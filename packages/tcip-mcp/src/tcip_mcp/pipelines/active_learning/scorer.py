"""Active learning scorers: rank unlabeled images by informativeness.

Scorers resolve from a scorer block through a small dict registry of factories taking the task
and the block's values as keywords (``resolve_scorer``), extensible with ``register_scorer``,
composing a new acquisition function (e.g. margin, least-confidence); the registered ones and
the values each takes are ``active_learning.scorer_catalog``'s.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from statistics import fmean
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch.distributions import Categorical
from torch.nn import functional

from tcip_mcp.pipelines.model_contract import DETECTOR_TASKS

if TYPE_CHECKING:
    from pathlib import Path

    from tcip_mcp.pipelines.active_learning import ScorerSpec
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor


class BaseScorer(ABC):
    """Rank images by how valuable they'd be to label next."""

    @abstractmethod
    def score(
        self,
        image_paths: list[Path | BandGroupRef],
        labeled: list[Path | BandGroupRef],
        predictor: GenericPredictor,
    ) -> list[tuple[Path | BandGroupRef, float]]:
        """Return (path, score) pairs of ``image_paths`` sorted descending, most valuable first.

        ``labeled`` are the images of the candidates' capture already finished, which a scorer
        measuring the candidates against what is labeled reads. Each image is read through the
        loaded checkpoint ``predictor``: its model, its device and its ``in_chans`` travel
        together.
        """
        ...


class UncertaintyScorer(BaseScorer):
    """Score by prediction uncertainty (entropy of softmax outputs).

    For detection: uses mean confidence of top-scoring detections.
    For classification/ordinal: uses entropy of class probabilities.
    """

    def __init__(self, task: str) -> None:
        """Hold the ``task`` whose model outputs the scorer reads."""
        self.task = task

    @torch.no_grad()
    def score(
        self,
        image_paths: list[Path | BandGroupRef],
        labeled: list[Path | BandGroupRef],
        predictor: GenericPredictor,
    ) -> list[tuple[Path | BandGroupRef, float]]:
        """Return (path, uncertainty) pairs sorted descending; ``labeled`` is not read.

        A detector's uncertainty is one minus its mean detection score, 0.0 for no detection;
        any other model's is the mean softmax entropy over its heads' logits. Raises
        ``ValueError`` for an output carrying no class logits.
        """
        model = predictor.model
        model.eval()
        scored: list[tuple[Path | BandGroupRef, float]] = []

        for path in image_paths:
            tensor = predictor.model_input(path)[0].unsqueeze(0)

            if self.task in DETECTOR_TASKS:
                outputs = model([tensor[0]])
                if isinstance(outputs, list):
                    outputs = outputs[0]
                scores = outputs["scores"]
                # A frame with no detections holds no ambiguous decision, so it ranks lowest.
                uncertainty = (
                    1.0 - scores.mean().item() if len(scores) else 0.0
                )
            else:
                outputs = model(tensor)
                heads = (
                    outputs.values()
                    if isinstance(outputs, dict)
                    else [outputs]
                )
                # Multi-head: average classification-style entropy across all heads.
                entropies = [
                    Categorical(logits=v).entropy().mean().item()
                    for v in heads
                    if isinstance(v, torch.Tensor) and v.dim() >= 2
                ]
                if not entropies:
                    raise ValueError(
                        f"the uncertainty scorer cannot score a {self.task} model: "
                        "its output carries no class logits to take an entropy of."
                    )
                uncertainty = fmean(entropies)

            scored.append((path, uncertainty))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


class DiversityScorer(BaseScorer):
    """Score by embedding distance from the labeled images.

    An image's embedding is its backbone's last feature map, average-pooled; images far from
    every labeled image in that space are most valuable.
    """

    @staticmethod
    def _embeddings(
        images: list[Path | BandGroupRef], predictor: GenericPredictor
    ) -> np.ndarray:
        """Return the ``[N, D]`` embeddings of ``images`` under ``predictor``'s backbone."""
        backbone = cast(Any, predictor.model).backbone
        embeddings = []
        for path in images:
            feats = backbone(predictor.model_input(path)[0].unsqueeze(0))
            feat = (
                list(feats.values())[-1] if isinstance(feats, dict) else feats
            )
            pooled = functional.adaptive_avg_pool2d(feat, 1).flatten()
            embeddings.append(pooled.cpu().numpy())
        return np.stack(embeddings)

    @torch.no_grad()
    def score(
        self,
        image_paths: list[Path | BandGroupRef],
        labeled: list[Path | BandGroupRef],
        predictor: GenericPredictor,
    ) -> list[tuple[Path | BandGroupRef, float]]:
        """Return (path, cosine distance to the nearest labeled image) pairs, descending.

        Raises ``ValueError`` when ``labeled`` is empty, and ``RuntimeError`` for a model exposing
        no ``backbone``.
        """
        if not labeled:
            raise ValueError(
                "diversity ranks each image by its distance to the capture's labeled images, "
                "and the capture has no image finished for the subject: name the subject, and "
                "mark the images annotated complete, or rank by 'uncertainty'."
            )
        model = predictor.model
        if not hasattr(model, "backbone"):
            raise RuntimeError(
                f"DiversityScorer embeds images through the model's .backbone attribute, a "
                f"module answering feature maps; {type(model).__name__} exposes none."
            )
        model.eval()
        nearest = cdist(
            self._embeddings(image_paths, predictor),
            self._embeddings(labeled, predictor),
            "cosine",
        ).min(axis=1)
        scored = [
            (path, float(d))
            for path, d in zip(image_paths, nearest, strict=True)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


class CombinedScorer(BaseScorer):
    """Weighted combination of uncertainty and diversity scores, each normalized to [0, 1]."""

    def __init__(
        self,
        task: str,
        uncertainty_weight: float,
        diversity_weight: float,
    ) -> None:
        """Combine an uncertainty scorer for ``task`` and a diversity scorer at their weights."""
        self.unc = UncertaintyScorer(task)
        self.div = DiversityScorer()
        self.uw = uncertainty_weight
        self.dw = diversity_weight

    def score(
        self,
        image_paths: list[Path | BandGroupRef],
        labeled: list[Path | BandGroupRef],
        predictor: GenericPredictor,
    ) -> list[tuple[Path | BandGroupRef, float]]:
        """Return (path, weighted sum of the normalized scores) pairs sorted descending.

        Each term is min-max scaled (``sklearn.preprocessing.minmax_scale``, a constant term
        scaling to zeros). The diversity term measures against ``labeled``
        (:meth:`DiversityScorer.score`, whose refusals propagate). The two rankings are joined by
        each candidate's file (:func:`source_path_of`).
        """
        from sklearn.preprocessing import minmax_scale

        from tcip_mcp.pipelines.image_utils import source_path_of

        def _normalize(
            scored: list[tuple[Path | BandGroupRef, float]],
        ) -> dict[str, float]:
            d = {source_path_of(p): s for p, s in scored}
            return dict(zip(d, minmax_scale(list(d.values())), strict=True))

        div_norm = _normalize(self.div.score(image_paths, labeled, predictor))
        unc_norm = _normalize(self.unc.score(image_paths, labeled, predictor))

        combined = []
        for p in image_paths:
            key = source_path_of(p)
            s = self.uw * unc_norm.get(key, 0) + self.dw * div_norm.get(key, 0)
            combined.append((p, s))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined


TASK_PARAMETER = "task"
"""The keyword a scorer factory is handed the checkpoint's task under (:func:`resolve_scorer`)."""

SCORER_REGISTRY: dict[str, Callable[..., BaseScorer]] = {
    "uncertainty": UncertaintyScorer,
    "diversity": lambda task: DiversityScorer(),
    "combined": CombinedScorer,
}
"""Each scorer a block may name, as its factory taking the task and the block's values."""


def register_scorer(name: str, factory: Callable[..., BaseScorer]) -> None:
    """Register an acquisition-function scorer's factory under ``name``.

    The factory takes the task under :data:`TASK_PARAMETER` and the block's values as keywords;
    a scorer block naming ``name`` resolves to it. Its values model is read first
    (:func:`~tcip_mcp.pipelines.active_learning.values_model`), whose refusal of a factory naming
    no :data:`TASK_PARAMETER` propagates.
    """
    from tcip_mcp.pipelines.active_learning import values_model

    values_model(factory)
    SCORER_REGISTRY[name] = factory


def resolve_scorer(block: ScorerSpec, task: str) -> BaseScorer:
    """Build the scorer ``block`` names for ``task`` from the block's own values.

    The factory is the one registered under ``block.name``, or the dotted ``module:factory`` it
    names (:func:`~tcip_mcp.pipelines.model_build.resolve_named`, whose refusals propagate), and
    is called with the block's values (``ScorerSpec.model_extra``) and ``task`` under
    :data:`TASK_PARAMETER`, all as keywords. The call refuses (``TypeError``) a value the factory
    requires that the block leaves unstated, one the block states that the factory does not take,
    a factory taking no :data:`TASK_PARAMETER`, and a block stating one, naming each; a factory
    that builds anything but a :class:`BaseScorer` refuses (``TypeError``) naming ``block.name``.
    """
    from tcip_mcp.pipelines.model_build import resolve_named

    target = resolve_named(
        block.name, SCORER_REGISTRY, kind="scorer", register="register_scorer"
    )
    scorer = target(**(block.model_extra or {}), **{TASK_PARAMETER: task})
    if not isinstance(scorer, BaseScorer):
        raise TypeError(
            f"scorer {block.name!r} built a {type(scorer).__name__}, not a BaseScorer: a "
            "scorer factory returns the BaseScorer it builds."
        )
    return scorer
