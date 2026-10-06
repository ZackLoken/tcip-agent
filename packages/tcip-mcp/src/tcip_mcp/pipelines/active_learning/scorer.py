"""Active learning scorers: rank unlabeled images by informativeness.

Scorers resolve through a small dict registry (``resolve_scorer``) the agent can extend with
``register_scorer``, composing a new acquisition function (e.g. margin, least-confidence). The
built-in reference scorers:
  - UncertaintyScorer: prediction entropy / confidence spread
  - DiversityScorer: embedding distance from labeled set
  - CombinedScorer: weighted combination of uncertainty + diversity
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from tcip_mcp.pipelines.active_learning import DEFAULT_SCORER
from tcip_mcp.pipelines.model_contract import DETECTION_TASKS

if TYPE_CHECKING:
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

logger = logging.getLogger(__name__)


def _entropy(logits: "torch.Tensor") -> float:
    """Mean softmax entropy of a ``[B, C]`` logits tensor (classification uncertainty)."""
    probs = F.softmax(logits, dim=-1)
    return -(probs * probs.clamp(min=1e-8).log()).sum(-1).mean().item()


class BaseScorer(ABC):
    """Rank images by how valuable they'd be to label next."""

    @abstractmethod
    def score(self, image_paths: list[str],
              predictor: GenericPredictor) -> list[tuple[str, float]]:
        """Return (path, score) pairs sorted descending (highest = most valuable), reading each
        candidate through the loaded checkpoint ``predictor``: its model, its device and its
        ``in_chans`` travel together."""
        ...


class UncertaintyScorer(BaseScorer):
    """Score by prediction uncertainty (entropy of softmax outputs).

    For detection: uses mean confidence of top-scoring detections.
    For classification/ordinal: uses entropy of class probabilities.
    """

    def __init__(self, task: str) -> None:
        self.task = task

    @torch.no_grad()
    def score(self, image_paths: list[str],
              predictor: GenericPredictor) -> list[tuple[str, float]]:
        model = predictor.model
        model.eval()
        scored: list[tuple[str, float]] = []

        for path in image_paths:
            tensor = predictor.model_input(path)[0].unsqueeze(0)

            if self.task in DETECTION_TASKS:
                outputs = model([tensor[0]])
                if isinstance(outputs, list):
                    outputs = outputs[0]
                scores = outputs["scores"]
                # A frame with no detections holds no ambiguous decision, so it ranks lowest.
                uncertainty = 1.0 - scores.mean().item() if len(scores) else 0.0
            else:
                outputs = model(tensor)
                heads = outputs.values() if isinstance(outputs, dict) else [outputs]
                # Multi-head: average classification-style entropy across all heads.
                entropies = [_entropy(v) for v in heads
                             if isinstance(v, torch.Tensor) and v.dim() >= 2]
                if not entropies:
                    raise ValueError(f"the uncertainty scorer cannot score a {self.task} model: "
                                     "its output carries no class logits to take an entropy of.")
                uncertainty = sum(entropies) / len(entropies)

            scored.append((path, uncertainty))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


class DiversityScorer(BaseScorer):
    """Score by embedding distance from the already-labeled set.

    Uses backbone features (before head) as embeddings.
    Images far from any labeled image in feature space are most valuable.
    """

    def __init__(self, labeled_embeddings: np.ndarray | None = None) -> None:
        self._labeled = labeled_embeddings  # [N, D] numpy array

    def set_labeled_embeddings(self, embeddings: np.ndarray) -> None:
        self._labeled = embeddings

    @torch.no_grad()
    def score(self, image_paths: list[str],
              predictor: GenericPredictor) -> list[tuple[str, float]]:
        model = predictor.model
        if not hasattr(model, "backbone"):
            # No silent random-noise embeddings: diversity needs real backbone features.
            raise RuntimeError(
                f"DiversityScorer requires a model exposing a .backbone attribute (e.g. one "
                f"built via build_detector, or a bespoke nn.Module exposing one) to extract "
                f"embeddings; got {type(model).__name__}."
            )
        model.eval()

        # Extract backbone features
        embeddings = []
        for path in image_paths:
            tensor = predictor.model_input(path)[0].unsqueeze(0)
            feats = cast(Any, model).backbone(tensor)
            feat = list(feats.values())[-1] if isinstance(feats, dict) else feats
            emb = F.adaptive_avg_pool2d(feat, 1).flatten(1).cpu().numpy()
            embeddings.append(emb[0])

        embeddings_arr = np.stack(embeddings)

        if self._labeled is None or len(self._labeled) == 0:
            # No labeled reference set -> diversity is uninformative. Make it visible
            # (uniform scores let CombinedScorer fall back to uncertainty cleanly).
            logger.warning(
                "DiversityScorer: no labeled embeddings set; returning uniform diversity "
                "scores. Call set_labeled_embeddings() for meaningful diversity ranking."
            )
            return [(p, 1.0) for p in image_paths]

        # Cosine distance to nearest labeled sample
        from numpy.linalg import norm
        scored = []
        for i, emb in enumerate(embeddings_arr):
            emb_norm = emb / (norm(emb) + 1e-8)
            labeled_norm = self._labeled / (norm(self._labeled, axis=1, keepdims=True) + 1e-8)
            similarities = emb_norm @ labeled_norm.T
            min_dist = 1.0 - similarities.max()
            scored.append((image_paths[i], float(min_dist)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


class CombinedScorer(BaseScorer):
    """Weighted combination of uncertainty and diversity scores."""

    def __init__(
        self,
        task: str,
        uncertainty_weight: float = 0.6,
        diversity_weight: float = 0.4,
        labeled_embeddings: np.ndarray | None = None,
    ) -> None:
        self.unc = UncertaintyScorer(task)
        self.div = DiversityScorer(labeled_embeddings)
        self.uw = uncertainty_weight
        self.dw = diversity_weight

    def score(self, image_paths: list[str],
              predictor: GenericPredictor) -> list[tuple[str, float]]:
        unc_scores = dict(self.unc.score(image_paths, predictor))
        div_scores = dict(self.div.score(image_paths, predictor))

        # Normalize each to [0, 1]
        def _normalize(d: dict[str, float]) -> dict[str, float]:
            vals = list(d.values())
            lo, hi = min(vals), max(vals)
            rng = hi - lo if hi > lo else 1.0
            return {k: (v - lo) / rng for k, v in d.items()}

        unc_norm = _normalize(unc_scores)
        div_norm = _normalize(div_scores)

        combined = []
        for p in image_paths:
            s = self.uw * unc_norm.get(p, 0) + self.dw * div_norm.get(p, 0)
            combined.append((p, s))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined


SCORER_REGISTRY: dict[str, Callable[[str], BaseScorer]] = {
    "uncertainty": lambda task: UncertaintyScorer(task=task),
    "diversity": lambda task: DiversityScorer(),
    DEFAULT_SCORER: lambda task: CombinedScorer(task=task),
}


def register_scorer(name: str, factory: Callable[[str], BaseScorer]) -> None:
    """Register an acquisition-function scorer under ``name`` so ``method=<name>`` resolves to
    it."""
    SCORER_REGISTRY[name] = factory


def resolve_scorer(method: str, task: str) -> BaseScorer:
    """The scorer the factory registered under ``method``, or the dotted ``module:factory`` it
    names, builds for ``task`` (:func:`~tcip_mcp.pipelines.model_build.resolve_named`, whose
    refusals propagate)."""
    from tcip_mcp.pipelines.model_build import resolve_named

    target = resolve_named(method, SCORER_REGISTRY, kind="scorer", register="register_scorer")
    return target(task) if callable(target) else target
