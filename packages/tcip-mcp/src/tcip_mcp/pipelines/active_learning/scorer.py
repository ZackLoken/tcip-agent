"""Active learning scorers: rank unlabeled images by informativeness.

An acquisition function is a capability, not a fixed menu: scorers resolve through a small dict
registry (``resolve_scorer``) the agent can extend with ``register_scorer``, composing a new
acquisition function (e.g. margin, least-confidence) and register it rather than pick from a welded
if/elif. The built-in reference scorers:
  - UncertaintyScorer: prediction entropy / confidence spread
  - DiversityScorer: embedding distance from labeled set
  - CombinedScorer: weighted combination of uncertainty + diversity
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _entropy(logits: "torch.Tensor") -> float:
    """Mean softmax entropy of a ``[B, C]`` logits tensor (classification uncertainty)."""
    probs = F.softmax(logits, dim=-1)
    return -(probs * probs.clamp(min=1e-8).log()).sum(-1).mean().item()


class BaseScorer(ABC):
    """Rank images by how valuable they'd be to label next."""

    @abstractmethod
    def score(self, image_paths: list[str], model: torch.nn.Module, device: torch.device) -> list[tuple[str, float]]:
        """Return (path, score) pairs sorted descending (highest = most valuable)."""
        ...


class UncertaintyScorer(BaseScorer):
    """Score by prediction uncertainty (entropy of softmax outputs).

    For detection: uses mean confidence of top-scoring detections.
    For classification/ordinal: uses entropy of class probabilities.
    """

    def __init__(self, task: str = "classification") -> None:
        self.task = task

    @torch.no_grad()
    def score(self, image_paths: list[str], model: torch.nn.Module, device: torch.device) -> list[tuple[str, float]]:
        from tcip_mcp.pipelines.image_utils import load_image, pil_to_tensor

        model.eval()
        scored: list[tuple[str, float]] = []

        for path in image_paths:
            img = load_image(path, 3)  # EXIF-oriented: score/embed in the same frame the model trained on
            tensor = pil_to_tensor(img).unsqueeze(0).to(device)

            if self.task in ("detection", "instance_seg"):
                outputs = model([tensor[0]])
                if isinstance(outputs, list):
                    outputs = outputs[0]
                scores = outputs.get("scores", torch.tensor([]))
                if len(scores) == 0:
                    # No detections = no ambiguous decision for uncertainty sampling to act on.
                    # Scoring these 1.0 floods the queue with empty frames; a missed object is a
                    # recall gap uncertainty sampling can't see from the model's own outputs (use
                    # diversity/coverage sampling for that), so an empty frame ranks low, not top.
                    uncertainty = 0.0
                else:
                    # Low confidence spread → uncertain
                    uncertainty = 1.0 - scores.mean().item()
            else:
                outputs = model(tensor)
                if isinstance(outputs, dict):
                    # Multi-head: average classification-style entropy across all heads
                    # (was first-head-only, which ignored every other head's uncertainty).
                    entropies = [
                        _entropy(v) for v in outputs.values()
                        if isinstance(v, torch.Tensor) and v.dim() >= 2
                    ]
                    uncertainty = sum(entropies) / len(entropies) if entropies else 0.5
                elif isinstance(outputs, torch.Tensor) and outputs.dim() >= 2:
                    uncertainty = _entropy(outputs)
                else:
                    uncertainty = 0.5

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
    def score(self, image_paths: list[str], model: torch.nn.Module, device: torch.device) -> list[tuple[str, float]]:
        from tcip_mcp.pipelines.image_utils import load_image, pil_to_tensor

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
            img = load_image(path, 3)  # EXIF-oriented: score/embed in the same frame the model trained on
            tensor = pil_to_tensor(img).unsqueeze(0).to(device)
            # A bespoke model's own opt-in attribute, not part of nn.Module's stub (checked above).
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
        task: str = "classification",
        uncertainty_weight: float = 0.6,
        diversity_weight: float = 0.4,
        labeled_embeddings: np.ndarray | None = None,
    ) -> None:
        self.unc = UncertaintyScorer(task)
        self.div = DiversityScorer(labeled_embeddings)
        self.uw = uncertainty_weight
        self.dw = diversity_weight

    def score(self, image_paths: list[str], model: torch.nn.Module, device: torch.device) -> list[tuple[str, float]]:
        unc_scores = dict(self.unc.score(image_paths, model, device))
        div_scores = dict(self.div.score(image_paths, model, device))

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


# The acquisition-function seam: one Protocol (BaseScorer) + one dict registry. Built-in reference
# scorers are pre-registered; the agent registers its own with register_scorer(name, factory) instead
# of editing a closed menu. A factory takes the task string and returns a BaseScorer.
SCORER_REGISTRY: dict[str, Callable[[str], BaseScorer]] = {
    "uncertainty": lambda task: UncertaintyScorer(task=task),
    "diversity": lambda task: DiversityScorer(),
    "combined": lambda task: CombinedScorer(task=task),
}


def register_scorer(name: str, factory: Callable[[str], BaseScorer]) -> None:
    """Register an acquisition-function scorer under ``name`` so ``method=<name>`` resolves to it."""
    SCORER_REGISTRY[name] = factory


def resolve_scorer(method: str, task: str) -> BaseScorer:
    """Resolve a scorer: a registered built-in name, else a dotted ``module:factory`` you wrote.

    Mirrors ``resolve_proposer``. An unresolvable name raises ``ValueError`` rather than falling
    back to the combined scorer: a silent substitution means the queue is ordered by an acquisition
    function the caller did not choose, while the result still reports the name that was asked for.
    A dotted name that fails to import raises ``ValueError`` too, so one ``except`` covers both.
    """
    factory = SCORER_REGISTRY.get(method)
    if factory is not None:
        return factory(task)
    if ":" in method or "." in method:
        from tcip_mcp.pipelines.model_build import _import_dotted

        try:
            target = _import_dotted(method)
        except Exception as exc:  # noqa: BLE001 (any import failure is an unresolvable name)
            raise ValueError(f"Could not import scorer {method!r}: {exc}") from exc
        return target(task) if callable(target) else target
    raise ValueError(
        f"Unknown scorer {method!r}. Use a built-in ({sorted(SCORER_REGISTRY)}), register one with "
        f"register_scorer, or pass a dotted 'module:factory' you wrote."
    )
