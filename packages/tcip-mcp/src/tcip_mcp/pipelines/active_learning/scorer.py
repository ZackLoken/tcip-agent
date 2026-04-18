"""Active learning scorers — rank unlabeled images by informativeness.

Scorers:
  - UncertaintyScorer: prediction entropy / confidence spread
  - DiversityScorer: embedding distance from labeled set
  - CombinedScorer: weighted combination of uncertainty + diversity
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F


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
        from tcip_mcp.pipelines.image_utils import pil_to_tensor
        from PIL import Image

        model.eval()
        scored: list[tuple[str, float]] = []

        for path in image_paths:
            img = Image.open(path).convert("RGB")
            tensor = pil_to_tensor(img).unsqueeze(0).to(device)

            if self.task in ("detection", "instance_seg"):
                outputs = model([tensor[0]])
                if isinstance(outputs, list):
                    outputs = outputs[0]
                scores = outputs.get("scores", torch.tensor([]))
                if len(scores) == 0:
                    uncertainty = 1.0  # no detections = high uncertainty
                else:
                    # Low confidence spread → uncertain
                    uncertainty = 1.0 - scores.mean().item()
            else:
                outputs = model(tensor)
                if isinstance(outputs, dict):
                    logits = next(iter(outputs.values()))
                else:
                    logits = outputs
                if isinstance(logits, torch.Tensor) and logits.dim() >= 2:
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * probs.clamp(min=1e-8).log()).sum(-1).mean().item()
                    uncertainty = entropy
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
        from tcip_mcp.pipelines.image_utils import pil_to_tensor
        from PIL import Image

        model.eval()

        # Extract backbone features
        embeddings = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            tensor = pil_to_tensor(img).unsqueeze(0).to(device)

            # Get backbone features if model is ComposedModel
            if hasattr(model, "backbone"):
                backbone: torch.nn.Module = model.backbone  # type: ignore[assignment]
                feats = backbone(tensor)
                if isinstance(feats, dict):
                    feat = list(feats.values())[-1]  # last scale
                else:
                    feat = feats
                emb = F.adaptive_avg_pool2d(feat, 1).flatten(1).cpu().numpy()
            else:
                emb = np.random.randn(1, 128)  # fallback
            embeddings.append(emb[0])

        embeddings_arr = np.stack(embeddings)

        if self._labeled is None or len(self._labeled) == 0:
            # No labeled set — all images equally diverse
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
