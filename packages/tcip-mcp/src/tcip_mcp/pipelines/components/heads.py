"""Task-specific heads: each knows its loss, metric, and output format.

Every head implements ``BaseHead`` with ``forward()``, ``compute_loss()``,
and ``decode()`` so the composer and trainer are task-agnostic.
"""

from __future__ import annotations

import abc
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseHead(nn.Module, abc.ABC):
    """Abstract base for all task heads."""

    task_type: str = ""
    default_loss: str = ""

    @abc.abstractmethod
    def forward(self, features: Any, targets: Any = None) -> dict[str, torch.Tensor]:
        """Forward pass. Return output dict (logits, boxes, masks, etc.)."""

    @abc.abstractmethod
    def compute_loss(
        self, outputs: dict[str, torch.Tensor], targets: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute loss dict from outputs and targets."""

    @abc.abstractmethod
    def decode(self, outputs: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Post-process outputs into human-readable predictions."""


# ====================================================================
# Classification Head
# ====================================================================

class ClassificationHead(BaseHead):
    """Multi-class classification from a flat feature vector."""

    task_type = "classification"
    default_loss = "cross_entropy"

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.0,
                 loss: str | None = None, class_weights: list | None = None) -> None:
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.num_classes = num_classes
        # Opt-in registry loss (e.g. focal / weighted_ce) as a submodule so its
        # class-weight buffer follows the model to device. None -> today's behavior.
        self._loss = None
        if loss is not None:
            from tcip_mcp.pipelines.components.losses import build_loss
            weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
            self._loss = build_loss(loss, weight=weight)

    def forward(self, features: torch.Tensor, targets: Any = None) -> dict[str, torch.Tensor]:
        logits = self.fc(self.drop(features))
        return {"logits": logits}

    def compute_loss(self, outputs, targets):
        if self._loss is not None:
            return {"cls_loss": self._loss(outputs["logits"], targets["labels"])}
        return {"cls_loss": F.cross_entropy(outputs["logits"], targets["labels"])}

    def decode(self, outputs):
        probs = F.softmax(outputs["logits"], dim=-1)
        preds = probs.argmax(dim=-1)
        confs = probs.max(dim=-1).values
        return {"labels": preds, "confidences": confs, "probabilities": probs}


# ====================================================================
# Ordinal Head (CORN)
# ====================================================================

class OrdinalHead(BaseHead):
    """Ordinal classification via CORN (Conditional Ordinal Regression).

    Trains K-1 binary classifiers where classifier k predicts
    P(Y > k | Y > k-1).  Ref: Shi et al. 2021.
    """

    task_type = "ordinal"
    default_loss = "corn"

    def __init__(self, in_channels: int, num_ranks: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_ranks = num_ranks
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # K-1 binary classifiers (conditional)
        self.classifiers = nn.Linear(in_channels, num_ranks - 1)

    def forward(self, features: torch.Tensor, targets: Any = None) -> dict[str, torch.Tensor]:
        logits = self.classifiers(self.drop(features))
        return {"logits": logits, "num_ranks": torch.tensor(self.num_ranks)}

    def compute_loss(self, outputs, targets):
        logits = outputs["logits"]  # [B, K-1]
        ranks = targets["ranks"]  # [B] int, 0-indexed
        loss = _corn_loss(logits, ranks, self.num_ranks)
        return {"ordinal_loss": loss}

    def decode(self, outputs):
        logits = outputs["logits"]
        probs = torch.sigmoid(logits)  # P(Y > k | Y > k-1) per rank
        # Convert conditional to cumulative
        cum_probs = torch.cumprod(probs, dim=-1)
        # Predicted rank = number of thresholds exceeded
        predicted_ranks = (cum_probs > 0.5).sum(dim=-1)
        # Per-instance confidence: the marginal probability mass CORN's own cumulative outputs
        # imply at the predicted rank, P(Y=k) = P(Y>=k) - P(Y>=k+1), from the extended sequence
        # P(Y>=0)=1, P(Y>=k)=cum_probs[:,k-1] for k=1..num_ranks-1, P(Y>=num_ranks)=0 - the CORN
        # analog of ClassificationHead.decode's confs = probs.max(dim=-1).values (probability mass
        # at the argmax), not an invented heuristic.
        batch = cum_probs.shape[0]
        p_ge = torch.cat(
            [cum_probs.new_ones((batch, 1)), cum_probs, cum_probs.new_zeros((batch, 1))], dim=-1)
        p_eq = p_ge[:, :-1] - p_ge[:, 1:]  # [B, num_ranks], marginal P(Y=k)
        confidences = p_eq.gather(1, predicted_ranks.unsqueeze(-1).long()).squeeze(-1)
        return {
            "ranks": predicted_ranks,
            "cumulative_probs": cum_probs,
            "confidences": confidences,
        }


def _corn_loss(logits: torch.Tensor, ranks: torch.Tensor, num_ranks: int) -> torch.Tensor:
    """CORN loss: conditional ordinal regression.

    For each rank k in [0, K-2], classifier k is trained only on
    samples where Y >= k, predicting P(Y > k | Y >= k).
    """
    loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    n_tasks = 0
    for k in range(num_ranks - 1):
        mask = ranks >= k
        if mask.sum() == 0:
            continue
        target_k = (ranks[mask] > k).float()
        loss = loss + F.binary_cross_entropy_with_logits(logits[mask, k], target_k)
        n_tasks += 1
    return loss / max(n_tasks, 1)


# ====================================================================
# Regression Head
# ====================================================================

class RegressionHead(BaseHead):
    """Continuous value regression from a flat feature vector."""

    task_type = "regression"
    default_loss = "smooth_l1"

    def __init__(self, in_channels: int, dropout: float = 0.0, loss: str | None = None) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(in_channels, 1),
        )
        # Opt-in registry loss (e.g. huber) as a submodule, same pattern as ClassificationHead.
        # None -> today's smooth_l1 behavior.
        self._loss = None
        if loss is not None:
            from tcip_mcp.pipelines.components.losses import build_loss
            self._loss = build_loss(loss)

    def forward(self, features: torch.Tensor, targets: Any = None) -> dict[str, torch.Tensor]:
        return {"values": self.fc(features).squeeze(-1)}

    def compute_loss(self, outputs, targets):
        if self._loss is not None:
            return {"reg_loss": self._loss(outputs["values"], targets["values"])}
        return {"reg_loss": F.smooth_l1_loss(outputs["values"], targets["values"])}

    def decode(self, outputs):
        return {"values": outputs["values"]}


# ====================================================================
# Semantic Segmentation Head
# ====================================================================

class SemanticSegHead(BaseHead):
    """Pixel-wise semantic segmentation (DeepLab-style).

    Takes no ``loss`` name, and ``default_loss`` is empty on purpose: this head computes its own
    CE + multi-class Dice blend in ``compute_loss``, and there is no registry loss to route to.
    ``build_loss("cross_entropy+dice")`` constructs but raises at forward: the registry's
    ``DiceLoss`` is binary (sigmoid + flatten) while this head emits multi-class logits.
    ``class_weights`` *is* honored, applied to the CE term.
    """

    task_type = "semantic_seg"
    default_loss = ""

    def __init__(self, in_channels: int, num_classes: int,
                 class_weights: list | None = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )
        # Optional per-class weight applied to the CE term (Dice term is unweighted); the
        # annotation states the buffer's real type since nn.Module's own __getattr__ stub can't.
        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.ce_weight: torch.Tensor | None
        self.register_buffer("ce_weight", weight)

    def forward(self, features, targets=None):
        # Use highest-resolution pyramid level
        if isinstance(features, dict):
            keys = sorted(features.keys())
            x = features[keys[0]]
        else:
            x = features
        return {"logits": self.conv(x)}

    def compute_loss(self, outputs, targets):
        logits = outputs["logits"]
        mask = targets["masks"]
        # Resize logits to match target
        if logits.shape[-2:] != mask.shape[-2:]:
            logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        ce = F.cross_entropy(logits, mask.long(), weight=self.ce_weight)
        # Dice loss
        probs = F.softmax(logits, dim=1)
        flat_probs = probs.flatten(2)
        flat_mask = F.one_hot(mask.long(), self.num_classes).permute(0, 3, 1, 2).float().flatten(2)
        intersection = (flat_probs * flat_mask).sum(-1)
        dice = 1.0 - (2.0 * intersection + 1e-6) / (flat_probs.sum(-1) + flat_mask.sum(-1) + 1e-6)
        return {"ce_loss": ce, "dice_loss": dice.mean()}

    def decode(self, outputs):
        logits = outputs["logits"]
        return {"masks": logits.argmax(dim=1), "probabilities": F.softmax(logits, dim=1)}
