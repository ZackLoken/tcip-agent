"""Task-specific heads: each knows its loss, metric, and output format.

Every head implements ``BaseHead`` with ``forward()``, ``compute_loss()``,
and ``decode()`` so the trainer is task-agnostic.
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
    target_key: str = ""
    """The key a head's target carries its truth under and its decode its prediction under."""

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
    target_key = "labels"

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.0,
                 loss: str | None = None, class_weights: list | None = None) -> None:
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.num_classes = num_classes
        # Opt-in registry loss (e.g. focal / weighted_ce) as a submodule so its
        # class-weight buffer follows the model to device. None uses cross_entropy.
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
            return {"cls_loss": self._loss(outputs["logits"], targets[self.target_key])}
        return {"cls_loss": F.cross_entropy(outputs["logits"], targets[self.target_key])}

    def decode(self, outputs):
        probs = F.softmax(outputs["logits"], dim=-1)
        preds = probs.argmax(dim=-1)
        confs = probs.max(dim=-1).values
        return {self.target_key: preds, "confidences": confs, "probabilities": probs}


# ====================================================================
# Ordinal Head (CORN)
# ====================================================================

class OrdinalHead(BaseHead):
    """Ordinal classification via CORN (conditional ordinal regression): K-1 binary classifiers
    where classifier k predicts P(Y > k | Y > k-1), trained by
    :func:`~tcip_mcp.pipelines.components.losses.corn_loss`."""

    task_type = "ordinal"
    target_key = "ranks"

    def __init__(self, in_channels: int, num_ranks: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_ranks = num_ranks
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # K-1 binary classifiers (conditional)
        self.classifiers = nn.Linear(in_channels, num_ranks - 1)

    def forward(self, features: torch.Tensor, targets: Any = None) -> dict[str, torch.Tensor]:
        logits = self.classifiers(self.drop(features))
        return {"logits": logits}

    def compute_loss(self, outputs, targets):
        from tcip_mcp.pipelines.components.losses import corn_loss

        return {"ordinal_loss": corn_loss(outputs["logits"], targets[self.target_key],
                                          self.num_ranks)}

    def decode(self, outputs):
        logits = outputs["logits"]
        probs = torch.sigmoid(logits)  # P(Y > k | Y > k-1) per rank
        # Convert conditional to cumulative
        cum_probs = torch.cumprod(probs, dim=-1)
        # Predicted rank = number of thresholds exceeded
        predicted_ranks = (cum_probs > 0.5).sum(dim=-1)
        # Confidence is the marginal P(Y=k) = P(Y>=k) - P(Y>=k+1) at the predicted rank.
        batch = cum_probs.shape[0]
        p_ge = torch.cat(
            [cum_probs.new_ones((batch, 1)), cum_probs, cum_probs.new_zeros((batch, 1))], dim=-1)
        p_eq = p_ge[:, :-1] - p_ge[:, 1:]  # [B, num_ranks], marginal P(Y=k)
        confidences = p_eq.gather(1, predicted_ranks.unsqueeze(-1).long()).squeeze(-1)
        return {
            self.target_key: predicted_ranks,
            "cumulative_probs": cum_probs,
            "confidences": confidences,
        }


# ====================================================================
# Regression Head
# ====================================================================

class RegressionHead(BaseHead):
    """Continuous value regression from a flat feature vector."""

    task_type = "regression"

    def __init__(self, in_channels: int, dropout: float = 0.0, loss: str | None = None) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(in_channels, 1),
        )
        # Opt-in registry loss (e.g. huber) as a submodule, same pattern as ClassificationHead.
        # None uses smooth_l1.
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
    """Pixel-wise semantic segmentation (DeepLab-style), trained by cross-entropy and
    multi-class Dice over the softmax; ``class_weights`` applies to the cross-entropy term."""

    task_type = "semantic_seg"

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
        from tcip_mcp.pipelines.components.losses import dice_loss

        ce = F.cross_entropy(logits, mask.long(), weight=self.ce_weight)
        one_hot = F.one_hot(mask.long(), self.num_classes).permute(0, 3, 1, 2).float()
        return {"ce_loss": ce,
                "dice_loss": dice_loss(F.softmax(logits, dim=1).flatten(2), one_hot.flatten(2))}

    def decode(self, outputs):
        logits = outputs["logits"]
        return {"masks": logits.argmax(dim=1), "probabilities": F.softmax(logits, dim=1)}
