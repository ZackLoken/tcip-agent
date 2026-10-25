"""Task-specific heads: each knows its loss, metric, and output format."""

from __future__ import annotations

import abc

import torch
import torch.nn as nn
from torch.nn import functional


class BaseHead(nn.Module, abc.ABC):
    """Abstract base for all task heads."""

    task_type: str = ""
    target_key: str = ""
    """The key a head's target carries its truth under and its decode its prediction under."""

    @abc.abstractmethod
    def forward(
        self, features: torch.Tensor, targets: object = None
    ) -> dict[str, torch.Tensor]:
        """Return output dict (logits, boxes, masks, etc.)."""

    @abc.abstractmethod
    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute loss dict from outputs and targets."""

    @abc.abstractmethod
    def decode(
        self, outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Post-process outputs into human-readable predictions."""


# ====================================================================
# Classification Head
# ====================================================================


class ClassificationHead(BaseHead):
    """Multi-class classification from a flat feature vector."""

    task_type = "classification"
    target_key = "labels"

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.0,
        loss: str | None = None,
        class_weights: list | None = None,
    ) -> None:
        """Build a linear classifier from ``in_channels`` features to ``num_classes`` logits.

        A ``dropout`` above zero drops features first. ``loss`` names a registry loss
        (``losses.build_loss``) weighted by ``class_weights``; ``None`` trains by cross-entropy.
        """
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.num_classes = num_classes
        # Opt-in registry loss (e.g. focal / weighted_ce) as a submodule so its
        # class-weight buffer follows the model to device. None uses cross_entropy.
        self._loss = None
        if loss is not None:
            from tcip_mcp.pipelines.components.losses import build_loss

            weight = (
                torch.tensor(class_weights, dtype=torch.float32)
                if class_weights is not None
                else None
            )
            self._loss = build_loss(loss, weight=weight)

    def forward(
        self, features: torch.Tensor, targets: object = None
    ) -> dict[str, torch.Tensor]:
        """Return ``{"logits"}`` for ``features``; ``targets`` are not read."""
        logits = self.fc(self.drop(features))
        return {"logits": logits}

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return ``{"cls_loss"}``, the head's loss of the logits against the target labels."""
        if self._loss is not None:
            return {
                "cls_loss": self._loss(
                    outputs["logits"], targets[self.target_key]
                )
            }
        return {
            "cls_loss": functional.cross_entropy(
                outputs["logits"], targets[self.target_key]
            )
        }

    def decode(
        self, outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return each sample's argmax label, its softmax confidence and every probability."""
        probs = functional.softmax(outputs["logits"], dim=-1)
        preds = probs.argmax(dim=-1)
        confs = probs.max(dim=-1).values
        return {
            self.target_key: preds,
            "confidences": confs,
            "probabilities": probs,
        }


# ====================================================================
# Ordinal Head (CORN)
# ====================================================================


class OrdinalHead(BaseHead):
    """Ordinal classification via CORN (conditional ordinal regression).

    K-1 binary classifiers, classifier k predicting P(Y > k | Y > k-1), trained by
    :func:`~tcip_mcp.pipelines.components.losses.corn_loss`.
    """

    task_type = "ordinal"
    target_key = "ranks"

    def __init__(
        self, in_channels: int, num_ranks: int, dropout: float = 0.0
    ) -> None:
        """Build ``num_ranks - 1`` conditional classifiers over ``in_channels`` features.

        A ``dropout`` above zero drops features first.
        """
        super().__init__()
        self.num_ranks = num_ranks
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # K-1 binary classifiers (conditional)
        self.classifiers = nn.Linear(in_channels, num_ranks - 1)

    def forward(
        self, features: torch.Tensor, targets: object = None
    ) -> dict[str, torch.Tensor]:
        """Return ``{"logits"}``, the conditional logits for ``features``; ``targets`` unread."""
        logits = self.classifiers(self.drop(features))
        return {"logits": logits}

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return ``{"ordinal_loss"}``, the CORN loss of the logits against the target ranks."""
        from tcip_mcp.pipelines.components.losses import corn_loss

        return {
            "ordinal_loss": corn_loss(
                outputs["logits"], targets[self.target_key], self.num_ranks
            )
        }

    def decode(
        self, outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return each sample's predicted rank, cumulative probabilities and confidence.

        The rank is the count of cumulative probabilities above 0.5; the confidence is the
        marginal probability of that rank.
        """
        logits = outputs["logits"]
        probs = torch.sigmoid(logits)  # P(Y > k | Y > k-1) per rank
        # Convert conditional to cumulative
        cum_probs = torch.cumprod(probs, dim=-1)
        # Predicted rank = number of thresholds exceeded
        predicted_ranks = (cum_probs > 0.5).sum(dim=-1)
        # Confidence is the marginal P(Y=k) = P(Y>=k) - P(Y>=k+1) at the predicted rank.
        batch = cum_probs.shape[0]
        p_eq = -torch.diff(  # [B, num_ranks], marginal P(Y=k)
            cum_probs,
            dim=-1,
            prepend=cum_probs.new_ones((batch, 1)),
            append=cum_probs.new_zeros((batch, 1)),
        )
        confidences = p_eq.gather(
            1, predicted_ranks.unsqueeze(-1).long()
        ).squeeze(-1)
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

    def __init__(
        self, in_channels: int, dropout: float = 0.0, loss: str | None = None
    ) -> None:
        """Build a linear regressor from ``in_channels`` features to one value.

        A ``dropout`` above zero drops features first. ``loss`` names a registry loss
        (``losses.build_loss``); ``None`` trains by smooth L1.
        """
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

    def forward(
        self, features: torch.Tensor, targets: object = None
    ) -> dict[str, torch.Tensor]:
        """Return ``{"values"}``, one value per sample of ``features``; ``targets`` unread."""
        return {"values": self.fc(features).squeeze(-1)}

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return ``{"reg_loss"}``, the head's loss of the values against the target values."""
        if self._loss is not None:
            return {
                "reg_loss": self._loss(outputs["values"], targets["values"])
            }
        return {
            "reg_loss": functional.smooth_l1_loss(
                outputs["values"], targets["values"]
            )
        }

    def decode(
        self, outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return the predicted ``values`` as they are."""
        return {"values": outputs["values"]}


# ====================================================================
# Semantic Segmentation Head
# ====================================================================


class SemanticSegHead(BaseHead):
    """Pixel-wise semantic segmentation (DeepLab-style).

    Trained by cross-entropy and multi-class Dice over the softmax; ``class_weights`` applies to
    the cross-entropy term.
    """

    task_type = "semantic_seg"

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        class_weights: list | None = None,
    ) -> None:
        """Build a conv head from ``in_channels`` features to ``num_classes`` logits per pixel."""
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
        weight = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )
        self.ce_weight: torch.Tensor | None
        self.register_buffer("ce_weight", weight)

    def forward(
        self,
        features: torch.Tensor | dict[str, torch.Tensor],
        targets: object = None,
    ) -> dict[str, torch.Tensor]:
        """Return ``{"logits"}`` over ``features``, a pyramid dict read at its first sorted key."""
        # Use highest-resolution pyramid level
        if isinstance(features, dict):
            keys = sorted(features.keys())
            x = features[keys[0]]
        else:
            x = features
        return {"logits": self.conv(x)}

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return ``{"ce_loss", "dice_loss"}`` of the logits against the target masks.

        Logits at another size than the masks are bilinearly resized to them first.
        """
        logits = outputs["logits"]
        mask = targets["masks"]
        # Resize logits to match target
        if logits.shape[-2:] != mask.shape[-2:]:
            logits = functional.interpolate(
                logits,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        from tcip_mcp.pipelines.components.losses import dice_loss

        ce = functional.cross_entropy(
            logits, mask.long(), weight=self.ce_weight
        )
        one_hot = (
            functional.one_hot(mask.long(), self.num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )
        probs = functional.softmax(logits, dim=1)
        return {
            "ce_loss": ce,
            "dice_loss": dice_loss(probs.flatten(2), one_hot.flatten(2)),
        }

    def decode(
        self, outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return each pixel's argmax class as ``masks`` beside the softmax probabilities."""
        logits = outputs["logits"]
        return {
            "masks": logits.argmax(dim=1),
            "probabilities": functional.softmax(logits, dim=1),
        }
