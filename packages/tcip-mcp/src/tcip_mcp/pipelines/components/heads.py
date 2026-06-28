"""Task-specific heads — each knows its loss, metric, and output format.

Every head implements ``BaseHead`` with ``forward()``, ``compute_loss()``,
and ``decode()`` so the composer and trainer are task-agnostic.
"""

from __future__ import annotations

import abc
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcip_mcp.pipelines.registry import HEADS


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
        return {
            "ranks": predicted_ranks,
            "cumulative_probs": cum_probs,
        }


def _corn_loss(logits: torch.Tensor, ranks: torch.Tensor, num_ranks: int) -> torch.Tensor:
    """CORN loss — conditional ordinal regression.

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

    def __init__(self, in_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(in_channels, 1),
        )

    def forward(self, features: torch.Tensor, targets: Any = None) -> dict[str, torch.Tensor]:
        return {"values": self.fc(features).squeeze(-1)}

    def compute_loss(self, outputs, targets):
        return {"reg_loss": F.smooth_l1_loss(outputs["values"], targets["values"])}

    def decode(self, outputs):
        return {"values": outputs["values"]}


# ====================================================================
# Detection Head (anchor-based, wraps torchvision)
# ====================================================================

class AnchorDetectionHead(BaseHead):
    """Two-stage anchor-based detection (Faster R-CNN style).

    Wraps torchvision's detection pipeline behind the BaseHead interface.
    The backbone+neck features are passed through separately, so this head
    builds its own RPN + ROI heads.
    """

    task_type = "detection"
    default_loss = "focal+smooth_l1"

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        min_size: int = 800,
        max_size: int = 1333,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self._min_size = min_size
        self._max_size = max_size
        self._model = None  # Lazy init — needs torchvision
        self._kwargs = kwargs

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        m = fasterrcnn_resnet50_fpn(
            weights=None, min_size=self._min_size, max_size=self._max_size,
        )
        in_features = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes + 1)
        self._model = m

    def forward(self, features, targets=None):
        self._ensure_model()
        # In composed mode the backbone is already run; for standalone
        # compatibility, pass images directly if list[Tensor]
        if isinstance(features, (list, tuple)) and isinstance(features[0], torch.Tensor):
            if targets is not None:
                return self._model(features, targets)
            return self._model(features)
        return {"features": features}

    def compute_loss(self, outputs, targets):
        # torchvision returns loss dict directly in training mode
        if isinstance(outputs, dict) and "loss_classifier" in outputs:
            return outputs
        return {}

    def decode(self, outputs):
        if isinstance(outputs, list):
            return outputs[0] if outputs else {}
        return outputs


# ====================================================================
# Anchor-free / single-stage Detection Head (FCOS / RetinaNet)
# ====================================================================

class AnchorFreeDetectionHead(BaseHead):
    """Anchor-free (FCOS) or single-stage (RetinaNet) detection via torchvision.

    Like ``AnchorDetectionHead``, the real wiring is ``compose_model`` routing this
    head name to ``DetectionModel`` (which builds the detector over the shared
    backbone+neck). This standalone path exists only for registry/validation parity.
    """

    task_type = "detection"
    default_loss = "focal+giou+ctrness"

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        detector: str = "fcos",
        min_size: int = 800,
        max_size: int = 1333,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.detector = detector
        self._min_size = min_size
        self._max_size = max_size
        self._model = None  # Lazy init — needs torchvision
        self._kwargs = kwargs

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if self.detector == "retinanet":
            from torchvision.models.detection import retinanet_resnet50_fpn
            self._model = retinanet_resnet50_fpn(
                weights=None, num_classes=self.num_classes + 1,
                min_size=self._min_size, max_size=self._max_size,
            )
        else:
            from torchvision.models.detection import fcos_resnet50_fpn
            self._model = fcos_resnet50_fpn(
                weights=None, num_classes=self.num_classes + 1,
                min_size=self._min_size, max_size=self._max_size,
            )

    def forward(self, features, targets=None):
        self._ensure_model()
        if isinstance(features, (list, tuple)) and isinstance(features[0], torch.Tensor):
            if targets is not None:
                return self._model(features, targets)
            return self._model(features)
        return {"features": features}

    def compute_loss(self, outputs, targets):
        # FCOS / RetinaNet return their loss dict directly in training mode.
        if isinstance(outputs, dict) and ("classification" in outputs or "bbox_regression" in outputs):
            return outputs
        return {}

    def decode(self, outputs):
        if isinstance(outputs, list):
            return outputs[0] if outputs else {}
        return outputs


# ====================================================================
# Semantic Segmentation Head
# ====================================================================

class SemanticSegHead(BaseHead):
    """Pixel-wise semantic segmentation (DeepLab-style)."""

    task_type = "semantic_seg"
    default_loss = "ce+dice"

    def __init__(self, in_channels: int, num_classes: int,
                 loss: str | None = None, class_weights: list | None = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self._loss_name = loss  # informational; semantic_seg always blends CE + Dice
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )
        # Optional per-class weight applied to the CE term (Dice term is unweighted).
        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
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


# ====================================================================
# Registration
# ====================================================================

HEADS.register_factory("classification", lambda **kw: ClassificationHead(**kw),
    category="image_level", metadata={
        "description": "Multi-class classification head",
        "valid_tasks": ["classification"],
        "input_format": "flat_vector",
        "required_params": ["in_channels", "num_classes"],
    })

HEADS.register_factory("ordinal", lambda **kw: OrdinalHead(**kw),
    category="image_level", metadata={
        "description": "Ordinal classification via CORN (conditional ordinal regression)",
        "valid_tasks": ["ordinal"],
        "input_format": "flat_vector",
        "required_params": ["in_channels", "num_ranks"],
        "when_to_use": "Disease scores (1-9), maturity ratings, ordinal trait scales",
    })

HEADS.register_factory("regression", lambda **kw: RegressionHead(**kw),
    category="image_level", metadata={
        "description": "Continuous value regression head",
        "valid_tasks": ["regression"],
        "input_format": "flat_vector",
        "required_params": ["in_channels"],
    })

HEADS.register_factory("anchor_detection", lambda **kw: AnchorDetectionHead(**kw),
    category="detection", metadata={
        "description": "Anchor-based detector (Faster R-CNN style)",
        "valid_tasks": ["detection"],
        "input_format": "multi_scale_dict",
        "required_params": ["in_channels", "num_classes"],
    })

HEADS.register_factory("anchor_free_detection", lambda **kw: AnchorFreeDetectionHead(**kw),
    category="detection", metadata={
        "description": "Anchor-free / single-stage detector (FCOS or RetinaNet)",
        "valid_tasks": ["detection"],
        "input_format": "multi_scale_dict",
        "required_params": ["in_channels", "num_classes"],
        "when_to_use": "Tiny/dense objects (FCOS, anchor-free) or a lighter single-stage detector",
    })

HEADS.register_factory("semantic_seg", lambda **kw: SemanticSegHead(**kw),
    category="segmentation", metadata={
        "description": "Pixel-wise semantic segmentation",
        "valid_tasks": ["semantic_seg"],
        "input_format": "multi_scale_dict",
        "required_params": ["in_channels", "num_classes"],
    })
