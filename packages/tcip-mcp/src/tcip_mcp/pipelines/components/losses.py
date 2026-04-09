"""Composable loss functions with self-describing metadata.

Each loss registers into the global LOSSES registry so the agent can
query valid losses per task and compose CombinedLoss from multiple terms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcip_mcp.pipelines.registry import LOSSES


class BaseLoss(nn.Module):
    """Abstract base for registered losses."""
    name: str = ""
    valid_tasks: list[str] = []

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ====================================================================
# Standard losses
# ====================================================================

class CrossEntropyLoss(BaseLoss):
    name = "cross_entropy"
    valid_tasks = ["classification", "semantic_seg"]

    def __init__(self, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    def forward(self, predictions, targets):
        return self.ce(predictions, targets)


class FocalLoss(BaseLoss):
    name = "focal"
    valid_tasks = ["detection", "classification"]

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions, targets):
        ce = F.cross_entropy(predictions, targets, reduction="none")
        p_t = torch.exp(-ce)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SmoothL1Loss(BaseLoss):
    name = "smooth_l1"
    valid_tasks = ["detection", "regression"]

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.loss = nn.SmoothL1Loss(beta=beta)

    def forward(self, predictions, targets):
        return self.loss(predictions, targets)


class HuberLoss(BaseLoss):
    name = "huber"
    valid_tasks = ["regression"]

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.loss = nn.HuberLoss(delta=delta)

    def forward(self, predictions, targets):
        return self.loss(predictions, targets)


class BCEWithLogitsLoss(BaseLoss):
    name = "bce"
    valid_tasks = ["instance_seg", "semantic_seg"]

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, predictions, targets):
        return self.loss(predictions, targets.float())


class DiceLoss(BaseLoss):
    name = "dice"
    valid_tasks = ["instance_seg", "semantic_seg"]

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        probs = torch.sigmoid(predictions)
        flat_p = probs.flatten(1)
        flat_t = targets.float().flatten(1)
        intersection = (flat_p * flat_t).sum(-1)
        dice = 1.0 - (2.0 * intersection + self.smooth) / (
            flat_p.sum(-1) + flat_t.sum(-1) + self.smooth
        )
        return dice.mean()


class GIoULoss(BaseLoss):
    name = "giou"
    valid_tasks = ["detection"]

    def forward(self, predictions, targets):
        return _generalized_box_iou_loss(predictions, targets)


# ====================================================================
# Ordinal losses
# ====================================================================

class CORNLoss(BaseLoss):
    """Conditional Ordinal Regression Network loss (Shi et al. 2021)."""
    name = "corn"
    valid_tasks = ["ordinal"]

    def __init__(self, num_ranks: int):
        super().__init__()
        self.num_ranks = num_ranks

    def forward(self, predictions, targets):
        # predictions: [B, K-1] logits, targets: [B] rank indices (0-indexed)
        loss = torch.tensor(0.0, device=predictions.device)
        n = 0
        for k in range(self.num_ranks - 1):
            mask = targets >= k
            if mask.sum() == 0:
                continue
            target_k = (targets[mask] > k).float()
            loss = loss + F.binary_cross_entropy_with_logits(predictions[mask, k], target_k)
            n += 1
        return loss / max(n, 1)


class CORALLoss(BaseLoss):
    """Consistent Rank Logits loss (Cao, Mirjalili, Raschka 2020)."""
    name = "coral"
    valid_tasks = ["ordinal"]

    def __init__(self, num_ranks: int):
        super().__init__()
        self.num_ranks = num_ranks

    def forward(self, predictions, targets):
        # predictions: [B, K-1] cumulative logits, targets: [B] rank indices
        levels = torch.zeros_like(predictions)
        for i in range(predictions.size(0)):
            levels[i, :targets[i]] = 1.0
        return F.binary_cross_entropy_with_logits(predictions, levels)


# ====================================================================
# Combined loss
# ====================================================================

class CombinedLoss(BaseLoss):
    """Weighted combination of multiple losses."""
    name = "combined"
    valid_tasks = ["all"]

    def __init__(self, losses: list[BaseLoss], weights: list[float] | None = None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights or [1.0] * len(losses)

    def forward(self, predictions, targets):
        total = torch.tensor(0.0, device=predictions.device if torch.is_tensor(predictions) else "cpu")
        for loss_fn, w in zip(self.losses, self.weights):
            total = total + w * loss_fn(predictions, targets)
        return total


# ====================================================================
# Helpers
# ====================================================================

def _generalized_box_iou_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """GIoU loss for bounding box regression. Boxes in xyxy format."""
    x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area_p = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_g = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = area_p + area_g - inter

    iou = inter / (union + 1e-7)

    ex1 = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
    ey1 = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
    ex2 = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
    ey2 = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])
    enclose = (ex2 - ex1) * (ey2 - ey1)

    giou = iou - (enclose - union) / (enclose + 1e-7)
    return (1.0 - giou).mean()


def build_loss(name: str, **kwargs) -> BaseLoss:
    """Build a loss by registry name, or parse combined like 'bce+dice'."""
    if "+" in name:
        parts = name.split("+")
        sub_losses = [build_loss(p.strip(), **kwargs) for p in parts]
        return CombinedLoss(sub_losses)
    return LOSSES.build(name, **kwargs)


# ====================================================================
# Registration
# ====================================================================

_LOSS_MAP: list[tuple[str, type, dict]] = [
    ("cross_entropy", CrossEntropyLoss, {"description": "Standard cross-entropy", "valid_tasks": ["classification", "semantic_seg"]}),
    ("focal", FocalLoss, {"description": "Focal loss for class imbalance", "valid_tasks": ["detection", "classification"]}),
    ("smooth_l1", SmoothL1Loss, {"description": "Smooth L1 (box regression / regression)", "valid_tasks": ["detection", "regression"]}),
    ("huber", HuberLoss, {"description": "Huber loss (outlier-robust regression)", "valid_tasks": ["regression"]}),
    ("bce", BCEWithLogitsLoss, {"description": "Binary cross-entropy with logits", "valid_tasks": ["instance_seg", "semantic_seg"]}),
    ("dice", DiceLoss, {"description": "Dice loss for region overlap", "valid_tasks": ["instance_seg", "semantic_seg"]}),
    ("giou", GIoULoss, {"description": "Generalized IoU loss for box regression", "valid_tasks": ["detection"]}),
    ("corn", CORNLoss, {"description": "CORN ordinal loss (Shi 2021)", "valid_tasks": ["ordinal"], "when_to_use": "Disease scores, maturity ratings, ordinal scales"}),
    ("coral", CORALLoss, {"description": "CORAL ordinal loss (Cao 2020)", "valid_tasks": ["ordinal"], "when_to_use": "Alternative to CORN with consistent rank logits"}),
]

for _name, _cls, _meta in _LOSS_MAP:
    LOSSES.register_factory(
        _name,
        lambda c=_cls, **kw: c(**kw),
        category=_meta.get("valid_tasks", ["all"])[0],
        metadata=_meta,
    )
