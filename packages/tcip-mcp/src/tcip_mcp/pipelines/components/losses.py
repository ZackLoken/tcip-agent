"""Loss functions for bespoke models: plain importable classes + a name->class map.

Bespoke model code imports a loss class directly or calls ``build_loss(name)``; the
``a+b`` syntax composes a ``CombinedLoss`` from multiple terms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 weight: torch.Tensor | list | None = None, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        if weight is not None and not torch.is_tensor(weight):
            weight = torch.tensor(weight, dtype=torch.float32)
        # Registered as a buffer so it moves with the model via .to(device); the annotation
        # states the buffer's real type since nn.Module's own __getattr__ stub can't.
        self.weight: torch.Tensor | None
        self.register_buffer("weight", weight)

    def forward(self, predictions, targets):
        if self.weight is not None:
            # Per-class weight subsumes the scalar alpha (RetinaNet-style): no double balance.
            ce = F.cross_entropy(predictions, targets, weight=self.weight, reduction="none")
            p_t = torch.exp(-ce)
            loss = (1 - p_t) ** self.gamma * ce
        else:
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


def compute_class_weights(
    class_distribution: dict[int, int],
    num_classes: int | None = None,
    scheme: str = "balanced",
    beta: float = 0.999,
    normalize: bool = True,
) -> torch.Tensor:
    """Per-class loss weights from a class-count distribution.

    Schemes: ``balanced`` (sklearn-style ``total/(n_present*count)``, the same
    inverse-frequency formula ``ClassBalancedSampler`` uses), ``inverse``
    (``1/count``), or ``effective`` (Cui et al. 2019, ``(1-beta)/(1-beta**count)``).
    Zero-count classes get weight 1.0. When ``normalize``, weights are rescaled so
    the mean over present classes is 1.0.
    """
    if num_classes is None:
        num_classes = (max(class_distribution) + 1) if class_distribution else 1
    counts = [int(class_distribution.get(c, 0)) for c in range(num_classes)]
    total = sum(counts)
    n_present = sum(1 for c in counts if c > 0)
    weights = []
    for cnt in counts:
        if cnt <= 0:
            weights.append(1.0)
        elif scheme == "inverse":
            weights.append(1.0 / cnt)
        elif scheme == "effective":
            eff = 1.0 - beta ** cnt
            weights.append((1.0 - beta) / eff if eff > 0 else 1.0)
        else:  # balanced
            weights.append(total / (n_present * cnt) if n_present > 0 else 1.0)
    w = torch.tensor(weights, dtype=torch.float32)
    if normalize and n_present > 0:
        present_mean = torch.tensor([weights[c] for c in range(num_classes) if counts[c] > 0]).mean()
        if present_mean > 0:
            w = w / present_mean
    return w


_WEIGHTABLE_LOSSES = {"cross_entropy", "weighted_ce", "focal"}

# Name → loss class. ``weighted_ce`` is a plain CrossEntropyLoss that expects a ``weight``.
_LOSS_CLASSES: dict[str, type[BaseLoss]] = {
    "cross_entropy": CrossEntropyLoss,
    "weighted_ce": CrossEntropyLoss,
    "focal": FocalLoss,
    "smooth_l1": SmoothL1Loss,
    "huber": HuberLoss,
    "bce": BCEWithLogitsLoss,
    "dice": DiceLoss,
    "giou": GIoULoss,
    "corn": CORNLoss,
    "coral": CORALLoss,
}


def _accepted_kwargs(loss_name: str) -> set[str] | None:
    """Constructor keyword names a loss accepts, or ``None`` if it takes ``**kwargs``."""
    import inspect

    cls = _LOSS_CLASSES.get(loss_name)
    if cls is None:
        return set()
    params = inspect.signature(cls).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return None
    return {n for n, p in params.items()
            if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) and n != "self"}


def build_loss(
    name: str, *, class_distribution: dict[int, int] | None = None,
    num_classes: int | None = None, weight_scheme: str = "balanced", **kwargs,
) -> BaseLoss:
    """Build a loss by name, or parse combined like 'bce+dice'.

    When ``class_distribution`` is supplied and the loss is weightable
    (``cross_entropy``/``weighted_ce``/``focal``), an inverse-frequency ``weight``
    tensor is injected unless ``weight`` was passed explicitly. Supplying it for a loss that
    cannot consume it raises rather than dropping it: imbalance handling that silently vanishes
    is worse than a build that refuses.

    In a combined loss each keyword goes to the terms whose constructor accepts it, so a per-term
    hyperparameter (``weight`` for the CE term, ``smooth`` for the dice term) reaches its own term
    instead of every term. A keyword no term accepts raises.
    """
    if "+" in name:
        parts = [p.strip() for p in name.split("+")]
        if class_distribution is not None and not any(p in _WEIGHTABLE_LOSSES for p in parts):
            raise ValueError(
                f"class_distribution was supplied for '{name}', but none of {parts} is weightable "
                f"(weightable: {sorted(_WEIGHTABLE_LOSSES)}); the weighting would have no effect. "
                "Compose a weightable term, or drop class_distribution."
            )
        # Route each hyperparameter to the terms that accept it. Broadcasting every kwarg to every
        # term makes a per-term argument a TypeError from whichever term lacks it; dropping it
        # silently builds a loss that is not the one asked for.
        accepted = {p: _accepted_kwargs(p) for p in parts}

        def _takes(p: str, k: str) -> bool:
            accepted_p = accepted[p]
            return accepted_p is None or k in accepted_p

        unusable = sorted(k for k in kwargs if not any(_takes(p, k) for p in parts))
        if unusable:
            raise ValueError(
                f"{unusable} not accepted by any term of '{name}'. Each term accepts: "
                + "; ".join(
                    f"{p}: {'any' if (accepted_p := accepted[p]) is None else sorted(accepted_p)}"
                    for p in parts
                )
            )
        sub_losses = [
            build_loss(p, num_classes=num_classes, weight_scheme=weight_scheme,
                       class_distribution=(class_distribution if p in _WEIGHTABLE_LOSSES else None),
                       **{k: v for k, v in kwargs.items() if _takes(p, k)})
            for p in parts
        ]
        return CombinedLoss(sub_losses)
    if class_distribution is not None and name not in _WEIGHTABLE_LOSSES:
        raise ValueError(
            f"class_distribution was supplied for '{name}', which is not weightable "
            f"(weightable: {sorted(_WEIGHTABLE_LOSSES)}); the weighting would have no effect. "
            "Use a weightable loss, or drop class_distribution."
        )
    if class_distribution is not None and "weight" not in kwargs:
        kwargs["weight"] = compute_class_weights(class_distribution, num_classes=num_classes, scheme=weight_scheme)
    try:
        cls = _LOSS_CLASSES[name]
    except KeyError:
        raise KeyError(f"Unknown loss '{name}'. Available: {sorted(_LOSS_CLASSES)}") from None
    return cls(**kwargs)
