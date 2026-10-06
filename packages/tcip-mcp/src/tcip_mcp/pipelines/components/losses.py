"""Loss functions for bespoke models: plain importable classes + a name->class map.

Bespoke model code imports a loss class directly or calls ``build_loss(name)``; the
``a+b`` syntax composes a ``CombinedLoss`` from multiple terms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional

from tcip_mcp.pipelines.derivations import num_classes_from_distribution


class BaseLoss(nn.Module):
    """Abstract base for registered losses."""
    name: str = ""

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def dice_loss(probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """One minus the Dice coefficient of ``probs`` against ``targets`` of the same shape, each
    summed over its last axis and the result averaged over every other axis."""
    intersection = (probs * targets).sum(-1)
    return (1.0 - (2.0 * intersection + smooth)
            / (probs.sum(-1) + targets.sum(-1) + smooth)).mean()


def corn_loss(logits: torch.Tensor, ranks: torch.Tensor, num_ranks: int) -> torch.Tensor:
    """The CORN loss of ``[B, num_ranks - 1]`` conditional ``logits`` against ``[B]`` 0-indexed
    ``ranks``: for each rank ``k``, binary cross-entropy of classifier ``k`` on the samples with
    a rank of at least ``k`` predicting a rank above ``k``, averaged over the classifiers that saw
    any sample, accumulated in the logits' own dtype."""
    loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    n_tasks = 0
    for k in range(num_ranks - 1):
        mask = ranks >= k
        if mask.sum() == 0:
            continue
        loss = loss + functional.binary_cross_entropy_with_logits(
            logits[mask, k], (ranks[mask] > k).float())
        n_tasks += 1
    return loss / max(n_tasks, 1)


class CrossEntropyLoss(BaseLoss):
    name = "cross_entropy"

    def __init__(self, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    def forward(self, predictions, targets):
        return self.ce(predictions, targets)


class FocalLoss(BaseLoss):
    name = "focal"

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
            # Per-class weight subsumes the scalar alpha: no double balance.
            ce = functional.cross_entropy(
                predictions, targets, weight=self.weight, reduction="none")
            p_t = torch.exp(-ce)
            loss = (1 - p_t) ** self.gamma * ce
        else:
            ce = functional.cross_entropy(predictions, targets, reduction="none")
            p_t = torch.exp(-ce)
            loss = self.alpha * (1 - p_t) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SmoothL1Loss(BaseLoss):
    name = "smooth_l1"

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.loss = nn.SmoothL1Loss(beta=beta)

    def forward(self, predictions, targets):
        return self.loss(predictions, targets)


class HuberLoss(BaseLoss):
    name = "huber"

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.loss = nn.HuberLoss(delta=delta)

    def forward(self, predictions, targets):
        return self.loss(predictions, targets)


class BCEWithLogitsLoss(BaseLoss):
    name = "bce"

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, predictions, targets):
        return self.loss(predictions, targets.float())


class DiceLoss(BaseLoss):
    """:func:`dice_loss` of each sample's sigmoid probabilities against its binary target."""

    name = "dice"

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        return dice_loss(torch.sigmoid(predictions).flatten(1), targets.float().flatten(1),
                         self.smooth)


class GIoULoss(BaseLoss):
    name = "giou"

    def forward(self, predictions, targets):
        return _generalized_box_iou_loss(predictions, targets)


class CORNLoss(BaseLoss):
    """:func:`corn_loss` over ``num_ranks`` ranks."""

    name = "corn"

    def __init__(self, num_ranks: int):
        super().__init__()
        self.num_ranks = num_ranks

    def forward(self, predictions, targets):
        return corn_loss(predictions, targets, self.num_ranks)


class CORALLoss(BaseLoss):
    """Consistent rank logits loss: binary cross-entropy of ``[B, K-1]`` cumulative logits against
    each sample's rank levels."""
    name = "coral"

    def __init__(self, num_ranks: int):
        super().__init__()
        self.num_ranks = num_ranks

    def forward(self, predictions, targets):
        levels = torch.zeros_like(predictions)
        for i in range(predictions.size(0)):
            levels[i, :targets[i]] = 1.0
        return functional.binary_cross_entropy_with_logits(predictions, levels)


class CombinedLoss(BaseLoss):
    """Weighted combination of multiple losses."""
    name = "combined"

    def __init__(self, losses: list[BaseLoss], weights: list[float] | None = None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights or [1.0] * len(losses)

    def forward(self, predictions, targets):
        total = torch.tensor(
            0.0, device=predictions.device if torch.is_tensor(predictions) else "cpu"
        )
        for loss_fn, w in zip(self.losses, self.weights):
            total = total + w * loss_fn(predictions, targets)
        return total


def _generalized_box_iou_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """GIoU loss for bounding box regression. Boxes in xyxy format."""
    from tcip_annotation.matching import box_areas

    x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    union = box_areas(pred_boxes) + box_areas(gt_boxes) - inter

    iou = inter / (union + 1e-7)

    ex1 = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
    ey1 = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
    ex2 = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
    ey2 = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])
    enclose = box_areas(torch.stack([ex1, ey1, ex2, ey2], dim=1))

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

    Schemes: ``balanced`` (``total/(n_present*count)``), ``inverse`` (``1/count``), or
    ``effective`` (``(1-beta)/(1-beta**count)``). Zero-count classes get weight 1.0. When
    ``normalize``, weights are rescaled so the mean over present classes is 1.0.

    ``num_classes`` unstated is the count the distribution itself implies
    (:func:`~tcip_mcp.pipelines.derivations.num_classes_from_distribution`); a distribution that
    counted nothing implies no classes and weights none.
    """
    if num_classes is None:
        num_classes = num_classes_from_distribution(class_distribution)
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
        present_mean = torch.tensor(
            [weights[c] for c in range(num_classes) if counts[c] > 0]
        ).mean()
        if present_mean > 0:
            w = w / present_mean
    return w


# Name -> loss class. ``weighted_ce`` is a plain CrossEntropyLoss that expects a ``weight``.
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


def build_loss(
    name: str, *, class_distribution: dict[int, int] | None = None,
    num_classes: int | None = None, weight_scheme: str = "balanced", **kwargs,
) -> BaseLoss:
    """Build a loss by name, or a :class:`CombinedLoss` from terms joined like ``'bce+dice'``.

    Each keyword goes to the terms whose constructor accepts it, so a per-term hyperparameter
    (``weight`` for the CE term, ``smooth`` for the dice term) reaches its own term; a keyword no
    term accepts refuses. A term is weightable when its constructor takes ``weight``: given a
    ``class_distribution``, an inverse-frequency ``weight`` (:func:`compute_class_weights`) goes
    to every weightable term unless ``weight`` was passed, and a ``class_distribution`` for a loss
    with no weightable term refuses. An unknown name refuses (``ValueError``) naming the
    registered ones.
    """
    from tcip_mcp.pipelines.model_build import keyword_parameters, resolve_named

    parts = [p.strip() for p in name.split("+")]
    accepted = {p: keyword_parameters(resolve_named(p, _LOSS_CLASSES, kind="loss"))
                for p in parts}

    def takes(p: str, k: str) -> bool:
        named, open_ended = accepted[p]
        return open_ended or k in named

    if class_distribution is not None:
        if not any(takes(p, "weight") for p in parts):
            raise ValueError(
                f"class_distribution was supplied for '{name}', which is not weightable (no term "
                "takes a weight); the weighting would have no effect. Compose a weightable term, "
                "or drop class_distribution.")
        kwargs.setdefault("weight", compute_class_weights(
            class_distribution, num_classes=num_classes, scheme=weight_scheme))
    unusable = sorted(k for k in kwargs if not any(takes(p, k) for p in parts))
    if unusable:
        raise ValueError(
            f"{unusable} not accepted by any term of '{name}'. Each term accepts: "
            + "; ".join(f"{p}: {'any' if accepted[p][1] else sorted(accepted[p][0])}"
                        for p in parts))
    terms = [_LOSS_CLASSES[p](**{k: v for k, v in kwargs.items() if takes(p, k)})
             for p in parts]
    return terms[0] if len(terms) == 1 else CombinedLoss(terms)
