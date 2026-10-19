"""Loss functions for bespoke models: plain importable classes + a name->class map.

The ``a+b`` syntax composes a ``CombinedLoss`` from multiple terms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional

from tcip_mcp.pipelines.derivations import num_classes_from_distribution


class BaseLoss(nn.Module):
    """Abstract base for registered losses."""

    name: str = ""

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Return the loss of ``predictions`` against ``targets``; each subclass provides it."""
        raise NotImplementedError


def dice_loss(
    probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    """Return one minus the Dice coefficient of ``probs`` against ``targets`` of the same shape.

    Each is summed over its last axis and the result averaged over every other axis.
    """
    intersection = (probs * targets).sum(-1)
    return (
        1.0
        - (2.0 * intersection + smooth)
        / (probs.sum(-1) + targets.sum(-1) + smooth)
    ).mean()


def corn_loss(
    logits: torch.Tensor, ranks: torch.Tensor, num_ranks: int
) -> torch.Tensor:
    """Return the CORN loss of ``[B, num_ranks - 1]`` conditional ``logits`` against ``ranks``.

    ``ranks`` are ``[B]`` and 0-indexed. For each rank ``k``, the loss is the binary cross-entropy
    of classifier ``k`` on the samples with a rank of at least ``k`` predicting a rank above
    ``k``, averaged over the classifiers that saw any sample, accumulated in the logits' own dtype.
    """
    loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    n_tasks = 0
    for k in range(num_ranks - 1):
        mask = ranks >= k
        if mask.sum() == 0:
            continue
        loss = loss + functional.binary_cross_entropy_with_logits(
            logits[mask, k], (ranks[mask] > k).float()
        )
        n_tasks += 1
    return loss / max(n_tasks, 1)


class CrossEntropyLoss(BaseLoss):
    """Cross-entropy over class logits (``nn.CrossEntropyLoss``)."""

    name = "cross_entropy"

    def __init__(
        self, weight: torch.Tensor | None = None, label_smoothing: float = 0.0
    ) -> None:
        """Take an optional per-class ``weight`` and the ``label_smoothing`` to apply."""
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=label_smoothing
        )

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Return the cross-entropy of ``predictions`` logits against class ``targets``."""
        return self.ce(predictions, targets)


class FocalLoss(BaseLoss):
    """Focal loss over class logits: cross-entropy scaled by ``(1 - p_t) ** gamma``."""

    name = "focal"

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        weight: torch.Tensor | list | None = None,
        reduction: str = "mean",
    ) -> None:
        """Take ``alpha``, ``gamma``, an optional per-class ``weight`` and the ``reduction``.

        A ``weight`` given as a list is held as a float32 tensor buffer.
        """
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

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Return the focal loss of ``predictions`` against ``targets`` under ``reduction``.

        A per-class ``weight`` replaces the scalar ``alpha``. ``"mean"`` averages, ``"sum"`` sums,
        and any other reduction returns the per-sample losses.
        """
        if self.weight is not None:
            # Per-class weight subsumes the scalar alpha: no double balance.
            ce = functional.cross_entropy(
                predictions, targets, weight=self.weight, reduction="none"
            )
            p_t = torch.exp(-ce)
            loss = (1 - p_t) ** self.gamma * ce
        else:
            ce = functional.cross_entropy(
                predictions, targets, reduction="none"
            )
            p_t = torch.exp(-ce)
            loss = self.alpha * (1 - p_t) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SmoothL1Loss(BaseLoss):
    """Smooth L1 loss (``nn.SmoothL1Loss``)."""

    name = "smooth_l1"

    def __init__(self, beta: float = 1.0) -> None:
        """Take the ``beta`` at which the loss turns from quadratic to linear."""
        super().__init__()
        self.loss = nn.SmoothL1Loss(beta=beta)

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Return the smooth L1 loss of ``predictions`` against ``targets``."""
        return self.loss(predictions, targets)


class HuberLoss(BaseLoss):
    """Huber loss (``nn.HuberLoss``)."""

    name = "huber"

    def __init__(self, delta: float = 1.0) -> None:
        """Take the ``delta`` at which the loss turns from quadratic to linear."""
        super().__init__()
        self.loss = nn.HuberLoss(delta=delta)

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Return the Huber loss of ``predictions`` against ``targets``."""
        return self.loss(predictions, targets)


class BCEWithLogitsLoss(BaseLoss):
    """Binary cross-entropy over logits (``nn.BCEWithLogitsLoss``)."""

    name = "bce"

    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        """Take an optional ``pos_weight`` on the positive class."""
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Return the binary cross-entropy of ``predictions`` logits against ``targets``."""
        return self.loss(predictions, targets.float())


class DiceLoss(BaseLoss):
    """:func:`dice_loss` of each sample's sigmoid probabilities against its binary target."""

    name = "dice"

    def __init__(self, smooth: float = 1e-6) -> None:
        """Take the ``smooth`` term added to the Dice numerator and denominator."""
        super().__init__()
        self.smooth = smooth

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Return :func:`dice_loss` of sigmoid ``predictions``, each sample flattened."""
        return dice_loss(
            torch.sigmoid(predictions).flatten(1),
            targets.float().flatten(1),
            self.smooth,
        )


class GIoULoss(BaseLoss):
    """Generalized IoU loss over xyxy boxes."""

    name = "giou"

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Return the mean GIoU loss of ``predictions`` boxes against ``targets`` boxes.

        Each predicted box is paired with the target box at its row; the loss is
        ``torchvision.ops.generalized_box_iou_loss``'s.
        """
        from torchvision.ops import generalized_box_iou_loss

        return generalized_box_iou_loss(predictions, targets, reduction="mean")


class CORNLoss(BaseLoss):
    """:func:`corn_loss` over ``num_ranks`` ranks."""

    name = "corn"

    def __init__(self, num_ranks: int) -> None:
        """Take the rank count ``num_ranks``."""
        super().__init__()
        self.num_ranks = num_ranks

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Return :func:`corn_loss` of ``predictions`` against ``targets`` ranks."""
        return corn_loss(predictions, targets, self.num_ranks)


class CORALLoss(BaseLoss):
    """Consistent rank logits loss.

    The loss is the binary cross-entropy of ``[B, K-1]`` cumulative logits against each sample's
    rank levels.
    """

    name = "coral"

    def __init__(self, num_ranks: int) -> None:
        """Take the rank count ``num_ranks``."""
        super().__init__()
        self.num_ranks = num_ranks

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Return the binary cross-entropy of ``predictions`` against each sample's rank levels."""
        levels = torch.zeros_like(predictions)
        for i in range(predictions.size(0)):
            levels[i, : targets[i]] = 1.0
        return functional.binary_cross_entropy_with_logits(predictions, levels)


class CombinedLoss(BaseLoss):
    """Weighted combination of multiple losses."""

    name = "combined"

    def __init__(
        self, losses: list[BaseLoss], weights: list[float] | None = None
    ) -> None:
        """Take the ``losses`` and their ``weights``, each weight 1.0 when none are given."""
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights or [1.0] * len(losses)

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Return the weighted sum of the losses of ``predictions`` against ``targets``.

        Each loss is paired with the weight at its position; the sum runs over those pairs.
        """
        total = torch.tensor(0.0, device=predictions.device)
        for loss_fn, w in zip(self.losses, self.weights):
            total = total + w * loss_fn(predictions, targets)
        return total


def compute_class_weights(
    class_distribution: dict[int, int],
    num_classes: int | None = None,
    scheme: str = "balanced",
    beta: float = 0.999,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute per-class loss weights from a class-count distribution.

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
            eff = 1.0 - beta**cnt
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
    name: str,
    *,
    class_distribution: dict[int, int] | None = None,
    num_classes: int | None = None,
    weight_scheme: str = "balanced",
    **kwargs: object,
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
    from tcip_mcp.pipelines.model_build import (
        keyword_parameters,
        resolve_named,
    )

    parts = [p.strip() for p in name.split("+")]
    accepted = {
        p: keyword_parameters(resolve_named(p, _LOSS_CLASSES, kind="loss"))
        for p in parts
    }

    def takes(p: str, k: str) -> bool:
        named, open_ended = accepted[p]
        return open_ended or k in named

    if class_distribution is not None:
        if not any(takes(p, "weight") for p in parts):
            raise ValueError(
                f"class_distribution was supplied for '{name}', which is not weightable (no term "
                "takes a weight); the weighting would have no effect. Compose a weightable term, "
                "or drop class_distribution."
            )
        kwargs.setdefault(
            "weight",
            compute_class_weights(
                class_distribution,
                num_classes=num_classes,
                scheme=weight_scheme,
            ),
        )
    unusable = sorted(k for k in kwargs if not any(takes(p, k) for p in parts))
    if unusable:
        raise ValueError(
            f"{unusable} not accepted by any term of '{name}'. Each term accepts: "
            + "; ".join(
                f"{p}: {'any' if accepted[p][1] else sorted(accepted[p][0])}"
                for p in parts
            )
        )
    terms = [
        _LOSS_CLASSES[p](**{k: v for k, v in kwargs.items() if takes(p, k)})
        for p in parts
    ]
    return terms[0] if len(terms) == 1 else CombinedLoss(terms)
