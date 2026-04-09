"""Loss function utilities.

Torchvision detection models compute their own loss internally when
called in training mode (model(images, targets)). This module provides
helpers for custom loss weighting, monitoring, and additional loss
functions (focal loss, GIoU) that can replace the built-in losses.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_loss(loss_dict: dict, weights: dict | None = None) -> float:
    """Compute weighted sum of a loss dictionary.

    Torchvision detection models return a dict like:
        {"loss_classifier": ..., "loss_box_reg": ..., "loss_objectness": ..., "loss_rpn_box_reg": ...}

    Args:
        loss_dict: Dict of named loss tensors.
        weights: Optional dict mapping loss names to weights. Defaults to 1.0 for all.

    Returns:
        Total weighted loss (float for logging; use tensor sum for backward).
    """
    if weights is None:
        weights = {}
    total = 0.0
    for name, val in loss_dict.items():
        w = weights.get(name, 1.0)
        total += w * val.item()
    return total


def sum_losses(loss_dict: dict, weights: dict | None = None):
    """Return a tensor that is the weighted sum of all losses (for .backward()).

    Args:
        loss_dict: Dict of named loss tensors.
        weights: Optional weight overrides.
    """
    if weights is None:
        weights = {}
    total = None
    for name, val in loss_dict.items():
        w = weights.get(name, 1.0)
        term = w * val
        total = term if total is None else total + term
    return total


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss for addressing class imbalance in detection.

    Lin et al., "Focal Loss for Dense Object Detection" (RetinaNet paper).
    Reduces loss contribution from easy negatives to focus on hard examples.

    Args:
        inputs: Predicted logits [N, C] (pre-sigmoid).
        targets: Ground truth class labels [N] (long).
        alpha: Weighting factor for the rare class. Default 0.25.
        gamma: Focusing parameter. Higher values down-weight easy examples more. Default 2.0.
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Focal loss tensor.
    """
    ce_loss = F.cross_entropy(inputs, targets, reduction="none")
    p_t = torch.exp(-ce_loss)  # probability of correct class
    focal_weight = alpha * (1 - p_t) ** gamma
    loss = focal_weight * ce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary sigmoid focal loss (for single-class or multi-label detection).

    Args:
        inputs: Predicted logits [N] or [N, C] (pre-sigmoid).
        targets: Binary targets, same shape as inputs, float in [0, 1].
        alpha: Weighting factor. Default 0.25.
        gamma: Focusing parameter. Default 2.0.
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Focal loss tensor.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (alpha * targets + (1 - alpha) * (1 - targets)) * (1 - p_t) ** gamma
    loss = focal_weight * ce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


# ---------------------------------------------------------------------------
# GIoU Loss
# ---------------------------------------------------------------------------

def box_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute Generalized IoU between two sets of boxes.

    Rezatofighi et al., "Generalized Intersection over Union".

    Args:
        boxes1: [N, 4] in (x1, y1, x2, y2) format.
        boxes2: [N, 4] in (x1, y1, x2, y2) format.

    Returns:
        GIoU values [N] in range [-1, 1].
    """
    # Intersection
    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter_area

    iou = inter_area / union.clamp(min=1e-6)

    # Enclosing box
    enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
    return giou


def giou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """GIoU loss for bounding box regression.

    Loss = 1 - GIoU, so a perfect prediction yields 0 loss.

    Args:
        pred_boxes: Predicted boxes [N, 4] (x1, y1, x2, y2).
        target_boxes: Ground truth boxes [N, 4] (x1, y1, x2, y2).
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        GIoU loss tensor.
    """
    giou = box_giou(pred_boxes, target_boxes)
    loss = 1.0 - giou

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


# ---------------------------------------------------------------------------
# DIoU and CIoU Loss (distance-aware)
# ---------------------------------------------------------------------------

def diou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Distance-IoU loss.

    Adds center distance penalty to IoU for faster convergence.

    Args:
        pred_boxes: [N, 4] (x1, y1, x2, y2).
        target_boxes: [N, 4] (x1, y1, x2, y2).
        reduction: 'mean', 'sum', or 'none'.
    """
    # IoU
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area1 = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area2 = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-6)

    # Center distance
    pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    tgt_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    tgt_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    center_dist_sq = (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2

    # Enclosing box diagonal
    enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2

    loss = 1.0 - iou + center_dist_sq / diag_sq.clamp(min=1e-6)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss
