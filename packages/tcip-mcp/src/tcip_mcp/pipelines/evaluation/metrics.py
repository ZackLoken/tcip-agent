"""Detection evaluation metrics — mAP, precision/recall curves."""

from __future__ import annotations

import numpy as np
import torch


def compute_ap(recalls: list[float], precisions: list[float]) -> float:
    """Compute Average Precision using the 11-point interpolation method.

    Args:
        recalls: List of recall values at each threshold.
        precisions: Corresponding precision values.

    Returns:
        AP score.
    """
    if not recalls or not precisions:
        return 0.0

    # Add sentinel values
    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [0.0] + list(precisions) + [0.0]

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        prec_at_recall = 0.0
        for r, p in zip(mrec, mpre):
            if r >= t:
                prec_at_recall = max(prec_at_recall, p)
        ap += prec_at_recall / 11.0

    return float(ap)


def compute_map(
    all_predictions: list[dict],
    all_targets: list[dict],
    iou_thresholds: list[float] | None = None,
    score_threshold: float = 0.01,
) -> dict:
    """Compute mAP over a dataset.

    Args:
        all_predictions: List of prediction dicts with 'boxes', 'scores', 'labels' tensors.
        all_targets: List of target dicts with 'boxes', 'labels' tensors.
        iou_thresholds: IoU thresholds for evaluation (default: [0.5]).
        score_threshold: Min score to consider.

    Returns:
        Dict with 'mAP', 'AP_per_class', per-threshold results.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    # Gather all class IDs
    all_classes: set[int] = set()
    for t in all_targets:
        if len(t["labels"]) > 0:
            all_classes.update(t["labels"].tolist())

    results_per_threshold: dict[float, dict] = {}

    for iou_thresh in iou_thresholds:
        ap_per_class: dict[int, float] = {}

        for cls in sorted(all_classes):
            # Collect all predictions and GT for this class
            pred_scores: list[float] = []
            pred_matched: list[bool] = []
            total_gt = 0

            for preds, targets in zip(all_predictions, all_targets):
                gt_boxes = targets["boxes"]
                gt_labels = targets["labels"]
                gt_mask = gt_labels == cls
                gt_cls_boxes = gt_boxes[gt_mask]
                total_gt += len(gt_cls_boxes)

                pred_boxes = preds["boxes"]
                pred_labels = preds["labels"]
                pred_scores_t = preds["scores"]

                cls_mask = (pred_labels == cls) & (pred_scores_t >= score_threshold)
                cls_boxes = pred_boxes[cls_mask]
                cls_scores = pred_scores_t[cls_mask]

                # Sort by confidence
                order = cls_scores.argsort(descending=True)
                cls_boxes = cls_boxes[order]
                cls_scores = cls_scores[order]

                matched_gt = set()
                for i in range(len(cls_boxes)):
                    pred_scores.append(cls_scores[i].item())
                    best_iou = 0.0
                    best_gt = -1
                    for j in range(len(gt_cls_boxes)):
                        if j in matched_gt:
                            continue
                        iou = _box_iou_single(cls_boxes[i], gt_cls_boxes[j])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = j
                    if best_iou >= iou_thresh and best_gt >= 0:
                        pred_matched.append(True)
                        matched_gt.add(best_gt)
                    else:
                        pred_matched.append(False)

            # Compute precision-recall curve
            if total_gt == 0:
                ap_per_class[cls] = 0.0
                continue

            # Sort all predictions by score
            indices = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)
            tp_cumsum = 0
            recalls = []
            precisions = []
            for rank, idx in enumerate(indices):
                if pred_matched[idx]:
                    tp_cumsum += 1
                precision = tp_cumsum / (rank + 1)
                recall = tp_cumsum / total_gt
                precisions.append(precision)
                recalls.append(recall)

            ap_per_class[cls] = compute_ap(recalls, precisions)

        mean_ap = sum(ap_per_class.values()) / max(len(ap_per_class), 1)
        results_per_threshold[iou_thresh] = {
            "mAP": round(mean_ap, 4),
            "AP_per_class": {k: round(v, 4) for k, v in ap_per_class.items()},
        }

    # Primary metric: mAP@0.5
    primary = results_per_threshold.get(0.5, results_per_threshold.get(iou_thresholds[0], {}))
    return {
        "mAP": primary.get("mAP", 0.0),
        "per_threshold": results_per_threshold,
    }


def _box_iou_single(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0].item(), box2[0].item())
    y1 = max(box1[1].item(), box2[1].item())
    x2 = min(box1[2].item(), box2[2].item())
    y2 = min(box1[3].item(), box2[3].item())

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]).item() * (box1[3] - box1[1]).item()
    area2 = (box2[2] - box2[0]).item() * (box2[3] - box2[1]).item()
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0
