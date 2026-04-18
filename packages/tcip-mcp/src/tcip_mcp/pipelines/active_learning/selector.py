"""Active learning selector — pick next images to annotate.

Uses scorers to rank unlabeled images, then selects top-N.
Also provides auto-accept (high-confidence) and review-queue
(medium-confidence) partitioning.
"""

from __future__ import annotations

import logging

import torch

from tcip_mcp.pipelines.active_learning.scorer import BaseScorer

logger = logging.getLogger(__name__)


def select_batch(
    scorer: BaseScorer,
    unlabeled_paths: list[str],
    model: torch.nn.Module,
    device: torch.device,
    budget: int = 50,
) -> list[str]:
    """Select the top-N most informative images to annotate next.

    Args:
        scorer: A BaseScorer instance (uncertainty, diversity, or combined).
        unlabeled_paths: List of paths to unlabeled images.
        model: Trained model for scoring.
        device: Torch device.
        budget: Number of images to select.

    Returns:
        List of image paths to annotate, ordered by informativeness.
    """
    scored = scorer.score(unlabeled_paths, model, device)
    return [path for path, _ in scored[:budget]]


def auto_accept(
    predictions: list[dict],
    threshold: float = 0.8,
) -> list[dict]:
    """Filter predictions confident enough for automatic labeling.

    Args:
        predictions: List of prediction dicts (from GenericPredictor).
        threshold: Minimum confidence score for auto-acceptance.

    Returns:
        Predictions where ALL detections/classifications exceed threshold.
    """
    accepted = []
    for pred in predictions:
        scores = pred.get("scores", [])
        if scores and all(s >= threshold for s in scores):
            accepted.append(pred)
        elif "output" in pred:
            # Classification: check max softmax prob
            output = pred["output"]
            if isinstance(output, list) and len(output) > 0:
                if isinstance(output[0], list):
                    max_prob = max(max(row) for row in output)
                else:
                    max_prob = max(output)
                if max_prob >= threshold:
                    accepted.append(pred)
    return accepted


def review_queue(
    predictions: list[dict],
    low: float = 0.3,
    high: float = 0.8,
) -> list[dict]:
    """Select predictions needing human review (medium confidence).

    Returns predictions where at least one score is between low and high,
    sorted by lowest confidence first (most uncertain = review first).
    """
    queue = []
    for pred in predictions:
        scores = pred.get("scores", [])
        if scores:
            min_score = min(scores)
            if low <= min_score < high:
                queue.append((min_score, pred))
        elif "output" in pred:
            output = pred["output"]
            if isinstance(output, list) and len(output) > 0:
                if isinstance(output[0], list):
                    max_prob = max(max(row) for row in output)
                else:
                    max_prob = max(output)
                if low <= max_prob < high:
                    queue.append((max_prob, pred))

    queue.sort(key=lambda x: x[0])
    return [pred for _, pred in queue]
