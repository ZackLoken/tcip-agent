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


def _confidence_values(pred: dict) -> list[float]:
    """Flat image-level confidences from a GenericPredictor prediction dict.

    Classification/ordinal heads emit per-image confidences under
    ``head{i}_confidences`` (via ComposedModel -> ``_format_other``); matching
    the suffix covers multi-head specs. ``*_probabilities`` is deliberately
    NOT matched — SemanticSegHead emits it as a 4-D nested list.
    """
    values: list[float] = []
    for key, val in pred.items():
        if key.endswith("_confidences") and isinstance(val, list):
            values.extend(float(v) for v in val)
    return values


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
        if scores:
            # Detection: every kept box must clear the threshold.
            if all(s >= threshold for s in scores):
                accepted.append(pred)
        else:
            # Classification/ordinal: every head's confidence must clear it.
            confs = _confidence_values(pred)
            if confs and all(c >= threshold for c in confs):
                accepted.append(pred)
    return accepted


def review_queue(
    predictions: list[dict],
    low: float = 0.3,
    high: float = 0.8,
) -> list[dict]:
    """Select predictions needing human review (medium confidence).

    Returns predictions whose least-confident detection or head confidence
    falls between low and high, sorted by lowest confidence first
    (most uncertain = review first).
    """
    queue = []
    for pred in predictions:
        scores = pred.get("scores", [])
        confs = scores if scores else _confidence_values(pred)
        if confs:
            min_conf = min(confs)
            if low <= min_conf < high:
                queue.append((min_conf, pred))

    queue.sort(key=lambda x: x[0])
    return [pred for _, pred in queue]
