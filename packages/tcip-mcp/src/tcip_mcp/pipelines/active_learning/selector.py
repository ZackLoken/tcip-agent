"""Active learning selector: partition a checkpoint's own predictions.

Auto-accept (high-confidence), review-queue (medium-confidence) and unscoreable partitioning over
prediction dicts. Ranking unlabeled images by informativeness is the scorers' own seam
(``active_learning.scorer``), which reads the predictor rather than these dicts.
"""

from __future__ import annotations


def _confidence_values(pred: dict) -> list[float]:
    """Flat image-level confidences from a GenericPredictor prediction dict.

    Classification/ordinal heads emit per-image confidences under ``head{i}_confidences``; matching
    the suffix covers a model with any number of heads. ``*_probabilities`` is not matched.
    """
    values: list[float] = []
    for key, val in pred.items():
        if key.endswith("_confidences") and isinstance(val, list):
            values.extend(float(v) for v in val)
    return values


def unscoreable(predictions: list[dict]) -> list[dict]:
    """Predictions with no confidence-bearing signal: no ``scores`` key (checked by presence;
    ``scores: []`` is a scored negative) and no ``*_confidences`` head output.
    """
    return [pred for pred in predictions
            if "scores" not in pred and not _confidence_values(pred)]


def auto_accept(
    predictions: list[dict],
    *,
    threshold: float,
) -> list[dict]:
    """Filter predictions confident enough for automatic labeling.

    Args:
        predictions: List of prediction dicts (from GenericPredictor).
        threshold: Minimum confidence score for auto-acceptance; required.

    Returns:
        Predictions where every detection/classification exceeds threshold.
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
