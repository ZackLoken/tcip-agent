"""Active learning selector: partition a checkpoint's own predictions.

Review-queue (medium-confidence) and unscoreable partitioning over prediction dicts. Ranking
unlabeled images by informativeness is the scorers' own seam (``active_learning.scorer``), which
reads the predictor rather than these dicts.
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
