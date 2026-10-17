"""Active learning selector: review-queue and unscoreable partitioning of predictions.

The review queue is the medium-confidence predictions; both partition a checkpoint's own
prediction dicts.
"""

from __future__ import annotations


def _confidence_values(pred: dict) -> list[float]:
    """Return flat image-level confidences from a GenericPredictor prediction dict.

    Classification/ordinal heads emit per-image confidences under ``head{i}_confidences``; matching
    the suffix covers a model with any number of heads. ``*_probabilities`` is not matched.
    """
    values: list[float] = []
    for key, val in pred.items():
        if key.endswith("_confidences") and isinstance(val, list):
            values.extend(float(v) for v in val)
    return values


def unscoreable(predictions: list[dict]) -> list[dict]:
    """Return the predictions with no confidence-bearing signal.

    That is no ``scores`` key (checked by presence; ``scores: []`` is a scored negative) and no
    ``*_confidences`` head output.
    """
    return [
        pred
        for pred in predictions
        if "scores" not in pred and not _confidence_values(pred)
    ]


def review_queue(
    predictions: list[dict], low: float, high: float
) -> list[dict]:
    """Select predictions needing human review (medium confidence).

    Returns predictions whose least-confident detection or head confidence
    falls in ``[low, high)``, sorted by lowest confidence first
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
