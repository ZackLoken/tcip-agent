"""Active learning selector: review-queue and unscoreable partitioning of predictions.

The review queue is the medium-confidence predictions; both partition a checkpoint's own passes.
"""

from __future__ import annotations

from typing import cast

from tcip_mcp.pipelines.inference.passes import DetectionPass, HeadPass


def _confidence_values(pred: DetectionPass | HeadPass) -> list[float]:
    """Return a pass's flat confidences: a detector's detection scores, a head's image-level ones.

    Classification/ordinal heads emit per-image confidences under ``head{i}_confidences``; matching
    the suffix covers a model with any number of heads. ``*_probabilities`` is not matched.
    """
    if isinstance(pred, DetectionPass):
        return [d.score for d in pred.detections]
    values: list[float] = []
    for key, val in pred.outputs.items():
        if key.endswith("_confidences") and isinstance(val, list):
            values.extend(map(float, cast("list[float]", val)))
    return values


def unscoreable(
    predictions: list[DetectionPass | HeadPass],
) -> list[DetectionPass | HeadPass]:
    """Return the predictions with no confidence-bearing signal.

    That is a head's pass with no ``*_confidences`` output; a detector's pass is scored, one with
    no detection a scored negative.
    """
    return [
        pred
        for pred in predictions
        if not isinstance(pred, DetectionPass) and not _confidence_values(pred)
    ]


def review_queue(
    predictions: list[DetectionPass | HeadPass], low: float, high: float
) -> list[DetectionPass | HeadPass]:
    """Select predictions needing human review (medium confidence).

    Returns predictions whose least-confident detection or head confidence
    falls in ``[low, high)``, sorted by lowest confidence first
    (most uncertain = review first).
    """
    queue = []
    for pred in predictions:
        confs = _confidence_values(pred)
        if confs:
            min_conf = min(confs)
            if low <= min_conf < high:
                queue.append((min_conf, pred))

    queue.sort(key=lambda x: x[0])
    return [pred for _, pred in queue]
