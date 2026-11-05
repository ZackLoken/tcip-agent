"""Active learning selector: review_queue / unscoreable partitioning.

The fixtures follow the GenericPredictor output contract: a detector's pass carries its
detections' scores; classification/ordinal checkpoints (ComposedModel -> ``_format_other``) carry
``head{i}_confidences`` among their outputs, never an ``output`` key; regression checkpoints carry
only ``head{i}_values``, no confidence-bearing output at all.
"""

from pydantic import JsonValue

from tcip_mcp.pipelines.active_learning.selector import (
    review_queue,
    unscoreable,
)
from tcip_mcp.pipelines.inference.passes import (
    Detection,
    DetectionPass,
    HeadPass,
)


def _det(image: str, scores: list[float]) -> DetectionPass:
    """Return a detector's pass over ``image`` with one detection scored at each of ``scores``."""
    return DetectionPass(
        image=image,
        width=640,
        height=480,
        cap=len(scores) + 1,
        detections=[
            Detection(
                box=(0.0, 0.0, 1.0, 1.0),
                score=s,
                label=1,
                attributes=[],
                mask=None,
            )
            for s in scores
        ],
        soft_masks=None,
    )


def _head(image: str, size: int = 640, **outputs: JsonValue) -> HeadPass:
    """Return a _format_other-shaped head pass over ``image`` carrying ``outputs``."""
    return HeadPass(image=image, width=size, height=size, outputs=outputs)


def _cls_pred(image: str, conf: float, **extra: JsonValue) -> HeadPass:
    """Return a _format_other-shaped classification prediction (single image)."""
    return _head(
        image,
        head0_labels=[2],
        head0_confidences=[conf],
        head0_probabilities=[[(1.0 - conf) / 2, (1.0 - conf) / 2, conf]],
        **extra,
    )


def _ordinal_pred(image: str, rank: int, conf: float) -> HeadPass:
    """Return a _format_other-shaped ordinal prediction of ``rank`` at confidence ``conf``.

    Its two cumulative probabilities both equal ``conf``.
    """
    return _head(
        image,
        head0_ranks=[rank],
        head0_confidences=[conf],
        head0_cumulative_probs=[[conf, conf]],
    )


def _reg_pred(image: str, value: float) -> HeadPass:
    """Return a _format_other-shaped regression prediction whose only head key is its values."""
    return _head(image, head0_values=[value])


def _seg_pred(image: str) -> HeadPass:
    """Return a _format_other-shaped semantic-seg prediction (no confidences)."""
    return _head(
        image,
        4,
        head0_masks=[[[0, 1], [1, 0]]],
        head0_probabilities=[
            [[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]]
        ],
    )


class TestReviewQueue:
    def test_detection_partitioning(self) -> None:
        predictions = [
            _det("a.png", [0.95]),
            _det("b.png", [0.5]),
            _det("c.png", [0.2]),
        ]
        queue = review_queue(predictions, low=0.3, high=0.8)
        assert [p.image for p in queue] == ["b.png"]

    def test_classification_partitioning_and_ordering(self) -> None:
        predictions = [
            _cls_pred("confident.png", 0.93),  # above high -> auto territory
            _cls_pred("mid.png", 0.6),
            _cls_pred("shaky.png", 0.35),
            _cls_pred("noise.png", 0.2),  # below low -> reject
        ]
        queue = review_queue(predictions, low=0.3, high=0.8)
        # Most uncertain first.
        assert [p.image for p in queue] == ["shaky.png", "mid.png"]

    def test_ordinal_partitioning_and_ordering(self) -> None:
        predictions = [
            _ordinal_pred(
                "confident.png", 2, 0.93
            ),  # above high -> auto territory
            _ordinal_pred("mid.png", 1, 0.6),
            _ordinal_pred("shaky.png", 1, 0.35),
            _ordinal_pred("noise.png", 0, 0.2),  # below low -> reject
        ]
        queue = review_queue(predictions, low=0.3, high=0.8)
        assert [p.image for p in queue] == ["shaky.png", "mid.png"]

    def test_multi_head_gates_on_least_confident_head(self) -> None:
        pred = _cls_pred("multi.png", 0.95, head1_confidences=[0.5])
        queue = review_queue([pred], low=0.3, high=0.8)
        assert queue == [pred]

    def test_seg_probabilities_are_ignored(self) -> None:
        assert review_queue([_seg_pred("mask.png")], low=0.0, high=1.0) == []

    def test_mixed_detection_and_classification_sorted_together(self) -> None:
        det = _det("det.png", [0.7, 0.4])
        cls = _cls_pred("cls.png", 0.6)
        queue = review_queue([det, cls], low=0.3, high=0.8)
        assert [p.image for p in queue] == ["det.png", "cls.png"]


class TestUnscoreable:
    def test_regression_prediction_has_no_confidence_signal(self) -> None:
        """A regression prediction is left out of review_queue and returned by unscoreable."""
        pred = _reg_pred("val.png", 0.42)
        assert review_queue([pred], low=0.0, high=1.0) == []
        assert unscoreable([pred]) == [pred]

    def test_seg_prediction_with_no_confidences_is_unscoreable(self) -> None:
        # 4-D head0_probabilities alone carries no usable per-instance confidence either.
        pred = _seg_pred("mask.png")
        assert unscoreable([pred]) == [pred]

    def test_detection_negative_is_not_unscoreable(self) -> None:
        """A detector's pass with no detection is not unscoreable.

        No detection found is a complete signal, and unscoreable leaves it out.
        """
        assert unscoreable([_det("neg.png", [])]) == []

    def test_detection_and_classification_and_ordinal_are_not_unscoreable(
        self,
    ) -> None:
        det = _det("det.png", [0.9])
        cls = _cls_pred("cls.png", 0.9)
        ordinal = _ordinal_pred("ord.png", 1, 0.9)
        assert unscoreable([det, cls, ordinal]) == []
