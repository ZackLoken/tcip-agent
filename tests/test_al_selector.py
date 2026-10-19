"""Active learning selector: review_queue / unscoreable partitioning.

The fixtures follow the GenericPredictor output contract: detection dicts carry ``scores``;
classification/ordinal checkpoints (ComposedModel -> ``_format_other``) carry
``head{i}_confidences``, never an ``output`` key; regression checkpoints carry only
``head{i}_values``, no confidence-bearing key at all.
"""

from tcip_mcp.pipelines.active_learning.selector import (
    review_queue,
    unscoreable,
)


def _cls_pred(image: str, conf: float) -> dict:
    """Return a _format_other-shaped classification prediction (single image)."""
    return {
        "image": image,
        "width": 640,
        "height": 480,
        "head0_labels": [2],
        "head0_confidences": [conf],
        "head0_probabilities": [[(1.0 - conf) / 2, (1.0 - conf) / 2, conf]],
    }


def _ordinal_pred(image: str, rank: int, conf: float) -> dict:
    """Return a _format_other-shaped ordinal prediction of ``rank`` at confidence ``conf``.

    Its two cumulative probabilities both equal ``conf``.
    """
    return {
        "image": image,
        "width": 640,
        "height": 480,
        "head0_ranks": [rank],
        "head0_confidences": [conf],
        "head0_cumulative_probs": [[conf, conf]],
    }


def _reg_pred(image: str, value: float) -> dict:
    """Return a _format_other-shaped regression prediction whose only head key is its values."""
    return {
        "image": image,
        "width": 640,
        "height": 480,
        "head0_values": [value],
    }


def _seg_pred(image: str) -> dict:
    """Return a _format_other-shaped semantic-seg prediction (no confidences)."""
    return {
        "image": image,
        "width": 4,
        "height": 4,
        "head0_masks": [[[0, 1], [1, 0]]],
        "head0_probabilities": [
            [[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]]
        ],
    }


# ====================================================================
# review_queue
# ====================================================================


class TestReviewQueue:
    def test_detection_partitioning(self) -> None:
        predictions = [
            {"image": "a.png", "scores": [0.95]},
            {"image": "b.png", "scores": [0.5]},
            {"image": "c.png", "scores": [0.2]},
        ]
        queue = review_queue(predictions, low=0.3, high=0.8)
        assert [p["image"] for p in queue] == ["b.png"]

    def test_classification_partitioning_and_ordering(self) -> None:
        predictions = [
            _cls_pred("confident.png", 0.93),  # above high -> auto territory
            _cls_pred("mid.png", 0.6),
            _cls_pred("shaky.png", 0.35),
            _cls_pred("noise.png", 0.2),  # below low -> reject
        ]
        queue = review_queue(predictions, low=0.3, high=0.8)
        # Most uncertain first.
        assert [p["image"] for p in queue] == ["shaky.png", "mid.png"]

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
        assert [p["image"] for p in queue] == ["shaky.png", "mid.png"]

    def test_multi_head_gates_on_least_confident_head(self) -> None:
        pred = _cls_pred("multi.png", 0.95)
        pred["head1_confidences"] = [0.5]
        queue = review_queue([pred], low=0.3, high=0.8)
        assert queue == [pred]

    def test_seg_probabilities_are_ignored(self) -> None:
        assert review_queue([_seg_pred("mask.png")], low=0.0, high=1.0) == []

    def test_mixed_detection_and_classification_sorted_together(self) -> None:
        det = {"image": "det.png", "scores": [0.7, 0.4]}
        cls = _cls_pred("cls.png", 0.6)
        queue = review_queue([det, cls], low=0.3, high=0.8)
        assert [p["image"] for p in queue] == ["det.png", "cls.png"]


# ====================================================================
# unscoreable
# ====================================================================


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
        """A detection with no boxes is not unscoreable.

        scores=[] (zero boxes found) is a complete signal, and unscoreable leaves it out.
        """
        assert unscoreable([{"image": "neg.png", "scores": []}]) == []

    def test_detection_and_classification_and_ordinal_are_not_unscoreable(
        self,
    ) -> None:
        det = {"image": "det.png", "scores": [0.9]}
        cls = _cls_pred("cls.png", 0.9)
        ordinal = _ordinal_pred("ord.png", 1, 0.9)
        assert unscoreable([det, cls, ordinal]) == []
