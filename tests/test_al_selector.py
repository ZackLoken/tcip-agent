"""Active learning selector — auto_accept / review_queue partitioning.

Covers the GenericPredictor output contract: detection dicts carry ``scores``;
classification/ordinal checkpoints (ComposedModel -> ``_format_other``) carry
``head{i}_confidences`` — never an ``output`` key.
"""

from tcip_mcp.pipelines.active_learning.selector import auto_accept, review_queue


def _cls_pred(image: str, conf: float) -> dict:
    """A _format_other-shaped classification prediction (single image)."""
    return {
        "image": image,
        "width": 640,
        "height": 480,
        "head0_labels": [2],
        "head0_confidences": [conf],
        "head0_probabilities": [[(1.0 - conf) / 2, (1.0 - conf) / 2, conf]],
    }


def _seg_pred(image: str) -> dict:
    """A _format_other-shaped semantic-seg prediction (no confidences)."""
    return {
        "image": image,
        "width": 4,
        "height": 4,
        "head0_masks": [[[0, 1], [1, 0]]],
        "head0_probabilities": [[[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]]],
    }


# ====================================================================
# auto_accept
# ====================================================================

class TestAutoAccept:
    def test_detection_partitioning(self):
        predictions = [
            {"image": "a.png", "scores": [0.95, 0.9]},
            {"image": "b.png", "scores": [0.3]},
            {"image": "c.png", "scores": [0.85, 0.82]},
            {"image": "d.png", "scores": [0.9, 0.5]},  # one weak box blocks accept
        ]
        accepted = auto_accept(predictions, threshold=0.8)
        assert [p["image"] for p in accepted] == ["a.png", "c.png"]

    def test_classification_partitioning(self):
        predictions = [
            _cls_pred("hi.png", 0.93),
            _cls_pred("lo.png", 0.55),
            _cls_pred("edge.png", 0.8),  # threshold is inclusive
        ]
        accepted = auto_accept(predictions, threshold=0.8)
        assert [p["image"] for p in accepted] == ["hi.png", "edge.png"]

    def test_integer_labels_not_mistaken_for_confidence(self):
        # head0_labels holds class indices (here 2 > threshold); only
        # head0_confidences may gate acceptance.
        pred = _cls_pred("lo.png", 0.1)
        assert auto_accept([pred], threshold=0.8) == []

    def test_multi_head_requires_all_heads_confident(self):
        pred = _cls_pred("multi.png", 0.95)
        pred["head1_confidences"] = [0.4]
        assert auto_accept([pred], threshold=0.8) == []
        pred["head1_confidences"] = [0.9]
        assert auto_accept([pred], threshold=0.8) == [pred]

    def test_seg_probabilities_are_ignored(self):
        # 4-D nested head0_probabilities must not be treated as confidence.
        assert auto_accept([_seg_pred("mask.png")], threshold=0.1) == []

    def test_empty_detection_not_accepted(self):
        assert auto_accept([{"image": "neg.png", "scores": []}], threshold=0.8) == []


# ====================================================================
# review_queue
# ====================================================================

class TestReviewQueue:
    def test_detection_partitioning(self):
        predictions = [
            {"image": "a.png", "scores": [0.95]},
            {"image": "b.png", "scores": [0.5]},
            {"image": "c.png", "scores": [0.2]},
        ]
        queue = review_queue(predictions, low=0.3, high=0.8)
        assert [p["image"] for p in queue] == ["b.png"]

    def test_classification_partitioning_and_ordering(self):
        predictions = [
            _cls_pred("confident.png", 0.93),  # above high -> auto territory
            _cls_pred("mid.png", 0.6),
            _cls_pred("shaky.png", 0.35),
            _cls_pred("noise.png", 0.2),  # below low -> reject
        ]
        queue = review_queue(predictions, low=0.3, high=0.8)
        # Most uncertain first.
        assert [p["image"] for p in queue] == ["shaky.png", "mid.png"]

    def test_multi_head_gates_on_least_confident_head(self):
        pred = _cls_pred("multi.png", 0.95)
        pred["head1_confidences"] = [0.5]
        queue = review_queue([pred], low=0.3, high=0.8)
        assert queue == [pred]

    def test_seg_probabilities_are_ignored(self):
        assert review_queue([_seg_pred("mask.png")], low=0.0, high=1.0) == []

    def test_mixed_detection_and_classification_sorted_together(self):
        det = {"image": "det.png", "scores": [0.7, 0.4]}
        cls = _cls_pred("cls.png", 0.6)
        queue = review_queue([det, cls], low=0.3, high=0.8)
        assert [p["image"] for p in queue] == ["det.png", "cls.png"]
