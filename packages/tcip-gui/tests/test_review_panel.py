"""Tests for the review panel."""

import pytest

from tcip_gui.panels.review_panel import ReviewPanel


class TestReviewPanel:
    def test_creation(self):
        panel = ReviewPanel()
        assert panel._current_idx == -1
        assert panel._detections == []

    def test_set_detections(self):
        panel = ReviewPanel()
        detections = [
            {"det_type": "tp", "class_id": 0, "conf": 0.95, "iou": 0.82},
            {"det_type": "fp", "class_id": 0, "conf": 0.60, "iou": 0.12},
            {"det_type": "fn", "class_id": 1, "conf": 0.0, "iou": 0.0},
        ]
        panel.set_detections(detections)
        assert panel._current_idx == 0
        assert panel._tp_label.text() == "TP: 1"
        assert panel._fp_label.text() == "FP: 1"
        assert panel._fn_label.text() == "FN: 1"

    def test_set_empty_detections(self):
        panel = ReviewPanel()
        panel.set_detections([])
        assert panel._current_idx == -1

    def test_navigation(self):
        panel = ReviewPanel()
        detections = [
            {"det_type": "tp", "class_id": 0, "conf": 0.9, "iou": 0.8},
            {"det_type": "fp", "class_id": 0, "conf": 0.5, "iou": 0.1},
        ]
        panel.set_detections(detections)
        assert panel._current_idx == 0

        panel._on_next()
        assert panel._current_idx == 1

        panel._on_next()  # at end, should stay
        assert panel._current_idx == 1

        panel._on_prev()
        assert panel._current_idx == 0

        panel._on_prev()  # at start, should stay
        assert panel._current_idx == 0

    def test_review_action_signal(self):
        panel = ReviewPanel()
        detections = [
            {"det_type": "tp", "class_id": 0, "conf": 0.9, "iou": 0.8},
            {"det_type": "fp", "class_id": 0, "conf": 0.5, "iou": 0.1},
        ]
        panel.set_detections(detections)

        received = []
        panel.review_action.connect(lambda idx, action: received.append((idx, action)))

        panel._on_action("accept")
        assert received == [(0, "accept")]
        # Should auto-advance to next
        assert panel._current_idx == 1
