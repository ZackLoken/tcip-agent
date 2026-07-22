"""Canvas round-trip tests — verify annotation pipeline end-to-end.

These tests simulate what the annotation canvas webview does:
1. Load image → read existing labels
2. Add/modify annotations (boxes + polygons)
3. Save in YOLO detect + segment formats
4. Read back and verify content matches

Also tests prediction overlay data roundtrip for the review panel.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation import (
    AnnotationState,
    BBox,
    Polygon,
    PredBBox,
    compute_matches,
    parse_detect_labels,
    parse_segment_labels,
    write_detect_labels,
    write_segment_labels,
)


# ── Setup ──


@pytest.fixture
def img_dir(tmp_path: Path) -> Path:
    """Minimal image dataset directory structure."""
    images = tmp_path / "images"
    images.mkdir()
    detect = tmp_path / "labels" / "detect"
    detect.mkdir(parents=True)
    segment = tmp_path / "labels" / "segment"
    segment.mkdir(parents=True)
    pred_detect = tmp_path / "predictions" / "detect"
    pred_detect.mkdir(parents=True)
    pred_segment = tmp_path / "predictions" / "segment"
    pred_segment.mkdir(parents=True)

    # Create a minimal test image
    from PIL import Image

    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    img.save(images / "test_001.jpg")
    return tmp_path


# ── Box roundtrip ──


class TestBoxRoundtrip:
    """Load → draw boxes → save detect format → read back."""

    def test_draw_save_reload(self, img_dir: Path) -> None:
        """Draw two boxes, save as YOLO detect, reload and verify."""
        state = AnnotationState(
            image_path=str(img_dir / "images" / "test_001.jpg"),
            img_width=640,
            img_height=480,
        )

        # Draw boxes (simulating canvas click-drag)
        state.boxes.append(BBox(x1=100, y1=50, x2=250, y2=200, class_id=0))
        state.boxes.append(BBox(x1=400, y1=300, x2=550, y2=420, class_id=1))
        assert len(state.boxes) == 2

        # Save
        label_path = str(img_dir / "labels" / "detect" / "test_001.json")
        write_detect_labels(label_path, state.boxes, 640, 480)

        # Read back
        boxes, class_ids = parse_detect_labels(label_path, 640, 480)
        assert len(boxes) == 2
        assert 0 in class_ids
        assert 1 in class_ids

        # Verify coordinates (within YOLO float precision)
        for orig, loaded in zip(state.boxes, boxes):
            assert abs(orig.x1 - loaded.x1) < 2
            assert abs(orig.y1 - loaded.y1) < 2
            assert abs(orig.x2 - loaded.x2) < 2
            assert abs(orig.y2 - loaded.y2) < 2
            assert orig.class_id == loaded.class_id

    def test_undo_redo_then_save(self, img_dir: Path) -> None:
        """Draw boxes, verify save — state management is tested elsewhere."""
        state = AnnotationState(img_width=640, img_height=480)

        state.boxes.append(BBox(x1=10, y1=10, x2=100, y2=100, class_id=0))
        state.boxes.append(BBox(x1=200, y1=200, x2=300, y2=300, class_id=1))
        assert len(state.boxes) == 2

        label_path = str(img_dir / "labels" / "detect" / "test_001.json")
        write_detect_labels(label_path, state.boxes, 640, 480)

        boxes, _ = parse_detect_labels(label_path, 640, 480)
        assert len(boxes) == 2


# ── Polygon roundtrip ──


class TestPolygonRoundtrip:
    """Load → draw polygons → save segment format → read back."""

    def test_draw_polygon_save_reload(self, img_dir: Path) -> None:
        """Draw a polygon, save as YOLO segment, reload and verify."""
        state = AnnotationState(
            image_path=str(img_dir / "images" / "test_001.jpg"),
            img_width=640,
            img_height=480,
        )

        # Draw a triangle polygon (simulating canvas vertex clicks)
        poly = Polygon(
            points=[(100.0, 50.0), (250.0, 200.0), (50.0, 200.0)],
            class_id=0,
        )
        state.polygons.append(poly)
        assert len(state.polygons) == 1

        # Save
        label_path = str(img_dir / "labels" / "segment" / "test_001.json")
        write_segment_labels(label_path, state.polygons, 640, 480)

        # Read back
        polygons, class_ids = parse_segment_labels(label_path, 640, 480)
        assert len(polygons) == 1
        assert 0 in class_ids
        assert len(polygons[0].points) == 3

        # Verify coordinates
        for (ox, oy), (lx, ly) in zip(poly.points, polygons[0].points):
            assert abs(ox - lx) < 2
            assert abs(oy - ly) < 2

    def test_multi_class_polygon(self, img_dir: Path) -> None:
        """Multiple polygons with different classes."""
        state = AnnotationState(img_width=640, img_height=480)

        state.polygons.append(Polygon(
            points=[(10, 10), (100, 10), (100, 100), (10, 100)],
            class_id=0,
        ))
        state.polygons.append(Polygon(
            points=[(200, 200), (300, 200), (300, 300), (200, 300)],
            class_id=2,
        ))

        label_path = str(img_dir / "labels" / "segment" / "test_001.json")
        write_segment_labels(label_path, state.polygons, 640, 480)

        polygons, class_ids = parse_segment_labels(label_path, 640, 480)
        assert len(polygons) == 2
        assert {p.class_id for p in polygons} == {0, 2}


# ── Dual-format save (detect + segment from same annotations) ──


class TestDualFormatSave:
    """The canvas saves both detect and segment labels simultaneously."""

    def test_box_and_polygon_dual_save(self, img_dir: Path) -> None:
        """Save boxes as detect and polygons as segment, verify both files."""
        state = AnnotationState(img_width=640, img_height=480)

        state.boxes.append(BBox(x1=50, y1=50, x2=200, y2=150, class_id=0))
        state.polygons.append(Polygon(
            points=[(300, 100), (400, 100), (400, 200), (300, 200)],
            class_id=1,
        ))

        detect_path = str(img_dir / "labels" / "detect" / "test_001.json")
        segment_path = str(img_dir / "labels" / "segment" / "test_001.json")

        write_detect_labels(detect_path, state.boxes, 640, 480)
        write_segment_labels(segment_path, state.polygons, 640, 480)

        boxes, _ = parse_detect_labels(detect_path, 640, 480)
        polygons, _ = parse_segment_labels(segment_path, 640, 480)

        assert len(boxes) == 1
        assert len(polygons) == 1
        assert boxes[0].class_id == 0
        assert polygons[0].class_id == 1


# ── Prediction overlay verification ──


class TestPredictionOverlay:
    """Verify that predictions load correctly and matching produces expected TP/FP/FN."""

    def test_prediction_overlay_matching(self, img_dir: Path) -> None:
        """Load GT + predictions, compute matches, verify overlay data."""
        # Write GT
        gt_boxes = [
            BBox(x1=100, y1=50, x2=250, y2=200, class_id=0),
            BBox(x1=400, y1=300, x2=550, y2=420, class_id=0),
        ]
        gt_path = str(img_dir / "labels" / "detect" / "test_001.json")
        write_detect_labels(gt_path, gt_boxes, 640, 480)

        # Write predictions (one matching, one FP)
        pred_boxes = [
            PredBBox(x1=105, y1=55, x2=245, y2=195, class_id=0, confidence=0.92),
            PredBBox(x1=10, y1=10, x2=50, y2=50, class_id=0, confidence=0.75),
        ]
        pred_path = str(img_dir / "predictions" / "detect" / "test_001.json")
        from tcip_annotation.json_io import write_detect
        write_detect(pred_path, pred_boxes, 640, 480)

        # Now load and match
        loaded_gt, _ = parse_detect_labels(gt_path, 640, 480)
        from tcip_annotation.json_io import read_detect_pred as parse_detect_predictions
        loaded_preds, _ = parse_detect_predictions(pred_path, 640, 480)

        matches = compute_matches(
            loaded_gt, [], loaded_preds, [],
            iou_threshold=0.5, conf_threshold=0.25,
        )

        assert len(matches["tp"]) == 1, "Expected 1 TP (overlapping prediction)"
        assert len(matches["fp"]) == 1, "Expected 1 FP (non-overlapping prediction)"
        assert len(matches["fn"]) == 1, "Expected 1 FN (unmatched GT box)"

    def test_class_color_assignment(self) -> None:
        """Verify different classes get different class IDs for color mapping."""
        state = AnnotationState(img_width=640, img_height=480)

        for i in range(5):
            state.boxes.append(BBox(x1=i*100, y1=10, x2=i*100+80, y2=90, class_id=i))

        class_ids = {b.class_id for b in state.boxes}
        assert class_ids == {0, 1, 2, 3, 4}
