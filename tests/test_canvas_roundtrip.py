"""Canvas round-trip tests: verify annotation pipeline end-to-end.

These tests simulate what the annotation canvas webview does:
1. Load image → read existing labels
2. Add/modify annotations (boxes + polygons)
3. Save the single name-based per-image label file
4. Read back and verify content matches

Also tests prediction overlay data roundtrip for the review panel.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation import (
    Annotation,
    AnnotationState,
    BBox,
    Polygon,
    compute_matches,
    read_annotations,
    write_annotations,
)


# ── Setup ──


@pytest.fixture
def img_dir(tmp_path: Path) -> Path:
    """Minimal image dataset directory structure."""
    images = tmp_path / "images"
    images.mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "predictions").mkdir()

    # Create a minimal test image
    from PIL import Image

    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    img.save(images / "test_001.jpg")
    return tmp_path


# ── Box roundtrip ──


class TestBoxRoundtrip:
    """Load → draw boxes → save → read back."""

    def test_draw_save_reload(self, img_dir: Path) -> None:
        """Draw two boxes, save, reload and verify."""
        state = AnnotationState(
            image_path=str(img_dir / "images" / "test_001.jpg"),
            img_width=640,
            img_height=480,
        )

        # Draw boxes (simulating canvas click-drag)
        state.annotations.append(Annotation(subject="bud", geometry=BBox(x1=100, y1=50, x2=250, y2=200)))
        state.annotations.append(Annotation(subject="nut", geometry=BBox(x1=400, y1=300, x2=550, y2=420)))
        assert len(state.annotations) == 2

        # Save
        label_path = str(img_dir / "labels" / "test_001.json")
        write_annotations(label_path, state.annotations, 640, 480)

        # Read back
        read_back = read_annotations(label_path)
        assert len(read_back) == 2
        assert {a.subject for a in read_back} == {"bud", "nut"}

        # Verify coordinates (within float precision)
        for orig, loaded in zip(state.annotations, read_back):
            assert abs(orig.geometry.x1 - loaded.geometry.x1) < 2
            assert abs(orig.geometry.y1 - loaded.geometry.y1) < 2
            assert abs(orig.geometry.x2 - loaded.geometry.x2) < 2
            assert abs(orig.geometry.y2 - loaded.geometry.y2) < 2
            assert orig.subject == loaded.subject

    def test_undo_redo_then_save(self, img_dir: Path) -> None:
        """Draw boxes, verify save: state management is tested elsewhere."""
        state = AnnotationState(img_width=640, img_height=480)

        state.annotations.append(Annotation(subject="bud", geometry=BBox(x1=10, y1=10, x2=100, y2=100)))
        state.annotations.append(Annotation(subject="nut", geometry=BBox(x1=200, y1=200, x2=300, y2=300)))
        assert len(state.annotations) == 2

        label_path = str(img_dir / "labels" / "test_001.json")
        write_annotations(label_path, state.annotations, 640, 480)

        read_back = read_annotations(label_path)
        assert len(read_back) == 2


# ── Polygon roundtrip ──


class TestPolygonRoundtrip:
    """Load → draw polygons → save → read back."""

    def test_draw_polygon_save_reload(self, img_dir: Path) -> None:
        """Draw a polygon, save, reload and verify."""
        state = AnnotationState(
            image_path=str(img_dir / "images" / "test_001.jpg"),
            img_width=640,
            img_height=480,
        )

        # Draw a triangle polygon (simulating canvas vertex clicks)
        poly = Polygon(rings=[[(100.0, 50.0), (250.0, 200.0), (50.0, 200.0)]])
        state.annotations.append(Annotation(subject="bud", geometry=poly))
        assert len(state.annotations) == 1

        # Save
        label_path = str(img_dir / "labels" / "test_001.json")
        write_annotations(label_path, state.annotations, 640, 480)

        # Read back
        read_back = read_annotations(label_path)
        assert len(read_back) == 1
        assert read_back[0].subject == "bud"
        assert isinstance(read_back[0].geometry, Polygon)
        assert len(read_back[0].geometry.rings[0]) == 3

        # Verify coordinates
        for (ox, oy), (lx, ly) in zip(poly.rings[0], read_back[0].geometry.rings[0]):
            assert abs(ox - lx) < 2
            assert abs(oy - ly) < 2

    def test_multi_class_polygon(self, img_dir: Path) -> None:
        """Multiple polygons with different subjects."""
        state = AnnotationState(img_width=640, img_height=480)

        state.annotations.append(Annotation(
            subject="bud", geometry=Polygon(rings=[[(10, 10), (100, 10), (100, 100), (10, 100)]])))
        state.annotations.append(Annotation(
            subject="leaf", geometry=Polygon(rings=[[(200, 200), (300, 200), (300, 300), (200, 300)]])))

        label_path = str(img_dir / "labels" / "test_001.json")
        write_annotations(label_path, state.annotations, 640, 480)

        read_back = read_annotations(label_path)
        assert len(read_back) == 2
        assert {a.subject for a in read_back} == {"bud", "leaf"}

    def test_occlusion_split_prediction_survives_the_round_trip(self, img_dir: Path) -> None:
        """A model-predicted mask the review canvas overlays can be occlusion-split: one instance,
        two contours. Both rings must come back so the reviewer sees the whole object."""
        rings = [
            [(100.0, 50.0), (150.0, 50.0), (150.0, 200.0), (100.0, 200.0)],
            [(300.0, 60.0), (350.0, 60.0), (350.0, 190.0), (300.0, 190.0)],
        ]
        pred_path = str(img_dir / "predictions" / "test_001.json")
        write_annotations(
            pred_path,
            [Annotation(subject="bud", geometry=Polygon(rings=rings), score=0.88)],
            640, 480)

        read_back = read_annotations(pred_path)
        assert len(read_back) == 1  # one instance, not one annotation per contour
        assert read_back[0].geometry.rings == rings
        assert read_back[0].score == 0.88


# ── Single-file save (boxes + polygons in one per-image file) ──


class TestSingleFileSave:
    """The canvas saves every subject, boxes and polygons alike, into one per-image file."""

    def test_box_and_polygon_single_file(self, img_dir: Path) -> None:
        """Save a box and a polygon together, verify both survive in one file."""
        state = AnnotationState(img_width=640, img_height=480)

        state.annotations.append(Annotation(subject="bud", geometry=BBox(x1=50, y1=50, x2=200, y2=150)))
        state.annotations.append(Annotation(
            subject="nut", geometry=Polygon(rings=[[(300, 100), (400, 100), (400, 200), (300, 200)]])))

        label_path = str(img_dir / "labels" / "test_001.json")
        write_annotations(label_path, state.annotations, 640, 480)

        read_back = read_annotations(label_path)
        assert len(read_back) == 2
        box_ann = next(a for a in read_back if isinstance(a.geometry, BBox))
        poly_ann = next(a for a in read_back if isinstance(a.geometry, Polygon))
        assert box_ann.subject == "bud"
        assert poly_ann.subject == "nut"


# ── Prediction overlay verification ──


class TestPredictionOverlay:
    """Verify that predictions load correctly and matching produces expected TP/FP/FN."""

    def test_prediction_overlay_matching(self, img_dir: Path) -> None:
        """Load GT + predictions, compute matches, verify overlay data."""
        # Write GT
        gt = [
            Annotation(subject="bud", geometry=BBox(x1=100, y1=50, x2=250, y2=200)),
            Annotation(subject="bud", geometry=BBox(x1=400, y1=300, x2=550, y2=420)),
        ]
        gt_path = str(img_dir / "labels" / "test_001.json")
        write_annotations(gt_path, gt, 640, 480)

        # Write predictions (one matching, one FP)
        preds = [
            Annotation(subject="bud", geometry=BBox(x1=105, y1=55, x2=245, y2=195), score=0.92),
            Annotation(subject="bud", geometry=BBox(x1=10, y1=10, x2=50, y2=50), score=0.75),
        ]
        pred_path = str(img_dir / "predictions" / "test_001.json")
        write_annotations(pred_path, preds, 640, 480)

        # Now load and match
        loaded_gt = read_annotations(gt_path)
        loaded_preds = read_annotations(pred_path)

        matches = compute_matches(loaded_gt, loaded_preds, iou_threshold=0.5, conf_threshold=0.25)

        assert len(matches["tp"]) == 1, "Expected 1 TP (overlapping prediction)"
        assert len(matches["fp"]) == 1, "Expected 1 FP (non-overlapping prediction)"
        assert len(matches["fn"]) == 1, "Expected 1 FN (unmatched GT box)"

    def test_five_distinct_subjects_survive_the_round_trip(self, img_dir: Path) -> None:
        """Coverage: five annotations of five distinct subjects, written and read back through
        the same per-image label file, all five subjects and the count surviving the round
        trip."""
        subjects = ["bud", "shoot", "leaf", "nut", "bush"]
        annotations = [
            Annotation(subject=subj, geometry=BBox(x1=i * 100, y1=10, x2=i * 100 + 80, y2=90))
            for i, subj in enumerate(subjects)
        ]

        label_path = str(img_dir / "labels" / "test_001.json")
        write_annotations(label_path, annotations, 640, 480)

        read_back = read_annotations(label_path)
        assert len(read_back) == 5
        assert {a.subject for a in read_back} == set(subjects)
