"""Canvas round-trip tests of the annotation pipeline, end to end.

These tests simulate what the annotation canvas webview does:
1. Load image → read existing labels
2. Add/modify annotations (boxes + polygons)
3. Save the single name-based per-image label document
4. Read back and verify content matches

Also tests prediction overlay data roundtrip for the proposals the editor shows.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation import Annotation, BBox, Polygon
from tcip_annotation.json_io import read_label_document, write_label_document
from tcip_annotation.matching import pair_proposals
from tcip_mcp.dataset_layout import UNDATED_BUCKET, label_key, prediction_key


def _write(key, annotations) -> list[Annotation]:
    """Write ``annotations`` as the 640x480 document ``key`` names and return them read back."""
    write_label_document(key, annotations, 640, 480)
    return read_label_document(key).annotations


# ── Setup ──


@pytest.fixture
def img_dir(tmp_path: Path) -> Path:
    """Write one 640x480 image under ``images/`` and return the dataset root."""
    images = tmp_path / "images"
    images.mkdir()

    # Create a minimal test image
    from PIL import Image

    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    img.save(images / "test_001.jpg")
    return tmp_path


def _label(root: Path):
    return label_key(root, UNDATED_BUCKET, "test_001")


def _prediction(root: Path):
    return prediction_key(root, "m", "test_001")


# ── Box roundtrip ──


class TestBoxRoundtrip:
    """The box round trip: load, draw boxes, save, read back."""

    def test_draw_save_reload(self, img_dir: Path) -> None:
        """Two drawn boxes come back with their subjects and corners less than 2 px off."""
        # Draw boxes (simulating canvas click-drag)
        annotations = [
            Annotation(
                subject="bud", geometry=BBox(x1=100, y1=50, x2=250, y2=200)
            ),
            Annotation(
                subject="nut", geometry=BBox(x1=400, y1=300, x2=550, y2=420)
            ),
        ]

        read_back = _write(_label(img_dir), annotations)
        assert len(read_back) == 2
        assert {a.subject for a in read_back} == {"bud", "nut"}

        # Verify coordinates (within float precision)
        for orig, loaded in zip(annotations, read_back):
            assert abs(orig.geometry.x1 - loaded.geometry.x1) < 2
            assert abs(orig.geometry.y1 - loaded.geometry.y1) < 2
            assert abs(orig.geometry.x2 - loaded.geometry.x2) < 2
            assert abs(orig.geometry.y2 - loaded.geometry.y2) < 2
            assert orig.subject == loaded.subject

    def test_five_distinct_subjects_survive_the_round_trip(
        self, img_dir: Path
    ) -> None:
        """Five box annotations of five distinct subjects survive the label document round trip.

        They are written to and read back from one per-image label document, all five subjects
        and the count surviving.
        """
        subjects = ["bud", "shoot", "leaf", "nut", "bush"]
        annotations = [
            Annotation(
                subject=subj,
                geometry=BBox(x1=i * 100, y1=10, x2=i * 100 + 80, y2=90),
            )
            for i, subj in enumerate(subjects)
        ]

        read_back = _write(_label(img_dir), annotations)
        assert len(read_back) == 5
        assert {a.subject for a in read_back} == set(subjects)


# ── Polygon roundtrip ──


class TestPolygonRoundtrip:
    """The polygon round trip: load, draw polygons, save, read back."""

    def test_draw_polygon_save_reload(self, img_dir: Path) -> None:
        """A drawn triangle comes back as one polygon, its subject kept, vertices under 2 px off."""
        # Draw a triangle polygon (simulating canvas vertex clicks)
        poly = Polygon(rings=[[(100.0, 50.0), (250.0, 200.0), (50.0, 200.0)]])

        read_back = _write(
            _label(img_dir), [Annotation(subject="bud", geometry=poly)]
        )
        assert len(read_back) == 1
        assert read_back[0].subject == "bud"
        assert isinstance(read_back[0].geometry, Polygon)
        assert len(read_back[0].geometry.rings[0]) == 3

        # Verify coordinates
        for (ox, oy), (lx, ly) in zip(
            poly.rings[0], read_back[0].geometry.rings[0]
        ):
            assert abs(ox - lx) < 2
            assert abs(oy - ly) < 2

    def test_multi_class_polygon(self, img_dir: Path) -> None:
        """Two polygons of different subjects both come back with their subjects."""
        annotations = [
            Annotation(
                subject="bud",
                geometry=Polygon(
                    rings=[[(10, 10), (100, 10), (100, 100), (10, 100)]]
                ),
            ),
            Annotation(
                subject="leaf",
                geometry=Polygon(
                    rings=[[(200, 200), (300, 200), (300, 300), (200, 300)]]
                ),
            ),
        ]

        read_back = _write(_label(img_dir), annotations)
        assert len(read_back) == 2
        assert {a.subject for a in read_back} == {"bud", "leaf"}

    def test_occlusion_split_prediction_survives_the_round_trip(
        self, img_dir: Path
    ) -> None:
        """An occlusion-split predicted mask comes back as one instance with both its rings.

        A model-predicted mask the editor overlays can be occlusion-split: one instance, two
        contours. Both rings come back on the one instance, with its score.
        """
        rings = [
            [(100.0, 50.0), (150.0, 50.0), (150.0, 200.0), (100.0, 200.0)],
            [(300.0, 60.0), (350.0, 60.0), (350.0, 190.0), (300.0, 190.0)],
        ]
        read_back = _write(
            _prediction(img_dir),
            [
                Annotation(
                    subject="bud", geometry=Polygon(rings=rings), score=0.88
                )
            ],
        )
        assert (
            len(read_back) == 1
        )  # one instance, not one annotation per contour
        assert read_back[0].geometry.rings == rings
        assert read_back[0].score == 0.88


# ── Single-document save (boxes + polygons in one per-image document) ──


class TestSingleDocumentSave:
    """Boxes and polygons of different subjects saved into one per-image document."""

    def test_box_and_polygon_single_document(self, img_dir: Path) -> None:
        """A box and a polygon saved together both come back from one document."""
        annotations = [
            Annotation(
                subject="bud", geometry=BBox(x1=50, y1=50, x2=200, y2=150)
            ),
            Annotation(
                subject="nut",
                geometry=Polygon(
                    rings=[[(300, 100), (400, 100), (400, 200), (300, 200)]]
                ),
            ),
        ]

        read_back = _write(_label(img_dir), annotations)
        assert len(read_back) == 2
        box_ann = next(a for a in read_back if isinstance(a.geometry, BBox))
        poly_ann = next(
            a for a in read_back if isinstance(a.geometry, Polygon)
        )
        assert box_ann.subject == "bud"
        assert poly_ann.subject == "nut"


# ── Prediction overlay verification ──


class TestPredictionOverlay:
    """Proposals loaded beside annotations and paired with the one they overlap."""

    def test_prediction_overlay_pairing(self, img_dir: Path) -> None:
        """A loaded proposal pairs with the annotation it overlaps."""
        loaded_gt = _write(
            _label(img_dir),
            [
                Annotation(
                    subject="bud", geometry=BBox(x1=100, y1=50, x2=250, y2=200)
                ),
                Annotation(
                    subject="bud",
                    geometry=BBox(x1=400, y1=300, x2=550, y2=420),
                ),
            ],
        )
        # Predictions: one matching, one FP.
        loaded_preds = _write(
            _prediction(img_dir),
            [
                Annotation(
                    subject="bud",
                    geometry=BBox(x1=105, y1=55, x2=245, y2=195),
                    score=0.92,
                ),
                Annotation(
                    subject="bud",
                    geometry=BBox(x1=10, y1=10, x2=50, y2=50),
                    score=0.75,
                ),
            ],
        )

        paired = pair_proposals(
            loaded_gt, loaded_preds, {"kind": "iou", "iou_threshold": 0.5}
        )

        # The overlapping proposal confirms the first annotation; the other pairs with nothing.
        assert paired.pairs == [(0, 0)]
