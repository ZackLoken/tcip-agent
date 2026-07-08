"""Tests for label_io.py — round-trip for all parse/write functions."""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation import (
    BBox,
    Polygon,
    PredBBox,
    PredPolygon,
    parse_detect_labels,
    parse_segment_labels,
    parse_detect_predictions,
    parse_segment_predictions,
    write_detect_labels,
    write_segment_labels,
)


IMG_W, IMG_H = 640, 480


@pytest.fixture
def label_dir(tmp_path: Path) -> Path:
    d = tmp_path / "labels"
    d.mkdir()
    return d


# ── Detect labels ────────────────────────────────────────────────────────


def test_parse_detect_labels(label_dir: Path):
    p = label_dir / "test.txt"
    p.write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.7 0.1 0.1\n")
    boxes, class_ids = parse_detect_labels(str(p), IMG_W, IMG_H)
    assert len(boxes) == 2
    assert class_ids == {0, 1}
    # Check pixel coordinates for first box (cx=320, cy=240, w=128, h=96)
    b = boxes[0]
    assert abs(b.x1 - (320 - 64)) < 0.01
    assert abs(b.y1 - (240 - 48)) < 0.01
    assert abs(b.x2 - (320 + 64)) < 0.01
    assert abs(b.y2 - (240 + 48)) < 0.01


def test_parse_detect_labels_missing_file(label_dir: Path):
    boxes, class_ids = parse_detect_labels(str(label_dir / "missing.txt"), IMG_W, IMG_H)
    assert boxes == []
    assert class_ids == set()


def test_write_and_read_detect_roundtrip(label_dir: Path):
    boxes = [BBox(100, 50, 200, 150, 0), BBox(300, 200, 400, 300, 1)]
    p = str(label_dir / "detect_rt.txt")
    write_detect_labels(p, boxes, IMG_W, IMG_H)
    read_back, class_ids = parse_detect_labels(p, IMG_W, IMG_H)
    assert len(read_back) == 2
    assert class_ids == {0, 1}
    for orig, read in zip(boxes, read_back):
        assert abs(orig.x1 - read.x1) < 1.0
        assert abs(orig.y1 - read.y1) < 1.0
        assert abs(orig.x2 - read.x2) < 1.0
        assert abs(orig.y2 - read.y2) < 1.0


def test_write_empty_deletes_by_default(label_dir: Path):
    # Default behaviour (used by pipeline writers) removes an emptied label file.
    p = Path(label_dir / "empty_default.txt")
    write_detect_labels(str(p), [BBox(100, 50, 200, 150, 0)], IMG_W, IMG_H)
    assert p.exists()
    write_detect_labels(str(p), [], IMG_W, IMG_H)
    assert not p.exists()


def test_write_empty_keep_empty_preserves_negative(label_dir: Path):
    # keep_empty=True (used by the interactive annotator) writes a 0-byte file so a
    # confirmed negative is preserved on disk instead of deleted — empty label files
    # are valid negatives (CLAUDE.md invariant), not noise to prune.
    p = Path(label_dir / "neg.txt")
    write_detect_labels(str(p), [], IMG_W, IMG_H, keep_empty=True)
    assert p.exists()
    assert p.read_text() == ""
    read_back, class_ids = parse_detect_labels(str(p), IMG_W, IMG_H)
    assert read_back == []
    assert class_ids == set()

    ps = Path(label_dir / "neg_seg.txt")
    write_segment_labels(str(ps), [], IMG_W, IMG_H, keep_empty=True)
    assert ps.exists()
    assert ps.read_text() == ""


# ── Segment labels ───────────────────────────────────────────────────────


def test_parse_segment_labels(label_dir: Path):
    p = label_dir / "seg.txt"
    # class_id x1 y1 x2 y2 x3 y3 (normalised)
    p.write_text("0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n")
    polys, class_ids = parse_segment_labels(str(p), IMG_W, IMG_H)
    assert len(polys) == 1
    assert class_ids == {0}
    assert len(polys[0].points) == 4


def test_write_and_read_segment_roundtrip(label_dir: Path):
    polys = [Polygon([(64, 48), (192, 48), (192, 144), (64, 144)], 0)]
    p = str(label_dir / "seg_rt.txt")
    write_segment_labels(p, polys, IMG_W, IMG_H)
    read_back, class_ids = parse_segment_labels(p, IMG_W, IMG_H)
    assert len(read_back) == 1
    assert class_ids == {0}
    for orig_pt, read_pt in zip(polys[0].points, read_back[0].points):
        assert abs(orig_pt[0] - read_pt[0]) < 1.0
        assert abs(orig_pt[1] - read_pt[1]) < 1.0


# ── Detect predictions ───────────────────────────────────────────────────


def test_parse_detect_predictions(label_dir: Path):
    p = label_dir / "pred_det.txt"
    p.write_text("0 0.95 0.5 0.5 0.2 0.2\n1 0.80 0.3 0.7 0.1 0.1\n")
    preds, class_ids = parse_detect_predictions(str(p), IMG_W, IMG_H)
    assert len(preds) == 2
    assert class_ids == {0, 1}
    assert isinstance(preds[0], PredBBox)
    assert abs(preds[0].confidence - 0.95) < 0.001
    assert abs(preds[1].confidence - 0.80) < 0.001


def test_parse_detect_predictions_missing_file(label_dir: Path):
    preds, class_ids = parse_detect_predictions(str(label_dir / "missing.txt"), IMG_W, IMG_H)
    assert preds == []
    assert class_ids == set()


# ── Segment predictions ──────────────────────────────────────────────────


def test_parse_segment_predictions(label_dir: Path):
    p = label_dir / "pred_seg.txt"
    p.write_text("0 0.92 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n")
    preds, class_ids = parse_segment_predictions(str(p), IMG_W, IMG_H)
    assert len(preds) == 1
    assert class_ids == {0}
    assert isinstance(preds[0], PredPolygon)
    assert abs(preds[0].confidence - 0.92) < 0.001
    assert len(preds[0].points) == 4


def test_parse_segment_predictions_missing_file(label_dir: Path):
    preds, class_ids = parse_segment_predictions(str(label_dir / "missing.txt"), IMG_W, IMG_H)
    assert preds == []
    assert class_ids == set()
