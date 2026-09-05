"""Tests for annotation tools and label I/O."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcip_annotation import (
    Annotation,
    BBox,
    read_annotations,
    write_annotations,
    box_iou,
    compute_matches,
)


def test_bbox_creation():
    b = BBox(x1=10, y1=20, x2=50, y2=60)
    assert (b.x1, b.y1, b.x2, b.y2) == (10, 20, 50, 60)


def test_read_annotations(tmp_path: Path):
    # Canonical on-disk label is the name-based per-image JSON: bbox is pixel xywh, subject by name.
    label_path = tmp_path / "img_001.json"
    label_path.write_text(
        json.dumps({
            "image": "img_001", "width": 640, "height": 480,
            "annotations": [
                {"subject": "bud", "bbox": [100, 100, 50, 50]},
                {"subject": "bud", "bbox": [200, 200, 40, 40]},
            ],
        })
    )
    anns = read_annotations(str(label_path))
    assert len(anns) == 2
    assert all(isinstance(a.geometry, BBox) for a in anns)
    assert {a.subject for a in anns} == {"bud"}


def test_write_and_read_roundtrip(tmp_path: Path):
    anns = [
        Annotation(subject="bud", geometry=BBox(x1=100, y1=100, x2=200, y2=200)),
        Annotation(subject="leaf", geometry=BBox(x1=300, y1=300, x2=350, y2=350)),
    ]
    path = tmp_path / "test.json"
    write_annotations(str(path), anns, 640, 480)
    read_back = read_annotations(str(path))
    assert len(read_back) == 2
    assert {a.subject for a in read_back} == {"bud", "leaf"}
    # Check approximate roundtrip (floating point tolerance)
    for orig, read in zip(anns, read_back):
        assert abs(orig.geometry.x1 - read.geometry.x1) < 2
        assert abs(orig.geometry.y1 - read.geometry.y1) < 2


def test_box_iou():
    b1 = BBox(x1=0, y1=0, x2=10, y2=10)
    b2 = BBox(x1=0, y1=0, x2=10, y2=10)
    assert box_iou(b1, b2) == 1.0

    b3 = BBox(x1=5, y1=5, x2=15, y2=15)
    iou = box_iou(b1, b3)
    assert 0.1 < iou < 0.2  # 25/175 ≈ 0.143


def test_box_iou_no_overlap():
    b1 = BBox(x1=0, y1=0, x2=10, y2=10)
    b2 = BBox(x1=20, y1=20, x2=30, y2=30)
    assert box_iou(b1, b2) == 0.0


def test_compute_matches():
    gt = [Annotation(subject="bud", geometry=BBox(x1=0, y1=0, x2=10, y2=10))]
    preds = [
        Annotation(subject="bud", geometry=BBox(x1=0, y1=0, x2=10, y2=10), score=0.9),
        Annotation(subject="bud", geometry=BBox(x1=50, y1=50, x2=60, y2=60), score=0.8),
    ]
    matches = compute_matches(gt, preds, iou_threshold=0.5, conf_threshold=0.25)
    assert len(matches["tp"]) == 1
    assert len(matches["fp"]) == 1
    assert len(matches["fn"]) == 0


def test_score_predictions_single_image(data_dir: Path):
    pytest.importorskip("tcip_mcp.tools.annotation_tools")
    from tcip_mcp.tools.annotation_tools import score_predictions

    img = data_dir / "images" / "2-11-26" / "img_001.jpg"
    result = score_predictions(str(img), iou_threshold=0.5, conf_threshold=0.25)
    assert "tp" in result
    assert "fp" in result
    assert "fn" in result
    assert result["tp"] + result["fn"] == 2  # 2 GT boxes
