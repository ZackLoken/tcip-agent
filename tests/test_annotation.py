"""Tests for annotation tools and label I/O."""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation import (
    BBox,
    PredBBox,
    parse_detect_labels,
    write_detect_labels,
    box_iou,
    compute_matches,
)


def test_bbox_creation():
    b = BBox(x1=10, y1=20, x2=50, y2=60, class_id=0)
    assert b.x1 == 10
    assert b.class_id == 0


def test_parse_detect_labels(data_dir: Path):
    label_path = data_dir / "labels" / "detect" / "img_001.txt"
    boxes, class_ids = parse_detect_labels(str(label_path), 640, 480)
    assert len(boxes) == 2
    assert all(isinstance(b, BBox) for b in boxes)
    assert 0 in class_ids


def test_write_and_read_roundtrip(tmp_path: Path):
    boxes = [
        BBox(x1=100, y1=100, x2=200, y2=200, class_id=0),
        BBox(x1=300, y1=300, x2=350, y2=350, class_id=1),
    ]
    path = tmp_path / "test.txt"
    write_detect_labels(str(path), boxes, 640, 480)
    read_back, class_ids = parse_detect_labels(str(path), 640, 480)
    assert len(read_back) == 2
    assert 0 in class_ids and 1 in class_ids
    # Check approximate roundtrip (floating point tolerance)
    for orig, read in zip(boxes, read_back):
        assert abs(orig.x1 - read.x1) < 2
        assert abs(orig.y1 - read.y1) < 2


def test_box_iou():
    b1 = BBox(x1=0, y1=0, x2=10, y2=10, class_id=0)
    b2 = BBox(x1=0, y1=0, x2=10, y2=10, class_id=0)
    assert box_iou(b1, b2) == 1.0

    b3 = BBox(x1=5, y1=5, x2=15, y2=15, class_id=0)
    iou = box_iou(b1, b3)
    assert 0.1 < iou < 0.2  # 25/175 ≈ 0.143


def test_box_iou_no_overlap():
    b1 = BBox(x1=0, y1=0, x2=10, y2=10, class_id=0)
    b2 = BBox(x1=20, y1=20, x2=30, y2=30, class_id=0)
    assert box_iou(b1, b2) == 0.0


def test_compute_matches():
    gt_boxes = [BBox(x1=0, y1=0, x2=10, y2=10, class_id=0)]
    pred_boxes = [
        PredBBox(x1=0, y1=0, x2=10, y2=10, class_id=0, confidence=0.9),
        PredBBox(x1=50, y1=50, x2=60, y2=60, class_id=0, confidence=0.8),
    ]
    matches = compute_matches(gt_boxes, [], pred_boxes, [], iou_threshold=0.5, conf_threshold=0.25)
    assert len(matches["tp"]) == 1
    assert len(matches["fp"]) == 1
    assert len(matches["fn"]) == 0


def test_evaluate_detections_tool(data_dir: Path):
    pytest.importorskip("tcip_mcp.tools.annotation_tools")
    from tcip_mcp.tools.annotation_tools import evaluate_detections

    img = data_dir / "images" / "img_001.jpg"
    result = evaluate_detections(str(img), iou_threshold=0.5, conf_threshold=0.25)
    assert "tp" in result
    assert "fp" in result
    assert "fn" in result
    assert result["tp"] + result["fn"] == 2  # 2 GT boxes
