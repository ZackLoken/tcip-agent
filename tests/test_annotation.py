"""Tests for annotation tools and label I/O."""

from __future__ import annotations

import json
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


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Minimal dataset in the canonical layout with per-image JSON labels/predictions.

    Overrides the conftest fixture: score_predictions reads GT and
    predictions through the json_io per-image schema (pixel COCO xywh + native ``score``).
    Canonical per-image JSON, resolved by the tools without a format hint.
    """
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import BBox, PredBBox

    date = "2-11-26"
    images_dir = tmp_path / "images" / date
    images_dir.mkdir(parents=True)
    labels_dir = tmp_path / "annotations" / "default" / date / "detect"
    labels_dir.mkdir(parents=True)
    preds_dir = tmp_path / "predictions" / "live" / date / "detect"
    preds_dir.mkdir(parents=True)

    for name in ("img_001", "img_002", "img_003"):
        Image.new("RGB", (640, 480), color=(128, 128, 128)).save(images_dir / f"{name}.jpg")
        # 2 GT boxes per image (pixel xyxy).
        json_io.write_detect(
            str(labels_dir / f"{name}.json"),
            [BBox(288, 216, 352, 264, 0), BBox(176, 132, 208, 156, 0)],
            640, 480,
        )
        # Predictions: 1 matching (TP) + 1 elsewhere (FP), confidence in the JSON score.
        json_io.write_detect(
            str(preds_dir / f"{name}.json"),
            [PredBBox(288, 216, 352, 264, 0, confidence=0.9),
             PredBBox(496, 372, 528, 396, 0, confidence=0.7)],
            640, 480,
        )
    return tmp_path


def test_bbox_creation():
    b = BBox(x1=10, y1=20, x2=50, y2=60, class_id=0)
    assert b.x1 == 10
    assert b.class_id == 0


def test_parse_detect_labels(tmp_path: Path):
    # Canonical on-disk label is per-image COCO/JSON: bbox is pixel xywh, class in category_id.
    label_path = tmp_path / "img_001.json"
    label_path.write_text(
        json.dumps({
            "image": "img_001", "width": 640, "height": 480,
            "objects": [
                {"category_id": 0, "bbox": [100, 100, 50, 50]},
                {"category_id": 0, "bbox": [200, 200, 40, 40]},
            ],
        })
    )
    boxes, class_ids = parse_detect_labels(str(label_path), 640, 480)
    assert len(boxes) == 2
    assert all(isinstance(b, BBox) for b in boxes)
    assert 0 in class_ids


def test_write_and_read_roundtrip(tmp_path: Path):
    boxes = [
        BBox(x1=100, y1=100, x2=200, y2=200, class_id=0),
        BBox(x1=300, y1=300, x2=350, y2=350, class_id=1),
    ]
    path = tmp_path / "test.json"
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


def test_score_predictions_single_image(data_dir: Path):
    pytest.importorskip("tcip_mcp.tools.annotation_tools")
    from tcip_mcp.tools.annotation_tools import score_predictions

    img = data_dir / "images" / "2-11-26" / "img_001.jpg"
    result = score_predictions(str(img), iou_threshold=0.5, conf_threshold=0.25)
    assert "tp" in result
    assert "fp" in result
    assert "fn" in result
    assert result["tp"] + result["fn"] == 2  # 2 GT boxes
