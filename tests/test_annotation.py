"""Tests for annotation tools and label I/O."""

from __future__ import annotations

import json
from pathlib import Path

from tcip_annotation import (
    Annotation,
    BBox,
    read_annotations,
    write_annotations,
)
from tcip_annotation.matching import iou_matrix, pair_proposals


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


def test_iou_matrix():
    iou = iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]])
    assert iou[0, 0] == 1.0
    assert 0.1 < iou[0, 1] < 0.2  # 25/175 ≈ 0.143
    assert iou[0, 2] == 0.0


def test_pair_proposals():
    gt = [Annotation(subject="bud", geometry=BBox(x1=0, y1=0, x2=10, y2=10))]
    preds = [
        Annotation(subject="bud", geometry=BBox(x1=0, y1=0, x2=10, y2=10), score=0.9),
        Annotation(subject="bud", geometry=BBox(x1=50, y1=50, x2=60, y2=60), score=0.8),
    ]
    assert pair_proposals(gt, preds, {"kind": "iou", "iou_threshold": 0.5}).pairs == [(0, 0)]


def test_score_predictions_single_image(data_dir: Path):
    from tcip_mcp.tools.annotation_tools import score_predictions

    img = data_dir / "images" / "2-11-26" / "img_001.jpg"
    result = score_predictions(str(img), str(data_dir / "predictions" / "live" / "2-11-26"),
                               iou_threshold=0.5, conf_threshold=0.25)
    assert "tp" in result
    assert "fp" in result
    assert "fn" in result
    assert result["tp"] + result["fn"] == 2  # 2 GT boxes
