"""Tests for annotation I/O — the canonical per-image JSON and the assembled COCO."""

import json


from tcip_annotation.state import BBox, Polygon
from tcip_annotation.format_io import (
    detect_format,
    load_annotations,
    save_annotations,
    parse_coco_detect,
    parse_coco_segment,
    write_coco_detect,
    write_coco_segment,
)


# ── detect_format ───────────────────────────────────────────────────────────


def test_detect_format_json(tmp_path):
    js = tmp_path / "annotations.json"
    js.write_text('{"images": [], "annotations": []}')
    assert detect_format(str(js)) == "coco"


def test_detect_format_dir_json_coco(tmp_path):
    (tmp_path / "annotations.json").write_text('{"images": [], "annotations": []}')
    assert detect_format(str(tmp_path)) == "coco"


def _sample_coco_detect():
    return {
        "images": [
            {"id": 1, "file_name": "IMG_0001.jpg", "width": 640, "height": 480}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [100, 200, 50, 60], "area": 3000, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [300, 100, 80, 40], "area": 3200, "iscrowd": 0},
        ],
        "categories": [{"id": 0, "name": "tree"}, {"id": 1, "name": "nut"}],
    }


def test_parse_coco_detect():
    coco = _sample_coco_detect()
    boxes, class_ids = parse_coco_detect(coco, file_name="IMG_0001.jpg")
    assert len(boxes) == 2
    assert class_ids == {0, 1}
    # COCO bbox [x, y, w, h] → BBox(x1, y1, x2, y2)
    assert boxes[0].x1 == 100
    assert boxes[0].y1 == 200
    assert boxes[0].x2 == 150
    assert boxes[0].y2 == 260


def test_parse_coco_detect_missing_image():
    coco = _sample_coco_detect()
    boxes, class_ids = parse_coco_detect(coco, file_name="MISSING.jpg")
    assert len(boxes) == 0


def test_write_coco_detect_roundtrip(tmp_path):
    boxes = [BBox(10, 20, 50, 80, 0), BBox(100, 100, 200, 150, 1)]
    path = str(tmp_path / "annotations.json")
    write_coco_detect(path, {"IMG_0001.jpg": (boxes, 640, 480)})

    with open(path) as f:
        coco = json.load(f)

    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 2
    assert coco["images"][0]["file_name"] == "IMG_0001.jpg"

    # Parse back
    parsed, cids = parse_coco_detect(coco, file_name="IMG_0001.jpg")
    assert len(parsed) == 2
    assert parsed[0].x1 == 10
    assert parsed[0].x2 == 50


# ── COCO segment parse/write round-trip ─────────────────────────────────────


def _sample_coco_segment():
    return {
        "images": [
            {"id": 1, "file_name": "IMG_0001.jpg", "width": 640, "height": 480}
        ],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 0,
                "segmentation": [[10.0, 20.0, 50.0, 20.0, 50.0, 80.0, 10.0, 80.0]],
                "bbox": [10, 20, 40, 60], "area": 2400, "iscrowd": 0,
            },
        ],
        "categories": [],
    }


def test_parse_coco_segment():
    coco = _sample_coco_segment()
    polygons, class_ids = parse_coco_segment(coco, file_name="IMG_0001.jpg")
    assert len(polygons) == 1
    assert polygons[0].class_id == 0
    assert len(polygons[0].points) == 4
    assert polygons[0].points[0] == (10.0, 20.0)


def test_write_coco_segment_roundtrip(tmp_path):
    poly = Polygon([(10, 20), (50, 20), (50, 80), (10, 80)], class_id=0)
    path = str(tmp_path / "seg.json")
    write_coco_segment(path, {"IMG_0001.jpg": ([poly], 640, 480)})

    with open(path) as f:
        coco = json.load(f)

    parsed, cids = parse_coco_segment(coco, file_name="IMG_0001.jpg")
    assert len(parsed) == 1
    assert len(parsed[0].points) == 4


# ── Unified load/save dispatch ──────────────────────────────────────────────


def test_load_annotations_coco(tmp_path):
    """load_annotations dispatches to COCO parser for .json files."""
    coco = _sample_coco_detect()
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(coco))
    boxes, cids = load_annotations(
        str(path), 640, 480, task="detect", file_name="IMG_0001.jpg"
    )
    assert len(boxes) == 2


def test_save_annotations_coco(tmp_path):
    """save_annotations dispatches to COCO writer for fmt='coco'."""
    boxes = [BBox(100, 200, 200, 300, 0)]
    path = str(tmp_path / "annotations.json")
    save_annotations(
        path, boxes, 640, 480, task="detect", fmt="coco", file_name="IMG_0001.jpg"
    )
    with open(path) as f:
        coco = json.load(f)
    assert len(coco["annotations"]) == 1


# ── PASCAL VOC round-trip ───────────────────────────────────────────────────


def test_detect_format_refuses_an_unrecognized_store(tmp_path):
    """A misdetected format reads real annotations as empty negatives, so a wrong answer here is
    worse than no answer. There is no fallback guess left to make."""
    import pytest

    odd = tmp_path / "labels.json"
    odd.write_text(json.dumps({"regions": [{"x": 1}]}))  # an in-house schema we do not know
    with pytest.raises(ValueError, match="Cannot determine the annotation format"):
        detect_format(str(odd))
    with pytest.raises(ValueError):
        detect_format(str(tmp_path / "nothing_here"))
