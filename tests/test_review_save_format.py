"""Phase 4.2b — review save_gt writes GT back in the image's source format."""

import json

import pytest


def _engine_and_ctx(tmp_path, source_format):
    from tcip_annotation.review_engine import ReviewContext, ReviewEngine
    from tcip_annotation.state import BBox, Polygon

    eng = ReviewEngine(state_dir=str(tmp_path / "state"), class_names={0: "cat"})
    ctx = ReviewContext(
        img_name="img0.jpg", img_width=100, img_height=100,
        gt_boxes=[BBox(10.5, 10.0, 50.0, 60.0, 0)],
        gt_polygons=[Polygon([(5, 5), (20, 5), (20, 20)], 0)],
        source_format=source_format,
    )
    return eng, ctx


def test_save_gt_voc_roundtrips(tmp_path):
    from tcip_annotation.format_io import parse_voc_detect

    eng, ctx = _engine_and_ctx(tmp_path, "voc")
    assert eng.save_gt(ctx, detect_path=str(tmp_path / "labels" / "img0.txt")) is True
    xml = tmp_path / "labels" / "img0.xml"
    assert xml.is_file()
    boxes, _, name_to_id = parse_voc_detect(str(xml))
    assert len(boxes) == 1
    assert boxes[0].x1 == pytest.approx(10.5)   # precision + format preserved
    assert "cat" in name_to_id                   # class name embedded


def test_save_gt_labelme_writes_box_and_polygon(tmp_path):
    eng, ctx = _engine_and_ctx(tmp_path, "labelme")
    assert eng.save_gt(ctx, detect_path=str(tmp_path / "labels" / "img0.txt")) is True
    js = json.loads((tmp_path / "labels" / "img0.json").read_text())
    assert sorted(s["shape_type"] for s in js["shapes"]) == ["polygon", "rectangle"]
    assert all(s["label"] == "cat" for s in js["shapes"])


def test_save_gt_yolo_default_unchanged(tmp_path):
    eng, ctx = _engine_and_ctx(tmp_path, "yolo")
    detect = tmp_path / "labels" / "img0.txt"
    assert eng.save_gt(ctx, detect_path=str(detect)) is True
    fields = detect.read_text().split()
    assert fields and len(fields) % 5 == 0       # YOLO "cls cx cy w h"


def test_save_gt_coco_falls_back_to_yolo(tmp_path):
    eng, ctx = _engine_and_ctx(tmp_path, "coco")
    detect = tmp_path / "labels" / "img0.txt"
    assert eng.save_gt(ctx, detect_path=str(detect)) is True
    assert detect.is_file()                       # GT not lost
    assert not (tmp_path / "labels" / "img0.xml").exists()
