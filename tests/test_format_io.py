"""Tests for multi-format annotation I/O (YOLO, COCO, VOC, LabelMe)."""

import json
import os


from tcip_annotation.state import BBox, Polygon
from tcip_annotation.format_io import (
    detect_format,
    load_annotations,
    save_annotations,
    parse_coco_detect,
    parse_coco_segment,
    write_coco_detect,
    write_coco_segment,
    parse_voc_detect,
    write_voc_detect,
    parse_labelme_detect,
    parse_labelme_segment,
    write_labelme,
)
from tcip_annotation.label_io import (
    parse_detect_labels,
)


# ── detect_format ───────────────────────────────────────────────────────────


def test_detect_format_txt(tmp_path):
    txt = tmp_path / "label.txt"
    txt.write_text("0 0.5 0.5 0.1 0.1\n")
    assert detect_format(str(txt)) == "yolo"


def test_detect_format_json(tmp_path):
    js = tmp_path / "annotations.json"
    js.write_text('{"images": [], "annotations": []}')
    assert detect_format(str(js)) == "coco"


def test_detect_format_xml(tmp_path):
    xml = tmp_path / "label.xml"
    xml.write_text("<annotation></annotation>")
    assert detect_format(str(xml)) == "voc"


def test_detect_format_labelme_json(tmp_path):
    lm = tmp_path / "label.json"
    lm.write_text('{"shapes": [], "imagePath": "img.jpg"}')
    assert detect_format(str(lm)) == "labelme"


def test_detect_format_dir_txt(tmp_path):
    (tmp_path / "label.txt").write_text("")
    assert detect_format(str(tmp_path)) == "yolo"


def test_detect_format_dir_json_coco(tmp_path):
    (tmp_path / "annotations.json").write_text('{"images": [], "annotations": []}')
    assert detect_format(str(tmp_path)) == "coco"


def test_detect_format_dir_json_labelme(tmp_path):
    (tmp_path / "img0.json").write_text('{"shapes": [], "imagePath": "img0.jpg"}')
    assert detect_format(str(tmp_path)) == "labelme"


def test_detect_format_dir_json_with_txt_coexisting(tmp_path):
    # COCO/LabelMe JSON should win over stray .txt files in the same directory
    (tmp_path / "annotations.json").write_text('{"images": [], "annotations": []}')
    (tmp_path / "classes.txt").write_text("catkin\nbush\n")
    assert detect_format(str(tmp_path)) == "coco"


def test_detect_format_dir_unknown_json_falls_back_to_yolo(tmp_path):
    # An unrecognized JSON (no annotation keys) should not force coco
    (tmp_path / "data.json").write_text("{}")
    assert detect_format(str(tmp_path)) == "yolo"


# ── COCO detect parse/write round-trip ──────────────────────────────────────


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


def test_load_annotations_yolo(tmp_path):
    """load_annotations dispatches to YOLO parser for .txt files."""
    txt = tmp_path / "label.txt"
    txt.write_text("0 0.5 0.5 0.1 0.2\n")
    boxes, cids = load_annotations(str(txt), 640, 480, task="detect")
    assert len(boxes) == 1
    assert cids == {0}


def test_load_annotations_coco(tmp_path):
    """load_annotations dispatches to COCO parser for .json files."""
    coco = _sample_coco_detect()
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(coco))
    boxes, cids = load_annotations(
        str(path), 640, 480, task="detect", file_name="IMG_0001.jpg"
    )
    assert len(boxes) == 2


def test_save_annotations_yolo(tmp_path):
    """save_annotations dispatches to YOLO writer for fmt='yolo'."""
    boxes = [BBox(100, 200, 200, 300, 0)]
    path = str(tmp_path / "label.txt")
    save_annotations(path, boxes, 640, 480, task="detect", fmt="yolo")
    assert os.path.exists(path)
    # Verify round-trip
    parsed, _ = parse_detect_labels(path, 640, 480)
    assert len(parsed) == 1


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


def test_parse_voc_detect(tmp_path):
    xml_content = """<?xml version="1.0" ?>
<annotation>
  <filename>IMG_0001.jpg</filename>
  <size><width>640</width><height>480</height><depth>3</depth></size>
  <object>
    <name>tree</name>
    <bndbox><xmin>100</xmin><ymin>200</ymin><xmax>300</xmax><ymax>400</ymax></bndbox>
  </object>
  <object>
    <name>nut</name>
    <bndbox><xmin>50</xmin><ymin>50</ymin><xmax>150</xmax><ymax>100</ymax></bndbox>
  </object>
</annotation>"""
    xml_path = tmp_path / "label.xml"
    xml_path.write_text(xml_content)
    boxes, class_ids, name_to_id = parse_voc_detect(str(xml_path))
    assert len(boxes) == 2
    assert name_to_id == {"nut": 0, "tree": 1}  # alphabetical
    assert boxes[0].x1 == 100  # first object is tree
    assert boxes[0].class_id == 1  # tree → id 1


def test_write_voc_detect_roundtrip(tmp_path):
    boxes = [BBox(10, 20, 100, 200, 0), BBox(50, 50, 150, 150, 1)]
    path = str(tmp_path / "label.xml")
    write_voc_detect(path, boxes, 640, 480, id_to_name={0: "tree", 1: "nut"})
    assert os.path.exists(path)
    parsed, cids, name_to_id = parse_voc_detect(path)
    assert len(parsed) == 2
    assert name_to_id == {"nut": 0, "tree": 1}


def test_load_annotations_voc(tmp_path):
    xml_content = """<?xml version="1.0" ?>
<annotation>
  <object>
    <name>tree</name>
    <bndbox><xmin>100</xmin><ymin>200</ymin><xmax>300</xmax><ymax>400</ymax></bndbox>
  </object>
</annotation>"""
    xml_path = tmp_path / "label.xml"
    xml_path.write_text(xml_content)
    boxes, cids = load_annotations(str(xml_path), 640, 480, task="detect")
    assert len(boxes) == 1


def test_save_annotations_voc(tmp_path):
    boxes = [BBox(10, 20, 100, 200, 0)]
    path = str(tmp_path / "label.xml")
    save_annotations(path, boxes, 640, 480, task="detect", fmt="voc")
    assert os.path.exists(path)
    parsed, _, _ = parse_voc_detect(path)
    assert len(parsed) == 1


# ── LabelMe round-trip ─────────────────────────────────────────────────────


def test_parse_labelme_detect(tmp_path):
    lm = {
        "shapes": [
            {"label": "tree", "points": [[10, 20], [100, 200]], "shape_type": "rectangle"},
            {"label": "nut", "points": [[50, 50], [150, 150]], "shape_type": "rectangle"},
        ],
        "imagePath": "IMG_0001.jpg",
        "imageWidth": 640,
        "imageHeight": 480,
    }
    path = tmp_path / "label.json"
    path.write_text(json.dumps(lm))
    boxes, class_ids, name_to_id = parse_labelme_detect(str(path))
    assert len(boxes) == 2
    assert name_to_id == {"nut": 0, "tree": 1}


def test_parse_labelme_segment(tmp_path):
    lm = {
        "shapes": [
            {"label": "tree", "points": [[10, 20], [50, 20], [50, 80], [10, 80]], "shape_type": "polygon"},
        ],
        "imagePath": "IMG_0001.jpg",
        "imageWidth": 640,
        "imageHeight": 480,
    }
    path = tmp_path / "label.json"
    path.write_text(json.dumps(lm))
    polygons, class_ids, name_to_id = parse_labelme_segment(str(path))
    assert len(polygons) == 1
    assert len(polygons[0].points) == 4


def test_write_labelme_roundtrip_boxes(tmp_path):
    boxes = [BBox(10, 20, 100, 200, 0)]
    path = str(tmp_path / "label.json")
    write_labelme(path, boxes, 640, 480, id_to_name={0: "tree"})
    assert os.path.exists(path)
    parsed, _, _ = parse_labelme_detect(path)
    assert len(parsed) == 1


def test_write_labelme_roundtrip_polygons(tmp_path):
    poly = Polygon([(10, 20), (50, 20), (50, 80), (10, 80)], class_id=0)
    path = str(tmp_path / "label.json")
    write_labelme(path, [poly], 640, 480, id_to_name={0: "tree"})
    assert os.path.exists(path)
    parsed, _, _ = parse_labelme_segment(path)
    assert len(parsed) == 1


def test_load_annotations_labelme(tmp_path):
    lm = {
        "shapes": [
            {"label": "tree", "points": [[10, 20], [100, 200]], "shape_type": "rectangle"},
        ],
        "imagePath": "img.jpg",
    }
    path = tmp_path / "label.json"
    path.write_text(json.dumps(lm))
    boxes, cids = load_annotations(str(path), 640, 480, task="detect")
    assert len(boxes) == 1


def test_save_annotations_labelme(tmp_path):
    boxes = [BBox(10, 20, 100, 200, 0)]
    path = str(tmp_path / "label.json")
    save_annotations(path, boxes, 640, 480, task="detect", fmt="labelme")
    with open(path) as f:
        data = json.load(f)
    assert "shapes" in data
    assert len(data["shapes"]) == 1
