"""Phase 4.2 — non-YOLO detection loaders (COCO/VOC/LabelMe) + parser correctness fixes."""

import json

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402


def test_parse_coco_segment_keeps_all_polygons():
    from tcip_annotation.format_io import parse_coco_segment
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 2,
                         "segmentation": [[0, 0, 10, 0, 10, 10], [20, 20, 30, 20, 30, 30]]}],
        "categories": [],
    }
    polys, cids = parse_coco_segment(coco, file_name="a.jpg")
    assert len(polys) == 2          # both polygon parts kept (was 1)
    assert cids == {2}


def test_write_voc_preserves_float_precision(tmp_path):
    from tcip_annotation.format_io import BBox, parse_voc_detect, write_voc_detect
    p = tmp_path / "a.xml"
    write_voc_detect(str(p), [BBox(10.5, 20.25, 30.75, 40.5, 0)], 100, 100, "a.jpg")
    boxes, _, _ = parse_voc_detect(str(p))
    assert boxes[0].x1 == pytest.approx(10.5)   # not truncated to whole pixels
    assert boxes[0].y2 == pytest.approx(40.5)


def _make_images(images_dir, n=1):
    images_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (100, 100)).save(images_dir / f"img{i}.jpg")


def test_build_dataset_voc(tmp_path):
    from tcip_annotation.format_io import BBox, write_voc_detect
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images_dir = tmp_path / "images"
    _make_images(images_dir)
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    write_voc_detect(str(labels_dir / "img0.xml"), [BBox(10, 10, 50, 50, 0)], 100, 100, "img0.jpg")

    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                       num_classes=1, label_format="voc")
    _, target = ds[0]
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].tolist() == [1]     # 0-indexed cid 0 -> 1-indexed (background 0)


def test_build_dataset_coco(tmp_path):
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images_dir = tmp_path / "images"
    _make_images(images_dir)
    coco = {"images": [{"id": 1, "file_name": "img0.jpg", "width": 100, "height": 100}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 40, 40]}],
            "categories": []}
    coco_path = tmp_path / "ann.json"
    coco_path.write_text(json.dumps(coco))

    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(images_dir),
                       num_classes=1, label_format="coco", coco_json=str(coco_path))
    _, target = ds[0]
    assert target["boxes"].shape == (1, 4)
    assert ds.class_distribution == {0: 1}


def test_build_dataset_labelme(tmp_path):
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images_dir = tmp_path / "images"
    _make_images(images_dir)
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    lm = {"shapes": [{"label": "cat", "shape_type": "rectangle", "points": [[10, 10], [50, 50]]}]}
    (labels_dir / "img0.json").write_text(json.dumps(lm))

    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                       num_classes=1, label_format="labelme")
    _, target = ds[0]
    assert target["boxes"].shape == (1, 4)
