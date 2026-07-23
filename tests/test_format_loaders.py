"""COCO detection/segmentation loaders + parser correctness."""

import json

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402


def test_parse_coco_annotations_decodes_names():
    """A single-file COCO parses to name-based Annotations, the subject decoded from categories."""
    from tcip_annotation.format_io import parse_coco_annotations
    from tcip_annotation.state import Polygon
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 2,
                         "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]]}],
        "categories": [{"id": 2, "name": "catkin"}],
    }
    anns = parse_coco_annotations(coco, file_name="a.jpg")
    assert len(anns) == 1
    assert anns[0].subject == "catkin"       # id 2 -> name, from the file's own categories
    assert isinstance(anns[0].geometry, Polygon)


def _make_images(images_dir, n=1):
    images_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (100, 100)).save(images_dir / f"img{i}.jpg")


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
