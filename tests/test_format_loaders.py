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
        "categories": [{"id": 2, "name": "bud"}],
    }
    anns = parse_coco_annotations(coco, file_name="a.jpg")
    assert len(anns) == 1
    assert anns[0].subject == "bud"       # id 2 -> name, from the file's own categories
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


def _bom_coco_path(tmp_path):
    coco = {"images": [{"id": 1, "file_name": "img0.jpg", "width": 100, "height": 100}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 40, 40]}],
            "categories": []}
    coco_path = tmp_path / "ann.json"
    coco_path.write_bytes(b"\xef\xbb\xbf" + json.dumps(coco).encode("utf-8"))
    return coco_path


@pytest.mark.parametrize("task", ["detection", "instance_seg"])
def test_build_dataset_coco_admits_a_byte_order_marked_document(tmp_path, task):
    """A UTF-8 byte-order mark encodes the same document as one without it: both training
    loaders admit it through the reader's one decode, the same as ``load_annotations`` does."""
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images_dir = tmp_path / "images"
    _make_images(images_dir)
    coco_path = _bom_coco_path(tmp_path)

    ds = build_dataset(task, images_dir=str(images_dir), labels_dir=str(images_dir),
                       num_classes=1, label_format="coco", coco_json=str(coco_path))
    assert list(ds.stems)


@pytest.mark.parametrize("task", ["detection", "instance_seg"])
def test_build_dataset_coco_admits_a_document_naming_its_own_schema_version(tmp_path, task):
    """COCO is interop (frozen=False by its own row): a legitimate external COCO document
    naming a schema_version this platform's own annotation_records store does not know must
    still build a dataset, never refuse against a store it never claimed to be."""
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images_dir = tmp_path / "images"
    _make_images(images_dir)
    coco = {"images": [{"id": 1, "file_name": "img0.jpg", "width": 100, "height": 100}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 40, 40]}],
            "categories": [], "schema_version": 999}
    coco_path = tmp_path / "ann.json"
    coco_path.write_text(json.dumps(coco))

    ds = build_dataset(task, images_dir=str(images_dir), labels_dir=str(images_dir),
                       num_classes=1, label_format="coco", coco_json=str(coco_path))
    assert list(ds.stems)


@pytest.mark.parametrize("task", ["detection", "instance_seg"])
def test_build_dataset_coco_refuses_an_undecodable_document(tmp_path, task):
    """A present COCO document that will not decode is a named refusal, not a raw parse error
    surfacing from whichever loader happens to touch it first."""
    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images_dir = tmp_path / "images"
    _make_images(images_dir)
    coco_path = tmp_path / "ann.json"
    coco_path.write_bytes(b"{not json")

    with pytest.raises(UnreadableLabelDocument):
        build_dataset(task, images_dir=str(images_dir), labels_dir=str(images_dir),
                      num_classes=1, label_format="coco", coco_json=str(coco_path))
