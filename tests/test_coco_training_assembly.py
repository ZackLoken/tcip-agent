"""COCO-training-assembly: training/eval reads the canonical per-image JSON label store by
assembling a dataset-level COCO on the fly (``build_dataset`` auto-routes a JSON label dir onto
the ``label_format='coco'`` path). Covers detection + instance-seg, format detection, and the
stratification line-count that must see JSON objects (not only YOLO ``.txt``)."""

import json

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import BBox, Polygon  # noqa: E402


def _make_images(images_dir, stems):
    images_dir.mkdir(parents=True, exist_ok=True)
    for s in stems:
        Image.new("RGB", (100, 100)).save(images_dir / f"{s}.jpg")


# ── format detection ────────────────────────────────────────────────────────

def test_dir_label_format_detects_json(tmp_path):
    from tcip_mcp.pipelines.data.datasets import dir_label_format
    d = tmp_path / "detect"
    d.mkdir()
    json_io.write_detect(d / "a.json", [BBox(10, 10, 50, 50, 0)], 100, 100)
    assert dir_label_format(d) == "json"


def test_dir_label_format_detects_yolo(tmp_path):
    from tcip_mcp.pipelines.data.datasets import dir_label_format
    d = tmp_path / "detect"
    d.mkdir()
    (d / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    assert dir_label_format(d) == "yolo"


def test_dir_label_format_ignores_non_canonical_json(tmp_path):
    """A LabelMe-style .json (``shapes``, no ``objects``) is not the canonical store — don't claim
    it, so it never gets silently assembled as though it were per-image JSON."""
    from tcip_mcp.pipelines.data.datasets import dir_label_format
    d = tmp_path / "labels"
    d.mkdir()
    (d / "a.json").write_text(json.dumps({"shapes": [{"label": "x", "points": [[1, 1], [2, 2]]}]}))
    assert dir_label_format(d) is None


def test_dir_label_format_empty(tmp_path):
    from tcip_mcp.pipelines.data.datasets import dir_label_format
    assert dir_label_format(tmp_path / "nope") is None


# ── assemble_coco ───────────────────────────────────────────────────────────

def test_assemble_coco_pairs_labels_with_images(tmp_path):
    from tcip_mcp.pipelines.data.datasets import assemble_coco
    images = tmp_path / "images"
    labels = tmp_path / "detect"
    labels.mkdir()
    _make_images(images, ["img0", "img1"])
    json_io.write_detect(labels / "img0.json", [BBox(10, 10, 50, 50, 0), BBox(0, 0, 20, 20, 1)], 100, 100)
    json_io.write_detect(labels / "img1.json", [], 100, 100, keep_empty=True)  # confirmed negative

    coco = assemble_coco(labels, images)
    assert {im["file_name"] for im in coco["images"]} == {"img0.jpg", "img1.jpg"}  # negative kept
    assert len(coco["annotations"]) == 2
    assert {a["category_id"] for a in coco["annotations"]} == {0, 1}


def test_assemble_coco_skips_stem_without_image(tmp_path):
    from tcip_mcp.pipelines.data.datasets import assemble_coco
    images = tmp_path / "images"
    labels = tmp_path / "detect"
    labels.mkdir()
    _make_images(images, ["img0"])
    json_io.write_detect(labels / "img0.json", [BBox(10, 10, 50, 50, 0)], 100, 100)
    json_io.write_detect(labels / "orphan.json", [BBox(5, 5, 9, 9, 0)], 100, 100)  # no image

    coco = assemble_coco(labels, images)
    assert [im["file_name"] for im in coco["images"]] == ["img0.jpg"]


# ── build_dataset auto-routes per-image JSON onto the COCO path ──────────────

def test_build_dataset_detection_autoresolves_json(tmp_path):
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images = tmp_path / "images"
    labels = tmp_path / "detect"
    labels.mkdir()
    _make_images(images, ["img0"])
    json_io.write_detect(labels / "img0.json", [BBox(10, 10, 50, 50, 0)], 100, 100)

    ds = build_dataset("detection", images_dir=str(images), labels_dir=str(labels), num_classes=1)
    assert ds.label_format == "coco"  # auto-routed
    _, target = ds[0]
    assert target["boxes"].shape == (1, 4)
    assert target["boxes"].tolist()[0] == pytest.approx([10, 10, 50, 50])
    assert target["labels"].tolist() == [1]  # 0-indexed cid 0 -> 1-indexed
    assert ds.class_distribution == {0: 1}


def test_build_dataset_detection_yolo_unaffected(tmp_path):
    """A legacy YOLO ``.txt`` dir still trains through the YOLO path (no assembly)."""
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images = tmp_path / "images"
    labels = tmp_path / "detect"
    labels.mkdir()
    _make_images(images, ["img0"])
    (labels / "img0.txt").write_text("0 0.3 0.3 0.4 0.4\n")

    ds = build_dataset("detection", images_dir=str(images), labels_dir=str(labels), num_classes=1)
    assert ds.label_format == "yolo"
    _, target = ds[0]
    assert target["boxes"].shape == (1, 4)


def test_build_dataset_respects_explicit_format(tmp_path):
    """An explicit label_format is never overridden by auto-resolve."""
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images = tmp_path / "images"
    labels = tmp_path / "detect"
    labels.mkdir()
    _make_images(images, ["img0"])
    json_io.write_detect(labels / "img0.json", [BBox(10, 10, 50, 50, 0)], 100, 100)

    ds = build_dataset("detection", images_dir=str(images), labels_dir=str(labels),
                       num_classes=1, label_format="yolo")
    assert ds.label_format == "yolo"  # honored, even though .json is present


def test_build_dataset_instance_seg_autoresolves_json(tmp_path):
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images = tmp_path / "images"
    labels = tmp_path / "segment"
    labels.mkdir()
    _make_images(images, ["img0"])
    json_io.write_segment(
        labels / "img0.json",
        [Polygon([(10, 10), (50, 10), (50, 50), (10, 50)], 0)], 100, 100)

    ds = build_dataset("instance_seg", images_dir=str(images), labels_dir=str(labels), num_classes=1)
    assert ds.label_format == "coco"
    _, target = ds[0]
    assert target["boxes"].shape == (1, 4)
    assert target["masks"].shape[0] == 1
    assert target["labels"].tolist() == [1]
    assert int(target["masks"].sum()) > 0  # polygon rasterized to a non-empty mask


def test_build_dataset_instance_seg_yolo_polygon_unaffected(tmp_path):
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images = tmp_path / "images"
    labels = tmp_path / "segment"
    labels.mkdir()
    _make_images(images, ["img0"])
    (labels / "img0.txt").write_text("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")

    ds = build_dataset("instance_seg", images_dir=str(images), labels_dir=str(labels), num_classes=1)
    assert ds.label_format == "yolo"
    _, target = ds[0]
    assert target["masks"].shape[0] == 1


# ── stratification line count ───────────────────────────────────────────────

def test_count_label_lines_reads_json_objects(tmp_path):
    from tcip_mcp.pipelines.data.splits import count_label_lines
    labels = tmp_path / "detect"
    labels.mkdir()
    json_io.write_detect(labels / "a.json", [BBox(0, 0, 10, 10, 0), BBox(0, 0, 20, 20, 0)], 100, 100)
    json_io.write_detect(labels / "neg.json", [], 100, 100, keep_empty=True)
    assert count_label_lines(labels, "a") == 2
    assert count_label_lines(labels, "neg") == 0
    assert count_label_lines(labels, "missing") == 0


def test_count_label_lines_txt_fallback(tmp_path):
    from tcip_mcp.pipelines.data.splits import count_label_lines
    labels = tmp_path / "detect"
    labels.mkdir()
    (labels / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.1 0.1\n")
    assert count_label_lines(labels, "a") == 2
