"""Tests for annotation I/O: the canonical per-image JSON and the assembled COCO."""

import json

import pytest

from tcip_annotation.state import Annotation, BBox, Polygon
from tcip_annotation.format_io import (
    detect_format,
    load_annotations,
    save_annotations,
    parse_coco_annotations,
    write_coco,
)


# ── detect_format ───────────────────────────────────────────────────────────


def test_detect_format_json(tmp_path):
    js = tmp_path / "annotations.json"
    js.write_text('{"images": [], "annotations": []}')
    assert detect_format(str(js)) == "coco"


def test_detect_format_dir_json_coco(tmp_path):
    (tmp_path / "annotations.json").write_text('{"images": [], "annotations": []}')
    assert detect_format(str(tmp_path)) == "coco"


def test_detect_format_per_image_json(tmp_path):
    js = tmp_path / "IMG_0001.json"
    js.write_text('{"image": "IMG_0001", "annotations": [{"subject": "catkin", "bbox": [1, 1, 9, 9]}]}')
    assert detect_format(str(js)) == "json"


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
    anns = parse_coco_annotations(coco, file_name="IMG_0001.jpg")
    assert len(anns) == 2
    assert {a.subject for a in anns} == {"tree", "nut"}
    # COCO bbox [x, y, w, h] → BBox(x1, y1, x2, y2), subject decoded from the file's categories.
    assert anns[0].geometry.x1 == 100
    assert anns[0].geometry.y1 == 200
    assert anns[0].geometry.x2 == 150
    assert anns[0].geometry.y2 == 260


def test_parse_coco_detect_missing_image():
    coco = _sample_coco_detect()
    anns = parse_coco_annotations(coco, file_name="MISSING.jpg")
    assert len(anns) == 0


def test_parse_coco_matches_a_manifest_named_logical_image_by_stem():
    """A ``.bandgroup`` manifest's own on-disk name never appears verbatim in an externally
    authored COCO document; its stem still ties the record recorded under one of its bands."""
    coco = _sample_coco_detect()
    anns = parse_coco_annotations(coco, file_name="IMG_0001.bandgroup")
    assert len(anns) == 2


def test_parse_coco_prefers_an_exact_file_name_match_over_the_stem_tie():
    coco = _sample_coco_detect()
    coco["images"].append(
        {"id": 2, "file_name": "IMG_0001.bandgroup", "width": 10, "height": 10})
    coco["annotations"].append(
        {"id": 3, "image_id": 2, "category_id": 0, "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0})
    anns = parse_coco_annotations(coco, file_name="IMG_0001.bandgroup")
    assert len(anns) == 1


def test_write_coco_roundtrip(tmp_path):
    anns = [Annotation(subject="tree", geometry=BBox(10, 20, 50, 80)),
            Annotation(subject="nut", geometry=BBox(100, 100, 200, 150))]
    path = str(tmp_path / "annotations.json")
    write_coco(path, {"IMG_0001.jpg": (anns, 640, 480)})

    with open(path) as f:
        coco = json.load(f)

    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 2
    assert coco["images"][0]["file_name"] == "IMG_0001.jpg"

    # Parse back: the categories written by write_coco decode the ids to names.
    parsed = parse_coco_annotations(coco, file_name="IMG_0001.jpg")
    assert len(parsed) == 2
    assert {a.subject for a in parsed} == {"tree", "nut"}
    assert parsed[0].geometry.x1 == 10
    assert parsed[0].geometry.x2 == 50


# ── COCO polygon parse/write round-trip ──────────────────────────────────────


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
        "categories": [{"id": 0, "name": "leaf"}],
    }


def test_parse_coco_segment():
    coco = _sample_coco_segment()
    anns = parse_coco_annotations(coco, file_name="IMG_0001.jpg")
    assert len(anns) == 1
    assert anns[0].subject == "leaf"
    assert isinstance(anns[0].geometry, Polygon)
    assert anns[0].geometry.rings == [[(10.0, 20.0), (50.0, 20.0), (50.0, 80.0), (10.0, 80.0)]]


def test_write_coco_polygon_roundtrip(tmp_path):
    poly = Polygon([[(10, 20), (50, 20), (50, 80), (10, 80)]])
    anns = [Annotation(subject="leaf", geometry=poly)]
    path = str(tmp_path / "seg.json")
    write_coco(path, {"IMG_0001.jpg": (anns, 640, 480)})

    with open(path) as f:
        coco = json.load(f)

    parsed = parse_coco_annotations(coco, file_name="IMG_0001.jpg")
    assert len(parsed) == 1
    assert isinstance(parsed[0].geometry, Polygon)
    assert len(parsed[0].geometry.rings[0]) == 4
    assert parsed[0].subject == "leaf"


# ── COCO multi-ring (occlusion-split instance) ───────────────────────────────

# Two disjoint lobes of one instance: a leaf crossed by a stem, a catkin behind a branch.
LOBE_A = [(10.0, 10.0), (30.0, 10.0), (30.0, 50.0), (10.0, 50.0)]
LOBE_B = [(70.0, 12.0), (90.0, 12.0), (90.0, 48.0), (70.0, 48.0)]


def test_parse_coco_multi_ring_segmentation_keeps_every_ring():
    # COCO's segmentation is a list of rings and always was; the reader must decode all of them into
    # one polygon rather than taking the first and dropping the rest.
    coco = _sample_coco_segment()
    coco["annotations"][0]["segmentation"] = [
        [10.0, 10.0, 30.0, 10.0, 30.0, 50.0, 10.0, 50.0],
        [70.0, 12.0, 90.0, 12.0, 90.0, 48.0, 70.0, 48.0],
    ]
    (ann,) = parse_coco_annotations(coco, file_name="IMG_0001.jpg")
    assert ann.geometry.rings == [LOBE_A, LOBE_B]


def test_write_coco_multi_ring_polygon_roundtrip(tmp_path):
    anns = [Annotation(subject="leaf", geometry=Polygon([LOBE_A, LOBE_B]))]
    path = str(tmp_path / "seg_multi.json")
    write_coco(path, {"IMG_0001.jpg": (anns, 640, 480)})

    with open(path) as f:
        coco = json.load(f)

    # One COCO annotation, two rings, in order, and its box/area span their union.
    (rec,) = coco["annotations"]
    assert rec["segmentation"] == [
        [10.0, 10.0, 30.0, 10.0, 30.0, 50.0, 10.0, 50.0],
        [70.0, 12.0, 90.0, 12.0, 90.0, 48.0, 70.0, 48.0],
    ]
    assert rec["bbox"] == [10.0, 10.0, 80.0, 40.0]
    assert rec["area"] == 80.0 * 40.0

    (parsed,) = parse_coco_annotations(coco, file_name="IMG_0001.jpg")
    assert parsed.geometry.rings == [LOBE_A, LOBE_B]


# ── Unified load/save dispatch ──────────────────────────────────────────────


def test_load_annotations_coco(tmp_path):
    """load_annotations detects and dispatches to the COCO parser for a dataset-level .json."""
    coco = _sample_coco_detect()
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(coco))
    anns = load_annotations(str(path), file_name="IMG_0001.jpg")
    assert len(anns) == 2


def test_save_annotations_coco(tmp_path):
    """save_annotations dispatches to the COCO writer for fmt='coco'."""
    anns = [Annotation(subject="tree", geometry=BBox(100, 200, 200, 300))]
    path = str(tmp_path / "annotations.json")
    save_annotations(path, anns, 640, 480, fmt="coco", file_name="IMG_0001.jpg")
    with open(path) as f:
        coco = json.load(f)
    assert len(coco["annotations"]) == 1


# ── format-detection refusals ───────────────────────────────────────────────


def test_detect_format_refuses_an_unrecognized_store(tmp_path):
    """A misdetected format reads real annotations as empty negatives, so a wrong answer here is
    worse than no answer. There is no fallback guess left to make."""
    odd = tmp_path / "labels.json"
    odd.write_text(json.dumps({"regions": [{"x": 1}]}))  # an in-house schema we do not know
    with pytest.raises(ValueError, match="Cannot determine the annotation format"):
        detect_format(str(odd))
    with pytest.raises(ValueError):
        detect_format(str(tmp_path / "nothing_here"))


def test_detect_format_refuses_older_objects_keyed_schema(tmp_path):
    """An older 'objects'-keyed schema is not sniffed: it raises rather than reading as zero
    annotations (which would train on fabricated empty negatives)."""
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"image": "a", "objects": [{"category_id": 0, "bbox": [1, 1, 9, 9]}]}))
    with pytest.raises(ValueError):
        detect_format(str(old))
