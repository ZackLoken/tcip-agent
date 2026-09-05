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


def test_detect_format_dir_excludes_a_bucket_sidecar(tmp_path):
    """The directory branch walks through prediction_documents, so a bucket's own provenance
    stamp is never read as a candidate label document, even one carrying a recognizable format
    marker of its own: a directory holding only one is undetectable, the same answer an empty
    directory gives."""
    (tmp_path / "operating_point.json").write_text('{"annotations": []}')
    with pytest.raises(ValueError):
        detect_format(str(tmp_path))


def test_detect_format_per_image_json(tmp_path):
    js = tmp_path / "IMG_0001.json"
    js.write_text('{"image": "IMG_0001", "annotations": [{"subject": "bud", "bbox": [1, 1, 9, 9]}]}')
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


def test_parse_coco_raises_on_a_category_id_that_will_not_coerce():
    from tcip_annotation.json_io import UnreadableLabelDocument

    coco = _sample_coco_detect()
    coco["annotations"][0]["category_id"] = "not-a-number"
    with pytest.raises(UnreadableLabelDocument, match="record 0"):
        parse_coco_annotations(coco, file_name="IMG_0001.jpg")


def test_parse_coco_raises_on_a_category_id_with_no_name_in_the_document():
    from tcip_annotation.json_io import UnreadableLabelDocument

    coco = _sample_coco_detect()
    coco["annotations"][0]["category_id"] = 99
    with pytest.raises(UnreadableLabelDocument, match="record 0"):
        parse_coco_annotations(coco, file_name="IMG_0001.jpg")


def test_parse_coco_admits_every_record_whose_category_resolves():
    """The refusal is per-record, not a document-wide reflex: a document whose every category
    resolves reads in full, the same shape ``test_parse_coco_detect`` already covers."""
    coco = _sample_coco_detect()
    anns = parse_coco_annotations(coco, file_name="IMG_0001.jpg")
    assert len(anns) == 2


def test_parse_coco_matches_a_manifest_named_logical_image_by_stem():
    """A ``.bandgroup`` manifest's own on-disk name never appears verbatim in an externally
    authored COCO document; looking it up ties by stem to the one recorded image that shares it."""
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


def test_parse_coco_non_manifest_lookup_with_no_exact_match_returns_empty():
    """A same-stem record never ties for a non-``.bandgroup`` lookup name: only an exact
    ``file_name`` match, or a ``.bandgroup`` manifest's stem tie, resolves an image."""
    coco = _sample_coco_detect()
    anns = parse_coco_annotations(coco, file_name="IMG_0001.png")
    assert len(anns) == 0


def test_parse_coco_refuses_an_ambiguous_stem_tie():
    coco = _sample_coco_detect()
    coco["images"].append(
        {"id": 2, "file_name": "IMG_0001.png", "width": 10, "height": 10})
    with pytest.raises(ValueError, match="unresolvable ambiguity"):
        parse_coco_annotations(coco, file_name="IMG_0001.bandgroup")


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

# Two disjoint lobes of one instance: a leaf crossed by a stem, a bud behind a branch.
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


# ── load_annotations: a supplied fmt is a claim, not a bypass ───────────────


def test_load_annotations_refuses_a_per_image_document_asked_for_as_coco(tmp_path):
    """A caller-supplied fmt='coco' over a document carrying only the per-image annotations key
    (no images/categories) must not silently answer zero results; it must name the mismatch."""
    js = tmp_path / "IMG_0001.json"
    js.write_text(json.dumps({"image": "IMG_0001", "width": 100, "height": 100,
                              "annotations": [{"subject": "bud", "bbox": [1, 1, 8, 8]}]}))
    with pytest.raises(ValueError, match="coco"):
        load_annotations(str(js), fmt="coco", file_name="IMG_0001.json")


def test_load_annotations_refuses_a_coco_document_asked_for_as_json(tmp_path):
    """The mirror direction: a dataset-level COCO document asked for as the per-image schema."""
    coco = _sample_coco_detect()
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="json"):
        load_annotations(str(path), fmt="json")


def test_load_annotations_accepts_a_coco_document_declared_as_coco(tmp_path):
    coco = _sample_coco_detect()
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(coco))
    anns = load_annotations(str(path), fmt="coco", file_name="IMG_0001.jpg")
    assert len(anns) == 2


def test_load_annotations_accepts_a_per_image_document_declared_as_json(tmp_path):
    js = tmp_path / "IMG_0001.json"
    js.write_text(json.dumps({"image": "IMG_0001", "width": 100, "height": 100,
                              "annotations": [{"subject": "bud", "bbox": [1, 1, 8, 8]}]}))
    anns = load_annotations(str(js), fmt="json")
    assert len(anns) == 1
    assert anns[0].subject == "bud"


def test_load_annotations_refuses_a_document_carrying_neither_shapes_markers(tmp_path):
    """A document whose keys are neither the per-image nor the COCO shape satisfies no stated
    fmt: it is refused under every fmt asked of it, not read as an empty store."""
    odd = tmp_path / "labels.json"
    odd.write_text(json.dumps({"shapes": [{"label": "bud", "points": [[1, 1], [8, 8]]}]}))
    with pytest.raises(ValueError, match="neither"):
        load_annotations(str(odd), fmt="coco", file_name="a.jpg")
    with pytest.raises(ValueError, match="neither"):
        load_annotations(str(odd), fmt="json")


def test_load_annotations_refuses_an_old_objects_schema_document_under_a_stated_fmt(tmp_path):
    """The old 'objects'-keyed schema is refused the same way a format-free read refuses it,
    rather than answering an empty result because it carries neither current shape's markers."""
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"image": "a", "objects": [{"category_id": 0, "bbox": [1, 1, 9, 9]}]}))
    with pytest.raises(ValueError, match="objects"):
        load_annotations(str(old), fmt="json")
    with pytest.raises(ValueError, match="objects"):
        load_annotations(str(old), fmt="coco", file_name="a.jpg")


def test_load_annotations_a_missing_path_stays_absent_under_a_stated_fmt(tmp_path):
    """A stated fmt is a claim checked against a present document's own shape; a path that does
    not exist has no shape to check and reads exactly as an omitted fmt would."""
    missing = tmp_path / "nothing_here.json"
    assert load_annotations(str(missing), fmt="json") == []


# ── the reader's one decode: a byte-order mark, or nothing to decode at all ──


def test_detect_format_and_load_annotations_admit_a_byte_order_marked_coco(tmp_path):
    """A UTF-8 byte-order mark encodes the same document as one without it: detection and the
    COCO parser must agree on that, not disagree at the second decode."""
    coco = _sample_coco_detect()
    path = tmp_path / "annotations.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(coco).encode("utf-8"))

    assert detect_format(str(path)) == "coco"
    anns = load_annotations(str(path), file_name="IMG_0001.jpg")
    assert len(anns) == 2


def test_parse_coco_json_admits_a_byte_order_marked_document(tmp_path):
    from tcip_annotation.format_io import _parse_coco_json

    coco = _sample_coco_detect()
    path = tmp_path / "annotations.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(coco).encode("utf-8"))
    assert _parse_coco_json(str(path))["categories"] == coco["categories"]


def test_parse_coco_json_refuses_an_undecodable_document(tmp_path):
    from tcip_annotation.format_io import _parse_coco_json
    from tcip_annotation.json_io import UnreadableLabelDocument

    path = tmp_path / "annotations.json"
    path.write_bytes(b"{not json")
    with pytest.raises(UnreadableLabelDocument):
        _parse_coco_json(str(path))


def test_write_coco_drops_a_geometry_the_stored_grid_collapses(tmp_path):
    """The interop export applies the per-image export's own drop: a polygon or box that rounds
    to no extent at the stored 2-decimal grid emits no record, while a real one still does."""
    collapsing_polygon = Polygon([[(10.001, 10.001), (10.002, 10.001), (10.002, 10.002)]])
    collapsing_box = BBox(10.001, 10.001, 10.004, 10.004)
    kept = Polygon([[(10, 20), (50, 20), (50, 80), (10, 80)]])
    anns = [Annotation(subject="leaf", geometry=collapsing_polygon),
            Annotation(subject="leaf", geometry=collapsing_box),
            Annotation(subject="leaf", geometry=kept)]
    path = str(tmp_path / "seg.json")
    write_coco(path, {"IMG_0001.jpg": (anns, 640, 480)})

    with open(path) as f:
        coco = json.load(f)

    assert len(coco["annotations"]) == 1
    assert coco["annotations"][0]["bbox"] == [10.0, 20.0, 40.0, 60.0]
