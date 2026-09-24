"""The external COCO document's reader: category naming, image grouping, the one decoder, and
every fault returned together rather than the first one raised."""

import json

import pytest

from tcip_annotation.format_io import coco_categories, parse_coco_annotations
from tcip_annotation.state import Polygon


def _no_run_length(segmentation):
    raise AssertionError(f"no run-length mask in this fixture, got {segmentation!r}")


def _parse(coco):
    return parse_coco_annotations(coco, decode_rle=_no_run_length)


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
    categories, by_image, problems = _parse(_sample_coco_detect())
    anns = by_image[1]
    assert problems == []
    assert categories == {0: "tree", 1: "nut"}
    assert len(anns) == 2
    assert {a.subject for a in anns} == {"tree", "nut"}
    # COCO bbox [x, y, w, h] → BBox(x1, y1, x2, y2), subject decoded from the file's categories.
    assert anns[0].geometry.x1 == 100
    assert anns[0].geometry.y1 == 200
    assert anns[0].geometry.x2 == 150
    assert anns[0].geometry.y2 == 260


def test_parse_coco_keys_records_by_the_image_each_names():
    coco = _sample_coco_detect()
    coco["images"].append({"id": 2, "file_name": "IMG_0002.jpg", "width": 640, "height": 480})
    coco["annotations"][1]["image_id"] = 2
    _, parsed, _ = _parse(coco)
    assert {k: [a.subject for a in v] for k, v in parsed.items()} == {1: ["tree"], 2: ["nut"]}


def test_a_crowd_record_reads_with_its_flag():
    coco = _sample_coco_detect()
    coco["annotations"][1]["iscrowd"] = 1
    _, by_image, problems = _parse(coco)
    assert problems == []
    assert [a.iscrowd for a in by_image[1]] == [False, True]


def test_parse_coco_refuses_a_document_without_the_coco_shape():
    """A per-image document handed to the COCO reader is refused by name rather than read as a
    document holding no annotations."""
    with pytest.raises(ValueError, match="not a dataset-level COCO document"):
        _parse({"image": "a", "annotations": []})


@pytest.mark.parametrize("category_id", ["0", 0.0, None, 99, True],
                         ids=["string", "float", "missing", "undeclared", "bool"])
def test_a_record_naming_no_valid_category_is_a_fault_by_index(category_id):
    coco = _sample_coco_detect()
    coco["annotations"][0]["category_id"] = category_id
    _, by_image, problems = _parse(coco)
    assert len(problems) == 1 and problems[0].startswith("record 0 names category_id"), problems
    assert [a.subject for a in by_image[1]] == ["nut"]


@pytest.mark.parametrize("image_id", [None, "1", 1.0], ids=["missing", "string", "float"])
def test_a_record_with_no_integer_image_id_is_a_fault_by_index(image_id):
    """A missing identity never associates under ``None``, and a coercible one never under the
    integer it would coerce to."""
    coco = _sample_coco_detect()
    coco["annotations"][0]["image_id"] = image_id
    _, by_image, problems = _parse(coco)
    assert problems == [f"record 0 names image_id {image_id!r}, not an integer image id"]
    assert list(by_image) == [1] and len(by_image[1]) == 1


@pytest.mark.parametrize("declaration", [
    {"id": 7.5, "name": None}, {"id": "7", "name": "leaf"}, "leaf",
], ids=["float_null", "string_id", "not_an_object"])
def test_a_malformed_category_declaration_is_a_fault_by_index_never_a_coercion(declaration):
    problems: list[str] = []
    categories = coco_categories({"categories": [{"id": 0, "name": "tree"}, declaration]},
                                 problems=problems)
    assert categories == {0: "tree"}
    assert len(problems) == 1 and problems[0].startswith("category 1 "), problems


def test_parse_coco_refuses_two_categories_declared_under_one_id():
    """A second declaration under an id would otherwise rename every record of the first."""
    coco = _sample_coco_detect()
    coco["categories"].append({"id": 0, "name": "leaf"})
    _, _, problems = _parse(coco)
    assert problems == ["category id 0 is declared twice ('tree' and 'leaf')"]


def test_every_independent_fault_is_returned_together():
    """The first malformed declaration or record hides nothing after it."""
    coco = _sample_coco_detect()
    coco["categories"].append({"id": 0, "name": "leaf"})
    coco["categories"].append({"id": 7.5, "name": None})
    coco["annotations"][0]["bbox"] = [100, 200, 0, 60]
    coco["annotations"][1]["category_id"] = 99
    _, _, problems = _parse(coco)
    assert [p.split(" ")[:2] for p in problems] == [
        ["category", "id"], ["category", "3"], ["record", "0"], ["record", "1"]], problems


@pytest.mark.parametrize("annotations, fault", [
    (None, "'annotations' is None, not a list"),
    ([None], "record 0 is None, not an annotation object"),
], ids=["null", "null_record"])
def test_parse_coco_names_a_malformed_annotations_container(annotations, fault):
    coco = _sample_coco_detect()
    coco["annotations"] = annotations
    assert _parse(coco)[2] == [fault]


@pytest.mark.parametrize("key, value, fault", [
    ("attributes", {"stage": ""}, "record 1 attributes"),
    ("bbox", [100, 200, 0, 60], "no positive extent"),
    ("score", "0.8", "record 1 score"),
    ("segmentation", [[1.0, 2.0, 3.0, 4.0]], "three or more points"),
])
def test_parse_coco_decodes_every_record_through_the_per_image_decoder(key, value, fault):
    """The per-image decoder's own terms decide a record, a supplied malformed value refused by
    record index exactly as a per-image document's record would be."""
    coco = _sample_coco_detect()
    coco["annotations"][1][key] = value
    _, by_image, problems = _parse(coco)
    assert len(problems) == 1 and problems[0].startswith("record 1 ") and fault in problems[0], (
        problems)
    assert [a.subject for a in by_image[1]] == ["tree"]


def test_parse_coco_annotations_keeps_an_empty_string_accepted_by():
    """An empty-string provenance value is kept, not dropped: the per-image reader keeps it (is
    not None, never a truthiness test), and the COCO records go through that reader."""
    coco = _sample_coco_detect()
    coco["annotations"][0]["accepted_by"] = ""
    parsed = _parse(coco)[1][1]
    assert parsed[0].accepted_by == ""


# ── COCO polygon parse ──────────────────────────────────────────────────────


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
    anns = _parse(_sample_coco_segment())[1][1]
    assert len(anns) == 1
    assert anns[0].subject == "leaf"
    assert isinstance(anns[0].geometry, Polygon)
    assert anns[0].geometry.rings == [[(10.0, 20.0), (50.0, 20.0), (50.0, 80.0), (10.0, 80.0)]]


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
    (ann,) = _parse(coco)[1][1]
    assert ann.geometry.rings == [LOBE_A, LOBE_B]


# ── the one decode of a document's bytes ────────────────────────────────────


def test_the_coco_reader_admits_a_byte_order_marked_document(tmp_path):
    """A UTF-8 byte-order mark encodes the same document as one without it."""
    from tcip_annotation.json_io import (
        decode_document_bytes, parse_json_document, read_document_bytes,
    )

    coco = _sample_coco_detect()
    path = tmp_path / "annotations.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(coco).encode("utf-8"))

    document = parse_json_document(
        decode_document_bytes(read_document_bytes(path), source=str(path)), source=str(path))
    assert len(_parse(document)[1][1]) == 2
