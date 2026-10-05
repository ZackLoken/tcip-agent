"""json_io: the canonical name-based per-image label document (GT + predictions), a record in the
dataset root's database.

Covers geometry round-trips (xyxy in memory <-> xywh in the record), score handling (predictions
only), the crowd flag, provenance persistence, the empty-document invariant (a present empty
document and a missing one both read as no annotations; neither is a negative until a person
confirms one), and malformed input: a supplied value that does not parse refuses the document by
record, only an absent or null key reads as absent, and a document of another shape refuses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store
from tcip_annotation.json_io import (
    UNASSESSED,
    UndeclaredValue,
    UnreadableLabelDocument,
    attribute_ids,
    attribute_values,
    read_label_document,
    require_reference_ground_truth,
    write_label_document,
)
from tcip_annotation.state import Annotation, BBox, Polygon, bbox_of
from tcip_mcp.dataset_layout import UNDATED_BUCKET, label_key

# Coordinates use binary-exact values (.0/.25/.5/.75) so the writer's 2-decimal rounding
# is an identity and geometry assertions can be exact.
SQUARE = [(10.0, 20.0), (110.0, 20.0), (110.0, 220.0), (10.0, 220.0)]
TRIANGLE = [(0.5, 0.25), (30.0, 0.25), (15.25, 40.75)]

# Two disjoint rings of one instance (an occlusion-split object: a bud behind a branch).
LEFT_LOBE = [(10.0, 10.0), (30.0, 10.0), (30.0, 50.0), (10.0, 50.0)]
RIGHT_LOBE = [(70.0, 12.0), (90.0, 12.0), (90.0, 48.0), (70.0, 48.0)]


def _key(tmp_path: Path, stem: str = "a") -> tcip_store.Key:
    """The label document key of the undated image ``stem`` under ``tmp_path``."""
    return label_key(tmp_path, UNDATED_BUCKET, stem)


def _read(key: tcip_store.Key) -> list[Annotation]:
    return read_label_document(key).annotations


def _stored(tmp_path: Path, payload, stem: str = "a") -> tcip_store.Key:
    """``payload`` stored as-is under the label document key of ``stem``, past the writer."""
    key = _key(tmp_path, stem)
    tcip_store.replace(key, payload)
    return key


# -- GT box round-trip --------------------------------------------------------


def test_gt_round_trip_geometry_and_subjects(tmp_path: Path) -> None:
    key = _key(tmp_path, "IMG_0001")
    anns = [Annotation(subject="leaf", geometry=BBox(10.0, 20.0, 110.5, 220.25)),
            Annotation(subject="bud", geometry=BBox(0.0, 0.0, 5.0, 5.0))]
    write_label_document(key, anns, 640, 480)

    got = _read(key)
    assert [a.subject for a in got] == ["leaf", "bud"]
    assert [(a.geometry.x1, a.geometry.y1, a.geometry.x2, a.geometry.y2) for a in got] == [
        (10.0, 20.0, 110.5, 220.25),
        (0.0, 0.0, 5.0, 5.0),
    ]


def test_the_crowd_flag_round_trips_through_the_writer_and_the_reader(tmp_path: Path) -> None:
    key = _key(tmp_path, "IMG_crowd")
    write_label_document(key, [Annotation(subject="bur", geometry=BBox(1.0, 2.0, 30.0, 40.0),
                                          iscrowd=True),
                               Annotation(subject="bur", geometry=BBox(50.0, 50.0, 60.0, 60.0))],
                         100, 100)

    crowd, single = tcip_store.read(key)["annotations"]
    assert crowd["iscrowd"] is True and "iscrowd" not in single  # written only when set
    got = _read(key)
    assert [a.iscrowd for a in got] == [True, False]


def test_the_payload_route_keeps_the_crowd_flag_and_shares_the_decoders_checks() -> None:
    from tcip_annotation.json_io import annotation_from_payload

    kept = annotation_from_payload(
        {"subject": "bur", "bbox": [1, 2, 30, 40], "iscrowd": True, "created_by": "user:a",
         "attributes": {"stage": "ripe"}})
    assert kept.iscrowd and kept.attributes == {"stage": "ripe"}
    b = kept.geometry
    assert isinstance(b, BBox) and (b.x1, b.y1, b.x2, b.y2) == (1.0, 2.0, 30.0, 40.0)
    for bad in ({"attributes": {"stage": 12}}, {"attributes": {"stage": ""}}, {"iscrowd": 2},
                {"iscrowd": "yes"}, {"bbox": [10, 10, 5, 5]}, {"bbox": ["1", "2", "3", "4"]}):
        with pytest.raises(ValueError):
            annotation_from_payload({"subject": "bur", **bad})
    assert not annotation_from_payload({"subject": "bur", "iscrowd": None}).iscrowd


@pytest.mark.parametrize("subject", [None, "", 7], ids=["absent", "empty", "not_a_string"])
def test_a_record_naming_no_subject_is_refused_where_an_annotation_is_made(
        tmp_path: Path, subject) -> None:
    """Construction states the subject rule once: a direct constructor refuses, and the decoder,
    reading through it, refuses the document naming the record."""
    with pytest.raises(ValueError, match="non-empty string subject"):
        Annotation(subject=subject)  # type: ignore[arg-type]
    key = _key(tmp_path)
    write_label_document(key, [Annotation(subject="bur", geometry=BBox(1.0, 1.0, 5.0, 5.0))],
                         10, 10)
    raw = tcip_store.read(key)
    raw["annotations"].append(
        {"bbox": [1, 1, 4, 4], **({} if subject is None else {"subject": subject})})
    tcip_store.replace(key, raw)
    with pytest.raises(UnreadableLabelDocument, match=r"record 1 .*non-empty string subject"):
        _read(key)


def test_provenance_is_the_stored_records_or_the_actors_never_the_payloads() -> None:
    """A payload's provenance keys are not read: a content the document already stores keeps
    that record's provenance, and any other is the actor's at the save's time."""
    from tcip_annotation.json_io import annotation_from_payload, stamped

    stored = Annotation(subject="bur", geometry=BBox(1.0, 2.0, 3.0, 4.0), created_by="model:m",
                        created_at="2026-01-02", accepted_by="user:a", accepted_at="2026-01-03")
    kept, new = stamped([
        annotation_from_payload({"subject": "bur", "bbox": [1, 2, 3, 4], "created_by": "user:x"}),
        annotation_from_payload({"subject": "bur", "bbox": [5, 6, 7, 8], "accepted_by": "user:x"}),
    ], [stored], actor="user:b", now="t")
    assert (kept.created_by, kept.created_at, kept.accepted_by, kept.accepted_at) == (
        "model:m", "2026-01-02", "user:a", "2026-01-03")
    assert (new.created_by, new.created_at, new.accepted_by) == ("user:b", "t", None)


@pytest.mark.parametrize("geometry", [
    {"rings": []}, {"points": []}, {"points": [[1.0, 2.0]]}, {"rings": [[[1, 2], [3, 4]]]},
    {"points": [[1.0, 2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]},
], ids=["no_rings", "no_points", "one_vertex", "two_point_ring", "three_value_vertex"])
def test_a_payload_polygon_that_is_no_shape_refuses_rather_than_reading_as_absent(geometry) -> None:
    """A supplied geometry that is empty or too short is a malformed value, refused by the same
    decoder a stored record passes, never read as a label with no geometry or dropped on write."""
    from tcip_annotation.json_io import annotation_from_payload

    with pytest.raises(ValueError, match="polygon|segmentation|vertex"):
        annotation_from_payload({"subject": "bur", **geometry})


def test_a_payload_polygon_translates_to_the_record_the_decoder_reads() -> None:
    from tcip_annotation.json_io import annotation_from_payload

    pairs = annotation_from_payload({"subject": "bur", "points": [[0, 0], [10, 0], [10, 10]]})
    mappings = annotation_from_payload(
        {"subject": "bur", "rings": [[{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 10, "y": 10}]]})
    assert pairs.geometry == mappings.geometry == Polygon([[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]])
    # A corner box lands on the stored grid through the one corner conversion.
    box = annotation_from_payload({"subject": "bur", "bbox": [1.004, 2.0, 3.006, 4.0]}).geometry
    assert (box.x1, box.y1, box.x2, box.y2) == (1.0, 2.0, 3.0, 4.0)


def test_a_typed_record_whose_attributes_its_reader_refuses_is_refused_at_write(
        tmp_path: Path) -> None:
    key = _key(tmp_path, "IMG_attr")
    with pytest.raises(ValueError, match="attributes"):
        write_label_document(key, [Annotation(subject="bur", geometry=BBox(1.0, 2.0, 3.0, 4.0),
                                              attributes={"stage": 12})], 100, 100)  # type: ignore[dict-item]
    assert not tcip_store.exists(key)
    write_label_document(key, [Annotation(subject="bur", geometry=BBox(1.0, 2.0, 3.0, 4.0),
                                          attributes={"stage": "ripe"})], 100, 100)
    assert _read(key)[0].attributes == {"stage": "ripe"}


def test_a_string_score_is_refused_by_the_reference_check(tmp_path: Path) -> None:
    """A supplied score that does not parse once read as no score, so the record passed as ground
    truth; it now refuses where the numeric one refuses as an unadjudicated prediction."""
    key = _stored(tmp_path, {"annotations": [
        {"subject": "bud", "bbox": [1.0, 2.0, 3.0, 4.0], "score": "0.8"}]})
    with pytest.raises(UnreadableLabelDocument, match="score"):
        require_reference_ground_truth(_read(key))


def test_gt_record_schema_is_coco_xywh_without_score(tmp_path: Path) -> None:
    key = _key(tmp_path, "IMG_0001")
    write_label_document(key, [Annotation(subject="bud", geometry=BBox(10.0, 20.0, 110.0, 220.0))],
                         640, 480)

    data = tcip_store.read(key)
    assert set(data) == {"width", "height", "annotations"}
    assert data["width"] == 640 and data["height"] == 480
    rec = data["annotations"][0]
    # In-memory is xyxy; the record is COCO xywh, keyed by subject name (no numeric class id).
    assert rec["bbox"] == [10.0, 20.0, 100.0, 200.0]
    assert rec["subject"] == "bud"
    # A GT record never carries a score.
    assert all("score" not in o for o in data["annotations"])


# -- prediction box round-trip ------------------------------------------------


def test_pred_round_trip_confidence_via_score(tmp_path: Path) -> None:
    key = _key(tmp_path, "IMG_0002")
    preds = [
        Annotation(subject="bud", geometry=BBox(10.0, 20.0, 110.0, 220.0), score=0.875),
        Annotation(subject="leaf", geometry=BBox(1.0, 2.0, 3.0, 4.0), score=0.5),
    ]
    write_label_document(key, preds, 640, 480)

    data = tcip_store.read(key)
    assert [o["score"] for o in data["annotations"]] == [0.875, 0.5]

    got = _read(key)
    assert [(a.geometry.x1, a.geometry.y1, a.geometry.x2, a.geometry.y2, a.subject, a.score)
            for a in got] == [
        (10.0, 20.0, 110.0, 220.0, "bud", 0.875),
        (1.0, 2.0, 3.0, 4.0, "leaf", 0.5),
    ]


# -- polygon GT + pred round-trip ---------------------------------------------


def test_polygon_gt_round_trip(tmp_path: Path) -> None:
    key = _key(tmp_path, "IMG_0003")
    write_label_document(key, [Annotation(subject="leaf", geometry=Polygon([SQUARE])),
                               Annotation(subject="bud", geometry=Polygon([TRIANGLE]))], 640, 480)

    data = tcip_store.read(key)
    # segmentation is [[flat pixel coords]] with >= 3 points.
    assert data["annotations"][0]["segmentation"] == [
        [10.0, 20.0, 110.0, 20.0, 110.0, 220.0, 10.0, 220.0]]
    assert data["annotations"][1]["segmentation"] == [[0.5, 0.25, 30.0, 0.25, 15.25, 40.75]]
    assert all("score" not in o for o in data["annotations"])
    # A polygon record stores its rings only; its box is derived on read, never a second spelling.
    assert all("bbox" not in o for o in data["annotations"])

    got = _read(key)
    assert len(got) == 2
    assert all(isinstance(a.geometry, Polygon) for a in got)
    assert [(a.geometry.rings, a.subject) for a in got] == [([SQUARE], "leaf"), ([TRIANGLE], "bud")]


def test_multi_ring_polygon_round_trip_keeps_every_ring_in_order(tmp_path: Path) -> None:
    """An occlusion-split instance is one annotation with more than one ring. Every ring must
    survive the write/read round trip, in authored order; a reader that kept only the first would
    silently shrink the object, and the box derived from it with it."""
    key = _key(tmp_path, "IMG_multi")
    write_label_document(
        key, [Annotation(subject="bud", geometry=Polygon([LEFT_LOBE, RIGHT_LOBE]), score=0.5)],
        640, 480)

    (rec,) = tcip_store.read(key)["annotations"]
    assert rec["segmentation"] == [
        [10.0, 10.0, 30.0, 10.0, 30.0, 50.0, 10.0, 50.0],
        [70.0, 12.0, 90.0, 12.0, 90.0, 48.0, 70.0, 48.0],
    ]
    got = _read(key)
    assert len(got) == 1  # one annotation, not one per ring
    assert got[0].geometry.rings == [LEFT_LOBE, RIGHT_LOBE]
    b = bbox_of(got[0].geometry)
    assert (b.x1, b.y1, b.x2, b.y2) == (10.0, 10.0, 90.0, 50.0)


@pytest.mark.parametrize("rings", [
    [[(1.0, 1.0), (2.0, 2.0)], LEFT_LOBE],   # a two-point ring beside a real one
    [[(1.0, 1.0)]],                          # one vertex
    [],                                      # no ring at all
], ids=["two_point_beside_a_ring", "one_vertex", "no_ring"])
def test_a_polygon_with_a_ring_that_is_no_shape_cannot_be_made(rings) -> None:
    """A ring that cannot be a shape is refused where the polygon is made, never carried to a
    writer that would drop it and write back fewer rings than were stated."""
    with pytest.raises(ValueError, match="three or more points"):
        Polygon(rings)
    assert Polygon([LEFT_LOBE, RIGHT_LOBE]).rings == [LEFT_LOBE, RIGHT_LOBE]


@pytest.mark.parametrize("bad_ring", [
    [1.0, 2.0, 3.0, 4.0],                 # < 3 points
    [1, 2, 3, 4, 5, 6, 7],                # odd coord count
    {"counts": "RLE", "size": [2, 2]},    # a run-length mask: the document carries rings only
], ids=["short", "odd", "run_length"])
def test_a_bad_ring_beside_good_ones_refuses_the_document_by_record(tmp_path: Path, bad_ring) -> None:
    """A supplied ring that is not a ring is a malformed value, never a ring to drop quietly: the
    document refuses naming the record rather than reading back as fewer rings than it states."""
    key = _stored(tmp_path, {
        "width": 100, "height": 100,
        "annotations": [{"subject": "bud", "bbox": [1.0, 1.0, 2.0, 2.0]}, {
            "subject": "bud",
            "segmentation": [
                [10.0, 10.0, 30.0, 10.0, 30.0, 50.0, 10.0, 50.0],
                bad_ring,
                [70.0, 12.0, 90.0, 12.0, 90.0, 48.0, 70.0, 48.0],
            ],
        }],
    })

    with pytest.raises(UnreadableLabelDocument, match="record 1 (segmentation|a polygon)"):
        _read(key)


def test_box_only_record_reads_as_single_bbox_annotation(tmp_path: Path) -> None:
    """A hand-drawn box (no segmentation) still reads as exactly one BBox annotation; the
    polygon's bbox co-storage must not make an ordinary box record ambiguous or double-counted."""
    key = _key(tmp_path, "IMG_box")
    write_label_document(key, [Annotation(subject="bud", geometry=BBox(10.0, 20.0, 110.0, 220.0))],
                         640, 480)

    obj = tcip_store.read(key)["annotations"][0]
    assert "bbox" in obj and "segmentation" not in obj

    got = _read(key)
    assert len(got) == 1 and isinstance(got[0].geometry, BBox)
    g = got[0].geometry
    assert (g.x1, g.y1, g.x2, g.y2) == (10.0, 20.0, 110.0, 220.0)


def test_polygon_pred_round_trip_confidence_via_score(tmp_path: Path) -> None:
    key = _key(tmp_path, "IMG_0004")
    write_label_document(
        key, [Annotation(subject="leaf", geometry=Polygon([TRIANGLE]), score=0.75)], 640, 480)

    assert tcip_store.read(key)["annotations"][0]["score"] == 0.75

    got = _read(key)
    assert got[0].geometry.rings == [TRIANGLE]
    assert got[0].subject == "leaf"
    assert got[0].score == 0.75


# -- provenance ---------------------------------------------------------------

PROV = {
    "created_by": "sam",
    "created_at": "2026-07-15T10:00:00Z",
    "accepted_by": "user:breeder",
    "accepted_at": "2026-07-15T11:00:00Z",
}


def _assert_prov(shape) -> None:
    for k, v in PROV.items():
        assert getattr(shape, k) == v


def test_provenance_round_trip_box_gt_and_pred(tmp_path: Path) -> None:
    gt = _key(tmp_path, "gt")
    write_label_document(
        gt, [Annotation(subject="bud", geometry=BBox(1.0, 2.0, 3.0, 4.0), **PROV)], 100, 100)
    (gt_box,) = _read(gt)
    _assert_prov(gt_box)

    pred = _key(tmp_path, "pred")
    write_label_document(
        pred, [Annotation(subject="bud", geometry=BBox(1.0, 2.0, 3.0, 4.0), score=0.5, **PROV)],
        100, 100)
    (pred_box,) = _read(pred)
    _assert_prov(pred_box)
    assert pred_box.score == 0.5


def test_provenance_round_trip_polygon_gt_and_pred(tmp_path: Path) -> None:
    gt = _key(tmp_path, "gt")
    write_label_document(
        gt, [Annotation(subject="leaf", geometry=Polygon([TRIANGLE]), **PROV)], 100, 100)
    (gt_poly,) = _read(gt)
    _assert_prov(gt_poly)

    pred = _key(tmp_path, "pred")
    write_label_document(
        pred, [Annotation(subject="leaf", geometry=Polygon([TRIANGLE]), score=0.875, **PROV)],
        100, 100)
    (pred_poly,) = _read(pred)
    _assert_prov(pred_poly)
    assert pred_poly.score == 0.875


def test_unset_provenance_omitted_from_the_record_not_null(tmp_path: Path) -> None:
    box = _key(tmp_path, "a")
    write_label_document(box, [Annotation(subject="bud", geometry=BBox(1.0, 2.0, 3.0, 4.0))],
                         100, 100)
    polygon = _key(tmp_path, "b")
    write_label_document(polygon, [Annotation(subject="bud", geometry=Polygon([TRIANGLE]))],
                         100, 100)
    for key in (box, polygon):
        obj = tcip_store.read(key)["annotations"][0]
        for k in ("created_by", "created_at", "accepted_by", "accepted_at"):
            assert k not in obj  # omitted entirely, never written as null


def test_partial_provenance_writes_only_set_fields(tmp_path: Path) -> None:
    key = _key(tmp_path)
    write_label_document(
        key, [Annotation(subject="bud", geometry=BBox(1.0, 2.0, 3.0, 4.0), created_by="claude")],
        100, 100)
    obj = tcip_store.read(key)["annotations"][0]
    assert obj["created_by"] == "claude"
    for k in ("created_at", "accepted_by", "accepted_at"):
        assert k not in obj
    (box,) = _read(key)
    assert box.created_by == "claude"
    assert box.created_at is None and box.accepted_by is None and box.accepted_at is None


def test_provenance_set_by_mutation_survives_write(tmp_path: Path) -> None:
    key = _key(tmp_path)
    write_label_document(
        key, [Annotation(subject="bud", geometry=BBox(1.0, 2.0, 3.0, 4.0), score=0.5)], 100, 100)
    (pb,) = _read(key)
    pb.created_by = "sam"  # mutating a parsed annotation is the documented pattern
    write_label_document(key, [pb], 100, 100)
    (again,) = _read(key)
    assert again.created_by == "sam"
    assert again.score == 0.5


# -- empty documents: keep_empty writes one, and neither it nor a missing one is a negative ----


def test_keep_empty_writes_a_present_empty_document(tmp_path: Path) -> None:
    key = _key(tmp_path, "IMG_0005")
    write_label_document(key, [], 640, 480, keep_empty=True)

    assert tcip_store.exists(key)  # present, and still unannotated until a person confirms it
    assert tcip_store.read(key)["annotations"] == []
    assert _read(key) == []


def test_empty_write_without_keep_empty_removes_an_existing_document(tmp_path: Path) -> None:
    key = _key(tmp_path, "IMG_0006")
    write_label_document(key, [Annotation(subject="bud", geometry=BBox(1.0, 2.0, 3.0, 4.0))],
                         640, 480)
    assert tcip_store.exists(key)

    write_label_document(key, [], 640, 480, keep_empty=False)
    assert not tcip_store.exists(key)  # back to unannotated


def test_a_present_empty_document_and_a_missing_one_both_read_empty(tmp_path: Path) -> None:
    from tcip_annotation.json_io import read_document_versioned

    present = _key(tmp_path, "present_empty")
    missing = _key(tmp_path, "unannotated")
    write_label_document(present, [], 640, 480, keep_empty=True)

    # Both read as empty and only one exists; neither records a person's confirmation.
    assert _read(present) == []
    assert read_document_versioned(missing)[0].annotations == []
    assert tcip_store.exists(present)
    assert not tcip_store.exists(missing)


def test_an_absent_document_reads_empty_at_the_absent_version(tmp_path: Path) -> None:
    from tcip_annotation.json_io import read_document_versioned

    doc, version = read_document_versioned(_key(tmp_path, "never_written"))
    assert doc.annotations == [] and doc.marks == {}
    assert version == tcip_store.Version.ABSENT


# -- robustness: a missing document reads empty; a present, unreadable one raises --------------


def test_a_record_that_does_not_decode_raises_the_one_unreadable_refusal(tmp_path: Path) -> None:
    from tests._record_damage_fixtures import damage_record

    key = _key(tmp_path)
    write_label_document(key, [], 10, 10, keep_empty=True)
    damage_record(key, b"not json {][")

    with pytest.raises(UnreadableLabelDocument, match="does not decode"):
        _read(key)


@pytest.mark.parametrize("payload", [[1, 2, 3], "a string", 42],
                         ids=["list", "string", "number"])
def test_a_record_that_is_not_an_object_raises(tmp_path: Path, payload) -> None:
    key = _stored(tmp_path, payload)

    with pytest.raises(UnreadableLabelDocument):
        _read(key)


def test_geometryless_subject_kept_as_image_level_label(tmp_path: Path) -> None:
    # A subject with no geometry is a real image-level label and is kept.
    key = _stored(tmp_path, {
        "width": 100, "height": 100,
        "annotations": [
            {"subject": "bud"},                                # image-level label: kept
            {"subject": "leaf", "bbox": [1.0, 2.0, 3.0, 4.0]},    # box: kept
        ],
    })

    got = _read(key)
    assert [a.subject for a in got] == ["bud", "leaf"]
    assert got[0].geometry is None  # geometry-less label is a real annotation, not dropped
    assert isinstance(got[1].geometry, BBox)


def test_entry_without_subject_raises(tmp_path: Path) -> None:
    """A name-based label is undecodable without a subject: the document raises rather than
    reading as one record short."""
    key = _stored(tmp_path, {
        "width": 100, "height": 100,
        "annotations": [{"bbox": [5.0, 6.0, 7.0, 8.0]}],
    })

    with pytest.raises(UnreadableLabelDocument):
        _read(key)


@pytest.mark.parametrize("key_name, value", [
    ("bbox", [1.0, 2.0, 3.0]),                  # wrong length
    ("bbox", [1, 2, 3, 4, 5]),                  # wrong length
    ("bbox", "10,20,30,40"),                    # not a list
    ("bbox", ["10", "20", "30", "40"]),         # numeric strings are not numbers
    ("point", [5.0]),                           # not two numbers
    ("point", ["5", "6"]),                      # not numbers
    ("score", "0.8"),                           # not a number
    ("score", True),                            # a flag is not a confidence
    ("attributes", {"stage": ""}),              # an empty value name
    ("attributes", {"stage": 12}),              # a value that is not a name
    ("attributes", ["stage"]),                  # not a mapping
    ("iscrowd", 2),                             # not a crowd flag
    ("iscrowd", "yes"),                         # not a crowd flag
])
def test_a_supplied_malformed_value_refuses_the_document_by_record(
        tmp_path: Path, key_name, value) -> None:
    """A present value that is not what the schema states is never read as absent: an absent
    geometry, score or attribute is a different fact from a supplied one that did not parse."""
    key = _stored(tmp_path, {
        "width": 100, "height": 100,
        "annotations": [
            {"subject": "bud", "bbox": [1.0, 2.0, 3.0, 4.0]},
            {"subject": "bud", "segmentation": [[0.0, 0.0, 9.0, 0.0, 5.0, 9.0]],
             key_name: value},
        ],
    })

    with pytest.raises(UnreadableLabelDocument, match=f"record 1 {key_name}"):
        _read(key)


def test_stored_box_with_no_positive_extent_raises(tmp_path: Path) -> None:
    key = _stored(tmp_path, {
        "width": 100, "height": 100,
        "annotations": [{"subject": "bud", "bbox": [5.0, 5.0, 0.0, 0.0]}],
    })

    with pytest.raises(UnreadableLabelDocument):
        _read(key)


@pytest.mark.parametrize("segmentation", [
    [[1.0, 2.0, 3.0, 4.0]],                        # < 3 points
    [[1, 2, 3, 4, 5, 6, 7]],                       # odd coord count
    {"counts": "RLE", "size": [2, 2]},             # a run-length mask: rings only in a record
    [],                                            # empty
    [["a", "b", "c", "d", "e", "f"]],              # non-numeric
], ids=["short", "odd", "run_length", "empty", "non_numeric"])
def test_a_bad_segmentation_refuses_rather_than_falling_back_to_the_box(
        tmp_path: Path, segmentation) -> None:
    """Geometry precedence runs over the keys present, never over the keys that parsed: a record
    stating a polygon that does not parse is not read as its box."""
    key = _stored(tmp_path, {
        "width": 100, "height": 100,
        "annotations": [{"subject": "leaf", "segmentation": segmentation,
                         "bbox": [1.0, 2.0, 3.0, 4.0]}],
    })

    with pytest.raises(UnreadableLabelDocument, match="record 0 (segmentation|a polygon)"):
        _read(key)


@pytest.mark.parametrize("bbox", [[0.0, 0.0, -1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
                         ids=["negative_width", "zero_extent"])
def test_a_box_with_no_extent_refuses_beside_valid_rings(tmp_path: Path, bbox) -> None:
    """A supplied box is checked where it is read, whichever geometry wins precedence: valid
    rings beside it do not make an invalid box readable."""
    key = _stored(tmp_path, {"annotations": [
        {"subject": "bud", "segmentation": [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0]], "bbox": bbox},
    ]})

    with pytest.raises(UnreadableLabelDocument, match="record 0 .*no positive extent"):
        _read(key)


def test_an_absent_key_still_reads_as_absent(tmp_path: Path) -> None:
    """The admitting half: a record with no geometry, score, attributes or crowd flag is an
    image-level ground-truth label, and a null reads as absent too."""
    key = _stored(tmp_path, {
        "width": 100, "height": 100,
        "annotations": [
            {"subject": "leaf"},
            {"subject": "bud", "bbox": [1.0, 2.0, 3.0, 4.0], "score": None, "attributes": None,
             "segmentation": None, "point": None, "iscrowd": None},
        ],
    })

    leaf, bud = _read(key)
    assert leaf.geometry is None and leaf.score is None and leaf.attributes == {}
    assert isinstance(bud.geometry, BBox) and bud.score is None and not bud.iscrowd


@pytest.mark.parametrize("payload", [{"annotations": None}, {"width": 100}],
                         ids=["null", "absent"])
def test_annotations_null_or_absent_raises(tmp_path: Path, payload) -> None:
    key = _stored(tmp_path, payload)

    with pytest.raises(UnreadableLabelDocument):
        _read(key)


def test_the_platforms_own_empty_document_still_reads_empty(tmp_path: Path) -> None:
    """The platform's own shape (what write_label_document(keep_empty=True) writes for a
    confirmed negative), unlike a null or absent annotations key, keeps reading as empty."""
    key = _stored(tmp_path, {"annotations": []})

    assert _read(key) == []


def test_readers_accept_a_document_of_only_valid_records(tmp_path: Path) -> None:
    key = _stored(tmp_path, {
        "annotations": [
            {"subject": "bud", "bbox": [1.0, 2.0, 3.0, 4.0]},
            {"subject": "leaf", "segmentation": [[0.0, 0.0, 9.0, 0.0, 5.0, 9.0]]},
        ],
    })
    got = _read(key)
    assert len(got) == 2
    assert isinstance(got[0].geometry, BBox)
    assert isinstance(got[1].geometry, Polygon)


@pytest.mark.parametrize("junk", [1, "x", None, [1, 2]])
def test_a_non_dict_annotation_record_raises(tmp_path: Path, junk) -> None:
    """A record that is not an object is undecodable the same way a subject-less one is: reading
    past it as if it weren't there would let a corrupt record read as a smaller label set."""
    key = _stored(tmp_path, {"annotations": [junk]})

    with pytest.raises(UnreadableLabelDocument):
        _read(key)


def test_a_null_score_reads_as_ground_truth_and_a_bad_one_refuses(tmp_path: Path) -> None:
    null = _stored(tmp_path, {"annotations": [
        {"subject": "bud", "bbox": [1.0, 2.0, 3.0, 4.0], "score": None}]}, "null")
    (got,) = _read(null)
    assert got.score is None  # null score -> None (a GT annotation), not dropped

    bad = _stored(tmp_path, {"annotations": [
        {"subject": "leaf", "segmentation": [[0.0, 0.0, 9.0, 0.0, 5.0, 9.0]], "score": "high"}]},
        "bad")
    with pytest.raises(UnreadableLabelDocument, match="record 0 score"):
        _read(bad)


def test_a_non_finite_score_is_refused_rather_than_written(tmp_path: Path) -> None:
    """A confidence that is no finite number is refused by the writer, never collapsed to a
    number that would read as one, and nothing lands."""
    key = _key(tmp_path)

    with pytest.raises(ValueError, match="not a finite number"):
        write_label_document(key, [Annotation(subject="bud", geometry=BBox(1.0, 2.0, 3.0, 4.0),
                                              score=float("nan"))], 100, 100)
    assert not tcip_store.exists(key)


@pytest.mark.parametrize("flag", [True, False])
def test_boolean_score_field_is_not_a_confidence(tmp_path: Path, flag) -> None:
    """``true``/``false`` in a ``score`` field is not a confidence.

    Reading a boolean as 1.0/0.0 would turn ground truth into a maximum-confidence (or a
    zero-confidence) prediction, and the score is the only thing separating the two; reading it
    as absent would turn a supplied value into no value, so the document refuses.
    """
    key = _stored(tmp_path, {"width": 320, "height": 240, "annotations": [
        {"subject": "bud", "bbox": [10.0, 20.0, 100.0, 200.0], "score": flag}]})
    with pytest.raises(UnreadableLabelDocument, match="record 0 score"):
        _read(key)


def test_polygon_wins_over_a_disagreeing_stored_box(tmp_path: Path) -> None:
    """The polygon is the source of truth, so a co-stored box that disagrees with it is ignored.

    A record written past the writer can carry a stale box next to a segmentation; reading that
    box would shrink the instance to whatever a previous edit left behind.
    """
    key = _stored(tmp_path, {
        "width": 320, "height": 240,
        "annotations": [{
            "subject": "leaf",
            "segmentation": [[c for xy in SQUARE for c in xy]],
            "bbox": [0.0, 0.0, 5.0, 5.0],
        }],
    })

    (got,) = _read(key)
    assert isinstance(got.geometry, Polygon)
    assert got.geometry.rings == [SQUARE]
    b = bbox_of(got.geometry)
    assert (b.x1, b.y1, b.x2, b.y2) == (10.0, 20.0, 110.0, 220.0)


# -- absence is the store's fact, never the value's ----------------------------


def test_absence_reads_as_no_document_and_a_stored_null_refuses(tmp_path: Path) -> None:
    """The editor's read answers no document only where the store holds no record; a present
    record holding ``null`` is a value that does not decode, refused like any other, and a reader
    requiring a record refuses its absence."""
    from tcip_annotation.json_io import NO_DOCUMENT, read_document_versioned

    key = _key(tmp_path)
    assert read_document_versioned(key) == (NO_DOCUMENT, tcip_store.Version.ABSENT)
    with pytest.raises(UnreadableLabelDocument, match="has no record"):
        read_label_document(key)

    _stored(tmp_path, None)

    with pytest.raises(UnreadableLabelDocument, match="not the object a label document is"):
        read_document_versioned(key)


# -- attribute ids: unassessed vs. undeclared ---------------------------------


def test_attribute_ids_distinguish_unassessed_from_undeclared() -> None:
    """An instance carrying no value for an attribute is unassessed for it and reads the mark in
    that column; a value the attribute does not declare refuses naming both; a record of another
    subject is no row at all; the encoder reads an id row back into values."""
    from tcip_mcp.subject_registry import Attribute

    attributes = (Attribute("opening", "categorical", ("open", "closed")),
                  Attribute("grade", "ordinal", ("low", "high")))
    unassessed = Annotation(subject="bud", geometry=BBox(0, 0, 1, 1), attributes={"grade": "high"})
    undeclared = Annotation(subject="bud", geometry=BBox(0, 0, 1, 1),
                            attributes={"opening": "not-a-real-value"})
    labeled = Annotation(subject="bud", geometry=BBox(0, 0, 1, 1),
                         attributes={"opening": "closed", "grade": "low"})

    assert attribute_ids(unassessed, "bud", attributes) == [UNASSESSED, 1]
    with pytest.raises(UndeclaredValue, match="opening='not-a-real-value'"):
        attribute_ids(undeclared, "bud", attributes)
    assert attribute_ids(labeled, "bud", attributes) == [1, 0]
    assert attribute_ids(labeled, "bush", attributes) is None
    assert attribute_values([1, 0], attributes) == {"opening": "closed", "grade": "low"}
    assert attribute_values([UNASSESSED, 1], attributes) == {"grade": "high"}


def test_whole_image_rating_never_becomes_a_target(tmp_path) -> None:
    """A geometry-less annotation is a whole-image rating, not a detection or segmentation target:
    it has no box to train or match on, so it takes no row, whatever attributes its scope
    declares."""
    from tcip_mcp import subject_registry as cr
    from tcip_mcp.pipelines.data.label_queries import json_det_targets, registry_scope
    from tests._producer_fixtures import registry_over

    key = _key(tmp_path, "rating")
    write_label_document(key, [Annotation(subject="bud", attributes={"vigor": "high"})], 10, 10)
    registry_over(tmp_path, cr.SubjectRegistry(subjects=(cr.Subject(name="bud", attributes=(
        cr.Attribute("opening", "categorical", ("closed", "open")),)),)))
    target = json_det_targets(_read(key), registry_scope(tmp_path / "images", "bud"))
    assert target["boxes"] == [] and target["attributes"].shape == (0, 1)


# -- the writer's own extent rail ---------------------------------------------


def test_a_box_that_would_round_to_zero_extent_is_refused_at_write(tmp_path: Path) -> None:
    """A box with real pre-round extent that collapses to nothing at the document's stored
    2-decimal quantum must never be written: the writer would otherwise hand the reader a
    document it refuses."""
    sliver = Annotation(subject="bud", geometry=BBox(1.0, 1.0, 1.003, 1.003))
    with pytest.raises(ValueError):
        write_label_document(_key(tmp_path), [sliver], 10, 10)


def test_a_polygon_that_would_round_to_a_zero_extent_box_is_refused_at_write(tmp_path: Path) -> None:
    """The polygon branch is checked against its rounded, stored box the same as the box branch:
    a sliver whose derived box collapses to nothing at the stored grid must never be written."""
    sliver = Annotation(subject="bud", geometry=Polygon(
        [[(1.0, 1.0), (1.002, 1.0), (1.002, 1.002)]]
    ))
    with pytest.raises(ValueError):
        write_label_document(_key(tmp_path), [sliver], 10, 10)


def test_a_polygon_whose_vertices_all_round_to_one_point_is_refused_at_write(tmp_path: Path) -> None:
    """A polygon's box is checked against the same rounded rings the document stores, not the raw
    ones: vertices that only collapse to one point at the stored 2-decimal grid must never write a
    bbox claiming extent the stored geometry does not have."""
    sliver = Annotation(subject="bud", geometry=Polygon(
        [[(0.996, 0.996), (1.004, 0.996), (1.004, 1.004)]]
    ))
    with pytest.raises(ValueError):
        write_label_document(_key(tmp_path), [sliver], 10, 10)


def test_a_box_that_rounds_to_positive_extent_still_writes_and_reads_back(tmp_path: Path) -> None:
    key = _key(tmp_path)
    real = Annotation(subject="bud", geometry=BBox(1.0, 1.0, 1.02, 1.02))
    write_label_document(key, [real], 10, 10)
    [back] = _read(key)
    assert back.subject == "bud"


def test_geometry_extent_ok_agrees_with_the_writer_on_a_collapsing_polygon() -> None:
    """The pre-write check a caller uses to drop a degenerate detection before it ever reaches
    the writer must reach the writer's own verdict, not a looser one: one implementation."""
    from tcip_annotation.json_io import geometry_extent_ok

    collapsing = Polygon([[(0.996, 0.996), (1.004, 0.996), (1.004, 1.004)]])
    assert geometry_extent_ok(collapsing) is False

    real = Polygon([[(1.0, 1.0), (5.0, 1.0), (5.0, 5.0), (1.0, 5.0)]])
    assert geometry_extent_ok(real) is True
