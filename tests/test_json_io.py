"""json_io: the canonical name-based per-image JSON label format (GT + predictions).

Covers geometry round-trips (xyxy in memory <-> xywh on disk), score handling (predictions
only), the crowd flag, provenance persistence, the empty-document invariant (a present empty
document and a missing one both read as no annotations; neither is a negative until a person
confirms one), and malformed input: a supplied value that does not parse refuses the document by
record, only an absent or null key reads as absent, and a document of another shape refuses.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tcip_annotation.json_io import (
    UNLABELED,
    read_annotations,
    require_reference_ground_truth,
    target_class_id,
    write_annotations,
)
from tcip_annotation.state import Annotation, BBox, Polygon, bbox_of

# Coordinates use binary-exact values (.0/.25/.5/.75) so the writer's 2-decimal rounding
# is an identity and geometry assertions can be exact.
SQUARE = [(10.0, 20.0), (110.0, 20.0), (110.0, 220.0), (10.0, 220.0)]
TRIANGLE = [(0.5, 0.25), (30.0, 0.25), (15.25, 40.75)]

# Two disjoint rings of one instance (an occlusion-split object: a bud behind a branch).
LEFT_LOBE = [(10.0, 10.0), (30.0, 10.0), (30.0, 50.0), (10.0, 50.0)]
RIGHT_LOBE = [(70.0, 12.0), (90.0, 12.0), (90.0, 48.0), (70.0, 48.0)]


def _raw(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -- GT box round-trip --------------------------------------------------------


def test_gt_round_trip_geometry_and_subjects(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "IMG_0001.json"
    anns = [Annotation(subject="leaf", geometry=BBox(10.0, 20.0, 110.5, 220.25)),
            Annotation(subject="bud", geometry=BBox(0.0, 0.0, 5.0, 5.0))]
    write_annotations(path, anns, 640, 480)

    got = read_annotations(path)
    assert [a.subject for a in got] == ["leaf", "bud"]
    assert [(a.geometry.x1, a.geometry.y1, a.geometry.x2, a.geometry.y2) for a in got] == [
        (10.0, 20.0, 110.5, 220.25),
        (0.0, 0.0, 5.0, 5.0),
    ]


def test_the_crowd_flag_round_trips_through_the_writer_and_the_reader(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "IMG_crowd.json"
    write_annotations(path, [Annotation(subject="bur", geometry=BBox(1.0, 2.0, 30.0, 40.0),
                                        iscrowd=True),
                             Annotation(subject="bur", geometry=BBox(50.0, 50.0, 60.0, 60.0))],
                      100, 100)

    crowd, single = _raw(path)["annotations"]
    assert crowd["iscrowd"] is True and "iscrowd" not in single  # written only when set
    got = read_annotations(path)
    assert [a.iscrowd for a in got] == [True, False]


def test_the_payload_route_keeps_the_crowd_flag_and_shares_the_decoders_checks() -> None:
    from tcip_annotation.json_io import annotation_from_payload

    kept = annotation_from_payload(
        {"subject": "bur", "bbox": [1, 2, 30, 40], "iscrowd": True, "created_by": "user:a",
         "attributes": {"stage": "ripe"}}, author="user:b", now="2026-01-01T00:00:00+00:00")
    assert kept.iscrowd and kept.attributes == {"stage": "ripe"}
    b = kept.geometry
    assert isinstance(b, BBox) and (b.x1, b.y1, b.x2, b.y2) == (1.0, 2.0, 30.0, 40.0)
    for bad in ({"attributes": {"stage": 12}}, {"attributes": {"stage": ""}}, {"iscrowd": 2},
                {"iscrowd": "yes"}, {"bbox": [10, 10, 5, 5]}, {"bbox": ["1", "2", "3", "4"]}):
        with pytest.raises(ValueError):
            annotation_from_payload({"subject": "bur", **bad}, author=None, now="t")
    assert not annotation_from_payload({"subject": "bur", "iscrowd": None}, author=None,
                                       now="t").iscrowd


@pytest.mark.parametrize("subject", [None, "", 7], ids=["absent", "empty", "not_a_string"])
def test_a_record_naming_no_subject_is_refused_where_an_annotation_is_made(
        tmp_path: Path, subject) -> None:
    """Construction states the subject rule once: a direct constructor refuses, and the decoder,
    reading through it, refuses the document naming the record."""
    from tcip_annotation.json_io import UnreadableLabelDocument

    with pytest.raises(ValueError, match="non-empty string subject"):
        Annotation(subject=subject)  # type: ignore[arg-type]
    path = tmp_path / "labels" / "a.json"
    write_annotations(path, [Annotation(subject="bur", geometry=BBox(1.0, 1.0, 5.0, 5.0))], 10, 10)
    raw = _raw(path)
    raw["annotations"].append({"bbox": [1, 1, 4, 4], **({} if subject is None else {"subject": subject})})
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(UnreadableLabelDocument, match=r"^record 1 .*non-empty string subject"):
        read_annotations(path)


def test_the_payload_route_reads_provenance_presence_as_the_decoder_does() -> None:
    """A payload carrying a provenance key is a round-trip by the decoder's own presence rule (any
    value but null), so every provenance key it carries is kept, an empty creator included; a
    payload carrying none is a new shape stamped to its author."""
    from tcip_annotation.json_io import annotation_from_payload

    kept = annotation_from_payload(
        {"subject": "bur", "created_by": "", "created_at": "2026-01-02", "accepted_by": "user:a",
         "accepted_at": "2026-01-03", "accepted_by_rule": "exp:abc"}, author="user:b", now="t")
    assert (kept.created_by, kept.created_at, kept.accepted_by, kept.accepted_at,
            kept.accepted_by_rule) == ("", "2026-01-02", "user:a", "2026-01-03", "exp:abc")
    new = annotation_from_payload({"subject": "bur", "accepted_by": "user:a"}, author="user:b",
                                  now="t")
    assert (new.created_by, new.created_at, new.accepted_by) == ("user:b", "t", None)


@pytest.mark.parametrize("geometry", [
    {"rings": []}, {"points": []}, {"points": [[1.0, 2.0]]}, {"rings": [[[1, 2], [3, 4]]]},
    {"points": [[1.0, 2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]},
], ids=["no_rings", "no_points", "one_vertex", "two_point_ring", "three_value_vertex"])
def test_a_payload_polygon_that_is_no_shape_refuses_rather_than_reading_as_absent(geometry) -> None:
    # A supplied geometry that is empty or too short is a malformed value, refused by the same
    # decoder a stored record passes, never read as a label with no geometry or dropped on write.
    from tcip_annotation.json_io import annotation_from_payload

    with pytest.raises(ValueError, match="polygon|segmentation|vertex"):
        annotation_from_payload({"subject": "bur", **geometry}, author=None, now="t")


def test_a_payload_polygon_translates_to_the_record_the_decoder_reads() -> None:
    from tcip_annotation.json_io import annotation_from_payload

    pairs = annotation_from_payload({"subject": "bur", "points": [[0, 0], [10, 0], [10, 10]]},
                                    author=None, now="t")
    mappings = annotation_from_payload(
        {"subject": "bur", "rings": [[{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 10, "y": 10}]]},
        author=None, now="t")
    assert pairs.geometry == mappings.geometry == Polygon([[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]])
    # A corner box lands on the stored grid through the one corner conversion.
    box = annotation_from_payload({"subject": "bur", "bbox": [1.004, 2.0, 3.006, 4.0]},
                                  author=None, now="t").geometry
    assert (box.x1, box.y1, box.x2, box.y2) == (1.0, 2.0, 3.0, 4.0)


def test_a_typed_record_whose_attributes_its_reader_refuses_is_refused_at_write(
        tmp_path: Path) -> None:
    path = tmp_path / "labels" / "IMG_attr.json"
    with pytest.raises(ValueError, match="attributes"):
        write_annotations(path, [Annotation(subject="bur", geometry=BBox(1.0, 2.0, 3.0, 4.0),
                                            attributes={"stage": 12})], 100, 100)  # type: ignore[dict-item]
    assert not path.exists()
    write_annotations(path, [Annotation(subject="bur", geometry=BBox(1.0, 2.0, 3.0, 4.0),
                                        attributes={"stage": "ripe"})], 100, 100)
    assert read_annotations(path)[0].attributes == {"stage": "ripe"}


def test_a_string_score_is_refused_by_the_reference_check(tmp_path: Path) -> None:
    # A supplied score that does not parse once read as no score, so the record passed as ground
    # truth; it now refuses where the numeric one refuses as an unadjudicated prediction.
    from tcip_annotation.json_io import UnreadableLabelDocument

    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "a.json").write_text(json.dumps({"image": "a", "annotations": [
        {"subject": "bud", "bbox": [1.0, 2.0, 3.0, 4.0], "score": "0.8"}]}), encoding="utf-8")
    with pytest.raises(UnreadableLabelDocument, match="score"):
        require_reference_ground_truth(reference)


def test_gt_disk_schema_is_coco_xywh_without_score(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "IMG_0001.json"
    write_annotations(path, [Annotation(subject="bud", geometry=BBox(10.0, 20.0, 110.0, 220.0))], 640, 480)

    data = _raw(path)
    assert data["image"] == "IMG_0001"
    assert data["width"] == 640 and data["height"] == 480
    rec = data["annotations"][0]
    # In-memory is xyxy; disk is COCO xywh, keyed by subject name (no numeric class id).
    assert rec["bbox"] == [10.0, 20.0, 100.0, 200.0]
    assert rec["subject"] == "bud"
    # A GT record never carries a score.
    assert all("score" not in o for o in data["annotations"])


# -- prediction box round-trip ------------------------------------------------


def test_pred_round_trip_confidence_via_score(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "IMG_0002.json"
    preds = [
        Annotation(subject="bud", geometry=BBox(10.0, 20.0, 110.0, 220.0), score=0.875),
        Annotation(subject="leaf", geometry=BBox(1.0, 2.0, 3.0, 4.0), score=0.5),
    ]
    write_annotations(path, preds, 640, 480)

    data = _raw(path)
    assert [o["score"] for o in data["annotations"]] == [0.875, 0.5]

    got = read_annotations(path)
    assert [(a.geometry.x1, a.geometry.y1, a.geometry.x2, a.geometry.y2, a.subject, a.score)
            for a in got] == [
        (10.0, 20.0, 110.0, 220.0, "bud", 0.875),
        (1.0, 2.0, 3.0, 4.0, "leaf", 0.5),
    ]


# -- polygon GT + pred round-trip ---------------------------------------------


def test_polygon_gt_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "IMG_0003.json"
    write_annotations(path, [Annotation(subject="leaf", geometry=Polygon([SQUARE])),
                             Annotation(subject="bud", geometry=Polygon([TRIANGLE]))], 640, 480)

    data = _raw(path)
    # segmentation is [[flat pixel coords]] with >= 3 points.
    assert data["annotations"][0]["segmentation"] == [[10.0, 20.0, 110.0, 20.0, 110.0, 220.0, 10.0, 220.0]]
    assert data["annotations"][1]["segmentation"] == [[0.5, 0.25, 30.0, 0.25, 15.25, 40.75]]
    assert all("score" not in o for o in data["annotations"])

    # Each polygon record also carries its derived box (COCO xywh of bbox_of(points)) alongside the
    # segmentation; the polygon stays the source of truth, its box travels with it on disk.
    def _xywh(pts: list[tuple[float, float]]) -> list[float]:
        b = bbox_of(Polygon([pts]))
        return [b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1]

    assert data["annotations"][0]["bbox"] == _xywh(SQUARE) == [10.0, 20.0, 100.0, 200.0]
    assert data["annotations"][1]["bbox"] == _xywh(TRIANGLE) == [0.5, 0.25, 29.5, 40.5]

    # The on-disk bbox is derived, not a second geometry: each record reads back as exactly one
    # polygon annotation (segmentation wins over the co-stored bbox), never a box and a polygon.
    got = read_annotations(path)
    assert len(got) == 2
    assert all(isinstance(a.geometry, Polygon) for a in got)
    assert [(a.geometry.rings, a.subject) for a in got] == [([SQUARE], "leaf"), ([TRIANGLE], "bud")]


def test_multi_ring_polygon_round_trip_keeps_every_ring_in_order(tmp_path: Path) -> None:
    # An occlusion-split instance is one annotation with more than one ring. Every ring must survive
    # the write/read round trip, in authored order; a reader that kept only the first would silently
    # shrink the object, and its derived box with it.
    path = tmp_path / "labels" / "IMG_multi.json"
    write_annotations(
        path, [Annotation(subject="bud", geometry=Polygon([LEFT_LOBE, RIGHT_LOBE]), score=0.5)],
        640, 480)

    (rec,) = _raw(path)["annotations"]
    assert rec["segmentation"] == [
        [10.0, 10.0, 30.0, 10.0, 30.0, 50.0, 10.0, 50.0],
        [70.0, 12.0, 90.0, 12.0, 90.0, 48.0, 70.0, 48.0],
    ]
    # The co-stored box spans both rings, not just the first.
    assert rec["bbox"] == [10.0, 10.0, 80.0, 40.0]

    got = read_annotations(path)
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
    # A ring that cannot be a shape is refused where the polygon is made, never carried to a
    # writer that would drop it and write back fewer rings than were stated.
    with pytest.raises(ValueError, match="three or more points"):
        Polygon(rings)
    assert Polygon([LEFT_LOBE, RIGHT_LOBE]).rings == [LEFT_LOBE, RIGHT_LOBE]


@pytest.mark.parametrize("bad_ring", [
    [1.0, 2.0, 3.0, 4.0],                 # < 3 points
    [1, 2, 3, 4, 5, 6, 7],                # odd coord count
    {"counts": "RLE", "size": [2, 2]},    # a run-length mask: the document carries rings only
], ids=["short", "odd", "run_length"])
def test_a_bad_ring_beside_good_ones_refuses_the_document_by_record(tmp_path: Path, bad_ring) -> None:
    # A supplied ring that is not a ring is a malformed value, never a ring to drop quietly: the
    # document refuses naming the record rather than reading back as fewer rings than it states.
    from tcip_annotation.json_io import UnreadableLabelDocument

    path = tmp_path / "mixed.json"
    payload = {
        "image": "mixed", "width": 100, "height": 100,
        "annotations": [{"subject": "bud", "bbox": [1.0, 1.0, 2.0, 2.0]}, {
            "subject": "bud",
            "segmentation": [
                [10.0, 10.0, 30.0, 10.0, 30.0, 50.0, 10.0, 50.0],
                bad_ring,
                [70.0, 12.0, 90.0, 12.0, 90.0, 48.0, 70.0, 48.0],
            ],
        }],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnreadableLabelDocument, match="record 1 (segmentation|a polygon)"):
        read_annotations(path)


def test_box_only_record_reads_as_single_bbox_annotation(tmp_path: Path) -> None:
    # A hand-drawn box (no segmentation) still reads as exactly one BBox annotation; the polygon's
    # bbox co-storage must not make an ordinary box record ambiguous or double-counted.
    path = tmp_path / "labels" / "IMG_box.json"
    write_annotations(path, [Annotation(subject="bud", geometry=BBox(10.0, 20.0, 110.0, 220.0))], 640, 480)

    obj = _raw(path)["annotations"][0]
    assert "bbox" in obj and "segmentation" not in obj

    got = read_annotations(path)
    assert len(got) == 1 and isinstance(got[0].geometry, BBox)
    g = got[0].geometry
    assert (g.x1, g.y1, g.x2, g.y2) == (10.0, 20.0, 110.0, 220.0)


def test_polygon_pred_round_trip_confidence_via_score(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "IMG_0004.json"
    write_annotations(path, [Annotation(subject="leaf", geometry=Polygon([TRIANGLE]), score=0.75)], 640, 480)

    assert _raw(path)["annotations"][0]["score"] == 0.75

    got = read_annotations(path)
    assert got[0].geometry.rings == [TRIANGLE]
    assert got[0].subject == "leaf"
    assert got[0].score == 0.75


# -- provenance ---------------------------------------------------------------

PROV = {
    "created_by": "sam",
    "created_at": "2026-07-15T10:00:00Z",
    "accepted_by": "user:breeder",
    "accepted_at": "2026-07-15T11:00:00Z",
    "accepted_by_rule": "exp-1:0123456789abcdef",
}


def _assert_prov(shape) -> None:
    for k, v in PROV.items():
        assert getattr(shape, k) == v


def test_provenance_round_trip_box_gt_and_pred(tmp_path: Path) -> None:
    gt_path = tmp_path / "labels" / "gt.json"
    write_annotations(gt_path, [Annotation(subject="bud", geometry=BBox(1.0, 2.0, 3.0, 4.0), **PROV)], 100, 100)
    (gt_box,) = read_annotations(gt_path)
    _assert_prov(gt_box)

    pred_path = tmp_path / "labels" / "pred.json"
    write_annotations(
        pred_path, [Annotation(subject="bud", geometry=BBox(1.0, 2.0, 3.0, 4.0), score=0.5, **PROV)], 100, 100)
    (pred_box,) = read_annotations(pred_path)
    _assert_prov(pred_box)
    assert pred_box.score == 0.5


def test_provenance_round_trip_polygon_gt_and_pred(tmp_path: Path) -> None:
    gt_path = tmp_path / "labels" / "gt.json"
    write_annotations(gt_path, [Annotation(subject="leaf", geometry=Polygon([TRIANGLE]), **PROV)], 100, 100)
    (gt_poly,) = read_annotations(gt_path)
    _assert_prov(gt_poly)

    pred_path = tmp_path / "labels" / "pred.json"
    write_annotations(
        pred_path, [Annotation(subject="leaf", geometry=Polygon([TRIANGLE]), score=0.875, **PROV)], 100, 100)
    (pred_poly,) = read_annotations(pred_path)
    _assert_prov(pred_poly)
    assert pred_poly.score == 0.875


def test_unset_provenance_omitted_from_json_not_null(tmp_path: Path) -> None:
    dpath = tmp_path / "labels" / "a.json"
    write_annotations(dpath, [Annotation(subject="bud", geometry=BBox(1.0, 2.0, 3.0, 4.0))], 100, 100)
    spath = tmp_path / "labels" / "b.json"
    write_annotations(spath, [Annotation(subject="bud", geometry=Polygon([TRIANGLE]))], 100, 100)
    for path in (dpath, spath):
        obj = _raw(path)["annotations"][0]
        for k in ("created_by", "created_at", "accepted_by", "accepted_at", "accepted_by_rule"):
            assert k not in obj  # omitted entirely, never written as null


def test_partial_provenance_writes_only_set_fields(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "a.json"
    write_annotations(path, [Annotation(subject="bud", geometry=BBox(1.0, 2.0, 3.0, 4.0), created_by="claude")], 100, 100)
    obj = _raw(path)["annotations"][0]
    assert obj["created_by"] == "claude"
    for k in ("created_at", "accepted_by", "accepted_at", "accepted_by_rule"):
        assert k not in obj
    (box,) = read_annotations(path)
    assert box.created_by == "claude"
    assert box.created_at is None and box.accepted_by is None and box.accepted_at is None
    assert box.accepted_by_rule is None


def test_provenance_set_by_mutation_survives_write(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "a.json"
    write_annotations(path, [Annotation(subject="bud", geometry=BBox(1.0, 2.0, 3.0, 4.0), score=0.5)], 100, 100)
    (pb,) = read_annotations(path)
    pb.created_by = "sam"  # mutating a parsed annotation is the documented pattern
    write_annotations(path, [pb], 100, 100)
    (again,) = read_annotations(path)
    assert again.created_by == "sam"
    assert again.score == 0.5


# -- empty documents: keep_empty writes one, and neither it nor a missing file is a negative --


def test_keep_empty_writes_a_present_empty_document(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "IMG_0005.json"
    write_annotations(path, [], 640, 480, keep_empty=True)

    assert os.path.exists(path)  # present, and still unannotated until a person confirms it
    assert _raw(path)["annotations"] == []
    assert read_annotations(path) == []


def test_empty_write_without_keep_empty_removes_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "IMG_0006.json"
    write_annotations(path, [Annotation(subject="bud", geometry=BBox(1.0, 2.0, 3.0, 4.0))], 640, 480)
    assert os.path.exists(path)

    write_annotations(path, [], 640, 480, keep_empty=False)
    assert not os.path.exists(path)  # back to unannotated
    assert read_annotations(path) == []


def test_empty_write_without_keep_empty_on_missing_file_is_noop(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "never_written.json"
    write_annotations(path, [], 640, 480, keep_empty=False)  # must not raise
    assert not os.path.exists(path)


def test_missing_file_reads_empty(tmp_path: Path) -> None:
    path = tmp_path / "nope" / "missing.json"
    assert not os.path.exists(path)
    assert read_annotations(path) == []


def test_a_present_empty_document_and_a_missing_one_both_read_empty(tmp_path: Path) -> None:
    present = tmp_path / "labels" / "present_empty.json"
    missing = tmp_path / "labels" / "unannotated.json"
    write_annotations(present, [], 640, 480, keep_empty=True)

    # Both read as empty and only one exists on disk; neither records a person's confirmation.
    assert read_annotations(present) == []
    assert read_annotations(missing) == []
    assert os.path.exists(present)
    assert not os.path.exists(missing)


# -- robustness: a missing file reads empty; a present, unreadable one raises ------------------


def test_non_json_file_raises(tmp_path: Path) -> None:
    from tcip_annotation.json_io import UnreadableLabelDocument

    path = tmp_path / "garbage.json"
    path.write_text("not json {][", encoding="utf-8")

    with pytest.raises(UnreadableLabelDocument):
        read_annotations(path)


def test_json_that_is_not_a_dict_raises(tmp_path: Path) -> None:
    from tcip_annotation.json_io import UnreadableLabelDocument

    for i, payload in enumerate(('[1, 2, 3]', '"a string"', "42", "null")):
        path = tmp_path / f"nondict_{i}.json"
        path.write_text(payload, encoding="utf-8")

        with pytest.raises(UnreadableLabelDocument):
            read_annotations(path)


@pytest.mark.parametrize("payload, named", [
    ({"images": [], "categories": [], "annotations": []}, "import_coco"),
    ({"categories": [{"id": 1, "name": "bud"}],
      "annotations": [{"subject": "bud", "bbox": [1, 1, 9, 9]}]}, "import_coco"),
    ({"image": "a", "objects": [{"label": "bud"}], "annotations": []}, "'objects'"),
], ids=["empty_coco", "subject_bearing_coco", "objects_schema"])
def test_a_document_of_another_shape_is_refused_by_the_one_reader(
    tmp_path: Path, payload: dict, named: str,
) -> None:
    """A dataset-level COCO, even an empty one or one whose records carry ``subject``, and the
    old ``objects`` schema are never read as this image's annotations."""
    from tcip_annotation.json_io import UnreadableLabelDocument

    path = tmp_path / "a.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UnreadableLabelDocument, match=named):
        read_annotations(path)


def test_geometryless_subject_kept_as_image_level_label(tmp_path: Path) -> None:
    # A subject with no geometry is a real image-level label and is kept.
    path = tmp_path / "a.json"
    payload = {
        "image": "a", "width": 100, "height": 100,
        "annotations": [
            {"subject": "bud"},                                # image-level label: kept
            {"subject": "leaf", "bbox": [1.0, 2.0, 3.0, 4.0]},    # box: kept
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    got = read_annotations(path)
    assert [a.subject for a in got] == ["bud", "leaf"]
    assert got[0].geometry is None  # geometry-less label is a real annotation, not dropped
    assert isinstance(got[1].geometry, BBox)


def test_entry_without_subject_raises(tmp_path: Path) -> None:
    # A name-based label is undecodable without a subject: the document raises rather than
    # reading as one record short.
    from tcip_annotation.json_io import UnreadableLabelDocument

    path = tmp_path / "a.json"
    payload = {
        "image": "a", "width": 100, "height": 100,
        "annotations": [{"bbox": [5.0, 6.0, 7.0, 8.0]}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnreadableLabelDocument):
        read_annotations(path)


@pytest.mark.parametrize("key, value", [
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
        tmp_path: Path, key, value) -> None:
    # A present value that is not what the schema states is never read as absent: an absent
    # geometry, score or attribute is a different fact from a supplied one that did not parse.
    from tcip_annotation.json_io import UnreadableLabelDocument

    path = tmp_path / "a.json"
    payload = {
        "image": "a", "width": 100, "height": 100,
        "annotations": [
            {"subject": "bud", "bbox": [1.0, 2.0, 3.0, 4.0]},
            {"subject": "bud", "segmentation": [[0.0, 0.0, 9.0, 0.0, 5.0, 9.0]], key: value},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnreadableLabelDocument, match=f"record 1 {key}"):
        read_annotations(path)


def test_stored_box_with_no_positive_extent_raises(tmp_path: Path) -> None:
    from tcip_annotation.json_io import UnreadableLabelDocument

    path = tmp_path / "a.json"
    payload = {
        "image": "a", "width": 100, "height": 100,
        "annotations": [{"subject": "bud", "bbox": [5.0, 5.0, 0.0, 0.0]}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnreadableLabelDocument):
        read_annotations(path)


@pytest.mark.parametrize("segmentation", [
    [[1.0, 2.0, 3.0, 4.0]],                        # < 3 points
    [[1, 2, 3, 4, 5, 6, 7]],                       # odd coord count
    {"counts": "RLE", "size": [2, 2]},             # a run-length mask: rings only on disk
    [],                                            # empty
    [["a", "b", "c", "d", "e", "f"]],              # non-numeric
], ids=["short", "odd", "run_length", "empty", "non_numeric"])
def test_a_bad_segmentation_refuses_rather_than_falling_back_to_the_box(
        tmp_path: Path, segmentation) -> None:
    # Geometry precedence runs over the keys present, never over the keys that parsed: a record
    # stating a polygon that does not parse is not read as its box.
    from tcip_annotation.json_io import UnreadableLabelDocument

    path = tmp_path / "a.json"
    payload = {
        "image": "a", "width": 100, "height": 100,
        "annotations": [{"subject": "leaf", "segmentation": segmentation,
                         "bbox": [1.0, 2.0, 3.0, 4.0]}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnreadableLabelDocument, match="record 0 (segmentation|a polygon)"):
        read_annotations(path)


@pytest.mark.parametrize("bbox", [[0.0, 0.0, -1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
                         ids=["negative_width", "zero_extent"])
def test_a_box_with_no_extent_refuses_beside_valid_rings(tmp_path: Path, bbox) -> None:
    # A supplied box is checked where it is read, whichever geometry wins precedence: valid rings
    # beside it do not make an invalid box readable.
    from tcip_annotation.json_io import UnreadableLabelDocument

    path = tmp_path / "a.json"
    path.write_text(json.dumps({"image": "a", "annotations": [
        {"subject": "bud", "segmentation": [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0]], "bbox": bbox},
    ]}), encoding="utf-8")

    with pytest.raises(UnreadableLabelDocument, match="record 0 .*no positive extent"):
        read_annotations(path)


def test_an_absent_key_still_reads_as_absent(tmp_path: Path) -> None:
    # The admitting half: a record with no geometry, score, attributes or crowd flag is an
    # image-level ground-truth label, and a null reads as absent too.
    path = tmp_path / "a.json"
    payload = {
        "image": "a", "width": 100, "height": 100,
        "annotations": [
            {"subject": "leaf"},
            {"subject": "bud", "bbox": [1.0, 2.0, 3.0, 4.0], "score": None, "attributes": None,
             "segmentation": None, "point": None, "iscrowd": None},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    leaf, bud = read_annotations(path)
    assert leaf.geometry is None and leaf.score is None and leaf.attributes == {}
    assert isinstance(bud.geometry, BBox) and bud.score is None and not bud.iscrowd


def test_annotations_null_or_absent_raises(tmp_path: Path) -> None:
    from tcip_annotation.json_io import UnreadableLabelDocument

    for i, payload in enumerate(('{"image": "a", "annotations": null}', '{"image": "a"}')):
        path = tmp_path / f"empty_{i}.json"
        path.write_text(payload, encoding="utf-8")

        with pytest.raises(UnreadableLabelDocument):
            read_annotations(path)


def test_the_platforms_own_empty_document_still_reads_empty(tmp_path: Path) -> None:
    # The platform's own shape (what write_annotations(keep_empty=True) writes for a confirmed
    # negative), unlike a null or absent annotations key, keeps reading as empty.
    path = tmp_path / "negative.json"
    path.write_text(json.dumps({"image": "a", "annotations": []}), encoding="utf-8")

    assert read_annotations(path) == []


def test_readers_accept_a_document_of_only_valid_records(tmp_path: Path) -> None:
    payload = {
        "image": "a",
        "annotations": [
            {"subject": "bud", "bbox": [1.0, 2.0, 3.0, 4.0]},
            {"subject": "leaf", "segmentation": [[0.0, 0.0, 9.0, 0.0, 5.0, 9.0]]},
        ],
    }
    path = tmp_path / "a.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    got = read_annotations(path)
    assert len(got) == 2
    assert isinstance(got[0].geometry, BBox)
    assert isinstance(got[1].geometry, Polygon)


@pytest.mark.parametrize("junk", [1, "x", None, [1, 2]])
def test_a_non_dict_annotation_record_raises(tmp_path: Path, junk) -> None:
    # A record that is not an object is undecodable the same way a subject-less one is: reading
    # past it as if it weren't there would let a corrupt record read as a smaller label set.
    from tcip_annotation.json_io import UnreadableLabelDocument

    payload = {"image": "a", "annotations": [junk]}
    path = tmp_path / "a.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnreadableLabelDocument):
        read_annotations(path)


def test_a_null_score_reads_as_ground_truth_and_a_bad_one_refuses(tmp_path: Path) -> None:
    from tcip_annotation.json_io import UnreadableLabelDocument

    null = tmp_path / "null.json"
    null.write_text(json.dumps({"image": "a", "annotations": [
        {"subject": "bud", "bbox": [1.0, 2.0, 3.0, 4.0], "score": None}]}), encoding="utf-8")
    (got,) = read_annotations(null)
    assert got.score is None  # null score -> None (a GT annotation), not dropped

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"image": "a", "annotations": [
        {"subject": "leaf", "segmentation": [[0.0, 0.0, 9.0, 0.0, 5.0, 9.0]], "score": "high"}]}),
        encoding="utf-8")
    with pytest.raises(UnreadableLabelDocument, match="record 0 score"):
        read_annotations(bad)


def test_non_finite_score_is_written_as_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "a.json"
    write_annotations(path, [Annotation(subject="bud", geometry=BBox(1.0, 2.0, 3.0, 4.0), score=float("nan"))], 100, 100)
    # File must be strict-valid JSON (no bare NaN literal); the non-finite score collapses to 0.0.
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["annotations"][0]["score"] == 0.0
    (ann,) = read_annotations(path)
    assert ann.score == 0.0


def test_boolean_score_field_is_not_a_confidence(tmp_path: Path) -> None:
    """``true``/``false`` in a ``score`` field is not a confidence.

    Reading a boolean as 1.0/0.0 would turn ground truth into a maximum-confidence (or a
    zero-confidence) prediction, and the score is the only thing separating the two; reading it
    as absent would turn a supplied value into no value, so the document refuses.
    """
    from tcip_annotation.json_io import UnreadableLabelDocument

    for flag in (True, False):
        payload = {"image": "a", "width": 320, "height": 240, "annotations": [
            {"subject": "bud", "bbox": [10.0, 20.0, 100.0, 200.0], "score": flag}]}
        path = tmp_path / f"IMG_{flag}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(UnreadableLabelDocument, match="record 0 score"):
            read_annotations(path)


def test_polygon_wins_over_a_disagreeing_stored_box(tmp_path: Path) -> None:
    """The polygon is the source of truth, so a co-stored box that disagrees with it is ignored.

    A hand-authored or hand-edited file can carry a stale box next to a segmentation; reading that
    box would shrink the instance to whatever a previous edit left behind.
    """
    path = tmp_path / "IMG_stale.json"
    payload = {
        "image": "IMG_stale", "width": 320, "height": 240,
        "annotations": [{
            "subject": "leaf",
            "segmentation": [[c for xy in SQUARE for c in xy]],
            "bbox": [0.0, 0.0, 5.0, 5.0],
        }],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    (got,) = read_annotations(path)
    assert isinstance(got.geometry, Polygon)
    assert got.geometry.rings == [SQUARE]
    b = bbox_of(got.geometry)
    assert (b.x1, b.y1, b.x2, b.y2) == (10.0, 20.0, 110.0, 220.0)


# -- target_class_id: unlabeled vs. undecodable -------------------------------


def test_target_class_id_distinguishes_unlabeled_from_undecodable() -> None:
    id_map = {"open": 0, "closed": 1}
    unlabeled = Annotation(subject="bud", geometry=BBox(0, 0, 1, 1), attributes={})
    undecodable = Annotation(subject="bud", geometry=BBox(0, 0, 1, 1),
                             attributes={"opening": "not-a-real-value"})
    labeled = Annotation(subject="bud", geometry=BBox(0, 0, 1, 1),
                         attributes={"opening": "closed"})

    # Default (allow_unlabeled=False): both failure shapes raise, unchanged original behavior.
    try:
        target_class_id(unlabeled, "bud", "opening", id_map)
        raise AssertionError("expected a ValueError")
    except ValueError:
        pass

    # allow_unlabeled=True: the soft gap becomes the distinguishable UNLABELED sentinel...
    assert target_class_id(unlabeled, "bud", "opening", id_map, allow_unlabeled=True) == UNLABELED
    # ...but a genuine decode bug (a value the registry doesn't know) still raises regardless.
    try:
        target_class_id(undecodable, "bud", "opening", id_map, allow_unlabeled=True)
        raise AssertionError("expected a ValueError")
    except ValueError:
        pass

    assert target_class_id(labeled, "bud", "opening", id_map, allow_unlabeled=True) == 1


def test_whole_image_rating_never_becomes_a_target(tmp_path) -> None:
    """A geometry-less annotation is a whole-image rating, not a detection or segmentation target:
    it has no box to train or match on, so it takes no class id, and it carries no attribute gap."""
    from tcip_mcp.pipelines.data.label_queries import json_det_targets

    path = tmp_path / "rating.json"
    write_annotations(path, [Annotation(subject="bud", attributes={"vigor": "high"})], 10, 10)
    assert json_det_targets(str(path), "bud", None, {"bud": 0})[0]["boxes"] == []
    target, n_unlabeled = json_det_targets(str(path), "bud", "opening", {"closed": 7, "open": 3})
    assert target["boxes"] == [] and n_unlabeled == 0


# -- the loader, the parser, and the sidecar exclusion ------------------------


def test_parse_label_document_raises_on_undecodable_text() -> None:
    from tcip_annotation.json_io import UnreadableLabelDocument, parse_label_document

    with pytest.raises(UnreadableLabelDocument):
        parse_label_document("not json {][", source="<test>")


def test_parse_label_document_raises_on_a_non_dict_document() -> None:
    from tcip_annotation.json_io import UnreadableLabelDocument, parse_label_document

    with pytest.raises(UnreadableLabelDocument):
        parse_label_document("[1, 2, 3]", source="<test>")


def test_parse_label_document_returns_the_dict() -> None:
    from tcip_annotation.json_io import parse_label_document

    assert parse_label_document('{"annotations": []}', source="<test>") == {"annotations": []}


def test_load_label_document_raises_on_an_unopenable_path(tmp_path: Path) -> None:
    from tcip_annotation.json_io import UnreadableLabelDocument, load_label_document

    with pytest.raises(UnreadableLabelDocument):
        load_label_document(tmp_path / "no_such_directory" / "a.json")


def test_load_label_document_returns_the_dict(tmp_path: Path) -> None:
    from tcip_annotation.json_io import load_label_document

    path = tmp_path / "a.json"
    path.write_text('{"annotations": []}', encoding="utf-8")
    assert load_label_document(path) == {"annotations": []}


def test_prediction_documents_excludes_every_sidecar_filename(tmp_path: Path) -> None:
    from tcip_annotation.json_io import SIDECAR_FILENAMES, prediction_documents

    bucket = tmp_path / "bucket"
    bucket.mkdir()
    write_annotations(bucket / "IMG_0001.json", [Annotation(subject="bud", geometry=BBox(1, 1, 2, 2))],
                      10, 10)
    for name in SIDECAR_FILENAMES:
        (bucket / name).write_text("{}", encoding="utf-8")

    documents = prediction_documents(bucket)

    assert [p.name for p in documents] == ["IMG_0001.json"]


def test_prediction_documents_excludes_a_case_variant_sidecar_filename(tmp_path: Path) -> None:
    """The exclusion is case-insensitive: a stamp saved under a different case still names the
    file a case-insensitive filesystem would collide it with."""
    from tcip_annotation.json_io import prediction_documents

    bucket = tmp_path / "bucket"
    bucket.mkdir()
    write_annotations(bucket / "IMG_0001.json", [Annotation(subject="bud", geometry=BBox(1, 1, 2, 2))],
                      10, 10)
    (bucket / "Operating_Point.json").write_text("{}", encoding="utf-8")

    documents = prediction_documents(bucket)

    assert [p.name for p in documents] == ["IMG_0001.json"]


def test_is_sidecar_name_is_case_insensitive() -> None:
    from tcip_annotation.json_io import is_sidecar_name

    assert is_sidecar_name("Operating_Point.json")
    assert is_sidecar_name("OPERATING_POINT.JSON")
    assert not is_sidecar_name("IMG_0001.json")


def test_prediction_documents_on_a_missing_directory_is_empty(tmp_path: Path) -> None:
    from tcip_annotation.json_io import prediction_documents

    assert prediction_documents(tmp_path / "nope") == []


def test_require_reference_ground_truth_admits_a_bucket_holding_sidecars(tmp_path: Path) -> None:
    # A calibration/holdout reference dir may itself be a prediction bucket carrying its own
    # provenance stamps; those are not label documents and must never be read as one.
    from tcip_annotation.json_io import SIDECAR_FILENAMES

    bucket = tmp_path / "bucket"
    bucket.mkdir()
    write_annotations(bucket / "IMG_0001.json", [Annotation(subject="bud", geometry=BBox(1, 1, 2, 2))],
                      10, 10)
    for name in SIDECAR_FILENAMES:
        (bucket / name).write_text("{}", encoding="utf-8")

    require_reference_ground_truth(bucket)  # must not raise


def test_load_label_document_raises_on_invalid_utf8_bytes(tmp_path: Path) -> None:
    from tcip_annotation.json_io import UnreadableLabelDocument, load_label_document

    path = tmp_path / "a.json"
    path.write_bytes(b'{"annotations": [{"subject": "cat\xffkin"}]}')
    with pytest.raises(UnreadableLabelDocument):
        load_label_document(path)


def test_read_annotations_raises_on_invalid_utf8_bytes(tmp_path: Path) -> None:
    """A write truncated mid multi-byte sequence is a present, unreadable document, not an empty
    one: the writer emits ensure_ascii=False, so this is reachable in practice."""
    from tcip_annotation.json_io import UnreadableLabelDocument

    path = tmp_path / "a.json"
    path.write_bytes(b'{"annotations": [{"subject": "cat\xffkin"}]}')
    with pytest.raises(UnreadableLabelDocument):
        read_annotations(path)


def test_read_annotations_versioned_raises_on_invalid_utf8_bytes(tmp_path: Path) -> None:
    import tcip_store
    from tcip_annotation.json_io import (
        UnreadableLabelDocument, annotation_record_key, read_annotations_versioned,
    )

    key = annotation_record_key(tmp_path, "a")
    tcip_store.put_blob(key, b'{"annotations": [{"subject": "cat\xffkin"}]}')
    with pytest.raises(UnreadableLabelDocument):
        read_annotations_versioned(key)


def test_load_label_document_reads_a_document_carrying_a_utf8_bom(tmp_path: Path) -> None:
    """A UTF-8 byte-order mark encodes the same text as the same document without one; a document
    written under a tool that stamps one must still read."""
    from tcip_annotation.json_io import load_label_document

    path = tmp_path / "a.json"
    path.write_bytes(b"\xef\xbb\xbf" + b'{"annotations": []}')
    assert load_label_document(path) == {"annotations": []}


def test_read_annotations_versioned_reads_a_document_carrying_a_utf8_bom(tmp_path: Path) -> None:
    import tcip_store
    from tcip_annotation.json_io import annotation_record_key, read_annotations_versioned

    key = annotation_record_key(tmp_path, "a")
    tcip_store.put_blob(key, b"\xef\xbb\xbf" + b'{"annotations": []}')
    annotations, _ = read_annotations_versioned(key)
    assert annotations == []


def test_read_annotations_versioned_reads_an_absent_document_as_empty(tmp_path: Path) -> None:
    from tcip_store import Version
    from tcip_annotation.json_io import annotation_record_key, read_annotations_versioned

    key = annotation_record_key(tmp_path, "never_written")
    annotations, version = read_annotations_versioned(key)
    assert annotations == []
    assert version == Version.ABSENT


def test_read_annotations_versioned_and_read_annotations_agree_on_the_same_bytes(
    tmp_path: Path,
) -> None:
    """One decode policy: whatever the file reader accepts or refuses, the store-backed reader
    over the identical bytes must agree."""
    from tcip_annotation.json_io import (
        UnreadableLabelDocument, annotation_record_key, read_annotations_versioned,
        write_annotations,
    )

    path = tmp_path / "a.json"
    write_annotations(path, [Annotation(subject="bud", geometry=BBox(1, 1, 2, 2))], 10, 10)
    key = annotation_record_key(tmp_path, "a")
    annotations, _ = read_annotations_versioned(key)
    assert [a.subject for a in annotations] == [a.subject for a in read_annotations(path)]

    corrupt = tmp_path / "b.json"
    corrupt.write_bytes(b"{not json")
    corrupt_key = annotation_record_key(tmp_path, "b")
    with pytest.raises(UnreadableLabelDocument):
        read_annotations(corrupt)
    with pytest.raises(UnreadableLabelDocument):
        read_annotations_versioned(corrupt_key)


def test_a_box_that_would_round_to_zero_extent_is_refused_at_write(tmp_path: Path) -> None:
    """A box with real pre-round extent that collapses to nothing at the document's stored
    2-decimal quantum must never be written: the writer would otherwise hand the reader a
    document it refuses."""
    path = tmp_path / "a.json"
    sliver = Annotation(subject="bud", geometry=BBox(1.0, 1.0, 1.003, 1.003))
    with pytest.raises(ValueError):
        write_annotations(path, [sliver], 10, 10)


def test_a_polygon_that_would_round_to_a_zero_extent_box_is_refused_at_write(tmp_path: Path) -> None:
    """The polygon branch is checked against its rounded, stored box the same as the box branch:
    a sliver whose derived box collapses to nothing at the stored grid must never be written."""
    path = tmp_path / "a.json"
    sliver = Annotation(subject="bud", geometry=Polygon(
        [[(1.0, 1.0), (1.002, 1.0), (1.002, 1.002)]]
    ))
    with pytest.raises(ValueError):
        write_annotations(path, [sliver], 10, 10)


def test_a_polygon_whose_vertices_all_round_to_one_point_is_refused_at_write(tmp_path: Path) -> None:
    """A polygon's box is checked against the same rounded rings the document stores, not the raw
    ones: vertices that only collapse to one point at the stored 2-decimal grid must never write a
    bbox claiming extent the stored geometry does not have."""
    path = tmp_path / "a.json"
    sliver = Annotation(subject="bud", geometry=Polygon(
        [[(0.996, 0.996), (1.004, 0.996), (1.004, 1.004)]]
    ))
    with pytest.raises(ValueError):
        write_annotations(path, [sliver], 10, 10)


def test_a_box_that_rounds_to_positive_extent_still_writes_and_reads_back(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    real = Annotation(subject="bud", geometry=BBox(1.0, 1.0, 1.02, 1.02))
    write_annotations(path, [real], 10, 10)
    [back] = read_annotations(path)
    assert back.subject == "bud"


def test_geometry_extent_ok_agrees_with_the_writer_on_a_collapsing_polygon() -> None:
    """The pre-write check a caller uses to drop a degenerate detection before it ever reaches
    the writer must reach the writer's own verdict, not a looser one: one implementation."""
    from tcip_annotation.json_io import geometry_extent_ok

    collapsing = Polygon([[(0.996, 0.996), (1.004, 0.996), (1.004, 1.004)]])
    assert geometry_extent_ok(collapsing) is False

    real = Polygon([[(1.0, 1.0), (5.0, 1.0), (5.0, 5.0), (1.0, 5.0)]])
    assert geometry_extent_ok(real) is True


def test_a_prediction_document_the_writer_lands_in_a_bucket_reads_back_through_the_bucket_readers(
    tmp_path: Path,
) -> None:
    """The writer is write_annotations, called the shape pipelines/postprocessing/export.py's
    write_predictions_json calls it: keep_empty=True, each Annotation carrying a score and a
    created_by stamped through tcip_mcp.pipelines.resolution.prediction_producer, landing at
    dataset_layout.prediction_dir(root, model, date) / label_filename(stem). The readers are
    read_annotations (the document's own records), prediction_documents (the bucket listing) and
    dataset_layout.models_with_predictions (the per-date model listing): a document only its
    writer's test has seen is one any of these three readers could silently disagree with.
    """
    from tcip_mcp.dataset_layout import label_filename, models_with_predictions, prediction_dir
    from tcip_mcp.pipelines.resolution import prediction_producer

    root, model, date, stem = tmp_path, "baseline", "2026-02-11", "IMG_1"
    created_by = prediction_producer("checkpoints/baseline.pt", "a" * 64)
    preds = [Annotation(subject="bud", geometry=BBox(10.0, 20.0, 110.0, 220.0), score=0.875,
                        created_by=created_by, created_at="2026-02-11T10:00:00Z")]
    target = prediction_dir(root, model, date) / label_filename(stem)
    write_annotations(str(target), preds, 640, 480, keep_empty=True)

    got = read_annotations(target)
    assert len(got) == 1
    assert got[0].subject == "bud" and got[0].score == 0.875
    assert got[0].created_by == created_by

    from tcip_annotation.json_io import prediction_documents

    assert prediction_documents(prediction_dir(root, model, date)) == [target]

    assert models_with_predictions(root, date) == [model]
    assert models_with_predictions(root, "2026-03-24") == []  # no bucket on this date


def test_prediction_documents_skips_a_directory_named_like_a_json_file(tmp_path: Path) -> None:
    from tcip_annotation.json_io import prediction_documents

    bucket = tmp_path / "bucket"
    bucket.mkdir()
    write_annotations(bucket / "IMG_0001.json", [Annotation(subject="bud", geometry=BBox(1, 1, 2, 2))],
                      10, 10)
    (bucket / "x.json").mkdir()

    documents = prediction_documents(bucket)

    assert [p.name for p in documents] == ["IMG_0001.json"]
