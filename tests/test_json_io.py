"""json_io: the canonical name-based per-image JSON label format (GT + predictions).

Covers geometry round-trips (xyxy in memory <-> xywh on disk), score handling (predictions
only), provenance persistence, the negative invariant (present-empty file == confirmed
negative, missing file == unannotated), malformed-input robustness, and dataset-COCO
assembly via to_coco_dataset (including a round-trip through format_io's COCO parser).
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
    to_coco_dataset,
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


def test_degenerate_ring_is_dropped_without_losing_its_siblings(tmp_path: Path) -> None:
    # A ring that is not a shape is dropped individually; the annotation and its valid rings survive.
    # (Before multi-ring support a bad first ring could take the whole annotation with it.)
    path = tmp_path / "labels" / "IMG_partial.json"
    write_annotations(
        path,
        [Annotation(subject="bud",
                    geometry=Polygon([[(1.0, 1.0), (2.0, 2.0)], LEFT_LOBE, RIGHT_LOBE]))],
        640, 480)

    (rec,) = _raw(path)["annotations"]
    assert len(rec["segmentation"]) == 2
    (got,) = read_annotations(path)
    assert got.geometry.rings == [LEFT_LOBE, RIGHT_LOBE]


def test_read_drops_only_the_bad_ring_of_a_mixed_segmentation(tmp_path: Path) -> None:
    # Same rule on the read side, from a hand-authored file: an unusable entry (too few coords, odd
    # coord count, an RLE dict) is skipped per-entry, and the usable rings still form the polygon.
    path = tmp_path / "mixed.json"
    payload = {
        "image": "mixed", "width": 100, "height": 100,
        "annotations": [{
            "subject": "bud",
            "segmentation": [
                [10.0, 10.0, 30.0, 10.0, 30.0, 50.0, 10.0, 50.0],
                [1.0, 2.0, 3.0, 4.0],            # < 3 points
                [1, 2, 3, 4, 5, 6, 7],           # odd coord count
                {"counts": "RLE", "size": [2, 2]},
                [70.0, 12.0, 90.0, 12.0, 90.0, 48.0, 70.0, 48.0],
            ],
        }],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    (got,) = read_annotations(path)
    assert got.geometry.rings == [LEFT_LOBE, RIGHT_LOBE]


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
        for k in ("created_by", "created_at", "accepted_by", "accepted_at"):
            assert k not in obj  # omitted entirely, never written as null


def test_partial_provenance_writes_only_set_fields(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "a.json"
    write_annotations(path, [Annotation(subject="bud", geometry=BBox(1.0, 2.0, 3.0, 4.0), created_by="claude")], 100, 100)
    obj = _raw(path)["annotations"][0]
    assert obj["created_by"] == "claude"
    for k in ("created_at", "accepted_by", "accepted_at"):
        assert k not in obj
    (box,) = read_annotations(path)
    assert box.created_by == "claude"
    assert box.created_at is None and box.accepted_by is None and box.accepted_at is None


def test_provenance_set_by_mutation_survives_write(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "a.json"
    write_annotations(path, [Annotation(subject="bud", geometry=BBox(1.0, 2.0, 3.0, 4.0), score=0.5)], 100, 100)
    (pb,) = read_annotations(path)
    pb.created_by = "sam"  # mutating a parsed annotation is the documented pattern
    write_annotations(path, [pb], 100, 100)
    (again,) = read_annotations(path)
    assert again.created_by == "sam"
    assert again.score == 0.5


# -- negative invariant: present-empty == confirmed negative, missing == unannotated --


def test_keep_empty_writes_present_confirmed_negative(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "IMG_0005.json"
    write_annotations(path, [], 640, 480, keep_empty=True)

    assert os.path.exists(path)  # present file == confirmed negative
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


def test_present_negative_is_distinct_from_missing(tmp_path: Path) -> None:
    negative = tmp_path / "labels" / "confirmed_negative.json"
    missing = tmp_path / "labels" / "unannotated.json"
    write_annotations(negative, [], 640, 480, keep_empty=True)

    # Both read as empty, but only the confirmed negative exists on disk.
    assert read_annotations(negative) == []
    assert read_annotations(missing) == []
    assert os.path.exists(negative)
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


def test_bad_bbox_yields_no_box_geometry(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    payload = {
        "image": "a", "width": 100, "height": 100,
        "annotations": [
            {"subject": "bud", "bbox": [1.0, 2.0, 3.0]},          # wrong length
            {"subject": "bud", "bbox": [1, 2, 3, 4, 5]},          # wrong length
            {"subject": "bud", "bbox": "10,20,30,40"},            # not a list
            {"subject": "bud", "bbox": ["a", "b", "c", "d"]},     # non-numeric
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    got = read_annotations(path)
    # No malformed bbox becomes a garbage box.
    assert len(got) == 4
    assert all(a.geometry is None for a in got)


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


def test_bad_segmentation_yields_no_polygon(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    payload = {
        "image": "a", "width": 100, "height": 100,
        "annotations": [
            {"subject": "leaf", "segmentation": [[1.0, 2.0, 3.0, 4.0]]},        # < 3 points
            {"subject": "leaf", "segmentation": [[1, 2, 3, 4, 5, 6, 7]]},       # odd coord count
            {"subject": "leaf", "segmentation": {"counts": "RLE", "size": [2, 2]}},  # RLE dict
            {"subject": "leaf", "segmentation": []},                             # empty
            {"subject": "leaf", "segmentation": [["a", "b", "c", "d", "e", "f"]]},  # non-numeric
            {"subject": "leaf"},                                                 # absent
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    got = read_annotations(path)
    # Every entry still reads back as its own annotation: an unusable segmentation costs the record
    # its geometry, never the record itself (a dropped record would read as a smaller label set).
    assert [a.subject for a in got] == ["leaf"] * 6
    assert all(not isinstance(a.geometry, Polygon) for a in got)  # no bad ring becomes a polygon
    assert all(a.geometry is None for a in got)  # and no other geometry is invented in its place


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


def test_write_skips_degenerate_polygon(tmp_path: Path) -> None:
    # A <3-point polygon is not a shape; the writer must skip it so it can't be written as a record
    # every reader then silently drops (which would masquerade as a confirmed negative).
    path = tmp_path / "labels" / "a.json"
    write_annotations(path, [Annotation(subject="bud", geometry=Polygon([[(1.0, 1.0), (2.0, 2.0)]])),
                             Annotation(subject="leaf", geometry=Polygon([TRIANGLE]))], 100, 100)
    got = read_annotations(path)
    assert [(a.geometry.rings, a.subject) for a in got] == [([TRIANGLE], "leaf")]  # only the valid polygon
    # write<->read symmetry: what a reader can't read, a writer must not write.
    assert all(len(ring) >= 6 for o in _raw(path)["annotations"] for ring in o["segmentation"])

    # A polygon whose every ring is degenerate yields no records -> removed (unannotated), not a
    # bogus file.
    only_bad = tmp_path / "labels" / "b.json"
    write_annotations(only_bad, [Annotation(subject="bud", geometry=Polygon([[(1.0, 1.0), (2.0, 2.0)]]))], 100, 100)
    assert not os.path.exists(only_bad)


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


def test_null_or_bad_score_reads_as_none(tmp_path: Path) -> None:
    payload = {
        "image": "a",
        "annotations": [
            {"subject": "bud", "bbox": [1.0, 2.0, 3.0, 4.0], "score": None},
            {"subject": "leaf", "segmentation": [[0.0, 0.0, 9.0, 0.0, 5.0, 9.0]], "score": "high"},
        ],
    }
    path = tmp_path / "a.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    got = read_annotations(path)
    assert len(got) == 2
    assert got[0].score is None  # null score -> None (a GT annotation), not dropped
    assert got[1].score is None  # non-numeric score -> None, no raise


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
    zero-confidence) prediction, and the score is the only thing separating the two.
    """
    payload = {
        "image": "a", "width": 320, "height": 240,
        "annotations": [
            {"subject": "bud", "bbox": [10.0, 20.0, 100.0, 200.0], "score": True},
            {"subject": "bud", "bbox": [5.0, 6.0, 30.0, 12.0], "score": False},
        ],
    }
    path = tmp_path / "IMG_bool.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    got = read_annotations(path)
    assert len(got) == 2
    assert [a.score for a in got] == [None, None]

    # ...and the assembled dataset keeps them ground truth: no score key rides along.
    coco = to_coco_dataset([(str(path), "IMG_bool.JPG")], subject="bud", id_map={"bud": 0})
    assert len(coco["annotations"]) == 2
    assert all("score" not in a for a in coco["annotations"])


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

    # Assembly re-derives the box from the rings rather than passing the stale one through.
    coco = to_coco_dataset([(str(path), "IMG_stale.JPG")], subject="leaf", id_map={"leaf": 0})
    (ann,) = coco["annotations"]
    assert ann["bbox"] == [10.0, 20.0, 100.0, 200.0]
    assert ann["area"] == 100.0 * 200.0


def test_written_polygon_box_covers_only_the_rings_that_survive(tmp_path: Path) -> None:
    """The box written beside a segmentation spans the kept rings, not the dropped ones.

    A degenerate ring is never written, so a box that still counted its coordinates would describe a
    region no segmentation supports and would stretch the instance toward stray points.
    """
    stray = [(500.0, 400.0), (501.0, 401.0)]  # 2 points: not a shape, dropped on write
    path = tmp_path / "labels" / "IMG_stray.json"
    write_annotations(path, [Annotation(subject="leaf", geometry=Polygon([SQUARE, stray]))], 640, 480)

    (rec,) = _raw(path)["annotations"]
    assert rec["segmentation"] == [[c for xy in SQUARE for c in xy]]
    assert rec["bbox"] == [10.0, 20.0, 100.0, 200.0]

    # The read side agrees, computed by the real box derivation over what was actually kept.
    (got,) = read_annotations(path)
    b = bbox_of(got.geometry)
    assert [b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1] == rec["bbox"]


# -- to_coco_dataset ----------------------------------------------------------


def _mixed_entries(tmp_path: Path) -> list[tuple[str, str]]:
    """GT box + pred box (w/ provenance), a polygon pred, a confirmed negative, a missing file."""
    d = tmp_path / "IMG_0001.json"
    write_annotations(
        d,
        [Annotation(subject="bud", geometry=BBox(10.0, 20.0, 110.0, 220.0)),
         Annotation(subject="bud", geometry=BBox(5.0, 5.0, 15.0, 25.0), score=0.875, **PROV)],
        640, 480,
    )
    s = tmp_path / "IMG_0002.json"
    write_annotations(s, [Annotation(subject="bud", geometry=Polygon([SQUARE]), score=0.5, created_by="sam")], 800, 600)
    neg = tmp_path / "IMG_0003.json"
    write_annotations(neg, [], 640, 480, keep_empty=True)
    missing = tmp_path / "IMG_0004.json"
    return [(str(d), "IMG_0001.JPG"), (str(s), "IMG_0002.JPG"),
            (str(neg), "IMG_0003.JPG"), (str(missing), "IMG_0004.JPG")]


def test_to_coco_dataset_assembles_mixed_entries(tmp_path: Path) -> None:
    coco = to_coco_dataset(_mixed_entries(tmp_path), subject="bud", id_map={"bud": 0},
                           confirmed_negative_names={"IMG_0003.JPG"})

    assert coco["categories"] == [{"id": 0, "name": "bud"}]
    # Present files yield an images record, including IMG_0003's empty file because a human
    # confirmed it negative (an empty file alone never trains as a negative). The missing
    # (unannotated) IMG_0004 is skipped entirely.
    assert [i["file_name"] for i in coco["images"]] == [
        "IMG_0001.JPG", "IMG_0002.JPG", "IMG_0003.JPG"]
    assert [i["id"] for i in coco["images"]] == [1, 2, 3]
    assert coco["images"][0]["width"] == 640 and coco["images"][0]["height"] == 480
    assert coco["images"][1]["width"] == 800 and coco["images"][1]["height"] == 600

    anns = coco["annotations"]
    assert len(anns) == 3  # 2 boxes + 1 polygon; the negative contributes none
    assert [a["id"] for a in anns] == [1, 2, 3]
    assert {a["image_id"] for a in anns} == {1, 2}
    for a in anns:
        assert a["iscrowd"] == 0
        assert a["category_id"] == 0  # scoped to the single subject
        assert len(a["bbox"]) == 4 and "area" in a

    gt_box, pred_box, seg = anns
    assert gt_box["bbox"] == [10.0, 20.0, 100.0, 200.0]
    assert gt_box["area"] == 100.0 * 200.0
    assert "score" not in gt_box and "created_by" not in gt_box

    # score + provenance ride along as COCO-extension keys.
    assert pred_box["score"] == 0.875
    for k, v in PROV.items():
        assert pred_box[k] == v

    # Polygon annotations carry segmentation and a bbox derived from the polygon.
    assert seg["segmentation"] == [[10.0, 20.0, 110.0, 20.0, 110.0, 220.0, 10.0, 220.0]]
    assert seg["bbox"] == [10.0, 20.0, 100.0, 200.0]
    assert seg["area"] == 100.0 * 200.0
    assert seg["score"] == 0.5 and seg["created_by"] == "sam"

    # The confirmed-negative image (id 3) is present but carries no annotations.
    assert not [a for a in anns if a["image_id"] == 3]


def test_to_coco_dataset_round_trips_through_format_io_parser(tmp_path: Path) -> None:
    from tcip_annotation.format_io import parse_coco_annotations

    coco = to_coco_dataset(_mixed_entries(tmp_path), subject="bud", id_map={"bud": 0})

    anns1 = parse_coco_annotations(coco, image_id=1)
    assert [a.subject for a in anns1] == ["bud", "bud"]
    assert [(a.geometry.x1, a.geometry.y1, a.geometry.x2, a.geometry.y2) for a in anns1] == [
        (10.0, 20.0, 110.0, 220.0),
        (5.0, 5.0, 15.0, 25.0),
    ]

    anns2 = parse_coco_annotations(coco, file_name="IMG_0002.JPG")
    assert len(anns2) == 1 and isinstance(anns2[0].geometry, Polygon)
    assert anns2[0].geometry.rings == [SQUARE]
    assert anns2[0].subject == "bud"

    # No confirmed negatives were passed, so the empty IMG_0003 and the missing IMG_0004 are absent.
    file_names = {i["file_name"] for i in coco["images"]}
    assert "IMG_0003.JPG" not in file_names
    assert "IMG_0004.JPG" not in file_names
    assert parse_coco_annotations(coco, image_id=99) == []  # no such image


def test_to_coco_dataset_keeps_every_ring_of_a_multi_ring_instance(tmp_path: Path) -> None:
    # Training assembly must carry the whole occlusion-split instance: one COCO annotation whose
    # segmentation lists both rings, with the box/area spanning their union.
    from tcip_annotation.format_io import parse_coco_annotations

    path = tmp_path / "IMG_multi.json"
    write_annotations(
        path, [Annotation(subject="bud", geometry=Polygon([LEFT_LOBE, RIGHT_LOBE]))], 200, 200)

    coco = to_coco_dataset([(str(path), "IMG_multi.JPG")], subject="bud", id_map={"bud": 0})
    (ann,) = coco["annotations"]
    assert ann["segmentation"] == [
        [10.0, 10.0, 30.0, 10.0, 30.0, 50.0, 10.0, 50.0],
        [70.0, 12.0, 90.0, 12.0, 90.0, 48.0, 70.0, 48.0],
    ]
    assert ann["bbox"] == [10.0, 10.0, 80.0, 40.0]
    assert ann["area"] == 80.0 * 40.0

    # ...and it survives back through the COCO parser as one two-ring polygon.
    (parsed,) = parse_coco_annotations(coco, file_name="IMG_multi.JPG")
    assert parsed.geometry.rings == [LEFT_LOBE, RIGHT_LOBE]


def test_to_coco_dataset_empty_entries(tmp_path: Path) -> None:
    coco = to_coco_dataset([], subject="bud", id_map={"bud": 0})
    assert coco == {"images": [], "annotations": [], "categories": [{"id": 0, "name": "bud"}],
                    "excluded_incomplete_attribute": []}


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


def test_to_coco_dataset_excludes_the_whole_image_when_any_instance_is_unlabeled(
    tmp_path: Path,
) -> None:
    """An instance the annotator hasn't assessed for `attribute` yet must not abort the whole
    assembly with a raise, and must also not be silently narrowed to just its labeled subset,
    training the image's other real objects as background. The whole image is excluded and
    disclosed, the same treatment a missing label file already gets. A genuinely undecodable
    value still raises -- that distinction is unaffected."""
    mixed_path = tmp_path / "IMG_A.json"
    write_annotations(mixed_path, [
        Annotation(subject="bud", geometry=BBox(10, 10, 30, 30),
                  attributes={"opening": "closed"}),
        Annotation(subject="bud", geometry=BBox(40, 40, 60, 60), attributes={}),  # unlabeled
    ], 100, 100)
    fully_labeled_path = tmp_path / "IMG_B.json"
    write_annotations(fully_labeled_path, [
        Annotation(subject="bud", geometry=BBox(10, 10, 30, 30),
                  attributes={"opening": "open"}),
    ], 100, 100)

    coco = to_coco_dataset(
        [(str(mixed_path), "IMG_A.JPG"), (str(fully_labeled_path), "IMG_B.JPG")],
        subject="bud", id_map={"open": 0, "closed": 1}, attribute="opening",
    )

    # The mixed image is excluded wholesale -- not present at all, not even with its labeled
    # instance -- and the exclusion is disclosed, never silent.
    assert [i["file_name"] for i in coco["images"]] == ["IMG_B.JPG"]
    assert len(coco["annotations"]) == 1
    assert coco["annotations"][0]["category_id"] == 0  # "open", from IMG_B only
    # Names, not just a count: the downstream partition (trainable_stems) needs to know which
    # images left, or it attributes their absence to whichever category it resembles and reports
    # a false reason.
    assert coco["excluded_incomplete_attribute"] == ["IMG_A.JPG"]

    undecodable_path = tmp_path / "IMG_C.json"
    write_annotations(undecodable_path, [
        Annotation(subject="bud", geometry=BBox(10, 10, 30, 30),
                  attributes={"opening": "not-a-real-value"}),
    ], 100, 100)
    try:
        to_coco_dataset(
            [(str(undecodable_path), "IMG_C.JPG")],
            subject="bud", id_map={"open": 0, "closed": 1}, attribute="opening",
        )
        raise AssertionError("expected a ValueError")
    except ValueError:
        pass


# -- subject scoping, class ids, and geometry-less labels in assembly ---------


def test_an_image_empty_of_this_subject_is_no_negative_without_confirmation(tmp_path: Path) -> None:
    """A negative is confirmed by a human, per subject, never inferred from an absence.

    An image full of another subject's annotations holds no information about this one, so admitting
    it as a negative would train real objects as background on the strength of a scope mismatch.
    """
    other_subject = tmp_path / "IMG_leaf.json"
    write_annotations(other_subject, [
        Annotation(subject="leaf", geometry=BBox(10.0, 20.0, 110.0, 220.0)),
        Annotation(subject="leaf", geometry=BBox(200.0, 30.0, 240.0, 90.0)),
    ], 640, 480)
    confirmed = tmp_path / "IMG_conf.json"
    write_annotations(confirmed, [], 640, 480, keep_empty=True)
    populated = tmp_path / "IMG_cat.json"
    write_annotations(populated, [Annotation(subject="bud", geometry=BBox(5.0, 6.0, 35.0, 26.0))],
                      640, 480)

    coco = to_coco_dataset(
        [(str(other_subject), "IMG_leaf.JPG"), (str(confirmed), "IMG_conf.JPG"),
         (str(populated), "IMG_cat.JPG")],
        subject="bud", id_map={"bud": 0}, confirmed_negative_names={"IMG_conf.JPG"},
    )

    assert [i["file_name"] for i in coco["images"]] == ["IMG_conf.JPG", "IMG_cat.JPG"]
    (ann,) = coco["annotations"]
    by_id = {i["id"]: i["file_name"] for i in coco["images"]}
    assert by_id[ann["image_id"]] == "IMG_cat.JPG"
    assert ann["bbox"] == [5.0, 6.0, 30.0, 20.0]


def test_categories_and_class_ids_follow_the_run_id_map(tmp_path: Path) -> None:
    """Class ids come from the run's own name to id assignment, which is neither dense nor ordered
    the way the names are, and the emitted categories cover exactly the ids the annotations use."""
    id_map = {"closed": 7, "open": 3}
    closed_img = tmp_path / "IMG_closed.json"
    write_annotations(closed_img, [Annotation(subject="bud", geometry=BBox(10.0, 20.0, 40.0, 90.0),
                                               attributes={"opening": "closed"})], 320, 240)
    open_img = tmp_path / "IMG_open.json"
    write_annotations(open_img, [Annotation(subject="bud", geometry=BBox(1.0, 2.0, 61.0, 12.0),
                                                 attributes={"opening": "open"})], 320, 240)

    coco = to_coco_dataset(
        [(str(closed_img), "IMG_closed.JPG"), (str(open_img), "IMG_open.JPG")],
        subject="bud", id_map=id_map, attribute="opening",
    )

    assert coco["categories"] == [{"id": 3, "name": "open"}, {"id": 7, "name": "closed"}]
    by_image = {i["id"]: i["file_name"] for i in coco["images"]}
    assert {by_image[a["image_id"]]: a["category_id"] for a in coco["annotations"]} == {
        "IMG_closed.JPG": 7, "IMG_open.JPG": 3}
    # Every annotation's class id is one the emitted categories declare.
    assert {a["category_id"] for a in coco["annotations"]} <= {c["id"] for c in coco["categories"]}


def test_whole_image_rating_is_kept_but_never_becomes_a_target(tmp_path: Path) -> None:
    """A geometry-less annotation is a whole-image rating, not a detection or segmentation target.

    It has no box to train or match on, so it takes no class id and adds no COCO annotation; it also
    carries no attribute gap, so its presence never pulls a fully labeled image into the
    incomplete-attribute exclusion.
    """
    id_map = {"closed": 7, "open": 3}
    rating = Annotation(subject="bud", attributes={"vigor": "high"})
    assert target_class_id(rating, "bud", None, {"bud": 0}) is None
    assert target_class_id(rating, "bud", "opening", id_map, allow_unlabeled=True) is None

    path = tmp_path / "IMG_rated.json"
    write_annotations(path, [
        Annotation(subject="bud", geometry=BBox(10.0, 20.0, 40.0, 90.0),
                   attributes={"opening": "closed"}),
        rating,
    ], 320, 240)

    coco = to_coco_dataset([(str(path), "IMG_rated.JPG")], subject="bud", id_map=id_map,
                           attribute="opening")
    assert coco["excluded_incomplete_attribute"] == []
    assert [i["file_name"] for i in coco["images"]] == ["IMG_rated.JPG"]
    (ann,) = coco["annotations"]
    assert ann["category_id"] == 7
    assert ann["bbox"] == [10.0, 20.0, 30.0, 70.0]


def test_rating_only_image_counts_as_annotated_with_no_targets(tmp_path: Path) -> None:
    """An image whose only annotation is a whole-image rating is annotated, not unannotated: it
    enters the dataset with zero targets rather than being dropped or excluded as incomplete."""
    id_map = {"closed": 7, "open": 3}
    path = tmp_path / "IMG_rating_only.json"
    write_annotations(path, [Annotation(subject="bud", attributes={"vigor": "low"})], 320, 240)

    coco = to_coco_dataset([(str(path), "IMG_rating_only.JPG")], subject="bud", id_map=id_map,
                           attribute="opening")
    assert [i["file_name"] for i in coco["images"]] == ["IMG_rating_only.JPG"]
    assert coco["annotations"] == []
    assert coco["excluded_incomplete_attribute"] == []
    assert coco["images"][0]["width"] == 320 and coco["images"][0]["height"] == 240


def test_to_coco_dataset_refuses_a_corrupt_document_behind_a_confirmed_negative_name(
    tmp_path: Path,
) -> None:
    # A confirmed negative is supposed to be an empty label file plus a human's Complete; a
    # corrupt document behind that name must never assemble as a fabricated zero-object image.
    from tcip_annotation.json_io import UnreadableLabelDocument

    path = tmp_path / "IMG_bad.json"
    path.write_text("not json {][", encoding="utf-8")

    with pytest.raises(UnreadableLabelDocument):
        to_coco_dataset([(str(path), "IMG_bad.JPG")], subject="bud", id_map={"bud": 0},
                        confirmed_negative_names={"IMG_bad.JPG"})


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
