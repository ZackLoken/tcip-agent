"""json_io — the canonical name-based per-image JSON label format (GT + predictions).

Covers geometry round-trips (xyxy in memory <-> xywh on disk), score handling (predictions
only), provenance persistence, the negative invariant (present-empty file == confirmed
negative, missing file == unannotated), malformed-input robustness, and dataset-COCO
assembly via to_coco_dataset (including a round-trip through format_io's COCO parser).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from tcip_annotation.json_io import (
    read_annotations,
    to_coco_dataset,
    write_annotations,
)
from tcip_annotation.state import Annotation, BBox, Polygon

# Coordinates use binary-exact values (.0/.25/.5/.75) so the writer's 2-decimal rounding
# is an identity and geometry assertions can be exact.
SQUARE = [(10.0, 20.0), (110.0, 20.0), (110.0, 220.0), (10.0, 220.0)]
TRIANGLE = [(0.5, 0.25), (30.0, 0.25), (15.25, 40.75)]


def _raw(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── GT box round-trip ────────────────────────────────────────────────────────


def test_gt_round_trip_geometry_and_subjects(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "IMG_0001.json"
    anns = [Annotation(subject="leaf", geometry=BBox(10.0, 20.0, 110.5, 220.25)),
            Annotation(subject="catkin", geometry=BBox(0.0, 0.0, 5.0, 5.0))]
    write_annotations(path, anns, 640, 480)

    got = read_annotations(path)
    assert [a.subject for a in got] == ["leaf", "catkin"]
    assert [(a.geometry.x1, a.geometry.y1, a.geometry.x2, a.geometry.y2) for a in got] == [
        (10.0, 20.0, 110.5, 220.25),
        (0.0, 0.0, 5.0, 5.0),
    ]


def test_gt_disk_schema_is_coco_xywh_without_score(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "IMG_0001.json"
    write_annotations(path, [Annotation(subject="catkin", geometry=BBox(10.0, 20.0, 110.0, 220.0))], 640, 480)

    data = _raw(path)
    assert data["image"] == "IMG_0001"
    assert data["width"] == 640 and data["height"] == 480
    rec = data["annotations"][0]
    # In-memory is xyxy; disk is COCO xywh, keyed by subject name (no numeric class id).
    assert rec["bbox"] == [10.0, 20.0, 100.0, 200.0]
    assert rec["subject"] == "catkin"
    # A GT record never carries a score.
    assert all("score" not in o for o in data["annotations"])


# ── prediction box round-trip ────────────────────────────────────────────────


def test_pred_round_trip_confidence_via_score(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "IMG_0002.json"
    preds = [
        Annotation(subject="catkin", geometry=BBox(10.0, 20.0, 110.0, 220.0), score=0.875),
        Annotation(subject="leaf", geometry=BBox(1.0, 2.0, 3.0, 4.0), score=0.5),
    ]
    write_annotations(path, preds, 640, 480)

    data = _raw(path)
    assert [o["score"] for o in data["annotations"]] == [0.875, 0.5]

    got = read_annotations(path)
    assert [(a.geometry.x1, a.geometry.y1, a.geometry.x2, a.geometry.y2, a.subject, a.score)
            for a in got] == [
        (10.0, 20.0, 110.0, 220.0, "catkin", 0.875),
        (1.0, 2.0, 3.0, 4.0, "leaf", 0.5),
    ]


# ── polygon GT + pred round-trip ─────────────────────────────────────────────


def test_polygon_gt_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "IMG_0003.json"
    write_annotations(path, [Annotation(subject="leaf", geometry=Polygon(SQUARE)),
                             Annotation(subject="catkin", geometry=Polygon(TRIANGLE))], 640, 480)

    data = _raw(path)
    # segmentation is [[flat pixel coords]] with >= 3 points.
    assert data["annotations"][0]["segmentation"] == [[10.0, 20.0, 110.0, 20.0, 110.0, 220.0, 10.0, 220.0]]
    assert data["annotations"][1]["segmentation"] == [[0.5, 0.25, 30.0, 0.25, 15.25, 40.75]]
    assert all("score" not in o for o in data["annotations"])

    got = read_annotations(path)
    assert [(a.geometry.points, a.subject) for a in got] == [(SQUARE, "leaf"), (TRIANGLE, "catkin")]


def test_polygon_pred_round_trip_confidence_via_score(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "IMG_0004.json"
    write_annotations(path, [Annotation(subject="leaf", geometry=Polygon(TRIANGLE), score=0.75)], 640, 480)

    assert _raw(path)["annotations"][0]["score"] == 0.75

    got = read_annotations(path)
    assert got[0].geometry.points == TRIANGLE
    assert got[0].subject == "leaf"
    assert got[0].score == 0.75


# ── provenance ───────────────────────────────────────────────────────────────

PROV = {
    "created_by": "sam",
    "created_at": "2026-07-15T10:00:00Z",
    "accepted_by": "user:zack",
    "accepted_at": "2026-07-15T11:00:00Z",
}


def _assert_prov(shape) -> None:
    for k, v in PROV.items():
        assert getattr(shape, k) == v


def test_provenance_round_trip_box_gt_and_pred(tmp_path: Path) -> None:
    gt_path = tmp_path / "labels" / "gt.json"
    write_annotations(gt_path, [Annotation(subject="catkin", geometry=BBox(1.0, 2.0, 3.0, 4.0), **PROV)], 100, 100)
    (gt_box,) = read_annotations(gt_path)
    _assert_prov(gt_box)

    pred_path = tmp_path / "labels" / "pred.json"
    write_annotations(
        pred_path, [Annotation(subject="catkin", geometry=BBox(1.0, 2.0, 3.0, 4.0), score=0.5, **PROV)], 100, 100)
    (pred_box,) = read_annotations(pred_path)
    _assert_prov(pred_box)
    assert pred_box.score == 0.5


def test_provenance_round_trip_polygon_gt_and_pred(tmp_path: Path) -> None:
    gt_path = tmp_path / "labels" / "gt.json"
    write_annotations(gt_path, [Annotation(subject="leaf", geometry=Polygon(TRIANGLE), **PROV)], 100, 100)
    (gt_poly,) = read_annotations(gt_path)
    _assert_prov(gt_poly)

    pred_path = tmp_path / "labels" / "pred.json"
    write_annotations(
        pred_path, [Annotation(subject="leaf", geometry=Polygon(TRIANGLE), score=0.875, **PROV)], 100, 100)
    (pred_poly,) = read_annotations(pred_path)
    _assert_prov(pred_poly)
    assert pred_poly.score == 0.875


def test_unset_provenance_omitted_from_json_not_null(tmp_path: Path) -> None:
    dpath = tmp_path / "labels" / "a.json"
    write_annotations(dpath, [Annotation(subject="catkin", geometry=BBox(1.0, 2.0, 3.0, 4.0))], 100, 100)
    spath = tmp_path / "labels" / "b.json"
    write_annotations(spath, [Annotation(subject="catkin", geometry=Polygon(TRIANGLE))], 100, 100)
    for path in (dpath, spath):
        obj = _raw(path)["annotations"][0]
        for k in ("created_by", "created_at", "accepted_by", "accepted_at"):
            assert k not in obj  # omitted entirely, never written as null


def test_partial_provenance_writes_only_set_fields(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "a.json"
    write_annotations(path, [Annotation(subject="catkin", geometry=BBox(1.0, 2.0, 3.0, 4.0), created_by="claude")], 100, 100)
    obj = _raw(path)["annotations"][0]
    assert obj["created_by"] == "claude"
    for k in ("created_at", "accepted_by", "accepted_at"):
        assert k not in obj
    (box,) = read_annotations(path)
    assert box.created_by == "claude"
    assert box.created_at is None and box.accepted_by is None and box.accepted_at is None


def test_provenance_set_by_mutation_survives_write(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "a.json"
    write_annotations(path, [Annotation(subject="catkin", geometry=BBox(1.0, 2.0, 3.0, 4.0), score=0.5)], 100, 100)
    (pb,) = read_annotations(path)
    pb.created_by = "sam"  # mutating a parsed annotation is the documented pattern
    write_annotations(path, [pb], 100, 100)
    (again,) = read_annotations(path)
    assert again.created_by == "sam"
    assert again.score == 0.5


# ── negative invariant: present-empty == confirmed negative, missing == unannotated ──


def test_keep_empty_writes_present_confirmed_negative(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "IMG_0005.json"
    write_annotations(path, [], 640, 480, keep_empty=True)

    assert os.path.exists(path)  # present file == confirmed negative
    assert _raw(path)["annotations"] == []
    assert read_annotations(path) == []


def test_empty_write_without_keep_empty_removes_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "IMG_0006.json"
    write_annotations(path, [Annotation(subject="catkin", geometry=BBox(1.0, 2.0, 3.0, 4.0))], 640, 480)
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


# ── robustness: malformed inputs are skipped/empty, never raise ──────────────


def test_non_json_file_reads_empty(tmp_path: Path) -> None:
    path = tmp_path / "garbage.json"
    path.write_text("not json {][", encoding="utf-8")
    assert read_annotations(path) == []


def test_json_that_is_not_a_dict_reads_empty(tmp_path: Path) -> None:
    for i, payload in enumerate(('[1, 2, 3]', '"a string"', "42", "null")):
        path = tmp_path / f"nondict_{i}.json"
        path.write_text(payload, encoding="utf-8")
        assert read_annotations(path) == []


def test_entry_without_subject_skipped_geometryless_kept(tmp_path: Path) -> None:
    # A name-based label is undecodable without a subject, so a subject-less entry is dropped; a
    # subject WITH no geometry is a real image-level label and is kept.
    path = tmp_path / "a.json"
    payload = {
        "image": "a", "width": 100, "height": 100,
        "annotations": [
            {"subject": "catkin"},                                # image-level label — kept
            {"subject": "leaf", "bbox": [1.0, 2.0, 3.0, 4.0]},    # box — kept
            {"bbox": [5.0, 6.0, 7.0, 8.0]},                       # no subject — skipped
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    got = read_annotations(path)
    assert [a.subject for a in got] == ["catkin", "leaf"]
    assert got[0].geometry is None  # geometry-less label is a real annotation, not dropped
    assert isinstance(got[1].geometry, BBox)


def test_bad_bbox_yields_no_box_geometry(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    payload = {
        "image": "a", "width": 100, "height": 100,
        "annotations": [
            {"subject": "catkin", "bbox": [1.0, 2.0, 3.0]},          # wrong length
            {"subject": "catkin", "bbox": [1, 2, 3, 4, 5]},          # wrong length
            {"subject": "catkin", "bbox": "10,20,30,40"},            # not a list
            {"subject": "catkin", "bbox": ["a", "b", "c", "d"]},     # non-numeric
            {"bbox": [1.0, 2.0, 3.0, 4.0]},                          # no subject — skipped entirely
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    got = read_annotations(path)
    # No malformed bbox becomes a garbage box; the subject-less entry is dropped.
    assert len(got) == 4
    assert all(a.geometry is None for a in got)


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
    assert all(not isinstance(a.geometry, Polygon) for a in got)  # no bad ring becomes a polygon


def test_annotations_null_or_absent_reads_empty(tmp_path: Path) -> None:
    for i, payload in enumerate(('{"image": "a", "annotations": null}', '{"image": "a"}')):
        path = tmp_path / f"empty_{i}.json"
        path.write_text(payload, encoding="utf-8")
        assert read_annotations(path) == []


def test_write_skips_degenerate_polygon(tmp_path: Path) -> None:
    # A <3-point polygon is not a shape; the writer must skip it so it can't be written as a record
    # every reader then silently drops (which would masquerade as a confirmed negative).
    path = tmp_path / "labels" / "a.json"
    write_annotations(path, [Annotation(subject="catkin", geometry=Polygon([(1.0, 1.0), (2.0, 2.0)])),
                             Annotation(subject="leaf", geometry=Polygon(TRIANGLE))], 100, 100)
    got = read_annotations(path)
    assert [(a.geometry.points, a.subject) for a in got] == [(TRIANGLE, "leaf")]  # only the valid polygon
    # write<->read symmetry: what a reader can't read, a writer must not write.
    assert all(len(o["segmentation"][0]) >= 6 for o in _raw(path)["annotations"])

    # A list of ONLY degenerate polygons yields no records -> removed (unannotated), not a bogus file.
    only_bad = tmp_path / "labels" / "b.json"
    write_annotations(only_bad, [Annotation(subject="catkin", geometry=Polygon([(1.0, 1.0), (2.0, 2.0)]))], 100, 100)
    assert not os.path.exists(only_bad)


def test_readers_tolerate_non_dict_annotations(tmp_path: Path) -> None:
    payload = {
        "image": "a",
        "annotations": [
            1, "x", None, [1, 2],  # junk entries that must not raise
            {"subject": "catkin", "bbox": [1.0, 2.0, 3.0, 4.0]},
            {"subject": "leaf", "segmentation": [[0.0, 0.0, 9.0, 0.0, 5.0, 9.0]]},
        ],
    }
    path = tmp_path / "a.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    got = read_annotations(path)
    assert len(got) == 2
    assert isinstance(got[0].geometry, BBox)
    assert isinstance(got[1].geometry, Polygon)


def test_null_or_bad_score_reads_as_none(tmp_path: Path) -> None:
    payload = {
        "image": "a",
        "annotations": [
            {"subject": "catkin", "bbox": [1.0, 2.0, 3.0, 4.0], "score": None},
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
    write_annotations(path, [Annotation(subject="catkin", geometry=BBox(1.0, 2.0, 3.0, 4.0), score=float("nan"))], 100, 100)
    # File must be STRICT-valid JSON (no bare NaN literal); the non-finite score collapses to 0.0.
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["annotations"][0]["score"] == 0.0
    (ann,) = read_annotations(path)
    assert ann.score == 0.0


# ── to_coco_dataset ──────────────────────────────────────────────────────────


def _mixed_entries(tmp_path: Path) -> list[tuple[str, str]]:
    """GT box + pred box (w/ provenance), a polygon pred, a confirmed negative, a missing file."""
    d = tmp_path / "IMG_0001.json"
    write_annotations(
        d,
        [Annotation(subject="catkin", geometry=BBox(10.0, 20.0, 110.0, 220.0)),
         Annotation(subject="catkin", geometry=BBox(5.0, 5.0, 15.0, 25.0), score=0.875, **PROV)],
        640, 480,
    )
    s = tmp_path / "IMG_0002.json"
    write_annotations(s, [Annotation(subject="catkin", geometry=Polygon(SQUARE), score=0.5, created_by="sam")], 800, 600)
    neg = tmp_path / "IMG_0003.json"
    write_annotations(neg, [], 640, 480, keep_empty=True)
    missing = tmp_path / "IMG_0004.json"
    return [(str(d), "IMG_0001.JPG"), (str(s), "IMG_0002.JPG"),
            (str(neg), "IMG_0003.JPG"), (str(missing), "IMG_0004.JPG")]


def test_to_coco_dataset_assembles_mixed_entries(tmp_path: Path) -> None:
    coco = to_coco_dataset(_mixed_entries(tmp_path), subject="catkin", id_map={"catkin": 0},
                           confirmed_negative_names={"IMG_0003.JPG"})

    assert coco["categories"] == [{"id": 0, "name": "catkin"}]
    # Present files yield an images record — including IMG_0003's empty file because a human
    # CONFIRMED it negative (an empty file alone never trains as a negative). The missing
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

    coco = to_coco_dataset(_mixed_entries(tmp_path), subject="catkin", id_map={"catkin": 0})

    anns1 = parse_coco_annotations(coco, image_id=1)
    assert [a.subject for a in anns1] == ["catkin", "catkin"]
    assert [(a.geometry.x1, a.geometry.y1, a.geometry.x2, a.geometry.y2) for a in anns1] == [
        (10.0, 20.0, 110.0, 220.0),
        (5.0, 5.0, 15.0, 25.0),
    ]

    anns2 = parse_coco_annotations(coco, file_name="IMG_0002.JPG")
    assert len(anns2) == 1 and isinstance(anns2[0].geometry, Polygon)
    assert anns2[0].geometry.points == SQUARE
    assert anns2[0].subject == "catkin"

    # No confirmed negatives were passed, so the empty IMG_0003 and the missing IMG_0004 are absent.
    file_names = {i["file_name"] for i in coco["images"]}
    assert "IMG_0003.JPG" not in file_names
    assert "IMG_0004.JPG" not in file_names
    assert parse_coco_annotations(coco, image_id=99) == []  # no such image


def test_to_coco_dataset_empty_entries(tmp_path: Path) -> None:
    coco = to_coco_dataset([], subject="catkin", id_map={"catkin": 0})
    assert coco == {"images": [], "annotations": [], "categories": [{"id": 0, "name": "catkin"}]}
