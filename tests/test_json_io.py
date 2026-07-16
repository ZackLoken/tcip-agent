"""json_io — the canonical per-image COCO-shaped JSON label format (GT + predictions).

Covers geometry round-trips (xyxy in memory <-> xywh on disk), score handling (predictions
only), provenance persistence, the negative invariant (present-empty file == confirmed
negative, missing file == unannotated), malformed-input robustness, and dataset-COCO
assembly via to_coco_dataset (including a round-trip through format_io's COCO parsers).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from tcip_annotation.format_io import parse_coco_detect, parse_coco_segment
from tcip_annotation.json_io import (
    read_detect,
    read_detect_pred,
    read_segment,
    read_segment_pred,
    to_coco_dataset,
    write_detect,
    write_segment,
)
from tcip_annotation.state import BBox, Polygon, PredBBox, PredPolygon

# Coordinates use binary-exact values (.0/.25/.5/.75) so the writer's 2-decimal rounding
# is an identity and geometry assertions can be exact.
SQUARE = [(10.0, 20.0), (110.0, 20.0), (110.0, 220.0), (10.0, 220.0)]
TRIANGLE = [(0.5, 0.25), (30.0, 0.25), (15.25, 40.75)]


def _raw(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── GT detect round-trip ─────────────────────────────────────────────────────


def test_detect_gt_round_trip_geometry_and_classes(tmp_path: Path) -> None:
    path = tmp_path / "detect" / "IMG_0001.json"
    boxes = [BBox(10.0, 20.0, 110.5, 220.25, 1), BBox(0.0, 0.0, 5.0, 5.0, 0)]
    write_detect(path, boxes, 640, 480)

    got, class_ids = read_detect(path)
    assert class_ids == {0, 1}
    assert [(b.x1, b.y1, b.x2, b.y2, b.class_id) for b in got] == [
        (10.0, 20.0, 110.5, 220.25, 1),
        (0.0, 0.0, 5.0, 5.0, 0),
    ]


def test_detect_gt_disk_schema_is_coco_xywh_without_score(tmp_path: Path) -> None:
    path = tmp_path / "detect" / "IMG_0001.json"
    write_detect(path, [BBox(10.0, 20.0, 110.0, 220.0, 3)], 640, 480)

    data = _raw(path)
    assert data["image"] == "IMG_0001"
    assert data["width"] == 640 and data["height"] == 480
    # In-memory is xyxy; disk is COCO xywh.
    assert data["objects"][0]["bbox"] == [10.0, 20.0, 100.0, 200.0]
    assert data["objects"][0]["category_id"] == 3
    # A GT file never carries a score.
    assert all("score" not in o for o in data["objects"])


# ── pred detect round-trip ───────────────────────────────────────────────────


def test_detect_pred_round_trip_confidence_via_score(tmp_path: Path) -> None:
    path = tmp_path / "detect" / "IMG_0002.json"
    preds = [
        PredBBox(10.0, 20.0, 110.0, 220.0, 0, confidence=0.875),
        PredBBox(1.0, 2.0, 3.0, 4.0, 2, confidence=0.5),
    ]
    write_detect(path, preds, 640, 480)

    data = _raw(path)
    assert [o["score"] for o in data["objects"]] == [0.875, 0.5]

    got, class_ids = read_detect_pred(path)
    assert class_ids == {0, 2}
    assert [(b.x1, b.y1, b.x2, b.y2, b.class_id, b.confidence) for b in got] == [
        (10.0, 20.0, 110.0, 220.0, 0, 0.875),
        (1.0, 2.0, 3.0, 4.0, 2, 0.5),
    ]


# ── GT + pred segment round-trip ─────────────────────────────────────────────


def test_segment_gt_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "segment" / "IMG_0003.json"
    write_segment(path, [Polygon(SQUARE, 1), Polygon(TRIANGLE, 0)], 640, 480)

    data = _raw(path)
    # segmentation is [[flat pixel coords]] with >= 3 points.
    assert data["objects"][0]["segmentation"] == [[10.0, 20.0, 110.0, 20.0, 110.0, 220.0, 10.0, 220.0]]
    assert data["objects"][1]["segmentation"] == [[0.5, 0.25, 30.0, 0.25, 15.25, 40.75]]
    assert all("score" not in o for o in data["objects"])

    got, class_ids = read_segment(path)
    assert class_ids == {0, 1}
    assert [(p.points, p.class_id) for p in got] == [(SQUARE, 1), (TRIANGLE, 0)]


def test_segment_pred_round_trip_confidence_via_score(tmp_path: Path) -> None:
    path = tmp_path / "segment" / "IMG_0004.json"
    write_segment(path, [PredPolygon(TRIANGLE, 2, confidence=0.75)], 640, 480)

    assert _raw(path)["objects"][0]["score"] == 0.75

    got, class_ids = read_segment_pred(path)
    assert class_ids == {2}
    assert got[0].points == TRIANGLE
    assert got[0].class_id == 2
    assert got[0].confidence == 0.75


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


def test_provenance_round_trip_detect_gt_and_pred(tmp_path: Path) -> None:
    gt_path = tmp_path / "detect" / "gt.json"
    write_detect(gt_path, [BBox(1.0, 2.0, 3.0, 4.0, 0, **PROV)], 100, 100)
    (gt_box,), _ = read_detect(gt_path)
    _assert_prov(gt_box)

    pred_path = tmp_path / "detect" / "pred.json"
    write_detect(pred_path, [PredBBox(1.0, 2.0, 3.0, 4.0, 0, confidence=0.5, **PROV)], 100, 100)
    (pred_box,), _ = read_detect_pred(pred_path)
    _assert_prov(pred_box)
    assert pred_box.confidence == 0.5


def test_provenance_round_trip_segment_gt_and_pred(tmp_path: Path) -> None:
    gt_path = tmp_path / "segment" / "gt.json"
    write_segment(gt_path, [Polygon(TRIANGLE, 1, **PROV)], 100, 100)
    (gt_poly,), _ = read_segment(gt_path)
    _assert_prov(gt_poly)

    pred_path = tmp_path / "segment" / "pred.json"
    write_segment(pred_path, [PredPolygon(TRIANGLE, 1, confidence=0.875, **PROV)], 100, 100)
    (pred_poly,), _ = read_segment_pred(pred_path)
    _assert_prov(pred_poly)
    assert pred_poly.confidence == 0.875


def test_unset_provenance_omitted_from_json_not_null(tmp_path: Path) -> None:
    dpath = tmp_path / "detect" / "a.json"
    write_detect(dpath, [BBox(1.0, 2.0, 3.0, 4.0, 0)], 100, 100)
    spath = tmp_path / "segment" / "a.json"
    write_segment(spath, [Polygon(TRIANGLE, 0)], 100, 100)
    for path in (dpath, spath):
        obj = _raw(path)["objects"][0]
        for k in ("created_by", "created_at", "accepted_by", "accepted_at"):
            assert k not in obj  # omitted entirely, never written as null


def test_partial_provenance_writes_only_set_fields(tmp_path: Path) -> None:
    path = tmp_path / "detect" / "a.json"
    write_detect(path, [BBox(1.0, 2.0, 3.0, 4.0, 0, created_by="claude")], 100, 100)
    obj = _raw(path)["objects"][0]
    assert obj["created_by"] == "claude"
    for k in ("created_at", "accepted_by", "accepted_at"):
        assert k not in obj
    (box,), _ = read_detect(path)
    assert box.created_by == "claude"
    assert box.created_at is None and box.accepted_by is None and box.accepted_at is None


def test_provenance_set_by_mutation_survives_write(tmp_path: Path) -> None:
    path = tmp_path / "detect" / "a.json"
    write_detect(path, [PredBBox(1.0, 2.0, 3.0, 4.0, 0, confidence=0.5)], 100, 100)
    (pb,), _ = read_detect_pred(path)
    pb.created_by = "sam"  # mutating a parsed shape is the documented pattern
    write_detect(path, [pb], 100, 100)
    (again,), _ = read_detect_pred(path)
    assert again.created_by == "sam"
    assert again.confidence == 0.5


# ── negative invariant: present-empty == confirmed negative, missing == unannotated ──


def test_keep_empty_writes_present_confirmed_negative_detect(tmp_path: Path) -> None:
    path = tmp_path / "detect" / "IMG_0005.json"
    write_detect(path, [], 640, 480, keep_empty=True)

    assert os.path.exists(path)  # present file == confirmed negative
    assert _raw(path)["objects"] == []
    assert read_detect(path) == ([], set())
    assert read_detect_pred(path) == ([], set())


def test_keep_empty_writes_present_confirmed_negative_segment(tmp_path: Path) -> None:
    path = tmp_path / "segment" / "IMG_0005.json"
    write_segment(path, [], 640, 480, keep_empty=True)

    assert os.path.exists(path)
    assert _raw(path)["objects"] == []
    assert read_segment(path) == ([], set())
    assert read_segment_pred(path) == ([], set())


def test_empty_write_without_keep_empty_removes_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "detect" / "IMG_0006.json"
    write_detect(path, [BBox(1.0, 2.0, 3.0, 4.0, 0)], 640, 480)
    assert os.path.exists(path)

    write_detect(path, [], 640, 480, keep_empty=False)
    assert not os.path.exists(path)  # back to unannotated
    assert read_detect(path) == ([], set())

    spath = tmp_path / "segment" / "IMG_0006.json"
    write_segment(spath, [Polygon(TRIANGLE, 0)], 640, 480)
    write_segment(spath, [], 640, 480, keep_empty=False)
    assert not os.path.exists(spath)


def test_empty_write_without_keep_empty_on_missing_file_is_noop(tmp_path: Path) -> None:
    path = tmp_path / "detect" / "never_written.json"
    write_detect(path, [], 640, 480, keep_empty=False)  # must not raise
    assert not os.path.exists(path)


def test_missing_file_reads_empty_from_all_readers(tmp_path: Path) -> None:
    path = tmp_path / "nope" / "missing.json"
    assert not os.path.exists(path)
    for reader in (read_detect, read_detect_pred, read_segment, read_segment_pred):
        assert reader(path) == ([], set())


def test_present_negative_is_distinct_from_missing(tmp_path: Path) -> None:
    negative = tmp_path / "detect" / "confirmed_negative.json"
    missing = tmp_path / "detect" / "unannotated.json"
    write_detect(negative, [], 640, 480, keep_empty=True)

    # Both read as empty, but only the confirmed negative exists on disk.
    assert read_detect(negative) == ([], set())
    assert read_detect(missing) == ([], set())
    assert os.path.exists(negative)
    assert not os.path.exists(missing)


# ── robustness: malformed inputs are skipped/empty, never raise ──────────────

ALL_READERS = (read_detect, read_detect_pred, read_segment, read_segment_pred)


def test_non_json_file_reads_empty(tmp_path: Path) -> None:
    path = tmp_path / "garbage.json"
    path.write_text("not json {][", encoding="utf-8")
    for reader in ALL_READERS:
        assert reader(path) == ([], set())


def test_json_that_is_not_a_dict_reads_empty(tmp_path: Path) -> None:
    for i, payload in enumerate(('[1, 2, 3]', '"a string"', "42", "null")):
        path = tmp_path / f"nondict_{i}.json"
        path.write_text(payload, encoding="utf-8")
        for reader in ALL_READERS:
            assert reader(path) == ([], set())


def test_object_missing_geometry_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    payload = {
        "image": "a", "width": 100, "height": 100,
        "objects": [
            {"category_id": 1},  # no bbox and no segmentation
            {"category_id": 0, "bbox": [1.0, 2.0, 3.0, 4.0]},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    boxes, class_ids = read_detect(path)
    assert len(boxes) == 1 and class_ids == {0}  # only the valid object survives
    assert read_segment(path) == ([], set())  # no object has a usable segmentation


def test_bbox_wrong_shape_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    payload = {
        "image": "a", "width": 100, "height": 100,
        "objects": [
            {"category_id": 0, "bbox": [1.0, 2.0, 3.0]},          # wrong length
            {"category_id": 0, "bbox": [1, 2, 3, 4, 5]},          # wrong length
            {"category_id": 0, "bbox": "10,20,30,40"},            # not a list
            {"category_id": 0, "bbox": ["a", "b", "c", "d"]},     # non-numeric
            {"category_id": "x", "bbox": [1.0, 2.0, 3.0, 4.0]},   # non-numeric class
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert read_detect(path) == ([], set())
    assert read_detect_pred(path) == ([], set())


def test_bad_segmentation_rings_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    payload = {
        "image": "a", "width": 100, "height": 100,
        "objects": [
            {"category_id": 0, "segmentation": [[1.0, 2.0, 3.0, 4.0]]},        # < 3 points
            {"category_id": 0, "segmentation": [[1, 2, 3, 4, 5, 6, 7]]},       # odd coord count
            {"category_id": 0, "segmentation": {"counts": "RLE", "size": [2, 2]}},  # RLE dict
            {"category_id": 0, "segmentation": []},                             # empty
            {"category_id": 0, "segmentation": [["a", "b", "c", "d", "e", "f"]]},  # non-numeric
            {"category_id": 0},                                                  # absent
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert read_segment(path) == ([], set())
    assert read_segment_pred(path) == ([], set())


def test_objects_null_or_absent_reads_empty(tmp_path: Path) -> None:
    for i, payload in enumerate(('{"image": "a", "objects": null}', '{"image": "a"}')):
        path = tmp_path / f"empty_{i}.json"
        path.write_text(payload, encoding="utf-8")
        for reader in ALL_READERS:
            assert reader(path) == ([], set())


def test_write_segment_skips_degenerate_polygon(tmp_path: Path) -> None:
    # A <3-point polygon is not a shape; the writer must skip it so it can't be written as an object
    # every reader then silently drops (which would masquerade as a confirmed negative).
    path = tmp_path / "segment" / "a.json"
    write_segment(path, [Polygon([(1.0, 1.0), (2.0, 2.0)], 0), Polygon(TRIANGLE, 1)], 100, 100)
    got, class_ids = read_segment(path)
    assert [(p.points, p.class_id) for p in got] == [(TRIANGLE, 1)]  # only the valid polygon
    assert class_ids == {1}
    # write<->read symmetry: what a reader can't read, a writer must not write.
    assert all(len(o["segmentation"][0]) >= 6 for o in _raw(path)["objects"])

    # A list of ONLY degenerate polygons yields no objects -> removed (unannotated), not a bogus file.
    only_bad = tmp_path / "segment" / "b.json"
    write_segment(only_bad, [Polygon([(1.0, 1.0), (2.0, 2.0)], 0)], 100, 100)
    assert not os.path.exists(only_bad)


def test_readers_tolerate_non_dict_objects(tmp_path: Path) -> None:
    payload = {
        "image": "a",
        "objects": [
            1, "x", None, [1, 2],  # junk entries that must not raise
            {"category_id": 0, "bbox": [1.0, 2.0, 3.0, 4.0]},
            {"category_id": 1, "segmentation": [[0.0, 0.0, 9.0, 0.0, 5.0, 9.0]]},
        ],
    }
    path = tmp_path / "a.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert len(read_detect(path)[0]) == 1  # only the dict-with-bbox
    assert len(read_segment(path)[0]) == 1  # only the dict-with-segmentation
    for reader in ALL_READERS:
        reader(path)  # never raises


def test_null_or_bad_score_reads_as_zero(tmp_path: Path) -> None:
    payload = {
        "image": "a",
        "objects": [
            {"category_id": 1, "bbox": [1.0, 2.0, 3.0, 4.0], "score": None},
            {"category_id": 2, "segmentation": [[0.0, 0.0, 9.0, 0.0, 5.0, 9.0]], "score": "high"},
        ],
    }
    path = tmp_path / "a.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    (pb,), _ = read_detect_pred(path)
    assert pb.confidence == 0.0  # null score -> 0.0, object not dropped
    (pp,), _ = read_segment_pred(path)
    assert pp.confidence == 0.0  # non-numeric score -> 0.0, no raise


def test_non_finite_score_is_written_as_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "detect" / "a.json"
    write_detect(path, [PredBBox(1.0, 2.0, 3.0, 4.0, 0, confidence=float("nan"))], 100, 100)
    # File must be STRICT-valid JSON (no bare NaN literal); the non-finite score collapses to 0.0.
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["objects"][0]["score"] == 0.0
    (pb,), _ = read_detect_pred(path)
    assert pb.confidence == 0.0


# ── to_coco_dataset ──────────────────────────────────────────────────────────


def _mixed_entries(tmp_path: Path) -> list[tuple[str, str]]:
    """Detect (GT + pred w/ provenance), segment (pred), confirmed negative, missing."""
    d = tmp_path / "detect" / "IMG_0001.json"
    write_detect(
        d,
        [BBox(10.0, 20.0, 110.0, 220.0, 0),
         PredBBox(5.0, 5.0, 15.0, 25.0, 1, confidence=0.875, **PROV)],
        640, 480,
    )
    s = tmp_path / "segment" / "IMG_0002.json"
    write_segment(s, [PredPolygon(SQUARE, 1, confidence=0.5, created_by="sam")], 800, 600)
    neg = tmp_path / "detect" / "IMG_0003.json"
    write_detect(neg, [], 640, 480, keep_empty=True)
    missing = tmp_path / "detect" / "IMG_0004.json"
    return [(str(d), "IMG_0001.JPG"), (str(s), "IMG_0002.JPG"),
            (str(neg), "IMG_0003.JPG"), (str(missing), "IMG_0004.JPG")]


def test_to_coco_dataset_assembles_mixed_entries(tmp_path: Path) -> None:
    cats = [{"id": 0, "name": "catkin"}, {"id": 1, "name": "leaf"}]
    coco = to_coco_dataset(_mixed_entries(tmp_path), categories=cats,
                           confirmed_negative_names={"IMG_0003.JPG"})

    assert coco["categories"] == cats
    # Present files yield an images record — including IMG_0003's empty file because a human
    # CONFIRMED it negative (an empty file alone never trains as a negative). The missing
    # (unannotated) IMG_0004 is skipped entirely.
    assert [i["file_name"] for i in coco["images"]] == [
        "IMG_0001.JPG", "IMG_0002.JPG", "IMG_0003.JPG"]
    assert [i["id"] for i in coco["images"]] == [1, 2, 3]
    assert coco["images"][0]["width"] == 640 and coco["images"][0]["height"] == 480
    assert coco["images"][1]["width"] == 800 and coco["images"][1]["height"] == 600

    anns = coco["annotations"]
    assert len(anns) == 3  # 2 detect + 1 segment; the negative contributes none
    assert [a["id"] for a in anns] == [1, 2, 3]
    assert {a["image_id"] for a in anns} == {1, 2}
    for a in anns:
        assert a["iscrowd"] == 0
        assert isinstance(a["category_id"], int)
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


def test_to_coco_dataset_round_trips_through_format_io_parsers(tmp_path: Path) -> None:
    coco = to_coco_dataset(_mixed_entries(tmp_path))

    boxes, class_ids = parse_coco_detect(coco, image_id=1)
    assert class_ids == {0, 1}
    assert [(b.x1, b.y1, b.x2, b.y2, b.class_id) for b in boxes] == [
        (10.0, 20.0, 110.0, 220.0, 0),
        (5.0, 5.0, 15.0, 25.0, 1),
    ]

    polys, seg_class_ids = parse_coco_segment(coco, file_name="IMG_0002.JPG")
    assert seg_class_ids == {1}
    assert [(p.points, p.class_id) for p in polys] == [(SQUARE, 1)]

    # Detect-only annotations have no segmentation; the confirmed negative (id 3) has nothing at all;
    # the missing image was skipped, so there is no id-4 record.
    assert parse_coco_segment(coco, image_id=1) == ([], set())
    assert parse_coco_detect(coco, image_id=3) == ([], set())
    assert "IMG_0004.JPG" not in {i["file_name"] for i in coco["images"]}


def test_to_coco_dataset_defaults(tmp_path: Path) -> None:
    coco = to_coco_dataset([])
    assert coco == {"images": [], "annotations": [], "categories": []}
