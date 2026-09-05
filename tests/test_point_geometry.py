"""``state.Point`` is a third annotation geometry, and every consumer decides about it.

A Point is a placed prompt (SAM-style) or a keypoint/landmark: a real annotation with a location, but
never a detection/segmentation target. It has no box and no area, so ``bbox_of`` refuses one rather
than fabricate a degenerate zero-area box that would read downstream as a real object. That refusal is
the backstop, not the guard: these tests pin the two behaviours that widening the union demands of
every consumer:

  * a training-target / IoU-matching / delivery-grade path skips a Point cleanly (never crashes on
    ``bbox_of``, never emits a fabricated extent), while still doing its normal job for the boxes and
    polygons alongside it; and
  * a serializer / write path represents a Point (the ``"point": [x, y]`` key, symmetric on read and
    write) instead of silently dropping the only thing that annotation says.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.matching import compute_matches
from tcip_annotation.state import Annotation, BBox, Point, Polygon, bbox_of

BOX = BBox(10.0, 10.0, 30.0, 30.0)
RING = [(50.0, 50.0), (70.0, 50.0), (70.0, 70.0)]


def _img(tmp_path: Path, name: str = "IMG_0001.JPG", size: tuple[int, int] = (100, 80)) -> Path:
    p = tmp_path / "images" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(p)
    return p


# ── the geometry itself ──────────────────────────────────────────────────────


def test_bbox_of_refuses_a_point_and_says_why() -> None:
    with pytest.raises(ValueError) as exc:
        bbox_of(Point(5.0, 6.0))
    msg = str(exc.value)
    assert "Point" in msg and "no bounding box" in msg


def test_bbox_of_still_reads_a_box_and_a_polygon() -> None:
    """The refusal must not have narrowed what bbox_of legitimately answers."""
    assert bbox_of(BOX) is BOX
    b = bbox_of(Polygon([RING]))
    assert (b.x1, b.y1, b.x2, b.y2) == (50.0, 50.0, 70.0, 70.0)


# ── on-disk round trip (json_io) ─────────────────────────────────────────────


def test_point_round_trips_through_the_per_image_json(tmp_path: Path) -> None:
    path = tmp_path / "IMG_0001.json"
    json_io.write_annotations(
        path, [Annotation(subject="bud", geometry=Point(12.5, 34.25))], 100, 80)

    raw = json.loads(path.read_text(encoding="utf-8"))
    (rec,) = raw["annotations"]
    assert rec["point"] == [12.5, 34.25]
    assert "bbox" not in rec  # no fabricated box travels with a point

    (back,) = json_io.read_annotations(path)
    assert isinstance(back.geometry, Point)
    assert (back.geometry.x, back.geometry.y) == (12.5, 34.25)


def test_a_point_alongside_a_box_and_a_polygon_all_survive_one_file(tmp_path: Path) -> None:
    path = tmp_path / "IMG_0002.json"
    json_io.write_annotations(path, [
        Annotation(subject="bud", geometry=BOX),
        Annotation(subject="bud", geometry=Polygon([RING])),
        Annotation(subject="bud", geometry=Point(1.0, 2.0)),
        Annotation(subject="bud"),  # image-level label, no geometry
    ], 100, 80)
    kinds = [type(a.geometry) for a in json_io.read_annotations(path)]
    assert kinds == [BBox, Polygon, Point, type(None)]


# ── target membership (the one shared decision) ──────────────────────────────


def test_target_class_id_returns_none_for_a_point_without_raising() -> None:
    a = Annotation(subject="bud", geometry=Point(1.0, 2.0))
    assert json_io.target_class_id(a, "bud", None, {"bud": 0}) is None
    # An attribute scope must not turn the point into a decode failure either: it is simply not a
    # target for this scope, which is a different thing from "a target the registry can't decode".
    assert json_io.target_class_id(a, "bud", "opening", {"open": 0}) is None


def test_target_class_id_still_assigns_a_box_its_class() -> None:
    a = Annotation(subject="bud", geometry=BOX)
    assert json_io.target_class_id(a, "bud", None, {"bud": 0}) == 0


# ── COCO assembly for training ───────────────────────────────────────────────


def _entries(tmp_path: Path, anns: list[Annotation], name: str = "IMG_0001.JPG"):
    label = tmp_path / f"{Path(name).stem}.json"
    json_io.write_annotations(label, anns, 100, 80)
    return [(str(label), name)]


def test_to_coco_dataset_counts_a_point_only_image_but_emits_no_annotation(tmp_path: Path) -> None:
    """A point is real content (the image is annotated, not an empty negative) yet has no COCO
    record: a zero-area box in ``annotations`` would train as an object."""
    coco = json_io.to_coco_dataset(
        _entries(tmp_path, [Annotation(subject="bud", geometry=Point(20.0, 20.0))]),
        subject="bud", id_map={"bud": 0})
    assert [i["file_name"] for i in coco["images"]] == ["IMG_0001.JPG"]
    assert coco["annotations"] == []


def test_to_coco_dataset_keeps_the_box_next_to_the_point(tmp_path: Path) -> None:
    coco = json_io.to_coco_dataset(
        _entries(tmp_path, [
            Annotation(subject="bud", geometry=Point(20.0, 20.0)),
            Annotation(subject="bud", geometry=BOX),
        ]),
        subject="bud", id_map={"bud": 0})
    (rec,) = coco["annotations"]
    assert rec["bbox"] == [10.0, 10.0, 20.0, 20.0]
    assert rec["area"] == 400.0


def test_write_coco_interop_export_skips_a_point(tmp_path: Path) -> None:
    from tcip_annotation.format_io import write_coco

    out = tmp_path / "dataset.json"
    write_coco(str(out), {"IMG_0001.JPG": (
        [Annotation(subject="bud", geometry=Point(20.0, 20.0)),
         Annotation(subject="bud", geometry=BOX)], 100, 80)})
    coco = json.loads(out.read_text(encoding="utf-8"))
    (rec,) = coco["annotations"]  # the box only; COCO has no honest record for a point
    assert rec["bbox"] == [10.0, 10.0, 20.0, 20.0]


# ── the loader's own per-image target read ───────────────────────────────────


def test_json_det_targets_yields_no_box_for_a_point(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.label_queries import json_det_targets

    label = tmp_path / "IMG_0001.json"
    json_io.write_annotations(label, [
        Annotation(subject="bud", geometry=Point(20.0, 20.0)),
        Annotation(subject="bud", geometry=BOX),
    ], 100, 80)
    boxes, labels, n_unlabeled = json_det_targets(str(label), "bud", None, {"bud": 0})
    assert boxes == [[10.0, 10.0, 30.0, 30.0]]
    assert labels == [1]
    assert n_unlabeled == 0  # a point is not an unlabeled instance either: it is not an instance


def test_a_point_only_image_is_not_a_trainable_sample(tmp_path: Path) -> None:
    """``_label_record_state``'s ``has_objects`` is target membership, not mere annotatedness: a
    point-only image kept on the direct-json path would train as a zero-object negative no human
    confirmed: the exact fabrication the function's docstring exists to prevent."""
    from tcip_mcp.pipelines.data.label_queries import _label_record_state

    labels = tmp_path / "annotations"
    labels.mkdir()
    json_io.write_annotations(labels / "IMG_0001.json",
                              [Annotation(subject="bud", geometry=Point(20.0, 20.0))], 100, 80)
    json_io.write_annotations(labels / "IMG_0002.json",
                              [Annotation(subject="bud", geometry=BOX)], 100, 80)

    assert _label_record_state("IMG_0001", labels, "bud") == (True, False)
    assert _label_record_state("IMG_0002", labels, "bud") == (True, True)


# ── IoU matching ─────────────────────────────────────────────────────────────


def test_compute_matches_ignores_a_point_on_either_side() -> None:
    gt = [Annotation(subject="bud", geometry=Point(20.0, 20.0))]
    preds = [Annotation(subject="bud", geometry=Point(20.0, 20.0), score=0.9)]
    m = compute_matches(gt, preds, iou_threshold=0.5, conf_threshold=0.1)
    # Not a TP (nothing overlapped), not an FP, not an FN: a point makes no spatial claim to score.
    assert (m["tp"], m["fp"], m["fn"]) == ([], [], [])


def test_compute_matches_still_matches_the_boxes_around_a_point() -> None:
    gt = [Annotation(subject="bud", geometry=Point(1.0, 1.0)),
          Annotation(subject="bud", geometry=BOX)]
    preds = [Annotation(subject="bud", geometry=Point(1.0, 1.0), score=0.9),
             Annotation(subject="bud", geometry=BOX, score=0.9)]
    m = compute_matches(gt, preds, iou_threshold=0.5, conf_threshold=0.1)
    assert len(m["tp"]) == 1 and not m["fp"] and not m["fn"]
    # The reported indices address the caller's own lists, so they must still point at the boxes.
    assert m["tp"][0]["gt_idx"] == 1 and m["tp"][0]["pred_idx"] == 1


# ── COCO scoring records ─────────────────────────────────────────────────────


def test_records_from_annotation_omits_a_point_and_its_category() -> None:
    from tcip_mcp.pipelines.training.evaluation import records_from_annotation

    iou_type, rec = records_from_annotation(
        [Annotation(subject="prompt", geometry=Point(5.0, 5.0)),
         Annotation(subject="bud", geometry=BOX)],
        [Annotation(subject="bud", geometry=BOX, score=0.8)],
        width=100, height=80)
    assert iou_type == "bbox"
    assert len(rec["gt"]) == 1 and len(rec["dt"]) == 1
    # 'prompt' minted no category, so the box keeps id 1 rather than being pushed to 2 by a subject
    # that contributes no record at all.
    assert rec["gt"][0]["category_id"] == 1


# ── triage heuristics / phenology counts ─────────────────────────────────────


def test_worst_predictions_does_not_count_a_point_as_a_detection(tmp_path: Path) -> None:
    from tcip_mcp.tools.vision_tools import get_worst_predictions

    gt_dir, pred_dir = tmp_path / "gt", tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    json_io.write_annotations(gt_dir / "IMG_0001.json",
                              [Annotation(subject="bud", geometry=BOX)], 100, 80)
    json_io.write_annotations(pred_dir / "IMG_0001.json", [
        Annotation(subject="bud", geometry=BOX, score=1.0),
        Annotation(subject="bud", geometry=Point(60.0, 60.0), score=1.0),
    ], 100, 80)

    res = get_worst_predictions(str(pred_dir), str(gt_dir))
    # 1 GT box vs 1 predicted box: no shortfall, no surplus, full confidence -> a zero error score.
    # Counting the point as a surplus prediction would score this perfect frame as wrong.
    assert res["worst_images"][0]["error_score"] == 0.0


def test_phenology_detection_counts_exclude_a_point(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.postprocessing.phenology import count_by_class
    from tcip_mcp.pipelines.resolution import BucketScope

    path = tmp_path / "IMG_0001.json"
    json_io.write_annotations(path, [
        Annotation(subject="bud", geometry=BOX, score=0.9,
                  attributes={"opening": "open"}),
        Annotation(subject="bud", geometry=Point(60.0, 60.0), score=0.9,
                  attributes={"opening": "open"}),
    ], 100, 80)
    scope = BucketScope(subject="bud", attribute="opening")
    total, positive, unclassified = count_by_class(
        path, {"open": 0, "closed": 1}, "open", scope=scope)
    assert (total, positive, unclassified) == (1, 1, 0)


# ── review engine: a verdict's box ───────────────────────────────────────────


def test_review_engine_reads_no_bbox_for_a_point(tmp_path: Path) -> None:
    from tcip_annotation import ReviewEngine

    eng = ReviewEngine(state_dir=tmp_path / "state")
    anns = [Annotation(subject="bud", geometry=Point(20.0, 20.0)),
            Annotation(subject="bud", geometry=BOX)]
    assert eng._bbox_of_annotation(anns, 0) is None  # like a geometry-less label, not a 0-area box
    assert eng._bbox_of_annotation(anns, 1) == (10.0, 10.0, 30.0, 30.0)


# ── spatial index (hit-testing, a different concern from bbox_of) ────────────


def test_annotation_engine_indexes_a_point_at_its_own_location(tmp_path: Path) -> None:
    from tcip_annotation import AnnotationEngine
    from tcip_annotation.state import AnnotationState

    state = AnnotationState(img_width=100, img_height=80)
    state.annotations = [Annotation(subject="bud", geometry=Point(20.0, 25.0))]
    AnnotationEngine(state).ensure_poly_bboxes()
    # A real hit-test cell at the point, not the (0,0,0,0) placeholder a geometry-less label gets.
    assert state._poly_bboxes == [(20.0, 25.0, 20.0, 25.0)]


# ── renderers ────────────────────────────────────────────────────────────────


def test_box_renderer_skips_a_point_and_discloses_the_skip() -> None:
    from tcip_mcp.tools.vision_tools import _boxable, _n_points, _point_note

    anns = [Annotation(subject="bud", geometry=Point(1.0, 1.0)),
            Annotation(subject="bud", geometry=BOX),
            Annotation(subject="bud")]
    assert _boxable(anns) == [anns[1]]
    assert _n_points(anns) == 1
    assert "not drawn" in _point_note(_n_points(anns))
    assert _point_note(0) == ""


def test_visualize_annotations_renders_the_box_and_reports_the_point(tmp_path: Path) -> None:
    from tcip_mcp.tools.vision_tools import _viz_annotations

    img = _img(tmp_path / "ds", "IMG_0001.JPG")
    label = tmp_path / "ds" / "annotations" / "IMG_0001.json"
    json_io.write_annotations(label, [
        Annotation(subject="bud", geometry=BOX),
        Annotation(subject="bud", geometry=Point(60.0, 60.0)),
    ], 100, 80)

    res = _viz_annotations(str(img), task="detect")
    assert "error" not in res  # the point did not crash the box renderer
    assert res["count"] == 1
    assert res["points_not_rendered"] == 1
    assert "point annotation(s) not drawn" in res["summary"]


# ── dict serializers (the read side of every agent/GUI surface) ──────────────


def test_mcp_ann_dict_emits_the_point_key() -> None:
    from tcip_mcp.tools.annotation_tools import _add_geom, _ann_dict

    d = _ann_dict(Annotation(subject="bud", geometry=Point(12.0, 34.0)))
    assert d["point"] == [12.0, 34.0]

    det: dict = {}
    _add_geom(det, Annotation(subject="bud", geometry=Point(12.0, 34.0)))
    assert det["point"] == [12.0, 34.0]


def test_mcp_read_annotations_tool_returns_a_point(tmp_path: Path) -> None:
    from tcip_mcp.tools.annotation_tools import read_annotations as read_annotations_tool

    img = _img(tmp_path / "ds", "IMG_0001.JPG")
    label = tmp_path / "ds" / "annotations" / "IMG_0001.json"
    json_io.write_annotations(label, [Annotation(subject="bud", geometry=Point(12.0, 34.0))],
                              100, 80)
    res = read_annotations_tool(str(img))
    (ann,) = res["labels"]["annotations"]
    assert ann["point"] == [12.0, 34.0]


def test_subject_task_names_a_point_only_frame(tmp_path: Path) -> None:
    """A point-only frame is annotated (a non-None task) but is neither 'detect' nor 'segment'."""
    from tcip_mcp.tools.gui_tools import _subject_task

    assert _subject_task([Annotation(subject="bud", geometry=Point(1.0, 1.0))], "bud") == "point"
    assert _subject_task([Annotation(subject="bud", geometry=BOX)], "bud") == "detect"
    assert _subject_task([Annotation(subject="bud", geometry=Polygon([RING]))], "bud") == "segment"
    assert _subject_task([Annotation(subject="bud")], "bud") is None


# ── the agent's own write door ────────────────────────────────────────────────


def test_save_annotations_tool_writes_an_incoming_point(tmp_path: Path) -> None:
    from tcip_mcp.tools.annotation_tools import save_annotations

    img = _img(tmp_path)
    out = tmp_path / "IMG_0001.json"
    res = save_annotations(str(img), annotations=[{"subject": "bud", "point": [12.0, 34.0]}],
                           path=str(out))
    assert "error" not in res
    (stored,) = json_io.read_annotations(out)
    assert isinstance(stored.geometry, Point)
    assert (stored.geometry.x, stored.geometry.y) == (12.0, 34.0)


def test_save_annotations_tool_keeps_points_and_point_distinct(tmp_path: Path) -> None:
    """``points`` is a polygon contour and ``point`` is one location: one spelling must not serve
    both, or a point and a one-vertex polygon become indistinguishable on disk."""
    from tcip_mcp.tools.annotation_tools import save_annotations

    img = _img(tmp_path)
    out = tmp_path / "IMG_0001.json"
    save_annotations(str(img), annotations=[
        {"subject": "bud", "points": [[50, 50], [70, 50], [70, 70]]},
        {"subject": "bud", "point": [12.0, 34.0]},
    ], path=str(out))
    kinds = [type(a.geometry) for a in json_io.read_annotations(out)]
    assert kinds == [Polygon, Point]


# ── web routes (the human's canvas) ──────────────────────────────────────────


@pytest.fixture
def client() -> TestClient:
    from tcip_web.app import app

    return TestClient(app, base_url="http://127.0.0.1")


def test_annotate_route_round_trips_a_point(client: TestClient, tmp_path: Path) -> None:
    img = _img(tmp_path)
    label = tmp_path / "labels" / "IMG_0001.json"

    resp = client.post("/api/annotate/labels", json={
        "image_path": str(img), "label_path": str(label),
        "annotations": [{"subject": "bud", "point": [12.0, 34.0]}],
    })
    assert resp.status_code == 200
    (stored,) = json_io.read_annotations(str(label))
    assert isinstance(stored.geometry, Point)

    body = client.get("/api/annotate/labels",
                      params={"image_path": str(img), "label_path": str(label)}).json()
    (ann,) = body["annotations"]
    assert ann["point"] == [12.0, 34.0]  # read back as itself, not as a geometry-less label


def test_annotate_route_round_trips_mixed_point_and_box_geometry(
    client: TestClient, tmp_path: Path
) -> None:
    img = _img(tmp_path)
    label = tmp_path / "labels" / "IMG_0001.json"
    resp = client.post("/api/annotate/labels", json={
        "image_path": str(img), "label_path": str(label),
        "annotations": [{"subject": "bud", "point": [12.0, 34.0]},
                        {"subject": "bud", "bbox": [10.0, 10.0, 30.0, 30.0]}],
    })
    assert resp.status_code == 200
    kinds = [type(a.geometry) for a in json_io.read_annotations(str(label))]
    assert kinds == [Point, BBox]


def test_review_matches_returns_a_point_gt_without_scoring_it(
    client: TestClient, tmp_path: Path
) -> None:
    img = _img(tmp_path)
    gt = tmp_path / "gt.json"
    pred = tmp_path / "pred.json"
    json_io.write_annotations(gt, [Annotation(subject="bud", geometry=Point(20.0, 20.0))], 100, 80)
    json_io.write_annotations(pred, [Annotation(subject="bud", geometry=BOX, score=0.9)], 100, 80)

    body = client.post("/api/review/matches", json={
        "dataset_root": str(tmp_path / "proj"),
        "image_name": "IMG_0001.JPG", "image_path": str(img),
        "gt_path": str(gt), "pred_path": str(pred),
        "iou_threshold": 0.3, "conf_threshold": 0.1,
    }).json()

    # The point is neither an FN nor a match; the box prediction is a plain FP against no GT.
    assert (body["n_tp"], body["n_fp"], body["n_fn"]) == (0, 1, 0)
    assert body["gt"][0]["point"] == [20.0, 20.0]  # still shown to the reviewer, with its location
