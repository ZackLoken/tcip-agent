"""Unit tests for tcip_annotation.review_engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation import (
    BBox,
    PredBBox,
    Polygon,
    ReviewContext,
    ReviewEngine,
    compute_matches,
)


@pytest.fixture
def engine(tmp_path: Path) -> ReviewEngine:
    return ReviewEngine(
        state_dir=tmp_path,
        class_names={0: "catkin", 1: "bud"},
        current_user="alice",
    )


@pytest.fixture
def ctx() -> ReviewContext:
    return ReviewContext(
        img_name="IMG_0133.JPG",
        img_width=1000,
        img_height=800,
        gt_boxes=[
            BBox(100, 100, 200, 200, class_id=0),
            BBox(400, 400, 500, 500, class_id=0),
        ],
        gt_polygons=[],
        pred_boxes=[
            PredBBox(105, 105, 205, 205, class_id=0, confidence=0.9),  # matches gt[0]
            PredBBox(700, 700, 800, 800, class_id=0, confidence=0.8),  # FP
        ],
        pred_polygons=[],
    )


def test_initial_state_is_empty(engine: ReviewEngine, tmp_path: Path) -> None:
    assert engine.raw_state == {}
    assert engine.review_state_path == tmp_path / "review_stats.json"
    assert not engine.is_image_reviewed("IMG_0133.JPG")
    assert engine.get_image_review_status("IMG_0133.JPG") == "not_started"


def test_persistence_round_trip(tmp_path: Path) -> None:
    eng1 = ReviewEngine(tmp_path)
    eng1.mark_image_reviewed("IMG_0001.JPG")
    eng2 = ReviewEngine(tmp_path)
    assert eng2.is_image_reviewed("IMG_0001.JPG")


def test_build_detection_list_tp_fp_fn(engine: ReviewEngine, ctx: ReviewContext) -> None:
    matches = compute_matches(
        gt_boxes=ctx.gt_boxes,
        gt_polygons=ctx.gt_polygons,
        pred_boxes=ctx.pred_boxes,
        pred_polygons=ctx.pred_polygons,
        iou_threshold=0.5,
        conf_threshold=0.25,
    )
    assert len(matches["tp"]) == 1
    assert len(matches["fp"]) == 1
    assert len(matches["fn"]) == 1

    dets = engine.build_detection_list(ctx, matches)
    types = sorted(d.det_type for d in dets)
    assert types == ["fn", "fp", "tp"]


def test_build_detection_list_filter_type(engine: ReviewEngine, ctx: ReviewContext) -> None:
    matches = compute_matches(
        ctx.gt_boxes, ctx.gt_polygons, ctx.pred_boxes, ctx.pred_polygons,
        iou_threshold=0.5, conf_threshold=0.25,
    )
    tp_only = engine.build_detection_list(ctx, matches, filter_type="tp")
    assert len(tp_only) == 1 and tp_only[0].det_type == "tp"

    fp_only = engine.build_detection_list(ctx, matches, filter_type="fp")
    assert len(fp_only) == 1 and fp_only[0].det_type == "fp"


def test_build_detection_list_filter_class(engine: ReviewEngine, ctx: ReviewContext) -> None:
    matches = compute_matches(
        ctx.gt_boxes, ctx.gt_polygons, ctx.pred_boxes, ctx.pred_polygons,
        iou_threshold=0.5, conf_threshold=0.25,
    )
    all_class_0 = engine.build_detection_list(ctx, matches, filter_class=0)
    assert all(d.class_id == 0 for d in all_class_0)
    none = engine.build_detection_list(ctx, matches, filter_class=99)
    assert none == []


def test_record_and_find_reviewed(engine: ReviewEngine, ctx: ReviewContext) -> None:
    matches = compute_matches(
        ctx.gt_boxes, ctx.gt_polygons, ctx.pred_boxes, ctx.pred_polygons,
        iou_threshold=0.5, conf_threshold=0.25,
    )
    dets = engine.build_detection_list(ctx, matches)
    target = next(d for d in dets if d.det_type == "tp")

    # Before recording: not found
    assert engine.find_reviewed_entry(target, ctx) is None

    engine.record_detection_action(target, ctx, action="accepted")
    entry = engine.find_reviewed_entry(target, ctx)
    assert entry is not None
    assert entry["action"] == "accepted"
    assert entry["match_type"] == "TP"
    assert entry["reviewed_by"] == "alice"
    assert entry["class_name"] == "catkin"


def test_record_overrides_existing(engine: ReviewEngine, ctx: ReviewContext) -> None:
    matches = compute_matches(
        ctx.gt_boxes, ctx.gt_polygons, ctx.pred_boxes, ctx.pred_polygons,
        iou_threshold=0.5, conf_threshold=0.25,
    )
    dets = engine.build_detection_list(ctx, matches)
    target = dets[0]
    engine.record_detection_action(target, ctx, action="accepted")
    engine.record_detection_action(target, ctx, action="rejected")
    entry = engine.find_reviewed_entry(target, ctx)
    assert entry is not None
    assert entry["action"] == "rejected"
    # Should still be exactly one entry, not two
    img_data = engine.raw_state["image"][ctx.img_name]
    assert len(img_data["detections"]) == 1


def test_status_filter_isolates_reviewed(engine: ReviewEngine, ctx: ReviewContext) -> None:
    matches = compute_matches(
        ctx.gt_boxes, ctx.gt_polygons, ctx.pred_boxes, ctx.pred_polygons,
        iou_threshold=0.5, conf_threshold=0.25,
    )
    dets = engine.build_detection_list(ctx, matches)
    engine.record_detection_action(dets[0], ctx, action="accepted")

    reviewed = engine.build_detection_list(ctx, matches, status_filter="reviewed")
    not_reviewed = engine.build_detection_list(ctx, matches, status_filter="not_reviewed")
    assert len(reviewed) == 1
    assert len(not_reviewed) == 2


def test_check_image_review_complete(engine: ReviewEngine, ctx: ReviewContext) -> None:
    matches = compute_matches(
        ctx.gt_boxes, ctx.gt_polygons, ctx.pred_boxes, ctx.pred_polygons,
        iou_threshold=0.5, conf_threshold=0.25,
    )
    dets = engine.build_detection_list(ctx, matches)
    # Accept each detection one at a time
    for det in dets[:-1]:
        engine.record_detection_action(det, ctx, action="accepted")
    assert engine.check_image_review_complete(ctx.img_name, matches) is False

    engine.record_detection_action(dets[-1], ctx, action="accepted")
    assert engine.check_image_review_complete(ctx.img_name, matches) is True
    assert engine.is_image_reviewed(ctx.img_name)


def test_backup_original_labels_per_file(engine: ReviewEngine, tmp_path: Path) -> None:
    detect_dir = tmp_path / "detect"
    detect_dir.mkdir()
    (detect_dir / "IMG_0001.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    assert engine.backup_original_labels(detect_dir) == 1
    backup = detect_dir / ".original" / "IMG_0001.txt"
    assert backup.read_text() == "0 0.5 0.5 0.1 0.1\n"

    # Mutate the label on disk; a later backup must not overwrite its baseline
    (detect_dir / "IMG_0001.txt").write_text("1 0.5 0.5 0.1 0.1\n")
    assert engine.backup_original_labels(detect_dir) == 0
    assert backup.read_text() == "0 0.5 0.5 0.1 0.1\n"  # still original

    # A label added after the first backup still gets its own baseline captured
    (detect_dir / "IMG_0002.txt").write_text("0 0.3 0.3 0.1 0.1\n")
    assert engine.backup_original_labels(detect_dir) == 1
    assert (detect_dir / ".original" / "IMG_0002.txt").read_text() == "0 0.3 0.3 0.1 0.1\n"
    assert backup.read_text() == "0 0.5 0.5 0.1 0.1\n"


def test_save_gt_writes_both_files(engine: ReviewEngine, ctx: ReviewContext, tmp_path: Path) -> None:
    from tcip_annotation.json_io import read_detect, read_segment

    ctx.gt_boxes = [BBox(100, 100, 200, 200, class_id=0)]
    ctx.gt_polygons = [Polygon([(10.0, 10.0), (20.0, 10.0), (20.0, 20.0)], class_id=1)]
    det_path = tmp_path / "out_detect" / "IMG.json"
    seg_path = tmp_path / "out_segment" / "IMG.json"
    ok = engine.save_gt(ctx, detect_path=str(det_path), segment_path=str(seg_path))
    assert ok
    det_boxes, _ = read_detect(str(det_path))
    assert len(det_boxes) == 1 and det_boxes[0].class_id == 0
    seg_polys, _ = read_segment(str(seg_path))
    assert len(seg_polys) == 1 and seg_polys[0].class_id == 1
