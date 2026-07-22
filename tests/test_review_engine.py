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
    assert engine.shard_dir == tmp_path / "review"
    assert not engine.is_image_reviewed("IMG_0133.JPG")
    assert engine.get_image_review_status("IMG_0133.JPG") == "not_started"


def test_persistence_round_trip(tmp_path: Path) -> None:
    eng1 = ReviewEngine(tmp_path)
    eng1.mark_image_reviewed("IMG_0001.JPG")
    eng2 = ReviewEngine(tmp_path)
    assert eng2.is_image_reviewed("IMG_0001.JPG")


def test_save_review_state_is_compact_json(engine: ReviewEngine, tmp_path: Path) -> None:
    # No indent/whitespace: a shard is rewritten whole on every verdict to that image, so
    # compact serialization roughly halves the bytes serialized and written per save.
    engine.mark_image_reviewed("IMG_0133.JPG")
    raw = (tmp_path / "review" / "IMG_0133.JPG.json").read_text(encoding="utf-8")
    assert "\n" not in raw and "  " not in raw
    assert engine.raw_state["image"]["IMG_0133.JPG"]["img_status"] == "completed"


def test_verdict_writes_only_its_own_shard(engine: ReviewEngine, ctx: ReviewContext, tmp_path: Path) -> None:
    """A verdict on one image must not touch another image's shard file (O(dets on this
    image), not O(all-reviewed))."""
    engine.mark_image_reviewed("IMG_OTHER.JPG")
    other_shard = tmp_path / "review" / "IMG_OTHER.JPG.json"
    before = other_shard.stat().st_mtime_ns

    matches = compute_matches(
        ctx.gt_boxes, ctx.gt_polygons, ctx.pred_boxes, ctx.pred_polygons,
        iou_threshold=0.5, conf_threshold=0.25,
    )
    dets = engine.build_detection_list(ctx, matches)
    engine.record_detection_action(dets[0], ctx, action="accepted")

    assert (tmp_path / "review" / "IMG_0133.JPG.json").is_file()
    assert other_shard.stat().st_mtime_ns == before  # untouched


def test_verdict_calls_shard_writer_exactly_once(
    engine: ReviewEngine, ctx: ReviewContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    orig = ReviewEngine._save_image

    def spy(self: ReviewEngine, img_name: str) -> None:
        calls.append(img_name)
        orig(self, img_name)

    monkeypatch.setattr(ReviewEngine, "_save_image", spy)

    matches = compute_matches(
        ctx.gt_boxes, ctx.gt_polygons, ctx.pred_boxes, ctx.pred_polygons,
        iou_threshold=0.5, conf_threshold=0.25,
    )
    dets = engine.build_detection_list(ctx, matches)
    engine.record_detection_action(dets[0], ctx, action="accepted")
    assert calls == [ctx.img_name]  # exactly one shard write, for the touched image only


def test_verdicts_across_images_produce_one_shard_each(engine: ReviewEngine, ctx: ReviewContext) -> None:
    matches = compute_matches(
        ctx.gt_boxes, ctx.gt_polygons, ctx.pred_boxes, ctx.pred_polygons,
        iou_threshold=0.5, conf_threshold=0.25,
    )
    dets = engine.build_detection_list(ctx, matches)
    engine.record_detection_action(dets[0], ctx, action="accepted")
    engine.mark_image_reviewed("IMG_0200.JPG")

    shards = sorted(p.name for p in engine.shard_dir.glob("*.json"))
    assert shards == ["IMG_0133.JPG.json", "IMG_0200.JPG.json"]
    # Each shard holds only its own image's data.
    assert engine.raw_state["image"]["IMG_0133.JPG"]["detections"]
    assert engine.raw_state["image"]["IMG_0200.JPG"]["detections"] == []


def test_raw_state_round_trips_through_reload(engine: ReviewEngine, ctx: ReviewContext, tmp_path: Path) -> None:
    matches = compute_matches(
        ctx.gt_boxes, ctx.gt_polygons, ctx.pred_boxes, ctx.pred_polygons,
        iou_threshold=0.5, conf_threshold=0.25,
    )
    dets = engine.build_detection_list(ctx, matches)
    for det in dets:
        engine.record_detection_action(det, ctx, action="accepted")
    before = engine.raw_state

    reloaded = ReviewEngine(state_dir=tmp_path)
    assert reloaded.raw_state == before
def test_shard_key_with_separator_round_trips(tmp_path: Path) -> None:
    """A key bearing a path separator survives a save/reload without mutation (the shard filename
    is sanitized, but the true key is read back from the payload)."""
    eng = ReviewEngine(tmp_path)
    eng.mark_image_reviewed("sub/img.jpg")
    before = eng.raw_state

    reloaded = ReviewEngine(state_dir=tmp_path)
    assert reloaded.raw_state == before
    assert reloaded.is_image_reviewed("sub/img.jpg")


def test_shard_keys_colliding_after_sanitization_stay_distinct(tmp_path: Path) -> None:
    """'a/b.jpg' and 'a_b.jpg' sanitize to the same base but must not share a shard or merge state."""
    eng = ReviewEngine(tmp_path)
    eng.mark_image_reviewed("a/b.jpg")       # completed
    eng.mark_image_reviewed("a_b.jpg")
    eng.unmark_image_reviewed("a_b.jpg")     # -> not_started (no detections)
    assert len(list(eng.shard_dir.glob("*.json"))) == 2  # two distinct files, no clobber

    reloaded = ReviewEngine(state_dir=tmp_path)
    assert set(reloaded.raw_state["image"]) == {"a/b.jpg", "a_b.jpg"}  # keys preserved, not merged
    assert reloaded.is_image_reviewed("a/b.jpg")
    assert not reloaded.is_image_reviewed("a_b.jpg")


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


def test_build_detection_list_never_hides_reviewed(engine: ReviewEngine, ctx: ReviewContext) -> None:
    # Review status is image-level navigation, not per-detection visibility: reviewing a detection
    # must never drop it from the walkable list (else it can't be re-inspected or un-done).
    matches = compute_matches(
        ctx.gt_boxes, ctx.gt_polygons, ctx.pred_boxes, ctx.pred_polygons,
        iou_threshold=0.5, conf_threshold=0.25,
    )
    dets = engine.build_detection_list(ctx, matches)
    before = len(dets)
    engine.record_detection_action(dets[0], ctx, action="accepted")
    after = engine.build_detection_list(ctx, matches)
    assert len(after) == before  # the accepted detection is still walkable


def test_get_all_image_statuses(engine: ReviewEngine, ctx: ReviewContext) -> None:
    assert engine.get_all_image_statuses() == {}  # nothing touched yet
    matches = compute_matches(
        ctx.gt_boxes, ctx.gt_polygons, ctx.pred_boxes, ctx.pred_polygons,
        iou_threshold=0.5, conf_threshold=0.25,
    )
    dets = engine.build_detection_list(ctx, matches)
    engine.record_detection_action(dets[0], ctx, action="accepted")  # -> "started"
    engine.mark_image_reviewed("IMG_OTHER.JPG")  # -> "completed"
    statuses = engine.get_all_image_statuses()
    assert statuses[ctx.img_name] == "started"
    assert statuses["IMG_OTHER.JPG"] == "completed"


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
    orig = '{"objects": [{"category_id": 0, "bbox": [1, 1, 9, 9]}]}'
    (detect_dir / "IMG_0001.json").write_text(orig)
    assert engine.backup_original_labels(detect_dir) == 1
    backup = detect_dir / ".original" / "IMG_0001.json"
    assert backup.read_text() == orig

    # Mutate the label on disk; a later backup must not overwrite its baseline
    (detect_dir / "IMG_0001.json").write_text('{"objects": []}')
    assert engine.backup_original_labels(detect_dir) == 0
    assert backup.read_text() == orig  # still original

    # A label added after the first backup still gets its own baseline captured
    second = '{"objects": [{"category_id": 0, "bbox": [3, 3, 7, 7]}]}'
    (detect_dir / "IMG_0002.json").write_text(second)
    assert engine.backup_original_labels(detect_dir) == 1
    assert (detect_dir / ".original" / "IMG_0002.json").read_text() == second
    assert backup.read_text() == orig


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
