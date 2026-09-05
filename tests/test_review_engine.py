"""Unit tests for tcip_annotation.review_engine."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from tcip_annotation import (
    Annotation,
    BBox,
    Polygon,
    ReviewContext,
    ReviewDetection,
    ReviewEngine,
    compute_classified_trait_matches,
    compute_matches,
)
from tcip_annotation.review_engine import bucket_dirname, capture_label_baseline

# The prediction bucket these verdicts are recorded against, spelled the way
# prediction_buckets.bucket_key_of spells one: relative to the dataset root.
BUCKET = "predictions/baseline/2026-02-11"


@pytest.fixture
def engine(tmp_path: Path) -> ReviewEngine:
    return ReviewEngine(state_dir=tmp_path, current_user="alice")


@pytest.fixture
def bucket_dir(tmp_path: Path) -> Path:
    return tmp_path / "review" / bucket_dirname(BUCKET)


@pytest.fixture
def ctx() -> ReviewContext:
    return ReviewContext(
        img_name="IMG_0133.JPG",
        img_width=1000,
        img_height=800,
        gt=[
            Annotation(subject="bud", geometry=BBox(100, 100, 200, 200)),
            Annotation(subject="bud", geometry=BBox(400, 400, 500, 500)),
        ],
        preds=[
            Annotation(subject="bud", geometry=BBox(105, 105, 205, 205), score=0.9),  # matches gt[0]
            Annotation(subject="bud", geometry=BBox(700, 700, 800, 800), score=0.8),  # FP
        ],
    )


@pytest.fixture
def unordered_ctx() -> ReviewContext:
    """A non-square image whose GT and prediction lists are in different orders.

    Every box has a width unequal to its height and the two lists never pair a GT with the
    prediction at the same position, so an index transposition and a width normalized against
    the wrong image dimension both change the numbers rather than cancelling out.
    """
    return ReviewContext(
        img_name="IMG_0501.JPG",
        img_width=1200,
        img_height=500,
        gt=[
            Annotation(subject="bud", geometry=BBox(100, 100, 300, 200)),
            Annotation(subject="bud", geometry=BBox(700, 260, 900, 420)),
            Annotation(subject="bud", geometry=BBox(50, 400, 150, 460)),  # no prediction: an FN
        ],
        preds=[
            Annotation(subject="bud", geometry=BBox(704, 264, 904, 424), score=0.90),  # gt[1]
            Annotation(subject="bud", geometry=BBox(1000, 20, 1080, 100), score=0.80),  # an FP
            Annotation(subject="bud", geometry=BBox(104, 104, 304, 204), score=0.72),  # gt[0]
        ],
    )


def _unordered_matches(ctx: ReviewContext) -> dict:
    return compute_matches(ctx.gt, ctx.preds, iou_threshold=0.5, conf_threshold=0.25)


def test_initial_state_is_empty(engine: ReviewEngine, tmp_path: Path) -> None:
    assert engine.raw_state == {}
    assert engine.shard_dir == tmp_path / "review"
    assert not engine.is_image_reviewed(BUCKET, "IMG_0133.JPG")
    assert engine.get_image_review_status(BUCKET, "IMG_0133.JPG") == "not_started"


def test_persistence_round_trip(tmp_path: Path) -> None:
    eng1 = ReviewEngine(tmp_path)
    eng1.mark_image_reviewed(BUCKET, "IMG_0001.JPG")
    eng2 = ReviewEngine(tmp_path)
    assert eng2.is_image_reviewed(BUCKET, "IMG_0001.JPG")


def test_save_review_state_writes_the_canonical_record_spelling(
    engine: ReviewEngine, bucket_dir: Path
) -> None:
    """A shard is spelled the way every record is, so a breeder who opens one and a reader
    that parses one meet the same document.

    Bound to the file backend on purpose: the claim is about the exact bytes on disk, which only
    the file backend exposes as a file at all.
    """
    import tcip_store
    from tcip_store import RECORD_JSON
    from tcip_store.file_backend import FileBackend

    tcip_store.bind(FileBackend())
    engine.mark_image_reviewed(BUCKET, "IMG_0133.JPG")
    raw = (bucket_dir / "IMG_0133.JPG.json").read_bytes()

    assert raw == RECORD_JSON.encode(RECORD_JSON.decode(raw))
    assert engine.raw_state["verdicts"][(BUCKET, "IMG_0133.JPG")]["img_status"] == "completed"


def test_verdict_writes_only_its_own_shard(
    engine: ReviewEngine, ctx: ReviewContext, tmp_path: Path
) -> None:
    """A verdict on one image must not touch another image's shard record (O(dets on this
    image), not O(all-reviewed))."""
    import tcip_store
    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE, review_verdict_key

    engine.mark_image_reviewed(BUCKET, "IMG_OTHER.JPG")
    other_key = review_verdict_key(tmp_path, BUCKET, "IMG_OTHER.JPG")
    before = tcip_store.read_versioned(other_key).version

    matches = compute_matches(ctx.gt, ctx.preds, iou_threshold=0.5, conf_threshold=0.25)
    dets = engine.build_detection_list(ctx, matches)
    engine.record_detection_action(BUCKET, dets[0], ctx, action="accepted")

    assert tcip_store.exists(review_verdict_key(tmp_path, BUCKET, "IMG_0133.JPG"))
    assert tcip_store.read_versioned(other_key).version == before  # untouched
    assert {k.parts[1] for k in tcip_store.keys(REVIEW_VERDICTS_STORE, str(tmp_path))} == {
        "IMG_OTHER.JPG", "IMG_0133.JPG"}


def test_verdict_calls_shard_writer_exactly_once(
    engine: ReviewEngine, ctx: ReviewContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []
    orig = ReviewEngine._save_image

    def spy(self: ReviewEngine, bucket: str, img_name: str) -> None:
        calls.append((bucket, img_name))
        orig(self, bucket, img_name)

    monkeypatch.setattr(ReviewEngine, "_save_image", spy)

    matches = compute_matches(ctx.gt, ctx.preds, iou_threshold=0.5, conf_threshold=0.25)
    dets = engine.build_detection_list(ctx, matches)
    engine.record_detection_action(BUCKET, dets[0], ctx, action="accepted")
    assert calls == [(BUCKET, ctx.img_name)]  # exactly one shard write, for the touched image only


def test_verdicts_across_images_produce_one_shard_each(
    engine: ReviewEngine, ctx: ReviewContext, tmp_path: Path
) -> None:
    import tcip_store
    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE

    matches = compute_matches(ctx.gt, ctx.preds, iou_threshold=0.5, conf_threshold=0.25)
    dets = engine.build_detection_list(ctx, matches)
    engine.record_detection_action(BUCKET, dets[0], ctx, action="accepted")
    engine.mark_image_reviewed(BUCKET, "IMG_0200.JPG")

    shards = sorted(k.parts[1] for k in tcip_store.keys(REVIEW_VERDICTS_STORE, str(tmp_path)))
    assert shards == ["IMG_0133.JPG", "IMG_0200.JPG"]
    # Each shard holds only its own image's data.
    assert engine.raw_state["verdicts"][(BUCKET, "IMG_0133.JPG")]["detections"]
    assert engine.raw_state["verdicts"][(BUCKET, "IMG_0200.JPG")]["detections"] == []


def test_raw_state_round_trips_through_reload(engine: ReviewEngine, ctx: ReviewContext, tmp_path: Path) -> None:
    matches = compute_matches(ctx.gt, ctx.preds, iou_threshold=0.5, conf_threshold=0.25)
    dets = engine.build_detection_list(ctx, matches)
    for det in dets:
        engine.record_detection_action(BUCKET, det, ctx, action="accepted")
    before = engine.raw_state

    reloaded = ReviewEngine(state_dir=tmp_path)
    assert reloaded.raw_state == before
def test_shard_key_with_separator_round_trips(tmp_path: Path) -> None:
    """A key bearing a path separator survives a save/reload without mutation (the shard filename
    is sanitized, but the true key is read back from the payload)."""
    eng = ReviewEngine(tmp_path)
    eng.mark_image_reviewed(BUCKET, "sub/img.jpg")
    before = eng.raw_state

    reloaded = ReviewEngine(state_dir=tmp_path)
    assert reloaded.raw_state == before
    assert reloaded.is_image_reviewed(BUCKET, "sub/img.jpg")


def test_shard_keys_colliding_after_sanitization_stay_distinct(tmp_path: Path) -> None:
    """'a/b.jpg' and 'a_b.jpg' sanitize to the same base but must not share a shard or merge state."""
    eng = ReviewEngine(tmp_path)
    eng.mark_image_reviewed(BUCKET, "a/b.jpg")       # completed
    eng.mark_image_reviewed(BUCKET, "a_b.jpg")
    eng.unmark_image_reviewed(BUCKET, "a_b.jpg")     # -> not_started (no detections)
    import tcip_store
    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE

    assert len(tcip_store.keys(REVIEW_VERDICTS_STORE, str(tmp_path))) == 2  # two distinct records

    reloaded = ReviewEngine(state_dir=tmp_path)
    # keys preserved, not merged
    assert set(reloaded.raw_state["verdicts"]) == {(BUCKET, "a/b.jpg"), (BUCKET, "a_b.jpg")}
    assert reloaded.is_image_reviewed(BUCKET, "a/b.jpg")
    assert not reloaded.is_image_reviewed(BUCKET, "a_b.jpg")


def test_mark_image_reviewed_merges_a_second_subjects_coverage_without_erasing_the_first(
    engine: ReviewEngine,
) -> None:
    """A second Complete under another subject on the same image adds its own entry to the
    coverage map rather than overwriting what the first one confirmed."""
    engine.mark_image_reviewed(BUCKET, "IMG_0300.JPG", adjudication_covered={"bud": True})
    engine.mark_image_reviewed(BUCKET, "IMG_0300.JPG", adjudication_covered={"leaf": False})
    img_data = engine.raw_state["verdicts"][(BUCKET, "IMG_0300.JPG")]
    assert img_data["adjudication_covered"] == {"bud": True, "leaf": False}


def test_mark_image_reviewed_refuses_to_merge_over_a_non_map_existing_value(
    engine: ReviewEngine,
) -> None:
    """A bare boolean already recorded for a shard is a shape no current writer produces; the
    merge raises by name rather than silently discarding it."""
    engine.mark_image_reviewed(BUCKET, "IMG_0301.JPG", adjudication_covered={"bud": True})
    engine.raw_state["verdicts"][(BUCKET, "IMG_0301.JPG")]["adjudication_covered"] = True
    with pytest.raises(ValueError, match="adjudication_covered"):
        engine.mark_image_reviewed(BUCKET, "IMG_0301.JPG", adjudication_covered={"bud": True})


def test_build_detection_list_tp_fp_fn(engine: ReviewEngine, ctx: ReviewContext) -> None:
    matches = compute_matches(ctx.gt, ctx.preds, iou_threshold=0.5, conf_threshold=0.25)
    assert len(matches["tp"]) == 1
    assert len(matches["fp"]) == 1
    assert len(matches["fn"]) == 1

    dets = engine.build_detection_list(ctx, matches)
    types = sorted(d.det_type for d in dets)
    assert types == ["fn", "fp", "tp"]


def test_build_detection_list_filter_type(engine: ReviewEngine, ctx: ReviewContext) -> None:
    matches = compute_matches(ctx.gt, ctx.preds, iou_threshold=0.5, conf_threshold=0.25)
    tp_only = engine.build_detection_list(ctx, matches, filter_type="tp")
    assert len(tp_only) == 1 and tp_only[0].det_type == "tp"

    fp_only = engine.build_detection_list(ctx, matches, filter_type="fp")
    assert len(fp_only) == 1 and fp_only[0].det_type == "fp"


def test_build_detection_list_filter_class(engine: ReviewEngine, ctx: ReviewContext) -> None:
    matches = compute_matches(ctx.gt, ctx.preds, iou_threshold=0.5, conf_threshold=0.25)
    all_bud = engine.build_detection_list(ctx, matches, filter_class="bud")
    # The fixture's whole match set is bud (1 TP, 1 FP, 1 FN), so naming that class must keep
    # every one of them: an empty result would satisfy the per-item check vacuously.
    assert len(all_bud) == 3
    assert all(d.class_name == "bud" for d in all_bud)
    none = engine.build_detection_list(ctx, matches, filter_class="nonexistent")
    assert none == []


def test_class_filter_keeps_the_named_class_and_drops_the_others(engine: ReviewEngine) -> None:
    """Naming a class must select that class's detections and exclude the rest, with both sides
    of the partition non-empty so neither a pass-everything nor a drop-everything filter reads
    as correct."""
    ctx = ReviewContext(
        img_name="IMG_0777.JPG",
        img_width=900,
        img_height=400,
        gt=[
            Annotation(subject="bud", geometry=BBox(100, 100, 300, 180)),
            Annotation(subject="leaf", geometry=BBox(500, 40, 700, 300)),
            Annotation(subject="leaf", geometry=BBox(60, 250, 200, 380)),
        ],
        preds=[
            Annotation(subject="leaf", geometry=BBox(504, 44, 704, 304), score=0.88),
            Annotation(subject="bud", geometry=BBox(760, 300, 860, 360), score=0.61),
        ],
    )
    matches = compute_matches(ctx.gt, ctx.preds, iou_threshold=0.5, conf_threshold=0.25)

    buds = engine.build_detection_list(ctx, matches, filter_class="bud")
    leaves = engine.build_detection_list(ctx, matches, filter_class="leaf")
    assert len(buds) == 2  # the unmatched bud GT and the bud prediction that hit nothing
    assert len(leaves) == 2  # the matched leaf and the leaf GT nothing was predicted for
    assert {d.class_name for d in buds} == {"bud"}
    assert {d.class_name for d in leaves} == {"leaf"}
    assert sorted(d.det_type for d in buds) == ["fn", "fp"]
    assert sorted(d.det_type for d in leaves) == ["fn", "tp"]


def test_record_and_find_reviewed(engine: ReviewEngine, ctx: ReviewContext) -> None:
    matches = compute_matches(ctx.gt, ctx.preds, iou_threshold=0.5, conf_threshold=0.25)
    dets = engine.build_detection_list(ctx, matches)
    target = next(d for d in dets if d.det_type == "tp")

    # Before recording: not found
    assert engine.find_reviewed_entry(BUCKET, target, ctx) is None

    engine.record_detection_action(BUCKET, target, ctx, action="accepted")
    entry = engine.find_reviewed_entry(BUCKET, target, ctx)
    assert entry is not None
    assert entry["action"] == "accepted"
    assert entry["match_type"] == "TP"
    assert entry["reviewed_by"] == "alice"
    assert entry["class_name"] == "bud"


def test_record_overrides_existing(engine: ReviewEngine, ctx: ReviewContext) -> None:
    matches = compute_matches(ctx.gt, ctx.preds, iou_threshold=0.5, conf_threshold=0.25)
    dets = engine.build_detection_list(ctx, matches)
    target = dets[0]
    engine.record_detection_action(BUCKET, target, ctx, action="accepted")
    engine.record_detection_action(BUCKET, target, ctx, action="rejected")
    entry = engine.find_reviewed_entry(BUCKET, target, ctx)
    assert entry is not None
    assert entry["action"] == "rejected"
    # Should still be exactly one entry, not two
    img_data = engine.raw_state["verdicts"][(BUCKET, ctx.img_name)]
    assert len(img_data["detections"]) == 1


def test_record_detection_action_refuses_an_action_outside_the_declared_vocabulary(
    engine: ReviewEngine, ctx: ReviewContext
) -> None:
    """The engine's write boundary rejects an action the vocabulary doesn't declare, so a direct
    caller cannot store a verdict none of the consumers read."""
    matches = compute_matches(ctx.gt, ctx.preds, iou_threshold=0.5, conf_threshold=0.25)
    dets = engine.build_detection_list(ctx, matches)
    target = dets[0]
    with pytest.raises(ValueError, match="unknown verdict action"):
        engine.record_detection_action(BUCKET, target, ctx, action="approved")
    assert engine.raw_state == {}


def test_record_stamps_missed_object_attested_for_a_genuine_new_attestation(
    engine: ReviewEngine, ctx: ReviewContext
) -> None:
    # The "mark missed object" tool's exact call shape (ReviewTab.tsx's recordMissedObject) has
    # neither an existing GT nor an existing prediction to key off of. missed_object_attested is
    # stamped from that call-site fact directly, not reconstructed later from bbox geometry.
    det = ReviewDetection(det_type="fn", class_name="bud", conf=None, iou=None,
                          gt_idx=None, pred_idx=None, bbox=(10, 10, 20, 20))
    engine.record_detection_action(BUCKET, det, ctx, action="edited")
    entry = engine.raw_state["verdicts"][(BUCKET, ctx.img_name)]["detections"][0]
    assert entry["missed_object_attested"] is True


def test_record_does_not_mistake_an_existing_fn_rejection_for_a_missed_object_attestation(
    engine: ReviewEngine, ctx: ReviewContext
) -> None:
    # A rejected pre-existing FN (an existing, already-indexed GT box the breeder decided was wrong,
    # not a newly-attested miss) ends up with the same persisted bbox shape as a genuine attestation
    # once written (pred_bbox_norm=None, gt_bbox_norm=<box>): geometry alone can't tell them apart.
    # missed_object_attested must read False here, since the call site supplied a real gt_idx.
    matches = compute_matches(ctx.gt, ctx.preds, iou_threshold=0.5, conf_threshold=0.25)
    dets = engine.build_detection_list(ctx, matches)
    fn_det = next(d for d in dets if d.det_type == "fn")
    assert fn_det.gt_idx is not None and fn_det.pred_idx is None  # a real, indexed FN

    engine.record_detection_action(BUCKET, fn_det, ctx, action="rejected")
    entry = engine.find_reviewed_entry(BUCKET, fn_det, ctx)
    assert entry is not None
    assert entry["pred_bbox_norm"] is None and entry["gt_bbox_norm"] is not None  # the ambiguous shape
    assert entry["missed_object_attested"] is False


def test_build_detection_list_never_hides_reviewed(engine: ReviewEngine, ctx: ReviewContext) -> None:
    # Review status is image-level navigation, not per-detection visibility: reviewing a detection
    # must never drop it from the walkable list (else it can't be re-inspected or un-done).
    matches = compute_matches(ctx.gt, ctx.preds, iou_threshold=0.5, conf_threshold=0.25)
    dets = engine.build_detection_list(ctx, matches)
    before = len(dets)
    engine.record_detection_action(BUCKET, dets[0], ctx, action="accepted")
    after = engine.build_detection_list(ctx, matches)
    assert len(after) == before  # the accepted detection is still walkable


def test_get_all_image_statuses(engine: ReviewEngine, ctx: ReviewContext) -> None:
    assert engine.get_all_image_statuses() == {}  # nothing touched yet
    matches = compute_matches(ctx.gt, ctx.preds, iou_threshold=0.5, conf_threshold=0.25)
    dets = engine.build_detection_list(ctx, matches)
    engine.record_detection_action(BUCKET, dets[0], ctx, action="accepted")  # -> "started"
    engine.mark_image_reviewed(BUCKET, "IMG_OTHER.JPG")  # -> "completed"
    statuses = engine.get_all_image_statuses()
    assert statuses[ctx.img_name] == "started"
    assert statuses["IMG_OTHER.JPG"] == "completed"


def test_check_image_review_complete(engine: ReviewEngine, ctx: ReviewContext) -> None:
    matches = compute_matches(ctx.gt, ctx.preds, iou_threshold=0.5, conf_threshold=0.25)
    dets = engine.build_detection_list(ctx, matches)
    # Accept each detection one at a time
    for det in dets[:-1]:
        engine.record_detection_action(BUCKET, det, ctx, action="accepted")
    assert engine.check_image_review_complete(BUCKET, ctx, matches) is False

    engine.record_detection_action(BUCKET, dets[-1], ctx, action="accepted")
    assert engine.check_image_review_complete(BUCKET, ctx, matches) is True
    assert engine.is_image_reviewed(BUCKET, ctx.img_name)


def test_review_progress_counts_an_aliased_pair_as_one_reviewed_of_two(
    engine: ReviewEngine, ctx: ReviewContext
) -> None:
    """Two current FP detections at the same predicted centre both alias to the one stored entry
    the centre-only lookup finds: recording a verdict against each writes the same entry twice, so
    the pair reads as one reviewed of two, and the image does not complete while it is the last
    detection left."""
    aliased_ctx = replace(ctx, preds=[
        *ctx.preds,
        Annotation(subject="bud", geometry=BBox(700, 700, 800, 800), score=0.7),  # aliases the FP
    ])
    matches = compute_matches(aliased_ctx.gt, aliased_ctx.preds, iou_threshold=0.5, conf_threshold=0.25)
    dets = engine.build_detection_list(aliased_ctx, matches)
    fp_dets = [d for d in dets if d.det_type == "fp"]
    assert len(fp_dets) == 2

    for det in dets:
        engine.record_detection_action(BUCKET, det, aliased_ctx, action="accepted")

    reviewed, total = engine.review_progress(BUCKET, aliased_ctx, dets)
    assert total == len(dets)
    assert reviewed == total - 1
    assert engine.check_image_review_complete(BUCKET, aliased_ctx, matches) is False


def test_backup_original_labels_per_file(engine: ReviewEngine, tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    orig = '{"annotations": [{"subject": "bud", "bbox": [1, 1, 8, 8]}]}'
    (labels_dir / "IMG_0001.json").write_text(orig)
    assert engine.backup_original_labels(labels_dir) == 1
    backup = labels_dir / ".original" / "IMG_0001.json"
    assert backup.read_text() == orig

    # Mutate the label on disk; a later backup must not overwrite its baseline
    (labels_dir / "IMG_0001.json").write_text('{"annotations": []}')
    assert engine.backup_original_labels(labels_dir) == 0
    assert backup.read_text() == orig  # still original

    # A label added after the first backup still gets its own baseline captured
    second = '{"annotations": [{"subject": "bud", "bbox": [3, 3, 4, 4]}]}'
    (labels_dir / "IMG_0002.json").write_text(second)
    assert engine.backup_original_labels(labels_dir) == 1
    assert (labels_dir / ".original" / "IMG_0002.json").read_text() == second
    assert backup.read_text() == orig


def test_backup_sweep_and_per_file_capture_share_one_baseline(
    engine: ReviewEngine, tmp_path: Path
) -> None:
    """The directory sweep (``backup_original_labels``) and the per-file capture the review
    routes call (``capture_label_baseline``) both go through the identical create-only
    implementation, so whichever one reaches a file first is the baseline that survives, no
    matter which entry point runs second."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    label = labels_dir / "IMG_0001.json"
    first_writer_bytes = '{"annotations": [{"subject": "bud", "bbox": [1, 1, 8, 8]}]}'
    label.write_text(first_writer_bytes)

    assert capture_label_baseline(label) is True
    label.write_text('{"annotations": []}')
    assert engine.backup_original_labels(labels_dir) == 0

    backup = labels_dir / ".original" / "IMG_0001.json"
    assert backup.read_text() == first_writer_bytes


def test_plain_compute_matches_can_never_produce_a_tp_for_a_classified_trait() -> None:
    # The reproduced defect: an attribute-scoped detector's predictions carry the classified VALUE
    # on `subject` (a joint detect-and-classify class space, class_registry.assign_class_ids), while
    # GT keeps the real object type on `subject` and the confirmed value in `attributes[attribute]`.
    # Plain compute_matches groups strictly by identical `subject`, so these two vocabularies never
    # intersect -- a correctly classified instance could never register as a match, regardless of
    # model quality.
    gt = [Annotation(subject="bud", geometry=BBox(100, 100, 200, 200),
                     attributes={"opening": "open"})]
    preds = [Annotation(subject="open", geometry=BBox(102, 102, 198, 198), score=0.9)]
    matches = compute_matches(gt, preds, iou_threshold=0.5, conf_threshold=0.25)
    assert matches["tp"] == []
    assert len(matches["fn"]) == 1 and len(matches["fp"]) == 1  # never even compared


_OPENING_VOCABULARY = {"closed", "open"}


def test_compute_classified_trait_matches_produces_a_tp_for_a_correct_classification() -> None:
    gt = [Annotation(subject="bud", geometry=BBox(100, 100, 200, 200),
                     attributes={"opening": "open"})]
    preds = [Annotation(subject="bud", geometry=BBox(102, 102, 198, 198), score=0.9,
                       attributes={"opening": "open"})]
    matches = compute_classified_trait_matches(
        gt, preds, subject="bud", attribute="opening", vocabulary=_OPENING_VOCABULARY,
        iou_threshold=0.5, conf_threshold=0.25,
    )
    assert len(matches["tp"]) == 1
    assert matches["fp"] == [] and matches["fn"] == []
    tp = matches["tp"][0]
    assert tp["class_name"] == "open"
    assert tp["gt_idx"] == 0 and tp["pred_idx"] == 0  # indexes the caller's real, unprojected lists


def test_compute_classified_trait_matches_a_misclassification_is_an_fp_and_fn_pair() -> None:
    # The model found the object but called it the wrong value: an FN for the confirmed value
    # paired with an FP for the wrongly predicted one, the existing accept/reject vocabulary.
    gt = [Annotation(subject="bud", geometry=BBox(100, 100, 200, 200),
                     attributes={"opening": "closed"})]
    preds = [Annotation(subject="bud", geometry=BBox(102, 102, 198, 198), score=0.9,
                       attributes={"opening": "open"})]
    matches = compute_classified_trait_matches(
        gt, preds, subject="bud", attribute="opening", vocabulary=_OPENING_VOCABULARY,
        iou_threshold=0.5, conf_threshold=0.25,
    )
    assert matches["tp"] == []
    assert [m["class_name"] for m in matches["fn"]] == ["closed"]
    assert [m["class_name"] for m in matches["fp"]] == ["open"]


def test_compute_classified_trait_matches_excludes_unassessed_and_out_of_scope_instances() -> None:
    gt = [
        # never assessed for `opening` yet: a soft, expected gap, not a confirmed negative
        Annotation(subject="bud", geometry=BBox(100, 100, 200, 200)),
        # a different, enabling subject sharing the same labels dir -- must not enter the match pool
        Annotation(subject="bush", geometry=BBox(300, 300, 400, 400), attributes={"opening": "open"}),
    ]
    preds = [Annotation(subject="bud", geometry=BBox(102, 102, 198, 198), score=0.9,
                       attributes={"opening": "open"})]
    matches = compute_classified_trait_matches(
        gt, preds, subject="bud", attribute="opening", vocabulary=_OPENING_VOCABULARY,
        iou_threshold=0.5, conf_threshold=0.25,
    )
    assert matches["tp"] == [] and matches["fn"] == []  # neither GT instance was ever a real match
    assert len(matches["fp"]) == 1  # the prediction itself is still walkable, nothing confirms it


def test_check_image_review_complete_ignores_a_coverage_only_sweep_entry(
    engine: ReviewEngine, ctx: ReviewContext
) -> None:
    # "swept this image, found nothing more" (no gt_idx/pred_idx, no edited geometry) must not
    # inflate the reviewed count: it doesn't correspond to any of `matches`' TP/FP/FN entries, so
    # counting it could flip an image with real unreviewed detections to "completed" early.
    matches = compute_matches(ctx.gt, ctx.preds, iou_threshold=0.5, conf_threshold=0.25)
    sweep_det = ReviewDetection(det_type="sweep", class_name="", conf=None, iou=None,
                                gt_idx=None, pred_idx=None, bbox=(0, 0, ctx.img_width, ctx.img_height))
    engine.record_detection_action(BUCKET, sweep_det, ctx, action="swept")
    assert engine.check_image_review_complete(BUCKET, ctx, matches) is False

    dets = engine.build_detection_list(ctx, matches)
    for det in dets:
        engine.record_detection_action(BUCKET, det, ctx, action="accepted")
    assert engine.check_image_review_complete(BUCKET, ctx, matches) is True


def test_save_gt_writes_merged_file(engine: ReviewEngine, ctx: ReviewContext, tmp_path: Path) -> None:
    from tcip_annotation.json_io import read_annotations

    # One merged per-image file holds every subject: a box and a polygon together.
    ctx.gt = [
        Annotation(subject="bud", geometry=BBox(100, 100, 200, 200)),
        Annotation(subject="leaf", geometry=Polygon([[(10.0, 10.0), (20.0, 10.0), (20.0, 20.0)]])),
        # An accepted occlusion-split prediction keeps both of its contours through the GT write.
        Annotation(subject="nut", geometry=Polygon([
            [(30.0, 30.0), (40.0, 30.0), (40.0, 40.0)],
            [(60.0, 30.0), (70.0, 30.0), (70.0, 40.0)],
        ])),
    ]
    path = tmp_path / "out" / "IMG.json"
    ok = engine.save_gt(ctx, path=str(path))
    assert ok
    read_back = read_annotations(str(path))
    assert len(read_back) == 3
    box_ann = next(a for a in read_back if isinstance(a.geometry, BBox))
    assert box_ann.subject == "bud"
    poly_ann = next(a for a in read_back if a.subject == "leaf")
    assert poly_ann.geometry.rings == [[(10.0, 10.0), (20.0, 10.0), (20.0, 20.0)]]
    multi_ann = next(a for a in read_back if a.subject == "nut")
    assert multi_ann.geometry.rings == [
        [(30.0, 30.0), (40.0, 30.0), (40.0, 40.0)],
        [(60.0, 30.0), (70.0, 30.0), (70.0, 40.0)],
    ]


def test_a_matched_detection_pairs_the_gt_with_the_prediction_that_matched_it(
    engine: ReviewEngine, unordered_ctx: ReviewContext
) -> None:
    """A walkable TP must index the GT and the prediction that actually matched each other.

    The two lists are in different orders here, so a detection that read its GT out of the
    prediction list (or the reverse) would zoom the reviewer to another object's box and key the
    verdict to a pair nobody adjudicated.
    """
    matches = _unordered_matches(unordered_ctx)
    dets = engine.build_detection_list(unordered_ctx, matches, filter_type="tp")
    assert len(dets) == 2

    for det in dets:
        assert det.gt_idx is not None and det.pred_idx is not None
        # The confidence carried on the detection belongs to the prediction it indexes.
        assert unordered_ctx.preds[det.pred_idx].score == pytest.approx(det.conf)
        # A GT annotation is the one without a score; a prediction indexed as GT would carry one.
        assert unordered_ctx.gt[det.gt_idx].score is None

    strongest = next(d for d in dets if d.conf == pytest.approx(0.90))
    assert (strongest.gt_idx, strongest.pred_idx) == (1, 0)
    # The auto-zoom box spans that GT and that prediction together, no other object's.
    assert strongest.bbox == (700.0, 260.0, 904.0, 424.0)


def test_persisted_verdict_boxes_scale_width_by_image_width(
    engine: ReviewEngine, unordered_ctx: ReviewContext
) -> None:
    """A verdict's stored box is ``[cx, cy, w, h]`` with each term divided by its own dimension.

    The image is wider than it is tall and the boxes are not square, so a width divided by the
    height still produces plausible centers while every stored size is off by the aspect ratio,
    which is what the calibration reference built from these entries measures objects with.
    """
    matches = _unordered_matches(unordered_ctx)
    dets = engine.build_detection_list(unordered_ctx, matches, filter_type="tp")
    strongest = next(d for d in dets if d.conf == pytest.approx(0.90))

    engine.record_detection_action(BUCKET, strongest, unordered_ctx, action="accepted")
    entry = engine.raw_state["verdicts"][(BUCKET, unordered_ctx.img_name)]["detections"][0]

    assert entry["gt_bbox_norm"] == pytest.approx([0.666667, 0.68, 0.166667, 0.32], abs=1e-6)
    assert entry["pred_bbox_norm"] == pytest.approx([0.67, 0.688, 0.166667, 0.32], abs=1e-6)


def test_gt_preexisting_keeps_the_state_from_before_the_review_session(
    engine: ReviewEngine, unordered_ctx: ReviewContext
) -> None:
    """``gt_preexisting`` records whether the image had ground truth before review began.

    A session that authors ground truth of its own must not turn that recorded False into True on
    a later verdict: the pre-review fact is unrecoverable once overwritten.
    """
    empty = ReviewContext(
        img_name="IMG_0900.JPG",
        img_width=1200,
        img_height=500,
        gt=[],
        preds=[Annotation(subject="bud", geometry=BBox(200, 60, 320, 140), score=0.77)],
    )
    first = ReviewDetection(det_type="fp", class_name="bud", conf=0.77, iou=None,
                            gt_idx=None, pred_idx=0, bbox=(200, 60, 320, 140))
    engine.record_detection_action(BUCKET, first, empty, action="rejected")
    assert engine.raw_state["verdicts"][(BUCKET, "IMG_0900.JPG")]["gt_preexisting"] is False

    # The reviewer then draws a missed object, so the session's own ctx now carries ground truth.
    authored = replace(empty, gt=[Annotation(subject="bud", geometry=BBox(600, 200, 720, 300))])
    second = ReviewDetection(det_type="fn", class_name="bud", conf=None, iou=None,
                             gt_idx=0, pred_idx=None, bbox=(600, 200, 720, 300))
    engine.record_detection_action(BUCKET, second, authored, action="accepted")
    assert engine.raw_state["verdicts"][(BUCKET, "IMG_0900.JPG")]["gt_preexisting"] is False

    # An image that genuinely did have ground truth before review still records True.
    dets = engine.build_detection_list(unordered_ctx, _unordered_matches(unordered_ctx))
    engine.record_detection_action(BUCKET, dets[0], unordered_ctx, action="accepted")
    assert engine.raw_state["verdicts"][(BUCKET, unordered_ctx.img_name)]["gt_preexisting"] is True


def test_verdict_count_matches_bucket_stems_against_extensioned_log_keys(
    engine: ReviewEngine, unordered_ctx: ReviewContext
) -> None:
    """Verdicts are counted by image stem on both sides of the comparison.

    A prediction bucket enumerates bare ``<stem>.json`` names while the review log keys on the
    image filename with its extension. Comparing the two vocabularies unstemmed makes every
    reviewed bucket report zero verdicts, and bucket immutability stops holding silently.
    """
    matches = _unordered_matches(unordered_ctx)
    dets = engine.build_detection_list(unordered_ctx, matches)
    for det in dets[:3]:
        engine.record_detection_action(BUCKET, det, unordered_ctx, action="accepted")
    assert len(engine.raw_state["verdicts"][(BUCKET, unordered_ctx.img_name)]["detections"]) == 3

    assert engine.verdict_count_for_images(BUCKET, ["IMG_0501"]) == 3
    assert engine.verdict_count_for_images(BUCKET, ["IMG_0501", "IMG_9999"]) == 3
    assert engine.verdict_count_for_images(BUCKET, ["IMG_9999"]) == 0
    assert engine.verdict_count_for_images(BUCKET, []) == 0


def test_completion_gate_counts_every_detection_not_only_the_filtered_ones(
    engine: ReviewEngine, unordered_ctx: ReviewContext
) -> None:
    """Completion is judged against the whole match set, never the reviewer's active filter.

    A session filtered to one match type adjudicates a subset; declaring the image finished there
    would retire it with real, never-walked detections still unreviewed.
    """
    matches = _unordered_matches(unordered_ctx)
    tp_only = engine.build_detection_list(unordered_ctx, matches, filter_type="tp")
    assert len(tp_only) == 2
    for det in tp_only:
        engine.record_detection_action(BUCKET, det, unordered_ctx, action="accepted")
    assert engine.check_image_review_complete(BUCKET, unordered_ctx, matches) is False
    assert engine.get_image_review_status(BUCKET, unordered_ctx.img_name) == "started"

    for det in engine.build_detection_list(unordered_ctx, matches, filter_type="fp"):
        engine.record_detection_action(BUCKET, det, unordered_ctx, action="rejected")
    assert engine.check_image_review_complete(BUCKET, unordered_ctx, matches) is False

    for det in engine.build_detection_list(unordered_ctx, matches, filter_type="fn"):
        engine.record_detection_action(BUCKET, det, unordered_ctx, action="accepted")
    assert engine.check_image_review_complete(BUCKET, unordered_ctx, matches) is True


def test_an_attested_miss_stays_attested_when_keyed_to_the_authored_geometry(
    engine: ReviewEngine, unordered_ctx: ReviewContext
) -> None:
    """Marking a missed object authors ground truth and keys the entry to the drawn box, while
    the attestation itself is read from the call site's own shape (neither index supplied).

    Reading the attestation off the post-authoring geometry instead would report False for every
    genuine find, and adjudication coverage, the half of the calibration gate this control exists
    to supply, would silently read as unattested.
    """
    drawn = (500.0, 100.0, 620.0, 180.0)
    det = ReviewDetection(det_type="fn", class_name="bud", conf=None, iou=None,
                          gt_idx=None, pred_idx=None, bbox=drawn)
    authored = replace(
        unordered_ctx,
        gt=[*unordered_ctx.gt, Annotation(subject="bud", geometry=BBox(*drawn))],
    )
    engine.record_detection_action(BUCKET,
        det, unordered_ctx, action="edited",
        norm_det=replace(det, gt_idx=len(authored.gt) - 1), norm_ctx=authored,
    )

    entry = engine.raw_state["verdicts"][(BUCKET, unordered_ctx.img_name)]["detections"][0]
    assert entry["missed_object_attested"] is True
    assert entry["pred_bbox_norm"] is None
    assert entry["gt_bbox_norm"] == pytest.approx([0.466667, 0.28, 0.1, 0.16], abs=1e-6)


def test_a_shard_write_that_fails_to_land_is_refused_out_loud(
    engine: ReviewEngine, unordered_ctx: ReviewContext, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shard write that never lands raises rather than being logged and dropped.

    A caller told a verdict was recorded when it was not is a reviewer who will never revisit
    that detection. The staging file it wrote through must also be gone whether or not the swap
    succeeded, since the review dir is enumerated as shards, and the verdict already confirmed on
    disk stays complete and readable.

    Bound to the file backend on purpose: the shard-swap mechanism this monkeypatches
    (``os.replace``) is the file backend's own atomic-write step, which a database backend never
    calls.
    """
    import tcip_store
    from tcip_store.file_backend import FileBackend

    tcip_store.bind(FileBackend())
    matches = _unordered_matches(unordered_ctx)
    dets = engine.build_detection_list(unordered_ctx, matches)
    engine.record_detection_action(BUCKET, dets[0], unordered_ctx, action="accepted")

    def refuse(src, dst, **kwargs):
        raise OSError("shard swap refused")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError, match="shard swap refused"):
        engine.record_detection_action(BUCKET, dets[1], unordered_ctx, action="rejected")
    monkeypatch.undo()

    assert sorted(p.name for p in engine.shard_dir.rglob("*.json")) == ["IMG_0501.JPG.json"]
    assert [p.name for p in engine.shard_dir.rglob("*") if p.suffix == ".tmp"] == []

    reloaded = ReviewEngine(state_dir=tmp_path)
    persisted = reloaded.raw_state["verdicts"][(BUCKET, unordered_ctx.img_name)]["detections"]
    assert len(persisted) == 1
    assert persisted[0]["action"] == "accepted"
    assert persisted[0]["det_status"] == "reviewed"


def test_load_review_state_refuses_a_version_refused_shard_rather_than_skip_it(
    tmp_path: Path,
) -> None:
    """A shard at a schema_version this reader does not accept holds a real human verdict, so
    losing it the way an unrelated corrupt shard is dropped (logged and skipped) would silently
    drop that verdict; it must propagate instead.
    """
    import tcip_store
    from tcip_store import SchemaVersionRefused
    from tcip_store.file_backend import FileBackend

    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE, review_verdict_key

    tcip_store.bind(FileBackend())
    key = review_verdict_key(tmp_path, BUCKET, "IMG_0001.JPG")
    path = FileBackend().path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    poisoned = tcip_store.get_descriptor(REVIEW_VERDICTS_STORE).codec.encode(
        {"bucket": BUCKET, "img_name": "IMG_0001.JPG", "state": {"accepted": True},
         "schema_version": 2}
    )
    path.write_bytes(poisoned)

    with pytest.raises(SchemaVersionRefused):
        ReviewEngine(state_dir=tmp_path)
