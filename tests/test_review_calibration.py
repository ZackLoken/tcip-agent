"""The review-confirmation reference: validate the count operating point against a
breeder-confirmed sample of the model's own outputs, held to the identical gate the held-out-GT
path uses (the shared-reference principle), and blocked by the same conf-censoring guard.

Producer-identity scoping and FN-adjudication coverage are both exercised here: every fixture
below carries an explicit ``producer_identity`` on its verdict entries and an explicit
``bucket_identities`` argument at the call site: there is no legitimate call to
``review_to_records``/``resolve_operating_point_from_review`` without one.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")  # operating_point imports evaluation which imports torch

from tcip_mcp.pipelines.feedback import (  # noqa: E402
    resolve_operating_point_from_review,
    review_conf_threshold,
    review_reference_hash,
    review_to_records,
)
from tcip_mcp.pipelines.resolution import (  # noqa: E402
    VALIDATED_FALSE,
    VALIDATED_REVIEW_CONFIRMED,
)

# No built-in traits: seed_bud_trait_spec (conftest.py) writes a real bud.yml into this
# test's pinned platform state root so resolve_operating_point_from_review("bud_opening", ...) keeps
# resolving by default.
pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

_DIMS = {"A.jpg": (400, 400), "B.jpg": (400, 400)}

N_IMAGES = 16
OBJECTS_PER_IMAGE = 40

_IDENTITY_A = {"checkpoint_sha256": "sha-model-a", "experiment_id": None}
_IDENTITY_B = {"checkpoint_sha256": "sha-model-b", "experiment_id": None}


def _entry(mt, action, cid, gt, pred, conf, *, producer_identity=_IDENTITY_A, conf_threshold=None,
           missed_object_attested=False):
    return {"match_type": mt, "action": action, "class_id": cid,
            "gt_bbox_norm": gt, "pred_bbox_norm": pred, "conf": conf,
            "producer_identity": producer_identity, "conf_threshold": conf_threshold,
            "missed_object_attested": missed_object_attested}


def _floored_state():
    """Two disjoint completed images whose predictions include the low-conf tail (floored infer).
    ``gt_preexisting=True`` marks these as a GT-backed review session, not the
    previously-unlabeled scenario exercised separately below."""
    a = {"img_status": "completed", "gt_preexisting": True, "detections": [
        _entry("TP", "accepted", 0, [0.25, 0.25, 0.05, 0.05], [0.25, 0.25, 0.05, 0.05], 0.9),
        _entry("FP", "rejected", 0, None, [0.75, 0.75, 0.05, 0.05], 0.05)]}
    b = {"img_status": "completed", "gt_preexisting": True, "detections": [
        _entry("TP", "accepted", 0, [0.25, 0.25, 0.05, 0.05], [0.25, 0.25, 0.05, 0.05], 0.9),
        _entry("TP", "accepted", 0, [0.5, 0.5, 0.05, 0.05], [0.5, 0.5, 0.05, 0.05], 0.05)]}
    return {"image": {"A.jpg": a, "B.jpg": b}}


def _dense_review_state(n_images=N_IMAGES, objects_per_image=OBJECTS_PER_IMAGE, *, id_prefix="A",
                        conf=0.9, fp_conf=0.05, miss_pattern=None, fp_pattern=None,
                        gt_preexisting=True, producer_identity=None):
    """A dense, realistic review reference: ``objects_per_image`` confirmed matches per image
    (grid-laid-out normalized boxes, well outside the derived center-match tolerance), an
    optional per-image confirmed-miss count (``miss_pattern``, a gt-only, no-pred entry, i.e. a
    genuine FN attested via the "mark missed object" tool), and an optional per-image rejected-FP
    count at a low, realistic conf. Normalized unit-square coordinates (no ``image_dims``), so the
    boxes stay resolution-independent.

    ``gt_preexisting`` stamps the image-level fact a GT-backed review session would have
    recorded; ``False`` simulates a previously-unlabeled image, where the only
    adjudication-coverage evidence is a genuine gt-only entry from ``miss_pattern``.
    """
    identity = producer_identity or _IDENTITY_A
    miss_pattern = list(miss_pattern) if miss_pattern is not None else [0] * n_images
    fp_pattern = list(fp_pattern) if fp_pattern is not None else [0] * n_images
    cols = int(objects_per_image**0.5) + 2
    far_row = (objects_per_image // cols) + 2
    images = {}
    for i in range(n_images):
        dets = []
        # A tiny per-image jitter (well inside the derived center-match tolerance, so matching is
        # unaffected) keeps every image's GT content byte-distinct: without it every image shares
        # the exact same grid and the content-overlap gate flags the holdout as a full content
        # clone of calibration (real geometry never repeats identically image to image).
        jitter = i * 0.0001
        for k in range(objects_per_image):
            row, col = divmod(k, cols)
            box = [0.05 + col * 0.02 + jitter, 0.05 + row * 0.02, 0.01, 0.01]
            if k < miss_pattern[i]:
                dets.append(_entry("FN", "edited", 0, box, None, None, producer_identity=identity,
                                   missed_object_attested=True))
            else:
                dets.append(_entry("TP", "accepted", 0, box, box, conf, producer_identity=identity))
        for j in range(fp_pattern[i]):
            fp_box = [0.05 + j * 0.02, 0.05 + (far_row + i) * 0.02, 0.01, 0.01]
            dets.append(_entry("FP", "rejected", 0, None, fp_box, fp_conf, producer_identity=identity))
        images[f"{id_prefix}{i}.jpg"] = {"img_status": "completed", "detections": dets,
                                         "gt_preexisting": gt_preexisting}
    return {"image": images}


def _good_review_state():
    """A dense reference with a realistic low-conf-FP profile: a genuinely floored reference must
    still validate, so the count-unbiased pick lands at the high, correct-match conf once the
    low-conf FP is filtered out, uniformly across every image.
    """
    return _dense_review_state(fp_pattern=[1] * N_IMAGES)


def test_review_to_records_reconstructs_gt_and_dt():
    recs = review_to_records(_floored_state(), image_dims=_DIMS, bucket_identities=[_IDENTITY_A])
    by_id = {r["image_id"]: r for r in recs}
    # image_id is the stem, matching the GT path's convention and training stems:
    # an extensioned id here could never match a training stem in _train_disjointness.
    assert set(by_id) == {"A", "B"}
    # gt = affirmed boxes (accepted/edited/FN/accepted-FP); dt = every model prediction w/ its score.
    assert len(by_id["A"]["gt"]) == 1  # rejected FP is not gt
    assert sorted(d["score"] for d in by_id["A"]["dt"]) == [0.05, 0.9]  # both predictions kept
    assert len(by_id["B"]["gt"]) == 2
    # boxes are pixel xywh top-left (denormalized), category lifted to 1-indexed like the GT path.
    gt0 = by_id["A"]["gt"][0]
    assert gt0["category_id"] == 1
    assert gt0["bbox"] == pytest.approx([90.0, 90.0, 20.0, 20.0])


def test_review_only_completed_images():
    state = _floored_state()
    state["image"]["C.jpg"] = {"img_status": "started", "detections": [
        _entry("TP", "accepted", 0, [0.1, 0.1, 0.05, 0.05], [0.1, 0.1, 0.05, 0.05], 0.9)]}
    ids = {r["image_id"] for r in review_to_records(state, image_dims=_DIMS,
                                                    bucket_identities=[_IDENTITY_A])}
    assert ids == {"A", "B"}  # a partially-reviewed image is not a confirmed reference


def test_review_confirmed_stamps_when_the_same_gate_passes(tmp_path):
    # staged_conf_floor simulates what the review-path threading computes and passes (threaded
    # from routes/review.py); the seam here is a caller-supplied value.
    # tiled=False: this test is about conf-calibration shippability, not tiling; tile_size only
    # gates a bundle when tiled.
    b = resolve_operating_point_from_review(_good_review_state(), "bud_opening", staged_conf_floor=0.01,
                                            tiled=False, bucket_identities=[_IDENTITY_A], scope_root=tmp_path)
    conf = b.get("conf")
    # A disjoint, uncensored, count-bias-passing review reference earns review_confirmed (distinct
    # from VALIDATED_HELD_OUT so provenance records which reference validated) and is shippable.
    assert conf.validated_against == VALIDATED_REVIEW_CONFIRMED
    assert conf.derived_from == "count-unbiased center-match curve over review verdicts"
    assert conf.dataset_scoped is True
    assert b.is_shippable is True
    assert conf.gate_evidence["failures"] == []


def test_review_confirmed_fails_closed_without_a_staged_conf_floor(tmp_path):
    # With no staged_conf_floor asserted, even a geometrically perfect reference cannot validate:
    # the honest default, not a silent pass. Same fixture as the passing case above.
    b = resolve_operating_point_from_review(_good_review_state(), "bud_opening",
                                            tiled=True, bucket_identities=[_IDENTITY_A], scope_root=tmp_path)
    conf = b.get("conf")
    assert conf.validated_against == VALIDATED_FALSE
    assert b.is_shippable is False
    assert "conf_floor_unstated" in conf.gate_evidence["failures"]
    assert "conf_censored" not in conf.gate_evidence["failures"]


def test_conf_censored_review_reference_refused_when_picked_conf_at_or_below_the_staged_floor(tmp_path):
    # The asserted floor sits at the picked conf: the sweep could not have seen anything below
    # it, so it must refuse even though the split is disjoint and the counts genuinely agree.
    b = resolve_operating_point_from_review(_good_review_state(), "bud_opening", tiled=True, staged_conf_floor=0.95,
                                            bucket_identities=[_IDENTITY_A], scope_root=tmp_path)
    conf = b.get("conf")
    assert conf.validated_against == VALIDATED_FALSE
    assert b.is_shippable is False
    assert "conf_censored" in conf.gate_evidence["failures"]


def test_review_reference_hash_scopes_to_the_affirmed_reference():
    recs = review_to_records(_floored_state(), image_dims=_DIMS, bucket_identities=[_IDENTITY_A])
    h1 = review_reference_hash(recs)
    # a different affirmed-gt set is a different reference identity
    recs2 = review_to_records(_floored_state(), image_dims=_DIMS, bucket_identities=[_IDENTITY_A])
    recs2[0]["gt"].append({"category_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0], "iscrowd": 0})
    assert review_reference_hash(recs2) != h1
    assert len(h1) == 16


# ── producer-identity scoping ───────────────────────────────────────


def test_producer_identity_mismatch_excludes_verdicts_entirely():
    # The reproduced defect: model A's review verdicts must not validate model B's bucket.
    recs = review_to_records(_floored_state(), image_dims=_DIMS, bucket_identities=[_IDENTITY_B])
    assert recs == []


def test_producer_identity_matches_by_checkpoint_sha_regardless_of_bucket_dir():
    # A benign re-export of the same checkpoint into a fresh directory (a routine workflow) must
    # still validate: matching is on checkpoint_sha256/experiment_id, never a directory-string
    # comparison.
    state = _floored_state()
    for img in state["image"].values():
        for entry in img["detections"]:
            entry["producer_identity"] = {**_IDENTITY_A, "bucket_dir": "predictions/model/2026-01-01"}
    target = [{"checkpoint_sha256": _IDENTITY_A["checkpoint_sha256"], "experiment_id": None}]
    recs = review_to_records(state, image_dims=_DIMS, bucket_identities=target)
    assert {r["image_id"] for r in recs} == {"A", "B"}


def test_missing_producer_identity_fails_closed_not_grandfathered():
    # Verdicts with no producer_identity at all are excluded, never grandfathered in (CLAUDE.md's
    # no-back-compat rule).
    state = {"image": {"A.jpg": {"img_status": "completed", "gt_preexisting": True, "detections": [
        {"match_type": "TP", "action": "accepted", "class_id": 0,
         "gt_bbox_norm": [0.25, 0.25, 0.05, 0.05], "pred_bbox_norm": [0.25, 0.25, 0.05, 0.05],
         "conf": 0.9}
    ]}}}
    recs = review_to_records(state, image_dims=_DIMS, bucket_identities=[_IDENTITY_A])
    assert recs == []


def test_unresolvable_class_id_refuses_the_whole_reference_not_a_silent_drop():
    # A verdict whose class_id is None (the producing bucket's own id_map either doesn't exist or
    # doesn't recognize this verdict's class_name) must refuse the whole reference, not silently
    # exclude the one entry. A silent per-entry drop is a fail-open here: it can delete a
    # confirmed miss (FN) while keeping an in-vocabulary accepted-FP entry, making gt/dt agree by
    # construction and pass the count-bias gate on a reference missing real evidence.
    state = {"image": {"A.jpg": {"img_status": "completed", "gt_preexisting": True, "detections": [
        {"match_type": "TP", "action": "accepted", "class_id": None, "class_name": "bud",
         "gt_bbox_norm": [0.25, 0.25, 0.05, 0.05], "pred_bbox_norm": [0.25, 0.25, 0.05, 0.05],
         "conf": 0.9, "producer_identity": _IDENTITY_A}
    ]}}}
    with pytest.raises(ValueError, match="no resolvable class identity"):
        review_to_records(state, image_dims=_DIMS, bucket_identities=[_IDENTITY_A])


def test_missing_class_id_key_also_refuses_not_defaulted_to_class_one():
    # A verdict entry with no class_id key at all (a shape record_detection_action can write)
    # must not silently default to category_id 1 for every entry; it must refuse instead of
    # guessing.
    state = {"image": {"A.jpg": {"img_status": "completed", "gt_preexisting": True, "detections": [
        {"match_type": "TP", "action": "accepted", "class_name": "bud",
         "gt_bbox_norm": [0.25, 0.25, 0.05, 0.05], "pred_bbox_norm": [0.25, 0.25, 0.05, 0.05],
         "conf": 0.9, "producer_identity": _IDENTITY_A}
    ]}}}
    with pytest.raises(ValueError, match="no resolvable class identity"):
        review_to_records(state, image_dims=_DIMS, bucket_identities=[_IDENTITY_A])


def test_class_id_unresolvable_message_is_drawn_from_the_shared_failure_vocabulary():
    # The refusal text must read in the same breeder-facing "Not yet." voice
    # describe_review_validation's own _FAILURE_MESSAGES entries use, not an independently
    # authored string, so a breeder sees one consistent voice regardless of which check refused.
    state = {"image": {"A.jpg": {"img_status": "completed", "gt_preexisting": True, "detections": [
        {"match_type": "TP", "action": "accepted", "class_id": None, "class_name": "bud",
         "gt_bbox_norm": [0.25, 0.25, 0.05, 0.05], "pred_bbox_norm": [0.25, 0.25, 0.05, 0.05],
         "conf": 0.9, "producer_identity": _IDENTITY_A}
    ]}}}
    with pytest.raises(ValueError) as exc_info:
        review_to_records(state, image_dims=_DIMS, bucket_identities=[_IDENTITY_A])
    assert str(exc_info.value).startswith("Not yet.")
    assert "no resolvable class identity" in str(exc_info.value)


def test_a_coverage_only_attestation_needs_no_resolvable_class_id():
    # "swept this image, found nothing more" (ReviewTab.tsx's recordSweepAttested): neither
    # gt_bbox_norm nor pred_bbox_norm set, class_id unresolved (nothing was classified). This must
    # not refuse the reference -- the entry carries no class-scoped evidence to admit either way --
    # and its missed_object_attested stamp must still count toward adjudication coverage.
    state = {"image": {"A.jpg": {"img_status": "completed", "gt_preexisting": False, "detections": [
        {"match_type": "sweep", "action": "swept", "class_id": None, "class_name": "",
         "gt_bbox_norm": None, "pred_bbox_norm": None, "conf": None,
         "producer_identity": _IDENTITY_A, "missed_object_attested": True}
    ]}}}
    recs = review_to_records(state, image_dims=_DIMS, bucket_identities=[_IDENTITY_A])
    assert len(recs) == 1
    assert recs[0]["gt"] == [] and recs[0]["dt"] == []
    assert recs[0]["adjudication_covered"] is True


def test_confirmed_negative_image_carries_its_own_producer_identity():
    # A confirmed negative (mark_complete, zero verdict entries) stamps identity at the image
    # level; it must remain in the reference, correctly attributed, not silently dropped just
    # because it has nothing to carry a per-entry identity on.
    state = {"image": {"NEG.jpg": {"img_status": "completed", "gt_preexisting": True,
                                    "detections": [], "producer_identity": _IDENTITY_A}}}
    recs = review_to_records(state, image_dims=_DIMS, bucket_identities=[_IDENTITY_A])
    assert len(recs) == 1
    assert recs[0]["gt"] == [] and recs[0]["dt"] == []

    # Scoped to a different bucket, the same negative must not be attributed to it.
    recs_other = review_to_records(state, image_dims=_DIMS, bucket_identities=[_IDENTITY_B])
    assert recs_other == []


# ── subject-scoped zero-verdict coverage ─────────────────────────────


def test_covers_reads_the_subjects_own_entry():
    from tcip_mcp.pipelines.feedback.review_calibration import covers

    slot = {"bud": True, "leaf": False}
    assert covers(slot, "bud") is True
    assert covers(slot, "leaf") is False


def test_covers_falls_back_to_the_star_entry_for_a_subject_less_complete():
    # A subject-less Complete's "*" entry is a claim about every subject, so any subject's
    # coverage check must read it as covered.
    from tcip_mcp.pipelines.feedback.review_calibration import covers

    slot = {"*": True}
    assert covers(slot, "bud") is True
    assert covers(slot, None) is True


def test_covers_reads_false_for_an_unrecorded_subject_and_no_star_entry():
    from tcip_mcp.pipelines.feedback.review_calibration import covers

    slot = {"leaf": True}
    assert covers(slot, "bud") is False


def test_covers_reads_false_for_no_recorded_map_at_all():
    from tcip_mcp.pipelines.feedback.review_calibration import covers

    assert covers(None, "bud") is False


def test_covers_reads_the_subjects_own_false_entry_over_a_star_entry():
    # An explicit per-subject False must not be overridden by an unrelated "*" entry: the
    # subject's own entry, whatever it says, wins once it is present.
    from tcip_mcp.pipelines.feedback.review_calibration import covers

    slot = {"*": True, "leaf": False}
    assert covers(slot, "leaf") is False


def test_covers_refuses_a_bare_boolean_by_name():
    # A bare boolean is not a shape any writer produces; reading one as covering everything or
    # nothing would guess at the subject a bare flag never named.
    from tcip_mcp.pipelines.feedback.review_calibration import covers

    with pytest.raises(ValueError, match="adjudication_covered"):
        covers(True, "bud")


def test_zero_verdict_image_coverage_is_read_for_the_subject_the_reference_validates():
    # A file holding another subject's boxes must not read as covered for the subject actually
    # being validated.
    state = {"image": {"NEG.jpg": {
        "img_status": "completed", "detections": [],
        "producer_identity": _IDENTITY_A,
        "adjudication_covered": {"bud": True, "leaf": False},
    }}}
    subject_recs = review_to_records(state, bucket_identities=[_IDENTITY_A], subject="bud")
    assert subject_recs[0]["adjudication_covered"] is True
    leaf_recs = review_to_records(state, bucket_identities=[_IDENTITY_A], subject="leaf")
    assert leaf_recs[0]["adjudication_covered"] is False


def test_zero_verdict_image_with_a_subject_less_star_entry_covers_the_validated_subject():
    state = {"image": {"NEG.jpg": {
        "img_status": "completed", "detections": [],
        "producer_identity": _IDENTITY_A,
        "adjudication_covered": {"*": True},
    }}}
    recs = review_to_records(state, bucket_identities=[_IDENTITY_A], subject="bud")
    assert recs[0]["adjudication_covered"] is True


# ── FN-adjudication coverage ────────────────────────────────────────


def test_previously_unlabeled_session_with_marked_misses_can_still_validate(tmp_path):
    # The breeder used the "mark missed object" tool at least once per image on images with no
    # pre-existing GT (gt_preexisting=False): the gate must be able to reach review_confirmed.
    # fp_pattern=2 (not 1): with exactly one confirmed miss per image, a single low-conf FP would
    # numerically cancel it at the low-conf grid point (both magnitude 1), landing the picker
    # there instead of the true high-conf agreement point; 2 avoids that coincidence while
    # staying realistic.
    #
    # objects_per_image=250, not the module default 40: the relative count-bias tolerance scales
    # with this reference's own typical per-image count, and at 40 a single permanent miss is a
    # ~2.5% relative error, over the default 1% tolerance. 250 keeps the same miss=1/fp=2 counts
    # while making that identical single miss a ~0.4% relative error, comfortably inside
    # tolerance.
    state = _dense_review_state(gt_preexisting=False, miss_pattern=[1] * N_IMAGES,
                                fp_pattern=[2] * N_IMAGES, objects_per_image=250)
    b = resolve_operating_point_from_review(state, "bud_opening", tiled=True, staged_conf_floor=0.01,
                                            bucket_identities=[_IDENTITY_A], scope_root=tmp_path)
    conf = b.get("conf")
    assert conf.validated_against == VALIDATED_REVIEW_CONFIRMED
    assert "insufficient_adjudication_coverage" not in conf.gate_evidence["failures"]


def test_previously_unlabeled_session_with_zero_adjudication_refuses_honestly(tmp_path):
    # Previously-unlabeled images, reviewed (accept/reject), but never checked for a missed
    # object, must fail with an honest reason naming the new tool rather than falling through to
    # the generic "counts didn't agree" message.
    state = _dense_review_state(gt_preexisting=False, fp_pattern=[1] * N_IMAGES)
    b = resolve_operating_point_from_review(state, "bud_opening", tiled=True, staged_conf_floor=0.01,
                                            bucket_identities=[_IDENTITY_A], scope_root=tmp_path)
    conf = b.get("conf")
    assert conf.validated_against == VALIDATED_FALSE
    assert conf.gate_evidence["failures"] == ["insufficient_adjudication_coverage"]


def test_gt_backed_session_passes_unaffected_by_the_coverage_gate(tmp_path):
    # A genuinely GT-backed review session (gt_preexisting=True, the default) must still pass,
    # unaffected by the adjudication-coverage gate.
    b = resolve_operating_point_from_review(_good_review_state(), "bud_opening", tiled=True, staged_conf_floor=0.01,
                                            bucket_identities=[_IDENTITY_A], scope_root=tmp_path)
    conf = b.get("conf")
    assert conf.validated_against == VALIDATED_REVIEW_CONFIRMED
    assert "insufficient_adjudication_coverage" not in conf.gate_evidence["failures"]


def test_rejected_fn_geometry_is_not_mistaken_for_a_missed_object_attestation():
    # A rejected pre-existing FN (the breeder decided an existing GT box was wrong and removed
    # it, not a newly-attested miss) leaves the identical pred_bbox_norm=None /
    # gt_bbox_norm=<box> shape a genuine "mark missed object" attestation would.
    # missed_object_attested (the explicit, call-site-derived fact record_detection_action
    # stamps, never bbox geometry) is what adjudication_covered must key off of.
    state = {"image": {"A.jpg": {"img_status": "completed", "gt_preexisting": False, "detections": [
        _entry("FN", "rejected", 0, [0.25, 0.25, 0.05, 0.05], None, None,
              missed_object_attested=False),
    ]}}}
    recs = review_to_records(state, image_dims=_DIMS, bucket_identities=[_IDENTITY_A])
    assert len(recs) == 1
    assert recs[0]["adjudication_covered"] is False


# ── review_conf_threshold ────────────


def test_review_conf_threshold_is_the_max_across_scoped_verdicts():
    state = {"image": {"A.jpg": {"img_status": "completed", "detections": [
        _entry("TP", "accepted", 0, [0.25, 0.25, 0.05, 0.05], [0.25, 0.25, 0.05, 0.05], 0.9,
              conf_threshold=0.1),
        _entry("FP", "rejected", 0, None, [0.75, 0.75, 0.05, 0.05], 0.05, conf_threshold=0.3)]},
        "B.jpg": {"img_status": "completed", "detections": [
        _entry("TP", "accepted", 0, [0.1, 0.1, 0.05, 0.05], [0.1, 0.1, 0.05, 0.05], 0.9,
              conf_threshold=0.2)]}}}
    assert review_conf_threshold(state, bucket_identities=[_IDENTITY_A]) == pytest.approx(0.3)


def test_review_conf_threshold_none_when_any_scoped_verdict_lacks_it():
    # A verdict with no conf_threshold at all leaves the review-side term unknown (never inferred
    # as 0 or skipped), so the caller's max(...) combination fails closed.
    state = {"image": {"A.jpg": {"img_status": "completed", "detections": [
        _entry("TP", "accepted", 0, [0.25, 0.25, 0.05, 0.05], [0.25, 0.25, 0.05, 0.05], 0.9,
              conf_threshold=0.1),
        _entry("FP", "rejected", 0, None, [0.75, 0.75, 0.05, 0.05], 0.05, conf_threshold=None)]}}}
    assert review_conf_threshold(state, bucket_identities=[_IDENTITY_A]) is None


def test_review_conf_threshold_scoped_to_the_matching_bucket_only():
    # A verdict recorded against a different producer must not inflate/deflate this bucket's
    # floor: the same _matches_any_bucket predicate review_to_records uses, not a second
    # implementation.
    state = {"image": {"A.jpg": {"img_status": "completed", "detections": [
        _entry("TP", "accepted", 0, [0.25, 0.25, 0.05, 0.05], [0.25, 0.25, 0.05, 0.05], 0.9,
              conf_threshold=0.1, producer_identity=_IDENTITY_A),
        _entry("TP", "accepted", 0, [0.5, 0.5, 0.05, 0.05], [0.5, 0.5, 0.05, 0.05], 0.9,
              conf_threshold=0.9, producer_identity=_IDENTITY_B)]}}}
    assert review_conf_threshold(state, bucket_identities=[_IDENTITY_A]) == pytest.approx(0.1)
    assert review_conf_threshold(state, bucket_identities=[_IDENTITY_B]) == pytest.approx(0.9)


def test_review_conf_threshold_none_when_nothing_scoped_to_this_bucket():
    state = {"image": {"A.jpg": {"img_status": "completed", "detections": [
        _entry("TP", "accepted", 0, [0.25, 0.25, 0.05, 0.05], [0.25, 0.25, 0.05, 0.05], 0.9,
              conf_threshold=0.1, producer_identity=_IDENTITY_A)]}}}
    assert review_conf_threshold(state, bucket_identities=[_IDENTITY_B]) is None


def test_review_conf_threshold_respects_only_completed():
    state = {"image": {"A.jpg": {"img_status": "started", "detections": [
        _entry("TP", "accepted", 0, [0.25, 0.25, 0.05, 0.05], [0.25, 0.25, 0.05, 0.05], 0.9,
              conf_threshold=0.7)]}}}
    assert review_conf_threshold(state, bucket_identities=[_IDENTITY_A], only_completed=True) is None
    assert review_conf_threshold(state, bucket_identities=[_IDENTITY_A],
                                 only_completed=False) == pytest.approx(0.7)
