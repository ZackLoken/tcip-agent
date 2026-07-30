"""W1-R2: the review-confirmation reference — validate the count operating point against a
breeder-confirmed sample of the model's own outputs, held to the IDENTICAL gate the held-out-GT
path uses (the shared-reference principle), and blocked by the same conf-censoring guard (Fix D).

Fix G (producer-identity scoping) and Fix H (FN-adjudication coverage) are both exercised here:
every fixture below carries an explicit ``producer_identity`` on its verdict entries and an
explicit ``bucket_identities`` argument at the call site — there is no legitimate call to
``review_to_records``/``resolve_operating_point_from_review`` without one (K2).
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

# Round 10 (2026-07-29): no built-in traits — seed_catkin_trait_spec (conftest.py) writes a real
# catkin.yml into this test's pinned project root so resolve_operating_point_from_review("catkin",
# ...) keeps resolving by default.
pytestmark = pytest.mark.usefixtures("seed_catkin_trait_spec")

_DIMS = {"A.jpg": (400, 400), "B.jpg": (400, 400)}

N_IMAGES = 16
OBJECTS_PER_IMAGE = 40

_IDENTITY_A = {"checkpoint_sha256": "sha-model-a", "experiment_id": None}
_IDENTITY_B = {"checkpoint_sha256": "sha-model-b", "experiment_id": None}


def _entry(mt, action, cid, gt, pred, conf, *, producer_identity=_IDENTITY_A, conf_threshold=None):
    return {"match_type": mt, "action": action, "class_id": cid,
            "gt_bbox_norm": gt, "pred_bbox_norm": pred, "conf": conf,
            "producer_identity": producer_identity, "conf_threshold": conf_threshold}


def _floored_state():
    """Two disjoint completed images whose predictions include the low-conf tail (floored infer).
    ``gt_preexisting=True`` (Fix H): these represent a GT-backed review session, not the
    previously-unlabeled scenario Fix H's own tests exercise separately below."""
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
    """A dense, realistic (rule 17) review reference: ``objects_per_image`` confirmed matches per
    image (grid-laid-out normalized boxes, well outside the derived center-match tolerance), an
    optional per-image confirmed-miss count (``miss_pattern`` — a gt-only, no-pred entry, i.e. a
    genuine FN attested via the "mark missed object" tool), and an optional per-image rejected-FP
    count at a low, realistic conf. Normalized unit-square coordinates (no ``image_dims``), so the
    boxes stay resolution-independent.

    ``gt_preexisting`` (Fix H) stamps the image-level fact a GT-backed review session would have
    recorded; ``False`` simulates a previously-unlabeled image, where the ONLY adjudication-coverage
    evidence is a genuine gt-only entry from ``miss_pattern``.
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
        # unaffected) keeps every image's GT content byte-distinct — without it every image shares
        # the exact same grid and K1's content-overlap gate flags the holdout as a full content clone
        # of calibration (real geometry never repeats identically image to image).
        jitter = i * 0.0001
        for k in range(objects_per_image):
            row, col = divmod(k, cols)
            box = [0.05 + col * 0.02 + jitter, 0.05 + row * 0.02, 0.01, 0.01]
            if k < miss_pattern[i]:
                dets.append(_entry("FN", "edited", 0, box, None, None, producer_identity=identity))
            else:
                dets.append(_entry("TP", "accepted", 0, box, box, conf, producer_identity=identity))
        for j in range(fp_pattern[i]):
            fp_box = [0.05 + j * 0.02, 0.05 + (far_row + i) * 0.02, 0.01, 0.01]
            dets.append(_entry("FP", "rejected", 0, None, fp_box, fp_conf, producer_identity=identity))
        images[f"{id_prefix}{i}.jpg"] = {"img_status": "completed", "detections": dets,
                                         "gt_preexisting": gt_preexisting}
    return {"image": images}


def _good_review_state():
    """A dense reference with a realistic low-conf-FP profile (matches Fix D's test obligation: a
    genuinely floored reference must still validate) — the count-unbiased pick lands at the high,
    correct-match conf once the low-conf FP is filtered out, uniformly across every image.
    """
    return _dense_review_state(fp_pattern=[1] * N_IMAGES)


def test_review_to_records_reconstructs_gt_and_dt():
    recs = review_to_records(_floored_state(), image_dims=_DIMS, bucket_identities=[_IDENTITY_A])
    by_id = {r["image_id"]: r for r in recs}
    # image_id is the STEM (K1 finding 2), matching the GT path's convention and training stems —
    # an extensioned id here could never match a training stem in _train_disjointness.
    assert set(by_id) == {"A", "B"}
    # gt = affirmed boxes (accepted/edited/FN/accepted-FP); dt = every model prediction w/ its score.
    assert len(by_id["A"]["gt"]) == 1  # rejected FP is NOT gt
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


def test_review_confirmed_stamps_when_the_same_gate_passes():
    # staged_conf_floor SIMULATES what the review-path threading computes and passes (Fix D item 4,
    # threaded from routes/review.py) — the seam here is a caller-supplied value.
    b = resolve_operating_point_from_review(_good_review_state(), "catkin", staged_conf_floor=0.01,
                                            bucket_identities=[_IDENTITY_A])
    conf = b.get("conf")
    # A disjoint, uncensored, count-bias-passing review reference earns review_confirmed (distinct
    # from VALIDATED_HELD_OUT so provenance records WHICH reference validated) and is shippable.
    assert conf.validated_against == VALIDATED_REVIEW_CONFIRMED
    assert conf.derived_from == "count-unbiased center-match sweep over review verdicts"
    assert conf.dataset_scoped is True
    assert b.is_shippable is True
    assert conf.sweep["failures"] == []


def test_review_confirmed_fails_closed_without_a_staged_conf_floor():
    # Fix D: with no staged_conf_floor asserted, even a geometrically perfect reference cannot
    # validate — the honest default, not a silent pass. Same fixture as the passing case above.
    b = resolve_operating_point_from_review(_good_review_state(), "catkin",
                                            bucket_identities=[_IDENTITY_A])
    conf = b.get("conf")
    assert conf.validated_against == VALIDATED_FALSE
    assert b.is_shippable is False
    assert "conf_censored" in conf.sweep["failures"]


def test_conf_censored_review_reference_refused_when_picked_conf_at_or_below_the_staged_floor():
    # The asserted floor sits AT the picked conf — the sweep could not have seen anything below it,
    # so it must refuse even though the split is disjoint and the counts genuinely agree.
    b = resolve_operating_point_from_review(_good_review_state(), "catkin", staged_conf_floor=0.95,
                                            bucket_identities=[_IDENTITY_A])
    conf = b.get("conf")
    assert conf.validated_against == VALIDATED_FALSE
    assert b.is_shippable is False
    assert "conf_censored" in conf.sweep["failures"]


def test_review_reference_hash_scopes_to_the_affirmed_reference():
    recs = review_to_records(_floored_state(), image_dims=_DIMS, bucket_identities=[_IDENTITY_A])
    h1 = review_reference_hash(recs)
    # a different affirmed-gt set is a different reference identity
    recs2 = review_to_records(_floored_state(), image_dims=_DIMS, bucket_identities=[_IDENTITY_A])
    recs2[0]["gt"].append({"category_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0], "iscrowd": 0})
    assert review_reference_hash(recs2) != h1
    assert len(h1) == 16


# ── Fix G — producer-identity scoping ───────────────────────────────────────


def test_producer_identity_mismatch_excludes_verdicts_entirely():
    # The reproduced defect: model A's review verdicts must not validate model B's bucket.
    recs = review_to_records(_floored_state(), image_dims=_DIMS, bucket_identities=[_IDENTITY_B])
    assert recs == []


def test_producer_identity_matches_by_checkpoint_sha_regardless_of_bucket_dir():
    # Rule 17: a benign re-export of the SAME checkpoint into a fresh directory (a routine
    # workflow) must still validate — matching is on checkpoint_sha256/experiment_id, never a
    # directory-string comparison.
    state = _floored_state()
    for img in state["image"].values():
        for entry in img["detections"]:
            entry["producer_identity"] = {**_IDENTITY_A, "bucket_dir": "predictions/model/2026-01-01"}
    target = [{"checkpoint_sha256": _IDENTITY_A["checkpoint_sha256"], "experiment_id": None}]
    recs = review_to_records(state, image_dims=_DIMS, bucket_identities=target)
    assert {r["image_id"] for r in recs} == {"A", "B"}


def test_missing_producer_identity_fails_closed_not_grandfathered():
    # Verdicts written before Fix G carry no producer_identity at all — excluded, never
    # grandfathered in (CLAUDE.md's no-back-compat rule).
    state = {"image": {"A.jpg": {"img_status": "completed", "gt_preexisting": True, "detections": [
        {"match_type": "TP", "action": "accepted", "class_id": 0,
         "gt_bbox_norm": [0.25, 0.25, 0.05, 0.05], "pred_bbox_norm": [0.25, 0.25, 0.05, 0.05],
         "conf": 0.9}
    ]}}}
    recs = review_to_records(state, image_dims=_DIMS, bucket_identities=[_IDENTITY_A])
    assert recs == []


def test_confirmed_negative_image_carries_its_own_producer_identity():
    # A confirmed negative (mark_complete, zero verdict entries) stamps identity at the image
    # level (Fix G item 2) — it must remain in the reference, correctly attributed, not silently
    # dropped just because it has nothing to carry a per-entry identity on.
    state = {"image": {"NEG.jpg": {"img_status": "completed", "gt_preexisting": True,
                                    "detections": [], "producer_identity": _IDENTITY_A}}}
    recs = review_to_records(state, image_dims=_DIMS, bucket_identities=[_IDENTITY_A])
    assert len(recs) == 1
    assert recs[0]["gt"] == [] and recs[0]["dt"] == []

    # Scoped to a DIFFERENT bucket, the same negative must not be attributed to it.
    recs_other = review_to_records(state, image_dims=_DIMS, bucket_identities=[_IDENTITY_B])
    assert recs_other == []


# ── Fix H — FN-adjudication coverage ────────────────────────────────────────


def test_previously_unlabeled_session_with_marked_misses_can_still_validate():
    # The breeder used the "mark missed object" tool at least once per image on images with no
    # pre-existing GT (gt_preexisting=False) — the gate must be able to reach review_confirmed.
    # fp_pattern=2 (not 1): with exactly one confirmed miss per image, a SINGLE low-conf FP would
    # numerically cancel it at the low-conf grid point (both magnitude 1), making the picker land
    # there instead of the true high-conf agreement point — an arithmetic coincidence of the
    # fixture, not a real ambiguity; 2 avoids it while staying realistic (rule 17).
    state = _dense_review_state(gt_preexisting=False, miss_pattern=[1] * N_IMAGES,
                                fp_pattern=[2] * N_IMAGES)
    b = resolve_operating_point_from_review(state, "catkin", staged_conf_floor=0.01,
                                            bucket_identities=[_IDENTITY_A])
    conf = b.get("conf")
    assert conf.validated_against == VALIDATED_REVIEW_CONFIRMED
    assert "insufficient_adjudication_coverage" not in conf.sweep["failures"]


def test_previously_unlabeled_session_with_zero_adjudication_refuses_honestly():
    # The corpus's reproduced scenario: previously-unlabeled images, reviewed (accept/reject), but
    # NEVER checked for a missed object — must fail, with an honest reason naming the new tool
    # rather than falling through to the generic "counts didn't agree" message.
    state = _dense_review_state(gt_preexisting=False, fp_pattern=[1] * N_IMAGES)
    b = resolve_operating_point_from_review(state, "catkin", staged_conf_floor=0.01,
                                            bucket_identities=[_IDENTITY_A])
    conf = b.get("conf")
    assert conf.validated_against == VALIDATED_FALSE
    assert conf.sweep["failures"] == ["insufficient_adjudication_coverage"]


def test_gt_backed_session_passes_unaffected_by_the_coverage_gate():
    # A genuinely GT-backed review session (gt_preexisting=True, the default) must still pass —
    # Fix H must not regress the mechanism that already worked before this cluster.
    b = resolve_operating_point_from_review(_good_review_state(), "catkin", staged_conf_floor=0.01,
                                            bucket_identities=[_IDENTITY_A])
    conf = b.get("conf")
    assert conf.validated_against == VALIDATED_REVIEW_CONFIRMED
    assert "insufficient_adjudication_coverage" not in conf.sweep["failures"]


# ── Fix D item 4 — review_conf_threshold (no unit coverage anywhere before this) ────────────


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
    # Verdicts written before this fix carry no conf_threshold at all — the review-side term is
    # UNKNOWN (never inferred as 0 or skipped), so the caller's max(...) combination fails closed.
    state = {"image": {"A.jpg": {"img_status": "completed", "detections": [
        _entry("TP", "accepted", 0, [0.25, 0.25, 0.05, 0.05], [0.25, 0.25, 0.05, 0.05], 0.9,
              conf_threshold=0.1),
        _entry("FP", "rejected", 0, None, [0.75, 0.75, 0.05, 0.05], 0.05, conf_threshold=None)]}}}
    assert review_conf_threshold(state, bucket_identities=[_IDENTITY_A]) is None


def test_review_conf_threshold_scoped_to_the_matching_bucket_only():
    # A verdict recorded against a DIFFERENT producer must not inflate/deflate this bucket's floor —
    # the SAME _matches_any_bucket predicate review_to_records uses, not a second implementation.
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
