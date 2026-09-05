"""The breeder GUI affordance that promotes a completed review into a validation reference.

Two layers: (1) ``describe_review_validation`` translates a resolved bundle into a plain-language,
breeder-facing result (torch-free); (2) the ``/api/review/validate_reference`` route runs the
identical review->calibration gate and stamps the bucket's ``operating_point.json``
review_confirmed (or an honest un-shippable placeholder), never a shortcut to validated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation.json_io import write_annotations
from tcip_mcp.pipelines.feedback import describe_review_validation
from tcip_mcp.pipelines.resolution import (
    VALIDATED_FALSE,
    VALIDATED_REVIEW_CONFIRMED,
    ResolvedBundle,
    derived,
)

# No built-in traits: seed_bud_trait_spec (conftest.py) writes a real bud.yml into this
# test's pinned platform state root so trait="bud_opening" call sites keep resolving.
pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")


def _bundle(*, validated: str, gate_evidence: dict) -> ResolvedBundle:
    conf = derived("conf", 0.42, requires_validation=True, validation_kind="annotations",
                   derived_from="count-unbiased center-match curve over review verdicts",
                   validated_against=validated, dataset_scoped=True, dataset_hash="abc",
                   gate_evidence=gate_evidence)
    return ResolvedBundle(trait="bud_opening", dataset_hash="abc", params={"conf": conf})


def test_describe_validated():
    b = _bundle(validated=VALIDATED_REVIEW_CONFIRMED,
                gate_evidence={"conf_censored": False, "disjoint": True, "passed_holdout": True,
                       "failures": [], "holdout_bias": {"tp": 8, "fn": 2}})
    out = describe_review_validation(b, reviewed_image_count=4)
    assert out["validated"] is True
    assert out["reference"] == VALIDATED_REVIEW_CONFIRMED
    assert out["conf"] == pytest.approx(0.42)
    assert "Validated" in out["reason"] and "4" in out["reason"]
    # The miss-coverage claim is read off the exact-conf holdout_bias entry, not asserted.
    assert "8 of 10" in out["reason"]


def test_describe_validated_names_no_producing_run_when_genuinely_none():
    b = _bundle(validated=VALIDATED_REVIEW_CONFIRMED,
                gate_evidence={"conf_censored": False, "disjoint": True, "passed_holdout": True,
                       "failures": [], "holdout_bias": {"tp": 8, "fn": 2},
                       "train_disjointness": {"checked": False, "experiment_id_ambiguous": False}})
    out = describe_review_validation(b, reviewed_image_count=4)
    assert "The bucket names no producing run" in out["reason"]
    assert "more than one run" not in out["reason"]


def test_describe_validated_names_more_than_one_run_when_buckets_disagree():
    b = _bundle(validated=VALIDATED_REVIEW_CONFIRMED,
                gate_evidence={"conf_censored": False, "disjoint": True, "passed_holdout": True,
                       "failures": [], "holdout_bias": {"tp": 8, "fn": 2},
                       "train_disjointness": {"checked": False, "experiment_id_ambiguous": True}})
    out = describe_review_validation(b, reviewed_image_count=4)
    assert "more than one run" in out["reason"]
    assert "The bucket names no producing run" not in out["reason"]


def test_describe_conf_censored():
    # The named-failure list (not the raw "conf_censored" key alone) drives the branch, and
    # "passed_holdout" must be present or the "too few images" branch wins first.
    b = _bundle(validated=VALIDATED_FALSE,
                gate_evidence={"conf_censored": True, "passed_holdout": False, "failures": ["conf_censored"]})
    out = describe_review_validation(b, reviewed_image_count=3)
    assert out["validated"] is False
    assert "confidence" in out["reason"].lower()


def test_describe_conf_floor_mismatch_is_non_gating_provenance_only():
    # conf_floor_mismatch is surfaced as provenance (gate_evidence["conf_floor_mismatch"]) but never gates
    # on its own; it never appears in "failures", so a bundle with no other named failure stays
    # Validated even when the floor mismatch is flagged.
    b = _bundle(validated=VALIDATED_REVIEW_CONFIRMED,
                gate_evidence={"conf_censored": False, "conf_floor_mismatch": True, "passed_holdout": True,
                       "failures": [], "holdout_bias": {"tp": 5, "fn": 0}})
    out = describe_review_validation(b, reviewed_image_count=4)
    assert out["validated"] is True


def test_conf_floor_mismatch_never_hijacks_a_real_failure_message():
    # A stronger companion to the above: conf_floor_mismatch=True present alongside a real, distinct
    # failure must not divert the message; there is no conf_floor_mismatch-specific branch at all
    # (it is non-gating provenance only), so the real failure's own message must still surface.
    b = _bundle(validated=VALIDATED_FALSE,
                gate_evidence={"conf_censored": False, "disjoint": True, "conf_floor_mismatch": True,
                       "passed_holdout": False,
                       "failures": ["count_bias_exceeds_tolerance"]})
    out = describe_review_validation(b, reviewed_image_count=4)
    assert out["validated"] is False
    assert "agree" in out["reason"]


def test_describe_not_enough_images():
    # One reviewed image -> no holdout was measured (gate_evidence carries no passed_holdout key).
    b = _bundle(validated=VALIDATED_FALSE, gate_evidence={"conf_censored": False, "note": "not held-out"})
    out = describe_review_validation(b, reviewed_image_count=1)
    assert out["validated"] is False
    assert "at least two" in out["reason"]


def test_describe_holdout_bias_failed():
    b = _bundle(validated=VALIDATED_FALSE,
                gate_evidence={"conf_censored": False, "disjoint": True, "passed_holdout": False,
                       "holdout_bias": {"count_bias_mean": 3.0}, "count_bias_tolerance_frac": 1.0,
                       "failures": ["count_bias_exceeds_tolerance"]})
    out = describe_review_validation(b, reviewed_image_count=6)
    assert out["validated"] is False
    assert "agree" in out["reason"]


def test_describe_no_adjudication_coverage():
    # "insufficient_adjudication_coverage" is a distinct, honest reason naming the affordance, not
    # a fallthrough to the generic "counts didn't agree" message.
    b = _bundle(validated=VALIDATED_FALSE,
                gate_evidence={"passed_holdout": False, "failures": ["insufficient_adjudication_coverage"]})
    out = describe_review_validation(b, reviewed_image_count=5)
    assert out["validated"] is False
    assert "mark missed object" in out["reason"].lower()


def test_describe_reports_every_applicable_failure_not_just_the_first():
    # A breeder who hits two blockers at once (e.g. an all-negative split side and a per-class
    # bias failure) must see both in one pass, not fix the first, resubmit, and only then discover
    # the second. _FAILURE_MESSAGES lists insufficient_calibration_gt before
    # count_bias_exceeds_tolerance, so both messages must appear, in that order.
    b = _bundle(validated=VALIDATED_FALSE,
                gate_evidence={"conf_censored": False, "disjoint": True, "passed_holdout": False,
                       "failures": ["count_bias_exceeds_tolerance", "insufficient_calibration_gt"]})
    out = describe_review_validation(b, reviewed_image_count=4)
    assert out["validated"] is False
    assert "all-negative" in out["reason"]  # insufficient_calibration_gt's own message fragment
    assert "agree" in out["reason"]  # count_bias_exceeds_tolerance's own message fragment
    # Priority order from _FAILURE_MESSAGES, not the order failures happened to list them in.
    assert out["reason"].index("all-negative") < out["reason"].index("agree")


def test_describe_never_asserts_the_counts_agree_when_the_pooled_bias_check_also_failed():
    # operating_point.py computes count_bias_ok, per_class_bias_failures, localization_floor_ok
    # and dispersion_ok independently (no mutual exclusion between their failures.append calls),
    # so count_bias_exceeds_tolerance can co-occur with any of the three messages below. Each of
    # those three must not presuppose the pooled check had passed ("the counts happened to
    # match" / "the counts agree on average" / "the overall number... looks right"), since every
    # applicable message is joined and a presupposing phrase would directly contradict
    # count_bias_exceeds_tolerance's own "the model's counts didn't agree closely enough" in the
    # same reason string.
    for co_failure in ("localization_quality_floor_failed", "count_error_dispersion_too_high",
                       "count_bias_exceeds_tolerance_per_class"):
        b = _bundle(validated=VALIDATED_FALSE,
                    gate_evidence={"conf_censored": False, "disjoint": True, "passed_holdout": False,
                           "failures": ["count_bias_exceeds_tolerance", co_failure]})
        out = describe_review_validation(b, reviewed_image_count=6)
        reason = out["reason"]
        assert "didn't agree closely enough" in reason  # count_bias_exceeds_tolerance's own claim
        for contradicting_phrase in ("happened to match", "agree on average", "looks right"):
            assert contradicting_phrase not in reason, (
                f"co_failure={co_failure!r} produced a self-contradictory reason: {reason!r}")


def test_describe_unrecognized_failure_name_is_a_loud_error():
    # Named-failure architecture (cross-cutting): an unmapped failure name must never fall through
    # to the generic message silently: it's a defect in describe_review_validation itself.
    b = _bundle(validated=VALIDATED_FALSE,
                gate_evidence={"conf_censored": False, "disjoint": True, "passed_holdout": False,
                       "failures": ["some_future_fix_g_or_h_failure_not_yet_mapped"]})
    with pytest.raises(AssertionError, match="unrecognized"):
        describe_review_validation(b, reviewed_image_count=2)


def test_describe_conf_floor_unstated_names_the_recognized_attribute_names():
    """A bespoke module holding an unrecognized attribute name (score_threshold, not score_thresh)
    learns why: the message names every attribute the holder looks for, and where."""
    b = _bundle(validated=VALIDATED_FALSE,
                gate_evidence={"conf_censored": False, "disjoint": True, "passed_holdout": False,
                       "failures": ["conf_floor_unstated"]})
    reason = describe_review_validation(b, reviewed_image_count=4)["reason"]
    for attr in ("score_thresh", "nms_thresh", "detections_per_img"):
        assert attr in reason
    assert "detector.roi_heads" in reason


# ── route ─────────────────────────────────────────────────────────────────


_IDENTITY = {"checkpoint_sha256": "sha-model-a", "experiment_id": None}
_OTHER_IDENTITY = {"checkpoint_sha256": "sha-model-b", "experiment_id": None}


def _entry(action, cid, gt, pred, conf, *, producer_identity=_IDENTITY, conf_threshold=None):
    return {"match_type": "TP", "action": action, "class_id": cid,
            "gt_bbox_norm": gt, "pred_bbox_norm": pred, "conf": conf,
            "producer_identity": producer_identity, "conf_threshold": conf_threshold}


def _write_shard(review_dir: Path, bucket_dir: Path, name: str, detections: list, *,
                 gt_preexisting: bool = True) -> None:
    """One completed image's shard, filed under the prediction bucket it was reviewed on.

    Each verdict records the identity of the prediction document the reviewer saw, exactly as the
    route does, so the promotion has the recorded digest it compares against the file on disk. The
    bucket's prediction file for ``name`` must therefore already be written when this is called.
    Written through the seam the engine itself writes through, so the record lands wherever the
    selected backend actually keeps it.
    """
    import tcip_store
    from tcip_annotation.review_engine import review_verdict_key
    from tcip_mcp.prediction_buckets import bucket_key_of

    digest = _pred_digest(bucket_dir, name)
    entries = [{**d, "producer_identity": {**d["producer_identity"], "prediction_digest": digest}}
               if isinstance(d.get("producer_identity"), dict) else d
               for d in detections]
    bucket = bucket_key_of(bucket_dir)
    state_dir = Path(review_dir).parent  # review_verdict_key wants the state dir, review_dir's parent
    tcip_store.replace(
        review_verdict_key(state_dir, bucket, name),
        {"bucket": bucket, "img_name": name,
         "state": {"img_status": "completed",
                   "gt_preexisting": gt_preexisting,
                   "detections": entries}},
        expect=tcip_store.Version.ABSENT)



def _write_empty_prediction_document(path: Path) -> None:
    """An empty prediction document in the shape the platform's own writer produces, the only
    shape the reader admits; the frame is nominal, since no image backs these fixtures."""
    write_annotations(str(path), [], 1, 1, keep_empty=True)

def _pred_digest(bucket_dir: Path, image_name: str) -> str | None:
    """The identity the route records for one image's prediction document, through its own helper."""
    from tcip_web.routes.review import _prediction_digest

    return _prediction_digest(str(bucket_dir), image_name)


def _write_sidecar(pred_dir: Path, identity: dict, *, generation_conf: float | None = None,
                   tile_size_op: dict | None = None) -> None:
    # "tiled" is always stamped, matching a real bucket's sidecar, so the route's tiled_vals
    # resolution has a real value to read, never None; these tests aren't about tiling.
    op: dict = {"tiled": {"value": False}}
    if generation_conf is not None:
        # The conf the bucket's predictions were actually generated/floored at: the route reads
        # this straight off the sidecar (never re-typed) to build staged_conf_floor.
        op["conf"] = {"value": generation_conf}
    if tile_size_op is not None:
        op["tiled"] = {"value": True}
        op["tile_size"] = tile_size_op
    sidecar = {
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "experiment_id": identity["experiment_id"],
        "validated": False,
        "operating_point": op,
        "subject": "bud",
        "attribute": None,
    }
    import tcip_store
    from tcip_mcp.pipelines.resolution import sidecar_key

    tcip_store.replace(sidecar_key(pred_dir, "operating_point"), sidecar,
                       expect=tcip_store.Version.ABSENT)


def _make_project(tmp_path: Path, *, floored: bool, producer_identity: dict = _IDENTITY,
                  sidecar_identity: dict | None = None,
                  gt_preexisting: bool = True) -> tuple[str, str]:
    """A project with two completed-review images + a prediction bucket. ``floored`` includes the
    low-conf tail (the sweep can reach it -> validated); otherwise every conf is above the display
    floor (conf-censored -> refused). Verdicts are recorded against ``producer_identity``; the
    bucket's own sidecar carries ``sidecar_identity`` (defaults to the same identity, a matching,
    scoped reference; some tests pass a deliberately different one to reproduce the mismatch)."""
    proj = tmp_path / "proj"
    review_dir = proj / ".tcip" / "state" / "review"
    pred_dir = proj / "predictions" / "model" / "2026-01-01" / "detect"
    lo = 0.05 if floored else 0.8
    pred_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("A", "B"):
        _write_empty_prediction_document(pred_dir / f"{stem}.json")
    _write_shard(review_dir, pred_dir, "A.jpg", [
        _entry("accepted", 0, [0.25, 0.25, 0.05, 0.05], [0.25, 0.25, 0.05, 0.05], 0.9,
               producer_identity=producer_identity),
        _entry("rejected", 0, None, [0.75, 0.75, 0.05, 0.05], lo,
               producer_identity=producer_identity)], gt_preexisting=gt_preexisting)
    _write_shard(review_dir, pred_dir, "B.jpg", [
        _entry("accepted", 0, [0.25, 0.25, 0.05, 0.05], [0.25, 0.25, 0.05, 0.05], 0.9,
               producer_identity=producer_identity),
        _entry("accepted", 0, [0.5, 0.5, 0.05, 0.05], [0.5, 0.5, 0.05, 0.05], lo,
               producer_identity=producer_identity)], gt_preexisting=gt_preexisting)
    _write_sidecar(pred_dir, sidecar_identity or producer_identity)
    return str(proj), str(pred_dir)


def _make_dense_reviewed_project(tmp_path: Path, *, n_images: int = 6, gt_preexisting: bool = True,
                                 producer_identity: dict = _IDENTITY,
                                 tile_size_op: dict | None = None) -> tuple[str, str]:
    """A project with ``n_images`` completed-review images (>= 2 per side of the locked cal/holdout
    split, since ``_make_project``'s 2-image fixture cannot clear the non-degeneracy floor: it can
    only ever produce a single holdout image) and a realistic staged conf floor: every verdict
    carries its own recorded ``conf_threshold`` and the bucket's sidecar carries its own recorded
    generation conf, so ``routes/review.py`` threads a real, non-``None`` ``staged_conf_floor`` end
    to end rather than always failing closed as conf-censored.

    Every image gets 2 accepted, exactly-matching verdicts at a real detection score (0.9), each
    geometrically distinct from every other image's (so the content-overlap gate never fires on a
    holdout that happens to duplicate calibration's boxes). ``gt_preexisting`` toggles adjudication
    coverage for every image at once: ``True`` clears it (a genuine pre-existing-GT image the
    breeder reviewed), ``False`` reproduces the refusal (no evidence a missed object was ever
    checked for).
    """
    proj = tmp_path / "proj"
    review_dir = proj / ".tcip" / "state" / "review"
    pred_dir = proj / "predictions" / "model" / "2026-01-01" / "detect"
    pred_dir.mkdir(parents=True, exist_ok=True)
    conf_threshold = 0.05
    for i in range(n_images):
        name = chr(ord("A") + i)
        x = 0.05 + 0.1 * i
        gt1, gt2 = [x, 0.2, 0.05, 0.05], [x, 0.6, 0.05, 0.05]
        _write_empty_prediction_document(pred_dir / f"{name}.json")
        _write_shard(review_dir, pred_dir, f"{name}.jpg", [
            _entry("accepted", 0, gt1, gt1, 0.9, producer_identity=producer_identity,
                   conf_threshold=conf_threshold),
            _entry("accepted", 0, gt2, gt2, 0.9, producer_identity=producer_identity,
                   conf_threshold=conf_threshold),
        ], gt_preexisting=gt_preexisting)
    _write_sidecar(pred_dir, producer_identity, generation_conf=conf_threshold,
                  tile_size_op=tile_size_op)
    return str(proj), str(pred_dir)


@pytest.fixture
def client():
    pytest.importorskip("torch")  # resolve_operating_point imports evaluation which imports torch
    from fastapi.testclient import TestClient

    import tcip_web.routes.review as review_mod
    from tcip_web.app import app

    review_mod._engines.clear()  # a stale cached engine would read another test's shards
    return TestClient(app, base_url="http://127.0.0.1")


def _read_sidecar(pred_dir: str) -> dict:
    import tcip_store
    from tcip_mcp.pipelines.resolution import sidecar_key

    return tcip_store.read(sidecar_key(pred_dir, "operating_point"))


def test_route_validates_and_stamps_review_confirmed(client, tmp_path: Path):
    # routes/review.py threads a real staged_conf_floor into resolve_operating_point_from_review:
    # max(generation_conf, review_conf_threshold), the generation half read off the bucket's own
    # operating_point.json sidecar, the review half read off the verdicts' own recorded
    # conf_threshold. This is the design's mandatory acceptance test for the review path: a
    # realistic, disjoint, count-agreeing, adjudication-covered review reference must actually
    # reach review_confirmed end to end through the route.
    proj, pred_dir = _make_dense_reviewed_project(tmp_path)
    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "bud_opening", "pred_dir": pred_dir, "subject": "bud"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is True
    assert body["reference"] == "reviewer_confirmed_annotations"
    # _IDENTITY names no experiment_id: a foreign-checkpoint promotion, the case with no producing
    # run to check the reviewed images' train-disjointness against.
    assert "not checked against that run's training split" in body["reason"]
    sc = _read_sidecar(pred_dir)
    assert sc["validated"] is True
    assert sc["validated_reference"] == "reviewer_confirmed_annotations"
    # The claim is earned: a record outside the bucket answers for it, under the trait it names.
    assert sc["trait"] == "bud_opening"
    from tcip_mcp.experiments import find_validation
    from tcip_mcp.pipelines.resolution import verify_stamp_binding

    pointer = sc["validated_by"]
    row = find_validation(pointer["experiment_id"], pointer["record_digest"])
    assert row is not None
    assert row["trait"] == "bud_opening"
    assert row["reference_identity"]["stated_values"]["review_image_count"] == 6
    assert row["train_disjointness"] == {"checked": False, "group_check": None}
    binding = verify_stamp_binding(sc, pred_dir, document="operating_point")
    assert binding.ok is True
    assert binding.train_disjointness == {"checked": False, "group_check": None}


def test_route_promotion_stamps_schema_version_2_and_carries_an_old_vintage_member(
    client, tmp_path: Path,
):
    """The carried-subrecord contract: promotion writes schema_version 2 for the fields it itself
    touches, but a top-level member the producing run already wrote under an older provenance
    vocabulary (here a hand-built ``mask_binarize`` payload spelled with the retired ``has_sweep``
    key) is not rewritten, so it still reads under that older spelling after promotion. The
    version-2 validation row's own claim carries that same mixed member, since ``mask_binarize``
    is one of operating_point's declared claim keys.
    """
    import tcip_store
    from tcip_mcp.pipelines.resolution import sidecar_key

    proj, pred_dir = _make_dense_reviewed_project(tmp_path)
    key = sidecar_key(pred_dir, "operating_point")
    with tcip_store.transaction(key) as txn:
        current = txn.read(key)
        txn.write(key, {**current, "mask_binarize": {"has_sweep": True, "threshold": 0.5}})

    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "bud_opening", "pred_dir": pred_dir, "subject": "bud"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["validated"] is True

    sc = _read_sidecar(pred_dir)
    assert sc["schema_version"] == 2
    assert sc["mask_binarize"] == {"has_sweep": True, "threshold": 0.5}

    from tcip_mcp.experiments import find_validation

    pointer = sc["validated_by"]
    row = find_validation(pointer["experiment_id"], pointer["record_digest"])
    assert row is not None
    assert row["schema_version"] == 2
    assert row["claim"]["mask_binarize"] == {"has_sweep": True, "threshold": 0.5}


def test_route_promotion_reports_store_contention_as_retryable_not_a_bad_request(
    client, tmp_path: Path, monkeypatch,
):
    """A lock timeout inside the promotion write is an infrastructure fault, so the route answers
    503 with the store's own message, never a 400 that tells the breeder their request was
    malformed (the dataset select route's StoreBusy handling is the platform's precedent). The
    contention is injected because a real lock timeout is not constructible through one client."""
    import tcip_mcp.pipelines.resolution as resolution_mod
    from tcip_mcp.pipelines.resolution import sidecar_key
    from tcip_store.errors import StoreBusy

    proj, pred_dir = _make_dense_reviewed_project(tmp_path)
    busy_key = sidecar_key(pred_dir, "operating_point")

    def _busy(*args, **kwargs):
        raise StoreBusy((busy_key,), busy_key, 5.0)

    monkeypatch.setattr(resolution_mod, "update_sidecar", _busy)
    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "bud_opening", "pred_dir": pred_dir, "subject": "bud"})
    assert resp.status_code == 503, resp.text


def test_route_validates_a_review_that_includes_a_confirmed_negative(client, tmp_path: Path):
    """An image the bucket predicted nothing for is reviewed by marking it Complete. The verdict
    records the absence of a prediction document as the value it is, and the promotion compares that
    absence against the absence still on disk rather than skipping the comparison, so a review
    holding one still earns its record."""
    proj, pred_dir = _make_dense_reviewed_project(tmp_path)
    import tcip_store
    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE

    negative = client.post("/api/review/mark_complete", json={
        "dataset_root": proj, "image_name": "Z.jpg", "pred_dir": pred_dir})
    assert negative.status_code == 200, negative.text
    (shard_key,) = [k for k in tcip_store.keys(REVIEW_VERDICTS_STORE,
                                                str(Path(proj) / ".tcip" / "state"))
                    if k.parts[1] == "Z.jpg"]
    shard = tcip_store.read(shard_key)
    assert shard["state"]["producer_identity"]["prediction_digest"] is None
    assert shard["state"]["adjudication_covered"] == {"*": True}

    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "bud_opening", "pred_dir": pred_dir, "subject": "bud"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is True
    assert body["buckets_stamped"] == [pred_dir]


def test_route_validates_a_subject_less_confirmed_negative_for_any_named_subject(
    client, tmp_path: Path
):
    """A subject-less Complete's ``"*"`` coverage claim is a claim about every subject, so naming
    a subject on the validation request must not withhold the confirmed negative from it. The
    same request naming no subject at all is refused rather than read as every subject."""
    proj, pred_dir = _make_dense_reviewed_project(tmp_path)
    # A document for the negative image, so its stem reaches the reference rather than being
    # dropped before covers() is ever consulted.
    _write_empty_prediction_document(Path(pred_dir) / "Z.json")

    negative = client.post("/api/review/mark_complete", json={
        "dataset_root": proj, "image_name": "Z.jpg", "pred_dir": pred_dir})
    assert negative.status_code == 200, negative.text

    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "bud_opening", "pred_dir": pred_dir, "subject": "bud"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is True
    assert body["reviewed_image_count"] == 7

    refused = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "bud_opening", "pred_dir": pred_dir})
    assert refused.status_code == 400
    assert "subject" in refused.json()["detail"].lower()


def test_route_validates_a_complete_recorded_under_a_named_subject(client, tmp_path: Path):
    """A Complete recorded under a named subject validates a reference requested for that same
    subject end to end, through the real routes."""
    import tcip_store
    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE
    from tcip_mcp.pipelines.resolution import sidecar_key

    proj, pred_dir = _make_dense_reviewed_project(tmp_path)
    stored = tcip_store.read_versioned(sidecar_key(pred_dir, "operating_point"))
    sidecar = stored.value
    sidecar["id_map"] = {"bud": 0}
    tcip_store.replace(sidecar_key(pred_dir, "operating_point"), sidecar, expect=stored.version)
    _write_empty_prediction_document(Path(pred_dir) / "Z.json")

    negative = client.post("/api/review/mark_complete", json={
        "dataset_root": proj, "image_name": "Z.jpg", "pred_dir": pred_dir, "subject": "bud"})
    assert negative.status_code == 200, negative.text
    (shard_key,) = [k for k in tcip_store.keys(REVIEW_VERDICTS_STORE,
                                                str(Path(proj) / ".tcip" / "state"))
                    if k.parts[1] == "Z.jpg"]
    assert tcip_store.read(shard_key)["state"]["adjudication_covered"] == {"bud": True}

    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "bud_opening", "pred_dir": pred_dir, "subject": "bud"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["validated"] is True


def test_route_with_no_pred_dir_stamps_nothing_and_says_why(client, tmp_path: Path):
    # A dataset root with no bucket is a shape the API accepts and answers honestly.
    proj, _ = _make_dense_reviewed_project(tmp_path)
    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "bud_opening", "subject": "bud"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is False
    assert body["buckets_stamped"] == []
    assert "No predictions are selected" in body["reason"]


def test_route_never_upgrades_a_native_ratio_tile_size_to_persisted_geometry(client, tmp_path: Path):
    """A bucket produced by the native-size ratio tier stamps tile_size with
    source="derived" (the same source a real persisted-training-geometry stamp uses) and
    validated_against the tier's own reference: the two bases are distinguishable only by
    validated_against, never by source alone. This route re-derives tile_size_source off the
    bucket's own sidecar to thread through resolve_operating_point_from_review; if it read the
    bare source field it would silently re-resolve a native-ratio bucket as validated persisted
    geometry the moment a breeder reviews it, defeating the platform's own ranking of the two
    tiers."""
    import tcip_mcp.pipelines.resolution as resolution_mod

    native_ref = getattr(resolution_mod, "VALIDATED_NATIVE_FRAME_GEOMETRY", None)
    proj, pred_dir = _make_dense_reviewed_project(tmp_path, tile_size_op={
        "value": 128, "source": "derived",
        "derived_from": "the checkpoint's own uniform untiled training frame",
        "validated_against": native_ref, "requires_validation": True,
        "validation_kind": "geometry",
    })
    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "bud_opening", "pred_dir": pred_dir, "subject": "bud"})
    assert resp.status_code == 200, resp.text
    sc = _read_sidecar(pred_dir)
    restamped = sc["operating_point"]["tile_size"]
    assert restamped["validated_against"] == native_ref
    assert restamped["value"] == 128


def test_route_refuses_an_explicit_tile_edge_with_no_derived_from_text(client, tmp_path: Path):
    """A bucket stamped ``explicit_caller_stated_geometry`` but carrying no ``derived_from`` text
    (an older stamp, or a stamp this route cannot resolve one text for) must not reach the
    resolver's own bare ``ValueError``: the route refuses first, naming the bucket."""
    from tcip_mcp.pipelines.resolution import VALIDATED_EXPLICIT_GEOMETRY

    proj, pred_dir = _make_dense_reviewed_project(tmp_path, tile_size_op={
        "value": 512, "source": "explicit",
        "validated_against": VALIDATED_EXPLICIT_GEOMETRY, "requires_validation": True,
        "validation_kind": "geometry",
    })
    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "bud_opening", "pred_dir": pred_dir, "subject": "bud"})
    assert resp.status_code == 400, resp.text
    # The bucket path travels inside a dict's repr(), which doubles its own backslashes.
    assert repr(pred_dir).strip("'") in resp.json()["detail"]


def test_route_promotes_an_explicit_tile_edge_carrying_its_stamps_own_text(client, tmp_path: Path):
    """A bucket stamped explicit with the real text the producing run composed it with is
    promoted, and the validated row carries that same text forward, never a placeholder."""
    from tcip_mcp.pipelines.resolution import VALIDATED_EXPLICIT_GEOMETRY

    derived_from = "equal to the checkpoint's persisted training tile geometry"
    proj, pred_dir = _make_dense_reviewed_project(tmp_path, tile_size_op={
        "value": 512, "source": "explicit", "derived_from": derived_from,
        "validated_against": VALIDATED_EXPLICIT_GEOMETRY, "requires_validation": True,
        "validation_kind": "geometry",
    })
    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "bud_opening", "pred_dir": pred_dir, "subject": "bud"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is True
    sc = _read_sidecar(pred_dir)
    restamped = sc["operating_point"]["tile_size"]
    assert restamped["source"] == "explicit"
    assert restamped["derived_from"] == derived_from
    assert restamped["validated_against"] == VALIDATED_EXPLICIT_GEOMETRY


def test_route_never_upgrades_an_edge_with_no_accepted_reference_to_a_validated_geometry(
        client, tmp_path: Path):
    """A stamp naming validated_against="false" beside a real tile edge (a stamp nothing in the
    current vocabulary answers for) keeps the edge (re-read back as tile_size_source "recorded")
    and must stay unvalidated through the same re-resolution, never silently promoted to any
    accepted reference nor dropped to None."""
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE

    proj, pred_dir = _make_dense_reviewed_project(tmp_path, tile_size_op={
        "value": 128, "source": "derived",
        "derived_from": "a recorded stamp whose geometry reference the current vocabulary does "
                        "not accept",
        "validated_against": VALIDATED_FALSE, "requires_validation": True,
        "validation_kind": "geometry",
    })
    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "bud_opening", "pred_dir": pred_dir, "subject": "bud"})
    assert resp.status_code == 200, resp.text
    sc = _read_sidecar(pred_dir)
    restamped = sc["operating_point"]["tile_size"]
    assert restamped["validated_against"] == VALIDATED_FALSE
    assert restamped["value"] == 128


def test_route_floors_an_edge_whose_stamp_names_no_geometry_reference(client, tmp_path: Path):
    """A stamp carrying a tile edge and no ``validated_against`` key at all (a stamp shaped
    before the geometry dimension existed) is re-resolved as ``recorded``: the edge is kept, the
    dimension is operative and floored, so a delivery over the bucket still gates on geometry.
    Carrying the stored block forward instead would leave the dimension non-operative and let
    the delivery ship ungated."""
    proj, pred_dir = _make_dense_reviewed_project(tmp_path, tile_size_op={
        "value": 640, "source": "default",
    })
    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "bud_opening", "pred_dir": pred_dir, "subject": "bud"})
    assert resp.status_code == 200, resp.text
    sc = _read_sidecar(pred_dir)
    restamped = sc["operating_point"]["tile_size"]
    assert restamped["value"] == 640
    assert restamped["requires_validation"] is True
    assert restamped["validated_against"] == VALIDATED_FALSE
    from tcip_mcp.pipelines.resolution import tile_size_gate_flag

    assert tile_size_gate_flag(sc["operating_point"]) == VALIDATED_FALSE


def test_route_refuses_conf_censored_and_stamps_honest_placeholder(client, tmp_path: Path):
    # The identical gate refuses a display-floored reference, surfaced honestly, not upgraded.
    proj, pred_dir = _make_project(tmp_path, floored=False)
    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "bud_opening", "pred_dir": pred_dir, "subject": "bud"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is False
    assert "confidence" in body["reason"].lower()
    sc = _read_sidecar(pred_dir)
    assert sc["validated"] is False
    assert sc["validated_reference"] == VALIDATED_FALSE


def test_route_no_completed_reviews(client, tmp_path: Path):
    proj = tmp_path / "proj"
    pred_dir = proj / "predictions" / "model" / "2026-01-01" / "detect"
    pred_dir.mkdir(parents=True, exist_ok=True)
    _write_empty_prediction_document(pred_dir / "A.json")
    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": str(proj), "trait": "bud_opening", "pred_dir": str(pred_dir),
        "subject": "bud"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is False
    assert body["reviewed_image_count"] == 0
    assert "No completed reviews" in body["reason"]
    assert not (pred_dir / "operating_point.json").exists()  # nothing to stamp


def test_route_unknown_trait_is_honest_400(client, tmp_path: Path):
    proj, pred_dir = _make_project(tmp_path, floored=True)
    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "annotations", "pred_dir": pred_dir, "subject": "bud"})
    assert resp.status_code == 400
    assert "not defined for trait" in resp.json()["detail"]


def test_route_honestly_refuses_when_class_id_unresolvable(client, tmp_path: Path):
    # A verdict with no resolvable class identity (class_id=None, e.g. from a bucket whose
    # id_map never recognized its class_name) must make the route refuse loudly (400, naming the
    # real cause) rather than silently stamp VALIDATED_REVIEW_CONFIRMED on a reference the dead
    # `class_id` field would otherwise default to category_id 1 for every entry.
    import tcip_store
    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE

    proj, pred_dir = _make_dense_reviewed_project(tmp_path)
    state_dir = Path(proj) / ".tcip" / "state"
    (shard_key,) = [k for k in tcip_store.keys(REVIEW_VERDICTS_STORE, str(state_dir))
                    if k.parts[1] == "A.jpg"]
    stored = tcip_store.read_versioned(shard_key)
    shard = stored.value
    for entry in shard["state"]["detections"]:
        entry["class_id"] = None
    tcip_store.replace(shard_key, shard, expect=stored.version)

    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "bud_opening", "pred_dir": pred_dir, "subject": "bud"})
    assert resp.status_code == 400
    assert "no resolvable class identity" in resp.json()["detail"]
    # The refusal happens before any stamping: the sidecar's own generation-time `validated: False`
    # is untouched, never silently upgraded to VALIDATED_REVIEW_CONFIRMED on bad data.
    sc = _read_sidecar(pred_dir)
    assert sc["validated"] is False
    assert "validated_reference" not in sc


def test_route_does_not_downgrade_a_validation_a_record_answers_for(client, tmp_path: Path):
    # A later press reports already validated and leaves the earned claim as it stands.
    proj, pred_dir = _make_dense_reviewed_project(tmp_path)
    first = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "bud_opening", "pred_dir": pred_dir, "subject": "bud"})
    assert first.json()["validated"] is True, first.text
    earned = _read_sidecar(pred_dir)

    again = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "bud_opening", "pred_dir": pred_dir, "subject": "bud"})
    assert again.status_code == 200, again.text
    body = again.json()
    assert body["validated"] is True
    assert body["reference"] == "reviewer_confirmed_annotations"
    assert body["buckets_stamped"] == []  # left untouched
    assert _read_sidecar(pred_dir) == earned  # the same claim, the same record, the same timestamp


def test_route_promotes_over_a_validated_stamp_no_record_answers_for(client, tmp_path: Path):
    """A stamp claiming validation that nothing outside the bucket answers for is an assertion, not
    a validation. Reading the raw flag would let a forged or orphaned claim both survive and deny
    the breeder the real validation their review earned, so the route verifies before it decides and
    promotes over it."""
    import tcip_store
    from tcip_mcp.pipelines.resolution import sidecar_key

    proj, pred_dir = _make_dense_reviewed_project(tmp_path)
    stored = tcip_store.read_versioned(sidecar_key(pred_dir, "operating_point"))
    asserted = stored.value
    asserted["validated"] = True
    asserted["validated_reference"] = "held_out_annotations"
    asserted["operating_point"]["conf"] = {"validated_against": "held_out_annotations", "value": 0.31}
    tcip_store.replace(sidecar_key(pred_dir, "operating_point"), asserted, expect=stored.version)

    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "bud_opening", "pred_dir": pred_dir, "subject": "bud"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is True
    assert body["reference"] == "reviewer_confirmed_annotations"  # earned here, not the claim found
    assert body["buckets_stamped"] == [pred_dir]
    sc = _read_sidecar(pred_dir)
    assert sc["validated_reference"] == "reviewer_confirmed_annotations"
    assert sc["validated_by"]["record_digest"]


def test_route_refuses_when_verdicts_belong_to_a_different_producer(client, tmp_path: Path):
    # Verdicts recorded against model A must not validate model B's bucket (retrain -> re-run
    # inference into a new bucket -> press Validate). The sidecar names a different producer than
    # the one the verdicts were recorded against.
    proj, pred_dir = _make_project(
        tmp_path, floored=True, producer_identity=_IDENTITY, sidecar_identity=_OTHER_IDENTITY)
    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "bud_opening", "pred_dir": pred_dir, "subject": "bud"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is False
    assert body["reviewed_image_count"] == 2  # the images were reviewed, just not for this bucket
    sc = _read_sidecar(pred_dir)
    assert sc["validated"] is False


def test_route_refuses_when_no_image_ever_recorded_fn_adjudication(client, tmp_path: Path):
    # Previously-unlabeled images (gt_preexisting=False), reviewed (accept/reject), but never
    # checked for a missed object, must refuse, naming the tool. Uses the dense,
    # realistically-floored fixture (not _make_project's 2-image one) so a real, non-None
    # staged_conf_floor is threaded and conf_censored does not mask this failure.
    proj, pred_dir = _make_dense_reviewed_project(tmp_path, gt_preexisting=False)
    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "bud_opening", "pred_dir": pred_dir, "subject": "bud"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is False
    assert "mark missed object" in body["reason"].lower()
    sc = _read_sidecar(pred_dir)
    assert sc["validated"] is False
