"""The breeder GUI affordance that promotes a completed review into a validation reference.

Two layers: (1) ``describe_review_validation`` translates a resolved bundle into a plain-language,
breeder-facing result (torch-free); (2) the ``/api/review/validate_reference`` route runs the
identical review->calibration gate and stamps the bucket's ``operating_point.json``
review_confirmed (or an honest un-shippable placeholder), never a shortcut to validated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcip_mcp.pipelines.feedback import describe_review_validation
from tcip_mcp.pipelines.resolution import (
    VALIDATED_FALSE,
    VALIDATED_REVIEW_CONFIRMED,
    ResolvedBundle,
    derived,
)

# No built-in traits: seed_catkin_trait_spec (conftest.py) writes a real catkin.yml into this
# test's pinned project root so trait="catkin" call sites keep resolving.
pytestmark = pytest.mark.usefixtures("seed_catkin_trait_spec")


def _bundle(*, validated: str, sweep: dict) -> ResolvedBundle:
    conf = derived("conf", 0.42, requires_validation=True, validation_kind="annotations",
                   derived_from="count-unbiased center-match sweep over review verdicts",
                   validated_against=validated, dataset_scoped=True, dataset_hash="abc", sweep=sweep)
    return ResolvedBundle(trait="catkin", dataset_hash="abc", params={"conf": conf})


def test_describe_validated():
    b = _bundle(validated=VALIDATED_REVIEW_CONFIRMED,
                sweep={"conf_censored": False, "disjoint": True, "passed_holdout": True,
                       "failures": [], "holdout_bias": {"tp": 8, "fn": 2}})
    out = describe_review_validation(b, reviewed_image_count=4)
    assert out["validated"] is True
    assert out["reference"] == VALIDATED_REVIEW_CONFIRMED
    assert out["conf"] == pytest.approx(0.42)
    assert "Validated" in out["reason"] and "4" in out["reason"]
    # The miss-coverage claim is read off the exact-conf holdout_bias entry, not asserted.
    assert "8 of 10" in out["reason"]


def test_describe_conf_censored():
    # The named-failure list (not the raw "conf_censored" key alone) drives the branch, and
    # "passed_holdout" must be present or the "too few images" branch wins first.
    b = _bundle(validated=VALIDATED_FALSE,
                sweep={"conf_censored": True, "passed_holdout": False, "failures": ["conf_censored"]})
    out = describe_review_validation(b, reviewed_image_count=3)
    assert out["validated"] is False
    assert "confidence" in out["reason"].lower()


def test_describe_conf_floor_mismatch_is_non_gating_provenance_only():
    # conf_floor_mismatch is surfaced as provenance (sweep["conf_floor_mismatch"]) but never gates
    # on its own; it never appears in "failures", so a bundle with no other named failure stays
    # Validated even when the floor mismatch is flagged.
    b = _bundle(validated=VALIDATED_REVIEW_CONFIRMED,
                sweep={"conf_censored": False, "conf_floor_mismatch": True, "passed_holdout": True,
                       "failures": [], "holdout_bias": {"tp": 5, "fn": 0}})
    out = describe_review_validation(b, reviewed_image_count=4)
    assert out["validated"] is True


def test_conf_floor_mismatch_never_hijacks_a_real_failure_message():
    # A stronger companion to the above: conf_floor_mismatch=True present alongside a real, distinct
    # failure must not divert the message; there is no conf_floor_mismatch-specific branch at all
    # (it is non-gating provenance only), so the real failure's own message must still surface.
    b = _bundle(validated=VALIDATED_FALSE,
                sweep={"conf_censored": False, "disjoint": True, "conf_floor_mismatch": True,
                       "passed_holdout": False,
                       "failures": ["count_bias_exceeds_tolerance"]})
    out = describe_review_validation(b, reviewed_image_count=4)
    assert out["validated"] is False
    assert "agree" in out["reason"]


def test_describe_not_enough_images():
    # One reviewed image -> no holdout was measured (sweep carries no passed_holdout key).
    b = _bundle(validated=VALIDATED_FALSE, sweep={"conf_censored": False, "note": "not held-out"})
    out = describe_review_validation(b, reviewed_image_count=1)
    assert out["validated"] is False
    assert "at least two" in out["reason"]


def test_describe_holdout_bias_failed():
    b = _bundle(validated=VALIDATED_FALSE,
                sweep={"conf_censored": False, "disjoint": True, "passed_holdout": False,
                       "holdout_bias": {"count_bias_mean": 3.0}, "count_bias_tolerance_frac": 1.0,
                       "failures": ["count_bias_exceeds_tolerance"]})
    out = describe_review_validation(b, reviewed_image_count=6)
    assert out["validated"] is False
    assert "agree" in out["reason"]


def test_describe_no_adjudication_coverage():
    # "insufficient_adjudication_coverage" is a distinct, honest reason naming the affordance, not
    # a fallthrough to the generic "counts didn't agree" message.
    b = _bundle(validated=VALIDATED_FALSE,
                sweep={"passed_holdout": False, "failures": ["insufficient_adjudication_coverage"]})
    out = describe_review_validation(b, reviewed_image_count=5)
    assert out["validated"] is False
    assert "mark missed object" in out["reason"].lower()


def test_describe_reports_every_applicable_failure_not_just_the_first():
    # A breeder who hits two blockers at once (e.g. an all-negative split side and a per-class
    # bias failure) must see both in one pass, not fix the first, resubmit, and only then discover
    # the second. _FAILURE_MESSAGES lists insufficient_calibration_gt before
    # count_bias_exceeds_tolerance, so both messages must appear, in that order.
    b = _bundle(validated=VALIDATED_FALSE,
                sweep={"conf_censored": False, "disjoint": True, "passed_holdout": False,
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
                    sweep={"conf_censored": False, "disjoint": True, "passed_holdout": False,
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
                sweep={"conf_censored": False, "disjoint": True, "passed_holdout": False,
                       "failures": ["some_future_fix_g_or_h_failure_not_yet_mapped"]})
    with pytest.raises(AssertionError, match="unrecognized"):
        describe_review_validation(b, reviewed_image_count=2)


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
    """
    from tcip_annotation.review_engine import bucket_dirname
    from tcip_mcp.prediction_buckets import bucket_key_of

    digest = _pred_digest(bucket_dir, name)
    entries = [{**d, "producer_identity": {**d["producer_identity"], "prediction_digest": digest}}
               if isinstance(d.get("producer_identity"), dict) else d
               for d in detections]
    bucket = bucket_key_of(bucket_dir)
    shard_dir = review_dir / bucket_dirname(bucket)
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / f"{name}.json").write_text(
        json.dumps({"bucket": bucket, "img_name": name,
                    "state": {"img_status": "completed",
                              "gt_preexisting": gt_preexisting,
                              "detections": entries}}),
        encoding="utf-8")


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
    }
    (pred_dir / "operating_point.json").write_text(json.dumps(sidecar), encoding="utf-8")


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
        (pred_dir / f"{stem}.json").write_text(json.dumps({"objects": []}), encoding="utf-8")
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
        (pred_dir / f"{name}.json").write_text(json.dumps({"objects": []}), encoding="utf-8")
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
    return TestClient(app)


def _read_sidecar(pred_dir: str) -> dict:
    return json.loads((Path(pred_dir) / "operating_point.json").read_text(encoding="utf-8"))


def test_route_validates_and_stamps_review_confirmed(client, tmp_path: Path):
    # routes/review.py threads a real staged_conf_floor into resolve_operating_point_from_review:
    # max(generation_conf, review_conf_threshold), the generation half read off the bucket's own
    # operating_point.json sidecar, the review half read off the verdicts' own recorded
    # conf_threshold. This is the design's mandatory acceptance test for the review path: a
    # realistic, disjoint, count-agreeing, adjudication-covered review reference must actually
    # reach review_confirmed end to end through the route.
    proj, pred_dir = _make_dense_reviewed_project(tmp_path)
    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "catkin", "pred_dir": pred_dir})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is True
    assert body["reference"] == "reviewer_confirmed_annotations"
    sc = _read_sidecar(pred_dir)
    assert sc["validated"] is True
    assert sc["validated_reference"] == "reviewer_confirmed_annotations"
    # The claim is earned: a record outside the bucket answers for it, under the trait it names.
    assert sc["trait"] == "catkin"
    from tcip_mcp.experiments import find_validation
    from tcip_mcp.pipelines.resolution import verify_stamp_binding

    pointer = sc["validated_by"]
    row = find_validation(pointer["experiment_id"], pointer["record_digest"])
    assert row is not None
    assert row["trait"] == "catkin"
    assert row["reference_identity"]["stated_values"]["review_image_count"] == 6
    assert verify_stamp_binding(sc, pred_dir, document="operating_point").ok is True


def test_route_validates_a_review_that_includes_a_confirmed_negative(client, tmp_path: Path):
    """An image the bucket predicted nothing for is reviewed by marking it Complete. The verdict
    records the absence of a prediction document as the value it is, and the promotion compares that
    absence against the absence still on disk rather than skipping the comparison, so a review
    holding one still earns its record."""
    proj, pred_dir = _make_dense_reviewed_project(tmp_path)
    negative = client.post("/api/review/mark_complete", json={
        "dataset_root": proj, "image_name": "Z.jpg", "pred_dir": pred_dir})
    assert negative.status_code == 200, negative.text
    shard = json.loads(
        next(iter(sorted((Path(proj) / ".tcip" / "state" / "review").rglob("Z.jpg.json"))))
        .read_text(encoding="utf-8"))
    assert shard["state"]["producer_identity"]["prediction_digest"] is None
    assert shard["state"]["adjudication_covered"] is True

    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "catkin", "pred_dir": pred_dir})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is True
    assert body["buckets_stamped"] == [pred_dir]


def test_route_with_no_pred_dir_stamps_nothing_and_says_why(client, tmp_path: Path):
    # A dataset root with no bucket is a shape the API accepts and answers honestly.
    proj, _ = _make_dense_reviewed_project(tmp_path)
    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "catkin"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is False
    assert body["buckets_stamped"] == []
    assert "No predictions are selected" in body["reason"]


def test_route_never_upgrades_a_native_ratio_tile_size_to_persisted_geometry(client, tmp_path: Path):
    """A bucket produced by the native-size ratio tier stamps tile_size with
    source="derived" (the same source a real persisted-training-geometry stamp uses) but
    validated_against=false: the two bases are distinguishable only by validated_against, never
    by source alone. This route re-derives tile_size_source off the bucket's own sidecar to
    thread through resolve_operating_point_from_review; if it read the bare source field it
    would silently re-resolve a native-ratio bucket as validated persisted geometry the moment a
    breeder reviews it, defeating the platform's own decision that native-ratio never clears a
    delivery gate on its own."""
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE

    proj, pred_dir = _make_dense_reviewed_project(tmp_path, tile_size_op={
        "value": 128, "source": "derived",
        "derived_from": "native-size ratio (not an independently validated geometry basis)",
        "validated_against": VALIDATED_FALSE, "requires_validation": True,
        "validation_kind": "geometry",
    })
    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "catkin", "pred_dir": pred_dir})
    assert resp.status_code == 200, resp.text
    sc = _read_sidecar(pred_dir)
    restamped = sc["operating_point"]["tile_size"]
    assert restamped["validated_against"] == VALIDATED_FALSE
    assert restamped["value"] == 128


def test_route_refuses_conf_censored_and_stamps_honest_placeholder(client, tmp_path: Path):
    # The identical gate refuses a display-floored reference, surfaced honestly, not upgraded.
    proj, pred_dir = _make_project(tmp_path, floored=False)
    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "catkin", "pred_dir": pred_dir})
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
    (pred_dir / "A.json").write_text(json.dumps({"objects": []}), encoding="utf-8")
    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": str(proj), "trait": "catkin", "pred_dir": str(pred_dir)})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is False
    assert body["reviewed_image_count"] == 0
    assert "No completed reviews" in body["reason"]
    assert not (pred_dir / "operating_point.json").exists()  # nothing to stamp


def test_route_unknown_trait_is_honest_400(client, tmp_path: Path):
    proj, pred_dir = _make_project(tmp_path, floored=True)
    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "annotations", "pred_dir": pred_dir})
    assert resp.status_code == 400
    assert "not defined for trait" in resp.json()["detail"]


def test_route_honestly_refuses_when_class_id_unresolvable(client, tmp_path: Path):
    # A verdict with no resolvable class identity (class_id=None, e.g. from a bucket whose
    # id_map never recognized its class_name) must make the route refuse loudly (400, naming the
    # real cause) rather than silently stamp VALIDATED_REVIEW_CONFIRMED on a reference the dead
    # `class_id` field would otherwise default to category_id 1 for every entry.
    proj, pred_dir = _make_dense_reviewed_project(tmp_path)
    review_dir = Path(proj) / ".tcip" / "state" / "review"
    shard_path = next(iter(sorted(review_dir.rglob("A.jpg.json"))))
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    for entry in shard["state"]["detections"]:
        entry["class_id"] = None
    shard_path.write_text(json.dumps(shard), encoding="utf-8")

    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "catkin", "pred_dir": pred_dir})
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
        "dataset_root": proj, "trait": "catkin", "pred_dir": pred_dir})
    assert first.json()["validated"] is True, first.text
    earned = _read_sidecar(pred_dir)

    again = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "catkin", "pred_dir": pred_dir})
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
    proj, pred_dir = _make_dense_reviewed_project(tmp_path)
    asserted = _read_sidecar(pred_dir)
    asserted["validated"] = True
    asserted["validated_reference"] = "held_out_annotations"
    asserted["operating_point"]["conf"] = {"validated_against": "held_out_annotations", "value": 0.31}
    Path(pred_dir, "operating_point.json").write_text(json.dumps(asserted), encoding="utf-8")

    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": proj, "trait": "catkin", "pred_dir": pred_dir})
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
        "dataset_root": proj, "trait": "catkin", "pred_dir": pred_dir})
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
        "dataset_root": proj, "trait": "catkin", "pred_dir": pred_dir})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is False
    assert "mark missed object" in body["reason"].lower()
    sc = _read_sidecar(pred_dir)
    assert sc["validated"] is False
