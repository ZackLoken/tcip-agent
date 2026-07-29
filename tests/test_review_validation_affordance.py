"""D17 — the breeder GUI affordance that promotes a completed review into a validation reference.

Two layers: (1) ``describe_review_validation`` translates a resolved bundle into a plain-language,
breeder-facing result (torch-free); (2) the ``/api/review/validate_reference`` route runs the IDENTICAL
review->calibration gate and stamps the bucket's ``operating_point.json`` review_confirmed (or an honest
un-shippable placeholder), never a shortcut to validated.
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

# Round 10 (2026-07-29): no built-in traits — seed_catkin_trait_spec (conftest.py) writes a real
# catkin.yml into this test's pinned project root so trait="catkin" call sites keep resolving.
pytestmark = pytest.mark.usefixtures("seed_catkin_trait_spec")


def _bundle(*, validated: str, sweep: dict) -> ResolvedBundle:
    conf = derived("conf", 0.42, derivation_class="calibration",
                   derived_from="count-unbiased center-match sweep over review verdicts",
                   validated_vs_gt=validated, dataset_scoped=True, dataset_hash="abc", sweep=sweep)
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
    # Fix I: the miss-coverage claim is read off the exact-conf holdout_bias entry, not asserted.
    assert "8 of 10" in out["reason"]


def test_describe_conf_censored():
    # The named-failure list (not the raw "conf_censored" key alone) drives the branch, and
    # "passed_holdout" must be present or the "too few images" branch wins first (stage-6 review).
    b = _bundle(validated=VALIDATED_FALSE,
                sweep={"conf_censored": True, "passed_holdout": False, "failures": ["conf_censored"]})
    out = describe_review_validation(b, reviewed_image_count=3)
    assert out["validated"] is False
    assert "confidence" in out["reason"].lower()


def test_describe_conf_floor_mismatch_is_non_gating_provenance_only():
    # Fix D reconciliation (stage-6 review): conf_floor_mismatch is surfaced as provenance
    # (sweep["conf_floor_mismatch"]) but never gates on its own — it never appears in "failures", so
    # a bundle with no OTHER named failure stays Validated even when the floor mismatch is flagged.
    b = _bundle(validated=VALIDATED_REVIEW_CONFIRMED,
                sweep={"conf_censored": False, "conf_floor_mismatch": True, "passed_holdout": True,
                       "failures": [], "holdout_bias": {"tp": 5, "fn": 0}})
    out = describe_review_validation(b, reviewed_image_count=4)
    assert out["validated"] is True


def test_conf_floor_mismatch_never_hijacks_a_real_failure_message():
    # A stronger companion to the above: conf_floor_mismatch=True present ALONGSIDE a real, distinct
    # failure must not divert the message — there is no conf_floor_mismatch-specific branch at all
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
                       "holdout_bias": {"count_bias_mean": 3.0}, "count_bias_tolerance": 1.0,
                       "failures": ["count_bias_exceeds_tolerance"]})
    out = describe_review_validation(b, reviewed_image_count=6)
    assert out["validated"] is False
    assert "agree" in out["reason"]


def test_describe_no_adjudication_coverage():
    # Fix H's own new failure name ("insufficient_adjudication_coverage") — a distinct, honest
    # reason naming the affordance, not a fallthrough to the generic "counts didn't agree" message.
    b = _bundle(validated=VALIDATED_FALSE,
                sweep={"passed_holdout": False, "failures": ["insufficient_adjudication_coverage"]})
    out = describe_review_validation(b, reviewed_image_count=5)
    assert out["validated"] is False
    assert "mark missed object" in out["reason"].lower()


def test_describe_unrecognized_failure_name_is_a_loud_error():
    # Named-failure architecture (cross-cutting): an unmapped failure name must never fall through
    # to the generic message silently — it's a defect in describe_review_validation itself.
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


def _write_shard(review_dir: Path, name: str, detections: list, *, gt_preexisting: bool = True) -> None:
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / f"{name}.json").write_text(
        json.dumps({"img_name": name, "state": {"img_status": "completed",
                                                 "gt_preexisting": gt_preexisting,
                                                 "detections": detections}}),
        encoding="utf-8")


def _write_sidecar(pred_dir: Path, identity: dict, *, generation_conf: float | None = None) -> None:
    sidecar = {
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "experiment_id": identity["experiment_id"],
        "validated": False,
    }
    if generation_conf is not None:
        # Fix D item 4: the conf the bucket's predictions were actually generated/floored at — the
        # route reads this straight off the sidecar (never re-typed) to build staged_conf_floor.
        sidecar["operating_point"] = {"conf": {"value": generation_conf}}
    (pred_dir / "operating_point.json").write_text(json.dumps(sidecar), encoding="utf-8")


def _make_project(tmp_path: Path, *, floored: bool, producer_identity: dict = _IDENTITY,
                  sidecar_identity: dict | None = None,
                  gt_preexisting: bool = True) -> tuple[str, str]:
    """A project with two completed-review images + a prediction bucket. ``floored`` includes the
    low-conf tail (the sweep can reach it -> validated); otherwise every conf is above the display
    floor (conf-censored -> refused). Verdicts are recorded against ``producer_identity``; the
    bucket's own sidecar carries ``sidecar_identity`` (defaults to the SAME identity — a matching,
    scoped reference; Fix G's tests pass a deliberately DIFFERENT one to reproduce the mismatch)."""
    proj = tmp_path / "proj"
    review_dir = proj / ".tcip" / "state" / "review"
    lo = 0.05 if floored else 0.8
    _write_shard(review_dir, "A.jpg", [
        _entry("accepted", 0, [0.25, 0.25, 0.05, 0.05], [0.25, 0.25, 0.05, 0.05], 0.9,
               producer_identity=producer_identity),
        _entry("rejected", 0, None, [0.75, 0.75, 0.05, 0.05], lo,
               producer_identity=producer_identity)], gt_preexisting=gt_preexisting)
    _write_shard(review_dir, "B.jpg", [
        _entry("accepted", 0, [0.25, 0.25, 0.05, 0.05], [0.25, 0.25, 0.05, 0.05], 0.9,
               producer_identity=producer_identity),
        _entry("accepted", 0, [0.5, 0.5, 0.05, 0.05], [0.5, 0.5, 0.05, 0.05], lo,
               producer_identity=producer_identity)], gt_preexisting=gt_preexisting)
    pred_dir = proj / "predictions" / "model" / "2026-01-01" / "detect"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("A", "B"):
        (pred_dir / f"{stem}.json").write_text(json.dumps({"objects": []}), encoding="utf-8")
    _write_sidecar(pred_dir, sidecar_identity or producer_identity)
    return str(proj), str(pred_dir)


def _make_dense_reviewed_project(tmp_path: Path, *, n_images: int = 6, gt_preexisting: bool = True,
                                 producer_identity: dict = _IDENTITY) -> tuple[str, str]:
    """A project with ``n_images`` completed-review images (>= 2 per side of the locked cal/holdout
    split, clearing Fix C's non-degeneracy floor — ``_make_project``'s 2-image fixture cannot, since
    it can only ever produce a single holdout image) and a REALISTIC staged conf floor: every verdict
    carries its own recorded ``conf_threshold`` (Fix D item 4) and the bucket's sidecar carries its
    own recorded generation conf (Fix D), so ``routes/review.py`` threads a real, non-``None``
    ``staged_conf_floor`` end to end rather than always failing closed as conf-censored.

    Every image gets 2 accepted, exactly-matching verdicts at a real detection score (0.9), each
    geometrically distinct from every other image's (so the content-overlap gate never fires on a
    holdout that happens to duplicate calibration's boxes). ``gt_preexisting`` toggles Fix H
    adjudication coverage for every image at once — ``True`` clears it (a genuine pre-existing-GT
    image the breeder reviewed), ``False`` reproduces Fix H's refusal (no evidence a missed object
    was ever checked for).
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
        _write_shard(review_dir, f"{name}.jpg", [
            _entry("accepted", 0, gt1, gt1, 0.9, producer_identity=producer_identity,
                   conf_threshold=conf_threshold),
            _entry("accepted", 0, gt2, gt2, 0.9, producer_identity=producer_identity,
                   conf_threshold=conf_threshold),
        ], gt_preexisting=gt_preexisting)
        (pred_dir / f"{name}.json").write_text(json.dumps({"objects": []}), encoding="utf-8")
    _write_sidecar(pred_dir, producer_identity, generation_conf=conf_threshold)
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
    # K2 (Fix D): routes/review.py DOES thread a real staged_conf_floor into
    # resolve_operating_point_from_review — max(generation_conf, review_conf_threshold), the
    # generation half read off the bucket's own operating_point.json sidecar, the review half read
    # off the verdicts' own recorded conf_threshold. This is the design's mandatory acceptance test
    # for the review path: a realistic, disjoint, count-agreeing, adjudication-covered review
    # reference must actually reach review_confirmed end to end through the route.
    proj, pred_dir = _make_dense_reviewed_project(tmp_path)
    resp = client.post("/api/review/validate_reference", json={
        "project_root": proj, "trait": "catkin", "pred_dir": pred_dir})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is True
    assert body["reference"] == "review_confirmed"
    sc = _read_sidecar(pred_dir)
    assert sc["validated"] is True
    assert sc["validated_reference"] == "review_confirmed"


def test_route_refuses_conf_censored_and_stamps_honest_placeholder(client, tmp_path: Path):
    # The identical gate refuses a display-floored reference — surfaced honestly, not upgraded.
    proj, pred_dir = _make_project(tmp_path, floored=False)
    resp = client.post("/api/review/validate_reference", json={
        "project_root": proj, "trait": "catkin", "pred_dir": pred_dir})
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
        "project_root": str(proj), "trait": "catkin", "pred_dir": str(pred_dir)})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is False
    assert body["reviewed_image_count"] == 0
    assert "No completed reviews" in body["reason"]
    assert not (pred_dir / "operating_point.json").exists()  # nothing to stamp


def test_route_unknown_trait_is_honest_400(client, tmp_path: Path):
    proj, pred_dir = _make_project(tmp_path, floored=True)
    resp = client.post("/api/review/validate_reference", json={
        "project_root": proj, "trait": "annotations", "pred_dir": pred_dir})
    assert resp.status_code == 400
    assert "not defined for trait" in resp.json()["detail"]


def test_route_does_not_downgrade_already_validated(client, tmp_path: Path):
    # A bucket already validated against held-out GT must not be lowered by a (censored) review.
    proj, pred_dir = _make_project(tmp_path, floored=False)
    Path(pred_dir, "operating_point.json").write_text(json.dumps({
        "validated": True,
        "operating_point": {"conf": {"validated_vs_gt": "validated_held_out", "value": 0.31}},
    }), encoding="utf-8")
    resp = client.post("/api/review/validate_reference", json={
        "project_root": proj, "trait": "catkin", "pred_dir": pred_dir})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is True
    assert body["reference"] == "validated_held_out"
    assert body["buckets_stamped"] == []  # left untouched
    sc = _read_sidecar(pred_dir)
    assert sc["validated"] is True  # not downgraded


def test_route_refuses_when_verdicts_belong_to_a_different_producer(client, tmp_path: Path):
    # Fix G's reproduced defect: verdicts recorded against model A must not validate model B's
    # bucket (retrain -> re-run inference into a new bucket -> press Validate). The sidecar names a
    # DIFFERENT producer than the one the verdicts were recorded against.
    proj, pred_dir = _make_project(
        tmp_path, floored=True, producer_identity=_IDENTITY, sidecar_identity=_OTHER_IDENTITY)
    resp = client.post("/api/review/validate_reference", json={
        "project_root": proj, "trait": "catkin", "pred_dir": pred_dir})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is False
    assert body["reviewed_image_count"] == 2  # the images were reviewed — just not for this bucket
    sc = _read_sidecar(pred_dir)
    assert sc["validated"] is False


def test_route_refuses_when_no_image_ever_recorded_fn_adjudication(client, tmp_path: Path):
    # Fix H's reproduced scenario: previously-unlabeled images (gt_preexisting=False), reviewed
    # (accept/reject), but never checked for a missed object — must refuse, naming the tool. Uses
    # the dense, realistically-floored fixture (not _make_project's 2-image one) so a real,
    # non-None staged_conf_floor is threaded and conf_censored does not mask this failure.
    proj, pred_dir = _make_dense_reviewed_project(tmp_path, gt_preexisting=False)
    resp = client.post("/api/review/validate_reference", json={
        "project_root": proj, "trait": "catkin", "pred_dir": pred_dir})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is False
    assert "mark missed object" in body["reason"].lower()
    sc = _read_sidecar(pred_dir)
    assert sc["validated"] is False
