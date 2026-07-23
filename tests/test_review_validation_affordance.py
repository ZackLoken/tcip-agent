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


def _bundle(*, validated: str, sweep: dict) -> ResolvedBundle:
    conf = derived("conf", 0.42, derivation_class="calibration",
                   derived_from="count-unbiased center-match sweep over review verdicts",
                   validated_vs_gt=validated, dataset_scoped=True, dataset_hash="abc", sweep=sweep)
    return ResolvedBundle(trait="catkin", dataset_hash="abc", params={"conf": conf})


def test_describe_validated():
    b = _bundle(validated=VALIDATED_REVIEW_CONFIRMED,
                sweep={"conf_censored": False, "disjoint": True, "passed_holdout": True})
    out = describe_review_validation(b, reviewed_image_count=4)
    assert out["validated"] is True
    assert out["reference"] == VALIDATED_REVIEW_CONFIRMED
    assert out["conf"] == pytest.approx(0.42)
    assert "Validated" in out["reason"] and "4" in out["reason"]


def test_describe_conf_censored():
    b = _bundle(validated=VALIDATED_FALSE, sweep={"conf_censored": True})
    out = describe_review_validation(b, reviewed_image_count=3)
    assert out["validated"] is False
    assert "confidence" in out["reason"].lower()


def test_describe_not_enough_images():
    # One reviewed image -> no holdout was measured (sweep carries no passed_holdout key).
    b = _bundle(validated=VALIDATED_FALSE, sweep={"conf_censored": False, "note": "not held-out"})
    out = describe_review_validation(b, reviewed_image_count=1)
    assert out["validated"] is False
    assert "at least two" in out["reason"]


def test_describe_holdout_bias_failed():
    b = _bundle(validated=VALIDATED_FALSE,
                sweep={"conf_censored": False, "disjoint": True, "passed_holdout": False,
                       "holdout_bias": {"count_bias_mean": 3.0}, "count_bias_tolerance": 1.0})
    out = describe_review_validation(b, reviewed_image_count=6)
    assert out["validated"] is False
    assert "agree" in out["reason"]


# ── route ─────────────────────────────────────────────────────────────────


def _entry(action, cid, gt, pred, conf):
    return {"match_type": "TP", "action": action, "class_id": cid,
            "gt_bbox_norm": gt, "pred_bbox_norm": pred, "conf": conf}


def _write_shard(review_dir: Path, name: str, detections: list) -> None:
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / f"{name}.json").write_text(
        json.dumps({"img_name": name, "state": {"img_status": "completed",
                                                 "detections": detections}}),
        encoding="utf-8")


def _make_project(tmp_path: Path, *, floored: bool) -> tuple[str, str]:
    """A project with two completed-review images + a prediction bucket. ``floored`` includes the
    low-conf tail (the sweep can reach it -> validated); otherwise every conf is above the display
    floor (conf-censored -> refused)."""
    proj = tmp_path / "proj"
    review_dir = proj / ".tcip" / "state" / "review"
    lo = 0.05 if floored else 0.8
    _write_shard(review_dir, "A.jpg", [
        _entry("accepted", 0, [0.25, 0.25, 0.05, 0.05], [0.25, 0.25, 0.05, 0.05], 0.9),
        _entry("rejected", 0, None, [0.75, 0.75, 0.05, 0.05], lo)])
    _write_shard(review_dir, "B.jpg", [
        _entry("accepted", 0, [0.25, 0.25, 0.05, 0.05], [0.25, 0.25, 0.05, 0.05], 0.9),
        _entry("accepted", 0, [0.5, 0.5, 0.05, 0.05], [0.5, 0.5, 0.05, 0.05], lo)])
    pred_dir = proj / "predictions" / "model" / "2026-01-01" / "detect"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("A", "B"):
        (pred_dir / f"{stem}.json").write_text(json.dumps({"objects": []}), encoding="utf-8")
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
    proj, pred_dir = _make_project(tmp_path, floored=True)
    resp = client.post("/api/review/validate_reference", json={
        "project_root": proj, "trait": "catkin", "pred_dir": pred_dir})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validated"] is True
    assert body["reference"] == VALIDATED_REVIEW_CONFIRMED
    assert body["reviewed_image_count"] == 2
    assert pred_dir in body["buckets_stamped"]
    sc = _read_sidecar(pred_dir)
    assert sc["validated"] is True
    assert sc["validated_reference"] == VALIDATED_REVIEW_CONFIRMED
    assert sc["operating_point"]["conf"]["validated_vs_gt"] == VALIDATED_REVIEW_CONFIRMED


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
