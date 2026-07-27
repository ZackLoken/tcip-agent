"""W1-R2: the review-confirmation reference — validate the count operating point against a
breeder-confirmed sample of the model's own outputs, held to the IDENTICAL gate the held-out-GT
path uses (the shared-reference principle), and blocked by the same conf-censoring guard (G1)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")  # operating_point imports evaluation which imports torch

from tcip_mcp.pipelines.feedback import (  # noqa: E402
    resolve_operating_point_from_review,
    review_reference_hash,
    review_to_records,
)
from tcip_mcp.pipelines.resolution import (  # noqa: E402
    VALIDATED_FALSE,
    VALIDATED_REVIEW_CONFIRMED,
)

_DIMS = {"A.jpg": (400, 400), "B.jpg": (400, 400)}


def _entry(mt, action, cid, gt, pred, conf):
    return {"match_type": mt, "action": action, "class_id": cid,
            "gt_bbox_norm": gt, "pred_bbox_norm": pred, "conf": conf}


def _floored_state():
    """Two disjoint completed images whose predictions include the low-conf tail (floored infer)."""
    a = {"img_status": "completed", "detections": [
        _entry("TP", "accepted", 0, [0.25, 0.25, 0.05, 0.05], [0.25, 0.25, 0.05, 0.05], 0.9),
        _entry("FP", "rejected", 0, None, [0.75, 0.75, 0.05, 0.05], 0.05)]}
    b = {"img_status": "completed", "detections": [
        _entry("TP", "accepted", 0, [0.25, 0.25, 0.05, 0.05], [0.25, 0.25, 0.05, 0.05], 0.9),
        _entry("TP", "accepted", 0, [0.5, 0.5, 0.05, 0.05], [0.5, 0.5, 0.05, 0.05], 0.05)]}
    return {"image": {"A.jpg": a, "B.jpg": b}}


def test_review_to_records_reconstructs_gt_and_dt():
    recs = review_to_records(_floored_state(), image_dims=_DIMS)
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
    ids = {r["image_id"] for r in review_to_records(state, image_dims=_DIMS)}
    assert ids == {"A", "B"}  # a partially-reviewed image is not a confirmed reference


def test_review_confirmed_stamps_when_the_same_gate_passes():
    b = resolve_operating_point_from_review(_floored_state(), "catkin", image_dims=_DIMS)
    conf = b.get("conf")
    # A disjoint, uncensored, count-bias-passing review reference earns review_confirmed (distinct
    # from validated_held_out so provenance records WHICH reference validated) and is shippable.
    assert conf.validated_vs_gt == VALIDATED_REVIEW_CONFIRMED
    assert conf.derived_from == "count-unbiased center-match sweep over review verdicts"
    assert conf.dataset_scoped is True
    assert b.is_shippable is True


def test_conf_censored_review_reference_cannot_validate():
    # Predictions staged at the display floor (all conf >= 0.5) — the sweep can't reach the low-conf
    # tail, so the G1 guard refuses to stamp validated even though the split is disjoint.
    a = {"img_status": "completed", "detections": [
        _entry("TP", "accepted", 0, [0.25, 0.25, 0.05, 0.05], [0.25, 0.25, 0.05, 0.05], 0.9)]}
    b = {"img_status": "completed", "detections": [
        _entry("TP", "accepted", 0, [0.5, 0.5, 0.05, 0.05], [0.5, 0.5, 0.05, 0.05], 0.7)]}
    bundle = resolve_operating_point_from_review({"image": {"A.jpg": a, "B.jpg": b}},
                                                 "catkin", image_dims=_DIMS)
    assert bundle.get("conf").validated_vs_gt == VALIDATED_FALSE
    assert bundle.is_shippable is False


def test_review_reference_hash_scopes_to_the_affirmed_reference():
    recs = review_to_records(_floored_state(), image_dims=_DIMS)
    h1 = review_reference_hash(recs)
    # a different affirmed-gt set is a different reference identity
    recs2 = review_to_records(_floored_state(), image_dims=_DIMS)
    recs2[0]["gt"].append({"category_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0], "iscrowd": 0})
    assert review_reference_hash(recs2) != h1
    assert len(h1) == 16
