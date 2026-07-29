"""Center-match count-unbiased operating-point sweep (count-trait calibration)."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")  # evaluation.py imports torch at module load

from tests._dense_op_fixtures import dense_records  # noqa: E402
from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    gt_class_avg_size,
    pick_count_unbiased,
    pick_f1_max,
    sweep_operating_point,
)

# Round 10 (2026-07-29): no built-in traits — seed_catkin_trait_spec (conftest.py) writes a real
# catkin.yml into this test's pinned project root so resolve_operating_point("catkin", ...) keeps
# resolving by default.
pytestmark = pytest.mark.usefixtures("seed_catkin_trait_spec")

N_IMAGES = 20
OBJECTS_PER_IMAGE = 80


def _box(cx: float, cy: float, s: float = 20.0) -> list[float]:
    return [cx - s / 2, cy - s / 2, s, s]  # xywh centered at (cx, cy)


def _ann(cx, cy, cid=0, score=None):
    a = {"category_id": cid, "bbox": _box(cx, cy)}
    if score is not None:
        a["score"] = score
    return a


def _records(idp="c", *, shift: float = 0.0):
    """Constructed so the count-unbiased conf (0.6) differs from the F1-max conf (0.9).

    Image A: 1 GT; a correct det @0.9 + a spurious far det @0.6.
    Image B: 2 GT; a correct det @0.9 + a hesitant-but-correct det @0.3.
    conf 0.6 -> A over-counts (+1, spurious kept), B under-counts (-1, hesitant dropped) => net bias 0.
    conf 0.9 -> A unbiased, B under-counts (-1) => net bias -0.5 but higher F1 (no spurious FP).

    ``shift`` offsets every GT box's center by that many px (well inside the ~10px center-match
    tolerance derived from these boxes) while leaving the detections in place — used to give a
    holdout fixture genuinely different GT content from calibration's (K1's content-overlap gate
    would otherwise flag a byte-identical-content holdout, differing only by ``image_id``, as a
    clone unable to function as an independent check).
    """
    a = {"width": 400, "height": 400, "image_id": f"{idp}_a",
         "gt": [_ann(100 + shift, 100)],
         "dt": [_ann(100, 100, score=0.9), _ann(300, 300, score=0.6)]}
    b = {"width": 400, "height": 400, "image_id": f"{idp}_b",
         "gt": [_ann(100 + shift, 100), _ann(200 + shift, 200)],
         "dt": [_ann(100, 100, score=0.9), _ann(200, 200, score=0.3)]}
    return [a, b]


def _good_cal_holdout(*, shift: float = 5.0):
    """A dense, realistic (rule 17) reference: a good detector with one low-conf spurious detection
    per image (a realistic false-positive profile) that vanishes once conf crosses it — the
    count-unbiased pick lands at the high, correct-match score (0.9), comfortably above a real
    calibration floor, with zero bias/dispersion and full recall/precision on the holdout.
    """
    miss = [0] * N_IMAGES
    fp = [1] * N_IMAGES
    cal = dense_records(n_images=N_IMAGES, objects_per_image=OBJECTS_PER_IMAGE, id_prefix="c",
                        miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05)
    hold = dense_records(n_images=N_IMAGES, objects_per_image=OBJECTS_PER_IMAGE, id_prefix="h",
                         shift=shift, miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05)
    return cal, hold


def test_gt_class_avg_size_derived_from_data():
    assert gt_class_avg_size(_records()) == pytest.approx(20.0)


def test_count_unbiased_differs_from_f1_max():
    recs = _records()
    tol = 0.5 * gt_class_avg_size(recs)  # derived tolerance = half class avg size
    sweep = sweep_operating_point(recs, tolerance=tol)

    cu = pick_count_unbiased(sweep)
    f1m = pick_f1_max(sweep)
    assert cu == pytest.approx(0.6)
    assert f1m == pytest.approx(0.0)  # F1 peaks at low conf (hesitant true det lifts recall)
    assert cu != f1m  # the whole point: a count phenotype is not optimized by F1

    by_conf = {round(c["conf"], 6): c for c in sweep["curve"]}
    # at the count-unbiased point the net per-image count bias vanishes...
    assert by_conf[round(cu, 6)]["count_bias_mean"] == pytest.approx(0.0)
    # ...while the F1-max point over-counts (keeps A's spurious det for recall's sake).
    assert by_conf[round(f1m, 6)]["count_bias_mean"] == pytest.approx(0.5)


def test_center_match_respects_tolerance():
    # a correct detection just outside tolerance must not count as a hit
    recs = [{"width": 400, "height": 400, "gt": [_ann(100, 100)],
             "dt": [_ann(100 + 100, 100, score=0.9)]}]  # 100px off, tolerance ~10
    sweep = sweep_operating_point(recs, tolerance=0.5 * gt_class_avg_size(recs))
    at0 = sweep["curve"][0]  # conf=0.0 is always the first (lowest) grid point
    assert at0["conf"] == pytest.approx(0.0)
    assert at0["tp"] == 0 and at0["fp"] == 1 and at0["fn"] == 1  # miss + false positive


def test_sweep_curve_carries_dispersion_and_reference_size_fields():
    # Fix B/C: every curve entry now also carries count_error_p90 / count_bias_std / n_images,
    # computed from the SAME per-image biases list, not a second pass over the data.
    recs = dense_records(n_images=4, objects_per_image=10,
                         miss_pattern=[0, 1, 0, 2], fp_pattern=[0, 0, 1, 0])
    sweep = sweep_operating_point(recs, tolerance=0.5 * gt_class_avg_size(recs))
    at09 = next(c for c in sweep["curve"] if c["conf"] == pytest.approx(0.9))
    # biases = fp - fn per image = [0, -1, 1, -2]
    assert at09["n_images"] == 4
    assert at09["count_bias_mean"] == pytest.approx(-0.5)
    assert at09["count_bias_std"] == pytest.approx(1.2909944, abs=1e-5)
    assert at09["count_error_p90"] == pytest.approx(1.7, abs=1e-6)


# --- resolve_operating_point + the in-model seam ---

def _two_stage():
    from types import SimpleNamespace
    return SimpleNamespace(detector=SimpleNamespace(
        roi_heads=SimpleNamespace(score_thresh=0.05, nms_thresh=0.5, detections_per_img=100)))


def _one_stage():
    from types import SimpleNamespace
    return SimpleNamespace(detector=SimpleNamespace(score_thresh=0.2, nms_thresh=0.6, detections_per_img=100))


def test_set_detector_operating_point_two_stage():
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point
    m = _two_stage()
    applied = set_detector_operating_point(m, score_thresh=0.4, nms_thresh=0.3, detections_per_img=300)
    assert m.detector.roi_heads.score_thresh == 0.4
    assert m.detector.roi_heads.nms_thresh == 0.3
    assert m.detector.roi_heads.detections_per_img == 300
    assert applied == {"score_thresh": 0.4, "nms_thresh": 0.3, "detections_per_img": 300}


def test_set_detector_operating_point_one_stage():
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point
    m = _one_stage()
    set_detector_operating_point(m, score_thresh=0.4, nms_thresh=0.35)
    assert m.detector.score_thresh == 0.4 and m.detector.nms_thresh == 0.35


def test_max_dets_from_density_scales_above_floor():
    # 1.5x the p99 GT-per-image count exceeds the 100 floor, so max_dets scales with
    # density instead of pinning to the floor (a dense scene must not be truncated).
    from tcip_mcp.pipelines.operating_point import _max_dets_from_density
    records = [{"gt": [_ann(0, 0)] * 80} for _ in range(20)]
    assert _max_dets_from_density(records) == 120  # ceil(1.5 * 80)


def test_max_dets_from_density_floors_sparse_scenes():
    from tcip_mcp.pipelines.operating_point import _max_dets_from_density
    records = [{"gt": [_ann(0, 0)] * 2} for _ in range(20)]
    assert _max_dets_from_density(records) == 100  # floor, not ceil(1.5 * 2)


def test_resolve_operating_point_validated_with_holdout():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    cal, hold = _good_cal_holdout()
    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=cal, holdout_records=hold,
                                staged_conf_floor=0.01)
    conf = b.get("conf")
    assert conf.derivation_class == "calibration"
    assert conf.validated_vs_gt == "validated_held_out"
    assert b.is_shippable
    assert conf.value == pytest.approx(0.9)  # count-unbiased pick: bias vanishes once the low-conf FP drops
    assert b.get("max_dets").value >= 100  # derived from GT density
    sweep = conf.sweep
    assert sweep["failures"] == []
    assert sweep["content_overlap_frac"] == pytest.approx(0.0)  # genuinely distinct holdout content
    assert sweep["train_disjointness"] == {"checked": False, "unresolvable": False,
                                           "leaked_groups": [], "leaked_stems": [],
                                           "group_check": None}
    assert set(sweep["calibration_image_ids"]) == {f"c_{i}" for i in range(N_IMAGES)}
    assert set(sweep["holdout_image_ids"]) == {f"h_{i}" for i in range(N_IMAGES)}


def test_resolve_operating_point_overlapping_holdout_not_validated():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    # same image ids in calibration and holdout -> not a real held-out split -> not validated
    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=_records("c"), holdout_records=_records("c"))
    assert b.get("conf").validated_vs_gt == "false"
    assert not b.is_shippable


def test_resolve_operating_point_missing_image_ids_fails_closed():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    # Records with no image_id: identity is unverifiable, so a held-out claim can't be proven —
    # the same records as cal+holdout must not be stamped validated (the firewall fails closed).
    # Before, empty id-sets made `disjoint` True, so an in-sample "holdout" passed as validated.
    recs = [{"width": 400, "height": 400, "gt": [_ann(100, 100)],
             "dt": [_ann(100, 100, score=0.9)]}]  # no image_id key
    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=recs, holdout_records=recs)
    assert b.get("conf").validated_vs_gt == "false"
    assert not b.is_shippable


def test_resolve_operating_point_biased_holdout_is_unshippable():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    cal, _ = _good_cal_holdout()
    # a dense holdout with a REAL, consistent per-image miss (not just a sparse fixture's one-off
    # spread) — count bias -3/image, well beyond tolerance regardless of dispersion/SE.
    biased_hold = dense_records(n_images=N_IMAGES, objects_per_image=OBJECTS_PER_IMAGE, id_prefix="h",
                                shift=5.0, miss_pattern=[3] * N_IMAGES, fp_pattern=[0] * N_IMAGES,
                                score=0.9)
    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=cal, holdout_records=biased_hold,
                                staged_conf_floor=0.01)
    # measured on the disjoint split but FAILED (bias > tolerance) -> not validated, firewall holds
    assert b.get("conf").validated_vs_gt == "false"
    assert not b.is_shippable
    assert "count_bias_exceeds_tolerance" in b.get("conf").sweep["failures"]


def test_resolve_operating_point_calibrated_but_no_holdout_is_unshippable():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    b = resolve_operating_point("catkin", dataset_hash="h1", calibration_records=_records())
    assert b.get("conf").validated_vs_gt == "false"
    assert not b.is_shippable


# --- K1: content-overlap + train-disjointness gates ---

def test_resolve_operating_point_content_clone_holdout_is_false():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    # Same GT content as calibration (only image_id differs, no shift) -> disjoint by image_id but
    # the holdout can't function as an independent check; the content-overlap gate must refuse it.
    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=_records("c"), holdout_records=_records("h"))
    conf = b.get("conf")
    assert conf.validated_vs_gt == "false"
    assert not b.is_shippable
    assert conf.sweep["content_overlap_frac"] == pytest.approx(1.0)


def test_resolve_operating_point_train_disjointness_fires(tmp_path, monkeypatch):
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    exp_dir = tmp_path / ".tcip" / "experiments" / "exp1"
    exp_dir.mkdir(parents=True)
    (exp_dir / "split.json").write_text(
        json.dumps({"train": ["a_0_0", "a_0_1"], "group_by": "tile_prefix"}), encoding="utf-8")

    # Calibration/holdout share tile group "a" (stem "a_0_2") with the training split above.
    cal = [{"width": 400, "height": 400, "image_id": "a_0_2", "gt": [_ann(100, 100)],
            "dt": [_ann(100, 100, score=0.9), _ann(300, 300, score=0.6)]}]
    hold = [{"width": 400, "height": 400, "image_id": "a_0_3", "gt": [_ann(100, 100 + 5)],
             "dt": [_ann(100, 100, score=0.9)]}]
    b = resolve_operating_point("catkin", dataset_hash="h1", calibration_records=cal,
                                holdout_records=hold, experiment_id="exp1")
    conf = b.get("conf")
    assert conf.validated_vs_gt == "false"
    assert conf.sweep["train_disjointness"]["leaked_groups"] == ["a"]


def test_resolve_operating_point_train_disjointness_unresolvable_when_split_missing(tmp_path, monkeypatch):
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    # A KNOWN experiment_id whose split.json can't be read fails closed (unresolvable), unlike the
    # experiment_id=None case (a foreign/unregistered checkpoint), per the owner decision.
    cal, hold = _good_cal_holdout()
    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=cal, holdout_records=hold,
                                staged_conf_floor=0.01, experiment_id="does-not-exist")
    conf = b.get("conf")
    assert conf.validated_vs_gt == "false"
    assert conf.sweep["train_disjointness"] == {"checked": False, "unresolvable": True,
                                                 "leaked_groups": [], "leaked_stems": [],
                                                 "group_check": None}
    assert "train_disjointness_unresolvable" in conf.sweep["failures"]


def test_resolve_operating_point_train_disjointness_resolvable_no_leak_still_validates(tmp_path, monkeypatch):
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    exp_dir = tmp_path / ".tcip" / "experiments" / "exp2"
    exp_dir.mkdir(parents=True)
    (exp_dir / "split.json").write_text(
        json.dumps({"train": ["z_0_0", "z_0_1"], "group_by": "tile_prefix"}), encoding="utf-8")

    # Calibration/holdout use id prefixes "c"/"h" — disjoint from training's "z" group.
    cal, hold = _good_cal_holdout()
    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=cal, holdout_records=hold,
                                staged_conf_floor=0.01, experiment_id="exp2")
    conf = b.get("conf")
    assert conf.validated_vs_gt == "validated_held_out"
    assert b.is_shippable
    assert conf.sweep["train_disjointness"] == {"checked": True, "unresolvable": False,
                                                 "leaked_groups": [], "leaked_stems": [],
                                                 "group_check": "performed"}


def test_resolve_operating_point_no_gt_placeholder_unshippable():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tcip_mcp.pipelines.resolution import UnvalidatedOperatingPointError
    b = resolve_operating_point("catkin", dataset_hash="hX")
    with pytest.raises(UnvalidatedOperatingPointError):
        _ = b.value("conf")  # firewall
    assert b.get("conf").unvalidated_value(acknowledge_unvalidated=True) == 0.5


def _overlap_records(idp="d"):
    """Calibration records whose GT boxes overlap (20px, offset 8px), so cross_tile_nms is derivable."""
    boxes = [_box(100, 100), _box(108, 100), _box(116, 100)]  # neighbor IoU ~0.43
    return [{"width": 400, "height": 400, "image_id": f"{idp}_{i}",
             "gt": [{"category_id": 1, "bbox": bx} for bx in boxes],
             "dt": [{"category_id": 1, "bbox": bx, "score": 0.9} for bx in boxes]}
            for i in range(2)]


def test_resolve_operating_point_derives_cross_tile_nms_from_gt():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    b = resolve_operating_point("catkin", dataset_hash="h1", calibration_records=_overlap_records())
    p = b.get("cross_tile_nms")
    assert p.source == "derived"
    assert p.derivation_class == "distribution"
    assert "neighbor-IoU" in p.derived_from
    assert 0.2 <= p.value <= 0.8
    assert p.value == pytest.approx(0.4286 + 0.05, abs=1e-2)  # p99 of the GT neighbor-IoU tail + margin


def test_resolve_operating_point_explicit_cross_tile_nms_not_labeled_derived():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    # An explicit override is honest even when overlapping GT was present to derive from.
    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=_overlap_records(), cross_tile_nms=0.55)
    p = b.get("cross_tile_nms")
    assert p.source == "explicit"
    assert p.value == pytest.approx(0.55)
    assert p.derived_from == "caller override"  # not a derivation costume on a caller-supplied number
    assert "neighbor-IoU" not in p.derived_from


def test_resolve_operating_point_cross_tile_nms_honest_default_when_underivable():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    # No GT at all -> honest default, never a derivation label on an underived number.
    p_no_gt = resolve_operating_point("catkin", dataset_hash="h1").get("cross_tile_nms")
    assert p_no_gt.source == "default"
    assert "neighbor-IoU" not in p_no_gt.derived_from
    # Sparse, non-overlapping GT is likewise underivable -> still an honest default.
    p_sparse = resolve_operating_point("catkin", dataset_hash="h1",
                                       calibration_records=_records("c")).get("cross_tile_nms")
    assert p_sparse.source == "default"


def test_resolve_operating_point_tile_size_derived():
    """K10 finding 3 residual: a bare truthy tile_size with no source claim used to be inferred as
    "derived" unconditionally (`if tile_size: derived(...)`) — so a fabricated fallback value (e.g.
    the checkpoint had no persisted geometry, and 640 was substituted one layer up) could be
    stamped "derived from persisted training geometry" when nothing was actually derived. The
    caller must now say which it was via ``tile_size_source``; omitting it is honestly "default"."""
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    b_no_claim = resolve_operating_point("catkin", dataset_hash="h1", tile_size=640)
    assert b_no_claim.get("tile_size").source == "default"
    assert b_no_claim.get("tile_size").value == 640

    b_derived = resolve_operating_point(
        "catkin", dataset_hash="h1", tile_size=640, tile_size_source="derived")
    assert b_derived.get("tile_size").source == "derived"
    assert b_derived.get("tile_size").value == 640


def test_classification_metrics_per_class_and_bias():
    from tcip_mcp.pipelines.training.evaluation import classification_metrics
    gt = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])    # 6 dormant, 4 elongated
    pred = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # classifier predicts 6 as elongated
    m = classification_metrics(pred, gt, num_classes=2)
    assert m["per_class"][1]["support"] == 4
    # over-predicting the elongated class inflates the elongated fraction: bias (6-4)/4 = +0.5
    assert m["count_bias"][1] == pytest.approx(0.5)
    assert "accuracy" in m and "f1" in m  # existing keys preserved (additive)
