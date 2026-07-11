"""Center-match count-unbiased operating-point sweep (count-trait calibration)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # evaluation.py imports torch at module load

from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    count_bias_at,
    gt_class_avg_size,
    pick_count_unbiased,
    pick_f1_max,
    sweep_operating_point,
)


def _box(cx: float, cy: float, s: float = 20.0) -> list[float]:
    return [cx - s / 2, cy - s / 2, s, s]  # xywh centered at (cx, cy)


def _ann(cx, cy, cid=0, score=None):
    a = {"category_id": cid, "bbox": _box(cx, cy)}
    if score is not None:
        a["score"] = score
    return a


def _records(idp="c"):
    """Constructed so the count-unbiased conf (0.6) differs from the F1-max conf (0.9).

    Image A: 1 GT; a correct det @0.9 + a spurious far det @0.6.
    Image B: 2 GT; a correct det @0.9 + a hesitant-but-correct det @0.3.
    conf 0.6 -> A over-counts (+1, spurious kept), B under-counts (-1, hesitant dropped) => net bias 0.
    conf 0.9 -> A unbiased, B under-counts (-1) => net bias -0.5 but higher F1 (no spurious FP).
    """
    a = {"width": 400, "height": 400, "image_id": f"{idp}_a",
         "gt": [_ann(100, 100)],
         "dt": [_ann(100, 100, score=0.9), _ann(300, 300, score=0.6)]}
    b = {"width": 400, "height": 400, "image_id": f"{idp}_b",
         "gt": [_ann(100, 100), _ann(200, 200)],
         "dt": [_ann(100, 100, score=0.9), _ann(200, 200, score=0.3)]}
    return [a, b]


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

    # at the count-unbiased point the net per-image count bias vanishes...
    assert count_bias_at(sweep, cu)["count_bias_mean"] == pytest.approx(0.0)
    # ...while the F1-max point over-counts (keeps A's spurious det for recall's sake).
    assert count_bias_at(sweep, f1m)["count_bias_mean"] == pytest.approx(0.5)


def test_center_match_respects_tolerance():
    # a correct detection just outside tolerance must NOT count as a hit
    recs = [{"width": 400, "height": 400, "gt": [_ann(100, 100)],
             "dt": [_ann(100 + 100, 100, score=0.9)]}]  # 100px off, tolerance ~10
    sweep = sweep_operating_point(recs, tolerance=0.5 * gt_class_avg_size(recs))
    at0 = count_bias_at(sweep, 0.0)
    assert at0["tp"] == 0 and at0["fp"] == 1 and at0["fn"] == 1  # miss + false positive


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


def test_resolve_operating_point_validated_with_holdout():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=_records("c"), holdout_records=_records("h"))
    conf = b.get("conf")
    assert conf.derivation_class == "calibration"
    assert conf.validated_vs_gt == "validated_held_out"
    assert b.is_shippable
    assert conf.value == pytest.approx(0.6)  # count-unbiased pick
    assert b.get("max_dets").value >= 100  # derived from GT density


def _biased_holdout():
    # each image: 1 GT + 2 spurious far high-conf detections, so the count over-counts by ~2 at the
    # calibration-chosen conf, i.e. it fails on this held-out split.
    return [{"width": 400, "height": 400, "image_id": f"h_{i}", "gt": [_ann(100, 100)],
             "dt": [_ann(100, 100, score=0.9), _ann(300, 300, score=0.9), _ann(50, 300, score=0.9)]}
            for i in range(3)]


def test_resolve_operating_point_overlapping_holdout_not_validated():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    # same image ids in calibration and holdout -> not a real held-out split -> not validated
    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=_records("c"), holdout_records=_records("c"))
    assert b.get("conf").validated_vs_gt == "false"
    assert not b.is_shippable


def test_resolve_operating_point_biased_holdout_is_unshippable():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=_records(), holdout_records=_biased_holdout())
    # measured on the disjoint split but FAILED (bias > tolerance) -> not validated, firewall holds
    assert b.get("conf").validated_vs_gt == "false"
    assert not b.is_shippable


def test_resolve_operating_point_calibrated_but_no_holdout_is_unshippable():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    b = resolve_operating_point("catkin", dataset_hash="h1", calibration_records=_records())
    assert b.get("conf").validated_vs_gt == "false"
    assert not b.is_shippable


def test_resolve_operating_point_no_gt_placeholder_unshippable():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tcip_mcp.pipelines.resolution import UnvalidatedOperatingPointError
    b = resolve_operating_point("catkin", dataset_hash="hX")
    with pytest.raises(UnvalidatedOperatingPointError):
        _ = b.value("conf")  # firewall
    assert b.get("conf").unvalidated_value(acknowledge_unvalidated=True) == 0.5


def test_resolve_operating_point_tile_size_derived():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    b = resolve_operating_point("catkin", dataset_hash="h1", tile_size=640)
    assert b.get("tile_size").source == "derived"
    assert b.get("tile_size").value == 640


def test_classification_metrics_per_class_and_bias():
    from tcip_mcp.pipelines.training.evaluation import classification_metrics
    gt = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])    # 6 dormant, 4 elongated
    pred = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # classifier predicts 6 as elongated
    m = classification_metrics(pred, gt, num_classes=2)
    assert m["per_class"][1]["support"] == 4
    # over-predicting the elongated class inflates the elongated fraction: bias (6-4)/4 = +0.5
    assert m["count_bias"][1] == pytest.approx(0.5)
    assert "accuracy" in m and "f1" in m  # existing keys preserved (additive)
