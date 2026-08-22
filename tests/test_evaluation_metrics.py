"""Evaluation metrics + composite selection objective.

Unit tests for the pycocotools metrics engine, the ported composite objective,
the in-house scalar metrics, and ``_selection_value``; plus light integration
tests that exercise the detection/classification ``_validate`` path end-to-end.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # evaluation.py imports torch at module load
pytest.importorskip("pycocotools")

from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    DEFAULT_SCORE_WEIGHTS,
    build_coco_image_record,
    classification_metrics,
    coco_detection_metrics,
    compute_composite_objective,
    concordance_correlation_coefficient,
    effective_iou_type,
    gt_class_avg_size,
    ordinal_metrics,
    pick_count_unbiased,
    pick_f1_max,
    quadratic_weighted_kappa,
    r_squared,
    regression_metrics,
    resolve_match_criterion,
    run_test_evaluation,
    sweep_operating_point,
)
from tcip_mcp.pipelines.training.generic_trainer import (  # noqa: E402
    _selection_value,
    resolve_selection_metric,
)

# No built-in traits: seed_catkin_trait_spec (conftest.py) writes a real catkin.yml into this
# test's pinned project root so trait="catkin" call sites keep resolving.
pytestmark = pytest.mark.usefixtures("seed_catkin_trait_spec")


# --------------------------------------------------------------------------
# Composite objective
# --------------------------------------------------------------------------

def test_composite_objective_matches_reference():
    assert compute_composite_objective(-1.0, 0.9, 0.9) == 1e6        # val_loss <= 0
    assert compute_composite_objective(2.0, 0.0, 0.0) == 1e6         # degenerate prune
    expected = 0.45 * 2.0 + 0.35 * 0.5 * 10 + 0.20 * 0.6 * 10        # 3.85
    assert compute_composite_objective(2.0, 0.5, 0.4) == pytest.approx(expected, abs=1e-9)
    # NaN coerces: val_loss -> inf branch, f1/map50 -> 0 -> degenerate sentinel.
    assert compute_composite_objective(float("nan"), float("nan"), float("nan")) == 1e6
    assert DEFAULT_SCORE_WEIGHTS == {"loss": 0.45, "f1": 0.35, "map50": 0.20}


# --------------------------------------------------------------------------
# pycocotools detection metrics
# --------------------------------------------------------------------------

def _rec(gt, dt, w=100, h=100):
    return build_coco_image_record(w, h, gt, dt)


def test_coco_map50_perfect():
    gt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]
    dt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}]
    m = coco_detection_metrics([_rec(gt, dt)])
    assert m["map50"] == pytest.approx(1.0)
    assert m["map"] == pytest.approx(1.0)


def test_coco_operating_point_tp_fp_fn():
    gt = [
        {"category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
        {"category_id": 1, "bbox": [50, 50, 10, 10], "area": 100, "iscrowd": 0},
    ]
    dt = [
        {"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},   # TP
        {"category_id": 1, "bbox": [80, 80, 10, 10], "score": 0.7},   # FP
    ]
    m = coco_detection_metrics([_rec(gt, dt)])
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["map50"] > 0.0


def test_coco_conf_threshold_filters():
    gt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]
    dt = [
        {"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},   # TP
        {"category_id": 1, "bbox": [80, 80, 10, 10], "score": 0.7},   # FP, below 0.8
    ]
    m = coco_detection_metrics([_rec(gt, dt)], conf_threshold=0.8)
    assert m["tp"] == 1
    assert m["fp"] == 0


def test_coco_no_stdout(capsys):
    gt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]
    dt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}]
    coco_detection_metrics([_rec(gt, dt)])
    captured = capsys.readouterr()
    assert captured.out == ""  # pycocotools prints must be redirected (MCP stdio safety)


def test_coco_segm_path():
    poly = [10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]
    gt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0, "segmentation": [poly]}]
    dt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9, "segmentation": [poly]}]
    m = coco_detection_metrics([_rec(gt, dt)], iou_type="segm")
    assert 0.0 <= m["map50"] <= 1.0
    assert m["map50"] == pytest.approx(1.0)


def test_coco_empty():
    gt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]
    dt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}]
    # (a) empty preds + non-empty GT -> guard the real loadRes([]) IndexError.
    m = coco_detection_metrics([_rec(gt, [])])
    assert m["fn"] == 1 and m["recall"] == 0.0 and m["map50"] == 0.0
    # (b) non-empty preds + empty GT -> guard the COCOeval stats == -1 sentinel.
    m = coco_detection_metrics([_rec([], dt)])
    assert m["map50"] == 0.0 and m["map50"] >= 0.0
    # (c) both empty.
    m = coco_detection_metrics([_rec([], [])])
    assert m["tp"] == 0 and m["fp"] == 0 and m["map50"] == 0.0


# --------------------------------------------------------------------------
# count-unbiased sweep numerics on fixed synthetic records.
# Pins the center-match sweep + operating-point pickers the calibration relies on.
# --------------------------------------------------------------------------

def _sweep_records():
    """Two images with hand-verifiable center-match outcomes at tolerance 10.

    All boxes are 20x20 (char size 20 → gt_class_avg_size 20; half = 10 tolerance).
    """
    def box(cx, cy, s=20.0):
        return [cx - s / 2, cy - s / 2, s, s]

    def ann(cx, cy, score=None):
        a = {"category_id": 0, "bbox": box(cx, cy)}
        if score is not None:
            a["score"] = score
        return a

    a = {"width": 400, "height": 400,
         "gt": [ann(100, 100)],
         "dt": [ann(100, 100, 0.9), ann(300, 300, 0.6)]}       # 1 TP + 1 far FP
    b = {"width": 400, "height": 400,
         "gt": [ann(100, 100), ann(200, 200)],
         "dt": [ann(100, 100, 0.9), ann(200, 200, 0.3)]}       # 2 TP (one low-conf)
    return [a, b]


def test_golden_gt_class_avg_size():
    assert gt_class_avg_size(_sweep_records(), class_id=0) == pytest.approx(20.0)


def test_golden_sweep_operating_point_curve():
    sweep = sweep_operating_point(_sweep_records(), tolerance=10.0, class_id=0)
    curve = sweep["curve"]
    assert [round(c["conf"], 2) for c in curve] == [0.0, 0.3, 0.6, 0.9]

    at = {round(c["conf"], 2): c for c in curve}
    # conf 0.6: image-a keeps both dets (1 TP + 1 FP), image-b drops the 0.3 det (1 TP, 1 FN).
    assert (at[0.6]["tp"], at[0.6]["fp"], at[0.6]["fn"]) == (2, 1, 1)
    assert at[0.6]["count_bias_mean"] == pytest.approx(0.0)   # +1 and -1 cancel → unbiased
    assert at[0.6]["abs_count_error_mean"] == pytest.approx(1.0)
    # conf 0.9: only the 0.9 dets survive → image-a 1 TP, image-b 1 TP + 1 FN.
    assert (at[0.9]["tp"], at[0.9]["fp"], at[0.9]["fn"]) == (2, 0, 1)
    assert at[0.9]["count_bias_mean"] == pytest.approx(-0.5)   # mean of [0, -1]


def test_golden_operating_point_pickers():
    sweep = sweep_operating_point(_sweep_records(), tolerance=10.0, class_id=0)
    assert pick_count_unbiased(sweep) == pytest.approx(0.6)   # zero count bias
    assert pick_f1_max(sweep) == pytest.approx(0.0)           # recall-max point
    at06 = next(c for c in sweep["curve"] if c["conf"] == pytest.approx(0.6))
    assert at06["count_bias_mean"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# resolve_match_criterion derives/records the localization kind once, reuses it, and warns
# (never silently switches) on divergence.
# --------------------------------------------------------------------------

def _write_bare_trait(name: str, **extra) -> None:
    """A minimal trait spec with no localization recorded (unlike seed_catkin_trait_spec's
    CATKIN, which already carries localization="center_match").

    Seeded at the directory the platform's own resolver returns for this test's pinned project
    root, so the fixture cannot state a specs location the registry does not read from.
    """
    import tcip_store as ts

    from tcip_mcp.project_paths import resolve_state
    from tcip_mcp.traits import _TRAIT_SPECS_RELPATH, trait_spec_key

    specs_dir = resolve_state(_TRAIT_SPECS_RELPATH)
    ts.replace(
        trait_spec_key(specs_dir, name), {"name": name, "delivers": ["leaf_length"], **extra},
        expect=ts.Version.ABSENT,
    )


def _per_image(boxes: list[tuple[float, float, float, float]]) -> list[dict]:
    return [{"gt": [{"bbox": list(b), "category_id": 0} for b in boxes]}]


def test_resolve_match_criterion_derives_and_persists_when_unrecorded(tmp_path: Path):
    from tcip_mcp.traits import get_trait

    _write_bare_trait("leaf")
    small_boxes = [(0, 0, 20, 20), (100, 0, 20, 20)]  # char size 20 -> center_match
    result = resolve_match_criterion("leaf", _per_image(small_boxes))
    assert result["kind"] == "center_match"
    assert result["kind_source"] == "data_derived_at_runtime"
    assert result["kind_diverged"] is False
    # persisted: a fresh read sees the derived value, not the original empty one.
    assert get_trait("leaf").localization == "center_match"


def test_resolve_match_criterion_audits_the_derived_localization_write(tmp_path: Path):
    """The persisted write above is a platform mutation, so it carries an audit line naming the
    trait, the field, the value and the derivation basis, in this project's own log."""
    import tcip_store as ts

    from tcip_mcp import audit as audit_module

    _write_bare_trait("leaf")
    small_boxes = [(0, 0, 20, 20), (100, 0, 20, 20)]
    result = resolve_match_criterion("leaf", _per_image(small_boxes))
    assert result["kind_source"] == "data_derived_at_runtime"

    key = audit_module.audit_log_key(audit_module.platform_audit_scope())
    rows = [r for r in ts.read_log(key).records if r["tool"] == "trait_spec_field_derived"]
    assert len(rows) == 1
    args = rows[0]["arguments"]
    assert args["trait"] == "leaf"
    assert args["field"] == "localization"
    assert args["value"] == "center_match"
    assert "GT boxes" in args["basis"]


def test_resolve_match_criterion_raises_when_the_derived_audit_append_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The spec write already committed by the time the audit line is attempted, so a caller told
    only by a swallowed warning would read the write as never having happened and blind-retry it.
    The append failing must raise and name that committed write, not vanish into a log line."""
    import tcip_store as ts

    from tcip_mcp import audit as audit_module
    from tcip_mcp.traits import get_trait

    _write_bare_trait("leaf")
    small_boxes = [(0, 0, 20, 20), (100, 0, 20, 20)]

    real_append = audit_module.append

    def _flaky_append(key, *args, **kwargs):
        if key.store == audit_module.AUDIT_LOG_STORE:
            raise RuntimeError("the audit log could not be appended to")
        return real_append(key, *args, **kwargs)

    monkeypatch.setattr(audit_module, "append", _flaky_append)

    with pytest.raises(audit_module.AuditEntryNotWritten):
        resolve_match_criterion("leaf", _per_image(small_boxes))

    # the spec write is not routed through audit_module.append, so it already landed.
    assert get_trait("leaf").localization == "center_match"
    key = audit_module.audit_log_key(audit_module.platform_audit_scope())
    rows = [r for r in ts.read_log(key).records if r["tool"] == "trait_spec_field_derived"]
    assert rows == []


def test_derived_localization_kind_is_read_back_as_recorded_on_the_next_call(tmp_path: Path):
    """A kind derived once has to land where the registry reads: the second call on the same
    project reports it as recorded rather than deriving it again from that call's own GT, which is
    what keeps one trait's governing criterion stable across sessions."""
    _write_bare_trait("leaf")
    small_boxes = [(0, 0, 20, 20), (100, 0, 20, 20)]
    first = resolve_match_criterion("leaf", _per_image(small_boxes))
    assert first["kind_source"] == "data_derived_at_runtime"

    second = resolve_match_criterion("leaf", _per_image(small_boxes))
    assert second["kind_source"] == "recorded"
    assert second["kind"] == first["kind"]
    assert second["kind_diverged"] is False


def test_resolve_match_criterion_reuses_recorded_kind_without_rederiving(tmp_path: Path):
    _write_bare_trait("leaf", localization="iou_match")
    # Small boxes would derive center_match fresh, but a recorded kind must be used as-is.
    small_boxes = [(0, 0, 20, 20), (100, 0, 20, 20)]
    result = resolve_match_criterion("leaf", _per_image(small_boxes))
    assert result["kind"] == "iou_match"
    assert result["kind_source"] == "recorded"


def test_resolve_match_criterion_flags_divergence_without_switching(tmp_path: Path):
    _write_bare_trait("leaf", localization="iou_match")
    # Small boxes: derive_localization_kind would say center_match, diverging from the recorded
    # iou_match. Must warn (kind_diverged=True), never silently switch what governs this call.
    small_boxes = [(0, 0, 20, 20), (100, 0, 20, 20)]
    result = resolve_match_criterion("leaf", _per_image(small_boxes))
    assert result["kind_diverged"] is True
    assert result["kind"] == "iou_match"  # unchanged despite the divergence


def test_resolve_match_criterion_no_divergence_when_kinds_agree(tmp_path: Path):
    _write_bare_trait("leaf", localization="center_match")
    small_boxes = [(0, 0, 20, 20), (100, 0, 20, 20)]
    result = resolve_match_criterion("leaf", _per_image(small_boxes))
    assert result["kind_diverged"] is False


def test_resolve_match_criterion_refuses_when_unrecorded_and_underivable(tmp_path: Path):
    _write_bare_trait("leaf")
    with pytest.raises(ValueError, match="no recorded localization kind"):
        resolve_match_criterion("leaf", [])  # no GT at all -> nothing to derive from


def test_resolve_match_criterion_no_trait_is_iou_comparability_convention():
    result = resolve_match_criterion(None, [])
    assert result["kind"] == "iou_match"
    assert result["trait"] is None


def test_resolve_match_criterion_iou_match_derives_a_real_threshold_not_pinned_0_5(tmp_path: Path):
    """iou_match's threshold must be genuinely derived from the GT in hand
    (derive_iou_match_threshold), not pinned to 0.5."""
    _write_bare_trait("leaf", localization="iou_match")
    # char size 300 -> derived threshold well above 0.5 (see test_derive_iou_match_threshold_*).
    large_boxes = [(0, 0, 300, 300), (500, 0, 300, 300)]
    result = resolve_match_criterion("leaf", _per_image(large_boxes))
    assert result["kind"] == "iou_match"
    assert result["iou_threshold"] > 0.5
    assert "achievable IoU" in result["derived_from"]


def test_resolve_match_criterion_iou_match_falls_back_honestly_when_underivable(tmp_path: Path):
    _write_bare_trait("leaf", localization="iou_match")
    result = resolve_match_criterion("leaf", [], iou_threshold=0.42)
    assert result["kind"] == "iou_match"
    assert result["iou_threshold"] == pytest.approx(0.42)  # caller/default, not a fabricated derivation
    assert "underivable" in result["derived_from"]


# --------------------------------------------------------------------------
# COCO box and mask mAP exact values on fixed synthetic records.
# --------------------------------------------------------------------------

def test_golden_coco_box_and_mask_map():
    gt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]
    dt_perfect = [{"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}]
    m = coco_detection_metrics([_rec(gt, dt_perfect)])
    assert (m["map"], m["map50"], m["map75"]) == pytest.approx((1.0, 1.0, 1.0))
    assert (m["precision"], m["recall"], m["f1"]) == pytest.approx((1.0, 1.0, 1.0))
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 0)

    # 1 TP + 1 FP against 2 GT → P=R=F1=0.5, counts 1/1/1.
    gt2 = [
        {"category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
        {"category_id": 1, "bbox": [50, 50, 10, 10], "area": 100, "iscrowd": 0},
    ]
    dt2 = [
        {"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
        {"category_id": 1, "bbox": [80, 80, 10, 10], "score": 0.7},
    ]
    m2 = coco_detection_metrics([_rec(gt2, dt2)])
    assert (m2["precision"], m2["recall"], m2["f1"]) == pytest.approx((0.5, 0.5, 0.5))
    assert (m2["tp"], m2["fp"], m2["fn"]) == (1, 1, 1)

    # segm: a perfect mask match scores map50 = 1.0.
    poly = [10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]
    gt_s = [{"category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0, "segmentation": [poly]}]
    dt_s = [{"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9, "segmentation": [poly]}]
    ms = coco_detection_metrics([_rec(gt_s, dt_s)], iou_type="segm")
    assert ms["iou_type"] == "segm"
    assert ms["map50"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# In-house scalar metrics
# --------------------------------------------------------------------------

def test_classification_metrics():
    pred = torch.tensor([0, 1, 1, 0])
    gt = torch.tensor([0, 1, 0, 0])
    m = classification_metrics(pred, gt, num_classes=2)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx((0.8 + 2 / 3) / 2, abs=1e-3)


def test_ordinal_metrics():
    m = ordinal_metrics(torch.tensor([0, 1, 2]), torch.tensor([0, 2, 2]))
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["rank_acc"] == pytest.approx(2 / 3)
    assert m["quadratic_weighted_kappa"] == pytest.approx(0.8)


def test_regression_metrics():
    m = regression_metrics(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 4.0]))
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["rmse"] == pytest.approx(math.sqrt(1 / 3))
    assert m["r_squared"] == pytest.approx(11 / 14)


def test_quadratic_weighted_kappa_perfect_agreement_is_one():
    kappa = quadratic_weighted_kappa(torch.tensor([0, 1, 2, 1]), torch.tensor([0, 1, 2, 1]))
    assert kappa == pytest.approx(1.0)


def test_quadratic_weighted_kappa_hand_computed():
    # true=[0,2,2], pred=[0,1,2]: one item off by one rank out of three, worked out by hand
    # against the expected-disagreement-under-independence formula -> kappa = 1 - 1/5.
    kappa = quadratic_weighted_kappa(torch.tensor([0, 1, 2]), torch.tensor([0, 2, 2]))
    assert kappa == pytest.approx(0.8)


def test_quadratic_weighted_kappa_reads_a_fractional_prediction_at_its_nearest_rank():
    """A head that emits continuous rank estimates is scored at the rank each estimate is nearest
    to, so predictions that all round onto the true ranks are perfect agreement. Reading 1.6 as
    rank 1 would report a disagreement the model does not have."""
    fractional = torch.tensor([0.4, 1.6, 2.4, 2.6])
    gt = torch.tensor([0, 2, 2, 3])
    assert quadratic_weighted_kappa(fractional, gt) == pytest.approx(1.0)


def test_quadratic_weighted_kappa_scores_a_single_rank_guess_at_chance():
    """Chance correction comes from the scored set's own rank marginals, so a predictor that
    always guesses the majority rank scores 0 however skewed that majority is: here it is exactly
    right on 7 of 10 items and still says nothing beyond the marginals."""
    gt = torch.tensor([0, 0, 0, 0, 0, 0, 0, 3, 3, 3])
    always_majority = torch.zeros(10)
    assert int((always_majority == gt).sum()) == 7
    assert quadratic_weighted_kappa(always_majority, gt) == pytest.approx(0.0, abs=1e-9)


def test_quadratic_weighted_kappa_empty_is_none():
    assert quadratic_weighted_kappa(torch.tensor([]), torch.tensor([])) is None


def test_quadratic_weighted_kappa_degenerate_single_rank_is_none():
    # every item shares the same true and predicted rank: no expected disagreement to correct for.
    kappa = quadratic_weighted_kappa(torch.tensor([1, 1, 1]), torch.tensor([1, 1, 1]), num_ranks=3)
    assert kappa is None


def test_r_squared_perfect_fit_is_one():
    assert r_squared(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])) == pytest.approx(1.0)


def test_r_squared_worse_than_mean_baseline_is_negative():
    r2 = r_squared(torch.tensor([5.0, -5.0, 5.0]), torch.tensor([1.0, 2.0, 3.0]))
    assert r2 < 0


def test_r_squared_empty_is_none():
    assert r_squared(torch.tensor([]), torch.tensor([])) is None


def test_r_squared_constant_gt_is_none():
    assert r_squared(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([5.0, 5.0, 5.0])) is None


def test_concordance_correlation_coefficient_perfect_agreement_is_one():
    ccc = concordance_correlation_coefficient(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0]))
    assert ccc == pytest.approx(1.0)


def test_concordance_correlation_coefficient_hand_computed():
    # pred=[1,2,3], gt=[1,2,4] (the same pair test_regression_metrics uses), population statistics:
    # pred_mean=2, gt_mean=7/3; pred_var=mean((pred-2)^2)=(1+0+1)/3=2/3;
    # gt_var=mean((gt-7/3)^2)=((-4/3)^2+(-1/3)^2+(5/3)^2)/3=(16/9+1/9+25/9)/3=(42/9)/3=14/9;
    # covariance=mean((pred-2)*(gt-7/3))=((-1)*(-4/3)+0*(-1/3)+1*(5/3))/3=(4/3+5/3)/3=1;
    # CCC = 2*covariance / (pred_var+gt_var+(pred_mean-gt_mean)^2)
    #     = 2*1 / (2/3+14/9+1/9) = 2 / (7/3) = 6/7.
    ccc = concordance_correlation_coefficient(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 4.0]))
    assert ccc == pytest.approx(6 / 7)


def test_concordance_correlation_coefficient_empty_is_none():
    assert concordance_correlation_coefficient(torch.tensor([]), torch.tensor([])) is None


def test_concordance_correlation_coefficient_zero_variance_is_none():
    # constant predictions against varying GT: pred_var=0, denominator undefined.
    assert concordance_correlation_coefficient(
        torch.tensor([2.0, 2.0, 2.0]), torch.tensor([1.0, 2.0, 3.0])) is None
    # constant GT against varying predictions: gt_var=0, same undefined denominator.
    assert concordance_correlation_coefficient(
        torch.tensor([1.0, 2.0, 3.0]), torch.tensor([5.0, 5.0, 5.0])) is None


def test_selection_value_prefers_objective_for_detection():
    assert _selection_value("detection", {"val_loss": 0.1, "val_objective": 5.0}, 0.2, "objective") == 5.0
    assert _selection_value("classification", {"val_loss": 0.1}, 0.2, "loss") == 0.1
    assert _selection_value("detection", {"val_loss": 0.1}, 0.2, "objective") == 0.1  # no objective -> val_loss


def test_resolve_selection_metric_defaults():
    assert resolve_selection_metric("detection", None, None) == "objective"
    assert resolve_selection_metric("instance_seg", None, None) == "objective"
    assert resolve_selection_metric("classification", None, None) == "loss"
    assert resolve_selection_metric("semantic_seg", None, None) == "loss"


def test_resolve_selection_metric_rejects_incoherent_explicit_choice():
    with pytest.raises(ValueError, match="comparability-only"):
        resolve_selection_metric("detection", "catkin", "map50")


def test_resolve_selection_metric_allows_coherent_explicit_choice():
    # A legitimate explicit choice must still succeed: a rail must admit valid work, not
    # only reject invalid work.
    assert resolve_selection_metric("detection", "catkin", "f1") == "f1"
    assert resolve_selection_metric("detection", "catkin", "recall") == "recall"
    assert resolve_selection_metric("detection", None, "map50") == "map50"  # no trait -> no gate


# --------------------------------------------------------------------------
# Effective iou_type: evaluate() scoring and run_test_evaluation metadata
# --------------------------------------------------------------------------

def test_effective_iou_type_resolution():
    assert effective_iou_type("detection", None) == "bbox"
    assert effective_iou_type("instance_seg", None) == "segm"   # segm AP by default
    assert effective_iou_type("instance_seg", "bbox") == "bbox"  # explicit override wins
    assert effective_iou_type("detection", "segm") == "segm"
    assert effective_iou_type("classification", None) == ""


def test_run_test_evaluation_records_effective_iou_type(tmp_path, monkeypatch):
    """test_results.json must record the iou_type evaluate() actually scored with
    (instance_seg defaults to segm AP; recording 'bbox' would misreport mask AP)."""
    import tcip_mcp.pipelines.model_build as model_build
    import tcip_mcp.pipelines.training.evaluation as evaluation

    class _DummyModel:
        def load_state_dict(self, state_dict):
            pass

        def to(self, device):
            pass

    ckpt_path = tmp_path / "model_best.pt"
    torch.save({"model_source": {"builder": "x:y"}, "model_state_dict": {}}, str(ckpt_path))
    monkeypatch.setattr(model_build, "build_model", lambda ckpt: _DummyModel())
    monkeypatch.setattr(evaluation, "evaluate", lambda *a, **k: {"loss": 0.1, "map50": 0.5})

    import tcip_store as ts

    from tcip_mcp.pipelines.training.evaluation import evaluation_results_key

    r = run_test_evaluation(str(ckpt_path), None, "cpu", "instance_seg", str(tmp_path / "seg"))
    assert r["iou_type"] == "segm"
    on_disk = ts.read(evaluation_results_key(tmp_path / "seg"))
    assert on_disk["iou_type"] == "segm"

    r = run_test_evaluation(str(ckpt_path), None, "cpu", "detection", str(tmp_path / "det"))
    assert r["iou_type"] == "bbox"

    r = run_test_evaluation(str(ckpt_path), None, "cpu", "instance_seg", str(tmp_path / "ovr"),
                            iou_type="bbox")
    assert r["iou_type"] == "bbox"  # explicit override still recorded as-is


def test_a_written_result_carries_one_byte_per_line_ending(tmp_path, monkeypatch):
    """A result document holds the same bytes wherever it was produced.

    A text-mode writer emits CRLF on one platform and LF on another, so the same measurement
    hashes differently depending on the machine that scored it. Bound to the file backend on
    purpose: the CRLF risk is a text-mode file-write concern, and a database backend stores the
    codec's own bytes in a column with no OS line-ending translation to go wrong.
    """
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    import tcip_mcp.pipelines.model_build as model_build
    import tcip_mcp.pipelines.training.evaluation as evaluation

    ts.bind(FileBackend())

    class _DummyModel:
        def load_state_dict(self, state_dict):
            pass

        def to(self, device):
            pass

    ckpt_path = tmp_path / "model_best.pt"
    torch.save({"model_source": {"builder": "x:y"}, "model_state_dict": {}}, str(ckpt_path))
    monkeypatch.setattr(model_build, "build_model", lambda ckpt: _DummyModel())
    monkeypatch.setattr(evaluation, "evaluate", lambda *a, **k: {"loss": 0.1, "map50": 0.5})

    r = run_test_evaluation(str(ckpt_path), None, "cpu", "detection", str(tmp_path / "out"))

    raw = Path(r["results_path"]).read_bytes()
    assert b"\r\n" not in raw
    assert b"\n" in raw


def test_run_test_evaluation_hands_back_the_file_it_wrote(tmp_path, monkeypatch):
    """``results_path`` names the readable test_results.json under the caller's output_dir, and its
    contents are the same result the call returned. This is the only handle a caller keeps on a
    finished evaluation, so a path that does not open, or opens onto different numbers, loses the
    evaluation. Bound to the file backend on purpose: ``results_path`` is a real filesystem path,
    and a database backend keeps the record in the database instead of at that path.
    """
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    import tcip_mcp.pipelines.model_build as model_build
    import tcip_mcp.pipelines.training.evaluation as evaluation

    ts.bind(FileBackend())

    class _DummyModel:
        def load_state_dict(self, state_dict):
            pass

        def to(self, device):
            pass

    ckpt_path = tmp_path / "model_best.pt"
    torch.save({"model_source": {"builder": "x:y"}, "model_state_dict": {}}, str(ckpt_path))
    monkeypatch.setattr(model_build, "build_model", lambda ckpt: _DummyModel())
    monkeypatch.setattr(evaluation, "evaluate",
                        lambda *a, **k: {"loss": 0.1, "map50": 0.5, "precision": 0.4, "recall": 0.75})

    out_dir = tmp_path / "runs" / "test"
    r = run_test_evaluation(str(ckpt_path), None, "cpu", "detection", str(out_dir))

    results_path = Path(r["results_path"])
    assert results_path == out_dir / "test_results.json"
    assert results_path.is_file()
    assert json.loads(results_path.read_text()) == {k: v for k, v in r.items() if k != "results_path"}


# --------------------------------------------------------------------------
# Light integration: _validate via train()
# --------------------------------------------------------------------------

torchvision = pytest.importorskip("torchvision")
from torch.utils.data import DataLoader  # noqa: E402

from tcip_mcp.pipelines.data.datasets import build_dataset  # noqa: E402
from tcip_mcp.pipelines.training.generic_trainer import create_run, task_collate, train  # noqa: E402

IMG = 64


def _save_png(path: Path) -> None:
    from torchvision.utils import save_image
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.rand(3, IMG, IMG) * 0.3, str(path))


def _cfg(model_source) -> dict:
    return {
        "model_source": model_source, "device": "cpu",
        "stages": [{"freeze_to": -1, "epochs": 1}], "mixed_precision": False,
        "optimizer": {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 0},
        "early_stopping": {"enabled": False},
    }


def test_validate_detection_returns_metrics_and_objective(tmp_path):
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        _save_png(images_dir / f"img{i}.png")
        json_io.write_annotations(str(labels_dir / f"img{i}.json"),
                                  [Annotation(subject="catkin", geometry=BBox(19.2, 19.2, 44.8, 44.8))],
                                  IMG, IMG, keep_empty=True)
    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                       subject="catkin")
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("detection"))

    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": {"num_classes": 1, "min_size": IMG, "max_size": IMG * 2},
                    "task": "detection"}
    run = create_run(_cfg(model_source), str(tmp_path / "out"))
    run = train(run, loader, val_loader=loader, task="detection")  # no AttributeError on model.heads

    assert run.status == "completed", getattr(run, "error", run.status)
    last = run.metrics_history[-1]
    for k in ("val_loss", "val_precision", "val_recall", "val_f1", "val_map50", "val_map", "val_objective"):
        assert k in last, f"missing {k}"
    assert (tmp_path / "out" / "model_best.pt").is_file()
    assert run.best_metric == pytest.approx(last["val_objective"])


def test_train_center_match_trait_records_governing_criterion(tmp_path):
    """Threading `trait` into _validate surfaces val_governing_criterion (a dict) and
    val_map50_role (a str) in val_metrics; the TensorBoard scalar loop must skip these
    non-numeric values rather than crash `add_scalar` on them."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        _save_png(images_dir / f"img{i}.png")
        json_io.write_annotations(str(labels_dir / f"img{i}.json"),
                                  [Annotation(subject="catkin", geometry=BBox(19.2, 19.2, 44.8, 44.8))],
                                  IMG, IMG, keep_empty=True)
    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                       subject="catkin")
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("detection"))

    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": {"num_classes": 1, "min_size": IMG, "max_size": IMG * 2},
                    "task": "detection"}
    cfg = _cfg(model_source)
    cfg["evaluation"] = {"trait": "catkin"}
    run = create_run(cfg, str(tmp_path / "out"))
    run = train(run, loader, val_loader=loader, task="detection")

    assert run.status == "completed", getattr(run, "error", run.status)
    last = run.metrics_history[-1]
    assert "val_governing_criterion" in last
    assert last["val_map50_role"] == "comparability_only"
    assert last["selection_metric"] == "objective"
    assert last["selection_trait"] == "catkin"


def test_validate_classification_metrics(tmp_path):
    images_dir = tmp_path / "images"
    rows = []
    for i in range(6):
        _save_png(images_dir / f"img{i}.png")
        rows.append((f"img{i}", i % 2))
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("stem", "label"))
        w.writerows(rows)
    ds = build_dataset("classification", images_dir=str(images_dir), csv_path=str(csv_path), num_classes=2)
    loader = DataLoader(ds, batch_size=3, collate_fn=task_collate("classification"))

    model_source = {"builder": "tests.bespoke_models:build_bespoke_classifier",
                    "builder_kwargs": {"num_classes": 2}, "task": "classification"}
    run = create_run(_cfg(model_source), str(tmp_path / "out"))
    run = train(run, loader, val_loader=loader, task="classification")

    assert run.status == "completed", getattr(run, "error", run.status)
    last = run.metrics_history[-1]
    assert "val_accuracy" in last and "val_f1" in last
    assert run.best_metric == pytest.approx(last["val_loss"])  # selection falls back to val_loss


@pytest.fixture
def json_data_dir(tmp_path: Path) -> Path:
    """Minimal dataset with per-image JSON labels/predictions in the canonical layout.

    score_predictions reads GT and predictions through the canonical json_io per-image schema
    (name-based, one file per image, pixel COCO xywh + native ``score``).
    """
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    date = "2-11-26"
    images_dir = tmp_path / "images" / date
    images_dir.mkdir(parents=True)
    labels_dir = tmp_path / "annotations" / date
    labels_dir.mkdir(parents=True)
    preds_dir = tmp_path / "predictions" / "live" / date
    preds_dir.mkdir(parents=True)

    for name in ("img_001", "img_002", "img_003"):
        Image.new("RGB", (640, 480), color=(128, 128, 128)).save(images_dir / f"{name}.jpg")
        json_io.write_annotations(
            str(labels_dir / f"{name}.json"),
            [Annotation(subject="catkin", geometry=BBox(288, 216, 352, 264)),
             Annotation(subject="catkin", geometry=BBox(176, 132, 208, 156))],
            640, 480,
        )
        # 1 matching prediction (TP) + 1 elsewhere (FP), confidence in each annotation's score.
        json_io.write_annotations(
            str(preds_dir / f"{name}.json"),
            [Annotation(subject="catkin", geometry=BBox(288, 216, 352, 264), score=0.9),
             Annotation(subject="catkin", geometry=BBox(496, 372, 528, 396), score=0.7)],
            640, 480,
        )
    return tmp_path


def test_score_predictions_folder_uses_pycocotools(json_data_dir):
    data_dir = json_data_dir
    from tcip_mcp.tools.annotation_tools import score_predictions
    r = score_predictions(str(data_dir))
    assert "map50" in r
    # fixture: each image has 2 GT, predictions = 1 TP + 1 FP -> tp=1,fp=1,fn=1 per image (x3 images).
    assert r["total_tp"] == 3 and r["total_fp"] == 3 and r["total_fn"] == 3
    assert r["precision"] == pytest.approx(0.5)
    assert all(p["tp"] == 1 and p["fp"] == 1 and p["fn"] == 1 for p in r["per_image"])
