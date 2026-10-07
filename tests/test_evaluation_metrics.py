"""Evaluation metrics + composite selection objective.

Unit tests for the one matcher's detection metrics, the composite objective,
the in-house scalar metrics, and ``_selection_value``; plus light integration
tests that exercise the detection/classification ``_validate`` path end-to-end.
"""

from __future__ import annotations

import csv
import math
from functools import partial

import pytest

torch = pytest.importorskip("torch")  # evaluation.py imports torch at module load

from tcip_annotation.matching import match_pairs  # noqa: E402
from tcip_mcp.dataset_layout import UNDATED_BUCKET  # noqa: E402
from tcip_mcp.pipelines.data.label_queries import registry_scope  # noqa: E402
from tcip_mcp.pipelines.data.selection import ClassScope  # noqa: E402
from tcip_mcp.pipelines.execution import Stated  # noqa: E402
from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    DEFAULT_SCORE_WEIGHTS,
    classification_metrics,
    compute_composite_objective,
    concordance_correlation_coefficient,
    detection_metrics,
    gt_class_avg_size,
    image_record,
    ordinal_metrics,
    pick_count_unbiased,
    pick_f1_max,
    quadratic_weighted_kappa,
    r_squared,
    regression_metrics,
    resolve_match_criterion,
    derive_operating_point_curve,
)
from tcip_mcp.pipelines.training.eval_runners import (  # noqa: E402
    evaluation_result,
    run_test_evaluation,
)
from tcip_mcp.pipelines.training.generic_trainer import (  # noqa: E402
    _selection_value,
    resolve_selection_metric,
)
from tests._image_fixtures import write_noise_image  # noqa: E402
from tests._producer_fixtures import checkpoint_admission  # noqa: E402
from tests._dense_op_fixtures import gt_only  # noqa: E402
from tests import _trait_fixtures as fx  # noqa: E402

# A test naming trait="bud_opening" proposes it in its project (conftest.seed_bud_trait_spec).
_with_bud_trait = pytest.mark.usefixtures("seed_bud_trait_spec")


# --------------------------------------------------------------------------
# Composite objective
# --------------------------------------------------------------------------

def test_composite_objective_matches_reference():
    expected = 0.45 * 2.0 + 0.35 * 0.5 * 10 + 0.20 * 0.6 * 10        # 3.85
    assert compute_composite_objective(2.0, 0.5, 0.4) == pytest.approx(expected, abs=1e-9)
    assert DEFAULT_SCORE_WEIGHTS == {"loss": 0.45, "f1": 0.35, "map50": 0.20}


def test_composite_objective_has_no_score_for_a_degenerate_epoch():
    """A non-positive or non-finite loss, or both quality terms at zero, is an epoch with no
    useful score: the objective is None, which a metrics row carries as null and the selection
    comparison treats as never improving, never a large number a chart would plot as a value."""
    assert compute_composite_objective(-1.0, 0.9, 0.9) is None
    assert compute_composite_objective(2.0, 0.0, 0.0) is None
    assert compute_composite_objective(float("nan"), float("nan"), float("nan")) is None
    assert compute_composite_objective(float("inf"), 0.5, 0.5) is None


# --------------------------------------------------------------------------
# detection metrics under the IoU convention
# --------------------------------------------------------------------------

def _rec(gt, dt, w=100, h=100):
    return image_record(w, h, gt, dt)


def test_detection_map50_perfect():
    gt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0}]
    dt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}]
    m = detection_metrics([_rec(gt, dt)], trait=None, conf_threshold=0.25, iou_threshold=0.5,
                          by_mask=False)
    assert m["map50"] == pytest.approx(1.0)
    assert m["map"] == pytest.approx(1.0)


def test_detection_operating_point_tp_fp_fn():
    gt = [
        {"category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0},
        {"category_id": 1, "bbox": [50, 50, 10, 10], "iscrowd": 0},
    ]
    dt = [
        {"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},   # TP
        {"category_id": 1, "bbox": [80, 80, 10, 10], "score": 0.7},   # FP
    ]
    m = detection_metrics([_rec(gt, dt)], trait=None, conf_threshold=0.25, iou_threshold=0.5,
                          by_mask=False)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["map50"] > 0.0


def test_detection_conf_threshold_filters():
    gt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0}]
    dt = [
        {"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},   # TP
        {"category_id": 1, "bbox": [80, 80, 10, 10], "score": 0.7},   # FP, below 0.8
    ]
    m = detection_metrics([_rec(gt, dt)], trait=None, conf_threshold=0.8, iou_threshold=0.5,
                          by_mask=False)
    assert m["tp"] == 1
    assert m["fp"] == 0


def test_detection_metrics_write_nothing_to_stdout(capsys):
    gt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0}]
    dt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}]
    detection_metrics([_rec(gt, dt)], trait=None, conf_threshold=0.25, iou_threshold=0.5,
                      by_mask=False)
    captured = capsys.readouterr()
    assert captured.out == ""  # nothing reaches stdout (MCP stdio safety)


def test_detection_mask_path():
    ring = [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)]
    gt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0, "rings": [ring]}]
    dt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9, "rings": [ring]}]
    m = detection_metrics([_rec(gt, dt)], trait=None, conf_threshold=0.25, iou_threshold=0.5,
                          by_mask=True)
    assert m["map50"] == pytest.approx(1.0)
    assert m["governing_criterion"]["kind"] == "mask_iou_match"


def test_a_mask_match_counts_by_region_where_the_box_match_would_not():
    """An L-shaped object whose box a detection's box covers exactly, but whose region the
    detection's region overlaps under half: matched by box, missed by mask."""
    l_shape = [(0.0, 0.0), (40.0, 0.0), (40.0, 10.0), (10.0, 10.0), (10.0, 40.0), (0.0, 40.0)]
    corner = [(10.0, 10.0), (40.0, 10.0), (40.0, 40.0), (10.0, 40.0)]
    gt = [{"category_id": 1, "bbox": [0, 0, 40, 40], "iscrowd": 0, "rings": [l_shape]}]
    dt = [{"category_id": 1, "bbox": [0, 0, 40, 40], "score": 0.9, "rings": [corner]}]
    by_box, by_mask = (detection_metrics([_rec(gt, dt)], trait=None, conf_threshold=0.25,
                                         iou_threshold=0.5, by_mask=mask) for mask in (False, True))
    assert (by_box["tp"], by_box["fp"], by_box["fn"]) == (1, 0, 0)
    assert (by_mask["tp"], by_mask["fp"], by_mask["fn"]) == (0, 1, 1)


def test_detection_metrics_over_an_empty_side():
    gt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0}]
    dt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}]

    def metrics(gt, dt):
        return detection_metrics([_rec(gt, dt)], trait=None, conf_threshold=0.25,
                                 iou_threshold=0.5, by_mask=False)

    # (a) no predictions over ground truth: every object missed.
    m = metrics(gt, [])
    assert m["fn"] == 1 and m["recall"] == 0.0 and m["map50"] == 0.0
    # (b) predictions over no ground truth: no object to find, so no precision to average.
    m = metrics([], dt)
    assert m["map50"] == 0.0
    # (c) both empty.
    m = metrics([], [])
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
        a = {"category_id": 0, "bbox": box(cx, cy), "iscrowd": 0}
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


CENTER_10 = {"kind": "center_match", "tolerance": 10.0}
"""A center match at a 10 px tolerance."""


def _points(centers: list[tuple[float, float]]) -> list[dict]:
    """Each center as the matcher's record of a zero-extent xywh box, so a center match reads its
    distance exactly."""
    return [{"bbox": [x, y, 0.0, 0.0]} for x, y in centers]


def test_golden_derive_operating_point_curve():
    sweep = derive_operating_point_curve(_sweep_records(), criterion=CENTER_10, class_id=0)
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
    sweep = derive_operating_point_curve(_sweep_records(), criterion=CENTER_10, class_id=0)
    assert pick_count_unbiased(sweep) == pytest.approx(0.6)   # zero count bias
    assert pick_f1_max(sweep) == pytest.approx(0.0)           # recall-max point
    at06 = next(c for c in sweep["curve"] if c["conf"] == pytest.approx(0.6))
    assert at06["count_bias_mean"] == pytest.approx(0.0)


# match_pairs: one matcher, two stated policies (score_first for the count, distance_first for
# the classifier pairing), under a center or an IoU criterion; coverage of the pinned tie rules.

def test_match_pairs_score_first_and_distance_first_disagree_on_cardinality():
    """Two ground truths, one detection within tolerance of both: score-first (walking
    detections in the given, score-descending order) claims one pair; distance-first (every
    pair sorted by distance ascending) claims two. Exact pairs, not just counts."""
    gt = _points([(0.0, 0.0), (10.0, 0.0)])
    dt = _points([(4.0, 0.0), (0.0, 0.0)])  # score 0.9 then 0.1, already score-descending
    criterion = {"kind": "center_match", "tolerance": 6.0}

    assert match_pairs(gt, dt, criterion, policy="score_first") == [(0, 0)]
    assert match_pairs(gt, dt, criterion, policy="distance_first") == [(0, 1), (1, 0)]


def test_match_pairs_score_first_tie_keeps_the_last_index():
    """Coverage: among equidistant unused ground truths, score-first keeps the last index."""
    pairs = match_pairs(_points([(0.0, 0.0), (2.0, 0.0)]), _points([(1.0, 0.0)]),
                        {"kind": "center_match", "tolerance": 1.0}, policy="score_first")
    assert pairs == [(1, 0)]


def test_match_pairs_distance_first_tie_breaks_by_gt_then_detection_index():
    """Coverage: distance-first breaks a tied distance by (gt index, detection index)
    ascending, so a fully degenerate 2x2 assigns each detection to the ground truth sharing
    its own index rather than crossing them."""
    pairs = match_pairs(_points([(0.0, 0.0), (0.0, 0.0)]), _points([(5.0, 0.0), (5.0, 0.0)]),
                        {"kind": "center_match", "tolerance": 10.0}, policy="distance_first")
    assert pairs == [(0, 0), (1, 1)]


def test_match_pairs_under_an_iou_criterion_matches_by_overlap():
    """The same matcher under an IoU criterion: a detection overlapping its ground truth past the
    threshold pairs, one beside it does not, however near its center sits."""
    gt = [{"bbox": [0.0, 0.0, 10.0, 10.0]}, {"bbox": [100.0, 0.0, 10.0, 10.0]}]
    dt = [{"bbox": [1.0, 0.0, 10.0, 10.0]}, {"bbox": [106.0, 0.0, 10.0, 10.0]}]
    pairs = match_pairs(gt, dt, {"kind": "iou_match", "iou_threshold": 0.5},
                        policy="score_first")
    assert pairs == [(0, 0)]


def test_match_pairs_refuses_an_unknown_policy_by_name():
    """Coverage: a policy outside the two stated ones refuses rather than falling through to
    either, so a typo never silently picks a matching rule."""
    with pytest.raises(ValueError, match="policy"):
        match_pairs(_points([(0.0, 0.0)]), _points([(0.0, 0.0)]),
                    {"kind": "center_match", "tolerance": 1.0}, policy="nearest")


def test_dt_score_refuses_a_record_without_a_score_or_with_none_by_name():
    """Coverage: the one confidence accessor the governing count reads through refuses a record
    with no score field, and one whose score is None, each naming the record; the curve's score
    grid reads through the same accessor, so neither reaches a bare KeyError or TypeError."""
    from tcip_mcp.pipelines.training.evaluation import _dt_score, derive_operating_point_curve

    with pytest.raises(ValueError, match="no 'score' field"):
        _dt_score({"bbox": [0, 0, 1, 1]})
    with pytest.raises(ValueError, match="is None"):
        _dt_score({"bbox": [0, 0, 1, 1], "score": None})
    per_image = [{"image_id": 1, "gt": [], "dt": [{"bbox": [0, 0, 1, 1], "category_id": 1}]}]
    with pytest.raises(ValueError, match="no 'score' field"):
        derive_operating_point_curve(per_image, criterion={"kind": "center_match",
                                                           "tolerance": 1.0})


# resolve_match_criterion reads the revision's stated localization and never derives one.

def test_resolve_match_criterion_refuses_an_unauthored_localization_asking_the_breeder():
    """No localization is derived from the boxes in hand: a revision that states none refuses,
    naming the field and the breeder's question, whatever the ground truth would suggest."""
    from tcip_mcp.traits import UnauthoredFieldError

    small_boxes = [(0, 0, 20, 20), (100, 0, 20, 20)]
    with pytest.raises(UnauthoredFieldError, match="localization"):
        resolve_match_criterion(fx.entry("leaf", ("leaf_length",)), gt_only(small_boxes))


def test_resolve_match_criterion_reads_the_stated_kind_as_is():
    """Boxes small enough to suit a center match change nothing when the revision states IoU."""
    trait = fx.entry("leaf", ("leaf_length",), localization="iou_match")
    small_boxes = [(0, 0, 20, 20), (100, 0, 20, 20)]
    assert resolve_match_criterion(trait, gt_only(small_boxes))["kind"] == "iou_match"


def test_resolve_match_criterion_scales_a_center_match_to_the_reference():
    """A center match carries its tolerance as a fraction of the average object size and the
    tolerance that fraction is on this reference."""
    trait = fx.entry("leaf", ("leaf_length",), localization="center_match")
    result = resolve_match_criterion(trait, gt_only([(0, 0, 20, 20), (500, 0, 20, 20)]))
    assert result["kind"] == "center_match"
    assert result["tolerance"] == pytest.approx(result["tolerance_frac"] * 20.0)


def test_resolve_match_criterion_no_trait_is_iou_comparability_convention():
    result = resolve_match_criterion(None, [])
    assert result["kind"] == "iou_match"
    assert result["trait"] is None


def test_resolve_match_criterion_iou_match_derives_a_real_threshold_not_pinned_0_5():
    """iou_match's threshold must be genuinely derived from the GT in hand
    (derive_iou_match_threshold), not pinned to 0.5."""
    from tcip_mcp.pipelines.derivations import IOU_MATCH_DERIVATION

    trait = fx.entry("leaf", ("leaf_length",), localization="iou_match")
    # char size 300 -> derived threshold well above 0.5 (see test_derive_iou_match_threshold_*).
    large_boxes = [(0, 0, 300, 300), (500, 0, 300, 300)]
    result = resolve_match_criterion(trait, gt_only(large_boxes))
    assert result["kind"] == "iou_match"
    assert result["iou_threshold"] > 0.5
    assert result["derived_from"] == IOU_MATCH_DERIVATION


def test_resolve_match_criterion_iou_match_refuses_a_reference_with_no_box_to_derive_from():
    """No conventional IoU stands in for a threshold the reference cannot derive."""
    trait = fx.entry("leaf", ("leaf_length",), localization="iou_match")
    with pytest.raises(ValueError, match="no ground-truth box"):
        resolve_match_criterion(trait, [], iou_threshold=0.42)


# --------------------------------------------------------------------------
# Box and mask mAP exact values on fixed synthetic records.
# --------------------------------------------------------------------------

def test_golden_box_and_mask_map():
    gt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0}]
    dt_perfect = [{"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}]
    m = detection_metrics([_rec(gt, dt_perfect)], trait=None, conf_threshold=0.25,
                          iou_threshold=0.5, by_mask=False)
    assert (m["map"], m["map50"]) == pytest.approx((1.0, 1.0))
    assert (m["precision"], m["recall"], m["f1"]) == pytest.approx((1.0, 1.0, 1.0))
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 0)

    # 1 TP + 1 FP against 2 GT → P=R=F1=0.5, counts 1/1/1.
    gt2 = [
        {"category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0},
        {"category_id": 1, "bbox": [50, 50, 10, 10], "iscrowd": 0},
    ]
    dt2 = [
        {"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
        {"category_id": 1, "bbox": [80, 80, 10, 10], "score": 0.7},
    ]
    m2 = detection_metrics([_rec(gt2, dt2)], trait=None, conf_threshold=0.25, iou_threshold=0.5,
                           by_mask=False)
    assert (m2["precision"], m2["recall"], m2["f1"]) == pytest.approx((0.5, 0.5, 0.5))
    assert (m2["tp"], m2["fp"], m2["fn"]) == (1, 1, 1)

    # segm: a perfect mask match scores map50 = 1.0.
    ring = [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)]
    gt_s = [{"category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0, "rings": [ring]}]
    dt_s = [{"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9, "rings": [ring]}]
    ms = detection_metrics([_rec(gt_s, dt_s)], trait=None, conf_threshold=0.25,
                           iou_threshold=0.5, by_mask=True)
    assert ms["governing_criterion"]["kind"] == "mask_iou_match"
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
    m = ordinal_metrics(torch.tensor([0, 1, 2]), torch.tensor([0, 2, 2]), 3)
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["rank_acc"] == pytest.approx(2 / 3)
    assert m["quadratic_weighted_kappa"] == pytest.approx(0.8)


def test_regression_metrics():
    m = regression_metrics(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 4.0]))
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["rmse"] == pytest.approx(math.sqrt(1 / 3))
    assert m["r_squared"] == pytest.approx(11 / 14)


def test_quadratic_weighted_kappa_perfect_agreement_is_one():
    kappa = quadratic_weighted_kappa(torch.tensor([0, 1, 2, 1]), torch.tensor([0, 1, 2, 1]), 3)
    assert kappa == pytest.approx(1.0)


def test_quadratic_weighted_kappa_hand_computed():
    # true=[0,2,2], pred=[0,1,2]: one item off by one rank out of three, worked out by hand
    # against the expected-disagreement-under-independence formula -> kappa = 1 - 1/5.
    kappa = quadratic_weighted_kappa(torch.tensor([0, 1, 2]), torch.tensor([0, 2, 2]), 3)
    assert kappa == pytest.approx(0.8)


def test_quadratic_weighted_kappa_reads_a_fractional_prediction_at_its_nearest_rank():
    """A head that emits continuous rank estimates is scored at the rank each estimate is nearest
    to, so predictions that all round onto the true ranks are perfect agreement. Reading 1.6 as
    rank 1 would report a disagreement the model does not have."""
    fractional = torch.tensor([0.4, 1.6, 2.4, 2.6])
    gt = torch.tensor([0, 2, 2, 3])
    assert quadratic_weighted_kappa(fractional, gt, 4) == pytest.approx(1.0)


def test_quadratic_weighted_kappa_scores_a_single_rank_guess_at_chance():
    """Chance correction comes from the scored set's own rank marginals, so a predictor that
    always guesses the majority rank scores 0 however skewed that majority is: here it is exactly
    right on 7 of 10 items and still says nothing beyond the marginals."""
    gt = torch.tensor([0, 0, 0, 0, 0, 0, 0, 3, 3, 3])
    always_majority = torch.zeros(10)
    assert int((always_majority == gt).sum()) == 7
    assert quadratic_weighted_kappa(always_majority, gt, 4) == pytest.approx(0.0, abs=1e-9)


def test_quadratic_weighted_kappa_empty_is_none():
    assert quadratic_weighted_kappa(torch.tensor([]), torch.tensor([]), 3) is None


def test_quadratic_weighted_kappa_degenerate_single_rank_is_none():
    # every item shares the same true and predicted rank: no expected disagreement to correct for.
    kappa = quadratic_weighted_kappa(torch.tensor([1, 1, 1]), torch.tensor([1, 1, 1]), num_ranks=3)
    assert kappa is None


def test_r_squared_perfect_fit_is_one():
    assert r_squared(
        torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])
    ) == pytest.approx(1.0)


def test_r_squared_worse_than_mean_baseline_is_negative():
    r2 = r_squared(torch.tensor([5.0, -5.0, 5.0]), torch.tensor([1.0, 2.0, 3.0]))
    assert r2 < 0


def test_r_squared_empty_is_none():
    assert r_squared(torch.tensor([]), torch.tensor([])) is None


def test_r_squared_constant_gt_is_none():
    assert r_squared(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([5.0, 5.0, 5.0])) is None


def test_concordance_correlation_coefficient_perfect_agreement_is_one():
    ccc = concordance_correlation_coefficient(
        torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])
    )
    assert ccc == pytest.approx(1.0)


def test_concordance_correlation_coefficient_hand_computed():
    # pred=[1,2,3], gt=[1,2,4] (the same pair test_regression_metrics uses), population statistics:
    # pred_mean=2, gt_mean=7/3; pred_var=mean((pred-2)^2)=(1+0+1)/3=2/3;
    # gt_var=mean((gt-7/3)^2)=((-4/3)^2+(-1/3)^2+(5/3)^2)/3=(16/9+1/9+25/9)/3=(42/9)/3=14/9;
    # covariance=mean((pred-2)*(gt-7/3))=((-1)*(-4/3)+0*(-1/3)+1*(5/3))/3=(4/3+5/3)/3=1;
    # CCC = 2*covariance / (pred_var+gt_var+(pred_mean-gt_mean)^2)
    #     = 2*1 / (2/3+14/9+1/9) = 2 / (7/3) = 6/7.
    ccc = concordance_correlation_coefficient(
        torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 4.0])
    )
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
    assert _selection_value(
        "detection", {"val_loss": 0.1, "val_objective": 5.0}, 0.2, "objective"
    ) == 5.0
    assert _selection_value("classification", {"val_loss": 0.1}, 0.2, "loss") == 0.1
    # No validation loader ran (val_metrics empty): selecting on loss falls back to the
    # training loss.
    assert _selection_value("detection", {}, 0.2, "loss") == 0.2


def test_selection_value_raises_when_the_metric_is_not_among_this_epochs_val_metrics():
    with pytest.raises(ValueError, match="not among this epoch's validation metrics"):
        _selection_value("detection", {"val_loss": 0.1}, 0.2, "objective")


def test_resolve_selection_metric_defaults():
    assert resolve_selection_metric("detection", None, None) == "objective"
    assert resolve_selection_metric("instance_seg", None, None) == "objective"
    assert resolve_selection_metric("classification", None, None) == "loss"
    assert resolve_selection_metric("semantic_seg", None, None) == "loss"


def test_resolve_selection_metric_with_no_val_loader_accepts_only_loss():
    assert resolve_selection_metric("classification", None, None, has_val_loader=False) == "loss"
    assert resolve_selection_metric(
        "classification", None, "loss", has_val_loader=False) == "loss"
    with pytest.raises(ValueError, match="needs a validation loader"):
        resolve_selection_metric("detection", None, None, has_val_loader=False)
    with pytest.raises(ValueError, match="needs a validation loader"):
        resolve_selection_metric("classification", None, "accuracy", has_val_loader=False)


@_with_bud_trait
def test_resolve_selection_metric_rejects_incoherent_explicit_choice(tmp_path):
    trait = fx.latest("bud_opening", tmp_path)
    with pytest.raises(ValueError, match="comparability-only"):
        resolve_selection_metric("detection", trait, "map50")


@_with_bud_trait
def test_resolve_selection_metric_allows_coherent_explicit_choice(tmp_path):
    # A legitimate explicit choice must still succeed: a rail must admit valid work, not
    # only reject invalid work.
    trait = fx.latest("bud_opening", tmp_path)
    assert resolve_selection_metric("detection", trait, "f1") == "f1"
    assert resolve_selection_metric("detection", trait, "recall") == "recall"
    assert resolve_selection_metric("detection", None, "map50") == "map50"  # no trait -> no gate


def test_resolve_selection_metric_rejects_a_metric_with_no_declared_direction():
    with pytest.raises(ValueError, match="no declared ranking direction"):
        resolve_selection_metric("detection", None, "not_a_real_metric")


# HIGHER_IS_BETTER_BY_METRIC held against what evaluate()/governing_counts really return.

def _detection_batch(num_images: int = 2, img_size: int = 64):
    from tcip_mcp.pipelines.training.collation import task_collate

    items = []
    boxes = [[10.0, 10.0, 40.0, 40.0], [5.0, 5.0, 25.0, 30.0]]
    for i in range(num_images):
        img = torch.rand(3, img_size, img_size)
        target = {"boxes": torch.tensor([boxes[i % len(boxes)]]),
                  "labels": torch.ones((1,), dtype=torch.long),
                  "iscrowd": torch.zeros((1,), dtype=torch.long), "image_id": i}
        items.append((img, target))
    return task_collate("detection")(items)


@_with_bud_trait
def test_higher_is_better_by_metric_matches_evaluate_and_governing_counts(tmp_path):
    """The declaration is exactly the numeric keys these two producers return: nothing declared
    that neither ever produces, nothing either produces that the declaration leaves unaccounted
    for (declared, or named in the not-a-ranking list below, or a ``_state`` companion)."""
    pytest.importorskip("torchvision")
    from tcip_store.values import NOT_FINITE_SUFFIX

    from tcip_mcp.pipelines.model_contract import _SYNTHESIZABLE_TASKS
    from tcip_mcp.pipelines.training.evaluation import (
        HIGHER_IS_BETTER_BY_METRIC,
        evaluate,
        governing_counts,
    )
    from tests import bespoke_models

    device = torch.device("cpu")
    img_size = 64
    returned: set[str] = set()

    for task in sorted(_SYNTHESIZABLE_TASKS):
        if task == "detection":
            model = bespoke_models.build_bespoke_detection(num_classes=1)
            images, targets = _detection_batch(img_size=img_size)
            loader = [(images, targets)]
        elif task == "instance_seg":
            model = bespoke_models.build_bespoke_instance_seg(num_classes=1)
            images, targets = _detection_batch(img_size=img_size)
            for t in targets:
                mask = torch.zeros((1, img_size, img_size), dtype=torch.uint8)
                mask[0, 10:40, 10:40] = 1
                t["masks"] = mask
            loader = [(images, targets)]
        elif task == "classification":
            model = bespoke_models.build_bespoke_classifier(num_classes=2)
            imgs = torch.stack([torch.rand(3, img_size, img_size) for _ in range(2)])
            loader = [(imgs, {"labels": torch.tensor([0, 1])})]
        elif task == "ordinal":
            model = bespoke_models.build_bespoke_ordinal(num_ranks=3)
            imgs = torch.stack([torch.rand(3, img_size, img_size) for _ in range(2)])
            loader = [(imgs, {"ranks": torch.tensor([0, 2])})]
        elif task == "regression":
            model = bespoke_models.build_bespoke_regressor()
            imgs = torch.stack([torch.rand(3, img_size, img_size) for _ in range(2)])
            loader = [(imgs, {"values": torch.tensor([0.2, 0.8])})]
        else:
            model = bespoke_models.build_bespoke_semantic_seg(num_classes=2)
            imgs = torch.stack([torch.rand(3, img_size, img_size) for _ in range(2)])
            m0 = torch.zeros((img_size, img_size), dtype=torch.long)
            m0[:, : img_size // 2] = 1
            m1 = 1 - m0
            loader = [(imgs, {"masks": torch.stack([m0, m1])})]

        # "bud_opening" (seeded center_match) exercises evaluate()'s center-match branch for
        # detection.
        trait = fx.latest("bud_opening", tmp_path) if task == "detection" else None
        dims = {"ordinal": {"num_ranks": 3}, "regression": {}}.get(task, {"num_classes": 2})
        result = evaluate(model, loader, device, task, dims={"in_chans": 3, **dims}, trait=trait)
        returned.update(result)

    per_image = [
        {"width": 64, "height": 64,
         "gt": [{"category_id": 1, "bbox": [10.0, 10.0, 30.0, 30.0], "iscrowd": 0}],
         "dt": [{"category_id": 1, "bbox": [11.0, 11.0, 29.0, 29.0], "score": 0.9}]},
        {"width": 64, "height": 64, "gt": [], "dt": []},
    ]
    returned.update(governing_counts(
        per_image, {"kind": "center_match", "tolerance": 5.0}, conf_threshold=0.25))

    not_a_ranking = {
        "per_class", "count_bias", "per_class_iou", "per_class_dice", "tp", "fp", "fn",
        "criterion", "governing_criterion", "map50_role",
    }
    ranking_returned = {k for k in returned if not k.endswith(NOT_FINITE_SUFFIX)}
    declared = set(HIGHER_IS_BETTER_BY_METRIC)

    undeclared = (ranking_returned - not_a_ranking) - declared
    assert not undeclared, f"returned but no declared direction: {sorted(undeclared)}"
    never_produced = declared - ranking_returned
    assert not never_produced, (
        f"declared but neither producer ever returns it: {sorted(never_produced)}"
    )
    assert ranking_returned - declared == not_a_ranking


def test_both_eval_regimes_share_common_keys_and_keep_their_own_apart(tmp_path, monkeypatch):
    """run_test_evaluation and run_full_frame_evaluation compose through one shared
    evaluation_result: both regimes' results carry the same common identity keys by name and
    presence, and the test regime carries none of the full-frame regime's fields."""
    from PIL import Image

    import tcip_mcp.pipelines.training.evaluation as evaluation
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation
    from tests._producer_fixtures import label_image
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    common_fields = {
        "model_path", "task", "checkpoint_sha256", "experiment_id",
        "iou_threshold", "execution", "eval_regime",
    }
    full_frame_only_fields = {
        "scored_images", "tallies", "max_dets_cap_saturated_frac",
    }

    from tcip_mcp.pipelines.execution import prepare_pass

    ckpt_path = registered_checkpoint(tmp_path)
    monkeypatch.setattr(evaluation, "evaluate",
                        lambda *a, **k: {"loss": 0.1, "precision": 0.4, "recall": 0.5, "f1": 0.44})
    checkpoint = load_registered_checkpoint(ckpt_path, project=tmp_path)
    test_result = run_test_evaluation(prepare_pass(checkpoint, Stated(tile=False)), None, "cpu")

    images_dir = tmp_path / "ff" / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(images_dir / "a.png")
    label_image(images_dir / "a.png", [Annotation(subject="bud", geometry=BBox(4, 4, 12, 12))],
                32, 32)

    from tests._predictor_fixtures import StubPredictor, install

    install(monkeypatch, StubPredictor(task="detection", in_chans=3, width=32, height=32,
                                       boxes=(), scores=()))
    ff_result = run_full_frame_evaluation(
        checkpoint, checkpoint_admission(checkpoint, images_dir),
        stated=Stated(tile_size=32, overlap=0.0))

    for field in common_fields:
        assert field in test_result, f"{field} missing from the test-regime record"
        assert field in ff_result, f"{field} missing from the full-frame-regime record"
    assert not (full_frame_only_fields & set(test_result))


def test_evaluation_result_refuses_a_key_extra_shares_with_common():
    """A key present in both common and extra is a programming error, not a precedence rule:
    extra silently shadowing a common identity field (or the reverse) would defeat the
    unification evaluation_result exists to enforce, so this refuses naming the key rather than
    pick a winner."""
    common = {
        "model_path": "m.pt", "task": "detection", "checkpoint_sha256": "abc",
        "experiment_id": "e1",
        "iou_threshold": 0.5, "execution": {"conf": 0.3, "max_dets": 100},
        "eval_regime": "full-frame-single-pass",
    }
    extra = {"precision": 0.9, "task": "classification"}
    with pytest.raises(ValueError, match="task"):
        evaluation_result(common, extra)


# --------------------------------------------------------------------------
# Light integration: _validate via train()
# --------------------------------------------------------------------------

torchvision = pytest.importorskip("torchvision")
from torch.utils.data import DataLoader  # noqa: E402

from tcip_mcp.pipelines.training.generic_trainer import train
from tcip_mcp.pipelines.training.collation import task_collate
from tests._producer_fixtures import run_over  # noqa: E402
from tests.tiny_trainer_fixtures import trainer_run  # noqa: E402

IMG = 64


_save_png = partial(write_noise_image, size=IMG)


def _four_bud_images(tmp_path):
    """Four noise images in ``tmp_path``'s undated capture, each labeled with one centered
    ``bud`` box."""
    from tcip_annotation.state import Annotation, BBox

    from tests._producer_fixtures import label_image

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    for i in range(4):
        _save_png(images_dir / f"img{i}.png")
        label_image(images_dir / f"img{i}.png",
                    [Annotation(subject="bud", geometry=BBox(19.2, 19.2, 44.8, 44.8))], IMG, IMG)
    return images_dir


def _stored(tmp_path, annotations, size: int = 100):
    """``annotations`` written as the label document of a ``size``-pixel image ``a`` under
    ``tmp_path`` and read back."""
    from tcip_annotation import json_io

    from tcip_mcp.dataset_layout import label_key

    key = label_key(tmp_path, UNDATED_BUCKET, "a")
    json_io.write_label_document(key, annotations, size, size, keep_empty=True)
    return json_io.read_label_document(key)


def _cfg(model_source, data: dict) -> dict:
    return {
        "model_source": model_source, "data": data, "device": "cpu",
        "stages": [{"freeze_to": -1, "epochs": 1}], "mixed_precision": False,
        "optimizer": {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 0},
        "early_stopping": {"enabled": False},
    }


def test_validate_detection_returns_metrics_and_objective(tmp_path):
    ds, data = run_over("detection", str(_four_bud_images(tmp_path)), subject="bud")
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("detection"))

    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": {"min_size": IMG, "max_size": IMG * 2},
                    "task": "detection"}
    run = trainer_run(_cfg(model_source, data), tmp_path / "out", project=tmp_path,
                      has_val_loader=True, id="auto-run-23")
    run = train(run, loader, val_loader=loader)  # no AttributeError on model.heads

    last = run.metrics_history[-1]
    for k in ("val_loss", "val_precision", "val_recall", "val_f1", "val_map50", "val_map",
              "val_objective"):
        assert k in last, f"missing {k}"
    # An untrained toy detector finds nothing on four images: the epoch has no useful score, so
    # no epoch is selectable and the run fails naming the stage rather than delivering weights.
    assert last["val_objective"] is None
    assert math.isnan(last["selection"])
    assert run.status == "failed"
    assert "stage 0 produced no selectable epoch" in run.status_error
    assert not (tmp_path / "out" / "model_best.pt").is_file()
    assert not (tmp_path / "out" / "model_final.pt").is_file()
    assert run.best_metric == math.inf


@_with_bud_trait
def test_train_center_match_trait_records_governing_criterion(tmp_path):
    """Threading `trait` into _validate surfaces val_governing_criterion (a dict) and
    val_map50_role (a str) in val_metrics; the TensorBoard scalar loop must skip these
    non-numeric values rather than crash `add_scalar` on them. The untrained toy detector never
    scores, so the run then fails as having no selectable epoch, after the row is logged."""
    from tcip_mcp.experiments import METRICS_FILE
    from tcip_mcp.pipelines.training.envelope import TrainContext

    ds, data = run_over("detection", str(_four_bud_images(tmp_path)), subject="bud")
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("detection"))

    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": {"min_size": IMG, "max_size": IMG * 2},
                    "task": "detection"}
    cfg = _cfg(model_source, data)
    cfg["evaluation"] = {"trait": "bud_opening"}
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / METRICS_FILE).touch()
    run = trainer_run(cfg, out_dir, project=tmp_path, has_val_loader=True, id="auto-run-24")
    ctx = TrainContext(run=run, train_loader=loader, val_loader=loader)
    run = ctx.default_train()
    ctx.tb.close()

    assert "no selectable epoch" in run.status_error
    last = run.metrics_history[-1]
    assert "val_governing_criterion" in last
    assert last["val_map50_role"] == "comparability_only"
    assert last["selection_metric"] == "objective"
    assert last["selection_trait"] == "bud_opening"


def test_validate_classification_metrics(tmp_path):
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    rows = []
    for i in range(6):
        _save_png(images_dir / f"img{i}.png")
        rows.append((f"img{i}", i % 2))
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("stem", "label"))
        w.writerows(rows)
    ds, data = run_over("classification", str(images_dir), str(csv_path))
    loader = DataLoader(ds, batch_size=3, collate_fn=task_collate("classification"))

    model_source = {"builder": "tests.bespoke_models:build_bespoke_classifier",
                    "task": "classification"}
    run = trainer_run(_cfg(model_source, data), tmp_path / "out", project=tmp_path,
                      has_val_loader=True, id="auto-run-25")
    run = train(run, loader, val_loader=loader)

    assert run.status == "completed", run.status_error
    last = run.metrics_history[-1]
    assert "val_accuracy" in last and "val_f1" in last
    assert run.best_metric == pytest.approx(last["val_loss"])  # selection falls back to val_loss


def test_score_predictions_folder_counts_every_image_through_the_matcher(data_dir):
    from tcip_mcp.tools.annotation_tools import score_predictions
    from tests.conftest import DATA_DIR_BUCKET

    r = score_predictions(str(data_dir / "images" / "2-11-26"), DATA_DIR_BUCKET)
    assert "map50" in r
    # fixture: each image has 2 GT, predictions = 1 TP + 1 FP -> tp=1,fp=1,fn=1 per image
    # (x3 images).
    assert r["total_tp"] == 3 and r["total_fp"] == 3 and r["total_fn"] == 3
    assert r["precision"] == pytest.approx(0.5)
    assert all(p["tp"] == 1 and p["fp"] == 1 and p["fn"] == 1 for p in r["per_image"])


# -- a crowd region: an ignore region, never one object ------------------------------------

_OBJECT = [10.0, 10.0, 20.0, 20.0]
_CROWD = [50.0, 50.0, 40.0, 40.0]


def _crowd_records() -> list[dict]:
    """One image whose object a detection matches and whose crowd region holds another
    detection, and one image holding only a crowd region no detection matches."""
    from tcip_mcp.pipelines.training.evaluation import gt_record

    return [
        image_record(100, 100, [gt_record(_OBJECT, 1, 0), gt_record(_CROWD, 1, 1)],
                                [{"category_id": 1, "bbox": _OBJECT, "score": 0.9},
                                 {"category_id": 1, "bbox": [55.0, 55.0, 30.0, 30.0],
                                  "score": 0.9}]),
        image_record(100, 100, [gt_record(_CROWD, 1, 1)], []),
    ]


def test_a_detection_in_a_crowd_region_is_neither_true_nor_false_and_the_region_no_miss():
    m = detection_metrics(_crowd_records(), trait=None, iou_threshold=0.5, conf_threshold=0.25,
                          by_mask=False)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 0)


def test_the_center_match_count_treats_a_crowd_region_as_the_iou_count_does():
    from tcip_mcp.pipelines.training.evaluation import governing_counts

    counts = governing_counts(_crowd_records(), {"kind": "center_match", "tolerance": 3.0},
                              conf_threshold=0.25)
    assert (counts["tp"], counts["fp"], counts["fn"]) == (1, 0, 0)


def test_a_targets_records_are_one_shape_on_the_stored_grid_whatever_the_target_holds(tmp_path):
    """The loader's own target, as lists, as an array and as tensors, becomes the same evaluation
    records, each box on the stored two-decimal grid: every reader of a target reads it here."""
    import numpy as np

    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.data.datasets import target_tensors
    from tcip_mcp.pipelines.data.label_queries import json_det_targets
    from tcip_mcp.pipelines.training.evaluation import gt_records

    document = _stored(tmp_path, [
        Annotation(subject="bur", geometry=BBox(1.25, 2.5, 30.75, 40.5)),
        Annotation(subject="bur", geometry=BBox(50.0, 50.0, 90.0, 90.0), iscrowd=True)])
    listed = json_det_targets(document.annotations, registry_scope(tmp_path, "bur"))
    arrays = {k: np.asarray(v) for k, v in listed.items()}
    off_grid = {**listed, "boxes": [[1.2504, 2.5, 30.7496, 40.5], [50.0, 50.0, 90.0, 90.0]]}

    expected = [{"category_id": 1, "bbox": [1.25, 2.5, 29.5, 38.0], "iscrowd": 0},
                {"category_id": 1, "bbox": [50.0, 50.0, 40.0, 40.0], "iscrowd": 1}]
    regions = [{"rings": [[(1.25, 2.5), (30.75, 2.5), (30.75, 40.5), (1.25, 40.5)]]},
               {"rings": [[(50.0, 50.0), (90.0, 50.0), (90.0, 90.0), (50.0, 90.0)]]}]
    for target in (listed, arrays, target_tensors(listed), off_grid):
        # A target carrying its rows' geometry states each row's region beside its box.
        assert gt_records(target) == ([{**e, **r} for e, r in zip(expected, regions)]
                                      if "geometry" in target else expected)
    flagless = {"boxes": [[0, 0, 10, 10]], "labels": [1]}
    with pytest.raises(ValueError, match="iscrowd"):
        gt_records(flagless)


def test_a_detector_output_stating_no_boxes_refuses_rather_than_reading_as_no_detections():
    from tcip_mcp.pipelines.training.evaluation import records_from_detector

    target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64),
              "iscrowd": torch.zeros((0,), dtype=torch.int64)}
    with pytest.raises(KeyError, match="boxes"):
        records_from_detector(target, {}, width=10, height=10)
    empty = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64),
             "scores": torch.zeros((0,))}
    assert records_from_detector(target, empty, width=10, height=10)["dt"] == []


def test_a_ground_truth_record_is_one_shape_from_a_target_and_from_its_annotation(tmp_path):
    """The loader's target route and the annotation route build a document's ground truth as
    the same records, its box, crowd flag and region stated alike, so the scorer fills nothing
    in."""
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.data.label_queries import json_det_targets
    from tcip_mcp.pipelines.training.evaluation import gt_records, records_from_annotation

    document = _stored(tmp_path, [
        Annotation(subject="bur", geometry=BBox(10.1, 10.1, 40.3, 30.3)),
        Annotation(subject="bur", geometry=BBox(50.0, 50.0, 90.0, 90.0), iscrowd=True)])
    listed = json_det_targets(document.annotations, registry_scope(tmp_path, "bur"))
    record = records_from_annotation(document.annotations, [], width=100,
                                     height=100, name_id={"bur": 1})
    # The annotation route also names each record's annotation, by its index in the document.
    assert [
        {k: v for k, v in g.items() if k != "index"} for g in record["gt"]
    ] == gt_records(listed)
    assert [g["index"] for g in record["gt"]] == [0, 1]


def test_a_detectors_record_reads_both_sides_on_the_stored_grid(tmp_path):
    """The training-time evaluation scores the box the document holds, never the loader's
    float32 corners, and each detection on the grid its ground truth is stored on."""
    torch = pytest.importorskip("torch")

    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.data.datasets import target_tensors
    from tcip_mcp.pipelines.data.label_queries import json_det_targets
    from tcip_mcp.pipelines.training.evaluation import records_from_detector

    document = _stored(tmp_path, [Annotation(subject="bur", geometry=BBox(10.3, 20.7, 40.1, 60.9))])
    listed = json_det_targets(document.annotations, registry_scope(tmp_path, "bur"))
    output = {"boxes": torch.tensor([[10.1, 10.1, 40.3, 30.3]]),
              "labels": torch.tensor([1]), "scores": torch.tensor([0.9])}
    record = records_from_detector(target_tensors(listed), output, width=100, height=100)
    assert [g["bbox"] for g in record["gt"]] == [[10.3, 20.7, 29.8, 40.2]]
    assert [d["bbox"] for d in record["dt"]] == [[10.1, 10.1, 30.2, 20.2]]


def _reference_records(tmp_path, reference: list, detections: list) -> list[dict]:
    """One image's records, its reference written and read back through the label document."""
    from tcip_mcp.pipelines.training.evaluation import records_from_annotation

    return [records_from_annotation(_stored(tmp_path, reference).annotations, detections,
                                    width=100, height=100)]


@pytest.mark.parametrize("crowd_only", [True, False], ids=["crowd_only", "empty"])
def test_a_detection_outside_a_reference_with_no_object_is_a_false_positive(tmp_path, crowd_only):
    """A reference holding no object, crowd regions alone or nothing, is still evaluated: a
    detection outside every crowd region is one false positive, by the IoU convention and by the
    center match alike, never hidden by a zero-object shortcut."""
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.training.evaluation import governing_counts

    reference = [Annotation(subject="bur", geometry=BBox(50, 50, 90, 90), iscrowd=True,
                            created_by="user:breeder")] if crowd_only else []
    detection = Annotation(subject="bur", geometry=BBox(5, 5, 20, 20), score=0.9)
    records = _reference_records(tmp_path, reference, [detection])

    m = detection_metrics(records, trait=None, iou_threshold=0.5, conf_threshold=0.25,
                          by_mask=False)
    assert (m["tp"], m["fp"], m["fn"]) == (0, 1, 0)
    counts = governing_counts(records, {"kind": "center_match", "tolerance": 3.0},
                              conf_threshold=0.25)
    assert (counts["tp"], counts["fp"], counts["fn"]) == (0, 1, 0)


def test_a_reference_with_objects_and_no_detection_scores_every_object_a_miss(tmp_path):
    from tcip_annotation.state import Annotation, BBox

    records = _reference_records(tmp_path, [
        Annotation(subject="bur", geometry=BBox(5, 5, 20, 20), created_by="user:breeder"),
        Annotation(subject="bur", geometry=BBox(40, 40, 60, 60), created_by="user:breeder")], [])

    m = detection_metrics(records, trait=None, iou_threshold=0.5, conf_threshold=0.25,
                          by_mask=False)
    assert (m["tp"], m["fp"], m["fn"]) == (0, 0, 2)


def test_a_crowd_region_is_no_object_in_a_ground_truth_count(tmp_path):
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.data.splits import count_label_lines
    from tcip_mcp.pipelines.training.evaluation import gt_class_typical_count

    assert gt_class_typical_count(_crowd_records()) == 1.0
    document = _stored(tmp_path, [
        Annotation(subject="bur", geometry=BBox(1, 1, 9, 9)),
        Annotation(subject="bur", geometry=BBox(20, 20, 60, 60), iscrowd=True)])
    assert count_label_lines(document, ClassScope(subject="bur")) == 1
    assert count_label_lines(document, ClassScope()) == 1
