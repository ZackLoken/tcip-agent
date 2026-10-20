"""Evaluation metrics + composite selection objective.

Unit tests for the one matcher's detection metrics, the composite objective,
the in-house scalar metrics, and ``_selection_value``; plus light integration
tests that exercise the detection/classification ``_validate`` path end-to-end.
"""

from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING, Unpack

import pytest

from tests._chain_fixtures import BESPOKE_CLASSIFIER, FRAME_STATISTICS

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor, device as torch_device, nn
    from torch.utils.data import DataLoader

    from tcip_annotation.json_io import LabelDocument
    from tcip_annotation.state import Annotation
    from tcip_mcp.pipelines.schemas import TrainConfigSchema
    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.pipelines.training.evaluation import EvaluationKeywords

torch = pytest.importorskip(
    "torch"
)  # evaluation.py imports torch at module load

from tcip_annotation.matching import match_pairs  # noqa: E402
from tcip_mcp.dataset_layout import UNDATED_BUCKET  # noqa: E402
from tcip_mcp.pipelines.data.label_queries import registry_scope  # noqa: E402
from tcip_mcp.pipelines.data.selection import ClassScope  # noqa: E402
from tcip_mcp.pipelines.execution import Stated  # noqa: E402
from tcip_mcp.pipelines.schemas import COMPARABILITY_IOU, ScoreWeights  # noqa: E402
from tcip_mcp.pipelines.training.eval_runners import (  # noqa: E402
    run_test_evaluation,
)
from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    classification_metrics,
    comparability_criterion,
    compute_composite_objective,
    concordance_correlation_coefficient,
    derive_operating_point_curve,
    detection_metrics,
    dt_record,
    gt_class_avg_size,
    gt_record,
    ordinal_metrics,
    pick_count_unbiased,
    pick_f1_max,
    prediction_record,
    quadratic_weighted_kappa,
    r_squared,
    regression_metrics,
    resolve_match_criterion,
)
from tcip_mcp.pipelines.training.generic_trainer import (  # noqa: E402
    _selection_value,
    resolve_selection_metric,
)
from tests import _trait_fixtures as fx  # noqa: E402
from tests._dense_op_fixtures import gt_only, toy_records  # noqa: E402
from tests._image_fixtures import write_noise_image  # noqa: E402
from tests._producer_fixtures import checkpoint_admission  # noqa: E402
from tests._training_values import (  # noqa: E402
    DISPERSION_QUANTILE,
    IOU_THRESHOLD,
    MASK_CUT,
    RATIONALE,
    SCORE_WEIGHTS,
    VALIDATION_CONF,
    drawn_split,
    evaluation_block,
)

WEIGHTS = ScoreWeights.model_validate(SCORE_WEIGHTS)

# A test naming trait="bud_opening" proposes it in its project (conftest.seed_bud_trait_spec).
_with_bud_trait = pytest.mark.usefixtures("seed_bud_trait_spec")


# --------------------------------------------------------------------------
# Composite objective
# --------------------------------------------------------------------------


def test_composite_objective_matches_reference() -> None:
    """The objective weights each term by the weights it is handed."""
    expected = 0.5 * 2.0 + 0.3 * 0.5 * 10 + 0.2 * 0.6 * 10  # 3.7
    assert compute_composite_objective(
        2.0, 0.5, 0.4, WEIGHTS
    ) == pytest.approx(expected, abs=1e-9)


def test_composite_objective_has_no_score_for_a_degenerate_epoch() -> None:
    """The composite objective is None for a degenerate epoch.

    A non-positive or non-finite loss, or both quality terms at zero, is an epoch with no
    useful score: the objective is None.
    """
    w = WEIGHTS
    assert compute_composite_objective(-1.0, 0.9, 0.9, w) is None
    assert compute_composite_objective(2.0, 0.0, 0.0, w) is None
    assert (
        compute_composite_objective(
            float("nan"), float("nan"), float("nan"), w
        )
        is None
    )
    assert compute_composite_objective(float("inf"), 0.5, 0.5, w) is None


# --------------------------------------------------------------------------
# detection metrics under the IoU convention
# --------------------------------------------------------------------------


def _rec(gt: list[dict], dt: list[dict], w: int = 100, h: int = 100) -> dict:
    return prediction_record(
        dt, gt, width=w, height=h, cap=None, count=len(dt)
    )


def _gt(*boxes: list[float]) -> list[dict]:
    """Return one class-1 object per ``[x, y, w, h]`` box, each as ``gt_record`` makes it."""
    return [gt_record(box, 1, 0) for box in boxes]


def _dt(*scored: tuple[list[float], float]) -> list[dict]:
    """Return one class-1 detection per ``([x, y, w, h], score)``, as ``dt_record`` makes it."""
    return [dt_record(box, 1, score) for box, score in scored]


def _masked(gt_ring: list, dt_ring: list) -> dict:
    """Return the ``records_from_annotation`` record of one masked object and one detection.

    The frame is 100 x 100, holding one object of region ``gt_ring`` and one detection at 0.9
    of region ``dt_ring``.
    """
    from tcip_annotation.state import Annotation, Polygon
    from tcip_mcp.pipelines.training.evaluation import records_from_annotation

    return records_from_annotation(
        [Annotation(subject="bud", geometry=Polygon([gt_ring]))],
        [Annotation(subject="bud", geometry=Polygon([dt_ring]), score=0.9)],
        width=100,
        height=100,
        cap=None,
    )


def test_detection_map50_perfect() -> None:
    gt = _gt([10, 10, 20, 20])
    dt = _dt(([10, 10, 20, 20], 0.9))
    m = detection_metrics(
        [_rec(gt, dt)],
        trait=None,
        conf_threshold=0.25,
        iou_threshold=0.5,
        by_mask=False,
    )
    assert m["map50"] == pytest.approx(1.0)
    assert m["map"] == pytest.approx(1.0)


def test_detection_operating_point_tp_fp_fn() -> None:
    gt = _gt([10, 10, 20, 20], [50, 50, 10, 10])
    dt = _dt(([10, 10, 20, 20], 0.9), ([80, 80, 10, 10], 0.7))  # a TP, an FP
    m = detection_metrics(
        [_rec(gt, dt)],
        trait=None,
        conf_threshold=0.25,
        iou_threshold=0.5,
        by_mask=False,
    )
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["map50"] > 0.0


def test_detection_conf_threshold_filters() -> None:
    gt = _gt([10, 10, 20, 20])
    # A TP, and an FP below 0.8.
    dt = _dt(([10, 10, 20, 20], 0.9), ([80, 80, 10, 10], 0.7))
    m = detection_metrics(
        [_rec(gt, dt)],
        trait=None,
        conf_threshold=0.8,
        iou_threshold=0.5,
        by_mask=False,
    )
    assert m["tp"] == 1
    assert m["fp"] == 0


def test_detection_metrics_write_nothing_to_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    gt = _gt([10, 10, 20, 20])
    dt = _dt(([10, 10, 20, 20], 0.9))
    detection_metrics(
        [_rec(gt, dt)],
        trait=None,
        conf_threshold=0.25,
        iou_threshold=0.5,
        by_mask=False,
    )
    captured = capsys.readouterr()
    assert captured.out == ""  # nothing reaches stdout (MCP stdio safety)


def test_detection_mask_path() -> None:
    ring = [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)]
    m = detection_metrics(
        [_masked(ring, ring)],
        trait=None,
        conf_threshold=0.25,
        iou_threshold=0.5,
        by_mask=True,
    )
    assert m["map50"] == pytest.approx(1.0)
    assert m["governing_criterion"]["kind"] == "mask_iou_match"


def test_a_mask_match_counts_by_region_where_the_box_match_would_not() -> None:
    """A detection matched by box is missed by mask when their regions do not overlap.

    An L-shaped object whose box the box of a detection filling its inner corner overlaps at
    IoU 0.5625, while their regions do not overlap: matched by box, missed by mask.
    """
    l_shape = [
        (0.0, 0.0),
        (40.0, 0.0),
        (40.0, 10.0),
        (10.0, 10.0),
        (10.0, 40.0),
        (0.0, 40.0),
    ]
    corner = [(10.0, 10.0), (40.0, 10.0), (40.0, 40.0), (10.0, 40.0)]
    by_box, by_mask = (
        detection_metrics(
            [_masked(l_shape, corner)],
            trait=None,
            conf_threshold=0.25,
            iou_threshold=0.5,
            by_mask=mask,
        )
        for mask in (False, True)
    )
    assert (by_box["tp"], by_box["fp"], by_box["fn"]) == (1, 0, 0)
    assert (by_mask["tp"], by_mask["fp"], by_mask["fn"]) == (0, 1, 1)


def test_detection_metrics_over_an_empty_side() -> None:
    gt = _gt([10, 10, 20, 20])
    dt = _dt(([10, 10, 20, 20], 0.9))

    def metrics(gt: list[dict], dt: list[dict]) -> dict:
        return detection_metrics(
            [_rec(gt, dt)],
            trait=None,
            conf_threshold=0.25,
            iou_threshold=0.5,
            by_mask=False,
        )

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


def test_golden_gt_class_avg_size() -> None:
    assert gt_class_avg_size(toy_records(), class_id=0) == pytest.approx(20.0)


CENTER_10 = {"kind": "center_match", "tolerance": 10.0}
"""A center match at a 10 px tolerance."""


def _points(centers: list[tuple[float, float]]) -> list[dict]:
    """Return each center as the matcher's record of a zero-extent xywh box."""
    return [{"bbox": [x, y, 0.0, 0.0]} for x, y in centers]


def test_golden_derive_operating_point_curve() -> None:
    sweep = derive_operating_point_curve(
        toy_records(),
        criterion=CENTER_10,
        class_id=0,
        dispersion_quantile=DISPERSION_QUANTILE,
    )
    curve = sweep["curve"]
    assert [round(c["conf"], 2) for c in curve] == [0.0, 0.3, 0.6, 0.9]

    at = {round(c["conf"], 2): c for c in curve}
    # conf 0.6: image-a keeps both dets (1 TP + 1 FP), image-b drops the 0.3 det (1 TP, 1 FN).
    assert (at[0.6]["tp"], at[0.6]["fp"], at[0.6]["fn"]) == (2, 1, 1)
    assert at[0.6]["count_bias_mean"] == pytest.approx(
        0.0
    )  # +1 and -1 cancel → unbiased
    assert at[0.6]["abs_count_error_mean"] == pytest.approx(1.0)
    # conf 0.9: only the 0.9 dets survive → image-a 1 TP, image-b 1 TP + 1 FN.
    assert (at[0.9]["tp"], at[0.9]["fp"], at[0.9]["fn"]) == (2, 0, 1)
    assert at[0.9]["count_bias_mean"] == pytest.approx(-0.5)  # mean of [0, -1]


def test_golden_operating_point_pickers() -> None:
    sweep = derive_operating_point_curve(
        toy_records(),
        criterion=CENTER_10,
        class_id=0,
        dispersion_quantile=DISPERSION_QUANTILE,
    )
    assert pick_count_unbiased(sweep) == pytest.approx(0.6)  # zero count bias
    assert pick_f1_max(sweep) == pytest.approx(0.0)  # recall-max point
    at06 = next(c for c in sweep["curve"] if c["conf"] == pytest.approx(0.6))
    assert at06["count_bias_mean"] == pytest.approx(0.0)


# match_pairs: one matcher, two stated policies (score_first for the count, distance_first for
# the classifier pairing), under a center or an IoU criterion; coverage of the pinned tie rules.


def test_match_pairs_score_first_and_distance_first_disagree_on_cardinality() -> (
    None
):
    """Score-first and distance-first matching disagree on how many pairs they claim.

    Over two ground truths and a detection within tolerance of both, score-first claims one
    pair and distance-first claims two.
    """
    gt = _points([(0.0, 0.0), (10.0, 0.0)])
    dt = _points(
        [(4.0, 0.0), (0.0, 0.0)]
    )  # score 0.9 then 0.1, already score-descending
    criterion = {"kind": "center_match", "tolerance": 6.0}

    assert match_pairs(gt, dt, criterion, policy="score_first") == [(0, 0)]
    assert match_pairs(gt, dt, criterion, policy="distance_first") == [
        (0, 1),
        (1, 0),
    ]


def test_match_pairs_score_first_tie_keeps_the_last_index() -> None:
    """Among equidistant unused ground truths, score-first keeps the last index."""
    pairs = match_pairs(
        _points([(0.0, 0.0), (2.0, 0.0)]),
        _points([(1.0, 0.0)]),
        {"kind": "center_match", "tolerance": 1.0},
        policy="score_first",
    )
    assert pairs == [(1, 0)]


def test_match_pairs_distance_first_tie_breaks_by_gt_then_detection_index() -> (
    None
):
    """Distance-first breaks a tied distance by ground-truth index, then detection index.

    The tie-break is ascending, so a fully degenerate 2x2 assigns each detection to the ground
    truth sharing its own index.
    """
    pairs = match_pairs(
        _points([(0.0, 0.0), (0.0, 0.0)]),
        _points([(5.0, 0.0), (5.0, 0.0)]),
        {"kind": "center_match", "tolerance": 10.0},
        policy="distance_first",
    )
    assert pairs == [(0, 0), (1, 1)]


def test_match_pairs_under_an_iou_criterion_matches_by_overlap() -> None:
    """``match_pairs`` under an IoU criterion pairs by overlap, not by nearness.

    At a 0.5 threshold, a detection 1 px off its ground truth pairs, and one 6 px off the other
    ground truth, overlapping it at IoU 0.25, does not.
    """
    gt = [{"bbox": [0.0, 0.0, 10.0, 10.0]}, {"bbox": [100.0, 0.0, 10.0, 10.0]}]
    dt = [{"bbox": [1.0, 0.0, 10.0, 10.0]}, {"bbox": [106.0, 0.0, 10.0, 10.0]}]
    pairs = match_pairs(
        gt,
        dt,
        {"kind": "iou_match", "iou_threshold": 0.5},
        policy="score_first",
    )
    assert pairs == [(0, 0)]


def test_match_pairs_refuses_an_unknown_policy_by_name() -> None:
    """``match_pairs`` refuses a policy outside the two stated ones, naming the policy."""
    with pytest.raises(ValueError, match="policy"):
        match_pairs(
            _points([(0.0, 0.0)]),
            _points([(0.0, 0.0)]),
            {"kind": "center_match", "tolerance": 1.0},
            policy="nearest",
        )


def test_dt_score_refuses_a_record_without_a_score_or_with_none_by_name() -> (
    None
):
    """``_dt_score`` refuses a record with no score or a None score, naming which.

    ``derive_operating_point_curve`` over a detection with no score refuses with the same
    "no 'score' field" message.
    """
    from tcip_mcp.pipelines.training.evaluation import (
        _dt_score,
        derive_operating_point_curve,
    )

    with pytest.raises(ValueError, match="no 'score' field"):
        _dt_score({"bbox": [0, 0, 1, 1]})
    with pytest.raises(ValueError, match="is None"):
        _dt_score({"bbox": [0, 0, 1, 1], "score": None})
    per_image = [
        {
            "image_id": 1,
            "gt": [],
            "dt": [{"bbox": [0, 0, 1, 1], "category_id": 1}],
        }
    ]
    with pytest.raises(ValueError, match="no 'score' field"):
        derive_operating_point_curve(
            per_image,
            criterion={"kind": "center_match", "tolerance": 1.0},
            dispersion_quantile=DISPERSION_QUANTILE,
        )


# resolve_match_criterion reads the revision's stated localization and never derives one.


def test_resolve_match_criterion_refuses_an_unauthored_localization_asking_the_breeder() -> (
    None
):
    """A revision stating no localization refuses at ``resolve_match_criterion``.

    The refusal names the ``localization`` field, with ground-truth boxes in hand.
    """
    from tcip_mcp.traits import UnauthoredFieldError

    small_boxes = [(0, 0, 20, 20), (100, 0, 20, 20)]
    with pytest.raises(UnauthoredFieldError, match="localization"):
        resolve_match_criterion(
            fx.entry("leaf", ("leaf_length",)), gt_only(small_boxes)
        )


def test_resolve_match_criterion_reads_the_stated_kind_as_is() -> None:
    """A revision stating IoU resolves an IoU match over boxes small enough for a center match."""
    trait = fx.entry(
        "leaf",
        ("leaf_length",),
        localization="iou_match",
        iou_jitter_px=fx.IOU_JITTER_PX,
        iou_margin=fx.IOU_MARGIN,
    )
    small_boxes = [(0, 0, 20, 20), (100, 0, 20, 20)]
    assert (
        resolve_match_criterion(trait, gt_only(small_boxes))["kind"]
        == "iou_match"
    )


def test_resolve_match_criterion_scales_a_center_match_to_the_reference() -> (
    None
):
    """A center match's tolerance scales with the reference's average object size.

    A center match carries its tolerance as a fraction of the average object size and the
    tolerance that fraction is on this reference.
    """
    trait = fx.entry("leaf", ("leaf_length",), localization="center_match")
    result = resolve_match_criterion(
        trait, gt_only([(0, 0, 20, 20), (500, 0, 20, 20)])
    )
    assert result["kind"] == "center_match"
    assert result["tolerance"] == pytest.approx(
        result["tolerance_frac"] * 20.0
    )


def test_the_comparability_criterion_is_iou_matching_under_no_trait() -> None:
    result = comparability_criterion(COMPARABILITY_IOU, by_mask=False)
    assert result["kind"] == "iou_match"
    assert result["iou_threshold"] == COMPARABILITY_IOU
    assert result["trait"] is None


def test_resolve_match_criterion_iou_match_derives_a_real_threshold_not_pinned_0_5() -> (
    None
):
    """An IoU match's threshold is derived from the ground truth in hand, not pinned to 0.5.

    The derivation is ``derive_iou_match_threshold``, and the record names it.
    """
    from tcip_mcp.pipelines.derivations import IOU_MATCH_DERIVATION

    trait = fx.entry(
        "leaf",
        ("leaf_length",),
        localization="iou_match",
        iou_jitter_px=fx.IOU_JITTER_PX,
        iou_margin=fx.IOU_MARGIN,
    )
    # char size 300 -> derived threshold well above 0.5 (see test_derive_iou_match_threshold_*).
    large_boxes = [(0, 0, 300, 300), (500, 0, 300, 300)]
    result = resolve_match_criterion(trait, gt_only(large_boxes))
    assert result["kind"] == "iou_match"
    assert result["iou_threshold"] > 0.5
    assert result["derived_from"] == IOU_MATCH_DERIVATION


def test_resolve_match_criterion_iou_match_refuses_an_unauthored_jitter_or_margin() -> (
    None
):
    """An IoU match refuses an unauthored jitter or margin by name.

    The jitter and margin the IoU derivation models are the trait's to author; nothing stands
    in for an unauthored one.
    """
    from tcip_mcp.traits import UnauthoredFieldError

    trait = fx.entry("leaf", ("leaf_length",), localization="iou_match")
    with pytest.raises(UnauthoredFieldError, match="iou_jitter_px"):
        resolve_match_criterion(trait, gt_only([(0, 0, 300, 300)]))
    half = fx.with_fields(trait, iou_jitter_px=fx.IOU_JITTER_PX)
    with pytest.raises(UnauthoredFieldError, match="iou_margin"):
        resolve_match_criterion(half, gt_only([(0, 0, 300, 300)]))


def test_resolve_match_criterion_iou_match_refuses_a_reference_with_no_box_to_derive_from() -> (
    None
):
    """An IoU match over a reference holding no box refuses naming it."""
    trait = fx.entry(
        "leaf",
        ("leaf_length",),
        localization="iou_match",
        iou_jitter_px=fx.IOU_JITTER_PX,
        iou_margin=fx.IOU_MARGIN,
    )
    with pytest.raises(ValueError, match="no ground-truth box"):
        resolve_match_criterion(trait, [])


# --------------------------------------------------------------------------
# Box and mask mAP exact values on fixed synthetic records.
# --------------------------------------------------------------------------


def test_golden_box_and_mask_map() -> None:
    gt = _gt([10, 10, 20, 20])
    dt_perfect = _dt(([10, 10, 20, 20], 0.9))
    m = detection_metrics(
        [_rec(gt, dt_perfect)],
        trait=None,
        conf_threshold=0.25,
        iou_threshold=0.5,
        by_mask=False,
    )
    assert (m["map"], m["map50"]) == pytest.approx((1.0, 1.0))
    assert (m["precision"], m["recall"], m["f1"]) == pytest.approx(
        (1.0, 1.0, 1.0)
    )
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 0)

    # 1 TP + 1 FP against 2 GT → P=R=F1=0.5, counts 1/1/1.
    gt2 = _gt([10, 10, 20, 20], [50, 50, 10, 10])
    dt2 = _dt(([10, 10, 20, 20], 0.9), ([80, 80, 10, 10], 0.7))
    m2 = detection_metrics(
        [_rec(gt2, dt2)],
        trait=None,
        conf_threshold=0.25,
        iou_threshold=0.5,
        by_mask=False,
    )
    assert (m2["precision"], m2["recall"], m2["f1"]) == pytest.approx(
        (0.5, 0.5, 0.5)
    )
    assert (m2["tp"], m2["fp"], m2["fn"]) == (1, 1, 1)

    # segm: a perfect mask match scores map50 = 1.0.
    ring = [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)]
    ms = detection_metrics(
        [_masked(ring, ring)],
        trait=None,
        conf_threshold=0.25,
        iou_threshold=0.5,
        by_mask=True,
    )
    assert ms["governing_criterion"]["kind"] == "mask_iou_match"
    assert ms["map50"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# In-house scalar metrics
# --------------------------------------------------------------------------


def test_classification_metrics() -> None:
    pred = torch.tensor([0, 1, 1, 0])
    gt = torch.tensor([0, 1, 0, 0])
    m = classification_metrics(pred, gt, num_classes=2)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx((0.8 + 2 / 3) / 2, abs=1e-3)


def test_ordinal_metrics() -> None:
    m = ordinal_metrics(torch.tensor([0, 1, 2]), torch.tensor([0, 2, 2]), 3)
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["rank_acc"] == pytest.approx(2 / 3)
    assert m["quadratic_weighted_kappa"] == pytest.approx(0.8)


def test_regression_metrics() -> None:
    m = regression_metrics(
        torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 4.0])
    )
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["rmse"] == pytest.approx(math.sqrt(1 / 3))
    assert m["r_squared"] == pytest.approx(11 / 14)


def test_quadratic_weighted_kappa_perfect_agreement_is_one() -> None:
    kappa = quadratic_weighted_kappa(
        torch.tensor([0, 1, 2, 1]), torch.tensor([0, 1, 2, 1]), 3
    )
    assert kappa == pytest.approx(1.0)


def test_quadratic_weighted_kappa_hand_computed() -> None:
    # true=[0,2,2], pred=[0,1,2]: one item off by one rank out of three, worked out by hand
    # against the expected-disagreement-under-independence formula -> kappa = 1 - 1/5.
    kappa = quadratic_weighted_kappa(
        torch.tensor([0, 1, 2]), torch.tensor([0, 2, 2]), 3
    )
    assert kappa == pytest.approx(0.8)


def test_quadratic_weighted_kappa_reads_a_fractional_prediction_at_its_nearest_rank() -> (
    None
):
    """Quadratic weighted kappa reads a fractional prediction at its nearest rank.

    A head that emits continuous rank estimates is scored at the rank each estimate is nearest
    to, so predictions that all round onto the true ranks are perfect agreement.
    """
    fractional = torch.tensor([0.4, 1.6, 2.4, 2.6])
    gt = torch.tensor([0, 2, 2, 3])
    assert quadratic_weighted_kappa(fractional, gt, 4) == pytest.approx(1.0)


def test_quadratic_weighted_kappa_scores_a_single_rank_guess_at_chance() -> (
    None
):
    """Quadratic weighted kappa scores a single-rank guess at chance.

    A predictor that always guesses the majority rank is right on 7 of 10 items and scores 0.
    """
    gt = torch.tensor([0, 0, 0, 0, 0, 0, 0, 3, 3, 3])
    always_majority = torch.zeros(10)
    assert int((always_majority == gt).sum()) == 7
    assert quadratic_weighted_kappa(always_majority, gt, 4) == pytest.approx(
        0.0, abs=1e-9
    )


def test_quadratic_weighted_kappa_empty_is_none() -> None:
    assert (
        quadratic_weighted_kappa(torch.tensor([]), torch.tensor([]), 3) is None
    )


def test_quadratic_weighted_kappa_degenerate_single_rank_is_none() -> None:
    # every item shares the same true and predicted rank: no expected disagreement to correct for.
    kappa = quadratic_weighted_kappa(
        torch.tensor([1, 1, 1]), torch.tensor([1, 1, 1]), num_ranks=3
    )
    assert kappa is None


def test_r_squared_perfect_fit_is_one() -> None:
    assert r_squared(
        torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])
    ) == pytest.approx(1.0)


def test_r_squared_worse_than_mean_baseline_is_negative() -> None:
    r2 = r_squared(
        torch.tensor([5.0, -5.0, 5.0]), torch.tensor([1.0, 2.0, 3.0])
    )
    assert r2 < 0


def test_r_squared_empty_is_none() -> None:
    assert r_squared(torch.tensor([]), torch.tensor([])) is None


def test_r_squared_constant_gt_is_none() -> None:
    assert (
        r_squared(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([5.0, 5.0, 5.0]))
        is None
    )


def test_concordance_correlation_coefficient_perfect_agreement_is_one() -> (
    None
):
    ccc = concordance_correlation_coefficient(
        torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])
    )
    assert ccc == pytest.approx(1.0)


def test_concordance_correlation_coefficient_hand_computed() -> None:
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


def test_concordance_correlation_coefficient_empty_is_none() -> None:
    assert (
        concordance_correlation_coefficient(torch.tensor([]), torch.tensor([]))
        is None
    )


def test_concordance_correlation_coefficient_zero_variance_is_none() -> None:
    # constant predictions against varying GT: pred_var=0, denominator undefined.
    assert (
        concordance_correlation_coefficient(
            torch.tensor([2.0, 2.0, 2.0]), torch.tensor([1.0, 2.0, 3.0])
        )
        is None
    )
    # constant GT against varying predictions: gt_var=0, same undefined denominator.
    assert (
        concordance_correlation_coefficient(
            torch.tensor([1.0, 2.0, 3.0]), torch.tensor([5.0, 5.0, 5.0])
        )
        is None
    )


def test_selection_value_prefers_objective_for_detection() -> None:
    assert (
        _selection_value(
            "detection",
            {"val_loss": 0.1, "val_objective": 5.0},
            0.2,
            "objective",
        )
        == 5.0
    )
    assert (
        _selection_value("classification", {"val_loss": 0.1}, 0.2, "loss")
        == 0.1
    )
    # No validation loader ran (val_metrics empty): selecting on loss falls back to the
    # training loss.
    assert _selection_value("detection", {}, 0.2, "loss") == 0.2


def test_selection_value_raises_when_the_metric_is_not_among_this_epochs_val_metrics() -> (
    None
):
    with pytest.raises(
        ValueError, match="not among this epoch's validation metrics"
    ):
        _selection_value("detection", {"val_loss": 0.1}, 0.2, "objective")


def test_resolve_selection_metric_admits_a_weighted_objective_and_a_plain_metric() -> (
    None
):
    resolve_selection_metric(
        None, "objective", score_weights=WEIGHTS, has_val_loader=True
    )
    resolve_selection_metric(
        None, "f1", score_weights=None, has_val_loader=True
    )


def test_resolve_selection_metric_with_no_val_loader_accepts_only_loss() -> (
    None
):
    resolve_selection_metric(
        None, "loss", score_weights=None, has_val_loader=False
    )
    for requested, weights in (("objective", WEIGHTS), ("accuracy", None)):
        with pytest.raises(ValueError, match="needs a validation loader"):
            resolve_selection_metric(
                None,
                requested,
                score_weights=weights,
                has_val_loader=False,
            )


@_with_bud_trait
def test_resolve_selection_metric_rejects_incoherent_explicit_choice(
    tmp_path: Path,
) -> None:
    trait = fx.latest("bud_opening", tmp_path)
    with pytest.raises(ValueError, match="comparability-only"):
        resolve_selection_metric(
            trait, "map50", score_weights=None, has_val_loader=True
        )


@_with_bud_trait
def test_resolve_selection_metric_allows_coherent_explicit_choice(
    tmp_path: Path,
) -> None:
    # A legitimate explicit choice must still succeed: a rail must admit valid work, not
    # only reject invalid work.
    trait = fx.latest("bud_opening", tmp_path)
    for metric in ("f1", "recall"):
        resolve_selection_metric(
            trait, metric, score_weights=None, has_val_loader=True
        )
    resolve_selection_metric(
        None, "map50", score_weights=None, has_val_loader=True
    )  # no trait -> no gate


def test_an_evaluation_block_selecting_on_a_metric_with_no_declared_direction_is_refused() -> (
    None
):
    """A selection metric with no declared ranking direction is refused by the field's type.

    ``selection_metric="not_a_real_metric"`` is refused naming the field, and ``"f1"`` is
    admitted.
    """
    from pydantic import ValidationError

    from tcip_mcp.pipelines.schemas import EvaluationSpec

    with pytest.raises(ValidationError, match="selection_metric"):
        EvaluationSpec(
            **evaluation_block(selection_metric="not_a_real_metric")
        )
    EvaluationSpec(**evaluation_block(selection_metric="f1"))


def test_a_detector_run_selecting_on_the_objective_refuses_unweighted_and_resolves_weighted(
    tmp_path: Path,
) -> None:
    """A detector selecting on the objective refuses unweighted and resolves weighted.

    A detector config selecting on ``objective`` with no ``score_weights`` refuses at its
    resolution naming the key; the same config stating them resolves, its objective weighted
    term by term by its recorded weights.
    """
    from tcip_mcp.pipelines.schemas import train_config
    from tcip_mcp.pipelines.training.generic_trainer import (
        resolve_trait_revision,
    )
    from tests._chain_fixtures import training_config, weights_statistics
    from tests._verified_checkpoint_fixtures import unbuilt_source

    source = weights_statistics(unbuilt_source("detection"))
    spec = train_config(
        training_config(
            tmp_path,
            source,
            {},
            evaluation=evaluation_block(score_weights=None),
        )
    )
    with pytest.raises(ValueError, match="evaluation.score_weights"):
        resolve_trait_revision(spec, project=tmp_path, has_val_loader=True)

    weights = {"loss": 1.0, "f1": 0.0, "map50": 0.0}
    spec = train_config(
        training_config(
            tmp_path,
            source,
            {},
            evaluation=evaluation_block(score_weights=weights),
        )
    )
    assert (
        resolve_trait_revision(spec, project=tmp_path, has_val_loader=True)
        is None
    )
    assert (
        compute_composite_objective(
            2.0, 0.5, 0.4, spec.evaluation.score_weights
        )
        == 2.0
    )


def test_score_weights_missing_a_term_or_stating_another_refuse_at_admission() -> (
    None
):
    """Score weights missing a term or stating another refuse at admission by name.

    A config whose ``score_weights`` omits a term refuses at admission naming each missing
    one, and one stating a fourth key refuses naming it; the three terms admit as the block.
    """
    from tcip_mcp.pipelines.schemas import train_config
    from tests import REPO_ROOT
    from tests._chain_fixtures import training_config, weights_statistics
    from tests._verified_checkpoint_fixtures import unbuilt_source

    def admitted(weights: dict) -> TrainConfigSchema:
        return train_config(
            training_config(
                REPO_ROOT,
                weights_statistics(unbuilt_source("detection")),
                {},
                evaluation=evaluation_block(score_weights=weights),
            )
        )

    with pytest.raises(
        ValueError, match=r"(?s)score_weights\.f1.*score_weights\.map50"
    ):
        admitted({"loss": 1.0})
    with pytest.raises(ValueError, match=r"score_weights\.recall"):
        admitted({**SCORE_WEIGHTS, "recall": 0.1})
    weights = admitted(dict(SCORE_WEIGHTS)).evaluation.score_weights
    assert weights.model_dump() == SCORE_WEIGHTS


# HIGHER_IS_BETTER_BY_METRIC held against what evaluate()/governing_counts really return.


def _detection_batch(
    num_images: int = 2, img_size: int = 64
) -> tuple[list[Tensor], list[dict]]:
    from tcip_mcp.pipelines.training.collation import task_collate

    items = []
    boxes = [[10.0, 10.0, 40.0, 40.0], [5.0, 5.0, 25.0, 30.0]]
    for i in range(num_images):
        img = torch.rand(3, img_size, img_size)
        target = {
            "boxes": torch.tensor([boxes[i % len(boxes)]]),
            "labels": torch.ones((1,), dtype=torch.long),
            "iscrowd": torch.zeros((1,), dtype=torch.long),
            "image_id": i,
        }
        items.append((img, target))
    return task_collate("detection")(items)


@_with_bud_trait
def test_higher_is_better_by_metric_matches_evaluate_and_governing_counts(
    tmp_path: Path,
) -> None:
    """``HIGHER_IS_BETTER_BY_METRIC`` matches the ranking keys the two metric producers return.

    Over each synthesizable task's ``evaluate`` and one ``governing_counts`` call, each declared
    key is returned, and each returned key outside ``not_a_ranking`` and the not-finite suffix
    is declared.
    """
    pytest.importorskip("torchvision")
    from tcip_mcp.pipelines.model_contract import _SYNTHESIZABLE_TASKS
    from tcip_mcp.pipelines.schemas import HIGHER_IS_BETTER_BY_METRIC
    from tcip_mcp.pipelines.training.evaluation import (
        evaluate,
        governing_counts,
    )
    from tcip_store.values import NOT_FINITE_SUFFIX
    from tests import bespoke_models
    from tests._chain_fixtures import band_statistics, detector_values
    from tests._producer_fixtures import painted_array

    device = torch.device("cpu")
    img_size = 64
    returned: set[str] = set()

    for task in sorted(_SYNTHESIZABLE_TASKS):
        if task == "detection":
            model = bespoke_models.build_bespoke_detection(
                num_classes=1, **detector_values()
            )
            images, targets = _detection_batch(img_size=img_size)
            loader: (
                list[tuple[list[Tensor], list[dict]]]
                | list[tuple[Tensor, dict[str, Tensor]]]
            ) = [(images, targets)]
        elif task == "instance_seg":
            model = bespoke_models.build_bespoke_instance_seg(
                num_classes=1, **detector_values()
            )
            images, targets = _detection_batch(img_size=img_size)
            for t in targets:
                t["masks"] = torch.as_tensor(
                    painted_array(img_size, img_size, [((10, 10, 40, 40), 1)])
                )[None]
            loader = [(images, targets)]
        elif task == "classification":
            model = bespoke_models.build_bespoke_classifier(
                num_classes=2, **band_statistics()
            )
            imgs = torch.stack(
                [torch.rand(3, img_size, img_size) for _ in range(2)]
            )
            loader = [(imgs, {"labels": torch.tensor([0, 1])})]
        elif task == "ordinal":
            model = bespoke_models.build_bespoke_ordinal(
                num_ranks=3, **band_statistics()
            )
            imgs = torch.stack(
                [torch.rand(3, img_size, img_size) for _ in range(2)]
            )
            loader = [(imgs, {"ranks": torch.tensor([0, 2])})]
        elif task == "regression":
            model = bespoke_models.build_bespoke_regressor(**band_statistics())
            imgs = torch.stack(
                [torch.rand(3, img_size, img_size) for _ in range(2)]
            )
            loader = [(imgs, {"values": torch.tensor([0.2, 0.8])})]
        else:
            model = bespoke_models.build_bespoke_semantic_seg(
                num_classes=2, **band_statistics()
            )
            imgs = torch.stack(
                [torch.rand(3, img_size, img_size) for _ in range(2)]
            )
            m0 = torch.as_tensor(
                painted_array(
                    img_size, img_size, [((0, 0, img_size // 2, img_size), 1)]
                )
            ).long()
            m1 = 1 - m0
            loader = [(imgs, {"masks": torch.stack([m0, m1])})]

        # "bud_opening" (seeded center_match) exercises evaluate()'s center-match branch for
        # detection.
        trait = (
            fx.latest("bud_opening", tmp_path) if task == "detection" else None
        )
        from tcip_mcp.pipelines.schemas import ModelDims

        dims = {
            "ordinal": ModelDims(in_chans=3, num_ranks=3),
            "regression": ModelDims(in_chans=3),
        }.get(task, ModelDims(in_chans=3, num_classes=2))
        from tcip_mcp.pipelines.model_contract import DETECTOR_TASKS

        detector = task in DETECTOR_TASKS
        result = evaluate(
            model,
            loader,
            device,
            task,
            dims=dims,
            trait=trait,
            conf_threshold=VALIDATION_CONF if detector else None,
            iou_threshold=IOU_THRESHOLD,
            score_weights=WEIGHTS if detector else None,
            density=1.0 if detector else None,
            mask_cut=MASK_CUT if task == "instance_seg" else None,
        )
        returned.update(result)

    per_image = [
        _rec(
            _gt([10.0, 10.0, 30.0, 30.0]),
            _dt(([11.0, 11.0, 29.0, 29.0], 0.9)),
            w=64,
            h=64,
        ),
        _rec([], [], w=64, h=64),
    ]
    returned.update(
        governing_counts(
            per_image,
            {"kind": "center_match", "tolerance": 5.0},
            conf_threshold=0.25,
        )
    )

    not_a_ranking = {
        "per_class",
        "count_bias",
        "per_class_iou",
        "per_class_dice",
        "tp",
        "fp",
        "fn",
        "criterion",
        "governing_criterion",
        "map50_role",
    }
    ranking_returned = {
        k for k in returned if not k.endswith(NOT_FINITE_SUFFIX)
    }
    declared = set(HIGHER_IS_BETTER_BY_METRIC)

    undeclared = (ranking_returned - not_a_ranking) - declared
    assert not undeclared, (
        f"returned but no declared direction: {sorted(undeclared)}"
    )
    never_produced = declared - ranking_returned
    assert not never_produced, (
        f"declared but neither producer ever returns it: {sorted(never_produced)}"
    )
    assert ranking_returned - declared == not_a_ranking


def test_both_eval_regimes_share_common_keys_and_keep_their_own_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both evaluation regimes carry the common keys and keep their own fields apart.

    The results of ``run_test_evaluation`` and ``run_full_frame_evaluation`` each carry the
    common identity keys, and ``run_test_evaluation``'s result carries none of the full-frame
    regime's fields.
    """
    from PIL import Image

    import tcip_mcp.pipelines.training.evaluation as evaluation
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.training.eval_runners import (
        run_full_frame_evaluation,
    )
    from tests._producer_fixtures import label_image
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_DETECTOR_PASS,
        registered_checkpoint,
    )

    common_fields = {
        "model_path",
        "task",
        "checkpoint_sha256",
        "experiment_id",
        "iou_threshold",
        "execution",
        "eval_regime",
    }
    full_frame_only_fields = {
        "scored_images",
        "tallies",
        "cap_saturated_frac",
    }

    from tcip_mcp.pipelines.execution import prepare

    ckpt_path = registered_checkpoint(tmp_path)
    monkeypatch.setattr(
        evaluation,
        "evaluate",
        lambda *a, **k: {
            "loss": 0.1,
            "precision": 0.4,
            "recall": 0.5,
            "f1": 0.44,
        },
    )
    checkpoint = load_registered_checkpoint(ckpt_path, project=tmp_path)
    test_result = run_test_evaluation(
        prepare(
            checkpoint, Stated(tile=False, **SAMPLE_DETECTOR_PASS)
        ).runnable(),
        None,
        iou_threshold=IOU_THRESHOLD,
        trait=None,
    )

    images_dir = tmp_path / "ff" / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(images_dir / "a.png")
    label_image(
        images_dir / "a.png",
        [Annotation(subject="bud", geometry=BBox(4, 4, 12, 12))],
        32,
        32,
    )

    from tests._predictor_fixtures import StubPredictor, install

    install(
        monkeypatch,
        StubPredictor(
            task="detection",
            in_chans=3,
            width=32,
            height=32,
            boxes=(),
            scores=(),
        ),
    )
    ff_result = run_full_frame_evaluation(
        checkpoint,
        checkpoint_admission(checkpoint, images_dir),
        stated=Stated(tile_size=32, overlap=0.0, **SAMPLE_DETECTOR_PASS),
        iou_threshold=IOU_THRESHOLD,
        trait=None,
    )

    for field in common_fields:
        assert field in test_result, (
            f"{field} missing from the test-regime record"
        )
        assert field in ff_result, (
            f"{field} missing from the full-frame-regime record"
        )
    assert not (full_frame_only_fields & set(test_result))


@pytest.mark.parametrize(
    "boxes, derives",
    [
        ([(4, 4, 16, 16), (10, 4, 22, 16)], True),
        ([(4, 4, 12, 12)], False),
    ],
)
def test_a_full_frame_evaluation_merges_at_the_threshold_its_ground_truth_derives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boxes: list[tuple[int, int, int, int]],
    derives: bool,
) -> None:
    """A full-frame evaluation merges at the threshold its ground truth derives.

    An unstated merge threshold is the one the evaluated reference's ground truth derives;
    where that ground truth derives none, the evaluation refuses naming ``cross_tile_nms``.
    """
    from PIL import Image

    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.derivations import (
        CROSS_TILE_NMS_DERIVATION,
        derive_cross_tile_nms,
    )
    from tcip_mcp.pipelines.execution import ExecutionRefusedError
    from tcip_mcp.pipelines.training.eval_runners import (
        run_full_frame_evaluation,
    )
    from tests._predictor_fixtures import StubPredictor, install
    from tests._producer_fixtures import label_image
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_CONF,
        registered_checkpoint,
    )

    checkpoint = load_registered_checkpoint(
        registered_checkpoint(tmp_path), project=tmp_path
    )
    images_dir = tmp_path / "ff" / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(images_dir / "a.png")
    label_image(
        images_dir / "a.png",
        [Annotation(subject="bud", geometry=BBox(*b)) for b in boxes],
        32,
        32,
    )
    install(
        monkeypatch,
        StubPredictor(
            task="detection",
            in_chans=3,
            width=32,
            height=32,
            boxes=(),
            scores=(),
        ),
    )
    admission = checkpoint_admission(checkpoint, images_dir)
    stated = Stated(
        tile_size=32,
        overlap=0.0,
        conf=SAMPLE_CONF,
        postprocess="nms",
        rationale=RATIONALE,
    )
    scored = {"iou_threshold": IOU_THRESHOLD, "trait": None}

    if not derives:
        with pytest.raises(ExecutionRefusedError, match="cross_tile_nms"):
            run_full_frame_evaluation(
                checkpoint, admission, stated=stated, **scored
            )
        return
    from tests._verified_checkpoint_fixtures import objects_over

    execution = run_full_frame_evaluation(
        checkpoint, admission, stated=stated, **scored
    )["execution"]
    assert execution["sources"]["cross_tile_nms"] == CROSS_TILE_NMS_DERIVATION
    assert execution["cross_tile_nms"] == pytest.approx(
        derive_cross_tile_nms(
            [objects_over([list(b) for b in boxes], 32 * 32)]
        )
    )


def test_an_evaluation_metric_sharing_an_identity_key_refuses_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regime's field sharing a key with the evaluation's identity refuses naming the key."""
    import tcip_mcp.pipelines.training.evaluation as evaluation
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import prepare
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_DETECTOR_PASS,
        registered_checkpoint,
    )

    monkeypatch.setattr(
        evaluation,
        "evaluate",
        lambda *a, **k: {"precision": 0.9, "task": "classification"},
    )
    checkpoint = load_registered_checkpoint(
        registered_checkpoint(tmp_path), project=tmp_path
    )
    with pytest.raises(ValueError, match="task"):
        run_test_evaluation(
            prepare(
                checkpoint, Stated(tile=False, **SAMPLE_DETECTOR_PASS)
            ).runnable(),
            None,
            iou_threshold=IOU_THRESHOLD,
            trait=None,
        )


def test_a_tile_level_evaluation_scores_on_the_device_its_pass_model_sits_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The evaluation is handed the pass's own predictor device, while torch sees CUDA."""
    import tcip_mcp.pipelines.training.evaluation as evaluation
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import prepare
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_DETECTOR_PASS,
        registered_checkpoint,
    )

    checkpoint = load_registered_checkpoint(
        registered_checkpoint(tmp_path), project=tmp_path
    )
    pass_ = prepare(
        checkpoint, Stated(tile=False, **SAMPLE_DETECTOR_PASS), device="cpu"
    ).runnable()
    handed: list[torch_device] = []

    def _observed(
        model: nn.Module,
        loader: DataLoader,
        device: torch_device,
        task: str,
        **keywords: Unpack[EvaluationKeywords],
    ) -> dict:
        handed.append(device)
        return {"precision": 0.9}

    monkeypatch.setattr(evaluation, "evaluate", _observed)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    run_test_evaluation(pass_, None, iou_threshold=IOU_THRESHOLD, trait=None)

    assert len(handed) == 1
    assert handed[0] is pass_.predictor.device


def test_an_evaluation_identity_stated_without_its_regime_refuses_naming_it(
    tmp_path: Path,
) -> None:
    """An evaluation identity stated without ``eval_regime`` refuses at construction naming it."""
    from pydantic import ValidationError

    import tcip_mcp.pipelines.training.eval_runners as eval_runners
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import prepare
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_DETECTOR_PASS,
        registered_checkpoint,
    )

    checkpoint = load_registered_checkpoint(
        registered_checkpoint(tmp_path), project=tmp_path
    )
    stated = {
        **dict(checkpoint.producer),
        "model_path": checkpoint.path,
        "task": checkpoint.task,
        "iou_threshold": IOU_THRESHOLD,
        "execution": prepare(
            checkpoint, Stated(tile=False, **SAMPLE_DETECTOR_PASS)
        )
        .runnable()
        .execution,
    }
    with pytest.raises(ValidationError, match="eval_regime"):
        eval_runners.EvaluationIdentity.model_validate(stated)


# --------------------------------------------------------------------------
# Light integration: _validate via train()
# --------------------------------------------------------------------------

torchvision = pytest.importorskip("torchvision")

from tcip_mcp.pipelines.training.generic_trainer import train  # noqa: E402
from tests._training_values import adamw_optimizer  # noqa: E402

IMG = 64


_save_png = partial(write_noise_image, size=IMG)


def _four_bud_images(tmp_path: Path) -> Path:
    """Write four noise images, each labeled with one ``bud`` box at (19.2, 19.2, 44.8, 44.8).

    The images land in ``tmp_path``'s undated capture; return their directory.
    """
    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import label_image

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    for i in range(4):
        _save_png(images_dir / f"img{i}.png")
        label_image(
            images_dir / f"img{i}.png",
            [Annotation(subject="bud", geometry=BBox(19.2, 19.2, 44.8, 44.8))],
            IMG,
            IMG,
        )
    return images_dir


def _stored(
    tmp_path: Path, annotations: list[Annotation], size: int = 100
) -> LabelDocument:
    """Write ``annotations`` as one image's label document and return it read back.

    The document is that of a ``size``-pixel image ``a`` under ``tmp_path``.
    """
    from tcip_annotation import json_io
    from tcip_mcp.dataset_layout import label_key

    key = label_key(tmp_path, UNDATED_BUCKET, "a")
    json_io.write_label_document(key, annotations, size, size, keep_empty=True)
    return json_io.read_label_document(key)


def _context(
    tmp_path: Path, model_source: dict, data: dict, **overrides: object
) -> TrainContext:
    """Return the context of a run of ``model_source`` over the data block ``data``.

    The run draws its split at a validation ratio of 0.25, with ``overrides`` in the config, and
    is opened by the launcher's own producer.
    """
    from tests._chain_fixtures import training_config
    from tests.tiny_trainer_fixtures import run_context

    return run_context(
        tmp_path,
        training_config(
            tmp_path,
            model_source,
            {**data, "split": drawn_split(val_ratio=0.25)},
            optimizer=adamw_optimizer(),
            **overrides,
        ),
    )


def test_validate_detection_returns_metrics_and_objective(
    tmp_path: Path,
) -> None:
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR

    ctx = _context(
        tmp_path,
        BUILT_DETECTOR,
        {
            "images_dir": str(_four_bud_images(tmp_path)),
            "scope": {"subject": "bud"},
        },
    )
    out_dir = ctx.run_dir
    # No AttributeError on model.heads.
    run = train(ctx.run, ctx.train_loader, val_loader=ctx.val_loader)

    last = run.metrics_history[-1]
    for k in (
        "val_loss",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_map50",
        "val_map",
        "val_objective",
    ):
        assert k in last, f"missing {k}"
    # An untrained BespokeDetection finds nothing on four images: the epoch has no useful score,
    # so no epoch is selectable and the run fails naming the stage rather than delivering weights.
    assert last["val_objective"] is None
    assert math.isnan(last["selection"])
    assert run.status == "failed"
    assert "stage 0 produced no selectable epoch" in run.status_error
    assert not (out_dir / "model_best.pt").is_file()
    assert not (out_dir / "model_final.pt").is_file()
    assert run.best_metric == math.inf


@_with_bud_trait
def test_train_center_match_trait_records_governing_criterion(
    tmp_path: Path,
) -> None:
    """A center-match trait's run records its governing criterion in the metrics row.

    The untrained detector's run fails as having no selectable epoch, and its last metrics row
    carries ``val_governing_criterion``, ``val_map50_role`` and the trait it selected on.
    """
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR

    ctx = _context(
        tmp_path,
        BUILT_DETECTOR,
        {
            "images_dir": str(_four_bud_images(tmp_path)),
            "scope": {"subject": "bud"},
        },
        evaluation=evaluation_block(trait="bud_opening"),
    )
    run = ctx.default_train()
    ctx.tb.close()

    assert "no selectable epoch" in run.status_error
    last = run.metrics_history[-1]
    assert "val_governing_criterion" in last
    assert last["val_map50_role"] == "comparability_only"
    assert last["selection_metric"] == "objective"
    assert last["selection_trait"] == "bud_opening"


def test_validate_classification_metrics(tmp_path: Path) -> None:
    from tests._chain_fixtures import BESPOKE_MODELS
    from tests.tiny_trainer_fixtures import (
        distinct_intensities,
        write_regression_dataset,
    )

    images_dir, csv_path = write_regression_dataset(
        tmp_path,
        distinct_intensities(6),
        [i % 2 for i in range(6)],
        height=IMG,
        width=IMG,
    )
    model_source = {
        "builder": BESPOKE_CLASSIFIER,
        "source_files": [BESPOKE_MODELS],
        "task": "classification",
        "normalization": FRAME_STATISTICS,
    }
    ctx = _context(
        tmp_path,
        model_source,
        {"images_dir": str(images_dir), "labels_dir": str(csv_path)},
        batch_size=3,
        evaluation=evaluation_block(selection_metric="loss"),
    )
    run = train(ctx.run, ctx.train_loader, val_loader=ctx.val_loader)

    assert run.status == "completed", run.status_error
    last = run.metrics_history[-1]
    assert "val_accuracy" in last and "val_f1" in last
    assert run.best_metric == pytest.approx(last["val_loss"])


def test_score_predictions_folder_counts_every_image_through_the_matcher(
    data_dir: Path,
) -> None:
    from tcip_mcp.tools.annotation_tools import score_predictions
    from tests.conftest import DATA_DIR_BUCKET

    r = score_predictions(
        str(data_dir / "images" / "2-11-26"),
        DATA_DIR_BUCKET,
        iou_threshold=0.5,
        conf_threshold=0.25,
    )
    assert "map50" in r
    # fixture: each image has 2 GT, predictions = 1 TP + 1 FP -> tp=1,fp=1,fn=1 per image
    # (x3 images).
    assert r["total_tp"] == 3 and r["total_fp"] == 3 and r["total_fn"] == 3
    assert r["precision"] == pytest.approx(0.5)
    assert all(
        p["tp"] == 1 and p["fp"] == 1 and p["fn"] == 1 for p in r["per_image"]
    )


# -- a crowd region: an ignore region, never one object ------------------------------------

_OBJECT = [10.0, 10.0, 20.0, 20.0]
_CROWD = [50.0, 50.0, 40.0, 40.0]


def _crowd_records() -> list[dict]:
    """Return the records of two images holding a crowd region.

    One image's object a detection matches and its crowd region holds another detection; the
    other image holds a crowd region alone, which no detection matches.
    """
    return [
        _rec(
            [gt_record(_OBJECT, 1, 0), gt_record(_CROWD, 1, 1)],
            _dt((_OBJECT, 0.9), ([55.0, 55.0, 30.0, 30.0], 0.9)),
        ),
        _rec([gt_record(_CROWD, 1, 1)], []),
    ]


def test_a_detection_in_a_crowd_region_is_neither_true_nor_false_and_the_region_no_miss() -> (
    None
):
    m = detection_metrics(
        _crowd_records(),
        trait=None,
        iou_threshold=0.5,
        conf_threshold=0.25,
        by_mask=False,
    )
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 0)


def test_the_center_match_count_treats_a_crowd_region_as_the_iou_count_does() -> (
    None
):
    from tcip_mcp.pipelines.training.evaluation import governing_counts

    counts = governing_counts(
        _crowd_records(),
        {"kind": "center_match", "tolerance": 3.0},
        conf_threshold=0.25,
    )
    assert (counts["tp"], counts["fp"], counts["fn"]) == (1, 0, 0)


def test_a_targets_records_are_one_shape_on_the_stored_grid_whatever_the_target_holds(
    tmp_path: Path,
) -> None:
    """``gt_records`` builds one record shape on the stored grid from lists, arrays and tensors.

    The loader's own target, as lists, as an array and as tensors, becomes the same evaluation
    records, each box on the stored two-decimal grid, and a target with no crowd flag refuses.
    """
    import numpy as np

    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.data.datasets import target_tensors
    from tcip_mcp.pipelines.data.label_queries import json_det_targets
    from tcip_mcp.pipelines.training.evaluation import gt_records

    document = _stored(
        tmp_path,
        [
            Annotation(subject="bur", geometry=BBox(1.25, 2.5, 30.75, 40.5)),
            Annotation(
                subject="bur",
                geometry=BBox(50.0, 50.0, 90.0, 90.0),
                iscrowd=True,
            ),
        ],
    )
    listed = json_det_targets(
        document.annotations, registry_scope(tmp_path, "bur")
    )
    arrays = {k: np.asarray(v) for k, v in listed.items()}
    off_grid = {
        **listed,
        "boxes": [[1.2504, 2.5, 30.7496, 40.5], [50.0, 50.0, 90.0, 90.0]],
    }

    expected = [
        {"category_id": 1, "bbox": [1.25, 2.5, 29.5, 38.0], "iscrowd": 0},
        {"category_id": 1, "bbox": [50.0, 50.0, 40.0, 40.0], "iscrowd": 1},
    ]
    regions = [
        {"rings": [[(1.25, 2.5), (30.75, 2.5), (30.75, 40.5), (1.25, 40.5)]]},
        {"rings": [[(50.0, 50.0), (90.0, 50.0), (90.0, 90.0), (50.0, 90.0)]]},
    ]
    for target in (listed, arrays, target_tensors(listed), off_grid):
        # A target carrying its rows' geometry states each row's region beside its box.
        assert gt_records(target) == (
            [{**e, **r} for e, r in zip(expected, regions)]
            if "geometry" in target
            else expected
        )
    flagless = {"boxes": [[0, 0, 10, 10]], "labels": [1]}
    with pytest.raises(ValueError, match="iscrowd"):
        gt_records(flagless)


def test_a_detector_output_stating_no_boxes_refuses_rather_than_reading_as_no_detections() -> (
    None
):
    from tcip_mcp.pipelines.training.evaluation import records_from_detector

    target = {
        "boxes": torch.zeros((0, 4)),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "iscrowd": torch.zeros((0,), dtype=torch.int64),
    }
    with pytest.raises(KeyError, match="boxes"):
        records_from_detector(
            target,
            {},
            width=10,
            height=10,
            cap=1,
            mask_cut=None,
        )
    empty = {
        "boxes": torch.zeros((0, 4)),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "scores": torch.zeros((0,)),
    }
    assert (
        records_from_detector(
            target,
            empty,
            width=10,
            height=10,
            cap=1,
            mask_cut=None,
        )["dt"]
        == []
    )


def test_a_ground_truth_record_is_one_shape_from_a_target_and_from_its_annotation(
    tmp_path: Path,
) -> None:
    """A ground-truth record is one shape from a target and from its annotation.

    The loader's target route and the annotation route build a document's ground truth as
    the same records, its box, crowd flag and region stated alike.
    """
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.data.label_queries import json_det_targets
    from tcip_mcp.pipelines.training.evaluation import (
        gt_records,
        records_from_annotation,
    )

    document = _stored(
        tmp_path,
        [
            Annotation(subject="bur", geometry=BBox(10.1, 10.1, 40.3, 30.3)),
            Annotation(
                subject="bur",
                geometry=BBox(50.0, 50.0, 90.0, 90.0),
                iscrowd=True,
            ),
        ],
    )
    listed = json_det_targets(
        document.annotations, registry_scope(tmp_path, "bur")
    )
    record = records_from_annotation(
        document.annotations,
        [],
        width=100,
        height=100,
        cap=None,
        name_id={"bur": 1},
    )
    # The annotation route also names each record's annotation, by its index in the document.
    assert [
        {k: v for k, v in g.items() if k != "index"} for g in record["gt"]
    ] == gt_records(listed)
    assert [g["index"] for g in record["gt"]] == [0, 1]


def test_a_detectors_record_reads_both_sides_on_the_stored_grid(
    tmp_path: Path,
) -> None:
    """``records_from_detector`` reads both sides on the stored label grid.

    The ground-truth record holds the box the document stores, and the detection's record sits
    on the same 2-decimal grid.
    """
    torch = pytest.importorskip("torch")

    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.data.datasets import target_tensors
    from tcip_mcp.pipelines.data.label_queries import json_det_targets
    from tcip_mcp.pipelines.training.evaluation import records_from_detector

    document = _stored(
        tmp_path,
        [Annotation(subject="bur", geometry=BBox(10.3, 20.7, 40.1, 60.9))],
    )
    listed = json_det_targets(
        document.annotations, registry_scope(tmp_path, "bur")
    )
    output = {
        "boxes": torch.tensor([[10.1, 10.1, 40.3, 30.3]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.9]),
    }
    record = records_from_detector(
        target_tensors(listed),
        output,
        width=100,
        height=100,
        cap=2,
        mask_cut=None,
    )
    assert [g["bbox"] for g in record["gt"]] == [[10.3, 20.7, 29.8, 40.2]]
    assert [d["bbox"] for d in record["dt"]] == [[10.1, 10.1, 30.2, 20.2]]


def _reference_records(
    tmp_path: Path, reference: list, detections: list
) -> list[dict]:
    """Return one image's records, its reference read back through the label document."""
    from tcip_mcp.pipelines.training.evaluation import records_from_annotation

    return [
        records_from_annotation(
            _stored(tmp_path, reference).annotations,
            detections,
            width=100,
            height=100,
            cap=None,
        )
    ]


@pytest.mark.parametrize(
    "crowd_only", [True, False], ids=["crowd_only", "empty"]
)
def test_a_detection_outside_a_reference_with_no_object_is_a_false_positive(
    tmp_path: Path, crowd_only: bool
) -> None:
    """A detection outside a reference with no object is one false positive.

    Over a reference holding a crowd region alone or nothing, a detection outside the crowd
    region is one false positive by the IoU convention and by the center match alike.
    """
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.training.evaluation import governing_counts

    reference = (
        [
            Annotation(
                subject="bur",
                geometry=BBox(50, 50, 90, 90),
                iscrowd=True,
                created_by="user:breeder",
            )
        ]
        if crowd_only
        else []
    )
    detection = Annotation(
        subject="bur", geometry=BBox(5, 5, 20, 20), score=0.9
    )
    records = _reference_records(tmp_path, reference, [detection])

    m = detection_metrics(
        records,
        trait=None,
        iou_threshold=0.5,
        conf_threshold=0.25,
        by_mask=False,
    )
    assert (m["tp"], m["fp"], m["fn"]) == (0, 1, 0)
    counts = governing_counts(
        records,
        {"kind": "center_match", "tolerance": 3.0},
        conf_threshold=0.25,
    )
    assert (counts["tp"], counts["fp"], counts["fn"]) == (0, 1, 0)


def test_a_reference_with_objects_and_no_detection_scores_every_object_a_miss(
    tmp_path: Path,
) -> None:
    from tcip_annotation.state import Annotation, BBox

    records = _reference_records(
        tmp_path,
        [
            Annotation(
                subject="bur",
                geometry=BBox(5, 5, 20, 20),
                created_by="user:breeder",
            ),
            Annotation(
                subject="bur",
                geometry=BBox(40, 40, 60, 60),
                created_by="user:breeder",
            ),
        ],
        [],
    )

    m = detection_metrics(
        records,
        trait=None,
        iou_threshold=0.5,
        conf_threshold=0.25,
        by_mask=False,
    )
    assert (m["tp"], m["fp"], m["fn"]) == (0, 0, 2)


def test_a_crowd_region_is_no_object_in_a_ground_truth_count(
    tmp_path: Path,
) -> None:
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.data.splits import count_label_lines
    from tcip_mcp.pipelines.training.evaluation import gt_class_typical_count

    assert gt_class_typical_count(_crowd_records()) == 1.0
    document = _stored(
        tmp_path,
        [
            Annotation(subject="bur", geometry=BBox(1, 1, 9, 9)),
            Annotation(
                subject="bur", geometry=BBox(20, 20, 60, 60), iscrowd=True
            ),
        ],
    )
    assert count_label_lines(document, ClassScope(subject="bur")) == 1
    assert count_label_lines(document, ClassScope()) == 1
