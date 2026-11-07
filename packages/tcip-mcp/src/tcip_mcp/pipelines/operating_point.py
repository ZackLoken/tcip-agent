"""The measurement criteria and the detector operating-point knobs.

Each criterion reads its tolerances, floors and objective from the trait entry it is handed.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

from tcip_mcp.assessment_evidence import (
    REFERENCE_FLOOR,
    ClassifierEvidence,
    CountEvidence,
    FailureName,
    ScalarEvidence,
    bias_equivalence,
)
from tcip_store import finite_or_none

if TYPE_CHECKING:
    import torch

    from tcip_mcp.pipelines.schemas import SpatialManifest

from tcip_mcp.pipelines.training.evaluation import (
    EvaluationRecord,
    classes_with_evidence,
    concordance_correlation_coefficient,
    derive_operating_point_curve,
    gt_class_typical_count,
    gt_objects,
    mean_of_present_counts,
    pick_count_unbiased,
    pick_f1_max,
    r_squared,
    resolve_match_criterion,
    scaled_to,
)
from tcip_mcp.traits import CountObjective, RegressionCriterion, TraitEntry

COUNT_OBJECTIVE_PICKERS: dict[
    CountObjective, tuple[Callable[[dict], float], str]
] = {
    "count_unbiased": (pick_count_unbiased, "count-unbiased count curve"),
    "detection_f1": (pick_f1_max, "F1-max count curve"),
    "presence": (pick_f1_max, "F1-max count curve"),
}
"""The count objectives a calibration can fit, each as the picker it runs over the calibration
curve and the label its conf is recorded under."""

REGRESSION_CRITERIA: dict[
    RegressionCriterion, Callable[[Any, Any], float | None]
] = {
    "r_squared": r_squared,
    "concordance_correlation_coefficient": concordance_correlation_coefficient,
}
"""The regression skill statistics a caller may assess a continuous prediction by."""


SCORE_THRESH = "score_thresh"
"""The knob holding the least score a detection keeps in the model's own output."""
DETECTIONS_PER_IMG = "detections_per_img"
"""The knob holding the most detections the model keeps for one frame."""


def set_detector_operating_point(
    model: torch.nn.Module, *, score_thresh: float, detections_per_img: int
) -> None:
    """Set both operating-point knobs on ``model`` itself.

    A knob the model does not expose refuses (``ValueError``) naming it.
    """
    _set_knob(model, SCORE_THRESH, score_thresh)
    _set_knob(model, DETECTIONS_PER_IMG, detections_per_img)


def _set_knob(model: torch.nn.Module, attr: str, value: float) -> None:
    """Set the operating-point knob ``attr`` to ``value`` on ``model``.

    Refuses (``ValueError``) naming the knob when ``model`` does not expose it.
    """
    if not hasattr(model, attr):
        raise ValueError(
            f"this model exposes no {attr} on itself: a detection model's operating point is "
            f"set on the model before its forward, so the model exposes {attr} and its eval "
            "forward honors it."
        )
    setattr(model, attr, value)


def governed_forward(
    model: torch.nn.Module, inputs: list, caps: list[int], *, conf: float
) -> list:
    """Return ``model``'s output for each of ``inputs``, in input order, under its own cap.

    Each output is produced under the in-model score threshold ``conf`` and its own detection cap
    of ``caps``, both set in the model: one forward per
    distinct cap, over the inputs it governs.
    """
    outputs: list = [None] * len(inputs)
    for cap in sorted(set(caps)):
        at = [i for i, c in enumerate(caps) if c == cap]
        set_detector_operating_point(
            model, score_thresh=conf, detections_per_img=cap
        )
        for i, output in zip(at, model([inputs[i] for i in at]), strict=True):
            outputs[i] = output
    return outputs


def _min_dt_score(records: list[EvaluationRecord]) -> float | None:
    """Return the lowest detection score across a reference, or None if it holds none."""
    scores = [d["score"] for rec in records for d in rec.dt]
    return min(scores) if scores else None


def cap_saturated_frac(records: list[EvaluationRecord]) -> float:
    """Return the fraction of ``records`` whose frame kept as many detections as its cap."""
    return (
        sum(r.frame_count == r.cap for r in records) / len(records)
        if records
        else 0.0
    )


def count_criterion(
    cal_records: list[EvaluationRecord],
    hold_records: list[EvaluationRecord],
    entry: TraitEntry,
    *,
    staged_conf_floor: float,
    equivalence_z: float,
    dispersion_quantile: float,
    held: dict | None = None,
) -> tuple[float, CountEvidence, list[FailureName]]:
    """Pick the count objective's conf on the calibration side and check the holdout at it.

    Returns ``(conf, evidence, failures)``. The records are per-image evaluation records (``gt``
    objects and every ``dt`` detection with its score) collected at ``staged_conf_floor``,
    matched under the trait's localization resolved once over the calibration records
    (:func:`~tcip_mcp.pipelines.training.evaluation.resolve_match_criterion`, the evidence's
    ``localization``) and applied to each side
    (:func:`~tcip_mcp.pipelines.training.evaluation.scaled_to`). The holdout passes when its
    present-scoped mean count bias, pooled and per class, is equivalent to zero at
    ``count_bias_tolerance_frac`` of that scope's typical count, the bias plus ``equivalence_z``
    one-sided standard errors (:meth:`CountEvidence.scope_equivalence`); when every class the
    calibration side evidences at the conf is evidenced on the holdout; when the held-out
    precision and recall both clear ``holdout_match_quality_floor`` (a holdout with ground truth
    and no kept detection has no precision and fails as ``holdout_nothing_to_score``, one with no
    ground truth has no recall, which this check leaves to ``insufficient_holdout_gt``); and when
    the held-out per-image count error at ``dispersion_quantile`` is within
    ``count_error_tolerance``. The entry carries every one of
    those fields; the evidence records ``equivalence_z``, the tolerance fraction, each scope's
    typical count, and each curve its quantile and the
    criterion it matched under. Every matching is kept in ``held`` where one is given
    (:func:`~tcip_mcp.pipelines.training.evaluation.matched`).
    A conf at or below the floor fails as ``conf_censored``.
    """
    picker, conf_label = COUNT_OBJECTIVE_PICKERS[entry.count_objective]
    criterion = resolve_match_criterion(entry, cal_records)
    cal_criterion = scaled_to(criterion, cal_records)
    hold_criterion = scaled_to(criterion, hold_records)
    cal_curve = derive_operating_point_curve(
        cal_records,
        criterion=cal_criterion,
        dispersion_quantile=dispersion_quantile,
        held=held,
    )
    conf = picker(cal_curve)
    hb, cb = (
        derive_operating_point_curve(
            records,
            criterion=side_criterion,
            dispersion_quantile=dispersion_quantile,
            conf_grid=[conf],
            held=held,
        )["curve"][0]
        for records, side_criterion in (
            (hold_records, hold_criterion),
            (cal_records, cal_criterion),
        )
    )
    evidence = CountEvidence(
        conf=conf,
        conf_derived_from=conf_label,
        calibration_curve=cal_curve,
        f1_max_conf=pick_f1_max(cal_curve),
        localization=criterion,
        holdout_at_conf=hb,
        calibration_at_conf=cb,
        count_bias_tolerance_frac=cast(float, entry.count_bias_tolerance_frac),
        typical_count=gt_class_typical_count(hold_records),
        per_class_typical_count={
            cid: gt_class_typical_count(hold_records, class_id=int(cid))
            for cid in hb["per_class"]
        },
        holdout_missing_classes=sorted(
            classes_with_evidence(cb) - classes_with_evidence(hb)
        ),
        equivalence_z=equivalence_z,
        staged_conf_floor=staged_conf_floor,
        observed_min_score=_min_dt_score(cal_records + hold_records),
        calibration_cap_saturated_frac=cap_saturated_frac(cal_records),
        holdout_cap_saturated_frac=cap_saturated_frac(hold_records),
    )
    floor = cast(float, entry.holdout_match_quality_floor)
    failures: list[FailureName] = []
    if conf <= staged_conf_floor:
        failures.append("conf_censored")
    if not sum(len(gt_objects(r)) for r in cal_records):
        failures.append("insufficient_calibration_gt")
    if not sum(len(gt_objects(r)) for r in hold_records):
        failures.append("insufficient_holdout_gt")
    if hb["n_present"] < REFERENCE_FLOOR:
        failures.append("insufficient_holdout_images")
    if not evidence.scope_equivalence()[0]:
        failures.append("count_bias_exceeds_tolerance")
    if any(not evidence.scope_equivalence(cid)[0] for cid in hb["per_class"]):
        failures.append("count_bias_exceeds_tolerance_per_class")
    if any(
        0 < c["n_present"] < REFERENCE_FLOOR for c in hb["per_class"].values()
    ):
        failures.append("insufficient_holdout_images_per_class")
    if evidence.holdout_missing_classes:
        failures.append("holdout_missing_class")
    if hb["recall"] is not None:
        if hb["precision"] is None:
            failures.append("holdout_nothing_to_score")
        elif min(hb["precision"], hb["recall"]) < floor:
            failures.append("localization_quality_floor_failed")
    if hb["count_error_at_quantile"] > cast(
        float, entry.count_error_tolerance
    ):
        failures.append("count_error_dispersion_too_high")
    return conf, evidence, failures


def _classification_kappa(items: list[dict]) -> float | None:
    """Return Cohen's kappa between the reference's and the model's positive/negative call.

    The calls are over matched instances, chance-corrected from the reference's own base rates;
    ``None`` for no items or a single class on either side.
    """
    n = len(items)
    if n == 0:
        return None
    true_pos = sum(1 for it in items if it["is_true_positive"])
    pred_pos = sum(1 for it in items if it["is_pred_positive"])
    agree = sum(
        1 for it in items if it["is_true_positive"] == it["is_pred_positive"]
    )
    if true_pos in (0, n) or pred_pos in (0, n):
        return None
    po = agree / n
    p_true_pos, p_pred_pos = true_pos / n, pred_pos / n
    pe = p_true_pos * p_pred_pos + (1 - p_true_pos) * (1 - p_pred_pos)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def classifier_criterion(
    cal_items: list[dict],
    hold_items: list[dict],
    entry: TraitEntry,
    *,
    equivalence_z: float,
) -> tuple[ClassifierEvidence, list[FailureName]]:
    """Assess the held-out agreement of a positive-state call over matched instances.

    Returns ``(evidence, failures)``. Each item is one reference instance matched to one
    prediction, ``{"image_id", "is_true_positive", "is_pred_positive"}``. The holdout passes when
    Cohen's kappa over its items is above zero and above ``classifier_agreement_floor``, and when
    its present-scoped per-image positive-count bias is equivalent to zero at
    ``count_bias_tolerance_frac`` of the typical positive count, at ``equivalence_z`` one-sided
    standard errors. Both sides must carry positive evidence and the holdout at least two items
    over at least two images.
    """
    by_image: dict[str, list[dict]] = {}
    for it in hold_items:
        by_image.setdefault(it["image_id"], []).append(it)
    biases = [
        sum(i["is_pred_positive"] for i in its)
        - sum(i["is_true_positive"] for i in its)
        for its in by_image.values()
        if any(i["is_pred_positive"] or i["is_true_positive"] for i in its)
    ]
    mean = statistics.fmean(biases) if biases else 0.0
    std = statistics.stdev(biases) if len(biases) > 1 else 0.0
    typical = mean_of_present_counts(
        sum(i["is_true_positive"] for i in its) for its in by_image.values()
    )
    bias_ok, tolerance = bias_equivalence(
        mean,
        std,
        len(biases),
        tolerance_frac=cast(float, entry.count_bias_tolerance_frac),
        typical_count=typical,
        equivalence_z=equivalence_z,
    )
    kappa = _classification_kappa(hold_items)
    floor = cast(float, entry.classifier_agreement_floor)

    failures: list[FailureName] = []
    if not any(it["is_true_positive"] for it in cal_items):
        failures.append("insufficient_calibration_positive_evidence")
    if not any(it["is_true_positive"] for it in hold_items):
        failures.append("insufficient_holdout_positive_evidence")
    if len(hold_items) < REFERENCE_FLOOR:
        failures.append("insufficient_holdout_items")
    if len(biases) < REFERENCE_FLOOR:
        failures.append("insufficient_holdout_images")
    if not bias_ok:
        failures.append("positive_count_bias_exceeds_tolerance")
    if kappa is None or kappa <= 0.0 or kappa <= floor:
        failures.append("classifier_agreement_below_floor")
    evidence = ClassifierEvidence(
        kappa=kappa,
        kappa_floor=floor,
        positive_count_bias=mean,
        positive_count_bias_std=std,
        positive_count_bias_images=len(biases),
        typical_positive_count=typical,
        positive_count_bias_tolerance=tolerance,
        n_calibration=len(cal_items),
        n_holdout=len(hold_items),
    )
    return evidence, failures


def scalar_criterion(
    hold_items: list[dict],
    *,
    score: Callable[[Any, Any], float | None],
    floor: float,
    criterion: str,
) -> tuple[ScalarEvidence, list[FailureName]]:
    """Assess a per-image scalar prediction's held-out skill.

    ``criterion``'s ``score`` over the holdout items' ``(predicted, true)`` values must be finite,
    above zero and above ``floor``. Returns ``(evidence, failures)``, the evidence's score
    ``None`` when it is undefined or not finite.
    """
    import torch

    true = torch.tensor([float(it["true"]) for it in hold_items])
    predicted = torch.tensor([float(it["predicted"]) for it in hold_items])
    value = score(predicted, true) if hold_items else None
    finite = finite_or_none(value)
    failures: list[FailureName] = []
    if len(hold_items) < REFERENCE_FLOOR:
        failures.append("insufficient_holdout_items")
    if value is None:
        failures.append("criterion_undefined")
    elif finite is None:
        failures.append("criterion_not_finite")
    elif not (finite > 0.0 and finite > floor):
        failures.append("criterion_below_floor")
    evidence = ScalarEvidence(
        criterion=criterion,
        score=finite,
        floor=floor,
        n_holdout=len(hold_items),
    )
    return evidence, failures


def spatial_disjointness(
    spatial: SpatialManifest, rects: Sequence[tuple[int, int, int, int]]
) -> list[str]:
    """Return the reference rects not held out from training under ``spatial``'s regions.

    Rects are in full-mosaic pixel coordinates; ``spatial`` is a within-image split's recorded
    regions. Each rect must lie inside one of its ``val_region``/``holdout_region``/
    ``calibration_region`` rects and overlap none of its ``train_region`` rects.
    """
    from tcip_mcp.pipelines.raster_source import (
        rect_contains_rect,
        rects_overlap,
    )

    train = spatial.train_region
    held = [
        *spatial.val_region,
        *spatial.holdout_region,
        *spatial.calibration_region,
    ]
    return sorted(
        str(list(rect))
        for rect in rects
        if not any(rect_contains_rect(h, rect) for h in held)
        or any(rects_overlap(t, rect) for t in train)
    )
