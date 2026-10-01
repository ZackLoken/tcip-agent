"""The measurement criteria an assessment computes, each reading its tolerances, floors and
objective from the trait entry it is handed, and the detector knobs an execution record sets."""

from __future__ import annotations

import math
import statistics
from typing import Any, Callable, Sequence, cast

from tcip_store import non_finite_state, stored_number

from tcip_mcp.pipelines.training.evaluation import (
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
from tcip_mcp.traits import COUNT_UNBIASED, DETECTION_F1, PRESENCE, TraitEntry

COUNT_OBJECTIVE_PICKERS: dict[str, tuple[Callable[[dict], float | None], str]] = {
    COUNT_UNBIASED: (pick_count_unbiased, "count-unbiased count curve"),
    DETECTION_F1: (pick_f1_max, "F1-max count curve"),
    PRESENCE: (pick_f1_max, "F1-max count curve"),
}
"""The count objectives a calibration can fit, each as the picker it runs over the calibration
curve and the label its conf is recorded under."""

_EQUIVALENCE_Z = 1.645
"""The one-sided multiplier of the mean-plus-standard-error equivalence test (about 95 percent)."""

REGRESSION_CRITERIA: dict[str, Callable[[Any, Any], float | None]] = {
    "r_squared": r_squared,
    "concordance_correlation_coefficient": concordance_correlation_coefficient,
}
"""The regression skill statistics a caller may assess a continuous prediction by."""


def _effective_count_bias_tolerance(tolerance_frac: float, typical_count: float, n: int) -> float:
    """The absolute per-image count-bias tolerance one scope (pooled, or one class) is held to:
    ``tolerance_frac`` of the scope's typical per-image count, floored at ``1 / n``, one whole
    miscount spread across the ``n`` samples the scope's evidence rests on; ``0.0`` at ``n == 0``.
    """
    return max(tolerance_frac * typical_count, 1.0 / n) if n > 0 else 0.0


def _bias_equivalence(mean: float, std: float, n: int, *, tolerance_frac: float,
                      typical_count: float) -> tuple[bool, float]:
    """Whether a bias measured over ``n`` present samples is equivalent to zero at the scope's
    tolerance (mean plus one-sided standard error within it), and that tolerance. ``n == 0`` never
    passes."""
    tolerance = _effective_count_bias_tolerance(tolerance_frac, typical_count, n)
    if n == 0:
        return False, tolerance
    return abs(mean) + _EQUIVALENCE_Z * std / math.sqrt(n) <= tolerance, tolerance


OPERATING_POINT_ATTRS = ("score_thresh", "detections_per_img")
"""The two knobs the detection operating point governs, wherever a module holds them."""


def detector_operating_point_holder(model: Any) -> tuple[Any, str | None]:
    """Where a model's operating-point knobs live, and the attribute path to name it by.

    Checked in this order, the first that exposes any of :data:`OPERATING_POINT_ATTRS`: the module
    itself, its ``.detector``'s ``roi_heads`` (two-stage detectors), its ``.detector`` (one-stage).
    Returns ``(None, None)`` when no candidate exposes either.

    Raises ``ValueError``, naming both locations, when more than one candidate exposes a knob.
    """
    det = getattr(model, "detector", None)
    candidates = ((model, "self"), (getattr(det, "roi_heads", None), "detector.roi_heads"),
                  (det, "detector"))
    matches = [(holder, path) for holder, path in candidates
               if holder is not None and any(hasattr(holder, attr) for attr in OPERATING_POINT_ATTRS)]
    if len(matches) > 1:
        raise ValueError(
            "this model exposes an operating-point knob at more than one location "
            f"({', '.join(path for _, path in matches)}); the platform cannot choose between "
            "them. Expose the knob at exactly one of self, .detector.roi_heads, or .detector."
        )
    return matches[0] if matches else (None, None)


def set_detector_operating_point(model: Any, *, score_thresh: float | None = None,
                                 detections_per_img: int | None = None,
                                 ) -> tuple[dict, str | None]:
    """Set the in-model thresholds so the operating point governs which boxes exist.

    Resolves where the knobs live through :func:`detector_operating_point_holder`. Returns
    ``(applied, attribute_path)``: ``applied`` holds only the knobs actually set, and
    ``attribute_path`` is the holder's own path, or ``None`` when nothing exposed any knob.
    """
    target, path = detector_operating_point_holder(model)
    applied: dict = {}
    if target is not None:
        for attr, val in zip(OPERATING_POINT_ATTRS, (score_thresh, detections_per_img)):
            if val is not None and hasattr(target, attr):
                setattr(target, attr, val)
                applied[attr] = val
    return applied, path


STAGED_CONF_FLOOR = 0.01
"""The conf an assessment collects its reference predictions at, so hesitant detections survive to
be swept; the floor applied is recorded beside the criterion."""
STAGED_CONF_FLOOR_SOURCE = "staged collection floor"
"""The source an execution record names for :data:`STAGED_CONF_FLOOR`."""


def _min_dt_score(records: list[dict]) -> float | None:
    """Lowest detection score across a reference, or None if it holds no detections."""
    scores = [d["score"] for rec in records for d in rec["dt"]]
    return min(scores) if scores else None


def cap_saturated_frac(records: list[dict]) -> float:
    """Fraction of ``records`` whose raw detection count hit the collection pass's per-image cap;
    every record states ``cap_hit``."""
    return sum(bool(r["cap_hit"]) for r in records) / len(records) if records else 0.0


def count_criterion(
    cal_records: list[dict], hold_records: list[dict], entry: TraitEntry, *,
    staged_conf_floor: float | None, staged_conf_floor_attribute_path: str | None,
) -> tuple[float, dict, list[str]]:
    """The conf the trait's count objective picks on the calibration side, and the held-out count
    check at that conf: ``(conf, evidence, failures)``.

    The records are per-image COCO records (``gt`` objects and every ``dt`` detection with its
    score) collected at ``staged_conf_floor``, matched under the trait's localization resolved
    once over the calibration records
    (:func:`~tcip_mcp.pipelines.training.evaluation.resolve_match_criterion`, the evidence's
    ``localization``) and applied to each side
    (:func:`~tcip_mcp.pipelines.training.evaluation.scaled_to`). The holdout passes when its
    present-scoped mean count bias, pooled and per class, is equivalent to zero at
    ``count_bias_tolerance_frac`` of that scope's typical count; when every class the calibration
    side evidences at the conf is evidenced on the holdout; when the held-out precision and recall
    both clear ``holdout_match_quality_floor``; and when the held-out 90th-percentile per-image
    count error is within ``count_error_tolerance``. The entry carries every one of those fields.
    A floor no module attribute took (``staged_conf_floor`` ``None``, beside the attribute path
    it was applied on) fails as ``conf_floor_unstated``, a conf at or below a stated one as
    ``conf_censored``. An objective with no registered picker, or a calibration curve the
    objective picks no conf on, refuses (``ValueError``).
    """
    if entry.count_objective not in COUNT_OBJECTIVE_PICKERS:
        raise ValueError(f"count_objective {entry.count_objective!r} has no registered picker "
                         f"(registered: {sorted(COUNT_OBJECTIVE_PICKERS)}).")
    picker, conf_label = COUNT_OBJECTIVE_PICKERS[entry.count_objective]
    criterion = resolve_match_criterion(entry, cal_records)
    cal_criterion = scaled_to(criterion, cal_records)
    hold_criterion = scaled_to(criterion, hold_records)
    cal_curve = derive_operating_point_curve(cal_records, criterion=cal_criterion)
    picked = picker(cal_curve)
    if picked is None:
        raise ValueError(
            f"the calibration side's count curve holds no conf the {entry.count_objective} "
            "objective picks: its predictions carry no detection to sweep. Assess a checkpoint "
            "that detects the subject, over a reference that holds it.")
    conf = float(picked)
    hb = derive_operating_point_curve(hold_records, criterion=hold_criterion,
                                      conf_grid=[conf])["curve"][0]
    cb = derive_operating_point_curve(cal_records, criterion=cal_criterion,
                                      conf_grid=[conf])["curve"][0]
    tolerance_frac = cast(float, entry.count_bias_tolerance_frac)
    pooled_typical = gt_class_typical_count(hold_records)
    pooled_ok, pooled_tolerance = _bias_equivalence(
        hb["count_bias_mean_present"], hb["count_bias_std_present"], hb["n_present"],
        tolerance_frac=tolerance_frac, typical_count=pooled_typical)
    per_class: dict[str, dict] = {}
    for cid, s in hb["per_class"].items():
        typical = gt_class_typical_count(hold_records, class_id=int(cid))
        ok, tolerance = _bias_equivalence(
            s["count_bias_mean_present"], s["count_bias_std_present"], s["n_present"],
            tolerance_frac=tolerance_frac, typical_count=typical)
        per_class[cid] = {"bias": s["count_bias_mean_present"], "typical_count": typical,
                          "tolerance": tolerance, "passed": ok, "n_present": s["n_present"]}
    missing_classes = sorted(classes_with_evidence(cb) - classes_with_evidence(hb))
    floor = cast(float, entry.holdout_match_quality_floor)
    failures: list[str] = []
    if staged_conf_floor is None:
        failures.append("conf_floor_unstated")
    elif conf <= staged_conf_floor:
        failures.append("conf_censored")
    if not sum(len(gt_objects(r)) for r in cal_records):
        failures.append("insufficient_calibration_gt")
    if not sum(len(gt_objects(r)) for r in hold_records):
        failures.append("insufficient_holdout_gt")
    if hb["n_present"] < 2:
        failures.append("insufficient_holdout_images")
    if not pooled_ok:
        failures.append("count_bias_exceeds_tolerance")
    if any(not c["passed"] for c in per_class.values()):
        failures.append("count_bias_exceeds_tolerance_per_class")
    if any(c["n_present"] == 1 for c in per_class.values()):
        failures.append("insufficient_holdout_images_per_class")
    if missing_classes:
        failures.append("holdout_missing_class")
    if hb["precision"] < floor or hb["recall"] < floor:
        failures.append("localization_quality_floor_failed")
    if hb["count_error_p90"] > cast(float, entry.count_error_tolerance):
        failures.append("count_error_dispersion_too_high")

    observed_min = _min_dt_score(cal_records + hold_records)
    evidence = {
        "conf": conf, "conf_derived_from": conf_label, "calibration_curve": cal_curve,
        "f1_max_conf": pick_f1_max(cal_curve), "localization": criterion,
        "holdout_at_conf": hb, "calibration_at_conf": cb,
        "pooled_typical_count": pooled_typical, "pooled_count_bias_tolerance": pooled_tolerance,
        "per_class": per_class, "holdout_missing_classes": missing_classes,
        "equivalence_z": _EQUIVALENCE_Z,
        "staged_conf_floor": staged_conf_floor,
        "staged_conf_floor_attribute_path": staged_conf_floor_attribute_path,
        "observed_min_score": observed_min,
        "conf_floor_mismatch": (staged_conf_floor is not None and observed_min is not None
                                and observed_min > staged_conf_floor + 0.05),
        "calibration_cap_saturated_frac": cap_saturated_frac(cal_records),
        "holdout_cap_saturated_frac": cap_saturated_frac(hold_records),
    }
    return conf, evidence, failures


def _classification_kappa(items: list[dict]) -> float | None:
    """Cohen's kappa between the reference's and the model's positive/negative call over matched
    instances, chance-corrected from the reference's own base rates; ``None`` for no items or a
    single class on either side."""
    n = len(items)
    if n == 0:
        return None
    true_pos = sum(1 for it in items if it["is_true_positive"])
    pred_pos = sum(1 for it in items if it["is_pred_positive"])
    agree = sum(1 for it in items if it["is_true_positive"] == it["is_pred_positive"])
    if true_pos in (0, n) or pred_pos in (0, n):
        return None
    po = agree / n
    p_true_pos, p_pred_pos = true_pos / n, pred_pos / n
    pe = p_true_pos * p_pred_pos + (1 - p_true_pos) * (1 - p_pred_pos)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def classifier_criterion(cal_items: list[dict], hold_items: list[dict],
                         entry: TraitEntry) -> tuple[dict, list[str]]:
    """The held-out agreement of a positive-state call over matched instances:
    ``(evidence, failures)``.

    Each item is one reference instance matched to one prediction, ``{"image_id",
    "is_true_positive", "is_pred_positive"}``. The holdout passes when Cohen's kappa over its items
    is above zero and above ``classifier_agreement_floor``, and when its present-scoped per-image
    positive-count bias is equivalent to zero at ``count_bias_tolerance_frac`` of the typical
    positive count. Both sides must carry positive evidence and the holdout at least two items over
    at least two images.
    """
    by_image: dict[str, list[dict]] = {}
    for it in hold_items:
        by_image.setdefault(it["image_id"], []).append(it)
    biases = [sum(i["is_pred_positive"] for i in its) - sum(i["is_true_positive"] for i in its)
              for its in by_image.values()
              if any(i["is_pred_positive"] or i["is_true_positive"] for i in its)]
    mean = statistics.fmean(biases) if biases else 0.0
    std = statistics.stdev(biases) if len(biases) > 1 else 0.0
    typical = mean_of_present_counts(sum(i["is_true_positive"] for i in its)
                                     for its in by_image.values())
    bias_ok, tolerance = _bias_equivalence(
        mean, std, len(biases), tolerance_frac=cast(float, entry.count_bias_tolerance_frac),
        typical_count=typical)
    kappa = _classification_kappa(hold_items)
    floor = cast(float, entry.classifier_agreement_floor)

    failures: list[str] = []
    if not any(it["is_true_positive"] for it in cal_items):
        failures.append("insufficient_calibration_positive_evidence")
    if not any(it["is_true_positive"] for it in hold_items):
        failures.append("insufficient_holdout_positive_evidence")
    if len(hold_items) < 2:
        failures.append("insufficient_holdout_items")
    if len(biases) < 2:
        failures.append("insufficient_holdout_images")
    if not bias_ok:
        failures.append("positive_count_bias_exceeds_tolerance")
    if kappa is None or kappa <= 0.0 or kappa <= floor:
        failures.append("classifier_agreement_below_floor")
    evidence = {
        "kappa": kappa, "kappa_floor": floor, "positive_count_bias": mean,
        "positive_count_bias_std": std, "positive_count_bias_images": len(biases),
        "typical_positive_count": typical, "positive_count_bias_tolerance": tolerance,
        "n_calibration": len(cal_items), "n_holdout": len(hold_items),
    }
    return evidence, failures


def scalar_criterion(hold_items: list[dict], *, score: Callable[[Any, Any], float | None],
                     floor: float, criterion: str) -> tuple[dict, list[str]]:
    """A per-image scalar prediction's held-out skill: ``criterion``'s ``score`` over the holdout
    items' ``(predicted, true)`` values must be finite, above zero and above ``floor``.
    ``(evidence, failures)``."""
    import torch

    true = torch.tensor([float(it["true"]) for it in hold_items])
    predicted = torch.tensor([float(it["predicted"]) for it in hold_items])
    value = score(predicted, true) if hold_items else None
    state = non_finite_state(value) if isinstance(value, float) else None
    failures: list[str] = []
    if len(hold_items) < 2:
        failures.append("insufficient_holdout_items")
    if value is None:
        failures.append("criterion_undefined")
    elif state is not None:
        failures.append("criterion_not_finite")
    elif not (value > 0.0 and value > floor):
        failures.append("criterion_below_floor")
    evidence = {"criterion": criterion, **stored_number("score", value), "floor": floor,
                "n_holdout": len(hold_items)}
    return evidence, failures


def spatial_disjointness(spatial: dict, rects: Sequence[tuple[int, int, int, int]]) -> list[str]:
    """The reference rects (full-mosaic pixel coordinates) not held out from training under a
    within-image split's persisted regions (``spatial``, a run's ``data.split.spatial_manifest``):
    each rect must lie inside one of its ``val_region``/``test_region``/``calibration_region``
    rects and overlap none of its ``train_region`` rects. A manifest missing any of those four keys
    refuses naming it."""
    from tcip_mcp.pipelines.raster_source import rect_contains_rect, rects_overlap

    missing = [k for k in ("train_region", "val_region", "test_region", "calibration_region")
               if k not in spatial]
    if missing:
        raise ValueError(f"the run's spatial manifest records no {missing}: the partition it "
                         "certifies is not stated whole, so containment cannot be checked.")
    train = [tuple(r) for r in spatial["train_region"]]
    held = [tuple(r) for key in ("val_region", "test_region", "calibration_region")
            for r in spatial[key]]
    return sorted(str(list(rect)) for rect in rects
                  if not any(rect_contains_rect(h, rect) for h in held)
                  or any(rects_overlap(t, rect) for t in train))
