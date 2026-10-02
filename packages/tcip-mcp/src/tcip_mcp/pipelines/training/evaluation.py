"""Task-aware evaluation metrics + composite selection objective:
  * the pycocotools-backed detection / instance_seg metrics (mAP + operating-point TP/FP/FN), the
  canonical COCO mAP definition;
  * in-house scalar metrics for classification / ordinal / regression;
  * the composite selection objective (lower = better);
  * a task-agnostic two-pass ``evaluate()``.

Every pycocotools call runs with stdout redirected, since it prints to stdout.
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from tcip_store import stored_number, stored_numbers

# Every box handed to pycocotools goes through this, both sides of a match on the one stored grid.
from tcip_annotation.json_io import xywh

if TYPE_CHECKING:
    from tcip_mcp.traits import TraitEntry

logger = logging.getLogger(__name__)

# Composite-objective weights. Note: in compute_composite_objective the F1 and
# mAP50 terms are multiplied by 10 to lift them onto the same scale as val_loss,
# so a weight here acts on that *scaled* term (a 0.35 f1 weight ~ 3.5 loss-units
# of pull at f1=0). See compute_composite_objective for the exact formula.
# These weights silently decide which checkpoint wins, so they are a caller-owned selection policy
# (validated=false, not a data derivation): overridable via the ``score_weights`` kwarg on every
# eval surface. Documented default, not a frozen truth, no derivation label is claimed for it.
DEFAULT_SCORE_WEIGHTS: dict[str, float] = {"loss": 0.45, "f1": 0.35, "map50": 0.20}

# The metric keys that ``evaluate()`` labels comparability-only (``map50_role``) once a center-match
# trait's own governing criterion takes over ``precision``/``recall``/``f1`` (see the center_match
# branch below), the AP@0.5-family keys plus the IoU@0.5-convention precision/recall/F1 that get
# relabeled ``iou_*`` at that point. The single source of truth for "is this metric governing or
# comparability-only for a center-match trait", ``resolve_selection_metric`` (generic_trainer.py)
# and ``rank_registered_models`` (model_tools.py) both import this rather than re-encoding the names.
CENTER_MATCH_COMPARABILITY_KEYS: frozenset[str] = frozenset({
    "map50", "map", "map_at_maxdets", "map50_at_maxdets",
    "iou_precision", "iou_recall", "iou_f1",
})

VAL_METRIC_PREFIX = "val_"
"""What ``_validate`` (generic_trainer.py) prefixes every metric key with before it reaches a
run's metrics log or a registry entry. Declared once here so a ranking reader strips it without
importing the training stack."""

HIGHER_IS_BETTER_BY_METRIC: dict[str, bool] = {
    "loss": False,
    "objective": False,
    "mae": False,
    "rmse": False,
    "precision": True,
    "recall": True,
    "f1": True,
    "map": True,
    "map50": True,
    "map_at_maxdets": True,
    "map50_at_maxdets": True,
    "iou_precision": True,
    "iou_recall": True,
    "iou_f1": True,
    "accuracy": True,
    "rank_acc": True,
    "quadratic_weighted_kappa": True,
    "r_squared": True,
    "mIoU": True,
    "dice": True,
    "pixel_acc": True,
}
"""Direction of a better value, keyed by the bare (un-``val_``-prefixed) metric name, for every
scalar ``evaluate()`` (or ``governing_counts``) returns across the tasks it scores. A raw count
(``tp``/``fp``/``fn``), a signed bias (``count_bias_mean``) and a non-finite value's state
companion (``tcip_store.values.NOT_FINITE_SUFFIX``) have no direction and are not listed."""


def _rounded(value):
    """One metric at the reported precision, leaving a non-finite or non-numeric value alone.

    Rounding a value that is not a number raises, and rounding a non-finite one changes
    nothing, so both are handed on for the caller to represent rather than forced through
    here.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return round(value, 6) if math.isfinite(value) else value


def _reported_metrics(values: dict) -> dict:
    """One task's metrics as the result record carries them.

    Scalars are rounded and a non-finite one becomes null beside a field naming its state;
    the per-class mappings some tasks return, which already carry None for an absent class,
    pass through untouched.
    """
    return stored_numbers({k: _rounded(v) for k, v in values.items()})


# ====================================================================
# Composite selection objective
# ====================================================================

def compute_composite_objective(
    val_loss: float, f1: float, map50: float, score_weights: dict | None = None
) -> float | None:
    """Lower-is-better selection/tuning score blending loss, F1 and mAP50, or ``None`` when
    the epoch has no useful score.

    ``w["loss"]*loss + w["f1"]*(1-f1)*10 + w["map50"]*(1-map50)*10``; the ``*10`` lifts the
    unit-interval quality terms to a typical loss magnitude. A degenerate epoch (a non-positive
    or non-finite loss, or both quality terms at zero) has no score at all, so it answers
    ``None`` rather than a number: a record carries ``None`` as "not measured", the selection
    comparison treats it as never improving, and no chart plots it as if it were a value.
    """
    w = score_weights or DEFAULT_SCORE_WEIGHTS
    vl = float(val_loss) if (val_loss is not None and math.isfinite(val_loss)) else float("inf")
    f1v = float(f1) if (f1 is not None and math.isfinite(f1)) else 0.0
    m50 = float(map50) if (map50 is not None and math.isfinite(map50)) else 0.0
    if vl <= 0 or not math.isfinite(vl):
        return None
    if f1v < 0.01 and m50 < 0.01:
        return None
    return w["loss"] * vl + w["f1"] * (1.0 - f1v) * 10 + w["map50"] * (1.0 - m50) * 10


# ====================================================================
# pycocotools detection / instance_seg metrics
# ====================================================================

def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision, recall and F1 from counts: each ``0.0`` when its denominator is empty."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def build_coco_image_record(width: float, height: float, gt: list[dict], dt: list[dict],
                            image_id=None) -> dict:
    """One per-image entry: ``{'width','height','gt':[ann...],'dt':[res...]}`` (+ optional image_id)."""
    rec = {"width": int(width), "height": int(height), "gt": list(gt), "dt": list(dt)}
    if image_id is not None:
        rec["image_id"] = image_id
    return rec


def _counts_at_operating_point(coco_eval, iou_threshold: float, conf_threshold: float) -> dict:
    """Walk ``COCOeval.evalImgs`` to extract TP/FP/FN at a (conf, iou) point."""
    p = coco_eval.params
    iou_thrs = list(p.iouThrs)
    t = min(range(len(iou_thrs)), key=lambda i: abs(iou_thrs[i] - iou_threshold))
    area_all = p.areaRng[0]

    tp = fp = total_gt = 0
    per_image: dict[int, dict] = {}
    for e in coco_eval.evalImgs:
        if e is None or e["aRng"] != area_all:
            continue
        img_id = e["image_id"]
        gt_ignore = np.asarray(e["gtIgnore"])
        n_gt = int((gt_ignore == 0).sum())
        total_gt += n_gt
        dt_scores = np.asarray(e["dtScores"])
        dt_matches = np.asarray(e["dtMatches"])
        dt_ignore = np.asarray(e["dtIgnore"])
        e_tp = e_fp = 0
        for d in range(dt_scores.shape[0] if dt_scores.size else 0):
            # strict > matches deployed torchvision's in-model score_thresh (keeps score > thresh)
            if dt_scores[d] <= conf_threshold or dt_ignore[t, d]:
                continue
            if dt_matches[t, d] > 0:
                e_tp += 1
            else:
                e_fp += 1
        tp += e_tp
        fp += e_fp
        rec = per_image.setdefault(img_id, {"image_id": img_id, "tp": 0, "fp": 0, "gt": 0})
        rec["tp"] += e_tp
        rec["fp"] += e_fp
        rec["gt"] += n_gt

    per_image_counts = [
        {"image_id": r["image_id"], "tp": r["tp"], "fp": r["fp"], "fn": max(r["gt"] - r["tp"], 0)}
        for r in per_image.values()
    ]
    return {"tp": tp, "fp": fp, "fn": max(total_gt - tp, 0), "per_image_counts": per_image_counts}


def _ap_from_precision(coco_eval, *, iou: float | None, maxdet: int) -> float:
    """Mean AP from ``coco_eval.eval['precision']`` at a given IoU / maxDet, mirroring
    pycocotools' ``_summarize(ap=1)`` (area='all'), but indexed explicitly so we can read AP at
    both the standard 100 cap and a non-100 operating cap without summarize()'s hardcoded 100."""
    p = coco_eval.params
    s = coco_eval.eval["precision"]  # [T(iou), R(rec), K(cat), A(area), M(maxDet)]
    if iou is not None:
        t = np.where(np.isclose(p.iouThrs, iou))[0]
        s = s[t]
    mind = list(p.maxDets).index(maxdet)
    s = s[:, :, :, 0, mind]  # area 'all' is index 0
    valid = s[s > -1]
    return float(valid.mean()) if valid.size else 0.0


def coco_detection_metrics(
    per_image: list[dict],
    *,
    iou_type: str = "bbox",
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    max_dets: int = 100,
) -> dict:
    """Run ``COCOeval`` once over ``per_image`` records and return COCO metrics.

    Returns mAP at the standard 100-detection cap (``map``/``map50``/``map75``, comparable across
    runs and caps) plus the same at the operating cap (``map_at_maxdets``/``map50_at_maxdets``),
    and operating-point ``precision``/``recall``/``f1``/``tp``/``fp``/``fn`` with per-image counts.
    Short-circuits to all-zero metrics, every object a miss, when there is no prediction at all
    (``loadRes([])`` raises ``IndexError``). A reference with no object, empty or crowd regions
    alone, is still evaluated: each detection outside a crowd region is a false positive.
    """
    images, annotations, results = [], [], []
    cat_ids: set[int] = set()
    ann_id = 1
    n_pred = 0
    # Objects each image's ground truth holds: a crowd region is none, it is COCOeval's ignore.
    n_objects = [len(gt_objects(rec)) for rec in per_image]
    for img_id, rec in enumerate(per_image, start=1):
        images.append({"id": img_id, "width": int(rec["width"]), "height": int(rec["height"])})
        for ann in rec["gt"]:
            a = dict(ann)
            a["id"] = ann_id
            a["image_id"] = img_id
            annotations.append(a)
            cat_ids.add(int(a["category_id"]))
            ann_id += 1
        for res in rec["dt"]:
            r = dict(res)
            r["image_id"] = img_id
            results.append(r)
            cat_ids.add(int(r["category_id"]))
            n_pred += 1
    n_gt = sum(n_objects)

    base = {
        "map": 0.0, "map50": 0.0, "map75": 0.0,
        "map_at_maxdets": 0.0, "map50_at_maxdets": 0.0,
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "tp": 0, "fp": 0, "fn": n_gt,
        "n_images": len(per_image), "n_gt": n_gt, "n_pred": n_pred,
        "per_image_counts": [
            {"image_id": i + 1, "tp": 0, "fp": 0, "fn": n} for i, n in enumerate(n_objects)
        ],
        "iou_type": iou_type, "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold, "max_dets": max_dets,
    }
    if n_pred == 0:
        return base

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    categories = [{"id": c, "name": str(c)} for c in sorted(cat_ids)]
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        coco_gt = COCO()
        coco_gt.dataset = {"images": images, "annotations": annotations, "categories": categories}
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        # Include 100 so map/map50/map75 stay the standard, cap-comparable AP; max_dets adds the operating-cap figures.
        coco_eval.params.maxDets = sorted({1, 100, int(max_dets)})
        coco_eval.params.imgIds = [im["id"] for im in images]
        coco_eval.evaluate()
        coco_eval.accumulate()
        m_ap = _ap_from_precision(coco_eval, iou=None, maxdet=100)
        m_ap50 = _ap_from_precision(coco_eval, iou=0.5, maxdet=100)
        m_ap75 = _ap_from_precision(coco_eval, iou=0.75, maxdet=100)
        m_ap_md = _ap_from_precision(coco_eval, iou=None, maxdet=int(max_dets))
        m_ap50_md = _ap_from_precision(coco_eval, iou=0.5, maxdet=int(max_dets))
        counts = _counts_at_operating_point(coco_eval, iou_threshold, conf_threshold)

    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "map": max(m_ap, 0.0), "map50": max(m_ap50, 0.0), "map75": max(m_ap75, 0.0),
        "map_at_maxdets": max(m_ap_md, 0.0), "map50_at_maxdets": max(m_ap50_md, 0.0),
        **precision_recall_f1(tp, fp, fn),
        "tp": tp, "fp": fp, "fn": fn,
        "n_images": len(per_image), "n_gt": n_gt, "n_pred": n_pred,
        "per_image_counts": counts["per_image_counts"],
        "iou_type": iou_type, "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold, "max_dets": max_dets,
    }


# ====================================================================
# Center-match counting sweep (for count-unbiased operating-point calibration)
# ====================================================================
# For small objects IoU is noisy relative to annotation jitter; a detection counts as finding an
# object when its center lands within a derived tolerance of a GT center (the object's actual
# scale for a given trait/dataset is gt_class_avg_size's job to measure, not a pinned constant
# here). The operating point (conf) is then
# derived to minimize the signed per-image count bias E[FP-FN], not F1, because the phenotype is a
# count (Sigma pred ~= Sigma gt) for a trait whose recorded count_objective/localization say so
# (traits.py, neither is authored, both are derived/decided once and recorded).

def gt_objects(rec: dict, *, crowd: bool = False) -> list[dict]:
    """A per-image record's ground-truth objects: every ``gt`` entry but a crowd region, which is
    COCO's ignore region and never one object in a count, a size or a spacing; with ``crowd``, the
    crowd regions instead.
    """
    return [a for a in rec["gt"] if bool(a["iscrowd"]) is crowd]


def gt_facts(rec: dict) -> list:
    """A per-image record's ground truth as content: each record's class, box and crowd flag, in
    a fixed order. The one projection every content identity of ground truth hashes."""
    return sorted([g["category_id"], g["bbox"], g["iscrowd"]] for g in rec["gt"])


def _char_size_xywh(a: dict) -> float:
    """Characteristic size of a box = sqrt(w*h), scale-robust for a tolerance basis."""
    w, h = float(a["bbox"][2]), float(a["bbox"][3])
    return (max(w, 0.0) * max(h, 0.0)) ** 0.5


def gt_class_avg_size(per_image: list[dict], class_id: int | None = None) -> float:
    """Average characteristic GT box size, the derived basis for the center-match tolerance.

    Derived from the data in hand (not pinned): the tolerance is ``half_class_avg_size`` (traits.py).
    """
    sizes = [
        _char_size_xywh(a)
        for rec in per_image for a in gt_objects(rec)
        if class_id is None or a["category_id"] == class_id
    ]
    return float(np.mean(sizes)) if sizes else 0.0


def mean_of_present_counts(counts: Iterable[int]) -> float:
    """Mean of the entries in ``counts`` that are actually positive (> 0): the "typical, when
    present" statistic behind a relative count-bias tolerance's derived denominator.
    """
    present = [c for c in counts if c > 0]
    return float(np.mean(present)) if present else 0.0


def gt_class_typical_count(per_image: list[dict], class_id: int | None = None) -> float:
    """Mean per-image GT count for ``class_id`` (all classes pooled when ``None``), the derived
    denominator a relative count-bias tolerance scales against
    (:func:`operating_point._bias_equivalence_ok`).

    GT-only and conf-independent, a distinct notion of "present" from ``_count_stats_at_conf``'s
    ``n_present`` (a counted object or detection, at one conf): a class with detections but no real
    GT anywhere derives 0.
    """
    counts = [
        sum(1 for a in gt_objects(rec) if class_id is None or a["category_id"] == class_id)
        for rec in per_image
    ]
    return mean_of_present_counts(counts)


def _match_image(gt: list[dict], dt: list[dict], criterion: dict) -> tuple[int, int, int]:
    """tp/fp/fn on one image from the one matcher
    (:func:`~tcip_annotation.matching.pair_detections`): a crowd region is no object to miss, and
    an ignored detection is neither a true nor a false positive."""
    from tcip_annotation.matching import pair_detections

    m = pair_detections(gt, dt, criterion)
    return len(m.pairs), len(m.unpaired), len(m.missed)


def localization_frac(trait: TraitEntry, boxes_per_image: list[list[list[float]]]
                      ) -> tuple[float, str]:
    """The center-match tolerance, as a fraction of the average object size, and its source: the
    ground truth's own nearest-neighbor spacing (``boxes_per_image`` one list of xywh boxes per
    image), else, when no two same-class objects sit close enough to derive it, the trait
    revision's stated ``localization_tolerance_frac``."""
    from tcip_mcp.pipelines.derivations import derive_localization_tolerance_frac

    frac = derive_localization_tolerance_frac(boxes_per_image)
    if frac is not None:
        return frac, "GT nearest-neighbor spacing (p10 + margin)"
    return trait.localization_tolerance_frac, (
        f"the trait's stated localization_tolerance_frac ({trait.localization_tolerance}); no "
        "same-class neighbor in this ground truth to derive one from")


def resolve_match_criterion(trait: TraitEntry | None, per_image: list[dict], *,
                            class_id: int | None = None, iou_threshold: float = 0.5) -> dict:
    """The localization criterion that governs a trait's phenotype count and model selection, as
    ``{kind, tolerance_frac and tolerance | iou_threshold, derived_from, trait}``, resolved once
    over the reference ``per_image`` and carried whole to every count and match over it.

    With no trait it is IoU matching at ``iou_threshold``, the labeled comparability convention
    (AP@0.5), which governs nothing on its own. With one, its stated ``localization`` governs, an
    unauthored one refusing (:class:`~tcip_mcp.traits.UnauthoredField`): a center match's tolerance
    is a fraction of the average object size (:func:`localization_frac`), scaled to ``per_image``
    here and to another reference by :func:`scaled_to`; an IoU match's threshold is the one the
    ground truth's own box sizes derive, and a reference with no box to derive it from refuses.
    """
    if trait is None:
        return {"kind": "iou_match", "iou_threshold": float(iou_threshold),
                "derived_from": "comparability convention (AP@0.5)", "trait": None}
    from tcip_mcp.pipelines.derivations import IOU_MATCH_DERIVATION, derive_iou_match_threshold
    from tcip_mcp.traits import CENTER_MATCH, authored

    authored(trait, ("localization",))
    boxes_per_image = [[a["bbox"] for a in gt_objects(rec)
                        if class_id is None or a["category_id"] == class_id]
                       for rec in per_image]
    if trait.localization == CENTER_MATCH:
        frac, frac_source = localization_frac(trait, boxes_per_image)
        return scaled_to({"kind": "center_match", "tolerance_frac": frac,
                          "derived_from": frac_source, "trait": trait.name}, per_image, class_id)
    threshold = derive_iou_match_threshold(boxes_per_image)
    if threshold is None:
        raise ValueError(
            f"trait {trait.name!r} matches by IoU, and this reference holds no ground-truth box "
            "to derive the IoU a match must reach from; evaluate against a labeled reference.")
    return {"kind": "iou_match", "iou_threshold": float(threshold),
            "derived_from": IOU_MATCH_DERIVATION, "trait": trait.name}


def scaled_to(criterion: dict, per_image: list[dict], class_id: int | None = None) -> dict:
    """``criterion`` as it applies to ``per_image``: a center match's tolerance scaled to that
    reference's own average object size; an IoU match unchanged."""
    if criterion["kind"] != "center_match":
        return criterion
    return {**criterion, "tolerance": float(
        criterion["tolerance_frac"] * gt_class_avg_size(per_image, class_id=class_id))}


def _dt_score(d: dict) -> float:
    """A detection record's confidence: a record with no ``score``, or a ``score`` of ``None``,
    refuses by name.
    """
    if "score" not in d:
        raise ValueError(f"detection record has no 'score' field, cannot count it: {d!r}")
    if d["score"] is None:
        raise ValueError(f"detection record's 'score' is None, cannot count it: {d!r}")
    return float(d["score"])


def governing_counts(per_image: list[dict], criterion: dict, *, conf_threshold: float,
                     class_id: int | None = None) -> dict:
    """tp/fp/fn/precision/recall/f1 at the criterion that governs the phenotype count, every
    conf-surviving detection matched under ``criterion`` (:func:`match_pairs`). This count is what a
    count-trait phenotype and model selection rest on, distinct from AP@0.5, which stays a labeled
    comparability metric that governs nothing.
    """
    m = _count_stats_at_conf(per_image, criterion=criterion, conf=conf_threshold,
                             class_id=class_id)
    return {"tp": int(m["tp"]), "fp": int(m["fp"]), "fn": int(m["fn"]),
            **{k: round(m[k], 6) for k in ("precision", "recall", "f1")}, "criterion": criterion}


def _count_stats_at_conf(per_image: list[dict], *, criterion: dict, conf: float,
                         class_id: int | None) -> dict:
    """Counting statistics under ``criterion`` over ``per_image`` at one conf, optionally for one
    class: the class-pooled curve entry and every per-class entry beside it.

    Two scopes of the same per-image bias travel side by side. The whole-reference statistics
    (``count_bias_mean``/``count_bias_std`` over ``n_images``) are what a conf picker compares
    across the grid: ``n_images`` is the same at every conf, so minimizing ``count_bias_mean``
    minimizes the reference's total signed miscount. The present-scoped statistics
    (``count_bias_mean_present``/``count_bias_std_present`` over ``n_present``) are what an
    equivalence gate compares against a relative tolerance scaled by a density measured over
    present images only (:func:`mean_of_present_counts`): an image with no GT and no surviving
    detection is excluded from them.
    """
    tp = fp = fn = 0
    biases: list[int] = []
    present_biases: list[int] = []
    for rec in per_image:
        gt = [a for a in rec["gt"] if class_id is None or a["category_id"] == class_id]
        dt = sorted(
            (d for d in rec["dt"]
             if _dt_score(d) >= conf and (class_id is None or d["category_id"] == class_id)),
            key=lambda d: -_dt_score(d),
        )
        t, f, n = _match_image(gt, dt, criterion)
        tp += t
        fp += f
        fn += n
        biases.append(f - n)
        if t + n + f:  # a counted object or detection; one ignored in a crowd region is neither
            present_biases.append(f - n)
    abs_biases = [abs(b) for b in biases]
    return {
        "tp": tp, "fp": fp, "fn": fn, **precision_recall_f1(tp, fp, fn),
        "count_bias_mean": float(np.mean(biases)) if biases else 0.0,
        "abs_count_error_mean": float(np.mean(abs_biases)) if biases else 0.0,
        # Tail dispersion, a p90 of |bias|, not another mean, since a mean can hide one
        # badly-off image among many. Reference-sufficiency terms, computed here, once,
        # so the gate never re-derives a second matcher over the same per-image biases.
        "count_error_p90": float(np.quantile(abs_biases, 0.9)) if abs_biases else 0.0,
        "count_bias_std": float(np.std(biases, ddof=1)) if len(biases) > 1 else 0.0,
        "n_images": len(biases),
        # Images holding a counted object or detection of this class, distinct from n_images
        # (the whole holdout) because a scope scarce in the reference gets the denominator of both
        # its bias and its standard error from how much evidence there really is, not diluted by
        # images that say nothing about it.
        "n_present": len(present_biases),
        "count_bias_mean_present": float(np.mean(present_biases)) if present_biases else 0.0,
        "count_bias_std_present": (float(np.std(present_biases, ddof=1))
                                   if len(present_biases) > 1 else 0.0),
    }


def _class_ids_present(per_image: list[dict], class_id: int | None = None) -> list[int]:
    """The class ids to break the sweep down by: ``[class_id]`` when given, whether or not the
    records carry it, else every ``category_id`` the records' gt and dt carry, sorted. An
    annotation or detection with no ``category_id`` raises ``KeyError``."""
    if class_id is not None:
        return [class_id]
    ids = {a["category_id"] for rec in per_image for a in rec["gt"]}
    ids |= {d["category_id"] for rec in per_image for d in rec["dt"]}
    return sorted(ids)


def derive_operating_point_curve(per_image: list[dict], *, criterion: dict,
                                 class_id: int | None = None,
                                 conf_grid: list[float] | None = None,
                                 max_thresholds: int = 80) -> dict:
    """Sweep the confidence threshold over ``per_image`` records, matched under ``criterion``.

    One model pass produces ``per_image`` (unfiltered dt with scores); this sweeps conf in Python,
    no re-forwarding. For each conf: aggregate TP/FP/FN and per-image count bias (FP-FN). An
    explicit ``conf_grid`` (e.g. a single-element ``[conf]``) evaluates exactly those points.
    Returns ``{criterion, class_id, curve:[{conf, tp, fp, fn, precision, recall, f1,
    count_bias_mean, abs_count_error_mean, count_error_p90, count_bias_std, n_images, n_present,
    count_bias_mean_present, count_bias_std_present, per_class}]}``. See
    :func:`_count_stats_at_conf` for the two bias scopes.

    ``per_class`` carries the same statistics measured within each class the records carry, keyed
    by ``str(category_id)`` (string keys so an in-memory sweep and one round-tripped through a
    stored JSON record have the same shape): matching is class-blind in the pooled entry, so a
    detector that calls every class-A object class B reports zero pooled bias while both per-class counts
    are wrong. Class ids come from the records themselves.
    """
    scores = sorted({_dt_score(d) for rec in per_image for d in rec["dt"]})
    if conf_grid is None:
        if len(scores) > max_thresholds:
            conf_grid = list(np.linspace(scores[0], scores[-1], max_thresholds))
        else:
            conf_grid = list(scores)
        conf_grid = sorted(set([0.0, *conf_grid]))
    class_ids = _class_ids_present(per_image, class_id)
    curve: list[dict] = []
    for conf in conf_grid:
        pooled = _count_stats_at_conf(per_image, criterion=criterion, conf=conf, class_id=class_id)
        if len(class_ids) == 1:
            # Filtering to the only class present is a no-op on both gt and dt, so the pooled entry
            # is that class's entry, reused rather than recomputed, which keeps the single-class
            # sweep (every reference the platform builds) at one pass's cost.
            per_class = {str(class_ids[0]): pooled}
        else:
            per_class = {str(cid): _count_stats_at_conf(per_image, criterion=criterion, conf=conf,
                                                        class_id=cid)
                         for cid in class_ids}
        curve.append({"conf": float(conf), **pooled, "per_class": per_class})
    return {"criterion": criterion, "class_id": class_id, "curve": curve}


def worst_class_count_bias(entry: dict) -> float:
    """The largest |mean per-image count bias| over the classes in one curve entry, the class this
    conf serves worst, and the one the gate's per-class equivalence test refuses on.

    Falls back to the pooled bias for an entry with no per-class breakdown; a single-class sweep
    reuses the pooled entry as its one class, so the two agree there by construction.
    """
    per_class = entry.get("per_class") or {}
    if not per_class:
        return abs(entry["count_bias_mean"])
    return max(abs(s["count_bias_mean"]) for s in per_class.values())


def pick_count_unbiased(sweep: dict) -> float | None:
    """The conf that minimizes the worst per-class |mean per-image count bias| (tie-break: lower
    pooled |bias|, higher F1, lower |error|, higher conf).

    This is the count-trait operating point, where the model's totals match GT totals, which is
    generally not the F1-max point (that optimizes matching, not count agreement). It targets the
    worst class, as the gate does: with three or more classes, a conf can buy pooled balance by
    trading one class's over-count against another's under-count. On a single-class reference the
    two objectives are the same number.

    The final ``-c["conf"]`` tie-break prefers the highest of exactly tied confs (a reference
    filtered to a floor ties everything below it), the most conservative candidate among equals.
    """
    curve = sweep.get("curve") or []
    if not curve:
        return None
    best = min(curve, key=lambda c: (worst_class_count_bias(c), abs(c["count_bias_mean"]), -c["f1"],
                                     c["abs_count_error_mean"], -c["conf"]))
    return best["conf"]


def classes_with_evidence(entry: dict) -> set[str]:
    """The classes one curve entry actually says something about: those with a GT object or a
    surviving detection at that conf (``tp + fp + fn > 0``). A class whose entry is all zeros
    carries no evidence of an unbiased count.
    """
    return {cid for cid, s in (entry.get("per_class") or {}).items()
            if s["tp"] + s["fp"] + s["fn"] > 0}


def pick_f1_max(sweep: dict) -> float | None:
    """The F1-max conf, reported alongside the count-unbiased point to show the trade-off."""
    curve = sweep.get("curve") or []
    return max(curve, key=lambda c: c["f1"])["conf"] if curve else None


# ---- converters -----------------------------------------------------

def _mask_to_rle(mask) -> dict:
    """Encode a binary/soft mask (``[H,W]`` or ``[1,H,W]``) as COCO RLE for segm metrics."""
    from pycocotools import mask as mask_utils

    m = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    if m.ndim == 3:  # predicted masks arrive as [1, H, W] soft probabilities
        m = m[0]
    binary = np.asfortranarray((m >= 0.5).astype(np.uint8))
    return mask_utils.encode(binary)


def gt_record(bbox: list[float], category_id: int, crowd: Any) -> dict:
    """One ground-truth evaluation record: its ``[x, y, w, h]`` box, the ``area`` that box
    states and its crowd flag. The one shape every ground-truth record is built in."""
    return {"category_id": int(category_id), "bbox": bbox, "area": float(bbox[2] * bbox[3]),
            "iscrowd": int(crowd)}


def dt_record(bbox: list[float], category_id: Any, score: Any) -> dict:
    """One detection's evaluation record: its ``[x, y, w, h]`` box as given, its class and its
    score. The one shape every detection record is built in, whatever coordinates it is on."""
    return {"category_id": int(category_id), "bbox": bbox, "score": float(score)}


def detection_record(box: Sequence[float], label: Any, score: Any) -> dict:
    """One detection's evaluation record from a predictor's corner ``box``, on the stored grid
    (:func:`~tcip_annotation.json_io.xywh`).
    """
    return dt_record(xywh(*box), label, score)


def prediction_record(result: Mapping[str, Any], gt: list[dict], *, image_id: str) -> dict:
    """One per-image evaluation record from a detection result (its ``width``, ``height``,
    corner ``boxes``, ``scores``, ``labels`` and ``cap_hit``) and the image's ground-truth records
    ``gt``, named ``image_id``. Boxes, scores and labels differing in length refuse
    (``ValueError``)."""
    dt = [detection_record(box, label, score) for box, score, label
          in zip(result["boxes"], result["scores"], result["labels"], strict=True)]
    return {**build_coco_image_record(int(result["width"]), int(result["height"]), gt, dt,
                                      image_id=image_id),
            "cap_hit": result["cap_hit"]}


def gt_records(target: Mapping[str, Any]) -> list[dict]:
    """A target's rows, corner ``boxes`` beside ``labels`` and the crowd flag, as evaluation
    ground-truth records on the stored grid (:func:`~tcip_annotation.json_io.xywh`), from a tensor,
    array or list target alike, its crowd flags read through
    :func:`~tcip_mcp.pipelines.data.datasets.crowd_of`.
    """
    from tcip_mcp.pipelines.data.datasets import crowd_of

    def rows(values: Any) -> list:
        return values.tolist() if hasattr(values, "tolist") else list(values)

    if not len(target["boxes"]):
        return []
    return [gt_record(xywh(*box), lab, crowd)
            for box, lab, crowd in zip(rows(target["boxes"]), rows(target["labels"]),
                                       rows(crowd_of(target)))]


def records_from_detector(target: dict, output: dict, *, width: int, height: int,
                          include_masks: bool = False, detections_cap: int | None = None) -> dict:
    """torchvision GT target + detector output -> one COCO per-image record.

    With ``include_masks`` (instance_seg / Mask R-CNN) each GT and prediction also carries an RLE
    ``segmentation``, so the record can be scored with ``iou_type='segm'``.

    ``detections_cap`` (non-gating provenance): when the caller knows the in-model
        ``detections_per_img`` this output was generated under, stamp ``cap_hit``, whether this
        image's raw detection count reached that cap.
    """
    gt = gt_records(target)
    if include_masks and target.get("masks") is not None:
        for ann, mask in zip(gt, target["masks"]):
            ann["segmentation"] = _mask_to_rle(mask)
    dt = []
    pboxes = output.get("boxes")
    pmasks = output.get("masks") if include_masks else None
    if pboxes is not None and len(pboxes):
        plabels = output["labels"].detach().cpu().tolist()
        pscores = output["scores"].detach().cpu().tolist()
        for i, (box, c, s) in enumerate(zip(pboxes.detach().cpu().tolist(), plabels, pscores)):
            res = detection_record(box, c, s)
            if pmasks is not None and i < len(pmasks):
                res["segmentation"] = _mask_to_rle(pmasks[i])
            dt.append(res)
    rec = build_coco_image_record(width, height, gt, dt, image_id=target.get("image_id"))
    if detections_cap is not None:
        rec["cap_hit"] = len(dt) >= detections_cap
    return rec


def _poly_flat(points) -> list[float]:
    return [float(c) for pt in points for c in (pt[0], pt[1])]


def subject_category_ids(annotations) -> dict[str, int]:
    """Each box-derivable annotation's subject mapped to a 1-indexed COCO category id (background
    0, like detector labels), in first-seen order; a geometry-less label and a ``Point`` mint
    none."""
    from tcip_annotation.state import box_derivable

    names: list[str] = []
    for a in annotations:
        if box_derivable(a.geometry) and a.subject not in names:
            names.append(a.subject)
    return {n: i + 1 for i, n in enumerate(names)}


def records_from_annotation(gt, preds, *, width: int, height: int, force_segm: bool = False,
                             name_id: dict[str, int] | None = None):
    """Name-based :class:`Annotation` GT + predictions -> (iou_type, COCO per-image record).

    ``gt`` / ``preds`` are ``Annotation`` lists (a prediction carries a ``score``). The COCO
    ``category_id`` is a 1-indexed id per distinct ``subject`` name, shared by GT and predictions.
    Pass ``name_id`` when scoring more than one image: pycocotools accumulates every per-image
    record into one eval, so a subject must map to the same id in every image; the per-image-local
    default (``name_id`` ``None``) is for a single image. ``force_segm`` makes every box carry a
    rectangular ``segmentation`` so a whole dataset can be scored with ``iou_type='segm'``.

    A geometry-less annotation and a :class:`~tcip_annotation.state.Point` contribute no record and
    no ``name_id`` entry: neither has a box to score.
    """
    from tcip_annotation.state import (
        bbox_of, box_derivable, is_detection, polygonal, prediction_score,
    )

    def _scorable(a) -> bool:
        return box_derivable(a.geometry)

    def _has_poly(anns):
        return any(polygonal(a.geometry) for a in anns)

    use_segm = force_segm or _has_poly(gt) or _has_poly(preds)
    iou_type = "segm" if use_segm else "bbox"

    if name_id is None:  # single-image scoring: a local map cannot disagree with itself
        name_id = subject_category_ids((*gt, *preds))

    def _box_seg(x1, y1, x2, y2):
        return [[float(x1), float(y1), float(x2), float(y1), float(x2), float(y2), float(x1), float(y2)]]

    def _record(a, *, is_pred):
        if not _scorable(a):
            return None
        box = bbox_of(a.geometry)
        corners = (box.x1, box.y1, box.x2, box.y2)
        rec = (detection_record(corners, name_id[a.subject], prediction_score(a)) if is_pred else gt_record(xywh(*corners), name_id[a.subject], a.iscrowd))
        if polygonal(a.geometry):
            rec["segmentation"] = [_poly_flat(ring) for ring in a.geometry.rings]
        elif use_segm:
            rec["segmentation"] = _box_seg(box.x1, box.y1, box.x2, box.y2)
        return rec

    gt_recs = [r for r in (_record(a, is_pred=False) for a in gt) if r is not None]
    dt_recs = [_record(a, is_pred=True) for a in preds if is_detection(a)]
    return iou_type, build_coco_image_record(width, height, gt_recs, dt_recs)


# ====================================================================
# In-house scalar metrics (expansion seam, segm AP already covers
# true instance segmentation once a mask head exists)
# ====================================================================

def classification_metrics(pred_labels: torch.Tensor, targets: torch.Tensor, num_classes: int) -> dict:
    """Accuracy + macro-F1 + per-class precision/recall/f1/support/count_bias, each per-class
    mapping keyed by the class index as a string (a JSON object's key).

    ``count_bias[c] = (predicted count - true count) / true count``: the phenotype is the
    positive-state fraction, so a class the classifier over-predicts inflates the fraction even at
    high accuracy.
    """
    pred = pred_labels.detach().cpu().long()
    gt = targets.detach().cpu().long()
    if gt.numel() == 0:
        return {"accuracy": 0.0, "f1": 0.0, "per_class": {}, "count_bias": {}}
    accuracy = (pred == gt).float().mean().item()
    per_class: dict[str, dict] = {}
    f1s = []
    for c in range(num_classes):
        tp = int(((pred == c) & (gt == c)).sum())
        fp = int(((pred == c) & (gt != c)).sum())
        fn = int(((pred != c) & (gt == c)).sum())
        support = int((gt == c).sum())
        pred_count = int((pred == c).sum())
        prf = precision_recall_f1(tp, fp, fn)
        f1s.append(prf["f1"])
        per_class[str(c)] = {**prf, "support": support,
                        "count_bias": (pred_count - support) / support if support > 0 else 0.0}
    return {
        "accuracy": accuracy,
        "f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "per_class": per_class,
        "count_bias": {c: per_class[c]["count_bias"] for c in per_class},
    }


def quadratic_weighted_kappa(
    pred_ranks: torch.Tensor, gt_ranks: torch.Tensor, num_ranks: int,
) -> float | None:
    """Chance-corrected ordinal agreement over ``num_ranks``, the run's own rank count: squared
    rank-distance weights, expected agreement from the scored set's own observed rank marginals,
    the ordinal counterpart to :func:`tcip_mcp.pipelines.operating_point._classification_kappa`.
    ``None`` when undefined: no items, or expected disagreement is zero (every populated
    true/predicted pair shares one rank).
    """
    pred = pred_ranks.detach().cpu().round().long()
    gt = gt_ranks.detach().cpu().round().long()
    n = gt.numel()
    if n == 0:
        return None
    k = num_ranks
    if k < 2:
        return None
    observed = torch.zeros((k, k))
    for t, p in zip(gt.tolist(), pred.tolist()):
        observed[t][p] += 1
    row_marginal = observed.sum(dim=1)
    col_marginal = observed.sum(dim=0)
    expected = torch.outer(row_marginal, col_marginal) / n
    idx = torch.arange(k, dtype=torch.float32)
    weights = (idx.unsqueeze(1) - idx.unsqueeze(0)) ** 2
    expected_disagreement = (weights * expected).sum().item()
    if expected_disagreement == 0.0:
        return None
    observed_disagreement = (weights * observed).sum().item()
    return 1.0 - observed_disagreement / expected_disagreement


def r_squared(pred_values: torch.Tensor, gt_values: torch.Tensor) -> float | None:
    """Fraction of variance explained beyond trivially predicting the scored set's own mean, the
    regression counterpart to :func:`quadratic_weighted_kappa`'s chance-correction (both express
    "how much better than the trivial baseline achievable from this set's own distribution").
    ``None`` when undefined: no items, or the set's own values are constant (no variance to
    explain).
    """
    pred = pred_values.detach().cpu().float()
    gt = gt_values.detach().cpu().float()
    if gt.numel() == 0:
        return None
    ss_tot = ((gt - gt.mean()) ** 2).sum().item()
    if ss_tot == 0.0:
        return None
    ss_res = ((gt - pred) ** 2).sum().item()
    return 1.0 - ss_res / ss_tot


def concordance_correlation_coefficient(pred_values: torch.Tensor, gt_values: torch.Tensor) -> float | None:
    """Lin's concordance correlation coefficient: agreement between ``pred_values`` and
    ``gt_values`` as precision (Pearson correlation) times an accuracy/bias penalty. A prediction
    perfectly correlated with GT but systematically offset (a constant bias, or a scale != 1)
    scores low here. ``None`` when undefined: no items, or either series has zero variance.

    ``CCC = 2*r*sigma_pred*sigma_gt / (sigma_pred^2 + sigma_gt^2 + (mean_pred - mean_gt)^2)``, with
    ``r`` the Pearson correlation and ``sigma`` the population (not sample) standard deviation, the
    convention :func:`r_squared` uses.
    """
    pred = pred_values.detach().cpu().float()
    gt = gt_values.detach().cpu().float()
    n = gt.numel()
    if n == 0:
        return None
    pred_mean, gt_mean = pred.mean(), gt.mean()
    pred_var = ((pred - pred_mean) ** 2).mean().item()
    gt_var = ((gt - gt_mean) ** 2).mean().item()
    if pred_var == 0.0 or gt_var == 0.0:
        return None
    covariance = ((pred - pred_mean) * (gt - gt_mean)).mean().item()
    return (2.0 * covariance) / (pred_var + gt_var + (pred_mean.item() - gt_mean.item()) ** 2)


def ordinal_metrics(pred_ranks: torch.Tensor, gt_ranks: torch.Tensor, num_ranks: int) -> dict:
    """Ordinal metrics over ``num_ranks``, the run's own rank count, which the scored half is
    never asked for: it may not reach every rank."""
    pred = pred_ranks.detach().cpu().float()
    gt = gt_ranks.detach().cpu().float()
    if gt.numel() == 0:
        return {"mae": 0.0, "rank_acc": 0.0, "quadratic_weighted_kappa": None}
    return {
        "mae": (pred - gt).abs().mean().item(),
        "rank_acc": (pred.round() == gt.round()).float().mean().item(),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(pred_ranks, gt_ranks, num_ranks),
    }


def regression_metrics(pred_values: torch.Tensor, gt_values: torch.Tensor) -> dict:
    pred = pred_values.detach().cpu().float()
    gt = gt_values.detach().cpu().float()
    if gt.numel() == 0:
        return {"mae": 0.0, "rmse": 0.0, "r_squared": None}
    return {
        "mae": (pred - gt).abs().mean().item(),
        "rmse": ((pred - gt) ** 2).mean().sqrt().item(),
        "r_squared": r_squared(pred_values, gt_values),
    }


def semantic_seg_metrics(preds: torch.Tensor, targets: torch.Tensor, num_classes: int,
                         ignore_index: int | None = None) -> dict:
    """Standard mean-IoU / Dice for semantic segmentation from per-pixel class maps.

    ``preds`` and ``targets`` are integer class-id tensors of matching shape (any shape;
    both are flattened). Per class: IoU = ``|P∩G| / |P∪G|`` and Dice = ``2|P∩G| / (|P|+|G|)``.
    ``mIoU`` / ``dice`` average only over classes present in preds or targets, a class absent
    from both has an undefined ratio (reported ``None`` per-class, excluded from the mean), the
    standard convention. ``pixel_acc`` is the fraction of correctly labeled pixels. Pixels equal
    to ``ignore_index`` in the GT are dropped before scoring.
    """
    pred = preds.detach().cpu().reshape(-1).long()
    gt = targets.detach().cpu().reshape(-1).long()
    if ignore_index is not None:
        keep = gt != ignore_index
        pred, gt = pred[keep], gt[keep]
    per_class_iou: dict[int, float | None] = {}
    per_class_dice: dict[int, float | None] = {}
    ious, dices = [], []
    for c in range(num_classes):
        if c == ignore_index:
            continue
        p = pred == c
        g = gt == c
        inter = int((p & g).sum())
        union = int((p | g).sum())
        denom = int(p.sum()) + int(g.sum())
        if union == 0:  # class absent from both, ratio undefined, exclude from the mean
            per_class_iou[c] = None
            per_class_dice[c] = None
            continue
        iou = inter / union
        dice = 2 * inter / denom if denom > 0 else 0.0
        per_class_iou[c] = iou
        per_class_dice[c] = dice
        ious.append(iou)
        dices.append(dice)
    pixel_acc = float((pred == gt).float().mean()) if gt.numel() else 0.0
    return {
        "mIoU": sum(ious) / len(ious) if ious else 0.0,
        "dice": sum(dices) / len(dices) if dices else 0.0,
        "pixel_acc": pixel_acc,
        "per_class_iou": per_class_iou,
        "per_class_dice": per_class_dice,
    }


# ====================================================================
# Task-agnostic evaluate(), loss pass + prediction pass
# ====================================================================

def effective_iou_type(task: str, iou_type: str | None) -> str:
    """Resolve the COCOeval ``iouType`` actually used to score ``task``: an explicit ``iou_type``
    wins; otherwise ``segm`` for instance_seg, ``bbox`` for detection, ``""`` for non-COCO tasks.
    """
    if iou_type:
        return iou_type
    if task == "instance_seg":
        return "segm"
    return "bbox" if task == "detection" else ""


@torch.no_grad()
def evaluate(
    model, loader, device, task: str, *, dims: Mapping[str, int],
    conf_threshold: float = 0.25, iou_threshold: float = 0.5,
    iou_type: str | None = None, max_dets: int = 100, score_weights: dict | None = None,
    trait: TraitEntry | None = None,
) -> dict:
    """Compute per-task validation/test metrics. Returns bare metric keys.

    ``dims`` is what the model was built at (:func:`~tcip_mcp.pipelines.model_build.model_dims`);
    a class or rank count is read from it, never off the half being scored.

    ``trait``: the trait's confirmed entry; when set, a count trait's derived localization
        criterion (traits.py, e.g. a
        center-match at half the class-average size) governs the reported detection count and the
        f1 the selection composite optimizes; map50 stays a labeled comparability metric. Absent ->
        the IoU@``iou_threshold`` convention governs.
    """
    is_detection = task in ("detection", "instance_seg")
    is_instance_seg = task == "instance_seg"
    eff_iou_type = effective_iou_type(task, iou_type)

    model.eval()
    total_loss = 0.0
    n_loss = 0
    per_image: list[dict] = []
    cls_p, cls_g, ord_p, ord_g, reg_p, reg_g = [], [], [], [], [], []
    seg_p, seg_g = [], []
    detector = getattr(model, "detector", None)

    for batch in loader:
        if is_detection:
            images, targets = batch
            images = [img.to(device) for img in images]
            targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
            # Loss pass over the full batch, all-negative (empty-box) images contribute their
            # background/objectness loss, so val_loss penalizes false positives on empty frames and
            # matches the train loop's distribution. BN stays eval via the train()+BN.eval() trick.
            # The heads get the train loop's own targets; scoring below reads every row.
            from tcip_mcp.pipelines.data.datasets import instance_targets

            head_targets = instance_targets(targets)
            if detector is not None:
                detector.train()
                for m in detector.modules():
                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                        m.eval()
                ld = detector(images, head_targets)
                total_loss += float(sum(ld.values()).item())
                n_loss += 1
            else:
                model.training = True
                for head in getattr(model, "heads", []):
                    head.training = True
                ld = model(images, head_targets)
                # sum() over an untyped model's loss dict resolves, by mypy's overload matching on
                # Any, to its int overload rather than the runtime Tensor the values actually are.
                total_loss += (
                    float(cast(Any, sum(ld.values())).item())
                    if isinstance(ld, dict) else float(ld)
                )
                n_loss += 1
            # Prediction pass.
            model.eval()
            outputs = model(images)
            for img, t, out in zip(images, targets, outputs):
                h, w = int(img.shape[-2]), int(img.shape[-1])
                per_image.append(records_from_detector(
                    t, out, width=w, height=h, include_masks=is_instance_seg))
        else:
            images, targets = batch
            images = images.to(device)
            targets = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in targets.items()}
            # Loss pass (BN stays in eval via the top-level training flag trick).
            model.training = True
            ld = model(images, targets)
            # sum() over an untyped model's loss dict resolves, by mypy's overload matching on
            # Any, to its int overload rather than the runtime Tensor the values actually are.
            total_loss += (
                float(cast(Any, sum(ld.values())).item())
                if isinstance(ld, dict) else float(ld)
            )
            n_loss += 1
            # Prediction pass.
            model.eval()
            out = model(images)
            if task == "classification" and "head0_labels" in out:
                cls_p.append(out["head0_labels"].detach().cpu())
                cls_g.append(targets["labels"].detach().cpu())
            elif task == "ordinal" and "head0_ranks" in out:
                ord_p.append(out["head0_ranks"].detach().cpu())
                ord_g.append(targets["ranks"].detach().cpu())
            elif task == "regression" and "head0_values" in out:
                reg_p.append(out["head0_values"].detach().cpu())
                reg_g.append(targets["values"].detach().cpu())
            elif task == "semantic_seg" and "head0_masks" in out:
                pm = out["head0_masks"].detach().cpu()
                gm = targets["masks"].detach().cpu()
                # decode() argmaxes feature-resolution logits; upsample the label map (nearest)
                # to the GT frame so metrics compare per-pixel at the annotation resolution.
                if pm.shape[-2:] != gm.shape[-2:]:
                    pm = torch.nn.functional.interpolate(
                        pm.unsqueeze(1).float(), size=gm.shape[-2:], mode="nearest").squeeze(1).long()
                seg_p.append(pm.reshape(-1))
                seg_g.append(gm.reshape(-1))

    model.eval()
    loss = total_loss / max(n_loss, 1)
    result: dict = stored_number("loss", _rounded(loss))

    if is_detection:
        m = coco_detection_metrics(per_image, iou_type=eff_iou_type, iou_threshold=iou_threshold,
                                   conf_threshold=conf_threshold, max_dets=max_dets)
        result.update({
            "precision": round(m["precision"], 6), "recall": round(m["recall"], 6),
            "f1": round(m["f1"], 6), "map50": round(m["map50"], 6), "map": round(m["map"], 6),
            "map_at_maxdets": round(m["map_at_maxdets"], 6),
            "map50_at_maxdets": round(m["map50_at_maxdets"], 6),
        })
        # A count trait's derived criterion governs the reported count + the selection f1;
        # map50 stays a labeled comparability metric. Without a trait the IoU convention governs.
        if trait is not None:
            criterion = resolve_match_criterion(trait, per_image)
            gc = governing_counts(per_image, criterion, conf_threshold=conf_threshold)
            result.update({
                "precision": gc["precision"], "recall": gc["recall"], "f1": gc["f1"],
                "governing_criterion": criterion, "map50_role": "comparability_only",
                "iou_precision": round(m["precision"], 6), "iou_recall": round(m["recall"], 6),
                "iou_f1": round(m["f1"], 6),
            })
        governing_f1 = result["f1"]
        result.update(stored_number(
            "objective",
            _rounded(compute_composite_objective(loss, governing_f1, m["map50"], score_weights)),
        ))
    elif task == "classification" and cls_p:
        result.update(_reported_metrics(classification_metrics(
            torch.cat(cls_p), torch.cat(cls_g), dims["num_classes"])))
    elif task == "ordinal" and ord_p:
        result.update(_reported_metrics(ordinal_metrics(
            torch.cat(ord_p), torch.cat(ord_g), dims["num_ranks"])))
    elif task == "regression" and reg_p:
        result.update(_reported_metrics(regression_metrics(torch.cat(reg_p), torch.cat(reg_g))))
    elif task == "semantic_seg" and seg_p:
        result.update(_reported_metrics(semantic_seg_metrics(
            torch.cat(seg_p), torch.cat(seg_g), dims["num_classes"])))

    return result
