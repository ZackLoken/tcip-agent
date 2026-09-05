"""Task-aware evaluation metrics + composite selection objective.

Single home for:
  * the pycocotools-backed detection / instance_seg metrics (mAP + operating-point
    TP/FP/FN), shared by training ``_validate``, ``eval_runners.run_test_evaluation`` and
    the agent's own ``score_predictions`` function (no GUI route calls it), one source of
    truth, the canonical COCO mAP definition;
  * in-house scalar metrics for classification / ordinal / regression (the seam
    where pycocotools ``iou_type='segm'`` can later cover true instance seg);
  * the composite selection objective (lower = better);
  * a task-agnostic two-pass ``evaluate()``, the metrics pass ``eval_runners.py``'s
    checkpoint-evaluation orchestration calls into.

pycocotools is imported lazily inside the COCO functions. Every pycocotools call
is wrapped in ``redirect_stdout`` because ``createIndex``/``loadRes``/``summarize``
print to stdout, which would corrupt the MCP stdio transport.
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import torch

from tcip_store import stored_number, stored_numbers

# Every box handed to pycocotools goes through this, both sides of a match on the one stored grid.
from tcip_annotation.json_io import xywh

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
scalar ``evaluate()`` (or ``governing_counts``) actually returns across the tasks it scores. A
metric's value alone never says which way is better, so a ranking that guessed from the key's
spelling could promote a worse model under an unfamiliar name. A raw count (``tp``/``fp``/``fn``)
or a signed bias has no such direction and is left out rather than assigned an arbitrary one; a
non-finite value's state companion (``tcip_store.values.NOT_FINITE_SUFFIX``) is excluded by that
suffix rule, not listed here.

``map75`` and the operating-point curve's ``abs_count_error_mean``/``count_error_p90``/
``count_bias_std`` are left out on purpose, not merely unnoticed: ``coco_detection_metrics``
computes ``map75`` internally but ``evaluate()`` never surfaces it, and the three curve
statistics come only from ``_count_stats_at_conf`` inside ``derive_operating_point_curve``'s
calibration path, never from ``evaluate()``/``governing_counts``. None of the four ever reaches
a checkpoint's ``metrics`` dict or a registry entry, so nothing here needs to rank them yet.
``count_bias_mean`` is signed (over- and under-counting are both present in the same value) and
has no direction to declare at all."""


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
# Composite selection objective (ported verbatim from chestnut-burr)
# ====================================================================

def compute_composite_objective(
    val_loss: float, f1: float, map50: float, score_weights: dict | None = None
) -> float:
    """Lower-is-better selection/tuning score blending loss, F1 and mAP50.

    ``w["loss"]*loss + w["f1"]*(1-f1)*10 + w["map50"]*(1-map50)*10`` with ``1e6``
    sentinels for degenerate runs. The ``*10`` lifts the unit-interval quality
    terms to a typical loss magnitude.
    """
    w = score_weights or DEFAULT_SCORE_WEIGHTS
    vl = float(val_loss) if (val_loss is not None and math.isfinite(val_loss)) else float("inf")
    f1v = float(f1) if (f1 is not None and math.isfinite(f1)) else 0.0
    m50 = float(map50) if (map50 is not None and math.isfinite(map50)) else 0.0
    if vl <= 0:
        return 1e6
    if f1v < 0.01 and m50 < 0.01:
        return 1e6
    return w["loss"] * vl + w["f1"] * (1.0 - f1v) * 10 + w["map50"] * (1.0 - m50) * 10


# ====================================================================
# pycocotools detection / instance_seg metrics
# ====================================================================

def build_coco_image_record(width: int, height: int, gt: list[dict], dt: list[dict],
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
    Short-circuits to all-zero metrics (no exception) for empty predictions
    (``loadRes([])`` raises ``IndexError``), empty GT (COCOeval ``stats == -1``),
    or a fully empty set.
    """
    images, annotations, results = [], [], []
    cat_ids: set[int] = set()
    ann_id = 1
    n_gt = n_pred = 0
    for img_id, rec in enumerate(per_image, start=1):
        images.append({"id": img_id, "width": int(rec.get("width", 0)), "height": int(rec.get("height", 0))})
        for ann in rec.get("gt", []):
            a = dict(ann)
            a["id"] = ann_id
            a["image_id"] = img_id
            a.setdefault("iscrowd", 0)
            if "area" not in a:
                bb = a["bbox"]
                a["area"] = float(bb[2] * bb[3])
            annotations.append(a)
            cat_ids.add(int(a["category_id"]))
            ann_id += 1
            n_gt += 1
        for res in rec.get("dt", []):
            r = dict(res)
            r["image_id"] = img_id
            results.append(r)
            cat_ids.add(int(r["category_id"]))
            n_pred += 1

    base = {
        "map": 0.0, "map50": 0.0, "map75": 0.0,
        "map_at_maxdets": 0.0, "map50_at_maxdets": 0.0,
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "tp": 0, "fp": 0, "fn": n_gt,
        "n_images": len(per_image), "n_gt": n_gt, "n_pred": n_pred,
        "per_image_counts": [
            {"image_id": i + 1, "tp": 0, "fp": 0, "fn": len(rec.get("gt", []))}
            for i, rec in enumerate(per_image)
        ],
        "iou_type": iou_type, "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold, "max_dets": max_dets,
    }
    if n_pred == 0 or n_gt == 0:
        return base

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    categories = [{"id": c, "name": str(c)} for c in sorted(cat_ids)]
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        coco_gt = COCO()
        coco_gt.dataset = {"images": images, "annotations": annotations, "categories": categories}
        coco_gt.createIndex()
        try:
            coco_dt = coco_gt.loadRes(results)
        except IndexError:
            return base
        coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        # Include 100 so map/map50/map75 stay the standard, cap-comparable AP (the old [1,10,max_dets]
        # made summarize() report AP=0.0 for any non-100 cap); max_dets adds the operating-cap figures.
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
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "map": max(m_ap, 0.0), "map50": max(m_ap50, 0.0), "map75": max(m_ap75, 0.0),
        "map_at_maxdets": max(m_ap_md, 0.0), "map50_at_maxdets": max(m_ap50_md, 0.0),
        # map/map50/map75 are standard COCO AP@100 as of this marker; older stored metrics
        # (no marker) used maxDets=max_dets and are not numerically comparable.
        "map_convention": "coco_ap100",
        "precision": precision, "recall": recall, "f1": f1,
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

def _centers_xywh(anns: list[dict]) -> list[tuple[float, float]]:
    return [(a["bbox"][0] + a["bbox"][2] / 2.0, a["bbox"][1] + a["bbox"][3] / 2.0) for a in anns]


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
        for rec in per_image for a in rec.get("gt", [])
        if class_id is None or a["category_id"] == class_id
    ]
    return float(np.mean(sizes)) if sizes else 0.0


def mean_of_present_counts(counts: Iterable[int]) -> float:
    """Mean of the entries in ``counts`` that are actually positive (> 0).

    The shared "typical, when present" statistic behind a relative count-bias tolerance's derived
    denominator: a 0 is not evidence of what a typical image carrying the
    thing being counted looks like, so including it would dilute the density figure toward zero for
    anything present on only some of the population, exactly the dilution already rejected for the
    equivalence test's own standard error (``n_present`` there). Shared by
    :func:`gt_class_typical_count` (GT per-image records) and
    :func:`operating_point.resolve_classifier_operating_point` (flat classified instances grouped by
    image) so both derive "typical count" the same way rather than each inventing its own.
    """
    present = [c for c in counts if c > 0]
    return float(np.mean(present)) if present else 0.0


def gt_class_typical_count(per_image: list[dict], class_id: int | None = None) -> float:
    """Mean per-image GT count for ``class_id`` (all classes pooled when ``None``), the derived
    denominator a relative count-bias tolerance scales against
    (:func:`operating_point._bias_equivalence_ok`).

    Deliberately GT-only and conf-independent, a distinct notion of "present" from
    ``_count_stats_at_conf``'s ``n_present`` (``gt or dt``, at one conf): a class with detections but
    no real GT anywhere has no genuine "typical count" to speak of (it should derive 0, not borrow
    density from its own false positives), and the relative tolerance must not shift as the sweep
    moves through conf values just because a different set of low-score detections happens to survive.
    """
    counts = [
        sum(1 for a in rec.get("gt", []) if class_id is None or a["category_id"] == class_id)
        for rec in per_image
    ]
    return mean_of_present_counts(counts)


def center_match_pairs(gt_centers: list[tuple[float, float]], dt_centers: list[tuple[float, float]],
                       tolerance: float, *, policy: str) -> list[tuple[int, int]]:
    """The one greedy nearest-center 1:1 matcher behind both the count and the classifier
    calibration's identity pairing, two stated policies rather than two implementations that
    could silently drift apart.

    Inputs are plain ``(x, y)`` centres, already reduced from whatever box shape the caller holds
    (an xywh detection box for the count, an ``Annotation``'s box for the calibration pairing);
    this primitive knows nothing about either representation. Distance is Euclidean, the
    tolerance inclusive (``d <= tolerance`` matches, so a pair exactly at the boundary counts).
    Returns ``(gt_index, dt_index)`` pairs, indices into the two input lists.

    ``policy="score_first"`` walks ``dt_centers`` in the order given (a caller passing detections
    score-descending resolves a duplicate claim on one ground truth by keeping the
    higher-confidence detection); among equidistant unused ground truths the last index wins, the
    count's existing semantics, unchanged by this primitive's introduction. A detection with no
    recorded score cannot be placed in that order at all, so a caller using this policy refuses
    such a record before it ever reaches here, never passing a stand-in score in its place.

    ``policy="distance_first"`` sorts every (gt, dt) pair within tolerance by distance ascending
    and claims the closest first, ties broken by ``(gt index, dt index)`` ascending, the order a
    plain ascending sort of the ``(distance, gt_index, dt_index)`` tuples already gives. This is
    the identity policy: acceptance drops a prediction's score once it is confirmed, so a partly
    reviewed bucket holds records with no place in a score order, and geometry alone is the
    evidence for which ground truth a prediction identifies.

    Neither policy deduplicates the false-positive count for a caller: an entry in ``dt_centers``
    that claims no pair is a false positive, so ``fp = len(dt_centers) - len(pairs)`` counts every
    detection that never claimed a ground truth, duplicates included, against the raw detection
    count.
    """
    if policy == "score_first":
        used = [False] * len(gt_centers)
        pairs: list[tuple[int, int]] = []
        for di, (dx, dy) in enumerate(dt_centers):
            best_gi, best_d = -1, tolerance
            for gi, (gx, gy) in enumerate(gt_centers):
                if used[gi]:
                    continue
                d = ((dx - gx) ** 2 + (dy - gy) ** 2) ** 0.5
                if d <= best_d:
                    best_d, best_gi = d, gi
            if best_gi >= 0:
                used[best_gi] = True
                pairs.append((best_gi, di))
        return pairs

    if policy == "distance_first":
        candidates: list[tuple[float, int, int]] = []
        for gi, (gx, gy) in enumerate(gt_centers):
            for di, (dx, dy) in enumerate(dt_centers):
                d = ((dx - gx) ** 2 + (dy - gy) ** 2) ** 0.5
                if d <= tolerance:
                    candidates.append((d, gi, di))
        candidates.sort()
        matched_gt: set[int] = set()
        matched_dt: set[int] = set()
        pairs = []
        for _, gi, di in candidates:
            if gi in matched_gt or di in matched_dt:
                continue
            matched_gt.add(gi)
            matched_dt.add(di)
            pairs.append((gi, di))
        return pairs

    raise ValueError(f"center_match_pairs: unknown policy {policy!r}, expected "
                     "'score_first' or 'distance_first'")


def _center_match_image(gt: list[dict], dt: list[dict], tolerance: float) -> tuple[int, int, int]:
    """tp/fp/fn under the count's score-first policy (``dt`` pre-sorted by score descending)."""
    gt_centers = _centers_xywh(gt)
    dt_centers = _centers_xywh(dt)
    # The count's identity question: a duplicate claim on one ground truth is resolved by keeping
    # the higher-confidence detection, never by geometry alone.
    tp = len(center_match_pairs(gt_centers, dt_centers, tolerance, policy="score_first"))
    return tp, len(dt) - tp, len(gt_centers) - tp


def resolve_match_criterion(trait_name: str | None, per_image: list[dict], *,
                            class_id: int | None = None, iou_threshold: float = 0.5) -> dict:
    """The one localization criterion that governs a trait's phenotype count + model selection.

    Reads the trait's recorded ``localization`` kind (center_match vs iou_match, traits.py) and
    derives its per-dataset tolerance from the GT in hand, never a pinned value. Returns
    ``{kind, tolerance | iou_threshold, derived_from, trait}``. With no trait (or an iou_match trait),
    it is IoU matching at ``iou_threshold``, the labeled comparability convention (AP@0.5), which
    governs nothing on its own; a count trait's derived center-match tolerance is what the phenotype
    and checkpoint selection rest on.

    ``localization`` is derived once, the first time real GT is available for a trait with no
    recorded kind (via ``derivations.derive_localization_kind``), persisted through
    ``traits.write_trait_spec_fields`` and recorded in the platform audit log naming the trait,
    the field, the value and the derivation basis, and read from the recorded value on every
    later call.
    A recorded kind is also
    cheaply re-checked against what the current data would derive, every real call, divergence
    surfaces a warning (``kind_diverged`` in the returned dict) rather than silently switching,
    per the standing "constrain by observation, not permission" rule; only an explicit re-derive
    changes the recorded value. This is also the single point every consumer of a trait's
    localization criterion goes through, ``generic_trainer.py`` reads the recorded field directly
    (it runs before any GT loads, so it cannot call this), but every site with real GT in hand
    (phenology_tools.py's classifier-calibration matching, this module's own count/selection
    metrics) calls this function rather than re-deriving or re-reading the field independently.
    """
    if not trait_name:
        return {"kind": "iou_match", "iou_threshold": float(iou_threshold),
                "derived_from": "comparability convention (AP@0.5)", "trait": None}
    from tcip_mcp.pipelines.derivations import (
        derive_iou_match_threshold, derive_localization_kind, derive_localization_tolerance_frac,
    )
    from tcip_mcp.traits import CENTER_MATCH, get_trait

    spec = get_trait(trait_name)
    boxes_per_image = [[a["bbox"] for a in rec.get("gt", [])
                        if class_id is None or a["category_id"] == class_id]
                       for rec in per_image]

    kind = spec.localization
    kind_source = "recorded"
    kind_diverged = False
    live_derived_kind = derive_localization_kind(boxes_per_image)
    if kind:
        if live_derived_kind is not None and live_derived_kind != kind:
            kind_diverged = True
            logger.warning(
                "trait %r: recorded localization kind %r diverges from what this call's own GT "
                "would derive (%r), not switched (observation, not permission); re-derive "
                "explicitly via revise_trait_spec if this data is now representative.",
                trait_name, kind, live_derived_kind)
    elif live_derived_kind is not None:
        kind = live_derived_kind
        kind_source = "data_derived_at_runtime"
        # Stamp via resolution.derived(), not aliased on import, so test_provenance_honesty.py's
        # AST scanner (which matches the literal call name "derived") actually sees this label.
        from tcip_mcp.pipelines.resolution import derived
        derived("localization_kind", kind,
               derived_from="achievable IoU under annotation jitter (GT characteristic size)")
        basis = (f"derived from {sum(len(b) for b in boxes_per_image)} GT boxes "
                 "(achievable IoU under jitter)")
        try:
            from tcip_mcp import traits as traits_module
            traits_module.write_trait_spec_fields(trait_name, {"localization": kind})
        except ValueError:
            logger.warning("could not persist derived localization kind for %r", trait_name, exc_info=True)
        else:
            from tcip_mcp.audit import record_event_or_raise
            record_event_or_raise("trait_spec_field_derived",
                                  {"trait": trait_name, "field": "localization", "value": kind,
                                   "basis": basis})
    else:
        raise ValueError(
            f"trait {trait_name!r} has no recorded localization kind and no GT in this call to "
            "derive one from, cannot resolve a match criterion. Calibrate or evaluate against a "
            "labeled reference at least once before this trait's localization kind can be known.")

    # Stamp via resolution.derived()/default(), not aliased on import, and with the derived_from
    # literal inlined directly into the call: passing it as a variable, even unaliased, is also
    # invisible to the AST scanner, it only reads a literal string or an
    # f-string's leading constant written directly at the call site, never a name reference.
    # derived() for a real per-dataset computation, default() for the honest "underivable, fell
    # back" case (never claimed as a derivation, and not scanned by test_provenance_honesty.py at
    # all, correctly, since it makes no derivation claim to check). The label text lives once, in
    # the call itself; `.derived_from` reads it back rather than a second, separately-typed copy.
    from tcip_mcp.pipelines.resolution import default, derived

    if kind == CENTER_MATCH:
        frac = derive_localization_tolerance_frac(boxes_per_image)
        if frac is not None:
            frac_source = derived("localization_tolerance_frac", frac,
                                  derived_from="GT nearest-neighbor spacing (p10 + margin)").derived_from
        else:
            frac = spec.localization_tolerance_frac
            frac_source = default(
                "localization_tolerance_frac", frac,
                derived_from=f"trait default (underivable: no same-class neighbor in this GT), "
                             f"{spec.localization_tolerance}",
            ).derived_from
        result = {"kind": "center_match",
                  "tolerance": float(frac * gt_class_avg_size(per_image, class_id=class_id)),
                  "derived_from": frac_source, "trait": trait_name,
                  "kind_source": kind_source, "kind_diverged": kind_diverged}
        return result
    derived_threshold = derive_iou_match_threshold(boxes_per_image)
    if derived_threshold is not None:
        threshold_source = derived(
            "iou_threshold", derived_threshold,
            derived_from="achievable IoU under annotation jitter, minus margin (GT characteristic size)",
        ).derived_from
    else:
        derived_threshold = iou_threshold
        threshold_source = f"caller/default (underivable: no valid GT boxes), trait localization={kind}"
        default("iou_threshold", derived_threshold, derived_from=threshold_source)
    result = {"kind": "iou_match", "iou_threshold": float(derived_threshold),
              "derived_from": threshold_source, "trait": trait_name,
              "kind_source": kind_source, "kind_diverged": kind_diverged}
    return result


def _dt_score(d: dict) -> float:
    """The one accessor every governing-count reader takes a detection record's confidence through.

    No default: a detection record with no ``score``, or a ``score`` of ``None``, cannot be
    ordered or thresholded by confidence, so this refuses by name rather than letting
    ``governing_counts``, ``_count_stats_at_conf`` and the calibration curve's score grid each
    risk a different silent stand-in for a field that measures the model's own certainty, the
    same no-default rule the classifier calibration path already applies to its own records.
    """
    if "score" not in d:
        raise ValueError(f"detection record has no 'score' field, cannot count it: {d!r}")
    if d["score"] is None:
        raise ValueError(f"detection record's 'score' is None, cannot count it: {d!r}")
    return float(d["score"])


def governing_counts(per_image: list[dict], criterion: dict, *, conf_threshold: float,
                     class_id: int | None = None, max_dets: int = 1000) -> dict:
    """tp/fp/fn/precision/recall/f1 at the criterion that governs the phenotype count.

    ``center_match`` uses greedy nearest-center matching at the derived tolerance; ``iou_match`` reuses
    the COCO IoU matcher. This count is what a count-trait phenotype and model selection rest on,
    distinct from AP@0.5, which stays a labeled comparability metric that governs nothing.
    """
    if criterion["kind"] == "center_match":
        # Deliberately uncapped by max_dets, unlike the iou_match branch below: a count
        # trait's total is every conf-surviving detection, not the COCOeval detection-cap
        # convention that AP@0.5 comparability uses.
        tol = float(criterion["tolerance"])
        tp = fp = fn = 0
        for rec in per_image:
            gt = [a for a in rec.get("gt", []) if class_id is None or a["category_id"] == class_id]
            dt = sorted(
                (d for d in rec.get("dt", [])
                 if _dt_score(d) >= conf_threshold
                 and (class_id is None or d["category_id"] == class_id)),
                key=lambda d: -_dt_score(d),
            )
            t, f, n = _center_match_image(gt, dt, tol)
            tp += t
            fp += f
            fn += n
    else:
        m = coco_detection_metrics(per_image, iou_threshold=criterion["iou_threshold"],
                                   conf_threshold=conf_threshold, max_dets=max_dets)
        tp, fp, fn = int(m["tp"]), int(m["fp"]), int(m["fn"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 6),
            "recall": round(recall, 6), "f1": round(f1, 6), "criterion": criterion}


def _count_stats_at_conf(per_image: list[dict], *, tolerance: float, conf: float,
                         class_id: int | None) -> dict:
    """Center-match counting statistics over ``per_image`` at one conf, optionally for one class.

    The single implementation of "match, count, and take the per-image count bias", the class-pooled
    curve entry and every per-class entry beside it both come from here, so a per-class bias can
    never be measured by a second matcher that drifts from the pooled one.

    Two scopes of the same per-image bias travel side by side. The whole-reference statistics
    (``count_bias_mean``/``count_bias_std`` over ``n_images``) are what a conf picker compares across
    the grid: ``n_images`` is the same at every conf, so ``count_bias_mean`` is the reference's total
    signed miscount up to one fixed constant and minimizing it minimizes that total. The
    present-scoped statistics (``count_bias_mean_present``/``count_bias_std_present``
    over ``n_present``) are what an equivalence gate compares against a relative tolerance: an image
    with no GT and no surviving detection contributes a certain zero and says nothing about how far
    off the count is on an image that carries the thing being counted. Including it divides the
    measured mean bias by exactly ``n_images / n_present`` and counts those empty images in the
    equivalence test's own sample size, while the tolerance the result is compared against is scaled
    by a density measured over present images only (:func:`mean_of_present_counts`). The two sides
    then describe different populations, and a systematic miscount on the images that carry
    something reads as that fraction of itself.
    """
    tp = fp = fn = 0
    biases: list[int] = []
    present_biases: list[int] = []
    for rec in per_image:
        gt = [a for a in rec.get("gt", []) if class_id is None or a["category_id"] == class_id]
        dt = sorted(
            (d for d in rec.get("dt", [])
             if _dt_score(d) >= conf and (class_id is None or d["category_id"] == class_id)),
            key=lambda d: -_dt_score(d),
        )
        t, f, n = _center_match_image(gt, dt, tolerance)
        tp += t
        fp += f
        fn += n
        biases.append(f - n)
        if gt or dt:
            present_biases.append(f - n)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    abs_biases = [abs(b) for b in biases]
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "count_bias_mean": float(np.mean(biases)) if biases else 0.0,
        "abs_count_error_mean": float(np.mean(abs_biases)) if biases else 0.0,
        # Tail dispersion, a p90 of |bias|, not another mean, since a mean can hide one
        # badly-off image among many. Reference-sufficiency terms, computed here, once,
        # so the gate never re-derives a second matcher over the same per-image biases.
        "count_error_p90": float(np.quantile(abs_biases, 0.9)) if abs_biases else 0.0,
        "count_bias_std": float(np.std(biases, ddof=1)) if len(biases) > 1 else 0.0,
        "n_images": len(biases),
        # Images that actually carried this class (gt or a surviving dt), distinct from n_images
        # (the whole holdout) because a scope scarce in the reference gets the denominator of both
        # its bias and its standard error from how much evidence there really is, not diluted by
        # images that say nothing about it.
        "n_present": len(present_biases),
        "count_bias_mean_present": float(np.mean(present_biases)) if present_biases else 0.0,
        "count_bias_std_present": (float(np.std(present_biases, ddof=1))
                                   if len(present_biases) > 1 else 0.0),
    }


def _class_ids_present(per_image: list[dict], class_id: int | None = None) -> list[int]:
    """The class ids to break the sweep down by: those the records carry in gt or dt, derived from
    the data in hand rather than a registry read or a pinned id space.

    An explicit ``class_id`` is returned as-is, the caller has already scoped the sweep to it, and
    it stays the breakdown's one key even on records that turn out to carry none of it. Every
    annotation and detection must carry ``category_id``; a per-class breakdown of records that do
    not identify their classes is not something to guess at.
    """
    if class_id is not None:
        return [class_id]
    ids = {a["category_id"] for rec in per_image for a in rec.get("gt", [])}
    ids |= {d["category_id"] for rec in per_image for d in rec.get("dt", [])}
    return sorted(ids)


def derive_operating_point_curve(per_image: list[dict], *, tolerance: float,
                                 class_id: int | None = None,
                                 conf_grid: list[float] | None = None,
                                 max_thresholds: int = 80) -> dict:
    """Sweep the confidence threshold over ``per_image`` records via center-matching.

    One model pass produces ``per_image`` (unfiltered dt with scores); this sweeps conf cheaply in
    Python, no re-forwarding. For each conf: aggregate TP/FP/FN and per-image count bias (FP-FN).
    Passing an explicit ``conf_grid`` (e.g. a single-element ``[conf]``) skips grid construction and
    evaluates exactly those points, the exact-conf holdout evaluation (no nearest-neighbor snap)
    relies on this. Returns ``{tolerance, class_id, curve:[{conf, tp, fp, fn, precision, recall, f1,
    count_bias_mean, abs_count_error_mean, count_error_p90, count_bias_std, n_images, n_present,
    count_bias_mean_present, count_bias_std_present, per_class}]}``
, the dispersion + reference-sufficiency terms are per-conf statistics across ``per_image`` that
    the operating-point gate reads, never recomputes. See :func:`_count_stats_at_conf` for why the
    bias travels in two scopes and which consumer reads which.

    ``per_class`` carries the same statistics measured within each class the records carry,
    keyed by ``str(category_id)`` (string keys so an in-memory sweep and one round-tripped through
    the JSON sidecar have the same shape). It exists because the pooled entry beside it cannot see a
    per-class error: matching is class-blind there, so a detector that calls every class-A object
    class B reports tp-only, zero pooled bias, while the delivered per-class counts, which are the
    phenotype for a fraction/ratio trait, are both wrong. Class ids come from the records themselves;
    which of them is the trait's positive class is not read here and is not needed to measure bias.
    """
    scores = sorted({_dt_score(d) for rec in per_image for d in rec.get("dt", [])})
    if conf_grid is None:
        if len(scores) > max_thresholds:
            conf_grid = list(np.linspace(scores[0], scores[-1], max_thresholds))
        else:
            conf_grid = list(scores)
        conf_grid = sorted(set([0.0, *conf_grid]))
    class_ids = _class_ids_present(per_image, class_id)
    curve: list[dict] = []
    for conf in conf_grid:
        pooled = _count_stats_at_conf(per_image, tolerance=tolerance, conf=conf, class_id=class_id)
        if len(class_ids) == 1:
            # Filtering to the only class present is a no-op on both gt and dt, so the pooled entry
            # is that class's entry, reused rather than recomputed, which keeps the single-class
            # sweep (every reference the platform builds today) at its original cost.
            per_class = {str(class_ids[0]): pooled}
        else:
            per_class = {str(cid): _count_stats_at_conf(per_image, tolerance=tolerance, conf=conf,
                                                        class_id=cid)
                         for cid in class_ids}
        curve.append({"conf": float(conf), **pooled, "per_class": per_class})
    return {"tolerance": float(tolerance), "class_id": class_id, "curve": curve}


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
    generally not the F1-max point (that optimizes matching, not count agreement).

    Aimed at the worst class rather than the pooled bias so the pick and the gate optimize
    the same thing. With two classes of opposite sign the two objectives coincide, but that does
    not hold with a third class: a conf can buy pooled balance by trading one class's over-count
    against another's under-count and be strictly worse for the worst class than a conf on the
    same curve that the gate would accept
    (see ``test_pick_serves_the_worst_class_not_the_pooled_total``). Picking pooled there refuses a
    model that has a valid operating point, and tells the breeder to fix a model that is not broken.
    On a single-class reference the two objectives are the same number, so this changes no operating
    point the platform picks today.

    The final ``-c["conf"]`` tie-break is a completion of the existing
    tie-break, not a new selection objective: when |bias| and F1 and |abs error| are all exactly
    tied across several confs, which happens on a reference filtered to a floor, since nothing
    below the floor is visible to distinguish them, defaulting to the lowest tied conf (e.g. the
    grid's seeded 0.0) would be generically the worst of the tied candidates in practice
    (it admits the most low-confidence noise for no better count agreement) and, combined with the
    conf-censoring guard, could make a genuinely trustworthy pick read as censored merely because the
    tie resolved to the search floor. Preferring the highest tied conf breaks ties toward the most
    conservative, best-supported candidate among equals, it does not change what is optimized.
    """
    curve = sweep.get("curve") or []
    if not curve:
        return None
    best = min(curve, key=lambda c: (worst_class_count_bias(c), abs(c["count_bias_mean"]), -c["f1"],
                                     c["abs_count_error_mean"], -c["conf"]))
    return best["conf"]


def classes_with_evidence(entry: dict) -> set[str]:
    """The classes one curve entry actually says something about: those with a GT object or a
    surviving detection at that conf (``tp + fp + fn > 0``).

    A class whose entry is all zeros is not evidence of an unbiased count for it, the records
    simply hold none of it at this conf, and a bias of 0.0 there is arithmetic, not measurement.
    Read off the sweep's own statistics so no caller re-derives a second notion of "present".
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


def records_from_detector(target: dict, output: dict, *, width: int, height: int,
                          include_masks: bool = False, detections_cap: int | None = None) -> dict:
    """torchvision GT target + detector output -> one COCO per-image record.

    With ``include_masks`` (instance_seg / Mask R-CNN) each GT and prediction also carries
    an RLE ``segmentation``, so the record can be scored with ``iou_type='segm'``.

    ``detections_cap`` (non-gating provenance): when the caller knows the in-model
    ``detections_per_img`` this output was generated under, stamp ``cap_hit``, whether this
    image's raw detection count reached that cap, so a reviewer can see per-image cap
    saturation without re-deriving it later from a number that's no longer available by then.
    """
    gt = []
    gboxes = target.get("boxes")
    gmasks = target.get("masks") if include_masks else None
    if gboxes is not None and len(gboxes):
        glabels = target["labels"].detach().cpu().tolist()
        for i, ((x1, y1, x2, y2), c) in enumerate(zip(gboxes.detach().cpu().tolist(), glabels)):
            ann = {"category_id": int(c), "bbox": xywh(x1, y1, x2, y2),
                   "area": float((x2 - x1) * (y2 - y1)), "iscrowd": 0}
            if gmasks is not None and i < len(gmasks):
                ann["segmentation"] = _mask_to_rle(gmasks[i])
            gt.append(ann)
    dt = []
    pboxes = output.get("boxes")
    pmasks = output.get("masks") if include_masks else None
    if pboxes is not None and len(pboxes):
        plabels = output["labels"].detach().cpu().tolist()
        pscores = output["scores"].detach().cpu().tolist()
        for i, ((x1, y1, x2, y2), c, s) in enumerate(
                zip(pboxes.detach().cpu().tolist(), plabels, pscores)):
            res = {"category_id": int(c), "bbox": xywh(x1, y1, x2, y2), "score": float(s)}
            if pmasks is not None and i < len(pmasks):
                res["segmentation"] = _mask_to_rle(pmasks[i])
            dt.append(res)
    rec = build_coco_image_record(width, height, gt, dt, image_id=target.get("image_id"))
    if detections_cap is not None:
        rec["cap_hit"] = len(dt) >= detections_cap
    return rec


def _poly_flat(points) -> list[float]:
    return [float(c) for pt in points for c in (pt[0], pt[1])]


def records_from_annotation(gt, preds, *, width: int, height: int, force_segm: bool = False,
                             name_id: dict[str, int] | None = None):
    """Name-based :class:`Annotation` GT + predictions -> (iou_type, COCO per-image record).

    ``gt`` / ``preds`` are ``Annotation`` lists (a prediction carries a ``score``). The COCO
    ``category_id`` is a 1-indexed id per distinct ``subject`` name, shared by GT and predictions.
    Pass ``name_id`` when scoring more than one image: pycocotools accumulates every per-image record
    into one eval, so a subject must map to the *same* id in every image, a per-image-local map (the
    default when ``name_id`` is ``None``, fine for a single image) pools different subjects into one
    category across images and corrupts per-class AP. ``force_segm`` makes every box carry a
    rectangular ``segmentation`` so a whole dataset can be scored with ``iou_type='segm'``.

    A geometry-less annotation and a :class:`~tcip_annotation.state.Point` contribute no record and no
    ``name_id`` entry: neither has a box to score, and emitting one would put a fabricated extent into
    a delivery-grade AP, as GT nothing can match, or as a detection matching nothing.
    """
    from tcip_annotation.state import Point, Polygon, bbox_of

    def _scorable(a) -> bool:
        return a.geometry is not None and not isinstance(a.geometry, Point)

    def _has_poly(anns):
        return any(isinstance(a.geometry, Polygon) for a in anns)

    use_segm = force_segm or _has_poly(gt) or _has_poly(preds)
    iou_type = "segm" if use_segm else "bbox"

    if name_id is None:  # single-image scoring: a local map cannot disagree with itself
        names: list[str] = []
        for a in (*gt, *preds):
            if _scorable(a) and a.subject not in names:
                names.append(a.subject)
        name_id = {n: i + 1 for i, n in enumerate(names)}  # 1-indexed (background 0), like detector labels

    def _box_seg(x1, y1, x2, y2):
        return [[float(x1), float(y1), float(x2), float(y1), float(x2), float(y2), float(x1), float(y2)]]

    def _record(a, *, is_pred):
        if not _scorable(a):
            return None
        box = bbox_of(a.geometry)
        rec: dict = {"category_id": name_id[a.subject],
                     "bbox": xywh(box.x1, box.y1, box.x2, box.y2)}
        if is_pred:
            rec["score"] = float(a.score if a.score is not None else 0.0)
        else:
            rec["area"] = float((box.x2 - box.x1) * (box.y2 - box.y1))
            rec["iscrowd"] = 0
        if isinstance(a.geometry, Polygon):
            rec["segmentation"] = [_poly_flat(ring) for ring in a.geometry.rings if len(ring) >= 3]
        elif use_segm:
            rec["segmentation"] = _box_seg(box.x1, box.y1, box.x2, box.y2)
        return rec

    gt_recs = [r for r in (_record(a, is_pred=False) for a in gt) if r is not None]
    dt_recs = [r for r in (_record(a, is_pred=True) for a in preds) if r is not None]
    return iou_type, build_coco_image_record(width, height, gt_recs, dt_recs)


# ====================================================================
# In-house scalar metrics (expansion seam, segm AP already covers
# true instance segmentation once a mask head exists)
# ====================================================================

def classification_metrics(pred_labels: torch.Tensor, targets: torch.Tensor, num_classes: int) -> dict:
    """Accuracy + macro-F1 + per-class precision/recall/f1/support/count_bias.

    ``count_bias[c] = (predicted count - true count) / true count`` matters for validating a
    positive-state classifier: the phenotype is the positive-state *fraction*, so a class the
    classifier over-predicts inflates the fraction even at high accuracy. This is what the
    phenology gate reads.
    """
    pred = pred_labels.detach().cpu().long()
    gt = targets.detach().cpu().long()
    if gt.numel() == 0:
        return {"accuracy": 0.0, "f1": 0.0, "per_class": {}, "count_bias": {}}
    accuracy = (pred == gt).float().mean().item()
    per_class: dict[int, dict] = {}
    f1s = []
    for c in range(num_classes):
        tp = int(((pred == c) & (gt == c)).sum())
        fp = int(((pred == c) & (gt != c)).sum())
        fn = int(((pred != c) & (gt == c)).sum())
        support = int((gt == c).sum())
        pred_count = int((pred == c).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
        per_class[c] = {"precision": p, "recall": r, "f1": f1, "support": support,
                        "count_bias": (pred_count - support) / support if support > 0 else 0.0}
    return {
        "accuracy": accuracy,
        "f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "per_class": per_class,
        "count_bias": {c: per_class[c]["count_bias"] for c in per_class},
    }


def quadratic_weighted_kappa(
    pred_ranks: torch.Tensor, gt_ranks: torch.Tensor, num_ranks: int | None = None,
) -> float | None:
    """Chance-corrected ordinal agreement: squared rank-distance weights, expected agreement from
    the scored set's own observed rank marginals (no authored constant), the ordinal counterpart
    to :func:`tcip_mcp.pipelines.operating_point._classification_kappa`'s compensating-error
    floor. ``None`` when undefined: no items, or expected disagreement is zero (every populated
    true/predicted pair shares one rank, degenerate).

    ``num_ranks`` derives from the data (``max(pred, gt) + 1``) when not given; pass it explicitly
    when the caller knows the head's true rank count and the batch may not cover every rank.
    """
    pred = pred_ranks.detach().cpu().round().long()
    gt = gt_ranks.detach().cpu().round().long()
    n = gt.numel()
    if n == 0:
        return None
    k = num_ranks if num_ranks is not None else int(max(pred.max(), gt.max()).item()) + 1
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
    ``gt_values`` as precision (Pearson correlation) times an accuracy/bias penalty, the standard
    measurement-agreement statistic (as opposed to :func:`r_squared`'s "variance explained beyond
    the trivial mean baseline", a more general ML-model-skill question). A prediction that is
    perfectly correlated with GT but systematically offset (a constant bias, or a scale != 1) scores
    high on correlation alone but low here, exactly the failure mode this statistic is meant to
    surface. ``None`` when undefined: no items, or either series has zero variance (the correlation
    term, and this statistic's denominator, are undefined).

    ``CCC = 2*r*sigma_pred*sigma_gt / (sigma_pred^2 + sigma_gt^2 + (mean_pred - mean_gt)^2)``, with
    ``r`` the Pearson correlation and ``sigma`` the population (not sample) standard deviation, so
    this and :func:`r_squared` are computed over the same population-statistics convention.
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


def ordinal_metrics(pred_ranks: torch.Tensor, gt_ranks: torch.Tensor) -> dict:
    pred = pred_ranks.detach().cpu().float()
    gt = gt_ranks.detach().cpu().float()
    if gt.numel() == 0:
        return {"mae": 0.0, "rank_acc": 0.0, "quadratic_weighted_kappa": None}
    return {
        "mae": (pred - gt).abs().mean().item(),
        "rank_acc": (pred.round() == gt.round()).float().mean().item(),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(pred_ranks, gt_ranks),
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
    standard convention. ``pixel_acc`` is the fraction of correctly labelled pixels. Pixels equal
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
    """Resolve the COCOeval ``iouType`` actually used to score ``task``.

    An explicit ``iou_type`` wins; otherwise ``segm`` for instance_seg, ``bbox``
    for detection, ``""`` for non-COCO tasks. Single source of truth so
    ``run_test_evaluation`` records the same value ``evaluate`` scores with.
    """
    if iou_type:
        return iou_type
    if task == "instance_seg":
        return "segm"
    return "bbox" if task == "detection" else ""


@torch.no_grad()
def evaluate(
    model, loader, device, task: str, *,
    conf_threshold: float = 0.25, iou_threshold: float = 0.5,
    iou_type: str | None = None, max_dets: int = 100, score_weights: dict | None = None,
    trait: str | None = None,
) -> dict:
    """Compute per-task validation/test metrics. Returns bare metric keys.

    ``trait``: when set, a count trait's derived localization criterion (traits.py, e.g. a
    center-match at half the class-average size) governs the reported detection count and the f1 the
    selection composite optimizes; map50 stays a labeled comparability metric. Absent -> the
    IoU@``iou_threshold`` convention governs (the prior behavior).
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
            if detector is not None:
                detector.train()
                for m in detector.modules():
                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                        m.eval()
                ld = detector(images, targets)
                total_loss += float(sum(ld.values()).item())
                n_loss += 1
            else:
                model.training = True
                for head in getattr(model, "heads", []):
                    head.training = True
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
        criterion = resolve_match_criterion(trait, per_image, iou_threshold=iou_threshold)
        if criterion["kind"] == "center_match":
            gc = governing_counts(per_image, criterion, conf_threshold=conf_threshold, max_dets=max_dets)
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
        num_classes = getattr(model.heads[0], "num_classes", int(torch.cat(cls_g).max()) + 1)
        result.update(_reported_metrics(
            classification_metrics(torch.cat(cls_p), torch.cat(cls_g), num_classes)))
    elif task == "ordinal" and ord_p:
        result.update(_reported_metrics(ordinal_metrics(torch.cat(ord_p), torch.cat(ord_g))))
    elif task == "regression" and reg_p:
        result.update(_reported_metrics(regression_metrics(torch.cat(reg_p), torch.cat(reg_g))))
    elif task == "semantic_seg" and seg_p:
        pred, gt = torch.cat(seg_p), torch.cat(seg_g)
        num_classes = getattr(model.heads[0], "num_classes", int(gt.max()) + 1)
        result.update(_reported_metrics(semantic_seg_metrics(pred, gt, num_classes)))

    return result
