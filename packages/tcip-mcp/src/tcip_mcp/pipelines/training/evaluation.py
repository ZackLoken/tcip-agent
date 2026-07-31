"""Task-aware evaluation metrics + composite selection objective.

Single home for:
  * the pycocotools-backed detection / instance_seg metrics (mAP + operating-point
    TP/FP/FN), shared by training ``_validate``, ``run_test_evaluation`` and the
    agent/GUI tool ``score_predictions`` — one source of
    truth, the canonical COCO mAP definition;
  * in-house scalar metrics for classification / ordinal / regression (the seam
    where pycocotools ``iou_type='segm'`` can later cover true instance seg);
  * the chestnut-burr composite selection objective (lower = better);
  * a task-agnostic two-pass ``evaluate()`` and a ``run_test_evaluation()``.

pycocotools is imported lazily inside the COCO functions. Every pycocotools call
is wrapped in ``redirect_stdout`` because ``createIndex``/``loadRes``/``summarize``
print to stdout, which would corrupt the MCP stdio transport.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch

from tcip_mcp.pipelines.resolution import DEFAULT_CONF

logger = logging.getLogger(__name__)

# Composite-objective weights. Note: in compute_composite_objective the F1 and
# mAP50 terms are multiplied by 10 to lift them onto the same scale as val_loss,
# so a weight here acts on that *scaled* term (a 0.35 f1 weight ~ 3.5 loss-units
# of pull at f1=0). See compute_composite_objective for the exact formula.
# These weights silently decide which checkpoint wins, so they are a caller-owned SELECTION POLICY
# (validated=false, not a data derivation): overridable via the ``score_weights`` kwarg on every
# eval surface. Documented default, not a frozen truth — no derivation label is claimed for it.
DEFAULT_SCORE_WEIGHTS: dict[str, float] = {"loss": 0.45, "f1": 0.35, "map50": 0.20}

# The metric keys that ``evaluate()`` labels comparability-only (``map50_role``) once a center-match
# trait's own governing criterion takes over ``precision``/``recall``/``f1`` (see the center_match
# branch below) — the AP@0.5-family keys plus the IoU@0.5-convention precision/recall/F1 that get
# relabeled ``iou_*`` at that point. K9: the single source of truth for "is this metric governing or
# comparability-only for a center-match trait" — ``resolve_selection_metric`` (generic_trainer.py)
# and ``select_best_model`` (model_tools.py) both import this rather than re-encoding the names.
CENTER_MATCH_COMPARABILITY_KEYS: frozenset[str] = frozenset({
    "map50", "map", "map_at_maxdets", "map50_at_maxdets",
    "iou_precision", "iou_recall", "iou_f1",
})


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

def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


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
    BOTH the standard 100 cap and a non-100 operating cap without summarize()'s hardcoded 100."""
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

    Returns mAP at the STANDARD 100-detection cap (``map``/``map50``/``map75`` — comparable across
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
# derived to minimize the signed per-image count bias E[FP-FN] — not F1 — because the phenotype is a
# count (Sigma pred ~= Sigma gt) for a trait whose recorded count_objective/localization say so
# (traits.py; K18 B3/B4 — neither is authored, both are derived/decided once and recorded).

def _centers_xywh(anns: list[dict]) -> list[tuple[float, float]]:
    return [(a["bbox"][0] + a["bbox"][2] / 2.0, a["bbox"][1] + a["bbox"][3] / 2.0) for a in anns]


def _char_size_xywh(a: dict) -> float:
    """Characteristic size of a box = sqrt(w*h) — scale-robust for a tolerance basis."""
    w, h = float(a["bbox"][2]), float(a["bbox"][3])
    return (max(w, 0.0) * max(h, 0.0)) ** 0.5


def gt_class_avg_size(per_image: list[dict], class_id: int | None = None) -> float:
    """Average characteristic GT box size — the DERIVED basis for the center-match tolerance.

    Derived from the data in hand (not pinned): the tolerance is ``half_class_avg_size`` (traits.py).
    """
    sizes = [
        _char_size_xywh(a)
        for rec in per_image for a in rec.get("gt", [])
        if class_id is None or a["category_id"] == class_id
    ]
    return float(np.mean(sizes)) if sizes else 0.0


def _center_match_image(gt: list[dict], dt: list[dict], tolerance: float) -> tuple[int, int, int]:
    """Greedy nearest-center 1:1 matching (dt pre-sorted by score desc). Returns (tp, fp, fn)."""
    gt_centers = _centers_xywh(gt)
    used = [False] * len(gt_centers)
    tp = 0
    for dx, dy in _centers_xywh(dt):
        best_j, best_d = -1, tolerance
        for j, (gx, gy) in enumerate(gt_centers):
            if used[j]:
                continue
            d = ((dx - gx) ** 2 + (dy - gy) ** 2) ** 0.5
            if d <= best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            used[best_j] = True
            tp += 1
    return tp, len(dt) - tp, len(gt_centers) - tp


def resolve_match_criterion(trait_name: str | None, per_image: list[dict], *,
                            class_id: int | None = None, iou_threshold: float = 0.5) -> dict:
    """The ONE localization criterion that GOVERNS a trait's phenotype count + model selection (R3/D9).

    Reads the trait's recorded ``localization`` kind (center_match vs iou_match, traits.py) and
    derives its per-dataset tolerance from the GT in hand — never a pinned value. Returns
    ``{kind, tolerance | iou_threshold, derived_from, trait}``. With no trait (or an iou_match trait),
    it is IoU matching at ``iou_threshold`` — the labeled comparability convention (AP@0.5), which
    governs nothing on its own; a count trait's derived center-match tolerance is what the phenotype
    and checkpoint selection rest on.

    K18 B3: ``localization`` is no longer authored — it is derived once, the first time real GT is
    available for a trait with no recorded kind (via ``derivations.derive_localization_kind``),
    persisted through ``traits.write_trait_spec_fields`` with ``data_derived_at_runtime``
    provenance, and read from the recorded value on every later call. A recorded kind is also
    cheaply re-checked against what the CURRENT data would derive, every real call — divergence
    surfaces a warning (``kind_diverged`` in the returned dict) rather than silently switching,
    per the standing "constrain by observation, not permission" rule; only an explicit re-derive
    changes the recorded value. This is also the single point every consumer of a trait's
    localization criterion goes through — ``generic_trainer.py`` reads the recorded field directly
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
                "would derive (%r) — not switched (observation, not permission); re-derive "
                "explicitly via update_trait_spec_fields if this data is now representative.",
                trait_name, kind, live_derived_kind)
    elif live_derived_kind is not None:
        kind = live_derived_kind
        kind_source = "data_derived_at_runtime"
        # Stamp via resolution.derived() — NOT aliased on import — so test_provenance_honesty.py's
        # AST scanner (which matches the literal call name "derived") actually sees this label; the
        # TraitSpec.provenance entry below is a separate, free-text record of WHO decided (for a
        # human/agent reader), not a substitute for this mechanical check.
        from tcip_mcp.pipelines.resolution import derived
        derived("localization_kind", kind,
               derived_from="achievable IoU under annotation jitter (GT characteristic size)")
        try:
            from tcip_mcp import traits as traits_module
            traits_module.write_trait_spec_fields(
                trait_name, {"localization": kind},
                [f"localization: data_derived_at_runtime — derived from "
                 f"{sum(len(b) for b in boxes_per_image)} GT boxes (achievable IoU under jitter)"],
            )
        except ValueError:
            logger.warning("could not persist derived localization kind for %r", trait_name, exc_info=True)
    else:
        raise ValueError(
            f"trait {trait_name!r} has no recorded localization kind and no GT in this call to "
            "derive one from — cannot resolve a match criterion. Calibrate or evaluate against a "
            "labeled reference at least once before this trait's localization kind can be known.")

    # Stamp via resolution.derived()/default() — NOT aliased on import, and with the derived_from
    # LITERAL inlined directly into the call (round-2 stage-6 review: passing it as a variable,
    # even unaliased, is ALSO invisible to the AST scanner — it only reads a literal string or an
    # f-string's leading constant written directly at the call site, never a name reference).
    # derived() for a real per-dataset computation, default() for the honest "underivable, fell
    # back" case (never claimed as a derivation — and not scanned by test_provenance_honesty.py at
    # all, correctly, since it makes no derivation claim to check). The label text lives ONCE, in
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


def governing_counts(per_image: list[dict], criterion: dict, *, conf_threshold: float,
                     class_id: int | None = None, max_dets: int = 1000) -> dict:
    """tp/fp/fn/precision/recall/f1 at the criterion that GOVERNS the phenotype count (R3).

    ``center_match`` uses greedy nearest-center matching at the derived tolerance; ``iou_match`` reuses
    the COCO IoU matcher. This count is what a count-trait phenotype and model selection rest on —
    distinct from AP@0.5, which stays a labeled comparability metric that governs nothing.
    """
    if criterion["kind"] == "center_match":
        # Deliberately uncapped by max_dets (K9), unlike the iou_match branch below: a count
        # trait's total is every conf-surviving detection, not the COCOeval detection-cap
        # convention that AP@0.5 comparability uses.
        tol = float(criterion["tolerance"])
        tp = fp = fn = 0
        for rec in per_image:
            gt = [a for a in rec.get("gt", []) if class_id is None or a["category_id"] == class_id]
            dt = sorted(
                (d for d in rec.get("dt", [])
                 if d.get("score", 1.0) >= conf_threshold
                 and (class_id is None or d["category_id"] == class_id)),
                key=lambda d: -d.get("score", 1.0),
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

    The single implementation of "match, count, and take the per-image count bias" — the class-pooled
    curve entry and every per-class entry beside it both come from here, so a per-class bias can
    never be measured by a second matcher that drifts from the pooled one.
    """
    tp = fp = fn = 0
    biases: list[int] = []
    n_present = 0
    for rec in per_image:
        gt = [a for a in rec.get("gt", []) if class_id is None or a["category_id"] == class_id]
        dt = sorted(
            (d for d in rec.get("dt", [])
             if d["score"] >= conf and (class_id is None or d["category_id"] == class_id)),
            key=lambda d: -d["score"],
        )
        t, f, n = _center_match_image(gt, dt, tolerance)
        tp += t
        fp += f
        fn += n
        biases.append(f - n)
        if gt or dt:
            n_present += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    abs_biases = [abs(b) for b in biases]
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "count_bias_mean": float(np.mean(biases)) if biases else 0.0,
        "abs_count_error_mean": float(np.mean(abs_biases)) if biases else 0.0,
        # Tail dispersion (Fix B) — a p90 of |bias|, not another mean, since a mean can hide one
        # badly-off image among many. Reference-sufficiency terms (Fix C) — computed here, once,
        # so the gate never re-derives a second matcher over the same per-image biases.
        "count_error_p90": float(np.quantile(abs_biases, 0.9)) if abs_biases else 0.0,
        "count_bias_std": float(np.std(biases, ddof=1)) if len(biases) > 1 else 0.0,
        "n_images": len(biases),
        # Images that actually carried this class (gt or a surviving dt) — distinct from n_images
        # (the whole holdout) because a class scarce in the reference gets its SE denominator from
        # how much evidence there really is, not diluted by images that say nothing about it.
        "n_present": n_present,
    }


def _class_ids_present(per_image: list[dict], class_id: int | None = None) -> list[int]:
    """The class ids to break the sweep down by: those the records carry in gt or dt, derived from
    the data in hand rather than a registry read or a pinned id space.

    An explicit ``class_id`` is returned as-is — the caller has already scoped the sweep to it, and
    it stays the breakdown's one key even on records that turn out to carry none of it. Every
    annotation and detection must carry ``category_id``; a per-class breakdown of records that do
    not identify their classes is not something to guess at.
    """
    if class_id is not None:
        return [class_id]
    ids = {a["category_id"] for rec in per_image for a in rec.get("gt", [])}
    ids |= {d["category_id"] for rec in per_image for d in rec.get("dt", [])}
    return sorted(ids)


def sweep_operating_point(per_image: list[dict], *, tolerance: float, class_id: int | None = None,
                          conf_grid: list[float] | None = None, max_thresholds: int = 80) -> dict:
    """Sweep the confidence threshold over ``per_image`` records via center-matching.

    One model pass produces ``per_image`` (unfiltered dt with scores); this sweeps conf cheaply in
    Python — no re-forwarding. For each conf: aggregate TP/FP/FN and per-image count bias (FP-FN).
    Passing an explicit ``conf_grid`` (e.g. a single-element ``[conf]``) skips grid construction and
    evaluates EXACTLY those points — the exact-conf holdout evaluation (no nearest-neighbor snap)
    relies on this. Returns ``{tolerance, class_id, curve:[{conf, tp, fp, fn, precision, recall, f1,
    count_bias_mean, abs_count_error_mean, count_error_p90, count_bias_std, n_images, n_present,
    per_class}]}``
    — the dispersion + reference-sufficiency terms are per-conf statistics across ``per_image`` that
    the operating-point gate reads, never recomputes.

    ``per_class`` (K4 #4) carries the same statistics measured within each class the records carry,
    keyed by ``str(category_id)`` (string keys so an in-memory sweep and one round-tripped through
    the JSON sidecar have the same shape). It exists because the pooled entry beside it cannot see a
    per-class error: matching is class-blind there, so a detector that calls every class-A object
    class B reports tp-only, zero pooled bias — while the delivered per-class counts, which are the
    phenotype for a fraction/ratio trait, are both wrong. Class ids come from the records themselves;
    which of them is the trait's positive class is not read here and is not needed to measure bias.
    """
    scores = sorted({d["score"] for rec in per_image for d in rec.get("dt", [])})
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
            # is that class's entry — reused rather than recomputed, which keeps the single-class
            # sweep (every reference the platform builds today) at its original cost.
            per_class = {str(class_ids[0]): pooled}
        else:
            per_class = {str(cid): _count_stats_at_conf(per_image, tolerance=tolerance, conf=conf,
                                                        class_id=cid)
                         for cid in class_ids}
        curve.append({"conf": float(conf), **pooled, "per_class": per_class})
    return {"tolerance": float(tolerance), "class_id": class_id, "curve": curve}


def worst_class_count_bias(entry: dict) -> float:
    """The largest |mean per-image count bias| over the classes in one curve entry — the class this
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

    This is the count-trait operating point — where the model's totals match GT totals — which is
    generally not the F1-max point (that optimizes matching, not count agreement).

    Aimed at the worst class rather than the pooled bias (K4 #4) so the pick and the gate optimize
    the same thing. With two classes of opposite sign the two objectives coincide, which is why the
    first draft of this fix left the pick pooled — but that reasoning does not survive a third class:
    a conf can buy pooled balance by trading one class's over-count against another's under-count and
    be strictly worse for the worst class than a conf on the same curve that the gate would accept
    (see ``test_pick_serves_the_worst_class_not_the_pooled_total``). Picking pooled there refuses a
    model that has a valid operating point, and tells the breeder to fix a model that is not broken.
    On a single-class reference the two objectives are the same number, so this changes no operating
    point the platform picks today.

    The final ``-c["conf"]`` tie-break (K2, stage-6 review) is a completion of the existing
    tie-break, not a new selection objective: when |bias| and F1 and |abs error| are ALL exactly
    tied across several confs — which happens on a reference filtered to a floor, since nothing
    below the floor is visible to distinguish them — the lowest tied conf (e.g. the grid's seeded
    0.0) used to win by default. That is generically the worst of the tied candidates in practice
    (it admits the most low-confidence noise for no better count agreement) and, combined with the
    conf-censoring guard, could make a genuinely trustworthy pick read as censored merely because the
    tie resolved to the search floor. Preferring the HIGHEST tied conf breaks ties toward the most
    conservative, best-supported candidate among equals — it does not change what is optimized.
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

    A class whose entry is all zeros is not evidence of an unbiased count for it — the records
    simply hold none of it at this conf, and a bias of 0.0 there is arithmetic, not measurement.
    Read off the sweep's own statistics so no caller re-derives a second notion of "present".
    """
    return {cid for cid, s in (entry.get("per_class") or {}).items()
            if s["tp"] + s["fp"] + s["fn"] > 0}


def pick_f1_max(sweep: dict) -> float | None:
    """The F1-max conf — reported alongside the count-unbiased point to show the trade-off."""
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

    ``detections_cap`` (Fix K, non-gating provenance): when the caller knows the in-model
    ``detections_per_img`` this output was generated under, stamp ``cap_hit`` — whether this
    image's raw detection count reached that cap — so a reviewer can see per-image cap
    saturation without re-deriving it later from a number that's no longer available by then.
    """
    gt = []
    gboxes = target.get("boxes")
    gmasks = target.get("masks") if include_masks else None
    if gboxes is not None and len(gboxes):
        glabels = target["labels"].detach().cpu().tolist()
        for i, ((x1, y1, x2, y2), c) in enumerate(zip(gboxes.detach().cpu().tolist(), glabels)):
            ann = {"category_id": int(c), "bbox": _xyxy_to_xywh(x1, y1, x2, y2),
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
            res = {"category_id": int(c), "bbox": _xyxy_to_xywh(x1, y1, x2, y2), "score": float(s)}
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
    into one eval, so a subject must map to the *same* id in every image — a per-image-local map (the
    default when ``name_id`` is ``None``, fine for a single image) pools different subjects into one
    category across images and corrupts per-class AP. ``force_segm`` makes every box carry a
    rectangular ``segmentation`` so a whole dataset can be scored with ``iou_type='segm'``.

    A geometry-less annotation and a :class:`~tcip_annotation.state.Point` contribute no record and no
    ``name_id`` entry: neither has a box to score, and emitting one would put a fabricated extent into
    a delivery-grade AP — as GT nothing can match, or as a detection matching nothing.
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
                     "bbox": _xyxy_to_xywh(box.x1, box.y1, box.x2, box.y2)}
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
# In-house scalar metrics (expansion seam — segm AP already covers
# true instance segmentation once a mask head exists)
# ====================================================================

def classification_metrics(pred_labels: torch.Tensor, targets: torch.Tensor, num_classes: int) -> dict:
    """Accuracy + macro-F1 + per-class precision/recall/f1/support/count_bias.

    ``count_bias[c] = (predicted count - true count) / true count`` matters for validating the
    elongation classifier: the phenotype is the elongated *fraction*, so a class the classifier
    over-predicts inflates the fraction even at high accuracy. This is what the phenology gate reads.
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


def ordinal_metrics(pred_ranks: torch.Tensor, gt_ranks: torch.Tensor) -> dict:
    pred = pred_ranks.detach().cpu().float()
    gt = gt_ranks.detach().cpu().float()
    if gt.numel() == 0:
        return {"mae": 0.0, "rank_acc": 0.0}
    return {
        "mae": (pred - gt).abs().mean().item(),
        "rank_acc": (pred.round() == gt.round()).float().mean().item(),
    }


def regression_metrics(pred_values: torch.Tensor, gt_values: torch.Tensor) -> dict:
    pred = pred_values.detach().cpu().float()
    gt = gt_values.detach().cpu().float()
    if gt.numel() == 0:
        return {"mae": 0.0, "rmse": 0.0}
    return {
        "mae": (pred - gt).abs().mean().item(),
        "rmse": ((pred - gt) ** 2).mean().sqrt().item(),
    }


def semantic_seg_metrics(preds: torch.Tensor, targets: torch.Tensor, num_classes: int,
                         ignore_index: int | None = None) -> dict:
    """Standard mean-IoU / Dice for semantic segmentation from per-pixel class maps.

    ``preds`` and ``targets`` are integer class-id tensors of matching shape (any shape;
    both are flattened). Per class: IoU = ``|P∩G| / |P∪G|`` and Dice = ``2|P∩G| / (|P|+|G|)``.
    ``mIoU`` / ``dice`` average only over classes present in preds or targets — a class absent
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
        if union == 0:  # class absent from both — ratio undefined, exclude from the mean
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
# Task-agnostic evaluate() — loss pass + prediction pass
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
    """Compute per-task validation/test metrics. Returns BARE metric keys.

    ``trait``: when set, a count trait's DERIVED localization criterion (traits.py, e.g. catkin's
    center-match at half the class-average size) governs the reported detection count and the f1 the
    selection composite optimizes; map50 stays a labeled comparability metric (D9). Absent -> the
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
            # Loss pass over the FULL batch — all-negative (empty-box) images contribute their
            # background/objectness loss, so val_loss penalizes false positives on empty frames and
            # matches the train loop's distribution (CV9). BN stays eval via the train()+BN.eval() trick.
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
                total_loss += float(sum(ld.values()).item()) if isinstance(ld, dict) else float(ld)
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
            total_loss += float(sum(ld.values()).item()) if isinstance(ld, dict) else float(ld)
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
    result: dict = {"loss": round(loss, 6)}

    if is_detection:
        m = coco_detection_metrics(per_image, iou_type=eff_iou_type, iou_threshold=iou_threshold,
                                   conf_threshold=conf_threshold, max_dets=max_dets)
        result.update({
            "precision": round(m["precision"], 6), "recall": round(m["recall"], 6),
            "f1": round(m["f1"], 6), "map50": round(m["map50"], 6), "map": round(m["map"], 6),
            "map_at_maxdets": round(m["map_at_maxdets"], 6),
            "map50_at_maxdets": round(m["map50_at_maxdets"], 6),
        })
        # R3/D9: a count trait's derived criterion governs the reported count + the selection f1;
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
        result["objective"] = round(compute_composite_objective(loss, governing_f1, m["map50"], score_weights), 6)
    elif task == "classification" and cls_p:
        num_classes = getattr(model.heads[0], "num_classes", int(torch.cat(cls_g).max()) + 1)
        # classification_metrics now also returns per_class/count_bias dicts — round only the scalars.
        result.update({k: (round(v, 6) if isinstance(v, (int, float)) else v)
                       for k, v in classification_metrics(torch.cat(cls_p), torch.cat(cls_g), num_classes).items()})
    elif task == "ordinal" and ord_p:
        result.update({k: round(v, 6) for k, v in ordinal_metrics(torch.cat(ord_p), torch.cat(ord_g)).items()})
    elif task == "regression" and reg_p:
        result.update({k: round(v, 6) for k, v in regression_metrics(torch.cat(reg_p), torch.cat(reg_g)).items()})
    elif task == "semantic_seg" and seg_p:
        pred, gt = torch.cat(seg_p), torch.cat(seg_g)
        num_classes = getattr(model.heads[0], "num_classes", int(gt.max()) + 1)
        m = semantic_seg_metrics(pred, gt, num_classes)
        # Scalars rounded; per-class dicts (with None for absent classes) passed through.
        result.update({k: (round(v, 6) if isinstance(v, (int, float)) else v) for k, v in m.items()})

    return result


def _producer_identity(ckpt_path: str) -> dict:
    """Producing-model identity for a test-results stamp (checkpoint sha + experiment id).

    Best-effort — a foreign checkpoint records the sha and leaves the experiment id null rather
    than failing the evaluation.
    """
    from tcip_mcp.model_registry import resolve_model_identity

    identity = resolve_model_identity(ckpt_path)
    return {"model_sha256": identity["sha256"], "experiment_id": identity["experiment_id"]}


def run_test_evaluation(
    ckpt_path: str, loader, device, task: str, output_dir: str, *,
    conf_threshold: float = DEFAULT_CONF, iou_threshold: float = 0.5,  # report at the ship point
    iou_type: str | None = None, max_dets: int = 100, score_weights: dict | None = None,
    tiling: dict | None = None, trait: str | None = None,
) -> dict:
    """Load ``model_best.pt``, evaluate ``loader``, write ``test_results.json``.

    ``tiling`` describes the eval dataset regime for provenance only (the loader is built by the
    caller): a tile-level run scores per-tile predictions against per-tile GT (a diagnostic that
    matches the training-run val mAP), not the delivery regime — the stamp keeps the two from being
    silently conflated. See CV1: use ``run_full_frame_evaluation`` for a delivery-grade metric.
    """
    from tcip_mcp.pipelines.model_build import build_model

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    metrics = evaluate(model, loader, device, task, conf_threshold=conf_threshold,
                       iou_threshold=iou_threshold, iou_type=iou_type, max_dets=max_dets,
                       score_weights=score_weights, trait=trait)
    tiled = bool(tiling and tiling.get("enabled", True) and task == "detection")
    result = {
        **metrics,
        "model_path": str(ckpt_path), "task": task,
        **_producer_identity(ckpt_path),
        "iou_type": effective_iou_type(task, iou_type),
        "iou_threshold": iou_threshold, "conf_threshold": conf_threshold, "max_dets": max_dets,
        "tiled": tiled,
        "eval_regime": "tile-level" if tiled else "full-frame-single-pass",
    }
    out = Path(output_dir) / "test_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    result["results_path"] = str(out)
    return result


def run_full_frame_evaluation(
    ckpt_path: str, images_dir: str, labels_dir: str, output_dir: str, *,
    subject: str | None = None, attribute: str | None = None,
    conf_threshold: float = DEFAULT_CONF, iou_threshold: float = 0.5,
    tile_size: int | None = None, overlap: float | None = None, global_nms_iou: float = 0.3,
    max_dets: int = 1000, postprocess: str = "nms", device: str | None = None,
    trait: str | None = None,
) -> dict:
    """Delivery-grade detection eval (CV1 Tier-2): tiled INFERENCE reconstructed to full frame,
    matched to full-frame GT.

    Unlike tile-level eval this exercises the cross-tile merge and scores against un-fragmented GT,
    so it answers "how well does the shipped full-frame count match ground truth" — the number that
    gates a phenotype delivery. Tile-level (``run_test_evaluation`` with ``tiling``) is a diagnostic
    that matches the training-run val mAP; it must not be reported as the delivery metric.

    Only call THIS with a checkpoint that was actually trained tiled (``predictor.train_tile_size``
    persisted), a foreign checkpoint whose geometry you can independently derive and state, or one
    where you intend to state a tile scale yourself. A checkpoint trained WITHOUT tiling has no
    "regime mismatch" to reconcile in the first place — ``evaluate_model``'s default
    (``use_tiled_inference=False``) full-frame single-pass path IS that model's correct delivery
    gate (same untiled regime end to end), and is the one to call instead of this function.

    ``tile_size``/``overlap`` are resolved (K10 CV3) by the SAME precedence
    ``run_inference`` uses — explicit > the checkpoint's own persisted training geometry > a
    documented default — via the shared ``resolve_tile_geometry``. Unlike the exploratory
    ``run_inference``, THIS is the delivery-gating call: an unresolvable ``tile_size`` (no explicit
    value and nothing persisted on the checkpoint) raises rather than silently fabricating 640, since
    a wrong tile scale here is a wrong number that gates a phenotype, not just a wrong preview.
    ``overlap`` alone falling back to a default does NOT raise — a checkpoint trained with no
    tiling overlap convention at all has no persisted overlap analog, which is a legitimate fact,
    not a missing derivation; only ``tile_size``'s absence changes the object count's scale.

    This is a BOX metric (``iou_type="bbox"``): it requests boxes-only tiled inference
    (``predict_tiled(require_masks=False)``), so an instance_seg checkpoint is gated here on its
    boxes/counts, never on its masks. A mask-quality gate is separate work; do not report this
    number as one.
    """
    from tcip_mcp.dataset_layout import annotation_date
    from tcip_mcp.pipelines.data.datasets import _json_det_targets, _resolve_registry_id_map
    from tcip_mcp.pipelines.inference.predictor import build_predictor, resolve_tile_geometry
    from tcip_mcp.pipelines.operating_point import _cap_saturated_frac

    predictor = build_predictor(
        checkpoint_path=str(ckpt_path), device=device,
        score_threshold=conf_threshold, nms_iou=global_nms_iou, max_dets=max_dets)

    resolved_tile, tile_size_source, resolved_overlap, overlap_source = resolve_tile_geometry(
        predictor, tile_size=tile_size, overlap=overlap)
    if tile_size_source == "default":
        raise ValueError(
            f"Cannot resolve a trustworthy tile_size for {ckpt_path}: no explicit tile_size was "
            "passed and the checkpoint carries no persisted training tile geometry. This is the "
            "delivery-grade gating path (\"Report THIS to gate a delivery\") — it refuses to silently "
            "score at a fabricated default rather than the model's real training scale. If this "
            "checkpoint was trained WITHOUT tiling, call evaluate_model with "
            "use_tiled_inference=False instead — that untiled regime IS its correct delivery gate, "
            "with no scale to reconcile. If you have genuinely derived (or intend to derive, e.g. "
            "from this dataset's object-size distribution vs. image resolution — 'Parameters: "
            "derive, don't pin') a tile scale for this checkpoint, pass it explicitly via the "
            "tiling= dict (and overlap, if known); it is NOT cross-checked against the checkpoint's "
            "actual training scale, so state it deliberately, not as a guess."
        )
    tile_size, overlap = resolved_tile, resolved_overlap

    img_dir, lbl_dir = Path(images_dir), Path(labels_dir)
    _lbl_date = annotation_date(lbl_dir)
    # The GT category ids come from the run's single assign_class_ids map, read through the same
    # loader-side reader (_json_det_targets), so delivery-grade GT never diverges from what trained.
    # No try/except here: _resolve_registry_id_map's only exception is its own deliberate
    # ValueError refusal (attribute classification with no registry to order values) — that must
    # reach the caller, not be silently swallowed into "no ground truth" for a delivery-grade
    # evaluation. Its one legitimate "no registry, that's fine" case (single-class, no attribute)
    # already returns normally without raising.
    _gt_id_map = None
    if subject:
        _reg, _gt_id_map = _resolve_registry_id_map(lbl_dir, subject, attribute)
    # Same negative rail the training set uses: an image with no label record has no ground truth,
    # so scoring it turns every correct detection into a false positive and drags down the very
    # precision this delivery-grade number gates a phenotype on.
    from tcip_mcp.pipelines.data.datasets import IMAGE_EXTS, image_name_map, trainable_stems

    names = image_name_map(img_dir)
    if lbl_dir.is_dir():
        keep, sample_counts = trainable_stems(lbl_dir, img_dir, subject=subject, date=_lbl_date)
        paths = [img_dir / names[s] for s in keep if s in names]
    else:
        # No label store, so no rail to apply and no ground truth either. Filtering to nothing here
        # would refuse a call that is merely metric-less, not wrong.
        sample_counts = {}
        paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    per_image: list[dict] = []
    n_excluded_incomplete = 0
    for p in paths:
        gt = []
        gt_file = lbl_dir / f"{p.stem}.json"
        if gt_file.is_file() and _gt_id_map is not None:
            # Same loader-side reader + id map the training targets use (1-indexed to match the
            # predictor's torchvision labels), so this delivery-grade GT can't diverge from training.
            gboxes, glabels, n_unlabeled = _json_det_targets(str(gt_file), subject, attribute, _gt_id_map)
            # Stage-6 review N2: an image with ANY instance unlabeled for `attribute` has
            # incomplete GT for this scope — excluded from delivery-grade scoring entirely (the
            # same precedent applied to a missing label file just above), never scored against its
            # labeled subset alone, which would turn its real, unlabeled objects into false
            # positives against the very number this function's docstring calls the delivery gate.
            if n_unlabeled:
                n_excluded_incomplete += 1
                continue
            for (x1, y1, x2, y2), lab in zip(gboxes, glabels):
                gt.append({"category_id": int(lab),
                           "bbox": _xyxy_to_xywh(x1, y1, x2, y2), "iscrowd": 0})
        # require_masks=False: this gate matches boxes to full-frame GT and never reads masks, so a
        # tile-trained instance_seg checkpoint evaluates here exactly as a detector does, instead of
        # being blocked by predict_tiled's mask contract for masks nothing on this path consumes.
        r = predictor.predict_tiled(str(p), tile_size=tile_size, overlap=overlap,
                                    global_nms_iou=global_nms_iou, postprocess=postprocess,
                                    require_masks=False)
        w, h = int(r["width"]), int(r["height"])
        dt = [{"category_id": int(lab), "bbox": _xyxy_to_xywh(*b), "score": float(s)}
              for b, s, lab in zip(r["boxes"], r["scores"], r["labels"])]
        rec = build_coco_image_record(w, h, gt, dt, image_id=p.stem)
        # K10 finding 2 residual: max_dets is honored verbatim on this gating path (no rescuing
        # sentinel) — stamp per-image cap saturation so a caller-explicit low max_dets that
        # truncates real detections is visible rather than silently assumed away.
        rec["cap_hit"] = len(dt) >= max_dets
        per_image.append(rec)

    m = coco_detection_metrics(per_image, iou_threshold=iou_threshold,
                               conf_threshold=conf_threshold, max_dets=max_dets)
    keys = ("map", "map50", "map75", "map_at_maxdets", "map50_at_maxdets",
            "precision", "recall", "f1", "tp", "fp", "fn", "n_images", "n_gt", "n_pred")
    result = {
        **{k: m[k] for k in keys},
        # task: the predictor's own real task (round-2 stage-6 finding — this used to hardcode
        # "detection" even for an instance_seg checkpoint, mislabeling its delivery artifact).
        # iou_type stays the literal "bbox": this gate always computes a box-only metric by design
        # (require_masks=False above), true regardless of task — see the docstring.
        "model_path": str(ckpt_path), "task": getattr(predictor, "task", "detection"),
        "iou_type": "bbox",
        **_producer_identity(ckpt_path),
        "iou_threshold": iou_threshold, "conf_threshold": conf_threshold, "max_dets": max_dets,
        "max_dets_cap_saturated_frac": _cap_saturated_frac(per_image),
        "tile_size": tile_size, "tile_size_source": tile_size_source,
        "overlap": overlap, "overlap_source": overlap_source,
        "tiled": True, "eval_regime": "full-frame-tiled-inference",
        # Which images this number was computed over, and which were held out for having no
        # ground truth or incomplete attribute labeling — so a reviewer can reconstruct the
        # denominator, not just the metric (stage-6 review N2: an excluded-incomplete image is
        # disclosed here, not silently scored against its labeled subset).
        "scored_images": len(per_image), "sample_counts": sample_counts,
        "n_excluded_incomplete_attribute": n_excluded_incomplete,
    }
    # R3/D9: for a count trait, the delivery-grade count that gates the phenotype is the derived
    # criterion's tp/fp/fn (center-match for catkin), NOT AP@0.5 — kept alongside, clearly labeled.
    criterion = resolve_match_criterion(trait, per_image, iou_threshold=iou_threshold)
    if criterion["kind"] == "center_match":
        gc = governing_counts(per_image, criterion, conf_threshold=conf_threshold, max_dets=max_dets)
        result.update({
            "governing_counts": gc, "governing_criterion": criterion,
            "map50_role": "comparability_only",
            "iou_tp": m["tp"], "iou_fp": m["fp"], "iou_fn": m["fn"],
            "iou_precision": m["precision"], "iou_recall": m["recall"], "iou_f1": m["f1"],
            "tp": gc["tp"], "fp": gc["fp"], "fn": gc["fn"],
            "precision": gc["precision"], "recall": gc["recall"], "f1": gc["f1"],
        })
    out = Path(output_dir) / "test_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    result["results_path"] = str(out)
    return result
