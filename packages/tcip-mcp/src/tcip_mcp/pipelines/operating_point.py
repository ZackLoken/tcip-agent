"""Resolve the detection operating point (conf/NMS/max_dets/tile) per dataset, at runtime.

This is the single place all four consumers — train-eval, test-eval, inference, export — get the
operating point, so the same model + images can't yield different counts by entry door (the audit's
divergent-defaults bug). The confidence threshold is a *calibration* param: derived by a center-match
count-unbiased sweep over a labeled subset and validated on a held-out split, or carried as
``validated=false`` when no GT exists (never a frozen literal). See the scope doc and traits.py.
"""

from __future__ import annotations

import math
from typing import Any

from tcip_mcp.pipelines.derivations import derive_cross_tile_nms
from tcip_mcp.pipelines.resolution import (
    VALIDATED_FALSE,
    VALIDATED_HELD_OUT,
    ResolvedBundle,
    ResolvedParam,
    default,
    derived,
)
from tcip_mcp.pipelines.training.evaluation import (
    count_bias_at,
    gt_class_avg_size,
    pick_count_unbiased,
    pick_f1_max,
    sweep_operating_point,
)
from tcip_mcp.traits import get_trait

# Documented defaults for the non-count operating-point params (overridable / derivable).
_DEFAULT_TILED = True
_DEFAULT_TILE_SIZE = 640
_DEFAULT_CROSS_TILE_NMS = 0.5  # merge tile-seam duplicates; a distribution derivation can refine
_DEFAULT_MAX_DETS = 300
_DEFAULT_CONF_PLACEHOLDER = 0.5  # only ever consumed via unvalidated_value(); never trustworthy


def set_detector_operating_point(model: Any, *, score_thresh: float | None = None,
                                 nms_thresh: float | None = None,
                                 detections_per_img: int | None = None) -> dict:
    """Set the *in-model* torchvision thresholds so the operating point governs which boxes EXIST.

    Two-stage detectors keep them on ``roi_heads``; one-stage (FCOS/RetinaNet) on the detector itself.
    The composed model wraps the torchvision net as ``.detector``. Returns what was applied.
    (The audit's finding: these were never set, so a post-hoc score filter could never recover a box
    the model's internal ``score_thresh``/``detections_per_img`` had already discarded.)
    """
    det = getattr(model, "detector", model)
    target = getattr(det, "roi_heads", None) or det
    applied: dict = {}
    for attr, val in (("score_thresh", score_thresh), ("nms_thresh", nms_thresh),
                      ("detections_per_img", detections_per_img)):
        if val is not None and hasattr(target, attr):
            setattr(target, attr, val)
            applied[attr] = val
    return applied


def records_over_loader(model: Any, loader: Any, device: Any, task: str) -> list[dict]:
    """One unfiltered model pass -> per-image COCO records (boxes + scores) for a conf sweep.

    Set the in-model score threshold low first (via ``set_detector_operating_point``) so hesitant
    detections survive to be swept.
    """
    import torch

    from tcip_mcp.pipelines.training.evaluation import records_from_detector

    include_masks = task == "instance_seg"
    stems = getattr(getattr(loader, "dataset", None), "stems", None)
    model.eval()
    records: list[dict] = []
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for img, tgt, out in zip(images, targets, outputs):
                rec = records_from_detector(tgt, out, width=img.shape[-1], height=img.shape[-2],
                                            include_masks=include_masks)
                idx = tgt.get("image_id")
                if stems is not None and isinstance(idx, int) and 0 <= idx < len(stems):
                    rec["image_id"] = stems[idx]  # globally-unique so cal/holdout overlap is detectable
                records.append(rec)
    return records


def _max_dets_from_density(records: list[dict], floor: int = 100) -> int:
    """A generous cap = ~1.5x the p99 GT objects-per-image, so dense scenes aren't truncated."""
    import numpy as np
    counts = [len(rec.get("gt", [])) for rec in records]
    if not counts:
        return _DEFAULT_MAX_DETS
    return max(floor, int(math.ceil(1.5 * float(np.quantile(counts, 0.99)))))


def resolve_operating_point(
    trait_name: str,
    *,
    dataset_hash: str | None,
    calibration_records: list[dict] | None = None,
    holdout_records: list[dict] | None = None,
    tile_size: int | None = None,
    tiled: bool | None = None,
    cross_tile_nms: float | None = None,
    max_dets: int | None = None,
) -> ResolvedBundle:
    """Resolve the operating point for (trait, dataset). Pure over records — callers pass the model
    pass output; ``records_over_loader`` produces it. ``tile_size`` may be model-derived (imgsz)."""
    trait = get_trait(trait_name)
    params: dict[str, ResolvedParam] = {}

    # --- conf: the count operating point (calibration) ---
    if calibration_records:
        tol = 0.5 * gt_class_avg_size(calibration_records)  # derived tolerance (half class avg size)
        cal_sweep = sweep_operating_point(calibration_records, tolerance=tol)
        conf = pick_count_unbiased(cal_sweep) if trait.count_objective == "count_unbiased" else pick_f1_max(cal_sweep)
        conf = _DEFAULT_CONF_PLACEHOLDER if conf is None else conf
        if holdout_records:
            # Disjointness can only be proven from image_ids, so fail closed (not disjoint) when
            # either set has none — else the same records passed as cal+holdout look validated.
            cal_ids = {r["image_id"] for r in calibration_records if "image_id" in r}
            hold_ids = {r["image_id"] for r in holdout_records if "image_id" in r}
            disjoint = bool(cal_ids) and bool(hold_ids) and not (cal_ids & hold_ids)
            hold_tol = 0.5 * gt_class_avg_size(holdout_records)
            hold_sweep = sweep_operating_point(holdout_records, tolerance=hold_tol)
            hb = count_bias_at(hold_sweep, conf)  # bias on the holdout at the calibration-chosen conf
            passed = disjoint and hb is not None and abs(hb["count_bias_mean"]) <= trait.count_bias_tolerance
            sweep_data = {"calibration": cal_sweep, "f1_max_conf": pick_f1_max(cal_sweep),
                          "holdout_bias": hb, "count_bias_tolerance": trait.count_bias_tolerance,
                          "disjoint": disjoint, "passed_holdout": passed}
            # validated only if the holdout is disjoint AND the bias passed there — not merely because
            # a holdout was supplied. Reference here is the annotations, not truth (bounded by them).
            validated = VALIDATED_HELD_OUT if passed else VALIDATED_FALSE
        else:
            sweep_data = {"calibration": cal_sweep, "note": "calibrated but not held-out-measured"}
            validated = VALIDATED_FALSE
        params["conf"] = derived("conf", float(conf), derivation_class="calibration",
                                 derived_from="count-unbiased center-match sweep",
                                 validated_vs_gt=validated, dataset_scoped=True,
                                 dataset_hash=dataset_hash, sweep=sweep_data)
        if max_dets is None:
            max_dets = _max_dets_from_density(calibration_records)
    else:
        # No GT for this dataset: cannot calibrate. Carry an unvalidated placeholder (un-shippable
        # via the firewall) — no valley heuristic, no chosen value dressed as trustworthy.
        params["conf"] = derived("conf", _DEFAULT_CONF_PLACEHOLDER, derivation_class="calibration",
                                 derived_from="no GT for this dataset; unvalidated placeholder",
                                 validated_vs_gt=VALIDATED_FALSE, dataset_scoped=True, dataset_hash=dataset_hash)

    # --- deterministic / distribution / documented-default params ---
    params["tile_size"] = (
        derived("tile_size", int(tile_size), derivation_class="deterministic",
                derived_from="model imgsz / persisted training geometry")
        if tile_size else default("tile_size", _DEFAULT_TILE_SIZE)
    )
    params["tiled"] = default("tiled", _DEFAULT_TILED if tiled is None else bool(tiled))
    # cross_tile_nms: an explicit override wins and is stamped as such; otherwise derive it from the
    # calibration GT's neighbor-IoU distribution; failing that (no GT / no genuine overlaps) an honest
    # default — never a derivation label on a number no derivation produced.
    if cross_tile_nms is not None:
        params["cross_tile_nms"] = ResolvedParam(
            "cross_tile_nms", float(cross_tile_nms), source="explicit",
            derivation_class="distribution", derived_from="caller override")
    else:
        nms = None
        if calibration_records:
            nms = derive_cross_tile_nms([[a["bbox"] for a in rec.get("gt", [])]
                                         for rec in calibration_records])
        params["cross_tile_nms"] = (
            derived("cross_tile_nms", nms, derivation_class="distribution",
                    derived_from="GT neighbor-IoU distribution (p99 + margin)")
            if nms is not None
            else default("cross_tile_nms", _DEFAULT_CROSS_TILE_NMS, derivation_class="distribution")
        )
    params["max_dets"] = (
        derived("max_dets", int(max_dets), derivation_class="distribution",
                derived_from="~1.5x p99 GT objects/image")
        if max_dets is not None else default("max_dets", _DEFAULT_MAX_DETS)
    )
    return ResolvedBundle(trait=trait_name, dataset_hash=dataset_hash, params=params)
