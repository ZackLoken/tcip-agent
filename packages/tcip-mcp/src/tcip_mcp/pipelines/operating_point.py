"""Resolve the detection operating point (conf/NMS/max_dets/tile) per dataset, at runtime.

This is the single place all four consumers — train-eval, test-eval, inference, export — get the
operating point, so the same model + images can't yield different counts by entry door (the audit's
divergent-defaults bug). The confidence threshold is a *calibration* param: derived by a center-match
count-unbiased sweep over a reference sized to the trait, and validated on a disjoint held-out split of
that reference — GT annotations (``validated_held_out``) OR a breeder-confirmed sample of the model's
own outputs (``review_confirmed``), the same gate either way — or carried as ``validated=false`` when
no reference exists (never a frozen literal). See the scope doc and traits.py.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from tcip_mcp.pipelines.derivations import derive_cross_tile_nms
from tcip_mcp.pipelines.resolution import (
    DEFAULT_CONF,
    DEFAULT_MAX_DETS,
    DEFAULT_NMS_IOU,
    DEFAULT_TILE_SIZE,
    DEFAULT_TILED,
    VALIDATED_FALSE,
    VALIDATED_HELD_OUT,
    VALIDATED_REVIEW_CONFIRMED,
    VALIDATED_SHIPPABLE,
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

# The non-count operating-point fallbacks all resolve to resolution.py's single source of truth
# (DEFAULT_TILED / DEFAULT_TILE_SIZE / DEFAULT_NMS_IOU / DEFAULT_MAX_DETS / DEFAULT_CONF) so the same
# model+images can't give a different count by entry door. cross_tile_nms shares the NMS-IoU knob (a
# distribution derivation refines it); the conf placeholder is only ever read via unvalidated_value().


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


def _min_dt_score(records: list[dict]) -> float | None:
    """Lowest detection score across a reference, or None if it holds no detections."""
    scores = [d["score"] for rec in records for d in rec.get("dt", []) if "score" in d]
    return min(scores) if scores else None


def _conf_censored(records: list[dict] | None, display_floor: float) -> bool:
    """True when a reference's detections were filtered above the calibration floor.

    The count-unbiased sweep is only trustworthy when the reference includes the low-conf tail —
    predictions generated at/below the calibration floor (``_calibrate_operating_point`` floors the
    detector to ``score_thresh=0.01`` for exactly this reason). A reference whose lowest detection
    score sits at or above the display conf floor (e.g. predictions a human reviewed, staged at
    ``DEFAULT_CONF``) is truncated: the sweep can't reach the boxes a lower conf would recover, so its
    count-unbiased point and held-out bias are not trustworthy and must not stamp a ``validated`` claim.
    """
    lo = _min_dt_score(records or [])
    return lo is not None and lo >= display_floor


def _max_dets_from_density(records: list[dict], floor: int = 100) -> int:
    """A generous cap = ~1.5x the p99 GT objects-per-image, so dense scenes aren't truncated."""
    import numpy as np
    counts = [len(rec.get("gt", [])) for rec in records]
    if not counts:
        return DEFAULT_MAX_DETS
    return max(floor, int(math.ceil(1.5 * float(np.quantile(counts, 0.99)))))


def _record_content_hash(rec: dict) -> str | None:
    """Content identity of one record's GT — ``(width, height, sorted (category_id, bbox))`` —
    ignoring ``image_id``. ``None`` for empty GT: a shared negative must not trip the
    content-overlap guard (a negative is first-class per CLAUDE.md; empty-GT records hash
    identically across cal/holdout by construction and that is expected, not leakage).
    """
    gt = rec.get("gt") or []
    if not gt:
        return None
    key = (
        int(rec.get("width", 0)), int(rec.get("height", 0)),
        tuple(sorted(
            (int(a.get("category_id", 0)), tuple(round(float(v), 6) for v in a.get("bbox", [])))
            for a in gt
        )),
    )
    return hashlib.sha256(json.dumps(key).encode("utf-8")).hexdigest()[:16]


def _content_overlap(cal_records: list[dict], hold_records: list[dict]) -> dict:
    """Whether the holdout's GT content is (fully) cloned from calibration's.

    ``duplicated`` fires only on full containment — holdout's non-empty content-hash set is a
    subset of calibration's — not on any overlap, so a holdout that merely shares ONE image with
    calibration (partially overlapping content) is not penalized; only a holdout whose entire
    content already exists in calibration (a byte-identical or re-labeled-copy holdout, unable to
    function as an independent check) is refused.
    """
    cal_hashes = {h for h in (_record_content_hash(r) for r in cal_records) if h is not None}
    hold_hashes = {h for h in (_record_content_hash(r) for r in hold_records) if h is not None}
    if not hold_hashes:
        return {"content_overlap_frac": 0.0, "duplicated": False}
    frac = len(hold_hashes & cal_hashes) / len(hold_hashes)
    return {"content_overlap_frac": frac, "duplicated": hold_hashes.issubset(cal_hashes)}


_UNRESOLVABLE_TRAIN_DISJOINTNESS = {
    "checked": False, "unresolvable": True, "leaked_groups": [], "leaked_stems": [],
    "group_check": None,
}


def _train_disjointness(experiment_id: str | None, cal_ids: set, hold_ids: set) -> dict:
    """Whether the cal/holdout images were also in the checkpoint's own training split (K1).

    ``experiment_id is None`` -> a foreign/unregistered checkpoint with no known training
    provenance to check against; allowed through per the owner decision (K1 design) — only a
    *known* ``experiment_id`` whose provenance can't be read/reconstructed fails closed.

    Two genuinely-unresolvable cases stay ``unresolvable: True`` (fail-closed, blocks ``passed``
    in ``resolve_operating_point``): ``split.json`` is missing/unreadable, or it IS readable but
    records no training stems at all — there is nothing here to check against, not even the
    stem-level fallback below.

    Otherwise (``split.json`` readable, with real training stems) this never blanket-refuses just
    because a GROUP policy can't be resolved — that was finding 1's headline bug: it permanently
    blocked the explicit-``val_images_dir`` route (``group_by="external"``, no computed grouping)
    and the ``group_key_map`` route (the map was never persisted) even though both are legitimate,
    disjoint training regimes. Instead, group-level resolution is attempted per stem:

      - a named, recognized strategy (``tile_prefix``/``stem``) resolves every stem, as before.
      - ``group_by == "explicit_map"`` resolves via the persisted ``group_key_map`` (K1 finding 1)
        for whichever stems it actually covers; stems it doesn't cover are treated as unresolvable
        for that stem only, not a blanket failure.
      - anything else (``"external"``, an unrecognized string, a missing field) resolves nothing
        at the group level.

    Every stem the group check couldn't cover falls back to the free, policy-independent check
    that's always available regardless of grouping: exact stem-set overlap between the training
    run's own stems and the calibration/holdout stem set (``leaked_stems``). ``group_check``
    records how much of the check was group-level: ``"performed"`` (every stem grouped),
    ``"partial"`` (some stems fell back to exact-stem), or ``"not_performed"`` (no group policy
    resolved at all, wholly exact-stem). A leak found by EITHER mechanism blocks ``passed``.
    """
    if experiment_id is None:
        return {"checked": False, "unresolvable": False, "leaked_groups": [], "leaked_stems": [],
                "group_check": None}
    from tcip_mcp.project_paths import project_root

    split_path = project_root() / ".tcip" / "experiments" / experiment_id / "split.json"
    if not split_path.is_file():
        return dict(_UNRESOLVABLE_TRAIN_DISJOINTNESS)
    try:
        split = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(_UNRESOLVABLE_TRAIN_DISJOINTNESS)

    train_stems = split.get("train") or []
    if not train_stems:
        # Nothing recorded to check against at all — not even the stem-overlap fallback has
        # anything to compare, so this is genuinely unresolvable, not merely ungrouped.
        return dict(_UNRESOLVABLE_TRAIN_DISJOINTNESS)

    from tcip_mcp.pipelines.data.splits import GROUP_KEY_FNS, resolve_group_key_fn

    cal_hold_stems = sorted(cal_ids | hold_ids)
    group_by = split.get("group_by")
    persisted_map = split.get("group_key_map") if group_by == "explicit_map" else None

    covered_train: list[str] = []
    covered_cal_hold: list[str] = []
    key_fn = None
    if persisted_map:
        covered_train = [s for s in train_stems if s in persisted_map]
        covered_cal_hold = [s for s in cal_hold_stems if s in persisted_map]
        if covered_train or covered_cal_hold:
            # The map, when present, always wins over group_by per resolve_group_key_fn's own
            # contract — the "tile_prefix" here is an inert placeholder, never consulted.
            key_fn = resolve_group_key_fn("tile_prefix", covered_train + covered_cal_hold,
                                          group_key_map=persisted_map)
    elif group_by and group_by in GROUP_KEY_FNS:
        key_fn = GROUP_KEY_FNS[group_by]
        covered_train, covered_cal_hold = list(train_stems), cal_hold_stems

    leaked_groups: list[str] = []
    if key_fn is not None:
        train_groups = {key_fn(s) for s in covered_train}
        cal_hold_groups = {key_fn(s) for s in covered_cal_hold}
        leaked_groups = sorted(train_groups & cal_hold_groups)

    uncovered_train = sorted(set(train_stems) - set(covered_train))
    uncovered_cal_hold = sorted(set(cal_hold_stems) - set(covered_cal_hold))
    leaked_stems = sorted(set(uncovered_train) & set(uncovered_cal_hold))

    if key_fn is None:
        group_check = "not_performed"
    elif uncovered_train or uncovered_cal_hold:
        group_check = "partial"
    else:
        group_check = "performed"

    return {
        "checked": True,
        "unresolvable": False,
        "leaked_groups": leaked_groups,
        "leaked_stems": leaked_stems,
        "group_check": group_check,
    }


def attach_split_policy_provenance(bundle: ResolvedBundle, locked: dict) -> None:
    """Copy the locked cal/holdout split's resolved policy + identity onto the conf param's sweep
    (K1 finding 5), so the operating-point provenance bundle is self-contained — a caller can see
    WHY particular ids ended up on which side without a separate lookup of the
    ``.tcip/artifacts/cal_holdout_split_<hash>.json`` lock file. Also carries any
    ``policy_divergence`` / ``unlocked_stems`` the lock resolution reported, so a declared-but-not-
    applied policy (a lock already existed with a different seed/ratio/grouping) is visible in the
    result, not only in a server log line.

    In-place: ``ResolvedParam`` is a frozen dataclass, but the ``sweep`` dict it holds is an
    ordinary mutable dict, so mutating its contents (never reassigning the attribute) is safe. A
    no-op when the bundle has no calibrated ``conf`` sweep to attach to (e.g. no GT at all).
    """
    conf = bundle.params.get("conf")
    if conf is None or conf.sweep is None:
        return
    conf.sweep["split_policy"] = {
        "group_by": locked.get("group_by"), "group_key_map": locked.get("group_key_map"),
        "seed": locked.get("seed"), "holdout_ratio": locked.get("holdout_ratio"),
        "identity_hash": locked.get("identity_hash"),
    }
    if locked.get("policy_divergence"):
        conf.sweep["split_policy_divergence"] = locked["policy_divergence"]
    if locked.get("unlocked_stems"):
        conf.sweep["split_unlocked_stems"] = locked["unlocked_stems"]


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
    validated_reference: str = VALIDATED_HELD_OUT,
    experiment_id: str | None = None,
) -> ResolvedBundle:
    """Resolve the operating point for (trait, dataset). Pure over records — callers pass the model
    pass output; ``records_over_loader`` produces it. ``tile_size`` may be model-derived (imgsz).

    ``validated_reference`` is the stamp a *passing* held-out gate earns: ``validated_held_out`` when
    the records came from GT annotations (default), ``review_confirmed`` when they were reconstructed
    from a breeder-confirmed sample of the model's own outputs (feedback.review_calibration). Both
    references pass the SAME disjoint + count-bias gate here — the stamp only records which one it was.

    ``experiment_id`` (K1) is the checkpoint's own training-run id, if known — it gates the SAME
    held-out pass on train-disjointness (the cal/holdout images must not also be in that run's
    training split); ``None`` (a foreign/unregistered checkpoint) skips the check rather than
    failing closed, per the owner decision that only a *known* run whose provenance can't be
    resolved should refuse.
    """
    if validated_reference not in VALIDATED_SHIPPABLE:
        raise ValueError(f"validated_reference must be one of {VALIDATED_SHIPPABLE}, got {validated_reference!r}")
    trait = get_trait(trait_name)
    review = validated_reference == VALIDATED_REVIEW_CONFIRMED
    conf_derived_from = ("count-unbiased center-match sweep over review verdicts"
                         if review else "count-unbiased center-match sweep")
    params: dict[str, ResolvedParam] = {}

    # --- conf: the count operating point (calibration) ---
    if calibration_records:
        tol = trait.localization_tolerance_frac * gt_class_avg_size(calibration_records)  # spec owns the "half"
        cal_sweep = sweep_operating_point(calibration_records, tolerance=tol)
        conf = pick_count_unbiased(cal_sweep) if trait.count_objective == "count_unbiased" else pick_f1_max(cal_sweep)
        conf = DEFAULT_CONF if conf is None else conf
        # conf-censoring guard: a count-unbiased 'validated' claim is only honest if the reference
        # shows the low-conf tail; a display-filtered reference (min score >= the display floor) is
        # truncated and cannot be stamped validated — carry it as an unvalidated placeholder instead.
        censored = _conf_censored(calibration_records, DEFAULT_CONF)
        if holdout_records:
            # Disjointness can only be proven from image_ids, so fail closed (not disjoint) when
            # either set has none — else the same records passed as cal+holdout look validated.
            cal_ids = {r["image_id"] for r in calibration_records if "image_id" in r}
            hold_ids = {r["image_id"] for r in holdout_records if "image_id" in r}
            disjoint = bool(cal_ids) and bool(hold_ids) and not (cal_ids & hold_ids)
            censored = censored or _conf_censored(holdout_records, DEFAULT_CONF)
            hold_tol = trait.localization_tolerance_frac * gt_class_avg_size(holdout_records)
            hold_sweep = sweep_operating_point(holdout_records, tolerance=hold_tol)
            hb = count_bias_at(hold_sweep, conf)  # bias on the holdout at the calibration-chosen conf
            # content-overlap gate (K1): a holdout whose GT content is fully cloned from calibration
            # (same boxes, different image_id) can't function as an independent check.
            content = _content_overlap(calibration_records, holdout_records)
            # train-disjointness gate (K1): the cal/holdout images must not also be in the producing
            # checkpoint's OWN training split, or the "held-out" bias check is measured partly on
            # data the model already trained on.
            td = _train_disjointness(experiment_id, cal_ids, hold_ids)
            train_provenance_blocked = (
                td["unresolvable"] or bool(td["leaked_groups"]) or bool(td["leaked_stems"])
            )
            passed = (disjoint and not censored and hb is not None
                      and abs(hb["count_bias_mean"]) <= trait.count_bias_tolerance
                      and not content["duplicated"] and not train_provenance_blocked)
            sweep_data = {"calibration": cal_sweep, "f1_max_conf": pick_f1_max(cal_sweep),
                          "holdout_bias": hb, "count_bias_tolerance": trait.count_bias_tolerance,
                          "disjoint": disjoint, "conf_censored": censored, "passed_holdout": passed,
                          "content_overlap_frac": content["content_overlap_frac"],
                          "content_duplicated": content["duplicated"],
                          "train_disjointness": td,
                          "calibration_image_ids": sorted(cal_ids), "holdout_image_ids": sorted(hold_ids)}
            # validated only if the holdout is disjoint AND uncensored AND the bias passed there AND
            # its content isn't a clone of calibration AND train-provenance isn't blocked — not merely
            # because a holdout was supplied. Reference here is the annotations/verdicts, not truth;
            # the stamp records which reference (GT vs review-confirmed) cleared the gate.
            validated = validated_reference if passed else VALIDATED_FALSE
        else:
            sweep_data = {"calibration": cal_sweep, "conf_censored": censored,
                          "note": "calibrated but not held-out-measured"}
            validated = VALIDATED_FALSE
        params["conf"] = derived("conf", float(conf), derivation_class="calibration",
                                 derived_from=conf_derived_from,
                                 validated_vs_gt=validated, dataset_scoped=True,
                                 dataset_hash=dataset_hash, sweep=sweep_data)
        if max_dets is None:
            max_dets = _max_dets_from_density(calibration_records)
    else:
        # No GT for this dataset: cannot calibrate. Carry an unvalidated placeholder (un-shippable
        # via the firewall) — no valley heuristic, no chosen value dressed as trustworthy.
        params["conf"] = derived("conf", DEFAULT_CONF, derivation_class="calibration",
                                 derived_from="no GT for this dataset; unvalidated placeholder",
                                 validated_vs_gt=VALIDATED_FALSE, dataset_scoped=True, dataset_hash=dataset_hash)

    # --- deterministic / distribution / documented-default params ---
    params["tile_size"] = (
        derived("tile_size", int(tile_size), derivation_class="deterministic",
                derived_from="model imgsz / persisted training geometry")
        if tile_size else default("tile_size", DEFAULT_TILE_SIZE)
    )
    params["tiled"] = default("tiled", DEFAULT_TILED if tiled is None else bool(tiled))
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
            else default("cross_tile_nms", DEFAULT_NMS_IOU, derivation_class="distribution")
        )
    params["max_dets"] = (
        derived("max_dets", int(max_dets), derivation_class="distribution",
                derived_from="~1.5x p99 GT objects/image")
        if max_dets is not None else default("max_dets", DEFAULT_MAX_DETS)
    )
    return ResolvedBundle(trait=trait_name, dataset_hash=dataset_hash, params=params)
