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
import statistics
from typing import Any, Callable

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
    classes_with_evidence,
    gt_class_avg_size,
    pick_count_unbiased,
    pick_f1_max,
    sweep_operating_point,
)
from tcip_mcp.traits import COUNT_OBJECTIVES, COUNT_UNBIASED, DETECTION_F1, PRESENCE, get_trait

# The non-count operating-point fallbacks all resolve to resolution.py's single source of truth
# (DEFAULT_TILED / DEFAULT_TILE_SIZE / DEFAULT_NMS_IOU / DEFAULT_MAX_DETS / DEFAULT_CONF) so the same
# model+images can't give a different count by entry door. cross_tile_nms shares the NMS-IoU knob (a
# distribution derivation refines it); the conf placeholder is only ever read via unvalidated_value().

# Count-objective -> (picker, derivation label). One registry, not two hardcoded lists to keep in
# sync (Fix E): the KEYS are ``traits.COUNT_OBJECTIVES`` (torch-free, so ``traits._spec_from_config``
# validates a config-authored count_objective against that set directly, never importing this
# torch-heavy module just to check a name) — this dict only adds the picker/label each objective
# resolves to. The label stamped on ``conf`` is derived from whichever picker actually ran
# (pick-then-label), never from the reference type alone. PRESENCE deliberately shares DETECTION_F1's
# F1-max picker/label: presence is a per-object find/no-find call, exactly what F1 (harmonic
# precision/recall) measures on matching quality — unlike COUNT_UNBIASED's sum-agreement objective
# (the phenotype is a count), which needs its own picker. An unregistered objective (only reachable
# via a code-authored TraitSpec, since config-authored specs are validated against COUNT_OBJECTIVES)
# falls back to the F1-max entry, same as before.
COUNT_OBJECTIVE_PICKERS: dict[str, tuple[Callable[[dict], float | None], str]] = {
    COUNT_UNBIASED: (pick_count_unbiased, "count-unbiased center-match sweep"),
    DETECTION_F1: (pick_f1_max, "F1-max center-match sweep"),
    PRESENCE: (pick_f1_max, "F1-max center-match sweep"),
}
assert set(COUNT_OBJECTIVE_PICKERS) == COUNT_OBJECTIVES, (
    "COUNT_OBJECTIVE_PICKERS must cover exactly traits.COUNT_OBJECTIVES — the single source of "
    "truth traits._spec_from_config validates config-authored objectives against")

# Fix C's one-sided confidence multiplier for the mean+SE equivalence criterion (~95%) — a stated
# CV-derivation convention, not a breeder-semantics decision, so it lives here as a named constant
# rather than buried in a formula.
_EQUIVALENCE_Z = 1.645

# K3's compensating-error floor, interim default (stage-6 review Finding B — corrected from the
# prior framing, which wrongly called this "a cited convention, like _EQUIVALENCE_Z"). Landis &
# Koch (1977)'s kappa scale is a descriptive LABEL for a magnitude, not a distributional fact
# entailed by a stated confidence level the way _EQUIVALENCE_Z's z-score is — "how much classifier
# agreement is enough to trust this trait's phenotype" is measurement semantics, the domain expert's
# call, the same as `TraitSpec.count_error_tolerance`. This value is a PROVISIONAL, platform-chosen
# placeholder used only when a trait hasn't authored `TraitSpec.classifier_agreement_floor` (None) —
# see that field's docstring. kappa==0 is exactly chance agreement; a floor there alone admits a
# classifier whose errors are compensating (net count-bias ~0) but substantial (round 1's finding).
_PROVISIONAL_KAPPA_FLOOR = 0.41


def _bias_equivalence_ok(mean: float, std: float, n: int, tolerance: float) -> bool:
    """Mean-plus-SE equivalence test: is a bias measured across ``n`` per-image samples small
    enough, relative to its own sampling uncertainty, to conclude equivalence with zero? Not a bare
    mean check — this degrades correctly at small ``n`` (SE grows, so less evidence is HARDER to
    pass, never easier). Shared by the detection path (:func:`resolve_operating_point`) and the
    classifier path (:func:`resolve_classifier_operating_point`) so both judge count bias in the
    same statistical shape and unit — a per-image mean — never two independently-derived criteria
    that happen to share a name and a tolerance field.
    """
    if n == 0:
        return False
    se = std / math.sqrt(n)
    return abs(mean) + _EQUIVALENCE_Z * se <= tolerance


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


def _current_detections_cap(model: Any) -> int | None:
    """The in-model ``detections_per_img`` a model is CURRENTLY set to, or ``None`` if unset.

    Read, not derived — whatever ``set_detector_operating_point`` last applied (or the framework
    default if nothing was ever set). Used only to stamp Fix K's non-gating cap-saturation signal
    at record-generation time, when the cap is actually known.
    """
    det = getattr(model, "detector", model)
    target = getattr(det, "roi_heads", None) or det
    return getattr(target, "detections_per_img", None)


def records_over_loader(model: Any, loader: Any, device: Any, task: str) -> list[dict]:
    """One unfiltered model pass -> per-image COCO records (boxes + scores) for a conf sweep.

    Set the in-model score threshold low first (via ``set_detector_operating_point``) so hesitant
    detections survive to be swept.
    """
    import torch

    from tcip_mcp.pipelines.training.evaluation import records_from_detector

    include_masks = task == "instance_seg"
    stems = getattr(getattr(loader, "dataset", None), "stems", None)
    cap = _current_detections_cap(model)
    model.eval()
    records: list[dict] = []
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for img, tgt, out in zip(images, targets, outputs):
                rec = records_from_detector(tgt, out, width=img.shape[-1], height=img.shape[-2],
                                            include_masks=include_masks, detections_cap=cap)
                idx = tgt.get("image_id")
                if stems is not None and isinstance(idx, int) and 0 <= idx < len(stems):
                    rec["image_id"] = stems[idx]  # globally-unique so cal/holdout overlap is detectable
                records.append(rec)
    return records


def _min_dt_score(records: list[dict]) -> float | None:
    """Lowest detection score across a reference, or None if it holds no detections."""
    scores = [d["score"] for rec in records for d in rec.get("dt", []) if "score" in d]
    return min(scores) if scores else None


def _cap_saturated_frac(records: list[dict] | None) -> float | None:
    """Fraction of records whose raw detection count hit the model's applied per-image cap.

    Fix K, non-gating provenance only: a per-image ``cap_hit`` flag is stamped by
    ``records_from_detector``/``records_over_loader`` when the cap was known at generation time, and
    (K10) by ``run_full_frame_evaluation`` for its own tiled-and-reconstructed pass. Records built
    some other way carry none and are excluded from the fraction entirely, not counted as an
    unsaturated 0 — ``None`` when nothing here carries the flag.
    """
    hits = [r["cap_hit"] for r in (records or []) if "cap_hit" in r]
    return (sum(hits) / len(hits)) if hits else None


def _conf_censored(chosen_conf: float, staged_conf_floor: float | None) -> bool:
    """True when the picked conf sits AT OR BELOW the floor the reference was staged/filtered at.

    The count-unbiased sweep is only trustworthy when the reference includes the low-conf tail below
    the picked conf. A conf picked strictly ABOVE the staging floor is fully supported by the
    reference — every surviving detection with ``score >= conf`` genuinely survived the floor's own
    filter, so the sweep saw everything it needed to. A conf at or below the floor means the sweep
    could not see whether an even-lower conf would have done better, so the pick is untrustworthy.
    ``staged_conf_floor is None`` (the caller made no assertion of what floor the reference was
    generated at) is always censored — there is nothing here to reconcile the pick against, so
    validation fails closed rather than trusting an unstated assumption.
    """
    return staged_conf_floor is None or chosen_conf <= staged_conf_floor


def _floor_mismatch(records: list[dict] | None, staged_conf_floor: float | None) -> bool:
    """True when the reference's OBSERVED lowest score is inconsistent with the ASSERTED floor.

    A material gap (>0.05) between what the caller asserts the reference was staged at and what the
    data actually shows is itself evidence the assertion is wrong, or that something else (a stale
    bucket, cap-trimmed tiles, a bespoke caller) truncated the reference after generation — a
    distinct failure mode from ``_conf_censored`` (which only compares the picked conf, not the
    reference's own data, against the asserted floor).
    """
    if staged_conf_floor is None:
        return False
    observed_min = _min_dt_score(records or [])
    return observed_min is not None and observed_min > staged_conf_floor + 0.05


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

    Known residual (stage-6 review N7, not fixed here — a design tradeoff to revisit, not an
    oversight): the classifier path calls this at INSTANCE granularity (one record per matched
    detection, via ``resolve_classifier_operating_point``'s ``_as_record``), where full-containment
    was designed for IMAGE-granularity records. This cuts both ways, not just one: (a) one extra,
    genuinely-independent instance in an otherwise wholesale-cloned holdout defeats the subset check
    — real duplication ESCAPES, the rail too permissive; and (b) two genuinely independent images
    that happen to share dimensions and produce a detection at the same pixel coordinates (plausible
    for a fixed-camera rig or center-cropped tiles) hash identically even though nothing was cloned —
    a valid, independent holdout can be FLAGGED duplicated, the rail too strict (the CLAUDE.md "a
    rail must admit valid work" failure mode, not just the more obvious permissive one).
    ``content_overlap_frac`` is computed but never itself gated (only the boolean ``duplicated`` is).
    A tighter classifier-path criterion (e.g. gating on the fraction directly, or grouping instances
    back to per-image records) is real follow-up work, not attempted in this pass.
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
    tile_size_source: str = "default",
    tiled: bool | None = None,
    tiled_source: str = "default",
    cross_tile_nms: float | None = None,
    max_dets: int | None = None,
    validated_reference: str = VALIDATED_HELD_OUT,
    experiment_id: str | None = None,
    staged_conf_floor: float | None = None,
    adjudication_covered: Callable[[dict], bool] | None = None,
) -> ResolvedBundle:
    """Resolve the operating point for (trait, dataset). Pure over records — callers pass the model
    pass output; ``records_over_loader`` produces it. ``tile_size`` may be model-derived (imgsz).

    ``tile_size_source``/``tiled_source`` (K10 finding 3) are the caller's own resolution of whether
    each value was an explicit override, derived from the checkpoint's persisted training geometry,
    or a documented default — not inferred here from mere truthiness. A truthy ``tile_size`` is not
    proof of derivation: a caller with no persisted geometry and no explicit value still passes a
    concrete fallback number, and without the source travelling separately this function used to
    stamp that fabricated value ``"derived"`` unconditionally.

    ``validated_reference`` is the stamp a *passing* held-out gate earns: ``validated_held_out`` when
    the records came from GT annotations (default), ``review_confirmed`` when they were reconstructed
    from a breeder-confirmed sample of the model's own outputs (feedback.review_calibration). Both
    references pass the SAME disjoint + count-bias gate here — the stamp only records which one it was.

    ``experiment_id`` (K1) is the checkpoint's own training-run id, if known — it gates the SAME
    held-out pass on train-disjointness (the cal/holdout images must not also be in that run's
    training split); ``None`` (a foreign/unregistered checkpoint) skips the check rather than
    failing closed, per the owner decision that only a *known* run whose provenance can't be
    resolved should refuse.

    ``staged_conf_floor`` (Fix D) is the floor the reference's predictions were actually generated /
    filtered at — a caller-supplied FACT, not inferred from the reference's own scores. ``None``
    (the caller asserted nothing) fails closed: see ``_conf_censored``. The GT/calibration callers
    thread the value ``set_detector_operating_point`` actually applied; the review path has no floor
    threaded here yet (see ``feedback.review_calibration.resolve_operating_point_from_review`` for
    that seam).

    ``adjudication_covered`` (Fix C item 5 / Fix H): an optional per-record predicate — when given,
    it is a GATE, not a filter: EVERY calibration and holdout record must satisfy it, or the whole
    reference is refused (``insufficient_adjudication_coverage``), before any bias/dispersion
    statistic is computed. Records are never silently dropped and re-measured on the survivors —
    stage-6 review of an earlier draft found that a per-record filter here is a fail-open: the
    excluded set is correlated with the very quantity being measured (an image survives only if a
    miss was attested), so a biased population could earn a clean stamp on a favorable subsample
    while the full reviewed population was off by several times the trait tolerance. ``None`` (the
    default; every GT-path caller) applies no requirement — correct there, since a labeled record is
    inherently adjudication-covered. ``feedback.review_calibration.resolve_operating_point_from_review``
    passes the real Fix H predicate (per-image FN-adjudication coverage).
    """
    if validated_reference not in VALIDATED_SHIPPABLE:
        raise ValueError(f"validated_reference must be one of {VALIDATED_SHIPPABLE}, got {validated_reference!r}")
    trait = get_trait(trait_name)
    review = validated_reference == VALIDATED_REVIEW_CONFIRMED
    # Fix H gate (NOT a filter — see the docstring above for why a filter fails open): every record
    # must satisfy the predicate or the whole reference is refused, unfiltered, further down.
    adjudication_ok = adjudication_covered is None or (
        all(adjudication_covered(r) for r in (calibration_records or []))
        and all(adjudication_covered(r) for r in (holdout_records or []))
    )
    picker, base_label = COUNT_OBJECTIVE_PICKERS.get(
        trait.count_objective, (pick_f1_max, "F1-max center-match sweep"))
    conf_derived_from = base_label + (" over review verdicts" if review else "")
    params: dict[str, ResolvedParam] = {}

    # --- conf: the count operating point (calibration) ---
    if calibration_records:
        tol = trait.localization_tolerance_frac * gt_class_avg_size(calibration_records)  # spec owns the "half"
        cal_sweep = sweep_operating_point(calibration_records, tolerance=tol)
        conf = picker(cal_sweep)
        conf = DEFAULT_CONF if conf is None else conf
        # conf-censoring guard (Fix D): a count-unbiased 'validated' claim is only honest if the
        # picked conf sits strictly above the floor the reference was staged at — not merely if the
        # reference's own scores happen to look low (that predicate is unfalsifiable from a caller
        # who mis-asserts the floor; see _floor_mismatch for the reconciling check).
        censored = _conf_censored(conf, staged_conf_floor)
        floor_mismatch = _floor_mismatch(calibration_records, staged_conf_floor)
        if holdout_records:
            # Disjointness can only be proven from image_ids, so fail closed (not disjoint) when
            # either set has none — else the same records passed as cal+holdout look validated.
            cal_ids = {r["image_id"] for r in calibration_records if "image_id" in r}
            hold_ids = {r["image_id"] for r in holdout_records if "image_id" in r}
            disjoint = bool(cal_ids) and bool(hold_ids) and not (cal_ids & hold_ids)
            floor_mismatch = floor_mismatch or _floor_mismatch(holdout_records, staged_conf_floor)
            hold_tol = trait.localization_tolerance_frac * gt_class_avg_size(holdout_records)
            # Fix F: exact-conf evaluation, not a nearest-grid-point snap — an explicit single-point
            # conf_grid makes sweep_operating_point evaluate EXACTLY the conf that will ship, never
            # an approximation from the holdout's own independently-built grid (which need not
            # contain, or be anywhere near, the calibration-chosen conf).
            hold_sweep = sweep_operating_point(holdout_records, tolerance=hold_tol, conf_grid=[conf])
            hb = hold_sweep["curve"][0]  # the exact-conf holdout bias entry
            # The calibration side re-measured at the SHIPPED conf (not read off its own grid, which
            # need not contain it) — the only comparable basis for asking which classes the holdout
            # was actually able to check, below.
            cb = sweep_operating_point(calibration_records, tolerance=tol, conf_grid=[conf])["curve"][0]
            # content-overlap gate (K1): a holdout whose GT content is fully cloned from calibration
            # (same boxes, different image_id) can't function as an independent check.
            content = _content_overlap(calibration_records, holdout_records)
            # train-disjointness gate (K1): the cal/holdout images must not also be in the producing
            # checkpoint's OWN training split, or the "held-out" bias check is measured partly on
            # data the model already trained on.
            td = _train_disjointness(experiment_id, cal_ids, hold_ids)

            # Fix C item 1: positive-evidence, unconditional, stated per-side (not a union) — an
            # all-negative reference on either side can't validate a count operating point.
            cal_gt_count = sum(len(r.get("gt", [])) for r in calibration_records)
            hold_gt_count = sum(len(r.get("gt", [])) for r in holdout_records)
            # Fix C items 2-4: the mean+SE equivalence/CI criterion, replacing a bare mean check —
            # degrades correctly at small n (SE grows, so less evidence is HARDER to pass, not
            # easier) and needs no second, unrelated tolerance constant.
            count_bias_ok = _bias_equivalence_ok(
                hb["count_bias_mean"], hb["count_bias_std"], hb["n_images"], trait.count_bias_tolerance)
            # K4 #4: the pooled test above is blind to a per-class error. Its matcher ignores
            # category, so a detector that calls every class-A object class B scores tp-only with
            # zero bias, and one that over-detects A exactly as much as it under-detects B nets to
            # zero too — either way a phenotype built from per-class counts (an elongated FRACTION,
            # a per-class total) is wrong while the stamp says validated. So every class the holdout
            # carries must clear the SAME equivalence test at the same trait tolerance, in the same
            # per-image-mean unit, over the same images (a class absent from an image contributes a
            # zero bias there, exactly as the pooled term does). Which class is the trait's positive
            # one is deliberately not consulted: that needs a name->id registry read this does not
            # have, and requiring every class to be unbiased is the stronger claim anyway.
            per_class_bias_failures = sorted(
                cid for cid, s in hb["per_class"].items()
                if not _bias_equivalence_ok(s["count_bias_mean"], s["count_bias_std"],
                                            s["n_images"], trait.count_bias_tolerance))
            # ...and a class the holdout never carries is not a class that passed: its entry is all
            # zeros, so the test above reads bias 0.0 and says nothing. Stage-6 review reached the
            # very hole this gate exists to close through exactly that shape — the confused class
            # sits wholly in the calibration half, so every per-class entry the gate can see reads
            # clean and the stamp lands anyway (reproduced in
            # `test_a_class_the_holdout_never_carries_cannot_be_validated_by_its_absence`). So every
            # class the calibration reference actually evidences at the shipped conf must be
            # evidenced in the holdout too, the same positive-evidence rule (never an inference from
            # absence) the per-side `insufficient_*_gt` conjuncts already apply to the pooled count.
            holdout_missing_classes = sorted(classes_with_evidence(cb) - classes_with_evidence(hb))
            # Fix B item 3: a real (still-categorical, not tuned) match-quality floor — mathematically
            # equivalent to tp > 0 (precision/recall are both 0 exactly when tp is 0), so it catches
            # the fully-degenerate case while count bias vanishes; it does NOT discriminate a trivial
            # 1-of-many match from a near-complete one (a continuous quality criterion is future
            # work, not part of this cluster — see the design doc).
            localization_floor_ok = hb["recall"] > 0 and hb["precision"] > 0
            # Fix B items 1-2: a p90 TAIL dispersion floor, gated only once a real value is authored
            # for this trait (no invented default — see TraitSpec.count_error_tolerance).
            dispersion_ok = (
                trait.count_error_tolerance is None
                or hb["count_error_p90"] <= trait.count_error_tolerance
            )
            # Both conjuncts above stay POOLED while count bias is now per-class, and the per-class
            # statistics they would need are computed and persisted beside them. Left that way
            # deliberately, not by oversight (stage-6 review raised both): each is its own
            # measurement question rather than a mechanical repeat of the bias one — a per-class
            # localization floor refuses a rare class whose single detection lands just outside
            # tolerance, and a per-class dispersion floor reads a tolerance no trait has authored
            # (count_error_tolerance is None everywhere today). A third, related residual is not
            # fixable here at all: count_bias_tolerance is an ABSOLUTE per-image count by the
            # breeder's own choice, so a class present on a few images of many is diluted toward
            # zero and can be wrong by 100% in relative terms while clearing it. Making that
            # judgement relative, or requiring a minimum per-class evidence, is trait semantics —
            # the domain expert's call, not one to infer here.

            # Named-failure architecture: every gate condition below is named here, once, so
            # describe_review_validation (and any future caller) maps failures to breeder-legible
            # reasons from this SAME list rather than re-deriving which check actually failed.
            # cap-saturation (Fix K) is intentionally absent — it is non-gating provenance only.
            # conf_floor_mismatch (Fix D reconciliation) is ALSO non-gating (stage-6 review): its
            # pinned +/-0.05 band is an ordinary property of a model's score distribution as often
            # as it is evidence of tampering, and gating on it re-created exactly the kind of
            # unsound pinned-constant refusal Fix D's redesign was meant to eliminate. It is still
            # computed and surfaced in sweep_data for a human/agent to notice, never blocking.
            failures: list[str] = []
            if not adjudication_ok:
                failures.append("insufficient_adjudication_coverage")
            if not disjoint:
                failures.append("not_disjoint")
            if censored:
                failures.append("conf_censored")
            if content["duplicated"]:
                failures.append("content_duplicated")
            if td["unresolvable"]:
                failures.append("train_disjointness_unresolvable")
            if td["leaked_groups"] or td["leaked_stems"]:
                failures.append("train_disjointness_leaked")
            if cal_gt_count == 0:
                failures.append("insufficient_calibration_gt")
            if hold_gt_count == 0:
                failures.append("insufficient_holdout_gt")
            if hb["n_images"] < 2:
                failures.append("insufficient_holdout_images")
            if not count_bias_ok:
                failures.append("count_bias_exceeds_tolerance")
            if per_class_bias_failures:
                failures.append("count_bias_exceeds_tolerance_per_class")
            if holdout_missing_classes:
                failures.append("holdout_missing_class")
            if not localization_floor_ok:
                failures.append("localization_quality_floor_failed")
            if not dispersion_ok:
                failures.append("count_error_dispersion_too_high")
            passed = not failures

            sweep_data = {"calibration": cal_sweep, "f1_max_conf": pick_f1_max(cal_sweep),
                          "holdout_bias": hb, "count_bias_tolerance": trait.count_bias_tolerance,
                          "per_class_count_bias_failures": per_class_bias_failures,
                          "holdout_missing_classes": holdout_missing_classes,
                          "calibration_bias_at_conf": cb,
                          "count_error_tolerance": trait.count_error_tolerance,
                          "equivalence_z": _EQUIVALENCE_Z,
                          "disjoint": disjoint, "conf_censored": censored,
                          "conf_floor_mismatch": floor_mismatch, "staged_conf_floor": staged_conf_floor,
                          "adjudication_covered": adjudication_ok,
                          "passed_holdout": passed, "failures": failures,
                          "content_overlap_frac": content["content_overlap_frac"],
                          "content_duplicated": content["duplicated"],
                          "train_disjointness": td,
                          "calibration_image_ids": sorted(cal_ids), "holdout_image_ids": sorted(hold_ids),
                          "calibration_observed_min_score": _min_dt_score(calibration_records),
                          "holdout_observed_min_score": _min_dt_score(holdout_records),
                          "calibration_cap_saturated_frac": _cap_saturated_frac(calibration_records),
                          "holdout_cap_saturated_frac": _cap_saturated_frac(holdout_records)}
            # validated only if the gate above raised NO named failure — not merely because a
            # holdout was supplied. Reference here is the annotations/verdicts, not truth; the stamp
            # records which reference (GT vs review-confirmed) cleared the gate.
            validated = validated_reference if passed else VALIDATED_FALSE
        else:
            sweep_data = {"calibration": cal_sweep, "conf_censored": censored,
                          "conf_floor_mismatch": floor_mismatch, "staged_conf_floor": staged_conf_floor,
                          "calibration_observed_min_score": _min_dt_score(calibration_records),
                          "calibration_cap_saturated_frac": _cap_saturated_frac(calibration_records),
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
    if tile_size and tile_size_source == "explicit":
        params["tile_size"] = ResolvedParam(
            "tile_size", int(tile_size), source="explicit",
            derivation_class="deterministic", derived_from="caller override")
    elif tile_size and tile_size_source == "derived":
        params["tile_size"] = derived(
            "tile_size", int(tile_size), derivation_class="deterministic",
            derived_from="model imgsz / persisted training geometry")
    else:
        params["tile_size"] = default("tile_size", tile_size or DEFAULT_TILE_SIZE)
    resolved_tiled = DEFAULT_TILED if tiled is None else bool(tiled)
    if tiled is not None and tiled_source == "explicit":
        params["tiled"] = ResolvedParam(
            "tiled", resolved_tiled, source="explicit",
            derivation_class="deterministic", derived_from="caller override")
    else:
        params["tiled"] = default("tiled", resolved_tiled)
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


def _classification_kappa(items: list[dict]) -> float | None:
    """Cohen's kappa between true and predicted positive/negative class over classification items.

    A derived-at-runtime compensating-error floor (K3): a mean count-bias check alone is blind to a
    classifier that flips k true positives to negative and k true negatives to positive (net bias
    ~0), and the detection path's own localization-quality floor (``recall > 0 and precision > 0``)
    admits any single true positive regardless of how corrupted the rest of the population is — the
    same gap applies here. Kappa corrects for chance agreement from the reference's OWN observed base
    rates (never an authored constant), so a classifier no better than always-guessing-the-majority-
    class scores ~0, and a classifier that inverts the call scores negative. ``None`` when there are
    too few items or only one class present to define a base rate (kappa undefined).
    """
    n = len(items)
    if n == 0:
        return None
    true_pos = sum(1 for it in items if it["is_true_positive"])
    pred_pos = sum(1 for it in items if it["is_pred_positive"])
    agree = sum(1 for it in items if it["is_true_positive"] == it["is_pred_positive"])
    if true_pos in (0, n) or pred_pos in (0, n):
        return None  # a single-class reference/prediction set has no chance-agreement rate to derive
    po = agree / n
    p_true_pos, p_pred_pos = true_pos / n, pred_pos / n
    pe = p_true_pos * p_pred_pos + (1 - p_true_pos) * (1 - p_pred_pos)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def resolve_classifier_operating_point(
    trait_name: str,
    *,
    calibration_items: list[dict] | None = None,
    holdout_items: list[dict] | None = None,
    experiment_id: str | None = None,
    validated_reference: str = VALIDATED_HELD_OUT,
    adjudication_covered: Callable[[dict], bool] | None = None,
) -> dict:
    """Classification-mode calibration gate for a trait's positive-class call (K3).

    Mirrors :func:`resolve_operating_point`'s rigor for a CLASSIFIER's call, not a detector's
    box-finding — calls the same shared primitives (:func:`_content_overlap`,
    :func:`_train_disjointness`) rather than reimplementing them, replacing the detection path's
    localization-quality floor with a derived compensating-error floor (:func:`_classification_kappa`)
    since there is no bbox-match concept here.

    Each item in ``calibration_items``/``holdout_items`` is one classified, already-localized
    instance: ``{"image_id": str, "is_true_positive": bool, "is_pred_positive": bool,
    "bbox": [x1, y1, x2, y2]}`` — whether the GT/reviewer-confirmed label and the classifier's own
    call are the trait's positive state, plus the instance's own GT geometry (required — see
    ``_as_record`` below for why a placeholder box cannot substitute for it).

    Returns a dict **structurally distinct** from a ``ResolvedParam``/``ResolvedBundle`` — never a
    shape a generic writer could mistake for the count operating point's ``conf`` param and stamp
    into the wrong sidecar (K3's distinct-return-shape requirement):
    ``{"validated_vs_gt", "passed", "failures", "sweep_data"}``. Callers write this into a
    classifier-scoped sidecar (``classifier_operating_point.json``, never ``operating_point.json``'s
    own fields) via :func:`tcip_mcp.pipelines.resolution.reconcile_classifier_validity`.

    ``experiment_id is None`` (a foreign/unregistered checkpoint) skips the train-disjointness check
    rather than failing closed, the same owner decision :func:`resolve_operating_point` follows — the
    classifier-validity *stamp* is still reachable for a foreign checkpoint whose cal/holdout is
    otherwise disjoint and unbiased; it is not reachable at all when no calibration/holdout is given.
    """
    if validated_reference not in VALIDATED_SHIPPABLE:
        raise ValueError(f"validated_reference must be one of {VALIDATED_SHIPPABLE}, got {validated_reference!r}")
    get_trait(trait_name)  # validates the trait exists; classification mode needs no trait-shaped fields today
    if not calibration_items or not holdout_items:
        return {
            "validated_vs_gt": VALIDATED_FALSE, "passed": False,
            "failures": ["no_calibration_or_holdout"],
            "sweep_data": {"note": "classifier calibration requires both calibration and holdout items"},
        }

    adjudication_ok = adjudication_covered is None or (
        all(adjudication_covered(r) for r in calibration_items)
        and all(adjudication_covered(r) for r in holdout_items)
    )

    cal_ids = {it["image_id"] for it in calibration_items if "image_id" in it}
    hold_ids = {it["image_id"] for it in holdout_items if "image_id" in it}
    disjoint = bool(cal_ids) and bool(hold_ids) and not (cal_ids & hold_ids)

    # Reuse the detection path's content-overlap/train-disjointness primitives by shaping each
    # classification item as a one-annotation image record — never a second implementation of
    # "is this holdout content actually cloned from calibration" or "was this in the training split".
    # The item's REAL GT bbox is required here, not a placeholder: _record_content_hash's whole
    # purpose is a per-instance content fingerprint, and every item of the same class would collapse
    # to one identical hash if the geometry were faked (stage-6 review, K3: this made the
    # content-duplication check fire on every well-formed reference and pass only a degenerate
    # single-class one — the exact inversion of what it's meant to catch).
    def _as_record(it: dict) -> dict:
        cid = 1 if it["is_true_positive"] else 0
        return {"image_id": it.get("image_id"), "width": 0, "height": 0,
                "gt": [{"category_id": cid, "bbox": it["bbox"]}]}

    content = _content_overlap([_as_record(it) for it in calibration_items],
                               [_as_record(it) for it in holdout_items])
    td = _train_disjointness(experiment_id, cal_ids, hold_ids)

    cal_pos = sum(1 for it in calibration_items if it["is_true_positive"])
    hold_pos = sum(1 for it in holdout_items if it["is_true_positive"])
    trait = get_trait(trait_name)
    # Per-image mean count-bias, via the SAME mean+SE equivalence test the detection path gates on
    # (stage-6 review, K3: the prior version summed a whole-holdout total and gated it against
    # trait.count_bias_tolerance — a per-image mean by its own docstring — so the gate silently got
    # stricter as the holdout grew, despite a comment claiming "the same absolute unit"). Grouped by
    # image_id since one image can carry several classified instances.
    by_image: dict[str | None, list[dict]] = {}
    for it in holdout_items:
        by_image.setdefault(it.get("image_id"), []).append(it)
    per_image_bias = [
        sum(1 for it in its if it["is_pred_positive"]) - sum(1 for it in its if it["is_true_positive"])
        for its in by_image.values()
    ]
    n_bias_images = len(per_image_bias)
    count_bias = statistics.fmean(per_image_bias) if per_image_bias else 0.0
    # Sample stdev (ddof=1/Bessel's correction), matching the detection path's np.std(biases,
    # ddof=1) exactly (stage-6 review Finding C/N4) — pstdev's population estimator was
    # systematically more permissive, worst at small n, which is exactly where the equivalence
    # test's SE penalty is supposed to bite hardest.
    count_bias_std = statistics.stdev(per_image_bias) if n_bias_images > 1 else 0.0
    count_bias_ok = _bias_equivalence_ok(
        count_bias, count_bias_std, n_bias_images, trait.count_bias_tolerance)

    kappa = _classification_kappa(holdout_items)
    # kappa is None only when the holdout is degenerate (a single class throughout) — a real
    # reference for a trait with two states should not be, so treat that as a failure to derive
    # rather than a pass. Two floors (stage-6 review Finding B): kappa > 0 is the universal,
    # domain-input-free minimum (better than pure chance — a classifier that flips a full 40% of
    # calls symmetrically, net count-bias ~0, clears this alone at kappa=0.2, exactly the
    # compensating-error case this check exists to catch); `agreement_floor` is the trait's own
    # authored bar (`TraitSpec.classifier_agreement_floor`), falling back to the platform's
    # provisional interim default only when the trait hasn't set one.
    agreement_floor = (
        trait.classifier_agreement_floor
        if trait.classifier_agreement_floor is not None else _PROVISIONAL_KAPPA_FLOOR)
    compensating_error_ok = kappa is not None and kappa > 0.0 and kappa > agreement_floor

    failures: list[str] = []
    if not adjudication_ok:
        failures.append("insufficient_adjudication_coverage")
    if not disjoint:
        failures.append("not_disjoint")
    if content["duplicated"]:
        failures.append("content_duplicated")
    if td["unresolvable"]:
        failures.append("train_disjointness_unresolvable")
    if td["leaked_groups"] or td["leaked_stems"]:
        failures.append("train_disjointness_leaked")
    if cal_pos == 0:
        failures.append("insufficient_calibration_positive_evidence")
    if hold_pos == 0:
        failures.append("insufficient_holdout_positive_evidence")
    if len(holdout_items) < 2:
        failures.append("insufficient_holdout_items")
    if n_bias_images < 2:
        # Same minimum the detection path requires (hb["n_images"] < 2) — stage-6 review Finding
        # C/N4: without this, a single-image holdout forces count_bias_std to 0.0 (no images to
        # vary across), so the equivalence test's SE penalty vanishes and a lone image can pass at
        # exactly the tolerance with zero uncertainty discount.
        failures.append("insufficient_holdout_images")
    if not count_bias_ok:
        failures.append("count_bias_exceeds_tolerance")
    if not compensating_error_ok:
        failures.append("compensating_error_floor_failed")
    passed = not failures

    sweep_data = {
        "content_overlap_frac": content["content_overlap_frac"], "content_duplicated": content["duplicated"],
        "train_disjointness": td, "disjoint": disjoint, "adjudication_covered": adjudication_ok,
        "count_bias": count_bias, "count_bias_std": count_bias_std, "count_bias_n_images": n_bias_images,
        "count_bias_tolerance": trait.count_bias_tolerance,
        "kappa": kappa, "kappa_floor": agreement_floor,
        "kappa_floor_source": ("trait" if trait.classifier_agreement_floor is not None
                               else "platform_provisional_default"),
        "n_calibration": len(calibration_items), "n_holdout": len(holdout_items),
    }
    return {
        "validated_vs_gt": validated_reference if passed else VALIDATED_FALSE,
        "passed": passed, "failures": failures, "sweep_data": sweep_data,
    }
