"""Resolve the calibrated operating points (detection conf/NMS/max_dets/tile, and the classifier,
ordinal and regression points) per dataset, at runtime.

This is the single place the calibrated consumers, train-eval, test-eval, inference, export and the
phenology delivery's classifier gate, get an operating point, so the same model + images can't yield
different counts by entry point (the audit's divergent-defaults bug); the raw and
block-calibrated-export regimes live in ``resolution.py`` and share ``resolve_tile_size_param``. The confidence threshold requires validation against an annotations
reference: derived by a center-match count-unbiased sweep over a reference sized to the trait,
and validated on a disjoint held-out split of that reference, GT annotations
(``VALIDATED_HELD_OUT``) OR a breeder-confirmed sample of the model's own outputs
(``VALIDATED_REVIEW_CONFIRMED``), the same gate either way, or carried as ``validated=false`` when
no reference exists (never a frozen literal). See the scope doc and traits.py.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from typing import Any, Callable, Sequence

from tcip_store import non_finite_state, stored_number

from tcip_mcp.pipelines.derivations import derive_cross_tile_nms, derive_localization_tolerance_frac
from tcip_mcp.pipelines.resolution import (
    DEFAULT_CONF,
    DEFAULT_MAX_DETS,
    DEFAULT_NMS_IOU,
    VALIDATED_FALSE,
    VALIDATED_HELD_OUT,
    VALIDATED_REVIEW_CONFIRMED,
    ResolvedBundle,
    ResolvedParam,
    accepted_references,
    default,
    derived,
    label_digests,
    resolve_tile_size_param,
)
from tcip_mcp.pipelines.training.evaluation import (
    classes_with_evidence,
    concordance_correlation_coefficient,
    derive_operating_point_curve,
    gt_class_avg_size,
    gt_class_typical_count,
    mean_of_present_counts,
    pick_count_unbiased,
    pick_f1_max,
    quadratic_weighted_kappa,
    r_squared,
)
from tcip_mcp.traits import COUNT_OBJECTIVES, COUNT_UNBIASED, DETECTION_F1, PRESENCE, get_trait

# The non-count operating-point fallbacks resolve to resolution.py's single source of truth
# (DEFAULT_NMS_IOU / DEFAULT_MAX_DETS / DEFAULT_CONF); tile_size/tiled carry no such constant.

# Count-objective -> (picker, derivation label), the currently implemented capability catalog, not
# a closed vocabulary (traits._spec_from_config does not validate count_objective against this; a
# trait can name any objective, but resolve_operating_point below can only run one that has a
# registered picker here). The label stamped on ``conf`` is derived from whichever picker actually
# ran (pick-then-label), never from the reference type alone. PRESENCE deliberately shares
# DETECTION_F1's F1-max picker/label: presence is a per-object find/no-find call, exactly what F1
# (harmonic precision/recall) measures on matching quality, unlike COUNT_UNBIASED's sum-agreement
# objective (the phenotype is a count), which needs its own picker. Add a new entry here (and a new
# picker function) when a trait's breeder-stated need doesn't match either existing one, the
# capability grows by registering a picker, not by widening a vocabulary check.
COUNT_OBJECTIVE_PICKERS: dict[str, tuple[Callable[[dict], float | None], str]] = {
    COUNT_UNBIASED: (pick_count_unbiased, "count-unbiased center-match curve"),
    DETECTION_F1: (pick_f1_max, "F1-max center-match curve"),
    PRESENCE: (pick_f1_max, "F1-max center-match curve"),
}
assert set(COUNT_OBJECTIVE_PICKERS) == COUNT_OBJECTIVES, (
    "COUNT_OBJECTIVE_PICKERS and traits.COUNT_OBJECTIVES must name the same currently-implemented "
    "objectives, two lists of the same capability set, kept in sync deliberately")

REVIEW_VERDICT_LABEL_SUFFIX = " over review verdicts"
"""What a picker's label gains when that picker swept confirmed review verdicts instead of GT.

Stated once, beside the labels it extends: the label a run stamps and the entries
``derivations.DERIVATION_IMPLEMENTATIONS`` holds for it are built from the same two pieces, so
registering a picker registers its label, and its review-reference variant, with it.
"""

# The one-sided confidence multiplier for the mean+SE equivalence criterion (~95%) is a stated
# CV-derivation convention, not a breeder-semantics decision, so it lives here as a named constant
# rather than buried in a formula.
_EQUIVALENCE_Z = 1.645

# The compensating-error floor here is an interim default, not a cited statistical convention like
# _EQUIVALENCE_Z: Landis & Koch (1977)'s kappa scale describes a magnitude, not a distributional fact.

# How much classifier agreement a trait's phenotype needs is measurement semantics, the domain
# expert's call, the same as `TraitSpec.count_error_tolerance`.

# This value is an interim, platform-chosen placeholder, never a validated or cited convention,
# used only when a trait hasn't authored `TraitSpec.classifier_agreement_floor` (None).

# kappa==0 is exactly chance agreement; a floor there alone admits a classifier whose errors are
# compensating (net count-bias ~0) but substantial.
_DEFAULT_KAPPA_FLOOR = 0.41

# The same "not yet authored for this trait" shape as `_DEFAULT_KAPPA_FLOOR`: how much relative
# per-image count error a trait's phenotype can tolerate is measurement semantics, the expert's call.

# This value is an interim, platform-chosen placeholder, never a validated or cited convention,
# used only when a trait hasn't authored `TraitSpec.count_bias_tolerance_frac` (None).
_DEFAULT_COUNT_BIAS_TOLERANCE_FRAC = 0.01

# The compensating-error criterion toolkits for the ordinal/regression calibration gates
# (:func:`resolve_ordinal_operating_point`/:func:`resolve_regression_operating_point`), a small,
# discoverable, growable set of named statistics rather than one hardcoded "the" criterion: which
# statistic is scientifically appropriate for a given trait's calibration is a CV-scientist judgment
# call the caller makes explicitly (the ``criterion`` argument, required, no default), never a
# platform prescription. Register a new criterion here (a function of ``(pred, gt)`` returning
# ``float | None``) rather than widening either function's own logic to special-case a new statistic.
ORDINAL_CRITERIA: dict[str, Callable[[Any, Any], float | None]] = {
    "quadratic_weighted_kappa": quadratic_weighted_kappa,
}
REGRESSION_CRITERIA: dict[str, Callable[[Any, Any], float | None]] = {
    "r_squared": r_squared,
    "concordance_correlation_coefficient": concordance_correlation_coefficient,
}

# The same interim-platform-default shape as `_DEFAULT_KAPPA_FLOOR`, and literally the same
# statistic: ordinal's only registered criterion is the classifier path's own kappa.

# An interim, platform-chosen placeholder, never a validated or cited convention, used only
# when a trait hasn't authored `TraitSpec.ordinal_agreement_floor` (None).
_DEFAULT_ORDINAL_AGREEMENT_FLOOR = 0.41

# The same interim-platform-default shape, for whichever regression criterion a calibration
# actually used: a plain "explains more than half the addressable skill" default.

# R² and CCC have different scales (see `TraitSpec.regression_skill_floor`'s docstring), so
# this single number is an interim, platform-chosen placeholder for either.

# Never a validated or cited convention: it applies only when a trait hasn't authored its own
# floor paired with its own criterion choice (None).
_DEFAULT_REGRESSION_SKILL_FLOOR = 0.5


def _effective_count_bias_tolerance(tolerance_frac: float, typical_count: float, n: int) -> float:
    """The absolute per-image count-bias tolerance one scope (pooled, or one class) is actually held
    to: the breeder-authored relative ``tolerance_frac`` scaled by that scope's own derived typical
    per-image count (:func:`training.evaluation.gt_class_typical_count` /
    :func:`training.evaluation.mean_of_present_counts`), floored at ``1 / n``, one whole miscount
    spread across the ``n`` samples this scope's own evidence rests on.

    The floor is itself derived from ``n`` (the same count :func:`_bias_equivalence_ok`'s own
    standard error already uses), never an authored or platform-invented constant, and via ``max()``
    can only ever raise the result above what ``tolerance_frac * typical_count`` alone would give,
    never lower it. As a function of ``n`` alone it shrinks monotonically as evidence grows and is
    bounded: exactly ``1.0`` at ``n == 1`` (``n`` is an integer, so ``1/n`` cannot exceed 1.0), and
    ``<= 0.5`` at every ``n >= 2``.

    ``n < 2`` is exactly the range every caller's own reference-sufficiency gate independently
    refuses on, so a floor this large is never what admits a reference: the pooled scope's
    ``hb["n_present"] < 2`` / ``insufficient_holdout_images`` conjunct, and the per-class scope's own
    ``s["n_present"] < 2`` / ``insufficient_holdout_images_per_class`` conjunct, the latter exists
    because a class present on exactly one holdout image could otherwise reach a passing per-class
    tolerance end to end (a real, ordinary, non-adversarial reference: a rare class that happens to
    show up once, in a denser-than-typical frame). Both conjuncts count the same presence-scoped
    evidence this function's ``n`` is, and both land in the same ``failures`` list its return value
    feeds, independent of what this function itself computes.

    ``n == 0`` returns 0.0 (an unachievable tolerance): :func:`_bias_equivalence_ok`'s own ``n == 0``
    branch already refuses before this matters, so this is a defined-but-moot edge, not a silent pass.
    """
    return max(tolerance_frac * typical_count, 1.0 / n) if n > 0 else 0.0


def _bias_equivalence_ok(mean: float, std: float, n: int, *, tolerance_frac: float,
                         typical_count: float) -> bool:
    """Mean-plus-SE equivalence test: is a bias measured across ``n`` per-image samples small
    enough, relative to its own sampling uncertainty, to conclude equivalence with zero? Not a bare
    mean check, this degrades correctly at small ``n`` (SE grows, so less evidence is harder to
    pass, never easier). Shared by the detection path (:func:`resolve_operating_point`) and the
    classifier path (:func:`resolve_classifier_operating_point`) so both judge count bias in the
    same statistical shape and unit, a per-image mean, never two independently-derived criteria
    that happen to share a name and a tolerance field.

    The tolerance itself is computed here, in one place, via :func:`_effective_count_bias_tolerance`,
    every caller passes the breeder-authored fraction and the scope's own derived typical count as
    keyword-only arguments, never a raw already-converted tolerance float, so a future caller cannot
    silently pass ``trait.count_bias_tolerance_frac`` straight through as if it were already an
    absolute count (the old, wrong shape this signature makes impossible to spell).

    ``mean``/``std``/``n`` must all be measured over the same population the ``typical_count`` was:
    the samples that actually carry the thing being counted. Passing a bias averaged over a wider
    population than the density was measured on loosens the effective relative tolerance by exactly
    the ratio of the two population sizes, silently and without any record of it.
    """
    if n == 0:
        return False
    se = std / math.sqrt(n)
    tolerance = _effective_count_bias_tolerance(tolerance_frac, typical_count, n)
    return abs(mean) + _EQUIVALENCE_Z * se <= tolerance


OPERATING_POINT_ATTRS = ("score_thresh", "nms_thresh", "detections_per_img")
"""The three knobs the detection operating point governs, wherever a module holds them."""


def detector_operating_point_holder(model: Any) -> tuple[Any, str | None]:
    """Where a model's operating-point knobs live, and the attribute path to name it by.

    Checked in this order, the first that exposes any of :data:`OPERATING_POINT_ATTRS`: the module
    itself, its ``.detector``'s ``roi_heads`` (two-stage detectors), its ``.detector`` (one-stage).
    A composed torchvision detector under ``.detector`` resolves the way it always has; a bespoke
    module exposing a knob on itself, with no ``.detector`` to route through, now reaches it too,
    independently of what a given call actually sets. Returns ``(None, None)`` when no candidate
    exposes any of the three: nothing here can govern which boxes exist.

    Raises ``ValueError``, naming both locations, when more than one candidate exposes a knob: the
    module itself and its ``.detector.roi_heads`` disagreeing about where the operating point lives
    is an ambiguous module, never a case to silently resolve by picking the first match.
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
                                 nms_thresh: float | None = None,
                                 detections_per_img: int | None = None,
                                 ) -> tuple[dict, str | None]:
    """Set the *in-model* torchvision thresholds so the operating point governs which boxes exist.

    Resolves where the knobs live through :func:`detector_operating_point_holder`, so the module
    itself, its ``.detector.roi_heads`` (two-stage detectors) or its ``.detector`` (one-stage) can
    each hold them. Returns ``(applied, attribute_path)``: ``applied`` holds only the knobs actually
    set (never a truthy dict for a model with nothing to set), and ``attribute_path`` is the
    holder's own path, or ``None`` when nothing exposed any knob. (Without this, a post-hoc score
    filter can never recover a box the model's internal ``score_thresh``/``detections_per_img`` had
    already discarded.)
    """
    target, path = detector_operating_point_holder(model)
    applied: dict = {}
    if target is not None:
        for attr, val in (("score_thresh", score_thresh), ("nms_thresh", nms_thresh),
                          ("detections_per_img", detections_per_img)):
            if val is not None and hasattr(target, attr):
                setattr(target, attr, val)
                applied[attr] = val
    return applied, path


def _current_detections_cap(model: Any) -> int | None:
    """The in-model ``detections_per_img`` a model is currently set to, or ``None`` if unset.

    Read, not derived, whatever ``set_detector_operating_point`` last applied (or the framework
    default if nothing was ever set), through the same :func:`detector_operating_point_holder` the
    setter itself resolves through, so the two never disagree about where the knob lives. Used only
    to stamp the non-gating cap-saturation signal at record-generation time, when the cap is
    actually known.
    """
    target, _path = detector_operating_point_holder(model)
    return getattr(target, "detections_per_img", None) if target is not None else None


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

    Non-gating provenance only: a per-image ``cap_hit`` flag is stamped by
    ``records_from_detector``/``records_over_loader`` when the cap was known at generation time, and
    by ``run_full_frame_evaluation`` for its own tiled-and-reconstructed pass. Records built some
    other way carry none and are excluded from the fraction entirely, not counted as an unsaturated
    0, ``None`` when nothing here carries the flag.
    """
    hits = [r["cap_hit"] for r in (records or []) if "cap_hit" in r]
    return (sum(hits) / len(hits)) if hits else None


def _conf_censored(chosen_conf: float, staged_conf_floor: float | None) -> bool:
    """True when a *stated* floor sits at or above the picked conf, so the sweep could not see
    whether an even-lower conf would have done better.

    The count-unbiased sweep is only trustworthy when the reference includes the low-conf tail below
    the picked conf. A conf picked strictly above the staging floor is fully supported by the
    reference, every surviving detection with ``score >= conf`` genuinely survived the floor's own
    filter, so the sweep saw everything it needed to. ``staged_conf_floor is None`` (the caller made
    no assertion of what floor the reference was generated at) is a distinct failure, not this one:
    see ``conf_floor_unstated``, gated separately so the two causes never share one name.
    """
    return staged_conf_floor is not None and chosen_conf <= staged_conf_floor


def _floor_mismatch(records: list[dict] | None, staged_conf_floor: float | None) -> bool:
    """True when the reference's observed lowest score is inconsistent with the asserted floor.

    A material gap (>0.05) between what the caller asserts the reference was staged at and what the
    data actually shows is itself evidence the assertion is wrong, or that something else (a stale
    bucket, cap-trimmed tiles, a bespoke caller) truncated the reference after generation, a
    distinct failure mode from ``_conf_censored`` (which only compares the picked conf, not the
    reference's own data, against the asserted floor).
    """
    if staged_conf_floor is None:
        return False
    observed_min = _min_dt_score(records or [])
    return observed_min is not None and observed_min > staged_conf_floor + 0.05


def derive_max_dets_from_counts(counts: list[int], floor: int = 100) -> int:
    """A generous cap = ~1.5x the p99 object count, so dense scenes aren't truncated.

    Shared by ``_max_dets_from_density`` (per-record GT counts, over already-collected records) and
    ``scripts/calibrate_operating_point.py`` (raw per-stem label-line counts, known before any model
    pass) so the record-collection cap and the eventually-resolved ``max_dets`` agree on one formula
    rather than two independently-typed derivations that could drift apart.
    """
    import numpy as np
    if not counts:
        return DEFAULT_MAX_DETS
    return max(floor, int(math.ceil(1.5 * float(np.quantile(counts, 0.99)))))


def _max_dets_from_density(records: list[dict], floor: int = 100) -> int:
    """A generous cap = ~1.5x the p99 GT objects-per-image, so dense scenes aren't truncated."""
    return derive_max_dets_from_counts([len(rec.get("gt", [])) for rec in records], floor=floor)


def _record_content_hash(rec: dict) -> str | None:
    """Content identity of one record's GT, ``(width, height, sorted (category_id, bbox))``,
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
    """Whether the holdout shares any image's GT content with calibration.

    ``shared`` fires on any overlap at all, not only full containment: a holdout that shares even
    one image's content with calibration is not independent for that image, so a holdout sharing
    images with the calibration set is refused rather than merely flagged. Both callers compare at
    image granularity, one record per image: the detection path's records already are one per
    image, and the classifier path (``resolve_classifier_operating_point``) groups its per-instance
    items back to one record per ``image_id`` before calling this, so two images sharing an
    instance's coordinates never hash equal unless their whole classified content agrees. The cost
    of the ruling this enforces: two genuinely independent images that happen to share both
    dimensions and identical labelled geometry read as shared content and refuse. On the classifier
    path the grouping key is the record's ``image_id``, which the platform's classifier door
    (``calibrate_classifier_operating_point``) sets to the label file's own stem.
    ``content_overlap_frac`` travels for provenance; ``shared`` is the boolean the gate reads.
    """
    cal_hashes = {h for h in (_record_content_hash(r) for r in cal_records) if h is not None}
    hold_hashes = {h for h in (_record_content_hash(r) for r in hold_records) if h is not None}
    if not hold_hashes:
        return {"content_overlap_frac": 0.0, "shared": False}
    frac = len(hold_hashes & cal_hashes) / len(hold_hashes)
    return {"content_overlap_frac": frac, "shared": frac > 0}


_UNRESOLVABLE_TRAIN_DISJOINTNESS = {
    "checked": False, "unresolvable": True, "leaked_groups": [], "leaked_stems": [],
    "group_check": None,
}


def _spatial_strip_geometric_disjointness(
    spatial: dict, cal_rects: dict | None, hold_rects: dict | None,
) -> dict:
    """The geometric form of the spatial_strip check: a cal/holdout rect must be fully contained
    in a persisted non-train region (``val_region``/``test_region``/``calibration_region``, the
    last only present on a four-way split) and disjoint from every persisted train region, read
    from ``spatial`` (the ``split.json`` ``"spatial"`` block ``persist_split_manifest`` writes).
    Compares real geometry, so it catches a leak the lexical stem-identity check can't: a rect
    that spills into the reserved train area from a source stem whose name never matches the
    training stem's own.
    """
    from tcip_mcp.pipelines.data.tiling import rect_contains_rect, rects_overlap

    train_regions = [tuple(r) for r in spatial.get("train_region", [])]
    non_train_regions = ([tuple(r) for r in spatial.get("val_region", [])]
                         + [tuple(r) for r in spatial.get("test_region", [])]
                         + [tuple(r) for r in spatial.get("calibration_region", [])])
    leaked_groups: list[str] = []
    for rects in (cal_rects or {}, hold_rects or {}):
        for stem, rect in rects.items():
            rect = tuple(rect)
            contained = any(rect_contains_rect(nt, rect) for nt in non_train_regions)
            overlaps_train = any(rects_overlap(tr, rect) for tr in train_regions)
            if not contained or overlaps_train:
                leaked_groups.append(stem)
    return {
        "checked": True, "unresolvable": False, "leaked_groups": sorted(set(leaked_groups)),
        "leaked_stems": [], "group_check": "spatial_strip_geometric",
    }


def _train_disjointness(
    experiment_id: str | None, cal_ids: set, hold_ids: set, *,
    cal_rects: dict[str, tuple[int, int, int, int]] | None = None,
    hold_rects: dict[str, tuple[int, int, int, int]] | None = None,
) -> dict:
    """Whether the cal/holdout images were also in the checkpoint's own training split.

    ``experiment_id is None`` -> a foreign/unregistered checkpoint with no known training
    provenance to check against; allowed through by design, only a *known* ``experiment_id`` whose
    provenance can't be read/reconstructed fails closed.

    Two genuinely-unresolvable cases stay ``unresolvable: True`` (fail-closed, blocks ``passed``
    in ``resolve_operating_point``): ``split.json`` is missing/unreadable, or it is readable but
    records no training stems at all, there is nothing here to check against, not even the
    stem-level fallback below.

    Otherwise (``split.json`` readable, with real training stems) this never blanket-refuses just
    because a group policy can't be resolved, a blanket refusal would permanently block the
    explicit-``val_images_dir`` route (``group_by="external"``, no computed grouping) and the
    ``group_key_map`` route (the map was never persisted) even though both are legitimate, disjoint
    training regimes. Instead, group-level resolution is attempted per stem:

      - a named, recognized strategy (``tile_prefix``/``stem``) resolves every stem, as before.
      - ``group_by == "explicit_map"`` resolves via the persisted ``group_key_map`` for whichever
        stems it actually covers; stems it doesn't cover are treated as unresolvable for that stem
        only, not a blanket failure.
      - anything else (``"external"``, an unrecognized string, a missing field) resolves nothing
        at the group level.

    Every stem the group check couldn't cover falls back to the free, policy-independent check
    that's always available regardless of grouping: exact stem-set overlap between the training
    run's own stems and the calibration/holdout stem set (``leaked_stems``). ``group_check``
    records how much of the check was group-level: ``"performed"`` (every stem grouped),
    ``"partial"`` (some stems fell back to exact-stem), ``"not_performed"`` (no group policy
    resolved at all, wholly exact-stem), ``"spatial_strip"`` (a within-image split, checked
    by source stem underneath each region identity), or ``"spatial_strip_geometric"`` (see below).
    A leak found by either mechanism blocks ``passed``. The group/stem resolution above (not the
    spatial-strip branch) is :func:`_resolve_group_stem_disjointness`, shared verbatim with
    :func:`_selection_disjointness`, which runs the identical check against a checkpoint's own
    selection (val) side instead of its training side.

    ``cal_rects``/``hold_rects`` (keyed by source stem, one pixel rect ``(x0, y0, x1, y1)`` per
    stem) are optional and additive: every existing caller omits them and gets exactly the
    behavior above, unchanged. When either is given and the persisted split is
    ``group_by == "spatial_strip"``, the check becomes geometric containment against the
    persisted ``train_region``/``val_region``/``test_region`` rects
    (:func:`_spatial_strip_geometric_disjointness`) instead of the lexical same-source check
    above, which this path drops entirely rather than running both.
    """
    if experiment_id is None:
        return {"checked": False, "unresolvable": False, "leaked_groups": [], "leaked_stems": [],
                "group_check": None}
    from tcip_mcp.experiments import read_split_manifest

    split = read_split_manifest(experiment_id)
    train_stems = split.get("train") or []
    if not train_stems:
        # Nothing recorded to check against at all, not even the stem-overlap fallback has
        # anything to compare, so this is genuinely unresolvable, not merely ungrouped.
        return dict(_UNRESOLVABLE_TRAIN_DISJOINTNESS)

    cal_hold_stems = sorted(cal_ids | hold_ids)
    group_by = split.get("group_by")

    if group_by == "spatial_strip":
        from tcip_mcp.pipelines.data.splits import stem_of_spatial_identity

        if cal_rects or hold_rects:
            return _spatial_strip_geometric_disjointness(
                split.get("spatial") or {}, cal_rects, hold_rects)
        # train_stems are per-region identities, not bare stems; only a same-source reference is
        # caught here (a caller with real rects gets the geometric check above instead).
        train_source_stems = {stem_of_spatial_identity(s) for s in train_stems}
        spatial_leaked_groups = sorted(train_source_stems & set(cal_hold_stems))
        return {
            "checked": True, "unresolvable": False, "leaked_groups": spatial_leaked_groups,
            "leaked_stems": [], "group_check": "spatial_strip",
        }

    persisted_map = split.get("group_key_map") if group_by == "explicit_map" else None
    return _resolve_group_stem_disjointness(train_stems, cal_hold_stems, group_by, persisted_map)


def _resolve_group_stem_disjointness(
    named_side_stems: Sequence[str], cal_hold_stems: Sequence[str], group_by: str | None,
    persisted_map: dict[str, str] | None,
) -> dict:
    """The group- and stem-level leak resolution :func:`_train_disjointness` and the selection
    check share: a named, recognized strategy (``tile_prefix``/``stem``) resolves every stem; an
    ``explicit_map`` policy resolves via ``persisted_map`` for whichever stems it covers; anything
    else resolves nothing at the group level. Every stem the group check couldn't cover falls back
    to the free, policy-independent check: exact stem-set overlap between ``named_side_stems`` and
    ``cal_hold_stems``. Never called for a ``spatial_strip`` record, which each caller branches on
    before reaching here.
    """
    from tcip_mcp.pipelines.data.splits import GROUP_KEY_FNS, resolve_group_key_fn

    covered_side: list[str] = []
    covered_cal_hold: list[str] = []
    key_fn = None
    if persisted_map:
        covered_side = [s for s in named_side_stems if s in persisted_map]
        covered_cal_hold = [s for s in cal_hold_stems if s in persisted_map]
        if covered_side or covered_cal_hold:
            # The map, when present, always wins over group_by per resolve_group_key_fn's own
            # contract, the "tile_prefix" here is an inert placeholder, never consulted.
            key_fn = resolve_group_key_fn("tile_prefix", covered_side + covered_cal_hold,
                                          group_key_map=persisted_map)
    elif group_by and group_by in GROUP_KEY_FNS:
        key_fn = GROUP_KEY_FNS[group_by]
        covered_side, covered_cal_hold = list(named_side_stems), list(cal_hold_stems)

    leaked_groups: list[str] = []
    if key_fn is not None:
        side_groups = {key_fn(s) for s in covered_side}
        cal_hold_groups = {key_fn(s) for s in covered_cal_hold}
        leaked_groups = sorted(side_groups & cal_hold_groups)

    uncovered_side = sorted(set(named_side_stems) - set(covered_side))
    uncovered_cal_hold = sorted(set(cal_hold_stems) - set(covered_cal_hold))
    leaked_stems = sorted(set(uncovered_side) & set(uncovered_cal_hold))

    if key_fn is None:
        group_check = "not_performed"
    elif uncovered_side or uncovered_cal_hold:
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


_NOT_APPLICABLE_SELECTION_SHAPE = {
    "checked": False, "unresolvable": False, "leaked_groups": [], "leaked_stems": [],
    "group_check": None,
}
_UNRESOLVABLE_SELECTION_SHAPE = {
    "checked": False, "unresolvable": True, "leaked_groups": [], "leaked_stems": [],
    "group_check": None,
}


def _resolve_label_movement(
    label_digests_block: dict | None, cal_ids: set,
    calibration_labels_dir: str | None, split_manifest_sha256: str | None,
) -> dict:
    """The four label-movement keys plus ``calibration_labels_dir``, from a bound run's own
    ``split.json``'s ``label_digests`` block (``at_split``/``at_run``/``manifest_sha256``) and
    the calibration's own labels directory, when it read one.

    All four keys ``None`` when the run recorded no ``label_digests`` block, or recorded one with
    an empty ``at_split`` (a run bound before that block existed, or an unbound run calibrated
    under a caller-named manifest): "not checked" must never read as "nothing moved". Otherwise
    the two digest dictionaries share one key set (the draw's), so a stem present only in
    ``at_run`` or added to the calibration universe after the draw is named by no key.

    The second window (``labels_moved_run_to_now``) is scoped to ``cal_ids``, the calibration's
    own universe, rather than to every stem of ``at_run``: ``calibration_labels_dir`` may be one
    of several already-split per-image directories (a classifier calibration's own GT dir, say),
    holding only the calibration side's own files, so recomputing over the run's whole bound set
    would read a train- or val-side stem's absence as a move it never made. When the scoped set is
    empty (the named directory holds none of ``at_run``'s documents), the second window is left
    unsealed (``None``) rather than sealed empty, since nothing in it was actually checked.
    """
    if not label_digests_block or not label_digests_block.get("at_split"):
        return {
            "labels_moved_draw_to_run": None,
            "labels_moved_run_to_now": None,
            "calibration_labels_moved": None,
            "manifest_redrawn": None,
            "calibration_labels_dir": calibration_labels_dir,
        }
    at_split = label_digests_block.get("at_split") or {}
    at_run = label_digests_block.get("at_run") or {}
    manifest_sha256 = label_digests_block.get("manifest_sha256")

    labels_moved_draw_to_run = sorted(
        stem for stem, digest in at_split.items() if at_run.get(stem) != digest)
    scoped = sorted(set(at_run) & cal_ids) if calibration_labels_dir is not None else []
    if scoped:
        # scoped is non-empty only when calibration_labels_dir is not None (see the ternary above).
        assert calibration_labels_dir is not None
        now = label_digests(calibration_labels_dir, scoped)
        labels_moved_run_to_now = sorted(
            stem for stem in scoped if now.get(stem) != at_run.get(stem))
    else:
        labels_moved_run_to_now = None

    moved = set(labels_moved_draw_to_run) | set(labels_moved_run_to_now or [])
    return {
        "labels_moved_draw_to_run": labels_moved_draw_to_run,
        "labels_moved_run_to_now": labels_moved_run_to_now,
        "calibration_labels_moved": sorted(moved & cal_ids),
        "manifest_redrawn": (
            split_manifest_sha256 != manifest_sha256 if split_manifest_sha256 is not None
            else None
        ),
        "calibration_labels_dir": calibration_labels_dir,
    }


def _selection_disjointness(
    experiment_id: str | None, cal_ids: set, hold_ids: set, *,
    split_manifest_dir: str | None = None, calibration_date: str | None = None,
    calibration_labels_dir: str | None = None, split_manifest_sha256: str | None = None,
) -> dict:
    """Whether the cal/holdout images were also on the checkpoint's own selection side (its
    ``split.json``'s ``val``), the check the manifest family's own ruling exists for: a checkpoint
    chosen on a side and then calibrated over that same side would clear every other gate while
    measuring the operating point on exactly the data the shipped weights were picked to fit.

    Unlike :func:`_train_disjointness` (date-blind, checked on every calibration), this check is
    ``applicable`` only when a selection side could plausibly be touched: the calibration names a
    split manifest (``split_manifest_dir``) or the checkpoint's own record carries a
    ``manifest_binding``, whichever is true. Not-applicable, each with a breeder-legible reason
    (``review_calibration.py`` renders it), for: no manifest named and no ``manifest_binding`` on
    the record; a ``spatial_strip`` record (the within-image route's own ``calibration_region`` is
    a different check, untouched here); an empty ``val``; ``resolved_group_by == "external"`` (the
    ``val`` came from a directory the record's ``date`` says nothing about); ``calibration_date is
    None`` (the caller derived no date to compare at all, never read as matching a flat record,
    whose own ``date`` is ``manifest_date_key``'s empty string, not ``None``); or a
    ``calibration_date`` other than the record's own ``date`` (a bare stem means the same image
    only under one date). Unresolvable, rather than not-applicable, when the calibration names a
    manifest but there is no experiment record to check it against
    (``experiment_id is None``): the ruling forbids exactly the number whose provenance can't be
    checked, and a foreign checkpoint's train check being merely skipped is not license to skip
    this one silently too.

    Returns the same shape :func:`_train_disjointness` does, plus ``applicable``/``reason``, and
    on the applicable path the four label-movement keys plus ``calibration_labels_dir``
    (:func:`_resolve_label_movement`): the four movement keys ``null``, ``calibration_labels_dir``
    preserved from the caller, and ``reason`` naming why, when the run recorded no
    ``label_digests`` block on its ``split.json`` or recorded one with an empty ``at_split``.
    """
    if experiment_id is None:
        if split_manifest_dir is not None:
            return {"applicable": True,
                    "reason": "the calibration names a split manifest but this checkpoint has no "
                              "recorded experiment to check its selection side against",
                    **_UNRESOLVABLE_SELECTION_SHAPE}
        return {"applicable": False,
                "reason": "no split manifest named and no recorded experiment to read a "
                          "selection side from",
                **_NOT_APPLICABLE_SELECTION_SHAPE}

    from tcip_mcp.experiments import read_split_manifest

    split = read_split_manifest(experiment_id)
    manifest_binding = split.get("manifest_binding")
    if split_manifest_dir is None and not manifest_binding:
        return {"applicable": False,
                "reason": "no split manifest named and this checkpoint's own run was not bound "
                          "to one",
                **_NOT_APPLICABLE_SELECTION_SHAPE}
    if not split:
        return {"applicable": True,
                "reason": "the calibration names a split manifest but this checkpoint's run "
                          "recorded no split.json to check its selection side against",
                **_UNRESOLVABLE_SELECTION_SHAPE}
    if split.get("group_by") == "spatial_strip":
        return {"applicable": False,
                "reason": "the run drew a within-image spatial split; its own calibration_region "
                          "is the selection check for that route, not this one",
                **_NOT_APPLICABLE_SELECTION_SHAPE}
    val_stems = split.get("val") or []
    if not val_stems:
        return {"applicable": False, "reason": "the run's split.json carries no val members",
                **_NOT_APPLICABLE_SELECTION_SHAPE}
    if split.get("group_by") == "external":
        return {"applicable": False,
                "reason": "the run validated against an explicit val_images_dir; its date says "
                          "nothing about that directory's own membership",
                **_NOT_APPLICABLE_SELECTION_SHAPE}
    if calibration_date is None:
        return {"applicable": False,
                "reason": "this calibration derived no date to check against the run's own "
                          "selection date",
                **_NOT_APPLICABLE_SELECTION_SHAPE}
    record_date = split.get("date")
    if calibration_date != record_date:
        return {"applicable": False,
                "reason": f"the calibration is dated {calibration_date!r}, the run's own val is "
                          f"dated {record_date!r}; a bare stem means the same image only under "
                          "one date",
                **_NOT_APPLICABLE_SELECTION_SHAPE}

    cal_hold_stems = sorted(cal_ids | hold_ids)
    persisted_map = split.get("group_key_map") if split.get("group_by") == "explicit_map" else None
    resolved = _resolve_group_stem_disjointness(
        val_stems, cal_hold_stems, split.get("group_by"), persisted_map)
    label_digests_block = split.get("label_digests")
    moved = _resolve_label_movement(
        label_digests_block, cal_ids, calibration_labels_dir, split_manifest_sha256)
    reason = None
    if not label_digests_block or not label_digests_block.get("at_split"):
        reason = (
            "this run's split.json recorded no label_digests block, so a calibration label "
            "moved since the draw cannot be named: the run was bound before that block existed, "
            "or was calibrated with no bound run under a caller-named manifest"
        )
    return {"applicable": True, "reason": reason, **resolved, **moved}


def attach_split_policy_provenance(bundle: ResolvedBundle, locked: dict) -> None:
    """Copy the locked cal/holdout split's resolved policy + identity onto the conf param's gate
    evidence, so the operating-point provenance bundle is self-contained, a caller can see why
    particular ids ended up on which side without a separate lookup of the
    ``.tcip/artifacts/cal_holdout_split_<hash>.json`` lock file. Also carries any
    ``policy_divergence`` / ``unlocked_stems`` the lock resolution reported, so a declared-but-not-
    applied policy (a lock already existed with a different seed/ratio/grouping) is visible in the
    result, not only in a server log line.

    In-place: ``ResolvedParam`` is a frozen dataclass, but the ``gate_evidence`` dict it holds is an
    ordinary mutable dict, so mutating its contents (never reassigning the attribute) is safe. A
    no-op when the bundle has no calibrated ``conf`` gate evidence to attach to (e.g. no GT at all).
    """
    conf = bundle.params.get("conf")
    if conf is None or conf.gate_evidence is None:
        return
    conf.gate_evidence["split_policy"] = {
        "group_by": locked.get("group_by"), "group_key_map": locked.get("group_key_map"),
        "seed": locked.get("seed"), "holdout_ratio": locked.get("holdout_ratio"),
        "identity_hash": locked.get("identity_hash"),
        "split_manifest_dir": locked.get("split_manifest_dir"),
    }
    if locked.get("policy_divergence"):
        conf.gate_evidence["split_policy_divergence"] = locked["policy_divergence"]
    if locked.get("unlocked_stems"):
        conf.gate_evidence["split_unlocked_stems"] = locked["unlocked_stems"]


def attach_spatial_split_kind_provenance(bundle: ResolvedBundle, spatial: dict) -> None:
    """Same target and shape as :func:`attach_split_policy_provenance`, for a block-calibrated
    bundle whose reference came from a mosaic's own persisted spatial-strip split (``spatial``,
    ``split.json``'s ``spatial`` manifest) rather than a locked cal/holdout draw over a labeled
    image set: there is no ``locked`` dict here to read a group policy off, only the split's own
    recorded geometry, so this writes the split-kind fact directly instead of reusing
    ``attach_split_policy_provenance``'s ``locked``-shaped signature. A no-op when the bundle has
    no calibrated ``conf`` gate evidence to attach to, same as its sibling.
    """
    conf = bundle.params.get("conf")
    if conf is None or conf.gate_evidence is None:
        return
    conf.gate_evidence["split_policy"] = {
        "group_by": "spatial_strip", "seed": spatial.get("seed"),
        "tile_size": spatial.get("tile_size"), "overlap": spatial.get("overlap"),
    }


def resolve_operating_point(
    trait_name: str,
    *,
    dataset_hash: str | None,
    calibration_records: list[dict] | None = None,
    holdout_records: list[dict] | None = None,
    tile_size: int | None = None,
    tile_size_source: str = "default",
    tile_size_derived_from: str | None = None,
    tiled: bool | None = None,
    tiled_source: str = "default",
    cross_tile_nms: float | None = None,
    max_dets: int | None = None,
    max_dets_derived_from: str | None = None,
    validated_reference: str = VALIDATED_HELD_OUT,
    experiment_id: str | None = None,
    staged_conf_floor: float | None = None,
    staged_conf_floor_attribute_path: str | None = None,
    adjudication_covered: Callable[[dict], bool] | None = None,
    cal_rects: dict[str, tuple[int, int, int, int]] | None = None,
    hold_rects: dict[str, tuple[int, int, int, int]] | None = None,
    split_manifest_dir: str | None = None,
    calibration_date: str | None = None,
    calibration_labels_dir: str | None = None,
    split_manifest_sha256: str | None = None,
) -> ResolvedBundle:
    """Resolve the operating point for (trait, dataset). Pure over records, callers pass the model
    pass output; ``records_over_loader`` produces it. ``tile_size`` may be model-derived (imgsz).

    ``cal_rects``/``hold_rects`` are optional and additive, forwarded verbatim to
    :func:`_train_disjointness` (see its own docstring): every existing caller omits them and gets
    exactly today's lexical/stem-based disjointness check, unchanged. A block-calibration caller
    (a mosaic's own reserved calibration/test regions) supplies them to get the geometric
    containment check instead, the only shape that can prove disjointness for a within-mosaic
    reference with no separate image identity of its own.

    ``max_dets_derived_from`` is how a caller-supplied ``max_dets`` was produced, in the caller's own
    words, and is the only way a caller-supplied cap earns a derivation label here: a caller that
    supplies a number without saying where it came from gets an explicit "caller override" stamp
    rather than this function's density-formula label on a number this function did not derive. It
    is ignored when ``max_dets`` is ``None``, since the cap is then derived here and labeled here.

    ``tile_size_source``/``tiled_source`` are the caller's own resolution of whether each value was
    an explicit override, derived from the checkpoint's persisted training geometry, or a documented
    default, not inferred here from mere truthiness. A truthy ``tile_size`` is not proof of
    derivation: a caller with no persisted geometry and no explicit value still passes a concrete
    fallback number that ``tile_size_source`` must distinguish from a genuinely derived one.

    ``validated_reference`` is the stamp a *passing* held-out gate earns: ``VALIDATED_HELD_OUT`` when
    the records came from GT annotations (default), ``VALIDATED_REVIEW_CONFIRMED`` when they were
    reconstructed from a breeder-confirmed sample of the model's own outputs
    (feedback.review_calibration), the two references ``accepted_references("annotations")``
    recognizes. Both pass the same disjoint + count-bias gate here; the stamp only records which one
    it was.

    ``experiment_id`` is the checkpoint's own training-run id, if known, it gates the same held-out
    pass on train-disjointness (the cal/holdout images must not also be in that run's training
    split); ``None`` (a foreign/unregistered checkpoint) skips the check rather than failing closed,
    since only a *known* run whose provenance can't be resolved should refuse.

    ``staged_conf_floor`` is the floor the reference's predictions were actually generated / filtered
    at, a caller-supplied fact, not inferred from the reference's own scores. ``None`` (the caller
    asserted nothing) gates as ``conf_floor_unstated``, a distinct name from ``conf_censored`` (a
    stated floor the picked conf does not clear): see ``_conf_censored``. The GT/calibration callers
    thread the value ``set_detector_operating_point`` actually applied; the review path threads
    ``max(generation_conf, review_conf_threshold)`` through the same seam
    (``feedback.review_calibration.resolve_operating_point_from_review``, computed at
    ``routes/review.py``, which has no floor to apply through a model and so has no attribute path
    either). ``staged_conf_floor_attribute_path`` is the module attribute the floor was applied on
    (``set_detector_operating_point``'s own ``"attribute_path"``), or ``None`` when the floor has no
    such producer; true for every producer of an unstated floor, never only the bespoke-module case.

    ``tile_size_derived_from`` is forwarded to :func:`~tcip_mcp.pipelines.resolution.
    resolve_tile_size_param` unchanged; it matters only for ``tile_size_source == "explicit"``. The
    GT/calibration callers compose it through
    :func:`~tcip_mcp.pipelines.inference.predictor.explicit_edge_provenance`; the review path
    forwards the stored stamp's own text (it holds no predictor to compose one from).

    ``split_manifest_dir``/``calibration_date`` gate ``selection_disjointness``, alongside
    ``train_disjointness``: whether the cal/holdout images were also on the checkpoint's own
    selection side (``split.json``'s ``val``), checked when either this calibration names a
    manifest or the checkpoint's own record carries a ``manifest_binding``, not-applicable
    otherwise (see :func:`_selection_disjointness`). A leak or an unresolvable check blocks
    ``passed`` the same way a train-disjointness one does. ``calibration_labels_dir`` is the
    directory this calibration read its own records from, so a label rewritten since the run
    bound can be named against the calibration's own live copy; ``split_manifest_sha256`` is the
    digest of the manifest record this calibration read, when it named one, so a manifest
    redrawn since the run bound is visible. Both feed :func:`_selection_disjointness` unchanged
    and open nothing themselves when omitted.

    ``adjudication_covered``: an optional per-record predicate, when given, it is a gate, not a
    filter: every calibration and holdout record must satisfy it, or the whole reference is refused
    (``insufficient_adjudication_coverage``), before any bias/dispersion statistic is computed.
    Records are never silently dropped and re-measured on the survivors, a per-record filter here is
    a fail-open: the excluded set is correlated with the very quantity being measured (an image
    survives only if a miss was attested), so a biased population could earn a clean stamp on a
    favorable subsample while the full reviewed population was off by several times the trait
    tolerance. ``None`` (the default; every GT-path caller) applies no requirement, correct there,
    since a labeled record is inherently adjudication-covered.
    ``feedback.review_calibration.resolve_operating_point_from_review`` passes the per-image
    FN-adjudication coverage predicate.
    """
    if validated_reference not in accepted_references("annotations"):
        raise ValueError(f"validated_reference must be one of {accepted_references('annotations')}, "
                         f"got {validated_reference!r}")
    if tiled is None:
        raise ValueError(
            "resolve_operating_point requires an explicit tiled=<bool>: this function is pure over "
            "records and carries no predictor to derive it from. The caller (which has a predictor "
            "in scope) must resolve it first, typically `predictor.train_tile_size is not None`, "
            "and pass that concrete bool here; never a silently-defaulted value."
        )
    trait = get_trait(trait_name)
    # "not yet authored for this trait" falls back to the platform's interim default fraction,
    # the same shape resolve_classifier_operating_point resolves its own kappa floor with.
    count_bias_tolerance_frac = (
        trait.count_bias_tolerance_frac if trait.count_bias_tolerance_frac is not None
        else _DEFAULT_COUNT_BIAS_TOLERANCE_FRAC)
    review = validated_reference == VALIDATED_REVIEW_CONFIRMED
    # This is a gate, not a filter, see the docstring above for why a filter fails open: every
    # record must satisfy the predicate or the whole reference is refused, unfiltered, further down.
    adjudication_ok = adjudication_covered is None or (
        all(adjudication_covered(r) for r in (calibration_records or []))
        and all(adjudication_covered(r) for r in (holdout_records or []))
    )
    # count_objective is a recorded breeder decision when the trait spec has one; an unset trait
    # defaults to COUNT_UNBIASED (errors canceling is the right tolerance for a fraction/ratio
    # phenotype, the common case) rather than refusing to calibrate at all. Nobody, not the agent,
    # not the breeder, can meaningfully answer "does every object need to be found correctly, or is
    # it fine if errors cancel out" before any result exists to judge; the real confirmation point
    # is the delivered result itself, via the review-confirmation loop, not a blind precondition.
    # Stamped as a real ResolvedParam below so a caller can see whether this run's objective was
    # breeder-authored or an agent default.
    count_objective_explicit = bool(trait.count_objective)
    count_objective = trait.count_objective or COUNT_UNBIASED
    if count_objective not in COUNT_OBJECTIVE_PICKERS:
        raise ValueError(
            f"trait {trait_name!r}'s count_objective {count_objective!r} has no registered "
            f"picker in COUNT_OBJECTIVE_PICKERS ({sorted(COUNT_OBJECTIVE_PICKERS)}), register one "
            "(a new picker function + a new entry in this dict) before calibrating this trait."
        )
    picker, base_label = COUNT_OBJECTIVE_PICKERS[count_objective]
    conf_derived_from = base_label + (REVIEW_VERDICT_LABEL_SUFFIX if review else "")
    params: dict[str, ResolvedParam] = {}
    if count_objective_explicit:
        params["count_objective"] = default("count_objective", count_objective,
                                            derived_from="trait-authored")
    else:
        params["count_objective"] = default(
            "count_objective", count_objective,
            derived_from="platform default (fraction/ratio phenotype tolerates canceling errors); "
                         "not breeder-confirmed, judge the delivered result instead")

    # --- conf: the count operating point (calibration) ---
    if calibration_records:
        # Derived once from the calibration GT's own nearest-neighbor spacing, then reused for the
        # holdout tolerance below too, the same "exact-conf, not independently re-picked"
        # discipline already applied to conf, never re-derived per side, or calibration and
        # holdout could disagree on what "a hit" means.
        loc_frac = derive_localization_tolerance_frac(
            [[a["bbox"] for a in rec.get("gt", [])] for rec in calibration_records])
        if loc_frac is not None:
            params["localization_tolerance_frac"] = derived(
                "localization_tolerance_frac", loc_frac,
                derived_from="GT nearest-neighbor spacing (p10 + margin)")
        else:
            loc_frac = trait.localization_tolerance_frac
            params["localization_tolerance_frac"] = default(
                "localization_tolerance_frac", loc_frac,
                derived_from="trait default (underivable: no same-class neighbor in this GT)")
        tol = loc_frac * gt_class_avg_size(calibration_records)
        cal_curve = derive_operating_point_curve(calibration_records, tolerance=tol)
        conf = picker(cal_curve)
        conf = DEFAULT_CONF if conf is None else conf
        # conf-censoring guard: a count-unbiased 'validated' claim is only honest if the picked conf
        # sits strictly above the floor the reference was staged at, not merely if the reference's
        # own scores happen to look low (that predicate is unfalsifiable from a caller who
        # mis-asserts the floor; see _floor_mismatch for the reconciling check).
        censored = _conf_censored(conf, staged_conf_floor)
        floor_mismatch = _floor_mismatch(calibration_records, staged_conf_floor)
        if holdout_records:
            # Disjointness can only be proven from image_ids, so fail closed (not disjoint) when
            # either set has none, else the same records passed as cal+holdout look validated.
            cal_ids = {r["image_id"] for r in calibration_records if "image_id" in r}
            hold_ids = {r["image_id"] for r in holdout_records if "image_id" in r}
            disjoint = bool(cal_ids) and bool(hold_ids) and not (cal_ids & hold_ids)
            floor_mismatch = floor_mismatch or _floor_mismatch(holdout_records, staged_conf_floor)
            hold_tol = loc_frac * gt_class_avg_size(holdout_records)  # same frac as calibration, above
            # Exact-conf evaluation, not a nearest-grid-point snap, an explicit single-point
            # conf_grid makes derive_operating_point_curve evaluate exactly the conf that will ship, never
            # an approximation from the holdout's own independently-built grid (which need not
            # contain, or be anywhere near, the calibration-chosen conf).
            hold_curve = derive_operating_point_curve(holdout_records, tolerance=hold_tol, conf_grid=[conf])
            hb = hold_curve["curve"][0]  # the exact-conf holdout bias entry
            # The calibration side re-measured at the shipped conf (not read off its own grid, which
            # need not contain it), the only comparable basis for asking which classes the holdout
            # was actually able to check, below.
            cb = derive_operating_point_curve(calibration_records, tolerance=tol, conf_grid=[conf])["curve"][0]
            # content-overlap gate: a holdout whose GT content is fully cloned from calibration
            # (same boxes, different image_id) can't function as an independent check.
            content = _content_overlap(calibration_records, holdout_records)
            # train-disjointness gate: the cal/holdout images must not also be in the producing
            # checkpoint's own training split, or the "held-out" bias check is measured partly on
            # data the model already trained on.
            td = _train_disjointness(
                experiment_id, cal_ids, hold_ids, cal_rects=cal_rects, hold_rects=hold_rects)
            sd = _selection_disjointness(
                experiment_id, cal_ids, hold_ids, split_manifest_dir=split_manifest_dir,
                calibration_date=calibration_date,
                calibration_labels_dir=calibration_labels_dir,
                split_manifest_sha256=split_manifest_sha256)

            # Positive-evidence, unconditional, stated per-side (not a union), an all-negative
            # reference on either side can't validate a count operating point.
            cal_gt_count = sum(len(r.get("gt", [])) for r in calibration_records)
            hold_gt_count = sum(len(r.get("gt", [])) for r in holdout_records)
            # The mean+SE equivalence/CI criterion, replacing a bare mean check, degrades correctly
            # at small n (SE grows, so less evidence is harder to pass, not easier) and needs no
            # second, unrelated tolerance constant.
            #
            # The tolerance `_bias_equivalence_ok` compares against is relative, the breeder-authored
            # fraction scaled by this scope's own typical per-image count, derived here from the
            # holdout GT alone (never calibration): `gt_class_avg_size`'s `loc_frac`/tolerance split
            # just above is the precedent for this shape (a shared, calibration-derived policy
            # multiplied by a scale measured on the side it applies to, never a scale borrowed from
            # the other side), the pooled/per-class typical counts here are that same scale, measured
            # on the holdout because they gate the holdout's own bias. Deriving from calibration too
            # would let a caller buy a looser holdout tolerance by padding calibration with denser
            # images of a class, with no compensating cost in the holdout's own measured bias, a new,
            # unconstrained lever the locked-split discipline (`resolve_locked_cal_holdout_split`)
            # does not otherwise close, since it balances total annotation count per group, not
            # per-class density between sides.
            #
            # Both sides of the comparison are measured over the same population: the images that
            # actually carry the thing being counted. `mean_of_present_counts` already scopes the
            # typical count that way, so the bias must be scoped that way too. An image with no GT
            # and no surviving detection contributes a certain zero to the bias and nothing to the
            # density, and counting it on one side only divides the measured bias by
            # `n_images / n_present` while leaving the tolerance untouched, so a reference carrying
            # confirmed negatives reads a systematic miscount as that fraction of itself. The
            # equivalence test's own sample size is over-counted by the same term (`n_images` counts
            # images the bias does not rest on), the dilution the per-class scope's `n_present`
            # denominator already closes.
            pooled_typical = gt_class_typical_count(holdout_records)
            count_bias_ok = _bias_equivalence_ok(
                hb["count_bias_mean_present"], hb["count_bias_std_present"], hb["n_present"],
                tolerance_frac=count_bias_tolerance_frac, typical_count=pooled_typical)
            # The pooled test above is blind to a per-class error. Its matcher ignores category, so a
            # detector that calls every class-A object class B scores tp-only with zero bias, and one
            # that over-detects A exactly as much as it under-detects B nets to zero too, either way
            # a phenotype built from per-class counts (an elongated fraction, a per-class total) is
            # wrong while the stamp says validated. So every class the holdout carries must clear the
            # same equivalence test at the same trait tolerance, in the same per-image-mean unit, over
            # the same images (a class absent from an image contributes a zero bias there, exactly as
            # the pooled term does). Which class is the trait's positive one is deliberately not
            # consulted: that needs a name->id registry read this does not have, and requiring every
            # class to be unbiased is the stronger claim anyway.
            holdout_typical_by_class = {
                cid: gt_class_typical_count(holdout_records, class_id=int(cid))
                for cid in hb["per_class"]
            }
            per_class_bias_failures = sorted(
                cid for cid, s in hb["per_class"].items()
                if not _bias_equivalence_ok(s["count_bias_mean_present"], s["count_bias_std_present"],
                                            s["n_present"],
                                            tolerance_frac=count_bias_tolerance_frac,
                                            typical_count=holdout_typical_by_class[cid]))
            # The pooled scope's own reference-sufficiency floor (`hb["n_present"] < 2` below) is
            # scoped to the whole reference, not to one class, and the per-class relative tolerance's
            # floor (1/n_present) means a class present on exactly one holdout image gets a tolerance
            # derived from that single image's own density, reachable end to end without any
            # adversarial construction (an ordinary rare class that happens to show up once, in an
            # unusually dense frame). Same positive-evidence discipline the pooled
            # `insufficient_holdout_gt`/`insufficient_holdout_images` conjuncts already apply: one
            # image is not enough evidence to certify any class's count bias, regardless of what
            # tolerance it would otherwise clear.
            #
            # Exactly ``== 1``, not ``< 2``: ``n_present == 0`` is a different, already-correctly-
            # named situation, a class with zero holdout presence at the shipped conf is exactly
            # what ``holdout_missing_class`` below (when the class was evidenced in calibration) or
            # the ordinary ``count_bias_exceeds_tolerance_per_class`` path
            # (``_bias_equivalence_ok``'s own ``n == 0`` branch, when it wasn't) already name
            # correctly. Catching it here too would let this failure's breeder message ("held back in
            # exactly one image") win the first-match lookup over the true, more specific one for a
            # class held back in no images at all.
            per_class_insufficient_images = sorted(
                cid for cid, s in hb["per_class"].items() if s["n_present"] == 1)
            # ...and a class the holdout never carries is not a class that passed: its entry is all
            # zeros, so the test above reads bias 0.0 and says nothing. A class confused entirely
            # within the calibration half would otherwise read clean on every per-class entry the
            # gate can see, and the stamp would land anyway (reproduced in
            # `test_a_class_the_holdout_never_carries_cannot_be_validated_by_its_absence`). So every
            # class the calibration reference actually evidences at the shipped conf must be
            # evidenced in the holdout too, the same positive-evidence rule (never an inference from
            # absence) the per-side `insufficient_*_gt` conjuncts already apply to the pooled count.
            holdout_missing_classes = sorted(classes_with_evidence(cb) - classes_with_evidence(hb))
            # A trait-authored floor (no platform default): held-out precision and recall of the
            # governing criterion at the shipped conf must both clear it; unauthored gates below.
            holdout_match_quality_floor = trait.holdout_match_quality_floor
            localization_floor_ok = (
                holdout_match_quality_floor is not None
                and hb["precision"] >= holdout_match_quality_floor
                and hb["recall"] >= holdout_match_quality_floor)
            # A p90 tail dispersion floor, gated only once a real value is authored for this trait
            # (no invented default, see TraitSpec.count_error_tolerance).
            dispersion_ok = (
                trait.count_error_tolerance is None
                or hb["count_error_p90"] <= trait.count_error_tolerance
            )
            # Both conjuncts above stay pooled while count bias is per-class, and the per-class
            # statistics they would need are computed and persisted beside them. This is deliberate,
            # not an oversight: each is its own measurement question rather than a mechanical repeat
            # of the bias one, a per-class localization floor refuses a rare class whose single
            # detection lands just outside tolerance, and a per-class dispersion floor reads a
            # tolerance no trait has authored (count_error_tolerance is None everywhere today).
            # Every equivalence test above, pooled and per-class alike, reads its scope's own
            # present-scoped bias, dispersion and evidence count, so the bias and the typical count
            # it is judged against are measured over the same images and a scope scarce in the
            # reference borrows neither a diluted bias nor statistical confidence it doesn't have.
            # count_bias_tolerance_frac is relative, a fraction of each scope's own derived typical
            # per-image count, rather than a flat absolute value.

            # Named-failure architecture: every gate condition below is named here, once, so
            # describe_review_validation (and any future caller) maps failures to breeder-legible
            # reasons from this same list rather than re-deriving which check actually failed.
            # cap-saturation is intentionally absent, it is non-gating provenance only.
            # conf_floor_mismatch is also non-gating: its pinned +/-0.05 band is an ordinary property
            # of a model's score distribution as often as it is evidence of tampering, and gating on
            # it would re-create the kind of unsound pinned-constant refusal this design avoids. It
            # is still computed and surfaced in gate_evidence for a human/agent to notice, never
            # blocking.
            failures: list[str] = []
            if not adjudication_ok:
                failures.append("insufficient_adjudication_coverage")
            if not disjoint:
                failures.append("not_disjoint")
            if staged_conf_floor is None:
                failures.append("conf_floor_unstated")
            elif censored:
                failures.append("conf_censored")
            if content["shared"]:
                failures.append("content_shared_with_calibration")
            if td["unresolvable"]:
                failures.append("train_disjointness_unresolvable")
            if td["leaked_groups"] or td["leaked_stems"]:
                failures.append("train_disjointness_leaked")
            if sd["applicable"] and sd["unresolvable"]:
                failures.append("selection_disjointness_unresolvable")
            if sd["applicable"] and (sd["leaked_groups"] or sd["leaked_stems"]):
                failures.append("selection_disjointness_leaked")
            if cal_gt_count == 0:
                failures.append("insufficient_calibration_gt")
            if hold_gt_count == 0:
                failures.append("insufficient_holdout_gt")
            # Scoped to the images that carry the thing being counted, the same population the
            # pooled equivalence test above measures over: a holdout of a hundred images where only
            # one carries anything is one image worth of evidence about count bias, whatever the
            # total is. This is also what keeps _effective_count_bias_tolerance's own 1/n floor
            # bounded below 0.5 wherever it is actually reachable.
            if hb["n_present"] < 2:
                failures.append("insufficient_holdout_images")
            if not count_bias_ok:
                failures.append("count_bias_exceeds_tolerance")
            if per_class_bias_failures:
                failures.append("count_bias_exceeds_tolerance_per_class")
            if per_class_insufficient_images:
                failures.append("insufficient_holdout_images_per_class")
            if holdout_missing_classes:
                failures.append("holdout_missing_class")
            if holdout_match_quality_floor is None:
                failures.append("holdout_match_quality_floor_unauthored")
            elif not localization_floor_ok:
                failures.append("localization_quality_floor_failed")
            if not dispersion_ok:
                failures.append("count_error_dispersion_too_high")
            passed = not failures

            gate_evidence = {"calibration": cal_curve, "f1_max_conf": pick_f1_max(cal_curve),
                          "holdout_bias": hb,
                          "count_bias_tolerance_frac": count_bias_tolerance_frac,
                          "count_bias_tolerance_frac_source": (
                              "trait" if trait.count_bias_tolerance_frac is not None
                              else "default"),
                          "holdout_match_quality_floor": holdout_match_quality_floor,
                          # Reconstructibility: the fraction alone does not say what a pass/refusal
                          # actually compared against, the derived typical count and the resulting
                          # absolute tolerance, per scope, so a reviewer can rebuild the gate's own
                          # arithmetic from this record alone.
                          "pooled_typical_count": pooled_typical,
                          "pooled_count_bias_tolerance": _effective_count_bias_tolerance(
                              count_bias_tolerance_frac, pooled_typical, hb["n_present"]),
                          "per_class_typical_count": holdout_typical_by_class,
                          "per_class_count_bias_tolerance": {
                              cid: _effective_count_bias_tolerance(
                                  count_bias_tolerance_frac, holdout_typical_by_class[cid],
                                  s["n_present"])
                              for cid, s in hb["per_class"].items()
                          },
                          "per_class_count_bias_failures": per_class_bias_failures,
                          "per_class_insufficient_images": per_class_insufficient_images,
                          "holdout_missing_classes": holdout_missing_classes,
                          "calibration_bias_at_conf": cb,
                          "count_error_tolerance": trait.count_error_tolerance,
                          "equivalence_z": _EQUIVALENCE_Z,
                          "disjoint": disjoint, "conf_censored": censored,
                          "conf_floor_mismatch": floor_mismatch, "staged_conf_floor": staged_conf_floor,
                          "staged_conf_floor_attribute_path": staged_conf_floor_attribute_path,
                          "adjudication_covered": adjudication_ok,
                          "passed_holdout": passed, "failures": failures,
                          "content_overlap_frac": content["content_overlap_frac"],
                          "content_shared_with_calibration": content["shared"],
                          "train_disjointness": td, "selection_disjointness": sd,
                          "calibration_image_ids": sorted(cal_ids), "holdout_image_ids": sorted(hold_ids),
                          "calibration_observed_min_score": _min_dt_score(calibration_records),
                          "holdout_observed_min_score": _min_dt_score(holdout_records),
                          "calibration_cap_saturated_frac": _cap_saturated_frac(calibration_records),
                          "holdout_cap_saturated_frac": _cap_saturated_frac(holdout_records)}
            # validated only if the gate above raised no named failure, not merely because a
            # holdout was supplied. Reference here is the annotations/verdicts, not truth; the stamp
            # records which reference (GT vs review-confirmed) cleared the gate.
            validated = validated_reference if passed else VALIDATED_FALSE
        else:
            gate_evidence = {"calibration": cal_curve, "conf_censored": censored,
                          "conf_floor_mismatch": floor_mismatch, "staged_conf_floor": staged_conf_floor,
                          "staged_conf_floor_attribute_path": staged_conf_floor_attribute_path,
                          "calibration_observed_min_score": _min_dt_score(calibration_records),
                          "calibration_cap_saturated_frac": _cap_saturated_frac(calibration_records),
                          "note": "calibrated but not held-out-measured"}
            validated = VALIDATED_FALSE
        params["conf"] = derived("conf", float(conf),
                                 derived_from=conf_derived_from,
                                 requires_validation=True, validation_kind="annotations",
                                 validated_against=validated, dataset_scoped=True,
                                 dataset_hash=dataset_hash, gate_evidence=gate_evidence)
        if max_dets is None:
            max_dets = _max_dets_from_density(calibration_records)
            max_dets_derived_from = "~1.5x p99 GT objects/image"
    else:
        # No GT for this dataset: cannot calibrate. Carry an unvalidated placeholder (un-shippable
        # via the firewall), no valley heuristic, no chosen value dressed as trustworthy.
        params["conf"] = derived("conf", DEFAULT_CONF,
                                 derived_from="no GT for this dataset; unvalidated placeholder",
                                 requires_validation=True, validation_kind="annotations",
                                 validated_against=VALIDATED_FALSE, dataset_scoped=True,
                                 dataset_hash=dataset_hash)
        params["localization_tolerance_frac"] = default(
            "localization_tolerance_frac", trait.localization_tolerance_frac,
            derived_from="trait default (no GT for this dataset)")

    # --- structural facts / distribution statistics / documented-default params ---
    # tile_size uses the same shared resolve_tile_size_param() raw_operating_point also calls.
    resolved_tiled = bool(tiled)  # already a concrete bool: the None-check above raised otherwise
    params["tile_size"] = resolve_tile_size_param(
        tile_size, tiled=resolved_tiled, tile_size_source=tile_size_source,
        tile_size_derived_from=tile_size_derived_from)
    if tiled_source == "explicit":
        params["tiled"] = ResolvedParam(
            "tiled", resolved_tiled, source="explicit", derived_from="caller override")
    else:
        params["tiled"] = default("tiled", resolved_tiled)
    # cross_tile_nms: an explicit override wins and is stamped as such; otherwise derive it from the
    # calibration GT's neighbor-IoU distribution; failing that (no GT / no genuine overlaps) an honest
    # default, never a derivation label on a number no derivation produced.
    if cross_tile_nms is not None:
        params["cross_tile_nms"] = ResolvedParam(
            "cross_tile_nms", float(cross_tile_nms), source="explicit",
            derived_from="caller override")
    else:
        nms = None
        if calibration_records:
            nms = derive_cross_tile_nms([[a["bbox"] for a in rec.get("gt", [])]
                                         for rec in calibration_records])
        params["cross_tile_nms"] = (
            derived("cross_tile_nms", nms,
                    derived_from="GT neighbor-IoU distribution (p99 + margin)")
            if nms is not None
            else default("cross_tile_nms", DEFAULT_NMS_IOU)
        )
    if max_dets is None:
        params["max_dets"] = default("max_dets", DEFAULT_MAX_DETS)
    elif max_dets_derived_from:
        params["max_dets"] = derived("max_dets", int(max_dets),
                                     derived_from=max_dets_derived_from)
    else:
        params["max_dets"] = ResolvedParam("max_dets", int(max_dets), source="explicit",
                                           derived_from="caller override")
    return ResolvedBundle(trait=trait_name, dataset_hash=dataset_hash, params=params)


def _classification_kappa(items: list[dict]) -> float | None:
    """Cohen's kappa between true and predicted positive/negative class over classification items.

    A derived-at-runtime compensating-error floor: a mean count-bias check alone is blind to a
    classifier that flips k true positives to negative and k true negatives to positive (net bias
    ~0), and the detection path's own localization-quality floor (``recall > 0 and precision > 0``)
    admits any single true positive regardless of how corrupted the rest of the population is, the
    same gap applies here. Kappa corrects for chance agreement from the reference's own observed base
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
    split_manifest_dir: str | None = None,
    calibration_date: str | None = None,
    calibration_labels_dir: str | None = None,
    split_manifest_sha256: str | None = None,
) -> dict:
    """Classification-mode calibration gate for a trait's positive-class call.

    Mirrors :func:`resolve_operating_point`'s rigor for a classifier's call, not a detector's
    box-finding, calls the same shared primitives (:func:`_content_overlap`,
    :func:`_train_disjointness`) rather than reimplementing them, replacing the detection path's
    localization-quality floor with a derived compensating-error floor (:func:`_classification_kappa`)
    since there is no bbox-match concept here.

    Each item in ``calibration_items``/``holdout_items`` is one classified, already-localized
    instance: ``{"image_id": str, "is_true_positive": bool, "is_pred_positive": bool,
    "bbox": [x1, y1, x2, y2]}``, whether the GT/reviewer-confirmed label and the classifier's own
    call are the trait's positive state, plus the instance's own GT geometry (required, see
    ``_grouped_records`` below for why a placeholder box cannot substitute for it).

    Returns a dict structurally distinct from a ``ResolvedParam``/``ResolvedBundle``, never a
    shape a generic writer could mistake for the count operating point's ``conf`` param and stamp
    into the wrong sidecar:
    ``{"validated_against", "passed", "failures", "gate_evidence"}``. Callers write this into a
    classifier-scoped sidecar (``classifier_operating_point.json``, never ``operating_point.json``'s
    own fields) via :func:`tcip_mcp.pipelines.resolution.reconcile_classifier_validity`.

    ``experiment_id is None`` (a foreign/unregistered checkpoint) skips the train-disjointness check
    rather than failing closed, the same owner decision :func:`resolve_operating_point` follows, the
    classifier-validity *stamp* is still reachable for a foreign checkpoint whose cal/holdout is
    otherwise disjoint and unbiased; it is not reachable at all when no calibration/holdout is given.

    ``split_manifest_dir``/``calibration_date`` gate ``selection_disjointness`` the same way
    :func:`resolve_operating_point` does; the classifier door draws no manifest today, so
    ``split_manifest_dir`` is never populated, and its one caller states ``calibration_date``
    from its own calibration GT directory (``manifest_date_key`` for a flat one, never the bare
    ``annotation_date`` result). ``calibration_date is None`` means the caller derived no date at
    all and the check reads not-applicable with that reason; it is never read as matching a flat
    run's own record, whose ``date`` is the same empty-string key, not ``None`` either.
    ``calibration_labels_dir`` is its caller's ``calibration_gt_dir``, forwarded unchanged so a
    label rewritten since the run bound can be named; ``split_manifest_sha256`` stays ``None``
    here, the same way ``split_manifest_dir`` does, since this door reads no manifest.
    """
    if validated_reference not in accepted_references("annotations"):
        raise ValueError(f"validated_reference must be one of {accepted_references('annotations')}, "
                         f"got {validated_reference!r}")
    get_trait(trait_name)  # validates the trait exists; classification mode needs no trait-shaped fields today
    if not calibration_items or not holdout_items:
        return {
            "validated_against": VALIDATED_FALSE, "passed": False,
            "failures": ["no_calibration_or_holdout"],
            "gate_evidence": {"note": "classifier calibration requires both calibration and holdout items"},
        }

    adjudication_ok = adjudication_covered is None or (
        all(adjudication_covered(r) for r in calibration_items)
        and all(adjudication_covered(r) for r in holdout_items)
    )

    cal_ids = {it["image_id"] for it in calibration_items if "image_id" in it}
    hold_ids = {it["image_id"] for it in holdout_items if "image_id" in it}
    disjoint = bool(cal_ids) and bool(hold_ids) and not (cal_ids & hold_ids)

    # Each item list is grouped by image_id once and the grouping reused below (content-overlap's
    # bbox fingerprint and the per-image bias stats), never re-grouped a second time.
    def _group_by_image(items: list[dict]) -> dict[str | None, list[dict]]:
        by_image: dict[str | None, list[dict]] = {}
        for it in items:
            by_image.setdefault(it.get("image_id"), []).append(it)
        return by_image

    def _content_record(image_id: str | None, items: list[dict]) -> dict:
        return {"image_id": image_id, "width": 0, "height": 0,
                "gt": [{"category_id": 1 if it["is_true_positive"] else 0, "bbox": it["bbox"]}
                      for it in items]}

    cal_by_image = _group_by_image(calibration_items)
    hold_by_image = _group_by_image(holdout_items)
    content = _content_overlap(
        [_content_record(iid, its) for iid, its in cal_by_image.items()],
        [_content_record(iid, its) for iid, its in hold_by_image.items()])
    td = _train_disjointness(experiment_id, cal_ids, hold_ids)
    sd = _selection_disjointness(
        experiment_id, cal_ids, hold_ids, split_manifest_dir=split_manifest_dir,
        calibration_date=calibration_date, calibration_labels_dir=calibration_labels_dir,
        split_manifest_sha256=split_manifest_sha256)

    cal_pos = sum(1 for it in calibration_items if it["is_true_positive"])
    hold_pos = sum(1 for it in holdout_items if it["is_true_positive"])
    trait = get_trait(trait_name)
    # "not yet authored for this trait" falls back to the platform's interim default fraction,
    # the same shape `agreement_floor` below resolves its own kappa floor with.
    count_bias_tolerance_frac = (
        trait.count_bias_tolerance_frac if trait.count_bias_tolerance_frac is not None
        else _DEFAULT_COUNT_BIAS_TOLERANCE_FRAC)
    # Per-image mean count-bias (mean+SE equivalence, as the detection path gates on), present-
    # scoped like typical_positive_count below: an all-negative image would dilute the measured bias.
    per_image_bias = []
    for its in hold_by_image.values():
        n_pred_pos = sum(1 for it in its if it["is_pred_positive"])
        n_true_pos = sum(1 for it in its if it["is_true_positive"])
        if n_true_pos or n_pred_pos:
            per_image_bias.append(n_pred_pos - n_true_pos)
    n_bias_images = len(per_image_bias)
    count_bias = statistics.fmean(per_image_bias) if per_image_bias else 0.0
    # Sample stdev (ddof=1/Bessel's correction), matching the detection path's
    # np.std(biases, ddof=1) exactly, pstdev's population estimator was systematically more
    # permissive, worst at small n, which is exactly where the equivalence test's SE penalty is
    # supposed to bite hardest.
    count_bias_std = statistics.stdev(per_image_bias) if n_bias_images > 1 else 0.0
    # The same relative-tolerance shape the detection path uses, the positive class's own typical
    # per-image count, reusing `hold_by_image` rather than a second pass over `holdout_items`.
    typical_positive_count = mean_of_present_counts(
        sum(1 for it in its if it["is_true_positive"]) for its in hold_by_image.values())
    count_bias_ok = _bias_equivalence_ok(
        count_bias, count_bias_std, n_bias_images,
        tolerance_frac=count_bias_tolerance_frac, typical_count=typical_positive_count)

    kappa = _classification_kappa(holdout_items)
    # kappa is None only when the holdout is degenerate (a single class throughout), a real
    # reference for a trait with two states should not be, so treat that as a failure to derive
    # rather than a pass. Two floors: kappa > 0 is the universal,
    # domain-input-free minimum (better than pure chance, a classifier that flips a full 40% of
    # calls symmetrically, net count-bias ~0, clears this alone at kappa=0.2, exactly the
    # compensating-error case this check exists to catch); `agreement_floor` is the trait's own
    # authored bar (`TraitSpec.classifier_agreement_floor`), falling back to the platform's
    # interim default only when the trait hasn't set one.
    agreement_floor = (
        trait.classifier_agreement_floor
        if trait.classifier_agreement_floor is not None else _DEFAULT_KAPPA_FLOOR)
    compensating_error_ok = kappa is not None and kappa > 0.0 and kappa > agreement_floor

    failures: list[str] = []
    if not adjudication_ok:
        failures.append("insufficient_adjudication_coverage")
    if not disjoint:
        failures.append("not_disjoint")
    if content["shared"]:
        failures.append("content_shared_with_calibration")
    if td["unresolvable"]:
        failures.append("train_disjointness_unresolvable")
    if td["leaked_groups"] or td["leaked_stems"]:
        failures.append("train_disjointness_leaked")
    if sd["applicable"] and sd["unresolvable"]:
        failures.append("selection_disjointness_unresolvable")
    if sd["applicable"] and (sd["leaked_groups"] or sd["leaked_stems"]):
        failures.append("selection_disjointness_leaked")
    if cal_pos == 0:
        failures.append("insufficient_calibration_positive_evidence")
    if hold_pos == 0:
        failures.append("insufficient_holdout_positive_evidence")
    if len(holdout_items) < 2:
        failures.append("insufficient_holdout_items")
    if n_bias_images < 2:
        # Same minimum the detection path requires (hb["n_present"] < 2, present-scoped like
        # n_bias_images now is): without this, a single-image holdout forces count_bias_std to 0.0
        # (no images to vary across), so the equivalence test's SE penalty vanishes and a lone image
        # can pass at exactly the tolerance
        # with zero uncertainty discount.
        failures.append("insufficient_holdout_images")
    if not count_bias_ok:
        failures.append("count_bias_exceeds_tolerance")
    if not compensating_error_ok:
        failures.append("compensating_error_floor_failed")
    passed = not failures

    gate_evidence = {
        "content_overlap_frac": content["content_overlap_frac"],
        "content_shared_with_calibration": content["shared"],
        "train_disjointness": td, "selection_disjointness": sd, "disjoint": disjoint,
        "adjudication_covered": adjudication_ok,
        "count_bias": count_bias, "count_bias_std": count_bias_std, "count_bias_n_images": n_bias_images,
        "count_bias_tolerance_frac": count_bias_tolerance_frac,
        "count_bias_tolerance_frac_source": ("trait" if trait.count_bias_tolerance_frac is not None
                                             else "default"),
        "typical_positive_count": typical_positive_count,
        # Never the bare "count_bias_tolerance" name once used for the authored value here: reusing
        # that exact name for the derived effective value would silently swap what the same key
        # means, the reuse-a-name-for-a-different-concept footgun CLAUDE.md's global rules warn
        # about. Deliberately not named to match the detector path's "pooled_count_bias_tolerance"
        # either, this sidecar has no "pooled" vs "per-class" split to distinguish from, so it needs
        # its own name, not a borrowed one.
        "count_bias_tolerance_absolute": _effective_count_bias_tolerance(
            count_bias_tolerance_frac, typical_positive_count, n_bias_images),
        "kappa": kappa, "kappa_floor": agreement_floor,
        "kappa_floor_source": ("trait" if trait.classifier_agreement_floor is not None
                               else "default"),
        "n_calibration": len(calibration_items), "n_holdout": len(holdout_items),
    }
    return {
        "validated_against": validated_reference if passed else VALIDATED_FALSE,
        "passed": passed, "failures": failures, "gate_evidence": gate_evidence,
    }


def _resolve_scalar_operating_point(
    trait_name: str,
    *,
    criterion: str,
    criteria: dict[str, Callable[[Any, Any], float | None]],
    true_key: str,
    pred_key: str,
    floor_field: str,
    default_floor: float,
    calibration_items: list[dict] | None,
    holdout_items: list[dict] | None,
    experiment_id: str | None,
    validated_reference: str,
    split_manifest_dir: str | None = None,
    calibration_date: str | None = None,
) -> dict:
    """Shared calibration-gate mechanics for :func:`resolve_ordinal_operating_point` and
    :func:`resolve_regression_operating_point`.

    Both validate a per-*image* scalar prediction (one rank or one continuous value per image,
    unlike the classifier path's per-instance, bbox-matched items: ``OrdinalDataset``/
    ``RegressionDataset`` are one CSV row per image stem, no bbox/geometry concept applies here)
    against a locked cal/holdout split the same way, disjointness, train-disjointness, then a
    derived compensating-error floor on the *holdout-only* criterion score, differing only in which
    criterion toolkit (``criteria``, ``ORDINAL_CRITERIA`` or ``REGRESSION_CRITERIA``) and which
    ``TraitSpec`` floor field apply. ``criterion`` is validated by both public callers before this is
    reached, this function trusts it is already a real key of ``criteria``.

    The criterion score is computed on holdout only, never calibration, which would be evaluating a
    criterion on the very data it was picked to look good on, not a validation of anything. A thin
    holdout (fewer than 2 items) still gets a real, non-fabricated score attempt (some criteria are
    defined, if noisy, on very few items; others return ``None``), it fails closed via the separate
    ``insufficient_holdout_items``/``criterion_undefined`` failures below rather than by silently
    substituting the calibration set.
    """
    trait = get_trait(trait_name)
    if not calibration_items or not holdout_items:
        return {
            "validated_against": VALIDATED_FALSE, "passed": False,
            "failures": ["no_calibration_or_holdout"],
            "gate_evidence": {"criterion": criterion,
                           "note": "calibration requires both calibration and holdout items"},
        }

    cal_ids = {it["image_id"] for it in calibration_items if "image_id" in it}
    hold_ids = {it["image_id"] for it in holdout_items if "image_id" in it}
    disjoint = bool(cal_ids) and bool(hold_ids) and not (cal_ids & hold_ids)
    # Reuses the shared train-disjointness primitive (stems/groups, no bbox). `_content_overlap`
    # fingerprints bbox content, which ordinal/regression items (one scalar each) carry none of.
    td = _train_disjointness(experiment_id, cal_ids, hold_ids)
    sd = _selection_disjointness(
        experiment_id, cal_ids, hold_ids, split_manifest_dir=split_manifest_dir,
        calibration_date=calibration_date)

    import torch

    holdout_true = torch.tensor([float(it[true_key]) for it in holdout_items])
    holdout_pred = torch.tensor([float(it[pred_key]) for it in holdout_items])
    score = criteria[criterion](holdout_pred, holdout_true)

    floor_authored = getattr(trait, floor_field)
    floor = floor_authored if floor_authored is not None else default_floor
    floor_source = "trait" if floor_authored is not None else "default"
    # A non-finite score compares false against every bound, so it is its own failure.
    score_state = non_finite_state(score) if isinstance(score, float) else None
    comparable = score is not None and score_state is None
    # score > 0.0 is the domain-input-free minimum (beat the criterion's chance baseline); floor
    # is the trait's authored bar or the platform's interim default, as in the kappa check.
    compensating_error_ok = comparable and score is not None and score > 0.0 and score > floor

    failures: list[str] = []
    if not disjoint:
        failures.append("not_disjoint")
    if td["unresolvable"]:
        failures.append("train_disjointness_unresolvable")
    if td["leaked_groups"] or td["leaked_stems"]:
        failures.append("train_disjointness_leaked")
    if sd["applicable"] and sd["unresolvable"]:
        failures.append("selection_disjointness_unresolvable")
    if sd["applicable"] and (sd["leaked_groups"] or sd["leaked_stems"]):
        failures.append("selection_disjointness_leaked")
    if len(holdout_items) < 2:
        failures.append("insufficient_holdout_items")
    if score is None:
        failures.append("criterion_undefined")
    elif score_state is not None:
        failures.append("criterion_not_finite")
    elif not compensating_error_ok:
        failures.append("compensating_error_floor_failed")
    passed = not failures

    gate_evidence = {
        "criterion": criterion, **stored_number("score", score),
        "floor": floor, "floor_source": floor_source,
        "disjoint": disjoint, "train_disjointness": td, "selection_disjointness": sd,
        "n_calibration": len(calibration_items), "n_holdout": len(holdout_items),
    }
    return {
        "validated_against": validated_reference if passed else VALIDATED_FALSE,
        "passed": passed, "failures": failures, "gate_evidence": gate_evidence,
    }


def resolve_ordinal_operating_point(
    trait_name: str,
    *,
    criterion: str,
    calibration_items: list[dict] | None = None,
    holdout_items: list[dict] | None = None,
    experiment_id: str | None = None,
    validated_reference: str = VALIDATED_HELD_OUT,
    split_manifest_dir: str | None = None,
    calibration_date: str | None = None,
) -> dict:
    """Ordinal-mode calibration gate for a trait's rank prediction.

    ``criterion`` is required, no default: which compensating-error statistic is scientifically
    appropriate for a given trait's calibration is a CV-scientist judgment call the caller makes
    explicitly (see ``ORDINAL_CRITERIA`` for the registered toolkit), never a platform-prescribed
    "the" statistic. Raises ``ValueError`` immediately, before any other work, when ``criterion``
    names no registered ordinal criterion.

    Each item in ``calibration_items``/``holdout_items`` is one image's rank prediction:
    ``{"image_id": str, "true_rank": int, "predicted_rank": int}`` (``OrdinalDataset`` is one CSV
    row per image stem, no bbox/geometry concept applies). Returns the same structurally-distinct
    shape :func:`resolve_classifier_operating_point` does: ``{"validated_against", "passed",
    "failures", "gate_evidence"}``, never a shape a generic writer could mistake for the count
    operating point's ``conf`` param. Callers write this into ``ordinal_operating_point.json`` via
    :func:`tcip_mcp.pipelines.resolution.reconcile_ordinal_validity`.

    ``split_manifest_dir``/``calibration_date`` gate ``selection_disjointness`` the same way
    :func:`resolve_operating_point` does; the caller's own CSV directory, when it sits under a
    capture date, is the date to state (``manifest_date_key`` for a flat one), ``None`` when it
    derived none, never a bare guess.

    See :func:`_resolve_scalar_operating_point` for the shared calibration mechanics (disjointness,
    train-disjointness, the holdout-only criterion score, the compensating-error floor).
    """
    if criterion not in ORDINAL_CRITERIA:
        raise ValueError(
            f"criterion {criterion!r} is not a registered ordinal criterion "
            f"({sorted(ORDINAL_CRITERIA)}); register a new criterion function in ORDINAL_CRITERIA "
            "before calibrating with it.")
    if validated_reference not in accepted_references("annotations"):
        raise ValueError(f"validated_reference must be one of {accepted_references('annotations')}, "
                         f"got {validated_reference!r}")
    return _resolve_scalar_operating_point(
        trait_name, criterion=criterion, criteria=ORDINAL_CRITERIA,
        true_key="true_rank", pred_key="predicted_rank", floor_field="ordinal_agreement_floor",
        default_floor=_DEFAULT_ORDINAL_AGREEMENT_FLOOR,
        calibration_items=calibration_items, holdout_items=holdout_items,
        experiment_id=experiment_id, validated_reference=validated_reference,
        split_manifest_dir=split_manifest_dir, calibration_date=calibration_date,
    )


def resolve_regression_operating_point(
    trait_name: str,
    *,
    criterion: str,
    calibration_items: list[dict] | None = None,
    holdout_items: list[dict] | None = None,
    experiment_id: str | None = None,
    validated_reference: str = VALIDATED_HELD_OUT,
    split_manifest_dir: str | None = None,
    calibration_date: str | None = None,
) -> dict:
    """Regression-mode calibration gate for a trait's continuous-value prediction.

    ``criterion`` is required, no default: ``r_squared`` and ``concordance_correlation_coefficient``
    (see ``REGRESSION_CRITERIA``) measure genuinely different things at different scales/conventions
    (R²: overall predictive skill vs. a trivial mean baseline, unbounded below; CCC: a bounded
    precision/accuracy decomposition, the standard measurement-agreement lens), so the caller states
    which one this calibration is judged against, never a platform default. Raises ``ValueError``
    immediately, before any other work, when ``criterion`` names no registered regression criterion.

    Each item in ``calibration_items``/``holdout_items`` is one image's value prediction:
    ``{"image_id": str, "true_value": float, "predicted_value": float}`` (``RegressionDataset`` is
    one CSV row per image stem, no bbox/geometry concept applies). Returns the same structurally-
    distinct shape :func:`resolve_classifier_operating_point` does: ``{"validated_against", "passed",
    "failures", "gate_evidence"}``. Callers write this into ``regression_operating_point.json`` via
    :func:`tcip_mcp.pipelines.resolution.reconcile_regression_validity`.

    ``split_manifest_dir``/``calibration_date`` gate ``selection_disjointness`` the same way
    :func:`resolve_ordinal_operating_point` states it.

    See :func:`_resolve_scalar_operating_point` for the shared calibration mechanics.
    """
    if criterion not in REGRESSION_CRITERIA:
        raise ValueError(
            f"criterion {criterion!r} is not a registered regression criterion "
            f"({sorted(REGRESSION_CRITERIA)}); register a new criterion function in "
            "REGRESSION_CRITERIA before calibrating with it.")
    if validated_reference not in accepted_references("annotations"):
        raise ValueError(f"validated_reference must be one of {accepted_references('annotations')}, "
                         f"got {validated_reference!r}")
    return _resolve_scalar_operating_point(
        trait_name, criterion=criterion, criteria=REGRESSION_CRITERIA,
        true_key="true_value", pred_key="predicted_value", floor_field="regression_skill_floor",
        default_floor=_DEFAULT_REGRESSION_SKILL_FLOOR,
        calibration_items=calibration_items, holdout_items=holdout_items,
        experiment_id=experiment_id, validated_reference=validated_reference,
        split_manifest_dir=split_manifest_dir, calibration_date=calibration_date,
    )
