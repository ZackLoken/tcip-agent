"""Reconstruct a calibration reference from human review verdicts.

Turns per-image review verdicts (the same shards ``materialize.py`` reads) into the COCO record
shape ``resolve_operating_point`` consumes, so a breeder-confirmed sample of the model's own
outputs can validate the count operating point, not only dense held-out GT (the shared-reference
principle, CLAUDE.md). Per record:

  - ``gt`` = the boxes the breeder affirmed exist: accepted/edited matches, confirmed misses (FN),
    and false-positives the breeder promoted to real (accepted FP). Rejected boxes never enter gt.
  - ``dt`` = the model's own predictions carried with their recorded confidence, regardless of
    verdict, so the sweep re-derives TP/FP/FN by center-matching dt against the affirmed gt exactly
    as the GT path does.

The review-confirmed reference passes the identical disjoint-split + count-bias gate the held-out-GT
path passes; ``resolve_operating_point`` stamps it ``VALIDATED_REVIEW_CONFIRMED`` (distinct from
``VALIDATED_HELD_OUT`` so provenance records which reference validated). The conf-censoring guard in
``resolve_operating_point`` still applies: verdicts whose predictions were staged above the display
floor are truncated and cannot stamp a validated claim, the reviewed predictions must have been
generated at a floored conf for the sweep to reach the low-conf tail.

Producer-identity scoping and FN-adjudication coverage both live here: every verdict (and
confirmed-negative image record) is scoped to the producing bucket(s) it was actually recorded
against before any gate statistic sees it, and every record carries whether the image was genuinely
adjudicated for missed objects, the two facts a review-confirmed reference must never let a caller
skip.

Torch-free.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tcip_mcp.pipelines.resolution import VALIDATED_REVIEW_CONFIRMED, ResolvedBundle

_POSITIVE_ACTIONS = {"accepted", "edited"}

# Every name resolve_operating_point can put in sweep["failures"] (cross-cutting named-failure
# architecture), paired with its breeder-facing message. describe_review_validation surfaces the
# message for every name present in "failures", not just the first, in this list's order, so
# order here is reading order for a breeder facing several at once, not a first-match-wins priority.
# Each message must therefore stay true on its own regardless of which other names are also present,
# never phrase one as if it were the only failure the reference could have: the
# localization/dispersion/per-class messages below must never assert "the counts agree"/"the total
# looks right", which becomes a false statement exactly when count_bias_exceeds_tolerance is also
# present, since all four gate conditions are computed independently and can co-occur,
# operating_point.py's failures.append calls have no mutual exclusion between them.
# This is the one place the name<->message association is spelled out, describe_review_validation's
# exhaustiveness check and its message selection both read from this same list, so there is no
# second, separately-maintained "known names" set to drift out of sync with it.
_FAILURE_MESSAGES: list[tuple[tuple[str, ...], str]] = [
    (("conf_censored",),
     "Not yet. The check can't see the borderline detections it needs, either the predictions were "
     "generated at too high a confidence cutoff, or this review session's own \"Conf ≥\" display "
     "filter hid the low-confidence detections from view (a session filter censors the reference "
     "exactly the same way a high generation cutoff does). If you set a Conf ≥ filter above 0, "
     "lower it and re-review the newly-visible detections first, that's usually the faster fix. If "
     "the filter is already at 0, re-run the predictions at a lower confidence, review those, then "
     "try again."),
    (("insufficient_adjudication_coverage",),
     'Not yet. At least one of these reviewed images shows no evidence that missed objects were '
     'checked for. For images that had no ground truth before this review: uncheck "Reviewed" on '
     'that image to unlock it, use the "mark missed object" tool at least once, even just to '
     'confirm nothing was missed, then mark it Reviewed again. Then try again.'),
    (("insufficient_calibration_gt", "insufficient_holdout_gt"),
     "Not yet. One side of the review split has no confirmed objects at all, an all-negative "
     "reference can't validate a count. Review some images with real objects represented on both "
     "sides, then try again."),
    (("insufficient_holdout_images",),
     "Not yet. Only one image was held back to check against, the platform needs at least two so "
     "it can judge how consistent the counts are, not just whether they happen to agree once. "
     "Review a few more images, then try again."),
    # One kind of object is held back, but on only a single image, not enough to judge its own count
    # consistency against, the same
    # evidence-sufficiency shape as insufficient_holdout_images above, scoped to one class rather
    # than the whole reference.
    (("insufficient_holdout_images_per_class",),
     "Not yet. One kind of object was held back to check, but only in a single image, the platform "
     "needs at least two images carrying that kind so it can judge how consistent its count is, not "
     "just whether it happened to agree once. Review a few more images containing that kind of "
     "object, then try again."),
    # An evidence-sufficiency refusal, so it sits with the two above rather than down among the
    # accuracy ones: nothing about the counts has been judged wrong, there is simply nothing held
    # back to judge this kind of object against.
    (("holdout_missing_class",),
     "Not yet. One kind of object appears in the images used to set the threshold but in none of "
     "the images held back to check it, so there is no independent evidence that the model counts "
     "that kind correctly. Review more images containing every kind of object you care about, then "
     "try again."),
    (("not_disjoint",),
     "Not yet. The reviewed images couldn't be split into independent groups to cross-check. Review "
     "more images, then try again."),
    (("train_disjointness_unresolvable",),
     "Not yet. This model's training record doesn't establish which images it trained on, so the "
     "platform can't confirm the reviewed images were actually held back. Retrain with the current "
     "data (which records this), or use a model whose training record is known."),
    (("train_disjointness_leaked",),
     "Not yet. Some of the reviewed images (or images from the same source, e.g. tiles of one "
     "photo) were also used to train this model, so they can't function as an independent check. "
     "Review a different set of images this model never trained on."),
    (("content_duplicated",),
     "Not yet. The held-back images you reviewed duplicate the calibration images' content, so they "
     "can't function as an independent check. Review a genuinely distinct set of images, then try "
     "again."),
    (("localization_quality_floor_failed",),
     "Not yet. On the held-back images, the model's predictions don't actually line up with what "
     "you confirmed, even where the counts agree, that agreement can be coincidental rather than "
     "real matching. Review more images, or improve the model."),
    (("count_error_dispersion_too_high",),
     "Not yet. Individual held-back images can be far off in opposite directions that cancel out "
     "in the total, the count isn't reliable image-to-image, whatever the total shows. Review "
     "more images, or improve the model."),
    (("count_bias_exceeds_tolerance",),
     "Not yet. On the held-back images, the model's counts didn't agree closely enough with your "
     "review to trust them yet. Reviewing more images, or improving the model, can help."),
    (("count_bias_exceeds_tolerance_per_class",),
     "Not yet. The split between kinds of object doesn't agree with your review, the model is "
     "finding too many of one kind and too few of another, in a way that can cancel out in the "
     "total even when the total itself agrees. Any result that separates the kinds (a percentage "
     "of one kind, for instance) would be wrong. Correcting the mislabelled kinds in your review, "
     "or improving the model, can help."),
    # Raised by review_to_records before the gate ever runs (a verdict's class identity couldn't be
    # resolved against its producing bucket), not one of resolve_operating_point's own sweep
    # failures, so it never appears in a bundle's "failures" list; still shares this vocabulary
    # rather than an independently-authored string, see review_to_records below.
    (("class_id_unresolvable",),
     "Not yet. At least one reviewed verdict can't be tied to a class this prediction bucket "
     "recognizes: no resolvable class identity was recorded for it, either the bucket never "
     "recorded a name->id map, or this class name isn't one of its keys. Class-aware "
     "review-confirmed validation isn't available for this bucket/trait combination yet."),
]
_KNOWN_FAILURE_NAMES = {name for names, _ in _FAILURE_MESSAGES for name in names}


def _breeder_message(name: str) -> str:
    """The breeder-facing text for one named failure in :data:`_FAILURE_MESSAGES`, the single
    vocabulary every review-based calibration refusal reads from rather than authoring its own
    prose inline."""
    for names, msg in _FAILURE_MESSAGES:
        if name in names:
            return msg
    raise AssertionError(f"{name!r} is not a name in _FAILURE_MESSAGES")


def _to_xywh(box_norm: list, img_w: float, img_h: float) -> list[float]:
    """Normalized center-form ``[cx, cy, w, h]`` -> top-left ``[x, y, w, h]`` scaled by image dims.

    With no image dimensions the unit square (1.0, 1.0) keeps every record on one consistent
    normalized scale, valid for the count sweep, whose tolerance is derived from the same records.
    """
    cx, cy, bw, bh = (float(v) for v in box_norm)
    return [(cx - bw / 2) * img_w, (cy - bh / 2) * img_h, bw * img_w, bh * img_h]


def _same_producer(entry_identity: dict, target: dict) -> bool:
    """True when ``entry_identity`` (a verdict/image's recorded producer fact) and ``target`` (a
    bucket's own identity) name the same producing model, never a directory-string comparison.

    Prefers ``checkpoint_sha256`` (the exact model bytes) when both sides recorded one; falls back
    to ``experiment_id`` only when a side has no sha to compare against. Missing on both sides is
    not a match, there is nothing here to reconcile, so an unresolvable identity fails closed
    rather than being treated as "unknown, so allow it".
    """
    e_sha, t_sha = entry_identity.get("checkpoint_sha256"), target.get("checkpoint_sha256")
    if e_sha is not None and t_sha is not None:
        return e_sha == t_sha
    e_exp, t_exp = entry_identity.get("experiment_id"), target.get("experiment_id")
    return e_exp is not None and t_exp is not None and e_exp == t_exp


def _matches_any_bucket(identity: dict | None, bucket_identities: list[dict]) -> bool:
    """True when ``identity`` (a verdict/image's own recorded producer fact, or ``None``) names the
    same producer as any of ``bucket_identities``. ``None``/empty never matches, a verdict with no
    recorded identity (written before producer-identity scoping existed, or authored by a caller
    that never resolved one) fails closed rather than being grandfathered in (CLAUDE.md's
    no-back-compat rule)."""
    if not identity:
        return False
    return any(_same_producer(identity, target) for target in bucket_identities)


def review_to_records(
    review_state: dict,
    *,
    bucket_identities: list[dict],
    image_dims: dict[str, tuple[int, int]] | None = None,
    only_completed: bool = True,
) -> list[dict]:
    """Reconstruct per-image COCO records (gt=affirmed, dt=model predictions) from review verdicts.

    ``bucket_identities`` (required, no default that would silently skip scoping): the
    producer identity/identities (``checkpoint_sha256``/``experiment_id``) of the prediction
    bucket(s) this reference is being built for. Only verdict entries (and confirmed-negative image
    records) recorded against a matching producer are included:

      - an image with verdict entries: only entries whose ``producer_identity`` matches any of
        ``bucket_identities`` contribute to ``gt``/``dt``. If none of an image's entries match, the
        whole image is dropped, it carries no evidence for this bucket, so it must not silently
        count as a zero-bias/zero-object agreement for it (this closes a real contamination path:
        model A's review verdicts must not validate model B's bucket).
      - an image with zero verdict entries (a confirmed negative via ``mark_complete``) carries its
        producer identity at the image level instead (``img_data["producer_identity"]``), checked
        the same way; a mismatch or missing stamp drops the image entirely rather than counting it
        as a negative for the wrong bucket.

    A verdict/image with no recorded identity at all (written before producer-identity scoping
    existed) always fails closed here, excluded, never grandfathered (CLAUDE.md's no-back-compat
    rule; this platform has no users yet).

    Each returned record also carries ``adjudication_covered``, ``True`` when there is
    positive evidence a human could have caught a missed object on this image:

      - a verdict-bearing image: the image's ``gt_preexisting`` fact is ``True``, or at least one
        of its (scoped) verdict entries carries ``missed_object_attested``, a fact
        ``record_detection_action`` stamps explicitly at the moment a verdict is recorded (from
        whether the caller supplied neither a ``gt_idx`` nor a ``pred_idx``, the exact shape only
        the "mark missed object" tool produces), never reconstructed here from the entry's bbox
        geometry. Geometry alone is ambiguous: a rejected or accepted pre-existing FN (an existing,
        already-indexed GT box being corrected or confirmed, not a newly-attested miss) ends up with
        the identical ``pred_bbox_norm=None, gt_bbox_norm=<box>`` shape once persisted, so inferring
        coverage from that shape would silently count an FN correction as if it were a
        swept-for-a-missed-object attestation.
      - a zero-verdict (``mark_complete``) image: the recorded ``adjudication_covered`` fact the
        route stamped at completion time, ``True`` only for a genuine negative
        (the route confirmed the bucket held zero predictions for this image, so there was nothing
        to individually adjudicate and Complete is itself the confirming act), never for a bulk-
        accept of a populated image the breeder never individually reviewed. A missing/unset fact
        is ``False``, fails closed, matching every other unrecorded-fact rule here.

    ``resolve_operating_point_from_review`` passes this field to ``resolve_operating_point`` as a
    gate, every record must satisfy it or the whole reference is refused, never a per-record
    filter (a filter here is a fail-open: the excluded set correlates with the quantity being
    measured, see ``resolve_operating_point``'s own docstring for the reproduced scenario).

    ``image_dims`` maps image name (with extension, as review state keys it) -> ``(width, height)``
    to denormalize boxes to pixels (the faithful scale); omit it to keep records on the normalized
    unit square. ``only_completed`` restricts to fully-reviewed images (a partially-reviewed image
    is not a confirmed reference).

    Each record carries ``image_id=Path(img_name).stem``, the stem, not the extensioned review-
    state key. Training stems (``split.json``'s ``"train"`` list) never carry an
    extension, so an extensioned ``image_id`` here could never match a training stem in
    ``_train_disjointness``, the leak the disjointness check exists to catch went entirely
    undetected on this path. Stemming also restores tile-group coherence: ``_TILE_GROUP_RE`` only
    matches a bare stem, so an extensioned id degenerated to one group per tile.
    """
    dims = image_dims or {}
    records: list[dict] = []
    for img_name, img_data in review_state.get("image", {}).items():
        if only_completed and img_data.get("img_status") != "completed":
            continue
        img_w, img_h = dims.get(img_name, (1.0, 1.0))
        detections = img_data.get("detections") or []
        gt_preexisting = bool(img_data.get("gt_preexisting"))

        if not detections:
            # A confirmed-negative / zero-verdict image (mark_complete): nothing here to carry a
            # per-entry identity, so it lives at the image level instead. Coverage is
            # the recorded fact the route stamped at completion time, True only for a genuine
            # negative (bucket held zero predictions here), never inferred from gt_preexisting:
            # gt_preexisting is never set on a zero-verdict image, since only
            # record_detection_action writes it, so treating it as the coverage fact would silently
            # drop every confirmed negative from the reference.
            if not _matches_any_bucket(img_data.get("producer_identity"), bucket_identities):
                continue
            records.append({"width": int(img_w), "height": int(img_h),
                            "image_id": Path(img_name).stem, "gt": [], "dt": [],
                            "adjudication_covered": bool(img_data.get("adjudication_covered"))})
            continue

        scoped = [e for e in detections
                 if _matches_any_bucket(e.get("producer_identity"), bucket_identities)]
        if not scoped:
            continue  # nothing on this image pertains to the bucket(s) being validated

        gt: list[dict] = []
        dt: list[dict] = []
        for entry in scoped:
            gt_norm = entry.get("gt_bbox_norm")
            pred_norm = entry.get("pred_bbox_norm")
            if gt_norm is None and pred_norm is None:
                # A coverage-only attestation ("swept this image, found nothing more", the Review
                # tab's "sweep" verdict: neither gt_idx nor pred_idx set and no edited geometry
                # either) carries no class-scoped evidence at all, only its own
                # missed_object_attested stamp (folded into has_missed_object_attestation below).
                # Requiring a resolvable class_id here would refuse the whole reference over an
                # entry that could never contribute a gt/dt box in the first place.
                continue
            action = entry.get("action")
            cid_raw = entry.get("class_id")
            if cid_raw is None:
                # An unresolved class identity (never recorded, or the producing bucket's own
                # id_map didn't recognize this verdict's class_name, e.g. an attribute-scoped
                # bucket, whose id_map is keyed by attribute values, being handed a GT annotation's
                # raw subject name) must refuse the whole reference, not silently drop this one
                # entry or default it to class 0. A partial drop is a fail-open here: dropping a
                # confirmed-miss (FN) entry while keeping an in-vocabulary accepted-FP entry can
                # make gt/dt agree by construction and pass the count-bias gate on a reference that
                # is missing real evidence, the opposite of what class-aware admission is for.
                raise ValueError(
                    f"{_breeder_message('class_id_unresolvable')} "
                    f"(image {img_name!r}, class {entry.get('class_name')!r})"
                )
            cid = int(cid_raw)
            conf = entry.get("conf")
            # dt: the model's own prediction with its recorded score (any verdict that has one).
            if pred_norm and len(pred_norm) == 4 and conf is not None:
                dt.append({"category_id": cid + 1, "bbox": _to_xywh(pred_norm, img_w, img_h),
                           "score": float(conf)})
            # gt: boxes the breeder affirmed exist (accepted FP carries only a predicted box).
            if action in _POSITIVE_ACTIONS:
                box = gt_norm or pred_norm
                if box and len(box) == 4:
                    gt.append({"category_id": cid + 1, "bbox": _to_xywh(box, img_w, img_h),
                               "iscrowd": 0})
        has_missed_object_attestation = any(e.get("missed_object_attested") for e in scoped)
        records.append({"width": int(img_w), "height": int(img_h),
                        "image_id": Path(img_name).stem, "gt": gt, "dt": dt,
                        "adjudication_covered": gt_preexisting or has_missed_object_attestation})
    return records


def review_reference_hash(records: list[dict]) -> str:
    """Content hash of the review-confirmed reference (image names + affirmed gt boxes).

    Scopes the derived conf to *this* reference so the firewall can flag it being inherited across a
    different one, the review analogue of ``resolution.dataset_hash`` over label bytes.
    """
    h = hashlib.sha256()
    for rec in sorted(records, key=lambda r: str(r.get("image_id", ""))):
        h.update(str(rec.get("image_id", "")).encode("utf-8"))
        h.update(b"\0")
        h.update(json.dumps(rec.get("gt", []), sort_keys=True).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def review_conf_threshold(
    review_state: dict, *, bucket_identities: list[dict], only_completed: bool = True,
) -> float | None:
    """The review session's effective confidence-display threshold, from recorded
    verdict facts, the max ``conf_threshold`` across every (bucket-scoped) verdict entry on the
    images this reference includes, never a re-typed default.

    Scoped by ``bucket_identities`` the same way :func:`review_to_records` scopes gt/dt (the same
    predicate, ``_matches_any_bucket``, one implementation, not a second one that could drift), so
    an unrelated review session over a different model's predictions never inflates or deflates the
    floor this reference was actually shown at.

    ``None`` when any image (with at least one bucket-scoped verdict entry) recorded no
    ``conf_threshold`` on any of them, the review-side term is then unknown, which the caller
    combines with the generation-side floor via ``max(...)``;
    either half unknown makes the combined ``staged_conf_floor`` ``None`` (fails closed per
    ``_conf_censored``, the same rule a missing producer identity follows). An image with zero
    bucket-scoped verdict entries (nothing walked/reviewed against this bucket) contributes
    nothing here, neither raising a value nor tripping the unknown case.
    """
    thresholds: list[float] = []
    for img_data in review_state.get("image", {}).values():
        if only_completed and img_data.get("img_status") != "completed":
            continue
        scoped = [e for e in (img_data.get("detections") or [])
                 if _matches_any_bucket(e.get("producer_identity"), bucket_identities)]
        if not scoped:
            continue
        image_thresholds = [e.get("conf_threshold") for e in scoped]
        if any(t is None for t in image_thresholds):
            return None
        thresholds.extend(float(t) for t in image_thresholds)
    return max(thresholds) if thresholds else None


def resolve_operating_point_from_review(
    review_state: dict,
    trait_name: str,
    *,
    bucket_identities: list[dict],
    image_dims: dict[str, tuple[int, int]] | None = None,
    only_completed: bool = True,
    tile_size: int | None = None,
    tile_size_source: str = "default",
    tiled: bool | None = None,
    tiled_source: str = "default",
    cross_tile_nms: float | None = None,
    max_dets: int | None = None,
    group_by: str = "tile_prefix",
    group_key_map: dict[str, str] | None = None,
    seed: int = 0,
    holdout_ratio: float = 0.5,
    experiment_id: str | None = None,
    staged_conf_floor: float | None = None,
) -> ResolvedBundle:
    """Resolve the count operating point from review verdicts (the review-confirmation reference).

    Splits the reviewed images into a locked, group-aware calibration/holdout split
    (``resolve_locked_cal_holdout_split``, keyed by the review reference's own content hash so a
    later call over the same verdicts returns the same split rather than a fresh cut) and hands
    both to ``resolve_operating_point`` with ``validated_reference=VALIDATED_REVIEW_CONFIRMED``, so the
    same disjoint + count-bias + content-overlap + train-disjointness gate decides whether the
    conf is shippable, and the conf-censoring guard still fails a display-floored reference
    closed. Returns a bundle whose conf is stamped ``VALIDATED_REVIEW_CONFIRMED`` only if that gate passes,
    else ``false``. ``seed``/``holdout_ratio`` only govern the first (locking) draw for this
    reference's identity hash, a later call over the same verdicts returns the locked split
    regardless, and any divergence is surfaced on the bundle's conf sweep, not just logged
    (``attach_split_policy_provenance``).

    ``bucket_identities`` (required, no default): threaded straight to
    ``review_to_records``, see there for the scoping/fail-closed semantics. There is no
    legitimate call to this function without a target bucket; a caller who genuinely has none must
    still say so explicitly by passing an empty list (which, per the same fail-closed rule, refuses
    every verdict rather than silently admitting them all).

    ``staged_conf_floor`` is the effective floor the reviewed predictions were staged/shown
    at, ``max(generation_conf, review_conf_threshold)`` per the design, computed by the caller
    (``routes/review.py``, which has both the buckets' ``operating_point.json`` sidecars and
    ``review_conf_threshold``'s recorded-verdict computation) and passed straight through to
    ``resolve_operating_point``. This function does not derive it.

    ``resolve_locked_cal_holdout_split`` raises ``ValueError`` when the lock references a stem no
    longer among the reviewed images, or when its lock file is corrupt, this
    propagates to the caller rather than crashing later on a missing dict lookup.
    """
    from tcip_mcp.pipelines.data.splits import resolve_locked_cal_holdout_split
    from tcip_mcp.pipelines.operating_point import (
        attach_split_policy_provenance, resolve_operating_point,
    )

    records = review_to_records(review_state, image_dims=image_dims, only_completed=only_completed,
                                bucket_identities=bucket_identities)
    ref_hash = review_reference_hash(records)
    by_id = {str(r.get("image_id", "")): r for r in records}
    stems = sorted(by_id)
    annotation_counts = {s: len(by_id[s].get("gt", [])) for s in stems}
    locked = resolve_locked_cal_holdout_split(
        stems, identity_hash=ref_hash, annotation_counts=annotation_counts,
        group_by=group_by, group_key_map=group_key_map, seed=seed, holdout_ratio=holdout_ratio,
    )
    cal_records = [by_id[s] for s in locked["calibration"] if s in by_id]
    hold_records = [by_id[s] for s in locked["holdout"] if s in by_id]
    bundle = resolve_operating_point(
        trait_name, dataset_hash=ref_hash,
        calibration_records=cal_records or None,
        holdout_records=hold_records or None,
        tile_size=tile_size, tile_size_source=tile_size_source,
        tiled=tiled, tiled_source=tiled_source, cross_tile_nms=cross_tile_nms, max_dets=max_dets,
        validated_reference=VALIDATED_REVIEW_CONFIRMED,
        experiment_id=experiment_id, staged_conf_floor=staged_conf_floor,
        adjudication_covered=lambda r: bool(r.get("adjudication_covered")),
    )
    attach_split_policy_provenance(bundle, locked)
    return bundle


def describe_review_validation(bundle: ResolvedBundle, *, reviewed_image_count: int) -> dict[str, Any]:
    """Translate a review-confirmed operating-point bundle into a breeder-legible validation result.

    Reads the conf param's own sweep diagnostics (the same gate output ``resolve_operating_point``
    already produced, never a re-run) and maps them to plain language a non-CV breeder can act on.
    ``resolve_operating_point``'s named ``failures`` list (cross-cutting) is the single source of
    truth for which check(s) refused; this function's job is only to translate each name to a
    message, exhaustively, so an unrecognized failure name is a loud error here, not a silent
    fallthrough to the generic "counts didn't agree" message. When more than one named failure
    applies at once (e.g. too few images and a censored floor), every one of them gets its own
    message rather than only the highest-priority match, a breeder who fixes the first blocker and
    resubmits must not discover the second only then. Pure over the bundle, no torch, no
    re-derivation.

    The "Validated" message's miss-coverage claim is read directly off the exact-conf holdout
    curve entry (``sweep['holdout_bias']``, already carrying ``tp``/``fn``/``recall``
    at precisely the shipped conf), never a second, independently-computed miss statistic that could
    drift from what the gate actually decided.
    """
    conf = bundle.params.get("conf")
    validated = bool(conf is not None and conf.is_shippable)
    reference = conf.validated_against if conf is not None else None
    # Report the derived number without shipping it, the honest raw-read accessor, not .value.
    conf_value = (float(conf.unvalidated_value(acknowledge_unvalidated=True))
                  if conf is not None else None)
    sweep = (conf.sweep if conf is not None else None) or {}
    failures = sweep.get("failures") or []
    # An elif chain that returns on the first recognized name would let an unmapped name riding
    # alongside a recognized one fall through silently instead of raising. Check exhaustiveness
    # unconditionally, over the whole list, before any branch runs.
    unrecognized = set(failures) - _KNOWN_FAILURE_NAMES
    if unrecognized:
        raise AssertionError(
            f"resolve_operating_point reported unrecognized gate failure(s) {sorted(unrecognized)} "
            f"(full list: {failures!r}), describe_review_validation has no breeder-facing message "
            "for one of these yet.")
    if validated:
        hb = sweep.get("holdout_bias") or {}
        tp, fn = hb.get("tp"), hb.get("fn")
        miss_note = ""
        if tp is not None and fn is not None and (tp + fn) > 0:
            miss_note = (f" On the held-back images, it found {tp} of {tp + fn} objects you "
                        f"confirmed ({100 * tp / (tp + fn):.0f}% recall).")
        reason = (f"Validated. Your review of {reviewed_image_count} reviewed image(s) confirms this "
                  f"model's counts closely enough to use as a validation reference for results.{miss_note}")
    elif "passed_holdout" not in sweep:
        # This branch must come before the _FAILURE_MESSAGES lookup. conf_censored is also present,
        # often truthy, in the no-holdout branch's sweep_data, which has no "failures" list at all,
        # checking those raw keys here would misdirect a "too few images reviewed" session into
        # "re-run at a low confidence" every time.
        reason = ("Not yet. Too few images have been reviewed, the check needs at least two fully "
                  "reviewed images so it can hold some back to test against. Review a few more, then "
                  "try again.")
    else:
        # Every applicable failure gets its own message, in _FAILURE_MESSAGES' order, not just the
        # first match. A breeder who hits two blockers at once (e.g. too few images and a censored
        # floor) must see both, not fix the first and only then discover the second.
        matched = [msg for names, msg in _FAILURE_MESSAGES if any(n in failures for n in names)]
        if not matched:
            raise AssertionError(
                f"resolve_operating_point set an unvalidated result with a completed holdout gate "
                f"but no recognized failure name (failures={failures!r}), "
                "describe_review_validation cannot explain this refusal.")
        reason = "\n\n".join(matched)
    return {"validated": validated, "reference": reference, "conf": conf_value, "reason": reason}
