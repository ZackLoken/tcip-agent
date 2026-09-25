"""Reconstruct a calibration reference from human review verdicts.

Turns per-image review verdicts into the COCO record shape ``resolve_operating_point`` consumes, so
a breeder-confirmed sample of the model's own outputs can validate the count operating point. Per
record:

  - ``gt`` = the boxes the breeder affirmed exist: accepted/edited matches, confirmed misses (FN),
    and false-positives the breeder promoted to real (accepted FP). Rejected boxes never enter gt.
  - ``dt`` = the model's own predictions carried with their recorded confidence, regardless of
    verdict, so the curve derivation re-derives TP/FP/FN by center-matching dt against the affirmed
    gt exactly as the GT path does.

Every verdict (and confirmed-negative image record) is scoped to the producing bucket(s) it was
recorded against before any gate statistic sees it, and every record carries whether the image was
adjudicated for missed objects.

Torch-free.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tcip_annotation.json_io import xywh
from tcip_annotation.state import BBox
from tcip_annotation.verdicts import Verdict, decode_verdict

from tcip_mcp.pipelines.data.splits import DEFAULT_CAL_SEED, DEFAULT_GROUP_BY, DEFAULT_HOLDOUT_RATIO
from tcip_mcp.pipelines.resolution import VALIDATED_REVIEW_CONFIRMED, ResolvedBundle

# Every name resolve_operating_point can put in gate_evidence["failures"] (cross-cutting named-failure
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
    (("conf_floor_unstated",),
     "Not yet. Nothing states the confidence floor these predictions were actually generated or "
     "filtered at, so the check has nothing to reconcile the picked confidence against, whether "
     "that is a generation cutoff or review filter this route could not read, or a bespoke model "
     "whose module exposes no operating-point knob under a recognized name: the platform looks "
     "for score_thresh, nms_thresh or detections_per_img on the module itself, its "
     "detector.roi_heads, or its detector. Review a bucket whose staging floor is known."),
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
    (("selection_disjointness_unresolvable",),
     "Not yet. This model's record does not establish which images chose it, so the platform "
     "can't confirm the reviewed images were held back from that choice. Retrain with the "
     "current data (which records this), or use a model whose training record is known."),
    (("selection_disjointness_leaked",),
     "Not yet. The images used to choose this model's best weights cannot check it. Review "
     "images the model neither trained on nor was chosen against."),
    (("content_shared_with_calibration",),
     "Not yet. One or more of the held-back images you reviewed share content with the calibration "
     "images, so they can't function as an independent check. Review a genuinely distinct set of "
     "images, then try again."),
    (("localization_quality_floor_failed",),
     "Not yet. On the held-back images, the model's predictions don't clear the held-out "
     "match-quality bar authored for this trait, even where the counts agree, that agreement can "
     "be coincidental rather than real matching. Review more images, or improve the model."),
    (("holdout_match_quality_floor_unauthored",),
     "Not yet. No held-out match-quality floor has been authored for this trait yet, so this check "
     "has nothing to hold the model's predictions to. This is for the agent: author one with "
     "author_trait_spec's holdout_match_quality_floor, with the breeder's confirmation."),
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
     "of one kind, for instance) would be wrong. Correcting the mislabeled kinds in your review, "
     "or improving the model, can help."),
    # Raised by review_to_records before the gate ever runs (a verdict's class identity couldn't be
    # resolved against its producing bucket), not one of resolve_operating_point's own gate-evidence
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
    """The breeder-facing text for one named failure in :data:`_FAILURE_MESSAGES`."""
    for names, msg in _FAILURE_MESSAGES:
        if name in names:
            return msg
    raise AssertionError(f"{name!r} is not a name in _FAILURE_MESSAGES")


def _selection_movement_sentence(gate_evidence: dict) -> str:
    """One sentence naming a calibration label that moved since the split was drawn, read from
    the sealed ``selection_disjointness``'s own ``calibration_labels_moved``; empty when that
    list is empty or absent (a run with no ``label_digests`` block, or nothing moved).

    Never a ``_FAILURE_MESSAGES`` entry: a moved label is disclosed, not a floor. This surface
    names no manifest and holds no labels directory of its own, so ``selection_redrawn`` and
    ``labels_moved_run_to_now`` are always ``null`` here and earn no sentence of their own.
    """
    moved = (gate_evidence.get("selection_disjointness") or {}).get("calibration_labels_moved")
    if not moved:
        return ""
    names = ", ".join(sorted(moved))
    return (f" The label(s) for {names} changed since this split was drawn, so this number was "
            "calibrated on a universe that moved: redraw the split or re-confirm those labels.")


_UNIT_SQUARE = (1.0, 1.0)
"""The frame a verdict's box is recorded on when its image's dimensions are unknown."""


def _to_xywh(box_norm: Sequence[float], dims: tuple[int, int] | None) -> list[float]:
    """A verdict's normalized box scaled to its image (:meth:`BBox.from_normalized_center`) as
    the stored ``[x, y, w, h]`` on the stored pixel grid (``json_io.xywh``).

    With no image dimensions the record stays on the unit square, off any pixel grid, keeping
    every record on one consistent normalized scale, valid for the count curve, whose tolerance
    is derived from the same records.
    """
    b = BBox.from_normalized_center(box_norm, *(dims or _UNIT_SQUARE))
    return xywh(b.x1, b.y1, b.x2, b.y2, on_grid=dims is not None)


def _same_producer(entry_identity: dict, target: dict) -> bool:
    """True when ``entry_identity`` (a verdict/image's recorded producer fact) and ``target`` (a
    bucket's own identity) name the same producing model, never a directory-string comparison.

    Prefers ``checkpoint_sha256`` (the exact model bytes) when both sides recorded one; falls back
    to ``experiment_id`` only when a side has no sha to compare against. Missing on both sides is
    not a match.
    """
    e_sha, t_sha = entry_identity.get("checkpoint_sha256"), target.get("checkpoint_sha256")
    if e_sha is not None and t_sha is not None:
        return e_sha == t_sha
    e_exp, t_exp = entry_identity.get("experiment_id"), target.get("experiment_id")
    return e_exp is not None and t_exp is not None and e_exp == t_exp


def _matches_any_bucket(identity: dict | None, bucket_identities: list[dict]) -> bool:
    """True when ``identity`` (a verdict/image's own recorded producer fact, or ``None``) names the
    same producer as any of ``bucket_identities``. ``None``/empty never matches: a verdict with no
    recorded identity fails closed."""
    if not identity:
        return False
    return any(_same_producer(identity, target) for target in bucket_identities)


def _scoped_verdicts(entries: list[dict], bucket_identities: list[dict]) -> list[Verdict]:
    """An image's verdict entries, decoded, recorded against one of ``bucket_identities``."""
    verdicts = [decode_verdict(e) for e in entries]
    return [v for v in verdicts if _matches_any_bucket(v.producer_identity, bucket_identities)]


def covers(slot: Any, subject: str | None) -> bool:
    """True when a zero-verdict image's own recorded ``adjudication_covered`` map confirms coverage
    for ``subject``, ``None`` read as ``"*"``: the subject's own entry when the map has one,
    whatever it says, never overridden by a subject-less Complete's ``"*"`` entry; the "*" entry
    answers only when the subject has no entry of its own.

    ``slot`` is that map, or ``None`` for an image with no such record. A bare boolean is refused
    by name.
    """
    if slot is None:
        return False
    if not isinstance(slot, dict):
        raise ValueError(
            f"adjudication_covered is {slot!r}, not the subject-keyed map a coverage check "
            "expects; a bare boolean here would guess which subject it was confirming."
        )
    key = subject if subject is not None else "*"
    if key in slot:
        return bool(slot[key])
    return bool(slot.get("*"))


def review_to_records(
    review_state: dict,
    *,
    bucket_identities: list[dict],
    image_dims: dict[str, tuple[int, int]] | None = None,
    only_completed: bool = True,
    subject: str | None = None,
) -> list[dict]:
    """Reconstruct per-image COCO records (gt=affirmed, dt=model predictions) from review verdicts.

    ``bucket_identities`` (required): the producer identity/identities
        (``checkpoint_sha256``/``experiment_id``) of the prediction bucket(s) this reference is
        being built for. Only verdict entries (and confirmed-negative image records) recorded
        against a matching producer are included:

      - an image with verdict entries: only entries whose ``producer_identity`` matches any of
        ``bucket_identities`` contribute to ``gt``/``dt``; if none match, the whole image is
        dropped.
      - an image with zero verdict entries (a confirmed negative via ``mark_complete``) carries its
        producer identity at the image level (``img_data["producer_identity"]``), checked the same
        way; a mismatch or missing stamp drops the image.

    A verdict/image with no recorded identity at all is excluded.

    Each returned record also carries ``adjudication_covered``, ``True`` when there is positive
    evidence a human could have caught a missed object on this image:

      - a verdict-bearing image: the image's ``gt_preexisting`` fact is ``True``, or at least one
        of its (scoped) verdict entries carries ``missed_object_attested``, the fact
        ``record_detection_action`` stamps when a verdict is recorded. Both facts are recorded at
        the image level, not scoped to a subject, so evidence about one subject admits a reference
        built for another subject on the same image; scoping them to the validated subject is
        unimplemented.
      - a zero-verdict (``mark_complete``) image: the recorded ``adjudication_covered`` map, read
        through :func:`covers` for ``subject`` (``"*"`` when ``subject`` is ``None``). A missing
        map, or a map with neither key set, is ``False``.

    ``subject`` (default ``None``, read as ``"*"``) is the object identity this reference is being
    built to validate, threaded to :func:`covers`.

    ``image_dims`` maps image name (with extension, as review state keys it) -> ``(width, height)``
    to denormalize boxes to pixels; omit it to keep records on the normalized unit square.
    ``only_completed`` restricts to fully-reviewed images.

    Each record carries ``image_id=Path(img_name).stem``, the stem, not the extensioned
    review-state key, so it matches a training stem in ``_train_disjointness`` and
    ``_TILE_GROUP_RE``'s bare-stem tile groups.
    """
    from tcip_mcp.pipelines.training.evaluation import build_coco_image_record, dt_record, gt_record

    dims = image_dims or {}
    records: list[dict] = []
    for img_name, img_data in review_state.get("image", {}).items():
        if only_completed and img_data.get("img_status") != "completed":
            continue
        img_dims = dims.get(img_name)
        img_w, img_h = img_dims or _UNIT_SQUARE
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
            records.append({
                **build_coco_image_record(img_w, img_h, [], [], image_id=Path(img_name).stem),
                "adjudication_covered": covers(img_data.get("adjudication_covered"), subject)})
            continue

        scoped = _scoped_verdicts(detections, bucket_identities)
        if not scoped:
            continue  # nothing on this image pertains to the bucket(s) being validated

        gt: list[dict] = []
        dt: list[dict] = []
        for verdict in scoped:
            if not verdict.geometry_recorded:
                # A coverage-only attestation ("swept this image, found nothing more", the Review
                # tab's "sweep" verdict: neither gt_idx nor pred_idx set and no edited geometry
                # either) carries no class-scoped evidence at all, only its own
                # missed_object_attested stamp (folded into has_missed_object_attestation below).
                # Requiring a resolvable class_id here would refuse the whole reference over an
                # entry that could never contribute a gt/dt box in the first place.
                continue
            if verdict.class_id is None:
                # An unresolved class identity (never recorded, or foreign to the producing bucket's
                # own id_map, e.g. a bare hand-split directory) must refuse the whole reference, not silently drop this one
                # entry or default it to class 0. A partial drop is a fail-open here: dropping a
                # confirmed-miss (FN) entry while keeping an in-vocabulary accepted-FP entry can
                # make gt/dt agree by construction and pass the count-bias gate on a reference that
                # is missing real evidence, the opposite of what class-aware admission is for.
                raise ValueError(
                    f"{_breeder_message('class_id_unresolvable')} "
                    f"(image {img_name!r}, class {verdict.class_name!r})"
                )
            cid = verdict.class_id
            # dt: the model's own prediction with its recorded score (any verdict that has one).
            if verdict.pred_box is not None and verdict.conf is not None:
                dt.append(dt_record(_to_xywh(verdict.pred_box, img_dims), cid + 1,
                                    verdict.conf))
            # gt: boxes the breeder affirmed exist (accepted FP carries only a predicted box).
            if verdict.is_positive and verdict.affirmed_box is not None:
                gt.append(gt_record(_to_xywh(verdict.affirmed_box, img_dims), cid + 1,
                                    verdict.iscrowd))
        has_missed_object_attestation = any(v.missed_object_attested for v in scoped)
        records.append({
            **build_coco_image_record(img_w, img_h, gt, dt, image_id=Path(img_name).stem),
            "adjudication_covered": gt_preexisting or has_missed_object_attestation})
    return records


def review_reference_hash(records: list[dict]) -> str:
    """Content hash of the review-confirmed reference: each image's name and its ground truth as
    content (:func:`~tcip_mcp.pipelines.training.evaluation.gt_facts`), and nothing derived from
    them, so a field a record builder adds never re-keys a reference.

    Scopes the derived conf to *this* reference so the firewall can flag it being inherited across a
    different one, the review analog of ``resolution.dataset_hash`` over label bytes.
    """
    from tcip_mcp.pipelines.training.evaluation import gt_facts

    h = hashlib.sha256()
    for rec in sorted(records, key=lambda r: str(r["image_id"])):
        h.update(str(rec["image_id"]).encode("utf-8"))
        h.update(b"\0")
        h.update(json.dumps(gt_facts(rec)).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def review_conf_threshold(
    review_state: dict, *, bucket_identities: list[dict], only_completed: bool = True,
) -> float | None:
    """The review session's effective confidence-display threshold, from recorded verdict facts:
    the max ``conf_threshold`` across every (bucket-scoped) verdict entry on the images this
    reference includes.

    Scoped by ``bucket_identities`` the same way :func:`review_to_records` scopes gt/dt
    (``_matches_any_bucket``).

    ``None`` when any image (with at least one bucket-scoped verdict entry) recorded no
    ``conf_threshold`` on any of them. An image with zero bucket-scoped verdict entries contributes
    nothing here.
    """
    thresholds: list[float] = []
    for img_data in review_state.get("image", {}).values():
        if only_completed and img_data.get("img_status") != "completed":
            continue
        for verdict in _scoped_verdicts(img_data.get("detections") or [], bucket_identities):
            if verdict.conf_threshold is None:
                return None
            thresholds.append(verdict.conf_threshold)
    return max(thresholds) if thresholds else None


def resolve_operating_point_from_review(
    review_state: dict,
    trait_name: str,
    *,
    scope_root: str | Path,
    bucket_identities: list[dict],
    image_dims: dict[str, tuple[int, int]] | None = None,
    only_completed: bool = True,
    tile_size: int | None = None,
    tile_size_source: str = "default",
    tile_size_derived_from: str | None = None,
    tiled: bool | None = None,
    tiled_source: str = "default",
    cross_tile_nms: float | None = None,
    max_dets: int | None = None,
    group_by: str = DEFAULT_GROUP_BY,
    group_key_map: dict[str, str] | None = None,
    seed: int = DEFAULT_CAL_SEED,
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
    experiment_id: str | None = None,
    staged_conf_floor: float | None = None,
    experiment_id_ambiguous: bool = False,
    subject: str | None = None,
    calibration_labels_dir: str | None = None,
) -> ResolvedBundle:
    """Resolve the count operating point from review verdicts (the review-confirmation reference).

    Splits the reviewed images into a locked, group-aware calibration/holdout split
    (``resolve_locked_cal_holdout_split``, keyed by the review reference's own content hash) and
    hands both to ``resolve_operating_point`` with
    ``validated_reference=VALIDATED_REVIEW_CONFIRMED``. Returns a bundle whose conf is stamped
    ``VALIDATED_REVIEW_CONFIRMED`` only if that gate passes, else ``false``.
    ``seed``/``holdout_ratio`` only govern the first (locking) draw for this reference's identity
    hash; any divergence is surfaced on the bundle's conf gate evidence
    (``attach_split_policy_provenance``).

    ``scope_root`` (required): the root the locked split is stored under, the dataset root the
        reviewed records themselves live under. See ``pipelines.data.splits.cal_holdout_lock_key``.

    ``bucket_identities`` (required): threaded to ``review_to_records``; an empty list refuses
        every verdict.

    ``staged_conf_floor`` is the effective floor the reviewed predictions were staged/shown at,
    ``max(generation_conf, review_conf_threshold)``, computed by the caller and passed through to
    ``resolve_operating_point``.

    ``tile_size_derived_from`` is the caller's own fact, read off the sidecar's stamp and forwarded
    unchanged.

    ``resolve_locked_cal_holdout_split``'s ``ValueError`` (a lock referencing a stem no longer
    among the reviewed images, or a corrupt lock file) propagates. This function takes no
    selection.

    ``experiment_id_ambiguous`` is true when the caller's buckets named more than one producing run
    rather than none; carried onto the sealed train-disjointness fact so
    :func:`describe_review_validation` can tell the two apart.

    ``subject`` (default ``None``) is forwarded to :func:`review_to_records`.

    ``calibration_labels_dir`` (the directory the reviewed bucket's own labels live in, when the
    caller can name one) is forwarded to ``resolve_operating_point``'s selection-disjointness
    check, applicable only when the checkpoint named by ``experiment_id`` carries a
    ``selection_binding``.
    """
    from tcip_mcp.pipelines.data.splits import resolve_locked_cal_holdout_split
    from tcip_mcp.pipelines.operating_point import (
        attach_split_policy_provenance, resolve_operating_point,
    )
    from tcip_mcp.pipelines.training.evaluation import gt_objects

    records = review_to_records(review_state, image_dims=image_dims, only_completed=only_completed,
                                bucket_identities=bucket_identities, subject=subject)
    ref_hash = review_reference_hash(records)
    by_id = {str(r.get("image_id", "")): r for r in records}
    stems = sorted(by_id)
    annotation_counts = {s: len(gt_objects(by_id[s])) for s in stems}
    locked = resolve_locked_cal_holdout_split(
        stems, identity_hash=ref_hash, scope_root=scope_root, annotation_counts=annotation_counts,
        group_by=group_by, group_key_map=group_key_map, seed=seed, holdout_ratio=holdout_ratio,
    )
    cal_records = [by_id[s] for s in locked["calibration"] if s in by_id]
    hold_records = [by_id[s] for s in locked["holdout"] if s in by_id]
    bundle = resolve_operating_point(
        trait_name, dataset_hash=ref_hash,
        calibration_records=cal_records or None,
        holdout_records=hold_records or None,
        tile_size=tile_size, tile_size_source=tile_size_source,
        tile_size_derived_from=tile_size_derived_from,
        tiled=tiled, tiled_source=tiled_source, cross_tile_nms=cross_tile_nms, max_dets=max_dets,
        validated_reference=VALIDATED_REVIEW_CONFIRMED,
        experiment_id=experiment_id, staged_conf_floor=staged_conf_floor,
        adjudication_covered=lambda r: bool(r.get("adjudication_covered")),
        calibration_labels_dir=calibration_labels_dir,
    )
    attach_split_policy_provenance(bundle, locked)
    conf = bundle.params.get("conf")
    if conf is not None and isinstance(conf.gate_evidence, dict):
        td = conf.gate_evidence.get("train_disjointness")
        if isinstance(td, dict):
            td["experiment_id_ambiguous"] = experiment_id_ambiguous
    return bundle


def describe_review_validation(bundle: ResolvedBundle, *, reviewed_image_count: int) -> dict[str, Any]:
    """Translate a review-confirmed operating-point bundle into a breeder-legible validation
    result.

    Reads the conf param's own gate evidence and maps each of ``resolve_operating_point``'s named
    ``failures`` to plain language a non-CV breeder can act on: every named failure gets its own
    message; an unknown name raises. Pure over the bundle, no torch.

    The "Validated" message's miss-coverage claim is read off ``gate_evidence['holdout_bias']``
    (``tp``/``fn``/``recall`` at the shipped conf).
    """
    conf = bundle.params.get("conf")
    validated = bool(conf is not None and conf.is_shippable)
    reference = conf.validated_against if conf is not None else None
    # Report the derived number without shipping it, the honest raw-read accessor, not .value.
    conf_value = (float(conf.unvalidated_value(acknowledge_unvalidated=True))
                  if conf is not None else None)
    gate_evidence = (conf.gate_evidence if conf is not None else None) or {}
    failures = gate_evidence.get("failures") or []
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
        hb = gate_evidence.get("holdout_bias") or {}
        tp, fn = hb.get("tp"), hb.get("fn")
        miss_note = ""
        if tp is not None and fn is not None and (tp + fn) > 0:
            miss_note = (f" On the held-back images, it found {tp} of {tp + fn} objects you "
                        f"confirmed ({100 * tp / (tp + fn):.0f}% recall).")
        td = gate_evidence.get("train_disjointness") or {}
        run_note = ""
        if not td.get("checked"):
            if td.get("experiment_id_ambiguous"):
                run_note = (" The reviewed predictions were produced by more than one run, so "
                            "there is no single training split to check the images against.")
            else:
                run_note = (" The bucket names no producing run, so the reviewed images were not "
                            "checked against that run's training split.")
        reason = (f"Validated. Your review of {reviewed_image_count} reviewed image(s) confirms this "
                  f"model's counts closely enough to use as a validation reference for "
                  f"results.{miss_note}{run_note}{_selection_movement_sentence(gate_evidence)}")
    elif "passed_holdout" not in gate_evidence:
        # This branch must come before the _FAILURE_MESSAGES lookup. conf_censored is also present,
        # often truthy, in the no-holdout branch's gate evidence, which has no "failures" list at all,
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
        reason = "\n\n".join(matched) + _selection_movement_sentence(gate_evidence)
    return {"validated": validated, "reference": reference, "conf": conf_value, "reason": reason}
