"""Inference MCP tools — run models on images, export results."""

from __future__ import annotations

import logging
from pathlib import Path

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited
from tcip_mcp.pipelines.postprocessing.export import export_detection_csv, write_predictions_json
from tcip_mcp.pipelines.resolution import (
    DEFAULT_CONF,
    DEFAULT_MAX_DETS,
    DEFAULT_NMS_IOU,
    DEFAULT_TILE_SIZE,
    DEFAULT_TILED,
)

logger = logging.getLogger(__name__)


def _calibrate_operating_point(predictor, trait, labels_dir, images_dir, *,
                               tile, tile_size, overlap, tile_batch_size,
                               global_nms_iou, postprocess, cross_tile_nms, max_dets,
                               tile_size_source="default", tiled_source="default",
                               group_by="tile_prefix", group_key_map=None, experiment_id=None,
                               seed=0, holdout_ratio=0.5):
    """Resolve a per-dataset operating point from a labeled split (CV0).

    Returns ``(bundle, hash, n_excluded_incomplete_attribute)`` — the third value is the count of
    cal/holdout stems dropped whole because an instance was unlabeled for ``attribute`` (see
    ``_records`` below); returned to the caller rather than silently filtered, matching
    ``evaluation.py``'s ``n_excluded_incomplete_attribute`` for the same exclusion. It is a separate
    return value, NOT a field on the bundle: ``run_inference`` surfaces it on its own response dict,
    and it does not travel into the persisted ``operating_point.json`` sidecar that
    ``export_predictions`` writes (round-4 review — an earlier version of this line said "disclosed
    on the returned bundle", which would have implied a persistence this has never had).

    The count-unbiased center-match sweep + held-out bias check run the SAME predictor path the
    delivery will use (same tile/tile_size/overlap/nms/postprocess) over a disjoint, LOCKED
    cal/holdout split of the labeled dir (K1 — ``resolve_locked_cal_holdout_split``: group-coherent,
    seeded, and stable across calls, not a fresh lexicographic cut every time), at a floor conf so
    hesitant detections survive to be swept — so the resolved conf is validated in the regime it
    ships through, not an untiled full-frame model pass. ``seed``/``holdout_ratio`` only take effect
    on the FIRST (locking) draw for this labeled dir's identity hash.

    Raises ``ValueError`` (propagated from ``resolve_locked_cal_holdout_split``) when the lock
    references a stem whose image/label no longer exists, or its lock file is corrupt (K1 finding
    4) — the caller (``run_inference``) turns this into a clean ``{"error": ...}`` rather than
    letting a bare ``KeyError`` surface from a stale ``stem_to_image`` lookup.

    ``tile_size_source``/``tiled_source`` (K10 finding 3) are the caller's already-resolved
    provenance for ``tile_size``/``tile`` — forwarded into ``resolve_operating_point`` so the
    calibrated bundle doesn't stamp a fabricated tile_size as ``"derived"``, and so it records
    whether calibration itself actually tiled rather than always asserting ``tiled=True``.
    """
    from tcip_mcp.pipelines.data.datasets import _json_det_targets, _resolve_registry_id_map
    from tcip_mcp.pipelines.data.splits import (
        count_label_lines, label_image_stems, resolve_locked_cal_holdout_split,
    )
    from tcip_mcp.pipelines.operating_point import (
        attach_split_policy_provenance, resolve_operating_point, set_detector_operating_point,
    )
    from tcip_mcp.pipelines.resolution import dataset_hash
    from tcip_mcp.pipelines.training.evaluation import build_coco_image_record

    labels_p = Path(labels_dir)
    dh = dataset_hash(labels_dir)
    # The run's subject + single id map (from predictor.config): calibration GT reads through the
    # same loader-side reader the training targets use, so the swept count can't diverge from training.
    _data_cfg = (getattr(predictor, "config", {}) or {}).get("data") or {}
    _subject, _attribute = _data_cfg.get("subject"), _data_cfg.get("attribute")
    _cal_id_map = None
    if _subject:
        try:
            _reg, _cal_id_map = _resolve_registry_id_map(labels_dir, _subject, _attribute)
        except Exception:  # noqa: BLE001 — fall back to a single-class GT read below
            _cal_id_map = None
    # Labels-intersect-images-on-disk (K1 finding 4): the shared scan force_redraw_cal_holdout_split
    # now also uses, so the two paths can't disagree about which stems exist. A stem whose image was
    # deleted/renamed never even enters the split universe here.
    stems, stem_to_image = label_image_stems(labels_dir, images_dir)
    annotation_counts = {s: count_label_lines(labels_dir, s) for s in stems}
    locked = resolve_locked_cal_holdout_split(
        stems, identity_hash=dh, annotation_counts=annotation_counts,
        group_by=group_by, group_key_map=group_key_map, seed=seed, holdout_ratio=holdout_ratio,
    )
    if locked.get("unlocked_stems"):
        logger.info(
            "cal/holdout split for %s has %d stem(s) not covered by the existing lock (new since "
            "it was drawn); excluded from this calibration: %s", dh,
            len(locked["unlocked_stems"]), locked["unlocked_stems"][:10],
        )
    cal_stems, hold_stems = locked["calibration"], locked["holdout"]

    # Floor the in-model + predictor conf so hesitant detections survive to be swept. The applied
    # value (not a re-typed 0.01 literal) is threaded into resolve_operating_point as the reference's
    # staged_conf_floor (Fix D) — the SAME value this call actually applied, per CLAUDE.md's "when
    # two paths must agree, call one from the other."
    applied = set_detector_operating_point(predictor.model, score_thresh=0.01)
    predictor.score_threshold = applied.get("score_thresh", 0.01)

    n_excluded_incomplete_attribute = 0

    def _records(sub_stems):
        nonlocal n_excluded_incomplete_attribute
        if not sub_stems:
            return []
        results = predictor.predict_batch(
            [str(stem_to_image[s]) for s in sub_stems], tile=tile, tile_size=tile_size,
            overlap=overlap, tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou,
            postprocess=postprocess,
        )
        recs = []
        for s, r in zip(sub_stems, results):
            dt = [{"category_id": int(lab), "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(sc)}
                  for (x1, y1, x2, y2), sc, lab in zip(r["boxes"], r["scores"], r["labels"])]
            # GT lifted to the predictor's 1-indexed labels via the loader-side reader (subject +
            # id map); with no run subject in scope, fall back to a single-class read of every box.
            gt_path = str(labels_p / f"{s}.json")
            if _subject and _cal_id_map is not None:
                gboxes, glabels, n_unlabeled = _json_det_targets(gt_path, _subject, _attribute, _cal_id_map)
                # Stage-6 review N2: an image with any instance unlabeled for `attribute` has
                # incomplete GT for this scope — excluded from the calibration/holdout record set
                # entirely (whole-image, the missing-label-file precedent), never scored against
                # its labeled subset alone. Counted and disclosed on the returned bundle (matching
                # evaluation.py's n_excluded_incomplete_attribute), not a silent filter.
                if n_unlabeled:
                    n_excluded_incomplete_attribute += 1
                    continue
                gt = [{"category_id": int(lab), "bbox": [x1, y1, x2 - x1, y2 - y1], "iscrowd": 0}
                      for (x1, y1, x2, y2), lab in zip(gboxes, glabels)]
            else:
                from tcip_annotation import json_io
                from tcip_annotation.state import bbox_of
                gt = []
                for a in json_io.read_annotations(gt_path):
                    if a.geometry is None:
                        continue
                    bx = bbox_of(a.geometry)
                    gt.append({"category_id": 1,
                               "bbox": [bx.x1, bx.y1, bx.x2 - bx.x1, bx.y2 - bx.y1], "iscrowd": 0})
            recs.append(build_coco_image_record(int(r["width"]), int(r["height"]), gt, dt, image_id=s))
        return recs

    cal_records = _records(cal_stems)
    hold_records = _records(hold_stems)
    bundle = resolve_operating_point(
        trait, dataset_hash=dh, calibration_records=cal_records,
        holdout_records=hold_records or None, tile_size=tile_size,
        tile_size_source=tile_size_source, tiled=tile, tiled_source=tiled_source,
        cross_tile_nms=cross_tile_nms, max_dets=max_dets, experiment_id=experiment_id,
        staged_conf_floor=applied.get("score_thresh"),
    )
    attach_split_policy_provenance(bundle, locked)
    return bundle, dh, n_excluded_incomplete_attribute


def _sweep_summary(conf_param) -> dict:
    """Compact, response-safe view of a calibration sweep (the full curve is written to disk).

    Includes ``disjoint``/``content_overlap_frac``/``train_disjointness`` (K1 finding 3) so the
    calling agent sees the real reason a calibration refused validation, not only the pass/fail
    booleans — a refusal from train-provenance or content-overlap looked identical to a plain
    holdout-bias failure before this. Also includes ``split_policy_divergence``/
    ``split_unlocked_stems`` (K1 finding 5, via ``attach_split_policy_provenance``) so a caller
    whose declared ``seed``/``holdout_ratio``/``group_by`` didn't take effect against an existing
    lock sees that here, in the tool's own response, rather than only in the persisted sweep
    artifact or a server log line.

    ``failures`` (K2, stage-6 review) is the SAME named-failure list ``describe_review_validation``
    reads for the breeder-facing message — surfaced here too, plus the individual new gate fields
    (``conf_floor_mismatch``, dispersion/localization terms), so an agent hitting one of K2's new
    refusal conditions on the GT path sees a real reason instead of every other field reading
    "fine" (the same defect K1 finding 3 fixed for the train-provenance/content-overlap terms,
    reintroduced one level up by K2's new conjuncts until now).
    """
    sweep = conf_param.sweep or {}
    hb = sweep.get("holdout_bias") or {}
    return {
        "count_unbiased_conf": conf_param._raw,
        "f1_max_conf": sweep.get("f1_max_conf"),
        "holdout_bias": hb.get("count_bias_mean") if isinstance(hb, dict) else None,
        # The pooled bias above is the one number a class-compensating refusal reads "fine" on, so
        # the per-class biases the gate actually judged travel beside it (K4 #4).
        "per_class_holdout_bias": {cid: s["count_bias_mean"]
                                   for cid, s in (hb.get("per_class") or {}).items()},
        "per_class_count_bias_failures": sweep.get("per_class_count_bias_failures"),
        "holdout_missing_classes": sweep.get("holdout_missing_classes"),
        "passed_holdout": sweep.get("passed_holdout"),
        "failures": sweep.get("failures"),
        "conf_censored": sweep.get("conf_censored"),
        "conf_floor_mismatch": sweep.get("conf_floor_mismatch"),
        "count_bias_tolerance": sweep.get("count_bias_tolerance"),
        "count_error_tolerance": sweep.get("count_error_tolerance"),
        "count_error_p90": hb.get("count_error_p90") if isinstance(hb, dict) else None,
        "disjoint": sweep.get("disjoint"),
        "content_overlap_frac": sweep.get("content_overlap_frac"),
        "content_duplicated": sweep.get("content_duplicated"),
        "train_disjointness": sweep.get("train_disjointness"),
        "split_policy_divergence": sweep.get("split_policy_divergence"),
        "split_unlocked_stems": sweep.get("split_unlocked_stems"),
    }


@mcp.tool()
@audited
def force_redraw_cal_holdout_split(
    labels_dir: str | None = None,
    images_dir: str | None = None,
    identity_hash: str | None = None,
    group_by: str = "tile_prefix",
    group_key_map: dict[str, str] | None = None,
    seed: int = 0,
    holdout_ratio: float = 0.5,
    reason: str = "",
) -> dict:
    """Deliberately redraw a LOCKED calibration/holdout split (K1 admin action).

    A cal/holdout split locks on its first draw (``resolve_locked_cal_holdout_split``) so the
    "held-out validation" gate can never silently pass on a different, weaker holdout drawn
    after the fact. Redrawing one is a real, audited decision — never automatic, never a hidden
    kwarg on a high-traffic tool like ``run_inference`` — so it is its own small tool. ``reason``
    is required and non-empty, and every redraw (this one included) is appended to the lock's
    ``redraw_history`` with its policy, timestamp, and the OLD split's membership captured before
    it is overwritten — so a redraw-until-it-passes pattern is visible on review even though
    nothing here enforces that a reason differ from a prior one; the defense is a reviewable
    audit trail, not an automatic block.

    Provide either ``labels_dir`` (the identity is derived as ``dataset_hash(labels_dir)``, and
    its stems are re-scanned) or ``identity_hash`` directly (e.g. a review-reference hash — in
    that case the existing lock's own calibration+holdout stems are reused as the redraw's stem
    universe, since a review reference has no labels directory to re-scan).

    Args:
        labels_dir: Labeled dir whose GT identity locked the split (mutually exclusive with
            ``identity_hash`` — if both are omitted, or ``identity_hash`` is given with no
            existing lock and no ``labels_dir``, this refuses).
        images_dir: Images for ``labels_dir``. When given, stems are the same labels-intersect-
            images-on-disk universe ``run_inference``'s calibration uses (K1 finding 4) — a stem
            whose image was deleted/renamed never enters the redraw's stem universe. Omitted ->
            every labeled stem is used regardless of whether an image still exists for it
            (the pre-K1 behavior), for a caller that has no images directory to check against.
        identity_hash: The locked split's identity hash directly.
        group_by: New grouping policy — ``"tile_prefix"`` / ``"stem"`` (ignored if
            ``group_key_map`` is given).
        group_key_map: Explicit ``{stem: group_key}`` map covering every stem, overriding
            ``group_by``.
        seed: New split seed.
        holdout_ratio: New calibration/holdout fraction.
        reason: Required, non-empty justification for this redraw — recorded in the audit log
            alongside the old and new split membership.
    """
    if not reason or not reason.strip():
        return {"error": "reason is required (a non-empty justification) for a force_redraw"}
    if not labels_dir and not identity_hash:
        return {"error": "provide either labels_dir or identity_hash"}

    from datetime import datetime, timezone

    from tcip_mcp.audit import record_event
    from tcip_mcp.pipelines.data.splits import (
        cal_holdout_lock_path, count_label_lines, label_image_stems,
        resolve_locked_cal_holdout_split,
    )
    from tcip_mcp.pipelines.resolution import dataset_hash
    from tcip_mcp.utils.atomic_io import read_json

    if identity_hash is None:
        identity_hash = dataset_hash(labels_dir)

    old_lock = read_json(cal_holdout_lock_path(identity_hash), default=None)
    old_membership = ({"calibration": old_lock.get("calibration", []),
                       "holdout": old_lock.get("holdout", [])} if old_lock else None)

    if labels_dir:
        # The SAME labels-intersect-images scan _calibrate_operating_point uses (K1 finding 4),
        # not a second independent glob — with images_dir omitted this degrades to the prior
        # labels-only scan (stem_to_image unused here either way).
        stems, _ = label_image_stems(labels_dir, images_dir)
        annotation_counts = {s: count_label_lines(labels_dir, s) for s in stems}
    elif old_lock:
        stems = sorted(set(old_lock.get("calibration", [])) | set(old_lock.get("holdout", [])))
        annotation_counts = None
    else:
        return {"error": f"no existing lock for identity_hash={identity_hash!r}, and no "
                          "labels_dir to derive stems from"}

    new_lock = resolve_locked_cal_holdout_split(
        stems, identity_hash=identity_hash, annotation_counts=annotation_counts,
        group_by=group_by, group_key_map=group_key_map, holdout_ratio=holdout_ratio, seed=seed,
        force_redraw=True, timestamp=datetime.now(timezone.utc).isoformat(),
    )
    new_membership = {"calibration": new_lock["calibration"], "holdout": new_lock["holdout"]}

    # @audited (the decorator on this tool) logs only the call's kwargs, never the RETURN value —
    # and the auditable fact here is what the redraw actually produced, not just that one was
    # requested. record_event brackets that the same way envelope.py's training loop closes the
    # "no audit record for what happened inside the call" hole. A distinct tool name (not this
    # function's own) so the two log entries — @audited's call-args line and this result line —
    # don't collide under one ambiguous schema.
    record_event(
        "force_redraw_cal_holdout_split_result",
        {"identity_hash": identity_hash, "group_by": group_by, "group_key_map": group_key_map,
         "seed": seed, "holdout_ratio": holdout_ratio, "reason": reason},
        old_membership=old_membership, new_membership=new_membership,
    )

    return {"identity_hash": identity_hash, "reason": reason,
            "old_membership": old_membership, "new_membership": new_membership}


@mcp.tool()
@audited
def run_inference(
    checkpoint_path: str,
    image_paths: list[str] | None = None,
    images_dir: str | None = None,
    conf_threshold: float = DEFAULT_CONF,
    device: str | None = None,
    tile: bool | None = None,
    tile_size: int | None = None,
    overlap: float | None = None,
    tile_batch_size: int = 96,
    global_nms_iou: float = DEFAULT_NMS_IOU,
    max_dets: int = DEFAULT_MAX_DETS,
    postprocess: str = "nms",
    dry_run: bool = False,
    trait: str | None = None,
    calibration_labels_dir: str | None = None,
    calibration_images_dir: str | None = None,
    experiment_id: str | None = None,
    group_by: str = "tile_prefix",
    group_key_map: dict[str, str] | None = None,
    split_seed: int = 0,
    split_holdout_ratio: float = 0.5,
) -> dict:
    """Run a trained model on images.

    Provide either image_paths (specific images) or images_dir (all images
    in a directory). Set ``tile=True`` for SAHI-style sliding-window detection on
    high-resolution imagery with many small objects (detection heads only).

    Args:
        checkpoint_path: Path to model .pt checkpoint.
        image_paths: List of specific image paths.
        images_dir: Directory containing images to process.
        conf_threshold: Minimum confidence score.
        device: Device to use ('cuda' or 'cpu').
        tile: Enable tiled (SAHI-style) detection inference. ``None`` (default) is a documented
            default (K10 finding 3), not silently ``False``/``True`` — its provenance is stamped
            ``"default"`` vs ``"explicit"`` so a caller who deliberately chose one way is
            distinguishable from one who left it unset.
        tile_size: Sliding-window tile edge (px). ``None`` (default) derives it from the
            checkpoint's training tile geometry so inference matches the trained scale; a value
            overrides. Foreign/legacy checkpoints with no geometry fall back to 640 with a warning.
        overlap: Fractional tile overlap (stride = tile_size*(1-overlap)). ``None`` derives from the
            checkpoint (else 0.2).
        tile_batch_size: Tiles per forward batch.
        global_nms_iou: Cross-tile global NMS IoU threshold.
        max_dets: Full-frame detection cap (after any tiled merge).
        postprocess: Cross-tile merge — "nms" suppresses overlaps, "nmm" unions boxes split
            across a tile seam (better for an object straddling a boundary).
        trait: Trait name (with ``calibration_labels_dir``) to DERIVE the confidence operating point
            per dataset instead of pinning a default — the count is the phenotype, so conf must be
            calibrated (CV0). Absent -> the byte-identical raw path (conf=score_threshold, unvalidated).
        calibration_labels_dir: Labeled dir for a disjoint cal/holdout split to calibrate + held-out
            validate the operating point. Its GT identity scopes the resolved conf (dataset firewall).
        calibration_images_dir: Images for the calibration labels (defaults to ``images_dir``).
        experiment_id: The run that produced the checkpoint, for provenance. Best-effort resolved
            (checkpoint's own stamp, then the registry) when omitted; a raw/foreign checkpoint
            legitimately has none. Also gates calibration's train-disjointness check (K1): a
            *known* run whose training split can't be read/reconstructed fails that check closed.
        group_by: Grouping policy for the LOCKED calibration/holdout split (K1) — ``"tile_prefix"``
            (default) or ``"stem"``. Ignored when ``group_key_map`` is given. Only the FIRST
            calibration call for a given calibration-labels identity draws the split; later calls
            return the same locked split regardless of this argument (see
            ``force_redraw_cal_holdout_split`` to redraw deliberately).
        group_key_map: An agent-derived ``{stem: group_key}`` map overriding ``group_by`` for the
            locked calibration/holdout split — must cover every stem in ``calibration_labels_dir``.
        split_seed: Split seed for the LOCKED calibration/holdout split (K1 finding 5) — like
            ``group_by``, only takes effect on the FIRST calibration call for a given
            calibration-labels identity; a later call's declared value is compared to the lock and
            any divergence is reported in ``sweep_summary``/the resolved bundle rather than
            silently ignored.
        split_holdout_ratio: Calibration/holdout fraction for the LOCKED split (K1 finding 5) —
            same first-call-only semantics as ``split_seed``.
    """
    if not Path(checkpoint_path).is_file():
        return {"error": f"Checkpoint not found: {checkpoint_path}"}

    # K10 finding 3: resolve the tiled bool ONCE, here, so every behavioral and provenance site
    # below reads the same resolved value — a caller who left ``tile`` unset gets DEFAULT_TILED
    # behavior everywhere, and the ORIGINAL None-or-bool is never passed further as a live value
    # (that would make e.g. predict_batch dispatch on falsy None and silently run untiled).
    tiled_source = "explicit" if tile is not None else "default"
    resolved_tile_bool = DEFAULT_TILED if tile is None else tile

    if dry_run:
        # Report the effective operating point without loading the model or running inference, so the
        # agent can see what conf/NMS/tiling will govern the object count before committing to a run.
        return {
            "dry_run": True,
            "checkpoint_path": checkpoint_path,
            "operating_point": {
                "conf": conf_threshold,
                "cross_tile_nms": global_nms_iou if resolved_tile_bool else None,
                "tiled": resolved_tile_bool,
                "tiled_source": tiled_source,
                # Stage-6 review: the checkpoint isn't loaded in a dry run, so neither value is
                # actually "derived" yet — an explicit value is known; an unset one is only a
                # PENDING derivation (or the fabricated default, if the checkpoint has none).
                "tile_size": tile_size if tile_size is not None else "pending-checkpoint-derivation",
                "overlap": overlap if overlap is not None else "pending-checkpoint-derivation",
                "max_dets": max_dets,
                "postprocess": postprocess,
            },
            "note": ("These operating-point values govern the object count (the phenotype for count "
                     "traits). For a trait with a labeled subset, resolve them per dataset "
                     "(resolve_operating_point) so the count is calibrated, not a default."),
        }

    # Lazy import to avoid torch import at module level
    from tcip_mcp.pipelines.inference.predictor import build_predictor
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point
    from tcip_mcp.pipelines.resolution import dataset_hash, raw_operating_point

    # Thread NMS IoU + the full-frame detection cap into the model so they govern which boxes exist
    # (torchvision's own in-model thresholds), not just cross-tile merge — else nms_iou has no
    # effect on an untiled run and dense scenes truncate at the framework default.
    predictor = build_predictor(
        checkpoint_path=checkpoint_path,
        device=device,
        score_threshold=conf_threshold,
        nms_iou=global_nms_iou,
        max_dets=max_dets,
    )

    # Producing-model identity resolved BEFORE calibration (K1), not after: the calibration's
    # train-disjointness gate needs the checkpoint's own experiment_id to check the cal/holdout
    # images against that run's training split. sha is cached (never re-hashed per call).
    from tcip_mcp.model_registry import resolve_model_identity

    identity = resolve_model_identity(checkpoint_path, experiment_id=experiment_id)

    # CV2/K10: derive the tile geometry from the checkpoint's training geometry unless the caller
    # pinned it, so a tiled run doesn't silently infer at a different scale than it trained at
    # (which shifts the object count — the phenotype). The shared resolver (K10) is a pure
    # fact-return — it never refuses — so the warn-and-proceed policy for this exploratory path
    # lives here, distinct from the delivery-gating path's refuse policy in
    # ``run_full_frame_evaluation``.
    from tcip_mcp.pipelines.inference.predictor import resolve_tile_geometry

    resolved_tile, tile_size_source, resolved_overlap, overlap_source = resolve_tile_geometry(
        predictor, tile_size=tile_size, overlap=overlap)
    geometry_warning = None
    if tile_size_source == "derived" and resolved_tile_bool and resolved_tile != DEFAULT_TILE_SIZE:
        # Loud, not just provenance: counts change vs the old pinned 640 for this checkpoint.
        logger.info("tile_size %d derived from the checkpoint's training geometry "
                    "(was pinned %d before derivation existed)", resolved_tile, DEFAULT_TILE_SIZE)
    elif tile_size_source == "default" and resolved_tile_bool:
        geometry_warning = (
            "checkpoint carries no training tile geometry; using default "
            f"{DEFAULT_TILE_SIZE} — counts may not match training scale. Retrain (geometry now "
            "persisted) or pass tile_size explicitly."
        )
        logger.warning(geometry_warning)
    # overlap_source == "default" is expected and unremarkable for a model with no persisted
    # overlap analog — only tile_size's absence changes the object count's scale, so only
    # tile_size's default fallback is worth a warning.

    if image_paths is None:
        if images_dir is None:
            return {"error": "Provide either image_paths or images_dir"}
        p = Path(images_dir)
        image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        image_paths = sorted(str(f) for f in p.iterdir() if f.suffix.lower() in image_exts)

    # Resolve the confidence operating point. CV0: with a trait + labeled calibration dir, DERIVE it
    # per dataset (count-unbiased + held-out validated); otherwise the byte-identical raw path.
    extra: dict = {}
    if trait and calibration_labels_dir:
        cal_images = calibration_images_dir or images_dir
        try:
            bundle, cal_hash, n_excluded_incomplete_attribute = _calibrate_operating_point(
                predictor, trait, calibration_labels_dir, cal_images,
                tile=resolved_tile_bool, tile_size=resolved_tile, overlap=resolved_overlap,
                tile_size_source=tile_size_source, tiled_source=tiled_source,
                tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou, postprocess=postprocess,
                cross_tile_nms=(global_nms_iou if global_nms_iou != DEFAULT_NMS_IOU else None),
                max_dets=(max_dets if max_dets != DEFAULT_MAX_DETS else None),
                group_by=group_by, group_key_map=group_key_map,
                experiment_id=identity["experiment_id"],
                seed=split_seed, holdout_ratio=split_holdout_ratio,
            )
        except ValueError as exc:
            # A locked cal/holdout split refusing this call (K1 finding 4): a calibration stem's
            # image/label was deleted/renamed since the split locked, or the lock file is corrupt.
            # Clean refusal here, not a bare KeyError from a stale stem_to_image lookup downstream.
            return {"error": str(exc)}
        conf_param = bundle.get("conf")
        conf = (conf_param.value if conf_param.is_shippable
                else conf_param.unvalidated_value(acknowledge_unvalidated=True))
        max_dets = int(bundle.get("max_dets")._raw)
        global_nms_iou = float(bundle.get("cross_tile_nms")._raw or global_nms_iou)
        # Apply the resolved operating point to the model so it governs which boxes exist.
        predictor.score_threshold = conf
        set_detector_operating_point(predictor.model, score_thresh=conf, detections_per_img=max_dets)
        op_bundle = bundle
        # Dataset-scope firewall: the conf is scoped to the calibration GT. The inference target is
        # usually UNLABELED, so its GT identity (a content hash) is undefined — pass None and record
        # 'not-comparable-unlabeled-target'. Only when inferencing the SAME labeled set it calibrated
        # on can we compare real hashes and flag cross-dataset inheritance.
        inf_stems = [Path(pp).stem for pp in image_paths]
        cal_label_stems = {pp.stem for pp in Path(calibration_labels_dir).glob("*.json")}
        same_images = calibration_images_dir is None or (
            images_dir is not None and Path(calibration_images_dir) == Path(images_dir))
        if same_images and inf_stems and set(inf_stems) == cal_label_stems:
            target_hash, cross_dataset_check = dataset_hash(calibration_labels_dir, stems=inf_stems), "same-labeled-set"
        else:
            target_hash, cross_dataset_check = None, "not-comparable-unlabeled-target"
        issues = bundle.shippable_issues(target_dataset_hash=target_hash)
        # Channel firewall (T6-3): probe ONE target raster and check its band count against the
        # checkpoint's in_chans via validate_resolved_bundle, so a channel-wrong inference surfaces in
        # the provenance rather than being silently coerced by the loader.
        if image_paths:
            from tcip_mcp.pipelines.derivations import probe_channels
            from tcip_mcp.pipelines.resolution import (
                ResolvedBundle, default as _resolved_default, validate_resolved_bundle,
            )
            try:
                probed = int(probe_channels(image_paths[0]))
            except Exception:
                probed = None
            if probed is not None:
                chan_bundle = ResolvedBundle(trait=trait or "", dataset_hash=None, params={
                    "in_chans": _resolved_default(
                        "in_chans", int(getattr(predictor, "in_chans", 3)),
                        derivation_class="deterministic")})
                issues = issues + validate_resolved_bundle(chan_bundle, probed_channels=probed)
        # validated only when held-out passed AND nothing is un-shippable under the target actually used.
        extra = {
            "validated": bool(bundle.is_shippable and not issues),
            "shippable_issues": issues,
            "cross_dataset_check": cross_dataset_check,
            "conf_source": "calibration",
            "dataset_hash": cal_hash,
            "sweep_summary": _sweep_summary(conf_param),
            "n_excluded_incomplete_attribute": n_excluded_incomplete_attribute,
        }
        # The full sweep can be large — persist it and return the path (provenance emits has_sweep).
        try:
            from tcip_mcp.project_paths import project_root
            from tcip_mcp.utils.atomic_io import atomic_write_json
            art = project_root() / ".tcip" / "artifacts"
            art.mkdir(parents=True, exist_ok=True)
            sweep_path = art / f"operating_point_sweep_{cal_hash}.json"
            atomic_write_json(sweep_path, {"trait": trait, "dataset_hash": cal_hash,
                                           "sweep": conf_param.sweep})
            extra["sweep_path"] = str(sweep_path)
        except Exception:
            logger.warning("could not persist operating-point sweep", exc_info=True)
    else:
        # Raw inference has no per-dataset calibration: the model already carries score_threshold as
        # its in-model conf; the bundle stamps it validated_vs_gt=false so the un-trustworthiness of
        # this uncalibrated operating point (the count is the phenotype) travels with the result.
        op_bundle = raw_operating_point(
            conf=conf_threshold, cross_tile_nms=global_nms_iou, tiled=resolved_tile_bool,
            tile_size=resolved_tile, max_dets=max_dets, tile_size_source=tile_size_source,
            tiled_source=tiled_source,
        )
        extra = {"validated": False, "conf_source": "default"}

    # Preflight: warn (don't fail) when a slow workload will run on CPU because CUDA isn't
    # available — full tiled inference over thousands of images is hours on CPU vs minutes on
    # a GPU. Install a CUDA torch build (see environment.yml) to use the card.
    cpu_warning = None
    if device != "cpu" and (resolved_tile_bool or len(image_paths) > 8):
        import torch

        if not torch.cuda.is_available():
            cpu_warning = (
                f"CUDA not available — running {len(image_paths)} image(s)"
                f"{' tiled' if resolved_tile_bool else ''} on CPU, which is much slower. Install a "
                "CUDA torch build (see environment.yml) to use the GPU."
            )
            logger.warning(cpu_warning)

    results = predictor.predict_batch(
        image_paths, tile=resolved_tile_bool, tile_size=resolved_tile, overlap=resolved_overlap,
        tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou, postprocess=postprocess,
    )
    total_detections = sum(r["count"] for r in results)

    # Producing-model identity (resolved above, before calibration) travels with the result so every
    # downstream deliverable can name the exact checkpoint (content hash) + run behind the count.
    from datetime import datetime, timezone

    # This run's name→id map, derived once here (assign_class_ids over the inference dataset's
    # classes.json for the config's subject/attribute) and reused for both recording and decode, so
    # export records it in operating_point.json and decodes predictions to names through this one map —
    # consistent within the run. Single-class detection is order-invariant ({subject: 0}), so this
    # matches the training run's map today. Binding it to the *training* run's recorded map — so a
    # registry whose value order was edited between train and inference cannot mis-decode a multi-value
    # attribute — lands with K4/K5, which is where attribute order first matters.
    id_map = None
    try:
        data_cfg = (getattr(predictor, "config", {}) or {}).get("data") or {}
        subject = data_cfg.get("subject")
        if subject and images_dir:
            from tcip_mcp.pipelines.data.datasets import _resolve_registry_id_map

            _reg, id_map = _resolve_registry_id_map(images_dir, subject, data_cfg.get("attribute"))
    except Exception:  # noqa: BLE001 — no run scope for the map; predictions decode by raw id then
        id_map = None
    out = {
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": identity["sha256"],
        "experiment_id": identity["experiment_id"],
        "images_dir": images_dir,
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "image_count": len(results),
        "total_detections": total_detections,
        "tiled": resolved_tile_bool,
        # Stage-6 review: overlap has no home in the ResolvedBundle's tracked params (only conf/
        # cross_tile_nms/tiled/tile_size/max_dets are) — surface the value this specific call
        # actually ran at directly, rather than silently drop it after resolving it.
        "overlap": resolved_overlap,
        "overlap_source": overlap_source,
        "operating_point": op_bundle.to_provenance()["operating_point"],
        "id_map": id_map,
        "results": results,
        **extra,
    }
    warning = "; ".join(w for w in (geometry_warning, cpu_warning) if w)
    if warning:
        out["warning"] = warning
    return out


@mcp.tool()
@audited
def export_predictions(
    checkpoint_path: str,
    images_dir: str,
    output_dir: str,
    conf_threshold: float = DEFAULT_CONF,
    device: str | None = None,
    tile: bool | None = None,
    tile_size: int | None = None,
    overlap: float | None = None,
    tile_batch_size: int = 96,
    global_nms_iou: float = DEFAULT_NMS_IOU,
    max_dets: int = DEFAULT_MAX_DETS,
    postprocess: str = "nms",
    trait: str | None = None,
    calibration_labels_dir: str | None = None,
    calibration_images_dir: str | None = None,
    experiment_id: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Run inference and save predictions as per-image COCO/JSON files.

    Routes through ``run_inference`` so this delivery door resolves the same firewalled
    operating point (conf/NMS/tiling/max_dets) — earlier it built its own bare predictor and
    so truncated the count at the framework default and shipped labels with no provenance.
    Writes ``<stem>.json`` per image plus an ``operating_point.json`` stamp beside them.

    A prediction bucket (``output_dir``) that already carries review verdicts is immutable: by
    default the export is redirected to a fresh run-scoped bucket (``<dir>@r2``, ``@r3`` — next
    free) and the dir actually written is returned as ``output_dir``. Pass ``overwrite=True`` to
    write in place only when the bucket has zero verdicts; with verdicts present it is refused
    (error names the count and a suggested dir) so a re-run never orphans recorded verdicts.

    Args:
        checkpoint_path: Path to model .pt checkpoint.
        images_dir: Directory containing input images.
        output_dir: Directory for output .json prediction files.
        conf_threshold: Minimum confidence score.
        device: Device to use.
        tile: Tiled (SAHI-style) inference for small dense objects. ``None`` (default) forwards to
            ``run_inference`` unresolved — see its own ``tile`` doc: a documented default distinct
            from an explicit choice, not silently ``False``.
        tile_size: Sliding-window tile edge (px).
        overlap: Fractional tile overlap.
        tile_batch_size: Tiles per forward batch.
        global_nms_iou: Cross-tile NMS IoU.
        max_dets: Full-frame detection cap.
        postprocess: Cross-tile merge — "nms" or "nmm".
        trait: Trait to calibrate the operating point per dataset (with ``calibration_labels_dir``).
        calibration_labels_dir: Labeled dir for calibrating + held-out validating the operating point.
        calibration_images_dir: Images for the calibration labels (defaults to ``images_dir``).
        overwrite: Write into ``output_dir`` even if it exists. Refused if the bucket has review
            verdicts; the default (False) auto-redirects to a fresh bucket instead.
    """
    from tcip_mcp.prediction_buckets import BucketHasVerdicts, resolve_writable_bucket
    from tcip_mcp.project_paths import resolve_state

    # Resolve the writable bucket before the (expensive) inference so a verdict-blocked overwrite
    # fails fast; verdicts live under the pinned project's ``.tcip/state``.
    out_path = Path(output_dir)
    parent, base_name = out_path.parent, out_path.name
    review_state_dir = resolve_state(Path(".tcip") / "state")
    try:
        resolution = resolve_writable_bucket(
            review_state_dir, base_name, lambda n: [parent / n], overwrite=overwrite)
    except BucketHasVerdicts as exc:
        return {"error": str(exc), "verdict_count": exc.count,
                "suggested_bucket": str(parent / exc.suggested)}
    out = parent / resolution.name

    result = run_inference(
        checkpoint_path=checkpoint_path, images_dir=images_dir, conf_threshold=conf_threshold,
        device=device, tile=tile, tile_size=tile_size, overlap=overlap,
        tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou, max_dets=max_dets,
        postprocess=postprocess, trait=trait,
        calibration_labels_dir=calibration_labels_dir, calibration_images_dir=calibration_images_dir,
        experiment_id=experiment_id,
    )
    if "error" in result:
        return result

    from tcip_mcp.utils.atomic_io import atomic_write_json

    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    # Producer carries the checkpoint's content hash so an accepted prediction's GT names the exact
    # model that produced it, not just a (collidable) filename stem.
    sha = result.get("checkpoint_sha256")
    producer = f"model:{Path(checkpoint_path).stem}" + (f"@{sha[:12]}" if sha else "")
    id_map = result.get("id_map")
    for r in result["results"]:
        out_json = out / f"{Path(r['image']).stem}.json"
        write_predictions_json(out_json, r, created_by=producer, id_map=id_map)
        written.append(str(out_json))

    # Stamp the operating point + producing-model identity beside the delivered labels. ``validated``
    # is derived from the run's resolved bundle (true only when a held-out calibration passed) —
    # never hardcoded, or a passing calibration would be recorded as unvalidated (and vice versa).
    atomic_write_json(out / "operating_point.json",
                      {"operating_point": result.get("operating_point"),
                       "id_map": id_map,
                       "validated": bool(result.get("validated", False)),
                       "shippable_issues": result.get("shippable_issues", []),
                       "checkpoint": Path(checkpoint_path).stem,
                       "checkpoint_sha256": sha,
                       "experiment_id": result.get("experiment_id"),
                       "images_dir": images_dir,
                       "produced_at": result.get("produced_at")})

    # Close the data→model→predictions chain: link this bucket into the producing run's lineage.
    # Additive first-write — the terminal-state lock permits it into a still-empty predictions field.
    exp_id = result.get("experiment_id")
    if exp_id:
        try:
            from tcip_mcp.experiments import update_lineage

            update_lineage(exp_id, predictions=str(out))
        except Exception:
            logger.warning("could not link predictions into experiment lineage", exc_info=True)

    return {"image_count": len(written), "output_dir": str(out), "files": written,
            "bucket_redirected": resolution.redirected,
            "requested_output_dir": output_dir if resolution.redirected else None,
            "operating_point": result.get("operating_point"),
            "validated": bool(result.get("validated", False)),
            "conf_source": result.get("conf_source"),
            "checkpoint_sha256": sha,
            "experiment_id": exp_id}


@mcp.tool()
@audited
def tabulate_counts(
    checkpoint_path: str,
    images_dir: str,
    output_path: str,
    conf_threshold: float = DEFAULT_CONF,
    device: str | None = None,
    tile: bool | None = None,
    tile_size: int | None = None,
    overlap: float | None = None,
    tile_batch_size: int = 96,
    global_nms_iou: float = DEFAULT_NMS_IOU,
    max_dets: int = DEFAULT_MAX_DETS,
    postprocess: str = "nms",
    trait: str | None = None,
    calibration_labels_dir: str | None = None,
    calibration_images_dir: str | None = None,
    experiment_id: str | None = None,
    acknowledge_unvalidated: bool = False,
) -> dict:
    """Run inference and export a CSV summary of detection counts per image.

    Routes through ``run_inference`` so the per-image counts resolve the same firewalled
    operating point (conf/NMS/tiling/max_dets) as ``run_inference``/``export_predictions`` —
    the CSV is a count-bearing deliverable (the count is the phenotype for count traits), so it
    must not be produced at a different, untiled, truncating operating point. Earlier this door
    hardcoded ``conf=0.5`` and passed no tiling/max_dets, under-reporting dense
    small-object counts relative to the other two doors.

    Delivery gate: the count is a phenotype, so the CSV is not written unless the count operating
    point is validated on the run's own resolved bundle (not a caller string) — or
    ``acknowledge_unvalidated=True`` writes a clearly-flagged provisional CSV stamped
    ``measurement_validated=false``. Calibrate per dataset (``trait`` + ``calibration_labels_dir``)
    to reach a validated count.

    Args:
        checkpoint_path: Path to model .pt checkpoint.
        images_dir: Directory containing input images.
        output_path: Path for the output CSV file.
        conf_threshold: Minimum confidence score.
        device: Device to use.
        tile: Tiled (SAHI-style) inference for small dense objects. ``None`` (default) forwards to
            ``run_inference`` unresolved — see its own ``tile`` doc: a documented default distinct
            from an explicit choice, not silently ``False``.
        tile_size: Sliding-window tile edge (px).
        overlap: Fractional tile overlap.
        tile_batch_size: Tiles per forward batch.
        global_nms_iou: Cross-tile NMS IoU.
        max_dets: Full-frame detection cap.
        postprocess: Cross-tile merge — "nms" or "nmm".
        trait: Trait to calibrate the operating point per dataset (with ``calibration_labels_dir``).
        calibration_labels_dir: Labeled dir for calibrating + held-out validating the operating point.
        calibration_images_dir: Images for the calibration labels (defaults to ``images_dir``).
        acknowledge_unvalidated: Write the count CSV even when the operating point is unvalidated,
            stamping it ``measurement_validated=false`` so the un-trustworthiness travels downstream.
    """
    result = run_inference(
        checkpoint_path=checkpoint_path,
        images_dir=images_dir,
        conf_threshold=conf_threshold,
        device=device,
        tile=tile,
        tile_size=tile_size,
        overlap=overlap,
        tile_batch_size=tile_batch_size,
        global_nms_iou=global_nms_iou,
        max_dets=max_dets,
        postprocess=postprocess,
        trait=trait,
        calibration_labels_dir=calibration_labels_dir,
        calibration_images_dir=calibration_images_dir,
        experiment_id=experiment_id,
    )
    if "error" in result:
        return result

    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE,
        VALIDATED_HELD_OUT,
        VALIDATED_SHIPPABLE,
        check_delivery_gate,
    )

    op = result.get("operating_point") or {}
    conf_prov = op.get("conf") or {}
    # The count operating point's validity is the run's RESOLVED bundle state, not a caller string:
    # prefer the conf param's on-the-run reference, falling back to the resolved validated bool.
    op_ref = conf_prov.get("validated_vs_gt")
    if op_ref not in VALIDATED_SHIPPABLE:
        op_ref = VALIDATED_HELD_OUT if result.get("validated") else VALIDATED_FALSE
    gate = check_delivery_gate({"operating_point": op_ref},
                               acknowledge_unvalidated=acknowledge_unvalidated)
    if not gate.ok:
        return {
            "error": gate.reason,
            "operating_point_validated": op_ref,
            "operating_point": result.get("operating_point"),
            "validated": False,
            "image_count": result["image_count"],
            "total_detections": result["total_detections"],
        }

    provenance = {
        "producer_model_sha256": result.get("checkpoint_sha256"),
        "experiment_id": result.get("experiment_id"),
        "operating_point_conf": op.get("conf"),
        "produced_at": result.get("produced_at"),
    }
    csv_path = export_detection_csv(
        result["results"], output_path, provenance=provenance,
        measurement_validated=gate.stamp["operating_point"],
        acknowledge_unvalidated=acknowledge_unvalidated,
    )
    return {
        "csv_path": csv_path,
        "image_count": result["image_count"],
        "total_detections": result["total_detections"],
        # Carry the operating point + producing model that produced these counts — the CSV is a
        # count-bearing deliverable; the numbers are only as trustworthy as what stands behind them.
        "operating_point": result.get("operating_point"),
        "validated": bool(result.get("validated", False)),
        "operating_point_validated": gate.stamp["operating_point"],
        "conf_source": result.get("conf_source"),
        "checkpoint_sha256": result.get("checkpoint_sha256"),
        "experiment_id": result.get("experiment_id"),
    }
