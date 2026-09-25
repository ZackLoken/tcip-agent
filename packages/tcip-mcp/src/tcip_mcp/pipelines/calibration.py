"""The calibrate/summarize pair every inference entry point shares: resolve a per-dataset operating
point from a labeled split, and its compact, response-safe gate-evidence summary.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tcip_mcp.pipelines.data.splits import DEFAULT_CAL_SEED, DEFAULT_HOLDOUT_RATIO

logger = logging.getLogger(__name__)


def calibration_reference_inputs(resolver_inputs: dict, locked: dict, stems: list[str]) -> dict:
    """Where a calibration's reference came from, read off the resolver's own inputs: the
    labels directory (with the drawn stems under a selection) and the split identity it drew."""
    drawn = (locked.get("redraw_history") or [{}])[-1]
    stated = {"split_identity_hash": resolver_inputs["dataset_hash"],
              "split_content_hash": drawn.get("new_content_hash")}
    labels = resolver_inputs["calibration_labels_dir"]
    selection_dir = resolver_inputs["selection_dir"]
    if selection_dir is None:
        return {"label_dirs": {"calibration": labels}, "stated_values": stated}
    stated["selection_dir"] = selection_dir
    return {"label_stems": {"calibration": {"path": labels, "stems": stems}},
            "stated_values": stated}


def calibrate_operating_point(predictor, trait, labels_dir, images_dir, *,
                               tile, tile_size, overlap, tile_batch_size,
                               global_nms_iou, postprocess, cross_tile_nms, max_dets,
                               tile_resize=None,
                               tile_size_source="default", tile_size_derived_from=None,
                               tiled_source="default",
                               group_by=None, group_key_map=None, experiment_id=None,
                               seed=DEFAULT_CAL_SEED, holdout_ratio=DEFAULT_HOLDOUT_RATIO,
                               selection_dir=None):
    """Resolve a per-dataset operating point from a labeled split.

    Returns ``(bundle, hash, n_excluded_incomplete_attribute, evidence)``. The third value is the
    count of cal/holdout stems dropped whole because an instance was unlabeled for ``attribute``;
    it is not a field on the bundle. ``evidence`` is the name of the resolver this ran, the
    arguments it ran it over (the dict passed to ``resolve_operating_point``, without the trait and
    the producing run), and the locations of the reference those arguments came from:
    ``label_dirs.calibration`` for a whole-directory draw, ``label_stems.calibration`` with
    ``stated_values.selection_dir`` under a selection, plus ``calibration_stems`` and, under a
    selection, ``excluded``.

    The count-unbiased center-match curve and held-out bias check run the predictor path the
    delivery will use (same tile/tile_size/tile_resize/overlap/nms/postprocess) at a floor conf,
    over a disjoint cal/holdout split of the labeled dir locked on its first draw
    (``resolve_locked_cal_holdout_split``); ``seed``/``holdout_ratio`` only take effect on that
    locking draw. The lock is scoped to the labeled dir's own root (``cal_holdout_scope_root``),
    which ``redraw_calibration_holdout`` states to address it.

    Raises ``ValueError`` when the lock references a stem whose image/label no longer exists or its
    lock file is corrupt; when the run's recorded scope (``run_scope``) names no subject; when an
    attribute run resolves no id map; and through ``json_io.require_reference_ground_truth`` when
    ``labels_dir`` is not an admissible reference.

    ``tile_size_source``/``tiled_source``/``tile_size_derived_from`` are the caller's resolved
    provenance for ``tile_size``/``tile``, forwarded into ``resolve_operating_point``.

    ``selection_dir`` restricts the calibration universe to the ``calibration`` samples of a
    selection whose label documents live under ``labels_dir``. A checkpoint bound to a different
    selection, or trained for a subject or attribute the selection was not drawn for, is refused by
    name, and so is ``group_by``/``group_key_map`` passed beside a selection. Without a selection,
    ``group_by`` defaults to ``splits.DEFAULT_GROUP_BY``. The identity (``dh``, the lock, the
    evidence's ``split_identity_hash``) is ``dataset_hash(labels_dir, stems=universe)`` under a
    selection.
    """
    from tcip_annotation.json_io import require_reference_ground_truth
    from tcip_mcp.pipelines.data.label_queries import json_det_targets
    from tcip_mcp.pipelines.data.splits import (
        cal_holdout_scope_root, count_label_lines, label_image_stems,
        resolve_locked_cal_holdout_split, same_directory, selection_policy_conflict,
    )
    from tcip_mcp.pipelines.operating_point import (
        STAGED_CONF_FLOOR, apply_operating_point, attach_split_policy_provenance,
        derive_max_dets_from_counts, resolve_operating_point,
    )
    from tcip_mcp.pipelines.resolution import dataset_hash
    from tcip_mcp.pipelines.training.evaluation import (
        build_coco_image_record, detection_record, gt_records,
    )
    from tcip_mcp.tools.inference_tools import (
        resolve_decode_id_map, run_scope, unmapped_classified_run,
    )

    # The run's own recorded class space, through the one reader of it: calibration GT reads
    # under the same scope the training targets did, so the swept count cannot diverge from it.
    _checkpoint_scope = run_scope(predictor)
    _subject = _checkpoint_scope.named_subject(repr(getattr(predictor, "path", predictor)))
    _attribute = _checkpoint_scope.attribute
    labels_p = Path(labels_dir)
    require_reference_ground_truth(labels_p)
    policy_conflict = selection_policy_conflict(selection_dir, group_by, group_key_map)
    if policy_conflict:
        raise ValueError(policy_conflict)
    from tcip_mcp.pipelines.data.split_construction import bound_selection_dir

    _data_cfg = (getattr(predictor, "config", {}) or {}).get("data") or {}
    _checkpoint_selection_dir = bound_selection_dir(_data_cfg.get("split") or {})
    if (selection_dir is not None and _checkpoint_selection_dir is not None
            and not same_directory(_checkpoint_selection_dir, selection_dir)):
        raise ValueError(
            f"this checkpoint is bound to the selection at {_checkpoint_selection_dir!r}, not the "
            f"{selection_dir!r} this calibration names: calibrating a bound checkpoint "
            "under a different selection would check its selection disjointness against a side "
            "the checkpoint was never trained or chosen with."
        )
    excluded = None
    selection_sha256 = None
    selection_id_map: dict[str, int] | None = None
    if selection_dir is not None:
        from tcip_mcp.pipelines.data.selection import read_selection
        from tcip_mcp.pipelines.data.splits import selection_calibration_universe
        from tcip_mcp.pipelines.image_utils import resolve_source_path
        from tcip_mcp.pipelines.resolution import selection_digest

        selection = read_selection(selection_dir)
        selection_sha256 = selection_digest(selection)
        if (_subject or None, _attribute or None) != (
                selection.subject or None, selection.attribute or None):
            raise ValueError(
                f"this checkpoint was trained for subject={_subject!r}, attribute="
                f"{_attribute!r}, and the selection at {selection_dir!r} was drawn for "
                f"subject={selection.subject!r}, attribute={selection.attribute!r}: the model "
                "only speaks its training vocabulary, so it cannot be measured against a "
                "reference drawn for another class space."
            )
        selection_id_map = dict(selection.id_map) if selection.id_map else None
        stems, group_by, group_key_map, excluded, annotation_counts, universe_samples = \
            selection_calibration_universe(selection, labels_dir)
        # Each stem's own recorded source and ground truth, never a directory listing's: two
        # directories can hold identically named files.
        stem_to_image = {s: resolve_source_path(universe_samples[s].source) for s in stems}
        gt_path_of = {s: universe_samples[s].ground_truth for s in stems}
    else:
        # The shared labels-intersect-images scan redraw_calibration_holdout also uses: a stem
        # whose image was deleted or renamed never enters the whole-directory universe.
        from tcip_mcp.dataset_layout import label_filename

        stems, stem_to_image = label_image_stems(labels_dir, images_dir)
        # The one caller here holding a name rather than a record composes its path once, here.
        gt_path_of = {s: str(labels_p / label_filename(s)) for s in stems}
    # A selection states the exact map its samples were admitted under; it wins over the run's
    # own decode map for a selection-restricted measurement.
    _cal_id_map = selection_id_map or resolve_decode_id_map(
        predictor, str(labels_p), scope=(_subject, _attribute))
    unmapped = unmapped_classified_run(_checkpoint_scope, _cal_id_map, images_dir=str(labels_p))
    if unmapped is not None:
        raise ValueError(unmapped)
    dh = dataset_hash(labels_dir, stems=(stems if selection_dir is not None else None))
    # The whole-directory universe holds names, and counts each member's document by the path it
    # composed; a selection's universe returned its own counts above.
    if selection_dir is None:
        annotation_counts = {
            s: count_label_lines(gt_path_of[s], subject=_subject, attribute=_attribute)
            for s in stems
        }
    # Detector-cap censoring: derive the collection-pass cap from this split's own density (same
    # formula tcip calibrate-operating-point uses), not the caller's possibly-unrelated max_dets.
    density_cap = derive_max_dets_from_counts(list(annotation_counts.values()))
    locked = resolve_locked_cal_holdout_split(
        stems, identity_hash=dh, scope_root=cal_holdout_scope_root(labels_dir),
        annotation_counts=annotation_counts,
        group_by=group_by, group_key_map=group_key_map, seed=seed, holdout_ratio=holdout_ratio,
        selection_dir=selection_dir,
    )
    if locked.get("unlocked_stems"):
        logger.info(
            "cal/holdout split for %s has %d stem(s) not covered by the existing lock (new since "
            "it was drawn); excluded from this calibration: %s", dh,
            len(locked["unlocked_stems"]), locked["unlocked_stems"][:10],
        )
    cal_stems, hold_stems = locked["calibration"], locked["holdout"]

    applied, applied_attribute_path = apply_operating_point(
        predictor, STAGED_CONF_FLOOR, density_cap)

    n_excluded_incomplete_attribute = 0

    def _records(sub_stems):
        nonlocal n_excluded_incomplete_attribute
        if not sub_stems:
            return []
        results = predictor.predict_batch(
            [stem_to_image[s] for s in sub_stems], tile=tile, tile_size=tile_size,
            overlap=overlap, tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou,
            postprocess=postprocess, tile_resize=tile_resize,
        )
        recs = []
        for s, r in zip(sub_stems, results):
            dt = [detection_record(b, lab, sc)
                  for b, sc, lab in zip(r["boxes"], r["scores"], r["labels"])]
            # GT lifted to the predictor's 1-indexed labels via the loader-side reader (subject +
            # id map), the same reading the run's training targets took.
            target, n_unlabeled = json_det_targets(gt_path_of[s], _subject, _attribute, _cal_id_map)
            # An image with any instance unlabeled for `attribute` is dropped whole from the
            # record set (the missing-label-file precedent), counted rather than silently filtered.
            if n_unlabeled:
                n_excluded_incomplete_attribute += 1
                continue
            recs.append(build_coco_image_record(int(r["width"]), int(r["height"]),
                                                gt_records(target), dt, image_id=s))
        return recs

    cal_records = _records(cal_stems)
    hold_records = _records(hold_stems)
    resolver_inputs = {
        "dataset_hash": dh, "calibration_records": cal_records,
        "holdout_records": hold_records or None, "tile_size": tile_size,
        "tile_size_source": tile_size_source, "tile_size_derived_from": tile_size_derived_from,
        "tiled": tile, "tiled_source": tiled_source,
        "cross_tile_nms": cross_tile_nms, "max_dets": max_dets,
        "staged_conf_floor": applied.get("score_thresh"),
        "staged_conf_floor_attribute_path": applied_attribute_path,
        "selection_dir": selection_dir,
        "calibration_labels_dir": str(labels_p), "selection_sha256": selection_sha256,
    }
    bundle = resolve_operating_point(trait, experiment_id=experiment_id, **resolver_inputs)
    attach_split_policy_provenance(bundle, locked)
    evidence = {
        "resolver": "resolve_operating_point", "inputs": resolver_inputs,
        "reference_inputs": calibration_reference_inputs(resolver_inputs, locked, stems),
        "calibration_stems": stems,
    }
    if excluded is not None:
        evidence["excluded"] = excluded
    return bundle, dh, n_excluded_incomplete_attribute, evidence


def gate_evidence_summary(conf_param) -> dict:
    """Compact, response-safe view of a calibration's gate evidence (the full curve is written to
    disk).

    Includes
    ``disjoint``/``content_overlap_frac``/``train_disjointness``/``selection_disjointness``,
    ``split_policy_divergence``/``split_unlocked_stems`` (via ``attach_split_policy_provenance``),
    ``failures`` and the individual gate fields (``conf_floor_mismatch``, dispersion/localization
    terms).
    """
    evidence = conf_param.gate_evidence or {}
    hb = evidence.get("holdout_bias") or {}
    return {
        "count_unbiased_conf": conf_param.unvalidated_value(acknowledge_unvalidated=True),
        "f1_max_conf": evidence.get("f1_max_conf"),
        "holdout_bias": hb.get("count_bias_mean") if isinstance(hb, dict) else None,
        # The pooled bias above is the one number a class-compensating refusal reads "fine" on, so
        # the per-class biases the gate actually judged travel beside it.
        "per_class_holdout_bias": {cid: s["count_bias_mean"]
                                   for cid, s in (hb.get("per_class") or {}).items()},
        "per_class_count_bias_failures": evidence.get("per_class_count_bias_failures"),
        "per_class_insufficient_images": evidence.get("per_class_insufficient_images"),
        "holdout_missing_classes": evidence.get("holdout_missing_classes"),
        "passed_holdout": evidence.get("passed_holdout"),
        "failures": evidence.get("failures"),
        "conf_censored": evidence.get("conf_censored"),
        "conf_floor_mismatch": evidence.get("conf_floor_mismatch"),
        "count_bias_tolerance_frac": evidence.get("count_bias_tolerance_frac"),
        "pooled_count_bias_tolerance": evidence.get("pooled_count_bias_tolerance"),
        # The pooled tolerance alone doesn't explain a per-class refusal: each class scales
        # against its own typical count.
        "per_class_count_bias_tolerance": evidence.get("per_class_count_bias_tolerance"),
        "pooled_typical_count": evidence.get("pooled_typical_count"),
        "per_class_typical_count": evidence.get("per_class_typical_count"),
        "count_error_tolerance": evidence.get("count_error_tolerance"),
        "count_error_p90": hb.get("count_error_p90") if isinstance(hb, dict) else None,
        "disjoint": evidence.get("disjoint"),
        "content_overlap_frac": evidence.get("content_overlap_frac"),
        "content_shared_with_calibration": evidence.get("content_shared_with_calibration"),
        "train_disjointness": evidence.get("train_disjointness"),
        "selection_disjointness": evidence.get("selection_disjointness"),
        "split_policy_divergence": evidence.get("split_policy_divergence"),
        "split_unlocked_stems": evidence.get("split_unlocked_stems"),
    }
