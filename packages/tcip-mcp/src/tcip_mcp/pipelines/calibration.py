"""The calibrate/summarize pair every inference entry point shares: resolve a per-dataset operating
point from a labeled split, and its compact, response-safe gate-evidence summary.

Both moved out of ``tools/inference_tools.py``: their consumers are cross-module
(``inference_tools._run_inference_verified`` calls both, and a dozen-plus test files exercise
them directly), so a single tools module was no longer their one home.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def calibrate_operating_point(predictor, trait, labels_dir, images_dir, *,
                               tile, tile_size, overlap, tile_batch_size,
                               global_nms_iou, postprocess, cross_tile_nms, max_dets,
                               tile_resize=None,
                               tile_size_source="default", tile_size_derived_from=None,
                               tiled_source="default",
                               group_by=None, group_key_map=None, experiment_id=None,
                               seed=0, holdout_ratio=0.5, split_manifest_dir=None):
    """Resolve a per-dataset operating point from a labeled split.

    Returns ``(bundle, hash, n_excluded_incomplete_attribute, evidence)``. The third value is the
    count of cal/holdout stems dropped whole because an instance was unlabeled for ``attribute``
    (see ``_records`` below); returned to the caller rather than silently filtered, matching
    ``evaluation.py``'s ``n_excluded_incomplete_attribute`` for the same exclusion. It is a separate
    return value, not a field on the bundle: ``run_inference`` surfaces it on its own response dict,
    and it does not travel into the persisted ``operating_point.json`` sidecar that
    ``run_inference`` writes.

    ``evidence`` is what a delivery door needs to earn the validation record behind a validated
    count: the name of the resolver this function just ran, the arguments it ran it over, and the
    locations of the reference those arguments came from. The arguments are the same dict passed to
    ``resolve_operating_point`` here, never a second assembly of them, so the gate a door reopens is
    the gate this calibration passed. The trait and the producing run are left out: they are
    ``open_validation``'s own arguments, and a second spelling of either could disagree with the
    record it is written into.

    The count-unbiased center-match curve + held-out bias check run the same predictor path the
    delivery will use (same tile/tile_size/tile_resize/overlap/nms/postprocess) over a disjoint, locked
    cal/holdout split of the labeled dir (``resolve_locked_cal_holdout_split``: group-coherent,
    seeded, and stable across calls, not a fresh lexicographic cut every time), at a floor conf so
    hesitant detections survive to be swept, so the resolved conf is validated in the regime it
    ships through, not an untiled full-frame model pass. ``seed``/``holdout_ratio`` only take effect
    on the first (locking) draw for this labeled dir's identity hash. That lock is scoped to the
    labeled dir's own root (``cal_holdout_scope_root``), so it is still the same lock after a
    project adoption repins the platform root, and ``redraw_calibration_holdout`` addresses it
    by stating that root.

    Raises ``ValueError`` (propagated from ``resolve_locked_cal_holdout_split``) when the lock
    references a stem whose image/label no longer exists, or its lock file is corrupt; also
    raised by name when ``split_manifest_dir`` is given with no ``images_dir``, a labels-only
    universe the redraw refuses the same way. The caller (``run_inference``) turns either into a
    clean ``{"error": ...}``.

    ``labels_dir`` is read as a measurement reference, so it goes through
    ``json_io.require_reference_ground_truth`` first (the same admissibility rule the classifier
    calibration path applies to its own GT dirs): a directory of the model's own predictions
    clears every numeric gate below, since the model agrees with itself, so nothing downstream
    can catch it.

    ``tile_size_source``/``tiled_source`` are the caller's already-resolved
    provenance for ``tile_size``/``tile``, forwarded into ``resolve_operating_point`` so the
    calibrated bundle doesn't stamp a fabricated tile_size as ``"derived"``, and so it records
    whether calibration itself actually tiled rather than always asserting ``tiled=True``.
    ``tile_size_derived_from`` is the caller's own composed ``derived_from`` text for an explicit
    edge (see ``predictor.explicit_edge_provenance``), forwarded unchanged; ``None`` for every
    other source.

    ``split_manifest_dir`` restricts the calibration universe to one capture date's
    ``calibration`` side of a ``split_manifest`` record (``data_tools.read_split_manifest_dir``)
    instead of every labelled stem with an image: the manifest's ``subject``/``attribute`` must
    equal this run's recorded training scope, the labels directory's date
    (``dataset_layout.annotation_date``) must be one the manifest holds members under, and the
    manifest's ``images_root`` for that date must be ``images_dir``, each refusing by name. A
    checkpoint bound to a different manifest than the one named here is refused by name too.
    ``group_by``/``group_key_map`` default to ``None``
    (resolved to ``"tile_prefix"`` when neither a manifest nor a value was given) so a value passed
    beside a manifest is detectable and refuses, naming both: the manifest's own grouping policy
    governs the locked draw instead. The identity (``dh``, the lock, the evidence's
    ``split_identity_hash``) is ``dataset_hash(labels_dir, stems=universe)`` rather than the whole
    directory's hash, so a manifest draw never addresses the lock a whole-directory draw locked,
    and the evidence records the swept universe under ``label_stems.calibration`` (with
    ``stated_values.split_manifest_dir``) instead of the whole directory under
    ``label_dirs.calibration``. ``evidence`` also carries ``calibration_stems`` (the swept stem
    list, every calibration) and, under a manifest, ``excluded``
    (``calibration_universe_from_manifest``'s own ``excluded_training_stems``/
    ``excluded_unassigned_stems``).
    """
    from tcip_annotation.json_io import require_reference_ground_truth
    from tcip_mcp.pipelines.data.label_queries import json_det_targets, resolve_registry_id_map
    from tcip_mcp.pipelines.data.splits import (
        cal_holdout_scope_root, count_label_lines, label_image_stems, manifest_date_key,
        resolve_locked_cal_holdout_split, same_directory,
    )
    from tcip_mcp.pipelines.operating_point import (
        attach_split_policy_provenance, derive_max_dets_from_counts, resolve_operating_point,
        set_detector_operating_point,
    )
    from tcip_mcp.pipelines.resolution import dataset_hash
    from tcip_mcp.pipelines.training.evaluation import build_coco_image_record
    from tcip_mcp.tools.inference_tools import _recorded_training_id_map

    from tcip_mcp.dataset_layout import annotation_date

    labels_p = Path(labels_dir)
    require_reference_ground_truth(labels_p)
    cal_date = annotation_date(labels_dir)
    if split_manifest_dir is not None and (group_by is not None or group_key_map is not None):
        raise ValueError(
            f"split_manifest_dir={split_manifest_dir!r} conflicts with group_by/group_key_map: "
            "the manifest's own grouping policy governs the locked draw; pass neither beside it."
        )
    # The run's subject + single id map (from predictor.config): calibration GT reads through the
    # same loader-side reader the training targets use, so the swept count can't diverge from training.
    _data_cfg = (getattr(predictor, "config", {}) or {}).get("data") or {}
    _subject, _attribute = _data_cfg.get("subject"), _data_cfg.get("attribute")
    _checkpoint_manifest_dir = (
        (_data_cfg.get("split") or {}).get("manifest_binding") or {}).get("manifest_dir")
    if (split_manifest_dir is not None and _checkpoint_manifest_dir is not None
            and not same_directory(_checkpoint_manifest_dir, split_manifest_dir)):
        raise ValueError(
            f"this checkpoint is bound to split manifest {_checkpoint_manifest_dir!r}, not the "
            f"{split_manifest_dir!r} this calibration names: calibrating a bound checkpoint "
            "under a different manifest would check its selection disjointness against a side "
            "the checkpoint was never trained or chosen with."
        )
    # Prefers the training run's own recorded map over a fresh registry read: the model only
    # speaks its training vocabulary, so an edited classes.json must not silently relabel the GT.
    _cal_id_map = None
    if _subject:
        _cal_id_map = _recorded_training_id_map(predictor)
        if _cal_id_map is None:
            # No try/except: resolve_registry_id_map's only exception is its own deliberate
            # ValueError, which must reach the caller rather than degrade to a single-class read.
            _reg, _cal_id_map = resolve_registry_id_map(labels_dir, _subject, _attribute)
    # The shared labels-intersect-images scan redraw_calibration_holdout also uses: a stem
    # whose image was deleted/renamed never enters the split universe here.
    stems, stem_to_image = label_image_stems(labels_dir, images_dir)
    excluded = None
    split_manifest_sha256 = None
    if split_manifest_dir is not None:
        if not images_dir:
            raise ValueError(
                "split_manifest_dir requires calibration_images_dir (or images_dir): a "
                "labels-only universe can include a stem whose image is gone, a lock the redraw "
                "would address that no manifest-restricted calibration ever draws."
            )
        from tcip_mcp.pipelines.data.splits import resolve_manifest_calibration_universe
        from tcip_mcp.pipelines.resolution import manifest_digest
        from tcip_mcp.tools.data_tools import read_split_manifest_dir

        manifest = read_split_manifest_dir(split_manifest_dir)
        split_manifest_sha256 = manifest_digest(manifest)
        stems, group_by, group_key_map, excluded, cal_date, _subject, _attribute = \
            resolve_manifest_calibration_universe(
                manifest, split_manifest_dir, labels_dir, images_dir, _subject, _attribute, stems)
        stem_to_image = {s: stem_to_image[s] for s in stems}
    else:
        group_by = group_by or "tile_prefix"
    dh = dataset_hash(labels_dir, stems=(stems if split_manifest_dir is not None else None))
    annotation_counts = {
        s: count_label_lines(labels_dir, s, subject=_subject, attribute=_attribute)
        for s in stems
    }
    # Detector-cap censoring: derive the collection-pass cap from this split's own density (same
    # formula tcip calibrate-operating-point uses), not the caller's possibly-unrelated max_dets.
    density_cap = derive_max_dets_from_counts(list(annotation_counts.values()))
    locked = resolve_locked_cal_holdout_split(
        stems, identity_hash=dh, scope_root=cal_holdout_scope_root(labels_dir),
        annotation_counts=annotation_counts,
        group_by=group_by, group_key_map=group_key_map, seed=seed, holdout_ratio=holdout_ratio,
        split_manifest_dir=split_manifest_dir,
    )
    if locked.get("unlocked_stems"):
        logger.info(
            "cal/holdout split for %s has %d stem(s) not covered by the existing lock (new since "
            "it was drawn); excluded from this calibration: %s", dh,
            len(locked["unlocked_stems"]), locked["unlocked_stems"][:10],
        )
    cal_stems, hold_stems = locked["calibration"], locked["holdout"]

    # Floor the in-model + predictor conf so hesitant detections survive to be swept; the applied
    # score_thresh (not a re-typed 0.01 literal) becomes resolve_operating_point's staged_conf_floor.
    applied, applied_attribute_path = set_detector_operating_point(
        predictor.model, score_thresh=0.01, detections_per_img=density_cap)
    predictor.score_threshold = applied.get("score_thresh", 0.01)

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
            dt = [{"category_id": int(lab), "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(sc)}
                  for (x1, y1, x2, y2), sc, lab in zip(r["boxes"], r["scores"], r["labels"])]
            # GT lifted to the predictor's 1-indexed labels via the loader-side reader (subject +
            # id map); with no run subject in scope, fall back to a single-class read of every box.
            gt_path = str(labels_p / f"{s}.json")
            if _subject and _cal_id_map is not None:
                gboxes, glabels, n_unlabeled = json_det_targets(gt_path, _subject, _attribute, _cal_id_map)
                # An image with any instance unlabeled for `attribute` is dropped whole from the
                # record set (the missing-label-file precedent), counted rather than silently filtered.
                if n_unlabeled:
                    n_excluded_incomplete_attribute += 1
                    continue
                gt = [{"category_id": int(lab), "bbox": [x1, y1, x2 - x1, y2 - y1], "iscrowd": 0}
                      for (x1, y1, x2, y2), lab in zip(gboxes, glabels)]
            else:
                from tcip_annotation import json_io
                from tcip_annotation.state import Point, bbox_of
                gt = []
                for a in json_io.read_annotations(gt_path):
                    if a.geometry is None or isinstance(a.geometry, Point):
                        continue
                    bx = bbox_of(a.geometry)
                    gt.append({"category_id": 1,
                               "bbox": [bx.x1, bx.y1, bx.x2 - bx.x1, bx.y2 - bx.y1], "iscrowd": 0})
            recs.append(build_coco_image_record(int(r["width"]), int(r["height"]), gt, dt, image_id=s))
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
        "split_manifest_dir": split_manifest_dir, "calibration_date": manifest_date_key(cal_date),
        "calibration_labels_dir": str(labels_p), "split_manifest_sha256": split_manifest_sha256,
    }
    bundle = resolve_operating_point(trait, experiment_id=experiment_id, **resolver_inputs)
    attach_split_policy_provenance(bundle, locked)
    drawn = (locked.get("redraw_history") or [{}])[-1]
    stated_values = {"split_identity_hash": dh, "split_content_hash": drawn.get("new_content_hash")}
    if split_manifest_dir is not None:
        stated_values["split_manifest_dir"] = split_manifest_dir
        reference_inputs = {
            "label_stems": {"calibration": {"path": str(labels_p), "stems": stems}},
            "stated_values": stated_values,
        }
    else:
        reference_inputs = {
            "label_dirs": {"calibration": str(labels_p)}, "stated_values": stated_values,
        }
    evidence = {
        "resolver": "resolve_operating_point", "inputs": resolver_inputs,
        "reference_inputs": reference_inputs, "calibration_stems": stems,
    }
    if excluded is not None:
        evidence["excluded"] = excluded
    return bundle, dh, n_excluded_incomplete_attribute, evidence


def gate_evidence_summary(conf_param) -> dict:
    """Compact, response-safe view of a calibration's gate evidence (the full curve is written to disk).

    Includes ``disjoint``/``content_overlap_frac``/``train_disjointness``/
    ``selection_disjointness`` so the calling agent sees the real reason a calibration refused
    validation, not only the pass/fail booleans, a refusal from train-provenance, selection-
    provenance or content-overlap would otherwise look identical to a plain holdout-bias
    failure. Also includes ``split_policy_divergence``/``split_unlocked_stems`` (via
    ``attach_split_policy_provenance``) so a caller whose declared ``seed``/``holdout_ratio``/
    ``group_by`` didn't take effect against an existing lock sees that here, in the tool's own
    response, rather than only in the persisted curve artifact or a server log line.

    ``failures`` is the same named-failure list ``describe_review_validation`` reads for the
    breeder-facing message, surfaced here too, plus the individual gate fields
    (``conf_floor_mismatch``, dispersion/localization terms), so an agent hitting a refusal
    condition on the GT path sees a real reason instead of every other field reading "fine".
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
