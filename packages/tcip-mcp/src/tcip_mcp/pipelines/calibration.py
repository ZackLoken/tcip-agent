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


def pass_resolver_inputs(p) -> dict:
    """The execution regime a prepared pass ``p`` collects under, as ``resolve_operating_point``
    takes it: the slicing record, the merge threshold's provenance, the tile edge with its source
    and derivation, and the stated cap (``None`` for one the calibration derives)."""
    return {
        "slicing": p.slicing, "cross_tile_nms": p.cross_tile_nms.to_provenance(),
        "tile_size": p.geometry.tile_size, "tile_size_source": p.geometry.tile_size_source,
        "tile_size_derived_from": p.geometry.tile_size_derived_from,
        "max_dets": p.max_dets if p.max_dets_stated else None,
    }


def resolve_pass_merge(p, calibration_gt: list[list[dict]]) -> None:
    """Set on ``p`` the merge threshold it collects and exports at: its stated one, else the one
    the calibration side's ground truth derives (``calibration_gt``, evaluation GT records per
    image; :func:`~tcip_mcp.pipelines.resolution.resolve_cross_tile_nms`)."""
    from tcip_mcp.pipelines.resolution import resolve_cross_tile_nms
    from tcip_mcp.pipelines.training.evaluation import gt_objects

    if p.cross_tile_nms.source != "explicit":
        p.cross_tile_nms = resolve_cross_tile_nms(None, p.slicing, [
            [a["bbox"] for a in gt_objects({"gt": gt})] for gt in calibration_gt])


def collect_calibration_records(p, cal_stems, hold_stems, source_of, gt_of):
    """``p``'s predictions over the calibration and holdout stems, each paired with its ground
    truth (``gt_of``, evaluation GT records per stem) into one COCO-shaped record, ``source_of``
    naming each stem's image, both collected at the merge threshold :func:`resolve_pass_merge`
    sets first. Returns ``(calibration_records, holdout_records)``."""
    from tcip_mcp.pipelines.training.evaluation import build_coco_image_record, detection_record

    resolve_pass_merge(p, [gt_of[s] for s in cal_stems])

    def records(stems):
        results = p.predict([source_of[s] for s in stems]) if stems else []
        return [build_coco_image_record(
            int(r["width"]), int(r["height"]), gt_of[s],
            [detection_record(b, lab, sc) for b, sc, lab in zip(r["boxes"], r["scores"], r["labels"])],
            image_id=s) for s, r in zip(stems, results)]

    return records(cal_stems), records(hold_stems)


def calibrate_operating_point(p, trait, labels_dir, images_dir, *, project: Path,
                               group_by=None, group_key_map=None, experiment_id=None,
                               seed=DEFAULT_CAL_SEED, holdout_ratio=DEFAULT_HOLDOUT_RATIO,
                               selection_dir=None):
    """Resolve a per-dataset operating point from a labeled split, over the prepared pass ``p``
    (``inference_tools._PreparedPass``) a delivery runs, for ``trait`` of ``project``, whose run
    ``experiment_id`` names.

    Returns ``(bundle, hash, n_excluded_incomplete_attribute, evidence)``. The third value is the
    count of cal/holdout stems dropped whole because an instance was unlabeled for ``attribute``;
    it is not a field on the bundle. ``evidence`` is the name of the resolver this ran, the
    arguments it ran it over (the dict passed to ``resolve_operating_point``, without the trait and
    the producing run), and the locations of the reference those arguments came from:
    ``label_dirs.calibration`` for a whole-directory draw, ``label_stems.calibration`` with
    ``stated_values.selection_dir`` under a selection, plus ``calibration_stems`` and, under a
    selection, ``excluded``.

    The count-unbiased center-match curve and held-out bias check run ``p`` at a floor conf
    (:func:`collect_calibration_records`) over a disjoint cal/holdout split of the labeled dir
    locked on its first draw (``resolve_locked_cal_holdout_split``); ``seed``/``holdout_ratio``
    only take effect on that locking draw. The lock is scoped to the labeled dir's own root
    (``cal_holdout_scope_root``), which ``redraw_calibration_holdout`` states to address it.

    Raises ``ValueError`` when the lock references a stem whose image/label no longer exists or its
    lock file is corrupt, and through ``json_io.require_reference_ground_truth`` when
    ``labels_dir`` is not an admissible reference. Ground truth is read under the run's recorded
    class space (``p.scope``).

    ``selection_dir`` restricts the calibration universe to the ``calibration`` samples of a
    selection whose label documents live under ``labels_dir``, re-admitted under ``p.scope``. A
    checkpoint whose producing run (``experiment_id``) bound a different selection is refused by
    name, and so is
    ``group_by``/``group_key_map`` passed beside a selection. Without a selection,
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
    from tcip_mcp.pipelines.training.evaluation import gt_records

    from tcip_mcp.pipelines.data.selection import DOCUMENT

    predictor = p.predictor
    # The run's own recorded class space: calibration GT reads under the same scope the training
    # targets did, so the swept count cannot diverge from it.
    scope = p.scope.admitted_for(DOCUMENT, f"the run calibrating against {labels_dir}")
    labels_p = Path(labels_dir)
    require_reference_ground_truth(labels_p)
    policy_conflict = selection_policy_conflict(selection_dir, group_by, group_key_map)
    if policy_conflict:
        raise ValueError(policy_conflict)
    if selection_dir is not None and experiment_id is not None:
        from tcip_mcp.experiments import run_resolution

        binding = run_resolution(experiment_id, project=project)["partition"]["selection"]
        if binding is not None and not same_directory(binding["selection_dir"], selection_dir):
            raise ValueError(
                f"this checkpoint is bound to the selection at {binding['selection_dir']!r}, not "
                f"the {selection_dir!r} this calibration names: calibrating a bound checkpoint "
                "under a different selection would check its selection disjointness against a "
                "side the checkpoint was never trained or chosen with."
            )
    excluded = None
    selection_sha256 = None
    if selection_dir is not None:
        from tcip_mcp.pipelines.data.selection import read_selection
        from tcip_mcp.pipelines.data.splits import selection_calibration_universe
        from tcip_mcp.pipelines.image_utils import resolve_source_path
        from tcip_mcp.pipelines.resolution import selection_digest

        selection = read_selection(selection_dir, project=project)
        selection_sha256 = selection_digest(selection, project)
        stems, group_by, group_key_map, excluded, annotation_counts, universe_samples = \
            selection_calibration_universe(selection, labels_dir, scope)
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
    dh = dataset_hash(labels_dir, stems=(stems if selection_dir is not None else None))
    # The whole-directory universe holds names, and counts each member's document by the path it
    # composed; a selection's universe returned its own counts above.
    if selection_dir is None:
        annotation_counts = {s: count_label_lines(gt_path_of[s], scope) for s in stems}
    # Detector-cap censoring: derive the collection-pass cap from this split's own density (same
    # formula tcip calibrate-operating-point uses), not the caller's possibly-unrelated max_dets.
    density_cap = derive_max_dets_from_counts(list(annotation_counts.values()))
    locked = resolve_locked_cal_holdout_split(
        stems, identity_hash=dh, scope_root=cal_holdout_scope_root(labels_dir),
        annotation_counts=annotation_counts,
        group_by=group_by, group_key_map=group_key_map, seed=seed, holdout_ratio=holdout_ratio,
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

    # GT read the way the run's training targets were; a stem with an unlabeled instance is counted.
    gt_of = {}
    for s in cal_stems + hold_stems:
        target, n_unlabeled = json_det_targets(gt_path_of[s], scope)
        if not n_unlabeled:
            gt_of[s] = gt_records(target)
    n_excluded_incomplete_attribute = len(cal_stems) + len(hold_stems) - len(gt_of)
    cal_records, hold_records = collect_calibration_records(
        p, [s for s in cal_stems if s in gt_of], [s for s in hold_stems if s in gt_of],
        stem_to_image, gt_of)
    resolver_inputs = {
        **pass_resolver_inputs(p),
        "dataset_hash": dh, "calibration_records": cal_records,
        "holdout_records": hold_records or None,
        "staged_conf_floor": applied.get("score_thresh"),
        "staged_conf_floor_attribute_path": applied_attribute_path,
        "selection_dir": selection_dir,
        "calibration_labels_dir": str(labels_p), "selection_sha256": selection_sha256,
    }
    bundle = resolve_operating_point(trait, project=project, experiment_id=experiment_id,
                                     **resolver_inputs)
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
