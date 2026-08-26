"""Inference MCP tools: run models on images, export results."""

from __future__ import annotations

import logging
import os
from pathlib import Path, PurePosixPath

from tcip_store import RECORD_JSON, BadKey, Key, StoreDescriptor, register_store

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited
from tcip_mcp.pipelines.postprocessing.export import (
    export_detection_csv,
    mask_binarize_provenance,
    positive_detections,
    write_predictions_json,
)
from tcip_mcp.pipelines.resolution import (
    DEFAULT_CONF,
    DEFAULT_MAX_DETS,
    DEFAULT_NMS_IOU,
)
from tcip_mcp.project_paths import resolve_output_path

logger = logging.getLogger(__name__)

_SWEEP_DIR = (".tcip", "artifacts")
_SWEEP_STEM = "operating_point_sweep_"


class _SweepArtifactLocator:
    """One conf sweep per set of inputs, named for the digest of the inputs it was swept over.

    The digest is in the filename rather than in a directory of its own, the same convention
    the locked calibration/holdout split beside it uses.
    """

    def relative_path(self, scope: str, parts: tuple[str, ...]) -> PurePosixPath:
        (body_hash,) = parts
        return PurePosixPath(*_SWEEP_DIR, f"{_SWEEP_STEM}{body_hash}.json")

    def parts_from(self, relative_path: PurePosixPath) -> tuple[str, ...] | None:
        segments = relative_path.parts
        if segments[:len(_SWEEP_DIR)] != _SWEEP_DIR or len(segments) != len(_SWEEP_DIR) + 1:
            return None
        name = segments[-1]
        if not name.startswith(_SWEEP_STEM) or not name.endswith(".json"):
            return None
        return (name[len(_SWEEP_STEM):-len(".json")],)


CONFIDENCE_SWEEP_STORE = "confidence_sweep"
register_store(
    StoreDescriptor(
        name=CONFIDENCE_SWEEP_STORE,
        kind="record",
        key_fields=("inputs_hash",),
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        locator=_SweepArtifactLocator(),
    )
)


def confidence_sweep_key(inputs_hash: str) -> Key:
    """The full confidence sweep one calibration produced.

    ``last_writer_wins``: the name is the digest of every input the curve depends on, so the
    same name is the same curve and a rerun writes what is already there.
    """
    if PurePosixPath(inputs_hash).name != inputs_hash or inputs_hash in ("", ".", ".."):
        raise BadKey(
            f"sweep identity {inputs_hash!r} is not a single name: an identity carrying a path "
            "separator would address a record outside the artifact store"
        )
    from tcip_mcp.project_paths import project_root

    return Key(CONFIDENCE_SWEEP_STORE, str(project_root().resolve()), (inputs_hash,))


def confidence_sweep_path(inputs_hash: str) -> Path:
    """Where that sweep lands on disk, for the provenance that names the file it was kept in."""
    from tcip_mcp.project_paths import project_root

    key = confidence_sweep_key(inputs_hash)
    return project_root().joinpath(
        *_SweepArtifactLocator().relative_path(key.root, key.parts).parts
    )


def _recorded_training_id_map(predictor) -> dict | None:
    """The training run's own recorded name->id map (``config["data"]["id_map"]``), or ``None``
    when the checkpoint carries none. The one read :func:`resolve_decode_id_map` and calibration's
    own GT-side id-map resolution (:func:`_calibrate_operating_point`) both share, so a checkpoint's
    recorded vocabulary is preferred identically wherever an id_map is derived for it, never
    re-checked independently by each caller."""
    data_cfg = (getattr(predictor, "config", {}) or {}).get("data") or {}
    recorded = data_cfg.get("id_map")
    if isinstance(recorded, dict) and recorded:
        return {str(k): int(v) for k, v in recorded.items()}
    return None


def resolve_decode_id_map(predictor, images_dir: str | None, *,
                          scope: tuple[str | None, str | None] | None = None) -> dict | None:
    """This run's name->id map for recording + decoding predictions.

    The one resolution every door that writes predictions to disk calls, this tool's own
    ``run_inference`` and the web GUI's inference worker (``tcip_web.routes.inference``), never a
    second implementation (CLAUDE.md: "when two code paths must agree, call one from the other").

    Prefers the *training* run's own recorded map (stamped onto ``config["data"]["id_map"]`` by
    ``subprocess_worker.py::run`` right after the dataset is built, so it travels on the checkpoint
    the same way ``subject``/``attribute`` already do) over re-deriving one from the inference
    dataset's live registry, the model can only speak the vocabulary it was trained on, so the
    training map is the correct decode map by definition, and it is immune to a ``classes.json``
    whose declared attribute-value order was edited after training. A checkpoint with no recorded
    map (a bespoke ``dataset_source`` with no registry scope, or a run trained from a pre-built COCO
    source whose id space isn't registry-derived, ``_resolve_run_id_map`` deliberately does not
    record one for either) falls through to the live-registry derivation, the same honest,
    order-invariant-for-single-class degraded path this already was.

    A registry read that fails for a real reason (corrupted file, an id-space mismatch) propagates
    loudly from here, but ``run_inference`` lets that reach its own caller, while the GUI worker
    (``routes/inference.py``) wraps this whole call in a broad except and degrades to ``id_map=None``
    on any failure; the two doors share this one resolution but choose different failure postures on
    top of it, not two different resolutions.

    ``scope`` is the ``(subject, attribute)`` the registry fallback derives against, defaulting to
    the predictor's own recorded training scope. A caller holding the run's scope from elsewhere
    (block calibration reads it from the training experiment's ``config.json``, and refuses without
    it) passes it here rather than restating the prefer-recorded-else-derive rule around its own.
    """
    recorded = _recorded_training_id_map(predictor)
    if recorded is not None:
        return recorded
    data_cfg = (getattr(predictor, "config", {}) or {}).get("data") or {}
    subject, attribute = scope if scope is not None else (
        data_cfg.get("subject"), data_cfg.get("attribute"))
    if not (subject and images_dir):
        return None

    from tcip_mcp.pipelines.data.datasets import _resolve_registry_id_map, resolved_classes_path

    # Precondition check, not a broad except: attribute-scoped decode with no classes.json for
    # this dataset is a legitimate, honest degraded case write_predictions_json already documents
    # and accepts ("the raw 0-indexed id is used as the name... never a re-derivation"), id_map
    # stays None rather than crashing on already-completed, valid prediction work. Reuses
    # resolved_classes_path, the same "does a registry exist" check _resolve_registry_id_map's
    # own refusal is built on, not a second independent re-derivation of that fact. A registry
    # that is present but fails to read/resolve for a real reason (corrupted file, an id-space
    # mismatch) still propagates loudly, this precondition only short-circuits the one case
    # that's supposed to degrade honestly, the same case _resolve_registry_id_map's own
    # attribute-without-registry ValueError names. Not the same shape as model_build.py's
    # resolve_contract_dims precondition (that one gates on subject alone and lets the
    # attribute-without-registry refusal reach the caller); this site's downstream consumer has
    # its own documented accepted-degradation contract that resolve_contract_dims's caller does
    # not.
    if attribute is not None and resolved_classes_path(images_dir) is None:
        return None
    _reg, id_map = _resolve_registry_id_map(images_dir, subject, attribute)
    return id_map


def _calibrate_operating_point(predictor, trait, labels_dir, images_dir, *,
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
    ``export_predictions`` writes.

    ``evidence`` is what a delivery door needs to earn the validation record behind a validated
    count: the name of the resolver this function just ran, the arguments it ran it over, and the
    locations of the reference those arguments came from. The arguments are the same dict passed to
    ``resolve_operating_point`` here, never a second assembly of them, so the gate a door reopens is
    the gate this calibration passed. The trait and the producing run are left out: they are
    ``open_validation``'s own arguments, and a second spelling of either could disagree with the
    record it is written into.

    The count-unbiased center-match sweep + held-out bias check run the same predictor path the
    delivery will use (same tile/tile_size/tile_resize/overlap/nms/postprocess) over a disjoint, locked
    cal/holdout split of the labeled dir (``resolve_locked_cal_holdout_split``: group-coherent,
    seeded, and stable across calls, not a fresh lexicographic cut every time), at a floor conf so
    hesitant detections survive to be swept, so the resolved conf is validated in the regime it
    ships through, not an untiled full-frame model pass. ``seed``/``holdout_ratio`` only take effect
    on the first (locking) draw for this labeled dir's identity hash. That lock is scoped to the
    labeled dir's own root (``cal_holdout_scope_root``), so it is still the same lock after a
    project adoption repins the platform root, and ``force_redraw_cal_holdout_split`` addresses it
    by stating that root.

    Raises ``ValueError`` (propagated from ``resolve_locked_cal_holdout_split``) when the lock
    references a stem whose image/label no longer exists, or its lock file is corrupt, the caller
    (``run_inference``) turns this into a clean ``{"error": ...}`` rather than letting a bare
    ``KeyError`` surface from a stale ``stem_to_image`` lookup.

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

    ``split_manifest_dir`` restricts the calibration universe to one capture date's held-out side
    of a ``split_manifest`` record (``data_tools.read_split_manifest_dir``) instead of every
    labelled stem with an image: the manifest's ``subject``/``attribute`` must equal this run's
    recorded training scope, the labels directory's date (``dataset_layout.annotation_date``) must
    be one the manifest holds members under, and the manifest's ``images_root`` for that date must
    be ``images_dir``, each refusing by name. ``group_by``/``group_key_map`` default to ``None``
    (resolved to ``"tile_prefix"`` when neither a manifest nor a value was given) so a value passed
    beside a manifest is detectable and refuses, naming both: the manifest's own grouping policy
    governs the locked draw instead. The identity (``dh``, the lock, the evidence's
    ``split_identity_hash``) is ``dataset_hash(labels_dir, stems=universe)`` rather than the whole
    directory's hash, so a manifest draw never addresses the lock a whole-directory draw locked,
    and the evidence records the swept universe under ``label_stems.calibration`` (with
    ``stated_values.split_manifest_dir``) instead of the whole directory under
    ``label_dirs.calibration``.
    """
    from tcip_annotation.json_io import require_reference_ground_truth
    from tcip_mcp.pipelines.data.datasets import _json_det_targets, _resolve_registry_id_map
    from tcip_mcp.pipelines.data.splits import (
        cal_holdout_scope_root, count_label_lines, label_image_stems,
        resolve_locked_cal_holdout_split,
    )
    from tcip_mcp.pipelines.operating_point import (
        attach_split_policy_provenance, derive_max_dets_from_counts, resolve_operating_point,
        set_detector_operating_point,
    )
    from tcip_mcp.pipelines.resolution import dataset_hash
    from tcip_mcp.pipelines.training.evaluation import build_coco_image_record

    labels_p = Path(labels_dir)
    require_reference_ground_truth(labels_p)
    if split_manifest_dir is not None and (group_by is not None or group_key_map is not None):
        raise ValueError(
            f"split_manifest_dir={split_manifest_dir!r} conflicts with group_by/group_key_map: "
            "the manifest's own grouping policy governs the locked draw; pass neither beside it."
        )
    # The run's subject + single id map (from predictor.config): calibration GT reads through the
    # same loader-side reader the training targets use, so the swept count can't diverge from training.
    _data_cfg = (getattr(predictor, "config", {}) or {}).get("data") or {}
    _subject, _attribute = _data_cfg.get("subject"), _data_cfg.get("attribute")
    # Prefers the training run's own recorded map over a fresh registry read, the same preference
    # resolve_decode_id_map applies to decode: the model can only speak the vocabulary it was
    # trained on, so a classes.json whose declared attribute-value order was edited since training
    # must not silently relabel the calibration GT. No try/except on the registry fallback:
    # _resolve_registry_id_map's only exception is its own deliberate ValueError refusal, which must
    # reach the caller on the calibration rail rather than silently degrade to a single-class GT
    # read. Its one legitimate "no registry, that's fine" case (single-class, no attribute) already
    # returns normally without raising.
    _cal_id_map = None
    if _subject:
        _cal_id_map = _recorded_training_id_map(predictor)
        if _cal_id_map is None:
            _reg, _cal_id_map = _resolve_registry_id_map(labels_dir, _subject, _attribute)
    # Labels-intersect-images-on-disk: the shared scan force_redraw_cal_holdout_split
    # now also uses, so the two paths can't disagree about which stems exist. A stem whose image was
    # deleted/renamed never even enters the split universe here.
    stems, stem_to_image = label_image_stems(labels_dir, images_dir)
    excluded = None
    if split_manifest_dir is not None:
        from tcip_mcp.dataset_layout import annotation_date
        from tcip_mcp.pipelines.data.splits import (
            calibration_universe_from_manifest, manifest_date_key,
        )
        from tcip_mcp.tools.data_tools import read_split_manifest_dir

        manifest = read_split_manifest_dir(split_manifest_dir)
        if (manifest.get("subject"), manifest.get("attribute")) != (_subject, _attribute):
            raise ValueError(
                f"split manifest at {split_manifest_dir!r} was drawn for subject="
                f"{manifest.get('subject')!r}, attribute={manifest.get('attribute')!r}, but this "
                f"run's recorded training scope is subject={_subject!r}, attribute={_attribute!r}."
            )
        cal_date = annotation_date(labels_dir)
        date_block = (manifest.get("members") or {}).get(manifest_date_key(cal_date))
        if date_block is None:
            raise ValueError(
                f"split manifest at {split_manifest_dir!r} holds no members under date "
                f"{cal_date!r}; it holds members under {sorted(manifest.get('members') or {})}."
            )
        manifest_images_root = date_block.get("images_root")
        if images_dir and manifest_images_root and not os.path.samefile(
                Path(images_dir).resolve(), Path(manifest_images_root).resolve()):
            raise ValueError(
                f"images_dir={images_dir!r} is not the split manifest's images_root for date "
                f"{cal_date!r} ({manifest_images_root!r})."
            )
        stems, group_by, group_key_map, excluded = calibration_universe_from_manifest(
            manifest, cal_date, stems)
        stem_to_image = {s: stem_to_image[s] for s in stems}
    else:
        group_by = group_by or "tile_prefix"
    dh = dataset_hash(labels_dir, stems=(stems if split_manifest_dir is not None else None))
    annotation_counts = {s: count_label_lines(labels_dir, s) for s in stems}
    # Detector-cap censoring: a flat max_dets can still truncate a dense calibration image's raw
    # detections the same way a too-high conf floor censors them, so derive the collection-pass cap
    # from this labeled split's own density (same formula scripts/calibrate_operating_point.py uses)
    # rather than measuring the sweep against the caller's (possibly unrelated) max_dets.
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

    # Floor the in-model + predictor conf so hesitant detections survive to be swept, and raise the
    # cap to this split's own density (derived above). The applied score_thresh (not a re-typed 0.01
    # literal) is threaded into resolve_operating_point as the reference's staged_conf_floor, the
    # same value this call actually applied, per CLAUDE.md's "when two paths must agree, call one
    # from the other."
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
                gboxes, glabels, n_unlabeled = _json_det_targets(gt_path, _subject, _attribute, _cal_id_map)
                # An image with any instance unlabeled for `attribute` has
                # incomplete GT for this scope, excluded from the calibration/holdout record set
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


def _sweep_summary(conf_param) -> dict:
    """Compact, response-safe view of a calibration sweep (the full curve is written to disk).

    Includes ``disjoint``/``content_overlap_frac``/``train_disjointness`` so the calling agent
    sees the real reason a calibration refused validation, not only the pass/fail booleans, a
    refusal from train-provenance or content-overlap would otherwise look identical to a plain
    holdout-bias failure. Also includes ``split_policy_divergence``/``split_unlocked_stems`` (via
    ``attach_split_policy_provenance``) so a caller whose declared ``seed``/``holdout_ratio``/
    ``group_by`` didn't take effect against an existing lock sees that here, in the tool's own
    response, rather than only in the persisted sweep artifact or a server log line.

    ``failures`` is the same named-failure list ``describe_review_validation`` reads for the
    breeder-facing message, surfaced here too, plus the individual gate fields
    (``conf_floor_mismatch``, dispersion/localization terms), so an agent hitting a refusal
    condition on the GT path sees a real reason instead of every other field reading "fine".
    """
    sweep = conf_param.sweep or {}
    hb = sweep.get("holdout_bias") or {}
    return {
        "count_unbiased_conf": conf_param.unvalidated_value(acknowledge_unvalidated=True),
        "f1_max_conf": sweep.get("f1_max_conf"),
        "holdout_bias": hb.get("count_bias_mean") if isinstance(hb, dict) else None,
        # The pooled bias above is the one number a class-compensating refusal reads "fine" on, so
        # the per-class biases the gate actually judged travel beside it.
        "per_class_holdout_bias": {cid: s["count_bias_mean"]
                                   for cid, s in (hb.get("per_class") or {}).items()},
        "per_class_count_bias_failures": sweep.get("per_class_count_bias_failures"),
        "per_class_insufficient_images": sweep.get("per_class_insufficient_images"),
        "holdout_missing_classes": sweep.get("holdout_missing_classes"),
        "passed_holdout": sweep.get("passed_holdout"),
        "failures": sweep.get("failures"),
        "conf_censored": sweep.get("conf_censored"),
        "conf_floor_mismatch": sweep.get("conf_floor_mismatch"),
        "count_bias_tolerance_frac": sweep.get("count_bias_tolerance_frac"),
        "pooled_count_bias_tolerance": sweep.get("pooled_count_bias_tolerance"),
        # A per-class refusal is unexplainable from this compact view without the actual bar each
        # class was held to, the pooled tolerance alone (above) is not it, since each class
        # scales against its own typical count.
        "per_class_count_bias_tolerance": sweep.get("per_class_count_bias_tolerance"),
        "pooled_typical_count": sweep.get("pooled_typical_count"),
        "per_class_typical_count": sweep.get("per_class_typical_count"),
        "count_error_tolerance": sweep.get("count_error_tolerance"),
        "count_error_p90": hb.get("count_error_p90") if isinstance(hb, dict) else None,
        "disjoint": sweep.get("disjoint"),
        "content_overlap_frac": sweep.get("content_overlap_frac"),
        "content_shared_with_calibration": sweep.get("content_shared_with_calibration"),
        "train_disjointness": sweep.get("train_disjointness"),
        "split_policy_divergence": sweep.get("split_policy_divergence"),
        "split_unlocked_stems": sweep.get("split_unlocked_stems"),
    }


@mcp.tool()
@audited(scope_arg="dataset_root")
def force_redraw_cal_holdout_split(
    dataset_root: str,
    labels_dir: str | None = None,
    images_dir: str | None = None,
    identity_hash: str | None = None,
    group_by: str | None = None,
    group_key_map: dict[str, str] | None = None,
    seed: int = 0,
    holdout_ratio: float = 0.5,
    reason: str = "",
    split_manifest_dir: str | None = None,
    subject: str | None = None,
    attribute: str | None = None,
) -> dict:
    """Deliberately redraw a locked calibration/holdout split.

    A cal/holdout split locks on its first draw (``resolve_locked_cal_holdout_split``) so the
    "held-out validation" gate can never silently pass on a different, weaker holdout drawn
    after the fact. Redrawing one is a real, audited decision, never automatic, never a hidden
    kwarg on a high-traffic tool like ``run_inference``, so it is its own small tool. ``reason``
    is required and non-empty, and every redraw (this one included) is appended to the lock's
    ``redraw_history`` with its policy, seed, and the old and new split's content hashes, so a
    redraw-until-it-passes pattern is visible on review even though nothing here enforces that a
    reason differ from a prior one; the old and new split membership itself is recorded in the
    audit log alongside the reason, not in ``redraw_history``; the defense is a reviewable audit
    trail, not an automatic block.

    Provide either ``labels_dir`` (the identity is derived as ``dataset_hash(labels_dir)``, and
    its stems are re-scanned) or ``identity_hash`` directly (e.g. a review-reference hash, in
    that case the existing lock's own calibration+holdout stems are reused as the redraw's stem
    universe, since a review reference has no labels directory to re-scan).

    Args:
        dataset_root: The root the lock is stored under, required, no default: a locked split
            travels with the data it was drawn over, and this tool holds an identity hash rather
            than anything the root can be read off. With ``labels_dir`` given, it is the root that
            dir's own lock lives under (its dataset root, or the dir itself when the layout places
            it under none), and a root disagreeing with it refuses rather than redrawing a lock
            nothing reads.
        labels_dir: Labeled dir whose GT identity locked the split (mutually exclusive with
            ``identity_hash``, if both are omitted, or ``identity_hash`` is given with no
            existing lock and no ``labels_dir``, this refuses).
        images_dir: Images for ``labels_dir``. When given, stems are the same labels-intersect-
            images-on-disk universe ``run_inference``'s calibration uses, a stem
            whose image was deleted/renamed never enters the redraw's stem universe. Omitted ->
            every labeled stem is used regardless of whether an image still exists for it,
            for a caller that has no images directory to check against.
        identity_hash: The locked split's identity hash directly.
        group_by: New grouping policy, ``"tile_prefix"`` / ``"stem"`` (ignored if
            ``group_key_map`` is given). ``None`` (default) resolves to ``"tile_prefix"`` when
            neither this nor a manifest was given; a value beside ``split_manifest_dir`` conflicts
            with the manifest's own grouping policy and refuses, naming both.
        group_key_map: Explicit ``{stem: group_key}`` map covering every stem, overriding
            ``group_by``. Conflicts with ``split_manifest_dir`` the same way ``group_by`` does.
        seed: New split seed.
        holdout_ratio: New calibration/holdout fraction.
        reason: Required, non-empty justification for this redraw, recorded in the audit log
            alongside the old and new split membership.
        split_manifest_dir: Restrict the redraw's universe to one capture date's held-out side of
            a split manifest (``data_tools.read_split_manifest_dir``), the same restriction
            ``run_inference`` applies, instead of every labelled stem with an image. Requires
            ``labels_dir`` and ``subject``: the manifest's own subject/attribute must equal
            ``subject``/``attribute``, the date ``labels_dir`` is under must be one the manifest
            holds members under, and the manifest's ``images_root`` for that date must be
            ``images_dir``, each refusing by name. The identity is
            ``dataset_hash(labels_dir, stems=universe)`` rather than the whole directory's hash,
            so the redraw addresses the same lock a manifest-restricted calibration locked.
        subject: The object class ``split_manifest_dir``'s admission was drawn for; required
            alongside it.
        attribute: The attribute ``split_manifest_dir``'s admission was scoped to, when it was.
    """
    if not reason or not reason.strip():
        return {"error": "reason is required (a non-empty justification) for a force_redraw"}
    if not labels_dir and not identity_hash:
        return {"error": "provide either labels_dir or identity_hash"}
    if split_manifest_dir is not None:
        if not labels_dir:
            return {"error": "split_manifest_dir requires labels_dir: the universe is drawn "
                             "from the manifest's held-out members under the labels' own date."}
        if not subject:
            return {"error": "split_manifest_dir requires subject: the manifest's own subject "
                             "must be checked against the door's."}
        if group_by is not None or group_key_map is not None:
            return {"error": f"split_manifest_dir={split_manifest_dir!r} conflicts with "
                             "group_by/group_key_map: the manifest's own grouping policy governs "
                             "the redraw; pass neither beside it."}

    from datetime import datetime, timezone

    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_store import DecodeError, store

    from tcip_mcp.audit import dataset_scope_of, record_event
    from tcip_mcp.pipelines.data.splits import (
        cal_holdout_lock_key, cal_holdout_scope_root, count_label_lines, label_image_stems,
        resolve_locked_cal_holdout_split,
    )
    from tcip_mcp.pipelines.resolution import dataset_hash

    manifest_stems: list[str] | None = None
    if split_manifest_dir is not None:
        from tcip_mcp.dataset_layout import annotation_date
        from tcip_mcp.pipelines.data.splits import (
            calibration_universe_from_manifest, manifest_date_key,
        )
        from tcip_mcp.tools.data_tools import read_split_manifest_dir

        manifest = read_split_manifest_dir(split_manifest_dir)
        if (manifest.get("subject"), manifest.get("attribute")) != (subject, attribute):
            return {"error": f"split manifest at {split_manifest_dir!r} was drawn for subject="
                             f"{manifest.get('subject')!r}, attribute="
                             f"{manifest.get('attribute')!r}, but this call states "
                             f"subject={subject!r}, attribute={attribute!r}."}
        cal_date = annotation_date(labels_dir)
        date_block = (manifest.get("members") or {}).get(manifest_date_key(cal_date))
        if date_block is None:
            return {"error": f"split manifest at {split_manifest_dir!r} holds no members under "
                             f"date {cal_date!r}; it holds members under "
                             f"{sorted(manifest.get('members') or {})}."}
        manifest_images_root = date_block.get("images_root")
        if images_dir and manifest_images_root and not os.path.samefile(
                Path(images_dir).resolve(), Path(manifest_images_root).resolve()):
            return {"error": f"images_dir={images_dir!r} is not the split manifest's "
                             f"images_root for date {cal_date!r} ({manifest_images_root!r})."}
        present, _ = label_image_stems(labels_dir, images_dir)
        try:
            manifest_stems, group_by, group_key_map, _excluded = calibration_universe_from_manifest(
                manifest, cal_date, present)
        except ValueError as exc:
            return {"error": str(exc)}

    scope_root = Path(dataset_root).resolve()
    if labels_dir:
        labels_scope = cal_holdout_scope_root(labels_dir)
        if labels_scope != scope_root:
            return {"error": f"labels_dir {labels_dir!r} locks its cal/holdout split under "
                             f"{str(labels_scope)!r}, and dataset_root states {str(scope_root)!r}. "
                             "A redraw under the stated root would replace a lock the calibration "
                             "never reads, so state the root those labels' own lock lives under."}

    if identity_hash is None:
        try:
            identity_hash = dataset_hash(labels_dir, stems=manifest_stems)
        except UnreadableLabelDocument as exc:
            return {"error": str(exc)}

    try:
        old_lock = store.read(cal_holdout_lock_key(identity_hash, scope_root=scope_root),
                              default=None)
    except DecodeError:
        # A redraw is the recovery for a lock whose bytes do not decode, so an unreadable one
        # is redrawn over rather than blocking the call; the entry it replaces is unknowable.
        logger.warning("the locked split for %s does not decode; redrawing over it",
                       identity_hash, exc_info=True)
        old_lock = None
    old_membership = ({"calibration": old_lock.get("calibration", []),
                       "holdout": old_lock.get("holdout", [])} if old_lock else None)

    if manifest_stems is not None:
        stems = manifest_stems
        try:
            annotation_counts = {s: count_label_lines(labels_dir, s) for s in stems}
        except UnreadableLabelDocument as exc:
            return {"error": str(exc)}
    elif labels_dir:
        # The same labels-intersect-images scan _calibrate_operating_point uses, not a second
        # independent glob (images_dir omitted degrades to the prior labels-only scan).
        stems, _ = label_image_stems(labels_dir, images_dir)
        try:
            annotation_counts = {s: count_label_lines(labels_dir, s) for s in stems}
        except UnreadableLabelDocument as exc:
            return {"error": str(exc)}
    elif old_lock:
        stems = sorted(set(old_lock.get("calibration", [])) | set(old_lock.get("holdout", [])))
        annotation_counts = None
    else:
        return {"error": f"no existing lock for identity_hash={identity_hash!r}, and no "
                          "labels_dir to derive stems from"}

    new_lock = resolve_locked_cal_holdout_split(
        stems, identity_hash=identity_hash, scope_root=scope_root,
        annotation_counts=annotation_counts,
        group_by=(group_by or "tile_prefix"), group_key_map=group_key_map,
        holdout_ratio=holdout_ratio, seed=seed,
        force_redraw=True, timestamp=datetime.now(timezone.utc).isoformat(),
        split_manifest_dir=split_manifest_dir,
    )
    new_membership = {"calibration": new_lock["calibration"], "holdout": new_lock["holdout"]}

    # A distinct tool name under the same scope: @audited logs the call, this logs what it made.
    record_event(
        "force_redraw_cal_holdout_split_result",
        {"identity_hash": identity_hash, "group_by": group_by, "group_key_map": group_key_map,
         "seed": seed, "holdout_ratio": holdout_ratio, "reason": reason,
         "split_manifest_dir": split_manifest_dir},
        scope=dataset_scope_of(str(scope_root)),
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
    conf_threshold: float | None = None,
    device: str | None = None,
    tile: bool | None = None,
    tile_size: int | None = None,
    overlap: float | None = None,
    tile_batch_size: int = 96,
    global_nms_iou: float | None = None,
    max_dets: int | None = None,
    postprocess: str = "nms",
    dry_run: bool = False,
    trait: str | None = None,
    calibration_labels_dir: str | None = None,
    calibration_images_dir: str | None = None,
    experiment_id: str | None = None,
    group_by: str | None = None,
    group_key_map: dict[str, str] | None = None,
    split_seed: int = 0,
    split_holdout_ratio: float = 0.5,
    split_manifest_dir: str | None = None,
) -> dict:
    """Run a trained model on images.

    Provide either image_paths (specific images) or images_dir (all images
    in a directory). Set ``tile=True`` for SAHI-style sliding-window detection on
    high-resolution imagery with many small objects (detection heads only).

    Args:
        checkpoint_path: Path to model .pt checkpoint.
        image_paths: List of specific image paths.
        images_dir: Directory containing images to process.
        conf_threshold: Minimum confidence score. ``None`` (default) states nothing and runs at the
            platform default, stamped ``"default"``; a stated value is honored as an explicit
            override and stamped as one, including when it happens to equal the platform default.
        device: Device to use ('cuda' or 'cpu').
        tile: Enable tiled (SAHI-style) detection inference. ``None`` (default) derives it from the
            checkpoint's own training tile geometry (``predictor.train_tile_size is not None``), not
            a fixed default, its provenance is stamped ``"default"`` vs ``"explicit"`` so a caller
            who deliberately chose one way is distinguishable from one who left it unset. Works for
            ``instance_seg`` too: each tiled result's ``masks`` (see
            ``GenericPredictor.predict_tiled``) are a tile-local patch plus its full-image-space
            offset, never the untiled path's dense full-image array, a downstream reader of
            ``results[i]["masks"]`` must not assume the two shapes are interchangeable.
        tile_size: Sliding-window tile edge (px). ``None`` (default) derives it from the
            checkpoint's training tile geometry so inference matches the trained scale; a value
            overrides. A checkpoint that trained untiled on frames that all shared one square size
            derives the edge from that frame instead (``"native_ratio"``), and each tile is run
            through the resize that run's own augmentation config recorded, so the model sees a tile
            the way it saw a training frame; that edge is a real geometry basis, weaker than a
            persisted one and stronger than an explicit caller edge, and a delivery door admits it
            on its own. A checkpoint with
            none of those has no real basis to tile at: if ``tile`` ends up ``True`` with no
            resolvable ``tile_size``, this refuses (names the missing basis) rather than fabricate one.
        overlap: Fractional tile overlap (stride = tile_size*(1-overlap)). ``None`` derives from the
            checkpoint (else 0.2).
        tile_batch_size: Tiles per forward batch.
        global_nms_iou: Cross-tile global NMS IoU threshold. ``None`` (default) means the caller
            stated nothing, so a calibrated run derives it from the calibration GT's own
            neighbor-IoU distribution; a stated value is honored as an explicit override and
            stamped as one, including when it happens to equal the platform default.
        max_dets: Full-frame detection cap (after any tiled merge). ``None`` (default) means the
            caller stated nothing, so a calibrated run derives it from the calibration GT's own
            object density; a stated value is honored as an explicit override and stamped as one,
            including when it happens to equal the platform default.
        postprocess: Cross-tile merge, "nms" suppresses overlaps, "nmm" unions boxes split
            across a tile seam (better for an object straddling a boundary).
        dry_run: Report the effective operating point (conf/tiling/max_dets/postprocess) without
            loading the model or running inference.
        trait: Trait name (with ``calibration_labels_dir``) to derive the confidence operating point
            per dataset instead of pinning a default, the count is the phenotype, so conf must be
            calibrated. Absent -> the byte-identical raw path (conf=score_threshold, unvalidated).
        calibration_labels_dir: Labeled dir for a disjoint cal/holdout split to calibrate + held-out
            validate the operating point. Its GT identity scopes the resolved conf (dataset firewall).
        calibration_images_dir: Images for the calibration labels (defaults to ``images_dir``).
        experiment_id: The run that produced the checkpoint, for provenance. Best-effort resolved
            (checkpoint's own stamp, then the registry) when omitted; a raw/foreign checkpoint
            legitimately has none. Also gates calibration's train-disjointness check: a
            *known* run whose training split can't be read/reconstructed fails that check closed.
        group_by: Grouping policy for the locked calibration/holdout split, ``"tile_prefix"``
            or ``"stem"``. Ignored when ``group_key_map`` is given. ``None`` (default) resolves
            to ``"tile_prefix"`` when neither this nor ``split_manifest_dir`` was given; a value
            beside ``split_manifest_dir`` conflicts with the manifest's own grouping policy and
            refuses, naming both. Only the first calibration call for a given calibration-labels
            identity draws the split; later calls return the same locked split regardless of this
            argument (see ``force_redraw_cal_holdout_split`` to redraw deliberately).
        group_key_map: An agent-derived ``{stem: group_key}`` map overriding ``group_by`` for the
            locked calibration/holdout split, must cover every stem in ``calibration_labels_dir``.
            Conflicts with ``split_manifest_dir`` the same way ``group_by`` does.
        split_seed: Split seed for the locked calibration/holdout split, like
            ``group_by``, only takes effect on the first calibration call for a given
            calibration-labels identity; a later call's declared value is compared to the lock and
            any divergence is reported in ``sweep_summary``/the resolved bundle rather than
            silently ignored.
        split_holdout_ratio: Calibration/holdout fraction for the locked split, same
            first-call-only semantics as ``split_seed``.
        split_manifest_dir: Restrict the calibration universe to one capture date's held-out side
            of a split manifest (``data_tools.read_split_manifest_dir``) instead of every
            labelled stem with an image, so the operating point is measured on exactly the set
            the checkpoint was chosen against, not diluted with training stems. The manifest's
            subject/attribute must equal the checkpoint's own recorded training scope, the
            calibration labels' date must be one the manifest holds members under, and the
            manifest's ``images_root`` for that date must be ``calibration_images_dir`` (or
            ``images_dir``), each refusing by name.
    """
    if not Path(checkpoint_path).is_file():
        return {"error": f"Checkpoint not found: {checkpoint_path}"}

    # An unstated cap falls to the shared platform default for the pass while staying unstated for
    # the resolver, the only thing that can derive one from the data.
    applied_nms_iou = DEFAULT_NMS_IOU if global_nms_iou is None else float(global_nms_iou)
    applied_max_dets = DEFAULT_MAX_DETS if max_dets is None else int(max_dets)
    max_dets_stated = max_dets is not None
    applied_conf = DEFAULT_CONF if conf_threshold is None else float(conf_threshold)
    conf_stated = conf_threshold is not None

    if dry_run:
        # No model load here: an unset ``tile`` is a pending derivation (the checkpoint decides
        # it), not a fabricated default, like tile_size/overlap already report.
        if tile is None:
            tiled_dry: bool | str = "pending-checkpoint-derivation"
            tiled_source_dry = "pending-checkpoint-derivation"
            cross_tile_nms_dry: float | None | str = "pending-checkpoint-derivation"
        else:
            tiled_dry, tiled_source_dry = tile, "explicit"
            cross_tile_nms_dry = applied_nms_iou if tile else None
        return {
            "dry_run": True,
            "checkpoint_path": checkpoint_path,
            "operating_point": {
                "conf": applied_conf,
                "cross_tile_nms": cross_tile_nms_dry,
                "tiled": tiled_dry,
                "tiled_source": tiled_source_dry,
                "tile_size": tile_size if tile_size is not None else "pending-checkpoint-derivation",
                "overlap": overlap if overlap is not None else "pending-checkpoint-derivation",
                "max_dets": applied_max_dets,
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

    # NMS IoU + the full-frame detection cap govern which boxes exist (in-model thresholds), not
    # just cross-tile merge, else nms_iou would have no effect on an untiled run.
    predictor = build_predictor(
        checkpoint_path=checkpoint_path,
        device=device,
        score_threshold=applied_conf,
        nms_iou=applied_nms_iou,
        max_dets=applied_max_dets,
    )

    # Resolve the tiled bool now the checkpoint's own persisted training geometry is in hand: an
    # unset ``tile`` gets the checkpoint's own tiled-or-not regime, never a fixed platform default.
    tiled_source = "explicit" if tile is not None else "default"
    resolved_tile_bool = (
        getattr(predictor, "train_tile_size", None) is not None) if tile is None else tile

    # Identity resolved before calibration: its train-disjointness gate needs the checkpoint's
    # experiment_id. sha is cached (never re-hashed per call).
    from tcip_mcp.model_registry import resolve_model_identity

    identity = resolve_model_identity(checkpoint_path, experiment_id=experiment_id)

    # Derive tile geometry from training geometry unless the caller pinned it; refuses only when a
    # stated edge contradicts the checkpoint's own recorded geometry.
    from tcip_mcp.pipelines.inference.predictor import (
        TileEdgeContradiction, explicit_edge_provenance, resolve_tile_regime,
    )

    # resolve_tile_regime resolves the resize half only when tiled, so an untiled run is never
    # sunk by an unreadable recorded augmentation config.
    try:
        resolved_tile, tile_size_source, resolved_overlap, overlap_source, tile_resize = (
            resolve_tile_regime(
                predictor, tiled=resolved_tile_bool, tile_size=tile_size, overlap=overlap))
    except TileEdgeContradiction as exc:
        return {"error": str(exc)}
    tile_size_derived_from = (
        explicit_edge_provenance(predictor, resolved_tile)
        if tile_size_source == "explicit" and resolved_tile is not None else None)
    if resolved_tile_bool and resolved_tile is None:
        # Tiling was requested but nothing justifies a scale: refuse rather than fabricate one.
        return {"error": (
            f"tile_size could not be resolved for {checkpoint_path}: this checkpoint carries no "
            "persisted training tile geometry, no tile_size was given explicitly, and its untiled "
            "training frame yields no tile edge either (none recorded, or a rectangular one, which "
            "no single square edge reproduces the scale of on both axes), so tiled inference has no "
            "real basis to run at. Pass tile_size explicitly, retrain with tile geometry persisted, "
            "or leave tile unset/False to run untiled."
        )}
    if tile_size_source == "derived":
        logger.info("tile_size %d derived from the checkpoint's training geometry", resolved_tile)
    elif resolved_tile_bool and tile_size_source == "native_ratio":
        resize_note = "" if tile_resize is None else (
            f", each tile run through its recorded train-time resize {tuple(tile_resize)}")
        logger.info(
            "tile_size %d derived from this checkpoint's own uniform untiled training frame%s",
            resolved_tile, resize_note)
    # overlap_source == "default" is unremarkable (no persisted overlap analog); only tile_size's
    # absence changes the object count's scale.

    if image_paths is None:
        if images_dir is None:
            return {"error": "Provide either image_paths or images_dir"}
        # Fold a `.bandgroup`-grouped capture into its one logical entry (list_logical_images),
        # the same enumeration every other reader in this platform shares, instead of this door's
        # own raw sibling-file listing enumerating each band file as its own (spurious) image.
        from tcip_mcp.pipelines.image_utils import list_logical_images

        logical = list_logical_images(images_dir)
        image_paths = [logical[stem] for stem in sorted(logical)]

    # Resolve the confidence operating point: with a trait + labeled calibration dir, derive it
    # per dataset (count-unbiased + held-out validated); otherwise the byte-identical raw path.
    extra: dict = {}
    if trait and calibration_labels_dir:
        from tcip_annotation.json_io import UnreadableLabelDocument

        cal_images = calibration_images_dir or images_dir
        try:
            bundle, cal_hash, n_excluded_incomplete_attribute, evidence = _calibrate_operating_point(
                predictor, trait, calibration_labels_dir, cal_images,
                tile=resolved_tile_bool, tile_size=resolved_tile, overlap=resolved_overlap,
                tile_resize=tile_resize,
                tile_size_source=tile_size_source, tile_size_derived_from=tile_size_derived_from,
                tiled_source=tiled_source,
                tile_batch_size=tile_batch_size, global_nms_iou=applied_nms_iou,
                postprocess=postprocess,
                cross_tile_nms=global_nms_iou, max_dets=max_dets,
                group_by=group_by, group_key_map=group_key_map,
                experiment_id=identity["experiment_id"],
                seed=split_seed, holdout_ratio=split_holdout_ratio,
                split_manifest_dir=split_manifest_dir,
            )
        except (ValueError, UnreadableLabelDocument) as exc:
            # An inadmissible reference, a locked split that no longer resolves, or a calibration
            # GT file that will not read: a clean refusal, not a bare KeyError downstream.
            return {"error": str(exc)}
        conf_param = bundle.get("conf")
        conf = (conf_param.value if conf_param.is_shippable
                else conf_param.unvalidated_value(acknowledge_unvalidated=True))
        applied_max_dets = int(bundle.get("max_dets").value)
        applied_nms_iou = float(bundle.get("cross_tile_nms").value or applied_nms_iou)
        # Apply the resolved operating point to the model so it governs which boxes exist.
        predictor.score_threshold = conf
        set_detector_operating_point(predictor.model, score_thresh=conf,
                                     detections_per_img=applied_max_dets)
        op_bundle = bundle
        # Dataset-scope firewall: the conf is scoped to the calibration GT. The inference target is
        # usually unlabeled, so its GT identity (a content hash) is undefined, pass None and record
        # 'not-comparable-unlabeled-target'. Only when inferencing the same labeled set it calibrated
        # on can we compare real hashes and flag cross-dataset inheritance.
        from tcip_mcp.pipelines.image_utils import stem_of

        from tcip_annotation.json_io import prediction_documents

        inf_stems = [stem_of(pp) for pp in image_paths]
        cal_label_stems = (
            set(evidence.get("calibration_stems", [])) if split_manifest_dir is not None
            else {pp.stem for pp in prediction_documents(calibration_labels_dir)}
        )
        same_images = calibration_images_dir is None or (
            images_dir is not None and Path(calibration_images_dir) == Path(images_dir))
        if same_images and inf_stems and set(inf_stems) == cal_label_stems:
            target_hash, cross_dataset_check = dataset_hash(calibration_labels_dir, stems=inf_stems), "same-labeled-set"
        else:
            target_hash, cross_dataset_check = None, "not-comparable-unlabeled-target"
        issues = bundle.shippable_issues(target_dataset_hash=target_hash)
        # Channel firewall: probe one target raster and check its band count against the
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
                        "in_chans", int(getattr(predictor, "in_chans", 3)))})
                issues = issues + validate_resolved_bundle(chan_bundle, probed_channels=probed)
        # validated only when held-out passed and nothing is un-shippable under the target actually used.
        validated = bool(bundle.is_shippable and not issues)
        if (conf_param.sweep or {}).get("conf_floor_mismatch"):
            # Read after `validated`: this one travels to the delivery surface without gating there.
            issues = issues + [
                "conf: the reference's own lowest detection score sits materially above the conf "
                "floor this calibration staged it at, so something truncated the reference after "
                "it was generated (a stale bucket, cap-trimmed tiles, a bespoke producer) and the "
                "swept curve never saw the low-conf tail it assumes"
            ]
        extra = {
            "validated": validated,
            "shippable_issues": issues,
            "cross_dataset_check": cross_dataset_check,
            "conf_source": "calibration",
            "dataset_hash": cal_hash,
            "sweep_summary": _sweep_summary(conf_param),
            "n_excluded_incomplete_attribute": n_excluded_incomplete_attribute,
        }
        # The full sweep can be large, persist it and return the path (provenance emits has_sweep).
        # Keyed on cal_hash alone, a second checkpoint (or the same checkpoint under
        # different tile/postprocess settings) calibrated on the same labels would silently overwrite the
        # prior curve. Content-address the filename by every dimension the sweep actually depends
        # on, checkpoint identity + the full predictor path, not just the labels, so it can't be
        # under-keyed again the next time a dimension is added; identical inputs harmlessly reuse
        # the same file, different inputs never collide.
        try:
            import hashlib
            import json as _json

            from tcip_store import store

            sweep_key = {
                "trait": trait,
                "dataset_hash": cal_hash,
                "checkpoint_sha256": identity["sha256"],
                "predictor_path": {
                    "tile": resolved_tile_bool, "tile_size": resolved_tile,
                    "overlap": resolved_overlap, "postprocess": postprocess,
                    "global_nms_iou": applied_nms_iou, "max_dets": applied_max_dets,
                },
            }
            body_hash = hashlib.sha256(
                _json.dumps(sweep_key, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
            # The evidence rides in the sweep artifact, read back by identity, never on this response.
            store.replace(confidence_sweep_key(body_hash),
                          {**sweep_key, "sweep": conf_param.sweep,
                           "calibration_evidence": evidence})
            extra["sweep_path"] = str(confidence_sweep_path(body_hash))
            extra["calibration_evidence_key"] = body_hash
        except Exception:
            logger.warning("could not persist operating-point sweep", exc_info=True)
    else:
        # Raw inference has no per-dataset calibration: the model already carries score_threshold as
        # its in-model conf; the bundle stamps it validated_against=false so the un-trustworthiness of
        # this uncalibrated operating point (the count is the phenotype) travels with the result.
        op_bundle = raw_operating_point(
            conf=applied_conf, cross_tile_nms=applied_nms_iou, tiled=resolved_tile_bool,
            tile_size=resolved_tile, max_dets=applied_max_dets, tile_size_source=tile_size_source,
            tile_size_derived_from=tile_size_derived_from,
            tiled_source=tiled_source, conf_stated=conf_stated,
            max_dets_stated=max_dets_stated,
        )
        extra = {"validated": False, "conf_source": "default"}

    # Preflight: warn (don't fail) when a slow workload will run on CPU because CUDA isn't
    # available, full tiled inference over thousands of images is hours on CPU vs minutes on
    # a GPU. Install a CUDA torch build (see environment.yml) to use the card.
    cpu_warning = None
    if device != "cpu" and (resolved_tile_bool or len(image_paths) > 8):
        import torch

        if not torch.cuda.is_available():
            cpu_warning = (
                f"CUDA not available, running {len(image_paths)} image(s)"
                f"{' tiled' if resolved_tile_bool else ''} on CPU, which is much slower. Install a "
                "CUDA torch build (see environment.yml) to use the GPU."
            )
            logger.warning(cpu_warning)

    results = predictor.predict_batch(
        image_paths, tile=resolved_tile_bool, tile_size=resolved_tile, overlap=resolved_overlap,
        tile_batch_size=tile_batch_size, global_nms_iou=applied_nms_iou, postprocess=postprocess,
        tile_resize=tile_resize,
    )
    # A degenerate box is no detection, so it is excluded here too; left at the raw per-image
    # count when masks are present, since only the writer's mask-to-polygon conversion decides.
    total_detections = sum(
        r["count"] if r.get("masks") is not None else positive_detections(r)[0] for r in results
    )

    # Producing-model identity (resolved above, before calibration) travels with the result so every
    # downstream deliverable can name the exact checkpoint (content hash) + run behind the count.
    from datetime import datetime, timezone

    # This run's name→id map, reused for both recording and decode, so export records it in
    # operating_point.json and decodes predictions to names through this one map, consistent
    # within the run. The one resolution both doors that write predictions to disk use (this tool
    # and the web GUI's own inference worker, routes/inference.py), never a second implementation.
    id_map = resolve_decode_id_map(predictor, images_dir)
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
        # cross_tile_nms/tiled/tile_size/max_dets are), surface the value this specific call
        # actually ran at directly, rather than silently drop it after resolving it.
        "overlap": resolved_overlap,
        "overlap_source": overlap_source,
        "operating_point": op_bundle.to_provenance()["operating_point"],
        "id_map": id_map,
        "results": results,
        **extra,
    }
    if cpu_warning:
        out["warning"] = cpu_warning
    return out


# --- earning the record a validated count claim names (the shared half of every door here) ---

_NO_DATASET_ROOT_NOTE = (
    "{bucket} sits under no dataset root, so two guarantees a bucket normally carries are absent "
    "here. The review-verdict immutability guard is inoperative: nothing checks whether a human "
    "has already recorded verdicts against predictions at this path before this run replaced them. "
    "And a count claim earned for these predictions has no dataset-relative key to be recorded "
    "under, so this bucket is stamped unvalidated whatever its operating point cleared. Write into "
    "a dataset's own predictions layout (<dataset_root>/predictions/<model>/<date>) for both."
)
"""What a door tells its caller about a bucket outside the dataset layout, rather than guarding it
against a verdict store that holds nothing about it or letting it claim a count nothing can verify."""


def _bucket_dataset_root(bucket: Path) -> Path | None:
    """The dataset root a bucket's count claim is recorded against, or ``None`` when it is under
    none. Resolved through ``dataset_root_of``, the same derivation the reader recomputes the
    covered-bucket key from, so a door cannot record a key the verifier will not look for."""
    from tcip_mcp.dataset_layout import dataset_root_of

    root = dataset_root_of(bucket)
    return root.resolve() if root is not None else None


def _resolve_writable_bucket_for(output_dir: str, *, overwrite: bool):
    """The bucket a run may write for ``output_dir``, its resolution, and its dataset root.

    Returns ``(out, resolution, dataset_root, refusal)``. ``refusal`` is the door's own error dict
    when the requested bucket carries review verdicts and the caller asked to overwrite it, and
    ``None`` otherwise. One resolution for both doors here that persist a bucket, so the canonical
    ``predictions/<model>/<date>`` redirect (which varies the model segment, the one every
    date-keyed reader enumerates) and the bespoke last-segment redirect cannot drift apart.
    ``dataset_root`` is ``None`` for a bucket under no dataset, whose verdict guard is inoperative.
    """
    from tcip_mcp.dataset_layout import prediction_dir
    from tcip_mcp.prediction_buckets import (
        BucketHasVerdicts,
        BucketResolution,
        resolve_prediction_bucket,
        resolve_writable_bucket,
        review_state_dir_of,
    )

    out_path = resolve_output_path(output_dir)
    parent, base_name = out_path.parent, out_path.name

    canonical_dataset_root = None
    if parent.name and parent.parent.name == "predictions":
        candidate_root = parent.parent.parent
        if Path(prediction_dir(candidate_root, parent.name, base_name)).resolve() == out_path.resolve():
            canonical_dataset_root = candidate_root

    # The guard reads the bucket's own dataset verdict store; no dataset root means no store to guard against.
    dataset_root = _bucket_dataset_root(out_path)
    review_state_dir = None if dataset_root is None else review_state_dir_of(dataset_root)

    try:
        if canonical_dataset_root is not None:
            out, resolution = resolve_prediction_bucket(
                canonical_dataset_root, parent.name, base_name,
                review_state_dir=review_state_dir, overwrite=overwrite)
        elif review_state_dir is not None:
            resolution = resolve_writable_bucket(
                review_state_dir, base_name, lambda n: [parent / n], overwrite=overwrite)
            out = parent / resolution.name
        else:
            resolution = BucketResolution(name=base_name, redirected=False, verdict_count=0,
                                          requested=base_name)
            out = out_path
    except BucketHasVerdicts as exc:
        suggested = (
            str(prediction_dir(canonical_dataset_root, exc.suggested, base_name))
            if canonical_dataset_root is not None
            else str(parent / exc.suggested)
        )
        return None, None, dataset_root, {
            "error": str(exc), "verdict_count": exc.count, "suggested_bucket": suggested}
    return out, resolution, dataset_root, None


def _calibration_evidence(result: dict) -> dict | None:
    """The evidence this run's calibration gate ran over, read back from the artifact it was kept
    in, or ``None`` for a run that resolved no calibrated operating point.

    Read rather than carried on the run's own response: the records are the largest thing a
    calibration produces, and only a door earning a validation record has any use for them.
    """
    identity = result.get("calibration_evidence_key")
    if not identity:
        return None
    from tcip_store import store

    return (store.read(confidence_sweep_key(identity), default=None) or {}).get(
        "calibration_evidence")


def _open_count_claim(evidence: dict, *, trait: str, checkpoint_sha256: str | None,
                      producing_experiment_id: str | None, dataset_root: Path):
    """Run the count document's own gate over the calibration evidence, before anything is written.

    Raises ``ValueError`` when the resolver's result clears no reference the count document accepts,
    which ends the run with the bucket still untouched.
    """
    from tcip_mcp.pipelines.resolution import open_validation

    return open_validation(
        document="operating_point",
        evidence={"resolver": evidence["resolver"], "inputs": evidence["inputs"]},
        trait=trait,
        checkpoint_sha256=checkpoint_sha256,
        producing_experiment_id=producing_experiment_id,
        reference_inputs={**evidence["reference_inputs"], "dataset_root": str(dataset_root)},
    )


def _draft_count_claim(result: dict, *, trait: str | None, bucket: Path,
                       dataset_root: Path | None, tile_size_validated: str | None):
    """The passed gate a validated count is stamped from, for a run that earned one.

    Returns ``(draft, refusal)``. ``refusal`` is the door's own error dict for a run that reports a
    validated operating point with no evidence left to earn a record from, which ends the run with
    the bucket still untouched. A run whose own dimensions did not all clear earns nothing, and so
    does a bucket under no dataset root, whose count claim would have no dataset-relative key to be
    recorded under; both stamp unvalidated rather than refusing, since producing predictions from a
    bespoke checkpoint or into a bespoke path is legitimate work.
    """
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE

    if not result.get("validated") or tile_size_validated == VALIDATED_FALSE:
        return None, None
    evidence = _calibration_evidence(result)
    if evidence is None:
        return None, {"error": (
            f"the run reports a validated operating point for trait {trait!r} but kept no "
            "calibration evidence to earn a validation record from, so these counts cannot be "
            "stamped validated. The evidence is written beside the confidence sweep at calibration "
            "time; re-run the calibration, or pass acknowledge_unvalidated=True to write an "
            "honestly-flagged provisional result."
        )}
    if dataset_root is None:
        logger.warning(_NO_DATASET_ROOT_NOTE.format(bucket=bucket))
        return None, None
    try:
        draft = _open_count_claim(
            evidence, trait=trait, checkpoint_sha256=result.get("checkpoint_sha256"),
            producing_experiment_id=result.get("experiment_id"), dataset_root=dataset_root)
    except ValueError as exc:
        return None, {"error": f"the count claim for trait {trait!r} was not earned: {exc}"}
    return draft, None


def _seal_and_stamp(out: Path, stamp_body: dict, draft) -> dict:
    """Append the record the gate earned over the files as they landed, then write the stamp last.

    The last two steps of the order every producer of a prediction bucket follows, in one place so
    the doors cannot drift into different ones. ``draft`` is ``None`` for a run that earned nothing,
    which stamps unvalidated with no pointer. A crash between the record and the stamp leaves a row
    no stamp names, which is inert; a crash before either leaves prediction files that floor.
    """
    from tcip_mcp.pipelines.resolution import seal_validation, write_sidecar

    if draft is not None:
        _digest, stamp_body = seal_validation(
            draft, dataset_root=draft.dataset_root, bucket_dirs=[out], stamp_body=stamp_body)
    write_sidecar(out, stamp_body)
    return stamp_body


def _publish_image_predictions(out: Path, result: dict, *, checkpoint_path: str,
                               trait: str | None, images_dir: str | None,
                               tile_size_validated: str | None, draft
                               ) -> tuple[list[str], int, dict]:
    """Write one prediction file per image, then earn and stamp over exactly what landed.

    The steps ``export_predictions`` and ``tabulate_counts`` share once each has resolved its own
    bucket and run its own gate: both persist the same run's per-image detections into a bucket and
    both stamp it, so the file naming, the producer string, the claim payload and the write order
    are one implementation rather than two that agree today. ``draft`` present is exactly the
    condition for a validated stamp: a door opens one only when the run's own dimensions all
    cleared and the bucket sits where a claim can be recorded.

    Returns ``(written, dropped_nonpositive_boxes, stamp_body)``: the middle value is the count of
    detections dropped for a zero-extent box across every image, for the caller's own summary.
    """
    from tcip_mcp.pipelines.resolution import operating_point_stamp, prediction_producer

    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    # The checkpoint's content hash, so an accepted prediction's GT names the exact model behind it.
    sha = result.get("checkpoint_sha256")
    producer = prediction_producer(checkpoint_path, sha)
    id_map = result.get("id_map")
    has_masks = False
    dropped = 0
    for r in result["results"]:
        out_json = out / f"{Path(r['image']).stem}.json"
        # Read before the write: a drop can empty a mask list that was genuinely there, and
        # has_masks must reflect what this run used, not what happened to survive the drop.
        has_masks = has_masks or bool(r.get("masks"))
        dropped += write_predictions_json(out_json, r, created_by=producer, id_map=id_map)
        written.append(str(out_json))

    op_stamp = operating_point_stamp(
        result.get("operating_point"),
        validated=draft is not None,
        validated_by=None,
        tile_size_validated=tile_size_validated,
        shippable_issues=result.get("shippable_issues", []),
        id_map=id_map,
        trait=trait,
        dataset_hash=result.get("dataset_hash"),
        checkpoint=Path(checkpoint_path).stem,
        checkpoint_sha256=sha,
        experiment_id=result.get("experiment_id"),
        images_dir=images_dir,
        raster_path=None,
        produced_at=result.get("produced_at"),
        sweep_path=result.get("sweep_path"),
        sweep_summary=result.get("sweep_summary"),
    )
    if has_masks:
        # The run-constant mask-binarize threshold travels once here rather than per-annotation.
        op_stamp["mask_binarize"] = mask_binarize_provenance()
    return written, dropped, _seal_and_stamp(out, op_stamp, draft)


def _export_predictions_raster(
    *, checkpoint_path: str, raster_path: str, out: Path, resolution, device: str | None,
    conf_threshold: float | None, tile_size: int | None, overlap: float | None, tile_batch_size: int,
    global_nms_iou: float | None, max_dets: int | None, postprocess: str, require_masks: bool,
    experiment_id: str | None, acknowledge_unvalidated: bool, trait: str | None = None,
) -> dict:
    """The windowed-raster regime of :func:`export_predictions`: tiled detection/instance_seg
    inference over a whole georeferenced (or merely huge) raster too large to decode whole, sourced
    from the windowed raster layer (:func:`~tcip_mcp.pipelines.raster_source.open_raster`) rather
    than an ordinary directory of per-image captures. Always tiled: there is no untiled option, the
    whole point of this regime is a raster too large for one. ``out``/``resolution`` are the bucket
    :func:`export_predictions` already resolved (immutability/redirect), shared with the ordinary
    regime rather than a second implementation of that check.

    Persists one prediction bucket: since there is no natural directory-of-per-plant-images shape
    for a whole-raster capture, "one image" is the whole raster, so the bucket holds exactly one
    ``<raster stem>.json`` prediction file (in full-raster pixel space) plus the same
    ``operating_point.json`` sidecar convention every other bucket carries.

    ``trait`` is ``None`` for every raster export with no reserved calibration region (the
    original, byte-identical raw path: the persisted operating point is never stamped validated
    here, conf has no per-dataset calibration for a raster source, a validated per-plant count is
    earned later at delivery). ``export_predictions`` only ever passes a real ``trait`` here once
    it has already confirmed the checkpoint's training experiment reserved a calibration region
    (:func:`~tcip_mcp.pipelines.block_calibration.reserved_calibration_region_available`); in that
    case this runs block calibration first (small reserved bands, not the whole mosaic), gates on
    the mosaic's own claim scope (:func:`~tcip_mcp.pipelines.raster_source.raster_identity_matches`
    -- the block-validated reference must be *this* raster, not a different one reusing the same
    checkpoint), then applies the calibrated conf/cross_tile_nms to the real whole-mosaic pass with
    ``max_dets`` deliberately uncapped (never the block bundle's own band-scoped density-derived
    value, which would truncate a whole-mosaic count to roughly one band's worth of objects).

    The tile_size gate (and, for the ``trait`` path, the claim-scope gate) runs pre-pass, before
    the always-expensive tiled pass. Unlike the ordinary regime (which can fall back to running
    untiled), this regime always tiles, so a checkpoint with no persisted geometry, no recoverable
    native-frame edge, and no explicit override has no real basis to tile at *at all*: that refusal
    is unconditional, never overridable via ``acknowledge_unvalidated`` (there is no value to
    provisionally proceed with). This door has no real-but-unvalidated tile scale left to admit on
    acknowledgement: every basis the gate below resolves to either clears it on its own or is this
    no-basis-at-all case. An explicit ``tile_size`` that contradicts the checkpoint's own recorded
    geometry refuses before that, from ``resolve_tile_regime`` itself. For the ``trait`` path, the
    block calibration's own reserved-region bands must also be tiled at this same resolved edge;
    a split manifest recorded at a different edge refuses there too.
    """
    from tcip_mcp.model_registry import resolve_model_identity
    from tcip_mcp.pipelines.inference.predictor import (
        TileEdgeContradiction, build_predictor, explicit_edge_provenance, resolve_tile_regime,
    )

    # An unstated cap falls to the shared platform default for the pass; this regime has no
    # per-dataset derivation of one to leave room for.
    applied_nms_iou = DEFAULT_NMS_IOU if global_nms_iou is None else float(global_nms_iou)
    applied_max_dets = DEFAULT_MAX_DETS if max_dets is None else int(max_dets)
    raw_max_dets_stated = max_dets is not None
    applied_conf = DEFAULT_CONF if conf_threshold is None else float(conf_threshold)
    raw_conf_stated = conf_threshold is not None

    predictor = build_predictor(
        checkpoint_path=checkpoint_path, device=device, score_threshold=applied_conf,
        nms_iou=applied_nms_iou, max_dets=applied_max_dets,
    )
    identity = resolve_model_identity(checkpoint_path, experiment_id=experiment_id)
    if identity["experiment_id"]:
        # Checked before the raster pass, not after: a blob write cannot join the record's own
        # transaction, so this is the one chance to refuse before the export writes anything.
        from tcip_mcp.experiments import pointer_frozen

        frozen = pointer_frozen(identity["experiment_id"], "lineage", "predictions", str(out))
        if frozen is not None:
            return {"error": frozen}

    # Resolved (resize included) before the raster is opened: an unreadable recorded augmentation
    # config, or a stated edge contradicting the checkpoint, refuses here, not mid-pass.
    try:
        resolved_tile, tile_size_source, resolved_overlap, overlap_source, tile_resize = (
            resolve_tile_regime(predictor, tiled=True, tile_size=tile_size, overlap=overlap))
    except TileEdgeContradiction as exc:
        return {"error": str(exc)}
    tile_size_derived_from = (
        explicit_edge_provenance(predictor, resolved_tile)
        if tile_size_source == "explicit" and resolved_tile is not None else None)
    if resolved_tile is None:
        return {"error": (
            f"tile_size could not be resolved for {checkpoint_path}: this checkpoint carries no "
            "persisted training tile geometry and no tile_size was given explicitly, so this "
            "always-tiled regime has no real basis to run at. Pass tile_size explicitly, or "
            "retrain with tile geometry persisted."
        )}

    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE, block_calibrated_export_operating_point, check_delivery_gate,
        operating_point_stamp, prediction_producer, raw_operating_point, tile_size_gate_flag,
    )

    conf_source = "default"
    block_prov: dict | None = None
    block_evidence: dict | None = None
    claim_scope_validated: str | None = None
    claim_scope_mismatch: str | None = None
    bucket_root = _bucket_dataset_root(out)
    draft = None

    if trait is not None:
        from tcip_mcp.pipelines.block_calibration import (
            BlockCalibrationRefused, resolve_block_calibration_records,
        )
        from tcip_mcp.pipelines.raster_source import (
            georeferenced_raster_identity_mismatch, raster_identity_matches,
        )
        from tcip_mcp.pipelines.resolution import (
            VALIDATED_SAME_MOSAIC_CONTENT_IDENTITY, VALIDATED_SAME_MOSAIC_IDENTITY,
        )

        try:
            block_bundle, block_prov, block_evidence = resolve_block_calibration_records(
                predictor, checkpoint_path=checkpoint_path, trait_name=trait,
                experiment_id=identity["experiment_id"], global_nms_iou=applied_nms_iou,
                export_tile_size=resolved_tile,
                tile_batch_size=tile_batch_size, postprocess=postprocess,
            )
        except BlockCalibrationRefused as exc:
            return {"error": str(exc)}

        training_identity = (block_prov["spatial_manifest"] or {}).get("raster_content_identity")
        if training_identity is None:
            return {"error": (
                "block calibration refused: no raster content identity was recorded for "
                f"experiment {block_prov['experiment_id']!r} at spatial-split time (an unreadable "
                "or unsupported training source); the claim-scope gate has nothing to compare "
                "this export target against."
            )}
        try:
            if training_identity.get("geotransform") is not None:
                claim_scope_mismatch = georeferenced_raster_identity_mismatch(
                    training_identity, raster_path)
                claim_scope_token = VALIDATED_SAME_MOSAIC_IDENTITY
            else:
                claim_scope_mismatch = (
                    None if raster_identity_matches(training_identity, raster_path)
                    else f"{raster_path} is not the raster this identity was recorded on"
                )
                claim_scope_token = VALIDATED_SAME_MOSAIC_CONTENT_IDENTITY
        except ValueError as exc:
            return {"error": f"claim-scope check refused: {exc}"}
        claim_scope_flag = (
            claim_scope_token if claim_scope_mismatch is None else VALIDATED_FALSE)

        conf_param = block_bundle.get("conf")
        conf = (conf_param.value if conf_param.is_shippable
                else conf_param.unvalidated_value(acknowledge_unvalidated=True))
        applied_nms_iou = float(block_bundle.get("cross_tile_nms").value)

        # Reset from the calibration-time floor to the real, calibrated point (full-frame cap
        # committed to None below, never the block bundle's own band-scoped density-derived one).
        from tcip_mcp.pipelines.operating_point import set_detector_operating_point

        predictor.score_threshold = conf
        set_detector_operating_point(predictor.model, score_thresh=conf,
                                     detections_per_img=applied_max_dets)
        predictor.max_dets = None

        op_bundle = block_calibrated_export_operating_point(
            block_bundle, trait=trait, tile_size=resolved_tile,
            tile_size_source=tile_size_source, tile_size_derived_from=tile_size_derived_from)
        op_provenance = op_bundle.to_provenance()["operating_point"]

        tile_ref = tile_size_gate_flag(op_provenance)
        gate_flags = {"claim_scope": claim_scope_flag}
        if tile_ref is not None:
            gate_flags["tile_size"] = tile_ref
        gate = check_delivery_gate(gate_flags, acknowledge_unvalidated=acknowledge_unvalidated)
        if not gate.ok:
            reason = gate.reason if claim_scope_mismatch is None else (
                f"{gate.reason} {claim_scope_mismatch}")
            return {"error": reason, "tile_size_validated": tile_ref,
                    "claim_scope_validated": claim_scope_flag}
        tile_size_validated = gate.stamp.get("tile_size")
        claim_scope_validated = gate.stamp.get("claim_scope")
        conf_source = "block_calibration"

        # The count claim's own gate, run before the always-expensive whole-mosaic pass.
        if (op_bundle.is_shippable and claim_scope_validated != VALIDATED_FALSE
                and tile_size_validated != VALIDATED_FALSE):
            if bucket_root is None:
                logger.warning(_NO_DATASET_ROOT_NOTE.format(bucket=out))
            else:
                try:
                    draft = _open_count_claim(
                        block_evidence, trait=trait, checkpoint_sha256=identity["sha256"],
                        producing_experiment_id=identity["experiment_id"],
                        dataset_root=bucket_root)
                except ValueError as exc:
                    return {"error": f"the count claim for trait {trait!r} was not earned: {exc}"}
    else:
        # Always tiled (a raster too large to load whole has no untiled alternative); every input
        # the gate needs is already resolved, so it runs here, before the expensive raster pass.
        op_bundle = raw_operating_point(
            conf=applied_conf, cross_tile_nms=applied_nms_iou, tiled=True,
            tile_size=resolved_tile, max_dets=applied_max_dets,
            tile_size_source=tile_size_source, tile_size_derived_from=tile_size_derived_from,
            tiled_source="default",
            conf_stated=raw_conf_stated, max_dets_stated=raw_max_dets_stated,
        )
        op_provenance = op_bundle.to_provenance()["operating_point"]

        tile_ref = tile_size_gate_flag(op_provenance)
        tile_flags = {"tile_size": tile_ref} if tile_ref is not None else {}
        gate = check_delivery_gate(tile_flags, acknowledge_unvalidated=acknowledge_unvalidated)
        if not gate.ok:
            return {"error": gate.reason, "tile_size_validated": tile_ref}
        tile_size_validated = gate.stamp.get("tile_size")

    from tcip_mcp.pipelines.raster_source import open_raster

    # The model's own in_chans is the channel routing hint; the reader's real band count is
    # checked against it inside predict_tiled before any tile is read.
    with open_raster(raster_path, predictor.in_chans) as reader:
        result = predictor.predict_tiled(
            reader, tile_size=resolved_tile, overlap=resolved_overlap,
            tile_batch_size=tile_batch_size, global_nms_iou=applied_nms_iou,
            postprocess=postprocess, require_masks=require_masks, source_label=str(raster_path),
            tile_resize=tile_resize,
        )

    # No images_dir for a raster source: a foreign checkpoint with no recorded id_map decodes to
    # the raw 0-indexed id as its name, the same honest degraded fallback documented elsewhere.
    id_map = resolve_decode_id_map(predictor, None)

    from datetime import datetime, timezone

    out.mkdir(parents=True, exist_ok=True)
    sha = identity["sha256"]
    producer = prediction_producer(checkpoint_path, sha)
    pred_path = out / f"{Path(raster_path).stem}.json"
    # Read before the write: a drop can empty a mask list that was genuinely there, and has_masks
    # must reflect what this run used, not what happened to survive the drop.
    has_masks = bool(result.get("masks"))
    dropped_boxes = write_predictions_json(pred_path, result, created_by=producer, id_map=id_map)

    produced_at = datetime.now(timezone.utc).isoformat()
    op_stamp = operating_point_stamp(
        op_provenance,
        validated=draft is not None,
        validated_by=None,
        tile_size_validated=tile_size_validated,
        shippable_issues=op_bundle.shippable_issues(),
        id_map=id_map,
        trait=op_bundle.trait or None,
        dataset_hash=op_bundle.dataset_hash,
        checkpoint=Path(checkpoint_path).stem,
        checkpoint_sha256=sha,
        experiment_id=identity["experiment_id"],
        images_dir=None,
        raster_path=str(raster_path),
        produced_at=produced_at,
    )
    if has_masks:
        op_stamp["mask_binarize"] = mask_binarize_provenance()
    if block_prov is not None:
        op_stamp["claim_scope_validated"] = claim_scope_validated
        # spatial_manifest is already carried on the training experiment's own split.json.
        op_stamp["block_calibration"] = {
            k: v for k, v in block_prov.items() if k != "spatial_manifest"
        }
    # Recorded for every run of this regime: a consumer resolving these boxes through a raster's
    # georeferencing has no other way to tell that raster is the one behind them.
    try:
        import dataclasses

        from tcip_mcp.pipelines.raster_source import content_identity

        export_identity = content_identity(raster_path, predictor.in_chans)
        op_stamp["raster_content_identity"] = dataclasses.asdict(export_identity)
    except Exception:
        logger.warning("export-time raster content identity could not be recorded", exc_info=True)
    op_stamp = _seal_and_stamp(out, op_stamp, draft)

    exp_id = identity["experiment_id"]
    if exp_id:
        try:
            from tcip_mcp.experiments import update_lineage

            update_lineage(exp_id, predictions=str(out))
        except Exception:
            logger.warning("could not link predictions into experiment lineage", exc_info=True)

    response = {
        "image_count": 1, "output_dir": str(out), "files": [str(pred_path)],
        "bucket_redirected": resolution.redirected,
        "requested_output_dir": str(out) if resolution.redirected else None,
        "operating_point": op_provenance,
        "validated": op_stamp["validated"],
        "tile_size_validated": tile_size_validated,
        "conf_source": conf_source,
        "checkpoint_sha256": sha,
        "experiment_id": exp_id,
        "tiles": result.get("tiles"),
        "verdict_guard_operative": bucket_root is not None,
        "dropped_nonpositive_boxes": dropped_boxes,
    }
    if bucket_root is None:
        response["note"] = _NO_DATASET_ROOT_NOTE.format(bucket=out)
    if block_prov is not None:
        response["claim_scope_validated"] = claim_scope_validated
        if claim_scope_mismatch is not None:
            response["claim_scope_note"] = claim_scope_mismatch
    return response


@mcp.tool()
@audited(scope_arg="output_dir", scope_via=resolve_output_path)
def export_predictions(
    checkpoint_path: str,
    images_dir: str | None = None,
    output_dir: str = "",
    conf_threshold: float | None = None,
    device: str | None = None,
    tile: bool | None = None,
    tile_size: int | None = None,
    overlap: float | None = None,
    tile_batch_size: int = 96,
    global_nms_iou: float | None = None,
    max_dets: int | None = None,
    postprocess: str = "nms",
    trait: str | None = None,
    calibration_labels_dir: str | None = None,
    calibration_images_dir: str | None = None,
    experiment_id: str | None = None,
    overwrite: bool = False,
    acknowledge_unvalidated: bool = False,
    raster_path: str | None = None,
    require_masks: bool = True,
) -> dict:
    """Run inference and save predictions as COCO/JSON prediction file(s).

    Two source regimes, sharing one bucket-resolution/immutability/gate/lineage contract so a
    breeder or agent has one door regardless of capture shape:

    - ``images_dir`` (an ordinary directory of per-image captures, the common case): routes through
      ``run_inference`` so this door resolves the same firewalled operating point (conf/NMS/tiling/
      max_dets) rather than building a bare predictor of its own, which would truncate the count at
      the framework default and ship labels with no provenance. Writes ``<stem>.json`` per image.
    - ``raster_path`` (a single raster, georeferenced or not, potentially too large to decode
      whole): sources tiles from the windowed raster layer instead
      (:func:`~tcip_mcp.pipelines.raster_source.open_raster`), always tiled (there is no untiled
      option for a raster too large to decode whole), and writes exactly one ``<raster stem>.json``
      prediction file (in full-raster pixel space), since there is no natural
      directory-of-per-plant-images shape for a whole-raster capture. ``calibration_labels_dir`` is
      not accepted with it (there is no separate labeled directory shape for one raster); ``trait``
      alone calibrates against the mosaic's own reserved regions instead
      (:func:`~tcip_mcp.pipelines.block_calibration.resolve_block_calibration_records`), when the
      checkpoint's own training experiment reserved one (``data.split.reserve_calibration_fraction``
      at training time). Without a reserved region, conf has no per-dataset calibration for this
      regime; a validated per-plant count is earned later, at delivery
      (``deliver_orthomosaic_plant_counts``).

    Provide exactly one of ``images_dir``/``raster_path``. Both regimes write the same
    ``operating_point.json`` stamp convention beside the prediction file(s), so downstream code that
    reads a bucket's sidecar generically needs no special case for which regime produced it.

    Neither regime's underlying pass (``run_inference``, or the raster pass) ever refuses on an
    unvalidated dimension on its own, each is the shared, honestly-stamped raw substrate this door
    builds on, the same contract an uncalibrated ``conf`` already has. This tool is the one that
    actually persists a prediction bucket other doors treat as ground truth, so it is where the
    refusal belongs: a tiled run whose tile_size has no real basis (no persisted training geometry,
    no recoverable native-frame edge, no explicit override) refuses to write here unless
    ``acknowledge_unvalidated=True``, the same
    gate ``tabulate_counts``/``compute_phenology``/the web results routes/``export_aggregated_csv``
    already apply, via the same shared :func:`tcip_mcp.pipelines.resolution.tile_size_gate_flag`.
    Both regimes gate before the (expensive) pass runs: the ``raster_path`` regime uses the
    predictor that pass then reuses; the ``images_dir`` regime sniffs the checkpoint's own stamped
    config (no weights load, never raises on a missing/unreadable checkpoint) to resolve the same
    geometry ``run_inference`` will, then re-checks against ``run_inference``'s own real result
    after the pass as the authoritative gate. An untiled run's tile_size is never operative and
    can't manufacture a refusal.

    A prediction bucket (``output_dir``) that already carries review verdicts is immutable: by
    default the export is redirected to a fresh run-scoped bucket (``<dir>@r2``, ``@r3``, next
    free) and the dir actually written is returned as ``output_dir``. Pass ``overwrite=True`` to
    write in place only when the bucket has zero verdicts; with verdicts present it is refused
    (error names the count and a suggested dir) so a re-run never orphans recorded verdicts. The
    verdicts consulted are the ones recorded in the bucket's own dataset, so a bucket written
    outside any dataset has no store to be guarded against: that export is written where it was
    asked for, ``verdict_guard_operative`` comes back false with a note saying so, and it is
    stamped unvalidated whatever its operating point cleared, since a count claim outside the
    dataset layout has no dataset-relative key a reader could locate it by.

    A bucket stamped validated names the validation record its claim was earned from. The gate for
    that record runs before any file is written, and the record is appended over the prediction
    files as they actually landed, so a run that dies partway leaves either predictions with no
    stamp or a record no stamp names, both of which floor to unvalidated at every delivery door.

    Args:
        checkpoint_path: Path to model .pt checkpoint.
        images_dir: Directory containing input images (mutually exclusive with ``raster_path``).
        output_dir: Directory for output .json prediction file(s). A relative path resolves
            against the project root, never the server process's cwd.
        conf_threshold: Minimum confidence score. ``None`` (default) states nothing and forwards
            that on, leaving the value to run at the platform default; a stated value is an
            explicit override even when it equals the platform default.
        device: Device to use.
        tile: Tiled (SAHI-style) inference for small dense objects (``images_dir`` regime only;
            ``raster_path`` is always tiled). ``None`` (default) forwards to ``run_inference``
            unresolved, see its own ``tile`` doc: a documented default distinct from an explicit
            choice, not silently ``False``. Works for ``instance_seg`` too; masks reach
            ``write_predictions_json`` either way, tiled or untiled, in whichever of the two
            coordinate shapes the producing pass produced.
        tile_size: Sliding-window tile edge (px).
        overlap: Fractional tile overlap.
        tile_batch_size: Tiles per forward batch.
        global_nms_iou: Cross-tile NMS IoU. ``None`` (default) states nothing and forwards that on,
            leaving the value to be derived; a stated value is an explicit override even when it
            equals the platform default. See ``run_inference``'s own doc.
        max_dets: Full-frame detection cap. ``None`` (default) states nothing and forwards that on,
            leaving the value to be derived; a stated value is an explicit override even when it
            equals the platform default. See ``run_inference``'s own doc.
        postprocess: Cross-tile merge, "nms" or "nmm".
        trait: Trait to calibrate the operating point per dataset. ``images_dir`` regime: with
            ``calibration_labels_dir``. ``raster_path`` regime: alone, against the checkpoint's own
            training mosaic's reserved calibration/test regions (requires
            ``data.split.reserve_calibration_fraction`` at training time; refuses by name
            otherwise).
        calibration_labels_dir: Labeled dir for calibrating + held-out validating the operating
            point (``images_dir`` regime only; not accepted with ``raster_path``, see ``trait``).
        calibration_images_dir: Images for the calibration labels (defaults to ``images_dir``).
        experiment_id: The run that produced the checkpoint, for provenance (forwarded to
            ``run_inference``; see its own doc for the best-effort resolution when omitted).
        overwrite: Write into ``output_dir`` even if it exists. Refused if the bucket has review
            verdicts; the default (False) auto-redirects to a fresh bucket instead.
        acknowledge_unvalidated: Write the bucket even when tile_size (a tiled run only) has no
            real basis, stamping ``tile_size_validated=false`` on the sidecar so the
            un-trustworthiness travels with it rather than writing silently.
        raster_path: A single raster, georeferenced or not, potentially too large to decode whole
            (mutually exclusive with ``images_dir``); see the regime description above.
        require_masks: Collect masks for an ``instance_seg`` checkpoint (``raster_path`` regime
            only; ignored for ``images_dir``, which always carries masks via ``run_inference``).
    """
    if not output_dir:
        return {"error": "output_dir is required"}
    if images_dir is None and raster_path is None:
        return {"error": "Provide either images_dir or raster_path"}
    if images_dir is not None and raster_path is not None:
        return {"error": "Provide only one of images_dir or raster_path, not both"}
    if raster_path is not None and calibration_labels_dir:
        return {"error": "calibration_labels_dir is not supported for a raster_path export: "
                         "block calibration (trait alone, see below) validates against the "
                         "mosaic's own reserved regions instead of a caller-supplied labeled dir."}
    if raster_path is not None:
        # The images_dir regime gets this for free from run_inference's own check; this regime
        # builds its predictor directly, so a missing file would otherwise raise uncaught.
        if not Path(checkpoint_path).is_file():
            return {"error": f"Checkpoint not found: {checkpoint_path}"}
        if not Path(raster_path).is_file():
            return {"error": f"raster_path not found: {raster_path}"}

    block_calibration_experiment_id = None
    if raster_path is not None and trait:
        from tcip_mcp.model_registry import resolve_model_identity
        from tcip_mcp.pipelines.block_calibration import reserved_calibration_region_available

        block_identity = resolve_model_identity(checkpoint_path, experiment_id=experiment_id)
        block_calibration_experiment_id = block_identity["experiment_id"]
        if block_calibration_experiment_id is None or not reserved_calibration_region_available(
            block_calibration_experiment_id
        ):
            return {"error": (
                "trait calibration for a raster_path export requires the checkpoint's own "
                "training experiment to have a spatial-strip split with a reserved calibration "
                "region (train it with data.split.reserve_calibration_fraction set); this "
                f"checkpoint's experiment_id ({block_calibration_experiment_id!r}) has none. "
                "Deliver a calibrated per-plant count via deliver_orthomosaic_plant_counts "
                "instead, or retrain with reserve_calibration_fraction set."
            )}

    # Resolve the writable bucket before the (expensive) inference so a verdict-blocked overwrite
    # fails fast.
    out, resolution, bucket_root, refusal = _resolve_writable_bucket_for(
        output_dir, overwrite=overwrite)
    if refusal is not None:
        return refusal

    if raster_path is not None:
        return _export_predictions_raster(
            checkpoint_path=checkpoint_path, raster_path=raster_path, out=out,
            resolution=resolution, device=device, conf_threshold=conf_threshold,
            tile_size=tile_size, overlap=overlap, tile_batch_size=tile_batch_size,
            global_nms_iou=global_nms_iou, max_dets=max_dets, postprocess=postprocess,
            require_masks=require_masks,
            experiment_id=block_calibration_experiment_id or experiment_id,
            acknowledge_unvalidated=acknowledge_unvalidated, trait=trait,
        )

    from tcip_mcp.pipelines.resolution import (
        check_delivery_gate, resolve_tile_size_param, tile_size_gate_flag,
    )

    # Gate before the (expensive) pass where possible, matching the raster_path regime: a light,
    # weights-free config sniff resolves the same tile geometry run_inference itself will.
    from types import SimpleNamespace

    from tcip_mcp.model_registry import read_checkpoint_data_config
    from tcip_mcp.pipelines.inference.predictor import (
        TileEdgeContradiction, explicit_edge_provenance, resolve_tile_regime,
    )

    data_cfg = read_checkpoint_data_config(checkpoint_path)
    tiling_cfg = data_cfg.get("tiling") or {}
    stub = SimpleNamespace(
        train_tile_size=tiling_cfg.get("tile_size"), train_overlap=tiling_cfg.get("overlap"),
        train_native_size=data_cfg.get("train_native_size"))
    pre_tiled = (stub.train_tile_size is not None) if tile is None else tile
    try:
        pre_tile, pre_tile_source, _pre_overlap, _pre_overlap_source, _pre_resize = resolve_tile_regime(
            stub, tiled=pre_tiled, tile_size=tile_size, overlap=overlap)
    except TileEdgeContradiction as exc:
        return {"error": str(exc)}
    pre_tile_derived_from = (
        explicit_edge_provenance(stub, pre_tile)
        if pre_tile_source == "explicit" and pre_tile is not None else None)
    pre_param = resolve_tile_size_param(
        pre_tile, tiled=pre_tiled, tile_size_source=pre_tile_source,
        tile_size_derived_from=pre_tile_derived_from)
    pre_tile_ref = tile_size_gate_flag({"tile_size": pre_param.to_provenance()})
    pre_gate = check_delivery_gate(
        {"tile_size": pre_tile_ref} if pre_tile_ref is not None else {},
        acknowledge_unvalidated=acknowledge_unvalidated)
    if not pre_gate.ok:
        return {"error": pre_gate.reason, "tile_size_validated": pre_tile_ref}

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

    # Re-checked against the real predictor's own resolution: the sniff above is an early
    # opt-out, this stays the authoritative gate.
    tile_ref = tile_size_gate_flag(result.get("operating_point"))
    tile_flags = {"tile_size": tile_ref} if tile_ref is not None else {}
    gate = check_delivery_gate(tile_flags, acknowledge_unvalidated=acknowledge_unvalidated)
    if not gate.ok:
        return {"error": gate.reason, "tile_size_validated": tile_ref}
    tile_size_validated = gate.stamp.get("tile_size")

    # The count claim's own gate, run before a single file is written.
    sha = result.get("checkpoint_sha256")
    draft, refusal = _draft_count_claim(
        result, trait=trait, bucket=out, dataset_root=bucket_root,
        tile_size_validated=tile_size_validated)
    if refusal is not None:
        return refusal

    exp_id = result.get("experiment_id")
    if exp_id:
        # Checked before the publisher writes the bucket: inference itself already ran and was
        # audited by this point, so this refuses only the file write that would follow it.
        from tcip_mcp.experiments import pointer_frozen

        frozen = pointer_frozen(exp_id, "lineage", "predictions", str(out))
        if frozen is not None:
            return {"error": frozen}

    written, dropped_boxes, op_stamp = _publish_image_predictions(
        out, result, checkpoint_path=checkpoint_path, trait=trait, images_dir=images_dir,
        tile_size_validated=tile_size_validated, draft=draft)

    # Close the data→model→predictions chain: link this bucket into the producing run's lineage.
    # Additive first-write, the terminal-state lock permits it into a still-empty predictions field.
    if exp_id:
        try:
            from tcip_mcp.experiments import update_lineage

            update_lineage(exp_id, predictions=str(out))
        except Exception:
            logger.warning("could not link predictions into experiment lineage", exc_info=True)

    response = {"image_count": len(written), "output_dir": str(out), "files": written,
                "bucket_redirected": resolution.redirected,
                "requested_output_dir": output_dir if resolution.redirected else None,
                "operating_point": result.get("operating_point"),
                "validated": op_stamp["validated"],
                "tile_size_validated": tile_size_validated,
                "conf_source": result.get("conf_source"),
                "checkpoint_sha256": sha,
                "experiment_id": exp_id,
                "verdict_guard_operative": bucket_root is not None,
                "dropped_nonpositive_boxes": dropped_boxes}
    if bucket_root is None:
        response["note"] = _NO_DATASET_ROOT_NOTE.format(bucket=out)
    # The run's warnings (a CPU-bound workload) belong on this door's own response too, otherwise
    # the reason a delivered bucket ran in the regime it did is visible only in the server log.
    if result.get("warning"):
        response["warning"] = result["warning"]
    return response


@mcp.tool()
@audited
def tabulate_counts(
    checkpoint_path: str,
    images_dir: str,
    output_path: str,
    trait: str,
    conf_threshold: float | None = None,
    device: str | None = None,
    tile: bool | None = None,
    tile_size: int | None = None,
    overlap: float | None = None,
    tile_batch_size: int = 96,
    global_nms_iou: float | None = None,
    max_dets: int | None = None,
    postprocess: str = "nms",
    calibration_labels_dir: str | None = None,
    calibration_images_dir: str | None = None,
    experiment_id: str | None = None,
    acknowledge_unvalidated: bool = False,
    predictions_dir: str | None = None,
) -> dict:
    """Run inference and export a CSV summary of detection counts per image.

    Routes through ``run_inference`` so the per-image counts resolve the same firewalled
    operating point (conf/NMS/tiling/max_dets) as ``run_inference``/``export_predictions``,
    the CSV is a count-bearing deliverable (the count is the phenotype for count traits), so it
    must not be produced at a different, untiled, truncating operating point.

    Delivery gate: the count is a phenotype, so the CSV is not written unless every gating dimension
    of the run's own resolved bundle is validated (not a caller string), or
    ``acknowledge_unvalidated=True`` writes a clearly-flagged provisional CSV stamped
    ``measurement_validated=false``. Calibrate per dataset (``trait`` + ``calibration_labels_dir``)
    to reach a validated conf. A tiled run's ``tile_size`` gates the same way, a run with no
    persisted training geometry, no recoverable native-frame edge, and no explicit caller override
    refuses here too (closing the asymmetry with ``run_full_frame_evaluation``, which already
    refuses outright for that same case); pass an explicit ``tile_size``, retrain with tile geometry
    persisted, or rely on a checkpoint's own uniform untiled training frame to reach a validated
    tile scale. An untiled run is never gated on tile_size at all.

    A validated CSV also needs the predictions its counts were read off to exist somewhere a
    reviewer can check them, so ``predictions_dir`` is the third gating dimension: without it the
    counts rest on one in-memory pass and nothing else, and the CSV can only be delivered
    provisionally. With it, this persists the predictions it counted into that bucket, stamps the
    bucket exactly as ``export_predictions`` does (the same bucket resolution, earning order and
    stamp), and reads the CSV's own validity back off that stamp.

    Meaning door: ``trait``'s per-image-count operationalization must be recorded and
    breeder-confirmed, and that is checked before the inference runs, not after. This function
    returns ``image_count`` and ``total_detections`` on its refusal paths as well as its success
    one, so a check placed after the pass would hand back the very numbers it refused to write.

    Args:
        checkpoint_path: Path to model .pt checkpoint.
        images_dir: Directory containing input images.
        output_path: Path for the output CSV file. A relative path resolves against the
            project root, never the server process's cwd.
        trait: The registered trait whose confirmed per-image-count operationalization this
            delivery rests on, and whose operating point is calibrated per dataset (with
            ``calibration_labels_dir``). Required.
        conf_threshold: Minimum confidence score. ``None`` (default) states nothing and forwards
            that on, leaving the value to run at the platform default; a stated value is an
            explicit override even when it equals the platform default.
        device: Device to use.
        tile: Tiled (SAHI-style) inference for small dense objects. ``None`` (default) forwards to
            ``run_inference`` unresolved, see its own ``tile`` doc: a documented default distinct
            from an explicit choice, not silently ``False``.
        tile_size: Sliding-window tile edge (px).
        overlap: Fractional tile overlap.
        tile_batch_size: Tiles per forward batch.
        global_nms_iou: Cross-tile NMS IoU. ``None`` (default) states nothing and forwards that on,
            leaving the value to be derived; a stated value is an explicit override even when it
            equals the platform default. See ``run_inference``'s own doc.
        max_dets: Full-frame detection cap. ``None`` (default) states nothing and forwards that on,
            leaving the value to be derived; a stated value is an explicit override even when it
            equals the platform default. See ``run_inference``'s own doc.
        postprocess: Cross-tile merge, "nms" or "nmm".
        calibration_labels_dir: Labeled dir for calibrating + held-out validating the operating point.
        calibration_images_dir: Images for the calibration labels (defaults to ``images_dir``).
        experiment_id: The run that produced the checkpoint, for provenance (forwarded to
            ``run_inference``; see its own doc for the best-effort resolution when omitted).
        acknowledge_unvalidated: Write the count CSV even when the operating point is unvalidated,
            stamping it ``measurement_validated=false`` so the un-trustworthiness travels downstream.
        predictions_dir: Directory to persist the counted predictions into, resolved and stamped
            the way ``export_predictions`` resolves and stamps a bucket (a relative path resolves
            against the project root; a bucket carrying review verdicts redirects to a fresh
            variant). Omitted, the CSV can only be delivered provisionally, since its counts would
            rest on no artifact anyone can re-read.
    """
    from tcip_mcp.operationalization import (
        PER_IMAGE_COUNT,
        check_operationalization,
        resolve_trait_and_record,
    )
    from tcip_mcp.project_paths import resolve_output_path
    from tcip_mcp.traits import TraitUnknownError

    output_path = str(resolve_output_path(output_path))
    # Ahead of the pass, so a refused delivery has no counts of its own to hand back.
    try:
        spec, record, _specs_dir = resolve_trait_and_record(trait, PER_IMAGE_COUNT)
    except TraitUnknownError as e:
        return {"error": str(e)}
    # A per_image_count delivery names no positive class, so check_operationalization ignores a
    # registry for this kind regardless of what one would resolve to.
    stated = check_operationalization(spec, record, PER_IMAGE_COUNT, registry=None)
    if not stated.ok:
        return {"error": stated.message}

    bucket = bucket_root = None
    resolution = None
    if predictions_dir is not None:
        bucket, resolution, bucket_root, refusal = _resolve_writable_bucket_for(
            predictions_dir, overwrite=False)
        if refusal is not None:
            return refusal

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
        accepted_references,
        check_delivery_gate,
        tile_size_gate_flag,
    )

    op = result.get("operating_point") or {}
    conf_prov = op.get("conf") or {}
    # The count operating point's validity is the reference the run's own conf param recorded, judged
    # against the references accepted for conf's own validation kind (annotations), never the bare
    # `validated` bool promoted to a reference the run never earned. That upgrade laundered a
    # missing/unrecognized/wrong-kind value into a shippable one (the same hole
    # resolution._sidecar_reference closed).
    op_ref = conf_prov.get("validated_against")
    if op_ref not in accepted_references("annotations"):
        op_ref = VALIDATED_FALSE
    flags = {"operating_point": op_ref}

    # tile_size gates the same way (closing the asymmetry with run_full_frame_evaluation): a
    # no-basis tile scale is as untrustworthy for a count as an uncalibrated conf; None if untiled.
    tile_ref = tile_size_gate_flag(op)
    if tile_ref is not None:
        flags["tile_size"] = tile_ref

    # With no bucket persisted there is nothing behind the count, so this dimension floors.
    if bucket is None:
        flags["persisted_predictions"] = VALIDATED_FALSE

    gate = check_delivery_gate(flags, acknowledge_unvalidated=acknowledge_unvalidated)
    if not gate.ok:
        reason = gate.reason
        if bucket is None:
            reason += (
                " These counts were read off an in-memory pass with no prediction bucket behind "
                "them: pass predictions_dir to persist and stamp the predictions they came from, "
                "which is what a validated count CSV rests on."
            )
        return {
            "error": reason,
            "operating_point_validated": op_ref,
            "tile_size_validated": tile_ref,
            "operating_point": result.get("operating_point"),
            "validated": False,
            "image_count": result["image_count"],
            "total_detections": result["total_detections"],
        }

    dropped_boxes = 0
    if bucket is not None:
        draft, refusal = _draft_count_claim(
            result, trait=trait, bucket=bucket, dataset_root=bucket_root,
            tile_size_validated=gate.stamp.get("tile_size"))
        if refusal is not None:
            return refusal
        _, dropped_boxes, _ = _publish_image_predictions(
            bucket, result, checkpoint_path=checkpoint_path, trait=trait, images_dir=images_dir,
            tile_size_validated=gate.stamp.get("tile_size"), draft=draft)

    provenance = {
        "producer_model_sha256": result.get("checkpoint_sha256"),
        "experiment_id": result.get("experiment_id"),
        "operating_point_conf": op.get("conf"),
        "produced_at": result.get("produced_at"),
    }
    # Read back off the bucket's own stamp, the same reconciliation the other bucket doors perform.
    csv_measurement_validated = gate.column_stamp("operating_point")
    csv_path = export_detection_csv(
        result["results"], output_path, provenance=provenance, trait=trait,
        measurement_validated=csv_measurement_validated,
        pred_dirs=[str(bucket)] if bucket is not None else None,
        acknowledge_unvalidated=acknowledge_unvalidated,
    )
    # This response carries the counts too, so it needs the proof at the end that the write did.
    spec_now, record_now, _ = resolve_trait_and_record(trait, PER_IMAGE_COUNT)
    still_stated = check_operationalization(
        spec_now, record_now, PER_IMAGE_COUNT, registry=None, basis=stated.basis)
    if not still_stated.ok:
        return {"error": still_stated.message}
    out = {
        "csv_path": csv_path,
        "image_count": result["image_count"],
        "total_detections": result["total_detections"],
        # Carry the operating point + producing model that produced these counts, the CSV is a
        # count-bearing deliverable; the numbers are only as trustworthy as what stands behind them.
        "operating_point": result.get("operating_point"),
        "validated": bool(result.get("validated", False)),
        "operating_point_validated": gate.stamp["operating_point"],
        "tile_size_validated": gate.stamp.get("tile_size"),
        "conf_source": result.get("conf_source"),
        "checkpoint_sha256": result.get("checkpoint_sha256"),
        "experiment_id": result.get("experiment_id"),
        "predictions_dir": str(bucket) if bucket is not None else None,
    }
    if bucket is not None:
        out["bucket_redirected"] = resolution.redirected
        out["verdict_guard_operative"] = bucket_root is not None
        out["dropped_nonpositive_boxes"] = dropped_boxes
        if bucket_root is None:
            out["note"] = _NO_DATASET_ROOT_NOTE.format(bucket=bucket)
    # run_inference's own warnings (a CPU-bound workload) are surfaced here too, so a count CSV
    # never ships with the regime it ran in disclosed only in the server log.
    if result.get("warning"):
        out["warning"] = result["warning"]
    return out
