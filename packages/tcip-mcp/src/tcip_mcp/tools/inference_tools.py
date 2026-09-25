"""Inference MCP tools: run_inference and deliver_per_image_counts."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from tcip_annotation.json_io import SIDECAR_FILENAMES
from tcip_store import (
    RECORD_JSON, BadKey, Key, StoreDescriptor, StoreError, Version, VersionConflict,
    register_store, store,
)
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited
from tcip_mcp.dataset_layout import bucket_dataset_root, label_filename
from tcip_mcp.pipelines.data.splits import DEFAULT_CAL_SEED, DEFAULT_HOLDOUT_RATIO, same_directory
from tcip_mcp.pipelines.postprocessing.export import (
    export_detection_csv,
    mask_binarize_provenance,
    positive_detections,
    unmapped_label_ids,
    write_predictions_json,
)
from tcip_mcp.pipelines.resolution import (
    DEFAULT_POSTPROCESS, DEFAULT_TILE_BATCH_SIZE, DeliveryRefused, applied_operating_point,
)
from tcip_mcp.project_paths import resolve_output_path

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.resolution import Acknowledgment

logger = logging.getLogger(__name__)

_CALIBRATION_CURVE_DIR = (".tcip", "artifacts")
_CALIBRATION_CURVE_STEM = "operating_point_sweep_"  # frozen on-disk prefix, the locator's own path contract


class _CalibrationCurveLocator:
    """One conf curve record per name, named for the digest of the record's own bytes
    (:func:`calibration_curve_identity`). Under ``last_writer_wins``, only a byte-identical rerun
    replaces what is already there.
    """

    def relative_path(self, scope: str, parts: tuple[str, ...]) -> PurePosixPath:
        (body_hash,) = parts
        return PurePosixPath(*_CALIBRATION_CURVE_DIR, f"{_CALIBRATION_CURVE_STEM}{body_hash}.json")

    def parts_from(self, relative_path: PurePosixPath) -> tuple[str, ...] | None:
        segments = relative_path.parts
        if (segments[:len(_CALIBRATION_CURVE_DIR)] != _CALIBRATION_CURVE_DIR
                or len(segments) != len(_CALIBRATION_CURVE_DIR) + 1):
            return None
        name = segments[-1]
        if not name.startswith(_CALIBRATION_CURVE_STEM) or not name.endswith(".json"):
            return None
        return (name[len(_CALIBRATION_CURVE_STEM):-len(".json")],)


CONFIDENCE_SWEEP_STORE = "confidence_sweep"  # frozen store name: the database backend keys existing rows by it
register_store(
    StoreDescriptor(
        name=CONFIDENCE_SWEEP_STORE,
        kind="record",
        key_fields=("record_digest",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        locator=_CalibrationCurveLocator(),
    )
)


def calibration_curve_key(record_digest: str) -> Key:
    """The full calibration curve one calibration produced.

    ``record_digest`` is :func:`calibration_curve_identity` over the record's own bytes, not a
    digest of the run's inputs alone. ``last_writer_wins``: only a byte-identical rerun replaces
    the record already under this key; a differing calibration keys a new record.
    """
    if PurePosixPath(record_digest).name != record_digest or record_digest in ("", ".", ".."):
        raise BadKey(
            f"curve identity {record_digest!r} is not a single name: an identity carrying a path "
            "separator would address a record outside the artifact store"
        )
    from tcip_mcp.project_paths import platform_state_root

    return Key(CONFIDENCE_SWEEP_STORE, str(platform_state_root().resolve()), (record_digest,))


def calibration_curve_identity(body: dict) -> str:
    """The sha256 identity of a confidence_sweep record's whole body, over the exact stored bytes
    (``RECORD_JSON.encode``); all sixty-four hex characters.
    """
    from tcip_mcp.model_registry import _sha256_of_bytes

    return _sha256_of_bytes(RECORD_JSON.encode(body))


def keep_calibration_curve(body: dict) -> str:
    """Keep one calibration's curve under its own identity and return that identity.

    The key is the body's own digest and the write is create-only, so a record already under it
    is these same bytes: the write conflicts, writes nothing and leaves no line, and of two
    concurrent first writes one lands. A write leaves its one ``calibration_curve_written`` line,
    whatever the calling door does next. Raises what the identity or the store raises.
    """
    from tcip_mcp.audit import record_event_or_raise

    identity = calibration_curve_identity(body)
    try:
        store.replace(calibration_curve_key(identity), body, expect=Version.ABSENT)
    except VersionConflict:
        return identity
    record_event_or_raise("calibration_curve_written", {"calibration_evidence_key": identity})
    return identity


RASTER_PASS_PROGRESS_STORE = "raster_pass_progress"
_RASTER_PASS_PROGRESS_LOCATOR = RootedFileLocator(
    prefix=(".tcip", "raster_pass_progress"), suffix=".json")
register_store(
    StoreDescriptor(
        name=RASTER_PASS_PROGRESS_STORE,
        kind="record",
        key_fields=("segment",),
        frozen=False,
        codec=RECORD_JSON,
        concurrency="cas",
        enumerable=True,
        locator=_RASTER_PASS_PROGRESS_LOCATOR,
    )
)
"""One tiled raster pass' resume state: an ``identity`` record naming the pass a bucket is mid-way
through, plus one ``batch-<index>`` record per tile batch already reconstructed. Sits under
``<bucket>/.tcip/raster_pass_progress/``, outside ``prediction_documents``' own non-recursive glob
of the bucket root, so an interrupted pass' own progress never reads as a prediction document.
Scratch, never a state that needs a receipt: written, resumed from and deleted with no audit line,
since what a pass leaves is its published bucket, which the publication records. Not frozen: its shape may still move, so a reader checks the identity record's own ``schema_version`` by
hand (``_RASTER_PASS_PROGRESS_SCHEMA_VERSION``) rather than relying on the seam, which only enforces
that ceiling for a store declared frozen."""

_RASTER_PASS_PROGRESS_SCHEMA_VERSION = 1


def run_scope(predictor) -> "ClassScope":
    """A run's own recorded class space (:class:`~tcip_mcp.pipelines.data.selection.ClassScope`):
    its subject, its attribute and the name->id map its loaders read targets under, read through
    :meth:`~tcip_mcp.pipelines.data.selection.ClassScope.recorded_in`. Refuses by name a run that
    declares an attribute with no subject.
    """
    from tcip_mcp.pipelines.data.selection import ClassScope

    scope = ClassScope.recorded_in((getattr(predictor, "config", {}) or {}).get("data") or {})
    if scope.attribute is not None and scope.subject is None:
        raise ValueError(
            f"{getattr(predictor, 'path', predictor)!r} declares attribute {scope.attribute!r} "
            "with no subject: a value with no object class names nothing a reader could hold "
            "predictions to. Retrain with data.subject stated beside data.attribute."
        )
    return scope


def unmapped_classified_run(
    scope: "ClassScope", id_map: dict | None, *, images_dir: str | None,
) -> str | None:
    """The composed refusal for a run that declared an attribute and resolved no ``id_map`` to
    decode predictions with, or ``None`` when there is nothing to refuse (a mapped run, or a run
    with no attribute at all).

    An ``images_dir`` whose dataset holds no ``subjects.json`` is told to run
    ``write_subject_registry`` for that dataset; a call with no ``images_dir`` at all is told to
    pass one.
    """
    if id_map is not None:
        return None
    attribute = scope.attribute
    if attribute is None:
        return None
    subject = scope.subject
    if images_dir is None:
        return (
            f"this run decoded along attribute {attribute!r} of subject {subject!r} from a "
            "registry-derived dataset, but no images_dir was given to read the decoding "
            "dataset's subjects.json from. Pass images_dir naming the dataset whose "
            "subjects.json decodes this run."
        )
    return (
        f"this run decoded along attribute {attribute!r} of subject {subject!r} from a "
        f"registry-derived dataset, but {images_dir!r} holds no subjects.json to decode it "
        "with. Run write_subject_registry for that dataset, then retry."
    )


def resolve_decode_id_map(predictor, images_dir: str | None, *,
                          scope: tuple[str | None, str | None] | None = None) -> dict | None:
    """This run's name->id map for recording + decoding predictions.

    The training run's own recorded map (``config["data"]["id_map"]``) when present; otherwise one
    derived from the live registry of ``images_dir``'s dataset for ``scope``, or ``None`` when the
    scope names no subject, no ``images_dir`` is given, or an attribute scope finds no
    ``subjects.json``. A registry read that fails for a real reason (corrupted file, an id-space
    mismatch) propagates.

    ``scope`` is the ``(subject, attribute)`` the registry derivation reads, defaulting to the
    predictor's own recorded training scope.
    """
    recorded = run_scope(predictor)
    if recorded.id_map is not None:
        return recorded.id_map
    subject, attribute = scope if scope is not None else (recorded.subject, recorded.attribute)
    if not (subject and images_dir):
        return None

    from tcip_mcp.pipelines.data.label_queries import resolve_registry_id_map, resolved_subjects_path

    # Precondition, not a broad except: an attribute-scoped run with no subjects.json here returns
    # None so unmapped_classified_run composes the write_subject_registry remedy, rather than crashing.
    if attribute is not None and resolved_subjects_path(images_dir) is None:
        return None
    _reg, id_map = resolve_registry_id_map(images_dir, subject, attribute)
    return id_map


@mcp.tool()
def run_inference(
    checkpoint_path: str,
    images_dir: str | None = None,
    raster_path: str | None = None,
    output_dir: str = "",
    conf_threshold: float | None = None,
    device: str | None = None,
    tile: bool | None = None,
    tile_size: int | None = None,
    overlap: float | None = None,
    tile_batch_size: int = DEFAULT_TILE_BATCH_SIZE,
    global_nms_iou: float | None = None,
    max_dets: int | None = None,
    postprocess: str = DEFAULT_POSTPROCESS,
    dry_run: bool = False,
    trait: str | None = None,
    calibration_labels_dir: str | None = None,
    calibration_images_dir: str | None = None,
    selection_dir: str | None = None,
    experiment_id: str | None = None,
    group_by: str | None = None,
    group_key_map: dict[str, str] | None = None,
    split_seed: int = DEFAULT_CAL_SEED,
    split_holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
    overwrite: bool = False,
    allow_unvalidated_staging: bool = False,
    require_masks: bool = True,
    resume: bool = False,
) -> dict:
    """Run a trained model over images or a raster, and persist the predictions as a bucket.

    Provide exactly one of ``images_dir`` (an ordinary directory of per-image captures) or
    ``raster_path`` (a single raster, georeferenced or not, potentially too large to decode whole).
    Every call publishes a prediction bucket at ``output_dir``, stamped with the operating point
    behind the files on disk.

    A prediction bucket is the directory these writes persist into: its identity is that
    directory's own path (relative to the dataset root when it sits under one, its own resolved
    path otherwise; see :func:`~tcip_mcp.prediction_buckets.bucket_key_of`), and a bucket under a
    dataset root turns immutable the moment a human review verdict lands on any image inside it.
    Two source regimes:

    - ``images_dir``: writes ``<stem>.json`` per image. Works for ``instance_seg`` too; each tiled
      result's ``masks`` (see ``GenericPredictor.predict_tiled``) are a tile-local patch plus its
      full-image-space offset, never the untiled path's dense full-image array.
    - ``raster_path``: sources tiles from the windowed raster layer instead
      (:func:`~tcip_mcp.pipelines.raster_source.open_raster`), always tiled, and writes exactly one
      ``<raster stem>.json`` prediction file (in full-raster pixel space).
      ``calibration_labels_dir``/``selection_dir`` are not accepted with it; ``trait`` alone
      calibrates against the mosaic's own reserved regions instead
      (:func:`~tcip_mcp.pipelines.block_calibration.resolve_block_calibration_records`), when the
      checkpoint's own training experiment reserved one
      (``data.split.reserve_calibration_fraction`` at training time). Without a reserved region,
      conf has no per-dataset calibration for this regime; a validated per-plant count is earned at
      delivery (``deliver_orthomosaic_plant_counts``). ``resume`` (below) applies only here.

    Both regimes write the same ``operating_point.json`` stamp convention beside the prediction
    file(s).

    A tiled run whose tile_size has no real basis (no persisted training geometry, no recoverable
    native-frame edge, no explicit override) refuses to write unless
    ``allow_unvalidated_staging=True``, the staging escape that clears only
    ``tile_size``/``claim_scope`` (:data:`tcip_mcp.pipelines.resolution.STAGING_DIMENSIONS`), never
    a phenotype's own delivered dimensions. The gate runs before any image is predicted. An untiled
    run's tile_size is never operative.

    A prediction bucket (``output_dir``) that already carries review verdicts is immutable: by
    default the write is redirected to a fresh run-scoped bucket (``<dir>@r2``, ``@r3``, next free)
    and the dir actually written is returned as ``output_dir``. Pass ``overwrite=True`` to write in
    place only when the bucket has zero verdicts; with verdicts present it is refused (error names
    the count and a suggested dir). The verdicts consulted are the ones recorded in the bucket's
    own dataset, so a bucket written outside any dataset has no store to be guarded against: that
    write lands where it was asked for, ``verdict_guard_operative`` comes back false with a note
    saying so, and it is stamped unvalidated whatever its operating point cleared.

    A bucket that already holds a prediction document from an earlier publish, with no verdict yet
    recorded against it, refuses this write outright (error names the document count and a
    suggested fresh bucket), whatever ``overwrite`` says; two runs racing into one bucket after
    both resolve it clean are not guarded. A terminal (completed or failed) experiment's bucket the
    lineage pointer already locks is unreachable through the suggested bucket too;
    ``clear_prediction_bucket`` is the audited remedy that clears one for republication, moving it
    into a dated archive.

    A bucket stamped validated names the validation record its claim was earned from. The gate for
    that record runs before any file is written, and the record is appended over the prediction
    files as they actually landed.

    Args:
        checkpoint_path: Path to model .pt checkpoint. Must be registered under this process's
            platform state root (``register_model``, explicit mode for a foreign or bespoke
            checkpoint) or this door refuses before loading it.
        images_dir: Directory containing input images (mutually exclusive with ``raster_path``).
        raster_path: A single raster, georeferenced or not, potentially too large to decode whole
            (mutually exclusive with ``images_dir``).
        output_dir: Directory for output .json prediction file(s). Required, including for
            ``dry_run``, which names the bucket the write would resolve to without writing it. A
            relative path resolves against the platform state root, never the server process's cwd.
        conf_threshold: Minimum confidence score. ``None`` (default) runs at the platform default,
            stamped ``"default"``; a stated value is stamped as an explicit override, including
            when it equals the platform default.
        device: Device to use ('cuda' or 'cpu').
        tile: Enable tiled (SAHI-style) detection inference (``images_dir`` regime only;
            ``raster_path`` is always tiled). ``None`` (default) derives it from the checkpoint's
            own training tile geometry (``predictor.train_tile_size is not None``); its provenance
            is stamped ``"default"`` vs ``"explicit"``.
        tile_size: Sliding-window tile edge (px). ``None`` (default) derives it from the
            checkpoint's training tile geometry; a value overrides. A checkpoint that trained
            untiled on frames that all shared one square size derives the edge from that frame
            instead (``"native_ratio"``), each tile run through the resize that run's own
            augmentation config recorded. A checkpoint with none of those has no basis to tile at:
            a tiled run with no resolvable ``tile_size`` refuses, naming the missing basis.
        overlap: Fractional tile overlap (stride = tile_size*(1-overlap)). ``None`` derives from
            the checkpoint (else 0.2).
        tile_batch_size: Tiles per forward batch.
        global_nms_iou: Cross-tile global NMS IoU threshold. ``None`` (default): a calibrated run
            derives it from the calibration GT's own neighbor-IoU distribution; a stated value is
            stamped as an explicit override.
        max_dets: Full-frame detection cap (after any tiled merge). ``None`` (default): a
            calibrated run derives it from the calibration GT's own object density; a stated value
            is stamped as an explicit override.
        postprocess: Cross-tile merge, "nms" suppresses overlaps, "nmm" unions boxes split across a
            tile seam.
        dry_run: Report the effective operating point (conf/tiling/max_dets/postprocess) and the
            bucket the write would resolve to, without loading the model or running inference;
            previews the same bucket refusal a real call would hit.
        trait: Trait name to derive the confidence operating point per dataset. ``images_dir``
            regime: with ``calibration_labels_dir``. ``raster_path`` regime: alone, against the
            checkpoint's own training mosaic's reserved calibration/test regions (requires
            ``data.split.reserve_calibration_fraction`` at training time; refuses by name
            otherwise). Absent -> the raw path (conf=score_threshold, unvalidated).
        calibration_labels_dir: Labeled dir for a disjoint cal/holdout split to calibrate +
            held-out validate the operating point (``images_dir`` regime only). Its GT identity
            scopes the resolved conf.
        calibration_images_dir: Images for the calibration labels (defaults to ``images_dir``).
        selection_dir: ``images_dir`` regime only. Restrict the calibration universe to a
            selection's ``calibration`` samples under ``calibration_labels_dir``
            (``pipelines.data.selection.read_selection``). A checkpoint bound to a different
            selection than the one named here is refused by name. The scope is the selection's own.
            The response carries ``n_excluded_training_stems`` and ``n_excluded_validation_stems``,
            the members the selection put elsewhere under this scope, beside
            ``n_excluded_incomplete_attribute``.
        experiment_id: The run that produced the checkpoint, by its record id
            (``tcip_mcp.experiments``), for provenance. Resolved best-effort (checkpoint's own
            stamp, then the registry) when omitted. A known run whose training split can't be read
            fails calibration's train-disjointness check.
        group_by: ``images_dir`` regime only. Grouping policy for the locked calibration/holdout
            split, ``"tile_prefix"`` or ``"stem"``; ignored when ``group_key_map`` is given.
            ``None`` resolves to ``"tile_prefix"`` without a ``selection_dir``; a value beside
            ``selection_dir`` refuses, naming both. Only the first calibration call for a given
            calibration-labels identity draws the split (see ``redraw_calibration_holdout``).
        group_key_map: ``images_dir`` regime only. An agent-derived ``{stem: group_key}`` map
            overriding ``group_by``; must cover every stem in ``calibration_labels_dir``. Conflicts
            with ``selection_dir`` the same way ``group_by`` does.
        split_seed: ``images_dir`` regime only. Split seed for the locked calibration/holdout
            split, first-call-only like ``group_by``; a later call's divergence from the lock is
            reported in ``gate_evidence_summary``/the resolved bundle.
        split_holdout_ratio: ``images_dir`` regime only. Calibration/holdout fraction for the
            locked split, same first-call-only semantics as ``split_seed``.
        overwrite: Write into ``output_dir`` even if it exists. Refused if the bucket has review
            verdicts; the default (False) auto-redirects to a fresh bucket instead. Never overrides
            the document refusal above. For a ``raster_path`` pass, also discards progress another
            pass recorded in a bucket that holds no document yet. Conflicts with ``resume=True``
            (refused by name). A bucket holding both a document and a progress record refuses on
            the document.
        allow_unvalidated_staging: Write the bucket even when tile_size (a tiled run only) has no
            real basis, stamping ``tile_size_validated=false`` on the sidecar. Clears only the
            staging dimensions (tile_size, claim_scope).
        require_masks: Collect masks for an ``instance_seg`` checkpoint (``raster_path`` regime
            only; ignored for ``images_dir``, which always carries masks).
        resume: ``raster_path`` regime only (refuses with ``images_dir``, and with
            ``overwrite=True``). Continue a raster pass this bucket carries progress from. The
            recorded pass' checkpoint, raster content, trait, experiment and tile batch size must
            all match this call's own, or it refuses naming what differs; the resumed pass then
            runs the remaining tiles at the recorded operating point (a block-calibrated pass
            applies the recorded conf/cross_tile_nms directly). Refuses when the bucket carries no
            progress, when the recorded progress is a schema version newer than this reader knows,
            and for a mask-bearing (``instance_seg`` with ``require_masks``) pass. See
            :func:`_export_predictions_raster`.
    """
    if not Path(checkpoint_path).is_file():
        return {"error": f"Checkpoint not found: {checkpoint_path}"}
    if not output_dir:
        return {"error": "output_dir is required"}

    # Every check below refuses on the call's own shape ahead of dry_run's preview, so a preview
    # previews the same refusal a real call would hit; "provide either" runs after dry_run instead.
    if images_dir is not None and raster_path is not None:
        return {"error": "Provide only one of images_dir or raster_path, not both"}
    if raster_path is not None and calibration_labels_dir:
        return {"error": "calibration_labels_dir is not supported for a raster_path export: "
                         "block calibration (trait alone, see below) validates against the "
                         "mosaic's own reserved regions instead of a caller-supplied labeled dir."}
    if raster_path is not None and selection_dir:
        return {"error": "selection_dir is not supported for a raster_path export: block "
                         "calibration draws no selection universe, so it would be silently "
                         "dropped rather than scoping anything."}
    if selection_dir and not calibration_labels_dir:
        return {"error": "selection_dir requires calibration_labels_dir: it scopes a "
                         "calibration this call has no trait/calibration_labels_dir to run, so "
                         "the selection would be silently dropped rather than bounding one."}
    if raster_path is not None and not Path(raster_path).is_file():
        return {"error": f"raster_path not found: {raster_path}"}
    if resume and images_dir is not None:
        return {"error": "resume=True only applies to the raster_path regime: the images_dir "
                         "regime has no resume, its all-at-the-end write is unchanged."}
    if resume and overwrite:
        return {"error": "resume=True and overwrite=True conflict: overwrite discards a bucket's "
                         "recorded progress and starts over, resume continues it. Pick one."}

    if dry_run:
        # No model load here: an unset ``tile`` is a pending derivation, not a fabricated default.
        # A preview needs no images_dir/raster_path: those are about the pass, not this preview.
        applied_conf, applied_nms_iou, applied_max_dets = applied_operating_point(
            conf_threshold, global_nms_iou, max_dets)
        if tile is None:
            tiled_dry: bool | str = "pending-checkpoint-derivation"
            tiled_source_dry = "pending-checkpoint-derivation"
            cross_tile_nms_dry: float | None | str = "pending-checkpoint-derivation"
        else:
            tiled_dry, tiled_source_dry = tile, "explicit"
            cross_tile_nms_dry = applied_nms_iou if tile else None
        out_preview, resolution_preview, _bucket_root_preview, refusal_preview = (
            _resolve_writable_bucket_for(output_dir, overwrite=overwrite))
        if refusal_preview is not None:
            return refusal_preview
        return {
            "dry_run": True,
            "checkpoint_path": checkpoint_path,
            "output_dir": str(out_preview),
            "bucket_redirected": resolution_preview.redirected,
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

    if images_dir is None and raster_path is None:
        return {"error": "Provide either images_dir or raster_path"}

    # Resolve the writable bucket before the checkpoint is read: a verdict-blocked overwrite must
    # still refuse before the file is touched at all.
    out, resolution, bucket_root, refusal = _resolve_writable_bucket_for(
        output_dir, overwrite=overwrite)
    if refusal is not None:
        return refusal

    from tcip_mcp.model_registry import UnregisteredCheckpoint, load_registered_checkpoint

    try:
        checkpoint = load_registered_checkpoint(checkpoint_path)
    except UnregisteredCheckpoint as exc:
        return {"error": str(exc)}

    block_calibration_experiment_id = None
    if raster_path is not None and trait:
        from tcip_mcp.model_registry import resolve_model_identity
        from tcip_mcp.pipelines.block_calibration import reserved_calibration_region_available

        block_identity = resolve_model_identity(checkpoint, experiment_id=experiment_id)
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

    if raster_path is not None:
        return _export_predictions_raster(
            checkpoint=checkpoint, raster_path=raster_path, out=out, resolution=resolution,
            output_dir=output_dir, dataset_root=bucket_root, device=device,
            conf_threshold=conf_threshold, tile_size=tile_size, overlap=overlap,
            tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou, max_dets=max_dets,
            postprocess=postprocess, require_masks=require_masks,
            experiment_id=block_calibration_experiment_id or experiment_id,
            allow_unvalidated_staging=allow_unvalidated_staging, trait=trait,
            resume=resume, overwrite=overwrite,
        )

    result = _run_inference_verified(
        checkpoint, images_dir=images_dir, conf_threshold=conf_threshold,
        device=device, tile=tile, tile_size=tile_size, overlap=overlap,
        tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou, max_dets=max_dets,
        postprocess=postprocess, trait=trait,
        calibration_labels_dir=calibration_labels_dir, calibration_images_dir=calibration_images_dir,
        selection_dir=selection_dir, experiment_id=experiment_id,
        group_by=group_by, group_key_map=group_key_map, split_seed=split_seed,
        split_holdout_ratio=split_holdout_ratio,
    )
    if "error" in result:
        return result

    pub = publish_bucket(
        result, out=out, trait=trait, dataset_root=bucket_root,
        allow_unvalidated_staging=allow_unvalidated_staging)
    if pub["refusal"] is not None:
        return pub["refusal"]
    counted = _bucket_csv_rows(out, pub["op_stamp"])
    # Every field the pass returned, minus ``results``, overlaid with what the write earned.
    response = {k: v for k, v in result.items() if k != "results"}
    response.update({
        **_bucket_response(pub, out=out, resolution=resolution, dataset_root=bucket_root,
                           requested_output_dir=output_dir),
        "image_count": len(counted), "total_detections": sum(r["count"] for r in counted),
    })
    return response


def _run_inference_verified(
    checkpoint,
    *,
    images_dir: str | None,
    conf_threshold: float | None,
    device: str | None,
    tile: bool | None,
    tile_size: int | None,
    overlap: float | None,
    tile_batch_size: int,
    global_nms_iou: float | None,
    max_dets: int | None,
    postprocess: str,
    trait: str | None,
    calibration_labels_dir: str | None,
    calibration_images_dir: str | None,
    experiment_id: str | None,
    group_by: str | None = None,
    group_key_map: dict[str, str] | None = None,
    split_seed: int = DEFAULT_CAL_SEED,
    split_holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
    selection_dir: str | None = None,
) -> dict:
    """A per-image pass over ``images_dir`` from a loaded checkpoint, calibrated when ``trait``
    and ``calibration_labels_dir`` are given: the run's own facts (:meth:`_PreparedPass.result`)
    with ``results`` a stream that predicts each image as it is consumed, or ``{"error": ...}``.
    """
    from tcip_mcp.pipelines.operating_point import apply_operating_point
    from tcip_mcp.pipelines.resolution import dataset_hash

    p = _prepare_pass(
        checkpoint, images_dir=images_dir, conf_threshold=conf_threshold, device=device, tile=tile, tile_size=tile_size,
        overlap=overlap, tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou,
        max_dets=max_dets, postprocess=postprocess, experiment_id=experiment_id)
    if isinstance(p, str):
        return {"error": p}

    # Resolve the confidence operating point: with a trait + labeled calibration dir, derive it
    # per dataset (count-unbiased + held-out validated); otherwise the byte-identical raw path.
    if not (trait and calibration_labels_dir):
        out = p.raw_result()
    else:
        from tcip_annotation.json_io import UnreadableLabelDocument

        from tcip_mcp.pipelines.calibration import calibrate_operating_point, gate_evidence_summary

        cal_images = calibration_images_dir or images_dir
        try:
            bundle, cal_hash, n_excluded_incomplete_attribute, evidence = calibrate_operating_point(
                p.predictor, trait, calibration_labels_dir, cal_images,
                tile=p.tiled, tile_size=p.tile_size, overlap=p.overlap,
                tile_resize=p.tile_resize,
                tile_size_source=p.tile_size_source,
                tile_size_derived_from=p.tile_size_derived_from, tiled_source=p.tiled_source,
                tile_batch_size=tile_batch_size, global_nms_iou=p.nms_iou,
                postprocess=postprocess,
                cross_tile_nms=global_nms_iou, max_dets=max_dets,
                group_by=group_by, group_key_map=group_key_map,
                experiment_id=p.identity["experiment_id"],
                seed=split_seed, holdout_ratio=split_holdout_ratio,
                selection_dir=selection_dir,
            )
        except (ValueError, UnreadableLabelDocument) as exc:
            # An inadmissible reference, a locked split that no longer resolves, or a calibration
            # GT file that will not read: a clean refusal, not a bare KeyError downstream.
            return {"error": str(exc)}
        conf_param = bundle.get("conf")
        conf = (conf_param.value if conf_param.is_shippable
                else conf_param.unvalidated_value(acknowledge_unvalidated=True))
        p.max_dets = int(bundle.get("max_dets").value)
        p.nms_iou = float(bundle.get("cross_tile_nms").value or p.nms_iou)
        apply_operating_point(p.predictor, conf, p.max_dets)
        # Dataset-scope firewall: the conf is scoped to the calibration GT. The inference target is
        # usually unlabeled, so its GT identity (a content hash) is undefined, pass None and record
        # 'not-comparable-unlabeled-target'. Only when inferencing the same labeled set it calibrated
        # on can we compare real hashes and flag cross-dataset inheritance.
        from tcip_mcp.pipelines.image_utils import stem_of

        from tcip_annotation.json_io import prediction_documents

        inf_stems = [stem_of(pp) for pp in p.paths]
        cal_label_stems = (
            set(evidence.get("calibration_stems", [])) if selection_dir is not None
            else {pp.stem for pp in prediction_documents(calibration_labels_dir)}
        )
        same_images = calibration_images_dir is None or (
            images_dir is not None and Path(calibration_images_dir) == Path(images_dir))
        # A selection's own calibration universe is a held-out subset of the labeled directory,
        # so inferring the whole directory is still the same labeled set the calibration drew.
        if selection_dir is not None:
            comparable = bool(
                same_images and inf_stems and cal_label_stems
                and cal_label_stems <= set(inf_stems)
            )
        else:
            comparable = bool(same_images and inf_stems and set(inf_stems) == cal_label_stems)
        if comparable:
            # The bundle's own hash covers the calibration universe under a manifest, not the
            # (larger) inference stem list, so the target must be hashed over that same universe.
            hashed_stems = evidence.get("calibration_stems", []) if selection_dir is not None \
                else inf_stems
            target_hash, cross_dataset_check = (
                dataset_hash(calibration_labels_dir, stems=hashed_stems), "same-labeled-set")
        else:
            target_hash, cross_dataset_check = None, "not-comparable-unlabeled-target"
        issues = bundle.shippable_issues(target_dataset_hash=target_hash)
        # validated only when held-out passed and nothing is un-shippable under the target actually used.
        validated = bool(bundle.is_shippable and not issues)
        if (conf_param.gate_evidence or {}).get("conf_floor_mismatch"):
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
            "gate_evidence_summary": gate_evidence_summary(conf_param),
            "n_excluded_incomplete_attribute": n_excluded_incomplete_attribute,
        }
        manifest_excluded = evidence.get("excluded")
        if manifest_excluded is not None:
            extra["n_excluded_training_stems"] = len(manifest_excluded["excluded_training_stems"])
            extra["n_excluded_validation_stems"] = len(
                manifest_excluded["excluded_validation_stems"])
        # The full curve can be large, persist it and return the path (provenance emits has_gate_evidence).
        # The record's own body is its identity, so a curve differing from a prior one is never lost.
        curve_body = {
            "trait": trait,
            "dataset_hash": cal_hash,
            "checkpoint_sha256": p.identity["sha256"],
            "predictor_path": {
                "tile": p.tiled, "tile_size": p.tile_size,
                "overlap": p.overlap, "postprocess": postprocess,
                "global_nms_iou": p.nms_iou, "max_dets": p.max_dets,
            },
            "gate_evidence": conf_param.gate_evidence,
            "calibration_evidence": evidence,
        }
        # The evidence rides in the curve artifact, read back by identity, never on this response.
        try:
            curve_identity_hex = keep_calibration_curve(curve_body)
        except (TypeError, ValueError) as exc:
            return {"error": f"the operating-point curve for trait {trait!r} could not be kept "
                             f"(its body cannot be recorded): {exc}"}
        except StoreError:
            logger.warning("could not persist operating-point curve", exc_info=True)
        else:
            extra["calibration_evidence_key"] = curve_identity_hex
        out = p.result(bundle.to_provenance()["operating_point"], extra)

    # Warn, never fail, when a slow workload will run on CPU because CUDA is not available.
    if device != "cpu" and (p.tiled or len(p.paths) > 8):
        import torch

        if not torch.cuda.is_available():
            out["warning"] = (
                f"CUDA not available, running {len(p.paths)} image(s)"
                f"{' tiled' if p.tiled else ''} on CPU, which is much slower. Install a "
                "CUDA torch build (see environment.yml) to use the GPU."
            )
            logger.warning(out["warning"])

    out["results"] = (result for path in p.paths for result in p.predict([path]))
    return out


@dataclass
class _PreparedPass:
    """A raw per-image pass resolved from a loaded checkpoint and its caller's stated values: the
    predictor, the images it runs over, the tile regime and operating point it runs at, and the
    run's identity and class scope.
    """

    checkpoint_path: str
    predictor: Any
    images_dir: str | None
    paths: list[str | Path | BandGroupRef]
    identity: dict
    scope: ClassScope
    id_map: dict | None
    tiled: bool
    tiled_source: str
    tile_size: int | None
    tile_size_source: str
    tile_size_derived_from: Any
    overlap: float
    tile_resize: Any
    conf: float
    nms_iou: float
    max_dets: int
    conf_stated: bool
    max_dets_stated: bool
    tile_batch_size: int
    postprocess: str

    def predict(self, paths: list[str | Path | BandGroupRef]) -> list[dict]:
        return self.predictor.predict_batch(
            paths, tile=self.tiled, tile_size=self.tile_size, overlap=self.overlap,
            tile_batch_size=self.tile_batch_size, global_nms_iou=self.nms_iou,
            postprocess=self.postprocess, tile_resize=self.tile_resize)

    def result(self, operating_point: dict, extra: dict) -> dict:
        """The run's own facts a publisher stamps from, before any image is predicted."""
        from datetime import datetime, timezone

        return {
            "checkpoint": self.checkpoint_path,
            "checkpoint_sha256": self.identity["sha256"],
            "experiment_id": self.identity["experiment_id"],
            "images_dir": self.images_dir,
            "raster_path": None,
            "produced_at": datetime.now(timezone.utc).isoformat(),
            "operating_point": operating_point,
            "id_map": self.id_map,
            "subject": self.scope.subject,
            "attribute": self.scope.attribute,
            "shippable_issues": [],
            "dataset_hash": None,
            "gate_evidence_summary": None,
            **extra,
        }

    def raw_result(self) -> dict:
        """:meth:`result` at the pass' own uncalibrated operating point: the model already carries
        it in-model, and the bundle stamps it validated_against=false so its untrustworthiness
        travels with the result."""
        from tcip_mcp.pipelines.resolution import raw_operating_point

        op_bundle = raw_operating_point(
            conf=self.conf, cross_tile_nms=self.nms_iou, tiled=self.tiled,
            tile_size=self.tile_size, max_dets=self.max_dets,
            tile_size_source=self.tile_size_source,
            tile_size_derived_from=self.tile_size_derived_from, tiled_source=self.tiled_source,
            conf_stated=self.conf_stated, max_dets_stated=self.max_dets_stated)
        return self.result(op_bundle.to_provenance()["operating_point"],
                           {"validated": False, "conf_source": "default"})


def _prepare_pass(
    checkpoint, *, images_dir: str | None,
    conf_threshold: float | None, device: str | None, tile: bool | None, tile_size: int | None,
    overlap: float | None, global_nms_iou: float | None, max_dets: int | None, postprocess: str,
    experiment_id: str | None, tile_batch_size: int,
) -> "_PreparedPass | str":
    """Resolve a pass from a loaded checkpoint and what its caller stated (``None`` for anything
    unstated), over ``images_dir``'s logical images or, with ``images_dir`` ``None``, over no
    image list (a raster pass), or the refusal naming why it cannot run: a stated tile edge the
    checkpoint's recorded geometry contradicts, a tiled run with no basis for its scale, or a
    classified run with no map to decode its predictions."""
    from tcip_mcp.model_registry import resolve_model_identity
    from tcip_mcp.pipelines.image_utils import list_logical_images
    from tcip_mcp.pipelines.inference.predictor import (
        TileEdgeContradiction, build_predictor, explicit_edge_provenance, resolve_tile_regime,
    )

    logical = list_logical_images(images_dir) if images_dir is not None else {}
    paths: list[str | Path | BandGroupRef] = [logical[stem] for stem in sorted(logical)]

    conf, nms_iou, applied_max_dets = applied_operating_point(
        conf_threshold, global_nms_iou, max_dets)
    # NMS IoU and the detection cap govern which boxes exist in-model, not only the tile merge.
    predictor = build_predictor(
        checkpoint, device=device, score_threshold=conf, nms_iou=nms_iou,
        max_dets=applied_max_dets)
    # An unset ``tile`` gets the checkpoint's own tiled-or-not regime, never a platform default.
    tiled = (getattr(predictor, "train_tile_size", None) is not None) if tile is None else tile
    # Identity before calibration: its train-disjointness gate needs the checkpoint's experiment.
    identity = resolve_model_identity(checkpoint, experiment_id=experiment_id)

    # The resize half resolves only when tiled, so an unreadable augmentation config never sinks
    # an untiled run.
    try:
        resolved_tile, tile_size_source, resolved_overlap, _overlap_source, tile_resize = (
            resolve_tile_regime(predictor, tiled=tiled, tile_size=tile_size, overlap=overlap))
    except TileEdgeContradiction as exc:
        return str(exc)
    if tiled and resolved_tile is None:
        return (
            f"tile_size could not be resolved for {checkpoint.path}: this checkpoint carries no "
            "persisted training tile geometry, no tile_size was given explicitly, and its untiled "
            "training frame yields no tile edge either (none recorded, or a rectangular one, which "
            "no single square edge reproduces the scale of on both axes), so tiled inference has no "
            "real basis to run at. Pass tile_size explicitly or retrain with tile geometry "
            "persisted; a pass over an images directory can also leave tile unset or False to run "
            "untiled."
        )
    if tile_size_source == "derived":
        logger.info("tile_size %d derived from the checkpoint's training geometry", resolved_tile)
    elif tiled and tile_size_source == "native_ratio":
        resize_note = "" if tile_resize is None else (
            f", each tile run through its recorded train-time resize {tuple(tile_resize)}")
        logger.info(
            "tile_size %d derived from this checkpoint's own uniform untiled training frame%s",
            resolved_tile, resize_note)

    # A classified run with no id_map refuses before any pass, so no calibration is spent on it.
    scope = run_scope(predictor)
    id_map = resolve_decode_id_map(predictor, images_dir)
    refusal = unmapped_classified_run(scope, id_map, images_dir=images_dir)
    if refusal is not None:
        return refusal
    return _PreparedPass(
        checkpoint_path=checkpoint.path, predictor=predictor, images_dir=images_dir, paths=paths,
        identity=identity, scope=scope, id_map=id_map, tiled=tiled,
        tiled_source="explicit" if tile is not None else "default",
        tile_size=resolved_tile, tile_size_source=tile_size_source,
        tile_size_derived_from=(
            explicit_edge_provenance(predictor, resolved_tile)
            if tile_size_source == "explicit" and resolved_tile is not None else None),
        overlap=resolved_overlap, tile_resize=tile_resize,
        conf=conf, nms_iou=nms_iou, max_dets=applied_max_dets,
        conf_stated=conf_threshold is not None, max_dets_stated=max_dets is not None,
        tile_batch_size=tile_batch_size, postprocess=postprocess)


# --- earning the record a validated count claim names (the shared half of every door here) ---

_NO_DATASET_ROOT_NOTE = (
    "{bucket} sits under no dataset root, so two guarantees a bucket normally carries are absent "
    "here. The review-verdict immutability guard is inoperative: nothing checks whether a human "
    "has already recorded verdicts against predictions at this path before this run replaced them. "
    "And a count claim earned for these predictions has no dataset-relative key to be recorded "
    "under, so this bucket is stamped unvalidated whatever its operating point cleared. The "
    "prediction-document refusal is unaffected by either absence: a bucket here that already holds "
    "a document from a prior run still refuses a second publish the same way one under a dataset "
    "root does. Write into a dataset's own predictions layout "
    "(<dataset_root>/predictions/<model>/<date>) for the two absent guarantees."
)
"""What a door tells its caller about a bucket outside the dataset layout, rather than guarding it
against a verdict store that holds nothing about it or letting it claim a count nothing can verify."""


def _resolve_writable_bucket_for(output_dir: str, *, overwrite: bool):
    """The bucket a run may write for ``output_dir``, its resolution, and its dataset root.

    Returns ``(out, resolution, dataset_root, refusal)``. ``refusal`` is the door's own error dict
    (``error``, the ``suggested_bucket`` path and its ``suggested_name``, and a ``verdict_count``
    or a ``document_stem_count``), or ``None``: a requested bucket carrying review verdicts that
    the caller asked to overwrite (``BucketHasVerdicts``), or one already holding a prediction
    document with no verdict recorded against it, whatever ``overwrite`` says
    (``BucketHoldsDocuments``). A canonical ``predictions/<model>/<date>`` bucket redirects by its
    model segment, any other by its last segment. ``dataset_root`` is ``None`` for a bucket under
    no dataset, whose verdict guard is inoperative.
    """
    from tcip_mcp.dataset_layout import canonical_prediction_bucket, prediction_dir
    from tcip_mcp.prediction_buckets import (
        BucketHasVerdicts,
        BucketHoldsDocuments,
        resolve_prediction_bucket,
        resolve_writable_bucket,
        review_state_dir_of,
    )

    out_path = resolve_output_path(output_dir)
    parent, base_name = out_path.parent, out_path.name

    canonical = canonical_prediction_bucket(out_path)
    canonical_dataset_root = canonical[0] if canonical is not None else None
    canonical_date = canonical[2] if canonical is not None else None

    # The guard reads the bucket's own dataset verdict store; no dataset root means no store to guard against.
    dataset_root = bucket_dataset_root(out_path)
    review_state_dir = None if dataset_root is None else review_state_dir_of(dataset_root)

    try:
        if canonical is not None:
            canonical_root, model, canonical_date = canonical
            out, resolution = resolve_prediction_bucket(
                canonical_root, model, canonical_date,
                review_state_dir=review_state_dir, overwrite=overwrite, refuse_documents=True)
        else:
            resolution = resolve_writable_bucket(
                review_state_dir, base_name, lambda n: [parent / n],
                overwrite=overwrite, refuse_documents=True)
            out = parent / resolution.name
    except (BucketHasVerdicts, BucketHoldsDocuments) as exc:
        suggested = None
        if exc.suggested is not None:
            suggested = (
                str(prediction_dir(canonical_dataset_root, exc.suggested, canonical_date))
                if canonical_dataset_root is not None
                else str(parent / exc.suggested)
            )
        error: dict = {"error": str(exc), "suggested_bucket": suggested,
                       "suggested_name": exc.suggested}
        if isinstance(exc, BucketHasVerdicts):
            error["verdict_count"] = exc.count
        else:
            error["document_stem_count"] = exc.document_stem_count
        return None, None, dataset_root, error
    return out, resolution, dataset_root, None


def _calibration_evidence(result: dict) -> dict | None:
    """The evidence this run's calibration gate ran over, read back from the artifact it was kept
    in, or ``None`` for a run that resolved no calibrated operating point.

    The record read back is re-encoded and its digest compared against ``identity`` (the run's own
    carried key, see :func:`calibration_curve_identity`); a difference raises ``ValueError`` naming
    both. An absent record returns ``None``.
    """
    identity = result.get("calibration_evidence_key")
    if not identity:
        return None
    body = store.read(calibration_curve_key(identity), default=None)
    if body is None:
        return None
    recomputed = calibration_curve_identity(body)
    if recomputed != identity:
        raise ValueError(
            f"the calibration-curve record under {identity!r} does not match the digest this "
            f"run's own response carried (recomputed {recomputed!r}): the evidence the count "
            "gate would run over is not what this run wrote."
        )
    return body.get("calibration_evidence")


def _draft_count_claim(result: dict, *, trait: str | None, bucket: Path,
                       dataset_root: Path | None, tile_size_validated: str | None,
                       evidence: dict | None):
    """The passed gate a validated count is stamped from, for a run that earned one.

    ``evidence`` is the reference evidence the claim opens from, or ``None`` to read the run's own
    kept calibration evidence (:func:`_calibration_evidence`). Returns ``(draft, refusal)``.
    ``refusal`` is the door's own error dict for a run that reports a validated operating point
    with no evidence left to earn a record from, which ends the run with the bucket still
    untouched. A run whose own dimensions did not all clear earns nothing, and so does a bucket
    under no dataset root; both stamp unvalidated.
    """
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, open_validation

    if not result["validated"] or tile_size_validated == VALIDATED_FALSE:
        return None, None
    try:
        evidence = evidence or _calibration_evidence(result)
    except ValueError as exc:
        return None, {"error": str(exc)}
    if evidence is None:
        return None, {"error": (
            f"the run reports a validated operating point for trait {trait!r} but kept no "
            "calibration evidence to earn a validation record from, so these counts cannot be "
            "stamped validated. The evidence is written beside the confidence sweep at calibration "
            "time; re-run the calibration. This has no acknowledgment route: it is a missing "
            "validation record for a run that already reports itself validated, not an ungated "
            "dimension a breeder can choose to ship unvalidated."
        )}
    if dataset_root is None:
        logger.warning(_NO_DATASET_ROOT_NOTE.format(bucket=bucket))
        return None, None
    # A validated result comes only from a calibration, which runs only for a named trait.
    assert trait is not None
    try:
        draft = open_validation(
            document="operating_point",
            evidence={"resolver": evidence["resolver"], "inputs": evidence["inputs"]},
            trait=trait, checkpoint_sha256=result["checkpoint_sha256"],
            producing_experiment_id=result["experiment_id"],
            reference_inputs={**evidence["reference_inputs"], "dataset_root": str(dataset_root)})
    except ValueError as exc:
        return None, {"error": f"the count claim for trait {trait!r} was not earned: {exc}"}
    return draft, None


def _publish_predictions(out: Path, predictions: Iterable[dict], stamp_body: dict, draft, *,
                         producer: str, dataset_root: Path | None
                         ) -> tuple[list[str], int, dict, bool | None]:
    """Write one prediction document per result as ``predictions`` yields it, then append the
    record the gate earned over the documents as they landed, write the stamp last, link the bucket
    into its run's lineage, and record the publication.

    ``predictions`` is a list or a stream that predicts, reports progress and stops on cancellation
    as it is consumed. The stamp names each written document's stem and its image's file name
    (``image_filenames``) and carries the mask-binarize threshold when a result carried masks.
    ``draft`` is ``None`` for a run that earned nothing, which stamps unvalidated with no pointer.

    A raise after the first document lands, before the stamp does, writes no stamp and leaves one
    ``prediction_bucket_published`` line under status ``failed`` naming the documents written and
    the error, then propagates; a raise before any document lands committed nothing and leaves no
    line. A lost audit line (``AuditEntryNotWritten``) propagates as it is. A completed pass leaves
    the stamp's own ``stamp_written`` line, which names every document, and one
    ``prediction_bucket_published`` line naming whether the lineage link landed (``None`` when the
    run names no experiment). ``dataset_root`` is the bucket's (``bucket_dataset_root``), the root
    both lines are recorded under.

    Returns ``(written, dropped_nonpositive_boxes, stamp_body, lineage_linked)``.
    """
    from tcip_mcp.audit import AuditEntryNotWritten, record_event_or_raise
    from tcip_mcp.pipelines.resolution import seal_validation, write_sidecar

    id_map, subject, attribute = stamp_body["id_map"], stamp_body["subject"], stamp_body["attribute"]
    written: list[str] = []
    names: dict[str, str] = {}
    dropped, has_masks = 0, False
    try:
        out.mkdir(parents=True, exist_ok=True)
        for r in predictions:
            image = Path(r["image"])
            unmapped = unmapped_label_ids([r], id_map) if attribute is not None else []
            if unmapped:
                raise ValueError(
                    f"{image.stem}: this classified run decoded to id(s) {unmapped}, not keys of "
                    f"its recorded id_map ({sorted((id_map or {}).values())}).")
            # Read before the write: a drop can empty a mask list that was genuinely there.
            has_masks = has_masks or bool(r.get("masks"))
            document = out / label_filename(image.stem)
            dropped += write_predictions_json(
                document, r, created_by=producer, id_map=id_map, subject=subject,
                attribute=attribute)
            written.append(str(document))
            names[image.stem] = image.name
        stamp_body["image_filenames"] = names
        if has_masks:
            stamp_body["mask_binarize"] = mask_binarize_provenance()
        if draft is not None:
            stamp_body = seal_validation(
                draft, dataset_root=draft.dataset_root, bucket_dirs=[out], stamp_body=stamp_body)
        write_sidecar(out, stamp_body)
    except AuditEntryNotWritten:
        raise
    except Exception as exc:
        if written:
            record_event_or_raise(
                "prediction_bucket_published",
                {"predictions_dir": str(out), "written": written, "error": str(exc)},
                status="failed", scope=dataset_root)
        raise
    exp_id = stamp_body["experiment_id"]
    lineage_linked = None
    if exp_id:
        try:
            from tcip_mcp.experiments import update_lineage

            update_lineage(exp_id, predictions=str(out))
            lineage_linked = True
        except Exception:
            logger.warning("could not link predictions into experiment lineage", exc_info=True)
            lineage_linked = False
    record_event_or_raise(
        "prediction_bucket_published",
        {"predictions_dir": str(out), "lineage_linked": lineage_linked}, scope=dataset_root)
    return written, dropped, stamp_body, lineage_linked


def _clear_door_refusal_reason(recorded_path: str) -> str | None:
    """Why ``clear_prediction_bucket`` would itself refuse ``recorded_path``, or ``None`` when the
    door would reach it.
    """
    from tcip_mcp.dataset_layout import canonical_prediction_bucket
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar, stamp_names_raster
    from tcip_mcp.prediction_buckets import bucket_key_of, review_state_count, review_state_dir_of

    canonical = canonical_prediction_bucket(recorded_path)
    if canonical is None:
        return "it is not a canonical bucket under a dataset root"
    dataset_root = canonical[0]
    stamp = read_operating_point_sidecar(Path(recorded_path))
    if stamp is not None and stamp_names_raster(stamp):
        return "its stamp names a whole-raster pass, out of that door's scope"
    count = review_state_count(review_state_dir_of(dataset_root), bucket_key_of(Path(recorded_path)))
    if count:
        return "it carries review state"
    return None


def _frozen_pointer_refusal(experiment_id: str | None, out: Path) -> str | None:
    """Why publishing into ``out`` is refused because ``experiment_id``'s lineage already points at
    another bucket it froze, or ``None`` when nothing refuses (no experiment, or a pointer still
    open). The reason names ``clear_prediction_bucket`` as the remedy only where that door would
    reach the recorded bucket.
    """
    if not experiment_id:
        return None
    from tcip_mcp.experiments import lineage_key, pointer_frozen, read_member

    frozen = pointer_frozen(experiment_id, "lineage", "predictions", str(out))
    if frozen is None:
        return None
    # pointer_frozen refuses exactly when terminal and the recorded pointer differs: that
    # recorded path already holds the experiment's own published documents.
    recorded_path = (read_member(lineage_key(experiment_id), {}) or {}).get("predictions")
    door_reason = _clear_door_refusal_reason(recorded_path) if recorded_path else None
    if door_reason is None:
        return (f"{frozen} {recorded_path!r} holds the experiment's own published documents; "
                f"clear_prediction_bucket(predictions_dir={recorded_path!r}, reason=...) "
                "clears it for re-publication into that recorded path.")
    return (f"{frozen} {recorded_path!r} holds the experiment's own published documents, and "
            f"clear_prediction_bucket refuses that bucket too, since {door_reason}; it stays as "
            "published.")


def publish_bucket(result: dict, *, out: Path, trait: str | None, dataset_root: Path | None,
                   allow_unvalidated_staging: bool, claim_evidence: dict | None = None,
                   stamp_extras: dict | None = None) -> dict:
    """Publish a run's predictions into ``out``: the tile gate, the count claim's own gate, the
    frozen-lineage-pointer refusal, then the writes, the stamp, the lineage link and the
    publication's line (:func:`_publish_predictions`).

    ``result`` is the run's own facts (:meth:`_PreparedPass.result`) with ``results`` a list or a
    stream that predicts as it is consumed; every gate runs before the first result is drawn.
    ``claim_evidence`` is the reference evidence a validated count claim opens from, or ``None``
    for the run's own kept calibration evidence; ``stamp_extras`` are the producer's own stamp
    fields. ``out`` has already cleared the bucket-immutability resolver.

    Returns ``{"refusal": <the door's own error dict>}`` with the bucket untouched, or ``refusal``
    ``None`` beside ``written``, ``dropped_boxes``, the ``op_stamp`` as written and
    ``lineage_linked`` (``True``/``False`` for an attempted link, ``None`` when the run named no
    experiment to link).
    """
    from tcip_mcp.pipelines.resolution import (
        check_delivery_gate, operating_point_stamp, prediction_producer, tile_size_gate_flag,
    )

    tile_ref = tile_size_gate_flag(result["operating_point"])
    tile_flags: dict[str, str | None] = {"tile_size": tile_ref} if tile_ref is not None else {}
    gate = check_delivery_gate(tile_flags, allow_unvalidated_staging=allow_unvalidated_staging)
    if not gate.ok:
        return {"refusal": {"error": gate.reason, "tile_size_validated": tile_ref}}
    tile_size_validated = gate.stamp.get("tile_size")

    # The count claim's own gate, run before a single file is written.
    draft, refusal = _draft_count_claim(
        result, trait=trait, bucket=out, dataset_root=dataset_root,
        tile_size_validated=tile_size_validated, evidence=claim_evidence)
    if refusal is not None:
        return {"refusal": refusal}

    frozen = _frozen_pointer_refusal(result["experiment_id"], out)
    if frozen is not None:
        return {"refusal": {"error": frozen}}

    sha = result["checkpoint_sha256"]
    op_stamp = operating_point_stamp(
        result["operating_point"], validated=draft is not None, validated_by=None,
        tile_size_validated=tile_size_validated, shippable_issues=result["shippable_issues"],
        id_map=result["id_map"], subject=result["subject"], attribute=result["attribute"],
        trait=trait, dataset_hash=result["dataset_hash"],
        checkpoint=Path(result["checkpoint"]).stem, checkpoint_sha256=sha,
        experiment_id=result["experiment_id"], images_dir=result["images_dir"],
        raster_path=result["raster_path"], produced_at=result["produced_at"],
        gate_evidence_summary=result["gate_evidence_summary"], **(stamp_extras or {}))
    written, dropped_boxes, op_stamp, lineage_linked = _publish_predictions(
        out, result["results"], op_stamp, draft,
        producer=prediction_producer(result["checkpoint"], sha), dataset_root=dataset_root)
    return {"refusal": None, "written": written, "dropped_boxes": dropped_boxes,
            "op_stamp": op_stamp, "lineage_linked": lineage_linked}


def bucket_location(out: Path | str, resolution, requested_output_dir: str) -> dict:
    """Where a publishing door's bucket lands: ``output_dir``, whether the write was redirected,
    and the bucket the caller asked for when it was."""
    return {"output_dir": str(out), "bucket_redirected": resolution.redirected,
            "requested_output_dir": requested_output_dir if resolution.redirected else None}


def _bucket_response(pub: dict, *, out: Path, resolution, dataset_root: Path | None,
                     requested_output_dir: str) -> dict:
    """The response fields every publishing door reports about the bucket it wrote: its
    :func:`bucket_location`, what the publisher stamped and dropped, and the dataset-root note
    when the bucket sits under none."""
    response = {
        **bucket_location(out, resolution, requested_output_dir), "files": pub["written"],
        "validated": pub["op_stamp"]["validated"],
        "tile_size_validated": pub["op_stamp"]["tile_size_validated"],
        "verdict_guard_operative": dataset_root is not None,
        "dropped_nonpositive_boxes": pub["dropped_boxes"],
        "lineage_linked": pub["lineage_linked"],
    }
    if dataset_root is None:
        response["note"] = _NO_DATASET_ROOT_NOTE.format(bucket=out)
    return response


# --- clearing a terminal experiment's bucket for re-publication ---

_OTHER_STAMP_DOCUMENTS = tuple(sorted(
    Path(name).stem for name in SIDECAR_FILENAMES if name != "operating_point.json"))
"""Every sidecar stamp beside ``operating_point``, derived from
:data:`~tcip_annotation.json_io.SIDECAR_FILENAMES` so a sixth stamp store registered there is
reconciled by this door too rather than left stranded at a half-cleared source. The door reconciles
``operating_point`` first, by its own value-aware rule, then each of these by the plain
copy-or-refuse rule every other stamp and every document share."""


def _reconcile_op_stamp(source: Path, destination: Path) -> tuple[bool, dict | None]:
    """Move (or finish moving) the ``operating_point`` stamp between ``source`` and
    ``destination``. Returns ``(moved_this_call, refusal)``.

    Absent at the destination with a value at the source: copied, then the source's is deleted.
    Present at both and equal: only the source's own delete remains. Present at both and different:
    the destination's copy is replaced by the source's current value (a merge writer landed on the
    source since the copy) before the source is deleted. Present at the destination alone: already
    finished. Absent at both: refused.
    """
    from tcip_mcp.pipelines.resolution import sidecar_key

    dest_key, source_key = sidecar_key(destination, "operating_point"), sidecar_key(source, "operating_point")
    dest_v = store.read_versioned(dest_key, default=None)
    source_v = store.read_versioned(source_key, default=None)
    dest_present, source_present = dest_v.value is not None, source_v.value is not None

    if not dest_present and not source_present:
        return False, {"error": f"operating_point.json is present at neither {source} nor "
                                f"{destination}: a store written past this door."}
    if not dest_present:
        store.replace(dest_key, source_v.value, expect=Version.ABSENT)
        store.delete(source_key, expect=source_v.version)
        return True, None
    if source_present:
        if dest_v.value != source_v.value:
            store.replace(dest_key, source_v.value, expect=dest_v.version)
        store.delete(source_key, expect=source_v.version)
        return True, None
    return False, None  # present at the destination alone: already finished


def _reconcile_secondary_stamp(
    source: Path, destination: Path, document: str, *, op_stamp_left_source: bool,
) -> tuple[bool, dict | None]:
    """Move (or finish moving) one of :data:`_OTHER_STAMP_DOCUMENTS` between ``source`` and
    ``destination``. Returns ``(moved_this_call, refusal)``.

    Absent at the destination with a value at the source: while ``operating_point.json`` still
    reads from ``source`` (``op_stamp_left_source`` is ``False``), copied then deleted; once it has
    left ``source``, refused (a fresh write landed on the half-cleared source). Present at both and
    equal: the source's delete finishes it. Present at both and different: refused naming them.
    Present at neither, or at the destination alone: nothing to do.
    """
    from tcip_mcp.pipelines.resolution import sidecar_key

    dest_key, source_key = sidecar_key(destination, document), sidecar_key(source, document)
    dest_v = store.read_versioned(dest_key, default=None)
    source_v = store.read_versioned(source_key, default=None)
    dest_present, source_present = dest_v.value is not None, source_v.value is not None

    fresh_write_refusal = {
        "error": f"{document}.json is present at {source} alone with operating_point.json "
                f"already at {destination}: resolve_scale, classifier_operating_point, "
                "ordinal_operating_point or regression_operating_point wrote a fresh stamp into "
                "the half-cleared source after this clear began; a person must reconcile it, no "
                "door removes it.",
    }

    if not dest_present and source_present:
        if op_stamp_left_source:
            return False, fresh_write_refusal
        store.replace(dest_key, source_v.value, expect=Version.ABSENT)
        store.delete(source_key, expect=source_v.version)
        return True, None
    if dest_present and source_present:
        if dest_v.value == source_v.value:
            store.delete(source_key, expect=source_v.version)
            return True, None
        return False, {"error": f"{document}.json differs between {source} and {destination}: "
                                "resolve_scale, classifier_operating_point, "
                                "ordinal_operating_point or regression_operating_point wrote a "
                                "fresh stamp into the source after this clear began; a person "
                                "must reconcile it, no door removes it."}
    return False, None


def _reconcile_document(source: Path, destination: Path, stem: str) -> tuple[bool, dict | None]:
    """Move (or finish moving) one prediction document between ``source`` and ``destination``.
    Returns ``(moved_this_call, refusal)``.

    Absent at the destination with bytes at the source: copied, then the source's deleted. Present
    at both with equal bytes: the source's delete finishes it. Present at both and different:
    refused naming it. Present at neither, or at the destination alone: nothing to do.
    """
    from tcip_annotation.json_io import annotation_record_key

    dest_key = annotation_record_key(destination, stem)
    source_key = annotation_record_key(source, stem)
    dest_v = store.read_blob_versioned(dest_key, default=None)
    source_v = store.read_blob_versioned(source_key, default=None)
    dest_present, source_present = dest_v.value is not None, source_v.value is not None

    if not dest_present and source_present:
        store.put_blob(dest_key, source_v.value, expect=Version.ABSENT)
        store.delete(source_key, expect=source_v.version)
        return True, None
    if dest_present and source_present:
        if dest_v.value == source_v.value:
            store.delete(source_key, expect=source_v.version)
            return True, None
        return False, {"error": f"{label_filename(stem)} differs between {source} and "
                                f"{destination}: the "
                                "staging door wrote a same-stem document over one this clear had "
                                "already copied; a person must reconcile it, no door removes it."}
    return False, None


def _cleared_artifact_matches(entry_path: str, dataset_root: Path, model: str, date: str | None) -> bool:
    """Whether an artifact's recorded ``path`` resolves, through ``cleared_bucket_of``, to the
    same dataset root, model and date as the source this clear (or resume) is scoped to."""
    from tcip_mcp.dataset_layout import cleared_bucket_of

    resolved = cleared_bucket_of(entry_path)
    if resolved is None:
        return False
    entry_root, entry_model, _entry_stamp, entry_date = resolved
    return same_directory(entry_root, dataset_root) and (entry_model, entry_date) == (model, date)


def _newest_cleared_artifact_for_source(
    experiment_id: str, dataset_root: Path, model: str, date: str | None,
) -> dict | None:
    """The newest ``cleared:`` artifact on ``experiment_id`` whose path names a cleared bucket of
    this source (``dataset_root``, ``model``, ``date``), by the artifacts' own ``recorded`` times,
    or ``None`` when it names none.
    """
    from tcip_mcp.experiments import artifacts_key, read_member

    artifacts = read_member(artifacts_key(experiment_id), {})
    if not isinstance(artifacts, dict):
        return None
    best: tuple[str, str, str] | None = None  # (recorded, name, path)
    for name, entry in artifacts.items():
        if not name.startswith("cleared:") or not isinstance(entry, dict):
            continue
        path = entry.get("path")
        if not path or not _cleared_artifact_matches(path, dataset_root, model, date):
            continue
        recorded = entry.get("recorded", "")
        if best is None or recorded > best[0]:
            best = (recorded, name, path)
    return None if best is None else {"name": best[1], "path": best[2], "recorded": best[0]}


def _find_cleared_candidate_with_no_source_stamp(
    dataset_root: Path, model: str, date: str | None,
) -> dict | None:
    """When the source holds no ``operating_point`` stamp at all: the resume candidate the no-stamp
    refusal names, found by walking the cleared archive. Every cleared bucket
    :func:`~tcip_mcp.dataset_layout.cleared_bucket_of` resolves to this source's model and date is
    read for its own stamp's experiment, and the newest ``cleared:`` artifact that experiment
    records for it is the candidate; one carrying no stamp, or whose experiment records no artifact
    for it, is never named.
    """
    from tcip_mcp.dataset_layout import cleared_bucket_of, is_cleared_bucket, prediction_bucket_dirs
    from tcip_mcp.pipelines.resolution import sidecar_key
    from tcip_store import StoreError

    best: tuple[str, str] | None = None  # (recorded, path)
    for cand in prediction_bucket_dirs(dataset_root, include_cleared=True):
        if not is_cleared_bucket(cand):
            continue
        resolved = cleared_bucket_of(cand)
        if resolved is None:
            continue
        cand_root, cand_model, _cand_stamp, cand_date = resolved
        if not (same_directory(cand_root, dataset_root) and (cand_model, cand_date) == (model, date)):
            continue
        try:
            cand_stamp = store.read(sidecar_key(cand, "operating_point"), default=None)
        except StoreError:
            continue
        if not isinstance(cand_stamp, dict):
            continue
        cand_exp_id = cand_stamp["experiment_id"]
        if not cand_exp_id:
            continue
        newest = _newest_cleared_artifact_for_source(cand_exp_id, dataset_root, model, date)
        if newest is None or not same_directory(newest["path"], cand):
            continue
        recorded = newest["recorded"]
        if best is None or recorded > best[0]:
            best = (recorded, str(cand))
    return None if best is None else {"path": best[1]}


@mcp.tool()
@audited(scope_arg="predictions_dir", scope_via=resolve_output_path)
def clear_prediction_bucket(
    predictions_dir: str, reason: str, cleared_bucket: str | None = None,
) -> dict:
    """Move a terminal experiment's own recorded prediction bucket into a dated archive under
    ``predictions/.cleared/``, so the path re-publishes.

    ``reason`` is required and non-empty, recorded as this door's own statement (never a ``user:``
    name minted from the string). The bucket cleared is ``resolve_output_path(predictions_dir)``,
    and it must be a canonical bucket under a dataset root (``predictions/<model>[/<date>]``).

    Refuses, before any write, each with its own sentence: an empty reason; a bucket already under
    the cleared archive; a bespoke bucket; a bucket carrying no ``operating_point.json`` stamp (and
    no resumable clear on record); any stamp that will not decode; a stamp naming a whole-raster
    pass; a stamp naming no experiment, one that is not terminal (``completed`` or ``failed``), or
    one whose recorded ``lineage.predictions`` is not this path; a bucket with no document to
    clear; an interrupted clear of this bucket already on record and unfinished; and a bucket
    carrying review state.

    The review-state refusal is a preflight over the state present when this call resolved the
    bucket, bucket-wide, run only on a fresh call, never on a resume (``cleared_bucket`` given).
    Review state that lands on the source between that preflight and the last document's delete is
    counted after the last delete and reported in ``review_state_landed_during_clear``. A publisher
    that resolved this bucket clean before the clear began can still write into it; a resume that
    finds a fresh ``operating_point`` stamp beside documents not yet moved reads it as a
    re-publication and leaves those documents where they landed.

    The move goes through the storage seam one key at a time (``operating_point`` first, then every
    other stamp present, then the documents), the artifact recorded on the experiment before any
    write, so a crash at any point leaves a state this door can finish: call again, naming
    ``cleared_bucket`` as the archive path this call (or an earlier refusal) reports; a resume
    creates the destination directory itself. A keyword-less call this door's own record shows as
    interrupted refuses naming that remedy. A conditional write inside a reconcile step that lands
    on a version another writer changed is refused naming that key, with everything already moved
    standing.

    Values move by value (a stamp under the record codec, a document byte for byte), never
    re-validated. The moved ``operating_point`` stamp keeps its ``validated: true`` and
    ``validated_by`` pointer, but the validation record names the source's own dataset-relative
    key, so a cleared bucket's stamp floors at every delivery door.

    Returns ``predictions_dir``, ``cleared_bucket``, ``resumed``, ``source_republished``,
    ``documents_moved_this_call``, ``stamps_moved_this_call``,
    ``review_state_landed_during_clear``, ``source_digest_before_call`` (the source's own document
    digest as this call found it), ``experiment_id``, ``checkpoint_sha256`` and ``validated_by``
    (read from wherever the ``operating_point`` stamp sits), ``cleared_artifact_recorded`` and
    ``reason``.
    """
    from tcip_mcp.dataset_layout import (
        canonical_prediction_bucket, cleared_bucket_of, cleared_prediction_dir,
        current_cleared_stamp, is_cleared_bucket,
    )
    from tcip_mcp.experiments import (
        _TERMINAL_STATES, lineage_key, read_member, record_artifact, status_key,
    )
    from tcip_mcp.pipelines.resolution import sidecar_key, stamp_names_raster
    from tcip_mcp.prediction_buckets import (
        bucket_content_digest, bucket_key_of, bucket_stems, review_state_count, review_state_dir_of,
    )
    from tcip_store import StoreError, VersionConflict

    if not (reason or "").strip():
        return {"error": "clear_prediction_bucket needs a non-empty reason: the confirmation "
                         "with the person this destructive act requires."}

    source = resolve_output_path(predictions_dir)

    if is_cleared_bucket(source):
        return {"error": f"{source} is already under the cleared archive; a bucket clears once."}

    canonical = canonical_prediction_bucket(source)
    if canonical is None:
        return {"error": f"{source} is not a canonical prediction bucket under a dataset root; "
                         "publish into the dataset's own predictions/<model>[/<date>] layout, "
                         "which is what this door clears, rather than a bespoke path."}
    dataset_root, model, date = canonical

    try:
        source_op = store.read_versioned(sidecar_key(source, "operating_point"), default=None)
    except StoreError as exc:
        return {"error": f"{source}: operating_point.json will not decode ({exc}); refused as "
                         "unreadable, since the record codec cannot move what it cannot decode."}

    review_state_dir = review_state_dir_of(dataset_root)
    source_key = bucket_key_of(source)
    resuming = cleared_bucket is not None
    destination: Path

    if source_op.value is None and not resuming:
        found = _find_cleared_candidate_with_no_source_stamp(dataset_root, model, date)
        if found is not None:
            return {"error": f"{source} carries no operating_point.json; this is not a "
                             "published bucket. A candidate on record may finish it: call again "
                             f"with cleared_bucket={found['path']!r}."}
        return {"error": f"{source} carries no operating_point.json and no cleared bucket on "
                         "record names it; this is not a published bucket."}

    if resuming:
        assert cleared_bucket is not None  # resuming is exactly cleared_bucket is not None
        destination = resolve_output_path(cleared_bucket)
        resolved = cleared_bucket_of(destination)
        if resolved is None or not _cleared_artifact_matches(str(destination), dataset_root, model, date):
            return {"error": f"{cleared_bucket!r} does not name {source}'s own cleared "
                             "destination."}
        try:
            dest_op = store.read_versioned(sidecar_key(destination, "operating_point"), default=None)
        except StoreError as exc:
            return {"error": f"{destination}: operating_point.json will not decode ({exc})."}
        if source_op.value is not None:
            live_stamp, live_bucket = source_op.value, source
        else:
            if dest_op.value is None:
                return {"error": f"operating_point.json is present at neither {source} nor "
                                 f"{destination}: a store written past this door."}
            live_stamp, live_bucket = dest_op.value, destination
        for document in _OTHER_STAMP_DOCUMENTS:
            try:
                store.read_versioned(sidecar_key(destination, document), default=None)
            except StoreError as exc:
                return {"error": f"{destination}: {document}.json will not decode ({exc})."}
    else:
        live_stamp, live_bucket = source_op.value, source

    for document in _OTHER_STAMP_DOCUMENTS:
        try:
            store.read_versioned(sidecar_key(source, document), default=None)
        except StoreError as exc:
            return {"error": f"{source}: {document}.json will not decode ({exc})."}

    if stamp_names_raster(live_stamp):
        return {"error": f"{live_bucket}: operating_point.json names raster_path; the raster "
                         "regime keeps its own resume state under <bucket>/.tcip/ and is out of "
                         "scope for this door."}

    experiment_id = live_stamp["experiment_id"]
    if not experiment_id:
        return {"error": f"{live_bucket}: operating_point.json names no experiment_id; "
                         "clear_prediction_bucket clears only a bucket a run recorded against a "
                         "specific experiment."}
    state = (read_member(status_key(experiment_id), {}) or {}).get("state")
    if state not in _TERMINAL_STATES:
        return {"error": f"experiment {experiment_id!r} is {state!r}, not terminal (completed or "
                         "failed); this door clears only a terminal experiment's own recorded "
                         "bucket. A non-terminal experiment's bucket already republishes in "
                         "place through run_inference's own <name>@r<n> redirect."}
    lineage = read_member(lineage_key(experiment_id), {}) or {}
    if lineage.get("predictions") != str(source):
        return {"error": f"experiment {experiment_id!r}'s recorded lineage.predictions is "
                         f"{lineage.get('predictions')!r}, not {str(source)!r}; this door clears "
                         "only the bucket an experiment's own pointer names."}

    if resuming:
        newest = _newest_cleared_artifact_for_source(experiment_id, dataset_root, model, date)
        if newest is None:
            return {"error": f"no cleared: artifact on experiment {experiment_id!r} names "
                             f"{cleared_bucket!r} as {source}'s own cleared destination."}
        if not same_directory(newest["path"], destination):
            return {"error": f"{cleared_bucket!r} is not the newest cleared bucket on record for "
                             f"{source}; the newest is {newest['name']!r}, naming "
                             f"{newest['path']!r}. Call again with cleared_bucket="
                             f"{newest['path']!r} to finish the interrupted clear."}
    else:
        if not bucket_stems(source):
            return {"error": f"{source} holds no prediction document; nothing to clear. A "
                             "stamp-only bucket is already re-publishable in place through "
                             "run_inference."}
        unfinished = _newest_cleared_artifact_for_source(experiment_id, dataset_root, model, date)
        if unfinished is not None and not bucket_stems(Path(unfinished["path"])):
            return {"error": f"an earlier clear of {source} is on record and unfinished "
                             f"({unfinished['name']!r} names {unfinished['path']!r}, which holds "
                             f"no document yet). Call again with "
                             f"cleared_bucket={unfinished['path']!r} to finish it."}

    if not resuming:
        count = review_state_count(review_state_dir, source_key)
        if count:
            noun = "entry" if count == 1 else "entries"
            return {"error": f"{source} carries {count} review {noun} (a detection verdict or a "
                             "bulk-accepted image); this door refuses a bucket carrying review "
                             "state, not only its documents."}

    source_digest_before_call = bucket_content_digest(source)
    cleared_artifact_recorded = True

    if not resuming:
        destination = cleared_prediction_dir(dataset_root, model, date, current_cleared_stamp())
        if destination.exists():
            return {"error": f"{destination} already exists (a same-second collision with "
                             "another clear); wait a second and retry."}
        from tcip_mcp.pipelines.resolution import bucket_relative_key

        relative_key = bucket_relative_key(destination, dataset_root, document="cleared")
        artifact = record_artifact(experiment_id, f"cleared:{relative_key}", str(destination))
        if "error" in artifact:
            return {"error": f"could not record the cleared artifact on {experiment_id!r}: "
                             f"{artifact['error']}"}

    destination.mkdir(parents=True, exist_ok=True)

    stamps_moved_this_call: list[str] = []
    documents_moved_this_call = 0
    source_republished = False
    # Whether operating_point.json had already left source before this call, the discriminator
    # _reconcile_secondary_stamp uses between this call's own unmoved stamp and a fresh write.
    op_stamp_left_source = resuming and source_op.value is None

    try:
        if len(bucket_stems(destination)) == 0:
            moved, refusal = _reconcile_op_stamp(source, destination)
            if refusal is not None:
                return refusal
            if moved:
                stamps_moved_this_call.append("operating_point")
            for document in _OTHER_STAMP_DOCUMENTS:
                moved, refusal = _reconcile_secondary_stamp(
                    source, destination, document, op_stamp_left_source=op_stamp_left_source)
                if refusal is not None:
                    return refusal
                if moved:
                    stamps_moved_this_call.append(document)
            for stem in sorted(bucket_stems(source) | bucket_stems(destination)):
                moved, refusal = _reconcile_document(source, destination, stem)
                if refusal is not None:
                    return refusal
                if moved:
                    documents_moved_this_call += 1
        elif store.read(sidecar_key(source, "operating_point"), default=None) is not None:
            source_republished = True
        else:
            for document in _OTHER_STAMP_DOCUMENTS:
                moved, refusal = _reconcile_secondary_stamp(
                    source, destination, document, op_stamp_left_source=op_stamp_left_source)
                if refusal is not None:
                    return refusal
                if moved:
                    stamps_moved_this_call.append(document)
            for stem in sorted(bucket_stems(source) | bucket_stems(destination)):
                moved, refusal = _reconcile_document(source, destination, stem)
                if refusal is not None:
                    return refusal
                if moved:
                    documents_moved_this_call += 1
    except VersionConflict as exc:
        moved_note = (
            f"{len(stamps_moved_this_call)} stamp(s) ({', '.join(stamps_moved_this_call)}) and "
            f"{documents_moved_this_call} document(s) already moved this call"
        )
        return {"error": f"{exc.key.store}{list(exc.key.parts)} changed under the door while "
                         f"clearing {source} into {destination}: {moved_note} stand. Call again "
                         f"with cleared_bucket={str(destination)!r} to finish it."}

    if not source_republished:
        arrived = bucket_stems(source)
        if arrived:
            return {"error": f"{sorted(arrived)} arrived at {source} during the clear (the "
                             "staging door admits a stampless bucket); call again with "
                             f"cleared_bucket={str(destination)!r} to move them."}

    review_state_landed_during_clear = review_state_count(review_state_dir, source_key)

    return {
        "predictions_dir": str(source),
        "cleared_bucket": str(destination),
        "resumed": resuming,
        "source_republished": source_republished,
        "documents_moved_this_call": documents_moved_this_call,
        "stamps_moved_this_call": stamps_moved_this_call,
        "review_state_landed_during_clear": review_state_landed_during_clear,
        "source_digest_before_call": source_digest_before_call,
        "experiment_id": experiment_id,
        "checkpoint_sha256": live_stamp["checkpoint_sha256"],
        "validated_by": live_stamp["validated_by"],
        "cleared_artifact_recorded": cleared_artifact_recorded,
        "reason": reason,
    }


# --- resuming an interrupted tiled raster pass (the raster regime only) ---


def _raster_pass_key(bucket: Path, segment: str) -> Key:
    """One raster pass' progress record under ``bucket``: the identity (``segment="identity"``)
    or one flushed tile batch (``segment=f"batch-{index:06d}"``)."""
    return Key(RASTER_PASS_PROGRESS_STORE, str(bucket), (segment,))


def _raster_pass_identity_body(
    *, raster_identity: dict, checkpoint_sha256: str | None, trait: str | None,
    experiment_id: str | None, tile_batch_size: int, conf: float, cross_tile_nms: float | None,
    max_dets: int | None, tile_size: int, overlap: float, tile_resize: tuple[int, int] | None,
    postprocess: str, require_masks: bool,
) -> dict:
    """The pass a raster-export bucket is mid-way through, as the plain dict a later resume
    compares its own call against. The device is deliberately absent: the same detections on
    other hardware are the same pass."""
    return {
        "schema_version": _RASTER_PASS_PROGRESS_SCHEMA_VERSION,
        "raster_identity": raster_identity,
        "checkpoint_sha256": checkpoint_sha256,
        "trait": trait,
        "experiment_id": experiment_id,
        "tile_batch_size": tile_batch_size,
        "operating_point": {
            "conf": conf,
            "cross_tile_nms": cross_tile_nms,
            "max_dets": max_dets,
            "tile_size": tile_size,
            "overlap": overlap,
            "tile_resize": list(tile_resize) if tile_resize is not None else None,
            "postprocess": postprocess,
            "require_masks": require_masks,
        },
    }


def _differing_fields(recorded: dict, current: dict, excluded: frozenset = frozenset()) -> list[str]:
    """Every field over both sides' keys, less ``excluded``, that one side lacks or the two state
    differently; ``None`` is a value like any other."""
    return [field for field in sorted((set(recorded) | set(current)) - excluded)
            if field not in recorded or field not in current or recorded[field] != current[field]]


def _raster_pass_input_mismatches(recorded: dict, current: dict) -> list[str]:
    """Every top-level identity field (everything but ``schema_version`` and ``operating_point``,
    compared by :func:`_raster_pass_identity_mismatches`) naming a difference between a recorded
    raster-pass identity and this call's own.
    """
    return _differing_fields(recorded, current, frozenset({"schema_version", "operating_point"}))


def _raster_pass_identity_mismatches(recorded: dict, current: dict) -> list[str]:
    """Every field naming a difference between a recorded raster-pass identity and this call's
    own, for a resume refusal to list by name."""
    return _raster_pass_input_mismatches(recorded, current) + [
        f"operating_point.{field}"
        for field in _differing_fields(recorded["operating_point"], current["operating_point"])]


def _load_raster_pass_prior(bucket: Path) -> dict:
    """Every tile batch a bucket's progress already holds, merged in tile order (by the numeric
    index parsed out of each ``batch-<index>`` key) into the shape
    ``GenericPredictor._tiled_infer_core`` seeds its own accumulators from.
    """
    indexed: list[tuple[int, Key]] = []
    for key in store.keys(RASTER_PASS_PROGRESS_STORE, str(bucket)):
        segment = key.parts[0]
        if not segment.startswith("batch-"):
            continue
        indexed.append((int(segment[len("batch-"):]), key))
    tile_info: list[dict] = []
    boxes: list = []
    scores: list = []
    labels: list = []
    for _index, key in sorted(indexed):
        batch = store.read(key)
        tile_info.extend(batch["tile_info"])
        boxes.extend(batch["boxes"])
        scores.extend(batch["scores"])
        labels.extend(batch["labels"])
    return {"tile_info": tile_info, "boxes": boxes, "scores": scores, "labels": labels}


def _clear_raster_pass_progress(bucket: Path) -> None:
    """Delete every progress record a raster pass over ``bucket`` left, in one transaction: a
    completed pass has nothing left to resume."""
    keys = store.keys(RASTER_PASS_PROGRESS_STORE, str(bucket))
    if not keys:
        return
    with store.transaction(*keys) as txn:
        for key in keys:
            txn.delete(key)


_SNAPSHOT_RESULT_KEYS = ("validated", "conf_source", "dataset_hash", "shippable_issues")
"""The run facts a block-calibrated raster pass records beside its identity, so a resume states
them as the interrupted attempt earned them rather than calibrating again."""


def _export_predictions_raster(
    *, checkpoint, raster_path: str, out: Path, resolution, output_dir: str,
    dataset_root: Path | None, device: str | None, conf_threshold: float | None,
    tile_size: int | None, overlap: float | None, tile_batch_size: int,
    global_nms_iou: float | None, max_dets: int | None, postprocess: str, require_masks: bool,
    experiment_id: str | None, allow_unvalidated_staging: bool, trait: str | None = None,
    resume: bool = False, overwrite: bool = False,
) -> dict:
    """The raster regime of :func:`run_inference`: one always-tiled pass over a raster read
    window by window (:func:`~tcip_mcp.pipelines.raster_source.open_raster`), published through
    :func:`publish_bucket` as one ``<raster stem>.json`` document in full-raster pixel space.

    ``out``/``resolution`` are the bucket :func:`run_inference` resolved for ``output_dir`` and
    ``dataset_root`` its dataset root. The pass is prepared by :func:`_prepare_pass` with no image
    list; a checkpoint with no basis for its tile edge refuses there, whatever
    ``allow_unvalidated_staging`` says. With ``trait`` ``None`` the pass runs at the stated or
    default operating point, unvalidated. With a ``trait`` (passed only once the checkpoint's
    training experiment reserved a calibration region) it runs block calibration first, refuses
    unless the block-validated reference is this raster (the claim-scope gate), then runs the
    whole-mosaic pass at the calibrated conf and cross-tile NMS with the full-frame cap lifted.

    A pass that is not mask-bearing (``instance_seg`` with ``require_masks``) records its identity
    and each flushed tile batch under ``<out>/.tcip/``, once the publisher's gates have passed.
    ``resume=True`` continues that record: the recorded checkpoint, raster content, trait,
    experiment, tile batch size and operating point must equal this call's, or it refuses naming
    what differs; a block-calibrated pass resumes at the operating point the interrupted attempt
    earned. ``resume=False`` over a bucket carrying progress refuses unless ``overwrite=True``,
    which discards it. A published pass deletes its progress.
    """
    p = _prepare_pass(
        checkpoint, images_dir=None, conf_threshold=conf_threshold, device=device, tile=True,
        tile_size=tile_size, overlap=overlap, global_nms_iou=global_nms_iou, max_dets=max_dets,
        postprocess=postprocess, experiment_id=experiment_id, tile_batch_size=tile_batch_size)
    if isinstance(p, str):
        return {"error": p}
    assert p.tile_size is not None  # a tiled pass with no edge refuses in _prepare_pass
    predictor, identity = p.predictor, p.identity

    if resume and predictor.task == "instance_seg" and require_masks:
        return {"error": (
            "resume=True refuses for a mask-bearing pass: an instance_seg checkpoint's tiled "
            "masks have no persisted per-batch representation this door records, so a "
            "mask-bearing raster pass records no progress and cannot be resumed. Run the whole "
            "pass again instead."
        )}
    record_progress = not (predictor.task == "instance_seg" and require_masks)

    import dataclasses

    from tcip_mcp.pipelines.raster_source import content_identity

    try:
        raster_identity = dataclasses.asdict(content_identity(raster_path, predictor.in_chans))
    except ValueError as exc:
        return {"error": f"raster content identity could not be computed for {raster_path}: {exc}"}
    # Round-tripped through the record codec so a tuple field (band_interpretations) reads the
    # same shape a recorded identity would, and later comparison is a plain dict equality.
    raster_identity = RECORD_JSON.decode(RECORD_JSON.encode(raster_identity))

    identity_key = _raster_pass_key(out, "identity")
    existing_pass_identity = store.read(identity_key, default=None)
    if resume and existing_pass_identity is None:
        return {"error": (
            f"resume=True but {out} carries no raster-pass identity record to resume from: "
            "either no pass has started here, or a prior pass already completed and cleared "
            "its own progress."
        )}
    if existing_pass_identity is not None and not resume:
        if overwrite:
            _clear_raster_pass_progress(out)
            existing_pass_identity = None
        else:
            return {"error": (
                f"{out} carries progress from an interrupted raster pass: pass resume=True to "
                "continue it, or overwrite=True to discard it and start over."
            )}
    if existing_pass_identity is not None:
        recorded_schema_version = existing_pass_identity["schema_version"]
        if (not isinstance(recorded_schema_version, int)
                or recorded_schema_version > _RASTER_PASS_PROGRESS_SCHEMA_VERSION):
            return {"error": (
                f"{out}'s raster-pass identity record is schema_version "
                f"{recorded_schema_version!r}, above the {_RASTER_PASS_PROGRESS_SCHEMA_VERSION} "
                "this reader knows: a newer writer produced it than this code understands."
            )}

    if resume:
        current_inputs = {
            "checkpoint_sha256": identity["sha256"], "trait": trait,
            "experiment_id": identity["experiment_id"], "tile_batch_size": tile_batch_size,
            "raster_identity": raster_identity,
        }
        input_mismatches = _raster_pass_input_mismatches(existing_pass_identity, current_inputs)
        if input_mismatches:
            return {"error": (
                f"resume=True but the recorded pass over {out} differs from this call in "
                f"{input_mismatches}: a resumed pass must be the identical pass, since merging "
                "tiles run at two different operating points would corrupt the count."
            )}

    from tcip_mcp.pipelines.operating_point import apply_operating_point
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE, block_calibrated_export_operating_point, check_delivery_gate,
    )

    stamp_extras: dict = {"raster_content_identity": raster_identity}
    claim_evidence: dict | None = None
    claim_scope_mismatch: str | None = None
    snapshot: dict | None = None
    if trait is None:
        result = p.raw_result()
    elif resume:
        snapshot = store.read(_raster_pass_key(out, "block-calibration"))
        recorded_op = existing_pass_identity["operating_point"]
        p.conf, p.nms_iou = recorded_op["conf"], recorded_op["cross_tile_nms"]
        apply_operating_point(predictor, p.conf, recorded_op["max_dets"])
        result = p.result(snapshot["operating_point"],
                          {k: snapshot[k] for k in _SNAPSHOT_RESULT_KEYS})
        stamp_extras.update(snapshot["stamp_extras"])
        claim_evidence = snapshot["block_evidence"]
    else:
        from tcip_mcp.pipelines.block_calibration import resolve_block_calibration_records
        from tcip_mcp.pipelines.raster_source import (
            georeferenced_raster_identity_mismatch, raster_identity_matches,
        )
        from tcip_mcp.pipelines.resolution import (
            VALIDATED_SAME_MOSAIC_CONTENT_IDENTITY, VALIDATED_SAME_MOSAIC_IDENTITY,
        )

        try:
            block_bundle, block_prov, block_evidence = resolve_block_calibration_records(
                predictor, trait_name=trait,
                experiment_id=identity["experiment_id"], global_nms_iou=p.nms_iou,
                export_tile_size=p.tile_size,
                tile_batch_size=tile_batch_size, postprocess=postprocess,
            )
        except ValueError as exc:  # a named block refusal, or the run's scope refused
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
        gate = check_delivery_gate(
            {"claim_scope": claim_scope_flag}, allow_unvalidated_staging=allow_unvalidated_staging)
        if not gate.ok:
            reason = gate.reason if claim_scope_mismatch is None else (
                f"{gate.reason} {claim_scope_mismatch}")
            return {"error": reason, "claim_scope_validated": claim_scope_flag}
        claim_scope_validated = gate.stamp.get("claim_scope")

        conf_param = block_bundle.get("conf")
        p.conf = (conf_param.value if conf_param.is_shippable
                  else conf_param.unvalidated_value(acknowledge_unvalidated=True))
        p.nms_iou = float(block_bundle.get("cross_tile_nms").value)
        # The whole mosaic runs uncapped, never at the block bundle's band-scoped density cap.
        apply_operating_point(predictor, p.conf, None)

        op_bundle = block_calibrated_export_operating_point(
            block_bundle, trait=trait, tile_size=p.tile_size,
            tile_size_source=p.tile_size_source, tile_size_derived_from=p.tile_size_derived_from)
        result = p.result(op_bundle.to_provenance()["operating_point"], {
            "validated": op_bundle.is_shippable and claim_scope_validated != VALIDATED_FALSE,
            "conf_source": "block_calibration", "dataset_hash": op_bundle.dataset_hash,
            "shippable_issues": op_bundle.shippable_issues()})
        stamp_extras.update(
            claim_scope_validated=claim_scope_validated,
            # The spatial manifest is already carried on the training experiment's own split.json.
            block_calibration={k: v for k, v in block_prov.items() if k != "spatial_manifest"})
        claim_evidence = block_evidence
        snapshot = {
            "operating_point": result["operating_point"],
            **{k: result[k] for k in _SNAPSHOT_RESULT_KEYS},
            "stamp_extras": {k: stamp_extras[k]
                             for k in ("claim_scope_validated", "block_calibration")},
            "block_evidence": block_evidence,
        }
    result["raster_path"] = str(raster_path)

    current_pass_identity = _raster_pass_identity_body(
        raster_identity=raster_identity, checkpoint_sha256=identity["sha256"], trait=trait,
        experiment_id=identity["experiment_id"], tile_batch_size=tile_batch_size,
        conf=p.conf, cross_tile_nms=p.nms_iou, max_dets=predictor.max_dets,
        tile_size=p.tile_size, overlap=p.overlap, tile_resize=p.tile_resize,
        postprocess=postprocess, require_masks=require_masks,
    )
    if existing_pass_identity is not None:
        mismatches = _raster_pass_identity_mismatches(existing_pass_identity, current_pass_identity)
        if mismatches:
            return {"error": (
                f"resume=True but the recorded pass over {out} differs from this call in "
                f"{mismatches}: a resumed pass must be the identical pass, since merging tiles "
                "run at two different operating points would corrupt the count."
            )}

    def _record_raster_pass_batch(start_index: int, _end_index: int, batch: dict) -> None:
        # Six digits also sorts the key into tile order lexically (covers any grid this
        # platform tiles); _load_raster_pass_prior parses the index itself rather than lean on that.
        store.replace(
            _raster_pass_key(out, f"batch-{start_index:06d}"),
            batch, expect=Version.ABSENT,
        )

    passed: dict = {}

    def raster_pass():
        """The one pass, run once the publisher's gates have passed: its progress record first,
        then the tiles."""
        from tcip_mcp.pipelines.raster_source import open_raster

        prior = None
        if existing_pass_identity is not None:
            prior = _load_raster_pass_prior(out)
        elif record_progress:
            store.replace(identity_key, current_pass_identity, expect=Version.ABSENT)
            if snapshot is not None:
                store.replace(_raster_pass_key(out, "block-calibration"), snapshot,
                              expect=Version.ABSENT)
        # The model's own in_chans is the channel routing hint; the reader's real band count is
        # checked against it inside predict_tiled before any tile is read.
        with open_raster(raster_path, predictor.in_chans) as reader:
            tiled = predictor.predict_tiled(
                reader, tile_size=p.tile_size, overlap=p.overlap,
                tile_batch_size=tile_batch_size, global_nms_iou=p.nms_iou,
                postprocess=postprocess, require_masks=require_masks,
                source_label=str(raster_path), tile_resize=p.tile_resize, prior=prior,
                progress=_record_raster_pass_batch if record_progress else None,
            )
        passed["tiles"] = tiled.get("tiles")
        yield tiled

    result["results"] = raster_pass()
    pub = publish_bucket(
        result, out=out, trait=trait, dataset_root=dataset_root,
        allow_unvalidated_staging=allow_unvalidated_staging, claim_evidence=claim_evidence,
        stamp_extras=stamp_extras)
    if pub["refusal"] is not None:
        return pub["refusal"]
    _clear_raster_pass_progress(out)

    response = {
        **_bucket_response(pub, out=out, resolution=resolution, dataset_root=dataset_root,
                           requested_output_dir=output_dir),
        "image_count": 1, "operating_point": result["operating_point"],
        "conf_source": result["conf_source"], "checkpoint_sha256": identity["sha256"],
        "experiment_id": identity["experiment_id"], "tiles": passed["tiles"],
    }
    if "claim_scope_validated" in stamp_extras:
        response["claim_scope_validated"] = stamp_extras["claim_scope_validated"]
        if claim_scope_mismatch is not None:
            response["claim_scope_note"] = claim_scope_mismatch
    return response


@mcp.tool()
def deliver_per_image_counts(
    checkpoint_path: str | None = None,
    images_dir: str | None = None,
    output_path: str = "",
    *,
    trait: str,
    conf_threshold: float | None = None,
    device: str | None = None,
    tile: bool | None = None,
    tile_size: int | None = None,
    overlap: float | None = None,
    tile_batch_size: int = DEFAULT_TILE_BATCH_SIZE,
    global_nms_iou: float | None = None,
    max_dets: int | None = None,
    postprocess: str = DEFAULT_POSTPROCESS,
    calibration_labels_dir: str | None = None,
    calibration_images_dir: str | None = None,
    selection_dir: str | None = None,
    experiment_id: str | None = None,
    allow_unvalidated_staging: bool = False,
    predictions_dir: str | None = None,
) -> dict:
    """Export a CSV summary of detection counts per image, from a live run or a persisted bucket.

    Two source regimes, exactly one stated:

    - Live (``checkpoint_path`` and ``images_dir``, both required together): the verified pass at
      the operating point ``run_inference`` resolves. Passing ``predictions_dir`` too persists the
      run's own predictions into that bucket through :func:`publish_bucket`
      (``allow_unvalidated_staging`` clears only its own tile-scale staging gate, never the CSV's
      own delivery gate), with ``run_inference``'s bucket-immutability resolution, and then reads
      the CSV's own validity back off the bucket it just wrote. A document refusal is returned
      before the checkpoint is loaded. Without ``predictions_dir`` an unvalidated live pass cannot
      be delivered. A published bucket can be promoted to validated later through the review
      validation route (``validate_reference``), then re-delivered through the bucket regime.
    - Bucket (``predictions_dir`` alone, ``checkpoint_path``/``images_dir`` both absent): no GPU,
      no predictor import. Reads an existing per-image prediction bucket's own
      ``operating_point.json`` stamp as its identity and validity source, counting real detections
      (a ``Point`` excluded) off each of its documents, sorted by stem. Every parameter meaningful
      only to a live run (conf/device/tiling/NMS/max_dets/calibration/selection/experiment_id)
      refuses here by name; ``postprocess``/``tile_batch_size`` refuse only away from their own
      documented default. A bucket recording ``raster_path`` refuses naming
      ``deliver_orthomosaic_plant_counts``. A stamp recording a different, non-``None`` trait
      refuses.

    Delivery gate, both regimes: inside ``export_detection_csv``, over the count operating point
    and (if tiled) the tile geometry, reconciled from the bucket's own sidecar; ships only when
    every dimension clears. This tool builds no acknowledgment, so an unvalidated dimension always
    refuses here. A refused delivery still names what happened to a ``predictions_dir`` the live
    regime published before the CSV's own gate ran (``output_dir``, ``files``,
    ``bucket_redirected``, ``lineage_linked``, beside ``csv_delivered: false``).

    Every row's image cell holds the source image's basename with its extension. The live regime
    without ``predictions_dir`` reads it off the pass's own per-image results; a bucket-reading
    path resolves it through the bucket's own stamp-recorded ``image_filenames`` map, and refuses a
    bucket whose stamp does not name each of its documents.

    ``trait``'s per-image-count operationalization must be recorded and breeder-confirmed, checked
    before the pass runs (live) or the bucket is read (bucket regime). Only the CSV's own
    delivery-gate refusal returns ``image_count`` and ``total_detections`` beside the error; an
    operationalization refusal, and every refusal the publisher raises before the CSV's own gate
    runs (a fabricated tile scale, an unearned count claim, a frozen lineage pointer), carry
    neither.

    The live regime's ``checkpoint_sha256``/``experiment_id`` are the run's asserted identity; the
    bucket regime's are the stamp's asserted identity, with no ``conf_source``. The CSV's own
    ``producer_model_sha256``/``producing_experiment_id`` columns, and this response's
    ``operating_point_validated``, are ``export_detection_csv``'s returned tail, corroborated
    against a record outside the stamp, so they can differ from the asserted identity.
    ``validated`` is live-only: the run's own verdict over the dimensions it resolved.
    ``unvalidated_dimensions`` names every gated dimension that did not validate.
    ``tile_size_validated`` is the gate's outcome when a bucket backs the delivery; on the live
    regime with no ``predictions_dir`` it is the run's own in-memory tile-scale flag, the CSV's
    ``unvalidated_dimensions`` can name only ``operating_point``, and the response adds
    ``run_conf_validated_against``, the run's own narrowed conf reference.

    Args:
        checkpoint_path: Path to model .pt checkpoint (live regime; required with ``images_dir``,
            absent for the bucket regime). Must be registered under this process's platform state
            root (``register_model``, explicit mode for a foreign or bespoke checkpoint) or this
            door refuses before loading it.
        images_dir: Directory containing input images (live regime; required with
            ``checkpoint_path``, absent for the bucket regime).
        output_path: Path for the output CSV file. Required; a relative path resolves against the
            platform state root, never the server process's cwd.
        trait: The registered trait whose confirmed per-image-count operationalization this
            delivery rests on. Required, in both regimes.
        conf_threshold: Live regime only. Minimum confidence score. ``None`` (default) runs at the
            platform default; a stated value is an explicit override even when it equals the
            platform default.
        device: Live regime only. Device to use.
        tile: Live regime only. Tiled (SAHI-style) inference for small dense objects; ``None``
            (default) as ``run_inference`` resolves it.
        tile_size: Live regime only. Sliding-window tile edge (px).
        overlap: Live regime only. Fractional tile overlap.
        tile_batch_size: Live regime only. Tiles per forward batch.
        global_nms_iou: Live regime only. Cross-tile NMS IoU; as ``run_inference`` resolves it.
        max_dets: Live regime only. Full-frame detection cap; as ``run_inference`` resolves it.
        postprocess: Live regime only. Cross-tile merge, "nms" or "nmm".
        calibration_labels_dir: Live regime only. Labeled dir for calibrating + held-out validating
            the operating point.
        calibration_images_dir: Live regime only. Images for the calibration labels (defaults to
            ``images_dir``).
        selection_dir: Live regime only. Restrict calibration to a selection's ``calibration``
            samples under the calibration labels directory (see ``run_inference``).
        experiment_id: Live regime only. The run that produced the checkpoint, by its record id
            (``tcip_mcp.experiments``), for provenance (see ``run_inference``).
        allow_unvalidated_staging: Live regime with ``predictions_dir`` only. Persist the bucket
            even when tile_size has no real basis, stamping ``tile_size_validated=false``; never a
            route to deliver the CSV itself unvalidated.
        predictions_dir: Live regime: directory to persist the counted predictions into, resolved
            and stamped the way ``run_inference`` resolves and stamps a bucket. Bucket regime: the
            existing bucket to read (required, resolved the same way; no redirect, since nothing is
            written).
    """
    from tcip_mcp.operationalization import (
        PER_IMAGE_COUNT,
        OperationalizationRefused,
        check_operationalization,
        resolve_trait_and_record,
    )
    from tcip_mcp.pipelines.resolution import CountDeliveryRefused
    from tcip_mcp.project_paths import resolve_output_path
    from tcip_mcp.traits import TraitUnknownError

    live = checkpoint_path is not None or images_dir is not None
    if live and (checkpoint_path is None or images_dir is None):
        return {"error": "the live regime requires both checkpoint_path and images_dir."}
    if not live and predictions_dir is None:
        return {"error": (
            "Provide either checkpoint_path and images_dir (a live run) or predictions_dir naming "
            "an existing, reviewed prediction bucket (a bucket regime call)."
        )}
    if live:
        assert checkpoint_path is not None  # the regime check above already requires it when live
        if not Path(checkpoint_path).is_file():
            return {"error": f"Checkpoint not found: {checkpoint_path}"}
    else:
        # A live-only parameter stated at its own default is indistinguishable from unstated.
        defaults = inspect.signature(deliver_per_image_counts).parameters
        stated_live_only = sorted(
            name for name, value in (
                ("conf_threshold", conf_threshold), ("device", device), ("tile", tile),
                ("tile_size", tile_size), ("overlap", overlap),
                ("global_nms_iou", global_nms_iou), ("max_dets", max_dets),
                ("calibration_labels_dir", calibration_labels_dir),
                ("calibration_images_dir", calibration_images_dir),
                ("selection_dir", selection_dir), ("experiment_id", experiment_id),
                ("postprocess", postprocess), ("tile_batch_size", tile_batch_size),
            ) if value != defaults[name].default)
        if stated_live_only:
            return {"error": (
                f"{stated_live_only} only apply to the live regime (checkpoint_path + "
                "images_dir); a bucket regime call (predictions_dir alone) reads an existing "
                "bucket's own stamp and cannot honor a stated live parameter, including one "
                "stated at its own documented default, since that is indistinguishable from "
                "stating nothing."
            )}
    if not output_path:
        return {"error": "output_path is required"}
    output_path = str(resolve_output_path(output_path))

    if not live:
        assert predictions_dir is not None  # the regime check above already requires it
        try:
            return per_image_counts_from_bucket(
                predictions_dir, output_path, trait=trait, project_root=None,
                acknowledgment=None)
        except OperationalizationRefused as exc:
            return {"error": exc.check.message}
        except DeliveryRefused as exc:
            reason = (
                f"{exc} This door takes no acknowledgment: acknowledge and re-export through "
                "the Results tab's count export, or validate the dimension named above, or "
                "promote this bucket to validated through the review validation route "
                "(validate_reference) and re-deliver."
            )
            return {"error": reason, **exc.facts}
        except CountDeliveryRefused as exc:
            return {"error": str(exc), **exc.facts}

    # live is True here (the bucket regime above always returns); the regime check higher up
    # already requires both when live.
    assert checkpoint_path is not None and images_dir is not None

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

    from tcip_mcp.model_registry import UnregisteredCheckpoint, load_registered_checkpoint

    try:
        checkpoint = load_registered_checkpoint(checkpoint_path)
    except UnregisteredCheckpoint as exc:
        return {"error": str(exc)}

    result = _run_inference_verified(
        checkpoint,
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
        selection_dir=selection_dir,
        experiment_id=experiment_id,
    )
    if "error" in result:
        return result

    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE, accepted_references, tile_size_gate_flag,
    )
    from tcip_store import StoreError

    op = result["operating_point"]
    conf_prov = op.get("conf") or {}
    # Judged against the references accepted for conf's own kind, never the bare validated bool
    # (that would launder a missing/unrecognized/wrong-kind value into a shippable one).
    op_ref = conf_prov.get("validated_against")
    if op_ref not in accepted_references("annotations"):
        op_ref = VALIDATED_FALSE

    bucket_fields: dict = {}
    if bucket is None:
        # tile_size gates the way conf does: a no-basis tile scale is as untrustworthy for a count
        # as an uncalibrated conf; None if untiled. A published bucket's stamp carries its own.
        tile_ref = tile_size_gate_flag(op)
        csv_rows = list(result["results"])
        # A detection the writer would not store is no detection, so it is not counted either.
        total_detections = sum(positive_detections(r)[0] for r in csv_rows)
    else:
        assert resolution is not None  # bucket and resolution are set together, above
        pub = publish_bucket(
            result, out=bucket, trait=trait, dataset_root=bucket_root,
            allow_unvalidated_staging=allow_unvalidated_staging)
        if pub["refusal"] is not None:
            return pub["refusal"]
        tile_ref = pub["op_stamp"]["tile_size_validated"]
        bucket_fields = _bucket_response(pub, out=bucket, resolution=resolution,
                                         dataset_root=bucket_root,
                                         requested_output_dir=str(predictions_dir))
        # Counted off the just-published documents, filenames off the run's own just-written stamp.
        csv_rows = _bucket_csv_rows(bucket, pub["op_stamp"])
        total_detections = sum(r["count"] for r in csv_rows)

    provenance = {
        "producer_model_sha256": result["checkpoint_sha256"],
        "producing_experiment_id": result["experiment_id"],
        "operating_point_conf": (op.get("conf") or {}).get("value"),
    }
    try:
        csv_path, tail, summary, _event_recorded = export_detection_csv(
            csv_rows, output_path, provenance=provenance, trait=trait,
            operating_point_validated=op_ref,
            pred_dirs=[str(bucket)] if bucket is not None else None,
        )
    except StoreError as exc:
        return {"error": str(exc)}
    except OperationalizationRefused as exc:
        return {"error": exc.check.message}
    except DeliveryRefused as exc:
        reason = str(exc)
        if bucket is None:
            reason += (
                " These counts were read off an in-memory pass with no prediction bucket behind "
                "them: pass predictions_dir to persist and stamp the predictions they came from, "
                "which is what a validated count CSV rests on; an unvalidated bucket can also be "
                "promoted to validated later through the review validation route, with no re-run."
            )
        else:
            reason += (
                " This door takes no acknowledgment for the CSV itself: validate the dimension "
                "named above, or promote this bucket to validated through the review validation "
                "route (validate_reference) and re-deliver."
            )
        refusal = {
            **bucket_fields,
            "error": reason,
            "operating_point_validated": exc.gate.stamp.get("operating_point", VALIDATED_FALSE),
            "tile_size_validated": exc.gate.stamp.get("tile_size", tile_ref),
            "unvalidated_dimensions": exc.gate.unvalidated_cell(),
            "operating_point": result["operating_point"],
            "validated": False,
            "image_count": len(csv_rows),
            "total_detections": total_detections,
        }
        if bucket is None:
            # The run's own narrowed reference, distinct from the gate stamp above: nothing on
            # disk backs it without a bucket, so it never answers for operating_point_validated.
            refusal["run_conf_validated_against"] = op_ref
        else:
            refusal["csv_delivered"] = False
        return refusal

    out = {
        **bucket_fields,
        "csv_path": csv_path,
        "image_count": len(csv_rows),
        "total_detections": total_detections,
        # Carry the operating point + producing model that produced these counts, the CSV is a
        # count-bearing deliverable; the numbers are only as trustworthy as what stands behind them.
        "operating_point": result["operating_point"],
        "validated": bool(result["validated"]),
        # The CSV's own written cell (floored across every gated dimension without a column of
        # its own); unvalidated_dimensions names every dimension that did not validate.
        "operating_point_validated": tail["operating_point_validated"],
        "tile_size_validated": (
            (summary["stamp"].get("tile_size") if summary["tile_size_operative"] else None)
            if bucket is not None else tile_ref),
        "unvalidated_dimensions": tail["unvalidated_dimensions"],
        "conf_source": result["conf_source"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "experiment_id": result["experiment_id"],
    }
    if bucket is None:
        # The run's own narrowed reference, distinct from operating_point_validated above,
        # which floors false here since nothing on disk backs it without a bucket.
        out["run_conf_validated_against"] = op_ref
    # run_inference's own warnings (a CPU-bound workload) are surfaced here too, so a count CSV
    # never ships with the regime it ran in disclosed only in the server log.
    if result.get("warning"):
        out["warning"] = result["warning"]
    return out


def _bucket_csv_rows(bucket_path: Path, stamp: dict) -> list[dict]:
    """A prediction bucket's own per-image documents as ``export_detection_csv``'s row source.

    Counted detections only (``detection_annotations``: a ``Point`` and a crowd region excluded),
    ordered by document stem.

    Each row's ``image`` value is the source filename the stamp records for the document's stem
    (``image_filenames``). A stamp without the map, or a document it does not name, is refused as
    ``CountDeliveryRefused`` naming the bucket and the documents.
    """
    from tcip_annotation.json_io import detection_annotations, prediction_documents, safe_score
    from tcip_annotation.state import prediction_score

    from tcip_mcp.pipelines.resolution import CountDeliveryRefused

    documents = sorted(prediction_documents(bucket_path), key=lambda p: p.stem)
    stated = stamp.get("image_filenames")
    names: dict = stated if isinstance(stated, dict) else {}
    unnamed = [doc.stem for doc in documents if doc.stem not in names]
    if unnamed:
        raise CountDeliveryRefused(
            f"{bucket_path}'s stamp names no source image for document(s) {unnamed}: every "
            "platform publisher records each document it writes in the stamp's image_filenames, "
            "so these documents are not a platform-published bucket's.")
    image_results = []
    for doc in documents:
        annotations = detection_annotations(doc)
        scores = [safe_score(prediction_score(a)) for a in annotations]
        image_results.append(
            {"image": names[doc.stem], "count": len(annotations), "scores": scores})
    return image_results


def per_image_counts_from_bucket(
    predictions_dir: str, output_path: str, *, trait: str,
    project_root: str | Path | None = None,
    acknowledgment: Acknowledgment | None = None,
) -> dict:
    """Count the detections in each document of the existing prediction bucket
    ``predictions_dir`` and write them as the per-image count CSV at ``output_path`` through
    ``export_detection_csv``, leaving the bucket untouched; returns the export's tail and counts.

    The stamp's ``images_dir``, ``raster_path`` and ``trait`` are checked before any document is
    counted; its ``operating_point``, ``checkpoint_sha256`` and ``experiment_id`` are read after
    counting, so a stamp lacking one of those raises ``KeyError`` naming it once the documents
    have been read and before anything is written.

    Runs the ``per_image_count`` meaning check first, against ``trait`` and ``project_root``,
    before the bucket is touched. ``OperationalizationRefused`` (``tcip_mcp.operationalization``)
    carries the failed check and no counts, raised from this call's own pre-check, from
    ``export_detection_csv``'s own pre-check (which also reads each recorded bucket's ``id_map``
    for the confirmed subject), or from that writer's post-gate re-check; ``DeliveryRefused``
    (``pipelines.resolution``) is the writer's own gate refusal, its ``facts`` attribute set to
    this call's counts-bearing facts; ``CountDeliveryRefused`` (``pipelines.resolution``) covers
    everything else this door refuses on (a missing stamp, a whole-raster bucket, an unknown or
    mismatched trait, a document the stamp's filename map does not name, an empty bucket), each
    carrying the same facts.
    """
    from tcip_mcp.operationalization import (
        PER_IMAGE_COUNT,
        OperationalizationRefused,
        check_operationalization,
        resolve_trait_and_record,
    )
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE, CountDeliveryRefused, read_operating_point_sidecar, stamp_names_raster,
    )
    from tcip_mcp.traits import TraitUnknownError
    from tcip_store import StoreError

    try:
        spec, record, _specs_dir = resolve_trait_and_record(
            trait, PER_IMAGE_COUNT, project_root=project_root)
    except TraitUnknownError as exc:
        raise CountDeliveryRefused(str(exc)) from exc
    # A per_image_count delivery names no positive class, so check_operationalization ignores a
    # registry for this kind regardless of what one would resolve to.
    stated = check_operationalization(spec, record, PER_IMAGE_COUNT, registry=None)
    if not stated.ok:
        raise OperationalizationRefused(stated)

    bucket_path = resolve_output_path(predictions_dir)
    sidecar = read_operating_point_sidecar(bucket_path)
    if sidecar is None:
        raise CountDeliveryRefused(
            f"{bucket_path} carries no readable operating_point.json: a bucket regime call reads "
            "a stamp a platform producer wrote (run_inference, deliver_per_image_counts's own "
            "live-with-predictions_dir path, or the web inference worker), never a directory of "
            "label JSON with no stamp.")
    if stamp_names_raster(sidecar):
        raise CountDeliveryRefused(
            f"{bucket_path} is a whole-raster bucket (its stamp records raster_path): one mosaic "
            "total is not a per-image count, and the per_image_count operationalization was never "
            "confirmed for it. Deliver a per-plant count from it through "
            "deliver_orthomosaic_plant_counts instead.")
    if not sidecar["images_dir"]:
        raise CountDeliveryRefused(
            f"{bucket_path}'s stamp records neither images_dir nor raster_path: it is not a "
            "per-image prediction bucket this door can read.")
    stamp_trait = sidecar["trait"]
    if stamp_trait is not None and stamp_trait != trait:
        raise CountDeliveryRefused(
            f"{bucket_path}'s stamp was recorded for trait {stamp_trait!r}, not {trait!r}: a "
            "bucket produced for one trait cannot deliver a per-image count under another.")

    image_results = _bucket_csv_rows(bucket_path, sidecar)
    if not image_results:
        raise CountDeliveryRefused(
            f"{bucket_path} carries a readable stamp but no prediction documents: an empty "
            "bucket is not a per-image count either.")
    image_count = len(image_results)
    total_detections = sum(r["count"] for r in image_results)

    op = sidecar["operating_point"] or {}
    provenance = {
        "producer_model_sha256": sidecar["checkpoint_sha256"],
        "producing_experiment_id": sidecar["experiment_id"],
        "operating_point_conf": (op.get("conf") or {}).get("value"),
    }
    try:
        csv_path, tail, summary, event_recorded = export_detection_csv(
            image_results, output_path, provenance=provenance, trait=trait,
            operating_point_validated=None, pred_dirs=[str(bucket_path)],
            acknowledgment=acknowledgment, project_root=project_root,
        )
    except StoreError as exc:
        raise CountDeliveryRefused(str(exc)) from exc
    except DeliveryRefused as exc:
        exc.facts = {
            "operating_point_validated": exc.gate.stamp.get("operating_point", VALIDATED_FALSE),
            "tile_size_validated": exc.gate.stamp.get("tile_size"),
            "unvalidated_dimensions": exc.gate.unvalidated_cell(),
            "operating_point": sidecar["operating_point"],
            "image_count": image_count,
            "total_detections": total_detections,
            "predictions_dir": str(bucket_path),
        }
        raise

    out = {
        "csv_path": csv_path,
        "image_count": image_count,
        "total_detections": total_detections,
        "operating_point": sidecar["operating_point"],
        "operating_point_validated": tail["operating_point_validated"],
        "tile_size_validated": (
            summary["stamp"].get("tile_size") if summary["tile_size_operative"] else None),
        "unvalidated_dimensions": tail["unvalidated_dimensions"],
        "acknowledged_by": tail["acknowledged_by"],
        "checkpoint_sha256": sidecar["checkpoint_sha256"],
        "experiment_id": sidecar["experiment_id"],
        "predictions_dir": str(bucket_path),
        "delivery_event_recorded": event_recorded,
    }
    return out
