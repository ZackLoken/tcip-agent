"""Inference MCP tools: run_inference and deliver_per_image_counts, sharing one verified body
(``_run_inference_verified``) so the firewalled operating point (conf/NMS/tiling/max_dets)
resolves identically for every entry point that runs a model over images."""

from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from tcip_store import RECORD_JSON, BadKey, Key, StoreDescriptor, Version, register_store, store
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited
from tcip_mcp.pipelines.postprocessing.export import (
    export_detection_csv,
    mask_binarize_provenance,
    positive_detections,
    unmapped_label_ids,
    write_predictions_json,
)
from tcip_mcp.pipelines.resolution import (
    DEFAULT_CONF,
    DEFAULT_MAX_DETS,
    DEFAULT_NMS_IOU,
    DeliveryRefused,
)
from tcip_mcp.project_paths import resolve_output_path

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.resolution import Acknowledgement

logger = logging.getLogger(__name__)

_CALIBRATION_CURVE_DIR = (".tcip", "artifacts")
_CALIBRATION_CURVE_STEM = "operating_point_sweep_"  # frozen on-disk prefix, the locator's own path contract


class _CalibrationCurveLocator:
    """One conf curve record per name, named for the digest of the record's own bytes
    (:func:`calibration_curve_identity`).

    The digest is in the filename rather than in a directory of its own, the same convention
    the locked calibration/holdout split beside it uses. Under ``last_writer_wins``, only a
    byte-identical rerun replaces what is already there; a calibration that differs in any byte
    the count gate would run over writes under a different name.
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


def calibration_curve_path(record_digest: str) -> Path:
    """Where that curve lands on disk, for the provenance that names the file it was kept in."""
    from tcip_mcp.project_paths import platform_state_root

    key = calibration_curve_key(record_digest)
    return platform_state_root().joinpath(
        *_CalibrationCurveLocator().relative_path(key.root, key.parts).parts
    )


def calibration_curve_identity(body: dict) -> str:
    """The sha256 identity of a confidence_sweep record's whole body, over the exact bytes the
    store writes for it (``RECORD_JSON.encode``, the codec's own check of what the body carries).

    The writer takes the key a record is written under from this function, and the reader
    recomputes it over what was read back and compares: two calibrations differing in any byte
    the count gate would run over now differ in identity, and a record edited after the run no
    longer matches the digest the run's own response carried. The full sixty-four hex characters,
    never truncated: a truncated key under ``last_writer_wins`` would let two legitimate bodies
    sharing a prefix overwrite each other.
    """
    from tcip_mcp.model_registry import _sha256_of_bytes

    return _sha256_of_bytes(RECORD_JSON.encode(body))


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
of the bucket root, so an interrupted pass' own progress never reads as a prediction document. Not
frozen: its shape may still move, so a reader checks the identity record's own ``schema_version`` by
hand (``_RASTER_PASS_PROGRESS_SCHEMA_VERSION``) rather than relying on the seam, which only enforces
that ceiling for a store declared frozen."""

_RASTER_PASS_PROGRESS_SCHEMA_VERSION = 1


def _recorded_training_id_map(predictor) -> dict | None:
    """The training run's own recorded name->id map (``config["data"]["id_map"]``), or ``None``
    when the checkpoint carries none. The one read :func:`resolve_decode_id_map` and calibration's
    own GT-side id-map resolution (:func:`~tcip_mcp.pipelines.calibration.calibrate_operating_point`)
    both share, so a checkpoint's
    recorded vocabulary is preferred identically wherever an id_map is derived for it, never
    re-checked independently by each caller."""
    data_cfg = (getattr(predictor, "config", {}) or {}).get("data") or {}
    recorded = data_cfg.get("id_map")
    if isinstance(recorded, dict) and recorded:
        return {str(k): int(v) for k, v in recorded.items()}
    return None


def run_scope(predictor) -> tuple[str | None, str | None]:
    """A run's own recorded ``(subject, attribute)``, the same pair the training run stamps onto
    its experiment config and every publishing door stamps onto the bucket's own
    ``operating_point.json``.

    Refuses by name a run that declares an attribute with no subject: a value with no object class
    names nothing a reader could hold predictions to. Every door that publishes a bucket or reads a
    checkpoint's scope calls this one function rather than re-reading ``config["data"]`` on its own.
    """
    data_cfg = (getattr(predictor, "config", {}) or {}).get("data") or {}
    subject, attribute = data_cfg.get("subject"), data_cfg.get("attribute")
    if attribute is not None and subject is None:
        raise ValueError(
            f"{getattr(predictor, 'path', predictor)!r} declares attribute {attribute!r} with no "
            "subject: a value with no object class names nothing a reader could hold predictions "
            "to. Retrain with data.subject stated beside data.attribute."
        )
    return subject, attribute


def unmapped_classified_run(
    data_cfg: dict, id_map: dict | None, *, images_dir: str | None,
) -> str | None:
    """The composed refusal for a run that declared an attribute and resolved no ``id_map`` to
    decode predictions with, or ``None`` when there is nothing to refuse (a mapped run, or a run
    with no attribute at all).

    The remedy names the run's own shape, decided by :func:`~tcip_mcp.pipelines.data.label_queries.
    targets_registry_derived`: a registry-derived run with an ``images_dir`` whose dataset holds
    no ``classes.json`` is told to run ``write_class_map`` for that dataset; a registry-derived run
    called with no ``images_dir`` at all (an ``image_paths``-only call) is told to pass one, since
    no dataset can otherwise be named to decode it against; any other run (a bespoke
    ``dataset_source``, a COCO-sourced run, or one with no registry to derive from) is told to
    state ``data.id_map`` in its launch config beside ``data.subject``/``data.attribute`` and
    retrain, the one route onto a bespoke run's checkpoint. The raster regime, which has no
    ``images_dir`` concept at all, always calls this with ``images_dir=None``; a raster run's own
    targets are never registry-derived, so its message always resolves to the retrain remedy.
    """
    if id_map is not None:
        return None
    attribute = data_cfg.get("attribute")
    if attribute is None:
        return None
    subject = data_cfg.get("subject")
    from tcip_mcp.pipelines.data.label_queries import targets_registry_derived

    if targets_registry_derived(data_cfg):
        if images_dir is None:
            return (
                f"this run decoded along attribute {attribute!r} of subject {subject!r} from a "
                "registry-derived dataset, but no images_dir was given to read the decoding "
                "dataset's classes.json from. Pass images_dir naming the dataset whose "
                "classes.json decodes this run."
            )
        return (
            f"this run decoded along attribute {attribute!r} of subject {subject!r} from a "
            f"registry-derived dataset, but {images_dir!r} holds no classes.json to decode it "
            "with. Run write_class_map for that dataset, then retry."
        )
    return (
        f"this run decoded along attribute {attribute!r} of subject {subject!r}, but its launch "
        "config recorded no id_map to decode predictions with. State data.id_map in the launch "
        "config beside data.subject and data.attribute and retrain: the launch config is the one "
        "producer of a map on a bespoke run's checkpoint, and re-registering the model does not "
        "add one, since the predictor never reads the registry entry's own config."
    )


def resolve_decode_id_map(predictor, images_dir: str | None, *,
                          scope: tuple[str | None, str | None] | None = None) -> dict | None:
    """This run's name->id map for recording + decoding predictions.

    The one resolution every entry point that writes predictions to disk calls, this tool's own
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
    on any failure; the two entry points share this one resolution but choose different failure postures on
    top of it, not two different resolutions.

    ``scope`` is the ``(subject, attribute)`` the registry fallback derives against, defaulting to
    the predictor's own recorded training scope. A caller holding the run's scope from elsewhere
    (block calibration reads it from the training experiment's ``config.json``, and refuses without
    it) passes it here rather than restating the prefer-recorded-else-derive rule around its own.
    """
    recorded = _recorded_training_id_map(predictor)
    if recorded is not None:
        return recorded
    subject, attribute = scope if scope is not None else run_scope(predictor)
    if not (subject and images_dir):
        return None

    from tcip_mcp.pipelines.data.label_queries import resolve_registry_id_map, resolved_classes_path

    # Precondition, not a broad except: an attribute-scoped run with no classes.json here returns
    # None so unmapped_classified_run composes the write_class_map remedy, rather than crashing.
    if attribute is not None and resolved_classes_path(images_dir) is None:
        return None
    _reg, id_map = resolve_registry_id_map(images_dir, subject, attribute)
    return id_map


@mcp.tool()
@audited(scope_arg="output_dir", scope_via=resolve_output_path)
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
    tile_batch_size: int = 96,
    global_nms_iou: float | None = None,
    max_dets: int | None = None,
    postprocess: str = "nms",
    dry_run: bool = False,
    trait: str | None = None,
    calibration_labels_dir: str | None = None,
    calibration_images_dir: str | None = None,
    split_manifest_dir: str | None = None,
    experiment_id: str | None = None,
    group_by: str | None = None,
    group_key_map: dict[str, str] | None = None,
    split_seed: int = 0,
    split_holdout_ratio: float = 0.5,
    overwrite: bool = False,
    allow_unvalidated_staging: bool = False,
    require_masks: bool = True,
    resume: bool = False,
) -> dict:
    """Run a trained model over images or a raster, and persist the predictions as a bucket.

    Provide exactly one of ``images_dir`` (an ordinary directory of per-image captures) or
    ``raster_path`` (a single raster, georeferenced or not, potentially too large to decode
    whole). Every call publishes a prediction bucket at ``output_dir``: this is the one door that
    runs a checkpoint over images and the one door that persists what it produced, so the
    operating point a caller sees is exactly the one behind the files on disk. The pass itself
    stays a private library call, :func:`_run_inference_verified`, an ordinary function with no
    audit line of its own; this tool's own audit line is the record of a call.

    A prediction bucket is the directory these writes persist into, not a score bin or a quota
    allocation: its identity is that directory's own path (relative to the dataset root when it
    sits under one, its own resolved path otherwise; see
    :func:`~tcip_mcp.prediction_buckets.bucket_key_of`), and a bucket under a dataset root turns
    immutable the moment a human review verdict lands on any image inside it (a bucket written
    outside any dataset root has no verdict store to guard it; see below). Two source regimes,
    sharing one bucket-resolution/immutability/gate/lineage contract so a breeder or agent has
    one door regardless of capture shape:

    - ``images_dir``: writes ``<stem>.json`` per image. Works for ``instance_seg`` too; each
      tiled result's ``masks`` (see ``GenericPredictor.predict_tiled``) are a tile-local patch
      plus its full-image-space offset, never the untiled path's dense full-image array.
    - ``raster_path``: sources tiles from the windowed raster layer instead
      (:func:`~tcip_mcp.pipelines.raster_source.open_raster`), always tiled (there is no untiled
      option for a raster too large to decode whole), and writes exactly one ``<raster
      stem>.json`` prediction file (in full-raster pixel space), since there is no natural
      directory-of-per-plant-images shape for a whole-raster capture. ``calibration_labels_dir``/
      ``split_manifest_dir`` are not accepted with it (there is no separate labeled directory
      shape for one raster); ``trait`` alone calibrates against the mosaic's own reserved regions
      instead (:func:`~tcip_mcp.pipelines.block_calibration.resolve_block_calibration_records`),
      when the checkpoint's own training experiment reserved one
      (``data.split.reserve_calibration_fraction`` at training time). Without a reserved region,
      conf has no per-dataset calibration for this regime; a validated per-plant count is earned
      later, at delivery (``deliver_orthomosaic_plant_counts``). ``resume`` (below) applies only
      here.

    Both regimes write the same ``operating_point.json`` stamp convention beside the prediction
    file(s), so downstream code that reads a bucket's sidecar generically needs no special case
    for which regime produced it.

    Neither regime's underlying pass ever refuses on an unvalidated dimension on its own; each is
    the shared, honestly-stamped raw substrate this door builds on, the same contract an
    uncalibrated ``conf`` already has. This tool is the one that actually persists a prediction
    bucket other doors treat as ground truth, so it is where the refusal belongs: a tiled run
    whose tile_size has no real basis (no persisted training geometry, no recoverable native-frame
    edge, no explicit override) refuses to write here unless ``allow_unvalidated_staging=True``, the
    staging escape that clears only ``tile_size``/``claim_scope``
    (:data:`tcip_mcp.pipelines.resolution.STAGING_DIMENSIONS`), never a phenotype's own delivered
    dimensions: it lets a raw, honestly-stamped bucket be persisted at an unproven scale, a
    different act from delivering a phenotype from one. Both regimes gate before the
    (expensive) pass runs: the ``raster_path`` regime uses the predictor that pass then reuses;
    the ``images_dir`` regime sniffs the checkpoint's own stamped config (no weights load, never
    raises on a missing/unreadable checkpoint) to resolve the same geometry the verified pass
    will, then re-checks against that pass' own real result after it runs as the authoritative
    gate. An untiled run's tile_size is never operative and can't manufacture a refusal.

    A prediction bucket (``output_dir``) that already carries review verdicts is immutable: by
    default the write is redirected to a fresh run-scoped bucket (``<dir>@r2``, ``@r3``, next
    free) and the dir actually written is returned as ``output_dir``. Pass ``overwrite=True`` to
    write in place only when the bucket has zero verdicts; with verdicts present it is refused
    (error names the count and a suggested dir) so a re-run never orphans recorded verdicts. The
    verdicts consulted are the ones recorded in the bucket's own dataset, so a bucket written
    outside any dataset has no store to be guarded against: that write lands where it was asked
    for, ``verdict_guard_operative`` comes back false with a note saying so, and it is stamped
    unvalidated whatever its operating point cleared, since a count claim outside the dataset
    layout has no dataset-relative key a reader could locate it by.

    A second, narrower immutability applies regardless of a dataset root or a verdict: a bucket
    that already holds a prediction document from an earlier publish, with no verdict yet
    recorded against it, refuses this write outright (error names the document count and a
    suggested fresh bucket), whatever ``overwrite`` says. No publish begins into a bucket that
    held prediction documents when this door resolved it; it does not guard against two runs
    racing into one bucket after both resolve it clean, the same window the verdict guard already
    has. A completed experiment's bucket the pointer below already locks is unreachable through
    the suggested bucket too, so a completed experiment's predictions publish once through this
    door with no audited remedy to clear one for republication.

    A bucket stamped validated names the validation record its claim was earned from. The gate
    for that record runs before any file is written, and the record is appended over the
    prediction files as they actually landed, so a run that dies partway leaves either
    predictions with no stamp or a record no stamp names, both of which floor to unvalidated at
    every delivery door.

    Args:
        checkpoint_path: Path to model .pt checkpoint. Must be registered under this process's
            platform state root (``register_model``, explicit mode for a foreign or bespoke
            checkpoint) or this door refuses before loading it.
        images_dir: Directory containing input images (mutually exclusive with ``raster_path``).
        raster_path: A single raster, georeferenced or not, potentially too large to decode whole
            (mutually exclusive with ``images_dir``); see the regime description above.
        output_dir: Directory for output .json prediction file(s). Required, including for
            ``dry_run``, which names the bucket the write would resolve to without writing it. A
            relative path resolves against the platform state root, never the server process's
            cwd.
        conf_threshold: Minimum confidence score. ``None`` (default) states nothing and runs at
            the platform default, stamped ``"default"``; a stated value is honored as an explicit
            override and stamped as one, including when it happens to equal the platform default.
        device: Device to use ('cuda' or 'cpu').
        tile: Enable tiled (SAHI-style) detection inference (``images_dir`` regime only;
            ``raster_path`` is always tiled). ``None`` (default) derives it from the checkpoint's
            own training tile geometry (``predictor.train_tile_size is not None``), not a fixed
            default, its provenance is stamped ``"default"`` vs ``"explicit"`` so a caller who
            deliberately chose one way is distinguishable from one who left it unset.
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
        dry_run: Report the effective operating point (conf/tiling/max_dets/postprocess) and the
            bucket the write would resolve to, without loading the model or running inference.
            Previews the same bucket refusal a real call would hit (verdicts, or a document with
            none), never a bucket a real call would in fact refuse to write.
        trait: Trait name to derive the confidence operating point per dataset instead of pinning
            a default, the count is the phenotype, so conf must be calibrated. ``images_dir``
            regime: with ``calibration_labels_dir``. ``raster_path`` regime: alone, against the
            checkpoint's own training mosaic's reserved calibration/test regions (requires
            ``data.split.reserve_calibration_fraction`` at training time; refuses by name
            otherwise). Absent -> the byte-identical raw path (conf=score_threshold, unvalidated).
        calibration_labels_dir: Labeled dir for a disjoint cal/holdout split to calibrate +
            held-out validate the operating point (``images_dir`` regime only; not accepted with
            ``raster_path``, see ``trait``). Its GT identity scopes the resolved conf (dataset
            firewall).
        calibration_images_dir: Images for the calibration labels (defaults to ``images_dir``).
        split_manifest_dir: Restrict the calibration universe to one capture date's
            ``calibration`` side of a split manifest (``data_tools.read_split_manifest_dir``,
            ``images_dir`` regime only; not accepted with ``raster_path``, block calibration draws
            no split-manifest universe) instead of every labelled stem with an image, so the
            operating point is measured on exactly the side the manifest held out for it, never
            the side the checkpoint was chosen on. A checkpoint bound to a different manifest than
            the one named here is refused by name. The manifest's subject/attribute must equal the
            checkpoint's own recorded training scope, the calibration labels' date must be one the
            manifest holds members under, and the manifest's ``images_root`` for that date must be
            ``calibration_images_dir`` (or ``images_dir``), each refusing by name. The response
            carries ``n_excluded_training_stems``, ``n_excluded_validation_stems`` and
            ``n_excluded_unassigned_stems``, the present stems the manifest's universe left out
            (its train side, its val side, and stems the draw never assigned), beside
            ``n_excluded_incomplete_attribute``.
        experiment_id: The run that produced the checkpoint, for provenance. Best-effort resolved
            (checkpoint's own stamp, then the registry) when omitted; a raw/foreign checkpoint
            legitimately has none. Also gates calibration's train-disjointness check: a
            *known* run whose training split can't be read/reconstructed fails that check closed.
        group_by: ``images_dir`` regime only. Grouping policy for the locked calibration/holdout
            split, ``"tile_prefix"`` or ``"stem"``. Ignored when ``group_key_map`` is given.
            ``None`` (default) resolves to ``"tile_prefix"`` when neither this nor
            ``split_manifest_dir`` was given; a value beside ``split_manifest_dir`` conflicts with
            the manifest's own grouping policy and refuses, naming both. Only the first
            calibration call for a given calibration-labels identity draws the split; later calls
            return the same locked split regardless of this argument (see
            ``redraw_calibration_holdout`` to redraw deliberately).
        group_key_map: ``images_dir`` regime only. An agent-derived ``{stem: group_key}`` map
            overriding ``group_by`` for the locked calibration/holdout split, must cover every
            stem in ``calibration_labels_dir``. Conflicts with ``split_manifest_dir`` the same way
            ``group_by`` does.
        split_seed: ``images_dir`` regime only. Split seed for the locked calibration/holdout
            split, like ``group_by``, only takes effect on the first calibration call for a given
            calibration-labels identity; a later call's declared value is compared to the lock and
            any divergence is reported in ``gate_evidence_summary``/the resolved bundle rather than
            silently ignored.
        split_holdout_ratio: ``images_dir`` regime only. Calibration/holdout fraction for the
            locked split, same first-call-only semantics as ``split_seed``.
        overwrite: Write into ``output_dir`` even if it exists. Refused if the bucket has review
            verdicts; the default (False) auto-redirects to a fresh bucket instead. Never
            overrides the document refusal above: a bucket holding a document with no verdict
            refuses this write whatever ``overwrite`` says. For a ``raster_path`` pass, also the
            way out of a bucket carrying another pass' progress (``resume``, below) when that
            bucket holds no document yet: it discards that progress and starts over. Conflicts
            with ``resume=True`` (refused by name): the two name opposite ways of handling the
            same recorded progress. A bucket that already holds both a document and a progress
            record (a crash between the two) refuses on the document before either ``resume`` or
            ``overwrite`` is consulted, leaving the progress record in place, inert.
        allow_unvalidated_staging: Write the bucket even when tile_size (a tiled run only) has no
            real basis, stamping ``tile_size_validated=false`` on the sidecar so the
            un-trustworthiness travels with it rather than writing silently. Clears only the
            staging dimensions (tile_size, claim_scope); never a delivered phenotype's own
            operating-point or classifier dimension, which no argument here can clear.
        require_masks: Collect masks for an ``instance_seg`` checkpoint (``raster_path`` regime
            only; ignored for ``images_dir``, which always carries masks).
        resume: ``raster_path`` regime only (refuses with ``images_dir``, and with
            ``overwrite=True``). Continue a raster pass this bucket carries progress from, rather
            than refusing over that progress or starting a new one. The recorded pass' checkpoint,
            raster content, trait, experiment and tile batch size must all match this call's own,
            or it refuses naming what differs; the resumed pass then runs the remaining tiles at
            the recorded operating point rather than re-deriving one (a block-calibrated pass in
            particular applies the recorded conf/cross_tile_nms directly, so it never re-runs the
            calibration bands a second time and can never refuse over a re-derivation's own float
            noise). Refuses outright when the bucket carries no progress at all, when the recorded
            progress is a schema version newer than this reader knows, and for a mask-bearing
            (``instance_seg`` with ``require_masks``) pass, which never records progress to resume
            from. See :func:`_export_predictions_raster`'s own docstring for the progress records
            this pass keeps.
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
    if raster_path is not None and split_manifest_dir:
        return {"error": "split_manifest_dir is not supported for a raster_path export: block "
                         "calibration draws no split-manifest universe, so it would be silently "
                         "dropped rather than scoping anything."}
    if split_manifest_dir and not calibration_labels_dir:
        return {"error": "split_manifest_dir requires calibration_labels_dir: it scopes a "
                         "calibration this call has no trait/calibration_labels_dir to run, so "
                         "the manifest would be silently dropped rather than bounding one."}
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
        applied_conf, applied_nms_iou, applied_max_dets = _applied_operating_point(
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
            checkpoint=checkpoint, raster_path=raster_path, out=out,
            resolution=resolution, device=device, conf_threshold=conf_threshold,
            tile_size=tile_size, overlap=overlap, tile_batch_size=tile_batch_size,
            global_nms_iou=global_nms_iou, max_dets=max_dets, postprocess=postprocess,
            require_masks=require_masks,
            experiment_id=block_calibration_experiment_id or experiment_id,
            allow_unvalidated_staging=allow_unvalidated_staging, trait=trait,
            resume=resume, overwrite=overwrite,
        )

    from tcip_mcp.pipelines.resolution import (
        check_delivery_gate, resolve_tile_size_param, tile_size_gate_flag,
    )

    # Gate before the (expensive) pass: the checkpoint's own stamped config, already loaded,
    # resolves the same tile geometry the verified pass itself will.
    from types import SimpleNamespace

    from tcip_mcp.pipelines.inference.predictor import (
        TileEdgeContradiction, explicit_edge_provenance, resolve_tile_regime,
    )

    data_cfg = checkpoint.data_config
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
        allow_unvalidated_staging=allow_unvalidated_staging)
    if not pre_gate.ok:
        return {"error": pre_gate.reason, "tile_size_validated": pre_tile_ref}

    result = _run_inference_verified(
        checkpoint, images_dir=images_dir, conf_threshold=conf_threshold,
        device=device, tile=tile, tile_size=tile_size, overlap=overlap,
        tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou, max_dets=max_dets,
        postprocess=postprocess, trait=trait,
        calibration_labels_dir=calibration_labels_dir, calibration_images_dir=calibration_images_dir,
        split_manifest_dir=split_manifest_dir, experiment_id=experiment_id,
        group_by=group_by, group_key_map=group_key_map, split_seed=split_seed,
        split_holdout_ratio=split_holdout_ratio,
    )
    if "error" in result:
        return result

    pub = _publish_bucket_bracket(
        result, out=out, checkpoint_path=checkpoint_path, trait=trait, images_dir=images_dir,
        dataset_root=bucket_root, allow_unvalidated_staging=allow_unvalidated_staging)
    if pub["refusal"] is not None:
        return pub["refusal"]
    written, dropped_boxes = pub["written"], pub["dropped_boxes"]
    op_stamp, tile_size_validated = pub["op_stamp"], pub["tile_size_validated"]

    # Every field the pass returned, minus ``results``, overlaid with the bucket fields the write
    # itself earned (the pass' own raw ``validated`` does not yet know about the tile-scale floor).
    response = dict(result)
    response.pop("results", None)
    response.update({
        "image_count": len(written), "output_dir": str(out), "files": written,
        "bucket_redirected": resolution.redirected,
        "requested_output_dir": output_dir if resolution.redirected else None,
        "validated": op_stamp["validated"],
        "tile_size_validated": tile_size_validated,
        "verdict_guard_operative": bucket_root is not None,
        "dropped_nonpositive_boxes": dropped_boxes,
    })
    if bucket_root is None:
        response["note"] = _NO_DATASET_ROOT_NOTE.format(bucket=out)
    return response


def _applied_operating_point(
    conf_threshold: float | None, global_nms_iou: float | None, max_dets: int | None,
) -> tuple[float, float, int]:
    """The stated-vs-platform-default resolution for conf/NMS/max_dets, shared by
    ``run_inference``'s ``dry_run`` preview and its verified body so the two can't diverge."""
    applied_nms_iou = DEFAULT_NMS_IOU if global_nms_iou is None else float(global_nms_iou)
    applied_max_dets = DEFAULT_MAX_DETS if max_dets is None else int(max_dets)
    applied_conf = DEFAULT_CONF if conf_threshold is None else float(conf_threshold)
    return applied_conf, applied_nms_iou, applied_max_dets


def _run_inference_verified(
    checkpoint,
    *,
    image_paths: list[str] | None = None,
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
    split_seed: int = 0,
    split_holdout_ratio: float = 0.5,
    split_manifest_dir: str | None = None,
) -> dict:
    """The verified body of ``run_inference``: everything after its checkpoint is loaded once.

    Carries no ``@audited`` of its own: the calling tool's audit line is the record of this call.
    ``deliver_per_image_counts``'s live regime calls this directly with the checkpoint it already
    loaded, never a second load through ``run_inference`` itself, so a door composing this pass
    never loads the file twice.
    """
    max_dets_stated = max_dets is not None
    conf_stated = conf_threshold is not None
    applied_conf, applied_nms_iou, applied_max_dets = _applied_operating_point(
        conf_threshold, global_nms_iou, max_dets)

    from tcip_mcp.pipelines.inference.predictor import build_predictor
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point
    from tcip_mcp.pipelines.resolution import dataset_hash, raw_operating_point

    # NMS IoU + the full-frame detection cap govern which boxes exist (in-model thresholds), not
    # just cross-tile merge, else nms_iou would have no effect on an untiled run.
    predictor = build_predictor(
        checkpoint,
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
    # experiment_id, off the object already loaded (no re-hash, no re-read).
    from tcip_mcp.model_registry import resolve_model_identity

    identity = resolve_model_identity(checkpoint, experiment_id=experiment_id)

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
            f"tile_size could not be resolved for {checkpoint.path}: this checkpoint carries no "
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

    # Holds the union predict_batch itself accepts (a directory scan can resolve a BandGroupRef),
    # never the narrower list[str] the image_paths parameter carries in from the caller.
    resolved_paths: list[str | Path | BandGroupRef]
    if image_paths is None:
        if images_dir is None:
            return {"error": "Provide either image_paths or images_dir"}
        # Fold a `.bandgroup`-grouped capture into its one logical entry (list_logical_images),
        # the same enumeration every other reader in this platform shares.
        from tcip_mcp.pipelines.image_utils import list_logical_images

        logical = list_logical_images(images_dir)
        resolved_paths = [logical[stem] for stem in sorted(logical)]
    else:
        resolved_paths = list(image_paths)

    # A classified run with no id_map cannot decode its own predictions: refuse before either
    # pass runs, so no calibration evidence is spent on a run that would refuse at the write.
    _subject, _attribute = run_scope(predictor)
    _door_id_map = resolve_decode_id_map(predictor, images_dir)
    _refusal = unmapped_classified_run(
        {"subject": _subject, "attribute": _attribute}, _door_id_map, images_dir=images_dir)
    if _refusal is not None:
        return {"error": _refusal}

    # Resolve the confidence operating point: with a trait + labeled calibration dir, derive it
    # per dataset (count-unbiased + held-out validated); otherwise the byte-identical raw path.
    extra: dict = {}
    if trait and calibration_labels_dir:
        from tcip_annotation.json_io import UnreadableLabelDocument

        from tcip_mcp.pipelines.calibration import calibrate_operating_point, gate_evidence_summary

        cal_images = calibration_images_dir or images_dir
        try:
            bundle, cal_hash, n_excluded_incomplete_attribute, evidence = calibrate_operating_point(
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

        inf_stems = [stem_of(pp) for pp in resolved_paths]
        cal_label_stems = (
            set(evidence.get("calibration_stems", [])) if split_manifest_dir is not None
            else {pp.stem for pp in prediction_documents(calibration_labels_dir)}
        )
        same_images = calibration_images_dir is None or (
            images_dir is not None and Path(calibration_images_dir) == Path(images_dir))
        # A manifest's own calibration universe is a held-out subset of the labelled directory,
        # so inferring the whole directory is still the same labelled set the calibration drew.
        if split_manifest_dir is not None:
            comparable = bool(
                same_images and inf_stems and cal_label_stems
                and cal_label_stems <= set(inf_stems)
            )
        else:
            comparable = bool(same_images and inf_stems and set(inf_stems) == cal_label_stems)
        if comparable:
            # The bundle's own hash covers the calibration universe under a manifest, not the
            # (larger) inference stem list, so the target must be hashed over that same universe.
            hashed_stems = evidence.get("calibration_stems", []) if split_manifest_dir is not None \
                else inf_stems
            target_hash, cross_dataset_check = (
                dataset_hash(calibration_labels_dir, stems=hashed_stems), "same-labeled-set")
        else:
            target_hash, cross_dataset_check = None, "not-comparable-unlabeled-target"
        issues = bundle.shippable_issues(target_dataset_hash=target_hash)
        # Channel firewall: probe one target raster and check its band count against the
        # checkpoint's in_chans via validate_resolved_bundle, so a channel-wrong inference surfaces in
        # the provenance rather than being silently coerced by the loader.
        if resolved_paths:
            from tcip_mcp.pipelines.derivations import probe_channels
            from tcip_mcp.pipelines.resolution import (
                ResolvedBundle, default as _resolved_default, validate_resolved_bundle,
            )
            try:
                probed = int(probe_channels(resolved_paths[0]))
            except Exception:
                probed = None
            if probed is not None:
                chan_bundle = ResolvedBundle(trait=trait or "", dataset_hash=None, params={
                    "in_chans": _resolved_default(
                        "in_chans", int(getattr(predictor, "in_chans", 3)))})
                issues = issues + validate_resolved_bundle(chan_bundle, probed_channels=probed)
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
            extra["n_excluded_unassigned_stems"] = len(manifest_excluded["excluded_unassigned_stems"])
        # The full curve can be large, persist it and return the path (provenance emits has_gate_evidence).
        # The record's own body is its identity, so a curve differing from a prior one is never lost.
        curve_body = {
            "trait": trait,
            "dataset_hash": cal_hash,
            "checkpoint_sha256": identity["sha256"],
            "predictor_path": {
                "tile": resolved_tile_bool, "tile_size": resolved_tile,
                "overlap": resolved_overlap, "postprocess": postprocess,
                "global_nms_iou": applied_nms_iou, "max_dets": applied_max_dets,
            },
            "gate_evidence": conf_param.gate_evidence,
            "calibration_evidence": evidence,
        }
        try:
            curve_identity_hex = calibration_curve_identity(curve_body)
        except (TypeError, ValueError) as exc:
            return {"error": f"the operating-point curve for trait {trait!r} could not be kept "
                             f"(its body cannot be recorded): {exc}"}
        # The evidence rides in the curve artifact, read back by identity, never on this response.
        try:
            store.replace(calibration_curve_key(curve_identity_hex), curve_body)
        except Exception:
            logger.warning("could not persist operating-point curve", exc_info=True)
        else:
            extra["calibration_curve_path"] = str(calibration_curve_path(curve_identity_hex))
            extra["calibration_evidence_key"] = curve_identity_hex
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
    if device != "cpu" and (resolved_tile_bool or len(resolved_paths) > 8):
        import torch

        if not torch.cuda.is_available():
            cpu_warning = (
                f"CUDA not available, running {len(resolved_paths)} image(s)"
                f"{' tiled' if resolved_tile_bool else ''} on CPU, which is much slower. Install a "
                "CUDA torch build (see environment.yml) to use the GPU."
            )
            logger.warning(cpu_warning)

    results = predictor.predict_batch(
        resolved_paths, tile=resolved_tile_bool, tile_size=resolved_tile, overlap=resolved_overlap,
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

    # This run's name->id map, resolved once above (ahead of calibration) and reused for decode.
    id_map = _door_id_map
    out = {
        "checkpoint": checkpoint.path,
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
        "subject": _subject,
        "attribute": _attribute,
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
    "under, so this bucket is stamped unvalidated whatever its operating point cleared. The "
    "prediction-document refusal is unaffected by either absence: a bucket here that already holds "
    "a document from a prior run still refuses a second publish the same way one under a dataset "
    "root does. Write into a dataset's own predictions layout "
    "(<dataset_root>/predictions/<model>/<date>) for the two absent guarantees."
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

    Returns ``(out, resolution, dataset_root, refusal)``. ``refusal`` is the door's own error
    dict, or ``None`` otherwise: raised when the requested bucket carries review verdicts and the
    caller asked to overwrite it (``BucketHasVerdicts``), or, since both callers of this helper
    (``run_inference``, ``deliver_per_image_counts``'s live path) opt into the document guard,
    when the requested bucket already holds a prediction document with no verdict yet recorded
    against it, whatever ``overwrite`` says (``BucketHoldsDocuments``). One resolution for both
    writers here that persist a bucket, so the canonical ``predictions/<model>/<date>`` redirect
    (which varies the model segment, the one every date-keyed reader enumerates) and the bespoke
    last-segment redirect cannot drift apart, on every branch including the one under no dataset
    root, whose verdict guard alone is inoperative (the document guard is not, see
    ``_NO_DATASET_ROOT_NOTE``). ``dataset_root`` is ``None`` for a bucket under no dataset.
    """
    from tcip_mcp.dataset_layout import prediction_dir
    from tcip_mcp.prediction_buckets import (
        BucketHasVerdicts,
        BucketHoldsDocuments,
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
                str(prediction_dir(canonical_dataset_root, exc.suggested, base_name))
                if canonical_dataset_root is not None
                else str(parent / exc.suggested)
            )
        error: dict = {"error": str(exc), "suggested_bucket": suggested}
        if isinstance(exc, BucketHasVerdicts):
            error["verdict_count"] = exc.count
        else:
            error["document_stem_count"] = exc.document_stem_count
        return None, None, dataset_root, error
    return out, resolution, dataset_root, None


def _calibration_evidence(result: dict) -> dict | None:
    """The evidence this run's calibration gate ran over, read back from the artifact it was kept
    in, or ``None`` for a run that resolved no calibrated operating point.

    Read rather than carried on the run's own response: the records are the largest thing a
    calibration produces, and only a door earning a validation record has any use for them.

    The record read back is re-encoded and its digest compared against ``identity`` (the run's
    own carried key, which is also its identity, see :func:`calibration_curve_identity`); a
    difference raises ``ValueError`` naming both, since the evidence the count gate would run
    over is then not what this run wrote. An absent record still returns ``None``.
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
    try:
        evidence = _calibration_evidence(result)
    except ValueError as exc:
        return None, {"error": str(exc)}
    if evidence is None:
        return None, {"error": (
            f"the run reports a validated operating point for trait {trait!r} but kept no "
            "calibration evidence to earn a validation record from, so these counts cannot be "
            "stamped validated. The evidence is written beside the confidence sweep at calibration "
            "time; re-run the calibration. This has no acknowledgement route: it is a missing "
            "validation record for a run that already reports itself validated, not an ungated "
            "dimension a breeder can choose to ship unvalidated."
        )}
    if dataset_root is None:
        logger.warning(_NO_DATASET_ROOT_NOTE.format(bucket=bucket))
        return None, None
    # result["validated"] is only ever set True while trait was truthy (_run_inference_verified
    # gates its calibration pass on trait), so a validated result always carries one here.
    assert trait is not None
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

    True once the bucket's own document refusal has already run: the resolver refuses a publish
    into a bucket a prior run's documents already occupy. ``out`` can still hold what that
    refusal does not check for: an earlier run's own ``operating_point.json`` with no document
    (a stamp-only bucket the document predicate admits), a non-``operating_point`` sidecar the
    document check never consults, or, for a directory reused across regimes, a raster pass'
    progress record under its own ``.tcip/``. Short of those and the residual race
    ``resolve_writable_bucket``'s own docstring states (two publishers resolving the same clean
    bucket before either writes), ``out`` holds nothing but what this call is about to add.

    The steps ``run_inference`` and ``deliver_per_image_counts`` share once each has resolved its own
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
    assert sha is not None  # _run_inference_verified always stamps its own checkpoint's real hash
    producer = prediction_producer(checkpoint_path, sha)
    id_map = result.get("id_map")
    subject, attribute = result.get("subject"), result.get("attribute")
    if attribute is not None:
        unmapped = unmapped_label_ids(result["results"], id_map)
        if unmapped:
            raise ValueError(
                f"{out}: this classified run decoded to id(s) {unmapped}, not keys of its "
                f"recorded id_map ({sorted((id_map or {}).values())}); refusing before any "
                "document is written."
            )
    has_masks = False
    dropped = 0
    for r in result["results"]:
        out_json = out / f"{Path(r['image']).stem}.json"
        # Read before the write: a drop can empty a mask list that was genuinely there, and
        # has_masks must reflect what this run used, not what happened to survive the drop.
        has_masks = has_masks or bool(r.get("masks"))
        dropped += write_predictions_json(
            out_json, r, created_by=producer, id_map=id_map, subject=subject, attribute=attribute)
        written.append(str(out_json))

    image_filenames = {Path(r["image"]).stem: Path(r["image"]).name for r in result["results"]}
    op_stamp = operating_point_stamp(
        result.get("operating_point"),
        validated=draft is not None,
        validated_by=None,
        tile_size_validated=tile_size_validated,
        shippable_issues=result.get("shippable_issues", []),
        id_map=id_map,
        subject=subject,
        attribute=attribute,
        trait=trait,
        dataset_hash=result.get("dataset_hash"),
        checkpoint=Path(checkpoint_path).stem,
        checkpoint_sha256=sha,
        experiment_id=result.get("experiment_id"),
        images_dir=images_dir,
        raster_path=None,
        produced_at=result.get("produced_at"),
        calibration_curve_path=result.get("calibration_curve_path"),
        gate_evidence_summary=result.get("gate_evidence_summary"),
        image_filenames=image_filenames,
    )
    if has_masks:
        # The run-constant mask-binarize threshold travels once here rather than per-annotation.
        op_stamp["mask_binarize"] = mask_binarize_provenance()
    return written, dropped, _seal_and_stamp(out, op_stamp, draft)


def _publish_bucket_bracket(result: dict, *, out: Path, checkpoint_path: str, trait: str | None,
                           images_dir: str | None, dataset_root: Path | None,
                           allow_unvalidated_staging: bool) -> dict:
    """Publish a live run's predictions into ``out``, gated and linked exactly as
    ``run_inference`` does: the authoritative post-inference tile gate, the count claim's own
    gate, the frozen-lineage-pointer refusal, the write, and the post-write lineage link. Shared so
    ``deliver_per_image_counts``' live-with-``predictions_dir`` path publishes under the identical contract
    rather than a second implementation. ``out`` has already cleared the bucket-immutability
    resolver (verdicts, and a prior run's own documents) before either caller reaches here; this
    function does not re-resolve it.

    Returns a dict: ``refusal`` (the door's own error dict, or ``None``), ``written``,
    ``dropped_boxes``, ``op_stamp``, ``tile_size_validated`` and ``lineage_linked`` (``True``/
    ``False`` for an attempted link, ``None`` when the run named no experiment to link). A non-
    ``None`` ``refusal`` leaves the bucket untouched and the other values empty/``None``.
    """
    from tcip_mcp.pipelines.resolution import check_delivery_gate, tile_size_gate_flag

    # Re-checked against the real predictor's own resolution: an earlier sniff (if any) is only an
    # early opt-out, this stays the authoritative gate.
    tile_ref = tile_size_gate_flag(result.get("operating_point"))
    tile_flags: dict[str, str | None] = {"tile_size": tile_ref} if tile_ref is not None else {}
    gate = check_delivery_gate(tile_flags, allow_unvalidated_staging=allow_unvalidated_staging)
    if not gate.ok:
        return {"refusal": {"error": gate.reason, "tile_size_validated": tile_ref},
                "written": [], "dropped_boxes": 0, "op_stamp": {},
                "tile_size_validated": tile_ref, "lineage_linked": None}
    tile_size_validated = gate.stamp.get("tile_size")

    # The count claim's own gate, run before a single file is written.
    draft, refusal = _draft_count_claim(
        result, trait=trait, bucket=out, dataset_root=dataset_root,
        tile_size_validated=tile_size_validated)
    if refusal is not None:
        return {"refusal": refusal, "written": [], "dropped_boxes": 0, "op_stamp": {},
                "tile_size_validated": tile_size_validated, "lineage_linked": None}

    exp_id = result.get("experiment_id")
    if exp_id:
        # Checked before the publisher writes the bucket, ahead of this door's own @audited entry
        # (appended only after the caller's body returns), so nothing on disk needs unwinding.
        from tcip_mcp.experiments import pointer_frozen

        frozen = pointer_frozen(exp_id, "lineage", "predictions", str(out))
        if frozen is not None:
            return {"refusal": {"error": frozen}, "written": [], "dropped_boxes": 0,
                    "op_stamp": {}, "tile_size_validated": tile_size_validated,
                    "lineage_linked": None}

    written, dropped_boxes, op_stamp = _publish_image_predictions(
        out, result, checkpoint_path=checkpoint_path, trait=trait, images_dir=images_dir,
        tile_size_validated=tile_size_validated, draft=draft)

    # Close the data->model->predictions chain: link this bucket into the producing run's lineage.
    # Additive first-write, the terminal-state lock permits it into a still-empty predictions field.
    lineage_linked = None
    if exp_id:
        try:
            from tcip_mcp.experiments import update_lineage

            update_lineage(exp_id, predictions=str(out))
            lineage_linked = True
        except Exception:
            logger.warning("could not link predictions into experiment lineage", exc_info=True)
            lineage_linked = False

    return {"refusal": None, "written": written, "dropped_boxes": dropped_boxes,
            "op_stamp": op_stamp, "tile_size_validated": tile_size_validated,
            "lineage_linked": lineage_linked}


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


def _raster_pass_input_mismatches(recorded: dict, current: dict) -> list[str]:
    """Every top-level identity field (everything but ``schema_version``, compared by the reader
    that reads it off the record directly, and ``operating_point``, compared by
    :func:`_raster_pass_identity_mismatches` below) naming a difference between a recorded
    raster-pass identity and this call's own.

    A key union over both sides, not a fixed field tuple, so a field :func:`_raster_pass_identity_body`
    gains later is compared here without a second list to keep in sync with it.
    """
    fields = sorted((set(recorded) | set(current)) - {"schema_version", "operating_point"})
    return [field for field in fields if recorded.get(field) != current.get(field)]


def _raster_pass_identity_mismatches(recorded: dict, current: dict) -> list[str]:
    """Every field naming a difference between a recorded raster-pass identity and this call's
    own, for a resume refusal to list by name."""
    differing = _raster_pass_input_mismatches(recorded, current)
    recorded_op = recorded.get("operating_point") or {}
    current_op = current.get("operating_point") or {}
    differing.extend(
        f"operating_point.{field}" for field in sorted(set(recorded_op) | set(current_op))
        if recorded_op.get(field) != current_op.get(field)
    )
    return differing


def _load_raster_pass_prior(bucket: Path) -> dict:
    """Every tile batch a bucket's progress already holds, merged in tile order into the shape
    ``GenericPredictor._tiled_infer_core`` seeds its own accumulators from.

    Ordered by the numeric index parsed out of each ``batch-<index>`` key, never by the order the
    store happens to hand keys back in: the zero-padded width only makes that order agree by
    coincidence, on a backend that lists keys lexically, for a grid this platform actually tiles.
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


def _export_predictions_raster(
    *, checkpoint, raster_path: str, out: Path, resolution, device: str | None,
    conf_threshold: float | None, tile_size: int | None, overlap: float | None, tile_batch_size: int,
    global_nms_iou: float | None, max_dets: int | None, postprocess: str, require_masks: bool,
    experiment_id: str | None, allow_unvalidated_staging: bool, trait: str | None = None,
    resume: bool = False, overwrite: bool = False,
) -> dict:
    """The windowed-raster regime of :func:`run_inference`: tiled detection/instance_seg
    inference over a whole georeferenced (or merely huge) raster too large to decode whole, sourced
    from the windowed raster layer (:func:`~tcip_mcp.pipelines.raster_source.open_raster`) rather
    than an ordinary directory of per-image captures. Always tiled: there is no untiled option, the
    whole point of this regime is a raster too large for one. ``out``/``resolution`` are the bucket
    :func:`run_inference` already resolved (immutability/redirect), shared with the ordinary
    regime rather than a second implementation of that check. ``allow_unvalidated_staging`` is
    :func:`run_inference`'s own staging escape, forwarded unchanged.

    Persists one prediction bucket: since there is no natural directory-of-per-plant-images shape
    for a whole-raster capture, "one image" is the whole raster, so the bucket holds exactly one
    ``<raster stem>.json`` prediction file (in full-raster pixel space) plus the same
    ``operating_point.json`` sidecar convention every other bucket carries.

    ``trait`` is ``None`` for every raster export with no reserved calibration region (the
    original, byte-identical raw path: the persisted operating point is never stamped validated
    here, conf has no per-dataset calibration for a raster source, a validated per-plant count is
    earned later at delivery). ``run_inference`` only ever passes a real ``trait`` here once
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
    is unconditional, never overridable via ``allow_unvalidated_staging`` (there is no value to
    provisionally proceed with). This door has no real-but-unvalidated tile scale left to admit on
    the staging escape: every basis the gate below resolves to either clears it on its own or is this
    no-basis-at-all case. An explicit ``tile_size`` that contradicts the checkpoint's own recorded
    geometry refuses before that, from ``resolve_tile_regime`` itself. For the ``trait`` path, the
    block calibration's own reserved-region bands must also be tiled at this same resolved edge;
    a split manifest recorded at a different edge refuses there too.

    Every call over a raster that is not mask-bearing (``instance_seg`` with ``require_masks``)
    records this pass' own identity and, as tile batches flush, its progress into
    ``RASTER_PASS_PROGRESS_STORE`` under ``<out>/.tcip/``, so an interruption anywhere in the pass
    leaves a recoverable trail rather than nothing. ``resume=True`` continues that trail: the
    recorded identity's checkpoint, raster content, trait, experiment and tile batch size must
    match this call's own, or it refuses naming what differs. A resumed pass then applies the
    recorded operating point directly rather than re-deriving one, since merging tiles run at two
    different operating points would corrupt the count; for the ``trait`` path this means the
    reserved-band calibration pass that earned that operating point runs once, at the interrupted
    attempt, never again at resume, and the recorded conf/cross_tile_nms/max_dets are applied as
    the pass' own fact. ``resume=False`` over a bucket already carrying progress refuses too,
    naming ``resume=True`` or ``overwrite=True`` (which discards the partial pass, and conflicts
    with ``resume=True``) as the two ways out. On completion every progress record for this bucket
    is deleted in one transaction, whether the pass ran straight through or resumed one
    interruption.
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
        checkpoint, device=device, score_threshold=applied_conf,
        nms_iou=applied_nms_iou, max_dets=applied_max_dets,
    )
    identity = resolve_model_identity(checkpoint, experiment_id=experiment_id)
    if identity["experiment_id"]:
        # Checked before the raster pass, not after: a blob write cannot join the record's own
        # transaction, so this is the one chance to refuse before the export writes anything.
        from tcip_mcp.experiments import pointer_frozen

        frozen = pointer_frozen(identity["experiment_id"], "lineage", "predictions", str(out))
        if frozen is not None:
            return {"error": frozen}

    # No images_dir for a raster source; resolved ahead of the pass (never after it) so a
    # classified run with no id_map refuses before the expensive tiled pass runs.
    raster_subject, raster_attribute = run_scope(predictor)
    id_map = resolve_decode_id_map(predictor, None)
    raster_refusal = unmapped_classified_run(
        {"subject": raster_subject, "attribute": raster_attribute}, id_map, images_dir=None)
    if raster_refusal is not None:
        return {"error": raster_refusal}

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
            f"tile_size could not be resolved for {checkpoint.path}: this checkpoint carries no "
            "persisted training tile geometry and no tile_size was given explicitly, so this "
            "always-tiled regime has no real basis to run at. Pass tile_size explicitly, or "
            "retrain with tile geometry persisted."
        )}

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
        recorded_schema_version = existing_pass_identity.get("schema_version", 1)
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
    block_calibration_snapshot: dict | None = None
    bundle_dataset_hash: str | None = None
    bundle_shippable_issues: list[str] = []

    if trait is not None:
        from tcip_mcp.pipelines.operating_point import set_detector_operating_point

        if resume:
            # Confirmed above to continue the recorded pass: apply its operating point directly
            # rather than re-run calibration and risk refusing over a re-derivation's float noise.
            snapshot = store.read(_raster_pass_key(out, "block-calibration"), default=None)
            if snapshot is None:
                return {"error": (
                    f"resume=True but {out}'s recorded pass carries no block-calibration snapshot "
                    "to resume the calibrated operating point from; it did not calibrate at all, "
                    "or was recorded before this door persisted one."
                )}
            recorded_op = existing_pass_identity["operating_point"]
            conf = recorded_op["conf"]
            applied_nms_iou = recorded_op["cross_tile_nms"]
            predictor.score_threshold = conf
            set_detector_operating_point(predictor.model, score_thresh=conf,
                                         detections_per_img=applied_max_dets)
            predictor.max_dets = recorded_op["max_dets"]
            op_provenance = snapshot["op_provenance"]
            tile_size_validated = snapshot["tile_size_validated"]
            claim_scope_validated = snapshot["claim_scope_validated"]
            conf_source = snapshot["conf_source"]
            block_prov = snapshot["block_prov"]
            bundle_dataset_hash = snapshot["dataset_hash"]
            bundle_shippable_issues = snapshot["shippable_issues"]
            if snapshot["count_claim_eligible"]:
                if bucket_root is None:
                    logger.warning(_NO_DATASET_ROOT_NOTE.format(bucket=out))
                else:
                    try:
                        draft = _open_count_claim(
                            snapshot["block_evidence"], trait=trait,
                            checkpoint_sha256=identity["sha256"],
                            producing_experiment_id=identity["experiment_id"],
                            dataset_root=bucket_root)
                    except ValueError as exc:
                        return {"error": f"the count claim for trait {trait!r} was not earned: {exc}"}
        else:
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
                    predictor, trait_name=trait,
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
            predictor.score_threshold = conf
            set_detector_operating_point(predictor.model, score_thresh=conf,
                                         detections_per_img=applied_max_dets)
            predictor.max_dets = None

            op_bundle = block_calibrated_export_operating_point(
                block_bundle, trait=trait, tile_size=resolved_tile,
                tile_size_source=tile_size_source, tile_size_derived_from=tile_size_derived_from)
            op_provenance = op_bundle.to_provenance()["operating_point"]

            tile_ref = tile_size_gate_flag(op_provenance)
            gate_flags: dict[str, str | None] = {"claim_scope": claim_scope_flag}
            if tile_ref is not None:
                gate_flags["tile_size"] = tile_ref
            gate = check_delivery_gate(gate_flags, allow_unvalidated_staging=allow_unvalidated_staging)
            if not gate.ok:
                reason = gate.reason if claim_scope_mismatch is None else (
                    f"{gate.reason} {claim_scope_mismatch}")
                return {"error": reason, "tile_size_validated": tile_ref,
                        "claim_scope_validated": claim_scope_flag}
            tile_size_validated = gate.stamp.get("tile_size")
            claim_scope_validated = gate.stamp.get("claim_scope")
            conf_source = "block_calibration"
            bundle_dataset_hash = op_bundle.dataset_hash
            bundle_shippable_issues = op_bundle.shippable_issues()

            # The count claim's own gate, run before the always-expensive whole-mosaic pass.
            count_claim_eligible = (
                op_bundle.is_shippable and claim_scope_validated != VALIDATED_FALSE
                and tile_size_validated != VALIDATED_FALSE)
            if count_claim_eligible:
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
            # Persisted beside this pass' identity record below, so a resume can apply this same
            # operating point without re-running calibration.
            block_calibration_snapshot = {
                "op_provenance": op_provenance, "claim_scope_validated": claim_scope_validated,
                "tile_size_validated": tile_size_validated, "conf_source": conf_source,
                "block_prov": block_prov, "block_evidence": block_evidence,
                "count_claim_eligible": count_claim_eligible,
                "dataset_hash": bundle_dataset_hash, "shippable_issues": bundle_shippable_issues,
            }
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
        tile_flags: dict[str, str | None] = {"tile_size": tile_ref} if tile_ref is not None else {}
        gate = check_delivery_gate(tile_flags, allow_unvalidated_staging=allow_unvalidated_staging)
        if not gate.ok:
            return {"error": gate.reason, "tile_size_validated": tile_ref}
        tile_size_validated = gate.stamp.get("tile_size")
        bundle_dataset_hash = op_bundle.dataset_hash
        bundle_shippable_issues = op_bundle.shippable_issues()

    from tcip_mcp.pipelines.raster_source import open_raster

    current_pass_identity = _raster_pass_identity_body(
        raster_identity=raster_identity, checkpoint_sha256=identity["sha256"], trait=trait,
        experiment_id=identity["experiment_id"], tile_batch_size=tile_batch_size,
        conf=predictor.score_threshold, cross_tile_nms=applied_nms_iou, max_dets=predictor.max_dets,
        tile_size=resolved_tile, overlap=resolved_overlap, tile_resize=tile_resize,
        postprocess=postprocess, require_masks=require_masks,
    )
    prior: dict | None = None
    if existing_pass_identity is not None:
        mismatches = _raster_pass_identity_mismatches(existing_pass_identity, current_pass_identity)
        if mismatches:
            return {"error": (
                f"resume=True but the recorded pass over {out} differs from this call in "
                f"{mismatches}: a resumed pass must be the identical pass, since merging tiles "
                "run at two different operating points would corrupt the count."
            )}
        prior = _load_raster_pass_prior(out)
    elif record_progress:
        store.replace(identity_key, current_pass_identity, expect=Version.ABSENT)
        if block_calibration_snapshot is not None:
            store.replace(
                _raster_pass_key(out, "block-calibration"), block_calibration_snapshot,
                expect=Version.ABSENT)

    def _record_raster_pass_batch(start_index: int, end_index: int, batch: dict) -> None:
        # Six digits also sorts the key into tile order lexically (covers any grid this
        # platform tiles); _load_raster_pass_prior parses the index itself rather than lean on that.
        store.replace(
            _raster_pass_key(out, f"batch-{start_index:06d}"),
            {"tile_index_range": [start_index, end_index], **batch}, expect=Version.ABSENT,
        )

    # The model's own in_chans is the channel routing hint; the reader's real band count is
    # checked against it inside predict_tiled before any tile is read.
    with open_raster(raster_path, predictor.in_chans) as reader:
        result = predictor.predict_tiled(
            reader, tile_size=resolved_tile, overlap=resolved_overlap,
            tile_batch_size=tile_batch_size, global_nms_iou=applied_nms_iou,
            postprocess=postprocess, require_masks=require_masks, source_label=str(raster_path),
            tile_resize=tile_resize, prior=prior,
            progress=_record_raster_pass_batch if record_progress else None,
        )

    from datetime import datetime, timezone

    out.mkdir(parents=True, exist_ok=True)
    sha = identity["sha256"]
    producer = prediction_producer(checkpoint.path, sha)
    pred_path = out / f"{Path(raster_path).stem}.json"
    if raster_attribute is not None:
        unmapped = unmapped_label_ids([result], id_map)
        if unmapped:
            raise ValueError(
                f"{pred_path}: this classified run decoded to id(s) {unmapped}, not keys of its "
                f"recorded id_map ({sorted((id_map or {}).values())}); refusing before writing."
            )
    # Read before the write: a drop can empty a mask list that was genuinely there, and has_masks
    # must reflect what this run used, not what happened to survive the drop.
    has_masks = bool(result.get("masks"))
    dropped_boxes = write_predictions_json(
        pred_path, result, created_by=producer, id_map=id_map,
        subject=raster_subject, attribute=raster_attribute)

    produced_at = datetime.now(timezone.utc).isoformat()
    op_stamp = operating_point_stamp(
        op_provenance,
        validated=draft is not None,
        validated_by=None,
        tile_size_validated=tile_size_validated,
        shippable_issues=bundle_shippable_issues,
        id_map=id_map,
        subject=raster_subject,
        attribute=raster_attribute,
        trait=trait or None,
        dataset_hash=bundle_dataset_hash,
        checkpoint=Path(checkpoint.path).stem,
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
    # Reused from the sampling taken at the top of this call, never resampled: a consumer
    # resolving these boxes through a raster's georeferencing needs this to name that raster.
    op_stamp["raster_content_identity"] = raster_identity
    op_stamp = _seal_and_stamp(out, op_stamp, draft)

    exp_id = identity["experiment_id"]
    if exp_id:
        try:
            from tcip_mcp.experiments import update_lineage

            update_lineage(exp_id, predictions=str(out))
        except Exception:
            logger.warning("could not link predictions into experiment lineage", exc_info=True)

    # The pass finished: whatever progress it left has nothing left to resume.
    _clear_raster_pass_progress(out)

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


_DELIVER_PER_IMAGE_COUNTS_LIVE_ONLY_DEFAULTS = {
    "conf_threshold": None, "device": None, "tile": None, "tile_size": None, "overlap": None,
    "global_nms_iou": None, "max_dets": None, "calibration_labels_dir": None,
    "calibration_images_dir": None, "split_manifest_dir": None, "experiment_id": None,
    "postprocess": "nms", "tile_batch_size": 96,
}
"""``deliver_per_image_counts`` parameters meaningful only for its live regime, mapped to the documented
default a bucket-regime call is judged against. A non-``None`` default (``postprocess``,
``tile_batch_size``) makes stated-at-default indistinguishable from stating nothing, so a call
naming one at its own default is honestly admitted rather than refused; every other parameter
here is ``None``-defaulted, so any non-``None`` value it carries is unambiguously a live-only
statement and refuses."""


@mcp.tool()
@audited
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
    tile_batch_size: int = 96,
    global_nms_iou: float | None = None,
    max_dets: int | None = None,
    postprocess: str = "nms",
    calibration_labels_dir: str | None = None,
    calibration_images_dir: str | None = None,
    split_manifest_dir: str | None = None,
    experiment_id: str | None = None,
    allow_unvalidated_staging: bool = False,
    predictions_dir: str | None = None,
) -> dict:
    """Export a CSV summary of detection counts per image, from a live run or a persisted bucket.

    A bucket here is the directory a prediction run persists into, not a score bin: it turns
    immutable once a review verdict lands on any image inside it.

    Two source regimes, exactly one stated:

    - Live (``checkpoint_path`` and ``images_dir``, both required together): routes through the
      private verified pass so the per-image counts resolve the same firewalled operating point
      (conf/NMS/tiling/max_dets) as ``run_inference``; the CSV is a
      count-bearing deliverable, so it must not be produced at a different, untiled, truncating
      operating point. Passing ``predictions_dir`` too persists the run's own predictions into
      that bucket, gated (``allow_unvalidated_staging`` clears only its own tile-scale staging gate,
      never the CSV's own delivery gate below) and linked exactly as ``run_inference`` publishes
      one (the same bucket-immutability resolution, refusing on a verdict or on a document a
      prior run left with none, the same frozen-lineage-pointer refusal, and the same lineage
      link), then reads the CSV's own validity back off the bucket it just wrote. A document
      refusal here is returned before the checkpoint is loaded, the same as ``run_inference``'s
      own. Without ``predictions_dir`` the counts rest on
      one in-memory pass with nothing a reviewer can re-open, and this door takes no acknowledgement
      to ship an unvalidated CSV anyway, so an unvalidated live pass with no ``predictions_dir``
      cannot be delivered at all; a bucket published this way (or by ``run_inference``) can also be
      promoted to validated later, with no re-run, through the review validation route
      (``validate_reference``), then re-delivered through this door's bucket regime below.
    - Bucket (``predictions_dir`` alone, ``checkpoint_path``/``images_dir`` both absent): no GPU,
      no predictor import, no checkpoint argument at all. Reads an existing, reviewed per-image
      prediction bucket's own ``operating_point.json`` stamp as its identity and validity source,
      counting real detections (a ``Point`` excluded) off each of its documents, sorted by stem.
      Every parameter meaningful only to a live run (conf/device/tiling/NMS/max_dets/calibration/
      split-manifest/experiment_id) refuses here by name, since a bucket regime call cannot honor
      a stated live parameter with nothing behind it; ``postprocess``/``tile_batch_size`` refuse
      only away from their own documented default, since stated-at-default is indistinguishable
      from unstated for a non-``None`` default. A bucket recording ``raster_path`` (a whole-mosaic
      bucket) refuses naming
      ``deliver_orthomosaic_plant_counts``: one mosaic total is not a per-image count. A stamp
      recording a different, non-``None`` trait refuses as a positive contradiction, validated or
      not.

    Delivery gate, both regimes: one composition, inside ``export_detection_csv``, over the count
    operating point and (if tiled) the tile geometry, reconciled from the bucket's own sidecar
    rather than trusted from a caller string; ships only when every dimension clears. This tool
    itself builds no acknowledgement (no MCP door ever does), so an unvalidated dimension always
    refuses here, whichever regime produced it; the bucket regime's own count is reachable
    provisionally through the web results route's count export, which builds a real
    ``Acknowledgement`` from the breeder's own reason and identity. The other documented route
    around a refusal is the one above: promote the bucket to validated through the review
    validation route, then re-deliver through the bucket regime. A refused
    delivery still names what happened to a
    ``predictions_dir`` the live regime published before the CSV's own gate ran
    (``bucket_published``, ``bucket_redirected``, ``lineage_linked``, beside ``csv_delivered:
    false``): the bucket is the caller's own stated ``predictions_dir`` intent, published under
    ``run_inference``'s own contract, and it survives a refused CSV so the review-promotion
    workflow above can proceed from it.

    Every row's image cell holds the source image's basename with its extension, never the bare
    document stem a prediction file is itself named for. The live regime without ``predictions_dir``
    reads it straight off the pass's own per-image results; a bucket-reading regime (the bucket
    regime proper, or the live regime with ``predictions_dir``, both reading documents by stem)
    resolves it through the bucket's own stamp-recorded ``image_filenames`` map. A stem the map does
    not name (a stamp written before the map existed, or a document left over in the bucket from an
    earlier publish this run's fresh map does not cover) falls that row back to the bare stem
    instead, and the response's ``image_note`` key discloses which stems fell back and why, carried
    on both a delivered response and a ``DeliveryRefused`` refusal, in either bucket-reading regime;
    absent when nothing fell back.

    Meaning door: ``trait``'s per-image-count operationalization must be recorded and
    breeder-confirmed, checked before the pass runs (live) or the bucket is read (bucket regime),
    not after. Only the CSV's own delivery-gate refusal, raised after the pass or the bucket read,
    returns ``image_count`` and ``total_detections`` beside the error, so a check placed after that
    gate would hand back the very numbers it refused to write; an operationalization refusal
    (withdrawn or never recorded), and every refusal the shared publish bracket raises before the
    CSV's own gate ever runs (a fabricated tile scale, an unearned count claim, a frozen lineage
    pointer), carry neither.

    The live regime's own ``checkpoint_sha256``/``experiment_id`` are the run's asserted identity,
    never corroborated; the bucket regime's are the stamp's own asserted identity, labelled so, no
    ``conf_source`` (not a stamp key, never fabricated from one). Either way the CSV's own
    ``producer_model_sha256``/``producing_experiment_id`` columns, and this response's
    ``operating_point_validated``, are ``export_detection_csv``'s returned tail, corroborated
    against what a record outside the stamp actually answers for, so the two can legitimately
    differ (or the tail can read unknown where the asserted identity does not). ``validated`` is
    live-only: the run's own verdict over the dimensions it resolved, absent from the bucket
    regime's response since no run happens there. The gate's own outcome travels in
    ``operating_point_validated`` in both regimes, with ``unvalidated_dimensions`` naming every
    gated dimension that did not validate, so a row where it and ``tile_size_validated`` disagree
    is still explained; ``tile_size_validated`` only joins the gate's own outcome when a bucket
    backs the delivery (live-with-``predictions_dir`` or the bucket regime), read off the same
    writer summary rather than re-entering the gate. On the live regime
    with no ``predictions_dir``, tile_size is never one of the gate's flags at all (nothing on disk
    for a second gate call to reconcile it from), so the CSV's own ``unvalidated_dimensions`` cell
    can only ever name ``operating_point`` on that path; ``tile_size_validated`` there is instead
    the run's own in-memory tile-scale flag, which never passed the gate, and this response
    carries one further fact nothing on disk backs: ``run_conf_validated_against``, the run's own
    narrowed conf reference, distinct from ``operating_point_validated`` (which floors false on
    that path, since nothing persisted answers for it).

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
            delivery rests on. Required, in both regimes; a bucket regime call also refuses when
            the bucket's own stamp recorded a different, non-``None`` trait.
        conf_threshold: Live regime only. Minimum confidence score. ``None`` (default) states
            nothing and forwards that on, leaving the value to run at the platform default; a
            stated value is an explicit override even when it equals the platform default.
        device: Live regime only. Device to use.
        tile: Live regime only. Tiled (SAHI-style) inference for small dense objects. ``None``
            (default) forwards to ``run_inference`` unresolved, see its own ``tile`` doc.
        tile_size: Live regime only. Sliding-window tile edge (px).
        overlap: Live regime only. Fractional tile overlap.
        tile_batch_size: Live regime only. Tiles per forward batch.
        global_nms_iou: Live regime only. Cross-tile NMS IoU. ``None`` (default) states nothing
            and forwards that on, leaving the value to be derived; a stated value is an explicit
            override even when it equals the platform default. See ``run_inference``'s own doc.
        max_dets: Live regime only. Full-frame detection cap. ``None`` (default) states nothing
            and forwards that on, leaving the value to be derived; a stated value is an explicit
            override even when it equals the platform default. See ``run_inference``'s own doc.
        postprocess: Live regime only. Cross-tile merge, "nms" or "nmm".
        calibration_labels_dir: Live regime only. Labeled dir for calibrating + held-out
            validating the operating point.
        calibration_images_dir: Live regime only. Images for the calibration labels (defaults to
            ``images_dir``).
        split_manifest_dir: Live regime only. Restrict calibration to one capture date's
            ``calibration`` side of a split manifest (forwarded to ``run_inference``; see its own
            doc), so a manifest-restricted calibration's evidence can earn a validation record
            through this door.
        experiment_id: Live regime only. The run that produced the checkpoint, for provenance
            (forwarded to ``run_inference``; see its own doc for the best-effort resolution when
            omitted).
        allow_unvalidated_staging: Live regime with ``predictions_dir`` only. Persist the bucket
            even when tile_size has no real basis, stamping ``tile_size_validated=false``; the
            staging escape a raw bucket write shares with ``run_inference``, never a route to
            deliver the CSV itself unvalidated (this door takes no acknowledgement for that).
        predictions_dir: Live regime: directory to persist the counted predictions into, resolved
            and stamped the way ``run_inference`` resolves and stamps a bucket (a relative
            path resolves against the platform state root; a bucket carrying review verdicts redirects to
            a fresh variant); omitted, an unvalidated live pass cannot be delivered at all. Bucket
            regime: the existing bucket to read (required, resolved the same way; no
            writable-bucket resolution or redirect, since nothing is written).
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
        stated_live_only = sorted(
            name for name, value in (
                ("conf_threshold", conf_threshold), ("device", device), ("tile", tile),
                ("tile_size", tile_size), ("overlap", overlap),
                ("global_nms_iou", global_nms_iou), ("max_dets", max_dets),
                ("calibration_labels_dir", calibration_labels_dir),
                ("calibration_images_dir", calibration_images_dir),
                ("split_manifest_dir", split_manifest_dir), ("experiment_id", experiment_id),
                ("postprocess", postprocess), ("tile_batch_size", tile_batch_size),
            ) if value != _DELIVER_PER_IMAGE_COUNTS_LIVE_ONLY_DEFAULTS[name])
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
                acknowledgement=None)
        except OperationalizationRefused as exc:
            return {"error": exc.check.message}
        except DeliveryRefused as exc:
            reason = (
                f"{exc} This door takes no acknowledgement: acknowledge and re-export through "
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
        split_manifest_dir=split_manifest_dir,
        experiment_id=experiment_id,
    )
    if "error" in result:
        return result

    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE, StampScopeUnstated, accepted_references, tile_size_gate_flag,
    )
    from tcip_store import StoreError

    op = result.get("operating_point") or {}
    conf_prov = op.get("conf") or {}
    # Judged against the references accepted for conf's own kind, never the bare validated bool
    # (that would launder a missing/unrecognized/wrong-kind value into a shippable one).
    op_ref = conf_prov.get("validated_against")
    if op_ref not in accepted_references("annotations"):
        op_ref = VALIDATED_FALSE
    # tile_size gates the same way (closing the asymmetry with run_full_frame_evaluation): a
    # no-basis tile scale is as untrustworthy for a count as an uncalibrated conf; None if untiled.
    tile_ref = tile_size_gate_flag(op)

    dropped_boxes = 0
    bucket_published = False
    lineage_linked = None
    image_note = None
    csv_rows = result["results"]
    if bucket is not None:
        pub = _publish_bucket_bracket(
            result, out=bucket, checkpoint_path=checkpoint_path, trait=trait, images_dir=images_dir,
            dataset_root=bucket_root, allow_unvalidated_staging=allow_unvalidated_staging)
        if pub["refusal"] is not None:
            return pub["refusal"]
        dropped_boxes = pub["dropped_boxes"]
        lineage_linked = pub["lineage_linked"]
        bucket_published = True
        # Counted off the just-published documents, filenames off the run's own just-written stamp;
        # a document from an earlier publish this run's map does not name falls back and discloses.
        filename_map = pub["op_stamp"].get("image_filenames")
        csv_rows, fallback_stems = _bucket_csv_rows(bucket, filename_map)
        image_note = _image_filename_fallback_note(bucket, filename_map, fallback_stems)

    provenance = {
        "producer_model_sha256": result.get("checkpoint_sha256"),
        "producing_experiment_id": result.get("experiment_id"),
        "operating_point_conf": (op.get("conf") or {}).get("value"),
    }
    try:
        csv_path, tail, summary, _event_recorded = export_detection_csv(
            csv_rows, output_path, provenance=provenance, trait=trait,
            operating_point_validated=op_ref,
            pred_dirs=[str(bucket)] if bucket is not None else None,
        )
    except (StampScopeUnstated, StoreError) as exc:
        return {"error": str(exc)}
    except DeliveryRefused as exc:
        reason = str(exc)
        op_validated = exc.gate.stamp.get("operating_point", VALIDATED_FALSE)
        tile_validated = exc.gate.stamp.get("tile_size", tile_ref)
        if bucket is None:
            reason += (
                " These counts were read off an in-memory pass with no prediction bucket behind "
                "them: pass predictions_dir to persist and stamp the predictions they came from, "
                "which is what a validated count CSV rests on; an unvalidated bucket can also be "
                "promoted to validated later through the review validation route, with no re-run."
            )
        else:
            reason += (
                " This door takes no acknowledgement for the CSV itself: validate the dimension "
                "named above, or promote this bucket to validated through the review validation "
                "route (validate_reference) and re-deliver."
            )
        refusal = {
            "error": reason,
            "operating_point_validated": op_validated,
            "tile_size_validated": tile_validated,
            "unvalidated_dimensions": exc.gate.unvalidated_cell(),
            "operating_point": result.get("operating_point"),
            "validated": False,
            "image_count": result["image_count"],
            "total_detections": result["total_detections"],
        }
        if bucket is None:
            # The run's own narrowed reference, distinct from the gate stamp above: nothing on
            # disk backs it without a bucket, so it never answers for operating_point_validated.
            refusal["run_conf_validated_against"] = op_ref
        if bucket is not None:
            assert resolution is not None  # bucket and resolution are set together, above
            refusal["bucket_published"] = bucket_published
            refusal["predictions_dir"] = str(bucket)
            refusal["bucket_redirected"] = resolution.redirected
            refusal["lineage_linked"] = lineage_linked
            refusal["csv_delivered"] = False
            refusal["dropped_nonpositive_boxes"] = dropped_boxes
            if bucket_root is None:
                refusal["note"] = _NO_DATASET_ROOT_NOTE.format(bucket=bucket)
            if image_note is not None:
                refusal["image_note"] = image_note
        return refusal

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
        # The CSV's own written cell (floored across every gated dimension without a column of
        # its own); unvalidated_dimensions names every dimension that did not validate.
        "operating_point_validated": tail["operating_point_validated"],
        "tile_size_validated": (
            (summary["stamp"].get("tile_size") if summary["tile_size_operative"] else None)
            if bucket is not None else tile_ref),
        "unvalidated_dimensions": tail["unvalidated_dimensions"],
        "conf_source": result.get("conf_source"),
        "checkpoint_sha256": result.get("checkpoint_sha256"),
        "experiment_id": result.get("experiment_id"),
        "predictions_dir": str(bucket) if bucket is not None else None,
    }
    if bucket is None:
        # The run's own narrowed reference, distinct from operating_point_validated above,
        # which floors false here since nothing on disk backs it without a bucket.
        out["run_conf_validated_against"] = op_ref
    if bucket is not None:
        assert resolution is not None  # bucket and resolution are set together, above
        out["bucket_redirected"] = resolution.redirected
        out["verdict_guard_operative"] = bucket_root is not None
        out["dropped_nonpositive_boxes"] = dropped_boxes
        out["lineage_linked"] = lineage_linked
        if bucket_root is None:
            out["note"] = _NO_DATASET_ROOT_NOTE.format(bucket=bucket)
        if image_note is not None:
            out["image_note"] = image_note
    # run_inference's own warnings (a CPU-bound workload) are surfaced here too, so a count CSV
    # never ships with the regime it ran in disclosed only in the server log.
    if result.get("warning"):
        out["warning"] = result["warning"]
    return out


def _bucket_csv_rows(
    bucket_path: Path, filename_map: dict[str, str] | None,
) -> tuple[list[dict], list[str]]:
    """A prediction bucket's own per-image documents as ``export_detection_csv``'s row source.

    Real detections only (``detection_annotations``, a ``Point`` excluded), ordered by document
    stem rather than trusted to ``prediction_documents``' own filename sort (which would diverge
    from the live regime's sorted-stem enumeration once ``.json`` changes a stem's relative
    order). The one reader every documents-backed CSV path shares, so a masked detection the
    write side kept on its stored polygon's extent cannot be dropped again by a different,
    box-based predicate downstream.

    Each row's ``image`` value is the stamp-recorded source filename (``filename_map[doc.stem]``,
    the basename with extension the publisher stamped): a document's own filename is only ever
    ``<stem>.json``, which carries no extension to recover. A ``filename_map`` that is absent
    entirely, or names no entry for a given stem, falls that row back to the bare stem; the second
    return value lists the stems that fell back, so a caller can disclose the fallback rather than
    let the CSV's own cells silently say less than the caller believes.
    """
    from tcip_annotation.json_io import detection_annotations, prediction_documents, safe_score

    documents = sorted(prediction_documents(bucket_path), key=lambda p: p.stem)
    image_results = []
    fallback_stems = []
    for doc in documents:
        annotations = detection_annotations(doc)
        scores = [safe_score(a.score) for a in annotations if a.score is not None]
        filename = (filename_map or {}).get(doc.stem)
        if filename is None:
            filename = doc.stem
            fallback_stems.append(doc.stem)
        image_results.append({"image": filename, "count": len(annotations), "scores": scores})
    return image_results, fallback_stems


def _image_filename_fallback_note(
    bucket_path: Path, filename_map: dict[str, str] | None, fallback_stems: list[str],
) -> str | None:
    """The fallback disclosure ``_bucket_csv_rows`` earns, shared by every count-bearing response
    that reads a bucket's image filename map, whether the call succeeds or a ``DeliveryRefused``
    refuses it: names which rows' image cells carry a bare document stem instead of the source
    filename, and why. ``None`` when every row resolved through the map.
    """
    if filename_map is None:
        return (
            f"{bucket_path}'s stamp carries no image filename map: every row derives from a "
            "document the map does not name, so every row's image cell carries the bare document "
            "stem, not the source image's filename."
        )
    if fallback_stems:
        return (
            f"{bucket_path}'s stamp's image filename map names no entry for stem(s) "
            f"{sorted(fallback_stems)}: those rows derive from documents the map does not name, "
            "and their image cells carry the bare stem."
        )
    return None


def per_image_counts_from_bucket(
    predictions_dir: str, output_path: str, *, trait: str,
    project_root: str | Path | None = None,
    acknowledgement: Acknowledgement | None = None,
) -> dict:
    """``deliver_per_image_counts``'s bucket regime, as a core the tool and the web results route
    both call: an existing, reviewed prediction bucket in, no GPU.

    No writable-bucket resolution and no verdict redirect: nothing is written, and reading a
    reviewed bucket is the point. Refuses on the bucket's own mandatory stamp shape before
    counting anything, so a mosaic bucket or a directory with no stamp at all is never counted.

    Runs the ``per_image_count`` meaning check first, against ``trait`` and ``project_root``,
    before the bucket is touched: a caller with its own response shape (the tool's own
    ``{"error": ...}``, the route's structured HTTP detail) composes it from which class this
    raises rather than parsing a message. ``OperationalizationRefused``
    (``tcip_mcp.operationalization``) carries the failed check and no counts, raised either from
    this call's own pre-check, from ``export_detection_csv``'s own pre-check (which additionally
    reads each recorded bucket's ``id_map`` for the confirmed subject, never checked here since
    the bucket is not yet touched), or from that writer's post-gate re-check (a confirmation
    withdrawn, or a spec field moved, since either check); ``DeliveryRefused``
    (``pipelines.resolution``) is the writer's own gate refusal, its ``facts`` attribute set here
    to this call's own counts-bearing facts; ``CountDeliveryRefused`` (``pipelines.resolution``)
    covers everything else this door refuses on (a missing stamp, a whole-raster bucket, an
    unknown or mismatched trait, a malformed filename map, an empty bucket), each carrying the
    same facts the tool's own ``{"error": ...}`` shape carries.
    """
    from tcip_mcp.operationalization import (
        PER_IMAGE_COUNT,
        OperationalizationRefused,
        check_operationalization,
        resolve_trait_and_record,
    )
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE, CountDeliveryRefused, StampScopeUnstated, bucket_scope,
        read_operating_point_sidecar, stamp_names_raster,
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
    try:
        bucket_scope(bucket_path)
    except StampScopeUnstated as exc:
        raise CountDeliveryRefused(str(exc)) from exc
    except StoreError as exc:
        raise CountDeliveryRefused(str(exc)) from exc
    if stamp_names_raster(sidecar):
        raise CountDeliveryRefused(
            f"{bucket_path} is a whole-raster bucket (its stamp records raster_path): one mosaic "
            "total is not a per-image count, and the per_image_count operationalization was never "
            "confirmed for it. Deliver a per-plant count from it through "
            "deliver_orthomosaic_plant_counts instead.")
    if not sidecar.get("images_dir"):
        raise CountDeliveryRefused(
            f"{bucket_path}'s stamp records neither images_dir nor raster_path: it is not a "
            "per-image prediction bucket this door can read.")
    stamp_trait = sidecar.get("trait")
    if stamp_trait is not None and stamp_trait != trait:
        raise CountDeliveryRefused(
            f"{bucket_path}'s stamp was recorded for trait {stamp_trait!r}, not {trait!r}: a "
            "bucket produced for one trait cannot deliver a per-image count under another.")

    filename_map = sidecar.get("image_filenames")
    if filename_map is not None and not isinstance(filename_map, dict):
        raise CountDeliveryRefused(
            f"{bucket_path}'s stamp's image_filenames is not a mapping (got "
            f"{type(filename_map).__name__}): a bucket regime call expects the stem-to-filename "
            "map run_inference, deliver_per_image_counts's live-with-predictions_dir path, or the "
            "web inference worker writes there, or nothing at all.")
    image_results, fallback_stems = _bucket_csv_rows(bucket_path, filename_map)
    if not image_results:
        raise CountDeliveryRefused(
            f"{bucket_path} carries a readable stamp but no prediction documents: an empty "
            "bucket is not a per-image count either.")
    image_count = len(image_results)
    total_detections = sum(r["count"] for r in image_results)

    image_note = _image_filename_fallback_note(bucket_path, filename_map, fallback_stems)

    op = sidecar.get("operating_point") or {}
    provenance = {
        "producer_model_sha256": sidecar.get("checkpoint_sha256"),
        "producing_experiment_id": sidecar.get("experiment_id"),
        "operating_point_conf": (op.get("conf") or {}).get("value"),
    }
    try:
        csv_path, tail, summary, event_recorded = export_detection_csv(
            image_results, output_path, provenance=provenance, trait=trait,
            operating_point_validated=None, pred_dirs=[str(bucket_path)],
            acknowledgement=acknowledgement, project_root=project_root,
        )
    except (StampScopeUnstated, StoreError) as exc:
        raise CountDeliveryRefused(str(exc)) from exc
    except DeliveryRefused as exc:
        exc.facts = {
            "operating_point_validated": exc.gate.stamp.get("operating_point", VALIDATED_FALSE),
            "tile_size_validated": exc.gate.stamp.get("tile_size"),
            "unvalidated_dimensions": exc.gate.unvalidated_cell(),
            "operating_point": sidecar.get("operating_point"),
            "image_count": image_count,
            "total_detections": total_detections,
            "predictions_dir": str(bucket_path),
        }
        if image_note is not None:
            exc.facts["image_note"] = image_note
        raise

    out = {
        "csv_path": csv_path,
        "image_count": image_count,
        "total_detections": total_detections,
        "operating_point": sidecar.get("operating_point"),
        "operating_point_validated": tail["operating_point_validated"],
        "tile_size_validated": (
            summary["stamp"].get("tile_size") if summary["tile_size_operative"] else None),
        "unvalidated_dimensions": tail["unvalidated_dimensions"],
        "acknowledged_by": tail["acknowledged_by"],
        "checkpoint_sha256": sidecar.get("checkpoint_sha256"),
        "experiment_id": sidecar.get("experiment_id"),
        "predictions_dir": str(bucket_path),
        "delivery_event_recorded": event_recorded,
    }
    if image_note is not None:
        out["image_note"] = image_note
    return out
