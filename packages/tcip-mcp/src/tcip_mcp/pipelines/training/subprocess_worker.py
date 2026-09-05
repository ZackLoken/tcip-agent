"""The subprocess entry point ``launch_training`` spawns to run one bespoke training run's
actual body, dataset/loader construction, the audited envelope, ``run_training_envelope()``, in
an isolated OS process, so a leak/OOM/hang in one run can't take down the launching process or any
other concurrent run's process. Everything here mirrors what running the same body synchronously
in-process would do; only the process boundary differs.

Invoked as ``python -m tcip_mcp.pipelines.training.subprocess_worker --run-id ... --experiment-id
... --output-dir ... --resume-from ...``, never imported for its functions elsewhere, only run as
``__main__``. The bootstrap config is read from the run's own output directory, which is the
record the launching process wrote it to, so the two processes cannot disagree on where it is.
"""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from tcip_mcp.pipelines.training.envelope import TrainContext

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume-from", default="")
    return p.parse_args()


def _patch_experiment_config(experiment_id: str, action: str,
                             mutate: Callable[[dict], None]) -> None:
    """Best-effort: open the durable experiment record's own ``config.json`` behind its status
    member, refuse a terminal record, and apply ``mutate`` to the config's ``data`` section
    inside the same transaction before writing it back, a patch of that section, never a
    rewrite of the whole record. ``action`` names the operation for both the terminal refusal
    and its audit line, one string every caller of this run's provenance patches shares with
    its own identity (``patch_experiment_config_tiling``, ``..._id_map``, ``..._split``). Never
    sinks the run if there is no experiment record to patch (experiment tracking is best-effort
    throughout this path, same as every other write in it); a terminal record's refusal is the
    one exception, since a lost provenance write there is itself a run failure, not a
    degradable one.
    """
    from tcip_mcp.experiments import ExperimentTerminal

    try:
        from tcip_store import store

        from tcip_mcp.experiments import config_key, refuse_if_terminal, status_key

        key, st_key = config_key(experiment_id), status_key(experiment_id)
        if not store.exists(key):
            return
        try:
            with store.transaction(key, st_key) as txn:
                state = (txn.read(st_key, default={}) or {}).get("state")
                refuse_if_terminal(experiment_id, action, state)
                cfg = txn.read(key, default={})
                data_cfg = cfg.setdefault("data", {})
                mutate(data_cfg)
                txn.write(key, cfg)
        except ExperimentTerminal as exc:
            from tcip_mcp.experiments import audit_refusal_reraising
            audit_refusal_reraising(experiment_id, action, {}, exc)
    except ExperimentTerminal:
        raise
    except Exception:
        logger.warning("experiment config patch-back failed for %s (%s)", experiment_id, action,
                       exc_info=True)


def _patch_experiment_config_tiling(experiment_id: str, tiling_cfg: dict, *,
                                    replace: bool = False,
                                    train_native_size: list | None = None) -> None:
    """Patch the effective tiling geometry into the durable experiment record. ``replace`` swaps
    the tiling record wholesale instead of merging: an untiled run's record must not keep a
    stale requested ``tile_size`` a merge would leave behind. ``train_native_size``, when
    stamped, lands beside it."""
    def mutate(data_cfg: dict) -> None:
        if replace:
            data_cfg["tiling"] = dict(tiling_cfg)
        else:
            data_cfg.setdefault("tiling", {}).update(tiling_cfg)
        if train_native_size is not None:
            data_cfg["train_native_size"] = list(train_native_size)

    _patch_experiment_config(experiment_id, "patch_experiment_config_tiling", mutate)


def _patch_experiment_config_id_map(experiment_id: str, subject: str, attribute: str | None,
                                    id_map: dict) -> None:
    """Patch this run's resolved name->id map into the durable experiment record. Called from
    ``run()`` right after the dataset is built (mirroring where the tiling patch fires), though
    unlike tile geometry this fact is a pure function of ``data_cfg`` and would be resolvable
    before the build too, the call site is chosen for symmetry with the tiling patch, not because
    the dataset build is a precondition for it."""
    def mutate(data_cfg: dict) -> None:
        data_cfg["subject"] = subject
        data_cfg["attribute"] = attribute
        data_cfg["id_map"] = dict(id_map)

    _patch_experiment_config(experiment_id, "patch_experiment_config_id_map", mutate)


def _is_manifest_bound_split(split_cfg: object) -> bool:
    """Whether a run's ``data.split`` block is a manifest-bound run's resolved block, the one
    shape :func:`_patch_experiment_config_split` exists for: a spatial or auto-split run's own
    resolved block carries per-region/per-stem member identities that stay out of the durable
    config, so only ``manifest_binding``'s presence qualifies."""
    return isinstance(split_cfg, dict) and "manifest_binding" in split_cfg


def _patch_experiment_config_split(experiment_id: str, split_cfg: dict) -> None:
    """Merge this run's resolved split policy into the durable experiment record. A binding to a
    named split manifest (``data.split.manifest_binding``) is what this exists for:
    ``launch_config.json``, written before the child exists, never carries it, so the durable
    record is the only other place a reviewer can see that a recorded partition, not a drawn
    one, governed the run."""
    def mutate(data_cfg: dict) -> None:
        data_cfg.setdefault("split", {}).update(split_cfg)

    _patch_experiment_config(experiment_id, "patch_experiment_config_split", mutate)


def _resolve_run_id_map(task: str, data_cfg: dict) -> tuple[str, str | None, dict] | None:
    """This run's resolved name->id map, or ``None`` when there is nothing to
    record. Returns ``(subject, attribute, id_map)``.

    Resolved independently of the built dataset object's own attributes:
    ``DetectionDataset`` only self-populates ``.id_map`` on its own direct-json build path; the
    COCO-assembled ``auto_val`` default and ``TiledDetectionDataset`` both build through a
    different internal path and expose neither ``.id_map`` nor ``.subject`` at all, so a
    ``getattr``-off-the-dataset read was silently a no-op on the default and every tiled run.
    ``assign_class_ids`` is a pure function of
    ``(registry, subject, attribute)``, "same registry + scope -> identical map, every call"
    (``class_registry.py``), so re-resolving it here from ``data_cfg``'s own
    subject/attribute/labels_dir, the same inputs ``auto_train_val`` already resolved it from
    internally (``training_tools.py``'s own COCO-assembly branch calls this exact function),
    reproduces the identical map without depending on which internal dataset shape got built.

    ``None`` when ``task`` isn't detection/instance_seg, no ``subject`` is configured, the run
    trains from a pre-built COCO source (``coco_json``/``label_format="coco"``) or a bespoke
    ``dataset_source`` (neither route's targets are guaranteed to come
    from this ``(labels_dir, subject, attribute)`` triple at all, a COCO file's own category ids
    can be authored in any order, and a bespoke builder owns its class space entirely, so
    re-deriving here could stamp a map that is the wrong id space for what the run actually
    trained on, exactly the class of error class-aware admission exists to prevent; ``build_dataset``
    itself only calls ``resolve_registry_id_map`` on the same predicate, datasets.py's own
    ``has_coco``/``dataset_source`` branch), or the one legitimate degraded case
    ``resolve_registry_id_map`` itself names (an attribute scope with no ``classes.json`` for this
    labels dir), honest: no map recorded, decode falls through to its own live-registry
    re-derivation.
    """
    if task not in ("detection", "instance_seg") or not data_cfg.get("subject"):
        return None
    from tcip_mcp.pipelines.data.label_queries import targets_registry_derived

    if not targets_registry_derived(data_cfg):
        return None
    from tcip_mcp.pipelines.data.label_queries import resolve_registry_id_map

    subject = data_cfg["subject"]
    attribute = data_cfg.get("attribute")
    try:
        _reg, id_map = resolve_registry_id_map(data_cfg.get("labels_dir", ""), subject, attribute)
    except ValueError:
        return None
    return (subject, attribute, id_map) if id_map else None


def run(run_id: str, experiment_id: str, output_dir: str, resume_from: str) -> None:
    """The training body, identical in substance to running synchronously in-process, just
    executing in this dedicated process instead."""
    from tcip_mcp.pipelines.raster_source import configure_gdal_cache

    # This is its own process entry point: without this the whole run reads through GDAL's
    # stock cache default instead of the platform budget the server/backend entry points set.
    configure_gdal_cache()
    # Its own process entry point, so it binds its own storage backend too.
    from tcip_store import store
    from tcip_store.binding import bind_default

    bind_default()

    from tcip_mcp.experiments import ExperimentTerminal

    try:
        ctx = _prepare_run_context(run_id, experiment_id, output_dir, resume_from, store)
    except ExperimentTerminal:
        # Already audited, and the record already terminal, by the raiser itself.
        raise
    except Exception as exc:
        # No run_training_envelope has opened its own "training_run" audit event yet, so
        # without this the record stays running and the crash goes unaudited.
        logger.exception("Pre-training setup failed for run %s: %s", run_id, exc)
        try:
            from tcip_mcp.audit import record_event
            from tcip_mcp.experiments import update_status

            update_status(experiment_id, "failed", error=str(exc))
            record_event("training_run", {"run_id": run_id, "experiment_id": experiment_id},
                        status="failed")
        except Exception:
            logger.warning("could not reconcile run %s to failed after its own setup crash",
                           run_id, exc_info=True)
        raise

    from tcip_mcp.pipelines.training.envelope import run_training_envelope

    run_training_envelope(ctx)


def _prepare_run_context(run_id: str, experiment_id: str, output_dir: str, resume_from: str,
                         store: Any) -> "TrainContext":
    """Build this run's ``TrainContext``: read the launch config, build the datasets and loaders,
    patch the durable experiment record's own provenance, and persist the split manifest.

    Isolated from :func:`run` so a crash anywhere here is reconciled to ``failed`` with its own
    ``training_run`` audit event, through ``run``'s own except clause, before it crashes the
    subprocess with its original traceback.
    """
    from tcip_mcp.pipelines.training.generic_trainer import (
        seeded_loader_kwargs, stamp_effective_data_geometry,
    )
    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.run_registry import attach_run
    from tcip_mcp.pipelines.data.split_construction import (
        auto_train_val, dataset_identity, persist_split_manifest,
    )
    from tcip_mcp.pipelines.model_build import MODEL_SOURCE_KEY
    from tcip_mcp.tools.training_tools import launch_config_key

    config = store.read(launch_config_key(output_dir))
    run_obj = attach_run(run_id, config, output_dir)

    model_source = config.get(MODEL_SOURCE_KEY, {})
    # setdefault, not get: the geometry stamp below mutates this dict and must land in config.
    data_cfg = config.setdefault("data", {})
    train_cfg = config.get("training", {})
    # Task drives collate + measurement routing: the bespoke model_source declares it, falling
    # back to the data section.
    task = model_source.get("task") or data_cfg.get("task", "detection")

    aug_config = config.get("augmentation", {})
    transforms = None
    if aug_config:
        from tcip_mcp.pipelines.data.augmentations import build_augmentation
        transforms = build_augmentation(aug_config)

    train_ds, val_ds, label_digests = auto_train_val(task, data_cfg, transforms)

    split_cfg = data_cfg.get("split")
    if _is_manifest_bound_split(split_cfg):
        _patch_experiment_config_split(experiment_id, split_cfg)

    # Stamp this run's resolved name->id map onto config["data"] in place, data_cfg
    # is the same dict object, so this lands on the checkpoint (generic_trainer persists run.config
    # into every checkpoint; GenericPredictor reads it back as predictor.config) as well as the
    # durable experiment record (_patch_experiment_config_id_map). Decode/record at inference time
    # (inference_tools.py::run_inference) then prefers this recorded map over re-deriving from the
    # inference dataset's live registry, so a classes.json whose declared attribute-value order
    # changes between train and inference can't silently mis-decode. See _resolve_run_id_map's own
    # docstring for why this is resolved independently of train_ds's own attributes.
    _resolved = _resolve_run_id_map(task, data_cfg)
    if _resolved is not None:
        _run_subject, _run_attribute, _run_id_map = _resolved
        data_cfg["id_map"] = dict(_run_id_map)
        _patch_experiment_config_id_map(experiment_id, _run_subject, _run_attribute, _run_id_map)

    # The effective input geometry is only knowable once the dataset is actually built, so it
    # is stamped here (into the config every checkpoint embeds) and mirrored to the experiment.
    stamped = stamp_effective_data_geometry(data_cfg, train_ds)
    _patch_experiment_config_tiling(experiment_id, stamped["tiling"],
                                    replace=stamped["tiling_replaced"],
                                    train_native_size=stamped["train_native_size"])

    from tcip_mcp.pipelines.data.samplers import build_sampler
    from torch.utils.data import DataLoader

    batch_size = train_cfg.get("batch_size", 2)
    num_workers = train_cfg.get("num_workers", 0)
    # Seeds the loader's shuffle/worker RNG from the run's own seed and scales each worker's
    # own GDAL cache share by num_workers.
    loader_kwargs = seeded_loader_kwargs(config.get("seed"), num_workers=num_workers)
    # Built after the loader context is known: read order depends on the worker regime too.
    sampler = build_sampler(config.get("sampler", "random"), train_ds,
                            num_workers=num_workers, batch_size=batch_size)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(sampler is None), sampler=sampler,
        collate_fn=task_collate(task),
        num_workers=num_workers,
        **loader_kwargs,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            shuffle=False,
            collate_fn=task_collate(task),
            num_workers=num_workers,
            **loader_kwargs,
        )
    if val_loader is None and task in ("detection", "instance_seg"):
        logger.warning(
            "No validation loader for %s run %s: best-model selection and early "
            "stopping will fall back to training loss (no val mAP/composite). "
            "Provide a val split (data.val_images_dir) or enable auto_val.",
            task, run_id,
        )

    # The dataset identity this run trains on, recomputed here (recompute-on-read is this fact's
    # own stated authority) rather than threaded across the process boundary; same deterministic
    # result the parent's own copy (used for the lineage record) already produced.
    ds_id, ds_fp = dataset_identity(data_cfg)
    persist_split_manifest(experiment_id, train_ds, val_ds, data_cfg,
                           dataset_id=ds_id, dataset_fingerprint=ds_fp,
                           label_digests=label_digests)

    from tcip_mcp.pipelines.training.envelope import TrainContext

    return TrainContext(
        run=run_obj, train_loader=train_loader, val_loader=val_loader,
        task=task, resume_from=resume_from, experiment_id=experiment_id,
    )


def main() -> None:
    args = _parse_args()
    run(args.run_id, args.experiment_id, args.output_dir, args.resume_from)


if __name__ == "__main__":
    main()
