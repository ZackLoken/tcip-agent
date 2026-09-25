"""The subprocess entry point ``launch_training`` spawns to run one bespoke training run's body,
dataset/loader construction and the audited envelope, ``run_training_envelope()``, in an isolated
OS process, so a leak/OOM/hang in one run can't take down the launching process or any other
concurrent run's process.

Invoked as ``python -m tcip_mcp.pipelines.training.subprocess_worker --experiment-id ...
--output-dir ... --resume-from ...``. The bootstrap config is read from the run's own output
directory.
"""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tcip_mcp.pipelines.training.envelope import TrainContext

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume-from", default="")
    return p.parse_args()


def _is_selection_bound_split(split_cfg: object) -> bool:
    """Whether a run's ``data.split`` block is a selection-bound run's resolved block: only then is
    it recorded in the durable config, since a drawn or spatial run's resolved block carries
    per-member identities that stay out of it."""
    return isinstance(split_cfg, dict) and "selection_binding" in split_cfg


def _mirror_data_section(experiment_id: str, data_cfg: dict) -> None:
    """Write this run's resolved data section whole into the durable experiment record's own
    ``config.json``: its admitted class space, its effective tiling and frame size, and its
    ``split`` block when the run is bound to a selection (the record's own ``split`` otherwise).

    No experiment record to write is no write. A terminal record refuses with
    :class:`~tcip_mcp.experiments.ExperimentTerminal`, and every write failure raises.
    """
    from tcip_store import store

    from tcip_mcp.experiments import config_key, rewrite_live_member

    key = config_key(experiment_id)
    if not store.exists(key):
        return
    bound = _is_selection_bound_split(data_cfg.get("split"))

    def update(cfg: dict) -> dict:
        section = dict(data_cfg)
        recorded = cfg.get("data") or {}
        if not bound:
            section.pop("split", None)
            if "split" in recorded:
                section["split"] = recorded["split"]
        return {**cfg, "data": section}

    rewrite_live_member(experiment_id, key, "mirror_data_section", update)


def run(experiment_id: str, output_dir: str, resume_from: str) -> None:
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
        ctx = _prepare_run_context(experiment_id, output_dir, resume_from, store)
    except ExperimentTerminal:
        raise  # the record is already terminal: nothing to reconcile, and a refusal has no line
    except Exception as exc:
        # No run_training_envelope has opened its own "training_run" audit event yet, so
        # without this the record stays running and the crash goes unaudited.
        logger.exception("Pre-training setup failed for run %s: %s", experiment_id, exc)
        try:
            from tcip_mcp.audit import record_event
            from tcip_mcp.experiments import update_status

            update_status(experiment_id, "failed", error=str(exc))
            record_event("training_run", {"experiment_id": experiment_id}, status="failed")
        except Exception:
            logger.warning("could not reconcile run %s to failed after its own setup crash",
                           experiment_id, exc_info=True)
        raise

    from tcip_mcp.pipelines.training.envelope import run_training_envelope

    run_training_envelope(ctx)


def _prepare_run_context(experiment_id: str, output_dir: str, resume_from: str,
                         store: Any) -> "TrainContext":
    """Build this run's ``TrainContext``: read the launch config, build the datasets and loaders,
    patch the durable experiment record's own provenance, and persist the run's partition.

    Isolated from :func:`run` so a crash anywhere here is reconciled to ``failed`` with its own
    ``training_run`` audit event, through ``run``'s own except clause, before it crashes the
    subprocess with its original traceback.
    """
    from tcip_mcp.pipelines.training.generic_trainer import (
        run_loaders, run_transforms, stamp_effective_data_geometry,
    )
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.experiments import lineage_key, read_member
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_run_partition
    from tcip_mcp.pipelines.model_build import run_task
    from tcip_mcp.tools.training_tools import launch_config_key

    config = store.read(launch_config_key(output_dir))
    run_obj = create_run(config, output_dir, id=experiment_id)

    # setdefault, not get: the geometry stamp below mutates this dict and must land in config.
    data_cfg = config.setdefault("data", {})
    task = run_task(config)
    train_ds, val_ds, partition = auto_train_val(task, data_cfg, run_transforms(config))

    # The effective input geometry is only knowable once the dataset is actually built, so it is
    # stamped here (into the config every checkpoint embeds) before the section is mirrored.
    stamp_effective_data_geometry(data_cfg, train_ds)
    _mirror_data_section(experiment_id, data_cfg)

    train_loader, val_loader = run_loaders(config, task, train_ds, val_ds, config.get("seed"))
    if val_loader is None and task in ("detection", "instance_seg"):
        logger.warning(
            "No validation loader for %s run %s: best-model selection and early "
            "stopping will fall back to training loss (no val mAP/composite). "
            "Bind a selection through data.split.selection_dir, or enable auto_val.",
            task, experiment_id,
        )

    # The dataset identity the launching process recorded on the run's own lineage.
    lineage = read_member(lineage_key(experiment_id), {})
    persist_run_partition(experiment_id, data_cfg,
                           dataset_id=lineage.get("dataset_id"),
                           dataset_fingerprint=lineage.get("dataset_fingerprint"),
                           partition=partition)

    from tcip_mcp.pipelines.training.envelope import TrainContext

    return TrainContext(
        run=run_obj, train_loader=train_loader, val_loader=val_loader,
        task=task, resume_from=resume_from, experiment_id=experiment_id,
    )


def main() -> None:
    args = _parse_args()
    run(args.experiment_id, args.output_dir, args.resume_from)


if __name__ == "__main__":
    main()
