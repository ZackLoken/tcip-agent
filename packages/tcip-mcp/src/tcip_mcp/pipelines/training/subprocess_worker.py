"""The process entry point of one run's body: ``python -m
tcip_mcp.pipelines.training.subprocess_worker --run-dir <dir>``, everything else it reads being in
the directory's ``run.json``.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tcip_mcp.experiments import RunObservation
    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.pipelines.training.run_registry import TrainRun

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def prepare_run_context(
    observation: "RunObservation", *, origin: str = "training", epoch_hook: Any = None,
) -> "TrainContext":
    """Build the ``TrainContext`` of the observed run from what its launch record resolved,
    resolving nothing again: its config with the resolved data section, its objective, and its
    datasets and loaders (:func:`~tcip_mcp.pipelines.data.split_construction.recorded_datasets`).
    ``origin`` and ``epoch_hook`` are the context's own.
    """
    from tcip_mcp.pipelines.data.split_construction import recorded_datasets
    from tcip_mcp.pipelines.model_build import run_task
    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.pipelines.training.generic_trainer import run_loaders, run_transforms
    from tcip_mcp.pipelines.training.run_registry import TrainRun

    run_dir = observation.directory
    run_record = observation.record
    resolved = run_record["resolved"]
    config = {**run_record["config"], "data": resolved["data"]}
    run_obj = TrainRun(id=run_dir.name, config=config, objective=resolved["objective"],
                       output_dir=str(run_dir), origin=origin)
    if run_record["max_wall_clock_seconds"] is not None:
        run_obj.deadline = time.time() + run_record["max_wall_clock_seconds"]

    task = run_task(config)
    train_ds, val_ds = recorded_datasets(task, resolved["data"], resolved["partition"],
                                         run_transforms(config))
    train_loader, val_loader = run_loaders(config, task, train_ds, val_ds, config.get("seed"))
    if val_loader is None and task in ("detection", "instance_seg"):
        logger.warning(
            "No validation loader for %s run %s: best-model selection and early stopping will "
            "fall back to training loss. Bind a selection through data.split.selection_dir, or "
            "enable auto_val.", task, run_dir.name,
        )
    return TrainContext(
        run=run_obj, train_loader=train_loader, val_loader=val_loader,
        resume_from=run_record["resume_from"] or "", epoch_hook=epoch_hook,
    )


def run_directory(
    run_dir: Path, *, origin: str = "training", epoch_hook: Any = None,
) -> "TrainRun":
    """Run the run whose directory is ``run_dir`` to its final status and return it: refuse one
    that has ended before anything is written (``experiments.require_open``), observe it once,
    keep its heartbeat, build its context from that observation (:func:`prepare_run_context`,
    which takes the other arguments), and run its body inside the envelope. A failure before the
    envelope opens writes the final status ``failed`` naming it, leaves a ``training_run`` audit
    line, and propagates.
    """
    from tcip_mcp.experiments import keep_heartbeat, observe, require_open, write_final_status
    from tcip_mcp.pipelines.training.envelope import run_training_envelope

    require_open(run_dir)
    observation = observe(run_dir)
    stop = keep_heartbeat(run_dir)
    try:
        try:
            ctx = prepare_run_context(observation, origin=origin, epoch_hook=epoch_hook)
        except Exception as exc:
            logger.exception("Pre-training setup failed for run %s: %s", run_dir.name, exc)
            from tcip_mcp.audit import record_event

            write_final_status(run_dir, "failed", str(exc), checkpoint=None)
            record_event("training_run", {"experiment_id": run_dir.name}, status="failed")
            raise
        run_training_envelope(ctx)
        return ctx.run
    finally:
        stop.set()


def main() -> None:
    from tcip_mcp.pipelines.raster_source import configure_gdal_cache

    # Its own process entry point: the platform's GDAL cache budget, and its own backend binding.
    configure_gdal_cache()
    from tcip_store.binding import bind_default

    bind_default()
    run_directory(Path(_parse_args().run_dir))


if __name__ == "__main__":
    main()
