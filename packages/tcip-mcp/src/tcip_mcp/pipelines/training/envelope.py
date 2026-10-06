"""The training envelope around any training body, the default trainer or an agent's custom
``train(ctx)``, and ``TrainContext``.

``TrainContext`` hands the training code the craft library (data / model / optim / eval utils) plus
the envelope-owned sinks (``log_metrics`` / ``save_checkpoint`` / ``record_artifact`` /
``should_cancel`` / ``tb`` / ``set_final_weights`` / ``report_objective``).

When no ``training_source`` is set, ``ctx.default_train()`` runs ``generic_trainer.train()``.
Every sink writes into the run's own directory, ``run.output_dir``, and refuses a run whose final
status is written (``experiments.require_open``).
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tcip_store import stored_number

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "artifacts"
"""The run-directory subdirectory a bespoke loop's recorded artifacts are copied into."""


@dataclass
class TrainContext:
    """The handle the envelope passes to a training body (default or custom).

    Carries the prebuilt loaders and run state, exposes the craft library as thin passthroughs,
    and owns the sinks a training body records metrics, checkpoints and artifacts through.
    """

    run: Any                      # TrainRun
    train_loader: Any
    val_loader: Any | None = None
    resume_from: str = ""
    epoch_hook: Any = None        # (epoch, metrics) -> None, fired for every metrics row
    _epoch: int = 0
    _tb: Any = None

    # ---- config / reproducibility ----
    @property
    def config(self) -> dict:
        return self.run.config

    @property
    def run_dir(self) -> Path:
        """The run's own directory."""
        return Path(self.run.output_dir)

    @property
    def task(self) -> str:
        """The task this run's config names (:func:`~tcip_mcp.pipelines.model_build.run_task`)."""
        from tcip_mcp.pipelines.model_build import run_task

        return run_task(self.config)

    @property
    def seed(self) -> Any:
        return self.config.get("seed")

    @property
    def device(self) -> Any:
        import torch

        return torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

    def set_seed(self, seed: int | None = None, deterministic: bool = False) -> None:
        from tcip_mcp.pipelines.training.generic_trainer import set_seed

        eff = self.seed if seed is None else seed
        if eff is not None:
            set_seed(int(eff), deterministic=deterministic)

    # ---- model ----
    def build_model(self) -> Any:
        """This run's model, at the width and count its config records
        (:func:`~tcip_mcp.pipelines.model_build.recorded_model_dims`)."""
        from tcip_mcp.pipelines.model_build import build_model, recorded_model_dims

        return build_model(self.config, recorded_model_dims(self.config))

    def _contract_args(self, **overrides: Any) -> dict:
        """What the smoke runs against: the dims this run resolved, or one batch off its own train
        loader when those dims cannot shape a batch (``no_batch_reason``).
        """
        from tcip_mcp.pipelines.model_build import recorded_model_dims, resolve_contract_dims
        from tcip_mcp.pipelines.model_contract import no_batch_reason

        dims = resolve_contract_dims(self.config, self.task, recorded_model_dims(self.config))
        if no_batch_reason(self.task, dims, None) is not None:
            batch = next(iter(self.train_loader or []), None)
            if batch is not None:
                return {"sample_batch": batch, **overrides}
        return {"dims": dims, **overrides}

    def check_contract(self, model: Any = None, **overrides: Any) -> dict:
        """Run the measurement-boundary contract on ``model`` (built if omitted) at the run's
        resolved dims, so a hand-rolled ``train(ctx)`` can self-prove before the full loop."""
        from tcip_mcp.pipelines.model_contract import check_model_contract

        return check_model_contract(
            model or self.build_model(), self.task, **self._contract_args(**overrides)
        )

    def overfit_check(self, model: Any = None, **overrides: Any) -> dict:
        """Voluntary diagnostic: drive a few steps on one tiny batch and confirm the loss falls,
        the cheap proof a from-scratch model actually learns. Non-gating."""
        from tcip_mcp.pipelines.model_contract import overfit_check

        return overfit_check(
            model or self.build_model(), self.task, **self._contract_args(**overrides)
        )

    # ---- the default trainer: one optional convenience ----
    def default_train(self) -> Any:
        """Run the default policy (progressive unfreeze / differential-LR / AMP+accum /
        selection+early-stop / checkpoint cadence)."""
        from tcip_mcp.pipelines.training.generic_trainer import train

        return train(self.run, self.train_loader, self.val_loader,
                     epoch_callback=self._epoch_sink, resume_from=self.resume_from)

    # ---- craft library passthroughs (compose, don't reinvent) ----
    def build_dataset(self, task: str | None = None, *, samples: Any,
                      sizes: "Mapping[str, int] | None" = None, **kwargs: Any) -> Any:
        """The factory, over the samples you were handed. ``sizes`` and ``scope`` unstated are
        the ones this run's data config records, so a loader you build here reads at its width
        and count whichever subset of samples it holds."""
        from tcip_mcp.pipelines.data.datasets import build_dataset, stated_sizes
        from tcip_mcp.pipelines.data.selection import ClassScope

        resolved_task = task or self.task
        data_cfg = self.config.get("data") or {}
        if sizes is None:
            sizes = stated_sizes(data_cfg)
        kwargs.setdefault("scope", ClassScope.of(data_cfg))
        return build_dataset(resolved_task, samples=samples, sizes=sizes, **kwargs)

    def tiled_dataset(self, base: Any, **kwargs: Any) -> Any:
        """Wrap a detection dataset in the native-resolution tiler (same derived sliver cutoff
        the default path uses); ``kwargs``: tile_size / overlap / sliver_frac / dedup_iou /
        skip_empty."""
        from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset

        return TiledDetectionDataset(base, **kwargs)

    def task_collate(self, task: str | None = None) -> Any:
        from tcip_mcp.pipelines.training.collation import task_collate

        return task_collate(task or self.task)

    def build_sampler(self, name: str, dataset: Any, *, num_workers: int | None = None,
                      batch_size: int | None = None) -> Any:
        from tcip_mcp.pipelines.data.samplers import build_sampler

        return build_sampler(name, dataset, num_workers=num_workers, batch_size=batch_size)

    def build_augmentation(self, cfg: dict) -> Any:
        from tcip_mcp.pipelines.data.augmentations import build_augmentation

        return build_augmentation(cfg)

    def compute_class_weights(self, *args: Any, **kwargs: Any) -> Any:
        from tcip_mcp.pipelines.components.losses import compute_class_weights

        return compute_class_weights(*args, **kwargs)

    def build_optimizer(self, *args: Any, **kwargs: Any) -> Any:
        from tcip_mcp.pipelines.training.optimizer_factory import build_optimizer

        return build_optimizer(*args, **kwargs)

    def build_scheduler(self, optimizer: Any, config: dict, epochs: int) -> Any:
        from tcip_mcp.pipelines.training.generic_trainer import _build_scheduler

        return _build_scheduler(optimizer, config, epochs)

    def apply_stage_freeze(self, model: Any, freeze_to: int, *, prev_trainable: int | None = None,
                           enforce_monotonic: bool = True) -> int:
        """Apply a stage's progressive-unfreeze policy (+ the monotonic guard) and return the new
        trainable-param count."""
        from tcip_mcp.pipelines.training.generic_trainer import apply_stage_freeze

        return apply_stage_freeze(model, freeze_to, prev_trainable=prev_trainable,
                                  enforce_monotonic=enforce_monotonic)

    def compute_lr_scale(self, *args: Any, **kwargs: Any) -> Any:
        from tcip_mcp.pipelines.training.optimizer_factory import compute_lr_scale

        return compute_lr_scale(*args, **kwargs)

    def snapshot_optimizer_state(self, *args: Any, **kwargs: Any) -> Any:
        from tcip_mcp.pipelines.training.optimizer_factory import snapshot_optimizer_state

        return snapshot_optimizer_state(*args, **kwargs)

    def restore_optimizer_state(self, *args: Any, **kwargs: Any) -> Any:
        from tcip_mcp.pipelines.training.optimizer_factory import restore_optimizer_state

        return restore_optimizer_state(*args, **kwargs)

    def evaluate(self, model: Any, loader: Any = None, **kwargs: Any) -> Any:
        from tcip_mcp.pipelines.model_build import recorded_model_dims
        from tcip_mcp.pipelines.training.evaluation import evaluate

        return evaluate(model, self.val_loader if loader is None else loader,
                        self.device, self.task, dims=recorded_model_dims(self.config), **kwargs)

    # ---- measurement primitives (compose for dimensional traits) ----
    def mask_geometry(self, *args: Any, **kwargs: Any) -> Any:
        from tcip_mcp.pipelines.measurement import mask_geometry

        return mask_geometry(*args, **kwargs)

    def instance_geometries(self, *args: Any, **kwargs: Any) -> Any:
        from tcip_mcp.pipelines.measurement import instance_geometries

        return instance_geometries(*args, **kwargs)

    # ---- envelope-owned sinks ----
    def _epoch_sink(self, epoch: int, metrics: dict) -> None:
        """Append one epoch's metrics to the run's ``metrics.jsonl`` in their stored form, stamped
        with ``epoch`` and the instant, after firing ``epoch_hook`` if attached with the metrics
        the body produced, so a diverged loss keeps comparing as the worst one. Refuses
        (``ValueError``) metrics carrying either stamp's key; a row JSON cannot hold raises,
        naming the field.
        """
        from tcip_mcp.audit import now_iso
        from tcip_mcp.experiments import (
            EPOCH_KEY, METRICS_FILE, TIMESTAMP_KEY, append_row, require_open,
        )
        from tcip_mcp.pipelines.training.generic_trainer import _checkpoint_metrics

        stamped = sorted({EPOCH_KEY, TIMESTAMP_KEY} & metrics.keys())
        if stamped:
            raise ValueError(
                f"the metrics carry {stamped}, which the metrics log stamps on each row itself; "
                "pass the epoch as the sink's own argument.")
        require_open(self.run_dir)
        self._epoch = epoch
        if self.epoch_hook is not None:
            self.epoch_hook(epoch, metrics)
        append_row(self.run_dir / METRICS_FILE,
                   {**_checkpoint_metrics(metrics), EPOCH_KEY: epoch, TIMESTAMP_KEY: now_iso()})

    def set_final_weights(self, tag: str) -> None:
        """Declare the checkpoint this body saved under ``tag`` (:meth:`save_checkpoint`) the
        run's deliverable. Unset, the run's ``model_best`` or else ``model_final`` is the one
        declared once the body returns. Refuses (``ValueError``) a tag this body saved nothing
        under."""
        from tcip_mcp.experiments import require_open

        require_open(self.run_dir)
        if tag not in self.run.saved:
            raise ValueError(f"run {self.run.id} saved no checkpoint under {tag!r}.")
        self.run.deliverable = self.run.saved[tag]

    def report_objective(self, value: float) -> None:
        """Record ``value`` as a metrics row stamping it the ``selection`` under the run's
        objective, at the last epoch logged (:meth:`_epoch_sink`)."""
        self._epoch_sink(self._epoch, {"selection": float(value),
                                       "selection_metric": self.run.objective["selection_metric"]})

    def log_metrics(self, epoch: int, metrics: dict) -> None:
        """Custom-loop metric sink: the run's own metrics log plus TensorBoard."""
        from tcip_store import scalar_number

        self._epoch_sink(epoch, metrics)
        if self.tb is not None:
            for k, v in metrics.items():
                if scalar_number(v):
                    self.tb.add_scalar(k, v, epoch)
            self.tb.flush()

    def save_checkpoint(self, state: dict, tag: str = "checkpoint") -> str:
        """Write ``state`` once under ``tag``, stamped with this run's ``config``
        without its data locations (``experiments.DATA_PATHS``; the run's own record keeps them),
        record it in ``run.saved``, and return the path written. A tag already written refuses
        with ``FileExistsError``. A ``metrics`` key in ``state`` is the deliverable's metrics,
        sourced ``training_source``.

        Refuses (``ValueError``) a ``state`` carrying a ``config`` key: the checkpoint's
        ``config`` is always this run's own.
        """
        from tcip_mcp.experiments import require_open
        from tcip_mcp.pipelines.model_build import CONFIG_KEY

        require_open(self.run_dir)
        if CONFIG_KEY in state:
            raise ValueError(
                "ctx.save_checkpoint: state carries a 'config' key, reserved for this run's own "
                "launch config, the record every publishing door reads this run's scope from; "
                "name a bespoke loop's own field something else."
            )
        import copy

        from tcip_mcp.experiments import DATA_PATHS
        from tcip_mcp.pipelines.training.generic_trainer import checkpoint_path, write_checkpoint

        config = copy.deepcopy(self.config)
        for field in DATA_PATHS:
            *parents, leaf = (step for step in field if step != "[]")
            node = config.get("data") or {}
            for parent in parents:
                node = node.get(parent) or {}
            node.pop(leaf, None)
        self.run.saved[tag] = write_checkpoint({**state, CONFIG_KEY: config},
                                               checkpoint_path(self.run_dir, tag))
        return str(self.run.saved[tag])

    def record_artifact(self, name: str, path: str) -> None:
        """Copy the file at ``path`` into the run's ``artifacts/`` directory under ``name``,
        refusing (``FileExistsError``) a name already recorded and (``BadKey``) one that is not a
        single file name.
        """
        from tcip_mcp.experiments import require_open, run_name

        require_open(self.run_dir)

        destination = self.run_dir / ARTIFACTS_DIR / run_name(name)
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"artifact {name!r} is already recorded for run {self.run.id}")
        shutil.copyfile(path, destination)

    def should_cancel(self) -> bool:
        return self.run.should_cancel()

    @property
    def tb(self) -> Any:
        if self._tb is None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                from tcip_mcp.experiments import TENSORBOARD_DIR

                self._tb = SummaryWriter(log_dir=str(self.run_dir / TENSORBOARD_DIR))
            except Exception:  # noqa: BLE001
                self._tb = None
        return self._tb


def dispatch_train_body(ctx: TrainContext) -> None:
    """Run the training body, an agent's ``training_source`` if set, else ``ctx.default_train()``,
    then declare the checkpoint the body saved under ``model_best``, else ``model_final``
    (``run.saved``), the deliverable when the body declared none.
    """
    run = ctx.run
    from tcip_mcp.experiments import FINAL_STATES
    from tcip_mcp.pipelines.model_build import TRAINING_SOURCE_KEY

    training_source = run.config.get(TRAINING_SOURCE_KEY)
    if training_source:
        from tcip_mcp.pipelines.model_build import _import_dotted

        agent_train = _import_dotted(training_source)
        agent_train(ctx)  # the agent's custom loop drives training through ctx
        if run.status not in FINAL_STATES:
            # A custom loop that never set a final status returned without canceling or raising.
            run.status = "canceled" if run.should_cancel() else "completed"
    else:
        ctx.default_train()  # the default trainer

    if run.deliverable is None:
        run.deliverable = run.saved.get("model_best") or run.saved.get("model_final")


def run_training_envelope(ctx: TrainContext) -> None:
    """Run a training body inside the audited envelope: open a ``training_run`` audit event,
    dispatch the body (:func:`dispatch_train_body`), settle how the run ended
    (:func:`_settle_run`), and close the run (:func:`close_run`), whatever the body did or
    omitted. An opening line that cannot be written fails the run naming why.
    """
    from tcip_mcp.audit import record_event_or_raise

    run = ctx.run
    audit_args = {"experiment_id": run.id, "task": ctx.task}

    t0 = time.monotonic()
    try:
        record_event_or_raise("training_run", audit_args, actor=None, status="running",
                              scope=run.project)
        dispatch_train_body(ctx)
    except Exception as exc:  # noqa: BLE001
        if run.status not in ("failed", "canceled"):
            run.status = "failed"
        run.error = run.error or str(exc)
        logger.exception("Training body failed for %s: %s", run.id, exc)

    checkpoint = _settle_run(ctx)
    run.status, run.error = close_run(
        ctx.run_dir, run.project, run.status or "failed", run.error or None,
        checkpoint=checkpoint,
        arguments={**audit_args, **stored_number("best_metric", run.best_metric)},
        duration_ms=round((time.monotonic() - t0) * 1000, 1))


def close_run(run_dir: Path, project: Path, state: str, error: str | None, *,
              checkpoint: dict | None, arguments: dict[str, Any],
              **extra: Any) -> tuple[str, str | None]:
    """Append the run's closing ``training_run`` line under ``project``, then write its final
    status once and return the state and error it names. A refused append ends the run
    ``failed`` with no checkpoint, its reason named after ``error`` when there is one."""
    from tcip_mcp.audit import AuditEntryNotWritten, record_event_or_raise
    from tcip_mcp.experiments import write_final_status

    try:
        record_event_or_raise("training_run", arguments, actor=None, status=state, scope=project,
                              **extra)
    except AuditEntryNotWritten as exc:
        state, checkpoint = "failed", None
        error = f"{error}; {exc}" if error else str(exc)
    write_final_status(run_dir, state, error, checkpoint=checkpoint)
    return state, error


def _settle_run(ctx: TrainContext) -> dict | None:
    """Settle how the run ended and return the checkpoint its final status names: for a completed
    run its deliverable (``run.deliverable``), named inside the run's directory with the sha256
    the verified checkpoint reader admitted (``model_registry.admitted_digest``), else ``None``. A
    run the wall clock stopped ends ``failed`` naming it; a completed run with no deliverable, or
    one that cannot be read or that the verified reader refuses, ends ``failed`` naming why.
    """
    from tcip_mcp.model_registry import admitted_digest

    run = ctx.run
    if run.wall_clock_exceeded and run.status != "failed":
        run.status, run.error = "failed", "exceeded max_wall_clock_seconds"
    checkpoint = None
    if run.status == "completed" and run.deliverable is None:
        run.status = "failed"
        run.error = "training completed but saved no final weights"
    elif run.status == "completed":
        try:
            checkpoint = {"path": str(run.deliverable),
                          "sha256": admitted_digest(run.deliverable)}
        except (OSError, ValueError) as exc:
            run.status, run.error = "failed", f"final weights could not be admitted: {exc}"
    return checkpoint
