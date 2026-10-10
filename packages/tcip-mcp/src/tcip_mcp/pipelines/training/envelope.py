"""The training envelope around any training body, the default trainer or an agent's custom
``train(ctx)``, and ``TrainContext``.

``TrainContext`` hands the training code the craft library (data / model / optim / eval utils) plus
the envelope-owned sinks (``log_metrics`` / ``log_batch`` / ``save_checkpoint`` /
``record_artifact`` / ``should_cancel`` / ``tb`` / ``set_final_weights`` / ``report_objective``).

When no ``training_source`` is set, ``ctx.default_train()`` runs ``generic_trainer.train()``.
Every sink writes into the run's own directory, ``run.output_dir``, and refuses a run whose final
status is written (``experiments.require_open``).
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tcip_store import stored_number

if TYPE_CHECKING:
    from tcip_mcp.pipelines.schemas import TrainConfigSchema

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
    epoch_hook: Any = None        # (epoch, metrics) -> None, fired for every epoch row
    _epoch: int = 0
    _tb: Any = None

    @property
    def run_dir(self) -> Path:
        """The run's own directory."""
        return Path(self.run.output_dir)

    @property
    def task(self) -> str:
        """The task this run's validated config names, ``model_source.task``."""
        return self.spec.model_source.task

    @property
    def spec(self) -> TrainConfigSchema:
        """The run's validated config (``TrainRun.spec``); a bespoke loop's own keys are its
        ``model_extra``."""
        return self.run.spec

    @property
    def seed(self) -> Any:
        """The validated config's ``seed``."""
        return self.spec.seed

    @property
    def device(self) -> Any:
        """The device the validated config trains on (``generic_trainer.run_device``)."""
        from tcip_mcp.pipelines.training.generic_trainer import run_device

        return run_device(self.spec)

    def set_seed(self, seed: int | None = None, deterministic: bool = False) -> None:
        from tcip_mcp.pipelines.training.generic_trainer import set_seed

        eff = self.seed if seed is None else seed
        if eff is not None:
            set_seed(int(eff), deterministic=deterministic)

    # ---- model ----
    def build_model(self) -> Any:
        """This run's model, built from its validated ``model_source``
        (:func:`~tcip_mcp.pipelines.model_build.build_from_model_source`) at the width and count
        its config records (:func:`~tcip_mcp.pipelines.model_build.recorded_model_dims`)."""
        from tcip_mcp.pipelines.model_build import build_from_model_source, recorded_model_dims

        return build_from_model_source(self.spec.model_source, self.run.layout,
                                       recorded_model_dims(self.spec))

    def _contract_args(self, **overrides: Any) -> dict:
        """What the smoke runs against: the dims this run resolved, or one batch off its own train
        loader when those dims cannot shape a batch (``no_batch_reason``).
        """
        from tcip_mcp.pipelines.model_build import recorded_model_dims, resolve_contract_dims
        from tcip_mcp.pipelines.model_contract import no_batch_reason

        dims = resolve_contract_dims(self.spec, recorded_model_dims(self.spec))
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
                     epoch_callback=self.log_metrics, batch_callback=self.log_batch,
                     resume_from=self.resume_from)

    # ---- craft library passthroughs (compose, don't reinvent) ----
    def build_dataset(self, *, samples: Any, transforms: Any = None) -> Any:
        """This run's loader over ``samples``, whole, at ``transforms``
        (``split_construction.run_loader``): by the builder its data block's dataset source
        names, or else by the platform's factory at the block's sizes and tiling, under the
        block's class space. A within-image split's region views are this run's own
        ``train_loader`` and ``val_loader``; this builds no view of them."""
        from tcip_mcp.pipelines.data.split_construction import run_loader

        return run_loader(self.task, self.spec.data, self.run.layout)(samples=samples,
                                                                      transforms=transforms)

    def tiled_dataset(self, base: Any, *, tile_size: int, overlap: float, **kwargs: Any) -> Any:
        """Wrap a detection dataset in the native-resolution tiler
        (``datasets.TiledDetectionDataset``) at the lattice you compose it at, ``tile_size`` and
        ``overlap``, both required. ``kwargs``: sliver_frac / dedup_iou / skip_empty /
        keep_regions / transforms."""
        from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset

        return TiledDetectionDataset(base, tile_size=tile_size, overlap=overlap, **kwargs)

    def task_collate(self) -> Any:
        from tcip_mcp.pipelines.training.collation import task_collate

        return task_collate(self.task)

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

    def _stated(self, block: str) -> Any:
        """The run's validated ``block``; refuses (``ValueError``) naming it when the run's
        config, naming its own ``training_source``, states none."""
        value = getattr(self.spec, block)
        if value is None:
            raise ValueError(f"this run's config states no {block!r} block for the loop to build "
                             "from; state one beside training_source")
        return value

    def build_optimizer(self, model: Any) -> Any:
        """The optimizer the run's validated ``optimizer`` block names over ``model``, at the
        block's own rates."""
        from tcip_mcp.pipelines.training.optimizer_factory import build_optimizer

        spec = self._stated("optimizer")
        return build_optimizer(spec, model, backbone_lr=spec.backbone_lr, head_lr=spec.head_lr)

    def build_scheduler(self, optimizer: Any, epochs: int) -> Any:
        """The scheduler the run's validated ``scheduler`` block names, over ``epochs`` epochs."""
        from tcip_mcp.pipelines.training.generic_trainer import _build_scheduler

        return _build_scheduler(optimizer, self._stated("scheduler"), epochs)

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

    def capture_training_state(self, *args: Any, **kwargs: Any) -> Any:
        from tcip_mcp.pipelines.training.optimizer_factory import capture_training_state

        return capture_training_state(*args, **kwargs)

    def restore_training_state(self, *args: Any, **kwargs: Any) -> Any:
        from tcip_mcp.pipelines.training.optimizer_factory import restore_training_state

        return restore_training_state(*args, **kwargs)

    def evaluate(self, model: Any, loader: Any = None, **kwargs: Any) -> Any:
        """``evaluation.evaluate`` of ``model`` over ``loader`` (the run's validation loader when
        omitted), a detector's boxes counted at the run's stated confidence
        (``TrainRun.reads``) and each frame capped at the run's recorded object density;
        ``kwargs`` are ``evaluate``'s other keywords, and one naming ``conf_threshold`` or
        ``density`` is refused by the call (``TypeError``)."""
        from tcip_mcp.pipelines.model_build import recorded_model_dims
        from tcip_mcp.pipelines.training.evaluation import evaluate

        return evaluate(model, self.val_loader if loader is None else loader,
                        self.device, self.task, dims=recorded_model_dims(self.spec),
                        conf_threshold=self.run.reads.conf_threshold,
                        density=self.spec.data.train_object_density, **kwargs)

    # ---- measurement primitives (compose for dimensional traits) ----
    def mask_geometry(self, *args: Any, **kwargs: Any) -> Any:
        from tcip_mcp.pipelines.measurement import mask_geometry

        return mask_geometry(*args, **kwargs)

    def instance_geometries(self, *args: Any, **kwargs: Any) -> Any:
        from tcip_mcp.pipelines.measurement import instance_geometries

        return instance_geometries(*args, **kwargs)

    # ---- envelope-owned sinks ----
    def _metrics_row(self, metrics: dict, stamps: dict) -> dict:
        """``metrics`` in their stored form with ``stamps`` and the instant laid over them, the
        run checked open (``experiments.require_open``). Refuses (``ValueError``) metrics carrying
        a key the log stamps (``experiments.EPOCH_KEY``, ``STEP_KEY``, ``TIMESTAMP_KEY``)."""
        from tcip_mcp.audit import now_iso
        from tcip_mcp.experiments import EPOCH_KEY, STEP_KEY, TIMESTAMP_KEY, require_open
        from tcip_mcp.pipelines.training.generic_trainer import _checkpoint_metrics

        stamped = sorted({EPOCH_KEY, STEP_KEY, TIMESTAMP_KEY} & metrics.keys())
        if stamped:
            raise ValueError(
                f"the metrics carry {stamped}, which the metrics log stamps on each row itself; "
                "pass the epoch and step as the sink's own arguments.")
        require_open(self.run_dir)
        return {**_checkpoint_metrics(metrics), **stamps, TIMESTAMP_KEY: now_iso()}

    def _epoch_sink(self, epoch: int, metrics: dict) -> None:
        """Append one epoch's row (:meth:`_metrics_row`, stamped with ``epoch``) to the run's
        ``metrics.jsonl``, after firing ``epoch_hook`` if attached with the metrics the body
        produced, so a diverged loss keeps comparing as the worst one. A row JSON cannot hold
        raises, naming the field."""
        from tcip_mcp.experiments import EPOCH_KEY, METRICS_FILE, append_row

        row = self._metrics_row(metrics, {EPOCH_KEY: epoch})
        self._epoch = epoch
        if self.epoch_hook is not None:
            self.epoch_hook(epoch, metrics)
        append_row(self.run_dir / METRICS_FILE, row)

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

    def _write_scalars(self, metrics: dict, step: int) -> None:
        """Each scalar of ``metrics`` at ``step`` in the run's TensorBoard (:attr:`tb`), tagged by
        its key."""
        from tcip_store import scalar_number

        for k, v in metrics.items():
            if scalar_number(v):
                self.tb.add_scalar(k, v, step)
        self.tb.flush()

    def log_metrics(self, epoch: int, metrics: dict) -> None:
        """Epoch metric sink: one epoch row in the run's own metrics log plus each scalar at
        ``epoch`` in TensorBoard."""
        self._epoch_sink(epoch, metrics)
        self._write_scalars(metrics, epoch)

    def log_batch(self, step: int, epoch: int, metrics: dict) -> None:
        """Per-batch sink: one per-batch row at ``step`` within ``epoch`` in the run's own metrics
        log (:meth:`_metrics_row`), never an epoch row, so ``epoch_hook`` does not fire; plus each
        scalar at ``step`` in TensorBoard."""
        from tcip_mcp.experiments import EPOCH_KEY, METRICS_FILE, STEP_KEY, append_row

        append_row(self.run_dir / METRICS_FILE,
                   self._metrics_row(metrics, {EPOCH_KEY: epoch, STEP_KEY: step}))
        self._write_scalars(metrics, step)

    def save_checkpoint(self, state: dict, tag: str = "checkpoint") -> str:
        """Write ``state`` once under ``tag``, stamped with this run's config and source snapshot
        (``generic_trainer.checkpoint_stamp``), record it in ``run.saved``, and return the path
        written.
        A tag already written refuses with ``FileExistsError``. A ``metrics`` key in ``state`` is
        the deliverable's metrics, sourced ``training_source``.

        Refuses (``ValueError``) a ``state`` carrying a key the stamp writes: the checkpoint's
        config and snapshot are always this run's own.
        """
        from tcip_mcp.experiments import require_open
        from tcip_mcp.pipelines.training.generic_trainer import (
            checkpoint_path,
            checkpoint_stamp,
            write_checkpoint,
        )

        require_open(self.run_dir)
        stamp = checkpoint_stamp(self.run)
        if reserved := sorted(set(state) & set(stamp)):
            raise ValueError(
                f"ctx.save_checkpoint: state carries {reserved}, reserved for this run's own "
                "launch config, the record every publishing door reads this run's scope from, and "
                "its source snapshot; name a bespoke loop's own field something else."
            )
        self.run.saved[tag] = write_checkpoint({**state, **stamp},
                                               checkpoint_path(self.run_dir, tag))
        return str(self.run.saved[tag])

    def record_artifact(self, name: str, path: str) -> None:
        """Copy the file at ``path`` into the run's ``artifacts/`` directory under ``name``,
        refusing (``FileExistsError``) a name already recorded and (``BadKeyError``) one that is not
        a single file name.
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
        """The run's one TensorBoard writer, over its ``tensorboard`` directory, opened on first
        use and closed by the envelope once the body returns."""
        if self._tb is None:
            from torch.utils.tensorboard import SummaryWriter

            from tcip_mcp.experiments import TENSORBOARD_DIR

            self._tb = SummaryWriter(log_dir=str(self.run_dir / TENSORBOARD_DIR))
        return self._tb


def dispatch_train_body(ctx: TrainContext) -> None:
    """Run the training body, an agent's ``training_source`` if set, else ``ctx.default_train()``,
    then declare the checkpoint the body saved under ``model_best``, else ``model_final``
    (``run.saved``), the deliverable when the body declared none. The run's TensorBoard writer
    (``ctx.tb``) is closed once the body returns or raises.
    """
    run = ctx.run
    from tcip_mcp.experiments import FINAL_STATES

    training_source = run.spec.training_source
    try:
        if training_source:
            from tcip_mcp.pipelines.model_build import import_source_builder

            agent_train = import_source_builder(training_source, run.layout)
            agent_train(ctx)  # the agent's custom loop drives training through ctx
            if run.status not in FINAL_STATES:
                # A custom loop that never set a final status returned without canceling or
                # raising.
                run.status = "canceled" if run.should_cancel() else "completed"
        else:
            ctx.default_train()  # the default trainer
    finally:
        if ctx._tb is not None:
            ctx._tb.close()

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
        run.status_error = run.status_error or str(exc)
        logger.exception("Training body failed for %s: %s", run.id, exc)

    checkpoint = _settle_run(ctx)
    run.status, run.status_error = close_run(
        ctx.run_dir, run.project, run.status or "failed", run.status_error or None,
        checkpoint=checkpoint,
        arguments={**audit_args, **stored_number("best_metric", run.best_metric)},
        duration_ms=round((time.monotonic() - t0) * 1000, 1))


def close_run(run_dir: Path, project: Path, state: str, status_error: str | None, *,
              checkpoint: dict | None, arguments: dict[str, Any],
              **extra: Any) -> tuple[str, str | None]:
    """Append the run's closing ``training_run`` line under ``project``, then write its final
    status once and return the state and ``status_error`` it names. A refused append ends the
    run ``failed`` with no checkpoint, its reason named after ``status_error`` when there is
    one."""
    from tcip_mcp.audit import AuditEntryNotWrittenError, record_event_or_raise
    from tcip_mcp.experiments import write_final_status

    try:
        record_event_or_raise("training_run", arguments, actor=None, status=state, scope=project,
                              **extra)
    except AuditEntryNotWrittenError as exc:
        state, checkpoint = "failed", None
        status_error = f"{status_error}; {exc}" if status_error else str(exc)
    write_final_status(run_dir, state, status_error, checkpoint=checkpoint)
    return state, status_error


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
        run.status, run.status_error = "failed", "exceeded max_wall_clock_seconds"
    checkpoint = None
    if run.status == "completed" and run.deliverable is None:
        run.status = "failed"
        run.status_error = "training completed but saved no final weights"
    elif run.status == "completed":
        try:
            checkpoint = {"path": str(run.deliverable),
                          "sha256": admitted_digest(run.deliverable)}
        except (OSError, ValueError) as exc:
            run.status, run.status_error = (
                "failed", f"final weights could not be admitted: {exc}")
    return checkpoint
