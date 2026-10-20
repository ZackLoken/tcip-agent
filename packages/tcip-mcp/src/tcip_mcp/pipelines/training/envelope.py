"""The training envelope around any training body, and the ``TrainContext`` it hands the body.

A body is the default trainer or an agent's custom ``train(ctx)``. ``TrainContext`` hands the
training code the craft library (data / model / optim / eval utils) plus the envelope-owned sinks
(``log_metrics`` / ``log_batch`` / ``save_checkpoint`` / ``record_artifact`` / ``should_cancel``
/ ``tb`` / ``set_final_weights`` / ``report_objective``).

The recording sinks (``log_metrics``, ``log_batch``, ``save_checkpoint``, ``record_artifact``,
``set_final_weights``, ``report_objective``) refuse a run whose final status is written
(``experiments.require_open``), and what they write lands in the run's own directory,
``run.output_dir``.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Unpack

from tcip_mcp.pipelines import measurement
from tcip_mcp.pipelines.components import losses
from tcip_mcp.pipelines.data import samplers
from tcip_mcp.pipelines.model_build import placement
from tcip_mcp.pipelines.training import generic_trainer, optimizer_factory
from tcip_store import stored_number

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch
    from pydantic import JsonValue
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    from tcip_mcp.pipelines.data.augmentations import Transforms
    from tcip_mcp.pipelines.data.datasets import (
        BaseDataset,
        DocumentDataset,
        TiledDetectionDataset,
    )
    from tcip_mcp.pipelines.data.selection import Sample
    from tcip_mcp.pipelines.model_contract import (
        ContractKeywords,
        ContractReport,
        OverfitKeywords,
        OverfitReport,
    )
    from tcip_mcp.pipelines.schemas import (
        SpatialManifest,
        TrainConfigSchema,
    )
    from tcip_mcp.pipelines.training.collation import Collate
    from tcip_mcp.pipelines.training.generic_trainer import EpochCallback
    from tcip_mcp.pipelines.training.run_registry import TrainRun

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "artifacts"
"""The run-directory subdirectory a bespoke loop's recorded artifacts are copied into."""


@dataclass
class TrainContext:
    """The handle the envelope passes to a training body (default or custom).

    Carries the prebuilt loaders and run state, exposes the craft library as thin passthroughs,
    and owns the sinks a training body records metrics, checkpoints and artifacts through.
    """

    run: TrainRun
    train_loader: Any
    val_loader: Any | None = None
    spatial: SpatialManifest | None = (
        None  # the within-image split the run's partition records
    )
    resume_from: str = ""
    epoch_hook: EpochCallback | None = None
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
        """The run's validated config (``TrainRun.spec``).

        A bespoke loop's own keys are its ``model_extra``.
        """
        return self.run.spec

    @property
    def seed(self) -> int:
        """The validated config's ``seed``."""
        return self.spec.seed

    @property
    def device(self) -> torch.device:
        """The device the validated config's ``device`` places on (``model_build.placement``)."""
        return placement(self.spec.device)

    def set_seed(self) -> None:
        """Seed every random stream at the validated config's ``seed`` and ``deterministic``.

        The seeding is ``generic_trainer.set_seed``'s.
        """
        generic_trainer.set_seed(
            self.spec.seed, deterministic=self.spec.deterministic
        )

    # ---- model ----
    def build_model(self) -> torch.nn.Module:
        """Build this run's model from its validated ``model_source``.

        The build (:func:`~tcip_mcp.pipelines.model_build.build_from_model_source`) runs at the
        width and count its config records
        (:func:`~tcip_mcp.pipelines.model_build.recorded_model_dims`) and the geometry its
        resolution recorded.
        """
        from tcip_mcp.pipelines.model_build import (
            build_from_model_source,
            recorded_model_dims,
        )

        return build_from_model_source(
            self.spec.model_source,
            self.run.layout,
            recorded_model_dims(self.spec),
            self.run.geometry,
        )

    def _contract_args(self) -> ContractKeywords:
        """Return the arguments the contract smoke runs against.

        They are the dims this run resolved, or one batch off its own train loader when those dims
        cannot shape a batch (``no_batch_reason``).
        """
        from tcip_mcp.pipelines.model_build import (
            recorded_model_dims,
            resolve_contract_dims,
        )
        from tcip_mcp.pipelines.model_contract import no_batch_reason

        dims = resolve_contract_dims(
            self.spec, recorded_model_dims(self.spec), self.run.geometry
        )
        if no_batch_reason(self.task, dims, None) is not None:
            batch = next(iter(self.train_loader or []), None)
            if batch is not None:
                return {"sample_batch": batch}
        return {"dims": dims}

    def check_contract(
        self,
        model: torch.nn.Module | None = None,
        **overrides: Unpack[ContractKeywords],
    ) -> ContractReport:
        """Run the measurement-boundary contract on ``model``, built if omitted, on :attr:`device`.

        The contract runs at the arguments :meth:`_contract_args` gives, ``overrides`` laid over
        them.
        """
        from tcip_mcp.pipelines.model_contract import check_model_contract

        return check_model_contract(
            model or self.build_model(),
            self.task,
            device=self.device,
            **(self._contract_args() | overrides),
        )

    def overfit_check(
        self,
        model: torch.nn.Module | None = None,
        **overrides: Unpack[OverfitKeywords],
    ) -> OverfitReport:
        """Drive a few steps on one tiny batch of ``model`` and report whether the loss falls.

        The steps (``model_contract.overfit_check``) run at the arguments :meth:`_contract_args`
        gives, ``overrides`` laid over them, on the run's ``seed``, ``deterministic`` and device.
        """
        from tcip_mcp.pipelines.model_contract import overfit_check

        return overfit_check(
            model or self.build_model(),
            self.task,
            seed=self.spec.seed,
            deterministic=self.spec.deterministic,
            device=self.device,
            **(self._contract_args() | overrides),
        )

    # ---- the default trainer: one optional convenience ----
    def default_train(self) -> TrainRun:
        """Run the default trainer over this run's loaders.

        Its policy is progressive unfreeze / differential-LR / AMP+accum / selection+early-stop /
        checkpoint cadence.
        """
        return generic_trainer.train(
            self.run,
            self.train_loader,
            self.val_loader,
            epoch_callback=self.log_metrics,
            batch_callback=self.log_batch,
            resume_from=self.resume_from,
        )

    # ---- craft library passthroughs (compose, don't reinvent) ----
    def build_dataset(
        self,
        *,
        samples: Sequence[Sample],
        transforms: Transforms | None = None,
    ) -> BaseDataset:
        """Build this run's dataset over ``samples`` at ``transforms``.

        The dataset (``split_construction.run_loader``) is built by the builder its data block's
        dataset source names, or else by the platform's factory at the block's sizes and tiling,
        under the block's class space. On a within-image split run (:attr:`spatial`) it is the
        run's one sample's train view, its tiles inside the recorded train region, and any other
        sample list refuses naming that sample.
        """
        from tcip_mcp.pipelines.data.split_construction import run_loader

        return run_loader(
            self.task, self.spec.data, self.run.layout, self.spatial
        )(samples=samples, transforms=transforms)

    def tiled_dataset(
        self,
        base: DocumentDataset,
        *,
        tile_size: int,
        overlap: float,
        skip_empty: bool,
        sliver_frac: float | None = None,
        transforms: Transforms | None = None,
        keep_regions: Sequence[tuple[int, int, int, int]] | None = None,
    ) -> TiledDetectionDataset:
        """Wrap a detection or instance-segmentation dataset in the native-resolution tiler.

        The tiler (``datasets.TiledDetectionDataset``) runs at the lattice you compose it at,
        ``tile_size`` and ``overlap``, and ``skip_empty``, whether a tile holding no object is
        dropped, each required; ``sliver_frac`` and ``keep_regions`` are the tiler's own, and
        ``transforms`` are admitted over ``base``'s samples first (``datasets.refuse_unchained``).
        """
        from tcip_mcp.pipelines.data.datasets import (
            TiledDetectionDataset,
            refuse_unchained,
        )

        refuse_unchained(
            [base.sample_of(k) for k in base.stems],
            base.expected_channels,
            transforms,
        )
        return TiledDetectionDataset(
            base,
            tile_size=tile_size,
            overlap=overlap,
            skip_empty=skip_empty,
            sliver_frac=sliver_frac,
            transforms=transforms,
            keep_regions=keep_regions,
        )

    def task_collate(self) -> Collate:
        """Return the collate function for this run's task (``collation.task_collate``)."""
        from tcip_mcp.pipelines.training.collation import task_collate

        return task_collate(self.task)

    build_sampler = staticmethod(samplers.build_sampler)

    def build_augmentation(self) -> Transforms:
        """Build the augmentation chain the run's validated ``augmentation`` block states.

        A config stating no block refuses (:meth:`_unstated`).
        """
        from tcip_mcp.pipelines.data.augmentations import build_augmentation

        if (augmentation := self.spec.augmentation) is None:
            raise self._unstated("augmentation")
        return build_augmentation(augmentation)

    compute_class_weights = staticmethod(losses.compute_class_weights)

    @staticmethod
    def _unstated(block: str) -> ValueError:
        """Return the refusal of a loop building from ``block``, which the run's config omits."""
        return ValueError(
            f"this run's config states no {block!r} block for the loop to build "
            "from; state one beside training_source"
        )

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """Build the optimizer the run's validated ``optimizer`` block names over ``model``.

        Its param groups take the block's own rates.
        """
        if (spec := self.spec.optimizer) is None:
            raise self._unstated("optimizer")
        return optimizer_factory.build_optimizer(
            spec, model, backbone_lr=spec.backbone_lr, head_lr=spec.head_lr
        )

    def build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Build the scheduler the run's validated ``scheduler`` block names over ``optimizer``.

        It runs at the block's own settings and horizon, in the direction of the run's selection
        metric.
        """
        from tcip_mcp.pipelines.schemas import selection_of

        if (scheduler := self.spec.scheduler) is None:
            raise self._unstated("scheduler")
        return generic_trainer._build_scheduler(
            optimizer,
            scheduler,
            higher_is_better=selection_of(
                self.spec.evaluation.selection_metric
            )[1],
        )

    apply_stage_freeze = staticmethod(generic_trainer.apply_stage_freeze)
    compute_lr_scale = staticmethod(optimizer_factory.compute_lr_scale)
    capture_training_state = staticmethod(
        optimizer_factory.capture_training_state
    )
    restore_training_state = staticmethod(
        optimizer_factory.restore_training_state
    )

    def evaluate(
        self, model: torch.nn.Module, loader: DataLoader | None = None
    ) -> dict:
        """Return ``evaluation.evaluate`` of ``model`` over ``loader`` at the run's keywords.

        ``loader`` is the run's validation loader when omitted, and the keywords are the ones the
        run resolved once (``TrainRun.evaluation_arguments``). Refuses (``ValueError``) naming
        ``val_loader`` when no loader is given and the run has none.
        """
        from tcip_mcp.pipelines.training.evaluation import evaluate

        loader = self.val_loader if loader is None else loader
        if loader is None:
            raise ValueError(
                "ctx.evaluate was given no loader and this run has no val_loader to score "
                "instead; pass loader=, or state a validation side in the run's split."
            )
        return evaluate(
            model,
            loader,
            self.device,
            self.task,
            **self.run.evaluation_arguments,
        )

    # ---- measurement primitives (compose for dimensional traits) ----
    mask_geometry = staticmethod(measurement.mask_geometry)
    instance_geometries = staticmethod(measurement.instance_geometries)

    # ---- envelope-owned sinks ----
    def _metrics_row(self, metrics: dict, stamps: dict) -> dict:
        """Return ``metrics`` in their stored form with ``stamps`` and the instant laid over them.

        The run is checked open (``experiments.require_open``). Refuses (``ValueError``) metrics
        carrying a key the log stamps (``experiments.EPOCH_KEY``, ``STEP_KEY``,
        ``TIMESTAMP_KEY``).
        """
        from tcip_mcp.audit import now_iso
        from tcip_mcp.experiments import (
            EPOCH_KEY,
            STEP_KEY,
            TIMESTAMP_KEY,
            require_open,
        )

        stamped = sorted({EPOCH_KEY, STEP_KEY, TIMESTAMP_KEY} & metrics.keys())
        if stamped:
            raise ValueError(
                f"the metrics carry {stamped}, which the metrics log stamps on each row itself; "
                "pass the epoch and step as the sink's own arguments."
            )
        require_open(self.run_dir)
        return {
            **generic_trainer._checkpoint_metrics(metrics),
            **stamps,
            TIMESTAMP_KEY: now_iso(),
        }

    def _epoch_sink(self, epoch: int, metrics: dict) -> None:
        """Append one epoch's row (:meth:`_metrics_row`) to the run's ``metrics.jsonl``.

        The row is stamped with ``epoch``. ``epoch_hook``, if attached, fires before the append
        with the metrics the body produced, not their stored form. A row JSON cannot hold raises,
        naming the field.
        """
        from tcip_mcp.experiments import EPOCH_KEY, METRICS_FILE, append_row

        row = self._metrics_row(metrics, {EPOCH_KEY: epoch})
        self._epoch = epoch
        if self.epoch_hook is not None:
            self.epoch_hook(epoch, metrics)
        append_row(self.run_dir / METRICS_FILE, row)

    def set_final_weights(self, tag: str) -> None:
        """Declare the checkpoint this body saved under ``tag`` the run's deliverable.

        The tag is one :meth:`save_checkpoint` wrote. Refuses (``ValueError``) a tag this body
        saved nothing under.
        """
        from tcip_mcp.experiments import require_open

        require_open(self.run_dir)
        if tag not in self.run.saved:
            raise ValueError(
                f"run {self.run.id} saved no checkpoint under {tag!r}."
            )
        self.run.deliverable = self.run.saved[tag]

    def report_objective(self, value: float) -> None:
        """Record ``value`` as the run's ``selection`` in a metrics row at the last epoch logged.

        The row (:meth:`_epoch_sink`) stamps ``value`` the ``selection`` under the run's config's
        ``evaluation.selection_metric``.
        """
        self._epoch_sink(
            self._epoch,
            {
                "selection": float(value),
                "selection_metric": self.spec.evaluation.selection_metric,
            },
        )

    def _write_scalars(self, metrics: dict, step: int) -> None:
        """Write each scalar of ``metrics`` at ``step`` to the run's TensorBoard (:attr:`tb`).

        Each is tagged by its key, and the writer is flushed.
        """
        from tcip_store import scalar_number

        for k, v in metrics.items():
            if scalar_number(v):
                self.tb.add_scalar(k, v, step)
        self.tb.flush()

    def log_metrics(self, epoch: int, metrics: dict) -> None:
        """Log one epoch's metrics to the run's own metrics log and to TensorBoard.

        The log gains one epoch row, TensorBoard each scalar at ``epoch``.
        """
        self._epoch_sink(epoch, metrics)
        self._write_scalars(metrics, epoch)

    def log_batch(self, step: int, epoch: int, metrics: dict) -> None:
        """Log one batch's metrics to the run's own metrics log and to TensorBoard.

        The log gains one per-batch row at ``step`` within ``epoch`` (:meth:`_metrics_row`), never
        an epoch row, so ``epoch_hook`` does not fire; TensorBoard gains each scalar at ``step``.
        """
        from tcip_mcp.experiments import (
            EPOCH_KEY,
            METRICS_FILE,
            STEP_KEY,
            append_row,
        )

        append_row(
            self.run_dir / METRICS_FILE,
            self._metrics_row(metrics, {EPOCH_KEY: epoch, STEP_KEY: step}),
        )
        self._write_scalars(metrics, step)

    def save_checkpoint(self, state: dict, tag: str = "checkpoint") -> str:
        """Write ``state`` once under ``tag`` and return the path written.

        The checkpoint is stamped with this run's config, source snapshot and geometry
        (``generic_trainer.checkpoint_stamp``) and recorded in ``run.saved``. A tag already
        written refuses with ``FileExistsError``.

        Refuses (``ValueError``) a ``state`` carrying a key the stamp writes: the checkpoint's
        config, snapshot and geometry are always this run's own.
        """
        from tcip_mcp.experiments import require_open

        require_open(self.run_dir)
        stamp = generic_trainer.checkpoint_stamp(self.run)
        if reserved := sorted(set(state) & set(stamp)):
            raise ValueError(
                f"ctx.save_checkpoint: state carries {reserved}, reserved for this run's own "
                "launch config, the record every publishing door reads this run's scope from, its "
                "source snapshot and the geometry its launch resolved; name a bespoke loop's own "
                "field something else."
            )
        self.run.saved[tag] = generic_trainer.write_checkpoint(
            {**state, **stamp},
            generic_trainer.checkpoint_path(self.run_dir, tag),
        )
        return str(self.run.saved[tag])

    def record_artifact(self, name: str, path: str) -> None:
        """Copy the file at ``path`` into the run's ``artifacts/`` directory under ``name``.

        Refuses (``FileExistsError``) a name already recorded and (``BadKeyError``) one that is
        not a single file name.
        """
        from tcip_mcp.experiments import require_open, run_name

        require_open(self.run_dir)

        destination = self.run_dir / ARTIFACTS_DIR / run_name(name)
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            raise FileExistsError(
                f"artifact {name!r} is already recorded for run {self.run.id}"
            )
        shutil.copyfile(path, destination)

    def should_cancel(self) -> bool:
        """Return whether the run should stop (``TrainRun.should_cancel``)."""
        return self.run.should_cancel()

    @property
    def tb(self) -> SummaryWriter:
        """The run's one TensorBoard writer, over its ``tensorboard`` directory.

        It is opened on first use.
        """
        if self._tb is None:
            from torch.utils.tensorboard import SummaryWriter

            from tcip_mcp.experiments import TENSORBOARD_DIR

            self._tb = SummaryWriter(
                log_dir=str(self.run_dir / TENSORBOARD_DIR)
            )
        return self._tb


def dispatch_train_body(ctx: TrainContext) -> None:
    """Run the training body, then declare its deliverable when the body declared none.

    The body is an agent's ``training_source`` if set, else ``ctx.default_train()``; the
    deliverable it falls back to is the checkpoint the body saved under ``model_best``, else
    ``model_final`` (``run.saved``). The run's TensorBoard writer (``ctx.tb``) is closed once the
    body returns or raises.
    """
    run = ctx.run
    from tcip_mcp.experiments import FINAL_STATES

    training_source = run.spec.training_source
    try:
        if training_source:
            from tcip_mcp.pipelines.model_build import import_source_builder

            agent_train = import_source_builder(training_source, run.layout)
            agent_train(
                ctx
            )  # the agent's custom loop drives training through ctx
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
        run.deliverable = run.saved.get("model_best") or run.saved.get(
            "model_final"
        )


def run_training_envelope(ctx: TrainContext) -> None:
    """Run a training body inside the audited envelope.

    The envelope opens a ``training_run`` audit event, dispatches the body
    (:func:`dispatch_train_body`), settles how the run ended (:func:`_settle_run`), and closes
    the run (:func:`close_run`), whatever the body did or omitted. An opening line that cannot
    be written fails the run naming why.
    """
    from tcip_mcp.audit import record_event_or_raise

    run = ctx.run
    audit_args = {"experiment_id": run.id, "task": ctx.task}

    t0 = time.monotonic()
    try:
        record_event_or_raise(
            "training_run",
            audit_args,
            actor=None,
            status="running",
            scope=run.project,
        )
        dispatch_train_body(ctx)
    except Exception as exc:  # noqa: BLE001
        if run.status not in ("failed", "canceled"):
            run.status = "failed"
        run.status_error = run.status_error or str(exc)
        logger.exception("Training body failed for %s: %s", run.id, exc)

    checkpoint = _settle_run(ctx)
    run.status, run.status_error = close_run(
        ctx.run_dir,
        run.project,
        run.status or "failed",
        run.status_error or None,
        checkpoint=checkpoint,
        arguments={
            **audit_args,
            **stored_number("best_metric", run.best_metric),
        },
        duration_ms=round((time.monotonic() - t0) * 1000, 1),
    )


def close_run(
    run_dir: Path,
    project: Path,
    state: str,
    status_error: str | None,
    *,
    checkpoint: dict | None,
    arguments: dict[str, Any],
    **extra: JsonValue,
) -> tuple[str, str | None]:
    """Append the run's closing ``training_run`` line, then write its final status once.

    The line goes under ``project``; the return is the state and ``status_error`` the final
    status names. A refused append ends the run ``failed`` with no checkpoint, its reason named
    after ``status_error`` when there is one.
    """
    from tcip_mcp.audit import AuditEntryNotWrittenError, record_event_or_raise
    from tcip_mcp.experiments import write_final_status

    try:
        record_event_or_raise(
            "training_run",
            arguments,
            actor=None,
            status=state,
            scope=project,
            **extra,
        )
    except AuditEntryNotWrittenError as exc:
        state, checkpoint = "failed", None
        status_error = f"{status_error}; {exc}" if status_error else str(exc)
    write_final_status(run_dir, state, status_error, checkpoint=checkpoint)
    return state, status_error


def _settle_run(ctx: TrainContext) -> dict | None:
    """Settle how the run ended and return the checkpoint its final status names.

    For a completed run that is its deliverable (``run.deliverable``), named inside the run's
    directory with the sha256 the verified checkpoint reader admitted
    (``model_registry.admitted_digest``), else ``None``. A run the wall clock stopped ends
    ``failed`` naming it; a completed run with no deliverable, or one that cannot be read or
    that the verified reader refuses, ends ``failed`` naming why.
    """
    from tcip_mcp.model_registry import admitted_digest

    run = ctx.run
    if run.wall_clock_exceeded and run.status != "failed":
        run.status, run.status_error = (
            "failed",
            "exceeded max_wall_clock_seconds",
        )
    checkpoint = None
    if run.status == "completed":
        if run.deliverable is None:
            run.status = "failed"
            run.status_error = "training completed but saved no final weights"
        else:
            try:
                checkpoint = {
                    "path": str(run.deliverable),
                    "sha256": admitted_digest(run.deliverable),
                }
            except (OSError, ValueError) as exc:
                run.status, run.status_error = (
                    "failed",
                    f"final weights could not be admitted: {exc}",
                )
    return checkpoint
