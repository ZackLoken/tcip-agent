"""The audited training envelope + ``TrainContext``.

The envelope is the integrity boundary the platform runs around any training body, the default
trainer or an agent's custom ``train(ctx)``: the run is on the platform's own audit log end to end,
its source/env provenance is snapshotted, its experiment status / lineage / registration are wired,
and any checkpoint it saves through ``ctx`` is stamped + atomic.

``TrainContext`` hands the training code the craft library (data / model / optim / eval utils) plus
the envelope-owned sinks (``log_metrics`` / ``save_checkpoint`` / ``record_artifact`` /
``should_cancel`` / ``tb`` / ``set_final_weights`` / ``report_objective``).

When no ``training_source`` is set, ``ctx.default_train()`` runs ``generic_trainer.train()``.
``dispatch_train_body`` is the dispatch-then-derive-final-weights step; an HPO trial runs it with
``experiment_id=None``, isolated from provenance/registration.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tcip_store import stored_number

logger = logging.getLogger(__name__)


@dataclass
class TrainContext:
    """The handle the envelope passes to a training body (default or custom).

    Carries the prebuilt leakage-free loaders + run state, exposes the craft library as thin
    passthroughs, and owns the audited/immutable sinks a custom loop must route through.
    """

    run: Any                      # TrainRun
    train_loader: Any
    val_loader: Any | None = None
    task: str = field(kw_only=True)
    resume_from: str = ""
    experiment_id: str | None = None  # None means no record; nothing reads run.id as this instead
    epoch_hook: Any = None        # (epoch, metrics) -> None; the stock trainer's per-epoch signal
    trial_report: Any = None      # (value: float) -> None; the raw HPO reporter, None outside HPO
    final_weights: str | None = None  # the shippable checkpoint path, see set_final_weights
    _tb: Any = None

    # ---- config / reproducibility ----
    @property
    def config(self) -> dict:
        return self.run.config

    @property
    def seed(self) -> Any:
        return self.config.get("seed")

    @property
    def device(self) -> Any:
        import torch

        return torch.device(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    def set_seed(self, seed: int | None = None, deterministic: bool = False) -> None:
        from tcip_mcp.pipelines.training.generic_trainer import set_seed

        eff = self.seed if seed is None else seed
        if eff is not None:
            set_seed(int(eff), deterministic=deterministic)

    # ---- model ----
    def build_model(self) -> Any:
        from tcip_mcp.pipelines.model_build import build_model

        return build_model(self.config)

    def _contract_args(self, **overrides: Any) -> dict:
        """What the smoke runs against: the dims this run resolved, or one batch off its own train
        loader when those dims cannot shape a batch (``no_batch_reason``).
        """
        from tcip_mcp.pipelines.data.datasets import stated_sizes
        from tcip_mcp.pipelines.data.selection import ClassScope
        from tcip_mcp.pipelines.model_build import resolve_contract_dims
        from tcip_mcp.pipelines.model_contract import no_batch_reason

        data_cfg = self.config.get("data") or {}
        dims = resolve_contract_dims(self.config, self.task,
                                     scope=ClassScope.recorded_in(data_cfg),
                                     sizes=stated_sizes(data_cfg))
        if no_batch_reason(self.task, dims, None) is not None:
            batch = next(iter(self.train_loader or []), None)
            if batch is not None:
                return {"sample_batch": batch, **overrides}
        return {"dims": dims, **overrides}

    def check_contract(self, model: Any = None, **overrides: Any) -> dict:
        """Run the measurement-boundary contract on ``model`` (built if omitted) at the run's
        resolved dims, so a hand-rolled ``train(ctx)`` can self-prove before the full loop."""
        from tcip_mcp.pipelines.model_contract import check_model_contract

        return check_model_contract(model or self.build_model(), self.task, **self._contract_args(**overrides))

    def overfit_check(self, model: Any = None, **overrides: Any) -> dict:
        """Voluntary diagnostic: drive a few steps on one tiny batch and confirm the loss falls,
        the cheap proof a from-scratch model actually learns. Non-gating."""
        from tcip_mcp.pipelines.model_contract import overfit_check

        return overfit_check(model or self.build_model(), self.task, **self._contract_args(**overrides))

    # ---- the default trainer: one optional convenience ----
    def default_train(self) -> Any:
        """Run the default policy (progressive unfreeze / differential-LR / AMP+accum /
        selection+early-stop / checkpoint cadence)."""
        from tcip_mcp.pipelines.training.generic_trainer import train

        return train(self.run, self.train_loader, self.val_loader, task=self.task,
                     epoch_callback=self._epoch_sink, resume_from=self.resume_from)

    # ---- craft library passthroughs (compose, don't reinvent) ----
    def build_dataset(self, task: str | None = None, *, samples: Any,
                      sizes: "Mapping[str, int] | None" = None, **kwargs: Any) -> Any:
        """The factory, over the samples you were handed. ``sizes`` unstated resolves from this
        run's own data config and those samples, so a loader you build here reads at its width."""
        from tcip_mcp.pipelines.data.datasets import build_dataset, resolve_sizes

        resolved_task = task or self.task
        if sizes is None:
            sizes = resolve_sizes(resolved_task, self.config.get("data") or {}, samples,
                                  kwargs.get("dataset_source"))
        return build_dataset(resolved_task, samples=samples, sizes=sizes, **kwargs)

    def tiled_dataset(self, base: Any, **kwargs: Any) -> Any:
        """Wrap a detection dataset in the native-resolution tiler (same derived sliver cutoff the
        default path uses); ``kwargs``: tile_size / overlap / sliver_frac / dedup_iou / skip_empty."""
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

    def auto_train_val(self, task: str | None = None, data_cfg: dict | None = None,
                       transforms: Any = None) -> Any:
        """``(train_ds, val_ds)``, the seam a bespoke ``train(ctx)`` body writes against; the
        recorded partition ``auto_train_val`` also resolves is not returned here.
        """
        from tcip_mcp.pipelines.data.split_construction import auto_train_val

        train_ds, val_ds, _label_digests = auto_train_val(
            task or self.task, data_cfg or self.config.get("data", {}), transforms)
        return train_ds, val_ds

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
        trainable-param count, the identical primitive the default trainer uses per stage."""
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
        from tcip_mcp.pipelines.training.evaluation import evaluate

        return evaluate(model, self.val_loader if loader is None else loader,
                        self.device, self.task, **kwargs)

    # ---- measurement primitives (compose for dimensional traits) ----
    def calibrate(self, trait_name: str, **kwargs: Any) -> Any:
        """Resolve the trait's operating point (conf/tile/max_dets) from record gate evidence. Pass
        calibration_records/holdout_records; kwargs mirror ``resolve_operating_point``, including
        ``tiled``, which is required (whether the pass that produced your records tiled).
        ``experiment_id`` defaults to this run's own id; a caller-supplied one wins.

        ``staged_conf_floor`` (pass it, or this can never validate): the confidence threshold your
            own inference pass floored detections to when it produced
            ``calibration_records``/``holdout_records``, e.g. the
            ``score_thresh``/``score_threshold`` you set on the model
            (``set_detector_operating_point``'s own return value, if you used it). Omitting it
            fails the reference as unstated (``conf_floor_unstated``).
        """
        from tcip_mcp.pipelines.operating_point import resolve_operating_point

        kwargs.setdefault("experiment_id", self.experiment_id)
        return resolve_operating_point(trait_name, **kwargs)

    def mask_geometry(self, *args: Any, **kwargs: Any) -> Any:
        from tcip_mcp.pipelines.measurement import mask_geometry

        return mask_geometry(*args, **kwargs)

    def instance_geometries(self, *args: Any, **kwargs: Any) -> Any:
        from tcip_mcp.pipelines.measurement import instance_geometries

        return instance_geometries(*args, **kwargs)

    # ---- envelope-owned sinks: keep a custom loop audited + immutable ----
    def _epoch_sink(self, epoch: int, metrics: dict) -> None:
        """Route one epoch's metrics to the log that owns them, and fire ``epoch_hook`` if attached
        (an HPO trial's per-epoch pruning signal; independent of ``experiment_id``).

        A run tracked as an experiment logs through ``experiments.log_metrics``, which owns that
        record's members and holds the terminal-state lock. An HPO trial has no experiment record,
        and its rows belong to the trial directory.

        ``epoch_hook`` is fired with the metrics the body produced, not the stored form: a diverged
        loss keeps comparing as the worst one.
        """
        if self.epoch_hook is not None:
            self.epoch_hook(epoch, metrics)
        from tcip_mcp.pipelines.training.generic_trainer import _checkpoint_metrics

        stored = _checkpoint_metrics(metrics)
        try:
            if self.experiment_id is None:
                from tcip_store import append

                from tcip_mcp.tools.training_tools import trial_metrics_key_for_dir

                append(trial_metrics_key_for_dir(self.run.output_dir),
                       {"epoch": epoch, **stored})
                return
            from tcip_mcp.experiments import log_metrics

            log_metrics(self.experiment_id, epoch, stored)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Metric log failed (%s epoch %s): %s",
                           self.experiment_id or self.run.id, epoch, exc)

    def set_final_weights(self, path: str) -> None:
        """Declare the shippable checkpoint for this run. ``dispatch_train_body`` derives
        this automatically from the ``model_best.pt``/``model_final.pt`` convention after the
        training body returns, call this yourself only when your loop's output doesn't follow
        that convention (e.g. a non-standard tag via ``save_checkpoint``)."""
        self.final_weights = path

    def report_objective(self, value: float) -> None:
        """Report a raw scalar directly to the active HPO trial's pruning scheduler, a
        no-op outside HPO (``trial_report`` is ``None`` on any non-trial run, so this is always
        safe to call unconditionally). The automatic ``epoch_hook`` path (fired from
        ``log_metrics``/``_epoch_sink``) only recognizes the stock trainer's own metric keys
        (``selection``/``val_objective``/``val_loss``); call this instead from a bespoke
        ``train(ctx)`` whose metrics use different names, with whatever value your loop knows
        represents trial progress, in the direction the sweep's resolved selection metric
        declares as better (``evaluation.HIGHER_IS_BETTER_BY_METRIC``), not a fixed convention."""
        if self.trial_report is not None:
            self.trial_report(float(value))

    def log_metrics(self, epoch: int, metrics: dict) -> None:
        """Custom-loop metric sink: the run's own metrics log plus TensorBoard."""
        self._epoch_sink(epoch, metrics)
        if self.tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    self.tb.add_scalar(k, v, epoch)
            self.tb.flush()

    def save_checkpoint(self, state: dict, tag: str = "checkpoint") -> str:
        """Stamped, atomic checkpoint save. Stamps ``kind`` + ``model_source`` + ``config`` so a
        hand-rolled loop can't emit an unstamped, un-routable ``.pt``.

        The tag contract: ``tag="model_best"`` or ``"model_final"`` is found automatically by
        ``dispatch_train_body`` after your loop returns and becomes the run's registered
        deliverable. Any other tag, including the default, ``"checkpoint"``, is saved and stamped
        but is not itself registered; call ``ctx.set_final_weights(path)`` with the path this
        method returns if you want a non-conventional tag to become the deliverable.

        A ``metrics`` key in ``state`` becomes the registered entry's ``metrics``, with
        ``metrics_source="training_source"``: the platform wrote it into the artifact but never
        measured it. Registering by ``metrics_source`` this way ranks only on request
        (``rank_registered_models(..., include_unverified=True)``).

        Refuses (``ValueError``) a ``state`` carrying a ``schema_version`` key (the platform's own
        checkpoint-version field) or a ``config`` key (the checkpoint's ``config`` is always this
        run's launch config, the record a run's ``(subject, attribute, id_map)`` scope is read
        from).
        """
        if "schema_version" in state:
            raise ValueError(
                f"ctx.save_checkpoint: state carries a 'schema_version' key "
                f"({state['schema_version']!r}), reserved for this platform's own checkpoint "
                "version field; name a bespoke loop's own field something else."
            )
        if "config" in state:
            raise ValueError(
                "ctx.save_checkpoint: state carries a 'config' key, reserved for this run's own "
                "launch config, the record every publishing door reads this run's scope from; "
                "name a bespoke loop's own field something else."
            )
        from tcip_mcp.pipelines.model_build import stamp_model_ref
        from tcip_mcp.pipelines.training.generic_trainer import checkpoint_key, write_checkpoint

        payload = dict(state)
        payload["config"] = self.config
        stamp_model_ref(payload, self.config, experiment_id=self.experiment_id)
        path = write_checkpoint(payload, checkpoint_key(self.run.output_dir, tag))
        return str(path)

    def record_artifact(self, name: str, path: str) -> None:
        """Record a named artifact against this run; a failure is logged rather than raised. The
        reserved name ``"model_weights"`` is routed to :meth:`set_final_weights` instead.
        """
        if name == "model_weights":
            logger.warning(
                "record_artifact(%r/'model_weights', %s) routed to set_final_weights: that "
                "name is the run's deliverable, and only completing the run records it.",
                self.experiment_id, path)
            self.set_final_weights(str(path))
            return
        if self.experiment_id is None:
            return
        try:
            from tcip_mcp.experiments import record_artifact

            record_artifact(self.experiment_id, name, str(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("record_artifact failed (%s/%s): %s", self.experiment_id, name, exc)

    def should_cancel(self) -> bool:
        return self.run.should_cancel()

    @property
    def tb(self) -> Any:
        if self._tb is None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(Path(self.run.output_dir) / "tensorboard"))
            except Exception:  # noqa: BLE001
                self._tb = None
        return self._tb


def _snapshot_run_provenance(ctx: TrainContext) -> None:
    """Snapshot env (+ bespoke model source) into the immutable experiment dir. Best-effort.

    ``env.json`` records the library versions, seed and model kind for every run. For a bespoke
    ``model_source`` / ``training_source`` run, the per-file source snapshot is added by
    ``snapshot_model_source``."""
    if ctx.experiment_id is None:
        return
    try:
        from tcip_store import store

        from tcip_mcp.experiments import env_key, experiment_dir, experiment_exists
        from tcip_mcp.pipelines.model_build import capture_env, snapshot_model_source
        from tcip_mcp.pipelines.inference.predictor import KIND_TCIP_MODULE

        kind = KIND_TCIP_MODULE
        env = {"env": capture_env(), "model_kind": kind, "resumed_from": ctx.resume_from or None}
        if experiment_exists(ctx.experiment_id):
            store.replace(env_key(ctx.experiment_id), env)
            # Bespoke run: copy the agent's model/training source (+ sha256) so it is reproducible
            # from an importable builder, not exec. No-op for the composed default path.
            snapshot_model_source(ctx.config, experiment_dir(ctx.experiment_id))
    except Exception:  # noqa: BLE001
        # A dropped provenance snapshot is a real gap in the model+env link, surface it, don't
        # bury it at debug (matches audit.py's own "a dropped audit line" stance).
        logger.warning("run provenance snapshot skipped", exc_info=True)


def dispatch_train_body(ctx: TrainContext) -> None:
    """Run the training body, an agent's ``training_source`` if set, else ``ctx.default_train()``,
    then resolve ``ctx.final_weights`` for either path from the
    ``model_best.pt``/``model_final.pt`` convention when the body set none.
    """
    run = ctx.run
    from tcip_mcp.pipelines.model_build import TRAINING_SOURCE_KEY
    training_source = run.config.get(TRAINING_SOURCE_KEY)
    if training_source:
        from tcip_mcp.pipelines.model_build import _import_dotted

        agent_train = _import_dotted(training_source)
        agent_train(ctx)  # the agent's custom loop drives training through ctx
        from tcip_mcp.experiments import _RECORDED_AS_DONE

        if run.status not in _RECORDED_AS_DONE:
            # A custom loop that never set a terminal status is treated as completed
            # (it returned without canceling or raising).
            run.status = "canceled" if run.should_cancel() else "completed"
    else:
        ctx.default_train()  # the default trainer

    if ctx.final_weights is None:
        from tcip_store import blob_path

        from tcip_mcp.pipelines.training.generic_trainer import checkpoint_key

        best = blob_path(checkpoint_key(run.output_dir, "model_best"))
        final = blob_path(checkpoint_key(run.output_dir, "model_final"))
        if best.is_file():
            ctx.set_final_weights(str(best))
        elif final.is_file():
            ctx.set_final_weights(str(final))


def run_training_envelope(ctx: TrainContext) -> None:
    """Run a training body inside the audited integrity envelope (background-thread entry).

    In order: snapshot source/env → open an audit event around the body → dispatch via
    ``dispatch_train_body`` → re-snapshot source/env → close status → register model + lineage + record artifact → close the audit
    event. Steps other than the dispatch happen regardless of what the training code does or omits.
    """
    from tcip_mcp.audit import record_event

    run = ctx.run
    exp_id = ctx.experiment_id
    audit_args = {"experiment_id": exp_id, "task": ctx.task}

    _snapshot_run_provenance(ctx)

    record_event("training_run", audit_args, status="running")
    t0 = time.monotonic()
    try:
        dispatch_train_body(ctx)
    except Exception as exc:  # noqa: BLE001
        if run.status not in ("failed", "canceled"):
            run.status = "failed"
        run.error = run.error or str(exc)
        logger.exception("Training body failed for %s: %s", run.id, exc)

    _snapshot_run_provenance(ctx)
    try:
        _finalize_run(ctx)
    finally:
        # Runs even if _finalize_run re-raises: run.status is already reconciled by then.
        record_event("training_run", {**audit_args, **stored_number("best_metric", run.best_metric)},
                     status=run.status or "failed",
                     duration_ms=round((time.monotonic() - t0) * 1000, 1))


def _reconcile_on_refusal(run: Any, result: dict[str, Any]) -> bool:
    """True when ``result`` is a refusal carrying the record's own state (a not-found result
    carries none): reconciles ``run.status``/``run.error`` to it, so the closing audit event
    reports the state the record actually holds rather than the one the child believed."""
    if "error" not in result or "state" not in result:
        return False
    run.status = result["state"]
    run.error = result["error"]
    return True


def _finalize_run(ctx: TrainContext) -> None:
    """Close status + register the model + record its weights artifact (the completion wiring).

    A refusal comes back as its dict, which :func:`_reconcile_on_refusal` reads. A registration
    that committed and could not write its own line propagates.
    """
    run = ctx.run
    exp_id = ctx.experiment_id
    if exp_id is None:
        return
    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_mcp.experiments import (
        complete_run,
        register_model_from_experiment,
        update_status,
    )

    try:
        if run.status == "completed" and ctx.final_weights is not None:
            result = complete_run(exp_id, ctx.final_weights)
            if "error" in result:
                if "state" in result:
                    # completed is the last durable write of a run: a refusal here means the
                    # record was already terminal (a wall-clock watchdog race to failed first).
                    _reconcile_on_refusal(run, result)
                    logger.warning("Run %s: completion refused (%s); weights at %s stay on "
                                   "disk, unregistered.", run.id, result["error"],
                                   ctx.final_weights)
                else:
                    # final_weights could not be read: mark failed, as the phantom-deliverable
                    # case below does, rather than completing with an unrecorded digest.
                    logger.warning(
                        "Run %s: completion refused (%s); marking failed instead of completing "
                        "with an unrecorded digest.", run.id, result["error"])
                    run.status = "failed"
                    run.error = run.error or result["error"]
                    _reconcile_on_refusal(run, update_status(exp_id, "failed"))
            else:
                try:
                    reg_result = register_model_from_experiment(exp_id, ctx.final_weights)
                except AuditEntryNotWritten:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Run %s: model registration failed for weights at %s: %s",
                                   run.id, ctx.final_weights, exc)
                else:
                    if "error" in reg_result:
                        logger.warning("Run %s: model registration refused for weights at %s: %s",
                                       run.id, ctx.final_weights, reg_result["error"])
        elif run.status == "completed":
            # No discoverable weights (no model_best.pt/model_final.pt, ctx.set_final_weights()
            # never called): a phantom deliverable, refuse rather than register a nonexistent path.
            logger.warning(
                "Run %s completed but produced no discoverable weights (no model_best.pt/"
                "model_final.pt and ctx.set_final_weights() was never called), marking failed "
                "instead of registering a nonexistent path.", run.id)
            run.status = "failed"
            run.error = run.error or "training completed but produced no final weights file"
            _reconcile_on_refusal(run, update_status(exp_id, "failed"))
        else:
            _reconcile_on_refusal(
                run, update_status(exp_id, run.status or "failed", error=run.error or None))
    except AuditEntryNotWritten:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Experiment completion wiring failed for %s: %s", exp_id, exc)
