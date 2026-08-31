"""The audited training envelope + ``TrainContext``.

The envelope is the fixed integrity boundary the platform runs around any training body, the
default trainer *or* an agent's custom ``train(ctx)``. Whatever the training code does, the
envelope guarantees (the rails CLAUDE.md protects): the run is on the platform's own audit log
end to end, its source/env provenance is snapshotted, its experiment status / lineage /
registration are wired, and any checkpoint it saves through ``ctx`` is stamped + atomic.

``TrainContext`` hands the training code the craft library (data / model / optim / eval utils)
plus the envelope-owned sinks (``log_metrics`` / ``save_checkpoint`` / ``record_artifact`` /
``should_cancel`` / ``tb`` / ``set_final_weights`` / ``report_objective``), the seams that keep a
hand-rolled loop audited + immutable.

When no ``training_source`` is set, ``ctx.default_train()`` runs today's
``generic_trainer.train()``, and the envelope adds only provenance/audit *around* it.
``dispatch_train_body`` is the shared dispatch-then-derive-final-weights step both the full
envelope (``run_training_envelope``) and an HPO trial (``training_tools._run_hpo_trial``) call,
a trial runs the same dispatch with ``experiment_id=None``, which keeps it isolated from
provenance/registration entirely (``_finalize_run`` never fires for a trial).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
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
    task: str = "detection"
    resume_from: str = ""
    experiment_id: str | None = None
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
        c = self.config
        return c.get("seed", c.get("training", {}).get("seed"))

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

    def _contract_dims(self, **overrides: Any) -> dict:
        from tcip_mcp.pipelines.model_build import resolve_contract_dims

        return {**resolve_contract_dims(self.config, self.task), **overrides}

    def check_contract(self, model: Any = None, **overrides: Any) -> dict:
        """Run the measurement-boundary contract on ``model`` (built if omitted) at the run's
        resolved dims, so a hand-rolled ``train(ctx)`` can self-prove before the full loop."""
        from tcip_mcp.pipelines.model_contract import check_model_contract

        return check_model_contract(model or self.build_model(), self.task, **self._contract_dims(**overrides))

    def overfit_check(self, model: Any = None, **overrides: Any) -> dict:
        """Voluntary diagnostic: drive a few steps on one tiny batch and confirm the loss falls,
        the cheap proof a from-scratch model actually learns. Non-gating."""
        from tcip_mcp.pipelines.model_contract import overfit_check

        return overfit_check(model or self.build_model(), self.task, **self._contract_dims(**overrides))

    # ---- the default trainer: one optional convenience ----
    def default_train(self) -> Any:
        """Run today's strong default policy (progressive unfreeze / differential-LR / AMP+accum /
        selection+early-stop / checkpoint cadence)."""
        from tcip_mcp.pipelines.training.generic_trainer import train

        return train(self.run, self.train_loader, self.val_loader, self.task,
                     epoch_callback=self._epoch_sink, resume_from=self.resume_from)

    # ---- craft library passthroughs (compose, don't reinvent) ----
    def build_dataset(self, task: str | None = None, **kwargs: Any) -> Any:
        from tcip_mcp.pipelines.data.datasets import build_dataset

        return build_dataset(task or self.task, **kwargs)

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
        """``(train_ds, val_ds)``, the seam a bespoke ``train(ctx)`` body writes against.

        ``auto_train_val`` also resolves a manifest-bound run's per-stem label digests as a
        third value; that value reaches ``persist_split_manifest`` through the internal bind
        path (``subprocess_worker.run`` and this envelope's own binder), never through this
        seam, so a caller written as ``train_ds, val_ds = ctx.auto_train_val()`` is not broken by
        a change to what the bind path itself records.
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
        """Resolve the trait's operating point (conf/tile/max_dets) from record gate evidence, the derived,
        held-out-validated point, not a pin. Pass calibration_records/holdout_records (kwargs mirror
        ``resolve_operating_point``, including ``tiled``: pass it explicitly, stating whether the
        pass that produced your records actually tiled, the same fact ``staged_conf_floor`` below
        is for conf; omitting it raises, ``resolve_operating_point`` has no predictor in scope to
        derive it from and refuses to guess). Defaults ``experiment_id`` to this run's own id, so
        the train-disjointness gate checks the calibration/holdout images against the training
        split this exact run drew, a caller-supplied ``experiment_id`` still wins.

        ``staged_conf_floor`` (pass it, or this can never validate): the confidence threshold
        your own inference pass floored detections to when it produced ``calibration_records``/
        ``holdout_records``, e.g. whatever ``score_thresh``/``score_threshold`` you set on the model
        before running it (``set_detector_operating_point``'s own return value, if you used it, is
        this fact, thread it straight through, never re-type the number). Omitting it fails the
        reference closed as unstated, never as censored (``resolve_operating_point``'s own
        docstring explains why: an unstated floor can't be reconciled against the picked conf, a
        different failure from a stated floor the pick doesn't clear). This is a real,
        caller-supplied fact about how your records were produced, not a default this method can
        derive for you."""
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
        """Route one epoch's metrics to the log that owns them, and fire ``epoch_hook`` if
        attached (an HPO trial's per-epoch pruning signal; independent of ``experiment_id``,
        since a trial runs with ``experiment_id=None``).

        Every training body's rows arrive here, the default trainer's and a bespoke loop's
        alike, so the row shape and the destination are decided in one place. A run tracked as
        an experiment logs through ``experiments.log_metrics``, which owns that record's
        members and holds the terminal-state lock. An HPO trial has no experiment record, and
        its rows belong to the trial directory the Tuning view reads them back from.

        ``epoch_hook`` is fired with the metrics the body produced, not the stored form: a
        pruner compares numbers, and a diverged loss has to keep comparing as the worst one.
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
                           self.experiment_id or self.run.run_id, epoch, exc)

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
        """Custom-loop metric sink: the run's own metrics log plus TensorBoard.

        ``default_train`` reaches the same log through ``_epoch_sink`` and writes its own
        TensorBoard scalars, so it never routes through here."""
        self._epoch_sink(epoch, metrics)
        if self.tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    self.tb.add_scalar(k, v, epoch)
            self.tb.flush()

    def save_checkpoint(self, state: dict, tag: str = "checkpoint") -> str:
        """Stamped, atomic checkpoint save. Stamps ``kind`` + ``model_source`` + ``config`` so a
        hand-rolled loop can't emit an unstamped, un-routable ``.pt``.

        The tag contract: ``tag="model_best"`` or ``"model_final"`` is found automatically
        by ``dispatch_train_body`` after your loop returns and becomes the run's registered
        deliverable. Any other tag, including the default, ``"checkpoint"``, is saved and
        stamped (audit/provenance are unconditional) but is not itself registered; call
        ``ctx.set_final_weights(path)`` with the path this method returns if you want a
        non-conventional tag to become the deliverable.

        A ``metrics`` key in ``state`` becomes the registered entry's ``metrics``, with
        ``metrics_source="training_source"``: the platform wrote it into the artifact but never
        measured it, since it is whatever this loop chose to put there. Registering by
        ``metrics_source`` this way ranks only on request (``select_best_model(...,
        include_unverified=True)``), never by default alongside the platform's own
        ``default_train`` runs.

        Refuses (``ValueError``) a ``state`` carrying a ``schema_version`` key: that name is
        reserved for this platform's own checkpoint-version field, read from the payload's
        top-level namespace by the load-time version check, and a bespoke loop's own key of the
        same name would collide with it silently.
        """
        if "schema_version" in state:
            raise ValueError(
                f"ctx.save_checkpoint: state carries a 'schema_version' key "
                f"({state['schema_version']!r}), reserved for this platform's own checkpoint "
                "version field; name a bespoke loop's own field something else."
            )
        from tcip_mcp.pipelines.model_build import stamp_model_ref
        from tcip_mcp.pipelines.training.generic_trainer import checkpoint_key, write_checkpoint

        payload = dict(state)
        payload.setdefault("config", self.config)
        stamp_model_ref(payload, self.config, experiment_id=self.experiment_id)
        path = write_checkpoint(payload, checkpoint_key(self.run.output_dir, tag))
        return str(path)

    def record_artifact(self, name: str, path: str) -> None:
        """Record a named artifact against this run, except the reserved name
        ``"model_weights"``: a loop that recorded under that name meant the run's deliverable,
        so it is routed to :meth:`set_final_weights` instead of being written under a name only
        ``_finalize_run``'s own completion write is allowed to populate. Every other name behaves
        as documented: recorded, with a failure logged rather than raised.
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

    Closes the 'no source/env provenance' hole for every run: ``env.json`` records the library
    versions + seed + model kind. For a bespoke ``model_source`` / ``training_source`` run, the
    per-file source snapshot is added in S3 (``snapshot_model_source``)."""
    if ctx.experiment_id is None:
        return
    try:
        from tcip_store import store

        from tcip_mcp.experiments import env_key, experiment_dir, experiment_exists
        from tcip_mcp.pipelines.model_build import capture_env, snapshot_model_source
        from tcip_mcp.pipelines.inference.predictor import KIND_TCIP_MODULE

        kind = KIND_TCIP_MODULE
        env = {"env": capture_env(), "seed": ctx.seed, "model_kind": kind,
               "run_id": ctx.run.run_id,
               "resumed_from": ctx.resume_from or None,
               "rng_state_restored": getattr(ctx.run, "rng_state_restored", None)}
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
    """Run the training body, an agent's ``training_source`` if set, else ``ctx.default_train()``
, then resolve ``ctx.final_weights`` generically for either path.

    The ``model_best.pt``/``model_final.pt`` convention must not live inside ``default_train()``
    alone, or a bespoke loop that never calls ``ctx.set_final_weights()`` itself would leave
    ``final_weights`` unset and its otherwise-legitimate run would be treated as producing no
    deliverable. This is the one dispatch decision both the full audited envelope
    (``run_training_envelope``) and an HPO trial (``_run_hpo_trial``) make, call one from the
    other, don't reimplement it.
    """
    run = ctx.run
    from tcip_mcp.pipelines.model_build import TRAINING_SOURCE_KEY
    training_source = run.config.get(TRAINING_SOURCE_KEY)
    if training_source:
        from tcip_mcp.pipelines.model_build import _import_dotted

        agent_train = _import_dotted(training_source)
        agent_train(ctx)  # the agent's custom loop drives training through ctx
        if run.status not in ("completed", "failed", "cancelled"):
            # A custom loop that never set a terminal status is treated as completed
            # (it returned without cancelling or raising).
            run.status = "cancelled" if run.should_cancel() else "completed"
    else:
        ctx.default_train()  # today's trainer

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
    ``dispatch_train_body`` → re-snapshot source/env (the resume/RNG outcome is only known after
    the body ran) → close status → register model + lineage + record artifact → close the audit
    event. Steps other than the dispatch happen regardless of what the training code does or omits.
    """
    from tcip_mcp.audit import record_event

    run = ctx.run
    exp_id = ctx.experiment_id
    audit_args = {"run_id": run.run_id, "experiment_id": exp_id, "task": ctx.task}

    _snapshot_run_provenance(ctx)

    record_event("training_run", audit_args, status="running")
    t0 = time.monotonic()
    try:
        dispatch_train_body(ctx)
    except Exception as exc:  # noqa: BLE001
        if run.status not in ("failed", "cancelled"):
            run.status = "failed"
        run.error = run.error or str(exc)
        logger.exception("Training body failed for %s: %s", run.run_id, exc)

    _snapshot_run_provenance(ctx)  # refresh with the real resume/RNG-restore outcome
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


def _reconcile_unaudited_refusal(run: Any, exp_id: str, exc: Exception) -> None:
    """A refusal's own audit line failed to write, so ``complete_run``/``update_status`` raised
    instead of returning the refusal dict :func:`_reconcile_on_refusal` reads: reconcile from the
    record itself, the only place the actual state still is, then re-raise so the failure reaches
    ``run_training_envelope`` rather than being logged away with the refusal it was recording."""
    from tcip_mcp.experiments import get_experiment

    record = get_experiment(exp_id)
    status = record.get("status") if "error" not in record else None
    if isinstance(status, dict):
        run.status = status.get("state", run.status)
        run.error = status.get("error", run.error)
    raise exc


def _finalize_run(ctx: TrainContext) -> None:
    """Close status + register the model + record its weights artifact (the completion wiring).

    A refusal whose own audit line failed to write propagates (see
    :func:`_reconcile_unaudited_refusal`) rather than being swallowed by the outer handler below:
    reconciliation still runs, from the record directly since the normal refusal return never
    came back, and ``run_training_envelope`` wraps this call in a ``finally`` so its closing
    event still reports the reconciled state.
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
            try:
                result = complete_run(exp_id, ctx.final_weights)
            except AuditEntryNotWritten as exc:
                _reconcile_unaudited_refusal(run, exp_id, exc)
            if "error" in result:
                if "state" in result:
                    # completed is the last durable write of a run: a refusal here means the
                    # record was already terminal (a wall-clock watchdog race to failed first).
                    _reconcile_on_refusal(run, result)
                    logger.warning("Run %s: completion refused (%s); weights at %s stay on "
                                   "disk, unregistered.", run.run_id, result["error"],
                                   ctx.final_weights)
                else:
                    # final_weights could not be read: mark failed, as the phantom-deliverable
                    # case below does, rather than completing with an unrecorded digest.
                    logger.warning(
                        "Run %s: completion refused (%s); marking failed instead of completing "
                        "with an unrecorded digest.", run.run_id, result["error"])
                    run.status = "failed"
                    run.error = run.error or result["error"]
                    try:
                        _reconcile_on_refusal(run, update_status(exp_id, "failed"))
                    except AuditEntryNotWritten as exc:
                        _reconcile_unaudited_refusal(run, exp_id, exc)
            else:
                from tcip_mcp.audit import record_event

                try:
                    reg_result = register_model_from_experiment(exp_id, ctx.final_weights)
                except AuditEntryNotWritten as exc:
                    _reconcile_unaudited_refusal(run, exp_id, exc)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Run %s: model registration failed for weights at %s: %s",
                                   run.run_id, ctx.final_weights, exc)
                    record_event("model_registration_failed", {
                        "run_id": run.run_id, "experiment_id": exp_id,
                        "weights_path": str(ctx.final_weights), "reason": str(exc),
                    })
                else:
                    if "error" in reg_result:
                        logger.warning("Run %s: model registration refused for weights at %s: %s",
                                       run.run_id, ctx.final_weights, reg_result["error"])
                        record_event("model_registration_failed", {
                            "run_id": run.run_id, "experiment_id": exp_id,
                            "weights_path": str(ctx.final_weights), "reason": reg_result["error"],
                        })
        elif run.status == "completed":
            # No discoverable weights (no model_best.pt/model_final.pt, ctx.set_final_weights()
            # never called): a phantom deliverable, refuse rather than register a nonexistent path.
            logger.warning(
                "Run %s completed but produced no discoverable weights (no model_best.pt/"
                "model_final.pt and ctx.set_final_weights() was never called), marking failed "
                "instead of registering a nonexistent path.", run.run_id)
            run.status = "failed"
            run.error = run.error or "training completed but produced no final weights file"
            try:
                _reconcile_on_refusal(run, update_status(exp_id, "failed"))
            except AuditEntryNotWritten as exc:
                _reconcile_unaudited_refusal(run, exp_id, exc)
        else:
            try:
                _reconcile_on_refusal(run, update_status(exp_id, run.status or "failed"))
            except AuditEntryNotWritten as exc:
                _reconcile_unaudited_refusal(run, exp_id, exc)
    except AuditEntryNotWritten:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Experiment completion wiring failed for %s: %s", exp_id, exc)
