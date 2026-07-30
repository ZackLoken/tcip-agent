"""The audited training envelope + ``TrainContext``.

The envelope is the fixed integrity boundary the platform runs AROUND any training body — the
default trainer *or* an agent's custom ``train(ctx)``. Whatever the training code does, the
envelope guarantees (the rails CLAUDE.md protects): the run is on the append-only audit log
end to end, its source/env provenance is snapshotted, its experiment status / lineage /
registration are wired, and any checkpoint it saves through ``ctx`` is stamped + atomic.

``TrainContext`` hands the training code the craft library (data / model / optim / eval utils)
plus the envelope-owned sinks (``log_metrics`` / ``save_checkpoint`` / ``record_artifact`` /
``should_cancel`` / ``tb`` / ``set_final_weights`` / ``report_objective``) — the seams that keep a
hand-rolled loop audited + immutable.

When no ``training_source`` is set, ``ctx.default_train()`` runs today's
``generic_trainer.train()``, and the envelope adds only provenance/audit *around* it.
``dispatch_train_body`` is the shared dispatch-then-derive-final-weights step both the full
envelope (``run_training_envelope``) and an HPO trial (``training_tools._run_hpo_trial``) call —
a trial runs the SAME dispatch with ``experiment_id=None``, which keeps it isolated from
provenance/registration entirely (``_finalize_run`` never fires for a trial).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    epoch_hook: Any = None        # (epoch, metrics) -> None; the stock trainer's per-epoch signal (K11)
    trial_report: Any = None      # (value: float) -> None; the raw HPO reporter, None outside HPO (K11)
    final_weights: str | None = None  # the shippable checkpoint path (K11) — see set_final_weights
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
        """Voluntary diagnostic: drive a few steps on one tiny batch and confirm the loss falls —
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
        from tcip_mcp.pipelines.training.generic_trainer import task_collate

        return task_collate(task or self.task)

    def build_sampler(self, name: str, dataset: Any) -> Any:
        from tcip_mcp.pipelines.data.samplers import build_sampler

        return build_sampler(name, dataset)

    def build_augmentation(self, cfg: dict) -> Any:
        from tcip_mcp.pipelines.data.augmentations import build_augmentation

        return build_augmentation(cfg)

    def auto_train_val(self, task: str | None = None, data_cfg: dict | None = None,
                       transforms: Any = None) -> Any:
        from tcip_mcp.tools.training_tools import _auto_train_val

        return _auto_train_val(task or self.task, data_cfg or self.config.get("data", {}), transforms)

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
        trainable-param count — the identical primitive the default trainer uses per stage."""
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
        """Resolve the trait's operating point (conf/tile/max_dets) from record sweeps — the derived,
        held-out-validated point, not a pin. Pass calibration_records/holdout_records (kwargs mirror
        ``resolve_operating_point``). Defaults ``experiment_id`` to this run's own id (K1), so the
        train-disjointness gate checks the calibration/holdout images against the training split
        this exact run drew — a caller-supplied ``experiment_id`` still wins.

        ``staged_conf_floor`` (K2 — pass it, or this can never validate): the confidence threshold
        YOUR OWN inference pass floored detections to when it produced ``calibration_records``/
        ``holdout_records`` — e.g. whatever ``score_thresh``/``score_threshold`` you set on the model
        before running it (``set_detector_operating_point``'s own return value, if you used it, IS
        this fact — thread it straight through, never re-type the number). Omitting it fails the
        reference closed as censored (``resolve_operating_point``'s own docstring explains why: an
        unstated floor can't be reconciled against the picked conf). This is a real, caller-supplied
        fact about how your records were produced, not a default this method can derive for you."""
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
        """Route epoch metrics into the experiment store, and fire ``epoch_hook`` if attached
        (K11 — an HPO trial's per-epoch pruning signal; independent of ``experiment_id``, since a
        trial runs with ``experiment_id=None``)."""
        if self.epoch_hook is not None:
            self.epoch_hook(epoch, metrics)
        if self.experiment_id is None:
            return
        try:
            from tcip_mcp.experiments import log_metrics

            log_metrics(self.experiment_id, epoch, metrics)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Experiment metric log failed (%s epoch %s): %s",
                           self.experiment_id, epoch, exc)

    def set_final_weights(self, path: str) -> None:
        """Declare the shippable checkpoint for this run (K11). ``dispatch_train_body`` derives
        this automatically from the ``model_best.pt``/``model_final.pt`` convention after the
        training body returns — call this yourself only when your loop's output doesn't follow
        that convention (e.g. a non-standard tag via ``save_checkpoint``)."""
        self.final_weights = path

    def report_objective(self, value: float) -> None:
        """Report a raw scalar directly to the active HPO trial's pruning scheduler (K11) — a
        no-op outside HPO (``trial_report`` is ``None`` on any non-trial run, so this is always
        safe to call unconditionally). The automatic ``epoch_hook`` path (fired from
        ``log_metrics``/``_epoch_sink``) only recognizes the stock trainer's own metric keys
        (``selection``/``val_objective``/``val_loss``); call this instead from a bespoke
        ``train(ctx)`` whose metrics use different names, with whatever value your loop knows
        represents trial progress (lower=better, matching the platform's minimize convention)."""
        if self.trial_report is not None:
            self.trial_report(float(value))

    def log_metrics(self, epoch: int, metrics: dict) -> None:
        """Custom-loop metric sink: experiment store + metrics.jsonl + TB (default_train writes
        metrics.jsonl/TB itself, so it uses ``_epoch_sink`` and does not double-write)."""
        self._epoch_sink(epoch, metrics)
        out = Path(self.run.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"epoch": epoch, **metrics}) + "\n")
        if self.tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    self.tb.add_scalar(k, v, epoch)
            self.tb.flush()

    def save_checkpoint(self, state: dict, tag: str = "checkpoint") -> str:
        """Stamped, atomic checkpoint save. Stamps ``kind`` + ``model_source`` + ``config`` so a
        hand-rolled loop can't emit an unstamped, un-routable ``.pt``.

        The TAG CONTRACT (K11): ``tag="model_best"`` or ``"model_final"`` is found automatically
        by ``dispatch_train_body`` after your loop returns and becomes the run's registered
        deliverable. Any other tag — including the default, ``"checkpoint"`` — is saved and
        stamped (audit/provenance are unconditional) but is NOT itself registered; call
        ``ctx.set_final_weights(path)`` with the path this method returns if you want a
        non-conventional tag to become the deliverable.
        """
        from tcip_mcp.pipelines.model_build import stamp_model_ref
        from tcip_mcp.pipelines.training.generic_trainer import _atomic_torch_save

        payload = dict(state)
        payload.setdefault("config", self.config)
        stamp_model_ref(payload, self.config, experiment_id=self.experiment_id)
        out = Path(self.run.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{tag}.pt"
        _atomic_torch_save(payload, path)
        return str(path)

    def record_artifact(self, name: str, path: str) -> None:
        if self.experiment_id is None:
            return
        try:
            from tcip_mcp.experiments import record_artifact

            record_artifact(self.experiment_id, name, str(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("record_artifact failed (%s/%s): %s", self.experiment_id, name, exc)

    def should_cancel(self) -> bool:
        return self.run.cancel_event.is_set()

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
        from tcip_mcp.experiments import experiments_dir
        from tcip_mcp.pipelines.model_build import capture_env, snapshot_model_source
        from tcip_mcp.pipelines.inference.predictor import KIND_TCIP_MODULE
        from tcip_mcp.utils.atomic_io import atomic_write_json

        kind = KIND_TCIP_MODULE
        env = {"env": capture_env(), "seed": ctx.seed, "model_kind": kind,
               "run_id": ctx.run.run_id,
               "resumed_from": ctx.resume_from or None,
               "rng_state_restored": getattr(ctx.run, "rng_state_restored", None)}
        exp_dir = experiments_dir() / ctx.experiment_id
        if exp_dir.is_dir():
            atomic_write_json(exp_dir / "env.json", env)
            # Bespoke run: copy the agent's model/training source (+ sha256) so it is reproducible
            # from an importable builder, not exec. No-op for the composed default path.
            snapshot_model_source(ctx.config, exp_dir)
    except Exception:  # noqa: BLE001
        # A dropped provenance snapshot is a real gap in the model+env link — surface it, don't
        # bury it at debug (K12: matches audit.py's own "a dropped audit line" stance).
        logger.warning("run provenance snapshot skipped", exc_info=True)


def dispatch_train_body(ctx: TrainContext) -> None:
    """Run the training body — an agent's ``training_source`` if set, else ``ctx.default_train()``
    — then resolve ``ctx.final_weights`` GENERICALLY for either path (K11).

    The ``model_best.pt``/``model_final.pt`` convention must not live inside ``default_train()``
    alone, or a bespoke loop that never calls ``ctx.set_final_weights()`` itself would leave
    ``final_weights`` unset and its otherwise-legitimate run would be treated as producing no
    deliverable. This is the ONE dispatch decision both the full audited envelope
    (``run_training_envelope``) and an HPO trial (``_run_hpo_trial``) make — call one from the
    other, don't reimplement it.
    """
    run = ctx.run
    training_source = run.config.get("training_source")
    if training_source:
        from tcip_mcp.pipelines.model_build import _import_dotted

        agent_train = _import_dotted(training_source)
        agent_train(ctx)  # the agent's custom loop drives training through ctx
        if run.status not in ("completed", "failed", "cancelled"):
            # A custom loop that never set a terminal status is treated as completed
            # (it returned without cancelling or raising).
            run.status = "cancelled" if run.cancel_event.is_set() else "completed"
    else:
        ctx.default_train()  # today's trainer

    if ctx.final_weights is None:
        out = Path(run.output_dir)
        best, final = out / "model_best.pt", out / "model_final.pt"
        if best.is_file():
            ctx.set_final_weights(str(best))
        elif final.is_file():
            ctx.set_final_weights(str(final))


def run_training_envelope(ctx: TrainContext) -> None:
    """Run a training body inside the audited integrity envelope (background-thread entry).

    In order: snapshot source/env → OPEN an audit event around the body → dispatch via
    ``dispatch_train_body`` → re-snapshot source/env (the resume/RNG outcome is only known after
    the body ran) → close status → register model + lineage + record artifact → CLOSE the audit
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

    _snapshot_run_provenance(ctx)  # refresh with the real resume/RNG-restore outcome (K11)
    _finalize_run(ctx)

    record_event("training_run", {**audit_args, "best_metric": run.best_metric},
                 status=run.status or "failed",
                 duration_ms=round((time.monotonic() - t0) * 1000, 1))


def _finalize_run(ctx: TrainContext) -> None:
    """Close status + register the model + record its weights artifact (the completion wiring)."""
    run = ctx.run
    exp_id = ctx.experiment_id
    if exp_id is None:
        return
    try:
        from tcip_mcp.experiments import (
            record_artifact,
            register_model_from_experiment,
            update_status,
        )

        if run.status == "completed" and ctx.final_weights is not None:
            update_status(exp_id, "completed")
            register_model_from_experiment(exp_id, ctx.final_weights)
            record_artifact(exp_id, "model_weights", ctx.final_weights)
        elif run.status == "completed":
            # K11: "completed" with no discoverable weights (no model_best.pt/model_final.pt,
            # and ctx.set_final_weights() was never called) is a phantom deliverable, not a
            # real one — refuse rather than register a nonexistent path.
            logger.warning(
                "Run %s completed but produced no discoverable weights (no model_best.pt/"
                "model_final.pt and ctx.set_final_weights() was never called) — marking failed "
                "instead of registering a nonexistent path.", run.run_id)
            run.status = "failed"
            run.error = run.error or "training completed but produced no final weights file"
            update_status(exp_id, "failed")
        else:
            update_status(exp_id, run.status or "failed")  # "failed" or "cancelled"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Experiment completion wiring failed for %s: %s", exp_id, exc)
