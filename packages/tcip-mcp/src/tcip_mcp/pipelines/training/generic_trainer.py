"""Task-agnostic training loop for ComposedModel.

This trainer works with *any* task type (detection, classification,
ordinal, regression, segmentation) because it delegates everything
to ComposedModel.forward() which returns a loss dict in train mode.

Preserves: TensorBoard, JSONL metrics, progressive unfreezing,
early stopping, mixed precision, gradient accumulation, checkpoints.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from tcip_mcp.pipelines.composer import compose_model, ComposedModel
from tcip_mcp.pipelines.training.evaluation import evaluate
from tcip_mcp.pipelines.training.optimizer_factory import (
    build_optimizer,
    compute_lr_scale,
    restore_optimizer_state,
    snapshot_optimizer_state,
)

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc,assignment]


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed random / numpy / torch (+ cuda) for reproducible runs (W7).

    ``deterministic`` additionally forces cuDNN deterministic algorithms
    (``cudnn.deterministic=True``, ``cudnn.benchmark=False``).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ====================================================================
# TrainConfig
# ====================================================================

@dataclass
class TrainConfig:
    """Everything the trainer needs — fully serializable for checkpoints."""
    model_spec: dict
    dataset: dict           # {task, images_dir, labels_dir, ...}
    augmentation: dict = field(default_factory=dict)
    sampler: str = "random"
    optimizer: dict = field(default_factory=lambda: {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 1e-4})
    stages: list[dict] = field(default_factory=lambda: [
        {"freeze_to": -1, "epochs": 5},   # train heads only (freeze all backbone)
        {"freeze_to": 2, "epochs": 10},    # unfreeze top backbone layers
        {"freeze_to": 0, "epochs": 5},     # full fine-tune
    ])
    early_stopping: dict = field(default_factory=lambda: {"enabled": True, "patience": 7, "min_delta": 1e-4})
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    checkpoint_every_n_epochs: int = 5
    batch_size: int = 4
    num_workers: int = 2
    scheduler: dict = field(default_factory=lambda: {"type": "cosine"})
    # W2 knobs (documentation/serialization only — train() reads run.config).
    stage_warmup_epochs: int = 0
    lr_scaling: dict = field(default_factory=lambda: {
        "enabled": False, "reference_effective_batch": 64, "scale_power": 0.5, "max_lr": None})
    enforce_monotonic_unfreeze: bool = True
    evaluation: dict = field(default_factory=dict)  # W1 eval params (doc-only; train() reads run.config)
    seed: int | None = None          # W7: serialization-only; runtime seeding reads run.config
    deterministic: bool = False      # W7: serialization-only


# ====================================================================
# TrainRun state
# ====================================================================

@dataclass
class TrainRun:
    run_id: str
    config: dict
    status: str = "created"
    current_epoch: int = 0
    current_stage: int = 0
    best_metric: float = float("inf")  # best selection objective (composite for detection/instance_seg, else val_loss; lower=better)
    metrics_history: list[dict] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    error: str = ""
    output_dir: str = ""
    # Set by cancel_run() to request a graceful stop; the train loop polls it.
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "current_epoch": self.current_epoch,
            "current_stage": self.current_stage,
            "best_metric": self.best_metric,
            "metrics_history": self.metrics_history,
            "elapsed_seconds": (self.end_time or time.time()) - self.start_time if self.start_time else 0,
        }


_RUNS: dict[str, TrainRun] = {}


def create_run(config: dict, output_dir: str) -> TrainRun:
    run_id = f"run_{int(time.time())}_{len(_RUNS)}"
    run = TrainRun(run_id=run_id, config=config, output_dir=output_dir)
    _RUNS[run_id] = run
    return run


def get_run(run_id: str) -> TrainRun | None:
    return _RUNS.get(run_id)


def list_runs() -> list[dict]:
    return [r.to_dict() for r in _RUNS.values()]


def cancel_run(run_id: str) -> bool:
    """Request a graceful cancellation of a training run. Returns False if unknown."""
    run = _RUNS.get(run_id)
    if run is None:
        return False
    run.cancel_event.set()
    return True


def _save_checkpoint(
    path: Path, *, model, optimizer, scheduler, scaler, config: dict,
    stage_idx: int, stage_epoch: int, run: "TrainRun",
    es_best: float, es_counter: int, global_step: int, seed, metrics: dict,
) -> None:
    """Write a resumable periodic checkpoint (W7).

    Superset of the previous payload — ``GenericPredictor`` reads only
    ``model_spec``/``model_state_dict`` and stays compatible.
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "model_spec": config["model_spec"],
        "config": config,
        "stage": stage_idx,
        "stage_epoch": stage_epoch,
        "epoch": run.current_epoch,
        "best_metric": run.best_metric,
        "es_best": es_best,
        "es_counter": es_counter,
        "global_step": global_step,
        "seed": seed,
        "metrics": metrics,
    }, path)


# ====================================================================
# Collate functions
# ====================================================================

def _detection_collate(batch):
    """Detection/instance_seg: list of (img, target) → (list[img], list[target])."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def _stack_collate(batch):
    """Classification/ordinal/regression/semantic_seg: stack into tensors."""
    images, targets = zip(*batch)
    images = torch.stack(images)
    # Merge target dicts — stack numeric values
    merged: dict[str, Any] = {}
    for key in targets[0]:
        vals = [t[key] for t in targets]
        if isinstance(vals[0], (int, float)):
            merged[key] = torch.tensor(vals)
        elif isinstance(vals[0], torch.Tensor):
            merged[key] = torch.stack(vals)
        else:
            merged[key] = vals
    return images, merged


def task_collate(task: str):
    """Return the right collate_fn for a task type."""
    if task in ("detection", "instance_seg"):
        return _detection_collate
    return _stack_collate


# ====================================================================
# Scheduler builder
# ====================================================================

def _build_scheduler(optimizer, config: dict, epochs: int):
    name = config.get("type", "cosine")
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.get("eta_min", 0))
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=config.get("factor", 0.5), patience=config.get("patience", 3))
    elif name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.get("max_lr", optimizer.defaults["lr"]), total_steps=epochs)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.get("step_size", 10), gamma=config.get("gamma", 0.1))
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


# ====================================================================
# Validation
# ====================================================================

@torch.no_grad()
def _validate(
    model: ComposedModel, val_loader: DataLoader, device: torch.device, task: str, *,
    conf_threshold: float = 0.25, iou_threshold: float = 0.5,
    iou_type: str | None = None, max_dets: int = 100, score_weights: dict | None = None,
) -> dict:
    """Task-aware validation — delegates to ``evaluation.evaluate`` and ``val_``-prefixes.

    detection/instance_seg → precision/recall/F1/mAP50/mAP + composite objective;
    classification → accuracy/F1; ordinal → MAE/rank_acc; regression → MAE/RMSE;
    semantic_seg stays loss-only. Always returns ``val_loss``.
    """
    metrics = evaluate(
        model, val_loader, device, task,
        conf_threshold=conf_threshold, iou_threshold=iou_threshold,
        iou_type=iou_type, max_dets=max_dets, score_weights=score_weights,
    )
    return {f"val_{k}": v for k, v in metrics.items()}


def _selection_value(task: str, val_metrics: dict, avg_loss: float) -> float:
    """Best-model/early-stopping driver: composite objective for detection, else val_loss."""
    if task in ("detection", "instance_seg") and "val_objective" in val_metrics:
        return val_metrics["val_objective"]
    return val_metrics.get("val_loss", avg_loss)


def _validate_input_channels(model_spec: dict, loader: DataLoader) -> None:
    """Fail loudly if the data's channel count doesn't match the backbone's ``in_chans``.

    Catches an N-channel/RGB mismatch up front with a clear message instead of an opaque
    conv-shape error deep in the first forward pass.
    """
    bb = model_spec.get("backbone", {})
    expected = bb.get("in_chans", 3) if isinstance(bb, dict) else 3
    batch = next(iter(loader), None)
    if batch is None:
        return
    imgs = batch[0]
    sample = imgs[0] if isinstance(imgs, (list, tuple)) else imgs
    if not hasattr(sample, "dim") or sample.dim() < 3:
        return
    channels = int(sample.shape[-3])
    if channels != expected:
        raise ValueError(
            f"Input images have {channels} channels but the model's backbone expects "
            f"in_chans={expected}. Set backbone.in_chans={channels} (or provide matching data)."
        )


# ====================================================================
# Main train loop
# ====================================================================

def train(
    run: TrainRun,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    task: str = "detection",
    epoch_callback=None,
    resume_from: str = "",
) -> TrainRun:
    """Execute a task-agnostic training run.

    The model is built from run.config["model_spec"] via compose_model().
    ``epoch_callback(epoch:int, epoch_metrics:dict)`` (optional) is invoked after
    each epoch's metrics are recorded — used by HPO to report intermediate values
    for pruning. It may raise to abort the run (e.g. ``optuna.TrialPruned``).
    """
    config = run.config
    run.status = "running"
    run.start_time = time.time()

    out_dir = Path(run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = None
    if SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))

    metrics_path = out_dir / "metrics.jsonl"

    # Early stopping
    es = config.get("early_stopping", {})
    es_enabled = es.get("enabled", val_loader is not None)
    es_patience = es.get("patience", 7)
    es_min_delta = es.get("min_delta", 1e-4)
    es_counter = 0
    es_best = float("inf")

    accum_steps = config.get("gradient_accumulation_steps", 1)
    ckpt_every = config.get("checkpoint_every_n_epochs", 5)

    try:
        device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # W7: seed before model build so pretrained=False init + shuffle are reproducible.
        seed = config.get("seed", config.get("training", {}).get("seed"))
        if seed is not None:
            set_seed(int(seed), deterministic=config.get(
                "deterministic", config.get("training", {}).get("deterministic", False)))

        model = compose_model(config["model_spec"])
        model.to(device)
        _validate_input_channels(config["model_spec"], train_loader)

        stages = config.get("stages", [{"freeze_to": 0, "epochs": 10}])
        use_amp = config.get("mixed_precision", True) and device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type) if use_amp else None

        opt_cfg = config.get("optimizer", {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 1e-4})
        sched_cfg = config.get("scheduler", {"type": "cosine"})

        # W2: progressive-unfreezing fidelity setup.
        base_backbone_lr = opt_cfg.get("backbone_lr", 1e-4)
        base_head_lr = opt_cfg.get("head_lr", 1e-3)
        lr_scaling_cfg = config.get("lr_scaling", {})
        stage_warmup_epochs = int(config.get("stage_warmup_epochs", 0))
        enforce_monotonic_unfreeze = config.get("enforce_monotonic_unfreeze", True)
        physical_batch = getattr(train_loader, "batch_size", None) or config.get("batch_size") or 1
        pending_snapshot = None   # best optimizer state from the previous stage
        prev_trainable = None     # trainable param count of the previous stage
        eval_cfg = config.get("evaluation", {})  # W1 metric / selection params

        global_step = 0
        stopped_early = False

        # W7: resume from a periodic checkpoint (model + optimizer + scheduler + scaler).
        resume_stage = -1
        resume_stage_epoch = 0
        ckpt = None
        if resume_from:
            ckpt = torch.load(resume_from, map_location=device, weights_only=False)
            missing = [k for k in ("model_state_dict", "optimizer_state_dict") if k not in ckpt]
            if missing:
                # Fail loudly instead of silently restarting from scratch (the old behavior).
                raise ValueError(
                    f"Cannot resume from {resume_from}: checkpoint is missing {missing} "
                    "(likely a legacy or non-resumable checkpoint, e.g. model_best.pt). Resume "
                    "from a periodic checkpoint_epoch_*.pt, or start a fresh run."
                )
            model.load_state_dict(ckpt["model_state_dict"])
            resume_stage = ckpt.get("stage", 0)
            resume_stage_epoch = ckpt.get("stage_epoch", 0)
            run.current_epoch = ckpt.get("epoch", 0)
            run.best_metric = ckpt.get("best_metric", run.best_metric)
            es_best = ckpt.get("es_best", es_best)
            es_counter = ckpt.get("es_counter", es_counter)
            global_step = ckpt.get("global_step", 0)
            logger.info("Resuming from %s at stage %d, stage_epoch %d (global epoch %d)",
                        resume_from, resume_stage, resume_stage_epoch, run.current_epoch)

        for stage_idx, stage in enumerate(stages):
            if stopped_early:
                break
            run.current_stage = stage_idx

            # W7: skip stages already completed before the resume checkpoint.
            if stage_idx < resume_stage:
                continue

            # Progressive unfreezing
            freeze_to = stage.get("freeze_to", 0)
            if freeze_to < 0:
                num_stages = getattr(model.backbone, "num_stages", 4)
                model.freeze_backbone(num_stages)
            elif freeze_to == 0:
                for p in model.parameters():
                    p.requires_grad = True
            else:
                model.freeze_backbone(freeze_to)

            # W2: progressive unfreezing must only ever grow the trainable set.
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if enforce_monotonic_unfreeze and prev_trainable is not None and trainable < prev_trainable:
                raise RuntimeError(
                    f"Non-decreasing unfreeze violated at stage {stage_idx}: "
                    f"{trainable} < {prev_trainable} trainable params"
                )
            prev_trainable = trainable

            # W2: per-stage accumulation + optional effective-batch LR scaling.
            stage_accum = stage.get("gradient_accumulation_steps", accum_steps)
            eff_batch = physical_batch * stage_accum
            stage_backbone_lr, stage_head_lr = base_backbone_lr, base_head_lr
            if lr_scaling_cfg.get("enabled", False):
                mult = compute_lr_scale(
                    eff_batch,
                    lr_scaling_cfg.get("reference_effective_batch", 64),
                    lr_scaling_cfg.get("scale_power", 0.5),
                )
                stage_backbone_lr *= mult
                stage_head_lr *= mult
                max_lr = lr_scaling_cfg.get("max_lr")
                if max_lr is not None:
                    stage_backbone_lr = min(stage_backbone_lr, max_lr)
                    stage_head_lr = min(stage_head_lr, max_lr)

            optimizer = build_optimizer(
                opt_cfg.get("name", "adamw"),
                model,
                backbone_lr=stage_backbone_lr,
                head_lr=stage_head_lr,
                weight_decay=opt_cfg.get("weight_decay", 1e-4),
            )

            # W2: hand off momentum from the previous stage's best epoch.
            if pending_snapshot is not None:
                restored = restore_optimizer_state(optimizer, model, pending_snapshot)
                logger.info("Stage %d: restored optimizer state for %d params", stage_idx, restored)

            target_lrs = [g["lr"] for g in optimizer.param_groups]
            prev_end_lrs = pending_snapshot.get("end_lrs") if pending_snapshot else None

            stage_epochs = stage.get("epochs", 10)
            # W2: inter-stage LR warmup (boundaries only; default off).
            warmup_n = (
                min(stage_warmup_epochs, stage_epochs)
                if (stage_idx > 0 and pending_snapshot is not None)
                else 0
            )
            sched_epochs = max(1, stage_epochs - warmup_n)
            scheduler = _build_scheduler(optimizer, sched_cfg, sched_epochs)
            is_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
            stage_best = float("inf")
            stage_snapshot = None

            # W7: for the resumed stage, restore optimizer/scheduler/scaler and start
            # mid-stage; later stages keep fresh state (and W2's handoff).
            start_epoch = 0
            if stage_idx == resume_stage and ckpt is not None:
                start_epoch = resume_stage_epoch
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
                        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
                        scaler.load_state_dict(ckpt["scaler_state_dict"])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Resume: optimizer/scheduler restore failed (%s); using fresh state.", exc)
                ckpt = None

            for epoch in range(start_epoch, stage_epochs):
                if stopped_early or run.cancel_event.is_set():
                    break
                run.current_epoch += 1
                model.train()
                epoch_loss = 0.0
                n_batches = 0
                optimizer.zero_grad()

                # W2: per-group linear LR warmup at the stage boundary.
                in_warmup = warmup_n > 0 and epoch < warmup_n
                if in_warmup:
                    alpha = (epoch + 1) / warmup_n
                    for gi, group in enumerate(optimizer.param_groups):
                        start = prev_end_lrs[gi] if (prev_end_lrs and gi < len(prev_end_lrs)) else 0.0
                        group["lr"] = start + alpha * (target_lrs[gi] - start)

                for batch_idx, batch in enumerate(train_loader):
                    if run.cancel_event.is_set():
                        break
                    if task in ("detection", "instance_seg"):
                        images, targets = batch
                        images = [img.to(device) for img in images]
                        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                    else:
                        images, targets = batch
                        images = images.to(device)
                        targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

                    if use_amp:
                        with torch.amp.autocast(device.type):
                            loss_dict = model(images, targets)
                            loss = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
                        scaled = loss / stage_accum
                        scaler.scale(scaled).backward()
                        if (batch_idx + 1) % stage_accum == 0 or (batch_idx + 1) == len(train_loader):
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                    else:
                        loss_dict = model(images, targets)
                        loss = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
                        scaled = loss / stage_accum
                        scaled.backward()
                        if (batch_idx + 1) % stage_accum == 0 or (batch_idx + 1) == len(train_loader):
                            optimizer.step()
                            optimizer.zero_grad()

                    epoch_loss += loss.item()
                    n_batches += 1
                    global_step += 1

                    if tb_writer and global_step % 10 == 0:
                        tb_writer.add_scalar("train/loss_step", loss.item(), global_step)

                avg_loss = epoch_loss / max(n_batches, 1)
                current_lr = optimizer.param_groups[0]["lr"]

                val_metrics = {}
                if val_loader is not None:
                    val_metrics = _validate(
                        model, val_loader, device, task,
                        conf_threshold=eval_cfg.get("conf_threshold", 0.25),
                        iou_threshold=eval_cfg.get("iou_threshold", 0.5),
                        iou_type=eval_cfg.get("iou_type"),
                        max_dets=eval_cfg.get("max_dets", 100),
                        score_weights=eval_cfg.get("score_weights"),
                    )
                sel = _selection_value(task, val_metrics, avg_loss)

                # W2: suppress the scheduler during warmup epochs.
                if not in_warmup:
                    if is_plateau:
                        scheduler.step(val_metrics.get("val_loss", avg_loss))
                    else:
                        scheduler.step()

                epoch_metrics = {
                    "epoch": run.current_epoch,
                    "stage": stage_idx,
                    "train_loss": round(avg_loss, 6),
                    "lr": current_lr,
                    "eff_batch": eff_batch,
                    "trainable_params": trainable,
                    "selection": round(sel, 6),
                    **val_metrics,
                }
                run.metrics_history.append(epoch_metrics)

                if tb_writer:
                    tb_writer.add_scalar("train/loss", avg_loss, run.current_epoch)
                    tb_writer.add_scalar("train/lr", current_lr, run.current_epoch)
                    for k, v in val_metrics.items():
                        tb_writer.add_scalar(f"val/{k}", v, run.current_epoch)
                    tb_writer.flush()

                with open(metrics_path, "a") as f:
                    f.write(json.dumps(epoch_metrics) + "\n")

                if epoch_callback is not None:
                    epoch_callback(run.current_epoch, epoch_metrics)

                logger.info("Epoch %d stage %d loss=%.4f val_loss=%.4f lr=%.2e",
                    run.current_epoch, stage_idx, avg_loss, val_metrics.get("val_loss", 0), current_lr)

                # Best model checkpoint — selected by the W1 selection objective.
                if sel < run.best_metric:
                    run.best_metric = sel
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "model_spec": config["model_spec"],
                        "config": config,
                        "metrics": epoch_metrics,
                        "stage": stage_idx, "epoch": run.current_epoch,
                    }, out_dir / "model_best.pt")

                # W2: remember this stage's best optimizer state for the handoff.
                if sel < stage_best:
                    stage_best = sel
                    stage_snapshot = snapshot_optimizer_state(optimizer, model)

                if ckpt_every > 0 and run.current_epoch % ckpt_every == 0:
                    _save_checkpoint(
                        out_dir / f"checkpoint_epoch_{run.current_epoch}.pt",
                        model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        config=config, stage_idx=stage_idx, stage_epoch=epoch + 1, run=run,
                        es_best=es_best, es_counter=es_counter, global_step=global_step,
                        seed=seed, metrics=epoch_metrics,
                    )

                # Early stopping — on the same selection objective.
                if es_enabled and val_loader is not None:
                    if sel < es_best - es_min_delta:
                        es_best = sel
                        es_counter = 0
                    else:
                        es_counter += 1
                        if es_counter >= es_patience:
                            logger.info("Early stopping at epoch %d", run.current_epoch)
                            stopped_early = True

            # W2: carry this stage's best optimizer state into the next stage.
            if stage_snapshot is not None:
                pending_snapshot = stage_snapshot

            if run.cancel_event.is_set():
                break  # stop before starting the next stage

        # Final checkpoint (saved even on cancellation so partial progress is recoverable).
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_spec": config["model_spec"],
            "config": config,
            "metrics": run.metrics_history,
        }, out_dir / "model_final.pt")

        if run.cancel_event.is_set():
            run.status = "cancelled"
            logger.info("Training run %s cancelled at epoch %d", run.run_id, run.current_epoch)
        else:
            run.status = "completed"

    except Exception as e:
        # Let HPO pruning signals propagate to Optuna (duck-typed to avoid the dep).
        if type(e).__name__ == "TrialPruned":
            raise
        run.status = "failed"
        run.error = str(e)
        logger.exception("Training failed: %s", e)

    finally:
        run.end_time = time.time()
        if tb_writer:
            tb_writer.close()

    return run
