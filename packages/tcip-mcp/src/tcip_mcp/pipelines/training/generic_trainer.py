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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from tcip_mcp.pipelines.composer import compose_model, ComposedModel, DetectionModel
from tcip_mcp.pipelines.training.optimizer_factory import build_optimizer

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc,assignment]


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
    best_metric: float = float("inf")  # best val loss (lower=better)
    metrics_history: list[dict] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    error: str = ""
    output_dir: str = ""

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
def _validate(model: ComposedModel, val_loader: DataLoader, device: torch.device, task: str) -> dict:
    """Task-agnostic validation — computes loss on val set."""
    model.eval()
    total_loss = 0.0
    n = 0

    for batch in val_loader:
        if task in ("detection", "instance_seg"):
            images, targets = batch
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        else:
            images, targets = batch
            images = images.to(device)
            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

        # ComposedModel.forward returns loss dict when targets given
        model.train()  # needed for loss computation in some heads
        with torch.no_grad():
            loss_dict = model(images, targets)
            if isinstance(loss_dict, dict):
                loss = sum(loss_dict.values())
            else:
                loss = loss_dict
            total_loss += loss.item()
            n += 1

    model.eval()
    return {"val_loss": round(total_loss / max(n, 1), 6)}


# ====================================================================
# Main train loop
# ====================================================================

def train(
    run: TrainRun,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    task: str = "detection",
) -> TrainRun:
    """Execute a task-agnostic training run.

    The model is built from run.config["model_spec"] via compose_model().
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
        model = compose_model(config["model_spec"])
        model.to(device)

        stages = config.get("stages", [{"freeze_to": 0, "epochs": 10}])
        use_amp = config.get("mixed_precision", True) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        opt_cfg = config.get("optimizer", {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 1e-4})
        sched_cfg = config.get("scheduler", {"type": "cosine"})

        global_step = 0
        stopped_early = False

        for stage_idx, stage in enumerate(stages):
            if stopped_early:
                break
            run.current_stage = stage_idx

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

            optimizer = build_optimizer(
                opt_cfg.get("name", "adamw"),
                model,
                backbone_lr=opt_cfg.get("backbone_lr", 1e-4),
                head_lr=opt_cfg.get("head_lr", 1e-3),
                weight_decay=opt_cfg.get("weight_decay", 1e-4),
            )

            stage_epochs = stage.get("epochs", 10)
            scheduler = _build_scheduler(optimizer, sched_cfg, stage_epochs)
            is_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

            for epoch in range(stage_epochs):
                if stopped_early:
                    break
                run.current_epoch += 1
                model.train()
                epoch_loss = 0.0
                n_batches = 0
                optimizer.zero_grad()

                for batch_idx, batch in enumerate(train_loader):
                    if task in ("detection", "instance_seg"):
                        images, targets = batch
                        images = [img.to(device) for img in images]
                        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                    else:
                        images, targets = batch
                        images = images.to(device)
                        targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            loss_dict = model(images, targets)
                            loss = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
                        scaled = loss / accum_steps
                        scaler.scale(scaled).backward()
                        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                    else:
                        loss_dict = model(images, targets)
                        loss = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
                        scaled = loss / accum_steps
                        scaled.backward()
                        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
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
                    val_metrics = _validate(model, val_loader, device, task)

                if is_plateau:
                    scheduler.step(val_metrics.get("val_loss", avg_loss))
                else:
                    scheduler.step()

                epoch_metrics = {
                    "epoch": run.current_epoch,
                    "stage": stage_idx,
                    "train_loss": round(avg_loss, 6),
                    "lr": current_lr,
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

                logger.info("Epoch %d stage %d loss=%.4f val_loss=%.4f lr=%.2e",
                    run.current_epoch, stage_idx, avg_loss, val_metrics.get("val_loss", 0), current_lr)

                # Best model checkpoint
                val_loss = val_metrics.get("val_loss", avg_loss)
                if val_loss < run.best_metric:
                    run.best_metric = val_loss
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "model_spec": config["model_spec"],
                        "config": config,
                        "metrics": epoch_metrics,
                        "stage": stage_idx, "epoch": run.current_epoch,
                    }, out_dir / "model_best.pt")

                if ckpt_every > 0 and run.current_epoch % ckpt_every == 0:
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_spec": config["model_spec"],
                        "config": config,
                        "stage": stage_idx, "epoch": run.current_epoch,
                    }, out_dir / f"checkpoint_epoch_{run.current_epoch}.pt")

                # Early stopping
                if es_enabled and val_loader is not None:
                    current_val = val_metrics.get("val_loss", float("inf"))
                    if current_val < es_best - es_min_delta:
                        es_best = current_val
                        es_counter = 0
                    else:
                        es_counter += 1
                        if es_counter >= es_patience:
                            logger.info("Early stopping at epoch %d", run.current_epoch)
                            stopped_early = True

        # Final checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_spec": config["model_spec"],
            "config": config,
            "metrics": run.metrics_history,
        }, out_dir / "model_final.pt")

        run.status = "completed"

    except Exception as e:
        run.status = "failed"
        run.error = str(e)
        logger.exception("Training failed: %s", e)

    finally:
        run.end_time = time.time()
        if tb_writer:
            tb_writer.close()

    return run
