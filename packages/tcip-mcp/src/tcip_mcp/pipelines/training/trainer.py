"""Detection trainer with multi-stage progressive unfreezing.

The training loop follows this pattern:
  Stage 1: Freeze backbone, train head only (high LR)
  Stage 2: Unfreeze top backbone layers, lower LR
  Stage 3: Unfreeze all, very low LR (full fine-tuning)

Uses torch.cuda.amp for mixed precision when available.
Writes TensorBoard event files, JSONL metrics, and periodic checkpoints.
Supports early stopping, gradient accumulation, and multiple LR schedulers.
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

from tcip_mcp.pipelines.models.builder import build_model
from tcip_mcp.pipelines.models.losses import sum_losses
from tcip_mcp.pipelines.training.stages import apply_stage, get_default_stages

logger = logging.getLogger(__name__)

# Optional TensorBoard import — graceful fallback
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc,assignment]
    logger.info("tensorboard not installed — TensorBoard logging disabled")


@dataclass
class TrainRun:
    """Tracks state of a single training run."""
    run_id: str
    config: dict
    status: str = "created"  # created | running | completed | failed | stopped
    current_epoch: int = 0
    current_stage: int = 0
    best_metric: float = 0.0
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


# Global run registry (in-process, no persistence needed for Phase 1)
_RUNS: dict[str, TrainRun] = {}


def create_run(config: dict, output_dir: str) -> TrainRun:
    """Create a new training run from config."""
    run_id = f"run_{int(time.time())}_{len(_RUNS)}"
    run = TrainRun(run_id=run_id, config=config, output_dir=output_dir)
    _RUNS[run_id] = run
    return run


def get_run(run_id: str) -> TrainRun | None:
    return _RUNS.get(run_id)


def list_runs() -> list[dict]:
    return [r.to_dict() for r in _RUNS.values()]


def validate_training_config(config: dict) -> list[str]:
    """Validate training config for semantic issues (issue #38)."""
    issues: list[str] = []

    model_cfg = config.get("model", {})
    nc = model_cfg.get("num_classes")
    if nc is not None:
        if not isinstance(nc, int) or nc < 1:
            issues.append(f"num_classes must be a positive integer, got {nc}")
        if nc > 1000:
            issues.append(f"num_classes={nc} is suspiciously large")

    for stage in config.get("stages", []):
        lr = stage.get("lr")
        if lr is not None:
            if not isinstance(lr, (int, float)) or lr <= 0:
                issues.append(f"lr must be positive, got {lr}")
            if lr > 1.0:
                issues.append(f"lr={lr} is unusually high (>1.0)")
        epochs = stage.get("epochs")
        if epochs is not None:
            if not isinstance(epochs, int) or epochs < 1:
                issues.append(f"epochs must be a positive integer, got {epochs}")
            if epochs > 500:
                issues.append(f"epochs={epochs} is very high, consider early stopping")

    bs = config.get("batch_size")
    if bs is not None:
        if not isinstance(bs, int) or bs < 1:
            issues.append(f"batch_size must be a positive integer, got {bs}")

    wd = config.get("weight_decay")
    if wd is not None:
        if not isinstance(wd, (int, float)) or wd < 0:
            issues.append(f"weight_decay must be non-negative, got {wd}")

    es = config.get("early_stopping", {})
    patience = es.get("patience")
    if patience is not None:
        if not isinstance(patience, int) or patience < 1:
            issues.append(f"early_stopping.patience must be a positive integer, got {patience}")

    return issues


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: dict,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build a learning rate scheduler from config (issue #25)."""
    name = scheduler_config.get("type", "cosine")
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs,
            eta_min=scheduler_config.get("eta_min", 0),
        )
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 3),
        )
    elif name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_config.get("max_lr", optimizer.defaults["lr"]),
            total_steps=epochs,
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 10),
            gamma=scheduler_config.get("gamma", 0.1),
        )
    else:
        logger.warning("Unknown scheduler '%s', falling back to cosine", name)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Run a validation pass and return loss + detection metrics."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_predictions: list[dict] = []
    all_targets: list[dict] = []

    for images, targets in val_loader:
        images_dev = [img.to(device) for img in images]
        targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Get loss (model must be in train mode for loss computation)
        model.train()
        with torch.no_grad():
            loss_dict = model(images_dev, targets_dev)
            loss = sum_losses(loss_dict)
            total_loss += loss.item()
            n_batches += 1

        # Get predictions (model in eval mode)
        model.eval()
        outputs = model(images_dev)
        for out, tgt in zip(outputs, targets):
            all_predictions.append({
                "boxes": out["boxes"].cpu(),
                "scores": out["scores"].cpu(),
                "labels": out["labels"].cpu(),
            })
            all_targets.append({
                "boxes": tgt["boxes"],
                "labels": tgt["labels"],
            })

    avg_val_loss = total_loss / max(n_batches, 1)

    # Compute mAP if we have predictions
    val_map = 0.0
    if all_predictions:
        try:
            from tcip_mcp.pipelines.evaluation.metrics import compute_map
            result = compute_map(all_predictions, all_targets, iou_thresholds=[0.5, 0.75])
            val_map = result.get("mAP", 0.0)
        except Exception as e:
            logger.warning("mAP computation failed: %s", e)

    return {
        "val_loss": round(avg_val_loss, 6),
        "val_mAP50": round(val_map, 4),
    }


def train(run: TrainRun, train_loader: DataLoader, val_loader: DataLoader | None = None) -> TrainRun:
    """Execute a training run synchronously.

    Supports:
    - TensorBoard event logging (issue #1)
    - Validation loop with mAP computation (issue #18→val_loader)
    - Periodic checkpoints (issue #19→every N epochs)
    - Early stopping (issue #26→patience-based on val_loss)
    - Gradient accumulation (issue #24 partial→accum_steps)
    - Configurable LR schedulers (issue #25)
    - JSONL metrics for GUI polling
    """
    config = run.config
    run.status = "running"
    run.start_time = time.time()

    out_dir = Path(run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard writer (issue #1)
    tb_writer = None
    if SummaryWriter is not None:
        tb_log_dir = out_dir / "tensorboard"
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
        logger.info("TensorBoard logging to %s", tb_log_dir)

    # JSONL metrics file for GUI polling
    metrics_jsonl_path = out_dir / "metrics.jsonl"

    # Early stopping config (issue #26)
    es_config = config.get("early_stopping", {})
    es_enabled = es_config.get("enabled", val_loader is not None)
    es_patience = es_config.get("patience", 7)
    es_min_delta = es_config.get("min_delta", 1e-4)
    es_counter = 0
    es_best_loss = float("inf")

    # Gradient accumulation (partial issue #24)
    accum_steps = config.get("gradient_accumulation_steps", 1)

    # Checkpoint frequency
    ckpt_every = config.get("checkpoint_every_n_epochs", 5)

    try:
        device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        model = build_model(config["model"])
        model.to(device)

        stages = config.get("stages", get_default_stages())
        use_amp = config.get("mixed_precision", True) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        global_step = 0
        stopped_early = False

        for stage_idx, stage_config in enumerate(stages):
            if stopped_early:
                break
            run.current_stage = stage_idx
            apply_stage(model, stage_config)

            lr = stage_config.get("lr", 1e-3)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                weight_decay=config.get("weight_decay", 1e-4),
            )

            stage_epochs = stage_config.get("epochs", 10)
            scheduler_config = stage_config.get("scheduler", {"type": "cosine"})
            scheduler = _build_scheduler(optimizer, scheduler_config, stage_epochs)
            is_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

            for epoch in range(stage_epochs):
                if stopped_early:
                    break
                run.current_epoch += 1
                model.train()
                epoch_loss = 0.0
                n_batches = 0

                optimizer.zero_grad()
                for batch_idx, (images, targets) in enumerate(train_loader):
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            loss_dict = model(images, targets)
                            loss = sum_losses(loss_dict)
                        # Scale loss for gradient accumulation
                        scaled = loss / accum_steps
                        scaler.scale(scaled).backward()
                        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                    else:
                        loss_dict = model(images, targets)
                        loss = sum_losses(loss_dict)
                        scaled = loss / accum_steps
                        scaled.backward()
                        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                            optimizer.step()
                            optimizer.zero_grad()

                    epoch_loss += loss.item()
                    n_batches += 1
                    global_step += 1

                    # TensorBoard: batch-level loss
                    if tb_writer is not None and global_step % 10 == 0:
                        tb_writer.add_scalar("train/loss_step", loss.item(), global_step)

                avg_loss = epoch_loss / max(n_batches, 1)
                current_lr = optimizer.param_groups[0]["lr"]

                # Validation (issue #18)
                val_metrics = {}
                if val_loader is not None:
                    val_metrics = _validate(model, val_loader, device)

                # LR scheduler step
                if is_plateau:
                    scheduler.step(val_metrics.get("val_loss", avg_loss))
                else:
                    scheduler.step()

                # Compose epoch metrics
                epoch_metrics = {
                    "epoch": run.current_epoch,
                    "stage": stage_idx,
                    "train_loss": round(avg_loss, 6),
                    "lr": current_lr,
                    **val_metrics,
                }
                run.metrics_history.append(epoch_metrics)

                # TensorBoard: epoch-level scalars (issue #1)
                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", avg_loss, run.current_epoch)
                    tb_writer.add_scalar("train/lr", current_lr, run.current_epoch)
                    if "val_loss" in val_metrics:
                        tb_writer.add_scalar("val/loss", val_metrics["val_loss"], run.current_epoch)
                    if "val_mAP50" in val_metrics:
                        tb_writer.add_scalar("val/mAP50", val_metrics["val_mAP50"], run.current_epoch)
                    tb_writer.flush()

                # JSONL metrics for GUI polling
                with open(metrics_jsonl_path, "a") as f:
                    f.write(json.dumps(epoch_metrics) + "\n")

                logger.info(
                    "Epoch %d stage %d loss=%.4f val_loss=%.4f mAP50=%.4f lr=%.2e",
                    run.current_epoch, stage_idx, avg_loss,
                    val_metrics.get("val_loss", 0), val_metrics.get("val_mAP50", 0),
                    current_lr,
                )

                # Track best metric
                val_map = val_metrics.get("val_mAP50", 0.0)
                if val_map > run.best_metric:
                    run.best_metric = val_map
                    # Save best model
                    best_path = out_dir / "model_best.pt"
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "stage": stage_idx,
                        "epoch": run.current_epoch,
                        "config": config,
                        "metrics": epoch_metrics,
                    }, best_path)

                # Periodic checkpoint (issue #19)
                if ckpt_every > 0 and run.current_epoch % ckpt_every == 0:
                    ckpt_path = out_dir / f"checkpoint_epoch_{run.current_epoch}.pt"
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if scaler else None,
                        "stage": stage_idx,
                        "epoch": run.current_epoch,
                        "config": config,
                    }, ckpt_path)

                # Early stopping (issue #26)
                if es_enabled and val_loader is not None:
                    current_val_loss = val_metrics.get("val_loss", float("inf"))
                    if current_val_loss < es_best_loss - es_min_delta:
                        es_best_loss = current_val_loss
                        es_counter = 0
                    else:
                        es_counter += 1
                        if es_counter >= es_patience:
                            logger.info("Early stopping triggered at epoch %d", run.current_epoch)
                            stopped_early = True

            # Save checkpoint after each stage
            ckpt_path = out_dir / f"stage_{stage_idx}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "stage": stage_idx,
                "epoch": run.current_epoch,
                "config": config,
            }, ckpt_path)

        # Save final model
        final_path = out_dir / "model_final.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "metrics": run.metrics_history,
        }, final_path)

        run.status = "completed"

    except Exception as e:
        run.status = "failed"
        run.error = str(e)
        logger.exception("Training failed: %s", e)

    finally:
        if tb_writer is not None:
            tb_writer.close()

    run.end_time = time.time()

    # Write run summary
    summary_path = out_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(run.to_dict(), f, indent=2)

    return run


def collate_fn(batch: list) -> tuple:
    """Custom collate for detection (list of images, list of targets)."""
    images, targets = zip(*batch)
    return list(images), list(targets)

