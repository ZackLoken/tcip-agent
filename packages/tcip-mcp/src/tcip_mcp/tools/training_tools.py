"""Training MCP tools — config validation, launch training, HPO, status."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited

logger = logging.getLogger(__name__)

# Lazy imports of heavy dependencies inside tool functions to keep server startup fast.


@mcp.tool()
@audited
def validate_config(config: dict) -> dict:
    """Validate a training configuration before launching.

    Config structure (new composable format):
        model_spec: {backbone, neck, heads: [{name, task, ...}], loss}
        data: {images_dir, labels_dir, task}
        training: {batch_size, device, stages, mixed_precision, ...}

    Args:
        config: Full training configuration dict.
    """
    from tcip_mcp.pipelines.composer import validate_model_spec

    issues: list[str] = []

    # Model spec validation via composable system
    model_spec = config.get("model_spec") or config.get("model")
    if not model_spec:
        issues.append("Missing 'model_spec' section")
    else:
        spec_issues = validate_model_spec(model_spec)
        issues.extend(spec_issues)

    # Data config validation
    data_cfg = config.get("data")
    if not data_cfg:
        issues.append("Missing 'data' section")
    else:
        for key in ("images_dir", "labels_dir"):
            path = data_cfg.get(key)
            if not path:
                issues.append(f"Missing 'data.{key}'")
            elif not Path(path).is_dir():
                issues.append(f"Directory not found: data.{key} = '{path}'")

    # Training config validation
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 2)
    if not isinstance(batch_size, int) or batch_size < 1:
        issues.append("'training.batch_size' must be a positive integer")

    stages = train_cfg.get("stages", [{"lr": 1e-3, "epochs": 10}])
    for i, stage in enumerate(stages):
        if "lr" not in stage:
            issues.append(f"Stage {i} missing 'lr'")
        if "epochs" not in stage:
            issues.append(f"Stage {i} missing 'epochs'")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
    }


@mcp.tool()
@audited
def launch_training(config: dict, output_dir: str) -> dict:
    """Launch a training run asynchronously using the composable model system.

    The run will proceed in a background thread. Use check_training_status
    to monitor progress.

    Args:
        config: Full training configuration dict with model_spec, data, training sections.
        output_dir: Directory for checkpoints and logs.
    """
    validation = validate_config(config)
    if not validation["valid"]:
        return {"error": "Invalid config", "issues": validation["issues"]}

    from tcip_mcp.pipelines.training.generic_trainer import TrainConfig, train, create_run

    model_spec = config.get("model_spec") or config["model"]
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    train_config = TrainConfig(
        model_spec=model_spec,
        dataset=data_cfg,
        augmentation=config.get("augmentation", {}),
        sampler=config.get("sampler", "random"),
        optimizer=config.get("optimizer", {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 1e-4}),
        stages=train_cfg.get("stages", [{"freeze_to": -1, "epochs": 5}, {"freeze_to": 2, "epochs": 10}]),
        mixed_precision=train_cfg.get("mixed_precision", True),
        batch_size=train_cfg.get("batch_size", 2),
        num_workers=train_cfg.get("num_workers", 0),
    )

    run = create_run(config, output_dir)

    # Determine task from model spec heads
    heads = model_spec.get("heads", [{}])
    task = heads[0].get("task", "detection") if heads else "detection"

    # Build data loaders
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.data.samplers import build_sampler
    from tcip_mcp.pipelines.training.generic_trainer import task_collate
    from torch.utils.data import DataLoader

    # Build augmentation transforms if config specifies them
    aug_config = config.get("augmentation", {})
    transforms = None
    if aug_config:
        from tcip_mcp.pipelines.data.augmentations import build_augmentation
        transforms = build_augmentation(aug_config)

    train_ds = build_dataset(task, images_dir=data_cfg.get("images_dir", ""),
                             labels_dir=data_cfg.get("labels_dir", ""),
                             transforms=transforms)
    sampler = build_sampler(train_config.sampler, train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=train_config.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        collate_fn=task_collate(task),
        num_workers=train_config.num_workers,
    )

    # Auto-create experiment if not already tracked
    experiment_id = config.get("experiment_id") or run.run_id
    try:
        from tcip_mcp.experiments import create_experiment, update_status

        create_experiment(experiment_id, config, data_source=data_cfg.get("images_dir"))
        update_status(experiment_id, "running")
    except Exception:
        pass  # Experiment tracking is best-effort

    thread = threading.Thread(
        target=train, args=(run, train_loader, None, task), daemon=True
    )
    thread.start()

    # Launch TensorBoard for live monitoring
    tb_info = {}
    try:
        from tcip_mcp.pipelines.training.tensorboard_manager import launch_tensorboard
        tb_dir = str(Path(output_dir) / "tensorboard")
        tb_info = launch_tensorboard(tb_dir, run_id=run.run_id)
    except Exception:
        pass  # TensorBoard launch is best-effort

    return {
        "run_id": run.run_id,
        "experiment_id": experiment_id,
        "status": "launched",
        "output_dir": output_dir,
        "tensorboard": tb_info,
    }


@mcp.tool()
@audited
def check_training_status(run_id: str) -> dict:
    """Check the status of a training run.

    Args:
        run_id: Training run identifier.
    """
    from tcip_mcp.pipelines.training.generic_trainer import get_run
    run = get_run(run_id)
    if run is None:
        return {"error": f"Run not found: {run_id}"}

    # Check for running TensorBoard
    tb_url = None
    try:
        from tcip_mcp.pipelines.training.tensorboard_manager import _TB_PROCESSES
        proc = _TB_PROCESSES.get(run_id)
        if proc and proc.poll() is None:
            tb_url = f"http://localhost:{proc._tb_port}"
    except Exception:
        pass

    return {
        "run_id": run.run_id,
        "status": run.status,
        "epoch": run.current_epoch,
        "best_metric": run.best_metric,
        "output_dir": run.output_dir,
        "tensorboard_url": tb_url,
    }


@mcp.tool()
@audited
def list_training_runs() -> dict:
    """List all training runs in this session."""
    from tcip_mcp.pipelines.training.generic_trainer import list_runs
    return {"runs": list_runs()}


@mcp.tool()
@audited
def run_hpo(
    base_config: dict,
    param_space: dict | None = None,
    n_trials: int = 5,
    output_dir: str = "",
    use_optuna: bool = False,
    direction: str = "maximize",
) -> dict:
    """Run hyperparameter optimization with optional TensorBoard logging.

    Two modes:
      - Random search (default): generates trial configs for separate training runs.
      - Optuna (use_optuna=True): runs TPE/ASHA search with per-trial TensorBoard logging.

    TensorBoard logs are written to output_dir/hpo_tensorboard/trial_{n}/ for
    each trial, enabling side-by-side comparison in the TensorBoard HParams plugin.

    Args:
        base_config: Base training config to modify.
        param_space: Dict mapping param names to candidate values (random) or
                    Optuna space dicts (optuna).
        n_trials: Number of trials to generate/run.
        output_dir: Base output directory for trial results.
        use_optuna: If True, use Optuna TPE search with TensorBoard logging.
        direction: 'maximize' (for mAP) or 'minimize' (for loss). Only for Optuna.
    """
    if use_optuna:
        from tcip_mcp.pipelines.training.hpo import optuna_search, get_default_optuna_space

        if param_space is None:
            param_space = get_default_optuna_space()

        tb_logdir = str(Path(output_dir) / "hpo_tensorboard") if output_dir else None

        def objective_fn(trial_params: dict) -> float:
            """Run a full training trial and return the objective metric."""
            merged = _deep_merge(base_config, _param_dict_to_config(trial_params))

            # Build training components
            from tcip_mcp.pipelines.training.generic_trainer import (
                TrainConfig, create_run, train, task_collate,
            )
            from tcip_mcp.pipelines.data.datasets import build_dataset
            from tcip_mcp.pipelines.data.samplers import build_sampler
            from torch.utils.data import DataLoader

            model_spec = merged.get("model_spec") or merged.get("model")
            if not model_spec:
                return float("inf") if direction == "minimize" else 0.0

            data_cfg = merged.get("data", {})
            train_cfg = merged.get("training", {})
            heads = (model_spec.get("heads") or [{}])
            task = heads[0].get("task", "detection") if heads else "detection"

            trial_dir = str(Path(output_dir) / f"trial_{len(_RUNS) if '_RUNS' in dir() else 0}")
            run = create_run(merged, trial_dir)

            try:
                train_ds = build_dataset(
                    task,
                    images_dir=data_cfg.get("images_dir", ""),
                    labels_dir=data_cfg.get("labels_dir", ""),
                )
                sampler = build_sampler(
                    merged.get("sampler", "random"), train_ds
                )
                train_loader = DataLoader(
                    train_ds,
                    batch_size=train_cfg.get("batch_size", trial_params.get("batch_size", 4)),
                    shuffle=(sampler is None),
                    sampler=sampler,
                    collate_fn=task_collate(task),
                    num_workers=train_cfg.get("num_workers", 0),
                )
                train(run, train_loader, None, task=task)

                # Return the best metric from the run
                if direction == "minimize":
                    return run.best_metric if run.best_metric != float("inf") else 999.0
                else:
                    # For maximize, invert since we track best val_loss (lower=better)
                    return -run.best_metric if run.best_metric != float("inf") else 0.0
            except Exception as e:
                logger.warning("HPO trial failed: %s", e)
                return float("inf") if direction == "minimize" else 0.0

        result = optuna_search(
            objective_fn=objective_fn,
            param_space=param_space,
            n_trials=n_trials,
            direction=direction,
            tb_logdir=tb_logdir,
        )

        # Auto-launch TensorBoard for HPO results
        tb_info = {}
        if tb_logdir:
            try:
                from tcip_mcp.pipelines.training.tensorboard_manager import launch_tensorboard
                tb_info = launch_tensorboard(tb_logdir, run_id=f"hpo_{result.get('study_name', 'search')}")
            except Exception:
                pass

        result["tensorboard"] = tb_info
        return result

    # Random search fallback
    from tcip_mcp.pipelines.training.hpo import (
        random_search, validate_param_space, get_default_param_space,
    )
    if param_space is None:
        param_space = get_default_param_space()

    issues = validate_param_space(param_space)
    if issues:
        return {"error": "Invalid param_space", "issues": issues}

    trials = random_search(param_space, n_trials=n_trials)

    # Merge each trial's params into the base config
    configs = []
    for i, trial_params in enumerate(trials):
        config = _deep_merge(base_config, _param_dict_to_config(trial_params))
        config["_trial_id"] = i
        config["_trial_params"] = trial_params
        configs.append(config)

    return {
        "n_trials": len(configs),
        "param_space": param_space,
        "trials": [{"trial_id": c["_trial_id"], "params": c["_trial_params"]} for c in configs],
        "configs": configs,
    }


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _param_dict_to_config(params: dict) -> dict:
    """Map flat HPO params to nested config structure."""
    config: dict = {}
    mapping = {
        "lr": ("training", "stages"),  # handled specially
        "batch_size": ("training", "batch_size"),
        "head": ("model", "head"),
        "weight_decay": ("training", "weight_decay"),
        "min_size": ("model", "min_size"),
    }
    for key, value in params.items():
        if key == "lr":
            # Apply LR to default 3-stage schedule
            stages = [
                {"lr": value, "epochs": 5, "freeze_backbone": True},
                {"lr": value * 0.1, "epochs": 10, "freeze_backbone": False},
                {"lr": value * 0.01, "epochs": 5, "freeze_backbone": False},
            ]
            config.setdefault("training", {})["stages"] = stages
        elif key in mapping:
            parts = mapping[key]
            if len(parts) == 2:
                config.setdefault(parts[0], {})[parts[1]] = value
    return config


@mcp.tool()
@audited
def get_training_metrics_path(run_id: str) -> dict:
    """Return the path to the live metrics JSONL file for a training run.

    The GUI reads this file independently to populate the training dashboard charts.
    Each line is a JSON object with keys: epoch, train_loss, val_loss, map50, lr, stage.

    Args:
        run_id: Training run identifier.
    """
    from tcip_mcp.pipelines.training.generic_trainer import get_run

    run = get_run(run_id)
    if run is None:
        return {"error": f"Run not found: {run_id}"}
    metrics_path = Path(run.output_dir) / "metrics.jsonl"
    return {
        "run_id": run_id,
        "metrics_path": str(metrics_path),
        "exists": metrics_path.is_file(),
    }


@mcp.tool()
@audited
def get_worst_predictions(
    predictions_dir: str,
    labels_dir: str,
    n: int = 8,
) -> dict:
    """Return the N images with the worst prediction quality (highest loss / lowest confidence).

    Compares prediction files to ground-truth labels and ranks images by
    descending error (missed detections + false positives + low confidence).

    Args:
        predictions_dir: Directory with prediction label files (YOLO format).
        labels_dir: Directory with ground-truth label files (YOLO format).
        n: Number of worst images to return.
    """
    pred_path = Path(predictions_dir)
    gt_path = Path(labels_dir)

    if not pred_path.is_dir():
        return {"error": f"Predictions directory not found: {predictions_dir}"}
    if not gt_path.is_dir():
        return {"error": f"Labels directory not found: {labels_dir}"}

    scores: list[tuple[str, float]] = []
    for pred_file in pred_path.glob("*.txt"):
        gt_file = gt_path / pred_file.name
        pred_lines = pred_file.read_text().strip().splitlines()
        gt_lines = gt_file.read_text().strip().splitlines() if gt_file.is_file() else []

        n_pred = len(pred_lines)
        n_gt = len(gt_lines)

        # Simple error heuristic: |pred - gt| + missed + extra + low confidence
        missed = max(0, n_gt - n_pred)
        extra = max(0, n_pred - n_gt)
        avg_conf = 0.0
        if n_pred > 0:
            confs = []
            for line in pred_lines:
                parts = line.split()
                if len(parts) >= 6:  # detection format: cls x y w h conf
                    try:
                        confs.append(float(parts[5]))
                    except ValueError:
                        pass
            avg_conf = sum(confs) / len(confs) if confs else 0.5

        # Higher score = worse prediction
        error_score = missed * 2.0 + extra * 1.0 + (1.0 - avg_conf)
        stem = pred_file.stem
        scores.append((stem, error_score))

    # Also include GT images with no predictions at all (completely missed)
    for gt_file in gt_path.glob("*.txt"):
        pred_file = pred_path / gt_file.name
        if not pred_file.is_file():
            gt_lines = gt_file.read_text().strip().splitlines()
            if gt_lines:
                scores.append((gt_file.stem, len(gt_lines) * 3.0))

    scores.sort(key=lambda x: x[1], reverse=True)
    worst = scores[:n]

    return {
        "worst_images": [{"stem": s, "error_score": round(sc, 3)} for s, sc in worst],
        "total_evaluated": len(scores),
    }
