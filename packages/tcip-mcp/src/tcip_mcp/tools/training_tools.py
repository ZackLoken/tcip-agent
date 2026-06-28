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
    from tcip_mcp.pipelines.schemas import validate_train_config_schema

    # Pydantic schema: type/structure + (via ModelSpecSchema) registry/channel-compat.
    issues: list[str] = list(validate_train_config_schema(config))

    # Model spec presence (keep the exact alias callers/tests rely on).
    model_spec = config.get("model_spec") or config.get("model")
    if not model_spec:
        issues.append("Missing 'model_spec' section")

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
def launch_training(config: dict, output_dir: str, resume_from: str = "") -> dict:
    """Launch a training run asynchronously using the composable model system.

    The run will proceed in a background thread. Use check_training_status
    to monitor progress.

    Args:
        config: Full training configuration dict with model_spec, data, training sections.
        output_dir: Directory for checkpoints and logs.
        resume_from: Optional path to a ``checkpoint_epoch_*.pt`` to resume from
            (restores model + optimizer + scheduler + scaler and continues).
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
        seed=train_cfg.get("seed"),
        deterministic=train_cfg.get("deterministic", False),
    )

    run = create_run(config, output_dir)

    # Determine task from model spec heads
    heads = model_spec.get("heads", [{}])
    task = heads[0].get("task", "detection") if heads else "detection"

    # Build data loaders (build_dataset is used inside _auto_train_val).
    from tcip_mcp.pipelines.data.samplers import build_sampler
    from tcip_mcp.pipelines.training.generic_trainer import task_collate
    from torch.utils.data import DataLoader

    # Build augmentation transforms if config specifies them
    aug_config = config.get("augmentation", {})
    transforms = None
    if aug_config:
        from tcip_mcp.pipelines.data.augmentations import build_augmentation
        transforms = build_augmentation(aug_config)

    train_ds, val_ds = _auto_train_val(task, data_cfg, transforms)
    sampler = build_sampler(train_config.sampler, train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=train_config.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        collate_fn=task_collate(task),
        num_workers=train_config.num_workers,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=train_config.batch_size,
            shuffle=False,
            collate_fn=task_collate(task),
            num_workers=train_config.num_workers,
        )

    # W8: inject imbalance loss + (auto) class weights into image-level head specs.
    # train() composes the model from run.config["model_spec"] (== config), so editing
    # the head specs here before the thread starts reaches the model.
    loss_cfg = config.get("loss")
    if isinstance(loss_cfg, dict):
        cw = loss_cfg.get("class_weights")
        for h in model_spec.get("heads", []):
            if h.get("name") not in ("classification", "semantic_seg"):
                continue
            h["loss"] = loss_cfg.get("name", "weighted_ce")
            if cw == "auto":
                from tcip_mcp.pipelines.components.losses import compute_class_weights
                weights = compute_class_weights(
                    getattr(train_ds, "class_distribution", {}),
                    num_classes=h.get("num_classes"),
                    scheme=loss_cfg.get("weight_scheme", "balanced"),
                )
                h["class_weights"] = weights.tolist()
            elif isinstance(cw, list):
                h["class_weights"] = cw

    # Auto-create experiment if not already tracked
    experiment_id = config.get("experiment_id") or run.run_id
    try:
        from tcip_mcp.experiments import create_experiment, update_status

        create_experiment(experiment_id, config, data_source=data_cfg.get("images_dir"))
        update_status(experiment_id, "running")
    except Exception:
        pass  # Experiment tracking is best-effort

    thread = threading.Thread(
        target=train, args=(run, train_loader, val_loader, task),
        kwargs={"resume_from": resume_from}, daemon=True,
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
    pruner: str = "asha",
    grace_period: int = 5,
    reduction_factor: int = 3,
    warm_start: bool = False,
    baseline_params: dict | None = None,
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

        import optuna

        def objective_fn(trial_params: dict, trial) -> float:
            """Run a full training trial; report per-epoch for ASHA pruning.

            ``run.best_metric`` is the composite selection objective (W1, lower=better)
            for detection when a val_loader exists, else val/train loss; ``maximize``
            inverts it (``-best_metric``), so the existing direction handling stays correct.
            """
            merged = _deep_merge(base_config, _param_dict_to_config(trial_params))

            from tcip_mcp.pipelines.training.generic_trainer import create_run, train, task_collate
            from tcip_mcp.pipelines.data.samplers import build_sampler
            from torch.utils.data import DataLoader

            model_spec = merged.get("model_spec") or merged.get("model")
            if not model_spec:
                return float("inf") if direction == "minimize" else 0.0

            data_cfg = merged.get("data", {})
            train_cfg = merged.get("training", {})
            heads = (model_spec.get("heads") or [{}])
            task = heads[0].get("task", "detection") if heads else "detection"

            trial_dir = str(Path(output_dir) / f"trial_{trial.number}")
            run = create_run(merged, trial_dir)

            try:
                # W4 auto-val gives the val_loader that W1's composite / ASHA need.
                train_ds, val_ds = _auto_train_val(task, data_cfg, None)
                sampler = build_sampler(merged.get("sampler", "random"), train_ds)
                batch_size = train_cfg.get("batch_size", trial_params.get("batch_size", 4))
                num_workers = train_cfg.get("num_workers", 0)
                train_loader = DataLoader(
                    train_ds, batch_size=batch_size, shuffle=(sampler is None),
                    sampler=sampler, collate_fn=task_collate(task), num_workers=num_workers,
                )
                val_loader = None
                if val_ds is not None:
                    val_loader = DataLoader(
                        val_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=task_collate(task), num_workers=num_workers,
                    )

                def epoch_cb(epoch: int, metrics: dict) -> None:
                    value = metrics.get("val_objective", metrics.get("val_loss"))
                    if value is None:
                        return
                    trial.report(value, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                train(run, train_loader, val_loader, task=task, epoch_callback=epoch_cb)

                if direction == "minimize":
                    return run.best_metric if run.best_metric != float("inf") else 999.0
                return -run.best_metric if run.best_metric != float("inf") else 0.0
            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.warning("HPO trial failed: %s", e)
                return float("inf") if direction == "minimize" else 0.0

        result = optuna_search(
            objective_fn=objective_fn,
            param_space=param_space,
            n_trials=n_trials,
            direction=direction,
            pruner=pruner,
            grace_period=grace_period,
            reduction_factor=reduction_factor,
            warm_start=warm_start,
            baseline_params=baseline_params,
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


def _auto_train_val(task: str, data_cfg: dict, transforms):
    """Build ``(train_ds, val_ds)`` for a run, deriving a leakage-free val split.

    Resolution order:
      1. ``data.val_images_dir`` set -> build val from it explicitly.
      2. ``data.auto_val`` (default True) and a stem-capable task
         (detection / instance_seg / semantic_seg / classification) -> derive a
         group-aware train/val split (no held-out test) so the trainer receives
         a real validation loader. Train keeps augmentation; val gets none.
      3. ordinal / regression, ``auto_val`` disabled, a tiny/single-group set, or
         any failure -> ``(full_train_ds, None)``. Never raises into the caller.

    Reads ``auto_val`` / ``val_*`` / ``split.*`` from ``data_cfg`` (== config["data"]).
    """
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.data.splits import (
        group_balanced_split, count_label_lines, GROUP_KEY_FNS, default_group_key,
    )

    STEM_TASKS = {"detection", "instance_seg", "semantic_seg", "classification"}

    def _source_kwargs() -> dict:
        if task in ("detection", "instance_seg"):
            return {"images_dir": data_cfg.get("images_dir", ""),
                    "labels_dir": data_cfg.get("labels_dir", "")}
        if task == "semantic_seg":
            return {"images_dir": data_cfg.get("images_dir", ""),
                    "masks_dir": data_cfg.get("masks_dir", data_cfg.get("labels_dir", ""))}
        kw = {"images_dir": data_cfg.get("images_dir", "")}
        if data_cfg.get("csv_path"):
            kw["csv_path"] = data_cfg["csv_path"]
        return kw

    src = _source_kwargs()
    tiling = data_cfg.get("tiling")  # W3: detection tiling (None for other tasks/configs)

    # 1. Explicit validation source.
    val_images = data_cfg.get("val_images_dir")
    if val_images:
        try:
            train_ds = build_dataset(task, **src, transforms=transforms, tiling=tiling)
            val_src = dict(src)
            val_src["images_dir"] = val_images
            if task in ("detection", "instance_seg"):
                val_src["labels_dir"] = data_cfg.get("val_labels_dir", data_cfg.get("labels_dir", ""))
            elif task == "semantic_seg":
                val_src["masks_dir"] = data_cfg.get("val_masks_dir", data_cfg.get("masks_dir", ""))
            return train_ds, build_dataset(task, **val_src, transforms=None, tiling=tiling)
        except Exception as exc:
            logger.warning("Explicit val build failed (%s); training without validation.", exc)
            return build_dataset(task, **src, transforms=transforms, tiling=tiling), None

    if not data_cfg.get("auto_val", True) or task not in STEM_TASKS:
        return build_dataset(task, **src, transforms=transforms, tiling=tiling), None

    # 2. Auto group-aware train/val split.
    try:
        full_ds = build_dataset(task, **src, transforms=transforms)
        stems = list(getattr(full_ds, "stems", None) or getattr(full_ds, "_stems", []))
        if len(stems) < 2:
            return full_ds, None

        split_cfg = data_cfg.get("split", {})
        val_ratio = float(split_cfg.get("val_ratio", 0.2))
        seed = int(split_cfg.get("seed", 42))
        group_by = split_cfg.get("group_by", "tile_prefix")
        stratify = split_cfg.get("stratify_foreground", True)
        group_key_fn = GROUP_KEY_FNS.get(group_by, default_group_key)

        annotation_counts = None
        if stratify and task in ("detection", "instance_seg"):
            labels_dir = data_cfg.get("labels_dir", "")
            annotation_counts = {s: count_label_lines(labels_dir, s) for s in stems}

        parts = group_balanced_split(
            stems, annotation_counts=annotation_counts, group_key_fn=group_key_fn,
            splits=(1.0 - val_ratio, val_ratio, 0.0), seed=seed,
        )
        train_stems, val_stems = parts["train"], parts["val"]
        if not val_stems or not train_stems:
            return full_ds, None

        if task == "classification":
            stem_to_label = dict(zip(getattr(full_ds, "_stems", []), getattr(full_ds, "_labels", [])))
            train_ds = build_dataset(
                task, images_dir=src["images_dir"], transforms=transforms,
                stems=train_stems, labels=[stem_to_label[s] for s in train_stems])
            val_ds = build_dataset(
                task, images_dir=src["images_dir"], transforms=None,
                stems=val_stems, labels=[stem_to_label[s] for s in val_stems])
        else:
            train_ds = build_dataset(task, **src, transforms=transforms, stems=train_stems, tiling=tiling)
            val_ds = build_dataset(task, **src, transforms=None, stems=val_stems, tiling=tiling)
        logger.info("Auto train/val split for %s: %d train / %d val stems.",
                    task, len(train_stems), len(val_stems))
        return train_ds, val_ds
    except Exception as exc:
        logger.warning("Auto train/val split failed (%s); training without validation.", exc)
        return build_dataset(task, **src, transforms=transforms, tiling=tiling), None


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


@mcp.tool()
@audited
def evaluate_model(
    run_id_or_ckpt: str,
    images_dir: str,
    labels_dir: str = "",
    task: str = "detection",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    iou_type: str = "bbox",
    max_dets: int = 100,
) -> dict:
    """Evaluate a trained checkpoint on a (held-out) dataset and write test_results.json.

    Computes the same per-task metrics as validation — detection/instance_seg get
    pycocotools mAP + precision/recall/F1; classification/ordinal/regression get the
    in-house scalar metrics — and writes ``test_results.json`` beside the checkpoint.

    Args:
        run_id_or_ckpt: A training run id (uses its ``model_best.pt``) or a checkpoint path.
        images_dir: Images directory for the evaluation split.
        labels_dir: Labels dir (detection/instance_seg) or masks dir (semantic_seg).
        task: Task type.
        conf_threshold: Operating confidence for P/R/F1.
        iou_threshold: Operating IoU (on COCOeval's grid; 0.5 -> index 0).
        iou_type: 'bbox' or 'segm'.
        max_dets: COCOeval max detections per image.
    """
    import torch
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.training.generic_trainer import get_run, task_collate
    from tcip_mcp.pipelines.training.evaluation import run_test_evaluation
    from tcip_mcp.pipelines.data.datasets import build_dataset

    ckpt = run_id_or_ckpt
    if not Path(ckpt).is_file():
        run = get_run(run_id_or_ckpt)
        if run is None:
            return {"error": f"Not a checkpoint path or known run id: {run_id_or_ckpt}"}
        ckpt = str(Path(run.output_dir) / "model_best.pt")
    if not Path(ckpt).is_file():
        return {"error": f"Checkpoint not found: {ckpt}"}

    ds_kwargs = {"images_dir": images_dir}
    if task in ("detection", "instance_seg"):
        ds_kwargs["labels_dir"] = labels_dir
    elif task == "semantic_seg":
        ds_kwargs["masks_dir"] = labels_dir
    try:
        dataset = build_dataset(task, **ds_kwargs)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Failed to build dataset: {exc}"}

    loader = DataLoader(dataset, batch_size=4, collate_fn=task_collate(task))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return run_test_evaluation(
        ckpt, loader, device, task, str(Path(ckpt).parent),
        conf_threshold=conf_threshold, iou_threshold=iou_threshold,
        iou_type=iou_type, max_dets=max_dets,
    )
