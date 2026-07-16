"""Training MCP tools — config validation, launch training, HPO, status."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited
from tcip_mcp.pipelines.resolution import DEFAULT_CONF

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

    # Per-stage 'epochs' is required; 'lr' is optional (StageSpec) and the trainer
    # reads learning rates from config['optimizer'], never from a stage. Absent
    # stages are fine — launch_training supplies its own default schedule.
    for i, stage in enumerate(train_cfg.get("stages") or []):
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

    # Canonicalize the shape: the GUI/validated schema nests stages/mixed_precision/batch_size
    # under ``training``, but the trainer reads them from the top level of run.config — without
    # this hoist a GUI-launched run silently trains the default single stage. (run_hpo already
    # normalizes inside _apply_hpo_params.)
    from tcip_mcp.pipelines.schemas import normalize_train_config
    config = normalize_train_config(config)

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
    # Nest each run's artifacts under its run_id. The GUI (and typical callers) pass a
    # shared base such as ``<project>/.tcip/experiments``; without nesting, sequential
    # runs write ``metrics.jsonl`` / ``model_best.pt`` to the *same* flat directory and
    # clobber each other — violating experiment immutability. Nesting also makes the
    # trainer write exactly where the web metrics stream reads (``<base>/<run_id>/``).
    run.output_dir = str(Path(output_dir) / run.run_id)

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

    # CV2: persist the EFFECTIVE tiling geometry (the 224/0.2 defaults used when the tiling dict
    # omitted them, not just caller-pinned values) into config["data"]["tiling"], so every checkpoint
    # + the experiment config carry the true tile scale and inference can derive it. data_cfg is the
    # same object as config["data"], so this lands in the snapshot recorded below.
    tiling_cfg = data_cfg.get("tiling")
    if tiling_cfg and tiling_cfg.get("enabled", True):
        eff_tile = getattr(train_ds, "tile_size", None)
        eff_overlap = getattr(train_ds, "overlap", None)
        if eff_tile is not None:
            tiling_cfg["tile_size"] = int(eff_tile)
        if eff_overlap is not None:
            tiling_cfg["overlap"] = float(eff_overlap)

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
    if val_loader is None and task in ("detection", "instance_seg"):
        logger.warning(
            "No validation loader for %s run %s: best-model selection and early "
            "stopping will fall back to training loss (no val mAP/composite). "
            "Provide a val split (data.val_images_dir) or enable auto_val.",
            task, run.run_id,
        )

    # Fill spec params the agent didn't pin (in_chans from the raster, num_classes from the labels;
    # explicit wins) at the same pre-compose seam as _inject_imbalance_loss. A hiccup mustn't fail the launch.
    try:
        from tcip_mcp.pipelines.derivations import resolve_spec_derivations
        _base_ds = getattr(train_ds, "base", train_ds)
        _img_dir = getattr(_base_ds, "images_dir", None) or data_cfg.get("images_dir")
        _sample = None
        if _img_dir:
            for _p in sorted(Path(str(_img_dir)).glob("*")):
                if _p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
                    _sample = str(_p)
                    break
        # union of train+val so a class landing only in val isn't dropped from the head
        _cd = dict(getattr(train_ds, "class_distribution", None) or {})
        _val_cd = getattr(val_ds, "class_distribution", None) if val_ds is not None else None
        for _k, _v in (_val_cd or {}).items():
            _cd[_k] = _cd.get(_k, 0) + _v
        resolve_spec_derivations(model_spec, sample_image=_sample, class_distribution=_cd or None)
    except Exception:
        logger.debug("spec derivation skipped", exc_info=True)

    # W8: inject imbalance loss + (auto) class weights into image-level head specs.
    # train() composes the model from run.config["model_spec"] (== config), so editing
    # the head specs here before the thread starts reaches the model.
    _inject_imbalance_loss(config.get("loss"), model_spec, train_ds)

    # Auto-create experiment if not already tracked. Experiments are immutable:
    # reusing an id that already has a run would interleave metrics histories and
    # overwrite lineage/registry entries, so such relaunches get a fresh id.
    experiment_id = config.get("experiment_id") or run.run_id
    try:
        from tcip_mcp.experiments import update_status

        experiment_id = _ensure_experiment(
            experiment_id, config, data_cfg.get("images_dir"), resume_from, run.run_id,
        )
        update_status(experiment_id, "running")
    except Exception as exc:  # Experiment tracking is best-effort, but failures must be visible.
        logger.warning("Experiment tracking failed for %s: %s", experiment_id, exc)

    def _run_with_tracking() -> None:
        """Run training and wire its lifecycle into the experiment + model registry."""
        from tcip_mcp.experiments import (
            log_metrics, register_model_from_experiment, update_status,
        )

        def _epoch_cb(epoch: int, epoch_metrics: dict) -> None:
            try:
                log_metrics(experiment_id, epoch, epoch_metrics)
            except Exception as exc:
                logger.warning("Experiment metric log failed (%s epoch %s): %s", experiment_id, epoch, exc)

        train(run, train_loader, val_loader, task, epoch_callback=_epoch_cb, resume_from=resume_from)

        # train() set run.status to completed/failed (it does not re-raise normal failures).
        try:
            if run.status == "completed":
                out = Path(run.output_dir)
                best = out / "model_best.pt"
                weights = str(best if best.is_file() else out / "model_final.pt")
                update_status(experiment_id, "completed")
                register_model_from_experiment(experiment_id, weights)
            else:
                update_status(experiment_id, run.status or "failed")  # "failed" or "cancelled"
        except Exception as exc:
            logger.warning("Experiment completion wiring failed for %s: %s", experiment_id, exc)

    thread = threading.Thread(target=_run_with_tracking, daemon=True)
    thread.start()

    # Launch TensorBoard for live monitoring
    tb_info = {}
    try:
        from tcip_mcp.pipelines.training.tensorboard_manager import launch_tensorboard
        tb_dir = str(Path(run.output_dir) / "tensorboard")
        tb_info = launch_tensorboard(tb_dir, run_id=run.run_id)
    except Exception:
        pass  # TensorBoard launch is best-effort

    return {
        "run_id": run.run_id,
        "experiment_id": experiment_id,
        "status": "launched",
        "output_dir": run.output_dir,
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
def cancel_training(run_id: str) -> dict:
    """Request graceful cancellation of a running training run.

    The trainer stops at the next batch/epoch boundary, still saves ``model_final.pt``
    (so partial progress is recoverable), and sets the run + its experiment to
    'cancelled'. Status updates asynchronously, so the returned status may still read
    'running' immediately after the request.

    Args:
        run_id: Training run identifier (from launch_training).
    """
    from tcip_mcp.pipelines.training.generic_trainer import cancel_run, get_run
    if not cancel_run(run_id):
        return {"error": f"Run not found: {run_id}"}
    return {"run_id": run_id, "status": get_run(run_id).status, "cancel_requested": True}


@mcp.tool()
@audited
def run_hpo(
    base_config: dict,
    param_space: dict | None = None,
    n_trials: int = 5,
    output_dir: str = "",
    direction: str = "maximize",
    pruner: str = "asha",
    grace_period: int = 5,
    reduction_factor: int = 3,
    warm_start: bool = False,
    baseline_params: dict | None = None,
) -> dict:
    """Run Optuna hyperparameter optimization with per-trial TensorBoard logging.

    Runs a TPE/ASHA search that actually trains each trial and reports per-epoch for pruning.
    (The former ``use_optuna=False`` random-search branch was removed — it only *enumerated*
    trial configs without training, so it could never select hyperparameters; the agent can
    assemble and launch configs itself for a manual sweep.)

    TensorBoard logs are written to output_dir/hpo_tensorboard/trial_{n}/ for
    each trial, enabling side-by-side comparison in the TensorBoard HParams plugin.

    Args:
        base_config: Base training config to modify.
        param_space: Dict mapping param names to Optuna space dicts. Defaults to the
            built-in space when omitted.
        n_trials: Number of trials to run.
        output_dir: Base output directory for trial results.
        direction: 'maximize' (for mAP) or 'minimize' (for loss).
    """
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
        merged = _apply_hpo_params(base_config, trial_params)

        from tcip_mcp.pipelines.training.generic_trainer import create_run, train, task_collate
        from tcip_mcp.pipelines.data.samplers import build_sampler
        from torch.utils.data import DataLoader

        model_spec = merged.get("model_spec") or merged.get("model")
        if not model_spec:
            # Worst-possible value in either direction — a dead trial must never
            # outrank a real one (0.0 beat every -composite under maximize).
            return float("inf") if direction == "minimize" else float("-inf")

        data_cfg = merged.get("data", {})
        train_cfg = merged.get("training", {})
        heads = (model_spec.get("heads") or [{}])
        task = heads[0].get("task", "detection") if heads else "detection"

        trial_dir = str(Path(output_dir) / f"trial_{trial.number}")
        # Tag as an HPO trial so it stays out of the Training-tab run list.
        run = create_run(merged, trial_dir, origin="hpo_trial")

        try:
            # Trials must train under the same regime as the final launch_training
            # run they tune for: same augmentation and same imbalance loss/class
            # weights — otherwise the selected hyperparameters don't transfer.
            transforms = None
            aug_cfg = merged.get("augmentation", {})
            if aug_cfg:
                from tcip_mcp.pipelines.data.augmentations import build_augmentation
                transforms = build_augmentation(aug_cfg)

            # W4 auto-val gives the val_loader that W1's composite / ASHA need.
            train_ds, val_ds = _auto_train_val(task, data_cfg, transforms)
            _inject_imbalance_loss(merged.get("loss"), model_spec, train_ds)
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
                # value is lower=better; report in the study's direction (matching
                # the -best_metric final return) so ASHA keeps the improving
                # trials — raw reports under maximize pruned the *best* trials.
                trial.report(-value if direction == "maximize" else value, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            train(run, train_loader, val_loader, task=task, epoch_callback=epoch_cb)

            # best_metric is lower=better; inf (train() caught a failure or never
            # produced a metric) maps to the worst-possible value in either
            # direction so a dead trial can never become the study's best.
            return run.best_metric if direction == "minimize" else -run.best_metric
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.warning("HPO trial failed: %s", e)
            return float("inf") if direction == "minimize" else float("-inf")

    # Persist the study to sqlite + a result file so a restart doesn't lose the trials
    # (the web sweep was ephemeral in-memory — Optuna defaulted to storage=None). One
    # uniquely-named study per call, under output_dir or the platform state root.
    import uuid

    from tcip_mcp.project_paths import project_root

    hpo_dir = Path(output_dir) if output_dir else project_root() / ".tcip" / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    study_name = f"hpo_{uuid.uuid4().hex[:8]}"
    storage = f"sqlite:///{(hpo_dir / 'hpo.db').as_posix()}"  # as_posix so Windows paths are valid URLs

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
        study_name=study_name,
        storage=storage,
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
    result["storage"] = storage
    # Durable result file alongside the sqlite study (best-effort — a write hiccup must not
    # sink a completed sweep).
    try:
        from tcip_mcp.utils.atomic_io import atomic_write_json
        atomic_write_json(hpo_dir / f"{study_name}.json", result)
    except Exception:
        logger.warning("could not persist hpo result json for %s", study_name, exc_info=True)
    return result


def _apply_hpo_params(base_config: dict, params: dict) -> dict:
    """Apply flat HPO params onto a deep copy of ``base_config``, in the right places.

    A param only counts if it lands where ``generic_trainer.train()`` actually reads
    it. Architecture params go into ``model_spec`` (the prior implementation dropped
    ``backbone`` entirely and wrote ``head`` / ``min_size`` to an unread
    ``config["model"]`` key, so every trial trained the same model). Optimizer params
    go into TOP-LEVEL ``config["optimizer"]`` and the unfreeze schedule into
    TOP-LEVEL ``config["stages"]`` — the trainer reads both from the top level and
    never reads ``training["stages"]`` or a stage's ``lr`` (the prior implementation
    wrote there, so every lr/weight_decay trial trained identically):

      - ``backbone``     -> ``model_spec["backbone"]["name"]``
      - ``head``         -> ``model_spec["heads"][0]["detector"]`` (faster_rcnn/fcos/retinanet)
      - ``min_size``     -> ``model_spec["heads"][0]["min_size"]``
      - ``lr``           -> ``optimizer["head_lr"]`` (+ ``backbone_lr`` at lr*0.1) plus a
                            3-stage ``freeze_to`` progressive-unfreeze schedule
      - ``weight_decay`` -> ``optimizer["weight_decay"]``
      - anything else (``batch_size``, ...) -> ``training``
    """
    import copy

    from tcip_mcp.pipelines.schemas import normalize_train_config

    cfg = normalize_train_config(copy.deepcopy(base_config))
    spec = cfg.get("model_spec")
    training = cfg.setdefault("training", {})

    def _first_head() -> dict:
        heads = spec.get("heads") or [{}]
        if not isinstance(heads[0], dict):
            heads[0] = {}
        spec["heads"] = heads
        return heads[0]

    for key, value in params.items():
        if key == "lr":
            lr = float(value)
            optimizer = cfg.setdefault("optimizer", {})
            optimizer["head_lr"] = lr
            optimizer["backbone_lr"] = lr * 0.1
            # 10 epochs total, matching the trainer's previous per-trial budget so a
            # sweep's runtime doesn't double. No per-stage 'lr' keys: the trainer
            # applies head_lr/backbone_lr per stage and ignores stage-level lr.
            cfg["stages"] = [
                {"freeze_to": -1, "epochs": 3},  # heads only
                {"freeze_to": 2, "epochs": 4},   # unfreeze top stages
                {"freeze_to": 0, "epochs": 3},   # full fine-tune
            ]
        elif key == "batch_size":
            training["batch_size"] = value
        elif key == "weight_decay":
            cfg.setdefault("optimizer", {})["weight_decay"] = value
        elif spec is None:
            # No model spec to mutate — keep scalar params on training as a fallback.
            training[key] = value
        elif key == "backbone":
            bb = spec.get("backbone")
            if isinstance(bb, dict):
                bb["name"] = value
            else:
                spec["backbone"] = {"name": value}
        elif key == "head":
            _first_head()["detector"] = value
        elif key == "min_size":
            _first_head()["min_size"] = value
        else:
            training[key] = value
    return cfg


def _ensure_experiment(
    experiment_id: str, config: dict, data_source, resume_from: str, run_id: str,
) -> str:
    """Create or attach the experiment for a run, enforcing experiment immutability.

    Returns the experiment id actually used. An existing id may be reused only when
    the experiment is pristine (agent pre-created it: state 'created', no metrics)
    or when ``resume_from`` continues that experiment's own checkpoint. Anything
    else mints a fresh ``<id>_<run_id>`` (with the old id as parent lineage) so the
    prior run's status, metrics, lineage, and registry entry stay intact.
    """
    from tcip_mcp.experiments import create_experiment, get_experiment

    created = create_experiment(experiment_id, config, data_source=data_source)
    if "error" not in created:
        return experiment_id

    existing = get_experiment(experiment_id, metrics_limit=1)
    pristine = (
        existing.get("status", {}).get("state") == "created"
        and not existing.get("n_epochs")
    )
    if pristine or resume_from:
        return experiment_id

    fresh_id = f"{experiment_id}_{run_id}"
    logger.warning(
        "experiment_id %s already has a run; experiments are immutable — tracking "
        "this run as %s instead.", experiment_id, fresh_id,
    )
    create_experiment(fresh_id, config, parent_experiment=experiment_id, data_source=data_source)
    return fresh_id


def _inject_imbalance_loss(loss_cfg, model_spec: dict, train_ds) -> None:
    """W8: inject imbalance loss + (auto) class weights into image-level head specs.

    Mutates ``model_spec["heads"]`` in place — the trainer composes the model from
    the run config's ``model_spec``, so editing head specs before train() starts
    reaches the model. The 'auto' scheme reads the built train dataset's
    ``class_distribution``, so this must run after the dataset is built. No-op for
    detection heads and non-dict ``loss_cfg``.
    """
    if not isinstance(loss_cfg, dict):
        return
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
            kw = {"images_dir": data_cfg.get("images_dir", ""),
                  "labels_dir": data_cfg.get("labels_dir", "")}
            # Thread the on-disk label format through to the dataset. Dropping it here
            # silently defaults to YOLO parsing, so a VOC/LabelMe/COCO dataset reads as
            # all-empty negatives — the undetected-format mismatch CLAUDE.md warns about.
            if data_cfg.get("label_format"):
                kw["label_format"] = data_cfg["label_format"]
            if data_cfg.get("coco_json"):
                kw["coco_json"] = data_cfg["coco_json"]
            return kw
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
        # Backend#3: assemble the dataset-level COCO ONCE (JSON detection labels) and thread it into
        # the full + train + val builds below, instead of re-assembling the same COCO three times.
        # Annotations are matched by image file name, so the full COCO is correct for any stem subset.
        build_src = dict(src)
        if task in ("detection", "instance_seg") and not (
            data_cfg.get("label_format") or data_cfg.get("coco_json")
        ):
            from tcip_mcp.pipelines.data.datasets import assemble_coco, dir_label_format
            _labels, _images = src.get("labels_dir", ""), src.get("images_dir", "")
            if _labels and _images and dir_label_format(_labels) == "json":
                build_src["coco_data"] = assemble_coco(_labels, _images)
                build_src["label_format"] = "coco"

        full_ds = build_dataset(task, **build_src, transforms=transforms)
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
            train_ds = build_dataset(task, **build_src, transforms=transforms, stems=train_stems, tiling=tiling)
            val_ds = build_dataset(task, **build_src, transforms=None, stems=val_stems, tiling=tiling)
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
    top_k: int = 8,
) -> dict:
    """Return the ``top_k`` images ranked worst by a count-mismatch + low-confidence triage heuristic.

    This is a cheap triage signal, not a quality metric: it does no IoU matching and computes
    no loss. The score is ``2·|n_gt−n_pred as a shortfall| + |surplus| + (1−avg_conf)`` — purely
    the difference in box *counts* plus mean confidence, so an image with the right count but
    every box mislocated scores as good. Use it to surface likely-bad frames for a human to look
    at; for true TP/FP/FN ranking use ``evaluate_detections`` (``detail=True``, IoU-matched).

    Args:
        predictions_dir: Directory with per-image JSON prediction files
            (``<stem>.json``) written by run_inference / the review engine.
        labels_dir: Directory with per-image JSON ground-truth label files.
        top_k: Number of worst images to return.
    """
    pred_path = Path(predictions_dir)
    gt_path = Path(labels_dir)

    if not pred_path.is_dir():
        return {"error": f"Predictions directory not found: {predictions_dir}"}
    if not gt_path.is_dir():
        return {"error": f"Labels directory not found: {labels_dir}"}

    from tcip_annotation import json_io

    scores: list[tuple[str, float]] = []
    for pred_file in pred_path.glob("*.json"):
        gt_file = gt_path / pred_file.name
        preds, _ = json_io.read_detect_pred(pred_file)
        gt_boxes, _ = json_io.read_detect(gt_file) if gt_file.is_file() else ([], set())

        n_pred = len(preds)
        n_gt = len(gt_boxes)

        # Simple error heuristic: |pred - gt| + missed + extra + low confidence
        missed = max(0, n_gt - n_pred)
        extra = max(0, n_pred - n_gt)
        avg_conf = 0.0
        if n_pred > 0:
            confs = [p.confidence for p in preds]
            avg_conf = sum(confs) / len(confs) if confs else 0.5

        # Higher score = worse prediction
        error_score = missed * 2.0 + extra * 1.0 + (1.0 - avg_conf)
        stem = pred_file.stem
        scores.append((stem, error_score))

    # Also include GT images with no predictions at all (completely missed)
    for gt_file in gt_path.glob("*.json"):
        pred_file = pred_path / gt_file.name
        if not pred_file.is_file():
            gt_boxes, _ = json_io.read_detect(gt_file)
            if gt_boxes:
                scores.append((gt_file.stem, len(gt_boxes) * 3.0))

    scores.sort(key=lambda x: x[1], reverse=True)
    worst = scores[:top_k]

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
    conf_threshold: float = DEFAULT_CONF,  # report/select at the ship point
    iou_threshold: float = 0.5,
    iou_type: str | None = None,
    max_dets: int = 100,
    tiling: dict | None = None,
    use_tiled_inference: bool = False,
) -> dict:
    """Evaluate a trained checkpoint on a (held-out) dataset and write test_results.json.

    Computes the same per-task metrics as validation — detection/instance_seg get
    pycocotools mAP + precision/recall/F1; classification/ordinal/regression get the
    in-house scalar metrics — and writes ``test_results.json`` beside the checkpoint.

    Two detection eval regimes (CV1):
      * default (single full-res forward pass) or ``tiling`` set -> tile-level DIAGNOSTIC that
        matches the training-run val mAP. When ``run_id_or_ckpt`` is a run id and ``tiling`` is
        left None, the run's own training tiling is reused so held-out eval matches that regime;
        an explicit checkpoint PATH stays untiled unless ``tiling`` is passed.
      * ``use_tiled_inference=True`` -> the delivery-grade full-frame metric (tiled inference
        reconstructed to full frame, matched to full-frame GT). Report THIS to gate a delivery.

    Args:
        run_id_or_ckpt: A training run id (uses its ``model_best.pt``) or a checkpoint path.
        images_dir: Images directory for the evaluation split.
        labels_dir: Labels dir (detection/instance_seg) or masks dir (semantic_seg).
        task: Task type.
        conf_threshold: Operating confidence for P/R/F1.
        iou_threshold: Operating IoU (on COCOeval's grid; 0.5 -> index 0).
        iou_type: 'bbox' or 'segm'. Default (None) auto-resolves from the task — 'segm' for
            instance_seg, 'bbox' otherwise — so a mask model isn't silently scored as boxes.
        max_dets: COCOeval max detections per image.
        tiling: Optional detection tiling dict ({enabled, tile_size, overlap, ...}) for a
            tile-level eval. None + a run id reuses the run's training tiling; None + a
            checkpoint path stays untiled.
        use_tiled_inference: Score the delivery regime (full-frame via tiled inference).
    """
    import torch
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.training.generic_trainer import get_run, task_collate
    from tcip_mcp.pipelines.training.evaluation import (
        run_full_frame_evaluation, run_test_evaluation,
    )
    from tcip_mcp.pipelines.data.datasets import build_dataset

    ckpt = run_id_or_ckpt
    run = None
    if not Path(ckpt).is_file():
        run = get_run(run_id_or_ckpt)
        if run is None:
            return {"error": f"Not a checkpoint path or known run id: {run_id_or_ckpt}"}
        ckpt = str(Path(run.output_dir) / "model_best.pt")
    if not Path(ckpt).is_file():
        return {"error": f"Checkpoint not found: {ckpt}"}

    run_tiling = (run.config.get("data", {}) or {}).get("tiling") if run is not None else None

    # Delivery-grade full-frame path (tiled inference + full-frame GT matching).
    if use_tiled_inference and task == "detection":
        tcfg = tiling or run_tiling or {}
        return run_full_frame_evaluation(
            ckpt, images_dir, labels_dir, str(Path(ckpt).parent),
            conf_threshold=conf_threshold, iou_threshold=iou_threshold,
            tile_size=int(tcfg.get("tile_size", 640)), overlap=float(tcfg.get("overlap", 0.2)),
            max_dets=max_dets if max_dets > 100 else 1000,
        )

    # Tile-level diagnostic (or untiled). Only detection tiles; a run id reuses its training tiling.
    if tiling is None and run is not None:
        tiling = run_tiling
    if task != "detection":
        tiling = None

    ds_kwargs = {"images_dir": images_dir}
    if task in ("detection", "instance_seg"):
        ds_kwargs["labels_dir"] = labels_dir
    elif task == "semantic_seg":
        ds_kwargs["masks_dir"] = labels_dir
    try:
        dataset = build_dataset(task, **ds_kwargs, tiling=tiling)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Failed to build dataset: {exc}"}

    loader = DataLoader(dataset, batch_size=4, collate_fn=task_collate(task))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return run_test_evaluation(
        ckpt, loader, device, task, str(Path(ckpt).parent),
        conf_threshold=conf_threshold, iou_threshold=iou_threshold,
        iou_type=iou_type, max_dets=max_dets, tiling=tiling,
    )
