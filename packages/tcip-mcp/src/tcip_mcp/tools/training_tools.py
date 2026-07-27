"""Training MCP tools — config validation, launch training, HPO, status."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited
from tcip_mcp.pipelines.resolution import DEFAULT_CONF, DEFAULT_NMS_IOU

logger = logging.getLogger(__name__)

# Lazy imports of heavy dependencies inside tool functions to keep server startup fast.


@mcp.tool()
@audited
def preflight_config(config: dict, smoke: bool = False, overfit: bool = False) -> dict:
    """Validate a training configuration before launching.

    Config structure:
        model_source: {builder, builder_kwargs, task, in_chans}
        data: {images_dir, labels_dir, task}  # known loaders, OR a bespoke
              # {dataset_source: {builder, builder_kwargs, source_files, task}, task}
        training: {batch_size, device, stages, mixed_precision, ...}

    Args:
        config: Full training configuration dict.
        smoke: When True, actually build the model and run ``check_model_contract`` (a train+eval
            forward at the resolved in_chans/num_classes/img_size). A contract failure is a
            guaranteed real-run failure, so it is appended to ``issues`` and blocks the launch —
            ``launch_training`` runs this before spawning the training thread. For a task the
            contract has no synthetic batch schema for, one real batch is built from ``data`` and
            used instead; if no batch can be built either, the boundary is unproven and that also
            blocks. Default False keeps the always-on web ``/validate`` path to structural checks
            plus a builder import — no model construction and no forward pass.
        overfit: When True (with ``smoke``), also run the voluntary ``overfit_check`` diagnostic and
            report it under ``overfit_check`` — never gating (a noisy-but-valid model can fail it).
    """
    from tcip_mcp.pipelines.schemas import validate_train_config_schema

    # Pydantic schema: type/structure of data/training.
    issues: list[str] = list(validate_train_config_schema(config))

    # model_source presence + builder importability (the one build path).
    model_source = config.get("model_source")
    if not model_source:
        issues.append("Missing 'model_source' section")
    elif not isinstance(model_source, dict) or not model_source.get("builder"):
        issues.append("model_source must be a dict with a 'builder' (module:function)")
    else:
        from tcip_mcp.pipelines.model_build import _import_dotted
        try:
            _import_dotted(model_source["builder"])
        except Exception as exc:
            issues.append(f"model_source.builder not importable: {exc}")

    # Data config validation
    data_cfg = config.get("data")
    if not data_cfg:
        issues.append("Missing 'data' section")
    elif data_cfg.get("dataset_source") is not None:
        # Bespoke dataset seam (mirrors model_source): the agent's builder owns loading, so the
        # known-loader images_dir/labels_dir aren't required — only the builder must import.
        dataset_source = data_cfg["dataset_source"]
        if not isinstance(dataset_source, dict) or not dataset_source.get("builder"):
            issues.append("data.dataset_source must be a dict with a 'builder' (module:function)")
        else:
            from tcip_mcp.pipelines.model_build import _import_dotted
            try:
                _import_dotted(dataset_source["builder"])
            except Exception as exc:
                issues.append(f"data.dataset_source.builder not importable: {exc}")
    else:
        for key in ("images_dir", "labels_dir"):
            path = data_cfg.get(key)
            if not path:
                issues.append(f"Missing 'data.{key}'")
            elif not Path(path).is_dir():
                issues.append(f"Directory not found: data.{key} = '{path}'")

    # Channel firewall (T6-3): probe one sample raster and check its band count against the declared
    # in_chans, so a channel-wrong train is caught here rather than deep in the daemon thread. Only
    # fires when a raster is actually readable — never a false-fail on an empty/absent dir.
    if isinstance(model_source, dict) and model_source.get("in_chans") is not None and data_cfg:
        images_dir = data_cfg.get("images_dir")
        if images_dir and Path(images_dir).is_dir():
            from tcip_mcp.pipelines.data.datasets import IMAGE_EXTS
            sample = next((f for f in sorted(Path(images_dir).iterdir())
                           if f.suffix.lower() in IMAGE_EXTS), None)
            if sample is not None:
                from tcip_mcp.pipelines.derivations import probe_channels
                from tcip_mcp.pipelines.resolution import (
                    ResolvedBundle, default as _resolved_default, validate_resolved_bundle,
                )
                try:
                    probed = int(probe_channels(sample))
                except Exception:
                    probed = None
                if probed is not None:
                    b = ResolvedBundle(trait="", dataset_hash=None, params={
                        "in_chans": _resolved_default(
                            "in_chans", int(model_source["in_chans"]), derivation_class="deterministic")})
                    issues.extend(validate_resolved_bundle(b, probed_channels=probed))

    # Split-policy validation (K1): mirrors the channel firewall above — only fires when a
    # grouping policy is actually declared, probes the dataset's stems the same way the channel
    # check probes a sample image, and never false-fails on an empty/absent/unreadable dir. Catches
    # an unrecognized ``group_by`` or an incomplete ``group_key_map`` here, at preflight, rather
    # than deep in ``_auto_train_val`` where it would now raise (T7).
    split_cfg = data_cfg.get("split") if isinstance(data_cfg, dict) else None
    if isinstance(split_cfg, dict) and (split_cfg.get("group_by") or split_cfg.get("group_key_map")):
        images_dir = data_cfg.get("images_dir")
        if images_dir and Path(images_dir).is_dir():
            from tcip_mcp.pipelines.data.datasets import IMAGE_EXTS
            stems = sorted(f.stem for f in Path(images_dir).iterdir()
                           if f.suffix.lower() in IMAGE_EXTS)
            if stems:
                from tcip_mcp.pipelines.data.splits import resolve_group_key_fn
                try:
                    resolve_group_key_fn(split_cfg.get("group_by", "tile_prefix"), stems,
                                         group_key_map=split_cfg.get("group_key_map"))
                except ValueError as exc:
                    issues.append(f"data.split: {exc}")

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

    result: dict = {"valid": False, "issues": issues}

    # Smoke: build the model and run the correctness contract at the RESOLVED dims, so a broken
    # bespoke builder is caught here (before the daemon thread spawns) rather than surfacing only as
    # run.status='failed'. Only attempt once the structural checks pass — otherwise the config can't
    # build and the contract would just re-report the same failure. Overfit stays a voluntary,
    # non-gating diagnostic (a valid model can fail 20 steps on noise).
    if smoke and not issues:
        try:
            from tcip_mcp.pipelines.model_build import build_model, resolve_contract_dims
            from tcip_mcp.pipelines.model_contract import check_model_contract, overfit_check

            ms = config.get("model_source") or {}
            task = ms.get("task") or (config.get("data") or {}).get("task", "detection")
            dims = resolve_contract_dims(config, task)
            model = build_model(config)
            report = check_model_contract(model, task, **dims)
            batch, why_no_batch = None, None
            if report.get("not_smokeable"):
                # The contract has no synthetic batch schema for this task. Rather than enumerate
                # tasks (a taxonomy) or skip the check (a rail made optional), smoke it against a
                # real batch from the run's own dataset — a better reference than a synthetic one
                # for every task, and the only one for a task the platform does not enumerate.
                batch, why_no_batch = _one_real_batch(task, config)
                if batch is not None:
                    report = check_model_contract(model, task, sample_batch=batch, **dims)
            # ``dims`` shape the synthetic batch only, so they describe nothing once a real batch
            # is used — record which reference actually proved the contract.
            result["smoke"] = {**report, "task": task,
                               "batch_source": "dataset" if batch is not None else "synthetic",
                               "dims": None if batch is not None else dims}
            if report.get("not_smokeable"):
                issues.append(
                    f"model contract: {report['not_smokeable']} Building one from the run's data "
                    f"config failed too ({why_no_batch}), so the measurement boundary is unproven."
                )
            elif not report["ok"]:
                issues.extend(f"model contract: {msg}" for msg in report["issues"])
            if overfit:
                # Same batch the contract used — otherwise this re-synthesizes and reports a false
                # "does not learn" for exactly the bespoke tasks the real-batch path exists for.
                result["overfit_check"] = overfit_check(model, task, sample_batch=batch, **dims)
        except Exception as exc:  # noqa: BLE001 — a build/contract crash is itself a blocking issue
            issues.append(f"model smoke build failed: {exc}")

    result["valid"] = len(issues) == 0
    return result


@mcp.tool()
@audited
def launch_training(config: dict, output_dir: str, resume_from: str = "") -> dict:
    """Launch a training run asynchronously from a bespoke ``model_source`` builder.

    The run will proceed in a background thread. Use check_training_status
    to monitor progress.

    Args:
        config: Full training configuration dict with model_source, data, training sections.
        output_dir: Directory for checkpoints and logs.
        resume_from: Optional path to a ``checkpoint_epoch_*.pt`` to resume from
            (restores model + optimizer + scheduler + scaler and continues).
    """
    # smoke=True: build the model and run the correctness contract before spawning the training
    # thread, so a broken builder returns here instead of wasting a full audited run.
    validation = preflight_config(config, smoke=True)
    if not validation["valid"]:
        return {"error": "Invalid config", "issues": validation["issues"]}

    # Canonicalize the shape: the GUI/validated schema nests stages/mixed_precision/batch_size
    # under ``training``, but the trainer reads them from the top level of run.config — without
    # this hoist a GUI-launched run silently trains the default single stage. (run_hpo already
    # normalizes inside _apply_hpo_params.)
    from tcip_mcp.pipelines.schemas import normalize_train_config
    config = normalize_train_config(config)

    from tcip_mcp.pipelines.training.generic_trainer import TrainConfig, create_run

    model_source = config.get("model_source", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    train_config = TrainConfig(
        model_source=model_source,
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

    # Task drives collate + measurement routing: the bespoke model_source declares it,
    # falling back to the data section.
    task = model_source.get("task") or data_cfg.get("task", "detection")

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

    # Auto-create experiment if not already tracked. Experiments are immutable:
    # reusing an id that already has a run would interleave metrics histories and
    # overwrite lineage/registry entries, so such relaunches get a fresh id.
    experiment_id = config.get("experiment_id") or run.run_id
    try:
        from tcip_mcp.experiments import update_status

        # The dataset identity this run trains on — computed ONCE and passed to both immutable
        # records (lineage + split.json) so they cannot disagree about which data produced the number.
        ds_id, ds_fp = _dataset_identity(data_cfg)
        experiment_id = _ensure_experiment(
            experiment_id, config, data_cfg.get("images_dir"), resume_from, run.run_id,
            dataset_id=ds_id, dataset_fingerprint=ds_fp,
        )
        # Thread the resolved id into the live config so the default trainer's checkpoints carry it
        # (the envelope's ctx.save_checkpoint stamps it explicitly). run.config is this same dict.
        config["experiment_id"] = experiment_id
        update_status(experiment_id, "running")
        _persist_split_manifest(experiment_id, train_ds, val_ds, data_cfg,
                                dataset_id=ds_id, dataset_fingerprint=ds_fp)
    except Exception as exc:  # Experiment tracking is best-effort, but failures must be visible.
        logger.warning("Experiment tracking failed for %s: %s", experiment_id, exc)

    # The training body runs inside the audited integrity envelope: it snapshots source/env,
    # brackets the body with audit open/close events (closing the old un-audited daemon-thread
    # hole), stamps checkpoints, and wires status/registration/lineage — around whatever training
    # code runs. Absent `training_source`, the envelope calls ctx.default_train() (today's trainer,
    # byte-identical); with it, the agent's custom train(ctx). See pipelines/training/envelope.py.
    from tcip_mcp.pipelines.training.envelope import TrainContext, run_training_envelope

    ctx = TrainContext(
        run=run, train_loader=train_loader, val_loader=val_loader,
        task=task, resume_from=resume_from, experiment_id=experiment_id,
    )
    thread = threading.Thread(target=run_training_envelope, args=(ctx,), daemon=True)
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


def _run_hpo_trial(config: dict, report, base_config: dict, trial_dir: str) -> None:
    """Train one HPO trial and ``report`` its composite objective (lower=better).

    ``report(value)`` feeds the Ray Tune searcher/scheduler; call it each epoch (so a
    scheduler can prune) and once at the end. Failures report ``+inf`` so a dead trial can
    never win a minimize sweep. Trials train under the final run's regime (same augmentation
    and imbalance handling), or the selected hyperparameters won't transfer.
    """
    merged = _apply_hpo_params(base_config, config)

    from tcip_mcp.pipelines.training.generic_trainer import create_run, train, task_collate
    from tcip_mcp.pipelines.data.samplers import build_sampler
    from torch.utils.data import DataLoader

    model_source = merged.get("model_source")
    if not model_source:
        report(float("inf"))
        return

    data_cfg = merged.get("data", {})
    train_cfg = merged.get("training", {})
    task = model_source.get("task") or data_cfg.get("task", "detection")

    # Tag as an HPO trial so it stays out of the Training-tab run list.
    run = create_run(merged, trial_dir, origin="hpo_trial")

    try:
        transforms = None
        aug_cfg = merged.get("augmentation", {})
        if aug_cfg:
            from tcip_mcp.pipelines.data.augmentations import build_augmentation
            transforms = build_augmentation(aug_cfg)

        # W4 auto-val gives the val_loader that W1's composite / the scheduler need.
        train_ds, val_ds = _auto_train_val(task, data_cfg, transforms)
        sampler = build_sampler(merged.get("sampler", "random"), train_ds)
        batch_size = train_cfg.get("batch_size", config.get("batch_size", 4))
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
            if value is not None:
                report(value)  # composite lower=better; mode='min' keeps improving trials

        train(run, train_loader, val_loader, task=task, epoch_callback=epoch_cb)
        report(run.best_metric)  # final composite (inf if train() never produced a metric)
    except Exception as e:
        logger.warning("HPO trial failed: %s", e)
        report(float("inf"))


@mcp.tool()
@audited
def run_hpo(
    base_config: dict,
    param_space: dict | None = None,
    n_trials: int = 5,
    output_dir: str = "",
    search_alg: str = "random",
    scheduler: str = "asha",
    grace_period: int = 5,
    reduction_factor: int = 3,
    warm_start: bool = False,
    baseline_params: dict | None = None,
    max_concurrent: int = 1,
) -> dict:
    """Run hyperparameter optimization on Ray Tune, training each trial for real.

    The search *algorithm* and trial *scheduler* are yours to choose per task/data — pick
    from what is installed on this machine (call the ``hpo`` module's ``available_search_algs``
    / ``available_schedulers`` for the live list); the defaults below are a sane starting
    point, not a recipe:
      - ``search_alg``: ``random``/``grid`` (native), or a backend — ``optuna``, ``bayesopt``,
        ``hyperopt``, ``nevergrad``, ``ax``, ``hebo``, ``zoopt``, ``bohb``.
      - ``scheduler``: ``asha`` (async HyperBand), ``hyperband``, ``bohb`` (pair with the bohb
        searcher), ``pbt``, ``median``, or ``none`` to run every trial to completion.

    Trials minimize the composite selection objective (lower=better); each trains under the
    base config's regime so the chosen hyperparameters transfer to ``launch_training``. Ray
    persists trial results under ``output_dir`` (also the TensorBoard logdir), and a result
    JSON is written alongside.

    Args:
        base_config: Base training config each trial modifies.
        param_space: Param-space dict (see ``hpo.get_default_space``); default when omitted.
        n_trials: Number of trials.
        output_dir: Base output directory for trial results (defaults under ``.tcip/hpo``).
        max_concurrent: Trials to run at once (default 1 — safe for single-GPU training).
    """
    from tcip_mcp.pipelines.training.hpo import tune_search, get_default_space

    if param_space is None:
        param_space = get_default_space()

    import uuid

    from tcip_mcp.project_paths import project_root

    hpo_dir = Path(output_dir) if output_dir else project_root() / ".tcip" / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    study_name = f"hpo_{uuid.uuid4().hex[:8]}"

    def objective_fn(config: dict, report) -> None:
        try:
            from ray import tune as _tune
            tid = _tune.get_context().get_trial_id()
        except Exception:
            tid = uuid.uuid4().hex[:8]
        _run_hpo_trial(config, report, base_config, str(hpo_dir / f"trial_{tid}"))

    result = tune_search(
        objective_fn=objective_fn,
        param_space=param_space,
        metric="objective",
        mode="min",
        num_samples=n_trials,
        search_alg=search_alg,
        scheduler=scheduler,
        grace_period=grace_period,
        reduction_factor=reduction_factor,
        warm_start=warm_start,
        baseline_params=baseline_params,
        max_concurrent=max_concurrent,
        storage_path=str(hpo_dir),
        study_name=study_name,
    )

    # Auto-launch TensorBoard on Ray's per-trial event files.
    tb_info: dict = {}
    tb_logdir = result.get("tensorboard_logdir")
    if tb_logdir:
        try:
            from tcip_mcp.pipelines.training.tensorboard_manager import launch_tensorboard
            tb_info = launch_tensorboard(tb_logdir, run_id=f"hpo_{study_name}")
        except Exception:
            pass

    result["tensorboard"] = tb_info
    # Durable result file (best-effort — a write hiccup must not sink a completed sweep).
    try:
        from tcip_mcp.utils.atomic_io import atomic_write_json
        atomic_write_json(hpo_dir / f"{study_name}.json", result)
    except Exception:
        logger.warning("could not persist hpo result json for %s", study_name, exc_info=True)
    return result


def _apply_hpo_params(base_config: dict, params: dict) -> dict:
    """Apply flat HPO params onto a deep copy of ``base_config``, where ``train()`` reads them.

    Architecture is owned by the bespoke ``model_source`` builder (unknown to the sweep),
    so only optimizer / schedule / batch axes are applied here. Optimizer params go into
    TOP-LEVEL ``config["optimizer"]`` and the unfreeze schedule into TOP-LEVEL
    ``config["stages"]`` — the trainer reads both from the top level and never reads
    ``training["stages"]`` or a stage's ``lr``:

      - ``lr``           -> ``optimizer["head_lr"]`` (+ ``backbone_lr`` at lr*0.1) plus a
                            3-stage ``freeze_to`` progressive-unfreeze schedule
      - ``weight_decay`` -> ``optimizer["weight_decay"]``
      - anything else (``batch_size``, ...) -> ``training``
    """
    import copy

    from tcip_mcp.pipelines.schemas import normalize_train_config

    cfg = normalize_train_config(copy.deepcopy(base_config))
    training = cfg.setdefault("training", {})

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
        else:
            training[key] = value
    return cfg


def _dataset_identity(data_cfg: dict) -> tuple[str | None, str | None]:
    """``(dataset_id, dataset_fingerprint)`` for the run's dataset — the content end of the
    reproduce-a-number chain. The fingerprint is recomputed here (recompute-on-read is authority); the
    id comes from the dataset's ``dataset.json`` if it was registered. ``(None, None)`` for a bespoke /
    imageless run (no dataset_root), matching ``dataset_hash=None`` rather than fabricating identity.
    """
    images_dir = data_cfg.get("images_dir")
    if not images_dir:
        return None, None
    import json

    from tcip_mcp.dataset_layout import dataset_identity_path, dataset_root_of
    from tcip_mcp.pipelines.resolution import dataset_fingerprint

    root = dataset_root_of(images_dir)
    if root is None:
        return None, None
    try:
        fp = dataset_fingerprint(root)
    except OSError as exc:
        # A fingerprint read failure must not sink the whole experiment record (lineage,
        # split.json, status) for a run that otherwise trains fine — degrade to an honest
        # None, matching the bespoke/imageless case, rather than fabricating or propagating.
        logger.warning("dataset_fingerprint failed for %s: %s", root, exc)
        fp = None
    ds_id = None
    ident = dataset_identity_path(root)
    if ident.is_file():
        try:
            ds_id = json.loads(ident.read_text(encoding="utf-8")).get("id")
        except (OSError, ValueError):
            ds_id = None
    return ds_id, fp


def _ensure_experiment(
    experiment_id: str, config: dict, data_source, resume_from: str, run_id: str,
    *, dataset_id: str | None = None, dataset_fingerprint: str | None = None,
) -> str:
    """Create or attach the experiment for a run, enforcing experiment immutability.

    Returns the experiment id actually used. An existing id may be reused only when
    the experiment is pristine (agent pre-created it: state 'created', no metrics)
    or when ``resume_from`` continues that experiment's own checkpoint. Anything
    else mints a fresh ``<id>_<run_id>`` (with the old id as parent lineage) so the
    prior run's status, metrics, lineage, and registry entry stay intact.
    """
    from tcip_mcp.experiments import create_experiment, get_experiment

    created = create_experiment(experiment_id, config, data_source=data_source,
                                dataset_id=dataset_id, dataset_fingerprint=dataset_fingerprint)
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
    create_experiment(fresh_id, config, parent_experiment=experiment_id, data_source=data_source,
                      dataset_id=dataset_id, dataset_fingerprint=dataset_fingerprint)
    return fresh_id


def _persist_split_manifest(experiment_id: str, train_ds, val_ds, data_cfg: dict, *,
                            dataset_id: str | None = None,
                            dataset_fingerprint: str | None = None) -> None:
    """Persist which stems (+ seed + dataset_hash + dataset identity) produced this run's metrics (R5).

    The same seed yields a different split if the label set changes, so a metric is only reproducible
    with the exact train/val membership recorded beside it. The whole-dataset ``dataset_fingerprint``
    (+ id) records the content identity too, so this artifact is literally "fingerprint + split" —
    content identity + membership + seed in one immutable record. Best-effort — a provenance write must
    never sink a launch.
    """
    def _stems(ds) -> list[str]:
        return sorted(getattr(ds, "stems", None) or getattr(ds, "_stems", []) or [])

    try:
        from tcip_mcp.experiments import experiments_dir, record_artifact
        from tcip_mcp.pipelines.resolution import dataset_hash
        from tcip_mcp.utils.atomic_io import atomic_write_json

        labels_dir = data_cfg.get("labels_dir", "")
        dh = None
        if labels_dir and Path(labels_dir).is_dir():
            dh = dataset_hash(labels_dir)
        split = data_cfg.get("split", {})
        resolved_group_by = split.get("resolved_group_by")
        manifest = {
            "train": _stems(train_ds),
            "val": _stems(val_ds) if val_ds is not None else [],
            "seed": int(split.get("seed", 42)),
            "dataset_hash": dh,
            "dataset_id": dataset_id,
            "dataset_fingerprint": dataset_fingerprint,
            # The actually resolved grouping (not the requested string) — "explicit_map" when
            # ``_auto_train_val`` resolved a caller group_key_map, "external" for the two-directory
            # explicit-val path (no computed grouping at all), the named strategy otherwise, or None
            # when no split manifest logic ran. K1's train-disjointness gate (operating_point.py)
            # reads this to recompute the training run's own group keys — a missing/None value here
            # fails that check closed rather than guessing a policy.
            "group_by": resolved_group_by,
        }
        if resolved_group_by == "explicit_map" and split.get("group_key_map"):
            # The map itself (K1 finding 1.1) — without it, _train_disjointness has a policy NAME
            # but no way to actually compute group keys for stems outside this run, which is what
            # made the group_key_map capability effectively unusable (exercising it permanently
            # blocked that model's later calibration).
            manifest["group_key_map"] = split["group_key_map"]
        exp_dir = experiments_dir() / experiment_id
        if exp_dir.is_dir():
            path = exp_dir / "split.json"
            atomic_write_json(path, manifest)
            record_artifact(experiment_id, "split", str(path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("split manifest persist failed for %s: %s", experiment_id, exc)


def _dataset_source_kwargs(task: str, data_cfg: dict) -> dict:
    """The ``build_dataset`` kwargs for a run's data config.

    One definition shared by the training path and the preflight smoke, so the batch the contract
    is proved against is built from the same keys as the batch the run will train on.
    """
    if task in ("detection", "instance_seg"):
        kw = {"images_dir": data_cfg.get("images_dir", ""),
              "labels_dir": data_cfg.get("labels_dir", "")}
        # The run's subject (and optional attribute): required to read name-based labels and to
        # derive the single assign_class_ids map. Threaded so every train/val build uses one map.
        if data_cfg.get("subject"):
            kw["subject"] = data_cfg["subject"]
        if data_cfg.get("attribute"):
            kw["attribute"] = data_cfg["attribute"]
        # Thread the on-disk label format through to the dataset (json | coco).
        if data_cfg.get("label_format"):
            kw["label_format"] = data_cfg["label_format"]
        if data_cfg.get("coco_json"):
            kw["coco_json"] = data_cfg["coco_json"]
    elif task == "semantic_seg":
        kw = {"images_dir": data_cfg.get("images_dir", ""),
              "masks_dir": data_cfg.get("masks_dir", data_cfg.get("labels_dir", ""))}
    else:
        kw = {"images_dir": data_cfg.get("images_dir", "")}
        if data_cfg.get("csv_path"):
            kw["csv_path"] = data_cfg["csv_path"]
    if data_cfg.get("dataset_source"):
        # Bespoke seam (mirrors model_source): route build_dataset to the agent's builder for a
        # task the known loaders don't cover. Threaded through src so the split machinery still
        # passes it (with stems) to every train/val build below.
        kw["dataset_source"] = data_cfg["dataset_source"]
    return kw


def _one_real_batch(task: str, config: dict, n: int = 2):
    """``(batch, reason_it_failed)`` — one collated ``(images, targets)`` from the run's dataset.

    Lets the model contract smoke a task it has no synthetic schema for, without the platform
    enumerating tasks. Built through the same source kwargs and augmentation the run itself uses,
    so a batch that smokes here is the batch that trains. Best-effort by design: a config that
    cannot yield a batch returns ``(None, reason)`` and the caller decides what that means — this
    function never decides whether a run proceeds. The reason is returned rather than only logged,
    so a caller that blocks can say what actually failed.
    """
    data_cfg = config.get("data") or {}
    try:
        from tcip_mcp.pipelines.data.datasets import build_dataset
        from tcip_mcp.pipelines.training.generic_trainer import task_collate

        transforms = None
        if config.get("augmentation"):
            from tcip_mcp.pipelines.data.augmentations import build_augmentation
            transforms = build_augmentation(config["augmentation"])

        src = _dataset_source_kwargs(task, data_cfg)
        ds = build_dataset(task, **src, transforms=transforms, tiling=data_cfg.get("tiling"))
        items = [ds[i] for i in range(min(n, len(ds)))]
        if not items:
            return None, "the dataset built but is empty"
        return task_collate(task)(items), None
    except Exception as exc:  # noqa: BLE001 — an unbuildable batch is a caller decision, not a crash
        logger.info("could not build a real batch to smoke task %r: %s", task, exc)
        return None, f"{type(exc).__name__}: {exc}"


def _auto_train_val(task: str, data_cfg: dict, transforms):
    """Build ``(train_ds, val_ds)`` for a run, deriving a leakage-free val split.

    Resolution order:
      1. ``data.val_images_dir`` set -> build val from it explicitly.
      2. ``data.auto_val`` (default True) and a stem-capable task
         (detection / instance_seg / semantic_seg / classification) -> derive a
         group-aware train/val split (no held-out test) so the trainer receives
         a real validation loader. Train keeps augmentation; val gets none.
      3. ordinal / regression, ``auto_val`` disabled, a tiny/single-group set, or
         most failures -> ``(full_train_ds, None)``. ``resolve_group_key_fn`` (an
         unrecognized ``split.group_by`` or a ``split.group_key_map`` missing stem
         coverage) is called outside any handler here and its ``ValueError`` propagates
         to the caller — silently training without validation on a policy error the
         caller could have fixed is worse than surfacing it. Every other failure in this
         function (dataset build errors, a malformed ``val_ratio``/``seed``, a
         ``group_balanced_split`` failure) still degrades to ``(full_train_ds, None)``.

    Reads ``auto_val`` / ``val_*`` / ``split.*`` from ``data_cfg`` (== config["data"]).
    """
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.data.splits import (
        group_balanced_split, count_label_lines, resolve_group_key_fn,
    )

    STEM_TASKS = {"detection", "instance_seg", "semantic_seg", "classification"}

    src = _dataset_source_kwargs(task, data_cfg)
    tiling = data_cfg.get("tiling")  # W3: detection tiling (None for other tasks/configs)

    # 1. Explicit validation source.
    val_images = data_cfg.get("val_images_dir")
    if val_images:
        # This path builds train/val from two SEPARATE directories with no computed grouping —
        # there is no group policy to persist. Record that shape explicitly (K1 finding 1.2) so
        # _persist_split_manifest writes a distinct "external" marker rather than leaving the field
        # unset, which _train_disjointness would otherwise be unable to tell apart from "no split
        # manifest was ever written" (a split.json this old, from before this check existed).
        data_cfg.setdefault("split", {})["resolved_group_by"] = "external"
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
            from tcip_mcp.pipelines.data.datasets import (
                assemble_coco, dir_label_format, _resolve_registry_id_map,
            )
            _labels, _images = src.get("labels_dir", ""), src.get("images_dir", "")
            if _labels and _images and dir_label_format(_labels) == "json":
                _subject, _attribute = src.get("subject"), src.get("attribute")
                _reg, _id_map = _resolve_registry_id_map(_labels, _subject, _attribute)
                build_src["coco_data"] = assemble_coco(
                    _labels, _images, subject=_subject, attribute=_attribute, id_map=_id_map)
                build_src["label_format"] = "coco"
                build_src["num_classes"] = len(_id_map)

        full_ds = build_dataset(task, **build_src, transforms=transforms)
        stems = list(getattr(full_ds, "stems", None) or getattr(full_ds, "_stems", []))
    except Exception as exc:
        logger.warning("Auto train/val split failed (%s); training without validation.", exc)
        return build_dataset(task, **src, transforms=transforms, tiling=tiling), None

    if len(stems) < 2:
        return full_ds, None

    # setdefault (not get): the resolved grouping is written back into data_cfg["split"] below
    # so _persist_split_manifest (called separately, after this returns) can record what was
    # actually used — the same "write the effective value back" pattern the tiling geometry
    # above uses.
    split_cfg = data_cfg.setdefault("split", {})
    group_by = split_cfg.get("group_by", "tile_prefix")
    group_key_map = split_cfg.get("group_key_map")
    # Deliberately outside any try/except: a malformed grouping policy is a caller-config
    # error, not a runtime split failure, and must reach the caller rather than degrade
    # silently to "training without validation" like the failures handled below.
    group_key_fn = resolve_group_key_fn(group_by, stems, group_key_map=group_key_map)
    split_cfg["resolved_group_by"] = "explicit_map" if group_key_map else group_by

    try:
        val_ratio = float(split_cfg.get("val_ratio", 0.2))
        seed = int(split_cfg.get("seed", 42))
        stratify = split_cfg.get("stratify_foreground", True)

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
    at; for true TP/FP/FN ranking use ``score_predictions`` (``detail=True``, IoU-matched).

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

    from tcip_annotation.json_io import read_annotations

    scores: list[tuple[str, float]] = []
    for pred_file in pred_path.glob("*.json"):
        if pred_file.name == "operating_point.json":
            continue
        gt_file = gt_path / pred_file.name
        preds = [a for a in read_annotations(str(pred_file)) if a.geometry is not None]
        gt_anns = [a for a in read_annotations(str(gt_file)) if a.geometry is not None] if gt_file.is_file() else []

        n_pred = len(preds)
        n_gt = len(gt_anns)

        # Simple error heuristic: |pred - gt| + missed + extra + low confidence
        missed = max(0, n_gt - n_pred)
        extra = max(0, n_pred - n_gt)
        avg_conf = 0.0
        if n_pred > 0:
            confs = [p.score for p in preds if p.score is not None]
            avg_conf = sum(confs) / len(confs) if confs else 0.5

        # Higher score = worse prediction
        error_score = missed * 2.0 + extra * 1.0 + (1.0 - avg_conf)
        scores.append((pred_file.stem, error_score))

    # Also include GT images with no predictions at all (completely missed)
    for gt_file in gt_path.glob("*.json"):
        pred_file = pred_path / gt_file.name
        if not pred_file.is_file():
            gt_anns = [a for a in read_annotations(str(gt_file)) if a.geometry is not None]
            if gt_anns:
                scores.append((gt_file.stem, len(gt_anns) * 3.0))

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
    global_nms_iou: float = DEFAULT_NMS_IOU,
    postprocess: str = "nms",
    trait: str | None = None,
    subject: str | None = None,
    attribute: str | None = None,
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
        trait: When set, the trait's DERIVED localization criterion (traits.py — catkin's
            center-match) governs the reported count and the selection f1; AP@0.5 (``iou_threshold``)
            is kept as a labeled comparability metric (D9). Absent -> the IoU convention governs.
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

    run_data_cfg = (run.config.get("data", {}) or {}) if run is not None else {}
    run_tiling = run_data_cfg.get("tiling")
    # The eval scope's subject/attribute: caller-supplied wins, else the producing run's config, so
    # the name-based GT reads through the same id map the run trained with.
    if subject is None:
        subject = run_data_cfg.get("subject")
    if attribute is None:
        attribute = run_data_cfg.get("attribute")

    # Delivery-grade full-frame path (tiled inference + full-frame GT matching).
    if use_tiled_inference and task == "detection":
        tcfg = tiling or run_tiling or {}
        # Thread the merge settings through — evaluating at a derived (non-default) NMS is
        # exactly the point of this path; dropping them silently re-pins 0.3.
        return run_full_frame_evaluation(
            ckpt, images_dir, labels_dir, str(Path(ckpt).parent),
            subject=subject, attribute=attribute,
            conf_threshold=conf_threshold, iou_threshold=iou_threshold,
            tile_size=int(tcfg.get("tile_size", 640)), overlap=float(tcfg.get("overlap", 0.2)),
            global_nms_iou=global_nms_iou, postprocess=postprocess,
            max_dets=max_dets if max_dets > 100 else 1000, trait=trait,
        )

    # Tile-level diagnostic (or untiled). Only detection tiles; a run id reuses its training tiling.
    if tiling is None and run is not None:
        tiling = run_tiling
    if task != "detection":
        tiling = None

    ds_kwargs: dict = {"images_dir": images_dir}
    if task in ("detection", "instance_seg"):
        ds_kwargs["labels_dir"] = labels_dir
        ds_kwargs["subject"] = subject
        ds_kwargs["attribute"] = attribute
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
        iou_type=iou_type, max_dets=max_dets, tiling=tiling, trait=trait,
    )
