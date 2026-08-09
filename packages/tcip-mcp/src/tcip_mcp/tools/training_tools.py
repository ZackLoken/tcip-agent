"""Training MCP tools, config validation, launch training, HPO, status."""

from __future__ import annotations

import itertools
import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited
from tcip_mcp.pipelines.resolution import DEFAULT_CONF, DEFAULT_NMS_IOU

logger = logging.getLogger(__name__)

# Round-robins unpinned concurrent launches across available GPUs (no-op with 0-1 devices).
_gpu_round_robin = itertools.count()

# Lazy imports of heavy dependencies inside tool functions to keep server startup fast.


@mcp.tool()
@audited
def preflight_config(config: dict, smoke: bool = False, overfit: bool = False) -> dict:
    """Validate a training configuration before launching.

    Config structure:
        model_source: {builder, builder_kwargs, task, in_chans}
        data: {images_dir, labels_dir, task}  # known loaders, OR a bespoke
              # {dataset_source: {builder, builder_kwargs, source_files, task}, task}
        training: {batch_size, ...}  # the full key list generic_trainer.train() reads
              # (device/seed/deterministic/mixed_precision/stages/optimizer/scheduler/
              # lr_scaling/stage_warmup_epochs/enforce_monotonic_unfreeze/
              # gradient_accumulation_steps/checkpoint_every_n_epochs/early_stopping/evaluation)
              # is documented on train()'s own docstring, not repeated here, read that for the
              # canonical, always-current list. training_source: optional custom train(ctx) loop.

    Args:
        config: Full training configuration dict.
        smoke: When True, actually build the model and run ``check_model_contract`` (a train+eval
            forward at the resolved in_chans/num_classes/img_size). A contract failure is a
            guaranteed real-run failure, so it is appended to ``issues`` and blocks the launch,
            ``launch_training`` runs this before spawning the training thread. For a task the
            contract has no synthetic batch schema for, one real batch is built from ``data`` and
            used instead; if no batch can be built either, the boundary is unproven and that also
            blocks. Default False keeps the always-on web ``/validate`` path to structural checks
            plus a builder import, no model construction and no forward pass.
        overfit: When True (with ``smoke``), also run the voluntary ``overfit_check`` diagnostic and
            report it under ``overfit_check``, never gating (a noisy-but-valid model can fail it).
    """
    from tcip_mcp.pipelines.schemas import validate_train_config_schema

    # Pydantic schema: type/structure of data/training.
    issues: list[str] = list(validate_train_config_schema(config))
    warnings: list[str] = []

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

    # training_source seam (mirrors model_source/dataset_source above), a bare "module:function"
    # string, not a dict.
    training_source = config.get("training_source")
    if training_source is not None:
        if not isinstance(training_source, str) or not training_source:
            issues.append("training_source must be a non-empty 'module:function' string")
        else:
            from tcip_mcp.pipelines.model_build import _import_dotted
            try:
                _import_dotted(training_source)
            except Exception as exc:
                issues.append(f"training_source not importable: {exc}")

    # Data config validation
    data_cfg = config.get("data")
    if not data_cfg:
        issues.append("Missing 'data' section")
    elif data_cfg.get("dataset_source") is not None:
        # Bespoke dataset seam (mirrors model_source): the agent's builder owns loading, so the
        # known-loader images_dir/labels_dir aren't required, only the builder must import.
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

    # Channel firewall: probe one sample raster and check its band count against the declared
    # in_chans, so a channel-wrong train is caught here rather than deep in the training subprocess.
    # Only fires when a raster is actually readable, never a false-fail on an empty/absent dir.
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
                            "in_chans", int(model_source["in_chans"]))})
                    issues.extend(validate_resolved_bundle(b, probed_channels=probed))

    # Split-policy validation: mirrors the channel firewall above, only fires when a
    # grouping policy is actually declared, probes the dataset's stems the same way the channel
    # check probes a sample image, and never false-fails on an empty/absent/unreadable dir. Catches
    # an unrecognized ``group_by`` or an incomplete ``group_key_map`` here, at preflight, rather
    # than deep in ``_auto_train_val`` where it would otherwise raise.
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

    # Trainable-sample coverage: trainable_stems' own partition was computed
    # by DetectionDataset/InstanceSegDataset and then thrown away, a run whose label store admits
    # only a fraction of its annotated images (an unconfirmed-empty backlog, a stale-schema
    # quarantine, incomplete attribute coverage) reported "valid, no warnings" with no visibility
    # into what would silently train on far fewer images than the operator expects. Never gating,
    # a real project legitimately has unconfirmed/unannotated images, and only fires for the known
    # loaders (a dataset_source's own admission logic is the agent's to report, not this rail's).
    task_for_coverage = (model_source.get("task") if isinstance(model_source, dict) else None) \
        or (data_cfg.get("task", "detection") if isinstance(data_cfg, dict) else "detection")
    if (isinstance(data_cfg, dict) and data_cfg.get("dataset_source") is None
            and task_for_coverage in ("detection", "instance_seg")):
        images_dir, labels_dir = data_cfg.get("images_dir"), data_cfg.get("labels_dir")
        if images_dir and labels_dir and Path(images_dir).is_dir() and Path(labels_dir).is_dir():
            try:
                from tcip_mcp.pipelines.data.datasets import trainable_stems
                stems, sample_counts = trainable_stems(
                    labels_dir, images_dir, subject=data_cfg.get("subject"), date=data_cfg.get("date"))
            except Exception:
                stems, sample_counts = None, None
            if sample_counts:
                dropped = {k: v for k, v in sample_counts.items()
                          if k not in ("annotated", "confirmed_negative") and v}
                total = sum(sample_counts.values())
                n_dropped = sum(dropped.values())
                if n_dropped and total:
                    warnings.append(
                        f"data: {n_dropped}/{total} candidate images ({n_dropped / total:.0%}) will "
                        f"not train, {dict(sorted(dropped.items()))}. {len(stems)} stem(s) admitted.")

    # Training config validation
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 2)
    if not isinstance(batch_size, int) or batch_size < 1:
        issues.append("'training.batch_size' must be a positive integer")

    # Per-stage 'epochs' is required; 'lr' is optional (StageSpec) and the trainer
    # reads learning rates from config['optimizer'], never from a stage. Absent
    # stages are fine, launch_training supplies its own default schedule.
    for i, stage in enumerate(train_cfg.get("stages") or []):
        if "epochs" not in stage:
            issues.append(f"Stage {i} missing 'epochs'")
        if "lr" in stage:
            warnings.append(
                f"training.stages[{i}].lr is set but ignored, the trainer reads learning rate "
                "only from the top-level 'optimizer' block (backbone_lr/head_lr), applied "
                "uniformly across every stage. Move the value into 'optimizer' if you meant to "
                "change it."
            )

    # Fail fast on an incoherent explicit selection_metric (a metric a center-match trait's
    # own criterion demotes to comparability-only) at validation time, not mid-run.
    eval_cfg = train_cfg.get("evaluation") or config.get("evaluation") or {}
    sel_metric = eval_cfg.get("selection_metric")
    trait_name = eval_cfg.get("trait")
    if sel_metric and trait_name:
        from tcip_mcp.pipelines.training.generic_trainer import resolve_selection_metric

        task_for_check = (model_source.get("task") if isinstance(model_source, dict) else None) \
            or (data_cfg.get("task", "detection") if isinstance(data_cfg, dict) else "detection")
        try:
            resolve_selection_metric(task_for_check, trait_name, sel_metric)
        except ValueError as exc:
            issues.append(str(exc))

    result: dict = {"valid": False, "issues": issues, "warnings": warnings}

    # Smoke: build the model and run the correctness contract at the resolved dims, so a broken
    # bespoke builder is caught here (before the training subprocess spawns) rather than surfacing
    # only as run.status='failed'. Only attempt once the structural checks pass, otherwise the
    # config can't build and the contract would just re-report the same failure. Overfit stays a
    # voluntary, non-gating diagnostic (a valid model can fail 20 steps on noise).
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
                # real batch from the run's own dataset, a better reference than a synthetic one
                # for every task, and the only one for a task the platform does not enumerate.
                batch, why_no_batch = _one_real_batch(task, config)
                if batch is not None:
                    report = check_model_contract(model, task, sample_batch=batch, **dims)
            # ``dims`` shape the synthetic batch only, so they describe nothing once a real batch
            # is used, record which reference actually proved the contract.
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
                # Same batch the contract used, otherwise this re-synthesizes and reports a false
                # "does not learn" for exactly the bespoke tasks the real-batch path exists for.
                result["overfit_check"] = overfit_check(model, task, sample_batch=batch, **dims)
        except Exception as exc:  # noqa: BLE001, a build/contract crash is itself a blocking issue
            issues.append(f"model smoke build failed: {exc}")

    result["valid"] = len(issues) == 0
    return result


@mcp.tool()
@audited
def launch_training(
    config: dict, output_dir: str = "", resume_from: str = "",
    max_wall_clock_seconds: float | None = None,
) -> dict:
    """Launch a training run in an isolated subprocess from a bespoke ``model_source`` builder.

    The run's actual training body (dataset build, model forward/backward, checkpointing) executes
    in a separate OS process, not this one, a bug/OOM/hang in one run can't take down this
    process or any other concurrent run's process. Use check_training_status to monitor progress;
    it reads the run's own status/metrics from disk, not shared memory.

    Args:
        config: Full training configuration dict with model_source, data, training sections.
        output_dir: Directory for checkpoints and logs. Empty defaults to the experiment store
            (``<project>/.tcip/experiments``, the same base the experiment records use); a
            relative path resolves against the project root, never the server process's cwd.
        resume_from: Optional path to a ``checkpoint_epoch_*.pt`` to resume from
            (restores model + optimizer + scheduler + scaler and continues).
        max_wall_clock_seconds: Optional hard timeout. If the training process hasn't exited on its
            own by then, it is terminated and the run marked failed with that reason, no
            cooperative grace period is attempted (a hung process isn't responding to cooperative
            signals). Omit for no timeout (the default).
    """
    # smoke=True: build the model and run the correctness contract before spawning the training
    # subprocess, so a broken builder returns here instead of wasting a full audited run.
    validation = preflight_config(config, smoke=True)
    if not validation["valid"]:
        return {"error": "Invalid config", "issues": validation["issues"]}

    # Canonicalize the shape: the GUI/validated schema nests stages/mixed_precision/batch_size
    # under ``training``, but the trainer reads them from the top level of run.config, without
    # this hoist a GUI-launched run silently trains the default single stage. (run_hpo already
    # normalizes inside _apply_hpo_params.)
    from tcip_mcp.pipelines.schemas import normalize_train_config
    config = normalize_train_config(config)

    from tcip_mcp.experiments import experiments_dir
    from tcip_mcp.pipelines.training.generic_trainer import create_run
    from tcip_mcp.project_paths import resolve_output_path

    # Training artifacts (weights, tensorboard, metrics) live with the project the run belongs
    # to, same as its experiment record; only an absolute output_dir points anywhere else.
    output_dir = str(resolve_output_path(output_dir) if output_dir else experiments_dir())

    data_cfg = config.get("data", {})

    run = create_run(config, output_dir)
    # Nest each run's artifacts under its run_id. The GUI (and typical callers) pass a
    # shared base such as ``<project>/.tcip/experiments``; without nesting, sequential
    # runs write ``metrics.jsonl`` / ``model_best.pt`` to the *same* flat directory and
    # clobber each other, violating experiment immutability. Nesting also makes the
    # trainer write exactly where the web metrics stream reads (``<base>/<run_id>/``), and gives
    # the subprocess a single directory to write into and the cancel sentinel to live in.
    run.output_dir = str(Path(output_dir) / run.run_id)

    # Auto-create experiment if not already tracked. Experiments are immutable:
    # reusing an id that already has a run would interleave metrics histories and
    # overwrite lineage/registry entries, so such relaunches get a fresh id.
    experiment_id = config.get("experiment_id") or run.run_id
    try:
        from tcip_mcp.experiments import update_status

        # The dataset identity this run trains on, computed once and passed to the immutable
        # lineage record. The child recomputes the identical fingerprint independently for
        # split.json, cheap, and "recompute-on-read" is this fact's own stated authority, so
        # there's no need to thread it across the process boundary.
        ds_id, ds_fp = _dataset_identity(data_cfg)
        experiment_id = _ensure_experiment(
            experiment_id, config, data_cfg.get("images_dir"), resume_from, run.run_id,
            output_dir=run.output_dir, dataset_id=ds_id, dataset_fingerprint=ds_fp,
        )
        # Thread the resolved id into the live config so the child's checkpoints carry it (the
        # envelope's ctx.save_checkpoint stamps it explicitly).
        config["experiment_id"] = experiment_id
        update_status(experiment_id, "running")
    except Exception as exc:  # Experiment tracking is best-effort, but failures must be visible.
        logger.warning("Experiment tracking failed for %s: %s", experiment_id, exc)

    # The child reads its own bootstrap config from here, independent of whether experiment
    # tracking above succeeded, a filesystem hiccup in .tcip/experiments degrades tracking (as it
    # always has) without also preventing the run from training at all. experiment_id is never
    # read from here (see subprocess_worker.py), only passed as the explicit CLI arg below,
    # because this file is written before config["experiment_id"] is guaranteed resolved in the
    # fresh-id-relaunch branch.
    from tcip_mcp.utils.atomic_io import atomic_write_json
    # Created only after the experiment resolution above: under the default base this is the
    # experiment record's own dir, and pre-creating it reads as an existing experiment there.
    Path(run.output_dir).mkdir(parents=True, exist_ok=True)
    launch_config_path = Path(run.output_dir) / "launch_config.json"
    atomic_write_json(launch_config_path, config)

    child_env = _child_env_for_launch(config)

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "tcip_mcp.pipelines.training.subprocess_worker",
            "--run-id", run.run_id,
            "--experiment-id", experiment_id,
            "--config-path", str(launch_config_path),
            "--output-dir", run.output_dir,
            "--resume-from", resume_from,
        ],
        env=child_env,
    )
    run.pid = proc.pid

    if max_wall_clock_seconds is not None:
        _watch_wall_clock(proc, run, experiment_id, max_wall_clock_seconds)

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
        "pid": proc.pid,
    }


def _child_env_for_launch(config: dict) -> dict[str, str]:
    """Subprocess env for a launch: round-robin GPU pinning when the config names no
    explicit device, left untouched when it does. ``CUDA_VISIBLE_DEVICES`` remaps device
    *indices* inside the child, pinning it would ask an explicit ``device: "cuda:1"`` config for
    an ordinal invalid in the child's own remapped view, so pinning applies only to the unpinned
    case it's meant to spread out.

    Also propagates this process's own import search path via ``PYTHONPATH``, the child is a
    fresh interpreter with only sys.path's own defaults, not whatever got this process's bespoke
    ``model_source``/``training_source``/``dataset_source`` module importable in the first place
    (an editable install's extra path entries, a test runner's rootdir insertion, an agent's own
    working-directory convention). Without this, a bespoke module importable to the caller can
    become unimportable to the child purely because of the process boundary, a correctness gap,
    not just a convenience.
    """
    import os
    import sys

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    path_entries = [p for p in sys.path if p]
    if existing_pythonpath:
        path_entries = path_entries + [existing_pythonpath]
    env["PYTHONPATH"] = os.pathsep.join(path_entries)

    if config.get("device") or config.get("training", {}).get("device"):
        return env

    try:
        import torch
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        count = 0
    if count > 1:
        from tcip_mcp.pipelines.training.generic_trainer import _RUNS_LOCK
        with _RUNS_LOCK:
            idx = next(_gpu_round_robin) % count
        env["CUDA_VISIBLE_DEVICES"] = str(idx)
    return env


def _watch_wall_clock(proc: subprocess.Popen, run: Any, experiment_id: str,
                      timeout_seconds: float) -> None:
    """Daemon watcher: hard-terminates ``proc`` if it outlives ``timeout_seconds`` and
    records the reason through the same status channel every other terminal state uses, never an
    in-memory-only mark, since ``check_training_status`` always defers to disk for a pid-bearing
    run and would otherwise never surface it. No cooperative grace period: a hung process isn't
    responding to cooperative signals, so this is a hard kill, not the cancel path."""
    def _watch() -> None:
        try:
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.terminate()
            reason = f"exceeded max_wall_clock_seconds ({timeout_seconds})"
            run.status = "failed"
            run.error = reason
            try:
                from tcip_mcp.experiments import update_status
                update_status(experiment_id, "failed", error=reason)
            except Exception:
                logger.warning("wall-clock timeout status update failed for %s",
                               experiment_id, exc_info=True)

    threading.Thread(target=_watch, daemon=True).start()


@mcp.tool()
@audited
def check_training_status(run_id: str) -> dict:
    """Check the status of a training run.

    Reads the run's own status/metrics from disk whenever its training body runs in a
    subprocess, the in-memory record for a subprocess-delegated run is a launch-time placeholder
    only, since the subprocess mutates its own separate copy in its own process memory, or when
    this process never held the run in memory at all (a different process launched it).

    Args:
        run_id: Training run identifier.
    """
    from tcip_mcp.pipelines.training.generic_trainer import get_run
    run = get_run(run_id)

    result: dict[str, Any] | None = None
    if run is None or run.pid is not None:
        from tcip_mcp.experiments import reconstruct_run_status
        disk = reconstruct_run_status(run_id)
        if disk is not None:
            result = {
                "run_id": disk["run_id"],
                "status": disk["status"],
                "epoch": disk["current_epoch"],
                "best_metric": disk["best_metric"],
                "output_dir": disk["output_dir"],
                "error": disk.get("error"),
            }

    if result is None:
        if run is None:
            return {"error": f"Run not found: {run_id}"}
        result = {
            "run_id": run.run_id,
            "status": run.status,
            "epoch": run.current_epoch,
            "best_metric": run.best_metric,
            "output_dir": run.output_dir,
        }

    # Check for running TensorBoard
    tb_url = None
    try:
        from tcip_mcp.pipelines.training.tensorboard_manager import _TB_PROCESSES
        proc = _TB_PROCESSES.get(run_id)
        if proc and proc.poll() is None:
            tb_url = f"http://localhost:{proc._tb_port}"
    except Exception:
        pass
    result["tensorboard_url"] = tb_url
    return result


@mcp.tool()
@audited
def list_training_runs() -> dict:
    """List all training runs in this session.

    Overlays disk-reconstructed status onto any subprocess-delegated run (``pid`` set) whose
    in-memory record is a stale launch-time placeholder; a run whose training body never left this
    process (every existing synchronous test, and any future non-subprocess caller) is reported
    from the live in-memory record exactly as before, untouched.
    """
    from tcip_mcp.pipelines.training.generic_trainer import list_runs
    from tcip_mcp.experiments import reconstruct_run_status

    runs = list_runs()
    for r in runs:
        if r.get("pid") is not None:
            disk = reconstruct_run_status(r["run_id"])
            if disk is not None:
                r["status"] = disk["status"]
                if disk["current_epoch"] is not None:
                    r["current_epoch"] = disk["current_epoch"]
                if disk.get("error"):
                    r["error"] = disk["error"]
    return {"runs": runs}


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
    run = get_run(run_id)
    if run is not None:
        status = run.status
    else:
        # Cancelled via the disk fallback, this process never held the run locally, so
        # there's no in-memory status to read; reflect the same disk record cancel_run itself
        # resolved to write the sentinel, if it's still discoverable.
        from tcip_mcp.experiments import reconstruct_run_status
        disk = reconstruct_run_status(run_id)
        status = disk["status"] if disk is not None else "running"
    return {"run_id": run_id, "status": status, "cancel_requested": True}


@mcp.tool()
@audited
def inspect_compute_resources() -> dict:
    """Report the host's current compute headroom, a fact to reason with before launching
    another concurrent training/HPO run, not an enforced cap. This platform doesn't cap memory/CPU
    per run (no portable, non-pinned way to do that across POSIX/Windows without guessing a number
    that's wrong on the next host); it gives you the real numbers and trusts you to judge whether
    another candidate run fits, the same way you'd judge any other CV-scientist tradeoff.

    Returns:
        ``cpu``: ``{logical_count, percent_used}``, ``percent_used`` is ``None`` without
            ``psutil`` installed.
        ``memory``: ``{total_bytes, available_bytes}``, both ``None`` without ``psutil``.
        ``gpus``: ``[{index, free_bytes, total_bytes}, ...]``, always populated when CUDA is
            available (``torch.cuda.mem_get_info``, no extra dependency); ``[]`` otherwise.
        ``active_training_runs``: count of runs currently reporting ``"running"``, reads through
            ``list_training_runs``'s own disk overlay, since a subprocess-delegated run's parent-
            side in-memory record never mutates past its launch-time placeholder.
    """
    import os

    cpu: dict[str, Any] = {"logical_count": os.cpu_count(), "percent_used": None}
    memory: dict[str, Any] = {"total_bytes": None, "available_bytes": None}
    try:
        import psutil
        cpu["percent_used"] = psutil.cpu_percent(interval=0.1)
        vm = psutil.virtual_memory()
        memory["total_bytes"] = vm.total
        memory["available_bytes"] = vm.available
    except Exception:
        logger.info("psutil unavailable or failed; cpu/memory visibility degraded to None",
                   exc_info=True)

    gpus: list[dict[str, Any]] = []
    try:
        import torch
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                free_b, total_b = torch.cuda.mem_get_info(idx)
                gpus.append({"index": idx, "free_bytes": free_b, "total_bytes": total_b})
    except Exception:
        logger.info("GPU visibility unavailable", exc_info=True)

    # list_training_runs (not the raw generic_trainer.list_runs), its disk overlay is what makes
    # a subprocess-delegated run's real status visible; the parent's own in-memory copy never
    # leaves its launch-time placeholder once the child starts mutating its own separate copy.
    active = sum(1 for r in list_training_runs()["runs"] if r.get("status") == "running")

    return {"cpu": cpu, "memory": memory, "gpus": gpus, "active_training_runs": active}


# Keys _apply_hpo_params gives purpose-built handling (routed into nested optimizer/
# training structures train() reads unconditionally), tracking them by their own top-level
# name would be meaningless. Only the "else"-routed passthrough keys (the actual risk case:
# a swept axis that lands somewhere no consumer reads) are checked against consumption.
_HPO_KNOWN_KEYS = {"lr", "batch_size", "weight_decay"}


class _AccessTrackingConfig(dict):
    """Dict subclass recording which top-level keys are ever read via ``__getitem__``/``get``/
    ``__contains__``, installed on ``run.config`` for one HPO trial's dispatch, so
    ``unconsumed_params`` reflects genuine runtime access (did anything read this key during
    this trial), not a static comparison against ``train()``'s known key list, which would
    falsely flag a bespoke ``training_source``'s own legitimate custom sweep key.

    Real, stated limitations (never gates the run, warn-only, so a false positive costs a log
    line, not a failed trial): top-level only (a nested read like
    ``ctx.config["optimizer"]["custom_key"]`` isn't seen); ``dict(cfg)``/``**cfg`` copies bypass
    the overrides entirely (CPython copies at the C level); whole-dict iteration
    (``.items()``/``.values()``/``.keys()``) isn't tracked per-key.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.accessed: set[str] = set()

    def __getitem__(self, key: Any) -> Any:
        self.accessed.add(key)
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        self.accessed.add(key)
        return super().get(key, default)

    def __contains__(self, key: Any) -> bool:
        self.accessed.add(key)
        return super().__contains__(key)


def hpo_root(output_dir: str = "") -> Path:
    """Where HPO sweeps live: ``output_dir`` when the caller named one, else ``.tcip/hpo``
    under the platform state root. A relative ``output_dir`` resolves against the project
    root, never the server process's cwd.

    The one resolver for that decision. Anything that has to find a sweep on disk (the
    Tuning routes included) calls this rather than rebuilding the same default.
    """
    from tcip_mcp.project_paths import project_root, resolve_output_path

    return resolve_output_path(output_dir) if output_dir else project_root() / ".tcip" / "hpo"


def sweep_dir(study_name: str, output_dir: str = "") -> Path:
    """One sweep's own directory: its manifest, its ``trial_<id>`` dirs, and (because Ray
    is handed ``storage_path=hpo_root`` and ``name=study_name``) Ray's experiment store."""
    return hpo_root(output_dir) / study_name


def _run_hpo_trial(config: dict, report, base_config: dict, trial_dir: str) -> None:
    """Train one HPO trial and ``report`` its composite objective (lower=better).

    ``report(value)`` feeds the Ray Tune searcher/scheduler; call it each epoch (so a
    scheduler can prune) and once at the end. Failures report ``+inf`` so a dead trial can
    never win a minimize sweep. Trials train under the final run's regime, same augmentation,
    imbalance handling, and dispatch: a ``training_source`` in ``base_config`` actually
    runs under that loop here too, not always the stock trainer, or the selected
    hyperparameters won't transfer.
    """
    merged = _apply_hpo_params(base_config, config)

    from tcip_mcp.pipelines.training.envelope import TrainContext, dispatch_train_body
    from tcip_mcp.pipelines.training.generic_trainer import (
        create_run, task_collate, seeded_loader_kwargs, stamp_effective_data_geometry,
    )
    from tcip_mcp.pipelines.data.samplers import build_sampler
    from torch.utils.data import DataLoader

    model_source = merged.get("model_source")
    if not model_source:
        report(float("inf"))
        return

    # setdefault, not get: the geometry stamp below mutates this dict and must land in the
    # resolved-config snapshot written from merged.
    data_cfg = merged.setdefault("data", {})
    train_cfg = merged.get("training", {})
    task = model_source.get("task") or data_cfg.get("task", "detection")

    # Track which top-level keys the trial actually reads, so a swept param that never
    # reaches any consumer is caught by observation rather than gated by a whitelist that would
    # forbid a bespoke training_source from sweeping its own custom axes.
    tracked_config = _AccessTrackingConfig(merged)
    # Tag as an HPO trial so it stays out of the Training-tab run list.
    run = create_run(tracked_config, trial_dir, origin="hpo_trial")

    try:
        transforms = None
        aug_cfg = merged.get("augmentation", {})
        if aug_cfg:
            from tcip_mcp.pipelines.data.augmentations import build_augmentation
            transforms = build_augmentation(aug_cfg)

        # Auto-val gives the val_loader that the composite objective / the scheduler need.
        train_ds, val_ds = _auto_train_val(task, data_cfg, transforms)
        # Stamped before training so a pruned/failed trial's resolved-config snapshot still
        # records the geometry the trial actually trained on.
        stamp_effective_data_geometry(data_cfg, train_ds)
        batch_size = train_cfg.get("batch_size", config.get("batch_size", 4))
        num_workers = train_cfg.get("num_workers", 0)
        # Built after the loader context is known: a sampler whose read order depends on the
        # worker regime and batching consumes both.
        sampler = build_sampler(merged.get("sampler", "random"), train_ds,
                                num_workers=num_workers, batch_size=batch_size)
        # run.config's seed is create_run-resolved (auto-drawn if base_config left it unset),
        # read it off run.config, not merged, so the loader is seeded with the value actually used.
        loader_kwargs = seeded_loader_kwargs(run.config.get("seed"), num_workers=num_workers)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=(sampler is None),
            sampler=sampler, collate_fn=task_collate(task), num_workers=num_workers,
            **loader_kwargs,
        )
        val_loader = None
        if val_ds is not None:
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                collate_fn=task_collate(task), num_workers=num_workers,
                **loader_kwargs,
            )

        def epoch_cb(epoch: int, metrics: dict) -> None:
            # resolve_selection_metric governs which key actually decides checkpoint
            # choice once evaluation.trait/evaluation.selection_metric are set, prefer it over
            # the raw composite so pruning ranks trials on the same criterion selection uses.
            value = metrics.get("selection", metrics.get("val_objective", metrics.get("val_loss")))
            if value is not None:
                report(value)  # composite lower=better; mode='min' keeps improving trials

        # Dispatch through the same training_source-or-default_train() decision the full
        # audited envelope uses. experiment_id=None is deliberate, dispatch_train_body never
        # reaches _finalize_run/register_model_from_experiment, so a trial stays isolated from
        # the registry (origin="hpo_trial") while still actually training under a bespoke loop
        # when base_config carries one, instead of silently falling back to the stock trainer.
        # trial_report wires ctx.report_objective(value) for a bespoke loop whose own metrics
        # don't share the stock trainer's key names, so it can still report progress for pruning.
        ctx = TrainContext(run=run, train_loader=train_loader, val_loader=val_loader, task=task,
                           experiment_id=None, epoch_hook=epoch_cb, trial_report=report)
        dispatch_train_body(ctx)
        report(run.best_metric)  # final composite (inf if train() never produced a metric)
    except Exception as e:
        logger.warning("HPO trial failed: %s", e)
        report(float("inf"))
    finally:
        # Surface any swept param no consumer touched. Warn-only, never gates the trial.
        unconsumed = sorted((set(config.keys()) - _HPO_KNOWN_KEYS) - tracked_config.accessed)
        try:
            from tcip_mcp.utils.atomic_io import atomic_write_json
            Path(trial_dir).mkdir(parents=True, exist_ok=True)
            # trial_params is the sampled point itself, the only record of which axes this
            # sweep actually varied (the merged config can't say which of its values came
            # from the search space).
            atomic_write_json(Path(trial_dir) / "resolved_config.json",
                              {**merged, "trial_params": dict(config),
                               "unconsumed_params": unconsumed})
        except Exception:
            logger.warning("could not persist resolved_config.json for %s", trial_dir, exc_info=True)
        if unconsumed:
            logger.warning(
                "HPO trial %s: swept params %s were never read by the training body, check "
                "_apply_hpo_params' routing, or (for a bespoke training_source) confirm the "
                "loop actually reads them from ctx.config.", trial_dir, unconsumed)


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
    resources_per_trial: dict | None = None,
) -> dict:
    """Run hyperparameter optimization on Ray Tune, training each trial for real.

    The search *algorithm* and trial *scheduler* are yours to choose per task/data, pick
    from what is installed on this machine (call the ``hpo`` module's ``available_search_algs``
    / ``available_schedulers`` for the live list); the defaults below are a sane starting
    point, not a recipe:
      - ``search_alg``: ``random``/``grid`` (native), or a backend, ``optuna``, ``bayesopt``,
        ``hyperopt``, ``nevergrad``, ``ax``, ``hebo``, ``zoopt``, ``bohb``.
      - ``scheduler``: ``asha`` (async HyperBand), ``hyperband``, ``bohb`` (pair with the bohb
        searcher), ``pbt``, ``median``, or ``none`` to run every trial to completion.

    Trials minimize the composite selection objective (lower=better); each trains under the
    base config's regime so the chosen hyperparameters transfer to ``launch_training``.

    Everything one sweep writes lands under ``<output_dir or .tcip/hpo>/<study_name>/``: a
    ``manifest.json`` stamped ``running`` before the first trial starts (so a sweep is
    visible while it runs, not only once it ends) and updated when the sweep ends, one
    ``trial_<id>/`` directory per trial, and Ray's own experiment store (also the
    TensorBoard logdir). The full result is written alongside as ``<study_name>.json``.

    Args:
        base_config: Base training config each trial modifies.
        param_space: Param-space dict (see ``hpo.get_default_space``); default when omitted.
        n_trials: Number of trials.
        output_dir: Base output directory for trial results (defaults under ``.tcip/hpo``).
        search_alg: Search algorithm, see the list above; call ``hpo.available_search_algs()``
            for what's actually installed on this box.
        scheduler: Trial scheduler, see the list above.
        grace_period: Minimum epochs before a halving scheduler (``asha``/``hyperband``) can
            stop a trial early.
        reduction_factor: Halving factor for ``asha``/``hyperband`` (fraction of trials kept
            at each rung).
        warm_start: Seed the search with ``baseline_params`` as a known-good starting point.
        baseline_params: Hyperparameter values to seed the search with when ``warm_start=True``.
        max_concurrent: Trials to run at once (default 1, safe for single-GPU training).
        resources_per_trial: Ray resource request per trial, omit to derive one from the
            host's real GPU count and ``max_concurrent`` (see ``hpo._default_trial_resources``);
            an explicit value always wins over the derivation.
    """
    from tcip_mcp.pipelines.training.hpo import tune_search, get_default_space

    if param_space is None:
        param_space = get_default_space()

    import uuid
    from datetime import datetime, timezone

    from tcip_mcp.utils.atomic_io import atomic_write_json

    hpo_dir = hpo_root(output_dir)
    hpo_dir.mkdir(parents=True, exist_ok=True)
    study_name = f"hpo_{uuid.uuid4().hex[:8]}"
    sweep_root = sweep_dir(study_name, output_dir)
    sweep_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "study_name": study_name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "n_trials": n_trials,
        "search_alg": search_alg,
        "scheduler": scheduler,
        "param_space": param_space,
        "sweep_dir": str(sweep_root),
    }
    manifest_path = sweep_root / "manifest.json"
    atomic_write_json(manifest_path, manifest)

    def objective_fn(config: dict, report) -> None:
        try:
            from ray import tune as _tune
            tid = _tune.get_context().get_trial_id()
        except Exception:
            tid = uuid.uuid4().hex[:8]
        _run_hpo_trial(config, report, base_config, str(sweep_root / f"trial_{tid}"))

    try:
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
            resources_per_trial=resources_per_trial,
        )
    except Exception as exc:
        manifest.update(status="failed", error=str(exc),
                        finished_at=datetime.now(timezone.utc).isoformat())
        atomic_write_json(manifest_path, manifest)
        raise

    # Auto-launch TensorBoard on the sweep root: Ray's per-trial event files and each
    # trial's own tensorboard dir both sit under it.
    tb_info: dict = {}
    tb_logdir = result.get("tensorboard_logdir")
    if tb_logdir:
        try:
            from tcip_mcp.pipelines.training.tensorboard_manager import launch_tensorboard
            tb_info = launch_tensorboard(tb_logdir, run_id=f"hpo_{study_name}")
        except Exception:
            pass

    result["tensorboard"] = tb_info
    manifest.update(
        status="completed",
        finished_at=datetime.now(timezone.utc).isoformat(),
        result={k: result.get(k) for k in ("best_params", "best_value", "n_trials")},
    )
    # Durable result files (best-effort, a write hiccup must not sink a completed sweep).
    try:
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(hpo_dir / f"{study_name}.json", result)
    except Exception:
        logger.warning("could not persist hpo result json for %s", study_name, exc_info=True)
    return result


def _apply_hpo_params(base_config: dict, params: dict) -> dict:
    """Apply flat HPO params onto a deep copy of ``base_config``, where ``train()`` reads them.

    Architecture is owned by the bespoke ``model_source`` builder (unknown to the sweep), so
    only optimizer/batch axes get purpose-built handling here; ``base_config``'s own progressive-
    unfreeze schedule is left untouched, preserving whatever schedule the agent configured:

      - ``lr``           -> ``optimizer["head_lr"]``, plus ``optimizer["backbone_lr"]`` scaled by
                            whatever backbone/head ratio ``base_config`` already expressed
                            (derived, not pinned, a frozen ``lr*0.1`` would discard an agent's
                            own deliberate ratio)
      - ``weight_decay`` -> ``optimizer["weight_decay"]``
      - ``batch_size``   -> ``training["batch_size"]``
      - anything else    -> the top level of ``cfg`` (not nested under ``training``, which runs
                            after ``normalize_train_config``'s hoist and would never reach
                            ``train()``'s top-level config reads), free for a bespoke
                            ``training_source`` to sweep its own axes; no whitelist, no reject.
    """
    import copy

    from tcip_mcp.pipelines.schemas import normalize_train_config

    cfg = normalize_train_config(copy.deepcopy(base_config))
    training = cfg.setdefault("training", {})

    # The ratio the agent already configured, read from base_config's own optimizer block
    # before this loop overwrites head_lr. Default to 1.0 (not a frozen 0.1) only when the agent
    # expressed no explicit backbone/head split at all.
    base_optimizer = base_config.get("optimizer") or {}
    base_backbone_lr = base_optimizer.get("backbone_lr")
    base_head_lr = base_optimizer.get("head_lr")
    backbone_head_ratio = (
        base_backbone_lr / base_head_lr if (base_head_lr and base_backbone_lr) else 1.0
    )

    for key, value in params.items():
        if key == "lr":
            lr = float(value)
            optimizer = cfg.setdefault("optimizer", {})
            optimizer["head_lr"] = lr
            optimizer["backbone_lr"] = lr * backbone_head_ratio
        elif key == "batch_size":
            training["batch_size"] = value
        elif key == "weight_decay":
            cfg.setdefault("optimizer", {})["weight_decay"] = value
        else:
            cfg[key] = value
    return cfg


def _dataset_identity(data_cfg: dict) -> tuple[str | None, str | None]:
    """``(dataset_id, dataset_fingerprint)`` for the run's dataset, the content end of the
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
        # split.json, status) for a run that otherwise trains fine, degrade to an honest
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
    *, output_dir: str, dataset_id: str | None = None, dataset_fingerprint: str | None = None,
) -> str:
    """Create or attach the experiment for a run, enforcing experiment immutability.

    Returns the experiment id actually used. An existing id may be reused only when the experiment
    is pristine (agent pre-created it: state 'created', no metrics), in which case its
    ``config.json`` (written before tiling/seed resolution) is refreshed with the config this run
    is actually launching. Anything else, including a ``resume_from`` that targets
    an id which already has recorded history, mints a fresh ``<id>_<run_id>`` (with the old id as
    parent lineage) so the prior run's status, metrics, lineage, and registry entry stay intact
    (resuming into a non-pristine id without this would silently reuse it, discarding the
    resumed run's own metrics/lineage writes behind the terminal-state lock and letting the model
    registry replace the original's entry by name with no record of what was superseded).

    Every branch below stamps ``run_id``/``output_dir`` into the resolved experiment's
    ``status.json`` before returning, unconditionally, once, regardless of which branch
    resolved the id, so a different process can later discover this run's real artifact directory
    from ``experiment_id`` alone (``resolve_experiment_dir_for_run``/``reconstruct_run_status``/the
    disk-based ``cancel_run`` fallback all depend on this). Deliberately not a ``create_experiment``
    param: that would only cover the fresh-creation branch and silently miss the pristine-reuse
    branch, which never otherwise touches ``status.json`` at all.
    """
    from tcip_mcp.experiments import (
        create_experiment, get_experiment, overwrite_config_if_pristine, stamp_run_identity,
    )

    created = create_experiment(experiment_id, config, data_source=data_source,
                                dataset_id=dataset_id, dataset_fingerprint=dataset_fingerprint)
    if "error" not in created:
        stamp_run_identity(experiment_id, run_id, output_dir)
        return experiment_id

    existing = get_experiment(experiment_id, metrics_limit=1)
    pristine = (
        existing.get("status", {}).get("state") == "created"
        and not existing.get("n_epochs")
    )
    if pristine:
        overwrite_config_if_pristine(experiment_id, config)
        stamp_run_identity(experiment_id, run_id, output_dir)
        return experiment_id

    fresh_id = f"{experiment_id}_{run_id}"
    logger.warning(
        "experiment_id %s already has a run; experiments are immutable, tracking "
        "this run as %s instead.", experiment_id, fresh_id,
    )
    create_experiment(fresh_id, config, parent_experiment=experiment_id, data_source=data_source,
                      dataset_id=dataset_id, dataset_fingerprint=dataset_fingerprint)
    stamp_run_identity(fresh_id, run_id, output_dir)
    return fresh_id


def _persist_split_manifest(experiment_id: str, train_ds, val_ds, data_cfg: dict, *,
                            dataset_id: str | None = None,
                            dataset_fingerprint: str | None = None) -> None:
    """Persist which stems (+ seed + dataset_hash + dataset identity) produced this run's metrics.

    The same seed yields a different split if the label set changes, so a metric is only reproducible
    with the exact train/val membership recorded beside it. The whole-dataset ``dataset_fingerprint``
    (+ id) records the content identity too, so this artifact is literally "fingerprint + split",
    content identity + membership + seed in one immutable record. Best-effort, a provenance write must
    never sink a launch.
    """
    def _stems(ds) -> list[str]:
        # set(): a tiled dataset's ``stems`` repeats one entry per tile, and a manifest member
        # list is a set of units, never a per-example list.
        return sorted(set(getattr(ds, "stems", None) or getattr(ds, "_stems", []) or []))

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
        # A spatial_strip split's members are per-region identities, never the bare stem;
        # _auto_train_val already computed and stashed them (the dataset only knows tile positions).
        spatial = split.get("spatial_manifest") if resolved_group_by == "spatial_strip" else None
        train_members = spatial["train_identities"] if spatial else _stems(train_ds)
        val_members = (spatial["val_identities"] if spatial
                       else (_stems(val_ds) if val_ds is not None else []))
        manifest = {
            "train": train_members,
            "val": val_members,
            "seed": int(split.get("seed", 42)),
            "dataset_hash": dh,
            "dataset_id": dataset_id,
            "dataset_fingerprint": dataset_fingerprint,
            # The actually resolved grouping ("explicit_map"/"external"/a named strategy/
            # "spatial_strip"/None); _train_disjointness recomputes group keys from this.
            "group_by": resolved_group_by,
        }
        if resolved_group_by == "explicit_map" and split.get("group_key_map"):
            # The map itself: without it _train_disjointness has a policy name but no way to
            # compute group keys for stems outside this run.
            manifest["group_key_map"] = split["group_key_map"]
        if spatial:
            manifest["spatial"] = spatial
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
    """``(batch, reason_it_failed)``, one collated ``(images, targets)`` from the run's dataset.

    Lets the model contract smoke a task it has no synthetic schema for, without the platform
    enumerating tasks. Built through the same source kwargs and augmentation the run itself uses,
    so a batch that smokes here is the batch that trains. Best-effort by design: a config that
    cannot yield a batch returns ``(None, reason)`` and the caller decides what that means, this
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
    except Exception as exc:  # noqa: BLE001, an unbuildable batch is a caller decision, not a crash
        logger.info("could not build a real batch to smoke task %r: %s", task, exc)
        return None, f"{type(exc).__name__}: {exc}"


def _spatial_single_source_split(
    stem: str, data_cfg: dict, tiling: dict, base, split_cfg: dict, transforms,
) -> tuple:
    """A train/val split over one detection source's own tile lattice, by disjoint pixel strips.

    Called only from ``_auto_train_val``'s single-source branch: there is no second stem to hold
    out whole, but a tiled source has many tiles, and :func:`~tcip_mcp.pipelines.data.splits.
    spatial_strip_split` can hold out disjoint, buffered regions of them. ``base`` is the
    already-built (untiled) ``DetectionDataset`` for this one source, reused for every view (its
    construction, and the class/subject/id-map resolution inside it, cost nothing extra to
    share). A test region is derived and reserved alongside train/val (excluded from both, so it
    is genuinely held out) but no dataset is built for it: nothing downstream consumes a third
    dataset from this function today, so only its geometry and kept-tile count are recorded,
    material a later block-aware calibration mechanism can consume without recomputing the split.
    Returns ``(train_ds, val_ds)``, or ``None`` when the extent is unknown or no strip layout can
    populate both train and val, in which case the caller falls back to training without
    validation.
    """
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset, tile_kwargs_from_tiling
    from tcip_mcp.pipelines.data.splits import image_extent_from_labels, spatial_strip_split

    extent = image_extent_from_labels(data_cfg.get("labels_dir", ""), stem)
    if extent is None:
        logger.warning(
            "Spatial train/val split for %r skipped: its label file carries no width/height; "
            "training without validation.", stem,
        )
        return None
    width, height = extent

    raw_kwargs = tile_kwargs_from_tiling(tiling)
    # keep_regions is this function's own to set from the derived split, never inherited from
    # the caller's tiling dict (which has no meaningful keep_regions in a single-source launch).
    tile_kwargs = {k: v for k, v in raw_kwargs.items() if k != "keep_regions"}
    # Matches TiledDetectionDataset.__init__'s own defaults: the geometry below must agree with
    # what the datasets built further down actually resolve to.
    tile_size = tile_kwargs.get("tile_size", 224)
    overlap = tile_kwargs.get("overlap", 0.2)
    val_ratio = float(split_cfg.get("val_ratio", 0.2))
    test_ratio = float(split_cfg.get("test_ratio", 0.1))
    train_ratio = 1.0 - val_ratio - test_ratio
    seed = int(split_cfg.get("seed", 42))

    try:
        spatial = spatial_strip_split(
            width, height, tile_size, overlap,
            fractions=(train_ratio, val_ratio, test_ratio), seed=seed, buffer=tiling.get("buffer"),
        )
    except ValueError as exc:
        logger.warning(
            "Spatial train/val split for %r could not be derived (%s); training without "
            "validation.", stem, exc,
        )
        return None

    train_ds = TiledDetectionDataset(
        base, transforms=transforms, keep_regions=spatial.regions["train"], **tile_kwargs)
    val_ds = TiledDetectionDataset(
        base, transforms=None, keep_regions=spatial.regions["val"], **tile_kwargs)
    if train_ds.num_samples == 0 or val_ds.num_samples == 0:
        logger.warning(
            "Spatial train/val split for %r yielded an empty side after tile filtering; "
            "training without validation.", stem,
        )
        return None

    def _identities(ds) -> list[str]:
        return sorted({spatial.identity_for(s, tx, ty) for s, tx, ty in ds.tile_entries} - {None})

    split_cfg["resolved_group_by"] = "spatial_strip"
    split_cfg["spatial_manifest"] = {
        "train_identities": _identities(train_ds), "val_identities": _identities(val_ds),
        "test_region": spatial.regions.get("test", []),
        "kept_test_tiles": spatial.kept_tiles.get("test", 0),
        "width": spatial.width, "height": spatial.height, "tile_size": spatial.tile_size,
        "overlap": spatial.overlap, "axis": spatial.axis, "buffer": spatial.buffer,
        "seed": spatial.seed, "requested_fractions": dict(zip(spatial.split_names,
                                                               spatial.requested_fractions)),
        "realized_fractions": spatial.realized_fractions,
        "realized_discard_fraction": spatial.realized_discard_fraction,
        "kept_train_tiles": spatial.kept_tiles.get("train", 0),
        "kept_val_tiles": spatial.kept_tiles.get("val", 0),
        "tiles_dropped_past_extent": spatial.tiles_dropped_past_extent,
        "tiles_dropped_outside_regions": spatial.tiles_dropped_outside_regions,
    }
    logger.info(
        "Spatial train/val split for %r: %d train / %d val tiles (axis=%s, "
        "realized_fractions=%s, realized_discard_fraction=%.3f).",
        stem, train_ds.num_samples, val_ds.num_samples, spatial.axis,
        spatial.realized_fractions, spatial.realized_discard_fraction,
    )
    return train_ds, val_ds


def _auto_train_val(task: str, data_cfg: dict, transforms):
    """Build ``(train_ds, val_ds)`` for a run, deriving a leakage-free val split.

    Resolution order:
      1. ``data.val_images_dir`` set -> build val from it explicitly (a CSV-driven task -
         classification/ordinal/regression - also requires ``data.val_csv_path``; there is no
         graceful fallback to the train CSV the way the geometry tasks fall back to the train
         labels/masks dir, see the CSV branch below for why).
      2. ``data.auto_val`` (default True) and a stem-capable task
         (detection / instance_seg / semantic_seg / classification) -> derive a
         group-aware train/val split (no held-out test) so the trainer receives
         a real validation loader. Train keeps augmentation; val gets none.
      3. ordinal / regression, ``auto_val`` disabled, a tiny/single-group set, or
         most failures -> ``(full_train_ds, None)``. ``resolve_group_key_fn`` (an
         unrecognized ``split.group_by`` or a ``split.group_key_map`` missing stem
         coverage) is called outside any handler here and its ``ValueError`` propagates
         to the caller, silently training without validation on a policy error the
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
    tiling = data_cfg.get("tiling")  # detection tiling (None for other tasks/configs)

    # 1. Explicit validation source.
    val_images = data_cfg.get("val_images_dir")
    if val_images:
        # This path builds train/val from two separate directories with no computed grouping,
        # there is no group policy to persist. Record that shape explicitly so
        # _persist_split_manifest writes a distinct "external" marker rather than leaving the field
        # unset, which _train_disjointness would otherwise be unable to tell apart from a
        # split.json where the field was never set at all.
        data_cfg.setdefault("split", {})["resolved_group_by"] = "external"
        try:
            train_ds = build_dataset(task, **src, transforms=transforms, tiling=tiling)
            val_src = dict(src)
            val_src["images_dir"] = val_images
            if task in ("detection", "instance_seg"):
                val_src["labels_dir"] = data_cfg.get("val_labels_dir", data_cfg.get("labels_dir", ""))
            elif task == "semantic_seg":
                val_src["masks_dir"] = data_cfg.get("val_masks_dir", data_cfg.get("masks_dir", ""))
            elif task in ("classification", "ordinal", "regression"):
                val_csv = data_cfg.get("val_csv_path")
                if not val_csv:
                    # A CSV dataset reads every row eagerly as a real item and only fails per-item
                    # at __getitem__ time (deep inside a later training-loop iteration) if a row's
                    # image isn't in val_images_dir - unlike the geometry tasks above, where a
                    # missing per-image label file degrades gracefully to "no label". Falling back
                    # to the train csv_path here would risk silently building a val_ds that crashes
                    # mid-training instead of failing now, so require it explicitly.
                    raise ValueError(
                        "val_images_dir set for a CSV-driven task also requires "
                        "data.val_csv_path; the train CSV's rows won't generally match a "
                        "different val_images_dir."
                    )
                val_src["csv_path"] = val_csv
            return train_ds, build_dataset(task, **val_src, transforms=None, tiling=tiling)
        except Exception as exc:
            logger.warning("Explicit val build failed (%s); training without validation.", exc)
            return build_dataset(task, **src, transforms=transforms, tiling=tiling), None

    if not data_cfg.get("auto_val", True) or task not in STEM_TASKS:
        return build_dataset(task, **src, transforms=transforms, tiling=tiling), None

    # 2. Auto group-aware train/val split.
    try:
        # Assemble the dataset-level COCO once (JSON detection labels) and thread it into
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
        # A single source can't hold out a whole stem, but a tiled detection source can still
        # hold out disjoint pixel blocks of its own tile lattice, see _spatial_single_source_split.
        split_cfg = data_cfg.setdefault("split", {})
        if task == "detection" and tiling and tiling.get("enabled", True):
            spatial_ds = _spatial_single_source_split(
                stems[0], data_cfg, tiling, full_ds, split_cfg, transforms)
            if spatial_ds is not None:
                return spatial_ds
        return build_dataset(task, **src, transforms=transforms, tiling=tiling), None

    # setdefault (not get): the resolved grouping is written back so a later
    # _persist_split_manifest call can record what was actually used.
    split_cfg = data_cfg.setdefault("split", {})
    group_by = split_cfg.get("group_by", "tile_prefix")
    group_key_map = split_cfg.get("group_key_map")
    # Deliberately outside any try/except: a malformed grouping policy is a caller-config error
    # and must reach the caller, not degrade silently like the failures handled below.
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
        if (not val_stems or not train_stems) and group_by != "stem" and not group_key_map:
            # Too few *groups* under the requested policy starved val (e.g. two sources whose
            # tile-prefix collapses to one group); retry at stem-level grouping before giving up.
            stem_key_fn = resolve_group_key_fn("stem", stems)
            retry_parts = group_balanced_split(
                stems, annotation_counts=annotation_counts, group_key_fn=stem_key_fn,
                splits=(1.0 - val_ratio, val_ratio, 0.0), seed=seed,
            )
            if retry_parts["train"] and retry_parts["val"]:
                logger.info(
                    "Auto train/val split for %s: group_by=%r left val empty (too few groups); "
                    "retried at stem-level grouping.", task, group_by,
                )
                parts = retry_parts
                train_stems, val_stems = parts["train"], parts["val"]
                split_cfg["resolved_group_by"] = "stem"
        if not val_stems or not train_stems:
            logger.warning(
                "Auto train/val split for %s: no grouping policy could populate both sides; "
                "training without validation.", task,
            )
            return build_dataset(task, **src, transforms=transforms, tiling=tiling), None

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
    no loss. The score is ``2·|n_gt−n_pred as a shortfall| + |surplus| + (1−avg_conf)``, purely
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
    from tcip_annotation.state import Point

    def _boxes(path) -> list:
        """The annotations this count heuristic counts, a geometry-less label and a ``Point`` are
        not detections, so neither belongs in a box count on either side of the comparison."""
        return [a for a in read_annotations(str(path))
                if a.geometry is not None and not isinstance(a.geometry, Point)]

    scores: list[tuple[str, float]] = []
    for pred_file in pred_path.glob("*.json"):
        if pred_file.name == "operating_point.json":
            continue
        gt_file = gt_path / pred_file.name
        preds = _boxes(pred_file)
        gt_anns = _boxes(gt_file) if gt_file.is_file() else []

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
            gt_anns = _boxes(gt_file)
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
    max_dets: int | None = None,
    tiling: dict | None = None,
    use_tiled_inference: bool = False,
    global_nms_iou: float = DEFAULT_NMS_IOU,
    postprocess: str = "nms",
    trait: str | None = None,
    subject: str | None = None,
    attribute: str | None = None,
) -> dict:
    """Evaluate a trained checkpoint on a (held-out) dataset and write test_results.json.

    Computes the same per-task metrics as validation, detection/instance_seg get
    pycocotools mAP + precision/recall/F1; classification/ordinal/regression get the
    in-house scalar metrics, and writes ``test_results.json`` beside the checkpoint.

    Three detection eval regimes:
      * Untiled default (no ``tiling``, checkpoint trained without tiling) -> single full-res
        forward pass, ``eval_regime="full-frame-single-pass"``. For a checkpoint that was never
        tile-trained, this is the correct delivery gate, untiled training, untiled eval, untiled
        inference are all the same regime, so there is nothing to reconcile. Do not reach for
        ``use_tiled_inference`` for such a checkpoint; it has no persisted tile geometry to gate
        against and will refuse (see below).
      * ``tiling`` set (or a run id whose training was tiled, reused automatically) -> tile-level
        diagnostic that matches the training-run val mAP. This is not the delivery metric, it
        scores fragmented tiles against fragmented GT, not the shipped full-frame count.
      * ``use_tiled_inference=True`` -> the delivery-grade full-frame metric for a tile-trained
        checkpoint (tiled inference reconstructed to full frame, matched to full-frame GT). Report
        this to gate a delivery for such a checkpoint. Tile geometry is resolved from the
        checkpoint's own persisted training geometry (or an explicit override); a checkpoint with
        neither refuses rather than silently fabricating a scale, see
        ``run_full_frame_evaluation``'s docstring for the full precedence and the escape hatch.

    Args:
        run_id_or_ckpt: A training run id (uses its ``model_best.pt``) or a checkpoint path.
        images_dir: Images directory for the evaluation split.
        labels_dir: Labels dir (detection/instance_seg), masks dir (semantic_seg), or the GT CSV
            path (classification/ordinal/regression, one row per image stem).
        task: Task type.
        conf_threshold: Operating confidence for P/R/F1.
        iou_threshold: Operating IoU (on COCOeval's grid; 0.5 -> index 0).
        iou_type: 'bbox' or 'segm'. Default (None) auto-resolves from the task, 'segm' for
            instance_seg, 'bbox' otherwise, so a mask model isn't silently scored as boxes.
        max_dets: Full-frame/COCOeval detection cap. ``None`` (default) resolves
            per-regime, 100 (the COCOeval ``maxDets`` convention) on the tile-level diagnostic
            path, 1000 (``DEFAULT_MAX_DETS``, dense full-frame scenes aren't truncated) on the
            delivery-grade ``use_tiled_inference`` path. An explicit value is always honored
            verbatim on both paths (no rescuing substitution), the delivery-grade path stamps a
            per-image ``cap_hit``/``max_dets_cap_saturated_frac`` so an explicit cap that actually
            truncates real detections is visible rather than silently assumed safe.
        tiling: Optional detection tiling dict ({enabled, tile_size, overlap, ...}) for a
            tile-level eval. None + a run id reuses the run's training tiling; None + a
            checkpoint path stays untiled.
        use_tiled_inference: Score the delivery regime (full-frame via tiled inference).
        global_nms_iou: Cross-tile global NMS IoU threshold (tiled paths only).
        postprocess: Cross-tile merge, "nms" suppresses overlaps, "nmm" unions boxes split
            across a tile seam.
        trait: When set, the trait's derived localization criterion (traits.py, e.g. a count
            trait's center-match) governs the reported count and the selection f1; AP@0.5 (``iou_threshold``)
            is kept as a labeled comparability metric. Absent -> the IoU convention governs.
        subject: Name-based GT scope. Caller-supplied wins; else resolved from the producing
            run's own config so the eval reads GT through the same id map the run trained with.
        attribute: Attribute scope for the same name-based GT resolution as ``subject``.
    """
    import torch
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.training.generic_trainer import get_run, task_collate
    from tcip_mcp.pipelines.training.evaluation import (
        run_full_frame_evaluation, run_test_evaluation,
    )
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.resolution import DEFAULT_MAX_DETS

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
        # An explicit caller max_dets is honored verbatim (no rescuing sentinel);
        # None resolves to the delivery-grade default (dense full-frame scenes aren't truncated).
        resolved_max_dets = DEFAULT_MAX_DETS if max_dets is None else max_dets
        # tile_size/overlap pass through as None-if-absent, run_full_frame_evaluation
        # itself resolves them from the checkpoint's persisted training geometry (or refuses) rather
        # than this wrapper silently defaulting to 640/0.2. Thread the merge settings through too,
        # evaluating at a derived (non-default) NMS is exactly the point of this path; dropping them
        # silently re-pins 0.3.
        try:
            return run_full_frame_evaluation(
                ckpt, images_dir, labels_dir, str(Path(ckpt).parent),
                subject=subject, attribute=attribute,
                conf_threshold=conf_threshold, iou_threshold=iou_threshold,
                tile_size=tcfg.get("tile_size"), overlap=tcfg.get("overlap"),
                global_nms_iou=global_nms_iou, postprocess=postprocess,
                max_dets=resolved_max_dets, trait=trait,
            )
        except ValueError as exc:
            return {"error": str(exc)}

    # Tile-level diagnostic (or untiled). Only detection tiles; a run id reuses its training tiling.
    if tiling is None and run is not None:
        tiling = run_tiling
    if task != "detection":
        tiling = None

    # One kwargs-builder shared with the training path (_dataset_source_kwargs), not a second
    # hand-rolled copy: two independent implementations of "which data_cfg keys does this task
    # read" is exactly what let classification/ordinal/regression drift out of sync with training
    # (evaluate_model never threaded a CSV path, so OrdinalDataset/RegressionDataset construction
    # always failed here). labels_dir doubles as the CSV path for the non-geometry tasks, the same
    # single "wherever this task's GT lives" slot it already serves for masks_dir/semantic_seg.
    data_cfg = {"images_dir": images_dir, "labels_dir": labels_dir, "masks_dir": labels_dir,
                "csv_path": labels_dir, "subject": subject, "attribute": attribute}
    ds_kwargs = _dataset_source_kwargs(task, data_cfg)
    try:
        dataset = build_dataset(task, **ds_kwargs, tiling=tiling)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Failed to build dataset: {exc}"}

    loader = DataLoader(dataset, batch_size=4, collate_fn=task_collate(task))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 100 is the COCOeval maxDets convention for this tile-level/diagnostic
    # regime, distinct from the delivery-grade path's 1000 above, resolved here (not via a
    # shared sentinel value) so an explicit caller max_dets<=100 is never silently substituted.
    resolved_max_dets = 100 if max_dets is None else max_dets
    return run_test_evaluation(
        ckpt, loader, device, task, str(Path(ckpt).parent),
        conf_threshold=conf_threshold, iou_threshold=iou_threshold,
        iou_type=iou_type, max_dets=resolved_max_dets, tiling=tiling, trait=trait,
    )
