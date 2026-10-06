"""Training MCP tools, config validation, launch training, HPO, status."""

from __future__ import annotations

import itertools
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, NamedTuple, Sized

from tcip_store import StoreError, canonical_path, check_json_value

from tcip_mcp import experiments
from tcip_mcp.experiments import SWEEP_FILE, SweepGroup, TrainingDetail
from tcip_mcp.server import tool
from tcip_mcp.audit import audited, now_iso, record_event_or_raise
from tcip_mcp.pipelines.data.split_construction import (
    ResolvedRun, data_dir_issues, resolve_run, selection_compatibility, split_seed,
)
from tcip_mcp.pipelines.execution import Stated
from tcip_mcp.pipelines.model_build import run_task

logger = logging.getLogger(__name__)

# Round-robins unpinned concurrent launches across available GPUs (no-op with 0-1 devices);
# next() on a count is one C call, atomic under the GIL.
_gpu_round_robin = itertools.count()

# Serializes the overfit diagnostic's reseed-run-restore, since the RNG streams are process-global.
_OVERFIT_CHECK_LOCK = threading.Lock()


def candidate_config_with_selection(config: dict, selection_dir: str) -> dict:
    """The launch config choosing ``selection_dir`` over ``config``'s own data section would
    build: ``data.split`` replaced wholesale by ``{"selection_dir": selection_dir}``, and the
    stated ``scope`` and ``labels_dir`` dropped, since a bound run reads its scope and each
    sample's ground truth off the selection.
    """
    data_cfg_raw = config.get("data")
    data_cfg: dict = {**data_cfg_raw} if isinstance(data_cfg_raw, dict) else {}
    data_cfg.pop("scope", None)
    data_cfg.pop("labels_dir", None)
    data_cfg["split"] = {"selection_dir": selection_dir}
    return {**config, "data": data_cfg}

# Lazy imports of heavy dependencies inside tool functions to keep server startup fast.


def preflight_config(project: Path, config: dict, smoke: bool = False,
                     overfit: bool = False) -> dict:
    """Validate a training configuration before launching (:func:`_preflight`'s report)."""
    return _preflight(project, config, smoke=smoke, overfit=overfit)[0]


def _preflight(project: Path, config: dict, *, smoke: bool,
               overfit: bool) -> tuple[dict, ResolvedRun | None]:
    """Validate a training configuration before launching a run of ``project`` and resolve it once
    (:func:`~tcip_mcp.pipelines.data.split_construction.resolve_run`) when its structure admits
    that. Returns the report and the resolution, ``None`` when structure refused first.

    Config structure, one placement for everything::

        model_source: {builder, builder_kwargs, task}
        data: {images_dir, labels_dir?, scope}  # known loaders, or a bespoke
                                                # {dataset_source: {builder, ...}}
        batch_size, stages, mixed_precision, device, seed, ...   # every key
                                                # generic_trainer.train() reads, at the top
                                                # level; train()'s own docstring is the list
        evaluation: {trait, selection_metric, ...}
        training_source: optional custom train(ctx) loop.

    A nested ``training`` section is refused by name (``schemas.TrainConfigSchema``).

    Args:
        config: Full training configuration dict.
        smoke: When True, actually build the model and run ``check_model_contract`` (a train+eval
            forward at the run's resolved dims and img_size, every attribute head included). A
            contract failure is appended
            to ``issues`` and blocks the launch. For a task the contract has no synthetic batch
            schema for, one real batch of the resolved train dataset is used instead; if it yields
            none, that also blocks.
        overfit: When True (with ``smoke``), also run the voluntary ``overfit_check`` diagnostic
            and report it under ``overfit_check``, never gating. The stored report is already
            rendered (``model_contract.render_overfit_report``), with non-finite losses rendered.
    """
    from tcip_mcp.pipelines.data.label_queries import admits
    from tcip_mcp.pipelines.model_build import MODEL_SOURCE_KEY

    issues = _structural_issues(config)
    warnings: list[str] = []
    model_source = config.get(MODEL_SOURCE_KEY)
    data_cfg = config.get("data")
    data_cfg_dict: dict = data_cfg if isinstance(data_cfg, dict) else {}
    split_cfg_raw = data_cfg_dict.get("split")
    split_cfg_dict: dict = split_cfg_raw if isinstance(split_cfg_raw, dict) else {}
    if split_cfg_dict.get("selection_dir") and split_cfg_dict.get("redraw_within_selection"):
        warnings.append(
            "data.split.redraw_within_selection=true: this run redraws train and val inside the "
            "selection's own train and val samples at this seed; the selection's calibration side "
            "stays untouched."
        )

    eval_cfg = config.get("evaluation") or {}
    if not isinstance(eval_cfg, dict):
        issues.append(
            f"'evaluation' must be a mapping (trait/selection_metric/... keys), got "
            f"{type(eval_cfg).__name__}"
        )

    # Normalization provenance: per-band builder_kwargs statistics must carry which images produced them.
    sampling_record = None
    if isinstance(model_source, dict):
        bk = model_source.get("builder_kwargs")
        bk = bk if isinstance(bk, dict) else {}
        if bk.get("image_mean") is not None or bk.get("image_std") is not None:
            from pydantic import ValidationError

            from tcip_mcp.pipelines.schemas import ImageStatsSampling

            raw_sampling = model_source.get("image_stats_sampling")
            if isinstance(raw_sampling, dict):
                try:
                    sampling_record = ImageStatsSampling.model_validate(raw_sampling)
                except ValidationError:
                    sampling_record = None
            if sampling_record is None or not sampling_record.windows:
                issues.append(
                    "model_source.builder_kwargs carries image_mean/image_std with no "
                    "model_source.image_stats_sampling beside it: per-band statistics must carry "
                    "a non-empty 'windows' list and a 'pixel_fraction', naming which images they "
                    "were derived from (see derivations.image_stats_provenance)."
                )

    # The run resolved once, the way its launch records it: every leg below reads this.
    resolution: ResolvedRun | None = None
    if not issues:
        counts: dict[str, int] = {}
        try:
            resolution = resolve_run(config, project=project, tallies_out=counts)
        except Exception as exc:  # noqa: BLE001, whatever stops the resolution stops the launch
            issues.append(str(exc))
        # Trainable-sample coverage, never gating: a run admitting a fraction of its annotated
        # images would otherwise read "valid, no warnings" while training on far fewer.
        dropped = {k: v for k, v in counts.items() if not admits(k) and v}
        total, n_dropped = sum(counts.values()), sum(dropped.values())
        if n_dropped and total:
            warnings.append(
                f"data: {n_dropped}/{total} candidate images ({n_dropped / total:.0%}) will "
                f"not train, {dict(sorted(dropped.items()))}. "
                f"{total - n_dropped} stem(s) admitted.")

    result: dict = {"valid": False, "issues": issues, "warnings": warnings}
    if sampling_record is not None and resolution is None:
        result["image_stats_containment"] = "not_checked"
    if resolution is not None:
        from tcip_mcp.pipelines.data.split_construction import partition_samples

        resolved_data = resolution.record["data"]
        known = {str(Path(s.source).resolve())
                 for s in partition_samples(resolution.record["partition"])}
        from tcip_mcp.pipelines.data.selection import REFERENCE_SIDES

        reserved = sorted(f"{side}_ratio" for side in REFERENCE_SIDES
                          if split_cfg_dict.get(f"{side}_ratio"))
        if reserved and "spatial_manifest" not in resolved_data["split"]:
            issues.append(
                f"data.split {reserved} have no effect: this run over {len(known)} admitted "
                "sources did not resolve to the single-source spatial-strip split a reserved "
                "region is cut from (a detection task with tiling enabled over one admitted "
                "source); draw a selection with draw_splits for a reference.")
        if sampling_record is not None:
            bad = sorted({label for label, _ in sampling_record.windows
                          if str(Path(label).resolve()) not in known})
            result["image_stats_containment"] = "checked"
            if bad:
                issues.append(
                    f"model_source.image_stats_sampling names path(s) {bad} outside this "
                    f"run's own {len(known)} source(s)."
                )

    # Smoke: build the model and run the correctness contract at the resolved dims, so a broken
    # bespoke builder is caught here (before the training subprocess spawns) rather than surfacing
    # only as run.status='failed'. Only attempt once the structural checks pass, otherwise the
    # config can't build and the contract would just re-report the same failure. Overfit stays a
    # voluntary, non-gating diagnostic (a valid model can fail 20 steps on noise).
    if smoke and not issues:
        assert resolution is not None, "a config with no issue resolved"
        try:
            from tcip_mcp.pipelines.model_build import (
                build_model, recorded_model_dims, resolve_contract_dims,
            )
            from tcip_mcp.pipelines.model_contract import (
                check_model_contract, overfit_check, render_overfit_report,
            )

            task = run_task(config)
            resolved_config = {**config, "data": resolution.record["data"]}
            built_at = recorded_model_dims(resolved_config)
            dims = resolve_contract_dims(resolved_config, task, built_at)
            model = build_model(resolved_config, built_at)
            report = check_model_contract(model, task, dims=dims)
            batch, why_no_batch = None, None
            if report.get("not_smokeable"):
                # No synthetic batch for this task or this width: smoke against a real batch from
                # the run's own dataset, the only reference the platform has not guessed at.
                batch, why_no_batch = _one_real_batch(task, resolution.train_ds)
                if batch is not None:
                    report = check_model_contract(model, task, sample_batch=batch)
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
                from tcip_mcp.pipelines.training.generic_trainer import (
                    capture_rng_state, restore_rng_state,
                )

                # Same batch the contract used, otherwise this re-synthesizes and reports a false
                # "does not learn" for exactly the bespoke tasks the real-batch path exists for.
                with _OVERFIT_CHECK_LOCK:
                    rng_state = capture_rng_state()
                    try:
                        raw_report = overfit_check(model, task, sample_batch=batch, dims=dims)
                    except Exception as exc:  # noqa: BLE001, becomes the report's issue only
                        raw_report = {
                            "passed": False, "losses": [], "initial": None, "final": None,
                            "issue": f"overfit check failed: {exc}",
                        }
                    finally:
                        restore_rng_state(rng_state)
                # Rendered before storage: this tool answers over JSON-RPC, which a raw non-finite
                # loss cannot cross, so a diverging model's report is sanitized here, once.
                result["overfit_check"] = render_overfit_report(raw_report)
        except Exception as exc:  # noqa: BLE001, a build/contract crash is itself a blocking issue
            issues.append(f"model smoke build failed: {exc}")

    result["valid"] = len(issues) == 0
    return result, resolution


def _structural_issues(config: dict) -> list[str]:
    """What stops ``config`` before anything reads its data: its schema
    (``schemas.validate_train_config_schema``), a ``model_source``, ``training_source`` or
    ``data.dataset_source`` builder that is missing or does not import, and a ``data`` section
    that is missing or names no locations (``split_construction.data_dir_issues``)."""
    from tcip_mcp.pipelines.schemas import validate_train_config_schema
    from tcip_mcp.pipelines.model_build import (
        DATASET_SOURCE_KEY, MODEL_SOURCE_KEY, TRAINING_SOURCE_KEY, _import_dotted,
        import_source_builder,
    )

    issues: list[str] = list(validate_train_config_schema(config))

    model_source = config.get(MODEL_SOURCE_KEY)
    if not model_source:
        issues.append("Missing 'model_source' section")
    elif not isinstance(model_source, dict) or not model_source.get("builder"):
        issues.append("model_source must be a dict with a 'builder' (module:function)")
    else:
        try:
            import_source_builder(model_source)
        except Exception as exc:
            issues.append(f"model_source.builder not importable: {exc}")

    training_source = config.get(TRAINING_SOURCE_KEY)
    if training_source is not None:
        if not isinstance(training_source, str) or not training_source:
            issues.append("training_source must be a non-empty 'module:function' string")
        else:
            try:
                _import_dotted(training_source)
            except Exception as exc:
                issues.append(f"training_source not importable: {exc}")

    data_cfg = config.get("data")
    if not data_cfg:
        issues.append("Missing 'data' section")
    elif not isinstance(data_cfg, dict):
        issues.append("'data' must be a dict")
    else:
        if data_cfg.get(DATASET_SOURCE_KEY) is not None:
            dataset_source = data_cfg[DATASET_SOURCE_KEY]
            if not isinstance(dataset_source, dict) or not dataset_source.get("builder"):
                issues.append(
                    "data.dataset_source must be a dict with a 'builder' (module:function)")
            else:
                try:
                    import_source_builder(dataset_source)
                except Exception as exc:
                    issues.append(f"data.dataset_source.builder not importable: {exc}")
        issues.extend(data_dir_issues(data_cfg))
    return issues


def open_run(
    run_dir: Path, config: dict, resolved: dict | None, *, relaunched_from: str | None = None, resume_from: str | None = None,
    max_wall_clock_seconds: float | None = None, model_contract: dict | None = None,
    trial_params: dict | None = None,
) -> None:
    """Open the run directory ``run_dir`` (``experiments.open_run_directory``) with its
    ``run.json``: ``config`` as the run's input, ``resolved`` as what that input resolved to
    (``split_construction.resolve_run``'s record, ``None`` for an HPO trial whose resolution
    failed, which then ends ``failed``), the environment and the dataset identity of
    ``config``'s own data section (``split_construction.dataset_identity``), a bespoke run's
    sources copied into the directory, the run it was relaunched from and the checkpoint it
    resumes from, its wall clock, the model contract preflight proved, and an HPO
    trial's sampled point. A seed is drawn onto ``config`` when it states none. Refuses an
    existing directory (``experiments.RunDirectoryExists``)."""
    from tcip_mcp.pipelines.data.split_construction import dataset_identity, partition_samples
    from tcip_mcp.pipelines.model_build import capture_env, snapshot_model_source
    from tcip_mcp.pipelines.training.run_registry import draw_seed_if_unset

    draw_seed_if_unset(config)
    dataset_id, fingerprint = dataset_identity(
        config.get("data") or {}, partition_samples(resolved["partition"]) if resolved else ())
    experiments.open_run_directory(run_dir, lambda directory: {
        "created": now_iso(), "config": config, "resolved": resolved,
        "environment": capture_env(),
        "dataset": {"id": dataset_id, "fingerprint": fingerprint},
        "source": snapshot_model_source(config, directory),
        "relaunched_from": relaunched_from, "resume_from": resume_from,
        "max_wall_clock_seconds": max_wall_clock_seconds, "model_contract": model_contract,
        "trial_params": trial_params,
    })


@tool()
def launch_training(
    project: Path, config: dict, resume_from: str = "",
    max_wall_clock_seconds: float | None = None, overfit_check: bool = False,
    relaunched_from: str | None = None, *, actor: str | None,
) -> dict:
    """Launch a training run by ``actor`` in an isolated subprocess from a bespoke
    ``model_source`` builder.

    The run's training body (dataset build, model forward/backward, checkpointing) executes in a
    separate OS process, writing into the run's own directory
    (``<project>/.tcip/experiments/<experiment_id>/``). Use monitor_training to monitor progress
    and cancel_training to stop a run. The platform itself stops a run only when it is dead (two
    consecutive full training passes with no finite batch loss) or stagnant against its own
    validation metric (early stopping); a run launched with no validation loader gets divergence as
    its only automatic stop.

    Writes the run's ``run.json`` once, then the act's one audit line naming the minted
    ``experiment_id`` (``AuditEntryNotWritten`` when it cannot be appended, and no subprocess
    starts), then starts the subprocess.

    Args:
        config: Full training configuration dict: model_source and data, with every training
            setting (batch_size, stages, evaluation, seed, device) at the top level. The run's
            id is minted here and answered as ``experiment_id``; a config naming one refuses.
        resume_from: Optional path to a ``checkpoint_epoch_*.pt`` to resume from (restores model +
            optimizer + scheduler + scaler and continues) in this new run's directory.
        max_wall_clock_seconds: Optional hard timeout. The run stops at its next batch boundary
            past it and ends failed naming it; a process still alive one heartbeat window after
            it is terminated and reads interrupted. Omit for no timeout (the default).
        overfit_check: When True, runs the voluntary ``overfit_check`` diagnostic (twenty optimizer
            steps at the training tile edge, on the CPU, inside this synchronous call) on the same
            batch the contract proved, before the subprocess spawns, and records the result on the
            run's ``model_contract`` under ``overfit_check``. Never gates. Default False.
        relaunched_from: The run this one relaunches, recorded as its lineage.

    Before the run's directory exists: preflight, normalization, the model contract, and the
    dataset identity read (an unreadable identity refuses the launch by name). A spawn failure
    after it leaves a directory with no process, which reads ``interrupted`` once its heartbeat
    window passes.
    """
    check_json_value(config, path="config")
    if "experiment_id" in config:
        return {"error": "launch_training: config.experiment_id is not a setting; the platform "
                         "mints every run's id and answers it as experiment_id."}
    # smoke=True: build the model and run the correctness contract before spawning the training
    # subprocess, so a broken builder returns here instead of wasting a full audited run.
    validation, resolution = _preflight(project, config, smoke=True, overfit=overfit_check)
    if not validation["valid"]:
        return {"error": "Invalid config", "issues": validation["issues"]}
    assert resolution is not None, "a valid config resolved"

    # The top-level key, never the smoke sub-report: overfit_check runs beside the contract's
    # build, on the same batch; preflight_config already rendered it for storage.
    rendered_overfit_report = validation.get("overfit_check")

    smoke_report = validation.get("smoke") or {}
    model_contract_record = {
        "subject": "the model as built at launch, before any training step",
        "gating": True,
        "batch_source": smoke_report.get("batch_source"),
        "dims": smoke_report.get("dims"),
        "issues": smoke_report.get("issues", []),
        "gradient_magnitudes": smoke_report.get("gradient_magnitudes"),
        "operating_point_knobs": smoke_report.get("operating_point_knobs"),
        "overfit_check": rendered_overfit_report,
    }

    # A copy: the launch records onto it, never onto the caller's own dict.
    config = dict(config)
    experiment_id = experiments.mint_experiment_id()
    try:
        run_dir = experiments.experiment_dir(experiment_id, project=project)
        open_run(run_dir, config, resolution.record, relaunched_from=relaunched_from,
                 resume_from=resume_from or None, max_wall_clock_seconds=max_wall_clock_seconds,
                 model_contract=model_contract_record)
    except (StoreError, ValueError, OSError) as exc:
        return {"error": f"launch_training: {exc}"}
    record_event_or_raise("launch_training", {
        "experiment_id": experiment_id, "relaunched_from": relaunched_from,
        "resume_from": resume_from or None}, actor=actor, scope=project)

    proc = _start_worker(run_dir, _child_env_for_launch(config))

    if max_wall_clock_seconds is not None:
        _watch_wall_clock(proc, max_wall_clock_seconds)

    # Keyed by its own log directory (the manager's default), never by the run id.
    tb_info = {}
    try:
        from tcip_mcp.pipelines.training.tensorboard_manager import launch_tensorboard
        tb_info = launch_tensorboard(str(experiments.board_of(run_dir)))
    except Exception:
        pass  # TensorBoard launch is best-effort

    return {
        "experiment_id": experiment_id,
        "status": "launched",
        "output_dir": str(run_dir),
        "tensorboard": tb_info,
        "pid": proc.pid,
        "overfit_check": rendered_overfit_report,
    }


def _worker_env() -> dict[str, str]:
    """This process's environment with its import search path propagated via ``PYTHONPATH``."""
    from tcip_mcp.pipelines.model_build import child_pythonpath

    return {**os.environ, "PYTHONPATH": child_pythonpath()}


def _child_env_for_launch(config: dict) -> dict[str, str]:
    """Subprocess env for a run's launch (:func:`_worker_env`): round-robin GPU pinning
    (``CUDA_VISIBLE_DEVICES``) when the config names no device, untouched when it does.
    """
    env = _worker_env()
    if config.get("device"):
        return env

    try:
        import torch
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        count = 0
    if count > 1:
        env["CUDA_VISIBLE_DEVICES"] = str(next(_gpu_round_robin) % count)
    return env


def _start_worker(directory: Path, env: dict[str, str]) -> subprocess.Popen:
    """Start ``subprocess_worker`` over the opened run or sweep ``directory`` in its own process
    under ``env``."""
    return subprocess.Popen(
        [sys.executable, "-m", "tcip_mcp.pipelines.training.subprocess_worker",
         "--run-dir", str(directory)], env=env)


def _watch_wall_clock(proc: subprocess.Popen, timeout_seconds: float) -> None:
    """Daemon watcher: terminates ``proc`` if it is still alive one heartbeat window
    (``experiments.HEARTBEAT_STALE_SECONDS``) past ``timeout_seconds``. The child stops itself at
    its deadline and writes its own final status; a child that did not is hung, and reads
    interrupted."""
    def _watch() -> None:
        try:
            proc.wait(timeout=timeout_seconds + experiments.HEARTBEAT_STALE_SECONDS)
        except subprocess.TimeoutExpired:
            proc.terminate()

    threading.Thread(target=_watch, daemon=True).start()


def training_detail(project: Path, experiment_id: str) -> TrainingDetail | None:
    """The run, trial or sweep of ``project`` ``experiment_id`` names
    (``experiments.named_directory``) as its ``experiments.TrainingDetail``, its
    ``tensorboard_url`` the one a TensorBoard this process started serves its board at
    (``experiments.board_of``), or ``None``."""
    from tcip_mcp.pipelines.training.tensorboard_manager import running_url

    directory = experiments.named_directory(experiment_id, project=project)
    if directory is None:
        return None
    url = running_url(str(experiments.board_of(directory)))
    if (directory / SWEEP_FILE).is_file():
        return TrainingDetail(run=None, tensorboard_url=url,
                              sweep=experiments.read_sweep(directory, experiments.run_rows(project)))
    run = experiments.observe(directory)
    return TrainingDetail(sweep=None, tensorboard_url=url, run=experiments.run_summary(
        run, experiments.read_rows(run.metrics_log)[0],
        experiments.launch_declarations(project).get(run.directory.name)))


@tool()
def monitor_training(project: Path, experiment_id: str) -> dict:
    """Check a training run, a sweep's trial included, or a hyperparameter sweep, by the id that
    names it (:func:`training_detail`): ``{"run", "sweep", "tensorboard_url"}``. An id naming
    neither answers ``{"error": ...}``.

    Args:
        experiment_id: The run's id from launch_training, a trial's id from its sweep, or a
            sweep's id from run_hyperparameter_search.
    """
    detail = training_detail(project, experiment_id)
    return {"error": f"Run not found: {experiment_id}"} if detail is None else detail.model_dump()


def list_split_choices(project: Path, experiment_id: str) -> dict:
    """Every choice this config's own "Data" control offers a relaunch of ``experiment_id``: the
    data section its launch stated (``run.json``'s config), and every selection directory this
    project's own bound runs or the dataset's own ``splits`` directory hold, each
    compatibility-checked the way a launch would check it.

    Every check is :func:`selection_compatibility` over a selection this reader read itself,
    through :func:`~tcip_mcp.pipelines.data.selection.read_selection_checked`. A candidate
    selection is checked against the config :func:`candidate_config_with_selection` builds
    (``data.split`` replaced wholesale); "As recorded" is checked against the stated config
    unchanged, plus the directory-presence issues :func:`preflight_config` would raise.

    The listing: the selection directories other runs of this project bound (the picked run's
    own excluded, since it is "As recorded"), plus, when
    ``dataset_root_of(data.images_dir)`` resolves, that root's ``splits`` directory (offered only
    when something is recorded there directly) and every directory one level under it holding a
    selection. The own-binding exclusion and the candidate dedupe compare each directory by
    ``tcip_store.canonical_path``.

    Returns ``{"error": ...}`` for an unknown ``experiment_id``. Otherwise: ``{"as_recorded":
    {"case": "bound"|"drawn", "line": str, "compatible": bool, "reason": str | None}, "selections":
    [{"selection_dir": str, "enabled": bool, "reason": str | None, "seed": int | None, "group_by":
    str | None, "train": int, "val": int, "calibration": int, "replaced_split_keys": list[str]},
    ...]}``. The counts are the selection's whole sides. ``replaced_split_keys`` names every
    recorded ``data.split`` key other than ``selection_dir`` choosing any offered partition drops,
    the same set for every candidate. A bound config anchors the search for sibling selections at
    its own selection's first sample source; an unbound one at ``data.images_dir``. With neither
    readable, ``selections`` is empty.
    """
    from tcip_mcp.dataset_layout import dataset_root_of
    from tcip_mcp.pipelines.data.selection import read_selection_checked

    runs = experiments.run_observations(project)
    own = next((obs for obs in runs if obs.directory.name == experiment_id), None)
    if own is None:
        return {"error": f"Experiment not found: {experiment_id}"}
    config = own.record["config"]
    data_cfg = config.get("data") or {}
    split_cfg = data_cfg.get("split") or {}
    own_selection_dir = split_cfg.get("selection_dir")
    replaced_split_keys = sorted(
        k for k, v in split_cfg.items() if k != "selection_dir" and v is not None
    )

    own_selection = None
    if own_selection_dir:
        as_recorded = {"case": "bound", "line": "on the partition it bound",
                       "compatible": True, "reason": None}
        own_selection, own_error = read_selection_checked(own_selection_dir, project=project)
        if own_selection is None:
            as_recorded["compatible"] = False
            as_recorded["reason"] = own_error or (
                f"no selection recorded under {own_selection_dir}; run draw_splits first."
            )
        else:
            own_issues = selection_compatibility(data_cfg, own_selection, own_selection_dir)
            if own_issues:
                as_recorded["compatible"] = False
                as_recorded["reason"] = "; ".join(own_issues)
    elif own.resolution is None:
        as_recorded = {"case": "drawn", "line": "resolved to nothing at its launch",
                       "compatible": False, "reason": own.error}
    else:
        seed = own.resolution["partition"]["seed"]
        as_recorded = {
            "case": "drawn",
            "line": f"draws its split again with seed {seed} over the labels as they are now",
            "compatible": True, "reason": None,
        }

    dir_issues = data_dir_issues(data_cfg)
    if dir_issues:
        as_recorded["compatible"] = False
        combined_reason = list(dir_issues)
        prior_reason = as_recorded["reason"]
        if prior_reason:
            combined_reason.insert(0, str(prior_reason))
        as_recorded["reason"] = "; ".join(combined_reason)

    # A bound config names no locations, so its own selection's first sample source anchors the
    # sibling search the way an unbound config's images_dir does.
    if own_selection is not None and own_selection.samples:
        root_anchor = str(Path(own_selection.samples[0].source).parent)
    elif not own_selection_dir and not dir_issues:
        root_anchor = str(data_cfg["images_dir"])
    else:
        return {"as_recorded": as_recorded, "selections": []}

    own_norm = canonical_path(own_selection_dir) if own_selection_dir else None
    candidate_dirs: list[str] = []
    seen: set[str] = set()
    for other in runs:
        binding = other.resolution["partition"]["selection"] if other.resolution else None
        if other is own or binding is None:
            continue
        candidate = binding["selection_dir"]
        candidate_norm = canonical_path(candidate)
        if candidate_norm == own_norm or candidate_norm in seen:
            continue
        seen.add(candidate_norm)
        candidate_dirs.append(candidate)
    dataset_root = dataset_root_of(root_anchor)
    if dataset_root is not None:
        default_dir = str(dataset_root / "splits")
        default_norm = canonical_path(default_dir)
        if default_norm != own_norm and default_norm not in seen:
            seen.add(default_norm)
            candidate_dirs.append(default_dir)
        # One level down: where freeze_selection writes a frozen run's own partition.
        splits_dir = Path(default_dir)
        if splits_dir.is_dir():
            for sub in sorted(p for p in splits_dir.iterdir() if p.is_dir()):
                sub_norm = canonical_path(str(sub))
                if sub_norm == own_norm or sub_norm in seen:
                    continue
                seen.add(sub_norm)
                candidate_dirs.append(str(sub))

    selections: list[dict] = []
    for candidate_dir in candidate_dirs:
        selection, error_text = read_selection_checked(candidate_dir, project=project)
        if selection is None:
            if error_text is None:
                continue  # nothing recorded there; not a real candidate
            selections.append({
                "selection_dir": candidate_dir, "enabled": False, "reason": error_text,
                "seed": None, "group_by": None, "train": 0, "val": 0, "calibration": 0,
                "replaced_split_keys": replaced_split_keys,
            })
            continue
        candidate_config = candidate_config_with_selection(config, candidate_dir)
        issues = selection_compatibility(candidate_config["data"], selection, candidate_dir)
        counts = selection.counts()
        entry: dict = {
            "selection_dir": candidate_dir, "seed": selection.seed,
            "group_by": selection.group_by, "train": counts["train"],
            "val": counts["val"], "calibration": counts["calibration"],
            "replaced_split_keys": replaced_split_keys,
        }
        if issues:
            entry["enabled"] = False
            entry["reason"] = "; ".join(issues)
        else:
            entry["enabled"] = True
            entry["reason"] = None
        selections.append(entry)

    return {"as_recorded": as_recorded, "selections": selections}


@tool()
@audited
def cancel_training(project: Path, experiment_id: str, *, actor: str | None) -> dict:
    """Request graceful cancellation of a running training run, a sweep's trial or a whole
    hyperparameter sweep, by ``actor``, named by its id (``experiments.named_directory``).

    A run stops at its next batch/epoch boundary, still saves ``model_final.pt`` (so partial
    progress is recoverable), and writes its final status 'canceled'; a run whose divergence
    verdict lands first (two consecutive full training passes with no finite batch loss, checked
    ahead of cancellation at the same boundary) ends 'failed' instead, with no ``model_final.pt``.
    A sweep stops each running trial at its next batch boundary and launches no more. The
    returned ``state`` is the one read now, so it may still read 'running'.

    Refuses an id naming no run or sweep, and one that has already ended
    (``experiments.request_cancel``); a repeated request keeps the first one's time.

    Args:
        experiment_id: The run's, trial's or sweep's id.
    """
    directory = experiments.named_directory(experiment_id, project=project)
    if directory is None:
        return {"error": f"Run not found: {experiment_id}"}
    try:
        experiments.request_cancel(directory)
    except ValueError as exc:
        return {"error": str(exc)}
    return {"experiment_id": experiment_id, "cancel_requested": True,
            "state": experiments.observe(directory).state}


def inspect_compute_resources(project: Path) -> dict:
    """Report the host's current compute headroom, a fact to reason with before launching another
    concurrent training/HPO run of ``project``, not an enforced cap.

    Returns:
        ``cpu``: ``{logical_count, percent_used}``, ``percent_used`` is ``None`` without ``psutil``
            installed.
        ``memory``: ``{total_bytes, available_bytes}``, both ``None`` without ``psutil``.
        ``gpus``: ``[{index, free_bytes, total_bytes}, ...]``, always populated when CUDA is
            available (``torch.cuda.mem_get_info``, no extra dependency); ``[]`` otherwise.
        ``active_training_runs``: count of every training run whose state is ``"running"``
            (``experiments.run_observations``).
    """
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

    active = sum(1 for obs in experiments.run_observations(project) if obs.state == "running")

    return {"cpu": cpu, "memory": memory, "gpus": gpus, "active_training_runs": active}


_CANCEL_BEFORE_START_REASON = "canceled before the sweep's first trial started"
_CANCEL_DURING_RUN_REASON = "the sweep was canceled by request before it could finish"


def open_trial(sweep: Path, trial_id: str, point: dict) -> Path:
    """Open the trial ``trial_id`` of the sweep at ``sweep`` as a run directory beneath it named
    ``<sweep id>_<trial_id>``, and return it. The sweep's base config with ``point`` applied is
    resolved against its project through the one run producer (``split_construction.resolve_run``,
    at the sweep's own objective) and the directory opened (:func:`open_run`) with ``point`` as
    its ``trial_params``, whatever the resolution did: a resolution that fails is the trial's
    final status ``failed`` naming why."""
    record = experiments.observe(sweep).record
    config = _apply_hpo_params(record["input"]["base_config"], point)
    trial_dir = sweep / f"{sweep.name}_{trial_id}"
    try:
        resolved, error = resolve_run(config, project=experiments.project_of_run(sweep),
                                      objective=record["objective"]).record, None
    except Exception as exc:  # noqa: BLE001, whatever stops the resolution fails the trial
        resolved, error = None, str(exc)
    open_run(trial_dir, config, resolved, trial_params=point)
    if error is not None:
        experiments.write_final_status(trial_dir, "failed", error, checkpoint=None)
    return trial_dir


def _run_hpo_trial(point: dict, report, sweep: Path, trial_id: str) -> None:
    """Train one HPO trial of the sweep at ``sweep`` (:func:`open_trial`), reporting every
    ``selection`` its metrics log records, which is the trial's result. A trial whose point failed
    to resolve ends at its opening; otherwise its body runs through
    ``subprocess_worker.run_directory``. A sweep canceled before the trial started opens nothing.
    """
    from tcip_mcp.pipelines.training.subprocess_worker import run_directory

    if experiments.cancel_requested(sweep):
        return

    def epoch_cb(epoch: int, metrics: dict) -> None:
        if "selection" in metrics:
            report(float(metrics["selection"]))

    trial_dir = open_trial(sweep, trial_id, point)
    if (trial_dir / experiments.FINAL_STATUS_FILE).is_file():
        return
    try:
        run_directory(trial_dir, origin="hpo_trial", epoch_hook=epoch_cb)
    except Exception as exc:  # noqa: BLE001, run_directory recorded it as the trial's failure
        logger.warning("HPO trial %s failed: %s", trial_dir.name, exc)


@tool()
def run_hyperparameter_search(
    project: Path,
    base_config: dict,
    param_space: dict | None = None,
    n_trials: int = 5,
    search_alg: str = "random",
    scheduler: str = "asha",
    grace_period: int = 5,
    reduction_factor: int = 3,
    warm_start: bool = False,
    baseline_params: dict | None = None,
    max_concurrent: int = 1,
    resources_per_trial: dict | None = None,
    auto_tensorboard: bool = True,
    split_draws: int = 1,
    split_draw_seeds: list[int] | None = None,
    *,
    search_seed: int,
    trial_budget: int | None = None,
    relaunched_from: str | None = None,
) -> dict:
    """Run hyperparameter optimization on Ray Tune, training each trial for real.

    The search algorithm and trial scheduler are the caller's choice (call the ``hpo`` module's
    ``available_search_algs`` / ``available_schedulers`` for the live list):
      - ``search_alg``: ``random``/``grid`` (native), or a backend, ``optuna``, ``bayesopt``,
        ``hyperopt``, each constructed with ``search_seed``.
      - ``scheduler``: ``asha`` (async HyperBand), ``hyperband``, ``pbt``, ``median``, or ``none``
        to run every trial to completion.

    Trials optimize the objective ``base_config`` resolves to once (its selection metric and the
    direction that metric's declaration says is better), recorded in the sweep's input and in
    every trial's launch record; each trains under the base config's regime and reports the
    ``selection`` rows its metrics log records.

    Everything one sweep writes lands in its own directory, ``.tcip/experiments/<sweep_id>/``
    beside the project's runs, whose names it shares (:func:`open_sweep`, :func:`run_sweep`): its ``sweep.json`` written once before the first
    trial starts, carrying every argument this call resolved (``base_config``, the resolved
    ``param_space`` and the objective included); a heartbeat this call keeps while it runs; one
    run directory per trial, ``<sweep_id>_<ray trial id>``, a run like any other; Ray's own
    experiment store (also the TensorBoard logdir); and its final status, written once when the
    sweep ends. An existing directory refuses by name. The result, as ``monitor_training``'s at any
    time, is a ``TrainingDetail`` whose ``sweep`` is the sweep's group: its trial rows and what
    they amount to (``experiments.sweep_outcome``).

    Refuses (``{"error": ..., "issues": [...]}``, nothing minted): a ``base_config`` that fails
        preflight; an unimportable builder or training source, or a config with no ``data``
        section, at every point the search space could resolve a trial's config to; a
        ``param_space`` axis naming ``data.split.seed`` while ``split_draws`` draws at
        most one partition (:func:`caller_split_seed_refusal`); ``split_draws`` above 1 on an
        unbound, built-in detection config with tiling on that admits exactly one trainable source
        (:func:`_split_draws_refusal`); a ``split_draws`` that is not an integer, or below 1
        (:func:`_split_draws_argument_refusal`, checked first); and, on a launch above one draw or
        any call naming a ``trial_budget``, the bound refusals of :func:`_trial_budget_refusal`. A
        cancel (``cancel_training``) requested before or during the run ends the sweep
        ``{"state": "canceled", ...}``, its final status recording the same.

    Args:
        base_config: Base training config each trial modifies.
        param_space: Param-space dict (see ``hpo.get_default_space``); default when omitted.
            A ``data.split.seed`` axis is refused whenever ``split_draws`` draws at most one
            partition; above 1 it belongs to ``split_draws``/``split_draw_seeds``.
        n_trials: Number of trials; a whole number of at least one on a call that reads a
            ``trial_budget`` bound.
        search_alg: Search algorithm, see the list above.
        scheduler: Trial scheduler, see the list above.
        grace_period: Minimum epochs before a halving scheduler (``asha``/``hyperband``) can stop a
            trial early.
        reduction_factor: Halving factor for ``asha``/``hyperband`` (fraction of trials kept at
            each rung).
        warm_start: Seed the search with ``baseline_params`` as a known-good starting point.
        baseline_params: Hyperparameter values to seed the search with when ``warm_start=True``.
        max_concurrent: Trials to run at once (default 1, safe for single-GPU training).
        resources_per_trial: Ray resource request per trial, omit to derive one from the host's
            real GPU count and ``max_concurrent`` (see ``hpo._default_trial_resources``); an
            explicit value always wins.
        auto_tensorboard: Launch a TensorBoard over a completed sweep's directory, whose URL the
            answer's ``tensorboard_url`` names.
        relaunched_from: The sweep this one replays, recorded in this sweep's input, ``None``
            when this sweep was not a relaunch. Refused when it names no sweep directory of this
            project. A stated ``trial_budget`` is checked on a relaunch exactly as on a launch.
        search_seed: The search algorithm's own seed, recorded in the sweep's input; required, and
            distinct from the split seed a trial's data draw uses.
        trial_budget: The most trials this sweep may launch, counted the way Ray will launch them
            (see :func:`~tcip_mcp.pipelines.training.hpo.planned_trial_count`). Required above one
            draw on a launch that is not a relaunch; checked whenever stated, including at one
            draw. Recorded in the sweep's input beside ``split_draws``, ``None`` when the caller
            stated none.
        split_draws: Above 1, adds ``data.split.seed`` to the search space as a grid over
            ``split_draw_seeds`` (default: the base config's own ``data.split.seed`` plus the draw
            index), paired with every sampled point through Ray's
            own ``BasicVariantGenerator(constant_grid_search=True)`` so each point trains once per
            seed. A ``base_config`` bound to a selection gains
            ``data.split.redraw_within_selection: true`` on its own copy (``data.split.seed``
            stated), so every trial redraws train and val inside the
            selection's own train-plus-val samples, calibration untouched; refused when those
            samples resolve to fewer than two foreground groups. Otherwise refused when
            ``data.auto_val`` is off, ``search_alg`` is not a native one
            (``random``/``grid``/``variant_generator``), ``scheduler`` is not ``none``,
            ``split_draw_seeds`` is given at a length other than ``split_draws`` or names the same
            seed twice, ``warm_start``'s ``baseline_params`` names ``data.split.seed``,
            ``param_space`` already sweeps ``data.split.seed`` itself, or ``param_space`` sweeps
            any other ``data.*`` axis. The outcome groups trials by point (params minus the seed)
            and chooses the best by mean over each point's draws; see ``best_value_spread``, and
            every point's own block at ``split_sensitivity``. 1 is the default; zero or below is
            refused naming the value.
        split_draw_seeds: The seeds ``split_draws`` pairs with every sampled point, one per draw;
            omit for the derived default (see ``split_draws``).
    """
    opened = open_sweep(
        project, base_config, param_space, n_trials=n_trials, search_alg=search_alg, scheduler=scheduler,
        grace_period=grace_period, reduction_factor=reduction_factor, warm_start=warm_start,
        baseline_params=baseline_params, max_concurrent=max_concurrent,
        resources_per_trial=resources_per_trial, split_draws=split_draws,
        split_draw_seeds=split_draw_seeds, search_seed=search_seed, trial_budget=trial_budget,
        relaunched_from=relaunched_from, actor=None)
    if isinstance(opened, dict):
        return opened
    group = run_sweep(opened)
    url = None
    if auto_tensorboard and group.state == "completed":
        from tcip_mcp.pipelines.training.tensorboard_manager import launch_tensorboard

        url = launch_tensorboard(str(experiments.board_of(opened))).get("url")
    return TrainingDetail(run=None, sweep=group, tensorboard_url=url).model_dump()


def open_sweep(
    project: Path, base_config: dict, param_space: dict | None, *, n_trials: int, search_alg: str,
    scheduler: str, grace_period: int, reduction_factor: int, warm_start: bool,
    baseline_params: dict | None, max_concurrent: int, resources_per_trial: dict | None,
    split_draws: int, split_draw_seeds: list[int] | None,
    search_seed: int, trial_budget: int | None, relaunched_from: str | None, actor: str | None,
) -> Path | dict:
    """Check a sweep's arguments: the first point the search space could resolve a trial to
    (:func:`_preflight_points`) resolved once (:func:`_preflight`), whose objective every trial
    records and whose partition answers whether its draws can vary (:func:`_spatial_draws_issue`),
    then the structure (:func:`_structural_issues`) of every other point. Create the sweep's
    directory under ``project``, named by ``experiments.mint_experiment_id("hpo")``, with its
    ``sweep.json`` written once, carrying the objective and
    every argument resolved as its ``input``, then the act's one audit line by ``actor`` naming
    the sweep (``AuditEntryNotWritten`` when it cannot be appended). Returns the opened sweep's
    directory, or the refusal ``{"error", "issues"}`` with nothing created."""
    from tcip_mcp.pipelines.training.hpo import get_default_space, split_draw_search_space

    if param_space is None:
        param_space = get_default_space()

    # Both reach a written record: the space into the sweep's input, the base config into every
    # trial's run.json once a sampled point is applied to it.
    check_json_value(param_space, path="param_space")
    check_json_value(base_config, path="base_config")

    if relaunched_from is not None and experiments.find_sweep(relaunched_from,
                                                              project=project) is None:
        return {"error": f"relaunched_from names no sweep of this project: {relaunched_from!r}",
                "issues": []}

    argument_refusal = _split_draws_argument_refusal(split_draws)
    if argument_refusal is not None:
        return {"error": argument_refusal, "issues": []}

    # Below the leg that makes split_draws an integer, so this comparison never meets another
    # type, and computed once so every leg that reads it agrees on whether a bound is read.
    reads_bound = trial_budget is not None or (split_draws > 1 and relaunched_from is None)

    # A bound base_config admitted to split_draws redraws inside its selection from here on.
    base_config = _base_config_for_split_draws(base_config, split_draws)

    # Checked ahead of preflight, so its own reason is what an auto_val refusal reads as, not
    # whatever preflight would have hit first.
    draws_refusal = _split_draws_refusal(
        base_config, param_space, search_alg, scheduler,
        split_draws, split_draw_seeds, warm_start, baseline_params)
    if draws_refusal is not None:
        return {"error": draws_refusal, "issues": []}

    seed_axis_refusal = caller_split_seed_refusal(param_space, split_draws)
    if seed_axis_refusal is not None:
        return {"error": f"{seed_axis_refusal.reason} {seed_axis_refusal.remedy}", "issues": []}

    (first_label, first_point), *rest = _preflight_points(param_space)
    preflight, resolution = _preflight(project, _apply_hpo_params(base_config, first_point),
                                       smoke=False, overfit=False)
    if resolution is None or not preflight["valid"]:
        return {"error": f"the sweep's base config fails preflight at {first_label}",
                "issues": preflight["issues"]}
    for label, point in rest:
        try:
            issues = _structural_issues(_apply_hpo_params(base_config, point))
        except ValueError as exc:
            issues = [str(exc)]
        if issues:
            return {"error": f"the sweep's base config fails preflight at {label}",
                    "issues": issues}
    spatial_refusal = _spatial_draws_issue(resolution.record, split_draws)
    if spatial_refusal is not None:
        return {"error": spatial_refusal, "issues": []}
    objective = resolution.record["objective"]

    search_param_space, resolved_draw_seeds = split_draw_search_space(
        param_space, base_config, split_draws, split_draw_seeds)

    budget_refusal = _trial_budget_refusal(
        reads_bound=reads_bound, split_draws=split_draws, n_trials=n_trials,
        search_alg=search_alg, warm_start=warm_start, trial_budget=trial_budget,
        relaunched_from=relaunched_from, search_param_space=search_param_space,
        baseline_params=baseline_params,
    )
    if budget_refusal is not None:
        return {"error": budget_refusal, "issues": []}

    try:
        directory = experiments.create_run_directory(experiments.experiment_dir(
            experiments.mint_experiment_id("hpo"), project=project))
    except (StoreError, ValueError, OSError) as exc:
        return {"error": str(exc), "issues": []}
    record = {"created": now_iso(), "objective": objective, "input": {
        "n_trials": n_trials, "search_alg": search_alg, "scheduler": scheduler,
        "grace_period": grace_period, "reduction_factor": reduction_factor,
        "max_concurrent": max_concurrent, "warm_start": warm_start,
        "baseline_params": baseline_params, "resources_per_trial": resources_per_trial,
        "param_space": param_space, "base_config": base_config,
        "relaunched_from": relaunched_from, "split_draws": split_draws,
        "split_draw_seeds": resolved_draw_seeds, "search_seed": search_seed,
        "trial_budget": trial_budget,
    }}
    experiments.write_record(directory / SWEEP_FILE, record)
    record_event_or_raise("open_sweep", {"sweep_id": directory.name,
                                         "relaunched_from": relaunched_from},
                          actor=actor, scope=project)
    return directory


def launch_sweep(project: Path, source: Path, *, actor: str | None) -> dict:
    """Open a new sweep of ``project`` by ``actor`` (:func:`open_sweep`, which mints its id) from
    the recorded input of the sweep directory ``source``, relaunched from it, and run it
    (:func:`run_sweep`) in its own worker process. Returns ``{"sweep_id", "status": "launched"}``,
    or the refusal dict when that input no longer opens."""
    given = experiments.observe(source).record["input"]
    opened = open_sweep(project, **{**given, "relaunched_from": source.name}, actor=actor)
    if isinstance(opened, dict):
        return opened
    _start_worker(opened, _worker_env())
    return {"sweep_id": opened.name, "status": "launched"}


def run_sweep(directory: Path) -> SweepGroup:
    """Run the opened sweep at ``directory`` (:func:`open_sweep`) on Ray Tune to its final status
    (``experiments.write_final_status``), keeping its heartbeat while it runs and training each
    trial through :func:`_run_hpo_trial` in a run directory named for the sweep and Ray's trial
    id. A cancel requested before or during the run ends it ``canceled``. Returns the ended
    sweep's group (``experiments.read_sweep``)."""
    import uuid

    from tcip_mcp.pipelines.training.hpo import split_draw_search_space, tune_search

    project = experiments.project_of_run(directory)
    record = experiments.observe(directory).record
    given = record["input"]

    def end(state: str, error: str | None) -> SweepGroup:
        experiments.write_final_status(directory, state, error)
        return experiments.read_sweep(directory, experiments.run_rows(project))

    if experiments.cancel_requested(directory):
        return end("canceled", _CANCEL_BEFORE_START_REASON)

    def objective_fn(config: dict, report) -> None:
        try:
            from ray import tune as _tune
            tid = _tune.get_context().get_trial_id()
        except Exception:
            tid = uuid.uuid4().hex[:8]
        _run_hpo_trial(config, report, directory, tid)

    search_param_space, _ = split_draw_search_space(
        given["param_space"], given["base_config"], given["split_draws"], given["split_draw_seeds"])
    stop_heartbeat = experiments.keep_heartbeat(directory)
    try:
        tune_search(
            objective_fn=objective_fn,
            param_space=search_param_space,
            metric="objective",
            mode="max" if record["objective"]["higher_is_better"] else "min",
            num_samples=given["n_trials"],
            search_alg=given["search_alg"],
            scheduler=given["scheduler"],
            grace_period=given["grace_period"],
            reduction_factor=given["reduction_factor"],
            seed=given["search_seed"],
            warm_start=given["warm_start"],
            baseline_params=given["baseline_params"],
            max_concurrent=given["max_concurrent"],
            sweep_dir=directory,
            resources_per_trial=given["resources_per_trial"],
            split_draws=given["split_draws"],
        )
    except Exception as exc:
        if experiments.cancel_requested(directory):
            return end("canceled", _CANCEL_DURING_RUN_REASON)
        end("failed", str(exc))
        raise
    finally:
        stop_heartbeat.set()

    if experiments.cancel_requested(directory):
        return end("canceled", _CANCEL_DURING_RUN_REASON)
    return end("completed", None)


def _apply_hpo_params(base_config: dict, params: dict) -> dict:
    """Apply flat HPO params onto a deep copy of ``base_config``, its progressive-unfreeze
    schedule left untouched:

      - ``lr``           -> ``optimizer["head_lr"]``, plus ``optimizer["backbone_lr"]`` scaled by
                            whatever backbone/head ratio ``base_config`` already expressed
      - ``weight_decay`` -> ``optimizer["weight_decay"]``
      - anything else    -> the top level of ``cfg`` (``batch_size`` included)
    """
    import copy

    cfg = copy.deepcopy(base_config)

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
        elif key == "weight_decay":
            cfg.setdefault("optimizer", {})["weight_decay"] = value
        elif "." in key:
            # A dotted path (e.g. "model_source.builder") reaches the nested field it names,
            # rather than landing as a literal top-level key nothing reads.
            *path, leaf = key.split(".")
            node = cfg
            for i, part in enumerate(path):
                if not isinstance(node, dict):
                    raise ValueError(
                        f"hpo param {key!r} cannot reach {'.'.join(path[:i])!r}: base_config "
                        f"holds a {type(node).__name__} there, not a mapping to walk into"
                    )
                node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(
                    f"hpo param {key!r} cannot be set: base_config holds a "
                    f"{type(node).__name__} at {'.'.join(path)!r}, not a mapping"
                )
            node[leaf] = value
        else:
            cfg[key] = value
    return cfg


def _base_config_for_split_draws(base_config: dict, split_draws: int) -> dict:
    """``base_config`` as a sweep over ``split_draws`` draws is minted from: unchanged unless
    ``split_draws`` is above 1 and the config is bound to a selection, in which case a copy carries
    ``data.split.redraw_within_selection: true``; a config stating no ``data.split.seed`` refuses
    (``ValueError``).
    """
    if split_draws <= 1:
        return base_config
    data_cfg = base_config.get("data") or {}
    split_cfg = data_cfg.get("split") or {}
    if not split_cfg.get("selection_dir"):
        return base_config
    new_split = {**split_cfg, "redraw_within_selection": True}
    new_split["seed"] = split_seed(split_cfg)
    return {**base_config, "data": {**data_cfg, "split": new_split}}


def _split_draws_argument_refusal(split_draws: object) -> str | None:
    """Whether ``split_draws`` itself is a draw count at all, checked before every other leg
    (including :func:`_base_config_for_split_draws`): a value that is not an ``int`` (a ``bool`` is
    an ``int`` and reads as the integer it names) refuses by name, and a value below one refuses by
    name.
    """
    if not isinstance(split_draws, int):
        return (f"split_draws={split_draws!r} is not a draw count; pass a whole number of "
                "partitions to pair.")
    if split_draws < 1:
        return (f"split_draws={split_draws} pairs no partition; a sweep pairs at least one, so "
                "pass 1 (the default, one draw and no spread) or the number of partitions to "
                "pair.")
    return None


def _trial_budget_refusal(
    *, reads_bound: bool, split_draws: int, n_trials: object, search_alg: str, warm_start: bool,
    trial_budget: object, relaunched_from: str | None, search_param_space: dict,
    baseline_params: dict | None,
) -> str | None:
    """Whether this call's own trial count fits the bound it must state or check, run only when
    ``reads_bound`` says one is read (a launch, never a relaunch, above one draw; or any call
    naming ``trial_budget``).

    Runs its two argument clauses first (an ``n_trials`` or ``trial_budget`` that is not a positive
    ``int``, a ``bool`` excluded by name), then counts Ray's own variant count over
    ``search_param_space`` via :func:`~tcip_mcp.pipelines.training.hpo.planned_trial_count` (a
    space that cannot be counted or generated answers its own refusal, naming the exception), then
    its two bound clauses: a launch above one draw naming no ``trial_budget`` refuses naming the
    budget Ray's own count would admit; a stated ``trial_budget`` the count exceeds refuses naming
    the count, the budget, and whether even one draw exceeds it (``count // split_draws`` is one
    draw's own count). Everything else answers ``None``.
    """
    if not reads_bound:
        return None
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials <= 0:
        return (f"n_trials={n_trials!r} is not a count of trials; Ray runs a negative count as "
                "an unbounded sweep and a zero count as none, so pass the number of sampled "
                "points, a whole number of at least one.")
    if trial_budget is not None and (
        not isinstance(trial_budget, int) or isinstance(trial_budget, bool) or trial_budget <= 0
    ):
        return (f"trial_budget={trial_budget!r} is not a count of trials; pass the most trials "
                "this sweep may launch, a whole number of at least one, or omit it at one draw.")

    from tcip_mcp.pipelines.training.hpo import planned_trial_count

    try:
        count = planned_trial_count(
            search_param_space, n_trials, search_alg, split_draws, warm_start, baseline_params)
    except (ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
        return (f"the sweep's param_space cannot be counted as a search space: {exc!r}; correct "
                "the axis that exception names, since tune_search builds the same space and "
                "would meet it too, or run at split_draws=1 stating no trial_budget, where no "
                "count is taken.")

    per_draw = count // split_draws
    warm_state = "on" if warm_start else "off"
    built = (f"Ray's own variant count over the space this call builds: n_trials={n_trials} "
             f"sample(s), search_alg={search_alg!r}, warm start {warm_state}, {per_draw} per draw")

    if split_draws > 1 and trial_budget is None and relaunched_from is None:
        return (f"split_draws={split_draws} pairs every sampled point with {split_draws} "
                f"partitions, so this sweep would launch {count} trials ({built}), a total the "
                f"call never stated; pass trial_budget={count} ({count} admits it as "
                "configured), or run at split_draws=1.")
    if trial_budget is not None and count > trial_budget:
        if per_draw > trial_budget:
            return (f"this sweep would launch {count} trials ({built}) against "
                    f"trial_budget={trial_budget}, which this sweep exceeds at one draw "
                    f"({per_draw} trials per draw); lower n_trials, narrow the gridded axes, or "
                    "state a larger trial_budget.")
        admitted_draws = trial_budget // per_draw
        return (f"this sweep would launch {count} trials ({built}) against "
                f"trial_budget={trial_budget}, which admits at most {admitted_draws} draw(s) at "
                f"these settings; lower split_draws to {admitted_draws}, lower n_trials, narrow "
                "the gridded axes, or state a larger trial_budget.")
    return None


def _split_draws_refusal(
    base_config: dict, param_space: dict | None, search_alg: str,
    scheduler: str, split_draws: int, split_draw_seeds: list[int] | None, warm_start: bool,
    baseline_params: dict | None,
) -> str | None:
    """Every reason of the paired path's own a sweep refuses ``split_draws`` above 1 for before
    its first point resolves. ``None`` when nothing here objects, and for one draw. A
    ``base_config`` bound to a selection skips the ``auto_val`` leg; whether its selection admits
    a redraw is the first point's resolution's answer."""
    if split_draws <= 1:
        return None
    from tcip_mcp.pipelines.training.hpo import (
        SPLIT_DRAW_SEED_KEY, _NATIVE_SEARCH, _NO_SCHEDULER, search_alg_key,
    )

    data_cfg = base_config.get("data") or {}
    split_cfg = data_cfg.get("split") or {}
    bound = bool(split_cfg.get("selection_dir"))
    if not bound and not data_cfg.get("auto_val", True):
        return ("split_draws needs a drawn validation split, and base_config sets "
                "data.auto_val=False.")
    if search_alg_key(search_alg) not in _NATIVE_SEARCH:
        native = sorted(_NATIVE_SEARCH)
        return (f"split_draws pairs a grid axis through Ray's own BasicVariantGenerator, which "
                f"only a native search_alg ({native}) builds; search_alg={search_alg!r} does not.")
    if (scheduler or "none").lower() not in _NO_SCHEDULER:
        return (f"split_draws makes each draw a blocked comparison, and a pruning scheduler "
                f"({scheduler!r}) could end one draw before another completes; pass "
                "scheduler='none' with split_draws.")
    if split_draw_seeds is not None and len(split_draw_seeds) != split_draws:
        return (f"split_draw_seeds has {len(split_draw_seeds)} seed(s) but split_draws="
                f"{split_draws}: one seed per draw.")
    if split_draw_seeds is not None and len(set(split_draw_seeds)) != len(split_draw_seeds):
        repeated = sorted({s for s in split_draw_seeds if split_draw_seeds.count(s) > 1})
        return (f"split_draw_seeds repeats {repeated}: a spread over the same partition drawn "
                "twice is not a spread, and group_split_draws only counts a point eligible on "
                "its distinct planned seeds, not on one seed completed twice.")
    if warm_start and baseline_params and SPLIT_DRAW_SEED_KEY in baseline_params:
        return (f"warm_start's baseline_params names {SPLIT_DRAW_SEED_KEY!r}: Ray's own "
                "preset-variant pinning would pin every draw to that one seed instead of "
                f"pairing the grid; drop {SPLIT_DRAW_SEED_KEY!r} from baseline_params.")
    if SPLIT_DRAW_SEED_KEY in (param_space or {}):
        return (f"param_space already sweeps {SPLIT_DRAW_SEED_KEY}, the same axis split_draws "
                "adds as a paired grid; state the crossing through split_draws/"
                "split_draw_seeds, not a second data.split.seed axis.")
    other_data_axes = sorted(
        k for k in (param_space or {}) if k.startswith("data.") and k != SPLIT_DRAW_SEED_KEY)
    if other_data_axes:
        return (f"split_draws pairs the identical partition with every sampled point at each "
                f"draw, and param_space sweeps {other_data_axes} beside it: a data.* axis other "
                f"than {SPLIT_DRAW_SEED_KEY} changes what a point admits or how it draws, so "
                "draw k would no longer be the same partition for every point.")
    return None


def _spatial_draws_issue(resolved: dict, split_draws: int) -> str | None:
    """The refusal ``split_draws`` above 1 meets when the sweep's ``resolved`` first point split
    one source over its own pixels (its ``data.split.spatial_manifest``), a partition no draw of
    ``data.split.seed`` varies; ``None`` otherwise."""
    if split_draws <= 1 or resolved["data"]["split"].get("spatial_manifest") is None:
        return None
    return (
        f"split_draws={split_draws} redraws the split, and base_config's one admitted source is "
        "split over its own pixels, a partition placed by declared order that does not vary with "
        "data.split.seed: no draw holds a different partition out and the spread would be "
        "training-seed noise. Run at split_draws=1, or sweep a dataset with two or more admitted "
        "sources or a config bound to a selection."
    )


class SeedAxisRefusal(NamedTuple):
    """Why a caller-supplied ``data.split.seed`` axis in ``param_space`` refuses at one draw:
    ``reason`` names the fact and the breeder's own next step; ``remedy`` names the Ray mechanics
    and what the tool's caller passes instead.
    """

    reason: str
    remedy: str


_SEED_AXIS_REASON = (
    "this sweep varied the split seed itself, so it cannot be replayed as recorded; "
    "ask the agent to run it again"
)
_SEED_AXIS_REMEDY = (
    "at one draw a caller-supplied data.split.seed axis is sampled or gridded with the point, "
    "the best is Ray's own pick over seeds and points together (so best_params would name a "
    "seed), a pruning scheduler can end one seed's trial early, and the sweep computes no "
    "spread; drop data.split.seed from param_space and pass split_draws=<n> (with "
    "split_draw_seeds for chosen, distinct seeds), which pairs every seed with every point and "
    "picks the best by mean over draws. The paired path's own conditions apply, bound and "
    "unbound alike: a config bound to a selection redraws train and val inside the selection's "
    "own train and val samples (the selection must be readable and resolve at least two "
    "foreground groups across them); an unbound config keeps auto_val on; search_alg "
    "is one the native generator builds (random, grid, variant_generator, or unset); scheduler "
    "prunes nothing (none, fifo, or unset); split_draw_seeds is one per draw and distinct; no "
    "baseline_params names the seed under a warm start; no other data.* axis is in param_space; "
    "a trial_budget is stated on a launch that is not a relaunch, and Ray's variant count over "
    "the sweep fits under it; and, for a built-in detection config with tiling on, more than one "
    "trainable source is admitted under data.images_dir (a single admitted source's own "
    "single-source spatial-strip path pairs no distinct partition with any draw). A single fixed "
    "seed belongs in "
    "base_config's own data.split.seed: the drawn path's partition depends on it, and the "
    "single-source spatial path's, or a selection-bound config's without redraw_within_selection, "
    "never does (the spatial path still records the config's value on the split record, a "
    "different fact, not claimed here)."
)


def caller_split_seed_refusal(param_space: dict, split_draws: int) -> SeedAxisRefusal | None:
    """Whether ``param_space`` names ``data.split.seed`` as its own axis while ``split_draws``
    draws at most one partition: refused whatever the sampler.
    """
    from tcip_mcp.pipelines.training.hpo import SPLIT_DRAW_SEED_KEY

    if split_draws > 1 or SPLIT_DRAW_SEED_KEY not in param_space:
        return None
    return SeedAxisRefusal(reason=_SEED_AXIS_REASON, remedy=_SEED_AXIS_REMEDY)


def _first_sampled_point(param_space: dict) -> dict:
    """One deterministic point from ``param_space``, spanning its declared range or choices."""
    point: dict = {}
    for key, spec in param_space.items():
        if not isinstance(spec, dict):
            # Not this platform's own {"type": ...} shape (a caller-composed space bypassing
            # get_default_space): take a value outright rather than guess a range from it.
            point[key] = spec[0] if isinstance(spec, list) and spec else spec
            continue
        kind = spec.get("type")
        if kind == "categorical":
            choices = spec.get("choices") or [None]
            point[key] = choices[0]
        elif kind in ("loguniform", "uniform", "int"):
            point[key] = spec["low"] if "low" in spec else spec.get("high", 0)
        else:
            point[key] = spec.get("low", spec.get("choices", [None])[0])
    return point


def _preflight_points(param_space: dict) -> list[tuple[str, dict]]:
    """Every point a sweep's preflight checks: the first sampled corner,
    plus one variant per categorical choice and one per numeric bound, each holding every other
    axis at its first sampled value.
    """
    base = _first_sampled_point(param_space)
    points: list[tuple[str, dict]] = [("the first sampled point", dict(base))]
    for key, spec in param_space.items():
        if not isinstance(spec, dict):
            continue
        kind = spec.get("type")
        if kind == "categorical":
            for choice in spec.get("choices") or []:
                variant = dict(base)
                variant[key] = choice
                points.append((f"{key}={choice!r}", variant))
        elif kind in ("loguniform", "uniform", "int"):
            for bound in ("low", "high"):
                if bound in spec:
                    variant = dict(base)
                    variant[key] = spec[bound]
                    points.append((f"{key} {bound}={spec[bound]!r}", variant))
    return points


def _one_real_batch(task: str, train_ds: Any, n: int = 2):
    """``(batch, reason_it_failed)``: one collated ``(images, targets)`` of the first ``n`` items
    of the run's resolved train dataset, or ``(None, reason)`` when it yields none."""
    try:
        from tcip_mcp.pipelines.training.collation import task_collate

        assert isinstance(train_ds, Sized), "every build_dataset task backend defines __len__"
        indexable: Any = train_ds
        items = [indexable[i] for i in range(min(n, len(train_ds)))]
        if not items:
            return None, "the dataset built but is empty"
        return task_collate(task)(items), None
    except Exception as exc:  # noqa: BLE001, an unbuildable batch is a caller decision, not a crash
        logger.info("could not build a real batch to smoke task %r: %s", task, exc)
        return None, f"{type(exc).__name__}: {exc}"


@tool()
def evaluate_model(
    project: Path,
    experiment_id_or_ckpt: str,
    images_dir: str,
    labels_dir: str | None = None,
    stated: Stated | None = None,
    iou_threshold: float = 0.5,
    tiling: dict | None = None,
    use_tiled_inference: bool = False,
    trait: str | None = None,
) -> dict:
    """Evaluate a trained checkpoint on a (held-out) dataset and return the result.

    Computes the same per-task metrics as validation: detection/instance_seg get the one
    matcher's precision/recall/F1 and average precision (by mask for instance_seg);
    classification/ordinal/regression get the in-house scalar metrics. Writes
    nothing: the training run's directory is never touched. The reference is read under the class
    space the checkpoint records, its map included.

    Three detection eval regimes:
      * Untiled default (no ``tiling``, checkpoint trained without tiling) -> single full-res
      forward pass, ``eval_regime="full-frame-single-pass"``: the delivery gate for a checkpoint
      never tile-trained. ``use_tiled_inference`` for such a checkpoint refuses (see below).
      * ``tiling`` set (or a run id whose training was tiled, reused automatically) -> tile-level
      diagnostic that matches the training-run val mAP; not the delivery metric.
      * ``use_tiled_inference=True`` -> the delivery-grade full-frame metric for a tile-trained
      checkpoint (tiled inference reconstructed to full frame, matched to full-frame GT). Tile
      geometry is resolved from the checkpoint's own persisted or native-frame training geometry,
      or the stated one; a checkpoint with none of those refuses (see
      ``run_full_frame_evaluation``).

    Args:
        experiment_id_or_ckpt: A completed run's id (uses the checkpoint its final status
            names) or a checkpoint path. Either way the resolved checkpoint must be registered in
            this project's registry (``register_model``, explicit mode for a foreign or bespoke
            checkpoint) or this door refuses before loading it.
        images_dir: The capture directory of the evaluation split.
        labels_dir: The masks dir (semantic_seg) or the GT CSV path
            (classification/ordinal/regression, one row per image stem); omitted for label
            documents (detection/instance_seg), which the images' own keys address. Admitted once
            with ``images_dir`` (``label_queries.admit``) before any regime runs, a refusal naming
            both. The task is the checkpoint's own.
        stated: The execution values to state rather than derive (``execution.Stated``): the
            operating ``conf`` and detection cap ``max_dets`` P/R/F1 are reported at, and on the
            delivery-grade path the ``tile_size``, ``overlap``, ``postprocess`` and
            ``cross_tile_nms``. The resolved record, each value's source with it, is returned under
            ``execution``.
        iou_threshold: The IoU a match must reach under the IoU convention.
        tiling: Optional detection tiling dict ({enabled, tile_size, overlap, ...}) for a
            tile-level eval. None + a run id reuses the run's training tiling; None + a checkpoint
            path stays untiled.
        use_tiled_inference: Score the delivery regime (full-frame via tiled inference).
        trait: When set, the derived localization criterion of the trait's latest confirmed
            revision (traits.py, e.g. a count trait's center-match) governs the reported count
            and the selection f1; AP@0.5 (``iou_threshold``) is kept as a labeled comparability
            metric. Absent -> the IoU convention governs.
    """
    import torch
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.eval_runners import (
        run_full_frame_evaluation, run_test_evaluation,
    )
    from tcip_mcp.pipelines.data.datasets import build_dataset, resolve_sizes
    from tcip_mcp.pipelines.execution import prepare_pass

    from tcip_mcp.operationalization import OperationalizationRefused, latest_confirmed
    from tcip_mcp.traits import TraitUnknownError

    try:
        trait_entry = latest_confirmed(trait, project).entry if trait else None
    except (TraitUnknownError, OperationalizationRefused) as exc:
        return {"error": str(exc)}

    ckpt = experiment_id_or_ckpt
    by_run = not Path(ckpt).is_file()
    if by_run:
        observation = experiments.find_observation(experiment_id_or_ckpt, project=project)
        completed = observation.checkpoint if observation is not None else None
        if completed is None:
            return {"error": f"Not a checkpoint path or a completed run's id: "
                             f"{experiment_id_or_ckpt}"}
        ckpt = completed["path"]
    if not Path(ckpt).is_file():
        return {"error": f"Checkpoint not found: {ckpt}"}

    from tcip_mcp.model_registry import UnregisteredCheckpoint, load_registered_checkpoint

    from tcip_mcp.pipelines.data.selection import ClassScope

    try:
        checkpoint = load_registered_checkpoint(ckpt, project=project)
        task = checkpoint.task
        scope = ClassScope.of(checkpoint.data_config)
    except (UnregisteredCheckpoint, ValueError) as exc:
        return {"error": str(exc)}
    run_tiling = checkpoint.data_config.get("tiling")
    stated = stated or Stated()

    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_mcp.pipelines.data.label_queries import admit, require_admitted

    try:
        admitted = admit(images_dir, labels_dir, scope=scope)
        require_admitted(admitted)
    except (ValueError, UnreadableLabelDocument) as exc:
        return {"error": f"the ground truth of images_dir={images_dir!r} with "
                         f"labels_dir={labels_dir!r} admits nothing to evaluate: {exc}"}

    if use_tiled_inference and task == "detection":
        try:
            return run_full_frame_evaluation(
                checkpoint, admitted, stated=stated, iou_threshold=iou_threshold,
                trait=trait_entry)
        except (ValueError, UnreadableLabelDocument) as exc:
            return {"error": str(exc)}

    # Tile-level diagnostic (or untiled). Only detection tiles; a run id reuses its training tiling.
    if tiling is None and by_run:
        tiling = run_tiling
    if task != "detection":
        tiling = None

    # The checkpoint's own untiled pass: the width its predictor reads images at sizes the loader,
    # and its execution record governs the model that scores them.
    try:
        pass_ = prepare_pass(checkpoint, stated.model_copy(update={"tile": False}))
    except ValueError as exc:
        return {"error": str(exc)}
    predictor = pass_.predictor

    try:
        measured_samples = admitted.every_sample()
        # Read at the width the predictor reads at: the model scores these tensors, so a loader
        # sized off the references instead would hand it images of another shape.
        dataset = build_dataset(
            task, samples=measured_samples, tiling=tiling, scope=scope,
            sizes=resolve_sizes(task, {"num_channels": predictor.in_chans}, measured_samples))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Failed to build dataset: {exc}"}

    loader = DataLoader(dataset, batch_size=4, collate_fn=task_collate(task))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return run_test_evaluation(
            pass_, loader, device, iou_threshold=iou_threshold, tiling=tiling, trait=trait_entry)
    except ValueError as exc:
        return {"error": str(exc)}
