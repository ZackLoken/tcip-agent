"""Training MCP tools, config validation, launch training, HPO, status."""

from __future__ import annotations

import itertools
import logging
import os
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from tcip_mcp import experiments
from tcip_mcp.audit import audited, now_iso, record_event_or_raise
from tcip_mcp.experiments import SWEEP_FILE, SweepGroup, TrainingDetail
from tcip_mcp.pipelines.data.split_construction import (
    ResolvedRun,
    data_dir_issues,
    resolve_run,
    selection_compatibility,
)
from tcip_mcp.pipelines.execution import Stated
from tcip_mcp.pipelines.model_contract import LaunchContract
from tcip_mcp.pipelines.schemas import (
    BoundSplit,
    DrawnSplit,
    ImageStatsSampling,
    TilingSpec,
    checked_train_config,
)
from tcip_mcp.pipelines.training.hpo import Count
from tcip_mcp.registry_paths import located
from tcip_mcp.server import tool
from tcip_store import StoreError, canonical_path

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.datasets import BaseDataset
    from tcip_mcp.pipelines.model_build import StagedSources
    from tcip_mcp.pipelines.schemas import DataSpec, TrainConfigSchema
    from tcip_mcp.pipelines.training.collation import Batch
    from tcip_mcp.pipelines.training.hpo import Report

logger = logging.getLogger(__name__)

# Round-robins unpinned concurrent launches across available GPUs (no-op with 0-1 devices);
# next() on a count is one C call, atomic under the GIL.
_gpu_round_robin = itertools.count()

# Serializes the overfit diagnostic's reseed-run-restore, since the RNG streams are process-global.
_OVERFIT_CHECK_LOCK = threading.Lock()


def data_with_selection(
    data: DataSpec, selection_dir: str, project: Path
) -> DataSpec:
    """Return the data block ``data`` would launch with bound to the selection ``selection_dir``.

    ``selection_dir`` is located against ``project``. ``split`` is replaced wholesale by one
    naming only ``selection_dir``, and the stated ``scope`` and ``labels_dir`` dropped.
    """
    from tcip_mcp.pipelines import schemas

    kept = data.record(
        exclude={"scope": True, "labels_dir": True, "split": True}
    )
    return schemas.DataSpec.model_validate(
        {**kept, "split": {"selection_dir": selection_dir}},
        context={"project": project},
    )


# Lazy imports of heavy dependencies inside tool functions to keep server startup fast.


def preflight_config(
    project: Path, config: dict, smoke: bool = False, overfit: bool = False
) -> dict:
    """Validate a training configuration before launching (:func:`_preflight`'s report)."""
    return _preflight(
        project, *_admitted(config, {}, project), smoke=smoke, overfit=overfit
    )[0]


def _admitted(
    config: dict, point: dict, project: Path
) -> tuple[TrainConfigSchema | None, StagedSources | None, list[str]]:
    """Admit ``config`` with ``point`` applied, validated once against ``project``.

    ``point`` is applied by :func:`_apply_hpo_params`, the result validated once against
    ``project`` (``schemas.checked_train_config``) and, when it validates, its augmentation judged
    against the project record's capture invariances (``augmentations.forbidden_transforms``) and
    its structure (:func:`_structural_issues`). Returns the spec, ``None`` when it did not
    validate, its staged sources, ``None`` when none were taken, and every issue.
    """
    from tcip_mcp.pipelines.data.augmentations import forbidden_transforms
    from tcip_mcp.project_record import ProjectRecordMissingError, read_record

    try:
        spec, issues = checked_train_config(
            _apply_hpo_params(config, point), project
        )
    except ValueError as exc:
        return None, None, [str(exc)]
    if spec is None:
        return None, None, issues
    if spec.augmentation is not None:
        try:
            issues = forbidden_transforms(
                spec.augmentation, read_record(project).invariances
            )
        except (ValueError, ProjectRecordMissingError) as exc:
            issues = [str(exc)]
    sources, structural = _structural_issues(spec, project)
    return spec, sources, [*issues, *structural]


def _preflight(
    project: Path,
    spec: TrainConfigSchema | None,
    sources: StagedSources | None,
    issues: list[str],
    *,
    smoke: bool,
    overfit: bool,
) -> tuple[dict, ResolvedRun | None]:
    """Check an admitted training configuration of ``project``; resolve it when no issue remains.

    The configuration is admitted once by :func:`_admitted`: ``spec``, ``None`` when the schema
    refused, its staged ``sources``, every import below reading their layout, and its ``issues``.
    It is checked before launching a run of ``project`` and resolved once
    (:func:`~tcip_mcp.pipelines.data.split_construction.resolve_run`) when no issue stands.
    Returns the report and the resolution, ``None`` when an issue stood first.

    Config structure, one placement for everything::

        model_source: {builder, builder_kwargs, task, normalization,
                       detector_geometry?}
        data: {images_dir, labels_dir?, scope}  # known loaders, or a bespoke
                                                # {dataset_source: {builder, ...}}
        batch_size, stages, mixed_precision, device, seed, ...   # at the top level
        evaluation: {trait, selection_metric, ...}
        training_source: optional custom train(ctx) loop.

    A nested ``training`` section is refused by name (``schemas.TrainConfigSchema``).

    Args:
        smoke: When True, actually build the model and run ``check_model_contract`` (a train+eval
            forward at the run's resolved dims and frame, every attribute head included). A
            contract failure is appended to ``issues`` and blocks the launch. For a run that
            resolved no frame, or a task the contract has no synthetic batch schema for, one real
            batch of the resolved train dataset is used instead; if it yields none, that also
            blocks.
        overfit: When True (with ``smoke``), also run the voluntary ``overfit_check`` diagnostic
            and report it under ``overfit_check``, never gating. The stored report is already
            rendered (``model_contract.render_overfit_report``), with non-finite losses rendered.

    """
    from tcip_mcp.pipelines.data.label_queries import admits

    issues = list(issues)
    warnings: list[str] = []
    split = spec.data.split if spec is not None else None
    if isinstance(split, BoundSplit) and split.seed is not None:
        warnings.append(
            f"data.split.seed={split.seed} beside data.split.selection_dir: this run redraws "
            "train and val inside the selection's own train and val samples at this seed; the "
            "selection's calibration side stays untouched."
        )

    source = spec.model_source.normalization.source if spec else None
    sampling_record = (
        source if isinstance(source, ImageStatsSampling) else None
    )

    # The run resolved once, the way its launch records it: every leg below reads this.
    resolution: ResolvedRun | None = None
    if not issues:
        counts: dict[str, int] = {}
        try:
            assert spec is not None and sources is not None, (
                "a config with no issue was staged"
            )
            resolution = resolve_run(
                spec, sources.layout, project=project, tallies_out=counts
            )
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
                f"{total - n_dropped} stem(s) admitted."
            )

    result: dict = {"valid": False, "issues": issues, "warnings": warnings}
    if sampling_record is not None and resolution is None:
        result["image_stats_containment"] = "not_checked"
    if resolution is not None:
        known = {str(s.source) for s in resolution.partition.samples}
        if sampling_record is not None:
            bad = sorted(
                {
                    label
                    for label, _ in sampling_record.windows
                    if label not in known
                }
            )
            result["image_stats_containment"] = "checked"
            if bad:
                issues.append(
                    f"model_source.normalization.source names path(s) {bad} outside this "
                    f"run's own {len(known)} source(s)."
                )

    # Smoke: build the model and run the correctness contract at the resolved dims, so a broken
    # bespoke builder is caught here (before the training subprocess spawns) rather than surfacing
    # only as run.status='failed'. Only attempt once the structural checks pass, otherwise the
    # config can't build and the contract would just re-report the same failure. Overfit stays a
    # voluntary, non-gating diagnostic (a valid model can fail 20 steps on noise).
    if smoke and not issues:
        assert resolution is not None and sources is not None, (
            "a config with no issue resolved"
        )
        try:
            from tcip_mcp.pipelines.model_build import (
                build_from_model_source,
                recorded_model_dims,
                resolve_contract_dims,
            )
            from tcip_mcp.pipelines.model_contract import (
                check_model_contract,
                overfit_check,
                render_overfit_report,
            )

            resolved_spec = resolution.spec.model_copy(
                update={"data": resolution.data}
            )
            task = resolved_spec.model_source.task
            built_at = recorded_model_dims(resolved_spec)
            dims = resolve_contract_dims(
                resolved_spec, built_at, resolution.geometry
            )
            model = build_from_model_source(
                resolved_spec.model_source,
                sources.layout,
                built_at,
                resolution.geometry,
            )
            report = check_model_contract(model, task, dims=dims)
            batch: Batch | None = None
            why_no_batch: str | None = None
            if report.get("not_smokeable"):
                # No synthetic batch for this task or this width: smoke against a real batch from
                # the run's own dataset, the only reference the platform has not guessed at.
                real = _one_real_batch(task, resolution.train_ds)
                if isinstance(real, str):
                    why_no_batch = real
                else:
                    batch = real
                    report = check_model_contract(
                        model, task, sample_batch=batch
                    )
            # ``dims`` shape the synthetic batch only, so they describe nothing once a real batch
            # is used, record which reference actually proved the contract.
            result["smoke"] = {
                **report,
                "task": task,
                "batch_source": "dataset"
                if batch is not None
                else "synthetic",
                "dims": None if batch is not None else dims,
            }
            if report.get("not_smokeable"):
                issues.append(
                    f"model contract: {report['not_smokeable']} Building one from the run's data "
                    f"config failed too ({why_no_batch}), so the measurement boundary is unproven."
                )
            elif not report["ok"]:
                issues.extend(
                    f"model contract: {msg}" for msg in report["issues"]
                )
            if overfit:
                from tcip_mcp.pipelines.training.generic_trainer import (
                    capture_rng_state,
                    restore_rng_state,
                )

                # Same batch the contract used, otherwise this re-synthesizes and reports a false
                # "does not learn" for exactly the bespoke tasks the real-batch path exists for.
                with _OVERFIT_CHECK_LOCK:
                    rng_state = capture_rng_state(None)
                    try:
                        raw_report = overfit_check(
                            model,
                            task,
                            seed=resolved_spec.seed,
                            deterministic=resolved_spec.deterministic,
                            sample_batch=batch,
                            dims=dims,
                        )
                    except Exception as exc:  # noqa: BLE001, becomes the report's issue only
                        raw_report = {
                            "passed": False,
                            "losses": [],
                            "initial": None,
                            "final": None,
                            "issue": f"overfit check failed: {exc}",
                        }
                    finally:
                        restore_rng_state(rng_state, None)
                # Rendered before storage: this tool answers over JSON-RPC, which a raw non-finite
                # loss cannot cross, so a diverging model's report is sanitized here, once.
                result["overfit_check"] = render_overfit_report(raw_report)
        except Exception as exc:  # noqa: BLE001, a build/contract crash is itself a blocking issue
            issues.append(f"model smoke build failed: {exc}")

    result["valid"] = len(issues) == 0
    return result, resolution


def _structural_issues(
    spec: TrainConfigSchema, project: Path
) -> tuple[StagedSources | None, list[str]]:
    """Stage the validated config ``spec``'s sources and list what stops it before any data read.

    Returns the staged sources of ``spec`` in ``project`` (``model_build.staged_sources``,
    ``None`` when they refuse) and what stops it before anything reads its data: what training
    reads left unstated (``TrainConfigSchema.trainer_reads``'s refusal), declared source files a
    run of ``project`` cannot place or read (the snapshot's refusal, before anything imports), a
    ``model_source`` or ``data.dataset_source`` builder or a ``training_source`` that does not
    import from their layout, and a ``data`` block that names no locations
    (``split_construction.data_dir_issues``).
    """
    from tcip_mcp.pipelines.model_build import (
        import_source_builder,
        source_seams,
        staged_sources,
    )

    issues: list[str] = []
    try:
        spec.trainer_reads()
    except ValueError as exc:
        issues.append(str(exc))
    try:
        sources = staged_sources(spec, project)
    except (ValueError, OSError) as exc:
        return None, [
            *issues,
            f"source_files: {exc}",
            *data_dir_issues(spec.data),
        ]
    for field, dotted, _files in source_seams(spec):
        try:
            import_source_builder(dotted, sources.layout)
        except Exception as exc:
            issues.append(f"{field} not importable: {exc}")
    issues.extend(data_dir_issues(spec.data))
    return sources, issues


def open_run(
    run_dir: Path,
    spec: TrainConfigSchema,
    sources: StagedSources,
    resolved: experiments.ResolvedInput | None,
    *,
    relaunched_from: str | None = None,
    resume_from: str | None = None,
    max_wall_clock_seconds: float | None = None,
    model_contract: LaunchContract | None = None,
    trial_params: dict | None = None,
) -> None:
    """Open the run directory ``run_dir`` with its ``run.json``.

    The directory is opened by ``experiments.open_run_directory``, its ``run.json`` holding the
    validated launch config ``spec``, recorded as it dumps (``TrainConfigSchema.record``), as the
    run's input, ``resolved`` as what that input resolved to (``split_construction.ResolvedRun``'s
    record, ``None`` for an HPO trial refused at its admission or whose resolution failed), the
    environment and the dataset identity of ``spec``'s data block
    (``split_construction.dataset_identity``), the run's source snapshot its admission took
    (``sources``, ``model_build.snapshot_model_source``'s record and copies, the copies laid out
    in the directory), the run it was relaunched from and the checkpoint it resumes from, its wall
    clock, the model contract preflight proved, and an HPO trial's sampled point. Refuses an
    existing directory (``experiments.RunDirectoryExistsError``).
    """
    from tcip_mcp.pipelines.data.split_construction import dataset_identity
    from tcip_mcp.pipelines.model_build import capture_env

    dataset_id, fingerprint = dataset_identity(
        spec.data, resolved.partition.samples if resolved else ()
    )
    experiments.open_run_directory(
        run_dir,
        experiments.RunRecord(
            created=now_iso(),
            config=spec.record(),
            resolved=resolved,
            environment=capture_env(),
            dataset=experiments.DatasetIdentity(
                id=dataset_id, fingerprint=fingerprint
            ),
            source=sources.record,
            relaunched_from=relaunched_from,
            resume_from=resume_from,
            max_wall_clock_seconds=max_wall_clock_seconds,
            model_contract=model_contract,
            trial_params=trial_params,
        ),
        sources.copies,
    )


@tool()
def launch_training(
    project: Path,
    config: dict,
    resume_from: str = "",
    max_wall_clock_seconds: float | None = None,
    overfit_check: bool = False,
    relaunched_from: str | None = None,
    *,
    actor: str | None,
) -> dict:
    """Launch a training run by ``actor`` in an isolated subprocess.

    The run builds its model from a bespoke ``model_source`` builder.

    The run's training body (dataset build, model forward/backward, checkpointing) executes in a
    separate OS process, writing into the run's own directory
    (``<project>/.tcip/experiments/<experiment_id>/``). Use monitor_training to monitor progress
    and cancel_training to stop a run. The platform itself stops a run when it is dead (its
    ``early_stopping.nonfinite_passes`` consecutive full training passes with no finite batch
    loss) or when its stages end it: its
    last stage ended (on its ``early_stopping`` plateau, or at a horizon-bound schedule's
    horizon), or a stage whose best did not improve by ``min_delta`` on the stage before it, on
    the run's selection value (the training loss for a run with no validation loader).

    Writes the run's ``run.json`` once, then the act's one audit line naming the minted
    ``experiment_id`` (``AuditEntryNotWrittenError`` when it cannot be appended, and no subprocess
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
    if "experiment_id" in config:
        return {
            "error": "launch_training: config.experiment_id is not a setting; the platform "
            "mints every run's id and answers it as experiment_id."
        }
    resume = str(located(resume_from, project)) if resume_from else None
    # smoke=True: build the model and run the correctness contract before spawning the training
    # subprocess, so a broken builder returns here instead of wasting a full audited run.
    spec, sources, issues = _admitted(config, {}, project)
    validation, resolution = _preflight(
        project, spec, sources, issues, smoke=True, overfit=overfit_check
    )
    if not validation["valid"]:
        return {"error": "Invalid config", "issues": validation["issues"]}
    assert resolution is not None and sources is not None, (
        "a valid config resolved"
    )

    # The top-level key, never the smoke sub-report: overfit_check runs beside the contract's
    # build, on the same batch; preflight_config already rendered it for storage.
    rendered_overfit_report = validation.get("overfit_check")

    smoke_report = validation.get("smoke") or {}
    model_contract_record = LaunchContract(
        subject="the model as built at launch, before any training step",
        gating=True,
        batch_source=smoke_report.get("batch_source"),
        dims=smoke_report.get("dims"),
        issues=smoke_report.get("issues", []),
        gradient_magnitudes=smoke_report.get("gradient_magnitudes"),
        overfit_check=rendered_overfit_report,
    )

    experiment_id = experiments.mint_experiment_id()
    try:
        run_dir = experiments.experiment_dir(experiment_id, project=project)
        open_run(
            run_dir,
            resolution.spec,
            sources,
            resolution.record,
            relaunched_from=relaunched_from,
            resume_from=resume,
            max_wall_clock_seconds=max_wall_clock_seconds,
            model_contract=model_contract_record,
        )
    except (StoreError, ValueError, OSError) as exc:
        return {"error": f"launch_training: {exc}"}
    record_event_or_raise(
        "launch_training",
        {
            "experiment_id": experiment_id,
            "relaunched_from": relaunched_from,
            "resume_from": resume,
        },
        actor=actor,
        scope=project,
    )

    proc = _start_worker(run_dir, _child_env_for_launch(resolution.spec))

    if max_wall_clock_seconds is not None:
        _watch_wall_clock(proc, max_wall_clock_seconds)

    # Keyed by its own log directory (the manager's default), never by the run id.
    tb_info = {}
    try:
        from tcip_mcp.pipelines.training.tensorboard_manager import (
            launch_tensorboard,
        )

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
    """Return this process's environment with its import search path set as ``PYTHONPATH``."""
    from tcip_mcp.pipelines.model_build import child_pythonpath

    return {**os.environ, "PYTHONPATH": child_pythonpath()}


def _child_env_for_launch(spec: TrainConfigSchema) -> dict[str, str]:
    """Return the subprocess environment for a run's launch (:func:`_worker_env`).

    Adds round-robin GPU pinning (``CUDA_VISIBLE_DEVICES``) when the validated config ``spec``
    names no device and more than one GPU is visible; untouched otherwise.
    """
    env = _worker_env()
    if spec.device:
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
    """Start ``subprocess_worker`` over the opened run or sweep ``directory`` in its own process.

    The process runs under ``env``.
    """
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tcip_mcp.pipelines.training.subprocess_worker",
            "--run-dir",
            str(directory),
        ],
        env=env,
    )


def _watch_wall_clock(proc: subprocess.Popen, timeout_seconds: float) -> None:
    """Start a daemon watcher that terminates ``proc`` once it overruns ``timeout_seconds``.

    The watcher terminates ``proc`` if it is still alive one heartbeat window
    (``experiments.HEARTBEAT_STALE_SECONDS``) past ``timeout_seconds``.
    """

    def _watch() -> None:
        try:
            proc.wait(
                timeout=timeout_seconds + experiments.HEARTBEAT_STALE_SECONDS
            )
        except subprocess.TimeoutExpired:
            proc.terminate()

    threading.Thread(target=_watch, daemon=True).start()


def training_detail(
    project: Path, experiment_id: str
) -> TrainingDetail | None:
    """Return the run, trial or sweep ``experiment_id`` names as its ``TrainingDetail``, or None.

    The directory is :func:`experiments.named_directory`'s in ``project``, its ``tensorboard_url``
    the one a TensorBoard this process started serves its board at (``experiments.board_of``).
    """
    from tcip_mcp.pipelines.training.tensorboard_manager import running_url

    directory = experiments.named_directory(experiment_id, project=project)
    if directory is None:
        return None
    url = running_url(str(experiments.board_of(directory)))
    if (directory / SWEEP_FILE).is_file():
        return TrainingDetail(
            run=None,
            tensorboard_url=url,
            sweep=experiments.read_sweep(
                directory, experiments.run_rows(project)
            ),
        )
    run = experiments.observe_run(directory)
    return TrainingDetail(
        sweep=None,
        tensorboard_url=url,
        run=experiments.run_summary(
            run,
            experiments.read_rows(run.metrics_log)[0],
            experiments.launch_declarations(project).get(run.directory.name),
        ),
    )


@tool()
def monitor_training(project: Path, experiment_id: str) -> dict:
    """Check a training run, a sweep's trial included, or a hyperparameter sweep, by its id.

    Answers :func:`training_detail`'s ``{"run", "sweep", "tensorboard_url"}``. An id naming
    neither answers ``{"error": ...}``.

    Args:
        experiment_id: The run's id from launch_training, a trial's id from its sweep, or a
            sweep's id from run_hyperparameter_search.

    """
    detail = training_detail(project, experiment_id)
    return (
        {"error": f"Run not found: {experiment_id}"}
        if detail is None
        else detail.model_dump()
    )


def list_split_choices(project: Path, experiment_id: str) -> dict:
    """List every data choice a relaunch of ``experiment_id`` can take.

    The choices are the data section its launch stated (``run.json``'s config), and every
    selection directory this project's own bound runs or the dataset's own ``splits`` directory
    hold, each compatibility-checked the way a launch would check it.

    Every check is :func:`selection_compatibility` over a selection this reader read itself,
    through :func:`~tcip_mcp.pipelines.data.selection.read_selection_checked`. A candidate
    selection is checked against the data block :func:`data_with_selection` builds
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
    own = next(
        (obs for obs in runs if obs.directory.name == experiment_id), None
    )
    if own is None:
        return {"error": f"Experiment not found: {experiment_id}"}
    data = own.record.spec.data
    split = data.split
    own_selection_dir = (
        split.selection_dir if isinstance(split, BoundSplit) else None
    )
    replaced_split_keys = sorted(
        set(split.model_dump(exclude_unset=True, exclude_none=True))
        - {"selection_dir"}
    )

    own_selection = None
    if isinstance(split, BoundSplit):
        as_recorded = {
            "case": "bound",
            "line": "on the partition it bound",
            "compatible": True,
            "reason": None,
        }
        own_selection, own_error = read_selection_checked(
            split.selection_dir, project=project
        )
        if own_selection is None:
            as_recorded["compatible"] = False
            as_recorded["reason"] = own_error or (
                f"no selection recorded under {split.selection_dir}; run draw_splits first."
            )
        else:
            own_issues = selection_compatibility(data, split, own_selection)
            if own_issues:
                as_recorded["compatible"] = False
                as_recorded["reason"] = "; ".join(own_issues)
    elif own.record.resolved is None:
        as_recorded = {
            "case": "drawn",
            "line": "resolved to nothing at its launch",
            "compatible": False,
            "reason": own.status_error,
        }
    else:
        seed = own.record.resolved.partition.seed
        as_recorded = {
            "case": "drawn",
            "line": f"draws its split again with seed {seed} over the labels as they are now",
            "compatible": True,
            "reason": None,
        }

    dir_issues = data_dir_issues(data)
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
        root_anchor = str(data.images_dir)
    else:
        return {"as_recorded": as_recorded, "selections": []}

    own_norm = canonical_path(own_selection_dir) if own_selection_dir else None
    candidate_dirs: list[str] = []
    seen: set[str] = set()
    for other in runs:
        binding = (
            other.record.resolved.partition.selection
            if other.record.resolved
            else None
        )
        if other is own or binding is None:
            continue
        candidate = binding.selection_dir
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
        selection, error_text = read_selection_checked(
            candidate_dir, project=project
        )
        if selection is None:
            if error_text is None:
                continue  # nothing recorded there; not a real candidate
            selections.append(
                {
                    "selection_dir": candidate_dir,
                    "enabled": False,
                    "reason": error_text,
                    "seed": None,
                    "group_by": None,
                    "train": 0,
                    "val": 0,
                    "calibration": 0,
                    "replaced_split_keys": replaced_split_keys,
                }
            )
            continue
        chosen = data_with_selection(data, candidate_dir, project)
        issues = selection_compatibility(
            chosen, cast(BoundSplit, chosen.split), selection
        )
        counts = selection.counts()
        entry: dict = {
            "selection_dir": candidate_dir,
            "seed": selection.seed,
            "group_by": selection.group_by,
            "train": counts["train"],
            "val": counts["val"],
            "calibration": counts["calibration"],
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
def cancel_training(
    project: Path, experiment_id: str, *, actor: str | None
) -> dict:
    """Request graceful cancellation of a running training run, a sweep's trial or a whole sweep.

    The request is ``actor``'s, the run or sweep named by its id (``experiments.named_directory``).

    A run stops at its next batch/epoch boundary, still saves ``model_final.pt``, the weights it
    stopped at (with no metrics once a batch ran past the epoch row those weights were taken
    at, since no row describes them), and writes its final status 'canceled'; a run whose
    divergence verdict lands first (its ``early_stopping.nonfinite_passes`` consecutive full
    training passes with no finite batch loss, checked ahead of cancellation at the same
    boundary) ends 'failed' instead, with no
    ``model_final.pt``.
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
    return {
        "experiment_id": experiment_id,
        "cancel_requested": True,
        "state": experiments.observe(directory).state,
    }


def inspect_compute_resources(project: Path) -> dict:
    """Report the host's current compute headroom and ``project``'s active training runs.

    The report enforces no cap.

    Returns:
        ``cpu``: ``{logical_count, percent_used}``, ``percent_used`` is ``None`` without ``psutil``
            installed.
        ``memory``: ``{total_bytes, available_bytes}``, both ``None`` without ``psutil``.
        ``gpus``: ``[{index, free_bytes, total_bytes}, ...]``, one entry per device whose query
            succeeds when CUDA is available (``torch.cuda.mem_get_info``, no extra dependency);
            ``[]`` without CUDA, and empty or partial when a device or memory query fails.
        ``active_training_runs``: count of every training run whose state is ``"running"``
            (``experiments.run_observations``).

    """
    cpu: dict[str, Any] = {
        "logical_count": os.cpu_count(),
        "percent_used": None,
    }
    memory: dict[str, Any] = {"total_bytes": None, "available_bytes": None}
    try:
        import psutil

        cpu["percent_used"] = psutil.cpu_percent(interval=0.1)
        vm = psutil.virtual_memory()
        memory["total_bytes"] = vm.total
        memory["available_bytes"] = vm.available
    except Exception:
        logger.info(
            "psutil unavailable or failed; cpu/memory visibility degraded to None",
            exc_info=True,
        )

    gpus: list[dict[str, Any]] = []
    try:
        import torch

        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                free_b, total_b = torch.cuda.mem_get_info(idx)
                gpus.append(
                    {
                        "index": idx,
                        "free_bytes": free_b,
                        "total_bytes": total_b,
                    }
                )
    except Exception:
        logger.info("GPU visibility unavailable", exc_info=True)

    active = sum(
        1
        for obs in experiments.run_observations(project)
        if obs.state == "running"
    )

    return {
        "cpu": cpu,
        "memory": memory,
        "gpus": gpus,
        "active_training_runs": active,
    }


_CANCEL_BEFORE_START_REASON = "canceled before the sweep's first trial started"
_CANCEL_DURING_RUN_REASON = (
    "the sweep was canceled by request before it could finish"
)


def open_trial(sweep: Path, trial_id: str, point: dict) -> Path:
    """Open the trial ``trial_id`` of the sweep at ``sweep`` as a run directory and return it.

    The directory lies beneath the sweep, named ``<sweep id>_<trial_id>``. The sweep's recorded
    base config with ``point`` applied is admitted (:func:`_admitted`) and, when it stands,
    resolved against its project through ``split_construction.resolve_run`` at the sweep's own
    trait revision, and the directory opened (:func:`open_run`) with ``point`` as its
    ``trial_params``: an admission issue or a resolution that fails is the trial's final status
    ``failed`` naming why. A config the schema
    refuses, or whose sources cannot be staged, raises ``ValueError`` naming every issue before
    any directory is opened.
    """
    observed = experiments.observe_sweep(sweep).record
    project = experiments.project_of_run(sweep)
    spec, sources, issues = _admitted(
        observed.input.base_config, point, project
    )
    if spec is None or sources is None:
        raise ValueError(
            f"trial {trial_id}'s config is refused: {'; '.join(issues)}"
        )
    trial_dir = sweep / f"{sweep.name}_{trial_id}"
    resolved, status_error = None, "; ".join(issues) or None
    if status_error is None:
        try:
            resolved = resolve_run(
                spec,
                sources.layout,
                project=project,
                trait_revision=observed.trait_revision,
            ).record
        except Exception as exc:  # noqa: BLE001, whatever stops the resolution fails the trial
            status_error = str(exc)
    open_run(trial_dir, spec, sources, resolved, trial_params=point)
    if status_error is not None:
        experiments.write_final_status(
            trial_dir, "failed", status_error, checkpoint=None
        )
    return trial_dir


def _run_hpo_trial(
    point: dict,
    report: Report,
    sweep: Path,
    trial_id: str,
) -> None:
    """Train one HPO trial of the sweep at ``sweep`` (:func:`open_trial`).

    Reports every ``selection`` its metrics log records, which is the trial's result. A trial
    whose point was refused at its admission or failed to resolve ends at its opening; otherwise
    its body runs through ``subprocess_worker.run_directory``. A sweep canceled before the trial
    started opens nothing.
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
    param_space: dict,
    baseline_params: dict | None = None,
    max_concurrent: int = 1,
    resources_per_trial: dict | None = None,
    auto_tensorboard: bool = True,
    split_draws: Count = 1,
    split_draw_seeds: list[int] | None = None,
    *,
    n_trials: Count,
    search_alg: str,
    scheduler: dict,
    search_seed: int,
    trial_budget: Count | None = None,
    relaunched_from: str | None = None,
) -> dict:
    """Run hyperparameter optimization on Ray Tune, training each trial for real.

    The search algorithm, constructed with ``search_seed``, and the trial scheduler are the
    caller's choice; the ``hpo`` module's ``available_search_algs`` and ``available_schedulers``
    list them.

    Trials optimize the ``evaluation.selection_metric`` ``base_config`` states, in the direction
    that metric's declaration says is better (``schemas.selection_of``), under the trait
    revision the base config resolves to once, recorded in the sweep's record and in every
    trial's launch record; a ``param_space`` axis sampling another metric is refused. Each trial
    trains under the base config's regime and reports the ``selection`` rows its metrics log
    records.

    Everything one sweep writes lands in its own directory, ``.tcip/experiments/<sweep_id>/``
    beside the project's runs, whose names it shares (:func:`open_sweep`, :func:`run_sweep`):
    its ``sweep.json`` written once before the first
    trial starts, carrying every argument this call resolved (``base_config``, the resolved
    ``param_space`` and the trait revision included); a heartbeat this call keeps while it runs; one
    run directory per trial, ``<sweep_id>_<ray trial id>``, a run like any other; Ray's own
    experiment store (also the TensorBoard logdir); and its final status, written once when the
    sweep ends. An existing directory refuses by name. The result, as ``monitor_training``'s at any
    time, is a ``TrainingDetail`` whose ``sweep`` is the sweep's group: its trial rows and what
    they amount to (``experiments.sweep_outcome``).

    Refuses (``{"error": ..., "issues": [...]}``, nothing minted): a ``param_space`` that is
        not a search space (``hpo.space_axes``, a grid's int axis past its bound included); a
        ``base_config`` that fails preflight, an unimportable builder or training source, or a
        config with no ``data`` section, at any point :func:`_preflight_points` checks (the
        first corner and each single-axis choice and range end, each point once; every trial's
        own point is admitted when its trial opens); a
        ``param_space`` axis naming ``data.split.seed`` while ``split_draws`` draws at
        most one partition (:func:`caller_split_seed_refusal`); ``split_draws`` above 1 on an
        unbound, built-in config with tiling on that admits exactly one trainable source
        (:func:`_spatial_draws_issue`); and, on a launch above one draw or any call naming a
        ``trial_budget``, the bound refusals (:func:`_trial_budget_refusal`). ``split_draws``,
        ``n_trials`` and a stated ``trial_budget`` are each a
        :data:`~tcip_mcp.pipelines.training.hpo.Count`, refused by the type at the MCP boundary
        otherwise. A
        cancel (``cancel_training``) requested before or during the run ends the sweep
        ``{"state": "canceled", ...}``, its final status recording the same.

    Args:
        base_config: Base training config each trial modifies; it may leave unstated a value
            ``param_space`` names, and every other value a training config requires it states.
        param_space: Param-space dict (see ``hpo.space_axes``), each key a config key or a
            dotted path into one (``optimizer.head_lr``). A ``data.split.seed`` axis is refused
            whenever ``split_draws`` draws at most one partition; above 1 it belongs to
            ``split_draws``/``split_draw_seeds``.
        n_trials: Number of sampled points.
        search_alg: Search algorithm, one ``available_search_algs`` lists.
        scheduler: Trial scheduler, a block naming one ``available_schedulers`` lists as its
            ``name`` beside every setting that scheduler's Ray class takes
            (``hpo.scheduler_settings``), its objective and ``pbt``'s mutations excepted. Refused
            naming each setting left unstated or not taken.
        baseline_params: Hyperparameter values to seed the search with, ``None`` for none.
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
            seed. Over a ``base_config`` bound to a selection each trial's seed redraws train
            and val inside the selection's own train-plus-val samples, calibration untouched;
            refused when those
            samples resolve to fewer than two foreground groups. Otherwise refused when
            ``data.split.val_ratio`` is zero, ``search_alg`` is not a native one, ``scheduler``
            prunes,
            ``split_draw_seeds`` is given at a length other than ``split_draws`` or names the same
            seed twice, ``baseline_params`` names ``data.split.seed``,
            ``param_space`` already sweeps ``data.split.seed`` itself, or ``param_space`` sweeps
            any other ``data.*`` axis. The outcome groups trials by point (params minus the seed)
            and chooses the best by mean over each point's draws; see ``best_value_spread``, and
            every point's own block at ``split_sensitivity``. 1 is the default.
        split_draw_seeds: The seeds ``split_draws`` pairs with every sampled point, one per draw;
            omit for the derived default (see ``split_draws``).

    """
    opened = open_sweep(
        project,
        base_config,
        param_space,
        n_trials=n_trials,
        search_alg=search_alg,
        scheduler=scheduler,
        baseline_params=baseline_params,
        max_concurrent=max_concurrent,
        resources_per_trial=resources_per_trial,
        split_draws=split_draws,
        split_draw_seeds=split_draw_seeds,
        search_seed=search_seed,
        trial_budget=trial_budget,
        relaunched_from=relaunched_from,
        actor=None,
    )
    if isinstance(opened, dict):
        return opened
    group = run_sweep(opened)
    url = None
    if auto_tensorboard and group.state == "completed":
        from tcip_mcp.pipelines.training.tensorboard_manager import (
            launch_tensorboard,
        )

        url = launch_tensorboard(str(experiments.board_of(opened))).get("url")
    return TrainingDetail(
        run=None, sweep=group, tensorboard_url=url
    ).model_dump()


def open_sweep(
    project: Path,
    base_config: dict,
    param_space: dict,
    *,
    n_trials: Count,
    search_alg: str,
    scheduler: dict,
    baseline_params: dict | None,
    max_concurrent: int,
    resources_per_trial: dict | None,
    split_draws: Count,
    split_draw_seeds: list[int] | None,
    search_seed: int,
    trial_budget: Count | None,
    relaunched_from: str | None,
    actor: str | None,
) -> Path | dict:
    """Check a sweep's arguments and open its directory under ``project``.

    The checks: its scheduler block (``hpo.scheduler_settings``), the first point
    :func:`_preflight_points` lists applied to ``base_config`` and admitted once
    (:func:`_admitted`), its data block, which every split-draw check reads, a ``param_space``
    axis sampling an ``evaluation.selection_metric`` other than the base config's refused, then
    resolved once (:func:`_preflight`), whose partition answers whether its draws can vary
    (:func:`_spatial_draws_issue`), then each other point it lists admitted the same way.

    Creates the sweep's directory under ``project``, named by
    ``experiments.mint_experiment_id("hpo")``, with its ``sweep.json`` written once, carrying the
    trait revision and, as its ``input``, every argument: ``base_config`` as the caller stated it,
    never a sampled point's values, its paths stored against ``project``
    (``experiments.RECORD_PATHS``); then the act's one audit line by ``actor`` naming the sweep
    (``AuditEntryNotWrittenError`` when it cannot be appended). Returns the opened sweep's
    directory, or the refusal ``{"error", "issues"}`` with nothing created.
    """
    from tcip_mcp.pipelines.training.hpo import (
        resolved_draw_seeds,
        scheduler_settings,
        space_axes,
        split_draw_search_space,
    )

    if (
        relaunched_from is not None
        and experiments.find_sweep(relaunched_from, project=project) is None
    ):
        return {
            "error": f"relaunched_from names no sweep of this project: {relaunched_from!r}",
            "issues": [],
        }

    try:
        scheduler_settings(scheduler)
    except ValueError as exc:
        return {"error": str(exc), "issues": []}

    try:
        (first_label, first_point), *rest = _preflight_points(
            param_space, search_alg
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "error": f"param_space is not a search space: {exc!r}",
            "issues": [],
        }
    first_spec, first_sources, first_issues = _admitted(
        base_config, first_point, project
    )
    if first_spec is None:
        return {
            "error": f"the sweep's base config fails preflight at {first_label}",
            "issues": first_issues,
        }

    # Checked ahead of preflight, so its own reason is what a zero validation share's refusal
    # reads as, not whatever preflight would have hit first.
    draws_refusal = _split_draws_refusal(
        first_spec.data,
        param_space,
        search_alg,
        scheduler,
        split_draws,
        split_draw_seeds,
        baseline_params,
    )
    if draws_refusal is not None:
        return {"error": draws_refusal, "issues": []}

    metric = (base_config.get("evaluation") or {}).get("selection_metric")
    sampled = {
        value.get("selection_metric") if isinstance(value, dict) else value
        for name, (_kind, values) in space_axes(
            param_space, grid=False
        ).items()
        if name in ("evaluation", "evaluation.selection_metric")
        for value in values
    }
    if sampled - {metric}:
        return {
            "error": "a sweep ranks every trial by the one evaluation.selection_metric its "
            f"base_config states ({metric!r}); state it there and sample no other.",
            "issues": [],
        }

    seed_axis_refusal = caller_split_seed_refusal(param_space, split_draws)
    if seed_axis_refusal is not None:
        return {
            "error": f"{seed_axis_refusal.reason} {seed_axis_refusal.remedy}",
            "issues": [],
        }

    preflight, resolution = _preflight(
        project,
        first_spec,
        first_sources,
        first_issues,
        smoke=False,
        overfit=False,
    )
    if resolution is None or not preflight["valid"]:
        return {
            "error": f"the sweep's base config fails preflight at {first_label}",
            "issues": preflight["issues"],
        }
    for label, point in rest:
        issues = _admitted(base_config, point, project)[2]
        if issues:
            return {
                "error": f"the sweep's base config fails preflight at {label}",
                "issues": issues,
            }
    spatial_refusal = _spatial_draws_issue(resolution, split_draws)
    if spatial_refusal is not None:
        return {"error": spatial_refusal, "issues": []}

    draw_seeds = resolved_draw_seeds(
        resolution.spec.data.split, split_draws, split_draw_seeds
    )
    search_param_space = split_draw_search_space(param_space, draw_seeds)

    budget_refusal = _trial_budget_refusal(
        split_draws=split_draws,
        n_trials=n_trials,
        search_alg=search_alg,
        trial_budget=trial_budget,
        relaunched_from=relaunched_from,
        search_param_space=search_param_space,
        baseline_params=baseline_params,
    )
    if budget_refusal is not None:
        return {"error": budget_refusal, "issues": []}

    try:
        record = experiments.SweepRecord(
            created=now_iso(),
            trait_revision=resolution.trait_revision,
            input=experiments.SweepInput(
                n_trials=n_trials,
                search_alg=search_alg,
                scheduler=scheduler,
                max_concurrent=max_concurrent,
                baseline_params=baseline_params,
                resources_per_trial=resources_per_trial,
                param_space=param_space,
                base_config=base_config,
                relaunched_from=relaunched_from,
                split_draw_seeds=draw_seeds,
                search_seed=search_seed,
                trial_budget=trial_budget,
            ),
        )
        directory = experiments.create_run_directory(
            experiments.experiment_dir(
                experiments.mint_experiment_id("hpo"), project=project
            )
        )
    except (StoreError, ValueError, OSError) as exc:
        return {"error": str(exc), "issues": []}
    experiments.write_record(directory / SWEEP_FILE, record)
    record_event_or_raise(
        "open_sweep",
        {"sweep_id": directory.name, "relaunched_from": relaunched_from},
        actor=actor,
        scope=project,
    )
    return directory


def launch_sweep(project: Path, source: Path, *, actor: str | None) -> dict:
    """Relaunch the sweep directory ``source`` as a new sweep of ``project`` by ``actor``.

    The new sweep is opened by :func:`open_sweep`, which mints its id, from the recorded input of
    ``source``, relaunched from it, and run (:func:`run_sweep`) in its own worker process.
    Returns ``{"sweep_id", "status": "launched"}``, or the refusal dict when that input no longer
    opens.
    """
    given = experiments.observe_sweep(source).record.input
    opened = open_sweep(
        project,
        given.base_config,
        given.param_space,
        n_trials=given.n_trials,
        search_alg=given.search_alg,
        scheduler=given.scheduler,
        baseline_params=given.baseline_params,
        max_concurrent=given.max_concurrent,
        resources_per_trial=given.resources_per_trial,
        split_draws=given.split_draws,
        split_draw_seeds=given.split_draw_seeds,
        search_seed=given.search_seed,
        trial_budget=given.trial_budget,
        relaunched_from=source.name,
        actor=actor,
    )
    if isinstance(opened, dict):
        return opened
    _start_worker(opened, _worker_env())
    return {"sweep_id": opened.name, "status": "launched"}


def run_sweep(directory: Path) -> SweepGroup:
    """Run the opened sweep at ``directory`` (:func:`open_sweep`) on Ray Tune to its final status.

    The final status is written by ``experiments.write_final_status``. The sweep keeps its
    heartbeat while it runs and trains each trial through :func:`_run_hpo_trial` in a run
    directory named for the sweep and Ray's trial id. A cancel requested before or during the run
    ends it ``canceled``. Returns the ended sweep's group (``experiments.read_sweep``).
    """
    import uuid

    from tcip_mcp.pipelines.training.hpo import (
        split_draw_search_space,
        tune_search,
    )

    project = experiments.project_of_run(directory)
    given = experiments.observe_sweep(directory).record.input

    def end(state: str, status_error: str | None) -> SweepGroup:
        experiments.write_final_status(directory, state, status_error)
        return experiments.read_sweep(directory, experiments.run_rows(project))

    if experiments.cancel_requested(directory):
        return end("canceled", _CANCEL_BEFORE_START_REASON)

    def objective_fn(config: dict, report: Report) -> None:
        try:
            from ray import tune as _tune

            tid = _tune.get_context().get_trial_id()
        except Exception:
            tid = uuid.uuid4().hex[:8]
        _run_hpo_trial(config, report, directory, tid)

    search_param_space = split_draw_search_space(
        given.param_space, given.split_draw_seeds
    )
    stop_heartbeat = experiments.keep_heartbeat(directory)
    try:
        tune_search(
            objective_fn=objective_fn,
            param_space=search_param_space,
            metric="objective",
            mode="max" if experiments.sweep_selection(given)[1] else "min",
            num_samples=given.n_trials,
            search_alg=given.search_alg,
            scheduler=given.scheduler,
            seed=given.search_seed,
            baseline_params=given.baseline_params,
            max_concurrent=given.max_concurrent,
            sweep_dir=directory,
            resources_per_trial=given.resources_per_trial,
            split_draws=given.split_draws,
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
    """Apply HPO params onto a deep copy of ``base_config`` and return the copy.

    ``base_config`` may leave unstated a value ``params`` names: a dotted key
    (``optimizer.head_lr``) lands at the nested field it names, any other key (``batch_size``) at
    the top level.
    """
    import copy

    cfg = copy.deepcopy(base_config)

    for key, value in params.items():
        if "." in key:
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


def _trial_budget_refusal(
    *,
    split_draws: Count,
    n_trials: Count,
    search_alg: str,
    trial_budget: Count | None,
    relaunched_from: str | None,
    search_param_space: dict,
    baseline_params: dict | None,
) -> str | None:
    """Return why this call's trial count violates its required budget, or None.

    A bound is read when ``trial_budget`` is stated, or on a launch (``relaunched_from`` unset)
    above one draw: Ray's own variant count over ``search_param_space``
    (:func:`~tcip_mcp.pipelines.training.hpo.planned_trial_count`) meets its two bound clauses:
    such a launch naming no ``trial_budget`` refuses naming the budget Ray's own count would
    admit; a stated ``trial_budget`` the count exceeds refuses naming the count, the budget, and
    whether even one draw exceeds it (``count // split_draws`` is one draw's own count).
    Everything else answers ``None``.
    """
    if trial_budget is None and (
        split_draws == 1 or relaunched_from is not None
    ):
        return None

    from tcip_mcp.pipelines.training.hpo import planned_trial_count

    count = planned_trial_count(
        search_param_space, n_trials, search_alg, split_draws, baseline_params
    )
    per_draw = count // split_draws
    baseline = "given" if baseline_params else "none"
    built = (
        f"Ray's own variant count over the space this call builds: n_trials={n_trials} "
        f"sample(s), search_alg={search_alg!r}, baseline {baseline}, {per_draw} per draw"
    )

    if trial_budget is None:
        return (
            f"split_draws={split_draws} pairs every sampled point with {split_draws} "
            f"partitions, so this sweep would launch {count} trials ({built}), a total the "
            f"call never stated; pass trial_budget={count} ({count} admits it as "
            "configured), or run at split_draws=1."
        )
    if count > trial_budget:
        if per_draw > trial_budget:
            return (
                f"this sweep would launch {count} trials ({built}) against "
                f"trial_budget={trial_budget}, which this sweep exceeds at one draw "
                f"({per_draw} trials per draw); lower n_trials, narrow the gridded axes, or "
                "state a larger trial_budget."
            )
        admitted_draws = trial_budget // per_draw
        return (
            f"this sweep would launch {count} trials ({built}) against "
            f"trial_budget={trial_budget}, which admits at most {admitted_draws} draw(s) at "
            f"these settings; lower split_draws to {admitted_draws}, lower n_trials, narrow "
            "the gridded axes, or state a larger trial_budget."
        )
    return None


def _split_draws_refusal(
    data: DataSpec,
    param_space: dict,
    search_alg: str,
    scheduler: dict,
    split_draws: int,
    split_draw_seeds: list[int] | None,
    baseline_params: dict | None,
) -> str | None:
    """Return why a sweep refuses ``split_draws`` above 1 before its first point resolves.

    The answer is the first of the paired path's own reasons that holds, its base config's
    validated data block being ``data``; ``None`` when nothing here objects, and for one draw. A
    base config bound to a selection skips the validation-share leg.
    """
    if split_draws <= 1:
        return None
    from tcip_mcp.pipelines.training.hpo import (
        _NATIVE_SEARCH,
        FIFO_SCHEDULER,
        SPLIT_DRAW_SEED_KEY,
        search_alg_key,
    )

    if isinstance(data.split, DrawnSplit) and not data.split.val_ratio:
        return (
            "split_draws needs a drawn validation split, and base_config states "
            "data.split.val_ratio zero."
        )
    if search_alg_key(search_alg) not in _NATIVE_SEARCH:
        native = sorted(_NATIVE_SEARCH)
        return (
            f"split_draws pairs a grid axis through Ray's own BasicVariantGenerator, which "
            f"only a native search_alg ({native}) builds; search_alg={search_alg!r} does not."
        )
    if scheduler["name"] != FIFO_SCHEDULER:
        return (
            f"split_draws makes each draw a blocked comparison, and a pruning scheduler "
            f"({scheduler['name']!r}) could end one draw before another completes; pass "
            f"scheduler={{'name': {FIFO_SCHEDULER!r}}} with split_draws."
        )
    if split_draw_seeds is not None and len(split_draw_seeds) != split_draws:
        return (
            f"split_draw_seeds has {len(split_draw_seeds)} seed(s) but split_draws="
            f"{split_draws}: one seed per draw."
        )
    repeated = sorted(s for s, n in Counter(split_draw_seeds).items() if n > 1)
    if repeated:
        return (
            f"split_draw_seeds repeats {repeated}: a spread over the same partition drawn "
            "twice is not a spread, and group_split_draws only counts a point eligible on "
            "its distinct planned seeds, not on one seed completed twice."
        )
    if baseline_params and SPLIT_DRAW_SEED_KEY in baseline_params:
        return (
            f"baseline_params names {SPLIT_DRAW_SEED_KEY!r}: Ray's own "
            "preset-variant pinning would pin every draw to that one seed instead of "
            f"pairing the grid; drop {SPLIT_DRAW_SEED_KEY!r} from baseline_params."
        )
    if SPLIT_DRAW_SEED_KEY in param_space:
        return (
            f"param_space already sweeps {SPLIT_DRAW_SEED_KEY}, the same axis split_draws "
            "adds as a paired grid; state the crossing through split_draws/"
            "split_draw_seeds, not a second data.split.seed axis."
        )
    other_data_axes = sorted(
        k
        for k in param_space
        if k.startswith("data.") and k != SPLIT_DRAW_SEED_KEY
    )
    if other_data_axes:
        return (
            f"split_draws pairs the identical partition with every sampled point at each "
            f"draw, and param_space sweeps {other_data_axes} beside it: a data.* axis other "
            f"than {SPLIT_DRAW_SEED_KEY} changes what a point admits or how it draws, so "
            "draw k would no longer be the same partition for every point."
        )
    return None


def _spatial_draws_issue(
    resolution: ResolvedRun, split_draws: int
) -> str | None:
    """Return the refusal ``split_draws`` above 1 meets on a spatially split first point.

    The refusal applies when the sweep's first point, resolved as ``resolution``, split one source
    over its own pixels (its ``spatial`` split), a partition no draw of ``data.split.seed``
    varies; ``None`` otherwise.
    """
    if split_draws <= 1 or resolution.partition.spatial is None:
        return None
    return (
        f"split_draws={split_draws} redraws the split, and base_config's one admitted source is "
        "split over its own pixels, a partition placed by declared order that does not vary with "
        "data.split.seed: no draw holds a different partition out and the spread would be "
        "training-seed noise. Run at split_draws=1, or sweep a dataset with two or more admitted "
        "sources or a config bound to a selection."
    )


class SeedAxisRefusal(NamedTuple):
    """Why a caller-supplied ``data.split.seed`` axis in ``param_space`` refuses at one draw.

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
    "foreground groups across them); an unbound config states val_ratio above zero; search_alg "
    "is a native search algorithm; the scheduler prunes nothing; split_draw_seeds is one per "
    "draw and distinct; baseline_params names no seed; no other data.* axis is in param_space; "
    "a trial_budget is stated on a launch that is not a relaunch, and Ray's variant count over "
    "the sweep fits under it; and, for a built-in config with tiling on, more than one "
    "trainable source is admitted under data.images_dir (a single admitted source's own "
    "single-source spatial-strip path pairs no distinct partition with any draw). A single fixed "
    "seed belongs in "
    "base_config's own data.split.seed: the drawn path's partition and a selection-bound "
    "config's redraw depend on it, and the single-source spatial path's never does (the spatial "
    "path still records the config's value on the split record, a different fact, not claimed "
    "here)."
)


def caller_split_seed_refusal(
    param_space: dict, split_draws: int
) -> SeedAxisRefusal | None:
    """Return the refusal of a ``data.split.seed`` axis of ``param_space`` at one draw, or None.

    ``param_space`` naming ``data.split.seed`` as its own axis while ``split_draws`` draws at most
    one partition is refused whatever the sampler.
    """
    from tcip_mcp.pipelines.training.hpo import SPLIT_DRAW_SEED_KEY

    if split_draws > 1 or SPLIT_DRAW_SEED_KEY not in param_space:
        return None
    return SeedAxisRefusal(reason=_SEED_AXIS_REASON, remedy=_SEED_AXIS_REMEDY)


def _preflight_points(
    param_space: dict, search_alg: str
) -> list[tuple[str, dict]]:
    """List the points a sweep's preflight checks, each with its label.

    The points are read through the space's shape reader (``hpo.space_axes`` as
    ``search_alg`` grids it, refusing as it refuses): the corner at every axis's first value, plus
    one variant per categorical choice and one per range end, each holding every other axis at
    that corner, each point listed once. Combinations of several axes away from the corner, and a
    range's interior, are not among them.
    """
    from tcip_mcp.pipelines.training.hpo import (
        GRID_SEARCH,
        search_alg_key,
        space_axes,
    )

    axes = space_axes(
        param_space, grid=search_alg_key(search_alg) == GRID_SEARCH
    )
    base = {key: values[0] for key, (_kind, values) in axes.items()}
    points: list[tuple[str, dict]] = [("the first sampled point", dict(base))]
    for key, (kind, values) in axes.items():
        labels = (
            [f"{key}={v!r}" for v in values]
            if kind == "categorical"
            else [f"{key} low={values[0]!r}", f"{key} high={values[1]!r}"]
        )
        for label, value in zip(labels, values):
            point = {**base, key: value}
            if all(point != listed for _, listed in points):
                points.append((label, point))
    return points


def _one_real_batch(
    task: str, train_ds: BaseDataset, n: int = 2
) -> Batch | str:
    """Collate one real batch of the run's resolved train dataset.

    Returns one collated ``(images, targets)`` of the first ``n`` items of ``train_ds``, or the
    reason it failed when it yields none or cannot be built.
    """
    try:
        from tcip_mcp.pipelines.training.collation import task_collate

        items = [train_ds[i] for i in range(min(n, len(train_ds)))]
        if not items:
            return "the dataset built but is empty"
        return task_collate(task)(items)
    except Exception as exc:  # noqa: BLE001, an unbuildable batch is a caller decision, not a crash
        logger.info(
            "could not build a real batch to smoke task %r: %s", task, exc
        )
        return f"{type(exc).__name__}: {exc}"


@tool()
def evaluate_model(
    project: Path,
    experiment_id_or_ckpt: str,
    images_dir: str,
    labels_dir: str | None = None,
    *,
    stated: Stated,
    iou_threshold: float,
    tiling: TilingSpec | None = None,
    use_tiled_inference: bool = False,
    trait: str | None = None,
) -> dict:
    """Evaluate a trained checkpoint on a (held-out) dataset and return the result.

    Computes the same per-task metrics as validation: detection/instance_seg get the
    matcher's precision/recall/F1 and average precision (by mask for an instance_seg pass that
    collects and cuts masks, by box for one stating ``masks`` false);
    classification/ordinal/regression get the in-house scalar metrics. Writes
    nothing: the training run's directory is never touched. The reference is read under the class
    space the checkpoint records, its map included.

    Three detection and instance_seg eval regimes:
      * Untiled default (no ``tiling``, checkpoint trained without tiling) -> single full-res
      forward pass, ``eval_regime="full-frame-single-pass"``.
      * ``tiling`` set (or a run id whose training was tiled, reused automatically) -> tile-level
      diagnostic over per-tile ground truth; not the delivery metric.
      * ``use_tiled_inference=True`` -> the delivery-grade full-frame metric (tiled inference
      reconstructed to full frame, matched to full-frame GT). Tile
      geometry is the stated one, else the checkpoint's own persisted or native-frame training
      geometry, else derived from the evaluated ground truth (see ``run_full_frame_evaluation``).

    Args:
        experiment_id_or_ckpt: The id of a run of this project (uses the checkpoint its final
            status names; a run that completed none refuses), else a checkpoint path. Either way
            the resolved checkpoint must be registered in
            this project's registry (``register_model``, explicit mode for a foreign or bespoke
            checkpoint) or this door refuses before loading it.
        images_dir: The capture directory of the evaluation split.
        labels_dir: The masks dir (semantic_seg) or the GT CSV path
            (classification/ordinal/regression, one row per image stem); omitted for label
            documents (detection/instance_seg), which the images' own keys address. Admitted once
            with ``images_dir`` (``label_queries.admit``) before any regime runs, a refusal naming
            both. The task is the checkpoint's own.
        stated: The execution values to state rather than derive (``execution.Stated``),
            ``{}`` stating none: the operating ``conf`` P/R/F1 are reported at, required of a
            detector, whether an instance segmentation pass collects ``masks`` and, when it does,
            the ``mask_cut`` it cuts them by (``execution.MaskCut``), and on the delivery-grade
            path the ``postprocess`` merge, required, and the ``tile_size``, ``overlap`` and
            ``cross_tile_nms`` (derived from the evaluated ground truth when unstated), with the
            ``rationale`` for them. The resolved record,
            each value's source with it, is returned under ``execution``.
        iou_threshold: The IoU a match must reach, stated (the comparability convention is
            ``schemas.COMPARABILITY_IOU``).
        tiling: Optional tiling (``schemas.TilingSpec``: tile_size, overlap, skip_empty, ...) for
            a tile-level eval, refused for a task that does not tile
            (``schemas.run_tiling``). None + a run id reuses the run's training tiling;
            None + a checkpoint path stays untiled.
        use_tiled_inference: Score the delivery regime (full-frame via tiled inference);
            refused for a checkpoint whose task does not tile.
        trait: When set, the derived localization criterion of the trait's latest confirmed
            revision (traits.py, e.g. a count trait's center-match) governs the reported count
            and the selection f1; the convention's counts at ``iou_threshold`` are kept as
            labeled comparability metrics. Absent -> the IoU convention governs.

    """
    import torch
    from torch.utils.data import DataLoader

    from tcip_mcp.operationalization import (
        OperationalizationRefusedError,
        latest_confirmed,
    )
    from tcip_mcp.pipelines.data.split_construction import predictor_dataset
    from tcip_mcp.pipelines.execution import prepare
    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.eval_runners import (
        run_full_frame_evaluation,
        run_test_evaluation,
    )
    from tcip_mcp.traits import TraitUnknownError

    try:
        trait_entry = latest_confirmed(trait, project).entry if trait else None
    except (TraitUnknownError, OperationalizationRefusedError) as exc:
        return {"error": str(exc)}

    images_dir = str(located(images_dir, project))
    labels_dir = labels_dir and str(located(labels_dir, project))

    from tcip_mcp.model_registry import (
        CheckpointNotFoundError,
        UnregisteredCheckpointError,
        load_registered_checkpoint,
    )

    run_dir = experiments.find_run(experiment_id_or_ckpt, project=project)
    by_run = run_dir is not None
    try:
        if run_dir is not None:
            completed = experiments.observe_run(run_dir).checkpoint
            if completed is None:
                return {
                    "error": f"run {experiment_id_or_ckpt} has completed no checkpoint"
                }
            ckpt = Path(completed.path)
        else:
            ckpt = located(experiment_id_or_ckpt, project)
        checkpoint = load_registered_checkpoint(ckpt, project=project)
        task = checkpoint.task
        scope = checkpoint.spec.data.recorded_scope
    except (
        CheckpointNotFoundError,
        UnregisteredCheckpointError,
        ValueError,
    ) as exc:
        return {"error": str(exc)}
    from tcip_mcp.pipelines.model_contract import TILING_TASKS

    if use_tiled_inference and task not in TILING_TASKS:
        return {
            "error": f"use_tiled_inference is set for a checkpoint of task {task!r}, and "
            f"only a {sorted(TILING_TASKS)} checkpoint predicts over tiles; drop "
            "use_tiled_inference."
        }
    from tcip_annotation.json_io import UnreadableLabelDocumentError
    from tcip_mcp.pipelines.data.label_queries import admit, require_admitted

    try:
        admitted = admit(images_dir, labels_dir, scope=scope)
        require_admitted(admitted)
    except (ValueError, UnreadableLabelDocumentError) as exc:
        return {
            "error": f"the ground truth of images_dir={images_dir!r} with "
            f"labels_dir={labels_dir!r} admits nothing to evaluate: {exc}"
        }

    if use_tiled_inference:
        try:
            return run_full_frame_evaluation(
                checkpoint,
                admitted,
                stated=stated,
                iou_threshold=iou_threshold,
                trait=trait_entry,
            )
        except (ValueError, UnreadableLabelDocumentError) as exc:
            return {"error": str(exc)}

    # Tile-level diagnostic (or untiled); a run id reuses its training tiling.
    if tiling is None and by_run:
        tiling = checkpoint.spec.data.tiling

    # The checkpoint's own untiled pass: the width its predictor reads images at sizes the loader,
    # and its execution record governs the model that scores them.
    try:
        pass_ = prepare(
            checkpoint, stated.model_copy(update={"tile": False})
        ).runnable()
    except ValueError as exc:
        return {"error": str(exc)}
    predictor = pass_.predictor

    try:
        dataset = predictor_dataset(
            task, admitted.every_sample(), scope, predictor, tiling
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Failed to build dataset: {exc}"}

    loader = DataLoader(dataset, batch_size=4, collate_fn=task_collate(task))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return run_test_evaluation(
            pass_,
            loader,
            device,
            iou_threshold=iou_threshold,
            tiling=tiling,
            trait=trait_entry,
        )
    except ValueError as exc:
        return {"error": str(exc)}
