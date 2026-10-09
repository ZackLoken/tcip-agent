"""A run as a directory: ``<project>/.tcip/experiments/<experiment_id>/``.

Each file in it has one writer:

- ``run.json``: the launcher's record of the run, its input and what that input resolved to,
  written once before the child starts.
- ``metrics.jsonl``: the child's epoch rows and per-batch rows (:func:`partition_rows`), created
  empty with the directory and appended.
- ``heartbeat``: touched by the child while it lives.
- ``cancel_requested.json``: the first cancellation request (:func:`request_cancel`).
- ``final_status.json``: the outcome, written once at exit (:func:`write_final_status`).

Checkpoints, TensorBoard logs, a bespoke run's source snapshot and its recorded artifacts are
files beside them. A sweep is a directory beside the runs, told from one by its ``sweep.json``,
holding its trial run directories with its own input, heartbeat, cancellation and final status
(``tools.training_tools``); a trial is a run like any other, named by its directory.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, BinaryIO, NamedTuple, cast

from pydantic import BaseModel

from tcip_store import (
    BadKeyError, DecodeError, decode_value, encode_log_line, encode_record, finite_number,
    stored_number,
)
from tcip_store.file_backend import is_sharing_violation, retry_while_denied
from tcip_store.values import NOT_FINITE_SUFFIX

from tcip_mcp.audit import now_iso
from tcip_mcp.pipelines.data.selection import SAMPLE_PATHS
from tcip_mcp.registry_paths import PathFields, recorded_paths, runtime_paths, within

if TYPE_CHECKING:
    from tcip_mcp.pipelines.schemas import DataSpec, SpatialManifest, TrainConfigSchema

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(".tcip/experiments")

RUN_FILE = "run.json"
METRICS_FILE = "metrics.jsonl"
HEARTBEAT_FILE = "heartbeat"
CANCEL_FILE = "cancel_requested.json"
FINAL_STATUS_FILE = "final_status.json"
SWEEP_FILE = "sweep.json"
"""A sweep directory's input, written once before its first trial."""
TENSORBOARD_DIR = "tensorboard"
"""A run directory's TensorBoard event directory."""
EPOCH_KEY = "epoch"
TIMESTAMP_KEY = "timestamp"
"""The keys every metrics row carries beside its metrics: its epoch and the instant it landed."""
STEP_KEY = "step"
"""The key a per-batch metrics row carries, the training step it was logged at; an epoch row
never carries it, so it is the row's kind (:func:`partition_rows`)."""

DATA_PATHS: PathFields = (
    ("images_dir",), ("labels_dir",), ("plant_csv_paths", "[]"), ("split", "selection_dir"))
"""The fields of a run's data section that name a path."""

_PARAMETER_PATHS: PathFields = tuple(
    (".".join(("data", *(step for step in field if step != "[]"))),
     *(step for step in field if step == "[]"))
    for field in DATA_PATHS)
"""The same fields in a flat map of sweep parameters, each named by its dotted config path."""

_SPACE_PATHS: PathFields = tuple((name, "choices", "[]", *rest) for name, *rest in _PARAMETER_PATHS)

RECORD_PATHS: dict[str, PathFields] = {
    RUN_FILE: (
        *within(("config", "data"), DATA_PATHS), *within(("resolved", "data"), DATA_PATHS),
        *within(("resolved", "partition", "samples", "[]"), SAMPLE_PATHS),
        ("resolved", "partition", "selection", "selection_dir"),
        ("resume_from",), *within(("trial_params",), _PARAMETER_PATHS)),
    SWEEP_FILE: (
        *within(("input", "base_config", "data"), DATA_PATHS),
        *within(("input", "baseline_params"), _PARAMETER_PATHS),
        *within(("input", "param_space"), _SPACE_PATHS)),
    FINAL_STATUS_FILE: (("checkpoint", "path"),),
}
"""The fields of each record of a run or sweep directory that name a path, each stored against
the project the directory lies under (:func:`project_of_run`)."""

FINAL_STATES = ("completed", "failed", "canceled")
"""The states a final status names."""
TERMINAL_STATES = frozenset({*FINAL_STATES, "interrupted"})
"""Every state a run or sweep ends in: a final status's, and ``interrupted``, the state a directory
with no final status reads as once its heartbeat is stale."""

HEARTBEAT_STALE_SECONDS = float(os.environ.get("TCIP_HEARTBEAT_STALE_SECONDS", "600"))
"""How long a directory with no final status reads ``running`` after its last sign of life,
from ``$TCIP_HEARTBEAT_STALE_SECONDS``."""


def experiments_dir(project: Path | str) -> Path:
    """The directory every run and sweep directory of ``project`` sits in, the one place a name
    is reserved (:func:`create_run_directory`)."""
    return Path(project) / EXPERIMENTS_DIR


def project_of_run(run_dir: Path) -> Path:
    """The project a run directory, a sweep directory or a sweep's trial directory lies under:
    the parent of the ``.tcip`` directory on its path. Refuses a directory under no ``.tcip``."""
    for parent in Path(run_dir).parents:
        if parent.name == ".tcip":
            return parent.parent
    raise ValueError(f"{run_dir} lies under no project's .tcip directory")


def run_name(name: str) -> str:
    """``name`` once it is known to be one directory name. A separator, a drive, an empty name or
    a dot name refuses with ``BadKeyError``."""
    if not name or name in (".", "..") or PureWindowsPath(name).name != name:
        raise BadKeyError(
            f"{name!r} is not a single directory name: a name carrying a separator, a drive or a "
            "parent reference would address a directory outside the one it names."
        )
    return name


def experiment_dir(experiment_id: str, *, project: Path | str) -> Path:
    """The directory of the run or sweep ``experiment_id`` names. Refuses an id that is not a
    single directory name (:func:`run_name`)."""
    return experiments_dir(project) / run_name(experiment_id)


def find_run(experiment_id: str, *, project: Path | str) -> Path | None:
    """The run directory of ``project`` (:func:`run_dirs`) named ``experiment_id``, a sweep's
    trial included, or ``None``."""
    return next((d for d in run_dirs(project) if d.name == experiment_id), None)


def find_sweep(sweep_id: str, *, project: Path | str) -> Path | None:
    """The sweep directory of ``project`` (:func:`sweep_dirs`) named ``sweep_id``, or ``None``."""
    return next((d for d in sweep_dirs(project) if d.name == sweep_id), None)


def named_directory(name: str, *, project: Path | str) -> Path | None:
    """The run, trial or sweep directory of ``project`` named ``name``, or ``None``."""
    return next((d for d in (*run_dirs(project), *sweep_dirs(project)) if d.name == name), None)


def board_of(directory: Path) -> Path:
    """The TensorBoard log directory of a run or sweep directory: a sweep's own directory, every
    trial's events beneath it, or a run's :data:`TENSORBOARD_DIR`."""
    return directory if (directory / SWEEP_FILE).is_file() else directory / TENSORBOARD_DIR


def mint_experiment_id(prefix: str = "run") -> str:
    """A fresh run id: ``<prefix>_<epoch-seconds>_<6 hex chars>``."""
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:6]}"


# ── the files ────────────────────────────────────────────────────────────────


class RunDirectoryExistsError(FileExistsError):
    """A launch named a run or sweep directory that already exists."""


def create_run_directory(directory: Path) -> Path:
    """Create ``directory`` and its parents, refusing with :class:`RunDirectoryExistsError` one that
    already exists: runs and sweeps share one parent, so this creation is the name's reservation."""
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise RunDirectoryExistsError(
            f"{directory} already exists: runs, trials and sweeps share one namespace, so a new "
            "one takes a new name."
        ) from None
    return directory


def open_run_directory(run_dir: Path, compose: Callable[[Path], dict]) -> None:
    """Create the run directory ``run_dir`` (:func:`create_run_directory`) with its empty metrics
    log, then write the record ``compose(run_dir)`` returns as its ``run.json`` once, so a
    directory holding a launch record holds every file a reader of a live run reads. ``compose``
    may write files of its own into the directory first."""
    create_run_directory(run_dir)
    (run_dir / METRICS_FILE).touch(exist_ok=False)
    write_record(run_dir / RUN_FILE, compose(run_dir))


def write_record(path: Path, value: Any) -> None:
    """Publish the run or sweep record ``value`` at ``path`` once (:func:`write_once`), each path
    :data:`RECORD_PATHS` names for its file stored against the directory's project."""
    write_once(path, recorded_paths(value, RECORD_PATHS[path.name], project_of_run(path)))


def read_run_record(path: Path) -> Any:
    """The run or sweep record at ``path`` (:func:`read_record`), each path :data:`RECORD_PATHS`
    names for its file resolved against the directory's project."""
    return runtime_paths(read_record(path), RECORD_PATHS[path.name], project_of_run(path))


def publish_once(path: Path, write: Callable[[BinaryIO], object]) -> None:
    """Write a file's whole content through ``write`` into a staging file beside ``path``, then
    publish it under ``path`` without replacing anything, so a file under its final name is always
    whole. A name already published refuses with ``FileExistsError``; a write that fails publishes
    nothing."""
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.staging")
    try:
        with open(staging, "xb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        # Windows' rename refuses an existing destination, where a POSIX rename would replace it.
        if os.name == "nt":
            os.rename(staging, path)
        else:
            os.link(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def write_once(path: Path, value: Any) -> None:
    """Publish ``value`` as the JSON record at ``path`` (:func:`publish_once`). Refuses a value
    JSON cannot hold before anything is created, and a path already written with
    ``FileExistsError``."""
    data = encode_record(value)
    publish_once(path, lambda handle: handle.write(data))


def write_final_status(directory: Path, state: str, status_error: str | None,
                       **outcome: Any) -> None:
    """Write the final status of the run or sweep at ``directory`` once: ``state``, the instant,
    ``status_error``, the error behind it, and ``outcome``, what it produced (a run's
    ``checkpoint``). Refuses (``ValueError``) a state outside :data:`FINAL_STATES`, and a status
    already written with ``FileExistsError``."""
    if state not in FINAL_STATES:
        raise ValueError(f"{state!r} is not a final state; a final status names one of "
                         f"{list(FINAL_STATES)}.")
    write_record(directory / FINAL_STATUS_FILE,
                 {"state": state, "ended": now_iso(), "status_error": status_error, **outcome})


def append_row(path: Path, row: dict) -> None:
    """Append ``row`` as one line of the JSON log at ``path``, refusing a row JSON cannot hold
    and (``FileNotFoundError``) a log its directory was not opened with."""
    line = encode_log_line(row) + b"\n"
    with open(path, "r+b") as handle:
        handle.seek(0, os.SEEK_END)
        handle.write(line)


SHARING_RETRY_BUDGET_S = 0.35
"""Provisional, chosen rather than measured: how long a record read keeps retrying a sharing
violation (``tcip_store.file_backend.retry_while_denied``) before raising it."""


def read_record(path: Path) -> Any:
    """The JSON record at ``path``. A Windows sharing violation is retried while
    :data:`SHARING_RETRY_BUDGET_S` is unspent and raised once it is, no read starting after
    the deadline; every other ``PermissionError``,
    a missing file (``FileNotFoundError``) and bytes that do not decode (``DecodeError`` naming
    the file) raise at once."""
    data = retry_while_denied(path.read_bytes, SHARING_RETRY_BUDGET_S,
                              denied=is_sharing_violation)
    try:
        return decode_value(data)
    except ValueError as exc:
        raise DecodeError(f"{path} does not decode as JSON: {exc}") from exc


def read_rows(path: Path, *, after: int = 0) -> tuple[list[dict], int]:
    """Every complete row of the JSON log at ``path`` past byte offset ``after``, and the offset
    past the last complete one. A line still being appended is left for the next read. A log that
    does not exist raises ``FileNotFoundError``; a complete line that does not decode raises
    ``DecodeError`` naming the file."""
    with open(path, "rb") as handle:
        handle.seek(after)
        data = handle.read()
    end = data.rfind(b"\n") + 1
    rows: list[dict] = []
    for line in data[:end].splitlines():
        try:
            rows.append(decode_value(line))
        except ValueError as exc:
            raise DecodeError(f"{path} holds a row that does not decode: {exc}") from exc
    return rows, after + end


# ── liveness ─────────────────────────────────────────────────────────────────


def touch_heartbeat(directory: Path) -> None:
    """Mark ``directory``'s process alive now."""
    (directory / HEARTBEAT_FILE).touch()


def keep_heartbeat(directory: Path) -> threading.Event:
    """Touch ``directory``'s heartbeat now and every tenth of :data:`HEARTBEAT_STALE_SECONDS` on a
    daemon thread until the returned event is set."""
    stop = threading.Event()
    touch_heartbeat(directory)

    def beat() -> None:
        while not stop.wait(HEARTBEAT_STALE_SECONDS / 10):
            try:
                touch_heartbeat(directory)
            except OSError:
                logger.warning("could not touch the heartbeat of %s", directory, exc_info=True)

    threading.Thread(target=beat, daemon=True).start()
    return stop


def last_alive(directory: Path, record_file: str = RUN_FILE) -> float:
    """The latest sign of life of a run or sweep directory, as a POSIX time: its heartbeat's
    modification time, or its launch record's before the heartbeat was first touched."""
    heartbeat = directory / HEARTBEAT_FILE
    return (heartbeat if heartbeat.exists() else directory / record_file).stat().st_mtime


class RunEndedError(ValueError):
    """A write addressed a run or sweep whose final status is written."""


def require_open(directory: Path) -> None:
    """Refuse (:class:`RunEndedError`) the run or sweep at ``directory`` once its final status is
    written: from then on it takes no write."""
    if (directory / FINAL_STATUS_FILE).exists():
        raise RunEndedError(
            f"{directory.name} has ended: its final status is written, so it takes no "
            "more writes.")


def request_cancel(directory: Path) -> str:
    """Record a cancellation request of the run or sweep at ``directory`` and return the time of
    its first one: a later request leaves that record as written. Refuses a directory that has
    ended (:func:`require_open`)."""
    require_open(directory)
    try:
        write_once(directory / CANCEL_FILE, {"requested": now_iso()})
    except FileExistsError:
        pass
    return read_record(directory / CANCEL_FILE)["requested"]


def cancel_requested(directory: Path) -> bool:
    """Whether a cancellation of the run or sweep at ``directory`` has been requested."""
    return (directory / CANCEL_FILE).exists()


# ── reading a run ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunObservation:
    """One run or sweep directory read once: its launch record, its final status (``None`` while
    it has not ended), its latest sign of life as a POSIX time (:func:`last_alive`), and the state
    those derive."""

    directory: Path
    record: dict
    final: dict | None
    alive: float
    state: str

    @property
    def metrics_log(self) -> Path:
        """The run's metrics log, which exists from the moment the run opened."""
        return self.directory / METRICS_FILE

    @property
    def checkpoint(self) -> dict | None:
        """The checkpoint a completed training run's final status selects, or ``None`` for a run
        that has not completed."""
        if self.final is None or self.state != "completed":
            return None
        return self.final["checkpoint"]

    @property
    def heartbeat(self) -> str:
        """Its latest sign of life as an ISO-8601 instant in UTC."""
        return datetime.fromtimestamp(self.alive, timezone.utc).isoformat()

    @property
    def status_error(self) -> str | None:
        """The error the final status names, ``None`` while it has not ended."""
        return self.final["status_error"] if self.final is not None else None

    @property
    def resolution(self) -> dict | None:
        """What a run's input resolved to at launch (its ``data``, ``partition`` and
        ``objective``), ``None`` for a sweep trial whose sampled point was refused at its
        admission or failed to resolve."""
        return self.record["resolved"]

    @cached_property
    def spec(self) -> TrainConfigSchema:
        """A run's launch ``config`` validated once (``schemas.train_config``, which refuses an
        invalid one)."""
        from tcip_mcp.pipelines.schemas import train_config

        return train_config(self.record["config"])

    @cached_property
    def resolved_data(self) -> DataSpec | None:
        """The data block a run's launch resolved (:attr:`resolution`'s), validated once;
        ``None`` for a run that resolved to nothing."""
        from tcip_mcp.pipelines.schemas import DataSpec

        resolved = self.resolution
        return None if resolved is None else DataSpec.model_validate(resolved["data"])

    @property
    def sweep(self) -> str | None:
        """The sweep a trial run belongs to, ``None`` for a run launched on its own."""
        parent = self.directory.parent
        return parent.name if (parent / SWEEP_FILE).is_file() else None


def observe(directory: Path) -> RunObservation:
    """Read the sweep (holding its ``sweep.json``) or run at ``directory`` once. Its state is the
    one its final status names once one is written; before that ``running`` while its latest sign
    of life is within :data:`HEARTBEAT_STALE_SECONDS`, else ``interrupted``. A run holds its
    metrics log from the moment it opened; one missing it refuses with ``FileNotFoundError``
    naming it. A completed run's checkpoint is read, and refused when absent, by whatever loads
    it: an archive may carry a run without its weights."""
    record_file = SWEEP_FILE if (directory / SWEEP_FILE).is_file() else RUN_FILE
    record = read_run_record(directory / record_file)
    final_path = directory / FINAL_STATUS_FILE
    final = read_run_record(final_path) if final_path.exists() else None
    alive = last_alive(directory, record_file)
    if final is not None:
        state = final["state"]
    else:
        state = "running" if time.time() - alive <= HEARTBEAT_STALE_SECONDS else "interrupted"
    observation = RunObservation(directory, record, final, alive, state)
    if record_file == RUN_FILE and not observation.metrics_log.is_file():
        raise FileNotFoundError(f"run {directory.name} ({state}) is missing "
                                f"{observation.metrics_log}.")
    return observation


def _holding(parent: Path, record_file: str) -> list[Path]:
    """Every directory under ``parent`` holding ``record_file``, in name order."""
    return (
        sorted(d for d in parent.iterdir() if (d / record_file).is_file())
        if parent.is_dir()
        else []
    )


def sweep_dirs(project: Path | str) -> list[Path]:
    """Every sweep directory of ``project`` that holds its input, in name order."""
    return _holding(experiments_dir(project), SWEEP_FILE)


def run_dirs(project: Path | str) -> list[Path]:
    """Every run directory of ``project`` that holds a launch record, each sweep's trials included,
    in name order."""
    return sorted([*_holding(experiments_dir(project), RUN_FILE),
                   *(d for sweep in sweep_dirs(project) for d in _holding(sweep, RUN_FILE))],
                  key=lambda d: d.name)


def run_observations(project: Path | str) -> list[RunObservation]:
    """Every run of ``project`` (:func:`run_dirs`), each observed once."""
    return [observe(d) for d in run_dirs(project)]


def live_run_conflict(project: Path) -> str | None:
    """The refusal a run, trial or sweep of ``project`` (:func:`training_listing`) still running
    answers with, or ``None``."""
    listing = training_listing(project)
    trials = [trial for sweep in listing.sweeps for trial in sweep.trials]
    running = [sweep.sweep_id for sweep in listing.sweeps if sweep.state == "running"]
    running += [row.experiment_id for row in (*listing.runs, *trials) if row.state == "running"]
    if not running:
        return None
    return (f"{running[0]!r} is running; ask the agent to cancel it (cancel_training) or wait for "
            "it to finish")


def find_observation(experiment_id: str, *, project: Path | str) -> RunObservation | None:
    """The run ``experiment_id`` names (:func:`find_run`), observed, or ``None``."""
    run_dir = find_run(experiment_id, project=project)
    return observe(run_dir) if run_dir is not None else None


class RunResolution(NamedTuple):
    """What a run resolved at launch: its data block (``RunObservation.resolved_data``), its
    ``partition`` and the within-image split that partition records
    (``split_construction.partition_spatial``)."""

    data: DataSpec
    partition: dict
    spatial: SpatialManifest | None


def run_resolution(experiment_id: str, *, project: Path | str) -> RunResolution:
    """What the run ``experiment_id`` names resolved at launch (:func:`find_observation`).
    Refuses (``ValueError``) an id naming no run directory or one whose input resolved to
    nothing."""
    observation = find_observation(experiment_id, project=project)
    data = observation.resolved_data if observation is not None else None
    if observation is None or observation.resolution is None or data is None:
        raise ValueError(f"no run directory records what {experiment_id!r} resolved to.")
    from tcip_mcp.pipelines.data.split_construction import partition_spatial

    partition = observation.resolution["partition"]
    return RunResolution(data, partition, partition_spatial(partition))


def partition_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A run's metrics-log ``rows`` split by kind, each in logged order: its epoch rows, then its
    per-batch rows, those carrying :data:`STEP_KEY`. The one place a row's kind is read."""
    epochs: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    for row in rows:
        (batches if STEP_KEY in row else epochs).append(row)
    return epochs, batches


def epoch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A run's metrics-log ``rows`` as one row per epoch, in the order each epoch was first
    logged: the epoch rows (:func:`partition_rows`) of one epoch merged, a later row's keys laid
    over the earlier's, a metric and its non-finite state companion as one value."""
    merged: dict[str, dict[str, Any]] = {}
    for row in partition_rows(rows)[0]:
        epoch = json.dumps(row.get(EPOCH_KEY), sort_keys=True, default=str)
        restated = {f"{key}{NOT_FINITE_SUFFIX}" for key in row}
        earlier = {k: v for k, v in merged.get(epoch, {}).items() if k not in restated}
        merged[epoch] = {**earlier, **row}
    return list(merged.values())


def best_selection(rows: list[dict[str, Any]], objective: dict) -> float | None:
    """The best finite ``selection`` among a run's metrics-log epoch rows
    (:func:`partition_rows`) in the direction of ``objective``, the run's recorded
    ``{"selection_metric", "higher_is_better"}``; ``None`` when no row carries one."""
    values = [float(row["selection"]) for row in partition_rows(rows)[0]
              if finite_number(row.get("selection"))]
    if not values:
        return None
    return max(values) if objective["higher_is_better"] else min(values)


class RunRow(BaseModel):
    """One run as every listing, monitor and stream shows it."""

    experiment_id: str
    state: str
    created: str
    relaunched_from: str | None
    sweep: str | None
    trial_params: dict[str, Any] | None
    builder: str
    task: str
    images_dir: str | None
    current_epoch: float | None
    best_metric: float | None
    best_metric_name: str | None
    output_dir: str
    status_error: str | None
    heartbeat: str
    launch: dict[str, str | None] | None


def run_summary(observation: RunObservation, rows: list[dict[str, Any]],
                launch: dict[str, str | None] | None) -> RunRow:
    """One run's :class:`RunRow` over its metrics-log ``rows`` (:func:`read_rows` of its
    ``metrics_log``, read as :func:`epoch_rows`) and the agent identity its ``launch`` event
    declared (:func:`launch_declarations`, ``None`` for a run no launch event names): its
    record's own
    fields, its last logged epoch, and its best selection value under the objective its launch
    resolved (:func:`best_selection`) with that objective's metric name, both ``None`` for a run
    that resolved to nothing."""
    record, resolution, spec = observation.record, observation.resolution, observation.spec
    objective = resolution["objective"] if resolution is not None else None
    rows = epoch_rows(rows)
    return RunRow(
        experiment_id=observation.directory.name, state=observation.state,
        created=record["created"], relaunched_from=record["relaunched_from"],
        sweep=observation.sweep, trial_params=record["trial_params"],
        builder=spec.model_source.builder, task=spec.model_source.task,
        images_dir=spec.data.images_dir,
        current_epoch=rows[-1].get(EPOCH_KEY) if rows else None,
        best_metric=best_selection(rows, objective) if objective is not None else None,
        best_metric_name=objective["selection_metric"] if objective is not None else None,
        output_dir=str(observation.directory), status_error=observation.status_error,
        heartbeat=observation.heartbeat, launch=launch,
    )


def run_rows(project: Path | str) -> list[RunRow]:
    """Every run of ``project`` (:func:`run_observations`) as its :func:`run_summary` row."""
    launches = launch_declarations(project)
    return [run_summary(obs, read_rows(obs.metrics_log)[0], launches.get(obs.directory.name))
            for obs in run_observations(project)]


def _completed_with_value(trial: RunRow) -> bool:
    """Whether a sweep's trial completed reporting a value under the sweep's objective."""
    return trial.state == "completed" and trial.best_metric is not None


def group_split_draws(trials: list[RunRow], planned_seeds: list[int]) -> list[dict]:
    """Group a sweep's trial rows by the point each draw shares (every param but
    ``hpo.SPLIT_DRAW_SEED_KEY``), each group carrying its ``split_draws`` block.

    ``planned_seeds`` names every seed the sweep asked for; a group is ``eligible`` for best only
    when every one of them completed for that point and the group holds no trial that did not.
    Pass an empty list to accept any single completed trial per point regardless of seed identity.

    A group's block always carries ``n`` (every trial of the point), ``n_complete`` (trials among
    them that completed with a value) and ``seeds_complete`` (the distinct seeds among those,
    sorted), plus ``seeds`` (one entry per completed trial, not deduplicated), ``values``,
    ``mean``, ``std`` (the sample standard deviation, ``None`` under two values), ``min`` and
    ``max`` over ``values``.
    """
    import statistics

    from tcip_mcp.pipelines.training.hpo import SPLIT_DRAW_SEED_KEY

    planned = set(planned_seeds)
    groups: dict[str, dict] = {}
    for row in trials:
        point = {k: v for k, v in (row.trial_params or {}).items() if k != SPLIT_DRAW_SEED_KEY}
        key = json.dumps(point, sort_keys=True, default=str)
        groups.setdefault(key, {"point": point, "rows": []})["rows"].append(row)

    out: list[dict] = []
    for entry in groups.values():
        rows = entry["rows"]
        complete = [r for r in rows if _completed_with_value(r)]
        values = [float(r.best_metric) for r in complete]
        seeds = [(r.trial_params or {}).get(SPLIT_DRAW_SEED_KEY) for r in complete]
        seeds_complete = sorted(set(seeds), key=lambda s: (s is None, s))
        block = {
            "seeds": seeds,
            "values": values,
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "n": len(rows),
            "n_complete": len(values),
            "seeds_complete": seeds_complete,
        }
        eligible = len(complete) == len(rows) and planned <= set(seeds_complete)
        out.append({"point": entry["point"], "block": block, "eligible": eligible})
    return out


def sweep_outcome(trials: list[RunRow], sweep: dict) -> dict[str, Any]:
    """What a sweep's trial rows amount to under the objective its record ``sweep`` states:
    ``best_params`` and ``best_value`` over its completed trials in the objective's direction;
    above one draw, the best point by mean over its draws (:func:`group_split_draws`), with
    ``best_value_spread`` and every point's ``split_sensitivity``. ``best_params`` is ``None``
    when no trial or point qualifies."""
    choose = max if sweep["objective"]["higher_is_better"] else min
    if sweep["input"]["split_draws"] > 1:
        groups = group_split_draws(trials, sweep["input"]["split_draw_seeds"])
        eligible = [g for g in groups if g["eligible"]]
        best_group = choose(eligible, key=lambda g: g["block"]["mean"]) if eligible else None
        return {
            "best_params": best_group["point"] if best_group else None,
            **stored_number("best_value", best_group["block"]["mean"] if best_group else None),
            "best_value_spread": best_group["block"] if best_group else None,
            "split_sensitivity": groups,
        }
    done = [t for t in trials if _completed_with_value(t)]
    best_trial = choose(done, key=lambda t: cast(float, t.best_metric)) if done else None
    return {"best_params": best_trial.trial_params if best_trial else None,
            **stored_number("best_value", best_trial.best_metric if best_trial else None)}


class SweepGroup(BaseModel):
    """One sweep as every listing and monitor shows it: its record's own fields, its trial rows
    and what they amount to."""

    sweep_id: str
    state: str
    status_error: str | None
    input: dict[str, Any]
    objective: dict[str, Any]
    cancel_requested: bool
    heartbeat: str
    trials: list[RunRow]
    outcome: dict[str, Any]


def read_sweep(directory: Path, rows: list[RunRow]) -> SweepGroup:
    """The sweep at ``directory`` as its :class:`SweepGroup`, its trials the ``rows`` of the
    project's runs (:func:`run_rows`) it holds, and their outcome (:func:`sweep_outcome`)."""
    sweep = observe(directory)
    trials = [row for row in rows if row.sweep == directory.name]
    return SweepGroup(
        sweep_id=directory.name, state=sweep.state, status_error=sweep.status_error,
        input=sweep.record["input"], objective=sweep.record["objective"],
        cancel_requested=cancel_requested(directory), heartbeat=sweep.heartbeat,
        trials=trials, outcome=sweep_outcome(trials, sweep.record),
    )


class TrainingDetail(BaseModel):
    """One run, trial or sweep: a run's row or a sweep's group, the other ``None``, and the URL a
    TensorBoard serves its board at, ``None`` while none does."""

    run: RunRow | None
    sweep: SweepGroup | None
    tensorboard_url: str | None


class TrainingListing(BaseModel):
    """Every run of a project launched on its own as its row, and every sweep as its group."""

    runs: list[RunRow]
    sweeps: list[SweepGroup]


def training_listing(project: Path | str) -> TrainingListing:
    """``project``'s runs (:func:`run_rows`) launched on their own, and its sweeps
    (:func:`read_sweep`) with their trial rows under them."""
    rows = run_rows(project)
    return TrainingListing(runs=[row for row in rows if row.sweep is None],
                           sweeps=[read_sweep(d, rows) for d in sweep_dirs(project)])


def launch_declarations(project: Path | str) -> dict[str, dict[str, str | None]]:
    """The agent identity each run's ``launch_training`` line in ``project``'s audit log carries,
    by the experiment id it names: empty for a launch no agent declared itself to."""
    from tcip_mcp import agent_identity
    from tcip_mcp.audit import acts_of

    return {entry["arguments"]["experiment_id"]:
            {field: entry[field] for field in agent_identity.RECORD_FIELDS if field in entry}
            for entry in acts_of(project, ("launch_training",))[0]}


def get_experiment(
    experiment_id: str, *, project: Path | str, metrics_limit: int | None = None,
    metrics_offset: int = 0,
) -> dict[str, Any]:
    """One run's directory read whole: its launch record, its final status (``None`` until
    written), its state, and its metrics as one row per epoch (:func:`epoch_rows`) paginated by
    ``metrics_offset`` and ``metrics_limit``, ``n_epochs`` the number paged over.
    ``{"error": ...}`` for an id naming no run."""
    observation = find_observation(experiment_id, project=project)
    if observation is None:
        return {"error": f"Experiment not found: {experiment_id}"}
    rows = epoch_rows(read_rows(observation.metrics_log)[0])
    end = (metrics_offset + metrics_limit) if metrics_limit is not None else None
    return {
        "experiment_id": experiment_id,
        "run": observation.record,
        "final_status": observation.final,
        "state": observation.state,
        "n_epochs": len(rows),
        "metrics": rows[metrics_offset:end],
        "metrics_offset": metrics_offset,
    }


def _split_summary(observation: RunObservation) -> dict[str, Any]:
    """The partition column of a comparison of the observed run: ``{"case": "bound",
    "selection_dir", "seed", "redraw"}`` for a run bound to a selection, ``{"case": "spatial"}``
    for a within-image split, ``{"case": "drawn", "seed"}`` for a drawn one, and ``{"case":
    "none"}`` for a run that resolved to nothing."""
    resolved, data = observation.resolution, observation.resolved_data
    if resolved is None or data is None:
        return {"case": "none"}
    partition = resolved["partition"]
    binding = partition["selection"]
    if binding is not None:
        return {"case": "bound", "selection_dir": binding["selection_dir"],
                "seed": partition["seed"], "redraw": binding["redraw"]}
    from tcip_mcp.pipelines.data.split_construction import partition_spatial

    if partition_spatial(partition) is not None:
        return {"case": "spatial"}
    return {"case": "drawn", "seed": partition["seed"]}


def _row_instant(row: dict, experiment_id: str) -> datetime:
    """A metric row's own ``timestamp`` as an instant. Refuses (``ValueError``) a row whose
    timestamp is absent or does not decode, naming the row and the run."""
    try:
        return datetime.fromisoformat(row[TIMESTAMP_KEY])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"metric row {row} of {experiment_id} carries no decodable timestamp "
                         f"({exc!r}); the run's metrics log cannot be compared") from exc


def compare_experiments(experiment_ids: list[str], *, project: Path | str) -> dict[str, Any]:
    """Side-by-side comparison of training runs.

    Per run: ``state``, ``n_epochs``, ``last_logged_metrics`` (the last epoch's row,
    :func:`epoch_rows`, not a verified result), ``rows_after_end`` (rows whose own
    ``timestamp`` is a later instant than
    the final status's ``ended``; ``None`` before a final status), the final status's
    ``status_error``, ``registry`` (a completed run's own registry entry,
    ``model_registry.run_entry``, with the metrics and source its checkpoint carries,
    ``model_registry.entry_facts``, as a one-entry list, empty otherwise), ``split``
    (:func:`_split_summary`), and the builder, task, subject and dataset identity its records
    carry. An id naming no run is an entry carrying only ``error``.
    ``same_dataset_fingerprint`` is ``None`` when any entry is an error or any fingerprint is
    unset, else whether every run names one fingerprint.
    """
    from tcip_mcp.model_registry import entry_facts, run_entry

    comparisons: list[dict[str, Any]] = []
    for eid in experiment_ids:
        observation = find_observation(eid, project=project)
        if observation is None:
            comparisons.append({"experiment_id": eid, "error": f"Experiment not found: {eid}"})
            continue
        run, final, spec = observation.record, observation.final, observation.spec
        metrics = read_rows(observation.metrics_log)[0]
        epochs = epoch_rows(metrics)
        summary: dict[str, Any] = {
            "experiment_id": eid, "state": observation.state, "n_epochs": len(epochs),
        }
        if epochs:
            summary["last_logged_metrics"] = epochs[-1]
        try:
            instants = [_row_instant(row, eid) for row in metrics]
        except ValueError as exc:
            comparisons.append({"experiment_id": eid, "error": str(exc)})
            continue
        rows_after_end = None if final is None else sum(
            1 for at in instants if at > datetime.fromisoformat(final["ended"]))
        summary["rows_after_end"] = rows_after_end
        summary["status_error"] = observation.status_error
        entry = run_entry(observation)
        summary["registry"] = [] if entry is None else [{
            "name": entry["name"], "registered_at": entry["registered_at"],
            **entry_facts(entry),
        }]
        data = observation.resolved_data
        summary["split"] = _split_summary(observation)
        summary["model"] = spec.model_source.builder
        summary["task"] = spec.model_source.task
        summary["subject"] = data.recorded_scope.subject if data is not None else None
        summary["dataset_id"] = run["dataset"]["id"]
        summary["dataset_fingerprint"] = run["dataset"]["fingerprint"]
        comparisons.append(summary)

    any_error = any("error" in c for c in comparisons)
    fps = {c.get("dataset_fingerprint") for c in comparisons if "error" not in c}
    same_dataset = None if (any_error or not fps or None in fps) else len(fps) == 1
    return {"experiments": comparisons, "count": len(comparisons),
            "same_dataset_fingerprint": same_dataset}


def get_experiment_lineage(experiment_id: str, *, project: Path | str) -> dict[str, Any]:
    """A training run's data-to-model chain read off its records: the data section it resolved,
    its dataset identity, the run it was relaunched from and the checkpoint it resumed from, and
    the checkpoint its completion selected. ``{"error": ...}`` for an id naming no training
    run."""
    observation = find_observation(experiment_id, project=project)
    if observation is None:
        return {"error": f"Experiment not found: {experiment_id}"}
    run, data = observation.record, observation.resolved_data
    return {"experiment_id": experiment_id, "lineage": {
        "data": data.record() if data is not None else None,
        "dataset_id": run["dataset"]["id"],
        "dataset_fingerprint": run["dataset"]["fingerprint"],
        "relaunched_from": run["relaunched_from"],
        "resume_from": run["resume_from"],
        "checkpoint": observation.checkpoint,
    }}
