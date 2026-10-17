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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path, PureWindowsPath
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Literal,
    NamedTuple,
    cast,
    get_args,
)

from pydantic import BaseModel

from tcip_mcp.audit import now_iso
from tcip_mcp.pipelines.data.selection import SAMPLE_PATHS
from tcip_mcp.pipelines.schemas import TraitRevisionRef, selection_of
from tcip_mcp.registry_paths import (
    PathFields,
    recorded_paths,
    runtime_paths,
    within,
)
from tcip_store import (
    BadKeyError,
    DecodeError,
    decode_value,
    encode_log_line,
    encode_record,
    finite_number,
    stored_number,
)
from tcip_store.file_backend import is_sharing_violation, retry_while_denied
from tcip_store.values import NOT_FINITE_SUFFIX

if TYPE_CHECKING:
    from tcip_mcp.pipelines.schemas import (
        DataSpec,
        SpatialManifest,
        TrainConfigSchema,
        TrainGeometry,
    )

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
    ("images_dir",),
    ("labels_dir",),
    ("plant_csv_paths", "[]"),
)
"""The fields of a run's data section that name where its samples are."""

_DATA_BLOCK_PATHS: PathFields = (
    *DATA_PATHS,
    ("split", "selection_dir"),
    ("dataset_source", "source_files", "[]"),
)
"""Every field of a run's data section that names a path: :data:`DATA_PATHS`, the selection a
bound split names and the source files its dataset builder imports."""

CONFIG_PATHS: PathFields = (
    ("model_source", "source_files", "[]"),
    ("model_source", "normalization", "source", "windows", "[]", "0"),
    *within(("data",), _DATA_BLOCK_PATHS),
)
"""Every field of a run's config that names a path."""

_PARAMETER_PATHS: PathFields = tuple(
    (".".join(field[:depth]), *field[depth:])
    for field in CONFIG_PATHS
    for depth in range(
        1, (field.index("[]") if "[]" in field else len(field)) + 1
    )
)
"""The same fields in a flat map of sweep parameters, whose keys are dotted config paths: each
field under every key a parameter may name it by, the mapping holding it or the field itself."""

_SPACE_PATHS: PathFields = tuple(
    (name, "choices", "[]", *rest) for name, *rest in _PARAMETER_PATHS
)

RECORD_PATHS: dict[str, PathFields] = {
    RUN_FILE: (
        *within(("config",), CONFIG_PATHS),
        *within(("resolved", "data"), _DATA_BLOCK_PATHS),
        *within(("resolved", "partition", "samples", "[]"), SAMPLE_PATHS),
        ("resolved", "partition", "selection", "selection_dir"),
        ("resume_from",),
        *within(("trial_params",), _PARAMETER_PATHS),
    ),
    SWEEP_FILE: (
        *within(("input", "base_config"), CONFIG_PATHS),
        *within(("input", "baseline_params"), _PARAMETER_PATHS),
        *within(("input", "param_space"), _SPACE_PATHS),
    ),
    FINAL_STATUS_FILE: (("checkpoint", "path"),),
}
"""The fields of each record of a run or sweep directory that name a path, each stored against
the project the directory lies under (:func:`project_of_run`)."""

FinalState = Literal["completed", "failed", "canceled"]
"""The states a final status names."""
FINAL_STATES: tuple[FinalState, ...] = get_args(FinalState)
TERMINAL_STATES = frozenset({*FINAL_STATES, "interrupted"})
"""Every state a run or sweep ends in: a final status's, and ``interrupted``, the state a directory
with no final status reads as once its heartbeat is stale."""

HEARTBEAT_STALE_SECONDS = float(
    os.environ.get("TCIP_HEARTBEAT_STALE_SECONDS", "600")
)
"""How long a directory with no final status reads ``running`` after its last sign of life,
from ``$TCIP_HEARTBEAT_STALE_SECONDS``."""


def experiments_dir(project: Path | str) -> Path:
    """Return the directory every run and sweep directory of ``project`` sits in."""
    return Path(project) / EXPERIMENTS_DIR


def project_of_run(run_dir: Path) -> Path:
    """Return the project a run, sweep or trial directory lies under.

    The project is the parent of the ``.tcip`` directory on its path. Refuses (``ValueError``) a
    directory under no ``.tcip``.
    """
    for parent in Path(run_dir).parents:
        if parent.name == ".tcip":
            return parent.parent
    raise ValueError(f"{run_dir} lies under no project's .tcip directory")


def run_name(name: str) -> str:
    """Return ``name`` once it is known to be one directory name.

    A separator, a drive, an empty name or a dot name refuses with ``BadKeyError``.
    """
    if not name or name in (".", "..") or PureWindowsPath(name).name != name:
        raise BadKeyError(
            f"{name!r} is not a single directory name: a name carrying a separator, a drive or a "
            "parent reference would address a directory outside the one it names."
        )
    return name


def experiment_dir(experiment_id: str, *, project: Path | str) -> Path:
    """Return the directory of the run or sweep ``experiment_id`` names.

    Refuses an id that is not a single directory name (:func:`run_name`).
    """
    return experiments_dir(project) / run_name(experiment_id)


def find_run(experiment_id: str, *, project: Path | str) -> Path | None:
    """Return the run directory of ``project`` named ``experiment_id``, or ``None``.

    The runs are :func:`run_dirs`'s, a sweep's trial included.
    """
    return next(
        (d for d in run_dirs(project) if d.name == experiment_id), None
    )


def find_sweep(sweep_id: str, *, project: Path | str) -> Path | None:
    """Return the sweep directory of ``project`` named ``sweep_id``, or ``None``.

    The sweeps are :func:`sweep_dirs`'s.
    """
    return next((d for d in sweep_dirs(project) if d.name == sweep_id), None)


def named_directory(name: str, *, project: Path | str) -> Path | None:
    """Return the run, trial or sweep directory of ``project`` named ``name``, or ``None``."""
    return next(
        (
            d
            for d in (*run_dirs(project), *sweep_dirs(project))
            if d.name == name
        ),
        None,
    )


def board_of(directory: Path) -> Path:
    """Return the TensorBoard log directory of a run or sweep directory.

    That is a sweep's own directory, every trial's events beneath it, or a run's
    :data:`TENSORBOARD_DIR`.
    """
    return (
        directory
        if (directory / SWEEP_FILE).is_file()
        else directory / TENSORBOARD_DIR
    )


def mint_experiment_id(prefix: str = "run") -> str:
    """Return a fresh run id: ``<prefix>_<epoch-seconds>_<6 hex chars>``."""
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:6]}"


# ── the files ────────────────────────────────────────────────────────────────


class RunDirectoryExistsError(FileExistsError):
    """A launch named a run or sweep directory that already exists."""


def create_run_directory(directory: Path) -> Path:
    """Create ``directory`` and its parents, and return it.

    One that already exists refuses with :class:`RunDirectoryExistsError`: runs and sweeps share
    one parent, so this creation is the name's reservation.
    """
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise RunDirectoryExistsError(
            f"{directory} already exists: runs, trials and sweeps share one namespace, so a new "
            "one takes a new name."
        ) from None
    return directory


def open_run_directory(
    run_dir: Path, record: dict, copies: Mapping[str, bytes]
) -> None:
    """Create the run directory ``run_dir`` with its snapshot and metrics log, then its record.

    The directory (:func:`create_run_directory`) holds its source snapshot's ``copies`` laid out
    (``model_build.lay_out``) and its empty metrics log; ``record`` is then written as its
    ``run.json`` once.
    """
    from tcip_mcp.pipelines.model_build import lay_out

    create_run_directory(run_dir)
    lay_out(run_dir, copies)
    (run_dir / METRICS_FILE).touch(exist_ok=False)
    write_record(run_dir / RUN_FILE, record)


def write_record(path: Path, value: Any) -> None:
    """Publish the run or sweep record ``value`` at ``path`` once (:func:`write_once`).

    Each path :data:`RECORD_PATHS` names for its file is stored against the directory's project.
    """
    write_once(
        path,
        recorded_paths(value, RECORD_PATHS[path.name], project_of_run(path)),
    )


def read_run_record(path: Path) -> Any:
    """Return the run or sweep record at ``path`` (:func:`read_record`).

    Each path :data:`RECORD_PATHS` names for its file is resolved against the directory's
    project.
    """
    return runtime_paths(
        read_record(path), RECORD_PATHS[path.name], project_of_run(path)
    )


def publish_once(path: Path, write: Callable[[BinaryIO], object]) -> None:
    """Write a file's whole content through ``write`` and publish it under ``path``, replacing none.

    The content goes into a staging file beside ``path`` first, so a file under its final name is
    always whole. A name already published refuses with ``FileExistsError``; a write that fails
    publishes nothing.
    """
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
    """Publish ``value`` as the JSON record at ``path`` (:func:`publish_once`).

    Refuses a value JSON cannot hold before anything is created, and a path already written with
    ``FileExistsError``.
    """
    data = encode_record(value)
    publish_once(path, lambda handle: handle.write(data))


def write_final_status(
    directory: Path, state: str, status_error: str | None, **outcome: Any
) -> None:
    """Write the final status of the run or sweep at ``directory`` once.

    The status holds ``state``, the instant, ``status_error``, the error behind it, and
    ``outcome``, what it produced (a run's ``checkpoint``). Refuses (``ValueError``) a state
    outside :data:`FINAL_STATES`, and a status already written with ``FileExistsError``.
    """
    if state not in FINAL_STATES:
        raise ValueError(
            f"{state!r} is not a final state; a final status names one of "
            f"{list(FINAL_STATES)}."
        )
    write_record(
        directory / FINAL_STATUS_FILE,
        {
            "state": state,
            "ended": now_iso(),
            "status_error": status_error,
            **outcome,
        },
    )


def append_row(path: Path, row: dict) -> None:
    """Append ``row`` as one line of the JSON log at ``path``.

    Refuses a row JSON cannot hold and (``FileNotFoundError``) a log its directory was not opened
    with.
    """
    line = encode_log_line(row) + b"\n"
    with open(path, "r+b") as handle:
        handle.seek(0, os.SEEK_END)
        handle.write(line)


SHARING_RETRY_BUDGET_S = 0.35
"""Provisional, chosen rather than measured: how long a record read keeps retrying a sharing
violation (``tcip_store.file_backend.retry_while_denied``) before raising it."""


def read_record(path: Path) -> Any:
    """Return the JSON record at ``path``.

    A Windows sharing violation is retried while :data:`SHARING_RETRY_BUDGET_S` is unspent and
    raised once it is, no read starting after the deadline; every other ``PermissionError``, a
    missing file (``FileNotFoundError``) and bytes that do not decode (``DecodeError`` naming the
    file) raise at once.
    """
    data = retry_while_denied(
        path.read_bytes, SHARING_RETRY_BUDGET_S, denied=is_sharing_violation
    )
    try:
        return decode_value(data)
    except ValueError as exc:
        raise DecodeError(f"{path} does not decode as JSON: {exc}") from exc


def read_rows(path: Path, *, after: int = 0) -> tuple[list[dict], int]:
    """Return every complete row of the JSON log at ``path`` past byte offset ``after``.

    Beside them is the offset past the last complete one. A line still being appended is left for
    the next read. A log that does not exist raises ``FileNotFoundError``; a complete line that
    does not decode raises ``DecodeError`` naming the file.
    """
    with open(path, "rb") as handle:
        handle.seek(after)
        data = handle.read()
    end = data.rfind(b"\n") + 1
    rows: list[dict] = []
    for line in data[:end].splitlines():
        try:
            rows.append(decode_value(line))
        except ValueError as exc:
            raise DecodeError(
                f"{path} holds a row that does not decode: {exc}"
            ) from exc
    return rows, after + end


# ── liveness ─────────────────────────────────────────────────────────────────


def touch_heartbeat(directory: Path) -> None:
    """Mark ``directory``'s process alive now."""
    (directory / HEARTBEAT_FILE).touch()


def keep_heartbeat(directory: Path) -> threading.Event:
    """Touch ``directory``'s heartbeat now and on a daemon thread until the returned event is set.

    The thread touches it every tenth of :data:`HEARTBEAT_STALE_SECONDS`.
    """
    stop = threading.Event()
    touch_heartbeat(directory)

    def beat() -> None:
        while not stop.wait(HEARTBEAT_STALE_SECONDS / 10):
            try:
                touch_heartbeat(directory)
            except OSError:
                logger.warning(
                    "could not touch the heartbeat of %s",
                    directory,
                    exc_info=True,
                )

    threading.Thread(target=beat, daemon=True).start()
    return stop


def last_alive(directory: Path, record_file: str = RUN_FILE) -> float:
    """Return the latest sign of life of a run or sweep directory, as a POSIX time.

    That is its heartbeat's modification time, or its launch record's before the heartbeat was
    first touched.
    """
    heartbeat = directory / HEARTBEAT_FILE
    return (
        (heartbeat if heartbeat.exists() else directory / record_file)
        .stat()
        .st_mtime
    )


class RunEndedError(ValueError):
    """A write addressed a run or sweep whose final status is written."""


def require_open(directory: Path) -> None:
    """Refuse (:class:`RunEndedError`) the run or sweep at ``directory`` once it has ended.

    It has ended once its final status is written: from then on it takes no write.
    """
    if (directory / FINAL_STATUS_FILE).exists():
        raise RunEndedError(
            f"{directory.name} has ended: its final status is written, so it takes no "
            "more writes."
        )


def request_cancel(directory: Path) -> str:
    """Record a cancellation request of the run or sweep at ``directory``; return its first time.

    A later request leaves the first record as written. Refuses a directory that has ended
    (:func:`require_open`).
    """
    require_open(directory)
    try:
        write_once(directory / CANCEL_FILE, {"requested": now_iso()})
    except FileExistsError:
        pass
    return read_record(directory / CANCEL_FILE)["requested"]


def cancel_requested(directory: Path) -> bool:
    """Return whether a cancellation of the run or sweep at ``directory`` has been requested."""
    return (directory / CANCEL_FILE).exists()


# ── reading a run ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunObservation:
    """One run or sweep directory read once.

    It holds its launch record, its final status (``None`` while it has not ended), its latest
    sign of life as a POSIX time (:func:`last_alive`), the state those derive, and the record file
    it holds (``RUN_FILE`` or ``SWEEP_FILE``).
    """

    directory: Path
    record: dict
    final: dict | None
    alive: float
    state: str
    record_file: str

    @property
    def metrics_log(self) -> Path:
        """The run's metrics log, which exists from the moment the run opened."""
        return self.directory / METRICS_FILE

    @property
    def checkpoint(self) -> dict | None:
        """The checkpoint a completed training run's final status selects, or ``None``.

        ``None`` is a run that has not completed.
        """
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
        """What a run's input resolved to at launch, or ``None``.

        It holds the run's ``data``, ``geometry``, ``partition`` and ``trait_revision``; ``None``
        is a sweep trial whose sampled point was refused at its admission or failed to resolve.
        """
        return self.record["resolved"]

    @cached_property
    def spec(self) -> TrainConfigSchema:
        """A run's launch ``config`` validated once.

        The validation is ``schemas.train_config``, which refuses an invalid one.
        """
        from tcip_mcp.pipelines.schemas import train_config

        return train_config(self.record["config"])

    @cached_property
    def resolved_data(self) -> DataSpec | None:
        """The data block a run's launch resolved (:attr:`resolution`'s), validated once.

        ``None`` for a run that resolved to nothing.
        """
        from tcip_mcp.pipelines.schemas import DataSpec

        resolved = self.resolution
        return (
            None
            if resolved is None
            else DataSpec.model_validate(resolved["data"])
        )

    @cached_property
    def resolved_geometry(self) -> TrainGeometry | None:
        """The geometry a run's launch read off its training frames, validated once.

        It is :attr:`resolution`'s; ``None`` for a run that resolved to nothing.
        """
        from tcip_mcp.pipelines.schemas import TrainGeometry

        resolved = self.resolution
        return (
            None
            if resolved is None
            else TrainGeometry.model_validate(resolved["geometry"])
        )

    @cached_property
    def trait_revision(self) -> TraitRevisionRef | None:
        """The confirmed trait revision the record carries, validated once.

        It is a sweep's own, else what a run's launch resolved (:attr:`resolution`'s); ``None``
        for one naming no trait. Refuses (``ValueError``) naming the run when it resolved to
        nothing.
        """
        held = (
            self.record if self.record_file == SWEEP_FILE else self.resolution
        )
        if held is None:
            raise ValueError(
                f"run {self.directory.name} resolved to nothing and carries no trait revision"
            )
        ref = held["trait_revision"]
        return None if ref is None else TraitRevisionRef.model_validate(ref)

    @property
    def sweep(self) -> str | None:
        """The sweep a trial run belongs to, ``None`` for a run launched on its own."""
        parent = self.directory.parent
        return parent.name if (parent / SWEEP_FILE).is_file() else None


def observe(directory: Path) -> RunObservation:
    """Read the sweep (holding its ``sweep.json``) or run at ``directory`` once.

    Its state is the one its final status names once one is written; before that ``running``
    while its latest sign of life is within :data:`HEARTBEAT_STALE_SECONDS`, else
    ``interrupted``. A run holds its metrics log from the moment it opened; one missing it
    refuses with :class:`IncompleteRunError` naming it. A completed run's checkpoint is not read.
    """
    record_file = (
        SWEEP_FILE if (directory / SWEEP_FILE).is_file() else RUN_FILE
    )
    record = read_run_record(directory / record_file)
    final_path = directory / FINAL_STATUS_FILE
    final = read_run_record(final_path) if final_path.exists() else None
    alive = last_alive(directory, record_file)
    if final is not None:
        state = final["state"]
    else:
        state = (
            "running"
            if time.time() - alive <= HEARTBEAT_STALE_SECONDS
            else "interrupted"
        )
    observation = RunObservation(
        directory, record, final, alive, state, record_file
    )
    if record_file == RUN_FILE and not observation.metrics_log.is_file():
        raise IncompleteRunError(
            f"run {directory.name} ({state}) is missing "
            f"{observation.metrics_log}."
        )
    return observation


class IncompleteRunError(ValueError):
    """A run directory missing a file every opened run holds."""


def _holding(parent: Path, record_file: str) -> list[Path]:
    """Return every directory under ``parent`` holding ``record_file``, in name order."""
    return (
        sorted(d for d in parent.iterdir() if (d / record_file).is_file())
        if parent.is_dir()
        else []
    )


def sweep_dirs(project: Path | str) -> list[Path]:
    """Return every sweep directory of ``project`` that holds its input, in name order."""
    return _holding(experiments_dir(project), SWEEP_FILE)


def run_dirs(project: Path | str) -> list[Path]:
    """Return every run directory of ``project`` that holds a launch record, in name order.

    Each sweep's trials are included.
    """
    return sorted(
        [
            *_holding(experiments_dir(project), RUN_FILE),
            *(
                d
                for sweep in sweep_dirs(project)
                for d in _holding(sweep, RUN_FILE)
            ),
        ],
        key=lambda d: d.name,
    )


def run_observations(project: Path | str) -> list[RunObservation]:
    """Return every run of ``project`` (:func:`run_dirs`), each observed once."""
    return [observe(d) for d in run_dirs(project)]


def live_run_conflict(project: Path) -> str | None:
    """Return the refusal a still-running run, trial or sweep of ``project`` answers with, or None.

    The runs, trials and sweeps are :func:`training_listing`'s; ``None`` when none is running.
    """
    listing = training_listing(project)
    trials = [trial for sweep in listing.sweeps for trial in sweep.trials]
    running = [
        sweep.sweep_id for sweep in listing.sweeps if sweep.state == "running"
    ]
    running += [
        row.experiment_id
        for row in (*listing.runs, *trials)
        if row.state == "running"
    ]
    if not running:
        return None
    return (
        f"{running[0]!r} is running; ask the agent to cancel it (cancel_training) or wait for "
        "it to finish"
    )


def find_observation(
    experiment_id: str, *, project: Path | str
) -> RunObservation | None:
    """Return the run ``experiment_id`` names (:func:`find_run`), observed, or ``None``."""
    run_dir = find_run(experiment_id, project=project)
    return observe(run_dir) if run_dir is not None else None


class RunResolution(NamedTuple):
    """What a run resolved at launch.

    It holds its data block (``RunObservation.resolved_data``), its ``partition`` and the
    within-image split that partition records (``split_construction.partition_spatial``).
    """

    data: DataSpec
    partition: dict
    spatial: SpatialManifest | None


def run_resolution(
    experiment_id: str, *, project: Path | str
) -> RunResolution:
    """Return what the run ``experiment_id`` names resolved at launch (:func:`find_observation`).

    Refuses (``ValueError``) an id naming no run directory or one whose input resolved to
    nothing.
    """
    observation = find_observation(experiment_id, project=project)
    data = observation.resolved_data if observation is not None else None
    if observation is None or observation.resolution is None or data is None:
        raise ValueError(
            f"no run directory records what {experiment_id!r} resolved to."
        )
    from tcip_mcp.pipelines.data.split_construction import partition_spatial

    partition = observation.resolution["partition"]
    return RunResolution(data, partition, partition_spatial(partition))


def partition_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a run's metrics-log ``rows`` by kind, each in logged order.

    Returns its epoch rows, then its per-batch rows, those carrying :data:`STEP_KEY`.
    """
    epochs: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    for row in rows:
        (batches if STEP_KEY in row else epochs).append(row)
    return epochs, batches


def epoch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a run's metrics-log ``rows`` as one row per epoch, in first-logged order.

    The epoch rows (:func:`partition_rows`) of one epoch are merged, a later row's keys laid over
    the earlier's, a metric and its non-finite state companion as one value.
    """
    merged: dict[str, dict[str, Any]] = {}
    for row in partition_rows(rows)[0]:
        epoch = json.dumps(row.get(EPOCH_KEY), sort_keys=True, default=str)
        restated = {f"{key}{NOT_FINITE_SUFFIX}" for key in row}
        earlier = {
            k: v for k, v in merged.get(epoch, {}).items() if k not in restated
        }
        merged[epoch] = {**earlier, **row}
    return list(merged.values())


def best_selection(
    rows: list[dict[str, Any]], higher_is_better: bool
) -> float | None:
    """Return the best finite ``selection`` among a run's metrics-log epoch rows.

    The epoch rows are :func:`partition_rows`'s; the best is the highest when
    ``higher_is_better`` and else the lowest, ``None`` when no row carries one.
    """
    values = [
        float(row["selection"])
        for row in partition_rows(rows)[0]
        if finite_number(row.get("selection"))
    ]
    if not values:
        return None
    return max(values) if higher_is_better else min(values)


class RunRow(BaseModel):
    """One run's status and summary fields."""

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


def run_summary(
    observation: RunObservation,
    rows: list[dict[str, Any]],
    launch: dict[str, str | None] | None,
) -> RunRow:
    """Return one run's :class:`RunRow` over its metrics-log ``rows`` and its ``launch`` identity.

    ``rows`` are :func:`read_rows` of its ``metrics_log``, read as :func:`epoch_rows`; ``launch``
    is the agent identity its launch event declared (:func:`launch_declarations`, ``None`` for a
    run no launch event names). The row holds its record's own fields, its last logged epoch,
    and its best selection value (:func:`best_selection`) under its config's selection metric and
    direction (``schemas.selection_of``) with that metric's name, both ``None`` for a run that
    resolved to nothing.
    """
    record, spec = observation.record, observation.spec
    selection = (
        selection_of(spec.evaluation.selection_metric)
        if observation.resolution is not None
        else None
    )
    rows = epoch_rows(rows)
    return RunRow(
        experiment_id=observation.directory.name,
        state=observation.state,
        created=record["created"],
        relaunched_from=record["relaunched_from"],
        sweep=observation.sweep,
        trial_params=record["trial_params"],
        builder=spec.model_source.builder,
        task=spec.model_source.task,
        images_dir=spec.data.images_dir,
        current_epoch=rows[-1].get(EPOCH_KEY) if rows else None,
        best_metric=best_selection(rows, selection[1])
        if selection is not None
        else None,
        best_metric_name=selection[0] if selection is not None else None,
        output_dir=str(observation.directory),
        status_error=observation.status_error,
        heartbeat=observation.heartbeat,
        launch=launch,
    )


def run_rows(project: Path | str) -> list[RunRow]:
    """Return every run of ``project`` (:func:`run_observations`) as its summary row."""
    launches = launch_declarations(project)
    return [
        run_summary(
            obs,
            read_rows(obs.metrics_log)[0],
            launches.get(obs.directory.name),
        )
        for obs in run_observations(project)
    ]


def _completed_with_value(trial: RunRow) -> bool:
    """Return whether a sweep's trial completed reporting a value under the sweep's selection."""
    return trial.state == "completed" and trial.best_metric is not None


def group_split_draws(
    trials: list[RunRow], planned_seeds: list[int]
) -> list[dict]:
    """Group a sweep's trial rows by the point their draws share.

    Each group carries its ``split_draws`` block; the point is every param but
    ``hpo.SPLIT_DRAW_SEED_KEY``. ``planned_seeds`` names every seed the sweep asked for; a group
    is ``eligible`` for best only when every one of them completed for that point and the group
    holds no trial that did not. An empty list accepts any group whose trials all completed with
    a value, regardless of seed identity.

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
        point = {
            k: v
            for k, v in (row.trial_params or {}).items()
            if k != SPLIT_DRAW_SEED_KEY
        }
        key = json.dumps(point, sort_keys=True, default=str)
        groups.setdefault(key, {"point": point, "rows": []})["rows"].append(
            row
        )

    out: list[dict] = []
    for entry in groups.values():
        rows = entry["rows"]
        complete = [r for r in rows if _completed_with_value(r)]
        values = [float(r.best_metric) for r in complete]
        seeds = [
            (r.trial_params or {}).get(SPLIT_DRAW_SEED_KEY) for r in complete
        ]
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
        eligible = len(complete) == len(rows) and planned <= set(
            seeds_complete
        )
        out.append(
            {"point": entry["point"], "block": block, "eligible": eligible}
        )
    return out


def sweep_selection(given: dict[str, Any]) -> tuple[str, bool]:
    """Return the metric and direction a sweep whose recorded input is ``given`` ranks trials by.

    They are ``schemas.selection_of`` its base config's ``evaluation.selection_metric``, which no
    trial's point sweeps.
    """
    return selection_of(given["base_config"]["evaluation"]["selection_metric"])


def sweep_outcome(
    trials: list[RunRow], given: dict[str, Any]
) -> dict[str, Any]:
    """Return what a sweep's trial rows amount to under its recorded input ``given``.

    That is ``best_params`` and ``best_value`` over its completed trials in its selection's
    direction (:func:`sweep_selection`); above one draw, the best point by mean over its draws
    (:func:`group_split_draws`), with ``best_value_spread`` and every point's
    ``split_sensitivity``. ``best_params`` is ``None`` when no trial or point qualifies.
    ``best_at_range_end`` names each numeric axis of the stated space (``hpo.space_axes``) whose
    ``low`` or ``high`` the best params sit at, the end beside it: a best the search could not
    look past.
    """
    from tcip_mcp.pipelines.training.hpo import space_axes

    choose = max if sweep_selection(given)[1] else min
    if given["split_draws"] > 1:
        groups = group_split_draws(trials, given["split_draw_seeds"])
        eligible = [g for g in groups if g["eligible"]]
        best_group = (
            choose(eligible, key=lambda g: g["block"]["mean"])
            if eligible
            else None
        )
        best = {
            "best_params": best_group["point"] if best_group else None,
            **stored_number(
                "best_value",
                best_group["block"]["mean"] if best_group else None,
            ),
            "best_value_spread": best_group["block"] if best_group else None,
            "split_sensitivity": groups,
        }
    else:
        done = [t for t in trials if _completed_with_value(t)]
        best_trial = (
            choose(done, key=lambda t: cast(float, t.best_metric))
            if done
            else None
        )
        best = {
            "best_params": best_trial.trial_params if best_trial else None,
            **stored_number(
                "best_value", best_trial.best_metric if best_trial else None
            ),
        }
    params = best["best_params"] or {}
    return {
        **best,
        "best_at_range_end": {
            name: end
            for name, (ptype, ends) in space_axes(
                given["param_space"], grid=False
            ).items()
            if ptype != "categorical" and name in params
            for end, bound in zip(("low", "high"), ends)
            if params[name] == bound
        },
    }


class SweepGroup(BaseModel):
    """One sweep: its record's own fields, its trial rows and what they amount to."""

    sweep_id: str
    state: str
    status_error: str | None
    input: dict[str, Any]
    cancel_requested: bool
    heartbeat: str
    trials: list[RunRow]
    outcome: dict[str, Any]


def read_sweep(directory: Path, rows: list[RunRow]) -> SweepGroup:
    """Return the sweep at ``directory`` as its :class:`SweepGroup`.

    Its trials are the ``rows`` of the project's runs (:func:`run_rows`) it holds, with their
    outcome (:func:`sweep_outcome`).
    """
    sweep = observe(directory)
    trials = [row for row in rows if row.sweep == directory.name]
    return SweepGroup(
        sweep_id=directory.name,
        state=sweep.state,
        status_error=sweep.status_error,
        input=sweep.record["input"],
        cancel_requested=cancel_requested(directory),
        heartbeat=sweep.heartbeat,
        trials=trials,
        outcome=sweep_outcome(trials, sweep.record["input"]),
    )


class TrainingDetail(BaseModel):
    """One run, trial or sweep, with the URL a TensorBoard serves its board at.

    It holds a run's row or a sweep's group, the other ``None``; the URL is ``None`` while no
    TensorBoard serves it.
    """

    run: RunRow | None
    sweep: SweepGroup | None
    tensorboard_url: str | None


class TrainingListing(BaseModel):
    """Every run of a project launched on its own as its row, and every sweep as its group."""

    runs: list[RunRow]
    sweeps: list[SweepGroup]


def training_listing(project: Path | str) -> TrainingListing:
    """Return ``project``'s runs launched on their own, and its sweeps with their trial rows.

    The runs are :func:`run_rows`'s and the sweeps :func:`read_sweep`'s.
    """
    rows = run_rows(project)
    return TrainingListing(
        runs=[row for row in rows if row.sweep is None],
        sweeps=[read_sweep(d, rows) for d in sweep_dirs(project)],
    )


def launch_declarations(
    project: Path | str,
) -> dict[str, dict[str, str | None]]:
    """Return the agent identity each run's ``launch_training`` audit line carries, by run id.

    The lines are ``project``'s audit log's, keyed by the experiment id each names; an identity is
    empty for a launch no agent declared itself to.
    """
    from tcip_mcp import agent_identity
    from tcip_mcp.audit import acts_of

    return {
        entry["arguments"]["experiment_id"]: {
            field: entry[field]
            for field in agent_identity.RECORD_FIELDS
            if field in entry
        }
        for entry in acts_of(project, ("launch_training",))[0]
    }


def get_experiment(
    experiment_id: str,
    *,
    project: Path | str,
    metrics_limit: int | None = None,
    metrics_offset: int = 0,
) -> dict[str, Any]:
    """Read one run's directory whole.

    Returns its launch record, its final status (``None`` until written), its state, and its
    metrics as one row per epoch (:func:`epoch_rows`) paginated by ``metrics_offset`` and
    ``metrics_limit``, ``n_epochs`` the number paged over. ``{"error": ...}`` for an id naming no
    run.
    """
    observation = find_observation(experiment_id, project=project)
    if observation is None:
        return {"error": f"Experiment not found: {experiment_id}"}
    rows = epoch_rows(read_rows(observation.metrics_log)[0])
    end = (
        (metrics_offset + metrics_limit) if metrics_limit is not None else None
    )
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
    """Return the partition column of a comparison of the observed run.

    It is ``{"case": "bound", "selection_dir", "seed", "redraw"}`` for a run bound to a
    selection, ``{"case": "spatial"}`` for a within-image split, ``{"case": "drawn", "seed"}`` for
    a drawn one, and ``{"case": "none"}`` for a run that resolved to nothing.
    """
    resolved, data = observation.resolution, observation.resolved_data
    if resolved is None or data is None:
        return {"case": "none"}
    partition = resolved["partition"]
    binding = partition["selection"]
    if binding is not None:
        return {
            "case": "bound",
            "selection_dir": binding["selection_dir"],
            "seed": partition["seed"],
            "redraw": binding["redraw"],
        }
    from tcip_mcp.pipelines.data.split_construction import partition_spatial

    if partition_spatial(partition) is not None:
        return {"case": "spatial"}
    return {"case": "drawn", "seed": partition["seed"]}


def _row_instant(row: dict, experiment_id: str) -> datetime:
    """Return a metric row's own ``timestamp`` as an instant.

    Refuses (``ValueError``) a row whose timestamp is absent or does not decode, naming the row
    and the run.
    """
    try:
        return datetime.fromisoformat(row[TIMESTAMP_KEY])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"metric row {row} of {experiment_id} carries no decodable timestamp "
            f"({exc!r}); the run's metrics log cannot be compared"
        ) from exc


def compare_experiments(
    experiment_ids: list[str], *, project: Path | str
) -> dict[str, Any]:
    """Compare training runs side by side.

    Per run: ``state``, ``n_epochs``, ``last_logged_metrics`` (the last epoch's row,
    :func:`epoch_rows`, not a verified result), ``rows_after_end`` (rows whose own ``timestamp``
    is a later instant than the final status's ``ended``; ``None`` before a final status), the
    final status's ``status_error``, ``registry`` (a completed run's own registry entry,
    ``model_registry.run_entry``, with the metrics and source its checkpoint carries,
    ``model_registry.entry_facts``, as a one-entry list, empty otherwise), ``split``
    (:func:`_split_summary`), and the builder, task, subject and dataset identity its records
    carry. An id naming no run is an entry carrying ``experiment_id`` and ``error``.
    ``same_dataset_fingerprint`` is ``None`` when any entry is an error or any fingerprint is
    unset, else whether every run names one fingerprint.
    """
    from tcip_mcp.model_registry import entry_facts, run_entry

    comparisons: list[dict[str, Any]] = []
    for eid in experiment_ids:
        observation = find_observation(eid, project=project)
        if observation is None:
            comparisons.append(
                {"experiment_id": eid, "error": f"Experiment not found: {eid}"}
            )
            continue
        run, final, spec = (
            observation.record,
            observation.final,
            observation.spec,
        )
        metrics = read_rows(observation.metrics_log)[0]
        epochs = epoch_rows(metrics)
        summary: dict[str, Any] = {
            "experiment_id": eid,
            "state": observation.state,
            "n_epochs": len(epochs),
        }
        if epochs:
            summary["last_logged_metrics"] = epochs[-1]
        try:
            instants = [_row_instant(row, eid) for row in metrics]
        except ValueError as exc:
            comparisons.append({"experiment_id": eid, "error": str(exc)})
            continue
        rows_after_end = (
            None
            if final is None
            else sum(
                1
                for at in instants
                if at > datetime.fromisoformat(final["ended"])
            )
        )
        summary["rows_after_end"] = rows_after_end
        summary["status_error"] = observation.status_error
        entry = run_entry(observation)
        summary["registry"] = (
            []
            if entry is None
            else [
                {
                    "name": entry["name"],
                    "registered_at": entry["registered_at"],
                    **entry_facts(
                        entry, project_of_run(observation.directory)
                    ),
                }
            ]
        )
        data = observation.resolved_data
        summary["split"] = _split_summary(observation)
        summary["model"] = spec.model_source.builder
        summary["task"] = spec.model_source.task
        summary["subject"] = (
            data.recorded_scope.subject if data is not None else None
        )
        summary["dataset_id"] = run["dataset"]["id"]
        summary["dataset_fingerprint"] = run["dataset"]["fingerprint"]
        comparisons.append(summary)

    any_error = any("error" in c for c in comparisons)
    fps = {
        c.get("dataset_fingerprint") for c in comparisons if "error" not in c
    }
    same_dataset = (
        None if (any_error or not fps or None in fps) else len(fps) == 1
    )
    return {
        "experiments": comparisons,
        "count": len(comparisons),
        "same_dataset_fingerprint": same_dataset,
    }


def get_experiment_lineage(
    experiment_id: str, *, project: Path | str
) -> dict[str, Any]:
    """Return a training run's data-to-model chain read off its records.

    The chain is the data section it resolved, its dataset identity, the run it was relaunched
    from and the checkpoint it resumed from, and the checkpoint its completion selected.
    ``{"error": ...}`` for an id naming no training run.
    """
    observation = find_observation(experiment_id, project=project)
    if observation is None:
        return {"error": f"Experiment not found: {experiment_id}"}
    run, data = observation.record, observation.resolved_data
    return {
        "experiment_id": experiment_id,
        "lineage": {
            "data": data.record() if data is not None else None,
            "dataset_id": run["dataset"]["id"],
            "dataset_fingerprint": run["dataset"]["fingerprint"],
            "relaunched_from": run["relaunched_from"],
            "resume_from": run["resume_from"],
            "checkpoint": observation.checkpoint,
        },
    }
