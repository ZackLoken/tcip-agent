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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Literal,
    cast,
    get_args,
)

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    computed_field,
)

from tcip_mcp.agent_identity import RecordedIdentity
from tcip_mcp.audit import now
from tcip_mcp.model_registry import Listed, metric_keys
from tcip_mcp.pipelines.data.selection import SAMPLE_PATHS
from tcip_mcp.pipelines.data.split_construction import Partition
from tcip_mcp.pipelines.model_build import Environment, SourceSnapshot
from tcip_mcp.pipelines.model_contract import LaunchContract
from tcip_mcp.pipelines.schemas import (
    DataSpec,
    EvaluationSpec,
    TrainGeometry,
    TraitRevisionRef,
    declared_higher_is_better,
    selection_of,
)
from tcip_mcp.pipelines.training.tensorboard_manager import TensorboardServing
from tcip_mcp.registry_paths import (
    PathFields,
    recorded_paths,
    runtime_paths,
    within,
)
from tcip_mcp.workspace import path_segment
from tcip_store import (
    DecodeError,
    FiniteJsonValue,
    decode_value,
    encode_log_line,
    encode_record,
    finite_number,
    finite_or_none,
    non_finite_state,
    stored_numbers,
)
from tcip_store.file_backend import is_sharing_violation, retry_while_denied
from tcip_store.values import NOT_FINITE_SUFFIX

if TYPE_CHECKING:
    from tcip_mcp.pipelines.schemas import TrainConfigSchema

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


class MetricsRow(BaseModel):
    """One metrics-log row: the stamps its writer lays beside its logged values.

    Every row carries its ``epoch`` and the ``timestamp`` it landed at; a per-batch row also its
    training ``step``, ``None`` on an epoch row. Every logged value is a finite JSON value.
    """

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, FiniteJsonValue] = Field(init=False)

    epoch: int
    timestamp: AwareDatetime
    step: int | None = None


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


def experiment_dir(experiment_id: str, *, project: Path | str) -> Path:
    """Return the directory of the run or sweep ``experiment_id`` names.

    Refuses an id that is no :data:`~tcip_mcp.workspace.PathSegment`.
    """
    return experiments_dir(project) / path_segment(experiment_id)


def named_directory(name: str, among: Iterable[Path]) -> Path | None:
    """Return the directory of ``among`` named ``name``, or ``None``.

    ``among`` is :func:`run_dirs` for a run (a sweep's trial included), :func:`sweep_dirs` for a
    sweep, or :func:`experiment_dirs` for either.
    """
    return next((d for d in among if d.name == name), None)


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


class DatasetIdentity(BaseModel):
    """The dataset a run's data block names: its id and its fingerprint, each ``None`` unset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str | None
    fingerprint: str | None


class ResolvedInput(BaseModel):
    """What a run's input resolved to at launch, as ``split_construction``'s resolution records it.

    That is its data block, the geometry read off its training frames, its ``partition`` (the
    resolution's own record of its samples and how they were drawn,
    ``split_construction.Partition``) and the confirmed trait revision it trains for, ``None``
    for none.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    data: DataSpec
    geometry: TrainGeometry
    partition: Partition
    trait_revision: TraitRevisionRef | None


class RunRecord(BaseModel):
    """A run's ``run.json``: its input and what that input resolved to, written once at launch.

    It holds the instant it was created, its launch ``config`` as the config records itself,
    ``resolved`` (``None`` for a sweep trial whose sampled point was refused at its admission or
    failed to resolve), the ``environment`` it ran in, the ``dataset`` its data block names, the
    ``source`` snapshot record its admission took, the run it was relaunched from and the
    checkpoint it resumes from, its wall clock, the model contract preflight proved, and a
    sweep trial's sampled point.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", ser_json_inf_nan="constants"
    )

    created: AwareDatetime
    config: dict[str, JsonValue]
    resolved: ResolvedInput | None
    environment: Environment
    dataset: DatasetIdentity
    source: SourceSnapshot
    relaunched_from: str | None
    resume_from: str | None
    max_wall_clock_seconds: float | None
    model_contract: LaunchContract | None
    trial_params: dict[str, JsonValue] | None

    @cached_property
    def spec(self) -> TrainConfigSchema:
        """The launch ``config`` validated once; ``schemas.train_config`` refuses an invalid one."""
        from tcip_mcp.pipelines.schemas import train_config

        return train_config(self.config)

    @property
    def resolution(self) -> ResolvedInput:
        """What its launch resolved to.

        Refuses (``ValueError``) a run that resolved to nothing, a sweep trial refused at its
        admission or whose resolution failed, which never ran.
        """
        if self.resolved is None:
            raise ValueError(
                "this run resolved to nothing: its input was refused at admission or failed "
                "to resolve, so it never ran"
            )
        return self.resolved


class SweepInput(BaseModel):
    """What a sweep was asked for: its search, its trial budget and the config its trials vary.

    ``base_config`` is the base training config each trial modifies and ``param_space`` the
    space its points are drawn from (``hpo.space_axes``); ``split_draw_seeds`` names the seeds
    each point is drawn under, one draw per seed, ``None`` for a single draw.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_trials: int
    search_alg: str
    scheduler: dict[str, JsonValue]
    max_concurrent: int
    baseline_params: dict[str, JsonValue] | None
    resources_per_trial: dict[str, JsonValue] | None
    param_space: dict[str, JsonValue]
    base_config: dict[str, JsonValue]
    relaunched_from: str | None
    split_draw_seeds: list[int] | None
    search_seed: int
    trial_budget: int | None

    @property
    def split_draws(self) -> int:
        """How many partitions the sweep draws: one per recorded seed, one for none."""
        return (
            1 if self.split_draw_seeds is None else len(self.split_draw_seeds)
        )


class SweepRecord(BaseModel):
    """A sweep's ``sweep.json``, written once before its first trial.

    It holds the instant it was created, the confirmed trait revision its trials train for
    (``None`` for none) and its :class:`SweepInput`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    created: AwareDatetime
    trait_revision: TraitRevisionRef | None
    input: SweepInput


def open_run_directory(
    run_dir: Path, record: RunRecord, copies: Mapping[str, bytes]
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


def write_record(
    path: Path, record: RunRecord | SweepRecord | FinalStatus
) -> None:
    """Publish ``record`` at ``path`` once (:func:`write_once`), as its model dumps it.

    Each path :data:`RECORD_PATHS` names for its file is stored against the directory's project.
    """
    write_once(
        path,
        recorded_paths(
            record.model_dump(mode="json", exclude_unset=True),
            RECORD_PATHS[path.name],
            project_of_run(path),
        ),
    )


def read_run_record(path: Path) -> JsonValue:
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


def write_once(path: Path, value: JsonValue) -> None:
    """Publish ``value`` as the JSON record at ``path`` (:func:`publish_once`).

    Refuses a value JSON cannot hold before anything is created, and a path already written with
    ``FileExistsError``.
    """
    data = encode_record(value)
    publish_once(path, lambda handle: handle.write(data))


class RunCheckpoint(BaseModel):
    """The checkpoint a completed run selected: its path and the sha256 of its bytes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    sha256: str


class FinalStatus(BaseModel):
    """How a run or sweep ended: its state, the instant, the error behind it, and its checkpoint.

    ``checkpoint`` is ``None`` for a sweep and for a run that selected none.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: FinalState
    ended: AwareDatetime
    status_error: str | None
    checkpoint: RunCheckpoint | None


def write_final_status(
    directory: Path,
    state: FinalState,
    status_error: str | None,
    *,
    ended: datetime,
    checkpoint: RunCheckpoint | None = None,
) -> None:
    """Write the final status of the run or sweep at ``directory`` once, a :class:`FinalStatus`.

    The status holds ``state``, the instant it ``ended``, ``status_error``, the error behind it,
    and the ``checkpoint`` a run produced. Refuses a status already written with
    ``FileExistsError``.
    """
    status = FinalStatus(
        state=state, ended=ended, status_error=status_error, checkpoint=checkpoint
    )
    write_record(directory / FINAL_STATUS_FILE, status)


def metrics_row(metrics: dict, epoch: int, step: int | None) -> MetricsRow:
    """Return ``metrics`` as the metrics log row of ``epoch`` and ``step``, stamped now.

    ``step`` is ``None`` on an epoch row. The metrics are in their stored form
    (``stored_numbers``). Refuses (``ValueError``) metrics carrying a key the row stamps (a
    :class:`MetricsRow` field) and (``pydantic.ValidationError``) a value that is not finite JSON,
    naming its position.
    """
    stamped = sorted(MetricsRow.model_fields.keys() & metrics.keys())
    if stamped:
        raise ValueError(
            f"the metrics carry {stamped}, which the metrics log stamps on each row itself; "
            "pass the epoch and step as the sink's own arguments."
        )
    return MetricsRow(
        epoch=epoch, timestamp=now(), step=step, **stored_numbers(metrics)
    )


def append_row(path: Path, row: MetricsRow) -> None:
    """Append ``row`` as one line of the metrics log at ``path``.

    Refuses (``FileNotFoundError``) a log its directory was not opened with.
    """
    line = encode_log_line(row.model_dump(mode="json")) + b"\n"
    with open(path, "r+b") as handle:
        handle.seek(0, os.SEEK_END)
        handle.write(line)


SHARING_RETRY_BUDGET_S = 0.35
"""Provisional, chosen rather than measured: how long a record read keeps retrying a sharing
violation (``tcip_store.file_backend.retry_while_denied``) before raising it."""


def read_record(path: Path) -> JsonValue:
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


def read_rows(path: Path, *, after: int = 0) -> tuple[list[MetricsRow], int]:
    """Return every complete row of the metrics log at ``path`` past byte offset ``after``.

    Beside them is the offset past the last complete one. A line still being appended is left for
    the next read. A log that does not exist raises ``FileNotFoundError``; a complete line that
    does not decode as a :class:`MetricsRow` raises ``DecodeError`` naming the file.
    """
    with open(path, "rb") as handle:
        handle.seek(after)
        data = handle.read()
    end = data.rfind(b"\n") + 1
    rows: list[MetricsRow] = []
    for line in data[:end].splitlines():
        try:
            rows.append(MetricsRow.model_validate(decode_value(line)))
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


def request_cancel(directory: Path) -> datetime:
    """Record a cancellation request of the run or sweep at ``directory``; return its first time.

    A later request leaves the first record as written. Refuses a directory that has ended
    (:func:`require_open`).
    """
    require_open(directory)
    try:
        write_once(
            directory / CANCEL_FILE,
            CancelRequest(requested=now()).model_dump(mode="json"),
        )
    except FileExistsError:
        pass
    return cancel_requested_at(directory)


class CancelRequest(BaseModel):
    """The first cancellation request of a run or sweep: the instant it was made."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requested: AwareDatetime


def cancel_requested_at(directory: Path) -> datetime:
    """Return when the cancellation of the run or sweep at ``directory`` was first requested.

    The record is a :class:`CancelRequest`; none written raises ``FileNotFoundError``.
    """
    return CancelRequest.model_validate(
        read_record(directory / CANCEL_FILE)
    ).requested


def cancel_requested(directory: Path) -> bool:
    """Return whether a cancellation of the run or sweep at ``directory`` has been requested."""
    return (directory / CANCEL_FILE).exists()


# ── reading a run ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunObservation[R: RunRecord | SweepRecord]:
    """One run or sweep directory read once.

    It holds its launch record (a :class:`RunRecord` or a :class:`SweepRecord`), its final status
    (``None`` while it has not ended), its latest sign of life as a POSIX time
    (:func:`last_alive`) and the state those derive.
    """

    directory: Path
    record: R
    final: FinalStatus | None
    alive: float
    state: str

    @property
    def metrics_log(self) -> Path:
        """The run's metrics log, which exists from the moment the run opened."""
        return self.directory / METRICS_FILE

    @property
    def checkpoint(self) -> RunCheckpoint | None:
        """The checkpoint a completed training run's final status selects, or ``None``.

        ``None`` is a run that has not completed.
        """
        if self.final is None or self.state != "completed":
            return None
        return self.final.checkpoint

    @property
    def heartbeat(self) -> datetime:
        """Its latest sign of life, an instant in UTC."""
        return datetime.fromtimestamp(self.alive, timezone.utc)

    @property
    def status_error(self) -> str | None:
        """The error the final status names, ``None`` while it has not ended."""
        return self.final.status_error if self.final is not None else None

    @property
    def sweep(self) -> str | None:
        """The sweep a trial run belongs to, ``None`` for a run launched on its own."""
        parent = self.directory.parent
        return parent.name if (parent / SWEEP_FILE).is_file() else None


def _observed[R: RunRecord | SweepRecord](
    directory: Path, record: R, record_file: str
) -> RunObservation[R]:
    """Return the observation of ``directory`` holding ``record`` in its ``record_file``.

    Its state is the one its final status names once one is written; before that ``running``
    while its latest sign of life is within :data:`HEARTBEAT_STALE_SECONDS`, else
    ``interrupted``.
    """
    final_path = directory / FINAL_STATUS_FILE
    final = (
        FinalStatus.model_validate(read_run_record(final_path))
        if final_path.exists()
        else None
    )
    alive = last_alive(directory, record_file)
    state: str
    if final is not None:
        state = final.state
    else:
        state = (
            "running"
            if time.time() - alive <= HEARTBEAT_STALE_SECONDS
            else "interrupted"
        )
    return RunObservation(directory, record, final, alive, state)


def read_run(run_dir: Path) -> RunRecord:
    """Return the :class:`RunRecord` the run at ``run_dir`` holds, its paths read as resolved."""
    return RunRecord.model_validate(read_run_record(run_dir / RUN_FILE))


def observe_run(directory: Path) -> RunObservation[RunRecord]:
    """Read the run at ``directory`` once (:func:`read_run`).

    A run holds its metrics log from the moment it opened; one missing it refuses with
    :class:`IncompleteRunError` naming it. A completed run's checkpoint is not read.
    """
    observation = _observed(directory, read_run(directory), RUN_FILE)
    if not observation.metrics_log.is_file():
        raise IncompleteRunError(
            f"run {directory.name} ({observation.state}) is missing "
            f"{observation.metrics_log}."
        )
    return observation


def observe_sweep(directory: Path) -> RunObservation[SweepRecord]:
    """Read the sweep at ``directory`` once, its :class:`SweepRecord` validated."""
    return _observed(
        directory,
        SweepRecord.model_validate(read_run_record(directory / SWEEP_FILE)),
        SWEEP_FILE,
    )


def observe(
    directory: Path,
) -> RunObservation[RunRecord] | RunObservation[SweepRecord]:
    """Read the sweep (holding its ``sweep.json``) or run at ``directory`` once.

    It is :func:`observe_sweep`'s or :func:`observe_run`'s.
    """
    if (directory / SWEEP_FILE).is_file():
        return observe_sweep(directory)
    return observe_run(directory)


class IncompleteRunError(ValueError):
    """A run directory missing a file every opened run holds."""


def _holding(parent: Path, record_file: str) -> list[Path]:
    """Return every directory under ``parent`` holding ``record_file``."""
    return (
        [d for d in parent.iterdir() if (d / record_file).is_file()]
        if parent.is_dir()
        else []
    )


def sweep_dirs(project: Path | str) -> list[Path]:
    """Return every sweep directory of ``project`` that holds its input."""
    return _holding(experiments_dir(project), SWEEP_FILE)


def run_dirs(project: Path | str) -> list[Path]:
    """Return every run directory of ``project`` that holds a launch record.

    Each sweep's trials are included.
    """
    return [
        *_holding(experiments_dir(project), RUN_FILE),
        *(
            d
            for sweep in sweep_dirs(project)
            for d in _holding(sweep, RUN_FILE)
        ),
    ]


def experiment_dirs(project: Path | str) -> list[Path]:
    """Return every run, trial and sweep directory of ``project``."""
    return [*run_dirs(project), *sweep_dirs(project)]


def run_observations(project: Path | str) -> list[RunObservation[RunRecord]]:
    """Return every run of ``project`` (:func:`run_dirs`), each observed once, in creation order."""
    return sorted(
        (observe_run(d) for d in run_dirs(project)),
        key=lambda observation: observation.record.created,
    )


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
) -> RunObservation[RunRecord] | None:
    """Return the run ``experiment_id`` names (:func:`run_dirs`), observed, or ``None``."""
    run_dir = named_directory(experiment_id, run_dirs(project))
    return observe_run(run_dir) if run_dir is not None else None


def run_resolution(
    experiment_id: str, *, project: Path | str
) -> ResolvedInput:
    """Return what the run ``experiment_id`` names resolved at launch (:func:`find_observation`).

    Refuses (``ValueError``) an id naming no run directory or one whose input resolved to
    nothing.
    """
    observation = find_observation(experiment_id, project=project)
    resolved = observation.record.resolved if observation is not None else None
    if resolved is None:
        raise ValueError(
            f"no run directory records what {experiment_id!r} resolved to."
        )
    return resolved


def partition_rows(
    rows: list[MetricsRow],
) -> tuple[list[MetricsRow], list[MetricsRow]]:
    """Split a run's metrics-log ``rows`` by kind, each in logged order.

    Returns its epoch rows, then its per-batch rows, those stating a ``step``.
    """
    epochs: list[MetricsRow] = []
    batches: list[MetricsRow] = []
    for row in rows:
        (epochs if row.step is None else batches).append(row)
    return epochs, batches


def epoch_rows(rows: list[MetricsRow]) -> list[MetricsRow]:
    """Return a run's metrics-log ``rows`` as one row per epoch, in first-logged order.

    The epoch rows (:func:`partition_rows`) of one epoch are merged, a later row's values laid
    over the earlier's, a metric and its non-finite state companion as one value.
    """
    merged: dict[int, dict[str, Any]] = {}
    for row in partition_rows(rows)[0]:
        laid = row.model_dump()
        restated = {f"{key}{NOT_FINITE_SUFFIX}" for key in laid}
        earlier = merged.get(row.epoch, {})
        merged[row.epoch] = {
            **{k: v for k, v in earlier.items() if k not in restated},
            **laid,
        }
    return [MetricsRow.model_construct(**row) for row in merged.values()]


def best_selection(
    rows: list[MetricsRow], higher_is_better: bool
) -> float | None:
    """Return the best finite ``selection`` among a run's metrics-log epoch rows.

    The epoch rows are :func:`partition_rows`'s; the best is the highest when
    ``higher_is_better`` and else the lowest, ``None`` when no row carries one.
    """
    values = [
        float(selection)
        for row in partition_rows(rows)[0]
        if finite_number(selection := getattr(row, "selection", None))
    ]
    if not values:
        return None
    return max(values) if higher_is_better else min(values)


class RunRow(BaseModel):
    """One run's status and summary fields."""

    experiment_id: str
    state: str
    created: AwareDatetime
    relaunched_from: str | None
    sweep: str | None
    trial_params: dict[str, JsonValue] | None
    builder: str
    task: str
    images_dir: str | None
    current_epoch: float | None
    best_metric: float | None
    best_metric_name: str | None
    output_dir: str
    status_error: str | None
    heartbeat: AwareDatetime
    launch: RecordedIdentity | None


def run_summary(
    observation: RunObservation[RunRecord],
    rows: list[MetricsRow],
    launches: Mapping[str, RecordedIdentity],
) -> RunRow:
    """Return one run's :class:`RunRow` over its metrics-log ``rows`` and its launch identity.

    ``rows`` are :func:`read_rows` of its ``metrics_log``, read as :func:`epoch_rows`;
    ``launches`` are its project's :func:`launch_declarations`, which the run's own identity is
    read from by its id (``None`` for a run no launch event names). The row holds its record's
    own fields, its last logged epoch,
    and its best selection value (:func:`best_selection`) under its config's selection metric and
    direction (``schemas.selection_of``) with that metric's name, both ``None`` for a run that
    resolved to nothing.
    """
    record = observation.record
    spec = record.spec
    selection = (
        selection_of(spec.evaluation.selection_metric)
        if record.resolved is not None
        else None
    )
    rows = epoch_rows(rows)
    return RunRow(
        experiment_id=observation.directory.name,
        state=observation.state,
        created=record.created,
        relaunched_from=record.relaunched_from,
        sweep=observation.sweep,
        trial_params=record.trial_params,
        builder=spec.model_source.builder,
        task=spec.model_source.task,
        images_dir=spec.data.images_dir,
        current_epoch=rows[-1].epoch if rows else None,
        best_metric=best_selection(rows, selection[1])
        if selection is not None
        else None,
        best_metric_name=selection[0] if selection is not None else None,
        output_dir=str(observation.directory),
        status_error=observation.status_error,
        heartbeat=observation.heartbeat,
        launch=launches.get(observation.directory.name),
    )


def run_rows(project: Path | str) -> list[RunRow]:
    """Return every run of ``project`` (:func:`run_observations`) as its summary row."""
    launches = launch_declarations(project)
    return [
        run_summary(obs, read_rows(obs.metrics_log)[0], launches)
        for obs in run_observations(project)
    ]


def _completed_with_value(trial: RunRow) -> bool:
    """Return whether a sweep's trial completed reporting a value under the sweep's selection."""
    return trial.state == "completed" and trial.best_metric is not None


class DrawSpread(BaseModel):
    """What one point's split draws amount to (:func:`group_split_draws`).

    ``n`` is every trial of the point, ``n_complete`` the trials among them that completed with a
    value and ``seeds_complete`` the distinct seeds among those, sorted; ``seeds`` holds one entry
    per completed trial, not deduplicated, beside its ``values``, and ``mean``, ``std`` (the
    sample standard deviation, ``None`` under two values), ``min`` and ``max`` are over
    ``values``, each ``None`` with no value.
    """

    seeds: list[int | None]
    values: list[float]
    mean: float | None
    std: float | None
    min: float | None
    max: float | None
    n: int
    n_complete: int
    seeds_complete: list[int | None]


class DrawGroup(BaseModel):
    """One point of a sweep, every param but its draw seed, and its draws' :class:`DrawSpread`.

    ``eligible`` is whether the point may be chosen best (:func:`group_split_draws`).
    """

    point: dict[str, Any]
    block: DrawSpread
    eligible: bool


def group_split_draws(
    trials: list[RunRow], planned_seeds: list[int]
) -> list[DrawGroup]:
    """Group a sweep's trial rows by the point their draws share.

    The point is every param but ``hpo.SPLIT_DRAW_SEED_KEY``. ``planned_seeds`` names every seed
    the sweep asked for; a group is ``eligible`` for best only when every one of them completed
    for that point and the group holds no trial that did not. An empty list accepts any group
    whose trials all completed with a value, regardless of seed identity.
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

    out: list[DrawGroup] = []
    for entry in groups.values():
        rows = entry["rows"]
        complete = [r for r in rows if _completed_with_value(r)]
        values = [float(r.best_metric) for r in complete]
        seeds = [
            (r.trial_params or {}).get(SPLIT_DRAW_SEED_KEY) for r in complete
        ]
        seeds_complete = sorted(set(seeds), key=lambda s: (s is None, s))
        block = DrawSpread(
            seeds=seeds,
            values=values,
            mean=statistics.fmean(values) if values else None,
            std=statistics.stdev(values) if len(values) > 1 else None,
            min=min(values) if values else None,
            max=max(values) if values else None,
            n=len(rows),
            n_complete=len(values),
            seeds_complete=seeds_complete,
        )
        eligible = len(complete) == len(rows) and planned <= set(
            seeds_complete
        )
        out.append(
            DrawGroup(point=entry["point"], block=block, eligible=eligible)
        )
    return out


def sweep_selection(given: SweepInput) -> tuple[str, bool]:
    """Return the metric and direction a sweep whose recorded input is ``given`` ranks trials by.

    They are ``schemas.selection_of`` its base config's ``evaluation.selection_metric``, which no
    trial's point sweeps; the ``evaluation`` block is validated as an ``EvaluationSpec``.
    """
    return selection_of(
        EvaluationSpec.model_validate(
            given.base_config["evaluation"]
        ).selection_metric
    )


RangeEnd = Literal["low", "high"]
"""An end of a numeric axis's stated range."""


class SweepOutcome(BaseModel):
    """What a sweep's trial rows amount to (:func:`sweep_outcome`).

    ``best_params`` is ``None`` when no trial or point qualifies; ``best_value`` is the best
    value, ``None`` beside ``best_value_state`` naming why a value is not finite
    (``tcip_store.stored_number``). ``best_value_spread`` and ``split_sensitivity`` are a sweep's
    above one draw (:func:`group_split_draws`), ``None`` otherwise. ``best_at_range_end`` names
    each numeric axis whose end the best params sit at.
    """

    best_params: dict[str, Any] | None
    best_value: float | None
    best_value_state: str | None = None
    best_value_spread: DrawSpread | None = None
    split_sensitivity: list[DrawGroup] | None = None
    best_at_range_end: dict[str, RangeEnd]


def sweep_outcome(trials: list[RunRow], given: SweepInput) -> SweepOutcome:
    """Return what a sweep's trial rows amount to under its recorded input ``given``.

    That is ``best_params`` and ``best_value`` over its completed trials in its selection's
    direction (:func:`sweep_selection`); above one draw, the best point by mean over its draws
    (:func:`group_split_draws`), with ``best_value_spread`` and every point's
    ``split_sensitivity``. ``best_at_range_end`` names each numeric axis of the stated space
    (``hpo.space_axes``) whose ``low`` or ``high`` the best params sit at, the end beside it: a
    best the search could not look past.
    """
    from tcip_mcp.pipelines.training.hpo import space_axes

    choose = max if sweep_selection(given)[1] else min
    if given.split_draw_seeds is not None:
        groups = group_split_draws(trials, given.split_draw_seeds)
        eligible = [g for g in groups if g.eligible]
        best_group = (
            choose(eligible, key=lambda g: cast(float, g.block.mean))
            if eligible
            else None
        )
        best_params = best_group.point if best_group else None
        best_value = best_group.block.mean if best_group else None
        spread = best_group.block if best_group else None
        sensitivity: list[DrawGroup] | None = groups
    else:
        done = [t for t in trials if _completed_with_value(t)]
        best_trial = (
            choose(done, key=lambda t: cast(float, t.best_metric))
            if done
            else None
        )
        best_params = best_trial.trial_params if best_trial else None
        best_value = best_trial.best_metric if best_trial else None
        spread, sensitivity = None, None
    params = best_params or {}
    return SweepOutcome(
        best_params=best_params,
        best_value=finite_or_none(best_value),
        best_value_state=None if best_value is None else non_finite_state(best_value),
        best_value_spread=spread,
        split_sensitivity=sensitivity,
        best_at_range_end={
            name: end
            for name, (ptype, ends) in space_axes(
                given.param_space, grid=False
            ).items()
            if ptype != "categorical" and name in params
            for end, bound in zip(get_args(RangeEnd), ends)
            if params[name] == bound
        },
    )


class SweepGroup(BaseModel):
    """One sweep: its record's own fields, its trial rows and what they amount to."""

    sweep_id: str
    state: str
    status_error: str | None
    input: SweepInput
    cancel_requested: bool
    heartbeat: AwareDatetime
    trials: list[RunRow]
    outcome: SweepOutcome

    @computed_field  # type: ignore[prop-decorator]
    @property
    def selection_metric(self) -> str:
        """The metric the sweep ranks its trials by (:func:`sweep_selection`)."""
        return sweep_selection(self.input)[0]


def read_sweep(
    sweep: RunObservation[SweepRecord], rows: list[RunRow]
) -> SweepGroup:
    """Return the observed ``sweep`` as its :class:`SweepGroup`.

    Its trials are the ``rows`` of the project's runs (:func:`run_rows`) it holds, with their
    outcome (:func:`sweep_outcome`).
    """
    directory = sweep.directory
    trials = [row for row in rows if row.sweep == directory.name]
    return SweepGroup(
        sweep_id=directory.name,
        state=sweep.state,
        status_error=sweep.status_error,
        input=sweep.record.input,
        cancel_requested=cancel_requested(directory),
        heartbeat=sweep.heartbeat,
        trials=trials,
        outcome=sweep_outcome(trials, sweep.record.input),
    )


class TrainingDetail(BaseModel):
    """One run, trial or sweep, with the TensorBoard serving its board.

    It holds a run's row or a sweep's group, the other ``None``; ``tensorboard`` is ``None``
    while no TensorBoard of this process serves the board.
    """

    run: RunRow | None
    sweep: SweepGroup | None
    tensorboard: TensorboardServing | None


class TrainingListing(BaseModel):
    """Every run of a project launched on its own as its row, and every sweep as its group."""

    runs: list[RunRow]
    sweeps: list[SweepGroup]


def training_listing(project: Path | str) -> TrainingListing:
    """Return ``project``'s runs launched on their own, and its sweeps with their trial rows.

    The runs are :func:`run_rows`'s and the sweeps :func:`read_sweep`'s, each in creation order.
    """
    rows = run_rows(project)
    sweeps = sorted(
        (observe_sweep(d) for d in sweep_dirs(project)),
        key=lambda sweep: sweep.record.created,
    )
    return TrainingListing(
        runs=[row for row in rows if row.sweep is None],
        sweeps=[read_sweep(sweep, rows) for sweep in sweeps],
    )


class _Launched(BaseModel):
    """The run a ``launch_training`` audit line's arguments name."""

    experiment_id: str


def launch_declarations(project: Path | str) -> dict[str, RecordedIdentity]:
    """Return the agent identity each run's ``launch_training`` audit line carries, by run id.

    The lines are ``project``'s audit log's, keyed by the experiment id each names; an identity's
    fields are ``None`` for a launch no agent declared itself to.
    """
    from tcip_mcp.audit import acts_of

    return {
        _Launched.model_validate(entry.arguments).experiment_id: (
            RecordedIdentity.model_validate(entry, from_attributes=True)
        )
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
        "run": observation.record.model_dump(mode="json", exclude_unset=True),
        "final_status": observation.final
        and observation.final.model_dump(mode="json"),
        "state": observation.state,
        "n_epochs": len(rows),
        "metrics": [
            row.model_dump(mode="json") for row in rows[metrics_offset:end]
        ],
        "metrics_offset": metrics_offset,
    }


class SplitColumn(BaseModel):
    """A comparison's partition column for one run.

    ``case`` is ``bound`` for a run bound to a selection, with its ``selection_dir``, ``seed``
    and ``redraw``; ``drawn`` for a drawn split, with its ``seed``; ``spatial`` for a
    within-image split; ``none`` for a run that resolved to nothing. A field its case does not
    state is ``None``.
    """

    case: Literal["bound", "drawn", "spatial", "none"]
    selection_dir: str | None = None
    seed: int | None = None
    redraw: bool | None = None


class ComparedRun(BaseModel):
    """One run's column in a comparison, each value from the record it names.

    ``run`` is the run's own summary row (:func:`run_summary`). ``last_logged_metrics`` is the
    last epoch's row (:func:`epoch_rows`, not a verified result), ``None`` before one;
    ``rows_after_end`` counts rows whose own ``timestamp`` is a later instant than the final
    status's ``ended``, ``None`` before a final status; ``registry`` is a completed run's own
    entry as listed, a one-entry list, empty otherwise. The subject and dataset identity are its
    records'.
    """

    run: RunRow
    n_epochs: int
    last_logged_metrics: MetricsRow | None
    rows_after_end: int | None
    registry: list[Listed]
    split: SplitColumn
    subject: str | None
    dataset_id: str | None
    dataset_fingerprint: str | None


class UncomparedRun(BaseModel):
    """An id a comparison could not read, and why."""

    experiment_id: str
    error: str


FingerprintMatch = Literal["same", "different", "unread", "unset"]
"""Whether compared runs name one dataset fingerprint: ``same`` or ``different`` when every run
was read and carries one, ``unread`` when some run could not be read, ``unset`` when a run read
carries none."""


class Comparison(BaseModel):
    """Runs side by side, and whether they name one fingerprint (:data:`FingerprintMatch`).

    ``higher_is_better`` holds each metric key their registered checkpoints carry
    (``model_registry.metric_keys``) with its declared direction
    (``schemas.declared_higher_is_better``), ``None`` where none is declared.
    """

    experiments: list[ComparedRun | UncomparedRun]
    dataset_fingerprints: FingerprintMatch
    higher_is_better: dict[str, bool | None]


def _split_summary(record: RunRecord) -> SplitColumn:
    """Return the partition column of a comparison of the run whose record is ``record``."""
    if record.resolved is None:
        return SplitColumn(case="none")
    partition = record.resolved.partition
    if (bound := partition.selection) is not None:
        return SplitColumn(
            case="bound",
            selection_dir=bound.selection_dir,
            seed=partition.seed,
            redraw=bound.redraw,
        )
    if partition.spatial is not None:
        return SplitColumn(case="spatial")
    return SplitColumn(case="drawn", seed=partition.seed)


def compare_experiments(
    experiment_ids: list[str], *, project: Path | str
) -> Comparison:
    """Compare training runs side by side.

    Each run is a :class:`ComparedRun`: its registry entry is ``model_registry.run_entry``'s as
    listed (``model_registry.listed_entry``), and its split :func:`_split_summary`'s. An id naming
    no run, or one whose metrics log :func:`read_rows` refuses, is an :class:`UncomparedRun`.
    """
    from tcip_mcp.model_registry import listed_entry, run_entry

    launches = launch_declarations(project)
    comparisons: list[ComparedRun | UncomparedRun] = []
    for eid in experiment_ids:
        observation = find_observation(eid, project=project)
        if observation is None:
            comparisons.append(
                UncomparedRun(
                    experiment_id=eid, error=f"Experiment not found: {eid}"
                )
            )
            continue
        run, final = observation.record, observation.final
        try:
            metrics = read_rows(observation.metrics_log)[0]
        except DecodeError as exc:
            comparisons.append(
                UncomparedRun(experiment_id=eid, error=str(exc))
            )
            continue
        epochs = epoch_rows(metrics)
        entry = run_entry(observation)
        registry = (
            []
            if entry is None
            else [
                listed_entry(str(project_of_run(observation.directory)), entry)
            ]
        )
        comparisons.append(
            ComparedRun(
                run=run_summary(observation, metrics, launches),
                n_epochs=len(epochs),
                last_logged_metrics=epochs[-1] if epochs else None,
                rows_after_end=None
                if final is None
                else sum(1 for row in metrics if row.timestamp > final.ended),
                registry=registry,
                split=_split_summary(run),
                subject=run.resolved.data.recorded_scope.subject
                if run.resolved is not None
                else None,
                dataset_id=run.dataset.id,
                dataset_fingerprint=run.dataset.fingerprint,
            )
        )

    compared = [c for c in comparisons if isinstance(c, ComparedRun)]
    fps = {c.dataset_fingerprint for c in compared}
    match: FingerprintMatch = (
        "unread"
        if len(compared) < len(comparisons)
        else "unset"
        if not fps or None in fps
        else "same"
        if len(fps) == 1
        else "different"
    )
    keys = metric_keys([listed for c in compared for listed in c.registry])
    return Comparison(
        experiments=comparisons,
        dataset_fingerprints=match,
        higher_is_better={k: declared_higher_is_better(k) for k in keys},
    )


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
    run = observation.record
    return {
        "experiment_id": experiment_id,
        "lineage": {
            "data": run.resolved.data.record()
            if run.resolved is not None
            else None,
            "dataset_id": run.dataset.id,
            "dataset_fingerprint": run.dataset.fingerprint,
            "relaunched_from": run.relaunched_from,
            "resume_from": run.resume_from,
            "checkpoint": observation.checkpoint
            and observation.checkpoint.model_dump(mode="json"),
        },
    }
