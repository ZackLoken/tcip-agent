"""A run as a directory: ``<project>/.tcip/experiments/<experiment_id>/``.

Each file in it has one writer:

- ``run.json``: the launcher's record of the run, its input and what that input resolved to,
  written once before the child starts.
- ``metrics.jsonl``: the child's epoch rows, created empty with the directory and appended.
- ``heartbeat``: touched by the child while it lives.
- ``cancel_requested.json``: the first cancellation request (:func:`request_cancel`).
- ``final_status.json``: the outcome, written once at exit (:func:`write_final_status`).

Checkpoints, TensorBoard logs, a bespoke run's source snapshot and its recorded artifacts are
files beside them. A sweep is a directory of trial run directories with its own input, heartbeat,
cancellation and final status (``tools.training_tools``).
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, BinaryIO

import tcip_store
from tcip_store import (
    BadKey, DecodeError, decode_value, encode_log_line, encode_record,
)

from tcip_mcp.audit import now_iso
from tcip_mcp.pipelines.data.selection import SAMPLE_PATHS
from tcip_mcp.registry_paths import PathFields, recorded_paths, runtime_paths, within

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(".tcip/experiments")
SWEEPS_DIR = Path(".tcip/hpo")

RUN_FILE = "run.json"
METRICS_FILE = "metrics.jsonl"
HEARTBEAT_FILE = "heartbeat"
CANCEL_FILE = "cancel_requested.json"
FINAL_STATUS_FILE = "final_status.json"
SWEEP_FILE = "sweep.json"
"""A sweep directory's input, written once before its first trial."""
TRIAL_DIR_PREFIX = "trial_"
"""The name every trial run directory of a sweep starts with."""

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
        ("resolved", "partition", "ground_truth_digests", "{}"),
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
    """The directory every run directory of ``project`` sits in."""
    return Path(project) / EXPERIMENTS_DIR


def sweeps_dir(project: Path | str) -> Path:
    """The directory every HPO sweep directory of ``project`` sits in."""
    return Path(project) / SWEEPS_DIR


def project_of_run(run_dir: Path) -> Path:
    """The project a run directory, a sweep directory or a sweep's trial directory lies under:
    the parent of the ``.tcip`` directory on its path. Refuses a directory under no ``.tcip``."""
    for parent in Path(run_dir).parents:
        if parent.name == ".tcip":
            return parent.parent
    raise ValueError(f"{run_dir} lies under no project's .tcip directory")


def run_name(name: str) -> str:
    """``name`` once it is known to be one directory name. A separator, a drive, an empty name or
    a dot name refuses with ``BadKey``."""
    if not name or name in (".", "..") or PureWindowsPath(name).name != name:
        raise BadKey(
            f"{name!r} is not a single directory name: a name carrying a separator, a drive or a "
            "parent reference would address a directory outside the one it names."
        )
    return name


def experiment_dir(experiment_id: str, *, project: Path | str) -> Path:
    """One run's directory. Refuses an id that is not a single directory name (:func:`run_name`)."""
    return experiments_dir(project) / run_name(experiment_id)


def find_run(experiment_id: str, *, project: Path | str) -> Path | None:
    """The run directory ``experiment_id`` names, or ``None`` for an id that is not a single
    directory name or names no directory holding a launch record."""
    try:
        run_dir = experiment_dir(experiment_id, project=project)
    except BadKey:
        return None
    return run_dir if (run_dir / RUN_FILE).is_file() else None


def mint_experiment_id(prefix: str = "run") -> str:
    """A fresh run id: ``<prefix>_<epoch-seconds>_<6 hex chars>``."""
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:6]}"


# ── the files ────────────────────────────────────────────────────────────────


class RunDirectoryExists(FileExistsError):
    """A launch named a run or sweep directory that already exists."""


def create_run_directory(directory: Path) -> Path:
    """Create ``directory`` and its parents, refusing one that already exists with
    :class:`RunDirectoryExists` naming it."""
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise RunDirectoryExists(
            f"{directory} already exists: a run directory is written by one launch, so a new run "
            "takes a new id."
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


def write_final_status(directory: Path, state: str, error: str | None, **outcome: Any) -> None:
    """Write the final status of the run or sweep at ``directory`` once: ``state``, the instant,
    the error behind it, and ``outcome``, what it produced (a run's ``checkpoint``). Refuses
    (``ValueError``) a state outside :data:`FINAL_STATES`, and a status already written with
    ``FileExistsError``."""
    if state not in FINAL_STATES:
        raise ValueError(f"{state!r} is not a final state; a final status names one of "
                         f"{list(FINAL_STATES)}.")
    write_record(directory / FINAL_STATUS_FILE,
                 {"state": state, "ended": now_iso(), "error": error, **outcome})


def append_row(path: Path, row: dict) -> None:
    """Append ``row`` as one line of the JSON log at ``path``, refusing a row JSON cannot hold
    and (``FileNotFoundError``) a log its directory was not opened with."""
    line = encode_log_line(row) + b"\n"
    with open(path, "r+b") as handle:
        handle.seek(0, os.SEEK_END)
        handle.write(line)


def read_record(path: Path) -> Any:
    """The JSON record at ``path``. A missing file raises ``FileNotFoundError``; bytes that do not
    decode raise ``DecodeError`` naming the file."""
    data = path.read_bytes()
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


class RunEnded(ValueError):
    """A write addressed a run or sweep whose final status is written."""


def require_open(directory: Path) -> None:
    """Refuse (:class:`RunEnded`) the run or sweep at ``directory`` once its final status is
    written: from then on it takes no write."""
    if (directory / FINAL_STATUS_FILE).exists():
        raise RunEnded(f"{directory.name} has ended: its final status is written, so it takes no "
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


def observe(directory: Path, record_file: str = RUN_FILE) -> RunObservation:
    """Read the run or sweep at ``directory`` once. Its state is the one its final status names
    once one is written; before that ``running`` while its latest sign of life is within
    :data:`HEARTBEAT_STALE_SECONDS`, else ``interrupted``. A run (``record_file`` its ``run.json``)
    holds its metrics log from the moment it opened; one missing it refuses with
    ``FileNotFoundError`` naming it. A completed run's checkpoint is read, and refused when
    absent, by whatever loads it: an archive may carry a run without its weights."""
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


def run_dirs(project: Path | str) -> list[Path]:
    """Every run directory of ``project`` that holds a launch record, sorted by
    id."""
    parent = experiments_dir(project)
    if not parent.is_dir():
        return []
    return sorted(d for d in parent.iterdir() if (d / RUN_FILE).is_file())


def run_observations(project: Path | str) -> list[RunObservation]:
    """Every run of ``project`` (:func:`run_dirs`), each observed once."""
    return [observe(d) for d in run_dirs(project)]


def live_run_conflict(project: Path) -> str | None:
    """The refusal a run or HPO sweep of ``project`` still running answers with, or
    ``None``."""
    for run in run_observations(project):
        if run.state == "running":
            return (f"experiment {run.directory.name!r} is running; ask the agent to cancel it "
                    "(cancel_training) or wait for it to finish")
    sweeps = sweeps_dir(project)
    for sweep in sorted(sweeps.iterdir()) if sweeps.is_dir() else []:
        if (sweep / SWEEP_FILE).is_file() and observe(sweep, SWEEP_FILE).state == "running":
            return (f"HPO sweep {sweep.name!r} is running; ask the agent to cancel it "
                    "(cancel_hyperparameter_search) or wait for it to finish")
    return None


def find_observation(experiment_id: str, *, project: Path | str) -> RunObservation | None:
    """The run ``experiment_id`` names (:func:`find_run`), observed, or ``None``."""
    run_dir = find_run(experiment_id, project=project)
    return observe(run_dir) if run_dir is not None else None


def run_resolution(experiment_id: str, *, project: Path | str) -> dict:
    """What the run ``experiment_id`` names resolved at launch (:func:`find_observation`): its
    ``data`` section, its ``partition`` and its ``objective``. Refuses (``ValueError``) an id
    naming no run directory."""
    observation = find_observation(experiment_id, project=project)
    if observation is None:
        raise ValueError(f"no run directory records {experiment_id!r}, so nothing it resolved "
                         "can be read.")
    return observation.record["resolved"]


def best_selection(rows: list[dict[str, Any]], objective: dict) -> float | None:
    """The best finite ``selection`` among a run's metrics-log ``rows`` in the direction of
    ``objective``, the run's recorded ``{"selection_metric", "higher_is_better"}``; ``None`` when
    no row carries one."""
    values = [float(row["selection"]) for row in rows
              if isinstance(row.get("selection"), (int, float))
              and not isinstance(row["selection"], bool) and math.isfinite(row["selection"])]
    if not values:
        return None
    return max(values) if objective["higher_is_better"] else min(values)


def run_summary(observation: RunObservation, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One run's status row over its metrics-log ``rows`` (:func:`read_rows` of its
    ``metrics_log``): its state, its last logged epoch, its directory, its final status's error,
    its last sign of life, and its best selection value under the objective its launch recorded,
    folded from ``rows`` (:func:`best_selection`), with that objective's metric name."""
    run_dir = observation.directory
    objective = observation.record["resolved"]["objective"]
    final = observation.final
    return {
        "experiment_id": run_dir.name,
        "status": observation.state,
        "current_epoch": rows[-1].get("epoch") if rows else None,
        "best_metric": best_selection(rows, objective),
        "best_metric_name": objective["selection_metric"],
        "output_dir": str(run_dir),
        "error": final["error"] if final is not None else None,
        "heartbeat": datetime.fromtimestamp(observation.alive, timezone.utc).isoformat(),
    }


def launch_declarations(project: Path | str) -> dict[str, dict[str, Any]]:
    """The agent identity each run's ``launch_training`` line in ``project``'s audit log carries,
    by the experiment id it names: empty for a launch no agent declared itself to."""
    from tcip_mcp import agent_identity
    from tcip_mcp.audit import audit_log_key

    return {entry["arguments"]["experiment_id"]:
            {field: entry[field] for field in agent_identity.RECORD_FIELDS if field in entry}
            for entry in tcip_store.read_log(audit_log_key(project)).records
            if entry.get("tool") == "launch_training"}


def _distinct_epoch_count(rows: list[dict[str, Any]]) -> int:
    """The number of distinct ``epoch`` values among ``rows``, compared as canonical JSON text."""
    return len({json.dumps(row.get("epoch"), sort_keys=True, default=str) for row in rows})


def get_experiment(
    experiment_id: str, *, project: Path | str, metrics_limit: int | None = None,
    metrics_offset: int = 0,
) -> dict[str, Any]:
    """One run's directory read whole: its launch record, its final status (``None`` until
    written), its state, its metrics rows paginated by ``metrics_offset`` and ``metrics_limit``
    over the row list (``n_rows`` bounds the paging, ``n_epochs`` counts distinct epochs).
    ``{"error": ...}`` for an id naming no run."""
    observation = find_observation(experiment_id, project=project)
    if observation is None:
        return {"error": f"Experiment not found: {experiment_id}"}
    rows = read_rows(observation.metrics_log)[0]
    end = (metrics_offset + metrics_limit) if metrics_limit is not None else None
    return {
        "experiment_id": experiment_id,
        "run": observation.record,
        "final_status": observation.final,
        "state": observation.state,
        "n_epochs": _distinct_epoch_count(rows),
        "n_rows": len(rows),
        "metrics": rows[metrics_offset:end],
        "metrics_offset": metrics_offset,
    }


def list_experiments(project: Path | str) -> list[dict[str, Any]]:
    """Every run directory of ``project``: its id, state and creation time."""
    return [{
        "experiment_id": obs.directory.name,
        "state": obs.state,
        "created": obs.record["created"],
    } for obs in run_observations(project)]


def _split_summary(resolved: dict) -> dict[str, Any]:
    """The partition column of a comparison: ``{"case": "bound", "selection_dir", "seed",
    "redraw"}`` for a run bound to a selection, ``{"case": "spatial"}`` for a within-image split,
    and ``{"case": "drawn", "seed"}`` otherwise."""
    partition = resolved["partition"]
    binding = partition["selection"]
    if binding is not None:
        return {"case": "bound", "selection_dir": binding["selection_dir"],
                "seed": partition["seed"], "redraw": binding["redraw"]}
    if "spatial_manifest" in resolved["data"]["split"]:
        return {"case": "spatial"}
    return {"case": "drawn", "seed": partition["seed"]}


def _row_instant(row: dict, experiment_id: str) -> datetime:
    """A metric row's own ``timestamp`` as an instant. Refuses (``ValueError``) a row whose
    timestamp is absent or does not decode, naming the row and the run."""
    try:
        return datetime.fromisoformat(row["timestamp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"metric row {row} of {experiment_id} carries no decodable timestamp "
                         f"({exc!r}); the run's metrics log cannot be compared") from exc


def compare_experiments(experiment_ids: list[str], *, project: Path | str) -> dict[str, Any]:
    """Side-by-side comparison of training runs.

    Per run: ``state``, ``n_epochs``/``n_rows``, ``last_logged_metrics`` (the log's last row, not
    a verified result), ``rows_after_end`` (rows whose own ``timestamp`` is a later instant than
    the final status's ``ended``; ``None`` before a final status), the final status's ``error`` as
    ``status_error``, ``registry`` (a completed run's own registry entry,
    ``model_registry.run_entry``, with the metrics and source its checkpoint carries,
    ``model_registry.entry_facts``, as a one-entry list, empty otherwise), ``split``
    (:func:`_split_summary`), and the builder, task, subject and dataset identity its records
    carry. An id naming no run is an entry carrying only ``error``.
    ``same_dataset_fingerprint`` is ``None`` when any entry is an error or any fingerprint is
    unset, else whether every run names one fingerprint.
    """
    from tcip_mcp.model_registry import entry_facts, run_entry
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.model_build import MODEL_SOURCE_KEY, run_task

    comparisons: list[dict[str, Any]] = []
    for eid in experiment_ids:
        observation = find_observation(eid, project=project)
        if observation is None:
            comparisons.append({"experiment_id": eid, "error": f"Experiment not found: {eid}"})
            continue
        run, final = observation.record, observation.final
        config = run["config"]
        metrics = read_rows(observation.metrics_log)[0]
        summary: dict[str, Any] = {
            "experiment_id": eid, "state": observation.state,
            "n_epochs": _distinct_epoch_count(metrics), "n_rows": len(metrics),
        }
        if metrics:
            summary["last_logged_metrics"] = metrics[-1]
        try:
            instants = [_row_instant(row, eid) for row in metrics]
        except ValueError as exc:
            comparisons.append({"experiment_id": eid, "error": str(exc)})
            continue
        rows_after_end = None if final is None else sum(
            1 for at in instants if at > datetime.fromisoformat(final["ended"]))
        summary["rows_after_end"] = rows_after_end
        summary["status_error"] = final["error"] if final is not None else None
        entry = run_entry(observation)
        summary["registry"] = [] if entry is None else [{
            "name": entry["name"], "registered_at": entry["registered_at"],
            **entry_facts(entry),
        }]
        resolved = run["resolved"]
        summary["split"] = _split_summary(resolved)
        summary["model"] = config[MODEL_SOURCE_KEY]["builder"]
        summary["task"] = run_task(config)
        summary["subject"] = ClassScope.of(resolved["data"]).subject
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
    run = observation.record
    return {"experiment_id": experiment_id, "lineage": {
        "data": run["resolved"]["data"],
        "dataset_id": run["dataset"]["id"],
        "dataset_fingerprint": run["dataset"]["fingerprint"],
        "parent_experiment": run["parent_experiment"],
        "resume_from": run["resume_from"],
        "checkpoint": observation.checkpoint,
    }}
