"""A run as a directory: ``<project>/.tcip/experiments/<experiment_id>/``.

Each file in it has one writer:

- ``run.json``: the launcher's record of the run, its input and what that input resolved to,
  written once before the child starts.
- ``metrics.jsonl``: the child's epoch rows, created empty with the directory and appended.
- ``heartbeat``: touched by the child while it lives.
- ``cancel_requested.json``: the first cancellation request (:func:`request_cancel`).
- ``final_status.json``: the outcome, written once at exit (:func:`write_final_status`).
- ``validations.jsonl``: the claims the seal door earned against this run, created empty with the
  directory and appended.

Checkpoints, TensorBoard logs, a bespoke run's source snapshot and its recorded artifacts are
files beside them. A sweep is a directory of trial run directories with its own input, heartbeat,
cancellation and final status (``tools.training_tools``).
"""

from __future__ import annotations

import hashlib
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

from tcip_store import LOG_JSON, RECORD_JSON, BadKey, DecodeError, check_json_value

from tcip_mcp.project_paths import resolve_state

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(".tcip/experiments")
SWEEPS_DIR = Path(".tcip/hpo")

RUN_FILE = "run.json"
METRICS_FILE = "metrics.jsonl"
HEARTBEAT_FILE = "heartbeat"
CANCEL_FILE = "cancel_requested.json"
FINAL_STATUS_FILE = "final_status.json"
VALIDATIONS_FILE = "validations.jsonl"
SWEEP_FILE = "sweep.json"
"""A sweep directory's input, written once before its first trial."""
TRIAL_DIR_PREFIX = "trial_"
"""The name every trial run directory of a sweep starts with."""

FINAL_STATES = ("completed", "failed", "canceled")
"""The states a final status names."""
TERMINAL_STATES = frozenset({*FINAL_STATES, "interrupted"})
"""Every state a run or sweep ends in: a final status's, and ``interrupted``, the state a directory
with no final status reads as once its heartbeat is stale."""

HEARTBEAT_STALE_SECONDS = float(os.environ.get("TCIP_HEARTBEAT_STALE_SECONDS", "600"))
"""How long a directory with no final status reads ``running`` after its last sign of life,
from ``$TCIP_HEARTBEAT_STALE_SECONDS``."""


def experiments_dir(root: Path | str | None = None) -> Path:
    """The directory every run directory of a project sits in: under ``root`` when given, else
    under the pinned platform root."""
    return resolve_state(EXPERIMENTS_DIR) if root is None else Path(root) / EXPERIMENTS_DIR


def sweeps_dir(root: Path | str | None = None) -> Path:
    """The directory every HPO sweep directory of a project sits in: under ``root`` when given,
    else under the pinned platform root."""
    return resolve_state(SWEEPS_DIR) if root is None else Path(root) / SWEEPS_DIR


def run_name(name: str) -> str:
    """``name`` once it is known to be one directory name. A separator, a drive, an empty name or
    a dot name refuses with ``BadKey``."""
    if not name or name in (".", "..") or PureWindowsPath(name).name != name:
        raise BadKey(
            f"{name!r} is not a single directory name: a name carrying a separator, a drive or a "
            "parent reference would address a directory outside the one it names."
        )
    return name


def experiment_dir(experiment_id: str, *, root: Path | str | None = None) -> Path:
    """One run's directory. Refuses an id that is not a single directory name (:func:`run_name`)."""
    return experiments_dir(root) / run_name(experiment_id)


def find_run(experiment_id: str, *, root: Path | str | None = None) -> Path | None:
    """The run directory ``experiment_id`` names, or ``None`` for an id that is not a single
    directory name or names no directory holding a launch record."""
    try:
        run_dir = experiment_dir(experiment_id, root=root)
    except BadKey:
        return None
    return run_dir if (run_dir / RUN_FILE).is_file() else None


def mint_experiment_id() -> str:
    """A fresh run id: ``run_<epoch-seconds>_<6 hex chars>``."""
    return f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"


def now_iso() -> str:
    """The current instant as an ISO-8601 UTC string."""
    return datetime.now(timezone.utc).isoformat()


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
    and validations logs, then write the record ``compose(run_dir)`` returns as its ``run.json``
    once, so a directory holding a launch record holds every file a reader of a live run reads.
    ``compose`` may write files of its own into the directory first."""
    create_run_directory(run_dir)
    for log in (METRICS_FILE, VALIDATIONS_FILE):
        (run_dir / log).touch(exist_ok=False)
    write_once(run_dir / RUN_FILE, compose(run_dir))


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
    check_json_value(value, path=path.name)
    data = RECORD_JSON.encode(value)
    publish_once(path, lambda handle: handle.write(data))


def write_final_status(directory: Path, state: str, error: str | None, **outcome: Any) -> None:
    """Write the final status of the run or sweep at ``directory`` once: ``state``, the instant,
    the error behind it, and ``outcome``, what it produced (a run's ``checkpoint``). Refuses
    (``ValueError``) a state outside :data:`FINAL_STATES`, and a status already written with
    ``FileExistsError``."""
    if state not in FINAL_STATES:
        raise ValueError(f"{state!r} is not a final state; a final status names one of "
                         f"{list(FINAL_STATES)}.")
    write_once(directory / FINAL_STATUS_FILE,
               {"state": state, "ended": now_iso(), "error": error, **outcome})


def append_row(path: Path, row: dict) -> None:
    """Append ``row`` as one line of the JSON log at ``path``, refusing a row JSON cannot hold
    and (``FileNotFoundError``) a log its directory was not opened with."""
    check_json_value(row, path=path.name)
    with open(path, "r+b") as handle:
        handle.seek(0, os.SEEK_END)
        handle.write(LOG_JSON.encode(row) + b"\n")


def read_record(path: Path) -> Any:
    """The JSON record at ``path``. A missing file raises ``FileNotFoundError``; bytes that do not
    decode raise ``DecodeError`` naming the file."""
    data = path.read_bytes()
    try:
        return RECORD_JSON.decode(data)
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
            rows.append(LOG_JSON.decode(line))
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
    written: from then on it takes no write but a validation."""
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
    def validations_log(self) -> Path:
        """The run's validations log, which exists from the moment the run opened."""
        return self.directory / VALIDATIONS_FILE

    @property
    def checkpoint(self) -> dict | None:
        """The checkpoint a completed training run's final status selects, ``path`` made
        absolute, or ``None`` for a run that has not completed and for a calibration run, which
        trains nothing."""
        if self.final is None or self.state != "completed" or self.record["config"] is None:
            return None
        checkpoint = self.final["checkpoint"]
        return {**checkpoint, "path": str(self.directory / checkpoint["path"])}


def observe(directory: Path, record_file: str = RUN_FILE) -> RunObservation:
    """Read the run or sweep at ``directory`` once. Its state is the one its final status names
    once one is written; before that ``running`` while its latest sign of life is within
    :data:`HEARTBEAT_STALE_SECONDS`, else ``interrupted``. A run (``record_file`` its ``run.json``)
    holds its two logs from the moment it opened; one missing either refuses with
    ``FileNotFoundError`` naming each. A completed run's checkpoint is read, and refused when
    absent, by whatever loads it: an archive may carry a run without its weights."""
    record = read_record(directory / record_file)
    final_path = directory / FINAL_STATUS_FILE
    final = read_record(final_path) if final_path.exists() else None
    alive = last_alive(directory, record_file)
    if final is not None:
        state = final["state"]
    else:
        state = "running" if time.time() - alive <= HEARTBEAT_STALE_SECONDS else "interrupted"
    observation = RunObservation(directory, record, final, alive, state)
    if record_file == RUN_FILE:
        missing = [str(path) for path in (observation.metrics_log, observation.validations_log)
                   if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"run {directory.name} ({state}) is missing "
                                    f"{', '.join(missing)}.")
    return observation


def run_dirs(root: Path | str | None = None) -> list[Path]:
    """Every run directory of the project under ``root`` (default: the pinned platform root) that
    holds a launch record, sorted by id."""
    parent = experiments_dir(root)
    if not parent.is_dir():
        return []
    return sorted(d for d in parent.iterdir() if (d / RUN_FILE).is_file())


def run_observations(root: Path | str | None = None) -> list[RunObservation]:
    """Every run of the project under ``root`` (:func:`run_dirs`), each observed once."""
    return [observe(d) for d in run_dirs(root)]


def training_runs(root: Path | str | None = None) -> list[RunObservation]:
    """Every run of the project under ``root`` that trains a model (:func:`run_observations`); a
    calibration run, whose launch record carries no config, is not one."""
    return [obs for obs in run_observations(root) if obs.record["config"] is not None]


def find_observation(experiment_id: str, *,
                     root: Path | str | None = None) -> RunObservation | None:
    """The run ``experiment_id`` names (:func:`find_run`), observed, or ``None``."""
    run_dir = find_run(experiment_id, root=root)
    return observe(run_dir) if run_dir is not None else None


def run_resolution(experiment_id: str) -> dict:
    """What the run ``experiment_id`` names resolved at launch (:func:`find_observation`): its
    ``data`` section, its ``partition`` and its ``objective``. Refuses (``ValueError``) an id
    naming no run directory and one naming a calibration run, which resolved nothing."""
    observation = find_observation(experiment_id)
    if observation is None:
        raise ValueError(f"no run directory records {experiment_id!r}, so nothing it resolved "
                         "can be read.")
    if observation.record["config"] is None:
        raise ValueError(f"{experiment_id!r} is a calibration run: it trained nothing and "
                         "resolved no data.")
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
    who launched it, its last sign of life, and its best selection value under the objective its
    launch recorded, folded from ``rows`` (:func:`best_selection`), with that objective's metric
    name."""
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
        "launched_by": observation.record["launched_by"],
        "heartbeat": datetime.fromtimestamp(observation.alive, timezone.utc).isoformat(),
    }


def _distinct_epoch_count(rows: list[dict[str, Any]]) -> int:
    """The number of distinct ``epoch`` values among ``rows``, compared as canonical JSON text."""
    return len({json.dumps(row.get("epoch"), sort_keys=True, default=str) for row in rows})


def get_experiment(
    experiment_id: str, *, metrics_limit: int | None = None, metrics_offset: int = 0,
) -> dict[str, Any]:
    """One run's directory read whole: its launch record, its final status (``None`` until
    written), its state, its metrics rows paginated by ``metrics_offset`` and ``metrics_limit``
    over the row list (``n_rows`` bounds the paging, ``n_epochs`` counts distinct epochs), and its
    validations. ``{"error": ...}`` for an id naming no run."""
    observation = find_observation(experiment_id)
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
        "validations": read_rows(observation.validations_log)[0],
    }


def list_experiments() -> list[dict[str, Any]]:
    """Every run directory of the project: its id, state, creation time, and whether it trains a
    model (``has_model_source``: its launch record carries a config, which a calibration run's
    does not)."""
    return [{
        "experiment_id": obs.directory.name,
        "state": obs.state,
        "created": obs.record["created"],
        "has_model_source": obs.record["config"] is not None,
    } for obs in run_observations()]


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


def compare_experiments(experiment_ids: list[str]) -> dict[str, Any]:
    """Side-by-side comparison of training runs.

    Per run: ``state``, ``n_epochs``/``n_rows``, ``last_logged_metrics`` (the log's last row, not
    a verified result), ``rows_after_end`` (rows whose own ``timestamp`` is a later instant than
    the final status's ``ended``; ``None`` before a final status), the final status's ``error`` as
    ``status_error``, ``registry`` (a completed run's own registry entry,
    ``model_registry.run_entry``, with the metrics and source its checkpoint carries,
    ``model_registry.entry_facts``, as a one-entry list, empty otherwise), ``split``
    (:func:`_split_summary`), and the builder, task, subject and dataset identity its records
    carry. An id naming no run, or a calibration run, is an entry carrying only ``error``.
    ``same_dataset_fingerprint`` is ``None`` when any entry is an error or any fingerprint is
    unset, else whether every run names one fingerprint.
    """
    from tcip_mcp.model_registry import entry_facts, run_entry
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.model_build import MODEL_SOURCE_KEY, run_task

    comparisons: list[dict[str, Any]] = []
    for eid in experiment_ids:
        observation = find_observation(eid)
        if observation is None:
            comparisons.append({"experiment_id": eid, "error": f"Experiment not found: {eid}"})
            continue
        run, final = observation.record, observation.final
        config = run["config"]
        if config is None:
            comparisons.append({"experiment_id": eid,
                                "error": f"{eid} is a calibration run; it trained nothing"})
            continue
        metrics = read_rows(observation.metrics_log)[0]
        summary: dict[str, Any] = {
            "experiment_id": eid, "state": observation.state,
            "n_epochs": _distinct_epoch_count(metrics), "n_rows": len(metrics),
        }
        if metrics:
            summary["last_logged_metrics"] = metrics[-1]
        rows_after_end = None
        if final is not None:
            ended = datetime.fromisoformat(final["ended"])
            rows_after_end = sum(
                1 for row in metrics
                if isinstance(row.get("timestamp"), str)
                and datetime.fromisoformat(row["timestamp"]) > ended
            )
        summary["rows_after_end"] = rows_after_end
        summary["status_error"] = final["error"] if final is not None else None
        entry = run_entry(observation)
        summary["registry"] = [] if entry is None else [{
            "name": entry["name"], "registered_at": entry["registered_at"],
            **{k: v for k, v in entry_facts(entry).items() if k != "kind"},
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


def get_experiment_lineage(experiment_id: str) -> dict[str, Any]:
    """A training run's data-to-model chain read off its records: the data section it resolved,
    its dataset identity, the run it was relaunched from and the checkpoint it resumed from, and
    the checkpoint its completion selected. ``{"error": ...}`` for an id naming no training
    run."""
    observation = find_observation(experiment_id)
    if observation is None:
        return {"error": f"Experiment not found: {experiment_id}"}
    run = observation.record
    if run["config"] is None:
        return {"error": f"{experiment_id} is a calibration run; it trained nothing"}
    return {"experiment_id": experiment_id, "lineage": {
        "data": run["resolved"]["data"],
        "dataset_id": run["dataset"]["id"],
        "dataset_fingerprint": run["dataset"]["fingerprint"],
        "parent_experiment": run["parent_experiment"],
        "resume_from": run["resume_from"],
        "checkpoint": observation.checkpoint,
    }}


# ── validations ──────────────────────────────────────────────────────────────


_VALIDATION_FIELDS = (
    "document",
    "trait",
    "claim",
    "validated_against",
    "checkpoint_sha256",
    "producing_experiment_id",
    "reference_identity",
    "covered_buckets",
    "dataset_root",
    "recorded_at",
    "train_disjointness",
    "selection_disjointness",
)
"""Every field a validation row carries, all required.

``train_disjointness`` is ``{"checked": bool, "group_check": str | None}`` for the four documents
whose gate runs the check, or ``null`` for ``resolve_scale``. ``selection_disjointness`` is the
same two facts plus ``applicable``/``reason``, ``null`` for ``resolve_scale``.
"""


def _content_digest(value: dict[str, Any]) -> str:
    """A mapping's content identity: sha256 over its canonical JSON, first 16 hex characters."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def validation_digest(body: dict[str, Any]) -> str:
    """The content identity of a validation row, recomputed from the row a reader holds."""
    return _content_digest(body)


def append_validation(run_dir: Path, body: dict[str, Any]) -> str:
    """Append one earned claim to ``run_dir``'s validations and record the append, returning the
    row's :func:`validation_digest`. Refuses (``ValueError``) a row missing any of
    :data:`_VALIDATION_FIELDS` and a directory holding no run."""
    missing = [field for field in _VALIDATION_FIELDS if field not in body]
    if missing:
        raise ValueError(f"Validation record is missing {', '.join(missing)}; every field of a "
                         "record is required and none has a default.")
    if not (run_dir / RUN_FILE).is_file():
        raise ValueError(f"Experiment not found: {run_dir.name}")
    append_row(run_dir / VALIDATIONS_FILE, body)
    digest = validation_digest(body)
    from tcip_mcp.audit import record_event_or_raise

    record_event_or_raise("experiment_validation_recorded",
                          {"experiment_id": run_dir.name, "document": body["document"],
                           "trait": body["trait"], "record_digest": digest})
    return digest


def find_validation(observation: RunObservation, digest: str) -> dict[str, Any] | None:
    """The row of the observed run's validations whose own recomputed identity is ``digest``, or
    ``None`` when no row has it."""
    for row in read_rows(observation.validations_log)[0]:
        if validation_digest(row) == digest:
            return row
    return None


def open_calibration_run(calibrated: dict[str, Any]) -> Path:
    """Open a fresh run directory (:func:`open_run_directory`) for one calibration of a checkpoint
    no run of this project produced, and record its creation. Its launch record carries no config
    and ``calibrated``: the ``document``, ``checkpoint_sha256``, ``reference_identity`` and
    ``trait`` the claim is earned for, and what it was ``derived_from``."""
    run_dir = experiment_dir(f"calibration_{mint_experiment_id()}")
    open_run_directory(run_dir, lambda _: {"created": now_iso(), "config": None,
                                           "launched_by": None, "calibrated": calibrated})
    from tcip_mcp.audit import record_event_or_raise

    record_event_or_raise("calibration_experiment_created",
                          {"experiment_id": run_dir.name, "document": calibrated["document"],
                           "trait": calibrated["trait"]})
    return run_dir
