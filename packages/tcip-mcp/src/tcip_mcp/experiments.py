"""Experiment tracking for ML training runs.

An experiment is one run's immutable record: named by the caller before the run or minted at
launch, and nothing groups runs into anything larger. A relaunch of a record that already has
history forks a new record instead of reopening it, its ``parent_experiment`` naming the one it
forked from. ``experiment_id`` is that record's id wherever a store key, a tool parameter, a route
path or a delivered column carries the term.

Stores experiment state in .tcip/experiments/<experiment_id>/:
  config.json, full training config snapshot
  metrics.jsonl, epoch-by-epoch metrics (append-only)
  artifacts.json, pointers to model weights, predictions
  lineage.json, data → model → predictions chain
  status.json, current state and timestamps
  split.json, the train/val membership, seed, capture date and dataset identity a metric is
    reproducible with (plus a bound run's calibration-side counts, held out from both)
  env.json, the library versions, seed and model kind behind a reproducible run
  validations.jsonl, the claims earned against this run's evidence (append-only)

This module declares the record's members and their key constructors; ``experiment_dir`` serves the
run artifacts that live beside the record without being members of it (checkpoints, TensorBoard
logs, a bespoke run's source snapshot).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any

from tcip_store import (
    LOG_JSON,
    RECORD_JSON,
    BadKey,
    DecodeError,
    Key,
    StoreDescriptor,
    Version,
    VersionConflict,
    check_json_value,
    register_store,
    store,
)
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.project_paths import resolve_state

logger = logging.getLogger(__name__)

# Relative default (tests rebind this constant). Consumers must go through
# ``experiments_dir()`` so the store anchors to ``$TCIP_STATE_ROOT`` when pinned (no
# subdir fragmentation) while a rebound absolute path / unpinned cwd still work.
EXPERIMENTS_DIR = Path(".tcip/experiments")


def experiments_dir() -> Path:
    """The experiment store, resolved against the pinned platform root at use time."""
    return resolve_state(EXPERIMENTS_DIR)


def experiment_dir(experiment_id: str) -> Path:
    """One experiment's directory: the record's members plus the run artifacts beside them.

    Serves the files the record's own layout does not name: weights, TensorBoard event files, and
    the per-file source snapshot a bespoke run copies in.
    """
    return experiments_dir() / experiment_id


# ── the experiment stores ────────────────────────────────────────────────────

_MEMBER_DOC = RootedFileLocator(suffix=".json")
"""One member document inside its experiment's directory."""

_MEMBER_LOG = RootedFileLocator(suffix=".jsonl")
"""One append-only member log inside its experiment's directory."""


def experiments_scope(root: Path | str | None = None) -> str:
    """The root every experiment key hangs off: the experiment store, made absolute.

    Resolved per call because ``EXPERIMENTS_DIR`` is relative until a platform root is pinned, and
    a pin can land mid-process. ``root`` names a different platform root than this process's own.
    """
    if root is None:
        return str(experiments_dir().resolve())
    return str((Path(root) / EXPERIMENTS_DIR).resolve())


def _member_key(store_name: str, experiment_id: str, document: str, root: Path | str | None) -> Key:
    if "/" in experiment_id or "\\" in experiment_id or experiment_id in ("", ".", ".."):
        raise BadKey(
            f"experiment id {experiment_id!r} is not a single name: an id carrying a path "
            "separator would address a record outside the experiment store"
        )
    return Key(store_name, experiments_scope(root), (experiment_id, document))


EXPERIMENT_CONFIG_STORE = "experiment_config"
register_store(
    StoreDescriptor(
        name=EXPERIMENT_CONFIG_STORE,
        kind="record",
        key_fields=("experiment_id", "document"),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        enumerable=True,
        locator=_MEMBER_DOC,
    )
)


def config_key(experiment_id: str, *, root: Path | str | None = None) -> Key:
    """The config snapshot a run trained under.

    ``last_writer_wins``: it is written whole at creation and replaced only while the record
    is still pristine, never merged into.
    """
    return _member_key(EXPERIMENT_CONFIG_STORE, experiment_id, "config", root)


EXPERIMENT_STATUS_STORE = "experiment_status"
register_store(
    StoreDescriptor(
        name=EXPERIMENT_STATUS_STORE,
        kind="record",
        key_fields=("experiment_id", "document"),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="cas",
        enumerable=True,
        locator=_MEMBER_DOC,
    )
)


STATUS_DOCUMENT = "status"


def status_key(experiment_id: str, *, root: Path | str | None = None) -> Key:
    """The run's state, timestamps, liveness heartbeat and launcher declaration.

    ``cas``: every writer here reads the document and updates fields inside it, from the training
        subprocess and the tool process at once.

    ``launched_by`` is a mapping naming who launched the run, stamped by
    :func:`stamp_run_identity`: ``{"launcher": "gui"}`` for a launch through the web app's own
    route, ``{"launcher": "agent", **agent_identity.audit_fields()}`` for a launch inside an MCP
    handshake, ``{"launcher": "process"}`` for a launch from neither. Absent on a record whose
    experiment tracking never reached the stamp, or whose stamp otherwise failed.
    """
    return _member_key(EXPERIMENT_STATUS_STORE, experiment_id, STATUS_DOCUMENT, root)


EXPERIMENT_LINEAGE_STORE = "experiment_lineage"
register_store(
    StoreDescriptor(
        name=EXPERIMENT_LINEAGE_STORE,
        kind="record",
        key_fields=("experiment_id", "document"),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="cas",
        enumerable=True,
        locator=_MEMBER_DOC,
    )
)


def lineage_key(experiment_id: str, *, root: Path | str | None = None) -> Key:
    """The data to model to predictions chain, written compare-and-set."""
    return _member_key(EXPERIMENT_LINEAGE_STORE, experiment_id, "lineage", root)


EXPERIMENT_ARTIFACTS_STORE = "experiment_artifacts"
register_store(
    StoreDescriptor(
        name=EXPERIMENT_ARTIFACTS_STORE,
        kind="record",
        key_fields=("experiment_id", "document"),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="cas",
        enumerable=True,
        locator=_MEMBER_DOC,
    )
)


def artifacts_key(experiment_id: str, *, root: Path | str | None = None) -> Key:
    """The run's artifact pointers, written compare-and-set."""
    return _member_key(EXPERIMENT_ARTIFACTS_STORE, experiment_id, "artifacts", root)


EXPERIMENT_ENV_STORE = "experiment_env"
register_store(
    StoreDescriptor(
        name=EXPERIMENT_ENV_STORE,
        kind="record",
        key_fields=("experiment_id", "document"),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        enumerable=True,
        locator=_MEMBER_DOC,
    )
)


def env_key(experiment_id: str, *, root: Path | str | None = None) -> Key:
    """The environment capture behind a reproducible run, last writer wins."""
    return _member_key(EXPERIMENT_ENV_STORE, experiment_id, "env", root)


EXPERIMENT_SPLIT_STORE = "experiment_split"
register_store(
    StoreDescriptor(
        name=EXPERIMENT_SPLIT_STORE,
        kind="record",
        key_fields=("experiment_id", "document"),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        enumerable=True,
        locator=_MEMBER_DOC,
    )
)


def split_key(experiment_id: str, *, root: Path | str | None = None) -> Key:
    """The train/val membership, seed and dataset identity of this run, last writer wins."""
    return _member_key(EXPERIMENT_SPLIT_STORE, experiment_id, "split", root)


EXPERIMENT_METRICS_STORE = "experiment_metrics"
register_store(
    StoreDescriptor(
        name=EXPERIMENT_METRICS_STORE,
        kind="log",
        key_fields=("experiment_id", "document"),
        frozen=True,
        codec=LOG_JSON,
        enumerable=True,
        locator=_MEMBER_LOG,
    )
)


def metrics_key(experiment_id: str, *, root: Path | str | None = None) -> Key:
    """The run's epoch-by-epoch metrics, one entry per row, append only."""
    return _member_key(EXPERIMENT_METRICS_STORE, experiment_id, "metrics", root)


EXPERIMENT_VALIDATIONS_STORE = "experiment_validations"
register_store(
    StoreDescriptor(
        name=EXPERIMENT_VALIDATIONS_STORE,
        kind="log",
        key_fields=("experiment_id", "document"),
        frozen=True,
        codec=LOG_JSON,
        enumerable=True,
        locator=_MEMBER_LOG,
    )
)


def validations_key(experiment_id: str, *, root: Path | str | None = None) -> Key:
    """The claims earned against this run's evidence, one row per claim, append only.

    A re-validation appends rather than rewriting, so the member holds the whole history and
    a stamp names the one row it was minted from.
    """
    return _member_key(EXPERIMENT_VALIDATIONS_STORE, experiment_id, "validations", root)


# Once terminal, a record is immutable and additive-only. Excludes "canceled": that record
# stays writable; a resume always mints a fresh id via _ensure_experiment rather than reopening it.
_TERMINAL_STATES = {"completed", "failed"}

# A different concept sharing similar vocabulary, states reconstruct_run_status trusts as
# already-decided and never re-derives from heartbeat freshness. Unlike _TERMINAL_STATES above,
# this does include "canceled": a gracefully canceled run recorded its own final state honestly
# (model_final.pt was written, cancel_training's own documented contract), and re-deriving it from
# heartbeat staleness would misreport it as "running" then permanently as "interrupted", implying
# a crash that never happened. Named separately rather than reusing _TERMINAL_STATES so the two
# purposes (mutation-lock vs. heartbeat-reconstruction) can never silently drift onto each other.
_RECORDED_AS_DONE = {"completed", "failed", "canceled"}


def read_member(key: Key, default: Any = None) -> Any:
    """One member document, with an unreadable record folded onto ``default``."""
    try:
        return store.read(key, default=default)
    except DecodeError:
        logger.warning("experiment member %s does not decode", list(key.parts), exc_info=True)
        return default


def experiment_exists(experiment_id: str, *, root: Path | str | None = None) -> bool:
    """Whether this id names a real experiment record, by its config snapshot.

    ``root`` defaults to the current platform root; a caller resolving a run under a root other
    than the one it started under passes the launch root explicitly.
    """
    return store.exists(config_key(experiment_id, root=root))


def recorded_state(status: Mapping[str, Any]) -> str:
    """The ``state`` a status record states, which :func:`create_experiment` writes on every one."""
    return status["state"]


def _current_state(experiment_id: str, *, root: Path | str | None = None) -> str | None:
    """The experiment's recorded ``state``, or ``None`` when no status record exists for it."""
    status = read_member(status_key(experiment_id, root=root), None)
    return None if status is None else recorded_state(status)


class ExperimentTerminal(RuntimeError):
    """A write reached an experiment record already in a terminal state (completed/failed;
    canceled stays resumable and is never terminal here).
    """


def refuse_if_terminal(experiment_id: str, op: str, state: str | None) -> None:
    """Raise :class:`ExperimentTerminal` if ``state`` is terminal.

    ``state`` is the value the caller already read, inside its own transaction when it holds one
    (so the check and the write it guards see the same value) or via :func:`_current_state` when it
    doesn't.
    """
    if state in _TERMINAL_STATES:
        raise ExperimentTerminal(f"Experiment {experiment_id} is {state} (terminal); refusing to {op}.")


def rewrite_live_member(experiment_id: str, key: Any, op: str,
                        update: Callable[[Any], Any]) -> None:
    """Rewrite one member of a live experiment record inside its status lock: ``update`` takes the
    member's stored value (``None`` when absent) and returns what is written.

    Raises :class:`ExperimentTerminal` naming ``op`` for a terminal record, and whatever the store
    raises on any other read or write failure.
    """
    st_key = status_key(experiment_id)
    with store.transaction(key, st_key) as txn:
        refuse_if_terminal(experiment_id, op, recorded_state(txn.read(st_key)))
        txn.write(key, update(txn.read(key, default=None)))


def mint_experiment_id() -> str:
    """A fresh, unclaimed experiment id: ``run_<epoch-seconds>_<6 hex chars>``.

    The uuid suffix, not a counter, so two ids minted in the same process, or in two different
    processes sharing one experiment store, never collide on the same clock second.
    """
    return f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"


def create_experiment(
    experiment_id: str,
    config: dict[str, Any],
    *,
    parent_experiment: str | None = None,
    data_source: str | None = None,
    dataset_id: str | None = None,
    dataset_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Create a new experiment record with its config snapshot.

    The config snapshot is written create-only, so an id that already names an experiment is
    refused inside the write's own lock.

    ``dataset_id`` / ``dataset_fingerprint`` record the identity of the data this run trained on,
    written into the immutable lineage at creation. They are set once here and never via
    ``update_lineage``.

    The config is checked against what JSON can hold before the write, naming the offending field.
    """
    check_json_value(config, path="config")
    try:
        store.replace(config_key(experiment_id), config, expect=Version.ABSENT)
    except VersionConflict:
        return {"error": f"Experiment already exists: {experiment_id}"}

    status = {
        "state": "created",
        "created": datetime.now(timezone.utc).isoformat(),
        "started": None,
        "ended": None,
    }
    store.replace(status_key(experiment_id), status, expect=Version.ABSENT)

    lineage = {
        "data_source": data_source,
        "dataset_id": dataset_id,
        "dataset_fingerprint": dataset_fingerprint,
        "parent_experiment": parent_experiment,
        "model_weights": None,
        "model_weights_sha256": None,
        "predictions": None,
    }
    store.replace(lineage_key(experiment_id), lineage, expect=Version.ABSENT)
    store.replace(artifacts_key(experiment_id), {}, expect=Version.ABSENT)

    return {
        "experiment_id": experiment_id,
        "path": str(experiment_dir(experiment_id)),
        "state": "created",
    }


def is_pristine(state: str | None, metrics_logged: bool) -> bool:
    """Whether an experiment record with this ``state`` and ``metrics_logged`` was never launched,
    and so may still take a full ``config.json`` rewrite: ``state == "created"`` and no metrics
    logged yet.
    """
    return state == "created" and not metrics_logged


def metrics_logged_of(status: dict[str, Any] | None) -> bool:
    """Whether a status record carries the ``metrics_logged`` marker :func:`log_metrics` stamps."""
    return bool(status.get("metrics_logged")) if isinstance(status, dict) else False


def overwrite_config_if_pristine(
    experiment_id: str, config: dict[str, Any], *, root: Path | str | None = None,
) -> dict[str, Any]:
    """Rewrite ``config.json`` with the config actually launched, but only while the experiment is
    still pristine (state == "created" and no epochs logged yet).

    Refuses once :func:`is_pristine` says the record is no longer pristine. Both of its inputs,
    ``state`` and ``metrics_logged``, are read from the one status record this opens a transaction
    over.

    ``root`` names a platform root other than this process's own.
    """
    check_json_value(config, path="config")
    if not experiment_exists(experiment_id, root=root):
        return {"error": f"Experiment not found: {experiment_id}"}
    cfg_key, st_key = config_key(experiment_id, root=root), status_key(experiment_id, root=root)
    state: str | None = None
    metrics_logged = False
    refused = False
    with store.transaction(cfg_key, st_key) as txn:
        status = txn.read(st_key)
        state = recorded_state(status)
        metrics_logged = metrics_logged_of(status)
        refused = not is_pristine(state, metrics_logged)
        if not refused:
            txn.write(cfg_key, config)
    if refused:
        return {"error": f"Experiment {experiment_id} is no longer pristine; refusing to "
                         f"overwrite its config.json."}
    return {"experiment_id": experiment_id, "overwritten": True}


def _mark_completed(status: dict[str, Any]) -> None:
    """Write the completed state, heartbeat and ended timestamp into a status record in place."""
    now = datetime.now(timezone.utc).isoformat()
    status["state"] = "completed"
    status["heartbeat"] = now
    status["ended"] = now


def update_status(
    experiment_id: str,
    state: str,
    *,
    error: str | None = None,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Update experiment state (created → running → completed | failed).

    A repeat of the record's current state is idempotent: nothing restamps (not ``heartbeat``, not
    ``ended``), and ``error`` lands only when the record does not already carry one, so a second
    reason never overwrites a first. Any other write to a terminal record
    (``completed``/``failed``, the other terminal state included) refuses through
    :func:`refuse_if_terminal`. ``canceled`` is not terminal here, so a record in that state still
    takes any write, including back to ``running``.

    ``error`` records a specific failure reason (e.g. a wall-clock-timeout kill) into
    ``status.json["error"]``; outside the idempotent-repeat case above, omitted/``None`` never
    clears a previously-recorded error, only an explicit new value overwrites it.

    ``root`` defaults to the current platform root; a caller passes another to reach a run's own
    record under a root other than this process's.
    """
    if not experiment_exists(experiment_id, root=root):
        return {"error": f"Experiment not found: {experiment_id}"}

    key = status_key(experiment_id, root=root)
    current: str | None = None
    refused = False
    with store.transaction(key) as txn:
        status = txn.read(key)
        current = recorded_state(status)
        if state == current:
            if error is not None and status.get("error") is None:
                status["error"] = error
                txn.write(key, status)
        else:
            try:
                refuse_if_terminal(experiment_id, "update_status", current)
            except ExperimentTerminal:
                refused = True

            if not refused:
                if state == "completed":
                    _mark_completed(status)
                else:
                    status["state"] = state
                    now = datetime.now(timezone.utc).isoformat()
                    status["heartbeat"] = now  # liveness stamp: a fresh heartbeat means a live process
                    if state == "running" and not status.get("started"):
                        status["started"] = now
                    if state == "failed":
                        status["ended"] = now
                if error is not None:
                    status["error"] = error

                txn.write(key, status)

    if refused:
        return {"error": f"Experiment {experiment_id} is {current} (terminal); refusing to "
                         f"move it to {state!r}.", "state": current}
    return {"experiment_id": experiment_id, "state": state}


def complete_run(
    experiment_id: str, final_weights: str, *, root: Path | str | None = None,
) -> dict[str, Any]:
    """Mark a run completed and record its final weights pointer and their digest, as one
    transaction.

    Hashes ``final_weights`` before opening the transaction. A missing or unreadable file returns
    an error naming the path and the read failure, and writes nothing (no transaction is even
    opened).

    Names the artifacts key before the lineage key, and the lineage key before the status key: a
    file-backend transaction applies its writes in named-key order and is not crash-atomic across
    keys, so a crash mid-apply leaves a detectably stale record (a pointer with no digest, or a
    digest recorded on a record still ``running``), never a ``completed`` record carrying a
    mismatched or absent digest. Refuses a run already terminal, naming the weights file; the
    refusal's ``state`` carries the state the record actually holds.

    ``root`` names a platform root other than this process's own.
    """
    if not experiment_exists(experiment_id, root=root):
        return {"error": f"Experiment not found: {experiment_id}"}

    from tcip_mcp.model_registry import _compute_sha256

    try:
        digest = _compute_sha256(final_weights)
    except OSError as exc:
        return {"error": f"complete_run: final_weights {final_weights!r} could not be read "
                         f"({exc}); refusing to complete with an unrecorded digest.",
                "final_weights": final_weights}

    art_key, lin_key, st_key = (
        artifacts_key(experiment_id, root=root), lineage_key(experiment_id, root=root),
        status_key(experiment_id, root=root),
    )
    current: str | None = None
    try:
        with store.transaction(art_key, lin_key, st_key) as txn:
            status = txn.read(st_key)
            current = recorded_state(status)
            refuse_if_terminal(experiment_id, "complete_run", current)

            recorded_at = datetime.now(timezone.utc).isoformat()
            artifacts = txn.read(art_key, default={})
            artifacts["model_weights"] = {
                "path": final_weights, "sha256": digest, "recorded": recorded_at,
            }
            txn.write(art_key, artifacts)

            lineage = txn.read(lin_key, default={})
            lineage["model_weights"] = final_weights
            lineage["model_weights_sha256"] = digest
            txn.write(lin_key, lineage)

            _mark_completed(status)
            txn.write(st_key, status)
    except ExperimentTerminal as exc:
        return {"error": f"{exc} Final weights at {final_weights!r} were not recorded.",
                "final_weights": final_weights, "state": current}

    return {"experiment_id": experiment_id, "state": "completed", "model_weights": final_weights,
            "model_weights_sha256": digest}


class StampPreconditionFailed(RuntimeError):
    """Raised by :func:`stamp_run_identity` when the record is not a fresh, unstamped one: its
    state is not ``"created"``, it already carries an ``output_dir``, or (when the stamp was given
    a config to write) it already carries a metrics row.
    """


def stamp_run_identity(
    experiment_id: str, output_dir: str, *, launched_by: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> None:
    """Stamp a launch onto this experiment's ``status.json``, moving it to ``running`` in the same
    write: ``output_dir`` (the real, caller-influenced artifact directory), ``launched_by`` (who
    launched it), and the ``state``/``heartbeat``/``started`` triple :func:`update_status` writes
    for that transition.

    One compare-and-set transaction, requiring ``state == "created"``, no ``output_dir`` already
    stamped, and, when ``config`` is given, no metrics logged yet. Raises
    :class:`StampPreconditionFailed` when that precondition fails, so two launches racing to stamp
    one pristine record never both win it, and raises whatever the store itself raises on any other
    write failure.

    ``config``, given, makes the transaction also span ``config_key``, written before
    ``status_key`` (a file-backend transaction applies its writes in the order its keys were
    named), so the config actually launched lands in the pre-created record's snapshot in the same
    write as the stamp. Omitted, the transaction spans ``status_key`` alone.
    """
    key = status_key(experiment_id)
    cfg_key = config_key(experiment_id) if config is not None else None
    keys = (cfg_key, key) if cfg_key is not None else (key,)
    with store.transaction(*keys) as txn:
        status = txn.read(key)
        logged = config is not None and metrics_logged_of(status)
        if not is_pristine(recorded_state(status), logged) or status.get("output_dir"):
            raise StampPreconditionFailed(
                f"experiment {experiment_id!r} is not a fresh, unstamped record "
                f"(state={status.get('state')!r}, output_dir={status.get('output_dir')!r}); "
                "another launch has already claimed it."
            )
        if cfg_key is not None:
            txn.write(cfg_key, config)
        now = datetime.now(timezone.utc).isoformat()
        status["output_dir"] = output_dir
        status["launched_by"] = launched_by
        status["state"] = "running"
        status["heartbeat"] = now
        status["started"] = now
        txn.write(key, status)


def experiment_ids_with_status(root: Path | str | None = None) -> list[str]:
    """Every experiment id the store holds a status record for, sorted."""
    found = store.keys(EXPERIMENT_STATUS_STORE, experiments_scope(root))
    return sorted(key.parts[0] for key in found if key.parts[1] == STATUS_DOCUMENT)


def derived_state(status: dict[str, Any], stale_seconds: float) -> str:
    """The state a status record reads as once heartbeat freshness applies: a state already
    recorded as done (:data:`_RECORDED_AS_DONE`) is trusted as-is; any other state derives to
    ``"running"`` while the heartbeat is fresh, else ``"interrupted"``.
    """
    state = recorded_state(status)
    heartbeat = status.get("heartbeat")
    if state not in _RECORDED_AS_DONE:
        state = "running" if _heartbeat_fresh(heartbeat, stale_seconds) else "interrupted"
    return state


def reconstruct_from_status(
    experiment_id: str, status: dict[str, Any], *, stale_seconds: float, read_progress: bool,
) -> dict[str, Any]:
    """One record's run row, reconstructed from a status document the caller already read.
    ``current_epoch``, ``best_metric`` and ``best_metric_name`` cost one metrics-log read and are
    included only when ``read_progress`` is true, read back through :func:`best_selection_from_log`
    from what the run itself stamped.

    ``launched_by`` carries the record's own stamped declaration (see :func:`stamp_run_identity`)
    whole, or ``None`` for a record never launched.

    ``heartbeat`` carries the record's own last-touched instant (see :func:`_touch_heartbeat`)
    whole, the same value :func:`derived_state` reads to decide ``running`` vs ``interrupted``; no
    process id is persisted.
    """
    current_epoch = None
    best_metric_name = None
    best_metric = None
    if read_progress:
        rows = read_metrics(experiment_id)
        current_epoch = rows[-1].get("epoch") if rows else None
        best_metric_name, best_metric = best_selection_from_log(rows)
    return {
        "experiment_id": experiment_id,
        "status": derived_state(status, stale_seconds),
        "current_epoch": current_epoch,
        "best_metric": best_metric,
        "best_metric_name": best_metric_name,
        "output_dir": status.get("output_dir"),
        "error": status.get("error"),
        "launched_by": status.get("launched_by"),
        "heartbeat": status.get("heartbeat"),
    }


def reconstruct_run_status(
    experiment_id: str, *, stale_seconds: float,
) -> dict[str, Any] | None:
    """Reconstruct one experiment's status from disk; ``None`` for an absent or unaddressable id (a
    path separator, an empty or dot name). ``stale_seconds`` is the heartbeat window.
    """
    try:
        key = status_key(experiment_id)
    except BadKey:
        return None
    status = read_member(key)
    if not isinstance(status, dict):
        return None
    return reconstruct_from_status(experiment_id, status, stale_seconds=stale_seconds,
                                   read_progress=True)


def _parse_iso_instant(value: Any) -> datetime | None:
    """One timestamp read as an instant: a non-string, or a string ``datetime.fromisoformat`` can't
    parse (a trailing ``Z`` and an offset both parse fine), is ``None`` rather than a raise. A
    naive value is read as UTC.
    """
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _heartbeat_fresh(hb_iso: str | None, stale_seconds: float) -> bool:
    """True if ``hb_iso`` (ISO-8601) is within the staleness window, a process is still actively
    updating this run. Missing/unparseable → not fresh (treat as dead).
    """
    hb = _parse_iso_instant(hb_iso)
    if hb is None:
        return False
    return (datetime.now(timezone.utc) - hb).total_seconds() <= stale_seconds


def _touch_heartbeat(experiment_id: str, *, root: Path | str | None = None) -> None:
    """Best-effort: stamp the current time into ``status.json['heartbeat']``.

    Called each epoch so a run still training in another process reads as live. Never raises, a
    heartbeat failure must not break metric logging.
    """
    key = status_key(experiment_id, root=root)
    if not store.exists(key):
        return
    try:
        with store.transaction(key) as txn:
            status = txn.read(key, default={})
            status["heartbeat"] = datetime.now(timezone.utc).isoformat()
            txn.write(key, status)
    except Exception:
        pass


def read_metrics(experiment_id: str, *, root: Path | str | None = None) -> list[dict[str, Any]]:
    """Every epoch row this run has logged, in order, oldest first.

    An entry still being appended when the log is read is left for the next read rather than
    returned half-formed, so a live run's tail is never served as a truncated row.
    """
    page = store.read_log(metrics_key(experiment_id, root=root))
    if page.corrupt:
        logger.warning("experiment %s metrics log has %d undecodable entries",
                       experiment_id, len(page.corrupt))
    if page.version_refused:
        logger.warning("experiment %s metrics log has %d entries at a schema_version this "
                       "reader does not accept", experiment_id, len(page.version_refused))
    return [dict(record) for record in page.records]


def best_selection_from_log(rows: list[dict[str, Any]]) -> tuple[str | None, float | None]:
    """The selection metric name and its best value, read from a run's own metrics-log rows.

    Every row a training body writes carries ``selection_metric`` (the bare name
    ``generic_trainer.train()`` resolved once, before the first epoch) and ``selection`` (that
    epoch's value on it); this reads those back.

    The name is the most recently stamped one; the best is compared over every row stamped with
    that name in its declared ranking direction (``evaluation.HIGHER_IS_BETTER_BY_METRIC`` on the
    bare name). A row with no name or a non-finite value is skipped when finding the best. No name
    in any row, or a name the declaration table does not carry, leaves both ``None``.
    """
    name: str | None = None
    for row in reversed(rows):
        candidate = row.get("selection_metric")
        if isinstance(candidate, str) and candidate:
            name = candidate
            break
    if name is None:
        return None, None

    from tcip_mcp.pipelines.training.evaluation import HIGHER_IS_BETTER_BY_METRIC

    higher_is_better = HIGHER_IS_BETTER_BY_METRIC.get(name)
    if higher_is_better is None:
        return None, None

    best: float | None = None
    for row in rows:
        if row.get("selection_metric") != name:
            continue
        value = row.get("selection")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            continue
        if best is None or (value > best if higher_is_better else value < best):
            best = float(value)
    return name, best


def log_metrics(
    experiment_id: str,
    epoch: int,
    metrics: dict[str, Any],
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Append epoch metrics to the run's metrics log and refresh its liveness heartbeat.

    A bespoke loop's row is its own dict, so it is checked field by field first: a tensor or a
    non-finite loss is named here.

    Stamps ``status.json["metrics_logged"] = True`` before appending, so a marker written but then
    an append that fails still reads non-pristine, never the reverse. A failure inside that marker
    transaction raises.

    ``root`` names a platform root other than this process's own.
    """
    check_json_value(metrics, path="metrics")
    if not experiment_exists(experiment_id, root=root):
        return {"error": f"Experiment not found: {experiment_id}"}

    # Terminal-state lock: a completed/failed run's metric history is frozen, no new epochs.
    try:
        refuse_if_terminal(experiment_id, "log_metrics", _current_state(experiment_id, root=root))
    except ExperimentTerminal as exc:
        return {"error": str(exc)}

    key = status_key(experiment_id, root=root)
    with store.transaction(key) as txn:
        status = txn.read(key, default={})
        status["metrics_logged"] = True
        txn.write(key, status)

    entry = {
        "epoch": epoch,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }
    store.append(metrics_key(experiment_id, root=root), entry)
    _touch_heartbeat(experiment_id, root=root)

    return {"experiment_id": experiment_id, "epoch": epoch, "logged": True}


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
"""Every field a validation row carries. All required, none defaulted.

A claim reads as provenance only when all of it is there: what was claimed, for which trait,
against which reference, produced by which checkpoint and run, over which content, and when.
A field a writer could omit would be a field a reader could not compare.

``train_disjointness`` is ``{"checked": bool, "group_check": str | None}`` for the four documents
whose gate runs the check (never a bare ``true`` over a check the gate's own record says did not
run), or ``null`` for ``resolve_scale``, whose gate has no training run to check against.

``selection_disjointness`` is the same two facts plus ``applicable``/``reason``: whether the
checkpoint's own selection side (its ``split.json``'s ``val``) was also checked disjoint from the
reference, applicable only when the calibration named a selection or the checkpoint carries
a ``selection_binding``, ``null`` for ``resolve_scale``.
"""


def _content_digest(value: dict[str, Any]) -> str:
    """A mapping's content identity: sha256 over its canonical JSON, first 16 hex characters."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def validation_digest(body: dict[str, Any]) -> str:
    """The content identity of a validation row.

    Pure: a reader recomputes it from a row it read. It is not stored in the row.
    """
    return _content_digest(body)


def _append_validation(experiment_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Append one earned claim to this experiment's validations log.

    The row is checked against ``_VALIDATION_FIELDS`` first: a missing field is refused, never
    filled in.
    """
    check_json_value(body, path="validation")
    if not experiment_exists(experiment_id):
        return {"error": f"Experiment not found: {experiment_id}"}

    missing = [field for field in _VALIDATION_FIELDS if field not in body]
    if missing:
        return {"error": f"Validation record is missing {', '.join(missing)}; every field of a "
                         f"record is required and none has a default."}

    # No terminal-state check: a validation is a statement made about a run after it ended.
    store.append(validations_key(experiment_id), body)
    digest = validation_digest(body)
    from tcip_mcp.audit import record_event_or_raise

    # Platform log: the row above is already on disk, so a failed append raises rather than
    # leaving a provenance gap nobody is told about.
    record_event_or_raise("experiment_validation_recorded",
                 {"experiment_id": experiment_id, "document": body["document"],
                  "trait": body["trait"], "record_digest": digest})
    return {"experiment_id": experiment_id, "record_digest": digest}


def read_validations(
    experiment_id: str, *, root: Path | str | None = None
) -> list[dict[str, Any]]:
    """Every claim this experiment has earned, in order, oldest first.

    A repeated validation of one claim appends a second row rather than replacing the first,
    so the history is the whole list and a stamp names one row of it.
    """
    page = store.read_log(validations_key(experiment_id, root=root))
    if page.corrupt:
        logger.warning("experiment %s validations log has %d undecodable entries",
                       experiment_id, len(page.corrupt))
    if page.version_refused:
        logger.warning("experiment %s validations log has %d entries at a schema_version "
                       "this reader does not accept", experiment_id, len(page.version_refused))
    return [dict(record) for record in page.records]


def find_validation(
    experiment_id: str, digest: str, *, root: Path | str | None = None
) -> dict[str, Any] | None:
    """The row whose own content identity is ``digest``, or ``None`` when no row has it.

    The identity is recomputed from each stored row rather than read off it, so a row answers
    for the content it actually holds.
    """
    for row in read_validations(experiment_id, root=root):
        if validation_digest(row) == digest:
            return row
    return None


def ensure_calibration_experiment(
    *,
    document: str,
    checkpoint_sha256: str | None,
    reference_identity: dict[str, Any],
    trait: str,
    config: dict[str, Any],
) -> str:
    """The experiment a calibration's claims hang off, created when it does not exist yet.

    The id is derived from the same content that constitutes the claim's identity, so a second
    calibration of the same document, checkpoint, reference and trait resolves to the same
    experiment.

    ``config`` is the free text describing the calibration. The identity fields are written here
    from the arguments the id was derived from, so a config restating one is refused.
    """
    identity = {
        "document": document,
        "checkpoint_sha256": checkpoint_sha256,
        "reference_identity": reference_identity,
        "trait": trait,
    }
    restated = sorted(set(config) & set(identity))
    if restated:
        raise ValueError(
            f"calibration config restates {', '.join(restated)}: those fields are written from "
            "the content the experiment id is derived from, so a second spelling of them could "
            "disagree with the id itself"
        )

    experiment_id = f"calibration_{_content_digest(identity)}"
    # create_experiment's create-only refusal is the existence check; a repeat names this same calibration.
    created = create_experiment(experiment_id, {**identity, **config})
    if "error" not in created:
        from tcip_mcp.audit import record_event_or_raise

        # Platform log: the experiment above is already created, so a failed append raises
        # rather than leaving a provenance gap nobody is told about.
        record_event_or_raise("calibration_experiment_created",
                     {"experiment_id": experiment_id, "document": document, "trait": trait})
    return experiment_id


def _pointer_populated(doc: dict[str, Any], field: str) -> bool:
    """Whether ``field`` already carries a real value in ``doc`` (an artifacts or lineage record):
    present and not an empty placeholder.
    """
    return doc.get(field) not in (None, "", [], {})


def _artifact_write_refused(doc: dict[str, Any], field: str, value: Any) -> bool:
    """The artifacts member's additive lock: a name already present freezes, regardless of
    whether the value recorded under it is itself falsy. Presence, not populated-ness, is the
    lock's baseline semantics here, so a name recorded with a falsy entry stays frozen too."""
    return field in doc


def _lineage_write_refused(doc: dict[str, Any], field: str, value: Any) -> bool:
    """The lineage member's additive lock: a populated field freezes unless the write would record
    the same value it already holds.
    """
    return _pointer_populated(doc, field) and doc.get(field) != value


# One predicate per member, evaluated by pointer_frozen, record_artifact and update_lineage alike
# so the three never carry separate copies of what "frozen" means for that member.
_MEMBER_WRITE_REFUSED = {"artifacts": _artifact_write_refused, "lineage": _lineage_write_refused}
_POINTER_MEMBER_KEYS = {"artifacts": artifacts_key, "lineage": lineage_key}


def pointer_frozen(experiment_id: str, member: str, field: str, value: Any) -> str | None:
    """Whether writing ``value`` into ``field`` of the named member (``"artifacts"`` or
    ``"lineage"``) would be refused right now: the experiment terminal and the member's own
    additive lock (:data:`_MEMBER_WRITE_REFUSED`).

    An untransacted pre-check for a caller about to write a file outside the store; the writer's
    own transactional refusal still decides. Returns the refusal text, or ``None`` when the write
    would be admitted.
    """
    doc = read_member(_POINTER_MEMBER_KEYS[member](experiment_id), {})
    state = _current_state(experiment_id)
    if state in _TERMINAL_STATES and _MEMBER_WRITE_REFUSED[member](
        doc if isinstance(doc, dict) else {}, field, value
    ):
        return (f"Experiment {experiment_id} is {state} (terminal); {member}.{field} is already "
                "recorded and is immutable.")
    return None


def record_artifact(
    experiment_id: str,
    name: str,
    path: str,
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Register an artifact (model weights, predictions, etc.).

    ``root`` names a platform root other than this process's own.
    """
    if not experiment_exists(experiment_id, root=root):
        return {"error": f"Experiment not found: {experiment_id}"}

    key, state = artifacts_key(experiment_id, root=root), status_key(experiment_id, root=root)
    current: str | None = None
    refused_overwrite = False
    with store.transaction(key, state) as txn:
        artifacts = txn.read(key, default={})
        current = recorded_state(txn.read(state))
        # Terminal-state lock (additive-only): a new artifact name may be recorded post-completion,
        # but an existing one is frozen, no silent overwrite of a delivered pointer.
        refused_overwrite = current in _TERMINAL_STATES and _MEMBER_WRITE_REFUSED["artifacts"](
            artifacts, name, path)
        if not refused_overwrite:
            artifacts[name] = {"path": path, "recorded": datetime.now(timezone.utc).isoformat()}
            txn.write(key, artifacts)

    if refused_overwrite:
        try:
            refuse_if_terminal(experiment_id, "record_artifact", current)
        except ExperimentTerminal as exc:
            return {"error": f"{exc} Artifact {name!r} is already recorded and is immutable; "
                             f"the file at {path!r} was not recorded.",
                    "artifact": name}
    return {"experiment_id": experiment_id, "artifact": name, "path": path}


_COMPLETION_ONLY_LINEAGE_FIELDS = ("model_weights", "model_weights_sha256")
"""The two lineage fields only ``complete_run`` writes: the digest completion recorded, and the
path it was taken over. Unlike the identity fields below, naming either here is a caller error,
not a state a still-empty field could legitimately take later, so the whole call refuses before
any field lands, never merged in and never silently dropped."""


def update_lineage(
    experiment_id: str,
    *,
    root: Path | str | None = None,
    **updates: Any,
) -> dict[str, Any]:
    """Update lineage fields (predictions, data_source, review_session, etc.).

    The updates are the caller's own kwargs, merged whole into the stored document, so they are
    checked against what JSON can hold before any of them lands. ``model_weights`` and
    ``model_weights_sha256`` are ``complete_run``'s alone: naming either raises ``ValueError``
    before any field, including a legitimate companion in the same call, lands.

    A field refused (a dataset identity field, or a populated field of a terminal run) is named
    under ``refused`` in the result, and the others land.

    ``root`` names a platform root other than this process's own.
    """
    check_json_value(updates, path="updates")
    completion_fields = sorted(f for f in _COMPLETION_ONLY_LINEAGE_FIELDS if f in updates)
    if completion_fields:
        raise ValueError(
            f"update_lineage: {', '.join(completion_fields)} is complete_run's alone to write; "
            "no other caller populates the run's recorded digest."
        )
    if not experiment_exists(experiment_id, root=root):
        return {"error": f"Experiment not found: {experiment_id}"}

    # Dataset identity is set once at creation and is immutable, never a lineage edge to backfill: the additive-only lock below would otherwise permit a first write to an empty identity field even post-terminal.
    # That write would be a silent change to what data the run trained on.
    identity_updates = {k: updates.pop(k) for k in ("dataset_id", "dataset_fingerprint") if k in updates}

    key, state = lineage_key(experiment_id, root=root), status_key(experiment_id, root=root)
    refused: dict[str, Any] = {}
    with store.transaction(key, state) as txn:
        lineage = txn.read(key, default={})
        current_state = recorded_state(txn.read(state))
        # Additive-only, unlike refuse_if_terminal's own all-or-nothing refusal: a still-empty
        # field may take its first write post-terminal, only a populated one is frozen.
        try:
            refuse_if_terminal(experiment_id, "update_lineage", current_state)
        except ExperimentTerminal:
            refused = {k: v for k, v in updates.items()
                       if _MEMBER_WRITE_REFUSED["lineage"](lineage, k, v)}
            if refused:
                updates = {k: v for k, v in updates.items() if k not in refused}
        lineage.update(updates)
        txn.write(key, lineage)

    refused_fields = sorted([*identity_updates, *refused])
    return {"experiment_id": experiment_id, "lineage": lineage,
            **({"refused": refused_fields} if refused_fields else {})}


def register_model_from_experiment(
    experiment_id: str,
    checkpoint_path: str,
    *,
    project_path: str = "",
    name: str | None = None,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Bind the registry's own entry to the run that produced it: the digest completion recorded.

    Requires a completed run whose completion recorded a digest (``complete_run``'s own write): a
    run in ``created`` or ``running``, or one that ended ``failed``/``canceled``, refuses by name.
    Hashes the caller's ``checkpoint_path`` through the same function completion hashed with and
    refuses when the two digests differ, naming both and the path completion recorded: the caller's
    path may be the recorded path or any byte-identical copy, never a different file.
    ``project_path``, when given, must be the directory the experiment's own keys hang off
    (compared through ``splits.same_directory``, tolerant of a different spelling of the same
    directory); any other directory refuses by name, naming the root this call would otherwise have
    searched.

    Pulls the experiment's config and the checkpoint's own metrics, the epoch that produced this
    checkpoint (e.g. ``model_best.pt``'s best epoch), not necessarily the last training epoch.
    Registers with no tags and writes only the registry's own entry, with ``experiment_id`` set to
    this run. A checkpoint that carries no metrics dict, or that will not load at all, registers
    with an empty ``metrics`` and a ``metrics_source`` of ``None``. The unpickle and its
    ``schema_version`` check are ``model_registry._load_verified_payload``.

    ``metrics_source`` records which path produced the numbers, not that anyone verified them:
    ``"trainer"`` when the run's config carries no ``training_source`` (the platform's own
    ``default_train`` computed them), ``"training_source"`` when it does (a bespoke loop's own
    saved state), and ``None`` when the checkpoint carries no metrics.

    A name a prior run bound is not evicted by this call unless ``experiment_id`` is that same run:
    the registry's own eviction rail refuses, returned here as an error naming the run that holds
    the name (see ``model_registry.EntryOwnedByRun``).

    ``root`` names the platform root the experiment's own keys hang off; the default is this
    process's pinned platform root, and ``project_path``'s own check is against whichever one
    applies.
    """
    if not experiment_exists(experiment_id, root=root):
        return {"error": f"Experiment not found: {experiment_id}"}

    from tcip_mcp.pipelines.data.splits import same_directory
    from tcip_mcp.project_paths import platform_state_root

    platform_root = str(Path(root).resolve()) if root is not None else str(platform_state_root())
    if project_path and not same_directory(project_path, platform_root):
        return {"error": f"register_model_from_experiment: project_path {project_path!r} is not "
                         f"the root experiment {experiment_id!r}'s own keys hang off "
                         f"({platform_root!r})."}
    # Resolved once confirmed the same root: the index key refuses a non-absolute one, so a
    # relative or forward-slash spelling of the root must still reach it.
    registry_root = str(Path(project_path).resolve()) if project_path else platform_root

    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        return {"error": f"register_model_from_experiment: checkpoint_path {checkpoint_path!r} "
                         "does not exist."}
    with open(ckpt, "rb") as f:
        data = f.read()

    from tcip_mcp.model_registry import _sha256_of_bytes

    digest = _sha256_of_bytes(data)

    state = _current_state(experiment_id, root=root)
    lineage = read_member(lineage_key(experiment_id, root=root), {})
    recorded_digest = lineage.get("model_weights_sha256") if isinstance(lineage, dict) else None
    recorded_path = lineage.get("model_weights") if isinstance(lineage, dict) else None
    if state != "completed" or not recorded_digest:
        return {"error": f"experiment {experiment_id!r} has not completed with a recorded "
                         f"digest (state={state!r}): its run has not said what it produced. "
                         "complete_run records the digest when the run finishes."}

    if digest != recorded_digest:
        return {"error": f"{checkpoint_path} (sha256 {digest}) is not the bytes experiment "
                         f"{experiment_id!r}'s completion recorded (sha256 {recorded_digest}, at "
                         f"{recorded_path!r}): the caller's path must be the recorded path or a "
                         "byte-identical copy of it."}

    config = read_member(config_key(experiment_id, root=root), {})

    # Metrics stored in the checkpoint describe the epoch it was saved at (never a later epoch's).
    # Read through the same unpickle+version-check load_registered_checkpoint uses.
    final_metrics: dict[str, Any] = {}
    kind: str | None = None
    payload: dict | None = None
    try:
        from tcip_mcp.model_registry import _load_verified_payload

        payload = _load_verified_payload(data, source=f"{checkpoint_path} (sha256 {digest})")
    except Exception as exc:
        logger.warning(
            "checkpoint %s would not load (%s); registering experiment %s with no metrics "
            "rather than substituting a different epoch's numbers.", ckpt, exc, experiment_id,
        )
    if payload is not None:
        kind = payload.get("kind")
        stamped = payload.get("metrics")
        if isinstance(stamped, dict) and stamped:
            final_metrics = dict(stamped)
            if payload.get("epoch") is not None:
                final_metrics.setdefault("epoch", payload["epoch"])

    metrics_source: str | None = None
    if final_metrics:
        from tcip_mcp.pipelines.model_build import TRAINING_SOURCE_KEY

        metrics_source = "training_source" if config.get(TRAINING_SOURCE_KEY) else "trainer"

    from tcip_mcp.model_registry import EntryOwnedByRun, _register_entry
    from tcip_mcp.registry_paths import resolved_registry_path

    try:
        entry = _register_entry(
            registry_root, name=name or experiment_id, checkpoint_path=checkpoint_path,
            config=config, metrics=final_metrics, tags=[], kind=kind,
            metrics_source=metrics_source, experiment_id=experiment_id, sha256=digest,
        )
    except EntryOwnedByRun as exc:
        return {"error": str(exc)}
    return {
        "experiment_id": experiment_id,
        "registered": entry["name"],
        "checkpoint": str(resolved_registry_path(registry_root, entry["checkpoint_path"])),
        "sha256": digest,
        "metrics": final_metrics,
        "metrics_source": metrics_source,
    }


def _distinct_epoch_count(rows: list[dict[str, Any]]) -> int:
    """The number of distinct ``epoch`` values among ``rows``, compared as each row's own
    canonical JSON text so an ``epoch`` of any type, including a bespoke loop's own unhashable
    one, counts without raising."""
    seen = {json.dumps(row.get("epoch"), sort_keys=True, default=str) for row in rows}
    return len(seen)


def get_experiment(
    experiment_id: str, *, metrics_limit: int | None = None, metrics_offset: int = 0,
) -> dict[str, Any]:
    """Read full experiment state.

    ``metrics`` is the run's own log: every row the run's own :func:`log_metrics` appended, in
    order, oldest first. Its last row is only the last one logged, not a verified result.

    ``metrics`` can be paginated for long runs: ``metrics_offset`` and ``metrics_limit`` index into
    the row list, so ``n_rows`` (the row count) is the paging bound, not ``n_epochs`` (the count of
    distinct ``epoch`` values; the stock loop logs one row per epoch, a bespoke one may log
    several). Defaults return all metrics. ``validations`` is the whole claim history, unpaginated.
    """
    if not experiment_exists(experiment_id):
        return {"error": f"Experiment not found: {experiment_id}"}

    result: dict[str, Any] = {"experiment_id": experiment_id}

    members = {"config": config_key, "status": status_key,
               "artifacts": artifacts_key, "lineage": lineage_key}
    for name, key_of in members.items():
        document = read_member(key_of(experiment_id))
        if document is not None:
            result[name] = document

    rows = read_metrics(experiment_id)
    end = (metrics_offset + metrics_limit) if metrics_limit is not None else None
    result["n_epochs"] = _distinct_epoch_count(rows)
    result["n_rows"] = len(rows)
    result["metrics"] = rows[metrics_offset:end]
    result["metrics_offset"] = metrics_offset
    result["validations"] = read_validations(experiment_id)

    return result


def list_experiments() -> list[dict[str, Any]]:
    """List every experiment the store holds a status record for, run or not.

    Covers a calibration experiment (id derived from a claim's content), a review-feedback lineage,
    a pre-created experiment never launched, and a launched one. The ids come from
    :func:`experiment_ids_with_status`. ``has_model_source`` is whether the config carries a
    ``model_source`` (a training run) versus an experiment that tracks something else.
    """
    from tcip_mcp.pipelines.model_build import MODEL_SOURCE_KEY

    experiments = []
    for experiment_id in experiment_ids_with_status(None):
        status = read_member(status_key(experiment_id))
        if isinstance(status, dict):
            config = read_member(config_key(experiment_id), {})
            experiments.append({
                "experiment_id": experiment_id,
                "state": recorded_state(status),
                "created": status.get("created"),
                "has_model_source": bool(isinstance(config, dict) and config.get(MODEL_SOURCE_KEY)),
            })

    return experiments


def _split_summary(experiment_id: str) -> dict[str, Any]:
    """The partition column for one experiment: :func:`read_run_partition_checked` reduced to
    the four states a comparison names. ``{"case": "error", "error": ...}`` for a record that
    exists but will not decode; ``{"case": "none"}`` for a run that never wrote one;
    ``{"case": "bound", "selection_dir": ..., "seed": ..., "redraw": bool}`` for a run bound to
    a named selection, ``selection_dir`` and ``redraw`` as ``split.json``'s own
    ``selection_binding`` records them; ``{"case": "drawn", "seed": ...}`` otherwise.
    """
    partition, decode_error = read_run_partition_checked(experiment_id)
    if decode_error is not None:
        return {"case": "error", "error": decode_error}
    if not partition:
        return {"case": "none"}
    if "selection_binding" in partition:
        binding = partition["selection_binding"]
        return {"case": "bound", "selection_dir": binding["selection_dir"],
                "seed": partition["seed"], "redraw": binding["redraw"]}
    return {"case": "drawn", "seed": partition["seed"]}


def _index_registry_entries(
    experiment_ids: list[str],
) -> tuple[dict[str, list[dict[str, Any]]] | None, str | None]:
    """Every registered model entry naming one of ``experiment_ids`` as its producer, indexed by
    experiment id, from one read of the platform root's registry index. Returns ``(index, error)``:
    ``index`` is ``None`` and ``error`` names why when the index document could not be read (an
    absent registry reads as an empty index). Otherwise ``index`` holds every match, each entry
    reduced to ``name``, ``metrics``, ``metrics_source``, ``registered_at``; an experiment with no
    registered entry is absent from the index.
    """
    from tcip_mcp.model_registry import read_registry_index
    from tcip_mcp.project_paths import platform_state_root

    try:
        entries = read_registry_index(platform_state_root())
    except Exception as exc:
        return None, f"registry unreadable: {exc}"

    wanted = set(experiment_ids)
    index: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        eid = e["experiment_id"]
        if eid not in wanted:
            continue
        index.setdefault(eid, []).append({
            "name": e["name"], "metrics": e["metrics"],
            "metrics_source": e["metrics_source"], "registered_at": e["registered_at"],
        })
    return index, None


def compare_experiments(experiment_ids: list[str], *, stale_seconds: float) -> dict[str, Any]:
    """Side-by-side comparison of multiple experiments.

    ``stale_seconds`` is the heartbeat freshness window :func:`derived_state` applies.

    Per experiment: ``recorded_state`` (the stored state) and ``state``, the heartbeat-derived
    state via :func:`derived_state`, but only for a record no longer pristine (:func:`is_pristine`): a
    pre-created experiment never launched reports ``state`` equal to ``recorded_state``
    (``"created"``). ``log_locked``, true when ``recorded_state`` is in the mutation lock's
    terminal set, meaning only that :func:`log_metrics` refuses further rows;
    ``last_logged_metrics``, the run's own log's last row (not a verified result, see
    :func:`get_experiment`); ``rows_after_end``, the count of rows whose ``timestamp`` is a later
    instant than the record's own ``ended``, ``None`` when the record has no ``ended``, and a row
    whose own ``timestamp`` is missing or unparseable never counted; and ``n_epochs``/``n_rows``,
    always present.

    Also per experiment: ``task``/``subject`` from the config already read; ``status_error``, the
    status record's own failure reason (``None`` for a run that never failed, distinct from a
    comparison entry's own top-level ``error`` when :func:`get_experiment` could not read it at
    all); ``model``, the config's builder, ``None`` when the config names none; ``split`` (see
    :func:`_split_summary`); and ``registry``, this experiment's own registered entries (see
    :func:`_index_registry_entries`), absent, with ``registry_error`` naming why, when the
    project's registry index can't be read or matched at all (an experiment with no registered
    entry still carries ``registry: []``). ``same_dataset_fingerprint`` is ``None``, never
    ``True``, when any compared id is an error entry.
    """
    from tcip_mcp.pipelines.model_build import MODEL_SOURCE_KEY, run_task

    comparisons: list[dict[str, Any]] = []
    registry_index, registry_error = _index_registry_entries(experiment_ids)

    for eid in experiment_ids:
        exp = get_experiment(eid)
        if "error" in exp:
            comparisons.append({"experiment_id": eid, "error": exp["error"]})
            continue

        status_doc = exp.get("status")
        status_doc = status_doc if isinstance(status_doc, dict) else {}
        stored_state = recorded_state(status_doc)
        ended = status_doc.get("ended")
        launched = not is_pristine(stored_state, metrics_logged_of(status_doc))

        summary: dict[str, Any] = {
            "experiment_id": eid,
            "recorded_state": stored_state,
            "state": derived_state(status_doc, stale_seconds) if launched else stored_state,
            "log_locked": stored_state in _TERMINAL_STATES,
            "n_epochs": exp["n_epochs"],
            "n_rows": exp["n_rows"],
        }

        metrics = exp.get("metrics", [])
        if metrics:
            summary["last_logged_metrics"] = metrics[-1]
        rows_after_end = None
        ended_instant = _parse_iso_instant(ended) if ended else None
        if ended_instant is not None:
            rows_after_end = sum(
                1 for row in metrics
                if (row_instant := _parse_iso_instant(row.get("timestamp"))) is not None
                and row_instant > ended_instant
            )
        summary["rows_after_end"] = rows_after_end
        # status_error, not "error": that key already marks a comparison entry get_experiment
        # could not even read (the sentinel with_fp/same_dataset_fingerprint filter on below).
        summary["status_error"] = status_doc.get("error")

        if registry_index is not None:
            summary["registry"] = registry_index.get(eid, [])
        else:
            summary["registry_error"] = registry_error

        summary["split"] = _split_summary(eid)

        # Get config summary
        config = exp.get("config", {})
        config = config if isinstance(config, dict) else {}
        model_source = config.get(MODEL_SOURCE_KEY, {})
        model_source = model_source if isinstance(model_source, dict) else {}
        summary["model"] = model_source.get("builder")
        summary["task"] = run_task(config) if config else None
        data_cfg = config.get("data", {})
        summary["subject"] = data_cfg.get("subject") if isinstance(data_cfg, dict) else None

        # Dataset identity (the content end of the reproduce-a-number chain), from the immutable lineage.
        lin = exp.get("lineage")
        if isinstance(lin, dict):
            summary["dataset_id"] = lin.get("dataset_id")
            summary["dataset_fingerprint"] = lin.get("dataset_fingerprint")

        comparisons.append(summary)

    # Whether every compared run trained on the same dataset content; an unset fingerprint, or a
    # record this call could not read, makes it unknown, not "same".
    any_error = any("error" in c for c in comparisons)
    fps = {c.get("dataset_fingerprint") for c in comparisons if "error" not in c}
    same_dataset = None if (any_error or not fps or None in fps) else len(fps) == 1
    return {"experiments": comparisons, "count": len(comparisons), "same_dataset_fingerprint": same_dataset}


def get_experiment_lineage(experiment_id: str) -> dict[str, Any]:
    """Trace the full data → model → predictions chain."""
    if not experiment_exists(experiment_id):
        return {"error": f"Experiment not found: {experiment_id}"}

    lineage = read_member(lineage_key(experiment_id))
    if lineage is None:
        return {"error": "No lineage file found"}

    from tcip_mcp.pipelines.model_build import run_task

    config = read_member(config_key(experiment_id))
    if isinstance(config, dict):
        data_cfg = config.get("data", {})
        lineage["data_config"] = {
            "images_dir": data_cfg.get("images_dir"),
            "labels_dir": data_cfg.get("labels_dir"),
            "task": run_task(config),
        }

    return {"experiment_id": experiment_id, "lineage": lineage}


def read_run_partition_checked(
    experiment_id: str, *, root: Path | str | None = None,
) -> tuple[dict[str, Any], str | None]:
    """The run's persisted partition, and the decode failure behind an unreadable one.

    Returns ``(manifest, decode_error)``. ``manifest`` is ``{}`` with ``decode_error`` ``None`` for
    a run that never wrote one. A record that exists but will not decode also answers
    ``manifest={}``, but ``decode_error`` names why. ``root`` resolves the same project a caller's
    own checkpoint lookup used.
    """
    from tcip_store import DecodeError

    try:
        manifest = store.read(split_key(experiment_id, root=root), default={})
    except DecodeError as exc:
        return {}, str(exc)
    return (manifest if isinstance(manifest, dict) else {}), None


def read_run_partition(experiment_id: str) -> dict[str, Any]:
    """The run's persisted partition, or ``{}`` when it was never written or could not be decoded."""
    manifest, _ = read_run_partition_checked(experiment_id)
    return manifest
