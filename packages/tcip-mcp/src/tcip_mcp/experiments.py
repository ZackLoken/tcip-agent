"""Experiment tracking for ML training runs.

Stores experiment state in .tcip/experiments/<experiment_id>/:
  config.json, full training config snapshot
  metrics.jsonl, epoch-by-epoch metrics (append-only)
  artifacts.json, pointers to model weights, predictions
  lineage.json, data → model → predictions chain
  status.json, current state and timestamps
  split.json, the train/val membership, seed and dataset identity a metric is reproducible with
  env.json, the library versions, seed and model kind behind a reproducible run
  validations.jsonl, the claims earned against this run's evidence (append-only)

This module declares the record's members, so it is also the one place they are addressed:
every reader and writer takes a key from a constructor here rather than composing a path of
its own, and ``experiment_dir`` serves the run artifacts that live beside the record without
being members of it (checkpoints, TensorBoard logs, a bespoke run's source snapshot).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
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

from tcip_mcp.project_paths import project_root, resolve_state

logger = logging.getLogger(__name__)

# Relative default (tests rebind this constant). Consumers must go through
# ``experiments_dir()`` so the store anchors to ``$TCIP_PROJECT_ROOT`` when pinned (no
# subdir fragmentation) while a rebound absolute path / unpinned cwd still work.
EXPERIMENTS_DIR = Path(".tcip/experiments")


def experiments_dir() -> Path:
    """The experiment store, resolved against the pinned platform root at use time."""
    return resolve_state(EXPERIMENTS_DIR)


def experiment_dir(experiment_id: str) -> Path:
    """One experiment's directory: the record's members plus the run artifacts beside them.

    A caller that needs a member document asks for its key instead. This serves the files the
    record's own layout does not name: weights, TensorBoard event files, and the per-file
    source snapshot a bespoke run copies in.
    """
    return experiments_dir() / experiment_id


# ── the experiment stores ────────────────────────────────────────────────────

_MEMBER_DOC = RootedFileLocator(suffix=".json")
"""One member document inside its experiment's directory."""

_MEMBER_LOG = RootedFileLocator(suffix=".jsonl")
"""One append-only member log inside its experiment's directory."""


def experiments_scope(root: Path | str | None = None) -> str:
    """The root every experiment key hangs off: the experiment store, made absolute.

    Absolute because a key names a root rather than a process's current directory, and
    resolved per call because ``EXPERIMENTS_DIR`` is relative until a platform root is
    pinned, and a pin can land mid-process. ``root`` names a different platform root than
    this process's own, for a caller (the web backend serving a run of the project the
    browser has open) whose subject is a project it is not itself pinned to.
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
        codec=RECORD_JSON,
        concurrency="cas",
        enumerable=True,
        locator=_MEMBER_DOC,
    )
)


STATUS_DOCUMENT = "status"


def status_key(experiment_id: str, *, root: Path | str | None = None) -> Key:
    """The run's state, timestamps and liveness heartbeat.

    ``cas``: every writer here reads the document and updates fields inside it, from the
    training subprocess and the tool process at once, so an unconditional write drops the
    heartbeat or the run identity another writer just stamped.
    """
    return _member_key(EXPERIMENT_STATUS_STORE, experiment_id, STATUS_DOCUMENT, root)


EXPERIMENT_LINEAGE_STORE = "experiment_lineage"
register_store(
    StoreDescriptor(
        name=EXPERIMENT_LINEAGE_STORE,
        kind="record",
        key_fields=("experiment_id", "document"),
        codec=RECORD_JSON,
        concurrency="cas",
        enumerable=True,
        locator=_MEMBER_DOC,
    )
)


def lineage_key(experiment_id: str, *, root: Path | str | None = None) -> Key:
    """The data to model to predictions chain.

    ``cas``: ``update_lineage`` merges fields into the stored document under a lock, so an
    unconditional write erases an edge another writer recorded.
    """
    return _member_key(EXPERIMENT_LINEAGE_STORE, experiment_id, "lineage", root)


EXPERIMENT_ARTIFACTS_STORE = "experiment_artifacts"
register_store(
    StoreDescriptor(
        name=EXPERIMENT_ARTIFACTS_STORE,
        kind="record",
        key_fields=("experiment_id", "document"),
        codec=RECORD_JSON,
        concurrency="cas",
        enumerable=True,
        locator=_MEMBER_DOC,
    )
)


def artifacts_key(experiment_id: str, *, root: Path | str | None = None) -> Key:
    """The run's artifact pointers.

    ``cas``: ``record_artifact`` adds one name to the stored mapping under a lock, so an
    unconditional write drops the pointers already recorded.
    """
    return _member_key(EXPERIMENT_ARTIFACTS_STORE, experiment_id, "artifacts", root)


EXPERIMENT_ENV_STORE = "experiment_env"
register_store(
    StoreDescriptor(
        name=EXPERIMENT_ENV_STORE,
        kind="record",
        key_fields=("experiment_id", "document"),
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        enumerable=True,
        locator=_MEMBER_DOC,
    )
)


def env_key(experiment_id: str, *, root: Path | str | None = None) -> Key:
    """The environment capture behind a reproducible run.

    ``last_writer_wins``: the envelope writes it whole, once, from state it already holds.
    """
    return _member_key(EXPERIMENT_ENV_STORE, experiment_id, "env", root)


EXPERIMENT_SPLIT_STORE = "experiment_split"
register_store(
    StoreDescriptor(
        name=EXPERIMENT_SPLIT_STORE,
        kind="record",
        key_fields=("experiment_id", "document"),
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        enumerable=True,
        locator=_MEMBER_DOC,
    )
)


def split_key(experiment_id: str, *, root: Path | str | None = None) -> Key:
    """The train/val membership, seed and dataset identity this run's metrics belong to.

    ``last_writer_wins``: one writer composes the whole manifest once, from the datasets the
    run actually built, and nothing merges into it afterwards.
    """
    return _member_key(EXPERIMENT_SPLIT_STORE, experiment_id, "split", root)


EXPERIMENT_METRICS_STORE = "experiment_metrics"
register_store(
    StoreDescriptor(
        name=EXPERIMENT_METRICS_STORE,
        kind="log",
        key_fields=("experiment_id", "document"),
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


# Once a run reaches a terminal state its record is immutable (experiments are immutable). The lock
# is additive-only: populated fields freeze, but a still-empty field may take its first write,
# so the post-completion predictions link + model registration still land. Deliberately excludes
# "cancelled", a cancelled run's record stays reopenable (e.g. resumed via resume_from), so it
# must not be lock-frozen the way a genuinely finished run is.
_TERMINAL_STATES = {"completed", "failed"}

# A different concept sharing similar vocabulary, states reconstruct_run_status trusts as
# already-decided and never re-derives from heartbeat freshness. Unlike _TERMINAL_STATES above,
# this does include "cancelled": a gracefully cancelled run recorded its own final state honestly
# (model_final.pt was written, cancel_training's own documented contract), and re-deriving it from
# heartbeat staleness would misreport it as "running" then permanently as "interrupted", implying
# a crash that never happened. Named separately rather than reusing _TERMINAL_STATES so the two
# purposes (mutation-lock vs. heartbeat-reconstruction) can never silently drift onto each other.
_RECORDED_AS_DONE = {"completed", "failed", "cancelled"}


def read_member(key: Key, default: Any = None) -> Any:
    """One member document, with an unreadable record folded onto ``default``.

    Every caller of an experiment member already treats a corrupt one the way it treats an absent
    one, so the fold is stated once rather than repeated at each read. The read every consumer of
    a member document goes through, in this module and outside it, so none of them reaches past
    the seam for the bytes.
    """
    try:
        return store.read(key, default=default)
    except DecodeError:
        logger.warning("experiment member %s does not decode", list(key.parts), exc_info=True)
        return default


def experiment_exists(experiment_id: str) -> bool:
    """Whether this id names a real experiment record, by its config snapshot."""
    return store.exists(config_key(experiment_id))


def _current_state(experiment_id: str) -> str | None:
    status = read_member(status_key(experiment_id), {})
    return status.get("state") if isinstance(status, dict) else None


class ExperimentTerminal(RuntimeError):
    """A write reached an experiment record already in a terminal state (completed/failed;
    cancelled stays resumable and is never terminal here). Raised by :func:`refuse_if_terminal`;
    a caller that reports refusal as a return value (``log_metrics``, ``record_artifact``) catches
    it and maps it to that value, a caller for whom the lost write is itself a run failure (the
    training worker's provenance patches, the split manifest) lets it propagate uncaught.
    """


def refuse_if_terminal(experiment_id: str, op: str, state: str | None) -> None:
    """Raise :class:`ExperimentTerminal` if ``state`` is terminal.

    ``state`` is the value the caller already read, inside its own transaction when it holds
    one (so the check and the write it guards see the same value) or via :func:`_current_state`
    when it doesn't. The one implementation of "is this experiment terminal" every writer of an
    experiment member consults, rather than each comparing against ``_TERMINAL_STATES`` itself.
    :func:`update_status` is the one exception: its rule differs (a terminal record refuses only a
    move to a *non*-terminal state, not a move between two terminal states or a repeat of the
    current one), so it keeps its own comparison rather than calling this.

    Never audits itself: an audit line is a log append, which cannot run inside a record
    transaction (``store.transaction`` only ever holds ``kind="record"`` keys), and a caller
    checking this from inside its own transaction would have the append raise
    ``TransactionMisuse``. The caller audits the refusal once its transaction has closed (or
    immediately, when it holds none).
    """
    if state in _TERMINAL_STATES:
        raise ExperimentTerminal(f"Experiment {experiment_id} is {state} (terminal); refusing to {op}.")


def _audit_refused(experiment_id: str, op: str, detail: dict[str, Any]) -> None:
    """Record a refused post-terminal mutation on the append-only audit log (best-effort)."""
    try:
        from tcip_mcp.audit import record_event

        record_event("experiment_mutation_refused", {"experiment_id": experiment_id, "op": op,
                                                      **detail}, status="refused")
    except Exception:
        logger.debug("could not audit refused mutation", exc_info=True)


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
    refused inside the write's own lock rather than after a separate existence check that two
    callers could both pass.

    ``dataset_id`` / ``dataset_fingerprint`` record the identity of the data this run trained on (the
    content end of the reproduce-a-number chain), written into the immutable lineage at creation. They
    are set once here and never via ``update_lineage`` (identity, not a mutable edge).

    The config is the caller's own dict, so it is checked against what JSON can hold before
    the write, naming the offending field rather than leaving the codec to refuse a payload
    it can only describe by store and key.
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
        "predictions": None,
    }
    store.replace(lineage_key(experiment_id), lineage, expect=Version.ABSENT)
    store.replace(artifacts_key(experiment_id), {}, expect=Version.ABSENT)

    return {
        "experiment_id": experiment_id,
        "path": str(experiment_dir(experiment_id)),
        "state": "created",
    }


def is_pristine(state: str | None, has_metrics: bool) -> bool:
    """Whether an experiment record with this ``state`` and ``has_metrics`` may still take a full
    ``config.json`` rewrite: ``state == "created"`` and no metrics logged yet.

    The one implementation of the pristine predicate. :func:`overwrite_config_if_pristine` calls
    it from inside the transaction that reads ``state``; ``_ensure_experiment`` calls it first, on
    its own untransacted read, to decide whether to attempt that overwrite at all, so a relaunch
    under a non-pristine id mints its fresh id straight away instead of provoking (and auditing) a
    refusal nothing needed.
    """
    return state == "created" and not has_metrics


def overwrite_config_if_pristine(experiment_id: str, config: dict[str, Any]) -> dict[str, Any]:
    """Rewrite ``config.json`` with the config actually launched, but only while the experiment is
    still pristine (state == "created" and no epochs logged yet).

    A pre-created experiment's ``config.json`` is written once, at ``create_experiment`` time,
    before effective tiling geometry and the training seed are resolved (see
    ``training_tools.launch_training``). Reusing that id via ``_ensure_experiment``'s pristine-reuse
    branch would otherwise ship a permanently stale snapshot describing a config that was never
    trained. Refuses (and audits the refusal) once :func:`is_pristine` says the record is no
    longer pristine, a "created" record that already has metrics rows must stay protected too, not
    just the terminal-state lock alone.

    The state read and the write are one transaction (closing the race where a concurrent
    ``update_status``/patch flips the record between the check and the write); the metrics-log
    read is not, a log key cannot join a record transaction (``tcip_store.transaction`` requires
    ``kind="record"``), so a metric logged in that narrow window is the one residual this cannot
    close, it can only ever make the record look non-pristine, never the reverse.
    """
    check_json_value(config, path="config")
    if not experiment_exists(experiment_id):
        return {"error": f"Experiment not found: {experiment_id}"}
    has_metrics = bool(read_metrics(experiment_id))
    cfg_key, st_key = config_key(experiment_id), status_key(experiment_id)
    state: str | None = None
    refused = False
    with store.transaction(cfg_key, st_key) as txn:
        status = txn.read(st_key, default={})
        state = status.get("state") if isinstance(status, dict) else None
        refused = not is_pristine(state, has_metrics)
        if not refused:
            txn.write(cfg_key, config)
    if refused:
        _audit_refused(experiment_id, "overwrite_config_if_pristine",
                       {"state": state, "has_metrics": has_metrics})
        return {"error": f"Experiment {experiment_id} is no longer pristine; refusing to "
                         f"overwrite its config.json."}
    return {"experiment_id": experiment_id, "overwritten": True}


def update_status(experiment_id: str, state: str, *, error: str | None = None) -> dict[str, Any]:
    """Update experiment state (created → running → completed | failed).

    ``error`` records a specific failure reason (e.g. a wall-clock-timeout kill) into
    ``status.json["error"]``, omitted/``None`` never clears a previously-recorded error, only an
    explicit new value overwrites it.
    """
    if not experiment_exists(experiment_id):
        return {"error": f"Experiment not found: {experiment_id}"}

    key = status_key(experiment_id)
    with store.transaction(key) as txn:
        status = txn.read(key, default={})
        current = status.get("state")
        # Terminal-state lock: a completed/failed run cannot be re-opened to a non-terminal state.
        refused_reopen = (
            current in _TERMINAL_STATES and state != current and state not in _TERMINAL_STATES
        )
        if not refused_reopen:
            status["state"] = state
            if error is not None:
                status["error"] = error

            now = datetime.now(timezone.utc).isoformat()
            status["heartbeat"] = now  # liveness stamp: a fresh heartbeat means a live process
            if state == "running" and not status.get("started"):
                status["started"] = now
            if state in ("completed", "failed"):
                status["ended"] = now

            txn.write(key, status)

    if refused_reopen:
        _audit_refused(experiment_id, "update_status", {"from": current, "to": state})
        return {"error": f"Experiment {experiment_id} is {current} (terminal); refusing to "
                         f"re-open to {state!r}.", "state": current}
    return {"experiment_id": experiment_id, "state": state}


def stamp_run_identity(experiment_id: str, run_id: str, output_dir: str) -> None:
    """Record which ``run_id``/``output_dir`` produced this experiment, into ``status.json``.

    Best-effort, like ``_touch_heartbeat``, a dropped stamp must not break the launch it's
    recording. Called unconditionally by ``_ensure_experiment`` regardless of which of its three
    branches resolved ``experiment_id`` (fresh creation, pristine pre-created-experiment reuse, or a
    fresh-id conflict), those are the only paths that mint a real, running experiment, and this
    is what makes the real artifact directory (``output_dir``, a separately-computed, caller-influenced
    path that only coincides with the experiment directory by convention) discoverable from
    ``experiment_id``/``run_id`` alone by a different process.
    """
    key = status_key(experiment_id)
    if not store.exists(key):
        return
    try:
        with store.transaction(key) as txn:
            status = txn.read(key, default={})
            status["run_id"] = run_id
            status["output_dir"] = output_dir
            txn.write(key, status)
    except Exception:
        logger.warning("stamp_run_identity failed for %s/%s", experiment_id, run_id, exc_info=True)


def resolve_experiment_dir_for_run(run_id: str) -> Path | None:
    """This run's experiment directory, or ``None`` when no record claims the run.

    The identity question is answered by :func:`resolve_experiment_for_run`; this is the form
    for a caller that then wants the run artifacts beside the record.
    """
    experiment_id = resolve_experiment_for_run(run_id)
    return None if experiment_id is None else experiment_dir(experiment_id)


def resolve_experiment_for_run(run_id: str, *, root: Path | str | None = None) -> str | None:
    """Find the experiment id for ``run_id`` without assuming ``experiment_id == run_id``.

    Tries the exact match first (the common case, ``experiment_id == run_id``). Then the
    fresh-id relaunch format (``f"{experiment_id}_{run_id}"``, always suffixed ``_<run_id>``), by
    the suffix. Neither naming convention covers a *custom-named* experiment (an agent/breeder
    pre-created it via the standalone ``create_experiment`` tool, e.g. ``"exp-001-<crop>-<trait>-
    det"``, before any ``run_id`` existed, then launched training against it later, a real, tested
    workflow, not theoretical: ``_ensure_experiment``'s pristine-reuse branch), its id
    bears no naming relationship to ``run_id`` at all. For that case, falls back to reading every
    experiment's own stamped ``status.json["run_id"]`` (the authoritative fact
    ``stamp_run_identity`` records, not a naming guess), a full scan, but reached only once both
    naming shortcuts miss, and it's also what disambiguates the (negligible-probability, per
    ``run_id``'s own timestamp+uuid entropy) case of more than one suffix match, rather than
    refusing a resolvable run just because the fast path was ambiguous. Returns ``None`` only when no
    record's stamped identity matches at all, the caller (``cancel_run``'s disk fallback,
    ``reconstruct_run_status``) must then refuse honestly rather than act against an unverified path.

    The candidates come from enumerating the status records, not from listing directories: what
    names an experiment is a record the store holds, and a directory the store's own backend keeps
    beside them is not a candidate id.

    ``root`` names a platform root other than this process's own, for the web backend serving a
    run of the project the browser has open.
    """
    try:
        if store.exists(status_key(run_id, root=root)):
            return run_id
    except BadKey:
        return None
    candidates = experiment_ids_with_status(root)
    matches = [name for name in candidates if name.endswith(f"_{run_id}")]
    if len(matches) == 1:
        return matches[0]

    for name in candidates:
        status = read_member(status_key(name, root=root), {})
        if isinstance(status, dict) and status.get("run_id") == run_id:
            return name
    return None


def experiment_ids_with_status(root: Path | str | None = None) -> list[str]:
    """Every experiment id the store holds a status record for, sorted.

    Every experiment gets a status record at creation, so this is the whole set, and it is the
    enumeration for any caller that wants the experiments themselves: what names an experiment is a
    record the store holds, so directories a backend happens to keep beside them are not candidate
    ids and a backend that keeps none at all still answers. Enumerating one member store answers
    with every member key under the scope, and the document part says which member a key names.
    """
    found = store.keys(EXPERIMENT_STATUS_STORE, experiments_scope(root))
    return sorted(key.parts[0] for key in found if key.parts[1] == STATUS_DOCUMENT)


def reconstruct_run_status(run_id: str, *, stale_seconds: float = 600.0) -> dict[str, Any] | None:
    """Reconstruct a run's status from disk for a caller whose in-memory registry doesn't
    have it, either it was never in this process (a different process launched it) or it was
    subprocess-delegated and the in-memory record is stale by design.

    Returns ``None`` when the run can't be resolved on disk at all (an honestly unknown run, not a
    guess). ``current_epoch`` comes from the last ``metrics.jsonl`` row when present; ``best_metric``
    is left ``None``, a running best isn't recoverable from the metrics log alone without
    re-deriving the selection policy, and a fabricated approximation would be worse than an honest
    gap (matches the pre-existing convention this function replaces, which also reported ``None``).
    ``stale_seconds`` lets a caller (``routes/training.py``) keep its own configurable heartbeat
    window rather than being pinned to this module's default.
    """
    experiment_id = resolve_experiment_for_run(run_id)
    if experiment_id is None:
        return None
    status = read_member(status_key(experiment_id))
    if not isinstance(status, dict):
        return None

    state = status.get("state", "unknown")
    heartbeat = status.get("heartbeat")
    if state not in _RECORDED_AS_DONE:
        state = "running" if _heartbeat_fresh(heartbeat, stale_seconds) else "interrupted"

    rows = read_metrics(experiment_id)
    current_epoch = rows[-1].get("epoch") if rows else None

    return {
        "run_id": status.get("run_id", experiment_id),
        "experiment_id": experiment_id,
        "status": state,
        "current_epoch": current_epoch,
        "best_metric": None,
        "output_dir": status.get("output_dir"),
        "error": status.get("error"),
    }


def _heartbeat_fresh(hb_iso: str | None, stale_seconds: float = 600.0) -> bool:
    """True if ``hb_iso`` (ISO-8601) is within the staleness window, a process is still actively
    updating this run. Missing/unparseable → not fresh (treat as dead). Mirrors
    ``routes/training.py``'s own threshold; kept independent since this module has no FastAPI/env
    dependency and the two consumers (web route, MCP tool) can reasonably differ in the future."""
    if not hb_iso:
        return False
    try:
        hb = datetime.fromisoformat(hb_iso)
    except (ValueError, TypeError):
        return False
    if hb.tzinfo is None:
        hb = hb.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - hb).total_seconds() <= stale_seconds


def _touch_heartbeat(experiment_id: str) -> None:
    """Best-effort: stamp the current time into ``status.json['heartbeat']``.

    Called each epoch so a run still training in another process (e.g. the MCP agent) reads
    as live to a web client reconstructing run state, instead of being flagged interrupted.
    Never raises, a heartbeat failure must not break metric logging.
    """
    key = status_key(experiment_id)
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
    return [dict(record) for record in page.records]


def log_metrics(
    experiment_id: str,
    epoch: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Append epoch metrics to the run's metrics log and refresh its liveness heartbeat.

    The one writer of that log: a training body routes its rows here rather than opening the
    file beside it, so the module that declares the record's members is the module that
    appends to them and the terminal-state lock cannot be written around.

    A bespoke loop's row is its own dict, so it is checked field by field first: a tensor or
    a non-finite loss is named here, where the caller can see which metric it was.
    """
    check_json_value(metrics, path="metrics")
    if not experiment_exists(experiment_id):
        return {"error": f"Experiment not found: {experiment_id}"}

    # Terminal-state lock: a completed/failed run's metric history is frozen, no new epochs.
    try:
        refuse_if_terminal(experiment_id, "log_metrics", _current_state(experiment_id))
    except ExperimentTerminal as exc:
        _audit_refused(experiment_id, "log_metrics", {"epoch": epoch})
        return {"error": str(exc)}

    entry = {
        "epoch": epoch,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }
    store.append(metrics_key(experiment_id), entry)
    _touch_heartbeat(experiment_id)

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
)
"""Every field a validation row carries. All required, none defaulted.

A claim reads as provenance only when all of it is there: what was claimed, for which trait,
against which reference, produced by which checkpoint and run, over which content, and when.
A field a writer could omit would be a field a reader could not compare.
"""


def _content_digest(value: dict[str, Any]) -> str:
    """A mapping's content identity: sha256 over its canonical JSON, first 16 hex characters.

    Canonical spelling because the digest is an agreement between separate processes: a writer
    and every later reader must spell the same mapping the same way to compute the same
    identity. The width matches the platform's other content identities.
    """
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def validation_digest(body: dict[str, Any]) -> str:
    """The content identity of a validation row.

    Pure: a reader recomputes it from a row it read, which is what lets a stamp name one
    specific row rather than an experiment in general. It is not stored in the row, since a
    row carrying its own digest would be vouching for itself.
    """
    return _content_digest(body)


def _append_validation(experiment_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Append one earned claim to this experiment's validations log.

    Module-private: a validation is earned by running the gate over the evidence, so the only
    caller is the primitive that ran it, and there is no supported appender a caller can hand
    a verdict to. The storage seam's own generic append against this key stays reachable and
    is a stated residual rather than something this module closes.

    The row is checked against ``_VALIDATION_FIELDS`` first: a missing field is refused, never
    filled in, because a defaulted provenance field is a claim nobody made.
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
    try:
        from tcip_mcp.audit import record_event

        # Platform log: the record this mutates is a platform-scoped experiment member.
        record_event("experiment_validation_recorded",
                     {"experiment_id": experiment_id, "document": body["document"],
                      "trait": body["trait"], "record_digest": digest})
    except Exception:
        logger.debug("could not audit validation append", exc_info=True)
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

    A claim is earned against a real experiment record, and a door that calibrated predictions
    no experiment produced (a bespoke or unregistered checkpoint) holds none. The id is derived
    from the same content that constitutes the claim's identity, so a second calibration of the
    same document, checkpoint, reference and trait resolves to the same experiment and agrees
    with its config by construction rather than by comparison.

    ``config`` is the free text describing the calibration. The identity fields are written
    here from the arguments the id was derived from, so a config restating one is refused
    rather than left free to contradict the id.
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
        try:
            from tcip_mcp.audit import record_event

            # Platform log: the experiment record lives at the platform root.
            record_event("calibration_experiment_created",
                         {"experiment_id": experiment_id, "document": document, "trait": trait})
        except Exception:
            logger.debug("could not audit calibration experiment creation", exc_info=True)
    return experiment_id


def record_artifact(
    experiment_id: str,
    name: str,
    path: str,
) -> dict[str, Any]:
    """Register an artifact (model weights, predictions, etc.)."""
    if not experiment_exists(experiment_id):
        return {"error": f"Experiment not found: {experiment_id}"}

    key, state = artifacts_key(experiment_id), status_key(experiment_id)
    current: str | None = None
    refused_overwrite = False
    with store.transaction(key, state) as txn:
        artifacts = txn.read(key, default={})
        current = (txn.read(state, default={}) or {}).get("state")
        # Terminal-state lock (additive-only): a new artifact name may be recorded post-completion,
        # but an existing one is frozen, no silent overwrite of a delivered pointer.
        refused_overwrite = name in artifacts and current in _TERMINAL_STATES
        if not refused_overwrite:
            artifacts[name] = {"path": path, "recorded": datetime.now(timezone.utc).isoformat()}
            txn.write(key, artifacts)

    if refused_overwrite:
        try:
            refuse_if_terminal(experiment_id, "record_artifact", current)
        except ExperimentTerminal as exc:
            _audit_refused(experiment_id, "record_artifact", {"artifact": name})
            return {"error": f"{exc} Artifact {name!r} is already recorded and is immutable.",
                    "artifact": name}
    return {"experiment_id": experiment_id, "artifact": name, "path": path}


def update_lineage(
    experiment_id: str,
    **updates: Any,
) -> dict[str, Any]:
    """Update lineage fields (model_weights, predictions, etc.).

    The updates are the caller's own kwargs, merged whole into the stored document, so they
    are checked against what JSON can hold before any of them lands.
    """
    check_json_value(updates, path="updates")
    if not experiment_exists(experiment_id):
        return {"error": f"Experiment not found: {experiment_id}"}

    # Dataset identity is set once at creation and is immutable, never a lineage edge to backfill.
    # (The additive-only lock below would otherwise permit a first write to an empty identity field
    # even post-terminal, which would be a silent change to what data the run trained on.)
    identity_updates = {k: updates.pop(k) for k in ("dataset_id", "dataset_fingerprint") if k in updates}
    if identity_updates:
        _audit_refused(experiment_id, "update_lineage_identity", {"fields": sorted(identity_updates)})

    key, state = lineage_key(experiment_id), status_key(experiment_id)
    refused: dict[str, Any] = {}
    with store.transaction(key, state) as txn:
        lineage = txn.read(key, default={})
        current_state = (txn.read(state, default={}) or {}).get("state")
        # Additive-only, unlike refuse_if_terminal's own all-or-nothing refusal: a still-empty
        # field may take its first write post-terminal, only a populated one is frozen.
        try:
            refuse_if_terminal(experiment_id, "update_lineage", current_state)
        except ExperimentTerminal:
            refused = {k: v for k, v in updates.items()
                       if lineage.get(k) not in (None, "", [], {}) and lineage.get(k) != v}
            if refused:
                updates = {k: v for k, v in updates.items() if k not in refused}
        lineage.update(updates)
        txn.write(key, lineage)

    if refused:
        _audit_refused(experiment_id, "update_lineage", {"fields": sorted(refused)})
    return {"experiment_id": experiment_id, "lineage": lineage}


def register_model_from_experiment(
    experiment_id: str,
    checkpoint_path: str,
    *,
    project_path: str = "",
    name: str | None = None,
) -> dict[str, Any]:
    """Register a completed experiment's model in the project registry.

    Pulls the experiment's config and the checkpoint's own metrics, the epoch that produced
    this checkpoint (e.g. ``model_best.pt``'s best epoch), not necessarily the last training
    epoch, falling back to the experiment's final ``metrics.jsonl`` row if the checkpoint
    carries none. Registers with an ``experiment:<id>`` back-reference tag and records it in
    the experiment's lineage (``model_weights``). Metrics are read, never fabricated.
    """
    if not experiment_exists(experiment_id):
        return {"error": f"Experiment not found: {experiment_id}"}

    config = read_member(config_key(experiment_id), {})

    # Prefer metrics stored IN the checkpoint (they describe the epoch it was saved at, so a
    # best-checkpoint isn't mislabelled with a later, worse epoch's numbers). Fall back to the
    # experiment's last metrics.jsonl row only if the checkpoint carries none.
    final_metrics: dict[str, Any] = {}
    kind: str | None = None
    ckpt = Path(checkpoint_path)
    if ckpt.is_file():
        try:
            import torch  # local checkpoint the caller is registering deliberately

            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(payload, dict):
                kind = payload.get("kind")  # stamped by the trainer; None on older checkpoints
                if isinstance(payload.get("metrics"), dict):
                    final_metrics = dict(payload["metrics"])
                    if payload.get("epoch") is not None:
                        final_metrics.setdefault("epoch", payload["epoch"])
        except Exception:
            final_metrics = {}
    if not final_metrics:
        rows = read_metrics(experiment_id)
        if rows:
            final_metrics = rows[-1]

    from tcip_mcp.model_registry import ModelRegistry

    # Registry co-locates with the experiment store under the platform root (the adopted
    # project after set_active_project) unless an explicit path overrides.
    registry_root = project_path or str(project_root())
    entry = ModelRegistry(registry_root).register_model(
        name or experiment_id, checkpoint_path, config,
        metrics=final_metrics, tags=[f"experiment:{experiment_id}"], kind=kind,
    )
    update_lineage(experiment_id, model_weights=checkpoint_path)
    return {
        "experiment_id": experiment_id,
        "registered": entry["name"],
        "checkpoint": checkpoint_path,
        "metrics": final_metrics,
    }


def get_experiment(
    experiment_id: str, *, metrics_limit: int | None = None, metrics_offset: int = 0,
) -> dict[str, Any]:
    """Read full experiment state.

    ``metrics`` can be paginated for long runs: ``metrics_offset`` skips epochs and
    ``metrics_limit`` caps how many are returned; ``n_epochs`` is always the true total.
    Defaults return all metrics. ``validations`` is the whole claim history, unpaginated:
    a run earns few claims where it logs many epochs.
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
    result["n_epochs"] = len(rows)
    result["metrics"] = rows[metrics_offset:end]
    result["metrics_offset"] = metrics_offset
    result["validations"] = read_validations(experiment_id)

    return result


def list_experiments() -> list[dict[str, Any]]:
    """List all experiments with summary info.

    The ids come from the same status-record enumeration :func:`resolve_experiment_for_run`
    resolves against, so a record the resolver finds is a record this lists.
    """
    experiments = []
    for experiment_id in experiment_ids_with_status(None):
        status = read_member(status_key(experiment_id))
        if isinstance(status, dict):
            experiments.append({
                "experiment_id": experiment_id,
                "state": status.get("state", "unknown"),
                "created": status.get("created"),
            })

    return experiments


def compare_experiments(experiment_ids: list[str]) -> dict[str, Any]:
    """Side-by-side comparison of multiple experiments."""
    comparisons: list[dict[str, Any]] = []

    for eid in experiment_ids:
        exp = get_experiment(eid)
        if "error" in exp:
            comparisons.append({"experiment_id": eid, "error": exp["error"]})
            continue

        summary: dict[str, Any] = {
            "experiment_id": eid,
            "state": exp.get("status", {}).get("state"),
        }

        # Get final metrics
        metrics = exp.get("metrics", [])
        if metrics:
            summary["final_metrics"] = metrics[-1]
            summary["n_epochs"] = len(metrics)

        # Get config summary
        config = exp.get("config", {})
        model_source = config.get("model_source", {})
        summary["model"] = model_source.get("builder", "unknown")

        # Dataset identity (the content end of the reproduce-a-number chain), from the immutable lineage.
        lin = exp.get("lineage")
        if isinstance(lin, dict):
            summary["dataset_id"] = lin.get("dataset_id")
            summary["dataset_fingerprint"] = lin.get("dataset_fingerprint")

        comparisons.append(summary)

    # Whether every compared run trained on the same dataset content, a metric comparison across
    # different data is not apples-to-apples, so surface it rather than let the caller assume.
    # A run with no recorded fingerprint (bespoke/imageless) makes the comparison's data identity
    # unknown, not "same" by default, so an unset fingerprint must not be filtered out before the
    # equality check the way an errored comparison is.
    fps = {c.get("dataset_fingerprint") for c in comparisons if "error" not in c}
    same_dataset = None if (not fps or None in fps) else len(fps) == 1
    return {"experiments": comparisons, "count": len(comparisons), "same_dataset_fingerprint": same_dataset}


def get_experiment_lineage(experiment_id: str) -> dict[str, Any]:
    """Trace the full data → model → predictions chain."""
    if not experiment_exists(experiment_id):
        return {"error": f"Experiment not found: {experiment_id}"}

    lineage = read_member(lineage_key(experiment_id))
    if lineage is None:
        return {"error": "No lineage file found"}

    config = read_member(config_key(experiment_id))
    if isinstance(config, dict):
        data_cfg = config.get("data", {})
        lineage["data_config"] = {
            "images_dir": data_cfg.get("images_dir"),
            "labels_dir": data_cfg.get("labels_dir"),
            "task": data_cfg.get("task"),
        }

    return {"experiment_id": experiment_id, "lineage": lineage}


def read_split_manifest(experiment_id: str) -> dict[str, Any]:
    """The run's persisted split manifest, or ``{}`` when it was never written.

    The one reader of that record: the membership question and the spatial-region questions
    a calibration path asks are answered from one parse, so a consumer never re-derives the
    record's location or its key names.
    """
    manifest = read_member(split_key(experiment_id), {})
    return manifest if isinstance(manifest, dict) else {}
