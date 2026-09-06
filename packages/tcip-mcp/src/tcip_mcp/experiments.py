"""Experiment tracking for ML training runs.

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

This module declares the record's members, so it is also the one place they are addressed:
every reader and writer takes a key from a constructor here rather than composing a path of
its own, and ``experiment_dir`` serves the run artifacts that live beside the record without
being members of it (checkpoints, TensorBoard logs, a bespoke run's source snapshot).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
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

    ``cas``: every writer here reads the document and updates fields inside it, from the
    training subprocess and the tool process at once, so an unconditional write drops the
    heartbeat or the run identity another writer just stamped.

    ``launched_by`` is a mapping naming who launched the run, stamped by
    :func:`stamp_run_identity`: ``{"launcher": "gui"}`` for a launch through the web app's own
    route, ``{"launcher": "agent", **agent_identity.audit_fields()}`` for a launch inside an MCP
    handshake, ``{"launcher": "process"}`` for a launch from neither. Absent on a record whose
    experiment tracking never reached the stamp, or whose stamp otherwise failed; a reader
    treats an absent field as "launcher not recorded" rather than guessing. Provenance only:
    nothing reads it to decide anything.
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
        frozen=True,
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
        frozen=True,
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
        frozen=True,
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


# Once terminal, a record is immutable and additive-only. Excludes "cancelled": that record
# stays writable; a resume always mints a fresh id via _ensure_experiment rather than reopening it.
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


def experiment_exists(experiment_id: str, *, root: Path | str | None = None) -> bool:
    """Whether this id names a real experiment record, by its config snapshot.

    ``root`` defaults to the current platform root; a caller resolving a run under a root
    other than the one it started under (a launch's own watchdog, after the process has
    since adopted a different project) passes the launch root explicitly.
    """
    return store.exists(config_key(experiment_id, root=root))


def _current_state(experiment_id: str, *, root: Path | str | None = None) -> str | None:
    status = read_member(status_key(experiment_id, root=root), {})
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
    :func:`record_artifact` (and the shared :func:`pointer_frozen` predicate the callers outside
    this module consult) apply a narrower, additive-only rule instead: a still-empty field takes
    its first write even past terminal, only a populated one is frozen, so ``record_artifact``
    decides by that rule first and calls this only to build the message once it already has.
    :func:`update_lineage` calls this first instead, inside its own transaction, as its initial
    terminal gate; only when it raises does ``update_lineage`` fall back to the same
    additive-only rule to decide which of the fields it was given the refusal actually blocks.

    Never audits itself: an audit line is a log append, which cannot run inside a record
    transaction (``store.transaction`` only ever holds ``kind="record"`` keys), and a caller
    checking this from inside its own transaction would have the append raise
    ``TransactionMisuse``. The caller audits the refusal once its transaction has closed (or
    immediately, when it holds none).
    """
    if state in _TERMINAL_STATES:
        raise ExperimentTerminal(f"Experiment {experiment_id} is {state} (terminal); refusing to {op}.")


def _audit_refused(
    experiment_id: str, op: str, detail: dict[str, Any], *, root: Path | str | None = None
) -> None:
    """Record a refused post-terminal mutation in the log ``root`` names.

    Through :func:`record_event_or_raise`: the mutation this refusal reports already committed
    (or, for update_status/complete_run, decided not to), so a line that cannot be appended
    raises :class:`AuditEntryNotWritten` to the caller rather than vanishing silently.

    ``root`` is the root the refused write was scoped to (a launch's own watchdog passes the
    root it captured at launch); the default files the line under the current platform root,
    unchanged for every caller that has no root of its own to name.
    """
    from tcip_mcp.audit import record_event_or_raise

    record_event_or_raise("experiment_mutation_refused", {"experiment_id": experiment_id, "op": op,
                                                           **detail}, status="refused", scope=root)


def audit_refusal_reraising(experiment_id: str, op: str, detail: dict[str, Any],
                            refusal: ExperimentTerminal, *,
                            root: Path | str | None = None) -> None:
    """Audit a refusal and re-raise it, whether or not the audit line itself could be written.

    For every caller that lets an :class:`ExperimentTerminal` propagate rather than report it as
    a return value (``subprocess_worker.py``'s two provenance patches, ``training_tools.py``'s
    split-manifest write). Those callers sit under an outer ``except Exception`` that would
    swallow an :class:`~tcip_mcp.audit.AuditEntryNotWritten` raised on its own, together with
    the refusal it was recording, so a failed append is chained onto ``refusal`` (``raise refusal
    from audit_exc``) instead: the refusal always reaches the caller and the append failure
    stays visible on it.

    ``root`` names a platform root other than this process's own, the same escape hatch
    :func:`update_status` offers a caller resolving a run under a root other than the one it
    started under.
    """
    try:
        _audit_refused(experiment_id, op, detail, root=root)
    except Exception as audit_exc:
        raise refusal from audit_exc
    raise refusal


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
    """Whether an experiment record with this ``state`` and ``metrics_logged`` may still take a
    full ``config.json`` rewrite: ``state == "created"`` and no metrics logged yet.

    The one implementation of the pristine predicate. ``metrics_logged`` is the status record's
    own field (see :func:`log_metrics`), read by both callers: :func:`overwrite_config_if_pristine`
    reads it from inside the transaction that also reads ``state``; ``_ensure_experiment`` reads it
    first, on its own outside read, to decide whether to attempt that overwrite at all, so a
    relaunch under a non-pristine id mints its fresh id straight away instead of provoking (and
    auditing) a refusal nothing needed.
    """
    return state == "created" and not metrics_logged


def metrics_logged_of(status: dict[str, Any] | None) -> bool:
    """Whether a status record already carries the ``metrics_logged`` marker :func:`log_metrics`
    stamps before its first append: the one read of that field, shared by
    :func:`overwrite_config_if_pristine`'s own transaction and ``_ensure_experiment``'s outside
    read, rather than each re-deriving it from the record."""
    return bool(status.get("metrics_logged")) if isinstance(status, dict) else False


def overwrite_config_if_pristine(
    experiment_id: str, config: dict[str, Any], *, root: Path | str | None = None,
) -> dict[str, Any]:
    """Rewrite ``config.json`` with the config actually launched, but only while the experiment is
    still pristine (state == "created" and no epochs logged yet).

    A pre-created experiment's ``config.json`` is written once, at ``create_experiment`` time,
    before effective tiling geometry and the training seed are resolved (see
    ``training_tools.launch_training``). Reusing that id via ``_ensure_experiment``'s pristine-reuse
    branch would otherwise ship a permanently stale snapshot describing a config that was never
    trained. Refuses (and audits the refusal) once :func:`is_pristine` says the record is no
    longer pristine, a "created" record that already has metrics rows must stay protected too, not
    just the terminal-state lock alone.

    Both of :func:`is_pristine`'s inputs, ``state`` and ``metrics_logged``, are read from the
    same status record this opens one transaction over, closing the race a log-key read outside
    the transaction could not: a ``log_metrics`` call now decides pristineness by writing to that
    same record, under the same lock, rather than a key no record transaction can name.

    ``root`` names a platform root other than this process's own, the same escape hatch
    :func:`update_status` offers a caller resolving a run under a root other than the one it
    started under.
    """
    check_json_value(config, path="config")
    if not experiment_exists(experiment_id, root=root):
        return {"error": f"Experiment not found: {experiment_id}"}
    cfg_key, st_key = config_key(experiment_id, root=root), status_key(experiment_id, root=root)
    state: str | None = None
    metrics_logged = False
    refused = False
    with store.transaction(cfg_key, st_key) as txn:
        status = txn.read(st_key, default={})
        state = status.get("state") if isinstance(status, dict) else None
        metrics_logged = metrics_logged_of(status)
        refused = not is_pristine(state, metrics_logged)
        if not refused:
            txn.write(cfg_key, config)
    if refused:
        _audit_refused(experiment_id, "overwrite_config_if_pristine",
                       {"state": state, "metrics_logged": metrics_logged}, root=root)
        return {"error": f"Experiment {experiment_id} is no longer pristine; refusing to "
                         f"overwrite its config.json."}
    return {"experiment_id": experiment_id, "overwritten": True}


def _mark_completed(status: dict[str, Any]) -> None:
    """Write the terminal completed state into a status record in place: state, heartbeat and
    ended timestamp together, the one transition :func:`update_status` and :func:`complete_run`
    both apply when a run finishes normally, so neither carries its own copy of the triple."""
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

    A repeat of the record's current state is idempotent: nothing restamps (not ``heartbeat``,
    not ``ended``), and ``error`` lands only when the record does not already carry one, so the
    watchdog's reasoned ``failed`` landing after the child's own reasonless ``failed`` still
    records the wall-clock reason, and a second reason never overwrites a first. Any other write
    to a terminal record (``completed``/``failed``, the other terminal state included) refuses
    through :func:`refuse_if_terminal`, audited once the transaction closes. ``cancelled`` is not
    terminal here, so a record in that state still takes any write, including back to ``running``.

    ``error`` records a specific failure reason (e.g. a wall-clock-timeout kill) into
    ``status.json["error"]``; outside the idempotent-repeat case above, omitted/``None`` never
    clears a previously-recorded error, only an explicit new value overwrites it.

    ``root`` defaults to the current platform root; a launch's wall-clock watchdog passes the
    root it captured at launch, so its write reaches the run's own record even after this
    process has since adopted a different project.
    """
    if not experiment_exists(experiment_id, root=root):
        return {"error": f"Experiment not found: {experiment_id}"}

    key = status_key(experiment_id, root=root)
    current: str | None = None
    refused = False
    with store.transaction(key) as txn:
        status = txn.read(key, default={})
        current = status.get("state")
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
        _audit_refused(experiment_id, "update_status", {"from": current, "to": state}, root=root)
        return {"error": f"Experiment {experiment_id} is {current} (terminal); refusing to "
                         f"move it to {state!r}.", "state": current}
    return {"experiment_id": experiment_id, "state": state}


def complete_run(
    experiment_id: str, final_weights: str, *, root: Path | str | None = None,
) -> dict[str, Any]:
    """Mark a run completed and record its final weights pointer and their digest, as one
    transaction.

    Hashes ``final_weights`` before opening the transaction: the store's locks cover records, not
    a checkpoint read, and holding three record locks through that read against the file backend's
    lock timeout would buy nothing. A declared deliverable this run cannot read is refused, not
    completed with a phantom pointer: a missing or unreadable file returns an error naming the
    path and the read failure, and writes nothing (no transaction is even opened).

    Names the artifacts key before the lineage key, and the lineage key before the status key: a
    file-backend transaction applies its writes in named-key order and is not crash-atomic across
    keys, so a crash mid-apply leaves a detectably stale record (a pointer with no digest, or a
    digest recorded on a record still ``running``), never a ``completed`` record carrying a
    mismatched or absent digest. Refuses (and audits, once the transaction has closed) a run
    already terminal, naming the weights file that exists on disk so an operator can find it; the
    refusal's ``state`` carries the state the record actually holds, so a caller can reconcile to
    it. The digest is what this call observed of the file, sealed into the transaction that makes
    the run terminal, so nothing a caller does to the path afterwards changes what the run
    recorded.

    ``root`` names a platform root other than this process's own, the same escape hatch
    :func:`update_status` offers a caller resolving a run under a root other than the one it
    started under.
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
            status = txn.read(st_key, default={})
            current = status.get("state") if isinstance(status, dict) else None
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
        _audit_refused(experiment_id, "complete_run", {"final_weights": final_weights}, root=root)
        return {"error": f"{exc} Final weights at {final_weights!r} were not recorded.",
                "final_weights": final_weights, "state": current}

    return {"experiment_id": experiment_id, "state": "completed", "model_weights": final_weights,
            "model_weights_sha256": digest}


def stamp_run_identity(
    experiment_id: str, run_id: str, output_dir: str, *, launched_by: dict[str, Any],
) -> None:
    """Record which ``run_id``/``output_dir`` produced this experiment, and who launched it, into
    ``status.json``.

    Best-effort, like ``_touch_heartbeat``, a dropped stamp must not break the launch it's
    recording. Called unconditionally by ``_ensure_experiment`` regardless of which of its three
    branches resolved ``experiment_id`` (fresh creation, pristine pre-created-experiment reuse, or a
    fresh-id conflict), those are the only paths that mint a real, running experiment, and this
    is what makes the real artifact directory (``output_dir``, a separately-computed, caller-influenced
    path that only coincides with the experiment directory by convention) discoverable from
    ``experiment_id``/``run_id`` alone by a different process.

    ``launched_by`` is the run currently being stamped, resolved once by ``launch_training``
    before ``_ensure_experiment`` runs: ``{"launcher": "agent", **agent_identity.audit_fields()}``
    inside an MCP handshake, ``{"launcher": "gui"}`` for the Training tab's launch route,
    ``{"launcher": "process"}`` for a caller with neither. Written whole, in the same transaction
    as ``run_id``/``output_dir``, so a relaunch that re-stamps a pristine id never leaves it
    disagreeing with the identity it was just re-stamped with. Provenance only: nothing reads it
    to decide anything, since a best-effort field that can go missing cannot guard a decision. A
    record with no ``launched_by`` at all is one that predates the field or whose stamp was
    dropped; every reader treats the two cases the same, as "launcher not recorded".
    """
    key = status_key(experiment_id)
    if not store.exists(key):
        return
    try:
        with store.transaction(key) as txn:
            status = txn.read(key, default={})
            status["run_id"] = run_id
            status["output_dir"] = output_dir
            status["launched_by"] = launched_by
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


def is_launched(status: dict[str, Any] | None) -> bool:
    """Whether a status record names a launched run rather than one only created: a stamped
    ``run_id``, a state other than ``"created"``, or the ``metrics_logged`` marker
    :func:`log_metrics` stamps before its first append, so a launch whose best-effort
    :func:`stamp_run_identity` or status write was lost still counts. The one implementation of
    "was this ever launched", shared by ``training_tools.py``'s own run enumeration and
    :func:`compare_experiments`, which consults it before deriving a heartbeat state at all: a
    pre-created, never-launched record carries a heartbeat of ``None`` and would otherwise derive
    to ``"interrupted"``, misreporting a run that never started as a crashed one.
    """
    if not isinstance(status, dict):
        return False
    return bool(status.get("run_id")) or status.get("state") != "created" or bool(status.get("metrics_logged"))


def derived_state(status: dict[str, Any], stale_seconds: float) -> str:
    """The state a status record reads as once heartbeat freshness applies: a state already
    recorded as done (:data:`_RECORDED_AS_DONE`) is trusted as-is; any other state derives to
    ``"running"`` while the heartbeat is fresh, else ``"interrupted"``. The one implementation
    :func:`reconstruct_from_status` and :func:`compare_experiments` both read through, so a run
    whose process died reads the same way everywhere rather than as ``"running"`` in one place.
    """
    state = status.get("state", "unknown") if isinstance(status, dict) else "unknown"
    heartbeat = status.get("heartbeat") if isinstance(status, dict) else None
    if state not in _RECORDED_AS_DONE:
        state = "running" if _heartbeat_fresh(heartbeat, stale_seconds) else "interrupted"
    return state


def reconstruct_from_status(
    experiment_id: str, status: dict[str, Any], *, stale_seconds: float, read_progress: bool,
) -> dict[str, Any]:
    """One record's run row, reconstructed from a status document the caller already read: the
    shape :func:`reconstruct_run_status` returns for the one record it resolved a ``run_id`` to,
    and the shape the run enumeration in ``training_tools.py`` builds per record without a
    separate resolver round-trip. ``current_epoch``, ``best_metric`` and ``best_metric_name``
    cost one metrics-log read and are included only when ``read_progress`` is true, read back
    through :func:`best_selection_from_log` from what the run itself stamped, never re-derived
    from the config.

    ``launched_by`` carries the record's own stamped declaration (see :func:`stamp_run_identity`)
    whole, or ``None`` for a record that predates the field or whose stamp was dropped; a reader
    treats both the same way, as "launcher not recorded".
    """
    current_epoch = None
    best_metric_name = None
    best_metric = None
    if read_progress:
        rows = read_metrics(experiment_id)
        current_epoch = rows[-1].get("epoch") if rows else None
        best_metric_name, best_metric = best_selection_from_log(rows)
    return {
        "run_id": status.get("run_id", experiment_id),
        "experiment_id": experiment_id,
        "status": derived_state(status, stale_seconds),
        "current_epoch": current_epoch,
        "best_metric": best_metric,
        "best_metric_name": best_metric_name,
        "output_dir": status.get("output_dir"),
        "error": status.get("error"),
        "launched_by": status.get("launched_by"),
    }


def reconstruct_run_status(run_id: str, *, stale_seconds: float = 600.0) -> dict[str, Any] | None:
    """Reconstruct a run's status from disk for a caller whose in-memory registry doesn't
    have it, either it was never in this process (a different process launched it) or it was
    subprocess-delegated and the in-memory record is stale by design.

    Returns ``None`` when the run can't be resolved on disk at all (an honestly unknown run, not a
    guess). ``stale_seconds`` lets ``training_tools.py``'s own callers (``monitor_training``,
    ``cancel_training``) pass their configured heartbeat window (``TCIP_HEARTBEAT_STALE_SECONDS``)
    rather than being pinned to this module's default. The reconstruction itself is
    :func:`reconstruct_from_status`, over the one record this function resolves ``run_id`` to.
    """
    experiment_id = resolve_experiment_for_run(run_id)
    if experiment_id is None:
        return None
    status = read_member(status_key(experiment_id))
    if not isinstance(status, dict):
        return None
    return reconstruct_from_status(experiment_id, status, stale_seconds=stale_seconds,
                                   read_progress=True)


def _parse_iso_instant(value: Any) -> datetime | None:
    """One timestamp read as an instant: a non-string, or a string ``datetime.fromisoformat``
    can't parse (a trailing ``Z`` and an offset both parse fine on this platform's Python), is
    ``None`` rather than a raise, so a caller comparing many rows skips a malformed one instead of
    aborting the whole comparison. Every platform writer stamps UTC with an explicit offset; a
    naive value (a row a bespoke loop appended with its own clock) is read as UTC so it compares
    against those on one clock rather than raising. Shared by :func:`_heartbeat_fresh` and
    :func:`compare_experiments`'s ``rows_after_end``, the one place either fact is parsed.
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


def _heartbeat_fresh(hb_iso: str | None, stale_seconds: float = 600.0) -> bool:
    """True if ``hb_iso`` (ISO-8601) is within the staleness window, a process is still actively
    updating this run. Missing/unparseable → not fresh (treat as dead). This module's own
    default window; a caller with a configured one (``training_tools.py``'s
    ``TCIP_HEARTBEAT_STALE_SECONDS``, read from ``$TCIP_HEARTBEAT_STALE_SECONDS``) passes its own
    ``stale_seconds`` through rather than being pinned to this default."""
    hb = _parse_iso_instant(hb_iso)
    if hb is None:
        return False
    return (datetime.now(timezone.utc) - hb).total_seconds() <= stale_seconds


def _touch_heartbeat(experiment_id: str, *, root: Path | str | None = None) -> None:
    """Best-effort: stamp the current time into ``status.json['heartbeat']``.

    Called each epoch so a run still training in another process (e.g. the MCP agent) reads
    as live to a web client reconstructing run state, instead of being flagged interrupted.
    Never raises, a heartbeat failure must not break metric logging.
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
    epoch's value on it); this reads those back rather than re-deriving a name from the run's
    config, which is a second, disagreeing resolution (a bespoke loop's rows, or a config whose
    ``training.evaluation`` and top-level ``evaluation`` blocks differ, are exactly where a second
    resolution drifts from the trainer's own).

    The name is the most recently stamped one; the best is compared over every row stamped with
    that name in its declared ranking direction (``evaluation.HIGHER_IS_BETTER_BY_METRIC`` on the
    bare name, the same declaration ``resolve_selection_metric`` enforces before a run can select
    on it). A row with no name or a non-finite value is skipped when finding the best. No name in
    any row, or a name the declaration table does not carry, leaves both ``None``: a metric this
    function cannot rank is never guessed at.
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

    The one writer of that log: a training body routes its rows here rather than opening the
    file beside it, so the module that declares the record's members is the module that
    appends to them and the terminal-state lock cannot be written around.

    A bespoke loop's row is its own dict, so it is checked field by field first: a tensor or
    a non-finite loss is named here, where the caller can see which metric it was.

    ``check_json_value`` admits any JSON-encodable value, wider than the frontend's own
    ``MetricRow`` type (``number | string | null | undefined`` per key), which renders only the metric
    shapes it recognizes and drops the rest silently.

    Stamps ``status.json["metrics_logged"] = True`` before appending, the record
    :func:`is_pristine` reads instead of a log key no record transaction can name. The marker
    goes before the append, not after, so a marker written but then an append that fails still
    reads non-pristine, never the reverse. A failure inside that marker transaction is left to
    raise rather than caught: no marker means no row follows it, the safe direction.

    ``root`` names a platform root other than this process's own, the same escape hatch
    :func:`update_status` offers a caller resolving a run under a root other than the one it
    started under.
    """
    check_json_value(metrics, path="metrics")
    if not experiment_exists(experiment_id, root=root):
        return {"error": f"Experiment not found: {experiment_id}"}

    # Terminal-state lock: a completed/failed run's metric history is frozen, no new epochs.
    try:
        refuse_if_terminal(experiment_id, "log_metrics", _current_state(experiment_id, root=root))
    except ExperimentTerminal as exc:
        _audit_refused(experiment_id, "log_metrics", {"epoch": epoch}, root=root)
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
reference, applicable only when the calibration named a split manifest or the checkpoint carries
a ``manifest_binding``, ``null`` for ``resolve_scale``.
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


def _pointer_populated(doc: dict[str, Any], field: str) -> bool:
    """Whether ``field`` already carries a real value in ``doc`` (an artifacts or lineage
    record): present and not an empty placeholder. The additive-lock's own definition of
    "populated", shared by every writer and pre-checker that consults it."""
    return doc.get(field) not in (None, "", [], {})


def _artifact_write_refused(doc: dict[str, Any], field: str, value: Any) -> bool:
    """The artifacts member's additive lock: a name already present freezes, regardless of
    whether the value recorded under it is itself falsy. Presence, not populated-ness, is the
    lock's baseline semantics here, so a name recorded with a falsy entry stays frozen too."""
    return field in doc


def _lineage_write_refused(doc: dict[str, Any], field: str, value: Any) -> bool:
    """The lineage member's additive lock: a populated field freezes unless the write would
    record the same value it already holds, so a repeat of an idempotent write is admitted rather
    than refused (a re-export whose own bucket resolved to no document on its first, empty-input
    pass, so a second, real pass in place records the same path a terminal experiment already
    holds)."""
    return _pointer_populated(doc, field) and doc.get(field) != value


# One predicate per member, evaluated by pointer_frozen, record_artifact and update_lineage alike
# so the three never carry separate copies of what "frozen" means for that member.
_MEMBER_WRITE_REFUSED = {"artifacts": _artifact_write_refused, "lineage": _lineage_write_refused}
_POINTER_MEMBER_KEYS = {"artifacts": artifacts_key, "lineage": lineage_key}


def pointer_frozen(experiment_id: str, member: str, field: str, value: Any) -> str | None:
    """Whether writing ``value`` into ``field`` of the named member (``"artifacts"`` or
    ``"lineage"``) would be refused right now: the experiment terminal and the member's own
    additive lock (:data:`_MEMBER_WRITE_REFUSED`), the same predicate :func:`record_artifact` and
    :func:`update_lineage` evaluate inside their transactions.

    An untransacted pre-check for a caller about to write a file outside the store (a blob write
    cannot join a record transaction), so it can refuse by name before writing anything, rather
    than after. The window between this read and the write is real, the record could still turn
    terminal in between, and the writer's own transactional refusal, sharing this same predicate,
    still catches that residual; this only spares the ordinary case its file write. Returns the
    refusal text, or ``None`` when the write would be admitted.
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

    ``root`` names a platform root other than this process's own, the same escape hatch
    :func:`update_status` offers a caller resolving a run under a root other than the one it
    started under.
    """
    if not experiment_exists(experiment_id, root=root):
        return {"error": f"Experiment not found: {experiment_id}"}

    key, state = artifacts_key(experiment_id, root=root), status_key(experiment_id, root=root)
    current: str | None = None
    refused_overwrite = False
    with store.transaction(key, state) as txn:
        artifacts = txn.read(key, default={})
        current = (txn.read(state, default={}) or {}).get("state")
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
            _audit_refused(experiment_id, "record_artifact", {"artifact": name, "path": path},
                          root=root)
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

    The updates are the caller's own kwargs, merged whole into the stored document, so they
    are checked against what JSON can hold before any of them lands. ``model_weights`` and
    ``model_weights_sha256`` are ``complete_run``'s alone: naming either raises ``ValueError``
    before any field, including a legitimate companion in the same call, lands.

    A dropped identity update is audited before the transaction below (it never reaches the
    transaction at all, see the comment on ``identity_updates``); when that append itself fails,
    the failure is not raised here, it would abort this call before the transaction applies the
    other, legitimate updates it was given. It is deferred and raised at the end instead, once
    those have landed, so the append failure still reaches the caller rather than being logged
    away, and never at the cost of a write that should have gone through.

    ``root`` names a platform root other than this process's own, the same escape hatch
    :func:`update_status` offers a caller resolving a run under a root other than the one it
    started under.
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
    identity_audit_exc: Exception | None = None
    if identity_updates:
        try:
            _audit_refused(experiment_id, "update_lineage_identity",
                          {"fields": sorted(identity_updates)}, root=root)
        except Exception as exc:
            identity_audit_exc = exc

    key, state = lineage_key(experiment_id, root=root), status_key(experiment_id, root=root)
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
                       if _MEMBER_WRITE_REFUSED["lineage"](lineage, k, v)}
            if refused:
                updates = {k: v for k, v in updates.items() if k not in refused}
        lineage.update(updates)
        txn.write(key, lineage)

    if refused:
        # Names the orphaned values themselves: for a path-like field (predictions) that value
        # is the file this refusal left unrecorded.
        _audit_refused(experiment_id, "update_lineage", {"fields": sorted(refused), **refused},
                      root=root)
    if identity_audit_exc is not None:
        raise identity_audit_exc
    return {"experiment_id": experiment_id, "lineage": lineage}


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
    run in ``created`` or ``running``, or one that ended ``failed``/``cancelled``, has not said
    what it produced and refuses by name. Hashes the caller's ``checkpoint_path`` through the same
    function completion hashed with and refuses when the two digests differ, naming both and the
    path completion recorded: the caller's path may be the recorded path or any byte-identical
    copy, never a different file. ``project_path``, when given, must be the directory the
    experiment's own keys hang off (compared through ``splits.same_directory``, tolerant of a
    different spelling of the same directory); any other directory refuses by name, naming the
    root this call would otherwise have searched.

    Pulls the experiment's config and the checkpoint's own metrics, the epoch that produced this
    checkpoint (e.g. ``model_best.pt``'s best epoch), not necessarily the last training epoch.
    Registers with no tags (the retired ``experiment:<id>`` convention named no verified fact) and
    writes only the registry's own entry, with ``experiment_id`` set to this run: the run's own
    lineage pointer (``model_weights``, ``model_weights_sha256``) is ``complete_run``'s alone, not
    this call's. Metrics are read, never fabricated: a checkpoint that carries no metrics dict, or
    that will not load at all, registers with an empty ``metrics`` and a ``metrics_source`` of
    ``None`` rather than substituting a different epoch's numbers (the run's own metrics log
    describes a different model state than the checkpoint being registered). The unpickle and its
    ``schema_version`` check are ``model_registry._load_verified_payload``, the same function
    ``load_registered_checkpoint`` calls after its own registry-name identity check: this call's
    identity check is the digest match above instead, but the payload rules (``weights_only=True``,
    the version ceiling) are one implementation, never a second ``torch.load`` of its own.

    ``metrics_source`` records which path produced the numbers, not that anyone verified them:
    ``"trainer"`` when the run's config carries no ``training_source`` (the platform's own
    ``default_train`` computed them), ``"training_source"`` when it does (a bespoke loop's own
    saved state), and ``None`` when the checkpoint carries no metrics.

    A name a prior run bound is not evicted by this call unless ``experiment_id`` is that same
    run: the registry's own eviction rail refuses, returned here as an error naming the run that
    holds the name (see ``model_registry.EntryOwnedByRun``).

    ``root`` names the platform root the experiment's own keys hang off, other than this
    process's own, the same escape hatch :func:`update_status` offers a caller resolving a run
    under a root other than the one it started under; the default is this process's pinned
    platform root, and ``project_path``'s own check is against whichever one applies.
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

    status = read_member(status_key(experiment_id, root=root), {})
    state = status.get("state") if isinstance(status, dict) else None
    lineage = read_member(lineage_key(experiment_id, root=root), {})
    recorded_digest = lineage.get("model_weights_sha256") if isinstance(lineage, dict) else None
    recorded_path = lineage.get("model_weights") if isinstance(lineage, dict) else None
    if state != "completed" or not recorded_digest:
        _audit_refused(experiment_id, "register_model_from_experiment", {
            "checkpoint_path": checkpoint_path, "caller_sha256": digest,
            "recorded_sha256": recorded_digest, "recorded_path": recorded_path,
        }, root=root)
        return {"error": f"experiment {experiment_id!r} has not completed with a recorded "
                         f"digest (state={state!r}): its run has not said what it produced. "
                         "complete_run records the digest when the run finishes."}

    if digest != recorded_digest:
        _audit_refused(experiment_id, "register_model_from_experiment", {
            "checkpoint_path": checkpoint_path, "caller_sha256": digest,
            "recorded_sha256": recorded_digest, "recorded_path": recorded_path,
        }, root=root)
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
        kind = payload.get("kind")  # stamped by the trainer; None on older checkpoints
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
    order, oldest first. Its last row is only the last one logged, not a verified result, and
    nothing binds a row written to that log outside :func:`log_metrics` itself, so a row a
    bespoke loop appended reaches display through this reader (and :func:`compare_experiments`'s
    ``last_logged_metrics``) exactly like any other, and neither one is a promotion decision:
    registering a checkpoint (:func:`register_model_from_experiment`) reads that checkpoint's own
    stamped metrics, never this log, and ranking a registered model (``model_registry.
    best_model``) reads the registry entry's own ``metrics_source``, not this log either.

    ``metrics`` can be paginated for long runs: ``metrics_offset`` and ``metrics_limit`` index
    into the row list, so ``n_rows`` (the row count) is the paging bound, not ``n_epochs`` (the
    count of distinct ``epoch`` values; the stock loop logs one row per epoch, a bespoke one may
    log several). Defaults return all metrics. ``validations`` is the whole claim history,
    unpaginated: a run earns few claims where it logs many epochs.
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

    Covers a calibration experiment (id derived from a claim's content, unreconstructable any
    other way), a review-feedback lineage, a pre-created experiment never launched, and a
    launched one whose ``run_id`` stamp was lost, none of which the tool door's
    ``launched_only=True`` view (a launched record only) lists. The ids come from the same
    status-record enumeration
    :func:`resolve_experiment_for_run` resolves against, so a record the resolver finds is a
    record this lists. ``run_id`` is the stamp :func:`stamp_run_identity` recorded, or ``None``
    when a launch never reached it or the record was never launched at all;
    ``has_model_source`` is whether the config carries a ``model_source`` (a training run) versus
    an experiment that tracks something else.
    """
    from tcip_mcp.pipelines.model_build import MODEL_SOURCE_KEY

    experiments = []
    for experiment_id in experiment_ids_with_status(None):
        status = read_member(status_key(experiment_id))
        if isinstance(status, dict):
            config = read_member(config_key(experiment_id), {})
            experiments.append({
                "experiment_id": experiment_id,
                "state": status.get("state", "unknown"),
                "created": status.get("created"),
                "run_id": status.get("run_id"),
                "has_model_source": bool(isinstance(config, dict) and config.get(MODEL_SOURCE_KEY)),
            })

    return experiments


def _index_refused_mutations(experiment_ids: list[str]) -> dict[str, list[dict[str, Any]]] | None:
    """Every ``experiment_mutation_refused`` audit entry naming one of ``experiment_ids``, indexed
    by ``arguments.experiment_id``, from one scan of the platform audit log, the read
    :func:`compare_experiments` shares across every experiment it compares rather than repeating
    per experiment. ``None`` for the whole call when the log can't be read: the read itself
    raised, or :func:`~tcip_store.read_log` reports entries this reader could not use, corrupt
    bytes on ``page.corrupt`` or an unsupported schema_version on ``page.version_refused``,
    either kept from raising so an unreadable page would otherwise look like a page with
    nothing to report. Either way every experiment's own field is then absent, never an empty
    list, so "no refusals" and "couldn't read the log" are never confused for each other. An id
    present in the index only when it has at least one entry; an id with none is absent from the
    index and the caller reads that as an empty list, not as unreadable. A refusal line lands
    only under the root that holds the record, which is the root this reader must be pinned to
    for the experiment to resolve at all, so the one-root scan is complete for every experiment
    it can answer for.
    """
    try:
        from tcip_store import read_log

        from tcip_mcp.audit import audit_log_key

        page = read_log(audit_log_key())
    except Exception:
        return None
    if page.corrupt:
        return None
    if page.version_refused:
        return None
    wanted = set(experiment_ids)
    index: dict[str, list[dict[str, Any]]] = {}
    for e in page.records:
        if e.get("tool") != "experiment_mutation_refused":
            continue
        arguments = e.get("arguments", {})
        eid = arguments.get("experiment_id")
        if eid not in wanted:
            continue
        index.setdefault(eid, []).append({"timestamp": e.get("timestamp"), "arguments": arguments})
    return index


def _split_summary(experiment_id: str) -> dict[str, Any]:
    """The partition column for one experiment: :func:`read_split_manifest_checked` reduced to
    the four states a comparison names. ``{"case": "error", "error": ...}`` for a record that
    exists but will not decode; ``{"case": "none"}`` for a run that never wrote one;
    ``{"case": "bound", "manifest_dir": ..., "seed": ..., "redrawn_within_manifest": bool}`` for
    a run bound to a named split manifest (``split.json``'s own ``manifest_binding``), the flag
    read from ``split.json``'s own top-level ``redrawn_within_manifest`` so a manifest bound at
    seed 42 and a redraw inside that same manifest at seed 42 never compare as the same data;
    ``{"case": "drawn", "seed": ...}`` otherwise.
    """
    manifest, decode_error = read_split_manifest_checked(experiment_id)
    if decode_error is not None:
        return {"case": "error", "error": decode_error}
    if not manifest:
        return {"case": "none"}
    binding = manifest.get("manifest_binding")
    if isinstance(binding, dict) and binding.get("manifest_dir"):
        return {
            "case": "bound", "manifest_dir": binding["manifest_dir"], "seed": manifest.get("seed"),
            "redrawn_within_manifest": bool(manifest.get("redrawn_within_manifest")),
        }
    return {"case": "drawn", "seed": manifest.get("seed")}


def _index_registry_entries(
    experiment_ids: list[str],
) -> tuple[dict[str, list[dict[str, Any]]] | None, str | None]:
    """Every registered model entry naming one of ``experiment_ids`` as its producer, indexed by
    experiment id, from one read of the platform root's registry index, the root every column
    this comparison shows is read from. Returns ``(index, error)``: ``index`` is ``None`` and
    ``error`` names why whenever the caller must not treat "no entries matched" as an answer:
    the index document itself could not be read (an absent registry reads as a legitimate empty
    index, not this case), or the registry carries an entry with no ``metrics_source`` key at
    all (an entry predating the field, the same condition :meth:`ModelRegistry.best_model`
    refuses ranking on), named rather than silently matching nothing. Otherwise ``index`` holds
    every match, each entry reduced to ``name``, ``metrics``, ``metrics_source``,
    ``registered_at``; an experiment with no registered entry is simply absent from the index.
    """
    from tcip_mcp.model_registry import read_registry_index
    from tcip_mcp.project_paths import platform_state_root

    try:
        entries = read_registry_index(platform_state_root())
    except Exception as exc:
        return None, f"registry unreadable: {exc}"

    stale = sorted(str(e.get("name")) for e in entries if "metrics_source" not in e)
    if stale:
        return None, (f"registry entries {stale} predate metrics_source and cannot be matched "
                       "to a producing experiment or ranked until conformed")

    wanted = set(experiment_ids)
    index: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        eid = e.get("experiment_id")
        if eid not in wanted:
            continue
        index.setdefault(eid, []).append({
            "name": e.get("name"), "metrics": e.get("metrics"),
            "metrics_source": e.get("metrics_source"), "registered_at": e.get("registered_at"),
        })
    return index, None


def compare_experiments(experiment_ids: list[str], *, stale_seconds: float = 600.0) -> dict[str, Any]:
    """Side-by-side comparison of multiple experiments.

    ``stale_seconds`` is the heartbeat freshness window :func:`derived_state` applies; this
    module's own default, a caller with the configured window (``training_tools.py``'s
    ``TCIP_HEARTBEAT_STALE_SECONDS``) passes it through rather than being pinned to this default,
    the same knob :func:`reconstruct_run_status` and the run enumeration take.

    Per experiment: ``recorded_state`` (the stored state) and ``state``, the same
    heartbeat-derived state the run enumeration reports, via :func:`derived_state`, so a run
    whose process died compares as ``"interrupted"`` rather than ``"running"``, but only for a
    launched record (:func:`is_launched`): a pre-created experiment never launched reports
    ``state`` equal to ``recorded_state`` (``"created"``), never a heartbeat-derived
    ``"interrupted"`` implying a crash that never happened. ``log_locked``, true when
    ``recorded_state`` is in the mutation lock's terminal set, no tamper claim, only that
    :func:`log_metrics` refuses further rows (a cancelled run reads ``log_locked`` false: the
    lock admits rows there, even though no production flow appends to one); ``last_logged_metrics``,
    the run's own log's last row (not ``"final"``: an unlocked log can still take a row after it,
    and this is not a verified result, see :func:`get_experiment`); ``rows_after_end``, the count
    of rows whose ``timestamp`` is a later instant than the record's own ``ended`` (the one row an
    unlocked log's own append can admit after the mark, or any row an outside writer appended
    later), ``None`` when the record has no ``ended``, and a row whose own ``timestamp`` is
    missing or unparseable never counted rather than raising; ``n_epochs``/``n_rows``, always
    present; and ``refused_mutations``, every refused write the platform audit log recorded
    against this experiment (see :func:`_index_refused_mutations`), absent rather than empty when
    that log itself can't be read.

    Also per experiment: ``task``/``subject`` from the config already read; ``status_error``, the
    status record's own failure reason (``None`` for a run that never failed, distinct from a
    comparison entry's own top-level ``error`` when :func:`get_experiment` could not read it at
    all); ``model``, the config's
    builder, ``None`` when the config names none (never a fabricated ``"unknown"``); ``split``
    (see :func:`_split_summary`), so two runs of one config on different partitions never compare
    as the same data; and ``registry``, this experiment's own registered entries (see
    :func:`_index_registry_entries`), absent, with ``registry_error`` naming why, when the
    project's registry index can't be read or matched at all (an experiment with no registered
    entry still carries ``registry: []``). ``same_dataset_fingerprint`` is ``None``, never
    ``True``, when any compared id is an error entry: a record this call could not even read must
    never be silently dropped from the same-data judgment.

    Reading refused mutations and the registry each cost one scan for the whole call, on top of
    one :func:`get_experiment` and one :func:`_split_summary` per experiment compared.
    """
    from tcip_mcp.pipelines.model_build import MODEL_SOURCE_KEY

    comparisons: list[dict[str, Any]] = []
    refused_index = _index_refused_mutations(experiment_ids)
    registry_index, registry_error = _index_registry_entries(experiment_ids)

    for eid in experiment_ids:
        exp = get_experiment(eid)
        if "error" in exp:
            comparisons.append({"experiment_id": eid, "error": exp["error"]})
            continue

        status_doc = exp.get("status")
        status_doc = status_doc if isinstance(status_doc, dict) else {}
        recorded_state = status_doc.get("state")
        ended = status_doc.get("ended")

        summary: dict[str, Any] = {
            "experiment_id": eid,
            "recorded_state": recorded_state,
            "state": derived_state(status_doc, stale_seconds) if is_launched(status_doc) else recorded_state,
            "log_locked": recorded_state in _TERMINAL_STATES,
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

        if refused_index is not None:
            summary["refused_mutations"] = refused_index.get(eid, [])

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
        summary["task"] = model_source.get("task")
        data_cfg = config.get("data", {})
        summary["subject"] = data_cfg.get("subject") if isinstance(data_cfg, dict) else None

        # Dataset identity (the content end of the reproduce-a-number chain), from the immutable lineage.
        lin = exp.get("lineage")
        if isinstance(lin, dict):
            summary["dataset_id"] = lin.get("dataset_id")
            summary["dataset_fingerprint"] = lin.get("dataset_fingerprint")

        comparisons.append(summary)

    # Whether every compared run trained on the same dataset content, so a caller doesn't assume
    # apples-to-apples; an unset or formula-unrecorded fingerprint makes it unknown, not "same".
    from tcip_mcp.pipelines.data.dataset_fingerprint import (
        FINGERPRINT_FORMULA_VERSION, fingerprint_formula_version,
    )

    any_error = any("error" in c for c in comparisons)
    with_fp = [c for c in comparisons if "error" not in c]
    for c in with_fp:
        fp = c.get("dataset_fingerprint")
        if fp is not None and fingerprint_formula_version(fp) != FINGERPRINT_FORMULA_VERSION:
            c["fingerprint_formula_unrecorded"] = True
    fps = {c.get("dataset_fingerprint") for c in with_fp}
    any_unrecorded = any(c.get("fingerprint_formula_unrecorded") for c in with_fp)
    # A record this call could not even read (any_error) must never be silently dropped from the
    # judgment: the remaining columns matching each other says nothing about the missing one.
    same_dataset = None if (any_error or not fps or None in fps or any_unrecorded) else len(fps) == 1
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


def read_split_manifest_checked(
    experiment_id: str, *, root: Path | str | None = None,
) -> tuple[dict[str, Any], str | None]:
    """The run's persisted split manifest, and the decode failure behind an unreadable one.

    Returns ``(manifest, decode_error)``. ``manifest`` is ``{}`` with ``decode_error`` ``None``
    for a run that never wrote one: nothing to say. A record that exists but will not decode also
    answers ``manifest={}``, but ``decode_error`` names why, so a caller that must not read a
    bound run's corrupted record as an unbound one (a calibration-side mark that would otherwise
    guess membership from a manifest that no longer exists) can tell the two apart.
    ``root`` resolves the same project a caller's own checkpoint lookup used, so an explicit one
    cannot pair one project's producer with another project's record.
    """
    from tcip_store import DecodeError

    try:
        manifest = store.read(split_key(experiment_id, root=root), default={})
    except DecodeError as exc:
        return {}, str(exc)
    return (manifest if isinstance(manifest, dict) else {}), None


def read_split_manifest(experiment_id: str) -> dict[str, Any]:
    """The run's persisted split manifest, or ``{}`` when it was never written or could not be
    decoded.

    Folds a decode failure onto the same ``{}`` an absent record answers: every consumer here
    already treats a corrupt manifest the way it treats a missing one, and none needs the two
    told apart. :func:`read_split_manifest_checked` is the one reader that keeps them apart, for
    a caller that must not guess membership from a manifest that no longer exists.
    """
    manifest, _ = read_split_manifest_checked(experiment_id)
    return manifest
