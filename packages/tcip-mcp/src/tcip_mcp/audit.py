"""Audit logging decorator for MCP tools.

Every tool call is logged with timestamp, tool name, arguments, result status, and duration, into
an append-only store: entries are added, never rewritten. The bound backend decides where those
entries sit, one JSON object per line on the file backend.

One store, ``audit_log``, addressed under three kinds of root: the platform audit log (the
pinned platform state root, the default when a caller names no other), a dataset's audit log
(a record that travels with the data), and a project's audit log (a record that is the project's
own, such as a delivery or a plant-mapping build). A project that has been adopted (see
``project_paths``) coincides with the platform root, so from then on a project's own log and the
platform log are one file at one key. Each event is written once, to the one log its scope names:
:func:`audited` for the platform's doors (every MCP tool in ``tools/``, plus the script-invoked
doors demoted from them, keeping ``@audited`` without registering), which name the argument
carrying the dataset or project location with ``@audited(scope_arg=...)``, and :func:`record_event`
/ :func:`record_event_or_raise` for code that is neither.

An entry's ``scope`` field is stamped by :func:`_stamp_scope`, the one implementation
:func:`audited`, :func:`record_event` and :func:`record_event_or_raise` all resolve a caller's
scope through. When present, ``scope`` names the resolved root the entry was filed under, which
can be a dataset's, a project's, or the platform's own (a writer that resolved and passed the
platform root stamps it; presence never means non-platform); its absence means the writer took
the platform default, never that the line is non-platform. Every line carries no
``schema_version`` (the frozen version 1: absence is the store's own lazy default, held to it by
``frozen-formats.json``). Three disclosures for a moved log: a stamped absolute scope travels in a
shared or imported archive exactly as the log body's other absolute paths already do, unredacted;
``scope`` records a write-time fact, never a location claim, so a relocated import's lines still
name the exporting machine's own root and nothing reconciles them against where the archive now
sits; and a project archive is provenance-preserving, not path-sanitized, by the same standing
choice.

An append the decorator cannot make is a refusal, not a warning, because the append runs after
the tool body: see :class:`MutationCommittedWithoutAuditLine`.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tcip_store import LOG_JSON, Key, StoreDescriptor, append, register_store
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp import agent_identity
from tcip_mcp.project_paths import resolve_state

logger = logging.getLogger(__name__)

# Relative default (tests rebind this constant). At write time ``resolve_state`` anchors it to
# ``$TCIP_STATE_ROOT`` when pinned, so processes from different dirs don't fragment the log.
AUDIT_ROOT = Path(".")

_AUDIT_LOG = RootedFileLocator(prefix=(".tcip",), suffix=".jsonl")
"""The append-only log under a root's own ``.tcip/``."""

AUDIT_LOG_STORE = "audit_log"
_AUDIT_PARTS = ("audit",)
register_store(
    StoreDescriptor(
        name=AUDIT_LOG_STORE,
        kind="log",
        key_fields=("document",),
        frozen=True,
        codec=LOG_JSON,
        locator=_AUDIT_LOG,
    )
)

# Fields to redact from logged arguments
_REDACTED_FIELDS = {"api_key", "token", "password", "secret"}


class MutationCommittedWithoutAuditLine(RuntimeError):
    """A tool body ran to completion and the audit entry that follows it did not land.

    Raised rather than warned because of where the append sits: :func:`audited` writes the entry
    after the body returns, so by the time an append can fail the state change is already on
    disk. A caller told only by a log line would read the missing entry as a failed call and
    blind-retry a mutation that already happened, and the trail would then say neither ran. The
    caller is told instead, and told which of the two happened.

    The conservative behavior, warning and continuing with no audit line, is the removal of the
    single ``raise`` of this error in :func:`audited`; the warning beside it stands either way.
    """

    def __init__(self, tool: str, cause: BaseException) -> None:
        super().__init__(
            f"{tool} completed and its audit entry could not be written: {cause}. Whatever the "
            "call changed is committed and unrecorded, so do not retry it blind: repair the "
            "audit log's destination, then reconcile the trail against what the call did."
        )
        self.tool = tool


class AuditEntryNotWritten(RuntimeError):
    """A call outside ``@audited`` recorded a mutation that already committed, and the append for
    it failed.

    :func:`record_event_or_raise`'s sibling to :class:`MutationCommittedWithoutAuditLine`: the
    same "committed and unrecorded, do not blind-retry" shape, for a call site with no tool body
    of its own to have already run. A sibling rather than a subclass, since the two guard different
    things (a decorator's own control flow around a body versus an explicit call with none), not
    one specialization of the other.
    """

    def __init__(self, tool: str, cause: BaseException) -> None:
        super().__init__(
            f"{tool} completed and its audit entry could not be written: {cause}. Whatever the "
            "call changed is committed and unrecorded, so do not retry it blind: repair the "
            "audit log's destination, then reconcile the trail against what the call did."
        )
        self.tool = tool


def platform_audit_scope() -> Path:
    """The root a platform event is recorded under, resolved at write time."""
    return resolve_state(AUDIT_ROOT)


def audit_log_key(scope: str | Path | None = None) -> Key:
    """The audit log one event belongs in.

    ``scope`` is the root the event's subject hangs off: a dataset root when the event changed a
    record that travels with the data, a project root when the event is the project's own outward
    action, the platform root (the default) for everything else. One store under three kinds of
    root; writers address it through :func:`_stamp_scope`, readers through this function
    directly, and both are one resolution because the stamper calls this.
    """
    root = Path(scope) if scope is not None else platform_audit_scope()
    return Key(AUDIT_LOG_STORE, str(root.resolve()), _AUDIT_PARTS)


def _stamp_scope(entry: dict[str, Any], scope: str | Path | None) -> Key:
    """Stamp ``entry`` from the same Key :func:`audit_log_key` builds, and return that Key: the
    stamped scope is the key's own root, so the root a line names and the root its Key addresses
    are one resolved value by construction, never two separately-resolved answers.

    ``entry["scope"]`` is stamped only when the caller passed a scope; the value may equal the
    platform root (a project door after adoption, or a writer that resolved the platform root
    itself and passed it), so the field means the writer named its root explicitly, and its
    absence means the writer took the platform default, never that the line is non-platform.
    """
    key = audit_log_key(scope)
    if scope is not None:
        entry["scope"] = key.root
    return key


def _redact(args: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive fields from tool arguments."""
    return {
        k: "***REDACTED***" if k in _REDACTED_FIELDS else v
        for k, v in args.items()
    }


def _entry(
    tool: str,
    arguments: dict[str, Any] | None,
    status: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The one shape every audit entry starts from: the clock, the tool, its redacted arguments,
    a caller's extra facts, and the agent identity this process established at its MCP handshake,
    if it has one.

    Shared by the decorator and both plain emitters, so the stamp an entry carries is decided in
    one place; an emitter building its own dict would be the drift this module exists to prevent.
    The identity keys are reserved: a caller's ``extra`` cannot set one, whether to override the
    handshake's value or to supply one the handshake left absent, so an entry's identity is only
    ever what the handshake established.
    """
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "arguments": _redact(arguments) if arguments else {},
    }
    if status is not None:
        entry["status"] = status
    entry.update({k: v for k, v in (extra or {}).items() if k not in agent_identity.RECORD_FIELDS})
    entry.update(agent_identity.audit_fields())
    return entry


def _write_entry(entry: dict[str, Any], scope: str | Path | None = None) -> None:
    """Append one audit entry to the log ``scope`` names (lock-guarded + fsync'd), never raising.

    What :func:`record_event` writes through, for the emitters that are not MCP tools. Its callers
    are built on never raising here: the training envelope brackets a body already running in a
    background thread, and each GUI route states that recording never fails the request. Standing
    residual: some of those callers record after their own mutation, the position :func:`audited`
    refuses from, so a dropped line there is a provenance gap nobody is told about. The refusal
    covers the decorator; extending it to these emitters means changing what each caller does with
    the failure, not this function.
    """
    try:
        append(_stamp_scope(entry, scope), entry)
    except Exception:
        # A dropped audit line is a real provenance gap, surface it, don't bury it at debug.
        logger.warning("Failed to write audit entry", exc_info=True)


def record_event(
    tool: str,
    arguments: dict[str, Any] | None = None,
    *,
    status: str = "ok",
    scope: str | Path | None = None,
    **extra: Any,
) -> None:
    """Emit one audit line for code that isn't an ``@audited`` MCP tool.

    The one writer of an audit entry: the training envelope brackets the training body (which
    runs in a background thread, outside any ``@audited`` MCP call) with open/close events,
    and the GUI routes record the mutations a browser request makes, so a consumer reads one
    stream rather than several files written by several spellings. ``scope`` names the root
    whose log the entry belongs in (see :func:`audit_log_key`). Best-effort, never raises.
    """
    _write_entry(_entry(tool, arguments, status, extra), scope)


def record_event_or_raise(
    tool: str,
    arguments: dict[str, Any] | None = None,
    *,
    status: str = "ok",
    scope: str | Path | None = None,
    **extra: Any,
) -> None:
    """Emit one audit line for a confirmation write that must not land silently unrecorded.

    Identical shape to :func:`record_event`, for a caller recording a mutation it already made,
    with no tool body of its own for ``@audited`` to bracket. Unlike :func:`record_event`, a
    failed append is not swallowed: it is raised as :class:`AuditEntryNotWritten`, naming the
    mutation that already committed and is now unrecorded, so the caller cannot blind-retry it.
    :func:`record_event`'s own callers are unaffected; this is a new sibling, not a change to it.
    """
    entry = _entry(tool, arguments, status, extra)
    try:
        append(_stamp_scope(entry, scope), entry)
    except Exception as exc:
        logger.warning("Failed to write the audit entry for %s", tool, exc_info=True)
        raise AuditEntryNotWritten(tool, exc) from exc


def dataset_scope_of(value: Any) -> Path | None:
    """The dataset root ``value`` names, or ``None`` when it does not name one.

    ``value`` is whatever a tool's declared scope argument holds: a path inside the dataset
    (an annotations or predictions directory, an image), or the dataset root itself. A path
    under a canonical dataset segment resolves through :func:`dataset_layout.dataset_root_of`,
    the one resolver for that shape. A path that is not under one counts as a root only when it
    is a directory that actually carries dataset or project state (its own ``.tcip/`` or a class
    registry); anything else is not evidence of a dataset and yields ``None``, since a guessed
    root would file the event against a log nobody can trace it back to.
    """
    from tcip_mcp.dataset_layout import CLASSES_FILENAME, dataset_root_of

    if not isinstance(value, (str, Path)) or not str(value):
        return None
    root = dataset_root_of(value)
    if root is None:
        candidate = Path(value)
        if not candidate.is_dir() or not (
            (candidate / ".tcip").is_dir() or (candidate / CLASSES_FILENAME).is_file()
        ):
            return None
        root = candidate
    # Resolved, so the scope an entry names is the same root its key was built from.
    return root.resolve()


def audited(
    fn: Callable | None = None,
    *,
    scope_arg: str | None = None,
    scope_via: Callable[[Any], Any] | None = None,
) -> Callable:
    """Decorator that logs MCP tool calls to the audit log their scope names.

    Bare (``@audited``), a call is a platform event and is recorded in the platform's log.
    ``@audited(scope_arg=...)`` declares which of the tool's own arguments carries the dataset
    or project location the call mutates a record of: that argument's value is resolved at call
    time (:func:`dataset_scope_of` for a dataset argument; a project argument resolves as the
    root it names, which after adoption can equal the platform root), and the entry goes to that
    root's log carrying a ``scope`` field naming it. An argument that is ``None``, absent, or
    resolves to no root leaves the call a platform event. Exactly one log receives each entry.

    ``scope_via`` is for a tool whose body canonicalizes that argument before writing through it,
    such as a relative output path anchored to the platform state root rather than the process
    cwd. Pass the resolver the body itself calls, so the scope is resolved along the identical
    path the write takes; a second implementation of that anchoring would file entries at a
    location the tool never wrote to.

    Three outcomes an entry can fail on, and what each does, all decided by the fact that the
    entry is written after the body:

    - The body returned and the append failed: :class:`MutationCommittedWithoutAuditLine`, so a
      caller cannot mistake a committed mutation for one it may retry.
    - The body raised: the body's exception is what the caller gets. The failed audit-of-failure
      is logged and never allowed to mask it.
    - A declared scope argument was given and resolving it raised: the call refuses, since
      rerouting a declared dataset event to the platform log would file it where nobody tracing
      that dataset looks. A resolution that cleanly answers "no dataset" is not a failure, and
      leaves the call a platform event as documented above.

    Binds positional args to their parameter names so a caller that invokes the
    tool positionally, e.g. the training relaunch route, which calls ``launch_training(config)``
    rather than by keyword, is recorded with the same fidelity as a keyword
    call, instead of writing an empty ``arguments`` dict. Binding failures never abort the call this
    decorator only observes; they fall back to the kwargs-only record, and to the platform log,
    since the scope argument's value is not recoverable from a failed binding.
    """
    def decorate(func: Callable) -> Callable:
        sig = inspect.signature(func)
        if scope_arg is not None and scope_arg not in sig.parameters:
            raise ValueError(
                f"@audited(scope_arg={scope_arg!r}) on {func.__name__} names no parameter of it; "
                f"it takes {tuple(sig.parameters)}"
            )
        if scope_via is not None and scope_arg is None:
            raise ValueError(
                f"@audited(scope_via=...) on {func.__name__} has no scope_arg to apply it to"
            )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tool_name = func.__name__
            t0 = time.monotonic()
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                logged_args: dict[str, Any] = dict(bound.arguments)
            except TypeError:
                logged_args = dict(kwargs)
            entry = _entry(tool_name, logged_args)

            def record() -> None:
                """Resolve the scope, stamp the duration, and append. Raises what it cannot do."""
                # Resolved after the body, so a tool that creates the dataset it names is
                # recorded in that dataset's own log rather than the platform's.
                scope = None
                raw = logged_args.get(scope_arg) if scope_arg else None
                if raw is not None:
                    scope = dataset_scope_of(scope_via(raw) if scope_via else raw)
                entry["duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
                append(_stamp_scope(entry, scope), entry)

            # One entry per call by construction: the two paths are exclusive, and neither
            # writer sits inside a handler that could run the other.
            try:
                result = func(*args, **kwargs)
            except Exception as body_exc:
                entry["status"] = "exception"
                entry["error"] = str(body_exc)
                try:
                    record()
                except Exception:
                    # The body's exception is the caller's answer; this one never displaces it.
                    logger.warning("Failed to audit the failed %s call", tool_name, exc_info=True)
                raise
            entry["status"] = "error" if isinstance(result, dict) and "error" in result else "ok"
            try:
                record()
            except Exception as exc:
                logger.warning("Failed to write the audit entry for %s", tool_name, exc_info=True)
                raise MutationCommittedWithoutAuditLine(tool_name, exc) from exc
            return result

        return wrapper

    return decorate(fn) if fn is not None else decorate
