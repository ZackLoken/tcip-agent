"""Audit logging decorator for the platform's mutating doors.

Every call of a mutating door is logged with timestamp, tool name, arguments, status, and duration,
into an append-only store scoped to a dataset, a project, or the platform (:func:`audit_log_key`).
``status`` is ``ok`` when the body returned and ``exception`` when it raised. :func:`audited`
decorates a door; :func:`record_event` / :func:`record_event_or_raise` record for code that is not
one. An entry's ``scope`` field names the resolved root the entry was filed under when the writer
passed one (:func:`_stamp_scope`).

An append the decorator cannot make is a refusal, not a warning, because the append runs after the
tool body: see :class:`MutationCommittedWithoutAuditLine`.
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
    """A tool body ran to completion and the audit entry that follows it did not land."""

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

    ``arguments`` are the facts the unwritten line would have carried, so a caller can name what
    committed without reading it back.
    """

    def __init__(self, tool: str, cause: BaseException, *,
                 arguments: dict[str, Any] | None = None) -> None:
        super().__init__(
            f"{tool} completed and its audit entry could not be written: {cause}. Whatever the "
            "call changed is committed and unrecorded, so do not retry it blind: repair the "
            "audit log's destination, then reconcile the trail against what the call did."
        )
        self.tool = tool
        self.arguments = arguments or {}


def platform_audit_scope() -> Path:
    """The root a platform event is recorded under, resolved at write time."""
    return resolve_state(AUDIT_ROOT)


def audit_log_key(scope: str | Path | None = None) -> Key:
    """The audit log one event belongs in.

    ``scope`` is the root the event's subject hangs off: a dataset root when the event changed a
    record that travels with the data, a project root when the event is the project's own outward
    action or another project's door files a line about it there, the platform root (the default)
    for everything else.
    """
    root = Path(scope) if scope is not None else platform_audit_scope()
    return Key(AUDIT_LOG_STORE, str(root.resolve()), _AUDIT_PARTS)


def _stamp_scope(entry: dict[str, Any], scope: str | Path | None) -> Key:
    """Stamp ``entry`` from the same Key :func:`audit_log_key` builds, and return that Key: the
    stamped scope is the key's own root.

    ``entry["scope"]`` is stamped only when the caller passed a scope; the value may equal the
    platform root, so its absence means the writer took the platform default, never that the line
    is non-platform.
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
    """The shape every audit entry starts from: the clock, the tool, its redacted arguments, a
    caller's extra facts, and the agent identity this process established at its MCP handshake, if
    it has one.

    The identity keys are reserved: a caller's ``extra`` cannot set one.
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
    """Append one audit entry to the log ``scope`` names (lock-guarded + fsync'd), never raising."""
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
    """Emit one best-effort audit line for a caller that is not an ``@audited`` door. ``scope``
    names the root whose log the entry belongs in (see :func:`audit_log_key`). Never raises.
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
    with no tool body of its own for ``@audited`` to bracket. A failed append is raised as
    :class:`AuditEntryNotWritten`, naming the mutation that already committed and is now
    unrecorded.
    """
    entry = _entry(tool, arguments, status, extra)
    try:
        append(_stamp_scope(entry, scope), entry)
    except Exception as exc:
        logger.warning("Failed to write the audit entry for %s", tool, exc_info=True)
        raise AuditEntryNotWritten(tool, exc, arguments=arguments) from exc


def dataset_scope_of(value: Any) -> Path | None:
    """The dataset root ``value`` names, or ``None`` when it does not name one.

    ``value`` is whatever a tool's declared scope argument holds: a path inside the dataset (an
    annotations or predictions directory, an image), or the dataset root itself. A path under a
    canonical dataset segment resolves through :func:`dataset_layout.dataset_root_of`. A path that
    is not under one counts as a root only when it is a directory that actually carries dataset or
    project state (its own ``.tcip/`` or a subject registry); anything else yields ``None``.
    """
    from tcip_mcp.dataset_layout import SUBJECTS_FILENAME, dataset_root_of

    if not isinstance(value, (str, Path)) or not str(value):
        return None
    root = dataset_root_of(value)
    if root is None:
        candidate = Path(value)
        if not candidate.is_dir() or not (
            (candidate / ".tcip").is_dir() or (candidate / SUBJECTS_FILENAME).is_file()
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
    """Decorator that logs a mutating door's calls to the audit log their scope names.

    Bare (``@audited``), a call is a platform event and is recorded in the platform's log.
    ``@audited(scope_arg=...)`` declares which of the tool's own arguments carries the dataset or
    project location the call mutates a record of: that argument's value is resolved at call time
    (:func:`dataset_scope_of` for a dataset argument; a project argument resolves as the root it
    names), and the entry goes to that root's log carrying a ``scope`` field naming it. An argument
    that is ``None``, absent, or resolves to no root leaves the call a platform event. Exactly one
    log receives each entry.

    ``scope_via`` is the resolver the body itself calls to canonicalize that argument before
    writing through it, so the scope is resolved along the identical path the write takes.

    A body that returns leaves its ``ok`` line, except a body returning a dict whose ``"error"`` is
    set, which leaves no line; a body that raises leaves its ``exception`` line.

    Outcomes when the entry cannot be written:

    - The body returned and the append failed: :class:`MutationCommittedWithoutAuditLine`.
    - The body raised: the body's exception is what the caller gets; the failed audit-of-failure is
      logged.
    - A declared scope argument was given and resolving it raised: the call refuses. A resolution
      that cleanly answers "no dataset" leaves the call a platform event.

    Binds positional args to their parameter names, so a positional call is recorded like a keyword
    one. Binding failures never abort the call; they fall back to the kwargs-only record, and to
    the platform log.
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
            if isinstance(result, dict) and result.get("error") is not None:
                return result  # a refusal returned as the error dict every tool returns: no act
            entry["status"] = "ok"
            try:
                record()
            except Exception as exc:
                logger.warning("Failed to write the audit entry for %s", tool_name, exc_info=True)
                raise MutationCommittedWithoutAuditLine(tool_name, exc) from exc
            return result

        return wrapper

    return decorate(fn) if fn is not None else decorate
