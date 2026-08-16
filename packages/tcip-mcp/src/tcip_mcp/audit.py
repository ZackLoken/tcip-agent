"""Audit logging decorator for MCP tools.

Every tool call is logged with timestamp, tool name, arguments, result status, and duration.
Append-only JSONL format.

The log is scoped: an event whose subject is a record that travels with a dataset is recorded
in that dataset's own log, so the provenance travels with the data, and a platform event is
recorded in the platform's log. Each event is written once, to the one log its scope names:
:func:`record_event` for code that is not an MCP tool, and :func:`audited` for the tools, which
name the argument carrying the dataset location with ``@audited(scope_arg=...)``.
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

from tcip_mcp.project_paths import resolve_state

logger = logging.getLogger(__name__)

# Relative default (tests rebind this constant). At write time ``resolve_state`` anchors it to
# ``$TCIP_PROJECT_ROOT`` when pinned, so processes from different dirs don't fragment the log.
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
        codec=LOG_JSON,
        locator=_AUDIT_LOG,
    )
)

# Fields to redact from logged arguments
_REDACTED_FIELDS = {"api_key", "token", "password", "secret"}


def platform_audit_scope() -> Path:
    """The root a platform event is recorded under, resolved at write time."""
    return resolve_state(AUDIT_ROOT)


def audit_log_key(scope: str | Path | None = None) -> Key:
    """The audit log one event belongs in.

    ``scope`` is the root the event's subject hangs off: a dataset root when the event
    changed a record that travels with the data, so an audit trail moves with the dataset it
    describes; the platform root (the default) for everything else.
    """
    root = Path(scope) if scope is not None else platform_audit_scope()
    return Key(AUDIT_LOG_STORE, str(root.resolve()), _AUDIT_PARTS)


def _redact(args: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive fields from tool arguments."""
    return {
        k: "***REDACTED***" if k in _REDACTED_FIELDS else v
        for k, v in args.items()
    }


def _write_entry(entry: dict[str, Any], scope: str | Path | None = None) -> None:
    """Append a single audit entry to the log ``scope`` names (lock-guarded + fsync'd)."""
    try:
        append(audit_log_key(scope), entry)
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
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "arguments": _redact(arguments) if arguments else {},
        "status": status,
    }
    entry.update(extra)
    _write_entry(entry, scope)


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
    ``@audited(scope_arg="dataset_root")`` declares which of the tool's own arguments carries
    the dataset the call mutates a record of: that argument's value is resolved at call time by
    :func:`dataset_scope_of`, and the entry goes to that dataset's log instead, carrying a
    ``scope`` field naming the root. An argument that is ``None``, absent, or resolves to no
    dataset leaves the call a platform event. Exactly one log receives each entry.

    ``scope_via`` is for a tool whose body canonicalizes that argument before writing through it,
    such as a relative output path anchored to the platform state root rather than the process
    cwd. Pass the resolver the body itself calls, so the scope is resolved along the identical
    path the write takes; a second implementation of that anchoring would file entries at a
    location the tool never wrote to.

    Binds positional args to their parameter names so a caller that invokes the
    tool positionally, e.g. the web routes, which call ``launch_training(payload.config,
    payload.output_dir)`` rather than by keyword, is recorded with the same fidelity as a keyword
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
            entry: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": tool_name,
                "arguments": _redact(logged_args) if logged_args else {},
            }

            def record() -> None:
                # Resolved after the body, so a tool that creates the dataset it names is
                # recorded in that dataset's own log rather than the platform's.
                scope = None
                raw = logged_args.get(scope_arg) if scope_arg else None
                if raw is not None:
                    try:
                        scope = dataset_scope_of(scope_via(raw) if scope_via else raw)
                    except Exception:
                        # A location that cannot be resolved leaves the entry a platform event,
                        # rather than failing the call this decorator only observes.
                        logger.warning("Failed to resolve the audit scope for %s", tool_name,
                                       exc_info=True)
                if scope is not None:
                    entry["scope"] = str(scope)
                entry["duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
                _write_entry(entry, scope)

            # One entry per call by construction: the two paths are exclusive, and neither
            # writer sits inside a handler that could run the other.
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                entry["status"] = "exception"
                entry["error"] = str(exc)
                record()
                raise
            entry["status"] = "error" if isinstance(result, dict) and "error" in result else "ok"
            record()
            return result

        return wrapper

    return decorate(fn) if fn is not None else decorate
