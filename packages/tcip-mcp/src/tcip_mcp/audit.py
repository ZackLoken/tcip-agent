"""Audit logging decorator for MCP tools.

Every tool call is logged with timestamp, tool name, arguments, result status, and duration.
Append-only JSONL format.

The log is scoped: an event whose subject is a record that travels with a dataset is recorded
in that dataset's own log, so the provenance travels with the data, and a platform event is
recorded in the platform's log. Each event is written once, by :func:`record_event`, to the
one log its scope names.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tcip_store import Key, StoreDescriptor, append, json_codec, register_store
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
        codec=json_codec(indent=None),
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


def audited(fn: Callable) -> Callable:
    """Decorator that logs MCP tool calls to the platform's audit log.

    Binds positional args to their parameter names so a caller that invokes the
    tool positionally, e.g. the web routes, which call ``launch_training(payload.config,
    payload.output_dir)`` rather than by keyword, is recorded with the same fidelity as a keyword
    call, instead of writing an empty ``arguments`` dict. Binding failures never abort the call this
    decorator only observes; they fall back to the kwargs-only record.
    """
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tool_name = fn.__name__
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

        try:
            result = fn(*args, **kwargs)
            entry["status"] = "error" if isinstance(result, dict) and "error" in result else "ok"
            entry["duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
            _write_entry(entry)
            return result
        except Exception as exc:
            entry["status"] = "exception"
            entry["error"] = str(exc)
            entry["duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
            _write_entry(entry)
            raise

    return wrapper
