"""Audit logging decorator for MCP tools.

Every tool call is logged to .tcip/audit.jsonl with timestamp, tool name,
arguments, result status, and duration. Append-only JSONL format.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tcip_mcp.project_paths import resolve_state

logger = logging.getLogger(__name__)

# Relative default (tests rebind this constant). At write time ``resolve_state`` anchors it to
# ``$TCIP_PROJECT_ROOT`` when pinned, so processes from different dirs don't fragment the log.
AUDIT_PATH = Path(".tcip/audit.jsonl")

# Fields to redact from logged arguments
_REDACTED_FIELDS = {"api_key", "token", "password", "secret"}


def _redact(args: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive fields from tool arguments."""
    return {
        k: "***REDACTED***" if k in _REDACTED_FIELDS else v
        for k, v in args.items()
    }


def _write_entry(entry: dict[str, Any]) -> None:
    """Append a single audit entry to the JSONL file (lock-guarded + fsync'd)."""
    try:
        from tcip_mcp.utils.atomic_io import append_jsonl

        append_jsonl(resolve_state(AUDIT_PATH), entry)
    except Exception:
        # A dropped audit line is a real provenance gap — surface it, don't bury it at debug.
        logger.warning("Failed to write audit entry", exc_info=True)


def record_event(
    tool: str, arguments: dict[str, Any] | None = None, *, status: str = "ok", **extra: Any
) -> None:
    """Emit one standalone audit line for code that isn't an ``@audited`` MCP tool.

    The training envelope uses this to bracket the training body (which runs in a background
    thread, outside any ``@audited`` MCP call) with open/close events on the same append-only
    ``.tcip/audit.jsonl`` sink — closing the hole where the whole ``train()`` body + registration
    left no audit record. Best-effort, never raises.
    """
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "arguments": _redact(arguments) if arguments else {},
        "status": status,
    }
    entry.update(extra)
    _write_entry(entry)


def audited(fn: Callable) -> Callable:
    """Decorator that logs MCP tool calls to .tcip/audit.jsonl.

    Binds positional args to their parameter names (K12 finding 6) so a caller that invokes the
    tool positionally — e.g. the web routes, which call ``launch_training(payload.config,
    payload.output_dir)`` rather than by keyword — is recorded with the same fidelity as a keyword
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
