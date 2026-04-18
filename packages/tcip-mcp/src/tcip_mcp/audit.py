"""Audit logging decorator for MCP tools.

Every tool call is logged to .tcip/audit.jsonl with timestamp, tool name,
arguments, result status, and duration. Append-only JSONL format.
"""

from __future__ import annotations

import functools
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

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
    """Append a single audit entry to the JSONL file."""
    try:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        logger.debug("Failed to write audit entry", exc_info=True)


def audited(fn: Callable) -> Callable:
    """Decorator that logs MCP tool calls to .tcip/audit.jsonl."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tool_name = fn.__name__
        t0 = time.monotonic()
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
            "arguments": _redact(kwargs) if kwargs else {},
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
