"""Per-project status pointer: a small, persisted summary of recent activity.

A locator module in the same spirit as :mod:`tcip_mcp.dataset_layout`: pure path/read/write
helpers, no business logic elsewhere duplicates. Read back by ``inspect_project`` and
``set_active_project`` so one call gives the picture that today takes 2-3 separate reads
(``inspect_project``'s live counts, plus ``load_project_memory`` once per kind).

Deliberately persists status/history only, never a "next step" or plan. A retrospective's
``would_do_differently``/``knowledge_for_future`` fields are forward-looking; this module never
caches their text, only a pointer (path + timestamp) to the retrospective that holds them, so
reading the actual content (with its caveats intact) is always one explicit
``load_project_memory(kind='retrospectives')`` call away, never silently resurfaced.

Unlike ``.tcip/reports/``/``.tcip/retrospectives/`` (expected to be pruned eventually), this file is
meant to be a permanent fixture a project operates against for its whole life, so a corrupted
file is reported as corrupt, not silently treated as "no history yet."
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

STATUS_FILENAME = "project_status.json"


def project_status_path(project_path: str | Path) -> Path:
    """``<project_path>/.tcip/state/project_status.json``."""
    return Path(project_path, ".tcip", "state", STATUS_FILENAME)


def read_project_status(project_path: str | Path) -> dict[str, Any]:
    """The project's status summary, or ``{}`` if none exists yet.

    Distinguishes absence from corruption: a missing file is genuinely "no history yet" and
    returns ``{}``; a file that exists but fails to parse, or parses to something other than a
    dict (mirrors :func:`tcip_mcp.dataset_layout.normalize_status_store`'s shape guard), returns
    ``{"_corrupt": True}`` instead, so callers surface that honestly rather than silently reading a
    permanent-fixture store as if it were a clean slate.
    """
    import json

    path = project_status_path(project_path)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # ValueError covers json.JSONDecodeError and UnicodeDecodeError (both its subclasses):
        # a non-UTF-8 file must not slip past this into an unhandled exception, which would break
        # the "status-file trouble must never break the write it's attached to" invariant _update
        # exists for.
        return {"_corrupt": True}
    if not isinstance(raw, dict):
        return {"_corrupt": True}
    return raw


def _update(project_path: str | Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    """Best-effort locked read-modify-write: never raises, never blocks the caller it's attached
    to. A status-file write failing must not fail the report/retrospective/distillation-pass write
    it's recording, same shape as ``audit.py``'s own best-effort append.

    ``mutate`` reads and mutates the current dict *in place*, entirely inside the
    ``file_transaction`` lock: an increment (``data[k] = data.get(k, 0) + 1``) computed from a
    read taken outside the lock, then passed in as an absolute value, would lose updates under
    concurrent callers (two callers reading the same pre-increment value, each writing the same
    post-increment one). Putting the read-and-decide step inside ``mutate`` is what makes
    ``file_transaction`` actually serialize the increment, not just the write.
    """
    from tcip_mcp.utils.atomic_io import atomic_write_json, file_transaction, read_json

    path = project_status_path(project_path)
    try:
        with file_transaction(path):
            data = read_json(path, default={})
            if not isinstance(data, dict):
                data = {}
            data.pop("_corrupt", None)
            mutate(data)
            atomic_write_json(path, data)
    except Exception:
        pass


def record_report(project_path: str | Path) -> None:
    """Call after a ``claude_reports`` write: bump both since-last-X counters."""
    now = datetime.now(timezone.utc).isoformat()

    def mutate(data: dict[str, Any]) -> None:
        data["last_activity"] = now
        data["reports_since_last_retrospective"] = (
            int(data.get("reports_since_last_retrospective") or 0) + 1
        )
        data["reports_since_last_distillation"] = (
            int(data.get("reports_since_last_distillation") or 0) + 1
        )

    _update(project_path, mutate)


def record_retrospective(project_path: str | Path, project_id: str, retro_path: str | Path) -> None:
    """Call after a ``project_retrospective`` write: reset the report counter, bump the
    distillation-retrospective counter, and point at the retrospective file (no cached text)."""
    now = datetime.now(timezone.utc).isoformat()

    def mutate(data: dict[str, Any]) -> None:
        data["last_activity"] = now
        data["reports_since_last_retrospective"] = 0
        data["retrospectives_since_last_distillation"] = (
            int(data.get("retrospectives_since_last_distillation") or 0) + 1
        )
        data["last_retrospective"] = {
            "project_id": project_id,
            "path": str(retro_path),
            "modified_at": now,
        }

    _update(project_path, mutate)


def record_distillation(project_path: str | Path) -> None:
    """Call after an owner-reviewed distillation pass (``record_distillation_pass`` MCP tool):
    reset both distillation counters. Bookkeeping only: records that a pass happened, not what
    came of it; promoting anything from a worksheet to a skill/CLAUDE.md stays the owner's own,
    separate, explicit act."""
    now = datetime.now(timezone.utc).isoformat()

    def mutate(data: dict[str, Any]) -> None:
        data["reports_since_last_distillation"] = 0
        data["retrospectives_since_last_distillation"] = 0
        data["last_distillation_at"] = now

    _update(project_path, mutate)
