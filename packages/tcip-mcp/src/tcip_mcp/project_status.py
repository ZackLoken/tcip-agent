"""Per-project status pointer: a small, persisted summary of recent activity.

A locator module in the same spirit as :mod:`tcip_mcp.dataset_layout`: pure path/read/write
helpers, no business logic elsewhere duplicates. Read back by ``inspect_project`` and
``set_active_project`` so one call gives the picture that today takes 2-3 separate reads
(``inspect_project``'s live counts, plus ``load_project_memory`` once per kind).

Deliberately persists status/history only, never a "next step" or plan. A retrospective's
``would_do_differently``/``knowledge_for_future`` fields are forward-looking; this module never
caches their text, only a pointer (project_id + timestamp) to the retrospective that holds them,
so reading the actual content (with its caveats intact) is always one explicit
``load_project_memory(kind='retrospectives')`` call away, never silently resurfaced. The
project_id, not a path, is what a reader resolves back to the retrospective: a path is
backend-dependent (the database backend keeps no such file) while the project_id resolves
through ``retrospective_key``/``read_retrospective`` under either backend.

Unlike ``.tcip/reports/``/``.tcip/retrospectives/`` (expected to be pruned eventually), this file is
meant to be a permanent fixture a project operates against for its whole life, so a corrupted
file is reported as corrupt, not silently treated as "no history yet."
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tcip_store import (
    RECORD_JSON,
    DecodeError,
    Key,
    SchemaVersionRefused,
    StoreDescriptor,
    read,
    register_store,
    transaction,
)
from tcip_store.file_backend import RootedFileLocator

logger = logging.getLogger(__name__)

_STATUS_DOC = RootedFileLocator(prefix=(".tcip", "state"), suffix=".json")
"""The status pointer, one document per project."""

PROJECT_STATUS_STORE = "project_status"
_STATUS_PARTS = ("project_status",)
register_store(
    StoreDescriptor(
        name=PROJECT_STATUS_STORE,
        kind="record",
        key_fields=("document",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="cas",
        locator=_STATUS_DOC,
    )
)


def project_status_key(project_path: str | Path) -> Key:
    """The project's status pointer.

    ``cas``: every writer here increments a counter it just read, so the read and the write
    have to be one serialized step. :func:`_update` names this key in a transaction and does
    the read-and-decide inside it.
    """
    return Key(PROJECT_STATUS_STORE, str(project_path), _STATUS_PARTS)


def project_status_path(project_path: str | Path) -> Path:
    """``<project_path>/.tcip/state/project_status.json``."""
    root = Path(project_path)
    return root.joinpath(*_STATUS_DOC.relative_path(str(root), _STATUS_PARTS).parts)


def read_project_status(project_path: str | Path) -> dict[str, Any]:
    """The project's status summary, or ``{}`` if none exists yet.

    Distinguishes absence from corruption: a missing file is genuinely "no history yet" and
    returns ``{}``; a file that exists but fails to decode, or decodes to something other than a
    dict (mirrors :func:`tcip_mcp.dataset_layout.normalize_status_store`'s shape guard), returns
    ``{"_corrupt": True}`` instead, so callers surface that honestly rather than silently reading a
    permanent-fixture store as if it were a clean slate. A file at a schema_version this reader
    does not accept is a distinct fact, not corruption, so it returns ``{"_version_refused": True}``
    instead: the bytes are a well-formed document from a newer writer, not garbage.
    """
    try:
        raw = read(project_status_key(project_path), default={})
    except SchemaVersionRefused:
        return {"_version_refused": True}
    except (OSError, DecodeError):
        # An unreadable pointer is reported as corrupt, not as absent: this store is a
        # permanent fixture, so "cannot be read" is never "no history yet".
        return {"_corrupt": True}
    if not isinstance(raw, dict):
        return {"_corrupt": True}
    return raw


def _update(project_path: str | Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    """Best-effort locked read-modify-write: never raises, never blocks the caller it's attached
    to. A status-file write failing must not fail the report/retrospective/distillation-pass write
    it's recording, same shape as ``audit.py``'s own best-effort append.

    ``mutate`` reads and mutates the current dict *in place*, entirely inside the transaction
    that holds this key: an increment (``data[k] = data.get(k, 0) + 1``) computed from a read
    taken outside the lock, then passed in as an absolute value, would lose updates under
    concurrent callers (two callers reading the same pre-increment value, each writing the same
    post-increment one). Putting the read-and-decide step inside ``mutate`` is what makes the
    transaction actually serialize the increment, not just the write.
    """
    key = project_status_key(project_path)
    try:
        with transaction(key) as txn:
            try:
                data = txn.read(key, default={})
            except DecodeError:
                # A corrupt pointer is replaced rather than propagated: the counters it held
                # are unreadable, and refusing here would fail the write being recorded.
                data = {}
            if not isinstance(data, dict):
                data = {}
            data.pop("_corrupt", None)
            mutate(data)
            txn.write(key, data)
    except SchemaVersionRefused:
        # A newer writer's pointer: left untouched rather than overwritten with counters that
        # would erase whatever fields that writer added, unlike the corrupt-bytes case above.
        logger.warning(
            "project status pointer for %s is at a schema_version this reader does not "
            "accept; its counters were left untouched", project_path)
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


def record_retrospective(project_path: str | Path, project_id: str) -> None:
    """Call after a ``project_retrospective`` write: reset the report counter, bump the
    distillation-retrospective counter, and point at the retrospective by its project_id (no
    cached text, no path: a path is backend-dependent and the database backend keeps no file)."""
    now = datetime.now(timezone.utc).isoformat()

    def mutate(data: dict[str, Any]) -> None:
        data["last_activity"] = now
        data["reports_since_last_retrospective"] = 0
        data["retrospectives_since_last_distillation"] = (
            int(data.get("retrospectives_since_last_distillation") or 0) + 1
        )
        data["last_retrospective"] = {
            "project_id": project_id,
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
