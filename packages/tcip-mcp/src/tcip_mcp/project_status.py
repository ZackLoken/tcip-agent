"""Per-project status pointer: a small, persisted summary of recent activity.

Persists status/history only, never a "next step" or plan: a retrospective is pointed at by its
project_id and timestamp, never cached as text. A corrupted file is reported as corrupt, not
treated as "no history yet."
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
    """The project's status pointer, written compare-and-swap."""
    return Key(PROJECT_STATUS_STORE, str(project_path), _STATUS_PARTS)


def project_status_path(project_path: str | Path) -> Path:
    """``<project_path>/.tcip/state/project_status.json``."""
    root = Path(project_path)
    return root.joinpath(*_STATUS_DOC.relative_path(str(root), _STATUS_PARTS).parts)


def read_project_status(project_path: str | Path) -> dict[str, Any]:
    """The project's status summary, or ``{}`` if none exists yet.

    A missing file returns ``{}``; a file that exists but fails to decode, or decodes to something
    other than a dict, returns ``{"_corrupt": True}``; a file at a schema_version this reader does
    not accept returns ``{"_version_refused": True}``.
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
    to.

    ``mutate`` reads and mutates the current dict in place, entirely inside the transaction that
    holds this key, so an increment is serialized with its read.
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
    """Call after a ``report_friction`` write: bump both since-last-X counters."""
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
    """Call after a ``write_retrospective`` write: reset the report counter, bump the
    distillation-retrospective counter, and point at the retrospective by its project_id.
    """
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
    """Call after a distillation pass (``record_distillation_pass`` MCP tool): reset both
    distillation counters. Records only that a pass happened, not what came of it.
    """
    now = datetime.now(timezone.utc).isoformat()

    def mutate(data: dict[str, Any]) -> None:
        data["reports_since_last_distillation"] = 0
        data["retrospectives_since_last_distillation"] = 0
        data["last_distillation_at"] = now

    _update(project_path, mutate)
