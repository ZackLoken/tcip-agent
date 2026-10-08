"""The workspace: the folder whose child directories are the projects the GUI lists.

A workspace root holds one directory per project, and beside them only the backend's port record
and the last-opened pointer, the id of the project the backend opened last. A removed project
moves into its ``.removed/``. Every function here takes the workspace a process resolved once at
its entry (:func:`workspace_from_environment`).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Optional

import tcip_store
from tcip_store import Key

logger = logging.getLogger(__name__)

REMOVED_DIRNAME = ".removed"
"""The workspace's holding directory for removed projects and their archives. It carries no
``.tcip`` of its own, so it is never listed as a project."""

RENAME_BUDGET_S = 5.0
"""How long :func:`remove_project` retries a move a transient handle (a scanner, an indexer)
denies; a process still holding the project's database open is not transient."""


def workspace_from_environment() -> Path:
    """The workspace root ``TCIP_WORKSPACE`` names (``~`` expanded), created and resolved, its path
    logged. Read once, at a process's entry. Refuses (``ValueError``) a variable that is unset or
    blank."""
    raw = os.environ.get("TCIP_WORKSPACE", "").strip()
    if not raw:
        raise ValueError("TCIP_WORKSPACE is unset: set it to the folder whose child directories "
                         "are the projects this process serves")
    root = Path(raw).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    logger.info("TCIP workspace: %s", root)
    return root


class BoundProject(NamedTuple):
    """The project a process was started for: its directory, the id its record held at start
    (:func:`~tcip_mcp.project_record.existing_project`), and the workspace naming the backend
    that serves it."""

    root: Path
    id: str
    workspace: Path


def is_valid_name(name: str) -> bool:
    """True if ``name`` is a single, safe path segment."""
    seg = (name or "").strip()
    return bool(seg) and not any(c in seg for c in ("/", "\\", ":")) and seg not in (".", "..")


# ── the last-opened pointer ──────────────────────────────────────────────────

LAST_OPENED_STORE = "workspace_last_opened"
_POINTER_PARTS = ("last_opened",)


def last_opened_key(workspace: Path) -> Key:
    """The workspace's last-opened pointer: the id of the project the backend opened last, a JSON
    string replaced whole on every open."""
    return Key(LAST_OPENED_STORE, str(workspace), _POINTER_PARTS)


def read_last_opened(workspace: Path) -> Optional[str]:
    """The id the last-opened pointer holds, as stored, or ``None`` when the backend has opened
    nothing in this workspace yet. A pointer the store cannot read raises the store's own error."""
    return tcip_store.read(last_opened_key(workspace), default=None)


def write_last_opened(workspace: Path, project_id: str) -> None:
    """Point the workspace's last-opened pointer at ``project_id``."""
    tcip_store.replace(last_opened_key(workspace), project_id)


# ── the projects ─────────────────────────────────────────────────────────────


def project_dirs(workspace: Path) -> list[Path]:
    """Every child directory of the workspace carrying a ``.tcip`` directory, sorted by name."""
    return sorted(p for p in workspace.iterdir() if p.is_dir() and (p / ".tcip").is_dir())


def project_records(workspace: Path) -> list[tuple[Path, dict]]:
    """Every workspace project (:func:`project_dirs`) with its record's fields as
    :func:`tcip_mcp.project_record.record_fields` reads them."""
    from tcip_mcp.project_record import record_fields

    return [(project, record_fields(project)) for project in project_dirs(workspace)]


def project_by_id(records: Iterable[tuple[Path, dict]], project_id: str) -> tuple[Path, dict]:
    """The project among ``records`` (:func:`project_records`) whose record holds ``project_id``,
    with its fields. Raises ``LookupError`` naming the id when no project holds it, with every
    project whose record would not read and why; and naming both directories when two hold it (a
    copied project)."""
    found: list[tuple[Path, dict]] = []
    unreadable: list[str] = []
    for project, fields in records:
        if fields["record_problem"] is not None:
            unreadable.append(f"{project}: {fields['record_problem']}")
        elif fields["id"] == project_id:
            found.append((project, fields))
    if not found:
        detail = f"; these projects' records would not read: {'; '.join(unreadable)}" if (
            unreadable) else ""
        raise LookupError(f"no project in the workspace has id {project_id!r}{detail}")
    if len(found) > 1:
        raise LookupError(f"projects {', '.join(str(p) for p, _ in found)} all have id "
                          f"{project_id!r}; one is a copy, so neither is opened by it")
    return found[0]


def remove_project(workspace: Path, project: Path, *, actor: str,
                   release: Callable[[], None]) -> dict:
    """Archive a workspace project, models included, into the workspace's ``.removed/``, then move
    its directory there, and record one ``project_removed`` line by ``actor`` in the moved
    project's own log.

    Refuses (``ValueError``) while a run or sweep of the project is live, and when the archive
    refuses; nothing is written and ``release`` is not called then. Once admitted and archived,
    ``release`` lets the caller let go of the project, then every store connection this process
    holds on the project's tree is released before the move. A move the filesystem denies past
    :data:`RENAME_BUDGET_S` removes the archive just written and raises the ``OSError``, leaving the
    project where it was. Returns ``{"archive_path", "moved_to"}``.
    """
    from tcip_store.file_backend import retry_while_denied

    from tcip_mcp.audit import record_event_or_raise
    from tcip_mcp.experiments import live_run_conflict
    from tcip_mcp.tools.project_tools import write_archive

    conflict = live_run_conflict(project)
    if conflict is not None:
        raise ValueError(conflict)
    removed = workspace / REMOVED_DIRNAME
    removed.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = removed / f"{project.name}-{stamp}.zip"
    moved_to = removed / f"{project.name}-{stamp}"
    result = write_archive(project, output_path=str(archive_path), include_models=True)
    if "error" in result:
        raise ValueError(result["error"])
    release()
    tcip_store.release_root(project)
    try:
        retry_while_denied(lambda: os.rename(project, moved_to), RENAME_BUDGET_S)
    except OSError:
        archive_path.unlink(missing_ok=True)
        raise
    record_event_or_raise("project_removed", {"archive_path": str(archive_path)}, actor=actor,
                          scope=moved_to)
    return {"archive_path": str(archive_path), "moved_to": str(moved_to)}
