"""The workspace's projects: listing them, opening one, removing one and renaming one.

Lists the projects under the backend's workspace by each one's own record
(:mod:`tcip_mcp.project_record`). Opening a project makes it the backend's open
project and points the workspace's last-opened pointer at its id; removal and rename are
:func:`tcip_mcp.workspace.remove_project` and :func:`tcip_mcp.project_record.rename_project`.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import anyio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_mcp import dataset_layout, workspace
from tcip_mcp.buckets import buckets_by_date
from tcip_mcp.identity import actor
from tcip_mcp.project_record import record_fields, rename_project
from tcip_web.state import store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["projects"])

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".bmp"}


class ProjectSummary(BaseModel):
    # The record's three fields, each null beside record_problem when the record will not read.
    id: str | None
    display_name: str | None
    site: str | None
    record_problem: str | None
    path: str
    created: float
    modified: float
    dates: list[str]
    subjects: list[str]
    # Per-date availability: which subjects have labels / which buckets are published on each
    # date (by name), so the pickers never offer a date with nothing there.
    subjects_by_date: dict[str, list[str]]
    buckets_by_date: dict[str, list[str]]
    image_count: int
    # The first date's labels that would not read, naming the documents; the project still lists.
    label_problem: str | None


def _summarize(project_dir: Path) -> ProjectSummary:
    from tcip_web.routes.dataset import _subjects_by_date

    st = project_dir.stat()
    images_dir = dataset_layout.image_root(project_dir)
    image_count = 0
    if images_dir.is_dir():
        image_count = sum(
            1 for f in images_dir.rglob("*") if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        )
    dates = dataset_layout.list_dates(project_dir)
    record = record_fields(project_dir)
    subjects_by_date, label_problem = _subjects_by_date(project_dir, dates)
    return ProjectSummary(
        **record,
        path=str(project_dir),
        created=st.st_ctime,
        modified=st.st_mtime,
        dates=dates,
        subjects=dataset_layout.list_subjects(project_dir),
        subjects_by_date=subjects_by_date,
        buckets_by_date=buckets_by_date(project_dir, dates),
        image_count=image_count,
        label_problem=label_problem,
    )


@router.get("")
def list_projects() -> dict:
    """The workspace's projects, newest first, with the open project's id (``open_id``).
    ``last_opened_problem`` names a last-opened pointer whose project is no longer in the
    workspace while nothing is open, and is null otherwise."""
    projects: list[ProjectSummary] = []
    for child in workspace.project_dirs(store.workspace):
        try:
            projects.append(_summarize(child))
        except OSError:
            # A project deleted mid-listing must not 500 the whole list.
            continue
    projects.sort(key=lambda p: p.modified, reverse=True)
    problem = None
    pointer = workspace.read_last_opened(store.workspace)
    if store.project_id is None and pointer is not None:
        try:
            workspace.project_by_id(store.workspace, pointer)
        except LookupError as exc:
            problem = str(exc)
    return {
        "workspace": str(store.workspace),
        "open_id": store.project_id,
        "last_opened_problem": problem,
        "projects": [p.model_dump() for p in projects],
    }


async def open_project_by_id(project_id: str) -> Path:
    """Open the workspace project whose record holds ``project_id`` and point the last-opened
    pointer at it. Raises ``LookupError`` naming the id when no project, or more than one, holds
    it."""
    project = workspace.project_by_id(store.workspace, project_id)
    await store.open_project(project)
    workspace.write_last_opened(store.workspace, project_id)
    return project


async def open_last_opened() -> None:
    """Open the project the workspace's last-opened pointer names; with no pointer, open nothing.
    A pointer naming no project in the workspace leaves nothing open and is logged; the project
    list names it (``last_opened_problem``)."""
    project_id = workspace.read_last_opened(store.workspace)
    if project_id is None:
        return
    try:
        await open_project_by_id(project_id)
    except LookupError as exc:
        logger.warning("Opening no project at start: %s", exc)


class OpenRequest(BaseModel):
    id: str
    user: str


@router.post("/open")
async def open_project(req: OpenRequest) -> dict:
    """Open the project ``req.id`` names; 400 for a request naming no one, 404 when the workspace
    holds no single project with it, 409 naming why when its record or its persisted GUI state
    will not read. Returns ``{id, display_name, path}``."""
    from tcip_store import StoreError

    actor(req.user)
    try:
        project = await open_project_by_id(req.id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, StoreError) as exc:
        raise HTTPException(409, str(exc)) from exc
    record = record_fields(project)
    return {"id": record["id"], "display_name": record["display_name"], "path": str(project)}


def _job_conflict(target: Path) -> str | None:
    """The first unfinished job of this process's two in-memory job registries that runs for
    ``target``."""
    from tcip_mcp.experiments import TERMINAL_STATES

    from tcip_web.routes import annotate, inference

    for job in inference._registry.list():
        if job.status not in TERMINAL_STATES and job.project == str(target):
            return (f"inference job {job.job_id!r} is not finished; ask the agent to cancel it "
                    "or wait for it to finish")
    for job in annotate._pq_registry.list():
        if job.status not in TERMINAL_STATES and job.project == str(target):
            return (f"a review priority-queue scoring pass ({job.job_id!r}) is not finished; "
                    "wait for it to finish")
    return None


class RemovalRequest(BaseModel):
    id: str
    # The project's display name, typed by the person removing it.
    confirm_name: str
    user: str


@router.post("/remove")
async def remove_project(req: RemovalRequest) -> dict:
    """Archive the project ``req.id`` names into the workspace's ``.removed/`` and move it there,
    closing it, when it is the open project, only once the removal is admitted. 404 for no such
    project, 400 for a confirm name that is not its display name, 409 while a job or run of it is
    live or the archive refuses, 400 for a request naming no one. Returns ``{archive_path,
    moved_to}``."""
    person = actor(req.user)
    try:
        project = workspace.project_by_id(store.workspace, req.id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    display_name = record_fields(project)["display_name"]
    if req.confirm_name != display_name:
        raise HTTPException(400, f"type the project's name {display_name!r} to remove it")
    conflict = _job_conflict(project)
    if conflict is not None:
        raise HTTPException(409, conflict)

    def release() -> None:
        if store.project_id == req.id:
            anyio.from_thread.run(store.close_project)

    try:
        return await anyio.to_thread.run_sync(functools.partial(
            workspace.remove_project, store.workspace, project, actor=person, release=release))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


class RenameRequest(BaseModel):
    id: str
    display_name: str
    user: str


@router.post("/rename")
def rename_project_route(req: RenameRequest) -> dict:
    """Change the display name of the project ``req.id`` names; its directory is left as it is.
    404 for no such project, 400 for a display name the record refuses or a request naming no
    one. Returns ``{id, display_name, previous_display_name}``."""
    person = actor(req.user)
    try:
        project = workspace.project_by_id(store.workspace, req.id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    try:
        return rename_project(project, req.display_name, actor=person)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
