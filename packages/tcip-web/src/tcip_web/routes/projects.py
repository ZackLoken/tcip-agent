"""The workspace's projects: list, open, remove, rename and correct one's capture invariances.

Lists the projects under the backend's workspace by each one's own record
(:mod:`tcip_mcp.project_record`). Opening a project makes it the backend's open
project; removal, rename and the
invariances' correction are :func:`tcip_mcp.workspace.remove_project`,
:func:`tcip_mcp.project_record.rename_project` and
:func:`tcip_mcp.project_record.replace_invariances`.
"""

from __future__ import annotations

import threading
from pathlib import Path

import anyio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_mcp import dataset_layout, workspace
from tcip_mcp.buckets import buckets_by_date
from tcip_mcp.identity import actor
from tcip_mcp.project_record import (
    CaptureInvariances,
    InvariancesCorrection,
    rename_project,
    replace_invariances,
)
from tcip_web.state import OpenProject, SnapshotProject, store

router = APIRouter(prefix="/api/projects", tags=["projects"])

_WORKSPACE_TURN = threading.Lock()
"""Held by every read of the workspace's project records and by a removal, so no read reopens a
project's store while its folder is being moved."""


class ProjectSummary(BaseModel):
    """One workspace project: its record's fields and what its data holds."""

    # The record's fields, each null beside record_problem when the record will not read.
    id: str | None
    display_name: str | None
    site: str | None
    invariances: CaptureInvariances | None
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


def _summarize(project_dir: Path, record: dict) -> ProjectSummary:
    """Summarize ``project_dir`` with ``record``, its record's fields."""
    from tcip_mcp.pipelines.image_utils import logical_image_count
    from tcip_web.routes.dataset import _subjects_by_date

    st = project_dir.stat()
    image_count = logical_image_count(project_dir)
    dates = dataset_layout.list_dates(project_dir)
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
    """List the workspace's projects, newest first, with the open project's id (``open_id``)."""
    projects: list[ProjectSummary] = []
    with _WORKSPACE_TURN:
        for child, record in workspace.project_records(store.workspace):
            try:
                projects.append(_summarize(child, record))
            except OSError:
                # A project deleted mid-listing must not 500 the whole list.
                continue
    projects.sort(key=lambda p: p.modified, reverse=True)
    opened = store.opened
    return {
        "workspace": str(store.workspace),
        "open_id": None if opened is None else opened.id,
        "projects": [p.model_dump() for p in projects],
    }


def workspace_project(project_id: str) -> tuple[Path, dict]:
    """Return the directory and record fields of the workspace project with ``project_id``.

    The project is the one whose record holds that id
    (:func:`tcip_mcp.workspace.project_by_id`); 404 naming why when none, or two, do.
    """
    try:
        with _WORKSPACE_TURN:
            records = workspace.project_records(store.workspace)
        return workspace.project_by_id(records, project_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


class OpenRequest(BaseModel):
    """A request to open the project ``id`` names, by the person ``user`` names."""

    id: str
    user: str


class OpenAnswer(BaseModel):
    """The project an open made the backend's open one, and its record's display name."""

    project: SnapshotProject
    display_name: str


@router.post("/open")
async def open_project(req: OpenRequest) -> OpenAnswer:
    """Open the project ``req.id`` names and return it (:class:`OpenAnswer`).

    400 for a request naming no one, 404 when the workspace holds no single project with it (a
    project whose record will not read is not found by its id), 409 naming why when its
    persisted GUI state will not read.
    """
    from tcip_store import StoreError

    actor(req.user)
    root, record = workspace_project(req.id)
    project = OpenProject(root, record["id"])
    try:
        await store.open_project(project)
    except (ValueError, StoreError) as exc:
        raise HTTPException(409, str(exc)) from exc
    return OpenAnswer(
        project=SnapshotProject.of(project),
        display_name=record["display_name"],
    )


def _job_conflict(target: Path) -> str | None:
    """Return the refusal naming the first unfinished job that runs for ``target``, else None.

    The jobs looked through are those of this process's two in-memory job registries.
    """
    from tcip_web.jobstore import live
    from tcip_web.routes import annotate, inference

    for job in inference._registry.list(str(target)):
        if live(job):
            return (
                f"inference job {job.job_id!r} is not finished; ask the agent to cancel it "
                "or wait for it to finish"
            )
    for queue_job in annotate._pq_registry.list(str(target)):
        if live(queue_job):
            return (
                f"a review priority-queue scoring pass ({queue_job.job_id!r}) is not finished; "
                "wait for it to finish"
            )
    return None


class RemovalRequest(BaseModel):
    """A request to remove the project ``id`` names, confirmed by its typed display name."""

    id: str
    # The project's display name, typed by the person removing it.
    confirm_name: str
    user: str


@router.post("/remove")
async def remove_project(req: RemovalRequest) -> dict:
    """Archive the project ``req.id`` names into the workspace's ``.removed/`` and move it there.

    The project, when it is the open one, is closed only once the removal is admitted. 404 for
    no such project, 400 for a confirm name that is not its display name, 409 while a job or run
    of it is live, the archive refuses or another process holds the folder past the move's retry
    budget, 400 for a request naming no one. Returns ``{archive_path, moved_to}``.
    """
    person = actor(req.user)
    project, fields = workspace_project(req.id)
    display_name = fields["display_name"]
    if req.confirm_name != display_name:
        raise HTTPException(
            400, f"type the project's name {display_name!r} to remove it"
        )
    conflict = _job_conflict(project)
    if conflict is not None:
        raise HTTPException(409, conflict)

    def release() -> None:
        anyio.from_thread.run(store.close_project, req.id)

    def remove() -> dict:
        with _WORKSPACE_TURN:
            return workspace.remove_project(
                store.workspace, project, actor=person, release=release
            )

    try:
        return await anyio.to_thread.run_sync(remove)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            409,
            f"another process holds the project's folder open, so it was left where it "
            f"is: {exc}. Close whatever has it open (an agent terminal working on it, a "
            "file browser) and remove it again",
        ) from exc


class RenameRequest(BaseModel):
    """A request to give the project ``id`` names a new display name."""

    id: str
    display_name: str
    user: str


@router.post("/rename")
def rename_project_route(req: RenameRequest) -> dict:
    """Change the display name of the project ``req.id`` names; its directory is left as it is.

    404 for no such project, 400 for a display name the record refuses or a request naming no
    one. Returns ``{id, display_name, previous_display_name}``.
    """
    person = actor(req.user)
    project, _ = workspace_project(req.id)
    try:
        return rename_project(project, req.display_name, actor=person)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class InvariancesRequest(BaseModel):
    """A correction of the capture invariances the project ``id`` names records."""

    id: str
    invariances: CaptureInvariances
    user: str


@router.post("/invariances")
def replace_invariances_route(
    req: InvariancesRequest,
) -> InvariancesCorrection:
    """Correct the capture invariances the project ``req.id`` names records.

    404 for no such project, 400 for a request naming no one, and 422 for a block the record's
    type refuses.
    """
    person = actor(req.user)
    project, _ = workspace_project(req.id)
    return replace_invariances(project, req.invariances, actor=person)
