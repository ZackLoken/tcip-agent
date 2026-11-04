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
from typing import Annotated

import anyio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, TypeAdapter

from tcip_mcp import workspace
from tcip_mcp.audit import AuditEntryNotWrittenError
from tcip_mcp.identity import actor
from tcip_mcp.project_record import (
    CaptureInvariances,
    ProjectRecord,
    ReadRecord,
    RecordCorrection,
    UnreadRecord,
    rename_project,
    replace_invariances,
)
from tcip_web.routes.audit_gap import audit_gap_409
from tcip_web.routes.dataset import DatasetTree, tree_of
from tcip_web.state import OpenProject, SnapshotProject, store

router = APIRouter(prefix="/api/projects", tags=["projects"])

_WORKSPACE_TURN = threading.Lock()
"""Held by every read of the workspace's project records and by a removal, so no read reopens a
project's store while its folder is being moved."""


class ProjectFacts(BaseModel):
    """What a workspace project's directory holds, whether or not its record reads.

    ``tree`` is the project root's :class:`~tcip_web.routes.dataset.DatasetTree`, its
    ``dataset_root`` the project's directory, so the pickers never offer a date with nothing
    there.
    """

    modified: float
    tree: DatasetTree
    image_count: int


class ReadProject(ReadRecord, ProjectFacts):
    """A workspace project whose record read: the record and its directory's facts."""


class UnreadProject(UnreadRecord, ProjectFacts):
    """A workspace project whose record will not read: why, and its directory's facts."""


type ProjectSummary = Annotated[
    ReadProject | UnreadProject, Field(discriminator="kind")
]
"""One workspace project as listed, told apart by its record reading's ``kind``."""

_SUMMARY: TypeAdapter[ProjectSummary] = TypeAdapter(ProjectSummary)


def _summarize(listed: workspace.ListedProject) -> ProjectSummary:
    """Summarize the workspace project ``listed``: its reading beside its directory's facts.

    The directory itself is served once, as the facts' ``tree.dataset_root``.
    """
    from tcip_mcp.pipelines.image_utils import logical_image_counts

    root = listed.root
    return _SUMMARY.validate_python(
        {
            **{field: value for field, value in listed if field != "root"},
            "modified": root.stat().st_mtime,
            "tree": tree_of(root),
            "image_count": sum(logical_image_counts(root).values()),
        }
    )


class ProjectsListing(BaseModel):
    """The workspace's projects, newest first, with the id of the one the backend has open.

    ``open_id`` is ``None`` while none is open.
    """

    workspace: str
    open_id: str | None
    projects: list[ProjectSummary]


@router.get("")
def list_projects() -> ProjectsListing:
    """List the workspace's projects (:class:`ProjectsListing`)."""
    projects: list[ProjectSummary] = []
    with _WORKSPACE_TURN:
        for listed in workspace.project_records(store.workspace):
            try:
                projects.append(_summarize(listed))
            except OSError:
                # A project deleted mid-listing must not 500 the whole list.
                continue
    projects.sort(key=lambda p: p.modified, reverse=True)
    opened = store.opened
    return ProjectsListing(
        workspace=str(store.workspace),
        open_id=None if opened is None else opened.id,
        projects=projects,
    )


def workspace_project(project_id: str) -> tuple[Path, ProjectRecord]:
    """Return the directory and record of the workspace project with ``project_id``.

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
    """The project an open made the backend's open one."""

    project: SnapshotProject


@router.post("/open")
async def open_project(req: OpenRequest) -> OpenAnswer:
    """Open the project ``req.id`` names and return it (:class:`OpenAnswer`).

    400 for a request naming no one, 404 when the workspace holds no single project with it (a
    project whose record will not read is not found by its id), 409 when its persisted GUI
    state, or the stats record of the project it replaces, will not read, its detail the
    :data:`~tcip_store.ReadRefusal` saying why, and 409 carrying the answer when the open
    committed and a departed session's audit line could not be written
    (:class:`~tcip_web.routes.audit_gap.AuditGapRefusal`).
    """
    from tcip_store import ReadRefusedError

    actor(req.user)
    root, record = workspace_project(req.id)
    project = OpenProject(root, record.id)
    answer = OpenAnswer(project=SnapshotProject.of(project))
    try:
        await store.open_project(project)
    except ReadRefusedError as exc:
        raise HTTPException(409, exc.refusal.model_dump()) from exc
    except AuditEntryNotWrittenError as exc:
        raise audit_gap_409(exc, answer) from exc
    return answer


def _job_conflict(target: Path) -> str | None:
    """Return the refusal naming the first unfinished job that runs for ``target``, else None.

    The jobs looked through are those of this process's two in-memory job registries.
    """
    from tcip_web.routes import annotate, inference

    unended = [
        *inference._registry.list(str(target), unended=True),
        *annotate._pq_registry.list(str(target), unended=True),
    ]
    if not unended:
        return None
    job = unended[0]
    return f"{job.kind} {job.job_id!r} is not finished; wait for it to finish"


class RemovalRequest(BaseModel):
    """A request to remove the project ``id`` names, confirmed by its typed display name."""

    id: str
    # The project's display name, typed by the person removing it.
    confirm_name: str
    user: str


@router.post("/remove")
async def remove_project(req: RemovalRequest) -> workspace.ProjectRemoval:
    """Archive the project ``req.id`` names into the workspace's ``.removed/`` and move it there.

    The project, when it is the open one, is closed only once the removal is admitted. 404 for
    no such project, 400 for a confirm name that is not its display name, 409 while a job or run
    of it is live, the archive refuses or another process holds the folder past the move's retry
    budget, 400 for a request naming no one.
    """
    person = actor(req.user)
    project, record = workspace_project(req.id)
    display_name = record.display_name
    if req.confirm_name != display_name:
        raise HTTPException(
            400, f"type the project's name {display_name!r} to remove it"
        )
    conflict = _job_conflict(project)
    if conflict is not None:
        raise HTTPException(409, conflict)

    def release() -> None:
        anyio.from_thread.run(store.close_project, req.id)

    def remove() -> workspace.ProjectRemoval:
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
def rename_project_route(req: RenameRequest) -> RecordCorrection:
    """Change the display name of the project ``req.id`` names; its directory is left as it is.

    404 for no such project, 400 for a display name the record refuses or a request naming no
    one.
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
) -> RecordCorrection:
    """Correct the capture invariances the project ``req.id`` names records.

    404 for no such project, 400 for a request naming no one, and 422 for a block the record's
    type refuses.
    """
    person = actor(req.user)
    project, _ = workspace_project(req.id)
    return replace_invariances(project, req.invariances, actor=person)
