"""Workspace project discovery + the active-project marker.

The front door: instead of browsing the filesystem for a project root and a dataset
root, the GUI lists the projects the agent built under the workspace
(``TCIP_WORKSPACE``, default ``~/tcip-projects/``) and opens one. The workspace and the
active-project marker are resolved through :mod:`tcip_mcp.workspace`: the single source
of truth shared with the ``ingest_images`` tool and the ``set_active_project`` tool, so a
project the agent creates is exactly the project this route lists.

Trust boundary: same as every other REST route (``tcip_web.trust_boundary``). Listing is
inherently confined to the workspace directory, and the active-project name is validated
as a single path segment, so neither can be coaxed into reaching outside the workspace.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_mcp import dataset_layout, workspace

router = APIRouter(prefix="/api/projects", tags=["projects"])

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".bmp"}


class ProjectSummary(BaseModel):
    name: str
    path: str
    created: float
    modified: float
    dates: list[str]
    subjects: list[str]
    models: list[str]
    # Per-date availability: which subjects actually have labels / which models actually
    # have predictions on each date. The subject/model pickers filter to these so a date
    # with no labels for a subject doesn't offer it (which would open an empty canvas).
    subjects_by_date: dict[str, list[str]]
    models_by_date: dict[str, list[str]]
    image_count: int
    is_active: bool


class ActiveProject(BaseModel):
    name: str | None = None
    path: str | None = None


def _summarize(project_dir: Path, active_name: str | None) -> ProjectSummary:
    st = project_dir.stat()
    images_dir = project_dir / "images"
    image_count = 0
    if images_dir.is_dir():
        image_count = sum(
            1 for f in images_dir.rglob("*") if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        )
    dates = dataset_layout.list_dates(project_dir)
    return ProjectSummary(
        name=project_dir.name,
        path=str(project_dir),
        created=st.st_ctime,
        modified=st.st_mtime,
        dates=dates,
        subjects=dataset_layout.list_subjects(project_dir),
        models=dataset_layout.list_models(project_dir),
        subjects_by_date={d: dataset_layout.subjects_with_labels(project_dir, d) for d in dates},
        models_by_date={d: dataset_layout.models_with_predictions(project_dir, d) for d in dates},
        image_count=image_count,
        is_active=project_dir.name == active_name,
    )


@router.get("")
def list_projects() -> dict:
    """List workspace projects (directories containing ``.tcip/``), newest first."""
    root = workspace.workspace_root()
    found = workspace.active_project_if_present()
    active = found[0] if found else None
    active_path = str(found[1]) if found else None
    projects: list[ProjectSummary] = []
    for child in root.iterdir():
        if child.is_dir() and (child / ".tcip").is_dir():
            try:
                projects.append(_summarize(child, active))
            except OSError:
                # A project deleted/renamed mid-listing must not 500 the whole list.
                continue
    projects.sort(key=lambda p: p.modified, reverse=True)
    return {
        "workspace": str(root),
        "active": active,
        "active_path": active_path,
        "projects": [p.model_dump() for p in projects],
    }


class SetActiveRequest(BaseModel):
    name: str


@router.post("/active")
def set_active_project(req: SetActiveRequest) -> ActiveProject:
    """Set the active project (the marker the GUI auto-opens). Name must be a workspace
    project; traversal/separators are rejected, and its ``.tcip`` must already exist."""
    try:
        path = workspace.project_path(req.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        workspace.set_active_project(req.name)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return ActiveProject(name=req.name, path=str(path))
