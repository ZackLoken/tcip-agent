"""Workspace project discovery + the active-project marker.

The front door: instead of browsing the filesystem for a project root and a dataset
root, the GUI lists the projects the agent built under the workspace
(``TCIP_WORKSPACE``, default ``~/tcip-projects/``) and opens one. The workspace and the
active-project marker are resolved through :mod:`tcip_mcp.workspace`: the single source
of truth shared with the ``ingest_images`` tool and the ``activate_project`` tool, so a
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
from tcip_mcp.project_record import site_fields

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
    # Exactly one is set (site_fields never raises), so a recordless or damaged project still lists.
    site: str | None
    site_problem: str | None
    # The first date's labels that would not read, naming the file; the project still lists, and
    # its subjects_by_date reports that date empty rather than aborting the scan.
    label_problem: str | None


class ActiveProject(BaseModel):
    name: str | None = None
    path: str | None = None


def _subjects_by_date(project_dir: Path, dates: list[str]) -> tuple[dict[str, list[str]], str | None]:
    """``subjects_with_labels`` per date, and the first date's problem when one won't read.

    The one implementation, shared with the per-dataset tree (``routes.dataset``): a date
    whose labels won't read reports an empty subject list for that date, never aborts the
    project's own listing.
    """
    from tcip_web.routes.dataset import _subjects_by_date as _dataset_subjects_by_date

    return _dataset_subjects_by_date(project_dir, dates)


def _summarize(project_dir: Path, active_name: str | None) -> ProjectSummary:
    st = project_dir.stat()
    images_dir = project_dir / "images"
    image_count = 0
    if images_dir.is_dir():
        image_count = sum(
            1 for f in images_dir.rglob("*") if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        )
    dates = dataset_layout.list_dates(project_dir)
    site = site_fields(project_dir)
    subjects_by_date, label_problem = _subjects_by_date(project_dir, dates)
    return ProjectSummary(
        name=project_dir.name,
        path=str(project_dir),
        created=st.st_ctime,
        modified=st.st_mtime,
        dates=dates,
        subjects=dataset_layout.list_subjects(project_dir),
        models=dataset_layout.list_models(project_dir),
        subjects_by_date=subjects_by_date,
        models_by_date={d: dataset_layout.models_with_predictions(project_dir, d) for d in dates},
        image_count=image_count,
        is_active=project_dir.name == active_name,
        site=site["site"],
        site_problem=site["site_problem"],
        label_problem=label_problem,
    )


@router.get("")
def list_projects() -> dict:
    """List workspace projects (directories containing ``.tcip/``), newest first.

    Carries ``platform_root``/``platform_root_source`` when this backend has bound one
    (:func:`tcip_mcp.project_paths.root_binding`, populated once the app has served its first
    request or repinned via ``activate_project``, never merely imported): the backend's own
    platform-state root, so the GUI can show it disagreeing with ``active``/``active_path`` in
    the window before a repin lands.

    ``job_registry_startup_refusals`` names every job-registry rehydrate this process has
    refused (an unconformed document, :func:`tcip_web.jobstore.startup_refusals`), each error
    text already naming the conform script; empty when nothing was refused.
    """
    from tcip_mcp.project_paths import root_binding

    from tcip_web import jobstore

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
    result = {
        "workspace": str(root),
        "active": active,
        "active_path": active_path,
        "projects": [p.model_dump() for p in projects],
        "job_registry_startup_refusals": jobstore.startup_refusals(),
    }
    binding = root_binding()
    if binding is not None:
        result["platform_root"] = str(binding.root)
        result["platform_root_source"] = binding.source
    return result


class SetActiveRequest(BaseModel):
    name: str


@router.post("/active")
def activate_project(req: SetActiveRequest) -> ActiveProject:
    """Set the active project (the marker the GUI auto-opens). Name must be a workspace
    project; traversal/separators are rejected, and its ``.tcip`` must already exist."""
    try:
        path = workspace.project_path(req.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        workspace.activate_project(req.name)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return ActiveProject(name=req.name, path=str(path))
