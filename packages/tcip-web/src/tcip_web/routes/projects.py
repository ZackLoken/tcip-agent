"""Workspace project discovery + the active-project marker + project removal.

The front door: instead of browsing the filesystem for a project root and a dataset
root, the GUI lists the projects the agent built under the workspace
(``TCIP_WORKSPACE``, default ``~/tcip-projects/``) and opens one. The workspace and the
active-project marker are resolved through :mod:`tcip_mcp.workspace`: the single source
of truth shared with the ``ingest_images`` tool and the ``activate_project`` tool, so a
project the agent creates is exactly the project this route lists.

Trust boundary: same as every other REST route (``tcip_web.trust_boundary``). Listing is
inherently confined to the workspace directory, and the active-project name is validated
as a single path segment, so neither can be coaxed into reaching outside the workspace.

Seven doors write here. ``POST /active`` writes the active-project marker only, its own
line staying wherever the MCP tool's own caller emits one (this route emits none itself).
``GET /{name}/removal-preview`` writes nothing (:func:`tcip_mcp.project_removal.
removal_preview`). ``POST /remove`` (:func:`tcip_mcp.project_removal.request_project_removal`)
is GUI-only and, on success, leaves the removed project's own last line, naming the marker just
written, in the target's own log; this request's own line joins the archive door's own line in
a bound project's own log when this process is bound to one (``recorded_in_open_project`` true
in the response), or joins the target's own line in the target's own log otherwise, so a
workspace with nothing bound still records the whole request. Phase two's own completion line
lands on the moved tree at the next backend start. ``POST /{name}/release-binding``
(:func:`tcip_mcp.project_removal.release_project_binding`) clears the marker when it names the
project, marks the canvas-open binding released when it does (never deleting either the binding
record or repinning this process), and, when either changed, records one line under the
project's own root.

``GET /{name}/rename-preview`` writes nothing (:func:`tcip_mcp.project_rename.rename_preview`).
``POST /rename`` (:func:`tcip_mcp.project_rename.request_project_rename`) is GUI-only, marks the
project pending rename with no archive taken, leaves the target's own last line naming the
marker in its own log, one ``dependency_pending_rename`` line per dependent in that dependent's
own log, and the route's own line in the bound project's own log or the target's own otherwise.
``POST /rename/withdraw`` (:func:`tcip_mcp.project_rename.withdraw_project_rename`) clears a
pending-rename marker with no rename of its own, for a request whose destination name was taken
before phase two ran.

``tcip_mcp.project_removal`` and ``tcip_mcp.project_rename`` import nothing from ``tcip_web``:
this module resolves the requesting identity through :mod:`tcip_web.identity` and supplies
:func:`_job_conflict`, the walk over the three job registries both doors' own refusal chains
call through, as their required keyword arguments.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_mcp import dataset_layout, project_removal, workspace
from tcip_mcp.project_record import site_fields
from tcip_web import identity

router = APIRouter(prefix="/api/projects", tags=["projects"])

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".bmp"}


class DependencyWarning(BaseModel):
    dataset_id: str
    dataset_path: str
    # The dependency's own workspace-child name.
    target: str
    # False once the dependency's own directory no longer exists at the location named.
    present: bool
    # From the target's own pending-removal marker while present is true; null for a pending
    # rename (which archives nothing) or once present is false.
    archive_path: str | None = None
    holding_dir: str | None = None
    # "removal" or "rename" while present is true; absent when the target is gone rather than
    # pending. Read by the Remove... and Rename... cards' own warning line.
    pending_kind: str | None = None
    # The name the target takes, from its own pending-rename marker; null for a removal or once
    # present is false. The card names it, so the dependent's owner knows what to re-register at.
    new_name: str | None = None


class ProjectSummary(BaseModel):
    name: str
    path: str
    created: float
    modified: float
    dates: list[str]
    subjects: list[str]
    models: list[str]
    # Per-date availability: which subjects have labels / which models have predictions on each
    # date, so the pickers never offer a date with nothing there (an empty canvas).
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
    # project_removal.identity_conflict's own text, or null; the control is disabled only for a
    # refusal removal_releasable cannot clear.
    removal_refusal: str | None
    # Whether the control stays enabled beside removal_refusal: a release would clear it.
    removal_releasable: bool
    # Every dataset this project registered under another workspace project now pending removal,
    # pending rename or gone; each entry's pending_kind says which.
    dependency_warnings: list[DependencyWarning]
    # A complete sentence, or null: a registry that will not decode leaves dependency_warnings
    # empty beside it; an entry with a path and no id leaves the other entries' warnings beside it.
    dependency_problem: str | None


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


def _summarize(
    project_dir: Path, active_name: str | None, open_state: project_removal.OpenProjectState,
) -> ProjectSummary:
    st = project_dir.stat()
    images_dir = dataset_layout.image_root(project_dir)
    image_count = 0
    if images_dir.is_dir():
        image_count = sum(
            1 for f in images_dir.rglob("*") if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        )
    dates = dataset_layout.list_dates(project_dir)
    site = site_fields(project_dir)
    subjects_by_date, label_problem = _subjects_by_date(project_dir, dates)
    warnings, dependency_problem = project_removal.dependency_warnings(project_dir)
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
        removal_refusal=project_removal.identity_conflict(project_dir, open_state),
        removal_releasable=project_removal.binding_release_available(project_dir, open_state),
        dependency_warnings=[DependencyWarning(**w) for w in warnings],
        dependency_problem=dependency_problem,
    )


@router.get("")
def list_projects() -> dict:
    """List workspace projects (directories containing ``.tcip/``), newest first.

    A project carrying either marker is never one of ``projects``: it is read before
    ``_summarize`` and carried instead under ``pending_removal``
    (``[{name, requested_at, archive_path, holding_dir}]``) or ``pending_rename``
    (``[{name, new_name, requested_at}]``), the way every other workspace walker skips it from
    the moment its marker lands.

    Carries ``platform_root``/``platform_root_source`` when this backend has bound one
    (:func:`tcip_mcp.project_paths.root_binding`, populated once the app has served its first
    request or repinned via ``activate_project``, never merely imported): the backend's own
    platform-state root, so the GUI can show it disagreeing with ``active``/``active_path`` in
    the window before a repin lands. ``removal_startup_outcomes`` and ``rename_startup_outcomes``
    carry the last ``complete_pending_removals``/``complete_pending_renames`` run's own outcomes
    (:func:`tcip_mcp.project_removal.startup_outcomes`,
    :func:`tcip_mcp.project_rename.rename_startup_outcomes`).

    ``job_registry_startup_refusals`` names every job-registry rehydrate this process has
    refused (an unconformed document, :func:`tcip_web.jobstore.startup_refusals`), each error
    text already naming the conform script; empty when nothing was refused.
    """
    from tcip_mcp import project_rename
    from tcip_mcp.project_paths import root_binding

    from tcip_web import jobstore

    root = workspace.workspace_root()
    found = workspace.active_project_if_present()
    active = found[0] if found else None
    active_path = str(found[1]) if found else None
    open_state = project_removal.read_open_project_state()
    projects: list[ProjectSummary] = []
    pending_removal: list[dict] = []
    pending_rename: list[dict] = []
    for child in root.iterdir():
        if not (child.is_dir() and (child / ".tcip").is_dir()):
            continue
        pending = workspace.pending_marker_or_none(child)
        if pending is not None:
            if pending.kind == "removal":
                pending_removal.append({
                    "name": child.name,
                    "requested_at": pending.record["requested_at"],
                    "archive_path": pending.record["archive_path"],
                    "holding_dir": pending.record["holding_dir"],
                })
            else:
                pending_rename.append({
                    "name": child.name,
                    "new_name": pending.record["new_name"],
                    "requested_at": pending.record["requested_at"],
                })
            continue
        try:
            projects.append(_summarize(child, active, open_state))
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
        "pending_removal": pending_removal,
        "pending_rename": pending_rename,
        "removal_startup_outcomes": project_removal.startup_outcomes(),
        "rename_startup_outcomes": project_rename.rename_startup_outcomes(),
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
    project; traversal/separators are rejected, its ``.tcip`` must already exist, and it must
    not be pending removal or pending rename (409, naming when the request was made and, for a
    rename, the new name)."""
    try:
        path = workspace.project_path(req.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        workspace.activate_project(req.name)
    except (workspace.ProjectPendingRemoval, workspace.ProjectPendingRename) as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return ActiveProject(name=req.name, path=str(path))


class DependentProject(BaseModel):
    project: str
    dataset_id: str | None = None
    dataset_path: str | None = None
    pending: bool | None = None
    # A complete sentence the dialog renders verbatim, set instead of the three fields above when
    # the dependent's registry will not read or the matching entry carries a path and no id.
    unreadable: str | None = None


class ExternalRoot(BaseModel):
    path: str
    layouts: list[str]
    # Whether the path still exists on disk: the preview can list a root under a project a
    # prior removal already moved.
    present: bool


class RemovalPreview(BaseModel):
    external_roots: list[ExternalRoot]
    dependent_projects: list[DependentProject]
    refusal: str | None = None
    # Set instead of an external_roots list when the target's own registry will not read;
    # the request is still admitted, and the dialog renders this as a warning.
    external_roots_unreadable: str | None = None
    # Whether a release (POST .../release-binding) would clear refusal: project_removal.
    # binding_release_available's own answer, set whether or not refusal is set.
    releasable: bool


class RemovalRequest(BaseModel):
    name: str
    confirm_name: str
    user: str = ""


class RemovalResponse(BaseModel):
    name: str
    archive_path: str
    holding_dir: str
    external_roots: list[ExternalRoot]
    dependent_projects: list[DependentProject]
    completes: str
    audit_scope: str
    # True when the route's own line landed in a bound project's log; false when, with none
    # bound, it landed in the target's own log instead.
    recorded_in_open_project: bool
    # Names where the request's own two lines and the archive door's own line went.
    audit_note: str


class ReleaseBindingRequest(BaseModel):
    user: str = ""


class ReleaseResponse(BaseModel):
    name: str
    marker_cleared: bool
    canvas_binding_released: bool
    # A fresh identity_conflict/binding_release_available read after the release, for a caller
    # that does not itself re-fetch the preview (the agent); the dialog re-fetches instead.
    refusal: str | None
    releasable: bool


class RenamePreview(BaseModel):
    # The shared refusal chain plus the records refusal, or null; the new-name-specific checks
    # are excluded since none is typed yet.
    refusal: str | None = None
    releasable: bool
    dependent_projects: list[DependentProject]
    # The project's own stores holding a record (project_rename.project_records_present);
    # empty for a project the rename door would admit.
    records_present: list[str]


class RenameRequest(BaseModel):
    name: str
    new_name: str
    confirm_name: str
    user: str = ""


class RenameResponse(BaseModel):
    name: str
    new_name: str
    completes: str
    dependent_projects: list[DependentProject]
    audit_scope: str
    # True when the route's own line landed in a bound project's log; false when, with none
    # bound, it landed in the target's own log instead.
    recorded_in_open_project: bool
    audit_note: str


def _job_conflict(target: Path) -> str | None:
    """Every non-terminal job in the three registries whose ``platform_root`` is ``target``, or
    whose server-recorded directories resolve under it. The client's own raw
    ``requested_dataset_root`` is never consulted, only what the server itself resolved. Lives
    here rather than in ``tcip_mcp.project_removal``: walking the job registries needs edges into
    tcip-web's own routes, and that module imports nothing from this package."""
    from tcip_mcp.registry_paths import nearest_containing_ancestor
    from tcip_mcp.tools import training_tools

    from tcip_web import jobstore
    from tcip_web.routes import inference, review, tuning

    def _under(value: str) -> bool:
        if not value:
            return False
        return nearest_containing_ancestor(Path(value), target, tolerant=True) is not None

    target_str = str(target)
    for job in inference._registry.list():
        if job.status in jobstore.TERMINAL_STATUSES:
            continue
        if job.platform_root == target_str or any(
            _under(v) for v in (job.checkpoint_path, job.images_dir, job.output_dir)
        ):
            return (f"inference job {job.job_id!r} is not finished; ask the agent to cancel it "
                     "or wait for it to finish")
    for job in tuning._registry.list():
        if job.status in jobstore.TERMINAL_STATUSES:
            continue
        if job.platform_root == target_str:
            return (f"HPO sweep {job.sweep_id!r} is not finished; ask the agent to cancel it or "
                     "wait for it to finish")
        # A rehydrated job can carry an empty platform_root; never compose a sweep directory
        # from an empty root string.
        if job.platform_root and _under(
            str(training_tools.sweep_dir(job.sweep_id, root=job.platform_root))
        ):
            return (f"HPO sweep {job.sweep_id!r} is not finished; ask the agent to cancel it or "
                     "wait for it to finish")
    for job in review._pq_registry.list():
        if job.status in jobstore.TERMINAL_STATUSES:
            continue
        if job.platform_root == target_str or any(
            _under(v) for v in (job.checkpoint_path, job.images_dir, job.dataset_root)
        ):
            return (f"a review priority-queue scoring pass ({job.job_id!r}) is not finished; "
                    "wait for it to finish or restart the backend, which marks it interrupted")
    return None


@router.get("/{name}/removal-preview")
def removal_preview_route(name: str) -> RemovalPreview:
    """A snapshot of what removing ``name`` would answer: the refusal the door's own ordered
    chain would give (or ``None``), every external root the project's own state reaches outside
    its tree, and every other project's dataset registered under it. The dialog fetches this on
    open; the request itself answers the same refusal fresh if anything changed meanwhile."""
    return RemovalPreview(**project_removal.removal_preview(name, job_conflict=_job_conflict))


@router.post("/remove")
def remove_project(req: RemovalRequest) -> RemovalResponse:
    """Archive ``name`` and mark it pending removal (:func:`tcip_mcp.project_removal.
    request_project_removal`); the only caller of that door. Refuses with the status the door
    itself names (400 malformed request, 404 no such project, 409 every other refusal)."""
    requested_by = identity.user_id(identity.resolve_user(req.user))
    result = project_removal.request_project_removal(
        req.name, req.confirm_name, requested_by=requested_by, job_conflict=_job_conflict,
    )
    if "error" in result:
        raise HTTPException(result.get("status", 400), result["error"])
    return RemovalResponse(**result)


@router.post("/{name}/release-binding")
def release_binding_route(name: str, req: ReleaseBindingRequest) -> ReleaseResponse:
    """Stop ``name`` opening by default and forget it as the GUI's own open project
    (:func:`tcip_mcp.project_removal.release_project_binding`); the only caller of that door.
    Never repins this process and never touches removal state; a project the marker and the
    canvas binding both leave alone answers with both flags false and no line."""
    requested_by = identity.user_id(identity.resolve_user(req.user))
    result = project_removal.release_project_binding(name, released_by=requested_by)
    if "error" in result:
        raise HTTPException(result.get("status", 400), result["error"])
    return ReleaseResponse(**result)


@router.get("/{name}/rename-preview")
def rename_preview_route(name: str) -> RenamePreview:
    """A snapshot of what renaming ``name`` would answer: the shared refusal chain plus the
    records refusal (or ``None``), every other project's dataset registered under it, and which
    of its own stores hold a record. The dialog fetches this on open, before a new name is
    typed; the request itself answers fresh if anything changed meanwhile."""
    from tcip_mcp import project_rename

    return RenamePreview(**project_rename.rename_preview(name, job_conflict=_job_conflict))


@router.post("/rename")
def rename_project(req: RenameRequest) -> RenameResponse:
    """Mark ``name`` pending rename to ``new_name`` (:func:`tcip_mcp.project_rename.
    request_project_rename`); the only caller of that door. Refuses with the status the door
    itself names (400 malformed request, 404 no such project, 409 every other refusal)."""
    from tcip_mcp import project_rename

    requested_by = identity.user_id(identity.resolve_user(req.user))
    result = project_rename.request_project_rename(
        req.name, req.new_name, req.confirm_name, requested_by=requested_by,
        job_conflict=_job_conflict,
    )
    if "error" in result:
        raise HTTPException(result.get("status", 400), result["error"])
    return RenameResponse(**result)


class WithdrawRenameRequest(BaseModel):
    name: str
    user: str = ""


@router.post("/rename/withdraw")
def withdraw_rename_route(req: WithdrawRenameRequest) -> dict:
    """Clear ``req.name``'s pending-rename marker with no rename of its own
    (:func:`tcip_mcp.project_rename.withdraw_project_rename`); the only caller of that door.
    Not destructive, so it takes no typed confirm name."""
    from tcip_mcp import project_rename

    requested_by = identity.user_id(identity.resolve_user(req.user))
    result = project_rename.withdraw_project_rename(req.name, requested_by=requested_by)
    if "error" in result:
        raise HTTPException(result.get("status", 400), result["error"])
    return result
