"""Project removal: archive now, mark for removal, move at the next backend start.

Two doors, GUI-only (the picker's "Remove..." dialog; the agent has no MCP tool for this, per
the owner's ruling): :func:`request_project_removal` (phase one, run from the removal route)
and :func:`complete_pending_removals` (phase two, run once at backend startup, after
``bind_startup_root``'s early return, or through ``tcip complete-removals``). Phase one
archives the project (``tcip_mcp.tools.project_tools.archive_project``, models included) into
the workspace's ``.removed/`` holding directory, then writes a ``pending_removal`` marker on
the project itself; every workspace opener and walker that already exists refuses or skips a
marked project from that moment (``tcip_mcp.workspace.adoptable_project_root``,
``tcip_web.paths.allowed_roots``'s excluded roots). Phase one leaves the project's directory
exactly where it was: nothing is deleted, and the archive round-trips through
``tcip import-project``. Phase two walks the workspace and renames each marked project's
directory onto its own holding path, deleting the marker only once the rename and its own
completion line have landed.

A completed request leaves two lines in the open project's own log (the archive door's own
``archive_project`` line, then the route's own line about the request), never in the target's:
the marker is written before any log line, so a refused or failed archive leaves the target's
own state, and its own log, untouched. The removed project's own log gets one further line, its
own last: the request line at phase one, the completion line at phase two, so a moved tree's own
audit trail ends by saying what happened to it.

A denied rename in phase two means some process still holds the target's database open: this
backend's own request thread (refusal checks read the target's experiment records and keep that
connection open for the process's life), the agent's MCP server (which closes no connection it
has opened until it exits), or a second backend on the same workspace. The retry budget
(:data:`RENAME_BUDGET_S`) is for a transient handle only (a scanner, an indexer); a held
database is not transient, and the marker stays for the next start to retry. The holding path is
26 characters longer than the project's own (``.removed/``, the hyphen, and the sixteen-
character UTC instant), which can matter against the Windows path-length limit on a project
already near it. Archiving costs disk (every database exported to loose files, then a full ZIP
of the tree) and time proportional to the tree's size; a crash between the archive and the
marker leaves an abandoned ZIP with no marker beside it, which nothing else lists and an
operator identifies by hand.

MCP tools and console commands addressed by path (``inspect_project``, ``tcip doctor``,
``archive_project`` itself, ``initialize_project``, ``register_dataset``) still reach a pending
project's tree until phase two moves it: the ruling's openers are the workspace adopters and the
GUI, not every path-addressed door. A write the agent's MCP process makes into the target
(``run_inference``, which leaves no experiment status and no backend job) is invisible to every
refusal here; on Windows under the database backend its held connection denies phase two's
rename and is reported; on POSIX or under the file backend the move lands under that writer.

Importing this module pulls in nothing that imports ``tcip_mcp.server``: the three job
registries, ``archive_project``, ``read_datasets_raw`` and ``dataset_entry_path`` are imported
inside :func:`request_project_removal`'s and :func:`removal_preview`'s own bodies, so the MCP
server's tool registration reaches a web backend process on the first preview or removal
request, never at startup.
"""

from __future__ import annotations

import errno
import logging
import os
import stat
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import tcip_store
from tcip_store.file_backend import retry_while_denied

from tcip_mcp import audit, workspace

logger = logging.getLogger(__name__)

REMOVED_DIRNAME = ".removed"
"""The workspace's holding directory for an archived, not-yet-moved project. Created on first
use; never listed as a project (it carries no ``.tcip`` of its own) and excluded from the
client-path allow-set by name (``tcip_web.paths._excluded_by_name``)."""

RENAME_BUDGET_S = 5.0
"""How long :func:`complete_pending_removals` retries a rename a transient handle (a scanner, an
indexer) denies. A held database is not transient, so this bounds the delay a blocked removal
adds to every start rather than waiting on a holder that will not release."""

_request_lock = threading.Lock()
"""Serializes :func:`request_project_removal` within one process: the second of two concurrent
requests meets the first one's freshly written marker rather than racing its own archive."""

_startup_outcomes: list[dict] = []
_startup_outcomes_lock = threading.Lock()


@dataclass(frozen=True)
class _Refusal:
    status: int
    message: str


def _is_link_or_junction(path: Path) -> bool:
    """A symbolic link (POSIX) or a Windows junction/reparse point.

    Either would let the archive follow the link's target while a rename moves the link entry
    alone, leaving the two out of step, so the door refuses rather than acting on it.
    """
    try:
        st = os.lstat(path)
    except OSError:
        return False
    if getattr(st, "st_file_attributes", 0) & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
        return True
    return stat.S_ISLNK(st.st_mode)


def _same_path(target: Path, other: str | os.PathLike | None) -> bool:
    if not other:
        return False
    try:
        return os.path.samefile(target, other)
    except OSError:
        return False


def _open_project_conflict(target: Path) -> Optional[str]:
    """The target compared, by filesystem identity, against the three spellings of "the open
    project": the workspace's active-project marker, this process's own platform root, and the
    GUI's canvas-open binding. Each is its own message naming the spelling and its remedy."""
    from tcip_mcp.project_paths import root_binding
    from tcip_mcp.web_client import GuiBindingUnreadable, read_canvas_binding

    marker_name = workspace.read_active_project()
    if marker_name:
        try:
            marker_root = workspace.project_path(marker_name, create=False)
        except ValueError:
            marker_root = None
        if marker_root is not None and _same_path(target, marker_root):
            return (f"{marker_name!r} is the workspace's active project (the marker that opens "
                    "it); adopt a different project first")

    binding = root_binding()
    if binding is not None and _same_path(target, binding.root):
        return "this backend's own platform root is the project; adopt a different project first"

    try:
        canvas = read_canvas_binding()
    except GuiBindingUnreadable as exc:
        return f"the GUI's open-project binding could not be read: {exc}"
    if canvas and (canvas.get("project_name") or canvas.get("root")):
        root = canvas.get("root")
        if root and _same_path(target, root):
            label = canvas.get("project_name") or root
            return f"{label!r} is the project the GUI has open; open a different project first"
    return None


def _live_run_conflict(target: Path) -> Optional[str]:
    from tcip_mcp import experiments
    from tcip_mcp.tools.training_tools import TCIP_HEARTBEAT_STALE_SECONDS

    for exp_id in experiments.experiment_ids_with_status(root=target):
        status = experiments.read_member(experiments.status_key(exp_id, root=target), {})
        if experiments.derived_state(status, TCIP_HEARTBEAT_STALE_SECONDS) == "running":
            return (f"experiment {exp_id!r} is running; ask the agent to cancel it "
                     "(cancel_training) or wait for it to finish")
    return None


def _under_target(value: str, target: Path) -> bool:
    from tcip_mcp.registry_paths import nearest_containing_ancestor

    if not value:
        return False
    return nearest_containing_ancestor(Path(value), target, tolerant=True) is not None


def _job_conflict(target: Path) -> Optional[str]:
    """Every non-terminal job in the three registries whose ``platform_root`` is the target, or
    whose server-recorded directories resolve under it. The client's own raw
    ``requested_dataset_root`` is never consulted, only what the server itself resolved."""
    from tcip_mcp.tools import training_tools
    from tcip_web import jobstore
    from tcip_web.routes import inference, review, tuning

    target_str = str(target)
    for job in inference._registry.list():
        if job.status in jobstore.TERMINAL_STATUSES:
            continue
        if job.platform_root == target_str or any(
            _under_target(v, target) for v in (job.checkpoint_path, job.images_dir, job.output_dir)
        ):
            return (f"inference job {job.job_id!r} is not finished; ask the agent to cancel it "
                     "or wait for it to finish")
    for job in tuning._registry.list():
        if job.status in jobstore.TERMINAL_STATUSES:
            continue
        sweep_directory = str(training_tools.sweep_dir(job.sweep_id, root=job.platform_root))
        if job.platform_root == target_str or _under_target(sweep_directory, target):
            return (f"HPO sweep {job.sweep_id!r} is not finished; ask the agent to cancel it or "
                     "wait for it to finish")
    for job in review._pq_registry.list():
        if job.status in jobstore.TERMINAL_STATUSES:
            continue
        if job.platform_root == target_str or any(
            _under_target(v, target) for v in (job.checkpoint_path, job.images_dir, job.dataset_root)
        ):
            return (f"a review priority-queue scoring pass ({job.job_id!r}) is not finished; "
                    "wait for it to finish or restart the backend, which marks it interrupted")
    return None


def _ordered_refusal(name: str, confirm_name: str) -> Optional[_Refusal]:
    """Decision order: name shape, confirmation, existence, a link, the marker, this backend
    having a project open at all, the three spellings, a live run, a non-terminal job. The first
    refusal wins; ``None`` means the removal may proceed."""
    if not workspace.is_valid_name(name):
        return _Refusal(400, f"invalid project name: {name!r}")
    if name != name.strip():
        return _Refusal(400, f"{name!r} carries leading or trailing whitespace")
    if name == REMOVED_DIRNAME:
        return _Refusal(400, f"{REMOVED_DIRNAME!r} is the workspace's own holding directory, "
                              "not a project")
    if confirm_name != name:
        return _Refusal(400, "the typed name does not match the project's name")

    try:
        project = workspace.workspace_project_root(name)
    except ValueError as exc:
        return _Refusal(404, str(exc))

    if _is_link_or_junction(project):
        return _Refusal(409, f"{name!r} is a symbolic link or a junction; the archive would "
                              "follow it while a rename would move the link alone, so remove it "
                              "by hand instead")

    try:
        pending = workspace.pending_removal_record(project)
    except (tcip_store.StoreError, tcip_store.DecodeError, tcip_store.SchemaVersionRefused) as exc:
        return _Refusal(409, f"{name!r}'s pending-removal marker could not be read: {exc}")
    if pending is not None:
        return _Refusal(
            409,
            f"{name!r} already has a pending-removal marker (requested "
            f"{pending['requested_at']}); it moves to the workspace's holding directory at the "
            "next backend start, or through tcip complete-removals",
        )

    from tcip_mcp.project_paths import root_binding

    binding = root_binding()
    open_name = workspace.workspace_project_name(binding.root) if binding is not None else None
    if open_name is None:
        bound = binding.root if binding is not None else "none"
        return _Refusal(
            409,
            f"no project is open in this backend (bound root: {bound}); open a project first "
            "so the request is recorded in its log",
        )

    conflict = _open_project_conflict(project)
    if conflict is not None:
        return _Refusal(409, conflict)

    live_run = _live_run_conflict(project)
    if live_run is not None:
        return _Refusal(409, live_run)

    job_conflict = _job_conflict(project)
    if job_conflict is not None:
        return _Refusal(409, job_conflict)

    return None


def removal_preview(name: str) -> dict:
    """Every fact the removal dialog needs before a name is even typed.

    ``refusal`` is the text of the first refusal :func:`_ordered_refusal` would answer with
    (confirmation trivially matches here, since this door takes no confirm_name of its own), or
    ``None``. ``external_roots`` is every path ``tcip_mcp.store_catalogue.project_roots`` names
    for this project that is not under it, listed once per path with the layouts it serves.
    ``dependent_projects`` is every other workspace project's dataset registry entry that
    resolves under this project's tree, pending ones included and marked ``pending``; a project
    whose own registry will not read is listed as ``{project, unreadable}`` rather than dropped.

    Reads the target's experiment records (the live-run refusal check), which opens its database
    on this request's thread and keeps it open for the process's life: previewing and cancelling
    still leaves this backend holding the target until it restarts.
    """
    from tcip_mcp.registry_paths import nearest_containing_ancestor
    from tcip_mcp.store_catalogue import project_roots
    from tcip_mcp.tools.project_tools import dataset_entry_path, read_datasets_raw

    refusal = _ordered_refusal(name, name)

    try:
        project = workspace.workspace_project_root(name)
    except ValueError:
        return {"external_roots": [], "dependent_projects": [],
                "refusal": refusal.message if refusal else None}

    roots_by_path: dict[str, list[str]] = {}
    try:
        # A damaged target registry is the archive step's own refusal to surface, not a crash here.
        target_roots = project_roots(project)
    except Exception:  # noqa: BLE001 - degrades to no known external roots, never crashes
        target_roots = ()
    for root_path, layout in target_roots:
        if nearest_containing_ancestor(Path(root_path), project, tolerant=True) is None:
            roots_by_path.setdefault(root_path, []).append(layout)
    external_roots = [
        {"path": path, "layouts": sorted(layouts)}
        for path, layouts in sorted(roots_by_path.items())
    ]

    dependent_projects: list[dict] = []
    ws = workspace.workspace_root(create=False)
    for child in sorted(p for p in ws.iterdir() if p.is_dir()):
        if _same_path(child, project) or not (child / ".tcip").is_dir():
            continue
        pending = workspace.pending_removal_or_none(child)
        try:
            entries = read_datasets_raw(child)
        except Exception as exc:  # noqa: BLE001 - reported per project, never aborts the scan
            dependent_projects.append({"project": child.name, "unreadable": str(exc)})
            continue
        for entry in entries:
            if not entry.get("path"):
                continue
            try:
                entry_path = dataset_entry_path(child, entry)
            except ValueError:
                continue
            if nearest_containing_ancestor(entry_path, project, tolerant=True) is not None:
                dependent_projects.append({
                    "project": child.name, "dataset_id": entry.get("id"),
                    "dataset_path": str(entry_path), "pending": pending is not None,
                })

    return {
        "external_roots": external_roots,
        "dependent_projects": dependent_projects,
        "refusal": refusal.message if refusal else None,
    }


def request_project_removal(name: str, confirm_name: str, user: str) -> dict:
    """Phase one: archive the named workspace project, then mark it pending removal.

    Refuses (see :func:`_ordered_refusal`) before anything is written. On success: the archive
    is under the workspace's holding directory, the marker is on the project, and this response
    carries ``{name, archive_path, holding_dir, external_roots, dependent_projects, completes,
    audit_scope}``. Two audit lines land in the open project's own log (never the target's): the
    archive door's own line, then this door's own line about the request. Nothing is deleted.
    """
    from tcip_web import identity

    with _request_lock:
        refusal = _ordered_refusal(name, confirm_name)
        preview = removal_preview(name)
        if refusal is not None:
            return {"error": refusal.message, "status": refusal.status}

        project = workspace.workspace_project_root(name)
        external_roots = preview["external_roots"]
        dependent_projects = preview["dependent_projects"]

        from tcip_mcp.tools.project_tools import archive_project

        ws = workspace.workspace_root()
        removed_dir = ws / REMOVED_DIRNAME
        removed_dir.mkdir(exist_ok=True)
        requested_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_path = removed_dir / f"{name}-{requested_at}.zip"
        holding_dir = removed_dir / f"{name}-{requested_at}"

        result = archive_project(
            project_path=str(project), output_path=str(archive_path), include_models=True,
        )
        if "error" in result:
            return {"error": result["error"], "status": 409}

        requested_by = identity.user_id(identity.resolve_user(user))
        record = {
            "requested_at": requested_at,
            "requested_by": requested_by,
            "archive_path": str(archive_path),
            "holding_dir": str(holding_dir),
            "external_roots": external_roots,
            "dependent_projects": dependent_projects,
        }
        try:
            tcip_store.replace(
                workspace.pending_removal_key(project), record, expect=tcip_store.Version.ABSENT,
            )
        except tcip_store.VersionConflict as exc:
            return {"error": f"another request already marked {name!r} for removal: {exc}",
                    "status": 409}

        try:
            audit.record_event_or_raise(
                "project_removal_requested",
                {"name": name, "archive_path": str(archive_path), "holding_dir": str(holding_dir),
                 "requested_by": requested_by},
                scope=project,
            )
        except audit.AuditEntryNotWritten as exc:
            return {"error": f"the marker for {name!r} is written but its own log line is not: "
                              f"{exc}. The removal still completes at the next backend start.",
                    "status": 409}

        from tcip_mcp.project_paths import root_binding

        open_root = root_binding().root  # type: ignore[union-attr]  # non-None: checked above
        try:
            audit.record_event_or_raise(
                "gui_project_removal_requested",
                {"name": name, "archive_path": str(archive_path), "holding_dir": str(holding_dir),
                 "external_roots": external_roots, "dependent_projects": dependent_projects},
                scope=open_root,
            )
        except audit.AuditEntryNotWritten as exc:
            return {"error": f"the marker for {name!r} and its own log line are written, but "
                              f"this backend's line is not: {exc}. The removal still completes "
                              "at the next backend start.",
                    "status": 409}

        return {
            "name": name,
            "archive_path": str(archive_path),
            "holding_dir": str(holding_dir),
            "external_roots": external_roots,
            "dependent_projects": dependent_projects,
            "completes": "at the next backend start, or tcip complete-removals",
            "audit_scope": str(open_root),
        }


def complete_pending_removals(workspace_root: Path) -> list[dict]:
    """Phase two: move every workspace project carrying a pending-removal marker onto its own
    holding directory. Runs once per process, before any other store is opened in intent (each
    project's own store is opened on this thread to read its marker, and closed before that
    project's rename); a per-project store refusal is skipped rather than stopping the walk.

    Each outcome is ``{name, moved_to, archive_path}`` on a completed move, ``{name, blocked_by,
    archive_path}`` when the rename was denied past :data:`RENAME_BUDGET_S` or crosses a
    filesystem boundary (the marker stays for the next start to retry), or ``{name, skipped,
    archive_path: None}`` when the marker itself could not be read (a loose layout, an
    undecodable or over-version document): that project is left exactly as it was.
    """
    outcomes: list[dict] = []
    ws = workspace_root
    if not ws.is_dir():
        return outcomes

    for child in sorted(p for p in ws.iterdir() if p.is_dir() and (p / ".tcip").is_dir()):
        name = child.name
        try:
            versioned = tcip_store.read_versioned(workspace.pending_removal_key(child), default=None)
        except (tcip_store.StoreError, tcip_store.DecodeError, tcip_store.SchemaVersionRefused) as exc:
            outcomes.append({"name": name, "skipped": str(exc), "archive_path": None})
            continue
        record = versioned.value
        if record is None:
            continue
        version = versioned.version
        removed_dir = ws / REMOVED_DIRNAME
        holding_dir = Path(record["holding_dir"])
        archive_path = record.get("archive_path")

        tcip_store.close_connections()
        try:
            removed_dir.mkdir(exist_ok=True)

            def _rename(src: Path = child, dst: Path = holding_dir) -> None:
                os.rename(str(src), str(dst))

            retry_while_denied(_rename, RENAME_BUDGET_S)
        except OSError as exc:
            reason = (f"{holding_dir} is on another filesystem: {exc}"
                      if getattr(exc, "errno", None) == errno.EXDEV else str(exc))
            outcomes.append({"name": name, "blocked_by": reason, "archive_path": archive_path})
            continue

        notes: list[str] = []
        try:
            audit.record_event_or_raise(
                "project_removal_completed",
                {"name": name, "moved_to": str(holding_dir), "archive_path": archive_path},
                scope=holding_dir,
            )
        except audit.AuditEntryNotWritten as exc:
            notes.append(str(exc))

        try:
            tcip_store.delete(workspace.pending_removal_key(holding_dir), expect=version)
        except tcip_store.StoreError as exc:
            notes.append(str(exc))

        tcip_store.close_connections()

        outcome = {"name": name, "moved_to": str(holding_dir), "archive_path": archive_path}
        if notes:
            outcome["note"] = "; ".join(notes)
        outcomes.append(outcome)

    with _startup_outcomes_lock:
        _startup_outcomes.clear()
        _startup_outcomes.extend(outcomes)
    return outcomes


def startup_outcomes() -> list[dict]:
    """Every outcome the last :func:`complete_pending_removals` run recorded, kept for this
    process's life the way ``tcip_web.jobstore.startup_refusals`` is."""
    with _startup_outcomes_lock:
        return list(_startup_outcomes)
