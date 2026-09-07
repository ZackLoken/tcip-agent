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

A successful archive re-exports every database under the target back out as loose files inside
its own ``.tcip`` before the marker is written, so the moved tree carries them rather than a
database file a plain archive would leave unreadable outside this process. "Open" carries two
senses that a request tells apart: the workspace's active-project marker names a default that
need not be this process's own, while a *bound* project is this process's own platform-state
root (:func:`tcip_mcp.project_paths.root_binding`); the marker, the bound root and the GUI's
canvas-open binding are the three spellings :func:`identity_conflict` refuses a request naming.
A completed request leaves the removed project's own last line, naming the marker just written,
in the target's own log; one ``dependency_pending_removal`` line per dependent project, in that
dependent's own log; and the route's own line about the request, in the bound project's own log
when one is bound, and in the target's own log otherwise, so a workspace with nothing bound
still records the whole request. The archive door's own line is not part of that choice: it is
a bare ``@audited`` door, so it lands wherever ``$TCIP_STATE_ROOT`` names at write time, unmoved
by which project (if any) is bound. A refused or failed archive leaves the target's own state,
and its own log, untouched, since the marker is written before any log line. Four crash windows
follow the marker: between it and the target's own request line landing (a marker with no
request line on the target, the next start then moving a tree whose log ends with the completion
and no request); between it and a dependent's line (that dependent's log without it, the marker
record's own ``dependent_projects`` still naming the dependent); between the last dependent's
line and the route's own (the target's log ending with ``project_removal_requested`` and no
route line); and, unrelated to the request, between the rename and the marker's own delete in
phase two (a moved tree still carrying its marker, one row-delete wide, so a tree moved back by
hand inside that window is pending again). Phase two's own completion line lands on the moved
tree at the next backend start, so a moved tree's own audit trail ends by saying what happened to
it. Undoing a completed move by hand (outside ``tcip import-project``) means renaming
``<name>-<stamp>/`` back under the workspace as ``<name>``, since the stamped holding name is not
itself a project name.

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

Importing this module pulls in nothing that imports ``tcip_mcp.server``: ``archive_project``,
``read_datasets_raw`` and ``dataset_entry_path`` are imported inside
:func:`request_project_removal`'s and :func:`removal_preview`'s own bodies, so the MCP server's
tool registration reaches a web backend process on the first preview or removal request, never
at startup. This module also imports nothing from ``tcip_web``: the resolved ``requested_by``
identity and the ``job_conflict`` callable that walks the three job registries are the caller's
own to supply (:mod:`tcip_web.routes.projects`, the only caller), rather than edges this module
holds into a package one layer above it.
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
from typing import Callable, Optional

import tcip_store
from tcip_store.file_backend import retry_while_denied

from tcip_mcp import audit, workspace

logger = logging.getLogger(__name__)

JobConflict = Callable[[Path], Optional[str]]
"""A function of a project's resolved root answering the refusal text of the first non-terminal
job found under it, or ``None``: the caller's own edge into tcip-web's job registries, since this
module holds none (:func:`request_project_removal`, :func:`removal_preview`)."""

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


@dataclass(frozen=True)
class OpenProjectState:
    """The three spellings of "the open project", read once: the active-project marker's raw
    name, this process's own bound platform root (:func:`tcip_mcp.project_paths.root_binding`),
    and the GUI's canvas-open binding (or the message naming why it would not read). Built by
    :func:`read_open_project_state` and passed to :func:`identity_conflict` for every project a
    caller checks, rather than each check re-reading the marker, the binding and the canvas
    record on its own."""
    marker_name: Optional[str]
    binding_root: Optional[Path]
    canvas: Optional[dict]
    canvas_problem: Optional[str]


def read_open_project_state() -> OpenProjectState:
    """Read the marker, the bound platform root and the canvas binding exactly once. The
    workspace listing calls this once per request and passes the result to
    :func:`identity_conflict` for every project it summarizes; the door's own refusal chain
    (:func:`_preview`) does the same for its own single target, rather than either caller
    re-reading the three per project or per call."""
    from tcip_mcp.project_paths import root_binding
    from tcip_mcp.web_client import GuiBindingUnreadable, read_canvas_binding

    binding = root_binding()
    canvas: Optional[dict] = None
    canvas_problem: Optional[str] = None
    try:
        canvas = read_canvas_binding()
    except GuiBindingUnreadable as exc:
        canvas_problem = str(exc)
    return OpenProjectState(
        marker_name=workspace.read_active_project(),
        binding_root=binding.root if binding is not None else None,
        canvas=canvas,
        canvas_problem=canvas_problem,
    )


def _open_project_conflict(target: Path, state: OpenProjectState) -> Optional[str]:
    """The target compared, by filesystem identity, against the three spellings of "the open
    project" carried on ``state``: the workspace's active-project marker, this process's own
    platform root, and the GUI's canvas-open binding (a released binding never matches: it names
    what the GUI last had open, not what it has open now). Each is its own message naming the
    spelling, that a breeder-triggered release clears it, and the remedy otherwise, the same
    strings :func:`identity_conflict` answers the listing's own per-project reason with.
    """
    if state.marker_name:
        try:
            marker_root = workspace.project_path(state.marker_name, create=False)
        except ValueError:
            marker_root = None
        if marker_root is not None and _same_path(target, marker_root):
            return (f"{state.marker_name} is the project this workspace opens by default; "
                    "choose a different default first, or release it as the default")

    if state.binding_root is not None and _same_path(target, state.binding_root):
        name = workspace.workspace_project_name(target) or target.name
        return (f"{name} is the project this backend started on; restart the backend first, "
                "started without TCIP_STATE_ROOT naming it")

    if state.canvas_problem is not None:
        return f"the GUI's open-project binding could not be read: {state.canvas_problem}"
    if state.canvas and not state.canvas.get("released") and (
        state.canvas.get("project_name") or state.canvas.get("root")
    ):
        root = state.canvas.get("root")
        if root and _same_path(target, root):
            label = state.canvas.get("project_name") or root
            return (f"{label} is the project the GUI has open; open a different project first, "
                    "or release it as the open project")
    return None


def identity_conflict(target: Path, state: OpenProjectState) -> Optional[str]:
    """The three spellings of "the open project" (:func:`_open_project_conflict`): the cheap
    identity checks in :func:`_ordered_refusal`'s own chain, none of them the live-run or
    job-registry scans that open a database connection. With no project bound and the marker
    and the canvas binding both silent, every project answers ``None`` here: the request still
    proceeds, its own line landing in the target's own log rather than a bound project's
    (:func:`request_project_removal`). Shared verbatim by the door's own refusal chain and by
    the workspace listing's own per-project ``removal_refusal``, so the two answer with the same
    string rather than each composing its own. ``state`` is the marker/binding/canvas triple
    every caller reads once (:func:`read_open_project_state`) and passes in, per project when
    checking several, so no call pays the reads on its own."""
    return _open_project_conflict(target, state)


def _live_run_conflict(target: Path) -> Optional[str]:
    from tcip_mcp import experiments
    from tcip_mcp.tools.training_tools import TCIP_HEARTBEAT_STALE_SECONDS

    for exp_id in experiments.experiment_ids_with_status(root=target):
        status = experiments.read_member(experiments.status_key(exp_id, root=target), {})
        if experiments.derived_state(status, TCIP_HEARTBEAT_STALE_SECONDS) == "running":
            return (f"experiment {exp_id!r} is running; ask the agent to cancel it "
                     "(cancel_training) or wait for it to finish")
    return None


def _name_shape_refusal(name: str) -> Optional[_Refusal]:
    """The three syntactic checks that open :func:`_ordered_refusal`'s own chain: an invalid
    name, one carrying leading or trailing whitespace, or the workspace's own holding-directory
    name. Also called by :func:`request_project_removal` ahead of its own confirm-name mismatch
    check, so a malformed name answers with its own text rather than the mismatch's, whatever
    ``confirm_name`` carries; cheap enough to run twice rather than thread a result across that
    boundary."""
    if not workspace.is_valid_name(name):
        return _Refusal(400, f"invalid project name: {name!r}")
    if name != name.strip():
        return _Refusal(400, f"{name!r} carries leading or trailing whitespace")
    if name == REMOVED_DIRNAME:
        return _Refusal(400, f"{REMOVED_DIRNAME!r} is the workspace's own holding directory, "
                              "not a project")
    return None


def _ordered_refusal(
    name: str, job_conflict: JobConflict, state: OpenProjectState,
) -> Optional[_Refusal]:
    """Decision order: name shape (:func:`_name_shape_refusal`), existence, a link, the marker,
    the two cheap identity checks (:func:`identity_conflict`), a live run, a non-terminal job.
    The first refusal wins; ``None`` means the removal may proceed. This chain runs once per
    request either way (:func:`_preview`), never once for the preview and again for the door.
    ``job_conflict`` is the caller's own function of the target root answering the refusal text
    or ``None``, the one edge into tcip-web's job registries this module holds no import of.
    ``state`` is :func:`_preview`'s own single :func:`read_open_project_state` call, passed
    through to :func:`identity_conflict` rather than read again here."""
    shape_refusal = _name_shape_refusal(name)
    if shape_refusal is not None:
        return shape_refusal

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

    conflict = identity_conflict(project, state)
    if conflict is not None:
        return _Refusal(409, conflict)

    live_run = _live_run_conflict(project)
    if live_run is not None:
        return _Refusal(409, live_run)

    job_result = job_conflict(project)
    if job_result is not None:
        return _Refusal(409, job_result)

    return None


def _preview(
    name: str, job_conflict: JobConflict, state: OpenProjectState,
) -> tuple[dict, Optional[_Refusal]]:
    """The shared implementation behind :func:`removal_preview` and :func:`request_project_removal`:
    runs :func:`_ordered_refusal` exactly once, returning the dialog's own dict beside the full
    refusal (status and text), so the door answers with the status the chain actually found
    rather than re-deriving it, and the chain never runs a second time for the same request. A
    refusal returns at once, with empty ``external_roots``/``dependent_projects``, before the
    external-root/dependent scan below ever walks the sibling registries: a refused request has
    nothing for the dialog to act on from that scan, so it never runs one. ``state`` is the
    caller's own single :func:`read_open_project_state` read (:func:`removal_preview` takes its
    own; :func:`request_project_removal` reuses the read it needs for the route's own line), so
    this function never reads the marker, the binding or the canvas record a second time.

    ``external_roots`` is every path ``tcip_mcp.store_catalogue.project_roots`` names for this
    project that is not under it, listed once per path with the layouts it serves and whether the
    path still exists on disk (``present``); a target registry that will not read carries
    ``external_roots_unreadable`` (the message) instead of an empty list, since the two mean
    different things to the dialog. ``dependent_projects`` is every other workspace project's
    dataset registry entry that resolves under this project's tree, pending ones included and
    marked ``pending``; a project whose own registry will not read is listed as
    ``{project, unreadable}`` rather than dropped, the same treatment a matching entry that
    carries a path but no id gets, naming the path rather than a null id.

    Reads the target's experiment records (the live-run refusal check), which opens its database
    on this request's thread and keeps it open for the process's life: previewing and cancelling
    still leaves this backend holding the target until it restarts.
    """
    from tcip_mcp.registry_paths import nearest_containing_ancestor
    from tcip_mcp.store_catalogue import project_roots
    from tcip_mcp.tools.project_tools import dataset_entry_path, read_datasets_raw

    refusal = _ordered_refusal(name, job_conflict, state)
    if refusal is not None:
        return {"external_roots": [], "dependent_projects": [], "refusal": refusal.message}, refusal

    try:
        project = workspace.workspace_project_root(name)
    except ValueError as exc:
        # The chain above just resolved this name; a ValueError here means it vanished in the
        # narrow window since, which the chain has not seen yet.
        return {"external_roots": [], "dependent_projects": [], "refusal": str(exc)}, _Refusal(404, str(exc))

    roots_by_path: dict[str, list[str]] = {}
    external_roots_unreadable: Optional[str] = None
    try:
        target_roots = project_roots(project)
    except Exception as exc:  # noqa: BLE001 - named for the dialog, never crashes the preview
        target_roots = ()
        external_roots_unreadable = str(exc)
    for root_path, layout in target_roots:
        if nearest_containing_ancestor(Path(root_path), project, tolerant=True) is None:
            roots_by_path.setdefault(root_path, []).append(layout)
    external_roots = [
        {"path": path, "layouts": sorted(layouts), "present": Path(path).exists()}
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
            if nearest_containing_ancestor(entry_path, project, tolerant=True) is None:
                continue
            dataset_id = entry.get("id")
            if not dataset_id:
                dependent_projects.append({
                    "project": child.name,
                    "unreadable": f"registry entry {entry_path} carries no id",
                })
                continue
            dependent_projects.append({
                "project": child.name, "dataset_id": dataset_id,
                "dataset_path": str(entry_path), "pending": pending is not None,
            })

    result: dict = {
        "external_roots": external_roots,
        "dependent_projects": dependent_projects,
        "refusal": None,
    }
    if external_roots_unreadable is not None:
        result["external_roots_unreadable"] = external_roots_unreadable
    return result, None


def removal_preview(name: str, *, job_conflict: JobConflict) -> dict:
    """Every fact the removal dialog needs before a name is even typed (see :func:`_preview`).
    ``job_conflict`` is the caller's own function of the target root answering a non-terminal
    job's refusal text, or ``None``."""
    result, _ = _preview(name, job_conflict, read_open_project_state())
    return result


def _bound_project_root(state: OpenProjectState) -> Optional[Path]:
    """The bound-root spelling of "the open project", or ``None`` when this process is bound to
    no workspace project at all: :func:`workspace.workspace_project_name` on ``state``'s own
    binding, which also answers ``None`` for a bound root outside the workspace (a registered
    dataset root, a ``TCIP_IMAGE_ROOTS`` entry). The one place :func:`request_project_removal`
    decides where its own line goes, so a caller reading the same ``state`` twice never derives
    two different answers to "is a project bound"."""
    if state.binding_root is None:
        return None
    if workspace.workspace_project_name(state.binding_root) is None:
        return None
    return state.binding_root


def request_project_removal(
    name: str, confirm_name: str, *, requested_by: str, job_conflict: JobConflict,
) -> dict:
    """Phase one: archive the named workspace project, then mark it pending removal.

    The name-shape checks that open :func:`_ordered_refusal`'s own chain
    (:func:`_name_shape_refusal`) run first, ahead of the confirm-name mismatch check, since only
    this door takes a ``confirm_name`` of its own and a malformed name answers with its own text
    whatever ``confirm_name`` carries. Once both pass, one preview (:func:`_preview`) computes
    the rest of the ordered refusal chain and the external-root/dependent scan exactly once, and
    this door answers with its ``refusal`` rather than running the chain a second time. On
    success: the archive is under the workspace's holding directory, the marker is on the
    project, and this response carries ``{name, archive_path, holding_dir, external_roots,
    dependent_projects, completes, audit_scope, recorded_in_open_project, audit_note}``.

    The removed project's own last line, naming the marker just written, lands in the target's
    own log first. Next, one ``dependency_pending_removal`` line per dependent project (the
    preview's own ``dependent_projects`` entries carrying a ``dataset_id``, grouped by
    ``project``), naming every dataset id and path a dependent registers under the target, in
    that dependent's own log; an unreadable dependent gets none. Last, the route's own line about
    the request, in the bound project's own log when this process is bound to one
    (``recorded_in_open_project`` true), or in the target's own log otherwise, so a workspace with
    nothing bound still records the whole request against the project it names. ``audit_scope``
    is that root either way; ``audit_note`` names, in one sentence, where the request's two lines
    and the archive door's own line went (the archive door's own line is a bare ``@audited`` door,
    unmoved by this decision: it always lands wherever ``$TCIP_STATE_ROOT`` names). A dependent's
    line failing to append never costs another dependent its own line, or the route its own
    attempt: every failure is collected and the loop continues, and one 409 at the end names each
    line that did not land beside the ones that did, stating that the marker and the target's own
    line stand and that the removal still completes at the next start; the target's own line keeps
    its immediate 409, unlike every line after it. See the module docstring for the four crash
    windows a request can leave. Nothing is deleted. ``requested_by`` is the caller's own resolved
    identity string (``tcip_web.identity.user_id(tcip_web.identity.resolve_user(...))``, the
    caller's own edge into that package, not this module's).
    """
    with _request_lock:
        shape_refusal = _name_shape_refusal(name)
        if shape_refusal is not None:
            return {"error": shape_refusal.message, "status": shape_refusal.status}
        if confirm_name != name:
            return {"error": "the typed name does not match the project's name", "status": 400}

        state = read_open_project_state()
        preview, refusal = _preview(name, job_conflict, state)
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

        unwritten: list[str] = []
        ws_root = workspace.workspace_root(create=False)
        by_dependent: dict[str, dict] = {}
        for dep in dependent_projects:
            if "unreadable" in dep:
                continue
            bucket = by_dependent.setdefault(dep["project"], {"dataset_ids": [], "dataset_paths": []})
            bucket["dataset_ids"].append(dep["dataset_id"])
            bucket["dataset_paths"].append(dep["dataset_path"])
        for dep_name, bucket in by_dependent.items():
            try:
                audit.record_event_or_raise(
                    "dependency_pending_removal",
                    {"dataset_ids": bucket["dataset_ids"], "dataset_paths": bucket["dataset_paths"],
                     "target": name, "target_root": str(project), "archive_path": str(archive_path),
                     "holding_dir": str(holding_dir), "requested_by": requested_by},
                    scope=ws_root / dep_name,
                )
            except audit.AuditEntryNotWritten as exc:
                unwritten.append(f"{dep_name}'s own line: {exc}")

        bound_root = _bound_project_root(state)
        recorded_in_open_project = bound_root is not None
        open_root = bound_root if bound_root is not None else project
        try:
            audit.record_event_or_raise(
                "gui_project_removal_requested",
                {"name": name, "archive_path": str(archive_path), "holding_dir": str(holding_dir),
                 "external_roots": external_roots, "dependent_projects": dependent_projects},
                scope=open_root,
            )
        except audit.AuditEntryNotWritten as exc:
            unwritten.append(f"the route's own line: {exc}")

        if unwritten:
            return {"error": f"the marker and {name!r}'s own line stand; every other line was "
                              f"written except: {'; '.join(unwritten)}. The removal still "
                              "completes at the next backend start.",
                    "status": 409}

        if recorded_in_open_project:
            bound_name = workspace.workspace_project_name(bound_root)
            audit_note = (
                f"{name!r}'s own line is in its own log; the route's own line and the archive "
                f"door's own line are in {bound_name!r}'s log."
            )
        else:
            audit_note = (
                f"this backend has no project bound, so {name!r}'s own line and the route's own "
                "line are both in its own log; the archive door's own line is under the root "
                "$TCIP_STATE_ROOT names."
            )

        return {
            "name": name,
            "archive_path": str(archive_path),
            "holding_dir": str(holding_dir),
            "external_roots": external_roots,
            "dependent_projects": dependent_projects,
            "completes": "at the next backend start, or tcip complete-removals",
            "audit_scope": str(open_root),
            "recorded_in_open_project": recorded_in_open_project,
            "audit_note": audit_note,
        }


def complete_pending_removals(workspace_root: Path) -> list[dict]:
    """Phase two: move every workspace project carrying a pending-removal marker onto its own
    holding directory. Runs once per process, before any other store is opened in intent (each
    project's own store is opened on this thread to read its marker, and closed before that
    project's rename); a per-project store refusal is skipped rather than stopping the walk.

    Each outcome is ``{name, moved_to, archive_path}`` on a completed move, ``{name, blocked_by,
    blocked_errno, archive_path}`` when the rename was denied past :data:`RENAME_BUDGET_S` or
    crosses a filesystem boundary (the marker stays for the next start to retry; ``blocked_by``
    stays the ``OSError``'s own text verbatim for the agent, and ``blocked_errno`` is its
    ``errno``, ``None`` when the error carried none, so a reader can tell a held handle
    (``EACCES``/``EPERM``) from a filesystem boundary (``EXDEV``) without parsing the text), or
    ``{name, skipped, archive_path: None}`` when the marker itself could not be read (a loose
    layout, an undecodable or over-version document): that project is left exactly as it was.
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
            exc_errno = getattr(exc, "errno", None)
            reason = (f"{holding_dir} is on another filesystem: {exc}"
                      if exc_errno == errno.EXDEV else str(exc))
            outcomes.append({"name": name, "blocked_by": reason, "blocked_errno": exc_errno,
                              "archive_path": archive_path})
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
