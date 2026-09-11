"""Project removal: archive now, mark for removal, move at the next backend start.

Three doors, GUI-only (the picker's "Remove..." dialog; the agent has no MCP tool for any of
this): :func:`request_project_removal` (phase one, run from the removal
route), :func:`complete_pending_removals` (phase two, run once at backend startup, after
``bind_startup_root``'s early return, or through ``tcip complete-removals``), and
:func:`release_project_binding` (clears the marker and/or marks the canvas-open binding
released for a project that refuses removal only because it opens by default or the GUI has it
open, with no archiving or marking of its own). Phase one archives the project
(``tcip_mcp.tools.project_tools.archive_project``, models included) into the workspace's
``.removed/`` holding directory, then writes a ``pending_removal`` marker on the project itself;
every workspace opener and walker that already exists refuses or skips a marked project from
that moment (``tcip_mcp.workspace.adoptable_project_root``, ``tcip_web.paths.allowed_roots``'s
excluded roots). Phase one leaves the project's directory exactly where it was: nothing is
deleted, and the archive round-trips through ``tcip import-project``. Phase two walks the
workspace and renames each marked project's directory onto its own holding path, deleting the
marker only once the rename and its own completion line have landed.

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
project's tree until phase two moves it: only the workspace adopters and the GUI are restricted
from opening a marked project, not every path-addressed door. A write the agent's MCP process
makes into the target
(``run_inference``, which leaves no experiment status and no backend job) is invisible to every
refusal here; on Windows under the database backend its held connection denies phase two's
rename and is reported; on POSIX or under the file backend the move lands under that writer.

Importing this module pulls in nothing that imports ``tcip_mcp.server``: ``archive_project``,
``read_datasets_raw`` and ``dataset_entry_path`` are imported inside
:func:`request_project_removal`'s and :func:`removal_preview`'s own bodies, so the MCP server's
tool registration reaches a web backend process on the first preview or removal request, never
at startup. This module also imports nothing from ``tcip_web``: the resolved ``requested_by``
identity and the ``job_conflict`` callable that walks the three job registries are the caller's
own to supply (:mod:`tcip_web.routes.projects`, the route module's own caller for both this
door and ``tcip_mcp.project_rename``'s), rather than edges this module holds into a package one
layer above it.

A fourth door lives in a sibling module, ``tcip_mcp.project_rename``: the project-rename door,
which shares :data:`_request_lock`, calls :func:`_ordered_refusal` (naming itself with
``door="rename"``) so the two doors' shared refusals are read from one implementation, and calls
:func:`dependent_projects_of`, :func:`read_open_project_state`, :func:`identity_conflict` and
:func:`binding_release_available` rather than re-deriving any of them. This module never imports
that one back: the dependency runs one way.
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
"""Serializes both :func:`request_project_removal` and
``tcip_mcp.project_rename.request_project_rename`` within one process: whichever of a removal or
a rename request enters second meets the first one's freshly written marker rather than racing
its own archive or its own write. A ``threading.Lock``, so it holds only within this process: two
backends bound to the same workspace can still each pass this lock in their own process and each
write their own marker (a removal marker and a rename marker on the same project, from two
processes); nothing in this platform prevents two backends from sharing one workspace, and that
residual is carried rather than hidden. Within one process it is exact: a crash inside either
door's own chain writes at most one marker, since each write is the chain's last step."""

_startup_outcomes: list[dict] = []
_startup_outcomes_lock = threading.Lock()


@dataclass(frozen=True)
class _Refusal:
    status: int
    message: str


def is_link_or_junction(path: Path) -> bool:
    """A symbolic link (POSIX) or a Windows junction/reparse point.

    Either would let the archive follow the link's target while a rename moves the link entry
    alone, leaving the two out of step, so the door refuses rather than acting on it. Public
    because both the removal and the rename door's shared refusal chain (:func:`_ordered_refusal`)
    call it.
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
    spelling and its own remedy: the marker and canvas messages name a breeder-triggered release
    (:func:`release_project_binding`); the bound-root message, which no release clears, names a
    restart instead. The same strings :func:`identity_conflict` answers the listing's own
    per-project reason with.
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
                "and start it without a state root naming this project")

    from tcip_mcp.web_client import binding_released_or_absent

    if state.canvas_problem is not None:
        return f"the GUI's open-project binding could not be read: {state.canvas_problem}"
    canvas = state.canvas
    if canvas is not None and not binding_released_or_absent(canvas) and (
        canvas.get("project_name") or canvas.get("root")
    ):
        root = canvas.get("root")
        if root and _same_path(target, root):
            label = canvas.get("project_name") or root
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
    checking several, so no call pays the reads on its own. An unreadable canvas record
    (``state.canvas_problem``) is the one remaining state in which every project refuses,
    fail-closed: with no project bound and the marker silent, that unreadable record still
    answers every target rather than letting a request through on a binding this process
    could not confirm."""
    return _open_project_conflict(target, state)


def _workspace_child_of(path: Path, workspace_root: Path) -> Optional[Path]:
    """The workspace's own top-level child ``path`` sits under, one level below
    ``workspace_root`` on the ancestor chain :func:`~tcip_mcp.registry_paths.
    nearest_containing_ancestor` walks (``tolerant=True``, so a missing leaf still resolves
    through parents that do exist): ``None`` for a path outside the workspace (no such ancestor),
    for the workspace root itself (checked by :func:`_same_path` up front, so a path equal to the
    workspace never has to be indexed), and for a path resolving into the workspace's own holding
    directory (:data:`REMOVED_DIRNAME`, never a project). Every other path answers the directory
    name immediately below the workspace root, whether or not that directory, or ``path`` itself,
    still exists on disk. Shared by :func:`dependency_warnings` and by :func:`_preview`'s own
    dependent scan, so the scan that lists a dependent before a request and the card that warns
    about it afterward agree on one containment rule; the one case they can still differ on is a
    hand-edited registry entry whose containment is visible only through a link the stored path
    preserves, which :func:`~tcip_mcp.tools.project_tools.registry_path_for`'s own resolved
    storage never produces.
    """
    from tcip_mcp.registry_paths import nearest_containing_ancestor

    if _same_path(path, workspace_root):
        return None
    chain = [path, *path.parents]
    anchor = nearest_containing_ancestor(path, workspace_root, tolerant=True)
    if anchor is None:
        return None
    idx = chain.index(anchor)
    child = chain[idx - 1]
    if child.name == REMOVED_DIRNAME:
        return None
    return child


def dependency_warnings(project_root: Path) -> tuple[list[dict], Optional[str]]:
    """Every warning ``project_root``'s own registry earns from a dataset it registered under
    another workspace project that is now pending removal, pending rename, or gone, plus the
    registry's own problem when it will not read at all or carries an entry with a path and no
    id. ``pending_kind`` (``"removal"`` or ``"rename"``, absent when the target is gone rather
    than pending) is set from :func:`~tcip_mcp.workspace.pending_marker_or_none` so a dependent's
    card can render the sentence for whichever kind is under way, never the removal sentence for
    a target that is only being renamed.

    Reads through :func:`~tcip_mcp.tools.project_tools.read_datasets_raw`; a registry that will
    not read (``StoreError``, ``DecodeError``, ``SchemaVersionRefused``, or a fingerprint
    ``ValueError``) answers ``([], <message>)`` rather than raising, so one damaged registry costs
    the listing one project's warnings, never the whole page. Each entry carrying a ``path`` is
    resolved through :func:`~tcip_mcp.tools.project_tools.dataset_entry_path` and named to its
    workspace child through :func:`_workspace_child_of`; an entry ``_workspace_child_of`` answers
    ``None`` for (outside the workspace, the holding directory) and an entry whose child is
    ``project_root`` itself (a project's own dataset registered under its own tree, compared by
    :func:`_same_path`) earn no warning, since neither names a dependency on another project. An
    entry that does name a dependency but carries no ``id`` is malformed, the same treatment
    :func:`_preview`'s own dependent scan gives it: it earns the problem naming the entry rather
    than a warning with a null id; the first such entry names the problem, and every other
    entry is still considered.
    The condition is the entry itself, never the holding directory an eventual move leaves
    behind: a child directory that no longer exists on disk is a warning with ``present`` false,
    ``pending_kind`` absent, and ``archive_path``/``holding_dir`` both null, whatever
    ``.removed/`` holds, since the holding directory is not state this platform owns and a
    project gone by any means leaves the entry dangling the same way; a child directory that
    still exists, still carries ``.tcip``, and carries either marker
    (:func:`~tcip_mcp.workspace.pending_marker_or_none`) is a warning with ``present`` true,
    ``pending_kind`` naming which marker, and ``archive_path``/``holding_dir`` from the marker
    record when it is a removal (both null for a rename, which archives nothing), so an agent
    reading the warning can answer "from where" without the dependent's own log; every other
    entry earns none. The warning's own lifetime is bounded by what the
    registry can express: no door removes an entry, and ``register_dataset`` refreshes an
    entry's own path by id, so a
    warning stands while the entry's path resolves under a pending or absent workspace child and
    clears once the entry is refreshed to a path that resolves elsewhere, or the child is present
    again; a project re-created under the removed one's own name resolves the entry into an
    unrelated tree and clears the warning too, which the dataset-identity rail (the fingerprint
    the entry carries against the data actually there) catches at training time and this function
    does not. Each warning is ``{dataset_id, dataset_path, target, present, archive_path,
    holding_dir, pending_kind}``, ``target`` the dependency's own workspace-child name.
    """
    from tcip_mcp.tools.project_tools import dataset_entry_path, read_datasets_raw

    ws = workspace.workspace_root(create=False)
    try:
        entries = read_datasets_raw(project_root)
    except (tcip_store.StoreError, tcip_store.DecodeError,
            tcip_store.SchemaVersionRefused, ValueError) as exc:
        return [], f"its dataset registry could not be read: {exc}"

    warnings: list[dict] = []
    problem: Optional[str] = None
    for entry in entries:
        if not entry.get("path"):
            continue
        try:
            entry_path = dataset_entry_path(project_root, entry)
        except ValueError:
            continue
        child = _workspace_child_of(entry_path, ws)
        if child is None or _same_path(child, project_root):
            continue
        dataset_id = entry.get("id")
        if not dataset_id:
            if problem is None:
                problem = f"its dataset registry has an entry with a path and no id: {entry_path}"
            continue
        if not child.exists():
            warnings.append({"dataset_id": dataset_id, "dataset_path": str(entry_path),
                              "target": child.name, "present": False,
                              "archive_path": None, "holding_dir": None})
            continue
        if not (child / ".tcip").is_dir():
            continue
        pending = workspace.pending_marker_or_none(child)
        if pending is not None:
            archive_path = pending.record.get("archive_path") if pending.kind == "removal" else None
            holding_dir = pending.record.get("holding_dir") if pending.kind == "removal" else None
            new_name = pending.record.get("new_name") if pending.kind == "rename" else None
            warnings.append({"dataset_id": dataset_id, "dataset_path": str(entry_path),
                              "target": child.name, "present": True,
                              "archive_path": archive_path, "holding_dir": holding_dir,
                              "pending_kind": pending.kind, "new_name": new_name})
    return warnings, problem


def binding_release_available(target: Optional[Path], state: OpenProjectState) -> bool:
    """Whether :func:`release_project_binding` would have something to clear for ``target``:
    true when the marker or the canvas binding (not itself already released) names it by
    identity, the same two checks :func:`_open_project_conflict` runs for its own marker and
    canvas spellings, so a control this answers ``True`` for stays enabled beside a refusal a
    release would clear. ``target`` may be ``None`` (a name that resolved to nothing safely
    addressable), which never matches. Shared by :func:`_preview`'s own ``releasable`` field and
    the workspace listing's own per-project ``removal_releasable``
    (``tcip_web.routes.projects._summarize``), the same split :func:`identity_conflict` and
    ``removal_refusal`` already keep."""
    from tcip_mcp.web_client import binding_released_or_absent

    if target is None:
        return False
    if state.marker_name:
        try:
            marker_root = workspace.project_path(state.marker_name, create=False)
        except ValueError:
            marker_root = None
        if marker_root is not None and _same_path(target, marker_root):
            return True
    canvas = state.canvas
    if canvas is not None and not binding_released_or_absent(canvas):
        root = canvas.get("root")
        if root and _same_path(target, root):
            return True
    return False


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
    name: str, job_conflict: JobConflict, state: OpenProjectState, *, door: str = "removal",
) -> Optional[_Refusal]:
    """Decision order: name shape (:func:`_name_shape_refusal`), existence, a link, both pending
    markers, the two cheap identity checks (:func:`identity_conflict`), a live run, a
    non-terminal job. The first refusal wins; ``None`` means the request may proceed. This chain
    runs once per request either way (:func:`_preview`, ``tcip_mcp.project_rename.rename_preview``),
    never once for a preview and again for the door it backs.

    ``door`` (``"removal"`` or ``"rename"``) names which door is asking, read only by the two
    pending-marker checks: a project already carrying a marker of the asking door's own kind
    answers with that door's own "already pending" wording, while a project carrying the other
    door's marker answers with a refusal naming what that marker completes into and, for a
    removal asking about a project pending rename, that the renamed project can be removed once
    the rename lands. Sharing this one chain is why the two doors' refusals cannot silently
    drift apart (CLAUDE.md: when two code paths must agree, call one from the other).

    ``job_conflict`` is the caller's own function of the target root answering the refusal text
    or ``None``, the one edge into tcip-web's job registries this module holds no import of.
    ``state`` is the caller's own single :func:`read_open_project_state` call, passed through to
    :func:`identity_conflict` rather than read again here."""
    shape_refusal = _name_shape_refusal(name)
    if shape_refusal is not None:
        return shape_refusal

    try:
        project = workspace.workspace_project_root(name)
    except ValueError as exc:
        return _Refusal(404, str(exc))

    if is_link_or_junction(project):
        return _Refusal(409, f"{name!r} is a symbolic link or a junction; the archive would "
                              "follow it while a rename would move the link alone, so remove it "
                              "by hand instead")

    try:
        removal_pending = workspace.pending_removal_record(project)
    except (tcip_store.StoreError, tcip_store.DecodeError, tcip_store.SchemaVersionRefused) as exc:
        return _Refusal(409, f"{name}'s pending-removal marker could not be read: {exc}")
    if removal_pending is not None:
        if door == "removal":
            return _Refusal(
                409,
                f"{name!r} already has a pending-removal marker (requested "
                f"{removal_pending['requested_at']}); it moves to the workspace's holding "
                "directory at the next backend start, or through tcip complete-removals",
            )
        return _Refusal(
            409,
            f"{name!r} is pending removal (requested {removal_pending['requested_at']}); it "
            "moves to the workspace's holding directory at the next backend start, or through "
            "tcip complete-removals",
        )

    try:
        rename_pending = workspace.pending_rename_record(project)
    except (tcip_store.StoreError, tcip_store.DecodeError, tcip_store.SchemaVersionRefused) as exc:
        return _Refusal(409, f"{name}'s pending-rename marker could not be read: {exc}")
    if rename_pending is not None:
        if door == "rename":
            return _Refusal(
                409,
                f"{name!r} already has a pending-rename marker (requested "
                f"{rename_pending['requested_at']}); it renames at the next backend start, or "
                "through tcip complete-renames",
            )
        return _Refusal(
            409,
            f"{name!r} is pending rename to {rename_pending['new_name']!r} (requested "
            f"{rename_pending['requested_at']}); it renames at the next backend start, or "
            "through tcip complete-renames, and the renamed project can be removed then",
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


def dependent_projects_of(project: Path, ws: Path) -> list[dict]:
    """Every other workspace project whose dataset registry names an entry resolving under
    ``project``'s own tree, pending ones included and marked ``pending``.

    Extracted from :func:`_preview`'s own dependent scan so the removal door's preview and
    ``tcip_mcp.project_rename``'s own door (its preview and its request) list dependents through
    one implementation rather than two that could silently drift (CLAUDE.md). A dependent project
    whose own registry will not read is listed as ``{project, unreadable}``, the same treatment a
    matching entry that carries a path but no id gets, naming the path rather than a null id.
    Containment is decided through :func:`_workspace_child_of`, the same helper
    :func:`dependency_warnings` calls, so the scan that lists a dependent here and the card that
    warns about it afterward agree on one rule. ``pending`` is true for a dependent carrying
    either marker (:func:`~tcip_mcp.workspace.pending_marker_or_none`), removal or rename alike:
    this scan is read-only and names no consequence of which kind it is, unlike
    :func:`dependency_warnings`'own ``pending_kind``.
    """
    from tcip_mcp.tools.project_tools import dataset_entry_path, read_datasets_raw

    dependent_projects: list[dict] = []
    for child in sorted(p for p in ws.iterdir() if p.is_dir()):
        if _same_path(child, project) or not (child / ".tcip").is_dir():
            continue
        pending = workspace.pending_marker_or_none(child)
        try:
            entries = read_datasets_raw(child)
        except Exception as exc:  # noqa: BLE001 - reported per project, never aborts the scan
            dependent_projects.append({
                "project": child.name,
                "unreadable": f"its dataset registry could not be read: {exc}",
            })
            continue
        for entry in entries:
            if not entry.get("path"):
                continue
            try:
                entry_path = dataset_entry_path(child, entry)
            except ValueError:
                continue
            entry_child = _workspace_child_of(entry_path, ws)
            if entry_child is None or not _same_path(entry_child, project):
                continue
            dataset_id = entry.get("id")
            if not dataset_id:
                dependent_projects.append({
                    "project": child.name,
                    "unreadable": f"its dataset registry has an entry with a path and no id: "
                                   f"{entry_path}",
                })
                continue
            dependent_projects.append({
                "project": child.name, "dataset_id": dataset_id,
                "dataset_path": str(entry_path), "pending": pending is not None,
            })
    return dependent_projects


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
    carries a path but no id gets, naming the path rather than a null id. Containment is decided
    through :func:`_workspace_child_of`, the same helper :func:`dependency_warnings` calls, so the
    scan that lists a dependent here and the card that warns about it afterward agree on one rule.

    Reads the target's experiment records (the live-run refusal check), which opens its database
    on this request's thread and keeps it open for the process's life: previewing and cancelling
    still leaves this backend holding the target until it restarts. ``releasable`` is
    :func:`binding_release_available` against whatever name resolves, computed before the refusal
    chain runs so a refused preview still tells the dialog whether a release would help.
    """
    from tcip_mcp.registry_paths import nearest_containing_ancestor
    from tcip_mcp.store_catalogue import project_roots

    try:
        candidate: Optional[Path] = workspace.project_path(name, create=False)
    except ValueError:
        candidate = None
    releasable = binding_release_available(candidate, state)

    refusal = _ordered_refusal(name, job_conflict, state)
    if refusal is not None:
        return {"external_roots": [], "dependent_projects": [], "refusal": refusal.message,
                "releasable": releasable}, refusal

    try:
        project = workspace.workspace_project_root(name)
    except ValueError as exc:
        # The chain above just resolved this name; a ValueError here means it vanished in the
        # narrow window since, which the chain has not seen yet.
        return {"external_roots": [], "dependent_projects": [], "refusal": str(exc),
                "releasable": releasable}, _Refusal(404, str(exc))

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

    ws = workspace.workspace_root(create=False)
    dependent_projects = dependent_projects_of(project, ws)

    result: dict = {
        "external_roots": external_roots,
        "dependent_projects": dependent_projects,
        "refusal": None,
        "releasable": releasable,
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
    and the archive door's own line went (the archive door's own line is a bare ``@audited`` door
    that always lands wherever ``$TCIP_STATE_ROOT`` names, whatever project this request
    concerns). A dependent's
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
            return {"error": f"the marker and {name}'s own line stand; every other line was "
                              f"written except: {'; '.join(unwritten)}. The removal still "
                              "completes at the next backend start.",
                    "status": 409}

        if bound_root is not None:
            bound_name = workspace.workspace_project_name(bound_root)
            audit_note = (
                f"{name}'s own line is in its own log; the route's own line and the archive "
                f"door's own line are in {bound_name}'s log."
            )
        else:
            from tcip_mcp.audit import platform_audit_scope

            audit_note = (
                f"this backend has no project open, so {name}'s own line and the route's own "
                f"line are both in its own log; the archive door's own line is under "
                f"{str(platform_audit_scope())!r}."
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


def release_project_binding(name: str, *, released_by: str) -> dict:
    """Stop ``name`` opening by default and forget it as the GUI's own open project, so the
    breeder can clear a marker or canvas-binding refusal a removal request meets without a
    restart, and try the request again once the bound-root spelling (if that is what remains) has
    one.

    ``name`` must name an existing workspace project (404 on a ``ValueError``, the same shape
    :func:`~tcip_mcp.workspace.workspace_project_root` raises). Each binding is read back with its
    own version and compared to the project by identity right before it is changed, never on the
    strength of an earlier read, so a caller cannot delete a marker or bump a binding that moved to
    another project in between.

    The marker: read through :func:`~tcip_mcp.workspace.active_project_key`, cleared with
    ``delete(key, expect=version)`` only when it names ``name`` by identity; a ``VersionConflict``
    (``activate_project`` landing between the read and the delete) is answered 409 naming it, with
    nothing cleared and no line. An unreadable marker (``OSError``, ``DecodeError``) folds to "no
    marker" the way its sibling reader (:func:`~tcip_mcp.workspace.read_active_project`) folds
    it, so a marker this door cannot read never 500s the release of a project the canvas binding
    alone names; nothing is cleared for it either way. The canvas-open binding is never deleted,
    so a caller still
    holding its generation keeps failing its own fence and its own push exactly as it would
    against any other stale generation: inside one transaction on
    :func:`~tcip_mcp.web_client.canvas_open_binding_key`, the record is read, and when its
    ``root`` names ``name`` by identity it is rewritten with ``generation + 1``, ``released: True``
    and a fresh ``issued_at``, ``root`` and ``project_name`` kept. Any ``tcip_store.StoreError``
    the transaction raises reading or writing the record (``StoreBusy``, ``VersionConflict``, a
    ``DecodeError`` or ``SchemaVersionRefused`` on a damaged record, or a record naming ``name``
    that carries no ``generation`` field) is caught and answered after the marker's own outcome
    is settled, never silently.

    When either binding actually changed, ``project_binding_released`` is recorded under the
    project's own root before any answer, naming ``released_by``, ``marker_cleared`` and
    ``canvas_binding_released``; a canvas failure after the marker was cleared still gets that
    line (``marker_cleared`` true, ``canvas_binding_released`` false) before the 409 that names
    both what was cleared and recorded and what was not, and a failed append itself answers 409
    naming what was cleared and that the line was not written. Neither binding naming the project
    writes nothing and answers both flags false with no line. This door never repins this
    process's own platform-state root and never touches :class:`~tcip_web.state.StateStore`'s
    in-memory open project: the platform root stays bound until the process restarts (the
    bound-root refusal says so), and the Results doors keep serving the last root the GUI
    selected until another select, which the guard's own excluded-roots check closes the moment a
    marker lands on that root. The response is ``{name, marker_cleared, canvas_binding_released,
    refusal, releasable}``, ``refusal`` and ``releasable`` read fresh after the release through
    :func:`identity_conflict` and :func:`binding_release_available`.
    """
    from tcip_mcp.web_client import canvas_open_binding_key

    try:
        project = workspace.workspace_project_root(name)
    except ValueError as exc:
        return {"error": str(exc), "status": 404}

    marker_cleared = False
    marker_key = workspace.active_project_key()
    try:
        versioned: Optional[tcip_store.Versioned] = tcip_store.read_versioned(marker_key, default=None)
    except (OSError, tcip_store.DecodeError):
        versioned = None
    marker_value = (versioned.value or "").strip() if versioned is not None else ""
    if versioned is not None and marker_value:
        try:
            marker_root: Optional[Path] = workspace.project_path(marker_value, create=False)
        except ValueError:
            marker_root = None
        if marker_root is not None and _same_path(project, marker_root):
            try:
                tcip_store.delete(marker_key, expect=versioned.version)
                marker_cleared = True
            except tcip_store.VersionConflict as exc:
                return {"error": f"the active-project marker changed while releasing {name!r}: "
                                  f"{exc}", "status": 409}

    canvas_binding_released = False
    canvas_key = canvas_open_binding_key()
    canvas_failure: Optional[Exception] = None
    try:
        with tcip_store.transaction(canvas_key) as txn:
            current = txn.read(canvas_key, default=None)
            if current is not None:
                if not isinstance(current, dict):
                    raise tcip_store.DecodeError(
                        f"the canvas-open binding record is {type(current).__name__}, not a "
                        "mapping"
                    )
                bound_root = current.get("root")
                if bound_root is not None and not isinstance(bound_root, str):
                    raise tcip_store.DecodeError(
                        f"the canvas-open binding record's root is {type(bound_root).__name__}, "
                        "not a path"
                    )
                if bound_root is not None and _same_path(project, bound_root):
                    generation = current.get("generation")
                    if not isinstance(generation, int) or isinstance(generation, bool):
                        raise tcip_store.DecodeError(
                            f"the canvas-open binding record for {bound_root} carries no usable "
                            f"generation: {generation!r}"
                        )
                    txn.write(canvas_key, {
                        "generation": generation + 1,
                        "root": bound_root,
                        "project_name": current.get("project_name"),
                        "released": True,
                        "issued_at": datetime.now(timezone.utc).isoformat(),
                    })
                    canvas_binding_released = True
    except tcip_store.StoreError as exc:
        canvas_failure = exc

    if marker_cleared or canvas_binding_released:
        try:
            audit.record_event_or_raise(
                "project_binding_released",
                {"name": name, "released_by": released_by, "marker_cleared": marker_cleared,
                 "canvas_binding_released": canvas_binding_released},
                scope=project,
            )
        except audit.AuditEntryNotWritten as exc:
            return {"error": f"{name}'s binding was cleared (marker_cleared={marker_cleared}, "
                              f"canvas_binding_released={canvas_binding_released}) but the line "
                              f"was not written: {exc}",
                    "status": 409}

    if canvas_failure is not None:
        return {"error": f"releasing {name}'s canvas-open binding failed: {canvas_failure}. "
                          f"marker_cleared={marker_cleared}.",
                "status": 409}

    state = read_open_project_state()
    return {
        "name": name,
        "marker_cleared": marker_cleared,
        "canvas_binding_released": canvas_binding_released,
        "refusal": identity_conflict(project, state),
        "releasable": binding_release_available(project, state),
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
