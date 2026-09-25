"""Project rename: mark for rename, move onto the new name at the next backend start.

:func:`request_project_rename` (phase one) writes a ``pending_rename`` marker naming the old and
the new name; every reader of :func:`~tcip_mcp.workspace.pending_marker_or_none` refuses or skips a
marked project from that moment. :func:`complete_pending_renames` (phase two, run at backend
startup ahead of ``complete_pending_removals``, or through ``tcip complete-renames``) renames each
marked project's directory onto its own new name, deleting the marker only once the rename and its
own completion line have landed; a directory already carrying the new name writes the completion
line and deletes the marker with no rename of its own. :func:`withdraw_project_rename` clears a
pending marker with no rename, for a request whose destination name was taken before phase two ran;
nothing reserves a destination name during the pending window.

:func:`project_records_present` refuses a project that holds any experiment, plant mapping, plant
registry, delivery event, persisted job or HPO sweep, since those stores can carry the project's
own path absolutely and no door here re-points a record's own path. A dependent project's own
dataset-registry entry is warned, never refused on and never re-pointed: before the request through
the preview and after it through :func:`~tcip_mcp.project_removal.dependency_warnings`'s
``pending_kind``. The dependent's own remedy is ``register_dataset`` at the renamed path, which
keeps the id its own ``dataset.json`` carries.

A removal and a rename request against the same project serialize on
:data:`tcip_mcp.project_removal._request_lock`, and the shared refusal chain is
:func:`tcip_mcp.project_removal._ordered_refusal`, called here with ``door="rename"``.
"""

from __future__ import annotations

import errno
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import tcip_store
from tcip_store.file_backend import retry_while_denied

from tcip_mcp import audit, project_removal, workspace
from tcip_mcp.project_removal import JobConflict

_rename_startup_outcomes: list[dict] = []
_rename_startup_outcomes_lock = threading.Lock()


def project_records_present(project: Path) -> list[str]:
    """The names of ``project``'s own stores that hold any record, over the six stores whose
    records can carry the project's own path absolutely: ``experiments``, ``plant_mapping``,
    ``plant_registries``, ``delivery_events``, ``job_registry`` and ``hpo_sweep_manifest``. Returns
    ``[]`` for a project that has not yet trained, mapped or delivered.
    """
    from tcip_mcp import experiments
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        PLANT_MAPPING_STORE, PLANT_REGISTRY_STORE,
    )
    from tcip_mcp.pipelines.resolution import DELIVERY_EVENTS_STORE, delivery_events_scope
    from tcip_mcp.tools import training_tools
    from tcip_mcp.web_client import JOB_REGISTRY_DOCUMENTS, job_registry_key

    present: list[str] = []
    if experiments.experiment_ids_with_status(root=project):
        present.append("experiments")

    state_root = str(Path(project).absolute() / ".tcip" / "state")
    if tcip_store.keys(PLANT_MAPPING_STORE, state_root):
        present.append("plant_mapping")
    if tcip_store.keys(PLANT_REGISTRY_STORE, state_root):
        present.append("plant_registries")
    if tcip_store.keys(DELIVERY_EVENTS_STORE, str(delivery_events_scope(project))):
        present.append("delivery_events")

    if any(
        tcip_store.read(job_registry_key(doc, root=project), default=[])
        for doc in JOB_REGISTRY_DOCUMENTS
    ):
        present.append("job_registry")

    hpo_root = training_tools.hpo_root(root=project)
    if hpo_root.is_dir():
        for sub in sorted(p for p in hpo_root.iterdir() if p.is_dir()):
            try:
                manifest = tcip_store.read(
                    training_tools.sweep_manifest_key(sub.name, root=project), default=None,
                )
            except (tcip_store.StoreError, tcip_store.DecodeError,
                    tcip_store.SchemaVersionRefused):
                present.append("hpo_sweep_manifest")
                break
            if manifest is not None:
                present.append("hpo_sweep_manifest")
                break
    return present


_RECORD_IN_PLAIN_WORDS = {
    "experiments": "a training run",
    "plant_mapping": "a plant mapping",
    "plant_registries": "a plant location list",
    "delivery_events": "a delivered result",
    "job_registry": "a saved job",
    "hpo_sweep_manifest": "a tuning sweep",
}
"""What each bound store means to the breeder reading the refusal, since the dialog renders that
sentence verbatim and a store's own name says nothing to the person holding the mouse."""


def _records_refusal(project: Path) -> Optional[str]:
    """:func:`project_records_present`'s answer rendered as the door's own refusal text, or
    ``None`` when the project holds no such record.
    """
    records = project_records_present(project)
    if not records:
        return None
    held = ", ".join(_RECORD_IN_PLAIN_WORDS.get(r, r) for r in records)
    return (
        f"{project.name} has already been worked on, so its name is fixed: it holds {held}. "
        "Those saved records name this project by its current folder, and nothing re-points "
        "them, so renaming would leave them pointing at a folder that no longer exists. "
        "Renaming is for a project that has not yet been trained, mapped or delivered from. "
        f"The records found were in {', '.join(records)}."
    )


def rename_preview(name: str, *, job_conflict: JobConflict) -> dict:
    """Every fact the rename dialog needs before a new name is even typed: the shared refusal chain
    plus the records refusal (:func:`project_records_present`), the new-name-specific checks
    excluded.

    ``dependent_projects`` is :func:`~tcip_mcp.project_removal.dependent_projects_of`'s own answer,
    listed whether or not the request would be refused on records. ``releasable`` is
    :func:`~tcip_mcp.project_removal.binding_release_available` against whatever name resolves,
    computed before the refusal chain runs.
    """
    state = project_removal.read_open_project_state()
    try:
        candidate: Optional[Path] = workspace.project_path(name, create=False)
    except ValueError:
        candidate = None
    releasable = project_removal.binding_release_available(candidate, state)

    refusal = project_removal._ordered_refusal(name, job_conflict, state, door="rename")
    if refusal is not None:
        return {"refusal": refusal.message, "releasable": releasable,
                "dependent_projects": [], "records_present": []}

    try:
        project = workspace.workspace_project_root(name)
    except ValueError as exc:
        # The chain above just resolved this name; a ValueError here means it vanished in the
        # narrow window since, which the chain has not seen yet.
        return {"refusal": str(exc), "releasable": releasable,
                "dependent_projects": [], "records_present": []}

    records = project_records_present(project)
    ws = workspace.workspace_root(create=False)
    dependent_projects = project_removal.dependent_projects_of(project, ws)

    return {
        "refusal": _records_refusal(project) if records else None,
        "releasable": releasable,
        "dependent_projects": dependent_projects,
        "records_present": records,
    }


def request_project_rename(
    name: str, new_name: str, confirm_name: str, *, requested_by: str, job_conflict: JobConflict,
) -> dict:
    """Phase one: mark ``name`` pending rename to ``new_name``.

    The shared refusal chain (:func:`tcip_mcp.project_removal._ordered_refusal`, ``door="rename"``)
    runs first, under the shared lock; then the records refusal
    (:func:`project_records_present`); then this door's own additions, in order: ``new_name`` not
    a safe segment, carrying leading/trailing whitespace, outside ``crop_subject_phenotype``, or
    equal to ``name`` (each 400); ``confirm_name != name`` (400); the destination already taken in
    the workspace (409, naming what sits there; never an overwrite). No archive is taken: a
    rename loses nothing.

    On success: the marker is written, and this response carries ``{name, new_name, completes,
    dependent_projects, audit_scope, recorded_in_open_project, audit_note}``. The target's own
    line lands first, naming the marker just written; then one ``dependency_pending_rename`` line
    per dependent project (grouped the way ``dependency_pending_removal`` is), naming every
    dataset id and path a dependent registers under the target, in that dependent's own log; an
    unreadable dependent gets none; last, the route's own line, in the bound project's own log
    when this process is bound to one (``recorded_in_open_project`` true), or in the target's own
    log otherwise. A dependent's line failing to append never costs another dependent its own
    line, or the route its own attempt: every failure is collected and the loop continues, and one
    409 at the end names each line that did not land beside the ones that did, stating that the
    marker stands and the rename still completes at the next start; the target's own line keeps
    its immediate 409, unlike every line after it.
    """
    with project_removal._request_lock:
        shape_refusal = project_removal._name_shape_refusal(name)
        if shape_refusal is not None:
            return {"error": shape_refusal.message, "status": shape_refusal.status}
        if confirm_name != name:
            return {"error": "the typed name does not match the project's name", "status": 400}

        state = project_removal.read_open_project_state()
        refusal = project_removal._ordered_refusal(name, job_conflict, state, door="rename")
        if refusal is not None:
            return {"error": refusal.message, "status": refusal.status}

        project = workspace.workspace_project_root(name)

        records = project_records_present(project)
        if records:
            return {"error": _records_refusal(project), "status": 409}

        if not workspace.is_valid_name(new_name):
            return {"error": f"invalid project name: {new_name!r}", "status": 400}
        if new_name != new_name.strip():
            return {"error": f"{new_name!r} carries leading or trailing whitespace",
                    "status": 400}
        try:
            workspace.parse_project_name(new_name)
        except ValueError as exc:
            return {"error": str(exc), "status": 400}
        if new_name == name:
            return {"error": f"{new_name!r} is the project's own current name", "status": 400}

        ws = workspace.workspace_root(create=False)
        destination = ws / new_name
        if os.path.lexists(destination):
            return {"error": f"{new_name!r} is already taken in the workspace ({destination})",
                    "status": 409}

        dependent_projects = project_removal.dependent_projects_of(project, ws)

        requested_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        record = {
            "requested_at": requested_at, "requested_by": requested_by,
            "old_name": name, "new_name": new_name,
        }
        try:
            tcip_store.replace(
                workspace.pending_rename_key(project), record, expect=tcip_store.Version.ABSENT,
            )
        except tcip_store.VersionConflict as exc:
            return {"error": f"another request already marked {name!r} for rename: {exc}",
                    "status": 409}

        try:
            audit.record_event_or_raise(
                "project_rename_requested",
                {"name": name, "new_name": new_name, "requested_by": requested_by},
                scope=project,
            )
        except audit.AuditEntryNotWritten as exc:
            return {"error": f"the marker for {name!r} is written but its own log line is not: "
                              f"{exc}. The rename still completes at the next backend start.",
                    "status": 409}

        unwritten: list[str] = []
        by_dependent: dict[str, dict] = {}
        for dep in dependent_projects:
            if "unreadable" in dep:
                continue
            bucket = by_dependent.setdefault(
                dep["project"], {"dataset_ids": [], "dataset_paths": []},
            )
            bucket["dataset_ids"].append(dep["dataset_id"])
            bucket["dataset_paths"].append(dep["dataset_path"])
        for dep_name, bucket in by_dependent.items():
            try:
                audit.record_event_or_raise(
                    "dependency_pending_rename",
                    {"dataset_ids": bucket["dataset_ids"], "dataset_paths": bucket["dataset_paths"],
                     "target": name, "target_root": str(project), "new_name": new_name,
                     "requested_by": requested_by},
                    scope=ws / dep_name,
                )
            except audit.AuditEntryNotWritten as exc:
                unwritten.append(f"{dep_name}'s own line: {exc}")

        bound_root = project_removal._bound_project_root(state)
        recorded_in_open_project = bound_root is not None
        open_root = bound_root if bound_root is not None else project
        try:
            audit.record_event_or_raise(
                "gui_project_rename_requested",
                {"name": name, "new_name": new_name, "dependent_projects": dependent_projects},
                scope=open_root,
            )
        except audit.AuditEntryNotWritten as exc:
            unwritten.append(f"the route's own line: {exc}")

        if unwritten:
            return {"error": f"the marker for {name!r} stands; every other line was written "
                              f"except: {'; '.join(unwritten)}. The rename still completes at "
                              "the next backend start.",
                    "status": 409}

        if bound_root is not None:
            bound_name = workspace.workspace_project_name(bound_root)
            audit_note = (
                f"{name}'s own line is in its own log; the route's own line is in "
                f"{bound_name}'s log."
            )
        else:
            audit_note = (
                f"this backend has no project open, so {name}'s own line and the route's own "
                "line are both in its own log."
            )

        return {
            "name": name,
            "new_name": new_name,
            "completes": "at the next backend start, or tcip complete-renames",
            "dependent_projects": dependent_projects,
            "audit_scope": str(open_root),
            "recorded_in_open_project": recorded_in_open_project,
            "audit_note": audit_note,
        }


def withdraw_project_rename(name: str, *, requested_by: str) -> dict:
    """Clear ``name``'s pending-rename marker with no rename of its own. Nothing is moved or
    deleted but the marker, so it takes no typed name.

    Writes ``project_rename_withdrawn`` (``{name, new_name, requested_by}``, scoped to the project)
    before deleting the marker at the version just read, under the shared lock so a concurrent
    rename or removal request cannot land between the read and the delete. 404 when ``name`` names
    no workspace project or carries no pending-rename marker; 409 naming the marker read that
    changed underneath, or the line that did not write (the marker then stands).
    """
    with project_removal._request_lock:
        try:
            project = workspace.workspace_project_root(name)
        except ValueError as exc:
            return {"error": str(exc), "status": 404}

        try:
            versioned = tcip_store.read_versioned(
                workspace.pending_rename_key(project), default=None,
            )
        except (tcip_store.StoreError, tcip_store.DecodeError,
                tcip_store.SchemaVersionRefused) as exc:
            return {"error": f"{name}'s pending-rename marker could not be read: {exc}",
                    "status": 409}
        if versioned.value is None:
            return {"error": f"{name!r} carries no pending-rename marker", "status": 404}

        record = versioned.value
        new_name = record["new_name"]

        if name == new_name:
            return {"error": f"{name!r} is already at its new name: the move landed and only the "
                             "completion line and the marker are outstanding, so withdrawing "
                             "would record a withdrawal of a rename that happened. It finishes "
                             "at the next backend start, or through tcip complete-renames.",
                    "status": 409}

        try:
            tcip_store.delete(workspace.pending_rename_key(project), expect=versioned.version)
        except tcip_store.StoreError as exc:
            return {"error": f"the marker changed while withdrawing: {exc}", "status": 409}

        try:
            audit.record_event_or_raise(
                "project_rename_withdrawn",
                {"name": name, "new_name": new_name, "requested_by": requested_by},
                scope=project,
            )
        except audit.AuditEntryNotWritten as exc:
            return {"error": f"{name!r} is withdrawn and its marker is gone, but the withdraw's "
                             f"own line was not written: {exc}. The project opens again; the log "
                             "does not say why it stopped being pending.",
                    "status": 409}

        return {"withdrawn": True, "name": name, "new_name": new_name}


def complete_pending_renames(workspace_root: Path) -> list[dict]:
    """Phase two: rename every workspace project carrying a pending-rename marker onto its own new
    name. Runs once per process, before
    :func:`~tcip_mcp.project_removal.complete_pending_removals`.

    Three branches per marked project. Normal: ``os.path.lexists(<ws>/<new_name>)`` answers
    ``{name, new_name, blocked_by}`` naming what sits there (never an overwrite); else
    :func:`tcip_store.close_connections`, ``os.rename`` under
    :data:`tcip_mcp.project_removal.RENAME_BUDGET_S` (a denied or cross-device rename answers
    ``{name, new_name, blocked_by, blocked_errno}`` and the marker stays), then the completion line
    ``project_rename_completed`` on the renamed tree, an ``AuditEntryNotWritten`` folded into the
    outcome's ``note``, then the marker delete at the version read under the new root's key, then
    :func:`tcip_store.close_connections` again; the outcome is ``{name, new_name}``. Resume: when
    the child's own name already equals the marker's ``new_name``, no rename; the completion line
    is written first, its failure folded into ``note`` the same way, then the marker is deleted at
    the version read, and the outcome says ``{name, new_name, already_renamed, note}``. Skipped:
    the marker itself could not be read (a loose layout, an undecodable or over-version document),
    answering ``{name, skipped}`` with that project left exactly as it was.
    """
    outcomes: list[dict] = []
    ws = workspace_root
    if not ws.is_dir():
        return outcomes

    for child in sorted(p for p in ws.iterdir() if p.is_dir() and (p / ".tcip").is_dir()):
        name = child.name
        try:
            versioned = tcip_store.read_versioned(
                workspace.pending_rename_key(child), default=None,
            )
        except (tcip_store.StoreError, tcip_store.DecodeError,
                tcip_store.SchemaVersionRefused) as exc:
            outcomes.append({"name": name, "skipped": str(exc)})
            continue
        record = versioned.value
        if record is None:
            continue
        version = versioned.version
        new_name = record["new_name"]
        old_name = record["old_name"]

        if name == new_name:
            notes: list[str] = []
            try:
                audit.record_event_or_raise(
                    "project_rename_completed",
                    {"name": old_name, "new_name": new_name, "old_root": str(ws / old_name)},
                    scope=child,
                )
                line_written = True
            except audit.AuditEntryNotWritten as exc:
                line_written = False
                notes.append(f"{exc}; the marker stands so the next start writes it")
            if line_written:
                try:
                    tcip_store.delete(workspace.pending_rename_key(child), expect=version)
                except tcip_store.StoreError as exc:
                    notes.append(str(exc))
            outcome = {"name": old_name, "new_name": new_name, "already_renamed": True}
            if notes:
                outcome["note"] = "; ".join(notes)
            outcomes.append(outcome)
            continue

        destination = ws / new_name
        if os.path.lexists(destination):
            outcomes.append({"name": name, "new_name": new_name,
                              "blocked_by": f"{destination} is already taken in the workspace"})
            continue

        tcip_store.close_connections()
        try:
            def _rename(src: Path = child, dst: Path = destination) -> None:
                os.rename(str(src), str(dst))

            retry_while_denied(_rename, project_removal.RENAME_BUDGET_S)
        except OSError as exc:
            exc_errno = getattr(exc, "errno", None)
            reason = (f"{destination} is on another filesystem: {exc}"
                      if exc_errno == errno.EXDEV else str(exc))
            outcomes.append({"name": name, "new_name": new_name, "blocked_by": reason,
                              "blocked_errno": exc_errno})
            continue

        notes = []
        try:
            audit.record_event_or_raise(
                "project_rename_completed",
                {"name": name, "new_name": new_name, "old_root": str(child)},
                scope=destination,
            )
            line_written = True
        except audit.AuditEntryNotWritten as exc:
            line_written = False
            notes.append(f"{exc}; the marker stands so the next start writes it")
        if line_written:
            try:
                tcip_store.delete(workspace.pending_rename_key(destination), expect=version)
            except tcip_store.StoreError as exc:
                notes.append(str(exc))
        tcip_store.close_connections()

        outcome = {"name": name, "new_name": new_name}
        if notes:
            outcome["note"] = "; ".join(notes)
        outcomes.append(outcome)

    with _rename_startup_outcomes_lock:
        _rename_startup_outcomes.clear()
        _rename_startup_outcomes.extend(outcomes)
    return outcomes


def rename_startup_outcomes() -> list[dict]:
    """Every outcome the last :func:`complete_pending_renames` run recorded, kept for this
    process's life the way :func:`tcip_mcp.project_removal.startup_outcomes` is."""
    with _rename_startup_outcomes_lock:
        return list(_rename_startup_outcomes)
