"""Annotation session telemetry in a workspace project's annotation-stats record: each
session's person, start and end, and one entry per contribution to an image naming its activity.
Every throughput figure is derived from those entries when they are read."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Literal, Optional, get_args

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import tcip_store

from tcip_mcp import audit
from tcip_mcp.audit import now_iso
from tcip_mcp.identity import actor
from tcip_mcp.web_client import annotation_stats_key

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

Activity = Literal["new_annotation", "review", "negative_confirmation"]
"""What one contribution to an image was: annotations added, a negative confirmed, or review."""


def _session_start(sessions: list[dict[str, Any]]) -> str:
    """The start stamp of a session opening now among ``sessions``, latest first: this instant,
    or one microsecond past the latest's stamp when this instant does not exceed it, so no two
    sessions of a project share a stamp and the stamp identifies the session."""
    now = datetime.fromisoformat(now_iso())
    latest = datetime.fromisoformat(sessions[0]["started"]) if sessions else None
    if latest is not None and now <= latest:
        now = latest + timedelta(microseconds=1)
    return now.isoformat()


class ImageEntry(BaseModel):
    """One contribution to an image: the identity the page minted for it, the seconds spent, the
    annotations added and its activity."""

    contribution_id: str = Field(min_length=1)
    image_name: str
    seconds: float = Field(ge=0)
    annotations_added: int = Field(ge=0)
    activity: Activity


class ImageEventPayload(ImageEntry):
    """One contribution, by the person ``user`` names, to the project whose id it names, in the
    session of that project whose start stamp ``started`` is, or ``None`` when the page records
    no session of that person there."""

    user: str
    project_id: str
    started: Optional[str]


class SessionRef(BaseModel):
    """One session, as a page names it to end it: the project it was recorded in and the start
    stamp the contribution that opened it answered with."""

    project_id: str
    started: str


class SessionPlace(BaseModel):
    """The session a contribution is recorded in: its start stamp, and whether it had ended when
    the answer was made (another page may end it afterwards)."""

    started: str
    ended: bool


class SessionWrite(BaseModel):
    """What a session write answers: ``ok`` when it touched a session, ``noop`` when it touched
    none, and ``session``, the session the contribution is recorded in, whether it landed now or
    is a repeat of one recorded before; ``None`` for a contribution with nothing to record and
    for every end answer."""

    status: Literal["ok", "noop"]
    session: Optional[SessionPlace]


class SessionSummary(BaseModel):
    """One session as recorded, with the throughput its entries amount to: the images given
    annotations, the annotations added, the seconds spent in all and by activity, and the seconds
    per added annotation (``None`` when none was added)."""

    user: str
    started: str
    ended: Optional[str]
    entries: list[ImageEntry]
    images_annotated: int
    total_annotations: int
    total_time_seconds: float
    seconds_by_activity: dict[Activity, float]
    avg_seconds_per_annotation: Optional[float]


@router.post("/image_event")
async def image_event(payload: ImageEventPayload) -> SessionWrite:
    """Record one contribution by the person ``user`` names and leave one ``image_event`` line by
    that person; answers the session the contribution is recorded in (:class:`SessionWrite`).

    A contribution naming a session (``started``) lands in it, in the workspace project its
    ``project_id`` names whatever project is open, while that session is the person's and
    unended: the one fact a departure (:func:`end_sessions`) changes, checked inside the
    record's transaction. One naming none lands in the open project
    (:meth:`~tcip_web.state.StateStore.admitted`), in the latest session when it is the
    person's and unended, else in a new one. A contribution whose ``contribution_id`` the record
    already holds is a repeat: it lands nowhere, leaves no line, and answers ``noop`` with the
    session it first landed in, ended or not. A contribution with no time, no annotations and
    no confirmed negative records nothing and answers ``noop`` with no session. Answers 404 for
    a ``project_id`` no workspace project holds, 409 for a named session that is not the
    person's unended session, for a ``project_id`` that is not the open project's when no
    session is named, and when the line cannot follow the write."""
    from tcip_web.routes.projects import workspace_project
    from tcip_web.state import store

    person = actor(payload.user)
    entry = payload.model_dump(exclude={"user", "project_id", "started"})
    if not (payload.seconds or payload.annotations_added
            or payload.activity == "negative_confirmation"):
        return SessionWrite(status="noop", session=None)

    def land(root: Path) -> tuple[Path, SessionWrite]:
        return root, _land(root, payload.started, entry, person)

    if payload.started is None:
        root, written = await store.admitted(payload.project_id, land)
    else:
        root, written = await asyncio.to_thread(land, workspace_project(payload.project_id)[0])
    if written.status == "ok":
        await asyncio.to_thread(_answering, written, lambda: audit.record_event_or_raise(
            "image_event", entry, actor=person, scope=root))
    return written


def _land(root: Path, started: Optional[str], entry: dict[str, Any],
          person: str) -> SessionWrite:
    """``entry`` appended, in one transaction on ``root``'s stats record, to the session
    ``started`` names, refused (409 naming why) unless it is ``person``'s and unended; with
    ``started`` ``None``, to the latest session when it is ``person``'s and unended, else to a
    new one. An entry whose ``contribution_id`` a session already holds is not appended: the
    answer is ``noop`` with that session."""
    key = annotation_stats_key(str(root))
    with tcip_store.transaction(key) as txn:
        data = txn.read(key, default={"sessions": []})
        sessions: list[dict[str, Any]] = data["sessions"]
        for held in sessions:
            if any(e["contribution_id"] == entry["contribution_id"] for e in held["entries"]):
                return SessionWrite(status="noop", session=SessionPlace(
                    started=held["started"], ended=held["ended"] is not None))
        if started is None:
            if not sessions or sessions[0]["ended"] is not None or sessions[0]["user"] != person:
                sessions.insert(0, {"user": person, "started": _session_start(sessions),
                                    "ended": None, "entries": []})
            session = sessions[0]
        else:
            named = next((s for s in sessions if s["started"] == started), None)
            if named is None or named["ended"] is not None or named["user"] != person:
                why = ("carries no session started then" if named is None
                       else "ended that session" if named["ended"] is not None
                       else "holds that session for another person")
                raise HTTPException(
                    status_code=409, detail=f"the project {why}: {started}; contribution refused")
            session = named
        session["entries"].append(entry)
        txn.write(key, data)
    return SessionWrite(status="ok", session=SessionPlace(started=session["started"], ended=False))


@router.post("/end")
def end_session(payload: SessionRef) -> SessionWrite:
    """Mark the session ``payload`` names as ended (:func:`end_sessions`), answering ``ok``, or
    ``noop`` for a session already ended, and no session either way. Answers 409 for a
    ``project_id`` that is not the open project's, for a ``started`` no session of that project
    carries, and when the line cannot follow the write."""
    from tcip_web.state import store

    project = store.admit(payload.project_id)
    try:
        ended = end_sessions(project.root, payload.started)
    except LookupError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    written = SessionWrite(status="ok" if ended else "noop", session=None)
    _answering(written, lambda: record_ended(project.root, ended))
    return written


def end_sessions(root: Path, started: Optional[str]) -> list[dict[str, Any]]:
    """Mark ended, in one transaction on ``root``'s stats record, the session ``started`` names,
    or every unended session (each person's) when ``started`` is ``None``; the sessions it ended,
    each as recorded. Raises ``LookupError`` for a ``started`` no session of the project
    carries."""
    key = annotation_stats_key(str(root))
    with tcip_store.transaction(key) as txn:
        data = txn.read(key, default={"sessions": []})
        named = [s for s in data["sessions"] if started in (None, s["started"])]
        if started is not None and not named:
            raise LookupError(f"no session of the open project started at {started}")
        ended = [s for s in named if s["ended"] is None]
        stamp = now_iso()
        for session in ended:
            session["ended"] = stamp
        if ended:
            txn.write(key, data)
    return ended


def record_ended(root: Path, ended: list[dict[str, Any]]) -> None:
    """One ``session_ended`` line in ``root``'s log per session of ``ended``
    (:func:`end_sessions`), by its person; raises ``AuditEntryNotWrittenError`` when one cannot
    follow."""
    for session in ended:
        audit.record_event_or_raise("session_ended", {"started": session["started"]},
                                    actor=session["user"], scope=root)


def _answering(written: SessionWrite, record: Callable[[], None]) -> None:
    """Run ``record`` for a session write already made; a line that cannot be appended is raised
    as the 409 carrying ``written``, the answer the write earned."""
    from tcip_web.routes.audit_gap import audit_gap_409

    try:
        record()
    except audit.AuditEntryNotWrittenError as exc:
        raise audit_gap_409(exc, written) from exc


@router.get("/load")
def load_sessions() -> dict[str, list[SessionSummary]]:
    """Every recorded session, latest first, each with the throughput its entries amount to."""
    from tcip_web.state import store

    data = tcip_store.read(annotation_stats_key(str(store.held().root)), default={"sessions": []})
    return {"sessions": [_summary(s) for s in data["sessions"]]}


def _summary(session: dict[str, Any]) -> SessionSummary:
    """``session`` with the figures derived from its entries."""
    entries = [ImageEntry.model_validate(e) for e in session["entries"]]
    by_activity: dict[Activity, float] = dict.fromkeys(get_args(Activity), 0.0)
    for e in entries:
        by_activity[e.activity] += e.seconds
    added = sum(e.annotations_added for e in entries)
    return SessionSummary(
        user=session["user"], started=session["started"], ended=session["ended"],
        entries=entries,
        images_annotated=len({e.image_name for e in entries if e.annotations_added}),
        total_annotations=added,
        total_time_seconds=round(sum(e.seconds for e in entries), 2),
        seconds_by_activity={k: round(v, 2) for k, v in by_activity.items()},
        avg_seconds_per_annotation=(round(by_activity["new_annotation"] / added, 2)
                                    if added else None),
    )
