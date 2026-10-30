"""Annotation session telemetry in a workspace project's annotation-stats record.

The record holds each session's person, start and end, and one entry per contribution to an
image naming its activity. Every throughput figure is derived from those entries when they are
read.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, get_args

from fastapi import APIRouter, HTTPException
from pydantic import AwareDatetime, BaseModel, Field, JsonValue
from pydantic_core import to_jsonable_python

import tcip_store
from tcip_mcp import audit
from tcip_mcp.audit import now
from tcip_mcp.identity import actor
from tcip_mcp.web_client import annotation_stats_key

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

Activity = Literal["new_annotation", "review", "negative_confirmation"]
"""What one contribution to an image was: annotations added, a negative confirmed, or review."""


class ImageEntry(BaseModel):
    """One contribution to an image: its identity, seconds, annotations added and activity."""

    contribution_id: str = Field(min_length=1)
    image_name: str
    seconds: float = Field(ge=0)
    annotations_added: int = Field(ge=0)
    activity: Activity


class Session(BaseModel):
    """One session as recorded: its person, its start and end stamps, and its contributions."""

    user: str
    started: AwareDatetime
    ended: AwareDatetime | None
    entries: list[ImageEntry]


class AnnotationStats(BaseModel):
    """A project's annotation-stats record: its sessions, latest first."""

    sessions: list[Session]


def read_stats(raw: JsonValue) -> AnnotationStats:
    """Return the annotation-stats record ``raw`` decodes to, holding no session for ``None``.

    A document that is not the record's whole shape raises ``ValidationError``.
    """
    if raw is None:
        return AnnotationStats(sessions=[])
    return AnnotationStats.model_validate(raw)


def _session_start(sessions: list[Session]) -> datetime:
    """Return the start stamp of a session opening now among ``sessions``, latest first.

    The stamp is this instant, or one microsecond past the latest's stamp when this instant does
    not exceed it, so no two sessions of a project share a stamp and the stamp identifies the
    session.
    """
    stamp = now()
    if sessions and stamp <= sessions[0].started:
        stamp = sessions[0].started + timedelta(microseconds=1)
    return stamp


class ImageEventPayload(ImageEntry):
    """One contribution, by the person ``user`` names, to the project whose id it names.

    It lands in the session of that project whose start stamp ``started`` is, or ``started`` is
    ``None`` for a contribution naming no session.
    """

    user: str
    project_id: str
    started: AwareDatetime | None


class SessionRef(BaseModel):
    """One session: its project and its start stamp.

    The project is the one it was recorded in, and the stamp the one the contribution that
    opened it answered with.
    """

    project_id: str
    started: AwareDatetime


class SessionPlace(BaseModel):
    """The session a contribution is recorded in: its start stamp and whether it had ended.

    Whether it had ended is as of the answer; it may be ended afterwards.
    """

    started: AwareDatetime
    ended: bool


class SessionWrite(BaseModel):
    """What a session write answers: its status and the session the contribution is in.

    The status is ``ok`` when it touched a session and ``noop`` when it touched none.
    ``session`` is the session the contribution is recorded in, whether it landed now or is a
    repeat of one recorded before; ``None`` for a contribution with nothing to record and for
    every end answer.
    """

    status: Literal["ok", "noop"]
    session: SessionPlace | None


class SessionSummary(Session):
    """One session as recorded, with the throughput its entries amount to.

    The throughput is the images given annotations, the annotations added, the seconds spent in
    all and by activity, and the seconds per added annotation (``None`` when none was added).
    """

    images_annotated: int
    total_annotations: int
    total_time_seconds: float
    seconds_by_activity: dict[Activity, float]
    avg_seconds_per_annotation: float | None


@router.post("/image_event")
async def image_event(payload: ImageEventPayload) -> SessionWrite:
    """Record one contribution by the person ``user`` names and leave one ``image_event`` line.

    The line is by that person; answers the session the contribution is recorded in
    (:class:`SessionWrite`).

    A contribution naming a session (``started``) lands in it, in the workspace project its
    ``project_id`` names whatever project is open, while that session is the person's and
    unended, checked inside the record's transaction. One naming none lands in the open project
    (:meth:`~tcip_web.state.StateStore.admitted`), in the latest session when it is the
    person's and unended, else in a new one. A contribution whose ``contribution_id`` the record
    already holds is a repeat: it lands nowhere, leaves no line, and answers ``noop`` with the
    session it first landed in, ended or not. A contribution with no time, no annotations and
    no confirmed negative records nothing and answers ``noop`` with no session. Answers 404 for
    a ``project_id`` no workspace project holds, 409 for a named session that is not the
    person's unended session, for a ``project_id`` that is not the open project's when no
    session is named, and when the line cannot follow the write.
    """
    from tcip_web.routes.projects import workspace_project
    from tcip_web.state import store

    person = actor(payload.user)
    entry = ImageEntry(
        **{name: getattr(payload, name) for name in ImageEntry.model_fields}
    )
    if not (
        payload.seconds
        or payload.annotations_added
        or payload.activity == "negative_confirmation"
    ):
        return SessionWrite(status="noop", session=None)

    def land(root: Path) -> tuple[Path, SessionWrite]:
        return root, _land(root, payload.started, entry, person)

    if payload.started is None:
        root, written = await store.admitted(payload.project_id, land)
    else:
        root, written = await asyncio.to_thread(
            land, workspace_project(payload.project_id)[0]
        )
    if written.status == "ok":
        await asyncio.to_thread(
            _answering,
            written,
            lambda: audit.record_event_or_raise(
                "image_event", entry.model_dump(), actor=person, scope=root
            ),
        )
    return written


def _land(
    root: Path, started: datetime | None, entry: ImageEntry, person: str
) -> SessionWrite:
    """Append ``entry`` to a session in one transaction on ``root``'s stats record.

    The session is the one ``started`` names, refused (409 naming why) unless it is
    ``person``'s and unended; with ``started`` ``None``, the latest session when it is
    ``person``'s and unended, else a new one. An entry whose ``contribution_id`` a session
    already holds is not appended: the answer is ``noop`` with that session.
    """
    key = annotation_stats_key(str(root))
    with tcip_store.transaction(key) as txn:
        stats = read_stats(txn.read(key, default=None))
        sessions = stats.sessions
        for held in sessions:
            if any(
                e.contribution_id == entry.contribution_id
                for e in held.entries
            ):
                return SessionWrite(
                    status="noop",
                    session=SessionPlace(
                        started=held.started, ended=held.ended is not None
                    ),
                )
        if started is None:
            if (
                not sessions
                or sessions[0].ended is not None
                or sessions[0].user != person
            ):
                sessions.insert(
                    0,
                    Session(
                        user=person,
                        started=_session_start(sessions),
                        ended=None,
                        entries=[],
                    ),
                )
            session = sessions[0]
        else:
            named = next((s for s in sessions if s.started == started), None)
            if (
                named is None
                or named.ended is not None
                or named.user != person
            ):
                why = (
                    "carries no session started then"
                    if named is None
                    else "ended that session"
                    if named.ended is not None
                    else "holds that session for another person"
                )
                raise HTTPException(
                    status_code=409,
                    detail=f"the project {why}: {to_jsonable_python(started)}; "
                    "contribution refused",
                )
            session = named
        session.entries.append(entry)
        txn.write(key, stats.model_dump(mode="json"))
    return SessionWrite(
        status="ok",
        session=SessionPlace(started=session.started, ended=False),
    )


@router.post("/end")
def close_session(payload: SessionRef) -> SessionWrite:
    """Mark the session ``payload`` names as ended (:func:`end_sessions`).

    Answers ``ok``, or ``noop`` for a session already ended, and no session either way. Answers
    409 for a ``project_id`` that is not the open project's, for a ``started`` no session of
    that project carries, and when the line cannot follow the write.
    """
    from tcip_web.state import store

    project = store.admit(payload.project_id)
    try:
        ended = end_sessions(project.root, payload.started)
    except LookupError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    written = SessionWrite(status="ok" if ended else "noop", session=None)
    _answering(written, lambda: record_ended(project.root, ended))
    return written


def end_sessions(root: Path, started: datetime | None) -> list[audit.AuditLine]:
    """Mark sessions ended in one transaction on ``root``'s stats record.

    The session ended is the one ``started`` names, or every unended session (each person's)
    when ``started`` is ``None``. Returns each ended session's ``session_ended`` line, by its
    person, whose instant the session records as its end. Raises ``LookupError`` for a
    ``started`` no session of the project carries.
    """
    key = annotation_stats_key(str(root))
    with tcip_store.transaction(key) as txn:
        stats = read_stats(txn.read(key, default=None))
        named = [s for s in stats.sessions if started in (None, s.started)]
        if started is not None and not named:
            raise LookupError(
                "no session of the open project started at "
                f"{to_jsonable_python(started)}"
            )
        lines = []
        for session in (s for s in named if s.ended is None):
            line = audit.audit_line(
                "session_ended", {"started": session.started}, session.user, "ok"
            )
            session.ended = line.timestamp
            lines.append(line)
        if lines:
            txn.write(key, stats.model_dump(mode="json"))
    return lines


def record_ended(root: Path, lines: list[audit.AuditLine]) -> None:
    """Append each ``session_ended`` line of ``lines`` to ``root``'s log.

    Raises ``AuditEntryNotWrittenError`` when one cannot follow.
    """
    for line in lines:
        audit.append_or_raise(line, scope=root)


def _answering(written: SessionWrite, record: Callable[[], None]) -> None:
    """Run ``record`` for a session write already made.

    A line that cannot be appended is raised as the 409 carrying ``written``, the answer the
    write earned.
    """
    from tcip_web.routes.audit_gap import audit_gap_409

    try:
        record()
    except audit.AuditEntryNotWrittenError as exc:
        raise audit_gap_409(exc, written) from exc


class SessionsListing(BaseModel):
    """Every session a project records, latest first, each with its derived throughput."""

    sessions: list[SessionSummary]


@router.get("/load")
def load_sessions() -> SessionsListing:
    """Return every recorded session of the open project (:class:`SessionsListing`)."""
    from tcip_web.state import store

    stats = read_stats(
        tcip_store.read(
            annotation_stats_key(str(store.held().root)), default=None
        )
    )
    return SessionsListing(sessions=[_summary(s) for s in stats.sessions])


def _summary(session: Session) -> SessionSummary:
    """Return ``session`` with the figures derived from its entries."""
    entries = session.entries
    by_activity: dict[Activity, float] = dict.fromkeys(get_args(Activity), 0.0)
    for e in entries:
        by_activity[e.activity] += e.seconds
    added = sum(e.annotations_added for e in entries)
    return SessionSummary(
        **dict(session),
        images_annotated=len(
            {e.image_name for e in entries if e.annotations_added}
        ),
        total_annotations=added,
        total_time_seconds=round(sum(e.seconds for e in entries), 2),
        seconds_by_activity={k: round(v, 2) for k, v in by_activity.items()},
        avg_seconds_per_annotation=(
            round(by_activity["new_annotation"] / added, 2) if added else None
        ),
    )
