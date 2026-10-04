"""Annotation session telemetry in the open project's annotation-stats record: each
session's person, start and end, and one entry per contribution to an image naming its activity.
Every throughput figure is derived from those entries when they are read."""

from __future__ import annotations

from typing import Any, Literal, Optional, get_args

from fastapi import APIRouter
from pydantic import BaseModel, Field

import tcip_store
from tcip_store import Key

from tcip_mcp.audit import now_iso
from tcip_mcp.identity import actor
from tcip_mcp.web_client import annotation_stats_key
from tcip_web.routes._body_common import EmptyBodyPayload

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

Activity = Literal["new_annotation", "review", "negative_confirmation"]
"""What one contribution to an image was: annotations added, a negative confirmed, or review."""


def _stats_key() -> Key:
    """The open project's stats key; raises ``NoProjectOpen`` when none is open."""
    from tcip_web.state import store

    return annotation_stats_key(str(store.open_root()))


class ImageEntry(BaseModel):
    """One contribution to an image: the seconds spent, the annotations added and its activity."""

    image_name: str
    seconds: float = Field(ge=0)
    annotations_added: int = Field(ge=0)
    activity: Activity


class ImageEventPayload(ImageEntry):
    """One contribution, by the person ``user`` names, to the project whose id it names."""

    user: str
    project_id: str


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
def image_event(payload: ImageEventPayload) -> dict:
    """Record one contribution in the open session of the person ``user`` names, opening a new
    session when the latest one has ended or is another person's, and leave one ``image_event``
    line by that person. A contribution with no time and no annotations records nothing. Answers
    409 for a ``project_id`` that is not the open project's, and when the line cannot follow the
    write."""
    from tcip_web.state import store

    person = actor(payload.user)
    key = annotation_stats_key(str(store.admit(payload.project_id)))
    entry = payload.model_dump(exclude={"user", "project_id"})
    if not (payload.seconds or payload.annotations_added):
        return {"status": "noop"}
    with tcip_store.transaction(key) as txn:
        data = txn.read(key, default={"sessions": []})
        sessions: list[dict[str, Any]] = data["sessions"]
        if not sessions or sessions[0]["ended"] is not None or sessions[0]["user"] != person:
            sessions.insert(0, {"user": person, "started": now_iso(), "ended": None,
                                "entries": []})
        sessions[0]["entries"].append(entry)
        txn.write(key, data)
    _record("image_event", entry, person)
    return {"status": "ok"}


@router.post("/end")
def end_session(payload: EmptyBodyPayload) -> dict:
    """Mark the latest session as ended, leaving one ``session_ended`` line by its person; answers
    409 when the line cannot follow the write."""
    key = _stats_key()
    with tcip_store.transaction(key) as txn:
        data = txn.read(key, default={"sessions": []})
        sessions = data["sessions"]
        if not sessions or sessions[0]["ended"] is not None:
            return {"status": "noop"}
        sessions[0]["ended"] = now_iso()
        txn.write(key, data)
    _record("session_ended", {"started": sessions[0]["started"]}, sessions[0]["user"])
    return {"status": "ok"}


def _record(tool: str, arguments: dict[str, Any], person: str) -> None:
    """One line in the open project's log for a session write already made, by ``person``."""
    from tcip_mcp.audit import AuditEntryNotWritten, record_event_or_raise
    from tcip_web.routes.audit_gap import audit_gap_409
    from tcip_web.state import store

    try:
        record_event_or_raise(tool, arguments, actor=person, scope=store.open_root())
    except AuditEntryNotWritten as exc:
        raise audit_gap_409(exc, {"status": "ok"}) from exc


@router.get("/load")
def load_sessions() -> dict[str, list[SessionSummary]]:
    """Every recorded session, latest first, each with the throughput its entries amount to."""
    data = tcip_store.read(_stats_key(), default={"sessions": []})
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
