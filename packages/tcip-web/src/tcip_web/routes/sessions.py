"""Session-tracking routes: annotation_stats.json equivalent.

A per-image annotation timer + session aggregate at the open project's
``.tcip/state/annotation_stats.json`` with this shape::

    {
        "sessions": [
            {
                "user": "exx",
                "started": "26-04-2026 09:00:00",
                "ended":   "26-04-2026 09:32:14",
                "images_annotated": 7,
                "total_annotations": 154,
                "total_time_seconds": 1934.21,
                "avg_seconds_per_annotation": 12.56,
                "images": {
                    "IMG_0001.JPG": {
                        "session_seconds": 273.4,
                        "annotations_added": 8,
                        "final_annotation_count": 29,
                        "avg_seconds_per_annotation": 34.2
                    },
                    ...
                }
            },
            ...
        ]
    }
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import tcip_store
from tcip_store import Key

from tcip_mcp.web_client import annotation_stats_key
from tcip_web.paths import allowed_path
from tcip_web.routes._body_common import EmptyBodyPayload

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _stats_key() -> Key:
    """The open project's stats key; raises ``NoProjectOpen`` when none is open."""
    from tcip_web.state import store

    return annotation_stats_key(str(store.open_root()))


# ── Per-image session telemetry ─────────────────────────────────────────


class ImageEventPayload(BaseModel):
    image_name: str
    session_seconds_delta: float = 0.0       # incremental time added
    annotations_added_delta: int = 0         # new annotations created during this slice
    final_annotation_count: int              # boxes + polygons after the slice
    # Where this image's label document lives, so a read-time classification of this session's
    # time (new annotation / review / negative confirmation) can look it up later.
    # Optional: a caller with no dataset context in hand still gets recorded, just unclassifiable.
    dataset_root: Optional[str] = None
    subject: Optional[str] = None
    date: Optional[str] = None


@router.post("/image_event")
def image_event(payload: ImageEventPayload) -> dict:
    """Record per-image session activity, adding this call's deltas to the image's totals."""
    key = _stats_key()
    dataset_root = str(allowed_path(payload.dataset_root)) if payload.dataset_root else None
    with tcip_store.transaction(key) as txn:
        data = txn.read(key, default={"sessions": []})
        sessions: list[dict[str, Any]] = data["sessions"]
        if not sessions:
            sessions.insert(0, _new_session_entry())
        s = sessions[0]
        images: dict[str, Any] = s["images"]
        img = images.setdefault(
            payload.image_name,
            {
                "session_seconds": 0.0,
                "annotations_added": 0,
                "final_annotation_count": 0,
                "avg_seconds_per_annotation": 0.0,
            },
        )

        if dataset_root:
            img["dataset_root"] = dataset_root
            img["subject"] = payload.subject
            img["date"] = payload.date

        img["session_seconds"] = round(
            img["session_seconds"] + max(0.0, payload.session_seconds_delta), 2)
        img["annotations_added"] += max(0, payload.annotations_added_delta)
        img["final_annotation_count"] = int(payload.final_annotation_count)

        if img["annotations_added"] > 0:
            img["avg_seconds_per_annotation"] = round(
                img["session_seconds"] / img["annotations_added"], 2
            )
        else:
            img["avg_seconds_per_annotation"] = 0.0

        # Drop entries that ended up empty (no time + no adds + no final count)
        if (
            img["session_seconds"] == 0.0
            and img["annotations_added"] == 0
            and img["final_annotation_count"] == 0
        ):
            images.pop(payload.image_name, None)

        _refresh_session_aggregate(s)
        txn.write(key, data)
    return {"status": "ok"}


# ── Session lifecycle ──────────────────────────────────────────────────


class StartSessionPayload(BaseModel):
    user: str = ""


@router.post("/start")
def start_session(payload: StartSessionPayload) -> dict:
    """Insert a new session row."""
    key = _stats_key()
    with tcip_store.transaction(key) as txn:
        data = txn.read(key, default={"sessions": []})
        sessions: list[dict[str, Any]] = data["sessions"]
        if sessions and not sessions[0]["ended"]:
            # Already an open session, keep it.
            return {"status": "ok", "session": sessions[0]}
        entry = _new_session_entry(user=payload.user)
        sessions.insert(0, entry)
        txn.write(key, data)
    return {"status": "ok", "session": entry}


@router.post("/end")
def end_session(payload: EmptyBodyPayload) -> dict:
    """Mark the latest session as ended and roll up totals."""
    key = _stats_key()
    with tcip_store.transaction(key) as txn:
        data = txn.read(key, default={"sessions": []})
        sessions = data["sessions"]
        if not sessions:
            return {"status": "noop"}
        s = sessions[0]
        s["ended"] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        _refresh_session_aggregate(s)
        txn.write(key, data)
    return {"status": "ok", "session": s}


@router.get("/load")
def load_sessions() -> dict:
    data = tcip_store.read(_stats_key(), default={"sessions": []})
    for s in data["sessions"]:
        s.update(_classify_session_seconds(s))
    return data


# ── helpers ────────────────────────────────────────────────────────────


def _new_session_entry(user: str = "") -> dict[str, Any]:
    return {
        "user": user,
        "started": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "ended": "",
        "images_annotated": 0,
        "total_annotations": 0,
        "total_time_seconds": 0.0,
        "avg_seconds_per_annotation": 0.0,
        "images": {},
    }


def _refresh_session_aggregate(s: dict[str, Any]) -> None:
    # total_time_seconds is every image with real session time, negative confirmation and pure
    # review included, not just images that gained a new annotation; avg_seconds_per_annotation
    # keeps its own narrower time sum so it stays a per-new-annotation figure, not diluted by time
    # that produced no new annotation.
    images = s["images"]
    images_with_adds = [v for v in images.values() if v["annotations_added"] > 0]
    total_seconds = round(sum(v["session_seconds"] for v in images.values()), 2)
    annotation_seconds = sum(v["session_seconds"] for v in images_with_adds)
    total_adds = sum(v["annotations_added"] for v in images.values())
    s["images_annotated"] = len(images_with_adds)
    s["total_annotations"] = total_adds
    s["total_time_seconds"] = total_seconds
    s["avg_seconds_per_annotation"] = (
        round(annotation_seconds / total_adds, 2) if total_adds > 0 else 0.0
    )


def _marked_negative(dataset_root: str, subject: str, date: str | None, image_name: str) -> bool:
    """Whether ``image_name``'s label document under ``dataset_root`` and ``date`` says ``subject``
    is a negative (:meth:`~tcip_annotation.json_io.LabelDocument.state`). A recorded root the
    allow-set does not admit is refused with a 403 naming it, nothing outside the allowed roots
    read; a document that will not read raises."""
    from pathlib import Path

    from tcip_annotation.json_io import read_label_document

    from tcip_mcp.dataset_layout import annotation_path
    from tcip_web.paths import assert_path_allowed

    try:
        allowed = assert_path_allowed(dataset_root)
    except ValueError as exc:
        raise HTTPException(403, f"a session records time on images under {dataset_root}, "
                                 f"which this server may not read, so that time cannot be "
                                 f"classified: {exc}") from exc
    label = annotation_path(allowed, date, Path(image_name).stem)
    return read_label_document(label).state(subject) == "negative"


def _classify_session_seconds(s: dict[str, Any]) -> dict[str, float]:
    """This session's time, split into new-annotation / review / negative-confirmation seconds,
    read against each image's label document as it stands. An image with no dataset_root or
    subject recorded counts as review time.
    """
    images = s["images"]
    negative_seconds = review_seconds = annotation_seconds = 0.0
    for name, img in images.items():
        seconds = img["session_seconds"]
        if img["annotations_added"] > 0:
            annotation_seconds += seconds
            continue
        dataset_root, subject = img.get("dataset_root"), img.get("subject")
        if dataset_root and subject and _marked_negative(dataset_root, subject, img.get("date"),
                                                         name):
            negative_seconds += seconds
        else:
            review_seconds += seconds
    return {
        "negative_confirmation_seconds": round(negative_seconds, 2),
        "review_seconds": round(review_seconds, 2),
        "new_annotation_seconds": round(annotation_seconds, 2),
    }
