"""Session-tracking routes: annotation_stats.json equivalent.

A per-image annotation timer + session aggregate at
``<project_root>/.tcip/state/annotation_stats.json`` with this shape::

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

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _guarded_stats_key(project_root: str) -> Key:
    """The stats key of a client-supplied project root, confined first (403 on escape)."""
    from tcip_web.paths import assert_project_root_allowed

    try:
        return annotation_stats_key(str(assert_project_root_allowed(project_root)))
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc


def _guarded_dataset_root(dataset_root: str) -> str:
    """A client-supplied dataset root, confined and resolved before it is persisted."""
    from tcip_web.paths import assert_path_allowed

    try:
        return str(assert_path_allowed(dataset_root))
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc


# ── Per-image session telemetry ─────────────────────────────────────────


class ImageEventPayload(BaseModel):
    project_root: str
    image_name: str
    session_seconds_delta: float = 0.0       # incremental time added
    annotations_added_delta: int = 0         # new annotations created during this slice
    final_annotation_count: int              # boxes + polygons after the slice
    # Where this image's own image_status.json entry lives, so a read-time classification of
    # this session's time (new annotation / review / negative confirmation) can look it up later.
    # Optional: a caller with no dataset context in hand still gets recorded, just unclassifiable.
    dataset_root: Optional[str] = None
    subject: Optional[str] = None
    date: Optional[str] = None


@router.post("/image_event")
def image_event(payload: ImageEventPayload) -> dict:
    """Record per-image session activity, adding this call's deltas to the image's totals."""
    key = _guarded_stats_key(payload.project_root)
    dataset_root = _guarded_dataset_root(payload.dataset_root) if payload.dataset_root else None
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
    project_root: str
    user: str = ""


@router.post("/start")
def start_session(payload: StartSessionPayload) -> dict:
    """Insert a new session row."""
    key = _guarded_stats_key(payload.project_root)
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


class EndSessionPayload(BaseModel):
    project_root: str


@router.post("/end")
def end_session(payload: EndSessionPayload) -> dict:
    """Mark the latest session as ended and roll up totals."""
    key = _guarded_stats_key(payload.project_root)
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
def load_sessions(project_root: str) -> dict:
    data = tcip_store.read(_guarded_stats_key(project_root), default={"sessions": []})
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


def _status_bucket_for(cache: dict[str, dict[str, str]], dataset_root: str,
                       subject: str | None, date: str | None) -> dict[str, str]:
    """One (dataset_root, subject, date) bucket of image_name -> status, read at most once per call
    to :func:`_classify_session_seconds` regardless of how many images in a session share it.

    A dataset with no confirmations yet has no bucket, which is an empty one. A recorded root the
    allow-set does not admit is refused with a 403 naming it, nothing outside the allowed roots
    read; a store that will not decode raises ``tcip_store.DecodeError``.
    """
    from tcip_mcp.dataset_layout import image_status_key, status_tokens, status_bucket
    from tcip_web.paths import assert_path_allowed

    key = f"{dataset_root}\0{subject or ''}\0{date or ''}"
    if key not in cache:
        try:
            allowed = assert_path_allowed(dataset_root)
        except ValueError as exc:
            raise HTTPException(403, f"a session records time on images under {dataset_root}, "
                                     f"which this server may not read, so that time cannot be "
                                     f"classified: {exc}") from exc
        raw = tcip_store.read(image_status_key(allowed), default={})
        cache[key] = status_tokens(raw).get(status_bucket(subject or "", date), {})
    return cache[key]


def _classify_session_seconds(s: dict[str, Any]) -> dict[str, float]:
    """This session's time, split into new-annotation / review / negative-confirmation seconds,
    read against image_status.json's current state. An image with no dataset_root recorded counts
    as review time.
    """
    from tcip_mcp.dataset_layout import is_confirmed_negative

    images = s["images"]
    cache: dict[str, dict[str, str]] = {}
    negative_seconds = review_seconds = annotation_seconds = 0.0
    for name, img in images.items():
        seconds = img["session_seconds"]
        if img["annotations_added"] > 0:
            annotation_seconds += seconds
            continue
        dataset_root = img.get("dataset_root")
        status = (
            _status_bucket_for(cache, dataset_root, img.get("subject"), img.get("date")).get(name)
            if dataset_root else None
        )
        if is_confirmed_negative(status):
            negative_seconds += seconds
        else:
            review_seconds += seconds
    return {
        "negative_confirmation_seconds": round(negative_seconds, 2),
        "review_seconds": round(review_seconds, 2),
        "new_annotation_seconds": round(annotation_seconds, 2),
    }
