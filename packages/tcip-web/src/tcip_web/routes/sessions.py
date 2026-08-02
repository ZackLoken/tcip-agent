"""Session-tracking routes: annotation_stats.json equivalent.

A per-image annotation timer + session aggregate at
``<project_root>/.tcip/state/annotation_stats.json`` with this shape::

    {
        "sessions": [
            {
                "user": "exx",
                "started": "2026-04-26 09:00:00",
                "ended":   "2026-04-26 09:32:14",
                "images_annotated": 7,
                "total_annotations": 154,
                "total_time_seconds": 1934.21,
                "avg_seconds_per_annotation": 12.56,
                "images": {
                    "IMG_0001.JPG": {
                        "session_seconds": 273.4,
                        "loaded_annotation_count": 21,
                        "annotations_added": 8,
                        "final_annotation_count": 29,
                        "avg_seconds_per_annotation": 34.2
                    },
                    ...
                }
            },
            ...
        ],
        "image_status": { "IMG_0001.JPG": "complete", ... }
    }

The image_status dict is co-owned with the classes-route's image_status.json
endpoint and is duplicated here only for the annotation-stats file shape.
The class-route file is canonical.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from tcip_mcp.utils.atomic_io import atomic_write_json, file_transaction, read_json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

STATS_FILENAME = "annotation_stats.json"


def _stats_path(project_root: str) -> Path:
    return Path(project_root) / ".tcip" / "state" / STATS_FILENAME


def _read(path: Path) -> dict[str, Any]:
    data = read_json(path, default=None)
    if not isinstance(data, dict):
        return {"sessions": [], "image_status": {}}
    return data


def _write(path: Path, data: dict[str, Any]) -> None:
    atomic_write_json(path, data)


# ── Per-image session telemetry ─────────────────────────────────────────


class ImageEventPayload(BaseModel):
    project_root: str
    image_name: str
    session_seconds_delta: float = 0.0       # incremental time added
    annotations_added_delta: int = 0         # new annotations created during this slice
    final_annotation_count: int              # boxes + polygons after the slice
    loaded_annotation_count: Optional[int] = None  # only set on first load


@router.post("/image_event")
def image_event(payload: ImageEventPayload) -> dict:
    """Record per-image session activity. Idempotent and additive.

    Called by the GUI on image-leave (Prev/Next/tab-switch/save).
    """
    path = _stats_path(payload.project_root)
    with file_transaction(path):
        data = _read(path)
        sessions: list[dict[str, Any]] = data.setdefault("sessions", [])
        if not sessions:
            sessions.insert(0, _new_session_entry())
        s = sessions[0]
        images: dict[str, Any] = s.setdefault("images", {})
        img = images.setdefault(
            payload.image_name,
            {
                "session_seconds": 0.0,
                "loaded_annotation_count": payload.loaded_annotation_count or 0,
                "annotations_added": 0,
                "final_annotation_count": 0,
                "avg_seconds_per_annotation": 0.0,
            },
        )

        if payload.loaded_annotation_count is not None and "loaded_annotation_count" not in img:
            img["loaded_annotation_count"] = int(payload.loaded_annotation_count)

        img["session_seconds"] = round(
            float(img.get("session_seconds", 0.0)) + max(0.0, payload.session_seconds_delta),
            2,
        )
        img["annotations_added"] = int(img.get("annotations_added", 0)) + max(
            0, payload.annotations_added_delta
        )
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
        _write(path, data)
    return {"status": "ok"}


# ── Session lifecycle ──────────────────────────────────────────────────


class StartSessionPayload(BaseModel):
    project_root: str
    user: str = ""


@router.post("/start")
def start_session(payload: StartSessionPayload) -> dict:
    """Insert a new session row. The GUI calls this when the user opens a
    project (or on load if no session is currently open)."""
    path = _stats_path(payload.project_root)
    with file_transaction(path):
        data = _read(path)
        sessions: list[dict[str, Any]] = data.setdefault("sessions", [])
        if sessions and not sessions[0].get("ended"):
            # Already an open session, keep it.
            return {"status": "ok", "session": sessions[0]}
        entry = _new_session_entry(user=payload.user)
        sessions.insert(0, entry)
        _write(path, data)
    return {"status": "ok", "session": entry}


class EndSessionPayload(BaseModel):
    project_root: str


@router.post("/end")
def end_session(payload: EndSessionPayload) -> dict:
    """Mark the latest session as ended and roll up totals."""
    path = _stats_path(payload.project_root)
    with file_transaction(path):
        data = _read(path)
        sessions = data.setdefault("sessions", [])
        if not sessions:
            return {"status": "noop"}
        s = sessions[0]
        s["ended"] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        _refresh_session_aggregate(s)
        _write(path, data)
    return {"status": "ok", "session": s}


@router.get("/load")
def load_sessions(project_root: str) -> dict:
    return _read(_stats_path(project_root))


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
    images = s.get("images", {})
    images_with_adds = [v for v in images.values() if v.get("annotations_added", 0) > 0]
    total_seconds = round(sum(v.get("session_seconds", 0.0) for v in images_with_adds), 2)
    total_adds = sum(int(v.get("annotations_added", 0)) for v in images.values())
    s["images_annotated"] = len(images_with_adds)
    s["total_annotations"] = total_adds
    s["total_time_seconds"] = total_seconds
    s["avg_seconds_per_annotation"] = (
        round(total_seconds / total_adds, 2) if total_adds > 0 else 0.0
    )
