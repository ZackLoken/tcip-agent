"""Class registry routes — mirrors yolo-annotator's ``classes.json``.

Stored at ``<state_dir>/classes.json`` where ``state_dir`` is
``<project_root>/.tcip/state/`` by default. File format::

    {
        "0": {"name": "catkin", "color": "#FF0000"},
        "1": {"name": "bud",    "color": "#00FFFF"}
    }

Both ``name`` and ``color`` are optional on read; ``name`` falls back to
``f"class_{cid}"``, ``color`` is auto-assigned from a high-contrast palette
keyed by class id.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_mcp.utils.atomic_io import atomic_write_json, file_transaction, read_json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/classes", tags=["classes"])


# High-contrast default palette (same as yolo-annotator's DEFAULT_CLASS_COLORS)
DEFAULT_CLASS_COLORS = [
    "#FF0000",
    "#00FFFF",
    "#FFFF00",
    "#FF00FF",
    "#FF8C00",
    "#00FF00",
    "#FFFFFF",
    "#4169E1",
    "#FF69B4",
    "#00CED1",
]


def auto_color(class_id: int) -> str:
    return DEFAULT_CLASS_COLORS[class_id % len(DEFAULT_CLASS_COLORS)]


class ClassEntry(BaseModel):
    id: int
    name: str
    color: str


class ClassRegistry(BaseModel):
    classes: list[ClassEntry]


def _classes_path(project_root: str) -> Path:
    return Path(project_root) / ".tcip" / "state" / "classes.json"


@router.get("/load")
def load_classes(project_root: str) -> ClassRegistry:
    path = _classes_path(project_root)
    if not path.exists():
        return ClassRegistry(classes=[])
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(500, f"could not parse {path}: {exc}") from exc

    entries: list[ClassEntry] = []
    for k, v in sorted(raw.items(), key=lambda kv: int(kv[0])):
        cid = int(k)
        entries.append(
            ClassEntry(
                id=cid,
                name=v.get("name", f"class_{cid}"),
                color=v.get("color", auto_color(cid)),
            )
        )
    return ClassRegistry(classes=entries)


class SaveClassesPayload(BaseModel):
    project_root: str
    classes: list[ClassEntry]


@router.post("/save")
def save_classes(payload: SaveClassesPayload) -> dict:
    path = _classes_path(payload.project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, dict] = {}
    for entry in payload.classes:
        data[str(entry.id)] = {"name": entry.name, "color": entry.color}
    try:
        atomic_write_json(path, data)
    except OSError as exc:
        raise HTTPException(500, f"could not write {path}: {exc}") from exc
    return {"status": "ok", "n_classes": len(payload.classes)}


@router.get("/auto_color/{class_id}")
def get_auto_color(class_id: int) -> dict:
    return {"class_id": class_id, "color": auto_color(class_id)}


# ── Per-image status (used by Complete checkbox + status filter) ─────────


VALID_STATUSES = ("complete", "partial", "negative", "unannotated")


class ImageStatusPayload(BaseModel):
    project_root: str
    image_name: str
    status: str  # "complete" | "partial" | "negative" | "unannotated"


def _image_status_path(project_root: str) -> Path:
    return Path(project_root) / ".tcip" / "state" / "image_status.json"


@router.get("/image_status")
def get_image_status(project_root: str) -> dict:
    path = _image_status_path(project_root)
    if not path.exists():
        return {"statuses": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"statuses": {}}
    return {"statuses": raw}


@router.post("/image_status")
def set_image_status(payload: ImageStatusPayload) -> dict:
    if payload.status not in VALID_STATUSES:
        raise HTTPException(400, f"invalid status: {payload.status}")
    path = _image_status_path(payload.project_root)
    with file_transaction(path):
        raw = read_json(path, default={})
        if not isinstance(raw, dict):
            raw = {}
        raw[payload.image_name] = payload.status
        atomic_write_json(path, {k: raw[k] for k in sorted(raw)})
    return {"status": "ok"}


class ImageStatusBulkPayload(BaseModel):
    project_root: str
    statuses: dict[str, str]  # image_name → status


@router.post("/image_status/bulk")
def set_image_status_bulk(payload: ImageStatusBulkPayload) -> dict:
    path = _image_status_path(payload.project_root)
    with file_transaction(path):
        raw = read_json(path, default={})
        if not isinstance(raw, dict):
            raw = {}
        for name, st in payload.statuses.items():
            if st in VALID_STATUSES:
                raw[name] = st
        atomic_write_json(path, {k: raw[k] for k in sorted(raw)})
    return {"status": "ok", "n": len(payload.statuses)}


class DerivePayload(BaseModel):
    project_root: str
    annotations_detect_dir: Optional[str] = None
    annotations_segment_dir: Optional[str] = None
    image_list: list[str]
    complete_override: list[str] = []


@router.post("/image_status/derive")
def derive_image_status(payload: DerivePayload) -> dict:
    """Compute initial per-image status based on whether label files exist
    and contain any annotations. Images in ``complete_override`` are forced
    to ``complete``.

    A label file that exists but is empty is a **confirmed negative** (the
    annotator reviewed the image and recorded no objects) — distinct from an
    image with no label file at all (never looked at):

      - any non-empty label line -> ``partial``
      - label file exists but empty -> ``negative``
      - no label file -> ``unannotated``
    """
    det = Path(payload.annotations_detect_dir) if payload.annotations_detect_dir else None
    seg = Path(payload.annotations_segment_dir) if payload.annotations_segment_dir else None
    complete_set = set(payload.complete_override)

    statuses: dict[str, str] = {}
    for name in payload.image_list:
        if name in complete_set:
            statuses[name] = "complete"
            continue
        stem = name.rsplit(".", 1)[0]
        has_any = False
        file_exists = False
        for label_dir in (det, seg):
            if not label_dir:
                continue
            txt = label_dir / f"{stem}.txt"
            if not txt.exists():
                continue
            file_exists = True
            try:
                with txt.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            has_any = True
                            break
            except Exception:
                pass
            if has_any:
                break
        if has_any:
            statuses[name] = "partial"
        elif file_exists:
            statuses[name] = "negative"
        else:
            statuses[name] = "unannotated"

    return {"statuses": statuses}
