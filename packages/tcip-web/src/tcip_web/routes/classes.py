"""Class registry routes — mirrors yolo-annotator's ``classes.json``.

Class ids are **scoped to a trait**: each trait keeps its own map at
``<project_root>/.tcip/state/classes/<trait>.json``, so ``catkin`` and ``bush`` can each use
class ``0`` for their own object without colliding. File format::

    {
        "0": {"name": "catkin", "color": "#FF0000"},
        "1": {"name": "bud",    "color": "#00FFFF"}
    }

A load with no ``trait`` reads the legacy project-global
``<project_root>/.tcip/state/classes.json`` (backward compatibility). Both ``name`` and
``color`` are optional on read; ``name`` falls back to ``f"class_{cid}"``, ``color`` is
auto-assigned from a high-contrast palette keyed by class id.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_mcp.utils.atomic_io import atomic_write_json, file_transaction, read_json
from tcip_mcp.workspace import is_valid_name

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


def _classes_path(project_root: str, trait: str | None = None) -> Path:
    state = Path(project_root) / ".tcip" / "state"
    if trait:
        return state / "classes" / f"{trait}.json"
    return state / "classes.json"  # legacy project-global map


def _read_registry(path: Path) -> ClassRegistry:
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries: list[ClassEntry] = []
    for k, v in sorted(raw.items(), key=lambda kv: int(kv[0])):
        cid = int(k)
        entries.append(
            ClassEntry(id=cid, name=v.get("name", f"class_{cid}"), color=v.get("color", auto_color(cid)))
        )
    return ClassRegistry(classes=entries)


def _class_ids_in_dir(d: Path) -> set[int]:
    ids: set[int] = set()
    for txt in d.glob("*.txt"):
        try:
            for line in txt.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts:
                    try:
                        ids.add(int(float(parts[0])))
                    except ValueError:
                        continue
        except OSError:
            continue
    return ids


def _derive_from_labels(detect_dir: str | None, segment_dir: str | None) -> list[ClassEntry]:
    """The class ids present in a trait's label files → a provisional registry (names default to
    ``class_<id>``). The safety net so a trait that has labels but no saved map never loads empty;
    a real map (with names) supersedes it once saved."""
    ids: set[int] = set()
    for d in (detect_dir, segment_dir):
        if d and Path(d).is_dir():
            ids |= _class_ids_in_dir(Path(d))
    return [ClassEntry(id=c, name=f"class_{c}", color=auto_color(c)) for c in sorted(ids)]


@router.get("/load")
def load_classes(
    project_root: str,
    trait: Optional[str] = None,
    annotations_detect_dir: Optional[str] = None,
    annotations_segment_dir: Optional[str] = None,
) -> ClassRegistry:
    """Load the class map for a trait. Ids are trait-scoped, so ``catkin`` and ``bush`` each keep
    their own ``0``. Resolution: the trait's saved map → else derived from the trait's labels
    (provisional ``class_<id>`` names) → else the legacy project-global map → else empty."""
    if trait and not is_valid_name(trait):
        raise HTTPException(400, f"invalid trait: {trait!r}")

    if trait:
        p = _classes_path(project_root, trait)
        if p.exists():
            try:
                return _read_registry(p)
            except Exception as exc:
                raise HTTPException(500, f"could not parse {p}: {exc}") from exc

    derived = _derive_from_labels(annotations_detect_dir, annotations_segment_dir)
    if derived:
        return ClassRegistry(classes=derived)

    legacy = _classes_path(project_root, None)
    if legacy.exists():
        try:
            return _read_registry(legacy)
        except Exception as exc:
            raise HTTPException(500, f"could not parse {legacy}: {exc}") from exc

    return ClassRegistry(classes=[])


class SaveClassesPayload(BaseModel):
    project_root: str
    trait: Optional[str] = None
    classes: list[ClassEntry]


@router.post("/save")
def save_classes(payload: SaveClassesPayload) -> dict:
    if payload.trait and not is_valid_name(payload.trait):
        raise HTTPException(400, f"invalid trait: {payload.trait!r}")
    path = _classes_path(payload.project_root, payload.trait)
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
    """Compute initial per-image status from the label files.

    A negative is intentional, not a side effect of an empty file: an image is a confirmed negative
    only when explicitly completed with no objects. Mapping: completed+objects -> complete;
    completed+empty -> negative; objects (not completed) -> partial; empty-or-missing -> unannotated.
    """
    det = Path(payload.annotations_detect_dir) if payload.annotations_detect_dir else None
    seg = Path(payload.annotations_segment_dir) if payload.annotations_segment_dir else None
    complete_set = set(payload.complete_override)

    statuses: dict[str, str] = {}
    for name in payload.image_list:
        stem = name.rsplit(".", 1)[0]
        has_any = False
        for label_dir in (det, seg):
            if not label_dir:
                continue
            txt = label_dir / f"{stem}.txt"
            if not txt.exists():
                continue
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
        if name in complete_set:
            statuses[name] = "complete" if has_any else "negative"
        elif has_any:
            statuses[name] = "partial"
        else:
            statuses[name] = "unannotated"

    return {"statuses": statuses}
