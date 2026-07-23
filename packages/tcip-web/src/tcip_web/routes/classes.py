"""Class registry routes.

The dataset's class registry is a single nested ``<dataset_root>/classes.json`` describing every
subject, its attributes, and their value names — never integer ids or colors (a label references
these names; an id is a per-training-run artifact and a color is GUI-local). Shape::

    {
      "bush":   {"description": "one hazelnut bush crown"},
      "catkin": {"description": "a hazelnut catkin",
                 "attributes": {"elongation": {"type": "categorical",
                                               "values": ["dormant", "elongated"]}}}
    }

Read/written through :mod:`tcip_mcp.class_registry` (the one registry authority), so the GUI and the
agent tools agree by construction. The registry travels with the image set — a name-based label is
undecodable without it.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_mcp.utils.atomic_io import atomic_write_json, file_transaction, read_json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/classes", tags=["classes"])


# ── Label-JSON memo (mtime-keyed, bounded) ────────────────────────────────
# derive_image_status and the label-derived subject list both re-parse label JSONs on every call —
# a dataset-selection change or a /load with no saved registry re-scans the same files repeatedly.
# Memoize per (path, mtime_ns) so an unchanged file is parsed once; a write bumps mtime_ns and the
# next read re-parses it.
_LABEL_JSON_CACHE_MAX = 4096
_label_json_cache: "OrderedDict[str, tuple[int, object]]" = OrderedDict()


def _cached_label_json(path: Path) -> object:
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return None
    key = str(path)
    cached = _label_json_cache.get(key)
    if cached is not None and cached[0] == mtime_ns:
        _label_json_cache.move_to_end(key)
        return cached[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    _label_json_cache[key] = (mtime_ns, data)
    _label_json_cache.move_to_end(key)
    if len(_label_json_cache) > _LABEL_JSON_CACHE_MAX:
        _label_json_cache.popitem(last=False)
    return data


def _resolve_dataset_root(dataset_root: str | None, annotations_dir: str | None) -> str | None:
    """The dataset root, taken from ``dataset_root`` or derived from a per-image label dir path."""
    if dataset_root:
        return dataset_root
    from tcip_mcp.dataset_layout import dataset_root_of

    if annotations_dir and (root := dataset_root_of(annotations_dir)) is not None:
        return str(root)
    return None


def _subjects_in_dir(d: Path) -> set[str]:
    """Distinct subject names present in a dir's per-image label files."""
    subjects: set[str] = set()
    for jf in d.glob("*.json"):
        data = _cached_label_json(jf)
        if not isinstance(data, dict):
            continue
        for o in data.get("annotations") or []:
            if isinstance(o, dict) and isinstance(o.get("subject"), str) and o["subject"]:
                subjects.add(o["subject"])
    return subjects


@router.get("/load")
def load_classes(
    project_root: str,
    dataset_root: Optional[str] = None,
    annotations_dir: Optional[str] = None,
) -> dict:
    """Load the dataset's nested class registry.

    Resolution: the dataset's saved ``classes.json`` -> else a provisional registry of the subjects
    actually present in the labels (detection-only, no attributes) -> else empty. Returns
    ``{"subjects": <nested registry mapping>}``.
    """
    from tcip_mcp.class_registry import (
        ClassRegistry,
        RegistryError,
        Subject,
        read_registry,
        registry_to_dict,
    )
    from tcip_mcp.dataset_layout import classes_path

    root = _resolve_dataset_root(dataset_root, annotations_dir)
    if root:
        p = classes_path(root)
        if p.exists():
            try:
                return {"subjects": registry_to_dict(read_registry(p))}
            except (OSError, RegistryError) as exc:
                raise HTTPException(500, f"could not parse {p}: {exc}") from exc

    if annotations_dir and Path(annotations_dir).is_dir():
        subjects = _subjects_in_dir(Path(annotations_dir))
        if subjects:
            reg = ClassRegistry(subjects=tuple(Subject(name=s) for s in sorted(subjects)))
            return {"subjects": registry_to_dict(reg)}

    return {"subjects": {}}


class SaveClassesPayload(BaseModel):
    project_root: str
    subjects: dict  # the nested registry mapping (subjects -> attributes -> values)
    dataset_root: Optional[str] = None
    annotations_dir: Optional[str] = None


@router.post("/save")
def save_classes(payload: SaveClassesPayload) -> dict:
    from tcip_mcp.class_registry import RegistryError, registry_from_dict, write_registry
    from tcip_mcp.dataset_layout import classes_path

    root = _resolve_dataset_root(payload.dataset_root, payload.annotations_dir)
    if not root:
        raise HTTPException(400, "cannot locate the dataset to save the class registry into; "
                                 "pass dataset_root or an annotations dir")
    try:
        registry = registry_from_dict(payload.subjects)
    except RegistryError as exc:
        raise HTTPException(400, f"invalid class registry: {exc}") from exc
    path = classes_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_registry(path, registry)
    except OSError as exc:
        raise HTTPException(500, f"could not write {path}: {exc}") from exc
    return {"status": "ok", "n_subjects": len(registry.subjects), "classes_path": str(path)}


# ── Per-image status (used by Complete checkbox + status filter) ─────────


VALID_STATUSES = ("complete", "partial", "negative", "unannotated")


class ImageStatusPayload(BaseModel):
    project_root: str
    image_name: str
    status: str  # "complete" | "partial" | "negative" | "unannotated"
    subject: str | None = None  # the object a Complete is scoped to (not necessarily a trait)
    date: str | None = None


def _image_status_path(project_root: str) -> Path:
    return Path(project_root) / ".tcip" / "state" / "image_status.json"


def _load_status_store(path: Path) -> dict[str, dict[str, str]]:
    """The status store, normalized. One definition, shared with doctor.py."""
    from tcip_mcp.dataset_layout import normalize_status_store

    return normalize_status_store(read_json(path, default={}))


def _bucket(subject: str | None, date: str | None) -> str:
    from tcip_mcp.dataset_layout import status_bucket

    return status_bucket(subject or "", date)


@router.get("/image_status")
def get_image_status(project_root: str, subject: str | None = None,
                     date: str | None = None) -> dict:
    """Statuses for one subject/date."""
    path = _image_status_path(project_root)
    if not path.exists():
        return {"statuses": {}}
    return {"statuses": _load_status_store(path).get(_bucket(subject, date), {})}


@router.post("/image_status")
def set_image_status(payload: ImageStatusPayload) -> dict:
    if payload.status not in VALID_STATUSES:
        raise HTTPException(400, f"invalid status: {payload.status}")
    path = _image_status_path(payload.project_root)
    bucket = _bucket(payload.subject, payload.date)
    with file_transaction(path):
        store = _load_status_store(path)
        store.setdefault(bucket, {})[payload.image_name] = payload.status
        atomic_write_json(path, {k: dict(sorted(store[k].items())) for k in sorted(store)})
    return {"status": "ok"}


class ImageStatusBulkPayload(BaseModel):
    project_root: str
    statuses: dict[str, str]  # image_name → status
    subject: str | None = None
    date: str | None = None


@router.post("/image_status/bulk")
def set_image_status_bulk(payload: ImageStatusBulkPayload) -> dict:
    path = _image_status_path(payload.project_root)
    bucket = _bucket(payload.subject, payload.date)
    with file_transaction(path):
        store = _load_status_store(path)
        target = store.setdefault(bucket, {})
        for name, st in payload.statuses.items():
            if st in VALID_STATUSES:
                target[name] = st
        atomic_write_json(path, {k: dict(sorted(store[k].items())) for k in sorted(store)})
    return {"status": "ok", "n": len(payload.statuses)}


class DerivePayload(BaseModel):
    project_root: str
    annotations_dir: Optional[str] = None
    subject: Optional[str] = None
    image_list: list[str]
    complete_override: list[str] = []


@router.post("/image_status/derive")
def derive_image_status(payload: DerivePayload) -> dict:
    """Compute initial per-image status from the per-image label files.

    A negative is intentional, not a side effect of an empty file: an image is a confirmed negative
    only when explicitly completed with no annotations. Mapping: completed+objects -> complete;
    completed+empty -> negative; objects (not completed) -> partial; empty-or-missing -> unannotated.
    When a ``subject`` is given, only annotations of that subject count (per-subject scoping).
    """
    adir = Path(payload.annotations_dir) if payload.annotations_dir else None
    complete_set = set(payload.complete_override)

    statuses: dict[str, str] = {}
    for name in payload.image_list:
        stem = name.rsplit(".", 1)[0]
        has_any = False
        if adir:
            data = _cached_label_json(adir / f"{stem}.json")
            if isinstance(data, dict):
                anns = data.get("annotations") or []
                if payload.subject:
                    has_any = any(
                        isinstance(o, dict) and o.get("subject") == payload.subject for o in anns
                    )
                else:
                    has_any = bool(anns)
        if name in complete_set:
            statuses[name] = "complete" if has_any else "negative"
        elif has_any:
            statuses[name] = "partial"
        else:
            statuses[name] = "unannotated"

    return {"statuses": statuses}
