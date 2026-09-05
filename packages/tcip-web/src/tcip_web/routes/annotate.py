"""Annotation label CRUD routes for the Annotate tab.

Reads / writes the canonical per-image label file (one JSON per image, holding every subject's
annotations by name) via :mod:`tcip_annotation.json_io`. The label path is supplied by the caller
so the backend doesn't have to guess a dataset layout.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from tcip_annotation import BBox, Point, Polygon
from tcip_annotation.json_io import (
    UnreadableLabelDocument,
    annotation_from_payload,
    authorship_of,
    read_annotations_versioned,
    write_annotations,
)
from tcip_annotation.state import Annotation
from tcip_mcp.pipelines.image_utils import (
    AmbiguousImageStem, image_dimensions, resolve_image_source,
)
from tcip_store import Version, VersionConflict
from tcip_web.identity import resolve_user, user_id
from tcip_web.paths import assert_path_allowed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/annotate", tags=["annotate"])


class AnnotationPayload(BaseModel):
    """One annotation: its ``subject`` (the object it is about), a geometry (a box, a polygon, a
    point, or none of them for an image-level label), and its attribute values by name."""

    subject: str
    bbox: Optional[list[float]] = None          # [x1, y1, x2, y2], pixel
    points: Optional[list[list[float]]] = None  # single-ring polygon vertices, pixel: a shape the
                                                 # canvas itself drew/edited by hand
    rings: Optional[list[list[list[float]]]] = None  # multi-ring polygon (a loaded, unedited
                                                       # occlusion-split instance_seg shape round-
                                                       # tripping through save): takes precedence
                                                       # over `points` when both are present
    point: Optional[list[float]] = None         # [x, y], pixel: one placed prompt / keypoint.
                                                 # Singular, deliberately distinct from `points`:
                                                 # a point and a one-vertex contour are not the
                                                 # same geometry.
    attributes: dict[str, str] = {}
    # Provenance round-trips through the client: a loaded shape carries its original created_by
    # back on save (keep-original-creator policy), so a re-save never wholesale re-stamps existing
    # labels to the current annotator. New shapes omit it.
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    accepted_by: Optional[str] = None
    accepted_at: Optional[str] = None


class SavePayload(BaseModel):
    image_path: str
    # Non-empty: a save with nowhere to write can never succeed, so it is refused (422) rather
    # than accepted as a no-op.
    label_path: str = Field(min_length=1)
    annotations: list[AnnotationPayload] = []
    # Accepted for wire compatibility and read nowhere in this module; the save's audit line
    # files under the label's own dataset root (_guarded_audit_root), never this field.
    project_root: Optional[str] = None
    # The label document's version token as the client loaded it: with one, the save is a
    # compare-and-set and a 409 says it changed underneath. Omit to skip the comparison.
    base_mtime: Optional[str] = None
    # GUI-set annotator identity (bare name, e.g. "breeder"); stamped as created_by ("user:<name>").
    # Omitted by non-GUI callers -> backend falls back to the OS/env user.
    user: Optional[str] = None


def _image_dims(path: str) -> tuple[int, int]:
    try:
        p = assert_path_allowed(path)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc
    if not p.is_file():
        raise HTTPException(404, f"image not found: {path}")
    # Channel-aware: resolve_image_source folds a `.bandgroup` manifest (or a genuinely
    # multi-band raster) into the frame image_dimensions measures, not a bare PIL header read.
    try:
        return image_dimensions(resolve_image_source(p.parent, p.stem))
    except AmbiguousImageStem as exc:
        raise HTTPException(400, str(exc)) from exc


def _guard_label_path(path: Optional[str]) -> Optional[str]:
    """Confine a client-supplied label path and hand back its resolved spelling, or None.

    Label read/write paths are attacker-controlled and ``write_annotations`` would otherwise be
    an arbitrary file write/delete primitive, so every read and write uses the path this returns.
    """
    if not path:
        return None
    try:
        return str(assert_path_allowed(path))
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc


def _guarded_audit_root(label_path: Optional[str]) -> Optional[str]:
    """The dataset root a label write is audited under, confined before anything is written.

    Labels travel with their dataset, so their trail belongs beside them rather than in whichever
    project happened to open the dataset. A label path outside a dataset tree names no such log,
    so the write proceeds unaudited and says so; a dataset root the allow-set does not admit
    refuses the write before it happens.
    """
    from tcip_mcp.dataset_layout import dataset_root_of

    root = dataset_root_of(label_path) if label_path else None
    if root is None:
        logger.warning("no dataset root for %s; label write not audited", label_path)
        return None
    try:
        return str(assert_path_allowed(root))
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc


def _ann_dict(a: Annotation) -> dict:
    """Serialize an :class:`Annotation` for the canvas (pixel coords + provenance).

    ``rings`` (not ``points``) for a polygon: a loaded GT annotation can be a multi-ring
    occlusion-split instance_seg prediction accepted through Review, so the canvas always receives
    every ring rather than silently only the first. The canvas itself still only ever *draws* a
    single ring by hand (see ``AnnotationPayload.points`` below, the save side). ``point`` is the
    singular ``[x, y]`` of a placed prompt / keypoint, the same key the on-disk schema uses.
    ``authorship`` is this load response's own field, derived from the four provenance fields
    through :func:`authorship_of`; the label document itself carries no such field.
    """
    out: dict = {"subject": a.subject, "attributes": dict(a.attributes)}
    geom = a.geometry
    if isinstance(geom, Polygon):
        out["rings"] = [[list(pt) for pt in ring] for ring in geom.rings]
    elif isinstance(geom, BBox):
        out["bbox"] = [geom.x1, geom.y1, geom.x2, geom.y2]
    elif isinstance(geom, Point):
        out["point"] = [geom.x, geom.y]
    if a.score is not None:
        out["score"] = a.score
    out["created_by"] = a.created_by
    out["created_at"] = a.created_at
    out["accepted_by"] = a.accepted_by
    out["accepted_at"] = a.accepted_at
    out["authorship"] = authorship_of(a)
    return out


def _audit_gui_write(payload: "SavePayload", label_path: str, root: str) -> None:
    """Record a GUI label-write in that dataset's own audit log.

    ``root`` is the dataset root :func:`_guarded_audit_root` admitted before the write.
    """
    from tcip_mcp.audit import record_event

    record_event(
        "gui_save_labels",
        {
            "image_path": payload.image_path,
            "label_path": label_path,
            "n_annotations": len(payload.annotations),
        },
        source="gui",
        scope=root,
    )


@router.get("/labels")
def load_labels(image_path: str, label_path: Optional[str] = None) -> dict:
    """Read existing labels for an image and return them in pixel coords."""
    w, h = _image_dims(image_path)
    label_path = _guard_label_path(label_path)
    annotations: list[dict] = []
    token: Optional[str] = None
    if label_path:
        try:
            stored, version = read_annotations_versioned(label_path)
        except UnreadableLabelDocument as exc:
            raise HTTPException(400, str(exc)) from exc
        annotations = [_ann_dict(a) for a in stored]
        token = version.token
    return {
        "image_path": image_path,
        "img_width": w,
        "img_height": h,
        "annotations": annotations,
        # Version token the client echoes back on save for the lost-update guard.
        "base_mtime": token,
    }


@router.post("/labels")
def save_labels(payload: SavePayload) -> dict:
    """Write labels for an image to its single per-image JSON file.

    An empty annotation list is written as ``{"annotations": []}`` (``keep_empty=True``) rather than
    deleted, so clearing all annotations keeps the record instead of erasing it. That record is not a
    negative on its own: it trains as one only once the breeder marks the image Complete
    (``image_status.json``); until then it reads as unannotated (CLAUDE.md's negative invariant).
    """
    w, h = _image_dims(payload.image_path)
    label_path = _guard_label_path(payload.label_path)
    assert label_path is not None  # payload.label_path is non-empty; the guard only confines it
    audit_root = _guarded_audit_root(label_path)

    author = user_id(resolve_user(payload.user))
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        annotations = [
            annotation_from_payload(ap.model_dump(), author=author, now=now_iso)
            for ap in payload.annotations
        ]
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    # The lost-update guard, inside the store's own lock: with a token the write is refused
    # unless the stored document still matches it, and the client resolves the 409 by reloading.
    expect = Version(payload.base_mtime) if payload.base_mtime is not None else None
    try:
        version = write_annotations(label_path, annotations, w, h, keep_empty=True, expect=expect)
    except VersionConflict as exc:
        raise HTTPException(409, {"error": "label file changed since it was loaded"}) from exc
    except OSError as exc:
        raise HTTPException(500, f"could not write labels: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    token = version.token if version is not None else None
    if audit_root is not None:
        _audit_gui_write(payload, label_path, audit_root)

    return {
        "status": "ok",
        "image_path": payload.image_path,
        "n_annotations": len(annotations),
        # New version token so the client can save again without a reload.
        "base_mtime": token,
    }
