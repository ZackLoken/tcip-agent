"""Annotation label CRUD routes for the Annotate tab.

Reads / writes the canonical per-image label files using the shared
:mod:`tcip_annotation` engine. Paths are supplied by the caller so the
backend doesn't have to guess a dataset layout.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_annotation import (
    BBox,
    Polygon,
    boxes_from_polygons,
    parse_detect_labels,
    parse_segment_labels,
    write_detect_labels,
    write_segment_labels,
)
from tcip_annotation.utils import get_image_dimensions
from tcip_web.identity import resolve_user, user_id
from tcip_web.paths import assert_path_allowed
from tcip_web.state import PredictionReference, store

router = APIRouter(prefix="/api/annotate", tags=["annotate"])


class LabelsPayload(BaseModel):
    image_path: str
    detect_path: Optional[str] = None
    segment_path: Optional[str] = None


class BoxPayload(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = 0
    # Provenance round-trips through the client: a loaded shape carries its original
    # created_by back on save (keep-original-creator policy), so a re-save never
    # wholesale re-stamps existing labels to the current annotator. New shapes omit it.
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    accepted_by: Optional[str] = None
    accepted_at: Optional[str] = None


class PolygonPayload(BaseModel):
    points: list[list[float]]
    class_id: int = 0
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    accepted_by: Optional[str] = None
    accepted_at: Optional[str] = None


class SavePayload(BaseModel):
    image_path: str
    detect_path: Optional[str] = None
    segment_path: Optional[str] = None
    boxes: list[BoxPayload] = []
    polygons: list[PolygonPayload] = []
    # Project root for the audit trail (optional; skipped if absent).
    project_root: Optional[str] = None
    # The label-file mtime tokens the client loaded, keyed "detect"/"segment". When
    # present, a write is rejected (409) if the file changed underneath the client —
    # a concurrent agent or second browser tab — so its edits aren't clobbered.
    # Omit to skip the check (backward compatible for non-GUI callers).
    base_mtimes: Optional[dict[str, Optional[str]]] = None
    # GUI-set annotator identity (bare name, e.g. "zack"); stamped as created_by ("user:<name>").
    # Omitted by non-GUI callers -> backend falls back to the OS/env user.
    user: Optional[str] = None


def _image_dims(path: str) -> tuple[int, int]:
    try:
        p = assert_path_allowed(path)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc
    if not p.is_file():
        raise HTTPException(404, f"image not found: {path}")
    return get_image_dimensions(str(p))  # header-only (w, h); never decodes pixels


def _guard_label_path(path: Optional[str]) -> None:
    """Reject a client-supplied label path that escapes the configured image roots.

    Label read/write paths are attacker-controlled, so an exposed deployment
    (``TCIP_IMAGE_ROOTS`` set) must confine them exactly like image serving —
    otherwise ``write_detect_labels`` is an arbitrary file write/delete primitive.
    """
    if not path:
        return
    try:
        assert_path_allowed(path)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc


def _mtime_token(path: Optional[str]) -> Optional[str]:
    """Modification time (ns) of a label file as an opaque string token, or None if absent.

    A string, not an int: the ns value exceeds JavaScript's 2**53 exact-integer range, so a
    numeric token is silently rounded by the browser's JSON parse and every echo mismatches
    (the 409-on-every-save bug). The client never inspects it — it only echoes it back.
    """
    if not path:
        return None
    try:
        return str(os.stat(path).st_mtime_ns)
    except OSError:
        return None


def _audit_gui_write(payload: "SavePayload") -> None:
    """Append a GUI label-write to the project's audit log, mirroring @audited MCP tools.

    Best-effort and project-scoped (``<project_root>/.tcip/audit.jsonl``); a missing
    project_root or any I/O error just skips the entry — never fails the save.
    """
    if not payload.project_root:
        return
    try:
        from tcip_mcp.utils.atomic_io import append_jsonl

        append_jsonl(
            os.path.join(payload.project_root, ".tcip", "audit.jsonl"),
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": "gui_save_labels",
                "source": "gui",
                "arguments": {
                    "image_path": payload.image_path,
                    "detect_path": payload.detect_path,
                    "segment_path": payload.segment_path,
                    "n_boxes": len(payload.boxes),
                    "n_polygons": len(payload.polygons),
                },
                "status": "ok",
            },
        )
    except Exception:
        pass


@router.get("/labels")
def load_labels(
    image_path: str,
    detect_path: Optional[str] = None,
    segment_path: Optional[str] = None,
) -> dict:
    """Read existing labels for an image and return them in pixel coords."""
    w, h = _image_dims(image_path)
    _guard_label_path(detect_path)
    _guard_label_path(segment_path)
    boxes: list[dict] = []
    polygons: list[dict] = []

    if detect_path:
        parsed, _ = parse_detect_labels(detect_path, w, h)
        for b in parsed:
            boxes.append({
                "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                "class_id": b.class_id,
                "created_by": b.created_by, "created_at": b.created_at,
                "accepted_by": b.accepted_by, "accepted_at": b.accepted_at,
            })

    if segment_path:
        parsed_polys, _ = parse_segment_labels(segment_path, w, h)
        for poly in parsed_polys:
            polygons.append({
                "points": [list(p) for p in poly.points],
                "class_id": poly.class_id,
                "created_by": poly.created_by, "created_at": poly.created_at,
                "accepted_by": poly.accepted_by, "accepted_at": poly.accepted_at,
            })

    return {
        "image_path": image_path,
        "img_width": w,
        "img_height": h,
        "boxes": boxes,
        "polygons": polygons,
        # Version tokens the client echoes back on save for the lost-update guard.
        "base_mtimes": {
            "detect": _mtime_token(detect_path),
            "segment": _mtime_token(segment_path),
        },
    }


@router.post("/labels")
def save_labels(payload: SavePayload) -> dict:
    """Write labels for an image. Either detect_path or segment_path may be omitted.

    Empty box/polygon lists are written as 0-byte files (``keep_empty=True``) rather than
    deleted: an annotator who clears all annotations is recording a *confirmed negative*,
    and empty label files are valid negatives (CLAUDE.md invariant), not noise to prune.
    """
    w, h = _image_dims(payload.image_path)
    _guard_label_path(payload.detect_path)
    _guard_label_path(payload.segment_path)

    # Lost-update guard: reject if a label file changed since the client loaded it
    # (a concurrent agent write or a second browser tab). The client resolves the
    # 409 by reloading. Omitting base_mtimes skips the check.
    if payload.base_mtimes is not None:
        conflicts = [
            key
            for key, path in (("detect", payload.detect_path), ("segment", payload.segment_path))
            if path and _mtime_token(path) != payload.base_mtimes.get(key)
        ]
        if conflicts:
            raise HTTPException(
                409, {"error": "label file changed since it was loaded", "conflicts": conflicts}
            )

    # Human-authored GT: a round-tripped shape keeps its original created_by (the creator
    # stays the creator through edits); only shapes with no provenance — new ones — are
    # stamped to the current annotator. json_io persists all four fields natively.
    author = user_id(resolve_user(payload.user))
    now_iso = datetime.now(timezone.utc).isoformat()

    def _prov(shape) -> dict:
        # accepted_* only ride along on round-tripped shapes (created_by present) — a NEW shape
        # claiming acceptance would mint review sign-off that never happened.
        round_tripped = bool(shape.created_by)
        return {
            "created_by": shape.created_by or author,
            "created_at": shape.created_at if round_tripped else now_iso,
            "accepted_by": shape.accepted_by if round_tripped else None,
            "accepted_at": shape.accepted_at if round_tripped else None,
        }

    boxes = [
        BBox(b.x1, b.y1, b.x2, b.y2, class_id=b.class_id, **_prov(b))
        for b in payload.boxes
    ]
    polygons = [
        Polygon(points=[tuple(pt) for pt in p.points], class_id=p.class_id, **_prov(p))
        for p in payload.polygons
    ]

    # Detect is a derived view of segment: when polygons exist, the detect boxes are
    # their bounding boxes, so editing a polygon can't leave a stale box twin behind
    # (the two label files stay in lockstep). With no polygons, drawn boxes stand.
    # Each derived box inherits its source polygon's provenance (derived geometry keeps
    # the polygon's author — the valley-farm derived:user:* convention).
    detect_derived = bool(polygons) and payload.detect_path is not None
    if polygons:
        detect_boxes = boxes_from_polygons(polygons)
        with_points = [p for p in polygons if p.points]  # boxes_from_polygons skips empty ones
        for db, src in zip(detect_boxes, with_points):
            db.created_by, db.created_at = src.created_by, src.created_at
            db.accepted_by, db.accepted_at = src.accepted_by, src.accepted_at
    else:
        detect_boxes = boxes

    ok = True
    if payload.detect_path:
        try:
            os.makedirs(os.path.dirname(payload.detect_path) or ".", exist_ok=True)
            write_detect_labels(payload.detect_path, detect_boxes, w, h, keep_empty=True)
        except OSError as exc:
            raise HTTPException(500, f"could not write detect labels: {exc}") from exc

    if payload.segment_path:
        try:
            os.makedirs(os.path.dirname(payload.segment_path) or ".", exist_ok=True)
            write_segment_labels(payload.segment_path, polygons, w, h, keep_empty=True)
        except OSError as exc:
            raise HTTPException(500, f"could not write segment labels: {exc}") from exc

    _audit_gui_write(payload)

    return {
        "status": "ok" if ok else "partial",
        "image_path": payload.image_path,
        "detect_written": payload.detect_path is not None,
        "segment_written": payload.segment_path is not None,
        "detect_derived": detect_derived,
        "n_boxes": len(detect_boxes),
        "n_polygons": len(polygons),
        # New version tokens so the client can save again without a reload.
        "base_mtimes": {
            "detect": _mtime_token(payload.detect_path),
            "segment": _mtime_token(payload.segment_path),
        },
    }


class OpenImagePayload(BaseModel):
    image_path: str
    image_index: Optional[int] = None
    scale: Optional[float] = None
    offset_x: Optional[float] = None
    offset_y: Optional[float] = None
    mode: Optional[str] = None
    pred_reference: Optional[PredictionReference] = None


@router.post("/open")
async def open_image(payload: OpenImagePayload) -> dict:
    """Command the Annotate tab to load an image with an optional view + pred-reference.

    Used by the Review tab's Edit / FP-Accept flow to drive the Annotate tab
    with the same zoom and a dashed blue pred-reference overlay.
    """
    updates: dict = {"active_tab": "annotate"}
    view_update = {}
    if payload.scale is not None:
        view_update["scale"] = payload.scale
    if payload.offset_x is not None:
        view_update["offset_x"] = payload.offset_x
    if payload.offset_y is not None:
        view_update["offset_y"] = payload.offset_y
    if view_update:
        view = store.state.view.model_copy(update=view_update)
        updates["view"] = view
    if payload.mode is not None:
        updates["mode"] = payload.mode
    if payload.pred_reference is not None:
        updates["pred_reference"] = payload.pred_reference
    else:
        updates["pred_reference"] = None

    if payload.image_index is not None:
        dataset = store.state.dataset.model_copy(update={"current_image_index": payload.image_index})
        updates["dataset"] = dataset

    await store.mutate(updates)
    return {"status": "ok", "image_path": payload.image_path}
