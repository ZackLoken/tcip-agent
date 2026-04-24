"""Annotation label CRUD routes for the Annotate tab.

Reads / writes YOLO-format label files using the shared
:mod:`tcip_annotation` engine. Paths are supplied by the caller so the
backend doesn't have to guess a dataset layout.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel

from tcip_annotation import (
    AnnotationState,
    BBox,
    Polygon,
    parse_detect_labels,
    parse_segment_labels,
    write_detect_labels,
    write_segment_labels,
)
from tcip_annotation.utils import auto_orient_image
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


class PolygonPayload(BaseModel):
    points: list[list[float]]
    class_id: int = 0


class SavePayload(BaseModel):
    image_path: str
    detect_path: Optional[str] = None
    segment_path: Optional[str] = None
    boxes: list[BoxPayload] = []
    polygons: list[PolygonPayload] = []


def _image_dims(path: str) -> tuple[int, int]:
    p = Path(path)
    if not p.is_file():
        raise HTTPException(404, f"image not found: {path}")
    with Image.open(p) as raw:
        im = auto_orient_image(raw)
        return im.size  # (w, h)


@router.get("/labels")
def load_labels(
    image_path: str,
    detect_path: Optional[str] = None,
    segment_path: Optional[str] = None,
) -> dict:
    """Read existing YOLO labels for an image and return them in pixel coords."""
    w, h = _image_dims(image_path)
    boxes: list[dict] = []
    polygons: list[dict] = []

    if detect_path:
        parsed, _ = parse_detect_labels(detect_path, w, h)
        for b in parsed:
            boxes.append({
                "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                "class_id": b.class_id,
            })

    if segment_path:
        parsed_polys, _ = parse_segment_labels(segment_path, w, h)
        for poly in parsed_polys:
            polygons.append({
                "points": [list(p) for p in poly.points],
                "class_id": poly.class_id,
            })

    return {
        "image_path": image_path,
        "img_width": w,
        "img_height": h,
        "boxes": boxes,
        "polygons": polygons,
    }


@router.post("/labels")
def save_labels(payload: SavePayload) -> dict:
    """Write YOLO labels for an image. Either detect_path or segment_path may be omitted."""
    w, h = _image_dims(payload.image_path)

    boxes = [
        BBox(b.x1, b.y1, b.x2, b.y2, class_id=b.class_id)
        for b in payload.boxes
    ]
    polygons = [
        Polygon(points=[tuple(pt) for pt in p.points], class_id=p.class_id)
        for p in payload.polygons
    ]

    ok = True
    if payload.detect_path:
        try:
            os.makedirs(os.path.dirname(payload.detect_path) or ".", exist_ok=True)
            write_detect_labels(payload.detect_path, boxes, w, h)
        except OSError as exc:
            raise HTTPException(500, f"could not write detect labels: {exc}") from exc

    if payload.segment_path:
        try:
            os.makedirs(os.path.dirname(payload.segment_path) or ".", exist_ok=True)
            write_segment_labels(payload.segment_path, polygons, w, h)
        except OSError as exc:
            raise HTTPException(500, f"could not write segment labels: {exc}") from exc

    return {
        "status": "ok" if ok else "partial",
        "image_path": payload.image_path,
        "detect_written": payload.detect_path is not None,
        "segment_written": payload.segment_path is not None,
        "n_boxes": len(boxes),
        "n_polygons": len(polygons),
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
