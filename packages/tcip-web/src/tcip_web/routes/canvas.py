"""Live canvas-state bridge: the GUI pushes what it is rendering; the agent reads it back.

The frontend posts here on a hybrid cadence — a tiny heartbeat (image, viewport, classes,
counts; ``shapes`` omitted) on view/meta changes, and the full display-resolved geometry only
when shapes actually change. State is split across two files under
``<project_root>/.tcip/state/`` so the cadences never contend:

  - ``canvas_live.json``   — the small meta document; overwritten atomically by every push.
  - ``canvas_shapes.json`` — the geometry blob; written only by full pushes.

There is no read-modify-write merge (so no lock and no interleaving that can resurrect stale
geometry): each file is written atomically, and the reader (``visualize_canvas``) treats the
geometry as valid only when its ``(image_path, tab)`` identity matches the meta document —
a heartbeat for a different image/tab implicitly invalidates stale shapes.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from tcip_mcp.utils.atomic_io import atomic_write_json

router = APIRouter(prefix="/api/canvas", tags=["canvas"])


class CanvasStatePayload(BaseModel):
    schema_version: int = 1
    project_root: str
    tab: str  # "annotate" | "review"
    image_path: str
    image: str
    img_width: int = 0
    img_height: int = 0
    viewport: Optional[dict] = None  # {x, y, w, h, scale} in image coords
    mode: Optional[str] = None
    active_class: Optional[int] = None
    dirty: Optional[bool] = None
    user: Optional[str] = None
    classes: list[dict] = []  # [{id, name, color}]
    legend: Optional[dict] = None  # e.g. review {tp, fp, fn, active} hex colors
    counts: Optional[dict] = None
    # None = heartbeat (geometry file untouched); a list = full geometry push.
    shapes: Optional[list[dict]] = None


def meta_path(project_root: str) -> Path:
    return Path(project_root) / ".tcip" / "state" / "canvas_live.json"


def shapes_path(project_root: str) -> Path:
    return Path(project_root) / ".tcip" / "state" / "canvas_shapes.json"


@router.post("/state")
def push_canvas_state(payload: CanvasStatePayload) -> dict:
    now = time.time()
    mp = meta_path(payload.project_root)
    mp.parent.mkdir(parents=True, exist_ok=True)

    if payload.shapes is not None:
        # Geometry first, meta second: a reader pairing the new meta with the old geometry
        # sees an identity mismatch (stale), never a false match.
        atomic_write_json(shapes_path(payload.project_root), {
            "image_path": payload.image_path,
            "tab": payload.tab,
            "shapes": payload.shapes,
            "received_at": now,
        })

    atomic_write_json(mp, {
        "schema_version": payload.schema_version,
        "received_at": now,
        "received_at_iso": datetime.now(timezone.utc).isoformat(),
        "project_root": payload.project_root,
        "tab": payload.tab,
        "image_path": payload.image_path,
        "image": payload.image,
        "img_width": payload.img_width,
        "img_height": payload.img_height,
        "viewport": payload.viewport,
        "mode": payload.mode,
        "active_class": payload.active_class,
        "dirty": payload.dirty,
        "user": payload.user,
        "classes": payload.classes,
        "legend": payload.legend,
        "counts": payload.counts,
    })
    return {"status": "ok", "shapes_written": payload.shapes is not None}
