"""Live canvas-state bridge: the GUI pushes what it is rendering; the agent reads it back.

A heartbeat (image, viewport, classes, counts; ``shapes`` omitted) arrives on view/meta changes,
and the full display-resolved geometry only when shapes change. State is split across two files
under the open project's ``.tcip/state/``:

  - ``canvas_live.json``: the small meta document; overwritten atomically by every push.
  - ``canvas_shapes.json``: the geometry blob; written only by full pushes.

Each document is replaced whole; the geometry is valid only when its ``(image_path, tab)`` identity
matches the meta document. Both records declare ``durable=False``. A push names the project it was
built for by id; one naming any project but the backend's open one answers 409 and writes nothing.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict
import tcip_store as ts

from tcip_mcp.registry_paths import stored_path
from tcip_mcp.web_client import canvas_geometry_key, canvas_meta_key
from tcip_web.state import store

router = APIRouter(prefix="/api/canvas", tags=["canvas"])


class CanvasStatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The id of the project the GUI built this push for.
    project_id: str
    tab: str  # "annotate" | "review"
    image_path: str
    image: str
    img_width: int = 0
    img_height: int = 0
    viewport: Optional[dict] = None  # {x, y, w, h, scale} in image coords
    mode: Optional[str] = None
    active_subject: Optional[str] = None
    # Whether the Annotate cut tool is armed (sticky across a completed cut or a refusal alike).
    cut_armed: Optional[bool] = None
    dirty: Optional[bool] = None
    user: Optional[str] = None
    classes: list[dict] = []  # [{name, color}]
    legend: Optional[dict] = None  # e.g. review {tp, fp, fn, active} hex colors
    counts: Optional[dict] = None
    # None = heartbeat (geometry file untouched); a list = full geometry push.
    shapes: Optional[list[dict]] = None


@router.post("/state")
def push_canvas_state(payload: CanvasStatePayload) -> dict:
    project = store.admit(payload.project_id)
    root = str(project)
    image_path = stored_path(payload.image_path, project)
    now = time.time()

    if payload.shapes is not None:
        # Geometry first, meta second: a reader pairing the new meta with the old geometry
        # sees an identity mismatch (stale), never a false match.
        ts.replace(canvas_geometry_key(root), {
            "image_path": image_path,
            "tab": payload.tab,
            "shapes": payload.shapes,
            "received_at": now,
        })

    ts.replace(canvas_meta_key(root), {
        "received_at": now,
        "received_at_iso": datetime.now(timezone.utc).isoformat(),
        "tab": payload.tab,
        "image_path": image_path,
        "image": payload.image,
        "img_width": payload.img_width,
        "img_height": payload.img_height,
        "viewport": payload.viewport,
        "mode": payload.mode,
        "active_subject": payload.active_subject,
        "cut_armed": payload.cut_armed,
        "dirty": payload.dirty,
        "user": payload.user,
        "classes": payload.classes,
        "legend": payload.legend,
        "counts": payload.counts,
    })
    return {"status": "ok", "shapes_written": payload.shapes is not None}
