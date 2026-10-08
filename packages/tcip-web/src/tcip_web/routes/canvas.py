"""Live canvas-state bridge: the GUI pushes what it is rendering; the agent reads it back.

A heartbeat (image, viewport, classes, counts; ``shapes`` omitted) arrives on view/meta changes,
and the full display-resolved geometry only when shapes change. State is split across two records
of the open project:

  - the meta record (``canvas_meta_key``): replaced by every push.
  - the geometry record (``canvas_geometry_key``): written only by full pushes.

Each record is replaced whole; the geometry is valid only when its ``(image_path, tab)`` identity
matches the meta record. A push names the project it was built for by id; one naming any project
but the backend's open one answers 409 and writes nothing.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict
import tcip_store as ts

from tcip_mcp.audit import now_iso
from tcip_mcp.identity import actor
from tcip_mcp.registry_paths import stored_path
from tcip_mcp.web_client import ActiveTab, canvas_geometry_key, canvas_meta_key
from tcip_web.state import store

router = APIRouter(prefix="/api/canvas", tags=["canvas"])


class CanvasStatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The id of the project the GUI built this push for.
    project_id: str
    tab: ActiveTab
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
    user: str
    classes: list[dict] = []  # [{name, color}]
    counts: Optional[dict] = None
    # None = heartbeat (geometry file untouched); a list = full geometry push.
    shapes: Optional[list[dict]] = None


@router.post("/state")
def push_canvas_state(payload: CanvasStatePayload) -> dict:
    """Store the pushed canvas for the open project, stamped with the instant it arrived; a push
    stating a ``user`` that names no one is refused before anything is written."""
    person = actor(payload.user)
    project = store.admit(payload.project_id)
    root = str(project.root)
    image_path = stored_path(payload.image_path, project.root)
    now = now_iso()

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
        "user": person,
        "classes": payload.classes,
        "counts": payload.counts,
    })
    return {"status": "ok", "shapes_written": payload.shapes is not None}
