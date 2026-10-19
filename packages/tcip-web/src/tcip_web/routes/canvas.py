"""Live canvas-state bridge: a pushed canvas state stored as the open project's records.

A heartbeat (``shapes`` ``null``) carries no geometry, and a full push carries the
display-resolved geometry. State is split across two records of the open
project:

  - the meta record (``canvas_meta_key``): replaced by every push.
  - the geometry record (``canvas_geometry_key``): written only by full pushes.

Each record is replaced whole; the geometry is valid only when its ``(image_path, tab)`` identity
matches the meta record. A push names the project it was built for by id; one naming any project
but the backend's open one answers 409 and writes nothing.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import ConfigDict

import tcip_store as ts
from tcip_annotation.viz import CanvasShape
from tcip_mcp.audit import now_iso
from tcip_mcp.identity import actor
from tcip_mcp.registry_paths import stored_path
from tcip_mcp.web_client import (
    CanvasGeometry,
    CanvasMeta,
    CanvasView,
    canvas_geometry_key,
    canvas_meta_key,
)
from tcip_web.state import store

router = APIRouter(prefix="/api/canvas", tags=["canvas"])


class CanvasStatePayload(CanvasView):
    """A canvas state push for the project ``project_id`` names.

    It is a heartbeat when ``shapes`` is ``None``, which leaves the geometry record untouched,
    else a full push.
    """

    model_config = ConfigDict(extra="forbid")

    project_id: str
    shapes: list[CanvasShape] | None


@router.post("/state")
def push_canvas_state(payload: CanvasStatePayload) -> dict:
    """Store the pushed canvas for the open project, stamped with the instant it arrived.

    A push stating a ``user`` that names no one is refused before anything is written.
    """
    person = actor(payload.user)
    project = store.admit(payload.project_id)
    root = str(project.root)
    image_path = stored_path(payload.image_path, project.root)
    now = now_iso()

    if payload.shapes is not None:
        # Geometry first, meta second: a reader pairing the new meta with the old geometry
        # sees an identity mismatch (stale), never a false match.
        geometry = CanvasGeometry(
            image_path=image_path,
            tab=payload.tab,
            shapes=payload.shapes,
            received_at=now,
        )
        ts.replace(canvas_geometry_key(root), geometry.model_dump(mode="json"))

    view = payload.model_dump(include=set(CanvasView.model_fields))
    meta = CanvasMeta.model_validate(
        {**view, "image_path": image_path, "user": person, "received_at": now}
    )
    ts.replace(canvas_meta_key(root), meta.model_dump(mode="json"))
    return {"status": "ok", "shapes_written": payload.shapes is not None}
