"""Live canvas-state bridge: the GUI pushes what it is rendering; the agent reads it back.

The frontend posts here on a hybrid cadence: a tiny heartbeat (image, viewport, classes,
counts; ``shapes`` omitted) on view/meta changes, and the full display-resolved geometry only
when shapes actually change. State is split across two files under
``<project_root>/.tcip/state/`` so the cadences never contend:

  - ``canvas_live.json``: the small meta document; overwritten atomically by every push.
  - ``canvas_shapes.json``: the geometry blob; written only by full pushes.

There is no read-modify-write merge, so nothing can interleave and resurrect stale geometry:
each document is replaced whole, and the reader (``capture_live_canvas``) treats the geometry
as valid only when its ``(image_path, tab)`` identity matches the meta document: a heartbeat
for a different image/tab implicitly invalidates stale shapes.

Both records declare ``durable=False``, so a push returns after the atomic replace and before
any flush: this is ephemeral live-view state re-pushed every heartbeat (as often as every
debounce cycle), not durable review/annotation history, and a crash losing the last push costs
nothing, the next push repaints it.

The web backend pins one platform root per process; a push lands only under that pinned root
(``_pinned_root_matching``), never under whatever the browser's own payload claims, since
``capture_live_canvas`` always reads the pinned root and a push under any other one would
write a file it never looks at.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from tcip_store import replace

from tcip_mcp.project_paths import project_root as platform_root
from tcip_mcp.web_client import canvas_geometry_key, canvas_meta_key
from tcip_web.paths import assert_path_allowed

router = APIRouter(prefix="/api/canvas", tags=["canvas"])


class CanvasStatePayload(BaseModel):
    project_root: str
    tab: str  # "annotate" | "review"
    image_path: str
    image: str
    img_width: int = 0
    img_height: int = 0
    viewport: Optional[dict] = None  # {x, y, w, h, scale} in image coords
    mode: Optional[str] = None
    active_subject: Optional[str] = None
    dirty: Optional[bool] = None
    user: Optional[str] = None
    classes: list[dict] = []  # [{name, color}]
    legend: Optional[dict] = None  # e.g. review {tp, fp, fn, active} hex colors
    counts: Optional[dict] = None
    # None = heartbeat (geometry file untouched); a list = full geometry push.
    shapes: Optional[list[dict]] = None


def _guard_project_root(project_root: str) -> str:
    """Confine a client-supplied project_root to the allowed roots and hand back its resolved
    spelling; 403 on escape. Confinement only: whether it is also the root this push must land
    under is :func:`_pinned_root_matching` below.
    """
    try:
        return str(assert_path_allowed(project_root))
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc


def _pinned_root_matching(stated_root: str) -> str:
    """This process's pinned platform root, once ``stated_root`` is confirmed to be it.

    ``capture_live_canvas`` (the only reader of what this route writes) anchors to
    ``tcip_mcp.project_paths.project_root()``, this process's own pinned root, never to a
    caller-supplied value; a push landing under any other root would write a file the reader
    never looks at. Compared by filesystem identity, not by string equality, the same rail
    ``results._open_project_root`` uses for its own open-project check: a resolved payload path
    and an unresolved pinned one can name the same directory without being the same string.
    """
    root = str(platform_root())
    try:
        same = os.path.samefile(stated_root, root)
    except OSError as exc:
        raise HTTPException(
            403, f"project_root {stated_root} cannot be compared with the pinned platform "
                 f"root {root}: {exc}") from exc
    if not same:
        raise HTTPException(
            403, f"project_root {stated_root} is not this process's pinned platform root "
                 f"{root}; capture_live_canvas reads live canvas state from the pinned root, "
                 "so a push under a different one would never be seen. Adopt this project "
                 "(set_active_project) before pushing canvas state for it.")
    return root


@router.post("/state")
def push_canvas_state(payload: CanvasStatePayload) -> dict:
    stated_root = _guard_project_root(payload.project_root)
    project_root = _pinned_root_matching(stated_root)
    now = time.time()

    if payload.shapes is not None:
        # Geometry first, meta second: a reader pairing the new meta with the old geometry
        # sees an identity mismatch (stale), never a false match.
        replace(canvas_geometry_key(project_root), {
            "image_path": payload.image_path,
            "tab": payload.tab,
            "shapes": payload.shapes,
            "received_at": now,
        })

    replace(canvas_meta_key(project_root), {
        "received_at": now,
        "received_at_iso": datetime.now(timezone.utc).isoformat(),
        "project_root": project_root,
        "tab": payload.tab,
        "image_path": payload.image_path,
        "image": payload.image,
        "img_width": payload.img_width,
        "img_height": payload.img_height,
        "viewport": payload.viewport,
        "mode": payload.mode,
        "active_subject": payload.active_subject,
        "dirty": payload.dirty,
        "user": payload.user,
        "classes": payload.classes,
        "legend": payload.legend,
        "counts": payload.counts,
    })
    return {"status": "ok", "shapes_written": payload.shapes is not None}
