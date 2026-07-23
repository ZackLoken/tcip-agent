"""Live canvas-state bridge: the GUI pushes what it is rendering; the agent reads it back.

The frontend posts here on a hybrid cadence — a tiny heartbeat (image, viewport, classes,
counts; ``shapes`` omitted) on view/meta changes, and the full display-resolved geometry only
when shapes actually change. State is split across two files under
``<project_root>/.tcip/state/`` so the cadences never contend:

  - ``canvas_live.json``   — the small meta document; overwritten atomically by every push.
  - ``canvas_shapes.json`` — the geometry blob; written only by full pushes.

There is no read-modify-write merge (so no lock and no interleaving that can resurrect stale
geometry): each file is written atomically, and the reader (``capture_live_canvas``) treats the
geometry as valid only when its ``(image_path, tab)`` identity matches the meta document —
a heartbeat for a different image/tab implicitly invalidates stale shapes.

Both writes skip ``fsync``: this is ephemeral live-view state re-pushed every heartbeat (as
often as every debounce cycle), not durable review/annotation history — a crash losing the
last push costs nothing, the next push repaints it.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_web.paths import assert_path_allowed

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
    active_subject: Optional[str] = None
    dirty: Optional[bool] = None
    user: Optional[str] = None
    classes: list[dict] = []  # [{name, color}]
    legend: Optional[dict] = None  # e.g. review {tp, fp, fn, active} hex colors
    counts: Optional[dict] = None
    # None = heartbeat (geometry file untouched); a list = full geometry push.
    shapes: Optional[list[dict]] = None


def meta_path(project_root: str) -> Path:
    return Path(project_root) / ".tcip" / "state" / "canvas_live.json"


def shapes_path(project_root: str) -> Path:
    return Path(project_root) / ".tcip" / "state" / "canvas_shapes.json"


def _guard_project_root(project_root: str) -> None:
    """403 if a client-supplied project_root escapes the configured image roots.

    This route writes files under project_root, so an exposed deployment
    (``TCIP_IMAGE_ROOTS`` set) must confine it, like review.py's ``_guard_path``.
    """
    try:
        assert_path_allowed(project_root)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc


def _write_json_no_fsync(path: Path, obj: dict) -> None:
    """Atomic replace (temp file + ``os.replace``) without ``fsync`` — see module docstring."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=2, default=str)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        _replace_with_retry(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _replace_with_retry(src: str, dst: Path, *, attempts: int = 5, delay: float = 0.05) -> None:
    """``os.replace`` with a short retry — a Windows virus scanner / indexer can momentarily
    hold the destination. Mirrors ``tcip_mcp.utils.atomic_io._replace_with_retry``."""
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


@router.post("/state")
def push_canvas_state(payload: CanvasStatePayload) -> dict:
    _guard_project_root(payload.project_root)
    now = time.time()
    mp = meta_path(payload.project_root)
    mp.parent.mkdir(parents=True, exist_ok=True)

    if payload.shapes is not None:
        # Geometry first, meta second: a reader pairing the new meta with the old geometry
        # sees an identity mismatch (stale), never a false match.
        _write_json_no_fsync(shapes_path(payload.project_root), {
            "image_path": payload.image_path,
            "tab": payload.tab,
            "shapes": payload.shapes,
            "received_at": now,
        })

    _write_json_no_fsync(mp, {
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
        "active_subject": payload.active_subject,
        "dirty": payload.dirty,
        "user": payload.user,
        "classes": payload.classes,
        "legend": payload.legend,
        "counts": payload.counts,
    })
    return {"status": "ok", "shapes_written": payload.shapes is not None}
