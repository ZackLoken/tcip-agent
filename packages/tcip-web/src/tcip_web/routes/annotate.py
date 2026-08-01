"""Annotation label CRUD routes for the Annotate tab.

Reads / writes the canonical per-image label file (one JSON per image, holding every subject's
annotations by name) via :mod:`tcip_annotation.json_io`. The label path is supplied by the caller
so the backend doesn't have to guess a dataset layout.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_annotation import BBox, Point, Polygon
from tcip_annotation.json_io import read_annotations, write_annotations
from tcip_annotation.state import Annotation
from tcip_mcp.pipelines.image_utils import image_dimensions, resolve_image_source
from tcip_web.identity import resolve_user, user_id
from tcip_web.paths import assert_path_allowed
from tcip_web.state import PredictionReference, store

router = APIRouter(prefix="/api/annotate", tags=["annotate"])


class AnnotationPayload(BaseModel):
    """One annotation: its ``subject`` (the object it is about), a geometry (a box, a polygon, a
    point, or none of them for an image-level label), and its attribute values by name."""

    subject: str
    bbox: Optional[list[float]] = None          # [x1, y1, x2, y2], pixel
    points: Optional[list[list[float]]] = None  # single-ring polygon vertices, pixel — a shape the
                                                 # canvas itself drew/edited by hand
    rings: Optional[list[list[list[float]]]] = None  # multi-ring polygon (a loaded, unedited
                                                       # occlusion-split instance_seg shape round-
                                                       # tripping through save) — takes precedence
                                                       # over `points` when both are present
    point: Optional[list[float]] = None         # [x, y], pixel — one placed prompt / keypoint.
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
    label_path: Optional[str] = None
    annotations: list[AnnotationPayload] = []
    # Project root for the audit trail (optional; skipped if absent).
    project_root: Optional[str] = None
    # The label-file mtime token the client loaded. When present, a write is rejected (409) if the
    # file changed underneath the client — a concurrent agent or second browser tab — so its edits
    # aren't clobbered. Omit to skip the check.
    base_mtime: Optional[str] = None
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
    # Channel-aware: resolve_image_source folds a `.bandgroup` manifest (or a genuinely
    # multi-band raster) into the real frame image_dimensions measures, instead of a bare PIL
    # header read misreporting its axes.
    return image_dimensions(resolve_image_source(p.parent, p.stem))


def _guard_label_path(path: Optional[str]) -> None:
    """Reject a client-supplied label path that escapes the configured image roots.

    Label read/write paths are attacker-controlled, so an exposed deployment
    (``TCIP_IMAGE_ROOTS`` set) must confine them exactly like image serving —
    otherwise ``write_annotations`` is an arbitrary file write/delete primitive.
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


def _ann_dict(a: Annotation) -> dict:
    """Serialize an :class:`Annotation` for the canvas (pixel coords + provenance).

    ``rings`` (not ``points``) for a polygon — a loaded GT annotation can be a multi-ring
    occlusion-split instance_seg prediction accepted through Review, so the canvas always receives
    every ring rather than silently only the first. The canvas itself still only ever *draws* a
    single ring by hand (see ``AnnotationPayload.points`` below, the save side). ``point`` is the
    singular ``[x, y]`` of a placed prompt / keypoint, the same key the on-disk schema uses.
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
    return out


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
                    "label_path": payload.label_path,
                    "n_annotations": len(payload.annotations),
                },
                "status": "ok",
            },
        )
    except Exception:
        pass


@router.get("/labels")
def load_labels(image_path: str, label_path: Optional[str] = None) -> dict:
    """Read existing labels for an image and return them in pixel coords."""
    w, h = _image_dims(image_path)
    _guard_label_path(label_path)
    annotations: list[dict] = []
    if label_path:
        annotations = [_ann_dict(a) for a in read_annotations(label_path)]
    return {
        "image_path": image_path,
        "img_width": w,
        "img_height": h,
        "annotations": annotations,
        # Version token the client echoes back on save for the lost-update guard.
        "base_mtime": _mtime_token(label_path),
    }


@router.post("/labels")
def save_labels(payload: SavePayload) -> dict:
    """Write labels for an image to its single per-image JSON file.

    An empty annotation list is written as ``{"annotations": []}`` (``keep_empty=True``) rather than
    deleted, so clearing all annotations keeps the record instead of erasing it. That record is not a
    negative on its own — it trains as one only once the breeder marks the image Complete
    (``image_status.json``); until then it reads as unannotated (CLAUDE.md's negative invariant).
    """
    w, h = _image_dims(payload.image_path)
    _guard_label_path(payload.label_path)

    # Lost-update guard: reject if the label file changed since the client loaded it (a concurrent
    # agent write or a second browser tab). The client resolves the 409 by reloading. Omitting
    # base_mtime skips the check.
    if payload.base_mtime is not None and payload.label_path:
        if _mtime_token(payload.label_path) != payload.base_mtime:
            raise HTTPException(409, {"error": "label file changed since it was loaded"})

    # Human-authored GT: a round-tripped shape keeps its original created_by (the creator stays the
    # creator through edits); only shapes with no provenance — new ones — are stamped to the current
    # annotator. json_io persists all four provenance fields natively.
    author = user_id(resolve_user(payload.user))
    now_iso = datetime.now(timezone.utc).isoformat()

    def _to_annotation(ap: AnnotationPayload) -> Annotation:
        geometry = None
        if ap.rings:
            geometry = Polygon(rings=[[(float(p[0]), float(p[1])) for p in ring] for ring in ap.rings])
        elif ap.points:
            # The canvas draws one contour by hand — single-ring input. Polygon itself supports
            # multiple rings (occlusion-split model output), but a freshly-drawn shape is always one.
            geometry = Polygon(rings=[[(float(p[0]), float(p[1])) for p in ap.points]])
        elif ap.bbox is not None:
            geometry = BBox(*ap.bbox)
        elif ap.point is not None:
            geometry = Point(float(ap.point[0]), float(ap.point[1]))
        # accepted_* only ride along on round-tripped shapes (created_by present) — a new shape
        # claiming acceptance would mint review sign-off that never happened.
        round_tripped = bool(ap.created_by)
        return Annotation(
            subject=ap.subject,
            geometry=geometry,
            attributes=dict(ap.attributes),
            created_by=ap.created_by or author,
            created_at=ap.created_at if round_tripped else now_iso,
            accepted_by=ap.accepted_by if round_tripped else None,
            accepted_at=ap.accepted_at if round_tripped else None,
        )

    annotations = [_to_annotation(ap) for ap in payload.annotations]

    written = payload.label_path is not None
    if payload.label_path:
        try:
            os.makedirs(os.path.dirname(payload.label_path) or ".", exist_ok=True)
            write_annotations(payload.label_path, annotations, w, h, keep_empty=True)
        except OSError as exc:
            raise HTTPException(500, f"could not write labels: {exc}") from exc

    _audit_gui_write(payload)

    return {
        "status": "ok",
        "image_path": payload.image_path,
        "label_written": written,
        "n_annotations": len(annotations),
        # New version token so the client can save again without a reload.
        "base_mtime": _mtime_token(payload.label_path),
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
