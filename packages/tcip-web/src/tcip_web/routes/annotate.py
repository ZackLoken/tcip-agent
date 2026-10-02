"""Annotation label CRUD routes for the Annotate tab.

Reads / writes the canonical per-image label file (one JSON per image, holding every subject's
annotations by name) via :mod:`tcip_annotation.json_io`, at the label path the caller supplies.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from tcip_annotation.json_io import (
    UnreadableLabelDocument,
    authorship_of,
    client_annotation,
    read_annotations_versioned,
)
from tcip_annotation.state import Annotation
from tcip_store import Version, VersionConflict
from tcip_web.identity import resolve_user, user_id
from tcip_web.paths import allowed_image_dimensions, allowed_optional, allowed_path

router = APIRouter(prefix="/api/annotate", tags=["annotate"])


class AnnotationPayload(BaseModel):
    """One annotation: its ``subject`` (the object it is about), a geometry (``bbox`` corners,
    ``points`` for the one ring the canvas draws by hand, ``rings`` for a loaded multi-ring shape
    round-tripping, ``point`` for a placed prompt or keypoint, or none of them for an image-level
    label), its attribute values by name and its crowd flag.

    Every field but the subject is carried uninterpreted: its one reading is
    :func:`~tcip_annotation.json_io.annotation_from_payload`.
    """

    subject: str
    bbox: Any = None
    points: Any = None
    rings: Any = None
    point: Any = None
    attributes: Any = None
    iscrowd: Any = None
    # Keep-original-creator: a loaded shape's created_by, sign-off and rule marker round-trip
    # back on save, so a re-save never re-stamps existing labels; new shapes carry none of them.
    created_by: Any = None
    created_at: Any = None
    accepted_by: Any = None
    accepted_at: Any = None
    accepted_by_rule: Any = None


class SavePayload(BaseModel):
    image_path: str
    # Non-empty: a save with nowhere to write can never succeed, so it is refused (422) rather
    # than accepted as a no-op.
    label_path: str = Field(min_length=1)
    annotations: list[AnnotationPayload] = []
    # The label document's version token as the client loaded it: with one, the save is a
    # compare-and-set and a 409 says it changed underneath. Omit to skip the comparison.
    base_mtime: Optional[str] = None
    # GUI-set annotator identity (bare name, e.g. "breeder"); stamped as created_by ("user:<name>").
    # Omitted by non-GUI callers -> backend falls back to the OS/env user.
    user: Optional[str] = None


def annotation_dict(a: Annotation) -> dict:
    """An :class:`Annotation` for the canvas: the library's client projection
    (:func:`~tcip_annotation.json_io.client_annotation`) plus this response's own ``authorship``,
    derived through :func:`authorship_of`; the label document itself carries no such field.
    """
    return {**client_annotation(a), "authorship": authorship_of(a)}


@router.get("/labels")
def load_labels(image_path: str, label_path: Optional[str] = None) -> dict:
    """Read existing labels for an image and return them in pixel coords."""
    w, h = allowed_image_dimensions(image_path)
    label_path = allowed_optional(label_path)
    annotations: list[dict] = []
    token: Optional[str] = None
    if label_path:
        try:
            stored, version = read_annotations_versioned(label_path)
        except UnreadableLabelDocument as exc:
            raise HTTPException(400, str(exc)) from exc
        annotations = [annotation_dict(a) for a in stored]
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
    """Write labels for an image to its single per-image JSON file, an empty list as an empty
    document.

    The save is :func:`~tcip_mcp.dataset_layout.save_label_document`; a write that commits and
    cannot be recorded answers 409 with the marker and the save's recorded facts.
    """
    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_mcp.dataset_layout import save_label_document
    from tcip_web.routes.audit_gap import audit_gap_409
    from tcip_web.state import store

    def saved(token: Optional[str]) -> dict:
        # The new version token, so the client can save again without a reload.
        return {"status": "ok", "image_path": payload.image_path,
                "n_annotations": len(payload.annotations), "base_mtime": token}

    w, h = allowed_image_dimensions(payload.image_path)
    label_path = allowed_path(payload.label_path)
    # The lost-update guard, inside the store's own lock: with a token the write is refused
    # unless the stored document still matches it, and the client resolves the 409 by reloading.
    expect = Version(payload.base_mtime) if payload.base_mtime is not None else None
    try:
        version = save_label_document(
            store.project_root, payload.image_path, label_path,
            [ap.model_dump() for ap in payload.annotations], width=w, height=h,
            author=user_id(resolve_user(payload.user)), expect=expect)
    except VersionConflict as exc:
        raise HTTPException(409, {"error": "label file changed since it was loaded"}) from exc
    except AuditEntryNotWritten as exc:
        raise audit_gap_409(exc, saved(exc.arguments["version"])) from exc
    except OSError as exc:
        raise HTTPException(500, f"could not write labels: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return saved(version.token if version is not None else None)
