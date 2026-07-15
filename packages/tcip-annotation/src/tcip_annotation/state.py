"""Annotation and review data model.  No GUI dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field


# Provenance travels with a shape: who authored it (`created_by`, e.g. "user:zack", "sam",
# "claude", "model:baseline@<path>") and, once a prediction is accepted into GT, who accepted it.
# The four fields below are keyword-only so `PredBBox`/`PredPolygon`'s positional `confidence`
# field is unaffected (verified: `PredBBox(x1,y1,x2,y2,class_id,confidence)` still works).
@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    created_by: "str | None" = field(default=None, kw_only=True)
    created_at: "str | None" = field(default=None, kw_only=True)
    accepted_by: "str | None" = field(default=None, kw_only=True)
    accepted_at: "str | None" = field(default=None, kw_only=True)


@dataclass
class Polygon:
    """Polygon annotation in pixel coordinates."""

    points: list[tuple[float, float]]
    class_id: int
    created_by: "str | None" = field(default=None, kw_only=True)
    created_at: "str | None" = field(default=None, kw_only=True)
    accepted_by: "str | None" = field(default=None, kw_only=True)
    accepted_at: "str | None" = field(default=None, kw_only=True)


def boxes_from_polygons(polygons: list["Polygon"]) -> list["BBox"]:
    """Axis-aligned bounding box of each polygon, preserving class id.

    Used where the detect layout is derived from the segment layout: when polygons
    exist they are the source of truth and their boxes are a pure function of them,
    so the two label files can never silently diverge.
    """
    out: list[BBox] = []
    for p in polygons:
        if not p.points:
            continue
        xs = [pt[0] for pt in p.points]
        ys = [pt[1] for pt in p.points]
        out.append(BBox(min(xs), min(ys), max(xs), max(ys), class_id=p.class_id))
    return out


@dataclass
class PredBBox(BBox):
    """Predicted bounding box with confidence score."""

    confidence: float = 0.0


@dataclass
class PredPolygon(Polygon):
    """Predicted polygon with confidence score."""

    confidence: float = 0.0


@dataclass
class AnnotationState:
    """All annotation data for one image.  GUI-free data model."""

    image_path: str = ""
    img_width: int = 0
    img_height: int = 0

    boxes: list[BBox] = field(default_factory=list)
    polygons: list[Polygon] = field(default_factory=list)

    pred_boxes: list[PredBBox] = field(default_factory=list)
    pred_polygons: list[PredPolygon] = field(default_factory=list)

    current_polygon: list[tuple[float, float]] = field(default_factory=list)
    mode: str = "box"
    active_class: int = 0

    # Undo / redo
    _undo_stack: list = field(default_factory=list, repr=False)
    _redo_stack: list = field(default_factory=list, repr=False)

    # Spatial index cache
    _poly_bboxes: list[tuple[float, float, float, float]] = field(
        default_factory=list, repr=False
    )
    _poly_bboxes_dirty: bool = field(default=True, repr=False)

    # Selection
    selected_polygon_idx: int | None = None
