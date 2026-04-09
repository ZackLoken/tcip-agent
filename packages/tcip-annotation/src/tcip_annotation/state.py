"""Annotation and review data model.  No GUI dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int


@dataclass
class Polygon:
    """Polygon annotation in pixel coordinates."""

    points: list[tuple[float, float]]
    class_id: int


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
