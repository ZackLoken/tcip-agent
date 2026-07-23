"""Annotation and review data model.  No GUI dependencies.

An :class:`Annotation` is the unit: *what* it is (``subject`` plus attribute values by name),
optionally *where* (``geometry``), and its provenance.  A geometry-less annotation is an image- or
plant-level label (e.g. a whole-plant rating).  ``score`` is set for a model prediction and ``None``
for ground truth.  Integer class ids do **not** live here — a name→id assignment is a per-training-run
artifact (see :mod:`tcip_mcp.class_registry`), never stored on an annotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates (``x1,y1,x2,y2``)."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Polygon:
    """Polygon annotation in pixel coordinates."""

    points: list[tuple[float, float]]


@dataclass
class Annotation:
    """One annotation on an image.

    ``subject`` is the object it is about (``catkin``, ``bush``, ``efb``).  ``geometry`` is a box or a
    polygon, or ``None`` for an image/plant-level label.  ``attributes`` maps an attribute name to its
    value name (e.g. ``{"elongation": "elongated"}``) — names, never a numeric class id.  ``score`` set
    means this is a prediction.  Provenance travels with the annotation: who authored it and, once a
    prediction is accepted into ground truth, who accepted it.
    """

    subject: str
    geometry: BBox | Polygon | None = None
    attributes: dict[str, str] = field(default_factory=dict)
    score: float | None = None
    created_by: str | None = None
    created_at: str | None = None
    accepted_by: str | None = None
    accepted_at: str | None = None


def bbox_of(geometry: BBox | Polygon) -> BBox:
    """The axis-aligned bounding box of a geometry — the box itself, or a polygon's enclosing box.

    Lets a detection consumer read a box from polygon ground truth: where polygons exist they are the
    source of truth and their boxes are a pure function of them, so the two can never silently diverge.
    """
    if isinstance(geometry, BBox):
        return geometry
    xs = [pt[0] for pt in geometry.points]
    ys = [pt[1] for pt in geometry.points]
    return BBox(min(xs), min(ys), max(xs), max(ys))


@dataclass
class AnnotationState:
    """All annotation data for one image.  GUI-free data model."""

    image_path: str = ""
    img_width: int = 0
    img_height: int = 0

    annotations: list[Annotation] = field(default_factory=list)
    predictions: list[Annotation] = field(default_factory=list)

    current_polygon: list[tuple[float, float]] = field(default_factory=list)
    mode: str = "box"
    # The subject new geometry is authored under; attribute values are set per annotation.
    active_subject: str = ""

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
