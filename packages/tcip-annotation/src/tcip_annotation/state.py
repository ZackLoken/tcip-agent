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
    """One or more simple closed contours (rings) in pixel coordinates.

    Most annotations are a single ring — a person draws one contour. A model-predicted mask can be
    more than one: an occlusion-split instance (a catkin behind a branch, a leaf crossed by a stem —
    routine in this imagery) is genuinely more than one region, and holding every ring is what makes
    that a represented fact instead of a silently truncated one.
    """

    rings: list[list[tuple[float, float]]]


@dataclass
class Point:
    """A single labeled location in pixel coordinates — a placed prompt (human- or agent-supplied,
    for a promptable method like SAM) or a keypoint/landmark.

    Deliberately has no bounding box and no area: a Point is not a detection/segmentation target, and
    :func:`bbox_of` refuses one rather than fabricate a degenerate zero-area box that could silently
    pass as a real one (a real hazard: a fabricated zero-area box entering an assembled COCO dataset
    as a training target, or matching nothing at any IoU in delivery-grade evaluation while reading
    as a legitimate miss rather than a category error). Every consumer that assembles training
    targets, computes IoU/matching, or reads a delivery-grade box must filter Point geometries out
    explicitly, the same way a geometry-less annotation already is — never rely on bbox_of's refusal
    as the only guard.
    """

    x: float
    y: float


@dataclass
class Annotation:
    """One annotation on an image.

    ``subject`` is the object it is about (``catkin``, ``bush``, ``efb``).  ``geometry`` is a box, a
    polygon, a point, or ``None`` for an image/plant-level label.  ``attributes`` maps an attribute
    name to its value name (e.g. ``{"elongation": "elongated"}``) — names, never a numeric class id.
    ``score`` set means this is a prediction.  Provenance travels with the annotation: who authored it
    and, once a prediction is accepted into ground truth, who accepted it.
    """

    subject: str
    geometry: BBox | Polygon | Point | None = None
    attributes: dict[str, str] = field(default_factory=dict)
    score: float | None = None
    created_by: str | None = None
    created_at: str | None = None
    accepted_by: str | None = None
    accepted_at: str | None = None


def bbox_of(geometry: BBox | Polygon) -> BBox:
    """The axis-aligned bounding box of a geometry — the box itself, or a polygon's enclosing box
    (over every ring, so a multi-ring instance's box covers all of its parts).

    Lets a detection consumer read a box from polygon ground truth: where polygons exist they are the
    source of truth and their boxes are a pure function of them, so the two can never silently diverge.

    Raises for a :class:`Point` — it has no bounding box, and returning a fabricated degenerate one
    would let it silently pass as a real detection/segmentation target downstream. Callers that may
    see a Point must filter it out before calling this, not rely on the raise as the only guard.
    """
    if isinstance(geometry, BBox):
        return geometry
    if isinstance(geometry, Point):
        raise ValueError(
            "bbox_of: a Point has no bounding box — it is not a detection/segmentation target. "
            "Filter Point geometries out before calling bbox_of (the same way a geometry-less "
            "annotation is already filtered), rather than relying on this raise."
        )
    xs = [pt[0] for ring in geometry.rings for pt in ring]
    ys = [pt[1] for ring in geometry.rings for pt in ring]
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
