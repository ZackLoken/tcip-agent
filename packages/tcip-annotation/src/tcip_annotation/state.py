"""Annotation and review data model.  No GUI dependencies.

An :class:`Annotation` is the unit: *what* it is (``subject`` plus attribute values by name),
optionally *where* (``geometry``), and its provenance.  A geometry-less annotation is an image- or
plant-level label (e.g. a whole-plant rating).  ``score`` is set for a model prediction and ``None``
for ground truth.  Integer class ids do not live here: a name→id assignment is a per-training-run
artifact (see :mod:`tcip_mcp.subject_registry`), never stored on an annotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeGuard


@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates (``x1,y1,x2,y2``)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_normalized_centre(cls, box, img_w: float, img_h: float) -> "BBox":
        """A normalized centre-form ``[cx, cy, w, h]`` scaled to an ``img_w`` x ``img_h`` image:
        the one conversion every reader of a verdict's or a staged proposal's box takes."""
        cx, cy, w, h = (float(v) for v in box)
        return cls((cx - w / 2) * img_w, (cy - h / 2) * img_h,
                   (cx + w / 2) * img_w, (cy + h / 2) * img_h)


def is_ring(ring) -> bool:
    """Whether ``ring`` can be a shape: three or more points. The one statement of the rule, read
    by :class:`Polygon`'s construction (which refuses a ring that is not one) and by the contour
    extractor (which drops such a contour, a stray pixel beside a real component)."""
    return len(ring) >= 3


@dataclass
class Polygon:
    """One or more simple closed contours (rings) in pixel coordinates.

    Most annotations are a single ring: a person draws one contour. A model-predicted mask can be
    more than one: an occlusion-split instance (a leaf crossed by a stem, a fruit behind a branch,
    routine in this imagery) is genuinely more than one region, and holding every ring is what makes
    that a represented fact instead of a silently truncated one.

    A polygon is valid where it is made: at least one ring, every ring three or more points, or
    ``ValueError``. A ring that cannot be a shape is refused where it is produced, never carried
    to a reader or a writer that would have to drop it.
    """

    rings: list[list[tuple[float, float]]]

    def __post_init__(self) -> None:
        if not self.rings or not all(is_ring(ring) for ring in self.rings):
            raise ValueError(
                f"a polygon needs at least one ring of three or more points; got rings of "
                f"{[len(ring) for ring in self.rings]} points")


@dataclass
class Point:
    """A single labeled location in pixel coordinates: a placed prompt (human- or agent-supplied,
    for a promptable method like SAM) or a keypoint/landmark.

    Deliberately has no bounding box and no area: a Point is not a detection/segmentation target, and
    :func:`bbox_of` refuses one rather than fabricate a degenerate zero-area box that could silently
    pass as a real one (a real hazard: a fabricated zero-area box entering a loader's targets
    as a training target, or matching nothing at any IoU in delivery-grade evaluation while reading
    as a legitimate miss rather than a category error). Every consumer that assembles training
    targets, computes IoU/matching, or reads a delivery-grade box must filter Point geometries out
    explicitly, the same way a geometry-less annotation already is, never rely on bbox_of's refusal
    as the only guard.
    """

    x: float
    y: float


@dataclass
class Annotation:
    """One annotation on an image.

    ``subject`` is the object it is about (``bush``, ``leaf``, ``efb``).  ``geometry`` is a box, a
    polygon, a point, or ``None`` for an image/plant-level label.  ``attributes`` maps an attribute
    name to its value name (e.g. ``{"thorns": "present"}``): names, never a numeric class id.
    ``score`` set means this is a prediction, and a prediction carries the same shape ground truth
    does: a classified prediction's ``subject`` is still the object class, with the classifier's
    decoded call sitting under ``attributes``, never the value alone in ``subject``. Provenance
    travels with the annotation: who authored it and, once a prediction is accepted into ground
    truth, who accepted it. ``accepted_by_rule`` names the validation record a rule-based
    admission was verified against (``<experiment_id>:<record_digest>``), set only by the
    producer that verified the claim; it is never a substitute for ``accepted_by``.

    ``iscrowd`` (COCO's own spelling) marks a region holding many objects of ``subject`` that
    were not separated: it is never one instance, so it never trains as a positive box, never
    counts as a missed object at evaluation and never counts as one object in a count.

    An annotation names what it is about where it is made: ``subject`` a non-empty string, or
    ``ValueError``. A record with no subject is undecodable by name, so no door carries one.
    """

    subject: str
    geometry: BBox | Polygon | Point | None = None
    attributes: dict[str, str] = field(default_factory=dict)
    score: float | None = None
    iscrowd: bool = False
    created_by: str | None = None
    created_at: str | None = None
    accepted_by: str | None = None
    accepted_at: str | None = None
    accepted_by_rule: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.subject, str) or not self.subject:
            raise ValueError(f"an annotation needs a non-empty string subject; got {self.subject!r}")


def instances(annotations: list[Annotation], *, crowd: bool = False) -> list[Annotation]:
    """The annotations that are each one object: every one but a crowd region; with ``crowd``,
    the crowd regions instead, the other half of the one split. The one place an
    annotation-level count, pairing or size asks which records are objects."""
    return [a for a in annotations if a.iscrowd == crowd]


def box_derivable(geometry: "BBox | Polygon | Point | None") -> TypeGuard[BBox | Polygon]:
    """Whether a geometry yields a box: a rect or a polygon, never a point and never nothing.

    The one statement of which geometries a detection or segmentation target can be read from.
    A loader declares it as the geometry it reads, the readers that extract targets filter by it
    and the writers that assemble them skip by it, so what a sample is admitted as and what is
    read out of it cannot disagree about what a target is. :func:`bbox_of` is what reads the box
    once this has answered.
    """
    return geometry is not None and not isinstance(geometry, Point)


def is_detection(a: Annotation) -> bool:
    """Whether ``a`` is one detected object: one object (:func:`instances`) with a box or region
    (:func:`box_derivable`). The one selection every match and detection count reads."""
    return bool(instances([a])) and box_derivable(a.geometry)


def prediction_score(a: Annotation) -> float:
    """A prediction's confidence, the model's own statement, or ``ValueError`` naming the record
    that states none: the one read of it every match, threshold and score record takes."""
    if a.score is None:
        raise ValueError(f"a prediction states its 'score'; this {a.subject!r} record states none")
    return a.score


def polygonal(geometry: "BBox | Polygon | Point | None") -> TypeGuard[Polygon]:
    """Whether a geometry is a polygon, the one shape an instance mask is rasterized from.

    Declared by the instance segmentation loader and asked by every reader of its rings, so the
    same rule admits a sample and extracts its masks.
    """
    return isinstance(geometry, Polygon)


def bbox_of(geometry: BBox | Polygon) -> BBox:
    """The axis-aligned bounding box of a geometry: the box itself, or a polygon's enclosing box
    (over every ring, so a multi-ring instance's box covers all of its parts).

    Lets a detection consumer read a box from polygon ground truth: where polygons exist they are the
    source of truth and their boxes are a pure function of them, so the two can never silently diverge.

    Raises for a :class:`Point`: it has no bounding box, and returning a fabricated degenerate one
    would let it silently pass as a real detection/segmentation target downstream. Callers that may
    see a Point must filter it out before calling this, not rely on the raise as the only guard.
    """
    if isinstance(geometry, BBox):
        return geometry
    if isinstance(geometry, Point):
        raise ValueError(
            "bbox_of: a Point has no bounding box; it is not a detection/segmentation target. "
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
