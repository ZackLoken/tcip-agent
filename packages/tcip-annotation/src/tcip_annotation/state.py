"""The annotation data model: :class:`Annotation` and the geometries it carries."""

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
    def from_normalized_center(cls, box, img_w: float, img_h: float) -> "BBox":
        """A normalized center-form ``[cx, cy, w, h]`` scaled to an ``img_w`` x ``img_h`` image."""
        cx, cy, w, h = (float(v) for v in box)
        return cls((cx - w / 2) * img_w, (cy - h / 2) * img_h,
                   (cx + w / 2) * img_w, (cy + h / 2) * img_h)


def is_ring(ring) -> bool:
    """Whether ``ring`` can be a shape: three or more points."""
    return len(ring) >= 3


@dataclass
class Polygon:
    """One or more simple closed contours (rings) in pixel coordinates; an occlusion-split instance
    holds several.

    At least one ring, every ring three or more points, or ``ValueError``.
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

    Has no bounding box and no area; :func:`bbox_of` refuses one. Every consumer that assembles
    training targets, computes IoU/matching, or reads a delivery-grade box filters Point geometries
    out itself.
    """

    x: float
    y: float


@dataclass
class Annotation:
    """One annotation on an image.

    ``subject`` is the object it is about, a non-empty string or ``ValueError``. ``geometry`` is a
    box, a polygon, a point, or ``None`` for an image/plant-level label. ``attributes`` maps an
    attribute name to its value name. ``score`` set means this is a prediction; a prediction's
    ``subject`` is the object class and each attribute head's call sits under ``attributes``.
    ``created_by``/``created_at`` name who authored it and ``accepted_by``/``accepted_at`` who
    accepted it into ground truth. ``iscrowd`` marks a region of unseparated objects of
    ``subject``, never one instance.
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

    def __post_init__(self) -> None:
        if not isinstance(self.subject, str) or not self.subject:
            raise ValueError(f"an annotation needs a non-empty string subject; got {self.subject!r}")


def object_rows(iscrowd) -> list[bool]:
    """For each of the crowd flags ``iscrowd`` (a list, an array or a tensor, one per row),
    whether its row is one object rather than a crowd region."""
    return [not bool(flag) for flag in iscrowd]


def instances(annotations: list[Annotation], *, crowd: bool = False) -> list[Annotation]:
    """The annotations that are each one object (:func:`object_rows`): every one but a crowd
    region; with ``crowd``, the crowd regions instead.
    """
    return [a for a, one in zip(annotations, object_rows([a.iscrowd for a in annotations]))
            if one is not crowd]


def box_derivable(geometry: "BBox | Polygon | Point | None") -> TypeGuard[BBox | Polygon]:
    """Whether a geometry yields a box: a rect or a polygon, never a point and never nothing."""
    return geometry is not None and not isinstance(geometry, Point)


def is_detection(a: Annotation) -> bool:
    """Whether ``a`` is one detected object: one object (:func:`instances`) with a box or region
    (:func:`box_derivable`). The one selection every match and detection count reads."""
    return bool(instances([a])) and box_derivable(a.geometry)


def prediction_score(a: Annotation) -> float:
    """A prediction's confidence, or ``ValueError`` naming a record that states none."""
    if a.score is None:
        raise ValueError(f"a prediction states its 'score'; this {a.subject!r} record states none")
    return a.score


def polygonal(geometry: "BBox | Polygon | Point | None") -> TypeGuard[Polygon]:
    """Whether a geometry is a polygon, the one shape an instance mask is rasterized from."""
    return isinstance(geometry, Polygon)


def bbox_of(geometry: BBox | Polygon) -> BBox:
    """The axis-aligned bounding box of a geometry: the box itself, or a polygon's enclosing box
    (over every ring, so a multi-ring instance's box covers all of its parts). Raises for a
    :class:`Point`.
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
