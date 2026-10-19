"""The annotation data model: :class:`Annotation` and the geometries it carries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Literal, TypeGuard, get_args

from pydantic import AfterValidator, ConfigDict, Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tcip_annotation.text import Text

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

type CrowdFlags = list[bool] | np.ndarray | Tensor
"""A detection target's per-row crowd flags: a list, an array or a tensor."""


@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates (``x1,y1,x2,y2``)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_normalized_center(
        cls,
        box: tuple[float, float, float, float],
        img_w: float,
        img_h: float,
    ) -> BBox:
        """Scale a normalized center-form ``[cx, cy, w, h]`` to an ``img_w`` x ``img_h`` image."""
        cx, cy, w, h = box
        return cls(
            (cx - w / 2) * img_w,
            (cy - h / 2) * img_h,
            (cx + w / 2) * img_w,
            (cy + h / 2) * img_h,
        )


RING_MIN_POINTS = 3
"""The fewest points a ring encloses an area with."""

Ring = list[tuple[float, float]]
"""One closed contour: its ``(x, y)`` pixel points in order."""

Rings = list[Ring]
"""One region's contours, an occlusion-split instance holding several."""


def is_ring(ring: Ring) -> bool:
    """Return whether ``ring`` can be a shape: :data:`RING_MIN_POINTS` or more points."""
    return len(ring) >= RING_MIN_POINTS


@dataclass
class Polygon:
    """One or more simple closed contours (rings) in pixel coordinates.

    An occlusion-split instance holds several. At least one ring, every ring one :func:`is_ring`
    admits, or ``ValueError``.
    """

    rings: Rings

    def __post_init__(self) -> None:
        """Refuse (``ValueError``) no ring, or a ring :func:`is_ring` does not admit."""
        if not self.rings or not all(is_ring(ring) for ring in self.rings):
            raise ValueError(
                f"a polygon needs at least one ring of {RING_MIN_POINTS} or more points; got "
                f"rings of {[len(ring) for ring in self.rings]} points"
            )


@dataclass
class Point:
    """A single labeled location in pixel coordinates.

    It is a placed prompt (human- or agent-supplied, for a promptable method like SAM) or a
    keypoint/landmark. Has no bounding box and no area.
    """

    x: float
    y: float


AttributeType = Literal["categorical", "ordinal"]
"""The attribute kinds a subject may carry: unordered, or ranked by declared order."""

ATTR_TYPES: tuple[AttributeType, ...] = get_args(AttributeType)


def _distinct(values: tuple[str, ...]) -> tuple[str, ...]:
    """Return ``values``, refused (``ValueError``) when one is repeated."""
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate values: {list(values)}")
    return values


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class Attribute:
    """One classification axis of a subject's instances.

    ``values`` are ordered, and a value's id is its position there; for an ``ordinal`` attribute
    that order is the rank (severity 0 < 1 < 2). Construction validates every field
    (``pydantic.ValidationError``): a known ``type`` and a non-empty list of distinct non-empty
    names.
    """

    name: str
    type: AttributeType
    values: Annotated[
        tuple[Text, ...], Field(min_length=1), AfterValidator(_distinct)
    ]


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
        """Refuse (``ValueError``) an empty ``subject``."""
        if not self.subject:
            raise ValueError(
                f"an annotation needs a non-empty string subject; got {self.subject!r}"
            )


def object_rows(iscrowd: CrowdFlags) -> list[bool]:
    """Return, for each crowd flag, whether its row is one object rather than a crowd region."""
    return [not bool(flag) for flag in iscrowd]


def instances(
    annotations: list[Annotation], *, crowd: bool = False
) -> list[Annotation]:
    """Return the annotations that are each one object (:func:`object_rows`).

    That is every one but a crowd region; with ``crowd``, the crowd regions instead.
    """
    return [
        a
        for a, one in zip(
            annotations, object_rows([a.iscrowd for a in annotations])
        )
        if one is not crowd
    ]


def box_derivable(
    geometry: BBox | Polygon | Point | None,
) -> TypeGuard[BBox | Polygon]:
    """Return whether a geometry yields a box: a rect or a polygon, never a point or nothing."""
    return geometry is not None and not isinstance(geometry, Point)


type GeometryReader = Callable[
    [BBox | Polygon | Point | None], TypeGuard[BBox | Polygon]
]
"""A predicate admitting the geometries a reader turns into a box: :func:`box_derivable` or
narrower."""


def is_detection(a: Annotation) -> bool:
    """Return whether ``a`` is one detected object.

    That is one object (:func:`instances`) with a box or region (:func:`box_derivable`).
    """
    return bool(instances([a])) and box_derivable(a.geometry)


def prediction_score(a: Annotation) -> float:
    """Return a prediction's confidence; ``ValueError`` naming a record that states none."""
    if a.score is None:
        raise ValueError(
            f"a prediction states its 'score'; this {a.subject!r} record states none"
        )
    return a.score


def polygonal(geometry: BBox | Polygon | Point | None) -> TypeGuard[Polygon]:
    """Return whether a geometry is a polygon."""
    return isinstance(geometry, Polygon)


def bbox_of(geometry: BBox | Polygon) -> BBox:
    """Return the axis-aligned bounding box of a geometry.

    That is the box itself, or a polygon's enclosing box (over every ring, so a multi-ring
    instance's box covers all of its parts).
    """
    if isinstance(geometry, BBox):
        return geometry
    xs = [pt[0] for ring in geometry.rings for pt in ring]
    ys = [pt[1] for ring in geometry.rings for pt in ring]
    return BBox(min(xs), min(ys), max(xs), max(ys))
