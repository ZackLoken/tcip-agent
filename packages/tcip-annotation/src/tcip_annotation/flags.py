"""Review flags: a person's request for a second look, and the reply that resolves it.

A flag is a person's comment asking for a second look at one place on an image, one of a
bucket's proposals, or the image as a whole. One record per image holds every flag raised on
it; a resolved flag is kept, never removed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Annotated, TypedDict

from pydantic import (
    AwareDatetime,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    TypeAdapter,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tcip_annotation.json_io import Person
from tcip_annotation.state import Annotation, BBox, Point, Polygon
from tcip_annotation.text import Text
from tcip_store import Key

if TYPE_CHECKING:
    from datetime import datetime

    from pydantic import JsonValue

REVIEW_FLAGS_STORE = "review_flags"

_RECORD = ConfigDict(extra="forbid", allow_inf_nan=False)


@pydantic_dataclass(frozen=True, config=_RECORD)
class PointTarget:
    """A flag's target at a pixel: ``point`` ``(x, y)`` and the ``subject`` annotated there."""

    point: tuple[float, float]
    subject: Text


@pydantic_dataclass(frozen=True, config=_RECORD)
class ProposalTarget:
    """A flag's target at a proposal: its ``bucket`` and its ``index`` in that bucket's document."""

    bucket: Text
    index: Annotated[StrictInt, Field(ge=0)]


@pydantic_dataclass(frozen=True, config=_RECORD)
class FlagRequest:
    """A flag's comment, ``text``, and its ``target``, ``None`` for the image as a whole.

    Construction validates every field (``pydantic.ValidationError``).
    """

    text: Text
    target: PointTarget | ProposalTarget | None


@pydantic_dataclass(frozen=True, config=_RECORD)
class Resolution:
    """Who resolved a flag (a person) and when, the ``reply`` given, and whether by ``removed``.

    ``removed`` is whether the save that resolved it did so by removing the annotation it sat on.
    """

    by: Person
    at: AwareDatetime
    reply: str
    removed: StrictBool


@pydantic_dataclass(frozen=True, config=_RECORD)
class Flag:
    """One flag: its id, its request, its raiser and time, and its resolution.

    ``id`` names it, ``request`` is the comment and target it was raised with, ``by`` who raised
    it (a person's recorded identity, the label record's own spelling) and ``at`` when;
    ``resolution`` is ``None`` while it is open. Construction validates every field
    (``pydantic.ValidationError``).
    """

    id: Text
    request: FlagRequest
    by: Person
    at: AwareDatetime
    resolution: Resolution | None

    @property
    def open(self) -> bool:
        """Whether nobody has resolved it."""
        return self.resolution is None

    def resolved(
        self, *, by: str, at: datetime, reply: str = "", removed: bool = False
    ) -> Flag:
        """Return this flag resolved by ``by`` at ``at``, validated as a :class:`Flag`."""
        return replace(
            self,
            resolution=Resolution(by=by, at=at, reply=reply, removed=removed),
        )


def raised(request: FlagRequest, *, by: str, at: datetime) -> Flag:
    """Return ``request`` as a new open flag by ``by`` at ``at`` under a freshly minted id.

    A flag that does not validate refuses as :class:`Flag` does.
    """
    import uuid

    return Flag(
        id=uuid.uuid4().hex, request=request, by=by, at=at, resolution=None
    )


def flag_key(label_key: Key) -> Key:
    """Return the flags record key of the image whose label document ``label_key`` names."""
    return Key(REVIEW_FLAGS_STORE, label_key.root, label_key.parts)


class FlagsRecord(TypedDict):
    """One image's flags record: every flag raised on it, oldest first."""

    flags: list[Flag]


_FLAGS_RECORD = TypeAdapter(FlagsRecord)


def flags_of(record: JsonValue) -> list[Flag]:
    """Return the flags a stored record holds, oldest first; an absent record holds none.

    A record that does not validate as a :class:`FlagsRecord` refuses
    (``pydantic.ValidationError``) naming each field.
    """
    if record is None:
        return []
    return _FLAGS_RECORD.validate_python(record)["flags"]


def flags_record(flags: list[Flag]) -> dict[str, JsonValue]:
    """Return ``flags`` as the :class:`FlagsRecord` its key stores."""
    return _FLAGS_RECORD.dump_python({"flags": flags}, mode="json")


def read_flags(key: Key) -> list[Flag]:
    """Return every flag the record ``key`` names holds (:func:`flags_of`)."""
    import tcip_store

    return flags_of(tcip_store.read(key, default=None))


def covers(
    annotation: Annotation, subject: str, point: tuple[float, float]
) -> bool:
    """Return whether ``annotation`` is one of ``subject`` whose geometry holds ``point``.

    Holding is inside or on the edge of a box or a polygon's rings, or exactly a placed point's
    own coordinate.
    """
    if annotation.subject != subject:
        return False
    x, y = point
    geometry = annotation.geometry
    if isinstance(geometry, BBox):
        return (
            geometry.x1 <= x <= geometry.x2 and geometry.y1 <= y <= geometry.y2
        )
    if isinstance(geometry, Polygon):
        from shapely.geometry import Point as ShapelyPoint

        from tcip_annotation.matching import shapely_geometry

        return bool(
            shapely_geometry(geometry.rings).intersects(ShapelyPoint(x, y))
        )
    if isinstance(geometry, Point):
        return (geometry.x, geometry.y) == (x, y)
    return False


def resolved_by_removal(
    flags: list[Flag],
    before: list[Annotation],
    after: list[Annotation],
    *,
    by: str,
    at: datetime,
) -> list[Flag]:
    """Return ``flags`` with each open one whose annotation was removed resolved as ``removed``.

    An open flag resolves when an annotation of its subject held its point in ``before`` and none
    does in ``after``; every other flag is unchanged.
    """

    def removed(flag: Flag) -> bool:
        target = flag.request.target
        if not flag.open or not isinstance(target, PointTarget):
            return False
        return any(
            covers(a, target.subject, target.point) for a in before
        ) and not any(covers(a, target.subject, target.point) for a in after)

    return [
        flag.resolved(by=by, at=at, removed=True) if removed(flag) else flag
        for flag in flags
    ]
