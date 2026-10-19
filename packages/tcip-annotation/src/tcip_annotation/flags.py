"""Review flags: a person's request for a second look, and the reply that resolves it.

A flag is a person's comment asking for a second look at one place on an image, one of a
bucket's proposals, or the image as a whole. One record per image holds every flag raised on
it; a resolved flag is kept, never removed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Annotated, TypedDict

from pydantic import (
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
    from pydantic import JsonValue

REVIEW_FLAGS_STORE = "review_flags"


@pydantic_dataclass(
    frozen=True, config=ConfigDict(extra="forbid", allow_inf_nan=False)
)
class Flag:
    """One flag: its id, comment, raiser and time, its target, and its resolution.

    ``id`` names it, ``text`` is the comment, ``by`` who raised it (a person's recorded identity,
    the label record's own spelling) and ``at`` when. Its target is one of: ``point`` (pixel
    ``(x, y)``) with the ``subject`` of the annotation flagged there; ``proposal`` (``(bucket,
    index)``, the index in that bucket's document for the image); or neither, the image as a
    whole. Once resolved it carries ``resolved_by`` (a person) and ``resolved_at``, the
    ``reply`` given, and ``removed`` when the save that resolved it did so by removing the
    annotation it sat on. Construction validates every field (``pydantic.ValidationError``).
    """

    id: Text
    text: Text
    by: Person
    at: Text
    point: tuple[float, float] | None
    subject: Text | None
    proposal: tuple[Text, Annotated[StrictInt, Field(ge=0)]] | None
    resolved_by: Person | None
    resolved_at: str | None
    reply: str
    removed: StrictBool

    def __post_init__(self) -> None:
        """Refuse a target or a resolution stated by halves (``ValueError``).

        A point without its subject or a subject without its point, a point beside a proposal,
        and a resolver without a time or a time without a resolver refuse.
        """
        if (
            (self.point is None) != (self.subject is None)
            or (self.point is not None and self.proposal is not None)
            or (self.resolved_by is None) != (self.resolved_at is None)
        ):
            raise ValueError(
                f"flag {self.id!r} is not a comment by a person at a time on a point with its "
                "subject, a proposal or the image, resolved by a person at a time if at all"
            )

    @property
    def open(self) -> bool:
        """Whether nobody has resolved it."""
        return self.resolved_by is None

    def resolved(
        self, *, by: str, at: str, reply: str = "", removed: bool = False
    ) -> Flag:
        """Return this flag resolved by ``by`` at ``at``, validated as a :class:`Flag`."""
        return replace(
            self, resolved_by=by, resolved_at=at, reply=reply, removed=removed
        )


@dataclass(frozen=True)
class FlagRequest:
    """A flag a save raises: its comment and its target, as :class:`Flag` states one."""

    text: str
    point: tuple[float, float] | None = None
    subject: str | None = None
    proposal: tuple[str, int] | None = None


def raised(request: FlagRequest, *, by: str, at: str) -> Flag:
    """Return ``request`` as a new open flag by ``by`` at ``at`` under a freshly minted id.

    The comment is stripped; a flag that does not validate refuses as :class:`Flag` does.
    """
    import uuid

    return Flag(
        id=uuid.uuid4().hex,
        text=request.text.strip(),
        by=by,
        at=at,
        point=request.point,
        subject=request.subject,
        proposal=request.proposal,
        resolved_by=None,
        resolved_at=None,
        reply="",
        removed=False,
    )


def flag_key(label_key: Key) -> Key:
    """Return the flags record key of the image whose label document ``label_key`` names."""
    return Key(REVIEW_FLAGS_STORE, label_key.root, label_key.parts)


_FLAG = TypeAdapter(Flag)


def encode_flag(flag: Flag) -> dict[str, JsonValue]:
    """Return ``flag`` as the entry its record stores."""
    return _FLAG.dump_python(flag, mode="json")


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
    at: str,
) -> list[Flag]:
    """Return ``flags`` with each open one whose annotation was removed resolved as ``removed``.

    An open flag resolves when an annotation of its subject held its point in ``before`` and none
    does in ``after``; every other flag is unchanged.
    """

    def removed(flag: Flag) -> bool:
        subject, point = flag.subject, flag.point
        if not flag.open or subject is None or point is None:
            return False
        return any(covers(a, subject, point) for a in before) and not any(
            covers(a, subject, point) for a in after
        )

    return [
        flag.resolved(by=by, at=at, removed=True) if removed(flag) else flag
        for flag in flags
    ]
