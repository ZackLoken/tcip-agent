"""Review flags: a person's comment asking for a second look at one place on an image, one of a
bucket's proposals, or the image as a whole, and the reply that resolves it. One record per
image holds every flag raised on it; a resolved flag is kept, never removed."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace

from tcip_store import Key

from tcip_annotation.state import Annotation, BBox, Point, Polygon

REVIEW_FLAGS_STORE = "review_flags"


@dataclass(frozen=True)
class Flag:
    """One flag: ``id`` names it, ``text`` is the comment, ``by`` who raised it (the label
    record's own spelling) and ``at`` when. Its target is one of: ``point`` (pixel ``(x, y)``) with
    the ``subject`` of the annotation flagged there; ``proposal`` (``(bucket, index)``, the
    index in that bucket's document for the image); or neither, the image as a whole. Once
    resolved it carries ``resolved_by`` and ``resolved_at``, the ``reply`` given, and ``removed``
    when the save that resolved it did so by removing the annotation it sat on."""

    id: str
    text: str
    by: str
    at: str
    point: tuple[float, float] | None = None
    subject: str | None = None
    proposal: tuple[str, int] | None = None
    resolved_by: str | None = None
    resolved_at: str | None = None
    reply: str = ""
    removed: bool = False

    @property
    def open(self) -> bool:
        """Whether nobody has resolved it."""
        return self.resolved_by is None

    def resolved(self, *, by: str, at: str, reply: str = "", removed: bool = False) -> "Flag":
        """This flag resolved by ``by`` at ``at``."""
        return replace(self, resolved_by=by, resolved_at=at, reply=reply, removed=removed)


@dataclass(frozen=True)
class FlagRequest:
    """A flag a save raises: its comment and its target, as :class:`Flag` states one."""

    text: str
    point: tuple[float, float] | None = None
    subject: str | None = None
    proposal: tuple[str, int] | None = None


def raised(request: FlagRequest, *, by: str, at: str) -> Flag:
    """``request`` as a new open flag by ``by`` at ``at`` under a freshly minted id; a blank
    comment, or a target :func:`decode_flag` would not read back, refuses (``ValueError``)."""
    import uuid

    flag = Flag(id=uuid.uuid4().hex, text=request.text.strip() if isinstance(request.text, str)
                else request.text, by=by, at=at, point=request.point, subject=request.subject,
                proposal=request.proposal)
    return decode_flag(encode_flag(flag))


def flag_key(label_key: Key) -> Key:
    """The flags record of the image whose label document ``label_key`` names."""
    return Key(REVIEW_FLAGS_STORE, label_key.root, label_key.parts)


def encode_flag(flag: Flag) -> dict:
    """``flag`` as the entry its record stores."""
    return {**asdict(flag), "point": list(flag.point) if flag.point else None,
            "proposal": list(flag.proposal) if flag.proposal else None}


def _is_text(value) -> bool:
    return isinstance(value, str) and bool(value)


def decode_flag(entry: Mapping) -> Flag:
    """One stored entry as a :class:`Flag`, or ``ValueError`` naming the entry when a key is
    missing, the id, comment, person or time is not a non-empty string, the point is not two
    numbers, the proposal is not a bucket name and an index, a point comes without a subject or
    beside a proposal, or the resolution states a person without a time."""
    try:
        point, proposal = entry["point"], entry["proposal"]
        flag = Flag(
            id=entry["id"], text=entry["text"], by=entry["by"], at=entry["at"],
            point=None if point is None else (float(point[0]), float(point[1])),
            subject=entry["subject"],
            proposal=None if proposal is None else (proposal[0], proposal[1]),
            resolved_by=entry["resolved_by"], resolved_at=entry["resolved_at"],
            reply=entry["reply"], removed=entry["removed"])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"flag entry {dict(entry)!r} does not read: {exc!r}") from exc
    well_formed = (
        all(_is_text(v) for v in (flag.id, flag.text, flag.by, flag.at))
        and (flag.point is None) == (flag.subject is None)
        and (flag.subject is None or _is_text(flag.subject))
        and not (flag.point is not None and flag.proposal is not None)
        and (flag.proposal is None or (_is_text(flag.proposal[0])
                                       and isinstance(flag.proposal[1], int)
                                       and not isinstance(flag.proposal[1], bool)
                                       and flag.proposal[1] >= 0))
        and (flag.resolved_by is None) == (flag.resolved_at is None)
        and (flag.resolved_by is None or _is_text(flag.resolved_by))
        and isinstance(flag.reply, str) and isinstance(flag.removed, bool))
    if not well_formed:
        raise ValueError(f"flag entry {dict(entry)!r} is not a comment by a person at a time on "
                         "a point with its subject, a proposal or the image")
    return flag


def flags_of(record: Mapping | None) -> list[Flag]:
    """The flags a stored record holds, oldest first; an absent record holds none. A record that
    is not ``{"flags": [...]}`` or an entry that will not decode refuses (``ValueError``)."""
    if record is None:
        return []
    entries = record.get("flags") if isinstance(record, Mapping) else None
    if not isinstance(entries, list):
        raise ValueError("a flags record holds a list under 'flags'")
    return [decode_flag(entry) for entry in entries]


def flags_record(flags: list[Flag]) -> dict:
    """``flags`` as the record :func:`flags_of` reads back."""
    return {"flags": [encode_flag(flag) for flag in flags]}


def read_flags(key: Key) -> list[Flag]:
    """Every flag the record ``key`` names holds (:func:`flags_of`)."""
    import tcip_store

    return flags_of(tcip_store.read(key, default=None))


def covers(annotation: Annotation, subject: str, point: tuple[float, float]) -> bool:
    """Whether ``annotation`` is one of ``subject`` whose geometry holds ``point``: inside or on
    the edge of a box or a polygon's rings, or exactly a placed point's own coordinate."""
    if annotation.subject != subject:
        return False
    x, y = point
    geometry = annotation.geometry
    if isinstance(geometry, BBox):
        return geometry.x1 <= x <= geometry.x2 and geometry.y1 <= y <= geometry.y2
    if isinstance(geometry, Polygon):
        from shapely.geometry import Point as ShapelyPoint

        from tcip_annotation.matching import shapely_geometry

        return bool(shapely_geometry(geometry.rings).intersects(ShapelyPoint(x, y)))
    if isinstance(geometry, Point):
        return (geometry.x, geometry.y) == (x, y)
    return False


def resolved_by_removal(flags: list[Flag], before: list[Annotation], after: list[Annotation],
                        *, by: str, at: str) -> list[Flag]:
    """``flags`` with each open one resolved as ``removed`` when an annotation of its subject
    held its point in ``before`` and none does in ``after``; every other flag unchanged."""
    def removed(flag: Flag) -> bool:
        subject, point = flag.subject, flag.point
        if not flag.open or subject is None or point is None:
            return False
        return (any(covers(a, subject, point) for a in before)
                and not any(covers(a, subject, point) for a in after))

    return [flag.resolved(by=by, at=at, removed=True) if removed(flag) else flag
            for flag in flags]
