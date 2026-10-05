"""The per-image document, ground truth or predictions, one record per image under its store key.

Schema::

    { "width": W, "height": H,
      "annotations": [
        { "subject": "<subject>",
          "bbox": [x, y, w, h],                 # COCO xywh, pixel, a box only
          "segmentation": [[x1,y1, ...], ...],  # pixel polygon, one or more rings (optional)
          "point": [x, y],                      # pixel point, a prompt or keypoint (optional)
          "attributes": {"<attribute>": "<value>"},   # attr name -> value name
          "score": 0.91,                        # predictions only
          "iscrowd": true,                      # a region of unseparated objects (optional)
          "created_by": "model:<sha>", "created_at": "...",
          "accepted_by": "user:breeder", "accepted_at": "..." } ],
      "complete": {"<subject>": [
        { "rect": [x, y, w, h], "by": "user:breeder", "at": "...",
          "digest": "<the subject's annotations>", "proposals_hidden": false } ]} }

A stored value that is not this object (``null`` included), an ``annotations`` that is not a
list, a record :func:`annotation_of_record` refuses or a completion mark :func:`completion_marks`
refuses raises :class:`UnreadableLabelDocument` naming its key.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, cast

import tcip_store
from tcip_store import Key, Version

from tcip_annotation.state import (
    Annotation, BBox, Point, Polygon, bbox_of, is_detection, polygonal,
    prediction_score,
)

ANNOTATIONS_KEY = "annotations"  # the one top-level list key
_PROV_KEYS = ("created_by", "created_at", "accepted_by", "accepted_at")


class UnreadableLabelDocument(Exception):
    """A present label document this platform's readers cannot make sense of; not a
    :class:`ValueError` subclass.
    """


def safe_score(x) -> float:
    """A confidence as it is stored: rounded to four places. A non-finite one (NaN or infinity)
    refuses (``ValueError``) naming it: no stored number stands in for a score the model did not
    give."""
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"confidence {x!r} is not a finite number, so it cannot be stored as one.")
    return round(v, 4)


def _numbers(value, key: str, count: int | None = None) -> list[float]:
    """``value`` as floats: a list of JSON numbers, exactly ``count`` of them when given. Raises
    ``ValueError`` naming ``key`` for anything else, a numeric string included.
    """
    if (not isinstance(value, list) or (count is not None and len(value) != count)
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)):
        raise ValueError(f"{key} {value!r} is not a list of {count or 'some'} numbers")
    return [float(v) for v in value]


def _polygon_of(segmentation) -> Polygon:
    """A ``segmentation``'s flat ``x, y`` rings as a :class:`Polygon`, which refuses a ring too
    short to be a shape, or ``ValueError`` naming what the value is instead."""
    if not isinstance(segmentation, list):
        raise ValueError(f"segmentation {segmentation!r} is not a list of rings")
    rings: list[list[tuple[float, float]]] = []
    for ring in segmentation:
        flat = _numbers(ring, "segmentation ring")
        if len(flat) % 2:
            raise ValueError(f"segmentation ring {ring!r} is not x, y pairs")
        rings.append(list(zip(flat[0::2], flat[1::2])))
    return Polygon(rings)


def attributes_of(raw) -> dict[str, str]:
    """Supplied attribute values, name to value name: anything but a mapping of names to non-empty
    value names raises ``ValueError``.
    """
    if not isinstance(raw, dict) or not all(isinstance(v, str) and v for v in raw.values()):
        raise ValueError(f"attributes {raw!r} are not attribute names mapped to value names")
    return dict(raw)


def iscrowd_of(raw) -> bool:
    """A supplied crowd flag: COCO's ``0``/``1`` or ``true``/``false``, else ``ValueError``."""
    if not isinstance(raw, int) or raw not in (0, 1):
        raise ValueError(f"iscrowd {raw!r} is not 0, 1, true or false")
    return bool(raw)


def box_extent_ok(bbox: BBox) -> bool:
    """Whether ``bbox`` has real extent: ``x2 > x1`` and ``y2 > y1``, on the raw corners;
    :func:`check_box_extent` refuses by the same rule, and :func:`stored_box_extent_ok` checks the
    stored grid.
    """
    return bbox.x2 > bbox.x1 and bbox.y2 > bbox.y1


def stored_box_extent_ok(bbox: BBox) -> bool:
    """Whether ``bbox`` still has positive extent once rounded to the stored 2-decimal quantum: a
    box can pass :func:`box_extent_ok` on its raw corners and still round to zero width or height
    at the grid :func:`xywh` stores.
    """
    _, _, w, h = xywh(bbox.x1, bbox.y1, bbox.x2, bbox.y2)
    return w > 0 and h > 0


def check_box_extent(bbox: BBox, *, where: str) -> None:
    """Refuse a box with no real extent: ``x2 <= x1`` or ``y2 <= y1``. ``where`` names what is
    being written (a subject, a reviewer action, or a record index).
    """
    if not box_extent_ok(bbox):
        raise ValueError(
            f"{where}: box (x1={bbox.x1}, y1={bbox.y1}, x2={bbox.x2}, y2={bbox.y2}) has no "
            "positive extent; a box needs x2 > x1 and y2 > y1"
        )


def ring_vertex(vertex) -> tuple:
    """A polygon ring vertex's ``(x, y)`` as given, from an ``[x, y]`` pair or an ``{"x":, "y":}``
    mapping, or ``ValueError`` for any other shape. The values are not interpreted here.
    """
    if isinstance(vertex, Mapping) and "x" in vertex and "y" in vertex:
        return vertex["x"], vertex["y"]
    if isinstance(vertex, (list, tuple)) and len(vertex) == 2:
        return vertex[0], vertex[1]
    raise ValueError(f"ring vertex {vertex!r} is not an [x, y] pair or an {{x, y}} mapping")


def annotation_from_payload(payload: Mapping) -> Annotation:
    """One client payload dict's content as an :class:`Annotation`, carrying no provenance.

    The payload is translated into the document's own record shape and decoded by
    :func:`annotation_of_record`, so a supplied value that route refuses raises ``ValueError`` here
    too. The translation: ``bbox`` corners ``[x1, y1, x2, y2]`` become the record's ``[x, y, w,
    h]``; ``rings`` (a list of rings) and ``points`` (one ring) become ``segmentation``, ``rings``
    winning when both are given, each vertex an ``[x, y]`` pair or an ``{"x", "y"}`` mapping;
    ``point`` (a single placed prompt or keypoint) is kept as it is. A polygon then wins over a box
    and a box over a point, as in a stored record. A payload's provenance keys and ``score`` are
    not read. A payload box quantizes to the stored two-decimal grid (:func:`xywh`).
    """
    payload = annotation_object(payload)
    rings, corners = payload.get("rings"), payload.get("bbox")
    if rings is None and payload.get("points") is not None:
        rings = [payload["points"]]
    segmentation = rings if not isinstance(rings, list) else [
        [c for v in ring for c in ring_vertex(v)] if isinstance(ring, list) else ring
        for ring in rings]
    return annotation_of_record({
        "subject": payload.get("subject"),
        "segmentation": segmentation,
        "bbox": xywh(*_numbers(corners, "bbox", 4)) if corners is not None else None,
        "point": payload.get("point"),
        "attributes": payload.get("attributes"),
        "iscrowd": payload.get("iscrowd"),
    })


def stamped(contents: Iterable[Annotation], stored: Iterable[Annotation], *, actor: str,
            now: str) -> list[Annotation]:
    """``contents`` with their provenance: each one whose stored content (:func:`stored_content`)
    equals a not yet claimed record of ``stored`` takes that record's provenance, and every other
    one is authored by ``actor`` at ``now`` with no sign-off. Refuses (``ValueError``) a blank
    ``actor``."""
    if not actor.strip():
        raise ValueError("a saved annotation records its producer; name who or what wrote it")
    unclaimed: dict[str, list[Annotation]] = {}
    for record in stored:
        unclaimed.setdefault(_content_key(record), []).append(record)
    out = []
    for content in contents:
        held = unclaimed.get(_content_key(content))
        source = held.pop(0) if held else Annotation(content.subject, created_by=actor,
                                                     created_at=now)
        out.append(replace(content, **_held_provenance(source)))
    return out


def _content_key(a: Annotation) -> str:
    """``a``'s stored content as one canonical string."""
    return json.dumps(stored_content(a), sort_keys=True)


def annotation_object(o) -> dict:
    """``o`` when it is an annotation object (a JSON object), else ``ValueError`` naming it."""
    if not isinstance(o, dict):
        raise ValueError(f"is {o!r}, not an annotation object")
    return o


def annotation_of_record(o) -> Annotation:
    """One record of a per-image document as an :class:`Annotation`: the one per-record decoder.

    A key that is absent or ``null`` reads as absent, for every optional field: a ground-truth
    record has no ``score``, an image-level label has no geometry, a record with no ``iscrowd``
    is no crowd region. A key that is present with a value that is not what the schema states
    raises ``ValueError`` naming the key, never reads as absent, whichever geometry wins
    precedence: a ``bbox`` that is not four numbers or has no positive extent
    (:func:`check_box_extent`), a ``segmentation`` that is not rings of three or more points
    (:func:`_polygon_of`), a ``point`` that is not two numbers, a ``score`` that is not a number,
    attributes that are not value names (:func:`attributes_of`), a crowd flag that is not one
    (:func:`iscrowd_of`). So does a record that is not an object (:func:`annotation_object`) or
    that :class:`Annotation` refuses to construct (no non-empty string ``subject``). Geometry
    precedence is rings, then box, then point, over the keys present: a polygon is the source of
    truth and its box is derived from it.
    """
    o = annotation_object(o)
    subject = o.get("subject")
    present = {k: o[k] for k in ("segmentation", "bbox", "point", "score", "attributes", "iscrowd",
                                 *_PROV_KEYS) if o.get(k) is not None}
    polygon = _polygon_of(present["segmentation"]) if "segmentation" in present else None
    box = None
    if "bbox" in present:
        x, y, w, h = _numbers(present["bbox"], "bbox", 4)
        box = BBox(x, y, x + w, y + h)
        check_box_extent(box, where=f"subject {subject!r}")
    point = _numbers(present["point"], "point", 2) if "point" in present else None
    score = _numbers([present["score"]], "score", 1)[0] if "score" in present else None
    return Annotation(
        subject=subject, geometry=polygon or box or (Point(*point) if point else None),
        attributes=attributes_of(present["attributes"]) if "attributes" in present else {},
        score=score, iscrowd="iscrowd" in present and iscrowd_of(present["iscrowd"]),
        **{k: present[k] for k in _PROV_KEYS if k in present},
    )


def _annotations_of(data: Any) -> list[Annotation]:
    """Parse a stored per-image document into :class:`Annotation` records.

    Raises :class:`UnreadableLabelDocument` for a document that is not an object (``null``
    included), one whose ``annotations`` is not a list (covers it being absent or ``null`` too),
    and for any record :func:`annotation_of_record` refuses, naming the record's index.
    """
    if not isinstance(data, dict):
        raise UnreadableLabelDocument(f"is a {type(data).__name__}, not the object a label "
                                      "document is")
    raw = data.get(ANNOTATIONS_KEY)
    if not isinstance(raw, list):
        raise UnreadableLabelDocument(
            f"{ANNOTATIONS_KEY!r} is {raw!r}, not the list a label document holds"
        )
    out: list[Annotation] = []
    for i, o in enumerate(raw):
        try:
            out.append(annotation_of_record(o))
        except ValueError as exc:
            raise UnreadableLabelDocument(f"record {i} {exc}") from exc
    return out


# ── completion marks ───────────────────────────────────────────────────────

COMPLETION_KEY = "complete"
"""The document key holding each subject's completion marks."""

SubjectState = Literal["complete", "negative", "partial", "unannotated"]
"""What one document says about one subject: finished with or without annotations of it
(``complete``, ``negative``), or not finished with or without them (``partial``,
``unannotated``)."""

_STATE_OF: dict[tuple[bool, bool], SubjectState] = {
    (True, True): "complete", (True, False): "negative",
    (False, True): "partial", (False, False): "unannotated"}
"""Each subject state by whether the subject is finished and whether annotations hold it."""

FINISHED_STATES = frozenset(s for (finished, _), s in _STATE_OF.items() if finished)
"""The states :meth:`LabelDocument.state` answers for a subject :meth:`LabelDocument.finished`."""


@dataclass(frozen=True)
class CompletionMark:
    """A person's attestation that every instance of one subject inside ``rect`` (pixel ``[x, y,
    w, h]``) is annotated: who (``by``) and when (``at``), the :func:`subject_digest` of that
    subject's annotations when it was made, and whether proposals were hidden while it was made.
    """

    rect: tuple[float, float, float, float]
    by: str
    at: str
    digest: str
    proposals_hidden: bool


def canonical_digest(value) -> str:
    """The platform's content digest of a JSON value: sha256 over its compact serialization, in
    the value's own key order, first sixteen hex digits."""
    canonical = json.dumps(value, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def subject_digest(annotations: Iterable[Annotation], subject: str) -> str:
    """The :func:`canonical_digest` of ``subject``'s annotations as stored, provenance aside, in
    an order independent of the order they were drawn in."""
    return canonical_digest(sorted(_content_key(a) for a in annotations if a.subject == subject))


def _live(marks: Mapping[str, list[CompletionMark]],
          annotations: list[Annotation]) -> dict[str, list[CompletionMark]]:
    """Each subject's marks whose digest still names that subject's annotations."""
    live = {}
    for subject, held in marks.items():
        digest = subject_digest(annotations, subject)
        kept = [m for m in held if m.digest == digest]
        if kept:
            live[subject] = kept
    return live


def completion_marks(data: dict,
                     annotations: list[Annotation]) -> dict[str, list[CompletionMark]]:
    """The live completion marks a parsed document holds, by subject: the ones whose digest still
    names that subject's ``annotations``. Raises :class:`UnreadableLabelDocument` naming the
    subject and index of a mark that is not a rect of four numbers, a person, a time, a digest and
    a ``proposals_hidden`` flag."""
    raw = data.get(COMPLETION_KEY) or {}
    if not isinstance(raw, dict):
        raise UnreadableLabelDocument(f"{COMPLETION_KEY!r} is {raw!r}, not marks by subject")
    marks: dict[str, list[CompletionMark]] = {}
    for subject, held in raw.items():
        if not isinstance(held, list):
            raise UnreadableLabelDocument(f"{subject!r} completion marks are {held!r}, not a list")
        for i, m in enumerate(held):
            try:
                marks.setdefault(subject, []).append(CompletionMark(
                    rect=cast(tuple, tuple(_numbers(m["rect"], "rect", 4))),
                    by=_text(m["by"], "by"), at=_text(m["at"], "at"),
                    digest=_text(m["digest"], "digest"),
                    proposals_hidden=_flag(m["proposals_hidden"])))
            except (KeyError, TypeError, ValueError) as exc:
                raise UnreadableLabelDocument(
                    f"{subject!r} completion mark {i} is not a mark: {exc!r}") from exc
    return _live(marks, annotations)


def _text(value, key: str) -> str:
    """``value`` when it is a non-empty string, else ``ValueError`` naming ``key``."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} {value!r} is not a non-empty string")
    return value


def _flag(value) -> bool:
    """``value`` when it is a boolean, else ``ValueError``."""
    if not isinstance(value, bool):
        raise ValueError(f"proposals_hidden {value!r} is not true or false")
    return value


def covers(marks: Iterable[CompletionMark], rect: tuple[float, float, float, float]) -> bool:
    """Whether the union of ``marks``' rects covers the half-open pixel ``rect`` ``(x0, y0, x1,
    y1)``, every point of it."""
    boxes = [(x, y, x + w, y + h) for x, y, w, h in (m.rect for m in marks)]
    x0, y0, x1, y1 = rect
    xs = sorted({x0, x1, *(min(max(v, x0), x1) for b in boxes for v in (b[0], b[2]))})
    ys = sorted({y0, y1, *(min(max(v, y0), y1) for b in boxes for v in (b[1], b[3]))})
    return all(
        any(b[0] <= (ax + bx) / 2 <= b[2] and b[1] <= (ay + by) / 2 <= b[3] for b in boxes)
        for ax, bx in zip(xs, xs[1:]) for ay, by in zip(ys, ys[1:]))


@dataclass(frozen=True)
class LabelDocument:
    """One image's label document as read: its annotations, its ``width`` and ``height``
    (``None`` for a document that states none), and each subject's live completion marks."""

    annotations: list[Annotation]
    width: int | None
    height: int | None
    marks: dict[str, list[CompletionMark]]

    def finished(self, subject: str) -> bool:
        """Whether this document's live marks for ``subject`` cover the whole image."""
        return bool(self.width and self.height and covers(
            self.marks.get(subject, []), (0, 0, self.width, self.height)))

    def state(self, subject: str) -> SubjectState:
        """What this document says about ``subject``: :meth:`finished` or not, holding it by
        :func:`annotations_hold_subject` or not."""
        return _STATE_OF[self.finished(subject),
                         annotations_hold_subject(self.annotations, subject)]


def annotations_hold_subject(annotations: Iterable[Annotation], subject: str) -> bool:
    """Whether any of ``annotations`` names ``subject``, geometry or not: an image-level record
    counts."""
    return any(a.subject == subject for a in annotations)


def label_document(data: dict) -> LabelDocument:
    """A stored per-image document as a :class:`LabelDocument`; refuses as
    :func:`_annotations_of` and :func:`completion_marks` do."""
    annotations = _annotations_of(data)
    return LabelDocument(annotations=annotations, width=data.get("width"),
                         height=data.get("height"), marks=completion_marks(data, annotations))


NO_DOCUMENT = LabelDocument(annotations=[], width=None, height=None, marks={})
"""What :func:`read_document_versioned` answers for a key holding no record."""


# ── reader ─────────────────────────────────────────────────────────────────


def document_at(key: Key, stored: tcip_store.Versioned) -> LabelDocument:
    """``stored``, what the store answered for ``key``, as a :class:`LabelDocument`:
    :data:`NO_DOCUMENT` at ``Version.ABSENT``, else its value decoded, a value
    :func:`label_document` refuses raising :class:`UnreadableLabelDocument` naming ``key``."""
    if stored.version == Version.ABSENT:
        return NO_DOCUMENT
    try:
        return label_document(stored.value)
    except UnreadableLabelDocument as exc:
        raise UnreadableLabelDocument(
            f"{key.store}{list(key.parts)} under {key.root}: {exc}") from exc


def read_stored(key: Key, **default: Any) -> tcip_store.Versioned:
    """The store's read of ``key`` (:func:`tcip_store.read_versioned`, ``default`` passed
    through); a record that does not decode, and an absent one read with no default
    (``NotFound``), raise :class:`UnreadableLabelDocument` with the store's own message."""
    try:
        return tcip_store.read_versioned(key, **default)
    except (tcip_store.DecodeError, tcip_store.NotFound) as exc:
        raise UnreadableLabelDocument(str(exc)) from exc


def read_document_versioned(key: Key) -> tuple[LabelDocument, Version]:
    """The editor's read: the document ``key`` names (:func:`document_at`, :data:`NO_DOCUMENT`
    where the store holds none) and the version it was read at. A record that does not decode
    raises :class:`UnreadableLabelDocument` naming it."""
    stored = read_stored(key, default=None)
    return document_at(key, stored), stored.version


def read_label_document(key: Key, version: str | None = None) -> LabelDocument:
    """The document ``key`` names, read through the store's required read, at ``version`` (a
    version token) when one is given. No record, another version, or a record that does not
    decode raises :class:`UnreadableLabelDocument` naming it."""
    stored = read_stored(key)
    if version is not None and stored.version.token != version:
        raise UnreadableLabelDocument(f"{key.store}{list(key.parts)} under {key.root} is at "
                                      f"version {stored.version.token}, not {version}, the one "
                                      "measured")
    return document_at(key, stored)


def read_predictions(key: Key) -> list[Annotation]:
    """A prediction document's records, each stating its ``score``: a record stating none raises
    :class:`UnreadableLabelDocument` naming it (:func:`~tcip_annotation.state.prediction_score`).
    """
    annotations = read_label_document(key).annotations
    for i, a in enumerate(annotations):
        try:
            prediction_score(a)
        except ValueError as exc:
            raise UnreadableLabelDocument(f"record {i} {exc}") from exc
    return annotations


def detection_annotations(annotations: Iterable[Annotation]) -> list[Annotation]:
    """``annotations`` narrowed to the detections a count counts
    (:func:`~tcip_annotation.state.is_detection`: a crowd region and a ``Point`` excluded).
    """
    return [a for a in annotations if is_detection(a)]


# ── reference admissibility ────────────────────────────────────────────────

PERSON_IDENTITY_PREFIX = "user:"


def is_person(identity: str | None) -> bool:
    """Whether ``identity`` names a person: :data:`PERSON_IDENTITY_PREFIX` followed by a
    non-blank name."""
    return (identity is not None and identity.startswith(PERSON_IDENTITY_PREFIX)
            and bool(identity.removeprefix(PERSON_IDENTITY_PREFIX).strip()))


def is_person_signoff(a: Annotation) -> bool:
    """True when ``a``'s ``accepted_by`` names a person (:func:`is_person`)."""
    return is_person(a.accepted_by)


def is_unadjudicated_prediction(a: Annotation) -> bool:
    """True when ``a`` is a model's own output that no human has taken responsibility for: a set
    ``score`` (see :class:`~tcip_annotation.state.Annotation`). Accepting a prediction into ground
    truth drops its ``score`` and stamps ``accepted_by``.
    """
    return a.score is not None


def is_unadjudicated_agent_authorship(a: Annotation) -> bool:
    """True when ``a`` names a producer that is not a person and no reviewer has signed off on it.

    A person's recorded identity is :func:`is_person`'s, while a tool producer stays bare
    (``sam``, an agent's own name) or carries a ``model:<checkpoint>`` stamp. An annotation
    carrying no ``created_by`` at all is not claimed by this rule. A set ``accepted_by`` is a
    reviewer taking responsibility for the record.
    """
    if not a.created_by or is_person(a.created_by):
        return False
    return not a.accepted_by


def authorship_of(a: Annotation) -> str:
    """``a``'s authorship, one of ``"person"``, ``"tool"``, ``"tool_accepted"`` or
    ``"unattributed"``; ``"tool"`` is :func:`is_unadjudicated_agent_authorship`.
    """
    if not a.created_by:
        return "unattributed"
    if is_unadjudicated_agent_authorship(a):
        return "tool"
    if is_person(a.created_by):
        return "person"
    return "tool_accepted"


@dataclass
class ProvenanceFacts:
    """The provenance classification over one list of annotations: ``scored``
    (:func:`is_unadjudicated_prediction`) and ``machine_authored``
    (:func:`is_unadjudicated_agent_authorship`), plus ``no_created_by`` and
    ``not_positively_a_persons``, index lists into the annotations passed in.
    """

    total: int
    scored: int
    machine_authored: list[str]
    no_created_by: list[int]
    not_positively_a_persons: list[int]
    """Index of every record whose ``created_by`` is not a person's and whose ``accepted_by`` is
    not a person's either (present or not): the canopy rule's own stricter test, which checks
    ``accepted_by``'s identity rather than merely its presence the way
    :func:`is_unadjudicated_agent_authorship` does. Never includes a record already counted under
    ``no_created_by``: that record's ``created_by`` names nobody to test as a person or not."""


def provenance_facts(annotations: list[Annotation]) -> ProvenanceFacts:
    """The :class:`ProvenanceFacts` classification over ``annotations``."""
    scored = 0
    machine_authored: list[str] = []
    no_created_by: list[int] = []
    not_positively_a_persons: list[int] = []
    for i, a in enumerate(annotations):
        if is_unadjudicated_prediction(a):
            scored += 1
        if is_unadjudicated_agent_authorship(a):
            machine_authored.append(str(a.created_by))
        if not a.created_by:
            no_created_by.append(i)
        elif not is_person(a.created_by):
            if not is_person_signoff(a):
                not_positively_a_persons.append(i)
    return ProvenanceFacts(
        total=len(annotations), scored=scored, machine_authored=machine_authored,
        no_created_by=no_created_by, not_positively_a_persons=not_positively_a_persons,
    )


def require_reference_ground_truth(annotations: list[Annotation]) -> None:
    """Refuse the label documents' ``annotations`` as a measurement reference when only the model
    stands behind them.

    Refuses on a record carrying a prediction ``score`` and on a record an agent authored as
    ground truth with no person's ``accepted_by``. Refuses the whole reference, never by dropping
    the offending records.
    """
    facts = provenance_facts(annotations)
    scored, total, agent_authored = facts.scored, facts.total, facts.machine_authored
    if scored:
        raise ValueError(
            f"{scored} of {total} annotations in the reference carry a prediction score, so they are "
            "the model's own output that no human has ruled on, and a reference built from them "
            "measures the model against itself rather than against a measurement. Annotate this "
            "reference, accept the model's proposals in the editor so each record is a person's "
            "call and carries their accepted_by, or validate against a breeder-confirmed sample of "
            "the model's outputs instead."
        )
    if agent_authored:
        producers = ", ".join(sorted(set(agent_authored)))
        raise ValueError(
            f"{len(agent_authored)} of {total} annotations in the reference are authored by "
            f"{producers}, which names no person under this platform's {PERSON_IDENTITY_PREFIX}"
            "<name> convention, and carry no accepted_by. Ground truth an agent wrote and no "
            "human has adjudicated is not a calibration or holdout reference. The lighter path "
            "is to have a person confirm these records in the editor so each one carries their "
            "accepted_by, rather than hand-annotating the whole reference."
        )


# ── writer ─────────────────────────────────────────────────────────────────


def xywh(x1: float, y1: float, x2: float, y2: float, *, on_grid: bool = True) -> list[float]:
    """A box in corner form as this schema's ``bbox``: COCO ``[x, y, w, h]``.

    A pixel box is put on the 2-decimal quantum, the grid the stored document lives on and the grid
    anything comparing a stored box against another box has to be on. ``on_grid=False`` is for a
    box on no pixel grid at all (a record kept on the normalized unit square).

    The inverse, reading such a record back, is ``BBox(x, y, x + w, y + h)`` in
    :func:`_annotations_of`.
    """
    box = [x1, y1, x2 - x1, y2 - y1]
    return [round(v, 2) for v in box] if on_grid else box


def _stored_bbox_or_raise(bbox: BBox, *, where: str) -> list[float]:
    """A box's stored ``xywh``, refusing one whose rounded extent is not positive
    (:func:`stored_box_extent_ok`).
    """
    if not stored_box_extent_ok(bbox):
        raise ValueError(
            f"{where}: box (x1={bbox.x1}, y1={bbox.y1}, x2={bbox.x2}, y2={bbox.y2}) rounds to no "
            "positive extent at the document's stored grid"
        )
    return xywh(bbox.x1, bbox.y1, bbox.x2, bbox.y2)


def _rounded_rings(rings: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """A polygon's rings with each vertex rounded to the document's stored 2-decimal grid."""
    return [[(round(float(x), 2), round(float(y), 2)) for x, y in ring] for ring in rings]


def geometry_extent_ok(geometry: BBox | Polygon) -> bool:
    """Whether ``geometry`` still has positive extent once written to its stored grid. A
    :class:`Polygon`'s vertices round to two decimals before its box is derived, the same order
    :func:`document_payload` stores them in.
    """
    if polygonal(geometry):
        return stored_box_extent_ok(bbox_of(Polygon(_rounded_rings(geometry.rings))))
    return stored_box_extent_ok(cast(BBox, geometry))


def stored_content(a: Annotation) -> dict:
    """One annotation as its stored JSON object, provenance aside: its content on the stored grid.
    Its attributes are read through :func:`attributes_of`, raising ``ValueError`` on a value the
    reader would refuse.
    """
    rec: dict = {"subject": a.subject}
    geom = a.geometry
    if polygonal(geom):
        if not geometry_extent_ok(geom):
            raise ValueError(f"{a.subject!r} annotation's polygon rounds to no positive extent at "
                             "the document's stored grid")
        rec["segmentation"] = [[c for xy in ring for c in xy] for ring in _rounded_rings(geom.rings)]
    elif isinstance(geom, BBox):
        rec["bbox"] = _stored_bbox_or_raise(geom, where=f"{a.subject!r} annotation")
    elif isinstance(geom, Point):
        rec["point"] = [round(geom.x, 2), round(geom.y, 2)]
    if a.attributes:
        rec["attributes"] = attributes_of(a.attributes)
    if a.score is not None:
        rec["score"] = safe_score(a.score)
    if a.iscrowd:
        rec["iscrowd"] = True
    return rec


def _held_provenance(a: Annotation) -> dict:
    """The provenance fields ``a`` holds, under the schema's own key names; a field it does not
    hold is absent, never ``None``."""
    return {k: getattr(a, k) for k in _PROV_KEYS if getattr(a, k) is not None}


def client_annotation(a: Annotation) -> dict:
    """``a`` as the dict a client reads.

    ``bbox`` in corner form ``[x1, y1, x2, y2]``; ``rings`` for a polygon, every ring; ``point``
    for a placed prompt or keypoint, the stored key; ``attributes``; ``score`` for a prediction;
    ``iscrowd`` always stated; and the provenance fields the record holds
    (:func:`_held_provenance`).
    """
    out: dict = {"subject": a.subject, "attributes": dict(a.attributes), "iscrowd": a.iscrowd}
    geom = a.geometry
    if polygonal(geom):
        out["rings"] = [[[x, y] for x, y in ring] for ring in geom.rings]
    elif isinstance(geom, BBox):
        out["bbox"] = [geom.x1, geom.y1, geom.x2, geom.y2]
    elif isinstance(geom, Point):
        out["point"] = [geom.x, geom.y]
    if a.score is not None:
        out["score"] = a.score
    return {**out, **_held_provenance(a)}


def document_payload(annotations, width: int, height: int, *,
                     keep_empty: bool = False,
                     marks: Mapping[str, list[CompletionMark]] | None = None) -> dict | None:
    """The record one image's per-image document holds: its frame, its annotations and, of
    ``marks``, each subject's whose digest still names that subject's annotations here. Refuses
    (``ValueError``) the first geometry the stored grid collapses. ``None`` when neither a record
    nor a mark survives and ``keep_empty`` is not set.
    """
    annotations = list(annotations)
    records = [{**stored_content(a), **_held_provenance(a)} for a in annotations]
    live = _live(marks or {}, annotations)
    if not records and not live and not keep_empty:
        return None
    payload: dict = {"width": int(width), "height": int(height), ANNOTATIONS_KEY: records}
    if live:
        payload[COMPLETION_KEY] = {
            subject: [{"rect": list(m.rect), "by": m.by, "at": m.at, "digest": m.digest,
                       "proposals_hidden": m.proposals_hidden} for m in held]
            for subject, held in sorted(live.items())}
    return payload


def write_label_document(key: Key, annotations, width: int, height: int, *,
                         keep_empty: bool = False, expect: Version | None = None,
                         marks: Mapping[str, list[CompletionMark]] | None = None,
                         ) -> Version | None:
    """Write all of an image's annotations, and the completion ``marks`` still live over them
    (:func:`document_payload`), as the document ``key`` names. An empty list writes an empty
    document under ``keep_empty``, else removes the document. ``expect`` (a version
    :func:`read_document_versioned` answered) makes the write a compare-and-set raising
    ``VersionConflict`` with nothing written. Returns the new version, or ``None`` when the
    document was removed.
    """
    payload = document_payload(annotations, width, height, keep_empty=keep_empty, marks=marks)
    if payload is None:
        tcip_store.delete(key, expect=expect)
        return None
    return tcip_store.replace(key, payload, expect=expect)


# ── attribute values as ids ──────────────────────────────────────────────────


UNASSESSED = -1
"""The id an attribute column carries for an instance nobody assessed for that attribute."""


class AttributeRecord(Protocol):
    """An attribute as a scope declares it: its ``name`` and its ``values`` in declared order, a
    value's id being its position there."""

    @property
    def name(self) -> str: ...

    @property
    def values(self) -> tuple[str, ...]: ...


class UndeclaredValue(ValueError):
    """A record carries a value its attribute does not declare."""


def attribute_ids(a: Annotation, subject: str,
                  attributes: Iterable[AttributeRecord]) -> list[int] | None:
    """The id of ``a``'s value under each of ``attributes``, in their order, :data:`UNASSESSED`
    where ``a`` carries no value for one; ``None`` for a record of a subject other than
    ``subject``. Refuses (:class:`UndeclaredValue`) a value its attribute does not declare, naming
    both."""
    if a.subject != subject:
        return None
    row = []
    for attribute in attributes:
        value = a.attributes.get(attribute.name)
        if value is not None and value not in attribute.values:
            raise UndeclaredValue(
                f"a record of {subject!r} carries {attribute.name}={value!r}, which that "
                f"attribute does not declare (it declares {list(attribute.values)}).")
        row.append(UNASSESSED if value is None else attribute.values.index(value))
    return row


def attribute_values(ids: Iterable[int], attributes: Iterable[AttributeRecord]) -> dict[str, str]:
    """``{name: value}`` for each of ``attributes`` the id row ``ids`` assesses, the inverse of
    :func:`attribute_ids`."""
    return {attribute.name: attribute.values[int(i)]
            for attribute, i in zip(attributes, ids, strict=True) if int(i) != UNASSESSED}
