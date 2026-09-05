"""Per-image JSON: the canonical on-disk label format (ground truth + predictions).

One JSON file per image, holding every subject's annotations by name. Each annotation carries its
``subject``, an optional geometry (``bbox`` xywh, ``segmentation`` polygon, ``point`` [x,y], or none
for an image/plant-level label), its attribute values by name, an optional ``score`` (predictions),
and provenance (``created_by/at``, ``accepted_by/at``), so a prediction's origin travels with it into
ground truth on accept, with no sidecar.

Schema::

    { "image": "<stem>", "width": W, "height": H,
      "annotations": [
        { "subject": "<subject>",
          "bbox": [x, y, w, h],                 # COCO xywh, pixel      (optional)
          "segmentation": [[x1,y1, ...], ...],  # pixel polygon, one or more rings (optional)
          "point": [x, y],                      # pixel point, a prompt or keypoint (optional)
          "attributes": {"<attribute>": "<value>"},   # attr name -> value name
          "score": 0.91,                        # predictions only
          "created_by": "sam", "created_at": "...",
          "accepted_by": "user:breeder", "accepted_at": "..." } ] }

Integer class ids never appear on disk; a name→id assignment is a per-training-run artifact
(:mod:`tcip_mcp.class_registry`). :func:`to_coco_dataset` takes that run's ``id_map`` (a plain dict,
so this package never imports tcip-mcp) and assigns COCO ``category_id`` from it.

Negative invariant: a missing file is unannotated, and a present file with ``"annotations": []`` is
*still* unannotated until a human marks that image Complete, recorded as ``"negative"`` in
``.tcip/state/image_status.json``, scoped to the subject. Only that confirmation makes it a training
negative; :func:`to_coco_dataset` skips an unconfirmed empty. An annotation with a subject but no
geometry (an image-level label) is a real annotation: it keeps the image out of the empty-negative
bucket and never collapses to nothing.

A missing file reads as unannotated (``[]``), and so does the platform's own empty document
(``{"annotations": []}``). A *present* document this format cannot make sense of (undecodable text,
a non-dict document, an ``annotations`` that is not a list, a record that is not a dict or carries
no string ``subject``, or a stored box with no positive extent) raises
:class:`UnreadableLabelDocument` instead of reading as empty: an unreadable file is not the same
fact as no file, and reading one as empty would train a corrupt document as a confirmed negative.
Writers and readers are symmetric, and a degenerate polygon (<3 points) is skipped on write so it
can never masquerade as an empty record.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import tcip_store
from tcip_store import (
    Key,
    SchemaVersionRefused,
    StoreDescriptor,
    Version,
    check_schema_version,
    get_descriptor,
    register_store,
)
from tcip_store.file_backend import RootedFileLocator

from tcip_annotation.state import Annotation, BBox, Point, Polygon, bbox_of

ANNOTATIONS_KEY = "annotations"  # the one top-level list key; format_io.detect_format shares it
_PROV_KEYS = ("created_by", "created_at", "accepted_by", "accepted_at")


# ── the store (tcip-annotation must not depend on tcip-mcp) ───────────────────

ANNOTATION_RECORDS_STORE = "annotation_records"
_ANNOTATION_RECORD_LOCATOR = RootedFileLocator(suffix=".json")
register_store(
    StoreDescriptor(
        name=ANNOTATION_RECORDS_STORE,
        kind="blob",
        key_fields=("stem",),
        frozen=True,
        path_readable=True,
        locator=_ANNOTATION_RECORD_LOCATOR,
    )
)


def annotation_record_key(directory: str | Path, stem: str) -> Key:
    """One image's per-image JSON document, addressed by the directory that holds it.

    The generic form, for a tree no layout resolver describes (a materialized split's
    ``labels/``, an export bucket) and for a caller using this package on its own. A caller that
    holds a dataset root mints a layout-aware key from its own resolver instead and hands that to
    :func:`write_annotations`; both address the same file and take the same lock, since the lock
    is the file's, not the store name's.
    """
    return Key(ANNOTATION_RECORDS_STORE, str(Path(directory).absolute()), (str(stem),))


def _record_key(target: Key | str | Path) -> Key:
    """The key ``target`` names: a key passed straight through, or a path placed generically."""
    if isinstance(target, Key):
        return target
    path = Path(target).absolute()
    return annotation_record_key(path.parent, path.stem)


def _document_bytes(payload: dict) -> bytes:
    """The exact on-disk bytes of one per-image document.

    ``allow_nan=False``: a non-finite value here is a bug sanitized before this call, and failing
    loudly beats emitting non-standard ``NaN`` that strict parsers (JS, jq) reject.
    """
    return json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8")


class UnreadableLabelDocument(Exception):
    """A present label document this platform's readers cannot make sense of.

    Deliberately not a :class:`ValueError`: a handler written to catch format detection's own
    ``ValueError`` (an unrecognized-but-decodable shape, a legitimate "not this kind of file"
    answer) must not also absorb a document that is flatly broken and silently render or train it
    as unlabeled.
    """


def parse_json_document(text: str, *, source: str) -> dict:
    """A JSON document's parsed dict, from its raw text: decode and dict-shape only.

    Raises :class:`UnreadableLabelDocument`, naming ``source``, for text that does not decode as
    JSON or that decodes to something other than a dict. Carries no opinion on
    ``schema_version``: a caller that has not yet decided whether ``source`` is this platform's
    own per-image document or an interop shape (COCO, or an unrecognized one) reads through here
    first, and applies :func:`check_annotation_record_version` itself only once it knows which.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise UnreadableLabelDocument(f"{source} does not decode as JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise UnreadableLabelDocument(
            f"{source} decodes to a {type(data).__name__}, not the object a label document is"
        )
    return data


def check_annotation_record_version(data: dict, *, source: str) -> None:
    """Refuse ``data`` when it carries a ``schema_version`` this platform's own per-image reader
    (:data:`ANNOTATION_RECORDS_STORE`) does not accept.

    Applies only once a caller has confirmed ``data`` is this platform's own annotation-records
    document: an interop shape (a COCO document's own top-level ``schema_version``, should it
    carry one) is never checked against this store's ceiling, since the two formats' version
    fields name unrelated documents.
    """
    try:
        check_schema_version(get_descriptor(ANNOTATION_RECORDS_STORE), data)
    except SchemaVersionRefused as exc:
        raise UnreadableLabelDocument(f"{source}: {exc}") from exc


def parse_label_document(text: str, *, source: str) -> dict:
    """A per-image label document's parsed dict, from its raw text.

    Raises :class:`UnreadableLabelDocument`, naming ``source``, for text that does not decode as
    JSON, that decodes to something other than a dict (:func:`parse_json_document`), or that
    carries a ``schema_version`` this reader does not accept (:func:`check_annotation_record_version`):
    none of the three is a document this schema can read, and reading any of them as empty would
    train an unreadable file as an unannotated image. The one choke point every reader of *this
    platform's own* per-image document shares, so ``load_label_document`` and
    ``annotations_from_bytes`` cannot disagree about what a document means. A caller reading a
    document whose format is not yet decided (format detection, an interop parse) reads through
    :func:`parse_json_document` instead, applying this platform's version ceiling only once it
    has confirmed the document is this store's own shape.
    """
    data = parse_json_document(text, source=source)
    check_annotation_record_version(data, source=source)
    return data


def _decode_label_bytes(data: bytes, *, source: str) -> str:
    """A label document's bytes, decoded strictly as UTF-8, a leading byte-order mark accepted.

    ``utf-8-sig``: a UTF-8 byte-order mark encodes the same text as the same document without
    one, so a document this platform (or an operator's conforming script) wrote under a tool
    that stamps one must read identically to one that does not. Bytes that decode with a BOM
    stripped but are not valid UTF-8 otherwise, or that carry no valid UTF-8 at all (a UTF-16
    document, for instance), still raise :class:`UnreadableLabelDocument`, naming ``source``: the
    one decode policy every reader of a label document's raw bytes shares, so a file reader and a
    store-backed reader over the same bytes cannot disagree about whether they read.
    """
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UnreadableLabelDocument(f"{source} is not valid UTF-8: {exc}") from exc


def load_label_document(path: str | Path) -> dict:
    """A per-image label document's parsed dict, read from ``path``.

    Raises :class:`UnreadableLabelDocument`, naming ``path``, when the file cannot be opened (a
    permission error, a directory where a file was expected), when its bytes are not valid UTF-8,
    or when :func:`parse_label_document` refuses its contents. Callers check for a missing path
    themselves: this function is only ever called once a document is known to be present, so it
    does not special-case absence.
    """
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as exc:
        raise UnreadableLabelDocument(f"{p} could not be opened: {exc}") from exc
    return parse_label_document(_decode_label_bytes(data, source=str(p)), source=str(p))


def load_json_document(path: str | Path) -> dict:
    """A JSON document's parsed dict, read from ``path``: decode and dict-shape only, no
    ``schema_version`` check.

    For a reader that has not yet decided whether ``path`` is this platform's own per-image
    document or an interop shape (format detection, a COCO parse): see :func:`load_label_document`
    for the version-checked per-image reader. Raises :class:`UnreadableLabelDocument`, naming
    ``path``, for the same unopenable-file and undecodable-bytes cases that one raises for.
    """
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as exc:
        raise UnreadableLabelDocument(f"{p} could not be opened: {exc}") from exc
    return parse_json_document(_decode_label_bytes(data, source=str(p)), source=str(p))


def annotations_from_bytes(data: bytes, *, source: str) -> list[Annotation]:
    """The typed annotation records a label document's raw bytes hold.

    The one path from bytes to records: the shared decode, :func:`parse_label_document`, then
    the record construction :func:`read_annotations` performs, so a caller holding a document's
    bytes (a store-backed reader, a parse memo keyed on the bytes it read) reads exactly what the
    file reader would. Raises :class:`UnreadableLabelDocument`, naming ``source``, for anything
    the file reader would refuse.
    """
    return _annotations_of(parse_label_document(_decode_label_bytes(data, source=source),
                                                source=source))


SIDECAR_FILENAMES = frozenset({
    "operating_point.json",
    "classifier_operating_point.json",
    "ordinal_operating_point.json",
    "regression_operating_point.json",
    "resolve_scale.json",
})
"""Every provenance stamp a prediction bucket carries beside its per-image documents.

A stamp is not a per-image label: a reader enumerating a bucket's prediction files excludes
these, or it invents an image stem no image has and reads a stamp as if it were detections.
Stated once here, so a stamp added for a new measurement dimension is excluded on every path
that enumerates a bucket, in this package and in any package that imports this set rather than
declaring its own copy.
"""


def is_sidecar_name(filename: str) -> bool:
    """Whether ``filename`` names one of a bucket's own provenance stamps.

    Case-insensitive: the filesystem this platform's stamps are written on may itself be
    case-insensitive, so a stamp write named in lowercase would replace a document named in
    another case, and comparing the stored case alone would then read the collided document as
    an ordinary label. The one predicate every reserved-name check shares, so ingest, band
    grouping and this enumeration cannot disagree about which stem is reserved.
    """
    return filename.lower() in SIDECAR_FILENAMES


def prediction_documents(bucket: str | Path) -> list[Path]:
    """Every per-image document in a prediction bucket, sorted, its own sidecar stamps excluded.

    The one enumeration every bucket walk uses, so a stamp added for a new measurement dimension
    is excluded everywhere at once rather than in each caller's own inline filter. A missing or
    non-directory ``bucket`` yields nothing.
    """
    d = Path(bucket)
    if not d.is_dir():
        return []
    return sorted(f for f in d.glob("*.json") if f.is_file() and not is_sidecar_name(f.name))


def _load(path: str) -> dict | None:
    """The parsed document at ``path``, or ``None`` when there is no file there.

    A present file that will not read raises :class:`UnreadableLabelDocument`: a bad document is
    not the same fact as no document, and reading it as absent would train an unreadable file as
    an empty negative.
    """
    if not os.path.exists(path):
        return None
    return load_label_document(path)


def safe_score(x) -> float:
    """A JSON-safe confidence: finite, rounded; non-finite (NaN/inf) collapses to 0.0.

    The one quantization every writer of a score applies, on the way into a persisted prediction
    document and again here for any caller (a CSV export averaging scores) that needs the same
    persisted precision, never a re-spelled ``round(x, 4)`` that would diverge on a non-finite
    score this collapses to 0.0.
    """
    v = float(x)
    return round(v, 4) if math.isfinite(v) else 0.0


def _score_of(obj: dict) -> float | None:
    """A prediction score, or ``None`` when absent/non-numeric (ground truth has none)."""
    s = obj.get("score")
    return float(s) if isinstance(s, (int, float)) and not isinstance(s, bool) else None


def _coerce_bbox(bb) -> list[float] | None:
    """A 4-list of floats, or None (wrong shape / non-numeric)."""
    if not isinstance(bb, list) or len(bb) != 4:
        return None
    try:
        return [float(v) for v in bb]
    except (TypeError, ValueError):
        return None


def _rings_from_segmentation(seg) -> list[list[tuple[float, float]]] | None:
    """Every valid polygon ring of a ``segmentation`` → pixel points per ring, or None if none usable.

    Each entry needs an even coord count and >=3 points (6 coords); a ring that fails this is dropped
    individually, not the whole annotation, the same treatment an RLE dict or other non-ring entry gets.
    Returns None only when no ring survives (an all-degenerate or empty ``segmentation``).
    """
    if not isinstance(seg, list) or not seg:
        return None
    rings: list[list[tuple[float, float]]] = []
    for ring in seg:
        if not isinstance(ring, list) or len(ring) < 6 or len(ring) % 2 != 0:
            continue
        try:
            rings.append([(float(ring[i]), float(ring[i + 1])) for i in range(0, len(ring), 2)])
        except (TypeError, ValueError):
            continue
    return rings or None


def _coerce_point(pt) -> tuple[float, float] | None:
    """A 2-list of floats as an (x, y) pair, or None (wrong shape / non-numeric)."""
    if not isinstance(pt, list) or len(pt) != 2:
        return None
    try:
        return (float(pt[0]), float(pt[1]))
    except (TypeError, ValueError):
        return None


def box_extent_ok(bbox: BBox) -> bool:
    """Whether ``bbox`` has real extent: ``x2 > x1`` and ``y2 > y1``.

    For a caller that wants to drop or count a degenerate box (a detector's own output clipped
    to nothing at an image edge) rather than raise: :func:`check_box_extent` enforces the same
    rule for a request that must be refused rather than silently dropped. Checks the raw corners;
    :func:`stored_box_extent_ok` is the check against what a write would actually store.
    """
    return bbox.x2 > bbox.x1 and bbox.y2 > bbox.y1


def stored_box_extent_ok(bbox: BBox) -> bool:
    """Whether ``bbox`` still has positive extent once rounded to the stored 2-decimal quantum.

    A box can pass :func:`box_extent_ok` on its raw corners and still round to zero width or
    height at the grid :func:`xywh` stores (a sub-centipixel detection). A caller deciding
    whether to keep or drop a detection before it ever reaches :func:`write_annotations` checks
    this, not the raw check, since the raw check alone would let such a box through to a writer
    that refuses it.
    """
    _, _, w, h = xywh(bbox.x1, bbox.y1, bbox.x2, bbox.y2)
    return w > 0 and h > 0


def check_box_extent(bbox: BBox, *, where: str) -> None:
    """Refuse a box with no real extent: ``x2 <= x1`` or ``y2 <= y1``.

    ``where`` names what is being written (a subject, a reviewer action, or a record index), so
    the refusal tells a caller which box failed rather than surfacing as an opaque write.
    """
    if not box_extent_ok(bbox):
        raise ValueError(
            f"{where}: box (x1={bbox.x1}, y1={bbox.y1}, x2={bbox.x2}, y2={bbox.y2}) has no "
            "positive extent; a box needs x2 > x1 and y2 > y1"
        )


def bbox_from_corners(x1: float, y1: float, x2: float, y2: float, *, where: str) -> BBox:
    """A :class:`BBox` from its corners, refusing an inverted or zero-extent one up front.

    The one place a raw request builds a box, so its refusal names ``where`` (the subject or
    action the request was about) rather than surfacing later at the writer's own record index.
    """
    box = BBox(x1, y1, x2, y2)
    check_box_extent(box, where=where)
    return box


def ring_vertex(vertex) -> tuple[float, float]:
    """A polygon ring vertex as ``(x, y)``, from an ``[x, y]`` pair or an ``{"x":, "y":}`` mapping.

    The producers of ring data disagree on the vertex shape (a canvas round-trip sends pairs, a
    segmentation prompt sends mappings), so the one conversion door takes either. Public: every
    door that stages a multi-ring polygon proposal (not just the ground-truth save path here)
    parses its vertices through this one function, never a second unpacking of its own.
    """
    if isinstance(vertex, Mapping):
        return float(vertex["x"]), float(vertex["y"])
    return float(vertex[0]), float(vertex[1])


def annotation_from_payload(payload: Mapping, *, author: str | None, now: str) -> Annotation:
    """One client payload dict as an :class:`Annotation`: the single conversion every save door uses.

    Geometry precedence is ``rings``, then ``points``, then ``bbox``, then ``point``. A payload
    carrying more than one of them never loses the richer shape: a polygon is the source of truth
    and its box is derived on write, so letting a box win would collapse it to a box-only record.
    An empty ``points`` list falls through to ``bbox`` rather than becoming a degenerate polygon.
    ``point`` is a single placed prompt or keypoint, deliberately a different key from ``points``:
    a one-vertex contour and a point are not the same geometry.

    Provenance: a payload carrying ``created_by`` is a shape round-tripping back through the
    client, so it keeps its own ``created_at`` and its review sign-off
    (``accepted_by``/``accepted_at``) verbatim, and the creator stays the creator through edits.
    One that does not is new: it is stamped to ``author`` at ``now`` and claims no sign-off, since
    a new shape minting acceptance would record a review that never happened. With no ``author``
    resolved either, a new shape carries no provenance rather than a time with nobody attached.
    """
    geometry: BBox | Polygon | Point | None = None
    if payload.get("rings"):
        geometry = Polygon(rings=[[ring_vertex(v) for v in ring] for ring in payload["rings"]])
    elif payload.get("points"):
        geometry = Polygon(rings=[[ring_vertex(v) for v in payload["points"]]])
    elif payload.get("bbox") is not None:
        x1, y1, x2, y2 = (float(v) for v in payload["bbox"])
        geometry = bbox_from_corners(x1, y1, x2, y2, where=f"subject {payload.get('subject')!r}")
    elif payload.get("point") is not None:
        geometry = Point(float(payload["point"][0]), float(payload["point"][1]))
    round_tripped = bool(payload.get("created_by"))
    created_by = payload.get("created_by") or author
    return Annotation(
        subject=str(payload["subject"]),
        geometry=geometry,
        attributes={str(k): str(v) for k, v in (payload.get("attributes") or {}).items()},
        created_by=created_by,
        created_at=payload.get("created_at") if round_tripped else (now if created_by else None),
        accepted_by=payload.get("accepted_by") if round_tripped else None,
        accepted_at=payload.get("accepted_at") if round_tripped else None,
    )


def _attributes_of(obj: dict) -> dict[str, str]:
    """The annotation's attribute values (name → value name); non-string entries dropped."""
    raw = obj.get("attributes")
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, str) and v}


def _prov_kwargs(obj: dict) -> dict:
    return {k: obj[k] for k in _PROV_KEYS if obj.get(k) is not None}


def _annotations_of(data: dict | None) -> list[Annotation]:
    """Parse a loaded per-image dict into :class:`Annotation` records (the one shared parse).

    Prefers a polygon geometry over a box when both are present (the polygon is the source of truth and
    its box is derivable). An annotation with a ``subject`` but no geometry is kept (an image-level
    label).

    Raises :class:`UnreadableLabelDocument` for a document whose ``annotations`` is present but not
    a list (covers it being absent or ``null`` too), for a record that is not a dict or carries no
    string ``subject`` (naming the record's index: skipping it silently would let a one-record file
    whose only record is malformed read as an empty document, and an empty read is what every
    zero-object training path is built on), and for a record whose stored box has no positive
    extent (:func:`check_box_extent`).
    """
    if data is None:
        return []
    raw = data.get(ANNOTATIONS_KEY)
    if not isinstance(raw, list):
        raise UnreadableLabelDocument(
            f"{ANNOTATIONS_KEY!r} is {raw!r}, not the list a label document holds"
        )
    out: list[Annotation] = []
    for i, o in enumerate(raw):
        if not isinstance(o, dict):
            raise UnreadableLabelDocument(f"record {i} is {o!r}, not an annotation object")
        subject = o.get("subject")
        if not isinstance(subject, str) or not subject:
            raise UnreadableLabelDocument(f"record {i} carries no string subject: {o!r}")
        geometry: BBox | Polygon | Point | None = None
        rings = _rings_from_segmentation(o.get("segmentation"))
        if rings is not None:
            geometry = Polygon(rings)
        else:
            bb = _coerce_bbox(o.get("bbox"))
            if bb is not None:
                x, y, w, h = bb
                geometry = BBox(x, y, x + w, y + h)
                try:
                    check_box_extent(geometry, where=f"record {i} ({subject!r})")
                except ValueError as exc:
                    raise UnreadableLabelDocument(str(exc)) from exc
            else:
                pt = _coerce_point(o.get("point"))
                if pt is not None:
                    geometry = Point(pt[0], pt[1])
        out.append(Annotation(
            subject=subject, geometry=geometry, attributes=_attributes_of(o),
            score=_score_of(o), **_prov_kwargs(o),
        ))
    return out


def annotations_of_document(document: dict) -> list[Annotation]:
    """The typed annotation records a parsed per-image document holds.

    The public counterpart of :func:`_annotations_of`, for a caller that has already parsed one
    document (through :func:`parse_label_document` or :func:`annotations_from_bytes`'s own
    decode) and wants its typed annotations without discarding the document's own ``image``,
    ``width`` and ``height`` fields the way :func:`read_annotations` does: reading both from the
    one parsed dict is one byte snapshot, never two independent reads of a file that could change
    in between. Raises :class:`UnreadableLabelDocument` for the same shapes ``_annotations_of``
    refuses (see its own docstring).
    """
    return _annotations_of(document)


# ── reader ─────────────────────────────────────────────────────────────────
# A missing file reads as []; a present, unreadable document raises UnreadableLabelDocument.


def read_annotations(path) -> list[Annotation]:
    return _annotations_of(_load(str(path)))


def detection_annotations(path: str | Path) -> list[Annotation]:
    """A prediction document's annotations narrowed to real detections, a ``Point`` excluded.

    A ``Point`` records no detection (nothing a detector localized as a box or region), so
    counting one would inflate whatever this feeds: a phenology curve's denominator, or a
    per-image detection count. The one predicate ``count_by_class`` and a per-image count CSV's
    bucket-regime reader share, extracted here so neither path can drift from the other about
    what counts as a detection.
    """
    return [a for a in read_annotations(str(path))
           if a.geometry is not None and not isinstance(a.geometry, Point)]


def read_annotations_versioned(target: Key | str | Path) -> tuple[list[Annotation], Version]:
    """An image's annotations and the version of the document they came from, read together.

    What a load-edit-save client needs: the token names exactly the bytes the client was shown,
    so a document that changed in between cannot pass the comparison on the way back in. An
    absent document reads as no annotations at ``Version.ABSENT``, which is the token that says
    "create this, or refuse", told apart from a present document by the store's own version, never
    by the bytes (a present document can be zero bytes long and is still present). A present
    document that will not decode or will not parse raises :class:`UnreadableLabelDocument`, the
    same reader contract :func:`read_annotations` keeps, decoded through the same strict policy.
    """
    stored = tcip_store.read_blob_versioned(_record_key(target), default=b"")
    if stored.version == Version.ABSENT:
        return [], stored.version
    return annotations_from_bytes(stored.value, source=str(target)), stored.version


# ── reference admissibility ────────────────────────────────────────────────

PERSON_IDENTITY_PREFIX = "user:"  # spelled here, not imported: this package depends only on tcip-store


def is_unadjudicated_prediction(a: Annotation) -> bool:
    """True when ``a`` is a model's own output that no human has taken responsibility for.

    A set ``score`` is this format's own marker that the record is a prediction (see
    :class:`~tcip_annotation.state.Annotation`), and a scored record is by construction one no
    reviewer has ruled on: accepting a prediction into ground truth drops its ``score`` and stamps
    ``accepted_by``, so an accepted record no longer answers True here.
    """
    return a.score is not None


def is_unadjudicated_agent_authorship(a: Annotation) -> bool:
    """True when ``a`` names a producer that is not a person and no reviewer has signed off on it.

    A person's recorded identity opens with :data:`PERSON_IDENTITY_PREFIX`, while a tool producer
    stays bare (``sam``, an agent's own name) or carries a ``model:<checkpoint>`` stamp, so an
    authorship without that prefix is something other than a person's statement. An annotation
    carrying no ``created_by`` at all is not claimed by this rule: it is unattributed, not
    attributed to a machine, and the pre-provenance hand labels a project already holds read that
    way. A set ``accepted_by`` is a reviewer taking responsibility for the record, which is what
    turns machine-origin ground truth back into something a person stands behind.
    """
    if not a.created_by or a.created_by.startswith(PERSON_IDENTITY_PREFIX):
        return False
    return not a.accepted_by


def authorship_of(a: Annotation) -> str:
    """``a``'s authorship, one of ``"person"``, ``"tool"``, ``"tool_accepted"`` or ``"unattributed"``,
    for a caller that draws or names it (a canvas symbology, a load response) rather than only
    gating on it.

    Built on :func:`is_unadjudicated_agent_authorship` for the ``tool`` classification, so a shape
    this reads as ``"tool"`` and the reference/canopy admissibility rules read as agent-authored
    are the same shape by construction, never two independent readings of the same ``created_by``.
    """
    if not a.created_by:
        return "unattributed"
    if is_unadjudicated_agent_authorship(a):
        return "tool"
    if a.created_by.startswith(PERSON_IDENTITY_PREFIX):
        return "person"
    return "tool_accepted"


@dataclass
class ProvenanceFacts:
    """The provenance classification over one list of annotations, computed once so
    :func:`require_reference_ground_truth` (the reference rule, which admits a record with no
    ``created_by`` at all as a pre-provenance hand label) and
    ``segment_attribution.load_canopy_segments`` (the canopy rule, stricter: it admits none of
    those) apply their own different admissibility policy to one shared classification, rather
    than each re-deriving what a record's provenance says.

    ``scored`` and ``machine_authored`` are exactly what :func:`require_reference_ground_truth`
    already computed inline before this extraction (the same :func:`is_unadjudicated_prediction`/
    :func:`is_unadjudicated_agent_authorship` predicates, so its own refusal wording holds
    unchanged); ``no_created_by``/``not_positively_a_persons`` are index lists into the annotations
    passed in, for a caller (the canopy rule) that names the specific record its own refusal is
    about.
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
    """The :class:`ProvenanceFacts` classification over ``annotations``, the one computation both
    admissibility rules apply their own policy to (see that class's own docstring)."""
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
        elif not a.created_by.startswith(PERSON_IDENTITY_PREFIX):
            if not (a.accepted_by and a.accepted_by.startswith(PERSON_IDENTITY_PREFIX)):
                not_positively_a_persons.append(i)
    return ProvenanceFacts(
        total=len(annotations), scored=scored, machine_authored=machine_authored,
        no_created_by=no_created_by, not_positively_a_persons=not_positively_a_persons,
    )


def require_reference_ground_truth(directory: str | Path) -> None:
    """Refuse ``directory`` as a measurement reference when only the model stands behind it.

    The rule every reference read shares: a calibration or held-out reference is only a measurement
    if something other than the model produced it. Two shapes fail it, both provenance rather than
    a number. A record carrying a prediction ``score`` is the model's own output. A record an agent
    authored as ground truth, with no reviewer's ``accepted_by``, is the same output with the score
    dropped. Either clears every numeric gate a calibration applies, since the model agrees with
    itself, so the gates cannot catch it and the provenance has to.

    Refuses on the whole directory, never by dropping the offending records: a mixed directory
    silently narrowed to its admissible subset would validate against a reference nobody chose.
    An absent or empty directory raises nothing here; a caller that needs a reference to exist
    checks that itself. Walks the directory through :func:`prediction_documents`, so a reference
    that happens to be a prediction bucket is read by its per-image documents alone, never one of
    its own sidecar stamps (a dict with no ``annotations`` list, which the reader would otherwise
    refuse as unreadable).
    """
    directory = Path(directory)
    annotations: list[Annotation] = []
    for path in prediction_documents(directory):
        annotations.extend(read_annotations(path))
    facts = provenance_facts(annotations)
    scored, total, agent_authored = facts.scored, facts.total, facts.machine_authored
    if scored:
        raise ValueError(
            f"{scored} of {total} annotations in {directory} carry a prediction score, so they are "
            "the model's own output that no human has ruled on, and a reference built from them "
            "measures the model against itself rather than against a measurement. Annotate this "
            "reference, accept the model's proposals through review so each record is a reviewer's "
            "call and carries their accepted_by, or validate against a breeder-confirmed sample of "
            "the model's outputs instead."
        )
    if agent_authored:
        producers = ", ".join(sorted(set(agent_authored)))
        raise ValueError(
            f"{len(agent_authored)} of {total} annotations in {directory} are authored by "
            f"{producers}, which names no person under this platform's {PERSON_IDENTITY_PREFIX}"
            "<name> convention, and carry no accepted_by. Ground truth an agent wrote and no "
            "human has adjudicated is not a calibration or holdout reference. The lighter path "
            "is the review-confirmation loop: have a reviewer confirm these records so each one "
            "carries their accepted_by, rather than hand-annotating the whole reference."
        )


# ── writer ─────────────────────────────────────────────────────────────────


def xywh(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    """A pixel box in corner form as this schema's ``bbox``: COCO ``[x, y, w, h]``, 2 decimals.

    The 2-decimal quantum is the grid the stored document lives on, so it is also the grid anything
    comparing a stored box against another box has to be on. Public because that comparison happens
    outside this package too (evaluation puts both sides of a match on this grid before scoring);
    one implementation, so a reader and a scorer cannot land on different grids.

    The inverse, reading such a record back, is ``BBox(x, y, x + w, y + h)`` in
    :func:`_annotations_of`, which is what keeps write and read symmetric.
    """
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def _stored_bbox_or_raise(bbox: BBox, *, where: str) -> list[float]:
    """A box's stored ``xywh``, refusing one whose rounded extent is not positive.

    Checked against the rounded, stored box (:func:`stored_box_extent_ok`), not the raw corners:
    a box, or a polygon's derived box, can pass on its raw extent and still round to zero width
    or height at the document's stored 2-decimal grid, and a document the writer lets through
    must be one the reader accepts. Shared by every geometry a writer stores as a ``bbox``, so
    the box and polygon branches cannot drift onto two different checks.
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
    """Whether ``geometry`` still has positive extent once written to its stored grid.

    A :class:`Polygon`'s vertices round to two decimals before its box is derived, the same
    order :func:`write_annotations` stores them in, so a caller pre-filtering a candidate
    detection before it ever reaches the writer reaches the identical verdict the writer would.
    An empty ring list (every ring shorter than three points) has no shape and is not ok.
    """
    if isinstance(geometry, Polygon):
        rings = [r for r in geometry.rings if len(r) >= 3]
        if not rings:
            return False
        return stored_box_extent_ok(bbox_of(Polygon(_rounded_rings(rings))))
    return stored_box_extent_ok(geometry)


def _annotation_record(a: Annotation) -> dict | None:
    """One annotation → its JSON object, or None if a degenerate polygon should be skipped."""
    rec: dict = {"subject": a.subject}
    geom = a.geometry
    if isinstance(geom, Polygon):
        valid_rings = [r for r in geom.rings if len(r) >= 3]
        if not valid_rings:
            return None  # no ring is a real shape; skip so write<->read stays symmetric
        rounded_rings = _rounded_rings(valid_rings)
        rec["segmentation"] = [[c for xy in ring for c in xy] for ring in rounded_rings]
        # Boxed from the rounded rings the document stores, not the raw ones, so a ring that
        # only collapses at the stored grid can't write a box claiming extent it lost.
        poly_box = bbox_of(Polygon(rounded_rings))
        rec["bbox"] = _stored_bbox_or_raise(poly_box, where=f"{a.subject!r} annotation's polygon")
    elif isinstance(geom, BBox):
        rec["bbox"] = _stored_bbox_or_raise(geom, where=f"{a.subject!r} annotation")
    elif isinstance(geom, Point):
        rec["point"] = [round(geom.x, 2), round(geom.y, 2)]
    if a.attributes:
        rec["attributes"] = dict(a.attributes)
    if a.score is not None:
        rec["score"] = safe_score(a.score)
    for k in _PROV_KEYS:
        v = getattr(a, k, None)
        if v is not None:
            rec[k] = v
    return rec


def write_annotations(target, annotations, img_w: int, img_h: int, *,
                      keep_empty: bool = False, expect: Version | None = None) -> Version | None:
    """Write all of an image's annotations to its per-image JSON document.

    ``target`` is either the document's storage key, minted by whichever resolver owns the tree it
    lives in, or its path, which is placed generically by :func:`annotation_record_key` here so this
    package never has to learn a layout.

    Empty list: ``keep_empty`` writes ``{"annotations": []}`` (unannotated until a human confirms
    it), else removes the document; writing an empty document never manufactures a negative.

    ``expect`` is the version the caller read (:func:`tcip_store.read_blob_versioned`), turning the
    write into a compare-and-set: anything that changed underneath raises ``VersionConflict`` and
    nothing is written. Returns the new version, or ``None`` when the document was removed.
    """
    records = [r for r in (_annotation_record(a) for a in annotations) if r is not None]
    key = _record_key(target)
    if not records and not keep_empty:
        tcip_store.delete(key, expect=expect)
        return None
    payload = {"image": key.parts[-1], "width": int(img_w), "height": int(img_h),
               ANNOTATIONS_KEY: records}
    return tcip_store.put_blob(key, _document_bytes(payload), expect=expect)


# ── the one target-membership decision (shared by assembly and the loader) ───


UNLABELED = "unlabeled"  # a real target this scope covers, but not yet assessed for `attribute`


def target_class_id(a: Annotation, subject: str, attribute: str | None,
                    id_map: dict[str, int], *, allow_unlabeled: bool = False
                    ) -> int | None | str:
    """The 0-indexed class id ``a`` trains as for ``(subject, attribute)``.

    Returns ``None`` if ``a`` is not a detection/segmentation target for this scope at all (a
    different subject, a geometry-less label, or a :class:`~tcip_annotation.state.Point`, since a point
    has no box/area and is never a detection/segmentation training target). For a genuine
    target, two different failure shapes exist and must not be conflated: the instance was
    never assessed for ``attribute`` at all (``a.attributes.get(attribute) is None``, the annotator
    hasn't gotten to it yet, a soft/expected gap), versus the instance was assessed but with a value
    the registry cannot decode (a real decode bug, the registry and the labels disagree). The first
    case returns the distinguishable sentinel ``UNLABELED`` when ``allow_unlabeled=True`` (opt-in,
    default ``False`` preserves this function's original all-undecodable-cases-raise behavior for any
    caller that hasn't been updated to handle the three-way split); the second always raises,
    regardless of ``allow_unlabeled``, since a real annotation read as nothing is a measurement bug,
    never something to drop silently.

    The single membership+id decision: :func:`to_coco_dataset` (training assembly) and the loader's
    per-image target reader both call this, so the assembled COCO and the calibration/eval GT can
    never disagree about which annotation is a target or which class it is.
    """
    if a.subject != subject or a.geometry is None or isinstance(a.geometry, Point):
        return None
    key = a.attributes.get(attribute) if attribute else subject
    if key is None:
        if allow_unlabeled:
            return UNLABELED
        raise ValueError(
            f"annotation of subject {subject!r} has no value for attribute {attribute!r}: "
            "the registry cannot decode its own labels")
    if key not in id_map:
        raise ValueError(
            f"annotation of subject {subject!r} has class key {key!r} not in the run's id map "
            f"(known: {sorted(id_map)}): the registry cannot decode its own labels")
    return id_map[key]


class ClassifiedRecordRefused(ValueError):
    """A record under a classified scope carries no usable value: not the object class, no value
    under the attribute, or a value outside the bucket's own vocabulary.

    Named for one prediction document and record index (``source``); the remedy is
    ``scripts/conform_classified_predictions.py`` (a ``tcip-mcp`` script) run over the bucket.
    """


def classified_value_of(a: Annotation, *, subject: str, attribute: str) -> str | None:
    """The value a record carries under ``(subject, attribute)``, or ``None`` when it carries
    none (a record of a different subject, or one with nothing recorded under ``attribute``).

    The one read every classified-scope reader projects a record through: a prediction's decoded
    value sits under ``attributes[attribute]`` with ``subject`` the object class (the shape
    :func:`~tcip_mcp.pipelines.postprocessing.export.write_predictions_json` writes under an
    attribute), the same shape ground truth carries.
    """
    if a.subject != subject:
        return None
    return a.attributes.get(attribute)


def require_classified_record(
    a: Annotation, *, subject: str, attribute: str, vocabulary, source: str,
) -> str:
    """The value ``a`` carries under ``(subject, attribute)``, held to ``vocabulary`` (the keys of
    the bucket's own recorded ``id_map``, the vocabulary its producer decoded through).

    Raises :class:`ClassifiedRecordRefused`, naming ``source`` (the document and record index),
    when ``a.subject`` is not ``subject``, the record carries no value under ``attribute``, or the
    value is not a member of ``vocabulary``: a classified bucket holds one subject and every
    record its writer produced carries a value the map declares, so a record that does not is a
    pre-conform record or a foreign document, never a legitimate gap.
    """
    if a.subject != subject:
        raise ClassifiedRecordRefused(
            f"{source}: record's subject is {a.subject!r}, not {subject!r}, the object class "
            "this classified bucket's every record is of. Run "
            "scripts/conform_classified_predictions.py over this bucket."
        )
    value = a.attributes.get(attribute)
    if value is None:
        raise ClassifiedRecordRefused(
            f"{source}: record of {subject!r} carries no value under attribute {attribute!r}. "
            "Run scripts/conform_classified_predictions.py over this bucket."
        )
    if value not in vocabulary:
        raise ClassifiedRecordRefused(
            f"{source}: record's value {value!r} is not a member of this bucket's own vocabulary "
            f"({sorted(vocabulary)}). Run scripts/conform_classified_predictions.py over this "
            "bucket."
        )
    return value


# ── dataset-COCO assembly (for training / export) ────────────────────────────


def to_coco_dataset(
    entries: list[tuple[str, str]],
    *,
    subject: str,
    id_map: dict[str, int],
    attribute: str | None = None,
    confirmed_negative_names: set[str] | None = None,
) -> dict:
    """Concatenate per-image JSON files into one COCO dataset dict, scoped to one subject.

    ``entries``: ``[(label_json_path, image_file_name), ...]``. ``id_map`` is the run's name→id
    assignment (``tcip_mcp.class_registry.assign_class_ids``): keyed by the ``attribute``'s value names
    when ``attribute`` is set, else by the ``subject`` itself. Each kept annotation's ``category_id`` is
    ``id_map[key]`` where ``key`` is its attribute value (``attribute`` set) or the subject.

    Only annotations of ``subject`` that carry a geometry become COCO annotations (a geometry-less
    annotation has no training destination this slice, but its presence still marks the image
    *annotated*, keeping it out of the negative bucket). A missing label file is skipped; an image
    with no annotations of ``subject`` is included only as a negative when its ``file_name`` is in
    ``confirmed_negative_names`` (a human marked it Complete-with-nothing). ``categories`` are emitted
    from ``id_map``. An annotation whose class key is not in ``id_map`` raises, since a real annotation
    the registry cannot decode is a measurement bug, not something to drop silently.

    When ``attribute`` is set, an image with any instance never assessed for it is excluded
    wholesale, not trained on its labeled subset alone: silently narrowing to the labeled instances
    would leave the image's other real, unlabeled objects to train as background noise. The excluded
    ``file_name``s are reported in the returned dict's own ``excluded_incomplete_attribute`` list,
    alongside ``images``/``annotations``/``categories``, so a downstream partition
    (``trainable_stems``) can attribute the drop to its real reason rather than re-deriving one from
    the image's mere absence, which reads identically to an empty label file nobody confirmed.

    A missing label file is skipped, unannotated; a present, unreadable one raises
    :class:`UnreadableLabelDocument` rather than assembling as a zero-object image, so a corrupt
    document behind a confirmed-negative name never enters the assembled dataset as a fabricated
    negative.
    """
    categories = [{"id": cid, "name": name} for name, cid in sorted(id_map.items(), key=lambda kv: kv[1])]
    coco: dict = {"images": [], "annotations": [], "categories": categories,
                 "excluded_incomplete_attribute": []}
    negatives = confirmed_negative_names or set()
    ann_id = 1
    img_id = 0
    for label_path, file_name in entries:
        if not os.path.exists(str(label_path)):
            continue  # unannotated, not part of the training set
        data = load_label_document(str(label_path))
        scoped = [a for a in _annotations_of(data) if a.subject == subject]
        if not scoped and file_name not in negatives:
            continue  # no annotations of this subject and not a confirmed negative, skip
        # allow_unlabeled=True: an instance never assessed for `attribute` is a soft, expected gap,
        # not a decode bug, and must not abort the whole assembly; computed once per instance and reused below.
        cids = [target_class_id(a, subject, attribute, id_map, allow_unlabeled=True) for a in scoped]
        if attribute is not None and UNLABELED in cids:
            coco["excluded_incomplete_attribute"].append(file_name)
            continue  # incomplete GT for this scope: the whole image, not just the gap, is excluded
        img_id += 1
        coco["images"].append({
            "id": img_id, "file_name": file_name,
            "width": int(data.get("width", 0) or 0), "height": int(data.get("height", 0) or 0),
        })
        for a, cid in zip(scoped, cids):
            if cid is None or a.geometry is None:
                continue  # image-level label: counts the image as annotated, no detection/seg target
            assert not isinstance(a.geometry, Point), (
                "target_class_id already returned None for a Point geometry"
            )
            box = bbox_of(a.geometry)
            rec: dict = {
                "id": ann_id, "image_id": img_id, "category_id": cid, "iscrowd": 0,
                "bbox": xywh(box.x1, box.y1, box.x2, box.y2),
                "area": round((box.x2 - box.x1) * (box.y2 - box.y1), 2),
            }
            if isinstance(a.geometry, Polygon):
                rec["segmentation"] = [[round(float(c), 2) for xy in ring for c in xy]
                                       for ring in a.geometry.rings if len(ring) >= 3]
            if a.score is not None:
                rec["score"] = safe_score(a.score)
            for k in _PROV_KEYS:
                v = getattr(a, k, None)
                if v is not None:
                    rec[k] = v
            coco["annotations"].append(rec)
            ann_id += 1
    return coco
