"""Per-image JSON: the canonical on-disk label format (ground truth + predictions).

One JSON file per image, holding every subject's annotations by name. Each annotation carries its
``subject``, an optional geometry (``bbox`` xywh, ``segmentation`` polygon, ``point`` [x,y], or none
for an image/plant-level label), its attribute values by name, an optional ``score`` (predictions),
and provenance (``created_by/at``, ``accepted_by/at``, ``accepted_by_rule``), so a prediction's
origin travels with it into ground truth on accept, with no sidecar.

Schema::

    { "image": "<stem>", "width": W, "height": H,
      "annotations": [
        { "subject": "<subject>",
          "bbox": [x, y, w, h],                 # COCO xywh, pixel      (optional)
          "segmentation": [[x1,y1, ...], ...],  # pixel polygon, one or more rings (optional)
          "point": [x, y],                      # pixel point, a prompt or keypoint (optional)
          "attributes": {"<attribute>": "<value>"},   # attr name -> value name
          "score": 0.91,                        # predictions only
          "iscrowd": true,                      # a region of unseparated objects (optional)
          "created_by": "sam", "created_at": "...",
          "accepted_by": "user:breeder", "accepted_at": "...",
          "accepted_by_rule": "<experiment_id>:<record_digest>" } ] }

Integer class ids never appear on disk; a name→id assignment is a per-training-run artifact
(:mod:`tcip_mcp.subject_registry`).

Negative invariant: a missing file is unannotated, and a present file with ``"annotations": []`` is
*still* unannotated until a human marks that image Complete, recorded as ``"negative"`` in
``.tcip/state/image_status.json``, scoped to the subject. Only that confirmation makes it a training
negative. An annotation with a subject but no
geometry (an image-level label) is a real annotation: it keeps the image out of the empty-negative
bucket and never collapses to nothing.

A missing file reads as unannotated (``[]``), and so does the platform's own empty document
(``{"annotations": []}``). A *present* document this format cannot make sense of (undecodable text,
a non-dict document, an ``annotations`` that is not a list, or a record
:func:`annotation_of_record` refuses, a supplied value that does not parse among them) raises
:class:`UnreadableLabelDocument` instead of reading as empty: an unreadable file is not the same
fact as no file, and reading one as empty would train a corrupt document as a confirmed negative.
Writers and readers are symmetric: a record the writer stores is one the reader accepts.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

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

from tcip_annotation.state import (
    Annotation, BBox, Point, Polygon, bbox_of, is_detection, polygonal,
    prediction_score,
)

ANNOTATIONS_KEY = "annotations"  # the one top-level list key
LABEL_SUFFIX = ".json"
"""The suffix of every per-image label and prediction document, stated once for every package."""
_PROV_KEYS = ("created_by", "created_at", "accepted_by", "accepted_at", "accepted_by_rule")


# ── the store (tcip-annotation must not depend on tcip-mcp) ───────────────────

ANNOTATION_RECORDS_STORE = "annotation_records"
_ANNOTATION_RECORD_LOCATOR = RootedFileLocator(suffix=LABEL_SUFFIX)
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

    The generic form, for a tree no layout resolver describes (a curated dataset's ``labels/``,
    an export bucket) and for a caller using this package on its own. A caller that
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

    Deliberately not a :class:`ValueError`: a handler written to catch a caller's own
    ``ValueError`` must not also absorb a document nobody can read and silently render or train
    it as unlabeled.
    """


def parse_json_document(text: str, *, source: str) -> dict:
    """A JSON document's parsed dict, from its raw text: decode and dict-shape only.

    Raises :class:`UnreadableLabelDocument`, naming ``source``, for text that does not decode as
    JSON or that decodes to something other than a dict. Carries no opinion on the document's
    shape or ``schema_version``: the COCO import reads an external document through here, and
    :func:`parse_label_document` adds the per-image document's own checks.
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


def is_dataset_level_document(data: dict) -> bool:
    """Whether a parsed document carries a dataset-level COCO's keys (``images`` or
    ``categories``) rather than being one image's label document."""
    return "images" in data or "categories" in data


def parse_label_document(text: str, *, source: str) -> dict:
    """A per-image label document's parsed dict, from its raw text.

    Raises :class:`UnreadableLabelDocument`, naming ``source``, for text that does not decode as
    JSON, that decodes to something other than a dict (:func:`parse_json_document`), that carries
    a dataset-level COCO's keys or the old ``objects`` schema, or that carries a
    ``schema_version`` this reader does not accept (:func:`check_annotation_record_version`):
    none is a document this schema can read, and reading any of them as a label would train an
    unreadable file, or another image's records, as this image's. The one choke point every
    reader of a per-image document shares (``read_annotations``, ``load_label_document``,
    ``annotations_from_bytes``), so every reader, training and calibration included, gets the
    same answer about what a document is.
    """
    data = parse_json_document(text, source=source)
    if is_dataset_level_document(data):
        raise UnreadableLabelDocument(
            f"{source} is a dataset-level COCO document (an 'images' or 'categories' key), not a "
            "per-image label document: convert it with import_coco"
        )
    if "objects" in data:
        raise UnreadableLabelDocument(
            f"{source} is the old 'objects' label schema, which is not read in place: convert it "
            "to the name-based per-image schema"
        )
    check_annotation_record_version(data, source=source)
    return data


def decode_document_bytes(data: bytes, *, source: str) -> str:
    """A JSON document's bytes, decoded strictly as UTF-8, a leading byte-order mark accepted.

    ``utf-8-sig``: a UTF-8 byte-order mark encodes the same text as the same document without
    one, so a document this platform (or an operator's conforming script) wrote under a tool
    that stamps one must read identically to one that does not. Bytes that decode with a BOM
    stripped but are not valid UTF-8 otherwise, or that carry no valid UTF-8 at all (a UTF-16
    document, for instance), still raise :class:`UnreadableLabelDocument`, naming ``source``: the
    one decode policy every reader of a document's raw bytes shares, so a file reader and a
    store-backed reader over the same bytes cannot disagree about whether they read.
    """
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UnreadableLabelDocument(f"{source} is not valid UTF-8: {exc}") from exc


def read_document_bytes(path: str | Path) -> bytes:
    """A present document's bytes, the one file read every document loader shares.

    Raises :class:`UnreadableLabelDocument`, naming ``path``, when the file cannot be opened (a
    permission error, a directory where a file was expected). Callers check for a missing path
    themselves: this is only ever called once a document is known to be present.
    """
    p = Path(path)
    try:
        return p.read_bytes()
    except OSError as exc:
        raise UnreadableLabelDocument(f"{p} could not be opened: {exc}") from exc


def load_label_document(path: str | Path) -> dict:
    """A per-image label document's parsed dict, read from ``path``.

    Raises :class:`UnreadableLabelDocument`, naming ``path``, for a file that will not open, bytes
    that are not valid UTF-8, or contents :func:`parse_label_document` refuses.
    """
    source = str(path)
    return parse_label_document(
        decode_document_bytes(read_document_bytes(path), source=source), source=source)


def annotations_from_bytes(data: bytes, *, source: str) -> list[Annotation]:
    """The typed annotation records a label document's raw bytes hold.

    The one path from bytes to records: the shared decode, :func:`parse_label_document`, then
    the record construction :func:`read_annotations` performs, so a caller holding a document's
    bytes (a store-backed reader, a parse memo keyed on the bytes it read) reads exactly what the
    file reader would. Raises :class:`UnreadableLabelDocument`, naming ``source``, for anything
    the file reader would refuse.
    """
    return _annotations_of(parse_label_document(decode_document_bytes(data, source=source),
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
    return sorted(f for f in d.glob(f"*{LABEL_SUFFIX}")
                  if f.is_file() and not is_sidecar_name(f.name))


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


def _numbers(value, key: str, count: int | None = None) -> list[float]:
    """``value`` as floats: a list of JSON numbers, exactly ``count`` of them when given.

    Raises ``ValueError`` naming ``key`` for anything else, a numeric string included: a supplied
    value that is not what the schema states is refused, never read as absent.
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
    """Supplied attribute values, name to value name, read by the decoder and the writer alike:
    anything but a mapping of names to non-empty value names raises ``ValueError``, since a value
    that is not a name is not the same fact as no value."""
    if not isinstance(raw, dict) or not all(isinstance(v, str) and v for v in raw.values()):
        raise ValueError(f"attributes {raw!r} are not attribute names mapped to value names")
    return dict(raw)


def iscrowd_of(raw) -> bool:
    """A supplied crowd flag: COCO's ``0``/``1`` or ``true``/``false``, else ``ValueError``."""
    if not isinstance(raw, int) or raw not in (0, 1):
        raise ValueError(f"iscrowd {raw!r} is not 0, 1, true or false")
    return bool(raw)


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


def ring_vertex(vertex) -> tuple:
    """A polygon ring vertex's ``(x, y)`` as given, from an ``[x, y]`` pair or an
    ``{"x":, "y":}`` mapping, or ``ValueError`` for any other shape.

    The producers of ring data disagree on the vertex shape (a canvas round-trip sends pairs, a
    segmentation prompt sends mappings), so the one unpacking takes either, and every door taking
    ring vertices (the save conversion here, proposal staging) goes through it. The values are
    not interpreted here: the reader of the ring checks they are numbers.
    """
    if isinstance(vertex, Mapping) and "x" in vertex and "y" in vertex:
        return vertex["x"], vertex["y"]
    if isinstance(vertex, (list, tuple)) and len(vertex) == 2:
        return vertex[0], vertex[1]
    raise ValueError(f"ring vertex {vertex!r} is not an [x, y] pair or an {{x, y}} mapping")


def annotation_from_payload(payload: Mapping, *, author: str | None, now: str) -> Annotation:
    """One client payload dict as an :class:`Annotation`: the single conversion every save door uses.

    The payload is translated into the document's own record shape and decoded by
    :func:`annotation_of_record`, so every check a stored record passes is the one a saved payload
    passes, and a supplied value that route refuses raises ``ValueError`` here too. The
    translation: ``bbox`` corners ``[x1, y1, x2, y2]`` become the record's ``[x, y, w, h]``;
    ``rings`` (a list of rings) and ``points`` (one ring) become ``segmentation``, ``rings``
    winning when both are given, each vertex an ``[x, y]`` pair or an ``{"x", "y"}`` mapping;
    ``point`` (a single placed prompt or keypoint, a different geometry from a one-vertex contour)
    is kept as it is. A polygon then wins over a box and a box over a point, as in a stored record.

    Provenance: a payload carrying ``created_by`` (present as the decoder reads presence: any
    value but ``null``) is a shape round-tripping back through the client, so it keeps every
    provenance key it carries verbatim, its review sign-off included, and the creator stays the
    creator through edits. One that does not is new: it is stamped to ``author`` at ``now`` and
    claims no sign-off, since a new shape minting acceptance would record a review that never
    happened. With no ``author`` resolved either, a new shape carries no provenance rather than a
    time with nobody attached. A payload's ``score`` is not read: a saved shape is ground truth.
    A payload box quantizes to the stored two-decimal grid (:func:`xywh`) in translation, so what
    is saved is what a later read of the document returns.
    """
    payload = annotation_object(payload)
    rings, corners = payload.get("rings"), payload.get("bbox")
    if rings is None and payload.get("points") is not None:
        rings = [payload["points"]]
    segmentation = rings if not isinstance(rings, list) else [
        [c for v in ring for c in ring_vertex(v)] if isinstance(ring, list) else ring
        for ring in rings]
    provenance = ({k: payload.get(k) for k in _PROV_KEYS} if payload.get("created_by") is not None
                  else {"created_by": author, "created_at": now if author else None})
    return annotation_of_record({
        "subject": payload.get("subject"),
        "segmentation": segmentation,
        "bbox": xywh(*_numbers(corners, "bbox", 4)) if corners is not None else None,
        "point": payload.get("point"),
        "attributes": payload.get("attributes"),
        "iscrowd": payload.get("iscrowd"),
        **provenance,
    })


def annotation_object(o) -> dict:
    """``o`` when it is an annotation object (a JSON object), else ``ValueError`` naming it: the
    one check every reader of a raw record, native or COCO, makes before reading its fields."""
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


def _annotations_of(data: dict | None) -> list[Annotation]:
    """Parse a loaded per-image dict into :class:`Annotation` records (the one shared parse).

    Raises :class:`UnreadableLabelDocument` for a document whose ``annotations`` is present but not
    a list (covers it being absent or ``null`` too), and for any record
    :func:`annotation_of_record` refuses, naming the record's index: skipping it silently would let
    a one-record file whose only record is malformed read as an empty document, and an empty read
    is what every zero-object training path is built on.
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
        try:
            out.append(annotation_of_record(o))
        except ValueError as exc:
            raise UnreadableLabelDocument(f"record {i} {exc}") from exc
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


def read_predictions(path) -> list[Annotation]:
    """A prediction document's records, each stating its ``score``, the one decoder a model's own
    output is read through: a record stating none raises :class:`UnreadableLabelDocument` naming
    it (:func:`~tcip_annotation.state.prediction_score`), never a stand-in confidence."""
    annotations = read_annotations(path)
    for i, a in enumerate(annotations):
        try:
            prediction_score(a)
        except ValueError as exc:
            raise UnreadableLabelDocument(f"record {i} {exc}") from exc
    return annotations


def detection_annotations(path: str | Path) -> list[Annotation]:
    """A prediction document's annotations narrowed to the detections a count counts
    (:func:`~tcip_annotation.state.is_detection`: a crowd region and a ``Point`` excluded).

    A ``Point`` records no detection and a crowd region is never one object, so counting either
    would inflate whatever this feeds: a phenology curve's denominator, a per-image detection
    count, a per-plant count. The one selection every counting reader of a per-image document
    shares (``count_by_class``, the per-image count CSV's bucket reader, the orthomosaic per-plant
    count, the worst-predictions triage on both sides), so no path can drift from another about
    what counts as a detection.
    """
    return [a for a in read_annotations(str(path)) if is_detection(a)]


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


def is_person_signoff(a: Annotation) -> bool:
    """True when ``a`` carries a person's own sign-off: ``accepted_by`` is set and opens with
    :data:`PERSON_IDENTITY_PREFIX`."""
    return a.accepted_by is not None and a.accepted_by.startswith(PERSON_IDENTITY_PREFIX)


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
    unchanged); ``no_created_by``/``not_positively_a_persons``/``rule_admitted_unsigned`` are index
    lists into the annotations passed in, for a caller (the canopy rule, or the reference rail's
    own third arm) that names the specific record its own refusal is about.
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
    rule_admitted_unsigned: list[int]
    """Index of every record whose ``accepted_by_rule`` is not ``None`` (an empty string counts:
    never a truthiness test) and which carries no person's sign-off
    (:func:`is_person_signoff`): the reference rail's own third arm."""


def provenance_facts(annotations: list[Annotation]) -> ProvenanceFacts:
    """The :class:`ProvenanceFacts` classification over ``annotations``, the one computation both
    admissibility rules apply their own policy to (see that class's own docstring)."""
    scored = 0
    machine_authored: list[str] = []
    no_created_by: list[int] = []
    not_positively_a_persons: list[int] = []
    rule_admitted_unsigned: list[int] = []
    for i, a in enumerate(annotations):
        if is_unadjudicated_prediction(a):
            scored += 1
        if is_unadjudicated_agent_authorship(a):
            machine_authored.append(str(a.created_by))
        if not a.created_by:
            no_created_by.append(i)
        elif not a.created_by.startswith(PERSON_IDENTITY_PREFIX):
            if not is_person_signoff(a):
                not_positively_a_persons.append(i)
        if a.accepted_by_rule is not None and not is_person_signoff(a):
            rule_admitted_unsigned.append(i)
    return ProvenanceFacts(
        total=len(annotations), scored=scored, machine_authored=machine_authored,
        no_created_by=no_created_by, not_positively_a_persons=not_positively_a_persons,
        rule_admitted_unsigned=rule_admitted_unsigned,
    )


def require_reference_ground_truth(directory: str | Path) -> None:
    """Refuse ``directory`` as a measurement reference when only the model stands behind it.

    The rule every reference read shares: a calibration or held-out reference is only a measurement
    if something other than the model produced it. Two shapes fail it, both provenance rather than
    a number. A record carrying a prediction ``score`` is the model's own output. A record an agent
    authored as ground truth, with no reviewer's ``accepted_by``, is the same output with the score
    dropped. Either clears every numeric gate a calibration applies, since the model agrees with
    itself, so the gates cannot catch it and the provenance has to.

    A third shape fails it too: a record carrying ``accepted_by_rule`` (a rule-based admission
    was verified for it) with no person's sign-off. No platform producer writes such a record (the
    Review accept stamps the person in the same write that stamps the rule), so this arm is
    defence in depth against a hand-edited or round-tripped document, never a shape a normal
    accept can reach.

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
    # (path, index within that document): the third arm's own walk, so its refusal can name a
    # file to open rather than an index into the concatenation nothing else reconstructs.
    sources: list[tuple[Path, int]] = []
    for path in prediction_documents(directory):
        doc = read_annotations(path)
        annotations.extend(doc)
        sources.extend((path, i) for i in range(len(doc)))
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
    if facts.rule_admitted_unsigned:
        indices = facts.rule_admitted_unsigned
        named = ", ".join(f"{sources[i][0]} record {sources[i][1]}" for i in indices)
        raise ValueError(
            f"{len(indices)} of {total} annotations in {directory} ({named}) carry "
            "accepted_by_rule with no person's accepted_by: a rule-based admission was verified "
            "for these records, but nobody has signed off on them, so no person stands behind "
            "them yet. Confirm each one through Review so it carries the person's sign-off, or "
            "delete it."
        )


# ── writer ─────────────────────────────────────────────────────────────────


def xywh(x1: float, y1: float, x2: float, y2: float, *, on_grid: bool = True) -> list[float]:
    """A box in corner form as this schema's ``bbox``: COCO ``[x, y, w, h]``, the one conversion.

    A pixel box is put on the 2-decimal quantum, the grid the stored document lives on, so it is
    also the grid anything comparing a stored box against another box has to be on. Public
    because that comparison happens outside this package too (evaluation puts both sides of a
    match on this grid before scoring); one implementation, so a reader and a scorer cannot land
    on different grids. ``on_grid=False`` is for a box on no pixel grid at all (a record kept on
    the normalized unit square), which the quantum would erase.

    The inverse, reading such a record back, is ``BBox(x, y, x + w, y + h)`` in
    :func:`_annotations_of`, which is what keeps write and read symmetric.
    """
    box = [x1, y1, x2 - x1, y2 - y1]
    return [round(v, 2) for v in box] if on_grid else box


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
    """
    if polygonal(geometry):
        return stored_box_extent_ok(bbox_of(Polygon(_rounded_rings(geometry.rings))))
    return stored_box_extent_ok(cast(BBox, geometry))


def stored_content(a: Annotation) -> dict:
    """One annotation as its stored JSON object, provenance aside: its content on the stored
    grid, which the writer stores and a content digest of stored annotations hashes. Its
    attributes are read through :func:`attributes_of`, so a record the writer stores is one the
    reader accepts, raising ``ValueError`` otherwise."""
    rec: dict = {"subject": a.subject}
    geom = a.geometry
    if polygonal(geom):
        rounded_rings = _rounded_rings(geom.rings)
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
    """``a`` as the dict a client reads, the one projection every read door shares (a tool
    response, the Annotate and Review canvases).

    ``bbox`` in corner form ``[x1, y1, x2, y2]``; ``rings`` for a polygon, every ring, since a
    stored polygon can be an occlusion-split instance of more than one; ``point`` for a placed
    prompt or keypoint, the on-disk key; ``attributes``; ``score`` for a prediction; ``iscrowd``
    always stated, so a crowd region round-trips back on save; and the provenance fields the
    record holds (:func:`_held_provenance`), which is what reference admissibility turns on.
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


def encode_annotations(target, annotations, img_w: int, img_h: int, *,
                       keep_empty: bool = False) -> tuple[Key, bytes | None]:
    """The key ``target`` names and the exact bytes its per-image document holds.

    The writer's one encoder, apart from the write so a caller placing several documents can
    encode every one of them, and refuse on the first geometry the stored grid collapses
    (``ValueError``), before any lands. ``None`` for the bytes when no record survives encoding
    and ``keep_empty`` is not set: such a document is removed rather than written.
    """
    records = [{**stored_content(a), **_held_provenance(a)} for a in annotations]
    key = _record_key(target)
    if not records and not keep_empty:
        return key, None
    payload = {"image": key.parts[-1], "width": int(img_w), "height": int(img_h),
               ANNOTATIONS_KEY: records}
    return key, _document_bytes(payload)


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
    key, data = encode_annotations(target, annotations, img_w, img_h, keep_empty=keep_empty)
    if data is None:
        tcip_store.delete(key, expect=expect)
        return None
    return tcip_store.put_blob(key, data, expect=expect)


# ── the one target-membership decision (shared by assembly and the loader) ───


UNLABELED = "unlabeled"  # a real target this scope covers, but not yet assessed for `attribute`


def target_class_id(a: Annotation, subject: str, attribute: str | None,
                    id_map: dict[str, int], *, allow_unlabeled: bool = False
                    ) -> int | None | str:
    """The 0-indexed class id ``a`` trains as for ``(subject, attribute)``.

    Returns ``None`` if ``a`` is of a different subject, no target of this scope at all. Which
    geometry a target is read from is the caller's statement (the loader's ``reads_geometry``,
    through ``label_queries.json_det_targets``), never asked again here. For a genuine
    target, two different failure shapes exist and must not be conflated: the instance was
    never assessed for ``attribute`` at all (``a.attributes.get(attribute) is None``, the annotator
    hasn't gotten to it yet, a soft/expected gap), versus the instance was assessed but with a value
    the registry cannot decode (a real decode bug, the registry and the labels disagree). The first
    case returns the distinguishable sentinel ``UNLABELED`` when ``allow_unlabeled=True`` (opt-in,
    default ``False`` raises for both failure shapes unless the caller opts in); the second always raises,
    regardless of ``allow_unlabeled``, since a real annotation read as nothing is a measurement bug,
    never something to drop silently.

    The single membership+id decision every per-image target reader calls, so training and
    calibration/eval ground truth can never disagree about which annotation is a target or which
    class it is.
    """
    if a.subject != subject:
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
    ``tcip repair-classified-predictions`` (a ``tcip-mcp`` command) run over the bucket.
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
            "tcip repair-classified-predictions over this bucket."
        )
    value = a.attributes.get(attribute)
    if value is None:
        raise ClassifiedRecordRefused(
            f"{source}: record of {subject!r} carries no value under attribute {attribute!r}. "
            "Run tcip repair-classified-predictions over this bucket."
        )
    if value not in vocabulary:
        raise ClassifiedRecordRefused(
            f"{source}: record's value {value!r} is not a member of this bucket's own vocabulary "
            f"({sorted(vocabulary)}). Run tcip repair-classified-predictions over this "
            "bucket."
        )
    return value
