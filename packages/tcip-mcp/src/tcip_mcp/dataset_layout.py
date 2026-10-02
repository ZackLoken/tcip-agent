"""Canonical dataset-layout resolver: where an image's ground-truth labels and model predictions
live on disk.

Canonical layout (the label tree mirrors ``images/<date>/`` so stem-pairing is trivial and capture
dates never collide). Labels are one file per image, holding every subject's annotations by name;
the on-disk path carries no subject or task segment: those are properties of the records inside the
file, resolved through the dataset's single subject registry::

    <dataset_root>/
        images/<date>/<stem>.<imgext>
        annotations/<date>/<stem>.json      # ground truth (all subjects for the image)
        predictions/<model>/<date>/<stem>.json   # model outputs
        subjects.json                        # the nested registry: subjects -> attributes ->
        values

The subject registry lives in the dataset and travels with the labels. This module never parses
``subjects.json`` (its contents belong to :mod:`tcip_mcp.subject_registry`). It owns the
dataset-root stores it registers below and the one label save.

``<date>`` of ``None`` (non-dated datasets) simply omits that segment.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

import tcip_store
from tcip_store import (
    RECORD_JSON,
    Key,
    StoreDescriptor,
    check_schema_version,
    get_descriptor,
    register_store,
)
from tcip_store.file_backend import RootedFileLocator

from tcip_annotation.json_io import LABEL_SUFFIX

if TYPE_CHECKING:
    from tcip_mcp.buckets import Bucket

#: Geometry kinds a task authors, kept as a selector, not a label-path segment.
TASKS = ("detect", "segment")
SUBJECTS_FILENAME = "subjects.json"

UNDATED_BUCKET = "undated"
"""The bucket a dateless capture lands in: ``ingest_images`` writes it, and any store key that
would otherwise hold an empty date segment (the proposal-staging address included) addresses it
under this token instead of a spelling of its own."""


def is_bucket_name(name: str) -> bool:
    """Whether ``name`` is legal as a bucket directory name under ``images/`` or a prediction
    model directory under ``predictions/``: a single safe path segment (see
    ``workspace.is_valid_name``) that does not start with a dot, so a hidden directory (an
    editor's swap file, platform cruft) is never mistaken for one."""
    from tcip_mcp.workspace import is_valid_name

    return is_valid_name(name) and not name.startswith(".")


# ── the dataset-root stores ──────────────────────────────────────────────────

_STATE_DOC = RootedFileLocator(prefix=(".tcip", "state"), suffix=".json")
"""A dataset's own private state documents, one of each per dataset."""

_DATASET_DOC = RootedFileLocator(suffix=".json")
"""The documents that travel with the image set, at the dataset root itself."""

_IMAGE_TREE = RootedFileLocator(prefix=("images",))
"""The ingested imagery, one file per capture under its date bucket. No suffix on the locator:
the extension is part of the file's own name, because a dataset holds whatever formats its
captures came in."""

_LABEL_TREE = RootedFileLocator(prefix=("annotations",), suffix=LABEL_SUFFIX)
"""Ground truth, one file per image under its capture date."""

_PREDICTION_TREE = RootedFileLocator(prefix=("predictions",), suffix=LABEL_SUFFIX)
"""Model outputs, one file per image under a model bucket and capture date."""


def _entry_path(locator: RootedFileLocator, scope: str | Path, parts: tuple[str, ...]) -> Path:
    """The absolute path a locator places an entry at under ``scope``."""
    return Path(scope, *locator.relative_path(str(scope), parts).parts)


def _document_of(filename: str) -> tuple[str, ...]:
    """The key parts addressing a store that holds exactly one document per scope."""
    return (Path(filename).stem,)


def parse_image_path(image_path: str | Path) -> tuple[Path, Optional[str], str]:
    """Return ``(dataset_root, date, stem)`` for an image path.

    Handles both canonical date-nested (``<root>/images/<date>/<stem>``) and flat
    (``<root>/images/<stem>``) images; ``date`` is ``None`` for the flat form. Raises on any other
    shape.
    """
    img = Path(image_path)
    stem = img.stem
    parent = img.parent
    if parent.name == "images":
        return parent.parent, None, stem
    if parent.parent.name == "images":
        return parent.parent.parent, parent.name, stem
    raise ValueError(
        f"parse_image_path: {image_path!r} is not under a recognized dataset image tree "
        "(<root>/images/<date>/<stem> or <root>/images/<stem>), refusing to guess a dataset root."
    )


def capture_of(source: str | Path) -> tuple[Optional[str], Optional[str]]:
    """``(dataset_id, date)`` of the capture an image source belongs to: the identity record's id
    of the dataset root whose image tree holds it, and its capture date folder. Each is ``None``
    when the source sits in no dataset image tree, when the root records no identity, or, for the
    date, when the image sits in the flat undated tree."""
    try:
        root, date, _stem = parse_image_path(source)
    except ValueError:
        return None, None
    identity = read_dataset_identity(root)
    return (identity["id"] if identity is not None else None), date


def _date_seg(date: Optional[str]) -> tuple[str, ...]:
    """A capture date as the path segment it names; ``ValueError`` for one that is not a single
    safe segment."""
    from tcip_mcp.workspace import is_valid_name

    if date and not is_valid_name(date):
        raise ValueError(
            f"date must be a single safe path segment (no separators/'..'), got {date!r}")
    return (date,) if date else ()


def image_root(dataset_root: str | Path) -> Path:
    """``<dataset_root>/images/``: the whole image tree, every capture date under it."""
    return Path(dataset_root, *_IMAGE_TREE.prefix)


def image_dir(dataset_root: str | Path, date: Optional[str]) -> Path:
    """``<dataset_root>/images/[<date>/]``: where an image's bytes live."""
    return image_root(dataset_root).joinpath(*_date_seg(date))


def image_filename(stem: str, ext: str) -> str:
    """The file name one capture's bytes are stored under (``ext`` includes the leading dot)."""
    return f"{stem}{ext}"


def image_path(dataset_root: str | Path, date: Optional[str], stem: str, ext: str) -> Path:
    """Canonical write path for an image (``ext`` includes the leading dot)."""
    return _entry_path(_IMAGE_TREE, dataset_root, (*_date_seg(date), image_filename(stem, ext)))


def resolve_images_dir(dataset_root: str | Path, date: Optional[str]) -> Path:
    """The directory one date's images actually live in: ``images/<date>/`` when that bucket exists
    on disk, else the flat ``images/`` root.
    """
    dated = image_dir(dataset_root, date)
    return dated if dated.is_dir() else image_dir(dataset_root, None)


def resolve_image_name(dataset_root: str | Path, date: Optional[str], stem: str) -> Optional[str]:
    """The on-disk display name of the logical image at ``stem`` for one capture date.

    Resolves through :func:`resolve_images_dir`, then within that directory through the same
    resolution :func:`~tcip_mcp.pipelines.image_utils.resolve_image_source` gives every other
    by-name reader, over that module's own extension set. ``None`` when no logical image at
    ``stem`` resolves in that directory. A stem sitting at the flat ``images/`` root while a dated
    bucket for the same date also exists on disk is a mixed layout this does not pair:
    :func:`resolve_images_dir` picks the dated bucket whenever one exists, with no per-stem
    fallback to the flat root.

    Lets :class:`~tcip_mcp.pipelines.image_utils.AmbiguousImageStem` propagate uncaught.
    """
    from tcip_mcp.pipelines.image_utils import logical_image_name, resolve_image_source

    try:
        source = resolve_image_source(resolve_images_dir(dataset_root, date), stem)
    except FileNotFoundError:
        return None
    return logical_image_name(source)


IMAGERY_STORE = "imagery"
register_store(
    StoreDescriptor(
        name=IMAGERY_STORE,
        kind="blob",
        key_fields=("date", "filename"),
        frozen=True,
        cannot_carry_field="raw capture bytes (JPEG/PNG/TIFF/GeoTIFF/NPZ), nothing to version",
        path_readable=True,
        locator=_IMAGE_TREE,
    )
)


def image_key(dataset_root: str | Path, date: str, stem: str, ext: str) -> Key:
    """One ingested capture's bytes, a path-readable blob.

    ``date`` is required: the ingest key is dated by design, with no undated form of its own. The
    flat ``images/`` root (:func:`image_dir` with ``date=None``) is the undated imagery form,
    addressed by path rather than through this key.
    """
    if not date:
        raise ValueError(
            f"image_key needs a capture date for {stem!r}: the undated dataset layout "
            f"({image_dir(dataset_root, None)}) has no key shape yet"
        )
    return Key(IMAGERY_STORE, str(dataset_root), (date, image_filename(stem, ext)))


def list_dates(dataset_root: str | Path,
               tree: Callable[[str | Path], Path] | None = None) -> list[str]:
    """Sorted bucket names under ``images/``, or under the tree ``tree`` names
    (:func:`annotation_root`, say): an ISO ``YYYY-MM-DD`` date, ``UNDATED_BUCKET``, or a literal
    bucket (e.g. a plot name) ``ingest_images`` was told to use. A dot-prefixed directory is never
    a bucket (see ``is_bucket_name``) and is excluded, so a hidden directory is invisible to every
    listing here."""
    imgs = (tree or image_root)(dataset_root)
    if not imgs.is_dir():
        return []
    return sorted(p.name for p in imgs.iterdir() if p.is_dir() and is_bucket_name(p.name))


def annotation_root(dataset_root: str | Path) -> Path:
    """``<dataset_root>/annotations/``: the whole ground-truth tree, every capture date under it."""
    return Path(dataset_root, *_LABEL_TREE.prefix)


def annotation_dir(dataset_root: str | Path, date: Optional[str]) -> Path:
    """``<dataset_root>/annotations/[<date>/]`` (ground truth, one file per image, all subjects)."""
    return annotation_root(dataset_root).joinpath(*_date_seg(date))


def annotation_date(path: str | Path) -> Optional[str]:
    """The ``<date>`` an annotations dir/file lives under, or ``None`` (declared inverse of the
    ``annotations/<date>/`` layout; the only recoverable path fact, since subject/task live in the
    record, not the path).

    ``<root>/annotations`` and non-canonical trees (a split's ``labels/``) yield ``None``.
    """
    p = Path(path)
    parts = p.parts
    if "annotations" not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index("annotations")
    rest = parts[i + 1:]
    # A file (<date>/<stem>.json or <stem>.json) trims its trailing stem first.
    if rest and rest[-1].endswith(LABEL_SUFFIX):
        rest = rest[:-1]
    return rest[0] if len(rest) == 1 else None


#: The top-level segments under a dataset root; a path under any of them locates the root.
#: ``labels`` covers a tree whose label documents sit there rather than under ``annotations/``,
#: so the same locator resolves both shapes.
_DATASET_SEGMENTS = ("annotations", "predictions", "images", "labels")


def dataset_root_of(path: str | Path) -> Optional[Path]:
    """The ``<dataset_root>`` a canonical sub-path lives under, or ``None`` if it is not one.

    ``<dataset_root>/{annotations|predictions|images}/...`` -> ``<dataset_root>``. Lets a consumer
    that holds only a label or prediction dir locate the dataset-level ``subjects.json`` that decodes
    those names. Anchors on the *last* dataset segment in the path, so a dataset physically nested
    under an ancestor named ``images`` (or another segment) still resolves to the real root rather
    than the ancestor. A bare segment with nothing above it is not inside a dataset -> ``None``.
    """
    parts = Path(path).parts
    idxs = [k for k, p in enumerate(parts) if p in _DATASET_SEGMENTS]
    if not idxs:
        return None
    i = max(idxs)
    return Path(*parts[:i]) if i > 0 else None


def subjects_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/subjects.json``: the one nested registry that decodes the dataset's labels."""
    return _entry_path(_DATASET_DOC, dataset_root, _SUBJECT_REGISTRY_PARTS)


SUBJECT_REGISTRY_STORE = "subject_registry"
_SUBJECT_REGISTRY_PARTS = _document_of(SUBJECTS_FILENAME)
register_store(
    StoreDescriptor(
        name=SUBJECT_REGISTRY_STORE,
        kind="blob",
        key_fields=("document",),
        frozen=True,
        locator=_DATASET_DOC,
    )
)


def subject_registry_key(dataset_root: str | Path) -> Key:
    """The dataset's subject registry blob, encoded through ``RECORD_JSON`` in declared order."""
    return Key(SUBJECT_REGISTRY_STORE, str(dataset_root), _SUBJECT_REGISTRY_PARTS)


def dataset_identity_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/dataset.json``: the dataset's identity ({crop, id, fingerprint}).

    The stored fingerprint is a cache; recompute-on-read
    (``dataset_fingerprint.dataset_fingerprint``) is authority.
    """
    return _entry_path(_DATASET_DOC, dataset_root, _DATASET_IDENTITY_PARTS)


DATASET_IDENTITY_STORE = "dataset_identity"
_DATASET_IDENTITY_PARTS = _document_of("dataset.json")
register_store(
    StoreDescriptor(
        name=DATASET_IDENTITY_STORE,
        kind="blob",
        key_fields=("document",),
        frozen=True,
        locator=_DATASET_DOC,
    )
)


def dataset_identity_key(dataset_root: str | Path) -> Key:
    """The dataset's identity document, written compare-and-set through ``RECORD_JSON``."""
    return Key(DATASET_IDENTITY_STORE, str(dataset_root), _DATASET_IDENTITY_PARTS)


def decode_dataset_identity_document(data: bytes, *, dataset_root: str | Path) -> dict:
    """A dataset identity document's bytes, decoded and shape/version-checked.

    Raises ``ValueError`` for bytes that do not decode, or that decode to something other than a
    dict; a dict lacking ``id``, ``crop`` or ``fingerprint`` raises ``KeyError``. Propagates :class:`tcip_store.SchemaVersionRefused`, uncaught, for a
    ``schema_version`` this reader does not accept.
    """
    try:
        identity = RECORD_JSON.decode(data)
    except ValueError as exc:
        raise ValueError(
            f"{dataset_identity_path(dataset_root)} exists but does not decode as a dataset "
            f"identity ({exc}); re-register with register_dataset") from exc
    if not isinstance(identity, dict):
        raise ValueError(
            f"{dataset_identity_path(dataset_root)} exists but does not decode as a dataset "
            "identity; re-register with register_dataset")
    check_schema_version(get_descriptor(DATASET_IDENTITY_STORE), identity)
    _id, _crop, _fingerprint = identity["id"], identity["crop"], identity["fingerprint"]
    return identity


def read_dataset_identity(dataset_root: str | Path) -> dict | None:
    """The dataset's identity record (``{crop, id, fingerprint}``) decoded through
    :func:`decode_dataset_identity_document`, or ``None`` for a dataset never registered. A record
    that does not decode raises ``ValueError``; :class:`tcip_store.SchemaVersionRefused`
    propagates."""
    import tcip_store

    stored = tcip_store.read_blob_versioned(dataset_identity_key(dataset_root), default=None)
    if stored.value is None:
        return None
    return decode_dataset_identity_document(stored.value, dataset_root=dataset_root)


def require_dataset_identity(dataset_root: str | Path) -> dict:
    """:func:`read_dataset_identity`, refusing with ``ValueError`` naming ``register_dataset`` a
    dataset never registered."""
    identity = read_dataset_identity(dataset_root)
    if identity is None:
        raise ValueError(
            f"{dataset_root} carries no dataset identity record "
            f"({dataset_identity_path(dataset_root)} absent); register it first with "
            "register_dataset")
    return identity


def prediction_root(dataset_root: str | Path) -> Path:
    """``<dataset_root>/predictions/``: the whole prediction tree, every model bucket under it."""
    return Path(dataset_root, *_PREDICTION_TREE.prefix)


def label_filename(stem: str) -> str:
    """The file name one image's label or prediction record is written under."""
    return f"{stem}{LABEL_SUFFIX}"


def annotation_path(dataset_root: str | Path, date: Optional[str], stem: str) -> Path:
    return annotation_dir(dataset_root, date) / label_filename(stem)


def annotation_path_for_image(image_path: str | Path, *, date: Optional[str] = None) -> Path:
    """Canonical write path for an image's single label file (date derived from the image path)."""
    root, img_date, stem = parse_image_path(image_path)
    return annotation_path(root, date if date is not None else img_date, stem)


@dataclass(frozen=True)
class Gestures:
    """What one save decides beyond the annotations it writes: the proposals of the bucket at
    ``bucket`` it accepts and rejects, each by its index in that bucket's document for the image;
    each subject of ``complete`` marked complete over ``rect`` (pixel ``[x, y, w, h]``, the whole
    image when ``None``) or, mapped to ``False``, its marks withdrawn; and whether proposals were
    hidden while the person annotated."""

    bucket: Optional[str] = None
    accept: frozenset[int] = frozenset()
    reject: frozenset[int] = frozenset()
    complete: Mapping[str, bool] = field(default_factory=dict)
    rect: Optional[tuple[float, float, float, float]] = None
    proposals_hidden: bool = False


def image_proposals(bucket_dir: str | Path, image_path: str | Path) -> tuple["Bucket", list]:
    """The published bucket at ``bucket_dir`` and its proposals for ``image_path``, in document
    order; a bucket that names no document for the image refuses (``ValueError``)."""
    from tcip_annotation.json_io import read_predictions

    from tcip_mcp.buckets import read_bucket

    bucket = read_bucket(bucket_dir)
    document = bucket.document(image_path)
    if document is None:
        raise ValueError(f"{bucket.path} holds no proposals for {Path(image_path).name}")
    return bucket, read_predictions(str(document))


def verdict_key_of(project: str | Path | None, image_path: str | Path,
                   bucket_dir: str | Path) -> Key:
    """The verdict shard of ``image_path``'s proposals in the bucket at ``bucket_dir``, in the
    state directory of the dataset the image lies in, or of ``project`` when it lies in none;
    refuses (``ValueError``) with neither."""
    from tcip_annotation.verdicts import verdict_key

    from tcip_mcp.audit import dataset_scope_of
    from tcip_mcp.buckets import bucket_key_of
    from tcip_mcp.project_paths import project_state_dir

    scope = dataset_scope_of(image_path) or project
    if scope is None:
        raise ValueError(f"{image_path} lies in no dataset and no project is open to hold its "
                         "verdicts; open the project the image belongs to")
    return verdict_key(project_state_dir(scope), bucket_key_of(bucket_dir), Path(image_path).name)


def proposal_pairs(project: str | Path | None, bucket: "Bucket", annotations: list,
                   proposals: list) -> dict[int, int]:
    """Which annotation each of ``bucket``'s proposals pairs with, proposal index to annotation
    index, by the one matcher (:func:`~tcip_annotation.matching.pair_proposals`) under the
    localization criterion the assessment ``bucket`` was published under measured its count by;
    a bucket published under none, or under one that measured no count, pairs under the
    platform's comparability convention. Refuses (``ValueError``) a bucket's assessment with no
    ``project`` to read it from."""
    from tcip_annotation.matching import pair_proposals

    from tcip_mcp.assessment import read_assessment
    from tcip_mcp.pipelines.training.evaluation import resolve_match_criterion

    criterion = None
    if bucket.assessment_id is not None:
        if project is None:
            raise ValueError(f"{bucket.path} was published under assessment "
                             f"{bucket.assessment_id}, which only its project holds; open it")
        count = read_assessment(project, bucket.assessment_id).criterion.get("count")
        criterion = count["localization"] if count else None
    m = pair_proposals(annotations, proposals, criterion or resolve_match_criterion(None, []))
    return {p: g for g, p in m.pairs}


def save_label_document(
    project: str | Path | None, image_path: str | Path, label_path: str | Path,
    payloads: Iterable[dict], *, width: int, height: int, author: Optional[str],
    expect: Optional[tcip_store.Version] = None, gestures: Gestures = Gestures(),
) -> Optional[tcip_store.Version]:
    """Write ``image_path``'s label document at ``label_path`` and the save's one audit line, in
    the log of the dataset ``label_path`` lies in or ``project``'s when it lies in none. Returns
    the new version.

    The document holds every annotation parsed from ``payloads``, provenance stamped
    (:func:`~tcip_annotation.json_io.stamped`): a record unchanged since it was stored keeps its
    own, any other is ``author``'s at the save's time. Each accepted proposal of ``gestures``
    pairing no annotation (:func:`proposal_pairs`) joins it as ground truth, its geometry and
    subject only, every attribute unassessed, authored by its producer and accepted by
    ``author``; one that pairs confirms that annotation and adds nothing.
    The document's completion marks still live over the new annotations stay, beside the marks
    ``gestures`` makes; a subject mapped to ``False`` loses its marks. Each accepted and rejected
    proposal appends one entry to the image's verdict shard under that bucket, before the
    document is written.

    Raises, before writing anything: ``ValueError`` for a label path in no dataset with no
    ``project``, a payload that does not parse, a proposal index the bucket's document does not
    hold or that is both accepted and rejected, and a mark or a verdict with no ``author`` to
    record; ``VersionConflict`` when ``expect`` is not the version read. A document changed
    between that read and the write raises ``VersionConflict`` with the verdicts appended, and a
    landed write whose line cannot follow raises ``AuditEntryNotWritten``.
    """
    from tcip_annotation.json_io import (
        CompletionMark, annotation_from_payload, annotation_record_key, read_document_versioned,
        stamped, subject_digest, write_annotations,
    )
    from tcip_annotation.verdicts import Verdict, VerdictAction, record_verdicts

    from tcip_mcp.audit import dataset_scope_of, record_event_or_raise

    scope = dataset_scope_of(label_path) or project
    if scope is None:
        raise ValueError(f"{label_path} lies in no dataset and no project is open to record the "
                         "save in; open the project the labels belong to")
    if not author and (gestures.accept or gestures.reject or gestures.complete):
        raise ValueError("a completion mark and a verdict each record who made them; name the "
                         "person saving")
    now = datetime.now(timezone.utc).isoformat()
    contents = []
    for i, payload in enumerate(payloads):
        try:
            contents.append(annotation_from_payload(payload))
        except ValueError as exc:
            raise ValueError(f"annotation {i} {exc}") from exc
    stored, read = read_document_versioned(str(label_path))
    if expect is not None and expect != read:
        raise tcip_store.VersionConflict(
            annotation_record_key(Path(label_path).parent, Path(label_path).stem), expect, read)
    annotations = stamped(contents, stored.annotations, actor=author, now=now)
    verdicts: list[Verdict] = []
    if gestures.accept or gestures.reject:
        if gestures.bucket is None:
            raise ValueError("accepting or rejecting a proposal names the bucket it came from")
        both = gestures.accept & gestures.reject
        if both:
            raise ValueError(f"proposal(s) {sorted(both)} are both accepted and rejected; decide "
                             "each once")
        bucket, proposals = image_proposals(gestures.bucket, image_path)
        paired = proposal_pairs(project, bucket, annotations, proposals)
        decided: tuple[tuple[VerdictAction, frozenset[int]], ...] = (
            ("accepted", gestures.accept), ("rejected", gestures.reject))
        for action, indices in decided:
            for i in sorted(indices):
                if not 0 <= i < len(proposals):
                    raise ValueError(f"{bucket.path} holds {len(proposals)} proposals for "
                                     f"{Path(image_path).name}, not one at index {i}")
                if action == "accepted" and i not in paired:
                    annotations.append(replace(proposals[i], score=None, attributes={},
                                               accepted_by=author, accepted_at=now))
                verdicts.append(Verdict(proposal=i, action=action, by=cast(str, author), at=now))
    marks = {s: held for s, held in stored.marks.items() if gestures.complete.get(s, True)}
    for subject in (s for s, made in gestures.complete.items() if made):
        marks.setdefault(subject, []).append(CompletionMark(
            rect=gestures.rect or (0, 0, width, height), by=cast(str, author), at=now,
            digest=subject_digest(annotations, subject),
            proposals_hidden=gestures.proposals_hidden))
    if verdicts:
        record_verdicts(verdict_key_of(project, image_path, bucket.path), verdicts)
    Path(label_path).parent.mkdir(parents=True, exist_ok=True)
    version = write_annotations(str(label_path), annotations, width, height, keep_empty=True,
                                expect=read, marks=marks)
    record_event_or_raise("save_label_document", {
        "image_path": str(image_path), "label_path": str(Path(label_path).resolve()),
        "n_annotations": len(annotations), "version": version.token if version else None,
        "accepted": sorted(gestures.accept), "rejected": sorted(gestures.reject),
        "complete": dict(gestures.complete),
    }, scope=scope)
    return version


def list_subjects(dataset_root: str | Path) -> list[str]:
    """The dataset's subjects, in the registry's declared order. ``[]`` when there is no registry.

    A registry that is present but unreadable raises rather than reading as no subjects.
    """
    from tcip_mcp import subject_registry

    if not subjects_path(dataset_root).is_file():
        return []
    try:
        registry = subject_registry.read_registry(dataset_root)
    except OSError:
        return []
    return [s.name for s in registry.subjects]


def subjects_on_date(
    dataset_root: str | Path, date: Optional[str], *, reader: Optional[Callable] = None,
) -> list[str]:
    """Distinct subjects that actually appear in the per-image label files on ``date``, sorted.

    Enumerates ``annotations/<date>/`` through ``json_io.prediction_documents`` and reads each
    document through ``reader`` (``json_io.read_annotations`` by default). Raises
    :class:`~tcip_annotation.json_io.UnreadableLabelDocument` when a present label file on this
    date will not read; a missing ``annotations/<date>/`` directory reads as no subjects.
    """
    from tcip_annotation import json_io

    if reader is None:
        reader = json_io.read_annotations
    d = annotation_dir(dataset_root, date)
    if not d.is_dir():
        return []
    found: set[str] = set()
    for f in json_io.prediction_documents(d):
        for a in reader(f):
            found.add(a.subject)
    return sorted(found)


def subjects_with_labels(
    dataset_root: str | Path, date: Optional[str], *, reader: Optional[Callable] = None,
) -> list[str]:
    """Subjects with at least one label on ``date``."""
    return subjects_on_date(dataset_root, date, reader=reader)


def find_gt_label(image_path: str | Path, *, date: Optional[str] = None) -> Optional[Path]:
    """Find the existing ground-truth label file for an image (read-time resolver).

    One file per image, so this resolves ``annotations/<date>/<stem>.json`` directly. Returns
    the file, or ``None``.
    """
    root, img_date, stem = parse_image_path(image_path)
    cand = annotation_path(root, date if date is not None else img_date, stem)
    return cand if cand.is_file() else None
