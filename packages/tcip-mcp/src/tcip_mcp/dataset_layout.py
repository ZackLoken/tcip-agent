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
dataset-root stores it registers below, with their status vocabulary, derivation and writers.

``<date>`` of ``None`` (non-dated datasets) simply omits that segment.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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

from tcip_annotation.json_io import LABEL_SUFFIX, prediction_documents

DEFAULT_MODEL = "live"
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


def _date_seg(date: Optional[str]) -> tuple[str, ...]:
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


def list_dates(dataset_root: str | Path) -> list[str]:
    """Sorted bucket names under ``images/``: an ISO ``YYYY-MM-DD`` date, ``UNDATED_BUCKET``, or a
    literal bucket (e.g. a plot name) ``ingest_images`` was told to use. A dot-prefixed directory
    is never a bucket (see ``is_bucket_name``) and is excluded, so a hidden directory under
    ``images/`` is invisible to every listing here."""
    imgs = image_root(dataset_root)
    if not imgs.is_dir():
        return []
    return sorted(p.name for p in imgs.iterdir() if p.is_dir() and is_bucket_name(p.name))


def annotation_root(dataset_root: str | Path) -> Path:
    """``<dataset_root>/annotations/``: the whole ground-truth tree, every capture date under it."""
    return Path(dataset_root, *_LABEL_TREE.prefix)


def annotation_dir(dataset_root: str | Path, date: Optional[str]) -> Path:
    """``<dataset_root>/annotations/[<date>/]`` (ground truth, one file per image, all subjects)."""
    return annotation_root(dataset_root).joinpath(*_date_seg(date))


def prediction_bucket_date(path: str | Path) -> Optional[str]:
    """The ``<date>`` a prediction bucket lives under (``<root>/predictions/<model>/<date>/``), or
    ``None`` for an undated bucket (``<root>/predictions/<model>/``).

    Answers ``None`` for a path whose model slot fails :func:`is_bucket_name`, which is every shape
    under the cleared archive (``predictions/.cleared/...``).
    """
    p = Path(path)
    parts = p.parts
    if "predictions" not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index("predictions")
    rest = parts[i + 1:]
    if not rest:
        return None
    model, rest = rest[0], rest[1:]  # drop <model>
    if not is_bucket_name(model):
        return None
    if rest and rest[-1].endswith(LABEL_SUFFIX):
        rest = rest[:-1]
    return rest[0] if len(rest) == 1 else None


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
#: ``labels`` covers a curated tree whose label documents sit there rather than under
#: ``annotations/``, so the same locator resolves both shapes.
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


def bucket_dataset_root(bucket: str | Path) -> Optional[Path]:
    """The resolved dataset root a prediction bucket's current path sits under, or ``None``."""
    root = dataset_root_of(bucket)
    return root.resolve() if root is not None else None


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


def require_dataset_identity(dataset_root: str | Path) -> dict:
    """The dataset's identity record (``{crop, id, fingerprint}``), decoded through
    :func:`decode_dataset_identity_document`.

    Read through the store, never a bare file check. Raises ``ValueError`` naming
    ``register_dataset`` when the record is absent or malformed; a
    :class:`tcip_store.SchemaVersionRefused` propagates uncaught.
    """
    import tcip_store

    stored = tcip_store.read_blob_versioned(dataset_identity_key(dataset_root), default=None)
    if stored.value is None:
        raise ValueError(
            f"{dataset_root} carries no dataset identity record "
            f"({dataset_identity_path(dataset_root)} absent); register it first with "
            "register_dataset")
    return decode_dataset_identity_document(stored.value, dataset_root=dataset_root)


def image_status_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/.tcip/state/image_status.json``: the confirmed-negatives store.

    Shape: ``{bucket: {image_name: {status, recorded_by, recorded_at}}}``, bucket via
        :func:`status_bucket`. Each record says who set the status and when.
    """
    return _entry_path(_STATE_DOC, dataset_root, _IMAGE_STATUS_PARTS)


IMAGE_STATUS_STORE = "image_status"
_IMAGE_STATUS_PARTS = _document_of("image_status.json")
register_store(
    StoreDescriptor(
        name=IMAGE_STATUS_STORE,
        kind="record",
        key_fields=("document",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="cas",
        locator=_STATE_DOC,
    )
)


def image_status_key(dataset_root: str | Path) -> Key:
    """The dataset's confirmed-negative store, written compare-and-set."""
    return Key(IMAGE_STATUS_STORE, str(dataset_root), _IMAGE_STATUS_PARTS)


def read_image_status_store(dataset_root: str | Path) -> dict:
    """The dataset's stored image statuses as written, or ``{}`` when none is stored."""
    return tcip_store.read(image_status_key(dataset_root), default={})


def view_coverage_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/.tcip/state/view_coverage.json``: per-image record of two per-cell facts,
    the reference-grid cells the GUI has served at native resolution (a delivery fact) and, per
    cell, the tightest scale bound at which every one of its sub-cells has sat fully on screen
    (``cells_seen_at_scale``). Whether a seen cell counts as swept is derived in the browser
    against a subject's working-scale bar, never stored here.

    Shape: ``{bucket: {image_name: record}}``, bucket via :func:`status_bucket`; the record's own
        shape is declared once, by the web layer's coverage models (``CoverageRecord``: ``grid``,
        ``cells_served_at_native``, ``cells_seen_at_scale``, ``viewing``, ``updated_at``), with
        ``viewing`` the display context that layer's own ``CoverageViewing`` declares (bands,
        stretch, stats_source, display_bounds, base_served_size). Each record carries the grid
        geometry it was accumulated against.
    """
    return _entry_path(_STATE_DOC, dataset_root, _VIEW_COVERAGE_PARTS)


VIEW_COVERAGE_STORE = "view_coverage"
_VIEW_COVERAGE_PARTS = _document_of("view_coverage.json")
register_store(
    StoreDescriptor(
        name=VIEW_COVERAGE_STORE,
        kind="record",
        key_fields=("document",),
        frozen=False,
        codec=RECORD_JSON,
        concurrency="cas",
        locator=_STATE_DOC,
    )
)


def view_coverage_key(dataset_root: str | Path) -> Key:
    """The per-image view-coverage record, written compare-and-set."""
    return Key(VIEW_COVERAGE_STORE, str(dataset_root), _VIEW_COVERAGE_PARTS)


def image_status_digest_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/.tcip/state/image_status_digest.json``: ``{bucket: {image_name: digest}}``.

    Stamped per image by the writers of :func:`image_status_path` at confirmation time with the
    subject's attribute-schema digest in effect
    (:func:`tcip_mcp.subject_registry.attribute_schema_digest`). Absence of a stamp is not evidence
    of staleness: only a stamp that positively disagrees with the current schema is grounds to
    quarantine that one image.
    """
    return _entry_path(_STATE_DOC, dataset_root, _IMAGE_STATUS_DIGEST_PARTS)


IMAGE_STATUS_DIGEST_STORE = "image_status_digest"
_IMAGE_STATUS_DIGEST_PARTS = _document_of("image_status_digest.json")
register_store(
    StoreDescriptor(
        name=IMAGE_STATUS_DIGEST_STORE,
        kind="record",
        key_fields=("document",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="cas",
        locator=_STATE_DOC,
    )
)


def image_status_digest_key(dataset_root: str | Path) -> Key:
    """The schema stamps beside the confirmed-negative store, written compare-and-set."""
    return Key(IMAGE_STATUS_DIGEST_STORE, str(dataset_root), _IMAGE_STATUS_DIGEST_PARTS)


def region_completeness_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/.tcip/state/region_completeness.json``: per-subject attestations that every
    instance of a subject has been found within a reference-grid region's cells.

    Keyed by :func:`status_bucket` with the raster's own stem standing in for ``date``: one
    ``{grid, cells_complete, cells_attested_view, attested_by, attested_at, stem, date, subject}``
    record per bucket, not one per image name.
    """
    return _entry_path(_STATE_DOC, dataset_root, _REGION_COMPLETENESS_PARTS)


REGION_COMPLETENESS_STORE = "region_completeness"
_REGION_COMPLETENESS_PARTS = _document_of("region_completeness.json")
register_store(
    StoreDescriptor(
        name=REGION_COMPLETENESS_STORE,
        kind="record",
        key_fields=("document",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="cas",
        locator=_STATE_DOC,
    )
)


def region_completeness_key(dataset_root: str | Path) -> Key:
    """The region-completeness attestations, written compare-and-set."""
    return Key(REGION_COMPLETENESS_STORE, str(dataset_root), _REGION_COMPLETENESS_PARTS)


def region_completeness_digest_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/.tcip/state/region_completeness_digest.json``: ``{bucket: {cell_name:
    digest}}``.

    Stamped per cell at attestation time with a content digest of the subject's annotations found
    inside that cell (see :mod:`tcip_mcp.pipelines.region_completeness`), so a reader can tell an
    attestation whose cell content has since been edited or deleted from one still valid.
    """
    return _entry_path(_STATE_DOC, dataset_root, _REGION_COMPLETENESS_DIGEST_PARTS)


REGION_COMPLETENESS_DIGEST_STORE = "region_completeness_digest"
_REGION_COMPLETENESS_DIGEST_PARTS = _document_of("region_completeness_digest.json")
register_store(
    StoreDescriptor(
        name=REGION_COMPLETENESS_DIGEST_STORE,
        kind="record",
        key_fields=("document",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="cas",
        locator=_STATE_DOC,
    )
)


def coverage_grid_zoom_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/.tcip/state/coverage_grid_zoom.json``: the breeder-set inspection zoom the
    coverage lattice's cell size is derived from, one entry per subject.

    Shape: ``{subject: {zoom, set_by, set_at}}``. Advisory, like :func:`view_coverage_path`: no
        default zoom exists, and a subject absent from this store simply has no coverage lattice
        yet.
    """
    return _entry_path(_STATE_DOC, dataset_root, _COVERAGE_GRID_ZOOM_PARTS)


COVERAGE_GRID_ZOOM_STORE = "coverage_grid_zoom"
_COVERAGE_GRID_ZOOM_PARTS = _document_of("coverage_grid_zoom.json")
register_store(
    StoreDescriptor(
        name=COVERAGE_GRID_ZOOM_STORE,
        kind="record",
        key_fields=("document",),
        frozen=False,
        codec=RECORD_JSON,
        concurrency="cas",
        locator=_STATE_DOC,
    )
)


def coverage_grid_zoom_key(dataset_root: str | Path) -> Key:
    """The per-subject coverage-lattice zoom, written compare-and-set."""
    return Key(COVERAGE_GRID_ZOOM_STORE, str(dataset_root), _COVERAGE_GRID_ZOOM_PARTS)


def region_completeness_digest_key(dataset_root: str | Path) -> Key:
    """The content stamps beside the region-completeness attestations, written compare-and-set."""
    return Key(REGION_COMPLETENESS_DIGEST_STORE, str(dataset_root),
               _REGION_COMPLETENESS_DIGEST_PARTS)





def status_bucket(subject: str, date: Optional[str]) -> str:
    """The ``image_status.json`` key a confirmation belongs under.

    Scoped by subject and date but not task: a Complete covers detect and segment together.
    ``subject`` must be a real subject; there is no catch-all default.
    """
    return f"{subject}/{date}" if date else subject


def bucket_subject_date(bucket: str) -> tuple[str, Optional[str]]:
    """The ``(subject, date)`` a bucket key was built from: the declared inverse of
    :func:`status_bucket`, so a reader that has to take a key apart never re-derives the
    separator the writer used."""
    subject, _, date = bucket.partition("/")
    return subject, (date or None)


def status_of(record: Mapping[str, str]) -> str:
    """The status token a stored ``{"status", "recorded_by", "recorded_at"}`` record holds. A
    record lacking any of the three raises ``KeyError`` naming it.
    """
    status, _by, _at = record["status"], record["recorded_by"], record["recorded_at"]
    return status


def status_records(
    statuses: Mapping[str, str], *, recorded_by: str, recorded_at: Optional[str] = None
) -> dict[str, dict[str, str]]:
    """One bucket's ``{image_name: status}`` as stored records, attributed to ``recorded_by``.

    ``recorded_by`` names the actor the status came from under the platform's identity convention
    (:func:`tcip_mcp.identity.user_identity` for a person, a bare name for a tool producer).
    ``recorded_at`` defaults to the moment of this call, one timestamp across the names in it.
    Refuses an unattributed write.
    """
    if not (recorded_by or "").strip():
        raise ValueError(
            "an image status records who set it, so recorded_by is required; pass the person's "
            "user:<name> identity or the writing tool's own name"
        )
    at = recorded_at or datetime.now(timezone.utc).isoformat()
    return {name: {"status": status, "recorded_by": recorded_by, "recorded_at": at}
            for name, status in statuses.items()}


def status_confirmations(
    raw: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> dict[str, dict[str, dict[str, str]]]:
    """``{bucket: {image_name: record}}``: the stored records whole, attribution included, each
    read through :func:`status_of`. A bucket is ``status_bucket(subject, date)``;
    :func:`status_tokens` is its status-token projection.
    """
    return {bucket: {name: dict(record, status=status_of(record))
                     for name, record in records.items()}
            for bucket, records in raw.items()}


def status_tokens(
    raw: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> dict[str, dict[str, str]]:
    """``{bucket: {image_name: status}}``: the status-token projection of
    :func:`status_confirmations`.
    """
    return {bucket: {name: status_of(record) for name, record in records.items()}
            for bucket, records in raw.items()}


CONFIRMED_NEGATIVE = "negative"
"""The status token for an image a human marked done with none of the subject on it."""

IMAGE_STATUSES = ("complete", "partial", CONFIRMED_NEGATIVE, "unannotated")
"""Every status the store holds, and the only values a write may record.

``"complete"`` and ``"negative"`` are opposites, not degrees: both mean the human finished the
image, and they differ on whether anything of the subject is on it. Anything reading ``"complete"``
as a confirmed negative trains populated images as empty.
"""

FINISHED_STATUSES = ("complete", CONFIRMED_NEGATIVE)
"""The two statuses a human's own confirmation ends on, as opposed to
:func:`status_confirmations`'s wider sense of every stored record: a ``partial`` or
``unannotated`` status is not a person's assertion about the subject. ``is_finished_status`` is
the membership predicate; the same pair the frontend declares as ``FINISHED_STATUSES`` in
``api/subjects.ts``, held equal to this one by ``tests/test_frontend_dataset_vocabulary.py``.
"""


def derive_status(*, completed: bool, has_content: bool) -> str:
    """The status one image holds for one subject, from the human's Complete and what is labeled.

    ``has_content`` is whether the image carries any annotation of the subject in question. An
    uncompleted empty image is ``"unannotated"`` rather than a negative.
    """
    if completed:
        return "complete" if has_content else CONFIRMED_NEGATIVE
    return "partial" if has_content else "unannotated"


def annotations_hold_subject(annotations: Iterable, subject: str) -> bool:
    """Whether any of ``annotations`` (as :func:`tcip_annotation.json_io.read_annotations` returns
    them) names ``subject``, geometry or not.

    An image-level record (a subject with no geometry) counts as content for that subject.
    """
    return any(a.subject == subject for a in annotations)


def is_confirmed_negative(status: object) -> bool:
    """Whether a stored status is a human's confirmation that the image holds none of the subject."""
    return status == CONFIRMED_NEGATIVE


def is_finished_status(status: object) -> bool:
    """Whether a stored status is one of the two a human's own confirmation ends on (``complete``
    or ``negative``), as opposed to :func:`status_confirmations`'s wider sense of every stored
    record. A ``partial`` or ``unannotated`` status is not a person's assertion and is never
    finished.
    """
    return status in FINISHED_STATUSES


def confirmed_negative_names_any_subject(by_bucket: Mapping[str, Mapping[str, str]]) -> set[str]:
    """Every image file name confirmed negative for some subject, in any date bucket.

    Takes ``by_bucket`` already in :func:`status_tokens`'s shape rather than a root to read.
    """
    return {name for bucket in by_bucket.values() for name, status in bucket.items()
            if is_confirmed_negative(status)}


def _require_known_statuses(bucket: str, statuses: Iterable[str]) -> None:
    """Refuse a status outside :data:`IMAGE_STATUSES`."""
    unknown = sorted(set(statuses) - set(IMAGE_STATUSES))
    if unknown:
        raise ValueError(
            f"image status must be one of {IMAGE_STATUSES}, recorded with who set it and when; "
            f"refusing to record {unknown} for {bucket!r}"
        )


def record_image_statuses(
    dataset_root: str | Path, bucket: str, statuses: Mapping[str, str], *, recorded_by: str
) -> None:
    """Merge one bucket's per-image statuses into the dataset's confirmed-negative store.

    Merged, never replaced: every other subject's and date's confirmations stay exactly as they
    were. ``recorded_by`` is the actor this write is on behalf of, stamped onto each record by
    :func:`status_records`. A stored record lacking a field fails the merge at the read.
    """
    _require_known_statuses(bucket, statuses.values())
    records = status_records(statuses, recorded_by=recorded_by)
    key = image_status_key(dataset_root)
    with tcip_store.transaction(key) as txn:
        store = status_confirmations(txn.read(key, default={}))
        store.setdefault(bucket, {}).update(records)
        txn.write(key, {k: dict(sorted(store[k].items())) for k in sorted(store)})


def replace_image_status_store(
    dataset_root: str | Path, records_by_bucket: Mapping[str, Mapping[str, Mapping[str, str]]]
) -> None:
    """Write the whole confirmed-negative store for a dataset this call is producing.

    Takes whole records, not bare tokens: new confirmations are built with :func:`status_records`,
    and another dataset's confirmations pass through unchanged, attribution included.
    """
    for bucket, records in records_by_bucket.items():
        _require_known_statuses(bucket, (status_of(r) for r in records.values()))
    key = image_status_key(dataset_root)
    with tcip_store.transaction(key) as txn:
        txn.write(key, {k: {n: dict(records_by_bucket[k][n])
                            for n in sorted(records_by_bucket[k])}
                        for k in sorted(records_by_bucket)})


def bucket_digest_stamps(stamps: object, bucket: str) -> dict:
    """The ``bucket``-scoped image-to-digest map inside a raw digest-store document.

    Returns ``{}`` when ``stamps`` itself, or its value at ``bucket``, is not a dict: whatever a
    corrupt or absent read produced, never raised here. Always a copy, never the document's own
    inner dict, so a caller that mutates the result cannot reach back into ``stamps``.
    """
    if not isinstance(stamps, dict):
        return {}
    bucket_stamps = stamps.get(bucket)
    return dict(bucket_stamps) if isinstance(bucket_stamps, dict) else {}


def stamp_image_status_digests(
    dataset_root: str | Path, bucket: str, image_names: Iterable[str], digest: str,
    *, only_unstamped: bool = False,
) -> list[str]:
    """Record ``digest`` against each of ``image_names`` in ``bucket``, merging into what is there,
    and return the names this call stamped.

    ``only_unstamped`` leaves an image that already carries a stamp exactly as it is. The read and
    the write share one transaction, so a stamp landing between them is seen rather than clobbered.
    """
    key = image_status_digest_key(dataset_root)
    with tcip_store.transaction(key) as txn:
        stamps = txn.read(key, default={})
        if not isinstance(stamps, dict):
            stamps = {}
        bucket_stamps = bucket_digest_stamps(stamps, bucket)
        stamped = [name for name in image_names
                   if not (only_unstamped and isinstance(bucket_stamps.get(name), str))]
        if not stamped:
            return stamped  # nothing to record: leave the document, and its absence, untouched
        for name in stamped:
            bucket_stamps[name] = digest
        stamps[bucket] = dict(sorted(bucket_stamps.items()))
        txn.write(key, dict(sorted(stamps.items())))
    return stamped


def prediction_root(dataset_root: str | Path) -> Path:
    """``<dataset_root>/predictions/``: the whole prediction tree, every model bucket under it."""
    return Path(dataset_root, *_PREDICTION_TREE.prefix)


def prediction_dir(dataset_root: str | Path, model: Optional[str], date: Optional[str]) -> Path:
    """``<dataset_root>/predictions/<model>/[<date>/]`` (model outputs, one file per image)."""
    return prediction_root(dataset_root).joinpath(model or DEFAULT_MODEL, *_date_seg(date))


def canonical_prediction_bucket(
    path: str | Path,
) -> tuple[Path, str, Optional[str]] | None:
    """``(dataset_root, model, date)`` for a canonical prediction bucket path
    (``<dataset_root>/predictions/<model>[/<date>]``), or ``None`` when ``path`` is not one: not
    under a ``predictions/`` tree at all, its model segment fails :func:`is_bucket_name`, or the
    round trip through :func:`prediction_dir` does not spell ``path`` back exactly.
    """
    p = Path(path)
    parts = p.parts
    if "predictions" not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index("predictions")
    rest = parts[i + 1:]
    if not rest or len(rest) > 2:
        return None
    model, date = rest[0], (rest[1] if len(rest) == 2 else None)
    if not is_bucket_name(model):
        return None
    dataset_root = Path(*parts[:i])
    if prediction_dir(dataset_root, model, date) != p:
        return None
    return dataset_root, model, date


_CLEARED_SEGMENT = ".cleared"
"""The dot-prefixed directory a canonical prediction bucket moves into once
:func:`clear_prediction_bucket` clears it, invisible to every model reader (``is_bucket_name``
excludes any dot-prefixed segment) and enumerated only by the archive's own named walk
(``prediction_bucket_dirs(..., include_cleared=True)``)."""

CLEARED_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
"""The UTC stamp format a cleared bucket's own directory name carries, admitted by
``workspace.is_valid_name`` (which forbids ``:`` but admits ``@``) on every platform."""


def _is_cleared_stamp(value: str) -> bool:
    """Whether ``value`` parses as a :data:`CLEARED_STAMP_FORMAT` UTC stamp, with no filesystem or
    clock read.
    """
    try:
        datetime.strptime(value, CLEARED_STAMP_FORMAT)
    except ValueError:
        return False
    return True


def current_cleared_stamp() -> str:
    """The current UTC time in :data:`CLEARED_STAMP_FORMAT`."""
    return datetime.now(timezone.utc).strftime(CLEARED_STAMP_FORMAT)


def cleared_prediction_dir(
    dataset_root: str | Path, model: str, date: Optional[str], stamp: str,
) -> Path:
    """``<dataset_root>/predictions/.cleared/<model>@<stamp>/[<date>/]``: where
    :func:`clear_prediction_bucket` archives a canonical bucket it moves out from under a
    terminal experiment's own recorded path, one entry per call. ``stamp`` is the UTC time of the
    call, ``%Y%m%dT%H%M%SZ`` (``:`` forbidden by ``workspace.is_valid_name``, ``@`` admitted).
    """
    return prediction_root(dataset_root).joinpath(
        _CLEARED_SEGMENT, f"{model}@{stamp}", *_date_seg(date))


def cleared_bucket_of(
    path: str | Path,
) -> tuple[Path, str, str, Optional[str]] | None:
    """``(dataset_root, model, stamp, date)`` for a cleared-bucket path
    (:func:`cleared_prediction_dir`'s own shape), or ``None`` when ``path`` is not one: splits the
    cleared segment on its last ``@`` and requires the tail to parse as
    :data:`CLEARED_STAMP_FORMAT`. Reads the tail arity for the date, never the filesystem: a third
    segment past the cleared model directory is the date, a second is none.
    """
    p = Path(path)
    parts = p.parts
    if "predictions" not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index("predictions")
    rest = parts[i + 1:]
    if not rest or rest[0] != _CLEARED_SEGMENT:
        return None
    rest = rest[1:]
    if not rest or len(rest) > 2:
        return None
    cleared_segment, date = rest[0], (rest[1] if len(rest) == 2 else None)
    model, sep, stamp = cleared_segment.rpartition("@")
    if not sep or not model or not _is_cleared_stamp(stamp):
        return None
    dataset_root = Path(*parts[:i])
    return dataset_root, model, stamp, date


def is_cleared_bucket(path: str | Path) -> bool:
    """Whether ``path`` is a cleared bucket (:func:`cleared_bucket_of` recognizes it)."""
    return cleared_bucket_of(path) is not None


def label_filename(stem: str) -> str:
    """The file name one image's label or prediction record is written under."""
    return f"{stem}{LABEL_SUFFIX}"


def annotation_path(dataset_root: str | Path, date: Optional[str], stem: str) -> Path:
    return annotation_dir(dataset_root, date) / label_filename(stem)


def annotation_path_for_image(image_path: str | Path, *, date: Optional[str] = None) -> Path:
    """Canonical write path for an image's single label file (date derived from the image path)."""
    root, img_date, stem = parse_image_path(image_path)
    return annotation_path(root, date if date is not None else img_date, stem)


def prediction_path(
    dataset_root: str | Path, model: Optional[str], date: Optional[str], stem: str,
) -> Path:
    return prediction_dir(dataset_root, model, date) / label_filename(stem)


def list_subjects(dataset_root: str | Path) -> list[str]:
    """The dataset's subjects, in the registry's declared order. ``[]`` when there is no registry.

    A registry that is present but unreadable raises rather than reading as no subjects.
    """
    from tcip_mcp import subject_registry

    sp = subjects_path(dataset_root)
    if not sp.is_file():
        return []
    try:
        registry = subject_registry.read_registry(sp)
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
    date will not read; a missing ``annotations/<date>/`` directory reads as no subjects. A label
    whose filename is a bucket's own provenance stamp (``json_io.is_sidecar_name``) is excluded
    from the walk entirely.
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


def list_models(dataset_root: str | Path) -> list[str]:
    """Sorted live model bucket names under ``predictions/``. A dot-prefixed directory is never a
    bucket (see ``is_bucket_name``) and is excluded, which also excludes the cleared archive
    (``predictions/.cleared/``).
    """
    preds = prediction_root(dataset_root)
    if not preds.is_dir():
        return []
    return sorted(p.name for p in preds.iterdir() if p.is_dir() and is_bucket_name(p.name))


def prediction_bucket_dirs(dataset_root: str | Path, *, include_cleared: bool) -> list[Path]:
    """Every directory under ``predictions/`` a live bucket's own sidecar could sit in: each
    model's own directory, and each of its date subdirectories, whether or not either actually
    holds one.

    ``include_cleared`` takes no default. ``True`` appends a second walk over the cleared archive
    (``predictions/.cleared/``): each directory :func:`cleared_bucket_of` recognizes as a cleared
    model directory, and each of its date children in the same two shapes, following the first;
    never the ``.cleared`` directory itself, and never a directory under it the inverse does not
    recognize.
    """
    pred_root = prediction_root(dataset_root)
    if not pred_root.is_dir():
        return []
    found: list[Path] = []
    for model in list_models(dataset_root):
        model_dir = pred_root / model
        if not model_dir.is_dir():
            continue
        found.append(model_dir)
        found.extend(sorted(p for p in model_dir.iterdir() if p.is_dir()))
    if include_cleared:
        cleared_root = pred_root / _CLEARED_SEGMENT
        if cleared_root.is_dir():
            for entry in sorted(p for p in cleared_root.iterdir() if p.is_dir()):
                if not is_cleared_bucket(entry):
                    continue
                found.append(entry)
                found.extend(sorted(
                    p for p in entry.iterdir() if p.is_dir() and is_cleared_bucket(p)))
    return found


def models_with_predictions(dataset_root: str | Path, date: Optional[str]) -> list[str]:
    """Models whose bucket on ``date`` holds at least one per-image prediction document, in
    ``list_models`` order; a bucket holding only its stamp has none."""
    root = Path(dataset_root)
    return [model for model in list_models(root)
            if prediction_documents(prediction_dir(root, model, date))]


def find_gt_label(image_path: str | Path, *, date: Optional[str] = None) -> Optional[Path]:
    """Find the existing ground-truth label file for an image (read-time resolver).

    One file per image, so this resolves ``annotations/<date>/<stem>.json`` directly. Returns
    the file, or ``None``.
    """
    root, img_date, stem = parse_image_path(image_path)
    cand = annotation_path(root, date if date is not None else img_date, stem)
    return cand if cand.is_file() else None


def find_prediction(
    image_path: str | Path,
    *,
    model: Optional[str] = None,
    date: Optional[str] = None,
) -> Optional[Path]:
    """Find an existing prediction file for an image: a specific ``model`` if given, else every
    live model. Returns the first existing file, or ``None``.

    Raises ``ValueError`` when an explicit ``model`` fails :func:`is_bucket_name`, so a cleared
    bucket's archive segment is never reached through this resolver.
    """
    root, img_date, stem = parse_image_path(image_path)
    d = date if date is not None else img_date

    if model is not None and not is_bucket_name(model):
        raise ValueError(
            f"find_prediction: {model!r} is not a canonical model bucket name; the cleared "
            "archive and any other dot-prefixed segment are never addressed through this resolver"
        )
    models = [model] if model else list_models(root)
    for m in models:
        cand = prediction_path(root, m, d, stem)
        if cand.is_file():
            return cand
    return None
