"""Canonical dataset-layout resolver: the single source of truth for where an
image's ground-truth labels and model predictions live on disk.

Canonical layout (the label tree mirrors ``images/<date>/`` so stem-pairing is trivial and capture
dates never collide). Labels are one file per image, holding every subject's annotations by name;
the on-disk path carries no subject or task segment: those are properties of the records
inside the file, resolved through the dataset's single class registry::

    <dataset_root>/
        images/<date>/<stem>.<imgext>
        annotations/<date>/<stem>.json      # ground truth (all subjects for the image)
        predictions/<model>/<date>/<stem>.json   # model outputs
        classes.json                         # the nested registry: subjects -> attributes -> values

The class registry lives in the dataset and travels with the labels: a name-based label
(``subject``, attribute value) is undecodable without it. A second project opening the same image
set reads the same names. This module never parses ``classes.json`` (its contents belong to
:mod:`tcip_mcp.class_registry`); it only delegates to that module to list subjects. It does own the
dataset-root stores it registers below, so their status vocabulary, their derivation and their
writers live here beside the locators rather than in each caller.

``<date>`` of ``None`` (non-dated datasets) simply omits that segment. This is the single source of
truth for label/prediction locations: every producer and consumer resolves paths through here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import tcip_store
from tcip_store import (
    RECORD_JSON,
    DecodeError,
    Key,
    StoreDescriptor,
    check_schema_version,
    get_descriptor,
    register_store,
)
from tcip_store.file_backend import RootedFileLocator

# Per-image JSON is the canonical on-disk label format; ``coco`` is the assembled dataset view of it.
LABEL_EXT = {"json": ".json", "coco": ".json"}
_ANY_EXTS = (".json",)
DEFAULT_MODEL = "live"
#: Geometry kinds a task authors, kept as a selector, not a label-path segment.
TASKS = ("detect", "segment")
CLASSES_FILENAME = "classes.json"

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


def label_ext(fmt: Optional[str]) -> str:
    """File extension for a label format: always ``.json``; both formats are JSON on disk."""
    return LABEL_EXT.get((fmt or "json").lower(), ".json")


# ── the dataset-root stores ──────────────────────────────────────────────────

_STATE_DOC = RootedFileLocator(prefix=(".tcip", "state"), suffix=".json")
"""A dataset's own private state documents, one of each per dataset."""

_DATASET_DOC = RootedFileLocator(suffix=".json")
"""The documents that travel with the image set, at the dataset root itself."""

_IMAGE_TREE = RootedFileLocator(prefix=("images",))
"""The ingested imagery, one file per capture under its date bucket. No suffix on the locator:
the extension is part of the file's own name, because a dataset holds whatever formats its
captures came in."""

_LABEL_TREE = RootedFileLocator(prefix=("annotations",), suffix=LABEL_EXT["json"])
"""Ground truth, one file per image under its capture date."""

_PREDICTION_TREE = RootedFileLocator(prefix=("predictions",), suffix=LABEL_EXT["json"])
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
    shape: a guessed dataset root is a fabrication a downstream write (a label file, a staged
    prediction) would silently land at the wrong place, never a mitigation.
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
    """``<dataset_root>/images/``: the whole image tree, every capture date under it.

    The top-level call a consumer that walks or stats the tree needs, so an archiver, a scanner
    or a cache signature asks this module where the tree is instead of spelling the segment again.
    """
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
    """The directory one date's images actually live in: ``images/<date>/`` when that bucket
    exists on disk, else the flat ``images/`` root.

    A canonical dataset nests images under their capture date, but a labels tree can be dated
    while its images never were split into date buckets; a caller that needs one directory for a
    date (a label/image pair-up, an admission draw) resolves it here instead of assuming the
    dated bucket is where the bytes are.
    """
    dated = image_dir(dataset_root, date)
    return dated if dated.is_dir() else image_dir(dataset_root, None)


def resolve_image_name(dataset_root: str | Path, date: Optional[str], stem: str) -> Optional[str]:
    """The on-disk display name of the logical image at ``stem`` for one capture date.

    Resolves through :func:`resolve_images_dir`, the same directory a subject-scoped draw
    (``draw_splits``, a training run's own admission) reads that date's images from, so a label
    dated while its images were never split into date buckets resolves here the way it is
    admitted there, instead of naming a different directory than the draw does. Within that
    directory, the same resolution :func:`~tcip_mcp.pipelines.image_utils.resolve_image_source`
    gives every other by-name reader, over that module's own extension set, so a name behind a
    confirmed-negative lookup is never fabricated from a guessed extension. ``None`` when no
    logical image at ``stem`` resolves in that directory; a caller looking up a status-store entry
    treats that as unresolvable rather than guessing a name to look one up under. A stem sitting at
    the flat ``images/`` root while a dated bucket for the same date also exists on disk is a mixed
    layout this does not pair: :func:`resolve_images_dir` picks the dated bucket whenever one
    exists, with no per-stem fallback to the flat root.

    Lets :class:`~tcip_mcp.pipelines.image_utils.AmbiguousImageStem` propagate uncaught: an
    ambiguous directory is not one in which no image resolves, whatever stem is asked for, so it
    is never folded into the ``None`` answer.
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
    """One ingested capture's bytes.

    A blob: the imagery is the breeder's own data, it travels with the dataset, and nothing
    read-modify-writes it. Path-readable because the readers open it through libraries that
    take a path (rasterio, PIL) rather than a file object.

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
    """The ``<date>`` a prediction bucket lives under (``<root>/predictions/<model>/<date>/``),
    or ``None`` for an undated bucket (``<root>/predictions/<model>/``): the same declared-inverse
    contract :func:`annotation_date` states for the ``annotations/<date>/`` tree, mirrored for the
    ``predictions/<model>/`` tree instead, one model segment further in.
    """
    p = Path(path)
    parts = p.parts
    if "predictions" not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index("predictions")
    rest = parts[i + 1:]
    if not rest:
        return None
    rest = rest[1:]  # drop <model>
    if rest and rest[-1].endswith(".json"):
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
    if rest and rest[-1].endswith(".json"):
        rest = rest[:-1]
    return rest[0] if len(rest) == 1 else None


#: The top-level segments under a dataset root; a path under any of them locates the root.
#: ``labels`` covers a split-materialized tree (``draw_splits(materialize=True)`` writes
#: ``{split}/labels/*.json`` rather than ``annotations/``) so the same locator resolves both shapes.
_DATASET_SEGMENTS = ("annotations", "predictions", "images", "labels")


def dataset_root_of(path: str | Path) -> Optional[Path]:
    """The ``<dataset_root>`` a canonical sub-path lives under, or ``None`` if it is not one.

    ``<dataset_root>/{annotations|predictions|images}/...`` -> ``<dataset_root>``. Lets a consumer
    that holds only a label or prediction dir locate the dataset-level ``classes.json`` that decodes
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


def classes_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/classes.json``: the one nested registry that decodes the dataset's labels.

    In the dataset (not a project's private state) and shared across every subject, so it travels
    with the image set. A name-based label means nothing without it, so it is part of the data.
    """
    return _entry_path(_DATASET_DOC, dataset_root, _CLASS_REGISTRY_PARTS)


CLASS_REGISTRY_STORE = "class_registry"
_CLASS_REGISTRY_PARTS = _document_of(CLASSES_FILENAME)
register_store(
    StoreDescriptor(
        name=CLASS_REGISTRY_STORE,
        kind="blob",
        key_fields=("document",),
        frozen=True,
        locator=_DATASET_DOC,
    )
)


def class_registry_key(dataset_root: str | Path) -> Key:
    """The dataset's class registry.

    A blob because ``classes.json`` is part of the data: it travels with the image set, a
    breeder may open it, and an archive carries it as a file. ``class_registry`` encodes and
    decodes it through the canonical ``RECORD_JSON`` codec, whose ``sort_keys=False`` is what
    keeps the subject and attribute sequences in the order they were declared.
    """
    return Key(CLASS_REGISTRY_STORE, str(dataset_root), _CLASS_REGISTRY_PARTS)


def dataset_identity_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/dataset.json``: the dataset's identity ({crop, id, fingerprint}).

    Sibling of ``classes.json``: identity is part of the data, so it travels with the image set. The
    stored fingerprint is a cache; recompute-on-read (``dataset_fingerprint.dataset_fingerprint``)
    is authority.
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
    """The dataset's identity document.

    A blob for the same reason ``classes.json`` is one: identity is part of the data and
    travels with the image set as a file. ``register_dataset`` writes it compare-and-set
    against the version it read, so the id is minted once even when two registrations race,
    and encodes it through the canonical ``RECORD_JSON`` codec.
    """
    return Key(DATASET_IDENTITY_STORE, str(dataset_root), _DATASET_IDENTITY_PARTS)


def decode_dataset_identity_document(data: bytes, *, dataset_root: str | Path) -> dict:
    """A dataset identity document's bytes, decoded and shape/version-checked, whatever its
    ``fingerprint`` states: the raw layer :func:`decode_dataset_identity` (the general reader,
    which also refuses a bare pre-prefix fingerprint) builds on, and ``register_dataset``'s own
    re-register read calls directly. Re-registering is the fix for a bare fingerprint, so that
    read (which only ever preserves the minted ``id`` across the rewrite) must not itself refuse
    on the very value it is about to overwrite.

    Raises ``ValueError`` for bytes that do not decode, or that decode to something other than a
    dict carrying an ``id``. Propagates :class:`tcip_store.SchemaVersionRefused`, uncaught, for a
    ``schema_version`` this reader does not accept: a policy fact about a newer writer, never the
    same fact as a malformed document, so a caller that tolerates a plain ``ValueError`` as
    "nothing registered yet" must not fold this one in too.
    """
    try:
        identity = RECORD_JSON.decode(data)
    except ValueError as exc:
        raise ValueError(
            f"{dataset_identity_path(dataset_root)} exists but does not decode as a dataset "
            f"identity ({exc}); re-register with register_dataset") from exc
    if not isinstance(identity, dict) or not identity.get("id"):
        raise ValueError(
            f"{dataset_identity_path(dataset_root)} exists but does not decode as a dataset "
            "identity; re-register with register_dataset")
    check_schema_version(get_descriptor(DATASET_IDENTITY_STORE), identity)
    return identity


def decode_dataset_identity(data: bytes, *, dataset_root: str | Path) -> dict:
    """:func:`decode_dataset_identity_document`, plus a refusal for a non-null ``fingerprint``
    naming no formula version (a bare value from before the ``v<n>:`` prefix existed):
    re-registering through ``register_dataset`` is the remedy, never a reader that admits the
    bare value as the dataset's current identity. A null ``fingerprint`` (a dataset with no
    images or labels) is not this case. :func:`require_dataset_identity` is the one caller; a
    re-register read that must see the document's fingerprint whatever it states uses the raw
    layer instead.
    """
    from tcip_mcp.pipelines.data.dataset_fingerprint import fingerprint_formula_version

    identity = decode_dataset_identity_document(data, dataset_root=dataset_root)
    fingerprint = identity.get("fingerprint")
    if fingerprint is not None and fingerprint_formula_version(fingerprint) is None:
        raise ValueError(
            f"{dataset_identity_path(dataset_root)} carries a fingerprint {fingerprint!r} that "
            "names no formula version; re-register this dataset through register_dataset to "
            "bring it to the current shape")
    return identity


def _read_dataset_identity(dataset_root: str | Path, *, decode) -> dict:
    """The shared absence-check both :func:`require_dataset_identity` and
    :func:`read_dataset_identity_document` build on, differing only in which decoder checks the
    present document.

    Read through the store (``tcip_store.read_blob_versioned``), never a bare file check, which
    the database backend would fail: a directory that merely ends in ``images`` is not a dataset
    until ``register_dataset`` has minted an identity for it.
    """
    import tcip_store

    stored = tcip_store.read_blob_versioned(dataset_identity_key(dataset_root), default=None)
    if stored.value is None:
        raise ValueError(
            f"{dataset_root} carries no dataset identity record "
            f"({dataset_identity_path(dataset_root)} absent); register it first with "
            "register_dataset")
    return decode(stored.value, dataset_root=dataset_root)


def require_dataset_identity(dataset_root: str | Path) -> dict:
    """The dataset's identity record (``{crop, id, fingerprint}``), or the refusal naming
    ``register_dataset`` when it is absent.

    A present document's decode, shape and version are checked through
    :func:`decode_dataset_identity`, which also refuses a non-null ``fingerprint`` naming no
    formula version; its :class:`tcip_store.SchemaVersionRefused` propagates uncaught,
    distinguishable from the plain ``ValueError`` this function itself raises for absence or a
    malformed document. :func:`read_dataset_identity_document` is the raw counterpart for a
    caller whose job is fixing that very fingerprint.
    """
    return _read_dataset_identity(dataset_root, decode=decode_dataset_identity)


def read_dataset_identity_document(dataset_root: str | Path) -> dict:
    """:func:`require_dataset_identity`, but through :func:`decode_dataset_identity_document`:
    the identity record whatever its ``fingerprint`` states, never refusing on a bare pre-prefix
    value. For a caller re-registering through ``register_dataset`` (fixing that very value) and
    ``scripts/check_dataset_identity.py`` (diagnosing it), whose jobs are seeing the value
    rather than being refused before either can act on it.
    """
    return _read_dataset_identity(dataset_root, decode=decode_dataset_identity_document)


def image_status_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/.tcip/state/image_status.json``: the confirmed-negatives store.

    Shape: ``{bucket: {image_name: {status, recorded_by, recorded_at}}}``, bucket via
    :func:`status_bucket`. Each record says who set the status and when, so a person's Complete and
    a status a harvest wrote are distinguishable here rather than only in an audit log.

    Sibling of ``classes_path``/``dataset_identity_path``: a Complete is a fact about the dataset's
    content (what actually trains), so it travels with the dataset rather than living in whichever
    project's private ``.tcip/`` happens to be an ancestor. The single locator every writer
    (the GUI's review flow, ``materialize_dataset``, ``draw_splits``) and every reader
    (``confirmed_negative_names``, ``doctor.py``) must call; never reconstruct this path locally.
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
    """The dataset's confirmed-negative store.

    ``cas``: the GUI's confirmation routes read the whole document, set one image's status
    under one bucket and write it back, from a different process than the agent's own
    writers, so an unconditional write drops confirmations nobody re-reviewed.
    """
    return Key(IMAGE_STATUS_STORE, str(dataset_root), _IMAGE_STATUS_PARTS)


def read_image_status_store(dataset_root: str | Path) -> dict:
    """The dataset's stored image statuses as written, or ``{}`` when there is nothing to read.

    The one read behind every confirmed-negative reader, so an enumeration of a subject's buckets,
    the records taken from them, and a fingerprint over them all come from the same document and
    give the same answer to a store that is absent or will not decode. Reading the bytes off
    :func:`image_status_path` instead answers "no confirmations" whenever the backend holds them
    somewhere other than that file, which trains a human's confirmed negatives as unlabelled and
    silently drops them from a dataset's content identity.

    The shape guards (:func:`status_confirmations`, :func:`normalize_status_store`) stay the
    caller's, so a reader that needs the records whole and one that needs only the tokens still
    project the same raw document rather than a pre-narrowed view.
    """
    try:
        raw = tcip_store.read(image_status_key(dataset_root), default={})
    except DecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def view_coverage_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/.tcip/state/view_coverage.json``: per-image record of two per-cell facts,
    the reference-grid cells the GUI has served at native resolution (a delivery fact) and, per
    cell, the tightest scale bound at which every one of its sub-cells has sat fully on screen
    (``cells_seen_at_scale``). Neither is a claim about what the breeder examined; whether a seen
    cell counts as swept is derived in the browser against a subject's working-scale bar, never
    stored here.

    Shape: ``{bucket: {image_name: record}}``, bucket via :func:`status_bucket`; the record's own
    shape is declared once, by the web layer's coverage models (``CoverageRecord``: ``grid``,
    ``cells_served_at_native``, ``cells_seen_at_scale``, ``viewing``, ``updated_at``), with
    ``viewing`` the display context that layer's own ``CoverageViewing`` declares (bands, stretch,
    stats_source, display_bounds, base_served_size). Each record carries the grid geometry it was
    accumulated against, so a derivation change can never silently misread an old cell list.
    Advisory only: training never reads this store, and a Complete with unswept cells warns in the
    GUI rather than blocks. The negative definition (:func:`image_status_path`) is untouched by
    anything recorded here.
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
    """The dataset's per-image view-coverage record.

    ``cas``: the coverage route accumulates cells into an existing record under a lock, so
    an unconditional write would drop the cells another writer had just added.
    """
    return Key(VIEW_COVERAGE_STORE, str(dataset_root), _VIEW_COVERAGE_PARTS)


def image_status_digest_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/.tcip/state/image_status_digest.json``: ``{bucket: {image_name: digest}}``.

    Sibling of :func:`image_status_path`, stamped by the same writers at confirmation time with the
    subject's attribute-schema digest in effect (:func:`tcip_mcp.class_registry.attribute_schema_digest`).
    Stamped per image, not per bucket: a bucket holds every image a human has ever touched under
    one subject/date, so a bucket-wide stamp would be silently overwritten by the next unrelated
    write to that bucket, un-quarantining a stale confirmation nobody re-reviewed. Lets a reader tell
    a confirmation made under a since-changed attribute schema from one still valid, see
    ``confirmed_negative_names``'s quarantine logic. Absence of a stamp is not evidence of staleness
    (a rail must admit valid work, not only reject it): only a stamp that positively disagrees with
    the current schema is grounds to quarantine that one image.
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
    """The schema stamps beside the confirmed-negative store.

    ``cas``: each stamp is merged into the existing document under a lock, one image at a
    time, so an unconditional write un-quarantines stamps another writer had just set.
    """
    return Key(IMAGE_STATUS_DIGEST_STORE, str(dataset_root), _IMAGE_STATUS_DIGEST_PARTS)


def region_completeness_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/.tcip/state/region_completeness.json``: per-subject attestations that
    every instance of a subject has been found within a reference-grid region's cells.

    Sibling of :func:`image_status_path`: an attestation is a fact about the dataset's content
    (what a block-calibration split may treat as fully labeled), so it travels with the dataset
    rather than living in whichever project's private ``.tcip/`` happens to be an ancestor.
    Keyed by :func:`status_bucket` with the raster's own stem standing in for ``date`` (see
    :func:`normalize_region_completeness_store`): unlike ``image_status.json``, a raster's
    completeness is one record per bucket, not one per image name, since the stem already
    identifies exactly one raster and a date directory can hold many.
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
    """The dataset's region-completeness attestations.

    ``cas``: the coverage route adds or removes one cell inside an existing bucket's record
    under a lock, so an unconditional write drops the cells another writer just attested.
    """
    return Key(REGION_COMPLETENESS_STORE, str(dataset_root), _REGION_COMPLETENESS_PARTS)


def region_completeness_digest_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/.tcip/state/region_completeness_digest.json``: ``{bucket: {cell_name:
    digest}}``.

    Sibling of :func:`region_completeness_path`, stamped at attestation time with a content
    digest of the subject's annotations found inside that cell (see
    :mod:`tcip_mcp.pipelines.region_completeness`). Stamped per cell, not per bucket: a bucket
    accumulates every cell a human has ever attested complete under one subject/raster, so a
    bucket-wide stamp would be silently overwritten by the next cell's attestation, un-quarantining
    an earlier cell's stale attestation nobody re-reviewed (the same reasoning
    :func:`image_status_digest_path` states for its own per-image stamping). Lets a reader tell an
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
    """``<dataset_root>/.tcip/state/coverage_grid_zoom.json``: the breeder-set inspection zoom
    the coverage lattice's cell size is derived from, one entry per subject.

    Shape: ``{subject: {zoom, set_by, set_at}}``. Advisory, like :func:`view_coverage_path`: no
    default zoom exists, and a subject absent from this store simply has no coverage lattice yet.
    Sibling of ``view_coverage_path``: it travels with the dataset rather than a project's own
    ``.tcip/``, since a lattice zoom is a fact about how this dataset's imagery is inspected, not
    about any one project's session.
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
    """The dataset's per-subject coverage-lattice zoom.

    ``cas``: the grid-zoom route sets one subject's entry inside an existing document under a
    lock, so an unconditional write would drop another subject's zoom set moments earlier.
    """
    return Key(COVERAGE_GRID_ZOOM_STORE, str(dataset_root), _COVERAGE_GRID_ZOOM_PARTS)


def region_completeness_digest_key(dataset_root: str | Path) -> Key:
    """The content stamps beside the region-completeness attestations.

    ``cas``: each cell's stamp is merged into the existing document under a lock, so an
    unconditional write revives an earlier cell's stale stamp.
    """
    return Key(REGION_COMPLETENESS_DIGEST_STORE, str(dataset_root),
               _REGION_COMPLETENESS_DIGEST_PARTS)


def _is_completeness_record(value: object) -> bool:
    """Whether ``value`` is a region-completeness bucket record: a dict with a dict ``grid`` and a
    ``cells_complete`` list of strings.

    The one recognizer :func:`normalize_region_completeness_store` and
    :func:`unreadable_completeness_entries` both call, so an entry is never kept by one and
    reported unreadable by the other.
    """
    if not isinstance(value, dict) or not isinstance(value.get("grid"), dict):
        return False
    cells = value.get("cells_complete")
    return isinstance(cells, list) and all(isinstance(c, str) for c in cells)


def normalize_region_completeness_store(raw: object) -> dict[str, dict]:
    """``{bucket: {grid, cells_complete, attested_by, attested_at, stem, date, subject}}``: a
    shape guard, shared by every reader.

    A bucket is ``status_bucket(subject, stem)``: the raster's own stem stands in for
    ``image_status.json``'s ``date`` slot, since one raster's completeness is one record, not one
    per image name (contrast :func:`normalize_status_store`, which nests by image name because a
    date bucket can hold many images). An entry :func:`_is_completeness_record` does not recognize
    is not a completeness record and is dropped, so a malformed store yields no attestations rather
    than a wrong one. A caller about to merge a write into this store asks
    :func:`unreadable_completeness_entries` first, so an unrecognized bucket entry is never one a
    merge silently deletes; a raw document that is not a dict at all is a separate case that
    function does not report (there is no bucket to name), and a merging writer must refuse on
    that shape itself rather than let this function's own empty return read as an empty store.
    """
    if not isinstance(raw, dict):
        return {}
    return {key: value for key, value in raw.items() if _is_completeness_record(value)}


def unreadable_completeness_entries(raw: object) -> list[str]:
    """Bucket names in ``raw`` whose value :func:`normalize_region_completeness_store` does not
    recognize as a completeness record.

    What a read drops and a merging write would therefore delete. Mirrors
    :func:`unreadable_status_entries` for this store; unlike that one, a region-completeness store
    is flat (one record per bucket, not nested by image name), so the entries reported here are
    bare bucket names rather than ``bucket/name`` pairs.
    """
    if not isinstance(raw, dict):
        return []
    return sorted(key for key, value in raw.items() if not _is_completeness_record(value))


def status_bucket(subject: str, date: Optional[str]) -> str:
    """The ``image_status.json`` key a confirmation belongs under.

    Scoped by subject and date but not task: a Complete covers detect and segment together,
    which is how ``derive_image_status`` already evaluates them. A store keyed by image name alone
    re-applies one subject's confirmations to every other subject. ``subject`` must be a real subject
    (callers supply it); there is no catch-all default; a bush image confirmed empty of one
    subject is still positive for another.
    """
    return f"{subject}/{date}" if date else subject


def bucket_subject_date(bucket: str) -> tuple[str, Optional[str]]:
    """The ``(subject, date)`` a bucket key was built from: the declared inverse of
    :func:`status_bucket`, so a reader that has to take a key apart never re-derives the
    separator the writer used."""
    subject, _, date = bucket.partition("/")
    return subject, (date or None)


def status_of(record: object) -> Optional[str]:
    """The status token a stored record holds, or ``None`` when the value is not one.

    A record is ``{"status", "recorded_by", "recorded_at"}``, all three non-empty strings: a stored
    status says what it is, who set it and when, and a value missing any of those cannot be told
    apart from one a function wrote unattributed. One predicate, so the readers and the writers
    agree on what the store holds.
    """
    if not isinstance(record, Mapping):
        return None
    status = record.get("status")
    fields = (status, record.get("recorded_by"), record.get("recorded_at"))
    if all(isinstance(v, str) and v for v in fields):
        return str(status)
    return None


def status_records(
    statuses: Mapping[str, str], *, recorded_by: str, recorded_at: Optional[str] = None
) -> dict[str, dict[str, str]]:
    """One bucket's ``{image_name: status}`` as stored records, attributed to ``recorded_by``.

    ``recorded_by`` names the actor the status came from under the platform's identity convention
    (:func:`tcip_mcp.identity.user_identity` for a person, a bare name for a tool producer), so a
    reader can tell a human's Complete from a status a function wrote without a second store to
    consult. ``recorded_at`` defaults to the moment of this call, one timestamp across the names in
    it. Refuses an unattributed write rather than recording a status nobody answers for.
    """
    if not (recorded_by or "").strip():
        raise ValueError(
            "an image status records who set it, so recorded_by is required; pass the person's "
            "user:<name> identity or the writing tool's own name"
        )
    at = recorded_at or datetime.now(timezone.utc).isoformat()
    return {name: {"status": status, "recorded_by": recorded_by, "recorded_at": at}
            for name, status in statuses.items()}


def status_confirmations(raw: object) -> dict[str, dict[str, dict[str, str]]]:
    """``{bucket: {image_name: record}}``: the stored records whole, attribution included.

    A bucket is ``status_bucket(subject, date)``. The shape guard every reader that needs to know
    who recorded a status goes through; :func:`normalize_status_store` is its status-token
    projection. Anything :func:`status_of` does not recognize is not a recorded status and is
    dropped, so a malformed store yields no confirmations rather than a wrong one.
    """
    if not isinstance(raw, dict):
        return {}
    return {
        key: {k: dict(v) for k, v in value.items() if status_of(v) is not None}
        for key, value in raw.items() if isinstance(value, dict)
    }


def normalize_status_store(raw: object) -> dict[str, dict[str, str]]:
    """``{bucket: {image_name: status}}``: a shape guard, shared by every reader.

    The status-token projection of :func:`status_confirmations`, which it calls rather than
    re-deriving what counts as a stored status, for readers that decide admission and never ask
    who recorded it.
    """
    return {bucket: {name: str(record["status"]) for name, record in records.items()}
            for bucket, records in status_confirmations(raw).items()}


CONFIRMED_NEGATIVE = "negative"
"""The status token for an image a human marked done with none of the subject on it."""

IMAGE_STATUSES = ("complete", "partial", CONFIRMED_NEGATIVE, "unannotated")
"""Every status the store holds, and the only values a write may record.

``"complete"`` and ``"negative"`` are opposites, not degrees: both mean the human finished the
image, and they differ on whether anything of the subject is on it. Anything reading ``"complete"``
as a confirmed negative trains populated images as empty.
"""


def derive_status(*, completed: bool, has_content: bool) -> str:
    """The status one image holds for one subject, from the human's Complete and what is labeled.

    ``has_content`` is whether the image carries any annotation of the subject in question; the
    caller scopes it, since only the caller knows which subject it is asking about. A negative is
    intentional, never a side effect of an empty file: it takes the Complete, which is why an
    uncompleted empty image is ``"unannotated"`` rather than a negative.
    """
    if completed:
        return "complete" if has_content else CONFIRMED_NEGATIVE
    return "partial" if has_content else "unannotated"


def annotations_hold_subject(annotations: Iterable, subject: str) -> bool:
    """Whether any of ``annotations`` (as :func:`tcip_annotation.json_io.read_annotations` returns
    them) names ``subject``, geometry or not.

    An image-level record (a subject with no geometry) counts as content for that subject, the
    same rule :func:`derive_status` and its callers already apply to a whole label file.
    """
    return any(a.subject == subject for a in annotations)


def is_confirmed_negative(status: object) -> bool:
    """Whether a stored status is a human's confirmation that the image holds none of the subject.

    One predicate, so no reader can widen it to include ``"complete"``, which is its opposite.
    """
    return status == CONFIRMED_NEGATIVE


def confirmed_negative_names_any_subject(by_bucket: Mapping[str, Mapping[str, str]]) -> set[str]:
    """Every image file name confirmed negative for some subject, in any date bucket.

    An empty label file names no subject of its own to scope the question by, so the check that
    decides whether to flag it (rather than treat it as a confirmed negative) spans every bucket
    the store holds, unlike the per-subject checks elsewhere in this module. Takes ``by_bucket``
    already in :func:`normalize_status_store`'s shape rather than a root to read, so every reader
    of the confirmed-negative store, the doctor and the data-quality validator alike, shares this
    one rule over whatever ``by_bucket`` it read rather than each deciding admission on its own.
    """
    return {name for bucket in by_bucket.values() for name, status in bucket.items()
            if is_confirmed_negative(status)}


def _require_known_statuses(bucket: str, statuses: Iterable[Optional[str]]) -> None:
    """Refuse a status outside :data:`IMAGE_STATUSES`, or a value that is not a stored record.

    A token no reader understands is neither a confirmation nor a negative, and an unattributed
    value is one no reader can tell from a status a function wrote, so neither reaches the store.
    """
    unknown = sorted({"<unattributed>" if s is None else s for s in statuses}
                     - set(IMAGE_STATUSES))
    if unknown:
        raise ValueError(
            f"image status must be one of {IMAGE_STATUSES}, recorded with who set it and when; "
            f"refusing to record {unknown} for {bucket!r}"
        )


def unreadable_status_entries(raw: object) -> list[str]:
    """``bucket/image_name`` for every stored value :func:`status_of` does not recognize.

    What a read drops and a merging write would therefore delete. A caller that is about to
    rewrite the document asks first, so a store holding statuses in some other shape is reported
    rather than quietly emptied of them.
    """
    if not isinstance(raw, dict):
        return []
    return sorted(f"{bucket}/{name}"
                  for bucket, value in raw.items() if isinstance(value, dict)
                  for name, record in value.items() if status_of(record) is None)


def record_image_statuses(
    dataset_root: str | Path, bucket: str, statuses: Mapping[str, str], *, recorded_by: str
) -> None:
    """Merge one bucket's per-image statuses into the dataset's confirmed-negative store.

    Merged, never replaced: a bucket is one subject on one date, and a write for one of them must
    leave every other subject's and date's confirmations exactly as they were. ``recorded_by`` is
    the actor this write is on behalf of, stamped onto each record by :func:`status_records`.

    Refuses when the document already holds entries this reader cannot recognize, rather than
    rewriting it without them: a merge reads the whole document and writes it back, so entries a
    read drops are entries the write deletes, and a human's statuses are not something to lose to
    an unrelated confirmation.
    """
    _require_known_statuses(bucket, statuses.values())
    records = status_records(statuses, recorded_by=recorded_by)
    key = image_status_key(dataset_root)
    with tcip_store.transaction(key) as txn:
        raw = txn.read(key, default={})
        unreadable = unreadable_status_entries(raw)
        if unreadable:
            raise ValueError(
                f"the image status store under {dataset_root} holds {len(unreadable)} entries in a "
                f"shape this reader does not recognize, starting with {unreadable[:3]}; merging a "
                f"write into it would delete them. Conform the store to the recorded-status shape "
                f"({{status, recorded_by, recorded_at}}) first"
            )
        store = status_confirmations(raw)
        store.setdefault(bucket, {}).update(records)
        txn.write(key, {k: dict(sorted(store[k].items())) for k in sorted(store)})


def replace_image_status_store(
    dataset_root: str | Path, records_by_bucket: Mapping[str, Mapping[str, Mapping[str, str]]]
) -> None:
    """Write the whole confirmed-negative store for a dataset this call is producing.

    For a materializer that is authoring an output dataset's negatives outright (a split tree, a
    curated review dataset): what it writes is the complete set for that output, so a leftover
    entry from an earlier materialization into the same directory must not survive as a negative
    nobody re-derived.

    Takes whole records, not bare tokens: a call authoring new confirmations builds them with
    :func:`status_records`, and a call copying another dataset's confirmations passes that
    dataset's own records through unchanged, so who confirmed an image survives the copy instead
    of being re-attributed to whatever wrote it last.
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

    Stamped per image and merged, for the reason :func:`image_status_digest_path` states: a
    bucket-wide or whole-document write would drop another image's stamp and un-quarantine a
    confirmation made under a since-changed schema that nobody re-reviewed.

    ``only_unstamped`` leaves an image that already carries a stamp exactly as it is, for a caller
    recording after the fact which schema a confirmation was made under: the stamp the writer set at
    confirmation time is the direct evidence, and overwriting it with a later reconstruction would
    re-date a confirmation nobody re-reviewed. The read and the write share one transaction, so a
    stamp landing between them is seen rather than clobbered.
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


def label_filename(stem: str, fmt: str = "json") -> str:
    """The file name one image's label or prediction record is written under.

    The rule an image stem becomes a record name by, stated once: a consumer holding a directory
    this module did not hand it (a materialized split's ``labels/``) still asks here for the name
    rather than re-asserting the extension.
    """
    return f"{stem}{label_ext(fmt)}"


def annotation_path(
    dataset_root: str | Path,
    date: Optional[str],
    stem: str,
    fmt: str = "json",
) -> Path:
    return annotation_dir(dataset_root, date) / label_filename(stem, fmt)


def annotation_path_for_image(
    image_path: str | Path,
    fmt: str = "json",
    *,
    date: Optional[str] = None,
) -> Path:
    """Canonical write path for an image's single label file (date derived from the image path)."""
    root, img_date, stem = parse_image_path(image_path)
    return annotation_path(root, date if date is not None else img_date, stem, fmt)


def prediction_path(
    dataset_root: str | Path,
    model: Optional[str],
    date: Optional[str],
    stem: str,
    fmt: str = "json",
) -> Path:
    return prediction_dir(dataset_root, model, date) / label_filename(stem, fmt)


def list_subjects(dataset_root: str | Path) -> list[str]:
    """The dataset's subjects, in the registry's declared order (delegated to ``class_registry``;
    this module never parses ``classes.json`` itself). ``[]`` when there is no registry.

    A registry that is present but unreadable raises rather than reading as no subjects: every
    name-based label under it is undecodable without it, so absence and corruption are
    different answers here.
    """
    from tcip_mcp import class_registry

    cp = classes_path(dataset_root)
    if not cp.is_file():
        return []
    try:
        registry = class_registry.read_registry(cp)
    except OSError:
        return []
    return [s.name for s in registry.subjects]


def subjects_on_date(
    dataset_root: str | Path, date: Optional[str], *, reader: Optional[Callable] = None,
) -> list[str]:
    """Distinct subjects that actually appear in the per-image label files on ``date``: the one
    per-date label scan, shared by ``subjects_with_labels`` and the GUI's subject selector.

    Enumerates ``annotations/<date>/`` through ``json_io.prediction_documents`` and reads each
    document through ``reader`` (``json_io.read_annotations`` by default; a caller with its own
    memoized reader, keyed by path plus mtime and size, passes it here instead of parsing the
    same files twice), so the subjects offered are the subjects genuinely labeled there,
    sorted. Raises
    :class:`~tcip_annotation.json_io.UnreadableLabelDocument` when a present label file on this
    date will not read; a missing ``annotations/<date>/`` directory reads as no subjects. A label
    whose filename is a bucket's own provenance stamp (``json_io.is_sidecar_name``) is neither read
    nor raised over: it is excluded from the walk entirely, the same as everywhere a prediction
    bucket is enumerated. Such a file is a data-state fact the doctor reports, not one this scan
    surfaces.
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
    """Subjects that actually have ≥1 label on ``date``: what the GUI's subject selector offers.

    A subject the registry declares but that is unlabeled on the selected date lands the user on an
    empty canvas, so the selector is sourced from the labels present, not the registry.
    """
    return subjects_on_date(dataset_root, date, reader=reader)


def list_models(dataset_root: str | Path) -> list[str]:
    """Sorted model bucket names under ``predictions/``. A dot-prefixed directory is never a
    bucket (see ``is_bucket_name``) and is excluded, the same grammar ``list_dates`` applies."""
    preds = prediction_root(dataset_root)
    if not preds.is_dir():
        return []
    return sorted(p.name for p in preds.iterdir() if p.is_dir() and is_bucket_name(p.name))


def prediction_bucket_dirs(dataset_root: str | Path) -> list[Path]:
    """Every directory under ``predictions/`` a bucket's own sidecar could sit in: each model's
    own directory, and each of its date subdirectories, whether or not either actually holds one.

    The one walk ``doctor.py``'s registry check and ``scripts/_store_bootstrap.py``'s
    ``project_roots`` both read through, so a directory one calls a bucket is a directory the
    other calls one too.
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
    return found


def _dir_has_label_file(d: Path) -> bool:
    """True if ``d`` holds at least one label file (any supported extension).

    An *empty* label file counts here: this answers "was anything written on this date", which is
    what the GUI's selectors need. It does not mean the image is a confirmed negative: that
    requires a human Complete recorded in ``.tcip/state/image_status.json``, and training reads only
    that (``confirmed_negative_names``).
    """
    if not d.is_dir():
        return False
    return any(p.is_file() and p.suffix in _ANY_EXTS for p in d.iterdir())


def models_with_predictions(dataset_root: str | Path, date: Optional[str]) -> list[str]:
    """Models that actually have ≥1 prediction file on ``date`` (sorted, same order as
    ``list_models``). A model with no predictions on the selected date has nothing to overlay."""
    root = Path(dataset_root)
    return [
        model
        for model in list_models(root)
        if _dir_has_label_file(prediction_dir(root, model, date))
    ]


def find_gt_label(
    image_path: str | Path,
    *,
    date: Optional[str] = None,
    fmt: Optional[str] = None,
) -> Optional[Path]:
    """Find the existing ground-truth label file for an image (read-time resolver).

    One file per image, so this resolves ``annotations/<date>/<stem>.json`` directly. Returns
    the file, or ``None``. If ``fmt`` is given only that extension is considered, else any supported.
    """
    root, img_date, stem = parse_image_path(image_path)
    d = date if date is not None else img_date
    exts = [label_ext(fmt)] if fmt else list(_ANY_EXTS)
    adir = annotation_dir(root, d)
    for e in exts:
        cand = adir / f"{stem}{e}"
        if cand.is_file():
            return cand
    return None


def find_prediction(
    image_path: str | Path,
    *,
    model: Optional[str] = None,
    date: Optional[str] = None,
    fmt: Optional[str] = None,
) -> Optional[Path]:
    """Find an existing prediction file for an image: a specific ``model`` if given, else every
    model. Returns the first existing file, or ``None``."""
    root, img_date, stem = parse_image_path(image_path)
    d = date if date is not None else img_date
    exts = [label_ext(fmt)] if fmt else list(_ANY_EXTS)

    models = [model] if model else list_models(root)
    for m in models:
        pdir = prediction_dir(root, m, d)
        for e in exts:
            cand = pdir / f"{stem}{e}"
            if cand.is_file():
                return cand
    return None
