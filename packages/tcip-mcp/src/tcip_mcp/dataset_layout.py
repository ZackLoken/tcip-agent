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

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Optional

import tcip_store
from tcip_store import Key, StoreDescriptor, json_codec, register_store
from tcip_store.file_backend import RootedFileLocator

# Per-image JSON is the canonical on-disk label format; ``coco`` is the assembled dataset view of it.
LABEL_EXT = {"json": ".json", "coco": ".json"}
_ANY_EXTS = (".json",)
DEFAULT_MODEL = "live"
#: Geometry kinds a task authors, kept as a selector, not a label-path segment.
TASKS = ("detect", "segment")
#: Split names ``make_splits(materialize=True)`` emits.
SPLIT_NAMES = ("train", "val", "test")
CLASSES_FILENAME = "classes.json"


def label_ext(fmt: Optional[str]) -> str:
    """File extension for a label format: always ``.json``; both formats are JSON on disk."""
    return LABEL_EXT.get((fmt or "json").lower(), ".json")


# ── the dataset-root stores ──────────────────────────────────────────────────

_STATE_DOC = RootedFileLocator(prefix=(".tcip", "state"), suffix=".json")
"""A dataset's own private state documents, one of each per dataset."""

_DATASET_DOC = RootedFileLocator(suffix=".json")
"""The documents that travel with the image set, at the dataset root itself."""

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
    return Path(dataset_root, "images")


def image_dir(dataset_root: str | Path, date: Optional[str]) -> Path:
    """``<dataset_root>/images/[<date>/]``: where an image's bytes live."""
    return image_root(dataset_root).joinpath(*_date_seg(date))


def image_path(dataset_root: str | Path, date: Optional[str], stem: str, ext: str) -> Path:
    """Canonical write path for an image (``ext`` includes the leading dot)."""
    return image_dir(dataset_root, date) / f"{stem}{ext}"


def list_dates(dataset_root: str | Path) -> list[str]:
    """Sorted capture-date bucket names under ``images/`` (ISO ``YYYY-MM-DD``)."""
    imgs = image_root(dataset_root)
    if not imgs.is_dir():
        return []
    return sorted(p.name for p in imgs.iterdir() if p.is_dir())


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
    if rest and rest[-1].endswith(".json"):
        rest = rest[:-1]
    return rest[0] if len(rest) == 1 else None


#: The top-level segments under a dataset root; a path under any of them locates the root.
#: ``labels`` covers a split-materialized tree (``make_splits(materialize=True)`` writes
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
        kind="record",
        key_fields=("document",),
        codec=json_codec(default=None, trailing_newline=True),
        concurrency="last_writer_wins",
        locator=_DATASET_DOC,
    )
)


def class_registry_key(dataset_root: str | Path) -> Key:
    """The dataset's class registry.

    ``last_writer_wins``: ``class_registry.write_registry`` replaces the whole registry from a
    value its caller already holds, and no writer merges into the stored document.
    """
    return Key(CLASS_REGISTRY_STORE, str(dataset_root), _CLASS_REGISTRY_PARTS)


def dataset_identity_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/dataset.json``: the dataset's identity ({crop, id, fingerprint}).

    Sibling of ``classes.json``: identity is part of the data, so it travels with the image set. The
    stored fingerprint is a cache; recompute-on-read (``resolution.dataset_fingerprint``) is authority.
    """
    return _entry_path(_DATASET_DOC, dataset_root, _DATASET_IDENTITY_PARTS)


DATASET_IDENTITY_STORE = "dataset_identity"
_DATASET_IDENTITY_PARTS = _document_of("dataset.json")
register_store(
    StoreDescriptor(
        name=DATASET_IDENTITY_STORE,
        kind="record",
        key_fields=("document",),
        codec=json_codec(default=None, trailing_newline=True),
        concurrency="last_writer_wins",
        locator=_DATASET_DOC,
    )
)


def dataset_identity_key(dataset_root: str | Path) -> Key:
    """The dataset's identity document.

    ``last_writer_wins``: ``register_dataset`` reads the existing document only to keep the
    id it minted once, and every other field it writes is derived from the dataset's own
    content, so two writers racing produce the same document rather than losing an edit.
    """
    return Key(DATASET_IDENTITY_STORE, str(dataset_root), _DATASET_IDENTITY_PARTS)


def image_status_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/.tcip/state/image_status.json``: the confirmed-negatives store.

    Sibling of ``classes_path``/``dataset_identity_path``: a Complete is a fact about the dataset's
    content (what actually trains), so it travels with the dataset rather than living in whichever
    project's private ``.tcip/`` happens to be an ancestor. The single locator every writer
    (the GUI's review flow, ``materialize_dataset``, ``make_splits``) and every reader
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
        codec=json_codec(),
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


def view_coverage_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/.tcip/state/view_coverage.json``: per-image record of two per-cell facts,
    the reference-grid cells the GUI has served at native resolution (a delivery fact) and the
    cells swept in the viewport at or above the breeder's own working scale (a sweep fact).
    Neither is a claim about what the breeder examined.

    Shape: ``{bucket: {image_name: {grid, cells_served_at_native, cells_swept, viewing,
    updated_at}}}``, bucket via :func:`status_bucket`; ``viewing`` carries the display context
    including ``working_scale_bar`` as ``{value, source}``. Each record carries the grid geometry
    it was accumulated against, so a derivation change can never silently misread an old cell
    list. Advisory only: training never reads this store, and a Complete with unswept cells warns
    in the GUI rather than blocks. The negative definition (:func:`image_status_path`) is
    untouched by anything recorded here.
    """
    return _entry_path(_STATE_DOC, dataset_root, _VIEW_COVERAGE_PARTS)


VIEW_COVERAGE_STORE = "view_coverage"
_VIEW_COVERAGE_PARTS = _document_of("view_coverage.json")
register_store(
    StoreDescriptor(
        name=VIEW_COVERAGE_STORE,
        kind="record",
        key_fields=("document",),
        codec=json_codec(),
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
        codec=json_codec(),
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
        codec=json_codec(),
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
        codec=json_codec(),
        concurrency="cas",
        locator=_STATE_DOC,
    )
)


def region_completeness_digest_key(dataset_root: str | Path) -> Key:
    """The content stamps beside the region-completeness attestations.

    ``cas``: each cell's stamp is merged into the existing document under a lock, so an
    unconditional write revives an earlier cell's stale stamp.
    """
    return Key(REGION_COMPLETENESS_DIGEST_STORE, str(dataset_root),
               _REGION_COMPLETENESS_DIGEST_PARTS)


def normalize_region_completeness_store(raw: object) -> dict[str, dict]:
    """``{bucket: {grid, cells_complete, attested_by, attested_at, stem, date, subject}}``: a
    shape guard, shared by every reader.

    A bucket is ``status_bucket(subject, stem)``: the raster's own stem stands in for
    ``image_status.json``'s ``date`` slot, since one raster's completeness is one record, not one
    per image name (contrast :func:`normalize_status_store`, which nests by image name because a
    date bucket can hold many images). An entry missing a dict ``grid`` or a ``cells_complete``
    list of strings is not a completeness record and is dropped, so a malformed store yields no
    attestations rather than a wrong one.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict] = {}
    for key, value in raw.items():
        if not isinstance(value, dict) or not isinstance(value.get("grid"), dict):
            continue
        cells = value.get("cells_complete")
        if not isinstance(cells, list) or not all(isinstance(c, str) for c in cells):
            continue
        out[key] = value
    return out


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


def normalize_status_store(raw: object) -> dict[str, dict[str, str]]:
    """``{bucket: {image_name: status}}``: a shape guard, shared by every reader.

    A bucket is ``status_bucket(subject, date)``. Anything that is not a dict of strings is not a
    subject's confirmations and is ignored, so a malformed store yields no confirmations rather
    than a wrong one.
    """
    if not isinstance(raw, dict):
        return {}
    return {
        key: {k: v for k, v in value.items() if isinstance(v, str)}
        for key, value in raw.items() if isinstance(value, dict)
    }


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


def is_confirmed_negative(status: object) -> bool:
    """Whether a stored status is a human's confirmation that the image holds none of the subject.

    One predicate, so no reader can widen it to include ``"complete"``, which is its opposite.
    """
    return status == CONFIRMED_NEGATIVE


def record_image_statuses(
    dataset_root: str | Path, bucket: str, statuses: Mapping[str, str]
) -> None:
    """Merge one bucket's per-image statuses into the dataset's confirmed-negative store.

    Merged, never replaced: a bucket is one subject on one date, and a write for one of them must
    leave every other subject's and date's confirmations exactly as they were. Refuses a status
    outside :data:`IMAGE_STATUSES` rather than recording a token no reader understands.
    """
    unknown = sorted(set(statuses.values()) - set(IMAGE_STATUSES))
    if unknown:
        raise ValueError(
            f"image status must be one of {IMAGE_STATUSES}; refusing to record {unknown} for "
            f"{bucket!r}: a token no reader understands is neither a confirmation nor a negative"
        )
    key = image_status_key(dataset_root)
    with tcip_store.transaction(key) as txn:
        store = normalize_status_store(txn.read(key, default={}))
        store.setdefault(bucket, {}).update(statuses)
        txn.write(key, {k: dict(sorted(store[k].items())) for k in sorted(store)})


def replace_image_status_store(
    dataset_root: str | Path, statuses_by_bucket: Mapping[str, Mapping[str, str]]
) -> None:
    """Write the whole confirmed-negative store for a dataset this call is producing.

    For a materializer that is authoring an output dataset's negatives outright (a split tree, a
    curated review dataset): what it writes is the complete set for that output, so a leftover
    entry from an earlier materialization into the same directory must not survive as a negative
    nobody re-derived.
    """
    key = image_status_key(dataset_root)
    with tcip_store.transaction(key) as txn:
        txn.write(key, {k: dict(sorted(statuses_by_bucket[k].items()))
                        for k in sorted(statuses_by_bucket)})


def stamp_image_status_digests(
    dataset_root: str | Path, bucket: str, image_names: Iterable[str], digest: str
) -> None:
    """Record ``digest`` against each of ``image_names`` in ``bucket``, merging into what is there.

    Stamped per image and merged, for the reason :func:`image_status_digest_path` states: a
    bucket-wide or whole-document write would drop another image's stamp and un-quarantine a
    confirmation made under a since-changed schema that nobody re-reviewed.
    """
    key = image_status_digest_key(dataset_root)
    with tcip_store.transaction(key) as txn:
        stamps = txn.read(key, default={})
        if not isinstance(stamps, dict):
            stamps = {}
        bucket_stamps = stamps.get(bucket)
        if not isinstance(bucket_stamps, dict):
            bucket_stamps = {}
        for name in image_names:
            bucket_stamps[name] = digest
        stamps[bucket] = dict(sorted(bucket_stamps.items()))
        txn.write(key, dict(sorted(stamps.items())))


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


LABELS_STORE = "labels"
register_store(
    StoreDescriptor(
        name=LABELS_STORE,
        kind="blob",
        key_fields=("date", "stem"),
        enumerable=True,
        path_readable=True,
        locator=_LABEL_TREE,
    )
)


def label_key(dataset_root: str | Path, date: str, stem: str) -> Key:
    """One image's ground-truth labels, every subject's records in one document.

    A blob: the labels are the breeder's own data and travel with the image set under any
    backend, and their version is the hash of the document ``json_io`` encodes, which is what
    lets the GUI's load-edit-save pair compare and set instead of checking and hoping.

    ``date`` is required. A dataset whose images are not date-nested has labels this key
    cannot address, and which layout segment stands in for the date there is a question the
    storage design does not answer.
    """
    if not date:
        raise ValueError(
            f"label_key needs a capture date for {stem!r}: the undated dataset layout "
            f"({annotation_dir(dataset_root, None)}) has no key shape yet"
        )
    return Key(LABELS_STORE, str(dataset_root), (date, stem))


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


PREDICTIONS_STORE = "predictions"
register_store(
    StoreDescriptor(
        name=PREDICTIONS_STORE,
        kind="blob",
        key_fields=("model", "date", "stem"),
        enumerable=True,
        path_readable=True,
        locator=_PREDICTION_TREE,
    )
)


def prediction_key(dataset_root: str | Path, model: Optional[str], date: str, stem: str) -> Key:
    """One image's predictions inside one model bucket.

    A blob, on the same terms as :func:`label_key`: it is written whole from a run's own output
    and travels with the dataset. A bucket a human has reviewed is protected by
    ``prediction_buckets``, which redirects the run rather than replacing the file.

    ``date`` is required, on the same terms as :func:`label_key`.
    """
    if not date:
        raise ValueError(
            f"prediction_key needs a capture date for {stem!r}: the undated dataset layout "
            f"({prediction_dir(dataset_root, model, None)}) has no key shape yet"
        )
    return Key(PREDICTIONS_STORE, str(dataset_root), (model or DEFAULT_MODEL, date, stem))


def list_subjects(dataset_root: str | Path) -> list[str]:
    """The dataset's subjects, in the registry's declared order (delegated to ``class_registry``;
    this module never parses ``classes.json`` itself). ``[]`` when there is no registry."""
    from tcip_mcp import class_registry

    cp = classes_path(dataset_root)
    if not cp.is_file():
        return []
    try:
        registry = class_registry.read_registry(cp)
    except (OSError, class_registry.RegistryError):
        return []
    return [s.name for s in registry.subjects]


def subjects_on_date(dataset_root: str | Path, date: Optional[str]) -> list[str]:
    """Distinct subjects that actually appear in the per-image label files on ``date``: the one
    per-date label scan, shared by ``subjects_with_labels`` and the GUI's subject selector.

    Reads each ``annotations/<date>/<stem>.json`` through ``json_io.read_annotations`` (the single
    reader), so the subjects offered are the subjects genuinely labeled there, sorted."""
    from tcip_annotation import json_io

    d = annotation_dir(dataset_root, date)
    if not d.is_dir():
        return []
    found: set[str] = set()
    for f in d.glob("*.json"):
        for a in json_io.read_annotations(str(f)):
            found.add(a.subject)
    return sorted(found)


def subjects_with_labels(dataset_root: str | Path, date: Optional[str]) -> list[str]:
    """Subjects that actually have ≥1 label on ``date``: what the GUI's subject selector offers.

    A subject the registry declares but that is unlabeled on the selected date lands the user on an
    empty canvas, so the selector is sourced from the labels present, not the registry.
    """
    return subjects_on_date(dataset_root, date)


def list_models(dataset_root: str | Path) -> list[str]:
    preds = prediction_root(dataset_root)
    if not preds.is_dir():
        return []
    return sorted(p.name for p in preds.iterdir() if p.is_dir())


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
