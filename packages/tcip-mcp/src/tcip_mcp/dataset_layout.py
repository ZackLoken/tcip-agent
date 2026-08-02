"""Canonical dataset-layout resolver: the single source of truth for where an
image's ground-truth labels and model predictions live on disk.

Canonical layout (the label tree mirrors ``images/<date>/`` so stem-pairing is trivial and capture
dates never collide). Labels are **one file per image**, holding every subject's annotations by name;
the on-disk path carries no subject or task segment: those are properties of the records
inside the file, resolved through the dataset's single class registry::

    <dataset_root>/
        images/<date>/<stem>.<imgext>
        annotations/<date>/<stem>.json      # ground truth (all subjects for the image)
        predictions/<model>/<date>/<stem>.json   # model outputs
        classes.json                         # the nested registry: subjects -> attributes -> values

The class registry lives **in the dataset** and travels with the labels: a name-based label
(``subject``, attribute value) is undecodable without it. A second project opening the same image
set reads the same names. This module is a pure *locator*: it never parses ``classes.json`` (its
contents belong to :mod:`tcip_mcp.class_registry`); it only delegates to that module to list subjects.

``<date>`` of ``None`` (non-dated datasets) simply omits that segment. This is the single source of
truth for label/prediction locations: every producer and consumer resolves paths through here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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


def image_dir(dataset_root: str | Path, date: Optional[str]) -> Path:
    """``<dataset_root>/images/[<date>/]``: where an image's bytes live."""
    return Path(dataset_root, "images", *_date_seg(date))


def image_path(dataset_root: str | Path, date: Optional[str], stem: str, ext: str) -> Path:
    """Canonical write path for an image (``ext`` includes the leading dot)."""
    return image_dir(dataset_root, date) / f"{stem}{ext}"


def list_dates(dataset_root: str | Path) -> list[str]:
    """Sorted capture-date bucket names under ``images/`` (ISO ``YYYY-MM-DD``)."""
    imgs = Path(dataset_root) / "images"
    if not imgs.is_dir():
        return []
    return sorted(p.name for p in imgs.iterdir() if p.is_dir())


def annotation_dir(dataset_root: str | Path, date: Optional[str]) -> Path:
    """``<dataset_root>/annotations/[<date>/]`` (ground truth, one file per image, all subjects)."""
    return Path(dataset_root, "annotations", *_date_seg(date))


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
    return Path(dataset_root, CLASSES_FILENAME)


def dataset_identity_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/dataset.json``: the dataset's identity ({crop, id, fingerprint}).

    Sibling of ``classes.json``: identity is part of the data, so it travels with the image set. The
    stored fingerprint is a cache; recompute-on-read (``resolution.dataset_fingerprint``) is authority.
    """
    return Path(dataset_root, "dataset.json")


def image_status_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/.tcip/state/image_status.json``: the confirmed-negatives store.

    Sibling of ``classes_path``/``dataset_identity_path``: a Complete is a fact about the dataset's
    content (what actually trains), so it travels with the dataset rather than living in whichever
    project's private ``.tcip/`` happens to be an ancestor. The single locator every writer
    (the GUI's review flow, ``materialize_dataset``, ``make_splits``) and every reader
    (``confirmed_negative_names``, ``doctor.py``) must call; never reconstruct this path locally.
    """
    return Path(dataset_root, ".tcip", "state", "image_status.json")


def image_status_digest_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/.tcip/state/image_status_digest.json``: ``{bucket: {image_name: digest}}``.

    Sibling of :func:`image_status_path`, stamped by the same writers at confirmation time with the
    subject's attribute-schema digest in effect (:func:`tcip_mcp.class_registry.attribute_schema_digest`).
    Stamped **per image**, not per bucket: a bucket holds every image a human has ever touched under
    one subject/date, so a bucket-wide stamp would be silently overwritten by the next unrelated
    write to that bucket, un-quarantining a stale confirmation nobody re-reviewed. Lets a reader tell
    a confirmation made under a since-changed attribute schema from one still valid, see
    ``confirmed_negative_names``'s quarantine logic. Absence of a stamp is not evidence of staleness
    (a rail must admit valid work, not only reject it): only a stamp that positively disagrees with
    the current schema is grounds to quarantine that one image.
    """
    return Path(dataset_root, ".tcip", "state", "image_status_digest.json")


def status_bucket(subject: str, date: Optional[str]) -> str:
    """The ``image_status.json`` key a confirmation belongs under.

    Scoped by subject and date but not task: a Complete covers detect and segment together,
    which is how ``derive_image_status`` already evaluates them. A store keyed by image name alone
    re-applies one subject's confirmations to every other subject. ``subject`` must be a real subject
    (callers supply it); there is no catch-all default; a bush image confirmed empty of one
    subject is still positive for another.
    """
    return f"{subject}/{date}" if date else subject


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


def prediction_dir(dataset_root: str | Path, model: Optional[str], date: Optional[str]) -> Path:
    """``<dataset_root>/predictions/<model>/[<date>/]`` (model outputs, one file per image)."""
    return Path(dataset_root, "predictions", model or DEFAULT_MODEL, *_date_seg(date))


def annotation_path(
    dataset_root: str | Path,
    date: Optional[str],
    stem: str,
    fmt: str = "json",
) -> Path:
    return annotation_dir(dataset_root, date) / f"{stem}{label_ext(fmt)}"


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
    return prediction_dir(dataset_root, model, date) / f"{stem}{label_ext(fmt)}"


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
    preds = Path(dataset_root) / "predictions"
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
