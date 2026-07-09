"""Canonical dataset-layout resolver — the single source of truth for where an
image's ground-truth labels and model predictions live on disk.

Canonical layout (the label tree mirrors ``images/<date>/`` so stem-pairing is
trivial and capture dates never collide; role / trait / date / task are orthogonal
path segments and *class semantics live in ``classes.json``, never in filenames*)::

    <dataset_root>/
        images/<date>/<stem>.<imgext>
        annotations/<trait>/<date>/<task>/<stem>.<labelext>     # ground truth
        predictions/<model>/<date>/<task>/<stem>.<labelext>     # model outputs
        classes.json

``<trait>`` is the annotation campaign (the GUI's ``annotation_type``); ``<task>`` is
``detect`` | ``segment``. A ``<date>`` of ``None`` (non-dated datasets) simply omits
that segment.

This is the single source of truth for label/prediction locations — there is no
alternate ("flat") layout; every producer and consumer resolves paths through here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

LABEL_EXT = {"yolo": ".txt", "voc": ".xml", "coco": ".json", "labelme": ".json"}
_ANY_EXTS = (".txt", ".xml", ".json")
DEFAULT_TRAIT = "default"
DEFAULT_MODEL = "live"
TASKS = ("detect", "segment")


def label_ext(fmt: Optional[str]) -> str:
    """File extension for a label format (defaults to YOLO ``.txt``)."""
    return LABEL_EXT.get((fmt or "yolo").lower(), ".txt")


def parse_image_path(image_path: str | Path) -> tuple[Path, Optional[str], str]:
    """Return ``(dataset_root, date, stem)`` for an image path.

    Handles both canonical date-nested (``<root>/images/<date>/<stem>``) and flat
    (``<root>/images/<stem>``) images; ``date`` is ``None`` for the flat form.
    """
    img = Path(image_path)
    stem = img.stem
    parent = img.parent
    if parent.name == "images":
        return parent.parent, None, stem
    if parent.parent.name == "images":
        return parent.parent.parent, parent.name, stem
    # Unknown structure — best effort (treat the grandparent as the dataset root).
    return parent.parent, None, stem


def _date_seg(date: Optional[str]) -> tuple[str, ...]:
    return (date,) if date else ()


def image_dir(dataset_root: str | Path, date: Optional[str]) -> Path:
    """``<dataset_root>/images/[<date>/]`` — where an image's bytes live.

    ``date`` of ``None`` yields the flat ``images/`` form; a date value nests it
    under that bucket (canonical ISO ``YYYY-MM-DD`` capture-date buckets).
    """
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


def annotation_dir(dataset_root: str | Path, trait: Optional[str], date: Optional[str], task: str) -> Path:
    """``<dataset_root>/annotations/<trait>/[<date>/]<task>`` (ground truth)."""
    return Path(dataset_root, "annotations", trait or DEFAULT_TRAIT, *_date_seg(date), task)


def prediction_dir(dataset_root: str | Path, model: Optional[str], date: Optional[str], task: str) -> Path:
    """``<dataset_root>/predictions/<model>/[<date>/]<task>`` (model outputs)."""
    return Path(dataset_root, "predictions", model or DEFAULT_MODEL, *_date_seg(date), task)


def annotation_path(
    dataset_root: str | Path,
    trait: Optional[str],
    date: Optional[str],
    task: str,
    stem: str,
    fmt: str = "yolo",
) -> Path:
    return annotation_dir(dataset_root, trait, date, task) / f"{stem}{label_ext(fmt)}"


def annotation_path_for_image(
    image_path: str | Path,
    task: str,
    fmt: str = "yolo",
    *,
    trait: str = DEFAULT_TRAIT,
    date: Optional[str] = None,
) -> Path:
    """Canonical write path for an image's labels (date derived from the image path)."""
    root, img_date, stem = parse_image_path(image_path)
    return annotation_path(root, trait, date if date is not None else img_date, task, stem, fmt)


def prediction_path(
    dataset_root: str | Path,
    model: Optional[str],
    date: Optional[str],
    task: str,
    stem: str,
    fmt: str = "yolo",
) -> Path:
    return prediction_dir(dataset_root, model, date, task) / f"{stem}{label_ext(fmt)}"


def list_traits(dataset_root: str | Path) -> list[str]:
    ann = Path(dataset_root) / "annotations"
    if not ann.is_dir():
        return []
    return sorted(p.name for p in ann.iterdir() if p.is_dir())


def list_models(dataset_root: str | Path) -> list[str]:
    preds = Path(dataset_root) / "predictions"
    if not preds.is_dir():
        return []
    return sorted(p.name for p in preds.iterdir() if p.is_dir())


def find_gt_label(
    image_path: str | Path,
    task: str,
    *,
    trait: Optional[str] = None,
    date: Optional[str] = None,
    fmt: Optional[str] = None,
) -> Optional[Path]:
    """Find an existing ground-truth label file for an image (read-time resolver).

    Searches the canonical tree — a specific ``trait`` if given, else every trait
    campaign. Returns the first existing file, or ``None``. If ``fmt`` is given only
    that extension is considered, else any supported label extension.
    """
    root, img_date, stem = parse_image_path(image_path)
    d = date if date is not None else img_date
    exts = [label_ext(fmt)] if fmt else list(_ANY_EXTS)

    traits = [trait] if trait else list_traits(root)
    for t in traits:
        adir = annotation_dir(root, t, d, task)
        for e in exts:
            cand = adir / f"{stem}{e}"
            if cand.is_file():
                return cand
    return None


def find_prediction(
    image_path: str | Path,
    task: str,
    *,
    model: Optional[str] = None,
    date: Optional[str] = None,
    fmt: Optional[str] = None,
) -> Optional[Path]:
    """Find an existing prediction file for an image — a specific ``model`` if given,
    else every model. Returns the first existing file, or ``None``."""
    root, img_date, stem = parse_image_path(image_path)
    d = date if date is not None else img_date
    exts = [label_ext(fmt)] if fmt else list(_ANY_EXTS)

    models = [model] if model else list_models(root)
    for m in models:
        pdir = prediction_dir(root, m, d, task)
        for e in exts:
            cand = pdir / f"{stem}{e}"
            if cand.is_file():
                return cand
    return None
