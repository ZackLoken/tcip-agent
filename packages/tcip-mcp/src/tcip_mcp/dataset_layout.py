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

# Per-image COCO/JSON is the canonical on-disk label format; YOLO/VOC survive only for explicit
# import/export. ``json`` is the default, and ``.json`` is preferred when resolving an existing label.
LABEL_EXT = {"json": ".json", "yolo": ".txt", "voc": ".xml", "coco": ".json", "labelme": ".json"}
_ANY_EXTS = (".json", ".txt", ".xml")
DEFAULT_TRAIT = "default"
DEFAULT_MODEL = "live"
TASKS = ("detect", "segment")
#: Split names ``make_splits(materialize=True)`` emits.
SPLIT_NAMES = ("train", "val", "test")


def label_ext(fmt: Optional[str]) -> str:
    """File extension for a label format. The canonical on-disk format is per-image JSON, so an
    unspecified format resolves to ``.json`` (YOLO/VOC only via an explicit ``fmt``)."""
    return LABEL_EXT.get((fmt or "json").lower(), ".json")


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


def parse_annotation_dir(path: str | Path) -> Optional[tuple[str, Optional[str], str]]:
    """``(campaign, date, task)`` for an annotation dir — declared inverse of ``annotation_dir``.

    A *campaign* is the object class being isolated. Sometimes that is the trait's own subject
    (catkins, for ``catkin_50per_date``); often it is an enabling object no trait names — a bush
    isolated so anything can be aggregated per plant, a leaf isolated before leaf area is measured.
    So a campaign name is **not** required to be a ``crops.yml`` trait.

    Anchored on the literal ``annotations`` segment, with ``TASKS`` disambiguating the optional
    ``<date>``. Returns ``None`` for a non-canonical path rather than guessing: a wrong campaign
    would attribute a human's confirmation to work they never looked at.
    """
    parts = Path(path).parts
    if "annotations" in parts:
        i = len(parts) - 1 - parts[::-1].index("annotations")  # last, so nested roots resolve
        rest = parts[i + 1:]
        if len(rest) == 2 and rest[1] in TASKS:
            return rest[0], None, rest[1]
        if len(rest) == 3 and rest[2] in TASKS:
            return rest[0], rest[1], rest[2]
        return None
    # ``labels/<task>`` — the curated single-campaign dataset ``materialize_review_dataset``
    # emits. Not canonical layout, but a shape this platform produces itself, so resolving it to
    # the default campaign is a fact rather than a guess. Without it the review loop's own hard
    # negatives would be unreadable.
    if len(parts) >= 2 and parts[-2] == "labels" and parts[-1] in TASKS:
        return DEFAULT_TRAIT, None, parts[-1]
    # ``<split>/labels`` — the {train,val,test} tree ``make_splits(materialize=True)`` emits, which
    # carries its own status store (``_carry_confirmed_negatives``) under this same default
    # campaign. Anchored on the split name: resolving *any* dir called ``labels`` would turn an
    # unresolvable campaign into a silent empty negative set.
    if len(parts) >= 2 and parts[-1] == "labels" and parts[-2] in SPLIT_NAMES:
        return DEFAULT_TRAIT, None, TASKS[0]
    return None


def status_bucket(campaign: Optional[str], date: Optional[str]) -> str:
    """The ``image_status.json`` key a confirmation belongs under.

    Scoped by campaign and date but **not** task: a Complete covers detect and segment together,
    which is how ``derive_image_status`` already evaluates them. A store keyed by image name alone
    re-applies one campaign's confirmations to every other campaign.
    """
    return f"{campaign or DEFAULT_TRAIT}/{date}" if date else (campaign or DEFAULT_TRAIT)


def normalize_status_store(raw: object) -> dict[str, dict[str, str]]:
    """``{bucket: {image_name: status}}`` — a shape guard, shared by every reader.

    A bucket is ``status_bucket(campaign, date)``. Anything that is not a dict of strings is not a
    campaign's confirmations and is ignored, so a malformed store yields no confirmations rather
    than a wrong one.
    """
    if not isinstance(raw, dict):
        return {}
    return {
        key: {k: v for k, v in value.items() if isinstance(v, str)}
        for key, value in raw.items() if isinstance(value, dict)
    }


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


def _dir_has_label_file(d: Path) -> bool:
    """True if ``d`` holds at least one label file (any supported extension).

    An *empty* label file counts as a label file here — this answers "was anything written for
    this campaign on this date", which is what the GUI's selector needs. It does **not** mean the
    image is a confirmed negative: that requires a human Complete recorded in
    ``.tcip/state/image_status.json``, and training reads only that (``confirmed_negative_names``).
    """
    if not d.is_dir():
        return False
    return any(p.is_file() and p.suffix in _ANY_EXTS for p in d.iterdir())


def traits_with_labels(dataset_root: str | Path, date: Optional[str]) -> list[str]:
    """Traits that actually have ≥1 label file (detect or segment) on ``date``.

    This is what the GUI's trait selector should offer for a given date — a trait
    campaign with no labels on the selected date is not a meaningful choice there,
    so offering it (as the flat ``list_traits`` does) lands the user on an empty
    canvas. Sorted, same order as ``list_traits``.
    """
    root = Path(dataset_root)
    return [
        trait
        for trait in list_traits(root)
        if any(_dir_has_label_file(annotation_dir(root, trait, date, task)) for task in TASKS)
    ]


def models_with_predictions(dataset_root: str | Path, date: Optional[str]) -> list[str]:
    """Models that actually have ≥1 prediction file (detect or segment) on ``date``.

    The model selector's job is to overlay a model's predictions; a model with no
    predictions on the selected date has nothing to show there. Sorted, same order
    as ``list_models``.
    """
    root = Path(dataset_root)
    return [
        model
        for model in list_models(root)
        if any(_dir_has_label_file(prediction_dir(root, model, date, task)) for task in TASKS)
    ]


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
