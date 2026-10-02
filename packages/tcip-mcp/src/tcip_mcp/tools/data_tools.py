"""Data management tools: census a dataset, split data. Per-file quality checks live in
the ``doctor`` command's ``check_data_quality``."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import tcip_store

from tcip_mcp.server import tool
from tcip_mcp.audit import audited
from tcip_mcp.pipelines.data.splits import DEFAULT_GROUP_BY, DEFAULT_SEED, DEFAULT_VAL_RATIO


@tool()
@audited
def freeze_selection(project: Path, experiment_id: str, output_path: str | None = None) -> dict:
    """Freeze a finished run's own drawn train/val partition into a selection, so a later run can
    bind to the identical partition instead of drawing its own.

    A run of any task freezes, whatever shape its ground truth is: the selection is the run's own
    train and val samples as its resolved partition records them, each carrying the digest its
    ground truth had when the run read it.

    Reads the run's resolution (``experiments.run_resolution``) and refuses, naming the primitive,
    when: ``experiment_id`` names no training run; the split is spatial (region identities, not
    stems); the run's val side is empty; a member's ground truth has moved since the run (the file
    the record names now digests differently, the moved members named); a selection already exists
    at the output directory; or the selection its resolved ``scope`` composes is one
    :func:`~tcip_mcp.pipelines.data.selection.write_selection` refuses.

    The frozen selection's ``calibration`` and ``holdout`` sides are always empty, so an assessment
    against it refuses by name; this tool's answer carries a ``note`` saying so.

    Args:
        experiment_id: The finished run to freeze the drawn partition of.
        output_path: Where to write the selection. Defaults to
            ``<dataset_root>/splits/frozen-<experiment_id>``, resolved from the run's own recorded
            sources through ``dataset_root_of``; refused when that does not resolve (a source
            outside the canonical ``<dataset_root>/images/...`` layout).
    """
    from dataclasses import replace

    from tcip_mcp.dataset_layout import dataset_root_of
    from tcip_mcp.experiments import run_resolution
    from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint
    from tcip_mcp.pipelines.data.selection import (
        ClassScope, Selection, read_selection_checked, write_selection,
    )
    from tcip_mcp.pipelines.data.split_construction import moved_since_run, partition_samples

    try:
        resolved = run_resolution(experiment_id, project=project)
    except ValueError as exc:
        return {"error": str(exc)}
    partition = resolved["partition"]
    if "spatial_manifest" in resolved["data"]["split"]:
        return {"error": f"{experiment_id!r}'s split is spatial (region identities, not stems): "
                         "freeze_selection binds a stem-keyed partition, which a spatial split "
                         "never draws."}
    # The run's own class space, so what this selection records is what the checkpoint records.
    scope = ClassScope.of(resolved["data"])
    at_run = partition["ground_truth_digests"]
    samples = [replace(s, ground_truth_digest=at_run[s.ground_truth])
               for s in partition_samples(partition) if s.side in ("train", "val")]
    n_train = sum(1 for s in samples if s.side == "train")
    n_val = len(samples) - n_train
    moved = moved_since_run(samples, at_run)
    if not n_val:
        return {"error": f"{experiment_id!r} trained without validation (an empty val side): "
                         "a partition no bind can use."}
    if moved:
        return {"error": f"the ground truth of {len(moved)} member(s) changed since "
                         f"{experiment_id!r} trained ({moved[:5]}): a selection "
                         "composed from them would bind a later run to ground truth this run "
                         "never saw. Freeze a run whose members have not moved, or draw a fresh "
                         "split over the current data."}

    a_source = samples[0].source
    dataset_root = dataset_root_of(a_source)
    if dataset_root is None:
        return {"error": f"this run's own source {a_source!r} does not resolve under a dataset "
                         "root (dataset_root_of), so the frozen selection has no dataset to "
                         "record a fingerprint for."}
    if output_path is None:
        output_path = str(dataset_root / "splits" / f"frozen-{experiment_id}")
    out_dir = Path(output_path)
    existing, existing_error = read_selection_checked(out_dir, project=project)
    if existing is not None or existing_error is not None:
        return {"error": f"a selection already exists at {output_path!r}: "
                         f"{existing_error or 'freeze_selection never overwrites one.'}"}

    try:
        fingerprint = dataset_fingerprint(dataset_root)
    except tcip_store.SchemaVersionRefused as exc:
        return {"error": f"cannot fingerprint the dataset for the frozen selection: {exc}"}

    try:
        write_selection(out_dir, Selection(
            samples=tuple(samples), scope=scope, seed=partition["seed"],
            group_by=partition["group_by"], dataset_fingerprint=fingerprint,
        ), project=project)
    except ValueError as exc:
        return {"error": f"{experiment_id!r}'s resolved record: {exc}"}
    return {
        "selection_dir": str(out_dir), "train": n_train, "val": n_val, "calibration": 0,
        "holdout": 0,
        "note": "the reference sides are empty (a training run's own drawn partition records "
               "no reference draw): an assessment against this selection refuses by name.",
    }


def _scan_dataset(root: str) -> dict:
    """Scan a directory tree for images and labels.

    Labels are the name-based per-image JSON (one file per image, all subjects) under
    ``annotations/<date>/``, a review baseline directory's copies excluded. The census reads no
    document.

    ``labels`` is a raw ``rglob``, so it counts a file whose name is reserved for a prediction
    bucket's own record; ``reserved_name_labels`` names each one. ``reserved_name_images`` names
    every image whose own stem is reserved the same way. ``predictions`` are the documents of
    every published bucket (:func:`~tcip_mcp.buckets.bucket_dirs`).

    ``images`` is built per bucket through
    :func:`~tcip_mcp.pipelines.image_utils.list_logical_images`: a stem collision within one bucket
    raises :class:`~tcip_mcp.pipelines.image_utils.AmbiguousImageStem`, and a grouped capture
    counts once, its own manifest. Walks one level under ``images/``: the flat root itself plus
    each direct date-bucket subdirectory. Falls back to a raw walk of the whole dataset root only
    when there is no canonical ``images/`` tree.
    """
    from tcip_annotation.json_io import is_bucket_record, is_reserved_stem
    from tcip_annotation.review_engine import BASELINE_DIRNAME
    from tcip_mcp.buckets import bucket_dirs, read_bucket
    from tcip_mcp.dataset_layout import LABEL_SUFFIX, annotation_root, image_root
    from tcip_mcp.pipelines.image_utils import BandGroupRef, IMAGE_EXTS, list_logical_images

    root_path = Path(root)
    image_exts = IMAGE_EXTS
    images: list[str] = []
    labels: list[str] = []
    preds: list[str] = []
    reserved_name_labels: list[str] = []
    reserved_name_images: list[str] = []

    # Find images through the platform's own bucket enumeration: a stem collision refuses here
    # too, and a grouped capture counts once, its own manifest.
    images_dir = image_root(root_path)
    if images_dir.is_dir():
        buckets = [images_dir] + sorted(p for p in images_dir.iterdir() if p.is_dir())
        for bucket in buckets:
            for source in list_logical_images(bucket).values():
                f = source.manifest_path if isinstance(source, BandGroupRef) else source
                images.append(str(f))
                if is_reserved_stem(f.stem):
                    reserved_name_images.append(str(f))
    else:
        # No canonical images/ tree, so no bucket contract to route through this walk.
        for f in sorted(root_path.rglob("*")):
            if f.is_file() and f.suffix.lower() in image_exts:
                images.append(str(f))
                if is_reserved_stem(f.stem):
                    reserved_name_images.append(str(f))

    # Ground-truth labels: annotations/[<date>/]<stem>.json (one file per image, every subject),
    # a review baseline copy under BASELINE_DIRNAME excluded: it is a snapshot, not a label.
    ann_dir = annotation_root(root_path)
    if ann_dir.is_dir():
        labels = [
            str(f) for f in sorted(ann_dir.rglob(f"*{LABEL_SUFFIX}"))
            if f.is_file() and BASELINE_DIRNAME not in f.parts
        ]
        reserved_name_labels = [f for f in labels if is_bucket_record(Path(f).name)]

    preds = [str(f) for bucket in bucket_dirs(root_path) for f in read_bucket(bucket).document_paths]

    return {
        "images": images, "labels": labels, "predictions": preds,
        "reserved_name_labels": reserved_name_labels, "reserved_name_images": reserved_name_images,
    }


def scan_dataset(folder_path: str) -> dict:
    """Scan a folder for images, labels, and predictions.

    Reads the name-based per-image JSON labels (one file per image, all subjects).

    Expects the canonical layout (see tcip_mcp.dataset_layout):
        images/<date>/  annotations/<date>/<stem>.json  predictions/.../bucket.json

    ``reserved_name_labels`` names every label counted in ``labels_count`` whose filename is
    reserved for a prediction bucket's own record. ``reserved_name_images`` names every
    image counted in ``image_count`` whose own stem is reserved the same way; such an image
    otherwise sits in ``unlabeled_images``.

    Args:
        folder_path: Path to the dataset root directory.
    """
    if not Path(folder_path).is_dir():
        return {"error": f"Directory not found: {folder_path}"}

    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem
    from tcip_store import SchemaVersionRefused

    try:
        scan = _scan_dataset(folder_path)
    except AmbiguousImageStem as exc:
        return {"error": str(exc)}
    except SchemaVersionRefused as exc:
        return {"error": f"a .bandgroup manifest under {folder_path} could not be read: {exc}"}

    image_stems = {Path(p).stem: p for p in scan["images"]}
    label_stems = {Path(p).stem for p in scan["labels"]}

    paired = sum(1 for stem in image_stems if stem in label_stems)
    unlabeled = len(image_stems) - paired

    return {
        "path": folder_path,
        "image_count": len(scan["images"]),
        "labels_count": len(scan["labels"]),
        "predictions_count": len(scan["predictions"]),
        "paired_images": paired,
        "unlabeled_images": unlabeled,
        "image_stems_sample": sorted(image_stems.keys())[:10],
        "reserved_name_labels": scan["reserved_name_labels"],
        "reserved_name_images": scan["reserved_name_images"],
    }


@tool()
@audited
def draw_splits(
    project: Path,
    folder_path: str,
    train_ratio: float = 1.0 - DEFAULT_VAL_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    calibration_ratio: float = 0.0,
    holdout_ratio: float = 0.0,
    seed: int = DEFAULT_SEED,
    group_by: str = DEFAULT_GROUP_BY,
    group_key_map: dict[str, str] | None = None,
    stratify_foreground: bool = True,
    output_path: str | None = None,
    subject: str | None = None,
    attribute: str | None = None,
    ground_truth: str | None = None,
) -> dict:
    """Compute a leakage-free, annotation-stratified train/val/calibration/holdout selection.

    Non-destructive: it copies nothing and moves nothing. With ``output_path`` it writes a
    selection record listing, per sample, the image source, the label document, the group key and
    the side; without one it answers the same draw's statistics and writes nothing. Sibling tiles
    of one source image are kept in the same split, and, when ``stratify_foreground`` is set,
    splits are balanced by annotation count. Groups whole source images; a within-image split for
    a folder holding a single source is a training run's own route (``data.tiling`` in the run
    config).

    The draw is :func:`~tcip_mcp.pipelines.data.split_construction.admitted_membership` over
    whichever ground truth the place it is pointed at holds, then
    :func:`~tcip_mcp.pipelines.data.split_construction.draw_sides`.

    Without ``ground_truth``, the ground truth is the dataset's per-image label tree: for each
    capture date the dataset holds, every image carrying an annotation of ``subject`` (with every
    instance assessed for ``attribute``, when one is given) or a human's negative confirmation for
    it, so ``subject`` is required. Every admitted date enters one draw: a sample names its own
    source and its own label, and two dates holding a same-named image are two samples.
    ``stratify_foreground`` only toggles the annotation-count balancing.

    With ``ground_truth`` naming a directory of ``<stem>.png`` rasters, the ground truth is a
    per-image mask and a sample is admitted when its mask sits there beside its image; with
    ``ground_truth`` naming a ``.csv`` file, the ground truth is that table and a sample is one
    row, admitted when the image its key names exists. Neither takes ``subject``/``attribute``.
    Balancing by foreground count applies to label documents only.

    The ``calibration`` and ``holdout`` sides are the reference an assessment fits an operating
    point on and checks it against, drawn as one share and cut between the two at the same seed;
    a side whose ratio is zero is not drawn, and a negative ratio refuses. The draw refuses,
    before any write, when the tree holds fewer foreground groups of ``subject`` (and
    ``attribute``, when scoped) than one per requested side. The answer's
    ``calibration_foreground_groups`` reports how many of the reference's groups carry a
    foreground annotation, and ``realized_ratios`` each side's share of the draw actually
    delivered, which can diverge from the ratios asked for on a tree sized at the floor.

    Args:
        folder_path: Path to the dataset root directory.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        calibration_ratio: Fraction held out for an assessment to fit its operating point on.
        holdout_ratio: Fraction held out for an assessment to check its operating point against.
        seed: Random seed for reproducibility.
        group_by: Group selector: ``"tile_prefix"`` (strip a trailing ``_<x>_<y>`` tile offset) or
            ``"stem"`` (one group per member). Ignored when ``group_key_map`` is given. The
            resolved key is recorded on every sample.
        group_key_map: An agent-derived ``{identity: group_key}`` map overriding ``group_by``,
            keyed ``<date>/<stem>`` (the bare ``<stem>`` under a flat tree); must cover every
            admitted member. Recorded as ``group_by="explicit_map"``.
        stratify_foreground: Balance splits by foreground annotation count.
        output_path: Where to write the selection. Omitted, nothing is written and the answer is
            the draw's statistics only.
        subject: The object class the selection is drawn for. Required over label documents,
            the dataset's own per-image label tree or a ``ground_truth`` naming them.
        attribute: Scope the draw to instances already assessed for this attribute of ``subject``;
            an image carrying an instance never assessed for it is excluded entirely. ``None``
            draws over every instance of ``subject`` regardless of attribute state.
        ground_truth: Where this dataset's ground truth lives, named explicitly: a directory of
            label documents, a directory of ``<stem>.png`` masks, or a ``.csv`` table of one row
            per image; the images are the dataset's own ``images/`` tree either way.
    """
    if not Path(folder_path).is_dir():
        return {"error": f"Directory not found: {folder_path}"}

    from tcip_annotation.json_io import UnreadableLabelDocument, prediction_documents
    from tcip_mcp.dataset_layout import annotation_dir, annotation_root, list_dates
    from tcip_mcp.dataset_layout import resolve_images_dir
    from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint
    from tcip_mcp.pipelines.data.selection import (
        REFERENCE_SIDES, SIDES, ClassScope, Selection, write_selection,
    )
    from tcip_mcp.pipelines.data.split_construction import admitted_membership, draw_sides
    from tcip_mcp.pipelines.data.splits import foreground_group_count
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem, BandGroupIncomplete

    # Each place holding ground truth: (its name, images directory, ground truth).
    places: list[tuple[str, Path, str]]
    if ground_truth is not None:
        places = [(ground_truth, resolve_images_dir(folder_path, None), ground_truth)]
    else:
        labels = annotation_root(folder_path)
        dates = list_dates(folder_path, tree=annotation_root)
        places = [(d, resolve_images_dir(folder_path, d), str(annotation_dir(folder_path, d)))
                  for d in dates]
        if labels.is_dir() and (prediction_documents(labels) or not dates):
            places.append(("annotations/ (loose labels)", resolve_images_dir(folder_path, None),
                           str(labels)))
        if not places:
            return {"error": f"{folder_path} holds no per-image label tree (annotations/<date>/ "
                             "or a flat annotations/) for draw_splits to draw a subject-scoped "
                             "selection from; an external COCO document is converted into one "
                             "by import_coco first."}
        entries_by_images_dir: dict[Path, list[str]] = {}
        for name, entry_images_dir, _labels in places:
            entries_by_images_dir.setdefault(entry_images_dir, []).append(name)
        colliding = {d: names for d, names in entries_by_images_dir.items() if len(names) > 1}
        if colliding:
            detail = "; ".join(
                f"{img_dir}: {sorted(names)}" for img_dir, names in sorted(colliding.items())
            )
            return {"error": f"{folder_path} has label entries that resolve to the same images "
                             f"directory ({detail}): one image file would be admitted once per "
                             "entry and could land on both sides of the split. Give each date its "
                             "own images/<date>/ bucket, or merge the colliding label entries "
                             "into one."}

    ratios = {"train": train_ratio, "val": val_ratio, "calibration": calibration_ratio,
              "holdout": holdout_ratio}
    try:
        membership = admitted_membership(
            places, scope=ClassScope(subject=subject, attribute=attribute), group_by=group_by,
            group_key_map=group_key_map)
        drawn, counted = draw_sides(
            membership.samples, membership.scope, seed=seed, stratify=stratify_foreground,
            ratios={side: share for side, share in ratios.items() if share != 0})
    except tcip_store.SchemaVersionRefused as exc:
        return {"error": f"a .bandgroup manifest under {folder_path} could not be read: {exc}"}
    except (UnreadableLabelDocument, AmbiguousImageStem, BandGroupIncomplete,
            FileNotFoundError, ValueError, OSError) as exc:
        return {"error": str(exc)}

    by_side = {side: [key for key, sample in drawn.items() if sample.side == side]
               for side in SIDES}
    sizes = {side: len(keys) for side, keys in by_side.items()}
    group_of = {key: sample.group for key, sample in drawn.items()}.__getitem__
    response = {
        "splits": sizes,
        "foreground_annotations": {
            side: sum(counted[key] for key in keys) for side, keys in by_side.items()},
        "total_stems": len(drawn),
        "total_annotations": sum(counted.values()),
        "groups": len({sample.group for sample in drawn.values()}),
        "stratified": stratify_foreground,
        "seed": seed, "group_by": membership.group_by, "scope": asdict(membership.scope),
        "tallies": membership.tallies,
        "selection_dir": str(output_path) if output_path else None,
        "calibration_foreground_groups": foreground_group_count(
            [key for side in REFERENCE_SIDES for key in by_side[side]], counted, group_of),
        "realized_ratios": {side: size / len(drawn) for side, size in sizes.items()},
    }
    if not output_path:
        return response
    try:
        write_selection(output_path, Selection(
            samples=tuple(drawn[key] for key in sorted(drawn)), scope=membership.scope, seed=seed,
            group_by=membership.group_by, dataset_fingerprint=dataset_fingerprint(folder_path),
        ), project=project)
    except tcip_store.SchemaVersionRefused as exc:
        return {"error": f"cannot fingerprint the dataset for the selection: {exc}"}
    except ValueError as exc:
        return {"error": str(exc)}
    return response
