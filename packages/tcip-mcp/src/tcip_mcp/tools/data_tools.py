"""Data management tools: load datasets, validate quality, split data."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import tcip_store
from tcip_store import RECORD_JSON, Key, StoreDescriptor, register_store
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited

_SPLIT_DOC = RootedFileLocator(suffix=".json")
"""A split output directory's own documents. The directory is wherever the caller asked the
partition to be written, so no dataset resolver owns its layout and the entries are addressed
by name under it, the way a label tree no resolver describes is addressed."""

SPLIT_STEM_LIST_STORE = "split_stem_list"
register_store(
    StoreDescriptor(
        name=SPLIT_STEM_LIST_STORE,
        kind="record",
        key_fields=("split",),
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        locator=_SPLIT_DOC,
    )
)


def split_stem_list_key(split_dir: str | Path, split_name: str) -> Key:
    """The stems one split of a partition holds.

    ``last_writer_wins``: the whole list is written once from a partition the caller already
    computed, and no writer merges into the stored list.
    """
    return Key(SPLIT_STEM_LIST_STORE, str(Path(split_dir).absolute()), (split_name,))


SPLIT_MANIFEST_STORE = "split_manifest"
_SPLIT_MANIFEST_PARTS = ("split_manifest",)
register_store(
    StoreDescriptor(
        name=SPLIT_MANIFEST_STORE,
        kind="record",
        key_fields=("document",),
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        locator=_SPLIT_DOC,
    )
)


def split_manifest_key(split_dir: str | Path) -> Key:
    """How a split directory's partition was produced, so it can be reconstructed.

    ``last_writer_wins``: written once, whole, at the end of the partition that produced it.
    """
    return Key(SPLIT_MANIFEST_STORE, str(Path(split_dir).absolute()), _SPLIT_MANIFEST_PARTS)


ROOT_LABEL_CANDIDATES = ("annotations.json", "labels.json", "instances.json")
"""Candidate filenames for one assembled dataset-level label document at a dataset's root,
checked in this order; the first one present on disk is the dataset's label store. Shared by
:func:`_scan_dataset` and :func:`validate_data_quality` so both name the same three candidates."""


def _scan_dataset(root: str) -> dict:
    """Scan a directory tree for images and labels.

    Labels are the name-based per-image JSON (one file per image, all subjects) under
    ``annotations/<date>/`` (no detect/segment split), a review baseline directory's copies
    excluded, plus a single assembled dataset-level COCO at the root if one is present: the root
    candidate sits beside the per-image tree, never in place of it, so a dataset carrying both
    reports every one of them.

    An unreadable first-sorted label (undecodable, non-dict, or otherwise malformed) raises
    :class:`~tcip_annotation.json_io.UnreadableLabelDocument` rather than being folded into "format
    undetectable": the caller reports it as the named file it is, not a guess. ``format`` is an
    informational best guess (the per-image tree's first-sorted label's shape, or the root
    candidate's when there is no per-image tree), not a claim every label file shares it;
    :func:`validate_data_quality` decides format per file instead.

    ``labels`` is a raw ``rglob``, so it counts a file whose name is reserved for a prediction
    bucket's own provenance stamp the way :func:`~tcip_mcp.dataset_layout.subjects_on_date` and
    every bucket walk through ``prediction_documents`` would not; ``reserved_name_labels`` names
    each one, so a caller comparing this census against those walks can tell the difference is a
    known exclusion rather than a disagreement.
    """
    from tcip_annotation.json_io import is_sidecar_name, prediction_documents
    from tcip_annotation.review_engine import BASELINE_DIRNAME
    from tcip_mcp.dataset_layout import annotation_root, image_root, prediction_root
    from tcip_mcp.pipelines.image_utils import IMAGE_EXTS

    root_path = Path(root)
    image_exts = IMAGE_EXTS
    images: list[str] = []
    labels: list[str] = []
    preds: list[str] = []
    reserved_name_labels: list[str] = []
    detected_format: str | None = None

    # Find images (recurse to catch the canonical images/<date>/ layout).
    images_dir = image_root(root_path)
    scan_root = images_dir if images_dir.is_dir() else root_path
    for f in sorted(scan_root.rglob("*")):
        if f.is_file() and f.suffix.lower() in image_exts:
            images.append(str(f))

    # Ground-truth labels: annotations/[<date>/]<stem>.json (one file per image, every subject),
    # a review baseline copy under BASELINE_DIRNAME excluded: it is a snapshot, not a label.
    ann_dir = annotation_root(root_path)
    if ann_dir.is_dir():
        labels = [
            str(f) for f in sorted(ann_dir.rglob("*.json"))
            if f.is_file() and BASELINE_DIRNAME not in f.parts
        ]
        reserved_name_labels = [f for f in labels if is_sidecar_name(Path(f).name)]
        if labels:
            try:
                from tcip_annotation.format_io import detect_format
                detected_format = detect_format(labels[0])
            except ValueError:
                detected_format = None  # unrecognized: report nothing rather than a guess

    # A single COCO JSON at the dataset root: one more present label beside the per-image tree.
    root_candidate = _root_label_candidate(root, set(labels))
    if root_candidate is not None:
        labels.append(root_candidate)
        if detected_format is None:
            try:
                from tcip_annotation.format_io import detect_format
                detected_format = detect_format(root_candidate)
            except ValueError:
                pass

    # Predictions: predictions/<model>/[<date>/]<stem>.json; each model/date bucket is walked on
    # its own through prediction_documents, so the bucket's own stamps are excluded everywhere.
    pred_dir = prediction_root(root_path)
    if pred_dir.is_dir():
        preds = [
            str(f)
            for bucket in sorted({p.parent for p in pred_dir.rglob("*.json")})
            for f in prediction_documents(bucket)
        ]

    return {
        "images": images, "labels": labels, "predictions": preds, "format": detected_format,
        "reserved_name_labels": reserved_name_labels,
    }


@mcp.tool()
@audited
def scan_dataset(folder_path: str) -> dict:
    """Scan a folder for images, labels, and predictions.

    Reads the name-based per-image JSON labels (one file per image, all subjects), or an assembled
    dataset-level COCO.

    Expects the canonical layout (see tcip_mcp.dataset_layout):
        images/<date>/  annotations/<date>/<stem>.json  predictions/<model>/<date>/<stem>.json

    ``reserved_name_labels`` names every label counted in ``labels_count`` whose filename is
    reserved for a prediction bucket's own provenance stamp: this census walks with ``rglob`` and
    counts it, while the bucket walks this platform reads labels through exclude it.

    Args:
        folder_path: Path to the dataset root directory.
    """
    if not Path(folder_path).is_dir():
        return {"error": f"Directory not found: {folder_path}"}

    from tcip_annotation.json_io import UnreadableLabelDocument

    try:
        scan = _scan_dataset(folder_path)
    except UnreadableLabelDocument as exc:
        return {"error": str(exc)}

    image_stems = {Path(p).stem: p for p in scan["images"]}
    label_stems = {Path(p).stem for p in scan["labels"]}

    paired = sum(1 for stem in image_stems if stem in label_stems)
    unlabelled = len(image_stems) - paired

    return {
        "path": folder_path,
        "format": scan.get("format"),
        "image_count": len(scan["images"]),
        "labels_count": len(scan["labels"]),
        "predictions_count": len(scan["predictions"]),
        "paired_images": paired,
        "unlabelled_images": unlabelled,
        "image_stems_sample": sorted(image_stems.keys())[:10],
        "reserved_name_labels": scan["reserved_name_labels"],
    }


def _root_label_candidate(folder_path: str, already_present: set) -> str | None:
    """The dataset root's own assembled-label candidate, if one is present and not already
    counted among ``already_present``.

    The one walk of ``ROOT_LABEL_CANDIDATES`` in their declared first-match order, called by
    :func:`_scan_dataset` before the candidate joins its ``labels`` list and by any other caller
    that has its own already-counted set to check the candidate against, so a present root
    candidate can never be walked for twice by two diverging implementations. A candidate whose
    format cannot be determined is still returned: it is a present label file, not evidence the
    dataset carries none, and detecting its format is left to the caller.
    """
    root_path = Path(folder_path)
    for candidate in ROOT_LABEL_CANDIDATES:
        cpath = root_path / candidate
        if cpath.is_file():
            return None if str(cpath) in already_present else str(cpath)
    return None


@mcp.tool()
@audited
def validate_data_quality(folder_path: str) -> dict:
    """Run quality checks on a dataset (any supported annotation format).

    Checks, decided per label file rather than once for the whole dataset (a store mixing shapes
    cannot report valid because one file's shape happened to be detected first): stem matching
    between images and labels (a label with no matching image, or for COCO, a referenced image
    file not found in the images dir), an empty per-image label with no human confirmation that
    the image is a negative, a file whose format cannot be determined, and a file present but
    unreadable. Class consistency against a subject registry and coordinate-range validation are
    not implemented.

    ``total_labels`` counts exactly the files ``scan_dataset``'s own ``labels_count`` counts (the
    per-image tree, a review baseline excluded, plus a present root candidate): the two are the
    same list. ``format`` is the distinct shapes actually found among the label files this call
    could classify: one shape's name when every file agrees, the shapes sorted when they do not,
    and ``None`` when no label file's format could be determined at all (labels present but
    undetectable, or no labels present). ``reserved_name_labels`` names every one of those files
    whose filename is reserved for a prediction bucket's own provenance stamp, the same
    ``scan_dataset`` field: this scan's per-file checks still run on it (it is a present file the
    census counted), but no bucket walk elsewhere in the platform would ever read it as a label.

    Args:
        folder_path: Path to the dataset root directory.
    """
    if not Path(folder_path).is_dir():
        return {"error": f"Directory not found: {folder_path}"}

    from tcip_annotation import json_io
    from tcip_annotation.format_io import _parse_coco_json, detect_format
    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_mcp.dataset_layout import (
        annotation_date, confirmed_negative_names_any_subject, normalize_status_store,
        read_image_status_store, resolve_image_name,
    )

    try:
        scan = _scan_dataset(folder_path)
    except UnreadableLabelDocument as exc:
        return {"error": str(exc)}
    issues: list[dict] = []

    image_stems = {Path(p).stem for p in scan["images"]}
    label_paths = list(scan["labels"])

    negatives = confirmed_negative_names_any_subject(
        normalize_status_store(read_image_status_store(folder_path))
    )

    subjects: set[str] = set()
    shapes_found: set[str] = set()
    for label_path in label_paths:
        stem = Path(label_path).stem
        try:
            file_fmt = detect_format(label_path)
        except ValueError as exc:
            issues.append({"level": "error", "file": label_path,
                          "message": f"cannot determine annotation format: {exc}"})
            continue
        except UnreadableLabelDocument as exc:
            issues.append({"level": "error", "file": label_path,
                          "message": f"label file will not read: {exc}"})
            continue
        shapes_found.add(file_fmt)

        if file_fmt == "json":
            if stem not in image_stems:
                issues.append({"level": "error", "file": label_path, "message": "No matching image"})
            try:
                anns = json_io.read_annotations(label_path)
            except UnreadableLabelDocument as exc:
                issues.append({"level": "error", "file": label_path,
                              "message": f"label file will not read: {exc}"})
                continue
            for a in anns:
                subjects.add(a.subject)
            if not anns:
                name = resolve_image_name(folder_path, annotation_date(label_path), stem)
                if name is None or name not in negatives:
                    issues.append({"level": "error", "file": label_path,
                                  "message": "empty label file, not a confirmed negative for any "
                                  "subject; excluded from training"})
        elif file_fmt == "coco":
            try:
                coco = _parse_coco_json(label_path)
                for c in coco.get("categories", []):
                    if c.get("name"):
                        subjects.add(str(c["name"]))
                coco_fnames = {img.get("file_name", "") for img in coco.get("images", [])}
                for fn in coco_fnames:
                    if Path(fn).stem not in image_stems:
                        issues.append({"level": "warning", "file": label_path,
                                      "message": f"COCO image '{fn}' not found in images dir"})
            except Exception as e:
                issues.append({"level": "error", "file": label_path, "message": f"COCO parse error: {e}"})

    if not shapes_found:
        report_format = None
    elif len(shapes_found) == 1:
        report_format = next(iter(shapes_found))
    else:
        report_format = sorted(shapes_found)

    return {
        "path": folder_path,
        "format": report_format,
        "total_images": len(scan["images"]),
        "total_labels": len(label_paths),
        "subjects": sorted(subjects),
        "issues": issues,
        "issue_count": len(issues),
        "is_valid": all(i["level"] != "error" for i in issues),
        "reserved_name_labels": scan["reserved_name_labels"],
    }


@mcp.tool()
@audited
def make_splits(
    folder_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
    group_by: str = "tile_prefix",
    group_key_map: dict[str, str] | None = None,
    stratify_foreground: bool = True,
    output_path: str | None = None,
    materialize: bool = False,
    copy_files: bool = True,
    subject: str | None = None,
    spatial: bool = False,
    tile_size: int | None = None,
    overlap: float | None = None,
    buffer: int | None = None,
) -> dict:
    """Compute a leakage-free, annotation-stratified train/val split.

    Non-destructive by default: emits ``{train,val}.json`` stem manifests plus a stats
    dict. Sibling tiles of one source image are kept in the same split (no tree-/canopy-level
    leakage), and, when ``stratify_foreground`` is set, splits are balanced by annotation
    count so dense and sparse sources are proportionally represented.

    With ``materialize=True`` it additionally lays out a
    ``{train,val}/{images,labels}/`` tree under ``output_path`` (defaulting to
    ``folder_path/splits``), copying (or symlinking, ``copy_files=False``) each stem's image and
    label, and adds ``output_dir`` / ``structure`` to the return.

    Args:
        folder_path: Path to the dataset root directory.
        train_ratio: Fraction for training set. Defaults to 0.8, the complement of the
            unchanged 0.2 validation default once ``test_ratio`` is 0.
        val_ratio: Fraction for validation set.
        test_ratio: Must be 0. No launch path honours a held-out test list, so make_splits
            writes train and val only; a non-zero value is refused rather than writing a
            partition nothing downstream can consume.
        seed: Random seed for reproducibility.
        group_by: Group selector: ``"tile_prefix"`` (strip a trailing
            ``_<x>_<y>`` tile offset) or ``"stem"`` (one group per image). Ignored when
            ``group_key_map`` is given.
        group_key_map: An agent-derived ``{stem: group_key}`` map overriding ``group_by``;
            must cover every stem in the dataset. Recorded as ``group_by="explicit_map"`` in
            the result and manifest (the resolved policy, not the raw ``group_by`` string).
        stratify_foreground: Balance splits by foreground annotation count.
        output_path: Where to write manifests (and, when materializing, the file tree).
            Defaults to ``folder_path/splits`` when materializing, else manifests are
            written only if this is set.
        materialize: Also copy/symlink files into a {train,val}/{images,labels}/ tree.
        copy_files: Copy files (True) or create symlinks (False) when materializing.
        subject: The object the confirmed negatives are keyed under. A materialized split tree
            can't recover the subject from its own path, so an image a human confirmed negative
            is silently dropped from the materialized set (reads as an unconfirmed empty label)
            unless this is passed. Only relevant with `materialize=True`.
        spatial: Derive a within-image train/val split over one source's own tile lattice
            (:func:`~tcip_mcp.pipelines.data.splits.spatial_strip_split`) instead of grouping
            whole source images. Requires ``tile_size``/``overlap`` and a folder holding exactly
            one labeled image; refuses otherwise, naming a training run's own automatic route
            (``data.tiling`` in the run config, which derives this the same way when a run's
            dataset turns out to be single-source) as the alternative for the common multi-source
            case. Ignores ``group_by``/``group_key_map``/``stratify_foreground``/``copy_files``/
            ``subject``, and is not compatible with ``materialize`` (there is one source image,
            not a set of files to lay out into a split tree); ``train_ratio``/``val_ratio`` still
            apply, a zero ratio drops that side entirely, and ``test_ratio`` is refused the same
            way as the non-spatial path.
        tile_size: Tile edge in pixels, required when ``spatial=True``.
        overlap: Tile overlap fraction, required when ``spatial=True``.
        buffer: Minimum pixel gap kept at every boundary between two differently-assigned
            regions, ``spatial=True`` only. Defaults to ``tile_size`` (see
            :func:`spatial_strip_split`).
    """
    if test_ratio != 0:
        return {"error": "test_ratio must be 0: no launch path honours a held-out test list, so "
                         "make_splits writes train and val only."}

    if spatial:
        return _make_spatial_split(
            folder_path, tile_size=tile_size, overlap=overlap,
            fractions=(train_ratio, val_ratio),
            seed=seed, buffer=buffer, output_path=output_path, materialize=materialize,
        )

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        return {"error": "test_ratio must be 0 and train_ratio + val_ratio must sum to 1.0."}
    if not Path(folder_path).is_dir():
        return {"error": f"Directory not found: {folder_path}"}

    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_mcp.pipelines.data.splits import (
        group_balanced_split,
        count_label_lines,
        resolve_group_key_fn,
    )

    try:
        scan = _scan_dataset(folder_path)
    except UnreadableLabelDocument as exc:
        return {"error": str(exc)}
    image_map = {Path(p).stem: p for p in scan["images"]}
    label_map = {Path(p).stem: p for p in scan["labels"]}

    stratified = bool(stratify_foreground and label_map)
    if stratified:
        stems = sorted(set(image_map) & set(label_map))
    else:
        stems = sorted(image_map)
    if not stems:
        return {"error": "No images found to split"}

    annotation_counts = None
    if stratified:
        # count_label_lines is JSON-aware; raw count_lines would count pretty-printed JSON
        # lines as annotations (a {objects: []} negative reads as ~5 foreground objects).
        try:
            annotation_counts = {
                s: count_label_lines(Path(label_map[s]).parent, s) for s in stems
            }
        except UnreadableLabelDocument as exc:
            return {"error": str(exc)}

    try:
        group_key_fn = resolve_group_key_fn(group_by, stems, group_key_map=group_key_map)
    except ValueError as exc:
        return {"error": str(exc)}
    resolved_group_by = "explicit_map" if group_key_map else group_by
    parts = group_balanced_split(
        stems,
        annotation_counts=annotation_counts,
        group_key_fn=group_key_fn,
        splits=(train_ratio, val_ratio, test_ratio),
        seed=seed,
    )

    # Content hash of the labels this split partitions: two runs with the same seed still yield
    # different splits over different GT, so the hash + seed together identify the partition.
    dataset_hash = None
    labels_root: Path | None = None
    if label_map:
        from tcip_mcp.pipelines.resolution import dataset_hash as _dataset_hash
        labels_root = Path(next(iter(label_map.values()))).parent
        dataset_hash = _dataset_hash(labels_root, stems=stems)

    # make_splits emits no test partition (see test_ratio, refused above); train and val are the
    # only two names carried into the stem lists, the manifest and this call's own summary.
    kept_splits = ("train", "val")

    counts = annotation_counts or {}
    out_dir = Path(output_path) if output_path else (Path(folder_path) / "splits" if materialize else None)
    manifest_dir = None
    if out_dir is not None:
        from tcip_mcp.pipelines.resolution import dataset_fingerprint

        out_dir.mkdir(parents=True, exist_ok=True)
        for split_name in kept_splits:
            tcip_store.replace(split_stem_list_key(out_dir, split_name), sorted(parts[split_name]))
        manifest: dict[str, Any] = {
            "seed": seed, "dataset_hash": dataset_hash, "group_by": resolved_group_by,
            "labels_root": str(labels_root) if labels_root is not None else None,
            "dataset_fingerprint": dataset_fingerprint(folder_path),
            "splits": {k: sorted(parts[k]) for k in kept_splits},
        }
        if group_key_map:
            manifest["group_key_map"] = group_key_map
        tcip_store.replace(split_manifest_key(out_dir), manifest)
        manifest_dir = str(out_dir)

    result = {
        "splits": {k: len(parts[k]) for k in kept_splits},
        "foreground_annotations": {
            k: sum(int(counts.get(s, 0)) for s in parts[k]) for k in kept_splits
        },
        "total_stems": len(stems),
        "total_annotations": sum(int(v) for v in counts.values()),
        "groups": len({group_key_fn(s) for s in stems}),
        "seed": seed,
        "dataset_hash": dataset_hash,
        "group_by": resolved_group_by,
        "stratified": stratified,
        "manifest_dir": manifest_dir,
    }

    if materialize:
        # The carry is computed before anything is written: a refusal here must leave no
        # partial split tree on disk.
        negative_carry = _compute_negative_carry(label_map, parts, image_map, subject)
        place_fn = shutil.copy2 if copy_files else os.symlink
        for split_name in kept_splits:
            split_stems = parts[split_name]
            img_dir = out_dir / split_name / "images"
            lbl_dir = out_dir / split_name / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for stem in split_stems:
                src_img = Path(image_map[stem])
                dst_img = img_dir / src_img.name
                if not dst_img.exists():
                    place_fn(str(src_img), str(dst_img))
                if stem in label_map:
                    src_lbl = Path(label_map[stem])
                    dst_lbl = lbl_dir / src_lbl.name
                    if not dst_lbl.exists():
                        place_fn(str(src_lbl), str(dst_lbl))
        _apply_negative_carry(negative_carry, out_dir, subject)
        if negative_carry is not None and negative_carry.contradicted:
            result["contradicted_negatives"] = sorted(negative_carry.contradicted)
        result["output_dir"] = str(out_dir)
        result["structure"] = f"{out_dir}/{{train,val}}/{{images,labels}}/"

    return result


def _make_spatial_split(
    folder_path: str, *, tile_size: int | None, overlap: float | None,
    fractions: tuple[float, float], seed: int, buffer: int | None,
    output_path: str | None, materialize: bool,
) -> dict:
    """``make_splits(spatial=True)``'s body: a within-image train/val split for one source.

    No dataset is constructed here (this tool never builds a ``TiledDetectionDataset``), so the
    tile lattice is the pure geometry :func:`~tcip_mcp.pipelines.data.splits.spatial_strip_split`
    describes; a training run's own manifest (``_persist_split_manifest``) instead reads the
    identities off its actually-constructed datasets, which can differ when the run also drops
    tiles for reasons this tool has no way to apply (e.g. ``skip_empty``).
    """
    if tile_size is None or overlap is None:
        return {"error": "spatial=True requires tile_size and overlap; a training run's own "
                         "automatic route (data.tiling in the run config) derives these from the "
                         "run's tiling geometry instead."}
    if materialize:
        return {"error": "spatial=True does not support materialize: there is one source image, "
                         "not a set of files to lay out into a split tree."}
    if not Path(folder_path).is_dir():
        return {"error": f"Directory not found: {folder_path}"}

    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_mcp.pipelines.data.splits import image_extent_from_labels, spatial_strip_split
    from tcip_mcp.pipelines.data.tiling import tile_positions, tile_within_extent

    try:
        scan = _scan_dataset(folder_path)
    except UnreadableLabelDocument as exc:
        return {"error": str(exc)}
    image_map = {Path(p).stem: p for p in scan["images"]}
    label_map = {Path(p).stem: p for p in scan["labels"]}
    stems = sorted(set(image_map) & set(label_map))
    if len(stems) != 1:
        return {"error": f"spatial=True requires a single-stem folder ({len(stems)} found); use "
                         "the non-spatial grouped split for a multi-source folder, or a training "
                         "run's own automatic route."}
    stem = stems[0]
    labels_root = Path(label_map[stem]).parent
    try:
        extent = image_extent_from_labels(labels_root, stem)
    except UnreadableLabelDocument as exc:
        return {"error": str(exc)}
    if extent is None:
        return {"error": f"{stem}'s label file carries no width/height; cannot derive a spatial "
                         "split without the image extent."}
    width, height = extent

    try:
        split = spatial_strip_split(
            width, height, tile_size, overlap, fractions=fractions, split_names=("train", "val"),
            seed=seed, buffer=buffer,
        )
    except ValueError as exc:
        return {"error": str(exc)}

    lattice = tile_positions(height, width, split.tile_size, split.stride)
    in_extent = [(tx, ty) for tx, ty in lattice
                if tile_within_extent(tx, ty, split.tile_size, width, height)]
    ids: dict[str, set[str]] = {name: set() for name in split.regions}
    for tx, ty in in_extent:
        name = split.split_name_for(tx, ty)
        if name is not None:
            ids[name].add(split.identity_for(stem, tx, ty))

    spatial_manifest: dict[str, Any] = {
        f"{name}_identities": sorted(ids[name]) for name in split.regions
    }
    spatial_manifest.update({
        "width": split.width, "height": split.height, "tile_size": split.tile_size,
        "overlap": split.overlap, "axis": split.axis, "buffer": split.buffer,
        "seed": split.seed,
        "requested_fractions": dict(zip(split.split_names, split.requested_fractions)),
        "realized_fractions": split.realized_fractions,
        "realized_discard_fraction": split.realized_discard_fraction,
        "kept_tiles": split.kept_tiles,
        "tiles_dropped_past_extent": split.tiles_dropped_past_extent,
        "tiles_dropped_outside_regions": split.tiles_dropped_outside_regions,
    })

    from tcip_mcp.pipelines.resolution import dataset_fingerprint

    out_dir = Path(output_path) if output_path else Path(folder_path) / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in split.regions:
        tcip_store.replace(split_stem_list_key(out_dir, name), sorted(ids[name]))
    tcip_store.replace(split_manifest_key(out_dir), {
        "seed": seed, "group_by": "spatial_strip", "spatial": spatial_manifest,
        "labels_root": str(labels_root), "dataset_fingerprint": dataset_fingerprint(folder_path)})

    return {
        "splits": {name: len(ids[name]) for name in split.regions},
        "seed": seed,
        "group_by": "spatial_strip",
        "spatial": spatial_manifest,
        "manifest_dir": str(out_dir),
    }


class _NegativeCarry:
    """What :func:`_compute_negative_carry` found, before anything is written: each split's own
    slice of the source subject's confirmed negatives, the schema digest to stamp them with, and
    the names the label content contradicted (excluded from every slice, reported to the caller)."""

    def __init__(self, by_split: dict[str, dict[str, dict[str, str]]], contradicted: set[str],
                digest: str | None, src_classes: Path | None) -> None:
        self.by_split = by_split
        self.contradicted = contradicted
        self.digest = digest
        self.src_classes = src_classes


def _compute_negative_carry(label_map: dict, parts: dict, image_map: dict,
                            subject: str | None) -> "_NegativeCarry | None":
    """Reads the source subject's confirmed negatives and assigns each to the split holding its
    image, entirely before any split-tree file is written (see the call site).

    A split tree is ``{train,val}/labels`` by construction and cannot recover the subject from
    its path, so the confirmations are carried explicitly under the threaded ``subject`` (keyed by
    ``status_bucket(subject, None)``, since the split carries no date). Without this, every image a
    human confirmed negative reads as an unconfirmed empty in the split and is dropped from training.
    No subject threaded -> nothing to attribute the confirmations to, so ``None`` is returned.

    A split partitions whatever the source dataset holds, across every capture date at once, and
    lands it in one undated bucket. So the source side reads every bucket the store names this
    subject under, the keys writers actually stated, rather than a date derived from each label
    directory's path: a confirmation recorded under a key that does not spell the directory it sits
    in is still the human's, and deriving the key would drop it. A name whose label file now holds
    the subject is excluded the way :func:`confirmed_negative_records_every_date` always excludes
    one, named in the returned carry's ``contradicted`` rather than silently dropped.
    """
    if not subject:
        return None
    from tcip_mcp.class_registry import (
        RegistryError, attribute_schema_digest, read_registry,
    )
    from tcip_mcp.dataset_layout import classes_path, dataset_root_of
    from tcip_mcp.pipelines.data.datasets import confirmed_negative_records_every_date

    src_dirs = {Path(p).parent for p in label_map.values()}
    if not src_dirs:
        return None
    negatives: dict[str, dict[str, str]] = {}
    contradicted: set[str] = set()
    for d in sorted(src_dirs):
        negatives.update(confirmed_negative_records_every_date(
            d, subject=subject, contradicted_out=contradicted))
    if not negatives:
        return _NegativeCarry(by_split={}, contradicted=contradicted, digest=None, src_classes=None)

    # A best-effort registry read: without a stamp, a split tree can never quarantine a stale
    # confirmation later (a permanent no-op, not "admit until proven stale").
    digest = None
    src_classes: Path | None = None
    for d in src_dirs:
        root = dataset_root_of(d)
        if root is None:
            continue
        cp = classes_path(root)
        if not cp.is_file():
            continue
        try:
            candidate = attribute_schema_digest(read_registry(cp), subject)
        except (OSError, RegistryError):
            candidate = None
        if candidate is not None:
            digest, src_classes = candidate, cp
            break

    by_split: dict[str, dict[str, dict[str, str]]] = {}
    for split_name, split_stems in parts.items():
        names = {Path(image_map[s]).name for s in split_stems if s in image_map}
        carried = {n: negatives[n] for n in sorted(set(negatives) & names)}
        if carried:
            by_split[split_name] = carried
    return _NegativeCarry(by_split=by_split, contradicted=contradicted, digest=digest,
                          src_classes=src_classes)


def _apply_negative_carry(carry: "_NegativeCarry | None", out_dir: Path,
                          subject: str | None) -> None:
    """Writes what :func:`_compute_negative_carry` found into each split's own status store. Each
    confirmation is copied whole, so the split records who confirmed the image and when, the source
    dataset's own answer, rather than re-attributing the human's work to the split writer."""
    if carry is None or not carry.by_split or not subject:
        return
    from tcip_mcp.class_registry import copy_registry
    from tcip_mcp.dataset_layout import (
        classes_path, replace_image_status_store, stamp_image_status_digests, status_bucket,
    )

    bucket_key = status_bucket(subject, None)
    for split_name, carried in carry.by_split.items():
        split_root = out_dir / split_name
        split_root.mkdir(parents=True, exist_ok=True)
        replace_image_status_store(split_root, {bucket_key: carried})
        if carry.digest is not None and carry.src_classes is not None:
            copy_registry(carry.src_classes, classes_path(split_root))
            stamp_image_status_digests(split_root, bucket_key, carried, carry.digest)
