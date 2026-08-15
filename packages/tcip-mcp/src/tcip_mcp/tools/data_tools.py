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


def _scan_dataset(root: str) -> dict:
    """Scan a directory tree for images and labels.

    Labels are the name-based per-image JSON (one file per image, all subjects) under
    ``annotations/<date>/`` (no detect/segment split), or a single assembled dataset-level COCO.
    """
    from tcip_mcp.dataset_layout import annotation_root, image_root, prediction_root
    from tcip_mcp.pipelines.resolution import SIDECAR_FILENAMES

    root_path = Path(root)
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    images: list[str] = []
    labels: list[str] = []
    preds: list[str] = []
    detected_format: str | None = None

    # Find images (recurse to catch the canonical images/<date>/ layout).
    images_dir = image_root(root_path)
    scan_root = images_dir if images_dir.is_dir() else root_path
    for f in sorted(scan_root.rglob("*")):
        if f.is_file() and f.suffix.lower() in image_exts:
            images.append(str(f))

    # Ground-truth labels: annotations/[<date>/]<stem>.json (one file per image, every subject).
    ann_dir = annotation_root(root_path)
    if ann_dir.is_dir():
        labels = [str(f) for f in sorted(ann_dir.rglob("*.json")) if f.is_file()]
        if labels:
            try:
                from tcip_annotation.format_io import detect_format
                detected_format = detect_format(labels[0])
            except ValueError:
                detected_format = None  # unrecognized: report nothing rather than a guess

    # A single COCO JSON at the dataset root
    for candidate in ("annotations.json", "labels.json", "instances.json"):
        coco_path = root_path / candidate
        if coco_path.is_file():
            try:
                from tcip_annotation.format_io import detect_format
                if detect_format(str(coco_path)) == "coco":
                    labels = [str(coco_path)]
                    detected_format = "coco"
            except ValueError:
                pass
            break

    # Predictions: predictions/<model>/[<date>/]<stem>.json; the bucket's stamps are not labels.
    pred_dir = prediction_root(root_path)
    if pred_dir.is_dir():
        preds = [str(f) for f in sorted(pred_dir.rglob("*.json"))
                 if f.is_file() and f.name not in SIDECAR_FILENAMES]

    return {"images": images, "labels": labels, "predictions": preds, "format": detected_format}


@mcp.tool()
@audited
def scan_dataset(folder_path: str) -> dict:
    """Scan a folder for images, labels, and predictions.

    Reads the name-based per-image JSON labels (one file per image, all subjects), or an assembled
    dataset-level COCO.

    Expects the canonical layout (see tcip_mcp.dataset_layout):
        images/<date>/  annotations/<date>/<stem>.json  predictions/<model>/<date>/<stem>.json

    Args:
        folder_path: Path to the dataset root directory.
    """
    if not Path(folder_path).is_dir():
        return {"error": f"Directory not found: {folder_path}"}

    scan = _scan_dataset(folder_path)

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
    }


@mcp.tool()
@audited
def validate_data_quality(folder_path: str) -> dict:
    """Run quality checks on a dataset (any supported annotation format).

    Checks: empty label files, and stem matching between images and labels (a label with no
    matching image, or for COCO, a referenced image file not found in the images dir). Class
    consistency against a subject registry and coordinate-range validation are not implemented.

    Args:
        folder_path: Path to the dataset root directory.
    """
    if not Path(folder_path).is_dir():
        return {"error": f"Directory not found: {folder_path}"}

    scan = _scan_dataset(folder_path)
    issues: list[dict] = []
    fmt = scan.get("format")

    image_stems = {Path(p).stem for p in scan["images"]}

    # Per-image labels: check stem matching (a dataset-level COCO has no per-stem file)
    if fmt != "coco":
        for label_path in scan["labels"]:
            stem = Path(label_path).stem
            if stem not in image_stems:
                issues.append({"level": "error", "file": label_path, "message": "No matching image"})

    # Format-specific validation: the subjects present, not numeric ids (labels are name-based now).
    subjects: set[str] = set()

    if fmt == "json":  # the name-based per-image label file
        from tcip_annotation import json_io
        for label_path in scan["labels"]:
            try:
                for a in json_io.read_annotations(label_path):
                    subjects.add(a.subject)
            except Exception as e:
                issues.append({"level": "error", "file": label_path, "message": f"JSON parse error: {e}"})
    elif fmt == "coco":
        from tcip_annotation.format_io import _parse_coco_json
        for label_path in scan["labels"]:
            try:
                coco = _parse_coco_json(label_path)
                for c in coco.get("categories", []):
                    if c.get("name"):
                        subjects.add(str(c["name"]))
                coco_fnames = {img.get("file_name", "") for img in coco.get("images", [])}
                for fn in coco_fnames:
                    if Path(fn).stem not in image_stems:
                        issues.append({"level": "warning", "file": label_path, "message": f"COCO image '{fn}' not found in images dir"})
            except Exception as e:
                issues.append({"level": "error", "file": label_path, "message": f"COCO parse error: {e}"})
    if fmt != "coco":
        for label_path in scan["labels"]:
            if os.path.getsize(label_path) == 0:
                issues.append({"level": "warning", "file": label_path, "message": "Empty label file"})

    return {
        "path": folder_path,
        "format": fmt,
        "total_images": len(scan["images"]),
        "total_labels": len(scan["labels"]),
        "subjects": sorted(subjects),
        "issues": issues,
        "issue_count": len(issues),
        "is_valid": all(i["level"] != "error" for i in issues),
    }


@mcp.tool()
@audited
def make_splits(
    folder_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
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
    """Compute a leakage-free, annotation-stratified train/val/test split.

    Non-destructive by default: emits ``{train,val,test}.json`` stem manifests plus a stats
    dict. Sibling tiles of one source image are kept in the same split (no tree-/canopy-level
    leakage), and, when ``stratify_foreground`` is set, splits are balanced by annotation
    count so dense and sparse sources are proportionally represented.

    With ``materialize=True`` it additionally lays out a
    ``{train,val,test}/{images,labels}/`` tree under ``output_path`` (defaulting to
    ``folder_path/splits``), copying (or symlinking, ``copy_files=False``) each stem's image and
    label, and adds ``output_dir`` / ``structure`` to the return.

    Args:
        folder_path: Path to the dataset root directory.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
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
        materialize: Also copy/symlink files into a {train,val,test}/{images,labels}/ tree.
        copy_files: Copy files (True) or create symlinks (False) when materializing.
        subject: The object the confirmed negatives are keyed under. A materialized split tree
            can't recover the subject from its own path, so an image a human confirmed negative
            is silently dropped from the materialized set (reads as an unconfirmed empty label)
            unless this is passed. Only relevant with `materialize=True`.
        spatial: Derive a within-image train/val/test split over one source's own tile lattice
            (:func:`~tcip_mcp.pipelines.data.splits.spatial_strip_split`) instead of grouping
            whole source images. Requires ``tile_size``/``overlap`` and a folder holding exactly
            one labeled image; refuses otherwise, naming a training run's own automatic route
            (``data.tiling`` in the run config, which derives this the same way when a run's
            dataset turns out to be single-source) as the alternative for the common multi-source
            case. Ignores ``group_by``/``group_key_map``/``stratify_foreground``/``copy_files``/
            ``subject``, and is not compatible with ``materialize`` (there is one source image,
            not a set of files to lay out into a split tree); ``train_ratio``/``val_ratio``/
            ``test_ratio`` still apply, a zero ratio drops that side entirely.
        tile_size: Tile edge in pixels, required when ``spatial=True``.
        overlap: Tile overlap fraction, required when ``spatial=True``.
        buffer: Minimum pixel gap kept at every boundary between two differently-assigned
            regions, ``spatial=True`` only. Defaults to ``tile_size`` (see
            :func:`spatial_strip_split`).
    """
    if spatial:
        return _make_spatial_split(
            folder_path, tile_size=tile_size, overlap=overlap,
            fractions=(train_ratio, val_ratio, test_ratio),
            seed=seed, buffer=buffer, output_path=output_path, materialize=materialize,
        )

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        return {"error": "Ratios must sum to 1.0"}
    if not Path(folder_path).is_dir():
        return {"error": f"Directory not found: {folder_path}"}

    from tcip_mcp.pipelines.data.splits import (
        group_balanced_split,
        count_label_lines,
        resolve_group_key_fn,
    )

    scan = _scan_dataset(folder_path)
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
        annotation_counts = {
            s: count_label_lines(Path(label_map[s]).parent, s) for s in stems
        }

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
    if label_map:
        from tcip_mcp.pipelines.resolution import dataset_hash as _dataset_hash
        labels_root = Path(next(iter(label_map.values()))).parent
        dataset_hash = _dataset_hash(labels_root, stems=stems)

    counts = annotation_counts or {}
    out_dir = Path(output_path) if output_path else (Path(folder_path) / "splits" if materialize else None)
    manifest_dir = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_stems in parts.items():
            tcip_store.replace(split_stem_list_key(out_dir, split_name), sorted(split_stems))
        tcip_store.replace(split_manifest_key(out_dir), {
            "seed": seed, "dataset_hash": dataset_hash, "group_by": resolved_group_by,
            "splits": {k: sorted(v) for k, v in parts.items()}})
        manifest_dir = str(out_dir)

    result = {
        "splits": {k: len(v) for k, v in parts.items()},
        "foreground_annotations": {
            k: sum(int(counts.get(s, 0)) for s in v) for k, v in parts.items()
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
        # Lay out a {train,val,test}/{images,labels}/ tree from the split assignment.
        place_fn = shutil.copy2 if copy_files else os.symlink
        for split_name, split_stems in parts.items():
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
        _carry_confirmed_negatives(label_map, out_dir, parts, image_map, subject)
        result["output_dir"] = str(out_dir)
        result["structure"] = f"{out_dir}/{{train,val,test}}/{{images,labels}}/"

    return result


def _make_spatial_split(
    folder_path: str, *, tile_size: int | None, overlap: float | None,
    fractions: tuple[float, float, float], seed: int, buffer: int | None,
    output_path: str | None, materialize: bool,
) -> dict:
    """``make_splits(spatial=True)``'s body: a within-image train/val/test split for one source.

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

    from tcip_mcp.pipelines.data.splits import image_extent_from_labels, spatial_strip_split
    from tcip_mcp.pipelines.data.tiling import tile_positions, tile_within_extent

    scan = _scan_dataset(folder_path)
    image_map = {Path(p).stem: p for p in scan["images"]}
    label_map = {Path(p).stem: p for p in scan["labels"]}
    stems = sorted(set(image_map) & set(label_map))
    if len(stems) != 1:
        return {"error": f"spatial=True requires a single-stem folder ({len(stems)} found); use "
                         "the non-spatial grouped split for a multi-source folder, or a training "
                         "run's own automatic route."}
    stem = stems[0]
    labels_root = Path(label_map[stem]).parent
    extent = image_extent_from_labels(labels_root, stem)
    if extent is None:
        return {"error": f"{stem}'s label file carries no width/height; cannot derive a spatial "
                         "split without the image extent."}
    width, height = extent

    try:
        split = spatial_strip_split(
            width, height, tile_size, overlap, fractions=fractions, seed=seed, buffer=buffer,
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

    out_dir = Path(output_path) if output_path else Path(folder_path) / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in split.regions:
        tcip_store.replace(split_stem_list_key(out_dir, name), sorted(ids[name]))
    tcip_store.replace(split_manifest_key(out_dir), {
        "seed": seed, "group_by": "spatial_strip", "spatial": spatial_manifest})

    return {
        "splits": {name: len(ids[name]) for name in split.regions},
        "seed": seed,
        "group_by": "spatial_strip",
        "spatial": spatial_manifest,
        "manifest_dir": str(out_dir),
    }


def _carry_confirmed_negatives(label_map: dict, out_dir: Path, parts: dict,
                               image_map: dict, subject: str | None) -> None:
    """Copy the source subject's confirmed negatives into each split's own status store.

    A split tree is ``{train,val,test}/labels`` by construction and cannot recover the subject from
    its path, so the confirmations are carried explicitly under the threaded ``subject`` (keyed by
    ``status_bucket(subject, None)``, since the split carries no date). Without this, every image a
    human confirmed negative reads as an unconfirmed empty in the split and is dropped from training.
    No subject threaded -> nothing to attribute the confirmations to, so none are carried.
    """
    if not subject:
        return
    from tcip_mcp.class_registry import (
        RegistryError, attribute_schema_digest, copy_registry, read_registry,
    )
    from tcip_mcp.dataset_layout import (
        CONFIRMED_NEGATIVE, annotation_date, classes_path, dataset_root_of,
        replace_image_status_store, stamp_image_status_digests, status_bucket,
    )
    from tcip_mcp.pipelines.data.datasets import confirmed_negative_names

    src_dirs = {Path(p).parent for p in label_map.values()}
    if not src_dirs:
        return
    negatives: set[str] = set()
    for d in src_dirs:
        negatives |= confirmed_negative_names(d, subject=subject, date=annotation_date(d))
    if not negatives:
        return

    # Resolve one source dataset's registry (best-effort) to carry a fresh per-image schema stamp
    # alongside the negatives: without this, a split tree has no classes.json to compare against
    # and quarantine can never fire on it (a permanent no-op, not "admit until proven stale").
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

    bucket_key = status_bucket(subject, None)
    for split_name, split_stems in parts.items():
        names = {Path(image_map[s]).name for s in split_stems if s in image_map}
        carried = {n: CONFIRMED_NEGATIVE for n in sorted(negatives & names)}
        if not carried:
            continue
        split_root = out_dir / split_name
        split_root.mkdir(parents=True, exist_ok=True)
        replace_image_status_store(split_root, {bucket_key: carried})
        if digest is not None and src_classes is not None:
            copy_registry(src_classes, classes_path(split_root))
            stamp_image_status_digests(split_root, bucket_key, carried, digest)
