"""Data management tools — load datasets, validate quality, split data."""

from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited


def _scan_dataset(root: str) -> dict:
    """Scan a directory tree for images and labels in any supported format.

    Detects YOLO (.txt), PASCAL VOC (.xml), COCO (.json), and LabelMe (.json).
    """
    root_path = Path(root)
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    label_exts = {".txt", ".xml", ".json"}
    images: list[str] = []
    labels_detect: list[str] = []
    labels_segment: list[str] = []
    preds_detect: list[str] = []
    preds_segment: list[str] = []
    detected_format: str = "yolo"

    # Find images (recurse to catch the canonical images/<date>/ layout).
    images_dir = root_path / "images"
    scan_root = images_dir if images_dir.is_dir() else root_path
    for f in sorted(scan_root.rglob("*")):
        if f.is_file() and f.suffix.lower() in image_exts:
            images.append(str(f))

    # Ground-truth labels: annotations/<trait>/[<date>/]<task>/*.{txt,xml,json}
    ann_dir = root_path / "annotations"
    if ann_dir.is_dir():
        for f in sorted(ann_dir.rglob("*")):
            if not (f.is_file() and f.suffix in label_exts):
                continue
            if f.parent.name == "detect":
                labels_detect.append(str(f))
            elif f.parent.name == "segment":
                labels_segment.append(str(f))
        if labels_detect:
            # Auto-detect format from the first label file
            try:
                from tcip_annotation.format_io import detect_format
                detected_format = detect_format(labels_detect[0])
            except Exception:
                pass

    # A single COCO JSON at the dataset root
    for candidate in ("annotations.json", "labels.json", "instances.json"):
        coco_path = root_path / candidate
        if coco_path.is_file():
            try:
                from tcip_annotation.format_io import detect_format
                if detect_format(str(coco_path)) == "coco":
                    labels_detect = [str(coco_path)]
                    detected_format = "coco"
            except Exception:
                pass
            break

    # Predictions: predictions/<model>/[<date>/]<task>/*.{txt,xml,json}
    pred_dir = root_path / "predictions"
    if pred_dir.is_dir():
        for f in sorted(pred_dir.rglob("*")):
            if not (f.is_file() and f.suffix in label_exts):
                continue
            if f.parent.name == "detect":
                preds_detect.append(str(f))
            elif f.parent.name == "segment":
                preds_segment.append(str(f))

    return {
        "images": images,
        "labels_detect": labels_detect,
        "labels_segment": labels_segment,
        "predictions_detect": preds_detect,
        "predictions_segment": preds_segment,
        "format": detected_format,
    }


@mcp.tool()
@audited
def load_dataset(folder_path: str) -> dict:
    """Scan a folder for images, labels, and predictions.

    Supports YOLO (.txt), PASCAL VOC (.xml), COCO (.json), and LabelMe (.json).
    Format is auto-detected from file extensions and content.

    Expects the canonical layout (see tcip_mcp.dataset_layout):
        images/<date>/  annotations/<trait>/<date>/{detect,segment}/
        predictions/<model>/<date>/{detect,segment}/

    Args:
        folder_path: Path to the dataset root directory.
    """
    if not Path(folder_path).is_dir():
        return {"error": f"Directory not found: {folder_path}"}

    scan = _scan_dataset(folder_path)

    # Build stem-based pairing
    image_stems = {Path(p).stem: p for p in scan["images"]}
    label_stems = {Path(p).stem: p for p in scan["labels_detect"]}

    paired = 0
    unlabelled = 0
    for stem in image_stems:
        if stem in label_stems:
            paired += 1
        else:
            unlabelled += 1

    return {
        "path": folder_path,
        "format": scan.get("format", "yolo"),
        "image_count": len(scan["images"]),
        "labels_detect_count": len(scan["labels_detect"]),
        "labels_segment_count": len(scan["labels_segment"]),
        "predictions_detect_count": len(scan["predictions_detect"]),
        "predictions_segment_count": len(scan["predictions_segment"]),
        "paired_images": paired,
        "unlabelled_images": unlabelled,
        "image_stems_sample": sorted(image_stems.keys())[:10],
    }


@mcp.tool()
@audited
def validate_data_quality(folder_path: str) -> dict:
    """Run quality checks on a dataset (any supported annotation format).

    Checks: empty labels, missing images, class consistency, coordinate ranges.
    Auto-detects format (YOLO, COCO, PASCAL VOC, LabelMe).

    Args:
        folder_path: Path to the dataset root directory.
    """
    if not Path(folder_path).is_dir():
        return {"error": f"Directory not found: {folder_path}"}

    scan = _scan_dataset(folder_path)
    issues: list[dict] = []
    fmt = scan.get("format", "yolo")

    image_stems = {Path(p).stem for p in scan["images"]}

    # For per-file formats (yolo, voc, labelme), check stem matching
    if fmt != "coco":
        for label_path in scan["labels_detect"]:
            stem = Path(label_path).stem
            if stem not in image_stems:
                issues.append({"level": "error", "file": label_path, "message": "No matching image"})

    # Format-specific validation
    class_ids: set[int] = set()

    if fmt == "yolo":
        for label_path in scan["labels_detect"]:
            with open(label_path, "r") as f:
                for line_no, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) == 0:
                        continue
                    if len(parts) != 5:
                        issues.append(
                            {"level": "error", "file": label_path, "line": line_no,
                             "message": f"Expected 5 fields, got {len(parts)}"}
                        )
                        continue
                    try:
                        cid = int(parts[0])
                        vals = [float(v) for v in parts[1:]]
                        class_ids.add(cid)
                        for v in vals:
                            if v < 0 or v > 1:
                                issues.append(
                                    {"level": "warning", "file": label_path, "line": line_no,
                                     "message": f"Coordinate out of [0,1] range: {v:.4f}"}
                                )
                    except ValueError:
                        issues.append(
                            {"level": "error", "file": label_path, "line": line_no,
                             "message": "Non-numeric field"}
                        )
    elif fmt == "voc":
        from tcip_annotation.format_io import parse_voc_detect
        for label_path in scan["labels_detect"]:
            try:
                boxes, cids, _ = parse_voc_detect(label_path)
                class_ids.update(cids)
            except Exception as e:
                issues.append({"level": "error", "file": label_path, "message": f"VOC parse error: {e}"})
    elif fmt == "coco":
        from tcip_annotation.format_io import _parse_coco_json
        for label_path in scan["labels_detect"]:
            try:
                coco = _parse_coco_json(label_path)
                for ann in coco.get("annotations", []):
                    cid = ann.get("category_id", 0)
                    class_ids.add(cid)
                # Check all annotated images have files
                coco_fnames = {img.get("file_name", "") for img in coco.get("images", [])}
                for fn in coco_fnames:
                    stem = Path(fn).stem
                    if stem not in image_stems:
                        issues.append({"level": "warning", "file": label_path, "message": f"COCO image '{fn}' not found in images dir"})
            except Exception as e:
                issues.append({"level": "error", "file": label_path, "message": f"COCO parse error: {e}"})
    elif fmt == "labelme":
        from tcip_annotation.format_io import parse_labelme_detect
        for label_path in scan["labels_detect"]:
            try:
                boxes, cids, _ = parse_labelme_detect(label_path)
                class_ids.update(cids)
            except Exception as e:
                issues.append({"level": "error", "file": label_path, "message": f"LabelMe parse error: {e}"})

    # Check for empty label files (per-file formats only)
    if fmt != "coco":
        for label_path in scan["labels_detect"]:
            if os.path.getsize(label_path) == 0:
                issues.append({"level": "warning", "file": label_path, "message": "Empty label file"})

    return {
        "path": folder_path,
        "format": fmt,
        "total_images": len(scan["images"]),
        "total_labels": len(scan["labels_detect"]),
        "class_ids": sorted(class_ids),
        "issues": issues,
        "issue_count": len(issues),
        "is_valid": all(i["level"] != "error" for i in issues),
    }


@mcp.tool()
@audited
def split_dataset(
    folder_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    output_path: str | None = None,
    stratified: bool = False,
    copy_files: bool = True,
) -> dict:
    """Split dataset into train/val/test and create YOLO directory structure.

    Creates output_path/{train,val,test}/{images,labels}/ with actual file
    copies (or symlinks if copy_files=False). Also writes split manifest JSONs.

    When stratified=True, splits are balanced by the primary class in each label
    file so rare classes are proportionally represented in every split.

    Args:
        folder_path: Path to the dataset root directory.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.
        output_path: Directory for split output (defaults to folder_path/splits).
        stratified: Use stratified sampling by primary class.
        copy_files: Copy files (True) or create symlinks (False).
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        return {"error": "Ratios must sum to 1.0"}

    scan = _scan_dataset(folder_path)
    root = Path(folder_path)

    # Build stem → file path maps
    image_map: dict[str, str] = {}
    for p in scan["images"]:
        image_map[Path(p).stem] = p
    label_map: dict[str, str] = {}
    for p in scan["labels_detect"]:
        label_map[Path(p).stem] = p

    # Only split stems that have both image and label
    paired_stems = sorted(set(image_map) & set(label_map))
    if not paired_stems:
        return {"error": "No paired image+label files found"}

    rng = random.Random(seed)

    if stratified:
        # Determine primary class per stem (most frequent class in label file)
        stem_classes: dict[str, int] = {}
        for stem in paired_stems:
            counter: Counter[int] = Counter()
            with open(label_map[stem], "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            counter[int(parts[0])] += 1
                        except ValueError:
                            pass
            stem_classes[stem] = counter.most_common(1)[0][0] if counter else -1

        # Group by class
        class_groups: dict[int, list[str]] = {}
        for stem, cid in stem_classes.items():
            class_groups.setdefault(cid, []).append(stem)

        # Stratified split per class
        splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        for cid in sorted(class_groups):
            group = class_groups[cid]
            rng.shuffle(group)
            n = len(group)
            n_train = max(1, int(n * train_ratio)) if n >= 3 else n
            n_val = int(n * val_ratio) if n >= 3 else 0
            splits["train"].extend(group[:n_train])
            splits["val"].extend(group[n_train : n_train + n_val])
            splits["test"].extend(group[n_train + n_val :])
    else:
        rng.shuffle(paired_stems)
        n = len(paired_stems)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits = {
            "train": paired_stems[:n_train],
            "val": paired_stems[n_train : n_train + n_val],
            "test": paired_stems[n_train + n_val :],
        }

    out_dir = Path(output_path) if output_path else root / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create YOLO directory structure and copy/link files
    place_fn = shutil.copy2 if copy_files else os.symlink
    for split_name, stems in splits.items():
        img_dir = out_dir / split_name / "images"
        lbl_dir = out_dir / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            src_img = Path(image_map[stem])
            src_lbl = Path(label_map[stem])
            dst_img = img_dir / src_img.name
            dst_lbl = lbl_dir / src_lbl.name
            if not dst_img.exists():
                place_fn(str(src_img), str(dst_img))
            if not dst_lbl.exists():
                place_fn(str(src_lbl), str(dst_lbl))

    # Write manifest JSONs for reference
    for split_name, stems in splits.items():
        out_file = out_dir / f"{split_name}.json"
        with open(out_file, "w") as f:
            json.dump(sorted(stems), f, indent=2)

    return {
        "splits": {k: len(v) for k, v in splits.items()},
        "total": len(paired_stems),
        "seed": seed,
        "stratified": stratified,
        "output_dir": str(out_dir),
        "structure": f"{out_dir}/{{train,val,test}}/{{images,labels}}/",
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
    stratify_foreground: bool = True,
    output_path: str | None = None,
) -> dict:
    """Compute a leakage-free, annotation-stratified train/val/test split.

    Unlike ``split_dataset`` (which copies/symlinks files into a YOLO directory
    tree), this is a lightweight, non-destructive complement: it only emits
    ``{train,val,test}.json`` stem manifests plus a stats dict. Sibling tiles of
    one source image are kept in the same split (no tree-/canopy-level leakage),
    and — when ``stratify_foreground`` is set — splits are balanced by annotation
    count so dense and sparse sources are proportionally represented.

    Args:
        folder_path: Path to the dataset root directory.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.
        group_by: Group selector — ``"tile_prefix"`` (strip a trailing
            ``_<x>_<y>`` tile offset) or ``"stem"`` (one group per image).
        stratify_foreground: Balance splits by foreground annotation count.
        output_path: When set, write ``{train,val,test}.json`` manifests there.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        return {"error": "Ratios must sum to 1.0"}
    if not Path(folder_path).is_dir():
        return {"error": f"Directory not found: {folder_path}"}

    from tcip_mcp.pipelines.data.splits import (
        group_balanced_split,
        count_lines,
        GROUP_KEY_FNS,
        default_group_key,
    )

    scan = _scan_dataset(folder_path)
    image_map = {Path(p).stem: p for p in scan["images"]}
    label_map = {Path(p).stem: p for p in scan["labels_detect"]}

    stratified = bool(stratify_foreground and label_map)
    if stratified:
        stems = sorted(set(image_map) & set(label_map))
    else:
        stems = sorted(image_map)
    if not stems:
        return {"error": "No images found to split"}

    annotation_counts = None
    if stratified:
        annotation_counts = {s: count_lines(label_map[s]) for s in stems}

    group_key_fn = GROUP_KEY_FNS.get(group_by, default_group_key)
    parts = group_balanced_split(
        stems,
        annotation_counts=annotation_counts,
        group_key_fn=group_key_fn,
        splits=(train_ratio, val_ratio, test_ratio),
        seed=seed,
    )

    counts = annotation_counts or {}
    manifest_dir = None
    if output_path:
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        for split_name, split_stems in parts.items():
            with open(out / f"{split_name}.json", "w") as f:
                json.dump(sorted(split_stems), f, indent=2)
        manifest_dir = str(out)

    return {
        "splits": {k: len(v) for k, v in parts.items()},
        "foreground_annotations": {
            k: sum(int(counts.get(s, 0)) for s in v) for k, v in parts.items()
        },
        "total_stems": len(stems),
        "total_annotations": sum(int(v) for v in counts.values()),
        "groups": len({group_key_fn(s) for s in stems}),
        "seed": seed,
        "group_by": group_by,
        "stratified": stratified,
        "manifest_dir": manifest_dir,
    }
