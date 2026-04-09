"""Data management tools — load datasets, validate quality, split data."""

from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

from tcip_mcp.server import mcp


def _scan_yolo_dataset(root: str) -> dict:
    """Scan a directory tree for YOLO-format images and labels."""
    root_path = Path(root)
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    images: list[str] = []
    labels_detect: list[str] = []
    labels_segment: list[str] = []
    preds_detect: list[str] = []
    preds_segment: list[str] = []

    # Find images
    images_dir = root_path / "images"
    if images_dir.is_dir():
        for f in sorted(images_dir.iterdir()):
            if f.suffix.lower() in image_exts:
                images.append(str(f))
    else:
        for f in sorted(root_path.iterdir()):
            if f.suffix.lower() in image_exts:
                images.append(str(f))

    # Find labels
    for sub in ("labels/detect", "labels"):
        d = root_path / sub
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.suffix == ".txt":
                    labels_detect.append(str(f))
            break

    for sub in ("labels/segment",):
        d = root_path / sub
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.suffix == ".txt":
                    labels_segment.append(str(f))

    # Find predictions
    for sub in ("predictions/detect",):
        d = root_path / sub
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.suffix == ".txt":
                    preds_detect.append(str(f))

    for sub in ("predictions/segment",):
        d = root_path / sub
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.suffix == ".txt":
                    preds_segment.append(str(f))

    return {
        "images": images,
        "labels_detect": labels_detect,
        "labels_segment": labels_segment,
        "predictions_detect": preds_detect,
        "predictions_segment": preds_segment,
    }


@mcp.tool()
def load_dataset(folder_path: str) -> dict:
    """Scan a folder for YOLO-format images, labels, and predictions.

    Expects structure:
        folder/images/  folder/labels/detect/  folder/predictions/detect/

    Args:
        folder_path: Path to the dataset root directory.
    """
    if not Path(folder_path).is_dir():
        return {"error": f"Directory not found: {folder_path}"}

    scan = _scan_yolo_dataset(folder_path)

    # Build stem-based pairing
    image_stems = {Path(p).stem: p for p in scan["images"]}
    label_stems = {Path(p).stem: p for p in scan["labels_detect"]}
    pred_stems = {Path(p).stem: p for p in scan["predictions_detect"]}

    paired = 0
    unlabelled = 0
    for stem in image_stems:
        if stem in label_stems:
            paired += 1
        else:
            unlabelled += 1

    return {
        "path": folder_path,
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
def validate_data_quality(folder_path: str) -> dict:
    """Run quality checks on a YOLO dataset.

    Checks: empty labels, missing images, class consistency, coordinate ranges.

    Args:
        folder_path: Path to the dataset root directory.
    """
    scan = _scan_yolo_dataset(folder_path)
    issues: list[dict] = []

    image_stems = {Path(p).stem for p in scan["images"]}

    # Check labels have matching images
    for label_path in scan["labels_detect"]:
        stem = Path(label_path).stem
        if stem not in image_stems:
            issues.append({"level": "error", "file": label_path, "message": "No matching image"})

    # Check label format
    class_ids: set[int] = set()
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

    # Check for empty label files
    for label_path in scan["labels_detect"]:
        if os.path.getsize(label_path) == 0:
            issues.append({"level": "warning", "file": label_path, "message": "Empty label file"})

    return {
        "path": folder_path,
        "total_images": len(scan["images"]),
        "total_labels": len(scan["labels_detect"]),
        "class_ids": sorted(class_ids),
        "issues": issues,
        "issue_count": len(issues),
        "is_valid": all(i["level"] != "error" for i in issues),
    }


@mcp.tool()
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

    scan = _scan_yolo_dataset(folder_path)
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
