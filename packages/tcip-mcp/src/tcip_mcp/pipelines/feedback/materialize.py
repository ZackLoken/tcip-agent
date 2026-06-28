"""Materialize a curated YOLO detection dataset from human review verdicts (W5).

Pure / torch-free. Turns ``review_stats.json`` verdicts back into training data:
  - accepted / edited GT boxes  -> positive YOLO label files
  - rejected-only images        -> empty-label hard-negative backgrounds
plus a ``curated_manifest.json`` for provenance. The output layout (``images/`` +
``labels/detect/``) matches ``data_tools._scan_dataset`` so the loop chains straight
into ``split_dataset`` / ``launch_training`` with no glue.

Because the verdict log stores normalized YOLO center-form boxes
(``[cx, cy, w, h]``), positives are reconstructed directly — no inference re-run and
no image dimensions needed.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from tcip_annotation.label_io import write_detect_labels
from tcip_annotation.state import BBox

_POSITIVE_ACTIONS = {"accepted", "edited"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def partition_review_verdicts(review_state: dict, *, only_completed: bool = False) -> dict[str, dict]:
    """Partition per-image review verdicts into positives / hard-negatives / skip.

    Returns ``{img_name: {"positives": [(class_id, cx, cy, w, h)],
    "rejected_count": int, "status": "positive"|"hard_negative"|"skip"}}``.
    A detection's box is ``gt_bbox_norm or pred_bbox_norm`` (the fallback handles
    accepted-FP entries that carry only a predicted box).
    """
    result: dict[str, dict] = {}
    for img_name, img_data in review_state.get("image", {}).items():
        if only_completed and img_data.get("img_status") != "completed":
            continue
        positives: list[tuple] = []
        rejected = 0
        for entry in img_data.get("detections", []):
            action = entry.get("action")
            if action in _POSITIVE_ACTIONS:
                box = entry.get("gt_bbox_norm") or entry.get("pred_bbox_norm")
                if box and len(box) == 4:
                    cx, cy, w, h = (float(v) for v in box)
                    positives.append((int(entry.get("class_id", 0)), cx, cy, w, h))
            elif action == "rejected":
                rejected += 1
        status = "positive" if positives else ("hard_negative" if rejected else "skip")
        result[img_name] = {"positives": positives, "rejected_count": rejected, "status": status}
    return result


def _find_source_image(source_images_dir: str, img_name: str) -> Path | None:
    direct = Path(source_images_dir) / img_name
    if direct.is_file():
        return direct
    stem = Path(img_name).stem
    for ext in IMAGE_EXTS:
        cand = Path(source_images_dir) / f"{stem}{ext}"
        if cand.is_file():
            return cand
    return None


def _write_positive_label(path: Path, positives: list[tuple]) -> None:
    # Unit scale: write_detect_labels divides by img_w=img_h=1, so the normalized
    # [cx,cy,w,h] round-trips through its atomic write + 6-decimal formatter.
    boxes = [BBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cid)
             for (cid, cx, cy, w, h) in positives]
    write_detect_labels(str(path), boxes, img_w=1, img_h=1)


def materialize_dataset(
    review_state: dict,
    source_images_dir: str,
    output_dir: str,
    *,
    review_state_path: str = "",
    include_hard_negatives: bool = True,
    copy_files: bool = True,
    only_completed: bool = False,
) -> dict:
    """Write ``output_dir/images/`` + ``output_dir/labels/detect/`` + manifest."""
    partition = partition_review_verdicts(review_state, only_completed=only_completed)
    out = Path(output_dir)
    images_out = out / "images"
    labels_out = out / "labels" / "detect"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    place = shutil.copy2 if copy_files else os.symlink
    counts = {"positive": 0, "hard_negative": 0, "skipped": 0, "total_boxes": 0, "missing_images": 0}
    class_ids: set[int] = set()
    manifest_images: list[dict] = []

    for img_name, info in partition.items():
        status = info["status"]
        if status == "skip" or (status == "hard_negative" and not include_hard_negatives):
            counts["skipped"] += 1
            continue
        src = _find_source_image(source_images_dir, img_name)
        if src is None:
            counts["missing_images"] += 1
            continue

        dst_img = images_out / src.name
        if not dst_img.exists():
            place(str(src), str(dst_img))
        label_path = labels_out / f"{src.stem}.txt"

        if status == "positive":
            _write_positive_label(label_path, info["positives"])
            counts["positive"] += 1
            counts["total_boxes"] += len(info["positives"])
            class_ids.update(cid for (cid, *_rest) in info["positives"])
        else:  # hard_negative -> empty label (write 0 bytes directly; the writer deletes empties)
            label_path.write_text("")
            counts["hard_negative"] += 1

        manifest_images.append({
            "image": src.name, "status": status, "n_boxes": len(info["positives"]),
            "rejected_count": info["rejected_count"], "label": str(label_path),
        })

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "review_state": review_state_path,
        "source_images_dir": str(source_images_dir),
        "output_dir": str(out),
        "counts": counts,
        "class_ids": sorted(class_ids),
        "images": manifest_images,
    }
    (out / "curated_manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        **counts,
        "class_ids": sorted(class_ids),
        "output_dir": str(out),
        "structure": f"{out}/images/ + {out}/labels/detect/",
        "manifest": str(out / "curated_manifest.json"),
    }


def reviewed_image_names(review_state: dict) -> set[str]:
    """Image names whose ``img_status == 'completed'`` (same predicate as is_image_reviewed)."""
    return {name for name, d in review_state.get("image", {}).items()
            if d.get("img_status") == "completed"}


def select_unreviewed(image_paths: list[str], reviewed_names: set[str]) -> list[str]:
    """``image_paths`` whose basename is not in ``reviewed_names`` (order-preserving)."""
    return [p for p in image_paths if os.path.basename(p) not in reviewed_names]
