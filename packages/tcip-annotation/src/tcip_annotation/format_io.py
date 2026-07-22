"""Annotation I/O for the two on-disk formats — the canonical per-image JSON and the COCO
assembled from it.

The internal representation (BBox, Polygon, PredBBox, PredPolygon) is always pixel-coordinate;
format only matters at file I/O boundaries.

  - json  — one ``.json`` per image (the canonical json_io schema, an ``objects`` key)
  - coco  — a single dataset-level ``.json`` (an ``images``/``annotations`` key)

Usage:
    boxes, class_ids = load_annotations(path, img_w, img_h, task="detect")
    save_annotations(path, boxes, img_w, img_h, task="detect", fmt="json")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from tcip_annotation.state import BBox, Polygon

AnnotFormat = Literal["coco", "json"]
Task = Literal["detect", "segment"]


# ── Format detection ────────────────────────────────────────────────────────


def detect_format(path: str) -> AnnotFormat:
    """The annotation format of a file or directory, from its own contents.

    ``"json"`` is the canonical per-image label file (``json_io`` schema, keyed on ``objects``);
    ``"coco"`` is an assembled dataset-level COCO (keyed on ``images``/``annotations``). Raises for
    anything else rather than guessing — a misdetected format reads real annotations as empty
    negatives, so a wrong answer here is worse than no answer.
    """
    p = Path(path)
    candidates = sorted(p.glob("*.json")) if p.is_dir() else [p]
    for candidate in candidates:
        fmt = _detect_json_format(candidate)
        if fmt is not None:
            return fmt
    raise ValueError(
        f"Cannot determine the annotation format of {path}: expected the canonical per-image JSON "
        f"(an 'objects' key) or an assembled COCO (an 'images'/'annotations' key)."
    )


def _detect_json_format(path: Path) -> AnnotFormat | None:
    """``"json"`` / ``"coco"`` from a file's keys, or ``None`` if it is neither."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "objects" in data:
                return "json"  # the canonical per-image label file (json_io schema)
            if "images" in data or "annotations" in data:
                return "coco"
    except (json.JSONDecodeError, OSError):
        pass
    return None


# ── COCO JSON parsing ──────────────────────────────────────────────────────


def _parse_coco_json(path: str) -> dict:
    """Load and return a COCO-format JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _coco_image_annotations(
    coco: dict, image_id: int | None = None, file_name: str | None = None,
) -> tuple[list[dict], int, int]:
    """Extract annotations for a single image from COCO dict.

    Returns (annotations_list, img_width, img_height).
    """
    # Find image record
    img_record = None
    for img in coco.get("images", []):
        if image_id is not None and img["id"] == image_id:
            img_record = img
            break
        if file_name is not None and img.get("file_name") == file_name:
            img_record = img
            break
    if img_record is None:
        return [], 0, 0

    img_id = img_record["id"]
    w = img_record.get("width", 0)
    h = img_record.get("height", 0)

    anns = [a for a in coco.get("annotations", []) if a.get("image_id") == img_id]
    return anns, w, h


def parse_coco_detect(
    coco: dict, image_id: int | None = None, file_name: str | None = None,
) -> tuple[list[BBox], set[int]]:
    """Parse COCO detection annotations for one image into BBox objects.

    COCO bbox format: [x, y, width, height] in pixel coordinates.
    """
    anns, _, _ = _coco_image_annotations(coco, image_id, file_name)
    boxes: list[BBox] = []
    class_ids: set[int] = set()
    for ann in anns:
        bbox = ann.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue
        x, y, bw, bh = bbox
        cid = ann.get("category_id", 0)
        boxes.append(BBox(x, y, x + bw, y + bh, cid, **_coco_prov(ann)))
        class_ids.add(cid)
    return boxes, class_ids


def _coco_prov(ann: dict) -> dict:
    """Provenance extension keys of a COCO annotation record, when present."""
    return {k: ann[k] for k in ("created_by", "created_at", "accepted_by", "accepted_at")
            if ann.get(k)}


def parse_coco_segment(
    coco: dict, image_id: int | None = None, file_name: str | None = None,
) -> tuple[list[Polygon], set[int]]:
    """Parse COCO segmentation annotations for one image into Polygon objects.

    COCO segmentation format: list of [x1,y1,x2,y2,...] polygon arrays (pixel coords).
    All polygon parts of an annotation are kept (multi-part / disjoint masks); RLE
    segmentations (a dict, not a list) are skipped (no RLE support yet).
    """
    anns, _, _ = _coco_image_annotations(coco, image_id, file_name)
    polygons: list[Polygon] = []
    class_ids: set[int] = set()
    for ann in anns:
        segs = ann.get("segmentation")
        if not segs or not isinstance(segs, list):
            continue  # missing, or RLE (dict)
        cid = ann.get("category_id", 0)
        for coords in segs:  # every polygon part, not just the first
            if not isinstance(coords, list) or len(coords) < 6:
                continue
            points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            polygons.append(Polygon(points, cid, **_coco_prov(ann)))
            class_ids.add(cid)
    return polygons, class_ids


# ── COCO JSON writing ──────────────────────────────────────────────────────


def _emit_coco_extras(rec: dict, shape) -> None:
    """Carry provenance (and a Pred shape's score) into a COCO annotation record.

    Extension keys, same names as the canonical per-image JSON — without them a GT
    round-trip through dataset-COCO export silently strips created_by/accepted_by.
    """
    for key in ("created_by", "created_at", "accepted_by", "accepted_at"):
        val = getattr(shape, key, None)
        if val:
            rec[key] = val
    conf = getattr(shape, "confidence", None)
    if conf is not None:
        rec["score"] = float(conf)


def write_coco_detect(
    path: str,
    images_annotations: dict[str, tuple[list[BBox], int, int]],
    categories: list[dict] | None = None,
) -> None:
    """Write detection annotations to a COCO JSON file.

    Args:
        path: Output JSON file path.
        images_annotations: Dict mapping file_name → (boxes, img_w, img_h).
        categories: Optional list of {"id": int, "name": str} dicts.
    """
    coco: dict = {"images": [], "annotations": [], "categories": categories or []}
    ann_id = 1
    for img_id, (file_name, (boxes, img_w, img_h)) in enumerate(
        images_annotations.items(), start=1
    ):
        coco["images"].append({
            "id": img_id,
            "file_name": file_name,
            "width": img_w,
            "height": img_h,
        })
        for box in boxes:
            bw = box.x2 - box.x1
            bh = box.y2 - box.y1
            rec = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": box.class_id,
                "bbox": [round(box.x1, 2), round(box.y1, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
            }
            _emit_coco_extras(rec, box)
            coco["annotations"].append(rec)
            ann_id += 1

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)


def write_coco_segment(
    path: str,
    images_annotations: dict[str, tuple[list[Polygon], int, int]],
    categories: list[dict] | None = None,
) -> None:
    """Write segmentation annotations to a COCO JSON file.

    Args:
        path: Output JSON file path.
        images_annotations: Dict mapping file_name → (polygons, img_w, img_h).
        categories: Optional list of {"id": int, "name": str} dicts.
    """
    coco: dict = {"images": [], "annotations": [], "categories": categories or []}
    ann_id = 1
    for img_id, (file_name, (polygons, img_w, img_h)) in enumerate(
        images_annotations.items(), start=1
    ):
        coco["images"].append({
            "id": img_id,
            "file_name": file_name,
            "width": img_w,
            "height": img_h,
        })
        for poly in polygons:
            flat = []
            for x, y in poly.points:
                flat.extend([round(x, 2), round(y, 2)])
            xs = [p[0] for p in poly.points]
            ys = [p[1] for p in poly.points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            bw = x_max - x_min
            bh = y_max - y_min
            rec = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": poly.class_id,
                "segmentation": [flat],
                "bbox": [round(x_min, 2), round(y_min, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
            }
            _emit_coco_extras(rec, poly)
            coco["annotations"].append(rec)
            ann_id += 1

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)



def load_annotations(
    path: str,
    img_w: int,
    img_h: int,
    task: Task = "detect",
    fmt: AnnotFormat | None = None,
    image_id: int | None = None,
    file_name: str | None = None,
) -> tuple[list[BBox] | list[Polygon], set[int]]:
    """Load annotations. ``fmt`` of ``None`` detects it from the file's own keys.

    For COCO, either ``image_id`` or ``file_name`` must identify the target image.
    """
    if fmt is None:
        fmt = detect_format(path)

    if fmt == "json":  # the canonical per-image label file
        from tcip_annotation import json_io

        if task == "detect":
            return json_io.read_detect(path, img_w, img_h)
        return json_io.read_segment(path, img_w, img_h)
    if fmt == "coco":
        coco = _parse_coco_json(path)
        if task == "detect":
            return parse_coco_detect(coco, image_id, file_name)
        return parse_coco_segment(coco, image_id, file_name)
    raise ValueError(f"Unsupported annotation format: {fmt}")


def save_annotations(
    path: str,
    annotations: list[BBox] | list[Polygon],
    img_w: int,
    img_h: int,
    task: Task = "detect",
    fmt: AnnotFormat = "json",
    file_name: str | None = None,
    id_to_name: dict[int, str] | None = None,
    keep_empty: bool = False,
) -> None:
    """Save annotations.

    ``json`` writes the canonical per-image label file; ``coco`` writes/updates a dataset-level
    COCO (pass ``file_name`` to identify the image). ``keep_empty`` (json only): an empty list
    writes an ``objects: []`` record instead of deleting the label — without it a save of zero
    shapes erases the GT. An empty record is not a negative until a human confirms it.
    """
    if fmt == "json":  # the canonical per-image label file
        from tcip_annotation import json_io

        if task == "detect":
            json_io.write_detect(path, annotations, img_w, img_h, keep_empty=keep_empty)  # type: ignore[arg-type]
        else:
            json_io.write_segment(path, annotations, img_w, img_h, keep_empty=keep_empty)  # type: ignore[arg-type]
    elif fmt == "coco":
        fname = file_name or Path(path).stem
        images_dict = {fname: (annotations, img_w, img_h)}
        if task == "detect":
            write_coco_detect(path, images_dict)  # type: ignore[arg-type]
        else:
            write_coco_segment(path, images_dict)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unsupported annotation format: {fmt}")
