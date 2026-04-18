"""Format-agnostic annotation I/O — auto-detect and dispatch across formats.

The internal representation (BBox, Polygon, PredBBox, PredPolygon) is always
pixel-coordinate.  Format only matters at file I/O boundaries.

Supported formats:
  - yolo      — one .txt per image, normalized coords
  - coco      — single .json for entire dataset, pixel coords
  - voc       — one .xml per image (PASCAL VOC), pixel coords
  - labelme   — one .json per image (LabelMe), pixel coords

Usage:
    boxes, class_ids = load_annotations(path, img_w, img_h, task="detect")
    save_annotations(path, boxes, img_w, img_h, task="detect", fmt="yolo")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

from tcip_annotation.state import BBox, Polygon, PredBBox, PredPolygon
from tcip_annotation.label_io import (
    parse_detect_labels,
    parse_segment_labels,
    parse_detect_predictions,
    parse_segment_predictions,
    write_detect_labels,
    write_segment_labels,
)

AnnotFormat = Literal["yolo", "coco", "voc", "labelme"]
Task = Literal["detect", "segment"]


# ── Format detection ────────────────────────────────────────────────────────


def detect_format(path: str) -> AnnotFormat:
    """Infer annotation format from file extension and content.

    Rules:
      - .xml → voc (PASCAL VOC)
      - .txt → yolo
      - .json → inspect content: 'shapes' key → labelme, else → coco
      - directory → inspect files inside
    """
    p = Path(path)

    if p.suffix == ".xml":
        return "voc"

    if p.suffix == ".txt":
        return "yolo"

    if p.suffix == ".json":
        return _detect_json_format(p)

    if p.is_dir():
        has_xml = any(p.glob("*.xml"))
        has_json = any(p.glob("*.json"))
        has_txt = any(p.glob("*.txt"))
        if has_xml:
            return "voc"
        if has_json and not has_txt:
            # Check first JSON to distinguish COCO vs LabelMe
            first_json = next(p.glob("*.json"))
            return _detect_json_format(first_json)
        if has_txt:
            return "yolo"

    return "yolo"  # default fallback


def _detect_json_format(path: Path) -> AnnotFormat:
    """Distinguish COCO from LabelMe by inspecting JSON keys."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "shapes" in data:
                return "labelme"
            if "images" in data or "annotations" in data:
                return "coco"
    except (json.JSONDecodeError, OSError):
        pass
    return "coco"  # default for unknown JSON


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
        boxes.append(BBox(x, y, x + bw, y + bh, cid))
        class_ids.add(cid)
    return boxes, class_ids


def parse_coco_segment(
    coco: dict, image_id: int | None = None, file_name: str | None = None,
) -> tuple[list[Polygon], set[int]]:
    """Parse COCO segmentation annotations for one image into Polygon objects.

    COCO segmentation format: list of [x1,y1,x2,y2,...] polygon arrays (pixel coords).
    Only the first polygon per annotation is used (no RLE support yet).
    """
    anns, _, _ = _coco_image_annotations(coco, image_id, file_name)
    polygons: list[Polygon] = []
    class_ids: set[int] = set()
    for ann in anns:
        segs = ann.get("segmentation")
        if not segs or not isinstance(segs, list) or not isinstance(segs[0], list):
            continue
        coords = segs[0]  # first polygon
        if len(coords) < 6:
            continue
        cid = ann.get("category_id", 0)
        points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        polygons.append(Polygon(points, cid))
        class_ids.add(cid)
    return polygons, class_ids


# ── COCO JSON writing ──────────────────────────────────────────────────────


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
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": box.class_id,
                "bbox": [round(box.x1, 2), round(box.y1, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
            })
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
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": poly.class_id,
                "segmentation": [flat],
                "bbox": [round(x_min, 2), round(y_min, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
            })
            ann_id += 1

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

# ── PASCAL VOC XML parsing ──────────────────────────────────────────────────


def parse_voc_detect(path: str) -> tuple[list[BBox], set[int], dict[str, int]]:
    """Parse PASCAL VOC XML file into BBox objects.

    VOC stores class names as strings. Returns a name→id mapping built
    on the fly (alphabetical order).

    Returns (boxes, class_ids, name_to_id).
    """
    tree = ET.parse(path)
    root = tree.getroot()

    names: list[str] = []
    for obj in root.iter("object"):
        name = obj.findtext("name", "")
        if name and name not in names:
            names.append(name)
    names.sort()
    name_to_id = {n: i for i, n in enumerate(names)}

    boxes: list[BBox] = []
    class_ids: set[int] = set()
    for obj in root.iter("object"):
        name = obj.findtext("name", "")
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            x1 = float(bndbox.findtext("xmin", "0"))
            y1 = float(bndbox.findtext("ymin", "0"))
            x2 = float(bndbox.findtext("xmax", "0"))
            y2 = float(bndbox.findtext("ymax", "0"))
        except ValueError:
            continue
        cid = name_to_id.get(name, 0)
        boxes.append(BBox(x1, y1, x2, y2, cid))
        class_ids.add(cid)

    return boxes, class_ids, name_to_id


def write_voc_detect(
    path: str,
    boxes: list[BBox],
    img_w: int,
    img_h: int,
    file_name: str = "",
    id_to_name: dict[int, str] | None = None,
) -> None:
    """Write detection boxes to a PASCAL VOC XML file.

    Args:
        path: Output XML file path.
        boxes: List of BBox objects in pixel coordinates.
        img_w: Image width.
        img_h: Image height.
        file_name: Image filename to embed in the XML.
        id_to_name: Optional mapping from class_id to class name.
    """
    id_to_name = id_to_name or {}

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = file_name or Path(path).stem + ".jpg"
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_w)
    ET.SubElement(size, "height").text = str(img_h)
    ET.SubElement(size, "depth").text = "3"

    for box in boxes:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = id_to_name.get(box.class_id, str(box.class_id))
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(round(box.x1))
        ET.SubElement(bndbox, "ymin").text = str(round(box.y1))
        ET.SubElement(bndbox, "xmax").text = str(round(box.x2))
        ET.SubElement(bndbox, "ymax").text = str(round(box.y2))

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tree_out = ET.ElementTree(annotation)
    tree_out.write(path, encoding="UTF-8", xml_declaration=True)


# ── LabelMe JSON parsing ───────────────────────────────────────────────────


def parse_labelme_detect(path: str) -> tuple[list[BBox], set[int], dict[str, int]]:
    """Parse a LabelMe JSON file's rectangle shapes into BBox objects.

    LabelMe stores class names as strings. Returns a name→id mapping.

    Returns (boxes, class_ids, name_to_id).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    names: list[str] = sorted({
        s.get("label", "") for s in data.get("shapes", [])
        if s.get("label")
    })
    name_to_id = {n: i for i, n in enumerate(names)}

    boxes: list[BBox] = []
    class_ids: set[int] = set()
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        points = shape.get("points", [])
        shape_type = shape.get("shape_type", "")

        if shape_type == "rectangle" and len(points) == 2:
            (x1, y1), (x2, y2) = points
            cid = name_to_id.get(label, 0)
            boxes.append(BBox(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), cid))
            class_ids.add(cid)

    return boxes, class_ids, name_to_id


def parse_labelme_segment(path: str) -> tuple[list[Polygon], set[int], dict[str, int]]:
    """Parse a LabelMe JSON file's polygon shapes into Polygon objects.

    Returns (polygons, class_ids, name_to_id).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    names: list[str] = sorted({
        s.get("label", "") for s in data.get("shapes", [])
        if s.get("label")
    })
    name_to_id = {n: i for i, n in enumerate(names)}

    polygons: list[Polygon] = []
    class_ids: set[int] = set()
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        points = shape.get("points", [])
        shape_type = shape.get("shape_type", "")

        if shape_type == "polygon" and len(points) >= 3:
            cid = name_to_id.get(label, 0)
            pixel_points = [(float(p[0]), float(p[1])) for p in points]
            polygons.append(Polygon(pixel_points, cid))
            class_ids.add(cid)

    return polygons, class_ids, name_to_id


def write_labelme(
    path: str,
    annotations: list[BBox] | list[Polygon],
    img_w: int,
    img_h: int,
    file_name: str = "",
    id_to_name: dict[int, str] | None = None,
) -> None:
    """Write annotations to a LabelMe JSON file.

    Handles both BBox (→ rectangle) and Polygon (→ polygon) shapes.
    """
    id_to_name = id_to_name or {}
    shapes = []

    for ann in annotations:
        label = id_to_name.get(ann.class_id, str(ann.class_id))
        if isinstance(ann, BBox):
            shapes.append({
                "label": label,
                "points": [[ann.x1, ann.y1], [ann.x2, ann.y2]],
                "shape_type": "rectangle",
                "flags": {},
            })
        elif isinstance(ann, Polygon):
            shapes.append({
                "label": label,
                "points": [[x, y] for x, y in ann.points],
                "shape_type": "polygon",
                "flags": {},
            })

    data = {
        "version": "5.0.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": file_name or Path(path).stem + ".jpg",
        "imageHeight": img_h,
        "imageWidth": img_w,
        "imageData": None,
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── Unified dispatch API ───────────────────────────────────────────────────


def load_annotations(
    path: str,
    img_w: int,
    img_h: int,
    task: Task = "detect",
    fmt: AnnotFormat | None = None,
    image_id: int | None = None,
    file_name: str | None = None,
) -> tuple[list[BBox] | list[Polygon], set[int]]:
    """Load annotations from any supported format.

    If fmt is None, auto-detects from file extension/content.
    For COCO, either image_id or file_name must identify the target image.
    For VOC/LabelMe, the name→id mapping is built automatically.
    """
    if fmt is None:
        fmt = detect_format(path)

    if fmt == "yolo":
        if task == "detect":
            return parse_detect_labels(path, img_w, img_h)
        else:
            return parse_segment_labels(path, img_w, img_h)
    elif fmt == "coco":
        coco = _parse_coco_json(path)
        if task == "detect":
            return parse_coco_detect(coco, image_id, file_name)
        else:
            return parse_coco_segment(coco, image_id, file_name)
    elif fmt == "voc":
        if task == "detect":
            boxes, class_ids, _ = parse_voc_detect(path)
            return boxes, class_ids
        else:
            # VOC has no native polygon support — return empty
            return [], set()
    elif fmt == "labelme":
        if task == "detect":
            boxes, class_ids, _ = parse_labelme_detect(path)
            return boxes, class_ids
        else:
            polygons, class_ids, _ = parse_labelme_segment(path)
            return polygons, class_ids
    else:
        raise ValueError(f"Unsupported annotation format: {fmt}")


def save_annotations(
    path: str,
    annotations: list[BBox] | list[Polygon],
    img_w: int,
    img_h: int,
    task: Task = "detect",
    fmt: AnnotFormat = "yolo",
    file_name: str | None = None,
    id_to_name: dict[int, str] | None = None,
) -> None:
    """Save annotations to the specified format.

    For YOLO: writes a single .txt file (one per image).
    For COCO: writes/updates a .json file (pass file_name for image identification).
    For VOC: writes a single .xml file (detection only).
    For LabelMe: writes a single .json file (rectangles or polygons).
    """
    if fmt == "yolo":
        if task == "detect":
            write_detect_labels(path, annotations, img_w, img_h)  # type: ignore[arg-type]
        else:
            write_segment_labels(path, annotations, img_w, img_h)  # type: ignore[arg-type]
    elif fmt == "coco":
        fname = file_name or Path(path).stem
        images_dict = {fname: (annotations, img_w, img_h)}
        if task == "detect":
            write_coco_detect(path, images_dict)  # type: ignore[arg-type]
        else:
            write_coco_segment(path, images_dict)  # type: ignore[arg-type]
    elif fmt == "voc":
        write_voc_detect(
            path, annotations, img_w, img_h,  # type: ignore[arg-type]
            file_name=file_name or "", id_to_name=id_to_name,
        )
    elif fmt == "labelme":
        write_labelme(
            path, annotations, img_w, img_h,
            file_name=file_name or "", id_to_name=id_to_name,
        )
    else:
        raise ValueError(f"Unsupported annotation format: {fmt}")
