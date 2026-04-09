"""Annotation tools — load, save, and evaluate annotations via MCP."""

from __future__ import annotations

from pathlib import Path

from tcip_annotation import (
    parse_detect_labels,
    parse_detect_predictions,
    parse_segment_labels,
    parse_segment_predictions,
    write_detect_labels,
    write_segment_labels,
    compute_matches,
    BBox,
    Polygon,
)
from tcip_annotation.utils import get_image_dimensions

from tcip_mcp.server import mcp


@mcp.tool()
def load_annotations(image_path: str) -> dict:
    """Load YOLO labels and predictions for a single image.

    Looks for label/prediction .txt files at conventional YOLO paths
    relative to the image directory.

    Args:
        image_path: Absolute path to the image file.
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    w, h = get_image_dimensions(image_path)
    stem = img.stem

    result: dict = {"image": image_path, "width": w, "height": h}

    # Look for labels in conventional locations
    root = img.parent.parent  # e.g. data/ if images are in data/images/
    for label_dir in (root / "labels" / "detect", root / "labels"):
        txt = label_dir / f"{stem}.txt"
        if txt.is_file():
            boxes, class_ids = parse_detect_labels(str(txt), w, h)
            result["detect_labels"] = {
                "path": str(txt),
                "count": len(boxes),
                "class_ids": sorted(class_ids),
                "boxes": [
                    {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "class_id": b.class_id}
                    for b in boxes
                ],
            }
            break

    for label_dir in (root / "labels" / "segment",):
        txt = label_dir / f"{stem}.txt"
        if txt.is_file():
            polys, class_ids = parse_segment_labels(str(txt), w, h)
            result["segment_labels"] = {
                "path": str(txt),
                "count": len(polys),
                "class_ids": sorted(class_ids),
            }
            break

    # Look for predictions
    for pred_dir in (root / "predictions" / "detect",):
        txt = pred_dir / f"{stem}.txt"
        if txt.is_file():
            pred_boxes, class_ids = parse_detect_predictions(str(txt), w, h)
            result["detect_predictions"] = {
                "path": str(txt),
                "count": len(pred_boxes),
                "class_ids": sorted(class_ids),
                "boxes": [
                    {
                        "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                        "class_id": b.class_id, "confidence": b.confidence,
                    }
                    for b in pred_boxes
                ],
            }
            break

    for pred_dir in (root / "predictions" / "segment",):
        txt = pred_dir / f"{stem}.txt"
        if txt.is_file():
            pred_polys, class_ids = parse_segment_predictions(str(txt), w, h)
            result["segment_predictions"] = {
                "path": str(txt),
                "count": len(pred_polys),
                "class_ids": sorted(class_ids),
            }
            break

    return result


@mcp.tool()
def save_annotations(
    image_path: str,
    boxes: list[dict] | None = None,
    polygons: list[dict] | None = None,
) -> dict:
    """Write YOLO label files for an image.

    Args:
        image_path: Absolute path to the image file.
        boxes: List of dicts with x1, y1, x2, y2, class_id (pixel coords).
        polygons: List of dicts with points and class_id (pixel coords).
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    w, h = get_image_dimensions(image_path)
    stem = img.stem
    root = img.parent.parent

    written: list[str] = []

    if boxes is not None:
        typed_boxes = [BBox(x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"], class_id=b["class_id"]) for b in boxes]
        out_dir = root / "labels" / "detect"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}.txt"
        write_detect_labels(str(out_path), typed_boxes, w, h)
        written.append(str(out_path))

    if polygons is not None:
        typed_polys = [
            Polygon(points=[(pt[0], pt[1]) for pt in p["points"]], class_id=p["class_id"])
            for p in polygons
        ]
        out_dir = root / "labels" / "segment"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}.txt"
        write_segment_labels(str(out_path), typed_polys, w, h)
        written.append(str(out_path))

    return {"written": written, "count": len(written)}


@mcp.tool()
def evaluate_detections(
    image_path: str,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> dict:
    """Match predictions against ground truth for a single image.

    Args:
        image_path: Absolute path to the image file.
        iou_threshold: IoU threshold for a positive match.
        conf_threshold: Minimum confidence to consider a prediction.
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    w, h = get_image_dimensions(image_path)
    stem = img.stem
    root = img.parent.parent

    gt_boxes: list[BBox] = []
    gt_polys: list[Polygon] = []
    pred_boxes = []
    pred_polys = []

    # Load GT
    detect_label = root / "labels" / "detect" / f"{stem}.txt"
    if detect_label.is_file():
        gt_boxes, _ = parse_detect_labels(str(detect_label), w, h)

    segment_label = root / "labels" / "segment" / f"{stem}.txt"
    if segment_label.is_file():
        gt_polys, _ = parse_segment_labels(str(segment_label), w, h)

    # Load predictions
    detect_pred = root / "predictions" / "detect" / f"{stem}.txt"
    if detect_pred.is_file():
        pred_boxes, _ = parse_detect_predictions(str(detect_pred), w, h)

    segment_pred = root / "predictions" / "segment" / f"{stem}.txt"
    if segment_pred.is_file():
        pred_polys, _ = parse_segment_predictions(str(segment_pred), w, h)

    matches = compute_matches(
        gt_boxes, gt_polys, pred_boxes, pred_polys,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
    )

    tp = len(matches["tp"])
    fp = len(matches["fp"])
    fn = len(matches["fn"])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "image": image_path,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold,
        "matches": matches,
    }


@mcp.tool()
def evaluate_dataset(
    folder_path: str,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> dict:
    """Aggregate detection metrics across all images in a dataset.

    Args:
        folder_path: Path to dataset root.
        iou_threshold: IoU threshold for a positive match.
        conf_threshold: Minimum confidence.
    """
    root = Path(folder_path)
    images_dir = root / "images"
    if not images_dir.is_dir():
        images_dir = root

    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    images = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in image_exts)

    total_tp = total_fp = total_fn = 0
    per_image: list[dict] = []

    for img in images:
        r = evaluate_detections(str(img), iou_threshold, conf_threshold)
        if "error" in r:
            continue
        total_tp += r["tp"]
        total_fp += r["fp"]
        total_fn += r["fn"]
        per_image.append({"image": img.name, "tp": r["tp"], "fp": r["fp"], "fn": r["fn"]})

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "path": folder_path,
        "image_count": len(images),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_image": per_image,
    }
