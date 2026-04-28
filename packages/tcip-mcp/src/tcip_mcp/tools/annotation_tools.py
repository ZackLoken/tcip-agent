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
from tcip_annotation.format_io import (
    detect_format,
    detect_format_confident,
    load_annotations as format_load,
    save_annotations as format_save,
    AnnotFormat,
)
from tcip_annotation.utils import get_image_dimensions

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited


def _find_label_path(root: Path, stem: str, task: str = "detect", fmt: str | None = None) -> Path | None:
    """Find a label file for the given image stem in any supported format."""
    subdir = "detect" if task == "detect" else "segment"
    search_dirs = [root / "labels" / subdir, root / "labels"]

    # If format is specified, search for matching extension first
    fmt_ext_map = {"yolo": ".txt", "voc": ".xml", "coco": ".json", "labelme": ".json"}
    if fmt and fmt in fmt_ext_map:
        preferred_ext = fmt_ext_map[fmt]
        for d in search_dirs:
            if not d.is_dir():
                continue
            candidate = d / f"{stem}{preferred_ext}"
            if candidate.is_file():
                return candidate

    for d in search_dirs:
        if not d.is_dir():
            continue
        for ext in (".txt", ".xml", ".json"):
            candidate = d / f"{stem}{ext}"
            if candidate.is_file():
                return candidate
    return None


@mcp.tool()
@audited
def load_annotations(image_path: str, fmt: str | None = None) -> dict:
    """Load labels and predictions for a single image.

    Supports YOLO (.txt), PASCAL VOC (.xml), COCO (.json), and LabelMe (.json).
    Format is auto-detected from file extension unless fmt is specified.

    Args:
        image_path: Absolute path to the image file.
        fmt: Force annotation format ('yolo', 'voc', 'coco', 'labelme'). Auto-detects if omitted.
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    w, h = get_image_dimensions(image_path)
    stem = img.stem
    root = img.parent.parent  # e.g. data/ if images are in data/images/

    result: dict = {"image": image_path, "width": w, "height": h}

    # Find and load detection labels
    det_path = _find_label_path(root, stem, "detect", fmt=fmt)
    if det_path is not None:
        if fmt:
            file_fmt, confident = fmt, True
        else:
            file_fmt, confident = detect_format_confident(str(det_path))
        boxes, class_ids = format_load(str(det_path), w, h, task="detect", fmt=file_fmt)
        result["detect_labels"] = {
            "path": str(det_path),
            "format": file_fmt,
            "format_confident": confident,
            "count": len(boxes),
            "class_ids": sorted(class_ids),
            "boxes": [
                {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "class_id": b.class_id}
                for b in boxes
            ],
        }
        if not confident:
            result["warning"] = (
                f"Annotation format could not be confidently detected for {det_path.name}; "
                "defaulted to YOLO. If this is wrong, re-call with fmt='coco', 'voc', or 'labelme'."
            )

    # Find and load segment labels
    seg_path = _find_label_path(root, stem, "segment", fmt=fmt)
    if seg_path is not None:
        if fmt:
            file_fmt, confident = fmt, True
        else:
            file_fmt, confident = detect_format_confident(str(seg_path))
        polys, class_ids = format_load(str(seg_path), w, h, task="segment", fmt=file_fmt)
        result["segment_labels"] = {
            "path": str(seg_path),
            "format": file_fmt,
            "format_confident": confident,
            "count": len(polys),
            "class_ids": sorted(class_ids),
        }
        if not confident and "warning" not in result:
            result["warning"] = (
                f"Annotation format could not be confidently detected for {seg_path.name}; "
                "defaulted to YOLO. If this is wrong, re-call with fmt='coco', 'voc', or 'labelme'."
            )

    # Look for predictions (YOLO format — predictions are always YOLO)
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
@audited
def save_annotations(
    image_path: str,
    boxes: list[dict] | None = None,
    polygons: list[dict] | None = None,
    fmt: str = "yolo",
) -> dict:
    """Write annotation label files for an image.

    Supports YOLO (.txt), PASCAL VOC (.xml), and LabelMe (.json).

    Args:
        image_path: Absolute path to the image file.
        boxes: List of dicts with x1, y1, x2, y2, class_id (pixel coords).
        polygons: List of dicts with points and class_id (pixel coords).
        fmt: Output format — 'yolo' (default), 'voc', 'labelme', 'coco'.
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    w, h = get_image_dimensions(image_path)
    stem = img.stem
    root = img.parent.parent

    ext_map = {"yolo": ".txt", "voc": ".xml", "labelme": ".json", "coco": ".json"}
    ext = ext_map.get(fmt, ".txt")

    written: list[str] = []

    if boxes is not None:
        typed_boxes = [BBox(x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"], class_id=b["class_id"]) for b in boxes]
        out_dir = root / "labels" / "detect"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}{ext}"
        format_save(str(out_path), typed_boxes, w, h, task="detect", fmt=fmt, file_name=img.name)
        written.append(str(out_path))

    if polygons is not None:
        typed_polys = [
            Polygon(points=[(pt[0], pt[1]) for pt in p["points"]], class_id=p["class_id"])
            for p in polygons
        ]
        out_dir = root / "labels" / "segment"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}{ext}"
        format_save(str(out_path), typed_polys, w, h, task="segment", fmt=fmt, file_name=img.name)
        written.append(str(out_path))

    return {"written": written, "format": fmt, "count": len(written)}


@mcp.tool()
@audited
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
@audited
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


@mcp.tool()
@audited
def sam_predict(
    image_path: str,
    points: list[dict] | None = None,
    box: dict | None = None,
    grid_cells: list[str] | None = None,
    model_type: str = "hiera_b+",
) -> dict:
    """Run SAM (Segment Anything) prediction on an image.

    Provide point prompts, a box prompt, OR grid cell references.
    Grid cells (e.g. ['B3', 'D5']) are converted to pixel coordinates
    using the grid overlay system (8 cols x 6 rows by default).

    Args:
        image_path: Absolute path to the image file.
        points: List of point prompts, each with x, y, and label (1=fg, 0=bg).
        box: Box prompt with x1, y1, x2, y2 in pixel coordinates.
        grid_cells: List of grid cell references like ['B3', 'D5']. Each
            cell is treated as a foreground point prompt at the cell center.
        model_type: SAM2 variant — hiera_t / hiera_s / hiera_b+ (default) / hiera_l.
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    if points is None and box is None and grid_cells is None:
        return {"error": "Provide either points, box, or grid_cells prompt"}

    # Convert grid cells to point prompts
    if grid_cells is not None:
        from tcip_annotation.sam_wrapper import grid_to_pixel
        w, h = get_image_dimensions(image_path)
        points = []
        for cell in grid_cells:
            try:
                cx, cy = grid_to_pixel(cell, w, h)
                points.append({"x": cx, "y": cy, "label": 1})
            except ValueError as e:
                return {"error": f"Invalid grid cell {cell!r}: {e}"}

    try:
        from tcip_annotation.sam_wrapper import (
            predict_from_box,
            predict_from_point,
            predict_from_points,
        )
    except ImportError as e:
        return {"error": f"SAM dependencies not available: {e}"}

    try:
        if box is not None:
            polygon = predict_from_box(
                image_path,
                box["x1"], box["y1"], box["x2"], box["y2"],
                model_type=model_type,
            )
        elif points is not None and len(points) == 1:
            p = points[0]
            polygon = predict_from_point(
                image_path,
                p["x"], p["y"],
                label=p.get("label", 1),
                model_type=model_type,
            )
        else:
            pts = [(p["x"], p["y"]) for p in (points or [])]
            lbls = [p.get("label", 1) for p in (points or [])]
            polygon = predict_from_points(
                image_path, pts, lbls,
                model_type=model_type,
            )

        if not polygon:
            return {"error": "SAM produced empty mask", "polygon": []}

        return {
            "polygon": [{"x": x, "y": y} for x, y in polygon],
            "vertex_count": len(polygon),
        }
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"SAM prediction failed: {e}"}


@mcp.tool()
@audited
def run_matching(
    image_path: str,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> dict:
    """Run GT-vs-prediction matching for a single image.

    Returns detailed match data including per-detection TP/FP/FN classification
    with bounding box coordinates, class IDs, IoU values, and confidence scores.

    This is the low-level matching tool — use for annotation review,
    quality assessment, or feeding match data to the review panel.

    Args:
        image_path: Absolute path to the image file.
        iou_threshold: IoU threshold for matching (default 0.5).
        conf_threshold: Confidence threshold for filtering predictions (default 0.25).
    """
    from tcip_annotation import (
        parse_detect_labels,
        parse_detect_predictions,
        parse_segment_labels,
        parse_segment_predictions,
        compute_matches,
    )

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    root = img.parent.parent

    # Get image dimensions
    img_w, img_h = get_image_dimensions(image_path)

    stem = img.stem
    gt_det = str(root / "labels" / "detect" / f"{stem}.txt")
    gt_seg = str(root / "labels" / "segment" / f"{stem}.txt")
    pred_det = str(root / "predictions" / "detect" / f"{stem}.txt")
    pred_seg = str(root / "predictions" / "segment" / f"{stem}.txt")

    gt_boxes, _ = parse_detect_labels(gt_det, img_w, img_h)
    gt_polys, _ = parse_segment_labels(gt_seg, img_w, img_h)
    pred_boxes, _ = parse_detect_predictions(pred_det, img_w, img_h)
    pred_polys, _ = parse_segment_predictions(pred_seg, img_w, img_h)

    matches = compute_matches(gt_boxes, gt_polys, pred_boxes, pred_polys,
                              iou_threshold, conf_threshold)

    # Build detailed per-detection records with coordinates
    detections = []
    for m in matches["tp"]:
        d = {
            "tag": "tp", "class_id": m["class_id"],
            "iou": m["iou"], "confidence": m["conf"],
            "gt_type": m["gt_type"], "gt_idx": m["gt_idx"],
            "pred_type": m["pred_type"], "pred_idx": m["pred_idx"],
        }
        if m["pred_type"] == "box":
            b = pred_boxes[m["pred_idx"]]
            d["box"] = [b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1]
        else:
            p = pred_polys[m["pred_idx"]]
            d["polygon"] = [[pt[0], pt[1]] for pt in p.points]
        detections.append(d)

    for m in matches["fp"]:
        d = {
            "tag": "fp", "class_id": m["class_id"],
            "confidence": m["conf"],
            "pred_type": m["pred_type"], "pred_idx": m["pred_idx"],
        }
        if m["pred_type"] == "box":
            b = pred_boxes[m["pred_idx"]]
            d["box"] = [b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1]
        else:
            p = pred_polys[m["pred_idx"]]
            d["polygon"] = [[pt[0], pt[1]] for pt in p.points]
        detections.append(d)

    for m in matches["fn"]:
        d = {
            "tag": "fn", "class_id": m["class_id"], "confidence": 0,
            "gt_type": m["gt_type"], "gt_idx": m["gt_idx"],
        }
        if m["gt_type"] == "box":
            b = gt_boxes[m["gt_idx"]]
            d["box"] = [b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1]
        else:
            p = gt_polys[m["gt_idx"]]
            d["polygon"] = [[pt[0], pt[1]] for pt in p.points]
        detections.append(d)

    return {
        "image": str(img),
        "img_w": img_w,
        "img_h": img_h,
        "tp_count": len(matches["tp"]),
        "fp_count": len(matches["fp"]),
        "fn_count": len(matches["fn"]),
        "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold,
        "detections": detections,
    }


@mcp.tool()
@audited
def push_panel_data(
    panel: str,
    event_type: str,
    data: dict,
) -> dict:
    """Push structured data to a TCIP GUI panel via the tcip-web backend.

    Sends an HTTP POST to the running FastAPI server (see
    :mod:`tcip_mcp.web_client`); the backend broadcasts to any connected
    browsers via WebSocket. Replaces the legacy ``.tcip/events/`` file
    bridge.

    If the backend is not running the call returns
    ``{"status": "no_subscribers"}`` so the agent can proceed.

    Args:
        panel: Target panel — 'annotate', 'review', 'training', 'tuning',
            'inference', or 'results'. The legacy name 'hpo' is aliased to
            'tuning' for backwards compatibility.
        event_type: Event type the panel switches on (e.g. 'load_matches',
            'metrics_update').
        data: Arbitrary JSON data payload.
    """
    from tcip_mcp.web_client import post_panel_event

    valid_panels = {"annotate", "annotation", "review", "training", "tuning", "hpo", "inference", "results"}
    if panel not in valid_panels:
        return {"error": f"Unknown panel: {panel}. Valid: {sorted(valid_panels)}"}

    # Normalise legacy aliases
    if panel == "annotation":
        panel = "annotate"
    if panel == "hpo":
        panel = "tuning"

    result = post_panel_event(panel, event_type, data)
    result.setdefault("panel", panel)
    result.setdefault("event_type", event_type)
    return result
