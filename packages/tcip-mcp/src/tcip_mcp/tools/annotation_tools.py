"""Annotation tools — load, save, and evaluate annotations via MCP."""

from __future__ import annotations

from pathlib import Path

from tcip_annotation import (
    parse_detect_labels,
    parse_detect_predictions,
    parse_segment_labels,
    parse_segment_predictions,
    compute_matches,
    BBox,
    Polygon,
)
from tcip_annotation.format_io import (
    detect_format_confident,
    load_annotations as format_load,
    save_annotations as format_save,
)
from tcip_annotation.utils import get_image_dimensions

from tcip_mcp.dataset_layout import (
    DEFAULT_TRAIT,
    annotation_path_for_image,
    find_gt_label,
    find_prediction,
)
from tcip_mcp.server import mcp
from tcip_mcp.audit import audited


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

    result: dict = {"image": image_path, "width": w, "height": h}

    # Find and load detection labels
    det_path = find_gt_label(image_path, "detect", fmt=fmt)
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
    seg_path = find_gt_label(image_path, "segment", fmt=fmt)
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
    pred_det = find_prediction(image_path, "detect", fmt="yolo")
    if pred_det is not None:
        pred_boxes, class_ids = parse_detect_predictions(str(pred_det), w, h)
        result["detect_predictions"] = {
            "path": str(pred_det),
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

    pred_seg = find_prediction(image_path, "segment", fmt="yolo")
    if pred_seg is not None:
        pred_polys, class_ids = parse_segment_predictions(str(pred_seg), w, h)
        result["segment_predictions"] = {
            "path": str(pred_seg),
            "count": len(pred_polys),
            "class_ids": sorted(class_ids),
        }

    return result


@mcp.tool()
@audited
def save_annotations(
    image_path: str,
    boxes: list[dict] | None = None,
    polygons: list[dict] | None = None,
    fmt: str = "yolo",
    trait: str = DEFAULT_TRAIT,
    date: str | None = None,
    detect_path: str | None = None,
    segment_path: str | None = None,
) -> dict:
    """Write annotation label files for an image into the canonical dataset layout.

    Labels go to ``<dataset_root>/annotations/<trait>/<date>/<task>/<stem>.<ext>``
    (see :mod:`tcip_mcp.dataset_layout`); ``date`` is derived from the image path
    (``images/<date>/<stem>``) when not given. Pass ``detect_path`` / ``segment_path``
    to write to an explicit location instead. Supports YOLO (.txt), PASCAL VOC (.xml),
    and LabelMe (.json).

    Args:
        image_path: Absolute path to the image file.
        boxes: List of dicts with x1, y1, x2, y2, class_id (pixel coords).
        polygons: List of dicts with points and class_id (pixel coords).
        fmt: Output format — 'yolo' (default), 'voc', 'labelme', 'coco'.
        trait: Annotation campaign (e.g. 'catkin') — the layout's ``<trait>`` segment.
        date: Capture date; derived from the image path when omitted.
        detect_path: Explicit detect label path (overrides the canonical location).
        segment_path: Explicit segment label path (overrides the canonical location).
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    w, h = get_image_dimensions(image_path)
    stem = img.stem

    written: list[str] = []

    if boxes is not None:
        typed_boxes = [BBox(x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"], class_id=b["class_id"]) for b in boxes]
        out_path = (
            Path(detect_path)
            if detect_path
            else annotation_path_for_image(image_path, "detect", fmt, trait=trait, date=date)
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        format_save(str(out_path), typed_boxes, w, h, task="detect", fmt=fmt, file_name=img.name)
        written.append(str(out_path))

    if polygons is not None:
        typed_polys = [
            Polygon(points=[(pt[0], pt[1]) for pt in p["points"]], class_id=p["class_id"])
            for p in polygons
        ]
        out_path = (
            Path(segment_path)
            if segment_path
            else annotation_path_for_image(image_path, "segment", fmt, trait=trait, date=date)
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        format_save(str(out_path), typed_polys, w, h, task="segment", fmt=fmt, file_name=img.name)
        written.append(str(out_path))

    # Best-effort: notify a running GUI that the agent touched this image's labels,
    # so it can surface the activity and refresh if it is viewing the same file.
    if written:
        try:
            from tcip_mcp.web_client import post_panel_event

            post_panel_event(
                "annotate",
                "labels_written",
                {
                    "image_path": image_path,
                    "stem": stem,
                    "trait": trait,
                    "written": written,
                    "count": len(written),
                },
            )
        except Exception:
            pass

    return {"written": written, "format": fmt, "count": len(written)}


def _load_image_annotations(image_path: str):
    """Load GT + predictions for one image and build a COCO per-image record.

    Returns ``(iou_type, record, raw, width, height)`` where
    ``raw = (gt_boxes, gt_polys, pred_boxes, pred_polys)``; ``None`` if unreadable.
    """
    from tcip_mcp.pipelines.training.evaluation import records_from_annotation

    img = Path(image_path)
    if not img.is_file():
        return None
    w, h = get_image_dimensions(image_path)
    gt_boxes: list[BBox] = []
    gt_polys: list[Polygon] = []
    pred_boxes: list = []
    pred_polys: list = []

    detect_label = find_gt_label(image_path, "detect", fmt="yolo")
    if detect_label:
        gt_boxes, _ = parse_detect_labels(str(detect_label), w, h)
    segment_label = find_gt_label(image_path, "segment", fmt="yolo")
    if segment_label:
        gt_polys, _ = parse_segment_labels(str(segment_label), w, h)
    detect_pred = find_prediction(image_path, "detect", fmt="yolo")
    if detect_pred:
        pred_boxes, _ = parse_detect_predictions(str(detect_pred), w, h)
    segment_pred = find_prediction(image_path, "segment", fmt="yolo")
    if segment_pred:
        pred_polys, _ = parse_segment_predictions(str(segment_pred), w, h)

    iou_type, record = records_from_annotation(gt_boxes, gt_polys, pred_boxes, pred_polys, width=w, height=h)
    return iou_type, record, (gt_boxes, gt_polys, pred_boxes, pred_polys), w, h


@mcp.tool()
@audited
def evaluate_detections(
    image_path: str,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> dict:
    """Match predictions against ground truth for a single image (COCOeval).

    mAP / TP / FP / FN come from pycocotools; the ``matches`` block is a
    per-box overlay for the GUI review panel (``compute_matches``).

    Args:
        image_path: Absolute path to the image file.
        iou_threshold: IoU threshold for a positive match.
        conf_threshold: Minimum confidence to consider a prediction.
    """
    loaded = _load_image_annotations(image_path)
    if loaded is None:
        return {"error": f"Image not found: {image_path}"}
    iou_type, record, (gt_boxes, gt_polys, pred_boxes, pred_polys), _w, _h = loaded

    from tcip_mcp.pipelines.training.evaluation import coco_detection_metrics
    m = coco_detection_metrics([record], iou_type=iou_type,
                               iou_threshold=iou_threshold, conf_threshold=conf_threshold)
    matches = compute_matches(
        gt_boxes, gt_polys, pred_boxes, pred_polys,
        iou_threshold=iou_threshold, conf_threshold=conf_threshold,
    )
    return {
        "image": image_path,
        "tp": m["tp"],
        "fp": m["fp"],
        "fn": m["fn"],
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
        "map50": round(m["map50"], 4),
        "iou_type": iou_type,
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
    # Recurse to catch the canonical images/<date>/ layout.
    images = sorted(
        f for f in images_dir.rglob("*") if f.is_file() and f.suffix.lower() in image_exts
    )

    from tcip_mcp.pipelines.training.evaluation import coco_detection_metrics, records_from_annotation

    collected = []  # (iou_type, record, raw, w, h, img)
    for img in images:
        loaded = _load_image_annotations(str(img))
        if loaded is None:
            continue
        iou_type, record, raw, w, h = loaded
        collected.append((iou_type, record, raw, w, h, img))

    any_segm = any(c[0] == "segm" for c in collected)
    dataset_iou_type = "segm" if any_segm else "bbox"
    # When mixed, rebuild every record forcing segmentation so one COCOeval pass works.
    if any_segm:
        records = [records_from_annotation(*raw, width=w, height=h, force_segm=True)[1]
                   for (_it, _rec, raw, w, h, _img) in collected]
    else:
        records = [c[1] for c in collected]
    valid_images = [c[5] for c in collected]

    m = coco_detection_metrics(records, iou_type=dataset_iou_type,
                               iou_threshold=iou_threshold, conf_threshold=conf_threshold)

    counts_by_id = {c["image_id"]: c for c in m["per_image_counts"]}
    per_image = []
    for idx, img in enumerate(valid_images, start=1):
        c = counts_by_id.get(idx, {"tp": 0, "fp": 0, "fn": 0})
        per_image.append({"image": img.name, "tp": c["tp"], "fp": c["fp"], "fn": c["fn"]})

    return {
        "path": folder_path,
        "image_count": len(images),
        "map": round(m["map"], 4),
        "map50": round(m["map50"], 4),
        "total_tp": m["tp"],
        "total_fp": m["fp"],
        "total_fn": m["fn"],
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
        "iou_type": dataset_iou_type,
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

    # Get image dimensions
    img_w, img_h = get_image_dimensions(image_path)

    gt_det = find_gt_label(image_path, "detect", fmt="yolo")
    gt_seg = find_gt_label(image_path, "segment", fmt="yolo")
    pred_det = find_prediction(image_path, "detect", fmt="yolo")
    pred_seg = find_prediction(image_path, "segment", fmt="yolo")

    gt_boxes, _ = parse_detect_labels(str(gt_det), img_w, img_h) if gt_det else ([], set())
    gt_polys, _ = parse_segment_labels(str(gt_seg), img_w, img_h) if gt_seg else ([], set())
    pred_boxes, _ = (
        parse_detect_predictions(str(pred_det), img_w, img_h) if pred_det else ([], set())
    )
    pred_polys, _ = (
        parse_segment_predictions(str(pred_seg), img_w, img_h) if pred_seg else ([], set())
    )

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
            'inference', or 'results'.
        event_type: Event type the panel switches on (e.g. 'load_matches',
            'metrics_update').
        data: Arbitrary JSON data payload.
    """
    from tcip_mcp.web_client import post_panel_event

    valid_panels = {"annotate", "review", "training", "tuning", "inference", "results"}
    if panel not in valid_panels:
        return {"error": f"Unknown panel: {panel}. Valid: {sorted(valid_panels)}"}

    result = post_panel_event(panel, event_type, data)
    result.setdefault("panel", panel)
    result.setdefault("event_type", event_type)
    return result


@mcp.tool()
@audited
def focus_annotate(
    project_root: str,
    dataset_root: str,
    trait: str,
    date: str,
    mode: str | None = None,
    image_index: int | None = None,
) -> dict:
    """Drive the live Annotate tab to a (trait, date), in the right mode, on an annotated frame.

    Switching *only* the dataset selection lands the browser at image 0 in **box** mode — which
    hides polygons and usually shows an unannotated frame, so the human sees a blank canvas
    (the exact failure a real session hit on "switch to the bush polygons"). This posts an
    explicit ``annotate_focus`` event the GUI honors with local view setters — the same path the
    Review→Edit button uses — so the tab lands on the FIRST annotated image in the correct mode
    with no manual steps. Requires the GUI to be running; returns ``delivered: false`` if not.

    Args:
        project_root: Project root (== dataset_root for workspace projects).
        dataset_root: Dataset root holding ``images/`` and ``annotations/``.
        trait: Annotation campaign (e.g. "catkin", "bush").
        date: Capture-date bucket (e.g. "2026-03-02").
        mode: "box" or "polygon". Default: inferred from the labels present on that frame
            (segment → polygon, detect → box).
        image_index: Index into the date's sorted image list. Default: the first image with a
            non-empty label, so the canvas isn't blank.
    """
    from pathlib import Path

    from tcip_mcp.dataset_layout import annotation_dir, image_dir
    from tcip_mcp.web_client import post_panel_event

    img_exts = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".bmp"}
    idir = Path(image_dir(dataset_root, date))
    if not idir.is_dir():
        return {"error": f"no images for date {date} under {dataset_root}"}
    # Match the /api/dataset/select listing EXACTLY (p.is_file() + same ext set + sorted), so
    # the resolved index lines up with the frontend's image_list, not one frame off.
    images = sorted(p.name for p in idir.iterdir() if p.is_file() and p.suffix.lower() in img_exts)
    if not images:
        return {"error": f"no images on {date}"}

    seg_dir = Path(annotation_dir(dataset_root, trait, date, "segment"))
    det_dir = Path(annotation_dir(dataset_root, trait, date, "detect"))

    def _label_task(stem: str) -> str | None:
        # "segment"/"detect" if a NON-EMPTY label exists (something to show), else None.
        for task, d in (("segment", seg_dir), ("detect", det_dir)):
            f = d / f"{stem}.txt"
            if f.is_file() and f.stat().st_size > 0:
                return task
        return None

    def _first_class(stem: str, task: str) -> int | None:
        d = seg_dir if task == "segment" else det_dir
        try:
            for line in (d / f"{stem}.txt").read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts:
                    return int(float(parts[0]))
        except (OSError, ValueError):
            pass
        return None

    n_annotated = 0
    first_idx: int | None = None
    for i, name in enumerate(images):
        if _label_task(Path(name).stem) is not None:
            n_annotated += 1
            if first_idx is None:
                first_idx = i

    if image_index is None:
        image_index = first_idx if first_idx is not None else 0
    image_index = max(0, min(image_index, len(images) - 1))

    # Mode + active class come from the frame ACTUALLY shown (images[image_index]) — not the
    # first-annotated frame — so an explicit image_index gets a consistent mode/class. The
    # canvas only renders shapes of the active class, so setting it is what keeps it non-blank.
    resolved_stem = Path(images[image_index]).stem
    resolved_task = _label_task(resolved_stem)
    if mode is None:
        mode = "polygon" if resolved_task == "segment" else "box"
    if mode not in ("box", "polygon"):
        return {"error": f"mode must be 'box' or 'polygon', got {mode!r}"}
    active_class = _first_class(resolved_stem, resolved_task) if resolved_task else None

    payload = {
        "project_root": project_root,
        "dataset_root": dataset_root,
        "trait": trait,
        "date": date,
        "image_index": image_index,
        "mode": mode,
    }
    if active_class is not None:
        payload["active_class"] = active_class

    result = post_panel_event("app", "annotate_focus", payload)
    return {
        "delivered": result.get("delivered", False),
        "status": result.get("status"),
        "trait": trait,
        "date": date,
        "image_index": image_index,
        "mode": mode,
        "active_class": active_class,
        "n_images": len(images),
        "n_annotated": n_annotated,
        "image": images[image_index],
    }


@mcp.tool()
@audited
def focus_review(
    project_root: str,
    dataset_root: str,
    trait: str,
    date: str,
    model_name: str,
    image_index: int | None = None,
    detection_idx: int = 0,
    filter_type: str = "all",
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> dict:
    """Drive the live Review tab to a model's predictions on a frame, so the human sees exactly
    what the agent flagged (a false positive, a missed catkin) without hunting for it.

    Posts a ``review_focus`` event the GUI honors with local setters — the Review analog of
    ``focus_annotate`` — landing on the (trait, date) with ``model_name``'s predictions loaded,
    on ``image_index`` (default: the first frame that has predictions for this model, so the
    canvas isn't empty), at ``detection_idx`` under the given filter. Requires the GUI to be
    running; returns ``delivered: false`` if not.

    Args:
        project_root: Project root (== dataset_root for workspace projects).
        dataset_root: Dataset root holding ``images/``, ``annotations/``, ``predictions/``.
        trait: Annotation campaign (e.g. "catkin").
        date: Capture-date bucket (e.g. "2026-02-11").
        model_name: The model whose ``predictions/<model>/<date>/`` to review.
        image_index: Index into the date's sorted image list. Default: the first image that has
            a non-empty prediction file for this model.
        detection_idx: Which detection to center in the Review navigator.
        filter_type: "all" | "tp" | "fp" | "fn" — the Review match filter to apply.
        iou_threshold: IoU cutoff for the TP/FP/FN match classification.
        conf_threshold: Confidence cutoff for showing predictions.
    """
    from tcip_mcp.dataset_layout import image_dir, prediction_dir
    from tcip_mcp.web_client import post_panel_event

    if filter_type not in ("all", "tp", "fp", "fn"):
        return {"error": f"filter_type must be all|tp|fp|fn, got {filter_type!r}"}

    img_exts = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".bmp"}
    idir = Path(image_dir(dataset_root, date))
    if not idir.is_dir():
        return {"error": f"no images for date {date} under {dataset_root}"}
    # Match /api/dataset/select's listing exactly so the resolved index lines up with the frontend.
    images = sorted(p.name for p in idir.iterdir() if p.is_file() and p.suffix.lower() in img_exts)
    if not images:
        return {"error": f"no images on {date}"}

    pred_dir = Path(prediction_dir(dataset_root, model_name, date, "detect"))

    def _has_pred(stem: str) -> bool:
        f = pred_dir / f"{stem}.txt"
        return f.is_file() and f.stat().st_size > 0

    n_with_preds = 0
    first_idx: int | None = None
    for i, name in enumerate(images):
        if _has_pred(Path(name).stem):
            n_with_preds += 1
            if first_idx is None:
                first_idx = i

    if image_index is None:
        image_index = first_idx if first_idx is not None else 0
    image_index = max(0, min(image_index, len(images) - 1))

    payload = {
        "project_root": project_root,
        "dataset_root": dataset_root,
        "trait": trait,
        "date": date,
        "model_name": model_name,
        "image_index": image_index,
        "detection_idx": detection_idx,
        "filter_type": filter_type,
        "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold,
    }
    result = post_panel_event("app", "review_focus", payload)
    return {
        "delivered": result.get("delivered", False),
        "status": result.get("status"),
        "trait": trait,
        "date": date,
        "model_name": model_name,
        "image_index": image_index,
        "detection_idx": detection_idx,
        "filter_type": filter_type,
        "n_images": len(images),
        "n_with_predictions": n_with_preds,
        "image": images[image_index],
    }


@mcp.tool()
@audited
def stage_proposals(
    dataset_root: str,
    model_name: str,
    date: str,
    stem: str,
    boxes: list[dict],
) -> dict:
    """Stage agent-proposed detections to ``predictions/<model>/<date>/detect/<stem>.txt`` for
    canvas review — the "show on canvas before writing ground truth" guardrail.

    Proposals (a corrected detection, a SAM output, a model prediction the agent wants a human to
    vet) go to the PREDICTIONS tree, never ``annotations/``, so the human reviews them on the
    Review canvas and accepts/rejects/edits before they become GT. This never writes ground truth.
    Pair with ``focus_review`` to send the human straight to them.

    Args:
        dataset_root: Dataset root holding ``predictions/``.
        model_name: Predictions bucket to stage under (e.g. "agent_proposals").
        date: Capture-date bucket (e.g. "2026-02-11").
        stem: Image stem (filename without extension).
        boxes: ``[{class_id, conf, cx, cy, w, h}]`` with cx/cy/w/h normalized to [0, 1].
    """
    from tcip_mcp.dataset_layout import prediction_dir
    from tcip_mcp.utils.atomic_io import atomic_write_text
    from tcip_mcp.workspace import is_valid_name

    # Confine the path segments so a malformed model/date/stem (an absolute path, a stray ``..``)
    # can't escape predictions/ into annotations/ — this tool's whole promise is "never GT".
    for label, val in (("model_name", model_name), ("date", date), ("stem", stem)):
        if not is_valid_name(val):
            return {"error": f"{label} must be a single safe path segment (no separators/'..'), got {val!r}"}

    lines: list[str] = []
    for i, b in enumerate(boxes):
        try:
            cx, cy, w, h = float(b["cx"]), float(b["cy"]), float(b["w"]), float(b["h"])
            cls = int(b.get("class_id", 0))
            conf = float(b.get("conf", 1.0))
        except (KeyError, TypeError, ValueError):
            return {"error": f"box {i} needs numeric class_id, conf, cx, cy, w, h (normalized): {b!r}"}
        # Guard against pixel coords slipping in — they'd render off-canvas and corrupt the review.
        if any(v < -0.01 or v > 1.5 for v in (cx, cy, w, h)):
            return {"error": (f"box {i} coords {(cx, cy, w, h)} look un-normalized; cx/cy/w/h must be "
                              f"in [0,1] (divide pixel coords by image width/height)")}
        lines.append(f"{cls} {conf:.4f} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    pdir = Path(prediction_dir(dataset_root, model_name, date, "detect"))
    pdir.mkdir(parents=True, exist_ok=True)
    out = pdir / f"{stem}.txt"
    atomic_write_text(out, "\n".join(lines) + ("\n" if lines else ""))
    return {
        "staged": len(lines),
        "path": str(out),
        "model_name": model_name,
        "date": date,
        "stem": stem,
        "note": ("staged to predictions/ for canvas review — NOT committed as ground truth; the "
                 "human accepts on the Review tab before it becomes GT (focus_review to send them)"),
    }


@mcp.tool()
@audited
def write_class_map(labels_dir: str, class_names: str = "", output_path: str = "") -> dict:
    """Persist the class map (id -> name/color) derived from a label set to ``classes.json``.

    Enumerates the class ids actually present in ``<labels_dir>/*.txt`` (YOLO ``cls ...`` per line)
    and writes the canonical ``<project>/.tcip/state/classes.json`` the GUI and pipeline read — so
    class identity and ``num_classes`` have one durable, audited source (derived from the labels in
    hand) instead of a pinned integer. ``class_names`` is an optional comma-separated list indexed by
    class id; unnamed ids fall back to ``class_<id>``.
    """
    import json as _json

    from tcip_mcp.project_paths import resolve_state

    ld = Path(labels_dir)
    if not ld.is_dir():
        return {"error": f"labels_dir not found: {labels_dir}"}
    ids: set[int] = set()
    for txt in ld.glob("*.txt"):
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                try:
                    ids.add(int(float(parts[0])))
                except ValueError:
                    continue
    if not ids:
        return {"error": f"no class ids found in {labels_dir}"}
    names = [n.strip() for n in class_names.split(",")] if class_names else []
    palette = ["#FF0000", "#00FFFF", "#00FF00", "#FF00FF", "#FFFF00", "#0000FF", "#FF8000", "#8000FF"]
    class_map = {
        str(cid): {"name": names[cid] if cid < len(names) and names[cid] else f"class_{cid}",
                   "color": palette[cid % len(palette)]}
        for cid in sorted(ids)
    }
    out = Path(output_path) if output_path else resolve_state(Path(".tcip") / "state" / "classes.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(class_map, indent=2), encoding="utf-8")
    return {"classes_path": str(out), "num_classes": max(ids) + 1,
            "class_ids": sorted(ids), "class_map": class_map}
