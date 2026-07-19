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
from tcip_mcp.pipelines.resolution import DEFAULT_CONF
from tcip_mcp.server import mcp
from tcip_mcp.audit import audited


@mcp.tool()
@audited
def load_annotations(image_path: str, fmt: str | None = None) -> dict:
    """Load labels and predictions for a single image.

    Supports the canonical per-image COCO/JSON (.json), plus YOLO (.txt), PASCAL VOC (.xml),
    COCO (.json), and LabelMe (.json). Format is auto-detected from file extension and content
    unless fmt is specified.

    Args:
        image_path: Absolute path to the image file.
        fmt: Force annotation format ('json', 'yolo', 'voc', 'coco', 'labelme'). Auto-detects if omitted.
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

    # Look for predictions (per-image COCO/JSON, parsed by json_io)
    pred_det = find_prediction(image_path, "detect")
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

    pred_seg = find_prediction(image_path, "segment")
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
    fmt: str = "json",
    trait: str = DEFAULT_TRAIT,
    date: str | None = None,
    detect_path: str | None = None,
    segment_path: str | None = None,
    created_by: str | None = None,
) -> dict:
    """Write annotation label files for an image into the canonical dataset layout.

    Labels go to ``<dataset_root>/annotations/<trait>/<date>/<task>/<stem>.<ext>``
    (see :mod:`tcip_mcp.dataset_layout`); ``date`` is derived from the image path
    (``images/<date>/<stem>``) when not given. Pass ``detect_path`` / ``segment_path``
    to write to an explicit location instead. Supports the canonical per-image COCO/JSON
    (.json), plus YOLO (.txt), PASCAL VOC (.xml), and LabelMe (.json).

    Args:
        image_path: Absolute path to the image file.
        boxes: List of dicts with x1, y1, x2, y2, class_id (pixel coords).
        polygons: List of dicts with points and class_id (pixel coords).
        fmt: Output format — 'json' (canonical per-image, default), 'yolo', 'voc', 'labelme', 'coco'.
        trait: Annotation campaign (e.g. 'catkin') — the layout's ``<trait>`` segment.
        date: Capture date; derived from the image path when omitted.
        detect_path: Explicit detect label path (overrides the canonical location).
        segment_path: Explicit segment label path (overrides the canonical location).
        created_by: Producer to stamp on each written shape (e.g. "claude", "model:<run>"). No
            human is present at this tool, so pass the real source; omit to leave provenance unset
            rather than fabricate it. A per-shape ``created_by`` in a box/polygon dict overrides it.
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    # trait/date become path segments under annotations/, so confine them (like stage_proposals)
    # when the canonical layout is used — the explicit *_path args bypass it.
    from tcip_mcp.workspace import is_valid_name

    canonical_used = (boxes is not None and not detect_path) or (polygons is not None and not segment_path)
    if canonical_used:
        if not is_valid_name(trait):
            return {"error": f"trait must be a single safe path segment (no separators/'..'), got {trait!r}"}
        if date is not None and not is_valid_name(date):
            return {"error": f"date must be a single safe path segment (no separators/'..'), got {date!r}"}

    w, h = get_image_dimensions(image_path)
    stem = img.stem

    from datetime import datetime, timezone
    _now = datetime.now(timezone.utc).isoformat()

    def _prov(cb):  # created_at accompanies created_by; both stay unset when there's no producer
        return (cb, _now) if cb else (None, None)

    written: list[str] = []

    if boxes is not None:
        typed_boxes = []
        for b in boxes:
            cb, ca = _prov(b.get("created_by", created_by))
            typed_boxes.append(BBox(x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"],
                                    class_id=b["class_id"], created_by=cb, created_at=ca))
        out_path = (
            Path(detect_path)
            if detect_path
            else annotation_path_for_image(image_path, "detect", fmt, trait=trait, date=date)
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        format_save(str(out_path), typed_boxes, w, h, task="detect", fmt=fmt, file_name=img.name,
                    keep_empty=True)
        written.append(str(out_path))

    if polygons is not None:
        typed_polys = []
        for p in polygons:
            cb, ca = _prov(p.get("created_by", created_by))
            typed_polys.append(Polygon(points=[(pt[0], pt[1]) for pt in p["points"]],
                                       class_id=p["class_id"], created_by=cb, created_at=ca))
        out_path = (
            Path(segment_path)
            if segment_path
            else annotation_path_for_image(image_path, "segment", fmt, trait=trait, date=date)
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        format_save(str(out_path), typed_polys, w, h, task="segment", fmt=fmt, file_name=img.name,
                    keep_empty=True)
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

    detect_label = find_gt_label(image_path, "detect")
    if detect_label:
        gt_boxes, _ = parse_detect_labels(str(detect_label), w, h)
    segment_label = find_gt_label(image_path, "segment")
    if segment_label:
        gt_polys, _ = parse_segment_labels(str(segment_label), w, h)
    detect_pred = find_prediction(image_path, "detect")
    if detect_pred:
        pred_boxes, _ = parse_detect_predictions(str(detect_pred), w, h)
    segment_pred = find_prediction(image_path, "segment")
    if segment_pred:
        pred_polys, _ = parse_segment_predictions(str(segment_pred), w, h)

    iou_type, record = records_from_annotation(gt_boxes, gt_polys, pred_boxes, pred_polys, width=w, height=h)
    return iou_type, record, (gt_boxes, gt_polys, pred_boxes, pred_polys), w, h


def _detection_breakdown(matches, gt_boxes, gt_polys, pred_boxes, pred_polys) -> list[dict]:
    """Per-detection TP/FP/FN records (box/polygon coords + IoU + confidence) from a match result."""
    detections: list[dict] = []
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
    return detections


def _evaluate_image(
    image_path: str,
    iou_threshold: float = 0.5,
    conf_threshold: float = DEFAULT_CONF,
    detail: bool = False,
) -> dict:
    """Match predictions against ground truth for a single image (COCOeval).

    mAP / TP / FP / FN come from pycocotools; the ``matches`` block is a
    per-box overlay for the GUI review panel (``compute_matches``).
    """
    loaded = _load_image_annotations(image_path)
    if loaded is None:
        return {"error": f"Image not found: {image_path}"}
    iou_type, record, (gt_boxes, gt_polys, pred_boxes, pred_polys), w, h = loaded

    from tcip_mcp.pipelines.training.evaluation import coco_detection_metrics
    m = coco_detection_metrics([record], iou_type=iou_type,
                               iou_threshold=iou_threshold, conf_threshold=conf_threshold)
    matches = compute_matches(
        gt_boxes, gt_polys, pred_boxes, pred_polys,
        iou_threshold=iou_threshold, conf_threshold=conf_threshold,
    )
    out = {
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
    if detail:
        out["img_w"] = w
        out["img_h"] = h
        out["detections"] = _detection_breakdown(matches, gt_boxes, gt_polys, pred_boxes, pred_polys)
    return out


def _evaluate_folder(
    folder_path: str,
    iou_threshold: float = 0.5,
    conf_threshold: float = DEFAULT_CONF,
) -> dict:
    """Aggregate detection metrics across all images in a dataset."""
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
def evaluate_predictions(
    path: str,
    iou_threshold: float = 0.5,
    conf_threshold: float = DEFAULT_CONF,
    detail: bool = False,
) -> dict:
    """Score on-disk predictions against on-disk ground truth (COCOeval).

    Dispatches on the input: a single image file returns per-box ``matches`` (plus an optional
    per-detection ``detections`` breakdown with ``img_w`` / ``img_h`` when ``detail=True``) for
    the GUI review panel; a dataset directory returns aggregate metrics plus ``per_image`` TP/FP/FN.
    Both regimes share ``coco_detection_metrics``.

    Args:
        path: Absolute path to an image file (single-image match) or a dataset root (aggregate).
        iou_threshold: IoU threshold for a positive match.
        conf_threshold: Minimum confidence to consider a prediction. Defaults to the shared
            ``DEFAULT_CONF`` so the reported operating point matches evaluate_model / inference.
        detail: Single-image only — also return the per-detection ``detections`` breakdown.
    """
    p = Path(path)
    if p.is_file():
        return _evaluate_image(path, iou_threshold, conf_threshold, detail)
    if p.is_dir():
        return _evaluate_folder(path, iou_threshold, conf_threshold)
    return {"error": f"Path not found: {path}"}


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
            'inference', 'results', or 'app' (app-level events like
            annotate_focus / review_focus / project_changed).
        event_type: Event type the panel switches on (e.g. 'load_matches',
            'metrics_update').
        data: Arbitrary JSON data payload.
    """
    from tcip_mcp.web_client import post_panel_event

    # 'app' is a real target: focus_annotate/focus_review/set_active_project all post to it.
    valid_panels = {"app", "annotate", "review", "training", "tuning", "inference", "results"}
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
    # Match the /api/dataset/select listing exactly (p.is_file() + same ext set + sorted), so
    # the resolved index lines up with the frontend's image_list, not one frame off.
    images = sorted(p.name for p in idir.iterdir() if p.is_file() and p.suffix.lower() in img_exts)
    if not images:
        return {"error": f"no images on {date}"}

    seg_dir = Path(annotation_dir(dataset_root, trait, date, "segment"))
    det_dir = Path(annotation_dir(dataset_root, trait, date, "detect"))

    def _label_task(stem: str) -> str | None:
        # "segment"/"detect" if a label with objects exists (something to show), else None. A
        # confirmed negative (a present {"objects": []}) has size > 0 but nothing to show — read it.
        for task, d in (("segment", seg_dir), ("detect", det_dir)):
            f = d / f"{stem}.json"
            if not f.is_file():
                continue
            shapes, _ = (parse_segment_labels if task == "segment" else parse_detect_labels)(str(f))
            if shapes:
                return task
        return None

    def _first_class(stem: str, task: str) -> int | None:
        d = seg_dir if task == "segment" else det_dir
        parser = parse_segment_labels if task == "segment" else parse_detect_labels
        shapes, _ = parser(str(d / f"{stem}.json"))
        return shapes[0].class_id if shapes else None

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

    # Mode + active class come from the frame actually shown (images[image_index]) — not the
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
    conf_threshold: float = DEFAULT_CONF,
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
    from tcip_mcp.workspace import is_valid_name

    if filter_type not in ("all", "tp", "fp", "fn"):
        return {"error": f"filter_type must be all|tp|fp|fn, got {filter_type!r}"}

    # model_name/date become path segments, so confine them like stage_proposals (read-only here,
    # but keeps a traversal segment from probing paths / a later refactor becoming a write path).
    for label, val in (("model_name", model_name), ("date", date)):
        if not is_valid_name(val):
            return {"error": f"{label} must be a single safe path segment (no separators/'..'), got {val!r}"}

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
        boxes, _ = parse_detect_predictions(str(pred_dir / f"{stem}.json"))
        return bool(boxes)

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
    boxes: list[dict] | None = None,
    polygons: list[dict] | None = None,
    overwrite: bool = False,
) -> dict:
    """Stage model-/agent-proposed shapes to ``predictions/<model>/<date>/<task>/<stem>.json`` for
    canvas review — the "show on canvas before writing ground truth" guardrail.

    Anything a model produces (a SAM mask, a groundingDINO/baseline detection, a shape the agent
    wants a human to vet) goes to the PREDICTIONS tree, never ``annotations/``, so the human reviews
    it on the Review canvas and accepts/rejects/edits before it becomes GT. Only human-accepted
    shapes reach ground truth. Boxes land under ``detect/``, polygons (e.g. SAM masks) under
    ``segment/`` — pass either or both. This never writes ground truth. Pair with ``focus_review``
    to send the human straight to them.

    A prediction bucket that already carries review verdicts is immutable: by default a stage into
    it is redirected to a fresh run-scoped bucket (``<model>@r2``, ``@r3`` — next free), and the
    bucket actually written is returned as ``bucket``. Pass ``overwrite=True`` to write in place
    only when the bucket has zero verdicts; with verdicts present it is refused (error names the
    count and a suggested bucket) so a re-run never orphans recorded verdicts.

    Args:
        dataset_root: Dataset root holding ``predictions/``.
        model_name: Predictions bucket to stage under — the real producer, one per source (e.g.
            "sam", "claude", "groundingdino", "model:<run>"). It is stamped as each object's
            created_by, so name the actual origin rather than a generic placeholder.
        date: Capture-date bucket (e.g. "2026-02-11").
        stem: Image stem (filename without extension).
        boxes: ``[{class_id, conf, cx, cy, w, h}]`` with cx/cy/w/h normalized to [0, 1].
        polygons: ``[{class_id, conf, points: [[x, y], ...]}]`` with points normalized to [0, 1]
            (>=3 points each) — SAM's mask-quality score is a natural ``conf``.
        overwrite: Write in place even into an existing bucket. Refused if the bucket has review
            verdicts; the default (False) auto-redirects to a fresh bucket instead.
    """
    from tcip_annotation import json_io
    from tcip_annotation.state import PredBBox, PredPolygon

    from tcip_mcp.dataset_layout import image_dir, prediction_dir
    from tcip_mcp.prediction_buckets import BucketHasVerdicts, resolve_writable_bucket
    from tcip_mcp.workspace import is_valid_name

    # Confine the path segments so a malformed model/date/stem (an absolute path, a stray ``..``)
    # can't escape predictions/ into annotations/ — this tool's whole promise is "never GT".
    for label, val in (("model_name", model_name), ("date", date), ("stem", stem)):
        if not is_valid_name(val):
            return {"error": f"{label} must be a single safe path segment (no separators/'..'), got {val!r}"}

    boxes = boxes or []
    polygons = polygons or []
    if not boxes and not polygons:
        return {"error": "provide at least one of boxes or polygons to stage"}

    def _unnormalized(vals: tuple[float, ...]) -> bool:
        # Pixel coords slipping in would render off-canvas and corrupt the review.
        return any(v < -0.01 or v > 1.5 for v in vals)

    # Validate every normalized shape before touching the image or writing anything, so a bad shape
    # can't leave a partial stage. json_io is pixel-space, so we denormalize after resolving dims.
    norm_boxes: list[tuple[int, float, float, float, float, float]] | None = None
    if boxes:
        norm_boxes = []
        for i, b in enumerate(boxes):
            try:
                cx, cy, w, h = float(b["cx"]), float(b["cy"]), float(b["w"]), float(b["h"])
                cls = int(b.get("class_id", 0))
                conf = float(b.get("conf", 1.0))
            except (KeyError, TypeError, ValueError):
                return {"error": f"box {i} needs numeric class_id, conf, cx, cy, w, h (normalized): {b!r}"}
            if _unnormalized((cx, cy, w, h)):
                return {"error": (f"box {i} coords {(cx, cy, w, h)} look un-normalized; cx/cy/w/h must be "
                                  f"in [0,1] (divide pixel coords by image width/height)")}
            norm_boxes.append((cls, conf, cx, cy, w, h))

    norm_polys: list[tuple[int, float, list[tuple[float, float]]]] | None = None
    if polygons:
        norm_polys = []
        for i, p in enumerate(polygons):
            try:
                cls = int(p.get("class_id", 0))
                conf = float(p.get("conf", 1.0))
                pts = [(float(x), float(y)) for x, y in p["points"]]
            except (KeyError, TypeError, ValueError):
                return {"error": f"polygon {i} needs class_id, conf, points [[x,y],...] (normalized): {p!r}"}
            if len(pts) < 3:
                return {"error": f"polygon {i} needs at least 3 points, got {len(pts)}"}
            flat = tuple(v for xy in pts for v in xy)
            if _unnormalized(flat):
                return {"error": (f"polygon {i} points look un-normalized; x/y must be in [0,1] "
                                  f"(divide pixel coords by image width/height)")}
            norm_polys.append((cls, conf, pts))

    # Resolve the source image to convert normalized shapes to pixel space (json_io is pixel-space).
    img_file = None
    for idir in (image_dir(dataset_root, date), image_dir(dataset_root, None)):
        for cand in sorted(Path(idir).glob(f"{stem}.*")):
            if cand.is_file():
                img_file = cand
                break
        if img_file is not None:
            break
    if img_file is None:
        return {"error": f"no image found for stem {stem!r} under {image_dir(dataset_root, date)}"}
    img_w, img_h = get_image_dimensions(str(img_file))

    # Prediction-bucket immutability: don't overwrite a bucket that has review verdicts. Verdicts
    # (and the predictions they reference) colocate under the dataset's ``.tcip/state``.
    review_state_dir = Path(dataset_root) / ".tcip" / "state"

    def _bucket_dirs(name: str) -> list[Path]:
        return [Path(prediction_dir(dataset_root, name, date, "detect")),
                Path(prediction_dir(dataset_root, name, date, "segment"))]

    try:
        resolution = resolve_writable_bucket(review_state_dir, model_name, _bucket_dirs, overwrite=overwrite)
    except BucketHasVerdicts as exc:
        return {"error": str(exc), "verdict_count": exc.count, "suggested_bucket": exc.suggested}
    bucket = resolution.name

    # Stamp the real producer (model_name) as created_by + a stage-time created_at, so a staged
    # prediction's origin travels into GT natively when a human accepts it on the Review canvas.
    from datetime import datetime, timezone
    created_at = datetime.now(timezone.utc).isoformat()

    detect_path = None
    if norm_boxes is not None:
        pred_boxes = [
            PredBBox(
                (cx - w / 2) * img_w, (cy - h / 2) * img_h,
                (cx + w / 2) * img_w, (cy + h / 2) * img_h,
                cls, confidence=conf, created_by=model_name, created_at=created_at,
            )
            for (cls, conf, cx, cy, w, h) in norm_boxes
        ]
        out = Path(prediction_dir(dataset_root, bucket, date, "detect")) / f"{stem}.json"
        json_io.write_detect(out, pred_boxes, img_w, img_h)
        detect_path = str(out)

    segment_path = None
    if norm_polys is not None:
        pred_polys = [
            PredPolygon([(x * img_w, y * img_h) for x, y in pts], cls,
                        confidence=conf, created_by=model_name, created_at=created_at)
            for (cls, conf, pts) in norm_polys
        ]
        out = Path(prediction_dir(dataset_root, bucket, date, "segment")) / f"{stem}.json"
        json_io.write_segment(out, pred_polys, img_w, img_h)
        segment_path = str(out)

    note = ("staged to predictions/ for canvas review — not committed as ground truth; the "
            "human accepts on the Review tab before it becomes GT (focus_review to send them)")
    if resolution.redirected:
        note = (f"bucket {model_name!r} has {resolution.verdict_count} review verdict(s) — staged to a "
                f"fresh bucket {bucket!r} instead so the reviewed predictions stay intact; " + note)

    return {
        "staged": len(boxes) + len(polygons),
        "n_detect": len(boxes),
        "n_segment": len(polygons),
        "detect_path": detect_path,
        "segment_path": segment_path,
        "model_name": model_name,
        "bucket": bucket,
        "bucket_redirected": resolution.redirected,
        "date": date,
        "stem": stem,
        "note": note,
    }


@mcp.tool()
@audited
def write_class_map(labels_dir: str, class_names: str = "", output_path: str = "") -> dict:
    """Persist the class map (id -> name/color) derived from a label set to ``classes.json``.

    Enumerates the class ids actually present in ``<labels_dir>/*.json`` (per-image COCO/JSON,
    each object's ``category_id``) and writes the canonical ``<project>/.tcip/state/classes.json``
    the GUI and pipeline read — so
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
    for jf in ld.glob("*.json"):
        try:
            data = _json.loads(jf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        for o in data.get("objects") or []:
            if isinstance(o, dict) and "category_id" in o:
                try:
                    ids.add(int(o["category_id"]))
                except (TypeError, ValueError):
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
