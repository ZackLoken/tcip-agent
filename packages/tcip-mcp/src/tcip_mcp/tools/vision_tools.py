"""Vision tools — render annotations and predictions for visual analysis.

Each tool saves a rendered image to .tcip/artifacts/viz/ and returns the
path so the agent can use view_image to visually inspect it.
"""

from __future__ import annotations

import random
from pathlib import Path

from tcip_annotation.utils import get_image_dimensions
from tcip_annotation.viz import (
    render_candidates,
    render_canvas_state,
    render_comparison,
    render_confusion_examples,
    render_detections,
    render_grid,
    render_grid_overlay,
    render_segmentations,
)

from tcip_mcp.audit import audited
from tcip_mcp.pipelines.resolution import DEFAULT_CONF
from tcip_mcp.server import mcp


@mcp.tool()
@audited
def visualize(
    source: str,
    path: str,
    task: str = "detect",
    class_names: str = "",
    conf_threshold: float = DEFAULT_CONF,
    iou_threshold: float = 0.5,
    n: int = 16,
) -> dict:
    """Render annotations, predictions, a GT-vs-prediction comparison, or a sample grid.

    One entry point for the common renders (replaces the former visualize_annotations /
    visualize_predictions / visualize_comparison / visualize_dataset_sample). Saves to
    .tcip/artifacts/viz/ and returns ``image_path`` for view_image.

    Args:
        source: What to render —
            'annotations' = ground-truth labels on a single image (path = image file);
            'predictions' = model predictions on a single image (path = image file);
            'comparison'  = GT (green) vs predictions (red) with TP/FP/FN match stats
            (path = image file);
            'dataset'     = grid of n random annotated samples (path = dataset folder
            containing images/ and labels/).
        path: Image file (annotations/predictions/comparison) or dataset folder (dataset).
        task: 'detect' or 'segment'.
        class_names: Comma-separated class names (e.g. "catkin,nut,bud").
        conf_threshold: Minimum confidence — filters displayed predictions (source='predictions')
            and the predictions matched against GT (source='comparison'). Defaults to the shared
            ``DEFAULT_CONF`` so the comparison operating point matches inference/evaluate (it used to
            silently match at compute_matches' own 0.25 default).
        iou_threshold: IoU threshold for a positive match (source='comparison' only).
        n: Number of samples in the grid (source='dataset' only).
    """
    if source == "annotations":
        return _viz_annotations(path, task=task, class_names=class_names)
    if source == "predictions":
        return _viz_predictions(
            path, task=task, class_names=class_names, conf_threshold=conf_threshold
        )
    if source == "comparison":
        return _viz_comparison(
            path, task=task, iou_threshold=iou_threshold, class_names=class_names,
            conf_threshold=conf_threshold,
        )
    if source == "dataset":
        return _viz_dataset_sample(path, n=n, task=task, class_names=class_names)
    return {
        "error": f"Unknown source '{source}'. "
        "Use 'annotations', 'predictions', 'comparison', or 'dataset'."
    }


def _viz_annotations(
    image_path: str,
    task: str = "detect",
    class_names: str = "",
) -> dict:
    """Render ground-truth annotations on a single image. See ``visualize``."""
    from tcip_annotation.format_io import (
        detect_format,
        load_annotations as format_load,
    )
    from tcip_mcp.dataset_layout import find_gt_label

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    w, h = get_image_dimensions(image_path)
    stem = img.stem

    name_map = {}
    if class_names:
        name_map = {i: n.strip() for i, n in enumerate(class_names.split(","))}

    label_path = find_gt_label(image_path, task)
    if label_path is None:
        return {"error": f"No {task} labels found for {stem}"}

    fmt = detect_format(str(label_path))

    if task == "detect":
        boxes, class_ids = format_load(str(label_path), w, h, task="detect", fmt=fmt)
        box_dicts = [
            {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "class_id": b.class_id}
            for b in boxes
        ]
        out = render_detections(image_path, box_dicts, class_names=name_map)
        summary = f"Rendered {len(boxes)} detections on {img.name}"
        if name_map:
            from collections import Counter
            counts = Counter(name_map.get(b.class_id, str(b.class_id)) for b in boxes)
            summary += " — " + ", ".join(f"{v} {k}" for k, v in counts.most_common())
    else:
        polys, class_ids = format_load(str(label_path), w, h, task="segment", fmt=fmt)
        poly_dicts = [
            {"points": [(pt[0], pt[1]) for pt in p.points], "class_id": p.class_id}
            for p in polys
        ]
        out = render_segmentations(image_path, poly_dicts, class_names=name_map)
        summary = f"Rendered {len(polys)} segmentation masks on {img.name}"

    _count = len(boxes) if task == "detect" else len(polys)
    return {
        "image_path": out,
        "summary": summary,
        "format": fmt,
        # `count` is the stable key across all visualize sources; the source-specific alias stays.
        "count": _count,
        "annotation_count": _count,
    }


def _viz_predictions(
    image_path: str,
    task: str = "detect",
    class_names: str = "",
    conf_threshold: float = 0.0,
) -> dict:
    """Render model predictions on a single image. See ``visualize``."""
    from tcip_annotation import parse_detect_predictions, parse_segment_predictions

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    w, h = get_image_dimensions(image_path)
    stem = img.stem

    name_map = {}
    if class_names:
        name_map = {i: n.strip() for i, n in enumerate(class_names.split(","))}

    from tcip_mcp.dataset_layout import find_prediction

    pred_file = find_prediction(image_path, task)
    if pred_file is None:
        return {"error": f"No {task} predictions found for {stem}"}

    if task == "detect":
        pred_boxes, class_ids = parse_detect_predictions(str(pred_file), w, h)
        if conf_threshold > 0:
            pred_boxes = [b for b in pred_boxes if b.confidence >= conf_threshold]
        box_dicts = [
            {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
             "class_id": b.class_id, "confidence": b.confidence}
            for b in pred_boxes
        ]
        out = render_detections(image_path, box_dicts, class_names=name_map)
        summary = f"Rendered {len(pred_boxes)} predictions on {img.name}"
    else:
        pred_polys, class_ids = parse_segment_predictions(str(pred_file), w, h)
        poly_dicts = [
            {"points": [(pt[0], pt[1]) for pt in p.points], "class_id": p.class_id}
            for p in pred_polys
        ]
        out = render_segmentations(image_path, poly_dicts, class_names=name_map)
        summary = f"Rendered {len(pred_polys)} prediction masks on {img.name}"

    _count = len(pred_boxes) if task == "detect" else len(pred_polys)
    return {
        "image_path": out,
        "summary": summary,
        # `count` is the stable key across all visualize sources; the source-specific alias stays.
        "count": _count,
        "prediction_count": _count,
    }


def _viz_comparison(
    image_path: str,
    task: str = "detect",
    iou_threshold: float = 0.5,
    class_names: str = "",
    conf_threshold: float = DEFAULT_CONF,
) -> dict:
    """Render GT vs prediction comparison with match indicators. See ``visualize``.

    Green = ground truth, Red = predictions, Yellow lines = matched pairs.
    """
    from tcip_annotation import parse_detect_predictions
    from tcip_annotation.format_io import (
        detect_format,
        load_annotations as format_load,
    )
    from tcip_annotation.matching import compute_matches
    from tcip_mcp.dataset_layout import find_gt_label

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    w, h = get_image_dimensions(image_path)
    stem = img.stem

    name_map = {}
    if class_names:
        name_map = {i: n.strip() for i, n in enumerate(class_names.split(","))}

    # Load GT
    label_path = find_gt_label(image_path, task)
    if label_path is None:
        return {"error": f"No {task} labels found for {stem}"}

    fmt = detect_format(str(label_path))
    gt_boxes_raw, _ = format_load(str(label_path), w, h, task="detect", fmt=fmt)
    gt_dicts = [
        {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "class_id": b.class_id}
        for b in gt_boxes_raw
    ]

    # Load predictions
    from tcip_mcp.dataset_layout import find_prediction

    pred_file = find_prediction(image_path, task)
    pred_dicts: list[dict] = []
    matches_out: list[dict] = []
    if pred_file is not None:
        pred_boxes_raw, _ = parse_detect_predictions(str(pred_file), w, h)
        pred_dicts = [
            {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
             "class_id": b.class_id, "confidence": b.confidence}
            for b in pred_boxes_raw
        ]
        # Match at the caller's conf operating point (not compute_matches' silent 0.25 default).
        match_result = compute_matches(
            gt_boxes_raw, [], pred_boxes_raw, [],
            iou_threshold=iou_threshold, conf_threshold=conf_threshold,
        )
        tp = match_result.get("tp", 0)
        fp = match_result.get("fp", 0)
        fn = match_result.get("fn", 0)
    else:
        tp, fp, fn = 0, 0, len(gt_boxes_raw)

    out = render_comparison(
        image_path, gt_dicts, pred_dicts,
        matches=matches_out, class_names=name_map,
    )

    return {
        "image_path": out,
        "summary": f"GT={len(gt_dicts)}, Pred={len(pred_dicts)}, TP={tp}, FP={fp}, FN={fn}",
        "gt_count": len(gt_dicts),
        "pred_count": len(pred_dicts),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


@mcp.tool()
@audited
def visualize_worst_predictions(
    predictions_dir: str,
    labels_dir: str,
    images_dir: str = "",
    task: str = "detect",
    top_k: int = 10,
    class_names: str = "",
) -> dict:
    """Find and render the worst predictions for failure analysis.

    Returns a grid image and individual failure case images.

    Args:
        predictions_dir: Directory with prediction files.
        labels_dir: Directory with ground-truth label files.
        images_dir: Directory with source images. Auto-detected if empty.
        task: 'detect' or 'segment'.
        top_k: Number of worst cases to render.
        class_names: Comma-separated class names.
    """
    from tcip_mcp.tools.training_tools import get_worst_predictions

    # Auto-detect images_dir
    if not images_dir:
        labels_path = Path(labels_dir)
        candidate = labels_path.parent.parent / "images"
        if candidate.is_dir():
            images_dir = str(candidate)
        else:
            return {"error": "images_dir not specified and could not be auto-detected"}

    worst = get_worst_predictions(predictions_dir, labels_dir, top_k=top_k)
    if "error" in worst:
        return worst

    worst_items = worst.get("worst_images", [])
    if not worst_items:
        return {"summary": "No prediction errors found", "image_path": None}

    # Build failure case data for render_confusion_examples
    failure_cases = []
    img_dir = Path(images_dir)
    for item in worst_items:
        stem = item["stem"]
        # Find actual image path
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            candidate = img_dir / f"{stem}{ext}"
            if candidate.is_file():
                img_path = str(candidate)
                break
        if not img_path:
            continue

        w, h = get_image_dimensions(img_path)

        # Parse GT boxes
        from tcip_annotation import parse_detect_labels
        gt_file = Path(labels_dir) / f"{stem}.json"
        gt_dicts = []
        if gt_file.is_file():
            gt_boxes, _ = parse_detect_labels(str(gt_file), w, h)
            gt_dicts = [
                {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "class_id": b.class_id}
                for b in gt_boxes
            ]

        # Parse pred boxes
        from tcip_annotation import parse_detect_predictions
        pred_file = Path(predictions_dir) / f"{stem}.json"
        pred_dicts = []
        if pred_file.is_file():
            pred_boxes, _ = parse_detect_predictions(str(pred_file), w, h)
            pred_dicts = [
                {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                 "class_id": b.class_id, "confidence": b.confidence}
                for b in pred_boxes
            ]

        failure_cases.append({
            "image": stem,
            "image_path": img_path,
            "gt_boxes": gt_dicts,
            "pred_boxes": pred_dicts,
            "error_score": item["error_score"],
        })

    # Render individual failure cases
    case_paths = render_confusion_examples(failure_cases, images_dir=images_dir)

    # Render grid of all failure cases
    grid_path = None
    if case_paths:
        titles = [f"{fc['image']} (err={fc['error_score']:.1f})" for fc in failure_cases[:len(case_paths)]]
        grid_path = render_grid(case_paths, titles=titles, cols=min(4, len(case_paths)))

    return {
        "image_path": grid_path,
        "case_images": case_paths,
        "summary": f"Rendered {len(case_paths)} worst prediction cases (of {worst['total_evaluated']} evaluated)",
        "worst_images": worst_items,
    }


def _viz_dataset_sample(
    folder_path: str,
    n: int = 16,
    task: str = "detect",
    class_names: str = "",
) -> dict:
    """Render a grid of random annotated dataset samples. See ``visualize``."""
    from tcip_annotation.format_io import (
        detect_format,
        load_annotations as format_load,
    )
    from tcip_mcp.dataset_layout import find_gt_label

    root = Path(folder_path)
    images_dir = root / "images"
    if not images_dir.is_dir():
        return {"error": f"Images directory not found: {images_dir}"}

    name_map = {}
    if class_names:
        name_map = {i: n_name.strip() for i, n_name in enumerate(class_names.split(","))}

    # Collect all image paths (recurse for the canonical images/<date>/ layout).
    all_images = sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    )
    if not all_images:
        return {"error": "No images found in dataset"}

    # Sample
    sample = random.sample(all_images, min(n, len(all_images)))

    # Render each with annotations
    rendered_paths = []
    titles = []
    for img_path in sample:
        w, h = get_image_dimensions(str(img_path))
        label_path = find_gt_label(str(img_path), task)

        if label_path is not None:
            fmt = detect_format(str(label_path))
            if task == "detect":
                boxes, _ = format_load(str(label_path), w, h, task="detect", fmt=fmt)
                box_dicts = [
                    {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "class_id": b.class_id}
                    for b in boxes
                ]
                out = render_detections(str(img_path), box_dicts, class_names=name_map)
            else:
                polys, _ = format_load(str(label_path), w, h, task="segment", fmt=fmt)
                poly_dicts = [
                    {"points": [(pt[0], pt[1]) for pt in p.points], "class_id": p.class_id}
                    for p in polys
                ]
                out = render_segmentations(str(img_path), poly_dicts, class_names=name_map)
            count = len(boxes) if task == "detect" else len(polys)
            titles.append(f"{img_path.stem} ({count})")
        else:
            # No annotations — just use raw image
            out = str(img_path)
            titles.append(f"{img_path.stem} (no labels)")

        rendered_paths.append(out)

    grid_path = render_grid(rendered_paths, titles=titles, cols=min(4, len(rendered_paths)))

    return {
        "image_path": grid_path,
        "summary": f"Grid of {len(sample)} annotated samples from {root.name}",
        # `count` is the stable key across all visualize sources; the source-specific alias stays.
        "count": len(sample),
        "sample_count": len(sample),
        "total_images": len(all_images),
    }


@mcp.tool()
@audited
def sam_auto_label(
    image_path: str,
    model_type: str = "hiera_b+",
    points_per_side: int = 16,
    pred_iou_thresh: float = 0.86,
    stability_score_thresh: float = 0.92,
    min_mask_region_area: int = 100,
) -> dict:
    """Generate SAM candidate masks and render them for agent review.

    Runs SAM auto-mask generation, renders numbered candidates on the image,
    and returns both the render path and candidate data. The agent should
    use view_image on the rendered output, then call accept_candidates
    to assign classes and save.

    Args:
        image_path: Absolute path to the image file.
        model_type: SAM2 variant (hiera_t / hiera_s / hiera_b+ / hiera_l).
        points_per_side: Grid density for auto-mask generation.
        pred_iou_thresh: Minimum predicted IoU to keep a mask.
        stability_score_thresh: Minimum stability score to keep a mask.
        min_mask_region_area: Minimum mask area in pixels.
    """
    from tcip_annotation.sam_wrapper import auto_mask

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    try:
        candidates = auto_mask(
            image_path,
            model_type=model_type,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
        )
    except ImportError as e:
        return {"error": str(e)}
    except FileNotFoundError as e:
        return {"error": str(e)}

    if not candidates:
        return {
            "image_path": None,
            "summary": "No candidate masks generated",
            "candidates": [],
        }

    out = render_candidates(image_path, candidates)

    # Resolve state via the platform root (like write_class_map), not a CWD-relative path, so the
    # handoff to accept_candidates survives CWD != project root.
    from tcip_mcp.project_paths import resolve_state

    import json
    state_file = resolve_state(Path(".tcip") / "state" / f"candidates_{img.stem}.json")
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(candidates, default=str), encoding="utf-8")

    return {
        "image_path": out,
        "summary": f"Generated {len(candidates)} candidate masks. "
                   f"Review the numbered overlay, then call accept_candidates "
                   f"with class assignments.",
        "candidate_count": len(candidates),
        "candidates": [
            {
                "id": c["candidate_id"],
                "area": c["area"],
                "stability": round(c["stability_score"], 3),
                "iou": round(c["predicted_iou"], 3),
                "bbox": [round(v, 1) for v in c["bbox"]],
            }
            for c in candidates
        ],
    }


@mcp.tool()
@audited
def accept_candidates(
    image_path: str,
    assignments: list[dict],
) -> dict:
    """Assign classes to SAM candidates and stage them as SAM predictions for review.

    After reviewing sam_auto_label output, the agent calls this tool with a mapping from
    candidate IDs to class IDs. Rejected candidates are simply omitted from the assignments
    list. The masks are written to the predictions tree (``predictions/sam/<date>/<task>``) as
    per-image COCO/JSON with ``created_by="sam"`` and ``score`` = SAM's mask-quality
    (predicted_iou) — they are model output, so a human accepts them on the Review canvas before
    they become ground truth. This never writes GT.

    Args:
        image_path: Absolute path to the image (same as sam_auto_label).
        assignments: List of dicts, each with 'candidate_id' (int) and
            'class_id' (int). Only listed candidates are staged.
    """
    import json
    from tcip_annotation import json_io
    from tcip_annotation.state import PredBBox, PredPolygon

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    # Load cached candidates from the same platform-root state location sam_auto_label wrote to.
    from tcip_mcp.project_paths import resolve_state

    state_file = resolve_state(Path(".tcip") / "state" / f"candidates_{img.stem}.json")
    if not state_file.is_file():
        return {"error": f"No candidates found for {img.stem}. Run sam_auto_label first."}

    candidates = json.loads(state_file.read_text(encoding="utf-8"))
    cand_map = {c["candidate_id"]: c for c in candidates}

    w, h = get_image_dimensions(image_path)

    # Build SAM predictions from accepted candidates (created_by="sam", score = mask-quality).
    from datetime import datetime, timezone
    staged_at = datetime.now(timezone.utc).isoformat()
    boxes: list[PredBBox] = []
    polygons: list[PredPolygon] = []

    for assign in assignments:
        cid = assign["candidate_id"]
        class_id = assign["class_id"]
        cand = cand_map.get(cid)
        if cand is None:
            continue

        score = float(cand.get("predicted_iou", 0.0))  # SAM's mask-quality head, in [0, 1]
        poly_pts = cand["polygon"]
        if len(poly_pts) >= 3:
            polygons.append(PredPolygon(
                [(float(x), float(y)) for x, y in poly_pts], class_id,
                confidence=score, created_by="sam", created_at=staged_at,
            ))

        # Also stage a bounding box (from polygon extent)
        xs = [p[0] for p in poly_pts]
        ys = [p[1] for p in poly_pts]
        boxes.append(PredBBox(
            min(xs), min(ys), max(xs), max(ys), class_id,
            confidence=score, created_by="sam", created_at=staged_at,
        ))

    # Stage into the predictions tree (predictions/sam/<date>/<task>) — model output for a human to
    # accept on the Review canvas, never written straight to ground truth.
    from tcip_mcp.dataset_layout import parse_image_path, prediction_dir

    root, date, _stem = parse_image_path(str(img))
    if boxes:
        det_dir = Path(prediction_dir(root, "sam", date, "detect"))
        det_dir.mkdir(parents=True, exist_ok=True)
        json_io.write_detect(str(det_dir / f"{img.stem}.json"), boxes, w, h)
    if polygons:
        seg_dir = Path(prediction_dir(root, "sam", date, "segment"))
        seg_dir.mkdir(parents=True, exist_ok=True)
        json_io.write_segment(str(seg_dir / f"{img.stem}.json"), polygons, w, h)

    # Render final result for QA
    from tcip_annotation.viz import render_detections
    box_dicts = [
        {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "class_id": b.class_id}
        for b in boxes
    ]
    out = render_detections(image_path, box_dicts)

    return {
        "image_path": out,
        "summary": f"Staged {len(boxes)} detections and {len(polygons)} segmentations "
                   f"from {len(assignments)} SAM candidates as predictions (created_by='sam') "
                   f"for review — not ground truth.",
        "detection_count": len(boxes),
        "segmentation_count": len(polygons),
    }


@mcp.tool()
@audited
def visualize_canvas(
    refresh: bool = True,
    crop_to_viewport: bool = True,
    max_edge: int = 1600,
) -> dict:
    """Render exactly what the human's GUI canvas shows right now — image, shapes, viewport.

    The GUI continuously pushes its canvas state (image, viewport, classes, and the
    display-resolved shapes with the exact colors/tags it renders — including unsaved edits and
    an in-progress drawing) to ``.tcip/state/canvas_live.json``. This tool renders that state
    over the full-resolution image and returns the artifact path for ``view_image``, plus the
    classes schema, review legend, per-tag/per-creator counts, and the state's age.

    Args:
        refresh: Ping the GUI (via the panel-event hub) to push fresh state first, waiting
            briefly for it to land. Falls back to the last pushed state if no GUI responds.
        crop_to_viewport: Render only the region the human currently sees (their zoom/pan).
            Pass False for the full frame with the same overlays.
        max_edge: Downscale the rendered output to at most this edge (px).
    """
    import json
    import time as _time

    from tcip_mcp.project_paths import resolve_state

    meta_file = resolve_state(Path(".tcip") / "state" / "canvas_live.json")
    shapes_file = resolve_state(Path(".tcip") / "state" / "canvas_shapes.json")

    def _read(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    prev = _read(meta_file)
    prev_ts = (prev or {}).get("received_at", 0)
    refreshed = False
    ping_delivered = False
    if refresh:
        from tcip_mcp.web_client import post_panel_event

        res = post_panel_event("app", "canvas_state_request", {})
        ping_delivered = bool(res.get("delivered"))
        if ping_delivered:
            for _ in range(12):  # ~2.4s for the GUI's flush to land
                _time.sleep(0.2)
                cur = _read(meta_file)
                if cur and cur.get("received_at", 0) > prev_ts:
                    refreshed = True
                    break

    state = _read(meta_file)
    if state is None:
        return {"error": "No live canvas state found — is the GUI open with a project loaded? "
                         "The frontend pushes it to <project>/.tcip/state/canvas_live.json "
                         f"(looked at {meta_file}; if the GUI has a different project open, "
                         "set_active_project to it first)."}

    src_image = state.get("image_path") or ""
    if not Path(src_image).is_file():
        return {"error": f"Canvas state references a missing image: {src_image}"}

    # Geometry is valid only when its identity matches the meta document — a heartbeat for a
    # different image/tab means the stored shapes are stale and must not render.
    sdoc = _read(shapes_file) or {}
    shapes_valid = (
        sdoc.get("image_path") == state.get("image_path") and sdoc.get("tab") == state.get("tab")
    )
    shapes = (sdoc.get("shapes") or []) if shapes_valid else []
    out = render_canvas_state(
        src_image, shapes, viewport=state.get("viewport"),
        crop_to_viewport=crop_to_viewport, max_edge=max_edge,
    )

    now = _time.time()
    tag_counts: dict[str, int] = {}
    creator_counts: dict[str, int] = {}
    for s in shapes:
        if isinstance(s, dict):
            tag_counts[str(s.get("tag") or "untagged")] = tag_counts.get(str(s.get("tag") or "untagged"), 0) + 1
            cb = s.get("created_by")
            if cb:
                creator_counts[str(cb)] = creator_counts.get(str(cb), 0) + 1

    age = round(max(0.0, now - float(state.get("received_at") or now)), 1)
    live = refreshed or age < 5.0
    summary = (
        f"Rendered the live {state.get('tab')} canvas for {state.get('image')} ({len(shapes)} shapes)."
        if live else
        f"Rendered the LAST KNOWN {state.get('tab')} canvas for {state.get('image')} "
        f"({len(shapes)} shapes, {age}s old — the GUI did not answer the refresh ping; it may be "
        "closed, on another tab, or on a different project)."
    ) + " Call view_image on image_path to see it."
    return {
        "image_path": out,
        "source_image": src_image,
        "image": state.get("image"),
        "tab": state.get("tab"),
        "mode": state.get("mode"),
        "user": state.get("user"),
        "dirty": state.get("dirty"),
        "project_root": state.get("project_root"),
        "viewport": state.get("viewport"),
        "cropped_to_viewport": bool(crop_to_viewport and state.get("viewport")),
        "classes": state.get("classes") or [],
        "legend": state.get("legend"),
        "counts": state.get("counts"),
        "shape_counts_by_tag": tag_counts,
        "shape_counts_by_creator": creator_counts,
        "state_age_seconds": age,
        "shapes_age_seconds": (round(max(0.0, now - float(sdoc.get("received_at") or 0)), 1)
                               if shapes_valid and sdoc.get("received_at") else None),
        # True when no valid geometry exists for this image/tab yet (heartbeat-only or stale).
        "shapes_missing": not shapes_valid,
        # Did a fresh push land after our ping? False + delivered ping = GUI not listening here.
        "refreshed": refreshed,
        "refresh_ping_delivered": ping_delivered,
        "summary": summary,
    }


@mcp.tool()
@audited
def visualize_grid_overlay(
    image_path: str,
    cols: int = 8,
    rows: int = 6,
) -> dict:
    """Render image with a labeled grid overlay for spatial referencing.

    Creates a grid with letter columns (A-H) and number rows (1-6).
    The agent can reference grid cells like 'B3' to indicate regions
    of interest, which can be converted to SAM point prompts via
    sam_predict with grid cell references.

    Args:
        image_path: Absolute path to the image file.
        cols: Number of grid columns (default 8, labeled A-H).
        rows: Number of grid rows (default 6, labeled 1-6).
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    out = render_grid_overlay(image_path, cols=cols, rows=rows)

    return {
        "image_path": out,
        "summary": f"Grid overlay ({cols}x{rows}) rendered on {img.name}. "
                   f"Reference cells like 'A1' (top-left) to "
                   f"'{chr(ord('A') + cols - 1)}{rows}' (bottom-right).",
        "cols": cols,
        "rows": rows,
    }
