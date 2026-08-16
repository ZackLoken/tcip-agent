"""Annotation tools, load, save, and evaluate name-based annotations via MCP."""

from __future__ import annotations

from pathlib import Path

from tcip_annotation import (
    Annotation,
    BBox,
    Point,
    Polygon,
    compute_matches,
    detect_format,
    load_annotations_any,
    save_annotations_any,
)
from tcip_annotation.json_io import _PROV_KEYS, annotation_from_payload
from tcip_annotation.json_io import read_annotations as read_labels

from tcip_mcp.dataset_layout import (
    annotation_path_for_image,
    find_gt_label,
    find_prediction,
)
from tcip_mcp.pipelines.image_utils import (
    BandGroupIncomplete,
    image_dimensions,
    resolve_image_source,
)
from tcip_mcp.pipelines.resolution import DEFAULT_CONF
from tcip_mcp.server import mcp
from tcip_mcp.audit import audited


def _dims_for(image_path: str) -> tuple[int, int]:
    """``(width, height)`` for ``image_path``, channel-aware, resolves through the same
    enumeration/resolution primitive every other reader now shares (``resolve_image_source``),
    so a ``.bandgroup``-grouped capture measures its real stacked frame instead of ``PIL``
    misreading a manifest file (or a genuinely multi-band raster) as a photograph.
    """
    img = Path(image_path)
    source = resolve_image_source(img.parent, img.stem)
    return image_dimensions(source)


def _logical_image_names(images_dir) -> list[str]:
    """Every logical image's on-disk display name under ``images_dir``, a plain file's own name,
    or (for a ``.bandgroup``-grouped capture) its manifest's filename, the file every other
    by-name reader (``image_name_map``, the dataset gallery route) treats as that capture's name.
    Folding sibling band files into one name here is what lets this tool's frame index agree with
    the frontend's own image_list, which now enumerates the same way.
    """
    from tcip_mcp.pipelines.image_utils import BandGroupRef, list_logical_images

    return [
        src.manifest_path.name if isinstance(src, BandGroupRef) else src.name
        for src in list_logical_images(images_dir).values()
    ]


def _ann_dict(a: Annotation) -> dict:
    """A name-based annotation as a plain JSON dict for a tool response.

    ``rings`` (not ``points``) for a polygon, a stored annotation can genuinely carry more than
    one ring (an occlusion-split instance_seg prediction), so the read side always represents every
    ring rather than silently reporting only the first. ``point`` is the same ``[x, y]`` key the
    on-disk schema uses, so a prompt/keypoint reads back as itself instead of as a geometry-less label.

    Provenance travels out under the schema's own key names, and only where the record holds it, so
    who authored a label and who accepted it are readable here rather than write-only. A reference's
    admissibility turns on exactly those two fields
    (:func:`tcip_annotation.json_io.require_reference_ground_truth`), so a reader that dropped them
    could not tell agent-authored ground truth from a person's.
    """
    d: dict = {"subject": a.subject, "attributes": dict(a.attributes)}
    if isinstance(a.geometry, BBox):
        d["bbox"] = [a.geometry.x1, a.geometry.y1, a.geometry.x2, a.geometry.y2]
    elif isinstance(a.geometry, Polygon):
        d["rings"] = [[[p[0], p[1]] for p in ring] for ring in a.geometry.rings]
    elif isinstance(a.geometry, Point):
        d["point"] = [a.geometry.x, a.geometry.y]
    if a.score is not None:
        d["score"] = a.score
    for k in _PROV_KEYS:
        v = getattr(a, k, None)
        if v is not None:
            d[k] = v
    return d


@mcp.tool()
@audited
def read_annotations(image_path: str, fmt: str | None = None) -> dict:
    """Load the ground-truth labels and predictions for a single image.

    Both are the name-based per-image schema, one file per image, all subjects. Reads the canonical
    per-image JSON (an ``annotations`` key) or an assembled dataset-level COCO (an
    ``images``/``categories`` key), detected from the file's own keys unless ``fmt`` is given. An
    unrecognized store returns an ``error`` rather than a guess.

    Args:
        image_path: Absolute path to the image file.
        fmt: Force annotation format ('json' or 'coco'). Detected from the file's keys if omitted.
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    w, h = _dims_for(image_path)
    result: dict = {"image": image_path, "width": w, "height": h}

    gt_path = find_gt_label(image_path, fmt=fmt)
    if gt_path is not None:
        try:
            file_fmt = fmt or detect_format(str(gt_path))
        except ValueError as exc:
            return {"error": str(exc)}
        anns = load_annotations_any(str(gt_path), fmt=file_fmt, file_name=img.name)
        result["labels"] = {
            "path": str(gt_path), "format": file_fmt, "count": len(anns),
            "subjects": sorted({a.subject for a in anns}),
            "annotations": [_ann_dict(a) for a in anns],
        }

    pred_path = find_prediction(image_path)
    if pred_path is not None:
        preds = read_labels(str(pred_path))
        result["predictions"] = {
            "path": str(pred_path), "count": len(preds),
            "subjects": sorted({a.subject for a in preds}),
            "annotations": [_ann_dict(a) for a in preds],
        }

    return result


@mcp.tool()
@audited(scope_arg="image_path")
def save_annotations(
    image_path: str,
    annotations: list[dict] | None = None,
    fmt: str = "json",
    date: str | None = None,
    path: str | None = None,
    created_by: str | None = None,
) -> dict:
    """Write an image's annotations to its single per-image label file (all subjects, one file).

    The label goes to ``<dataset_root>/annotations/<date>/<stem>.json`` (see
    :mod:`tcip_mcp.dataset_layout`); ``date`` is derived from the image path when not given. Pass
    ``path`` to write to an explicit location instead. Each annotation is a dict carrying a
    ``subject`` (required, refused when absent, since a name-based label is undecodable without it),
    an optional geometry (``bbox`` = [x1,y1,x2,y2], ``points`` = [[x,y],...] for a single-ring polygon
    contour, ``rings`` = [[[x,y],...], ...] for a multi-ring polygon (an occlusion-split mask,
    e.g. ``segment_prompt``'s own output, whose ring vertices are ``{x,y}`` dicts and are accepted
    the same as ``[x,y]`` pairs), ``point`` = [x,y] for a single prompt/keypoint location, or none of
    them for an image-level label), and optional ``attributes`` (attribute name -> value name).

    Args:
        image_path: Absolute path to the image file.
        annotations: List of ``{subject, bbox?/points?/rings?/point?, attributes?}`` dicts (pixel
            coords).
        fmt: Output format, 'json' (canonical per-image, default) or 'coco'.
        date: Capture date; derived from the image path when omitted.
        path: Explicit label path (overrides the canonical location).
        created_by: Producer stamped on each written annotation. Omit to leave provenance unset.
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    anns_in = annotations or []
    if not anns_in:
        return {"error": "provide at least one annotation to save (each carrying a subject)"}
    for i, a in enumerate(anns_in):
        if not isinstance(a, dict) or not a.get("subject"):
            return {"error": f"annotation {i} needs a non-empty subject: {a!r}"}

    from tcip_mcp.workspace import is_valid_name

    if path is None and date is not None and not is_valid_name(date):
        return {"error": f"date must be a single safe path segment (no separators/'..'), got {date!r}"}

    w, h = _dims_for(image_path)

    from datetime import datetime, timezone
    _now = datetime.now(timezone.utc).isoformat()

    typed = [annotation_from_payload(a, author=created_by, now=_now) for a in anns_in]

    out_path = Path(path) if path else annotation_path_for_image(image_path, fmt, date=date)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_annotations_any(str(out_path), typed, w, h, fmt=fmt, file_name=img.name, keep_empty=True)

    try:
        from tcip_mcp.web_client import post_panel_event

        post_panel_event("annotate", "labels_written",
                         {"image_path": image_path, "stem": img.stem, "written": [str(out_path)]})
    except Exception:
        pass

    return {"written": [str(out_path)], "format": fmt, "count": len(typed)}


def _load_image_annotations(image_path: str):
    """Load GT + predictions for one image and build a COCO per-image record.

    Returns ``(iou_type, record, (gt, preds), width, height)`` where ``gt`` / ``preds`` are
    :class:`Annotation` lists; ``None`` if unreadable.
    """
    from tcip_mcp.pipelines.training.evaluation import records_from_annotation

    img = Path(image_path)
    if not img.is_file():
        return None
    w, h = _dims_for(image_path)
    gt: list[Annotation] = []
    preds: list[Annotation] = []

    gt_path = find_gt_label(image_path)
    if gt_path:
        gt = load_annotations_any(str(gt_path), file_name=img.name)
    pred_path = find_prediction(image_path)
    if pred_path:
        preds = read_labels(str(pred_path))

    iou_type, record = records_from_annotation(gt, preds, width=w, height=h)
    return iou_type, record, (gt, preds), w, h


def _add_geom(d: dict, a: Annotation) -> None:
    if isinstance(a.geometry, BBox):
        b = a.geometry
        d["box"] = [b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1]
    elif isinstance(a.geometry, Polygon):
        d["polygon_rings"] = [[[pt[0], pt[1]] for pt in ring] for ring in a.geometry.rings]
    elif isinstance(a.geometry, Point):
        d["point"] = [a.geometry.x, a.geometry.y]


def _detection_breakdown(matches: dict, gt: list[Annotation], preds: list[Annotation]) -> list[dict]:
    """Per-detection TP/FP/FN records (geometry + IoU + confidence + class name) from a match result."""
    detections: list[dict] = []
    for m in matches["tp"]:
        d = {"tag": "tp", "class_name": m["class_name"], "iou": m["iou"], "confidence": m["conf"],
             "gt_idx": m["gt_idx"], "pred_idx": m["pred_idx"]}
        _add_geom(d, preds[m["pred_idx"]])
        detections.append(d)
    for m in matches["fp"]:
        d = {"tag": "fp", "class_name": m["class_name"], "confidence": m["conf"],
             "pred_idx": m["pred_idx"]}
        _add_geom(d, preds[m["pred_idx"]])
        detections.append(d)
    for m in matches["fn"]:
        d = {"tag": "fn", "class_name": m["class_name"], "confidence": 0, "gt_idx": m["gt_idx"]}
        _add_geom(d, gt[m["gt_idx"]])
        detections.append(d)
    return detections


def _apply_governing_criterion(out: dict, records: list, *, trait: str | None,
                               iou_threshold: float, conf_threshold: float) -> dict:
    """Override the human-facing TP/FP/FN + P/R/F1 with the trait's derived criterion.

    A count trait using center-match governs the review count that feeds the phenotype; AP@0.5
    stays as a labeled comparability metric. With no trait, ``out`` is returned unchanged.
    """
    from tcip_mcp.pipelines.training.evaluation import governing_counts, resolve_match_criterion

    criterion = resolve_match_criterion(trait, records, iou_threshold=iou_threshold)
    if criterion["kind"] != "center_match":
        return out
    gc = governing_counts(records, criterion, conf_threshold=conf_threshold)
    out.update({
        "iou_tp": out.get("tp"), "iou_fp": out.get("fp"), "iou_fn": out.get("fn"),
        "tp": gc["tp"], "fp": gc["fp"], "fn": gc["fn"],
        "precision": round(gc["precision"], 4), "recall": round(gc["recall"], 4),
        "f1": round(gc["f1"], 4),
        "governing_criterion": criterion, "map50_role": "comparability_only",
    })
    return out


def _evaluate_image(
    image_path: str,
    iou_threshold: float = 0.5,
    conf_threshold: float = DEFAULT_CONF,
    detail: bool = False,
    trait: str | None = None,
) -> dict:
    """Match predictions against ground truth for a single image (COCOeval).

    mAP / TP / FP / FN come from pycocotools; the ``matches`` block is a per-box overlay for the GUI
    review panel (``compute_matches``). With a count ``trait`` the reported count is governed by the
    trait's derived criterion, map50 kept as comparability.
    """
    loaded = _load_image_annotations(image_path)
    if loaded is None:
        return {"error": f"Image not found: {image_path}"}
    iou_type, record, (gt, preds), w, h = loaded

    from tcip_mcp.pipelines.training.evaluation import coco_detection_metrics
    m = coco_detection_metrics([record], iou_type=iou_type,
                               iou_threshold=iou_threshold, conf_threshold=conf_threshold)
    matches = compute_matches(gt, preds, iou_threshold=iou_threshold, conf_threshold=conf_threshold)
    out = {
        "image": image_path,
        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
        "map50": round(m["map50"], 4),
        "iou_type": iou_type,
        "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold,
        "matches": matches,
    }
    out = _apply_governing_criterion(out, [record], trait=trait,
                                     iou_threshold=iou_threshold, conf_threshold=conf_threshold)
    if detail:
        out["img_w"] = w
        out["img_h"] = h
        out["detections"] = _detection_breakdown(matches, gt, preds)
    return out


def _evaluate_folder(
    folder_path: str,
    iou_threshold: float = 0.5,
    conf_threshold: float = DEFAULT_CONF,
    trait: str | None = None,
) -> dict:
    """Aggregate detection metrics across all images in a dataset."""
    root = Path(folder_path)
    images_dir = root / "images"
    if not images_dir.is_dir():
        images_dir = root

    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    images = sorted(
        f for f in images_dir.rglob("*") if f.is_file() and f.suffix.lower() in image_exts
    )

    from tcip_mcp.pipelines.training.evaluation import coco_detection_metrics, records_from_annotation

    collected = []  # (iou_type, record, (gt, preds), w, h, img)
    for img in images:
        loaded = _load_image_annotations(str(img))
        if loaded is None:
            continue
        iou_type, record, raw, w, h = loaded
        collected.append((iou_type, record, raw, w, h, img))

    any_segm = any(c[0] == "segm" for c in collected)
    dataset_iou_type = "segm" if any_segm else "bbox"
    # One subject->id map across the whole scored set: coco_detection_metrics accumulates every
    # per-image record into a single eval, so a subject must carry the same category id in every
    # image. Rebuild all records with it (the per-image records built by _load_image_annotations
    # used a per-image-local map, which would pool distinct subjects into one class across images).
    global_names: list[str] = []
    for (_it, _rec, (gt, preds), _w, _h, _img) in collected:
        for a in (*gt, *preds):
            # Same membership records_from_annotation applies when it builds the records this map
            # keys: a geometry-less label and a Point produce no scorable box, so neither may mint a
            # COCO category that no annotation ever lands in.
            if a.geometry is None or isinstance(a.geometry, Point):
                continue
            if a.subject not in global_names:
                global_names.append(a.subject)
    name_id = {n: i + 1 for i, n in enumerate(global_names)}
    records = [records_from_annotation(gt, preds, width=w, height=h,
                                       force_segm=any_segm, name_id=name_id)[1]
               for (_it, _rec, (gt, preds), w, h, _img) in collected]
    valid_images = [c[5] for c in collected]

    m = coco_detection_metrics(records, iou_type=dataset_iou_type,
                               iou_threshold=iou_threshold, conf_threshold=conf_threshold)

    counts_by_id = {c["image_id"]: c for c in m["per_image_counts"]}
    per_image = []
    for idx, img in enumerate(valid_images, start=1):
        c = counts_by_id.get(idx, {"tp": 0, "fp": 0, "fn": 0})
        per_image.append({"image": img.name, "tp": c["tp"], "fp": c["fp"], "fn": c["fn"]})

    out = {
        "path": folder_path,
        "image_count": len(images),
        "map": round(m["map"], 4),
        "map50": round(m["map50"], 4),
        "total_tp": m["tp"], "total_fp": m["fp"], "total_fn": m["fn"],
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
        "iou_type": dataset_iou_type,
        "per_image": per_image,
    }
    from tcip_mcp.pipelines.training.evaluation import governing_counts, resolve_match_criterion
    criterion = resolve_match_criterion(trait, records, iou_threshold=iou_threshold)
    if criterion["kind"] == "center_match":
        gc = governing_counts(records, criterion, conf_threshold=conf_threshold)
        out.update({
            "iou_total_tp": out["total_tp"], "iou_total_fp": out["total_fp"],
            "iou_total_fn": out["total_fn"],
            "total_tp": gc["tp"], "total_fp": gc["fp"], "total_fn": gc["fn"],
            "precision": round(gc["precision"], 4), "recall": round(gc["recall"], 4),
            "f1": round(gc["f1"], 4),
            "governing_criterion": criterion, "map50_role": "comparability_only",
        })
        out["per_image"] = [
            {"image": img.name,
             **{k: governing_counts([rec], criterion, conf_threshold=conf_threshold)[k]
                for k in ("tp", "fp", "fn")}}
            for rec, img in zip(records, valid_images)
        ]
    return out


@mcp.tool()
@audited
def score_predictions(
    path: str,
    iou_threshold: float = 0.5,
    conf_threshold: float = DEFAULT_CONF,
    detail: bool = False,
    trait: str | None = None,
) -> dict:
    """Score on-disk predictions against on-disk ground truth (COCOeval).

    Dispatches on the input: a single image file returns per-box ``matches`` (plus an optional
    per-detection ``detections`` breakdown with ``img_w`` / ``img_h`` when ``detail=True``) for the
    GUI review panel; a dataset directory returns aggregate metrics plus ``per_image`` TP/FP/FN.
    Both regimes share ``coco_detection_metrics``.

    Args:
        path: Absolute path to an image file (single-image match) or a dataset root (aggregate).
        iou_threshold: IoU threshold for a positive match (the AP@0.5 comparability convention).
        conf_threshold: Minimum confidence to consider a prediction.
        detail: Single-image only, also return the per-detection ``detections`` breakdown.
        trait: When set, the trait's derived localization criterion governs the reported TP/FP/FN
            count; map50 stays a labeled comparability metric. Absent -> the IoU convention governs.
    """
    p = Path(path)
    if p.is_file():
        return _evaluate_image(path, iou_threshold, conf_threshold, detail, trait)
    if p.is_dir():
        return _evaluate_folder(path, iou_threshold, conf_threshold, trait)
    return {"error": f"Path not found: {path}"}


@mcp.tool()
@audited
def segment_prompt(
    image_path: str,
    points: list[dict] | None = None,
    box: dict | None = None,
    grid_cells: list[str] | None = None,
    tile_size: int | None = None,
    overlap: float = 0.0,
    engine: str = "sam",
    engine_params: dict | None = None,
) -> dict:
    """Turn an interactive prompt (points, a box, or grid cells) into mask polygon rings, via an engine.

    Returns ``rings``, the mask's contours as ``[[{x, y}, ...], ...]``, one ring per connected region.
    An occlusion-split object (a leaf crossed by a stem) segments to more than one region and all of
    them come back; keeping only the largest would report part of an object as the whole of it.

    Provide point prompts, a box prompt, or grid-cell references (e.g. ['B3', 'D5'], converted to
    foreground point prompts). A cell name means nothing without the grid that produced it, so
    ``grid_cells`` requires an explicit ``tile_size``, the geometry the overlay whose cells are
    being named was rendered with (``overlay_reference_grid`` echoes ``tile_size`` and ``overlap``
    back for exactly this). There is no default grid to fall back on: guessing one resolves 'B3' to
    a pixel in a grid nobody looked at. The cells recompute here through the same
    ``reference_grid.reference_cells`` the overlay drew, so the resolved centers are the rendered
    cells' own. The segmentation method is a capability, not a hardcode: 'sam' is the built-in
    SAM2 reference engine; the agent can bring another prompted-segmentation engine behind the same
    seam (a dotted 'module:factory').

    Args:
        image_path: Absolute path to the image file.
        points: List of point prompts, each with x, y, and label (1=fg, 0=bg).
        box: Box prompt with x1, y1, x2, y2 in pixel coordinates.
        grid_cells: List of grid cell references like ['B3', 'D5']. Each is a foreground point.
        tile_size: Cell edge, in native pixels, of the grid the cells were read off. Required
            with ``grid_cells``.
        overlap: Overlap fraction of the grid the cells were read off.
        engine: Segmentation engine, 'sam' (built-in) or a dotted 'module:factory' the agent brings.
        engine_params: Engine-specific knobs forwarded to the engine (e.g. SAM's model_type).
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    if points is None and box is None and grid_cells is None:
        return {"error": "Provide either points, box, or grid_cells prompt"}

    if grid_cells is not None:
        if tile_size is None:
            return {"error": "grid_cells requires tile_size, the cell edge of the grid the "
                             "cells were read off (overlay_reference_grid echoes it back, with "
                             "overlap). Without it a cell name resolves against a grid nobody "
                             "rendered."}
        from tcip_annotation.sam_wrapper import grid_to_pixel

        from tcip_mcp.pipelines.reference_grid import reference_cells
        w, h = _dims_for(image_path)
        try:
            cells = reference_cells(w, h, tile_size, overlap, clamp=True)
        except ValueError as e:
            return {"error": str(e)}
        points = []
        for cell in grid_cells:
            try:
                cx, cy = grid_to_pixel(cell, cells)
                points.append({"x": cx, "y": cy, "label": 1})
            except ValueError as e:
                return {"error": f"Invalid grid cell {cell!r}: {e}"}

    from tcip_mcp.pipelines.proposal import resolve_proposer

    try:
        proposer = resolve_proposer(engine)
    except (ValueError, ImportError) as e:
        return {"error": str(e)}

    try:
        rings = proposer.segment(image_path, points=points, box=box, **(engine_params or {}))
    except ImportError as e:
        return {"error": f"segmentation engine dependencies not available: {e}"}
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"segmentation failed: {e}"}

    if not rings:
        return {"error": "engine produced empty mask", "rings": []}

    return {
        "rings": [[{"x": x, "y": y} for x, y in ring] for ring in rings],
        "ring_count": len(rings),
        "vertex_count": sum(len(ring) for ring in rings),
        "engine": engine,
    }


@mcp.tool()
@audited
def push_panel_data(
    panel: str,
    event_type: str,
    data: dict,
) -> dict:
    """Push structured data to a TCIP GUI panel via the tcip-web backend.

    Sends an HTTP POST to the running FastAPI server (see :mod:`tcip_mcp.web_client`); the backend
    broadcasts to any connected browsers via WebSocket. If the backend is not running the call
    returns ``{"status": "no_subscribers"}`` so the agent can proceed.

    Args:
        panel: Target panel: one per GUI tab, or 'app' for app-level events like annotate_focus /
            review_focus / project_changed. See ``web_client.VALID_PANELS`` for the current set.
        event_type: Event type the panel switches on (e.g. 'load_matches', 'metrics_update', or
            'banner', whose ``data['text']`` the GUI shows as a quiet note above that tab).
        data: Arbitrary JSON data payload.
    """
    from tcip_mcp.web_client import VALID_PANELS, post_panel_event

    if panel not in VALID_PANELS:
        return {"error": f"Unknown panel: {panel}. Valid: {sorted(VALID_PANELS)}"}

    result = post_panel_event(panel, event_type, data)
    result.setdefault("panel", panel)
    result.setdefault("event_type", event_type)
    return result


@mcp.tool()
@audited
def focus(
    tab: str,
    project_root: str,
    dataset_root: str,
    subject: str,
    date: str,
    image_index: int | None = None,
    mode: str | None = None,
    model_name: str | None = None,
    detection_idx: int = 0,
    filter_type: str = "all",
    iou_threshold: float = 0.5,
    conf_threshold: float = DEFAULT_CONF,
) -> dict:
    """Drive the live GUI to a (subject, date) frame, the Annotate tab or the Review tab.

    ``tab='annotate'`` lands the Annotate tab on the first frame annotated for ``subject`` in the
    right mode (emits ``annotate_focus``); ``tab='review'`` lands the Review tab on a model's
    predictions (emits ``review_focus``). Requires the GUI to be running; returns ``delivered:
    false`` if not.

    Args:
        tab: Which GUI surface to drive, 'annotate' or 'review'.
        project_root: Project root (== dataset_root for workspace projects).
        dataset_root: Dataset root holding ``images/`` and ``annotations/`` (plus ``predictions/``).
        subject: Annotation subject (e.g. "leaf", "bush").
        date: Capture-date bucket (e.g. "2026-03-02").
        image_index: Index into the date's sorted image list. Default: first frame labeled for
            ``subject`` (annotate) / with a prediction of ``subject`` for the model (review).
        mode: Annotate only, "box", "polygon" or "point" (default: inferred from the geometry the
            labels on that frame actually carry).
        model_name: Review only (required when ``tab='review'``), the model whose predictions.
        detection_idx: Review only, which detection to center in the Review navigator.
        filter_type: Review only, "all" | "tp" | "fp" | "fn" match filter.
        iou_threshold: Review only, IoU cutoff for the TP/FP/FN match classification.
        conf_threshold: Review only, confidence cutoff for showing predictions.
    """
    if tab == "annotate":
        return _focus_annotate(project_root, dataset_root, subject, date, mode=mode, image_index=image_index)
    if tab == "review":
        if not model_name:
            return {"error": "tab='review' requires model_name"}
        return _focus_review(
            project_root, dataset_root, subject, date, model_name,
            image_index=image_index, detection_idx=detection_idx, filter_type=filter_type,
            iou_threshold=iou_threshold, conf_threshold=conf_threshold,
        )
    return {"error": f"tab must be 'annotate' or 'review', got {tab!r}"}


def _subject_task(anns: list[Annotation], subject: str) -> str | None:
    """"segment" if ``subject`` has a polygon here, "detect" if it has a box, "point" if its only
    geometry here is a point, else None (no geometry-bearing annotation of ``subject``).

    ``"point"`` is a real answer, not a box: callers use a non-``None`` return as "this frame is
    annotated for the subject", so collapsing a point-only frame to ``None`` would hide it from the
    Annotate tab's own frame count, while calling it ``"detect"`` would claim a box nobody drew.
    """
    scoped = [a for a in anns if a.subject == subject and a.geometry is not None]
    if any(isinstance(a.geometry, Polygon) for a in scoped):
        return "segment"
    if any(isinstance(a.geometry, BBox) for a in scoped):
        return "detect"
    if scoped:
        return "point"
    return None


# The GUI drawing mode each resolved task is edited in, the frontend's own Mode union
# ("box" | "polygon" | "point", store/types.ts). A point-only frame lands in point mode: sending it
# in box mode would hand the human a tool that cannot edit what is on the canvas.
_TASK_MODE = {"segment": "polygon", "detect": "box", "point": "point"}


def _focus_annotate(
    project_root: str,
    dataset_root: str,
    subject: str,
    date: str,
    mode: str | None = None,
    image_index: int | None = None,
) -> dict:
    """Drive the live Annotate tab to a (subject, date), in the right mode, on a frame labeled for
    the subject. Posts an ``annotate_focus`` event the GUI honors with local view setters."""
    from tcip_mcp.dataset_layout import annotation_dir, image_dir, label_filename
    from tcip_mcp.web_client import post_panel_event

    idir = Path(image_dir(dataset_root, date))
    if not idir.is_dir():
        return {"error": f"no images for date {date} under {dataset_root}"}
    images = sorted(_logical_image_names(idir))
    if not images:
        return {"error": f"no images on {date}"}

    adir = Path(annotation_dir(dataset_root, date))

    def _task(stem: str) -> str | None:
        f = adir / label_filename(stem)
        return _subject_task(read_labels(str(f)), subject) if f.is_file() else None

    n_annotated = 0
    first_idx: int | None = None
    for i, name in enumerate(images):
        if _task(Path(name).stem) is not None:
            n_annotated += 1
            if first_idx is None:
                first_idx = i

    if image_index is None:
        image_index = first_idx if first_idx is not None else 0
    image_index = max(0, min(image_index, len(images) - 1))

    resolved_task = _task(Path(images[image_index]).stem)
    if mode is None:
        mode = _TASK_MODE.get(resolved_task or "", "box")
    if mode not in ("box", "polygon", "point"):
        return {"error": f"mode must be 'box', 'polygon' or 'point', got {mode!r}"}

    payload = {
        "project_root": project_root, "dataset_root": dataset_root,
        "subject": subject, "date": date, "image_index": image_index, "mode": mode,
    }
    result = post_panel_event("app", "annotate_focus", payload)
    return {
        "delivered": result.get("delivered", False),
        "status": result.get("status"),
        "subject": subject, "date": date, "image_index": image_index, "mode": mode,
        "n_images": len(images), "n_annotated": n_annotated, "image": images[image_index],
    }


def _focus_review(
    project_root: str,
    dataset_root: str,
    subject: str,
    date: str,
    model_name: str,
    image_index: int | None = None,
    detection_idx: int = 0,
    filter_type: str = "all",
    iou_threshold: float = 0.5,
    conf_threshold: float = DEFAULT_CONF,
) -> dict:
    """Drive the live Review tab to a model's predictions of ``subject`` on a frame. Posts a
    ``review_focus`` event the GUI honors with local setters."""
    from tcip_mcp.dataset_layout import image_dir, label_filename, prediction_dir
    from tcip_mcp.web_client import post_panel_event
    from tcip_mcp.workspace import is_valid_name

    if filter_type not in ("all", "tp", "fp", "fn"):
        return {"error": f"filter_type must be all|tp|fp|fn, got {filter_type!r}"}
    for label, val in (("model_name", model_name), ("date", date)):
        if not is_valid_name(val):
            return {"error": f"{label} must be a single safe path segment (no separators/'..'), got {val!r}"}

    idir = Path(image_dir(dataset_root, date))
    if not idir.is_dir():
        return {"error": f"no images for date {date} under {dataset_root}"}
    images = sorted(_logical_image_names(idir))
    if not images:
        return {"error": f"no images on {date}"}

    pred_dir = Path(prediction_dir(dataset_root, model_name, date))

    def _has_pred(stem: str) -> bool:
        f = pred_dir / label_filename(stem)
        return bool(f.is_file() and any(a.subject == subject and a.geometry is not None
                                        for a in read_labels(str(f))))

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
        "project_root": project_root, "dataset_root": dataset_root,
        "subject": subject, "date": date, "model_name": model_name,
        "image_index": image_index, "detection_idx": detection_idx, "filter_type": filter_type,
        "iou_threshold": iou_threshold, "conf_threshold": conf_threshold,
    }
    result = post_panel_event("app", "review_focus", payload)
    return {
        "delivered": result.get("delivered", False),
        "status": result.get("status"),
        "subject": subject, "date": date, "model_name": model_name,
        "image_index": image_index, "detection_idx": detection_idx, "filter_type": filter_type,
        "n_images": len(images), "n_with_predictions": n_with_preds, "image": images[image_index],
    }


@mcp.tool()
@audited(scope_arg="dataset_root")
def stage_proposals(
    dataset_root: str,
    model_name: str,
    date: str,
    stem: str,
    boxes: list[dict] | None = None,
    polygons: list[dict] | None = None,
    overwrite: bool = False,
) -> dict:
    """Stage model-/agent-proposed shapes to ``predictions/<model>/<date>/<stem>.json`` for canvas
    review, the "show on canvas before writing ground truth" guardrail.

    Anything a model produces (a SAM mask, a baseline detection, a shape the agent wants a human to
    vet) goes to the predictions tree, never ``annotations/``, so the human reviews it on the Review
    canvas and accepts/rejects/edits before it becomes GT. Boxes and polygons alike land in the one
    per-image prediction file, each carrying a ``subject`` name. This never writes ground truth. Pair
    with ``focus(tab='review')`` to send the human straight to them.

    A prediction bucket that already carries review verdicts is immutable: by default a stage into it
    is redirected to a fresh run-scoped bucket (``<model>@r2``, next free), and the bucket actually
    written is returned as ``bucket``. Pass ``overwrite=True`` to write in place only when the bucket
    has zero verdicts; with verdicts present it is refused.

    Args:
        dataset_root: Dataset root holding ``predictions/``.
        model_name: Predictions bucket to stage under, the real producer (stamped as created_by).
        date: Capture-date bucket (e.g. "2026-02-11").
        stem: Image stem (filename without extension).
        boxes: ``[{subject, conf, cx, cy, w, h}]`` with cx/cy/w/h normalized to [0, 1].
        polygons: ``[{subject, conf, points: [[x, y], ...]}]`` with points normalized to [0, 1].
        overwrite: Write in place even into an existing bucket. Refused if the bucket has verdicts.
    """
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.prediction_buckets import BucketHasVerdicts, stage_prediction_shapes
    from tcip_mcp.workspace import is_valid_name

    for label, val in (("model_name", model_name), ("date", date), ("stem", stem)):
        if not is_valid_name(val):
            return {"error": f"{label} must be a single safe path segment (no separators/'..'), got {val!r}"}

    boxes = boxes or []
    polygons = polygons or []
    if not boxes and not polygons:
        return {"error": "provide at least one of boxes or polygons to stage"}

    def _unnormalized(vals) -> bool:
        return any(v < -0.01 or v > 1.5 for v in vals)

    norm_boxes: list[tuple[str, float, float, float, float, float]] = []
    for i, b in enumerate(boxes):
        try:
            cx, cy, w, h = float(b["cx"]), float(b["cy"]), float(b["w"]), float(b["h"])
            conf = float(b.get("conf", 1.0))
            subject = str(b["subject"])
        except (KeyError, TypeError, ValueError):
            return {"error": f"box {i} needs a subject and numeric conf, cx, cy, w, h (normalized): {b!r}"}
        if not subject:
            return {"error": f"box {i} needs a non-empty subject"}
        if _unnormalized((cx, cy, w, h)):
            return {"error": f"box {i} coords {(cx, cy, w, h)} look un-normalized; cx/cy/w/h must be in [0,1]"}
        norm_boxes.append((subject, conf, cx, cy, w, h))

    norm_polys: list[tuple[str, float, list[tuple[float, float]]]] = []
    for i, p in enumerate(polygons):
        try:
            conf = float(p.get("conf", 1.0))
            subject = str(p["subject"])
            pts = [(float(x), float(y)) for x, y in p["points"]]
        except (KeyError, TypeError, ValueError):
            return {"error": f"polygon {i} needs a subject, conf, points [[x,y],...] (normalized): {p!r}"}
        if not subject:
            return {"error": f"polygon {i} needs a non-empty subject"}
        if len(pts) < 3:
            return {"error": f"polygon {i} needs at least 3 points, got {len(pts)}"}
        if _unnormalized([v for xy in pts for v in xy]):
            return {"error": f"polygon {i} points look un-normalized; x/y must be in [0,1]"}
        norm_polys.append((subject, conf, pts))

    img_source = None
    for idir in (image_dir(dataset_root, date), image_dir(dataset_root, None)):
        try:
            img_source = resolve_image_source(idir, stem)
            break
        except (FileNotFoundError, BandGroupIncomplete):
            continue
    if img_source is None:
        return {"error": f"no image found for stem {stem!r} under {image_dir(dataset_root, date)}"}
    img_w, img_h = image_dimensions(img_source)

    from datetime import datetime, timezone
    created_at = datetime.now(timezone.utc).isoformat()

    proposals: list[Annotation] = [
        Annotation(subject=subject,
                   geometry=BBox((cx - w / 2) * img_w, (cy - h / 2) * img_h,
                                 (cx + w / 2) * img_w, (cy + h / 2) * img_h),
                   score=conf, created_by=model_name, created_at=created_at)
        for (subject, conf, cx, cy, w, h) in norm_boxes
    ] + [
        Annotation(subject=subject,
                   geometry=Polygon(rings=[[(x * img_w, y * img_h) for x, y in pts]]),
                   score=conf, created_by=model_name, created_at=created_at)
        for (subject, conf, pts) in norm_polys
    ]

    try:
        staged = stage_prediction_shapes(
            dataset_root, model_name, date, stem,
            annotations=proposals, img_w=img_w, img_h=img_h, overwrite=overwrite,
        )
    except BucketHasVerdicts as exc:
        return {"error": str(exc), "verdict_count": exc.count, "suggested_bucket": exc.suggested}
    bucket = staged["bucket"]

    note = ("staged to predictions/ for canvas review, not committed as ground truth; the human "
            "accepts on the Review tab before it becomes GT (focus tab='review' to send them)")
    if staged["redirected"]:
        note = (f"bucket {model_name!r} has {staged['verdict_count']} review verdict(s), staged to a "
                f"fresh bucket {bucket!r} instead so the reviewed predictions stay intact; " + note)

    return {
        "staged": len(boxes) + len(polygons),
        "n_detect": len(boxes), "n_segment": len(polygons),
        "path": staged["path"],
        "model_name": model_name, "bucket": bucket, "bucket_redirected": staged["redirected"],
        "date": date, "stem": stem, "note": note,
    }


@mcp.tool()
@audited(scope_arg="dataset_root")
def write_class_map(dataset_root: str, subjects: dict, output_path: str = "") -> dict:
    """Author the dataset's nested class registry, a thin wrapper over ``class_registry``.

    ``subjects`` is the nested registry mapping the expert defines, subjects to their
    ``description`` / provenance and zero or more ``attributes`` (each ``categorical`` | ``ordinal``
    with ordered ``values``). It is validated through :func:`class_registry.registry_from_dict` (a
    malformed shape refuses loudly) and written to ``<dataset_root>/classes.json`` via
    :func:`class_registry.write_registry`. No numeric class ids, no colors, no id enumeration: a
    label-scan cannot infer an attribute's type or rank, the expert's fact is the input here.

    Changing a subject's attribute vocabulary invalidates the confirmations made under the old one,
    so before the new registry lands, :func:`class_registry.stamp_unstamped_confirmations` records
    the outgoing digest onto that subject's still-unstamped confirmations; they then read as
    predating the change rather than as made under the new vocabulary. What it stamped, and any
    warning if it could not, comes back under ``schema_change_sweep``.

    Args:
        dataset_root: Dataset root; the registry is written to ``<dataset_root>/classes.json``.
        subjects: Nested ``{subject: {description?, defined_by?, defined_at?, attributes?}}`` dict.
        output_path: Optional explicit path (overrides ``<dataset_root>/classes.json``).
    """
    from tcip_mcp import class_registry
    from tcip_mcp.dataset_layout import classes_path

    if not isinstance(subjects, dict) or not subjects:
        return {"error": "subjects must be a non-empty nested registry mapping"}
    try:
        registry = class_registry.registry_from_dict(subjects)
    except class_registry.RegistryError as exc:
        return {"error": f"invalid registry: {exc}"}

    out = Path(output_path) if output_path else classes_path(dataset_root)
    sweep = class_registry.stamp_unstamped_confirmations(out, registry)
    class_registry.write_registry(out, registry)
    return {"classes_path": str(out), "subjects": [s.name for s in registry.subjects],
            "schema_change_sweep": sweep}
