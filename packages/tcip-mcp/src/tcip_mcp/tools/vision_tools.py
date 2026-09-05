"""Vision tools: render annotations and predictions for visual analysis.

Each tool saves a rendered image to .tcip/artifacts/viz/ and returns the path so the agent
can call its client's own image-capable read tool on it to visually inspect it. The
proposal-workflow tools (propose_annotations, stage_proposals, segment_prompt) live in
proposal_tools.py; this module keeps the renderers.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Callable, NamedTuple

import tcip_store as ts

from tcip_annotation import Annotation, Point, Polygon, bbox_of, load_annotations_any
from tcip_annotation.json_io import UnreadableLabelDocument
from tcip_annotation.json_io import read_annotations as read_labels
from tcip_annotation.sam_wrapper import column_label
from tcip_annotation.viz import (
    render_canvas_state,
    render_comparison,
    render_detections,
    render_grid,
    render_grid_overlay,
    render_segmentations,
)

from tcip_mcp.audit import audited
from tcip_mcp.pipelines.display_bounds import VIZ_ARTIFACT_MAX_EDGE
from tcip_mcp.pipelines.resolution import DEFAULT_CONF
from tcip_mcp.server import mcp

if TYPE_CHECKING:
    import numpy as np

    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.raster_source import Rect


class DisplayRead(NamedTuple):
    """Display pixels for a renderer, with the frame facts that place annotations on them.

    ``pixels`` is uint8 RGB; ``rect`` is the region of the raster they were read from and ``scale``
    the served resolution as a fraction of native, the pair a renderer drawing a crop needs;
    ``native_size`` is the raster's own ``(width, height)``, the frame annotation coordinates are
    measured in.
    """

    pixels: np.ndarray
    rect: Rect
    scale: float
    native_size: tuple[int, int]


def _bounded_target(rect: Rect, max_edge: int) -> tuple[int, int] | None:
    """The aspect-preserving output size that holds ``rect``'s longest edge to ``max_edge``;
    ``None`` when the region already fits and reads at native resolution."""
    edge = max(rect.width, rect.height)
    if edge <= max_edge:
        return None
    k = max_edge / edge
    return max(1, round(rect.width * k)), max(1, round(rect.height * k))


def _clamped_rect(region: tuple[float, float, float, float], width: int, height: int) -> Rect:
    """An ``(x, y, w, h)`` region as a non-empty rect inside a ``width`` x ``height`` raster.

    A viewport can hang off any edge of the image (the human pans past it), so it is clamped here
    rather than refused: what the raster layer will not serve is an out-of-bounds or empty read.
    """
    from tcip_mcp.pipelines.raster_source import Rect

    def clamp(v: float, low: int, high: int) -> int:
        return max(low, min(int(v), high))

    x0 = clamp(region[0], 0, width - 1)
    y0 = clamp(region[1], 0, height - 1)
    return Rect(x0, y0,
                clamp(x0 + region[2], x0 + 1, width), clamp(y0 + region[3], y0 + 1, height))


def _read_for_display(source: "str | Path | BandGroupRef", *,
                      max_edge: int = VIZ_ARTIFACT_MAX_EDGE,
                      region: tuple[float, float, float, float] | None = None) -> DisplayRead:
    """Read ``source`` as display pixels a renderer can draw on.

    The one decode every visualization tool goes through: the raster layer serves the region (an
    ``(x, y, w, h)`` rectangle in the raster's own grid, or the whole frame) at or under
    ``max_edge``, so an overview-bearing raster costs a reduced read rather than a whole decode
    and nothing is ever materialized to a temp file.

    An 8-bit raster at 1/3/4 bands already holds display values, so it keeps its own pixels
    (grayscale repeated, alpha dropped) with no stretch, the plain-RGB reading a viewer is served:
    an ordinary photograph or RGB GeoTIFF must never reach the agent as a synthetic
    reinterpretation of its colors. Every other raster has no 8-bit reading of its own, so its
    first three bands are composited and independently min-max stretched. Both readings go through
    the shared ``composite_display_rgb``.
    """
    from tcip_mcp.pipelines.band_stats import composite_display_rgb
    from tcip_mcp.pipelines.derivations import probe_channels
    from tcip_mcp.pipelines.raster_source import Rect, open_raster

    with open_raster(source, probe_channels(source)) as raster:
        native = (int(raster.width), int(raster.height))
        rect = (Rect(0, 0, raster.width, raster.height) if region is None
                else _clamped_rect(region, raster.width, raster.height))
        pixels, spec = raster.read_region(rect, target_size=_bounded_target(rect, max_edge))
    bands = int(pixels.shape[-1])
    idxs = [0, 1, 2] if bands >= 3 else [0, 0, 0]
    plain = bands in (1, 3, 4) and pixels.dtype == "uint8"
    return DisplayRead(composite_display_rgb(pixels, idxs, "none" if plain else "minmax"),
                       rect, spec.scale, native)


def _display_for_stem(images_dir: str | Path, stem: str) -> DisplayRead | None:
    """Display pixels for ``stem`` in ``images_dir``; ``None`` if ``stem`` isn't a resolvable
    logical image (missing, or a stale band-group manifest)."""
    from tcip_mcp.pipelines import image_utils

    try:
        source = image_utils.resolve_image_source(images_dir, stem)
    except (FileNotFoundError, image_utils.BandGroupIncomplete):
        return None
    return _read_for_display(source)


def _source_for_path(image_path: str) -> "str | Path | BandGroupRef":
    """The logical image source behind ``image_path``, for a caller that has a path rather than a
    ``(dir, stem)`` pair.

    The enumeration primitive's own resolution of it, so a ``.bandgroup``-grouped capture reads as
    the group it names. A path the primitive doesn't resolve (one outside any recognized
    ``images/`` layout) is returned as itself, so the caller's own not-a-file handling surfaces the
    real error instead of this swallowing it.
    """
    from tcip_mcp.pipelines import image_utils

    img = Path(image_path)
    try:
        return image_utils.resolve_image_source(img.parent, img.stem)
    except (FileNotFoundError, image_utils.BandGroupIncomplete):
        return img


def _display_for_path(image_path: str, *, max_edge: int = VIZ_ARTIFACT_MAX_EDGE,
                      region: tuple[float, float, float, float] | None = None) -> DisplayRead:
    """As ``_display_for_stem``, for a caller that already has a path rather than a
    ``(dir, stem)`` pair.
    """
    return _read_for_display(_source_for_path(image_path), max_edge=max_edge, region=region)


def _subject_indexer() -> tuple[dict[str, int], Callable[[str], int]]:
    """A stable subject-name → color-index map for the (int-keyed) renderers, plus its indexer.

    Labels are name-based now; the viz layer colors by an integer and labels from a ``{index: name}``
    map, so each distinct subject in one render gets a stable index and its own name in the legend.
    """
    idx: dict[str, int] = {}

    def index(name: str) -> int:
        if name not in idx:
            idx[name] = len(idx)
        return idx[name]

    return idx, index


def _name_map(idx: dict[str, int]) -> dict[int, str]:
    return {i: name for name, i in idx.items()}


def _boxable(anns: list[Annotation]) -> list[Annotation]:
    """The annotations a box renderer can draw: geometry-bearing, with a Point excluded.

    A Point has no box (``bbox_of`` refuses one) and there is no point renderer in the viz layer yet,
    so it is skipped by the *draw* call the way a geometry-less label already is. Callers report how
    many they skipped (``_n_points``) rather than quietly shrinking the annotation count they show.
    """
    return [a for a in anns if a.geometry is not None and not isinstance(a.geometry, Point)]


def _n_points(anns: list[Annotation]) -> int:
    return sum(1 for a in anns if isinstance(a.geometry, Point))


def _point_note(n: int) -> str:
    return f" ({n} point annotation(s) not drawn, no point renderer yet)" if n else ""


def _legend_name(a: Annotation, *, scope) -> str:
    """The name a render's legend shows for ``a``: the classified value under a classified scope
    (the object class alone would be one name for every prediction), else ``a.subject`` as today.
    """
    if scope is not None and scope.classified:
        from tcip_annotation.json_io import classified_value_of

        value = classified_value_of(a, subject=scope.subject, attribute=scope.attribute)
        if value is not None:
            return value
    return a.subject


def _box_dict(a: Annotation, index: Callable[[str], int], *, scope=None) -> dict:
    geometry = a.geometry
    assert not isinstance(geometry, Point) and geometry is not None, \
        "every caller passes a _boxable-filtered or box-rendered annotation"
    b = bbox_of(geometry)
    d = {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
        "class_id": index(_legend_name(a, scope=scope))}
    if a.score is not None:
        d["confidence"] = a.score
    return d


def _poly_dict(a: Annotation, index: Callable[[str], int], *, scope=None) -> dict:
    geometry = a.geometry
    assert isinstance(geometry, Polygon), "called only for the segmentation task's own shapes"
    return {"rings": [[[p[0], p[1]] for p in ring] for ring in geometry.rings],
            "class_id": index(_legend_name(a, scope=scope))}


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

    Not an MCP tool: run through ``scripts/visualize.py``, per the admission standard
    (packages/tcip-mcp/CLAUDE.md), while staying importable for its own tests.

    One entry point for the common renders (replaces the former visualize_annotations /
    visualize_predictions / visualize_comparison / visualize_dataset_sample). Saves to
    .tcip/artifacts/viz/ and returns ``image_path`` for the agent's own image-capable read tool.

    Rendering conventions, shared across every source: boxes/masks color by class through the
    20-class palette in ``tcip_annotation.viz``, indexed by first-seen order within one render
    call; this is not the GUI annotation canvas's own coloring (a per-subject-name hash into its
    own smaller palette) and is not stable across renders. A 'comparison' render outlines GT
    green and predictions red, with yellow center-to-center lines joining matched pairs. A
    detection label carries the class name and, when the box carries a confidence score, the
    score; a segmentation label carries the class name only. Each source image is read at up to
    ``display_bounds.VIZ_ARTIFACT_MAX_EDGE`` (1024px) on its longest edge before rendering; for
    source='dataset' the per-sample renders built this way are then tiled into a grid that saves
    at ``cols`` x 256 by ``rows`` x 256 pixels, growing with ``n``.

    Args:
        source: What to render:
            'annotations' = ground-truth labels on a single image (path = image file);
            'predictions' = model predictions on a single image (path = image file);
            'comparison'  = GT (green) vs predictions (red) with TP/FP/FN match stats
            (path = image file);
            'dataset'     = grid of n random annotated samples (path = dataset folder
            containing images/ and labels/).
        path: Image file (annotations/predictions/comparison) or dataset folder (dataset).
        task: 'detect' or 'segment'.
        class_names: Comma-separated class names (e.g. "leaf,fruit,bud").
        conf_threshold: Minimum confidence; filters displayed predictions (source='predictions')
            and the predictions matched against GT (source='comparison'). Defaults to the shared
            ``DEFAULT_CONF`` so the comparison operating point matches inference/evaluate.
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
    from tcip_annotation.format_io import detect_format
    from tcip_mcp.dataset_layout import find_gt_label

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    stem = img.stem
    label_path = find_gt_label(image_path)
    if label_path is None:
        return {"error": f"No labels found for {stem}"}

    try:
        fmt = detect_format(str(label_path))
        anns = load_annotations_any(str(label_path), fmt=fmt, file_name=img.name)
    except (ValueError, UnreadableLabelDocument) as exc:
        return {"error": str(exc)}
    idx, index = _subject_indexer()

    n_points = _n_points(anns)
    read = _display_for_path(image_path)
    if task == "detect":
        shapes = _boxable(anns)
        out = render_detections(read.pixels, [_box_dict(a, index) for a in shapes],
                                native_size=read.native_size, class_names=_name_map(idx))
        summary = f"Rendered {len(shapes)} detections on {img.name}"
        if shapes:
            from collections import Counter
            counts = Counter(a.subject for a in shapes)
            summary += ": " + ", ".join(f"{v} {k}" for k, v in counts.most_common())
        summary += _point_note(n_points)
    else:
        shapes = [a for a in anns if isinstance(a.geometry, Polygon)]
        out = render_segmentations(read.pixels, [_poly_dict(a, index) for a in shapes],
                                   native_size=read.native_size, class_names=_name_map(idx))
        summary = f"Rendered {len(shapes)} segmentation masks on {img.name}" + _point_note(n_points)

    return {
        "image_path": out,
        "summary": summary,
        "format": fmt,
        # `count` is the stable key across all visualize sources; the source-specific alias stays.
        "count": len(shapes),
        "annotation_count": len(shapes),
        # Disclosed, not folded into `count`: these annotations are real but this renderer can't draw
        # them, and a silently smaller count would read as "the image has fewer annotations".
        "points_not_rendered": n_points,
    }


def _viz_predictions(
    image_path: str,
    task: str = "detect",
    class_names: str = "",
    conf_threshold: float = 0.0,
) -> dict:
    """Render model predictions on a single image. See ``visualize``.

    Under a classified bucket (its own recorded ``bucket_scope``), the legend keys each
    detection by its decoded value, not the object class every one of them shares.
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    stem = img.stem
    from tcip_mcp.dataset_layout import find_prediction
    from tcip_mcp.pipelines.resolution import StampScopeUnstated, bucket_scope

    pred_file = find_prediction(image_path)
    if pred_file is None:
        return {"error": f"No predictions found for {stem}"}

    try:
        preds = read_labels(str(pred_file))
        scope = bucket_scope(Path(pred_file).parent)
    except (UnreadableLabelDocument, StampScopeUnstated, ts.StoreError) as exc:
        return {"error": str(exc)}
    if conf_threshold > 0:
        preds = [a for a in preds if (a.score is None or a.score >= conf_threshold)]
    idx, index = _subject_indexer()

    n_points = _n_points(preds)
    read = _display_for_path(image_path)
    if task == "detect":
        shapes = _boxable(preds)
        out = render_detections(
            read.pixels, [_box_dict(a, index, scope=scope) for a in shapes],
            native_size=read.native_size, class_names=_name_map(idx))
        summary = f"Rendered {len(shapes)} predictions on {img.name}" + _point_note(n_points)
    else:
        shapes = [a for a in preds if isinstance(a.geometry, Polygon)]
        out = render_segmentations(
            read.pixels, [_poly_dict(a, index, scope=scope) for a in shapes],
            native_size=read.native_size, class_names=_name_map(idx))
        summary = f"Rendered {len(shapes)} prediction masks on {img.name}" + _point_note(n_points)

    return {
        "image_path": out,
        "summary": summary,
        # `count` is the stable key across all visualize sources; the source-specific alias stays.
        "count": len(shapes),
        "prediction_count": len(shapes),
        "points_not_rendered": n_points,
    }


def _viz_comparison(
    image_path: str,
    task: str = "detect",
    iou_threshold: float = 0.5,
    class_names: str = "",
    conf_threshold: float = DEFAULT_CONF,
) -> dict:
    """Render GT vs prediction comparison with match indicators. See ``visualize``.

    Green = ground truth, Red = predictions, Yellow lines = matched pairs. The match lines come
    from ``compute_matches`` over the two documents as written: on a conformed classified bucket
    (predictions carrying the object class in ``subject``, the same shape ground truth carries)
    they match by object class, while the legend still keys the prediction side by its decoded
    value (:func:`_legend_name`), so a correctly localized, wrongly classified pair now shows as
    a real match with two different legend colors rather than as an unrelated FP/FN.
    """
    from tcip_annotation.format_io import detect_format
    from tcip_annotation.matching import compute_matches
    from tcip_mcp.dataset_layout import find_gt_label, find_prediction
    from tcip_mcp.pipelines.resolution import StampScopeUnstated, bucket_scope

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    stem = img.stem
    idx, index = _subject_indexer()

    label_path = find_gt_label(image_path)
    if label_path is None:
        return {"error": f"No labels found for {stem}"}
    try:
        fmt = detect_format(str(label_path))
        gt = _boxable(load_annotations_any(str(label_path), fmt=fmt, file_name=img.name))
    except (ValueError, UnreadableLabelDocument) as exc:
        return {"error": str(exc)}
    gt_dicts = [_box_dict(a, index) for a in gt]

    pred_file = find_prediction(image_path)
    pred_dicts: list[dict] = []
    tp_matches: list[dict] = []
    if pred_file is not None:
        try:
            preds = _boxable(read_labels(str(pred_file)))
            scope = bucket_scope(Path(pred_file).parent)
        except (UnreadableLabelDocument, StampScopeUnstated, ts.StoreError) as exc:
            return {"error": str(exc)}
        pred_dicts = [_box_dict(a, index, scope=scope) for a in preds]
        # Match at the caller's conf operating point (not compute_matches' silent 0.25 default).
        match_result = compute_matches(gt, preds, iou_threshold=iou_threshold,
                                       conf_threshold=conf_threshold)
        tp_matches = match_result["tp"]
        tp = len(tp_matches)
        fp = len(match_result["fp"])
        fn = len(match_result["fn"])
    else:
        tp, fp, fn = 0, 0, len(gt)

    read = _display_for_path(image_path)
    out = render_comparison(read.pixels, gt_dicts, pred_dicts, native_size=read.native_size,
                            matches=tp_matches, class_names=_name_map(idx))

    return {
        "image_path": out,
        "summary": f"GT={len(gt_dicts)}, Pred={len(pred_dicts)}, TP={tp}, FP={fp}, FN={fn}",
        "gt_count": len(gt_dicts),
        "pred_count": len(pred_dicts),
        "tp": tp, "fp": fp, "fn": fn,
    }


def get_worst_predictions(
    predictions_dir: str,
    labels_dir: str,
    top_k: int = 8,
) -> dict:
    """Return the ``top_k`` images ranked worst by a count-mismatch + low-confidence triage heuristic.

    This is a cheap triage signal, not a quality metric: it does no IoU matching and computes
    no loss. The score is ``2·|n_gt−n_pred as a shortfall| + |surplus| + (1−avg_conf)``, purely
    the difference in box *counts* plus mean confidence, so an image with the right count but
    every box mislocated scores as good. Use it to surface likely-bad frames for a human to look
    at; for true TP/FP/FN ranking use ``score_predictions`` (``detail=True``, IoU-matched).

    Args:
        predictions_dir: Directory with per-image JSON prediction files
            (``<stem>.json``) written by run_inference / the review engine.
        labels_dir: Directory with per-image JSON ground-truth label files.
        top_k: Number of worst images to return.
    """
    pred_path = Path(predictions_dir)
    gt_path = Path(labels_dir)

    if not pred_path.is_dir():
        return {"error": f"Predictions directory not found: {predictions_dir}"}
    if not gt_path.is_dir():
        return {"error": f"Labels directory not found: {labels_dir}"}

    from tcip_annotation.json_io import prediction_documents, read_annotations
    from tcip_annotation.state import Point

    def _boxes(path) -> list:
        """The annotations this count heuristic counts, a geometry-less label and a ``Point`` are
        not detections, so neither belongs in a box count on either side of the comparison."""
        return [a for a in read_annotations(str(path))
                if a.geometry is not None and not isinstance(a.geometry, Point)]

    scores: list[tuple[str, float]] = []
    for pred_file in prediction_documents(pred_path):
        gt_file = gt_path / pred_file.name
        preds = _boxes(pred_file)
        gt_anns = _boxes(gt_file) if gt_file.is_file() else []

        n_pred = len(preds)
        n_gt = len(gt_anns)

        # Simple error heuristic: |pred - gt| + missed + extra + low confidence
        missed = max(0, n_gt - n_pred)
        extra = max(0, n_pred - n_gt)
        avg_conf = 0.0
        if n_pred > 0:
            confs = [p.score for p in preds if p.score is not None]
            avg_conf = sum(confs) / len(confs) if confs else 0.5

        # Higher score = worse prediction
        error_score = missed * 2.0 + extra * 1.0 + (1.0 - avg_conf)
        scores.append((pred_file.stem, error_score))

    # Also include GT images with no predictions at all (completely missed)
    for gt_file in prediction_documents(gt_path):
        pred_file = pred_path / gt_file.name
        if not pred_file.is_file():
            gt_anns = _boxes(gt_file)
            if gt_anns:
                scores.append((gt_file.stem, len(gt_anns) * 3.0))

    scores.sort(key=lambda x: x[1], reverse=True)
    worst = scores[:top_k]

    return {
        "worst_images": [{"stem": s, "error_score": round(sc, 3)} for s, sc in worst],
        "total_evaluated": len(scores),
    }


@audited
def render_failure_cases(
    predictions_dir: str,
    labels_dir: str,
    images_dir: str = "",
    task: str = "detect",
    top_k: int = 10,
    class_names: str = "",
) -> dict:
    """Find and render the worst predictions for failure analysis.

    Not an MCP tool: run through ``scripts/render_failure_cases.py``, per the admission standard
    (packages/tcip-mcp/CLAUDE.md), while staying importable for its own tests.

    Ranks by a count-mismatch + low-confidence heuristic (`get_worst_predictions`); no IoU
    matching, so an image with the right box count but every box mislocated scores as good. Not a
    substitute for `score_predictions`(`detail=True`)'s IoU-matched TP/FP/FN when mislocalization
    itself is the question.

    Returns a grid image and individual failure case images.

    Args:
        predictions_dir: Directory with prediction files.
        labels_dir: Directory with ground-truth label files.
        images_dir: Directory with source images. Auto-detected if empty.
        task: 'detect' or 'segment'.
        top_k: Number of worst cases to render.
        class_names: Comma-separated class names.
    """
    # Auto-detect images_dir
    if not images_dir:
        from tcip_mcp.dataset_layout import image_root

        labels_path = Path(labels_dir)
        candidate = image_root(labels_path.parent.parent)
        if candidate.is_dir():
            images_dir = str(candidate)
        else:
            return {"error": "images_dir not specified and could not be auto-detected"}

    try:
        worst = get_worst_predictions(predictions_dir, labels_dir, top_k=top_k)
    except UnreadableLabelDocument as exc:
        return {"error": str(exc)}
    if "error" in worst:
        return worst

    worst_items = worst.get("worst_images", [])
    if not worst_items:
        return {"summary": "No prediction errors found", "image_path": None}

    from tcip_mcp.project_paths import resolve_state

    out_dir = resolve_state(Path(".tcip") / "artifacts" / "viz" / "failures").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # One GT-vs-prediction render per case, titled in the same pass so a case that can't be
    # resolved drops its title with it.
    case_paths: list[str] = []
    titles: list[str] = []
    img_dir = Path(images_dir)
    for item in worst_items:
        stem = item["stem"]
        read = _display_for_stem(img_dir, stem)
        if read is None:
            continue

        idx, index = _subject_indexer()

        gt_file = Path(labels_dir) / f"{stem}.json"
        pred_file = Path(predictions_dir) / f"{stem}.json"
        try:
            gt_dicts = ([_box_dict(a, index) for a in _boxable(read_labels(str(gt_file)))]
                        if gt_file.is_file() else [])
            pred_dicts = ([_box_dict(a, index) for a in _boxable(read_labels(str(pred_file)))]
                          if pred_file.is_file() else [])
        except UnreadableLabelDocument as exc:
            return {"error": str(exc)}

        out = str(out_dir / f"failure_{len(case_paths):03d}_{stem}.png")
        render_comparison(read.pixels, gt_dicts, pred_dicts, native_size=read.native_size,
                          class_names=_name_map(idx), output_path=out)
        case_paths.append(out)
        titles.append(f"{stem} (err={item['error_score']:.1f})")

    # Render grid of all failure cases
    grid_path = None
    if case_paths:
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
    from tcip_annotation.format_io import detect_format
    from tcip_mcp.dataset_layout import find_gt_label, image_root
    from tcip_mcp.pipelines.image_utils import (
        BandGroupIncomplete, BandGroupRef, list_logical_images, resolve_image_source,
    )

    root = Path(folder_path)
    images_dir = image_root(root)
    if not images_dir.is_dir():
        return {"error": f"Images directory not found: {images_dir}"}

    # Every logical image at or under images_dir, folding sibling band files into one grouped
    # entry per capture: recurses into images/<date>/ subfolders (the canonical layout) and any
    # deeper nesting.
    dirs = {images_dir} | {p for p in images_dir.rglob("*") if p.is_dir()}
    all_images: list[tuple[Path, str]] = [
        (d, stem) for d in sorted(dirs) for stem in sorted(list_logical_images(d))
    ]
    if not all_images:
        return {"error": "No images found in dataset"}

    sample = random.sample(all_images, min(n, len(all_images)))
    rendered_paths = []
    titles = []
    for d, stem in sample:
        try:
            source = resolve_image_source(d, stem)
        except (FileNotFoundError, BandGroupIncomplete):
            continue
        rep_path = source.manifest_path if isinstance(source, BandGroupRef) else source
        label_path = find_gt_label(str(rep_path))
        if label_path is not None:
            try:
                fmt = detect_format(str(label_path))
            except UnreadableLabelDocument as exc:
                return {"error": str(exc)}
            except ValueError:
                label_path = None  # unrecognized store: render the image without labels
        read = _read_for_display(source)
        if label_path is not None:
            idx, index = _subject_indexer()
            try:
                anns = load_annotations_any(str(label_path), fmt=fmt, file_name=rep_path.name)
            except UnreadableLabelDocument as exc:
                return {"error": str(exc)}
            if task == "detect":
                shapes = _boxable(anns)
                out = render_detections(read.pixels, [_box_dict(a, index) for a in shapes],
                                        native_size=read.native_size, class_names=_name_map(idx))
            else:
                shapes = [a for a in anns if isinstance(a.geometry, Polygon)]
                out = render_segmentations(read.pixels, [_poly_dict(a, index) for a in shapes],
                                           native_size=read.native_size,
                                           class_names=_name_map(idx))
            titles.append(f"{stem} ({len(shapes)})")
        else:
            # An unlabeled sample renders too, with nothing drawn on it: the grid tiles rendered
            # artifacts, so every cell has to be one.
            out = render_detections(read.pixels, [], native_size=read.native_size)
            titles.append(f"{stem} (no labels)")

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
def capture_live_canvas(
    refresh: bool = True,
    crop_to_viewport: bool = True,
    max_edge: int = 1600,
    render_last_known: bool = False,
) -> dict:
    """Render exactly what the human's GUI canvas shows right now: image, shapes, viewport.

    The GUI continuously pushes its canvas state under ``.tcip/state/``, split into two documents
    so the cadences never contend: ``canvas_live.json``, a meta document written on every push
    (image, viewport, classes, legend, counts, tab, mode, active_subject, cut_armed, dirty and
    user), and
    ``canvas_shapes.json``, the full display-resolved geometry (the shapes with the exact
    colors/tags the canvas renders, including unsaved edits and an in-progress drawing), written
    only when the shapes themselves change. A heartbeat (meta only, no geometry write) fires on
    view and meta changes, and as the downgrade while a pointer interaction is live, with the
    full geometry pushed once on release rather than per tick. This tool reads the region being
    shown at up to ``max_edge``, renders that state over it, and returns the artifact path for
    the agent's own image-capable read tool, plus the classes schema, review legend,
    per-tag/per-creator counts, and the state's age.

    The GUI's currently open project is named by the ``canvas_open_binding`` record, checked
    against this process's own pinned project before rendering anything: a project the GUI has
    moved away from is not the live canvas, so a mismatch refuses by default rather than render
    another project's documents as if they were this one's. Pass ``render_last_known=True`` to
    render this process's own pinned project's last pushed canvas anyway, labelled not-live.

    Args:
        refresh: Ping the GUI (via the panel-event hub) to push fresh state first, waiting
            briefly for it to land. Falls back to the last pushed state if no GUI responds.
            No-op when the binding names another project: pinging would only make that other
            project's GUI push under its own root, not this one.
        crop_to_viewport: Render only the region the human currently sees (their zoom/pan).
            Pass False for the full frame with the same overlays.
        max_edge: Downscale the rendered output to at most this edge (px).
        render_last_known: When the GUI's open project differs from this process's own, render
            this process's own pinned project's last pushed canvas anyway (labelled not-live)
            instead of refusing. Ignored when the two agree.
    """
    import time as _time

    from tcip_mcp import workspace
    from tcip_mcp.project_paths import platform_state_root
    from tcip_mcp.web_client import (
        GuiBindingUnreadable, binding_divergence, canvas_geometry_key, canvas_meta_key,
        gui_binding_matches, read_canvas_binding,
    )

    root = str(platform_state_root())
    meta_doc = canvas_meta_key(root)
    shapes_doc = canvas_geometry_key(root)

    def _read(key: ts.Key) -> dict | None:
        try:
            return ts.read(key, default=None)
        except (OSError, ts.DecodeError):
            return None

    binding: dict | None = None
    same_root = False
    state: dict | None = None
    sdoc: dict = {}
    shapes: list = []
    shapes_valid = False
    region = None
    out = ""
    src_image = ""
    refreshed = False
    ping_delivered = False

    for attempt in range(2):
        try:
            same_root, binding = gui_binding_matches(root)
        except GuiBindingUnreadable as exc:
            return {"error": str(exc)}

        if binding is None:
            ws_root = str(workspace.workspace_root(create=False))
            return {"error": f"No current canvas binding exists under the workspace root "
                              f"{ws_root}; opening a project in the GUI creates one."}

        if not same_root and not render_last_known:
            return {
                "error": "The GUI's open project differs from this tool's own pinned project; "
                         "its canvas is not this project's live view.",
                "divergence": binding_divergence(binding, root),
            }

        prev = _read(meta_doc)
        prev_ts = (prev or {}).get("received_at", 0)
        refreshed = False
        ping_delivered = False
        if refresh and same_root:
            from tcip_mcp.web_client import PANEL_EVENT_CANVAS_STATE_REQUEST, post_panel_event

            res = post_panel_event("app", PANEL_EVENT_CANVAS_STATE_REQUEST, {})
            ping_delivered = bool(res.get("delivered"))
            if ping_delivered:
                for _ in range(12):  # ~2.4s for the GUI's flush to land
                    _time.sleep(0.2)
                    cur = _read(meta_doc)
                    if cur and cur.get("received_at", 0) > prev_ts:
                        refreshed = True
                        break

        state = _read(meta_doc)
        if state is None:
            if same_root:
                return {"error": "No live canvas state found; is the GUI open with a project "
                                  "loaded? The frontend pushes its canvas state to the project "
                                  f"the GUI has open, and it already agrees this is {root}; "
                                  "nothing has been pushed there yet."}
            return {
                "error": f"No live canvas state found under this project's own root ({root}); "
                         "the GUI has a different project open.",
                "divergence": binding_divergence(binding, root),
            }

        src_image = state.get("image_path") or ""
        if not Path(src_image).is_file():
            return {"error": f"Canvas state references a missing image: {src_image}"}

        # Geometry is valid only when its identity matches the meta document: a heartbeat for a
        # different image/tab means the stored shapes are stale and must not render.
        sdoc = _read(shapes_doc) or {}
        shapes_valid = (
            sdoc.get("image_path") == state.get("image_path") and sdoc.get("tab") == state.get("tab")
        )
        shapes = (sdoc.get("shapes") or []) if shapes_valid else []
        viewport = state.get("viewport")
        # Read exactly the region being rendered: the human's viewport is a rectangle in the
        # image's own grid, so a raster far too large to decode whole is still capturable.
        region = None
        if crop_to_viewport and viewport and viewport.get("w") and viewport.get("h"):
            region = (float(viewport.get("x", 0)), float(viewport.get("y", 0)),
                      float(viewport["w"]), float(viewport["h"]))
        read = _display_for_path(src_image, max_edge=max_edge, region=region)
        out = render_canvas_state(read.pixels, shapes,
                                  origin=(read.rect.x0, read.rect.y0), scale=read.scale)

        # The binding fence, only when this attempt started same_root: render_last_known's own
        # render reads only this project's documents, so a binding already elsewhere can't stale it.
        if same_root:
            try:
                binding_after = read_canvas_binding()
            except GuiBindingUnreadable as exc:
                return {"error": str(exc)}
            if binding_after is None or binding_after.get("generation") != binding.get("generation"):
                if attempt == 0:
                    continue
                return {
                    "error": "The GUI's open project changed while this call was rendering; "
                             "retry the call.",
                    "divergence": binding_divergence(binding_after or binding, root),
                }
        break

    # Every path that reaches here returned already unless the loop broke with both set.
    assert binding is not None and state is not None
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
    live = same_root and (refreshed or age < 5.0)
    if not same_root:
        summary = (
            f"Rendered this project's last known {state.get('tab')} canvas for "
            f"{state.get('image')} ({len(shapes)} shapes, {age}s old; not live: the GUI has "
            "another project open)."
        )
    elif live:
        summary = f"Rendered the live {state.get('tab')} canvas for {state.get('image')} ({len(shapes)} shapes)."
    else:
        summary = (
            f"Rendered the last known {state.get('tab')} canvas for {state.get('image')} "
            f"({len(shapes)} shapes, {age}s old; the GUI did not answer the refresh ping; it may "
            "be closed, on another tab, or on a different project)."
        )
    summary += " Read image_path with your own image-capable read tool to see it."
    result = {
        "image_path": out,
        "source_image": src_image,
        "image": state.get("image"),
        "tab": state.get("tab"),
        "mode": state.get("mode"),
        "cut_armed": state.get("cut_armed"),
        "user": state.get("user"),
        "dirty": state.get("dirty"),
        "project_root": state.get("project_root"),
        "viewport": state.get("viewport"),
        "cropped_to_viewport": region is not None,
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
    if not same_root:
        result["divergence"] = binding_divergence(binding, root)
    return result


@audited
def overlay_reference_grid(
    image_path: str,
    tile_size: int | None = None,
    overlap: float = 0.0,
) -> dict:
    """Render image with a labeled reference-grid overlay for spatial referencing.

    Not an MCP tool: run through ``scripts/overlay_reference_grid.py``, per the admission standard
    (packages/tcip-mcp/CLAUDE.md), while staying importable for its own tests and for
    ``segment_prompt(grid_cells=...)``, which shares its underlying grid geometry
    (``reference_grid.reference_cells``) without calling this function itself.

    The grid lives in the raster's native pixel frame: square cells of ``tile_size``
    native pixels named spreadsheet-style ('A1' top-left; letter columns A-Z then AA,
    AB, ..., 1-based number rows). Rendered in yellow on the cells' true boundaries; a cell
    against the image edge clips to the frame rather than drawing past it. A cell's name draws
    only when the rendered cell's short edge clears the label's legibility floor and the label's
    own width fits inside the cell; either check failing skips the name and leaves the boundary
    alone, a property of that cell's own size rather than a rule biased toward grid edges. When
    ``tile_size`` is omitted it derives from the image dims and the artifact bound
    (``reference_grid.derive_pointing_tile_size``) so the rendered labels stay legible. Every
    response echoes the full grid geometry
    (``tile_size``, ``overlap``, ``cols``, ``rows``, ``width``, ``height``): pass the
    echoed ``tile_size``/``overlap`` to ``segment_prompt(grid_cells=...)`` so a cell name
    resolves against the grid that was actually rendered.

    Args:
        image_path: Absolute path to the image file.
        tile_size: Cell edge in native pixels; omitted derives a legible default.
        overlap: Cell overlap as a fraction of tile_size, training tiling's semantics.
    """
    from tcip_mcp.pipelines.reference_grid import (
        derive_pointing_tile_size,
        grid_geometry,
        reference_cells,
    )

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    display = _display_for_path(image_path)
    w, h = display.native_size
    if tile_size is None:
        tile_size = derive_pointing_tile_size(w, h)
    try:
        cells = reference_cells(w, h, tile_size, overlap, clamp=True)
    except ValueError as e:
        return {"error": str(e)}
    out = render_grid_overlay(display.pixels, cells, native_size=(w, h))

    geometry = grid_geometry(w, h, tile_size, overlap)
    last = f"{column_label(geometry['cols'] - 1)}{geometry['rows']}"
    return {
        "image_path": out,
        "summary": f"Reference grid ({geometry['cols']}x{geometry['rows']}, tile_size "
                   f"{tile_size}) rendered on {img.name}. Reference cells like 'A1' "
                   f"(top-left) to '{last}' (bottom-right).",
        **geometry,
    }
