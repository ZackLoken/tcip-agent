"""Vision tools: render annotations and predictions for visual analysis.

Each tool saves a rendered image under the project's ``.tcip/artifacts/viz/`` and returns the path
so the agent can call its client's own image-capable read tool on it to visually inspect it.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, NamedTuple

import tcip_store as ts

from tcip_annotation import Annotation, Point, bbox_of
from tcip_annotation.state import box_derivable, polygonal, prediction_score
from tcip_annotation.json_io import UnreadableLabelDocument, read_predictions
from tcip_annotation.json_io import read_annotations as read_labels
from tcip_annotation.viz import (
    render_canvas_state,
    render_comparison,
    render_detections,
    render_grid,
    render_grid_overlay,
    render_segmentations,
)

from tcip_mcp.pipelines.display_bounds import VIZ_ARTIFACT_MAX_EDGE
from tcip_annotation.matching import REVIEW_CONF_FLOOR
from tcip_mcp.project_paths import viz_output_path
from tcip_mcp.server import tool

if TYPE_CHECKING:
    import numpy as np

    from tcip_mcp.buckets import Bucket
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
    """An ``(x, y, w, h)`` region clamped to a non-empty rect inside a ``width`` x ``height``
    raster.
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

    The raster layer serves the region (an ``(x, y, w, h)`` rectangle in the raster's own grid, or
    the whole frame) at or under ``max_edge``, and nothing is ever materialized to a temp file.

    An 8-bit raster at 1/3/4 bands already holds display values, so it keeps its own pixels
    (grayscale repeated, alpha dropped) with no stretch. Every other raster has its first three
    bands composited and independently min-max stretched, through ``composite_display_rgb``.
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
    ``images/`` layout) is returned as itself.
    """
    from tcip_mcp.pipelines import image_utils

    try:
        return image_utils.resolve_image_path(image_path)
    except (FileNotFoundError, image_utils.BandGroupIncomplete):
        return Path(image_path)


def _display_for_path(image_path: str, *, max_edge: int = VIZ_ARTIFACT_MAX_EDGE,
                      region: tuple[float, float, float, float] | None = None) -> DisplayRead:
    """As ``_display_for_stem``, for a caller that already has a path rather than a
    ``(dir, stem)`` pair.
    """
    return _read_for_display(_source_for_path(image_path), max_edge=max_edge, region=region)


def _subject_indexer() -> tuple[dict[str, int], Callable[[str], int]]:
    """A stable subject-name → color-index map for the (int-keyed) renderers, plus its indexer.

    Labels are name-based; the viz layer colors by an integer and labels from a ``{index: name}``
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
    """The annotations a box renderer can draw: geometry-bearing, with a Point excluded."""
    return [a for a in anns if box_derivable(a.geometry)]


def _n_points(anns: list[Annotation]) -> int:
    return sum(1 for a in anns if isinstance(a.geometry, Point))


def _point_note(n: int) -> str:
    return f" ({n} point annotation(s) not drawn, no point renderer yet)" if n else ""


def _legend_name(a: Annotation, *, scope) -> str:
    """The name a render's legend shows for ``a``: its subject followed by each value it carries
    under ``scope``'s attributes, in declared order (:func:`~tcip_annotation.json_io.
    attribute_ids` and its inverse), its subject alone under no scope, a scope naming no subject
    or declaring no attribute, or for a record of another subject.
    """
    from tcip_annotation.json_io import attribute_ids, attribute_values
    from tcip_mcp.pipelines.data.selection import DOCUMENT

    if scope is None or scope.subject is None:
        return a.subject
    attributes = scope.admitted_for(DOCUMENT, "the rendered bucket's scope").attributes
    ids = attribute_ids(a, scope.subject, attributes)
    return " ".join([a.subject, *(attribute_values(ids, attributes).values() if ids else ())])


def _box_dict(a: Annotation, index: Callable[[str], int], *, scope=None) -> dict:
    geometry = a.geometry
    assert box_derivable(geometry), \
        "every caller passes a _boxable-filtered or box-rendered annotation"
    b = bbox_of(geometry)
    d = {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
        "class_id": index(_legend_name(a, scope=scope))}
    if a.score is not None:
        d["confidence"] = a.score
    return d


def _poly_dict(a: Annotation, index: Callable[[str], int], *, scope=None) -> dict:
    geometry = a.geometry
    assert polygonal(geometry), "called only for the segmentation task's own shapes"
    return {"rings": [[[p[0], p[1]] for p in ring] for ring in geometry.rings],
            "class_id": index(_legend_name(a, scope=scope))}


def visualize(
    project: Path,
    source: str,
    path: str,
    task: str = "detect",
    class_names: str = "",
    conf_threshold: float = REVIEW_CONF_FLOOR,
    iou_threshold: float = 0.5,
    n: int = 16,
    predictions_dir: str = "",
) -> dict:
    """Render annotations, predictions, a GT-vs-prediction comparison, or a sample grid.

    Saves under ``project``'s ``.tcip/artifacts/viz/`` and returns ``image_path`` for the agent's
    own image-capable read tool.

    Rendering conventions, shared across every source: boxes/masks color by class through the
    20-class palette in ``tcip_annotation.viz``, indexed by first-seen order within one render
    call, and not stable across renders. A 'comparison' render outlines GT green and predictions
    red, with yellow center-to-center lines joining matched pairs. A detection label carries the
    class name and, when the box carries a confidence score, the score; a segmentation label
    carries the class name only. Each source image is read at up to
    ``display_bounds.VIZ_ARTIFACT_MAX_EDGE`` (1024px) on its longest edge before rendering; for
    source='dataset' the per-sample renders built this way are then tiled into a grid that saves at
    ``cols`` x 256 by ``rows`` x 256 pixels, growing with ``n``.

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
        class_names: Comma-separated class names (e.g. "fruit,shoot").
        conf_threshold: Minimum confidence; filters displayed predictions (source='predictions')
            and the predictions matched against GT (source='comparison'). Defaults to the viewing
            filter ``matching.REVIEW_CONF_FLOOR``.
        iou_threshold: IoU threshold for a positive match (source='comparison' only).
        n: Number of samples in the grid (source='dataset' only).
        predictions_dir: The published bucket whose document for the image is rendered
            (source='predictions' and 'comparison', where it is required).
    """
    if source == "annotations":
        return _viz_annotations(project, path, task=task, class_names=class_names)
    if source in ("predictions", "comparison") and not predictions_dir:
        return {"error": f"source={source!r} requires predictions_dir, the published bucket "
                         "whose predictions to render."}
    if source == "predictions":
        return _viz_predictions(
            project, path, predictions_dir, task=task, class_names=class_names,
            conf_threshold=conf_threshold)
    if source == "comparison":
        return _viz_comparison(
            project, path, predictions_dir, task=task, iou_threshold=iou_threshold,
            class_names=class_names, conf_threshold=conf_threshold)
    if source == "dataset":
        return _viz_dataset_sample(project, path, n=n, task=task, class_names=class_names)
    return {
        "error": f"Unknown source '{source}'. "
        "Use 'annotations', 'predictions', 'comparison', or 'dataset'."
    }


def _viz_annotations(
    project: Path,
    image_path: str,
    task: str = "detect",
    class_names: str = "",
) -> dict:
    """Render ground-truth annotations on a single image."""
    from tcip_mcp.dataset_layout import find_gt_label

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    stem = img.stem
    label_path = find_gt_label(image_path)
    if label_path is None:
        return {"error": f"No labels found for {stem}"}

    try:
        anns = read_labels(str(label_path))
    except UnreadableLabelDocument as exc:
        return {"error": str(exc)}
    idx, index = _subject_indexer()

    n_points = _n_points(anns)
    read = _display_for_path(image_path)
    if task == "detect":
        shapes = _boxable(anns)
        out = render_detections(read.pixels, [_box_dict(a, index) for a in shapes],
                                native_size=read.native_size, class_names=_name_map(idx),
                                output_path=viz_output_path(project, "detections"))
        summary = f"Rendered {len(shapes)} detections on {img.name}"
        if shapes:
            from collections import Counter
            counts = Counter(a.subject for a in shapes)
            summary += ": " + ", ".join(f"{v} {k}" for k, v in counts.most_common())
        summary += _point_note(n_points)
    else:
        shapes = [a for a in anns if polygonal(a.geometry)]
        out = render_segmentations(read.pixels, [_poly_dict(a, index) for a in shapes],
                                   native_size=read.native_size, class_names=_name_map(idx),
                                   output_path=viz_output_path(project, "segmentations"))
        summary = f"Rendered {len(shapes)} segmentation masks on {img.name}" + _point_note(n_points)

    return {
        "image_path": out,
        "summary": summary,
        # `count` is the stable key across all visualize sources; the source-specific alias stays.
        "count": len(shapes),
        # Disclosed, not folded into `count`: these annotations are real but this renderer can't draw
        # them, and a silently smaller count would read as "the image has fewer annotations".
        "points_not_rendered": n_points,
    }


def _bucket_document(project: Path, predictions_dir: str, image_path: str):
    """``(document path or None, class scope)`` for ``image_path`` in the published bucket at
    ``predictions_dir``; a directory that is no bucket, or whose record will not read, refuses
    (``ValueError``)."""
    from tcip_mcp.buckets import read_bucket

    bucket = read_bucket(Path(project, predictions_dir))
    return bucket.document(image_path), bucket.scope


def _viz_predictions(
    project: Path,
    image_path: str,
    predictions_dir: str,
    task: str = "detect",
    class_names: str = "",
    conf_threshold: float = 0.0,
) -> dict:
    """Render a bucket's predictions on a single image.

    The legend keys each detection by its subject and the values it carries under the attributes
    the bucket's recorded scope declares (:func:`_legend_name`).
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    try:
        pred_file, scope = _bucket_document(project, predictions_dir, image_path)
        if pred_file is None:
            return {"error": f"No predictions found for {img.stem} in {predictions_dir}"}
        preds = read_predictions(str(pred_file))
    except (UnreadableLabelDocument, ValueError) as exc:
        return {"error": str(exc)}
    preds = [a for a in preds if prediction_score(a) >= conf_threshold]
    idx, index = _subject_indexer()

    n_points = _n_points(preds)
    read = _display_for_path(image_path)
    if task == "detect":
        shapes = _boxable(preds)
        out = render_detections(
            read.pixels, [_box_dict(a, index, scope=scope) for a in shapes],
            native_size=read.native_size, class_names=_name_map(idx),
            output_path=viz_output_path(project, "detections"))
        summary = f"Rendered {len(shapes)} predictions on {img.name}" + _point_note(n_points)
    else:
        shapes = [a for a in preds if polygonal(a.geometry)]
        out = render_segmentations(
            read.pixels, [_poly_dict(a, index, scope=scope) for a in shapes],
            native_size=read.native_size, class_names=_name_map(idx),
            output_path=viz_output_path(project, "segmentations"))
        summary = f"Rendered {len(shapes)} prediction masks on {img.name}" + _point_note(n_points)

    return {
        "image_path": out,
        "summary": summary,
        # `count` is the stable key across all visualize sources; the source-specific alias stays.
        "count": len(shapes),
        "points_not_rendered": n_points,
    }


def _viz_comparison(
    project: Path,
    image_path: str,
    predictions_dir: str,
    task: str = "detect",
    iou_threshold: float = 0.5,
    class_names: str = "",
    conf_threshold: float = REVIEW_CONF_FLOOR,
) -> dict:
    """Render GT vs prediction comparison with match indicators.

    Green = ground truth, Red = predictions, Yellow lines = matched pairs, and the TP/FP/FN
    counts, all as the single-image scoring
    (:func:`~tcip_mcp.tools.annotation_tools.score_predictions`) states them at ``iou_threshold``
    over the predictions at or above ``conf_threshold``; the legend keys the prediction side by
    its decoded value (:func:`_legend_name`).
    """
    from tcip_annotation.matching import pair_proposals

    from tcip_mcp.dataset_layout import find_gt_label
    from tcip_mcp.pipelines.training.evaluation import resolve_match_criterion
    from tcip_mcp.tools.annotation_tools import score_predictions

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    stem = img.stem
    idx, index = _subject_indexer()

    label_path = find_gt_label(image_path)
    if label_path is None:
        return {"error": f"No labels found for {stem}"}
    try:
        gt_all = read_labels(str(label_path))
    except UnreadableLabelDocument as exc:
        return {"error": str(exc)}
    gt = _boxable(gt_all)
    gt_dicts = [_box_dict(a, index) for a in gt]

    try:
        pred_file, scope = _bucket_document(project, predictions_dir, image_path)
    except ValueError as exc:
        return {"error": str(exc)}
    pred_dicts: list[dict] = []
    tp_matches: list[tuple[int, int]] = []
    unpredicted = pair_proposals(
        gt_all, [], resolve_match_criterion(None, [], iou_threshold=iou_threshold))
    tp, fp, fn = 0, 0, len(unpredicted.missed)
    if pred_file is not None:
        try:
            preds_all = read_predictions(str(pred_file))
        except UnreadableLabelDocument as exc:
            return {"error": str(exc)}
        pred_dicts = [_box_dict(a, index, scope=scope) for a in _boxable(preds_all)]
        scored = score_predictions(image_path, str(Path(project, predictions_dir)),
                                   iou_threshold=iou_threshold, conf_threshold=conf_threshold,
                                   detail=True)
        if "error" in scored:
            return {"error": scored["error"]}
        # The scoring indexes the whole documents; the renderer draws their boxable entries.
        gpos = {i: k for k, i in enumerate(
            i for i, a in enumerate(gt_all) if box_derivable(a.geometry))}
        ppos = {i: k for k, i in enumerate(
            i for i, a in enumerate(preds_all) if box_derivable(a.geometry))}
        tp_matches = [(gpos[g], ppos[p]) for g, p in scored["matches"]]
        tp, fp, fn = scored["tp"], scored["fp"], scored["fn"]

    read = _display_for_path(image_path)
    out = render_comparison(read.pixels, gt_dicts, pred_dicts, native_size=read.native_size,
                            matches=tp_matches, class_names=_name_map(idx),
                            output_path=viz_output_path(project, "comparison"))

    return {
        "image_path": out,
        "summary": f"GT={len(gt_dicts)}, Pred={len(pred_dicts)}, TP={tp}, FP={fp}, FN={fn}",
        "gt_count": len(gt_dicts),
        "pred_count": len(pred_dicts),
        "tp": tp, "fp": fp, "fn": fn,
    }


def get_worst_predictions(bucket: Bucket, labels_dir: str, top_k: int = 8) -> dict:
    """Return the ``top_k`` images ranked worst by a count-mismatch + low-confidence triage heuristic.

    This is a cheap triage signal, not a quality metric: it does no IoU matching and computes
    no loss. The score is ``2·|n_gt−n_pred as a shortfall| + |surplus| + (1−avg_conf)``, purely
    the difference in box *counts* plus mean confidence, so an image with the right count but
    every box mislocated scores as good. Use it to surface likely-bad frames for a human to look
    at; for true TP/FP/FN ranking use ``score_predictions`` (``detail=True``, IoU-matched). Only
    the documents the bucket's record names are ranked; a labeled image it names none for was not
    predicted, and is listed under ``not_predicted`` rather than scored.

    Args:
        bucket: The published bucket whose recorded documents are ranked.
        labels_dir: Directory with per-image JSON ground-truth label files.
        top_k: Number of worst images to return.
    """
    gt_path = Path(labels_dir)
    if not gt_path.is_dir():
        return {"error": f"Labels directory not found: {labels_dir}"}

    from tcip_annotation.json_io import detection_annotations, prediction_documents

    # Both sides counted as a count counts: objects with a box, a crowd region and a Point none.
    scores: list[tuple[str, float]] = []
    for pred_file in bucket.document_paths:
        preds = detection_annotations(pred_file)
        gt_anns = detection_annotations(gt_path / pred_file.name)

        n_pred = len(preds)
        n_gt = len(gt_anns)

        # Simple error heuristic: |pred - gt| + missed + extra + low confidence
        missed = max(0, n_gt - n_pred)
        extra = max(0, n_pred - n_gt)
        avg_conf = sum(map(prediction_score, preds)) / n_pred if n_pred else 0.0

        # Higher score = worse prediction
        error_score = missed * 2.0 + extra * 1.0 + (1.0 - avg_conf)
        scores.append((pred_file.stem, error_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    worst = scores[:top_k]

    return {
        "worst_images": [{"stem": s, "error_score": round(sc, 3)} for s, sc in worst],
        "total_evaluated": len(scores),
        "not_predicted": [f.stem for f in prediction_documents(gt_path)
                          if f.stem not in bucket.documents],
    }


def render_failure_cases(
    project: Path,
    predictions_dir: str,
    labels_dir: str,
    images_dir: str = "",
    task: str = "detect",
    top_k: int = 10,
    class_names: str = "",
) -> dict:
    """Find and render the worst predictions for failure analysis.

    Ranks by a count-mismatch + low-confidence heuristic (`get_worst_predictions`); no IoU
    matching, so an image with the right box count but every box mislocated scores as good. Not a
    substitute for `score_predictions`(`detail=True`)'s IoU-matched TP/FP/FN when mislocalization
    itself is the question.

    Returns a grid image and individual failure case images.

    Args:
        predictions_dir: The published bucket whose documents are ranked and rendered.
        labels_dir: Directory with ground-truth label files.
        images_dir: Directory with source images. Auto-detected if empty.
        task: 'detect' or 'segment'.
        top_k: Number of worst cases to render.
        class_names: Comma-separated class names.
    """
    from tcip_mcp.buckets import read_bucket

    try:
        bucket = read_bucket(Path(project, predictions_dir))
    except ValueError as exc:
        return {"error": str(exc)}
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
        worst = get_worst_predictions(bucket, labels_dir, top_k=top_k)
    except UnreadableLabelDocument as exc:
        return {"error": str(exc)}
    if "error" in worst:
        return worst

    worst_items = worst.get("worst_images", [])
    if not worst_items:
        return {"summary": "No prediction errors found", "image_path": None}

    from tcip_mcp.dataset_layout import label_filename

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

        gt_file = Path(labels_dir) / label_filename(stem)
        pred_file = bucket.document(bucket.documents[stem]) if stem in bucket.documents else None
        try:
            gt_dicts = ([_box_dict(a, index) for a in _boxable(read_labels(str(gt_file)))]
                        if gt_file.is_file() else [])
            pred_dicts = ([_box_dict(a, index) for a in _boxable(read_predictions(str(pred_file)))]
                          if pred_file is not None else [])
        except UnreadableLabelDocument as exc:
            return {"error": str(exc)}

        out = viz_output_path(project, f"failure_{len(case_paths):03d}_{stem}")
        render_comparison(read.pixels, gt_dicts, pred_dicts, native_size=read.native_size,
                          class_names=_name_map(idx), output_path=out)
        case_paths.append(out)
        titles.append(f"{stem} (err={item['error_score']:.1f})")

    # Render grid of all failure cases
    grid_path = None
    if case_paths:
        grid_path = render_grid(case_paths, titles=titles, cols=min(4, len(case_paths)),
                                output_path=viz_output_path(project, "failures"))

    return {
        "image_path": grid_path,
        "case_images": case_paths,
        "summary": f"Rendered {len(case_paths)} worst prediction cases (of {worst['total_evaluated']} evaluated)",
        "worst_images": worst_items,
    }


def _viz_dataset_sample(
    project: Path,
    folder_path: str,
    n: int = 16,
    task: str = "detect",
    class_names: str = "",
) -> dict:
    """Render a grid of random annotated dataset samples."""
    from tcip_mcp.dataset_layout import find_gt_label, image_root
    from tcip_mcp.pipelines.image_utils import (
        BandGroupIncomplete, list_logical_images, refuse_incomplete_band_group, source_path_of,
    )

    root = Path(folder_path)
    images_dir = image_root(root)
    if not images_dir.is_dir():
        return {"error": f"Images directory not found: {images_dir}"}

    # Every logical image at or under images_dir, a grouped capture as one entry, at any depth.
    dirs = {images_dir} | {p for p in images_dir.rglob("*") if p.is_dir()}
    all_images = [(stem, src) for d in sorted(dirs)
                  for stem, src in sorted(list_logical_images(d).items())]
    if not all_images:
        return {"error": "No images found in dataset"}

    sample = random.sample(all_images, min(n, len(all_images)))
    rendered_paths = []
    titles = []
    for stem, enumerated in sample:
        try:
            source = refuse_incomplete_band_group(enumerated)
        except BandGroupIncomplete:
            continue
        rep_path = source_path_of(source)
        label_path = find_gt_label(str(rep_path))
        read = _read_for_display(source)
        if label_path is not None:
            idx, index = _subject_indexer()
            try:
                anns = read_labels(str(label_path))
            except UnreadableLabelDocument as exc:
                return {"error": str(exc)}
            if task == "detect":
                shapes = _boxable(anns)
                out = render_detections(read.pixels, [_box_dict(a, index) for a in shapes],
                                        native_size=read.native_size, class_names=_name_map(idx),
                                        output_path=viz_output_path(project, "detections"))
            else:
                shapes = [a for a in anns if polygonal(a.geometry)]
                out = render_segmentations(read.pixels, [_poly_dict(a, index) for a in shapes],
                                           native_size=read.native_size,
                                           class_names=_name_map(idx),
                                           output_path=viz_output_path(project, "segmentations"))
            titles.append(f"{stem} ({len(shapes)})")
        else:
            # An unlabeled sample renders too, with nothing drawn on it: the grid tiles rendered
            # artifacts, so every cell has to be one.
            out = render_detections(read.pixels, [], native_size=read.native_size,
                                    output_path=viz_output_path(project, "detections"))
            titles.append(f"{stem} (no labels)")

        rendered_paths.append(out)

    grid_path = render_grid(rendered_paths, titles=titles, cols=min(4, len(rendered_paths)),
                            output_path=viz_output_path(project, "grid"))

    return {
        "image_path": grid_path,
        "summary": f"Grid of {len(sample)} annotated samples from {root.name}",
        "count": len(sample),
        "total_images": len(all_images),
    }


def _received_at(document: dict, name: str) -> datetime:
    """The instant the backend received the canvas document ``name``. Refuses (``ValueError``) a
    document that carries none."""
    if "received_at" not in document:
        raise ValueError(f"the canvas document {name} carries no received_at")
    return datetime.fromisoformat(document["received_at"])


@tool()
def capture_live_canvas(
    project: Path,
    workspace: Path,
    refresh: bool = True,
    crop_to_viewport: bool = True,
    max_edge: int = 1600,
) -> dict:
    """Render exactly what the human's GUI canvas showed for this project: image, shapes, viewport.

    Reads the canvas state the GUI pushes under the project's ``.tcip/state/``:
    ``canvas_live.json`` (image, viewport, classes, counts, tab, mode, active_subject,
    cut_armed, dirty and user) and ``canvas_shapes.json`` (the full display-resolved geometry,
    including unsaved edits and an in-progress drawing). The backend writes a push only under the
    project it has open, so these documents are always this project's own. Renders the region
    being shown at up to ``max_edge`` and returns the artifact path for the agent's own
    image-capable read tool, plus the classes schema, per-tag/per-creator counts,
    and the state's age. A canvas document that will not read raises the store's own error.

    Args:
        refresh: Ping the GUI (via the panel-event hub) to push fresh state first, waiting briefly
            for it to land. The backend delivers the ping only while it has this project open;
            otherwise the last pushed state renders, labeled not live, with the backend's answer.
        crop_to_viewport: Render only the region the human currently sees (their zoom/pan). Pass
            False for the full frame with the same overlays.
        max_edge: Downscale the rendered output to at most this edge (px).
    """
    import time as _time

    from tcip_mcp.web_client import (
        PANEL_EVENT_CANVAS_STATE_REQUEST, canvas_geometry_key, canvas_meta_key, post_panel_event,
    )

    meta_doc = canvas_meta_key(str(project))
    shapes_doc = canvas_geometry_key(str(project))

    previous = ts.read(meta_doc, default=None)
    since = None if previous is None else _received_at(previous, "canvas_live")
    refreshed = False
    ping: dict = {}
    if refresh:
        ping = post_panel_event(project, workspace, "app", PANEL_EVENT_CANVAS_STATE_REQUEST, {})
        if ping.get("delivered"):
            for _ in range(12):  # ~2.4s for the GUI's flush to land
                _time.sleep(0.2)
                cur = ts.read(meta_doc, default=None)
                if cur and (since is None or _received_at(cur, "canvas_live") > since):
                    refreshed = True
                    break

    state = ts.read(meta_doc, default=None)
    if state is None:
        return {"error": "No canvas state has been pushed for this project; the GUI pushes its "
                         "canvas only while it has this project open.",
                "refresh_answer": ping}

    from tcip_mcp.registry_paths import resolved_registry_path

    src_image = str(resolved_registry_path(project, state["image_path"]))
    if not Path(src_image).is_file():
        return {"error": f"Canvas state references a missing image: {src_image}"}

    # Geometry is valid only when its identity matches the meta document: a heartbeat for a
    # different image/tab means the stored shapes are stale and must not render.
    sdoc = ts.read(shapes_doc, default=None) or {}
    shapes_valid = (
        sdoc.get("image_path") == state.get("image_path") and sdoc.get("tab") == state.get("tab")
    )
    shapes: list = (sdoc.get("shapes") or []) if shapes_valid else []
    viewport = state.get("viewport")
    # Read exactly the region being rendered: the human's viewport is a rectangle in the
    # image's own grid, so a raster far too large to decode whole is still capturable.
    region = None
    if crop_to_viewport and viewport and viewport.get("w") and viewport.get("h"):
        region = (float(viewport.get("x", 0)), float(viewport.get("y", 0)),
                  float(viewport["w"]), float(viewport["h"]))
    read = _display_for_path(src_image, max_edge=max_edge, region=region)
    out = render_canvas_state(read.pixels, shapes,
                              origin=(read.rect.x0, read.rect.y0), scale=read.scale,
                              output_path=viz_output_path(project, "canvas", suffix=".jpg"))
    ping_delivered = bool(ping.get("delivered"))

    now = datetime.now(timezone.utc)
    tag_counts: dict[str, int] = {}
    creator_counts: dict[str, int] = {}
    for s in shapes:
        if isinstance(s, dict):
            tag_counts[str(s.get("tag") or "untagged")] = tag_counts.get(str(s.get("tag") or "untagged"), 0) + 1
            cb = s.get("created_by")
            if cb:
                creator_counts[str(cb)] = creator_counts.get(str(cb), 0) + 1

    age = round((now - _received_at(state, "canvas_live")).total_seconds(), 1)
    if refreshed or age < 5.0:
        summary = f"Rendered the live {state.get('tab')} canvas for {state.get('image')} ({len(shapes)} shapes)."
    else:
        summary = (
            f"Rendered the last known {state.get('tab')} canvas for {state.get('image')} "
            f"({len(shapes)} shapes, {age}s old; not live: the GUI did not answer the refresh "
            "ping, see refresh_answer)."
        )
    summary += " Read image_path with your own image-capable read tool to see it."
    return {
        "image_path": out,
        "source_image": src_image,
        "image": state.get("image"),
        "tab": state.get("tab"),
        "mode": state.get("mode"),
        "cut_armed": state.get("cut_armed"),
        "user": state.get("user"),
        "dirty": state.get("dirty"),
        "viewport": state.get("viewport"),
        "cropped_to_viewport": region is not None,
        "classes": state.get("classes") or [],
        "counts": state.get("counts"),
        "shape_counts_by_tag": tag_counts,
        "shape_counts_by_creator": creator_counts,
        "state_age_seconds": age,
        "shapes_age_seconds": (round((now - _received_at(sdoc, "canvas_shapes")).total_seconds(), 1)
                               if shapes_valid else None),
        # True when no valid geometry exists for this image/tab yet (heartbeat-only or stale).
        "shapes_missing": not shapes_valid,
        # Did a fresh push land after our ping? False + delivered ping = GUI not listening here.
        "refreshed": refreshed,
        "refresh_ping_delivered": ping_delivered,
        "refresh_answer": ping,
        "summary": summary,
    }


def overlay_reference_grid(
    project: Path,
    image_path: str,
    tile_size: int | None = None,
    overlap: float = 0.0,
) -> dict:
    """Render image with a labeled reference-grid overlay for spatial referencing.

    The grid lives in the raster's native pixel frame: square cells of ``tile_size`` native pixels
    named spreadsheet-style ('A1' top-left; letter columns A-Z then AA, AB, ..., 1-based number
    rows). Rendered in yellow on the cells' true boundaries; a cell against the image edge clips to
    the frame rather than drawing past it. A cell's name draws only when the rendered cell's short
    edge clears the label's legibility floor and the label's own width fits inside the cell; either
    check failing skips the name and leaves the boundary alone. When ``tile_size`` is omitted it
    derives from the image dims and the artifact bound
    (``reference_grid.derive_pointing_tile_size``) so the rendered labels stay legible. Every
    response echoes the full grid geometry (``tile_size``, ``overlap``, ``cols``, ``rows``,
    ``width``, ``height``): pass the echoed ``tile_size``/``overlap`` to
    ``propose_annotations(grid_cells=...)`` so a cell name resolves against the grid that was
    actually rendered.

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
    out = render_grid_overlay(display.pixels, cells, native_size=(w, h),
                              output_path=viz_output_path(project, "grid_overlay"))

    geometry = grid_geometry(w, h, tile_size, overlap)
    return {
        "image_path": out,
        "summary": f"Reference grid ({geometry['cols']}x{geometry['rows']}, tile_size "
                   f"{tile_size}) rendered on {img.name}. Reference cells like 'A1' "
                   f"(top-left) to '{cells[-1].name}' (bottom-right).",
        **geometry,
    }
