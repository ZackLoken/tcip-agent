"""Vision tools: render annotations and predictions to an image under the project's
``.tcip/artifacts/viz/`` and return its path."""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, NamedTuple

import tcip_store as ts

from tcip_annotation import Annotation, Point, bbox_of
from tcip_annotation.state import box_derivable, polygonal, prediction_score
from tcip_annotation.json_io import (
    UnreadableLabelDocumentError, read_document_versioned, read_predictions,
)
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
from tcip_mcp.pipelines.raster_source import SourceHeader
from tcip_mcp.project_paths import viz_output_path
from tcip_mcp.registry_paths import located
from tcip_mcp.server import tool
from tcip_mcp.workspace import BoundProject

if TYPE_CHECKING:
    import numpy as np

    from tcip_mcp.buckets import Bucket
    from tcip_mcp.pipelines.raster_source import RasterSource, Rect


class DisplayRead(NamedTuple):
    """Display pixels for a renderer, with the frame facts that place annotations on them.

    ``pixels`` is uint8 RGB; ``rect`` is the region of the raster they were read from, so the
    pixels' own size against it is the served resolution on each axis, what a renderer drawing a
    crop needs; ``native_size`` is the raster's own ``(width, height)``, the frame annotation
    coordinates are measured in.
    """

    pixels: np.ndarray
    rect: Rect
    native_size: tuple[int, int]


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


def _read_for_display(raster: "RasterSource", *, max_edge: int = VIZ_ARTIFACT_MAX_EDGE,
                      region: tuple[float, float, float, float] | None = None) -> DisplayRead:
    """Read the open ``raster`` as display pixels a renderer can draw on.

    The raster layer serves the region (an ``(x, y, w, h)`` rectangle in the raster's own grid, or
    the whole frame) at or under ``max_edge`` through ``display_bounds.plan_read``, off the
    overview level the plan names, and nothing is ever materialized to a temp file.

    An 8-bit raster at 1/3/4 bands already holds display values, so it keeps its own pixels
    (grayscale repeated, alpha dropped) with no stretch. Every other raster has its first three
    bands composited and independently min-max stretched, through ``composite_display_rgb``.
    """
    from tcip_mcp.pipelines.band_stats import composite_display_rgb
    from tcip_mcp.pipelines.display_bounds import plan_read, planned_read
    from tcip_mcp.pipelines.raster_source import Rect

    native = (int(raster.width), int(raster.height))
    rect = (Rect(0, 0, raster.width, raster.height) if region is None
            else _clamped_rect(region, raster.width, raster.height))
    plan = plan_read(rect, raster.width, raster.height, raster.level_dims(),
                     rect.width * rect.height, max_edge)
    pixels, _spec = planned_read(raster, rect, plan)
    bands = int(pixels.shape[-1])
    idxs = [0, 1, 2] if bands >= 3 else [0, 0, 0]
    plain = bands in (1, 3, 4) and pixels.dtype == "uint8"
    return DisplayRead(composite_display_rgb(pixels, idxs, "none" if plain else "minmax"),
                       rect, native)


def _display_of(header: "SourceHeader", *, max_edge: int = VIZ_ARTIFACT_MAX_EDGE,
                region: tuple[float, float, float, float] | None = None) -> DisplayRead:
    """:func:`_read_for_display` of the source ``header`` describes, opened once at the image
    route's count (:meth:`~tcip_mcp.pipelines.raster_source.SourceHeader.open_at_route_count`)."""
    with header.open_at_route_count() as raster:
        return _read_for_display(raster, max_edge=max_edge, region=region)


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
        d["score"] = a.score
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
    bucket: str = "",
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
            'dataset'     = grid of n random annotated samples (path = dataset root folder
            containing images/).
        path: Image file (annotations/predictions/comparison) or dataset folder (dataset).
        task: 'detect' or 'segment'.
        class_names: Comma-separated class names (e.g. "fruit,shoot").
        conf_threshold: Minimum confidence; filters displayed predictions (source='predictions')
            and the predictions matched against GT (source='comparison'). Defaults to the viewing
            filter ``matching.REVIEW_CONF_FLOOR``.
        iou_threshold: IoU threshold for a positive match (source='comparison' only).
        n: Number of samples in the grid (source='dataset' only).
        bucket: The name of the bucket, published under the image's dataset root, whose document
            for the image is rendered (source='predictions' and 'comparison', where it is
            required).
    """
    path = str(located(path, project))
    if source == "annotations":
        return _viz_annotations(project, path, task=task, class_names=class_names)
    if source in ("predictions", "comparison") and not bucket:
        return {"error": f"source={source!r} requires bucket, the published bucket whose "
                         "predictions to render."}
    if source == "predictions":
        return _viz_predictions(
            project, path, bucket, task=task, class_names=class_names,
            conf_threshold=conf_threshold)
    if source == "comparison":
        return _viz_comparison(
            project, path, bucket, task=task, iou_threshold=iou_threshold,
            class_names=class_names, conf_threshold=conf_threshold)
    if source == "dataset":
        return _viz_dataset_sample(project, path, n=n, task=task, class_names=class_names)
    return {
        "error": f"Unknown source '{source}'. "
        "Use 'annotations', 'predictions', 'comparison', or 'dataset'."
    }


def _labels_at(key: ts.Key) -> list[Annotation] | None:
    """The annotations of the label document ``key``, ``None`` when there is none."""
    document, version = read_document_versioned(key)
    return None if version == ts.Version.ABSENT else document.annotations


def _viz_annotations(
    project: Path,
    image_path: str,
    task: str = "detect",
    class_names: str = "",
) -> dict:
    """Render ground-truth annotations on the logical image ``image_path`` names
    (:func:`~tcip_mcp.pipelines.image_utils.resolve_image_path`, whose refusals answer as an
    error), read from its own label document."""
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    from tcip_mcp.dataset_layout import label_key_of
    from tcip_mcp.pipelines.image_utils import resolve_image_path, source_path_of

    try:
        source = resolve_image_path(image_path)
        anns = _labels_at(label_key_of(source_path_of(source)))
    except (UnreadableLabelDocumentError, ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}
    if anns is None:
        return {"error": f"No labels found for {img.stem}"}
    idx, index = _subject_indexer()

    n_points = _n_points(anns)
    read = _display_of(SourceHeader(source))
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
        # Disclosed, not folded into `count`: these annotations are real but this renderer can't
        # draw them, and a silently smaller count would read as "the image has fewer annotations".
        "points_not_rendered": n_points,
    }


def _viz_predictions(
    project: Path,
    image_path: str,
    bucket: str,
    task: str = "detect",
    class_names: str = "",
    conf_threshold: float = 0.0,
) -> dict:
    """Render a bucket's predictions on the logical image ``image_path`` names
    (:func:`~tcip_mcp.pipelines.image_utils.resolve_image_path`, whose refusals answer as an
    error), read from the bucket's document for it.

    The legend keys each detection by its subject and the values it carries under the attributes
    the bucket's recorded scope declares (:func:`_legend_name`).
    """
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.dataset_layout import parse_image_path
    from tcip_mcp.pipelines.image_utils import resolve_image_path, source_path_of

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    try:
        source = resolve_image_path(image_path)
        root, _capture, stem = parse_image_path(source_path_of(source))
        found = read_bucket(root, bucket)
        pred_key, scope = found.document_key(stem), found.scope
        if pred_key is None:
            return {"error": f"No predictions found for {stem} in bucket {bucket!r}"}
        preds = read_predictions(pred_key).annotations
    except (UnreadableLabelDocumentError, ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}
    preds = [a for a in preds if prediction_score(a) >= conf_threshold]
    idx, index = _subject_indexer()

    n_points = _n_points(preds)
    read = _display_of(SourceHeader(source))
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
    bucket: str,
    task: str = "detect",
    iou_threshold: float = 0.5,
    class_names: str = "",
    conf_threshold: float = REVIEW_CONF_FLOOR,
) -> dict:
    """Render GT vs prediction comparison with match indicators.

    Green = ground truth, Red = predictions, Yellow lines = matched pairs, and the TP/FP/FN
    counts, all as the image's scoring
    (:func:`~tcip_mcp.pipelines.training.evaluation.score_bucket`) matches them at
    ``iou_threshold`` over the predictions at or above ``conf_threshold``; the legend keys the
    prediction side by its decoded value (:func:`_legend_name`). An image the bucket names no
    document for draws its ground truth alone, every object missed.
    """
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.dataset_layout import parse_image_path
    from tcip_mcp.pipelines.image_utils import resolve_image_path, source_path_of
    from tcip_mcp.pipelines.training.evaluation import score_bucket

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    idx, index = _subject_indexer()
    try:
        source = resolve_image_path(image_path)
        found = read_bucket(parse_image_path(source_path_of(source))[0], bucket)
        (one,), _metrics = score_bucket([source], found, iou_threshold=iou_threshold,
                                        conf_threshold=conf_threshold, trait=None)
    except (UnreadableLabelDocumentError, ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}
    preds_all = one.preds or []
    gt_dicts = [_box_dict(a, index) for a in _boxable(one.gt)]
    pred_dicts = [_box_dict(a, index, scope=found.scope) for a in _boxable(preds_all)]
    # The matching indexes the whole documents; the renderer draws their boxable entries.
    gpos = {i: k for k, i in enumerate(
        i for i, a in enumerate(one.gt) if box_derivable(a.geometry))}
    ppos = {i: k for k, i in enumerate(
        i for i, a in enumerate(preds_all) if box_derivable(a.geometry))}
    tp_matches = [(gpos[g], ppos[p]) for g, p in one.matching.pairs]
    tp, fp, fn = one.matching.counts

    read = _display_of(one.header)
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


def get_worst_predictions(bucket: Bucket, top_k: int = 8) -> dict:
    """Return the ``top_k`` images ranked worst by a count-mismatch + low-confidence triage
    heuristic.

    This is a cheap triage signal, not a quality metric: it does no IoU matching and computes
    no loss. The score is ``2·|n_gt−n_pred as a shortfall| + |surplus| + (1−avg_conf)``, purely
    the difference in box *counts* plus mean confidence, so an image with the right count but
    every box mislocated scores as good. Use it to surface likely-bad frames for a human to look
    at; for true TP/FP/FN ranking use ``score_predictions`` (``detail=True``, IoU-matched). Only
    the documents the bucket's record names are ranked, each against the label document of its
    image in the bucket's capture (answered as ``capture``); a labeled image of that capture it
    names none for was not predicted, and is listed under ``not_predicted`` rather than scored. A
    bucket recording no capture refuses (``ValueError``), a ranked image naming no logical image
    (``FileNotFoundError``), and a ranked image with no label document
    (``UnreadableLabelDocumentError``).

    Args:
        bucket: The published bucket whose recorded documents are ranked.
        top_k: Number of worst images to return.
    """
    return _ranked(bucket, top_k)[0]


def _ranked(bucket: Bucket, top_k: int) -> tuple[dict, dict[str, tuple]]:
    """The ``top_k`` worst-scoring images of ``bucket`` by the count-mismatch and confidence
    heuristic (its capture, the ranked stems with their scores, how many were scored and the
    capture's labeled stems the bucket holds no document for), beside the read the ranking was
    made over: each stem's source, ground truth and predictions."""
    from tcip_annotation.json_io import detection_annotations

    from tcip_mcp.dataset_layout import capture_label_keys, image_dir
    from tcip_mcp.pipelines.image_utils import resolve_image_paths, source_path_of
    from tcip_mcp.pipelines.training.evaluation import bucket_reads

    capture = bucket.date
    if capture is None:
        raise ValueError(f"bucket {bucket.name!r} records no capture, so no label document "
                         "answers for its images.")
    read = bucket_reads(resolve_image_paths(
        image_dir(bucket.root, capture) / name for _stem, name in sorted(bucket.documents.items())),
        bucket)
    # Every image is one of the bucket's own documents, so each carries its predictions.
    by_stem = {Path(source_path_of(src)).stem: (src, gt, document.annotations)
               for src, gt, document in read}
    # Both sides counted as a count counts: objects with a box, a crowd region and a Point none.
    scores: list[tuple[str, float]] = []
    for stem, (_img, gt, predicted) in by_stem.items():
        preds = detection_annotations(predicted)
        gt_anns = detection_annotations(gt)

        n_pred = len(preds)
        n_gt = len(gt_anns)

        # Simple error heuristic: |pred - gt| + missed + extra + low confidence
        missed = max(0, n_gt - n_pred)
        extra = max(0, n_pred - n_gt)
        avg_conf = sum(map(prediction_score, preds)) / n_pred if n_pred else 0.0

        # Higher score = worse prediction
        error_score = missed * 2.0 + extra * 1.0 + (1.0 - avg_conf)
        scores.append((stem, error_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    worst = scores[:top_k]

    return {
        "capture": capture,
        "worst_images": [{"stem": s, "error_score": round(sc, 3)} for s, sc in worst],
        "total_evaluated": len(scores),
        "not_predicted": [key.parts[-1] for key in capture_label_keys(bucket.root, capture)
                          if key.parts[-1] not in bucket.documents],
    }, by_stem


def render_failure_cases(
    project: Path,
    dataset_root: str,
    bucket: str,
    task: str = "detect",
    top_k: int = 10,
    class_names: str = "",
) -> dict:
    """Find and render the worst predictions for failure analysis.

    Ranks by a count-mismatch + low-confidence heuristic (`get_worst_predictions`); no IoU
    matching, so an image with the right box count but every box mislocated scores as good. Not a
    substitute for `score_predictions`(`detail=True`)'s IoU-matched TP/FP/FN when mislocalization
    itself is the question. Each case renders its image of the bucket's capture against the label
    document of that image.

    Returns a grid image and individual failure case images.

    Args:
        dataset_root: The dataset root the bucket is published under.
        bucket: The published bucket whose documents are ranked and rendered.
        task: 'detect' or 'segment'.
        top_k: Number of worst cases to render.
        class_names: Comma-separated class names.
    """
    from tcip_mcp.buckets import read_bucket

    try:
        worst, by_stem = _ranked(read_bucket(located(dataset_root, project), bucket), top_k)
    except (UnreadableLabelDocumentError, ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}

    worst_items = worst.get("worst_images", [])
    if not worst_items:
        return {"summary": "No prediction errors found", "image_path": None}

    case_paths: list[str] = []
    titles: list[str] = []
    for item in worst_items:
        stem = item["stem"]
        source, gt, preds = by_stem[stem]
        read = _display_of(SourceHeader(source))

        idx, index = _subject_indexer()
        gt_dicts = [_box_dict(a, index) for a in _boxable(gt)]
        pred_dicts = [_box_dict(a, index) for a in _boxable(preds)]

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
        "summary": (
            f"Rendered {len(case_paths)} worst prediction cases "
            f"(of {worst['total_evaluated']} evaluated)"
        ),
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
    from tcip_mcp.dataset_layout import image_dir, image_root, label_key, list_dates
    from tcip_mcp.pipelines.image_utils import (
        BandGroupIncompleteError, list_logical_images, refuse_incomplete_band_group,
    )

    root = Path(folder_path)
    images_dir = image_root(root)
    if not images_dir.is_dir():
        return {"error": f"Images directory not found: {images_dir}"}

    # Every logical image of every capture, a grouped capture as one entry.
    all_images = [(capture, stem, src) for capture in list_dates(root)
                  for stem, src in sorted(list_logical_images(image_dir(root, capture)).items())]
    if not all_images:
        return {"error": "No images found in dataset"}

    sample = random.sample(all_images, min(n, len(all_images)))
    rendered_paths = []
    titles = []
    for capture, stem, enumerated in sample:
        try:
            source = refuse_incomplete_band_group(enumerated)
        except BandGroupIncompleteError:
            continue
        try:
            anns = _labels_at(label_key(root, capture, stem))
        except UnreadableLabelDocumentError as exc:
            return {"error": str(exc)}
        read = _display_of(SourceHeader(source))
        if anns is not None:
            idx, index = _subject_indexer()
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
    bound: BoundProject,
    refresh: bool = True,
    crop_to_viewport: bool = True,
    max_edge: int = 1600,
) -> dict:
    """Render exactly what the human's GUI canvas showed for this project: image, shapes, viewport.

    Reads the canvas state the GUI pushes into the project's store: the meta record (image,
    viewport, classes, counts, tab, mode, active_subject, cut_armed, dirty and user) and the
    geometry record (the full display-resolved geometry,
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

    project = bound.root
    meta_doc = canvas_meta_key(str(project))
    shapes_doc = canvas_geometry_key(str(project))

    previous = ts.read(meta_doc, default=None)
    since = None if previous is None else _received_at(previous, "canvas_live")
    refreshed = False
    ping: dict = {}
    if refresh:
        ping = post_panel_event(bound, "app", PANEL_EVENT_CANVAS_STATE_REQUEST, {})
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

    from tcip_mcp.pipelines.image_utils import resolve_image_path
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
    read = _display_of(SourceHeader(resolve_image_path(src_image)), max_edge=max_edge,
                       region=region)
    try:
        out = render_canvas_state(read.pixels, shapes,
                                  region=(read.rect.x0, read.rect.y0, read.rect.x1, read.rect.y1),
                                  output_path=viz_output_path(project, "canvas", suffix=".jpg"))
    except ValueError as exc:
        return {"error": f"the pushed canvas state for {src_image} will not render: {exc}"}
    ping_delivered = bool(ping.get("delivered"))

    now = datetime.now(timezone.utc)
    from collections import Counter

    tag_counts = dict(Counter(str(s.get("tag") or "untagged") for s in shapes))
    creator_counts = dict(Counter(str(s["created_by"]) for s in shapes if s.get("created_by")))

    age = round((now - _received_at(state, "canvas_live")).total_seconds(), 1)
    if refreshed or age < 5.0:
        summary = (
            f"Rendered the live {state.get('tab')} canvas for {state.get('image')} "
            f"({len(shapes)} shapes)."
        )
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
        image_path: The image file (:func:`~tcip_mcp.registry_paths.located` against
            ``project``).
        tile_size: Cell edge in native pixels; omitted derives a legible default.
        overlap: Cell overlap as a fraction of tile_size, training tiling's semantics.
    """
    from tcip_mcp.pipelines.image_utils import resolve_image_path
    from tcip_mcp.pipelines.reference_grid import (
        derive_pointing_tile_size,
        grid_geometry,
        reference_cells,
    )

    img = located(image_path, project)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    display = _display_of(SourceHeader(resolve_image_path(img)))
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
