"""Vision tools: render annotations and predictions to an image and return its path.

The image is written under the project's ``.tcip/artifacts/viz/``.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

import tcip_store as ts
from tcip_annotation import Annotation, Point, bbox_of
from tcip_annotation.json_io import (
    UnreadableLabelDocumentError,
    read_document_versioned,
    read_predictions,
)
from tcip_annotation.matching import REVIEW_CONF_FLOOR
from tcip_annotation.state import box_derivable, polygonal, prediction_score
from tcip_annotation.viz import (
    render_canvas_state,
    render_comparison,
    render_detections,
    render_grid,
    render_grid_overlay,
    render_segmentations,
)
from tcip_mcp.audit import now
from tcip_mcp.pipelines.display_bounds import VIZ_ARTIFACT_MAX_EDGE
from tcip_mcp.pipelines.raster_source import SourceHeader
from tcip_mcp.project_paths import viz_output_path
from tcip_mcp.registry_paths import located
from tcip_mcp.server import tool
from tcip_mcp.workspace import BoundProject

if TYPE_CHECKING:
    import numpy as np

    from tcip_mcp.buckets import Bucket
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.raster_source import RasterSource, Rect


class DisplayRead(NamedTuple):
    """Display pixels, with the frame facts that place annotations on them.

    ``pixels`` is uint8 RGB; ``rect`` is the region of the raster they were read from, so the
    pixels' own size against it is the served resolution on each axis; ``native_size`` is the
    raster's own ``(width, height)``, the frame annotation coordinates are measured in.
    """

    pixels: np.ndarray
    rect: Rect
    native_size: tuple[int, int]


def _clamped_rect(
    region: tuple[float, float, float, float], width: int, height: int
) -> Rect:
    """Clamp an ``(x, y, w, h)`` region to a non-empty rect in a ``width`` x ``height`` raster."""
    from tcip_mcp.pipelines.raster_source import Rect

    def clamp(v: float, low: int, high: int) -> int:
        return max(low, min(int(v), high))

    x0 = clamp(region[0], 0, width - 1)
    y0 = clamp(region[1], 0, height - 1)
    return Rect(
        x0,
        y0,
        clamp(x0 + region[2], x0 + 1, width),
        clamp(y0 + region[3], y0 + 1, height),
    )


def _read_for_display(
    raster: RasterSource,
    *,
    plain: bool,
    max_edge: int = VIZ_ARTIFACT_MAX_EDGE,
    region: tuple[float, float, float, float] | None = None,
) -> DisplayRead:
    """Read the open ``raster`` as display pixels.

    The raster layer serves the region (an ``(x, y, w, h)`` rectangle in the raster's own grid, or
    the whole frame) at or under ``max_edge`` through ``display_bounds.plan_read``, off the
    overview level the plan names, and nothing is ever materialized to a temp file.

    A ``plain`` raster, one the image route shows as its own pixels
    (``raster_source.displays_plain`` of its header), keeps them (grayscale repeated, alpha
    dropped) with no stretch; every other one has its default bands
    (``raster_source.display_bands``) composited and independently stretched by
    ``band_stats.DEFAULT_STRETCH``, through ``composite_display_rgb``.
    """
    from tcip_mcp.pipelines.band_stats import (
        DEFAULT_STRETCH,
        composite_display_rgb,
    )
    from tcip_mcp.pipelines.display_bounds import plan_read, planned_read
    from tcip_mcp.pipelines.raster_source import Rect, display_bands

    native = (int(raster.width), int(raster.height))
    rect = (
        Rect(0, 0, raster.width, raster.height)
        if region is None
        else _clamped_rect(region, raster.width, raster.height)
    )
    plan = plan_read(
        rect,
        raster.width,
        raster.height,
        raster.level_dims(),
        rect.width * rect.height,
        max_edge,
    )
    pixels, _spec = planned_read(raster, rect, plan)
    return DisplayRead(
        composite_display_rgb(
            pixels,
            display_bands(raster.num_channels),
            "none" if plain else DEFAULT_STRETCH,
        ),
        rect,
        native,
    )


def _display_of(
    header: SourceHeader,
    *,
    max_edge: int = VIZ_ARTIFACT_MAX_EDGE,
    region: tuple[float, float, float, float] | None = None,
) -> DisplayRead:
    """Read the source ``header`` describes for display, opened once at the image route's count.

    Returns :func:`_read_for_display` of the source, opened through
    :meth:`~tcip_mcp.pipelines.raster_source.SourceHeader.open_at_route_count`.
    """
    from tcip_mcp.pipelines.raster_source import displays_plain

    with header.open_at_route_count() as raster:
        return _read_for_display(
            raster, plain=displays_plain(header), max_edge=max_edge, region=region
        )


def _subject_indexer() -> tuple[dict[str, int], Callable[[str], int]]:
    """Build an empty name-to-index map and the indexer that fills it.

    The indexer returns a name's index, giving each new name the next integer in first-seen order
    and recording it in the map.
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
    """Return the annotations whose geometry derives a box: geometry-bearing, a Point excluded."""
    return [a for a in anns if box_derivable(a.geometry)]


def _n_points(anns: list[Annotation]) -> int:
    return sum(1 for a in anns if isinstance(a.geometry, Point))


def _point_note(n: int) -> str:
    return (
        f" ({n} point annotation(s) not drawn, no point renderer yet)"
        if n
        else ""
    )


def _legend_name(a: Annotation, *, scope: ClassScope | None) -> str:
    """Return the name a render's legend shows for ``a``.

    The name is its subject followed by each value it carries under ``scope``'s attributes, in
    declared order (:func:`~tcip_annotation.json_io.attribute_ids` and its inverse), its subject
    alone under no scope, a scope naming no subject or declaring no attribute, or for a record of
    another subject.
    """
    from tcip_annotation.json_io import attribute_ids, attribute_values

    if scope is None or scope.subject is None:
        return a.subject
    read = scope.document_attributes("the rendered bucket's scope")
    ids = attribute_ids(a, read.subject, read.attributes)
    return " ".join(
        [
            a.subject,
            *(attribute_values(ids, read.attributes).values() if ids else ()),
        ]
    )


def _box_dict(
    a: Annotation,
    index: Callable[[str], int],
    *,
    scope: ClassScope | None = None,
) -> dict:
    geometry = a.geometry
    assert box_derivable(geometry), (
        "every caller passes a _boxable-filtered or box-rendered annotation"
    )
    b = bbox_of(geometry)
    d = {
        "x1": b.x1,
        "y1": b.y1,
        "x2": b.x2,
        "y2": b.y2,
        "class_id": index(_legend_name(a, scope=scope)),
    }
    if a.score is not None:
        d["score"] = a.score
    return d


def _poly_dict(
    a: Annotation,
    index: Callable[[str], int],
    *,
    scope: ClassScope | None = None,
) -> dict:
    geometry = a.geometry
    assert polygonal(geometry), (
        "called only for the segmentation task's own shapes"
    )
    return {
        "rings": [[[p[0], p[1]] for p in ring] for ring in geometry.rings],
        "class_id": index(_legend_name(a, scope=scope)),
    }


VizTask = Literal["detect", "segment"]
"""Whether a render draws boxes or segmentation masks."""


class AnnotationsRender(BaseModel):
    """Ground-truth labels drawn on one image, ``path`` the image file."""

    model_config = ConfigDict(extra="forbid")
    source: Literal["annotations"]
    path: str
    task: VizTask = "detect"


class PredictionsRender(BaseModel):
    """A published bucket's predictions drawn on one image, ``path`` the image file.

    ``bucket`` names the bucket, published under the image's dataset root, whose document for the
    image is drawn; ``conf_threshold`` filters the predictions drawn, the viewing filter
    ``matching.REVIEW_CONF_FLOOR`` unless stated.
    """

    model_config = ConfigDict(extra="forbid")
    source: Literal["predictions"]
    path: str
    bucket: str
    task: VizTask = "detect"
    conf_threshold: float = REVIEW_CONF_FLOOR


class ComparisonRender(BaseModel):
    """Ground truth (green) beside a bucket's predictions (red) on one image, with TP/FP/FN.

    ``path`` is the image file and ``bucket`` the published bucket compared; a match reaches
    ``iou_threshold`` (the comparability convention is ``schemas.COMPARABILITY_IOU``) over the
    predictions at or above ``conf_threshold``, the viewing filter unless stated.
    """

    model_config = ConfigDict(extra="forbid")
    source: Literal["comparison"]
    path: str
    bucket: str
    iou_threshold: float
    conf_threshold: float = REVIEW_CONF_FLOOR


class DatasetRender(BaseModel):
    """A grid of ``n`` random annotated samples, ``path`` the dataset root holding ``images/``."""

    model_config = ConfigDict(extra="forbid")
    source: Literal["dataset"]
    path: str
    task: VizTask = "detect"
    n: int = 16


VizRender = Annotated[
    AnnotationsRender | PredictionsRender | ComparisonRender | DatasetRender,
    Field(discriminator="source"),
]
"""One render :func:`visualize` makes, its ``source`` naming which and what it requires."""


@tool()
def visualize(project: Path, render: VizRender) -> dict:
    """Render annotations, predictions, a GT-vs-prediction comparison, or a sample grid.

    Saves under ``project``'s ``.tcip/artifacts/viz/`` and returns its path as ``image_path``.

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
        render: The render, by its ``source``: ``annotations``, ``predictions``, ``comparison``
            or ``dataset``, each with the fields it requires (``path`` located against the
            project).

    """
    path = str(located(render.path, project))
    match render:
        case AnnotationsRender():
            return _viz_annotations(project, path, task=render.task)
        case DatasetRender():
            return _viz_dataset_sample(
                project, path, n=render.n, task=render.task
            )
        case PredictionsRender():
            return _viz_predictions(
                project,
                path,
                render.bucket,
                task=render.task,
                conf_threshold=render.conf_threshold,
            )
        case ComparisonRender():
            return _viz_comparison(
                project,
                path,
                render.bucket,
                iou_threshold=render.iou_threshold,
                conf_threshold=render.conf_threshold,
            )


def _labels_at(key: ts.Key) -> list[Annotation] | None:
    """Return the annotations of the label document ``key``, ``None`` when there is none."""
    document, version = read_document_versioned(key)
    return None if version == ts.Version.ABSENT else document.annotations


def _viz_annotations(project: Path, image_path: str, *, task: VizTask) -> dict:
    """Render ground-truth annotations on the logical image ``image_path`` names.

    The image is resolved by :func:`~tcip_mcp.pipelines.image_utils.resolve_image_path`, whose
    refusals answer as an error, and its annotations are read from its own label document.
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    from tcip_mcp.dataset_layout import label_key_of
    from tcip_mcp.pipelines.image_utils import (
        resolve_image_path,
        source_path_of,
    )

    try:
        source = resolve_image_path(image_path)
        anns = _labels_at(label_key_of(source_path_of(source)))
    except (
        UnreadableLabelDocumentError,
        ValueError,
        FileNotFoundError,
    ) as exc:
        return {"error": str(exc)}
    if anns is None:
        return {"error": f"No labels found for {img.stem}"}
    idx, index = _subject_indexer()

    n_points = _n_points(anns)
    read = _display_of(SourceHeader(source))
    if task == "detect":
        shapes = _boxable(anns)
        out = render_detections(
            read.pixels,
            [_box_dict(a, index) for a in shapes],
            native_size=read.native_size,
            class_names=_name_map(idx),
            output_path=viz_output_path(project, "detections"),
        )
        summary = f"Rendered {len(shapes)} detections on {img.name}"
        if shapes:
            from collections import Counter

            counts = Counter(a.subject for a in shapes)
            summary += ": " + ", ".join(
                f"{v} {k}" for k, v in counts.most_common()
            )
        summary += _point_note(n_points)
    else:
        shapes = [a for a in anns if polygonal(a.geometry)]
        out = render_segmentations(
            read.pixels,
            [_poly_dict(a, index) for a in shapes],
            native_size=read.native_size,
            class_names=_name_map(idx),
            output_path=viz_output_path(project, "segmentations"),
        )
        summary = (
            f"Rendered {len(shapes)} segmentation masks on {img.name}"
            + _point_note(n_points)
        )

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
    *,
    task: VizTask,
    conf_threshold: float,
) -> dict:
    """Render a bucket's predictions on the logical image ``image_path`` names.

    The image is resolved by :func:`~tcip_mcp.pipelines.image_utils.resolve_image_path`, whose
    refusals answer as an error, and its predictions are read from the bucket's document for it.

    The legend keys each detection by its subject and the values it carries under the attributes
    the bucket's recorded scope declares (:func:`_legend_name`).
    """
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.dataset_layout import parse_image_path
    from tcip_mcp.pipelines.image_utils import (
        resolve_image_path,
        source_path_of,
    )

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    try:
        source = resolve_image_path(image_path)
        root, _capture, stem = parse_image_path(source_path_of(source))
        found = read_bucket(root, bucket)
        pred_key, scope = found.document_key(stem), found.scope
        if pred_key is None:
            return {
                "error": f"No predictions found for {stem} in bucket {bucket!r}"
            }
        preds = read_predictions(pred_key).annotations
    except (
        UnreadableLabelDocumentError,
        ValueError,
        FileNotFoundError,
    ) as exc:
        return {"error": str(exc)}
    preds = [a for a in preds if prediction_score(a) >= conf_threshold]
    idx, index = _subject_indexer()

    n_points = _n_points(preds)
    read = _display_of(SourceHeader(source))
    if task == "detect":
        shapes = _boxable(preds)
        out = render_detections(
            read.pixels,
            [_box_dict(a, index, scope=scope) for a in shapes],
            native_size=read.native_size,
            class_names=_name_map(idx),
            output_path=viz_output_path(project, "detections"),
        )
        summary = (
            f"Rendered {len(shapes)} predictions on {img.name}"
            + _point_note(n_points)
        )
    else:
        shapes = [a for a in preds if polygonal(a.geometry)]
        out = render_segmentations(
            read.pixels,
            [_poly_dict(a, index, scope=scope) for a in shapes],
            native_size=read.native_size,
            class_names=_name_map(idx),
            output_path=viz_output_path(project, "segmentations"),
        )
        summary = (
            f"Rendered {len(shapes)} prediction masks on {img.name}"
            + _point_note(n_points)
        )

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
    *,
    iou_threshold: float,
    conf_threshold: float,
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
    from tcip_mcp.pipelines.image_utils import (
        resolve_image_path,
        source_path_of,
    )
    from tcip_mcp.pipelines.training.evaluation import score_bucket

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    idx, index = _subject_indexer()
    try:
        source = resolve_image_path(image_path)
        found = read_bucket(
            parse_image_path(source_path_of(source))[0], bucket
        )
        header = SourceHeader(source)
        # The render's decode states the frame the scoring reads, so the file opens once.
        read = _display_of(header)
        (one,), _metrics = score_bucket(
            [header],
            found,
            iou_threshold=iou_threshold,
            conf_threshold=conf_threshold,
            trait=None,
        )
    except (
        UnreadableLabelDocumentError,
        ValueError,
        FileNotFoundError,
    ) as exc:
        return {"error": str(exc)}
    preds_all = one.preds or []
    gt_dicts = [_box_dict(a, index) for a in _boxable(one.gt)]
    pred_dicts = [
        _box_dict(a, index, scope=found.scope) for a in _boxable(preds_all)
    ]
    # The matching indexes the whole documents; the renderer draws their boxable entries.
    gpos = {
        i: k
        for k, i in enumerate(
            i for i, a in enumerate(one.gt) if box_derivable(a.geometry)
        )
    }
    ppos = {
        i: k
        for k, i in enumerate(
            i for i, a in enumerate(preds_all) if box_derivable(a.geometry)
        )
    }
    tp_matches = [(gpos[g], ppos[p]) for g, p in one.matching.pairs]
    tp, fp, fn = one.matching.counts

    out = render_comparison(
        read.pixels,
        gt_dicts,
        pred_dicts,
        native_size=read.native_size,
        matches=tp_matches,
        class_names=_name_map(idx),
        output_path=viz_output_path(project, "comparison"),
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


def get_worst_predictions(
    bucket: Bucket,
    top_k: int,
    shortfall_weight: float,
    surplus_weight: float,
    confidence_weight: float,
) -> dict:
    """Return the ``top_k`` images ranked worst by a count-mismatch + low-confidence heuristic.

    This is a cheap triage signal, not a quality metric: it does no IoU matching and computes
    no loss. The score is ``shortfall_weight * max(0, n_gt - n_pred) + surplus_weight *
    max(0, n_pred - n_gt) + confidence_weight * (1 - avg_conf)``, purely the difference in box
    counts plus mean confidence, so an image with the right count but
    every box mislocated scores as good. For true TP/FP/FN ranking use ``score_predictions``
    (``detail=True``, IoU-matched). Only
    the documents the bucket's record names are ranked, each against the label document of its
    image in the bucket's capture (answered as ``capture``); a labeled image of that capture it
    names none for was not predicted, and is listed under ``not_predicted`` rather than scored. A
    bucket recording no capture refuses (``ValueError``), a ranked image naming no logical image
    (``FileNotFoundError``), and a ranked image with no label document
    (``UnreadableLabelDocumentError``).

    Args:
        bucket: The published bucket whose recorded documents are ranked.
        top_k: Number of worst images to return.
        shortfall_weight: Weight of each ground-truth object the predictions fall short of.
        surplus_weight: Weight of each prediction beyond the ground-truth count.
        confidence_weight: Weight of one minus the predictions' mean confidence.

    """
    return _ranked(
        bucket, top_k, (shortfall_weight, surplus_weight, confidence_weight)
    )[0]


def _ranked(
    bucket: Bucket, top_k: int, weights: tuple[float, float, float]
) -> tuple[dict, dict[str, tuple]]:
    """Rank the ``top_k`` worst-scoring images of ``bucket`` by the count-mismatch heuristic.

    The heuristic is the count-mismatch and confidence one under ``weights`` (shortfall, surplus,
    confidence; :func:`get_worst_predictions`). Returns the ranking (its capture, the ranked stems
    with their scores, how many were scored and the capture's labeled stems the bucket holds no
    document for), beside the read the ranking was made over: each stem's source, ground truth
    and predictions.
    """
    from tcip_annotation.json_io import detection_annotations
    from tcip_mcp.dataset_layout import capture_label_keys, image_dir
    from tcip_mcp.pipelines.image_utils import (
        resolve_image_paths,
        source_path_of,
    )
    from tcip_mcp.pipelines.training.evaluation import bucket_reads

    capture = bucket.date
    if capture is None:
        raise ValueError(
            f"bucket {bucket.name!r} records no capture, so no label document "
            "answers for its images."
        )
    read = bucket_reads(
        resolve_image_paths(
            image_dir(bucket.root, capture) / name
            for _stem, name in sorted(bucket.documents.items())
        ),
        bucket,
    )
    # Every image is one of the bucket's own documents, so each carries its predictions.
    by_stem = {
        Path(source_path_of(src)).stem: (src, gt, document.annotations)
        for src, gt, document in read
    }
    # Both sides counted as a count counts: objects with a box, a crowd region and a Point none.
    shortfall_weight, surplus_weight, confidence_weight = weights
    scores: list[tuple[str, float]] = []
    for stem, (_img, gt, predicted) in by_stem.items():
        preds = detection_annotations(predicted)
        gt_anns = detection_annotations(gt)

        n_pred = len(preds)
        n_gt = len(gt_anns)

        missed = max(0, n_gt - n_pred)
        extra = max(0, n_pred - n_gt)
        avg_conf = (
            sum(map(prediction_score, preds)) / n_pred if n_pred else 0.0
        )

        # Higher score = worse prediction
        error_score = (
            missed * shortfall_weight
            + extra * surplus_weight
            + (1.0 - avg_conf) * confidence_weight
        )
        scores.append((stem, error_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    worst = scores[:top_k]

    return {
        "capture": capture,
        "worst_images": [
            {"stem": s, "error_score": round(sc, 3)} for s, sc in worst
        ],
        "total_evaluated": len(scores),
        "not_predicted": [
            key.parts[-1]
            for key in capture_label_keys(bucket.root, capture)
            if key.parts[-1] not in bucket.documents
        ],
    }, by_stem


@tool()
def render_failure_cases(
    project: Path,
    dataset_root: str,
    bucket: str,
    top_k: int,
    shortfall_weight: float,
    surplus_weight: float,
    confidence_weight: float,
) -> dict:
    """Find and render the worst predictions for failure analysis.

    Ranks by a count-mismatch + low-confidence heuristic (`get_worst_predictions`); no IoU
    matching, so an image with the right box count but every box mislocated scores as good. Each
    case renders its image of the bucket's capture against the label
    document of that image.

    Returns a grid image and individual failure case images.

    Args:
        dataset_root: The dataset root the bucket is published under.
        bucket: The published bucket whose documents are ranked and rendered.
        top_k: Number of worst cases to render.
        shortfall_weight / surplus_weight / confidence_weight: The ranking's weights, as
            :func:`get_worst_predictions` takes them.

    """
    from tcip_mcp.buckets import read_bucket

    try:
        worst, by_stem = _ranked(
            read_bucket(located(dataset_root, project), bucket),
            top_k,
            (shortfall_weight, surplus_weight, confidence_weight),
        )
    except (
        UnreadableLabelDocumentError,
        ValueError,
        FileNotFoundError,
    ) as exc:
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
        render_comparison(
            read.pixels,
            gt_dicts,
            pred_dicts,
            native_size=read.native_size,
            class_names=_name_map(idx),
            output_path=out,
        )
        case_paths.append(out)
        titles.append(f"{stem} (err={item['error_score']:.1f})")

    # Render grid of all failure cases
    grid_path = None
    if case_paths:
        grid_path = render_grid(
            case_paths,
            titles=titles,
            cols=min(4, len(case_paths)),
            output_path=viz_output_path(project, "failures"),
        )

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
    project: Path, folder_path: str, *, n: int, task: VizTask
) -> dict:
    """Render a grid of random annotated dataset samples."""
    from tcip_mcp.dataset_layout import image_root, label_key
    from tcip_mcp.pipelines.image_utils import (
        BandGroupIncompleteError,
        list_logical_images,
        refuse_incomplete_band_group,
        scan_captures,
    )

    root = Path(folder_path)
    images_dir = image_root(root)
    if not images_dir.is_dir():
        return {"error": f"Images directory not found: {images_dir}"}

    all_images = [
        (capture, stem, src)
        for capture, logical in scan_captures(
            root, list_logical_images
        ).items()
        for stem, src in sorted(logical.items())
    ]
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
                out = render_detections(
                    read.pixels,
                    [_box_dict(a, index) for a in shapes],
                    native_size=read.native_size,
                    class_names=_name_map(idx),
                    output_path=viz_output_path(project, "detections"),
                )
            else:
                shapes = [a for a in anns if polygonal(a.geometry)]
                out = render_segmentations(
                    read.pixels,
                    [_poly_dict(a, index) for a in shapes],
                    native_size=read.native_size,
                    class_names=_name_map(idx),
                    output_path=viz_output_path(project, "segmentations"),
                )
            titles.append(f"{stem} ({len(shapes)})")
        else:
            # An unlabeled sample renders too, with nothing drawn on it: the grid tiles rendered
            # artifacts, so every cell has to be one.
            out = render_detections(
                read.pixels,
                [],
                native_size=read.native_size,
                output_path=viz_output_path(project, "detections"),
            )
            titles.append(f"{stem} (no labels)")

        rendered_paths.append(out)

    grid_path = render_grid(
        rendered_paths,
        titles=titles,
        cols=min(4, len(rendered_paths)),
        output_path=viz_output_path(project, "grid"),
    )

    return {
        "image_path": grid_path,
        "summary": f"Grid of {len(sample)} annotated samples from {root.name}",
        "count": len(sample),
        "total_images": len(all_images),
    }


@tool()
def capture_live_canvas(
    bound: BoundProject,
    refresh: bool = True,
    crop_to_viewport: bool = True,
    max_edge: int = 1600,
) -> dict:
    """Render this project's GUI canvas from its last pushed state: image, viewport and shapes.

    Reads the canvas state pushed into the project's store: the meta record (image, viewport,
    classes, counts, tab, mode, active_subject, cut_armed, dirty and user) and the geometry record
    (the full display-resolved geometry, including unsaved edits and an in-progress drawing).
    A heartbeat push updates the meta record alone, and geometry pushed for another image or tab
    is not drawn, so the render carries no shapes whenever ``shapes_missing`` is true.
    Renders the region being shown at up to ``max_edge`` and returns the artifact path, plus the
    meta record as stored (``canvas``, a :class:`~tcip_mcp.web_client.CanvasMeta`), per-tag
    counts, and the state's age. A canvas document that will
    not read raises the store's own error.

    Args:
        refresh: Ping the GUI (via the panel-event hub) to push fresh state first, waiting briefly
            for it to land. The backend accepts the ping only while it has this project open.
            Either way the last pushed state renders: ``refreshed``, ``refresh_answer`` (the
            ping's :class:`~tcip_mcp.web_client.PanelEventAnswer`, ``None`` with no ping) and
            ``state_age_seconds`` describe the meta record, and the summary calls it live
            only when a fresh meta push landed or the meta record is under five seconds old;
            whether shapes were pushed and how old they are is ``shapes_missing`` and
            ``shapes_age_seconds``.
        crop_to_viewport: Render only the region the human currently sees (their zoom/pan). Pass
            False for the full frame with the same overlays.
        max_edge: Downscale the rendered output to at most this edge (px).

    """
    import time as _time

    from tcip_mcp.web_client import (
        CanvasStateRequest,
        PanelEventAnswer,
        post_panel_event,
        read_canvas_geometry,
        read_canvas_meta,
    )

    project = bound.root
    previous = read_canvas_meta(str(project))
    refreshed = False
    ping: PanelEventAnswer | None = None
    if refresh:
        ping = post_panel_event(bound, "app", CanvasStateRequest())
        if ping.reason is None:
            for _ in range(12):  # ~2.4s for the GUI's flush to land
                _time.sleep(0.2)
                cur = read_canvas_meta(str(project))
                if cur is not None and (
                    previous is None or cur.received_at > previous.received_at
                ):
                    refreshed = True
                    break

    state = read_canvas_meta(str(project))
    if state is None:
        return {
            "error": "No canvas state has been pushed for this project; the GUI pushes its "
            "canvas only while it has this project open.",
            "refresh_answer": None if ping is None else ping.model_dump(),
        }

    from tcip_mcp.pipelines.image_utils import (
        logical_image_name,
        resolve_image_path,
    )
    from tcip_mcp.registry_paths import resolved_registry_path

    image = logical_image_name(state.image_path)
    src_image = str(resolved_registry_path(project, state.image_path))
    if not Path(src_image).is_file():
        return {
            "error": f"Canvas state references a missing image: {src_image}"
        }

    geometry = read_canvas_geometry(str(project))
    if geometry is not None and (
        geometry.image_path != state.image_path or geometry.tab != state.tab
    ):
        geometry = None
    shapes = [] if geometry is None else geometry.shapes
    # Only the rendered region is read, so a raster too large to decode whole is still capturable.
    viewport = state.viewport
    region = (
        (viewport.x, viewport.y, viewport.w, viewport.h)
        if crop_to_viewport and viewport is not None
        else None
    )
    read = _display_of(
        SourceHeader(resolve_image_path(src_image)),
        max_edge=max_edge,
        region=region,
    )
    try:
        out = render_canvas_state(
            read.pixels,
            shapes,
            region=(read.rect.x0, read.rect.y0, read.rect.x1, read.rect.y1),
            output_path=viz_output_path(project, "canvas", suffix=".jpg"),
        )
    except ValueError as exc:
        return {
            "error": f"the pushed canvas state for {src_image} will not render: {exc}"
        }
    from collections import Counter

    rendered = now()
    tag_counts = dict(Counter(s.tag for s in shapes))

    age = round((rendered - state.received_at).total_seconds(), 1)
    if refreshed or age < 5.0:
        summary = (
            f"Rendered the live {state.tab} canvas for {image} "
            f"({len(shapes)} shapes)."
        )
    else:
        summary = (
            f"Rendered the last known {state.tab} canvas for {image} "
            f"({len(shapes)} shapes, {age}s old; not live: no fresh push landed for this call, "
            "see refresh_answer)."
        )
    summary += (
        " Read image_path with your own image-capable read tool to see it."
    )
    return {
        "image_path": out,
        "source_image": src_image,
        "image": image,
        "canvas": state.model_dump(mode="json"),
        "cropped_to_viewport": region is not None,
        "shape_counts_by_tag": tag_counts,
        "state_age_seconds": age,
        "shapes_age_seconds": (
            None
            if geometry is None
            else round((rendered - geometry.received_at).total_seconds(), 1)
        ),
        "shapes_missing": geometry is None,
        "refreshed": refreshed,
        "refresh_answer": None if ping is None else ping.model_dump(),
        "summary": summary,
    }


@tool()
def overlay_reference_grid(
    project: Path,
    image_path: str,
    overlap: float,
    tile_size: int | None = None,
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
    ``width``, ``height``): a ``propose_annotations`` region carries the echoed
    ``tile_size``/``overlap`` beside its cells, so a cell name resolves against the grid that was
    actually rendered.

    Args:
        image_path: The image file (:func:`~tcip_mcp.registry_paths.located` against
            ``project``).
        overlap: Cell overlap as a fraction of tile_size, training tiling's semantics.
        tile_size: Cell edge in native pixels; omitted derives a legible default.

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
    out = render_grid_overlay(
        display.pixels,
        cells,
        native_size=(w, h),
        output_path=viz_output_path(project, "grid_overlay"),
    )

    geometry = grid_geometry(w, h, tile_size, overlap)
    return {
        "image_path": out,
        "summary": f"Reference grid ({geometry['cols']}x{geometry['rows']}, tile_size "
        f"{tile_size}) rendered on {img.name}. Reference cells like 'A1' "
        f"(top-left) to '{cells[-1].name}' (bottom-right).",
        **geometry,
    }
