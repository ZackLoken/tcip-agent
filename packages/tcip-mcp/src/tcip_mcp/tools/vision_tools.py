"""Vision tools: render annotations and predictions for visual analysis.

Each tool saves a rendered image to .tcip/artifacts/viz/ and returns the path so the agent
can call its client's own image-capable read tool on it to visually inspect it.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Callable, NamedTuple

import tcip_store as ts
from tcip_store.file_backend import RootedFileLocator

from tcip_annotation import Annotation, Point, Polygon, bbox_of, load_annotations_any
from tcip_annotation.json_io import UnreadableLabelDocument
from tcip_annotation.json_io import read_annotations as read_labels
from tcip_annotation.sam_wrapper import column_label, grid_to_rect
from tcip_annotation.viz import (
    render_candidates,
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
from tcip_mcp.project_paths import project_root
from tcip_mcp.server import mcp

if TYPE_CHECKING:
    import numpy as np

    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.raster_source import Rect


_PROPOSAL_DOC = RootedFileLocator(prefix=(".tcip", "state", "proposals"), suffix=".json")
"""The proposal envelope one dataset image's run stages, under the dataset's own state tree."""

PROPOSAL_STAGING_STORE = "proposal_staging"
ts.register_store(
    ts.StoreDescriptor(
        name=PROPOSAL_STAGING_STORE,
        kind="record",
        key_fields=("date", "stem"),
        frozen=True,
        codec=ts.RECORD_JSON,
        concurrency="last_writer_wins",
        locator=_PROPOSAL_DOC,
    )
)


def proposal_staging_key(dataset_root: str | Path, date: str | None, stem: str) -> ts.Key:
    """The proposals one run staged for one dataset image, for ``accept_proposals`` to read back.

    ``last_writer_wins``: a run writes the whole envelope from the candidates it just
    produced, so a re-run replaces the previous one rather than merging into it. Scoped to the
    dataset root, the same as the labels and predictions the proposals eventually become: a
    same-named image in another dataset, or another date bucket of this one, addresses its own
    record. ``date`` is the image's own capture-date bucket, or ``None`` for a flat dataset's
    undated layout, addressed under ``dataset_layout.UNDATED_BUCKET``: a store key holds no empty
    part, so the missing date needs the same declared token ``ingest_images`` buckets a dateless
    source under, rather than a spelling of its own. A flat-layout image and an image in that
    literal bucket therefore share one key for a given stem, which is never a real collision:
    ``ingest_images`` never produces a flat layout beside a dated one in the same dataset.
    """
    from tcip_mcp.dataset_layout import UNDATED_BUCKET

    return ts.Key(PROPOSAL_STAGING_STORE, str(dataset_root), (date or UNDATED_BUCKET, stem))


class StagingAddress(NamedTuple):
    """The :func:`proposal_staging_key` for a dataset image, plus the dataset root and date
    :func:`~tcip_mcp.dataset_layout.parse_image_path` derived to reach it.

    ``accept_proposals`` needs the root and date too, to stage the accepted predictions at the
    same address; carrying them here means that address is derived once, not twice.
    """

    key: ts.Key
    root: Path
    date: str | None


def _staging_key_for(image_path: str) -> StagingAddress:
    """The :class:`StagingAddress` for the dataset image at ``image_path``.

    Runs :func:`~tcip_mcp.dataset_layout.parse_image_path` once, so ``propose_annotations`` and
    ``accept_proposals`` never derive two different addresses for the same image. Raises
    ``ValueError``, the resolver's own message, for a path outside any dataset's ``images/`` tree.
    """
    from tcip_mcp.dataset_layout import parse_image_path

    root, date, stem = parse_image_path(image_path)
    return StagingAddress(proposal_staging_key(root, date, stem), root, date)


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


def _unresolvable_staging_source(img: Path, exc: Exception) -> str:
    """A reason for ``propose_annotations`` to decline staging ``img``, when
    :func:`~tcip_mcp.pipelines.image_utils.resolve_image_source` raised ``exc`` for it (the same
    call ``accept_proposals`` will make on this path).

    A band-group member's own path (``capture_Red.tif`` when ``capture.bandgroup`` claims it)
    resolves to nothing: the resolver's own ``FileNotFoundError`` for it reads the same as one for
    a stem that names no image at all, "no image for stem". This names the manifest that claims
    it instead, so the refusal points at the path to propose on rather than repeating a generic
    not-found. ``BandGroupIncomplete`` (a manifest that resolves but is missing a sibling) already
    carries its own manifest-naming message and is returned unchanged.
    """
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.image_utils import BandGroupIncomplete, list_logical_images

    if isinstance(exc, BandGroupIncomplete):
        return str(exc)
    for source in list_logical_images(img.parent).values():
        if isinstance(source, BandGroupRef) and img in source.bands.values():
            return (f"{img} is one band of the group {source.manifest_path.name!r}; propose "
                    f"on {source.manifest_path} instead.")
    return str(exc)


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


def _box_dict(a: Annotation, index: Callable[[str], int]) -> dict:
    b = bbox_of(a.geometry)
    d = {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "class_id": index(a.subject)}
    if a.score is not None:
        d["confidence"] = a.score
    return d


def _poly_dict(a: Annotation, index: Callable[[str], int]) -> dict:
    return {"rings": [[[p[0], p[1]] for p in ring] for ring in a.geometry.rings],
            "class_id": index(a.subject)}


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
    """Render model predictions on a single image. See ``visualize``."""
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    stem = img.stem
    from tcip_mcp.dataset_layout import find_prediction

    pred_file = find_prediction(image_path)
    if pred_file is None:
        return {"error": f"No predictions found for {stem}"}

    try:
        preds = read_labels(str(pred_file))
    except UnreadableLabelDocument as exc:
        return {"error": str(exc)}
    if conf_threshold > 0:
        preds = [a for a in preds if (a.score is None or a.score >= conf_threshold)]
    idx, index = _subject_indexer()

    n_points = _n_points(preds)
    read = _display_for_path(image_path)
    if task == "detect":
        shapes = _boxable(preds)
        out = render_detections(read.pixels, [_box_dict(a, index) for a in shapes],
                                native_size=read.native_size, class_names=_name_map(idx))
        summary = f"Rendered {len(shapes)} predictions on {img.name}" + _point_note(n_points)
    else:
        shapes = [a for a in preds if isinstance(a.geometry, Polygon)]
        out = render_segmentations(read.pixels, [_poly_dict(a, index) for a in shapes],
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

    Green = ground truth, Red = predictions, Yellow lines = matched pairs.
    """
    from tcip_annotation.format_io import detect_format
    from tcip_annotation.matching import compute_matches
    from tcip_mcp.dataset_layout import find_gt_label, find_prediction

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
        except UnreadableLabelDocument as exc:
            return {"error": str(exc)}
        pred_dicts = [_box_dict(a, index) for a in preds]
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
        labels_path = Path(labels_dir)
        candidate = labels_path.parent.parent / "images"
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
    from tcip_mcp.dataset_layout import find_gt_label
    from tcip_mcp.pipelines.image_utils import (
        BandGroupIncomplete, BandGroupRef, list_logical_images, resolve_image_source,
    )

    root = Path(folder_path)
    images_dir = root / "images"
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


def _region_rect_from_cells(cells: list, names: list[str]) -> "Rect":
    """The bounding rect, in the grid's native-pixel frame, of the named reference-grid cells.

    A region-scoped proposal pass needs one rectangle to crop, not the point prompts
    ``segment_prompt`` turns grid cells into: each name resolves through the one cell lookup
    (``sam_wrapper.grid_to_rect``, so a malformed or out-of-grid name is refused here exactly as it
    is for a point prompt) and the matched cells union to their combined bounding box.
    """
    from tcip_mcp.pipelines.raster_source import Rect

    matched = [grid_to_rect(name, cells) for name in names]
    return Rect(int(min(r[0] for r in matched)), int(min(r[1] for r in matched)),
                int(max(r[2] for r in matched)), int(max(r[3] for r in matched)))


def _write_region_crop(pixels: "np.ndarray") -> Path:
    """Save an RGB region crop to a fresh temp PNG file; the caller deletes it once the engine has
    read it.

    The crop is taken from the raster layer's already-``auto_orient_image``'d frame (a photographic
    source is EXIF-oriented on decode), so it carries no EXIF orientation tag once saved (a PIL
    ``.save()`` never re-emits an orientation tag it didn't read from a source file):
    ``auto_mask``'s own internal re-orientation call is a no-op against this file, not a second,
    wrong rotation of pixels that are already upright.
    """
    import os
    import tempfile

    from PIL import Image

    fd, tmp = tempfile.mkstemp(suffix=".png", prefix="tcip_propose_crop_")
    os.close(fd)
    Image.fromarray(pixels, mode="RGB").save(tmp)
    return Path(tmp)


def _offset_candidates(candidates: list[dict], origin: tuple[float, float]) -> list[dict]:
    """Candidates proposed against a region crop's own pixels, translated into the source image's
    full-frame native coordinates by the crop's own origin.

    Both consumers downstream (``render_candidates``, and ``accept_proposals`` reading the cached
    envelope back later) expect ``bbox``/``rings`` in the source image's native frame, never
    crop-local pixels, so this runs before either sees the candidates.
    """
    ox, oy = origin
    shifted = []
    for c in candidates:
        c = dict(c)
        x1, y1, x2, y2 = c["bbox"]
        c["bbox"] = [x1 + ox, y1 + oy, x2 + ox, y2 + oy]
        c["rings"] = [[(x + ox, y + oy) for x, y in ring] for ring in c["rings"]]
        shifted.append(c)
    return shifted


@mcp.tool()
@audited(scope_arg="image_path")
def propose_annotations(
    image_path: str,
    engine: str = "sam",
    engine_params: dict | None = None,
    grid_cells: list[str] | None = None,
    tile_size: int | None = None,
    overlap: float = 0.0,
) -> dict:
    """Propose candidate annotations on an image for review, using a chosen auto-labeling engine.

    Runs the engine's whole-image proposal pass, renders the numbered candidates, and returns the
    render path and neutral candidate data. Read the render with your own image-capable read
    tool, then call accept_proposals to assign classes and stage the accepted ones as predictions.

    Each candidate renders as a colored, semi-transparent filled polygon (every ring of an
    occlusion-split candidate drawn, not just the largest) with a large numbered label at its
    centroid, colors cycling through the shared class palette; the candidate id in that number is
    the same id ``accept_proposals``' ``assignments`` parameter names.

    On an image under a dataset's ``images/`` tree, the candidates are staged keyed by the
    dataset, capture date and stem, alongside the content identity of the pixels the engine ran
    on: ``accept_proposals`` reads the record back by that same address and refuses if the
    image's content no longer matches it. On a path outside any dataset's ``images/`` tree, or a
    dataset path ``accept_proposals`` would itself fail to resolve (a band-group member's own
    path when its manifest claims it), the engine still runs and the render and candidates are
    returned the same way, but nothing is staged (the response's ``staged`` is ``false``, naming
    why): there is no address ``accept_proposals`` could ever read the record back by, so such a
    call cannot later be accepted.

    The engine is a capability, not a fixed method: 'sam' is the built-in SAM2 reference; the agent
    can register another engine (``register_proposal_engine``) or pass a dotted 'module:factory' it
    wrote, then trial and compare engines by how well each one's high-conf proposals survive breeder
    review, and pick the most useful for the task.

    ``grid_cells`` restricts the pass to a region instead of the whole frame: name the reference-
    grid cells the region spans (e.g. ``['B3', 'C3', 'B4', 'C4']``, the same grid
    ``overlay_reference_grid``/``segment_prompt`` use), and the engine proposes only over their
    bounding rect. Useful on a large or crowded frame where a whole-image pass returns too many or
    too coarse candidates to review, or where only part of the frame matters right now. The crop is
    taken and the results offset back to full-frame coordinates entirely on this side of the engine
    seam: the engine is handed an ordinary (if smaller) image and never told a region was involved,
    so a bespoke engine gets region support with no code of its own. The one real caveat: an engine
    that keys behavior off the image path itself (a cache, a sidecar lookup keyed by the original
    file) receives the temp crop's path, which it cannot resolve back to the source image. Omitting
    ``grid_cells`` runs the whole frame, unchanged.

    Args:
        image_path: Absolute path to the image file.
        engine: Proposal engine: 'sam' (built-in) or a dotted 'module:factory' the agent brings.
        engine_params: Engine-specific knobs forwarded to the engine (e.g. SAM's model_type,
            points_per_side, pred_iou_thresh, stability_score_thresh, min_mask_region_area). Omit for
            the engine's own defaults.
        grid_cells: Reference-grid cell names bounding the region to propose over (e.g.
            ['B3', 'D5']); the engine sees the bounding rect of the named cells, not the whole
            frame. Requires ``tile_size``. Omit for the whole frame.
        tile_size: Cell edge, in native pixels, of the grid the cells were read off. Required with
            ``grid_cells``.
        overlap: Overlap fraction of the grid the cells were read off, ``segment_prompt``'s same
            semantics.
    """
    from tcip_mcp.pipelines.proposal import resolve_proposer

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    try:
        proposer = resolve_proposer(engine)
    except (ValueError, ImportError) as e:
        return {"error": str(e)}

    # A region is cropped and offset entirely here, before the engine ever sees an image path.
    # grid_cells=None skips this branch, taking the exact whole-frame path below, unchanged.
    propose_path = image_path
    crop_tmp: Path | None = None
    origin = (0.0, 0.0)
    region_info: dict | None = None
    if grid_cells is not None:
        if not grid_cells:
            return {"error": "grid_cells is empty; name at least one cell to scope the region."}
        if tile_size is None:
            return {"error": "grid_cells requires tile_size, the cell edge of the grid the "
                             "cells were read off (overlay_reference_grid echoes it back, with "
                             "overlap). Without it a cell name resolves against a grid nobody "
                             "rendered."}
        from tcip_mcp.pipelines.image_utils import image_dimensions
        from tcip_mcp.pipelines.raster_source import open_raster
        from tcip_mcp.pipelines.reference_grid import reference_cells

        # One resolution of the source for both halves: the frame the cells are laid over is the
        # frame the crop is read from, so they can never come from two decisions.
        source = _source_for_path(image_path)
        try:
            w, h = image_dimensions(source)
            cells = reference_cells(w, h, tile_size, overlap, clamp=True)
            rect = _region_rect_from_cells(cells, grid_cells)
            with open_raster(source, 3) as src:
                pixels, _spec = src.read_region(rect)
        except ValueError as e:
            return {"error": str(e)}
        if pixels.dtype != "uint8" or pixels.shape[-1] != 3:
            return {"error": "A region crop is handed to the engine as an RGB image, and "
                             f"{img.name} reads as {pixels.shape[-1]} band(s) of {pixels.dtype}. "
                             "Propose over the whole frame instead, or bring an engine that reads "
                             "this source itself."}
        crop_tmp = _write_region_crop(pixels)
        propose_path = str(crop_tmp)
        origin = (float(rect.x0), float(rect.y0))
        region_info = {"grid_cells": list(grid_cells), "tile_size": tile_size, "overlap": overlap,
                       "rect": [rect.x0, rect.y0, rect.x1, rect.y1]}

    try:
        try:
            candidates = proposer.propose(propose_path, **(engine_params or {}))
        except ImportError as e:
            return {"error": str(e)}
        except FileNotFoundError as e:
            return {"error": str(e)}
    finally:
        if crop_tmp is not None:
            crop_tmp.unlink(missing_ok=True)

    if region_info is not None:
        candidates = _offset_candidates(candidates, origin)

    if not candidates:
        # A prior run's record must not outlive this one finding nothing to propose.
        try:
            stale = _staging_key_for(image_path)
        except ValueError:
            pass
        else:
            ts.delete(stale.key)
        return {
            "image_path": None,
            "engine": engine,
            "summary": f"Engine {engine!r} proposed no candidates",
            "staged": False,
            "candidates": [],
        }

    # A bespoke engine's candidates are its own dicts, so what a segmenter returns natively
    # (an array, a numpy scalar) is named here rather than stored as a repr of itself.
    try:
        ts.check_json_value(candidates, path="candidates")
    except (TypeError, ValueError) as exc:
        return {"error": f"Engine {engine!r} proposed a candidate the store cannot hold: {exc}"}

    read = _display_for_path(image_path)
    out = render_candidates(read.pixels, candidates, native_size=read.native_size)

    # The envelope records the engine so accept_proposals stamps the right producer.
    envelope: dict = {"engine": engine, "candidates": candidates}
    if region_info is not None:
        envelope["region"] = region_info

    try:
        address = _staging_key_for(image_path)
    except ValueError as exc:
        staged = False
        stage_note = f" Not staged: {exc}"
    else:
        from tcip_mcp.pipelines import image_utils

        try:
            # The same resolution accept_proposals will make on this path: staging over a
            # source accept could never reread would leave a record it can never confirm.
            source = image_utils.resolve_image_source(img.parent, img.stem)
        except (FileNotFoundError, image_utils.BandGroupIncomplete) as exc:
            staged = False
            stage_note = f" Not staged: {_unresolvable_staging_source(img, exc)}"
        else:
            import dataclasses

            from tcip_mcp.pipelines.raster_source import content_identity

            identity = content_identity(source)
            envelope["image_identity"] = dataclasses.asdict(identity)
            envelope["image_path"] = str(img.resolve())
            ts.replace(address.key, envelope)
            staged = True
            stage_note = ""

    region_note = f" (region {grid_cells})" if region_info is not None else ""
    return {
        "image_path": out,
        "engine": engine,
        "summary": f"Engine {engine!r} proposed {len(candidates)} candidates{region_note}."
                   f"{stage_note} Review the numbered overlay, then call accept_proposals "
                   f"with class assignments.",
        "candidate_count": len(candidates),
        "staged": staged,
        "candidates": [
            {
                "id": c["candidate_id"],
                "area": c["area"],
                "score": round(c["score"], 3),
                "bbox": [round(v, 1) for v in c["bbox"]],
            }
            for c in candidates
        ],
    }


@mcp.tool()
@audited(scope_arg="image_path")
def accept_proposals(
    image_path: str,
    assignments: list[dict],
) -> dict:
    """Assign classes to reviewed proposals and stage them as predictions for canvas review.

    After reviewing propose_annotations output, the agent calls this tool with a mapping from
    candidate IDs to class IDs. Rejected candidates are simply omitted from the assignments list.
    The masks are written to the predictions tree (``predictions/<engine>/<date>/<task>``) as
    per-image COCO/JSON with ``created_by=<engine>`` and ``score`` = the engine's proposal score;
    they are model output, so a human accepts them on the Review canvas before they become ground
    truth. Staging goes through the prediction-bucket verdict guard, so a re-run never overwrites
    reviewed predictions or orphans their verdicts. This never writes GT.

    Reads back the record propose_annotations staged for this exact image (dataset, capture date
    and stem) and refuses if the image's content no longer matches the content identity that run
    recorded: the proposals it staged were candidates over those pixels, not whatever now sits at
    this path. That check decodes sample windows of the image (the bound ``CONTENT_IDENTITY_*``
    constants in ``raster_source.py`` set how many and how large), never the whole frame.

    Args:
        image_path: Absolute path to the image (same as propose_annotations).
        assignments: List of dicts, each with 'candidate_id' (int) and 'subject' (name).
            Only listed candidates are staged.
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    try:
        address = _staging_key_for(image_path)
    except ValueError as exc:
        return {"error": str(exc)}

    from tcip_mcp.pipelines.image_utils import (
        BandGroupIncomplete, image_dimensions, resolve_image_source,
    )

    try:
        source = resolve_image_source(img.parent, img.stem)
    except (FileNotFoundError, BandGroupIncomplete) as exc:
        return {"error": str(exc)}

    # Load cached proposals from the same record propose_annotations staged them in.
    envelope = ts.read(address.key, default=None)
    if envelope is None:
        return {"error": f"No proposals found for {img.stem}. Run propose_annotations first."}

    from tcip_mcp.pipelines.raster_source import raster_identity_matches

    try:
        matches = raster_identity_matches(envelope["image_identity"], source)
    except ValueError as exc:
        return {"error": f"Could not verify {image_path} against its staged proposals: {exc}"}

    if not matches:
        return {"error": f"{image_path} does not match the image propose_annotations ran on: "
                          "its content has changed since that run staged these candidates. "
                          "Run propose_annotations again on the current image."}

    engine = envelope.get("engine", "unknown")
    candidates = envelope.get("candidates", [])
    cand_map = {c["candidate_id"]: c for c in candidates}

    w, h = image_dimensions(source)

    # Build name-based predictions from accepted candidates (created_by=<engine>, score = the
    # engine's proposal score). Each candidate becomes one Annotation under its subject carrying
    # every ring the engine proposed: an occlusion-split object stays split rather than being
    # accepted as its largest fragment.
    from datetime import datetime, timezone
    from tcip_annotation.state import Polygon as _Polygon
    staged_at = datetime.now(timezone.utc).isoformat()
    proposals: list[Annotation] = []
    n_poly = 0

    for assign in assignments:
        cid = assign["candidate_id"]
        subject = assign.get("subject")
        cand = cand_map.get(cid)
        if cand is None or not subject:
            continue
        score = float(cand.get("score", 0.0))  # neutral proposal score, in [0, 1]
        rings = [[(float(x), float(y)) for x, y in ring]
                 for ring in cand["rings"] if len(ring) >= 3]
        if rings:
            proposals.append(Annotation(
                subject=str(subject), geometry=_Polygon(rings=rings),
                score=score, created_by=engine, created_at=staged_at))
            n_poly += 1

    # Stage into the predictions tree through the shared verdict-guarded helper: model output for a
    # human to accept on the Review canvas, never written straight to ground truth.
    from tcip_mcp.prediction_buckets import BucketHasVerdicts, stage_prediction_shapes

    try:
        staged = stage_prediction_shapes(
            str(address.root), engine, address.date, img.stem,
            annotations=proposals, img_w=w, img_h=h, overwrite=False,
        )
    except BucketHasVerdicts as exc:
        return {"error": str(exc), "verdict_count": exc.count, "suggested_bucket": exc.suggested}
    except ValueError as exc:
        return {"error": str(exc)}
    bucket = staged["bucket"]

    # Render final result for QA
    idx, index = _subject_indexer()
    read = _display_for_path(image_path)
    out = render_detections(read.pixels, [_box_dict(a, index) for a in proposals],
                            native_size=read.native_size, class_names=_name_map(idx))

    note = (f"Staged {n_poly} proposal(s) from {len(assignments)} {engine!r} candidates as "
            f"predictions (created_by={engine!r}) for review, not ground truth.")
    if staged["redirected"]:
        note = (f"bucket {engine!r} has {staged['verdict_count']} review verdict(s), staged to a fresh "
                f"bucket {bucket!r} instead so the reviewed predictions stay intact. " + note)

    return {
        "image_path": out,
        "engine": engine,
        "bucket": bucket,
        "bucket_redirected": staged["redirected"],
        "summary": note,
        "proposal_count": n_poly,
    }


@mcp.tool()
@audited
def capture_live_canvas(
    refresh: bool = True,
    crop_to_viewport: bool = True,
    max_edge: int = 1600,
) -> dict:
    """Render exactly what the human's GUI canvas shows right now: image, shapes, viewport.

    The GUI continuously pushes its canvas state under ``.tcip/state/``, split into two documents
    so the cadences never contend: ``canvas_live.json``, a meta document written on every push
    (image, viewport, classes, legend, counts, tab, mode, active_subject, dirty and user), and
    ``canvas_shapes.json``, the full display-resolved geometry (the shapes with the exact
    colors/tags the canvas renders, including unsaved edits and an in-progress drawing), written
    only when the shapes themselves change. A heartbeat (meta only, no geometry write) fires on
    view and meta changes, and as the downgrade while a pointer interaction is live, with the
    full geometry pushed once on release rather than per tick. This tool reads the region being
    shown at up to ``max_edge``, renders that state over it, and returns the artifact path for
    the agent's own image-capable read tool, plus the classes schema, review legend,
    per-tag/per-creator counts, and the state's age.

    Args:
        refresh: Ping the GUI (via the panel-event hub) to push fresh state first, waiting
            briefly for it to land. Falls back to the last pushed state if no GUI responds.
        crop_to_viewport: Render only the region the human currently sees (their zoom/pan).
            Pass False for the full frame with the same overlays.
        max_edge: Downscale the rendered output to at most this edge (px).
    """
    import time as _time

    from tcip_mcp.web_client import canvas_geometry_key, canvas_meta_key

    root = str(project_root())
    meta_doc = canvas_meta_key(root)
    shapes_doc = canvas_geometry_key(root)

    def _read(key: ts.Key) -> dict | None:
        try:
            return ts.read(key, default=None)
        except (OSError, ts.DecodeError):
            return None

    prev = _read(meta_doc)
    prev_ts = (prev or {}).get("received_at", 0)
    refreshed = False
    ping_delivered = False
    if refresh:
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
        return {"error": "No live canvas state found; is the GUI open with a project loaded? "
                         "The frontend pushes its canvas state to the project the GUI has open "
                         f"(looked under {root}; if the GUI has a different project open, "
                         "set_active_project to it first)."}

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
    # Read exactly the region being rendered: the human's viewport is a rectangle in the image's
    # own grid, so a raster far too large to decode whole is still capturable.
    region = None
    if crop_to_viewport and viewport and viewport.get("w") and viewport.get("h"):
        region = (float(viewport.get("x", 0)), float(viewport.get("y", 0)),
                  float(viewport["w"]), float(viewport["h"]))
    read = _display_for_path(src_image, max_edge=max_edge, region=region)
    out = render_canvas_state(read.pixels, shapes,
                              origin=(read.rect.x0, read.rect.y0), scale=read.scale)

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
        f"Rendered the last known {state.get('tab')} canvas for {state.get('image')} "
        f"({len(shapes)} shapes, {age}s old; the GUI did not answer the refresh ping; it may be "
        "closed, on another tab, or on a different project)."
    ) + " Read image_path with your own image-capable read tool to see it."
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


@mcp.tool()
@audited
def overlay_reference_grid(
    image_path: str,
    tile_size: int | None = None,
    overlap: float = 0.0,
) -> dict:
    """Render image with a labeled reference-grid overlay for spatial referencing.

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
