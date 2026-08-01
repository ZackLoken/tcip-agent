"""Vision tools — render annotations and predictions for visual analysis.

Each tool saves a rendered image to .tcip/artifacts/viz/ and returns the
path so the agent can use view_image to visually inspect it.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

from tcip_annotation import Annotation, Point, Polygon, bbox_of, load_annotations_any
from tcip_annotation.json_io import read_annotations as read_labels
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


def _materialize_if_needed(source) -> str:
    """The real path to hand a (photographic-only) renderer for an already-resolved image source.

    A plain photographic file (jpg/png/heic/bmp) is returned unchanged. So is an ordinary
    photographic-shaped GeoTIFF (1/3/4 bands — a real, pre-existing supported format PIL decodes
    in true color, unchanged from before band-group support existed). Only a genuinely
    non-standard source — a ``.bandgroup``-grouped capture, or a raster whose band count PIL's own
    1/3/4-channel modes don't cover (npy/npz, or a >4-band GeoTIFF) — decodes through the
    channel-aware ``image_utils`` and is materialized to a throwaway 8-bit RGB preview (first three
    bands, independently min-max stretched; a single band is repeated across channels) under
    ``.tcip/artifacts/viz/_band_previews/``. Never a measurement artifact — visualization only.
    """
    from tcip_mcp.pipelines import image_utils
    from tcip_mcp.pipelines.derivations import probe_channels

    if isinstance(source, Path):
        ext = source.suffix.lower()
        if ext not in (".npy", ".npz", ".tif", ".tiff"):
            return str(source)
        if ext in (".tif", ".tiff") and probe_channels(source) in (1, 3, 4):
            # A photographic-shaped GeoTIFF: PIL decodes it directly, same as any other
            # supported format — no synthetic stretch for what is really an ordinary image.
            return str(source)

    n = probe_channels(source)
    arr = image_utils.load_multiband(source, n)
    return _band_preview_png(arr, source.stem)


def _band_preview_png(arr, stem: str) -> str:
    import numpy as np
    from PIL import Image as _Image

    n_bands = arr.shape[-1]
    idxs = [0, 1, 2] if n_bands >= 3 else [0, 0, 0]
    channels = []
    for i in idxs:
        band = arr[:, :, i].astype(np.float64)
        lo, hi = float(band.min()), float(band.max())
        stretched = (band - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(band)
        channels.append(stretched.astype(np.uint8))
    rgb = np.stack(channels, axis=-1)

    from tcip_mcp.project_paths import resolve_state

    out_dir = resolve_state(Path(".tcip") / "artifacts" / "viz" / "_band_previews")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.png"
    _Image.fromarray(rgb, mode="RGB").save(out_path)
    return str(out_path)


def _resolve_render_path(images_dir, stem: str) -> str | None:
    """The real path to hand a renderer for ``stem`` in ``images_dir`` — ``None`` if ``stem``
    isn't a resolvable logical image (missing, or a stale band-group manifest)."""
    from tcip_mcp.pipelines import image_utils

    try:
        source = image_utils.resolve_image_source(images_dir, stem)
    except (FileNotFoundError, image_utils.BandGroupIncomplete):
        return None
    return _materialize_if_needed(source)


def _renderable_path(image_path: str) -> str:
    """As ``_resolve_render_path``, for a caller that already has a path rather than a
    ``(dir, stem)`` pair. Falls back to ``image_path`` unchanged when it isn't resolvable through
    the enumeration primitive (e.g. a path outside any recognized ``images/`` layout) — the
    caller's own not-a-file / not-found handling surfaces the real error instead of this silently
    swallowing it.
    """
    img = Path(image_path)
    out = _resolve_render_path(img.parent, img.stem)
    return out if out is not None else image_path


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
    return f" ({n} point annotation(s) not drawn — no point renderer yet)" if n else ""


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
    except ValueError as exc:
        return {"error": str(exc)}
    anns = load_annotations_any(str(label_path), fmt=fmt, file_name=img.name)
    idx, index = _subject_indexer()

    n_points = _n_points(anns)
    render_path = _renderable_path(image_path)
    if task == "detect":
        shapes = _boxable(anns)
        out = render_detections(render_path, [_box_dict(a, index) for a in shapes],
                                class_names=_name_map(idx))
        summary = f"Rendered {len(shapes)} detections on {img.name}"
        if shapes:
            from collections import Counter
            counts = Counter(a.subject for a in shapes)
            summary += " — " + ", ".join(f"{v} {k}" for k, v in counts.most_common())
        summary += _point_note(n_points)
    else:
        shapes = [a for a in anns if isinstance(a.geometry, Polygon)]
        out = render_segmentations(render_path, [_poly_dict(a, index) for a in shapes],
                                   class_names=_name_map(idx))
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

    preds = read_labels(str(pred_file))
    if conf_threshold > 0:
        preds = [a for a in preds if (a.score is None or a.score >= conf_threshold)]
    idx, index = _subject_indexer()

    n_points = _n_points(preds)
    render_path = _renderable_path(image_path)
    if task == "detect":
        shapes = _boxable(preds)
        out = render_detections(render_path, [_box_dict(a, index) for a in shapes],
                                class_names=_name_map(idx))
        summary = f"Rendered {len(shapes)} predictions on {img.name}" + _point_note(n_points)
    else:
        shapes = [a for a in preds if isinstance(a.geometry, Polygon)]
        out = render_segmentations(render_path, [_poly_dict(a, index) for a in shapes],
                                   class_names=_name_map(idx))
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
    except ValueError as exc:
        return {"error": str(exc)}
    gt = _boxable(load_annotations_any(str(label_path), fmt=fmt, file_name=img.name))
    gt_dicts = [_box_dict(a, index) for a in gt]

    pred_file = find_prediction(image_path)
    pred_dicts: list[dict] = []
    if pred_file is not None:
        preds = _boxable(read_labels(str(pred_file)))
        pred_dicts = [_box_dict(a, index) for a in preds]
        # Match at the caller's conf operating point (not compute_matches' silent 0.25 default).
        match_result = compute_matches(gt, preds, iou_threshold=iou_threshold,
                                       conf_threshold=conf_threshold)
        tp = len(match_result["tp"])
        fp = len(match_result["fp"])
        fn = len(match_result["fn"])
    else:
        tp, fp, fn = 0, 0, len(gt)

    out = render_comparison(_renderable_path(image_path), gt_dicts, pred_dicts, matches=[],
                            class_names=_name_map(idx))

    return {
        "image_path": out,
        "summary": f"GT={len(gt_dicts)}, Pred={len(pred_dicts)}, TP={tp}, FP={fp}, FN={fn}",
        "gt_count": len(gt_dicts),
        "pred_count": len(pred_dicts),
        "tp": tp, "fp": fp, "fn": fn,
    }


@mcp.tool()
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

    Ranks by a count-mismatch + low-confidence heuristic (`get_worst_predictions`) — no IoU
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
        img_path = _resolve_render_path(img_dir, stem)
        if not img_path:
            continue

        idx, index = _subject_indexer()

        gt_file = Path(labels_dir) / f"{stem}.json"
        gt_dicts = []
        if gt_file.is_file():
            gt_dicts = [_box_dict(a, index) for a in _boxable(read_labels(str(gt_file)))]

        pred_file = Path(predictions_dir) / f"{stem}.json"
        pred_dicts = []
        if pred_file.is_file():
            pred_dicts = [_box_dict(a, index) for a in _boxable(read_labels(str(pred_file)))]

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
    # entry per capture — recurses into images/<date>/ subfolders (the canonical layout) and any
    # deeper nesting, mirroring the old flat rglob extension scan.
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
            except ValueError:
                label_path = None  # unrecognized store: render the image without labels
        render_path = _materialize_if_needed(source)
        if label_path is not None:
            idx, index = _subject_indexer()
            anns = load_annotations_any(str(label_path), fmt=fmt, file_name=rep_path.name)
            if task == "detect":
                shapes = _boxable(anns)
                out = render_detections(render_path, [_box_dict(a, index) for a in shapes],
                                        class_names=_name_map(idx))
            else:
                shapes = [a for a in anns if isinstance(a.geometry, Polygon)]
                out = render_segmentations(render_path, [_poly_dict(a, index) for a in shapes],
                                           class_names=_name_map(idx))
            titles.append(f"{stem} ({len(shapes)})")
        else:
            # No annotations — just use raw image
            out = render_path
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
def propose_annotations(
    image_path: str,
    engine: str = "sam",
    engine_params: dict | None = None,
) -> dict:
    """Propose candidate annotations on an image for review, using a chosen auto-labeling engine.

    Runs the engine's whole-image proposal pass, renders the numbered candidates, caches them, and
    returns the render path and neutral candidate data. Use view_image on the render, then call
    accept_proposals to assign classes and stage the accepted ones as predictions.

    The engine is a capability, not a fixed method: 'sam' is the built-in SAM2 reference; the agent
    can register another engine (``register_proposal_engine``) or pass a dotted 'module:factory' it
    wrote — then trial and compare engines by how well each one's high-conf proposals survive breeder
    review, and pick the most useful for the task.

    Args:
        image_path: Absolute path to the image file.
        engine: Proposal engine — 'sam' (built-in) or a dotted 'module:factory' the agent brings.
        engine_params: Engine-specific knobs forwarded to the engine (e.g. SAM's model_type,
            points_per_side, pred_iou_thresh, stability_score_thresh, min_mask_region_area). Omit for
            the engine's own defaults.
    """
    from tcip_mcp.pipelines.proposal import resolve_proposer

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    try:
        proposer = resolve_proposer(engine)
    except (ValueError, ImportError) as e:
        return {"error": str(e)}

    try:
        candidates = proposer.propose(image_path, **(engine_params or {}))
    except ImportError as e:
        return {"error": str(e)}
    except FileNotFoundError as e:
        return {"error": str(e)}

    if not candidates:
        return {
            "image_path": None,
            "engine": engine,
            "summary": f"Engine {engine!r} proposed no candidates",
            "candidates": [],
        }

    out = render_candidates(_renderable_path(image_path), candidates)

    # Resolve state via the platform root, not a CWD-relative path, so the
    # handoff to accept_proposals survives CWD != project root. The envelope records the engine so
    # accept_proposals stamps the right producer and stages into the matching bucket.
    from tcip_mcp.project_paths import resolve_state

    import json
    state_file = resolve_state(Path(".tcip") / "state" / f"proposals_{img.stem}.json")
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        json.dumps({"engine": engine, "candidates": candidates}, default=str), encoding="utf-8"
    )

    return {
        "image_path": out,
        "engine": engine,
        "summary": f"Engine {engine!r} proposed {len(candidates)} candidates. "
                   f"Review the numbered overlay, then call accept_proposals "
                   f"with class assignments.",
        "candidate_count": len(candidates),
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
@audited
def accept_proposals(
    image_path: str,
    assignments: list[dict],
) -> dict:
    """Assign classes to reviewed proposals and stage them as predictions for canvas review.

    After reviewing propose_annotations output, the agent calls this tool with a mapping from
    candidate IDs to class IDs. Rejected candidates are simply omitted from the assignments list.
    The masks are written to the predictions tree (``predictions/<engine>/<date>/<task>``) as
    per-image COCO/JSON with ``created_by=<engine>`` and ``score`` = the engine's proposal score —
    they are model output, so a human accepts them on the Review canvas before they become ground
    truth. Staging goes through the prediction-bucket verdict guard, so a re-run never overwrites
    reviewed predictions or orphans their verdicts. This never writes GT.

    Args:
        image_path: Absolute path to the image (same as propose_annotations).
        assignments: List of dicts, each with 'candidate_id' (int) and 'subject' (name).
            Only listed candidates are staged.
    """
    import json

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    # Load cached proposals from the same platform-root state location propose_annotations wrote to.
    from tcip_mcp.project_paths import resolve_state

    state_file = resolve_state(Path(".tcip") / "state" / f"proposals_{img.stem}.json")
    if not state_file.is_file():
        return {"error": f"No proposals found for {img.stem}. Run propose_annotations first."}

    envelope = json.loads(state_file.read_text(encoding="utf-8"))
    engine = envelope.get("engine", "unknown")
    candidates = envelope.get("candidates", [])
    cand_map = {c["candidate_id"]: c for c in candidates}

    from tcip_mcp.pipelines.image_utils import image_dimensions, resolve_image_source

    w, h = image_dimensions(resolve_image_source(img.parent, img.stem))

    # Build name-based predictions from accepted candidates (created_by=<engine>, score = the
    # engine's proposal score). Each candidate becomes ONE Annotation under its subject carrying
    # every ring the engine proposed — an occlusion-split object stays split rather than being
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

    # Stage into the predictions tree through the shared verdict-guarded helper — model output for a
    # human to accept on the Review canvas, never written straight to ground truth.
    from tcip_mcp.dataset_layout import parse_image_path
    from tcip_mcp.prediction_buckets import BucketHasVerdicts, stage_prediction_shapes

    root, date, _stem = parse_image_path(str(img))
    try:
        staged = stage_prediction_shapes(
            str(root), engine, date, img.stem,
            annotations=proposals, img_w=w, img_h=h, overwrite=False,
        )
    except BucketHasVerdicts as exc:
        return {"error": str(exc), "verdict_count": exc.count, "suggested_bucket": exc.suggested}
    bucket = staged["bucket"]

    # Render final result for QA
    idx, index = _subject_indexer()
    out = render_detections(_renderable_path(image_path), [_box_dict(a, index) for a in proposals],
                            class_names=_name_map(idx))

    note = (f"Staged {n_poly} proposal(s) from {len(assignments)} {engine!r} candidates as "
            f"predictions (created_by={engine!r}) for review — not ground truth.")
    if staged["redirected"]:
        note = (f"bucket {engine!r} has {staged['verdict_count']} review verdict(s) — staged to a fresh "
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
        _renderable_path(src_image), shapes, viewport=state.get("viewport"),
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
def overlay_reference_grid(
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

    out = render_grid_overlay(_renderable_path(image_path), cols=cols, rows=rows)

    return {
        "image_path": out,
        "summary": f"Grid overlay ({cols}x{rows}) rendered on {img.name}. "
                   f"Reference cells like 'A1' (top-left) to "
                   f"'{chr(ord('A') + cols - 1)}{rows}' (bottom-right).",
        "cols": cols,
        "rows": rows,
    }
