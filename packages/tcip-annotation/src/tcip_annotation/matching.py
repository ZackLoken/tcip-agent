"""Geometry helpers and GT-vs-prediction matching engine.

All functions are pure (no GUI dependencies). Ground truth and predictions are both
:class:`~tcip_annotation.state.Annotation` lists: a prediction is an annotation whose ``score`` is
set. Matching groups by *class name* (an annotation's ``subject``); an integer class id never appears.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.validation import make_valid

try:
    from shapely.errors import ShapelyError
except ImportError:  # pragma: no cover - older shapely
    ShapelyError = Exception

from tcip_annotation.state import Annotation, BBox, Point, Polygon

logger = logging.getLogger(__name__)


def box_iou(b1: BBox, b2: BBox) -> float:
    """Compute IoU between two axis-aligned bounding boxes."""
    x1 = max(b1.x1, b2.x1)
    y1 = max(b1.y1, b2.y1)
    x2 = min(b1.x2, b2.x2)
    y2 = min(b1.y2, b2.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (b1.x2 - b1.x1) * (b1.y2 - b1.y1)
    area2 = (b2.x2 - b2.x1) * (b2.y2 - b2.y1)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def polygon_iou(geom1: ShapelyPolygon, area1: float, geom2: ShapelyPolygon, area2: float) -> float:
    """Compute IoU between two pre-built Shapely geometries."""
    try:
        inter = geom1.intersection(geom2).area
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
    except (ShapelyError, ValueError, ZeroDivisionError) as exc:
        # Degenerate/invalid geometry: log and treat as no overlap (don't mask other bugs).
        logger.debug("polygon_iou failed (%s); returning 0.0", exc)
        return 0.0


# Row-block budget for the vectorized IoU matrix: chunk the GT axis so the transient
# (rows x preds) float64 arrays stay bounded on very large classes, while still emitting
# pairs in row-major (gt asc, pred asc) order.
_IOU_MATRIX_BUDGET = 25_000_000


def _append_box_iou_pairs(np, gt_arr, pred_arr, gis, pis, iou_threshold, pairs) -> None:
    """Vectorized box IoU → append ``(iou, gt_idx, pred_idx)`` in row-major order.

    Numeric ops mirror ``box_iou`` (same float64 operation order) so values and the
    ``>= iou_threshold`` boundary are bitwise-identical to the scalar path.
    """
    m = gt_arr.shape[0]
    k = pred_arr.shape[0]
    if m == 0 or k == 0:
        return
    px1 = pred_arr[:, 0]
    py1 = pred_arr[:, 1]
    px2 = pred_arr[:, 2]
    py2 = pred_arr[:, 3]
    areas_pred = (px2 - px1) * (py2 - py1)
    chunk = max(1, _IOU_MATRIX_BUDGET // k)
    for r0 in range(0, m, chunk):
        g = gt_arr[r0 : r0 + chunk]
        gx1 = g[:, 0][:, None]
        gy1 = g[:, 1][:, None]
        gx2 = g[:, 2][:, None]
        gy2 = g[:, 3][:, None]
        iw = np.maximum(0.0, np.minimum(gx2, px2[None, :]) - np.maximum(gx1, px1[None, :]))
        ih = np.maximum(0.0, np.minimum(gy2, py2[None, :]) - np.maximum(gy1, py1[None, :]))
        inter = iw * ih
        areas_g = ((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]))[:, None]
        union = areas_g + areas_pred[None, :] - inter
        iou = np.zeros_like(union)
        np.divide(inter, union, out=iou, where=union > 0)
        rows, cols = np.nonzero(iou >= iou_threshold)
        for rr, cc in zip(rows.tolist(), cols.tolist()):
            pairs.append((float(iou[rr, cc]), gis[r0 + rr], pis[cc]))


def _is_box(a: Annotation) -> bool:
    return isinstance(a.geometry, BBox)


def _rings_to_shapely(rings: list[list[tuple[float, float]]]):
    """One or more simple closed rings -> a Shapely Polygon (one ring) or MultiPolygon (several);
    every ring contributes, never just the first/largest."""
    valid = [r for r in rings if len(r) >= 3]
    if len(valid) == 1:
        return ShapelyPolygon(valid[0])
    return ShapelyMultiPolygon([ShapelyPolygon(r) for r in valid])


def _to_shapely(a: Annotation):
    """Convert an annotation's geometry to a Shapely polygon (or multipolygon) + area."""
    if isinstance(a.geometry, BBox):
        b = a.geometry
        pts = [(b.x1, b.y1), (b.x2, b.y1), (b.x2, b.y2), (b.x1, b.y2)]
        g = ShapelyPolygon(pts)
    elif isinstance(a.geometry, Polygon):
        g = _rings_to_shapely(a.geometry.rings)
    else:  # pragma: no cover - callers filter geometry-less annotations out first
        g = ShapelyPolygon([])
    if not g.is_valid:
        g = make_valid(g)
    return g, g.area


def compute_matches(
    gt: list[Annotation],
    preds: list[Annotation],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> dict:
    """Match predictions to GT; classify as TP / FP / FN.

    ``gt`` / ``preds`` are :class:`Annotation` lists (a prediction carries a ``score``). Matching is
    per class name (``subject``) using greedy IoU. Geometry-less annotations (image-level labels)
    carry no spatial extent and are ignored here, as is a :class:`~tcip_annotation.state.Point`,
    which has no area and so no IoU with anything: it can be neither matched, nor a FP, nor a FN
    without fabricating a spatial claim it does not make.

    Returns a dict with keys ``'tp'`` / ``'fp'`` / ``'fn'``:
      - ``tp``: ``{gt_idx, pred_idx, iou, class_name, conf}``
      - ``fp``: ``{pred_idx, class_name, conf}``
      - ``fn``: ``{gt_idx, class_name}``

    ``gt_idx`` / ``pred_idx`` index into ``gt`` / ``preds`` directly.
    """
    def _matchable(a: Annotation) -> bool:
        return a.geometry is not None and not isinstance(a.geometry, Point)

    gt_items: list[tuple[int, str, Annotation]] = [
        (i, a.subject, a) for i, a in enumerate(gt) if _matchable(a)
    ]
    pred_items: list[tuple[int, str, float, Annotation]] = [
        (i, a.subject, float(a.score if a.score is not None else 1.0), a)
        for i, a in enumerate(preds)
        if _matchable(a) and (a.score is None or a.score >= conf_threshold)
    ]

    gt_by_class: dict[str, list[int]] = defaultdict(list)
    for li, item in enumerate(gt_items):
        gt_by_class[item[1]].append(li)
    pred_by_class: dict[str, list[int]] = defaultdict(list)
    for li, item in enumerate(pred_items):
        pred_by_class[item[1]].append(li)

    gt_geom_cache: dict[int, tuple] = {}
    pred_geom_cache: dict[int, tuple] = {}

    def _gt_geom(li: int):
        if li not in gt_geom_cache:
            gt_geom_cache[li] = _to_shapely(gt_items[li][2])
        return gt_geom_cache[li]

    def _pred_geom(li: int):
        if li not in pred_geom_cache:
            pred_geom_cache[li] = _to_shapely(pred_items[li][3])
        return pred_geom_cache[li]

    # Compute all same-class IoU pairs. Pure-box classes (the common detection case) use a
    # vectorized numpy IoU matrix; any class involving a polygon falls back to the exact per-pair
    # loop so emit order and IoU values stay byte-identical.
    import numpy as np

    pairs: list[tuple[float, int, int]] = []
    for cname in gt_by_class:
        if cname not in pred_by_class:
            continue
        gis = gt_by_class[cname]
        pis = pred_by_class[cname]
        if all(_is_box(gt_items[li][2]) for li in gis) and all(
            _is_box(pred_items[li][3]) for li in pis
        ):
            gt_arr = np.array(
                [(d.geometry.x1, d.geometry.y1, d.geometry.x2, d.geometry.y2)
                 for d in (gt_items[li][2] for li in gis)],
                dtype=np.float64,
            )
            pred_arr = np.array(
                [(d.geometry.x1, d.geometry.y1, d.geometry.x2, d.geometry.y2)
                 for d in (pred_items[li][3] for li in pis)],
                dtype=np.float64,
            )
            _append_box_iou_pairs(np, gt_arr, pred_arr, gis, pis, iou_threshold, pairs)
            continue
        for gi in gis:
            for pi in pis:
                gt_ann = gt_items[gi][2]
                pred_ann = pred_items[pi][3]
                if _is_box(gt_ann) and _is_box(pred_ann):
                    iou = box_iou(gt_ann.geometry, pred_ann.geometry)
                else:
                    g1, a1 = _gt_geom(gi)
                    g2, a2 = _pred_geom(pi)
                    iou = polygon_iou(g1, a1, g2, a2)
                if iou >= iou_threshold:
                    pairs.append((iou, gi, pi))

    # Greedy matching (descending IoU)
    pairs.sort(key=lambda x: x[0], reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    tp_list: list[dict] = []

    for iou, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        gt_idx, gt_cname, _ = gt_items[gi]
        p_idx, _, p_conf, _ = pred_items[pi]
        tp_list.append({
            "gt_idx": gt_idx,
            "pred_idx": p_idx,
            "iou": round(iou, 4),
            "class_name": gt_cname,
            "conf": round(p_conf, 4),
        })

    # Unmatched predictions → FP
    fp_list: list[dict] = []
    for li, (p_idx, p_cname, p_conf, _) in enumerate(pred_items):
        if li not in matched_pred:
            fp_list.append({"pred_idx": p_idx, "class_name": p_cname, "conf": round(p_conf, 4)})

    # Unmatched GT → FN
    fn_list: list[dict] = []
    for li, (gt_idx, gt_cname, _) in enumerate(gt_items):
        if li not in matched_gt:
            fn_list.append({"gt_idx": gt_idx, "class_name": gt_cname})

    return {"tp": tp_list, "fp": fp_list, "fn": fn_list}


def point_in_polygon(x: float, y: float, polygon: Polygon) -> bool:
    """Test whether a point lies inside any ring of a polygon using Shapely."""
    geom = _rings_to_shapely(polygon.rings)
    if not geom.is_valid:
        geom = make_valid(geom)
    return geom.contains(ShapelyPoint(x, y))
