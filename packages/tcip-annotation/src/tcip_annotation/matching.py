"""Geometry helpers and GT-vs-prediction matching engine.

All functions are pure (no GUI dependencies).
"""

from __future__ import annotations

import logging
from collections import defaultdict

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.validation import make_valid

try:
    from shapely.errors import ShapelyError
except ImportError:  # pragma: no cover - older shapely
    ShapelyError = Exception

from tcip_annotation.state import BBox, Polygon, PredBBox, PredPolygon

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


def _bbox_of(item_type: str, data) -> tuple[float, float, float, float]:
    """Get axis-aligned bounding box for a box or polygon."""
    if item_type == "box":
        return (data.x1, data.y1, data.x2, data.y2)
    # polygon
    xs = [p[0] for p in data.points]
    ys = [p[1] for p in data.points]
    return (min(xs), min(ys), max(xs), max(ys))


def _to_shapely(item_type: str, data) -> tuple[ShapelyPolygon, float]:
    """Convert a box or polygon to a Shapely geometry + area."""
    if item_type == "box":
        pts = [(data.x1, data.y1), (data.x2, data.y1), (data.x2, data.y2), (data.x1, data.y2)]
    else:
        pts = data.points
    g = ShapelyPolygon(pts)
    if not g.is_valid:
        g = make_valid(g)
    return g, g.area


def compute_matches(
    gt_boxes: list[BBox],
    gt_polygons: list[Polygon],
    pred_boxes: list[PredBBox],
    pred_polygons: list[PredPolygon],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> dict:
    """Match predictions to GT; classify as TP / FP / FN.

    Uses greedy matching: sort all same-class GT-Pred pairs by IoU
    descending, then assign greedily.

    Returns
    -------
    dict with keys ``'tp'``, ``'fp'``, ``'fn'``.
      - ``tp``: list of dicts ``{gt_type, gt_idx, pred_type, pred_idx, iou, class_id, conf}``
      - ``fp``: list of dicts ``{pred_type, pred_idx, class_id, conf}``
      - ``fn``: list of dicts ``{gt_type, gt_idx, class_id}``
    """
    # Build unified lists with type tags
    gt_items: list[tuple[str, int, int, object]] = []
    for i, b in enumerate(gt_boxes):
        gt_items.append(("box", i, b.class_id, b))
    for i, p in enumerate(gt_polygons):
        gt_items.append(("polygon", i, p.class_id, p))

    pred_items: list[tuple[str, int, int, float, object]] = []
    for i, b in enumerate(pred_boxes):
        if b.confidence >= conf_threshold:
            pred_items.append(("box", i, b.class_id, b.confidence, b))
    for i, p in enumerate(pred_polygons):
        if p.confidence >= conf_threshold:
            pred_items.append(("polygon", i, p.class_id, p.confidence, p))

    # Group by class
    gt_by_class: dict[int, list[int]] = defaultdict(list)
    for gi, item in enumerate(gt_items):
        gt_by_class[item[2]].append(gi)
    pred_by_class: dict[int, list[int]] = defaultdict(list)
    for pi, item in enumerate(pred_items):
        pred_by_class[item[2]].append(pi)

    # Shapely geometry caches (lazy)
    gt_geom_cache: dict[int, tuple] = {}
    pred_geom_cache: dict[int, tuple] = {}

    def _gt_geom(gi: int):
        if gi not in gt_geom_cache:
            gt_geom_cache[gi] = _to_shapely(gt_items[gi][0], gt_items[gi][3])
        return gt_geom_cache[gi]

    def _pred_geom(pi: int):
        if pi not in pred_geom_cache:
            pred_geom_cache[pi] = _to_shapely(pred_items[pi][0], pred_items[pi][4])
        return pred_geom_cache[pi]

    # Compute all same-class IoU pairs. Pure-box classes (the common detection case)
    # use a vectorized numpy IoU matrix; any class involving a polygon falls back to
    # the exact per-pair loop so emit order and IoU values stay byte-identical.
    import numpy as np

    pairs: list[tuple[float, int, int]] = []
    for cid in gt_by_class:
        if cid not in pred_by_class:
            continue
        gis = gt_by_class[cid]
        pis = pred_by_class[cid]
        if all(gt_items[gi][0] == "box" for gi in gis) and all(
            pred_items[pi][0] == "box" for pi in pis
        ):
            gt_arr = np.array(
                [(d.x1, d.y1, d.x2, d.y2) for d in (gt_items[gi][3] for gi in gis)],
                dtype=np.float64,
            )
            pred_arr = np.array(
                [(d.x1, d.y1, d.x2, d.y2) for d in (pred_items[pi][4] for pi in pis)],
                dtype=np.float64,
            )
            _append_box_iou_pairs(np, gt_arr, pred_arr, gis, pis, iou_threshold, pairs)
            continue
        for gi in gis:
            for pi in pis:
                gt_type, _, _, gt_data = gt_items[gi]
                p_type, _, _, _, p_data = pred_items[pi]
                if gt_type == "box" and p_type == "box":
                    iou = box_iou(gt_data, p_data)
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
        gt_type, gt_idx, gt_cid, _ = gt_items[gi]
        p_type, p_idx, _, p_conf, _ = pred_items[pi]
        tp_list.append(
            {
                "gt_type": gt_type,
                "gt_idx": gt_idx,
                "pred_type": p_type,
                "pred_idx": p_idx,
                "iou": round(iou, 4),
                "class_id": gt_cid,
                "conf": round(p_conf, 4),
            }
        )

    # Unmatched predictions → FP
    fp_list: list[dict] = []
    for pi, (p_type, p_idx, p_cid, p_conf, _) in enumerate(pred_items):
        if pi not in matched_pred:
            fp_list.append({"pred_type": p_type, "pred_idx": p_idx, "class_id": p_cid, "conf": round(p_conf, 4)})

    # Unmatched GT → FN
    fn_list: list[dict] = []
    for gi, (gt_type, gt_idx, gt_cid, _) in enumerate(gt_items):
        if gi not in matched_gt:
            fn_list.append({"gt_type": gt_type, "gt_idx": gt_idx, "class_id": gt_cid})

    return {"tp": tp_list, "fp": fp_list, "fn": fn_list}


def point_in_polygon(x: float, y: float, polygon: Polygon) -> bool:
    """Test whether a point lies inside a polygon using Shapely."""
    geom = ShapelyPolygon(polygon.points)
    if not geom.is_valid:
        geom = make_valid(geom)
    return geom.contains(ShapelyPoint(x, y))
