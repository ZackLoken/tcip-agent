"""Equivalence + regression tests for the vectorized ``compute_matches``.

The vectorized IoU-pair computation must produce byte-identical output to the
prior pure-Python nested-loop implementation: same tp/fp/fn *order*, same tuple
fields, same rounded floats — not merely equal counts. To pin that, this file
keeps a verbatim copy of the old implementation (``_reference_compute_matches``,
self-contained down to its own IoU helpers) and asserts deep equality across a
seeded RNG sweep plus constructed edge cases.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict

import pytest

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid

try:  # pragma: no cover - older shapely
    from shapely.errors import ShapelyError
except ImportError:  # pragma: no cover
    ShapelyError = Exception

from tcip_annotation import BBox, Polygon, PredBBox, PredPolygon, compute_matches


# ── Reference: verbatim copy of the pre-vectorization implementation ─────────


def _ref_box_iou(b1, b2) -> float:
    x1 = max(b1.x1, b2.x1)
    y1 = max(b1.y1, b2.y1)
    x2 = min(b1.x2, b2.x2)
    y2 = min(b1.y2, b2.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (b1.x2 - b1.x1) * (b1.y2 - b1.y1)
    area2 = (b2.x2 - b2.x1) * (b2.y2 - b2.y1)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _ref_polygon_iou(geom1, area1, geom2, area2) -> float:
    try:
        inter = geom1.intersection(geom2).area
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
    except (ShapelyError, ValueError, ZeroDivisionError):
        return 0.0


def _ref_to_shapely(item_type, data):
    if item_type == "box":
        pts = [(data.x1, data.y1), (data.x2, data.y1), (data.x2, data.y2), (data.x1, data.y2)]
    else:
        pts = data.points
    g = ShapelyPolygon(pts)
    if not g.is_valid:
        g = make_valid(g)
    return g, g.area


def _reference_compute_matches(
    gt_boxes,
    gt_polygons,
    pred_boxes,
    pred_polygons,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> dict:
    gt_items = []
    for i, b in enumerate(gt_boxes):
        gt_items.append(("box", i, b.class_id, b))
    for i, p in enumerate(gt_polygons):
        gt_items.append(("polygon", i, p.class_id, p))

    pred_items = []
    for i, b in enumerate(pred_boxes):
        if b.confidence >= conf_threshold:
            pred_items.append(("box", i, b.class_id, b.confidence, b))
    for i, p in enumerate(pred_polygons):
        if p.confidence >= conf_threshold:
            pred_items.append(("polygon", i, p.class_id, p.confidence, p))

    gt_by_class = defaultdict(list)
    for gi, item in enumerate(gt_items):
        gt_by_class[item[2]].append(gi)
    pred_by_class = defaultdict(list)
    for pi, item in enumerate(pred_items):
        pred_by_class[item[2]].append(pi)

    gt_geom_cache: dict = {}
    pred_geom_cache: dict = {}

    def _gt_geom(gi):
        if gi not in gt_geom_cache:
            gt_geom_cache[gi] = _ref_to_shapely(gt_items[gi][0], gt_items[gi][3])
        return gt_geom_cache[gi]

    def _pred_geom(pi):
        if pi not in pred_geom_cache:
            pred_geom_cache[pi] = _ref_to_shapely(pred_items[pi][0], pred_items[pi][4])
        return pred_geom_cache[pi]

    pairs = []
    for cid in gt_by_class:
        if cid not in pred_by_class:
            continue
        for gi in gt_by_class[cid]:
            for pi in pred_by_class[cid]:
                gt_type, _, _, gt_data = gt_items[gi]
                p_type, _, _, _, p_data = pred_items[pi]
                if gt_type == "box" and p_type == "box":
                    iou = _ref_box_iou(gt_data, p_data)
                else:
                    g1, a1 = _gt_geom(gi)
                    g2, a2 = _pred_geom(pi)
                    iou = _ref_polygon_iou(g1, a1, g2, a2)
                if iou >= iou_threshold:
                    pairs.append((iou, gi, pi))

    pairs.sort(key=lambda x: x[0], reverse=True)
    matched_gt: set = set()
    matched_pred: set = set()
    tp_list = []
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

    fp_list = []
    for pi, (p_type, p_idx, p_cid, p_conf, _) in enumerate(pred_items):
        if pi not in matched_pred:
            fp_list.append({"pred_type": p_type, "pred_idx": p_idx, "class_id": p_cid, "conf": round(p_conf, 4)})

    fn_list = []
    for gi, (gt_type, gt_idx, gt_cid, _) in enumerate(gt_items):
        if gi not in matched_gt:
            fn_list.append({"gt_type": gt_type, "gt_idx": gt_idx, "class_id": gt_cid})

    return {"tp": tp_list, "fp": fp_list, "fn": fn_list}


# ── Synthetic data generators ───────────────────────────────────────────────


def _rand_box(rng, cls, space=1000.0, size=(20.0, 200.0)):
    w = rng.uniform(*size)
    h = rng.uniform(*size)
    x1 = rng.uniform(0.0, space - w)
    y1 = rng.uniform(0.0, space - h)
    return x1, y1, x1 + w, y1 + h, cls


def _gt_boxes(rng, n, n_classes, **kw):
    return [BBox(*_rand_box(rng, rng.randrange(n_classes), **kw)) for _ in range(n)]


def _pred_boxes(rng, gts, n, n_classes, jitter=15.0, from_gt=0.6, **kw):
    """Mix of GT-derived (overlapping) preds and random preds, with random confidence."""
    out = []
    for _ in range(n):
        conf = rng.uniform(0.0, 1.0)
        if gts and rng.random() < from_gt:
            g = rng.choice(gts)
            dx = rng.uniform(-jitter, jitter)
            dy = rng.uniform(-jitter, jitter)
            out.append(PredBBox(g.x1 + dx, g.y1 + dy, g.x2 + dx, g.y2 + dy, g.class_id, confidence=conf))
        else:
            x1, y1, x2, y2, cls = _rand_box(rng, rng.randrange(n_classes), **kw)
            out.append(PredBBox(x1, y1, x2, y2, cls, confidence=conf))
    return out


def _rand_polygon(rng, cls, space=1000.0, size=(30.0, 150.0)):
    w = rng.uniform(*size)
    h = rng.uniform(*size)
    x1 = rng.uniform(0.0, space - w)
    y1 = rng.uniform(0.0, space - h)
    pts = [(x1, y1), (x1 + w, y1 + rng.uniform(-5, 5)), (x1 + w, y1 + h), (x1, y1 + h)]
    return pts, cls


# ── Equivalence sweep ────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(12))
def test_vectorized_equivalence_box_sweep(seed):
    """Deep-equal output vs reference over varied n / classes / density / seeds."""
    rng = random.Random(seed)
    for n_gt, n_pred, n_classes, iou_t, conf_t in [
        (5, 5, 1, 0.5, 0.25),
        (5, 8, 2, 0.3, 0.5),
        (50, 40, 1, 0.5, 0.25),
        (50, 60, 3, 0.6, 0.0),
        (200, 200, 1, 0.5, 0.25),
        (200, 150, 4, 0.4, 0.3),
    ]:
        gts = _gt_boxes(rng, n_gt, n_classes)
        preds = _pred_boxes(rng, gts, n_pred, n_classes)
        got = compute_matches(gts, [], preds, [], iou_threshold=iou_t, conf_threshold=conf_t)
        ref = _reference_compute_matches(gts, [], preds, [], iou_threshold=iou_t, conf_threshold=conf_t)
        assert got == ref, f"mismatch seed={seed} n_gt={n_gt} n_pred={n_pred} classes={n_classes}"


@pytest.mark.parametrize("seed", range(6))
def test_vectorized_equivalence_dense_overlap(seed):
    """High overlap density (many preds jittered from GT) stresses tie-breaking."""
    rng = random.Random(1000 + seed)
    gts = _gt_boxes(rng, 60, 1, space=300.0, size=(40.0, 80.0))
    preds = _pred_boxes(rng, gts, 120, 1, jitter=5.0, from_gt=0.95, space=300.0, size=(40.0, 80.0))
    got = compute_matches(gts, [], preds, [], iou_threshold=0.3)
    ref = _reference_compute_matches(gts, [], preds, [], iou_threshold=0.3)
    assert got == ref


def test_all_overlapping_identical_boxes():
    """Every pred is geometrically identical to the single GT (IoU ties)."""
    gts = [BBox(0, 0, 100, 100, 0)]
    preds = [PredBBox(0, 0, 100, 100, 0, confidence=0.5 + 0.01 * i) for i in range(10)]
    got = compute_matches(gts, [], preds, [], iou_threshold=0.5)
    ref = _reference_compute_matches(gts, [], preds, [], iou_threshold=0.5)
    assert got == ref
    # One TP; the lowest pred_idx wins the tie (reference greedy behavior).
    assert len(got["tp"]) == 1
    assert got["tp"][0]["pred_idx"] == 0


def test_empty_gt():
    rng = random.Random(7)
    preds = _pred_boxes(rng, [], 20, 2)
    got = compute_matches([], [], preds, [])
    ref = _reference_compute_matches([], [], preds, [])
    assert got == ref
    assert got["tp"] == [] and got["fn"] == []


def test_empty_preds():
    rng = random.Random(8)
    gts = _gt_boxes(rng, 15, 2)
    got = compute_matches(gts, [], [], [])
    ref = _reference_compute_matches(gts, [], [], [])
    assert got == ref
    assert got["tp"] == [] and got["fp"] == []
    assert len(got["fn"]) == 15


def test_all_conf_filtered():
    """Every pred below conf_threshold → all GT become FN."""
    gts = [BBox(0, 0, 100, 100, 0), BBox(200, 200, 300, 300, 0)]
    preds = [PredBBox(0, 0, 100, 100, 0, confidence=0.1)]
    got = compute_matches(gts, [], preds, [], conf_threshold=0.5)
    ref = _reference_compute_matches(gts, [], preds, [], conf_threshold=0.5)
    assert got == ref
    assert len(got["fn"]) == 2 and got["tp"] == [] and got["fp"] == []


def test_sub_threshold_only():
    """Overlapping but every IoU below threshold → no TP, all FP/FN."""
    rng = random.Random(9)
    gts = _gt_boxes(rng, 30, 1)
    # Small jitter but a very high threshold so nothing qualifies.
    preds = _pred_boxes(rng, gts, 30, 1, jitter=40.0, from_gt=1.0)
    got = compute_matches(gts, [], preds, [], iou_threshold=0.999)
    ref = _reference_compute_matches(gts, [], preds, [], iou_threshold=0.999)
    assert got == ref


def test_iou_boundary_exactly_at_threshold():
    """IoU exactly == threshold must be included (>=), identically to reference."""
    # Boxes with intersection 50x100 and union chosen so IoU is exactly 1/3.
    gts = [BBox(0, 0, 100, 100, 0)]
    preds = [PredBBox(50, 0, 150, 100, 0, confidence=0.9)]  # inter=5000, union=15000 → 1/3
    thr = _ref_box_iou(gts[0], preds[0])
    got = compute_matches(gts, [], preds, [], iou_threshold=thr)
    ref = _reference_compute_matches(gts, [], preds, [], iou_threshold=thr)
    assert got == ref
    assert len(got["tp"]) == 1


def test_tie_lower_pred_idx_wins():
    """Two preds with identical IoU vs one GT → lower pred_idx matches."""
    gts = [BBox(0, 0, 100, 100, 0)]
    preds = [
        PredBBox(10, 0, 110, 100, 0, confidence=0.9),
        PredBBox(-10, 0, 90, 100, 0, confidence=0.9),  # mirror → identical IoU
    ]
    assert _ref_box_iou(gts[0], preds[0]) == _ref_box_iou(gts[0], preds[1])
    got = compute_matches(gts, [], preds, [], iou_threshold=0.5)
    ref = _reference_compute_matches(gts, [], preds, [], iou_threshold=0.5)
    assert got == ref
    assert got["tp"][0]["pred_idx"] == 0


def test_integer_and_float_coords_agree():
    """Int-valued and float-valued coords both stay bitwise-equal to reference."""
    gts_int = [BBox(0, 0, 100, 100, 0), BBox(200, 200, 250, 260, 1)]
    preds_int = [
        PredBBox(5, 5, 95, 95, 0, confidence=0.8),
        PredBBox(205, 205, 255, 265, 1, confidence=0.7),
    ]
    gts_f = [BBox(0.5, 0.25, 100.75, 100.1, 0)]
    preds_f = [PredBBox(0.6, 0.2, 100.4, 100.9, 0, confidence=0.9)]
    for gts, preds in [(gts_int, preds_int), (gts_f, preds_f)]:
        got = compute_matches(gts, [], preds, [], iou_threshold=0.3)
        ref = _reference_compute_matches(gts, [], preds, [], iou_threshold=0.3)
        assert got == ref


# ── Polygon / mixed fallback path (must match reference exactly too) ─────────


@pytest.mark.parametrize("seed", range(4))
def test_polygon_and_mixed_equivalence(seed):
    rng = random.Random(500 + seed)
    n_classes = 2
    gt_boxes = _gt_boxes(rng, 20, n_classes)
    pred_boxes = _pred_boxes(rng, gt_boxes, 20, n_classes)
    gt_polys = [Polygon(*_rand_polygon(rng, rng.randrange(n_classes))) for _ in range(15)]
    pred_polys = [
        PredPolygon(*_rand_polygon(rng, rng.randrange(n_classes)), confidence=rng.uniform(0.0, 1.0))
        for _ in range(15)
    ]
    got = compute_matches(gt_boxes, gt_polys, pred_boxes, pred_polys, iou_threshold=0.3)
    ref = _reference_compute_matches(gt_boxes, gt_polys, pred_boxes, pred_polys, iou_threshold=0.3)
    assert got == ref


def test_mixed_same_class_box_and_polygon():
    """A class containing both box and polygon items uses the exact fallback loop."""
    gt_boxes = [BBox(0, 0, 100, 100, 0)]
    gt_polys = [Polygon([(0, 0), (100, 0), (100, 100), (0, 100)], 0)]
    pred_boxes = [PredBBox(5, 5, 95, 95, 0, confidence=0.9)]
    pred_polys = [PredPolygon([(2, 2), (98, 2), (98, 98), (2, 98)], 0, confidence=0.8)]
    got = compute_matches(gt_boxes, gt_polys, pred_boxes, pred_polys, iou_threshold=0.4)
    ref = _reference_compute_matches(gt_boxes, gt_polys, pred_boxes, pred_polys, iou_threshold=0.4)
    assert got == ref


# ── Rough timing sanity (no strict threshold) ────────────────────────────────


def test_timing_sanity_vectorized_not_slower():
    """At a realistic review scale the vectorized path must beat the pure loop.

    Correctness is still asserted; the timing check is a rough regression guard,
    not a benchmark — it only fires the comparative assert when the reference
    workload is non-trivial, to avoid CI noise on tiny inputs.
    """
    rng = random.Random(42)
    gts = _gt_boxes(rng, 1500, 1)
    preds = _pred_boxes(rng, gts, 1500, 1)

    t0 = time.perf_counter()
    got = compute_matches(gts, [], preds, [], iou_threshold=0.5)
    t_new = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref = _reference_compute_matches(gts, [], preds, [], iou_threshold=0.5)
    t_ref = time.perf_counter() - t0

    assert got == ref
    assert t_new < 5.0  # generous absolute ceiling
    if t_ref > 0.2:  # only compare when the pure-Python path did real work
        assert t_new < t_ref


def test_row_chunking_preserves_order(monkeypatch):
    """Shrinking the matrix budget forces GT-axis chunking; output must not change."""
    import tcip_annotation.matching as matching

    rng = random.Random(77)
    gts = _gt_boxes(rng, 40, 1, space=200.0, size=(30.0, 60.0))
    preds = _pred_boxes(rng, gts, 40, 1, jitter=5.0, from_gt=0.9, space=200.0, size=(30.0, 60.0))
    ref = _reference_compute_matches(gts, [], preds, [], iou_threshold=0.3)

    monkeypatch.setattr(matching, "_IOU_MATRIX_BUDGET", 7)  # ~1 row per chunk
    got = compute_matches(gts, [], preds, [], iou_threshold=0.3)
    assert got == ref
