"""Equivalence + regression tests for the vectorized ``compute_matches``.

The vectorized IoU-pair computation must produce byte-identical output to a
straightforward per-pair reference: same tp/fp/fn *order*, same dict fields, same
rounded floats, not merely equal counts. To pin that, this file keeps an independent
reference implementation (``_reference_compute_matches``, self-contained down to its own
IoU helpers) mirroring the name-based ``compute_matches`` semantics, and asserts deep
equality across a seeded RNG sweep plus constructed edge cases.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict

import pytest

from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid

try:  # pragma: no cover - older shapely
    from shapely.errors import ShapelyError
except ImportError:  # pragma: no cover
    ShapelyError = Exception

from tcip_annotation import Annotation, BBox, Polygon, compute_matches


# ── Reference: an independent per-pair implementation (name-based) ───────────


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


def _ref_to_shapely(ann: Annotation):
    geom = ann.geometry
    if isinstance(geom, BBox):
        g = ShapelyPolygon(
            [(geom.x1, geom.y1), (geom.x2, geom.y1), (geom.x2, geom.y2), (geom.x1, geom.y2)])
    else:
        rings = [r for r in geom.rings if len(r) >= 3]
        g = (ShapelyPolygon(rings[0]) if len(rings) == 1
             else ShapelyMultiPolygon([ShapelyPolygon(r) for r in rings]))
    if not g.is_valid:
        g = make_valid(g)
    return g, g.area


def _reference_compute_matches(
    gt,
    preds,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> dict:
    gt_items = [(i, a.subject, a) for i, a in enumerate(gt) if a.geometry is not None]
    pred_items = [
        (i, a.subject, float(a.score if a.score is not None else 1.0), a)
        for i, a in enumerate(preds)
        if a.geometry is not None and (a.score is None or a.score >= conf_threshold)
    ]

    gt_by_class = defaultdict(list)
    for li, item in enumerate(gt_items):
        gt_by_class[item[1]].append(li)
    pred_by_class = defaultdict(list)
    for li, item in enumerate(pred_items):
        pred_by_class[item[1]].append(li)

    gt_geom_cache: dict = {}
    pred_geom_cache: dict = {}

    def _gt_geom(li):
        if li not in gt_geom_cache:
            gt_geom_cache[li] = _ref_to_shapely(gt_items[li][2])
        return gt_geom_cache[li]

    def _pred_geom(li):
        if li not in pred_geom_cache:
            pred_geom_cache[li] = _ref_to_shapely(pred_items[li][3])
        return pred_geom_cache[li]

    pairs = []
    for cname in gt_by_class:
        if cname not in pred_by_class:
            continue
        for gi in gt_by_class[cname]:
            for pi in pred_by_class[cname]:
                gt_ann = gt_items[gi][2]
                pred_ann = pred_items[pi][3]
                if isinstance(gt_ann.geometry, BBox) and isinstance(pred_ann.geometry, BBox):
                    iou = _ref_box_iou(gt_ann.geometry, pred_ann.geometry)
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
        gt_idx, gt_cname, _ = gt_items[gi]
        p_idx, _, p_conf, _ = pred_items[pi]
        tp_list.append(
            {
                "gt_idx": gt_idx,
                "pred_idx": p_idx,
                "iou": round(iou, 4),
                "class_name": gt_cname,
                "conf": round(p_conf, 4),
            }
        )

    fp_list = []
    for li, (p_idx, p_cname, p_conf, _) in enumerate(pred_items):
        if li not in matched_pred:
            fp_list.append({"pred_idx": p_idx, "class_name": p_cname, "conf": round(p_conf, 4)})

    fn_list = []
    for li, (gt_idx, gt_cname, _) in enumerate(gt_items):
        if li not in matched_gt:
            fn_list.append({"gt_idx": gt_idx, "class_name": gt_cname})

    return {"tp": tp_list, "fp": fp_list, "fn": fn_list}


# ── Synthetic data generators (integer class -> subject name) ────────────────


def _subj(cls: int) -> str:
    return f"c{cls}"


def _rand_box(rng, cls, space=1000.0, size=(20.0, 200.0)):
    w = rng.uniform(*size)
    h = rng.uniform(*size)
    x1 = rng.uniform(0.0, space - w)
    y1 = rng.uniform(0.0, space - h)
    return BBox(x1, y1, x1 + w, y1 + h), cls


def _gt_boxes(rng, n, n_classes, **kw):
    out = []
    for _ in range(n):
        box, cls = _rand_box(rng, rng.randrange(n_classes), **kw)
        out.append(Annotation(subject=_subj(cls), geometry=box))
    return out


def _pred_boxes(rng, gts, n, n_classes, jitter=15.0, from_gt=0.6, **kw):
    """Mix of GT-derived (overlapping) preds and random preds, with random confidence."""
    out = []
    for _ in range(n):
        conf = rng.uniform(0.0, 1.0)
        if gts and rng.random() < from_gt:
            g = rng.choice(gts)
            dx = rng.uniform(-jitter, jitter)
            dy = rng.uniform(-jitter, jitter)
            gb = g.geometry
            out.append(Annotation(
                subject=g.subject,
                geometry=BBox(gb.x1 + dx, gb.y1 + dy, gb.x2 + dx, gb.y2 + dy),
                score=conf,
            ))
        else:
            box, cls = _rand_box(rng, rng.randrange(n_classes), **kw)
            out.append(Annotation(subject=_subj(cls), geometry=box, score=conf))
    return out


def _rand_polygon(rng, cls, space=1000.0, size=(30.0, 150.0)):
    w = rng.uniform(*size)
    h = rng.uniform(*size)
    x1 = rng.uniform(0.0, space - w)
    y1 = rng.uniform(0.0, space - h)
    pts = [(x1, y1), (x1 + w, y1 + rng.uniform(-5, 5)), (x1 + w, y1 + h), (x1, y1 + h)]
    return Polygon([pts]), cls


def _rand_multi_ring_polygon(rng, cls, space=1000.0, size=(30.0, 150.0), n_rings=2):
    """One occlusion-split instance: ``n_rings`` disjoint lobes in a row, separated by gaps."""
    w = rng.uniform(*size)
    h = rng.uniform(*size)
    gap = rng.uniform(5.0, 30.0)
    span = n_rings * w + (n_rings - 1) * gap
    x1 = rng.uniform(0.0, max(0.0, space - span))
    y1 = rng.uniform(0.0, space - h)
    rings = []
    for k in range(n_rings):
        ox = x1 + k * (w + gap)
        rings.append([(ox, y1), (ox + w, y1), (ox + w, y1 + h), (ox, y1 + h)])
    return Polygon(rings), cls


def _some_polygon(rng, cls, multi_frac, **kw):
    if multi_frac and rng.random() < multi_frac:
        return _rand_multi_ring_polygon(rng, cls, n_rings=rng.choice([2, 3]), **kw)
    return _rand_polygon(rng, cls, **kw)


def _gt_polys(rng, n, n_classes, *, multi_frac=0.0, **kw):
    out = []
    for _ in range(n):
        poly, cls = _some_polygon(rng, rng.randrange(n_classes), multi_frac, **kw)
        out.append(Annotation(subject=_subj(cls), geometry=poly))
    return out


def _pred_polys(rng, n, n_classes, *, multi_frac=0.0, **kw):
    out = []
    for _ in range(n):
        poly, cls = _some_polygon(rng, rng.randrange(n_classes), multi_frac, **kw)
        conf = rng.uniform(0.0, 1.0)
        out.append(Annotation(subject=_subj(cls), geometry=poly, score=conf))
    return out


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
        got = compute_matches(gts, preds, iou_threshold=iou_t, conf_threshold=conf_t)
        ref = _reference_compute_matches(gts, preds, iou_threshold=iou_t, conf_threshold=conf_t)
        assert got == ref, f"mismatch seed={seed} n_gt={n_gt} n_pred={n_pred} classes={n_classes}"


@pytest.mark.parametrize("seed", range(6))
def test_vectorized_equivalence_dense_overlap(seed):
    """High overlap density (many preds jittered from GT) stresses tie-breaking."""
    rng = random.Random(1000 + seed)
    gts = _gt_boxes(rng, 60, 1, space=300.0, size=(40.0, 80.0))
    preds = _pred_boxes(rng, gts, 120, 1, jitter=5.0, from_gt=0.95, space=300.0, size=(40.0, 80.0))
    got = compute_matches(gts, preds, iou_threshold=0.3)
    ref = _reference_compute_matches(gts, preds, iou_threshold=0.3)
    assert got == ref


def test_all_overlapping_identical_boxes():
    """Every pred is geometrically identical to the single GT (IoU ties)."""
    gts = [Annotation(subject="c0", geometry=BBox(0, 0, 100, 100))]
    preds = [Annotation(subject="c0", geometry=BBox(0, 0, 100, 100), score=0.5 + 0.01 * i)
             for i in range(10)]
    got = compute_matches(gts, preds, iou_threshold=0.5)
    ref = _reference_compute_matches(gts, preds, iou_threshold=0.5)
    assert got == ref
    # One TP; the lowest pred_idx wins the tie (reference greedy behavior).
    assert len(got["tp"]) == 1
    assert got["tp"][0]["pred_idx"] == 0


def test_empty_gt():
    rng = random.Random(7)
    preds = _pred_boxes(rng, [], 20, 2)
    got = compute_matches([], preds)
    ref = _reference_compute_matches([], preds)
    assert got == ref
    assert got["tp"] == [] and got["fn"] == []


def test_empty_preds():
    rng = random.Random(8)
    gts = _gt_boxes(rng, 15, 2)
    got = compute_matches(gts, [])
    ref = _reference_compute_matches(gts, [])
    assert got == ref
    assert got["tp"] == [] and got["fp"] == []
    assert len(got["fn"]) == 15


def test_all_conf_filtered():
    """Every pred below conf_threshold → all GT become FN."""
    gts = [Annotation(subject="c0", geometry=BBox(0, 0, 100, 100)),
           Annotation(subject="c0", geometry=BBox(200, 200, 300, 300))]
    preds = [Annotation(subject="c0", geometry=BBox(0, 0, 100, 100), score=0.1)]
    got = compute_matches(gts, preds, conf_threshold=0.5)
    ref = _reference_compute_matches(gts, preds, conf_threshold=0.5)
    assert got == ref
    assert len(got["fn"]) == 2 and got["tp"] == [] and got["fp"] == []


def test_sub_threshold_only():
    """Overlapping but every IoU below threshold → no TP, all FP/FN."""
    rng = random.Random(9)
    gts = _gt_boxes(rng, 30, 1)
    # Small jitter but a very high threshold so nothing qualifies.
    preds = _pred_boxes(rng, gts, 30, 1, jitter=40.0, from_gt=1.0)
    got = compute_matches(gts, preds, iou_threshold=0.999)
    ref = _reference_compute_matches(gts, preds, iou_threshold=0.999)
    assert got == ref


def test_iou_boundary_exactly_at_threshold():
    """IoU exactly == threshold must be included (>=), identically to reference."""
    # Boxes with intersection 50x100 and union chosen so IoU is exactly 1/3.
    gt_box = BBox(0, 0, 100, 100)
    pred_box = BBox(50, 0, 150, 100)  # inter=5000, union=15000 → 1/3
    gts = [Annotation(subject="c0", geometry=gt_box)]
    preds = [Annotation(subject="c0", geometry=pred_box, score=0.9)]
    thr = _ref_box_iou(gt_box, pred_box)
    got = compute_matches(gts, preds, iou_threshold=thr)
    ref = _reference_compute_matches(gts, preds, iou_threshold=thr)
    assert got == ref
    assert len(got["tp"]) == 1


def test_tie_lower_pred_idx_wins():
    """Two preds with identical IoU vs one GT → lower pred_idx matches."""
    gt_box = BBox(0, 0, 100, 100)
    pred0 = BBox(10, 0, 110, 100)
    pred1 = BBox(-10, 0, 90, 100)  # mirror → identical IoU
    gts = [Annotation(subject="c0", geometry=gt_box)]
    preds = [Annotation(subject="c0", geometry=pred0, score=0.9),
             Annotation(subject="c0", geometry=pred1, score=0.9)]
    assert _ref_box_iou(gt_box, pred0) == _ref_box_iou(gt_box, pred1)
    got = compute_matches(gts, preds, iou_threshold=0.5)
    ref = _reference_compute_matches(gts, preds, iou_threshold=0.5)
    assert got == ref
    assert got["tp"][0]["pred_idx"] == 0


def test_integer_and_float_coords_agree():
    """Int-valued and float-valued coords both stay bitwise-equal to reference."""
    gts_int = [Annotation(subject="c0", geometry=BBox(0, 0, 100, 100)),
               Annotation(subject="c1", geometry=BBox(200, 200, 250, 260))]
    preds_int = [
        Annotation(subject="c0", geometry=BBox(5, 5, 95, 95), score=0.8),
        Annotation(subject="c1", geometry=BBox(205, 205, 255, 265), score=0.7),
    ]
    gts_f = [Annotation(subject="c0", geometry=BBox(0.5, 0.25, 100.75, 100.1))]
    preds_f = [Annotation(subject="c0", geometry=BBox(0.6, 0.2, 100.4, 100.9), score=0.9)]
    for gts, preds in [(gts_int, preds_int), (gts_f, preds_f)]:
        got = compute_matches(gts, preds, iou_threshold=0.3)
        ref = _reference_compute_matches(gts, preds, iou_threshold=0.3)
        assert got == ref


# ── Polygon / mixed fallback path (must match reference exactly too) ─────────


@pytest.mark.parametrize("seed", range(4))
def test_polygon_and_mixed_equivalence(seed):
    rng = random.Random(500 + seed)
    n_classes = 2
    gt_boxes = _gt_boxes(rng, 20, n_classes)
    pred_boxes = _pred_boxes(rng, gt_boxes, 20, n_classes)
    gt_polys = _gt_polys(rng, 15, n_classes)
    pred_polys = _pred_polys(rng, 15, n_classes)
    gt = gt_boxes + gt_polys
    preds = pred_boxes + pred_polys
    got = compute_matches(gt, preds, iou_threshold=0.3)
    ref = _reference_compute_matches(gt, preds, iou_threshold=0.3)
    assert got == ref


def test_mixed_same_class_box_and_polygon():
    """A class containing both box and polygon items uses the exact fallback loop."""
    gt = [
        Annotation(subject="c0", geometry=BBox(0, 0, 100, 100)),
        Annotation(subject="c0", geometry=Polygon([[(0, 0), (100, 0), (100, 100), (0, 100)]])),
    ]
    preds = [
        Annotation(subject="c0", geometry=BBox(5, 5, 95, 95), score=0.9),
        Annotation(subject="c0", geometry=Polygon([[(2, 2), (98, 2), (98, 98), (2, 98)]]), score=0.8),
    ]
    got = compute_matches(gt, preds, iou_threshold=0.4)
    ref = _reference_compute_matches(gt, preds, iou_threshold=0.4)
    assert got == ref


# ── Multi-ring (occlusion-split) instances ───────────────────────────────────


@pytest.mark.parametrize("seed", range(4))
def test_multi_ring_polygon_equivalence(seed):
    """Multi-ring instances take the same exact fallback loop and must match the reference, whose
    own geometry builder is an independent MultiPolygon union of every ring."""
    rng = random.Random(900 + seed)
    n_classes = 2
    gt = _gt_polys(rng, 20, n_classes, multi_frac=0.7)
    preds = _pred_polys(rng, 20, n_classes, multi_frac=0.7)
    assert any(len(a.geometry.rings) > 1 for a in gt)  # the sweep really produced multi-ring GT
    got = compute_matches(gt, preds, iou_threshold=0.3)
    ref = _reference_compute_matches(gt, preds, iou_threshold=0.3)
    assert got == ref


def test_multi_ring_gt_iou_counts_the_area_of_every_ring():
    """The IoU denominator is the union of all of an instance's rings.

    Lobe A is 20x40 = 800, lobe B is 50x50 = 2500. A prediction that recovered only lobe A therefore
    scores 800/3300 against the two-lobe GT: comparing against lobe A alone would score it 1.0 and
    hand a half-found object a perfect match.
    """
    lobe_a = [(10.0, 10.0), (30.0, 10.0), (30.0, 50.0), (10.0, 50.0)]
    lobe_b = [(70.0, 10.0), (120.0, 10.0), (120.0, 60.0), (70.0, 60.0)]
    gt = [Annotation(subject="c0", geometry=Polygon([lobe_a, lobe_b]))]
    preds = [Annotation(subject="c0", geometry=Polygon([lobe_a]), score=0.9)]

    got = compute_matches(gt, preds, iou_threshold=0.1)
    ref = _reference_compute_matches(gt, preds, iou_threshold=0.1)
    assert got == ref
    assert got["tp"][0]["iou"] == round(800.0 / 3300.0, 4)

    # And the area really is the union, not the first ring: the reference oracle agrees.
    _, area = _ref_to_shapely(gt[0])
    assert area == 3300.0


# ── Rough timing sanity (no strict threshold) ────────────────────────────────


def test_timing_sanity_vectorized_not_slower():
    """At a realistic review scale the vectorized path must beat the pure loop.

    Correctness is still asserted; the timing check is a rough regression guard,
    not a benchmark: it only fires the comparative assert when the reference
    workload is non-trivial, to avoid CI noise on tiny inputs.
    """
    rng = random.Random(42)
    gts = _gt_boxes(rng, 1500, 1)
    preds = _pred_boxes(rng, gts, 1500, 1)

    t0 = time.perf_counter()
    got = compute_matches(gts, preds, iou_threshold=0.5)
    t_new = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref = _reference_compute_matches(gts, preds, iou_threshold=0.5)
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
    ref = _reference_compute_matches(gts, preds, iou_threshold=0.3)

    monkeypatch.setattr(matching, "_IOU_MATRIX_BUDGET", 7)  # ~1 row per chunk
    got = compute_matches(gts, preds, iou_threshold=0.3)
    assert got == ref
