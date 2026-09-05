"""Tests for matching.py: known GT+pred → expected TP/FP/FN."""

from __future__ import annotations

import pytest

from tcip_annotation import (
    Annotation,
    BBox,
    Polygon,
    compute_matches,
    box_iou,
    polygon_iou,
    point_in_polygon,
)
from tcip_annotation.matching import box_ring


# ── box_iou ──────────────────────────────────────────────────────────────


def test_box_iou_perfect_overlap():
    b = BBox(0, 0, 100, 100)
    assert abs(box_iou(b, b) - 1.0) < 1e-6


def test_box_iou_no_overlap():
    b1 = BBox(0, 0, 50, 50)
    b2 = BBox(100, 100, 150, 150)
    assert box_iou(b1, b2) == 0.0


def test_box_iou_partial_overlap():
    b1 = BBox(0, 0, 100, 100)
    b2 = BBox(50, 50, 150, 150)
    # Intersection: 50×50=2500, Union: 10000+10000-2500=17500
    expected = 2500 / 17500
    assert abs(box_iou(b1, b2) - expected) < 1e-4


# ── box_ring ─────────────────────────────────────────────────────────────


def test_box_ring_corner_order():
    b = BBox(1.0, 2.0, 5.0, 8.0)
    assert box_ring(b) == [(1.0, 2.0), (5.0, 2.0), (5.0, 8.0), (1.0, 8.0)]


# ── polygon_iou ──────────────────────────────────────────────────────────


def test_polygon_iou_identical():
    from shapely.geometry import Polygon as SP
    g = SP([(0, 0), (100, 0), (100, 100), (0, 100)])
    assert abs(polygon_iou(g, g.area, g, g.area) - 1.0) < 1e-6


def test_polygon_iou_no_overlap():
    from shapely.geometry import Polygon as SP
    g1 = SP([(0, 0), (50, 0), (50, 50), (0, 50)])
    g2 = SP([(100, 100), (150, 100), (150, 150), (100, 150)])
    assert polygon_iou(g1, g1.area, g2, g2.area) == 0.0


# ── point_in_polygon ─────────────────────────────────────────────────────


def test_point_in_polygon_inside():
    poly = Polygon([[(0, 0), (100, 0), (100, 100), (0, 100)]])
    assert point_in_polygon(50, 50, poly) is True


def test_point_in_polygon_outside():
    poly = Polygon([[(0, 0), (100, 0), (100, 100), (0, 100)]])
    assert point_in_polygon(200, 200, poly) is False


def test_point_in_polygon_on_edge():
    """Points on the boundary are not inside (Shapely convention)."""
    poly = Polygon([[(0, 0), (100, 0), (100, 100), (0, 100)]])
    assert point_in_polygon(0, 50, poly) is False


# ── multi-ring (occlusion-split) instances ───────────────────────────────

# Two disjoint lobes of one instance: a bud split by a branch crossing in front of it.
LOBE_A = [(10.0, 10.0), (30.0, 10.0), (30.0, 50.0), (10.0, 50.0)]        # area 800
LOBE_B = [(70.0, 10.0), (120.0, 10.0), (120.0, 60.0), (70.0, 60.0)]     # area 2500
LOBES_AREA = 800.0 + 2500.0


def test_point_in_polygon_hits_a_ring_that_is_not_the_first():
    """Every ring is part of the instance, so a hit test consults all of them."""
    poly = Polygon([LOBE_A, LOBE_B])
    assert point_in_polygon(20, 30, poly) is True   # inside the first lobe
    assert point_in_polygon(95, 35, poly) is True   # inside the second lobe
    assert point_in_polygon(50, 30, poly) is False  # the occluded gap between them


def test_compute_matches_multi_ring_iou_spans_every_ring():
    """IoU is computed over the union of an instance's rings (a Shapely MultiPolygon)."""
    gt = [Annotation(subject="bud", geometry=Polygon([LOBE_A, LOBE_B]))]

    exact = [Annotation(subject="bud", geometry=Polygon([LOBE_A, LOBE_B]), score=0.9)]
    result = compute_matches(gt, exact)
    assert len(result["tp"]) == 1
    assert result["tp"][0]["iou"] == 1.0

    # A prediction that found only the first lobe leaves most of the instance unexplained: its IoU is
    # 800/3300, so at a 0.5 threshold it is an FP against an unmatched FN, not the perfect match a
    # first-ring-only comparison would report.
    partial = [Annotation(subject="bud", geometry=Polygon([LOBE_A]), score=0.9)]
    strict = compute_matches(gt, partial, iou_threshold=0.5)
    assert strict["tp"] == []
    assert len(strict["fp"]) == 1 and len(strict["fn"]) == 1

    lenient = compute_matches(gt, partial, iou_threshold=0.2)
    assert lenient["tp"][0]["iou"] == pytest.approx(800.0 / LOBES_AREA, abs=1e-4)


# ── compute_matches ──────────────────────────────────────────────────────


def test_compute_matches_basic():
    """One GT box, one matching pred, one FP pred."""
    gt = [Annotation(subject="bud", geometry=BBox(100, 100, 200, 200))]
    preds = [
        Annotation(subject="bud", geometry=BBox(105, 105, 195, 195), score=0.9),  # match → TP
        Annotation(subject="bud", geometry=BBox(400, 400, 450, 450), score=0.8),  # no match → FP
    ]
    result = compute_matches(gt, preds, iou_threshold=0.5)
    assert len(result["tp"]) == 1
    assert len(result["fp"]) == 1
    assert len(result["fn"]) == 0
    assert result["tp"][0]["class_name"] == "bud"
    assert result["tp"][0]["iou"] > 0.5


def test_compute_matches_fn():
    """One GT box, no predictions → 1 FN."""
    gt = [Annotation(subject="bud", geometry=BBox(100, 100, 200, 200))]
    result = compute_matches(gt, [])
    assert len(result["tp"]) == 0
    assert len(result["fp"]) == 0
    assert len(result["fn"]) == 1


def test_compute_matches_conf_filter():
    """Prediction below confidence threshold is excluded."""
    gt = [Annotation(subject="bud", geometry=BBox(100, 100, 200, 200))]
    preds = [Annotation(subject="bud", geometry=BBox(100, 100, 200, 200), score=0.1)]
    result = compute_matches(gt, preds, conf_threshold=0.5)
    assert len(result["tp"]) == 0
    assert len(result["fp"]) == 0
    assert len(result["fn"]) == 1  # GT unmatched because pred filtered out


def test_compute_matches_class_mismatch():
    """GT subject 'bud', pred subject 'leaf' → FN + FP."""
    gt = [Annotation(subject="bud", geometry=BBox(100, 100, 200, 200))]
    preds = [Annotation(subject="leaf", geometry=BBox(100, 100, 200, 200), score=0.9)]
    result = compute_matches(gt, preds)
    assert len(result["tp"]) == 0
    assert len(result["fp"]) == 1
    assert len(result["fn"]) == 1


def test_compute_matches_polygon():
    """Polygon GT + Polygon pred."""
    gt = [Annotation(subject="bud", geometry=Polygon([[(0, 0), (100, 0), (100, 100), (0, 100)]]))]
    preds = [Annotation(subject="bud",
                        geometry=Polygon([[(5, 5), (95, 5), (95, 95), (5, 95)]]), score=0.85)]
    result = compute_matches(gt, preds)
    assert len(result["tp"]) == 1
    assert result["tp"][0]["iou"] > 0.8


def test_compute_matches_iou_threshold():
    """Predictions with IoU below threshold don't match."""
    gt = [Annotation(subject="bud", geometry=BBox(0, 0, 100, 100))]
    preds = [Annotation(subject="bud", geometry=BBox(80, 80, 180, 180), score=0.9)]
    # IoU is low because overlap is small
    result = compute_matches(gt, preds, iou_threshold=0.5)
    # IoU ~ 400 / 19600 ≈ 0.02 → below 0.5
    assert len(result["tp"]) == 0
    assert len(result["fp"]) == 1
    assert len(result["fn"]) == 1
