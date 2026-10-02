"""Tests for matching.py: the one matcher's pairs under a stated criterion, and its geometry
helpers."""

from __future__ import annotations

from tcip_annotation import Annotation, BBox, Polygon, point_in_polygon
from tcip_annotation.matching import box_ring, iou_matrix, pair_detections, pair_proposals

IOU = {"kind": "iou", "iou_threshold": 0.5}


# ── iou_matrix ───────────────────────────────────────────────────────────


def test_iou_matrix_perfect_none_and_partial_overlap():
    iou = iou_matrix([[0, 0, 100, 100]], [[0, 0, 100, 100], [100, 100, 150, 150],
                                          [50, 50, 150, 150]])
    assert abs(iou[0, 0] - 1.0) < 1e-6
    assert iou[0, 1] == 0.0
    # Intersection: 50×50=2500, Union: 10000+10000-2500=17500
    assert abs(iou[0, 2] - 2500 / 17500) < 1e-4


# ── box_ring ─────────────────────────────────────────────────────────────


def test_box_ring_corner_order():
    b = BBox(1.0, 2.0, 5.0, 8.0)
    assert box_ring(b) == [(1.0, 2.0), (5.0, 2.0), (5.0, 8.0), (1.0, 8.0)]


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


def test_point_in_polygon_hits_a_ring_that_is_not_the_first():
    """Every ring is part of the instance, so a hit test consults all of them."""
    poly = Polygon([LOBE_A, LOBE_B])
    assert point_in_polygon(20, 30, poly) is True   # inside the first lobe
    assert point_in_polygon(95, 35, poly) is True   # inside the second lobe
    assert point_in_polygon(50, 30, poly) is False  # the occluded gap between them


def test_a_multi_ring_instance_pairs_by_the_box_spanning_every_ring():
    """The matcher compares an instance by the box over all its rings, so a proposal that found
    one lobe of a two-lobe instance overlaps only part of it."""
    gt = [Annotation(subject="bud", geometry=Polygon([LOBE_A, LOBE_B]))]
    exact = [Annotation(subject="bud", geometry=Polygon([LOBE_A, LOBE_B]), score=0.9)]
    assert pair_proposals(gt, exact, IOU).pairs == [(0, 0)]

    # The first lobe's box covers 800 of the spanning box's 5500: no pair at 0.5.
    partial = [Annotation(subject="bud", geometry=Polygon([LOBE_A]), score=0.9)]
    assert pair_proposals(gt, partial, IOU).pairs == []


# ── the one matcher ──────────────────────────────────────────────────────


def test_a_proposal_pairs_with_the_annotation_it_overlaps_and_no_other():
    gt = [Annotation(subject="bud", geometry=BBox(100, 100, 200, 200))]
    preds = [
        Annotation(subject="bud", geometry=BBox(105, 105, 195, 195), score=0.9),  # pairs
        Annotation(subject="bud", geometry=BBox(400, 400, 450, 450), score=0.8),  # pairs nothing
    ]
    m = pair_proposals(gt, preds, IOU)
    assert (m.pairs, m.unpaired, m.missed) == ([(0, 0)], [1], [])
    assert pair_proposals(gt, [], IOU).missed == [0]


def test_the_confidence_floor_leaves_a_low_scoring_proposal_out_of_every_count():
    gt = [Annotation(subject="bud", geometry=BBox(100, 100, 200, 200))]
    preds = [Annotation(subject="bud", geometry=BBox(105, 105, 195, 195), score=0.2),
             Annotation(subject="bud", geometry=BBox(400, 400, 450, 450), score=0.1)]
    m = pair_proposals(gt, preds, IOU, conf_threshold=0.5)
    assert (m.pairs, m.unpaired, m.missed) == ([], [], [0])


def test_a_proposal_of_another_subject_pairs_nothing():
    gt = [Annotation(subject="bud", geometry=BBox(100, 100, 200, 200))]
    preds = [Annotation(subject="leaf", geometry=BBox(100, 100, 200, 200), score=0.9)]
    m = pair_proposals(gt, preds, IOU)
    assert (m.pairs, m.unpaired, m.missed) == ([], [0], [0])


def test_a_polygon_pairs_by_its_box():
    gt = [Annotation(subject="bud", geometry=Polygon([[(0, 0), (100, 0), (100, 100), (0, 100)]]))]
    preds = [Annotation(subject="bud",
                        geometry=Polygon([[(5, 5), (95, 5), (95, 95), (5, 95)]]), score=0.85)]
    assert pair_proposals(gt, preds, IOU).pairs == [(0, 0)]


def test_the_criterion_limit_is_inclusive_and_a_lower_overlap_pairs_nothing():
    """A pair exactly at the stated IoU pairs; one below it does not."""
    gt = [{"bbox": [0.0, 0.0, 100.0, 100.0], "iscrowd": False}]
    at_half = [{"bbox": [0.0, 0.0, 50.0, 100.0]}]          # IoU exactly 0.5
    below = [{"bbox": [80.0, 80.0, 100.0, 100.0]}]         # IoU ~ 0.02
    assert pair_detections(gt, at_half, IOU).pairs == [(0, 0)]
    m = pair_detections(gt, below, IOU)
    assert (m.pairs, m.unpaired, m.ignored, m.missed) == ([], [0], set(), [0])


def test_an_unpaired_detection_centered_in_a_crowd_region_is_ignored():
    """COCO's crowd rule as evaluation states it: a detection left unpaired whose center lies in
    a crowd region's box is neither true nor false."""
    gt = [{"bbox": [0.0, 0.0, 50.0, 50.0], "iscrowd": True},
          {"bbox": [100.0, 100.0, 20.0, 20.0], "iscrowd": False}]
    dt = [{"bbox": [100.0, 100.0, 20.0, 20.0]}, {"bbox": [10.0, 10.0, 10.0, 10.0]},
          {"bbox": [300.0, 300.0, 10.0, 10.0]}]
    m = pair_detections(gt, dt, IOU)
    assert m.pairs == [(1, 0)]
    assert m.ignored == {1}
    assert m.unpaired == [2]
    assert m.missed == []


def test_a_center_match_pairs_within_its_tolerance():
    gt = [{"bbox": [0.0, 0.0, 10.0, 10.0], "iscrowd": False}]
    near = [{"bbox": [3.0, 0.0, 10.0, 10.0]}]   # centers 3 px apart
    far = [{"bbox": [6.0, 0.0, 10.0, 10.0]}]    # centers 6 px apart
    center = {"kind": "center_match", "tolerance": 5.0}
    assert pair_detections(gt, near, center).pairs == [(0, 0)]
    assert pair_detections(gt, far, center).pairs == []
