"""``state.Polygon`` is multi-ring, and ``bbox_of`` reads the whole instance.

A polygon annotation holds one or more simple closed contours. Most are one ring (a person draws one
contour); an occlusion-split instance (a bud behind a branch, a leaf crossed by a stem) is
genuinely more than one region, and every consumer that derives a box from a polygon must see all of
it. ``bbox_of`` is that derivation, so it is where a first-ring-only read would quietly shrink every
downstream box, area, crop and spatial-index entry.
"""

from __future__ import annotations

from tcip_annotation.state import BBox, Polygon, bbox_of

# Two disjoint lobes of ONE instance. Each lobe alone yields a strictly smaller box than the pair.
LOBE_A = [(10.0, 30.0), (30.0, 30.0), (30.0, 50.0), (10.0, 50.0)]
LOBE_B = [(70.0, 10.0), (90.0, 10.0), (90.0, 80.0), (70.0, 80.0)]


def test_bbox_of_a_single_ring_polygon_is_that_ring_s_box() -> None:
    b = bbox_of(Polygon([LOBE_A]))
    assert (b.x1, b.y1, b.x2, b.y2) == (10.0, 30.0, 30.0, 50.0)


def test_bbox_of_a_box_is_the_box_itself() -> None:
    box = BBox(1.0, 2.0, 3.0, 4.0)
    assert bbox_of(box) is box


def test_bbox_of_multi_ring_polygon_spans_every_ring() -> None:
    b = bbox_of(Polygon([LOBE_A, LOBE_B]))
    # x1/y2 come from the first lobe, x2/y1 from the second: no single ring produces this box, so
    # reading only one of them cannot pass.
    assert (b.x1, b.y1, b.x2, b.y2) == (10.0, 10.0, 90.0, 80.0)


def test_bbox_of_multi_ring_polygon_is_independent_of_ring_order() -> None:
    """Either ring order yields the same box spanning both lobes.

    Asserting only that the two agree passes trivially for a first-ring-only read too (both orders
    would just return their own first lobe's box), so the value is what discriminates.
    """
    a_first = bbox_of(Polygon([LOBE_A, LOBE_B]))
    b_first = bbox_of(Polygon([LOBE_B, LOBE_A]))
    assert (a_first.x1, a_first.y1, a_first.x2, a_first.y2) == (10.0, 10.0, 90.0, 80.0)
    assert (b_first.x1, b_first.y1, b_first.x2, b_first.y2) == (10.0, 10.0, 90.0, 80.0)


def test_bbox_of_grows_with_each_added_ring() -> None:
    """Every ring contributes: adding one can only widen the enclosing box, never be ignored."""
    one = bbox_of(Polygon([LOBE_A]))
    two = bbox_of(Polygon([LOBE_A, LOBE_B]))
    three = bbox_of(Polygon([LOBE_A, LOBE_B, [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)]]))

    assert (two.x2 - two.x1) > (one.x2 - one.x1)
    assert (two.y2 - two.y1) > (one.y2 - one.y1)
    assert (three.x1, three.y1) == (0.0, 0.0)
    assert (three.x2, three.y2) == (two.x2, two.y2)
