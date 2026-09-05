"""Tier-A derivations on asymmetric data: sparse class ids, open boxes, skewed spacing.

A derivation that reads a symmetric artifact (dense ids, square boxes, evenly spaced objects)
cannot show which statistic it actually computes, because every candidate statistic agrees there.
These fixtures are built so a wrong statistic and the right one return different numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from tcip_mcp.pipelines.derivations import (
    char_sizes_from_boxes,
    derive_block_scale_px,
    derive_localization_tolerance_frac,
    gt_aspect_ratios,
    num_classes_from_distribution,
)


def test_num_classes_sizes_the_head_for_the_highest_id_not_the_ids_present():
    """A label set whose observed ids are sparse (a class absent from this split, or ids the
    registry assigned 0 and 3) still has to size the model's output head for the highest id it can
    emit: a head built for the number of ids present indexes past its own last logit."""
    assert num_classes_from_distribution({0: 5, 3: 2}) == 4
    assert num_classes_from_distribution({2: 7}) == 3
    assert num_classes_from_distribution({0: 1, 1: 1, 5: 3}) == 6


@pytest.mark.parametrize("boxes,label_ratio", [
    ([(10.0, 40.0)] * 20, 4.0),   # uniformly tall
    ([(40.0, 10.0)] * 20, 0.25),  # uniformly wide
])
def test_anchor_ratios_keep_square_coverage_on_a_uniformly_open_dataset(boxes, label_ratio):
    """The derived anchor set spans the GT's own shape distribution and still covers a square-ish
    object. A class whose p10 and p90 both sit away from 1.0 would otherwise leave anchors that
    match nothing near square, so an instance closer to square than the rest of its class loses
    anchor coverage entirely."""
    ratios = gt_aspect_ratios(boxes)
    assert ratios is not None and len(ratios) >= 2
    assert 1.0 in ratios
    assert label_ratio in ratios
    assert min(ratios) <= 1.0 <= max(ratios)


def test_localization_tolerance_normalizes_by_the_characteristic_size_its_siblings_share():
    """The center-match tolerance is a fraction of ``sqrt(w*h)``, the same characteristic size
    ``char_sizes_from_boxes`` (and through it the localization kind and the IoU threshold) measures.
    Two datasets with the same characteristic size and the same neighbor spacing must therefore
    derive the same tolerance, however differently their boxes are shaped: a size measure that
    reads the sides instead splits them apart and the tolerance stops being comparable to the
    criterion that selected it."""
    square = [[(0, 0, 30, 30), (40, 0, 30, 30), (80, 0, 30, 30)]]
    open = [[(0, 0, 10, 90), (40, 0, 10, 90), (80, 0, 10, 90)]]

    assert float(np.mean(char_sizes_from_boxes(square))) == pytest.approx(30.0)
    assert float(np.mean(char_sizes_from_boxes(open))) == pytest.approx(30.0)

    tol_square = derive_localization_tolerance_frac(square)
    tol_open = derive_localization_tolerance_frac(open)
    assert tol_square is not None and tol_open is not None
    # 40px between neighboring centers, margin_frac 0.5, characteristic size 30.
    assert tol_square == pytest.approx(40 * 0.5 / 30)
    assert tol_open == pytest.approx(tol_square)
    assert 0.1 < tol_square < 0.75  # strictly inside the clamp, so neither end flattens the pair


def test_block_scale_takes_the_typical_spacing_not_one_dragged_up_by_isolated_objects():
    """Calibration blocks are sized by the median nearest-neighbor spacing, so the isolated
    objects real GT always carries cannot widen them past what the data supports. On a skewed
    spacing distribution (a dense row plus a far-apart pair) the mean sits well above every
    spacing the dense majority actually has."""
    dense = [(x, 0, 10, 10) for x in range(0, 600, 100)]      # 6 boxes, 100px apart
    isolated = [(5000, 0, 10, 10), (5900, 0, 10, 10)]         # 900px apart, far from the row
    px, source = derive_block_scale_px(
        tile_size=50, gt_boxes_per_image=[dense + isolated])
    assert "GT object-spacing" in source
    assert px == 100
    assert px > 50  # above the tile_size floor, so the floor is not what produced this number
