"""The integer geometry a display read is sized and planned with (``display_bounds``), and its
agreement with the geometry the raster reader admits (``raster_source``)."""

from __future__ import annotations

import pytest

from tcip_mcp.pipelines.display_bounds import fit_within, plan_read
from tcip_mcp.pipelines.raster_source import Rect, _check_target_size


@pytest.mark.parametrize("width,height,max_pixels,max_edge", [
    (4000, 3006, 2_073_600, None),
    (1920, 1086, 300_000, None),
    (1, 100_000, 20_000, None),
    (2, 100_000, 20_000, None),
    (100_000, 1, 20_000, None),
    (5, 101, 32, None),
    (100_000, 1, 8_294_400, 65_500),
    (239921, 141130, 3_686_400, 65_500),
])
def test_every_fitted_size_is_within_its_bounds_and_admitted_by_the_reader(
    width, height, max_pixels, max_edge,
):
    out_w, out_h = fit_within(width, height, max_pixels, max_edge)
    assert out_w * out_h <= max_pixels
    assert max_edge is None or max(out_w, out_h) <= max_edge
    assert _check_target_size(Rect(0, 0, width, height), (out_w, out_h)) == (out_w, out_h)


def test_the_plan_reads_the_finest_level_that_fits_and_names_it():
    plan = plan_read(Rect(0, 0, 1000, 1000), 1000, 1000, [(500, 500), (250, 250)], 300_000)
    assert (plan.level, plan.width, plan.height, plan.fits) == (1, 500, 500, True)


def test_with_no_level_fitting_the_plan_reads_the_coarsest_within_the_cap():
    plan = plan_read(Rect(0, 0, 1000, 1000), 1000, 1000, [(500, 500), (250, 250)], 10_000)
    assert (plan.level, plan.width, plan.height, plan.fits) == (2, 100, 100, False)


def test_an_edge_limited_plan_names_the_coarser_level_it_comes_off():
    """A one-row output far below the fitting level's resolution reads the coarsest level still
    holding it, not the finer one whose window fit."""
    plan = plan_read(Rect(0, 0, 5000, 64), 5000, 64, [(2500, 32), (1250, 16), (625, 8)],
                     1024 * 1024, 1024)
    assert (plan.width, plan.height) == (1024, 13)
    assert plan.level == 2


LEVELS = [(1024, 1024), (512, 512)]
"""The levels of a 2048-pixel square raster built down to 512."""


@pytest.mark.parametrize("rect,max_pixels,owed", [
    (Rect(0, 0, 1, 2048), 100, (2, 1, 100, False)),
    (Rect(0, 0, 1, 2048), 600, (2, 1, 512, True)),
    (Rect(1, 1, 513, 513), 65_536, (2, 129, 129, True)),
])
def test_a_crop_is_planned_on_the_window_the_reader_opens(rect, max_pixels, owed):
    """A one-pixel-wide crop's window on every level is one pixel wide, never a fraction of one,
    and an offset crop's window takes its origin rounded down and its end up, so the plan picks
    the level a reader can open within the cap and names it."""
    from tcip_mcp.pipelines.raster_source import level_window

    plan = plan_read(rect, 2048, 2048, LEVELS, max_pixels)
    assert (plan.level, plan.width, plan.height, plan.fits) == owed
    window = level_window(rect, 2048, 2048, *LEVELS[plan.level - 1])
    assert window.width >= plan.width and window.height >= plan.height
    assert not plan.fits or window.width * window.height <= max_pixels
