"""Pixel bounds for what the platform serves to a screen or writes as an agent-facing artifact, and
the one integer geometry every display-bound read is sized and planned with.

``VIZ_ARTIFACT_MAX_EDGE`` bounds agent-facing visualization artifacts, which are read back through
the model's own image reading. Source: agent image-reading practicality; deliberately smaller than
the display bound, which serves a human screen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tcip_mcp.pipelines.raster_source import Rect

VIZ_ARTIFACT_MAX_EDGE = 1024


def within_cap(width: int, height: int, max_pixels: int) -> bool:
    """Whether a ``width`` x ``height`` read is at most ``max_pixels`` pixels: the one test of a
    read or a cell against a display's area cap."""
    return width * height <= max_pixels


def fit_within(width: int, height: int, max_pixels: int, max_edge: int | None = None,
               max_scale: Fraction = Fraction(1)) -> tuple[int, int]:
    """A ``raster_source.scaled_size`` of a ``width`` x ``height`` rect under one scale of at most
    ``max_scale`` (never above 1, so never upscaled) that is at most ``max_pixels`` pixels and,
    when ``max_edge`` is given, has no edge longer than it.

    The scale is the continuous area bound rounded down, so the size is bounded but not always
    the largest admissible one. Either axis may limit; a one-pixel axis holds the other to
    ``max_pixels`` on its own.
    """
    from tcip_mcp.pipelines.raster_source import scaled_size

    scale = min(max_scale, Fraction(1),
                Fraction(math.isqrt(max_pixels * width * height), width * height))
    if max_edge is not None:
        scale = min(scale, Fraction(max_edge, max(width, height)))
    size = scaled_size(width, height, scale)
    if not within_cap(*size, max_pixels):
        size = scaled_size(width, height, min(scale, Fraction(max_pixels, max(width, height))))
    return size


@dataclass(frozen=True)
class ReadPlan:
    """One display read: the level it reads (0 native, ``i`` the ``i``-th overview level), the
    output size, and whether any level's window fit the area it was planned under."""

    level: int
    width: int
    height: int
    fits: bool


def plan_read(rect: Rect, raster_w: int, raster_h: int, level_dims: list[tuple[int, int]],
              max_pixels: int, max_edge: int | None = None) -> ReadPlan:
    """The read of ``rect`` of a ``raster_w`` x ``raster_h`` raster within ``max_pixels`` output
    pixels, over the overview levels of ``level_dims`` (each level's own ``(width, height)``,
    finest first).

    Every level is judged by the window the reader opens on it (``raster_source.level_window``).
    The output is :func:`fit_within` at a scale no finer than the finest level whose window fits
    ``max_pixels`` (the coarsest level when none does). The level read is the coarsest one whose
    window still holds the output on both axes, so the plan names the level its pixels come off.
    """
    from tcip_mcp.pipelines.raster_source import level_window

    windows = [level_window(rect, raster_w, raster_h, *dims)
               for dims in [(raster_w, raster_h), *level_dims]]
    fitting = [i for i, w in enumerate(windows) if within_cap(w.width, w.height, max_pixels)]
    chosen = windows[fitting[0] if fitting else len(windows) - 1]
    width, height = fit_within(
        rect.width, rect.height, max_pixels, max_edge,
        min(Fraction(chosen.width, rect.width), Fraction(chosen.height, rect.height)))
    level = max(i for i, w in enumerate(windows) if w.width >= width and w.height >= height)
    return ReadPlan(level, width, height, bool(fitting))


def level_dims(source, num_channels: int) -> list[tuple[int, int]]:
    """The overview levels the reader ``raster_source.open_raster(source, num_channels)`` serves
    a planned read or a level tile from (``overviews.overview_dims``), decided from headers
    alone: a TIFF the reader serves windowed through GDAL (``raster_source.opens_windowed``)
    reports its levels; every other source, a stacked TIFF the reader decodes whole included,
    has none."""
    from pathlib import Path

    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.overviews import overview_dims
    from tcip_mcp.pipelines.raster_source import opens_windowed

    if isinstance(source, BandGroupRef) or Path(source).suffix.lower() not in (".tif", ".tiff"):
        return []
    return overview_dims(source) if opens_windowed(source, num_channels) else []


def planned_read(raster, rect: Rect, plan: ReadPlan):
    """``raster.read_region`` of ``rect`` at the output size and off the level ``plan`` names,
    natively when that is the rect's own size."""
    if (plan.width, plan.height) == (rect.width, rect.height):
        return raster.read_region(rect)
    if plan.level:
        return raster.read_region(rect, target_size=(plan.width, plan.height), level=plan.level)
    return raster.read_region(rect, target_size=(plan.width, plan.height))
