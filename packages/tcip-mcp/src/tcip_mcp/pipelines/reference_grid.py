"""Named reference grid over a raster's native pixel frame: cells recomputed from the serializable
geometry dict (:func:`grid_geometry`) by :func:`reference_cells`, each named spreadsheet-style, a
bijective base-26 column letter plus a 1-based row number ("B3", ``tcip_annotation.grid``'s
``column_label``)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tcip_annotation.grid import column_label

from tcip_mcp.pipelines.display_bounds import DISPLAY_MAX_EDGE, VIZ_ARTIFACT_MAX_EDGE

POINTING_LEGIBLE_EDGE = 46
"""Smallest rendered cell edge, in artifact pixels, at which the overlay's cell labels read
clearly with most of the cell visible, measured by rendering ``render_grid_overlay`` at the
artifact bound over cell edges 32 to 128: at 32 two-letter, two-digit labels collide, at 40 they
cover most of the cell."""

@dataclass(frozen=True)
class Cell:
    """One grid cell: its name plus its half-open native-pixel rect (``x0 <= x < x1``,
    ``y0 <= y < y1``, the same convention as ``raster_source.Rect``). ``col``/``row`` are
    0-based grid coordinates; ``name`` is ``column_label(col)`` + 1-based row."""

    name: str
    col: int
    row: int
    x0: int
    y0: int
    x1: int
    y1: int


def reference_cells(
    width: int,
    height: int,
    tile_size: int,
    overlap: float = 0.0,
    *,
    clamp: bool = False,
) -> list[Cell]:
    """Named square cells of ``tile_size`` native pixels over a ``width`` x ``height`` frame.

    Cell origins step by ``int(tile_size * (1 - overlap))`` (at least 1) from 0 while they lie
    inside the frame, so with ``overlap`` 0 they sit at multiples of ``tile_size``; this grid is
    its own, not the tiled-inference lattice. ``clamp=False`` lets a far-edge rect run past the
    extent; ``clamp=True`` clips each rect to the extent (edge cells truncate, never shift), so
    with ``overlap`` 0 the cells are an exact partition: every pixel belongs to exactly one cell.
    A clamped edge cell keeps whatever remainder the extent leaves, down to one pixel.
    """
    if width < 1 or height < 1:
        raise ValueError(f"frame must be at least 1x1, got {width}x{height}")
    if tile_size < 1:
        raise ValueError(f"tile_size must be at least 1, got {tile_size}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap is a fraction of tile_size in [0, 1), got {overlap}")

    stride = max(1, int(tile_size * (1.0 - overlap)))
    cells = []
    for row, ty in enumerate(range(0, height, stride)):
        for col, tx in enumerate(range(0, width, stride)):
            x1, y1 = tx + tile_size, ty + tile_size
            if clamp:
                x1, y1 = min(x1, width), min(y1, height)
            cells.append(Cell(name=f"{column_label(col)}{row + 1}", col=col, row=row,
                              x0=tx, y0=ty, x1=x1, y1=y1))
    return cells


def derive_serving_tile_size(width: int, height: int) -> int:
    """Cell edge for region serving: the coarsest near-uniform square grid whose every cell, served
    at native resolution, fits one display-bounded serve.

    ``n = ceil(long_edge / DISPLAY_MAX_EDGE)`` cells along the long edge, so the returned edge is
    ``ceil(long_edge / n) <= DISPLAY_MAX_EDGE``; an image inside the display bound derives one cell
    spanning it. Deterministic in the image dims and the platform display bound
    (``display_bounds.DISPLAY_MAX_EDGE``).
    """
    long_edge = max(width, height)
    n = math.ceil(long_edge / DISPLAY_MAX_EDGE)
    return math.ceil(long_edge / n)


def derive_pointing_tile_size(width: int, height: int) -> int:
    """Cell edge for the agent pointing grid, sized so labels stay legible on the overlay
    artifact.

    The overlay renders at most ``VIZ_ARTIFACT_MAX_EDGE`` on its long edge and never
    upscales, so the grain is chosen for that render: ``n = ceil(long_rendered /
    POINTING_LEGIBLE_EDGE)`` cells along the long edge, returned as ``ceil(long_native /
    n)``. Deterministic in the image dims, the artifact bound
    (``display_bounds.VIZ_ARTIFACT_MAX_EDGE``) and the measured legibility floor
    (:data:`POINTING_LEGIBLE_EDGE`), nothing else.
    """
    long_native = max(width, height)
    long_rendered = min(long_native, VIZ_ARTIFACT_MAX_EDGE)
    n = max(1, math.ceil(long_rendered / POINTING_LEGIBLE_EDGE))
    return math.ceil(long_native / n)


def grid_geometry(width: int, height: int, tile_size: int, overlap: float = 0.0) -> dict:
    """The serializable parameter tuple every consumer echoes: ``{width, height, tile_size,
    overlap, cols, rows}``.

    Cells are recomputed from this dict via :func:`reference_cells` (clamped or not, the caller's
    choice). ``tile_size`` is required: :func:`derive_serving_tile_size` and
    :func:`derive_pointing_tile_size` each derive one for their own grid.
    """
    cells = reference_cells(width, height, tile_size, overlap)
    return {
        "width": int(width),
        "height": int(height),
        "tile_size": int(tile_size),
        "overlap": float(overlap),
        "cols": max(c.col for c in cells) + 1,
        "rows": max(c.row for c in cells) + 1,
    }
