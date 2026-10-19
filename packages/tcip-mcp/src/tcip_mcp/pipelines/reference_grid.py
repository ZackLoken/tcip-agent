"""Named reference grid over a raster's native pixel frame.

Cells are recomputed from the serializable geometry dict (:func:`grid_geometry`) by
:func:`reference_cells`, each named spreadsheet-style, a bijective base-26 column letter plus a
1-based row number ("B3", ``tcip_annotation.grid``'s ``column_label``).
"""

from __future__ import annotations

import math

from tcip_annotation.grid import Cell, column_label
from tcip_mcp.pipelines.display_bounds import VIZ_ARTIFACT_MAX_EDGE, within_cap

POINTING_LEGIBLE_EDGE = 46
"""Smallest rendered cell edge, in artifact pixels, at which the overlay's cell labels read
clearly with most of the cell visible, measured by rendering ``render_grid_overlay`` at the
artifact bound over cell edges 32 to 128: at 32 two-letter, two-digit labels collide, at 40 they
cover most of the cell."""


def reference_cells(
    width: int,
    height: int,
    tile_size: int,
    overlap: float,
    *,
    clamp: bool = False,
) -> list[Cell]:
    """Return named square cells of ``tile_size`` native pixels over a ``width`` x ``height`` frame.

    Cell origins step by ``int(tile_size * (1 - overlap))`` (at least 1) from 0 while they lie
    inside the frame, so with ``overlap`` 0 they sit at multiples of ``tile_size``.
    ``clamp=False`` lets a far-edge rect run past the
    extent; ``clamp=True`` clips each rect to the extent (edge cells truncate, never shift), so
    with ``overlap`` 0 the cells are an exact partition: every pixel belongs to exactly one cell.
    A clamped edge cell keeps whatever remainder the extent leaves, down to one pixel.
    """
    if width < 1 or height < 1:
        raise ValueError(f"frame must be at least 1x1, got {width}x{height}")
    if tile_size < 1:
        raise ValueError(f"tile_size must be at least 1, got {tile_size}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(
            f"overlap is a fraction of tile_size in [0, 1), got {overlap}"
        )

    stride = max(1, int(tile_size * (1.0 - overlap)))
    cells = []
    for row, ty in enumerate(range(0, height, stride)):
        for col, tx in enumerate(range(0, width, stride)):
            x1, y1 = tx + tile_size, ty + tile_size
            if clamp:
                x1, y1 = min(x1, width), min(y1, height)
            cells.append(
                Cell(
                    name=f"{column_label(col)}{row + 1}",
                    col=col,
                    row=row,
                    x0=tx,
                    y0=ty,
                    x1=x1,
                    y1=y1,
                )
            )
    return cells


def derive_serving_tile_size(
    width: int, height: int, max_pixels: int, max_edge: int | None = None
) -> int:
    """Derive the cell edge for region serving, a near-uniform square grid within the caps.

    Every cell, served at its grid's own resolution and clipped to the extent
    (``reference_cells(..., clamp=True)``), is at most ``max_pixels`` pixels and, when
    ``max_edge`` is given (the edge limit of the encoding the cells are served in), no edge longer
    than it. This grid is not claimed to be the coarsest. The edge bound is the long edge for an
    image of at most ``max_pixels``, ``isqrt(max_pixels)`` for any other, and never past
    ``max_edge``. The grid has ``n = ceil(long_edge / bound)`` cells along the long edge, an edge
    of ``ceil(long_edge / n) <= bound``. Under the ``isqrt(max_pixels)`` bound even an unclipped
    cell is within the cap; under the long-edge bound an unclipped square can exceed it, and the
    clipped cell lies inside the extent, which is within the cap. An image within both bounds is
    one cell spanning it.
    """
    long_edge = max(width, height)
    bound = (
        long_edge
        if within_cap(width, height, max_pixels)
        else math.isqrt(max_pixels)
    )
    if max_edge is not None:
        bound = min(bound, max_edge)
    n = math.ceil(long_edge / bound)
    return math.ceil(long_edge / n)


def derive_pointing_tile_size(width: int, height: int) -> int:
    """Derive the cell edge for the agent pointing grid, legible on the overlay artifact.

    The edge is sized so labels stay legible. With ``long_rendered`` the long edge capped at
    ``VIZ_ARTIFACT_MAX_EDGE`` and never upscaled, ``n = ceil(long_rendered /
    POINTING_LEGIBLE_EDGE)`` cells lie along the long edge, returned as ``ceil(long_native / n)``.
    Deterministic in the image dims, the artifact bound (``display_bounds.VIZ_ARTIFACT_MAX_EDGE``)
    and the measured legibility floor (:data:`POINTING_LEGIBLE_EDGE`), nothing else.
    """
    long_native = max(width, height)
    long_rendered = min(long_native, VIZ_ARTIFACT_MAX_EDGE)
    n = max(1, math.ceil(long_rendered / POINTING_LEGIBLE_EDGE))
    return math.ceil(long_native / n)


def grid_geometry(
    width: int, height: int, tile_size: int, overlap: float
) -> dict:
    """Return the serializable grid parameters.

    They are ``{width, height, tile_size, overlap, cols, rows}``. ``tile_size`` and ``overlap``
    are required.
    """
    cells = reference_cells(width, height, tile_size, overlap)
    return {
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "overlap": overlap,
        "cols": max(c.col for c in cells) + 1,
        "rows": max(c.row for c in cells) + 1,
    }
