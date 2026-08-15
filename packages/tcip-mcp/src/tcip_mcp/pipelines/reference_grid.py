"""Named reference grid over a raster's native pixel frame.

One geometry implementation for every consumer of a cell name: the agent's pointing
overlay (``overlay_reference_grid`` / ``segment_prompt``), the GUI's view-coverage
lattice, and any renderer drawing cell boundaries. Consumers exchange the serializable
geometry dict (:func:`grid_geometry`) and recompute cells deterministically from it via
:func:`reference_cells`; cell lists are never shipped between agent tool calls.

Cell names are spreadsheet-style: a bijective base-26 column letter plus a 1-based row
number ("B3"). The letter scheme is ``tcip_annotation.sam_wrapper``'s ``column_label`` /
``column_index``, imported rather than duplicated (tcip-mcp may depend on
tcip-annotation, never the reverse). Resolving a name back against a cell list is that
module's ``grid_to_rect`` (``grid_to_pixel`` for the cell's center): this module builds
the cells, that one lookup reads them, and every consumer of a cell name goes through it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tcip_annotation.sam_wrapper import column_label

from tcip_mcp.pipelines.data.tiling import compute_stride, tile_positions
from tcip_mcp.pipelines.display_bounds import DISPLAY_MAX_EDGE, VIZ_ARTIFACT_MAX_EDGE

POINTING_LEGIBLE_EDGE = 46
"""Smallest rendered cell edge, in artifact pixels, at which the overlay's cell labels
read comfortably. Measured, not designed: render ``render_grid_overlay`` on any textured
frame at the artifact bound with cell edges of 32, 40, 46, 51, 64, 79, 93 and 128 px and
read the artifacts back; at 32 the wide (two-letter, two-digit) labels collide and run
together, at 40 every label reads but covers most of its cell, and at 46 and above labels
read clearly with most of the cell content visible. The renders are session artifacts,
not repo files; the procedure above re-derives the value."""


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

    @property
    def center(self) -> tuple[float, float]:
        """The rect's center in native pixels."""
        return (self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0


def reference_cells(
    width: int,
    height: int,
    tile_size: int,
    overlap: float = 0.0,
    *,
    clamp: bool = False,
) -> list[Cell]:
    """Named square cells of ``tile_size`` native pixels over a ``width`` x ``height`` frame.

    Built on the training tiler's ``compute_stride``/``tile_positions``, so a reference
    grid and a training tiling of the same frame agree on tile origins by construction.
    ``clamp=False`` keeps the tiler's own semantics: rects may run past the extent and
    callers pad, exactly what training does. ``clamp=True`` clips each rect to the extent
    (edge cells truncate, never shift), so with ``overlap`` 0 the cells are an exact
    partition: every pixel belongs to exactly one cell.

    Known bounded behaviors, inherited from sharing the tiler: an extreme aspect ratio
    can leave sliver edge cells (a clamped edge cell keeps whatever remainder the extent
    leaves, down to one pixel), and the cell count is not always minimal (a dimension just
    over a multiple of ``tile_size`` adds one mostly-empty row or column).
    """
    if width < 1 or height < 1:
        raise ValueError(f"frame must be at least 1x1, got {width}x{height}")
    if tile_size < 1:
        raise ValueError(f"tile_size must be at least 1, got {tile_size}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap is a fraction of tile_size in [0, 1), got {overlap}")

    stride = compute_stride(tile_size, overlap)
    positions = tile_positions(height, width, tile_size, stride)
    col_of = {x: i for i, x in enumerate(sorted({tx for tx, _ in positions}))}
    row_of = {y: i for i, y in enumerate(sorted({ty for _, ty in positions}))}
    cells = []
    for tx, ty in positions:
        x1, y1 = tx + tile_size, ty + tile_size
        if clamp:
            x1, y1 = min(x1, width), min(y1, height)
        col, row = col_of[tx], row_of[ty]
        cells.append(Cell(name=f"{column_label(col)}{row + 1}", col=col, row=row,
                          x0=tx, y0=ty, x1=x1, y1=y1))
    return cells


def derive_coverage_tile_size(width: int, height: int) -> int:
    """Cell edge for the view-coverage lattice: the coarsest near-uniform square grid whose
    every cell, served at native resolution, fits one display-bounded serve.

    ``n = ceil(long_edge / DISPLAY_MAX_EDGE)`` cells along the long edge, so the returned
    edge is ``ceil(long_edge / n) <= DISPLAY_MAX_EDGE``; an image inside the display bound
    derives one cell spanning it. Deterministic in the image dims and the platform display
    bound (``display_bounds.DISPLAY_MAX_EDGE``), nothing else.
    """
    long_edge = max(width, height)
    n = math.ceil(long_edge / DISPLAY_MAX_EDGE)
    return math.ceil(long_edge / n)


def derive_large_raster_grid_tile_size(width: int, height: int, divisions: int = 16) -> int:
    """Cell edge for the view-coverage lattice over a large-raster (windowed) source, a fixed
    subdivision of the raster's own dimensions rather than :func:`derive_coverage_tile_size`'s
    display-resolution-derived edge.

    A large orthomosaic annotates canopy-scale objects (large and sparse), nothing like the dense
    small-object case the display-derived lattice was built for; applied there it produces far more,
    far smaller cells than the annotation task needs (an estimated ~2,500 cells inside a real
    mosaic's reserved calibration/test regions alone). ``divisions=16`` is a plain, documented
    default (source: docs/superpowers/specs/2026-08-10-region-completeness-batch-attestation-design.md,
    a fixed subdivision, not derived from object size or GT) -- deliberately provisional, to revisit
    once real large-orthomosaic annotation sessions exist to check the grain against. At real
    ValleyFarm dimensions (239921x141130) this derives a 16x10 lattice (160 cells), not the ~8,260
    the display-derived lattice would produce.
    """
    long_edge = max(width, height)
    return math.ceil(long_edge / divisions)


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
    """The serializable parameter tuple every consumer echoes: ``{width, height,
    tile_size, overlap, cols, rows}``.

    Cells are recomputed from this dict via :func:`reference_cells` (clamped or not, the
    caller's choice), never shipped between agent tool calls. ``tile_size`` is explicit: three
    derivations exist (:func:`derive_coverage_tile_size` for the ordinary coverage lattice,
    :func:`derive_large_raster_grid_tile_size` for a large-raster source,
    :func:`derive_pointing_tile_size` for the agent overlay), so a caller that has not chosen
    one has not chosen a grid, and a cell name means nothing without its grid.
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
