"""What a sampled raster identity claims about how much of the raster it read.

``RasterIdentity.pixel_fraction`` is the coverage claim that travels beside the checksum: the share
of the raster's pixels the checksum was computed over, counted by each sampled window's area, on
grids whose cells are not square.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tcip_mcp.pipelines.raster_source import (
    open_raster,
    raster_content_identity,
    sample_windows,
)

if TYPE_CHECKING:
    from tcip_mcp.pipelines.raster_source import RasterIdentity


def _identity(
    path: Path, *, seed: int, window_size: int, max_windows: int
) -> RasterIdentity:
    """Open the three-band ``path`` and return its ``raster_content_identity``."""
    with open_raster(path, 3) as src:
        return raster_content_identity(
            src, seed=seed, window_size=window_size, max_windows=max_windows
        )


def _content_array(height: int, width: int) -> np.ndarray:
    rows = np.arange(height, dtype=np.uint16)[:, None]
    cols = np.arange(width, dtype=np.uint16)[None, :]
    return np.stack(
        [((rows * 7 + cols * 3 + k) % 251).astype(np.uint8) for k in range(3)],
        axis=-1,
    )


def test_a_sample_that_takes_every_grid_cell_claims_the_whole_raster(
    tmp_path: Path,
) -> None:
    """A sample that takes every grid cell claims the whole raster.

    A window budget at or above the grid's own cell count takes every cell, and the recorded
    coverage is the whole raster though the grid includes cells that are not square.
    """
    height, width, window_size = 20, 30, 16
    path = tmp_path / "content.npy"
    np.save(str(path), _content_array(height, width))

    cells = sample_windows(
        width, height, seed=3, window_size=window_size, max_windows=50
    )
    assert len(cells) > 1
    assert any(cell.width != cell.height for cell in cells), (
        "the grid must hold cells that are not square"
    )

    identity = _identity(path, seed=3, window_size=window_size, max_windows=50)
    assert identity.pixel_fraction == pytest.approx(1.0)


def test_a_partial_sample_claims_only_the_share_of_pixels_it_read(
    tmp_path: Path,
) -> None:
    """A partial sample claims only the share of pixels it read.

    A raster whose grid is three equal cells, two of them sampled, records two thirds: the
    coverage claim tracks the sampled area, not the window edge it was cut on.
    """
    height, width, window_size = 12, 48, 16
    path = tmp_path / "content.npy"
    np.save(str(path), _content_array(height, width))

    cells = sample_windows(
        width, height, seed=5, window_size=window_size, max_windows=2
    )
    assert len(cells) == 2
    assert all((cell.width, cell.height) == (16, 12) for cell in cells)

    identity = _identity(path, seed=5, window_size=window_size, max_windows=2)
    assert identity.pixel_fraction == pytest.approx(2 / 3)


def test_a_taller_than_wide_sample_claims_the_same_share_as_its_transpose(
    tmp_path: Path,
) -> None:
    """A taller-than-wide sample claims the same share as its transpose.

    Two rasters of the same pixel count sampled over the same grid geometry, one standing and one
    lying down, make the same coverage claim: the share is an area, so it cannot depend on which
    of a window's two edges is the long one.
    """
    window_size = 16
    tall_path, wide_path = tmp_path / "tall.npy", tmp_path / "wide.npy"
    np.save(str(tall_path), _content_array(48, 12))
    np.save(str(wide_path), _content_array(12, 48))

    tall = _identity(tall_path, seed=5, window_size=window_size, max_windows=2)
    wide = _identity(wide_path, seed=5, window_size=window_size, max_windows=2)

    assert (tall.width, tall.height) == (wide.height, wide.width)
    assert tall.pixel_fraction == pytest.approx(wide.pixel_fraction)
    assert tall.pixel_fraction < 1.0
