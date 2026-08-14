"""What a sampled raster identity claims about how much of the raster it read.

``RasterIdentity.pixel_fraction`` is the coverage claim that travels beside the checksum: it says
what share of the raster's pixels the checksum was computed over, so a later reader knows whether
an identity match is a claim about the whole file or about a sample of it. The claim is only
usable if it counts each sampled window's area, which these fixtures exercise on grids whose cells
are not square.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tcip_mcp.pipelines.raster_source import raster_content_identity, sample_windows


def _content_array(height: int, width: int) -> np.ndarray:
    rows = np.arange(height, dtype=np.uint16)[:, None]
    cols = np.arange(width, dtype=np.uint16)[None, :]
    return np.stack([((rows * 7 + cols * 3 + k) % 251).astype(np.uint8) for k in range(3)],
                    axis=-1)


def test_a_sample_that_takes_every_grid_cell_claims_the_whole_raster(tmp_path: Path) -> None:
    """A window budget at or above the grid's own cell count reads every pixel exactly once, so
    the recorded coverage is the whole raster even though the grid's edge cells are short and
    none of them is square."""
    height, width, window_size = 20, 30, 16
    path = tmp_path / "content.npy"
    np.save(str(path), _content_array(height, width))

    cells = sample_windows(width, height, seed=3, window_size=window_size, max_windows=50)
    assert len(cells) > 1
    assert any(cell.width != cell.height for cell in cells), (
        "the grid must hold cells that are not square")

    identity = raster_content_identity(
        path, 3, seed=3, window_size=window_size, max_windows=50)
    assert identity.pixel_fraction == pytest.approx(1.0)


def test_a_partial_sample_claims_only_the_share_of_pixels_it_read(tmp_path: Path) -> None:
    """A raster whose grid is three equal cells, two of them sampled, records two thirds: the
    coverage claim tracks the sampled area, not the window edge it was cut on."""
    height, width, window_size = 12, 48, 16
    path = tmp_path / "content.npy"
    np.save(str(path), _content_array(height, width))

    cells = sample_windows(width, height, seed=5, window_size=window_size, max_windows=2)
    assert len(cells) == 2
    assert all((cell.width, cell.height) == (16, 12) for cell in cells)

    identity = raster_content_identity(
        path, 3, seed=5, window_size=window_size, max_windows=2)
    assert identity.pixel_fraction == pytest.approx(2 / 3)


def test_a_taller_than_wide_sample_claims_the_same_share_as_its_transpose(tmp_path: Path) -> None:
    """Two rasters of the same pixel count sampled over the same grid geometry, one standing and
    one lying down, make the same coverage claim: the share is an area, so it cannot depend on
    which of a window's two edges is the long one."""
    window_size = 16
    tall_path, wide_path = tmp_path / "tall.npy", tmp_path / "wide.npy"
    np.save(str(tall_path), _content_array(48, 12))
    np.save(str(wide_path), _content_array(12, 48))

    tall = raster_content_identity(tall_path, 3, seed=5, window_size=window_size, max_windows=2)
    wide = raster_content_identity(wide_path, 3, seed=5, window_size=window_size, max_windows=2)

    assert (tall.width, tall.height) == (wide.height, wide.width)
    assert tall.pixel_fraction == pytest.approx(wide.pixel_fraction)
    assert tall.pixel_fraction < 1.0
