"""Offset reads through a region view: which pixels of the parent a windowed pass actually gets.

A calibration/holdout block is read as an ordinary windowed source through ``_RegionView``, so the
translation from the view's local coordinates into the parent's own space is the whole measurement
claim: a displaced translation serves real training pixels through the offset and nothing downstream
can tell. These fixtures give the block an origin whose row and column differ and pixel content that
differs along the two axes, so a translation that mixes the two axes lands on different pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from tcip_mcp.pipelines.raster_source import Rect, _RegionView, open_raster

# A raster taller than it is wide, and a block whose x0 and y0 differ: the geometry any read that
# confuses the two axes lands off.
_HEIGHT, _WIDTH = 40, 30
_BLOCK = Rect(7, 3, 27, 18)


def _axis_distinct_array(height: int, width: int, channels: int = 3) -> np.ndarray:
    """Pixel content that differs along the two axes, so no two distinct rectangles of the same
    shape hold the same pixels."""
    rows = np.arange(height, dtype=np.uint16)[:, None]
    cols = np.arange(width, dtype=np.uint16)[None, :]
    planes = [(rows + 0 * cols), (0 * rows + cols), (5 * rows + 3 * cols)]
    return np.stack([(p % 251).astype(np.uint8) for p in planes[:channels]], axis=-1)


def _parent_path(tmp_path: Path, backend: str, arr: np.ndarray) -> Path:
    if backend == "gdal":
        path = tmp_path / "block.tif"
        tifffile.imwrite(str(path), arr, photometric="rgb", rowsperstrip=4)
        return path
    path = tmp_path / "block.npy"
    np.save(str(path), arr)
    return path


@pytest.mark.parametrize("backend", ["gdal", "npy"])
def test_a_region_view_serves_the_block_its_rect_names(tmp_path: Path, backend: str) -> None:
    """A full-extent read through the view is the parent's own ``[y0:y1, x0:x1]`` slice for the
    rect the view was built over, whichever backend the parent is."""
    arr = _axis_distinct_array(_HEIGHT, _WIDTH)
    assert arr[0, 1, 2] != arr[1, 0, 2], "the fixture must differ along the two axes"

    with open_raster(_parent_path(tmp_path, backend, arr), 3) as parent:
        view = _RegionView(parent, _BLOCK)
        assert (view.height, view.width) == (15, 20)
        served = view.read_window(0, view.height, 0, view.width)

    assert np.array_equal(served, arr[_BLOCK.y0:_BLOCK.y1, _BLOCK.x0:_BLOCK.x1])


@pytest.mark.parametrize("backend", ["gdal", "npy"])
def test_a_region_view_window_lands_at_its_own_origin_plus_the_local_offset(
    tmp_path: Path, backend: str,
) -> None:
    """A window inside the view is the parent's slice offset by the rect's row origin in rows and
    its column origin in columns, never the two exchanged: a tiled pass over the block asks for
    local windows and every one of them must land inside the block it named."""
    arr = _axis_distinct_array(_HEIGHT, _WIDTH)
    local_y0, local_y1, local_x0, local_x1 = 2, 11, 4, 17

    with open_raster(_parent_path(tmp_path, backend, arr), 3) as parent:
        view = _RegionView(parent, _BLOCK)
        window = view.read_window(local_y0, local_y1, local_x0, local_x1)

    expected = arr[_BLOCK.y0 + local_y0:_BLOCK.y0 + local_y1,
                   _BLOCK.x0 + local_x0:_BLOCK.x0 + local_x1]
    assert window.shape == (local_y1 - local_y0, local_x1 - local_x0, 3)
    assert np.array_equal(window, expected)


def test_every_window_of_a_tiled_pass_over_a_region_view_stays_inside_the_block(
    tmp_path: Path,
) -> None:
    """The traversal a windowed inference pass actually performs: the block reassembled from its
    own local tiles is the block, so no tile of the pass read pixels from outside it."""
    arr = _axis_distinct_array(_HEIGHT, _WIDTH)
    tile = 6

    with open_raster(_parent_path(tmp_path, "gdal", arr), 3) as parent:
        view = _RegionView(parent, _BLOCK)
        rows = []
        for y0 in range(0, view.height, tile):
            y1 = min(y0 + tile, view.height)
            cols = [view.read_window(y0, y1, x0, min(x0 + tile, view.width))
                    for x0 in range(0, view.width, tile)]
            assert len(cols) > 1, "the pass must span more than one tile per row"
            rows.append(np.concatenate(cols, axis=1))
    assert len(rows) > 1, "the pass must span more than one row of tiles"

    assembled = np.concatenate(rows, axis=0)
    assert np.array_equal(assembled, arr[_BLOCK.y0:_BLOCK.y1, _BLOCK.x0:_BLOCK.x1])
