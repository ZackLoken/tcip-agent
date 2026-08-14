"""Reading a raster that is larger than one block: the traversal and the cache budget behind it.

An orthomosaic is served window by window out of GDAL's block cache rather than decoded whole, so
two facts have to hold at a size where that machinery is actually engaged: a pass whose windows are
not aligned to the file's own internal tile grid still reassembles the file exactly, and the block
cache each process commits is the budget this module hands it. The fixture here is a small stand-in
for a real mosaic, sized to span many internal tiles on both axes while staying fast to build.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from tcip_mcp.pipelines import raster_source
from tcip_mcp.pipelines.raster_source import open_raster

# Wider than it is tall, and a multiple of neither the tile size nor the traversal step, so edge
# windows are short and interior windows straddle tile boundaries on both axes.
_WIDTH, _HEIGHT = 1280, 896
_TILE = 256
_STEP_X, _STEP_Y = 300, 180


def _mosaic_array(height: int, width: int) -> np.ndarray:
    """Content that varies along both axes and repeats on no period the tile grid shares, so a
    window served from the wrong block cannot match the right one."""
    rows = np.arange(height, dtype=np.int64)[:, None]
    cols = np.arange(width, dtype=np.int64)[None, :]
    planes = [(rows * 13 + cols * 7), (rows * 3 + cols * 29), (rows * 41 + cols * 11)]
    return np.stack([(p % 251).astype(np.uint8) for p in planes], axis=-1)


@pytest.fixture
def restored_gdal_cache():
    """The block cache budget is process-global; each test that sets it puts back what it found."""
    from rasterio.env import get_gdal_config, set_gdal_config

    original = get_gdal_config("GDAL_CACHEMAX")
    yield
    if original is not None:
        set_gdal_config("GDAL_CACHEMAX", int(original))


@pytest.fixture(scope="module")
def multi_tile_raster(tmp_path_factory) -> tuple[Path, np.ndarray]:
    arr = _mosaic_array(_HEIGHT, _WIDTH)
    path = tmp_path_factory.mktemp("mosaic") / "mosaic.tif"
    tifffile.imwrite(str(path), arr, photometric="rgb", tile=(_TILE, _TILE))
    return path, arr


def test_the_fixture_spans_many_internal_tiles_on_both_axes(multi_tile_raster) -> None:
    """The property every other test in this file rests on: the file really is internally tiled,
    with several tiles across and several down, so a read touches more than one block."""
    path, arr = multi_tile_raster
    with tifffile.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        assert page.is_tiled
        assert (page.tilewidth, page.tilelength) == (_TILE, _TILE)
    assert arr.shape[1] // _TILE >= 4
    assert arr.shape[0] // _TILE >= 3


def test_a_traversal_whose_windows_straddle_tile_boundaries_reassembles_the_raster(
    multi_tile_raster,
) -> None:
    """A windowed pass over a multi-tile raster, stepped so that no window edge falls on a tile
    edge, returns the same pixels a whole decode of the same file does."""
    path, arr = multi_tile_raster
    assert _STEP_X % _TILE and _STEP_Y % _TILE, "the pass must not be aligned to the tile grid"

    rows = []
    with open_raster(path, 3) as src:
        assert isinstance(src, raster_source.GdalSource)
        assert (src.width, src.height) == (_WIDTH, _HEIGHT)
        for y0 in range(0, src.height, _STEP_Y):
            y1 = min(y0 + _STEP_Y, src.height)
            cols = [src.read_window(y0, y1, x0, min(x0 + _STEP_X, src.width))
                    for x0 in range(0, src.width, _STEP_X)]
            assert len(cols) > 2, "each row of the pass must span several windows"
            rows.append(np.concatenate(cols, axis=1))
    assert len(rows) > 2, "the pass must span several rows of windows"

    assembled = np.concatenate(rows, axis=0)
    assert assembled.shape == (_HEIGHT, _WIDTH, 3)
    assert np.array_equal(assembled, arr)
    assert np.array_equal(assembled, tifffile.imread(str(path)))


def test_an_interior_window_of_a_multi_tile_raster_is_its_own_pixels(multi_tile_raster) -> None:
    """One window taken from the middle of the raster, spanning four internal tiles and sharing an
    edge with none of them, is exactly the slice it names."""
    path, arr = multi_tile_raster
    y0, y1, x0, x1 = 470, 790, 630, 1010

    with open_raster(path, 3) as src:
        window = src.read_window(y0, y1, x0, x1)

    assert x0 % _TILE and y0 % _TILE and x1 % _TILE and y1 % _TILE
    assert window.shape == (y1 - y0, x1 - x0, 3)
    assert np.array_equal(window, arr[y0:y1, x0:x1])


def test_a_worker_share_scales_the_block_cache_budget_it_commits(
    monkeypatch, restored_gdal_cache,
) -> None:
    """A process holding one of several peer caches (a DataLoader worker) commits its share of the
    budget, so the peers together commit what one process alone would, never that much each."""
    monkeypatch.setattr(raster_source, "_memory_budget_bytes", lambda: 8 * 1024 ** 3)

    raster_source.configure_gdal_cache()
    whole = raster_source.gdal_cache_bytes()
    raster_source.configure_gdal_cache(share=1 / 4)
    quarter = raster_source.gdal_cache_bytes()

    assert whole > raster_source._GDAL_CACHEMAX_BYTES_FLOOR
    assert quarter * 4 == whole


def test_a_host_too_small_to_budget_still_commits_an_unambiguous_byte_count(
    monkeypatch, restored_gdal_cache,
) -> None:
    """GDAL reads its cache setting as megabytes below a threshold and as bytes at or above it, so
    even a budget far under that threshold is committed at the floor that keeps it bytes."""
    monkeypatch.setattr(raster_source, "_memory_budget_bytes", lambda: 4096)

    raster_source.configure_gdal_cache()

    assert raster_source.gdal_cache_bytes() >= raster_source._GDAL_CACHEMAX_BYTES_FLOOR
