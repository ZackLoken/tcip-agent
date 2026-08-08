"""The raster reading layer: which backend serves which source, and what each one guarantees.

Covers the factory's dispatch, the windowed strip backend's pixel and caching behavior, the
copy-on-return and bounds contracts every backend shares, the strip-cache capacity derivation, and
the process-local pool of open sources.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from tcip_mcp.pipelines import raster_source
from tcip_mcp.pipelines.raster_source import (
    BandGroupSource,
    NpySource,
    NpzSource,
    PhotographicSource,
    Rect,
    StripTiffSource,
    TiffWholeSource,
    UnsupportedRasterLayout,
    derive_strip_cache_capacity,
    open_raster,
)


@pytest.fixture(autouse=True)
def _empty_source_pool():
    """Each test starts and ends with an empty pool: it is process-global state."""
    raster_source.close_source_pool()
    yield
    raster_source.close_source_pool()


def _distinctive_array(height: int, width: int, channels: int = 3) -> np.ndarray:
    """Each pixel encodes its own row/col so a returned sub-array can be checked exactly."""
    arr = np.zeros((height, width, channels), dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            arr[row, col, 0] = row % 256
            arr[row, col, 1] = col % 256
            if channels > 2:
                arr[row, col, 2] = (row + col) % 256
    return arr


def _write_striped_tiff(path: Path, arr: np.ndarray, *, rowsperstrip: int) -> None:
    extrasamples = ["unassalpha"] * (arr.shape[-1] - 3) if arr.shape[-1] > 3 else None
    kwargs = {"photometric": "rgb", "rowsperstrip": rowsperstrip}
    if extrasamples:
        kwargs["extrasamples"] = extrasamples
    tifffile.imwrite(str(path), arr, **kwargs)


# ── One source per backend ───────────────────────────────────────────────


def _photographic(tmp_path: Path):
    from PIL import Image

    path = tmp_path / "photo.png"
    Image.fromarray(_distinctive_array(12, 9)).save(path)
    return path, 3


def _strip_tiff(tmp_path: Path):
    path = tmp_path / "strip.tif"
    _write_striped_tiff(path, _distinctive_array(23, 17), rowsperstrip=4)
    return path, 3


def _whole_tiff(tmp_path: Path):
    """A channel-last 5-band raster: tifffile stores it one row-block per page, which the strip
    backend refuses, so the factory sends it to the whole decode."""
    path = tmp_path / "multipage.tif"
    tifffile.imwrite(str(path), _distinctive_array(20, 14, channels=5))
    return path, 5


def _npy(tmp_path: Path):
    path = tmp_path / "bands.npy"
    np.save(str(path), _distinctive_array(18, 11, channels=5))
    return path, 5


def _npz(tmp_path: Path):
    path = tmp_path / "bands.npz"
    np.savez(str(path), bands=_distinctive_array(18, 11, channels=5))
    return path, 5


def _band_group(tmp_path: Path):
    from tcip_mcp.pipelines.data.band_groups import read_band_group_manifest, write_band_group_manifest

    d = tmp_path / "grouped"
    d.mkdir()
    green = d / "cap_G.tif"
    red = d / "cap_R.tif"
    tifffile.imwrite(str(green), np.full((8, 8), 111, dtype=np.uint16))
    tifffile.imwrite(str(red), np.arange(64, dtype=np.uint16).reshape(8, 8))
    manifest = write_band_group_manifest(d, "cap", {"Green": green, "Red": red})
    return read_band_group_manifest(manifest), 2


_BACKENDS = {
    "band_group": (_band_group, BandGroupSource),
    "npy": (_npy, NpySource),
    "npz": (_npz, NpzSource),
    "photographic": (_photographic, PhotographicSource),
    "strip_tiff": (_strip_tiff, StripTiffSource),
    "tiff_whole": (_whole_tiff, TiffWholeSource),
}


# ── Factory dispatch ─────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(_BACKENDS))
def test_open_raster_picks_the_backend_each_source_needs(tmp_path: Path, name: str) -> None:
    build, expected = _BACKENDS[name]
    source, num_channels = build(tmp_path)
    with open_raster(source, num_channels) as src:
        assert isinstance(src, expected)


def test_a_photographic_extension_at_a_count_pil_has_no_mode_for_refuses(tmp_path: Path) -> None:
    """The same refusal, wording included, a caller asking for band data out of a photograph gets:
    the count is a routing hint, and there is no PIL mode for 5 channels."""
    source, _ = _photographic(tmp_path)
    with pytest.raises(ValueError, match=r"Cannot load a 5-channel image from '\.png'"):
        open_raster(source, 5)


@pytest.mark.parametrize("layout", ["multipage", "tiled", "planar"])
def test_a_tiff_layout_the_strip_backend_refuses_still_reads_whole(tmp_path: Path, layout: str) -> None:
    """Every layout ``StripTiffSource`` will not serve routes to the whole decode and returns
    exactly what ``tifffile.imread`` does, rather than being refused outright."""
    arr = _distinctive_array(32, 40, channels=5 if layout == "multipage" else 3)
    path = tmp_path / f"{layout}.tif"
    if layout == "multipage":
        tifffile.imwrite(str(path), arr)
    elif layout == "tiled":
        tifffile.imwrite(str(path), arr, photometric="rgb", tile=(16, 16))
    else:
        tifffile.imwrite(str(path), arr, photometric="rgb", planarconfig="separate", rowsperstrip=8)

    with open_raster(path, arr.shape[-1]) as src:
        assert isinstance(src, TiffWholeSource)
        region, _spec = src.read_region(Rect(0, 0, src.width, src.height))
    assert np.array_equal(region, tifffile.imread(str(path)))


def test_a_shape_the_whole_decode_would_transpose_is_not_served_by_strips(tmp_path: Path) -> None:
    """``load_multiband`` reads a channel-first-looking shape into channel-last order and
    ``image_dimensions`` applies the same heuristic to the header, so a raster the heuristic fires
    on has to go whole; served by strips it would report one frame and measure as another."""
    from tcip_mcp.pipelines.image_utils import image_dimensions

    path = tmp_path / "three_row.tif"
    arr = _distinctive_array(3, 20, channels=4)
    tifffile.imwrite(str(path), arr, photometric="rgb", extrasamples=["unassalpha"], rowsperstrip=1)

    with StripTiffSource(path) as strip:
        assert (strip.height, strip.num_channels) == (3, 4)  # the strip backend would serve it

    with open_raster(path, 3) as src:
        assert isinstance(src, TiffWholeSource)
        region, _spec = src.read_region(Rect(0, 0, src.width, src.height))
    assert region.shape == (20, 4, 3)
    assert image_dimensions(path, 3) == (4, 20)

    # At the count the heuristic leaves alone, the windowed backend serves it as it always did.
    with open_raster(path, 4) as src:
        assert isinstance(src, StripTiffSource)


def test_every_read_here_is_served_at_full_resolution(tmp_path: Path) -> None:
    """No backend in this module serves an overview, so every ``ReadSpec`` says so."""
    source, num_channels = _strip_tiff(tmp_path)
    with open_raster(source, num_channels) as src:
        _region, spec = src.read_region(Rect(0, 0, 4, 4))
    assert (spec.backend, spec.scale, spec.resample) == ("strip_tiff", 1.0, None)


# ── The contracts every backend shares ───────────────────────────────────


@pytest.mark.parametrize("name", sorted(_BACKENDS))
def test_a_returned_region_is_a_copy_a_caller_may_mutate(tmp_path: Path, name: str) -> None:
    """Mutating a returned region must not corrupt what a later read of the same region returns,
    whichever backend served it."""
    build = _BACKENDS[name][0]
    source, num_channels = build(tmp_path)
    with open_raster(source, num_channels) as src:
        rect = Rect(0, 0, min(4, src.width), min(4, src.height))
        first, _spec = src.read_region(rect)
        original = first.copy()
        first[:] = 0
        second, _spec = src.read_region(rect)
    assert original.any(), "the fixture region must not already be all zeros"
    assert np.array_equal(second, original)


@pytest.mark.parametrize("name", sorted(_BACKENDS))
def test_a_region_outside_the_raster_refuses(tmp_path: Path, name: str) -> None:
    build = _BACKENDS[name][0]
    source, num_channels = build(tmp_path)
    with open_raster(source, num_channels) as src:
        with pytest.raises(ValueError):
            src.read_region(Rect(0, 0, src.width + 1, src.height))
        with pytest.raises(ValueError):
            src.read_region(Rect(0, 0, 0, 0))


# ── Windowed reads ───────────────────────────────────────────────────────


def test_read_window_full_extent_matches_source(tmp_path: Path) -> None:
    arr = _distinctive_array(23, 17)
    path = tmp_path / "win.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)
    with StripTiffSource(path) as source:
        assert source.height == 23
        assert source.width == 17
        assert source.num_channels == 3
        window = source.read_window(0, 23, 0, 17)
    assert np.array_equal(window, arr)


def test_read_window_spans_multiple_internal_strips(tmp_path: Path) -> None:
    arr = _distinctive_array(23, 17)
    path = tmp_path / "win.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)  # strips at rows 0,4,8,12,16,20
    with StripTiffSource(path) as source:
        window = source.read_window(3, 13, 2, 10)  # spans strip boundaries at rows 4,8,12
    assert np.array_equal(window, arr[3:13, 2:10])


def test_read_window_partial_edge_window(tmp_path: Path) -> None:
    arr = _distinctive_array(23, 17)  # 23 rows at rowsperstrip=4: the last strip holds 3 rows
    path = tmp_path / "win.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)
    with StripTiffSource(path) as source:
        window = source.read_window(19, 23, 10, 17)
    assert np.array_equal(window, arr[19:23, 10:17])


def test_read_window_out_of_bounds_raises(tmp_path: Path) -> None:
    arr = _distinctive_array(10, 10)
    path = tmp_path / "win.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)
    with StripTiffSource(path) as source:
        with pytest.raises(ValueError):
            source.read_window(0, 11, 0, 10)


def test_windowed_and_whole_decodes_of_one_strip_tiff_agree(tmp_path: Path) -> None:
    """The load-bearing equivalence between the two TIFF backends: the same file assembled from
    windowed reads, decoded whole, and read by tifffile itself must be the same pixels."""
    arr = _distinctive_array(23, 17)
    path = tmp_path / "win.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)

    with StripTiffSource(path) as source:
        bands = [source.read_window(y0, min(y0 + 5, 23), 0, 17) for y0 in range(0, 23, 5)]
    assembled = np.concatenate(bands, axis=0)

    with TiffWholeSource(path, 3) as whole:
        decoded, _spec = whole.read_region(Rect(0, 0, whole.width, whole.height))

    assert np.array_equal(assembled, decoded)
    assert np.array_equal(decoded, tifffile.imread(str(path)))


def test_read_window_caches_strips_across_a_row_band(tmp_path: Path) -> None:
    """The access pattern a real tiling loop produces: many windows at the same ``y0``, scanning
    across the width, each spanning the same handful of strips as its row-band neighbours. Each
    strip must be decoded from disk once, not once per window that touches it.
    """
    height, width, rowsperstrip, tile = 64, 200, 8, 20
    arr = _distinctive_array(height, width)
    path = tmp_path / "win.tif"
    _write_striped_tiff(path, arr, rowsperstrip=rowsperstrip)
    with StripTiffSource(path) as source:
        y0, y1 = 8, 24  # spans strips 1..2 (rows 8-15, 16-23) exactly
        windows = [
            source.read_window(y0, y1, x0, min(x0 + tile, width))
            for x0 in range(0, width, tile)
        ]
        assert len(windows) > 3  # a real row-band, not a single tile
        assert source.strip_decode_count == 2  # the row-band's two strips, decoded exactly once
        for window, x0 in zip(windows, range(0, width, tile)):
            x1 = min(x0 + tile, width)
            assert np.array_equal(window, arr[y0:y1, x0:x1])

        # Advancing to the next row-band's own strips (3, 4) is still a genuine miss for both.
        source.read_window(y1, y1 + (y1 - y0), 0, tile)
        assert source.strip_decode_count == 4


def test_read_window_returns_a_copy_not_a_cached_view(tmp_path: Path) -> None:
    """A caller that mutates its returned window must not corrupt what a later window, served
    from the same cached strip, reads back."""
    arr = _distinctive_array(23, 17)
    path = tmp_path / "win.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)
    with StripTiffSource(path) as source:
        first = source.read_window(0, 4, 0, 17)
        first[:] = 255
        second = source.read_window(0, 4, 0, 17)
    assert np.array_equal(second, arr[0:4, 0:17])


def test_strip_source_refuses_tiled_tiff(tmp_path: Path) -> None:
    arr = _distinctive_array(64, 64)
    path = tmp_path / "tiled.tif"
    tifffile.imwrite(str(path), arr, photometric="rgb", tile=(16, 16))
    with pytest.raises(UnsupportedRasterLayout):
        StripTiffSource(path)


def test_strip_source_refuses_multipage_row_block_tiff(tmp_path: Path) -> None:
    """tifffile stores a plain channel-last N-band raster one row-block per page, so ``pages[0]``
    of this 64x80x5 raster describes a single row-block (height 80, width 5, one sample), not the
    raster. A source built from that page's shape would silently serve windows sliced from the
    wrong geometry, so construction must refuse instead."""
    arr = np.zeros((64, 80, 5), dtype=np.uint8)
    path = tmp_path / "rowblock.tif"
    tifffile.imwrite(str(path), arr)
    with tifffile.TiffFile(str(path)) as tif:
        assert len(tif.pages) > 1  # the layout under test, not a single-page file
        assert tuple(tif.series[0].shape) == (64, 80, 5)
    with pytest.raises(UnsupportedRasterLayout):
        StripTiffSource(path)


def test_strip_source_accepts_single_page_grayscale(tmp_path: Path) -> None:
    """A 1-sample raster's series shape is 2-D (H, W) while the page reports samplesperpixel=1;
    the whole-raster cross-check must treat those as agreeing, not refuse a valid layout."""
    arr = (np.arange(23 * 17, dtype=np.uint16) % 256).astype(np.uint8).reshape(23, 17)
    path = tmp_path / "gray.tif"
    tifffile.imwrite(str(path), arr, rowsperstrip=4)
    with StripTiffSource(path) as source:
        assert (source.height, source.width, source.num_channels) == (23, 17, 1)
        window = source.read_window(3, 13, 2, 10)
    assert np.array_equal(np.squeeze(window), arr[3:13, 2:10])


# ── Strip cache capacity ─────────────────────────────────────────────────


def test_derive_strip_cache_capacity_holds_a_band_and_what_the_overlap_carries() -> None:
    from tcip_mcp.pipelines.data.tiling import compute_stride

    # 640px tiles over 8-row strips: 80 strips per row-band, plus the 16 the 0.2 overlap re-reads.
    assert derive_strip_cache_capacity(8, 640, overlap=0.2) == 96
    assert derive_strip_cache_capacity(8, 640, stride=compute_stride(640, 0.2)) == 96
    # No overlap carries nothing forward: the row-band's own strips are the whole requirement.
    assert derive_strip_cache_capacity(8, 640, overlap=0.0) == 80


def test_deriving_a_capacity_without_a_geometry_refuses() -> None:
    with pytest.raises(ValueError, match="tiling geometry"):
        derive_strip_cache_capacity(8, 640)


def test_a_tile_geometry_sizes_the_cache_and_an_explicit_capacity_wins(tmp_path: Path) -> None:
    arr = _distinctive_array(64, 40)
    path = tmp_path / "cap.tif"
    _write_striped_tiff(path, arr, rowsperstrip=1)

    with StripTiffSource(path, tile_size=32, overlap=0.2) as derived:
        assert derived.strip_cache_capacity == derive_strip_cache_capacity(1, 32, overlap=0.2)
    with StripTiffSource(path, strip_cache_capacity=7, tile_size=32, overlap=0.2) as explicit:
        assert explicit.strip_cache_capacity == 7
    with StripTiffSource(path) as hintless:
        assert hintless.strip_cache_capacity == raster_source.DEFAULT_STRIP_CACHE_CAPACITY


def test_a_derived_capacity_over_the_memory_budget_is_capped_and_says_so(
    tmp_path: Path, monkeypatch, caplog,
) -> None:
    """A geometry needing more memory than the budget allows is served at the cap, with the
    re-decode cost that buys named in the log, never silently clamped and never refused."""
    arr = _distinctive_array(64, 40)
    path = tmp_path / "cap.tif"
    _write_striped_tiff(path, arr, rowsperstrip=1)
    strip_bytes = 40 * 3
    monkeypatch.setattr(raster_source, "_memory_budget_bytes", lambda: 4 * strip_bytes)

    with caplog.at_level(logging.WARNING, logger="tcip_mcp.pipelines.raster_source"):
        with StripTiffSource(path, tile_size=32, overlap=0.2) as source:
            assert source.strip_cache_capacity == 4
    assert "capping at 4 strips" in caplog.text

    # An explicit capacity is the caller's own call, budget or not.
    with StripTiffSource(path, strip_cache_capacity=40) as explicit:
        assert explicit.strip_cache_capacity == 40


def _row_major_scan(source: StripTiffSource, tile_size: int, stride: int) -> None:
    """Read every tile of a row-major sliding-window scan, the order a tiled inference pass walks."""
    from tcip_mcp.pipelines.data.tiling import tile_positions

    for tile_x, tile_y in tile_positions(source.height, source.width, tile_size, stride):
        source.read_window(tile_y, min(tile_y + tile_size, source.height),
                           tile_x, min(tile_x + tile_size, source.width))


def _amplification_raster(tmp_path: Path) -> tuple[Path, int]:
    """A raster whose tile row-band needs more strips than the hint-less default cache holds:
    one row per strip, and a width of several tiles so a band is really re-scanned."""
    path = tmp_path / "scan.tif"
    _write_striped_tiff(path, _distinctive_array(100, 300), rowsperstrip=1)
    return path, 100


def test_a_row_major_scan_at_a_derived_capacity_decodes_each_strip_once(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.tiling import compute_stride

    path, strip_count = _amplification_raster(tmp_path)
    with StripTiffSource(path, tile_size=100, overlap=0.2) as source:
        _row_major_scan(source, 100, compute_stride(100, 0.2))
        assert source.strip_decode_count == strip_count


def test_the_same_scan_re_decodes_strips_at_a_capacity_the_row_band_outgrows(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.tiling import compute_stride

    path, strip_count = _amplification_raster(tmp_path)
    with StripTiffSource(path, strip_cache_capacity=64) as source:
        assert source.strip_cache_capacity < strip_count  # the row-band cannot fit in the cache
        _row_major_scan(source, 100, compute_stride(100, 0.2))
        assert source.strip_decode_count > strip_count


# ── The process-local pool of open sources ───────────────────────────────


def test_the_pool_serves_one_open_source_per_file_and_channel_count(tmp_path: Path) -> None:
    path, _ = _strip_tiff(tmp_path)
    first = raster_source.pooled_source(path, 3)
    assert raster_source.pooled_source(path, 3) is first
    assert raster_source.pooled_source(path, 1) is not first


def test_the_pool_key_changes_when_a_band_member_is_rewritten(tmp_path: Path) -> None:
    ref, num_channels = _band_group(tmp_path)
    before = raster_source.source_pool_key(ref, num_channels)
    member = next(iter(ref.bands.values()))
    stat = member.stat()
    os.utime(member, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert raster_source.source_pool_key(ref, num_channels) != before


def test_a_forked_worker_starts_with_an_empty_pool(tmp_path: Path, monkeypatch) -> None:
    path, _ = _strip_tiff(tmp_path)
    parent_source = raster_source.pooled_source(path, 3)
    monkeypatch.setattr(os, "getpid", lambda: 424242)
    child_source = raster_source.pooled_source(path, 3)
    assert child_source is not parent_source
    assert not parent_source.closed  # the parent still owns what it opened
    parent_source.close()


def test_the_pool_evicts_the_least_recently_used_source_over_budget(tmp_path: Path, monkeypatch) -> None:
    first_path, _ = _strip_tiff(tmp_path)
    second_path = tmp_path / "other.tif"
    _write_striped_tiff(second_path, _distinctive_array(23, 17), rowsperstrip=4)

    first = raster_source.pooled_source(first_path, 3)
    monkeypatch.setattr(raster_source, "_memory_budget_bytes", lambda: 1)
    second = raster_source.pooled_source(second_path, 3)

    assert first.closed
    assert not second.closed
    assert raster_source.pooled_source(first_path, 3) is not first


# ── Reads that were always valid and must stay so ────────────────────────


def test_a_five_band_geotiff_opened_at_three_channels_still_reads_five_bands(tmp_path: Path) -> None:
    """The channel count routes; it never asserts anything about the file. A 5-band raster read at
    3 is still 5 bands, and the caller is the one who compares."""
    from tcip_mcp.pipelines.image_utils import load_image

    path = tmp_path / "five.tif"
    arr = _distinctive_array(24, 40, channels=5)
    tifffile.imwrite(str(path), arr)
    got = load_image(path, 3)
    assert np.array_equal(got, tifffile.imread(str(path)))
    assert got.shape == (24, 40, 5)


def test_image_dimensions_of_a_two_band_group_at_the_default_channel_count(tmp_path: Path) -> None:
    """A group's frame comes from its bands, so the default count (3) never has to match the two
    bands it actually holds."""
    from tcip_mcp.pipelines.image_utils import image_dimensions

    ref, _num_channels = _band_group(tmp_path)
    assert image_dimensions(ref) == (8, 8)


def test_a_band_group_whose_members_disagree_on_the_frame_refuses(tmp_path: Path) -> None:
    """A member larger than the group's frame must refuse like a smaller one always has, never be
    silently cropped down to fit: stacking bands that disagree on the frame is not a raster."""
    from tcip_mcp.pipelines.data.band_groups import read_band_group_manifest, write_band_group_manifest
    from tcip_mcp.pipelines.image_utils import load_multiband

    d = tmp_path / "grouped"
    d.mkdir()
    green = d / "cap_G.tif"
    red = d / "cap_R.tif"
    tifffile.imwrite(str(green), np.full((8, 8), 111, dtype=np.uint16))
    tifffile.imwrite(str(red), np.zeros((12, 12), dtype=np.uint16))
    ref = read_band_group_manifest(write_band_group_manifest(d, "cap", {"Green": green, "Red": red}))

    with pytest.raises(ValueError, match="disagree on the frame"):
        raster_source.open_raster(ref, 2)
    with pytest.raises(ValueError):
        load_multiband(ref, 2)


def test_a_channel_first_shaped_npy_is_left_alone_at_a_mismatched_count(tmp_path: Path) -> None:
    """The channel-first transpose fires only when the leading axis matches the count asked for:
    a (5, 40, 24) array read at 3 channels stays exactly as it was stored."""
    from tcip_mcp.pipelines.image_utils import load_multiband

    path = tmp_path / "cfirst.npy"
    arr = np.arange(5 * 40 * 24, dtype=np.uint8).reshape(5, 40, 24)
    np.save(str(path), arr)
    got = load_multiband(path, 3)
    assert got.shape == (5, 40, 24)
    assert np.array_equal(got, arr)
