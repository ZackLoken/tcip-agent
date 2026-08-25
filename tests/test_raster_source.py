"""The raster reading layer: which backend serves which source, and what each one guarantees.

Covers the factory's dispatch (GDAL-first with the tifffile series cross-check), the windowed
GDAL backend's pixel behavior, the copy-on-return, bounds and target-size contracts every backend
shares, and the process-local pool of open sources.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from tcip_mcp.pipelines import raster_source
from tcip_mcp.pipelines.raster_source import Rect, TiffWholeSource, open_raster


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


def _write_planar_tiff(path: Path, arr: np.ndarray, *, rowsperstrip: int = 8) -> None:
    """A genuine planar (band-separate) file: tifffile takes the pixels channel-first."""
    extrasamples = ["unassalpha"] * (arr.shape[-1] - 3) if arr.shape[-1] > 3 else None
    kwargs = {"photometric": "rgb", "planarconfig": "separate", "rowsperstrip": rowsperstrip}
    if extrasamples:
        kwargs["extrasamples"] = extrasamples
    tifffile.imwrite(str(path), np.moveaxis(arr, -1, 0), **kwargs)


# ── One source per backend ───────────────────────────────────────────────


def _photographic(tmp_path: Path):
    from PIL import Image

    path = tmp_path / "photo.png"
    Image.fromarray(_distinctive_array(12, 9)).save(path)
    return path, 3


def _gdal_tiff(tmp_path: Path):
    path = tmp_path / "striped.tif"
    _write_striped_tiff(path, _distinctive_array(23, 17), rowsperstrip=4)
    return path, 3


def _whole_tiff(tmp_path: Path):
    """A channel-last 5-band raster: tifffile stores it one row-block per page, a stacked layout
    GDAL's first-IFD data model misreads, so the factory sends it to the whole decode."""
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


# Expected backends by attribute name, resolved at test time.
_BACKENDS = {
    "band_group": (_band_group, "BandGroupSource"),
    "gdal_tiff": (_gdal_tiff, "GdalSource"),
    "npy": (_npy, "NpySource"),
    "npz": (_npz, "NpzSource"),
    "photographic": (_photographic, "PhotographicSource"),
    "tiff_whole": (_whole_tiff, "TiffWholeSource"),
}


# ── Factory dispatch ─────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(_BACKENDS))
def test_open_raster_picks_the_backend_each_source_needs(tmp_path: Path, name: str) -> None:
    build, expected_name = _BACKENDS[name]
    source, num_channels = build(tmp_path)
    with open_raster(source, num_channels) as src:
        assert isinstance(src, getattr(raster_source, expected_name))


def test_a_photographic_extension_at_a_count_pil_has_no_mode_for_refuses(tmp_path: Path) -> None:
    """The same refusal, wording included, a caller asking for band data out of a photograph gets:
    the count is a routing hint, and there is no PIL mode for 5 channels."""
    source, _ = _photographic(tmp_path)
    with pytest.raises(ValueError, match=r"Cannot load a 5-channel image from '\.png'"):
        open_raster(source, 5)


@pytest.mark.parametrize("layout", ["striped", "tiled", "planar"])
def test_single_dataset_tiff_layouts_are_served_windowed(tmp_path: Path, layout: str) -> None:
    """Striped, internally tiled, and planar (band-separate) single-dataset TIFFs all read
    through the windowed GDAL backend, and the pixels served are the source array's own
    regardless of the on-disk layout."""
    arr = _distinctive_array(32, 40, channels=3)
    path = tmp_path / f"{layout}.tif"
    if layout == "striped":
        _write_striped_tiff(path, arr, rowsperstrip=8)
    elif layout == "tiled":
        tifffile.imwrite(str(path), arr, photometric="rgb", tile=(16, 16))
    else:
        _write_planar_tiff(path, arr)

    with open_raster(path, 3) as src:
        assert isinstance(src, raster_source.GdalSource)
        region, spec = src.read_region(Rect(0, 0, src.width, src.height))
    assert np.array_equal(region, arr)
    assert (spec.backend, spec.scale, spec.resample) == ("gdal", 1.0, None)


def test_a_planar_raster_read_at_a_mismatched_count_still_reads_as_stored(tmp_path: Path) -> None:
    """A planar file is channel-first by its own header (tifffile's SYX series axes), so its real
    frame and band count survive whatever channel count the caller routes with; it must be served
    windowed in that frame, never whole-decoded into the raw band-first array."""
    arr = _distinctive_array(32, 40, channels=4)
    path = tmp_path / "planar4.tif"
    _write_planar_tiff(path, arr)

    with open_raster(path, 3) as src:
        assert (src.width, src.height, src.num_channels) == (40, 32, 4)
        region, _spec = src.read_region(Rect(0, 0, 40, 32))
    assert np.array_equal(region, arr)


def test_a_stacked_multipage_tiff_still_reads_whole(tmp_path: Path) -> None:
    """The multi-page stacks tifffile writes (one row-block or one band per page) have no
    single-frame GDAL reading, so they decode whole and return exactly what ``tifffile.imread``
    does, rather than being refused or served from the first page's wrong geometry."""
    path, num_channels = _whole_tiff(tmp_path)
    with tifffile.TiffFile(str(path)) as tif:
        assert len(tif.pages) > 1  # the layout under test, not a single-page file
    with open_raster(path, num_channels) as src:
        assert isinstance(src, TiffWholeSource)
        region, _spec = src.read_region(Rect(0, 0, src.width, src.height))
    assert np.array_equal(region, tifffile.imread(str(path)))


def test_a_shape_the_whole_decode_would_transpose_is_not_served_windowed(tmp_path: Path) -> None:
    """``load_multiband`` reads a channel-first-looking shape into channel-last order and
    ``image_dimensions`` applies the same reading to the header, so a raster the reinterpretation
    fires on has to go whole; served windowed it would report one frame and measure as another."""
    from tcip_mcp.pipelines.image_utils import image_dimensions

    path = tmp_path / "three_row.tif"
    arr = _distinctive_array(3, 20, channels=4)
    tifffile.imwrite(str(path), arr, photometric="rgb", extrasamples=["unassalpha"], rowsperstrip=1)

    with open_raster(path, 3) as src:
        assert isinstance(src, TiffWholeSource)
        region, _spec = src.read_region(Rect(0, 0, src.width, src.height))
    assert region.shape == (20, 4, 3)
    assert image_dimensions(path, 3) == (4, 20)

    # At the count the reinterpretation leaves alone, the windowed backend serves it as stored.
    with open_raster(path, 4) as src:
        assert isinstance(src, raster_source.GdalSource)
        assert (src.height, src.width, src.num_channels) == (3, 20, 4)


def test_an_unreadable_tiff_fails_naming_the_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.tif"
    path.write_bytes(b"II*\x00garbage that is not a real tiff")
    with pytest.raises(ValueError, match=r"broken\.tif"):
        open_raster(path, 3)


def test_a_plain_read_serves_full_resolution(tmp_path: Path) -> None:
    """Without a ``target_size`` every read is native resolution and the ``ReadSpec`` says so."""
    source, num_channels = _gdal_tiff(tmp_path)
    with open_raster(source, num_channels) as src:
        _region, spec = src.read_region(Rect(0, 0, 4, 4))
    assert (spec.backend, spec.scale, spec.resample) == ("gdal", 1.0, None)


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
    with open_raster(path, 3) as source:
        assert source.height == 23
        assert source.width == 17
        assert source.num_channels == 3
        window = source.read_window(0, 23, 0, 17)
    assert np.array_equal(window, arr)


def test_read_window_is_rows_first_and_matches_the_numpy_slice(tmp_path: Path) -> None:
    """``read_window(y0, y1, x0, x1)`` is exactly ``arr[y0:y1, x0:x1]``: the row-first argument
    order the tiled inference loop uses, against ``Rect``'s x-first order, on a deliberately
    asymmetric window a silent transposition could not survive."""
    arr = _distinctive_array(23, 17)
    path = tmp_path / "win.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)
    with open_raster(path, 3) as source:
        window = source.read_window(3, 13, 2, 10)
    assert window.shape == (10, 8, 3)
    assert np.array_equal(window, arr[3:13, 2:10])


def test_read_window_partial_edge_window(tmp_path: Path) -> None:
    arr = _distinctive_array(23, 17)
    path = tmp_path / "win.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)
    with open_raster(path, 3) as source:
        window = source.read_window(19, 23, 10, 17)
    assert np.array_equal(window, arr[19:23, 10:17])


def test_read_window_out_of_bounds_raises(tmp_path: Path) -> None:
    arr = _distinctive_array(10, 10)
    path = tmp_path / "win.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)
    with open_raster(path, 3) as source:
        with pytest.raises(ValueError):
            source.read_window(0, 11, 0, 10)


def test_windowed_and_whole_decodes_of_one_tiff_agree(tmp_path: Path) -> None:
    """The load-bearing equivalence between the two TIFF backends: the same file assembled from
    windowed reads, decoded whole, and read by tifffile itself must be the same pixels."""
    arr = _distinctive_array(23, 17)
    path = tmp_path / "win.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)

    with open_raster(path, 3) as source:
        bands = [source.read_window(y0, min(y0 + 5, 23), 0, 17) for y0 in range(0, 23, 5)]
    assembled = np.concatenate(bands, axis=0)

    with TiffWholeSource(path, 3) as whole:
        decoded, _spec = whole.read_region(Rect(0, 0, whole.width, whole.height))

    assert np.array_equal(assembled, decoded)
    assert np.array_equal(decoded, tifffile.imread(str(path)))


def test_read_window_returns_a_copy_not_a_cached_view(tmp_path: Path) -> None:
    """A caller that mutates its returned window must not corrupt what a later read of the same
    window, served from whatever cache the backend keeps, reads back."""
    arr = _distinctive_array(23, 17)
    path = tmp_path / "win.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)
    with open_raster(path, 3) as source:
        first = source.read_window(0, 4, 0, 17)
        first[:] = 255
        second = source.read_window(0, 4, 0, 17)
    assert np.array_equal(second, arr[0:4, 0:17])


def test_a_single_band_raster_serves_with_a_trailing_channel_axis(tmp_path: Path) -> None:
    arr = (np.arange(23 * 17, dtype=np.uint16) % 256).astype(np.uint8).reshape(23, 17)
    path = tmp_path / "gray.tif"
    tifffile.imwrite(str(path), arr, rowsperstrip=4)
    with open_raster(path, 1) as source:
        assert (source.height, source.width, source.num_channels) == (23, 17, 1)
        window = source.read_window(3, 13, 2, 10)
    assert window.shape == (10, 8, 1)
    assert np.array_equal(np.squeeze(window, -1), arr[3:13, 2:10])


def test_a_palette_tiff_expands_through_its_color_table_like_pil(tmp_path: Path) -> None:
    """A palette-color TIFF must serve the colors its own table names, exactly the pixels PIL's
    palette decode produced, never the raw one-band table indices."""
    from PIL import Image

    indices = (np.arange(23 * 17, dtype=np.uint16) % 256).astype(np.uint8).reshape(23, 17)
    img = Image.fromarray(indices, mode="P")
    palette = []
    for i in range(256):
        palette += [i, 255 - i, (i * 7) % 256]
    img.putpalette(palette)
    path = tmp_path / "palette.tif"
    img.save(str(path))

    expected = np.asarray(img.convert("RGB"))
    with open_raster(path, 3) as src:
        region, _spec = src.read_region(Rect(0, 0, src.width, src.height))
        window = src.read_window(3, 13, 2, 10)
        # Pixel identity first: it is the measurement claim, whatever backend served it.
        assert np.array_equal(region, expected)
        assert np.array_equal(window, expected[3:13, 2:10])
        assert isinstance(src, raster_source.GdalSource)
        assert (src.num_channels, src.dtype) == (3, np.dtype("uint8"))


def test_band_interpretations_name_each_served_channel(tmp_path: Path) -> None:
    """A consumer (the web serving layer) tells an alpha band from a spectral one through the
    file's own declared color interpretations, never by guessing from the channel count; a
    palette source reports the three channels its expansion actually serves."""
    from PIL import Image

    rgba_path = tmp_path / "rgba.tif"
    _write_striped_tiff(rgba_path, _distinctive_array(8, 8, channels=4), rowsperstrip=4)
    with open_raster(rgba_path, 4) as src:
        assert src.band_interpretations == ("red", "green", "blue", "alpha")

    indices = (np.arange(64, dtype=np.uint16) % 256).astype(np.uint8).reshape(8, 8)
    img = Image.fromarray(indices, mode="P")
    img.putpalette([v for i in range(256) for v in (i, 255 - i, (i * 7) % 256)])
    palette_path = tmp_path / "palette.tif"
    img.save(str(palette_path))
    with open_raster(palette_path, 3) as src:
        assert src.num_channels == 3
        assert src.band_interpretations == ("red", "green", "blue")


def test_a_raster_serves_its_own_dtype(tmp_path: Path) -> None:
    arr = (np.arange(64, dtype=np.uint16) * 500).reshape(8, 8)
    path = tmp_path / "u16.tif"
    tifffile.imwrite(str(path), arr, rowsperstrip=4)
    with open_raster(path, 1) as src:
        assert src.dtype == np.dtype("uint16")
        region, _spec = src.read_region(Rect(0, 0, 8, 8))
    assert region.dtype == np.dtype("uint16")
    assert np.array_equal(region[:, :, 0], arr)


# ── target_size reads ────────────────────────────────────────────────────


def test_a_target_size_read_downsamples_through_gdal(tmp_path: Path) -> None:
    """A GDAL-served ``target_size`` read returns the reduced buffer directly, with the
    ``ReadSpec`` recording the requested scale and average resampling; at an exact 2x decimation
    every output pixel is its 2x2 block's mean."""
    arr = _distinctive_array(32, 40)
    path = tmp_path / "down.tif"
    _write_striped_tiff(path, arr, rowsperstrip=8)
    with open_raster(path, 3) as src:
        region, spec = src.read_region(Rect(0, 0, 40, 32), target_size=(20, 16))
    assert region.shape == (16, 20, 3)
    assert (spec.backend, spec.scale, spec.resample) == ("gdal", 0.5, "average")
    blocks = arr.reshape(16, 2, 20, 2, 3).mean(axis=(1, 3))
    assert np.allclose(region, blocks, atol=1.0)


def test_a_target_size_read_downsamples_an_array_backend_by_area(tmp_path: Path) -> None:
    """A backend with no overview machinery slices native and area-downsamples, recording the
    same requested scale with its own resampling name."""
    arr = _distinctive_array(32, 40, channels=5)
    path = tmp_path / "bands.npy"
    np.save(str(path), arr)
    with open_raster(path, 5) as src:
        region, spec = src.read_region(Rect(0, 0, 40, 32), target_size=(20, 16))
    assert region.shape == (16, 20, 5)
    assert (spec.backend, spec.scale, spec.resample) == ("npy", 0.5, "area")
    blocks = arr.reshape(16, 2, 20, 2, 5).mean(axis=(1, 3))
    assert np.allclose(region, blocks, atol=1.0)


@pytest.mark.parametrize("name", ["gdal_tiff", "npy"])
def test_a_distorting_target_size_refuses(tmp_path: Path, name: str) -> None:
    """``target_size`` must preserve the region's aspect ratio: a resample in this layer never
    silently changes a raster's geometry."""
    build, _ = _BACKENDS[name]
    source, num_channels = build(tmp_path)
    with open_raster(source, num_channels) as src:
        with pytest.raises(ValueError, match="aspect ratio"):
            src.read_region(Rect(0, 0, src.width, src.height),
                            target_size=(src.width, max(1, src.height // 3)))


# ── The process-local pool of open sources ───────────────────────────────


def test_the_pool_serves_one_open_source_per_file_and_channel_count(tmp_path: Path) -> None:
    path, _ = _gdal_tiff(tmp_path)
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
    path, _ = _gdal_tiff(tmp_path)
    parent_source = raster_source.pooled_source(path, 3)
    monkeypatch.setattr(os, "getpid", lambda: 424242)
    child_source = raster_source.pooled_source(path, 3)
    assert child_source is not parent_source
    assert not parent_source.closed  # the parent still owns what it opened
    parent_source.close()


def test_the_pool_evicts_the_least_recently_used_source_over_budget(tmp_path: Path, monkeypatch) -> None:
    first_path, num_channels = _npy(tmp_path)
    second_path = tmp_path / "other.npy"
    np.save(str(second_path), _distinctive_array(18, 11, channels=5))

    first = raster_source.pooled_source(first_path, num_channels)
    monkeypatch.setattr(raster_source, "_memory_budget_bytes", lambda: 1)
    second = raster_source.pooled_source(second_path, num_channels)

    assert first.closed
    assert not second.closed
    assert raster_source.pooled_source(first_path, num_channels) is not first


def test_a_gdal_source_accounts_no_resident_pixels(tmp_path: Path) -> None:
    """A GDAL source holds only a dataset handle; its decoded blocks live in GDAL's own budgeted
    block cache, so the pool accounts it at zero resident bytes rather than double-counting."""
    path, _ = _gdal_tiff(tmp_path)
    with open_raster(path, 3) as src:
        assert isinstance(src, raster_source.GdalSource)
        assert src.resident_bytes == 0


def test_a_photographic_source_accounts_its_peak_resident_frames(tmp_path: Path) -> None:
    """read_region materializes an ndarray copy beside the PIL frame; the pool records
    resident_bytes once at insert, so the value must already cover that peak, and it must not
    change after the copy exists or the pool's byte accounting drifts."""
    path, _ = _photographic(tmp_path)
    with open_raster(path, 3) as src:
        frame_bytes = src.width * src.height * src.num_channels * src.dtype.itemsize
        before = src.resident_bytes
        src.read_region(Rect(0, 0, src.width, src.height))
        assert src.resident_bytes == before
        assert src.resident_bytes >= 2 * frame_bytes


@pytest.mark.parametrize("name", sorted(_BACKENDS))
def test_opens_windowed_names_the_backends_that_open_without_decoding(
        tmp_path: Path, name: str) -> None:
    """Only GDAL-served rasters and memory-mapped .npy open without a whole decode; an eager
    open of anything else would decode every pixel at construction time."""
    build, _backend = _BACKENDS[name]
    source, num_channels = build(tmp_path)
    expected = name in ("gdal_tiff", "npy")
    assert raster_source.opens_windowed(source, num_channels) is expected


def test_opens_windowed_answers_false_for_an_unopenable_tiff(tmp_path: Path) -> None:
    path = tmp_path / "broken.tif"
    path.write_bytes(b"II*\x00garbage")
    assert raster_source.opens_windowed(path, 3) is False


def test_is_georeferenced_true_only_for_a_real_geotransform(tmp_path: Path) -> None:
    """A raster with real ModelPixelScale/ModelTiepoint/GeoKeyDirectory tags (a stitched,
    georectified orthomosaic) answers True; an ordinary capture with none of those tags (this
    module's own fixtures, and an .npy, which has no tag mechanism at all) answers False,
    regardless of pixel dimensions -- the decider is georeferencing, never size."""
    plain_path = tmp_path / "plain.tif"
    _write_striped_tiff(plain_path, _distinctive_array(24, 20), rowsperstrip=4)
    assert raster_source.is_georeferenced(plain_path) is False

    npy_path = tmp_path / "content.npy"
    np.save(str(npy_path), _distinctive_array(24, 20))
    assert raster_source.is_georeferenced(npy_path) is False

    geo_path = tmp_path / "geo.tif"
    geokeys = (1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, 32615)  # UTM zone 15N
    tifffile.imwrite(
        str(geo_path), _distinctive_array(24, 20), photometric="rgb", rowsperstrip=4,
        extratags=[
            (33550, "d", 3, (1.0, 1.0, 0.0), False),
            (33922, "d", 6, (0.0, 0.0, 0.0, 500_000.0, 4_800_000.0, 0.0), False),
            (34735, "H", len(geokeys), geokeys, False),
        ],
    )
    assert raster_source.is_georeferenced(geo_path) is True


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


# ── _RegionView: an offset read window over an already-open parent source ───────────────────


def test_region_view_reports_the_rects_own_extent_not_the_parents(tmp_path: Path) -> None:
    """The dims invariant this class exists to hold: height/width always report the rect it was
    constructed over, never the full parent source's own dims."""
    from tcip_mcp.pipelines.raster_source import _RegionView

    arr = _distinctive_array(40, 30)
    path = tmp_path / "region.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)
    with open_raster(path, 3) as parent:
        view = _RegionView(parent, Rect(5, 5, 25, 20))
        assert (view.height, view.width) == (15, 20)
        assert (view.height, view.width) != (parent.height, parent.width)
        assert view.num_channels == parent.num_channels == 3


def test_region_view_read_window_translates_into_the_parents_coordinate_space(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.raster_source import _RegionView

    arr = _distinctive_array(40, 30)
    path = tmp_path / "region.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)
    with open_raster(path, 3) as parent:
        view = _RegionView(parent, Rect(5, 5, 25, 20))
        full = view.read_window(0, view.height, 0, view.width)
        partial = view.read_window(2, 10, 3, 12)
    assert np.array_equal(full, arr[5:20, 5:25])
    assert np.array_equal(partial, arr[7:15, 8:17])


def test_region_view_refuses_a_read_past_its_own_declared_bounds(tmp_path: Path) -> None:
    """The one place this design could silently read real pixels beyond the region a caller
    asked for: the parent source has plenty of real pixels past this rect (unlike a true
    raster edge, which zero-pads), so the view's own bounds must be enforced explicitly, not
    left as an emergent property of what happens to call it today."""
    from tcip_mcp.pipelines.raster_source import _RegionView

    arr = _distinctive_array(40, 30)
    path = tmp_path / "region.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)
    with open_raster(path, 3) as parent:
        view = _RegionView(parent, Rect(5, 5, 25, 20))  # height=15, width=20
        with pytest.raises(ValueError):
            view.read_window(0, view.height + 1, 0, view.width)
        with pytest.raises(ValueError):
            view.read_window(0, view.height, 0, view.width + 1)


def test_region_view_construction_refuses_a_rect_outside_the_parents_own_bounds(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.raster_source import _RegionView

    arr = _distinctive_array(40, 30)
    path = tmp_path / "region.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)
    with open_raster(path, 3) as parent:
        with pytest.raises(ValueError):
            _RegionView(parent, Rect(0, 0, 31, 20))  # x1=31 > parent.width=30


def test_region_view_forwards_the_parents_band_interpretations(tmp_path: Path) -> None:
    """A haloed calibration/holdout block reads through this view, never the parent directly;
    the alpha-vs-spectral-band decision (image_utils.to_pil_if_faithful) must resolve the same
    way there as it does for a whole-mosaic export of the same file, so the fact has to survive
    the wrap."""
    from tcip_mcp.pipelines.raster_source import _RegionView

    rgba_path = tmp_path / "rgba.tif"
    _write_striped_tiff(rgba_path, _distinctive_array(40, 30, channels=4), rowsperstrip=4)
    with open_raster(rgba_path, 4) as parent:
        view = _RegionView(parent, Rect(5, 5, 25, 20))
        assert view.band_interpretations == parent.band_interpretations == (
            "red", "green", "blue", "alpha",
        )


def test_region_view_reports_no_band_interpretations_for_a_parent_that_carries_none(
    tmp_path: Path,
) -> None:
    """A .npy-backed parent carries no color-interpretation metadata at all; the view must not
    fabricate one, same as raster_content_identity's own getattr convention."""
    from tcip_mcp.pipelines.raster_source import _RegionView

    path = tmp_path / "bands.npy"
    np.save(str(path), _distinctive_array(40, 30, channels=3))
    with open_raster(path, 3) as parent:
        assert not hasattr(parent, "band_interpretations")
        view = _RegionView(parent, Rect(5, 5, 25, 20))
        assert view.band_interpretations is None


# ── raster_content_identity: one raster file's own content identity ─────────────────────────


_IDENTITY_KW = dict(seed=7, window_size=8, max_windows=50)


def test_raster_content_identity_agrees_across_gdal_and_npy_backends_for_same_content(
    tmp_path: Path,
) -> None:
    """Two entirely different backends (a GDAL-served GeoTIFF, a memory-mapped .npy) reading the
    same pixel content resolve the same identity: the checksum is the discriminating term, never
    a GDAL-only attribute."""
    from tcip_mcp.pipelines.raster_source import raster_content_identity

    arr = _distinctive_array(24, 20)
    tif_path = tmp_path / "content.tif"
    npy_path = tmp_path / "content.npy"
    _write_striped_tiff(tif_path, arr, rowsperstrip=4)
    np.save(str(npy_path), arr)

    tif_identity = raster_content_identity(tif_path, 3, **_IDENTITY_KW)
    npy_identity = raster_content_identity(npy_path, 3, **_IDENTITY_KW)

    assert tif_identity.pixel_checksum == npy_identity.pixel_checksum
    assert (tif_identity.width, tif_identity.height, tif_identity.num_channels) == (
        npy_identity.width, npy_identity.height, npy_identity.num_channels)
    assert tif_identity.dtype == npy_identity.dtype


def test_raster_content_identity_differs_for_different_content_same_dimensions(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.raster_source import raster_content_identity

    a_path, b_path = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(str(a_path), _distinctive_array(24, 20))
    np.save(str(b_path), np.zeros((24, 20, 3), dtype=np.uint8))

    a_identity = raster_content_identity(a_path, 3, **_IDENTITY_KW)
    b_identity = raster_content_identity(b_path, 3, **_IDENTITY_KW)

    assert (a_identity.width, a_identity.height) == (b_identity.width, b_identity.height)
    assert a_identity.pixel_checksum != b_identity.pixel_checksum


def test_raster_content_identity_deterministic_under_matching_recorded_parameters(
    tmp_path: Path,
) -> None:
    """A training-time and an export-time call agree when both recompute under the identity's own
    recorded seed/window_size/max_windows -- the parameters that must travel with the identity."""
    from tcip_mcp.pipelines.raster_source import raster_content_identity

    path = tmp_path / "content.npy"
    np.save(str(path), _distinctive_array(30, 22))

    first = raster_content_identity(path, 3, **_IDENTITY_KW)
    second = raster_content_identity(
        path, 3, seed=first.seed, window_size=first.window_size, max_windows=first.max_windows)

    assert first.pixel_checksum == second.pixel_checksum
    assert (first.seed, first.window_size, first.max_windows) == (7, 8, 50)


def test_raster_content_identity_band_interpretations_present_only_on_gdal(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.raster_source import raster_content_identity

    tif_path = tmp_path / "content.tif"
    npy_path = tmp_path / "content.npy"
    arr = _distinctive_array(24, 20)
    _write_striped_tiff(tif_path, arr, rowsperstrip=4)
    np.save(str(npy_path), arr)

    tif_identity = raster_content_identity(tif_path, 3, **_IDENTITY_KW)
    npy_identity = raster_content_identity(npy_path, 3, **_IDENTITY_KW)

    assert tif_identity.band_interpretations == ("red", "green", "blue")
    assert npy_identity.band_interpretations is None


def test_raster_content_identity_geotransform_optional_never_load_bearing(tmp_path: Path) -> None:
    """A raster with no georeferencing tags (every fixture this module writes) still resolves a
    fully usable identity: the geotransform term is absent, the checksum is not."""
    from tcip_mcp.pipelines.raster_source import raster_content_identity

    tif_path = tmp_path / "content.tif"
    _write_striped_tiff(tif_path, _distinctive_array(24, 20), rowsperstrip=4)

    identity = raster_content_identity(tif_path, 3, **_IDENTITY_KW)
    assert identity.geotransform is None
    assert identity.pixel_checksum  # a real, usable identity despite no geotransform

    npy_path = tmp_path / "content.npy"
    np.save(str(npy_path), _distinctive_array(24, 20))
    npy_identity = raster_content_identity(npy_path, 3, **_IDENTITY_KW)
    assert npy_identity.geotransform is None  # not a GeoTIFF; still a fully usable identity


def test_raster_content_identity_refuses_only_when_unopenable(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.raster_source import raster_content_identity

    with pytest.raises(ValueError):
        raster_content_identity(tmp_path / "nonexistent.npy", 3, **_IDENTITY_KW)


# ── content_identity: raster_content_identity under the platform's own budget ────────────────


def test_content_identity_with_an_explicit_channel_count_matches_the_old_spelling(
    tmp_path: Path,
) -> None:
    """A caller with its own channel count (a model's ``in_chans``, a training-time probe) gets
    exactly the value the direct ``raster_content_identity`` call under the platform's constants
    used to compute, never the route's own derivation.

    A grayscale photograph is the discriminating input: ``probe_channels`` reads it at 1
    (its own band count) while ``image_route_channel_count`` reads it at 3 (a plain serve's PIL
    RGB expansion), so a helper that silently ignored the explicit argument and fell back to the
    route's own rule would still pass a same-count fixture; here the two counts disagree, so only
    a helper that actually threads the explicit count through can match."""
    from PIL import Image

    from tcip_mcp.pipelines.raster_source import (
        CONTENT_IDENTITY_MAX_WINDOWS, CONTENT_IDENTITY_SEED, CONTENT_IDENTITY_WINDOW_SIZE,
        content_identity, raster_content_identity,
    )

    path = tmp_path / "gray.png"
    Image.fromarray(_distinctive_gray(20, 16), mode="L").save(path)

    explicit = content_identity(path, 1)
    old_spelling = raster_content_identity(
        path, 1, seed=CONTENT_IDENTITY_SEED, window_size=CONTENT_IDENTITY_WINDOW_SIZE,
        max_windows=CONTENT_IDENTITY_MAX_WINDOWS)

    assert explicit == old_spelling
    assert explicit != content_identity(path)


def _distinctive_gray(height: int, width: int) -> np.ndarray:
    """A single-band 2-D array, distinctive the same way :func:`_distinctive_array` is."""
    arr = np.zeros((height, width), dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            arr[row, col] = (row + col) % 256
    return arr


def test_content_identity_with_no_channel_count_uses_the_image_route_rule(tmp_path: Path) -> None:
    """Omitting ``num_channels`` (the shape ``propose_annotations`` calls it at) resolves the
    same channel count :func:`image_route_channel_count` gives the source."""
    from PIL import Image

    from tcip_mcp.pipelines.raster_source import content_identity, image_route_channel_count

    path = tmp_path / "photo.png"
    Image.fromarray(_distinctive_gray(20, 16), mode="L").save(path)

    identity = content_identity(path)
    assert identity.num_channels == image_route_channel_count(path) == 3


def test_image_route_channel_count_expands_a_grayscale_photograph_to_three(tmp_path: Path) -> None:
    """A grayscale photographic frame opens at three channels on a plain serve (PIL's own RGB
    expansion), the same override :func:`content_identity`'s default relies on."""
    from PIL import Image

    from tcip_mcp.pipelines.raster_source import image_route_channel_count

    path = tmp_path / "gray.png"
    Image.fromarray(_distinctive_gray(20, 16), mode="L").save(path)

    assert image_route_channel_count(path) == 3


def test_image_route_channel_count_leaves_an_array_container_at_its_own_band_count(
    tmp_path: Path,
) -> None:
    """A non-photographic container (an .npy raster) is never subject to the photographic
    override: its own probed band count is what a plain serve opens it at."""
    from tcip_mcp.pipelines.raster_source import image_route_channel_count

    path = tmp_path / "single_band.npy"
    np.save(str(path), _distinctive_gray(20, 16))

    assert image_route_channel_count(path) == 1


# ── raster_identity_matches: the claim-scope comparison ──────────────────────────────────────


def test_raster_identity_matches_same_file(tmp_path: Path) -> None:
    from dataclasses import asdict

    from tcip_mcp.pipelines.raster_source import raster_content_identity, raster_identity_matches

    path = tmp_path / "content.npy"
    np.save(str(path), _distinctive_array(24, 20))
    recorded = asdict(raster_content_identity(path, 3, **_IDENTITY_KW))
    assert raster_identity_matches(recorded, path) is True


def test_raster_identity_matches_false_for_different_content(tmp_path: Path) -> None:
    from dataclasses import asdict

    from tcip_mcp.pipelines.raster_source import raster_content_identity, raster_identity_matches

    a_path, b_path = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(str(a_path), _distinctive_array(24, 20))
    np.save(str(b_path), np.zeros((24, 20, 3), dtype=np.uint8))
    recorded = asdict(raster_content_identity(a_path, 3, **_IDENTITY_KW))
    assert raster_identity_matches(recorded, b_path) is False


def test_raster_identity_matches_recomputes_under_the_recorded_parameters_not_a_new_default(
    tmp_path: Path,
) -> None:
    """The exact regression the design calls out: comparing under each call's own default sampling
    parameters (rather than the recorded ones) could false-refuse a genuinely identical raster.
    Here a deliberately different ad-hoc seed/window_size would produce a different checksum
    (since the sampled windows differ), so the match only holds because the recorded parameters,
    not this call's own default, actually drove the comparison."""
    from dataclasses import asdict

    from tcip_mcp.pipelines.raster_source import raster_content_identity, raster_identity_matches

    path = tmp_path / "content.npy"
    np.save(str(path), _distinctive_array(30, 22))
    recorded = asdict(raster_content_identity(path, 3, seed=1, window_size=4, max_windows=3))
    # A differently-parameterized fresh call would disagree on the checksum; matches() must not
    # take that path, it must recompute under recorded's own seed/window_size/max_windows.
    off_default = raster_content_identity(path, 3, seed=99, window_size=6, max_windows=2)
    assert off_default.pixel_checksum != recorded["pixel_checksum"]
    assert raster_identity_matches(recorded, path) is True


def test_raster_identity_matches_raises_when_source_unopenable(tmp_path: Path) -> None:
    from dataclasses import asdict

    from tcip_mcp.pipelines.raster_source import raster_content_identity, raster_identity_matches

    path = tmp_path / "content.npy"
    np.save(str(path), _distinctive_array(24, 20))
    recorded = asdict(raster_content_identity(path, 3, **_IDENTITY_KW))
    with pytest.raises(ValueError):
        raster_identity_matches(recorded, tmp_path / "nonexistent.npy")
