"""The raster reading layer: which backend serves which source, and what each one guarantees.

Covers the factory's dispatch (GDAL-first with the tifffile series cross-check), the windowed
GDAL backend's pixel behavior, the backends' copy-on-return, bounds and target-size contracts, and
the process-local pool of open sources.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile

from tcip_mcp.pipelines import raster_source
from tcip_mcp.pipelines.raster_source import Rect, TiffWholeSource, open_raster

if TYPE_CHECKING:
    from collections.abc import Iterator

    from PIL.ImageFile import ImageFile
    from rasterio.io import DatasetReader

    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.raster_source import RasterIdentity, RasterSource


@pytest.fixture(autouse=True)
def _empty_source_pool() -> Iterator[None]:
    """Clear the process-global source pool before and after yielding."""
    raster_source.close_source_pool()
    yield
    raster_source.close_source_pool()


def _distinctive_array(
    height: int, width: int, channels: int = 3
) -> np.ndarray:
    """Return a uint8 array whose planes hold the row, the column and their sum, each modulo 256.

    ``channels`` is at least 2; the sum plane is written when it is above 2, and planes past the
    third stay zero.
    """
    arr = np.zeros((height, width, channels), dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            arr[row, col, 0] = row % 256
            arr[row, col, 1] = col % 256
            if channels > 2:
                arr[row, col, 2] = (row + col) % 256
    return arr


def _write_striped_tiff(
    path: Path, arr: np.ndarray, *, rowsperstrip: int
) -> None:
    extrasamples = (
        ["unassalpha"] * (arr.shape[-1] - 3) if arr.shape[-1] > 3 else None
    )
    kwargs = {"photometric": "rgb", "rowsperstrip": rowsperstrip}
    if extrasamples:
        kwargs["extrasamples"] = extrasamples
    tifffile.imwrite(str(path), arr, **kwargs)


def _write_planar_tiff(
    path: Path, arr: np.ndarray, *, rowsperstrip: int = 8
) -> None:
    """Write channel-last ``arr`` as a band-separate RGB TIFF, bands past three as extra samples."""
    extrasamples = (
        ["unassalpha"] * (arr.shape[-1] - 3) if arr.shape[-1] > 3 else None
    )
    kwargs = {
        "photometric": "rgb",
        "planarconfig": "separate",
        "rowsperstrip": rowsperstrip,
    }
    if extrasamples:
        kwargs["extrasamples"] = extrasamples
    tifffile.imwrite(str(path), np.moveaxis(arr, -1, 0), **kwargs)


# ── One source per backend ───────────────────────────────────────────────


def _photographic(tmp_path: Path) -> tuple[Path, int]:
    from PIL import Image

    path = tmp_path / "photo.png"
    Image.fromarray(_distinctive_array(12, 9)).save(path)
    return path, 3


def _gdal_tiff(tmp_path: Path) -> tuple[Path, int]:
    path = tmp_path / "striped.tif"
    _write_striped_tiff(path, _distinctive_array(23, 17), rowsperstrip=4)
    return path, 3


def _whole_tiff(tmp_path: Path) -> tuple[Path, int]:
    """Write a five-plane array by ``tifffile.imwrite``; return its path and the count 5.

    The array is ``_distinctive_array(20, 14, channels=5)``.
    """
    path = tmp_path / "multipage.tif"
    tifffile.imwrite(str(path), _distinctive_array(20, 14, channels=5))
    return path, 5


def _npy(tmp_path: Path) -> tuple[Path, int]:
    path = tmp_path / "bands.npy"
    np.save(str(path), _distinctive_array(18, 11, channels=5))
    return path, 5


def _npz(tmp_path: Path) -> tuple[Path, int]:
    path = tmp_path / "bands.npz"
    np.savez(str(path), bands=_distinctive_array(18, 11, channels=5))
    return path, 5


def _band_group(tmp_path: Path) -> tuple[BandGroupRef, int]:
    from tcip_mcp.pipelines.data.band_groups import read_band_group_manifest
    from tests._producer_fixtures import write_band_group

    d = tmp_path / "grouped"
    d.mkdir()
    green = d / "cap_G.tif"
    red = d / "cap_R.tif"
    tifffile.imwrite(str(green), np.full((8, 8), 111, dtype=np.uint16))
    tifffile.imwrite(str(red), np.arange(64, dtype=np.uint16).reshape(8, 8))
    manifest = write_band_group(
        d, "cap", {"Green": green, "Red": red}, central_wavelength_nm=None
    )
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
def test_open_raster_picks_the_backend_each_source_needs(
    tmp_path: Path, name: str
) -> None:
    build, expected_name = _BACKENDS[name]
    source, num_channels = build(tmp_path)
    with open_raster(source, num_channels) as src:
        assert isinstance(src, getattr(raster_source, expected_name))


def test_a_photographic_extension_at_a_count_pil_has_no_mode_for_refuses(
    tmp_path: Path,
) -> None:
    """A photographic extension at a count PIL has no mode for refuses.

    The refusal says a 5-channel image cannot be loaded from a ``.png``.
    """
    source, _ = _photographic(tmp_path)
    with pytest.raises(
        ValueError, match=r"Cannot load a 5-channel image from '\.png'"
    ):
        open_raster(source, 5)


@pytest.mark.parametrize("layout", ["striped", "tiled", "planar"])
def test_single_dataset_tiff_layouts_are_served_windowed(
    tmp_path: Path, layout: str
) -> None:
    """Striped, tiled and planar single-dataset TIFFs are served windowed with their own pixels.

    Striped, internally tiled, and planar (band-separate) single-dataset TIFFs all read through
    the windowed GDAL backend, and the pixels served are the source array's own regardless of the
    on-disk layout.
    """
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
    assert (spec.backend, spec.resample) == ("gdal", None)


def test_a_planar_raster_read_at_a_mismatched_count_still_reads_as_stored(
    tmp_path: Path,
) -> None:
    """A planar raster read at a mismatched count still reads as stored.

    A four-band planar file opened at three channels serves its own 40x32 frame and four bands,
    and its full region equals the stored array.
    """
    arr = _distinctive_array(32, 40, channels=4)
    path = tmp_path / "planar4.tif"
    _write_planar_tiff(path, arr)

    with open_raster(path, 3) as src:
        assert (src.width, src.height, src.num_channels) == (40, 32, 4)
        region, _spec = src.read_region(Rect(0, 0, 40, 32))
    assert np.array_equal(region, arr)


def test_a_stacked_multipage_tiff_still_reads_whole(tmp_path: Path) -> None:
    """A stacked multi-page TIFF still reads whole, exactly as ``tifffile.imread`` does.

    The multi-page stack tifffile writes (one row-block per page) opens through the whole decode,
    and its full region equals ``tifffile.imread``'s array.
    """
    path, num_channels = _whole_tiff(tmp_path)
    with tifffile.TiffFile(str(path)) as tif:
        assert (
            len(tif.pages) > 1
        )  # the layout under test, not a single-page file
    with open_raster(path, num_channels) as src:
        assert isinstance(src, TiffWholeSource)
        region, _spec = src.read_region(Rect(0, 0, src.width, src.height))
    assert np.array_equal(region, tifffile.imread(str(path)))


def test_a_whole_decoded_tiff_is_opened_once_and_never_through_gdal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A whole-decoded TIFF is opened once and never through GDAL.

    Opening and reading a stacked TIFF constructs one ``tifffile.TiffFile`` and opens no GDAL
    dataset.
    """
    path, num_channels = _whole_tiff(tmp_path)
    tiffs: list = []
    real_init = tifffile.TiffFile.__init__

    def counted(
        self: tifffile.TiffFile, file: object, *args: object, **kwargs: object
    ) -> None:
        tiffs.append(file)
        real_init(self, file, *args, **kwargs)

    def no_gdal(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"GDAL opened {args}")

    monkeypatch.setattr(tifffile.TiffFile, "__init__", counted)
    monkeypatch.setattr(raster_source, "open_gdal_dataset", no_gdal)
    with open_raster(path, num_channels) as src:
        region, _spec = src.read_region(Rect(0, 0, src.width, src.height))
    assert region.shape == (20, 14, 5)
    assert len(tiffs) == 1


def test_an_array_reader_materializes_the_layout_it_is_handed() -> None:
    """An array reader materializes the layout it is handed.

    The decoded shape and axis order an array reader serves are its layout's: a layout stating a
    12x8 decode of 96 values stored 8x12 is the frame served.
    """
    values = np.arange(96, dtype=np.uint8).reshape(8, 12)
    layout = raster_source._array_layout((12, 8), 1, Path("stack.npz"))
    with raster_source.NpzSource(values, layout) as src:
        assert (src.height, src.width, src.num_channels) == (12, 8, 1)
        region, _spec = src.read_region(Rect(0, 0, 8, 12))
    assert np.array_equal(region[:, :, 0], values.reshape(12, 8))


def test_a_shape_the_whole_decode_would_transpose_is_not_served_windowed(
    tmp_path: Path,
) -> None:
    """A three-row, four-band TIFF opened at three channels is not served windowed.

    The whole decode serves it as a 20x4x3 frame, matching ``image_dimensions``; opened at four
    channels, GDAL serves it as stored, 3x20x4.
    """
    from tcip_mcp.pipelines.image_utils import image_dimensions

    path = tmp_path / "three_row.tif"
    arr = _distinctive_array(3, 20, channels=4)
    tifffile.imwrite(
        str(path),
        arr,
        photometric="rgb",
        extrasamples=["unassalpha"],
        rowsperstrip=1,
    )

    with open_raster(path, 3) as src:
        assert isinstance(src, TiffWholeSource)
        region, _spec = src.read_region(Rect(0, 0, src.width, src.height))
    assert region.shape == (20, 4, 3)
    assert image_dimensions(path, 3) == (4, 20)

    # At the count the reinterpretation leaves alone, the windowed backend serves it as stored.
    with open_raster(path, 4) as src:
        assert isinstance(src, raster_source.GdalSource)
        assert (src.height, src.width, src.num_channels) == (3, 20, 4)


def _one_row_rgb(tmp_path: Path) -> tuple[Path, int]:
    """Write a 1 by 200,000 three-plane array as an RGB TIFF; return its path and the count 3."""
    path = tmp_path / "rgb_row.tif"
    tifffile.imwrite(
        str(path),
        np.tile(np.array([20, 100, 220], dtype=np.uint8), (1, 200_000, 1)),
        photometric="rgb",
    )
    return path, 3


def _planar(tmp_path: Path) -> tuple[Path, int]:
    path = tmp_path / "planar.tif"
    _write_planar_tiff(path, _distinctive_array(32, 40, channels=4))
    return path, 4


def _palette(tmp_path: Path) -> tuple[Path, int]:
    """Write a 23 by 17 palette-color TIFF; return its path and the channel count 3.

    Its indices are row-major ``arange % 256``, and its table maps index ``i`` to
    ``(i, 255 - i, 7 * i % 256)``.
    """
    from PIL import Image

    indices = (
        (np.arange(23 * 17, dtype=np.uint16) % 256)
        .astype(np.uint8)
        .reshape(23, 17)
    )
    img = Image.fromarray(indices, mode="P")
    img.putpalette(
        [v for i in range(256) for v in (i, 255 - i, (i * 7) % 256)]
    )
    path = tmp_path / "palette.tif"
    img.save(str(path))
    return path, 3


_TIFF_LAYOUTS = {
    "one_row_rgb": _one_row_rgb,
    "palette": _palette,
    "planar": _planar,
    "striped": _gdal_tiff,
    "stacked": _whole_tiff,
}


@pytest.mark.parametrize("name", sorted(_TIFF_LAYOUTS))
def test_the_probe_and_the_frame_read_one_band_count_off_a_tiff_header(
    tmp_path: Path,
    name: str,
) -> None:
    """``probe_channels`` and ``image_dimensions`` read one band count off a TIFF header.

    The two read the header the same way: the probe's count is the count the file serves at, the
    frame served at that count is the one measured, and it is the frame the header's own layout
    states with no hint, for a one-row RGB, a palette, a planar, a striped and a stacked TIFF.
    """
    from tcip_mcp.pipelines.derivations import probe_channels
    from tcip_mcp.pipelines.image_utils import image_dimensions

    path, bands = _TIFF_LAYOUTS[name](tmp_path)
    probed = probe_channels(path)
    assert probed == bands
    own = raster_source._layout(
        raster_source.SourceHeader(path).tiff, None
    ).frame
    with open_raster(path, probed) as src:
        assert (src.height, src.width, src.num_channels) == own
        assert image_dimensions(path, probed) == (src.width, src.height)


def test_a_tiff_series_that_is_not_a_2d_image_is_refused_by_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A TIFF series that is not a 2D image is refused, naming its shape.

    A four-axis series has no frame this platform reads; the probe, the frame and the open refuse
    it, and the open refuses before any GDAL handle is acquired.
    """
    from tcip_mcp.pipelines.derivations import probe_channels

    path = tmp_path / "volume.tif"
    tifffile.imwrite(str(path), np.zeros((2, 5, 20, 30), dtype=np.uint8))
    with pytest.raises(ValueError, match=r"\(2, 5, 20, 30\)"):
        probe_channels(path)
    with pytest.raises(ValueError, match=r"\(2, 5, 20, 30\)"):
        raster_source.SourceHeader(path).frame_at(5)

    opened: list[str] = []
    real_open = raster_source.open_gdal_dataset

    def recording_open(
        *args: str | Path | int | None, **kwargs: int | None
    ) -> DatasetReader:
        opened.append(str(args[0]))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(raster_source, "open_gdal_dataset", recording_open)
    with pytest.raises(ValueError, match=r"\(2, 5, 20, 30\)"):
        open_raster(path, 5)
    assert opened == []


def test_the_whole_decode_serves_the_frame_the_header_describes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole decode serves the frame the header describes.

    A TIFF the whole decode serves, because GDAL cannot open it or the hint reinterprets it, comes
    out in the header's frame: a planar file channel-last at its own count and at a
    mismatched one, a palette file expanded through its table, each equal to the pixels GDAL
    serves for the same file.
    """
    planar, _ = _planar(tmp_path)
    palette, _ = _palette(tmp_path)
    expected = {}
    for path, bands in ((planar, 4), (palette, 3)):
        with open_raster(path, bands) as src:
            assert isinstance(src, raster_source.GdalSource)
            expected[path] = src.read_region(
                Rect(0, 0, src.width, src.height)
            )[0]

    def refuse(*_a: object, **_k: object) -> None:
        raise ValueError("GDAL cannot open raster")

    monkeypatch.setattr(raster_source, "open_gdal_dataset", refuse)
    for path, bands in ((planar, 4), (planar, 3), (palette, 3)):
        with open_raster(path, bands) as whole:
            assert isinstance(whole, TiffWholeSource)
            assert (whole.height, whole.width, whole.num_channels) == expected[
                path
            ].shape
            assert np.array_equal(
                whole.read_region(Rect(0, 0, whole.width, whole.height))[0],
                expected[path],
            )


def test_a_palette_row_reinterpreted_by_its_hint_serves_its_expanded_pixels(
    tmp_path: Path,
) -> None:
    """A palette row reinterpreted by its hint serves its expanded pixels.

    A one-row, 20-pixel palette TIFF opened at hint 1 is served whole at the header's 3 by 20
    frame, one channel, and its region is PIL's expanded colors transposed channel-first.
    """
    from PIL import Image

    indices = np.arange(20, dtype=np.uint8).reshape(1, 20)
    img = Image.fromarray(indices, mode="P")
    img.putpalette(
        [v for i in range(256) for v in (i, 255 - i, (i * 7) % 256)]
    )
    path = tmp_path / "palette_row.tif"
    img.save(str(path))
    rgb = np.asarray(img.convert("RGB"))

    assert raster_source.SourceHeader(path).frame_at(1) == (3, 20)
    with open_raster(path, 1) as src:
        assert isinstance(src, TiffWholeSource)
        assert (src.height, src.width, src.num_channels) == (20, 3, 1)
        region, _spec = src.read_region(Rect(0, 0, 3, 20))
    assert np.array_equal(region, np.transpose(rgb, (1, 2, 0)))


def _cut_colormap_to_six_entries(path: Path) -> None:
    """Cut ``path``'s ``ColorMap`` tag to six values in place.

    Only the IFD entry's count field is rewritten (tag 320, SHORT, 768 values, little-endian); the
    helper asserts the file holds exactly one such entry.
    """
    data = bytearray(path.read_bytes())
    entry = b"\x40\x01\x03\x00\x00\x03\x00\x00"
    at = data.find(entry)
    assert at >= 0 and data.find(entry, at + 1) < 0, (
        "one ColorMap entry of 768 values"
    )
    data[at + 4 : at + 8] = (6).to_bytes(4, "little")
    path.write_bytes(bytes(data))


def test_a_palette_tiff_the_table_cannot_expand_is_refused_everywhere(
    tmp_path: Path,
) -> None:
    """A palette TIFF its table cannot expand is refused by the probe, the frame and the open.

    The image's color table holds fewer entries than its uint8 indices address, and each of the
    three refuses saying it holds no RGB palette.
    """
    from tcip_mcp.pipelines.derivations import probe_channels

    path, _ = _palette(tmp_path)
    _cut_colormap_to_six_entries(path)
    with tifffile.TiffFile(str(path)) as tif:
        assert tif.pages[0].tags["ColorMap"].value.shape == (3, 2)

    with pytest.raises(
        ValueError, match=r"shaped \(3, 2\) holds no RGB palette"
    ):
        probe_channels(path)
    with pytest.raises(ValueError, match="holds no RGB palette"):
        raster_source.SourceHeader(path).frame_at(3)
    with pytest.raises(ValueError, match="holds no RGB palette"):
        open_raster(path, 3)


def test_an_unreadable_tiff_fails_naming_the_file(tmp_path: Path) -> None:
    """An unreadable TIFF is refused, named, by the probe, the frame and the open."""
    from tcip_mcp.pipelines.derivations import probe_channels

    path = tmp_path / "broken.tif"
    path.write_bytes(b"II*\x00garbage that is not a real tiff")
    with pytest.raises(ValueError, match=r"broken\.tif"):
        probe_channels(path)
    with pytest.raises(ValueError, match=r"broken\.tif"):
        raster_source.SourceHeader(path).frame_at(3)
    with pytest.raises(ValueError, match=r"broken\.tif"):
        open_raster(path, 3)


def test_a_plain_read_serves_full_resolution(tmp_path: Path) -> None:
    """A read without a ``target_size`` is native resolution, resampled by nothing."""
    source, num_channels = _gdal_tiff(tmp_path)
    with open_raster(source, num_channels) as src:
        region, spec = src.read_region(Rect(0, 0, 4, 4))
    assert region.shape[:2] == (4, 4)
    assert (spec.backend, spec.resample) == ("gdal", None)


# ── The contracts every backend shares ───────────────────────────────────


@pytest.mark.parametrize("name", sorted(_BACKENDS))
def test_a_returned_region_is_a_copy_a_caller_may_mutate(
    tmp_path: Path, name: str
) -> None:
    """A returned region is a copy a caller may mutate, whichever backend served it.

    Mutating it does not corrupt what a later read of the same region returns.
    """
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
def test_a_region_outside_the_raster_refuses(
    tmp_path: Path, name: str
) -> None:
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


def test_read_window_is_rows_first_and_matches_the_numpy_slice(
    tmp_path: Path,
) -> None:
    """``read_window(y0, y1, x0, x1)`` is exactly ``arr[y0:y1, x0:x1]``.

    The window is row-first against ``Rect``'s x-first order, checked on a 10 by 8 window.
    """
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
    """Windowed and whole decodes of one TIFF agree with tifffile's own read.

    The same file assembled from windowed reads, decoded whole, and read by tifffile itself is the
    same pixels.
    """
    arr = _distinctive_array(23, 17)
    path = tmp_path / "win.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)

    with open_raster(path, 3) as source:
        bands = [
            source.read_window(y0, min(y0 + 5, 23), 0, 17)
            for y0 in range(0, 23, 5)
        ]
    assembled = np.concatenate(bands, axis=0)

    header = raster_source.SourceHeader(path)
    with TiffWholeSource(
        header, raster_source._layout(header.tiff, 3)
    ) as whole:
        decoded, _spec = whole.read_region(
            Rect(0, 0, whole.width, whole.height)
        )

    assert np.array_equal(assembled, decoded)
    assert np.array_equal(decoded, tifffile.imread(str(path)))


def test_read_window_returns_a_copy_not_a_cached_view(tmp_path: Path) -> None:
    """``read_window`` returns a copy, not a cached view.

    After a returned window is overwritten, a later read of the same window equals the stored
    pixels.
    """
    arr = _distinctive_array(23, 17)
    path = tmp_path / "win.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)
    with open_raster(path, 3) as source:
        first = source.read_window(0, 4, 0, 17)
        first[:] = 255
        second = source.read_window(0, 4, 0, 17)
    assert np.array_equal(second, arr[0:4, 0:17])


def test_a_single_band_raster_serves_with_a_trailing_channel_axis(
    tmp_path: Path,
) -> None:
    arr = (
        (np.arange(23 * 17, dtype=np.uint16) % 256)
        .astype(np.uint8)
        .reshape(23, 17)
    )
    path = tmp_path / "gray.tif"
    tifffile.imwrite(str(path), arr, rowsperstrip=4)
    with open_raster(path, 1) as source:
        assert (source.height, source.width, source.num_channels) == (
            23,
            17,
            1,
        )
        window = source.read_window(3, 13, 2, 10)
    assert window.shape == (10, 8, 1)
    assert np.array_equal(np.squeeze(window, -1), arr[3:13, 2:10])


def test_a_palette_tiff_expands_through_its_color_table_like_pil(
    tmp_path: Path,
) -> None:
    """A palette TIFF expands through its color table exactly as PIL's palette decode does.

    GDAL serves it as three uint8 channels, its region and a window equal to PIL's colors.
    """
    from PIL import Image

    indices = (
        (np.arange(23 * 17, dtype=np.uint16) % 256)
        .astype(np.uint8)
        .reshape(23, 17)
    )
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
    """A source's band interpretations name each served channel's declared interpretation.

    An RGBA TIFF names its fourth band alpha; a palette source names the three channels its
    expansion serves.
    """
    from PIL import Image

    rgba_path = tmp_path / "rgba.tif"
    _write_striped_tiff(
        rgba_path, _distinctive_array(8, 8, channels=4), rowsperstrip=4
    )
    with open_raster(rgba_path, 4) as src:
        assert src.band_interpretations == ("red", "green", "blue", "alpha")

    indices = (
        (np.arange(64, dtype=np.uint16) % 256).astype(np.uint8).reshape(8, 8)
    )
    img = Image.fromarray(indices, mode="P")
    img.putpalette(
        [v for i in range(256) for v in (i, 255 - i, (i * 7) % 256)]
    )
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
    """A GDAL-served ``target_size`` read downsamples by averaging.

    The read returns the reduced buffer, with the ``ReadSpec`` recording average resampling; at an
    exact 2x decimation the output agrees with the 2x2 block means within ``np.allclose`` at
    ``atol=1.0``.
    """
    arr = _distinctive_array(32, 40)
    path = tmp_path / "down.tif"
    _write_striped_tiff(path, arr, rowsperstrip=8)
    with open_raster(path, 3) as src:
        region, spec = src.read_region(
            Rect(0, 0, 40, 32), target_size=(20, 16)
        )
    assert region.shape == (16, 20, 3)
    assert (spec.backend, spec.resample) == ("gdal", "average")
    blocks = arr.reshape(16, 2, 20, 2, 3).mean(axis=(1, 3))
    assert np.allclose(region, blocks, atol=1.0)


def test_a_target_size_read_downsamples_an_array_backend_by_area(
    tmp_path: Path,
) -> None:
    """An array backend's ``target_size`` read slices native and downsamples by area.

    The read records ``area`` resampling, and the output agrees with the 2x2 block means within
    ``np.allclose`` at ``atol=1.0``.
    """
    arr = _distinctive_array(32, 40, channels=5)
    path = tmp_path / "bands.npy"
    np.save(str(path), arr)
    with open_raster(path, 5) as src:
        region, spec = src.read_region(
            Rect(0, 0, 40, 32), target_size=(20, 16)
        )
    assert region.shape == (16, 20, 5)
    assert (spec.backend, spec.resample) == ("npy", "area")
    blocks = arr.reshape(16, 2, 20, 2, 5).mean(axis=(1, 3))
    assert np.allclose(region, blocks, atol=1.0)


@pytest.mark.parametrize("name", ["gdal_tiff", "npy"])
def test_a_distorting_target_size_refuses(tmp_path: Path, name: str) -> None:
    """A ``target_size`` that distorts the region's aspect ratio refuses, naming the ratio."""
    build, _ = _BACKENDS[name]
    source, num_channels = build(tmp_path)
    with open_raster(source, num_channels) as src:
        with pytest.raises(ValueError, match="aspect ratio"):
            src.read_region(
                Rect(0, 0, src.width, src.height),
                target_size=(src.width, max(1, src.height // 3)),
            )


# ── The process-local pool of open sources ───────────────────────────────


def _pooled(
    source: str | Path | BandGroupRef, num_channels: int
) -> RasterSource | None:
    """Return ``pooled_source`` of ``source`` through a fresh header."""
    return raster_source.pooled_source(
        raster_source.SourceHeader(source), num_channels
    )


def test_the_pool_serves_one_open_source_per_file_and_channel_count(
    tmp_path: Path,
) -> None:
    path, _ = _gdal_tiff(tmp_path)
    first = _pooled(path, 3)
    assert isinstance(first, raster_source.GdalSource)
    assert _pooled(path, 3) is first
    assert _pooled(path, 1) is not first


def test_the_pool_holds_no_reader_for_a_source_that_decodes_whole(
    tmp_path: Path,
) -> None:
    """The pool holds no reader for a source that decodes whole.

    A whole-decode source answers no reader and leaves the pool empty, so no decoded pixels sit
    resident there.
    """
    path, num_channels = _whole_tiff(tmp_path)
    assert _pooled(path, num_channels) is None
    assert not raster_source._POOL


def test_the_pool_key_changes_when_a_band_member_is_rewritten(
    tmp_path: Path,
) -> None:
    ref, num_channels = _band_group(tmp_path)
    before = raster_source.source_version(raster_source.SourceHeader(ref))
    member = next(iter(ref.bands.values()))
    stat = member.stat()
    os.utime(member, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert (
        raster_source.source_version(raster_source.SourceHeader(ref)) != before
    )


@pytest.mark.parametrize("name", ["band_group", "gdal_tiff"])
def test_a_headers_path_is_read_off_its_source_and_never_set_beside_it(
    tmp_path: Path, name: str
) -> None:
    from tcip_mcp.pipelines.image_utils import source_path_of

    source, _ = _BACKENDS[name][0](tmp_path)
    header = raster_source.SourceHeader(source)
    assert header.path == Path(source_path_of(source))
    with pytest.raises(AttributeError):
        header.path = tmp_path / "elsewhere.tif"  # type: ignore[misc]


def test_a_forked_worker_starts_with_an_empty_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _gdal_tiff(tmp_path)
    parent_source = _pooled(path, 3)
    assert parent_source is not None
    monkeypatch.setattr(os, "getpid", lambda: 424242)
    child_source = _pooled(path, 3)
    assert child_source is not parent_source
    assert not parent_source.closed  # the parent still owns what it opened
    parent_source.close()


def test_the_pool_evicts_the_least_recently_used_source_over_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_path, num_channels = _npy(tmp_path)
    second_path = tmp_path / "other.npy"
    np.save(str(second_path), _distinctive_array(18, 11, channels=5))

    first = _pooled(first_path, num_channels)
    monkeypatch.setattr(raster_source, "_memory_budget_bytes", lambda: 1)
    second = _pooled(second_path, num_channels)

    assert first is not None and second is not None
    assert first.closed
    assert not second.closed
    assert _pooled(first_path, num_channels) is not first


def test_a_gdal_source_accounts_no_resident_pixels(tmp_path: Path) -> None:
    """A GDAL source accounts zero resident bytes."""
    path, _ = _gdal_tiff(tmp_path)
    with open_raster(path, 3) as src:
        assert isinstance(src, raster_source.GdalSource)
        assert src.resident_bytes == 0


def test_a_photographic_source_accounts_its_peak_resident_frames(
    tmp_path: Path,
) -> None:
    """A photographic source accounts its peak resident frames from the start.

    Its resident bytes cover at least two frames and are unchanged by a full ``read_region``.
    """
    path, _ = _photographic(tmp_path)
    with open_raster(path, 3) as src:
        frame_bytes = (
            src.width * src.height * src.num_channels * src.dtype.itemsize
        )
        before = src.resident_bytes
        src.read_region(Rect(0, 0, src.width, src.height))
        assert src.resident_bytes == before
        assert src.resident_bytes >= 2 * frame_bytes


@pytest.mark.parametrize("name", sorted(_BACKENDS))
def test_the_header_opens_windowed_only_the_backends_that_open_without_decoding(
    tmp_path: Path, name: str
) -> None:
    """The header opens windowed only the backends that open without decoding.

    Of the six backends, a GDAL-served TIFF and a memory-mapped .npy come back as a windowed
    reader of the backend ``open_raster`` serves them through, at its frame with no levels; a band
    group, an .npz, a photograph and a whole-decoded TIFF come back as none. Each source's header
    frame is the frame its open reader serves.
    """
    build, backend = _BACKENDS[name]
    source, num_channels = build(tmp_path)
    header = raster_source.SourceHeader(source)
    served = header.windowed(num_channels)
    with open_raster(source, num_channels) as src:
        assert header.frame_at(num_channels) == (src.width, src.height)
        if name in ("gdal_tiff", "npy"):
            assert served is not None and type(served).__name__ == backend
            with served:
                assert (served.width, served.height) == (src.width, src.height)
                assert served.level_dims() == src.level_dims() == []
        else:
            assert served is None


def test_the_header_refuses_an_unopenable_tiff_naming_it(
    tmp_path: Path,
) -> None:
    """The header's frame question refuses an unopenable TIFF, naming it."""
    path = tmp_path / "broken.tif"
    path.write_bytes(b"II*\x00garbage")
    with pytest.raises(ValueError, match=r"broken\.tif"):
        raster_source.SourceHeader(path).frame_at(3)


# ── Reads that were always valid and must stay so ────────────────────────


def test_a_five_band_geotiff_opened_at_three_channels_still_reads_five_bands(
    tmp_path: Path,
) -> None:
    """A five-band GeoTIFF opened at three channels still reads five bands.

    The load returns tifffile's own five-band array.
    """
    from tcip_mcp.pipelines.image_utils import load_image

    path = tmp_path / "five.tif"
    arr = _distinctive_array(24, 40, channels=5)
    tifffile.imwrite(str(path), arr)
    got = load_image(path, 3)
    assert np.array_equal(got, tifffile.imread(str(path)))
    assert got.shape == (24, 40, 5)


def test_image_dimensions_of_a_two_band_group_at_any_channel_count(
    tmp_path: Path,
) -> None:
    """A two-band group's image dimensions come from its bands when asked at three channels.

    A count of 3 need not match the two bands the group actually holds.
    """
    from tcip_mcp.pipelines.image_utils import image_dimensions

    ref, _num_channels = _band_group(tmp_path)
    assert image_dimensions(ref, 3) == (8, 8)


def test_a_band_group_whose_members_disagree_on_the_frame_refuses(
    tmp_path: Path,
) -> None:
    """A band group whose members disagree on the frame refuses.

    A 12x12 member beside an 8x8 one makes the open refuse, saying the members disagree on the
    frame.
    """
    from tcip_mcp.pipelines.data.band_groups import read_band_group_manifest
    from tests._producer_fixtures import write_band_group

    d = tmp_path / "grouped"
    d.mkdir()
    green = d / "cap_G.tif"
    red = d / "cap_R.tif"
    tifffile.imwrite(str(green), np.full((8, 8), 111, dtype=np.uint16))
    tifffile.imwrite(str(red), np.zeros((12, 12), dtype=np.uint16))
    ref = read_band_group_manifest(
        write_band_group(
            d, "cap", {"Green": green, "Red": red}, central_wavelength_nm=None
        )
    )

    with pytest.raises(ValueError, match="disagree on the frame"):
        raster_source.open_raster(ref, 2)


def test_a_channel_first_shaped_npy_is_left_alone_at_a_mismatched_count(
    tmp_path: Path,
) -> None:
    """A channel-first-shaped .npy read at a mismatched count stays exactly as stored.

    A (5, 40, 24) array read at 3 channels comes back untransposed.
    """
    from tcip_mcp.pipelines.image_utils import load_image

    path = tmp_path / "cfirst.npy"
    arr = np.arange(5 * 40 * 24, dtype=np.uint8).reshape(5, 40, 24)
    np.save(str(path), arr)
    got = load_image(path, 3)
    assert got.shape == (5, 40, 24)
    assert np.array_equal(got, arr)


# ── _RegionView: an offset read window over an already-open parent source ───────────────────


def test_region_view_reports_the_rects_own_extent_not_the_parents(
    tmp_path: Path,
) -> None:
    """A region view reports its rect's own extent, not the parent's.

    ``_RegionView``'s height/width report the rect it was constructed over, not the full parent
    source's own dims.
    """
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


def test_region_view_refuses_a_read_past_its_own_declared_bounds(
    tmp_path: Path,
) -> None:
    """A region view refuses a read past its own declared bounds.

    A window one row past the view's height, and one column past its width, each refuse, though
    both lie inside the 40 by 30 parent.
    """
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


def test_region_view_forwards_the_parents_band_interpretations(
    tmp_path: Path,
) -> None:
    """A region view forwards its parent's declared band interpretations.

    The view's interpretations equal its parent's, the fourth band named alpha.
    """
    from tcip_mcp.pipelines.raster_source import _RegionView

    rgba_path = tmp_path / "rgba.tif"
    _write_striped_tiff(
        rgba_path, _distinctive_array(40, 30, channels=4), rowsperstrip=4
    )
    with open_raster(rgba_path, 4) as parent:
        view = _RegionView(parent, Rect(5, 5, 25, 20))
        assert (
            view.band_interpretations
            == parent.band_interpretations
            == (
                "red",
                "green",
                "blue",
                "alpha",
            )
        )


def test_region_view_reports_no_band_interpretations_for_a_parent_that_carries_none(
    tmp_path: Path,
) -> None:
    """A region view reports no band interpretations for a parent that carries none.

    A .npy-backed parent carries no color-interpretation metadata at all, and the view fabricates
    none.
    """
    from tcip_mcp.pipelines.raster_source import _RegionView

    path = tmp_path / "bands.npy"
    np.save(str(path), _distinctive_array(40, 30, channels=3))
    with open_raster(path, 3) as parent:
        assert not hasattr(parent, "band_interpretations")
        view = _RegionView(parent, Rect(5, 5, 25, 20))
        assert view.band_interpretations is None


# ── raster_content_identity: one raster file's own content identity ─────────────────────────


_IDENTITY_KW = dict(seed=7, window_size=8, max_windows=50)


def _identity(
    path: Path,
    num_channels: int,
    *,
    seed: int,
    window_size: int,
    max_windows: int,
) -> RasterIdentity:
    """Return ``raster_content_identity`` of ``path`` opened at ``num_channels``."""
    from tcip_mcp.pipelines.raster_source import raster_content_identity

    with open_raster(path, num_channels) as src:
        return raster_content_identity(
            src, seed=seed, window_size=window_size, max_windows=max_windows
        )


def _matches(recorded: RasterIdentity, path: Path) -> bool:
    """Return whether ``open_as_recorded`` of ``path`` answers a fresh identity.

    It asserts the reader opens at the recorded channel count and that a fresh identity it
    answers equals the recorded one.
    """
    from tcip_mcp.pipelines.raster_source import open_as_recorded

    src, fresh = open_as_recorded(recorded, path)
    with src:
        assert src.num_channels == recorded.num_channels
    assert fresh is None or fresh == recorded
    return fresh is not None


def test_raster_content_identity_agrees_across_gdal_and_npy_backends_for_same_content(
    tmp_path: Path,
) -> None:
    """GDAL and .npy backends resolve one content identity for the same pixels.

    Two entirely different backends (a GDAL-served GeoTIFF, a memory-mapped .npy) reading the same
    pixel content resolve the same checksum and dtype.
    """
    arr = _distinctive_array(24, 20)
    tif_path = tmp_path / "content.tif"
    npy_path = tmp_path / "content.npy"
    _write_striped_tiff(tif_path, arr, rowsperstrip=4)
    np.save(str(npy_path), arr)

    tif_identity = _identity(tif_path, 3, **_IDENTITY_KW)
    npy_identity = _identity(npy_path, 3, **_IDENTITY_KW)

    assert tif_identity.pixel_checksum == npy_identity.pixel_checksum
    assert (
        tif_identity.width,
        tif_identity.height,
        tif_identity.num_channels,
    ) == (npy_identity.width, npy_identity.height, npy_identity.num_channels)
    assert tif_identity.dtype == npy_identity.dtype


def test_raster_content_identity_differs_for_different_content_same_dimensions(
    tmp_path: Path,
) -> None:
    a_path, b_path = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(str(a_path), _distinctive_array(24, 20))
    np.save(str(b_path), np.zeros((24, 20, 3), dtype=np.uint8))

    a_identity = _identity(a_path, 3, **_IDENTITY_KW)
    b_identity = _identity(b_path, 3, **_IDENTITY_KW)

    assert (a_identity.width, a_identity.height) == (
        b_identity.width,
        b_identity.height,
    )
    assert a_identity.pixel_checksum != b_identity.pixel_checksum


def test_raster_content_identity_deterministic_under_matching_recorded_parameters(
    tmp_path: Path,
) -> None:
    """Content identity is deterministic under matching recorded parameters.

    Two calls agree when both recompute under the identity's own recorded seed, window size and
    window count: the parameters that travel with the identity.
    """
    path = tmp_path / "content.npy"
    np.save(str(path), _distinctive_array(30, 22))

    first = _identity(path, 3, **_IDENTITY_KW)
    second = _identity(
        path,
        3,
        seed=first.seed,
        window_size=first.window_size,
        max_windows=first.max_windows,
    )

    assert first.pixel_checksum == second.pixel_checksum
    assert (first.seed, first.window_size, first.max_windows) == (7, 8, 50)


def test_raster_content_identity_band_interpretations_present_only_on_gdal(
    tmp_path: Path,
) -> None:
    tif_path = tmp_path / "content.tif"
    npy_path = tmp_path / "content.npy"
    arr = _distinctive_array(24, 20)
    _write_striped_tiff(tif_path, arr, rowsperstrip=4)
    np.save(str(npy_path), arr)

    tif_identity = _identity(tif_path, 3, **_IDENTITY_KW)
    npy_identity = _identity(npy_path, 3, **_IDENTITY_KW)

    assert tif_identity.band_interpretations == ("red", "green", "blue")
    assert npy_identity.band_interpretations is None


def test_raster_content_identity_geotransform_optional_never_load_bearing(
    tmp_path: Path,
) -> None:
    """A raster with no georeferencing tags still resolves a fully usable content identity.

    The geotransform term is absent, the checksum is not.
    """
    tif_path = tmp_path / "content.tif"
    _write_striped_tiff(tif_path, _distinctive_array(24, 20), rowsperstrip=4)

    identity = _identity(tif_path, 3, **_IDENTITY_KW)
    assert identity.geotransform is None
    assert (
        identity.pixel_checksum
    )  # a real, usable identity despite no geotransform

    npy_path = tmp_path / "content.npy"
    np.save(str(npy_path), _distinctive_array(24, 20))
    npy_identity = _identity(npy_path, 3, **_IDENTITY_KW)
    assert (
        npy_identity.geotransform is None
    )  # not a GeoTIFF; still a fully usable identity


def test_an_identity_reads_its_geotransform_off_the_open_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An identity reads its geotransform off the open reader.

    A georeferenced GeoTIFF's identity carries the geotransform its tags state, and records it
    without opening the file again: the header the reader was opened from holds it.
    """
    import tifffile as tifffile_module

    from tests._geotiff_fixtures import (
        PIXEL_SCALE,
        TIEPOINT_NATIVE_X,
        TIEPOINT_NATIVE_Y,
        UTM_15N_EPSG,
        write_geotiff,
    )

    path = tmp_path / "geo.tif"
    write_geotiff(path, shape=(24, 20, 3))
    with open_raster(path, 3) as src:
        assert isinstance(src, raster_source.GdalSource)
        opened: list[str] = []
        real = tifffile_module.TiffFile

        def counting(*args: object, **kwargs: object) -> tifffile.TiffFile:
            opened.append(str(args[0]))
            return real(*args, **kwargs)

        monkeypatch.setattr(tifffile_module, "TiffFile", counting)
        identity = raster_source.raster_content_identity(src, **_IDENTITY_KW)
        monkeypatch.undo()
    assert opened == []
    assert identity.geotransform == {
        "tiepoint_pixel_x": 0.0,
        "tiepoint_pixel_y": 0.0,
        "tiepoint_native_x": TIEPOINT_NATIVE_X,
        "tiepoint_native_y": TIEPOINT_NATIVE_Y,
        "pixel_scale_x": PIXEL_SCALE,
        "pixel_scale_y": PIXEL_SCALE,
        "epsg": UTM_15N_EPSG,
    }


def test_a_georeferenced_mismatch_check_computes_one_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A georeferenced mismatch check computes one identity.

    Opening a raster as recorded computes its content identity once, and the georeferenced check
    over that fresh identity finds no mismatch.
    """
    from tests._geotiff_fixtures import write_geotiff

    path = tmp_path / "geo.tif"
    write_geotiff(path, shape=(24, 20, 3), random=True)
    with open_raster(path, 3) as src:
        recorded = raster_source.content_identity(src)
    computed: list = []
    real = raster_source.raster_content_identity

    def counting(*args: RasterSource, **kwargs: int) -> RasterIdentity:
        computed.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(raster_source, "raster_content_identity", counting)
    reader, fresh = raster_source.open_as_recorded(recorded, path)
    reader.close()
    assert (
        raster_source.georeferenced_raster_identity_mismatch(recorded, fresh)
        is None
    )
    assert len(computed) == 1


def test_a_frame_question_opens_no_gdal_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frame question opens no GDAL dataset.

    The frame of a GDAL-served TIFF, at a stated count or at the route's, comes off its header
    alone.
    """
    from tcip_mcp.pipelines.image_utils import image_dimensions

    path, num_channels = _gdal_tiff(tmp_path)
    with open_raster(path, num_channels) as src:
        assert isinstance(src, raster_source.GdalSource)
        served = (src.width, src.height)

    def refuse(*_a: object, **_k: object) -> None:
        raise AssertionError("a frame question opened a GDAL dataset")

    monkeypatch.setattr(raster_source, "open_gdal_dataset", refuse)
    assert image_dimensions(path, num_channels) == served
    assert raster_source.SourceHeader(path).display_frame == served


def _distinctive_gray(height: int, width: int) -> np.ndarray:
    """Return a single-band uint8 array of ``(row + col) % 256``."""
    arr = np.zeros((height, width), dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            arr[row, col] = (row + col) % 256
    return arr


def test_the_route_count_expands_a_grayscale_photograph_to_three(
    tmp_path: Path,
) -> None:
    """A grayscale photograph's header reports one channel and a route count of three."""
    from PIL import Image

    path = tmp_path / "gray.png"
    Image.fromarray(_distinctive_gray(20, 16), mode="L").save(path)

    header = raster_source.SourceHeader(path)
    assert (header.channels, header.route_channels) == (1, 3)


def test_a_photographs_header_and_its_decode_share_one_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A photograph's header and its decode share one PIL open.

    Reading a photograph's frame and then opening it at the route count calls ``PIL.Image.open``
    once.
    """
    import PIL.Image

    path, _ = _photographic(tmp_path)
    opened: list = []
    real = PIL.Image.open

    def counting(*args: object, **kwargs: object) -> ImageFile:
        opened.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(PIL.Image, "open", counting)
    header = raster_source.SourceHeader(path)
    frame = header.display_frame
    with header.open(header.route_channels) as src:
        assert (src.width, src.height) == frame
    assert len(opened) == 1


def test_a_grayscale_with_alpha_photograph_is_served_at_the_route_count(
    tmp_path: Path,
) -> None:
    """A grayscale-with-alpha photograph is served at the route count of three.

    A two-band photograph (PIL's ``LA``) reports two channels and a route count of three, and the
    reader at the route count serves its 16x20 display frame at three channels.
    """
    from PIL import Image

    path = tmp_path / "gray_alpha.png"
    Image.new("LA", (16, 20)).save(path)
    header = raster_source.SourceHeader(path)
    assert (header.channels, header.route_channels) == (2, 3)
    assert header.display_frame == (16, 20)
    with header.open_at_route_count() as src:
        assert (src.width, src.height, src.num_channels) == (16, 20, 3)


def test_the_route_count_leaves_an_array_container_at_its_own_band_count(
    tmp_path: Path,
) -> None:
    """The route count leaves an array container at its own band count.

    A single-band .npy raster's route count is 1.
    """
    path = tmp_path / "single_band.npy"
    np.save(str(path), _distinctive_gray(20, 16))

    assert raster_source.SourceHeader(path).route_channels == 1


# ── raster_identity_matches: the claim-scope comparison ──────────────────────────────────────


def test_raster_identity_matches_same_file(tmp_path: Path) -> None:
    path = tmp_path / "content.npy"
    np.save(str(path), _distinctive_array(24, 20))
    recorded = _identity(path, 3, **_IDENTITY_KW)
    assert _matches(recorded, path) is True


def test_raster_identity_matches_false_for_different_content(
    tmp_path: Path,
) -> None:
    a_path, b_path = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(str(a_path), _distinctive_array(24, 20))
    np.save(str(b_path), np.zeros((24, 20, 3), dtype=np.uint8))
    recorded = _identity(a_path, 3, **_IDENTITY_KW)
    assert _matches(recorded, b_path) is False


def test_raster_identity_matches_recomputes_under_the_recorded_parameters_not_a_new_default(
    tmp_path: Path,
) -> None:
    """An identity match recomputes under the recorded parameters, not a new default.

    A different seed and window size give a different checksum of the same raster, and the match
    against the recorded identity holds.
    """
    path = tmp_path / "content.npy"
    np.save(str(path), _distinctive_array(30, 22))
    recorded = _identity(path, 3, seed=1, window_size=4, max_windows=3)
    # A differently-parameterized fresh call would disagree on the checksum; matches() must not
    # take that path, it must recompute under recorded's own seed/window_size/max_windows.
    off_default = _identity(path, 3, seed=99, window_size=6, max_windows=2)
    assert off_default.pixel_checksum != recorded.pixel_checksum
    assert _matches(recorded, path) is True
