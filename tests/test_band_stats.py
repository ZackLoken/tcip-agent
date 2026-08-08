"""The display band-statistics primitive: the 8-bit stretch every band render goes through, the
exact per-band ranges the bands endpoint reports, and the sampled ranges a raster too large to
decode whole is described by.

The stretch tests hold the primitive to the arithmetic the band-composite route and the band
preview renderer each display, written out here independently so the check is against the
expression, never against the primitive itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from tcip_mcp.pipelines import raster_source
from tcip_mcp.pipelines.band_stats import (
    DISPLAY_CLIP_PERCENTILES,
    STRETCH_MODES,
    BandRange,
    band_ranges,
    clip_bounds,
    full_scale_denominator,
    sampled_band_ranges,
    stretch_band,
)


def _composite_route_stretch(band, mode, orig_dtype):
    """The per-band stretch a band composite is displayed through, band by band and independently:
    the ``uint8`` pixels the primitive has to reproduce byte for byte."""
    raw = band.astype(np.float64)
    if mode == "none":
        if np.issubdtype(orig_dtype, np.integer):
            denom = float(np.iinfo(orig_dtype).max) or 1.0
        else:
            denom = float(raw.max()) or 1.0
        out = raw / denom * 255.0
    elif mode == "percent_clip":
        lo, hi = np.percentile(raw, [2.0, 98.0])
        out = (raw - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(raw)
    else:
        lo, hi = float(raw.min()), float(raw.max())
        out = (raw - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(raw)
    return np.clip(out, 0, 255).astype(np.uint8)


def _preview_renderer_stretch(band):
    """The per-band stretch a throwaway band preview is written with: a min-max span with no clip
    of its own, which is the same pixels as the composite route's ``minmax`` because a band's own
    values can never leave its own min-max span."""
    raw = band.astype(np.float64)
    lo, hi = float(raw.min()), float(raw.max())
    stretched = (raw - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(raw)
    return stretched.astype(np.uint8)


def _bands() -> dict[str, np.ndarray]:
    """One band per case the stretch has to survive: each integer width, a float raster, a band
    with no spread at all, a float band whose values are negative, and an all-zero float band."""
    rng = np.random.default_rng(0)
    return {
        "uint8": rng.integers(0, 256, size=(7, 5)).astype(np.uint8),
        "uint16": rng.integers(0, 65536, size=(7, 5)).astype(np.uint16),
        "float32": (rng.standard_normal((7, 5)) * 100.0).astype(np.float32),
        "float32_negative": (rng.standard_normal((7, 5)) - 5.0).astype(np.float32),
        "uint16_constant": np.full((7, 5), 321, dtype=np.uint16),
        "float32_zeros": np.zeros((7, 5), dtype=np.float32),
    }


BANDS = _bands()


@pytest.mark.parametrize("mode", STRETCH_MODES)
@pytest.mark.parametrize("name", sorted(BANDS))
def test_stretch_band_is_byte_identical_to_the_composite_display_expression(name, mode):
    band = BANDS[name]
    assert np.array_equal(stretch_band(band, mode, band.dtype),
                          _composite_route_stretch(band, mode, band.dtype))


@pytest.mark.parametrize("name", sorted(BANDS))
def test_minmax_stretch_is_byte_identical_to_the_band_preview_expression(name):
    band = BANDS[name]
    assert np.array_equal(stretch_band(band, "minmax", band.dtype),
                          _preview_renderer_stretch(band))


@pytest.mark.parametrize("mode", STRETCH_MODES)
def test_stretch_band_returns_uint8_display_pixels(mode):
    out = stretch_band(BANDS["uint16"], mode, BANDS["uint16"].dtype)
    assert out.dtype == np.uint8
    assert out.shape == BANDS["uint16"].shape


def test_stretch_band_refuses_a_mode_it_does_not_implement():
    with pytest.raises(ValueError, match="stretch mode"):
        stretch_band(BANDS["uint8"], "bogus", BANDS["uint8"].dtype)


def test_band_ranges_reports_each_bands_own_exact_min_and_max():
    arr = np.stack([np.full((4, 3), 7, dtype=np.uint16),
                    np.arange(12, dtype=np.uint16).reshape(4, 3)], axis=-1)
    assert band_ranges(arr) == [BandRange(7.0, 7.0), BandRange(0.0, 11.0)]


def test_band_ranges_reads_a_2d_array_as_one_band():
    assert band_ranges(np.arange(12, dtype=np.uint16).reshape(4, 3)) == [BandRange(0.0, 11.0)]


def test_clip_bounds_cuts_at_the_display_clip_percentiles():
    band = BANDS["uint16"]
    expected = np.percentile(band.astype(np.float64), list(DISPLAY_CLIP_PERCENTILES))
    assert clip_bounds(band) == (float(expected[0]), float(expected[1]))


def test_full_scale_denominator_is_the_dtype_ceiling_for_an_integer_raster():
    assert full_scale_denominator(BANDS["uint16"], np.dtype("uint16")) == 65535.0
    assert full_scale_denominator(BANDS["uint8"], np.dtype("uint8")) == 255.0


def test_full_scale_denominator_is_a_float_bands_own_maximum():
    band = BANDS["float32"]
    assert full_scale_denominator(band, band.dtype) == float(band.max())
    assert full_scale_denominator(BANDS["float32_zeros"], np.dtype("float32")) == 1.0


def _multiband_strip_tiff(path: Path, *, height: int = 24, width: int = 20,
                          channels: int = 4, rowsperstrip: int = 6) -> np.ndarray:
    """A small multi-band raster the windowed strip backend serves: strip-based, contiguous
    samples, one page, with per-band value levels a range check can tell apart."""
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 4096, size=(height, width, channels)).astype(np.uint16)
    tifffile.imwrite(str(path), arr, rowsperstrip=rowsperstrip)
    return arr


@pytest.mark.parametrize("mode", STRETCH_MODES)
@pytest.mark.parametrize("tokens,idxs", [(None, [0, 1, 2]), (["3", "0", "1"], [3, 0, 1])])
def test_the_served_band_composite_is_the_display_expression_band_for_band(
    tmp_path: Path, mode, tokens, idxs,
):
    """The RGB the band-composite route hands its JPEG encoder, band selection included, is the
    display expression's own pixels: what a viewer is served is unchanged by routing the stretch
    through the shared primitive."""
    from tcip_web.routes.images import _composite_bands

    path = tmp_path / "capture.tif"
    arr = _multiband_strip_tiff(path)
    expected = np.stack([_composite_route_stretch(arr[:, :, i], mode, arr.dtype) for i in idxs],
                        axis=-1)
    assert np.array_equal(np.asarray(_composite_bands(path, tokens, mode)), expected)


def test_the_band_preview_render_is_the_preview_expression_band_for_band(tmp_path: Path,
                                                                        monkeypatch):
    """Same check for the throwaway multi-band preview the visualization tools materialize: its
    written pixels are the preview expression's own."""
    from PIL import Image

    from tcip_mcp.tools.vision_tools import _band_preview_png

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    arr = _multiband_strip_tiff(tmp_path / "unused.tif")
    written = np.asarray(Image.open(_band_preview_png(arr, "capture")))
    expected = np.stack([_preview_renderer_stretch(arr[:, :, i]) for i in (0, 1, 2)], axis=-1)
    assert np.array_equal(written, expected)


# ── Sampled ranges: read through the windowed raster layer, never a full decode ──────────


def test_sampled_band_ranges_read_through_the_windowed_backend(tmp_path: Path):
    path = tmp_path / "mosaic.tif"
    arr = _multiband_strip_tiff(path)
    with raster_source.open_raster(path, arr.shape[-1]) as src:
        assert isinstance(src, raster_source.StripTiffSource)


def test_sampled_band_ranges_over_full_coverage_equal_the_exact_ranges(tmp_path: Path):
    path = tmp_path / "mosaic.tif"
    arr = _multiband_strip_tiff(path)
    sampled = sampled_band_ranges(path, arr.shape[-1], seed=3, window_size=8, max_windows=999)
    assert sampled.sampling.pixel_fraction == 1.0
    assert sampled.ranges == band_ranges(arr)


def test_partial_sampled_band_ranges_are_the_exact_ranges_of_the_windows_recorded(tmp_path: Path):
    path = tmp_path / "mosaic.tif"
    arr = _multiband_strip_tiff(path)
    sampled = sampled_band_ranges(path, arr.shape[-1], seed=11, window_size=8, max_windows=3)

    assert len(sampled.sampling.windows) == 3
    assert 0.0 < sampled.sampling.pixel_fraction < 1.0
    covered = sum(r.width * r.height for _, r in sampled.sampling.windows)
    assert sampled.sampling.pixel_fraction == pytest.approx(covered / (arr.shape[0] * arr.shape[1]))
    assert {label for label, _ in sampled.sampling.windows} == {str(path)}

    read = np.concatenate([arr[r.y0:r.y1, r.x0:r.x1].reshape(-1, arr.shape[-1])
                           for _, r in sampled.sampling.windows], axis=0)
    assert sampled.ranges == band_ranges(read[None, :, :])


def test_sampled_band_ranges_repeat_exactly_for_the_same_seed(tmp_path: Path):
    path = tmp_path / "mosaic.tif"
    arr = _multiband_strip_tiff(path)
    kwargs = {"seed": 11, "window_size": 8, "max_windows": 3}
    assert (sampled_band_ranges(path, arr.shape[-1], **kwargs)
            == sampled_band_ranges(path, arr.shape[-1], **kwargs))


def test_sampled_band_ranges_label_says_the_numbers_are_a_sample(tmp_path: Path):
    path = tmp_path / "mosaic.tif"
    arr = _multiband_strip_tiff(path)
    sampled = sampled_band_ranges(path, arr.shape[-1], seed=11, window_size=8, max_windows=3)
    assert sampled.sampling.label.startswith("sampled from 3 pixel window(s), seed 11")


# ── The window sampler the sampled statistics are drawn with ────────────────────────────


def test_sample_windows_covers_every_pixel_once_when_it_can_take_every_cell():
    windows = raster_source.sample_windows(20, 24, seed=1, window_size=8, max_windows=999)
    covered = np.zeros((24, 20), dtype=np.int64)
    for r in windows:
        covered[r.y0:r.y1, r.x0:r.x1] += 1
    assert covered.min() == covered.max() == 1


def test_sample_windows_draws_the_requested_count_without_overlap():
    windows = raster_source.sample_windows(20, 24, seed=1, window_size=8, max_windows=4)
    assert len(windows) == 4
    covered = np.zeros((24, 20), dtype=np.int64)
    for r in windows:
        covered[r.y0:r.y1, r.x0:r.x1] += 1
    assert covered.max() == 1


def test_sample_windows_is_deterministic_per_seed():
    first = raster_source.sample_windows(200, 240, seed=5, window_size=8, max_windows=4)
    again = raster_source.sample_windows(200, 240, seed=5, window_size=8, max_windows=4)
    other = raster_source.sample_windows(200, 240, seed=6, window_size=8, max_windows=4)
    assert first == again
    assert first != other


def test_sample_windows_refuses_a_geometry_it_cannot_sample():
    with pytest.raises(ValueError, match="raster"):
        raster_source.sample_windows(0, 10, seed=1, window_size=4, max_windows=1)
    with pytest.raises(ValueError, match="positive"):
        raster_source.sample_windows(10, 10, seed=1, window_size=4, max_windows=0)
