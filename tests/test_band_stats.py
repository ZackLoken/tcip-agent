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
    composite_display_rgb,
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
        elif raw.max() > 0:
            denom = float(raw.max())
        elif raw.min() < 0:
            denom = float(-raw.min())
        else:
            denom = 1.0
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
    with no spread at all, a float band whose values are negative, a float band with both signs,
    and an all-zero float band."""
    rng = np.random.default_rng(0)
    return {
        "uint8": rng.integers(0, 256, size=(7, 5)).astype(np.uint8),
        "uint16": rng.integers(0, 65536, size=(7, 5)).astype(np.uint16),
        "float32": (rng.standard_normal((7, 5)) * 100.0).astype(np.float32),
        "float32_negative": (rng.standard_normal((7, 5)) - 5.0).astype(np.float32),
        "float32_mixed_sign": (rng.standard_normal((7, 5)) * 5.0).astype(np.float32),
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


def test_a_none_stretch_of_an_all_negative_float_band_renders_black():
    """A band with no positive data divides by a positive number (the magnitude of its own
    minimum), so every pixel is negative before the clip and the render is uniformly black."""
    band = BANDS["float32_negative"]
    assert np.array_equal(stretch_band(band, "none", band.dtype), np.zeros_like(band, dtype=np.uint8))


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


# ── Stretching against bounds read from somewhere wider than the band in hand ────────────


def _derived_bounds(band, mode, orig_dtype) -> tuple[float, float]:
    """The ``(low, high)`` a mode reads off the band itself, written out independently of the
    primitive: what passing bounds in has to reproduce exactly."""
    raw = band.astype(np.float64)
    if mode == "none":
        if np.issubdtype(orig_dtype, np.integer):
            return 0.0, float(np.iinfo(orig_dtype).max) or 1.0
        return float(raw.min()), float(raw.max())
    if mode == "percent_clip":
        lo, hi = np.percentile(raw, list(DISPLAY_CLIP_PERCENTILES))
        return float(lo), float(hi)
    return float(raw.min()), float(raw.max())


@pytest.mark.parametrize("mode", STRETCH_MODES)
@pytest.mark.parametrize("name", sorted(BANDS))
def test_bounds_in_stretch_reproduces_the_stretch_that_derives_its_own_bounds(name, mode):
    band = BANDS[name]
    assert np.array_equal(stretch_band(band, mode, band.dtype, _derived_bounds(band, mode,
                                                                              band.dtype)),
                          stretch_band(band, mode, band.dtype))


@pytest.mark.parametrize("mode", STRETCH_MODES)
def test_a_region_stretched_against_the_whole_bands_bounds_matches_that_bands_own_stretch(mode):
    """The property region serving needs: two regions of one raster stretched against one bounds
    pair are the pixels that raster's whole-view stretch would have shown there."""
    band = BANDS["uint16"]
    bounds = _derived_bounds(band, mode, band.dtype)
    whole = stretch_band(band, mode, band.dtype, bounds)
    assert np.array_equal(stretch_band(band[:3], mode, band.dtype, bounds), whole[:3])
    assert np.array_equal(stretch_band(band[3:], mode, band.dtype, bounds), whole[3:])


def test_a_none_stretch_of_an_integer_band_keeps_its_dtype_ceiling_whatever_bounds_say():
    band = BANDS["uint16"]
    assert np.array_equal(stretch_band(band, "none", band.dtype, (0.0, 100.0)),
                          stretch_band(band, "none", band.dtype))


def test_a_none_stretch_of_a_float_band_divides_by_the_maximum_it_is_handed():
    """A float raster has no dtype ceiling, so a region renders against the sampled maximum of the
    whole raster instead of against its own local one."""
    band = BANDS["float32"]
    sampled_max = float(band.max()) * 2.0
    expected = np.clip(band.astype(np.float64) / sampled_max * 255.0, 0, 255).astype(np.uint8)
    assert np.array_equal(stretch_band(band, "none", band.dtype, (0.0, sampled_max)), expected)


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


def test_full_scale_denominator_is_a_float_bands_own_maximum_when_positive():
    band = BANDS["float32"]
    assert full_scale_denominator(band, band.dtype) == float(band.max())
    assert full_scale_denominator(BANDS["float32_zeros"], np.dtype("float32")) == 1.0


def test_full_scale_denominator_is_the_magnitude_of_the_minimum_for_a_non_positive_band():
    band = BANDS["float32_negative"]
    assert full_scale_denominator(band, band.dtype) == float(-band.min())


def _multiband_strip_tiff(path: Path, *, height: int = 24, width: int = 20,
                          channels: int = 4, rowsperstrip: int = 6) -> np.ndarray:
    """A small multi-band raster the windowed strip backend serves: strip-based, contiguous
    samples, one page, with per-band value levels a range check can tell apart."""
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 4096, size=(height, width, channels)).astype(np.uint16)
    tifffile.imwrite(str(path), arr, rowsperstrip=rowsperstrip)
    return arr


def test_the_multiband_render_input_is_the_preview_expression_band_for_band(tmp_path: Path):
    """Same check for the pixels the visualization tools hand a renderer for a multi-band raster:
    the first three bands through the preview expression, composited in memory."""
    from tcip_mcp.tools.vision_tools import _display_for_path

    path = tmp_path / "capture.tif"
    arr = _multiband_strip_tiff(path, channels=6)
    expected = np.stack([_preview_renderer_stretch(arr[:, :, i]) for i in (0, 1, 2)], axis=-1)
    assert np.array_equal(_display_for_path(str(path)).pixels, expected)


# ── The shared band-select, stretch and stack composite ─────────────────────────────────


def _multiband_array(channels: int = 4, dtype="uint16") -> np.ndarray:
    """A small multi-band array with a different value level per band, so a band selection that
    reorders or repeats bands is visible in the composite."""
    rng = np.random.default_rng(21)
    arr = rng.integers(0, 4096, size=(6, 5, channels))
    return (arr + np.arange(channels) * 17).astype(dtype)


@pytest.mark.parametrize("mode", STRETCH_MODES)
@pytest.mark.parametrize("idxs", [[0, 1, 2], [3, 0, 1], [2, 2, 2]])
def test_the_composite_is_each_selected_band_through_the_display_expression(mode, idxs):
    arr = _multiband_array()
    expected = np.stack([_composite_route_stretch(arr[:, :, i], mode, arr.dtype) for i in idxs],
                        axis=-1)
    assert np.array_equal(composite_display_rgb(arr, idxs, mode), expected)


def test_the_composite_is_uint8_rgb_whatever_the_sources_dtype():
    for dtype in ("uint8", "uint16", "float32"):
        out = composite_display_rgb(_multiband_array(dtype=dtype), [0, 1, 2], "minmax")
        assert out.dtype == np.uint8
        assert out.shape == (6, 5, 3)


def test_a_composite_of_a_2d_array_reads_it_as_one_band():
    band = BANDS["uint16"]
    out = composite_display_rgb(band, [0, 0, 0], "minmax")
    assert out.shape == band.shape + (3,)
    assert np.array_equal(out[:, :, 0], _composite_route_stretch(band, "minmax", band.dtype))


def test_a_composite_stretches_each_band_against_the_bounds_it_is_handed():
    arr = _multiband_array()
    bounds = [(100.0, 900.0), (0.0, 4095.0), (2000.0, 2100.0)]
    raw = arr.astype(np.float64)
    expected = np.stack(
        [np.clip((raw[:, :, i] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
         for i, (lo, hi) in zip([1, 2, 3], bounds)], axis=-1)
    assert np.array_equal(composite_display_rgb(arr, [1, 2, 3], "percent_clip", bounds), expected)


def test_a_none_composite_of_an_integer_source_divides_by_that_dtypes_ceiling():
    arr = _multiband_array(dtype="uint16")
    expected = np.clip(arr[:, :, :3].astype(np.float64) / 65535.0 * 255.0, 0, 255).astype(np.uint8)
    assert np.array_equal(composite_display_rgb(arr, [0, 1, 2], "none"), expected)


def test_a_none_composite_of_a_float_source_divides_by_each_bands_own_maximum():
    arr = _multiband_array(dtype="float32")
    raw = arr.astype(np.float64)
    expected = np.stack([np.clip(raw[:, :, i] / float(raw[:, :, i].max()) * 255.0, 0, 255)
                         for i in range(3)], axis=-1).astype(np.uint8)
    assert np.array_equal(composite_display_rgb(arr, [0, 1, 2], "none"), expected)


def test_a_composite_refuses_a_selection_it_cannot_display():
    arr = _multiband_array()
    with pytest.raises(ValueError, match="3 bands"):
        composite_display_rgb(arr, [0, 1], "minmax")
    with pytest.raises(ValueError, match="out of range"):
        composite_display_rgb(arr, [0, 1, 9], "minmax")
    with pytest.raises(ValueError, match="out of range"):
        composite_display_rgb(arr, [0, 1, -1], "minmax")
    with pytest.raises(ValueError, match="one \\(low, high\\) pair"):
        composite_display_rgb(arr, [0, 1, 2], "minmax", [(0.0, 1.0)])
    with pytest.raises(ValueError, match="stretch mode"):
        composite_display_rgb(arr, [0, 1, 2], "bogus")


# ── Sampled ranges: read through the windowed raster layer, never a full decode ──────────


def test_sampled_band_ranges_read_through_the_windowed_backend(tmp_path: Path):
    path = tmp_path / "mosaic.tif"
    arr = _multiband_strip_tiff(path)
    with raster_source.open_raster(path, arr.shape[-1]) as src:
        assert isinstance(src, raster_source.GdalSource)


def test_sampled_band_ranges_over_full_coverage_equal_the_exact_ranges(tmp_path: Path):
    path = tmp_path / "mosaic.tif"
    arr = _multiband_strip_tiff(path)
    sampled = sampled_band_ranges(path, arr.shape[-1], seed=3, window_size=8, max_windows=999,
                                  reservoir_size=4096)
    assert sampled.sampling.pixel_fraction == 1.0
    assert sampled.ranges == band_ranges(arr)


def test_partial_sampled_band_ranges_are_the_exact_ranges_of_the_windows_recorded(tmp_path: Path):
    path = tmp_path / "mosaic.tif"
    arr = _multiband_strip_tiff(path)
    sampled = sampled_band_ranges(path, arr.shape[-1], seed=11, window_size=8, max_windows=3,
                                  reservoir_size=4096)

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
    kwargs = {"seed": 11, "window_size": 8, "max_windows": 3, "reservoir_size": 4096}
    assert (sampled_band_ranges(path, arr.shape[-1], **kwargs)
            == sampled_band_ranges(path, arr.shape[-1], **kwargs))


def test_sampled_band_ranges_label_says_the_numbers_are_a_sample(tmp_path: Path):
    path = tmp_path / "mosaic.tif"
    arr = _multiband_strip_tiff(path)
    sampled = sampled_band_ranges(path, arr.shape[-1], seed=11, window_size=8, max_windows=3,
                                  reservoir_size=4096)
    assert sampled.sampling.label.startswith("sampled from 3 pixel window(s), seed 11")


# ── Clip bounds off the sampled pass: the same walk, a bounded reservoir ────────────────


def test_sampled_clip_bounds_are_exact_when_the_reservoir_holds_every_pixel_walked(tmp_path: Path):
    """A reservoir at least as large as the pixels the windows cover keeps all of them, so the cut
    points are that band's own percentiles and not an estimate of them."""
    path = tmp_path / "mosaic.tif"
    arr = _multiband_strip_tiff(path)
    sampled = sampled_band_ranges(path, arr.shape[-1], seed=3, window_size=8, max_windows=999,
                                  reservoir_size=arr.shape[0] * arr.shape[1])

    assert sampled.sampling.pixel_fraction == 1.0
    assert sampled.clip_sample_size == arr.shape[0] * arr.shape[1]
    assert sampled.percentiles == DISPLAY_CLIP_PERCENTILES
    assert sampled.clip_bounds == [clip_bounds(arr[:, :, i]) for i in range(arr.shape[-1])]


def test_sampled_clip_bounds_answer_the_percentiles_they_were_asked_for(tmp_path: Path):
    path = tmp_path / "mosaic.tif"
    arr = _multiband_strip_tiff(path)
    sampled = sampled_band_ranges(path, arr.shape[-1], seed=3, window_size=8, max_windows=999,
                                  reservoir_size=arr.shape[0] * arr.shape[1],
                                  percentiles=(10.0, 90.0))

    assert sampled.percentiles == (10.0, 90.0)
    assert sampled.clip_bounds == [clip_bounds(arr[:, :, i], (10.0, 90.0))
                                   for i in range(arr.shape[-1])]


def test_the_clip_reservoir_holds_at_most_the_size_it_was_given(tmp_path: Path):
    """The bound that lets one seeded pass describe a raster of any size: the values the cut points
    are read off never outgrow the reservoir, however many pixels the windows cover."""
    path = tmp_path / "mosaic.tif"
    arr = _multiband_strip_tiff(path, height=200, width=160, channels=3, rowsperstrip=20)
    sampled = sampled_band_ranges(path, arr.shape[-1], seed=5, window_size=40, max_windows=999,
                                  reservoir_size=500)

    assert sampled.sampling.pixel_fraction == 1.0
    assert sampled.clip_sample_size == 500

    # The estimate off 500 of 32000 pixels is not the exact cut point; it lands inside the band's
    # own neighbouring quantiles rather than at an arbitrary value.
    for i, (low, high) in enumerate(sampled.clip_bounds):
        band = arr[:, :, i].astype(np.float64)
        assert np.percentile(band, 0.5) <= low <= np.percentile(band, 5.0)
        assert np.percentile(band, 95.0) <= high <= np.percentile(band, 99.5)


def test_sampled_clip_bounds_repeat_exactly_for_the_same_seed(tmp_path: Path):
    path = tmp_path / "mosaic.tif"
    arr = _multiband_strip_tiff(path, height=200, width=160, channels=3, rowsperstrip=20)
    kwargs = {"window_size": 40, "max_windows": 4, "reservoir_size": 300}
    first = sampled_band_ranges(path, arr.shape[-1], seed=5, **kwargs)
    again = sampled_band_ranges(path, arr.shape[-1], seed=5, **kwargs)
    other = sampled_band_ranges(path, arr.shape[-1], seed=6, **kwargs)

    assert first.clip_sample_size == 300  # the reservoir replaced, it did not just fill
    assert first == again
    assert first.clip_bounds != other.clip_bounds


def test_sampled_band_ranges_refuses_a_reservoir_it_cannot_fill(tmp_path: Path):
    path = tmp_path / "mosaic.tif"
    arr = _multiband_strip_tiff(path)
    with pytest.raises(ValueError, match="reservoir_size must be positive"):
        sampled_band_ranges(path, arr.shape[-1], seed=3, window_size=8, max_windows=999,
                            reservoir_size=0)


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
