"""Display band statistics, the 8-bit stretch every band render goes through, and the RGB composite
it stacks into.

Every number here is in the raster's own dtype units: a uint16 band's minimum is a raw digital
number, and a ``percent_clip`` bound is one too. That is a different unit system from
``derivations.band_normalization_stats``, whose per-band mean/std are the [0, 1] tensor units a
detector normalizes with, and the two never stand in for each other: these describe what a viewer
sees on screen, those describe what the model is fed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:
    from tcip_mcp.pipelines.raster_source import RasterSource

# The bounds a ``percent_clip`` render stretches between, as percentiles of the band's own values:
# the display clip the band-composite route offers a viewer.
DISPLAY_CLIP_PERCENTILES = (2.0, 98.0)

# The stretches :func:`stretch_band` implements and a caller may ask for; it refuses anything else.
# The one declaration of this vocabulary: a type annotation elsewhere uses StretchMode itself.
StretchMode = Literal["minmax", "percent_clip", "none"]
STRETCH_MODES = get_args(StretchMode)


@dataclass(frozen=True)
class BandRange:
    """One band's value range in the raster's own dtype units."""

    minimum: float
    maximum: float


@dataclass(frozen=True)
class SampledBandRanges:
    """Per-band display bounds read from a sample of a raster's pixels.

    ``ranges`` bounds the spread of the pixels actually read and never claims to be the raster's
    own min/max: a more extreme value in a window the sample missed is invisible here.
    ``clip_bounds`` holds, per band and in the same order, the requested percentile cut points of
    those same pixels, drawn from a bounded reservoir of them rather than from all of them: when
    the reservoir held every pixel the windows covered, the cut points are that sample's exact
    percentiles; below that they are estimates of them. ``seed`` reproduces the window draw and
    ``pixel_fraction`` is the share of the raster's pixels the windows covered (1.0 when they
    covered every pixel), so a caller can say what the numbers describe.
    """

    ranges: list[BandRange]
    clip_bounds: list[tuple[float, float]]
    seed: int
    pixel_fraction: float


def band_ranges(pixels) -> list[BandRange]:
    """Exact per-band min/max of an ``[H, W, C]`` array (or of one 2-D band), in its dtype units."""
    from tcip_mcp.pipelines.raster_source import hwc_array

    arr = hwc_array(pixels)
    return [BandRange(float(arr[:, :, i].min()), float(arr[:, :, i].max()))
            for i in range(arr.shape[-1])]


def clip_bounds(band, percentiles: tuple[float, float] = DISPLAY_CLIP_PERCENTILES
                ) -> tuple[float, float]:
    """The values ``percentiles`` cuts a band at, in the band's own dtype units."""
    import numpy as np

    lo, hi = np.percentile(np.asarray(band, dtype=np.float64), list(percentiles))
    return float(lo), float(hi)


def full_scale_denominator(band, source_dtype, *, sampled_maximum: float | None = None,
                           sampled_minimum: float | None = None) -> float:
    """The divisor that puts a band on its own full scale with no data-range stretch applied.

    An integer raster divides by its dtype's maximum, the scale ``image_utils.pil_to_tensor``
    applies for training; a float raster has no such ceiling and divides by its own maximum. A band
    with no positive data divides by the magnitude of its minimum instead; a band whose whole range
    is exactly zero divides by 1.0.

    ``sampled_maximum``/``sampled_minimum`` are the whole raster's sampled statistics, used in
    place of ``band``'s own range for a float raster. They never displace the dtype ceiling an
    integer raster divides by.
    """
    import numpy as np

    if np.issubdtype(source_dtype, np.integer):
        return float(np.iinfo(source_dtype).max) or 1.0
    if sampled_maximum is None:
        sampled_maximum = float(np.asarray(band).max())
    if sampled_maximum > 0:
        return float(sampled_maximum)
    if sampled_minimum is None:
        sampled_minimum = float(np.asarray(band).min())
    if sampled_minimum < 0:
        return float(-sampled_minimum)
    return 1.0


def _stretch_between(raw, low: float, high: float):
    """``raw`` (float64) rescaled so ``low``..``high`` spans the 8-bit display range.

    A band with no spread (``high <= low``) renders black rather than dividing by zero.
    """
    import numpy as np

    if high > low:
        return (raw - low) / (high - low) * 255.0
    return np.zeros_like(raw)


def stretch_band(band, mode: str, source_dtype, bounds: tuple[float, float] | None = None):
    """One band as ``uint8`` display pixels, stretched by ``mode``.

    ``minmax`` spans the band's own data range, ``percent_clip`` spans
    :data:`DISPLAY_CLIP_PERCENTILES` of it, and ``none`` applies no data-range stretch at all,
    scaling by :func:`full_scale_denominator`. ``source_dtype`` is the raster's own dtype, which
    only ``none`` reads.

    ``bounds`` is the ``(low, high)`` those same modes would otherwise derive from ``band``, taken
    from a wider view of the raster (the whole raster's bounds when rendering one region).
    ``minmax`` and ``percent_clip`` take the pair as their span; ``none`` reads both ends as the
    sampled minimum and maximum a float raster's divisor comes from, and an integer raster ignores
    the pair for its dtype ceiling. Omitting it derives the bounds from ``band``.
    """
    import numpy as np

    raw = np.asarray(band).astype(np.float64)
    if mode == "none":
        sampled_maximum, sampled_minimum = (
            (None, None) if bounds is None else (bounds[1], bounds[0])
        )
        out = raw / full_scale_denominator(
            raw, source_dtype, sampled_maximum=sampled_maximum,
            sampled_minimum=sampled_minimum) * 255.0
    elif mode == "percent_clip":
        low, high = clip_bounds(raw) if bounds is None else bounds
        out = _stretch_between(raw, low, high)
    elif mode == "minmax":
        low, high = (float(raw.min()), float(raw.max())) if bounds is None else bounds
        out = _stretch_between(raw, low, high)
    else:
        raise ValueError(f"stretch mode must be one of {sorted(STRETCH_MODES)}, got {mode!r}")
    return np.clip(out, 0, 255).astype(np.uint8)


def composite_display_rgb(pixels, band_indices, stretch: str,
                          bounds: list[tuple[float, float]] | None = None):
    """Three of ``pixels``' bands stretched independently and stacked into ``uint8 [H, W, 3]``.

    ``pixels`` is ``[H, W, C]`` (a 2-D array reads as one band) and ``band_indices`` names exactly
    three of its bands, in display order, repeats allowed; the stretch reads the array's own dtype,
    so an integer array's ``none`` mode divides by that dtype's ceiling. ``bounds`` is one ``(low,
    high)`` pair per selected band, in the same order, passed through to :func:`stretch_band`;
    omitting it derives each band's bounds from the array in hand.
    """
    import numpy as np

    from tcip_mcp.pipelines.raster_source import hwc_array

    arr = hwc_array(pixels)
    idxs = [int(i) for i in band_indices]
    if len(idxs) != 3:
        raise ValueError(f"a display composite is 3 bands, got {len(idxs)}")
    for i in idxs:
        if not 0 <= i < arr.shape[-1]:
            raise ValueError(f"band index {i} is out of range for a {arr.shape[-1]}-band array")
    if bounds is not None and len(bounds) != len(idxs):
        raise ValueError(f"bounds must hold one (low, high) pair per selected band, got "
                         f"{len(bounds)} for {len(idxs)} bands")
    per_band = [None, None, None] if bounds is None else list(bounds)
    return np.stack([stretch_band(arr[:, :, i], stretch, arr.dtype, b)
                     for i, b in zip(idxs, per_band)], axis=-1)


def _reservoir_take(reservoir, seen: int, values, size: int, rng):
    """Fold ``values`` (``[N, C]`` pixels, in stream order) into a reservoir of at most ``size``,
    returning the reservoir and how many pixels the stream has offered.

    Classic reservoir sampling over pixels rather than over per-band values: one draw decides
    whether a whole pixel is kept, so every band's retained values come from the same pixels and
    each band's reservoir is a uniform sample of that band. ``seen`` is how many pixels the stream
    has already offered; the first ``size`` are kept outright, after which the i-th pixel replaces
    a uniformly chosen slot with probability ``size / i``.
    """
    import numpy as np

    if reservoir is None:
        reservoir = np.empty((0, values.shape[1]), dtype=values.dtype)
    room = size - reservoir.shape[0]
    if room > 0:
        head = values[:room]
        reservoir = np.concatenate([reservoir, head])
        seen += head.shape[0]
        values = values[room:]
    if values.shape[0]:
        slots = rng.integers(0, np.arange(seen + 1, seen + values.shape[0] + 1))
        keep = slots < size
        # Repeated slots assign in stream order and the last write wins, which is what replacing
        # one pixel at a time would leave behind.
        reservoir[slots[keep]] = values[keep]
        seen += values.shape[0]
    return reservoir, seen


def sampled_band_ranges(src: "RasterSource", *, seed: int,
                        window_size: int, max_windows: int, reservoir_size: int,
                        percentiles: tuple[float, float] = DISPLAY_CLIP_PERCENTILES,
                        ) -> SampledBandRanges:
    """Per-band display bounds from a seeded sample of the open reader ``src``'s pixel windows,
    in one pass.

    A raster far too large to decode whole is described from the windows
    ``raster_source.sample_windows`` picks and from nothing else. One walk of those windows
    produces each band's min/max and each band's ``percentiles`` cut points off a bounded
    reservoir of the pixels walked. The result is a sample's bounds, never the raster's, and
    says so through its :class:`SampledBandRanges` type, seed and covered fraction;
    :func:`band_ranges` gives the exact range of decoded pixels.

    ``seed`` and ``reservoir_size`` are required; the reservoir holds at most ``reservoir_size`` x
    the reader's channel count values whatever the raster's size. Two calls with the same seed
    over the same raster return the same numbers.
    """
    import numpy as np

    from tcip_mcp.pipelines import raster_source

    if reservoir_size <= 0:
        raise ValueError(f"reservoir_size must be positive, got {reservoir_size}")

    windows = raster_source.sample_windows(
        src.width, src.height, seed=seed, window_size=window_size, max_windows=max_windows)
    rng = np.random.default_rng(seed)
    lows = highs = None
    reservoir = None
    seen = 0
    covered = 0
    for rect in windows:
        region = raster_source.hwc_array(src.read_region(rect)[0])
        flat = region.reshape(-1, region.shape[-1])
        band_lo = flat.min(axis=0).astype(np.float64)
        band_hi = flat.max(axis=0).astype(np.float64)
        lows = band_lo if lows is None else np.minimum(lows, band_lo)
        highs = band_hi if highs is None else np.maximum(highs, band_hi)
        reservoir, seen = _reservoir_take(reservoir, seen, flat, reservoir_size, rng)
        covered += rect.width * rect.height
    fraction = covered / float(src.width * src.height)
    # sample_windows always returns at least one window for a raster with positive dimensions,
    # so the loop above ran at least once.
    assert lows is not None and highs is not None and reservoir is not None
    return SampledBandRanges(
        ranges=[BandRange(float(lo), float(hi)) for lo, hi in zip(lows, highs)],
        clip_bounds=[clip_bounds(reservoir[:, i], percentiles)
                     for i in range(reservoir.shape[1])],
        seed=int(seed),
        pixel_fraction=float(fraction),
    )
