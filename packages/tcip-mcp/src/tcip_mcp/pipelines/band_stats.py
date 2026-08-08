"""Display band statistics and the 8-bit stretch a band preview renders through.

Every number here is in the raster's own dtype units: a uint16 band's minimum is a raw digital
number, and a ``percent_clip`` bound is one too. That is a different unit system from
``derivations.band_normalization_stats``, whose per-band mean/std are the [0, 1] tensor units a
detector normalizes with, and the two never stand in for each other: these describe what a viewer
sees on screen, those describe what the model is fed.

Heavy deps (numpy, the raster backends) are imported lazily so this stays cheap to import.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.raster_source import WindowSampling

# The bounds a ``percent_clip`` render stretches between, as percentiles of the band's own values:
# the display clip the band-composite route offers a viewer.
DISPLAY_CLIP_PERCENTILES = (2.0, 98.0)

# The stretches :func:`stretch_band` implements and a caller may ask for; it refuses anything else.
STRETCH_MODES = ("minmax", "percent_clip", "none")


@dataclass(frozen=True)
class BandRange:
    """One band's value range in the raster's own dtype units."""

    minimum: float
    maximum: float


@dataclass(frozen=True)
class SampledBandRanges:
    """Per-band ranges read from a sample of a raster's pixels, with the sample that produced them.

    ``ranges`` bounds the spread of the pixels actually read and never claims to be the raster's
    own min/max: a more extreme value in a window the sample missed is invisible here.
    ``sampling`` names the windows that were read, so a caller can say what the numbers describe.
    """

    ranges: list[BandRange]
    sampling: WindowSampling


def band_ranges(pixels) -> list[BandRange]:
    """Exact per-band min/max of an ``[H, W, C]`` array (or of one 2-D band), in its dtype units."""
    import numpy as np

    arr = np.asarray(pixels)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return [BandRange(float(arr[:, :, i].min()), float(arr[:, :, i].max()))
            for i in range(arr.shape[-1])]


def clip_bounds(band, percentiles: tuple[float, float] = DISPLAY_CLIP_PERCENTILES
                ) -> tuple[float, float]:
    """The values ``percentiles`` cuts a band at, in the band's own dtype units."""
    import numpy as np

    lo, hi = np.percentile(np.asarray(band, dtype=np.float64), list(percentiles))
    return float(lo), float(hi)


def full_scale_denominator(band, source_dtype) -> float:
    """The divisor that puts a band on its own full scale with no data-range stretch applied.

    An integer raster divides by its dtype's maximum, the scale ``image_utils.pil_to_tensor``
    applies for training; a float raster has no such ceiling and divides by its own maximum. A band
    whose maximum is zero divides by 1.0, so an empty float band renders black instead of raising.
    """
    import numpy as np

    if np.issubdtype(source_dtype, np.integer):
        return float(np.iinfo(source_dtype).max) or 1.0
    return float(np.asarray(band).max()) or 1.0


def stretch_band(band, mode: str, source_dtype):
    """One band as ``uint8`` display pixels, stretched by ``mode``.

    ``minmax`` spans the band's own data range, ``percent_clip`` spans
    :data:`DISPLAY_CLIP_PERCENTILES` of it, and ``none`` applies no data-range stretch at all,
    scaling by :func:`full_scale_denominator` so a band's absolute level survives instead of being
    normalized away. A band with no spread (``hi <= lo``) renders black rather than dividing by
    zero. ``source_dtype`` is the raster's own dtype, which only ``none`` reads.
    """
    import numpy as np

    raw = np.asarray(band).astype(np.float64)
    if mode == "none":
        out = raw / full_scale_denominator(raw, source_dtype) * 255.0
    elif mode == "percent_clip":
        lo, hi = clip_bounds(raw)
        out = (raw - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(raw)
    elif mode == "minmax":
        lo, hi = float(raw.min()), float(raw.max())
        out = (raw - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(raw)
    else:
        raise ValueError(f"stretch mode must be one of {sorted(STRETCH_MODES)}, got {mode!r}")
    return np.clip(out, 0, 255).astype(np.uint8)


def sampled_band_ranges(source: "str | Path | BandGroupRef", num_channels: int, *, seed: int,
                        window_size: int, max_windows: int) -> SampledBandRanges:
    """Per-band ranges from a seeded sample of ``source``'s pixel windows, without a full decode.

    Reads through ``raster_source.open_raster``, so a raster far too large to decode whole is
    described from the windows ``raster_source.sample_windows`` picks and from nothing else. The
    result is a sample's range, never the raster's, and says so through its own
    :class:`SampledBandRanges` type and the returned sampling record; a caller that needs the exact
    range calls :func:`band_ranges` on the decoded pixels instead. ``seed`` has no default, so what
    was sampled is always the caller's own stated choice.
    """
    import numpy as np

    from tcip_mcp.pipelines import raster_source

    with raster_source.open_raster(source, num_channels) as src:
        windows = raster_source.sample_windows(
            src.width, src.height, seed=seed, window_size=window_size, max_windows=max_windows)
        label = str(getattr(source, "manifest_path", source))
        lows = highs = None
        covered = 0
        for rect in windows:
            region = np.asarray(src.read_region(rect)[0])
            if region.ndim == 2:
                region = region[:, :, None]
            flat = region.reshape(-1, region.shape[-1])
            band_lo = flat.min(axis=0).astype(np.float64)
            band_hi = flat.max(axis=0).astype(np.float64)
            lows = band_lo if lows is None else np.minimum(lows, band_lo)
            highs = band_hi if highs is None else np.maximum(highs, band_hi)
            covered += rect.width * rect.height
        fraction = covered / float(src.width * src.height)
    sampling = raster_source.WindowSampling(
        tuple((label, rect) for rect in windows), int(seed), float(fraction))
    return SampledBandRanges([BandRange(float(lo), float(hi)) for lo, hi in zip(lows, highs)],
                             sampling)
