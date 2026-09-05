"""Raster reading: one open-and-read surface for every image source this platform decodes.

There is a backend per kind of source (a photographic frame through PIL, a GDAL-readable raster
served windowed through GDAL's block cache, a numpy container, a group of sibling single-band
files, and a whole tifffile decode for the stacked multi-page layouts GDAL misreads), and each
serves pixel regions through the same small surface, so a caller that wants one window of a 90 GB
orthomosaic and a caller that wants a whole 4-band capture compose the same way.

:func:`open_raster` picks the backend from the source itself. The channel count a caller passes is
a routing hint (which PIL mode to decode a photograph in, which axis order a numpy/TIFF array
carries), never an assertion about the file: a raster whose real band count disagrees is still read
exactly as it sits on disk, and checking that is the caller's own job.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from tcip_mcp.pipelines.data.band_groups import BandGroupRef

logger = logging.getLogger(__name__)

# The containers band data is read out of as an array. Any other extension is a photographic frame
# decoded through PIL, at the channel counts PIL's own modes cover.
ARRAY_CONTAINER_EXTS = (".npy", ".npz", ".tif", ".tiff")

_PIL_MODES = {1: "L", 3: "RGB", 4: "RGBA"}

# A plain, documented platform default, not derived from a measurement: the share of the host's
# physical RAM the decoded-pixel caches here may hold, leaving the model, tile batch and OS the rest.
_RAM_BUDGET_FRACTION = 0.25

# GDAL's block cache's share of that budget; the pooled registry of open sources budgets against
# the remainder. An even split: no measurement yet favors either consumer over the other.
_GDAL_CACHE_SHARE = 0.5

# Used only when psutil cannot be imported: a deliberately low assumed host size, so a host whose
# real memory can't be read is under-budgeted rather than over.
_ASSUMED_TOTAL_RAM_BYTES = 8 * 1024 ** 3

# raster_content_identity()'s default sampling budget, a plain, documented default (not
# measured against a real false-match rate).
CONTENT_IDENTITY_SEED = 0
CONTENT_IDENTITY_WINDOW_SIZE = 1024
CONTENT_IDENTITY_MAX_WINDOWS = 8

_total_ram_bytes: int | None = None

# GDAL reads GDAL_CACHEMAX as megabytes below this value and as bytes at or above it, so the
# budget handed to it is floored here to stay unambiguously in bytes.
_GDAL_CACHEMAX_BYTES_FLOOR = 100_000


def configure_gdal_cache(share: float = 1.0) -> None:
    """Hand GDAL's block cache its share of this module's memory budget.

    Called once at each process entry point (the MCP server's ``main``, the web backend's app
    startup, the training subprocess, and each of its spawned DataLoader workers), never at
    source construction: the cache is process-global, and a per-open call would re-decide a
    process-wide fact on every read path. ``share`` scales the budget down for a process that
    is one of several peers each holding their own GDAL cache (a DataLoader worker: pass
    ``1 / num_workers`` so ``num_workers`` peers together still commit the platform's intended
    budget rather than that amount each).
    """
    from rasterio.env import set_gdal_config

    budget = int(_memory_budget_bytes() * _GDAL_CACHE_SHARE * share)
    set_gdal_config("GDAL_CACHEMAX", max(budget, _GDAL_CACHEMAX_BYTES_FLOOR))


def gdal_cache_bytes() -> int:
    """The block-cache budget currently in force, in bytes.

    Read from the configuration :func:`configure_gdal_cache` set, falling back to what that
    function would have set for a process that never called it. This module owns the budget, so a
    consumer sizing work against it asks here rather than asking GDAL, and the two can never
    disagree about a number one of them chose.
    """
    from rasterio.env import get_gdal_config

    configured = get_gdal_config("GDAL_CACHEMAX")
    if configured is None:
        return int(_memory_budget_bytes() * _GDAL_CACHE_SHARE)
    return int(configured)


def open_gdal_dataset(path: str | Path):
    """A read-only GDAL dataset for ``path``, with GDAL's own failure wrapped in this layer's
    error naming the file.

    Served through rasterio, which bundles its own GDAL, so no separate ``osgeo`` binding is
    needed. The returned object is a rasterio dataset, not an ``osgeo.gdal.Dataset``.
    """
    import rasterio

    try:
        return rasterio.open(str(path))
    except Exception as exc:  # noqa: BLE001, rasterio raises driver-specific errors
        raise ValueError(f"GDAL cannot open raster '{path}': {exc}") from exc


@dataclass(frozen=True)
class Rect:
    """A half-open pixel rectangle in a raster's own full-resolution grid.

    Rows ``y0`` up to but excluding ``y1``, columns ``x0`` up to but excluding ``x1``: exactly the
    numpy slice ``[y0:y1, x0:x1]``.
    """

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class WindowSampling:
    """Exactly which pixels a sampled statistic was read from.

    ``windows`` pairs each source label with one rectangle read from it, ``seed`` is the seed that
    chose them, and ``pixel_fraction`` is the share of those sources' pixels the rectangles cover.
    A statistic carrying this describes a sample and not the whole raster, so whatever records the
    statistic records this beside it.
    """

    windows: tuple[tuple[str, Rect], ...]
    seed: int
    pixel_fraction: float

    @property
    def label(self) -> str:
        """One line naming this as a sampled statistic and what it was sampled from."""
        return (f"sampled from {len(self.windows)} pixel window(s), seed {self.seed}, covering "
                f"{self.pixel_fraction:.4f} of the source pixels")


def sample_windows(width: int, height: int, *, seed: int, window_size: int,
                   max_windows: int) -> list[Rect]:
    """A deterministic, seeded selection of pixel windows over a ``width`` x ``height`` raster.

    The raster is divided into a grid of ``window_size`` squares (edge cells are short, never
    padded and never overlapped) and ``max_windows`` of those cells are drawn without replacement
    from ``seed``, returned in row-major order. Grid cells rather than free-floating rectangles, so
    no pixel is ever counted twice and a ``max_windows`` at or above the grid's own cell count
    returns every pixel exactly once: a statistic over that sample is the statistic over a full
    decode, which is what lets a sampled reader be checked against an exact one.
    """
    import numpy as np

    if width <= 0 or height <= 0:
        raise ValueError(f"cannot sample windows from a {height}x{width} raster")
    if window_size <= 0 or max_windows <= 0:
        raise ValueError("window_size and max_windows must both be positive, got "
                         f"window_size={window_size}, max_windows={max_windows}")
    cells = [Rect(x, y, min(x + window_size, width), min(y + window_size, height))
             for y in range(0, height, window_size)
             for x in range(0, width, window_size)]
    if max_windows >= len(cells):
        return cells
    chosen = np.random.default_rng(seed).choice(len(cells), size=max_windows, replace=False)
    return [cells[i] for i in sorted(int(c) for c in chosen)]


@dataclass(frozen=True)
class ReadSpec:
    """How a read was served: which backend decoded it, the served resolution as a fraction of the
    raster's native resolution, and the resampling that produced it.

    A plain read serves full resolution: ``scale`` 1.0, ``resample`` ``None``. A read with a
    ``target_size`` records the requested output/region ratio and the resampling algorithm that
    was requested; which overview level GDAL satisfied a reduced read from is not observable
    through RasterIO and is not claimed here.
    """

    backend: str
    scale: float = 1.0
    resample: str | None = None


class RasterSource(Protocol):
    """The read surface every backend in this module exposes.

    :meth:`read_region` returns ``([H, W, C] pixels, ReadSpec)`` for a rectangle lying inside the
    raster; an empty or out-of-bounds rectangle raises ``ValueError`` rather than returning a
    silently clipped array, so a caller that wants an edge tile clips to the raster's own bounds
    and pads the result itself (``image_utils.pad_tile``). The pixels are always a copy: mutating
    them can never corrupt what a later read returns.

    ``target_size`` (output ``(width, height)``) serves the same rectangle resampled to that size;
    it must preserve the rectangle's aspect ratio to within the rounding of fitting either edge
    (``ValueError`` otherwise), and the returned :class:`ReadSpec` records the requested scale and
    resampling. :meth:`read_window` is the same read in the row-first argument order the tiled
    inference loop uses.
    """

    width: int
    height: int
    num_channels: int
    dtype: np.dtype

    @property
    def resident_bytes(self) -> int: ...

    def read_region(self, rect: Rect, *,
                    target_size: tuple[int, int] | None = None) -> tuple[np.ndarray, ReadSpec]: ...

    def read_window(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray: ...

    def close(self) -> None: ...

    def __enter__(self) -> "RasterSource": ...

    def __exit__(self, *exc_info: object) -> None: ...


class _ClosableSource:
    """Close-once and context-manager plumbing shared by the backends; each releases whatever it
    holds open in ``_release``."""

    closed = False

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._release()

    def _release(self) -> None:
        """Drop whatever this backend holds open. Called once, by :meth:`close`."""

    def read_region(self, rect: Rect, *,
                    target_size: tuple[int, int] | None = None) -> tuple[np.ndarray, "ReadSpec"]:
        """Return ``([H, W, C] pixels, ReadSpec)`` for ``rect``. Every subclass provides its own;
        declared here only so :meth:`read_window` can call it through ``self``."""
        raise NotImplementedError

    def read_window(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """The pixel window ``[y0:y1, x0:x1]``, in the row-first argument order the tiled
        inference loop uses.

        ``read_window`` orders rows first while :class:`Rect` orders x first; the flip between the
        two conventions happens exactly here and nowhere else.
        """
        region, _spec = self.read_region(Rect(x0, y0, x1, y1))
        return region

    def __enter__(self):
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _check_region(rect: Rect, height: int, width: int) -> None:
    """Raise ``ValueError`` unless ``rect`` is a non-empty region inside a ``height`` x ``width``
    raster."""
    if not (0 <= rect.y0 < rect.y1 <= height) or not (0 <= rect.x0 < rect.x1 <= width):
        raise ValueError(
            f"region [{rect.y0}:{rect.y1}, {rect.x0}:{rect.x1}] is out of bounds for a "
            f"{height}x{width} raster"
        )


class _RegionView:
    """A read-only offset view over an already-open :class:`RasterSource`, restricted to one
    sub-rectangle of its full extent.

    Exposes exactly the minimal duck-typed surface a windowed tile source needs
    (``read_window``/``height``/``width``/``num_channels``, the
    ``inference.generic_predictor.WindowedRasterReader`` Protocol): a rectangular sub-region of a
    mosaic then reads as an ordinary windowed raster source in its own local coordinate space, so
    tiled inference over a haloed calibration/holdout block runs through the exact same
    ``predict_tiled`` code path a whole-mosaic export does, never a second implementation of
    tiled inference for one region.

    Dims invariant, load-bearing for measurement integrity, not just an implementation detail:
    ``height``/``width`` always report this view's own rect extent, never the parent source's,
    and every read is translated into the parent's coordinate space by adding the rect's own
    origin. A read past this view's declared bounds raises rather than falling through to the
    parent's own out-of-bounds check, which validates against the *whole* mosaic and would
    otherwise happily serve real training/buffer pixels through the offset -- exactly what
    ``predict_tiled``'s own edge clip (``min(tile_y + edge, source.height)``, checked against
    this class's own reported ``height``) relies on to keep a windowed pass over one region from
    ever silently reading pixels outside it. This must hold for every future caller of this
    class, not only ``predict_tiled``.

    ``band_interpretations`` forwards the parent's own attribute verbatim when it has one (a
    ``GdalSource`` parent), absent otherwise, the same ``getattr(src, "band_interpretations",
    None)`` convention every other consumer of this fact uses: a haloed calibration/holdout block
    read through this view is still the same file the whole-mosaic export path reads, so the two
    must resolve the same alpha-vs-spectral-band decision (:func:`image_utils.to_pil_if_faithful`)
    rather than one seeing the real signal and the other silently seeing none.
    """

    def __init__(self, parent: RasterSource, rect: Rect) -> None:
        _check_region(rect, parent.height, parent.width)
        self._parent = parent
        self._rect = rect
        self.height = rect.height
        self.width = rect.width
        self.num_channels = parent.num_channels
        self.band_interpretations = getattr(parent, "band_interpretations", None)

    def read_window(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """The pixel window ``[y0:y1, x0:x1]`` in this view's own local coordinate space,
        translated into the parent source's coordinates by this view's own rect origin.

        Checked against this view's own declared ``height``/``width``, never the parent's: the
        dims invariant this class exists to hold.
        """
        _check_region(Rect(x0, y0, x1, y1), self.height, self.width)
        oy, ox = self._rect.y0, self._rect.x0
        return self._parent.read_window(oy + y0, oy + y1, ox + x0, ox + x1)


def _check_target_size(rect: Rect, target_size: tuple[int, int]) -> tuple[int, int]:
    """Validate a ``(width, height)`` output size against ``rect`` and return it as ints.

    The target must preserve the region's aspect ratio to within the rounding of fitting either
    edge; a distorting target raises ``ValueError``, since a resample here must never silently
    change a raster's geometry (a caller that wants a distortion resizes the returned pixels
    itself).
    """
    out_w, out_h = int(target_size[0]), int(target_size[1])
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"target_size must be positive, got {out_w}x{out_h}")
    fit_h = max(1, round(rect.height * out_w / rect.width))
    fit_w = max(1, round(rect.width * out_h / rect.height))
    if out_h != fit_h and out_w != fit_w:
        raise ValueError(
            f"target_size {out_w}x{out_h} does not preserve the aspect ratio of the "
            f"{rect.width}x{rect.height} region it resamples"
        )
    return out_w, out_h


def _area_downsample(region: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """``region`` resampled to ``out_h`` x ``out_w`` rows x cols by pixel-area averaging: cv2's
    ``INTER_AREA``, the one area resampler used for every backend that resamples in memory."""
    import cv2

    out = cv2.resize(region, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return out if out.ndim == 3 else out[:, :, None]


def _serve_region(region: np.ndarray, rect: Rect, backend: str,
                  target_size: tuple[int, int] | None) -> tuple[np.ndarray, ReadSpec]:
    """The read tail every in-memory backend shares: the native copy as-is, or area-downsampled to
    an aspect-preserving ``target_size`` with the :class:`ReadSpec` recording the requested
    scale."""
    if target_size is None:
        return region, ReadSpec(backend)
    out_w, out_h = _check_target_size(rect, target_size)
    return (_area_downsample(region, out_w, out_h),
            ReadSpec(backend, scale=out_w / rect.width, resample="area"))


def channel_first_reinterpreted(shape: tuple[int, ...], num_channels: int) -> bool:
    """Whether a channel-last reading of a 3-D ``shape`` is instead taken as channel-first: the
    leading axis matches the caller's expected band count while the trailing one does not, the one
    shape where channel-first is unambiguous against what the caller asked for.

    The single predicate behind ``_channel_last``'s transpose, :func:`tiff_frame`'s header
    measurement, and the TIFF dispatch's whole-decode routing, so a raster can never be measured
    through one reading and decoded through another.
    """
    return len(shape) == 3 and shape[0] == num_channels and shape[2] != num_channels


def _channel_last(arr: np.ndarray, num_channels: int) -> np.ndarray:
    """A decoded array as ``[H, W, C]``, using the caller's expected band count to tell a
    channel-first raster from a channel-last one.

    A 2-D array gains a trailing axis of 1. A 3-D array is transposed only when
    :func:`channel_first_reinterpreted` says its shape reads channel-first against the expected
    count; every other shape is returned as decoded, so a count that disagrees with the file never
    reshapes its pixels.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim == 3 and channel_first_reinterpreted(arr.shape, num_channels):
        return np.transpose(arr, (1, 2, 0))
    return arr


def _tiff_series_probe(path: str | Path) -> tuple[tuple[int, ...], str] | None:
    """Header-only TIFF series shape and axes string; ``None`` if the header can't be read.

    ``tif.series[0]``, not ``pages[0]``: a channel-last TIFF stores each row-block as its own
    page, so ``pages[0]`` of a 24x40x5 raster is ``(40, 5)``.
    """
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tif:
            series = tif.series[0]
            return tuple(int(x) for x in series.shape), str(series.axes)
    except Exception:  # noqa: BLE001, fall through to a full read rather than guess
        return None


def tiff_series_shape(path: str | Path) -> tuple[int, ...] | None:
    """Header-only TIFF series shape (no pixel decode); ``None`` if it can't be read this way.

    Shared with ``derivations.probe_channels`` so a full pixel read is never paid just to learn
    the shape.
    """
    probe = _tiff_series_probe(path)
    return probe[0] if probe is not None else None


def _series_frame(shape: tuple[int, ...], axes: str) -> tuple[int, int, int] | None:
    """A TIFF series' frame as ``(height, width, channels)``; ``None`` for an axes layout with no
    single-frame reading (the one-page-per-band or one-page-per-row-block stacks tifffile writes,
    which only a whole tifffile decode reads correctly)."""
    if len(axes) != len(shape):
        return None
    if axes == "YX":
        return (shape[0], shape[1], 1)
    if axes == "YXS":
        return (shape[0], shape[1], shape[2])
    if axes == "SYX":  # planar (band-separate) samples report channel-first
        return (shape[1], shape[2], shape[0])
    return None


def tiff_frame(path: str | Path, num_channels: int) -> tuple[int, int, int] | None:
    """The ``(height, width, channels)`` frame this module's TIFF dispatch will serve ``path`` in
    at ``num_channels``, from the header alone; ``None`` when the header can't be read and only a
    decode can answer.

    The same axes normalization and channel-first reading the dispatch itself applies
    (:func:`_series_frame`, :func:`channel_first_reinterpreted`), so a frame measured here and
    pixels decoded later come from one set of rules: an axes-normalizable series serves in its own
    frame unless the channel-first reinterpretation sends it to the whole decode (which
    transposes); a stacked multi-page series serves whole in tifffile's raw reading.
    """
    probe = _tiff_series_probe(path)
    if probe is None:
        return None
    shape, axes = probe
    frame = _series_frame(shape, axes)
    if frame is None:
        # The whole-decode route: tifffile.imread returns the raw series shape and _channel_last
        # applies the same reinterpretation to it.
        if len(shape) == 2:
            return (shape[0], shape[1], 1)
        if len(shape) != 3:
            return None
        if channel_first_reinterpreted(shape, num_channels):
            return (shape[1], shape[2], shape[0])
        return (shape[0], shape[1], shape[2])
    # A planar (SYX) series is channel-first by its own header: no reinterpretation applies.
    if axes != "SYX" and channel_first_reinterpreted(frame, num_channels):
        return (frame[1], frame[2], frame[0])
    return frame


class _ArraySource(_ClosableSource):
    """A backend whose pixels are one already-decoded ``[H, W, C]`` array held in memory.

    A subclass decodes that array in its own constructor and hands it to :meth:`_describe`; regions
    are copied out of it.
    """

    _backend = "array"

    def _describe(self, array: np.ndarray) -> None:
        self._array: np.ndarray | None = array
        self.height = int(array.shape[0])
        self.width = int(array.shape[1])
        self.num_channels = int(array.shape[2]) if array.ndim > 2 else 1
        self.dtype = array.dtype

    @property
    def resident_bytes(self) -> int:
        assert self._array is not None, "resident_bytes read after close(): a closed source holds no array"
        return int(self._array.nbytes)

    def read_region(self, rect: Rect, *,
                    target_size: tuple[int, int] | None = None) -> tuple[np.ndarray, ReadSpec]:
        assert self._array is not None, "read_region called after close(): a closed source holds no array"
        _check_region(rect, self.height, self.width)
        region = np.array(self._array[rect.y0:rect.y1, rect.x0:rect.x1])
        return _serve_region(region, rect, self._backend, target_size)

    def _release(self) -> None:
        self._array = None


class PhotographicSource(_ClosableSource):
    """A whole photographic frame decoded through PIL, EXIF-oriented and converted to the mode the
    caller's channel count names (1 -> L, 3 -> RGB, 4 -> RGBA).

    ``image`` is that PIL frame itself: the augmentation and tiling code downstream works on PIL
    images at these counts, so this backend exposes the native object rather than forcing every
    caller through an array copy of it. The frame is fully decoded and independent of the file,
    which is closed as soon as it has been read.

    The frame is EXIF-oriented before the mode conversion so it matches what
    ``get_image_dimensions`` measures: labels are authored in the upright frame, and an
    Orientation-6 JPEG's raw sensor frame has its axes swapped against it.
    """

    def __init__(self, path: str | Path, num_channels: int):
        from PIL import Image

        from tcip_annotation.utils import auto_orient_image

        self.path = Path(path)
        opened = Image.open(self.path)
        self.image = auto_orient_image(opened).convert(_PIL_MODES[num_channels])
        opened.close()
        self.width, self.height = self.image.size
        self.num_channels = len(self.image.getbands())
        self.dtype = np.dtype("uint8")  # L/RGB/RGBA are all 8-bit
        self._frame: np.ndarray | None = None

    @property
    def resident_bytes(self) -> int:
        """The peak this source can hold: the PIL frame plus the ndarray copy
        :meth:`read_region` materializes from it on first use.

        The pool records this value once, at insert, so it must never understate what the
        source may later hold; counting both frames up front keeps the accounting honest
        even when :meth:`read_region` is never called, where a value that grew after a
        first read would silently exceed what the pool recorded.
        """
        return int(2 * self.width * self.height * self.num_channels * self.dtype.itemsize)

    def read_region(self, rect: Rect, *,
                    target_size: tuple[int, int] | None = None) -> tuple[np.ndarray, ReadSpec]:
        _check_region(rect, self.height, self.width)
        if self._frame is None:
            frame = np.asarray(self.image)
            self._frame = frame[:, :, None] if frame.ndim == 2 else frame
        region = np.array(self._frame[rect.y0:rect.y1, rect.x0:rect.x1])
        return _serve_region(region, rect, "photographic", target_size)

    def _release(self) -> None:
        self._frame = None


class TiffWholeSource(_ArraySource):
    """A TIFF decoded whole by ``tifffile.imread``, only for the layouts GDAL's first-IFD data
    model misreads (the stacked multi-page files tifffile itself writes: a channel-last raster one
    row-block per page, a plain ``[C, H, W]`` stack one band per page) and for the shapes a whole
    decode reinterprets channel-first; every single-dataset layout is :class:`GdalSource`'s."""

    _backend = "tiff_whole"

    def __init__(self, path: str | Path, num_channels: int):
        import tifffile

        self.path = Path(path)
        self._describe(_channel_last(tifffile.imread(str(self.path)), num_channels))


class NpySource(_ArraySource):
    """A ``.npy`` array, memory-mapped so a region read touches only the pages it covers."""

    _backend = "npy"

    def __init__(self, path: str | Path, num_channels: int):
        self.path = Path(path)
        self._mapped = np.load(str(self.path), mmap_mode="r")
        self._describe(_channel_last(self._mapped, num_channels))

    def _release(self) -> None:
        # Dropping both references releases the mapping; closing it under the views numpy exports
        # from it would raise instead.
        self._array = None
        self._mapped = None


class NpzSource(_ArraySource):
    """A ``.npz`` container's first stored array."""

    _backend = "npz"

    def __init__(self, path: str | Path, num_channels: int):
        self.path = Path(path)
        with np.load(str(self.path)) as npz:
            arr = npz[npz.files[0]]
        self._describe(_channel_last(arr, num_channels))


class BandGroupSource(_ClosableSource):
    """Sibling single-band files read as one logical multi-band raster.

    Each member is opened on its own through :func:`open_array_source` and decodes exactly as it
    would alone; a region is every member's own region concatenated on the channel axis, in the
    manifest's declared band order, and a ``target_size`` read is each member's own resampled read
    (the returned spec carries the members' scale and resampling). The group's frame is its first
    band's, the same frame ``image_utils.image_dimensions`` reports for a group; a member covering
    a different extent is refused at open rather than cropped or resampled to fit, since the
    pixels of a group whose bands disagree on the frame cannot be stacked into one raster.
    """

    def __init__(self, ref: BandGroupRef, num_channels: int):
        self.ref = ref
        self._members = [open_array_source(p, 1) for p in ref.bands.values()]
        first = self._members[0]
        for name, member in zip(ref.bands, self._members):
            if (member.width, member.height) != (first.width, first.height):
                self._release()
                raise ValueError(
                    f"band group {ref.stem!r} ({ref.manifest_path}): band {name!r} is "
                    f"{member.width}x{member.height} but the group's frame is "
                    f"{first.width}x{first.height}; bands that disagree on the frame cannot "
                    "stack into one raster."
                )
        self.width = first.width
        self.height = first.height
        self.num_channels = sum(m.num_channels for m in self._members)
        self.dtype = np.result_type(*[m.dtype for m in self._members])

    @property
    def resident_bytes(self) -> int:
        return sum(int(m.resident_bytes) for m in self._members)

    def read_region(self, rect: Rect, *,
                    target_size: tuple[int, int] | None = None) -> tuple[np.ndarray, ReadSpec]:
        reads = [m.read_region(rect, target_size=target_size) for m in self._members]
        member_spec = reads[0][1]
        return (np.concatenate([pixels for pixels, _spec in reads], axis=-1),
                ReadSpec("band_group", scale=member_spec.scale, resample=member_spec.resample))

    def _release(self) -> None:
        for member in self._members:
            member.close()


class GdalSource(_ClosableSource):
    """A GDAL-readable raster (today: .tif/.tiff) opened read-only and served windowed.

    Regions decode through GDAL's own block cache (budgeted per process by
    :func:`configure_gdal_cache`), so repeated windows over a raster far too large to decode whole
    cost only the blocks they touch. A ``target_size`` read asks RasterIO for the reduced buffer
    directly (``resample_alg=Average``), which GDAL serves from the nearest overview level at or
    above the requested resolution when the raster carries one (see ``pipelines.overviews``).

    A single-band palette-color raster (``GCI_PaletteIndex``) is expanded through its own color
    table to uint8 RGB, the same pixels PIL's palette decode produced, and reports three
    channels; a ``target_size`` read of one expands at native resolution first and
    area-downsamples the RGB, since palette indices must never be resampled.

    ``band_interpretations`` names each served channel's GDAL color interpretation (lowercase,
    e.g. ``("red", "green", "blue", "alpha")``; ``"undefined"`` when the file declares none), so
    a consumer can tell an alpha band from a spectral one. Only this backend carries the
    attribute; read it with ``getattr(src, "band_interpretations", None)``.

    A GDAL dataset handle is not thread-safe: one instance must never be shared across concurrent
    threads.
    """

    _backend = "gdal"

    def __init__(self, path: str | Path):
        from rasterio.enums import ColorInterp

        self.path = Path(path)
        self._ds = open_gdal_dataset(self.path)
        self.width = int(self._ds.width)
        self.height = int(self._ds.height)
        self.num_channels = int(self._ds.count)
        self.dtype = np.dtype(self._ds.dtypes[0])
        self._palette_lut: np.ndarray | None = None
        if (self.num_channels == 1 and self.dtype == np.dtype("uint8")
                and self._ds.colorinterp[0] == ColorInterp.palette):
            table = self._ds.colormap(1)
            if table:
                lut = self._tiff_colormap_lut()
                if lut is None:
                    # A container with no readable ColorMap tag: GDAL's converted entries.
                    lut = np.zeros((256, 3), dtype=np.uint8)
                    for index, entry in table.items():
                        if 0 <= int(index) < 256:
                            lut[int(index)] = tuple(entry)[:3]
                self._palette_lut = lut
                self.num_channels = 3
                self.dtype = np.dtype("uint8")
        if self._palette_lut is not None:
            self.band_interpretations = ("red", "green", "blue")
        else:
            self.band_interpretations = tuple(
                interp.name.lower() for interp in self._ds.colorinterp)

    def _tiff_colormap_lut(self) -> "np.ndarray | None":
        """The palette as the file's own ColorMap tag states it, high byte per 16-bit entry.

        The tag's 16-bit entries carry the 8-bit palette scaled by 256 (PIL) or 257 (the TIFF
        specification's full-range convention), and the high byte recovers the original value
        exactly under either scaling. GDAL's converted color-table entries are not used because
        the conversion changed across GDAL versions (3.8 truncate-divides by 257, off by one
        against PIL's decode; 3.12 agrees with the high byte), measured on both.
        ``None`` when the tag cannot be read; the caller falls back to GDAL's entries.
        """
        try:
            import tifffile

            with tifffile.TiffFile(str(self.path)) as tif:
                # tifffile types page 0 as TiffPage | TiffFrame; only TiffPage carries parsed
                # tags, and a page with none reads exactly as a page with no ColorMap tag.
                tags = getattr(tif.pages[0], "tags", None)
                cmap = tags.get("ColorMap") if tags is not None else None
                if cmap is None:
                    return None
                raw = np.asarray(cmap.value)
        except Exception:  # noqa: BLE001, any unreadable tag falls back to GDAL's table
            return None
        if raw.ndim != 2 or raw.shape[0] < 3:
            return None
        lut = np.zeros((256, 3), dtype=np.uint8)
        n = min(256, raw.shape[1])
        lut[:n] = (raw[:3, :n].astype(np.uint16) >> 8).astype(np.uint8).T
        return lut

    @property
    def resident_bytes(self) -> int:
        # Handle-only: decoded blocks live in GDAL's own block cache (configure_gdal_cache's
        # share of the budget), not in this process's pooled accounting.
        return 0

    def read_region(self, rect: Rect, *,
                    target_size: tuple[int, int] | None = None) -> tuple[np.ndarray, ReadSpec]:
        from rasterio.enums import Resampling
        from rasterio.windows import Window

        _check_region(rect, self.height, self.width)
        window = Window(rect.x0, rect.y0, rect.width, rect.height)
        if self._palette_lut is not None:
            indices = self._ds.read(1, window=window)
            return _serve_region(self._palette_lut[indices], rect, self._backend, target_size)
        kwargs = {}
        scale, resample = 1.0, None
        if target_size is not None:
            out_w, out_h = _check_target_size(rect, target_size)
            kwargs = {"out_shape": (self.num_channels, out_h, out_w),
                      "resampling": Resampling.average}
            scale, resample = out_w / rect.width, "average"
        arr = self._ds.read(window=window, **kwargs)
        # GDAL returns [C, H, W]; contiguous copy in the platform's [H, W, C] order.
        arr = np.ascontiguousarray(np.transpose(arr, (1, 2, 0)))
        return arr, ReadSpec(self._backend, scale=scale, resample=resample)

    def _release(self) -> None:
        if self._ds is not None:
            self._ds.close()
        self._ds = None


def _memory_budget_bytes() -> int:
    """The decoded-pixel bytes this module lets itself hold: a fraction of the host's total
    physical RAM, read once per process."""
    global _total_ram_bytes
    if _total_ram_bytes is None:
        try:
            import psutil

            _total_ram_bytes = int(psutil.virtual_memory().total)
        except ImportError:
            _total_ram_bytes = _ASSUMED_TOTAL_RAM_BYTES
            logger.info(
                "psutil unavailable; budgeting raster caches against an assumed %.1f GiB host",
                _ASSUMED_TOTAL_RAM_BYTES / 1024 ** 3,
            )
    return int(_total_ram_bytes * _RAM_BUDGET_FRACTION)


def _pool_budget_bytes() -> int:
    """The pooled registry's budget: what the memory budget leaves after GDAL's block-cache
    share."""
    return int(_memory_budget_bytes() * (1.0 - _GDAL_CACHE_SHARE))


def photographic_container(source: "str | Path | BandGroupRef", num_channels: int) -> bool:
    """Whether ``source`` decodes as a whole photographic frame through PIL rather than as band
    data: any extension outside :data:`ARRAY_CONTAINER_EXTS`, at one of the channel counts PIL's
    own modes cover.

    The one routing decision this module's factory and ``image_utils``' dimension probe share, so a
    file can never be measured through one path and decoded through another.
    """
    if isinstance(source, BandGroupRef):
        return False
    return (num_channels in _PIL_MODES
            and Path(source).suffix.lower() not in ARRAY_CONTAINER_EXTS)


def image_route_channel_count(
    source: "str | Path | BandGroupRef", probed: int | None = None,
) -> int:
    """The channel count a plain (non-composited) image-route read opens ``source`` at:
    :func:`~tcip_mcp.pipelines.derivations.probe_channels`, or three in place of a photographic
    container's own band count, since a plain serve decodes a grayscale or palette frame through
    PIL's RGB expansion rather than its raw band count.

    ``probed`` lets a caller that already has :func:`probe_channels`'s answer pass it through
    instead of probing the header twice. The one place this rule is spelled, so the display route
    and a content identity computed at its default channel count can never drift apart.
    """
    from tcip_mcp.pipelines.derivations import probe_channels

    if probed is None:
        probed = probe_channels(source)
    return 3 if photographic_container(source, probed) else probed


def open_array_source(source: "str | Path | BandGroupRef", num_channels: int) -> RasterSource:
    """Open ``source`` as a plain ``[H, W, C]`` array raster: a band group, a numpy container, or a
    TIFF.

    A photographic container (any other extension) is refused at every channel count, since a
    caller here wants band data and a PIL frame is :func:`open_raster`'s business.
    """
    if isinstance(source, BandGroupRef):
        return BandGroupSource(source, num_channels)
    path = Path(source)
    ext = path.suffix.lower()
    if ext == ".npy":
        return NpySource(path, num_channels)
    if ext == ".npz":
        return NpzSource(path, num_channels)
    if ext in (".tif", ".tiff"):
        return _open_tiff(path, num_channels)
    raise ValueError(
        f"Cannot load a {num_channels}-channel image from '{ext}'. "
        "Use .npy/.npz or a multi-band GeoTIFF (.tif/.tiff)."
    )


def _tiff_needs_whole_decode(source: GdalSource, num_channels: int) -> bool:
    """Whether a GDAL-opened TIFF must instead decode whole through tifffile.

    GDAL reads a TIFF's first IFD as the dataset, which misreads the stacked multi-page layouts
    tifffile itself writes (a channel-last raster one row-block per page, a ``[C, H, W]`` stack
    one band per page), so GDAL's frame is checked against the file's own axes-normalized series
    shape (:func:`_series_frame`) and any disagreement, including an axes layout with no
    single-frame reading, decodes whole through tifffile. An unreadable series header routes to
    GDAL alone: tifffile could not decode such a file either. A raster whose shape a whole decode
    would reinterpret channel-first (:func:`channel_first_reinterpreted`) also decodes whole, so
    ``load_multiband`` and ``image_dimensions`` keep reporting the same frame for it.

    Header reads only, never a pixel decode: :func:`opens_windowed` answers from this too.
    """
    # The raw band count, not the served one: a palette raster serves as three expanded channels
    # while the file's own header (what tifffile reports) still says one band.
    gdal_frame = (source.height, source.width, int(source._ds.count))
    probe = _tiff_series_probe(source.path)
    if probe is None:
        return False
    shape, axes = probe
    series_frame = _series_frame(shape, axes)
    if series_frame is None or series_frame != gdal_frame:
        return True
    # A planar (SYX) series is channel-first by its own header: no reinterpretation applies.
    return axes != "SYX" and channel_first_reinterpreted(gdal_frame, num_channels)


def _open_tiff(path: Path, num_channels: int) -> RasterSource:
    """GDAL first; tifffile's own header read cross-checks that GDAL sees the whole raster
    (:func:`_tiff_needs_whole_decode` is the one place that check lives)."""
    try:
        source = GdalSource(path)
    except ValueError:
        # GDAL cannot open it; tifffile's whole decode is the only reader left with a claim.
        if _tiff_series_probe(path) is not None:
            return TiffWholeSource(path, num_channels)
        raise
    if _tiff_needs_whole_decode(source, num_channels):
        source.close()
        return TiffWholeSource(path, num_channels)
    return source


def opens_windowed(source: "str | Path | BandGroupRef", num_channels: int) -> bool:
    """Whether :func:`open_raster` would serve ``source`` through a backend that opens without
    decoding pixels and reads windows on demand (a GDAL-served raster, a memory-mapped ``.npy``).

    Every other backend decodes the whole raster at open (a photographic frame, an ``.npz``, a
    stacked TIFF, a band group), so a caller deciding whether an eager open is affordable asks
    here first. For a TIFF the answer needs GDAL's own header read; a probe source is opened and
    closed again, header-only, never decoding pixels. A file no backend can open answers
    ``False``: the open that would name the failure is the caller's to attempt.
    """
    if isinstance(source, BandGroupRef):
        return False
    path = Path(source)
    if photographic_container(path, num_channels):
        return False
    ext = path.suffix.lower()
    if ext == ".npy":
        return True
    if ext not in (".tif", ".tiff"):
        return False
    try:
        probe = GdalSource(path)
    except ValueError:
        return False
    try:
        return not _tiff_needs_whole_decode(probe, num_channels)
    finally:
        probe.close()


def open_raster(source: "str | Path | BandGroupRef", num_channels: int) -> RasterSource:
    """The backend that serves ``source`` at ``num_channels``.

    ``num_channels`` routes (which PIL mode a photograph decodes in, which axis order a numpy/TIFF
    array carries) and is never checked against the file: a 5-band GeoTIFF opened at 3 still reads
    as 5 bands. A photographic extension at a count PIL has no mode for raises ``ValueError``
    naming the containers that do carry band data.
    """
    if photographic_container(source, num_channels):
        assert not isinstance(source, BandGroupRef), (
            "photographic_container already returns False for a BandGroupRef")
        return PhotographicSource(source, num_channels)
    return open_array_source(source, num_channels)


# ── Process-local pool of open sources ───────────────────────────────────

_POOL: "OrderedDict[tuple, RasterSource]" = OrderedDict()
_POOL_PID: int | None = None
_POOL_BYTES = 0


def _stat_identity(path: Path) -> tuple[int, int]:
    st = path.stat()
    return int(st.st_mtime_ns), int(st.st_size)


def source_pool_key(source: "str | Path | BandGroupRef", num_channels: int) -> tuple:
    """The identity :func:`pooled_source` pools an open source under: the file's path, modification
    time and size, and the channel count it was opened at, so an edited file never serves stale
    pixels and a source opened at one count is never handed to a caller asking for another.

    A band group is keyed on its manifest plus every member's own name, modification time and size:
    the manifest can sit untouched while a member is rewritten.
    """
    if isinstance(source, BandGroupRef):
        members = tuple((name, *_stat_identity(p)) for name, p in source.bands.items())
        return (str(source.manifest_path), members, num_channels)
    path = Path(source)
    return (str(path), *_stat_identity(path), num_channels)


def pooled_source(source: "str | Path | BandGroupRef", num_channels: int) -> RasterSource:
    """An open source for ``source`` from this process's pool, opening one if it holds none.

    The pool keeps recently used sources open so a caller that revisits the same raster does not
    reopen and reparse it, and evicts least-recently-used sources (closing them) once what it holds
    exceeds this module's pool budget. It belongs to the process that filled it: a forked worker
    (a DataLoader's, say) finds it empty rather than reusing handles the parent owns.

    A caller must not close what this returns; the pool owns it. The ``image_utils`` loaders open
    and close their own sources; the tiled training dataset reads its windowed sources through
    here so every tile of one raster shares one open handle.

    A GDAL dataset handle is not thread-safe, so this pool must never vend one
    :class:`GdalSource` to two concurrent threads; a threaded consumer needs thread-keyed pooling
    or a per-source lock before it can read through here.
    """
    global _POOL_PID, _POOL_BYTES
    pid = os.getpid()
    if _POOL_PID != pid:
        # A forked child inherits the parent's open handles; drop them without closing, since the
        # parent still owns the originals.
        _POOL.clear()
        _POOL_BYTES = 0
        _POOL_PID = pid
    key = source_pool_key(source, num_channels)
    existing = _POOL.get(key)
    if existing is not None:
        _POOL.move_to_end(key)
        return existing
    opened = open_raster(source, num_channels)
    _POOL[key] = opened
    _POOL_BYTES += int(opened.resident_bytes)
    budget = _pool_budget_bytes()
    while len(_POOL) > 1 and _POOL_BYTES > budget:
        _evicted_key, evicted = _POOL.popitem(last=False)
        _POOL_BYTES -= int(evicted.resident_bytes)
        evicted.close()
    return opened


def close_source_pool() -> None:
    """Close and drop every source this process's pool holds."""
    global _POOL_BYTES
    while _POOL:
        _key, source = _POOL.popitem()
        source.close()
    _POOL_BYTES = 0


# ── Raster content identity ──────────────────────────────────────────────


@dataclass(frozen=True)
class RasterIdentity:
    """One raster file's own content identity: header facts plus a deterministic, seeded pixel
    checksum.

    A sibling primitive to ``resolution.dataset_hash``/``dataset_fingerprint.dataset_fingerprint``,
    never an extension of either: those identify a whole *dataset's* ground truth (or ground truth +
    pixels + registry + confirmations) across many images, a coarser granularity that answers a
    different question ("is this the same dataset") than the one this class answers ("is this the
    same raster file"). ``pixel_checksum`` is the discriminating term (two different rasters of
    identical dimensions checksum differently); ``width``/``height``/``num_channels``/``dtype``
    travel alongside it for a human-readable identity, never in place of the checksum.

    ``band_interpretations`` is present only when the backend that served this raster carries it
    (GDAL-only today) and is ``None`` otherwise; absence is never a refusal condition, only a
    narrower identity. ``geotransform`` is an optional strengthening term, present only when the
    raster is a georeferenced GeoTIFF whose affine tags this module can read, and is never
    load-bearing: an unprojected or non-GeoTIFF raster still gets a fully usable identity from the
    checksum alone.

    ``seed``/``window_size``/``max_windows``/``pixel_fraction`` are the sampling parameters that
    produced ``pixel_checksum``, recorded by name (not just implied by the windows actually read)
    so a later comparison can recompute the other side's identity under the exact same parameters
    rather than each side's own independent default -- otherwise a genuinely identical raster can
    checksum differently purely from parameter drift between the two calls, never from a real
    content difference.
    """

    width: int
    height: int
    num_channels: int
    dtype: str
    pixel_checksum: str
    seed: int
    window_size: int
    max_windows: int
    pixel_fraction: float
    band_interpretations: tuple[str, ...] | None
    geotransform: dict | None


def _optional_geotransform(source: "str | Path | BandGroupRef") -> dict | None:
    """``source``'s own affine georeferencing tags as a plain dict, or ``None`` when it is not a
    path (a :class:`BandGroupRef` has no single file to read tags from) or carries no
    readable/projected geotransform. Never raises: this term strengthens a raster content
    identity when present and is silently absent otherwise, never load-bearing for the identity
    as a whole."""
    if isinstance(source, BandGroupRef):
        return None
    from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import read_geotransform

    try:
        gt = read_geotransform(source)
    except Exception:  # noqa: BLE001, a missing/unresolvable/non-GeoTIFF geotransform is optional
        return None
    return {
        "tiepoint_pixel_x": gt.tiepoint_pixel_x, "tiepoint_pixel_y": gt.tiepoint_pixel_y,
        "tiepoint_native_x": gt.tiepoint_native_x, "tiepoint_native_y": gt.tiepoint_native_y,
        "pixel_scale_x": gt.pixel_scale_x, "pixel_scale_y": gt.pixel_scale_y, "epsg": gt.epsg,
    }


def is_georeferenced(source: "str | Path | BandGroupRef") -> bool:
    """Whether ``source`` carries a real per-pixel affine geotransform (a stitched, georectified
    orthomosaic), never a size proxy for one: an ordinary drone/ground capture typically carries
    only a single EXIF GPS point for the camera's own position at capture time, not the
    ModelPixelScaleTag/ModelTiepointTag/GeoKeyDirectoryTag trio this checks for (:func:`
    ~tcip_mcp.pipelines.postprocessing.orthomosaic_mapping.read_geotransform`), regardless of how
    large that capture's own pixel dimensions are."""
    return _optional_geotransform(source) is not None


def raster_content_identity(
    source: "str | Path | BandGroupRef", num_channels: int, *, seed: int, window_size: int,
    max_windows: int,
) -> RasterIdentity:
    """The content identity of one raster file, read through :func:`open_raster`.

    Backend-agnostic by construction: the checksum walks the same :func:`sample_windows`
    selection every backend serves through :meth:`RasterSource.read_region`, so a GDAL-served
    GeoTIFF and a memory-mapped ``.npy`` of identical pixel content resolve the same identity,
    and two different-content rasters of identical dimensions resolve different ones: the
    checksum is the discriminating term, since a GDAL-only attribute cannot provide one for a
    legitimate ``.npy``/``.npz``/whole-decode-TIFF training raster. ``band_interpretations`` is
    read with ``getattr(src, "band_interpretations", None)``, this module's own convention for
    the one GDAL-only attribute (see :class:`GdalSource`), never a refusal condition on its own.

    Raises ``ValueError`` naming the source when the raster genuinely cannot be opened or sampled
    at all (whatever the backend's own open or read failure was, wrapped uniformly here rather
    than leaking a backend-specific exception type); never refuses merely for lacking a GDAL-only
    attribute or a resolvable geotransform, both optional terms here.
    """
    try:
        with open_raster(source, num_channels) as src:
            windows = sample_windows(
                src.width, src.height, seed=seed, window_size=window_size, max_windows=max_windows)
            digest = hashlib.sha256()
            covered = 0
            for rect in windows:
                region = np.ascontiguousarray(src.read_region(rect)[0])
                digest.update(f"{rect.x0},{rect.y0},{rect.x1},{rect.y1}|".encode("ascii"))
                digest.update(region.tobytes())
                covered += rect.width * rect.height
            fraction = covered / float(src.width * src.height)
            identity = RasterIdentity(
                width=int(src.width), height=int(src.height), num_channels=int(src.num_channels),
                dtype=str(src.dtype), pixel_checksum=digest.hexdigest(), seed=int(seed),
                window_size=int(window_size), max_windows=int(max_windows),
                pixel_fraction=float(fraction),
                band_interpretations=getattr(src, "band_interpretations", None),
                geotransform=_optional_geotransform(source),
            )
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001, uniformly named as this function's own refusal
        raise ValueError(f"cannot open or read raster {source!r} for a content identity: {exc}") from exc
    return identity


def raster_identity_matches(recorded: dict, source: "str | Path | BandGroupRef") -> bool:
    """Whether ``source`` is content-identical to a previously recorded
    :func:`raster_content_identity` result (as its ``dataclasses.asdict`` form).

    Recomputes ``source``'s identity under the *exact* sampling parameters (``seed``/
    ``window_size``/``max_windows``) the recorded identity carries, never this call's own
    default: a genuinely identical raster must not read as different purely from parameter drift
    between a training-time and an export-time call. Raises ``ValueError`` (from
    :func:`raster_content_identity`) naming ``source`` when it cannot be opened/sampled at all,
    never silently reporting a false non-match for an unresolvable identity.
    """
    fresh = raster_content_identity(
        source, int(recorded["num_channels"]), seed=int(recorded["seed"]),
        window_size=int(recorded["window_size"]), max_windows=int(recorded["max_windows"]),
    )
    return (
        fresh.width == int(recorded["width"]) and fresh.height == int(recorded["height"])
        and fresh.num_channels == int(recorded["num_channels"])
        and fresh.dtype == recorded["dtype"]
        and fresh.pixel_checksum == recorded["pixel_checksum"]
    )


def content_identity(
    source: "str | Path | BandGroupRef", num_channels: int | None = None,
) -> RasterIdentity:
    """:func:`raster_content_identity` of ``source`` under the platform's own sampling budget.

    ``num_channels`` defaults to :func:`image_route_channel_count`'s rule: a caller with no
    channel count of its own (a proposal run addressing the image it just proposed on) gets the
    same count the image route would open the file at. A caller that already has one (a trained
    model's ``in_chans``, a training-time probe already computed for its own reasons) passes it
    through explicitly and gets exactly that value, never the route's own derivation.
    """
    if num_channels is None:
        num_channels = image_route_channel_count(source)
    return raster_content_identity(
        source, num_channels, seed=CONTENT_IDENTITY_SEED,
        window_size=CONTENT_IDENTITY_WINDOW_SIZE, max_windows=CONTENT_IDENTITY_MAX_WINDOWS)


def georeferenced_raster_identity_mismatch(
    recorded: dict, source: "str | Path | BandGroupRef",
) -> str | None:
    """``None`` when ``source`` is both content-identical to a recorded
    :func:`raster_content_identity` result and carries the georeferencing that result recorded;
    otherwise a summary naming which part mismatched and the values behind it.

    The georeferencing-inclusive companion to :func:`raster_identity_matches`, which it calls for
    the content half rather than restating that comparison. The two scopes answer different
    questions and both are real: pixels alone decide which mosaic a model was calibrated on, so a
    claim about that scope is content-only by construction; a consumer that resolves a pixel
    position to a real-world coordinate (per-plant attribution through an orthomosaic's affine
    tags) also depends on those tags, and to it a pixel-identical copy with a moved tiepoint is a
    different raster.

    Geotransform values compare exactly: both sides are read by the same tag reader and a JSON
    round-trip of a float returns the same value, so a genuine match cannot drift apart here and
    any difference found is a real one, never sampling or serialization noise.
    """
    if not raster_identity_matches(recorded, source):
        return (
            f"content mismatch: {source} is not the raster this identity was recorded on "
            f"(recorded {recorded['width']}x{recorded['height']}x{recorded['num_channels']} "
            f"{recorded['dtype']}, pixel checksum {str(recorded['pixel_checksum'])[:12]})"
        )
    recorded_gt = recorded.get("geotransform")
    supplied_gt = _optional_geotransform(source)
    if recorded_gt == supplied_gt:
        return None
    if recorded_gt is None or supplied_gt is None:
        return (f"georeferencing mismatch: recorded geotransform {recorded_gt!r}, "
                f"supplied {supplied_gt!r}")
    differing = sorted(k for k in set(recorded_gt) | set(supplied_gt)
                       if recorded_gt.get(k) != supplied_gt.get(k))
    return "georeferencing mismatch: " + ", ".join(
        f"{k} recorded {recorded_gt.get(k)!r}, supplied {supplied_gt.get(k)!r}" for k in differing)
