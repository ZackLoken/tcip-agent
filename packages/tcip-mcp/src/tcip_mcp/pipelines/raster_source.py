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
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from tcip_mcp.pipelines.data.band_groups import BandGroupRef

# The array containers that carry no georeferencing tags at all, whatever is inside them: a door
# that needs meters refuses these by name rather than opening one and reporting a read failure.
UNGEOREFERENCED_ARRAY_EXTS = (".npy", ".npz")

# The containers band data is read out of as an array. Any other extension is a photographic frame
# decoded through PIL, at the channel counts PIL's own modes cover.
ARRAY_CONTAINER_EXTS = UNGEOREFERENCED_ARRAY_EXTS + (".tif", ".tiff")

_PIL_MODES = {1: "L", 3: "RGB", 4: "RGBA"}

# A plain, documented platform default, not derived from a measurement: the share of the host's
# physical RAM the decoded-pixel caches here may hold, leaving the model, tile batch and OS the
# rest.
_RAM_BUDGET_FRACTION = 0.25

# GDAL's block cache's share of that budget; the pooled registry of open sources budgets against
# the remainder. An even split: no measurement yet favors either consumer over the other.
_GDAL_CACHE_SHARE = 0.5

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
    """Hand GDAL's block cache its share of this module's memory budget, once per process: the
    cache is process-global. ``share`` scales the budget down for a process that is one of several
    peers each holding their own GDAL cache (a DataLoader worker: pass ``1 / num_workers`` so
    ``num_workers`` peers together still commit the platform's intended budget).
    """
    from rasterio.env import set_gdal_config

    budget = int(_memory_budget_bytes() * _GDAL_CACHE_SHARE * share)
    set_gdal_config("GDAL_CACHEMAX", max(budget, _GDAL_CACHEMAX_BYTES_FLOOR))


def gdal_cache_bytes() -> int:
    """The block-cache budget currently in force, in bytes: the configuration
    :func:`configure_gdal_cache` set, or what that function would have set for a process that never
    called it.
    """
    from rasterio.env import get_gdal_config

    configured = get_gdal_config("GDAL_CACHEMAX")
    if configured is None:
        return int(_memory_budget_bytes() * _GDAL_CACHE_SHARE)
    return int(configured)


def open_gdal_dataset(path: str | Path, overview: int | None = None):
    """A read-only GDAL dataset for ``path``, or for its ``overview``-th reduced-resolution level
    (0 the finest) when one is named, with GDAL's own failure wrapped in this layer's error naming
    the file.

    The returned object is a rasterio dataset.
    """
    import rasterio

    try:
        if overview is not None:
            return rasterio.open(str(path), overview_level=overview)
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


def rect_contains_rect(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int],
) -> bool:
    """Whether half-open pixel rect ``inner`` lies fully inside half-open pixel rect ``outer``."""
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    return ox0 <= ix0 and oy0 <= iy0 and ix1 <= ox1 and iy1 <= oy1


def rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Whether two half-open pixel rects share any pixel; sharing only an edge does not count."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


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
    from ``seed``, returned in row-major order; a ``max_windows`` at or above the grid's own cell
    count returns every cell, so every pixel exactly once.
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
    """How a read was served: which backend decoded it and the resampling that produced it.

    A plain read serves full resolution with ``resample`` ``None``; a read with a ``target_size``
    records the resampling algorithm that was requested. The served resolution is the returned
    pixels' own size against the rect read, one ratio per axis.
    """

    backend: str
    resample: str | None = None


class RasterSource(Protocol):
    """The read surface every backend in this module exposes.

    :meth:`read_region` returns ``([H, W, C] pixels, ReadSpec)`` for a rectangle lying inside the
    raster; an empty or out-of-bounds rectangle raises ``ValueError`` rather than returning a
    silently clipped array, so a caller that wants an edge tile clips to the raster's own bounds
    itself. The pixels are always a copy: mutating them can never corrupt what a later read
    returns.

    ``target_size`` (output ``(width, height)``) serves the same rectangle resampled to that size;
    it must be the rectangle's :func:`scaled_size` under one scale (``ValueError`` otherwise), and
    the returned :class:`ReadSpec` records the resampling.
    :meth:`read_window` is the same read in row-first argument order.
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

    def level_dims(self) -> list[tuple[int, int]]: ...

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
        """Drop whatever this backend holds open, once."""

    def level_dims(self) -> list[tuple[int, int]]:
        """The ``(width, height)`` of each reduced-resolution level this reader serves a planned
        read or a level tile from, finest first: none, for every backend that holds its pixels
        decoded; :class:`GdalSource` answers its overview levels."""
        return []

    def read_region(self, rect: Rect, *,
                    target_size: tuple[int, int] | None = None) -> tuple[np.ndarray, "ReadSpec"]:
        """Return ``([H, W, C] pixels, ReadSpec)`` for ``rect``; every subclass provides its own."""
        raise NotImplementedError

    def read_window(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """The pixel window ``[y0:y1, x0:x1]``, rows first."""
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

    Exposes ``read_window``, ``height``, ``width`` and ``num_channels``, so a rectangular
    sub-region of a mosaic reads as an ordinary windowed raster source in its own local
    coordinate space.

    ``height``/``width`` always report this view's own rect extent, never the parent source's, and
    every read is translated into the parent's coordinate space by adding the rect's own origin. A
    read past this view's declared bounds raises, so a windowed pass over one region never reads
    pixels outside it.

    ``band_interpretations`` forwards the parent's own attribute verbatim when it has one (a
    ``GdalSource`` parent), absent otherwise.
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


def scaled_size(width: int, height: int, scale: Fraction) -> tuple[int, int]:
    """The ``(width, height)`` a ``width`` x ``height`` rect is read at under one ``scale``: each
    edge times the scale, rounded down, and at least one pixel. The one integer geometry a
    resampled read's size is admitted by."""
    return max(1, math.floor(width * scale)), max(1, math.floor(height * scale))


def level_window(rect: Rect, width: int, height: int, level_w: int, level_h: int) -> Rect:
    """``rect`` of a ``width`` x ``height`` raster mapped onto a level of ``level_w`` x
    ``level_h`` pixels: its origin rounded down and its exclusive end rounded up, so the window
    covers every level pixel the rect touches. The one window geometry a level read opens and a
    read plan is decided on."""
    return Rect(rect.x0 * level_w // width, rect.y0 * level_h // height,
                -(-rect.x1 * level_w // width), -(-rect.y1 * level_h // height))


def _check_target_size(rect: Rect, target_size: tuple[int, int]) -> tuple[int, int]:
    """Validate a ``(width, height)`` output size against ``rect`` and return it as ints.

    The target must be :func:`scaled_size` of the rect under some one scale, tried at the smallest
    scale either edge admits; a distorting target raises ``ValueError``.
    """
    out_w, out_h = int(target_size[0]), int(target_size[1])
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"target_size must be positive, got {out_w}x{out_h}")
    scale = max(Fraction(out_w, rect.width) if out_w > 1 else Fraction(0),
                Fraction(out_h, rect.height) if out_h > 1 else Fraction(0))
    if scaled_size(rect.width, rect.height, scale) != (out_w, out_h):
        raise ValueError(
            f"target_size {out_w}x{out_h} does not preserve the aspect ratio of the "
            f"{rect.width}x{rect.height} region it resamples"
        )
    return out_w, out_h


def _area_downsample(region: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """``region`` resampled to ``out_h`` x ``out_w`` rows x cols by pixel-area averaging: cv2's
    ``INTER_AREA``, the one area resampler used for every backend that resamples in memory."""
    import cv2

    return hwc_array(cv2.resize(region, (out_w, out_h), interpolation=cv2.INTER_AREA))


def _serve_region(region: np.ndarray, rect: Rect, backend: str,
                  target_size: tuple[int, int] | None) -> tuple[np.ndarray, ReadSpec]:
    """The read tail every in-memory backend shares: the native copy as-is, or area-downsampled to
    an aspect-preserving ``target_size``."""
    if target_size is None:
        return region, ReadSpec(backend)
    out_w, out_h = _check_target_size(rect, target_size)
    return _area_downsample(region, out_w, out_h), ReadSpec(backend, resample="area")


def channel_first_reinterpreted(shape: tuple[int, ...], num_channels: int) -> bool:
    """Whether a channel-last reading of a 3-D ``shape`` is instead taken as channel-first: the
    leading axis matches the caller's expected band count while the trailing one does not.
    """
    return len(shape) == 3 and shape[0] == num_channels and shape[2] != num_channels


def _channel_last(arr: np.ndarray, num_channels: int) -> np.ndarray:
    """A decoded array as ``[H, W, C]``, using the caller's expected band count to tell a
    channel-first raster from a channel-last one.

    A 2-D array gains a trailing axis of 1. A 3-D array is transposed when
    :func:`channel_first_reinterpreted` says its shape reads channel-first against the expected
    count, and returned as decoded otherwise.
    """
    arr = np.asarray(arr)
    if arr.ndim == 3 and channel_first_reinterpreted(arr.shape, num_channels):
        return np.transpose(arr, (1, 2, 0))
    return hwc_array(arr)


def hwc_array(img: Any) -> np.ndarray:
    """A decoded PIL image or array as ``[H, W, C]``: a 2-D one gains a trailing axis of 1."""
    arr = np.asarray(img)
    return arr[:, :, None] if arr.ndim == 2 else arr


@dataclass(frozen=True)
class _TiffHeader:
    """What a TIFF's header states about its first series: its shape, its axes string, and the
    color table of a one-band uint8 palette image as a ``(256, 3)`` uint8 lookup (``None`` for
    any other image)."""

    shape: tuple[int, ...]
    axes: str
    palette_lut: "np.ndarray | None"


def _palette_lut(colormap: np.ndarray) -> np.ndarray:
    """A TIFF ``ColorMap`` tag's 16-bit entries as the ``(256, 3)`` uint8 lookup they encode, the
    high byte of each entry. Raises ``ValueError`` for a tag that does not hold three rows of the
    256 entries a uint8 index addresses."""
    raw = np.asarray(colormap)
    if raw.shape != (3, 256):
        raise ValueError(
            f"a ColorMap tag shaped {raw.shape} holds no RGB palette for uint8 indices")
    return (raw.astype(np.uint16) >> 8).astype(np.uint8).T


def _tiff_header(path: str | Path) -> _TiffHeader:
    """``path``'s first-series header, read without a pixel decode. Raises ``ValueError`` naming
    the file when tifffile cannot open it, and for a palette image this module cannot expand: one
    that is not uint8, carries no ``ColorMap`` tag, or carries one that holds no RGB palette."""
    import tifffile

    try:
        tif = tifffile.TiffFile(str(path))
    except Exception as exc:  # noqa: BLE001, tifffile raises its own kinds for a file it cannot open
        raise ValueError(f"cannot read the TIFF header of '{path}': {exc}") from exc
    with tif:
        if not tif.series:
            raise ValueError(f"cannot read the TIFF header of '{path}': it holds no image series")
        series = tif.series[0]
        page = series.keyframe
        shape, axes, dtype = tuple(int(x) for x in series.shape), str(series.axes), series.dtype
        palette = page.photometric == tifffile.PHOTOMETRIC.PALETTE
        colormap = page.tags["ColorMap"].value if "ColorMap" in page.tags else None
    if not palette:
        return _TiffHeader(shape, axes, None)
    if dtype != np.dtype("uint8") or colormap is None:
        raise ValueError(
            f"'{path}' is a palette TIFF of {dtype} "
            f"{'with' if colormap is not None else 'without'} a ColorMap tag; the palette images "
            "this platform serves are uint8 with a ColorMap tag")
    return _TiffHeader(shape, axes, _palette_lut(colormap))


_CHANNEL_LAST = (0, 1, 2)
_CHANNEL_FIRST = (1, 2, 0)
_FRAME_AXES = ("YX", "YXS", "SYX")


@dataclass(frozen=True)
class _Layout:
    """How a TIFF's whole decode is laid out as the frame it serves: ``decoded`` is the 3-D shape
    the decode has once a palette is expanded and a 2-D image gains its trailing axis, ``order``
    the axis order that lays it out channel-last, and ``stacked`` whether the series' axes
    describe no single frame (the multi-page files tifffile writes)."""

    decoded: tuple[int, int, int]
    order: tuple[int, int, int]
    stacked: bool

    @property
    def frame(self) -> tuple[int, int, int]:
        """The ``(height, width, channels)`` frame the decode serves in."""
        return (self.decoded[self.order[0]], self.decoded[self.order[1]],
                self.decoded[self.order[2]])


def _layout(header: _TiffHeader, num_channels: int | None) -> _Layout:
    """The :class:`_Layout` ``header`` serves at ``num_channels``: a planar (``SYX``) series is
    channel-first by its axes; any other series that decodes three-dimensional (a sample axis, or
    a palette's expansion) is channel-first when :func:`channel_first_reinterpreted` says so at
    ``num_channels``, or, for a stacked series with no hint, when its leading axis is the smaller
    outer one. Raises ``ValueError`` for a series that is not a 2-D image."""
    shape, palette = header.shape, header.palette_lut is not None
    if len(shape) == 2:
        decoded, three_d = (shape[0], shape[1], 3 if palette else 1), palette
    elif len(shape) == 3 and not palette:
        decoded, three_d = (shape[0], shape[1], shape[2]), True
    else:
        raise ValueError(f"a TIFF series shaped {shape} is not a 2-D image")
    stacked = header.axes not in _FRAME_AXES
    if header.axes == "SYX":
        first = True
    elif num_channels is not None:
        first = three_d and channel_first_reinterpreted(decoded, num_channels)
    else:
        first = stacked and three_d and decoded[0] < decoded[2]
    return _Layout(decoded, _CHANNEL_FIRST if first else _CHANNEL_LAST, stacked)


def _whole_pixels(path: Path, header: _TiffHeader, layout: _Layout) -> np.ndarray:
    """``path`` decoded whole by ``tifffile.imread``, a palette image expanded through
    ``header``'s table, and laid out in the frame ``layout`` describes."""
    import tifffile

    arr = np.asarray(tifffile.imread(str(path)))
    if header.palette_lut is not None:
        arr = header.palette_lut[arr]
    return np.transpose(hwc_array(arr), layout.order)


def tiff_channel_count(path: str | Path) -> int:
    """The channel count a TIFF serves at, from its header alone (:func:`_layout` with no hint).
    Raises what :func:`_tiff_header` and :func:`_layout` raise."""
    return _layout(_tiff_header(path), None).frame[2]


def tiff_frame(path: str | Path, num_channels: int) -> tuple[int, int, int]:
    """The ``(height, width, channels)`` frame this module serves ``path`` in at
    ``num_channels``, from its header alone (:func:`_layout`). Raises what :func:`_tiff_header`
    and :func:`_layout` raise."""
    return _layout(_tiff_header(path), num_channels).frame


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
        self.num_channels = int(array.shape[2])
        self.dtype = array.dtype

    @property
    def resident_bytes(self) -> int:
        assert self._array is not None, (
            "resident_bytes read after close(): a closed source holds no array"
        )
        return int(self._array.nbytes)

    def read_region(self, rect: Rect, *,
                    target_size: tuple[int, int] | None = None) -> tuple[np.ndarray, ReadSpec]:
        assert self._array is not None, (
            "read_region called after close(): a closed source holds no array"
        )
        _check_region(rect, self.height, self.width)
        region = np.array(self._array[rect.y0:rect.y1, rect.x0:rect.x1])
        return _serve_region(region, rect, self._backend, target_size)

    def _release(self) -> None:
        self._array = None


class PhotographicSource(_ClosableSource):
    """A whole photographic frame decoded through PIL, EXIF-oriented and converted to the mode the
    caller's channel count names (1 -> L, 3 -> RGB, 4 -> RGBA).

    ``image`` is that PIL frame itself. The frame is fully decoded and independent of the file,
    which is closed as soon as it has been read.

    The frame is EXIF-oriented before the mode conversion so it matches what
    ``get_image_dimensions`` measures: labels are authored in the upright frame.
    """

    def __init__(self, path: str | Path | bytes, num_channels: int):
        """``path`` names the file, or is the file's bytes as one read already answered them."""
        import io

        from PIL import Image

        from tcip_annotation.utils import auto_orient_image

        opened = Image.open(io.BytesIO(path) if isinstance(path, bytes) else Path(path))
        self.image = auto_orient_image(opened).convert(_PIL_MODES[num_channels])
        opened.close()
        self.width, self.height = self.image.size
        self.num_channels = len(self.image.getbands())
        self.dtype = np.dtype("uint8")  # L/RGB/RGBA are all 8-bit
        self._frame: np.ndarray | None = None

    @property
    def resident_bytes(self) -> int:
        """The peak this source can hold, stated before any read: the PIL frame plus the ndarray
        copy :meth:`read_region` materializes from it on first use."""
        return int(2 * self.width * self.height * self.num_channels * self.dtype.itemsize)

    def read_region(self, rect: Rect, *,
                    target_size: tuple[int, int] | None = None) -> tuple[np.ndarray, ReadSpec]:
        _check_region(rect, self.height, self.width)
        if self._frame is None:
            self._frame = hwc_array(self.image)
        region = np.array(self._frame[rect.y0:rect.y1, rect.x0:rect.x1])
        return _serve_region(region, rect, "photographic", target_size)

    def _release(self) -> None:
        self._frame = None


class TiffWholeSource(_ArraySource):
    """A TIFF decoded whole (:func:`_whole_pixels`) in the frame its header describes: the
    stacked multi-page files GDAL's first-IFD reading misreads, the shapes a whole decode
    reinterprets channel-first, and a readable TIFF GDAL cannot open; every other TIFF is
    :class:`GdalSource`'s."""

    _backend = "tiff_whole"

    def __init__(self, path: str | Path, header: _TiffHeader, layout: _Layout):
        self.path = Path(path)
        self._describe(_whole_pixels(self.path, header, layout))


class NpySource(_ArraySource):
    """A ``.npy`` array, memory-mapped so a region read touches only the pages it covers. An array
    container carries no georeference."""

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
    (the returned spec carries the members' resampling). The group's frame is its first
    band's; a member covering a different extent is refused at open.
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
                ReadSpec("band_group", resample=member_spec.resample))

    def _release(self) -> None:
        for member in self._members:
            member.close()


class GdalSource(_ClosableSource):
    """A GDAL-readable raster (today: .tif/.tiff) opened read-only and served windowed.

    Regions decode through GDAL's own block cache (budgeted per process by
    :func:`configure_gdal_cache`), so repeated windows over a raster far too large to decode whole
    cost only the blocks they touch. A ``target_size`` read asks RasterIO for the reduced buffer
    directly (``resample_alg=Average``). ``level`` names the overview level the read comes off
    (0 native, ``i`` the ``i``-th of :func:`~tcip_mcp.pipelines.overviews.overview_dims`), its
    window the rect mapped onto that level's own dimensions (see ``pipelines.overviews``).

    A one-band palette-color TIFF (its header's ``ColorMap``, :func:`_tiff_header`) is expanded
    through that table to uint8 RGB, the same pixels PIL's palette decode produces, and reports
    three channels; a ``target_size`` read of one expands the named level's indices at that
    level's resolution first and area-downsamples the RGB, since palette indices are never
    averaged (its pyramid is built by nearest sampling).

    ``band_interpretations`` names each served channel's GDAL color interpretation (lowercase,
    e.g. ``("red", "green", "blue", "alpha")``; ``"undefined"`` when the file declares none), so
    a consumer can tell an alpha band from a spectral one. Only this backend carries the
    attribute; read it with ``getattr(src, "band_interpretations", None)``.

    A GDAL dataset handle is not thread-safe: one instance must never be shared across concurrent
    threads.
    """

    _backend = "gdal"

    def __init__(self, path: str | Path, header: _TiffHeader):
        self.path = Path(path)
        self._ds = open_gdal_dataset(self.path)
        self._levels: dict[int, Any] = {}
        self.width = int(self._ds.width)
        self.height = int(self._ds.height)
        self.num_channels = int(self._ds.count)
        self.dtype = np.dtype(self._ds.dtypes[0])
        self._palette_lut: np.ndarray | None = header.palette_lut
        if self._palette_lut is not None:
            self.num_channels = 3
            self.dtype = np.dtype("uint8")
            self.band_interpretations = ("red", "green", "blue")
        else:
            self.band_interpretations = tuple(
                interp.name.lower() for interp in self._ds.colorinterp)

    @property
    def resident_bytes(self) -> int:
        # Handle-only: decoded blocks live in GDAL's own block cache (configure_gdal_cache's
        # share of the budget), not in this process's pooled accounting.
        return 0

    def read_region(self, rect: Rect, *, target_size: tuple[int, int] | None = None,
                    level: int = 0) -> tuple[np.ndarray, ReadSpec]:
        _check_region(rect, self.height, self.width)
        if level and target_size is None:
            raise ValueError("a read off an overview level is always resampled: name target_size")
        ds = self._ds if level == 0 else self._level_dataset(level - 1)
        window = level_window(rect, self.width, self.height, int(ds.width), int(ds.height))
        out = None if target_size is None else _check_target_size(rect, target_size)
        pixels, resample = self._read_window(ds, window, out)
        if out is None:
            return pixels, ReadSpec(self._backend)
        return pixels, ReadSpec(self._backend, resample=resample)

    def read_level_region(self, level: int, rect: Rect) -> tuple[np.ndarray, ReadSpec]:
        """``rect`` in the pixel grid of overview level ``level`` (1 the finest), read at that
        level's own resolution."""
        ds = self._level_dataset(level - 1)
        _check_region(rect, int(ds.height), int(ds.width))
        pixels, _resample = self._read_window(ds, rect, None)
        return pixels, ReadSpec(self._backend)

    def _read_window(self, ds, window: Rect, out: "tuple[int, int] | None"
                     ) -> tuple[np.ndarray, "str | None"]:
        """``window`` of dataset ``ds`` as ``[H, W, C]``, area-averaged to ``out`` when given (a
        palette raster's indices expanded first, never averaged), and the resampling used."""
        from rasterio.enums import Resampling
        from rasterio.windows import Window

        win = Window(window.x0, window.y0, window.width, window.height)
        if self._palette_lut is not None:
            rgb = self._palette_lut[ds.read(1, window=win)]
            return (rgb, None) if out is None else (_area_downsample(rgb, *out), "area")
        if out is None:
            arr = ds.read(window=win)
        else:
            arr = ds.read(window=win, out_shape=(self.num_channels, out[1], out[0]),
                          resampling=Resampling.average)
        # GDAL returns [C, H, W]; contiguous copy in the platform's [H, W, C] order.
        return (np.ascontiguousarray(np.transpose(arr, (1, 2, 0))),
                None if out is None else "average")

    def level_dims(self) -> list[tuple[int, int]]:
        """This raster's overview levels (:func:`~tcip_mcp.pipelines.overviews.overview_dims`)."""
        from tcip_mcp.pipelines.overviews import overview_dims

        return overview_dims(self.path)

    def _level_dataset(self, overview: int):
        """The ``overview``-th reduced-resolution level, opened once and held until close."""
        if overview not in self._levels:
            self._levels[overview] = open_gdal_dataset(self.path, overview)
        return self._levels[overview]

    def _release(self) -> None:
        for level in self._levels.values():
            level.close()
        self._levels.clear()
        if self._ds is not None:
            self._ds.close()
        self._ds = None


def _memory_budget_bytes() -> int:
    """The decoded-pixel bytes this module lets itself hold: a fraction of the host's total
    physical RAM, read once per process."""
    global _total_ram_bytes
    if _total_ram_bytes is None:
        import psutil

        _total_ram_bytes = int(psutil.virtual_memory().total)
    return int(_total_ram_bytes * _RAM_BUDGET_FRACTION)


def _pool_budget_bytes() -> int:
    """The pooled registry's budget: what the memory budget leaves after GDAL's block-cache
    share."""
    return int(_memory_budget_bytes() * (1.0 - _GDAL_CACHE_SHARE))


def photographic_container(source: "str | Path | BandGroupRef", num_channels: int) -> bool:
    """Whether ``source`` decodes as a whole photographic frame through PIL rather than as band
    data: any extension outside :data:`ARRAY_CONTAINER_EXTS`, at one of the channel counts PIL's
    own modes cover.
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
    PIL's RGB expansion.

    ``probed`` lets a caller that already has :func:`probe_channels`'s answer pass it through.
    """
    from tcip_mcp.pipelines.derivations import probe_channels

    if probed is None:
        probed = probe_channels(source)
    return 3 if photographic_container(source, probed) else probed


def open_array_source(source: "str | Path | BandGroupRef", num_channels: int) -> RasterSource:
    """Open ``source`` as a plain ``[H, W, C]`` array raster: a band group, a numpy container, or a
    TIFF.

    A photographic container (any other extension) is refused at every channel count.
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


def _tiff_needs_whole_decode(
    source: GdalSource, header: _TiffHeader, served: tuple[int, int, int],
) -> bool:
    """Whether a GDAL-opened TIFF must instead decode whole through tifffile: when its header's
    axes describe no single frame (GDAL reads a stacked multi-page file's first page as the
    dataset), when the frame GDAL serves is not the one the axes describe, or when the frame
    to serve (``served``, :func:`_layout` at the caller's count) is not that one either."""
    own = _layout(header, None)
    return own.stacked or own.frame != (source.height, source.width, source.num_channels) \
        or served != own.frame


def _tiff_dispatch(path: Path, num_channels: int) -> "GdalSource | tuple[_TiffHeader, _Layout]":
    """The reader that serves a TIFF at ``num_channels``: the open :class:`GdalSource` when GDAL
    sees the whole raster, else the header and the layout the whole decode reads it by. The
    header and the layout are resolved first, so a file tifffile cannot read and a series that is
    not a 2-D image refuse before any handle opens."""
    header = _tiff_header(path)
    layout = _layout(header, num_channels)
    try:
        source = GdalSource(path, header)
    except ValueError:
        return header, layout
    if _tiff_needs_whole_decode(source, header, layout.frame):
        source.close()
        return header, layout
    return source


def _open_tiff(path: Path, num_channels: int) -> RasterSource:
    """:func:`_tiff_dispatch`'s GDAL source, or the whole decode in the layout it resolved."""
    served = _tiff_dispatch(path, num_channels)
    if isinstance(served, GdalSource):
        return served
    return TiffWholeSource(path, *served)


def palette_tiff(path: str | Path) -> bool:
    """Whether ``path`` is a palette TIFF this module expands through its color table, from the
    header alone (:func:`_tiff_header`, whose refusals propagate)."""
    return _tiff_header(path).palette_lut is not None


def _windowed_probe(source: "str | Path | BandGroupRef",
                    num_channels: int) -> "GdalSource | None":
    """The header-only :class:`GdalSource` :func:`_tiff_dispatch` opens for a TIFF ``source`` at
    ``num_channels``, for the caller to close; ``None`` for a TIFF that would decode whole and
    for any source that is not a TIFF. A TIFF's header refusals propagate."""
    if isinstance(source, BandGroupRef) or Path(source).suffix.lower() not in (".tif", ".tiff"):
        return None
    served = _tiff_dispatch(Path(source), num_channels)
    return served if isinstance(served, GdalSource) else None


def opens_windowed(source: "str | Path | BandGroupRef", num_channels: int) -> bool:
    """Whether :func:`open_raster` would serve ``source`` through a backend that opens without
    decoding pixels and reads windows on demand (a GDAL-served raster, a memory-mapped ``.npy``).

    A photographic frame, an ``.npz`` and a stacked TIFF decode whole at open, and a band group
    answers ``False`` whatever its members open through, so a caller deciding whether an eager
    open is affordable asks here first. For a TIFF the answer needs GDAL's own header read; a
    probe source is opened and closed again, header-only, never decoding pixels, and a TIFF
    whose header refuses raises that refusal.
    """
    if not isinstance(source, BandGroupRef) and photographic_container(source, num_channels):
        return False
    if not isinstance(source, BandGroupRef) and Path(source).suffix.lower() == ".npy":
        return True
    probe = _windowed_probe(source, num_channels)
    if probe is None:
        return False
    probe.close()
    return True


def level_dims(source: "str | Path | BandGroupRef", num_channels: int) -> list[tuple[int, int]]:
    """:meth:`RasterSource.level_dims` of the reader :func:`open_raster` would serve ``source``
    at ``num_channels`` through, decided from headers alone: a TIFF's probe source is opened and
    closed again without decoding pixels, and a TIFF whose header refuses raises that refusal;
    every other source has none."""
    probe = _windowed_probe(source, num_channels)
    if probe is None:
        return []
    with probe:
        return probe.level_dims()


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
    """The identity an open source is pooled under: the file's path, modification time and size,
    and the channel count it was opened at.

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

    The pool keeps recently used sources open and evicts least-recently-used sources (closing them)
    once what it holds exceeds this module's pool budget. It belongs to the process that filled it:
    a forked worker finds it empty.

    A caller must not close what this returns; the pool owns it.

    A GDAL dataset handle is not thread-safe, so this pool must never vend one :class:`GdalSource`
    to two concurrent threads.
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

    Identifies one raster file, not a dataset
    (``resolution.dataset_hash``/``dataset_fingerprint.dataset_fingerprint`` do that).
    ``pixel_checksum`` is the discriminating term (two different rasters of identical dimensions
    checksum differently); ``width``/``height``/``num_channels``/``dtype`` travel alongside it.

    ``band_interpretations`` is present only when the backend that served this raster carries it
    (GDAL) and is ``None`` otherwise. ``geotransform`` is optional, present only when the raster is
    a georeferenced GeoTIFF whose affine tags this module can read.

    ``seed``/``window_size``/``max_windows``/``pixel_fraction`` are the sampling parameters that
    produced ``pixel_checksum``, recorded so a later comparison recomputes under the same
    parameters.
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


def raster_content_identity(
    source: "str | Path | BandGroupRef", num_channels: int, *, seed: int, window_size: int,
    max_windows: int,
) -> RasterIdentity:
    """The content identity of one raster file, read through :func:`open_raster`.

    The checksum walks the same :func:`sample_windows` selection every backend serves through
    :meth:`RasterSource.read_region`, so a GDAL-served GeoTIFF and a memory-mapped ``.npy`` of
    identical pixel content resolve the same identity. ``band_interpretations`` is read with
    ``getattr(src, "band_interpretations", None)``.

    Raises ``ValueError`` naming the source when the raster cannot be opened or sampled at all;
    never refuses for lacking a GDAL-only attribute or a resolvable geotransform.
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
        raise ValueError(
            f"cannot open or read raster {source!r} for a content identity: {exc}"
        ) from exc
    return identity


def raster_identity_matches(recorded: dict, source: "str | Path | BandGroupRef") -> bool:
    """Whether ``source`` is content-identical to a previously recorded
    :func:`raster_content_identity` result (as its ``dataclasses.asdict`` form), recomputed under
    the recorded identity's own sampling parameters (``seed``/``window_size``/``max_windows``).
    Raises ``ValueError`` (from :func:`raster_content_identity`) naming ``source`` when it cannot
    be opened/sampled at all.
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
    ``num_channels`` defaults to :func:`image_route_channel_count`'s rule; a stated value is used
    as given.
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
    :func:`raster_content_identity` result (:func:`raster_identity_matches`) and carries the
    georeferencing that result recorded; otherwise a summary naming which part mismatched and the
    values behind it. Geotransform values compare exactly.
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
