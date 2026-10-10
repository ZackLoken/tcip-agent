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
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from tcip_mcp.pipelines.data.band_groups import (
    MANIFEST_EXT, BandGroupRef, read_band_group_manifest,
)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from tifffile import TiffFile

    from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import (
        GeoreferencingError, GeoTransform,
    )

# The containers band data is read out of as an array. Any other extension is a photographic frame
# decoded through PIL, at the channel counts PIL's own modes cover.
ARRAY_CONTAINER_EXTS = (".npy", ".npz", ".tif", ".tiff")

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
    raster; an empty or out-of-bounds rectangle raises ``ValueError``. The pixels are always a
    copy: mutating them can never corrupt what a later read returns.

    ``target_size`` (output ``(width, height)``) serves the same rectangle resampled to that size;
    it must be the rectangle's :func:`scaled_size` under one scale (``ValueError`` otherwise), and
    the returned :class:`ReadSpec` records the resampling.
    :meth:`read_window` is the same read in row-first argument order.
    """

    width: int
    height: int
    num_channels: int
    dtype: np.dtype
    georeference: GeoTransform | GeoreferencingError | None

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
    holds open in ``_release``. ``georeference`` is the outcome of a TIFF backend's one
    georeference read (:func:`tiff_header`), ``None`` for a backend that reads no TIFF tags."""

    closed = False
    georeference: GeoTransform | GeoreferencingError | None = None

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

    ``band_interpretations`` is the parent's own attribute verbatim when it has one (a
    ``GdalSource`` parent), ``None`` otherwise.
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


def channel_first_reinterpreted(shape: tuple[int, ...], num_channels: int | None) -> bool:
    """Whether a channel-last reading of a 3-D ``shape`` is instead taken as channel-first: the
    leading axis matches the caller's expected band count while the trailing one does not.
    """
    return len(shape) == 3 and shape[0] == num_channels and shape[2] != num_channels


def hwc_array(img: Any) -> np.ndarray:
    """A decoded PIL image or array as ``[H, W, C]``: a 2-D one gains a trailing axis of 1."""
    arr = np.asarray(img)
    return arr[:, :, None] if arr.ndim == 2 else arr


@dataclass(frozen=True)
class _TiffHeader:
    """What the header of the TIFF at ``path`` states about its first series: its shape, its axes
    string, its stored sample dtype, the color table of a one-band uint8 palette image as a
    ``(256, 3)`` uint8 lookup (``None`` for any other image), and the outcome of reading its
    georeferencing tags
    (:func:`~tcip_mcp.pipelines.postprocessing.orthomosaic_mapping.geotransform_of`): the
    geotransform, or the exception naming why the tags state none."""

    path: Path
    shape: tuple[int, ...]
    axes: str
    dtype: np.dtype
    palette_lut: "np.ndarray | None"
    georeference: GeoTransform | GeoreferencingError


def _palette_lut(colormap: np.ndarray) -> np.ndarray:
    """A TIFF ``ColorMap`` tag's 16-bit entries as the ``(256, 3)`` uint8 lookup they encode, the
    high byte of each entry. Raises ``ValueError`` for a tag that does not hold three rows of the
    256 entries a uint8 index addresses."""
    raw = np.asarray(colormap)
    if raw.shape != (3, 256):
        raise ValueError(
            f"a ColorMap tag shaped {raw.shape} holds no RGB palette for uint8 indices")
    return (raw.astype(np.uint16) >> 8).astype(np.uint8).T


def tiff_header(tif: "TiffFile") -> _TiffHeader:
    """The first-series header of the open TIFF ``tif``, at the path its file handle names, read
    without a pixel decode. Raises ``ValueError`` naming the file when tifffile cannot read it,
    and for a palette image this module cannot expand: one that is not uint8, carries no
    ``ColorMap`` tag, or carries one that holds no RGB palette."""
    import tifffile

    from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import (
        GeoreferencingError, geotransform_of,
    )

    path = Path(tif.filehandle.path)

    try:
        series = tif.series[0] if tif.series else None
        if series is not None:
            page = series.keyframe
            shape, axes, dtype = (tuple(int(x) for x in series.shape), str(series.axes),
                                  series.dtype)
            palette = page.photometric == tifffile.PHOTOMETRIC.PALETTE
            colormap = page.tags["ColorMap"].value if "ColorMap" in page.tags else None
            try:
                georeference: GeoTransform | GeoreferencingError = geotransform_of(
                    page.tags, path)
            except GeoreferencingError as refusal:
                georeference = refusal
    except Exception as exc:  # noqa: BLE001, tifffile raises its own kinds for a file it cannot read
        raise ValueError(f"cannot read the TIFF header of '{path}': {exc}") from exc
    if series is None:
        raise ValueError(f"cannot read the TIFF header of '{path}': it holds no image series")
    if not palette:
        return _TiffHeader(path, shape, axes, np.dtype(dtype), None, georeference)
    if dtype != np.dtype("uint8") or colormap is None:
        raise ValueError(
            f"'{path}' is a palette TIFF of {dtype} "
            f"{'with' if colormap is not None else 'without'} a ColorMap tag; the palette images "
            "this platform serves are uint8 with a ColorMap tag")
    return _TiffHeader(path, shape, axes, np.dtype(dtype), _palette_lut(colormap), georeference)


_CHANNEL_LAST = (0, 1, 2)
_CHANNEL_FIRST = (1, 2, 0)
_FRAME_AXES = ("YX", "YXS", "SYX")


@dataclass(frozen=True)
class _Layout:
    """How a decoded array (a numpy container's, or a TIFF's whole decode) is laid out as the frame
    it serves: ``decoded`` is its 3-D shape once a 2-D array gains its trailing axis, ``order``
    the axis order that lays it out channel-last, and ``stacked`` the TIFF fact that its series'
    axes describe no single frame (the multi-page files tifffile writes; never true of a numpy
    container)."""

    decoded: tuple[int, int, int]
    order: tuple[int, int, int]
    stacked: bool

    @property
    def frame(self) -> tuple[int, int, int]:
        """The ``(height, width, channels)`` frame the decode serves in."""
        return (self.decoded[self.order[0]], self.decoded[self.order[1]],
                self.decoded[self.order[2]])


def _array_layout(shape: tuple[int, ...], num_channels: int | None, path: Path, *,
                  planar: bool = False, stacked: bool = False) -> _Layout:
    """The :class:`_Layout` an array of ``shape`` decoded from ``path`` serves at
    ``num_channels``: a 2-D array gains a trailing one-band axis; a 3-D one is channel-first when
    its container states a ``planar`` sample axis, when :func:`channel_first_reinterpreted` says
    so at ``num_channels``, or, for a ``stacked`` TIFF series with no hint, when its leading axis
    is the smaller outer one. Raises ``ValueError`` naming ``path`` and the shape for an array of
    any other number of dimensions."""
    if len(shape) not in (2, 3):
        raise ValueError(f"'{path}' decodes to an array shaped {tuple(shape)}, which is not a "
                         "2-D image: an image array is (height, width) or three-dimensional with "
                         "one band axis.")
    decoded = (int(shape[0]), int(shape[1]), int(shape[2]) if len(shape) == 3 else 1)
    if planar or num_channels is not None:
        first = planar or channel_first_reinterpreted(shape, num_channels)
    else:
        first = stacked and len(shape) == 3 and decoded[0] < decoded[2]
    return _Layout(decoded, _CHANNEL_FIRST if first else _CHANNEL_LAST, stacked)


def _layout(header: _TiffHeader, num_channels: int | None) -> _Layout:
    """The :class:`_Layout` the TIFF ``header`` describes serves at ``num_channels``: its decode,
    a palette image's indices expanded to three bands, laid out by :func:`_array_layout` with
    the facts only its axes state (a planar ``SYX`` series, a stacked one)."""
    palette = header.palette_lut is not None
    shape = (*header.shape, 3) if palette else header.shape
    return _array_layout(shape, num_channels, header.path, planar=header.axes == "SYX",
                         stacked=header.axes not in _FRAME_AXES)


class _ArraySource(_ClosableSource):
    """A backend whose pixels are one already-decoded array held in memory, laid out ``[H, W, C]``
    in the frame its :class:`_Layout` describes; regions are copied out of it."""

    _backend = "array"

    def __init__(self, decoded: np.ndarray, layout: _Layout):
        """``decoded`` is the array as decoded, taken as ``layout``'s decoded shape and laid out by
        its axis order."""
        array = np.transpose(np.reshape(decoded, layout.decoded), layout.order)
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
    ``tcip_annotation.utils.oriented_size`` measures: labels are authored in the upright frame.
    """

    def __init__(self, opened: "PILImage", num_channels: int):
        """``opened`` is the photograph as ``PIL.Image.open`` answered it, its pixels not yet
        decoded; it is decoded here and closed."""
        from tcip_annotation.utils import auto_orient_image

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
    """A TIFF decoded whole in the frame its header describes: the stacked multi-page files
    GDAL's first-IFD reading misreads, the shapes a whole decode reinterprets channel-first, and a
    readable TIFF GDAL cannot open; every other TIFF is :class:`GdalSource`'s."""

    _backend = "tiff_whole"

    def __init__(self, source: "SourceHeader", layout: _Layout):
        """Decodes the TIFF ``source`` already holds open, a palette image expanded through its
        table, laid out in the frame ``layout`` describes."""
        header = source.tiff
        self.georeference = header.georeference
        arr = np.asarray(source.tiff_file.asarray())
        if header.palette_lut is not None:
            arr = header.palette_lut[arr]
        super().__init__(arr, layout)


class NpySource(_ArraySource):
    """A ``.npy`` array as :class:`SourceHeader` memory-mapped it, so a region read touches only
    the pages it covers. An array container carries no georeference."""

    _backend = "npy"


class NpzSource(_ArraySource):
    """A ``.npz`` container's first stored array as :class:`SourceHeader` loaded it."""

    _backend = "npz"


class BandGroupSource(_ClosableSource):
    """Sibling single-band files read as one logical multi-band raster.

    Each member, one band by its own header (:attr:`SourceHeader.members`), is opened on its own
    at that one band and decodes exactly as it would alone; a region is every member's own region
    concatenated on the channel axis, in the manifest's declared band order, and a
    ``target_size`` read is each member's own resampled read
    (the returned spec carries the members' resampling). The group's frame is its first
    band's; a member covering a different extent is refused at open.
    """

    def __init__(self, group: "SourceHeader"):
        ref = group.group
        self._members = [member.open(1) for member in group.members]
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
        self.num_channels = group.channels
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
    (0 native, ``i`` the ``i``-th of :meth:`level_dims`), its
    window the rect mapped onto that level's own dimensions (see ``pipelines.overviews``).

    A one-band palette-color TIFF (its header's ``ColorMap``, :func:`tiff_header`) is expanded
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

    def __init__(self, header: _TiffHeader):
        """Opens the TIFF ``header`` was read from (its ``path``)."""
        self.path = header.path
        self._ds = open_gdal_dataset(self.path)
        self._levels: dict[int, Any] = {}
        self.width = int(self._ds.width)
        self.height = int(self._ds.height)
        self.num_channels = int(self._ds.count)
        self.dtype = np.dtype(self._ds.dtypes[0])
        self.georeference = header.georeference
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
        """The ``(width, height)`` of each reduced-resolution level GDAL serves this raster at,
        finest first, each read off the level's own dataset (:meth:`_level_dataset`): internal
        overviews, or an external sidecar
        :func:`~tcip_mcp.pipelines.overviews.sidecar_valid` confirms; none otherwise."""
        from tcip_mcp.pipelines.overviews import overview_sidecar, sidecar_valid

        if overview_sidecar(self.path).is_file() and not sidecar_valid(self.path):
            return []
        levels = map(self._level_dataset, range(len(self._ds.overviews(1))))
        return [(int(level.width), int(level.height)) for level in levels]

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


class SourceHeader:
    """What ``source`` states, acquired once on first use and held: a TIFF opened through
    tifffile (its header read off it by :func:`tiff_header`, its pixels decoded off the same
    handle only when the whole decode serves it), a numpy container's array (a ``.npy``
    memory-mapped, a ``.npz``'s first array loaded, pixels and all), a photograph opened through
    PIL (its header parsed, its pixels decoded only by the reader :meth:`open` hands it to), a
    band group's members' headers. Every kind, frame, count, georeference and open of ``source``
    reads through this one record; a held file is closed when the reader that decodes it is done
    with it or when the record is collected.

    ``kind`` is ``"group"`` (a band group, or the ``.bandgroup`` manifest standing in for it),
    ``"photo"`` (any other extension outside :data:`ARRAY_CONTAINER_EXTS`, decoded whole through
    PIL), or an array container's extension cut to ``"tif"``, ``"npy"`` or ``"npz"``."""

    def __init__(self, source: "str | Path | BandGroupRef"):
        self.source = source if isinstance(source, BandGroupRef) else Path(source)
        ext = self.path.suffix.lower()
        self.kind = ("group" if ext == MANIFEST_EXT
                     else ext[1:4] if ext in ARRAY_CONTAINER_EXTS else "photo")

    @property
    def path(self) -> Path:
        """The file :attr:`source` names
        (:func:`~tcip_mcp.pipelines.image_utils.source_path_of`): a band group's manifest, any
        other source's own file."""
        from tcip_mcp.pipelines.image_utils import source_path_of

        return Path(source_path_of(self.source))

    @cached_property
    def tiff_file(self) -> "TiffFile":
        """The TIFF opened through tifffile, held until this record is collected. Raises
        ``ValueError`` naming the file when tifffile cannot open it."""
        import tifffile

        try:
            tif = tifffile.TiffFile(str(self.path))
        except Exception as exc:  # noqa: BLE001, tifffile raises its own kinds for a bad file
            raise ValueError(f"cannot read the TIFF header of '{self.path}': {exc}") from exc
        weakref.finalize(self, tif.close)
        return tif

    @cached_property
    def tiff(self) -> _TiffHeader:
        """The TIFF header read off :attr:`tiff_file` (:func:`tiff_header`, whose refusals
        propagate)."""
        return tiff_header(self.tiff_file)

    @cached_property
    def array(self) -> np.ndarray:
        """A ``.npy`` memory-mapped, or a ``.npz``'s first stored array loaded."""
        if self.kind == "npy":
            return np.load(str(self.path), mmap_mode="r")
        with np.load(str(self.path)) as npz:
            return npz[npz.files[0]]

    @cached_property
    def _photo_image(self) -> "PILImage":
        """The photograph as ``PIL.Image.open`` answers it, its pixels not decoded; closed by the
        reader that decodes it, or when this header is collected, whichever comes first."""
        from PIL import Image

        image = Image.open(self.path)
        weakref.finalize(self, image.close)
        return image

    @cached_property
    def _photo(self) -> tuple[int, tuple[int, int]]:
        """A photograph's band count and EXIF-upright ``(width, height)``, from its PIL header."""
        from tcip_annotation.utils import oriented_size

        return len(self._photo_image.getbands()), oriented_size(self._photo_image)

    @cached_property
    def group(self) -> BandGroupRef:
        """The band group: the one given, or the one its manifest names (whose refusals
        propagate)."""
        assert self.kind == "group", "only a band group's header names one"
        if isinstance(self.source, BandGroupRef):
            return self.source
        return read_band_group_manifest(self.path)

    @cached_property
    def members(self) -> "list[SourceHeader]":
        """A band group's members' headers, in the manifest's declared band order: each one band
        laid out as it is opened, at one band (:meth:`_layout_at`), what a member of a band group
        is. Raises ``ValueError`` naming a member that holds another count read that way."""
        members = [SourceHeader(p) for p in self.group.bands.values()]
        for name, member in zip(self.group.bands, members):
            bands = member._layout_at(1)[2]
            if bands != 1:
                raise ValueError(
                    f"band group {self.group.stem!r} ({self.group.manifest_path}): band {name!r} "
                    f"({member.path.name}) holds {bands} bands; each member of a band group is "
                    "one band.")
        return members

    @cached_property
    def channels(self) -> int:
        """The band count the source's own data holds at no hint (:meth:`_layout_at`)."""
        return self._layout_at(None)[2]

    @property
    def route_channels(self) -> int:
        """The count a plain (non-composited) image-route read opens the source at: three for
        every photograph, whatever its own bands, since a plain serve decodes a grayscale,
        grayscale-with-alpha or palette frame through PIL's RGB expansion; :attr:`channels`
        otherwise."""
        return 3 if self.kind == "photo" else self.channels

    @property
    def display_frame(self) -> tuple[int, int]:
        """``(width, height)`` the image route serves the source in (:meth:`frame_at` at
        :attr:`route_channels`): the frame a viewer draws and annotation coordinates are measured
        in."""
        return self.frame_at(self.route_channels)

    @property
    def georeference(self) -> "GeoTransform | GeoreferencingError | None":
        """A TIFF's georeference read outcome (:class:`_TiffHeader`); ``None`` for a source that
        carries no TIFF tags."""
        return self.tiff.georeference if self.kind == "tif" else None

    def _layout_at(self, num_channels: int | None) -> tuple[int, int, int]:
        """``(height, width, bands)`` of the source's own data laid out at ``num_channels``
        (``None``: no hint), from what this record acquired: a TIFF's header (:func:`_layout`,
        no pixel decode), a numpy array's shape (:func:`_array_layout`; a ``.npy`` memory-mapped,
        a ``.npz``'s array loaded), a photograph's PIL header (never re-laid out), a band group's
        first member at one band and one band per member. Raises what those layouts raise."""
        if self.kind == "group":
            height, width, _one = self.members[0]._layout_at(1)
            return height, width, len(self.members)
        if self.kind == "tif":
            return _layout(self.tiff, num_channels).frame
        if self.kind == "photo":
            width, height = self._photo[1]
            return height, width, self._photo[0]
        return _array_layout(self.array.shape, num_channels, self.path).frame

    def frame_at(self, num_channels: int) -> tuple[int, int]:
        """``(width, height)`` as the source is served at ``num_channels`` (:meth:`_layout_at`).
        Raises what :meth:`open` raises for a count the source cannot be served at."""
        if self.kind == "photo":
            self._refuse_unless_photographic(num_channels)
        height, width, _bands = self._layout_at(num_channels)
        return width, height

    def _refuse_unless_photographic(self, num_channels: int) -> None:
        """Refuse a photograph at a count PIL has no mode for, naming the containers that carry
        band data."""
        if num_channels not in _PIL_MODES:
            raise ValueError(
                f"Cannot load a {num_channels}-channel image from '{self.path.suffix}'. "
                "Use .npy/.npz or a multi-band GeoTIFF (.tif/.tiff).")

    def windowed(self, num_channels: int) -> "RasterSource | None":
        """The one dispatch: the reader that serves the source at ``num_channels`` without
        decoding pixels, for the caller to close, or ``None`` for one the whole decode serves.

        A memory-mapped ``.npy`` is windowed. A TIFF is GDAL-served unless its header already
        says the whole decode serves it (its axes describe no single frame, as GDAL's first-IFD
        read of a stacked multi-page file would misread, or the frame to serve at
        ``num_channels`` is not the one its axes describe), GDAL cannot open it, or the frame GDAL
        opens is not that one. Every other kind decodes whole."""
        if self.kind == "npy":
            return NpySource(self.array, _array_layout(self.array.shape, num_channels, self.path))
        if self.kind != "tif":
            return None
        own = _layout(self.tiff, None)
        if own.stacked or _layout(self.tiff, num_channels).frame != own.frame:
            return None
        try:
            source = self.gdal()
        except ValueError:
            return None
        if (source.height, source.width, source.num_channels) != own.frame:
            source.close()
            return None
        return source

    def gdal(self) -> "GdalSource":
        """The TIFF opened through GDAL as it sits on disk, whatever count it is served at: the
        reader of its physical overview pyramid, whether or not a read at some count would
        decode it whole. Raises ``ValueError`` where GDAL cannot open it."""
        return GdalSource(self.tiff)

    def open(self, num_channels: int) -> RasterSource:
        """The reader that serves the source at ``num_channels``: :meth:`windowed`'s, else the
        whole decode of what this record already holds. ``num_channels`` routes (which PIL mode
        a photograph decodes in, which axis order a numpy/TIFF array carries) and is never
        checked against the file. A photograph at a count PIL has no mode for raises
        ``ValueError`` naming the containers that carry band data; a photograph's record opens
        once."""
        reader = self.windowed(num_channels)
        if reader is not None:
            return reader
        if self.kind == "group":
            return BandGroupSource(self)
        if self.kind == "tif":
            return TiffWholeSource(self, _layout(self.tiff, num_channels))
        if self.kind == "npz":
            return NpzSource(self.array, _array_layout(self.array.shape, num_channels, self.path))
        self._refuse_unless_photographic(num_channels)
        return PhotographicSource(self._photo_image, num_channels)

    def open_at_route_count(self) -> RasterSource:
        """:meth:`open` at :attr:`route_channels`, the count a plain image-route read opens the
        source at."""
        return self.open(self.route_channels)


def open_raster(source: "str | Path | BandGroupRef", num_channels: int) -> RasterSource:
    """:meth:`SourceHeader.open` of ``source`` at ``num_channels``: a 5-band GeoTIFF opened at 3
    still reads as 5 bands."""
    return SourceHeader(source).open(num_channels)


# ── Process-local pool of open sources ───────────────────────────────────

_POOL: "OrderedDict[tuple, RasterSource]" = OrderedDict()
_POOL_PID: int | None = None
_POOL_BYTES = 0


def file_version(path: Path) -> tuple[str, int, int]:
    """A file's path, modification time and size, read off the filesystem without opening it.
    Raises ``FileNotFoundError`` for a path no file is at."""
    st = path.stat()
    return str(path), int(st.st_mtime_ns), int(st.st_size)


def source_version(header: "SourceHeader") -> tuple:
    """The version on disk of the source ``header`` describes (:func:`file_version`): its
    file's; a band group's manifest's beside each band's name and its member file's, since the
    manifest can be rewritten to name another file and a member can be rewritten under an
    untouched manifest."""
    if header.kind == "group":
        members = tuple((name, *file_version(p)) for name, p in header.group.bands.items())
        return file_version(header.path), members
    return file_version(header.path)


def pooled_source(header: SourceHeader, num_channels: int) -> "RasterSource | None":
    """The windowed reader for ``header``'s source from this process's pool, opening one through
    :meth:`SourceHeader.windowed` if it holds none; ``None`` for a source that decodes whole,
    and nothing is pooled.

    The pool keeps recently used sources open and evicts least-recently-used sources (closing them)
    once what it holds exceeds this module's pool budget. It belongs to the process that filled it:
    a forked worker finds it empty.

    A caller must not close a reader this returns; the pool owns it.

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
    key = (source_version(header), num_channels)
    existing = _POOL.get(key)
    if existing is not None:
        _POOL.move_to_end(key)
        return existing
    opened = header.windowed(num_channels)
    if opened is None:
        return None
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


def raster_content_identity(
    src: RasterSource, *, seed: int, window_size: int, max_windows: int,
) -> RasterIdentity:
    """The content identity of the open raster ``src``.

    The checksum walks the same :func:`sample_windows` selection every backend serves through
    :meth:`RasterSource.read_region`, so a GDAL-served GeoTIFF and a memory-mapped ``.npy`` of
    identical pixel content resolve the same pixel checksum and frame; their georeference and
    band interpretations still differ by backend. ``band_interpretations`` is read with
    ``getattr(src, "band_interpretations", None)``, ``geotransform`` off the reader's
    ``georeference``.

    Raises ``ValueError`` when the raster cannot be sampled at all; never refuses for lacking a
    GDAL-only attribute or a geotransform.
    """
    try:
        windows = sample_windows(
            src.width, src.height, seed=seed, window_size=window_size, max_windows=max_windows)
        digest = hashlib.sha256()
        covered = 0
        for rect in windows:
            region = np.ascontiguousarray(src.read_region(rect)[0])
            digest.update(f"{rect.x0},{rect.y0},{rect.x1},{rect.y1}|".encode("ascii"))
            digest.update(region.tobytes())
            covered += rect.width * rect.height
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001, uniformly named as this function's own refusal
        raise ValueError(f"cannot read this raster for a content identity: {exc}") from exc
    import dataclasses

    from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import GeoTransform

    georeference = src.georeference
    return RasterIdentity(
        width=int(src.width), height=int(src.height), num_channels=int(src.num_channels),
        dtype=str(src.dtype), pixel_checksum=digest.hexdigest(), seed=int(seed),
        window_size=int(window_size), max_windows=int(max_windows),
        pixel_fraction=float(covered / float(src.width * src.height)),
        band_interpretations=getattr(src, "band_interpretations", None),
        geotransform=(dataclasses.asdict(georeference)
                      if isinstance(georeference, GeoTransform) else None),
    )


def open_as_recorded(
    recorded: dict, source: "str | Path | BandGroupRef",
) -> tuple[RasterSource, RasterIdentity | None]:
    """``source`` opened at the count a previously recorded :func:`raster_content_identity`
    result (its ``dataclasses.asdict`` form) was taken at, for the caller to close, beside its
    identity recomputed under that result's own sampling parameters when it is
    content-identical to it, ``None`` when it is not. Raises what :func:`open_raster` and
    :func:`raster_content_identity` raise, the reader closed.
    """
    src = open_raster(source, int(recorded["num_channels"]))
    try:
        fresh = raster_content_identity(
            src, seed=int(recorded["seed"]), window_size=int(recorded["window_size"]),
            max_windows=int(recorded["max_windows"]),
        )
    except Exception:
        src.close()
        raise
    matches = (
        fresh.width == int(recorded["width"]) and fresh.height == int(recorded["height"])
        and fresh.num_channels == int(recorded["num_channels"])
        and fresh.dtype == recorded["dtype"]
        and fresh.pixel_checksum == recorded["pixel_checksum"]
    )
    return src, fresh if matches else None


def content_identity(src: RasterSource) -> RasterIdentity:
    """:func:`raster_content_identity` of the open raster ``src`` under the platform's own
    sampling budget."""
    return raster_content_identity(
        src, seed=CONTENT_IDENTITY_SEED, window_size=CONTENT_IDENTITY_WINDOW_SIZE,
        max_windows=CONTENT_IDENTITY_MAX_WINDOWS)


def georeferenced_raster_identity_mismatch(
    recorded: dict, fresh: RasterIdentity | None,
) -> str | None:
    """``None`` when ``fresh``, :func:`open_as_recorded`'s answer for a recorded
    :func:`raster_content_identity` result, is content-identical to it and carries the
    georeferencing it recorded; otherwise a summary naming which part mismatched and the values
    behind it. Geotransform values compare exactly.
    """
    if fresh is None:
        return (
            "content mismatch: this is not the raster the identity was recorded on "
            f"(recorded {recorded['width']}x{recorded['height']}x{recorded['num_channels']} "
            f"{recorded['dtype']}, pixel checksum {str(recorded['pixel_checksum'])[:12]})"
        )
    recorded_gt = recorded["geotransform"]
    supplied_gt = fresh.geotransform
    if recorded_gt == supplied_gt:
        return None
    if recorded_gt is None or supplied_gt is None:
        return (f"georeferencing mismatch: recorded geotransform {recorded_gt!r}, "
                f"supplied {supplied_gt!r}")
    differing = [k for k in recorded_gt if recorded_gt[k] != supplied_gt[k]]
    return "georeferencing mismatch: " + ", ".join(
        f"{k} recorded {recorded_gt[k]!r}, supplied {supplied_gt[k]!r}" for k in differing)
