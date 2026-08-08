"""Raster reading: one open-and-read surface for every image source this platform decodes.

There is a backend per kind of source (a photographic frame through PIL, a TIFF whole or strip by
strip, a numpy container, a group of sibling single-band files), and each serves pixel regions
through the same small surface, so a caller that wants one window of a 90 GB orthomosaic and a
caller that wants a whole 4-band capture compose the same way.

:func:`open_raster` picks the backend from the source itself. The channel count a caller passes is
a routing hint (which PIL mode to decode a photograph in, which axis order a numpy/TIFF array
carries), never an assertion about the file: a raster whose real band count disagrees is still read
exactly as it sits on disk, and checking that is the caller's own job.
"""

from __future__ import annotations

import logging
import math
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

# The strip cache a caller that supplies no tiling geometry gets: a plain bound on resident strips
# for an access pattern this module cannot know, not a figure derived from any file.
DEFAULT_STRIP_CACHE_CAPACITY = 64

# Provisional platform budget, not derived from a measurement: the share of the host's physical RAM
# the decoded-pixel caches here may hold, leaving the model, tile batch and OS the rest.
_RAM_BUDGET_FRACTION = 0.25

# Used only when psutil cannot be imported: a deliberately low assumed host size, so a host whose
# real memory can't be read is under-budgeted rather than over.
_ASSUMED_TOTAL_RAM_BYTES = 8 * 1024 ** 3

_total_ram_bytes: int | None = None


class UnsupportedRasterLayout(ValueError):
    """A TIFF's on-disk pixel layout isn't one :class:`StripTiffSource` decodes: an internally
    tiled raster, planar (band-separate) samples, or a multi-page file whose first page isn't the
    whole raster (e.g. a channel-last raster stored one row-block per page). It reads only the
    strip-based, contiguous-sample, single-page layout; :func:`open_raster` sends every other
    layout to a whole decode instead."""


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

    Every backend here serves full resolution only, so ``scale`` is always 1.0 and ``resample``
    always ``None``; the fields carry the parameters of an overview read for a source that serves
    one.
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
    """

    width: int
    height: int
    num_channels: int
    dtype: np.dtype
    resident_bytes: int

    def read_region(self, rect: Rect) -> tuple[np.ndarray, ReadSpec]: ...

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


def _channel_last(arr: np.ndarray, num_channels: int) -> np.ndarray:
    """A decoded array as ``[H, W, C]``, using the caller's expected band count to tell a
    channel-first raster from a channel-last one.

    A 2-D array gains a trailing axis of 1. A 3-D array is transposed only when its first axis
    matches the expected count and its last axis does not, the one shape where channel-first is
    unambiguous against what the caller asked for; every other shape is returned as decoded, so a
    count that disagrees with the file never reshapes its pixels.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim == 3 and arr.shape[0] == num_channels and arr.shape[2] != num_channels:
        return np.transpose(arr, (1, 2, 0))
    return arr


def tiff_series_shape(path: str | Path) -> tuple[int, ...] | None:
    """Header-only TIFF series shape (no pixel decode); ``None`` if it can't be read this way.

    ``tif.series[0].shape``, not ``pages[0]``: a channel-last TIFF stores each row-block as its own
    page, so ``pages[0]`` of a 24x40x5 raster is ``(40, 5)``. Shared by ``image_utils`` and
    ``derivations.probe_channels`` so a full pixel read is never paid just to learn the shape.
    """
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tif:
            return tuple(int(x) for x in tif.series[0].shape)
    except Exception:  # noqa: BLE001, fall through to a full read rather than guess
        return None


class _ArraySource(_ClosableSource):
    """A backend whose pixels are one already-decoded ``[H, W, C]`` array held in memory.

    A subclass decodes that array in its own constructor and hands it to :meth:`_describe`; regions
    are copied out of it.
    """

    _backend = "array"

    def _describe(self, array: np.ndarray) -> None:
        self._array = array
        self.height = int(array.shape[0])
        self.width = int(array.shape[1])
        self.num_channels = int(array.shape[2]) if array.ndim > 2 else 1
        self.dtype = array.dtype

    @property
    def resident_bytes(self) -> int:
        return int(self._array.nbytes)

    def read_region(self, rect: Rect) -> tuple[np.ndarray, ReadSpec]:
        _check_region(rect, self.height, self.width)
        region = self._array[rect.y0:rect.y1, rect.x0:rect.x1]
        return np.array(region), ReadSpec(self._backend)

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
        return int(self.width * self.height * self.num_channels * self.dtype.itemsize)

    def read_region(self, rect: Rect) -> tuple[np.ndarray, ReadSpec]:
        _check_region(rect, self.height, self.width)
        if self._frame is None:
            frame = np.asarray(self.image)
            self._frame = frame[:, :, None] if frame.ndim == 2 else frame
        region = self._frame[rect.y0:rect.y1, rect.x0:rect.x1]
        return np.array(region), ReadSpec("photographic")

    def _release(self) -> None:
        self._frame = None


class TiffWholeSource(_ArraySource):
    """A TIFF decoded whole by ``tifffile.imread``, for every layout :class:`StripTiffSource`
    cannot serve: internally tiled, planar (band-separate), and the multi-page layouts tifffile
    writes for a channel-last N-band raster or a plain ``[C, H, W]`` stack."""

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
    manifest's declared band order. The group's frame is its first band's, the same frame
    ``image_utils.image_dimensions`` reports for a group; a member covering a different extent is
    refused at open rather than cropped or resampled to fit, since the pixels of a group whose
    bands disagree on the frame cannot be stacked into one raster.
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

    def read_region(self, rect: Rect) -> tuple[np.ndarray, ReadSpec]:
        bands = [m.read_region(rect)[0] for m in self._members]
        return np.concatenate(bands, axis=-1), ReadSpec("band_group")

    def _release(self) -> None:
        for member in self._members:
            member.close()


class StripTiffSource(_ClosableSource):
    """Opens a strip-based TIFF once and serves cheap, repeated pixel-window reads from it.

    Decodes only the strips a requested window actually overlaps, never the whole file, so one
    instance composes with a tiling loop that requests many windows across a raster far too large
    to decode whole, instead of reopening and reparsing the file per tile. Use as a context manager
    or call :meth:`close`; the file handle stays open for the source's lifetime.

    Only a strip-based raster with contiguous (chunky) samples is served: an internally tiled TIFF
    or one with planar (band-separate) samples raises :class:`UnsupportedRasterLayout`. The first
    page must also be the whole raster, cross-checked against the file's series shape
    (:func:`tiff_series_shape`), because a channel-last TIFF stored one row-block per page makes
    ``pages[0]`` report a single row-block's shape, and windows sliced from that geometry would be
    silently wrong. Decoding depends on a working codec for the file's compression: LZW and JPEG
    come from ``imagecodecs``.

    A real orthomosaic's strips run the raster's full width at a small ``rowsperstrip``, so a
    tiling loop that scans left to right across one row-band before advancing (``tile_positions``'
    own order) requests the same strips again for every tile in that band. Decoded strips are
    cached (an LRU keyed by strip index) so a strip already decoded to serve one window is sliced
    from memory for the next rather than re-read and re-decoded. The capacity is
    ``strip_cache_capacity`` when given; otherwise it is derived from the tiling geometry
    (``tile_size`` plus ``overlap`` or ``stride``, see :func:`derive_strip_cache_capacity`), which
    is what holds a row-major scan to one decode per strip; with neither it is
    :data:`DEFAULT_STRIP_CACHE_CAPACITY`. A caller whose access pattern isn't row-major still gets
    correct results, just fewer cache hits.
    """

    def __init__(self, path: str | Path, *, strip_cache_capacity: int | None = None,
                 tile_size: int | None = None, overlap: float | None = None,
                 stride: int | None = None):
        import tifffile

        self.path = Path(path)
        self._tif = tifffile.TiffFile(str(self.path))
        page = self._tif.pages[0]
        if page.is_tiled:
            self._tif.close()
            raise UnsupportedRasterLayout(
                f"{self.path}: internally tiled TIFF; this source only decodes strip-based rasters."
            )
        if page.planarconfig != 1:
            self._tif.close()
            raise UnsupportedRasterLayout(
                f"{self.path}: planar (band-separate) sample layout; this source only decodes "
                "contiguous (chunky) samples."
            )
        height = int(page.imagelength)
        width = int(page.imagewidth)
        num_channels = int(page.samplesperpixel)
        series_shape = tiff_series_shape(self.path)
        if series_shape is None:
            self._tif.close()
            raise UnsupportedRasterLayout(
                f"{self.path}: the file's series shape can't be read, so there is no way to "
                "verify the first page spans the whole raster."
            )
        normalized = series_shape + (1,) if len(series_shape) == 2 else series_shape
        if normalized != (height, width, num_channels):
            self._tif.close()
            raise UnsupportedRasterLayout(
                f"{self.path}: first page's shape {(height, width, num_channels)} (H, W, C) "
                f"disagrees with the raster's series shape {series_shape}: a multi-page layout "
                "(e.g. a channel-last raster stored one row-block per page) whose pages this "
                "source's single-page strip decoding would misinterpret."
            )
        self._page = page
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.dtype = page.dtype
        self._rows_per_strip = int(page.rowsperstrip)
        self._strip_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.strip_cache_capacity = self._resolve_capacity(
            strip_cache_capacity, tile_size, overlap, stride)
        self.strip_decode_count = 0  # cache misses actually decoded from disk; a caching probe

    def _resolve_capacity(self, strip_cache_capacity: int | None, tile_size: int | None,
                          overlap: float | None, stride: int | None) -> int:
        """The number of decoded strips this source keeps resident.

        An explicit ``strip_cache_capacity`` is honored as given. A tiling geometry derives one
        (:func:`derive_strip_cache_capacity`), which is then held to this process's memory budget:
        a derivation over the budget is capped, with a warning naming the re-decode amplification
        the cap costs, never silently clamped and never refused.
        """
        if strip_cache_capacity is not None:
            return int(strip_cache_capacity)
        if tile_size is None:
            return DEFAULT_STRIP_CACHE_CAPACITY
        derived = derive_strip_cache_capacity(
            self._rows_per_strip, int(tile_size), overlap=overlap, stride=stride)
        budget = _memory_budget_bytes()
        capped = max(1, int(budget // max(1, self._strip_bytes())))
        if derived > capped:
            logger.warning(
                "%s: a %d-strip cache for %dpx tiles needs %.2f GiB, over this process's %.2f GiB "
                "budget; capping at %d strips, which re-decodes strips about %.1fx over the scan.",
                self.path, derived, int(tile_size), derived * self._strip_bytes() / 1024 ** 3,
                budget / 1024 ** 3, capped, derived / capped,
            )
            return capped
        return derived

    def _strip_bytes(self) -> int:
        """Decoded size of one strip. A file whose ``rowsperstrip`` exceeds its own height stores
        fewer rows than it declares."""
        rows = min(self._rows_per_strip, self.height)
        return int(rows * self.width * self.num_channels * self.dtype.itemsize)

    @property
    def resident_bytes(self) -> int:
        return int(self.strip_cache_capacity * self._strip_bytes())

    def _release(self) -> None:
        self._strip_cache.clear()
        self._tif.close()

    def read_region(self, rect: Rect) -> tuple[np.ndarray, ReadSpec]:
        return self.read_window(rect.y0, rect.y1, rect.x0, rect.x1), ReadSpec("strip_tiff")

    def read_window(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Decode and return the ``[y1-y0, x1-x0, num_channels]`` pixel window ``[y0:y1, x0:x1]``.

        Raises ``ValueError`` for an empty or out-of-bounds window; a caller that wants a padded
        edge tile clips the request to the raster's bounds first and pads the result separately
        (mirrors ``image_utils.crop_pad_tile``, which this source doesn't duplicate).

        A window spanning several strips is assembled from each strip sliced down to the requested
        x-range first: a real orthomosaic's strips run the raster's full width, so concatenating
        whole strips and discarding everything outside ``x0:x1`` would copy orders of magnitude
        more data than the window itself needs.
        """
        _check_region(Rect(x0, y0, x1, y1), self.height, self.width)
        rps = self._rows_per_strip
        strip0 = y0 // rps
        strip1 = (y1 - 1) // rps
        if strip0 == strip1:
            block = self._decode_strip(strip0)
            row_offset = strip0 * rps
            # A copy, not a view: `block` is a cached, reused strip array, and a caller mutating
            # its returned window must not corrupt what a later window reads back.
            return np.array(block[y0 - row_offset : y1 - row_offset, x0:x1])
        # np.concatenate always allocates, so the multi-strip branch copies on its own.
        pieces = []
        for strip_index in range(strip0, strip1 + 1):
            strip = self._decode_strip(strip_index)
            row_offset = strip_index * rps
            row_lo = max(y0, row_offset) - row_offset
            row_hi = min(y1, row_offset + strip.shape[0]) - row_offset
            pieces.append(strip[row_lo:row_hi, x0:x1])
        return np.concatenate(pieces, axis=0)

    def _decode_strip(self, strip_index: int) -> np.ndarray:
        """Return strip ``strip_index``'s decoded pixel rows, from the LRU cache if a previous
        window already decoded it, decoding from disk only on a genuine miss.
        """
        cached = self._strip_cache.get(strip_index)
        if cached is not None:
            self._strip_cache.move_to_end(strip_index)
            return cached
        fh = self._page.parent.filehandle
        fh.seek(self._page.dataoffsets[strip_index])
        data = fh.read(self._page.databytecounts[strip_index])
        segment, _position, _shape = self._page.decode(data, strip_index)
        decoded = segment[0]
        self._strip_cache[strip_index] = decoded
        if len(self._strip_cache) > self.strip_cache_capacity:
            self._strip_cache.popitem(last=False)
        self.strip_decode_count += 1
        return decoded


def derive_strip_cache_capacity(rows_per_strip: int, tile_size: int, *,
                                overlap: float | None = None, stride: int | None = None) -> int:
    """Strips to keep resident so a row-major tile scan decodes each strip once.

    One tile row-band spans ``ceil(tile_size / rows_per_strip)`` strips, and the next band starts
    ``stride`` rows further down, so it re-reads the ``ceil((tile_size - stride) / rows_per_strip)``
    strips the overlap keeps in view: holding both means no strip is decoded twice. ``stride``
    defaults to ``tiling.compute_stride(tile_size, overlap)``, the stride the tiler itself walks,
    never a second formula for it.
    """
    if stride is None:
        if overlap is None:
            raise ValueError(
                "deriving a strip cache capacity needs the tiling geometry: pass overlap (or the "
                "stride it produces) alongside tile_size, or pass strip_cache_capacity outright."
            )
        from tcip_mcp.pipelines.data.tiling import compute_stride

        stride = compute_stride(int(tile_size), float(overlap))
    rows_per_strip = max(1, int(rows_per_strip))
    band = math.ceil(int(tile_size) / rows_per_strip)
    carried = math.ceil(max(0, int(tile_size) - int(stride)) / rows_per_strip)
    return max(1, band + carried)


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


def _open_tiff(path: Path, num_channels: int) -> RasterSource:
    """The windowed strip backend when the file's layout supports it, else a whole decode.

    :class:`StripTiffSource` refuses every layout it cannot serve, so its own constructor is the
    layout probe and there is no second implementation of those rules to drift from it. A raster
    whose shape a whole decode would read channel-first goes whole as well, so ``load_multiband``
    and ``image_dimensions`` keep reporting the same frame for it.
    """
    try:
        source = StripTiffSource(path)
    except UnsupportedRasterLayout:
        return TiffWholeSource(path, num_channels)
    if source.height == num_channels and source.num_channels != num_channels:
        source.close()
        return TiffWholeSource(path, num_channels)
    return source


def open_raster(source: "str | Path | BandGroupRef", num_channels: int) -> RasterSource:
    """The backend that serves ``source`` at ``num_channels``.

    ``num_channels`` routes (which PIL mode a photograph decodes in, which axis order a numpy/TIFF
    array carries) and is never checked against the file: a 5-band GeoTIFF opened at 3 still reads
    as 5 bands. A photographic extension at a count PIL has no mode for raises ``ValueError``
    naming the containers that do carry band data.
    """
    if photographic_container(source, num_channels):
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
    exceeds this module's memory budget. It belongs to the process that filled it: a forked worker
    (a DataLoader's, say) finds it empty rather than reusing handles the parent owns.

    A caller must not close what this returns; the pool owns it. Nothing in the pipeline reads
    through the pool: the ``image_utils`` loaders open and close their own sources, so no caller's
    memory profile depends on it.
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
    budget = _memory_budget_bytes()
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
