"""Image serving: the one path pixels reach the browser through.

Every response here decodes through ``raster_source.open_raster``, so a photographic frame, a
grouped multi-band capture and a raster far too large to decode whole are all read the same way: a
rectangle at a requested resolution. A request with no ``bands`` selection serves the file's own
pixels as plain RGB; a selection (or a source with more bands than an RGB reading covers)
composites three of them through the shared display primitives in ``band_stats``.

A photographic frame is EXIF-oriented on the way out (``PhotographicSource``).
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

from tcip_mcp.pipelines.display_bounds import plan_read, planned_read, within_cap
from tcip_web import jobstore
from tcip_web.paths import allowed_file, resolved_image

router = APIRouter(prefix="/api/images", tags=["images"])


class ServingCell(BaseModel):
    """One read of a view: its name, the level it is read off (0 native, ``i`` the ``i``-th
    overview level), its half-open rect in that level's own pixel grid, and the native-pixel
    rect it covers (``nx0``..``ny1``), where a viewer places it."""

    name: str
    level: int
    x0: int
    y0: int
    x1: int
    y1: int
    nx0: float
    ny0: float
    nx1: float
    ny1: float


class ViewReads(BaseModel):
    """The reads one view of a raster is served by: the native region-serving cells it intersects,
    the tiles of the overview level its display read is planned off, or one display read of the
    view where the raster has no level to tile."""

    reads: list[ServingCell]


RENDER_CACHE_VERSION = 6
"""The render cache's shape version, part of every cache key and every image URL: bumped whenever
the cache key's inputs or the served headers' shape changes."""


@dataclass(frozen=True)
class _Encoding:
    """How one kind of serve is encoded: a PIL format, its save options, and the longest edge the
    codec accepts (``None`` where none was found). The cache-file suffix and the media type are
    the format's own name."""

    pil_format: str
    options: dict
    max_edge: "int | None"

    @property
    def suffix(self) -> str:
        return f".{self.pil_format.lower()}"

    @property
    def media_type(self) -> str:
        return f"image/{self.pil_format.lower()}"


_DISPLAY_ENCODING = _Encoding("JPEG", {"quality": 95, "subsampling": 0}, 65_500)
"""A display read, the whole frame, a region served scaled or a tile of an overview level: JPEG
at quality 95 with chroma subsampling off. Provisional, from one RGB orthomosaic (239921x141130,
one sensor, one site) read off its overviews at 4:4:4, as bytes and median encode time of three
runs taken under load: 1,006,644 bytes in 65 ms at quality 95 against 675,889 in 83 ms at 90 for a
1877x1104 view, 1,737,251 in 207 ms against 1,169,458 in 107 ms at 2503x1472, and 3,543,213 in
200 ms against 2,423,054 in 453 ms at 3754x2208. The edge limit is the widest single row PIL's JPEG
encoder accepted here: 65,500 encoded, 65,501 did not."""

_NATIVE_ENCODING = _Encoding("PNG", {"compress_level": 1}, None)
"""A region served at native resolution: lossless PNG at its fastest compression level, the
pixels a judgment is made on. Provisional, from three 2880x2880 native windows of that same
orthomosaic: level 1 took 775 to 986 ms and 25.8 to 26.0 MB, level 6 took 1902 to 1978 ms and
19.7 to 19.8 MB. PNG encoded a 10,000,000 pixel row here; no edge limit was found."""


def display_cap(display_pixels: int = Query(
        ..., ge=1, description="The requesting display's device-pixel count")) -> int:
    """The area cap one display-bound read for the requesting display is held to: that display's
    own device-pixel count."""
    return display_pixels


IMAGE_ERROR_HEADER = "X-TCIP-Image-Error"
"""The response header a refusal here names its condition through."""

OVERVIEWS_REQUIRED = "overviews_required"
"""The one condition this header carries: a read that needs a raster's overview pyramid and
has none."""

_CACHE_BUDGET_DIVISOR = 20
"""The rendered-variant cache's byte budget is the cache volume's free space divided by this: a
documented headroom choice, not a measured optimum."""

_cache_budget_bytes: "int | None" = None


def _cache_byte_budget(cache_dir: Path) -> int:
    """The cache's byte budget: free space on the cache volume over :data:`_CACHE_BUDGET_DIVISOR`,
    read once per process.
    """
    global _cache_budget_bytes
    if _cache_budget_bytes is None:
        import shutil

        _cache_budget_bytes = shutil.disk_usage(cache_dir).free // _CACHE_BUDGET_DIVISOR
    return _cache_budget_bytes


_STATS_SEED = 0
"""The seed every sampled read of a raster's display bounds uses.

Fixed, so two region requests against one raster stretch alike and a reported statistic is
reproducible across requests and processes.
"""

_STATS_WINDOW_SIZE = 256
"""Pixel window edge the stats sample reads in, passed to ``sample_windows`` as the edge of the
grid it draws windows from."""

_STATS_MAX_WINDOWS = 256
"""Windows the stats sample may read natively, about 16.8 million pixels at
:data:`_STATS_WINDOW_SIZE`. A GDAL-served raster with more pixels than these windows hold has its
statistics read off its overview pyramid instead; any other backend samples these windows
natively. Provisional: a documented bound on one native read's cost, not a measurement."""

_STATS_RESERVOIR_SIZE = 1 << 20
"""Pixels the percentile pass keeps, bounding what it holds to that many values per band in the
raster's own dtype (8 MB for a 4-band uint16 raster). A documented cap: a memory bound, not a
measured precision."""

_STATS_CACHE_MAX = 64
"""Rasters the per-raster stats cache describes at once. An entry is a few hundred bytes, so this
only stops a long session over many rasters from growing it without limit."""


@dataclass(frozen=True)
class _RasterStats:
    """One raster's display bounds, read once and reused by every region request against it, so two
    regions of one raster are never stretched differently.

    ``ranges`` and ``clip_bounds`` are per band, in band order. They come from one of two reads,
    and exactly one of these describes which: ``pixel_fraction`` is the share of the raster's
    pixels a seeded window sample covered (1.0 when the budget covered all of them, where the
    bounds are that raster's own exact bounds), and ``overview_size`` is the ``(width, height)`` a
    single reduced read of the whole frame was served at. Overview bounds are narrower than the
    raster's own and describe display scale only.

    ``interpretations`` names what each band holds where the backend knows (a GDAL raster's color
    interpretations: ``red``, ``alpha`` and the rest), and is ``None`` where nothing does, which is
    a different fact from a band whose interpretation is undefined.
    """

    dtype: str
    num_channels: int
    ranges: list
    clip_bounds: list
    seed: "int | None" = None
    pixel_fraction: "float | None" = None
    overview_size: "tuple[int, int] | None" = None
    interpretations: "tuple | None" = None

    @property
    def sampled(self) -> bool:
        """Whether these bounds describe part of the raster's pixels rather than all of them."""
        return self.pixel_fraction is not None and self.pixel_fraction < 1.0


_stats_cache: "OrderedDict[tuple, _RasterStats]" = OrderedDict()
_stats_lock = threading.Lock()


def _render_cache_dir() -> Path:
    """This machine's render cache, under the system temp directory: a cache of any served image,
    whichever project or root it lies under."""
    base = Path(tempfile.gettempdir()) / "tcip-img-cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


# Per-directory (known total at last walk, bytes written since), so a miss burst on a
# large cache pays a per-file directory walk only when the budget could have been crossed.
_cache_accounting: dict[str, tuple[int, int]] = {}


def _note_cache_write(cache_dir: Path, nbytes: int) -> None:
    key = str(cache_dir)
    known, unaccounted = _cache_accounting.get(key, (0, 0))
    _cache_accounting[key] = (known, unaccounted + nbytes)


def _evict_lru(cache_dir: Path) -> None:
    """Drop least recently used rendered variants until the cache fits its byte budget,
    each with the headers stored beside it. Walks the directory only when writes since
    the last accounting could have crossed the budget."""
    try:
        budget = _cache_byte_budget(cache_dir)
        key = str(cache_dir)
        accounted = _cache_accounting.get(key)
        if accounted is not None and sum(accounted) <= budget:
            return
        entries: list[tuple[float, Path, int]] = []
        total = 0
        for p in cache_dir.iterdir():
            if not (p.is_file()
                    and p.suffix in (_DISPLAY_ENCODING.suffix, _NATIVE_ENCODING.suffix)):
                continue
            st = p.stat()
            size = st.st_size
            try:
                size += p.with_suffix(".json").stat().st_size
            except OSError:
                pass
            entries.append((st.st_mtime, p, size))
            total += size
        if total > budget:
            entries.sort(key=lambda e: e[0])
            for _mtime, p, size in entries:
                p.unlink()
                p.with_suffix(".json").unlink(missing_ok=True)
                total -= size
                if total <= budget:
                    break
        _cache_accounting[key] = (total, 0)
    except OSError:
        pass


@router.get("/view")
def get_view_reads(
    path: str = Query(..., description="Absolute path to the image file"),
    x0: int = Query(..., ge=0, description="View left edge, full-resolution pixels"),
    y0: int = Query(..., ge=0, description="View top edge, full-resolution pixels"),
    x1: int = Query(..., ge=0, description="View right edge, exclusive"),
    y1: int = Query(..., ge=0, description="View bottom edge, exclusive"),
    area_cap: int = Depends(display_cap),
) -> ViewReads:
    """The reads that serve the view ``[x0, x1) x [y0, y1)`` of the raster at ``path``, clipped
    to its extent, measured off the header for a TIFF or a photograph and by loading the array
    for a numpy container (``image_dimensions``).

    A view of at most the display's area cap is served by the native region-serving cells it
    intersects. A larger view is planned as one display read (``display_bounds.plan_read``); when
    the plan comes off an overview level, the view is served by that level's own region-serving
    cells its window on the level intersects, each a read of one tile of that level at the
    level's resolution, so a pan fetches only the tiles that entered the view. A raster with no
    level to read off is served by one display read of the view itself, named ``view``. Cells
    are sized by :func:`~tcip_mcp.pipelines.reference_grid.derive_serving_tile_size` under the
    same cap and the edge limit of the encoding each is served in. The frame and the levels are
    the ones the image route's own plain read opens the raster at
    (``raster_source.image_route_channel_count``, read once for both). Refuses a view with no
    pixels inside the raster.
    """
    from tcip_mcp.pipelines.image_utils import image_dimensions
    from tcip_mcp.pipelines.raster_source import (
        Rect,
        image_route_channel_count,
        level_dims,
        level_window,
        rects_overlap,
    )
    from tcip_mcp.pipelines.reference_grid import derive_serving_tile_size, reference_cells

    source = resolved_image(path)[1]
    open_channels = image_route_channel_count(source)
    width, height = image_dimensions(source, open_channels)
    x1, y1 = min(x1, width), min(y1, height)
    if not (x0 < x1 and y0 < y1):
        raise HTTPException(400, f"view [{y0}:{y1}, {x0}:{x1}] holds no pixel of this "
                                 f"{width}x{height} raster")
    view = Rect(x0, y0, x1, y1)

    def cells(level: int, level_w: int, level_h: int, window: Rect) -> ViewReads:
        sx, sy = width / level_w, height / level_h
        encoding = _DISPLAY_ENCODING if level else _NATIVE_ENCODING
        edge = derive_serving_tile_size(level_w, level_h, area_cap, encoding.max_edge)
        return ViewReads(reads=[
            ServingCell(name=c.name, level=level, x0=c.x0, y0=c.y0, x1=c.x1, y1=c.y1,
                        nx0=c.x0 * sx, ny0=c.y0 * sy, nx1=c.x1 * sx, ny1=c.y1 * sy)
            for c in reference_cells(level_w, level_h, edge, 0.0, clamp=True)
            if rects_overlap((c.x0, c.y0, c.x1, c.y1),
                             (window.x0, window.y0, window.x1, window.y1))])

    if within_cap(view.width, view.height, area_cap):
        return cells(0, width, height, view)
    dims = level_dims(source, open_channels)
    plan = plan_read(view, width, height, dims, area_cap)
    if plan.level == 0:
        return ViewReads(reads=[ServingCell(name="view", level=0, x0=x0, y0=y0, x1=x1, y1=y1,
                                            nx0=x0, ny0=y0, nx1=x1, ny1=y1)])
    level_w, level_h = dims[plan.level - 1]
    return cells(plan.level, level_w, level_h,
                 level_window(view, width, height, level_w, level_h))


def _parse_band_tokens(raw: str) -> list[str]:
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if len(tokens) != 3:
        raise HTTPException(
            400, f"bands must name exactly 3 bands (R,G,B), got {len(tokens)}: {raw!r}"
        )
    return tokens


def _band_index(token: str, declared_names: "list[str] | None", total_bands: int) -> int:
    if declared_names and token in declared_names:
        return declared_names.index(token)
    try:
        idx = int(token)
    except ValueError:
        raise HTTPException(
            400,
            f"band {token!r} is not a declared band name"
            + (f" ({declared_names})" if declared_names else "")
            + " and not a valid 0-based index",
        ) from None
    if not (0 <= idx < total_bands):
        raise HTTPException(400, f"band index {idx} out of range for a {total_bands}-band image")
    return idx


def _sidecar_identity(source) -> "tuple[int, int] | None":
    """The overview sidecar's ``(mtime_ns, size)`` for a GDAL-readable raster, or ``None`` when it
    has none. Part of the render key.
    """
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.overviews import overview_sidecar

    if isinstance(source, BandGroupRef):
        return None
    if Path(source).suffix.lower() not in (".tif", ".tiff"):
        return None
    try:
        st = overview_sidecar(source).stat()
    except OSError:
        return None
    return int(st.st_mtime_ns), int(st.st_size)


def _overviews_required(path: str, reason: str) -> HTTPException:
    """The one refusal for a read that needs a raster's overview pyramid and has none, carrying the
    condition as a header and, after ``reason``, the request that builds the pyramid.
    """
    return HTTPException(
        400,
        f"{reason} Reading it needs its reduced-resolution overviews: build them with POST "
        f'/api/images/overviews {{"path": "{path}"}}, then ask again.',
        headers={IMAGE_ERROR_HEADER: OVERVIEWS_REQUIRED})


def _overview_stats(raster, dims: list[tuple[int, int]]):
    """Per-band ranges, clip cut points, and the planned ``(width, height)`` they were read at,
    from one reduced read of ``raster``'s whole frame, planned within
    ``overviews.PYRAMID_FLOOR_EDGE`` on each edge over the overview levels ``dims``: the same
    display primitives the sampled path reads, over pixels an overview level served instead of
    over native windows."""
    from tcip_mcp.pipelines.band_stats import band_ranges, clip_bounds
    from tcip_mcp.pipelines.overviews import PYRAMID_FLOOR_EDGE
    from tcip_mcp.pipelines.raster_source import Rect

    whole = Rect(0, 0, raster.width, raster.height)
    plan = plan_read(whole, raster.width, raster.height, dims, PYRAMID_FLOOR_EDGE**2,
                     PYRAMID_FLOOR_EDGE)
    pixels, _spec = planned_read(raster, whole, plan)
    clips = [clip_bounds(pixels[:, :, i]) for i in range(pixels.shape[-1])]
    return band_ranges(pixels), clips, (plan.width, plan.height)


def _source_identity(source, num_channels: int) -> tuple:
    """What identifies a raster to everything cached about it here: the identity the raster layer
    pools an open source under, plus its overview sidecar's, since a pyramid appearing changes
    which pixels a read of it returns."""
    from tcip_mcp.pipelines import raster_source

    return (raster_source.source_pool_key(source, num_channels), _sidecar_identity(source))


def _raster_stats(source, num_channels: int) -> _RasterStats:
    """``source``'s per-band display bounds, one set per raster: the bounds a region stretch uses
    and ``/api/images/bands`` reports.

    A raster the :data:`_STATS_MAX_WINDOWS` windows can hold is read from native pixels by the
    seeded window sample (covering all of them, and so exact, when its grid fits), and past it a
    single reduced read of the whole frame comes off an overview level. A GDAL-backed raster past
    it with no overview levels is refused; every other backend keeps reading native windows.

    A concurrent miss on the same raster computes twice and stores the same numbers.
    """
    key = _source_identity(source, num_channels)
    sample_budget = _STATS_MAX_WINDOWS * _STATS_WINDOW_SIZE**2
    with _stats_lock:
        hit = _stats_cache.get(key)
        if hit is not None:
            _stats_cache.move_to_end(key)
            return hit

    from tcip_mcp.pipelines import raster_source
    from tcip_mcp.pipelines.band_stats import sampled_band_ranges
    from tcip_mcp.pipelines.image_utils import source_path_of

    with raster_source.open_raster(source, num_channels) as raster:
        dtype = str(raster.dtype)
        channels = int(raster.num_channels)
        # Only a backend that reads them from the file exposes these; nothing here infers them.
        interpretations = getattr(raster, "band_interpretations", None)
        oversized = raster.width * raster.height > sample_budget
        if oversized and isinstance(raster, raster_source.GdalSource):
            dims = raster.level_dims()
            if not dims:
                raise _overviews_required(
                    str(raster.path),
                    f"this {raster.width}x{raster.height} raster holds "
                    f"{raster.width * raster.height} pixels, over the {sample_budget} its "
                    "display statistics can be read from native pixels for.")
            ranges, clips, served = _overview_stats(raster, dims)
            stats = _RasterStats(dtype=dtype, num_channels=channels, ranges=ranges,
                                 clip_bounds=clips, overview_size=served,
                                 interpretations=interpretations)
        else:
            sampled = sampled_band_ranges(
                raster, label=source_path_of(source), seed=_STATS_SEED,
                window_size=_STATS_WINDOW_SIZE, max_windows=_STATS_MAX_WINDOWS,
                reservoir_size=_STATS_RESERVOIR_SIZE)
            stats = _RasterStats(
                dtype=dtype, num_channels=channels, ranges=list(sampled.ranges),
                clip_bounds=list(sampled.clip_bounds), seed=sampled.sampling.seed,
                pixel_fraction=sampled.sampling.pixel_fraction,
                interpretations=interpretations)
    with _stats_lock:
        _stats_cache[key] = stats
        _stats_cache.move_to_end(key)
        while len(_stats_cache) > _STATS_CACHE_MAX:
            _stats_cache.popitem(last=False)
    return stats


def _sampled_bounds(stats: _RasterStats, idxs: "list[int]", stretch: str
                    ) -> "list[tuple[float, float]]":
    """The ``(low, high)`` pair per selected band a region render stretches between, in the same
    order as ``idxs``: the clip cut points for ``percent_clip``, the sampled range otherwise
    (``none`` reads both ends, as the sampled minimum and maximum a float raster's divisor comes
    from, and an integer raster ignores the pair for its dtype ceiling)."""
    if stretch == "percent_clip":
        return [stats.clip_bounds[i] for i in idxs]
    return [(stats.ranges[i].minimum, stats.ranges[i].maximum) for i in idxs]


def _plain_rgb(pixels, dtype, bounds: "tuple[float, float] | None"):
    """A 1/3/4-band array as the plain ``uint8`` RGB the file's own pixels read as: a single band
    replicated, a fourth band dropped, and no data-range stretch applied.

    A non-``uint8`` raster is scaled by ``band_stats.full_scale_denominator`` (the ``none``
    stretch), one denominator for the whole array. ``bounds`` carries the sampled ``(minimum,
    maximum)`` a float raster's denominator comes from when the pixels in hand are one region of
    it. Returns the rendered pixels.
    """
    import numpy as np

    from tcip_mcp.pipelines.band_stats import full_scale_denominator, stretch_band

    arr = np.asarray(pixels)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    divisor = full_scale_denominator(
        arr, dtype, sampled_maximum=None if bounds is None else bounds[1],
        sampled_minimum=None if bounds is None else bounds[0])
    return stretch_band(arr, "none", dtype, (0.0, divisor))


@router.get("")
def serve_image(
    request: Request,
    path: str = Query(..., description="Absolute path to the image file"),
    area_cap: int = Depends(display_cap),
    bands: str | None = Query(
        None, description="3 comma-separated band names or 0-based indices, e.g. "
                          "'NIR,Red,Green' or '3,2,1'; selects a live composite instead of the "
                          "file's own pixels as-is."),
    stretch: str = Query(
        "minmax", description="minmax|percent_clip|none; applied only when compositing bands "
                              "(bands given, or path names a .bandgroup-grouped capture)."),
    x0: int | None = Query(None, ge=0, description="Region left edge, full-resolution pixels"),
    y0: int | None = Query(None, ge=0, description="Region top edge, full-resolution pixels"),
    x1: int | None = Query(None, ge=0, description="Region right edge, exclusive"),
    y1: int | None = Query(None, ge=0, description="Region bottom edge, exclusive"),
    level: int = Query(0, ge=0, description="The overview level the region is a tile of (0 "
                                            "native), its corners in that level's pixel grid"),
) -> Response:
    """Serve a raster, whole or one region of it, within the requesting display's area cap.

    ``x0/y0/x1/y1`` (all four or none) name a half-open region in the raster's own full-resolution
    pixel grid; omitting them serves the whole frame. A region of at most the cap's pixels is
    served at native resolution, losslessly (:data:`_NATIVE_ENCODING`). The whole frame, and a
    region past the cap, is a display read (:data:`_DISPLAY_ENCODING`) as
    ``display_bounds.plan_read`` plans it, within the codec's edge limit. With ``level`` above 0
    the region is one tile of that overview level, in its own pixel grid and at most the cap's
    pixels, served at the level's resolution as a display read; a raster with no such level
    refuses it.

    With no ``bands`` selection a 1/3/4-band raster serves as its own plain RGB, with no data-range
    stretch: a single band replicated, a fourth band dropped. A ``bands`` selection, a
    ``.bandgroup``-grouped capture, or any other band count composites three bands through the
    shared display stretch instead. Whole-view stretch bounds come from the served pixels
    themselves; a region's come from the raster's seeded per-band sample, so two regions of one
    raster render alike. The size served is reported in ``X-TCIP-Served-Size``.

    A display read of a GDAL raster whose window is larger than the cap, with no overview levels,
    is refused naming ``POST /api/images/overviews`` (``X-TCIP-Image-Error: overviews_required``)
    when a pyramid can be built for it (``overviews.overview_levels``); a raster within the
    pyramid floor, which can have none, is read natively and resampled.

    An ETag keyed on the file's identity and every requested render param lets the browser
    revalidate with a cheap 304.
    """
    import numpy as np
    from PIL import Image

    from tcip_mcp.pipelines import raster_source
    from tcip_mcp.pipelines.band_stats import (
        STRETCH_MODES,
        band_ranges,
        composite_display_rgb,
    )
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.derivations import probe_channels
    from tcip_mcp.pipelines.overviews import overview_levels

    if stretch not in STRETCH_MODES:
        raise HTTPException(400, f"stretch must be one of {sorted(STRETCH_MODES)}, got {stretch!r}")
    src, source = resolved_image(path)

    corners = (x0, y0, x1, y1)
    if any(c is None for c in corners) and any(c is not None for c in corners):
        raise HTTPException(400, "a region needs all four of x0, y0, x1, y1, or none of them")
    whole_view = corners[0] is None
    native = False
    if not whole_view:
        # the all-or-none guard above already requires every corner set when not whole_view
        assert x0 is not None and y0 is not None and x1 is not None and y1 is not None
        if not (x0 < x1 and y0 < y1):
            raise HTTPException(
                400, f"region [{y0}:{y1}, {x0}:{x1}] is empty; x0 < x1 and y0 < y1 are required")
        native = within_cap(x1 - x0, y1 - y0, area_cap)
    if level and not (native and x0 is not None and y0 is not None and x1 is not None
                      and y1 is not None and _DISPLAY_ENCODING.max_edge is not None
                      and max(x1 - x0, y1 - y0) <= _DISPLAY_ENCODING.max_edge):
        raise HTTPException(
            400, f"a tile of overview level {level} is a region of at most this display's cap "
                 f"of {area_cap} pixels and no edge past {_DISPLAY_ENCODING.max_edge}")

    band_tokens = _parse_band_tokens(bands) if bands is not None else None
    composite_requested = bands is not None or isinstance(source, BandGroupRef)
    try:
        probed = probe_channels(source)
    except Exception as exc:
        # A truncated or otherwise unreadable file fails its header probe here, before any
        # raster is opened: the request's own fault, answered as one.
        raise HTTPException(400, f"could not open this image: {exc}") from exc
    open_channels = (
        probed if composite_requested
        else raster_source.image_route_channel_count(source, probed)
    )
    encoding = _NATIVE_ENCODING if native and not level else _DISPLAY_ENCODING

    # Requested params only: the scale a read is served at depends on the raster's own size and on
    # whether an overview level exists, neither known at lookup time, so it returns as a header.
    key = hashlib.md5(
        f"{RENDER_CACHE_VERSION}:{_source_identity(source, open_channels)}:{area_cap}:"
        f"{encoding.pil_format}:{sorted(encoding.options.items())}:{bands}:{stretch}:{corners}:"
        f"{level}".encode()
    ).hexdigest()
    etag = f'W/"{key}"'
    cache_headers = {"ETag": etag, "Cache-Control": "private, max-age=3600"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=cache_headers)

    cache_dir = _render_cache_dir()
    cached = cache_dir / f"{key}{encoding.suffix}"
    cached_headers = cache_dir / f"{key}.json"
    if cached.is_file() and cached_headers.is_file():
        try:
            extra = json.loads(cached_headers.read_text(encoding="utf-8"))
            cached.touch()  # refresh mtime so LRU eviction keeps the working set
            return FileResponse(cached, media_type=encoding.media_type,
                                headers={**cache_headers, **extra})
        except (OSError, ValueError):
            pass  # an unreadable cache entry is rendered again, never served as-is

    opening = True
    try:
        with raster_source.open_raster(source, open_channels) as raster:
            opening = False
            dims = raster.level_dims()
            grid_w, grid_h = dims[level - 1] if 0 < level <= len(dims) else (
                raster.width, raster.height)
            if level and (grid_w, grid_h) == (raster.width, raster.height):
                raise HTTPException(400, f"this raster has no overview level {level}")
            if whole_view:
                rect = raster_source.Rect(0, 0, raster.width, raster.height)
            else:
                # the all-or-none guard above already requires every corner set when not whole_view
                assert x0 is not None and y0 is not None and x1 is not None and y1 is not None
                rect = raster_source.Rect(x0, y0, x1, y1)
            if not whole_view and (rect.x1 > grid_w or rect.y1 > grid_h):
                raise HTTPException(
                    400,
                    f"region [{rect.y0}:{rect.y1}, {rect.x0}:{rect.x1}] is outside this "
                    f"{grid_w}x{grid_h} raster",
                )
            if level:
                # only a GDAL raster reports overview levels, so a level read has one
                assert isinstance(raster, raster_source.GdalSource)
                pixels, _spec = raster.read_level_region(level, rect)
                served_size = (rect.width, rect.height)
            else:
                plan = plan_read(rect, raster.width, raster.height, dims, area_cap,
                                 encoding.max_edge)
                # A raster no pyramid can be built for is at most the floor edge square: read
                # natively.
                if (not plan.fits and plan.level == 0
                        and isinstance(raster, raster_source.GdalSource)
                        and overview_levels(raster.width, raster.height)):
                    raise _overviews_required(
                        path,
                        f"serving this {rect.width}x{rect.height} read at {plan.width}x"
                        f"{plan.height} would decode {rect.width * rect.height} pixels natively, "
                        f"over this display's cap of {area_cap}.")
                pixels, _spec = planned_read(raster, rect, plan)
                served_size = (plan.width, plan.height)
            channels = int(raster.num_channels)
            dtype = raster.dtype

        composite = composite_requested or channels not in (1, 3, 4)
        declared_names = list(source.bands) if isinstance(source, BandGroupRef) else None
        if band_tokens is None:
            idxs = [min(i, channels - 1) for i in range(3)]
        else:
            idxs = [_band_index(t, declared_names, channels) for t in band_tokens]

        # A region's bounds come from the raster's own sample, a whole view's from the pixels it
        # served; a scale that reads no pixel statistic (a dtype ceiling) asks for neither.
        integer = np.issubdtype(dtype, np.integer)
        wants_bounds = (not (stretch == "none" and integer)) if composite else not integer
        sampled = (_raster_stats(source, open_channels)
                   if wants_bounds and not whole_view else None)

        if composite and stretch == "none" and integer:
            rgb = composite_display_rgb(pixels, idxs, stretch, None)
        elif composite:
            rgb = composite_display_rgb(pixels, idxs, stretch, (
                None if sampled is None else _sampled_bounds(sampled, idxs, stretch)))
        elif integer:
            rgb = _plain_rgb(pixels, dtype, None)
        else:
            # Only the bands a plain serve displays: a fourth band is dropped before the viewer
            # sees it, so its level must not set the scale the other three are divided by.
            ranges = band_ranges(pixels) if sampled is None else sampled.ranges
            band_bounds = (
                min(ranges[i].minimum for i in idxs), max(ranges[i].maximum for i in idxs))
            rgb = _plain_rgb(pixels, dtype, band_bounds)

        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgb), mode="RGB").save(
            buf, encoding.pil_format, **encoding.options)
        data = buf.getvalue()
    except HTTPException:
        raise
    except ValueError as exc:
        what = ("open this image" if opening
                else "read this image" if whole_view else "read this region")
        raise HTTPException(400, f"could not {what}: {exc}") from exc
    except Exception as exc:
        raise HTTPException(500, f"could not process image: {exc}") from exc

    extra = {"X-TCIP-Served-Size": f"{served_size[0]}x{served_size[1]}"}

    try:
        tmp = cache_dir / f"{key}.{threading.get_ident()}.tmp"
        tmp.write_bytes(data)
        tmp.replace(cached)
        cached_headers.write_text(json.dumps(extra), encoding="utf-8")
        _note_cache_write(cache_dir, len(data) + cached_headers.stat().st_size)
        _evict_lru(cache_dir)
    except OSError:
        pass  # cache is best-effort; the response below is already rendered

    return Response(content=data, media_type=encoding.media_type,
                    headers={**cache_headers, **extra})


@router.get("/bands")
def get_bands(path: str = Query(...)) -> dict:
    """Band count + per-band stats for ``path``: the picker's symbology data, and the one fact
    (``band_count > 3``) the frontend uses to decide whether to show the picker at all.

    ``path`` may be a plain raster or a ``.bandgroup`` manifest naming a grouped multi-band
    capture. ``band_count`` comes from the channel probe (``probe_channels``: the header for a
    TIFF or a photograph, the loaded array for a numpy container). A plain (non-grouped) raster
    at ``band_count <= 3`` reads nothing past that probe; a ``.bandgroup``-grouped capture always
    gets the full per-band stats, even at exactly 3 bands.

    The response says which read produced the stats. A raster whose pixels fit the
    native-sampling budget, and any raster a backend decodes whole at open, carries ``sampled``,
    ``pixel_fraction`` and ``seed``: ``sampled`` is false exactly when the sample covered every
    pixel (``pixel_fraction`` 1.0), where the numbers are the raster's own exact bounds. A
    larger GDAL-served one is read once off an overview level instead and carries ``sampled``
    false with ``overview_size``, the ``[width, height]`` it was read at; those bounds describe
    display scale only. Either way they are the same numbers this raster's region renders
    stretch through.

    A band carries ``interpretation`` (``red``, ``alpha``, and the rest) where the backend reads it
    from the file; the key is absent where nothing knows.
    """
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.derivations import probe_channels

    source = resolved_image(path)[1]
    n = probe_channels(source)
    if n <= 3 and not isinstance(source, BandGroupRef):
        return {"band_count": n, "bands": []}

    stats = _raster_stats(source, n)
    if isinstance(source, BandGroupRef):
        names = list(source.bands)
        wavelengths = source.central_wavelength_nm or {}
    else:
        names = [str(i) for i in range(stats.num_channels)]
        wavelengths = {}

    bands = []
    for i, name in enumerate(names):
        band = {
            "name": name,
            "wavelength_nm": wavelengths.get(name),
            "dtype": stats.dtype,
            "min": stats.ranges[i].minimum,
            "max": stats.ranges[i].maximum,
        }
        if stats.interpretations is not None and i < len(stats.interpretations):
            band["interpretation"] = stats.interpretations[i]
        bands.append(band)
    if stats.overview_size is not None:
        return {"band_count": len(names), "bands": bands, "sampled": False,
                "overview_size": list(stats.overview_size)}
    return {
        "band_count": len(names),
        "bands": bands,
        "sampled": stats.sampled,
        "pixel_fraction": stats.pixel_fraction,
        "seed": stats.seed,
    }


# ── Overview builds ─────────────────────────────────────────────────────


@dataclass
class OverviewJob:
    """One raster's overview build, running on a background thread."""

    job_id: str
    path: str
    status: str = "pending"  # pending | running | completed | failed
    progress: float = 0.0
    error: "str | None" = None


_overview_registry = jobstore.JobRegistry()
"""The live registry for overview-build jobs (``jobstore.JobRegistry``), with no root concept."""


class OverviewBuildPayload(BaseModel):
    path: str


def _overview_summary(job: OverviewJob) -> dict:
    return {"job_id": job.job_id, "path": job.path, "status": job.status,
            "progress": job.progress, "error": job.error}


def _overview_worker(job: OverviewJob) -> None:
    """Build the pyramid, recording progress and whatever stopped it.

    A refusal to rebuild over a pyramid that already exists is a completed outcome; anything else
    that stops the build is a failure. A raster GDAL cannot open at all answers that there is no
    pyramid.
    """
    from tcip_mcp.pipelines.overviews import build_overviews, overview_dims

    def record(fraction: float) -> None:
        job.progress = float(fraction)

    job.status = "running"
    try:
        build_overviews(job.path, progress_cb=record)
    except Exception as exc:  # noqa: BLE001, the job records what stopped it
        try:
            built = bool(overview_dims(job.path))
        except Exception:  # noqa: BLE001, a raster that will not open carries no pyramid
            built = False
        if not built:
            job.status = "failed"
            job.error = str(exc)
            return
    job.status = "completed"
    job.progress = 1.0


@router.post("/overviews")
def build_image_overviews(payload: OverviewBuildPayload) -> dict:
    """Start building ``path``'s reduced-resolution overview pyramid, the sidecar a scaled read of
    a raster larger than the display cap is served from.

    One build per raster: a request naming a path a build is already running for joins that job
    rather than starting a second one over the same sidecar. Poll ``/overviews/status``.
    """
    src = allowed_file(payload.path)
    job, created = _overview_registry.find_or_register(
        lambda existing: existing.path == str(src) and jobstore.live(existing),
        lambda: OverviewJob(job_id=f"ovr-{uuid.uuid4().hex[:8]}", path=str(src)),
    )
    if created:
        threading.Thread(target=_overview_worker, args=(job,), daemon=True).start()
    return _overview_summary(job)


@router.get("/overviews/status")
def get_overview_job(job_id: str = Query(...)) -> dict:
    """An overview build's status and completion fraction."""
    job = _overview_registry.get(job_id)
    if job is None:
        raise HTTPException(404, f"job not found: {job_id}")
    return _overview_summary(job)
