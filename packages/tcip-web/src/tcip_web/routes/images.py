"""Image serving: the one path pixels reach the browser through.

Every response here decodes through ``raster_source.open_raster``, so a photographic frame, a
grouped multi-band capture and a raster far too large to decode whole are all read the same way:
a rectangle at a requested resolution. A request with no ``bands`` selection serves the file's own
pixels as plain RGB; a selection (or a source with more bands than an RGB reading covers)
composites three of them through the shared display primitives in ``band_stats``, never through a
second stretch implementation here.

The browser must receive pixels in the same frame the annotations were authored against, so a
photographic frame is EXIF-oriented on the way out (``PhotographicSource``, which
``get_image_dimensions`` measures the same way).
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import tempfile
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

from tcip_mcp.pipelines.display_bounds import DISPLAY_MAX_EDGE, DISPLAY_MAX_PIXELS
from tcip_web import jobstore
from tcip_web.paths import assert_path_allowed
from tcip_web.routes._coverage_models import StatsSource

router = APIRouter(prefix="/api/images", tags=["images"])

RENDER_CACHE_VERSION = 2
"""Bumped whenever the render cache key's inputs or the served headers' shape changes, so a
warm entry written under the old shape is served by neither the disk cache nor conditional
revalidation. Read by ``scripts/generate_frontend_types.py`` and carried on every image URL the
browser builds (``api.images.url``), so a browser cache entry from before the bump is never the
response to a request built after it either.
"""

IMAGE_ERROR_HEADER = "X-TCIP-Image-Error"
"""The response header a refusal here names its condition through, since a DOM ``Image`` sees
only that a load failed. Read by ``scripts/generate_frontend_types.py``."""

OVERVIEWS_REQUIRED = "overviews_required"
"""The one condition this header carries: a read that needs a raster's overview pyramid and
has none. Read by ``scripts/generate_frontend_types.py``."""

_CACHE_BUDGET_DIVISOR = 20
"""The rendered-variant cache's byte budget is the cache volume's free space divided by this.

The decode, orient, resize and encode pipeline costs seconds of CPU per large frame and the
browser cache only covers one session, so a cold request (fresh session, refresh, prefetch) is a
sendfile instead of a re-render. Cell-aligned region serving stores one entry per cell view
rather than a handful per raster, so a fixed file count no longer tracks what the cache costs;
bytes do. A twentieth of free space keeps the cache an order of magnitude away from ever filling
the volume while still holding thousands of cell-sized JPEGs on any workstation disk; the
divisor is that headroom rationale, not a measured optimum.
"""

_cache_budget_bytes: "int | None" = None


def _cache_byte_budget(cache_dir: Path) -> int:
    """The cache's byte budget: free space on the cache volume over
    :data:`_CACHE_BUDGET_DIVISOR`, read once per process so eviction pressure cannot
    ratchet the budget down as the cache itself consumes space."""
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
"""Pixel window edge the stats sample reads in: ``sample_windows``' own grid-cell size."""

_STATS_MAX_WINDOWS = DISPLAY_MAX_PIXELS // (_STATS_WINDOW_SIZE**2)
"""Windows the stats sample may read, so its pixel budget is at most :data:`DISPLAY_MAX_PIXELS`.

That ties what describing a raster costs to the same bound that caps what is served from it.
"""

_STATS_RESERVOIR_SIZE = 1 << 20
"""Pixels the percentile pass keeps, bounding what it holds to that many values per band in the
raster's own dtype (8 MB for a 4-band uint16 raster). A documented cap: a memory bound, not a
measured precision."""

_STATS_SAMPLE_BUDGET = _STATS_MAX_WINDOWS * _STATS_WINDOW_SIZE**2
"""Full-resolution pixels a raster may hold before its display stats are read from an overview
level instead of from native windows.

The window sample is cheap in pixels and expensive in reads: its square windows scatter across a
raster stored as full-width strips, so each one decodes whole strips to keep a 256-pixel square.
Past this many pixels that cost stops tracking the sample's size, and a single reduced read of the
whole frame describes the raster for far less work.
"""

_STATS_OVERVIEW_MAX_EDGE = 1024
"""Longest output edge the display stats of an oversized raster are read at.

Below the deepest overview level any pyramid built here carries (levels stop at the first whose
longest edge fits ``DISPLAY_MAX_EDGE``, so the deepest level's longest edge is over half of it),
which is what makes the read come off that level rather than off native pixels.
"""

_STATS_CACHE_MAX = 64
"""Rasters the per-raster stats cache describes at once. An entry is a few hundred bytes, so this
only stops a long session over many rasters from growing it without limit."""


@dataclass(frozen=True)
class _RasterStats:
    """One raster's display bounds, read once and reused by every region request against it, so
    two regions of one raster are never stretched differently.

    ``ranges`` and ``clip_bounds`` are per band, in band order. They come from one of two reads,
    and exactly one of these describes which: ``pixel_fraction`` is the share of the raster's
    pixels a seeded window sample covered (1.0 when the budget covered all of them, where the
    bounds are that raster's own exact bounds), and ``overview_scale`` is the served/native
    resolution ratio a single reduced read of the whole frame was taken at. Overview pixels are
    averages of native ones, so bounds read from them are narrower than the raster's own: they
    describe what a viewer sees at display scale, which is all a display stretch asks of them, and
    nothing else may be derived from them.

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
    overview_scale: "float | None" = None
    interpretations: "tuple | None" = None

    @property
    def sampled(self) -> bool:
        """Whether these bounds describe part of the raster's pixels rather than all of them."""
        return self.pixel_fraction is not None and self.pixel_fraction < 1.0

    def stats_source(self) -> StatsSource:
        """The structured ``StatsSource`` a response reports these bounds under."""
        if self.overview_scale is not None:
            return StatsSource(read="overview", overview_scale=self.overview_scale)
        return StatsSource(read="window_sample", seed=self.seed, pixel_fraction=self.pixel_fraction)


_stats_cache: "OrderedDict[tuple, _RasterStats]" = OrderedDict()
_stats_lock = threading.Lock()


def _render_cache_dir() -> Path:
    from tcip_mcp.project_paths import resolve_state_or

    base = resolve_state_or(
        Path(".tcip") / "cache" / "img", Path(tempfile.gettempdir()) / "tcip-img-cache"
    )
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
            if not (p.is_file() and p.suffix == ".jpg"):
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


def _checked(path: str) -> Path:
    """Resolve + allow-list check an absolute client-supplied image path."""
    try:
        src = assert_path_allowed(path)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc
    if not src.is_file():
        raise HTTPException(404, f"not a file: {path}")
    return src


def _parse_band_tokens(raw: str) -> list[str]:
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if len(tokens) != 3:
        raise HTTPException(400, f"bands must name exactly 3 bands (R,G,B), got {len(tokens)}: {raw!r}")
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
    has none.

    Part of the render key: overview-served pixels are not the pixels a native read resamples to
    the same size, so a build that completes between two requests has to invalidate what the first
    one cached. Read for every GDAL container rather than only for the reads that turn out to be
    scaled, since which reads those are is not known before the raster is open; the cost of the
    wider rule is a re-render of a native read after a build, never a stale one.
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


def _overviews_required(path: str, detail: str) -> HTTPException:
    """The one refusal for a read that needs a raster's overview pyramid and has none.

    Carries the condition as a header because a DOM ``Image`` sees only that a load failed, and
    names the endpoint that builds the pyramid in the detail for whoever reads that instead.
    """
    return HTTPException(400, detail, headers={IMAGE_ERROR_HEADER: OVERVIEWS_REQUIRED})


def _reduced_reads_available(raster) -> bool:
    """Whether ``raster`` can serve a reduced-resolution read off an overview level instead of by
    decoding native pixels.

    Read from headers, never by reading pixels: GDAL reports the overview levels it can see on
    open (``overviews.has_overviews``), and an external sidecar whose tiles were never written is
    one of those levels while reading back as silent zeros, so it counts only once
    ``overviews.sidecar_valid`` has confirmed it. Only a GDAL-backed raster has such levels; every
    other backend decodes native pixels whatever resolution is asked of it.
    """
    from tcip_mcp.pipelines.overviews import has_overviews, overview_sidecar, sidecar_valid
    from tcip_mcp.pipelines.raster_source import GdalSource

    if not isinstance(raster, GdalSource):
        return False
    if not has_overviews(raster.path):
        return False
    return not overview_sidecar(raster.path).is_file() or sidecar_valid(raster.path)


def _overview_stats_target(width: int, height: int) -> tuple[int, int]:
    """The aspect-preserving output size a raster's display stats are read at."""
    out_w = max(1, round(width * _STATS_OVERVIEW_MAX_EDGE / max(width, height)))
    return out_w, max(1, round(height * out_w / width))


def _overview_stats(raster):
    """Per-band ranges, clip cut points, and the read's own spec, from one reduced read of
    ``raster``'s whole frame: the same display primitives the sampled path reads, over pixels an
    overview level served instead of over native windows."""
    from tcip_mcp.pipelines.band_stats import band_ranges, clip_bounds
    from tcip_mcp.pipelines.raster_source import Rect

    pixels, spec = raster.read_region(
        Rect(0, 0, raster.width, raster.height),
        target_size=_overview_stats_target(raster.width, raster.height))
    clips = [clip_bounds(pixels[:, :, i]) for i in range(pixels.shape[-1])]
    return band_ranges(pixels), clips, spec


def _source_identity(source, num_channels: int) -> tuple:
    """What identifies a raster to everything cached about it here: the identity the raster layer
    pools an open source under, plus its overview sidecar's, since a pyramid appearing changes
    which pixels a read of it returns."""
    from tcip_mcp.pipelines import raster_source

    return (raster_source.source_pool_key(source, num_channels), _sidecar_identity(source))


def _raster_stats(source, num_channels: int, key: tuple) -> _RasterStats:
    """``source``'s per-band display bounds, cached under ``key``.

    The one computation behind both the region stretch bounds and what ``/api/images/bands``
    reports, so the numbers a viewer's picker shows are the numbers their region renders through.

    Which read produces them is the raster's size: at or under :data:`_STATS_SAMPLE_BUDGET` the
    seeded window sample reads native pixels (covering all of them, and so exact, for anything the
    budget's grid fits), and past it a single reduced read of the whole frame comes off an overview
    level. A GDAL-backed raster over the budget with no overview levels is refused rather than
    described by a window sample that would decode most of the file to read a thousandth of it;
    every other backend keeps reading native windows, since no pyramid would serve it.

    A concurrent miss on the same raster computes twice and stores the same numbers (both reads are
    deterministic), which is cheaper than holding the lock across the read.
    """
    with _stats_lock:
        hit = _stats_cache.get(key)
        if hit is not None:
            _stats_cache.move_to_end(key)
            return hit

    from tcip_mcp.pipelines import raster_source
    from tcip_mcp.pipelines.band_stats import sampled_band_ranges

    stats = None
    with raster_source.open_raster(source, num_channels) as raster:
        dtype = str(raster.dtype)
        channels = int(raster.num_channels)
        # Only a backend that reads them from the file exposes these; nothing here infers them.
        interpretations = getattr(raster, "band_interpretations", None)
        oversized = raster.width * raster.height > _STATS_SAMPLE_BUDGET
        if oversized and isinstance(raster, raster_source.GdalSource):
            if not _reduced_reads_available(raster):
                raise _overviews_required(
                    str(raster.path),
                    f"this {raster.width}x{raster.height} raster holds "
                    f"{raster.width * raster.height} pixels, over the "
                    f"{_STATS_SAMPLE_BUDGET} its display statistics can be read from native "
                    f"pixels for. Reading them needs its reduced-resolution overviews: build "
                    f'them with POST /api/images/overviews {{"path": "{raster.path}"}}, then '
                    "ask again.")
            ranges, clips, spec = _overview_stats(raster)
            stats = _RasterStats(dtype=dtype, num_channels=channels, ranges=ranges,
                                 clip_bounds=clips, overview_scale=spec.scale,
                                 interpretations=interpretations)
    if stats is None:
        sampled = sampled_band_ranges(
            source, num_channels, seed=_STATS_SEED, window_size=_STATS_WINDOW_SIZE,
            max_windows=_STATS_MAX_WINDOWS, reservoir_size=_STATS_RESERVOIR_SIZE)
        stats = _RasterStats(
            dtype=dtype,
            num_channels=channels,
            ranges=list(sampled.ranges),
            clip_bounds=list(sampled.clip_bounds),
            seed=sampled.sampling.seed,
            pixel_fraction=sampled.sampling.pixel_fraction,
            interpretations=interpretations,
        )
    with _stats_lock:
        _stats_cache[key] = stats
        _stats_cache.move_to_end(key)
        while len(_stats_cache) > _STATS_CACHE_MAX:
            _stats_cache.popitem(last=False)
    return stats


def _finite_display_bounds(applied: "list[tuple[float, float]]"
                           ) -> "list[tuple[float | None, float | None]]":
    """``applied`` with every non-finite low/high (a NaN or an infinity, from a raster whose
    sampled or served pixels held one) mapped to ``None``, so the header this feeds never carries
    a JSON token a strict parser refuses."""
    return [(lo if math.isfinite(lo) else None, hi if math.isfinite(hi) else None)
            for lo, hi in applied]


def _sampled_bounds(stats: _RasterStats, idxs: "list[int]", stretch: str
                    ) -> "list[tuple[float, float]]":
    """The ``(low, high)`` pair per selected band a region render stretches between, in the same
    order as ``idxs``: the clip cut points for ``percent_clip``, the sampled range otherwise
    (``none`` reads both ends, as the sampled minimum and maximum a float raster's divisor comes
    from, and an integer raster ignores the pair for its dtype ceiling)."""
    if stretch == "percent_clip":
        return [stats.clip_bounds[i] for i in idxs]
    return [(stats.ranges[i].minimum, stats.ranges[i].maximum) for i in idxs]


def _served_array_bounds(pixels, idxs: "list[int]", stretch: str) -> "list[tuple[float, float]]":
    """The same pairs read from the served pixels themselves, through the primitives
    ``stretch_band`` derives them with when no bounds are passed, so stating them explicitly is
    the same render and the response can report what was applied."""
    from tcip_mcp.pipelines.band_stats import band_ranges, clip_bounds

    if stretch == "percent_clip":
        return [clip_bounds(pixels[:, :, i]) for i in idxs]
    ranges = band_ranges(pixels)
    return [(ranges[i].minimum, ranges[i].maximum) for i in idxs]


def _fit_output(rect_w: int, rect_h: int, max_width: int, whole_view: bool) -> tuple[int, int]:
    """The ``(width, height)`` a region is served at: ``max_width`` wide at most, never upscaled,
    and never more than :data:`DISPLAY_MAX_PIXELS` of output.

    A whole-view request scales to fit that area whatever it asked for, so an image of any shape
    still renders. An explicit region is refused instead: a caller that named the pixels it wants
    is told the request is over the cap rather than handed fewer pixels than it asked for.
    """
    out_w = min(rect_w, max_width)
    out_h = max(1, round(rect_h * out_w / rect_w))
    if out_w * out_h <= DISPLAY_MAX_PIXELS:
        return out_w, out_h
    if not whole_view:
        raise HTTPException(
            400,
            f"a {rect_w}x{rect_h} region served at {out_w}x{out_h} is {out_w * out_h} output "
            f"pixels, over the display cap of {DISPLAY_MAX_PIXELS}. Ask for a smaller max_width, "
            "or a smaller region.",
        )
    out_w = max(1, int(out_w * math.sqrt(DISPLAY_MAX_PIXELS / (out_w * out_h))))
    out_h = max(1, round(rect_h * out_w / rect_w))
    # Rounding the height back onto the aspect ratio can put the area a hair over the cap.
    while out_w > 1 and out_w * out_h > DISPLAY_MAX_PIXELS:
        out_w -= 1
        out_h = max(1, round(rect_h * out_w / rect_w))
    return out_w, out_h


def _plain_rgb(pixels, dtype, bounds: "tuple[float, float] | None"):
    """A 1/3/4-band array as the plain ``uint8`` RGB the file's own pixels read as: a single band
    replicated, a fourth band dropped, and no data-range stretch applied.

    A non-``uint8`` raster is scaled by ``band_stats.full_scale_denominator`` (the ``none``
    stretch), one denominator for the whole array rather than one per band, so a plain serve never
    shifts the colors the file holds. ``bounds`` carries the sampled ``(minimum, maximum)`` a
    float raster's denominator comes from when the pixels in hand are one region of it. Returns
    the rendered pixels and the divisor the render divided by, so a caller reporting what was
    applied reads the same number the render used rather than deriving its own; ``None`` for a
    ``uint8`` raster, which applies no stretch at all.
    """
    import numpy as np

    from tcip_mcp.pipelines.band_stats import full_scale_denominator, stretch_band

    arr = np.asarray(pixels)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr), None
    divisor = full_scale_denominator(
        arr, dtype, sampled_maximum=None if bounds is None else bounds[1],
        sampled_minimum=None if bounds is None else bounds[0])
    return stretch_band(arr, "none", dtype, (0.0, divisor)), divisor


@router.get("")
def serve_image(
    request: Request,
    path: str = Query(..., description="Absolute path to the image file"),
    max_width: int | None = Query(
        None, ge=1, description="Serve at this width at most; defaults to the display edge bound"),
    quality: int = Query(90, ge=1, le=100),
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
) -> Response:
    """Serve a JPEG of a raster, whole or of one region of it.

    ``x0/y0/x1/y1`` (all four or none) name a half-open region in the raster's own
    full-resolution pixel grid; omitting them serves the whole frame. ``max_width`` is the width
    the result is served at, at most, and defaults to the platform's display edge bound: a request
    is never upscaled, and the output is never more than the display area bound (a whole-view
    request scales to fit it, an explicit region over it is refused, naming the cap).

    With no ``bands`` selection a 1/3/4-band raster serves as its own plain RGB, with no
    data-range stretch: a single band replicated, a fourth band dropped. A ``bands`` selection, a
    ``.bandgroup``-grouped capture, or any other band count composites three bands through the
    shared display stretch instead. Whole-view stretch bounds come from the served pixels
    themselves; a region's come from the raster's seeded per-band sample, so two regions of one
    raster render alike. Both are reported back in ``X-TCIP-Stats-Source`` and, where bounds were
    applied, ``X-TCIP-Display-Bounds``; a bound the raster's own pixels left non-finite (a NaN or
    an infinity) reports as ``null`` rather than a JSON token no client parses.

    A scaled read of a raster larger than the display area bound needs the reduced-resolution
    overviews GDAL serves it from; without them the request is refused, naming
    ``POST /api/images/overviews`` (``X-TCIP-Image-Error: overviews_required``), since decoding it
    natively is what the bound exists to prevent.

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
        full_scale_denominator,
    )
    from tcip_mcp.pipelines.data.band_groups import BandGroupIncomplete, BandGroupRef
    from tcip_mcp.pipelines.derivations import probe_channels
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem, resolve_image_source

    src = _checked(path)
    if stretch not in STRETCH_MODES:
        raise HTTPException(400, f"stretch must be one of {sorted(STRETCH_MODES)}, got {stretch!r}")

    try:
        source = resolve_image_source(src.parent, src.stem)
    except BandGroupIncomplete as exc:
        raise HTTPException(409, str(exc)) from exc
    except AmbiguousImageStem as exc:
        raise HTTPException(400, str(exc)) from exc

    corners = (x0, y0, x1, y1)
    if any(c is None for c in corners) and any(c is not None for c in corners):
        raise HTTPException(400, "a region needs all four of x0, y0, x1, y1, or none of them")
    whole_view = corners[0] is None
    if not whole_view:
        # the all-or-none guard above already requires every corner set when not whole_view
        assert x0 is not None and y0 is not None and x1 is not None and y1 is not None
        if not (x0 < x1 and y0 < y1):
            raise HTTPException(
                400, f"region [{y0}:{y1}, {x0}:{x1}] is empty; x0 < x1 and y0 < y1 are required")

    band_tokens = _parse_band_tokens(bands) if bands is not None else None
    composite_requested = bands is not None or isinstance(source, BandGroupRef)
    probed = probe_channels(source)
    open_channels = (
        probed if composite_requested
        else raster_source.image_route_channel_count(source, probed)
    )
    target_width = DISPLAY_MAX_EDGE if max_width is None else max_width

    # Requested params only: the scale a read is served at depends on the raster's own size and on
    # whether an overview level exists, neither known at lookup time, so it returns as a header.
    key = hashlib.md5(
        f"{RENDER_CACHE_VERSION}:{_source_identity(source, open_channels)}:"
        f"{target_width}:{quality}:{bands}:{stretch}:{corners}".encode()
    ).hexdigest()
    etag = f'W/"{key}"'
    cache_headers = {"ETag": etag, "Cache-Control": "private, max-age=3600"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=cache_headers)

    cache_dir = _render_cache_dir()
    cached = cache_dir / f"{key}.jpg"
    cached_headers = cache_dir / f"{key}.json"
    if cached.is_file() and cached_headers.is_file():
        try:
            extra = json.loads(cached_headers.read_text(encoding="utf-8"))
            cached.touch()  # refresh mtime so LRU eviction keeps the working set
            return FileResponse(cached, media_type="image/jpeg",
                                headers={**cache_headers, **extra})
        except (OSError, ValueError):
            pass  # an unreadable cache entry is rendered again, never served as-is

    opening = True
    try:
        with raster_source.open_raster(source, open_channels) as raster:
            opening = False
            if whole_view:
                rect = raster_source.Rect(0, 0, raster.width, raster.height)
            else:
                # the all-or-none guard above already requires every corner set when not whole_view
                assert x0 is not None and y0 is not None and x1 is not None and y1 is not None
                rect = raster_source.Rect(x0, y0, x1, y1)
            if not whole_view and (rect.x1 > raster.width or rect.y1 > raster.height):
                raise HTTPException(
                    400,
                    f"region [{rect.y0}:{rect.y1}, {rect.x0}:{rect.x1}] is outside this "
                    f"{raster.width}x{raster.height} raster",
                )
            out_w, out_h = _fit_output(rect.width, rect.height, target_width, whole_view)
            scaled = out_w < rect.width
            if (scaled and rect.width * rect.height > DISPLAY_MAX_PIXELS
                    and isinstance(raster, raster_source.GdalSource)
                    and not _reduced_reads_available(raster)):
                raise _overviews_required(
                    path,
                    f"serving this {rect.width}x{rect.height} region at {out_w}x{out_h} needs "
                    f"reduced-resolution overviews: reading it natively would decode "
                    f"{rect.width * rect.height} pixels, over the display cap of "
                    f"{DISPLAY_MAX_PIXELS}. Build them with POST /api/images/overviews "
                    f'{{"path": "{path}"}}, then request this view again.')
            pixels, _spec = raster.read_region(
                rect, target_size=(out_w, out_h) if scaled else None)
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
        sampled = (_raster_stats(source, open_channels, _source_identity(source, open_channels))
                   if wants_bounds and not whole_view else None)

        if composite and stretch == "none" and integer:
            rgb = composite_display_rgb(pixels, idxs, stretch, None)
            stats_source, applied = StatsSource(read="dtype_full_scale"), []
        elif composite:
            applied = (_served_array_bounds(pixels, idxs, stretch) if sampled is None
                       else _sampled_bounds(sampled, idxs, stretch))
            rgb = composite_display_rgb(pixels, idxs, stretch, applied)
            if stretch == "none":
                # The divisor each band actually rendered against, not its sampled (min, max).
                applied = [
                    (0.0, full_scale_denominator(pixels, dtype, sampled_maximum=hi,
                                                 sampled_minimum=lo))
                    for lo, hi in applied
                ]
            stats_source = (
                StatsSource(read="served_array") if sampled is None else sampled.stats_source())
        elif integer:
            rgb, _divisor = _plain_rgb(pixels, dtype, None)
            stats_source = StatsSource(read="none" if dtype == np.uint8 else "dtype_full_scale")
            applied = []
        else:
            # Only the bands a plain serve displays: a fourth band is dropped before the viewer
            # sees it, so its level must not set the scale the other three are divided by.
            ranges = band_ranges(pixels) if sampled is None else sampled.ranges
            band_bounds = (
                min(ranges[i].minimum for i in idxs), max(ranges[i].maximum for i in idxs))
            rgb, divisor = _plain_rgb(pixels, dtype, band_bounds)
            applied = [(0.0, divisor)]
            stats_source = (
                StatsSource(read="served_array") if sampled is None else sampled.stats_source())

        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgb), mode="RGB").save(buf, "JPEG", quality=quality)
        data = buf.getvalue()  # no optimize=True: ~0.4 s for ~5% size
    except HTTPException:
        raise
    except ValueError as exc:
        what = ("open this image" if opening
                else "read this image" if whole_view else "read this region")
        raise HTTPException(400, f"could not {what}: {exc}") from exc
    except Exception as exc:
        raise HTTPException(500, f"could not process image: {exc}") from exc

    extra = {
        "X-TCIP-Stats-Source": json.dumps(stats_source.model_dump(), allow_nan=False),
        "X-TCIP-Served-Size": f"{out_w}x{out_h}",
    }
    if applied:
        extra["X-TCIP-Display-Bounds"] = json.dumps(
            _finite_display_bounds(applied), allow_nan=False)

    try:
        tmp = cache_dir / f"{key}.{threading.get_ident()}.tmp"
        tmp.write_bytes(data)
        tmp.replace(cached)
        cached_headers.write_text(json.dumps(extra), encoding="utf-8")
        _note_cache_write(cache_dir, len(data) + cached_headers.stat().st_size)
        _evict_lru(cache_dir)
    except OSError:
        pass  # cache is best-effort; the response below is already rendered

    return Response(content=data, media_type="image/jpeg",
                    headers={**cache_headers, **extra})


@router.get("/bands")
def get_bands(path: str = Query(...)) -> dict:
    """Band count + per-band stats for ``path``: the picker's symbology data, and the one fact
    (``band_count > 3``) the frontend uses to decide whether to show the picker at all.

    Resolves the same way ``serve_image`` does: ``path`` may be a plain raster or a
    ``.bandgroup`` manifest naming a grouped multi-band capture. ``band_count`` is always cheap
    (``probe_channels`` never decodes pixels for a photographic format, and reads only the TIFF
    header when possible); the per-band stats below only run when they can actually tell the
    picker something new: a plain (non-grouped) raster at ``band_count <= 3`` is an ordinary
    photographic image with no real per-band symbology to report, so that case reads no pixels at
    all. A ``.bandgroup``-grouped capture always gets the full per-band stats even at exactly 3
    bands: its bands are real, independently named/wavelength-tagged captures the picker
    legitimately shows, not RGB color channels.

    The stats never come from a whole decode, so a raster too large to decode is still
    describable, and the response says which read produced them. A raster whose pixels fit the
    native-sampling budget carries ``sampled``, ``pixel_fraction`` and ``seed``: ``sampled`` is
    false exactly when the sample covered every pixel (``pixel_fraction`` 1.0), where the numbers
    are the raster's own exact bounds. A larger one is read once off an overview level instead and
    carries ``sampled`` false with ``overview_scale``, the served/native resolution ratio it was
    read at; those bounds are narrower than the raster's own, since an overview pixel averages
    native ones, and they describe display scale only. Either way they are the same numbers this
    raster's region renders stretch through.

    A band carries ``interpretation`` (``red``, ``alpha``, and the rest) where the backend reads it
    from the file, which is what lets a caller tell an ordinary RGBA frame from a genuinely
    multi-band capture that happens to hold four bands. The key is absent where nothing knows.
    """
    from tcip_mcp.pipelines.data.band_groups import BandGroupIncomplete, BandGroupRef
    from tcip_mcp.pipelines.derivations import probe_channels
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem, resolve_image_source

    src = _checked(path)
    try:
        source = resolve_image_source(src.parent, src.stem)
    except BandGroupIncomplete as exc:
        raise HTTPException(409, str(exc)) from exc
    except AmbiguousImageStem as exc:
        raise HTTPException(400, str(exc)) from exc

    n = probe_channels(source)
    if n <= 3 and not isinstance(source, BandGroupRef):
        return {"band_count": n, "bands": []}

    stats = _raster_stats(source, n, _source_identity(source, n))
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
    if stats.overview_scale is not None:
        return {"band_count": len(names), "bands": bands, "sampled": False,
                "overview_scale": stats.overview_scale}
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
    thread: "threading.Thread | None" = field(default=None, repr=False)


_overview_registry = jobstore.JobRegistry(None)
"""The dict-plus-lock live registry for overview-build jobs (see ``jobstore.JobRegistry``); this
one carries no root concept and persists nothing, unlike inference.py's, tuning.py's and
review.py's priority queue, whose registries share the same home."""


class OverviewBuildPayload(BaseModel):
    path: str


def _overview_summary(job: OverviewJob) -> dict:
    return {"job_id": job.job_id, "path": job.path, "status": job.status,
            "progress": job.progress, "error": job.error}


def _overview_worker(job: OverviewJob) -> None:
    """Build the pyramid, recording progress and whatever stopped it.

    A refusal to rebuild over a pyramid that already exists is the outcome the caller wanted (a
    build for the same raster finished first); anything else that stops the build is a failure.
    Asking whether one exists can fail in its own right, on a raster GDAL cannot open at all: that
    answers the question (there is no pyramid) rather than ending the job's thread, which would
    leave a caller polling a job that never reaches a terminal state.
    """
    from tcip_mcp.pipelines.overviews import build_overviews, has_overviews

    def record(fraction: float) -> None:
        job.progress = float(fraction)

    job.status = "running"
    try:
        build_overviews(job.path, progress_cb=record)
    except Exception as exc:  # noqa: BLE001, the job records what stopped it
        try:
            built = has_overviews(job.path)
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
    src = _checked(payload.path)
    job, created = _overview_registry.find_or_register(
        lambda existing: existing.path == str(src) and existing.status in ("pending", "running"),
        lambda: OverviewJob(job_id=f"ovr-{uuid.uuid4().hex[:8]}", path=str(src)),
    )
    if created:
        thread = threading.Thread(target=_overview_worker, args=(job,), daemon=True)
        job.thread = thread
        thread.start()
    return _overview_summary(job)


@router.get("/overviews/status")
def get_overview_job(job_id: str = Query(...)) -> dict:
    """An overview build's status and completion fraction."""
    job = _overview_registry.get(job_id)
    if job is None:
        raise HTTPException(404, f"job not found: {job_id}")
    return _overview_summary(job)
