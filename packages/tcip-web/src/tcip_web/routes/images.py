"""Image serving with uniform EXIF orientation.

The browser must receive pixels in the same frame the annotations were authored
against. Valley_Farm labels were created on EXIF-transposed images, so we
apply ``auto_orient_image`` uniformly here.

Only one code path reads raw JPEGs from disk: this one. All other
components should receive images via :func:`serve_image`.
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from PIL import Image

from tcip_annotation.utils import auto_orient_image
from tcip_web.paths import assert_path_allowed

router = APIRouter(prefix="/api/images", tags=["images"])

# Rendered-variant disk cache: the full decode→orient→resize→encode pipeline costs
# ~2 s of CPU per 24MP frame, and the browser cache only covers this session. Capped
# by file count (~500 × ~5 MB ≈ 2.5 GB) with mtime-LRU eviction.
_CACHE_MAX_FILES = 500


def _render_cache_dir() -> Path:
    root = os.environ.get("TCIP_PROJECT_ROOT")
    base = (
        Path(root) / ".tcip" / "cache" / "img"
        if root
        else Path(tempfile.gettempdir()) / "tcip-img-cache"
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def _evict_lru(cache_dir: Path) -> None:
    try:
        files = [p for p in cache_dir.iterdir() if p.is_file() and p.suffix == ".jpg"]
        if len(files) <= _CACHE_MAX_FILES:
            return
        files.sort(key=lambda p: p.stat().st_mtime)
        for p in files[: len(files) - _CACHE_MAX_FILES]:
            p.unlink()
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


def _composite_bands(source, band_tokens: "list[str] | None", stretch: str) -> Image.Image:
    """Decode ``source`` (a plain multi-band raster or a ``BandGroupRef``), select 3 bands (by
    declared band name when the source has one, else by 0-index), stretch each independently, and
    composite to an 8-bit RGB image, the same live-composite mechanism the ArcGIS-Pro-styled
    picker drives (never a physically-stacked file on disk).
    """
    import numpy as np

    from tcip_mcp.pipelines.band_stats import stretch_band
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.derivations import probe_channels
    from tcip_mcp.pipelines.image_utils import load_image

    n = probe_channels(source)
    arr = np.asarray(load_image(source, n))
    if arr.ndim == 2:
        arr = arr[:, :, None]
    total_bands = arr.shape[-1]
    declared_names = list(source.bands) if isinstance(source, BandGroupRef) else None

    if band_tokens is None:
        # No explicit selection: the first 3 bands (repeating the last one short of 3).
        idxs = [min(i, total_bands - 1) for i in range(3)]
    else:
        idxs = [_band_index(t, declared_names, total_bands) for t in band_tokens]

    channels = [stretch_band(arr[:, :, i], stretch, arr.dtype) for i in idxs]
    rgb = np.stack(channels, axis=-1)
    return Image.fromarray(rgb, mode="RGB")


@router.get("")
def serve_image(
    request: Request,
    path: str = Query(..., description="Absolute path to the image file"),
    max_width: int | None = Query(None, ge=1, le=8192, description="Downsample to this width"),
    quality: int = Query(90, ge=1, le=100),
    bands: str | None = Query(
        None, description="3 comma-separated band names or 0-based indices, e.g. "
                          "'NIR,Red,Green' or '3,2,1'; selects a live composite instead of the "
                          "file's own pixels as-is."),
    stretch: str = Query(
        "minmax", description="minmax|percent_clip|none; applied only when compositing bands "
                              "(bands given, or path names a .bandgroup-grouped capture)."),
) -> Response:
    """Serve an EXIF-corrected JPEG.

    Optional ``max_width`` downsamples (preserving aspect): useful for
    thumbnail grids and the lo-res background layer. Full-res is served
    by omitting ``max_width``.

    ``path`` naming a ``.bandgroup`` manifest (a grouped multi-band capture, see
    ``pipelines.data.band_groups``) or an explicit ``bands`` selection routes through a live
    band-composite render instead of a bare ``Image.open``; omitting both keeps today's behavior
    byte-identical for an ordinary photographic file.

    An ETag keyed on the file's identity (mtime + size) and every render param (including
    ``bands``/``stretch``) lets the browser revalidate with a cheap 304; the disk cache makes a
    cold request (fresh session, page refresh, prefetch) a sendfile instead of a re-render.
    """
    from tcip_mcp.pipelines.band_stats import STRETCH_MODES

    src = _checked(path)
    if stretch not in STRETCH_MODES:
        raise HTTPException(400, f"stretch must be one of {sorted(STRETCH_MODES)}, got {stretch!r}")

    from tcip_mcp.pipelines.data.band_groups import BandGroupIncomplete, BandGroupRef
    from tcip_mcp.pipelines.image_utils import resolve_image_source

    try:
        source = resolve_image_source(src.parent, src.stem)
    except BandGroupIncomplete as exc:
        raise HTTPException(409, str(exc)) from exc

    band_tokens = _parse_band_tokens(bands) if bands is not None else None
    composite = bands is not None or isinstance(source, BandGroupRef)

    st = src.stat()
    # Without bands/stretch in the key, two different band-combination requests against the same
    # .bandgroup path would collide on cache key and the second would silently be served the
    # first's cached composite. Additive to the pre-existing key format, never removed from it.
    key = hashlib.md5(
        f"{src}:{st.st_mtime_ns}:{st.st_size}:{max_width}:{quality}:{bands}:{stretch}".encode()
    ).hexdigest()
    etag = f'W/"{key}"'
    cache_headers = {"ETag": etag, "Cache-Control": "private, max-age=3600"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=cache_headers)

    cache_dir = _render_cache_dir()
    cached = cache_dir / f"{key}.jpg"
    if cached.is_file():
        try:
            cached.touch()  # refresh mtime so LRU eviction keeps the working set
        except OSError:
            pass
        return FileResponse(cached, media_type="image/jpeg", headers=cache_headers)

    try:
        if composite:
            im = _composite_bands(source, band_tokens, stretch)
        else:
            with Image.open(src) as raw:
                im = auto_orient_image(raw).convert("RGB")
        if max_width is not None and im.size[0] > max_width:
            scale = max_width / im.size[0]
            im = im.resize(
                (max_width, int(im.size[1] * scale)),
                Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)  # no optimize=True: ~0.4 s for ~5% size
        data = buf.getvalue()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"could not process image: {exc}") from exc

    try:
        tmp = cache_dir / f"{key}.{threading.get_ident()}.tmp"
        tmp.write_bytes(data)
        tmp.replace(cached)
        _evict_lru(cache_dir)
    except OSError:
        pass  # cache is best-effort; the response below is already rendered

    return Response(content=data, media_type="image/jpeg", headers=cache_headers)


@router.get("/dimensions")
def get_dimensions(path: str = Query(...)) -> dict:
    """Return the EXIF-oriented (width, height) of an image (header-only where possible)."""
    from tcip_mcp.pipelines.data.band_groups import BandGroupIncomplete
    from tcip_mcp.pipelines.image_utils import image_dimensions, resolve_image_source

    src = _checked(path)
    # Channel-aware: resolve_image_source folds a `.bandgroup` manifest (or a genuinely
    # multi-band raster) into the real frame image_dimensions measures, instead of the bare
    # channel-blind get_image_dimensions misreporting a multi-band GeoTIFF's axes (same fix
    # already applied to annotate.py/review.py's own _image_dims).
    try:
        source = resolve_image_source(src.parent, src.stem)
    except BandGroupIncomplete as exc:
        raise HTTPException(409, str(exc)) from exc
    w, h = image_dimensions(source)
    return {"path": path, "width": w, "height": h}


@router.get("/bands")
def get_bands(path: str = Query(...)) -> dict:
    """Band count + per-band stats for ``path``: the picker's symbology data, and the one fact
    (``band_count > 3``) the frontend uses to decide whether to show the picker at all.

    Resolves the same way ``serve_image`` does: ``path`` may be a plain raster or a
    ``.bandgroup`` manifest naming a grouped multi-band capture. ``band_count`` is always cheap
    (``probe_channels`` never decodes pixels for a photographic format, and reads only the TIFF
    header when possible); the per-band min/max/dtype decode below only runs when it can actually
    tell the picker something new: a plain (non-grouped) raster at ``band_count <= 3`` is an
    ordinary photographic image with no real per-band symbology to report, so that case skips the
    decode entirely (progressive disclosure: an RGB request never pays for a full decode it would
    discard). A ``.bandgroup``-grouped capture always gets the full per-band stats even at exactly
    3 bands: its bands are real, independently named/wavelength-tagged captures the picker
    legitimately shows, not RGB color channels.
    """
    import numpy as np

    from tcip_mcp.pipelines.band_stats import band_ranges
    from tcip_mcp.pipelines.data.band_groups import BandGroupIncomplete, BandGroupRef
    from tcip_mcp.pipelines.derivations import probe_channels
    from tcip_mcp.pipelines.image_utils import load_image, resolve_image_source

    src = _checked(path)
    try:
        source = resolve_image_source(src.parent, src.stem)
    except BandGroupIncomplete as exc:
        raise HTTPException(409, str(exc)) from exc

    n = probe_channels(source)
    if n <= 3 and not isinstance(source, BandGroupRef):
        return {"band_count": n, "bands": []}

    arr = np.asarray(load_image(source, n))
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if isinstance(source, BandGroupRef):
        names = list(source.bands)
        wavelengths = source.central_wavelength_nm or {}
    else:
        names = [str(i) for i in range(arr.shape[-1])]
        wavelengths = {}

    ranges = band_ranges(arr)
    bands = [
        {
            "name": name,
            "wavelength_nm": wavelengths.get(name),
            "dtype": str(arr[:, :, i].dtype),
            "min": ranges[i].minimum,
            "max": ranges[i].maximum,
        }
        for i, name in enumerate(names)
    ]
    return {"band_count": len(names), "bands": bands}
