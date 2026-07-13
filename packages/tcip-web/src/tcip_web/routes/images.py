"""Image serving with uniform EXIF orientation.

The browser must receive pixels in the same frame the annotations were authored
against. Valley_Farm labels were created on EXIF-transposed images, so we
apply ``auto_orient_image`` uniformly here.

Only one code path reads raw JPEGs from disk — this one. All other
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

from tcip_annotation.utils import auto_orient_image, get_image_dimensions
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


@router.get("")
def serve_image(
    request: Request,
    path: str = Query(..., description="Absolute path to the image file"),
    max_width: int | None = Query(None, ge=1, le=8192, description="Downsample to this width"),
    quality: int = Query(90, ge=1, le=100),
) -> Response:
    """Serve an EXIF-corrected JPEG.

    Optional ``max_width`` downsamples (preserving aspect) — useful for
    thumbnail grids and the lo-res background layer. Full-res is served
    by omitting ``max_width``.

    An ETag keyed on the file's identity (mtime + size) and the render params lets the
    browser revalidate with a cheap 304; the disk cache makes a cold request (fresh
    session, page refresh, prefetch) a sendfile instead of a ~2 s render.
    """
    src = _checked(path)
    st = src.stat()
    key = hashlib.md5(
        f"{src}:{st.st_mtime_ns}:{st.st_size}:{max_width}:{quality}".encode()
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
    """Return the EXIF-oriented (width, height) of an image (header-only)."""
    src = _checked(path)
    w, h = get_image_dimensions(str(src))
    return {"path": path, "width": w, "height": h}
