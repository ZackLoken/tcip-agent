"""Image serving with uniform EXIF orientation.

The browser must receive pixels in the same frame the annotations were authored
against. Valley_Farm labels were created on EXIF-transposed images, so we
apply ``auto_orient_image`` uniformly here.

Only one code path reads raw JPEGs from disk — this one. All other
components should receive images via :func:`serve_image`.
"""

from __future__ import annotations

import io
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Response
from PIL import Image

from tcip_annotation.utils import auto_orient_image

router = APIRouter(prefix="/api/images", tags=["images"])


@router.get("")
def serve_image(
    path: str = Query(..., description="Absolute path to the image file"),
    max_width: int | None = Query(None, ge=1, le=8192, description="Downsample to this width"),
    quality: int = Query(85, ge=1, le=100),
) -> Response:
    """Serve an EXIF-corrected JPEG.

    Optional ``max_width`` downsamples (preserving aspect) — useful for
    thumbnail grids and the lo-res background layer. Full-res is served
    by omitting ``max_width``.
    """
    src = Path(path)
    if not src.is_file():
        raise HTTPException(404, f"not a file: {path}")

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
            im.save(buf, "JPEG", quality=quality, optimize=True)
            data = buf.getvalue()
    except Exception as exc:
        raise HTTPException(500, f"could not process image: {exc}") from exc

    return Response(content=data, media_type="image/jpeg")


@router.get("/dimensions")
def get_dimensions(path: str = Query(...)) -> dict:
    """Return the EXIF-oriented (width, height) of an image."""
    src = Path(path)
    if not src.is_file():
        raise HTTPException(404, f"not a file: {path}")
    with Image.open(src) as raw:
        im = auto_orient_image(raw)
        w, h = im.size
    return {"path": path, "width": w, "height": h}
