"""Shared utilities: image orientation, geometry helpers."""

from __future__ import annotations

import logging
from PIL import Image, ExifTags

logger = logging.getLogger(__name__)


def auto_orient_image(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation correction to a PIL Image."""
    try:
        exif = img._getexif()
        if exif is None:
            return img
        orientation_key = None
        for k, v in ExifTags.TAGS.items():
            if v == "Orientation":
                orientation_key = k
                break
        if orientation_key is None or orientation_key not in exif:
            return img
        orientation = exif[orientation_key]
        ops = {
            2: lambda i: i.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            3: lambda i: i.rotate(180, expand=True),
            4: lambda i: i.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
            5: lambda i: i.transpose(Image.Transpose.FLIP_LEFT_RIGHT).rotate(270, expand=True),
            6: lambda i: i.rotate(270, expand=True),
            7: lambda i: i.transpose(Image.Transpose.FLIP_LEFT_RIGHT).rotate(90, expand=True),
            8: lambda i: i.rotate(90, expand=True),
        }
        if orientation in ops:
            img = ops[orientation](img)
    except Exception:
        logger.debug("EXIF orientation correction failed", exc_info=True)
    return img


def get_image_dimensions(path: str) -> tuple[int, int]:
    """Return (width, height) of an image, applying EXIF orientation.

    Header-only: reads size + orientation without decoding pixels. The transpose-based
    path forced a full decode of a 24MP frame (~0.5 s) just to learn its dimensions,
    and this runs on every label load / save / review-matches call.
    """
    with Image.open(path) as img:
        w, h = img.size
        orientation = 1
        try:
            orientation = int(img.getexif().get(0x0112, 1) or 1)
        except Exception:
            pass
    # Orientations 5-8 include a 90°/270° rotation, so the oriented axes swap.
    return (h, w) if orientation in (5, 6, 7, 8) else (w, h)
