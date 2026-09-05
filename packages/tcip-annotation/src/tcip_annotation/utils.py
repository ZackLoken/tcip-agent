"""Shared utilities: image orientation, geometry helpers."""

from __future__ import annotations

import logging
from typing import Any, cast

from PIL import Image, ExifTags

logger = logging.getLogger(__name__)


def _read_orientation_tag(img: Image.Image) -> int | None:
    """The image's EXIF Orientation tag value, or ``None`` if it carries no EXIF data or no
    Orientation tag. The one orientation read `auto_orient_image` and `get_image_dimensions`
    both use, so the frame one rotates and the frame the other measures can't disagree.

    Reads the tag through ``_getexif``, a JPEG-family (JPEG, MPO) accessor PIL's TIFF plugin
    does not implement; the photographic path this serves admits only those containers.
    """
    try:
        # _getexif is a JpegImageFile/MpoImageFile accessor, absent from the Image.Image stub.
        exif = cast(Any, img)._getexif()
        if exif is None:
            return None
        orientation_key = None
        for k, v in ExifTags.TAGS.items():
            if v == "Orientation":
                orientation_key = k
                break
        if orientation_key is None or orientation_key not in exif:
            return None
        return int(exif[orientation_key])
    except Exception:
        return None


def auto_orient_image(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation correction to a PIL Image."""
    try:
        orientation = _read_orientation_tag(img)
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
    path forced a full decode of a 24MP frame (~0.5 s) just to learn its dimensions.
    Its one production caller is `image_utils.image_dimensions`, which routes every
    photographic container here.
    """
    with Image.open(path) as img:
        w, h = img.size
        orientation = _read_orientation_tag(img) or 1
    # Orientations 5-8 include a 90°/270° rotation, so the oriented axes swap.
    return (h, w) if orientation in (5, 6, 7, 8) else (w, h)
