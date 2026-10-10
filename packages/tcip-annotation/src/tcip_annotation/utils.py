"""Shared utilities: image orientation, geometry helpers."""

from __future__ import annotations

import logging
from typing import Any, cast

from PIL import Image, ExifTags

logger = logging.getLogger(__name__)


def _read_orientation_tag(img: Image.Image) -> int | None:
    """The image's EXIF Orientation tag value, or ``None`` if it carries no EXIF data or no
    Orientation tag, read through ``_getexif``, a JPEG-family (JPEG, MPO) accessor.
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


_ORIENTATION_OPS = {
    2: lambda i: i.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
    3: lambda i: i.rotate(180, expand=True),
    4: lambda i: i.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
    5: lambda i: i.transpose(Image.Transpose.FLIP_LEFT_RIGHT).rotate(270, expand=True),
    6: lambda i: i.rotate(270, expand=True),
    7: lambda i: i.transpose(Image.Transpose.FLIP_LEFT_RIGHT).rotate(90, expand=True),
    8: lambda i: i.rotate(90, expand=True),
}
"""The operation each EXIF Orientation tag value names, applied to bring a frame upright."""


def auto_orient_image(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation correction to a PIL Image."""
    try:
        op = _ORIENTATION_OPS.get(_read_orientation_tag(img) or 1)
        if op is not None:
            img = op(img)
    except Exception:
        logger.debug("EXIF orientation correction failed", exc_info=True)
    return img


def oriented_size(img: Image.Image) -> tuple[int, int]:
    """``(width, height)`` of the opened image ``img`` once its EXIF orientation is applied, from
    its header alone, no pixel decode: the tag's operation (:data:`_ORIENTATION_OPS`) applied to a
    two-pixel probe says whether the axes swap."""
    w, h = img.size
    op = _ORIENTATION_OPS.get(_read_orientation_tag(img) or 1)
    swaps = op is not None and op(Image.new("L", (2, 1))).size == (1, 2)
    return (h, w) if swaps else (w, h)
