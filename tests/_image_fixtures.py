"""Synthetic images a test writes to disk."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from PIL import Image


def _deg_to_dms(value: float) -> tuple[float, float, float]:
    v = abs(value)
    d = int(v)
    m_full = (v - d) * 60
    m = int(m_full)
    return (float(d), float(m), round((m_full - m) * 60, 4))


def capture_exif(when: datetime) -> Image.Exif:
    """Return a PIL EXIF block stating ``when`` (a ``datetime``) as the image's DateTimeOriginal."""
    from PIL import Image

    exif = Image.Exif()
    exif[0x8769] = {0x9003: when.strftime("%Y:%m:%d %H:%M:%S")}
    return exif


def write_geo_image(
    path: Path,
    lat: float,
    lon: float,
    when: datetime,
    image: Image.Image | None = None,
    *,
    h_pos_err: float | None = None,
    omit: tuple[int, ...] = (),
) -> None:
    """Save an image at ``path`` with EXIF DateTimeOriginal ``when`` and GPS ``lat``/``lon``.

    The EXIF also carries ``h_pos_err``, when given, as GPSHPositioningError (meters), less the
    GPS tags ``omit`` names (``0x0001`` the latitude's hemisphere, ``0x0003``/``0x0004`` the
    longitude's). The image saved is the PIL ``image`` given, else an 8 by 8 black RGB frame, at
    quality 100 with no chroma subsampling.
    """
    from PIL import Image, TiffImagePlugin

    path.parent.mkdir(parents=True, exist_ok=True)
    exif = capture_exif(when)
    gps = {
        0x0001: "N" if lat >= 0 else "S",
        0x0002: _deg_to_dms(lat),
        0x0003: "E" if lon >= 0 else "W",
        0x0004: _deg_to_dms(lon),
    }
    if h_pos_err is not None:
        gps[0x001F] = TiffImagePlugin.IFDRational(h_pos_err)
    exif[0x8825] = {
        tag: value for tag, value in gps.items() if tag not in omit
    }
    frame = image if image is not None else Image.new("RGB", (8, 8))
    frame.save(path, exif=exif, quality=100, subsampling=0)


def write_noise_image(
    path: Path, size: int, bright: bool = False, span: float = 0.3
) -> None:
    """Save unseeded uniform noise as a ``size`` by ``size`` image at ``path``, its parent made.

    The tensor saved through ``torchvision.utils.save_image`` is ``torch.rand(3, size, size)``
    times ``span``, plus 0.7 when ``bright``.
    """
    import torch
    from torchvision.utils import save_image

    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        torch.rand(3, size, size) * span + (0.7 if bright else 0.0), str(path)
    )
