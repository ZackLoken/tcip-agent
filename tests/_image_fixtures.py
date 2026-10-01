"""Synthetic images a test writes to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _deg_to_dms(value: float) -> tuple[float, float, float]:
    v = abs(value)
    d = int(v)
    m_full = (v - d) * 60
    m = int(m_full)
    return (float(d), float(m), round((m_full - m) * 60, 4))


def write_geo_image(path: Path, lat: float, lon: float, when: Any, image: Any = None) -> None:
    """A JPEG at ``path`` carrying EXIF DateTimeOriginal ``when`` and GPS ``lat``/``lon``: the
    PIL ``image`` given, else a tiny black frame."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    exif[0x8769] = {0x9003: when.strftime("%Y:%m:%d %H:%M:%S")}
    exif[0x8825] = {
        0x0001: "N" if lat >= 0 else "S", 0x0002: _deg_to_dms(lat),
        0x0003: "E" if lon >= 0 else "W", 0x0004: _deg_to_dms(lon),
    }
    frame = image if image is not None else Image.new("RGB", (8, 8))
    frame.save(path, exif=exif, quality=100, subsampling=0)


def write_image(path: Path, size: int = 100) -> None:
    """A flat gray ``size`` by ``size`` RGB image at ``path``, its parent created."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), (120, 120, 120)).save(path)


def write_noise_image(path: Path, size: int, bright: bool = False, span: float = 0.3) -> None:
    """A ``size`` by ``size`` RGB image of uniform noise at ``path``, its parent created: values
    in [0, ``span``), or offset by 0.7 when ``bright``."""
    import torch
    from torchvision.utils import save_image

    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.rand(3, size, size) * span + (0.7 if bright else 0.0), str(path))
