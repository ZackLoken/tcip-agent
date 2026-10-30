"""Write the bands route's answers as a fixture file, produced by the platform itself.

In a scratch session (``tools/fixture_session.py``), a project created through
``tests._web_fixtures.named_project`` and opened through the picker's route holds seven captures:
a band group of four 16-bit bands whose manifest states each one's central wavelength, a band
group of three 8-bit bands and one of one 16-bit band whose manifests state none (each grouped by
``tests._producer_fixtures.write_band_group``), a four-band 8-bit TIFF and a four-band 16-bit
one whose files name each band's color interpretation, a plain RGB photo
(``tests._producer_fixtures.write_image``), and a four-band ``.npy`` past the stats sample's window
budget, whose bounds come from part of its pixels. Each is described by
``GET /api/images/bands``, and the answers are written, normalized
(``fixture_session.normalized``), to :data:`GENERATED_PATH`, keyed ``four_spectral``,
``three_unlabeled``, ``single``, ``rgba``, ``deep_rgba``, ``plain`` and ``sampled``.

    python -m tools.generate_bands_fixture
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import tifffile

from tests import FRONTEND_SRC
from tools.fixture_session import (
    answered,
    normalized,
    opened,
    scratch_session,
    write_fixture,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue

GENERATED_PATH = FRONTEND_SRC / "test" / "bandAnswers.json"

FRAME = (12, 16)
"""Each capture's height and width in pixels, few enough that the stats read every pixel."""

SPECTRAL_NM = {"Blue": 475.0, "Green": 560.0, "Red": 650.0, "NIR": 840.0}
"""The four-band capture's bands and the central wavelengths its manifest states, example
values of a four-band sensor."""

UNLABELED = ("Red", "Green", "Blue")
"""The three-band capture's band names, its manifest stating no wavelength."""

SINGLE = ("NIR",)
"""The one-band capture's band name, its manifest stating no wavelength."""


def _ramp(dtype: type[np.integer], top: int) -> np.ndarray:
    """Return a :data:`FRAME` of ``dtype`` rising from 0 to ``top`` across its columns."""
    row = np.linspace(0, top, FRAME[1]).astype(dtype)
    return np.repeat(row[None, :], FRAME[0], axis=0)


def _group(
    images: Path, stem: str, names: tuple[str, ...], dtype: type[np.integer]
) -> dict[str, Path]:
    """Write one single-band TIFF of ``dtype`` per name, ``stem_<name>.tif``; return them by name.

    Each band rises to a top of its own, so no two report the same range.
    """
    tops = np.linspace(
        np.iinfo(dtype).max // 4, np.iinfo(dtype).max, len(names)
    )
    bands = {}
    for name, top in zip(names, tops, strict=True):
        bands[name] = images / f"{stem}_{name}.tif"
        tifffile.imwrite(str(bands[name]), _ramp(dtype, int(top)))
    return bands


def _past_the_sample_budget(path: Path) -> Path:
    """Write a four-band 8-bit ``.npy`` one sampling window wider and taller than the stats budget.

    The budget is the bands route's own (``routes.images._STATS_MAX_WINDOWS`` windows of
    ``_STATS_WINDOW_SIZE``), so the route reads its bounds off part of the raster's pixels.
    """
    from math import isqrt

    from tcip_web.routes import images as images_route

    edge = images_route._STATS_WINDOW_SIZE * (
        isqrt(images_route._STATS_MAX_WINDOWS) + 1
    )
    row = np.linspace(0, 255, edge).astype(np.uint8)
    np.save(
        str(path),
        np.ascontiguousarray(np.broadcast_to(row[None, :, None], (edge, edge, 4))),
    )
    return path


def produce() -> dict[str, JsonValue]:
    """Write the captures in a scratch session; return the bands route's answer for each.

    A route refusing what it should admit raises.
    """
    from tcip_mcp.dataset_layout import image_dir
    from tests._producer_fixtures import write_band_group, write_image
    from tests._web_fixtures import named_project

    with scratch_session() as (workspace, client):
        root = workspace / "bands"
        opened(client, named_project(root, "Bands"))
        images = image_dir(root, "2026-02-11")
        images.mkdir(parents=True)
        rgba = images / "rgba.tif"
        tifffile.imwrite(
            str(rgba), np.dstack([_ramp(np.uint8, 255)] * 4), photometric="rgb"
        )
        deep_rgba = images / "deep_rgba.tif"
        tifffile.imwrite(
            str(deep_rgba),
            np.dstack([_ramp(np.uint16, 65535)] * 4),
            photometric="rgb",
        )
        captures = {
            "sampled": _past_the_sample_budget(images / "sampled.npy"),
            "four_spectral": write_band_group(
                images,
                "spectral",
                _group(images, "spectral", tuple(SPECTRAL_NM), np.uint16),
                central_wavelength_nm=SPECTRAL_NM,
            ),
            "three_unlabeled": write_band_group(
                images,
                "unlabeled",
                _group(images, "unlabeled", UNLABELED, np.uint8),
                central_wavelength_nm=None,
            ),
            "rgba": rgba,
            "deep_rgba": deep_rgba,
            "plain": write_image(images / "plain.jpg", (FRAME[1], FRAME[0])),
            "single": write_band_group(
                images,
                "single",
                _group(images, "single", SINGLE, np.uint16),
                central_wavelength_nm=None,
            ),
        }
        return normalized(
            {
                scene: answered(
                    client, "get_bands", 200, params={"path": str(path)}
                )
                for scene, path in captures.items()
            },
            workspace,
        )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
