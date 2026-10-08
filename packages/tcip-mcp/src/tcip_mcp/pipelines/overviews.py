"""External overview pyramids (.ovr sidecars) for large rasters.

GDAL serves a reduced-resolution read from the nearest overview level at or above the requested
resolution, so a display-scale read of a huge raster costs the overview's pixels instead of the
native ones. :func:`build_overviews` writes the GDAL-standard external ``.ovr`` next to the raster
(GDAL's ``TIFF_USE_OVR`` option directs the build to the sidecar; the raster itself is never
rewritten);
:func:`overview_dims` and :func:`sidecar_valid` are the checks a caller gates on first.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from tcip_mcp.pipelines.raster_source import open_gdal_dataset, palette_tiff

PYRAMID_FLOOR_EDGE = 1024
"""Longest edge of a pyramid's deepest level, and the edge a raster too large to sample natively
has its display statistics read at, so that read always has a level to come off. Provisional: a
documented choice, not a measurement."""

# How often the parent samples the growing sidecar to report progress and honor a cancel.
_BUILD_POLL_SECONDS = 0.2

_BUILD_CHILD = """
import sys
import rasterio
from rasterio.enums import Resampling

path, levels, palette = sys.argv[1], [int(v) for v in sys.argv[2].split(",")], sys.argv[3] == "1"
with rasterio.Env(TIFF_USE_OVR=True):
    with rasterio.open(path, "r+") as ds:
        ds.build_overviews(levels, Resampling.nearest if palette else Resampling.average)
"""


def overview_sidecar(path: str | Path) -> Path:
    """The external overview sidecar GDAL pairs with ``path``: the full filename plus ``.ovr``."""
    return Path(str(path) + ".ovr")


def overview_dims(path: str | Path) -> list[tuple[int, int]]:
    """The ``(width, height)`` of each reduced-resolution level GDAL can serve for ``path``, finest
    first: internal overviews or an external sidecar :func:`sidecar_valid` confirms, each read off
    the level's own dataset rather than inferred from a factor. Empty when there are none."""
    if overview_sidecar(path).is_file() and not sidecar_valid(path):
        return []
    ds = open_gdal_dataset(path)
    try:
        count = len(ds.overviews(1))
    finally:
        ds.close()
    dims = []
    for overview in range(count):
        level = open_gdal_dataset(path, overview)
        try:
            dims.append((int(level.width), int(level.height)))
        finally:
            level.close()
    return dims


def _predicted_sidecar_bytes(width: int, height: int, count: int,
                             itemsize: int, levels: list[int]) -> int:
    """Uncompressed size of the pyramid's pixels, the denominator progress is reported against.

    A compressed sidecar lands under this, so the reported fraction is a floor on real progress
    rather than a measurement of it; the caller is told 1.0 only once the build actually returns.
    """
    per_level = sum(-(-width // lvl) * -(-height // lvl) for lvl in levels)
    return max(int(per_level * count * itemsize), 1)


def sidecar_valid(path: str | Path) -> bool:
    """Whether ``path``'s external sidecar exists and every tile in it was actually written.

    Header-only (tile byte counts, no pixel decode, milliseconds even on a deep pyramid): a build
    interrupted outside :func:`build_overviews`' own cleanup leaves a structurally complete
    sidecar whose unwritten tiles have zero-length byte counts and read back as silent zeros, so a
    zero-length tile, or a header tifffile cannot parse, marks the sidecar invalid.
    """
    sidecar = overview_sidecar(path)
    if not sidecar.is_file():
        return False
    import tifffile

    try:
        with tifffile.TiffFile(str(sidecar)) as tif:
            for page in tif.pages:
                if any(int(count) == 0 for count in page.databytecounts):
                    return False
    except Exception:  # noqa: BLE001, an unparseable sidecar is invalid, not an error
        return False
    return True


def overview_levels(width: int, height: int) -> list[int]:
    """Power-of-2 decimation levels down to the first whose longest edge, in the whole pixels
    GDAL sizes a level with (the edge over the factor, rounded up), is at most
    :data:`PYRAMID_FLOOR_EDGE`.

    Empty when the raster's own longest edge already is.
    """
    levels: list[int] = []
    factor = 1
    while max(-(-width // factor), -(-height // factor)) > PYRAMID_FLOOR_EDGE:
        factor *= 2
        levels.append(factor)
    return levels


def build_overviews(path: str | Path,
                    *, progress_cb: "Callable[[float], object] | None" = None) -> Path:
    """Build ``path``'s external ``.ovr`` pyramid (the power-of-2 levels :func:`overview_levels`
    derives from the raster's own size, AVERAGE resampling, or NEAREST for a palette raster
    (:func:`~tcip_mcp.pipelines.raster_source.palette_tiff`) whose indices are never averaged)
    and return the sidecar's path.

    Refuses when a valid sidecar or internal overviews already exist (rebuilding a good pyramid
    is minutes of wasted decode) and when the raster has no level to build (an empty level list
    would clear existing overviews instead of building any). An invalid sidecar (see
    :func:`sidecar_valid`) is deleted and rebuilt. On any build failure or cancellation the
    sidecar is deleted: an interrupted build otherwise leaves a valid-looking file whose unwritten
    tiles read back as silent zeros.

    ``progress_cb``, when given, receives the build's completion fraction in ``[0, 1]``; returning
    ``False`` cancels the build.

    The build runs in a child process. rasterio's ``build_overviews`` takes no progress or cancel
    callback, and a pyramid over a multi-gigabyte raster runs far too long to be uninterruptible,
    so the parent reports progress from the sidecar's growth and cancels by terminating the child.
    ``progress_cb`` is consulted once before the child starts, so a caller that cancels immediately
    is honored whatever the raster's size.
    """
    path = Path(path)
    sidecar = overview_sidecar(path)
    if sidecar.exists():
        if sidecar_valid(path):
            raise ValueError(
                f"{sidecar} already holds a valid overview pyramid; refusing to rebuild over it")
        sidecar.unlink()
    if overview_dims(path):
        raise ValueError(f"{path} already carries internal overviews; nothing to build")
    ds = open_gdal_dataset(path)
    try:
        levels = overview_levels(int(ds.width), int(ds.height))
        predicted = _predicted_sidecar_bytes(
            int(ds.width), int(ds.height), int(ds.count),
            np.dtype(ds.dtypes[0]).itemsize, levels)
    finally:
        ds.close()
    if not levels:
        raise ValueError(
            f"{path}'s longest edge is within {PYRAMID_FLOOR_EDGE} pixels; there is no overview "
            "level to build")

    def canceled(fraction: float) -> bool:
        return progress_cb is not None and progress_cb(fraction) is False

    def abandon(reason: str) -> None:
        if sidecar.exists():
            sidecar.unlink()
        raise RuntimeError(reason)

    if canceled(0.0):
        abandon(f"overview build for {path} canceled before it started")

    child = subprocess.Popen(
        [sys.executable, "-c", _BUILD_CHILD, str(path), ",".join(str(v) for v in levels),
         "1" if palette_tiff(path) else "0"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    while child.poll() is None:
        time.sleep(_BUILD_POLL_SECONDS)
        grown = sidecar.stat().st_size if sidecar.exists() else 0
        if canceled(min(grown / predicted, 0.99)):
            child.terminate()
            child.wait(timeout=30)
            abandon(f"overview build for {path} canceled")
    if child.returncode != 0:
        assert child.stderr is not None, "stderr=subprocess.PIPE was passed to Popen above"
        abandon(f"overview build for {path} failed: {(child.stderr.read() or '').strip()}")
    if progress_cb is not None:
        progress_cb(1.0)
    return sidecar
