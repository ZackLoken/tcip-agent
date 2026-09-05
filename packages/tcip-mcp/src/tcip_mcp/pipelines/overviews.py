"""External overview pyramids (.ovr sidecars) for large rasters.

GDAL serves a reduced-resolution read from the nearest overview level at or above the requested
resolution, so a display-scale read of a huge raster costs the overview's pixels instead of the
native ones. :func:`build_overviews` writes the GDAL-standard external ``.ovr`` next to the raster
(a read-only open forces the sidecar form; the raster itself is never rewritten);
:func:`has_overviews` and :func:`sidecar_valid` are the checks a caller gates on first.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from tcip_mcp.pipelines.display_bounds import DISPLAY_MAX_EDGE
from tcip_mcp.pipelines.raster_source import open_gdal_dataset

# How often the parent samples the growing sidecar to report progress and honor a cancel.
_BUILD_POLL_SECONDS = 0.2

_BUILD_CHILD = """
import sys
import rasterio
from rasterio.enums import Resampling

path, levels = sys.argv[1], [int(v) for v in sys.argv[2].split(",")]
with rasterio.Env(TIFF_USE_OVR=True):
    with rasterio.open(path, "r+") as ds:
        ds.build_overviews(levels, Resampling.average)
"""


def overview_sidecar(path: str | Path) -> Path:
    """The external overview sidecar GDAL pairs with ``path``: the full filename plus ``.ovr``."""
    return Path(str(path) + ".ovr")


def has_overviews(path: str | Path) -> bool:
    """Whether GDAL can serve any reduced-resolution level for ``path``: internal overviews or a
    readable external sidecar, both visible as band overview levels on open."""
    ds = open_gdal_dataset(path)
    try:
        return bool(ds.overviews(1))
    finally:
        ds.close()


def _predicted_sidecar_bytes(width: int, height: int, count: int,
                             itemsize: int, levels: list[int]) -> int:
    """Uncompressed size of the pyramid's pixels, the denominator progress is reported against.

    A compressed sidecar lands under this, so the reported fraction is a floor on real progress
    rather than a measurement of it; the caller is told 1.0 only once the build actually returns.
    """
    per_level = sum((width // lvl) * (height // lvl) for lvl in levels)
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


def overview_levels(width: int, height: int, *, max_edge: int = DISPLAY_MAX_EDGE) -> list[int]:
    """Power-of-2 decimation levels down to the first whose longest edge fits ``max_edge``.

    Empty when the raster already fits: there is no display-scale resolution an overview level
    would serve.
    """
    levels: list[int] = []
    factor = 2
    while max(width, height) > max_edge:
        levels.append(factor)
        if max(width, height) / factor <= max_edge:
            break
        factor *= 2
    return levels


def build_overviews(path: str | Path,
                    *, progress_cb: "Callable[[float], object] | None" = None) -> Path:
    """Build ``path``'s external ``.ovr`` pyramid (AVERAGE resampling, power-of-2 levels down to
    the display bound) and return the sidecar's path.

    Refuses when a valid sidecar or internal overviews already exist (rebuilding a good pyramid
    is minutes of wasted decode) and when the raster already fits the display bound (an empty
    level list would clear existing overviews instead of building any). An invalid sidecar (see
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
    if has_overviews(path):
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
            f"{path} already fits the display bound ({DISPLAY_MAX_EDGE}px longest edge); there "
            "is no overview level to build")

    def cancelled(fraction: float) -> bool:
        return progress_cb is not None and progress_cb(fraction) is False

    def abandon(reason: str) -> None:
        if sidecar.exists():
            sidecar.unlink()
        raise RuntimeError(reason)

    if cancelled(0.0):
        abandon(f"overview build for {path} cancelled before it started")

    child = subprocess.Popen(
        [sys.executable, "-c", _BUILD_CHILD, str(path), ",".join(str(v) for v in levels)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    while child.poll() is None:
        time.sleep(_BUILD_POLL_SECONDS)
        grown = sidecar.stat().st_size if sidecar.exists() else 0
        if cancelled(min(grown / predicted, 0.99)):
            child.terminate()
            child.wait(timeout=30)
            abandon(f"overview build for {path} cancelled")
    if child.returncode != 0:
        assert child.stderr is not None, "stderr=subprocess.PIPE was passed to Popen above"
        abandon(f"overview build for {path} failed: {(child.stderr.read() or '').strip()}")
    if progress_cb is not None:
        progress_cb(1.0)
    return sidecar
