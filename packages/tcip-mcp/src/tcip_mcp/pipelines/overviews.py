"""External overview pyramids (.ovr sidecars) for large rasters.

GDAL serves a reduced-resolution read from the nearest overview level at or above the requested
resolution, so a display-scale read of a huge raster costs the overview's pixels instead of the
native ones. :func:`build_overviews` writes the GDAL-standard external ``.ovr`` next to the raster
(GDAL's ``TIFF_USE_OVR`` option directs the build to the sidecar; the raster itself is never
rewritten);
:func:`overview_dims` and :func:`sidecar_valid` are the checks a caller holding no reader gates
on first.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, NoReturn

from tcip_mcp.pipelines.raster_source import SourceHeader

PYRAMID_FLOOR_EDGE = 1024
"""Longest edge of a pyramid's deepest level, and the edge a raster too large to sample natively
has its display statistics read at, so that read always has a level to come off: an engineering
bound, a level small enough to read whole in one native pass, chosen and not measured."""

# How often the parent samples the growing sidecar to report progress and honor a cancel.
_BUILD_POLL_SECONDS = 0.2

# The build's one acquisition of the raster: its header and the one GDAL open plan the levels and
# write them, first stating on stdout either the plan's levels or why there are none.
_BUILD_CHILD = """
import json
import sys

import rasterio
from rasterio.enums import Resampling

from tcip_mcp.pipelines.overviews import plan_overview_build

path = sys.argv[1]
with rasterio.Env(TIFF_USE_OVR=True):
    with rasterio.open(path, "r+") as ds:
        plan = plan_overview_build(ds)
        print(json.dumps(plan), flush=True)
        if "levels" in plan:
            ds.build_overviews(plan["levels"],
                               Resampling.nearest if plan["palette"] else Resampling.average)
"""


def overview_sidecar(path: str | Path) -> Path:
    """The external overview sidecar GDAL pairs with ``path``: the full filename plus ``.ovr``."""
    return Path(str(path) + ".ovr")


def overview_dims(path: str | Path) -> list[tuple[int, int]]:
    """The levels of ``path``'s physical pyramid
    (:meth:`~tcip_mcp.pipelines.raster_source.SourceHeader.gdal`'s ``level_dims``). Raises what
    its header read and GDAL's open raise."""
    with SourceHeader(path).gdal() as source:
        return source.level_dims()


def plan_overview_build(ds) -> dict:
    """The build of the pyramid of the raster the open GDAL dataset ``ds`` is (``ds.name``),
    planned over ``ds`` and the raster's TIFF header: ``{"refusal": why}`` when the raster already
    carries internal overviews or has no
    level to build, else its ``levels`` (:func:`overview_levels`) and whether it is a ``palette``
    raster (whose indices are resampled by nearest, never averaged)."""
    path = ds.name
    if ds.overviews(1):
        return {"refusal": f"{path} already carries internal overviews; nothing to build"}
    levels = overview_levels(int(ds.width), int(ds.height))
    if not levels:
        return {"refusal": f"{path}'s longest edge is within {PYRAMID_FLOOR_EDGE} pixels; there "
                           "is no overview level to build"}
    return {"levels": levels, "palette": SourceHeader(path).tiff.palette_lut is not None}


def sidecar_valid(path: str | Path) -> bool:
    """Whether ``path``'s external sidecar exists and every tile in it was actually written, read
    off its header (tile byte counts, no pixel decode): a zero-length tile, which reads back as
    silent zeros, or a header tifffile cannot parse marks the sidecar invalid."""
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
                    *, progress_cb: "Callable[[int], object] | None" = None) -> Path:
    """Build ``path``'s external ``.ovr`` pyramid as :func:`plan_overview_build` plans it and
    return the sidecar's path. The raster is read once, in the child process the build runs in:
    its header and one GDAL open plan and write the levels.

    Refuses (``ValueError``) when a valid sidecar already exists, and with the plan's refusal
    when internal overviews exist or there is no level to build (an empty level list would clear
    existing overviews instead of building any). An invalid sidecar (see :func:`sidecar_valid`)
    is deleted and rebuilt. On any build failure or cancellation the sidecar is deleted: an
    interrupted build otherwise leaves a valid-looking file whose unwritten tiles read back as
    silent zeros.

    ``progress_cb``, when given, receives the sidecar's size in bytes at each poll, ``0`` before
    the child starts, and the finished sidecar's size once the build succeeds; returning ``False``
    while the build runs cancels it. It is consulted once before the child starts, so a caller that
    cancels immediately is honored whatever the raster's size. The child process is what makes a
    running build cancelable: rasterio's ``build_overviews`` takes no progress or cancel
    callback, so progress is the sidecar's growth and a cancel terminates the child.
    """
    import json

    path = Path(path)
    sidecar = overview_sidecar(path)
    if sidecar.exists():
        if sidecar_valid(path):
            raise ValueError(
                f"{sidecar} already holds a valid overview pyramid; refusing to rebuild over it")
        sidecar.unlink()

    def canceled(written: int) -> bool:
        return progress_cb is not None and progress_cb(written) is False

    def abandon(reason: str) -> NoReturn:
        if sidecar.exists():
            sidecar.unlink()
        raise RuntimeError(reason)

    if canceled(0):
        abandon(f"overview build for {path} canceled before it started")

    child = subprocess.Popen(
        [sys.executable, "-c", _BUILD_CHILD, str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert child.stdout is not None and child.stderr is not None, "both were piped above"
    stated = child.stdout.readline()
    if not stated:
        child.wait()
        abandon(f"overview build for {path} failed: {(child.stderr.read() or '').strip()}")
    plan = json.loads(stated)
    if "refusal" in plan:
        child.wait()
        raise ValueError(plan["refusal"])
    while child.poll() is None:
        time.sleep(_BUILD_POLL_SECONDS)
        if canceled(sidecar.stat().st_size if sidecar.exists() else 0):
            child.terminate()
            child.wait(timeout=30)
            abandon(f"overview build for {path} canceled")
    if child.returncode != 0:
        abandon(f"overview build for {path} failed: {(child.stderr.read() or '').strip()}")
    if progress_cb is not None:
        progress_cb(sidecar.stat().st_size)
    return sidecar
