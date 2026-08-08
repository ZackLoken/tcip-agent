"""External overview pyramids (.ovr sidecars) for large rasters.

GDAL serves a reduced-resolution read from the nearest overview level at or above the requested
resolution, so a display-scale read of a huge raster costs the overview's pixels instead of the
native ones. :func:`build_overviews` writes the GDAL-standard external ``.ovr`` next to the raster
(a read-only open forces the sidecar form; the raster itself is never rewritten);
:func:`has_overviews` and :func:`sidecar_valid` are the checks a caller gates on first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from tcip_mcp.pipelines.display_bounds import DISPLAY_MAX_EDGE
from tcip_mcp.pipelines.raster_source import _gdal, open_gdal_dataset


def overview_sidecar(path: str | Path) -> Path:
    """The external overview sidecar GDAL pairs with ``path``: the full filename plus ``.ovr``."""
    return Path(str(path) + ".ovr")


def has_overviews(path: str | Path) -> bool:
    """Whether GDAL can serve any reduced-resolution level for ``path``: internal overviews or a
    readable external sidecar, both visible as band overview levels on open."""
    ds = open_gdal_dataset(path)
    count = int(ds.GetRasterBand(1).GetOverviewCount())
    del ds
    return count > 0


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
    levels = overview_levels(int(ds.RasterXSize), int(ds.RasterYSize))
    if not levels:
        del ds
        raise ValueError(
            f"{path} already fits the display bound ({DISPLAY_MAX_EDGE}px longest edge); there "
            "is no overview level to build")
    callback = None
    if progress_cb is not None:
        def callback(complete, _message, _data):
            return 0 if progress_cb(float(complete)) is False else 1
    _gdal()  # exceptions must be enabled before BuildOverviews so a failure raises, not returns
    try:
        ds.BuildOverviews("AVERAGE", levels, callback=callback)
    except Exception:
        # Release the dataset's own handle on the sidecar before deleting it.
        del ds
        if sidecar.exists():
            sidecar.unlink()
        raise
    del ds
    return sidecar
