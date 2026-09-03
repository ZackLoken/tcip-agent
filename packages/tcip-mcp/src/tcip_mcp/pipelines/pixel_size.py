"""The one place the platform turns a raster's georeferencing tags into a real-world pixel size
in metres. Every consumer that needs one (the completeness bar's working-scale derivation, the
block-aware calibration buffer) resolves through :func:`resolve_pixel_size` (or its two thin
wrappers, :func:`raster_pixel_size` and :func:`raster_pixel_size_reason`) so no second reading of
the raster's CRS unit can drift from this one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from tcip_mcp.pipelines.data.band_groups import BandGroupRef


@dataclass(frozen=True)
class PixelSize:
    """One raster's own real-world pixel size, resolved from its georeferencing tags alone
    (:func:`raster_pixel_size`): ``metres_per_px`` is the isotropic pixel edge in metres,
    ``source_clause`` the one-clause description of the geotransform it came from."""

    metres_per_px: float
    source_clause: str


_UNIT_CODES_TO_METRES = frozenset({"9001", "9002", "9003"})
"""``pyproj`` axis ``unit_code`` values this module converts to metres (metre, foot, US survey
foot); judged by code, never by the numeric factor, which ``pyproj`` reports for a degree axis
too."""

_ANISOTROPY_REL_TOL = 1e-6
"""Relative tolerance for treating ``pixel_scale_x``/``pixel_scale_y`` as equal: one part in a
million, so a tag written as ``0.030000001`` beside ``0.03`` reads as equal while any real
anisotropy still refuses. An anisotropic raster has no single pixel size to convert a longer
side through, since :func:`saved_extents` does not record which axis a box's longer side lay
along, so it is refused by name rather than averaged."""


def resolve_pixel_size(source: Path | BandGroupRef) -> tuple[PixelSize | None, str]:
    """The shared implementation behind :func:`raster_pixel_size` and
    :func:`raster_pixel_size_reason`, so the two can never disagree about why a raster carries no
    metres-per-pixel figure: returns ``(pixel_size, reason)``, ``reason`` the empty string on
    success and the first failing condition's clause otherwise, checked in this order:

    1. ``source`` is a raster at all (:func:`~tcip_mcp.pipelines.image_utils.capture_kind`); a
       photographic capture or a band group is never opened here.
    2. :func:`~tcip_mcp.pipelines.postprocessing.orthomosaic_mapping.read_geotransform` returns.
       Its own exceptions are mapped to a short clause never carrying the server's absolute path
       (the geographic case is refused there by the model-type check, which reads as
       :class:`GeoreferencingError` here): :class:`RotatedRasterError` -> "it is rotated or
       sheared", :class:`GeoreferencingError` -> "its georeferencing tags are incomplete"; a
       :class:`ValueError` from ``tifffile`` on a container that is not a TIFF (``.npy``/``.npz``,
       which ``capture_kind`` also calls rasters) -> "it is not a TIFF"; ``OSError`` -> "it could
       not be read".
    3. ``pyproj.CRS.from_epsg(epsg)`` resolves (a :class:`pyproj.exceptions.CRSError` -- a
       user-defined or unknown code -- is the reason).
    4. The CRS is not compound, checked before any unit code: a compound CRS reports an empty
       ``unit_code`` on its horizontal axes too and would otherwise refuse for the wrong reason.
    5. The CRS is projected.
    6. Every axis reports the same ``unit_code`` from ``{"9001", "9002", "9003"}`` (metre, foot,
       US survey foot); the factor to metres is that axis's own ``unit_conversion_factor``.
    7. ``pixel_scale_x`` and ``pixel_scale_y`` are both positive: a zero or negative tag admits no
       real pixel size and would otherwise divide cleanly by zero downstream.
    8. ``pixel_scale_x`` and ``pixel_scale_y`` agree within :data:`_ANISOTROPY_REL_TOL`.
    """
    from tcip_mcp.pipelines.image_utils import capture_kind

    if capture_kind(source) != "raster":
        return None, "it is not a raster"
    assert isinstance(source, Path)  # capture_kind's "raster" answer is Path-only, never a group

    import pyproj

    from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import (
        GeoreferencingError,
        RotatedRasterError,
        read_geotransform,
    )

    try:
        gt = read_geotransform(source)
    except RotatedRasterError:
        return None, "it is rotated or sheared"
    except GeoreferencingError:
        return None, "its georeferencing tags are incomplete"
    except ValueError:
        return None, "it is not a TIFF"
    except OSError:
        return None, "it could not be read"

    try:
        crs = pyproj.CRS.from_epsg(gt.epsg)
    except pyproj.exceptions.CRSError:
        return None, f"its CRS EPSG:{gt.epsg} is user-defined"
    if crs.is_compound:
        return None, f"its CRS EPSG:{gt.epsg} is compound"
    if not crs.is_projected:
        return None, "its georeferencing is not projected"
    codes = {axis.unit_code for axis in crs.axis_info}
    if len(codes) != 1 or next(iter(codes)) not in _UNIT_CODES_TO_METRES:
        return None, f"its CRS EPSG:{gt.epsg}'s units are not metre, foot or US survey foot"
    if gt.pixel_scale_x <= 0 or gt.pixel_scale_y <= 0:
        return None, "its pixel scale is zero or negative"
    if not math.isclose(gt.pixel_scale_x, gt.pixel_scale_y, rel_tol=_ANISOTROPY_REL_TOL):
        return None, "its pixel scales differ by axis"

    factor = crs.axis_info[0].unit_conversion_factor
    metres_per_px = gt.pixel_scale_x * factor
    clause = f"a projected geotransform (EPSG:{gt.epsg}, {metres_per_px:.6g} m/px)"
    return PixelSize(metres_per_px=metres_per_px, source_clause=clause), ""


def raster_pixel_size(source: Path | BandGroupRef) -> PixelSize | None:
    """``source``'s own real-world pixel size, from its georeferencing tags alone, or ``None``
    when ``source`` is not a raster this module can resolve one for (see
    :func:`resolve_pixel_size` for the checks, in order); call
    :func:`raster_pixel_size_reason` for why. Calls
    :func:`~tcip_mcp.pipelines.postprocessing.orthomosaic_mapping.read_geotransform` directly,
    never :func:`~tcip_mcp.pipelines.raster_source.is_georeferenced`, which swallows the reason.

    A photographic capture is never opened, and a ``band_group`` manifest is excluded: neither
    is a single raster this module reads georeferencing tags off of.
    """
    pixel_size, _reason = resolve_pixel_size(source)
    return pixel_size


def raster_pixel_size_reason(source: Path | BandGroupRef) -> str | None:
    """The one-clause reason :func:`raster_pixel_size` returned ``None`` for ``source``, or
    ``None`` when it did not (a pixel size was resolved)."""
    pixel_size, reason = resolve_pixel_size(source)
    return None if pixel_size is not None else reason
