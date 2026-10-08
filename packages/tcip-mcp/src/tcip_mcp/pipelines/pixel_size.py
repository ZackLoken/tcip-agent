"""The resolver from a raster's georeferencing tags to a real-world pixel size in meters
(:func:`resolve_pixel_size`)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from tcip_mcp.pipelines.data.band_groups import BandGroupRef


@dataclass(frozen=True)
class PixelSize:
    """One raster's own real-world pixel size, resolved from its georeferencing tags alone:
    ``meters_per_px`` is the isotropic pixel edge in meters, ``source_clause`` the one-clause
    description of the geotransform it came from."""

    meters_per_px: float
    source_clause: str


_UNIT_CODES_TO_METERS = frozenset({"9001", "9002", "9003"})
"""``pyproj`` axis ``unit_code`` values this module converts to meters (meter, foot, US survey
foot)."""

_ANISOTROPY_REL_TOL = 1e-6
"""Relative tolerance for treating ``pixel_scale_x``/``pixel_scale_y`` as equal: one part in a
million, so a tag written as ``0.030000001`` beside ``0.03`` reads as equal while any real
anisotropy still refuses."""


def meters_per_crs_unit(epsg: int) -> tuple[float | None, str]:
    """The meters one unit of the projected CRS ``epsg`` spans, as ``(factor, reason)``:
    ``reason`` the empty string on success and the first failing condition's clause otherwise.
    ``pyproj.CRS.from_epsg`` resolves it (a :class:`pyproj.exceptions.CRSError`, raised for a
    user-defined or unknown code, is the reason); it is not compound, checked before any unit code
    since a compound CRS reports an empty ``unit_code`` on its horizontal axes too; it is
    projected; and every axis reports the same ``unit_code`` from ``{"9001", "9002", "9003"}``
    (meter, foot, US survey foot), the factor that axis's own ``unit_conversion_factor``."""
    import pyproj

    try:
        crs = pyproj.CRS.from_epsg(epsg)
    except pyproj.exceptions.CRSError:
        return None, f"its CRS EPSG:{epsg} is user-defined"
    if crs.is_compound:
        return None, f"its CRS EPSG:{epsg} is compound"
    if not crs.is_projected:
        return None, "its georeferencing is not projected"
    codes = {axis.unit_code for axis in crs.axis_info}
    if len(codes) != 1 or next(iter(codes)) not in _UNIT_CODES_TO_METERS:
        return None, f"its CRS EPSG:{epsg}'s units are not meter, foot or US survey foot"
    return crs.axis_info[0].unit_conversion_factor, ""


def resolve_pixel_size(source: Path | BandGroupRef) -> tuple[PixelSize | None, str]:
    """``source``'s own real-world pixel size from its georeferencing tags alone, as
    ``(pixel_size, reason)``: ``reason`` the empty string on success and the first failing
    condition's clause otherwise, checked in this order:

    1. ``source`` is a raster at all (:func:`~tcip_mcp.pipelines.image_utils.capture_kind`); a
      photographic capture or a band group is never opened here.
    2. :func:`~tcip_mcp.pipelines.postprocessing.orthomosaic_mapping.read_geotransform` returns.
      Its own exceptions are mapped to a short clause never carrying the server's absolute path:
      :class:`RotatedRasterError` -> "it is rotated or sheared", :class:`GeoreferencingError` ->
      "its georeferencing tags are incomplete"; a :class:`ValueError` from ``tifffile`` on a
      container that is not a TIFF -> "it is not a TIFF"; ``OSError`` -> "it could not be read".
    3. Its CRS spans a known number of meters per unit (:func:`meters_per_crs_unit`).
    4. ``pixel_scale_x`` and ``pixel_scale_y`` are both positive.
    5. ``pixel_scale_x`` and ``pixel_scale_y`` agree within :data:`_ANISOTROPY_REL_TOL`.
    """
    from tcip_mcp.pipelines.image_utils import capture_kind

    if capture_kind(source) != "raster":
        return None, "it is not a raster"
    assert isinstance(source, Path)  # capture_kind's "raster" answer is Path-only, never a group

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

    factor, reason = meters_per_crs_unit(gt.epsg)
    if factor is None:
        return None, reason
    if gt.pixel_scale_x <= 0 or gt.pixel_scale_y <= 0:
        return None, "its pixel scale is zero or negative"
    if not math.isclose(gt.pixel_scale_x, gt.pixel_scale_y, rel_tol=_ANISOTROPY_REL_TOL):
        return None, "its pixel scales differ by axis"

    meters_per_px = gt.pixel_scale_x * factor
    clause = f"a projected geotransform (EPSG:{gt.epsg}, {meters_per_px:.6g} m/px)"
    return PixelSize(meters_per_px=meters_per_px, source_clause=clause), ""
