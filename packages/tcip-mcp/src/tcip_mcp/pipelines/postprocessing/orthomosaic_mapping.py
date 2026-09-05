"""Georeferencing for a whole-mosaic GeoTIFF.

A drone orthomosaic covers many plants in one raster (unlike every other image loader in this
package, which assumes one file per plant), so mapping a detection to a real-world plant needs
something no other module here does: reading the GeoTIFF's own georeferencing tags to turn a pixel
location into a real-world coordinate.

:class:`OrthomosaicGeoreference` resolves that pixel <-> real-world mapping, reading the raster's
own tags rather than assuming a CRS or zone: a rotated/sheared raster, or one whose CRS this module
can't determine, is refused rather than silently mis-georeferenced (see
:class:`RotatedRasterError` / :class:`GeoreferencingError`). Reading the pixels themselves is
``pipelines.raster_source``'s job (:class:`~tcip_mcp.pipelines.raster_source.GdalSource` for a
raster too large to decode whole).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Sequence, cast

if TYPE_CHECKING:
    import pyproj
    import tifffile

# GeoTIFF tag IDs this module reads (see the GeoTIFF spec; tifffile exposes each as a plain
# TiffTag keyed by these codes, no prior art for any of them elsewhere in this codebase).
_MODEL_PIXEL_SCALE_TAG = 33550
_MODEL_TIEPOINT_TAG = 33922
_GEO_KEY_DIRECTORY_TAG = 34735
_MODEL_TRANSFORMATION_TAG = 34264

# GeoKeyDirectoryTag key IDs this module resolves out of the flat (key, location, count, value)
# quad array.
_GT_MODEL_TYPE_GEOKEY = 1024
_PROJECTED_CS_TYPE_GEOKEY = 3072
_MODEL_TYPE_PROJECTED = 1

WGS84_EPSG = 4326


class RotatedRasterError(ValueError):
    """A GeoTIFF carries a ``ModelTransformationTag``: a rotated or sheared raster, for which
    the tiepoint + pixel-scale affine this module implements is the wrong math. Refused rather
    than silently mis-georeferencing every pixel this module would otherwise resolve."""


class GeoreferencingError(ValueError):
    """A GeoTIFF is missing (or carries an unusable form of) a tag this module needs to resolve
    a pixel's real-world coordinate: pixel scale, tiepoint, a geokey directory, or a projected
    CRS identified by that directory."""


@dataclass
class GeoTransform:
    """The affine mapping from a GeoTIFF's pixel grid to its native projected CRS.

    ``tiepoint_pixel_*``/``tiepoint_native_*`` are one matched pixel/real-world pair (usually,
    but not necessarily, pixel (0, 0)); ``pixel_scale_*`` is the real-world size of one pixel in
    that CRS's units. ``epsg`` identifies the CRS itself, read from the file rather than assumed.
    """

    tiepoint_pixel_x: float
    tiepoint_pixel_y: float
    tiepoint_native_x: float
    tiepoint_native_y: float
    pixel_scale_x: float
    pixel_scale_y: float
    epsg: int


def read_geotransform(path: str | Path) -> GeoTransform:
    """Read the georeferencing tags from a GeoTIFF's first page.

    Raises :class:`RotatedRasterError` if a ``ModelTransformationTag`` is present, and
    :class:`GeoreferencingError` if the pixel scale, tiepoint, or a projected CRS can't be
    determined from the tags that are.
    """
    import tifffile

    path = Path(path)
    with tifffile.TiffFile(str(path)) as tif:
        # pages[0] is always the keyframe, a full TiffPage, per tifffile's own page-caching
        # contract; only a later page can come back as a lighter TiffFrame sharing its tags.
        page = cast("tifffile.TiffPage", tif.pages[0])
        tags = page.tags
        if _MODEL_TRANSFORMATION_TAG in tags:
            raise RotatedRasterError(
                f"{path}: carries a ModelTransformationTag (rotation/shear). The simple "
                "tiepoint + pixel-scale affine this module implements is wrong for that case; "
                "refusing rather than guessing."
            )
        scale_tag = tags.get(_MODEL_PIXEL_SCALE_TAG)
        tiepoint_tag = tags.get(_MODEL_TIEPOINT_TAG)
        geokey_tag = tags.get(_GEO_KEY_DIRECTORY_TAG)
        if scale_tag is None or tiepoint_tag is None:
            raise GeoreferencingError(
                f"{path}: missing ModelPixelScaleTag and/or ModelTiepointTag; cannot resolve a "
                "pixel to a real-world coordinate without both."
            )
        if geokey_tag is None:
            raise GeoreferencingError(
                f"{path}: missing GeoKeyDirectoryTag; cannot determine the raster's CRS."
            )
        scale_x, scale_y = float(scale_tag.value[0]), float(scale_tag.value[1])
        tiepoint = tiepoint_tag.value
        epsg = _read_projected_epsg(geokey_tag.value, path)

    return GeoTransform(
        tiepoint_pixel_x=float(tiepoint[0]),
        tiepoint_pixel_y=float(tiepoint[1]),
        tiepoint_native_x=float(tiepoint[3]),
        tiepoint_native_y=float(tiepoint[4]),
        pixel_scale_x=scale_x,
        pixel_scale_y=scale_y,
        epsg=epsg,
    )


def _read_projected_epsg(geokeys: tuple[int, ...], path: Path) -> int:
    """Unpack a ``GeoKeyDirectoryTag``'s flat uint16 array and return its EPSG code.

    Layout: a 4-value header (``KeyDirectoryVersion``, ``KeyRevision``, ``MinorRevision``,
    ``NumberOfKeys``) followed by ``NumberOfKeys`` entries of ``(KeyID, TIFFTagLocation, Count,
    ValueOffset)``. A key with ``TIFFTagLocation == 0`` stores its value directly in
    ``ValueOffset``; any other location means the real value lives at that offset inside another
    tag (``GeoDoubleParamsTag``/``GeoAsciiParamsTag``), which this module doesn't resolve since
    ``ProjectedCSTypeGeoKey`` is a SHORT and always stored inline for a conformant file.
    """
    geokeys = tuple(geokeys)
    if len(geokeys) < 4:
        raise GeoreferencingError(f"{path}: GeoKeyDirectoryTag is too short to hold a header.")
    num_keys = int(geokeys[3])
    entries = geokeys[4 : 4 + num_keys * 4]
    if len(entries) < num_keys * 4:
        raise GeoreferencingError(
            f"{path}: GeoKeyDirectoryTag header claims {num_keys} keys but the array is short."
        )

    model_type: int | None = None
    projected_epsg: int | None = None
    for i in range(num_keys):
        key_id, location, _count, value = entries[i * 4 : i * 4 + 4]
        if key_id == _GT_MODEL_TYPE_GEOKEY:
            model_type = int(value)
        elif key_id == _PROJECTED_CS_TYPE_GEOKEY:
            if location != 0:
                raise GeoreferencingError(
                    f"{path}: ProjectedCSTypeGeoKey is stored out-of-line (TIFFTagLocation="
                    f"{location}); this module only reads an inline SHORT value for this key."
                )
            projected_epsg = int(value)

    if model_type != _MODEL_TYPE_PROJECTED:
        raise GeoreferencingError(
            f"{path}: GTModelTypeGeoKey is {model_type!r}, not Projected (1). This module maps "
            "pixels to a projected (e.g. UTM) CRS and refuses a geographic/geocentric/undefined "
            "raster rather than guess how to interpret its coordinates."
        )
    if projected_epsg is None:
        raise GeoreferencingError(f"{path}: GeoKeyDirectoryTag has no ProjectedCSTypeGeoKey.")
    return projected_epsg


def pixel_to_native(transform: GeoTransform, pixel_x: float, pixel_y: float) -> tuple[float, float]:
    """The (easting, northing) in ``transform``'s native projected CRS for pixel (column, row)
    ``(pixel_x, pixel_y)``, via the standard tiepoint + pixel-scale GeoTIFF affine.

    GeoTIFF pixel rows increase downward while northing increases upward, so the y term is
    subtracted where the x term is added; get this backwards and every coordinate this module
    produces is silently flipped north-for-south.
    """
    native_x = transform.tiepoint_native_x + (pixel_x - transform.tiepoint_pixel_x) * transform.pixel_scale_x
    native_y = transform.tiepoint_native_y - (pixel_y - transform.tiepoint_pixel_y) * transform.pixel_scale_y
    return native_x, native_y


def native_to_pixel(transform: GeoTransform, native_x: float, native_y: float) -> tuple[float, float]:
    """The pixel (column, row) for a native-CRS coordinate ``(native_x, native_y)``: the exact
    inverse of :func:`pixel_to_native`.

    Refuses by name when ``transform``'s own ``pixel_scale_x`` or ``pixel_scale_y`` is zero: the
    affine this module implements has no inverse at a degenerate scale, and dividing by it would
    silently produce an infinite or NaN pixel position instead of a real one.
    """
    if transform.pixel_scale_x == 0:
        raise ValueError(
            "native_to_pixel: transform.pixel_scale_x is zero; the tiepoint + pixel-scale affine "
            "has no inverse at a degenerate scale"
        )
    if transform.pixel_scale_y == 0:
        raise ValueError(
            "native_to_pixel: transform.pixel_scale_y is zero; the tiepoint + pixel-scale affine "
            "has no inverse at a degenerate scale"
        )
    pixel_x = transform.tiepoint_pixel_x + (native_x - transform.tiepoint_native_x) / transform.pixel_scale_x
    pixel_y = transform.tiepoint_pixel_y - (native_y - transform.tiepoint_native_y) / transform.pixel_scale_y
    return pixel_x, pixel_y


class OrthomosaicGeoreference:
    """Resolves a real-world coordinate for any pixel in a whole-mosaic GeoTIFF.

    Built once per file (``from_file``) and reused for every subsequent pixel lookup: building a
    ``pyproj.Transformer`` isn't cheap, so it's cached on first use rather than rebuilt per call.
    """

    def __init__(self, transform: GeoTransform):
        self.transform = transform
        self._to_wgs84: "pyproj.Transformer | None" = None

    @classmethod
    def from_file(cls, path: str | Path) -> "OrthomosaicGeoreference":
        return cls(read_geotransform(path))

    def pixel_to_native(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        """(easting, northing) in this raster's own native projected CRS."""
        return pixel_to_native(self.transform, pixel_x, pixel_y)

    def pixel_to_wgs84(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        """(lat, lon) in WGS84 for raster pixel ``(pixel_x, pixel_y)``.

        Returned as (lat, lon) to match ``plant_mapping``'s ``PlantRecord``/``haversine_m``
        convention; ``pyproj.Transformer.transform`` with ``always_xy=True`` itself returns
        (lon, lat), so the swap happens once here rather than leaving every caller to remember
        which order this particular coordinate pair is in.
        """
        import pyproj

        if self._to_wgs84 is None:
            self._to_wgs84 = pyproj.Transformer.from_crs(
                f"EPSG:{self.transform.epsg}", f"EPSG:{WGS84_EPSG}", always_xy=True
            )
        native_x, native_y = self.pixel_to_native(pixel_x, pixel_y)
        lon, lat = self._to_wgs84.transform(native_x, native_y)
        return lat, lon

    def wgs84_to_pixel(self, lat: float, lon: float) -> tuple[float, float]:
        """The pixel (column, row) in this raster for a WGS84 ``(lat, lon)``: the exact inverse
        of :meth:`pixel_to_wgs84`, run through the same cached transformer this instance already
        built (never a second one) with ``direction="INVERSE"``, then :func:`native_to_pixel`.
        """
        import pyproj

        if self._to_wgs84 is None:
            self._to_wgs84 = pyproj.Transformer.from_crs(
                f"EPSG:{self.transform.epsg}", f"EPSG:{WGS84_EPSG}", always_xy=True
            )
        native_x, native_y = self._to_wgs84.transform(lon, lat, direction="INVERSE")
        return native_to_pixel(self.transform, native_x, native_y)


# ── Per-detection plant assignment ──────────────────────────────────────
#
# plant_mapping.assign_plants anchors on a walker's capture *sequence* between separate image
# files: meaningless here, since every detection in one static mosaic frame is simultaneous, so
# there is no "row run" or timestamp order to segment. This is pure point-in/point-out: resolve
# each detection's own pixel location to (lat, lon) via OrthomosaicGeoreference, then the same
# nearest-neighbour primitives plant_mapping already owns (_nearest_plant/haversine_m/
# grid_pitch_m), reused directly rather than reimplemented.

from tcip_mcp.pipelines.postprocessing.plant_mapping import (  # noqa: E402
    NN_TOLERANCE_METERS,
    PlantRecord,
    _nearest_plant,
    grid_pitch_m,
)


@dataclass
class DetectionAssignment:
    """The plant a single detection resolves to, by nearest-neighbour GPS distance.

    ``detection_index`` is the detection's position in the source ``predict_tiled``-shaped
    result's ``boxes``/``scores``/``labels`` lists, so a caller joins this back to the detection
    it came from; ``pixel_x``/``pixel_y`` (the box centroid, in the same full-mosaic pixel space)
    is carried alongside for a caller that wants the location without re-deriving it.

    ``source`` is ``"nearest_neighbour"`` (a plant lies within tolerance) or ``"unmapped"`` (none
    does): unlike ``plant_mapping.Assignment.source``, there is no ``"sequence"`` case, since a
    single static mosaic frame carries no capture order to anchor on. Mirrors ``Assignment``'s own
    honesty: no plant within tolerance is recorded as unmapped, never force-assigned to the
    nearest one regardless of distance, and no fabricated 0-1 confidence, only ``distance_m``
    (see ``plant_mapping``'s module docstring for why a confidence score was removed there).
    """

    detection_index: int
    pixel_x: float
    pixel_y: float
    lat: float
    lon: float
    plot_name: str | None
    accession_name: str | None
    source: str  # "nearest_neighbour" | "unmapped"
    distance_m: float | None

    plant_attribution: ClassVar[str] = "detection"
    """The granularity at which this mapper attributes objects to plants: one detection per
    object, a whole-mosaic frame carrying no capture sequence to attribute at image granularity
    the way ``plant_mapping.MappingBuild`` does. A class attribute, like that one, so it names one
    mapper-wide fact rather than a per-assignment value nothing here computes differently."""


def detection_location(box: Sequence[float]) -> tuple[float, float]:
    """A detection's own location: its box centroid ``((x1+x2)/2, (y1+y2)/2)``, in the same pixel
    space the box itself is stated in, the point the detector's own geometry most directly stands
    for. The one centroid computation :func:`assign_detections_to_plants` and
    :func:`~tcip_mcp.pipelines.postprocessing.segment_attribution.assign_detections_to_segments`
    both call, so per-detection nearest-neighbour attribution and per-detection segment
    containment can never silently disagree about where a detection "is".
    """
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def plants_in_frame(
    plants: list[PlantRecord], georef: OrthomosaicGeoreference, *, width: int, height: int,
) -> tuple[list[PlantRecord], list[PlantRecord]]:
    """``(in_frame, outside)``: ``plants`` partitioned by whether their projected pixel position
    falls inside this raster's own frame, the half-open test ``0 <= px < width`` and
    ``0 <= py < height`` on the position :meth:`OrthomosaicGeoreference.wgs84_to_pixel` projects.

    ``width``/``height`` come from the raster's own verified recorded identity, never assumed:
    the one partition both attribution regimes share, so a registry point the raster does not
    picture is treated the same way whichever regime reads it. The nearest-neighbour delivery
    regime's own candidate list is the in-frame plants only, so a detection near a raster edge
    cannot map to a registry point outside the raster; the canopy segment tie's own containment
    test is meaningless for a plant the raster never pictures, so it partitions the same way
    before testing containment at all.
    """
    in_frame: list[PlantRecord] = []
    outside: list[PlantRecord] = []
    for p in plants:
        px, py = georef.wgs84_to_pixel(p.lat, p.lon)
        if 0 <= px < width and 0 <= py < height:
            in_frame.append(p)
        else:
            outside.append(p)
    return in_frame, outside


def resolve_nn_tolerance_m(
    plants: list[PlantRecord], nn_tolerance_m: float | None = None,
) -> dict:
    """The match tolerance (metres) :func:`assign_detections_to_plants` matches a detection to a
    plant within, and where it came from: ``{"value": float, "source": str}``.

    ``source`` is ``"stated"`` when the caller names a tolerance, ``"grid_pitch"`` when it is
    derived from the plant layout's own spacing (``grid_pitch_m(plants) / 6``), or ``"fallback"``
    (:data:`NN_TOLERANCE_METERS`) when the layout carries too few georeferenced plants (< 2) to
    derive a pitch from.

    A single-detection-per-object mosaic frame accepts a match at the tolerance itself, never
    ``plant_mapping.build_mapping``'s own sequence-anchored ceiling (a stated override capped at a
    sixth of the grid pitch): the two doors' match semantics differ enough that a stated override
    is never capped here, so this stays its own derivation rather than sharing ``build_mapping``'s.
    """
    if nn_tolerance_m is not None:
        return {"value": nn_tolerance_m, "source": "stated"}
    pitch = grid_pitch_m(plants)
    if pitch > 0:
        return {"value": pitch / 6, "source": "grid_pitch"}
    return {"value": NN_TOLERANCE_METERS, "source": "fallback"}


def assign_detections_to_plants(
    detections: dict,
    georeference: OrthomosaicGeoreference,
    plants: list[PlantRecord],
    *,
    nn_tolerance_m: float | None = None,
) -> list[DetectionAssignment]:
    """One :class:`DetectionAssignment` per box in a ``predict_tiled``-shaped ``detections`` result
    (``{"boxes": [[x1, y1, x2, y2], ...], ...}`` in full-mosaic pixel space, as returned by
    :meth:`GenericPredictor.predict_tiled` for either of its source kinds).

    Each detection's own location is its box centroid ``((x1+x2)/2, (y1+y2)/2)``: the point the
    detector's own geometry most directly stands for, resolved to (lat, lon) via
    ``georeference.pixel_to_wgs84``, then matched to the nearest plant in ``plants``.

    ``nn_tolerance_m`` resolves through :func:`resolve_nn_tolerance_m`; ``None`` (the default)
    derives it from the plant layout rather than pinning a constant. A detection farther than the
    tolerance from every plant is unmapped rather than force-assigned to the nearest one.
    """
    boxes = detections.get("boxes") or []
    tolerance_m = resolve_nn_tolerance_m(plants, nn_tolerance_m)["value"]

    out: list[DetectionAssignment] = []
    for i, box in enumerate(boxes):
        cx, cy = detection_location(box)
        lat, lon = georeference.pixel_to_wgs84(cx, cy)
        plant, distance_m = _nearest_plant(lat, lon, plants) if plants else (None, None)
        if plant is not None and distance_m is not None and distance_m <= tolerance_m:
            out.append(DetectionAssignment(
                detection_index=i, pixel_x=cx, pixel_y=cy, lat=lat, lon=lon,
                plot_name=plant.plot_name, accession_name=plant.accession_name,
                source="nearest_neighbour", distance_m=distance_m,
            ))
        else:
            out.append(DetectionAssignment(
                detection_index=i, pixel_x=cx, pixel_y=cy, lat=lat, lon=lon,
                plot_name=None, accession_name=None,
                source="unmapped", distance_m=distance_m,
            ))
    return out
