"""Georeferencing and windowed pixel access for a whole-mosaic GeoTIFF.

A drone orthomosaic covers many plants in one raster (unlike every other image loader in this
package, which assumes one file per plant), so mapping a detection to a real-world plant needs
two things neither of which exist elsewhere in the codebase: reading the GeoTIFF's own
georeferencing tags to turn a pixel location into a real-world coordinate, and reading a pixel
window from a file too large to decode in one call.

:class:`OrthomosaicGeoreference` resolves the pixel <-> real-world mapping; :class:`OrthomosaicWindowReader`
serves cheap, repeated windowed reads of the pixel data itself. Both read the raster's own tags
rather than assuming a CRS or zone: a rotated/sheared raster, or one whose CRS this module can't
determine, is refused rather than silently mis-georeferenced (see :class:`RotatedRasterError` /
:class:`GeoreferencingError`).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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


class UnsupportedRasterLayout(ValueError):
    """A GeoTIFF's on-disk pixel layout, an internally tiled raster or planar (band-separate)
    samples, isn't one :class:`OrthomosaicWindowReader` decodes. It only reads the strip-based,
    contiguous-sample layout the mosaic this module was built for actually uses."""


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
        tags = tif.pages[0].tags
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


class OrthomosaicGeoreference:
    """Resolves a real-world coordinate for any pixel in a whole-mosaic GeoTIFF.

    Built once per file (``from_file``) and reused for every subsequent pixel lookup: building a
    ``pyproj.Transformer`` isn't cheap, so it's cached on first use rather than rebuilt per call.
    """

    def __init__(self, transform: GeoTransform):
        self.transform = transform
        self._to_wgs84 = None

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


class OrthomosaicWindowReader:
    """Opens a whole-mosaic GeoTIFF once and serves cheap, repeated pixel-window reads.

    Decodes only the strips a requested window actually overlaps, never the whole file, so one
    instance composes with a tiling loop that requests many windows across the raster instead of
    reopening/reparsing the file per tile. Use as a context manager or call :meth:`close`
    explicitly; the underlying file handle stays open for the reader's lifetime.

    Only a strip-based raster with contiguous (chunky) samples is supported, the layout the
    orthomosaic this module was built for actually uses; an internally tiled TIFF or one with
    planar (band-separate) samples raises :class:`UnsupportedRasterLayout`. Decoding also
    depends on this environment having a working codec for the file's compression: LZW and JPEG
    need the optional ``imagecodecs`` package, which is not part of this environment.

    A real orthomosaic's strips run the raster's full width at a small ``rowsperstrip``, so a
    tiling loop that scans left to right across one row-band before advancing (``tile_positions``'
    own order) requests the *same* strips again for every tile in that band. Decoded strips are
    cached (an LRU keyed by strip index) so a strip already decoded to serve one window is sliced
    from memory for the next, rather than re-read from disk and re-decoded; the cache is capped
    well below the file's full strip count so it never approaches holding the whole raster, sized
    generously for one row-band's worth of strips rather than tuned to this file's own strip
    count. A caller whose access pattern isn't row-major still gets correct results, just fewer
    cache hits.
    """

    def __init__(self, path: str | Path, *, strip_cache_capacity: int = 64):
        import tifffile

        self._path = Path(path)
        self._tif = tifffile.TiffFile(str(self._path))
        page = self._tif.pages[0]
        if page.is_tiled:
            self._tif.close()
            raise UnsupportedRasterLayout(
                f"{self._path}: internally tiled TIFF; this reader only decodes strip-based rasters."
            )
        if page.planarconfig != 1:
            self._tif.close()
            raise UnsupportedRasterLayout(
                f"{self._path}: planar (band-separate) sample layout; this reader only decodes "
                "contiguous (chunky) samples."
            )
        self._page = page
        self.height = int(page.imagelength)
        self.width = int(page.imagewidth)
        self.num_channels = int(page.samplesperpixel)
        self.dtype = page.dtype
        self._rows_per_strip = int(page.rowsperstrip)
        self._strip_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._strip_cache_capacity = strip_cache_capacity
        self.strip_decode_count = 0  # cache misses actually decoded from disk; a caching test's probe

    def close(self) -> None:
        self._strip_cache.clear()
        self._tif.close()

    def __enter__(self) -> "OrthomosaicWindowReader":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def read_window(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Decode and return the ``[y1-y0, x1-x0, num_channels]`` pixel window ``[y0:y1, x0:x1]``.

        Raises ``ValueError`` for an empty or out-of-bounds window; a caller that wants a padded
        edge tile clips the request to the raster's bounds first and pads the result separately
        (mirrors ``image_utils.crop_pad_tile``, which this reader doesn't duplicate).
        """
        if not (0 <= y0 < y1 <= self.height) or not (0 <= x0 < x1 <= self.width):
            raise ValueError(
                f"window [{y0}:{y1}, {x0}:{x1}] is out of bounds for a "
                f"{self.height}x{self.width} raster"
            )
        rps = self._rows_per_strip
        strip0 = y0 // rps
        strip1 = (y1 - 1) // rps
        if strip0 == strip1:
            block = self._decode_strip(strip0)
            row_offset = strip0 * rps
            # A copy, not a view: `block` is a cached, reused strip array, and a caller mutating
            # its returned window must not corrupt what a later window reads back.
            return np.array(block[y0 - row_offset : y1 - row_offset, x0:x1])
        # A real orthomosaic's strips run the raster's full width, so slice each strip down to
        # the requested x-range before concatenating: concatenating full-width strips just to
        # discard everything outside x0:x1 copies orders of magnitude more data than the window
        # itself needs. `np.concatenate` always allocates a new array, so this is a copy on its
        # own, same as the single-strip branch's explicit one.
        pieces = []
        for strip_index in range(strip0, strip1 + 1):
            strip = self._decode_strip(strip_index)
            row_offset = strip_index * rps
            row_lo = max(y0, row_offset) - row_offset
            row_hi = min(y1, row_offset + strip.shape[0]) - row_offset
            pieces.append(strip[row_lo:row_hi, x0:x1])
        return np.concatenate(pieces, axis=0)

    def _decode_strip(self, strip_index: int) -> np.ndarray:
        """Return strip ``strip_index``'s decoded pixel rows, from the LRU cache if a previous
        window already decoded it, decoding from disk only on a genuine miss.
        """
        cached = self._strip_cache.get(strip_index)
        if cached is not None:
            self._strip_cache.move_to_end(strip_index)
            return cached
        fh = self._page.parent.filehandle
        fh.seek(self._page.dataoffsets[strip_index])
        data = fh.read(self._page.databytecounts[strip_index])
        segment, _position, _shape = self._page.decode(data, strip_index)
        decoded = segment[0]
        self._strip_cache[strip_index] = decoded
        if len(self._strip_cache) > self._strip_cache_capacity:
            self._strip_cache.popitem(last=False)
        self.strip_decode_count += 1
        return decoded


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


def assign_detections_to_plants(
    detections: dict,
    georeference: OrthomosaicGeoreference,
    plants: list[PlantRecord],
    *,
    nn_tolerance_m: float | None = None,
) -> list[DetectionAssignment]:
    """One :class:`DetectionAssignment` per box in a ``predict_tiled``-shaped ``detections`` result
    (``{"boxes": [[x1, y1, x2, y2], ...], ...}`` in full-mosaic pixel space, as returned by
    :meth:`GenericPredictor.predict_tiled` / ``predict_tiled_from_reader``).

    Each detection's own location is its box centroid ``((x1+x2)/2, (y1+y2)/2)``: the point the
    detector's own geometry most directly stands for, resolved to (lat, lon) via
    ``georeference.pixel_to_wgs84``, then matched to the nearest plant in ``plants``.

    ``nn_tolerance_m`` defaults to the same derivation ``plant_mapping.build_mapping`` already
    uses for its own ``nn_tolerance_m`` (mirrored exactly rather than a second formula for the same
    idea): ``grid_pitch_m(plants) / 6``, or the honest ``NN_TOLERANCE_METERS`` fallback when the
    plant layout has too few georeferenced plants to derive a pitch from (< 2). A detection farther
    than the tolerance from every plant is unmapped rather than force-assigned to the nearest one.
    """
    boxes = detections.get("boxes") or []
    if nn_tolerance_m is None:
        pitch = grid_pitch_m(plants)
        nn_tolerance_m = (pitch / 6) if pitch > 0 else NN_TOLERANCE_METERS

    out: list[DetectionAssignment] = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        lat, lon = georeference.pixel_to_wgs84(cx, cy)
        plant, distance_m = _nearest_plant(lat, lon, plants) if plants else (None, None)
        if plant is not None and distance_m is not None and distance_m <= nn_tolerance_m:
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
