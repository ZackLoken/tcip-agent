"""Plant-ID mapping across capture dates: each image assigned a plant by its capture sequence
along a row and its GPS, or by nearest-neighbor GPS where no sequence anchors it, every assignment
recording its match ``source`` and its GPS ``distance_m``. A mapping is named project state, bound
to the dataset it was built over and to the build receipt that names it."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import re
import statistics
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING, Any, ClassVar, Iterable, Literal, NamedTuple, Optional, Sequence, get_args,
)

import tcip_store
from PIL import ExifTags, Image
from pydantic import ConfigDict, TypeAdapter
from tcip_store import Key, encode_record

from tcip_mcp.audit import now_iso
from tcip_mcp.project_paths import project_state_dir

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tcip_mcp.buckets import Bucket
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_378_137.0

SEQUENCE_MATCH_FACTOR = 2
"""The sequence-anchored gate: a run's nearest unclaimed plant is accepted out to this many times
nn_tolerance_m before falling through to plain nearest-neighbor."""

NEAREST_MATCH_FACTOR = 3
"""The plain nearest-neighbor gate, the loosest a match is ever accepted at: this many times
nn_tolerance_m."""


@dataclass
class ImageStamp:
    """Per-capture metadata: EXIF for an ``image``, structural facts for a ``band_group`` or
    ``raster``. ``name`` is the capture's file name, ``readable`` whether an ``image``'s EXIF read
    (``None`` for the other kinds), and ``manifest_sha256``/``members`` a ``band_group``'s own."""

    path: str
    stem: str
    date_folder: str
    kind: str
    name: str
    timestamp: Optional[datetime]
    lat: Optional[float]
    lon: Optional[float]
    h_pos_err: Optional[float]
    readable: Optional[bool]
    manifest_sha256: Optional[str] = None
    members: tuple[str, ...] = ()

    @property
    def position(self) -> tuple[float, float] | None:
        """``(lat, lon)``, or ``None`` when the stamp lacks either."""
        return None if self.lat is None or self.lon is None else (self.lat, self.lon)


@dataclass
class PlantRecord:
    """One plant from a plant_locations CSV."""

    plot_name: str
    accession_name: str
    plot_number: Optional[float]
    row_number: Optional[float]
    col_number: Optional[float]
    lat: float
    lon: float


_DECODED = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

GpsSource = Literal["sequence", "nearest_neighbor", "unmapped"]
"""Where a GPS assignment's plant came from: the capture sequence, the plain nearest plant, or
none within the gate."""

UnattributedSegmentSource = Literal["outside_segments", "overlapping_segments",
                                    "segment_without_plant"]
"""Why a canopy-segment assignment names no plant: no segment holds the detection, several do, or
the one that does is tied to no plant."""

SegmentSource = Literal["segment_containment", UnattributedSegmentSource]
"""Where a canopy-segment assignment's plant came from: one tied segment containing the detection,
or why none did."""

UNATTRIBUTED_SEGMENT_SOURCES: tuple[UnattributedSegmentSource, ...] = get_args(
    UnattributedSegmentSource)

ToleranceSource = Literal["grid_pitch", "stated", "stated_capped"]
"""Where a match tolerance came from (:func:`resolve_nn_tolerance_m`): the plant grid's pitch,
the stated value, or the stated value capped to the grid pitch."""


@dataclass
class Assignment:
    """The mapping we produce for a single image, named by its file name in its date folder;
    ``distance_m`` is the GPS distance (m) to the nearest plant, ``None`` for a stamp with no
    position."""

    __pydantic_config__ = _DECODED

    image: str
    stem: str
    date_folder: str
    plot_name: Optional[str]
    accession_name: Optional[str]
    source: GpsSource
    distance_m: Optional[float]

    @classmethod
    def of(cls, stamp: "ImageStamp", plant: "PlantRecord | None", source: GpsSource,
           distance_m: float | None) -> "Assignment":
        """``stamp`` assigned to ``plant`` (``None`` for no plant) from ``source``."""
        return cls(image=stamp.name, stem=stamp.stem, date_folder=stamp.date_folder,
                   plot_name=plant.plot_name if plant else None,
                   accession_name=plant.accession_name if plant else None,
                   source=source, distance_m=distance_m)


def assignment_is_attributed(assignment: object) -> bool:
    """Whether ``assignment`` (any assignment record, or the plain dict row ``MappingBuild.rows``
    produces) names a real plant: a non-empty ``plot_name``, the rule a plant CSV's own blank name
    column and an unmapped capture (``plot_name=None``) both fail by.
    """
    plot_name = attr(assignment, "plot_name")
    return isinstance(plot_name, str) and plot_name != ""


def require_named_plants(plants: list[PlantRecord]) -> None:
    """Refuse (``ValueError``) a registry carrying a blank ``plot_name``
    (:func:`assignment_is_attributed`'s rule) or a duplicate one, naming the offending accession
    or names."""
    for p in plants:
        if not assignment_is_attributed({"plot_name": p.plot_name}):
            raise ValueError(
                f"the plant registry carries a blank plot_name (accession {p.accession_name!r}); "
                "every plant this delivery attributes detections to must carry a plot_name"
            )
    names = [p.plot_name for p in plants]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"the plant registry carries duplicate plot_name(s) {duplicates}; two rows sharing "
            "one identity would merge two trees' detections into one aggregation row"
        )


@dataclass
class MappingBuild:
    """One build's provenance plus its per-date assignments: the whole persisted record, in memory.

    ``dataset_id`` is the dataset identity record's minted id; ``dataset_root`` is spelled
    against the owning project by :func:`~tcip_mcp.registry_paths.stored_path`.
    ``capture_digests`` is ``capture_identity``'s per-capture counterpart
    (:func:`capture_digests`, one entry per stem, derived from the same row builder).
    """

    __pydantic_config__ = _DECODED

    name: str
    dataset_root: str
    dataset_id: str
    built_at: str
    dates_requested: Optional[list[str]]
    dates: list[str]
    nn_tolerance_m: dict
    plant_registry: dict
    """The named :data:`PLANT_REGISTRY_STORE` record this build's plants were read from:
    ``{"name": ..., "digest": ...}``."""
    capture_identity: dict[str, str]
    capture_digests: dict[str, dict[str, str]]
    unreadable: dict[str, list[str]]
    assignments: dict[str, list[Assignment]]
    supersedes: Optional[str] = None
    """The archived digest this record replaced, when a same-name rebuild was told to
    ``supersede`` a record a delivery event still cited; ``None`` otherwise."""
    record_sha256: str = ""
    """The digest of the stored record this build was decoded from (:meth:`from_record`); blank
    on a build not yet read back. Never written into the record itself."""

    plant_attribution: ClassVar[str] = "image"
    """The granularity at which this mapping's own assignments attribute objects to plants: one
    frame per plant, by the capture protocol the platform assumes for a walked, EXIF-geolocated
    scene. Never a field of the record."""

    def to_record(self) -> dict:
        """The stored document: every field but ``record_sha256``."""
        return _MAPPING_RECORD.dump_python(self, exclude={"record_sha256"})

    @classmethod
    def from_record(cls, raw: object, project: Path | str, name: str) -> "MappingBuild":
        """The build the stored record ``raw`` under ``name`` holds, carrying the record's own
        digest (:func:`record_digest`). A document that does not decode as this record, or whose
        ``capture_identity`` names a date ``capture_digests`` does not, raises ``ValueError``
        naming the project and the name."""
        from pydantic import ValidationError

        problem = f"plant mapping {name!r} under {project} is not a record this reader decodes"
        try:
            build = _MAPPING_RECORD.validate_json(encode_record(raw))
        except ValidationError as exc:
            raise ValueError(f"{problem}: {exc}") from exc
        undigested = sorted(set(build.capture_identity) - set(build.capture_digests))
        if undigested:
            raise ValueError(f"{problem}: capture_identity names {undigested}, which "
                             "capture_digests carries no digest map for")
        return replace(build, record_sha256=record_digest(raw))

    def rows(self) -> dict[str, list[dict]]:
        """The assignments as plain per-date dict rows, each a fresh dict carrying this build's
        own ``plant_attribution``."""
        return {
            date: [{**a.__dict__, "plant_attribution": self.plant_attribution} for a in assignments]
            for date, assignments in self.assignments.items()
        }

    def unattributed(self, dates: Optional[Iterable[str]] = None) -> int:
        """The number of assignments over ``dates`` (every date this mapping holds, when ``None``)
        for which :func:`assignment_is_attributed` is false.
        """
        scope = self.dates if dates is None else dates
        return sum(
            1
            for date in scope
            for a in self.assignments.get(date, [])
            if not assignment_is_attributed(a)
        )

    def summary(self) -> dict:
        """This build's own per-date and total counts: images, mapped, unattributed, and the mean
        GPS match distance (``None`` for a date with no recorded distance).
        """
        per_date: dict[str, dict] = {}
        total_images = 0
        total_mapped = 0
        total_unattributed = 0
        for date in self.dates:
            assignments = self.assignments.get(date, [])
            n_images = len(assignments)
            n_unattributed = self.unattributed([date])
            n_mapped = n_images - n_unattributed
            dists = [a.distance_m for a in assignments if a.distance_m is not None]
            per_date[date] = {
                "n_images": n_images,
                "n_mapped": n_mapped,
                "n_unattributed": n_unattributed,
                "avg_distance_m": (round(sum(dists) / len(dists), 2) if dists else None),
            }
            total_images += n_images
            total_mapped += n_mapped
            total_unattributed += n_unattributed
        return {
            "per_date": per_date,
            "totals": {
                "n_dates": len(self.dates),
                "n_images": total_images,
                "n_mapped": total_mapped,
                "n_unattributed": total_unattributed,
            },
        }

    def served(self) -> dict:
        """This build as a door answers it: its ``name``, :meth:`summary`, the captures PIL could
        not open per date, the tolerance record and that tolerance's loosest accepted distance
        (:func:`match_gates`)."""
        return {
            "name": self.name,
            "summary": self.summary(),
            "unreadable": self.unreadable,
            "nn_tolerance_m": self.nn_tolerance_m,
            "max_match_distance_m": match_gates(
                self.nn_tolerance_m["value"])["max_match_distance_m"],
        }

    def delivery_disclosure(self, verified: dict, dates: Iterable[str]) -> dict:
        """The ``plant_mapping`` dict a phenology delivery carries: this build's own identity,
        ``verify_mapping_inputs``'s disclosure, and this delivery's own unattributed-capture count
        scoped to ``dates`` (a delivery's own delivered dates, never the mapping's full span).
        """
        dates_delivered = sorted(dates)
        return {
            "name": self.name,
            "dataset_id": self.dataset_id,
            "dataset_root": self.dataset_root,
            "built_at": self.built_at,
            "record_sha256": self.record_sha256,
            "nn_tolerance_m": self.nn_tolerance_m,
            "capture_identity": self.capture_identity,
            "captures_unverified": verified["captures_unverified"],
            "plant_csvs_unverified": verified["plant_csvs_unverified"],
            "dates_delivered": dates_delivered,
            "images_unattributed": self.unattributed(dates_delivered),
            "images_unattributed_scope": "delivered_dates",
            "plant_attribution": self.plant_attribution,
        }


_MAPPING_RECORD = TypeAdapter(MappingBuild)


# ── EXIF extraction ──────────────────────────────────────────────────────

_GPS_IFD_TAG = 0x8825


def _exif_dms_to_decimal(dms, ref, *, axis: str, negative: str, positive: str,
                         path: Path) -> Optional[float]:
    """Degrees, minutes and seconds with their hemisphere reference as signed decimal degrees, or
    ``None`` when ``dms`` is absent. Raises ``ValueError`` naming ``path`` for a value that is not
    three numbers or a reference that is not ``negative`` or ``positive``."""
    if dms is None:
        return None
    try:
        d, m, s = (float(x) for x in dms)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}'s GPS {axis} {dms!r} is not degrees, minutes and seconds"
        ) from exc
    if ref not in (negative, positive):
        raise ValueError(f"{path}'s GPS {axis} reference {ref!r} is neither {negative!r} nor "
                         f"{positive!r}, so its hemisphere is not stated")
    val = d + m / 60 + s / 3600
    return -val if ref == negative else val


def read_image_stamp(path: Path, date_folder: str) -> ImageStamp:
    """One ``image`` capture's EXIF stamp.

    The ``try`` covers ``Image.open`` alone: a capture PIL cannot open (a HEIC with no decoder
    installed, a locked file) becomes a stamp with ``readable=False``; an image that opens and
    carries no EXIF stays ``readable=True`` with ``None`` fields. A capture time, GPS coordinate or
    positioning error present in malformed form, and a GPS coordinate whose hemisphere reference is
    missing, raise ``ValueError`` naming the image.
    """
    from tcip_mcp.pipelines.image_utils import exif_capture_time, parse_capture_time

    stamp = ImageStamp(
        path=str(path), stem=path.stem, date_folder=date_folder, kind="image", name=path.name,
        timestamp=None, lat=None, lon=None, h_pos_err=None, readable=True,
    )
    try:
        im = Image.open(path)
    except Exception:
        stamp.readable = False
        return stamp
    with im:
        exif = im.getexif()
        dt_raw = exif_capture_time(exif)
        if dt_raw is not None:
            stamp.timestamp = parse_capture_time(dt_raw)
            if stamp.timestamp is None:
                raise ValueError(f"{path}'s capture time {dt_raw!r} does not read")

        gps_raw = exif.get_ifd(_GPS_IFD_TAG)
        if gps_raw:
            gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
            stamp.lat = _exif_dms_to_decimal(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"),
                                             axis="latitude", negative="S", positive="N",
                                             path=path)
            stamp.lon = _exif_dms_to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"),
                                             axis="longitude", negative="W", positive="E",
                                             path=path)
            hpe = gps.get("GPSHPositioningError")
            if hpe is not None:
                try:
                    stamp.h_pos_err = float(hpe)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path}'s GPS positioning error {hpe!r} is not a number") from exc
    return stamp


def _read_date_stamps(
    logical: dict[str, "Path | BandGroupRef"], date_folder: str
) -> list[ImageStamp]:
    """A date's stamps for every logical capture ``list_logical_images`` enumerated: EXIF for an
    ``image``, the manifest's own digest and member names for a ``band_group``, bare identity for
    a ``raster`` (no EXIF to read for either)."""
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.image_utils import capture_kind

    stamps: list[ImageStamp] = []
    for stem in sorted(logical):
        source = logical[stem]
        kind = capture_kind(source)
        if kind == "band_group":
            assert isinstance(source, BandGroupRef)
            manifest_sha256 = hashlib.sha256(source.manifest_path.read_bytes()).hexdigest()
            members = tuple(sorted(p.name for p in source.bands.values()))
            stamps.append(ImageStamp(
                path=str(source.manifest_path), stem=stem, date_folder=date_folder, kind=kind,
                name=source.manifest_path.name, timestamp=None, lat=None, lon=None,
                h_pos_err=None, readable=None, manifest_sha256=manifest_sha256, members=members,
            ))
        elif kind == "raster":
            p = source
            assert isinstance(p, Path)
            stamps.append(ImageStamp(
                path=str(p), stem=stem, date_folder=date_folder, kind=kind, name=p.name,
                timestamp=None, lat=None, lon=None, h_pos_err=None, readable=None,
            ))
        else:
            assert isinstance(source, Path)
            stamps.append(read_image_stamp(source, date_folder))
    return stamps


def _capture_row(s: ImageStamp) -> list[object]:
    """The fields one capture's identity commits to: a manifest's own digest and the member names
    it claims for a ``band_group``, bare identity for a ``raster`` (neither carries EXIF), EXIF for
    an ``image``.
    """
    if s.kind == "band_group":
        return [s.name, s.kind, s.manifest_sha256, list(s.members)]
    if s.kind == "raster":
        return [s.name, s.kind]
    return [
        s.name, s.kind,
        s.timestamp.isoformat() if s.timestamp else None,
        s.lat, s.lon, s.h_pos_err, s.readable,
    ]


def _row_digest(row: Sequence[object]) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=False).encode("utf-8")).hexdigest()[:16]


def capture_identity(stamps: list[ImageStamp]) -> str:
    """The sha256[:16] over every capture ``list_logical_images`` enumerated for one date.

    An EXIF/manifest identity, never an image-content identity: it certifies the inputs the
    assignment was made from, not the pixels a phenotype was counted over (the prediction bucket's
    own stamps carry that).
    """
    rows = [_capture_row(s) for s in sorted(stamps, key=lambda s: s.name)]
    return _row_digest(rows)


def capture_digests(stamps: list[ImageStamp]) -> dict[str, str]:
    """One digest per capture, keyed by stem, over that capture's own :func:`_capture_row` alone:
    ``capture_identity``'s per-capture counterpart.
    """
    return {s.stem: _row_digest(_capture_row(s)) for s in stamps}


# ── Plant CSV parsing ───────────────────────────────────────────────────


PLANT_CSV_COLUMN_FOR_FIELD = {
    "plot_name": "plot_name",
    "accession_name": "accession_name",
    "plot_number": "plot_number",
    "row_number": "row_number",
    "col_number": "col_number",
    "lon": "WGS84_centroid_x",
    "lat": "WGS84_centroid_y",
}
"""Which CSV column carries each :class:`PlantRecord` field, in header order."""

PLANT_CSV_COLUMNS = list(PLANT_CSV_COLUMN_FOR_FIELD.values())
"""The seven column names in header order."""


def read_plant_csv_bytes(data: bytes) -> list[PlantRecord]:
    """Parse one plant-locations CSV's own bytes into :class:`PlantRecord` rows, a row with no
    numeric lat/lon skipped. Bytes that are not UTF-8 raise ``UnicodeDecodeError``."""
    import io

    records: list[PlantRecord] = []
    text = data.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    column = PLANT_CSV_COLUMN_FOR_FIELD
    for row in reader:
        try:
            lat = float(row[column["lat"]])
            lon = float(row[column["lon"]])
        except Exception:
            continue
        records.append(
            PlantRecord(
                plot_name=row.get(column["plot_name"], ""),
                accession_name=row.get(column["accession_name"], ""),
                plot_number=_maybe_float(row.get(column["plot_number"])),
                row_number=_maybe_float(row.get(column["row_number"])),
                col_number=_maybe_float(row.get(column["col_number"])),
                lat=lat,
                lon=lon,
            )
        )
    return records


def read_plant_csvs(paths: Iterable[Path]) -> list[PlantRecord]:
    """Read every ``paths`` entry that exists through :func:`read_plant_csv_bytes`, silently
    skipping a path that is not a file. A plant-locations shapefile is not one of these paths:
    convert it first with :func:`read_plant_shapefile`'s own CLI, ``tcip shp-to-plant-csv``."""
    records: list[PlantRecord] = []
    for path in paths:
        p = Path(path)
        if not p.is_file():
            continue
        records.extend(read_plant_csv_bytes(p.read_bytes()))
    return records


class ShapefileRows(NamedTuple):
    """One :func:`read_plant_shapefile` call's rows and bookkeeping. ``rows`` carries
    :data:`PLANT_CSV_COLUMNS`' seven keys per feature: the five attribute columns as the DBF
    value's ``str()`` verbatim (empty when the field is unresolved, exactly the rule
    :func:`_shapefile_field_value` states), and ``WGS84_centroid_x``/``WGS84_centroid_y`` as the
    reprojected floats. ``missing_fields`` names every CSV column with no resolvable source
    attribute; ``geometry_kinds`` is every geometry type actually read (``"point"``,
    ``"polygon"``, ``"multipolygon"``); ``skipped_null_geometry`` counts features with no
    geometry at all, invisible in ``rows``."""

    rows: list[dict[str, object]]
    missing_fields: list[str]
    geometry_kinds: list[str]
    skipped_null_geometry: int


_DEFAULT_SHAPEFILE_FIELD_MAP = {
    "plot_name": "plot_name",
    "accession_name": "accession_name",
    "plot_number": "plot_number",
    "row_number": "row_number",
    "col_number": "col_number",
}
"""Optional pass-through columns' default source-attribute field names, overridable per
shapefile through :func:`read_plant_shapefile`'s own ``field_map``: an ESRI Shapefile DBF caps
field names at 10 characters, so a source rarely matches these exactly."""


def _shapefile_field_value(properties: dict, field_name: Optional[str]) -> str:
    if field_name is None:
        return ""
    value = properties.get(field_name)
    return "" if value is None else str(value)


def read_plant_shapefile(
    path: Path | str, *, field_map: Optional[dict[str, str]] = None,
) -> ShapefileRows:
    """Read a plant-locations shapefile's own features into :data:`PLANT_CSV_COLUMNS`-shaped rows,
    reprojecting every feature's own coordinate to WGS84 with ``always_xy=True`` (GDAL 3's
    authority-compliant EPSG:4326 axis order is lat/lon). ``pyproj.Transformer.from_crs`` takes the
    layer's own CRS object directly, so a custom projected CRS with a valid ``.prj`` converts.

    A point's own coordinate is read; a polygon's or multipolygon's centroid is read; any other
    geometry type refuses by name (naming the feature index and ``geom_type``). A feature with null
    geometry is skipped and counted in the return's ``skipped_null_geometry``. Raises
    :class:`ShapefileCrsUnknownError` when the layer's CRS cannot be resolved, and
    :class:`ShapefileUnreadableError` when fiona cannot open the shapefile at all.

    Yields rows with the DBF value's own string, verbatim, never :class:`PlantRecord`, whose
    ``plot_number``, ``row_number`` and ``col_number`` narrow to ``Optional[float]``
    (:func:`_maybe_float`).
    """
    import fiona
    from pyproj import Transformer
    from shapely.geometry import shape

    shp_path = Path(path)
    fields = dict(_DEFAULT_SHAPEFILE_FIELD_MAP)
    if field_map:
        fields.update(field_map)

    geom_kinds: set[str] = set()
    rows: list[dict[str, object]] = []
    skipped_null_geometry = 0
    try:
        opened = fiona.open(str(shp_path))
    except Exception as exc:
        raise ShapefileUnreadableError(
            f"{shp_path}: fiona could not open this shapefile ({type(exc).__name__}: {exc}); "
            "check that the .shp, .shx and .dbf parts are all present and readable beside each "
            "other, then run tcip shp-to-plant-csv again"
        ) from exc
    with opened as layer:
        # fiona reports a missing or unparseable .prj as an empty CRS (probed directly, neither
        # raises), so one falsy check catches both without a second except clause.
        if not layer.crs:
            raise ShapefileCrsUnknownError(
                f"{shp_path}: no resolvable coordinate reference system (missing or unreadable "
                ".prj); refusing to guess a CRS, supply a .prj alongside the .shp"
            )
        # always_xy keeps both sides in (lon, lat) order; EPSG:4326's authority-declared order is
        # (lat, lon), and without this every coordinate lands in the wrong column.
        transform = Transformer.from_crs(layer.crs, "EPSG:4326", always_xy=True)

        available = set(layer.schema["properties"])
        resolved_fields: dict[str, Optional[str]] = {}
        for csv_col, shp_field in fields.items():
            if shp_field in available:
                resolved_fields[csv_col] = shp_field
            elif shp_field[:10] in available:
                # A source field name over 10 chars is truncated by the ESRI Shapefile DBF
                # driver; try that truncated form before giving up.
                resolved_fields[csv_col] = shp_field[:10]
            else:
                resolved_fields[csv_col] = None
        missing_fields = sorted(c for c, f in resolved_fields.items() if f is None)

        for index, feature in enumerate(layer):
            if feature.geometry is None:
                skipped_null_geometry += 1
                continue
            geom = shape(feature.geometry)
            if geom.geom_type == "Point":
                geom_kinds.add("point")
                x, y = geom.x, geom.y
            elif geom.geom_type in ("Polygon", "MultiPolygon"):
                geom_kinds.add("polygon" if geom.geom_type == "Polygon" else "multipolygon")
                centroid = geom.centroid
                x, y = centroid.x, centroid.y
            else:
                raise ValueError(
                    f"{shp_path}: feature {index} has geometry type {geom.geom_type!r}, which "
                    "this reader does not read a point or centroid from; only Point, Polygon "
                    "and MultiPolygon geometry is read"
                )
            lon, lat = transform.transform(x, y)
            properties = dict(feature.properties)
            column = PLANT_CSV_COLUMN_FOR_FIELD
            row: dict[str, object] = {
                column[field]: _shapefile_field_value(properties, resolved_fields[field])
                for field in ("plot_name", "accession_name", "plot_number", "row_number",
                              "col_number")
            }
            row[column["lon"]] = lon
            row[column["lat"]] = lat
            rows.append(row)

    return ShapefileRows(
        rows=rows, missing_fields=missing_fields, geometry_kinds=sorted(geom_kinds),
        skipped_null_geometry=skipped_null_geometry,
    )


def _maybe_float(x: Optional[str]) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def registry_content_digest(plants: list[PlantRecord]) -> str:
    """sha256 over ``plants``' own fields, order-independent, so two registrations of the same
    plants under two paths (or a different row order in one file) digest equal.
    """
    from dataclasses import asdict

    rows = sorted(
        (asdict(p) for p in plants), key=lambda r: json.dumps(r, sort_keys=True))
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()


class ShapefileCrsUnknownError(ValueError):
    """A shapefile's coordinate reference system cannot be resolved: fiona reports a missing
    ``.prj``, and a ``.prj`` whose WKT does not parse, as the same falsy ``layer.crs``. No CRS is
    ever guessed.
    """


class ShapefileUnreadableError(ValueError):
    """fiona could not open a shapefile at all: a missing or unreadable ``.shx``/``.dbf`` part, a
    file that is not a shapefile, a driver or IO failure. Raised in place of whatever fiona threw.
    A ``ValueError`` subclass.
    """


class NoGeoreferencedPlantsError(Exception):
    """A registration names a CSV that parsed no georeferenced, named plant. ``paths`` lists
    every file that failed the parse, so the caller's refusal names them all at once."""

    def __init__(self, message: str, *, paths: list[str]) -> None:
        super().__init__(message)
        self.paths = paths


class PlantRegistryNameConflictError(Exception):
    """A registration's name is already taken by a different set of plants."""


PLANT_REGISTRY_STORE = "plant_registries"

NAME_SEGMENT = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
"""A caller-chosen registry or mapping name's one legal shape: lowercase letters, digits, single
hyphens between groups."""

ARCHIVED_NAME_SEGMENT = re.compile(NAME_SEGMENT.pattern.removesuffix("$") + r"@[0-9a-f]{12}$")
"""An archived plant-mapping name's own shape: a legal :data:`NAME_SEGMENT` plus ``@`` and the
twelve-hex-digit prefix of the record digest it archived; it never matches plain
:data:`NAME_SEGMENT`."""


def _named(name: str, *patterns: re.Pattern[str]) -> tuple[str]:
    """``name`` as a key's one part; ``ValueError`` when it matches none of ``patterns``."""
    if not any(p.fullmatch(name) for p in patterns):
        raise ValueError(f"name {name!r} is not lowercase letters, digits and single hyphens "
                         f"({patterns[0].pattern})")
    return (name,)


def plant_registry_key(project: Path | str, name: str) -> Key:
    """One project's named plant registry: identity state for a plant-locations CSV set, under
    ``.tcip/state`` like :func:`plant_mapping_key`. A name outside ``NAME_SEGMENT`` raises
    ``ValueError``.
    """
    return Key(PLANT_REGISTRY_STORE, str(project_state_dir(project)), _named(name, NAME_SEGMENT))


class PlantRegistryNotFoundError(ValueError):
    """A plant registry name nothing is registered under in the project."""


def load_registry(project: Path | str, name: str) -> dict:
    """The named registry record. A name nothing is stored under (never registered, or deleted
    since a mapping recorded it) refuses (:class:`PlantRegistryNotFoundError`) naming it."""
    record = tcip_store.read(plant_registry_key(project, name), default=None)
    if record is None:
        raise PlantRegistryNotFoundError(
            f"plant registry not found: {name!r} under {project}; register it with "
            "register_plant_registry before naming it.")
    return record


def registry_csv_entries(record: dict, project: Path | str) -> list[dict]:
    """The ``{path, sha256, n_plants}`` entries a loaded registry record of ``project``
    carries, each ``path`` resolved through
    :func:`~tcip_mcp.registry_paths.resolved_registry_path`."""
    from tcip_mcp.registry_paths import resolved_registry_path

    return [{**e, "path": str(resolved_registry_path(project, e["path"]))}
            for e in record["csvs"]]


def registry_entries_or_refusal(
    build: "MappingBuild", project: Path | str,
) -> tuple[list[dict], Optional[str]]:
    """The CSV entries ``build.plant_registry`` names, or the refusal naming the registry, the
    mapping and ``project`` when they cannot be trusted: the named registry no longer loads,
    or its own ``digest`` no longer matches ``build.plant_registry["digest"]``.
    """
    registry_name = (build.plant_registry or {}).get("name")
    if not registry_name:
        return [], None
    try:
        record = load_registry(project, registry_name)
    except PlantRegistryNotFoundError as exc:
        return [], f"mapping {build.name!r} names it: {exc}"
    stored_digest = (build.plant_registry or {}).get("digest")
    if stored_digest is not None and record.get("digest") != stored_digest:
        return [], (
            f"plant registry {registry_name!r} under {project} named by mapping "
            f"{build.name!r} has moved (built against digest {stored_digest!r}, now "
            f"{record.get('digest')!r}); rebuild the mapping against the current registry")
    return registry_csv_entries(record, project), None


def verify_registry_csv_bytes(
    registry_entries: list[dict],
) -> tuple[list[str], Optional[str], dict[str, bytes]]:
    """Check each of ``registry_entries``'s (:func:`registry_csv_entries`'s own ``{path, sha256,
    n_plants}`` shape) recorded bytes against what is on disk now, reading each present file's
    bytes exactly once.

    Returns every missing path, the first rewritten file (or ``None`` when every present path's
    bytes still match what was registered), and the verified bytes this call read for every present
    path, keyed by the entry's own ``path``, for :func:`read_plant_csv_bytes` to parse. Reports the
    bytes facts only; the caller composes its own remedy.
    """
    missing: list[str] = []
    rewritten: Optional[str] = None
    bytes_by_path: dict[str, bytes] = {}
    for entry in registry_entries:
        p = Path(entry["path"])
        if not p.is_file():
            missing.append(entry["path"])
            continue
        data = p.read_bytes()
        bytes_by_path[entry["path"]] = data
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry["sha256"] and rewritten is None:
            rewritten = f"plant CSV {entry['path']} was rewritten since it was registered"
    return missing, rewritten, bytes_by_path


def parse_plant_registry_csvs(
    csv_paths: list[Path], project: Path | str,
) -> tuple[list[dict], str, int]:
    """Parse ``csv_paths`` into the registry's own ``{path, sha256, n_plants}`` entries, each
    ``path`` spelled against ``project`` by :func:`~tcip_mcp.registry_paths.stored_path`,
    the content digest over every parsed row and the total plant count, committing nothing.

    Raises :class:`NoGeoreferencedPlantsError`, naming every file that parsed no georeferenced,
    named plant or is not UTF-8 text (a binary file, a shapefile's own ``.shp``/``.shx``/``.dbf``
    included).
    """
    from tcip_mcp.registry_paths import stored_path

    failed: list[str] = []
    csvs_meta: list[dict] = []
    all_plants: list[PlantRecord] = []
    for p in csv_paths:
        data = Path(p).read_bytes() if Path(p).is_file() else b""
        try:
            records = read_plant_csv_bytes(data)
        except UnicodeDecodeError:
            records = []
        if not records:
            failed.append(str(p))
            continue
        all_plants.extend(records)
        csvs_meta.append({
            "path": stored_path(p, project),
            "sha256": hashlib.sha256(data).hexdigest(),
            "n_plants": len(records),
        })
    if failed:
        raise NoGeoreferencedPlantsError(
            f"{failed} parsed no georeferenced, named plant, or is not UTF-8 text (need columns "
            "plot_name, WGS84_centroid_x, WGS84_centroid_y with usable values); register only "
            "files that carry at least one",
            paths=failed,
        )
    return csvs_meta, registry_content_digest(all_plants), len(all_plants)


def register_plant_registry_record(
    project: Path | str,
    name: str,
    csv_paths: list[Path],
    *,
    crop: str,
    site: str,
) -> dict:
    """Register ``csv_paths`` under ``name`` in this project's plant registry, and return the
    stored record.

    Parses every path through :func:`parse_plant_registry_csvs`
    (:class:`NoGeoreferencedPlantsError` names the file that parsed no georeferenced, named
    plant); the record holds the ``{path, sha256, n_plants}`` entries plus ``crop``, ``site``,
    ``registered_at`` and the parsed content's digest. The read-then-write is
    one transaction (:func:`tcip_store.transaction`, this store's own ``concurrency="cas"``): a
    second registration under a taken name returns the existing record unchanged when the digest
    matches, and raises :class:`PlantRegistryNameConflictError` otherwise, naming the two digests.
    """
    csvs_meta, digest, n_plants = parse_plant_registry_csvs(csv_paths, project)
    key = plant_registry_key(project, name)
    with tcip_store.transaction(key) as txn:
        existing = txn.read(key, default=None)
        if existing is not None:
            if existing.get("digest") == digest:
                return existing
            raise PlantRegistryNameConflictError(
                f"plant registry {name!r} under {project} already names different plants "
                f"(digest {existing.get('digest')!r}, this registration would write "
                f"{digest!r}); register under a new name")
        record = {
            "name": name,
            "crop": crop,
            "site": site,
            "csvs": csvs_meta,
            "n_plants": n_plants,
            "digest": digest,
            "registered_at": now_iso(),
        }
        txn.write(key, record)
    return record


# ── Geometry helpers ───────────────────────────────────────────────────


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return EARTH_RADIUS_M * c


def attr(row: object, name: str) -> Any:
    """One field of an assignment row, an :class:`Assignment` or the plain dict row
    :meth:`MappingBuild.rows` produces; ``None`` when it carries none."""
    return row.get(name) if isinstance(row, dict) else getattr(row, name, None)


def stems_delivery_reads(rows: "Iterable[Assignment | dict]", bucket: "Bucket") -> set[str]:
    """Stems of ``rows`` (one date's assignment rows) :func:`assignment_is_attributed` calls
    attributed, whose stem also carries a prediction document in ``bucket``'s record: which
    prediction documents this delivery reads, never whether the underlying image still exists on
    disk.
    """
    pred_stems = set(bucket.documents)
    return {attr(row, "stem") for row in rows
            if assignment_is_attributed(row) and attr(row, "stem") in pred_stems}


def nearest_plant(
    position: tuple[float, float], plants: list[PlantRecord], *, within_m: float,
    skip: "set[int] | frozenset[int]" = frozenset(),
) -> tuple[Optional[int], Optional[float]]:
    """The index in ``plants`` (those in ``skip`` passed over) of the plant nearest the ``(lat,
    lon)`` ``position``, and its distance in meters; the index is ``None`` when that plant lies
    farther than ``within_m`` and both are ``None`` when no plant is left to measure."""
    distances = [(haversine_m(*position, p.lat, p.lon), i)
                 for i, p in enumerate(plants) if i not in skip]
    if not distances:
        return None, None
    distance, index = min(distances)
    return (index if distance <= within_m else None), distance


# ── Sequence anchoring ─────────────────────────────────────────────────


# Iglewicz & Hoaglin's modified z-score outlier test (1993, ASQC Quality Press), tested in
# log-space against the date's own consecutive-gap ratios; 3.5 is the convention's own cutoff.
_ROW_BREAK_MODIFIED_Z = 3.5


def _derive_row_break_threshold(gaps: list[float]) -> Optional[float]:
    """Find the natural break in a date's own consecutive-image gap distances, if one exists.

    A walker's path produces small, roughly-uniform steps within a row and one or more much
    larger jumps at a row transition, regardless of whether the row is straight or curved. This
    looks for that break as the largest relative (log-space) jump between consecutive values in
    the sorted gap sequence, then tests it against the rest of the sorted jumps via Iglewicz &
    Hoaglin's modified z-score (median/MAD-based, robust to the small, often near-degenerate
    samples a single date's image count produces). Returns ``None`` when there aren't enough gaps
    to judge, or the gaps are too uniform to support a split.
    """
    if len(gaps) < 2:
        return None
    sorted_gaps = sorted(gaps)
    # A near-zero gap carries no walking-pace information (its log/ratio would read as infinite);
    # excluded as a candidate split rather than treated as an automatic winner.
    candidates = [
        (i, math.log(hi / lo)) for i, (lo, hi) in enumerate(zip(sorted_gaps, sorted_gaps[1:]))
        if lo > 1e-9
    ]
    if not candidates:
        return None
    best_i, best_log_ratio = max(candidates, key=lambda pair: pair[1])

    other_log_ratios = [r for i, r in candidates if i != best_i]
    if not other_log_ratios:
        # Only one candidate split to compare against: no basis to call it a genuine transition.
        return None
    median_other = statistics.median(other_log_ratios)
    mad = statistics.median([abs(r - median_other) for r in other_log_ratios])
    if mad > 0:
        modified_z = 0.6745 * (best_log_ratio - median_other) / mad
        if modified_z < _ROW_BREAK_MODIFIED_Z:
            return None
    elif best_log_ratio <= median_other:
        # No measured spread in the rest of this date's steps: only a genuinely larger jump
        # counts, MAD's own limiting behavior as it -> 0, not an unconditional pass.
        return None

    return sorted_gaps[best_i] + (sorted_gaps[best_i + 1] - sorted_gaps[best_i]) / 2


def _segment_runs(stamps: list[ImageStamp]) -> list[list[ImageStamp]]:
    """Break an ordered list of stamps into row runs on GPS jumps that stand out from that
    date's own walking gaps (see ``_derive_row_break_threshold``)."""
    if not stamps:
        return []
    pairs: list[tuple[ImageStamp, Optional[float]]] = []
    for prev, cur in zip(stamps, stamps[1:]):
        d = None
        if prev.position is not None and cur.position is not None:
            d = haversine_m(*prev.position, *cur.position)
        pairs.append((cur, d))

    threshold = _derive_row_break_threshold([d for _, d in pairs if d is not None])

    runs: list[list[ImageStamp]] = [[stamps[0]]]
    for cur, d in pairs:
        if d is None or threshold is None or d <= threshold:
            runs[-1].append(cur)
        else:
            runs.append([cur])
    return runs


def _order_by_time(stamps: list[ImageStamp]) -> list[ImageStamp]:
    ordered = sorted(
        stamps, key=lambda s: (s.timestamp is None, s.timestamp, s.stem)
    )
    return ordered


def assign_plants(
    stamps: list[ImageStamp],
    plants: list[PlantRecord],
    *,
    nn_tolerance_m: float,
) -> list[Assignment]:
    """Assign each image to a plant by sequence-anchored NN matching: its nearest unclaimed plant
    within the sequence gate, else its nearest plant within the loosest gate
    (:func:`match_gates`), else none. A stamp with no position (a ``band_group`` or ``raster``, or
    an ``image`` with no usable EXIF position) is unmapped."""
    out: list[Assignment] = []
    gates = match_gates(nn_tolerance_m)
    claimed: set[int] = set()
    for run in _segment_runs(_order_by_time(stamps)):
        for s in run:
            if s.position is None or not plants:
                out.append(Assignment.of(s, None, "unmapped", None))
                continue
            index, distance = nearest_plant(
                s.position, plants, within_m=gates["sequence_match_distance_m"], skip=claimed)
            if index is not None:
                claimed.add(index)
                out.append(Assignment.of(s, plants[index], "sequence", distance))
                continue
            index, distance = nearest_plant(
                s.position, plants, within_m=gates["max_match_distance_m"])
            out.append(Assignment.of(s, None, "unmapped", distance) if index is None
                       else Assignment.of(s, plants[index], "nearest_neighbor", distance))

    by_image = {(a.date_folder, a.image): a for a in out}
    return [by_image[(s.date_folder, s.name)] for s in stamps]


# ── Whole-dataset driver ────────────────────────────────────────────────


def grid_pitch_m(plants: list[PlantRecord]) -> float:
    """Median nearest-neighbor spacing of the plant centroids, the planting grid pitch (m); 0.0
    for fewer than two plants."""
    pts = [(p.lat, p.lon) for p in plants]
    if len(pts) < 2:
        return 0.0
    nn = []
    for i, (la, lo) in enumerate(pts):
        nn.append(min(haversine_m(la, lo, la2, lo2) for j, (la2, lo2) in enumerate(pts) if j != i))
    nn.sort()
    return nn[len(nn) // 2]


class NoMatchToleranceError(ValueError):
    """No match tolerance is stated and the plant layout derives none."""


def resolve_nn_tolerance_m(plants: list[PlantRecord], stated: float | None = None) -> dict:
    """The tolerance (meters) a capture or detection is matched to a plant within, and where it
    came from: ``{"value": float, "source": str}``.

    Derived as the tolerance whose loosest gate (:func:`match_gates`) reaches half a grid cell
    (:func:`grid_pitch_m`), ``"grid_pitch"``. A stated value is honored (``"stated"``) up to that
    ceiling and capped at it (``"stated_capped"``). Raises :class:`NoMatchToleranceError` naming
    ``nn_tolerance_m`` when nothing is stated and the layout carries too few plants to derive a
    pitch from.
    """
    ceiling = grid_pitch_m(plants) / (2 * NEAREST_MATCH_FACTOR)
    if stated is None:
        if ceiling <= 0:
            raise NoMatchToleranceError(
                "the plant layout carries fewer than two georeferenced plants, so no grid pitch "
                "derives a match tolerance: state nn_tolerance_m (meters)")
        return {"value": ceiling, "source": "grid_pitch"}
    if 0 < ceiling < stated:
        return {"value": ceiling, "source": "stated_capped"}
    return {"value": stated, "source": "stated"}


def match_gates(nn_tolerance_m: float) -> dict:
    """The distances a match is accepted out to, given ``nn_tolerance_m``: the tolerance itself,
    a sequence-anchored match's ceiling, and ``max_match_distance_m``, the loosest any match is."""
    return {
        "nn_tolerance_m": nn_tolerance_m,
        "sequence_match_distance_m": nn_tolerance_m * SEQUENCE_MATCH_FACTOR,
        "max_match_distance_m": nn_tolerance_m * NEAREST_MATCH_FACTOR,
    }


def build_mapping(
    images_root: Path,
    plant_csv_paths: list[Path],
    *,
    name: str,
    dataset_root: Path | str,
    dataset_id: str,
    project: Path | str,
    plant_registry: dict,
    dates: Optional[list[str]] = None,
    nn_tolerance_m: Optional[float] = None,
) -> MappingBuild:
    """Build one project's named plant mapping: per-date assignments plus the provenance that binds
    the record to the inputs it was built from.

    ``name``/``dataset_root``/``dataset_id`` are the caller's own resolved facts, and
    ``project`` the project the mapping belongs to. ``plant_csv_paths`` are the files this build
    reads the plants from (resolved by
    the caller from ``plant_registry``'s own ``name``); ``plant_registry`` is the ``{"name": ...,
    "digest": ...}`` reference stored on the record in their place.

    ``nn_tolerance_m`` resolves through :func:`resolve_nn_tolerance_m`, whose value and source
    the record carries.

    A date's captures are enumerated through ``image_utils.list_logical_images``, so a band raster
    or a band group ingested under a mapped date is a capture the identity sees; its
    :class:`~tcip_mcp.pipelines.image_utils.AmbiguousImageStemError` propagates.

    Raises :class:`UngeoreferencedCaptureError`: naming ``images_root`` when the requested dates
    carry no capture at all, and with :func:`ungeoreferenced_capture_message` (naming any capture
    PIL could not open before the position clause) when every capture that was read carries no
    position this door reads.
    """
    from tcip_mcp.pipelines.image_utils import list_logical_images

    plant_csv_paths = [Path(p) for p in plant_csv_paths]
    plants = read_plant_csvs(plant_csv_paths)

    tolerance = resolve_nn_tolerance_m(plants, nn_tolerance_m)
    logger.info("nn_tolerance_m %.2f (%s)", tolerance["value"], tolerance["source"])

    images_root = Path(images_root)
    dates_walked: list[str] = []
    assignments: dict[str, list[Assignment]] = {}
    capture_ids: dict[str, str] = {}
    capture_digests_by_date: dict[str, dict[str, str]] = {}
    unreadable: dict[str, list[str]] = {}
    n_stamps = 0
    n_positioned = 0
    if images_root.is_dir():
        for date_dir in sorted(images_root.iterdir()):
            if not date_dir.is_dir():
                continue
            if dates is not None and date_dir.name not in dates:
                continue
            date = date_dir.name
            dates_walked.append(date)
            logical = list_logical_images(date_dir)
            stamps = _read_date_stamps(logical, date)
            n_stamps += len(stamps)
            n_positioned += sum(1 for s in stamps if s.position is not None)
            capture_ids[date] = capture_identity(stamps)
            capture_digests_by_date[date] = capture_digests(stamps)
            unreadable[date] = sorted(
                s.name for s in stamps if s.kind == "image" and s.readable is False)
            assignments[date] = assign_plants(stamps, plants, nn_tolerance_m=tolerance["value"])

    if n_stamps == 0:
        raise UngeoreferencedCaptureError(
            f"no capture under {images_root} on the requested dates")
    if n_positioned == 0:
        all_unreadable = sorted(
            {name for date in dates_walked for name in unreadable.get(date, [])})
        raise UngeoreferencedCaptureError(
            ungeoreferenced_capture_message(str(images_root), all_unreadable))

    from tcip_mcp.registry_paths import stored_path

    return MappingBuild(
        name=name,
        dataset_root=stored_path(dataset_root, project),
        dataset_id=dataset_id,
        built_at=now_iso(),
        dates_requested=list(dates) if dates is not None else None,
        dates=sorted(dates_walked),
        nn_tolerance_m=tolerance,
        plant_registry=plant_registry,
        capture_identity=capture_ids,
        capture_digests=capture_digests_by_date,
        unreadable=unreadable,
        assignments=assignments,
    )


PLANT_MAPPING_STORE = "plant_mapping"


def plant_mapping_key(project: Path | str, name: str) -> Key:
    """One project's named plant-mapping build, addressed by the project that owns it, under its
    ``.tcip/state``: a dataset can be read by more than one project, and each project's mapping is
    its own. A mapping is written whole in one call, and a later build under the same name
    replaces it. A name outside ``NAME_SEGMENT`` and the archived ``ARCHIVED_NAME_SEGMENT`` raises
    ``ValueError``.
    """
    return Key(PLANT_MAPPING_STORE, str(project_state_dir(project)),
               _named(name, NAME_SEGMENT, ARCHIVED_NAME_SEGMENT))


def plant_mapping_names(project: Path | str) -> list[str]:
    """Every mapping name persisted under this project that ``NAME_SEGMENT`` admits, sorted, from
    the store's own key listing."""
    root = str(project_state_dir(project))
    names = (key.parts[-1] for key in tcip_store.keys(PLANT_MAPPING_STORE, root))
    return sorted(name for name in names if NAME_SEGMENT.fullmatch(name))


def archived_mapping_name(name: str, record_sha256: str) -> str:
    """The key name a superseding rebuild of ``name`` moves its cited record ``record_sha256``
    to; :func:`plant_mapping_names` never lists it, since ``@`` fails ``NAME_SEGMENT``."""
    return f"{name}@{record_sha256[:12]}"


def record_digest(record: object) -> str:
    """The digest a stored mapping record earns, whatever its shape: sha256 over
    ``encode_record(record)``, what its receipt names."""
    return hashlib.sha256(encode_record(record)).hexdigest()


class MappingRebuildError(Exception):
    """A same-name rebuild would replace a mapping record a delivery event under this project
    still cites. ``event_ids`` names every citing event; ``status`` is the web door's HTTP status
    for it."""

    def __init__(self, message: str, *, event_ids: list[str], status: int = 409) -> None:
        super().__init__(message)
        self.event_ids = event_ids
        self.status = status


def _citing_delivery_event_ids(project: Path | str, name: str, digest: str) -> list[str]:
    """Every delivery event under this project whose own ``plant_mapping`` disclosure names
    (``name``, ``digest``): the events a same-name rebuild would strand by silently replacing the
    record they cite, sorted for a stable refusal message.
    """
    from tcip_mcp.delivery import read_delivery_events
    from tcip_mcp.pipelines.delivery_events_schema import PlantMappingDisclosure

    return sorted(
        event.event_id for event in read_delivery_events(project)
        if isinstance(event.plant_mapping, PlantMappingDisclosure)
        and (event.plant_mapping.name, event.plant_mapping.record_sha256) == (name, digest))


def persist_mapping(build: MappingBuild, project: Path | str, *, supersede: bool = False,
                    actor: str | None) -> None:
    """Write the mapping record under ``build.name``, then the receipt by ``actor`` that binds it
    to this build.

    The record is committed before the receipt (a log append cannot join a record transaction): a
    receipt that cannot be written fails loudly (``AuditEntryNotWrittenError`` propagates) and
    leaves a record no receipt names, which :func:`load_mapping` refuses to read until a rebuild
    replaces it.

    ``project`` names which log the receipt lands in.

    A rebuild under ``name`` whose current record is still cited by a delivery event under this
    project raises :class:`MappingRebuildError`, naming the citing events, unless
    ``supersede=True``. In that case the current record moves unchanged to
    :func:`archived_mapping_name`, where its own receipt still answers for it, and
    ``build.supersedes`` is set to its digest before this writes the new record, whose receipt
    names both digests. An uncited rebuild replaces the record. The current record is identified
    by its raw :func:`record_digest`, whatever its shape.
    """
    from tcip_mcp.audit import record_event_or_raise

    name = build.name
    existing_raw = tcip_store.read(plant_mapping_key(project, name), default=None)
    archived_digest: Optional[str] = None
    if existing_raw is not None:
        existing_digest = record_digest(existing_raw)
        citing = _citing_delivery_event_ids(project, name, existing_digest)
        if citing and not supersede:
            raise MappingRebuildError(
                f"plant mapping {name!r} under {project} is cited by delivery event(s) "
                f"{citing}: rebuilding under this name would strand them. Pass supersede=True to "
                "archive the current record and rebuild.",
                event_ids=citing,
            )
        if citing:
            tcip_store.replace(
                plant_mapping_key(project, archived_mapping_name(name, existing_digest)),
                existing_raw)
            archived_digest = existing_digest
            build.supersedes = archived_digest

    record = build.to_record()
    tcip_store.replace(plant_mapping_key(project, name), record)
    record_sha256 = record_digest(record)
    record_event_or_raise(
        "plant_mapping_built",
        {
            "name": name,
            "dataset_root": build.dataset_root,
            "built_at": build.built_at,
            "record_sha256": record_sha256,
            "supersedes": archived_digest,
        },
        actor=actor, scope=project,
    )


def build_plant_mapping(
    project: Path | str, name: str, images_root: Path | str, plant_registry: str, *,
    dates: Optional[list[str]] = None, nn_tolerance_m: Optional[float] = None,
    supersede: bool = False, actor: str | None,
) -> MappingBuild:
    """Map the captures under ``images_root`` to the plants of ``project``'s registry
    ``plant_registry`` (:func:`build_mapping`) and persist the mapping under ``name`` by ``actor``
    (:func:`persist_mapping`, whose receipt is the act's one audit line); return the build.

    Raises ``ValueError`` for a name the key refuses, for an ``images_root`` that is not a
    dataset's own ``images/`` directory (naming ``register_dataset``) and for a dataset with no
    identity record; :class:`PlantRegistryNotFoundError`; and every refusal of the two steps.
    """
    from tcip_mcp.dataset_layout import dataset_root_of, image_root, require_dataset_identity
    from tcip_mcp.pipelines.data.splits import same_directory

    plant_mapping_key(project, name)
    images = Path(images_root).resolve()
    dataset = dataset_root_of(images)
    if dataset is None or not same_directory(image_root(dataset), images):
        raise ValueError(f"{images_root} is not a dataset's own images/ root; a mapping maps the "
                         "image tree of a dataset registered with register_dataset")
    identity = require_dataset_identity(dataset)
    registry = load_registry(project, plant_registry)
    build = build_mapping(
        images, [Path(e["path"]) for e in registry_csv_entries(registry, project)],
        name=name, dataset_root=dataset, dataset_id=identity["id"], project=project,
        plant_registry={"name": plant_registry, "digest": registry["digest"]},
        dates=dates, nn_tolerance_m=nn_tolerance_m)
    persist_mapping(build, project, supersede=supersede, actor=actor)
    return build


def load_mapping_rows(project: Path | str, name: str) -> dict[str, list[dict]]:
    """The persisted mapping as plain per-date rows, read through :func:`load_mapping`. ``{}`` when
    no mapping is stored under ``name``.
    """
    build = load_mapping(project, name)
    return build.rows() if build is not None else {}


# Process-local memo for load_mapping's receipt check: a project's log cursor plus every
# plant_mapping_built (name, record_sha256) seen; a miss re-reads once before refusing.
_receipt_cursor: dict[str, str] = {}
_receipt_seen: dict[str, dict[str, set[str]]] = {}


def _scan_receipts(project: Path | str, root_key: str, *, after: Optional[str]) -> None:
    from tcip_mcp.audit import acts_of

    entries, cursor = acts_of(project, ("plant_mapping_built",), after=after)
    seen = _receipt_seen.setdefault(root_key, {})
    for entry in entries:
        seen.setdefault(entry["arguments"]["name"], set()).add(entry["arguments"]["record_sha256"])
    _receipt_cursor[root_key] = cursor


def _require_receipt(project: Path | str, name: str, record_sha256: str) -> None:
    """Refuse unless a ``plant_mapping_built`` event under ``name`` names ``record_sha256`` in
    the project's own audit log; any matching receipt admits, the latest or not."""
    root_key = str(Path(project).resolve())
    if root_key not in _receipt_cursor:
        _scan_receipts(project, root_key, after=None)
    if record_sha256 not in _receipt_seen.get(root_key, {}).get(name, set()):
        _scan_receipts(project, root_key, after=_receipt_cursor.get(root_key))
    if record_sha256 not in _receipt_seen.get(root_key, {}).get(name, set()):
        raise ValueError(
            f"plant mapping {name!r} under {project} carries no plant_mapping_built "
            f"receipt naming record {record_sha256}: this record was not written by "
            "build_plant_mapping (a forged or hand-restored record, or one whose receipt could "
            "not be written); rebuild with build_plant_mapping")


def load_mapping(project: Path | str, name: str) -> Optional[MappingBuild]:
    """One project's named, persisted plant-mapping build, or ``None`` when nothing is stored
    under that name.

    Decodes the record (:meth:`MappingBuild.from_record`), then requires the receipt its build
    wrote under the record's own name (see :func:`_require_receipt`) before trusting it: a record
    never built through the platform's own writers is refused, a forgery naming the real inputs'
    identities included, and so is a record stored under a name other than its own or the
    archive name (:func:`archived_mapping_name`) a superseding rebuild moved it to.
    """
    raw = tcip_store.read(plant_mapping_key(project, name), default=None)
    if raw is None:
        return None
    build = MappingBuild.from_record(raw, project, name)
    if name not in (build.name, archived_mapping_name(build.name, build.record_sha256)):
        raise ValueError(f"the plant mapping stored under {name!r} in {project} is the record "
                         f"of {build.name!r}; rebuild with build_plant_mapping")
    _require_receipt(project, build.name, build.record_sha256)
    return build


def resolved_mapping_key_for_citation(
    project: Path | str, name: str, record_sha256: str,
) -> Optional[str]:
    """The name a reader loads to see exactly the record a delivery event's own
    ``plant_mapping.record_sha256`` cites: ``name`` itself when the record currently stored under
    it still hashes to ``record_sha256``, the archive name a superseding rebuild moved it to
    (:func:`archived_mapping_name`) when that key holds a stored record, or ``None`` when
    neither does.
    """
    current = tcip_store.read(plant_mapping_key(project, name), default=None)
    if current is not None and record_digest(current) == record_sha256:
        return name
    archived_name = archived_mapping_name(name, record_sha256)
    if tcip_store.exists(plant_mapping_key(project, archived_name)):
        return archived_name
    return None


def _describe_capture(stamps_by_stem: dict[str, ImageStamp], stem: str) -> str:
    """Name one capture for a refusal message: its kind and file name when a fresh stamp for
    ``stem`` was read this call, else the bare stem.
    """
    s = stamps_by_stem.get(stem)
    if s is None:
        return stem
    if s.kind == "band_group":
        return f"band group manifest {s.name!r}"
    if s.kind == "raster":
        return f"raster {s.name!r}"
    return f"image {s.name!r}"


def verify_mapping_inputs(
    build: MappingBuild, dataset_root: Path | str, buckets: "Mapping[str, Bucket]", *,
    project: Path | str,
) -> dict:
    """Check what this delivery can verify about a mapping of ``project``'s recorded inputs
    against what is on disk now, for the captures it actually reads, at delivery time.
    ``buckets`` is the delivered buckets by recorded date.

    A mapped date no delivered bucket records is never walked (no enumeration, no EXIF):
    disclosed in ``captures_unverified`` as the bare date string, the same as a named date whose
    image folder is absent. A named date's folder is enumerated once
    (``image_utils.list_logical_images``, refusing by name on
    :class:`~tcip_mcp.pipelines.image_utils.AmbiguousImageStemError`);
    an enumerated stem the mapping's own assignment rows for that date do not name refuses,
    naming the date, the file(s) and the
    rebuild remedy. A recorded stem no longer enumerated is disclosed as ``"<date>/<name>"``, using
    the recorded row's own file name; so is a recorded, still-enumerated stem this delivery's own
    prediction bucket carries no document for (:func:`stems_delivery_reads`).

    Only a capture this delivery reads gets its fresh stamp read. For each: a capture readable when
    this mapping was built and unreadable now refuses by name. A row that recorded a plant position
    (:func:`assignment_is_attributed` true, with a ``distance_m``) refuses by name when the
    capture's fresh GPS position no longer sits ``distance_m`` from any plant of that name in the
    plant CSVs whose bytes this same call just verified, or when the fresh capture carries no GPS
    position at all. When that plant's own CSV is itself among ``plant_csvs_unverified``, the
    position is disclosed rather than compared.

    When every mapped capture of a date was read (``missing_stems`` empty and ``read_set`` equal to
    the recorded stems :func:`assignment_is_attributed` calls attributed), the whole date's
    identity (:func:`capture_identity`) is recomputed over every capture the date enumerates, the
    unmapped ones (a raster, a band group) included, and compared against the record, refusing on a
    mismatch and naming the capture(s) whose own :func:`capture_digests` entry moved; a capture
    verified this way is not also listed in ``captures_unverified``. Under a partial read that
    digest does not run, so an in-place EXIF timestamp or band-group manifest change on a read
    capture goes undetected when some other mapped capture of the same date was not read.

    Never raises: returns ``{"refusal": str}`` for any of the above; otherwise
    ``{"captures_unverified": [...], "plant_csvs_unverified": [...]}``, entries in ``build.dates``
    order and, within a date, sorted by name. A missing plant CSV is disclosed in
    ``plant_csvs_unverified``; a rewritten one refuses (restore the file's registered bytes, or
    register the current file under a new registry name, rebuild the mapping against it, and
    deliver under that name).
    """
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.image_utils import (
        AmbiguousImageStemError,
        list_logical_images,
        logical_image_name,
    )

    remedy = "rebuild with build_plant_mapping"

    # Plant CSVs first: the per-capture moved-position check below trusts only verified bytes.
    # A registry that no longer loads or whose digest moved refuses rather than verifying nothing.
    registry_entries, registry_refusal = registry_entries_or_refusal(build, project)
    if registry_refusal:
        return {"refusal": registry_refusal}
    plant_csvs_unverified, rewritten_fact, verified_csv_bytes = verify_registry_csv_bytes(
        registry_entries)
    if rewritten_fact:
        return {"refusal": (
            f"{rewritten_fact}: restore the file's registered bytes, or register the current "
            "file under a new registry name, rebuild the mapping against it, and deliver under "
            "that name")}
    verified_plants = [
        plant
        for entry in registry_entries if entry["path"] not in plant_csvs_unverified
        for plant in read_plant_csv_bytes(verified_csv_bytes[entry["path"]])
    ]

    captures_unverified: list[str] = []
    for date in build.dates:
        rows = build.assignments.get(date, [])
        recorded_by_stem = {a.stem: a for a in rows}
        recorded_stems = set(recorded_by_stem)

        bucket = buckets.get(date)
        if bucket is None:
            captures_unverified.append(date)
            continue

        date_dir = image_dir(dataset_root, date)
        if not date_dir.is_dir():
            captures_unverified.append(date)
            continue

        try:
            logical = list_logical_images(date_dir)
        except AmbiguousImageStemError as exc:
            return {"refusal": str(exc)}

        enumerated_stems = set(logical)
        added = enumerated_stems - recorded_stems
        if added:
            names = sorted(logical_image_name(logical[s]) for s in added)
            return {"refusal": (
                f"date {date} carries capture(s) {names} this mapping's assignments do not "
                f"name: the mapping does not cover what is on disk now; {remedy}")}

        missing_stems = recorded_stems - enumerated_stems
        present_stems = recorded_stems & enumerated_stems
        read_set = stems_delivery_reads(rows, bucket) & present_stems
        unread_stems = present_stems - read_set
        mapped_stems = {s for s in recorded_stems if assignment_is_attributed(recorded_by_stem[s])}
        full_mapped_coverage = not missing_stems and read_set == mapped_stems

        unverified_names: set[str] = set()
        if not full_mapped_coverage:
            # An unmapped capture (a raster, a band group) is never read through predictions, so
            # it always lands here unless the whole-date digest below re-checks it instead.
            unverified_names.update(
                recorded_by_stem[s].image for s in (missing_stems | unread_stems))

        # One observation: the whole date when its digest is re-checked below, else the read set.
        observed = _read_date_stamps(
            logical if full_mapped_coverage else {s: logical[s] for s in read_set}, date)
        stamps = [s for s in observed if s.stem in read_set]
        was_unreadable = set(build.unreadable.get(date, []))
        for s in stamps:
            row = recorded_by_stem[s.stem]
            if s.kind == "image":
                if s.name not in was_unreadable and s.readable is False:
                    return {"refusal": (
                        f"{s.name} (date {date}) was readable when this mapping was built and "
                        "could not be read now: retry, or rebuild if it is gone")}
            if assignment_is_attributed(row) and row.distance_m is not None:
                if s.position is None:
                    return {"refusal": (
                        f"{s.name} (date {date}) recorded a plant position when this mapping was "
                        "built and now carries no GPS position: rebuild, since its assignment "
                        "would differ")}
                distances = [
                    haversine_m(*s.position, p.lat, p.lon)
                    for p in verified_plants if p.plot_name == row.plot_name
                ]
                if not distances:
                    # The plant's own CSV is among plant_csvs_unverified: disclose this capture's
                    # recorded position as unrechecked rather than comparing it against nothing.
                    unverified_names.add(s.name)
                    continue
                if not any(abs(d - row.distance_m) <= 1e-6 for d in distances):
                    return {"refusal": (
                        f"{s.name} (date {date}) has moved since this mapping was built: no "
                        f"plant named {row.plot_name!r} is {row.distance_m} m away now; rebuild, "
                        "since its assignment would differ")}

        for name in sorted(unverified_names):
            captures_unverified.append(f"{date}/{name}")

        if full_mapped_coverage:
            new_identity = capture_identity(observed)
            if new_identity != build.capture_identity.get(date):
                new_digests = capture_digests(observed)
                recorded_digests = build.capture_digests.get(date, {})
                stamps_by_stem = {s.stem: s for s in observed}
                moved = sorted(
                    _describe_capture(stamps_by_stem, stem)
                    for stem in set(new_digests) | set(recorded_digests)
                    if new_digests.get(stem) != recorded_digests.get(stem)
                )
                detail = (
                    ", ".join(moved) if moved else "a capture whose identity does not decompose"
                )
                return {"refusal": (
                    f"date {date}: {detail} changed since this mapping was built; "
                    "rebuild to cover the images actually on disk")}

    return {
        "captures_unverified": captures_unverified,
        "plant_csvs_unverified": plant_csvs_unverified,
    }


def ungeoreferenced_capture_message(walked: str, unreadable: Sequence[str] = ()) -> str:
    """The refusal sentence for a walk whose captures carry no position this door reads.

    ``walked`` names what was walked. ``unreadable``, when given, opens the sentence by naming the
    captures PIL could not open at all, before the position clause.
    """
    prefix = ""
    if unreadable:
        prefix = (
            f"{', '.join(unreadable)} could not be opened, so their position could not be read; "
        )
    return (
        f"{prefix}no capture under {walked} on the requested dates carries a position this door "
        "reads (a photograph with no GPS position, or a raster or band-group capture, which never "
        "carries one here), so no capture can be assigned to a plant; per-plant identity for "
        "ungeoreferenced capture needs a plant-tag mechanism the platform does not have (README's "
        "roadmap), and a georeferenced orthomosaic delivers per plant through run_inference's "
        "raster regime and deliver_orthomosaic_plant_counts instead"
    )


class _StatusError(Exception):
    """A refusal whose ``str(exc)`` is the caller-facing message and ``status`` the web door's HTTP
    status for it."""

    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class UngeoreferencedCaptureError(_StatusError):
    """A plant mapping cannot be built or delivered from the captures at hand: none carries a
    position this door reads, or none could be read."""


class MappingDeliveryError(_StatusError):
    """A phenology delivery cannot proceed from a named mapping; ``str(exc)`` is the caller-facing
    message. ``status`` is the HTTP status for it (400 by default, 404 for a mapping that is not
    stored, 409 for a store-level problem reading it).
    """


def resolve_delivery_mapping(
    project: Path | str, name: str, buckets: "Mapping[str, Bucket]",
) -> tuple[MappingBuild, dict]:
    """Load the named mapping, refuse a delivered bucket's recorded date it does not cover
    (``buckets`` keyed by those dates), resolve the delivered buckets' one dataset root and require
    it to carry the mapping's own minted dataset id, then verify the mapping's recorded inputs
    against that resolved root for exactly the captures the buckets read.

    A delivery whose delivered dates attribute nothing (:func:`assignment_is_attributed` false for
    every one of them, a date recorded with no capture at all included) refuses on the record's own
    evidence, before ``verify_mapping_inputs`` re-reads anything. An empty date beside a fully
    attributed one ships, absent from every plant's series.

    Returns the loaded build and :func:`verify_mapping_inputs`'s disclosure; raises
    :class:`MappingDeliveryError`, naming the remedy, for every case a delivery must not proceed
    from, a ``StoreError`` reading the mapping store included.
    """
    from tcip_store import StoreError

    from tcip_mcp.buckets import shared_root
    from tcip_mcp.dataset_layout import require_dataset_identity

    try:
        mapping_build = load_mapping(project, name)
    except (StoreError, ValueError) as exc:
        raise MappingDeliveryError(
            f"could not read mapping {name!r}: {exc}", status=409) from exc
    if mapping_build is None:
        raise MappingDeliveryError(
            f"mapping not found: {name!r}; build one with build_plant_mapping before "
            "computing phenology", status=404)

    missing_dates = [d for d in buckets if d not in mapping_build.dates]
    if missing_dates:
        raise MappingDeliveryError(
            f"the delivered buckets record date(s) {missing_dates} the mapping {name!r} does not "
            "cover; rebuild the mapping to cover them, or drop those buckets")

    try:
        delivered_root = shared_root(buckets.values())
        delivered_identity = require_dataset_identity(delivered_root)
    except ValueError as exc:
        raise MappingDeliveryError(str(exc)) from exc
    if delivered_identity.get("id") != mapping_build.dataset_id:
        raise MappingDeliveryError(
            f"the predictions under {delivered_root} belong to a different dataset than the "
            f"mapping {name!r} was built over (mapping dataset_root "
            f"{mapping_build.dataset_root!r}, delivered dataset root {str(delivered_root)!r})")

    delivered_assignments = [
        a for date in buckets for a in mapping_build.assignments.get(date, [])
    ]
    if not delivered_assignments or not any(
        assignment_is_attributed(a) for a in delivered_assignments
    ):
        # A no-capture-at-all date is named below only inside this nothing-attributed scope.
        empty_dates = sorted(d for d in buckets if not mapping_build.assignments.get(d))
        if empty_dates:
            raise MappingDeliveryError(
                f"the mapping {name!r} recorded no capture at all for date(s) {empty_dates}: a "
                "date with no capture cannot be delivered; rebuild the mapping, or drop the "
                "date(s)")
        # Re-reading captures for a delivery that can attribute nothing would disclose about
        # nothing; refuse here, before verify_mapping_inputs, on the record's own evidence.
        registry_entries, registry_refusal = registry_entries_or_refusal(mapping_build, project)
        if registry_refusal:
            raise MappingDeliveryError(registry_refusal, status=409)
        paths = [entry["path"] for entry in registry_entries]
        n_plants = sum(entry["n_plants"] for entry in registry_entries)
        if n_plants == 0:
            raise MappingDeliveryError(
                f"the plant CSVs this mapping was built from ({paths}) parsed no plant with "
                "usable coordinates and a name, so no capture could be assigned; check the "
                "CSV's column headers against read_plant_csvs's and rebuild the mapping")
        if all(a.distance_m is None for a in delivered_assignments):
            raise MappingDeliveryError(
                ungeoreferenced_capture_message(
                    f"mapping {name!r} (dataset {mapping_build.dataset_root!r})"))
        gates = match_gates(mapping_build.nn_tolerance_m["value"])
        n_unpositioned = sum(1 for a in delivered_assignments if a.distance_m is None)
        unpositioned_note = (
            f", and {n_unpositioned} captures carry no position" if n_unpositioned else "")
        raise MappingDeliveryError(
            "every positioned capture on the delivered dates lies beyond the accepted match "
            f"distance ({gates['max_match_distance_m']} m from tolerance "
            f"{gates['nn_tolerance_m']} m, {mapping_build.nn_tolerance_m['source']}) of every "
            f"plant in {paths}{unpositioned_note}; check the plant CSV names this block, or "
            "rebuild the mapping with a stated tolerance")

    verified = verify_mapping_inputs(mapping_build, delivered_root, buckets, project=project)
    if "refusal" in verified:
        raise MappingDeliveryError(verified["refusal"])
    return mapping_build, verified


def plant_mapping_disclosure(project: Path | str, name: str, buckets: "Mapping[str, Bucket]",
                             plants: list[str]) -> dict:
    """The ``plant_mapping`` disclosure a per-plant delivery of the population ``plants`` carries
    when its plant ids came from mapping ``name``: the mapping resolved and verified against the
    delivered ``buckets`` by recorded date (:func:`resolve_delivery_mapping`, refusing as it does),
    and a population plant the mapping assigns on no delivered date refused
    (:class:`MappingDeliveryError`) naming each."""
    build, verified = resolve_delivery_mapping(project, name, buckets)
    assigned = {a.plot_name for d in buckets for a in build.assignments.get(d, [])
                if assignment_is_attributed(a)}
    unmapped = sorted(set(plants) - assigned)
    if unmapped:
        raise MappingDeliveryError(f"plant(s) {unmapped} are assigned to no plot by mapping "
                                     f"{name!r} on the delivered dates.")
    return build.delivery_disclosure(verified, list(buckets))
