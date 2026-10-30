"""Plant-ID mapping across capture dates.

Each image is assigned a plant by its capture sequence along a row and its GPS, or by
nearest-neighbor GPS where no sequence anchors it, every assignment recording its match ``source``
and its GPS ``distance_m``. A mapping is named project state, bound to the dataset it was built
over and to the build receipt that names it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields, replace
from datetime import datetime
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    NamedTuple,
    SupportsFloat,
    TypedDict,
    cast,
    get_args,
)

from PIL import ExifTags, Image
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    TypeAdapter,
    computed_field,
    with_config,
)
from pyproj import Geod

import tcip_store
from tcip_mcp.audit import now
from tcip_mcp.project_paths import project_state_dir
from tcip_store import Key, encode_record

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import JsonValue

    from tcip_mcp.buckets import Bucket
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import (
        DetectionAssignment,
    )

logger = logging.getLogger(__name__)

WGS84 = Geod(ellps="WGS84")
"""The datum every plant and capture position is stated in; ``WGS84.inv`` answers the geodesic
distance in meters between two of them."""


@dataclass
class ImageStamp:
    """Per-capture metadata: a photograph's EXIF, or a band group's or array raster's facts.

    ``kind`` is the capture's
    :attr:`~tcip_mcp.pipelines.raster_source.SourceHeader.kind` (``"photo"``, ``"group"``, or an
    array container's ``"tif"``/``"npy"``/``"npz"``), ``name`` its file name, ``readable`` whether
    a photograph's EXIF read (``None`` for the other kinds), and ``manifest_sha256``/``members`` a
    band group's own.
    """

    date_folder: str
    kind: str
    name: str
    timestamp: datetime | None
    lat: float | None
    lon: float | None
    h_pos_err: float | None
    readable: bool | None
    manifest_sha256: str | None = None
    members: tuple[str, ...] = ()

    @property
    def stem(self) -> str:
        """The capture's stem, its file name's."""
        return Path(self.name).stem

    @property
    def position(self) -> tuple[float, float] | None:
        """``(lat, lon)``, or ``None`` when the stamp lacks either."""
        return (
            None
            if self.lat is None or self.lon is None
            else (self.lat, self.lon)
        )


@dataclass
class PlantRecord:
    """One plant from a plant_locations CSV."""

    plant_id: str
    accession_name: str
    plot_number: float | None
    row_number: float | None
    col_number: float | None
    lat: float
    lon: float


_DECODED = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

GpsSource = Literal["sequence", "nearest_neighbor", "unmapped"]
"""Where a GPS assignment's plant came from: the capture sequence, the plain nearest plant, or
none within the date's match radius."""

UnattributedSegmentSource = Literal[
    "outside_segments",
    "overlapping_segments",
    "segment_without_plant",
    "segment_plant_within_position_error",
]
"""Why a canopy-segment assignment names no plant: no segment holds the detection, several do, the
one that does contains no plant, or it contains one whose position lies within the position error
of its boundary."""

SegmentSource = Literal["segment_containment", UnattributedSegmentSource]
"""Where a canopy-segment assignment's plant came from: one tied segment containing the detection,
or why none did."""

UNATTRIBUTED_SEGMENT_SOURCES: tuple[UnattributedSegmentSource, ...] = get_args(
    UnattributedSegmentSource
)

ToleranceSource = Literal["grid_pitch", "stated", "stated_capped"]
"""Where a match tolerance came from (:func:`resolve_nn_tolerance_m`): the plant grid's pitch,
the stated value, or the stated value capped to half the grid pitch."""

RadiusSource = Literal["h_pos_err", "stated"]
"""Where a date's match radius came from (:func:`date_match_radius`): the positioning error its
photographs report, or the caller's statement."""


class StatedRadius(BaseModel):
    """A match radius (meters) a mapping build states for its captures, and how it is known."""

    model_config = ConfigDict(extra="forbid")

    value: PositiveFloat
    rationale: str = Field(min_length=1)


class MatchRadius(BaseModel):
    """How far (meters) from a plant one date's captures were matched to it, and its source.

    ``rationale`` is the stated radius's own, ``None`` for a derived one.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    value: float
    source: RadiusSource
    rationale: str | None


class DateCounts(BaseModel):
    """One mapped date's counts: its images, those naming no plant, and the mean match distance.

    ``avg_distance_m`` is the mean GPS match distance, ``None`` for a date with no recorded
    distance.
    """

    n_images: int
    n_unattributed: int
    avg_distance_m: float | None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_mapped(self) -> int:
        """The date's images naming a plant."""
        return self.n_images - self.n_unattributed


class ServedMapping(BaseModel):
    """A persisted plant mapping as its build and load doors answer it.

    It holds its ``name``, the counts of every date it walked, the images naming no plant over
    all of them (:meth:`MappingBuild.unattributed`), the captures PIL could not open per date and
    each date's recorded match radius, absent for a date no capture of which carries a position.
    """

    name: str
    per_date: dict[str, DateCounts]
    images_unattributed: int
    unreadable: dict[str, list[str]]
    match_radius: dict[str, MatchRadius]


class MappingVerification(NamedTuple):
    """What :func:`verify_mapping_inputs` could not re-check: captures and plant CSVs, by name."""

    captures_unverified: list[str]
    plant_csvs_unverified: list[str]


class PlantMappingDisclosure(BaseModel):
    """The ``plant_mapping`` a phenology delivery attributed detections through.

    It holds the mapping's own identity, ``verify_mapping_inputs``'s :class:`MappingVerification`,
    and this delivery's own count of captures naming no plant over its ``dates_delivered``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    dataset_id: str
    dataset_root: str
    built_at: AwareDatetime
    record_sha256: str
    match_radius: dict[str, MatchRadius]
    capture_digests: dict[str, dict[str, str]]
    captures_unverified: list[str]
    plant_csvs_unverified: list[str]
    dates_delivered: list[str]
    images_unattributed: int
    plant_attribution: str


class PlantRegistryReference(BaseModel):
    """A registered plant registry (:class:`PlantRegistry`) by name and content digest."""

    model_config = ConfigDict(extra="forbid")

    name: str
    digest: str


@dataclass
class Assignment:
    """One image's plant assignment, the image named by its file name in its date folder.

    ``distance_m`` is the GPS distance in meters from the capture to the plant its matching
    measured (the plant assigned, or for an unmapped capture the nearest one), ``None`` when no
    distance was measured.
    """

    __pydantic_config__ = _DECODED

    image: str
    date_folder: str
    plant_id: str | None
    accession_name: str | None
    source: GpsSource
    distance_m: float | None

    @property
    def stem(self) -> str:
        """The image's stem, its file name's."""
        return Path(self.image).stem

    @classmethod
    def of(
        cls,
        stamp: ImageStamp,
        plant: PlantRecord | None,
        source: GpsSource,
        distance_m: float | None,
    ) -> Assignment:
        """Return an assignment of ``stamp`` to ``plant`` (``None`` for none) from ``source``."""
        return cls(
            image=stamp.name,
            date_folder=stamp.date_folder,
            plant_id=plant.plant_id if plant else None,
            accession_name=plant.accession_name if plant else None,
            source=source,
            distance_m=distance_m,
        )


def assignment_is_attributed(
    assignment: PlantRecord | Assignment | DetectionAssignment,
) -> bool:
    """Return whether ``assignment`` names a real plant: a non-empty ``plant_id``.

    ``assignment`` is a :class:`PlantRecord` or an assignment record; a plant CSV's own blank
    column and an unmapped capture (``plant_id=None``) both fail the rule.
    """
    return bool(assignment.plant_id)


def require_named_plants(plants: list[PlantRecord]) -> None:
    """Refuse (``ValueError``) a registry carrying a blank or duplicate ``plant_id``.

    Blank is :func:`assignment_is_attributed`'s rule; the refusal names the offending accession or
    names.
    """
    for p in plants:
        if not assignment_is_attributed(p):
            raise ValueError(
                f"the plant registry carries a blank plant_id (accession {p.accession_name!r}); "
                "every plant this delivery attributes detections to must carry a plant_id"
            )
    counts = Counter(p.plant_id for p in plants)
    duplicates = sorted(n for n, k in counts.items() if k > 1)
    if duplicates:
        raise ValueError(
            f"the plant registry carries duplicate plant_id(s) {duplicates}; two rows sharing "
            "one identity would merge two trees' detections into one aggregation row"
        )


@dataclass
class MappingBuild:
    """One build's provenance plus its per-date assignments: the whole persisted record, in memory.

    ``dataset_id`` is the dataset identity record's minted id; ``dataset_root`` is spelled
    against the owning project by :func:`~tcip_mcp.registry_paths.stored_path`.
    ``capture_digests`` is the identity of the captures each date was built from
    (:func:`capture_digests`, one entry per stem), and ``match_radius`` the radius each date's
    positioned captures were matched within (:func:`date_match_radius`): every date the build
    walked when one was stated, and a date with no positioned capture has no derived one.
    """

    __pydantic_config__ = _DECODED

    name: str
    dataset_root: str
    dataset_id: str
    built_at: AwareDatetime
    dates_requested: list[str] | None
    match_radius: dict[str, MatchRadius]
    plant_registry: PlantRegistryReference
    """The named :data:`PLANT_REGISTRY_STORE` record this build's plants were read from."""
    capture_digests: dict[str, dict[str, str]]
    unreadable: dict[str, list[str]]
    assignments: dict[str, list[Assignment]]
    supersedes: str | None = None
    """The archived digest this record replaced, when a same-name rebuild was told to
    ``supersede`` a record a delivery event still cited; ``None`` otherwise."""
    record_sha256: str = ""
    """The digest of the stored record this build was decoded from (:meth:`from_record`); blank
    on a build not yet read back. Never written into the record itself."""

    plant_attribution: ClassVar[str] = "image"
    """The granularity at which this mapping's own assignments attribute objects to plants: one
    frame per plant, by the capture protocol the platform assumes for a walked, EXIF-geolocated
    scene. Never a field of the record."""

    @property
    def dates(self) -> list[str]:
        """The dates this build walked, in walk order: those it holds assignments for."""
        return list(self.assignments)

    def to_record(self) -> dict:
        """Return the stored document: every field but ``record_sha256``."""
        return _MAPPING_RECORD.dump_python(
            self, mode="json", exclude={"record_sha256"}
        )

    @classmethod
    def from_record(
        cls, raw: JsonValue, project: Path | str, name: str
    ) -> MappingBuild:
        """Return the build the stored record ``raw`` under ``name`` holds, with its digest.

        The digest is :func:`record_digest`'s. A document that does not decode as this record
        raises ``ValueError`` naming the project and the name.
        """
        from pydantic import ValidationError

        try:
            build = _MAPPING_RECORD.validate_json(encode_record(raw))
        except ValidationError as exc:
            raise ValueError(
                f"plant mapping {name!r} under {project} is not a record this "
                f"reader decodes: {exc}"
            ) from exc
        return replace(build, record_sha256=record_digest(raw))

    def unattributed(self, dates: Iterable[str] | None = None) -> int:
        """Return the number of assignments over ``dates`` naming no plant.

        ``dates`` is every date this mapping holds when ``None``; an assignment names no plant
        when :func:`assignment_is_attributed` is false.
        """
        scope = self.dates if dates is None else dates
        return sum(
            1
            for date in scope
            for a in self.assignments.get(date, [])
            if not assignment_is_attributed(a)
        )

    def _date_counts(self, date: str) -> DateCounts:
        """Return ``date``'s own counts."""
        assignments = self.assignments.get(date, [])
        dists = [a.distance_m for a in assignments if a.distance_m is not None]
        return DateCounts(
            n_images=len(assignments),
            n_unattributed=self.unattributed([date]),
            avg_distance_m=round(sum(dists) / len(dists), 2)
            if dists
            else None,
        )

    def served(self) -> ServedMapping:
        """Return this build's served view, never its per-image assignments."""
        return ServedMapping(
            name=self.name,
            per_date={date: self._date_counts(date) for date in self.dates},
            images_unattributed=self.unattributed(),
            unreadable=self.unreadable,
            match_radius=self.match_radius,
        )

    def delivery_disclosure(
        self, verified: MappingVerification, dates: Iterable[str]
    ) -> PlantMappingDisclosure:
        """Return the ``plant_mapping`` disclosure a delivery carries.

        It holds this build's own identity, ``verify_mapping_inputs``'s disclosure ``verified``,
        and the delivery's own unattributed-capture count over ``dates`` (a delivery's own
        delivered dates, never the mapping's full span).
        """
        dates_delivered = sorted(dates)
        return PlantMappingDisclosure(
            name=self.name,
            dataset_id=self.dataset_id,
            dataset_root=self.dataset_root,
            built_at=self.built_at,
            record_sha256=self.record_sha256,
            match_radius=self.match_radius,
            capture_digests=self.capture_digests,
            captures_unverified=verified.captures_unverified,
            plant_csvs_unverified=verified.plant_csvs_unverified,
            dates_delivered=dates_delivered,
            images_unattributed=self.unattributed(dates_delivered),
            plant_attribution=self.plant_attribution,
        )


_MAPPING_RECORD = TypeAdapter(MappingBuild)


# ── EXIF extraction ──────────────────────────────────────────────────────

_GPS_IFD_TAG = 0x8825


def _exif_dms_to_decimal(
    dms: Iterable[SupportsFloat] | None,
    ref: object,
    *,
    axis: str,
    negative: str,
    positive: str,
    path: Path,
) -> float | None:
    """Return degrees, minutes and seconds with their hemisphere reference as decimal degrees.

    The degrees are signed; ``None`` when ``dms`` is absent. Raises ``ValueError`` naming ``path``
    for a value that is not three numbers or a reference that is not ``negative`` or ``positive``.
    """
    if dms is None:
        return None
    try:
        d, m, s = (float(x) for x in dms)
    except ValueError as exc:
        raise ValueError(
            f"{path}'s GPS {axis} {dms!r} is not degrees, minutes and seconds"
        ) from exc
    if ref not in (negative, positive):
        raise ValueError(
            f"{path}'s GPS {axis} reference {ref!r} is neither {negative!r} nor "
            f"{positive!r}, so its hemisphere is not stated"
        )
    val = d + m / 60 + s / 3600
    return -val if ref == negative else val


def read_image_stamp(path: Path, date_folder: str) -> ImageStamp:
    """Return one photograph's EXIF stamp.

    A photograph PIL cannot open (a HEIC with no decoder installed, a locked file) becomes a
    stamp with ``readable=False``; one that opens and carries no EXIF stays ``readable=True``
    with ``None`` fields. A capture time, GPS coordinate or positioning error present in
    malformed form, and a GPS coordinate whose hemisphere reference is missing, raise
    ``ValueError`` naming the image.
    """
    from tcip_mcp.pipelines.image_utils import (
        exif_capture_time,
        parse_capture_time,
    )

    stamp = ImageStamp(
        date_folder=date_folder,
        kind="photo",
        name=path.name,
        timestamp=None,
        lat=None,
        lon=None,
        h_pos_err=None,
        readable=True,
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
                raise ValueError(
                    f"{path}'s capture time {dt_raw!r} does not read"
                )

        gps_raw = exif.get_ifd(_GPS_IFD_TAG)
        if gps_raw:
            gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
            stamp.lat = _exif_dms_to_decimal(
                gps.get("GPSLatitude"),
                gps.get("GPSLatitudeRef"),
                axis="latitude",
                negative="S",
                positive="N",
                path=path,
            )
            stamp.lon = _exif_dms_to_decimal(
                gps.get("GPSLongitude"),
                gps.get("GPSLongitudeRef"),
                axis="longitude",
                negative="W",
                positive="E",
                path=path,
            )
            hpe = gps.get("GPSHPositioningError")
            if hpe is not None:
                try:
                    stamp.h_pos_err = float(hpe)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path}'s GPS positioning error {hpe!r} is not a number"
                    ) from exc
    return stamp


def _read_date_stamps(
    logical: dict[str, Path | BandGroupRef], date_folder: str
) -> list[ImageStamp]:
    """Return a date's stamps for every logical capture, in stem order, by its header's ``kind``.

    ``logical`` is what ``list_logical_images`` enumerated. A photograph's stamp is its EXIF, a
    band group's the manifest's own digest and member names, an array raster's bare identity (no
    EXIF to read for either).
    """
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.raster_source import SourceHeader

    stamps: list[ImageStamp] = []
    for stem in sorted(logical):
        source = logical[stem]
        kind = SourceHeader(source).kind
        if kind == "group":
            assert isinstance(source, BandGroupRef)
            manifest_sha256 = hashlib.sha256(
                source.manifest_path.read_bytes()
            ).hexdigest()
            members = tuple(sorted(source.manifest.bands.values()))
            stamps.append(
                ImageStamp(
                    date_folder=date_folder,
                    kind=kind,
                    name=source.manifest_path.name,
                    timestamp=None,
                    lat=None,
                    lon=None,
                    h_pos_err=None,
                    readable=None,
                    manifest_sha256=manifest_sha256,
                    members=members,
                )
            )
        elif kind == "photo":
            assert isinstance(source, Path)
            stamps.append(read_image_stamp(source, date_folder))
        else:
            p = source
            assert isinstance(p, Path)
            stamps.append(
                ImageStamp(
                    date_folder=date_folder,
                    kind=kind,
                    name=p.name,
                    timestamp=None,
                    lat=None,
                    lon=None,
                    h_pos_err=None,
                    readable=None,
                )
            )
    return stamps


def _capture_row(s: ImageStamp) -> list[object]:
    """Return the fields one capture's identity commits to.

    They are a manifest's own digest and the member names it claims for a band group, bare
    identity for an array raster (neither carries EXIF), EXIF for a photograph.
    """
    if s.kind == "group":
        return [s.name, s.kind, s.manifest_sha256, list(s.members)]
    if s.kind != "photo":
        return [s.name, s.kind]
    return [
        s.name,
        s.kind,
        s.timestamp.isoformat() if s.timestamp else None,
        s.lat,
        s.lon,
        s.h_pos_err,
        s.readable,
    ]


def _row_digest(row: Sequence[object]) -> str:
    return hashlib.sha256(
        json.dumps(row, sort_keys=False).encode("utf-8")
    ).hexdigest()[:16]


def capture_digests(stamps: list[ImageStamp]) -> dict[str, str]:
    """Return one digest per capture, keyed by stem, over that capture's own :func:`_capture_row`.

    An EXIF/manifest identity, never an image-content identity: it certifies the inputs the
    assignment was made from, not the pixels a phenotype was counted over.
    """
    return {s.stem: _row_digest(_capture_row(s)) for s in stamps}


# ── Plant CSV parsing ───────────────────────────────────────────────────


_COORDINATE_COLUMNS = {"lon": "WGS84_centroid_x", "lat": "WGS84_centroid_y"}

PLANT_CSV_COLUMN_FOR_FIELD = {
    f.name: _COORDINATE_COLUMNS.get(f.name, f.name)
    for f in fields(PlantRecord)
}
"""Which CSV column carries each :class:`PlantRecord` field, in the record's field order: each
field under its own name, the two coordinates under :data:`_COORDINATE_COLUMNS`."""

PLANT_CSV_COLUMNS = list(PLANT_CSV_COLUMN_FOR_FIELD.values())
"""The seven column names in header order."""


def read_plant_csv_bytes(data: bytes) -> list[PlantRecord]:
    """Parse one plant-locations CSV's own bytes into :class:`PlantRecord` rows.

    A row with no numeric lat/lon is skipped. Bytes that are not UTF-8 raise
    ``UnicodeDecodeError``.
    """
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
                plant_id=row.get(column["plant_id"], ""),
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
    """Read every ``paths`` entry that exists through :func:`read_plant_csv_bytes`.

    Each path is a plant-locations CSV; a path that is not a file is silently skipped.
    """
    records: list[PlantRecord] = []
    for path in paths:
        p = Path(path)
        if not p.is_file():
            continue
        records.extend(read_plant_csv_bytes(p.read_bytes()))
    return records


class ShapefileRows(NamedTuple):
    """A plant-locations shapefile's rows and bookkeeping.

    ``rows`` carries
    :data:`PLANT_CSV_COLUMNS`' seven keys per feature: the five attribute columns as the DBF
    value's ``str()`` verbatim (empty when the field is unresolved, exactly the rule
    :func:`_shapefile_field_value` states), and ``WGS84_centroid_x``/``WGS84_centroid_y`` as the
    reprojected floats. ``missing_fields`` names every CSV column with no resolvable source
    attribute; ``geometry_kinds`` is every geometry type actually read (``"point"``,
    ``"polygon"``, ``"multipolygon"``); ``skipped_null_geometry`` counts features with no
    geometry at all, invisible in ``rows``.
    """

    rows: list[dict[str, object]]
    missing_fields: list[str]
    geometry_kinds: list[str]
    skipped_null_geometry: int


SHAPEFILE_FIELD_DEFAULTS = {
    field: column
    for field, column in PLANT_CSV_COLUMN_FOR_FIELD.items()
    if field not in _COORDINATE_COLUMNS
}
"""Every pass-through field's default source-attribute name, its CSV column, overridable per
shapefile through :func:`read_plant_shapefile`'s own ``field_map``: an ESRI Shapefile DBF caps
field names at 10 characters, so a source rarely matches these exactly."""

# mypy reads only a literal member list; the members are SHAPEFILE_FIELD_DEFAULTS' own keys.
ShapefileField = StrEnum(  # type: ignore[misc]
    "ShapefileField", {field: field for field in SHAPEFILE_FIELD_DEFAULTS}
)
"""The pass-through fields a shapefile's ``field_map`` may name a source attribute for."""


def _shapefile_field_value(properties: dict, field_name: str | None) -> str:
    if field_name is None:
        return ""
    value = properties.get(field_name)
    return "" if value is None else str(value)


def read_plant_shapefile(
    path: Path | str,
    *,
    field_map: dict[ShapefileField, str] | None = None,
) -> ShapefileRows:
    """Read a plant-locations shapefile's own features into :data:`PLANT_CSV_COLUMNS`-shaped rows.

    Every feature's own coordinate is reprojected to WGS84 with ``always_xy=True`` (GDAL 3's
    authority-compliant EPSG:4326 axis order is lat/lon). ``pyproj.Transformer.from_crs`` takes
    the layer's own CRS object directly, so a custom projected CRS with a valid ``.prj`` converts.

    A point's own coordinate is read; a polygon's or multipolygon's centroid is read; any other
    geometry type refuses by name (naming the feature index and ``geom_type``). A feature with null
    geometry is skipped and counted in the return's ``skipped_null_geometry``. Raises
    :class:`ShapefileCrsUnknownError` when the layer's CRS cannot be resolved, and
    :class:`ShapefileUnreadableError` when fiona cannot open the shapefile at all.

    Returns rows with the DBF value's own string, verbatim.
    """
    import fiona
    from pyproj import Transformer
    from shapely.geometry import shape

    shp_path = Path(path)
    source_fields = dict(SHAPEFILE_FIELD_DEFAULTS)
    source_fields.update((field_map or {}).items())

    geom_kinds: set[str] = set()
    rows: list[dict[str, object]] = []
    skipped_null_geometry = 0
    try:
        opened = fiona.open(str(shp_path))
    except Exception as exc:
        raise ShapefileUnreadableError(
            f"{shp_path}: fiona could not open this shapefile ({type(exc).__name__}: {exc}); "
            "check that the .shp, .shx and .dbf parts are all present and readable beside each "
            "other, then run shp_to_plant_csv again"
        ) from exc
    with opened as layer:
        # fiona reports a missing or unparseable .prj as an empty CRS, raising for neither.
        if not layer.crs:
            raise ShapefileCrsUnknownError(
                f"{shp_path}: no resolvable coordinate reference system (missing or unreadable "
                ".prj); refusing to guess a CRS, supply a .prj alongside the .shp"
            )
        # always_xy keeps both sides in (lon, lat) order; EPSG:4326's authority-declared order is
        # (lat, lon), and without this every coordinate lands in the wrong column.
        transform = Transformer.from_crs(
            layer.crs, "EPSG:4326", always_xy=True
        )

        available = set(layer.schema["properties"])
        resolved_fields: dict[str, str | None] = {}
        for csv_col, shp_field in source_fields.items():
            if shp_field in available:
                resolved_fields[csv_col] = shp_field
            elif shp_field[:10] in available:
                # A source field name over 10 chars is truncated by the ESRI Shapefile DBF
                # driver; try that truncated form before giving up.
                resolved_fields[csv_col] = shp_field[:10]
            else:
                resolved_fields[csv_col] = None
        missing_fields = sorted(
            c for c, f in resolved_fields.items() if f is None
        )

        for index, feature in enumerate(layer):
            if feature.geometry is None:
                skipped_null_geometry += 1
                continue
            geom = shape(feature.geometry)
            if geom.geom_type == "Point":
                geom_kinds.add("point")
                x, y = geom.x, geom.y
            elif geom.geom_type in ("Polygon", "MultiPolygon"):
                geom_kinds.add(
                    "polygon"
                    if geom.geom_type == "Polygon"
                    else "multipolygon"
                )
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
                column[field]: _shapefile_field_value(
                    properties, resolved_fields[field]
                )
                for field in SHAPEFILE_FIELD_DEFAULTS
            }
            row[column["lon"]] = lon
            row[column["lat"]] = lat
            rows.append(row)

    return ShapefileRows(
        rows=rows,
        missing_fields=missing_fields,
        geometry_kinds=sorted(geom_kinds),
        skipped_null_geometry=skipped_null_geometry,
    )


def _maybe_float(x: str | None) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def registry_content_digest(plants: list[PlantRecord]) -> str:
    """Return the sha256 over ``plants``' own fields, order-independent.

    Two registrations of the same plants under two paths (or a different row order in one file)
    digest equal.
    """
    from dataclasses import asdict

    rows = sorted(
        (asdict(p) for p in plants),
        key=lambda r: json.dumps(r, sort_keys=True),
    )
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode("utf-8")
    ).hexdigest()


class ShapefileCrsUnknownError(ValueError):
    """A shapefile's coordinate reference system cannot be resolved.

    fiona reports a missing ``.prj``, and a ``.prj`` whose WKT does not parse, as the same falsy
    ``layer.crs``. No CRS is ever guessed.
    """


class ShapefileUnreadableError(ValueError):
    """fiona could not open a shapefile at all.

    The causes are a missing or unreadable ``.shx``/``.dbf`` part, a file that is not a
    shapefile, a driver or IO failure. Raised in place of whatever fiona threw. A ``ValueError``
    subclass.
    """


class NoGeoreferencedPlantsError(Exception):
    """A registration names a CSV that parsed no georeferenced, named plant.

    ``paths`` lists every file that failed the parse.
    """

    def __init__(self, message: str, *, paths: list[str]) -> None:
        """Take the refusal's message and every path that failed the parse."""
        super().__init__(message)
        self.paths = paths


class PlantRegistryNameConflictError(Exception):
    """A registration's name is already taken by a different set of plants."""


PLANT_REGISTRY_STORE = "plant_registries"

NAME_SEGMENT = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
"""A caller-chosen registry or mapping name's one legal shape: lowercase letters, digits, single
hyphens between groups."""

ARCHIVED_NAME_SEGMENT = re.compile(
    NAME_SEGMENT.pattern.removesuffix("$") + r"@[0-9a-f]{12}$"
)
"""An archived plant-mapping name's own shape: a legal :data:`NAME_SEGMENT` plus ``@`` and the
twelve-hex-digit prefix of the record digest it archived; it never matches plain
:data:`NAME_SEGMENT`."""


def _named(name: str, *patterns: re.Pattern[str]) -> tuple[str]:
    """Return ``name`` as a key's one part; ``ValueError`` when it matches none of ``patterns``."""
    if not any(p.fullmatch(name) for p in patterns):
        raise ValueError(
            f"name {name!r} is not lowercase letters, digits and single hyphens "
            f"({patterns[0].pattern})"
        )
    return (name,)


def plant_registry_key(project: Path | str, name: str) -> Key:
    """Return the key of one project's named plant registry.

    The registry is identity state for a plant-locations CSV set, under ``.tcip/state`` like
    :func:`plant_mapping_key`. A name outside ``NAME_SEGMENT`` raises ``ValueError``.
    """
    return Key(
        PLANT_REGISTRY_STORE,
        str(project_state_dir(project)),
        _named(name, NAME_SEGMENT),
    )


class PlantRegistryNotFoundError(ValueError):
    """A plant registry name nothing is registered under in the project."""


@with_config(ConfigDict(extra="forbid"))
class RegistryCsv(TypedDict):
    """One plant-locations CSV a registry names: its path, content digest and plant count."""

    path: str
    sha256: str
    n_plants: int


@with_config(ConfigDict(extra="forbid"))
class PlantRegistry(TypedDict):
    """A plant registry record, the one :func:`plant_registry_key` stores.

    It names its plant-locations CSVs, their ``crop`` and ``site``, the plant count, the content
    digest over every parsed row and when it was registered.
    """

    name: str
    crop: str
    site: str
    csvs: list[RegistryCsv]
    n_plants: int
    digest: str
    registered_at: AwareDatetime


_PLANT_REGISTRY = TypeAdapter(PlantRegistry)


def load_registry(project: Path | str, name: str) -> PlantRegistry:
    """Return the named registry record.

    A name nothing is stored under (never registered, or deleted since a mapping recorded it)
    refuses (:class:`PlantRegistryNotFoundError`) naming it; a record that does not validate as
    :class:`PlantRegistry` refuses with ``pydantic.ValidationError`` naming each field.
    """
    record = tcip_store.read(plant_registry_key(project, name), default=None)
    if record is None:
        raise PlantRegistryNotFoundError(
            f"plant registry not found: {name!r} under {project}; register it with "
            "register_plant_registry before naming it."
        )
    return _PLANT_REGISTRY.validate_python(record)


def registry_csv_entries(
    record: PlantRegistry, project: Path | str
) -> list[RegistryCsv]:
    """Return the CSV entries a loaded registry record carries, each path resolved.

    The record is ``project``'s; each ``path`` is resolved through
    :func:`~tcip_mcp.registry_paths.resolved_registry_path`.
    """
    from tcip_mcp.registry_paths import resolved_registry_path

    return [
        RegistryCsv(
            path=str(resolved_registry_path(project, e["path"])),
            sha256=e["sha256"],
            n_plants=e["n_plants"],
        )
        for e in record["csvs"]
    ]


def registry_entries_or_refusal(
    build: MappingBuild,
    project: Path | str,
) -> tuple[list[RegistryCsv], str | None]:
    """Return the CSV entries ``build.plant_registry`` names, or the refusal when untrusted.

    The refusal names the registry, the mapping and ``project``: the named registry no longer
    loads, or its own ``digest`` no longer matches ``build.plant_registry.digest``.
    """
    reference = build.plant_registry
    try:
        record = load_registry(project, reference.name)
    except PlantRegistryNotFoundError as exc:
        return [], f"mapping {build.name!r} names it: {exc}"
    if record["digest"] != reference.digest:
        return [], (
            f"plant registry {reference.name!r} under {project} named by mapping "
            f"{build.name!r} has moved (built against digest {reference.digest!r}, now "
            f"{record['digest']!r}); rebuild the mapping against the current registry"
        )
    return registry_csv_entries(record, project), None


def verify_registry_csv_bytes(
    registry_entries: list[RegistryCsv],
) -> tuple[list[str], str | None, dict[str, bytes]]:
    """Check each of ``registry_entries``'s recorded bytes against what is on disk now.

    The entries are :func:`registry_csv_entries`'; each present file's bytes are read exactly
    once. Returns every missing path, the first rewritten file (or ``None`` when every present
    path's bytes still match what was registered), and the verified bytes this call read for every
    present path, keyed by the entry's own ``path``.
    """
    missing: list[str] = []
    rewritten: str | None = None
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
    csv_paths: list[Path],
    project: Path | str,
) -> tuple[list[RegistryCsv], str, int]:
    """Parse the absolute ``csv_paths`` into registry entries, their digest and plant count.

    Returns the registry's own entries, each ``path`` spelled against ``project`` by
    :func:`~tcip_mcp.registry_paths.stored_path`, the content digest over every parsed row and the
    total plant count, committing nothing.

    Raises :class:`NoGeoreferencedPlantsError`, naming every file that parsed no georeferenced,
    named plant or is not UTF-8 text (a binary file, a shapefile's own ``.shp``/``.shx``/``.dbf``
    included).
    """
    from tcip_mcp.registry_paths import stored_path

    failed: list[str] = []
    csvs_meta: list[RegistryCsv] = []
    all_plants: list[PlantRecord] = []
    for p in csv_paths:
        data = p.read_bytes() if p.is_file() else b""
        try:
            records = read_plant_csv_bytes(data)
        except UnicodeDecodeError:
            records = []
        if not records:
            failed.append(str(p))
            continue
        all_plants.extend(records)
        csvs_meta.append(
            RegistryCsv(
                path=stored_path(p, project),
                sha256=hashlib.sha256(data).hexdigest(),
                n_plants=len(records),
            )
        )
    if failed:
        raise NoGeoreferencedPlantsError(
            f"{failed} parsed no georeferenced, named plant, or is not UTF-8 text (need columns "
            "plant_id, WGS84_centroid_x, WGS84_centroid_y with usable values); register only "
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
) -> PlantRegistry:
    """Register ``csv_paths`` under ``name`` in this project's plant registry; return the record.

    Parses every path through :func:`parse_plant_registry_csvs`
    (:class:`NoGeoreferencedPlantsError` names the file that parsed no georeferenced, named
    plant); the record (:class:`PlantRegistry`) holds those entries plus ``crop``, ``site``,
    ``registered_at`` and the parsed content's digest. The read-then-write is one transaction
    (:func:`tcip_store.transaction`, this store's own ``concurrency="cas"``): a second
    registration under a taken name returns the existing record unchanged when the digest
    matches, and raises :class:`PlantRegistryNameConflictError` otherwise, naming the two digests.
    """
    csvs_meta, digest, n_plants = parse_plant_registry_csvs(csv_paths, project)
    key = plant_registry_key(project, name)
    with tcip_store.transaction(key) as txn:
        stored = txn.read(key, default=None)
        if stored is not None:
            existing = _PLANT_REGISTRY.validate_python(stored)
            if existing["digest"] == digest:
                return existing
            raise PlantRegistryNameConflictError(
                f"plant registry {name!r} under {project} already names different plants "
                f"(digest {existing['digest']!r}, this registration would write "
                f"{digest!r}); register under a new name"
            )
        record = PlantRegistry(
            name=name,
            crop=crop,
            site=site,
            csvs=csvs_meta,
            n_plants=n_plants,
            digest=digest,
            registered_at=now(),
        )
        txn.write(key, _PLANT_REGISTRY.dump_python(record, mode="json"))
    return record


def stems_delivery_reads(
    rows: Iterable[Assignment], bucket: Bucket
) -> set[str]:
    """Return the attributed stems of ``rows`` that carry a prediction document in ``bucket``.

    ``rows`` are one date's assignment rows, attributed as :func:`assignment_is_attributed` calls
    them; a stem counts when ``bucket``'s record names a prediction document for it, never by
    whether the underlying image still exists on disk.
    """
    pred_stems = set(bucket.documents)
    return {
        row.stem
        for row in rows
        if assignment_is_attributed(row) and row.stem in pred_stems
    }


def nearest_plant(
    position: tuple[float, float],
    plants: list[PlantRecord],
    *,
    within_m: float,
    skip: set[int] | frozenset[int] = frozenset(),
) -> tuple[int | None, float | None]:
    """Return the index of the plant nearest the ``(lat, lon)`` ``position``, and its distance.

    The index is into ``plants``, those in ``skip`` passed over, and the distance in meters; the
    index is ``None`` when that plant lies farther than ``within_m`` and both are ``None`` when no
    plant is left to measure.
    """
    lat, lon = position
    distances = [
        (WGS84.inv(lon, lat, p.lon, p.lat)[2], i)
        for i, p in enumerate(plants)
        if i not in skip
    ]
    if not distances:
        return None, None
    distance, index = min(distances)
    return (index if distance <= within_m else None), distance


# ── Sequence anchoring ─────────────────────────────────────────────────


# Iglewicz & Hoaglin's modified z-score outlier test (1993, ASQC Quality Press), tested in
# log-space against the date's own consecutive-gap ratios; 3.5 is the convention's own cutoff.
_ROW_BREAK_MODIFIED_Z = 3.5


def _derive_row_break_threshold(gaps: list[float]) -> float | None:
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
    # A near-zero gap carries no walking-pace information, so it is no candidate split.
    candidates = [
        (math.log(hi / lo), lo, hi)
        for lo, hi in pairwise(sorted_gaps)
        if lo > 1e-9
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda candidate: candidate[0])
    best_log_ratio, best_lo, best_hi = best

    other_log_ratios = [c[0] for c in candidates if c is not best]
    if not other_log_ratios:
        # Only one candidate split to compare against: no basis to call it a genuine transition.
        return None
    from scipy.stats import median_abs_deviation

    median_other = statistics.median(other_log_ratios)
    mad = float(median_abs_deviation(other_log_ratios, scale="normal"))
    if mad > 0:
        modified_z = (best_log_ratio - median_other) / mad
        if modified_z < _ROW_BREAK_MODIFIED_Z:
            return None
    elif best_log_ratio <= median_other:
        # With no spread in the other steps, only a strictly larger jump splits (MAD's limit at 0).
        return None

    return best_lo + (best_hi - best_lo) / 2


def _segment_runs(stamps: list[ImageStamp]) -> list[list[ImageStamp]]:
    """Break an ordered list of stamps into row runs on GPS jumps.

    A jump breaks a run when it stands out from that date's own walking gaps (see
    ``_derive_row_break_threshold``).
    """
    if not stamps:
        return []
    pairs: list[tuple[ImageStamp, float | None]] = []
    for prev, cur in pairwise(stamps):
        d = None
        if prev.position is not None and cur.position is not None:
            (lat1, lon1), (lat2, lon2) = prev.position, cur.position
            d = WGS84.inv(lon1, lat1, lon2, lat2)[2]
        pairs.append((cur, d))

    threshold = _derive_row_break_threshold(
        [d for _, d in pairs if d is not None]
    )

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


class NoMatchRadiusError(ValueError):
    """A date's positioned captures report no positioning error and no match radius is stated."""


def date_match_radius(
    stamps: list[ImageStamp], stated: StatedRadius | None
) -> MatchRadius | None:
    """Return the radius one date's captures are matched to plants within.

    The radius is ``stated`` when given, else the largest horizontal positioning error
    (``h_pos_err``) its positioned captures report, each capture's position lying within its own
    reported error of where it was taken. ``None`` for a date stating none and no capture of which
    carries a position, which matches none.

    Raises :class:`NoMatchRadiusError` naming ``match_radius`` when nothing is stated and a
    positioned capture reports no positioning error.
    """
    positioned = [s for s in stamps if s.position is not None]
    if stated is not None:
        return MatchRadius(
            value=stated.value, source="stated", rationale=stated.rationale
        )
    unreported = sorted(s.name for s in positioned if s.h_pos_err is None)
    if unreported:
        raise NoMatchRadiusError(
            f"{unreported} on {positioned[0].date_folder} report no horizontal positioning "
            "error, so no match radius derives from them: state match_radius {value, "
            "rationale} (meters) for the build"
        )
    errors = [s.h_pos_err for s in positioned if s.h_pos_err is not None]
    if not errors:
        return None
    return MatchRadius(value=max(errors), source="h_pos_err", rationale=None)


def assign_plants(
    stamps: list[ImageStamp],
    plants: list[PlantRecord],
    *,
    radius: MatchRadius | None,
) -> list[Assignment]:
    """Assign each image to a plant by sequence-anchored NN matching within ``radius``.

    Each image takes its nearest unclaimed plant, else its nearest plant, else none; the
    assignments come back in ``stamps`` order. A stamp with no position (a band group or an array
    raster, or a photograph with no usable EXIF position) is unmapped.
    """
    out: list[Assignment] = []
    claimed: set[int] = set()
    for run in _segment_runs(_order_by_time(stamps)):
        for s in run:
            if s.position is None or radius is None or not plants:
                out.append(Assignment.of(s, None, "unmapped", None))
                continue
            index, distance = nearest_plant(
                s.position, plants, within_m=radius.value, skip=claimed
            )
            if index is not None:
                claimed.add(index)
                out.append(
                    Assignment.of(s, plants[index], "sequence", distance)
                )
                continue
            index, distance = nearest_plant(
                s.position, plants, within_m=radius.value
            )
            out.append(
                Assignment.of(s, None, "unmapped", distance)
                if index is None
                else Assignment.of(
                    s, plants[index], "nearest_neighbor", distance
                )
            )

    by_image = {(a.date_folder, a.image): a for a in out}
    return [by_image[(s.date_folder, s.name)] for s in stamps]


# ── Whole-dataset driver ────────────────────────────────────────────────


def grid_pitch_m(plants: list[PlantRecord]) -> float:
    """Return the planting grid pitch (m).

    The pitch is the median over the plants of each one's geodesic distance to its nearest
    neighbor, the mean of the two middle values for an even count; 0.0 for fewer than two plants.
    """
    if len(plants) < 2:
        return 0.0
    nearest = (
        nearest_plant((p.lat, p.lon), plants, within_m=math.inf, skip={i})
        for i, p in enumerate(plants)
    )
    return statistics.median(cast(float, distance) for _, distance in nearest)


class NoMatchToleranceError(ValueError):
    """No match tolerance is stated and the plant layout derives none."""


class MatchTolerance(BaseModel):
    """A nearest-neighbor match tolerance (meters) and the branch of its resolution behind it.

    The branch is :func:`resolve_nn_tolerance_m`'s.
    """

    model_config = ConfigDict(extra="forbid")

    value: float
    source: ToleranceSource


def resolve_nn_tolerance_m(
    plants: list[PlantRecord], stated: float | None = None
) -> MatchTolerance:
    """Return the tolerance (meters) a capture or detection is matched to a plant within.

    Derived as half the grid pitch (:func:`grid_pitch_m`), ``"grid_pitch"``: the farthest a
    position can lie from a plant and still be nearer it than any neighbor on the grid. A stated
    value is honored (``"stated"``) up to that ceiling and capped at it (``"stated_capped"``).
    Raises :class:`NoMatchToleranceError` naming ``nn_tolerance_m`` when nothing is stated and the
    layout carries too few plants to derive a pitch from.
    """
    ceiling = grid_pitch_m(plants) / 2
    if stated is None:
        if ceiling <= 0:
            raise NoMatchToleranceError(
                "the plant layout carries fewer than two georeferenced plants, so no grid pitch "
                "derives a match tolerance: state nn_tolerance_m (meters)"
            )
        return MatchTolerance(value=ceiling, source="grid_pitch")
    if 0 < ceiling < stated:
        return MatchTolerance(value=ceiling, source="stated_capped")
    return MatchTolerance(value=stated, source="stated")


def build_mapping(
    dataset_root: Path | str,
    plant_csv_paths: list[Path],
    *,
    name: str,
    dataset_id: str,
    project: Path | str,
    plant_registry: PlantRegistryReference,
    dates: list[str] | None = None,
    match_radius: StatedRadius | None = None,
) -> MappingBuild:
    """Build one project's named plant mapping: per-date assignments plus their provenance.

    The provenance binds the record to the inputs it was built from.
    ``name``/``dataset_root``/``dataset_id`` are taken as already resolved, and ``project`` is the
    project the mapping belongs to. ``plant_csv_paths`` are the files this build reads the plants
    from, those of the registry ``plant_registry`` names; ``plant_registry`` is the reference
    stored on the record in their place.

    Each date's captures are matched within its :func:`date_match_radius`, ``match_radius`` when
    stated and derived from the date's own positioning errors when not, which the record carries
    per date; :class:`NoMatchRadiusError` propagates.

    The captures are ``dataset_root``'s, those of ``dates`` when it is given
    (:func:`~tcip_mcp.pipelines.image_utils.scan_captures`), each enumerated through
    ``image_utils.list_logical_images``, so a band raster or a band group
    ingested under a mapped date is a capture the identity sees; its
    :class:`~tcip_mcp.pipelines.image_utils.AmbiguousImageStemError` propagates.

    Raises :class:`UngeoreferencedCaptureError` naming the dataset's image tree: when the requested
    dates carry no capture at all, and with :func:`ungeoreferenced_capture_message` (naming any
    capture PIL could not open before the position clause) when every capture that was read
    carries no position this door reads.
    """
    from tcip_mcp.dataset_layout import image_root
    from tcip_mcp.pipelines.image_utils import (
        list_logical_images,
        scan_captures,
    )

    plant_csv_paths = [Path(p) for p in plant_csv_paths]
    plants = read_plant_csvs(plant_csv_paths)

    images = image_root(dataset_root)
    assignments: dict[str, list[Assignment]] = {}
    radii: dict[str, MatchRadius] = {}
    capture_digests_by_date: dict[str, dict[str, str]] = {}
    unreadable: dict[str, list[str]] = {}
    n_stamps = 0
    n_positioned = 0
    captures = scan_captures(dataset_root, list_logical_images, dates)
    for date, logical in captures.items():
        stamps = _read_date_stamps(logical, date)
        n_stamps += len(stamps)
        n_positioned += sum(1 for s in stamps if s.position is not None)
        capture_digests_by_date[date] = capture_digests(stamps)
        unreadable[date] = sorted(
            s.name for s in stamps if s.kind == "photo" and s.readable is False
        )
        radius = date_match_radius(stamps, match_radius)
        if radius is not None:
            radii[date] = radius
        assignments[date] = assign_plants(stamps, plants, radius=radius)

    if n_stamps == 0:
        raise UngeoreferencedCaptureError(
            f"no capture under {images} on the requested dates"
        )
    if n_positioned == 0:
        all_unreadable = sorted(
            {name for names in unreadable.values() for name in names}
        )
        raise UngeoreferencedCaptureError(
            ungeoreferenced_capture_message(str(images), all_unreadable)
        )

    from tcip_mcp.registry_paths import stored_path

    return MappingBuild(
        name=name,
        dataset_root=stored_path(dataset_root, project),
        dataset_id=dataset_id,
        built_at=now(),
        dates_requested=list(dates) if dates is not None else None,
        match_radius=radii,
        plant_registry=plant_registry,
        capture_digests=capture_digests_by_date,
        unreadable=unreadable,
        assignments=assignments,
    )


PLANT_MAPPING_STORE = "plant_mapping"


def plant_mapping_key(project: Path | str, name: str) -> Key:
    """Return the key of one project's named plant-mapping build, under its ``.tcip/state``.

    The key is addressed by the project that owns it: a dataset can be read by more than one
    project, and each project's mapping is its own. A name outside ``NAME_SEGMENT`` and the
    archived ``ARCHIVED_NAME_SEGMENT`` raises ``ValueError``.
    """
    return Key(
        PLANT_MAPPING_STORE,
        str(project_state_dir(project)),
        _named(name, NAME_SEGMENT, ARCHIVED_NAME_SEGMENT),
    )


def plant_mapping_names(project: Path | str) -> list[str]:
    """Return every mapping name persisted under this project that ``NAME_SEGMENT`` admits, sorted.

    The names come from the store's own key listing.
    """
    root = str(project_state_dir(project))
    names = (
        key.parts[-1] for key in tcip_store.keys(PLANT_MAPPING_STORE, root)
    )
    return sorted(name for name in names if NAME_SEGMENT.fullmatch(name))


def archived_mapping_name(name: str, record_sha256: str) -> str:
    """Return the key name ``name``'s cited record ``record_sha256`` is archived under."""
    return f"{name}@{record_sha256[:12]}"


def record_digest(record: JsonValue) -> str:
    """Return the digest a stored mapping record earns, whatever its shape.

    The digest is sha256 over ``encode_record(record)``, what its receipt names.
    """
    return hashlib.sha256(encode_record(record)).hexdigest()


class MappingRebuildError(Exception):
    """A same-name rebuild would replace a mapping record a delivery event still cites.

    The event is one under this project. ``event_ids`` names every citing event; ``status`` is the
    HTTP status the refusal carries.
    """

    def __init__(
        self, message: str, *, event_ids: list[str], status: int = 409
    ) -> None:
        """Take the refusal's message, the citing event ids and its HTTP status."""
        super().__init__(message)
        self.event_ids = event_ids
        self.status = status


def _citing_delivery_event_ids(
    project: Path | str, name: str, digest: str
) -> list[str]:
    """Return the sorted ids of the delivery events citing the record (``name``, ``digest``).

    These are the events under this project whose own ``plant_mapping`` disclosure names it.
    """
    from tcip_mcp.delivery import read_delivery_events

    return sorted(
        event.event_id
        for event in read_delivery_events(project)
        if isinstance(event.plant_mapping, PlantMappingDisclosure)
        and (event.plant_mapping.name, event.plant_mapping.record_sha256)
        == (name, digest)
    )


def persist_mapping(
    build: MappingBuild,
    project: Path | str,
    *,
    supersede: bool = False,
    actor: str | None,
) -> None:
    """Write the mapping record under ``build.name``, then the receipt by ``actor`` binding it.

    The receipt binds the record to this build. The record is committed before the receipt: a
    receipt that cannot be written fails loudly (``AuditEntryNotWrittenError`` propagates) and
    leaves a record no receipt names.

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
    existing_raw = tcip_store.read(
        plant_mapping_key(project, name), default=None
    )
    archived_digest: str | None = None
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
                plant_mapping_key(
                    project, archived_mapping_name(name, existing_digest)
                ),
                existing_raw,
            )
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
            "record_sha256": record_sha256,
            "supersedes": archived_digest,
        },
        actor=actor,
        scope=project,
    )


def build_plant_mapping(
    project: Path | str,
    name: str,
    dataset_root: Path,
    plant_registry: str,
    *,
    dates: list[str] | None = None,
    match_radius: StatedRadius | None = None,
    supersede: bool = False,
    actor: str | None,
) -> MappingBuild:
    """Map the captures of ``dataset_root`` to a registry's plants and persist the mapping.

    ``dataset_root`` is an established location and the registry is ``project``'s
    ``plant_registry`` (:func:`build_mapping`); the mapping is persisted under ``name`` by
    ``actor`` (:func:`persist_mapping`, whose receipt is the act's one audit line), and the build
    is returned.

    Raises ``ValueError`` for a name the key refuses and for a root with no dataset identity
    record (:func:`~tcip_mcp.dataset_layout.require_dataset_identity`, naming
    ``register_dataset``); :class:`PlantRegistryNotFoundError`; and every refusal of the two steps.
    """
    from tcip_mcp.dataset_layout import require_dataset_identity

    plant_mapping_key(project, name)
    identity = require_dataset_identity(dataset_root)
    registry = load_registry(project, plant_registry)
    build = build_mapping(
        dataset_root,
        [Path(e["path"]) for e in registry_csv_entries(registry, project)],
        name=name,
        dataset_id=identity.id,
        project=project,
        plant_registry=PlantRegistryReference(
            name=registry["name"], digest=registry["digest"]
        ),
        dates=dates,
        match_radius=match_radius,
    )
    persist_mapping(build, project, supersede=supersede, actor=actor)
    return build


# Process-local memo of each project's log cursor and every plant_mapping_built
# (name, record_sha256) read so far; a miss re-reads the log once before refusing.
_receipt_cursor: dict[str, str] = {}
_receipt_seen: dict[str, dict[str, set[str]]] = {}


class _MappingBuilt(BaseModel):
    """The mapping a ``plant_mapping_built`` audit line's arguments name."""

    name: str
    record_sha256: str


def _scan_receipts(project: Path | str, *, after: str | None) -> None:
    from tcip_mcp.audit import acts_of

    root_key = tcip_store.canonical_path(project)
    entries, cursor = acts_of(project, ("plant_mapping_built",), after=after)
    seen = _receipt_seen.setdefault(root_key, {})
    for entry in entries:
        built = _MappingBuilt.model_validate(entry.arguments)
        seen.setdefault(built.name, set()).add(built.record_sha256)
    _receipt_cursor[root_key] = cursor


def _require_receipt(
    project: Path | str, name: str, record_sha256: str
) -> None:
    """Refuse (``ValueError``) unless a receipt under ``name`` names ``record_sha256``.

    The receipt is a ``plant_mapping_built`` event in the project's own audit log; any matching
    receipt admits, the latest or not.
    """
    root_key = tcip_store.canonical_path(project)
    if root_key not in _receipt_cursor:
        _scan_receipts(project, after=None)
    if record_sha256 not in _receipt_seen.get(root_key, {}).get(name, set()):
        _scan_receipts(project, after=_receipt_cursor.get(root_key))
    if record_sha256 not in _receipt_seen.get(root_key, {}).get(name, set()):
        raise ValueError(
            f"plant mapping {name!r} under {project} carries no plant_mapping_built "
            f"receipt naming record {record_sha256}: this record was not written by "
            "build_plant_mapping (a forged or hand-restored record, or one whose receipt could "
            "not be written); rebuild with build_plant_mapping"
        )


def load_mapping(project: Path | str, name: str) -> MappingBuild | None:
    """Return one project's named, persisted plant-mapping build, or ``None``.

    ``None`` when nothing is stored under that name.

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
    if name not in (
        build.name,
        archived_mapping_name(build.name, build.record_sha256),
    ):
        raise ValueError(
            f"the plant mapping stored under {name!r} in {project} is the record "
            f"of {build.name!r}; rebuild with build_plant_mapping"
        )
    _require_receipt(project, build.name, build.record_sha256)
    return build


def resolved_mapping_key_for_citation(
    project: Path | str,
    name: str,
    record_sha256: str,
) -> str | None:
    """Return the name under which the record a delivery event cites is stored, or ``None``.

    The record is the one the event's own ``plant_mapping.record_sha256`` cites. The name is
    ``name`` itself when the record currently stored under it still hashes to ``record_sha256``,
    the archive name a superseding rebuild moved it to (:func:`archived_mapping_name`) when that
    key holds a stored record, or ``None`` when neither does.
    """
    current = tcip_store.read(plant_mapping_key(project, name), default=None)
    if current is not None and record_digest(current) == record_sha256:
        return name
    archived_name = archived_mapping_name(name, record_sha256)
    if tcip_store.exists(plant_mapping_key(project, archived_name)):
        return archived_name
    return None


def _describe_capture(stamps_by_stem: dict[str, ImageStamp], stem: str) -> str:
    """Return one capture's name: its kind and file name, else the bare stem.

    The kind and file name come from a fresh stamp for ``stem`` in ``stamps_by_stem``.
    """
    s = stamps_by_stem.get(stem)
    if s is None:
        return stem
    if s.kind == "group":
        return f"band group manifest {s.name!r}"
    if s.kind != "photo":
        return f"raster {s.name!r}"
    return f"image {s.name!r}"


def verify_mapping_inputs(
    build: MappingBuild,
    dataset_root: Path | str,
    buckets: Mapping[str, Bucket],
    *,
    project: Path | str,
) -> MappingVerification:
    """Check a mapping's recorded inputs against what is on disk now, for the captures delivered.

    The mapping is ``project``'s; the check covers what a delivery can verify, for the captures it
    actually reads, at delivery time. ``buckets`` is the delivered buckets by recorded date.

    A mapped date no delivered bucket records is never walked (no enumeration, no EXIF):
    disclosed in ``captures_unverified`` as the bare date string, the same as a named date whose
    image folder is absent. The named dates' folders are enumerated once, in one walk
    (``image_utils.scan_captures`` over ``list_logical_images``, refusing by name on
    :class:`~tcip_mcp.pipelines.image_utils.AmbiguousImageStemError`); an enumerated stem the
    mapping's own assignment rows for that date do not name refuses, naming the date, the file(s)
    and the rebuild remedy. A recorded stem no longer enumerated is disclosed as
    ``"<date>/<name>"``, using the recorded row's own file name; so is a recorded,
    still-enumerated stem the delivery's own prediction bucket carries no document for
    (:func:`stems_delivery_reads`).

    Only a capture this delivery reads gets its fresh stamp read. For each: a capture readable when
    this mapping was built and unreadable now refuses by name. A row that recorded a plant position
    (:func:`assignment_is_attributed` true, with a ``distance_m``) refuses by name when the
    capture's fresh GPS position no longer sits ``distance_m`` from any plant of that name in the
    plant CSVs whose bytes this same call just verified, or when the fresh capture carries no GPS
    position at all. When that plant's own CSV is itself among ``plant_csvs_unverified``, the
    position is disclosed rather than compared.

    When every mapped capture of a date was read (``missing_stems`` empty and ``read_set`` equal to
    the recorded stems :func:`assignment_is_attributed` calls attributed), every capture the date
    enumerates, the unmapped ones (an array raster, a band group) included, has its digest
    (:func:`capture_digests`) recomputed and compared against the record, refusing by name on any
    capture whose digest moved; a capture verified this way is not also listed in
    ``captures_unverified``. Under a partial read those digests are not compared, so an in-place
    EXIF timestamp or band-group manifest change on a read capture goes undetected when some
    other mapped capture of the same date was not read.

    Raises :class:`MappingDeliveryError` for any of the above; otherwise returns the
    :class:`MappingVerification`, entries in ``build.dates`` order and, within a date, sorted by
    name. A missing plant CSV is disclosed in ``plant_csvs_unverified``; a rewritten one refuses
    (restore the file's registered bytes, or register the current file under a new registry name,
    rebuild the mapping against it, and deliver under that name).
    """
    from tcip_mcp.pipelines.image_utils import (
        AmbiguousImageStemError,
        list_logical_images,
        logical_image_name,
        scan_captures,
    )

    remedy = "rebuild with build_plant_mapping"

    # Plant CSVs first: the per-capture moved-position check below trusts only verified bytes.
    registry_entries, registry_refusal = registry_entries_or_refusal(
        build, project
    )
    if registry_refusal:
        raise MappingDeliveryError(registry_refusal)
    plant_csvs_unverified, rewritten_fact, verified_csv_bytes = (
        verify_registry_csv_bytes(registry_entries)
    )
    if rewritten_fact:
        raise MappingDeliveryError(
            f"{rewritten_fact}: restore the file's registered bytes, or register the current "
            "file under a new registry name, rebuild the mapping against it, and deliver under "
            "that name"
        )
    verified_plants = [
        plant
        for entry in registry_entries
        if entry["path"] not in plant_csvs_unverified
        for plant in read_plant_csv_bytes(verified_csv_bytes[entry["path"]])
    ]

    try:
        on_disk = scan_captures(
            dataset_root,
            list_logical_images,
            [date for date in build.dates if date in buckets],
        )
    except AmbiguousImageStemError as exc:
        raise MappingDeliveryError(str(exc)) from exc

    captures_unverified: list[str] = []
    for date in build.dates:
        rows = build.assignments.get(date, [])
        recorded_by_stem = {a.stem: a for a in rows}
        recorded_stems = set(recorded_by_stem)

        bucket = buckets.get(date)
        logical = on_disk.get(date)
        if bucket is None or logical is None:
            captures_unverified.append(date)
            continue

        enumerated_stems = set(logical)
        added = enumerated_stems - recorded_stems
        if added:
            names = sorted(logical_image_name(logical[s]) for s in added)
            raise MappingDeliveryError(
                f"date {date} carries capture(s) {names} this mapping's assignments do not "
                f"name: the mapping does not cover what is on disk now; {remedy}"
            )

        missing_stems = recorded_stems - enumerated_stems
        present_stems = recorded_stems & enumerated_stems
        read_set = stems_delivery_reads(rows, bucket) & present_stems
        unread_stems = present_stems - read_set
        mapped_stems = {
            s
            for s in recorded_stems
            if assignment_is_attributed(recorded_by_stem[s])
        }
        full_mapped_coverage = not missing_stems and read_set == mapped_stems

        unverified_names: set[str] = set()
        if not full_mapped_coverage:
            # An unmapped capture (a raster, a band group) is never read through predictions, so
            # it always lands here unless the whole-date digest below re-checks it instead.
            unverified_names.update(
                recorded_by_stem[s].image
                for s in (missing_stems | unread_stems)
            )

        # One observation: the whole date when its digest is re-checked below, else the read set.
        observed = _read_date_stamps(
            logical
            if full_mapped_coverage
            else {s: logical[s] for s in read_set},
            date,
        )
        stamps = [s for s in observed if s.stem in read_set]
        was_unreadable = set(build.unreadable.get(date, []))
        for s in stamps:
            row = recorded_by_stem[s.stem]
            if s.kind == "photo":
                if s.name not in was_unreadable and s.readable is False:
                    raise MappingDeliveryError(
                        f"{s.name} (date {date}) was readable when this mapping was built and "
                        "could not be read now: retry, or rebuild if it is gone"
                    )
            if assignment_is_attributed(row) and row.distance_m is not None:
                if s.position is None:
                    raise MappingDeliveryError(
                        f"{s.name} (date {date}) recorded a plant position when this "
                        "mapping was built and now carries no GPS position: rebuild, "
                        "since its assignment would differ"
                    )
                lat, lon = s.position
                distances = [
                    WGS84.inv(lon, lat, p.lon, p.lat)[2]
                    for p in verified_plants
                    if p.plant_id == row.plant_id
                ]
                if not distances:
                    # The plant's own CSV is among plant_csvs_unverified, so this capture's
                    # recorded position is disclosed as unrechecked.
                    unverified_names.add(s.name)
                    continue
                if not any(abs(d - row.distance_m) <= 1e-6 for d in distances):
                    raise MappingDeliveryError(
                        f"{s.name} (date {date}) has moved since this mapping was built: "
                        f"no plant named {row.plant_id!r} is {row.distance_m} m away now; "
                        "rebuild, since its assignment would differ"
                    )

        for name in sorted(unverified_names):
            captures_unverified.append(f"{date}/{name}")

        if full_mapped_coverage:
            new_digests = capture_digests(observed)
            recorded_digests = build.capture_digests[date]
            stamps_by_stem = {s.stem: s for s in observed}
            moved = sorted(
                _describe_capture(stamps_by_stem, stem)
                for stem in set(new_digests) | set(recorded_digests)
                if new_digests.get(stem) != recorded_digests.get(stem)
            )
            if moved:
                raise MappingDeliveryError(
                    f"date {date}: {', '.join(moved)} changed since this mapping was built; "
                    "rebuild to cover the images actually on disk"
                )

    return MappingVerification(captures_unverified, plant_csvs_unverified)


def ungeoreferenced_capture_message(
    walked: str, unreadable: Sequence[str] = ()
) -> str:
    """Return the refusal sentence for a walk whose captures carry no position this door reads.

    ``walked`` names what was walked. ``unreadable``, when given, opens the sentence by naming the
    captures PIL could not open at all, before the position clause.
    """
    prefix = ""
    if unreadable:
        prefix = (
            f"{', '.join(unreadable)} could not be opened, so their position could not be "
            "read; "
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
    """A refusal whose ``str(exc)`` is the caller-facing message and ``status`` its HTTP status."""

    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class UngeoreferencedCaptureError(_StatusError):
    """A plant mapping cannot be built or delivered from the captures at hand.

    None carries a position this door reads, or none could be read.
    """


class MappingDeliveryError(_StatusError):
    """A delivery cannot proceed from a named mapping.

    ``str(exc)`` is the caller-facing message. ``status`` is the HTTP status for it (400 by
    default, 404 for a mapping that is not stored, 409 for a store-level problem reading it).
    """


def resolve_delivery_mapping(
    project: Path | str,
    name: str,
    buckets: Mapping[str, Bucket],
) -> tuple[MappingBuild, MappingVerification]:
    """Load the named mapping and verify it for a delivery of ``buckets``.

    A delivered bucket's recorded date the mapping does not cover is refused (``buckets`` keyed
    by those dates); the delivered buckets' one dataset root is resolved and required to carry the
    mapping's own minted dataset id; then the mapping's recorded inputs are verified against that
    resolved root for exactly the captures the buckets read.

    A delivery whose delivered dates attribute nothing (:func:`assignment_is_attributed` false for
    every one of them, a date recorded with no capture at all included) refuses on the record's own
    evidence, before ``verify_mapping_inputs`` re-reads anything.

    Returns the loaded build and :func:`verify_mapping_inputs`'s disclosure; raises
    :class:`MappingDeliveryError`, naming the remedy, for every case a delivery must not proceed
    from, a ``StoreError`` reading the mapping store included.
    """
    from tcip_mcp.buckets import shared_root
    from tcip_mcp.dataset_layout import require_dataset_identity
    from tcip_store import StoreError

    try:
        mapping_build = load_mapping(project, name)
    except (StoreError, ValueError) as exc:
        raise MappingDeliveryError(
            f"could not read mapping {name!r}: {exc}", status=409
        ) from exc
    if mapping_build is None:
        raise MappingDeliveryError(
            f"mapping not found: {name!r}; build one with build_plant_mapping before "
            "computing phenology",
            status=404,
        )

    missing_dates = [d for d in buckets if d not in mapping_build.dates]
    if missing_dates:
        raise MappingDeliveryError(
            f"the delivered buckets record date(s) {missing_dates} the mapping {name!r} does not "
            "cover; rebuild the mapping to cover them, or drop those buckets"
        )

    try:
        delivered_root = shared_root(buckets.values())
        delivered_identity = require_dataset_identity(delivered_root)
    except ValueError as exc:
        raise MappingDeliveryError(str(exc)) from exc
    if delivered_identity.id != mapping_build.dataset_id:
        raise MappingDeliveryError(
            f"the predictions under {delivered_root} belong to a different dataset than the "
            f"mapping {name!r} was built over (mapping dataset_root "
            f"{mapping_build.dataset_root!r}, delivered dataset root {str(delivered_root)!r})"
        )

    delivered_assignments = [
        a for date in buckets for a in mapping_build.assignments.get(date, [])
    ]
    if not delivered_assignments or not any(
        assignment_is_attributed(a) for a in delivered_assignments
    ):
        # A no-capture-at-all date is named below only inside this nothing-attributed scope.
        empty_dates = sorted(
            d for d in buckets if not mapping_build.assignments.get(d)
        )
        if empty_dates:
            raise MappingDeliveryError(
                f"the mapping {name!r} recorded no capture at all for date(s) {empty_dates}: a "
                "date with no capture cannot be delivered; rebuild the mapping, or drop the "
                "date(s)"
            )
        # Refused on the record's own evidence, before verify_mapping_inputs reads any capture.
        registry_entries, registry_refusal = registry_entries_or_refusal(
            mapping_build, project
        )
        if registry_refusal:
            raise MappingDeliveryError(registry_refusal, status=409)
        paths = [entry["path"] for entry in registry_entries]
        n_plants = sum(entry["n_plants"] for entry in registry_entries)
        if n_plants == 0:
            raise MappingDeliveryError(
                f"the plant CSVs this mapping was built from ({paths}) parsed no plant with "
                "usable coordinates and a name, so no capture could be assigned; check the "
                "CSV's column headers against read_plant_csvs's and rebuild the mapping"
            )
        if all(a.distance_m is None for a in delivered_assignments):
            raise MappingDeliveryError(
                ungeoreferenced_capture_message(
                    f"mapping {name!r} (dataset {mapping_build.dataset_root!r})"
                )
            )
        radii = {
            date: f"{r.value} m ({r.source})"
            for date, r in mapping_build.match_radius.items()
            if date in buckets
        }
        n_unpositioned = sum(
            1 for a in delivered_assignments if a.distance_m is None
        )
        unpositioned_note = (
            f", and {n_unpositioned} captures carry no position"
            if n_unpositioned
            else ""
        )
        raise MappingDeliveryError(
            "every positioned capture on the delivered dates lies beyond its date's recorded "
            f"match radius ({radii}) of every plant in {paths}{unpositioned_note}; check the "
            "plant CSV names this block, or rebuild the mapping with a stated match_radius"
        )

    return mapping_build, verify_mapping_inputs(
        mapping_build, delivered_root, buckets, project=project
    )


def plant_mapping_disclosure(
    project: Path | str,
    name: str,
    buckets: Mapping[str, Bucket],
    plants: list[str],
) -> PlantMappingDisclosure:
    """Return the ``plant_mapping`` disclosure of a per-plant delivery of the population ``plants``.

    The delivery's plant ids came from mapping ``name``. The mapping is resolved and verified
    against the delivered ``buckets`` by recorded date (:func:`resolve_delivery_mapping`, refusing
    as it does), and a population plant the mapping assigns on no delivered date is refused
    (:class:`MappingDeliveryError`) naming each.
    """
    build, verified = resolve_delivery_mapping(project, name, buckets)
    assigned = {
        a.plant_id
        for d in buckets
        for a in build.assignments.get(d, [])
        if assignment_is_attributed(a)
    }
    unmapped = sorted(set(plants) - assigned)
    if unmapped:
        raise MappingDeliveryError(
            f"plant(s) {unmapped} are assigned to no plot by mapping "
            f"{name!r} on the delivered dates."
        )
    return build.delivery_disclosure(verified, list(buckets))
