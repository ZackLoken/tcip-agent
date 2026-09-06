"""Plant-ID mapping across image capture dates.

iPhone GPS is ~5 m accurate while the Valley_Farm plant grid is ~2.8 m between
adjacent plots, so nearest-neighbour GPS alone is ambiguous. We resolve the
ambiguity with a hybrid:

  1. Order images on each date by EXIF DateTime (walker's capture sequence).
  2. Detect "row breaks" as GPS jumps between consecutive images that stand out from the
     rest of that date's own walking gaps (derived per date, not a fixed distance, since a
     walker produces small, roughly uniform steps within a row and one or more much larger
     jumps at a row transition, whether the row itself is straight or curved).
  3. Within each row run, assign plants by matching the row end-points to the
     plant CSV and filling in plants sequentially along the row.

Fallback: when sequence anchoring fails (missing timestamps, unordered
capture), fall back to nearest-neighbour GPS with a configurable tolerance.
Each assignment records its match ``source`` and the GPS ``distance_m`` to the
matched plant, honest, interpretable signals. It deliberately does not emit a
0–1 "confidence": a linear ``1 − d/tol`` score read as a probability was
fabricated (uncalibrated against any hand-checked assignment) and has been
removed; use ``distance_m`` + ``source`` to judge a match (see the CLAUDE.md
measurement-integrity invariant).

A mapping is project state with a name, bound to the dataset it was built over and to its own
build receipt: ``build_mapping`` produces a :class:`MappingBuild` (provenance plus assignments),
``persist_mapping`` writes the record and then the receipt that binds it, and ``load_mapping``
refuses a record no receipt names. ``verify_mapping_inputs`` is the delivery-time check: for each
mapped date a delivery's own ``predictions_by_date`` actually names, it re-reads only the captures
the delivery reads (:func:`stems_delivery_reads`) plus the plant CSVs the record names, and
refuses (never raises) when what is on disk now no longer matches what the build was made from; a
date the delivery omits, or a capture of a delivered date the delivery does not read, is
disclosed rather than checked.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import statistics
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Iterable, Optional, Sequence

import tcip_store
from PIL import ExifTags, Image
from tcip_store import RECORD_JSON, Key, StoreDescriptor, register_store
from tcip_store.file_backend import RootedFileLocator

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_378_137.0

NN_TOLERANCE_METERS = 10.0

SEQUENCE_MATCH_FACTOR = 2
"""assign_plants' sequence-anchored gate: a run's nearest unclaimed plant is accepted out to this
many times nn_tolerance_m before falling through to plain nearest-neighbour."""

NEAREST_MATCH_FACTOR = 3
"""assign_plants' plain nearest-neighbour gate, the loosest a match is ever accepted at: this
many times nn_tolerance_m."""


@dataclass
class ImageStamp:
    """Per-capture metadata: EXIF for an ``image``, structural facts for a ``band_group`` or
    ``raster`` (``image_utils.capture_kind``). ``name`` is the file name (with extension)
    ``list_logical_images`` enumerated this stem under, ``readable`` is only meaningful for an
    ``image`` (``null`` for the other two kinds, which carry no EXIF to fail reading), and
    ``manifest_sha256``/``members`` only carry a value for a ``band_group``.
    """

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


@dataclass
class Assignment:
    """The mapping we produce for a single image."""

    image_path: str
    stem: str
    date_folder: str
    plot_name: Optional[str]
    accession_name: Optional[str]
    source: str        # "sequence" | "nearest_neighbour" | "unmapped"
    distance_m: Optional[float]  # GPS distance to the matched plant (m); None if unmapped


def assignment_is_attributed(assignment: "Assignment | dict") -> bool:
    """Whether ``assignment`` (an :class:`Assignment`, or the plain dict row ``MappingBuild.rows``
    produces) names a real plant: a non-empty ``plot_name``, the one rule a plant CSV's own blank
    name column and an unmapped capture (``plot_name=None``) both fail by, and nowhere else.
    ``stems_delivery_reads``, ``verify_mapping_inputs``, ``per_plant_series`` and
    :meth:`MappingBuild.unattributed` all decide attribution through this one predicate rather than
    each testing ``plot_name`` truthiness for itself.
    """
    plot_name = assignment.get("plot_name") if isinstance(assignment, dict) else assignment.plot_name
    return isinstance(plot_name, str) and plot_name != ""


def require_named_plants(plants: list[PlantRecord]) -> None:
    """Refuse a registry carrying a blank or duplicate ``plot_name``, the one check every
    raster-level attribution regime (nearest-neighbour distance, canopy-segment containment)
    runs over the same registry before attributing a single detection to it.

    A blank name fails :func:`assignment_is_attributed`'s own rule; a duplicate would merge two
    trees' detections into one row once ``aggregate_per_plant`` groups by ``plot_name``. Raises
    ``ValueError`` naming the offending accession or names, never silently dropping either plant
    from the registry it was asked to check.
    """
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
    """One build's provenance plus its per-date assignments: the whole persisted record, in
    memory, before :func:`persist_mapping` writes it and the receipt behind it.

    ``dataset_root``/``dataset_id`` are the door's own resolved facts (the dataset identity
    record's minted id, so a moved-and-re-registered dataset still delivers through this
    mapping); ``project_root``/``built_by``/``name`` are likewise the door's own facts, not
    re-derived here. ``capture_digests`` is ``capture_identity``'s per-capture counterpart
    (:func:`capture_digests`, one entry per stem, derived from the same row builder), so a
    whole-date identity mismatch can be narrowed to the capture that actually moved.
    """

    name: str
    project_root: str
    dataset_root: str
    dataset_id: str
    built_by: str
    built_at: str
    dates_requested: Optional[list[str]]
    dates: list[str]
    nn_tolerance_m: dict
    plant_registry: dict
    """The named :data:`PLANT_REGISTRY_STORE` record this build's plants were read from:
    ``{"name": ..., "digest": ...}``, resolved once by the calling door (``register_plant_registry``
    mints the record; this cell only names which one) and stored here rather than the CSV paths
    and hashes themselves, which the registry now holds. :func:`verify_mapping_inputs` and
    :func:`resolve_delivery_mapping`'s no-plant-count refusal both read the registry back through
    :func:`load_registry` using this name, never a copy of its own."""
    capture_identity: dict[str, str]
    capture_digests: dict[str, dict[str, str]]
    unreadable: dict[str, list[str]]
    assignments: dict[str, list[Assignment]] = field(default_factory=dict)
    supersedes: Optional[str] = None
    """The archived digest this record replaced, when a same-name rebuild was told to
    ``supersede`` a record a delivery event still cited (:func:`persist_mapping`); ``None`` for
    an ordinary build or an uncited rebuild, which replaces as it always has."""
    record_sha256: str = field(default="", metadata={"persisted": False})
    """The digest ``load_mapping`` verified this record's receipt against; blank on a build not
    yet read back through ``load_mapping`` (a fresh ``build_mapping`` result before persisting).
    Not one of :data:`_PERSISTED_FIELD_NAMES`: it names a fact about how this record was read,
    never a fact ``to_record`` writes or ``load_mapping`` reads back from the document itself."""

    plant_attribution: ClassVar[str] = "image"
    """The granularity at which this mapping's own assignments attribute objects to plants: one
    frame per plant, by the capture protocol the platform assumes for a walked, EXIF-geolocated
    scene. A class attribute, not an instance field, so it is excluded from the dataclass's own
    ``fields()`` the same way ``record_sha256``'s metadata excludes that field, and carries no
    per-build value to disagree with itself."""

    def to_record(self) -> dict:
        """The exact document ``persist_mapping`` writes and ``load_mapping`` reads back.

        Built over :data:`_PERSISTED_FIELD_NAMES`, the dataclass's own field list minus
        ``record_sha256``, so a field added to :class:`MappingBuild` needs its record shape
        decided once here, never a second key list kept in step by hand.
        """
        record = {field_name: getattr(self, field_name) for field_name in _PERSISTED_FIELD_NAMES
                  if field_name != "assignments"}
        record["assignments"] = {
            date: [a.__dict__ for a in assignments]
            for date, assignments in self.assignments.items()
        }
        return record

    def rows(self) -> dict[str, list[dict]]:
        """The assignments as plain per-date dict rows, the shape ``per_plant_phenology`` reads,
        each carrying this build's own ``plant_attribution`` so a caller composing image-
        granularity aggregation off these rows never hand-copies it from the build separately. A
        fresh dict per row, never ``a.__dict__`` itself, so this never leaks the extra key into
        what ``to_record`` persists."""
        return {date: [{**a.__dict__, "plant_attribution": self.plant_attribution} for a in assignments]
                for date, assignments in self.assignments.items()}

    def unattributed(self, dates: Optional[Iterable[str]] = None) -> int:
        """The number of assignments over ``dates`` (every date this mapping holds, when ``None``)
        for which :func:`assignment_is_attributed` is false: the one place this count is computed,
        called once per date for a per-date breakdown (:meth:`summary`) and once over a delivery's
        own delivered dates for its total (:meth:`delivery_disclosure`)."""
        scope = self.dates if dates is None else dates
        return sum(
            1
            for date in scope
            for a in self.assignments.get(date, [])
            if not assignment_is_attributed(a)
        )

    def summary(self) -> dict:
        """This build's own per-date and total counts: images, mapped, unattributed, and the mean
        GPS match distance (``None`` for a date with no recorded distance, never a fabricated
        zero). The one computation both the build and load routes answer their ``summary`` from,
        and the tool's own flat ``per_date``/totals fill from."""
        per_date: dict[str, dict] = {}
        total_images = 0
        total_mapped = 0
        total_unattributed = 0
        for date in self.dates:
            assignments = self.assignments.get(date, [])
            n_images = len(assignments)
            n_mapped = sum(1 for a in assignments if assignment_is_attributed(a))
            n_unattributed = n_images - n_mapped
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

    def delivery_disclosure(self, verified: dict, dates: Iterable[str]) -> dict:
        """The ``plant_mapping`` dict a phenology delivery carries: this build's own identity,
        ``verify_mapping_inputs``'s disclosure, and this delivery's own unattributed-capture count
        scoped to ``dates`` (a delivery's own delivered dates, never the mapping's full span), the
        one composition every phenology door (``deliver_phenology_milestones``, both web phenology routes)
        builds through rather than each assembling its own copy."""
        dates_delivered = sorted(dates)
        return {
            "name": self.name,
            "project_root": self.project_root,
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


# ── EXIF extraction ──────────────────────────────────────────────────────

_DT_ORIGINAL_TAG = 0x9003
# DateTimeOriginal: checked at the top level and in the Exif sub-IFD (0x8769) below, since a
# fresh PIL.Image.Exif writes it at the top level while a camera's own JPEG nests it under 0x8769.
_EXIF_IFD_TAG = 0x8769
_GPS_IFD_TAG = 0x8825


def _exif_dms_to_decimal(dms, ref: str) -> Optional[float]:
    if not dms:
        return None
    try:
        d, m, s = (float(x) for x in dms)
    except Exception:
        return None
    val = d + m / 60 + s / 3600
    if ref in ("S", "W"):
        val = -val
    return val


def read_image_stamp(path: Path, date_folder: str) -> ImageStamp:
    """One ``image`` capture's EXIF stamp.

    The ``try`` covers ``Image.open`` alone: a capture PIL cannot open (a HEIC with no decoder
    installed, a locked file) becomes a stamp with ``readable=False`` rather than being
    swallowed into an indistinguishable all-``None`` stamp; an image that opens and carries no
    EXIF (or an EXIF read through ``getexif()``, which the pinned Pillow defines for every
    format, unlike the JPEG-only legacy ``_getexif()``) stays ``readable=True`` with ``None``
    fields.
    """
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
        dt_raw = exif.get_ifd(_EXIF_IFD_TAG).get(_DT_ORIGINAL_TAG)
        if dt_raw is None:
            dt_raw = exif.get(_DT_ORIGINAL_TAG)
        if dt_raw is not None:
            try:
                stamp.timestamp = datetime.strptime(str(dt_raw), "%Y:%m:%d %H:%M:%S")
            except Exception:
                pass

        gps_raw = exif.get_ifd(_GPS_IFD_TAG)
        if gps_raw:
            gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
            stamp.lat = _exif_dms_to_decimal(gps.get("GPSLatitude"), str(gps.get("GPSLatitudeRef", "N")))
            stamp.lon = _exif_dms_to_decimal(gps.get("GPSLongitude"), str(gps.get("GPSLongitudeRef", "E")))
            hpe = gps.get("GPSHPositioningError")
            if hpe is not None:
                try:
                    stamp.h_pos_err = float(hpe)
                except Exception:
                    pass
    return stamp


def _read_date_stamps(logical: dict[str, "Path | BandGroupRef"], date_folder: str) -> list[ImageStamp]:
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
    an ``image``. The one row shape :func:`capture_identity` (joined across a date) and
    :func:`capture_digests` (kept per capture) both build from, so the two spellings cannot drift
    apart.
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
    assignment was made from, and no reader may cite it as provenance for the pixels a phenotype
    was counted over (the prediction bucket's own stamps carry that). ``build_mapping`` calls
    this once per date; the delivery's ``verify_mapping_inputs`` recomputes it for a date only
    when the delivery read every capture the date has, to detect a changed input set at no added
    cost precisely when it can (see that function's own docstring for the case it can't).
    """
    rows = [_capture_row(s) for s in sorted(stamps, key=lambda s: s.name)]
    return _row_digest(rows)


def capture_digests(stamps: list[ImageStamp]) -> dict[str, str]:
    """One digest per capture, keyed by stem, over that capture's own :func:`_capture_row` alone.

    ``capture_identity``'s counterpart: the same row shape, kept apart per capture instead of
    joined across the whole date, so a whole-date identity mismatch can be narrowed to the exact
    capture that moved (a band group's manifest rewritten in place, most usefully, since nothing
    else names that case) rather than only the date.
    """
    return {s.stem: _row_digest(_capture_row(s)) for s in stamps}


# ── Plant CSV parsing ───────────────────────────────────────────────────


def read_plant_csv_bytes(data: bytes) -> list[PlantRecord]:
    """Parse one plant-locations CSV's own bytes into :class:`PlantRecord` rows.

    The row parse :func:`read_plant_csvs` applies per path, extracted so a caller already holding
    a file's verified bytes (:func:`verify_registry_csv_bytes`'s own return) parses those bytes
    directly instead of re-opening the file a second time: the read-then-parse race a file
    replaced in between the two reads could otherwise let slip through.
    """
    import io

    records: list[PlantRecord] = []
    text = data.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    for row in reader:
        try:
            lat = float(row["WGS84_centroid_y"])
            lon = float(row["WGS84_centroid_x"])
        except Exception:
            continue
        records.append(
            PlantRecord(
                plot_name=row.get("plot_name", ""),
                accession_name=row.get("accession_name", ""),
                plot_number=_maybe_float(row.get("plot_number")),
                row_number=_maybe_float(row.get("row_number")),
                col_number=_maybe_float(row.get("col_number")),
                lat=lat,
                lon=lon,
            )
        )
    return records


def read_plant_csvs(paths: Iterable[Path]) -> list[PlantRecord]:
    records: list[PlantRecord] = []
    for path in paths:
        p = Path(path)
        if not p.is_file():
            continue
        records.extend(read_plant_csv_bytes(p.read_bytes()))
    return records


def _maybe_float(x: Optional[str]) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def registry_content_digest(plants: list[PlantRecord]) -> str:
    """sha256 over ``plants``' own fields, order-independent, so two registrations of the same
    plants under two paths (or a different row order in one file) are told apart from a real
    edit: :func:`register_plant_registry_record` compares this digest against a name already
    taken, and returns the existing record when it matches rather than refusing a harmless
    re-registration."""
    from dataclasses import asdict

    rows = sorted(
        (asdict(p) for p in plants), key=lambda r: json.dumps(r, sort_keys=True))
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()


class NoGeoreferencedPlantsRefusal(Exception):
    """A registration names a CSV that parsed no georeferenced, named plant. ``paths`` lists
    every file that failed the parse, so the caller's refusal names them all at once."""

    def __init__(self, message: str, *, paths: list[str]) -> None:
        super().__init__(message)
        self.paths = paths


class PlantRegistryNameConflict(Exception):
    """A registration's name is already taken by a different set of plants."""


PLANT_REGISTRY_STORE = "plant_registries"
_REGISTRY_DOC = RootedFileLocator(prefix=("plant_registries",), suffix=".json")
register_store(
    StoreDescriptor(
        name=PLANT_REGISTRY_STORE,
        kind="record",
        key_fields=("name",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="cas",
        enumerable=True,
        locator=_REGISTRY_DOC,
    )
)


def plant_registry_key(project_root: Path | str, name: str) -> Key:
    """One project's named plant registry: identity state for a plant-locations CSV set, the
    same ``.tcip/state``-scoped shape :func:`plant_mapping_key` uses, so a registry travels with
    the project that registered it rather than the dataset it happens to describe."""
    root = Path(project_root).absolute() / ".tcip" / "state"
    return Key(PLANT_REGISTRY_STORE, str(root), (name,))


def load_registry(project_root: Path | str, name: str) -> Optional[dict]:
    """The named registry record, or ``None`` when nothing is stored under that name (including
    a name that was never registered, or one deleted since a mapping recorded it)."""
    return tcip_store.read(plant_registry_key(project_root, name), default=None)


def registry_csv_entries(record: Optional[dict]) -> list[dict]:
    """The ``{path, sha256, n_plants}`` entries a loaded registry record carries, or an empty
    list when ``record`` is ``None``.

    A bare accessor over a record already in hand, never the decision of what a missing or
    mismatched registry means for a mapping that names it: that decision is
    :func:`registry_entries_or_refusal`'s, called by every mapping-facing reader
    (:func:`verify_mapping_inputs`, :func:`resolve_delivery_mapping`) so a vanished or moved
    registry refuses by name rather than silently verifying nothing.
    """
    return list(record["csvs"]) if record else []


def registry_entries_or_refusal(
    build: "MappingBuild", project_root: Path | str,
) -> tuple[list[dict], Optional[str]]:
    """The CSV entries ``build.plant_registry`` names, or the refusal naming the registry, the
    mapping and ``project_root`` when they cannot be trusted.

    Refuses when the named registry no longer loads (deleted since the mapping was built) and
    when it loads but its own ``digest`` no longer matches ``build.plant_registry["digest"]`` (the
    registration a mapping was built against has moved under its own name), rather than letting
    either case verify silently against nothing, the same document
    :func:`verify_mapping_inputs` and :func:`resolve_delivery_mapping` would each otherwise read
    on their own and could drift apart on.
    """
    registry_name = (build.plant_registry or {}).get("name")
    if not registry_name:
        return [], None
    record = load_registry(project_root, registry_name)
    if record is None:
        return [], (
            f"plant registry {registry_name!r} under {project_root} named by mapping "
            f"{build.name!r} no longer loads (deleted since this mapping was built); "
            "re-register it with register_plant_registry, or rebuild the mapping against a "
            "different registry")
    stored_digest = (build.plant_registry or {}).get("digest")
    if stored_digest is not None and record.get("digest") != stored_digest:
        return [], (
            f"plant registry {registry_name!r} under {project_root} named by mapping "
            f"{build.name!r} has moved (built against digest {stored_digest!r}, now "
            f"{record.get('digest')!r}); rebuild the mapping against the current registry")
    return registry_csv_entries(record), None


def verify_registry_csv_bytes(
    registry_entries: list[dict],
) -> tuple[list[str], Optional[str], dict[str, bytes]]:
    """Check each of ``registry_entries``'s (:func:`registry_csv_entries`'s own ``{path, sha256,
    n_plants}`` shape) recorded bytes against what is on disk now, reading each present file's
    bytes exactly once.

    Returns every missing path (the loop runs to completion rather than stopping at the first
    rewritten file), the fact that the first rewritten file was rewritten, naming it, or ``None``
    when every present path's bytes still match what was registered, and the verified bytes this
    call read for every present path, keyed by the entry's own ``path``. A caller parses a verified
    file from that third return rather than re-opening it a second time: reading bytes here and
    parsing them again from the path afterward is a read-then-parse race a file replaced in
    between could slip through, closed by parsing the one snapshot this call already took
    (:func:`read_plant_csv_bytes`). This function reports the bytes facts only, never a remedy: a
    caller decides for itself what a missing path means and composes its own wording for a
    rewritten one (:func:`verify_mapping_inputs` and
    :func:`~tcip_mcp.tools.orthomosaic_tools.deliver_orthomosaic_plant_counts` each refuse under a
    different remedy, since one can rebuild a mapping against a new registry and the other cannot).
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


def parse_plant_registry_csvs(csv_paths: list[Path]) -> tuple[list[dict], str, int]:
    """Parse ``csv_paths`` into the registry's own ``{path, sha256, n_plants}`` entries, the
    content digest over every parsed row and the total plant count: the read-only half of
    :func:`register_plant_registry_record` that commits nothing, so a preview can compute
    exactly what a real registration would write without writing it.

    Raises :class:`NoGeoreferencedPlantsRefusal`, naming every file that parsed no georeferenced,
    named plant, the same check :func:`register_plant_registry_record` runs before it writes.
    """
    failed: list[str] = []
    csvs_meta: list[dict] = []
    all_plants: list[PlantRecord] = []
    for p in csv_paths:
        records = read_plant_csvs([p])
        if not records:
            failed.append(str(p))
            continue
        all_plants.extend(records)
        csvs_meta.append({
            "path": str(p),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "n_plants": len(records),
        })
    if failed:
        raise NoGeoreferencedPlantsRefusal(
            f"{failed} parsed no georeferenced, named plant (need columns plot_name, "
            "WGS84_centroid_x, WGS84_centroid_y with usable values); register only files that "
            "carry at least one",
            paths=failed,
        )
    return csvs_meta, registry_content_digest(all_plants), len(all_plants)


def register_plant_registry_record(
    project_root: Path | str,
    name: str,
    csv_paths: list[Path],
    *,
    crop: str,
    site: str,
    registered_by: str,
) -> dict:
    """Register ``csv_paths`` under ``name`` in this project's plant registry, and return the
    stored record.

    Parses every path through :func:`parse_plant_registry_csvs`, so a refusal names exactly which
    file parsed no georeferenced, named plant (:class:`NoGeoreferencedPlantsRefusal`); the record
    holds the same ``{path, sha256, n_plants}`` entries the mapping record held before this door
    existed, plus ``crop``, ``site``, ``registered_by``, ``registered_at`` and the parsed content's
    digest. The read-then-write is one transaction (:func:`tcip_store.transaction`, this store's
    own ``concurrency="cas"``): a second registration under a taken name returns the existing
    record unchanged when the digest matches (the same plants again, read from a different path or
    in a different row order), and raises :class:`PlantRegistryNameConflict` otherwise, naming the
    two digests, since overwriting a taken name would silently move what every mapping already
    citing it means.
    """
    csvs_meta, digest, n_plants = parse_plant_registry_csvs(csv_paths)
    key = plant_registry_key(project_root, name)
    with tcip_store.transaction(key) as txn:
        existing = txn.read(key, default=None)
        if existing is not None:
            if existing.get("digest") == digest:
                return existing
            raise PlantRegistryNameConflict(
                f"plant registry {name!r} under {project_root} already names different plants "
                f"(digest {existing.get('digest')!r}, this registration would write "
                f"{digest!r}); register under a new name")
        record = {
            "name": name,
            "crop": crop,
            "site": site,
            "csvs": csvs_meta,
            "n_plants": n_plants,
            "digest": digest,
            "registered_by": registered_by,
            "registered_at": datetime.now(timezone.utc).isoformat(),
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


def stems_delivery_reads(rows: "Iterable[Assignment | dict]", pred_dir: Path | str) -> set[str]:
    """Stems of ``rows`` (one date's assignment rows, :class:`Assignment` objects or the plain
    dict rows :meth:`MappingBuild.rows` produces) :func:`assignment_is_attributed` calls
    attributed, whose stem also carries a prediction document under ``pred_dir``: the one
    predicate for "which prediction documents this delivery reads," never whether the underlying
    image still exists on disk.

    Both :func:`verify_mapping_inputs` (which of those stems it may re-check a fresh stamp for)
    and :func:`~tcip_mcp.pipelines.postprocessing.phenology.per_plant_series` (which of those
    stems it aggregates rather than counts missing) call this rather than each inlining the same
    test, so the two can never disagree about which predictions a delivery reads. Whether a read
    prediction's own capture can still be verified is a further partition ``verify_mapping_inputs``
    makes on its own, one this predicate does not speak to.
    """
    from tcip_mcp.prediction_buckets import bucket_stems

    def _attr(row: object, name: str) -> object:
        return getattr(row, name, None) if not isinstance(row, dict) else row.get(name)

    pred_stems = bucket_stems(pred_dir)
    out: set[str] = set()
    for row in rows:
        if not assignment_is_attributed(row):
            continue
        stem = _attr(row, "stem")
        if isinstance(stem, str) and stem in pred_stems:
            out.add(stem)
    return out


def _nearest_plant(
    lat: float,
    lon: float,
    plants: list[PlantRecord],
) -> tuple[Optional[PlantRecord], Optional[float]]:
    best: Optional[PlantRecord] = None
    best_d: Optional[float] = None
    for p in plants:
        d = haversine_m(lat, lon, p.lat, p.lon)
        if best_d is None or d < best_d:
            best = p
            best_d = d
    return best, best_d


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
    samples a single date's image count produces) rather than trusting it as the single largest of
    an otherwise smooth continuum. Returns ``None`` when there aren't enough gaps to judge, or the
    gaps are too uniform to support a split: the caller should then treat the whole sequence as one
    run rather than fabricating a break the data doesn't support.
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
        if prev.lat is not None and cur.lat is not None:
            d = haversine_m(prev.lat, prev.lon or 0.0, cur.lat, cur.lon or 0.0)
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
    nn_tolerance_m: float = NN_TOLERANCE_METERS,
) -> list[Assignment]:
    """Assign each image to a plant by sequence-anchored NN matching.

    A ``band_group``/``raster`` stamp carries no GPS fix (``lat``/``lon`` are always ``None``),
    so it always falls through to the ``unmapped`` branch below, the same way an ``image`` with
    no usable EXIF position does.
    """
    out: list[Assignment] = []
    if not stamps or not plants:
        for s in stamps:
            out.append(
                Assignment(
                    image_path=s.path,
                    stem=s.stem,
                    date_folder=s.date_folder,
                    plot_name=None,
                    accession_name=None,
                    source="unmapped",
                    distance_m=None,
                )
            )
        return out

    ordered = _order_by_time(stamps)
    runs = _segment_runs(ordered)

    # Each image picks its nearest unclaimed plant; degrades to NN when sequence signal is weak.
    claimed: set[int] = set()

    for run in runs:
        for s in run:
            if s.lat is None or s.lon is None:
                out.append(
                    Assignment(
                        image_path=s.path,
                        stem=s.stem,
                        date_folder=s.date_folder,
                        plot_name=None,
                        accession_name=None,
                        source="unmapped",
                        distance_m=None,
                    )
                )
                continue

            best_idx = -1
            best_d: Optional[float] = None
            for i, p in enumerate(plants):
                if i in claimed:
                    continue
                d = haversine_m(s.lat, s.lon, p.lat, p.lon)
                if best_d is None or d < best_d:
                    best_idx = i
                    best_d = d

            if best_idx < 0 or best_d is None or best_d > nn_tolerance_m * SEQUENCE_MATCH_FACTOR:
                # Fall through to plain NN, even if claimed: duplicates can happen
                plant, nn_d = _nearest_plant(s.lat, s.lon, plants)
                if plant is None or nn_d is None or nn_d > nn_tolerance_m * NEAREST_MATCH_FACTOR:
                    out.append(
                        Assignment(
                            image_path=s.path,
                            stem=s.stem,
                            date_folder=s.date_folder,
                            plot_name=None,
                            accession_name=None,
                            source="unmapped",
                            distance_m=nn_d,
                        )
                    )
                else:
                    out.append(
                        Assignment(
                            image_path=s.path,
                            stem=s.stem,
                            date_folder=s.date_folder,
                            plot_name=plant.plot_name,
                            accession_name=plant.accession_name,
                            source="nearest_neighbour",
                            distance_m=nn_d,
                        )
                    )
                continue

            p = plants[best_idx]
            claimed.add(best_idx)
            out.append(
                Assignment(
                    image_path=s.path,
                    stem=s.stem,
                    date_folder=s.date_folder,
                    plot_name=p.plot_name,
                    accession_name=p.accession_name,
                    source="sequence",
                    distance_m=best_d,
                )
            )

    # Return in the original stamp order so callers can merge with their image lists
    by_path = {a.image_path: a for a in out}
    return [by_path[s.path] for s in stamps]


# ── Whole-dataset driver ────────────────────────────────────────────────


def grid_pitch_m(plants: list[PlantRecord]) -> float:
    """Median nearest-neighbor spacing of the plant centroids = the planting grid pitch (m).

    Derived from the layout in hand (not pinned): used to cap the GPS match tolerance so a
    detection can't be attributed to a plant more than half a grid cell away; beyond that, the
    nearest plant is as likely to be the wrong (adjacent) plot as the right one.
    """
    pts = [(p.lat, p.lon) for p in plants if p.lat is not None and p.lon is not None]
    if len(pts) < 2:
        return 0.0
    nn = []
    for i, (la, lo) in enumerate(pts):
        nn.append(min(haversine_m(la, lo, la2, lo2) for j, (la2, lo2) in enumerate(pts) if j != i))
    nn.sort()
    return nn[len(nn) // 2]


def match_gates(nn_tolerance_m: float) -> dict:
    """The distances a match is actually accepted out to, given ``nn_tolerance_m``: the stated
    tolerance itself, the sequence-anchored gate's own ceiling, and ``max_match_distance_m``,
    ``assign_plants``' loosest gate (the plain nearest-neighbour fallback). Derived once here so
    a caller states the same ceiling ``assign_plants`` enforces rather than restating its factors.
    """
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
    project_root: Path | str,
    built_by: str,
    plant_registry: dict,
    dates: Optional[list[str]] = None,
    nn_tolerance_m: Optional[float] = None,
) -> MappingBuild:
    """Build one project's named plant mapping: per-date assignments plus the provenance that
    binds the record to the inputs it was built from.

    ``name``/``dataset_root``/``dataset_id``/``project_root``/``built_by`` are the caller's own
    resolved facts (the door already checked the dataset identity and the project record before
    calling here); this function does not re-derive them. ``plant_csv_paths`` are the files this
    build actually reads the plants from (resolved by the caller from ``plant_registry``'s own
    ``name``); ``plant_registry`` is the ``{"name": ..., "digest": ...}`` reference stored on the
    record in their place, so a later read finds the files through the registry rather than a
    copy of the paths and hashes here.

    ``nn_tolerance_m`` is derived from the plot's ``grid_pitch_m`` when the caller does not pin it
    (not a pinned 10 m): pitch/6, so assign_plants' loosest ``NEAREST_MATCH_FACTOR``-times gate
    keeps the effective match radius within half a grid cell. An explicit value is honored but
    still capped at that pitch-derived ceiling; ``nn_tolerance_m`` on the record carries which of
    the four branches (``grid_pitch``, ``fallback``, ``stated_capped``, ``stated``) produced the
    value.

    A date's captures are enumerated through ``image_utils.list_logical_images``, so a band
    raster or a band group ingested under a mapped date is a capture the identity sees. That
    enumeration raises :class:`~tcip_mcp.pipelines.image_utils.AmbiguousImageStem` when the
    bucket holds more than one logical identity under one case-folded stem, standalone-versus-
    standalone or standalone-versus-band-group alike; this function lets it propagate, and the
    calling door catches it and refuses in its own error shape.

    Raises :class:`UngeoreferencedCaptureRefusal`: naming ``images_root`` when the requested
    dates carry no capture at all, and with :func:`ungeoreferenced_capture_message` (naming any
    capture PIL could not open before the position clause) when every capture that was read
    carries no position this door reads.
    """
    from tcip_mcp.pipelines.image_utils import list_logical_images

    plant_csv_paths = [Path(p) for p in plant_csv_paths]
    plants = read_plant_csvs(plant_csv_paths)

    # The ceiling: assign_plants' loosest gate is NEAREST_MATCH_FACTOR x nn_tolerance, so pitch/6
    # -> effective radius <= pitch/2; derive the tolerance from the layout, cap only an override.
    pitch = grid_pitch_m(plants)
    _cap = pitch / 6
    tolerance_source: str
    if nn_tolerance_m is None:
        if pitch > 0:
            nn_tolerance_m, tolerance_source = _cap, "grid_pitch"
        else:
            nn_tolerance_m, tolerance_source = NN_TOLERANCE_METERS, "fallback"
        logger.info("nn_tolerance_m derived %.2f (%s)", nn_tolerance_m, tolerance_source)
    elif pitch > 0 and nn_tolerance_m > _cap:
        logger.info("capping nn_tolerance_m %.1f -> %.2f (grid pitch %.1f: effective radius <= pitch/2)",
                    nn_tolerance_m, _cap, pitch)
        nn_tolerance_m, tolerance_source = _cap, "stated_capped"
    else:
        tolerance_source = "stated"

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
            n_positioned += sum(1 for s in stamps if s.lat is not None and s.lon is not None)
            capture_ids[date] = capture_identity(stamps)
            capture_digests_by_date[date] = capture_digests(stamps)
            unreadable[date] = sorted(
                s.name for s in stamps if s.kind == "image" and s.readable is False)
            assignments[date] = assign_plants(stamps, plants, nn_tolerance_m=nn_tolerance_m)

    if n_stamps == 0:
        raise UngeoreferencedCaptureRefusal(
            f"no capture under {images_root} on the requested dates")
    if n_positioned == 0:
        all_unreadable = sorted(
            {name for date in dates_walked for name in unreadable.get(date, [])})
        raise UngeoreferencedCaptureRefusal(
            ungeoreferenced_capture_message(str(images_root), all_unreadable))

    return MappingBuild(
        name=name,
        project_root=str(project_root),
        dataset_root=str(dataset_root),
        dataset_id=dataset_id,
        built_by=built_by,
        built_at=datetime.now(timezone.utc).isoformat(),
        dates_requested=list(dates) if dates is not None else None,
        dates=sorted(dates_walked),
        nn_tolerance_m={"value": nn_tolerance_m, "source": tolerance_source},
        plant_registry=plant_registry,
        capture_identity=capture_ids,
        capture_digests=capture_digests_by_date,
        unreadable=unreadable,
        assignments=assignments,
    )


PLANT_MAPPING_STORE = "plant_mapping"
_MAPPING_DOC = RootedFileLocator(prefix=("plant_mappings",), suffix=".json")
register_store(
    StoreDescriptor(
        name=PLANT_MAPPING_STORE,
        kind="record",
        key_fields=("name",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        enumerable=True,
        locator=_MAPPING_DOC,
    )
)


def plant_mapping_key(project_root: Path | str, name: str) -> Key:
    """One project's named plant-mapping build, addressed by the project that owns it.

    A mapping is project state: a dataset can be read by more than one project, and each
    project's mapping is its own. The key root is ``<project_root>/.tcip/state``; the document
    lives at ``plant_mappings/<name>.json`` under it (the same ``STATE``-scoped shape
    ``delivery_events`` uses, and the shape the layout claim in ``tcip_store.layout_claims``
    declares for this store).

    ``last_writer_wins``: a mapping is assigned whole in memory and written in one call, and a
    later build under the same name is a fresh assignment replacing that one rather than a
    merge into it. No writer reads the record first.
    """
    root = Path(project_root).absolute() / ".tcip" / "state"
    return Key(PLANT_MAPPING_STORE, str(root), (name,))


def plant_mapping_names(project_root: Path | str) -> list[str]:
    """Every mapping name persisted under this project, sorted.

    Enumerated through the store's own key listing (the ``STATE``-scoped claim's
    ``enumerable=True``), filtered by ``NAME_SEGMENT``: the listing is by locator and would
    otherwise offer a stray, illegally-named file's stem as a name the door refuses.
    """
    from tcip_store.layout_claims import NAME_SEGMENT

    root = str(Path(project_root).absolute() / ".tcip" / "state")
    names = (key.parts[-1] for key in tcip_store.keys(PLANT_MAPPING_STORE, root))
    return sorted(name for name in names if NAME_SEGMENT.fullmatch(name))


def record_digest(record: dict) -> str:
    """The one digest a mapping record earns: sha256 over ``RECORD_JSON.encode(record)``.

    Called from :func:`persist_mapping` (what the receipt names) and :func:`load_mapping` (what
    the receipt is checked against), so the write side and the read side can never spell this
    digest differently.
    """
    return hashlib.sha256(RECORD_JSON.encode(record)).hexdigest()


class MappingRebuildRefusal(Exception):
    """A same-name rebuild would replace a mapping record a delivery event under this project
    still cites. ``event_ids`` names every citing event; ``status`` is the web door's HTTP status
    for it."""

    def __init__(self, message: str, *, event_ids: list[str], status: int = 409) -> None:
        super().__init__(message)
        self.event_ids = event_ids
        self.status = status


def _citing_delivery_event_ids(project_root: Path | str, name: str, digest: str) -> list[str]:
    """Every delivery event under this project whose own ``plant_mapping`` disclosure names
    (``name``, ``digest``): the events a same-name rebuild would strand by silently replacing the
    record they cite, sorted for a stable refusal message.

    Reads through :func:`~tcip_mcp.pipelines.resolution.read_delivery_events`, so an event under
    this project that does not validate refuses the whole call rather than being silently skipped
    from a citing list a rebuild refusal must be complete to trust.
    """
    from tcip_mcp.pipelines.delivery_events_schema import is_mapping_disclosure
    from tcip_mcp.pipelines.resolution import read_delivery_events

    ids = [
        record["event_id"] for record in read_delivery_events(project_root)
        if is_mapping_disclosure(record.get("plant_mapping"))
        and record["plant_mapping"]["name"] == name
        and record["plant_mapping"]["record_sha256"] == digest
    ]
    return sorted(ids)


def persist_mapping(
    build: MappingBuild, project_root: Path | str, name: str, *, supersede: bool = False,
) -> None:
    """Write the mapping record, then the receipt that binds it to this build.

    The record is committed before the receipt (a log append cannot join a record transaction):
    a receipt that cannot be written fails loudly (``AuditEntryNotWritten`` propagates, never
    swallowed) and leaves a record no receipt names, which :func:`load_mapping` refuses to read
    until a rebuild replaces it.

    ``project_root`` names which log the receipt lands in, and this function's two production
    callers pass two different roots for it: the ``build_plant_mapping`` MCP tool passes the
    process's own pinned platform root, so the receipt lands in the platform log's own file (a
    project's, once that root is an adopted project); the web build route passes its own guarded
    project root instead, which can differ from the process's pin when the browser has a
    different project open.

    A rebuild under ``name`` whose current record is still cited by a delivery event under this
    project raises :class:`MappingRebuildRefusal`, naming the citing events, unless
    ``supersede=True``. In that case the current record is archived first, under
    ``plant_mapping_key(project_root, f"{name}@{digest[:12]}")`` (a name :func:`plant_mapping_names`
    never lists, since it carries ``@`` and fails ``NAME_SEGMENT``), with a fresh
    ``plant_mapping_built`` receipt appended under that archived name so :func:`load_mapping` can
    still read it back; ``build.supersedes`` is set to the archived digest before this writes the
    new record, and the new record's own receipt names both digests. An uncited rebuild replaces
    as it always has, and records nothing extra.

    A rebuild under ``name`` whose *current* record is stored in a shape this reader no longer
    recognizes also raises :class:`MappingRebuildRefusal`: an unparseable existing record cannot
    be checked for citations, so it can never be safely archived or silently replaced either, and
    no operator door repairs it in place.
    """
    from tcip_mcp.audit import record_event_or_raise

    existing_raw = tcip_store.read(plant_mapping_key(project_root, name), default=None)
    archived_digest: Optional[str] = None
    if existing_raw is not None:
        try:
            existing_record = _validated_record(existing_raw, project_root, name)
        except ValueError as exc:
            raise MappingRebuildRefusal(
                f"plant mapping {name!r} under {project_root} is stored in a shape this reader "
                f"no longer recognizes ({exc}); a rebuild cannot tell whether a delivery event "
                "still cites it, and no operator door repairs an existing record in place, so "
                "this record must be corrected to the current shape before rebuilding",
                event_ids=[],
            ) from exc
        existing_digest = record_digest(existing_record)
        citing = _citing_delivery_event_ids(project_root, name, existing_digest)
        if citing and not supersede:
            raise MappingRebuildRefusal(
                f"plant mapping {name!r} under {project_root} is cited by delivery event(s) "
                f"{citing}: rebuilding under this name would strand them. Pass supersede=True to "
                "archive the current record and rebuild.",
                event_ids=citing,
            )
        if citing:
            archived_name = f"{name}@{existing_digest[:12]}"
            tcip_store.replace(plant_mapping_key(project_root, archived_name), existing_record)
            record_event_or_raise(
                "plant_mapping_built",
                {
                    "name": archived_name,
                    "project_root": str(project_root),
                    "dataset_root": existing_record.get("dataset_root"),
                    "built_at": existing_record.get("built_at"),
                    "record_sha256": existing_digest,
                },
                scope=project_root,
            )
            archived_digest = existing_digest
            build.supersedes = archived_digest

    record = build.to_record()
    tcip_store.replace(plant_mapping_key(project_root, name), record)
    record_sha256 = record_digest(record)
    record_event_or_raise(
        "plant_mapping_built",
        {
            "name": name,
            "project_root": str(project_root),
            "dataset_root": build.dataset_root,
            "built_at": build.built_at,
            "record_sha256": record_sha256,
            "supersedes": archived_digest,
        },
        scope=project_root,
    )


def load_mapping_rows(project_root: Path | str, name: str) -> dict[str, list[dict]]:
    """The persisted mapping as plain per-date rows, for a consumer that works in dicts.

    Reads through :func:`load_mapping`, so a caller handing the rows to the phenology pipeline
    gets the fields that reader fills in and the types it coerces, rather than whatever a
    particular writer happened to leave out. ``{}`` when no mapping is stored under ``name``.
    """
    build = load_mapping(project_root, name)
    return build.rows() if build is not None else {}


_PERSISTED_FIELD_NAMES: tuple[str, ...] = tuple(
    f.name for f in fields(MappingBuild) if f.metadata.get("persisted", True)
)
"""Every :class:`MappingBuild` field ``to_record``/``load_mapping`` carry to and from the
document, derived from the dataclass's own fields (minus ``record_sha256``) so a field added
there flows into the record shape and the constructor call without a second list to keep in
step. Two attributes are excluded from this tuple, each by its own mechanism: ``record_sha256``
by its own field metadata (``persisted: False``), and ``plant_attribution`` by never being a
dataclass field at all (a ``ClassVar``, which ``dataclasses.fields()`` never sees).
``_REQUIRED_TOP_KEYS`` still states each field's own type by hand (a runtime isinstance
check needs a real type, not the string form ``from __future__ import annotations`` leaves an
annotation as); the assertion below ties its key set back to this tuple, so the two cannot drift
apart silently."""

_REQUIRED_TOP_KEYS: dict[str, type | tuple[type, ...]] = {
    "name": str,
    "project_root": str,
    "dataset_root": str,
    "dataset_id": str,
    "built_by": str,
    "built_at": str,
    "dates_requested": (list, type(None)),
    "dates": list,
    "nn_tolerance_m": dict,
    "plant_registry": dict,
    "capture_identity": dict,
    "capture_digests": dict,
    "unreadable": dict,
    "assignments": dict,
    "supersedes": (str, type(None)),
}
assert set(_REQUIRED_TOP_KEYS) == set(_PERSISTED_FIELD_NAMES), (
    "MappingBuild's own fields and _REQUIRED_TOP_KEYS's key set have drifted apart")
_ASSIGNMENT_ROW_KEYS = (
    "image_path", "stem", "date_folder", "plot_name", "accession_name", "source", "distance_m")
_VALID_SOURCES = {"sequence", "nearest_neighbour", "unmapped"}


def _validated_record(raw: object, project_root: Path | str, name: str) -> dict:
    """``raw`` as a plant-mapping record, or the ``ValueError`` naming the project, the name and
    the field this reader does not recognize. A rebuild through ``build_plant_mapping`` is not
    named as the remedy here: :func:`persist_mapping` itself refuses a rebuild over a record this
    function cannot validate, and no operator door corrects an existing record in place."""
    remedy = "no operator door corrects an existing record in place"
    if not isinstance(raw, dict):
        raise ValueError(
            f"plant mapping {name!r} under {project_root} is not a record document "
            f"(found {type(raw).__name__}); {remedy}")
    for key, kind in _REQUIRED_TOP_KEYS.items():
        if key not in raw:
            raise ValueError(
                f"plant mapping {name!r} under {project_root} is missing {key!r}; {remedy}")
        if not isinstance(raw[key], kind):
            raise ValueError(
                f"plant mapping {name!r} under {project_root} carries {key!r} of the wrong type "
                f"(found {type(raw[key]).__name__}); {remedy}")
    for date in raw["capture_identity"]:
        if not isinstance(raw["capture_digests"].get(date), dict):
            raise ValueError(
                f"plant mapping {name!r} under {project_root}: capture_identity names date "
                f"{date!r} but capture_digests carries no digest map for it; {remedy}")
    for date, rows in raw["assignments"].items():
        if not isinstance(rows, list):
            raise ValueError(
                f"plant mapping {name!r} under {project_root}: assignments[{date!r}] is not a "
                f"list; {remedy}")
        for row in rows:
            if not isinstance(row, dict) or any(k not in row for k in _ASSIGNMENT_ROW_KEYS):
                raise ValueError(
                    f"plant mapping {name!r} under {project_root}: a row under "
                    f"assignments[{date!r}] is missing one of {_ASSIGNMENT_ROW_KEYS}; {remedy}")
            if row["source"] not in _VALID_SOURCES:
                raise ValueError(
                    f"plant mapping {name!r} under {project_root}: a row under "
                    f"assignments[{date!r}] carries source {row['source']!r}, not one of "
                    f"{sorted(_VALID_SOURCES)}; {remedy}")
            dm = row["distance_m"]
            if dm is not None and not (isinstance(dm, (int, float)) and math.isfinite(dm)):
                raise ValueError(
                    f"plant mapping {name!r} under {project_root}: a row under "
                    f"assignments[{date!r}] carries a non-finite distance_m ({dm!r}); {remedy}")
    return raw


# Process-local memo for load_mapping's receipt check: a project's log cursor plus every
# plant_mapping_built (name, record_sha256) seen; a miss re-reads once before refusing.
_receipt_cursor: dict[str, str] = {}
_receipt_seen: dict[str, dict[str, set[str]]] = {}


def _scan_receipts(project_root: Path | str, root_key: str, *, after: Optional[str]) -> None:
    from tcip_mcp.audit import audit_log_key

    key = audit_log_key(project_root)
    page = tcip_store.read_log(key, after=after)
    if page.corrupt:
        raise ValueError(
            f"the audit log at {key} carries {len(page.corrupt)} undecodable "
            f"entr{'y' if len(page.corrupt) == 1 else 'ies'}; repair the log before a "
            "plant-mapping receipt can be trusted")
    if page.version_refused:
        raise ValueError(
            f"the audit log at {key} carries {len(page.version_refused)} entr"
            f"{'y' if len(page.version_refused) == 1 else 'ies'} at a schema_version this reader "
            "does not know, not corruption; a plant-mapping receipt cannot be trusted while an "
            "entry could be hiding behind it unread")
    seen = _receipt_seen.setdefault(root_key, {})
    for entry in page.records:
        if entry.get("tool") != "plant_mapping_built":
            continue
        args = entry.get("arguments") or {}
        entry_name, sha = args.get("name"), args.get("record_sha256")
        if isinstance(entry_name, str) and isinstance(sha, str):
            seen.setdefault(entry_name, set()).add(sha)
    _receipt_cursor[root_key] = page.cursor


def _require_receipt(project_root: Path | str, name: str, record_sha256: str) -> None:
    """Refuse unless a ``plant_mapping_built`` event under ``name`` names ``record_sha256`` in
    the project's own audit log. Any matching receipt admits, not only the latest: the record
    write and the receipt append cannot be ordered across two concurrent legitimate builds of
    one name, so a latest-receipt rule could wedge a name permanently after an honest race."""
    root_key = str(Path(project_root).resolve())
    if root_key not in _receipt_cursor:
        _scan_receipts(project_root, root_key, after=None)
    if record_sha256 not in _receipt_seen.get(root_key, {}).get(name, set()):
        _scan_receipts(project_root, root_key, after=_receipt_cursor.get(root_key))
    if record_sha256 not in _receipt_seen.get(root_key, {}).get(name, set()):
        raise ValueError(
            f"plant mapping {name!r} under {project_root} carries no plant_mapping_built "
            f"receipt naming record {record_sha256}: this record was not written by "
            "build_plant_mapping or the web build route (a forged or hand-restored record, or "
            "one whose receipt could not be written); rebuild with build_plant_mapping")


def load_mapping(project_root: Path | str, name: str) -> Optional[MappingBuild]:
    """One project's named, persisted plant-mapping build, or ``None`` when nothing is stored
    under that name (the delivery's "build one first" refusal stands).

    Validates the record's shape, then requires a matching receipt (see :func:`_require_receipt`)
    before trusting it: a record shaped correctly but never built through the platform's own
    writers is refused, a forgery naming the real inputs' identities included.
    """
    raw = tcip_store.read(plant_mapping_key(project_root, name), default=None)
    if raw is None:
        return None
    record = _validated_record(raw, project_root, name)
    record_sha256 = record_digest(record)
    _require_receipt(project_root, name, record_sha256)
    assignments: dict[str, list[Assignment]] = {}
    for date, rows in record["assignments"].items():
        assignments[date] = [
            Assignment(
                image_path=r["image_path"], stem=r["stem"], date_folder=r["date_folder"],
                plot_name=r["plot_name"], accession_name=r["accession_name"],
                source=r["source"], distance_m=r["distance_m"],
            )
            for r in rows
        ]
    kwargs = {field_name: record[field_name] for field_name in _PERSISTED_FIELD_NAMES
              if field_name != "assignments"}
    return MappingBuild(**kwargs, assignments=assignments, record_sha256=record_sha256)


def resolved_mapping_key_for_citation(
    project_root: Path | str, name: str, record_sha256: str,
) -> Optional[str]:
    """The name a reader loads to see exactly the record a delivery event's own
    ``plant_mapping.record_sha256`` cites: ``name`` itself when the record currently stored
    under it still hashes to ``record_sha256``, the archived key a superseding rebuild moved
    it to (``f"{name}@{record_sha256[:12]}"``) when that key holds a stored record, or ``None``
    when neither does, so a caller renders the citation unresolved rather than a key that reads
    back nothing.

    For a delivery-event reader (the Results tab's panel route) to call, never for
    :func:`resolve_delivery_mapping`: a delivery resolves the mapping it is about to read by
    name, against the mapping's current inputs, not a historical record it may already have
    superseded.
    """
    current = tcip_store.read(plant_mapping_key(project_root, name), default=None)
    if isinstance(current, dict):
        try:
            if record_digest(_validated_record(current, project_root, name)) == record_sha256:
                return name
        except ValueError:
            pass
    archived_name = f"{name}@{record_sha256[:12]}"
    if tcip_store.exists(plant_mapping_key(project_root, archived_name)):
        return archived_name
    return None


def _describe_capture(stamps_by_stem: dict[str, ImageStamp], stem: str) -> str:
    """Name one capture for a refusal message: its kind and file name when a fresh stamp for
    ``stem`` was read this call, else the bare stem (a capture that vanished between the record
    and this recompute, which the added/missing-stem checks above already refuse before this is
    ever reached)."""
    s = stamps_by_stem.get(stem)
    if s is None:
        return stem
    if s.kind == "band_group":
        return f"band group manifest {s.name!r}"
    if s.kind == "raster":
        return f"raster {s.name!r}"
    return f"image {s.name!r}"


def verify_mapping_inputs(
    build: MappingBuild, dataset_root: Path | str, predictions_by_date: dict[str, str],
) -> dict:
    """Check what this delivery can verify about a mapping's recorded inputs against what is on
    disk now, for the captures it actually reads, at delivery time.

    The disclosures name what could not be verified, never merely what was not read: a delivered
    date whose image folder is absent, or an archived date, still has every one of its stems'
    predictions read and counted by the phenology aggregation (:func:`stems_delivery_reads` never
    checks whether the underlying image still exists), while its captures cannot be checked here
    and are disclosed unverified all the same; a date the delivery genuinely omits is unread in
    the ordinary sense too, every stem the mapping names for it counted missing downstream.

    A mapped date not named in ``predictions_by_date`` is never walked (no enumeration, no EXIF):
    disclosed in ``captures_unverified`` as the bare date string, the same as a named date whose
    image folder is absent. A named date's folder is enumerated once
    (``image_utils.list_logical_images``, still refusing by name on
    :class:`~tcip_mcp.pipelines.image_utils.AmbiguousImageStem`); an enumerated stem the
    mapping's own assignment rows for that date do not name means the mapping does not cover what
    is on disk, and refuses, naming the date, the file(s) and the rebuild remedy. A recorded stem
    no longer enumerated (its capture moved or was deleted) is disclosed as ``"<date>/<name>"``,
    using the recorded row's own file name; so is a recorded, still-enumerated stem this
    delivery's own prediction bucket carries no document for (:func:`stems_delivery_reads`),
    never opened to check.

    Only a capture this delivery reads gets its fresh stamp read. For each: a capture readable
    when this mapping was built and unreadable now refuses by name (the reverse cannot occur: an
    unreadable capture carries no GPS, so ``assign_plants`` never mapped it, and an unmapped
    capture is never among what a delivery reads through predictions). A row that recorded a
    plant position (:func:`assignment_is_attributed` true, with a ``distance_m``) refuses by
    name when the capture's fresh GPS position no longer sits ``distance_m`` from any plant of
    that name in the plant CSVs whose bytes this same call just verified, or when the fresh capture carries no GPS
    position at all. When no verified plant CSV can answer for this capture's own recorded plant
    (that plant's own CSV is itself among ``plant_csvs_unverified``), the position is disclosed
    rather than compared against nothing: the recorded fact stands unrechecked, not confirmed
    unchanged.

    When every mapped capture of a date was read (``missing_stems`` empty and ``read_set`` equal
    to the recorded stems :func:`assignment_is_attributed` calls attributed), the whole date's
    identity (:func:`capture_identity`) is recomputed over every capture the date enumerates, the unmapped
    ones (a raster, a band group) included, and compared against the record, refusing on a
    mismatch; a capture verified only this way is not also listed in ``captures_unverified``, since
    the digest just re-checked it. The refusal names the capture(s) whose own
    :func:`capture_digests` entry moved (a band group's manifest rewritten in place, most usefully,
    since nothing else names that case), not only the date, by comparing the record's per-capture
    digests against a fresh recompute over the same enumeration. Under a partial read, that digest
    does not run, so an in-place EXIF timestamp or band-group manifest change on a read capture goes
    undetected whenever some other mapped capture of the same date was not read: the position and
    readability checks above catch a moved plant or a capture gone unreadable, never a
    same-position, same-readability change in place.

    Never raises: returns ``{"refusal": str}`` for any of the above; otherwise
    ``{"captures_unverified": [...], "plant_csvs_unverified": [...]}``, entries in ``build.dates``
    order and, within a date, sorted by name. A missing plant CSV is still disclosed in
    ``plant_csvs_unverified`` as before; a rewritten one now refuses under this function's own
    remedy (restore the file's registered bytes, or register the current file under a new
    registry name, rebuild the mapping against it, and deliver under that name), composed here
    rather than read from :func:`verify_registry_csv_bytes`, which reports the bytes fact only.
    """
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.image_utils import (
        AmbiguousImageStem,
        list_logical_images,
        logical_image_name,
    )

    remedy = "rebuild with build_plant_mapping"

    # Plant CSVs first: the per-capture moved-position check below trusts only verified bytes.
    # A registry that no longer loads or whose digest has moved refuses rather than verifying nothing.
    registry_entries, registry_refusal = registry_entries_or_refusal(build, build.project_root)
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

        pred_dir = predictions_by_date.get(date)
        if pred_dir is None:
            captures_unverified.append(date)
            continue

        date_dir = image_dir(dataset_root, date)
        if not date_dir.is_dir():
            captures_unverified.append(date)
            continue

        try:
            logical = list_logical_images(date_dir)
        except AmbiguousImageStem as exc:
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
        read_set = stems_delivery_reads(rows, pred_dir) & present_stems
        unread_stems = present_stems - read_set
        mapped_stems = {s for s in recorded_stems if assignment_is_attributed(recorded_by_stem[s])}
        full_mapped_coverage = not missing_stems and read_set == mapped_stems

        unverified_names: set[str] = set()
        if not full_mapped_coverage:
            # An unmapped capture (a raster, a band group) is never read through predictions, so
            # it always lands here unless the whole-date digest below re-checks it instead.
            unverified_names.update(
                Path(recorded_by_stem[s].image_path).name for s in (missing_stems | unread_stems))

        read_logical = {s: logical[s] for s in read_set}
        stamps = _read_date_stamps(read_logical, date)
        was_unreadable = set(build.unreadable.get(date, []))
        for s in stamps:
            row = recorded_by_stem[s.stem]
            if s.kind == "image":
                if s.name not in was_unreadable and s.readable is False:
                    return {"refusal": (
                        f"{s.name} (date {date}) was readable when this mapping was built and "
                        "could not be read now: retry, or rebuild if it is gone")}
            if assignment_is_attributed(row) and row.distance_m is not None:
                if s.lat is None or s.lon is None:
                    return {"refusal": (
                        f"{s.name} (date {date}) recorded a plant position when this mapping was "
                        "built and now carries no GPS position: rebuild, since its assignment "
                        "would differ")}
                distances = [
                    haversine_m(s.lat, s.lon, p.lat, p.lon)
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
            whole_date_stamps = _read_date_stamps(logical, date)
            new_identity = capture_identity(whole_date_stamps)
            if new_identity != build.capture_identity.get(date):
                new_digests = capture_digests(whole_date_stamps)
                recorded_digests = build.capture_digests.get(date, {})
                stamps_by_stem = {s.stem: s for s in whole_date_stamps}
                moved = sorted(
                    _describe_capture(stamps_by_stem, stem)
                    for stem in set(new_digests) | set(recorded_digests)
                    if new_digests.get(stem) != recorded_digests.get(stem)
                )
                detail = ", ".join(moved) if moved else "a capture whose identity does not decompose"
                return {"refusal": (
                    f"date {date}: {detail} changed since this mapping was built; "
                    "rebuild to cover the images actually on disk")}

    return {"captures_unverified": captures_unverified, "plant_csvs_unverified": plant_csvs_unverified}


def ungeoreferenced_capture_message(walked: str, unreadable: Sequence[str] = ()) -> str:
    """The refusal sentence for a walk whose captures carry no position this door reads.

    ``walked`` names what was walked (the build passes ``images_root``; a delivery passes the
    mapping's name and dataset root), so the same sentence composer serves both doors this
    condition can refuse from. ``unreadable``, when given, opens the sentence by naming the
    captures PIL could not open at all, before the position clause: a capture that could not be
    read never carried a position to read either, and this says so rather than folding it into
    "no position" as if it had been opened and found blank.
    """
    prefix = ""
    if unreadable:
        prefix = f"{', '.join(unreadable)} could not be opened, so their position could not be read; "
    return (
        f"{prefix}no capture under {walked} on the requested dates carries a position this door "
        "reads (a photograph with no GPS position, or a raster or band-group capture, which never "
        "carries one here), so no capture can be assigned to a plant; per-plant identity for "
        "ungeoreferenced capture needs a plant-tag mechanism the platform does not have (README's "
        "roadmap), and a georeferenced orthomosaic delivers per plant through run_inference's "
        "raster regime and deliver_orthomosaic_plant_counts instead"
    )


class UngeoreferencedCaptureRefusal(Exception):
    """A plant mapping cannot be built or delivered from the captures at hand: none carries a
    position this door reads, or none could be read. ``str(exc)`` is the caller-facing message;
    ``status`` is the web door's HTTP status for it."""

    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class MappingDeliveryRefusal(Exception):
    """A phenology delivery cannot proceed from a named mapping; ``str(exc)`` is the caller-facing
    message. ``status`` is the web door's HTTP status for it (400 by default, 404 for a mapping
    that is not stored, 409 for a store-level problem reading it), unused by an MCP tool door,
    which reports ``str(exc)`` in its own ``{"error": ...}`` shape instead."""

    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def resolve_delivery_mapping(
    project_root: Path | str, name: str, predictions_by_date: dict[str, str],
) -> tuple[MappingBuild, dict]:
    """The two phenology doors' shared preamble: load the named mapping, refuse a
    ``predictions_by_date`` date it does not cover, resolve the delivered buckets' one dataset
    root and require it to carry the mapping's own minted dataset id, then verify the mapping's
    recorded inputs against that resolved root for exactly the captures
    ``predictions_by_date`` reads, so a dataset moved (or copied and the original renamed) after
    registration still verifies without re-reading a delivery's own unread captures. A delivery
    whose delivered dates attribute nothing (:func:`assignment_is_attributed` false for every one
    of them, a date recorded with no capture at all included) refuses on the record's own
    evidence, before ``verify_mapping_inputs`` re-reads anything, since a delivery that can
    attribute nothing has nothing for a re-read to disclose about; the no-capture-at-all reason is
    named only inside that same nothing-attributed scope, so a delivery naming a fully attributed
    date beside an empty one still ships: the empty date names no plant at all, so it is simply
    absent from every plant's own per-plant_phenology series rather than blocking the delivery.
    Returns the loaded build and :func:`verify_mapping_inputs`'s disclosure; raises
    :class:`MappingDeliveryRefusal`, naming the remedy, for every case a delivery must not
    proceed from, a ``StoreError`` reading the mapping store included, so a root whose state is
    still in the file layout refuses with the store's own sentence rather than escaping as an
    unhandled exception.
    """
    from tcip_store import StoreError

    from tcip_mcp.class_registry import RegistryError, dataset_root_for_pred_dirs
    from tcip_mcp.dataset_layout import require_dataset_identity

    try:
        mapping_build = load_mapping(project_root, name)
    except (StoreError, ValueError) as exc:
        raise MappingDeliveryRefusal(
            f"could not read mapping {name!r}: {exc}", status=409) from exc
    if mapping_build is None:
        raise MappingDeliveryRefusal(
            f"mapping not found: {name!r}; build one with build_plant_mapping before "
            "computing phenology", status=404)

    missing_dates = [d for d in predictions_by_date if d not in mapping_build.dates]
    if missing_dates:
        raise MappingDeliveryRefusal(
            f"predictions_by_date names date(s) {missing_dates} the mapping {name!r} does not "
            "cover; rebuild the mapping to cover them, or drop the date(s)")

    try:
        delivered_root = dataset_root_for_pred_dirs(list(predictions_by_date.values()))
    except RegistryError as exc:
        raise MappingDeliveryRefusal(str(exc)) from exc
    try:
        delivered_identity = require_dataset_identity(delivered_root)
    except ValueError as exc:
        raise MappingDeliveryRefusal(str(exc)) from exc
    if delivered_identity.get("id") != mapping_build.dataset_id:
        raise MappingDeliveryRefusal(
            f"the predictions under {delivered_root} belong to a different dataset than the "
            f"mapping {name!r} was built over (mapping dataset_root "
            f"{mapping_build.dataset_root!r}, delivered dataset root {str(delivered_root)!r})")

    delivered_assignments = [
        a for date in predictions_by_date for a in mapping_build.assignments.get(date, [])
    ]
    if not delivered_assignments or not any(
        assignment_is_attributed(a) for a in delivered_assignments
    ):
        # A no-capture-at-all date is named below only inside this nothing-attributed scope.
        empty_dates = sorted(d for d in predictions_by_date if not mapping_build.assignments.get(d))
        if empty_dates:
            raise MappingDeliveryRefusal(
                f"the mapping {name!r} recorded no capture at all for date(s) {empty_dates}: a "
                "date with no capture cannot be delivered; rebuild the mapping, or drop the "
                "date(s)")
        # Re-reading captures for a delivery that can attribute nothing would disclose about
        # nothing; refuse here, before verify_mapping_inputs, on the record's own evidence.
        registry_entries, registry_refusal = registry_entries_or_refusal(mapping_build, project_root)
        if registry_refusal:
            raise MappingDeliveryRefusal(registry_refusal, status=409)
        paths = [entry["path"] for entry in registry_entries]
        n_plants = sum(entry["n_plants"] for entry in registry_entries)
        if n_plants == 0:
            raise MappingDeliveryRefusal(
                f"the plant CSVs this mapping was built from ({paths}) parsed no plant with "
                "usable coordinates and a name, so no capture could be assigned; check the "
                "CSV's column headers against read_plant_csvs's and rebuild the mapping")
        if all(a.distance_m is None for a in delivered_assignments):
            raise MappingDeliveryRefusal(
                ungeoreferenced_capture_message(
                    f"mapping {name!r} (dataset {mapping_build.dataset_root!r})"))
        gates = match_gates(mapping_build.nn_tolerance_m["value"])
        n_unpositioned = sum(1 for a in delivered_assignments if a.distance_m is None)
        unpositioned_note = (
            f", and {n_unpositioned} captures carry no position" if n_unpositioned else "")
        raise MappingDeliveryRefusal(
            "every positioned capture on the delivered dates lies beyond the accepted match "
            f"distance ({gates['max_match_distance_m']} m from tolerance "
            f"{gates['nn_tolerance_m']} m, {mapping_build.nn_tolerance_m['source']}) of every "
            f"plant in {paths}{unpositioned_note}; check the plant CSV names this block, or "
            "rebuild the mapping with a stated tolerance")

    verified = verify_mapping_inputs(mapping_build, delivered_root, predictions_by_date)
    if "refusal" in verified:
        raise MappingDeliveryRefusal(verified["refusal"])
    return mapping_build, verified
