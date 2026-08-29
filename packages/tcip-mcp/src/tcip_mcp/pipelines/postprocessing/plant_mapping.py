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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

import tcip_store
from PIL import ExifTags, Image
from tcip_store import RECORD_JSON, Key, StoreDescriptor, register_store
from tcip_store.file_backend import RootedFileLocator

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_378_137.0

NN_TOLERANCE_METERS = 10.0


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
    plant_csvs: list[dict]
    capture_identity: dict[str, str]
    capture_digests: dict[str, dict[str, str]]
    unreadable: dict[str, list[str]]
    assignments: dict[str, list[Assignment]] = field(default_factory=dict)
    record_sha256: str = ""
    """The digest ``load_mapping`` verified this record's receipt against; blank on a build not
    yet read back through ``load_mapping`` (a fresh ``build_mapping`` result before persisting)."""

    def to_record(self) -> dict:
        """The exact document ``persist_mapping`` writes and ``load_mapping`` reads back."""
        return {
            "name": self.name,
            "project_root": self.project_root,
            "dataset_root": self.dataset_root,
            "dataset_id": self.dataset_id,
            "built_by": self.built_by,
            "built_at": self.built_at,
            "dates_requested": self.dates_requested,
            "dates": self.dates,
            "nn_tolerance_m": self.nn_tolerance_m,
            "plant_csvs": self.plant_csvs,
            "capture_identity": self.capture_identity,
            "capture_digests": self.capture_digests,
            "unreadable": self.unreadable,
            "assignments": {
                date: [a.__dict__ for a in assignments]
                for date, assignments in self.assignments.items()
            },
        }

    def rows(self) -> dict[str, list[dict]]:
        """The assignments as plain per-date dict rows, the shape ``per_plant_phenology`` reads."""
        return {date: [a.__dict__ for a in assignments]
                for date, assignments in self.assignments.items()}

    def delivery_disclosure(self, verified: dict) -> dict:
        """The ``plant_mapping`` dict a phenology delivery carries: this build's own identity
        plus ``verify_mapping_inputs``'s disclosure, the one composition every phenology door
        (``compute_phenology``, both web phenology routes) builds through rather than each
        assembling its own copy."""
        return {
            "name": self.name,
            "project_root": self.project_root,
            "dataset_id": self.dataset_id,
            "dataset_root": self.dataset_root,
            "built_at": self.built_at,
            "record_sha256": self.record_sha256,
            "capture_identity": self.capture_identity,
            "captures_unverified": verified["captures_unverified"],
            "plant_csvs_unverified": verified["plant_csvs_unverified"],
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


def _row_digest(row: list[object]) -> str:
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


def read_plant_csvs(paths: Iterable[Path]) -> list[PlantRecord]:
    records: list[PlantRecord] = []
    for path in paths:
        p = Path(path)
        if not p.is_file():
            continue
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
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


def _maybe_float(x: Optional[str]) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


# ── Geometry helpers ───────────────────────────────────────────────────


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return EARTH_RADIUS_M * c


def stems_delivery_reads(rows: Iterable[object], pred_dir: Path | str) -> set[str]:
    """Stems of ``rows`` (one date's assignment rows, :class:`Assignment` objects or the plain
    dict rows :meth:`MappingBuild.rows` produces) whose ``plot_name`` is truthy and whose stem
    carries a prediction document under ``pred_dir``: the one predicate for "which prediction
    documents this delivery reads," never whether the underlying image still exists on disk.

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
        if not _attr(row, "plot_name"):
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

            if best_idx < 0 or best_d is None or best_d > nn_tolerance_m * 2:
                # Fall through to plain NN, even if claimed: duplicates can happen
                plant, d = _nearest_plant(s.lat, s.lon, plants)
                if plant is None or d is None or d > nn_tolerance_m * 3:
                    out.append(
                        Assignment(
                            image_path=s.path,
                            stem=s.stem,
                            date_folder=s.date_folder,
                            plot_name=None,
                            accession_name=None,
                            source="unmapped",
                            distance_m=d,
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
                            distance_m=d,
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


def build_mapping(
    images_root: Path,
    plant_csv_paths: list[Path],
    *,
    name: str,
    dataset_root: Path | str,
    dataset_id: str,
    project_root: Path | str,
    built_by: str,
    dates: Optional[list[str]] = None,
    nn_tolerance_m: Optional[float] = None,
) -> MappingBuild:
    """Build one project's named plant mapping: per-date assignments plus the provenance that
    binds the record to the inputs it was built from.

    ``name``/``dataset_root``/``dataset_id``/``project_root``/``built_by`` are the caller's own
    resolved facts (the door already checked the dataset identity and the project record before
    calling here); this function does not re-derive them.

    ``nn_tolerance_m`` is derived from the plot's ``grid_pitch_m`` when the caller does not pin it
    (not a pinned 10 m): pitch/6, so assign_plants' loosest 3x gate keeps the effective match
    radius within half a grid cell. An explicit value is honored but still capped at that
    pitch-derived ceiling; ``nn_tolerance_m`` on the record carries which of the four branches
    (derived, capped, stated, fallback) produced the value.

    A date's captures are enumerated through ``image_utils.list_logical_images``, so a band
    raster or a band group ingested under a mapped date is a capture the identity sees. That
    enumeration raises :class:`~tcip_mcp.pipelines.image_utils.AmbiguousImageStem` when a
    standalone file's stem collides with a band group's; this function lets it propagate, and
    the calling door catches it and refuses in its own error shape.
    """
    from tcip_mcp.pipelines.image_utils import list_logical_images

    plant_csv_paths = [Path(p) for p in plant_csv_paths]
    plants = read_plant_csvs(plant_csv_paths)
    plant_csv_meta = [
        {
            "path": str(p),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "n_plants": len(read_plant_csvs([p])),
        }
        for p in plant_csv_paths if p.is_file()
    ]

    # The ceiling: assign_plants' loosest gate is 3x nn_tolerance, so pitch/6 -> effective radius
    # <= pitch/2. Derive the tolerance from the layout in hand; only cap an explicit override.
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
            capture_ids[date] = capture_identity(stamps)
            capture_digests_by_date[date] = capture_digests(stamps)
            unreadable[date] = sorted(
                s.name for s in stamps if s.kind == "image" and s.readable is False)
            assignments[date] = assign_plants(stamps, plants, nn_tolerance_m=nn_tolerance_m)

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
        plant_csvs=plant_csv_meta,
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


def persist_mapping(build: MappingBuild, project_root: Path | str, name: str) -> None:
    """Write the mapping record, then the receipt that binds it to this build.

    The record is committed before the receipt (a log append cannot join a record transaction):
    a receipt that cannot be written fails loudly (``AuditEntryNotWritten`` propagates, never
    swallowed) and leaves a record no receipt names, which :func:`load_mapping` refuses to read
    until a rebuild replaces it.
    """
    from tcip_mcp.audit import record_event_or_raise

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
    "plant_csvs": list,
    "capture_identity": dict,
    "capture_digests": dict,
    "unreadable": dict,
    "assignments": dict,
}
_ASSIGNMENT_ROW_KEYS = (
    "image_path", "stem", "date_folder", "plot_name", "accession_name", "source", "distance_m")
_VALID_SOURCES = {"sequence", "nearest_neighbour", "unmapped"}


def _validated_record(raw: object, project_root: Path | str, name: str) -> dict:
    """``raw`` as a plant-mapping record, or the ``ValueError`` naming the project, the name,
    the field and the remedy (rebuild)."""
    remedy = "rebuild with build_plant_mapping"
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
    return MappingBuild(
        name=record["name"], project_root=record["project_root"],
        dataset_root=record["dataset_root"], dataset_id=record["dataset_id"],
        built_by=record["built_by"], built_at=record["built_at"],
        dates_requested=record["dates_requested"], dates=record["dates"],
        nn_tolerance_m=record["nn_tolerance_m"], plant_csvs=record["plant_csvs"],
        capture_identity=record["capture_identity"],
        capture_digests=record["capture_digests"], unreadable=record["unreadable"],
        assignments=assignments, record_sha256=record_sha256,
    )


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
    plant position (a truthy ``plot_name`` with a ``distance_m``) refuses by name when the
    capture's fresh GPS position no longer sits ``distance_m`` from any plant of that name in the
    plant CSVs whose bytes this same call just verified, or when the fresh capture carries no GPS
    position at all. When no verified plant CSV can answer for this capture's own recorded plant
    (that plant's own CSV is itself among ``plant_csvs_unverified``), the position is disclosed
    rather than compared against nothing: the recorded fact stands unrechecked, not confirmed
    unchanged.

    When every mapped capture of a date was read (``missing_stems`` empty and ``read_set`` equal
    to the recorded stems whose row carries a truthy ``plot_name``), the whole date's identity
    (:func:`capture_identity`) is recomputed over every capture the date enumerates, the unmapped
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
    order and, within a date, sorted by name. A plant CSV missing or rewritten in place
    disclose/refuse exactly as before.
    """
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.image_utils import (
        AmbiguousImageStem,
        list_logical_images,
        logical_image_name,
    )

    remedy = "rebuild with build_plant_mapping"

    # Plant CSVs first: the per-capture moved-position check below trusts only verified bytes.
    plant_csvs_unverified: list[str] = []
    for entry in build.plant_csvs:
        p = Path(entry["path"])
        if not p.is_file():
            plant_csvs_unverified.append(entry["path"])
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            return {"refusal": (
                f"plant CSV {entry['path']} was rewritten since this mapping was built; "
                "rebuild")}
    verified_plants = read_plant_csvs(
        Path(entry["path"]) for entry in build.plant_csvs
        if entry["path"] not in plant_csvs_unverified
    )

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
        mapped_stems = {s for s in recorded_stems if recorded_by_stem[s].plot_name}
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
            if row.plot_name and row.distance_m is not None:
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
    registration still verifies without re-reading a delivery's own unread captures. Returns the
    loaded build and :func:`verify_mapping_inputs`'s disclosure; raises
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

    verified = verify_mapping_inputs(mapping_build, delivered_root, predictions_by_date)
    if "refusal" in verified:
        raise MappingDeliveryRefusal(verified["refusal"])
    return mapping_build, verified
