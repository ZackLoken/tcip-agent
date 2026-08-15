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
"""

from __future__ import annotations

import csv
import logging
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import tcip_store
from PIL import ExifTags, Image
from tcip_store import RECORD_JSON, Key, StoreDescriptor, register_store
from tcip_store.file_backend import RootedFileLocator

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_378_137.0

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".bmp")

NN_TOLERANCE_METERS = 10.0


@dataclass
class ImageStamp:
    """Per-image metadata extracted from EXIF."""

    path: str
    stem: str
    date_folder: str
    timestamp: Optional[datetime]
    lat: Optional[float]
    lon: Optional[float]
    h_pos_err: Optional[float]


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


# ── EXIF extraction ──────────────────────────────────────────────────────

_GPS_TAG = next((k for k, v in ExifTags.TAGS.items() if v == "GPSInfo"), None)
_DT_TAG = next((k for k, v in ExifTags.TAGS.items() if v == "DateTimeOriginal"), None)


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
    stamp = ImageStamp(
        path=str(path),
        stem=path.stem,
        date_folder=date_folder,
        timestamp=None,
        lat=None,
        lon=None,
        h_pos_err=None,
    )
    try:
        with Image.open(path) as im:
            exif = im._getexif() or {}
    except Exception:
        return stamp

    # DateTimeOriginal
    if _DT_TAG is not None and _DT_TAG in exif:
        try:
            stamp.timestamp = datetime.strptime(str(exif[_DT_TAG]), "%Y:%m:%d %H:%M:%S")
        except Exception:
            pass

    # GPS sub-dictionary
    if _GPS_TAG is not None and _GPS_TAG in exif:
        gps_raw = exif[_GPS_TAG]
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


def read_directory(date_dir: Path) -> list[ImageStamp]:
    if not date_dir.is_dir():
        return []
    stamps: list[ImageStamp] = []
    for p in sorted(date_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            stamps.append(read_image_stamp(p, date_dir.name))
    return stamps


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


# Iglewicz & Hoaglin's modified z-score outlier test (Iglewicz, B. and Hoaglin, D. (1993), "How to
# Detect and Handle Outliers", ASQC Quality Press), the standard median/MAD-based convention for
# flagging an outlier in a small, possibly non-normal sample: a candidate row-transition jump must
# clear this critical value on its modified z-score, computed in log-space (so "how many times
# bigger" becomes an additive, mean/spread-comparable quantity) against the rest of the date's own
# consecutive-gap ratios. 0.6745 is the constant that makes MAD a consistent estimator of the
# standard deviation for a normally-distributed population; 3.5 is the convention's own published
# cutoff, the same "cited statistical convention with a stated confidence level" shape as
# ``operating_point._EQUIVALENCE_Z``, not a multiplier tuned to pass a fixture.
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
    # A near-zero gap (two images at essentially the same GPS fix, e.g. a duplicate/cached EXIF
    # reading between two rapid captures) carries no walking-pace information: taking its log, or
    # dividing by it, would make whatever value follows look like an infinitely large jump purely
    # by construction, not because a row transition actually happened there. Excluded as a
    # candidate split point entirely, rather than treated as an automatic winner.
    candidates = [
        (i, math.log(hi / lo)) for i, (lo, hi) in enumerate(zip(sorted_gaps, sorted_gaps[1:]))
        if lo > 1e-9
    ]
    if not candidates:
        return None
    best_i, best_log_ratio = max(candidates, key=lambda pair: pair[1])

    other_log_ratios = [r for i, r in candidates if i != best_i]
    if not other_log_ratios:
        # Only one candidate split to look at: nothing to compare it against, so there's no basis
        # to call it a genuine transition rather than noise.
        return None
    median_other = statistics.median(other_log_ratios)
    mad = statistics.median([abs(r - median_other) for r in other_log_ratios])
    if mad > 0:
        modified_z = 0.6745 * (best_log_ratio - median_other) / mad
        if modified_z < _ROW_BREAK_MODIFIED_Z:
            return None
    elif best_log_ratio <= median_other:
        # The rest of this date's own steps show no measured spread at all (identical, or equal up
        # to floating-point precision): only a genuinely larger jump counts as a break, the modified
        # z-score's own limiting behavior as MAD -> 0, not an unconditional pass on any candidate.
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
    """Assign each image to a plant by sequence-anchored NN matching."""
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

    # Path-aware assignment within a run: each image picks its nearest
    # unclaimed plant, biasing toward plants close to the run's start/end
    # anchors. For now, the first pass just uses nearest-neighbour but
    # penalises duplicates; it degrades gracefully to NN when sequence
    # signal is weak.
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
    dates: Optional[list[str]] = None,
    nn_tolerance_m: Optional[float] = None,
) -> dict[str, list[Assignment]]:
    """Build per-date plant assignments for every image under ``images_root``.

    ``nn_tolerance_m`` is derived from the plot's ``grid_pitch_m`` when the caller does not pin it
    (not a pinned 10 m): pitch/6, so assign_plants' loosest 3x gate keeps the effective match radius
    within half a grid cell. An explicit value is honored but still capped at that ceiling. No
    derivable pitch (< 2 georeferenced plants) -> the honest ``NN_TOLERANCE_METERS`` fallback.
    """
    plants = read_plant_csvs(plant_csv_paths)
    # The ceiling: assign_plants' loosest gate is 3x nn_tolerance, so pitch/6 -> effective radius
    # <= pitch/2. Derive the tolerance from the layout in hand; only cap an explicit override.
    pitch = grid_pitch_m(plants)
    _cap = pitch / 6
    if nn_tolerance_m is None:
        nn_tolerance_m = _cap if pitch > 0 else NN_TOLERANCE_METERS
        logger.info("nn_tolerance_m derived %.2f from grid pitch %.1f (effective radius <= pitch/2)",
                    nn_tolerance_m, pitch)
    elif pitch > 0 and nn_tolerance_m > _cap:
        logger.info("capping nn_tolerance_m %.1f -> %.2f (grid pitch %.1f: effective radius <= pitch/2)",
                    nn_tolerance_m, _cap, pitch)
        nn_tolerance_m = _cap
    result: dict[str, list[Assignment]] = {}
    images_root = Path(images_root)
    if not images_root.is_dir():
        return result
    for date_dir in sorted(images_root.iterdir()):
        if not date_dir.is_dir():
            continue
        if dates is not None and date_dir.name not in dates:
            continue
        stamps = read_directory(date_dir)
        if not stamps:
            result[date_dir.name] = []
            continue
        assignments = assign_plants(stamps, plants, nn_tolerance_m=nn_tolerance_m)
        result[date_dir.name] = assignments
    return result


PLANT_MAPPING_STORE = "plant_mapping"
_MAPPING_DOC = RootedFileLocator(suffix=".json")
register_store(
    StoreDescriptor(
        name=PLANT_MAPPING_STORE,
        kind="record",
        key_fields=("document",),
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        locator=_MAPPING_DOC,
    )
)


def plant_mapping_key(path: Path | str) -> Key:
    """One build's per-date plant assignments, addressed by the document the caller named.

    The generic placement, because the caller chooses where a mapping lands (a project's state
    directory, an export bucket) and this module resolves no layout of its own.

    ``last_writer_wins``: a mapping is assigned whole in memory and written in one call, and a
    later build over the same document is a fresh assignment replacing that one rather than a
    merge into it. No writer reads the record first.
    """
    document = Path(path).absolute()
    return Key(PLANT_MAPPING_STORE, str(document.parent), (document.stem,))


def persist_mapping(mapping: dict[str, list[Assignment]], out_path: Path) -> None:
    serialisable = {
        date: [a.__dict__ for a in assignments] for date, assignments in mapping.items()
    }
    tcip_store.replace(plant_mapping_key(out_path), serialisable)


def load_mapping(path: Path) -> dict[str, list[Assignment]]:
    raw = tcip_store.read(plant_mapping_key(path), default=None)
    if raw is None:
        return {}
    out: dict[str, list[Assignment]] = {}
    for date, rows in raw.items():
        out[date] = []
        for r in rows:
            out[date].append(
                Assignment(
                    image_path=r.get("image_path", ""),
                    stem=r.get("stem", ""),
                    date_folder=r.get("date_folder", date),
                    plot_name=r.get("plot_name"),
                    accession_name=r.get("accession_name"),
                    source=r.get("source", "unknown"),
                    distance_m=(
                        float(r["distance_m"]) if r.get("distance_m") is not None else None
                    ),
                )
            )
    return out
