"""Results routes: plant-mapping, per-plant phenology curves, CSV export.

The Phase 1 target is a per-plant CSV with catkin_05 / 50 / 95per_date columns.
That pipeline looks like:

    predictions(date) ─► foreground mask(bush model) ─► count catkins per plant
                    ─► classify elongated vs dormant by bbox-height threshold
                    ─► ratio elongated/total per (plant, date)
                    ─► fit curve + find 5/50/95% crossings

For Phase 1 the backend owns everything except the model inference (which is
driven by the Inference tab). This module provides the HTTP endpoints the
Results tab calls.
"""

from __future__ import annotations

import csv
import logging
from io import StringIO
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from tcip_web import plant_mapping

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/results", tags=["results"])


# ── Plant mapping ──────────────────────────────────────────────────────


class BuildMappingPayload(BaseModel):
    images_root: str
    plant_csv_paths: list[str]
    dates: Optional[list[str]] = None
    nn_tolerance_m: float = 10.0
    persist_path: Optional[str] = None


@router.post("/plant_mapping/build")
def build_plant_mapping(payload: BuildMappingPayload) -> dict:
    mapping = plant_mapping.build_mapping(
        Path(payload.images_root),
        [Path(p) for p in payload.plant_csv_paths],
        dates=payload.dates,
        nn_tolerance_m=payload.nn_tolerance_m,
    )
    if payload.persist_path:
        plant_mapping.persist_mapping(mapping, Path(payload.persist_path))

    summary = {}
    for date, assignments in mapping.items():
        matched = sum(1 for a in assignments if a.plot_name)
        summary[date] = {
            "n_images": len(assignments),
            "n_mapped": matched,
            "avg_distance_m": (
                sum(a.distance_m for a in assignments if a.distance_m is not None)
                / max(1, sum(1 for a in assignments if a.distance_m is not None))
            ),
        }

    return {
        "mapping": {
            date: [a.__dict__ for a in assignments] for date, assignments in mapping.items()
        },
        "summary": summary,
    }


class LoadMappingPayload(BaseModel):
    persist_path: str


@router.post("/plant_mapping/load")
def load_plant_mapping(payload: LoadMappingPayload) -> dict:
    mapping = plant_mapping.load_mapping(Path(payload.persist_path))
    return {
        "mapping": {
            date: [a.__dict__ for a in assignments] for date, assignments in mapping.items()
        }
    }


# ── Per-plant curves ───────────────────────────────────────────────────


def _count_predictions(txt_path: Path) -> tuple[int, int]:
    """Return (total, n_elongated) based on bbox-height threshold.

    The prediction file is YOLO format ``class conf cx cy w h`` with
    normalised coordinates (0-1). Elongated catkins have substantially
    larger bbox height (longer vertical pendant); we use a threshold on
    normalised h that the caller can tune.
    """
    return _count_with_threshold(txt_path, elongation_height=0.020)


def _count_with_threshold(txt_path: Path, elongation_height: float) -> tuple[int, int]:
    if not txt_path.exists():
        return (0, 0)
    total = 0
    elongated = 0
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            try:
                h = float(parts[5])
            except ValueError:
                continue
            total += 1
            if h >= elongation_height:
                elongated += 1
    return total, elongated


class PerPlantCurvesPayload(BaseModel):
    project_root: str
    mapping_path: str  # .tcip/state/plant_mapping.json or equivalent
    # map date → predictions directory (YOLO detect txt files) for that date
    predictions_by_date: dict[str, str]
    elongation_height: float = 0.02


class PerPlantCurveRow(BaseModel):
    plant_id: str
    accession: Optional[str]
    date: str
    n_images: int
    n_total: int
    n_elongated: int
    ratio: float


@router.post("/per_plant_curves")
def per_plant_curves(payload: PerPlantCurvesPayload) -> dict:
    mapping = plant_mapping.load_mapping(Path(payload.mapping_path))
    if not mapping:
        raise HTTPException(404, f"no mapping at {payload.mapping_path}")

    rows: list[dict] = []
    per_plant: dict[str, dict[str, dict]] = {}

    for date, pred_dir in payload.predictions_by_date.items():
        pred_path = Path(pred_dir)
        if date not in mapping:
            continue
        for assignment in mapping[date]:
            if not assignment.plot_name:
                continue
            txt = pred_path / f"{assignment.stem}.txt"
            total, elongated = _count_with_threshold(txt, payload.elongation_height)
            plant_id = assignment.plot_name
            per_plant.setdefault(plant_id, {})
            entry = per_plant[plant_id].setdefault(
                date,
                {
                    "plant_id": plant_id,
                    "accession": assignment.accession_name,
                    "date": date,
                    "n_images": 0,
                    "n_total": 0,
                    "n_elongated": 0,
                },
            )
            entry["n_images"] += 1
            entry["n_total"] += total
            entry["n_elongated"] += elongated

    for plant_id, by_date in per_plant.items():
        for date, entry in by_date.items():
            ratio = entry["n_elongated"] / entry["n_total"] if entry["n_total"] else 0.0
            entry["ratio"] = ratio
            rows.append(entry)

    return {"rows": rows, "n_plants": len(per_plant)}


# ── Onset-date extraction + CSV export ─────────────────────────────────


def _date_key(date_str: str) -> tuple[int, int, int]:
    """Parse an ISO date (``YYYY-MM-DD``) → ``(year, month, day)`` for chronological sort.

    Dates are ISO platform-wide — ingestion buckets images by EXIF capture date into
    ``images/<YYYY-MM-DD>/``. A malformed value sorts first as ``(0, 0, 0)``.
    """
    parts = date_str.split("-")
    if len(parts) != 3:
        return (0, 0, 0)
    try:
        y, m, d = (int(x) for x in parts)
    except ValueError:
        return (0, 0, 0)
    return (y, m, d)


def _iso(date_str: str) -> str:
    y, m, d = _date_key(date_str)
    return f"{y:04d}-{m:02d}-{d:02d}"


def _crossing_date(
    sorted_points: list[tuple[str, float]], target_ratio: float
) -> Optional[str]:
    """Find the earliest date on which the ratio curve ≥ target_ratio.

    Linear interpolation between neighbouring points. If the final point is
    still below the target, return ``None``. If the first point is already
    at/above, return that date's ISO string.
    """
    if not sorted_points:
        return None
    # First point already meets target?
    if sorted_points[0][1] >= target_ratio:
        return _iso(sorted_points[0][0])
    for (d1, r1), (d2, r2) in zip(sorted_points, sorted_points[1:]):
        if r2 >= target_ratio:
            # Interpolate between d1 and d2
            y1, m1, day1 = _date_key(d1)
            y2, m2, day2 = _date_key(d2)
            # Straight interpolation using the ordinal delta as days is fine
            # for our tight ~6-week window (no month boundaries to worry about)
            if r2 == r1:
                return _iso(d2)
            t = (target_ratio - r1) / (r2 - r1)
            t = max(0.0, min(1.0, t))
            from datetime import date, timedelta

            d1_obj = date(y1, m1, day1)
            d2_obj = date(y2, m2, day2)
            delta = (d2_obj - d1_obj).days
            est = d1_obj + timedelta(days=round(t * delta))
            return est.isoformat()
    return None


class OnsetDatesPayload(BaseModel):
    curves: list[dict]  # output of per_plant_curves


@router.post("/onset_dates")
def onset_dates(payload: OnsetDatesPayload) -> dict:
    """Compute catkin_05/50/95per_date for each plant from its ratio curve."""
    plants: dict[str, list[dict]] = {}
    for row in payload.curves:
        plants.setdefault(row["plant_id"], []).append(row)

    out: list[dict] = []
    for plant_id, rows in plants.items():
        ordered = sorted(rows, key=lambda r: _date_key(r["date"]))
        series = [(r["date"], float(r["ratio"])) for r in ordered]
        out.append(
            {
                "plant_id": plant_id,
                "accession": rows[0].get("accession"),
                "n_datapoints": len(rows),
                "catkin_05per_date": _crossing_date(series, 0.05),
                "catkin_50per_date": _crossing_date(series, 0.50),
                "catkin_95per_date": _crossing_date(series, 0.95),
            }
        )
    return {"rows": out}


class ExportCsvPayload(BaseModel):
    rows: list[dict]
    filename: str = "catkin_phenology.csv"
    # Optional structural hint — the frontend can pre-compute onset-date rows
    # via /onset_dates and pass the rows directly. We honour whatever keys are
    # present.


@router.post("/export_csv")
def export_csv(payload: ExportCsvPayload) -> Response:
    if not payload.rows:
        raise HTTPException(400, "no rows to export")
    keys: list[str] = []
    seen: set[str] = set()
    for row in payload.rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    for row in payload.rows:
        writer.writerow(row)
    body = buf.getvalue()
    headers = {"Content-Disposition": f'attachment; filename="{payload.filename}"'}
    return Response(content=body, media_type="text/csv", headers=headers)


# ── List registered models (used by Inference tab) ─────────────────────


@router.get("/models/registered")
def registered_models(project_path: str, tag: Optional[str] = None) -> dict:
    from tcip_mcp.tools.model_tools import list_registered_models

    return list_registered_models(project_path, tag)
