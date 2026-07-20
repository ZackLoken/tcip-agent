"""Results routes: plant-mapping, per-plant phenology curves, CSV export.

The Phase 1 target is a per-plant CSV with catkin_05 / 50 / 95per_date columns.
That pipeline looks like:

    predictions(date) ─► per-plant catkin detections (via plant mapping)
                    ─► classify each catkin elongated vs not (validated classifier)
                    ─► fraction elongated / total per (plant, date)
                    ─► find the dates that fraction crosses 5/50/95%

Bloom is the *elongated fraction* of a plant's catkins — "elongated" being an expert-
defined morphological stage from a validated classifier, never a geometric proxy such
as bbox height (see the ``phenology`` skill + the CLAUDE.md measurement-integrity
invariant). The milestone math lives once in ``tcip_mcp...postprocessing.phenology``;
this module is the HTTP surface the Results tab calls and delegates to it.

For Phase 1 the backend owns everything except the model inference (which is
driven by the Inference tab).
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
from tcip_mcp.pipelines.postprocessing import phenology, plant_mapping

from tcip_web.paths import assert_path_allowed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/results", tags=["results"])


def _guard(*paths: str | None) -> None:
    """Confine client-supplied file paths to the allowed roots (no-op unless TCIP_IMAGE_ROOTS)."""
    for p in paths:
        if not p:
            continue
        try:
            assert_path_allowed(p)
        except ValueError as exc:
            raise HTTPException(403, str(exc)) from exc


# ── Plant mapping ──────────────────────────────────────────────────────


class BuildMappingPayload(BaseModel):
    images_root: str
    plant_csv_paths: list[str]
    dates: Optional[list[str]] = None
    nn_tolerance_m: float = 10.0
    persist_path: Optional[str] = None


@router.post("/plant_mapping/build")
def build_plant_mapping(payload: BuildMappingPayload) -> dict:
    _guard(payload.images_root, payload.persist_path, *payload.plant_csv_paths)
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
    _guard(payload.persist_path)
    mapping = plant_mapping.load_mapping(Path(payload.persist_path))
    return {
        "mapping": {
            date: [a.__dict__ for a in assignments] for date, assignments in mapping.items()
        }
    }


# ── Per-plant curves ───────────────────────────────────────────────────


class PerPlantCurvesPayload(BaseModel):
    project_root: str
    mapping_path: str  # .tcip/state/plant_mapping.json or equivalent
    # map date → predictions directory (YOLO detect txt files) for that date. Predictions
    # must come from the validated 2-class elongation classifier (class
    # ``elongated_class_id`` = elongated); raw single-class detections carry no elongation
    # signal and cannot yield a bloom ratio.
    predictions_by_date: dict[str, str]
    elongated_class_id: int = 1


class PerPlantCurveRow(BaseModel):
    # Required = what onset_dates dereferences (malformed -> 422, not a KeyError/500); rest stay optional.
    plant_id: str
    date: str
    ratio: float
    accession: Optional[str] = None
    n_images: int = 0
    n_total: int = 0
    n_elongated: int = 0


@router.post("/per_plant_curves")
def per_plant_curves(payload: PerPlantCurvesPayload) -> dict:
    """Per-(plant, date) elongated-catkin fraction from CLASSIFIED predictions.

    Bloom, per the trait definition, is the fraction of a plant's detected catkins that are
    elongated — "elongated" being an expert-defined morphological stage emitted by a
    *validated* classifier, never a geometric proxy. If the predictions carry no elongation
    class, ``elongation_classified`` is false and the ratios are not a valid bloom
    measurement (run + validate the classifier first — see the ``phenology`` skill).
    """
    _guard(payload.mapping_path, *payload.predictions_by_date.values())
    mapping = plant_mapping.load_mapping(Path(payload.mapping_path))
    if not mapping:
        raise HTTPException(404, f"no mapping at {payload.mapping_path}")

    rows: list[dict] = []
    per_plant: dict[str, dict[str, dict]] = {}
    all_classes: set[int] = set()

    for date, pred_dir in payload.predictions_by_date.items():
        pred_path = Path(pred_dir)
        if date not in mapping:
            continue
        for assignment in mapping[date]:
            if not assignment.plot_name:
                continue
            pred_json = pred_path / f"{assignment.stem}.json"
            total, elongated, classes_seen = phenology.count_by_class(
                pred_json, payload.elongated_class_id
            )
            all_classes |= classes_seen
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
            entry["ratio"] = entry["n_elongated"] / entry["n_total"] if entry["n_total"] else 0.0
            rows.append(entry)

    return {
        "rows": rows,
        "n_plants": len(per_plant),
        "classes_seen": sorted(all_classes),
        # Honest signal: were any predictions actually elongation-classified? If false, the
        # ratios above are not a valid bloom measurement — do not deliver curves from them.
        "elongation_classified": payload.elongated_class_id in all_classes,
    }


# ── Onset-date extraction + CSV export ─────────────────────────────────


class OnsetDatesPayload(BaseModel):
    curves: list[PerPlantCurveRow]  # output of per_plant_curves


@router.post("/onset_dates")
def onset_dates(payload: OnsetDatesPayload) -> dict:
    """Compute each plant's catkin phenology milestones from its elongated-fraction curve.

    Delegates to the canonical ``phenology`` module so a milestone date means the same
    thing here and in the ``compute_phenology`` MCP tool: ``catkin_05/50/95per_date`` are
    the dates the elongated fraction crosses those levels; ``catkin_elongation_date`` is the
    date most catkins have elongated (``crops.yml``) — the 95% majority crossing.
    """
    plants: dict[str, list[PerPlantCurveRow]] = {}
    for row in payload.curves:
        plants.setdefault(row.plant_id, []).append(row)

    out: list[dict] = []
    for plant_id, rows in plants.items():
        ordered = sorted(rows, key=lambda r: phenology.date_key(r.date))
        series = [(r.date, float(r.ratio)) for r in ordered]
        out.append(
            {
                "plant_id": plant_id,
                "accession": rows[0].accession,
                "n_datapoints": len(rows),
                **phenology.plant_milestones(series),
            }
        )
    return {"rows": out}


class ExportCsvPayload(BaseModel):
    rows: list[dict]
    filename: str = "catkin_phenology.csv"
    # Optional structural hint — the frontend can pre-compute onset-date rows
    # via /onset_dates and pass the rows directly. We honour whatever keys are
    # present.
    # When these rows are a bloom delivery, pass the prediction buckets the curve was
    # computed from so the gate reads the count operating point's validity from each
    # bucket's operating_point.json (the on-disk floor), not a caller-supplied row string.
    predictions_by_date: Optional[dict[str, str]] = None


@router.post("/export_csv")
def export_csv(payload: ExportCsvPayload) -> Response:
    if not payload.rows:
        raise HTTPException(400, "no rows to export")

    # Phenology-delivery guard: if these rows are a bloom-phenotype delivery (carry milestone dates),
    # every row must carry a validated classifier AND a validated count operating point — the same gate
    # compute_phenology enforces, applied here so this generic export can't become a second, un-gated
    # door that ships an unvalidated phenotype CSV. The operating point's validity is read from each
    # bucket's operating_point.json (floored against any row string), never trusted from the row alone.
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_SHIPPABLE,
        check_delivery_gate,
        reconcile_operating_point_validity,
    )
    _milestones = {"catkin_05per_date", "catkin_50per_date", "catkin_95per_date"}
    if any(_milestones & set(r.keys()) for r in payload.rows):
        op_state: str | None = None
        if payload.predictions_by_date:
            recon = reconcile_operating_point_validity(list(payload.predictions_by_date.values()))
            op_state = recon["validated"]
            if op_state not in VALIDATED_SHIPPABLE:
                raise HTTPException(
                    400,
                    "phenology delivery: the count operating point is not validated on disk "
                    f"(reconciled from operating_point.json = {op_state!r}; missing sidecars: "
                    f"{recon['missing_sidecars']}; unvalidated buckets: {recon['unvalidated_buckets']}). "
                    "Produce the predictions via a calibrated export_predictions, then compute_phenology.",
                )
        for r in payload.rows:
            # On-disk reconciliation wins when buckets were supplied; else fall back to the row's stamp
            # (itself written by compute_phenology's on-disk gate) — never upgrade an unvalidated row.
            row_op = op_state if op_state is not None else r.get("operating_point_validated")
            # The web door has no acknowledge escape — an unvalidated phenotype CSV is refused outright.
            gate = check_delivery_gate(
                {"classifier": r.get("elongation_classifier_validated"), "operating_point": row_op})
            if not gate.ok:
                raise HTTPException(
                    400,
                    "phenology delivery requires a validated classifier AND count operating point in "
                    "every row (elongation_classifier_validated + operating_point_validated a shippable "
                    "reference); produce it via compute_phenology, which reads the on-disk operating "
                    f"point, gates, and stamps it. Unvalidated: {list(gate.unvalidated)}.",
                )
    else:
        # Non-phenology delivery: a per-plant phenotype row (the delivery schema — a measured
        # trait_name + value) is equally a delivery door, so gate it against its stamped validity.
        # Scoped to phenotype rows (trait_name + value) so diagnostic / inventory tables — which
        # carry neither — still export freely (over-correction guard).
        pheno_rows = [r for r in payload.rows if "trait_name" in r and "value" in r]
        for r in pheno_rows:
            gate = check_delivery_gate({"measurement": r.get("measurement_validated")})
            if not gate.ok:
                raise HTTPException(
                    400,
                    "per-plant phenotype delivery requires a validated measurement in every row "
                    "(measurement_validated a shippable reference — held-out GT or a breeder-confirmed "
                    "sample); produce the CSV via export_aggregated_csv, which gates and stamps it. "
                    f"Unvalidated: {list(gate.unvalidated)}.",
                )
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
    _guard(project_path)
    from tcip_mcp.tools.model_tools import list_registered_models

    return list_registered_models(project_path, tag)
