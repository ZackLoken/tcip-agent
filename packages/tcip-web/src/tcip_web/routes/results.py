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
from typing import Literal, Optional

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
    # map date → predictions directory for that date. A trait's positive class id is resolved
    # server-side from each bucket's own recorded id_map (K6 #7/#3/#4/#6) — a client-supplied
    # class id is never honored, closing the bypass a caller-chosen id would otherwise open.
    predictions_by_date: dict[str, str]
    trait: str


class PerPlantCurveRow(BaseModel):
    # Required = what onset_dates dereferences (malformed -> 422, not a KeyError/500); rest stay
    # optional. n_unclassified/n_missing are ALSO required (stage-6 review Finding E): the
    # plant_fully_classified predicate reads them directly, and a `= 0` default let an
    # under-specified payload silently read as "fully classified" — more permissive for missing
    # input than the "ratio is not None" predicate it replaced, the opposite of what a
    # measurement-integrity gate should do when its own evidence is absent.
    plant_id: str
    date: str
    n_unclassified: int
    n_missing: int
    ratio: Optional[float] = None
    accession: Optional[str] = None
    n_images: int = 0
    n_total: int = 0
    n_positive: int = 0


@router.post("/per_plant_curves")
def per_plant_curves(payload: PerPlantCurvesPayload) -> dict:
    """Per-(plant, date) positive-fraction curve from CLASSIFIED predictions.

    Delegates to the canonical ``phenology.per_plant_series`` (K6: this route used to
    reimplement the aggregation loop independently — the same coverage rule, resolution, and
    disclosure fields ``compute_phenology`` uses now apply here too, so the two surfaces can't
    silently diverge). If a bucket never classified along the trait's positive-class axis,
    ``elongation_classified`` is false and the ratios are not a valid bloom measurement (run +
    validate the classifier first — see the ``phenology`` skill).
    """
    from tcip_mcp.traits import TraitUnknownError, get_trait

    try:
        spec = get_trait(payload.trait)
    except TraitUnknownError as e:
        raise HTTPException(400, str(e)) from e

    _guard(payload.mapping_path, *payload.predictions_by_date.values())
    mapping = plant_mapping.load_mapping(Path(payload.mapping_path))
    if not mapping:
        raise HTTPException(404, f"no mapping at {payload.mapping_path}")
    mapping_raw = {date: [a.__dict__ for a in assignments] for date, assignments in mapping.items()}

    positive_class_id, _msg = phenology.resolve_positive_class_id(spec, payload.predictions_by_date)

    per_plant = phenology.per_plant_series(mapping_raw, payload.predictions_by_date,
                                           positive_class_name=spec.positive_class_name)
    rows: list[dict] = []
    any_classified = False
    for plant_id, info in per_plant.items():
        for date_str, total, positive, unclassified, missing in info["series"]:
            classified = unclassified == 0 and missing == 0
            any_classified = any_classified or (classified and total > 0)
            rows.append({
                "plant_id": plant_id, "accession": info["accession"], "date": date_str,
                "n_images": 1, "n_total": total, "n_positive": positive,
                "n_unclassified": unclassified, "n_missing": missing,
                "ratio": (positive / total if classified and total else None),
            })

    return {
        "rows": rows,
        "n_plants": len(per_plant),
        "positive_class_id": positive_class_id,
        # Honest signal: was anything actually classified along the trait's axis? If false, the
        # ratios above are not a valid bloom measurement — do not deliver curves from them.
        "elongation_classified": any_classified,
    }


# ── Onset-date extraction + CSV export ─────────────────────────────────


class OnsetDatesPayload(BaseModel):
    curves: list[PerPlantCurveRow]  # output of per_plant_curves
    trait: str


@router.post("/onset_dates")
def onset_dates(payload: OnsetDatesPayload) -> dict:
    """Compute each plant's phenology milestones from its positive-fraction curve.

    Delegates to the canonical ``phenology`` module so a milestone date means the same
    thing here and in the ``compute_phenology`` MCP tool. ``trait`` is required (K4/K5): the
    milestone column names and crossing fractions are the trait's own spec, never a silent
    catkin default for a different trait's data.
    """
    from tcip_mcp.traits import TraitUnknownError, get_trait

    try:
        spec = get_trait(payload.trait)
    except TraitUnknownError as e:
        raise HTTPException(400, str(e)) from e

    plants: dict[str, list[PerPlantCurveRow]] = {}
    for row in payload.curves:
        plants.setdefault(row.plant_id, []).append(row)

    out: list[dict] = []
    for plant_id, rows in plants.items():
        ordered = sorted(rows, key=lambda r: phenology.date_key(r.date))
        # A row with no ratio (unclassified/missing, or total==0 for that date) can't contribute a
        # real observation to the crossing series — excluded, not coerced to 0.0.
        series = [(r.date, float(r.ratio)) for r in ordered if r.ratio is not None]
        # The SAME "usable date" predicate phenology.per_plant_phenology gates milestones on —
        # unclassified==0 and missing==0 for every date, not "ratio is not None" (stage-6 review:
        # ratio is also None for a legitimate zero-detection date, e.g. before catkins emerge, so
        # the old predicate marked a perfectly valid plant "not fully classified" and delivered
        # null milestones for it, while this door's own n_dates_unclassified/n_dates_missing_images
        # both correctly read 0 — a plant the GUI could then render "valid" next to blank cells).
        usable = [r for r in rows if r.n_unclassified == 0 and r.n_missing == 0]
        plant_fully_classified = len(usable) == len(rows) and bool(rows)
        milestones = (phenology.plant_milestones(series, spec) if plant_fully_classified
                     else {col: None for col in phenology.milestone_date_columns(spec)})
        out.append({
            "plant_id": plant_id,
            "accession": rows[0].accession,
            "n_datapoints": len(rows),
            "n_dates_unclassified": sum(1 for r in rows if r.n_unclassified),
            "n_dates_missing_images": sum(1 for r in rows if r.n_missing),
            # Stage-6 review N6: a plant can be fully classified AND fully observed (0 unclassified,
            # 0 missing) while STILL having zero real detections on every date (before emergence, or
            # a genuinely empty scene) — "valid" alone doesn't distinguish that from real bloom data,
            # so the GUI needs this count to render "no observations" rather than a plain "valid"
            # next to blank milestone cells.
            "n_observed_dates": len(series),
            **milestones,
        })
    return {"rows": out}


class ExportCsvPayload(BaseModel):
    rows: list[dict]
    filename: str = "catkin_phenology.csv"
    # What this export IS — declared by the caller, not guessed from the rows' column names.
    # Required, no default: an omitted kind is a 422, never a silent "probably diagnostic".
    #
    # Three successive rounds of stage-6 review defeated three successive attempts to INFER this
    # from row shape (all-keys subset -> any-one-key intersection -> two-of-five -> identity-pair).
    # The approach cannot work: the caller controls the row shape, so every predicate is both
    # evadable (rename plant_id to plot_name, which is the platform's own field name for it, and a
    # real bloom curve ships ungated) and over-eager (an inventory table keyed by plant and date is
    # refused with a phenology error). Round 4 was defeated in BOTH directions in a single probe.
    # Declaring intent replaces a guess about caller intent with a statement of it; lying now takes
    # a deliberate false declaration rather than an incidental column name, and buys nothing —
    # validity is still reconciled from each bucket's own on-disk sidecar, never from the rows.
    export_kind: Literal["phenology", "diagnostic"]
    # For a phenology delivery, the prediction buckets the curve was computed from, so the gate
    # reads the count operating point's validity from each bucket's operating_point.json (the
    # on-disk floor) rather than any caller-supplied string.
    predictions_by_date: Optional[dict[str, str]] = None


@router.post("/export_csv")
def export_csv(payload: ExportCsvPayload) -> Response:
    """Export rows as CSV, gated on the caller's declared ``export_kind`` (see the payload model).

    Known residual (stage-6 review Finding H, still not fixed here — pre-existing, not introduced by
    this diff, but this diff is what made this door reconcile real validity, so the gap is worth
    recording rather than carrying forward silently): this door writes whatever keys the caller's
    rows carry (``extrasaction="ignore"`` below), so a web-delivered phenology CSV still ends up with
    a DIFFERENT schema than the MCP door's ``write_phenology_csv`` — it lacks that door's provenance
    stamp columns (``operating_point_conf``, ``producer_experiment_id``, etc., which nothing here
    adds to a web-built row). The gate correctly refuses an unvalidated delivery either way; the two
    doors' delivered *shapes* have simply diverged. Reconciling them (a shared column-shaping step,
    or stamping provenance onto web rows before export) is real follow-up work.

    An earlier version of this note also claimed the web rows carry ``gap_days`` columns the
    canonical writer drops. That was wrong on both halves and is corrected here (round-4 review):
    ``onset_dates`` emits no ``gap_days`` column at all — ``Crossing.gap_days`` is computed but never
    made a column by any producer — and the ``*_date_bound`` columns it does emit are now part of
    ``phenology_csv_columns`` too, so they no longer diverge.
    """
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
        reconcile_classifier_validity,
        reconcile_operating_point_validity,
    )
    from tcip_mcp.traits import registered_traits, get_trait

    # TRAP 1 (K5/K6): the milestone-column trigger is the UNION of milestone columns over EVERY
    # registered trait — structural, so no caller can lower it by tagging a row with a different
    # trait name (a client-supplied `trait` field must never decide WHETHER the gate fires, only how
    # a row is formatted once it has). An unregistered trait still has no columns to include — the
    # corresponding hole stays open until a spec is authored for it (an honest scope limit, not a
    # claim of full closure).
    #
    # This is retained as a FLOOR beneath the declared `export_kind`, not replaced by it: a row
    # carrying `catkin_95per_date` IS a delivered phenology milestone whatever the caller calls the
    # export, and these names come from the trait registry rather than from the payload. So a
    # `diagnostic` declaration cannot smuggle out milestone rows. What the declaration adds is the
    # case the floor cannot see — a per-plant-per-date CURVE, whose columns are generic enough
    # (`ratio`, `n_total`) that no column-name rule can separate it from an unrelated table.
    _milestones: set[str] = set()
    for _t in registered_traits():
        _milestones |= set(phenology.milestone_date_columns(get_trait(_t)))
    # K15 finding #2: a Curves CSV export (per_plant_curves' own row shape — ratio/n_total/
    # n_positive/n_unclassified/n_missing, no milestone columns) carries the identical underlying
    # measurement and must be gated too — milestone-key presence alone let it ship un-acknowledged
    # by construction. A subset check (ALL curve keys required) let a caller drop just one (e.g.
    # n_positive) to bypass it; a single-key intersection (stage-6 review round-2 fix) over-corrected
    # the other way — every one of these names is generic enough that an unrelated table (a
    # class-balance QC table with its own "n_positive" column, a split-ratio table with its own
    # "ratio" column) collides and gets refused with a phenology error (stage-6 review Finding
    # D/N9), and the round-3 and round-4 successors were each defeated in turn — round 4's
    # identity-pair rule simultaneously let four real curve shapes through (renaming `plant_id` to
    # the platform's own `plot_name`, or `date` to `observation_date`, was enough) and 400'd
    # legitimate per-plant-per-date inventory tables. The trigger is now the caller's own declared
    # `export_kind`, with the milestone-column floor above still applying regardless of it.
    if payload.export_kind == "phenology" or any(_milestones & set(r.keys()) for r in payload.rows):
        # Both dimensions reconcile from the same buckets' sidecars (never the row's own asserted
        # string) — the classifier's stamp lives in classifier_operating_point.json beside the same
        # bucket operating_point.json is in (K3). ResultsTab.tsx/inference.ts (Commit 4) now send
        # predictions_by_date on every call; without it this door still fails closed for lack of
        # evidence, never a caller-trust fallback.
        op_state: str | None = None
        classifier_state: str | None = None
        if payload.predictions_by_date:
            pred_dirs = list(payload.predictions_by_date.values())
            recon = reconcile_operating_point_validity(pred_dirs)
            op_state = recon["validated"]
            classifier_state = reconcile_classifier_validity(pred_dirs)["validated"]
            if op_state not in VALIDATED_SHIPPABLE:
                raise HTTPException(
                    400,
                    "phenology delivery: the count operating point is not validated on disk "
                    f"(reconciled from operating_point.json = {op_state!r}; missing sidecars: "
                    f"{recon['missing_sidecars']}; unvalidated buckets: {recon['unvalidated_buckets']}). "
                    "Produce the predictions via a calibrated export_predictions, then compute_phenology.",
                )
        for r in payload.rows:
            # The web door has no acknowledge escape — an unvalidated phenotype CSV is refused outright.
            gate = check_delivery_gate({"classifier": classifier_state, "operating_point": op_state})
            if not gate.ok:
                raise HTTPException(
                    400,
                    "phenology delivery requires a validated classifier AND count operating point, "
                    "reconciled from the prediction buckets' own sidecars (never a caller-asserted row "
                    "string). Pass predictions_by_date so both can be reconciled, and produce them via "
                    "calibrate_classifier_operating_point + a calibrated export_predictions, then "
                    f"compute_phenology. Unvalidated: {list(gate.unvalidated)}.",
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
