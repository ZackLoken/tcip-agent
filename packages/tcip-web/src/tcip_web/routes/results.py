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


class BloomPayload(BaseModel):
    """The inputs a bloom measurement is computed FROM — never the measurement itself.

    Every Results door takes this shape. Rounds 2-5 accepted a caller-composed ``rows`` table here
    and tried to decide what it MEANT: four inference predicates over row shape were defeated (the
    caller controls the column names), and round 5's ``export_kind`` declaration was defeated too
    (the caller controls the declaration). Both failed for one reason — the server was classifying
    data it did not produce, and that information is not in the payload. A table with a ``ratio``
    column is a bloom curve or an unrelated QC table depending on where it came from, which is
    exactly what a caller-supplied payload erased. So no door accepts rows: they accept a request to
    COMPUTE rows, and the server knows what it produced because it produced it.
    """

    project_root: str
    mapping_path: str  # .tcip/state/plant_mapping.json or equivalent
    # map date → predictions directory for that date. A trait's positive class id is resolved
    # server-side from each bucket's own recorded id_map (K6 #7/#3/#4/#6) — a client-supplied
    # class id is never honored, closing the bypass a caller-chosen id would otherwise open.
    predictions_by_date: dict[str, str]
    trait: str
    # Show provisional numbers on screen rather than refusing outright — the same escape
    # ``compute_phenology`` offers, so a breeder whose operating point is not yet calibrated can see
    # what they have instead of a dead end. It never applies to a file leaving the platform.
    acknowledge_unvalidated: bool = False


class _Bloom:
    """One trait's per-plant bloom measurement plus the on-disk evidence that qualifies it."""

    def __init__(self, spec, plants: dict, validity: dict, gate, positive_class_id) -> None:
        self.spec, self.plants, self.validity, self.gate = spec, plants, validity, gate
        self.positive_class_id = positive_class_id

    @property
    def elongation_classified(self) -> bool:
        """Whether the trait's positive-class axis was assessed at all.

        Requires the bucket-level fact ``compute_phenology`` refuses on — some bucket's recorded
        ``id_map`` actually contains the trait's positive class — as well as a fully-classified date.
        ``per_plant_phenology``'s flag alone reads True for a date with zero detections even in a
        bucket that never had the axis, because zero detections are trivially "all classified".
        """
        return self.positive_class_id is not None and bool(self.plants["elongation_classified"])

    def curve_rows(self) -> list[dict]:
        """Per-(plant, date) rows — the milestone rows' own series, not a second aggregation."""
        return [
            {"plant_id": row["plant_id"], "accession": row["accession"], **point}
            for row in self.plants["rows"] for point in row["series"]
        ]

    def milestone_rows(self) -> list[dict]:
        return [{k: v for k, v in row.items() if k != "series"} for row in self.plants["rows"]]


def _bloom(payload: BloomPayload) -> _Bloom:
    """Compute a trait's bloom measurement and reconcile the evidence behind it — one producer.

    Every Results door routes through here, so the numbers and the validity that qualifies them are
    read in one place from the buckets' own sidecars. Before this, only the CSV door reconciled
    anything: the curve and milestone doors returned the same phenotype with no gate at all, so the
    breeder read unvalidated bloom dates on screen and met the refusal only on clicking Download.
    Rows come from the canonical ``per_plant_phenology`` — the same function ``compute_phenology``
    delivers from — rather than a second aggregation loop, so the two surfaces cannot diverge.
    """
    from tcip_mcp.pipelines.resolution import (
        bind_classifier_validity,
        check_delivery_gate,
        reconcile_classifier_validity,
        reconcile_operating_point_validity,
    )
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

    pred_dirs = list(payload.predictions_by_date.values())
    recon = reconcile_operating_point_validity(pred_dirs)
    classifier_recon = reconcile_classifier_validity(pred_dirs)
    # The same binding compute_phenology applies, from the same shared owner rather than a second
    # copy: a classifier stamp calibrated for another trait or against a run that did not produce
    # these predictions does not validate THIS delivery. Without it the web door accepted a stamp
    # the MCP door rejects — and this route writes that stamp into the delivered CSV.
    classifier_state, binding_note = bind_classifier_validity(
        classifier_recon["validated"], pred_dirs, pred_dirs, trait=payload.trait,
    )
    validity = {
        "operating_point": recon["validated"],
        "classifier": classifier_state,
        "operating_point_conf": recon["conf"],
        "missing_operating_point_sidecars": recon["missing_sidecars"],
        "unvalidated_buckets": recon["unvalidated_buckets"],
        "missing_classifier_sidecars": classifier_recon["missing_sidecars"],
        "classifier_binding_note": binding_note,
    }
    gate = check_delivery_gate(
        {"classifier": classifier_state, "operating_point": recon["validated"]},
        acknowledge_unvalidated=payload.acknowledge_unvalidated,
    )
    plants = phenology.per_plant_phenology(
        mapping_raw, payload.predictions_by_date,
        positive_class_name=spec.positive_class_name, spec=spec,
    )
    positive_class_id, _msg = phenology.resolve_positive_class_id(spec, payload.predictions_by_date)
    return _Bloom(spec, plants, validity, gate, positive_class_id)


def _refusal(bloom: _Bloom) -> str:
    return (
        "bloom delivery requires a validated classifier AND count operating point, reconciled from "
        "the prediction buckets' own sidecars (never a caller-asserted string). Unvalidated: "
        f"{list(bloom.gate.unvalidated)} (operating_point="
        f"{bloom.validity['operating_point']!r}, classifier={bloom.validity['classifier']!r}; "
        f"missing operating_point.json: {bloom.validity['missing_operating_point_sidecars']}; "
        f"unvalidated buckets: {bloom.validity['unvalidated_buckets']}; missing "
        f"classifier_operating_point.json: {bloom.validity['missing_classifier_sidecars']}). "
        "Produce the predictions via a calibrated export_predictions and calibrate the classifier "
        "via calibrate_classifier_operating_point."
        + (f" {bloom.validity['classifier_binding_note']}"
           if bloom.validity["classifier_binding_note"] else "")
    )


def _disclosure(bloom: _Bloom) -> dict:
    """What qualifies these numbers, returned beside them so no surface can render them bare."""
    return {
        "validated": bloom.gate.stamp,
        # True whenever a dimension lacked on-disk evidence — including when the caller acknowledged
        # it, which is exactly when a surface must not render these numbers as valid.
        "provisional": bool(bloom.gate.unvalidated),
        "validity_detail": bloom.validity,
        # Honest signal: was anything actually classified along the trait's axis? If false, the
        # ratios are not a valid bloom measurement — do not deliver curves from them.
        "elongation_classified": bloom.elongation_classified,
    }


@router.post("/per_plant_curves")
def per_plant_curves(payload: BloomPayload) -> dict:
    """Per-(plant, date) positive-fraction curve from CLASSIFIED predictions.

    Gated on the same reconciled evidence as the CSV door (see ``_bloom``): a curve IS the delivered
    bloom measurement, just un-summarised, so it is refused on unvalidated evidence unless the caller
    explicitly acknowledges — in which case it ships marked provisional rather than bare.
    """
    bloom = _bloom(payload)
    if not bloom.gate.ok:
        raise HTTPException(400, _refusal(bloom))
    return {
        "rows": bloom.curve_rows(),
        "n_plants": len(bloom.plants["rows"]),
        "positive_class_id": bloom.positive_class_id,
        **_disclosure(bloom),
    }


# ── Milestone dates + CSV export ───────────────────────────────────────


@router.post("/onset_dates")
def onset_dates(payload: BloomPayload) -> dict:
    """Each plant's phenology milestones, computed from the buckets rather than from caller rows.

    Takes the same inputs as ``per_plant_curves`` (it used to accept the curve rows a client handed
    back, which made the milestone dates a function of a caller-supplied table). Both doors now
    project one ``per_plant_phenology`` result, so a milestone date and the curve it was read off
    can never come from different numbers.
    """
    bloom = _bloom(payload)
    if not bloom.gate.ok:
        raise HTTPException(400, _refusal(bloom))
    return {"rows": bloom.milestone_rows(), **_disclosure(bloom)}


class ExportCsvPayload(BloomPayload):
    # WHICH server computation to export — a choice of producer, never a claim about what the rows
    # mean or whether they are valid. Picking the "wrong" one yields a correctly-gated CSV of the
    # other thing, so there is nothing here for a caller to defeat.
    payload: Literal["curves", "milestones"]
    filename: str = "catkin_phenology.csv"


@router.post("/export_csv")
def export_csv(payload: ExportCsvPayload) -> Response:
    """Write the CSV for a bloom measurement this route computes itself.

    The gate is unconditional because there is no longer anything to branch on: this door computes
    the rows from the buckets (``_bloom``) instead of accepting a caller-composed table, so the only
    question left is whether the evidence on disk supports delivering them. ``acknowledge_unvalidated``
    is deliberately ignored here — it lets the breeder LOOK at provisional numbers on screen, never
    write them to a file that leaves the platform without its evidence.

    Milestone rows are written in the canonical ``phenology_csv_columns`` schema, so a web-delivered
    CSV and the MCP door's ``write_phenology_csv`` no longer disagree about what a phenology
    delivery's columns are.
    """
    bloom = _bloom(payload)
    if bloom.gate.unvalidated:
        raise HTTPException(400, _refusal(bloom))
    # The same refusal compute_phenology makes: if no bucket anywhere ever assessed the trait's
    # positive-class axis, the fraction is not a measurement and there is nothing valid to deliver.
    if not bloom.elongation_classified:
        raise HTTPException(
            400,
            f"predictions carry no {bloom.spec.positive_class_name!r} class anywhere in this "
            "delivery — the classifier that produced them never assessed this trait's positive "
            "class, so the positive fraction is not a valid measurement. Run and validate the "
            "classifier first.",
        )

    # Stamp the provenance the canonical schemas declare, from the same reconciliation the gate just
    # used — a delivered phenotype must name the operating point and the checkpoint behind it, and a
    # column a schema declares but nothing fills is the phantom this round removed elsewhere. Both
    # payloads are the same measurement, so both carry the same chain. Uses phenology_tools' own
    # resolver rather than re-reading the sidecars here.
    from tcip_mcp.tools.phenology_tools import _resolve_producer_identity

    producer = _resolve_producer_identity(payload.predictions_by_date)
    stamp = {
        "operating_point_conf": bloom.validity["operating_point_conf"],
        "operating_point_validated": bloom.gate.stamp["operating_point"],
        "positive_state_classifier_validated": bloom.gate.stamp["classifier"],
        "producer_model_sha256": producer.get("sha256"),
        "producer_experiment_id": producer.get("experiment_id"),
    }
    if payload.payload == "milestones":
        if bloom.spec.majority_milestone:
            stamp[f"{bloom.spec.phenology_prefix}_{bloom.spec.majority_label}_provisional"] = (
                "true" if bloom.spec.majority_provisional else "false")
        rows = [{**row, **stamp} for row in bloom.milestone_rows()]
        keys = phenology.phenology_csv_columns(bloom.spec)
    else:
        rows = [{**row, **stamp} for row in bloom.curve_rows()]
        keys = phenology.curve_csv_columns()
    if not rows:
        raise HTTPException(400, "no rows to export")

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
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
