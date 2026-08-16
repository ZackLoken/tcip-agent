"""Results routes: plant-mapping, per-plant phenology curves, CSV export.

The delivered target is a per-plant CSV of a registered trait's own milestone-date columns
(``<phenology_prefix>_<NN>per_date`` for each milestone its ``TraitSpec`` declares; every
registered trait has its own prefix and milestone set, resolved from the spec, with no code
change here). That pipeline looks like:

    predictions(date) -> per-plant detections of the trait's object (via plant mapping)
                       -> classify each detection positive vs not (validated classifier)
                       -> positive fraction / total per (plant, date)
                       -> find the dates that fraction crosses each declared milestone

The positive-state fraction is the share of a plant's detections of the trait's object that are
in its positive/measured state: an expert-defined morphological stage from a validated classifier, never a
geometric proxy such as bbox height (see the ``phenology`` skill + the CLAUDE.md
measurement-integrity invariant). The milestone math lives once in
``tcip_mcp...postprocessing.phenology``; this module is the HTTP surface the Results tab calls
and delegates to it.

This module also serves the operationalization record: what a trait's delivered number means, who
recorded it, and the breeder's confirmation of it. The confirmation is written here and nowhere
else, because an agent able to write one would be confirming its own definition of the measurement.

The backend owns everything except the model inference (which is driven by the Inference tab).
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

from tcip_web.paths import assert_path_allowed, assert_project_root_allowed

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


def _guarded_project_root(project_root: str) -> Path:
    """Confine a request's project root and hand back the resolved path every later read uses.

    The returned path is the load-bearing half: a door that guards the raw string and then
    resolves that string again reopens exactly what the guard closed.
    """
    try:
        return assert_project_root_allowed(project_root)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc


def _audit(project_root: str, tool: str, arguments: dict, **extra: object) -> None:
    """Record a GUI results mutation in the project's audit log.

    A delivery and the plant mapping behind it are project state, not dataset state: the
    dataset can be read by more than one project, but the export is this project's own
    outward action. ``extra`` carries a fact the entry shape has no home for, such as the
    person an action is recorded as coming from. Never fails the request.
    """
    if not project_root:
        return
    from tcip_mcp.audit import record_event

    record_event(tool, arguments, source="gui", scope=project_root, **extra)


def _project_root_of_state_path(path: str) -> Optional[str]:
    """The project root owning a ``<project_root>/.tcip/...`` path, or ``None`` if it isn't one.

    Mirrors ``dataset_layout.dataset_root_of``, anchored on the ``.tcip`` marker directory itself
    (the one shared platform state dir, see root CLAUDE.md) rather than a fixed depth under it.
    """
    parts = Path(path).parts
    idxs = [k for k, p in enumerate(parts) if p == ".tcip"]
    if not idxs:
        return None
    i = idxs[-1]
    return str(Path(*parts[:i])) if i > 0 else None


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
        root = _project_root_of_state_path(payload.persist_path)
        if root:
            _audit(
                root,
                "gui_build_plant_mapping",
                {
                    "persist_path": payload.persist_path,
                    "images_root": payload.images_root,
                    "n_dates": len(mapping),
                },
            )

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


class PhenologyPayload(BaseModel):
    """The inputs a phenology measurement is computed from: never the measurement itself.

    Every Results door takes this shape. A caller-composed ``rows`` table was tried here before,
    with the server inferring what it meant (inference predicates over row shape, an
    ``export_kind`` declaration), but both were defeated because the caller controls the column
    names and the declaration. The real problem: the server was classifying data it did not
    produce, and that information is not in the payload. A table with a ``ratio`` column is a
    phenology curve or an unrelated QC table depending on where it came from, which is exactly what a
    caller-supplied payload erases. So no door accepts rows: they accept a request to compute
    rows, and the server knows what it produced because it produced it.
    """

    project_root: str
    mapping_path: str  # .tcip/state/plant_mapping.json or equivalent
    # map date → predictions directory for that date. A trait's positive class id is resolved
    # server-side from each bucket's own recorded id_map: a client-supplied
    # class id is never honored, closing the bypass a caller-chosen id would otherwise open.
    predictions_by_date: dict[str, str]
    trait: str
    # Show provisional numbers on screen rather than refusing outright: the same escape
    # ``compute_phenology`` offers, so a breeder whose operating point is not yet calibrated can see
    # what they have instead of a dead end. It never applies to a file leaving the platform.
    acknowledge_unvalidated: bool = False


class _PhenologyMeasurement:
    """One trait's per-plant phenology measurement plus the on-disk evidence that qualifies it."""

    def __init__(
        self, spec, plants: dict, validity: dict, gate, positive_class_id, project_root: Path
    ) -> None:
        self.spec, self.plants, self.validity, self.gate = spec, plants, validity, gate
        self.positive_class_id = positive_class_id
        # The guarded, resolved root every later write and audit entry resolves from.
        self.project_root = project_root

    @property
    def positive_class_assessed(self) -> bool:
        """Whether the trait's positive-class axis was assessed at all.

        Requires the bucket-level fact ``compute_phenology`` refuses on: some bucket's recorded
        ``id_map`` actually contains the trait's positive class, as well as a fully-classified date.
        ``per_plant_phenology``'s flag alone reads True for a date with zero detections even in a
        bucket that never had the axis, because zero detections are trivially "all classified".
        """
        return self.positive_class_id is not None and bool(self.plants["positive_class_assessed"])

    def curve_rows(self) -> list[dict]:
        """Per-(plant, date) rows: the milestone rows' own series, not a second aggregation."""
        return [
            {"plant_id": row["plant_id"], "accession": row["accession"], **point}
            for row in self.plants["rows"] for point in row["series"]
        ]

    def milestone_rows(self) -> list[dict]:
        return [{k: v for k, v in row.items() if k != "series"} for row in self.plants["rows"]]


def _measure_phenology(payload: PhenologyPayload) -> _PhenologyMeasurement:
    """Compute a trait's phenology measurement and reconcile the evidence behind it: one producer.

    Every Results door routes through here, so the numbers and the validity that qualifies them are
    read in one place from the buckets' own sidecars: the curve and milestone doors share the same
    gate as CSV, so a breeder never sees an unvalidated phenology date on screen only to have the
    refusal surface later on Download. Rows come from the canonical ``per_plant_phenology`` (the same
    function ``compute_phenology`` delivers from) rather than a second aggregation loop, so the two
    surfaces cannot diverge.
    """
    from tcip_mcp.pipelines.resolution import (
        bind_classifier_validity,
        check_delivery_gate,
        reconcile_classifier_validity,
        reconcile_operating_point_validity,
        reconcile_tile_size_validity,
    )
    from tcip_mcp.traits import TraitUnknownError, get_trait

    root = _guarded_project_root(payload.project_root)
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
    # these predictions does not validate this delivery. Without it the web door accepted a stamp
    # the MCP door rejects, and this route writes that stamp into the delivered CSV.
    classifier_state, binding_note = bind_classifier_validity(
        classifier_recon["validated"], pred_dirs, pred_dirs, trait=payload.trait,
    )
    # The tile scale the counts were produced at is the same count operating point's other gating
    # dimension: a curve is built from per-image counts, and a tile edge with no persisted or
    # caller-stated basis moves those counts as surely as an uncalibrated conf does. Only operative
    # for buckets that actually ran tiled, so an untiled delivery is never refused over it.
    tile_recon = reconcile_tile_size_validity(pred_dirs)
    validity = {
        "operating_point": recon["validated"],
        "classifier": classifier_state,
        "operating_point_conf": recon["conf"],
        "missing_operating_point_sidecars": recon["missing_sidecars"],
        "unvalidated_buckets": recon["unvalidated_buckets"],
        "missing_classifier_sidecars": classifier_recon["missing_sidecars"],
        "classifier_binding_note": binding_note,
        "tile_size": tile_recon["validated"],
        "unvalidated_tile_size_buckets": tile_recon["unvalidated_buckets"],
    }
    flags = {"classifier": classifier_state, "operating_point": recon["validated"]}
    if tile_recon["operative"]:
        flags["tile_size"] = tile_recon["validated"]
    gate = check_delivery_gate(flags, acknowledge_unvalidated=payload.acknowledge_unvalidated)
    plants = phenology.per_plant_phenology(
        mapping_raw, payload.predictions_by_date,
        positive_class_name=spec.positive_class_name, spec=spec,
    )
    positive_class_id, _msg = phenology.resolve_positive_class_id(spec, payload.predictions_by_date)
    return _PhenologyMeasurement(spec, plants, validity, gate, positive_class_id, root)


def _refusal(measurement: _PhenologyMeasurement) -> str:
    tile_note = ""
    if measurement.validity["unvalidated_tile_size_buckets"]:
        tile_note = (
            f" Tiled bucket(s) {measurement.validity['unvalidated_tile_size_buckets']} carry a "
            "tile_size with no persisted training geometry and no explicit caller override, so the "
            "scale their counts were produced at has no basis. Re-export with an explicit tile_size, "
            "or from a checkpoint whose training tile geometry was persisted."
        )
    return (
        "phenology delivery requires a validated classifier and count operating point, reconciled from "
        "the prediction buckets' own sidecars (never a caller-asserted string). Unvalidated: "
        f"{list(measurement.gate.unvalidated)} (operating_point="
        f"{measurement.validity['operating_point']!r}, classifier={measurement.validity['classifier']!r}, "
        f"tile_size={measurement.validity['tile_size']!r}; "
        f"missing operating_point.json: {measurement.validity['missing_operating_point_sidecars']}; "
        f"unvalidated buckets: {measurement.validity['unvalidated_buckets']}; missing "
        f"classifier_operating_point.json: {measurement.validity['missing_classifier_sidecars']}). "
        "Produce the predictions via a calibrated export_predictions and calibrate the classifier "
        "via calibrate_classifier_operating_point."
        + tile_note
        + (f" {measurement.validity['classifier_binding_note']}"
           if measurement.validity["classifier_binding_note"] else "")
    )


def _disclosure(measurement: _PhenologyMeasurement) -> dict:
    """What qualifies these numbers, returned beside them so no surface can render them bare."""
    return {
        "validated": measurement.gate.stamp,
        # True whenever a dimension lacked on-disk evidence: including when the caller acknowledged
        # it, which is exactly when a surface must not render these numbers as valid.
        "provisional": bool(measurement.gate.unvalidated),
        "validity_detail": measurement.validity,
        # Honest signal: was anything actually classified along the trait's axis? If false, the
        # ratios are not a valid phenology measurement: do not deliver curves from them.
        "positive_class_assessed": measurement.positive_class_assessed,
    }


@router.post("/per_plant_curves")
def per_plant_curves(payload: PhenologyPayload) -> dict:
    """Per-(plant, date) positive-fraction curve from classified predictions.

    Gated on the same reconciled evidence as the CSV door (see ``_measure_phenology``): a curve IS the delivered
    phenology measurement, just un-summarised, so it is refused on unvalidated evidence unless the caller
    explicitly acknowledges, in which case it ships marked provisional rather than bare.
    """
    measurement = _measure_phenology(payload)
    if not measurement.gate.ok:
        raise HTTPException(400, _refusal(measurement))
    return {
        "rows": measurement.curve_rows(),
        "n_plants": len(measurement.plants["rows"]),
        "positive_class_id": measurement.positive_class_id,
        **_disclosure(measurement),
    }


# ── Milestone dates + CSV export ───────────────────────────────────────


@router.post("/onset_dates")
def onset_dates(payload: PhenologyPayload) -> dict:
    """Each plant's phenology milestones, computed from the buckets rather than from caller rows.

    Takes the same inputs as ``per_plant_curves``: both doors project one ``per_plant_phenology``
    result rather than accepting a caller-composed table, so a milestone date and the curve it was
    read off can never come from different numbers.
    """
    measurement = _measure_phenology(payload)
    if not measurement.gate.ok:
        raise HTTPException(400, _refusal(measurement))
    return {"rows": measurement.milestone_rows(), **_disclosure(measurement)}


class ExportCsvPayload(PhenologyPayload):
    # Which server computation to export: a choice of producer, never a claim about what the rows
    # mean or whether they are valid. Picking the "wrong" one yields a correctly-gated CSV of the
    # other thing, so there is nothing here for a caller to defeat.
    payload: Literal["curves", "milestones"]
    filename: Optional[str] = None


@router.post("/export_csv")
def export_csv(payload: ExportCsvPayload) -> Response:
    """Write the CSV for a phenology measurement this route computes itself.

    The gate is unconditional because there is nothing else to branch on: this door computes
    the rows from the buckets (``_measure_phenology``) instead of accepting a caller-composed table, so the only
    question left is whether the evidence on disk supports delivering them. ``acknowledge_unvalidated``
    is deliberately ignored here: it lets the breeder look at provisional numbers on screen, never
    write them to a file that leaves the platform without its evidence.

    Milestone rows are written in the canonical ``phenology_csv_columns`` schema, so a web-delivered
    CSV and the MCP door's ``write_phenology_csv`` cannot disagree about what a phenology
    delivery's columns are.
    """
    measurement = _measure_phenology(payload)
    if measurement.gate.unvalidated:
        raise HTTPException(400, _refusal(measurement))
    # The same refusal compute_phenology makes: if no bucket anywhere ever assessed the trait's
    # positive-class axis, the fraction is not a measurement and there is nothing valid to deliver.
    if not measurement.positive_class_assessed:
        raise HTTPException(
            400,
            f"predictions carry no {measurement.spec.positive_class_name!r} class anywhere in this "
            "delivery. The classifier that produced them never assessed this trait's positive "
            "class, so the positive fraction is not a valid measurement. Run and validate the "
            "classifier first.",
        )

    # Stamp the provenance the canonical schemas declare, from the same reconciliation the gate just
    # used: a delivered phenotype must name the operating point and the checkpoint behind it, and a
    # column a schema declares but nothing fills is a phantom that must not reach the delivered CSV.
    # Both payloads are the same measurement, so both carry the same chain. Uses phenology_tools'
    # own resolver rather than re-reading the sidecars here.
    from tcip_mcp.tools.phenology_tools import _resolve_producer_identity

    producer = _resolve_producer_identity(payload.predictions_by_date)
    stamp = {
        "operating_point_conf": measurement.validity["operating_point_conf"],
        "operating_point_validated": measurement.gate.stamp["operating_point"],
        "positive_state_classifier_validated": measurement.gate.stamp["classifier"],
        "producer_model_sha256": producer.get("sha256"),
        "producer_experiment_id": producer.get("experiment_id"),
    }
    if payload.payload == "milestones":
        provisional_column = phenology.majority_provisional_column(measurement.spec)
        if provisional_column:
            stamp[provisional_column] = (
                "true" if measurement.spec.majority_provisional else "false")
        rows = [{**row, **stamp} for row in measurement.milestone_rows()]
        keys = phenology.phenology_csv_columns(measurement.spec)
    else:
        rows = [{**row, **stamp} for row in measurement.curve_rows()]
        keys = phenology.curve_csv_columns()
    if not rows:
        raise HTTPException(400, "no rows to export")

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    body = buf.getvalue()
    filename = payload.filename or f"{payload.trait}_{payload.payload}.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

    # The browser download lands wherever the breeder's browser puts it; the delivery itself
    # belongs to the project, so the same bytes are written to <project>/results_export/, audited.
    saved_path = measurement.project_root / "results_export" / Path(filename).name
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path.write_text(body, encoding="utf-8", newline="")
    _audit(str(measurement.project_root), "results.export_csv", {
        "trait": payload.trait, "payload": payload.payload, "saved_path": str(saved_path),
        "rows": len(rows),
    })
    headers["X-TCIP-Saved-To"] = str(saved_path)
    return Response(content=body, media_type="text/csv", headers=headers)


# ── What a delivered number means: the record, and the breeder's confirmation ──


class ConfirmOperationalizationPayload(BaseModel):
    """One breeder confirmation, or one withdrawal, for one trait's one delivery kind.

    ``record_seen`` is the content hash of the record the surface rendered. Confirming is an act
    over what the breeder read, so a record rewritten while the panel was open refuses rather than
    taking the click for text nobody displayed. ``user`` is the name the surface carries; when it is
    absent the backend falls back to its own process identity and records that it did.
    """

    project_root: str
    trait: str
    delivery_kind: str
    record_seen: str
    user: Optional[str] = None
    confirmed: bool = True


def _operationalization_body(project_root: Path, trait: str, delivery_kind: str) -> dict:
    """One record as the confirming surface reads it: what is stated, what covers it, what moved.

    ``confirmed_current`` and ``superseded`` come from the same ``check_operationalization`` the
    delivery doors run, so the panel and the door cannot disagree about whether a confirmation still
    holds, and no surface re-derives that comparison for itself. ``delivers`` quotes the crop
    vocabulary's own wording, so the breeder reads their definition before the agent's statement.
    Nothing stated yet reads as null statement fields with ``confirmed_current`` false.
    """
    from tcip_mcp import operationalization as op
    from tcip_mcp.traits import crops_definitions

    spec, record, _specs_dir = op.resolve_trait_and_record(
        trait, delivery_kind, project_root=project_root
    )
    check = op.check_operationalization(spec, record, delivery_kind)
    stated = record.value or {}
    definitions = crops_definitions()
    return {
        "trait": trait,
        "delivery_kind": delivery_kind,
        **{field: stated.get(field) for field in op.STATEMENT_FIELDS},
        "confirmed_by": stated.get("confirmed_by"),
        "confirmed_at": stated.get("confirmed_at"),
        "identity_from_request": stated.get("identity_from_request"),
        "confirmed_current": check.ok,
        "superseded": [dict(entry) for entry in check.superseded],
        "delivers": [{"name": name, "definition": definitions.get(name)} for name in spec.delivers],
        "record_seen": op.record_seen_hash(stated),
    }


@router.get("/operationalization")
def get_operationalization(project_root: str, trait: str, delivery_kind: str) -> dict:
    """What this trait's delivered number is recorded to mean, and whether it is confirmed now.

    The panel renders every field this returns and posts ``record_seen`` back with the confirmation,
    so what the breeder authorizes is what they were shown.
    """
    from tcip_mcp.traits import TraitUnknownError

    root = _guarded_project_root(project_root)
    try:
        return _operationalization_body(root, trait, delivery_kind)
    except (TraitUnknownError, ValueError) as e:
        raise HTTPException(400, str(e)) from e


@router.get("/operationalizations")
def list_operationalizations(project_root: str) -> dict:
    """Every operationalization record this project holds, one row per trait and delivery kind.

    The panel enumerates rather than selects: the record's key is a pair, so a surface that knew
    only a trait could not address one, and a kind selector would hardcode today's kinds into the
    browser. Rows for a kind the Results tab cannot compute are still listed, because a count or an
    aggregate record is confirmable whether or not this tab renders that delivery.

    ``unresolved`` names a record whose trait is no longer registered, or whose delivery kind is not
    one this platform declares, rather than dropping it: an unlistable record and no record at all
    would otherwise read identically.
    """
    import tcip_store as ts
    from tcip_mcp import operationalization as op
    from tcip_mcp.traits import TraitUnknownError

    root = _guarded_project_root(project_root)
    records: list[dict] = []
    unresolved: list[dict] = []
    for key in ts.keys(op.OPERATIONALIZATIONS_STORE, str(op.operationalizations_scope(root))):
        trait, delivery_kind = key.parts
        try:
            records.append(_operationalization_body(root, trait, delivery_kind))
        except (TraitUnknownError, ValueError) as e:
            unresolved.append(
                {"trait": trait, "delivery_kind": delivery_kind, "reason": str(e)}
            )
    return {"records": records, "unresolved": unresolved}


@router.post("/operationalization/confirm")
def confirm_operationalization(payload: ConfirmOperationalizationPayload) -> dict:
    """Record the breeder's confirmation of what is on file, or withdraw one they gave before.

    The one path a confirmation is ever written by. No MCP tool reaches this writer: an agent that
    could confirm its own statement would be confirming its own definition of the measurement.

    Refused with 400 when nothing is stated for this trait and kind, or the trait is not registered
    for this project, and with 409 when the record moved since the surface read it, the body then
    carrying what is on file so the panel re-renders that and the breeder confirms what they see.
    ``confirmed`` false withdraws, clearing the four confirmation fields and leaving the statement.

    Nothing verifies that the person at the keyboard is the person named. What this records is a
    name the request supplied, whether the request supplied one at all, and an audit entry saying
    the act happened; it is not authentication and nothing here may be read as one.
    """
    from tcip_mcp import operationalization as op
    from tcip_mcp.traits import TraitUnknownError

    from tcip_web.identity import resolve_user, user_id

    root = _guarded_project_root(payload.project_root)
    # The writer applies the user: convention to whatever name it is given, so it is passed bare.
    actor = resolve_user(payload.user)
    identity_from_request = bool((payload.user or "").strip())
    try:
        record = op.confirm_trait_operationalization(
            root,
            payload.trait,
            payload.delivery_kind,
            user=actor,
            record_seen=payload.record_seen,
            identity_from_request=identity_from_request,
            confirmed=payload.confirmed,
        )
    except op.RecordMoved as e:
        raise HTTPException(
            409,
            {
                "message": str(e),
                "record": _operationalization_body(root, payload.trait, payload.delivery_kind),
            },
        ) from e
    except (TraitUnknownError, ValueError) as e:
        raise HTTPException(400, str(e)) from e

    _audit(
        str(root),
        "results.confirm_trait_operationalization",
        {
            "trait": payload.trait,
            "delivery_kind": payload.delivery_kind,
            "confirmed": payload.confirmed,
            "identity_from_request": identity_from_request,
        },
        user=user_id(actor),
    )
    return {field: record[field] for field in op.CONFIRMATION_FIELDS}


# ── Registered traits (drives the Results tab's trait selection) ───────


@router.get("/traits")
def list_traits(project_root: str) -> dict:
    """Traits registered for this project, so the Results tab resolves which trait it is computing
    for from the project's own registry instead of assuming one.

    ``milestone_fractions_by_trait`` carries each trait's declared milestone fractions verbatim
    rather than a derived category, so a surface can tell whether a trait has milestones to compute
    without the server deciding what that means for it.

    ``invalid_specs`` names every spec file under this project's registry that failed to load and
    why, so a breeder facing zero (or fewer than expected) traits can tell "nothing registered" from
    "something is registered but broken" instead of the two reading identically.
    """
    _guard(project_root)
    from tcip_mcp.traits import load_trait_specs_with_errors

    specs, errors = load_trait_specs_with_errors(project_root=project_root)
    return {
        "traits": sorted(spec.name for spec in specs),
        "milestone_fractions_by_trait": {
            spec.name: list(spec.milestone_fractions) for spec in specs
        },
        "invalid_specs": errors,
    }


# ── List registered models (used by Inference tab) ─────────────────────


@router.get("/models/registered")
def registered_models(project_path: str, tag: Optional[str] = None) -> dict:
    _guard(project_path)
    from tcip_mcp.tools.model_tools import list_registered_models

    return list_registered_models(project_path, tag)
