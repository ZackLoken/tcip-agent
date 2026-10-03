"""Results routes: plant-mapping, per-plant phenology curves, CSV export, and the traits with the
breeder's confirmation of a trait revision.

The delivered target is a per-plant CSV of a registered trait's own milestone-date columns
(``<phenology_prefix>_<NN>per_date`` for each milestone its ``TraitEntry`` declares; every
registered trait has its own prefix and milestone set, resolved from the spec). That pipeline looks
like:

    predictions(date) -> per-plant detections of the trait's object (via plant mapping)
                       -> classify each detection positive vs not (validated classifier)
                       -> positive fraction / total per (plant, date)
                       -> find the dates that fraction crosses each declared milestone

The positive-state fraction is the share of a plant's detections of the trait's object that are in
its positive/measured state, from a validated classifier. The milestone math lives in
``tcip_mcp.pipelines.postprocessing.phenology``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field
from tcip_mcp.pipelines.postprocessing import phenology, plant_mapping

from tcip_web.paths import resolved_path, within
from tcip_web.state import store

if TYPE_CHECKING:
    from tcip_mcp.pipelines.postprocessing.phenology import PhenologyMeasurement
    from tcip_mcp.traits import TraitRecord, TraitRevision

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/results", tags=["results"])


def _evidence_roots(root: Path) -> list[Path]:
    """The open project's own tree plus every dataset root registered to it; a registry that will
    not decode answers 400.
    """
    from tcip_store import DecodeError

    from tcip_mcp.tools.project_tools import dataset_entry_path, read_datasets

    try:
        entries = read_datasets(root)
    except DecodeError as exc:
        raise HTTPException(400, str(exc)) from exc
    return [root, *(dataset_entry_path(root, e) for e in entries)]


def _belonging(root: Path, *paths: Optional[str]) -> list[Optional[Path]]:
    """Confine evidence paths to the open project: its tree or a dataset registered to it.

    Resolved paths come back in the order given (``None`` for an omitted one), and every later read
    uses them. A path inside the managed allow-set but outside this project's own set refuses,
    naming the project and the roots it checked.
    """
    roots = _evidence_roots(root)
    out: list[Optional[Path]] = []
    for p in paths:
        if not p:
            out.append(None)
            continue
        resolved = resolved_path(p)
        if not any(within(resolved, r) for r in roots):
            raise HTTPException(
                403, f"{p} does not belong to project {root}: it is under neither the project "
                     f"nor a dataset registered to it ({', '.join(str(r) for r in roots)})")
        out.append(resolved)
    return out


# ── Plant mapping ──────────────────────────────────────────────────────


class BuildMappingPayload(BaseModel):
    name: str
    images_root: str
    plant_registry: str
    dates: Optional[list[str]] = None
    nn_tolerance_m: Optional[float] = None
    supersede: bool = False
    user: str


@router.post("/plant_mapping/build")
def build_plant_mapping(payload: BuildMappingPayload) -> dict:
    """Build and persist the open project's plant mapping
    (:func:`~tcip_mcp.pipelines.postprocessing.plant_mapping.build_plant_mapping`) from an
    ``images_root`` confined to the project, by the person ``user`` names, answering the build as
    ``MappingBuild.served`` states it.

    A refusal answers 400, an unknown registry 404, a rebuild a delivery event still cites 409
    unless ``supersede``, and a receipt that could not be written 409.
    """
    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_mcp.identity import actor
    from tcip_web.routes.audit_gap import audit_gap_409

    person = actor(payload.user)
    root = store.open_root()
    [images_root] = _belonging(root, payload.images_root)
    try:
        return plant_mapping.build_plant_mapping(
            root, payload.name, images_root or "", payload.plant_registry, dates=payload.dates,
            nn_tolerance_m=payload.nn_tolerance_m, supersede=payload.supersede,
            actor=person).served()
    except plant_mapping.PlantRegistryNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except AuditEntryNotWritten as exc:
        raise audit_gap_409(exc, None) from exc
    except (plant_mapping.MappingRebuildRefusal,
            plant_mapping.UngeoreferencedCaptureRefusal) as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class LoadMappingPayload(BaseModel):
    name: str


@router.post("/plant_mapping/load")
def load_plant_mapping(payload: LoadMappingPayload) -> dict:
    """The persisted mapping under ``payload.name`` as ``MappingBuild.served`` states it. Nothing
    stored under the name answers 404; a name the key refuses, or a record that will not load,
    409.
    """
    from tcip_store import StoreError

    root = store.open_root()
    try:
        build = plant_mapping.load_mapping(root, payload.name)
    except (StoreError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc
    if build is None:
        raise HTTPException(404, f"no plant mapping named {payload.name!r} under {root}")
    return build.served()


@router.get("/plant_mapping/list")
def list_plant_mappings() -> dict:
    """Every mapping name persisted under the open project."""
    root = store.open_root()
    return {"names": plant_mapping.plant_mapping_names(root)}


# ── Per-plant curves ───────────────────────────────────────────────────


class PhenologyInputs(BaseModel):
    """The inputs a phenology measurement is computed from, never the measurement itself: no
    Results door accepts rows.
    """

    model_config = ConfigDict(extra="forbid")

    mapping_name: str  # a name persisted under the open project's own plant-mapping store
    # The published bucket of each delivered date; each stands for the date its record states.
    buckets: list[str]
    trait: str
    # The population: the plant ids this measurement is for, one row each, never every mapped plot.
    plants: list[str]
    # Whether a plant missing any delivered date is left out of the milestones.
    require_all_dates_complete: bool = phenology.REQUIRE_ALL_DATES_COMPLETE


class PhenologyPayload(PhenologyInputs):
    """The screen route's payload: the measurement inputs plus a display choice."""

    # Show unvalidated numbers on screen rather than refusing outright. Never an acknowledgment.
    show_unvalidated: bool = False


def _measure(payload: PhenologyInputs) -> "PhenologyMeasurement":
    """:func:`~tcip_mcp.pipelines.postprocessing.phenology.measure_phenology` over the buckets this
    route confines to the open project; each of its refusals answers 400."""
    from tcip_mcp.operationalization import OperationalizationRefused
    from tcip_mcp.subject_registry import RegistryError
    from tcip_mcp.traits import TraitUnknownError

    root = store.open_root()
    resolved = _belonging(root, *payload.buckets)
    try:
        return phenology.measure_phenology(
            root, trait=payload.trait, mapping_name=payload.mapping_name,
            buckets=[p for p in resolved if p is not None], plants=payload.plants,
            require_all_dates_complete=payload.require_all_dates_complete)
    except OperationalizationRefused as exc:
        raise HTTPException(400, exc.as_detail()) from exc
    except plant_mapping.MappingDeliveryRefusal as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    except phenology.measurement_refusals() as exc:
        raise HTTPException(400, str(exc)) from exc
    except (TraitUnknownError, RegistryError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/phenology_measurement")
def phenology_measurement(payload: PhenologyPayload) -> dict:
    """Both phenology projections (the per-(plant, date) curve and the per-plant milestone dates)
    from one measurement, with the milestone columns the producer writes, cleared through the one
    delivery gate: its refusal answers 400 with the refusal's detail unless ``show_unvalidated``,
    in which case both are returned with the refusal as ``unvalidated_reason`` and, as
    ``result_sha256``, the digest of each projection's delivered result keyed ``curves`` and
    ``milestones``, which a breeder's acknowledgment of exporting it binds to.
    """
    from tcip_mcp.delivery import DeliveryRefused, gate
    from tcip_mcp.operationalization import OperationalizationRefused
    from tcip_mcp.traits import STATE_CROSSING_DATES

    measurement = _measure(payload)
    reason = None
    digests: dict[str, str] = {}
    for projection in ("curves", "milestones"):
        try:
            gate(store.open_root(), list(measurement.buckets.values()),
                 delivery_kind=STATE_CROSSING_DATES, revision=measurement.revision,
                 result=measurement.result(curves=projection == "curves"))
        except OperationalizationRefused as exc:
            raise HTTPException(400, exc.as_detail()) from exc
        except DeliveryRefused as exc:
            if not payload.show_unvalidated or exc.result_sha256 is None:
                raise HTTPException(400, exc.as_detail()) from exc
            reason, digests[projection] = str(exc), exc.result_sha256
    disclosure = measurement.plant_mapping
    return {
        "curves": {"rows": measurement.curve_rows(), "n_plants": len(measurement.rows)},
        "milestones": {"rows": measurement.milestone_rows(),
                       "columns": phenology.milestone_date_columns(measurement.revision.entry)},
        "validated": reason is None, "unvalidated_reason": reason,
        "result_sha256": digests or None,
        "require_all_dates_complete": measurement.require_all_dates_complete,
        **measurement.revision.ref,
        "positive_class_assessed": measurement.positive_class_assessed,
        **{key: disclosure[key] for key in ("captures_unverified", "plant_csvs_unverified",
                                            "dates_delivered", "images_unattributed")},
    }


# ── CSV export ───────────────────────────────────────────────────────────


class AcknowledgmentPayload(BaseModel):
    """Why the exporting person ships the unvalidated result shown, and that result's digest
    (``result_sha256``, as the refusal or the screen measurement served it)."""

    model_config = ConfigDict(extra="forbid")

    reason: str
    result_sha256: str


def _recorded_acknowledgment(payload, person: str, request: Request) -> Optional[str]:
    """Record the acknowledgment ``payload`` carries (:func:`~tcip_mcp.delivery.
    record_acknowledgment`) by ``person``, and return its id; ``None`` when it carries none. A
    request with no browser ``Origin``, or one declaring an agent identity
    (:data:`~tcip_mcp.agent_identity.HEADERS`), refuses (403), and a blank reason refuses (400),
    before anything runs; an act recorded whose audit line could not follow answers 409.
    """
    if payload.acknowledgment is None:
        return None
    from tcip_mcp import agent_identity
    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_mcp.delivery import record_acknowledgment
    from tcip_web.routes.audit_gap import audit_gap_409

    if request.headers.get("origin") is None or any(
            request.headers.get(h) for h in agent_identity.HEADERS.values()):
        raise HTTPException(403, "an acknowledgment is the breeder's act in the Results tab: a "
                                 "request from no browser, or from an agent, cannot record one.")
    try:
        return record_acknowledgment(
            store.open_root(), acknowledged_by=person,
            reason=payload.acknowledgment.reason,
            result_sha256=payload.acknowledgment.result_sha256).acknowledgment_id
    except AuditEntryNotWritten as exc:
        raise audit_gap_409(exc, exc.arguments) from exc
    except ValueError as exc:
        raise HTTPException(400, f"an acknowledgment states why: {exc}") from exc


class ExportCsvPayload(PhenologyInputs):
    """The export door's payload: the measurement inputs, which computation to export, the person
    exporting it and the acknowledgment fields."""

    # Which server computation to export: a choice of producer, never a claim about what the rows
    # mean or whether they are valid. Picking the "wrong" one yields a correctly-gated CSV.
    payload: Literal["curves", "milestones"]
    filename: Optional[str] = None
    user: str
    # The breeder's own act of shipping this delivery unvalidated, or None for an ordinary
    # validated export.
    acknowledgment: Optional[AcknowledgmentPayload] = None


@router.post("/export_csv")
def export_csv(payload: ExportCsvPayload, request: Request) -> Response:
    """Deliver a phenology measurement this route computes itself as a CSV
    (:func:`~tcip_mcp.pipelines.postprocessing.phenology.deliver_phenology`), written to the
    project's ``results_export/`` and returned as the response body.

    The acknowledgment is recorded by :func:`_recorded_acknowledgment`; a delivery refusal
    answers 400 with its detail, and a delivery event its receipt could not follow answers 409.
    """
    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_mcp.delivery import DeliveryRefused
    from tcip_mcp.identity import actor
    from tcip_web.routes.audit_gap import audit_gap_409

    person = actor(payload.user)
    measurement = _measure(payload)
    acknowledgment_id = _recorded_acknowledgment(payload, person, request)
    filename = payload.filename or f"{payload.trait}_{payload.payload}.csv"
    saved_path = store.open_root() / "results_export" / Path(filename).name
    try:
        phenology.deliver_phenology(
            store.open_root(), measurement, curves=payload.payload == "curves",
            output_path=saved_path, acknowledgment_id=acknowledgment_id,
            door="results.export_csv", actor=person)
    except AuditEntryNotWritten as exc:
        raise audit_gap_409(exc, {"saved_path": str(saved_path)}) from exc
    except DeliveryRefused as exc:
        raise HTTPException(400, exc.as_detail()) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    headers = {"Content-Disposition": f'attachment; filename="{filename}"',
               "X-TCIP-Saved-To": str(saved_path)}
    return Response(content=saved_path.read_bytes(), media_type="text/csv", headers=headers)


# ── Count CSV export ────────────────────────────────────────────────────


class PerImageCountDelivery(BaseModel):
    """The bucket regime of a per-image count delivery: an existing, reviewed prediction bucket."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["per_image_count"]
    predictions_dir: str = Field(min_length=1)
    trait: str = Field(min_length=1)


class OrthomosaicPlantCountsDelivery(BaseModel):
    """A per-plant count delivery from a published whole-raster prediction bucket plus a registered
    plant registry, for the plants ``plants`` names. ``nn_tolerance_m`` is not exposed here: this
    door serves the derived tolerance the core resolves from the plant grid's own spacing, or the
    ``canopy_subject`` regime.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["orthomosaic_plant_counts"]
    predictions_dir: str = Field(min_length=1)
    plant_registry: str
    delivered_phenotype: str
    plants: list[str]
    crop: str = ""
    pipeline_version: str = ""
    canopy_subject: str = ""


class ExportCountCsvPayload(BaseModel):
    """The count-export door's own payload: a discriminated ``delivery`` naming which of the two
    count kinds this posts, the person exporting it, plus the acknowledgment fields.
    """

    model_config = ConfigDict(extra="forbid")

    delivery: Union[PerImageCountDelivery, OrthomosaicPlantCountsDelivery] = Field(
        discriminator="kind")
    filename: str = Field(min_length=1)
    user: str
    acknowledgment: Optional[AcknowledgmentPayload] = None


@router.post("/export_count_csv")
def export_count_csv(payload: ExportCountCsvPayload, request: Request) -> Response:
    """Deliver a per-image or per-plant count CSV over a bucket (and, for the per-plant kind, the
    registered plant registry) this route confines to the open project, written to the project's
    ``results_export/`` and returned as the response body.

    ``per_image_count`` is
    :func:`~tcip_mcp.pipelines.postprocessing.export.deliver_per_image_counts_csv`;
    ``orthomosaic_plant_counts`` is :func:`~tcip_mcp.tools.orthomosaic_tools.orthomosaic_plant_counts`.
    A delivery refusal answers 400 with
    ``{"kind", "message"}``, ``kind`` ``"operationalization"`` or ``"delivery"``; a delivery
    event its receipt could not follow answers 409. The response headers name the saved path,
    whether the delivery is validated and who acknowledged it when it is not.
    """
    from urllib.parse import quote

    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_mcp.delivery import DeliveryRefused
    from tcip_mcp.identity import actor
    from tcip_mcp.operationalization import OperationalizationRefused
    from tcip_mcp.traits import TraitUnknownError
    from tcip_web.routes.audit_gap import audit_gap_409

    person = actor(payload.user)
    root = store.open_root()
    saved_path = root / "results_export" / Path(payload.filename).name
    delivery = payload.delivery
    (predictions_dir,) = _belonging(root, delivery.predictions_dir)
    assert predictions_dir is not None, "the payload requires a non-empty predictions_dir"
    acknowledgment_id = _recorded_acknowledgment(payload, person, request)
    try:
        if delivery.kind == "per_image_count":
            from tcip_mcp.pipelines.postprocessing.export import deliver_per_image_counts_csv

            result = deliver_per_image_counts_csv(
                root, predictions_dir, str(saved_path), trait=delivery.trait,
                acknowledgment_id=acknowledgment_id, door="results.export_count_csv",
                actor=person)
        else:
            try:
                registry_record = plant_mapping.load_registry(root, delivery.plant_registry)
            except plant_mapping.PlantRegistryNotFound as exc:
                raise HTTPException(404, str(exc)) from exc
            from tcip_mcp.tools.orthomosaic_tools import orthomosaic_plant_counts

            result = orthomosaic_plant_counts(
                root, str(predictions_dir), registry_record, str(saved_path),
                delivery.delivered_phenotype, delivery.plants, crop=delivery.crop,
                pipeline_version=delivery.pipeline_version,
                canopy_subject=delivery.canopy_subject, acknowledgment_id=acknowledgment_id,
                door="results.export_count_csv", actor=person)
    except AuditEntryNotWritten as exc:
        raise audit_gap_409(exc, {"saved_path": str(saved_path)}) from exc
    except (OperationalizationRefused, DeliveryRefused) as exc:
        raise HTTPException(400, exc.as_detail()) from exc
    except (TraitUnknownError, ValueError) as exc:
        raise HTTPException(400, {"kind": "delivery", "message": str(exc)}) from exc
    headers = {
        "Content-Disposition": f'attachment; filename="{Path(payload.filename).name}"',
        "X-TCIP-Saved-To": str(saved_path),
        "X-TCIP-Validated": "true" if result["validated"] else "false",
        "X-TCIP-Acknowledged-By": quote(result["acknowledged_by"] or ""),
    }
    return Response(content=saved_path.read_bytes(), media_type="text/csv", headers=headers)


# ── What has shipped: the delivery-event record, read-only ─────────────


@router.get("/delivery-events")
def list_delivery_events() -> dict:
    """Every delivery event the open project holds (:func:`~tcip_mcp.delivery.read_delivery_events`):
    what shipped, under which trait revision and kind, over which buckets and with what each
    bucket's gate finding was. A record that will not decode refuses the whole listing (400).

    A record naming a walked-mapping ``plant_mapping`` (a dict carrying ``name`` and
    ``record_sha256``) also carries ``plant_mapping_resolved_key``: the name to load to see exactly
    the record this event cites, its own name when a rebuild has not moved past it, the archived
    key (``resolved_mapping_key_for_citation``) when a superseding rebuild has, or ``None`` when
    neither name holds a stored record any more. A record naming a whole-raster ``plant_mapping``
    carries no ``plant_mapping_resolved_key`` at all.
    """
    from pydantic import ValidationError
    from tcip_mcp.delivery import read_delivery_events
    from tcip_mcp.pipelines.delivery_events_schema import PlantMappingDisclosure
    from tcip_mcp.pipelines.postprocessing.plant_mapping import resolved_mapping_key_for_citation

    from tcip_store import StoreError

    root = store.open_root()
    try:
        events = read_delivery_events(root)
    except (StoreError, ValidationError) as exc:
        raise HTTPException(400, str(exc)) from exc
    records = []
    for event in events:
        record = event.model_dump(mode="json")
        pm = event.plant_mapping
        if isinstance(pm, PlantMappingDisclosure):
            record["plant_mapping_resolved_key"] = resolved_mapping_key_for_citation(
                root, pm.name, pm.record_sha256)
        records.append(record)
    return {"records": records}


# ── Traits and the breeder's confirmation of a revision ────────────────


def _served_revision(revision: TraitRevision) -> dict:
    """One revision as the trait routes serve it, with its ``confirmed`` state."""
    return {**revision.model_dump(mode="json"), "confirmed": revision.confirmed}


def _served(name: str, record: TraitRecord) -> dict:
    """One trait's record as the trait routes serve it: each revision as
    :func:`_served_revision` states it, and ``latest_confirmed``, the number of the revision a
    delivery reads, or ``None``."""
    latest_confirmed = record.latest_confirmed
    return {
        "trait": name,
        "revisions": [_served_revision(r) for r in record.revisions],
        "latest_confirmed": latest_confirmed.number if latest_confirmed else None,
    }


@router.get("/traits")
def list_traits() -> dict:
    """Every trait the open project holds, each as :func:`_served` states it, revisions oldest
    first.

    ``definitions`` quotes crops.yml's own definition of every phenotype a revision delivers.
    ``unreadable`` names each trait whose stored record its schema refuses, and why.
    """
    from pydantic import ValidationError
    from tcip_mcp.traits import crops_definitions, read_trait, trait_names

    root = store.open_root()
    records: list[dict] = []
    unreadable: list[dict] = []
    for name in trait_names(root):
        try:
            records.append(_served(name, read_trait(name, root)))
        except ValidationError as exc:
            unreadable.append({"trait": name, "reason": str(exc)})
    definitions = crops_definitions()
    delivered = {p for r in records for rev in r["revisions"] for p in rev["entry"]["delivers"]}
    return {
        "traits": records,
        "unreadable": unreadable,
        "definitions": {p: definitions[p] for p in sorted(delivered) if p in definitions},
    }


class ConfirmRevisionPayload(BaseModel):
    """The breeder's confirmation of one trait revision, or the withdrawal of one they gave.

    ``entry_sha256`` is the hash of the entry the surface showed. ``user`` is the name the surface
    carries.
    """

    model_config = ConfigDict(extra="forbid")

    trait: str
    revision: int
    entry_sha256: str
    user: str
    confirmed: bool


@router.post("/traits/confirm")
def confirm_trait_revision(payload: ConfirmRevisionPayload) -> dict:
    """Record the breeder's confirmation of a trait revision, or withdraw one, through
    ``traits.confirm_revision``.

    Refused with 409 when the hash is not the revision's own, the body carrying the trait's record
    as it stands, and with 400 for every other refusal. A committed confirmation whose audit line
    could not be written returns the revision with that failure as ``audit_warning``. Records the
    name the request supplied, refusing a blank one; it is not authentication.
    """
    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_mcp.identity import actor
    from tcip_mcp.traits import RevisionMoved, TraitUnknownError, confirm_revision, read_trait

    person = actor(payload.user)
    root = store.open_root()
    audit_warning: Optional[str] = None
    try:
        revision = confirm_revision(
            root, payload.trait, payload.revision, payload.entry_sha256, actor=person,
            confirmed=payload.confirmed)
    except AuditEntryNotWritten as e:
        revision = read_trait(payload.trait, root).revisions[payload.revision - 1]
        audit_warning = str(e)
    except RevisionMoved as e:
        record = _served(payload.trait, read_trait(payload.trait, root))
        raise HTTPException(409, {"message": str(e), "record": record}) from e
    except (TraitUnknownError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    return {**_served_revision(revision), "audit_warning": audit_warning}


# ── Registered models ───────────────────────────────────────────────────


@router.get("/models/registered")
def registered_models(tag: Optional[str] = None) -> dict:
    """The open project's registered models (:func:`~tcip_mcp.tools.model_tools.registered_listing`)."""
    from tcip_mcp.tools.model_tools import registered_listing

    return registered_listing(store.open_root(), tag=tag)
