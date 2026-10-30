"""Results routes: plant-mapping, per-plant phenology curves, CSV export and the traits.

The traits are served with the breeder's confirmation of a trait revision.

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
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, computed_field

from tcip_mcp.assessment import Assessment
from tcip_mcp.delivery import PhenologyRefusal, ProjectionDigests
from tcip_mcp.pipelines.delivery_events_schema import (
    DeliveryEventRecord,
    PlantMappingDisclosure,
)
from tcip_mcp.pipelines.postprocessing import phenology, plant_mapping
from tcip_mcp.pipelines.postprocessing.segment_attribution import PositionError
from tcip_mcp.pipelines.schemas import TraitRevisionRef
from tcip_mcp.tools.model_tools import RegisteredModels
from tcip_mcp.traits import TraitRevision
from tcip_web.paths import resolved_path, within
from tcip_web.state import store

if TYPE_CHECKING:
    from tcip_mcp.pipelines.postprocessing.phenology import (
        PhenologyMeasurement,
    )
    from tcip_mcp.traits import TraitRecord

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/results", tags=["results"])


def _evidence_roots(root: Path) -> list[Path]:
    """Return the open project's own tree plus every dataset root registered to it.

    A registry that will not decode answers 400.
    """
    from tcip_mcp.tools.project_tools import dataset_entry_path, read_datasets
    from tcip_store import DecodeError

    try:
        entries = read_datasets(root)
    except DecodeError as exc:
        raise HTTPException(400, str(exc)) from exc
    return [root, *(dataset_entry_path(root, e) for e in entries)]


def _is_root(resolved: Path, root: Path) -> bool:
    """Return whether ``resolved`` is the established root ``root`` itself."""
    return resolved == root


def _belonging(
    root: Path, path: str, admits: Callable[[Path, Path], bool] = within
) -> Path:
    """Resolve ``path`` when ``admits`` it against the open project or a dataset registered to it.

    By default ``admits`` means at or under one; any other path refuses (403), naming the roots
    checked.
    """
    roots = _evidence_roots(root)
    resolved = resolved_path(path)
    if not any(admits(resolved, r) for r in roots):
        raise HTTPException(
            403,
            f"{path} does not belong to project {root}, whose own roots are the project and "
            f"the datasets registered to it ({', '.join(str(r) for r in roots)})",
        )
    return resolved


# ── Plant mapping ──────────────────────────────────────────────────────


class BuildMappingPayload(BaseModel):
    """A plant-mapping build: its name, dataset root, plant registry, dates, radius and person."""

    name: str
    dataset_root: str = Field(min_length=1)
    plant_registry: str
    dates: list[str] | None = None
    match_radius: plant_mapping.StatedRadius | None = None
    supersede: bool = False
    user: str


@router.post("/plant_mapping/build")
def build_plant_mapping(
    payload: BuildMappingPayload,
) -> plant_mapping.ServedMapping:
    """Build and persist the open project's plant mapping, answering it as served.

    The build is :func:`~tcip_mcp.pipelines.postprocessing.plant_mapping.build_plant_mapping`'s,
    from a ``dataset_root`` confined to the project, by the person ``user`` names, answered as
    ``MappingBuild.served`` states it.

    A refusal answers 400, an unknown registry 404, a rebuild a delivery event still cites 409
    unless ``supersede``, and a receipt that could not be written 409.
    """
    from tcip_mcp.audit import AuditEntryNotWrittenError
    from tcip_mcp.identity import actor
    from tcip_web.routes.audit_gap import audit_gap_409

    person = actor(payload.user)
    root = store.held().root
    dataset_root = _belonging(root, payload.dataset_root)
    try:
        return plant_mapping.build_plant_mapping(
            root,
            payload.name,
            dataset_root,
            payload.plant_registry,
            dates=payload.dates,
            match_radius=payload.match_radius,
            supersede=payload.supersede,
            actor=person,
        ).served()
    except plant_mapping.PlantRegistryNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except AuditEntryNotWrittenError as exc:
        raise audit_gap_409(exc, None) from exc
    except (
        plant_mapping.MappingRebuildError,
        plant_mapping.UngeoreferencedCaptureError,
    ) as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class LoadMappingPayload(BaseModel):
    """The name of a persisted plant mapping to load."""

    name: str


@router.post("/plant_mapping/load")
def load_plant_mapping(
    payload: LoadMappingPayload,
) -> plant_mapping.ServedMapping:
    """Load the persisted mapping under ``payload.name`` as ``MappingBuild.served`` states it.

    Nothing stored under the name answers 404; a name the key refuses, or a record that will not
    load, 409.
    """
    from tcip_store import StoreError

    root = store.held().root
    try:
        build = plant_mapping.load_mapping(root, payload.name)
    except (StoreError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc
    if build is None:
        raise HTTPException(
            404, f"no plant mapping named {payload.name!r} under {root}"
        )
    return build.served()


class MappingNames(BaseModel):
    """Every mapping name persisted under a project."""

    names: list[str]


@router.get("/plant_mapping/list")
def list_plant_mappings() -> MappingNames:
    """List every mapping name persisted under the open project."""
    root = store.held().root
    return MappingNames(names=plant_mapping.plant_mapping_names(root))


# ── Per-plant curves ───────────────────────────────────────────────────


class PhenologyInputs(BaseModel):
    """The inputs a phenology measurement is computed from."""

    model_config = ConfigDict(extra="forbid")

    mapping_name: str  # a name persisted under the open project's own plant-mapping store
    dataset_root: (
        str  # the project or a dataset registered to it, the buckets' root
    )
    # The name of the published bucket of each delivered date; each stands for its record's date.
    buckets: list[str]
    trait: str
    # The population: the plant ids this measurement is for, one row each.
    plants: list[str]


class PhenologyPayload(PhenologyInputs):
    """The screen route's payload: the measurement inputs plus a display choice."""

    # Show unvalidated numbers on screen, which acknowledges nothing.
    show_unvalidated: bool = False


def _measure(root: Path, payload: PhenologyInputs) -> PhenologyMeasurement:
    """Measure phenology over the buckets this route confines to the project ``root``.

    The measurement is :func:`~tcip_mcp.pipelines.postprocessing.phenology.measure_phenology`'s;
    each of its refusals answers 400.
    """
    from tcip_mcp.operationalization import OperationalizationRefusedError
    from tcip_mcp.subject_registry import RegistryError
    from tcip_mcp.traits import TraitUnknownError

    dataset_root = _belonging(root, payload.dataset_root, _is_root)
    try:
        return phenology.measure_phenology(
            root,
            trait=payload.trait,
            mapping_name=payload.mapping_name,
            dataset_root=dataset_root,
            buckets=payload.buckets,
            plants=payload.plants,
        )
    except OperationalizationRefusedError as exc:
        raise HTTPException(400, exc.as_detail()) from exc
    except plant_mapping.MappingDeliveryError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    except phenology.measurement_refusals() as exc:
        raise HTTPException(400, str(exc)) from exc
    except (TraitUnknownError, RegistryError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


class PhenologyAnswer(BaseModel):
    """Both phenology projections of one measurement, under the trait revision it is measured by.

    ``curves`` is the per-(plant, date) projection. ``unvalidated`` is the delivery gate's
    refusal with the projections' digests when no assessment answers for every delivered bucket,
    ``None`` when the gate cleared. ``positive_class_assessed`` is the measurement's
    (:attr:`~tcip_mcp.pipelines.postprocessing.phenology.PhenologyMeasurement.positive_class_assessed`),
    and ``plant_mapping`` the delivery's mapping disclosure.
    """

    revision: TraitRevisionRef
    curves: list[phenology.CurveRow]
    milestones: list[phenology.MilestoneRow]
    unvalidated: PhenologyRefusal | None
    positive_class_assessed: bool
    plant_mapping: plant_mapping.PlantMappingDisclosure


@router.post("/phenology_measurement")
def phenology_measurement(payload: PhenologyPayload) -> PhenologyAnswer:
    """Return both phenology projections from one measurement, cleared through the delivery gate.

    The projections are the per-(plant, date) curve and the per-plant milestone crossings. The
    gate is the one delivery gate: its refusal
    answers 400 with the refusal's detail unless ``show_unvalidated``, in which case both are
    returned with the refusal and the digests (:class:`~tcip_mcp.delivery.PhenologyRefusal`).
    """
    from tcip_mcp.delivery import DeliveryRefusedError, gate
    from tcip_mcp.operationalization import OperationalizationRefusedError
    from tcip_mcp.traits import STATE_CROSSING_DATES

    root = store.held().root
    measurement = _measure(root, payload)
    reason = None
    digests: dict[phenology.Projection, str] = {}
    for projection in get_args(phenology.Projection):
        try:
            gate(
                root,
                list(measurement.buckets.values()),
                delivery_kind=STATE_CROSSING_DATES,
                revision=measurement.revision,
                result=measurement.result(projection),
            )
        except OperationalizationRefusedError as exc:
            raise HTTPException(400, exc.as_detail()) from exc
        except DeliveryRefusedError as exc:
            if not payload.show_unvalidated or exc.result_sha256 is None:
                raise HTTPException(400, exc.as_detail()) from exc
            reason, digests[projection] = str(exc), exc.result_sha256
    return PhenologyAnswer(
        revision=measurement.revision.ref,
        curves=measurement.phenology.curves,
        milestones=measurement.phenology.milestones,
        unvalidated=None
        if reason is None
        else PhenologyRefusal(message=reason, digests=ProjectionDigests(**digests)),
        positive_class_assessed=measurement.positive_class_assessed,
        plant_mapping=measurement.plant_mapping,
    )


# ── CSV export ───────────────────────────────────────────────────────────

EXPORT_HEADER = "X-TCIP-Export"
"""The header an export's CSV answer reports what it delivered in: its record's ASCII JSON
(:func:`exported`)."""


class SavedExport(BaseModel):
    """Where an export wrote its CSV, as its answer and an audit-gap refusal of it carry it.

    ``filename`` is the written file's own name, the one its download is named by.
    """

    saved_path: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def filename(self) -> str:
        """The written file's own name."""
        return Path(self.saved_path).name


class CountExport(SavedExport):
    """What a count export delivered: where, whether validated, and who acknowledged it if not.

    ``acknowledged_by`` is ``None`` for a validated delivery.
    """

    validated: bool
    acknowledged_by: str | None


def exported(saved: SavedExport) -> Response:
    """Answer the CSV ``saved`` names as an attachment named by its own file name, reporting it.

    The report is :data:`EXPORT_HEADER`'s.
    """
    return Response(
        content=Path(saved.saved_path).read_bytes(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{saved.filename}"',
            EXPORT_HEADER: saved.model_dump_json(ensure_ascii=True),
        },
    )


class AcknowledgmentPayload(BaseModel):
    """Why the exporting person ships the unvalidated result shown, and that result's digest.

    The digest is ``result_sha256``, as the refusal or the screen measurement served it.
    """

    model_config = ConfigDict(extra="forbid")

    reason: str
    result_sha256: str


def _recorded_acknowledgment(
    root: Path,
    payload: ExportCsvPayload | ExportCountCsvPayload,
    person: str,
    request: Request,
) -> str | None:
    """Record the acknowledgment ``payload`` carries and return its id; ``None`` when none.

    The acknowledgment is recorded through :func:`~tcip_mcp.delivery.record_acknowledgment` in
    the project ``root`` by ``person``. A request with no browser ``Origin``, or one declaring an
    agent identity (:data:`~tcip_mcp.agent_identity.HEADERS`), refuses (403), and a blank reason
    refuses (400), before anything runs; an act recorded whose audit line could not follow
    answers 409.
    """
    if payload.acknowledgment is None:
        return None
    from tcip_mcp import agent_identity
    from tcip_mcp.audit import AuditEntryNotWrittenError
    from tcip_mcp.delivery import record_acknowledgment
    from tcip_web.routes.audit_gap import audit_gap_409

    if request.headers.get("origin") is None or any(
        request.headers.get(h) for h in agent_identity.HEADERS.values()
    ):
        raise HTTPException(
            403,
            "an acknowledgment is the breeder's act in the Results tab: a "
            "request from no browser, or from an agent, cannot record one.",
        )
    try:
        return record_acknowledgment(
            root,
            acknowledged_by=person,
            reason=payload.acknowledgment.reason,
            result_sha256=payload.acknowledgment.result_sha256,
        ).acknowledgment_id
    except AuditEntryNotWrittenError as exc:
        raise audit_gap_409(exc, exc.arguments) from exc
    except ValueError as exc:
        raise HTTPException(
            400, f"an acknowledgment states why: {exc}"
        ) from exc


class ExportCsvPayload(PhenologyInputs):
    """The export door's payload: the measurement inputs and the export's own fields.

    Those are which computation to export, the person exporting it and the acknowledgment
    fields.
    """

    # Which server computation to export, a choice of producer; either answers a gated CSV.
    payload: phenology.Projection
    user: str
    # The breeder's own act of shipping this delivery unvalidated, or None for an ordinary
    # validated export.
    acknowledgment: AcknowledgmentPayload | None = None


@router.post("/export_csv")
def export_csv(payload: ExportCsvPayload, request: Request) -> Response:
    """Deliver a phenology measurement this route computes itself as a CSV.

    The delivery is :func:`~tcip_mcp.pipelines.postprocessing.phenology.deliver_phenology`'s,
    written to the project's ``results_export/`` under
    :func:`~tcip_mcp.pipelines.postprocessing.phenology.export_filename` and returned as the
    response body, its :class:`SavedExport` reported.

    The acknowledgment is recorded by :func:`_recorded_acknowledgment`; a delivery refusal
    answers 400 with its detail, and a delivery event its receipt could not follow answers 409.
    """
    from tcip_mcp.audit import AuditEntryNotWrittenError
    from tcip_mcp.delivery import DeliveryRefusedError
    from tcip_mcp.identity import actor
    from tcip_web.routes.audit_gap import audit_gap_409

    person = actor(payload.user)
    root = store.held().root
    measurement = _measure(root, payload)
    acknowledgment_id = _recorded_acknowledgment(
        root, payload, person, request
    )
    saved_path = (
        root / "results_export" / phenology.export_filename(payload.trait, payload.payload)
    )
    try:
        phenology.deliver_phenology(
            root,
            measurement,
            projection=payload.payload,
            output_path=saved_path,
            acknowledgment_id=acknowledgment_id,
            door="results.export_csv",
            actor=person,
        )
    except AuditEntryNotWrittenError as exc:
        raise audit_gap_409(
            exc, SavedExport(saved_path=str(saved_path))
        ) from exc
    except DeliveryRefusedError as exc:
        raise HTTPException(400, exc.as_detail()) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return exported(SavedExport(saved_path=str(saved_path)))


# ── Count CSV export ────────────────────────────────────────────────────


class PerImageCountDelivery(BaseModel):
    """The bucket regime of a per-image count delivery: an existing, reviewed prediction bucket."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["per_image_count"]
    dataset_root: str = Field(min_length=1)
    bucket: str = Field(min_length=1)
    trait: str = Field(min_length=1)


class OrthomosaicPlantCountsDelivery(BaseModel):
    """A per-plant count delivery from a published whole-raster bucket and a plant registry.

    The bucket is a published whole-raster prediction bucket and the registry a registered
    plant registry, for the plants ``plants`` names. The payload carries no ``nn_tolerance_m``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["orthomosaic_plant_counts"]
    dataset_root: str = Field(min_length=1)
    bucket: str = Field(min_length=1)
    plant_registry: str
    delivered_phenotype: str
    plants: list[str]
    crop: str = ""
    pipeline_version: str = ""
    canopy_subject: str = ""
    position_error_m: PositionError | None = None


class ExportCountCsvPayload(BaseModel):
    """The count-export door's own payload: its delivery, its person and the acknowledgment.

    ``delivery`` is discriminated, naming which of the two count kinds this posts; the person is
    the one exporting it.
    """

    model_config = ConfigDict(extra="forbid")

    delivery: PerImageCountDelivery | OrthomosaicPlantCountsDelivery = Field(
        discriminator="kind"
    )
    filename: str = Field(min_length=1)
    user: str
    acknowledgment: AcknowledgmentPayload | None = None


@router.post("/export_count_csv")
def export_count_csv(
    payload: ExportCountCsvPayload, request: Request
) -> Response:
    """Deliver a per-image or per-plant count CSV over a bucket this route confines to the project.

    For the per-plant kind the registered plant registry is confined likewise. The CSV is
    written to the project's ``results_export/`` and returned as the response body.

    ``per_image_count`` is
    :func:`~tcip_mcp.pipelines.postprocessing.export.deliver_per_image_counts_csv`;
    ``orthomosaic_plant_counts`` is
    :func:`~tcip_mcp.tools.orthomosaic_tools.orthomosaic_plant_counts`.
    A delivery refusal answers 400 with its detail
    (:class:`~tcip_mcp.operationalization.OperationalizationRefusal` or
    :class:`~tcip_mcp.delivery.DeliveryRefusal`); a delivery event its receipt could not follow
    answers 409. The answer reports a :class:`CountExport` (:func:`exported`).
    """
    from tcip_mcp.audit import AuditEntryNotWrittenError
    from tcip_mcp.delivery import DeliveryRefusal, DeliveryRefusedError
    from tcip_mcp.identity import actor
    from tcip_mcp.operationalization import OperationalizationRefusedError
    from tcip_mcp.traits import TraitUnknownError
    from tcip_web.routes.audit_gap import audit_gap_409

    person = actor(payload.user)
    root = store.held().root
    saved_path = root / "results_export" / Path(payload.filename).name
    delivery = payload.delivery
    dataset_root = _belonging(root, delivery.dataset_root, _is_root)
    acknowledgment_id = _recorded_acknowledgment(
        root, payload, person, request
    )
    try:
        if delivery.kind == "per_image_count":
            from tcip_mcp.pipelines.postprocessing.export import (
                deliver_per_image_counts_csv,
            )

            result = deliver_per_image_counts_csv(
                root,
                dataset_root,
                delivery.bucket,
                str(saved_path),
                trait=delivery.trait,
                acknowledgment_id=acknowledgment_id,
                door="results.export_count_csv",
                actor=person,
            )
        else:
            try:
                registry_record = plant_mapping.load_registry(
                    root, delivery.plant_registry
                )
            except plant_mapping.PlantRegistryNotFoundError as exc:
                raise HTTPException(404, str(exc)) from exc
            from tcip_mcp.tools.orthomosaic_tools import (
                orthomosaic_plant_counts,
            )

            result = orthomosaic_plant_counts(
                root,
                dataset_root,
                delivery.bucket,
                registry_record,
                saved_path,
                delivery.delivered_phenotype,
                delivery.plants,
                crop=delivery.crop,
                pipeline_version=delivery.pipeline_version,
                canopy_subject=delivery.canopy_subject,
                position_error_m=delivery.position_error_m,
                acknowledgment_id=acknowledgment_id,
                door="results.export_count_csv",
                actor=person,
            )
    except AuditEntryNotWrittenError as exc:
        raise audit_gap_409(
            exc, SavedExport(saved_path=str(saved_path))
        ) from exc
    except (OperationalizationRefusedError, DeliveryRefusedError) as exc:
        raise HTTPException(400, exc.as_detail()) from exc
    except (TraitUnknownError, ValueError) as exc:
        raise HTTPException(
            400,
            DeliveryRefusal(message=str(exc), result_sha256=None).model_dump(),
        ) from exc
    return exported(
        CountExport(
            saved_path=str(saved_path),
            validated=result["validated"],
            acknowledged_by=result["acknowledged_by"],
        )
    )


# ── What has shipped: the delivery-event record, read-only ─────────────


class ServedDeliveryEvent(BaseModel):
    """A stored delivery event as served, with the key the mapping it cites loads under.

    ``resolved_key`` is the name to load to see exactly the record a walked mapping's disclosure
    cites: the mapping's own name while no rebuild has moved past it, the archived key a
    superseding rebuild gave it (``plant_mapping.resolved_mapping_key_for_citation``), or ``None``
    once neither name holds a stored record, and for an event citing no walked mapping.
    """

    event: DeliveryEventRecord
    resolved_key: str | None


class DeliveryEvents(BaseModel):
    """Every delivery event a project holds, as served."""

    records: list[ServedDeliveryEvent]


@router.get("/delivery-events")
def list_delivery_events() -> DeliveryEvents:
    """List every delivery event the open project holds.

    The events are :func:`~tcip_mcp.delivery.read_delivery_events`'s: what shipped, under which
    trait revision and kind, over which buckets and with what each bucket's gate finding was,
    each a :class:`ServedDeliveryEvent`. A record that will not decode refuses the whole listing
    (400).
    """
    from pydantic import ValidationError

    from tcip_mcp.delivery import read_delivery_events
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        resolved_mapping_key_for_citation,
    )
    from tcip_store import StoreError

    root = store.held().root
    try:
        events = read_delivery_events(root)
    except (StoreError, ValidationError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return DeliveryEvents(
        records=[
            ServedDeliveryEvent(
                event=event,
                resolved_key=resolved_mapping_key_for_citation(
                    root, pm.name, pm.record_sha256
                )
                if isinstance(
                    pm := event.plant_mapping, PlantMappingDisclosure
                )
                else None,
            )
            for event in events
        ]
    )


# ── Traits and the breeder's confirmation of a revision ────────────────


class ServedRevision(TraitRevision):
    """One revision as the trait routes serve it: the stored revision and whether it stands."""

    confirmed = computed_field(TraitRevision.confirmed)


class ServedTrait(BaseModel):
    """One trait's stored record, revisions oldest first, with the revision a delivery reads.

    ``latest_confirmed`` is the latest confirmed revision, ``None`` while none is.
    """

    trait: str
    revisions: list[ServedRevision]
    latest_confirmed: ServedRevision | None

    @classmethod
    def of(cls, name: str, record: TraitRecord) -> ServedTrait:
        """Return the record ``name``'s trait holds as served."""
        latest = record.latest_confirmed
        return cls(
            trait=name,
            revisions=[
                ServedRevision.model_validate(r, from_attributes=True)
                for r in record.revisions
            ],
            latest_confirmed=None
            if latest is None
            else ServedRevision.model_validate(latest, from_attributes=True),
        )


class UnreadableTrait(BaseModel):
    """A trait whose stored record its schema refuses, and why."""

    trait: str
    reason: str


class TraitsListing(BaseModel):
    """Every trait a project holds, each trait it cannot read, and the phenotypes' definitions.

    ``definitions`` quotes crops.yml's own definition of every phenotype a revision delivers.
    """

    traits: list[ServedTrait]
    unreadable: list[UnreadableTrait]
    definitions: dict[str, str]


@router.get("/traits")
def list_traits() -> TraitsListing:
    """List every trait the open project holds (:class:`TraitsListing`)."""
    from pydantic import ValidationError

    from tcip_mcp.traits import crops_definitions, read_trait, trait_names

    root = store.held().root
    records: list[ServedTrait] = []
    unreadable: list[UnreadableTrait] = []
    for name in trait_names(root):
        try:
            records.append(ServedTrait.of(name, read_trait(name, root)))
        except ValidationError as exc:
            unreadable.append(UnreadableTrait(trait=name, reason=str(exc)))
    definitions = crops_definitions()
    delivered = {
        p for r in records for rev in r.revisions for p in rev.entry.delivers
    }
    return TraitsListing(
        traits=records,
        unreadable=unreadable,
        definitions={
            p: definitions[p] for p in sorted(delivered) if p in definitions
        },
    )


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


class ConfirmedRevision(BaseModel):
    """A confirmation's answer: the revision as it stands and the audit line's failure, if any.

    ``audit_warning`` is ``None`` when the confirmation's audit line was written.
    """

    revision: ServedRevision
    audit_warning: str | None


class RevisionMovedRefusal(BaseModel):
    """A confirmation refused for naming a hash that is not the revision's own.

    It carries the trait's record as it stands.
    """

    kind: Literal["revision_moved"] = "revision_moved"
    message: str
    record: ServedTrait


@router.post("/traits/confirm")
def confirm_trait_revision(
    payload: ConfirmRevisionPayload,
) -> ConfirmedRevision:
    """Record the breeder's confirmation of a trait revision, or withdraw one.

    Both go through ``traits.confirm_revision``.

    Refused with 409 when the hash is not the revision's own (:class:`RevisionMovedRefusal`),
    and with 400 for every other refusal. Records the name the request supplied, refusing a
    blank one; it is not authentication.
    """
    from tcip_mcp.audit import AuditEntryNotWrittenError
    from tcip_mcp.identity import actor
    from tcip_mcp.traits import (
        RevisionMovedError,
        TraitUnknownError,
        confirm_revision,
        read_trait,
    )

    person = actor(payload.user)
    root = store.held().root
    audit_warning: str | None = None
    try:
        revision = confirm_revision(
            root,
            payload.trait,
            payload.revision,
            payload.entry_sha256,
            actor=person,
            confirmed=payload.confirmed,
        )
    except AuditEntryNotWrittenError as e:
        revision = read_trait(payload.trait, root).revisions[
            payload.revision - 1
        ]
        audit_warning = str(e)
    except RevisionMovedError as e:
        record = ServedTrait.of(payload.trait, read_trait(payload.trait, root))
        raise HTTPException(
            409,
            RevisionMovedRefusal(message=str(e), record=record).model_dump(
                mode="json"
            ),
        ) from e
    except (TraitUnknownError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    return ConfirmedRevision(
        revision=ServedRevision.model_validate(revision, from_attributes=True),
        audit_warning=audit_warning,
    )


# ── Registered models ───────────────────────────────────────────────────


class AssessmentSummary(BaseModel):
    """One recorded assessment, whole, and whether it passed (:attr:`Assessment.passed`)."""

    assessment: Assessment

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        """Whether the assessment passed, its own :attr:`Assessment.passed`."""
        return self.assessment.passed


class Assessments(BaseModel):
    """The open project's recorded assessments."""

    assessments: list[AssessmentSummary]


@router.get("/assessments")
def assessments() -> Assessments:
    """List every assessment recorded under the open project.

    The listing is :func:`~tcip_mcp.assessment.list_assessments`'s.
    """
    from tcip_mcp.assessment import list_assessments

    return Assessments(
        assessments=[
            AssessmentSummary(assessment=a)
            for a in list_assessments(store.held().root)
        ]
    )


@router.get("/models/registered")
def registered_models(tag: str | None = None) -> RegisteredModels:
    """List the open project's registered models under ``tag``.

    The listing is :func:`~tcip_mcp.tools.model_tools.registered_listing`'s.
    """
    from tcip_mcp.tools.model_tools import registered_listing

    return registered_listing(store.held().root, tag=tag)
