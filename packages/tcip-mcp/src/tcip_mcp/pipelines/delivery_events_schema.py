"""The ``delivery_events`` record's declared shape, so its writer (``resolution.py``'s
``record_delivery_binding_event``) and its readers (``tcip_web``'s ``list_delivery_events`` route,
``scripts/conform_delivery_events.py``) agree on one shape rather than each independently
tolerating whatever the others happen to have written.

A sibling module to ``resolution.py`` rather than a class inside it, the same split
``pipelines/schemas.py`` already draws for its own pydantic models: ``resolution.py`` states its
own dependency surface as storage-seam-and-stdlib only, and a pydantic import belongs with the
other schema-only module instead of widening that statement.

``DeliveryEventRecord.plant_mapping`` carries one of two disclosure shapes, or ``None``: a walked
capture mapping's :class:`PlantMappingDisclosure` (the phenology doors, and
``deliver_per_plant_csv`` when its caller verified one), or a whole-raster frame's
:class:`PlantRegistryDisclosure` (``deliver_orthomosaic_plant_counts`` alone, which has no walked
mapping build to name). The two share no key, so pydantic resolves a stored dict to exactly one
model with no discriminator field added to either: every key either model declares is required and
both forbid an extra one, so a dict is a legal instance of at most one of the two.

Both models forbid an undeclared key, so a stored record or disclosure carrying one is refused by
name rather than silently accepted and later misread.
"""

from __future__ import annotations

from typing import Literal, Mapping, Optional, Union

from pydantic import BaseModel, ConfigDict, ValidationError


class PlantMappingDisclosure(BaseModel):
    """The ``plant_mapping`` a phenology delivery attributed detections through, exactly as
    :meth:`tcip_mcp.pipelines.postprocessing.plant_mapping.MappingBuild.delivery_disclosure`
    composes it: the mapping's own identity, ``verify_mapping_inputs``'s two unverified
    disclosures, and this delivery's own unattributed-capture count scoped to its delivered dates.

    Every key is required: none of ``dates_delivered``, ``images_unattributed`` or
    ``plant_attribution`` can be reconstructed from a record that lacks it, since none was ever
    computed for that delivery, so a record missing one is refused rather than read as if the gap
    meant something.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    project_root: str
    dataset_id: str
    dataset_root: str
    built_at: str
    record_sha256: str
    nn_tolerance_m: dict
    capture_identity: dict[str, str]
    captures_unverified: list[str]
    plant_csvs_unverified: list[str]
    dates_delivered: list[str]
    images_unattributed: int
    images_unattributed_scope: Literal["delivered_dates"]
    plant_attribution: str


class PlantRegistryDisclosure(BaseModel):
    """The ``plant_mapping`` an orthomosaic delivery attributed detections through, exactly as
    ``deliver_orthomosaic_plant_counts`` (``orthomosaic_tools.py``) composes it.

    A whole-mosaic frame carries no walked capture sequence to build a
    :class:`MappingBuild`-shaped mapping from, so this names only what that door verifies or
    computes itself: the plant registry it read, the raster identity every count in the delivery
    is attributed through, the tolerance it matched detections under, and this delivery's own
    unattributed-detection count. Every key is required, the same reasoning
    :class:`PlantMappingDisclosure` states: none is reconstructable from a record that lacks it.
    """

    model_config = ConfigDict(extra="forbid")

    plant_registry: dict
    project_root: str
    raster_identity: dict
    nn_tolerance_m: dict
    detections_unattributed: int
    detections_unattributed_scope: Literal["delivered_raster"]
    plant_attribution: str


class DocumentBinding(BaseModel):
    """One bucket's binding evidence, exactly as ``record_delivery_binding_event`` renders a
    :class:`tcip_mcp.pipelines.resolution.StampBinding` into the stored record's ``documents``
    mapping."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    claimed: bool
    experiment_id: Optional[str]
    producing_experiment_id: Optional[str]
    checkpoint_sha256: Optional[str]
    record_digest: Optional[str]
    note: str


class DeliveryEventRecord(BaseModel):
    """The stored per-delivery record, as ``record_delivery_binding_event`` writes it and
    ``list_delivery_events`` serves it back: what shipped, under which trait and kind, and the
    real per-bucket verification evidence the delivering door reconciled at the time."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    trait: Optional[str]
    delivery_kind: Optional[str]
    door: str
    output_path: Optional[str]
    output_sha256: Optional[str]
    measurement_documents: list[str]
    scale_document: Optional[str]
    # Who acknowledged this delivery unvalidated, and why: null on both when nothing was
    # acknowledged, the same pair DeliveryGateResult carries.
    acknowledged_by: Optional[str]
    acknowledgement_reason: Optional[str]
    plant_mapping: Optional[Union[PlantMappingDisclosure, PlantRegistryDisclosure]]
    documents: dict[str, DocumentBinding]
    produced_at: str


class DeliverySupersessionRecord(BaseModel):
    """One ``delivery_supersessions`` record, as ``supersede_delivery`` writes it: the superseded
    event's own id and digest, the replacement event when a re-delivery already exists, and the
    non-empty reason and actor behind the withdrawal."""

    model_config = ConfigDict(extra="forbid")

    superseded_event_id: str
    output_sha256: Optional[str]
    replacement_event_id: Optional[str]
    reason: str
    superseded_by: str
    superseded_at: str


def with_supersessions(
    events: list[dict], supersessions: Mapping[str, dict]
) -> list[dict]:
    """Every one of ``events`` (as ``list_delivery_events`` reads them back) with its own
    supersession attached under ``superseded``: the record ``supersede_delivery`` filed against
    that event's id, or ``None`` when nothing supersedes it.

    ``supersessions`` maps a superseded event's id to its own stored ``delivery_supersessions``
    record (:func:`tcip_mcp.pipelines.resolution.load_delivery_supersessions`'s own shape), read
    once by the caller rather than once per event. The one join every delivery-events reader (the
    Results tab's panel route, and ``read_audit_log``) composes through, so the two can never
    disagree about which record answers for which event.
    """
    return [
        {**event, "superseded": supersessions.get(event.get("event_id"))}
        for event in events
    ]


def validation_error_detail(exc: ValidationError) -> str:
    """``exc``'s errors rendered as one line, the one rendering both the results route's refusal
    and ``scripts/conform_delivery_events.py``'s outcome lines share, so a record's shape errors
    never break a caller's one-line-per-outcome rendering with pydantic's own multi-line dump."""
    return "; ".join(
        f"{'.'.join(str(p) for p in error['loc']) or 'record'}: {error['msg']}"
        for error in exc.errors()
    )
