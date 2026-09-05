"""The ``delivery_events`` record's declared shape, so its writer (``resolution.py``'s
``record_delivery_binding_event``) and its readers (``tcip_web``'s ``list_delivery_events`` route,
``scripts/conform_delivery_events.py``) agree on one shape rather than each independently
tolerating whatever the others happen to have written.

A sibling module to ``resolution.py`` rather than a class inside it, the same split
``pipelines/schemas.py`` already draws for its own pydantic models: ``resolution.py`` states its
own dependency surface as storage-seam-and-stdlib only, and a pydantic import belongs with the
other schema-only module instead of widening that statement.

``DeliveryEventRecord.plant_mapping`` carries one of three disclosure shapes, or ``None``: a
walked capture mapping's :class:`PlantMappingDisclosure` (the phenology doors, and
``deliver_per_plant_csv`` when its caller verified one), a whole-raster frame's
:class:`PlantRegistryDisclosure` (``deliver_orthomosaic_plant_counts``'s nearest-neighbour
regime, which has no walked mapping build to name), or that same door's canopy-segment regime's
:class:`CanopySegmentDisclosure`. No two of the three declare the same required key set (the
registry form has ``nn_tolerance_m``, the canopy form ``canopy_segments``, the mapping form
``name``), so with an extra key forbidden on every one a stored dict validates against at most
one: pydantic resolves it with no discriminator field added to any of them.

Every model forbids an undeclared key, so a stored record or disclosure carrying one is refused
by name rather than silently accepted and later misread.
"""

from __future__ import annotations

from typing import Literal, Mapping, Optional, TypeGuard, Union

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
    """The ``plant_mapping`` an orthomosaic delivery's nearest-neighbour regime attributed
    detections through, exactly as ``deliver_orthomosaic_plant_counts`` (``orthomosaic_tools.py``)
    composes it.

    A whole-mosaic frame carries no walked capture sequence to build a
    :class:`MappingBuild`-shaped mapping from, so this names only what that door verifies or
    computes itself: the plant registry it read, the raster identity every count in the delivery
    is attributed through, the tolerance it matched detections under, this delivery's own
    unattributed-detection count, and the registry plants the raster's own frame does not
    picture at all, by name. Every key is required, the same reasoning
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
    plants_outside_raster: list[str]


class CanopySegmentsDocument(BaseModel):
    """The label document a canopy-segment delivery read its boundaries from, named and hashed
    over the bytes it actually parsed."""

    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    subject: str
    n_segments: int


class SegmentTieDisclosure(BaseModel):
    """One resolved segment-to-plant tie: the attribution claim itself, and the derived margin
    (:class:`~tcip_mcp.pipelines.postprocessing.segment_attribution.TiedSegment.clearance_m``) a
    displaced registry position would have to exceed to leave this segment."""

    model_config = ConfigDict(extra="forbid")

    segment_index: int
    plot_name: str
    clearance_m: float


class UnattributedDetectionsBySource(BaseModel):
    """A canopy-segment delivery's own unattributed-detection count, broken out by the
    :class:`~tcip_mcp.pipelines.postprocessing.segment_attribution.SegmentAssignment.source`
    that left each one unattributed; ``detections_unattributed`` on the enclosing disclosure is
    the sum of these three, stated there as derived, never independent evidence.

    This sum excludes one further case the door's own response counts as unmapped: a detection
    whose containment resolved to exactly one tied segment (``source="segment_containment"``, so
    it never appears in any of these three counts) but whose only containing segment's plant was
    then dropped from delivery for an ambiguous detection elsewhere in that same segment. The two
    numbers can differ for that reason alone."""

    model_config = ConfigDict(extra="forbid")

    outside_segments: int
    overlapping_segments: int
    segment_without_plant: int


class CanopySegmentDisclosure(BaseModel):
    """The ``plant_mapping`` an orthomosaic delivery's canopy-segment regime attributed
    detections through, exactly as ``deliver_orthomosaic_plant_counts``'s ``canopy_subject``
    argument composes it.

    Names the registry and raster identity the same way :class:`PlantRegistryDisclosure` does,
    plus the canopy document this delivery read its boundaries from, the resolved segment-to-plant
    ties, and every plant this delivery's own rows do not cover, by name and by reason: outside
    the raster's frame, inside no segment, or inside a segment whose own detection was ambiguous
    (:data:`~tcip_mcp.pipelines.postprocessing.segment_attribution.SegmentAssignment`'s
    ``"overlapping_segments"`` source). Every key is required, the same reasoning
    :class:`PlantMappingDisclosure` states: none is reconstructable from a record that lacks it.
    """

    model_config = ConfigDict(extra="forbid")

    plant_registry: dict
    project_root: str
    raster_identity: dict
    canopy_segments: CanopySegmentsDocument
    segment_ties: list[SegmentTieDisclosure]
    segments_without_plant: int
    plants_outside_raster: list[str]
    plants_without_segment: list[str]
    plants_with_ambiguous_detections: list[str]
    detections_unattributed: int
    detections_unattributed_by_source: UnattributedDetectionsBySource
    detections_unattributed_scope: Literal["delivered_raster"]
    plant_attribution: str


def is_mapping_disclosure(pm: object) -> TypeGuard[dict]:
    """Whether ``pm`` is a walked-mapping :class:`PlantMappingDisclosure` dict rather than one of
    the two whole-raster shapes (:class:`PlantRegistryDisclosure`, :class:`CanopySegmentDisclosure`)
    or neither: the one key test every reader that needs to tell the mapping disclosure apart from
    the others (``list_delivery_events``, :func:`~tcip_mcp.pipelines.postprocessing.plant_mapping.
    _citing_delivery_event_ids`) calls, rather than each restating
    ``"name" in pm and "record_sha256" in pm`` on its own."""
    return isinstance(pm, dict) and "name" in pm and "record_sha256" in pm


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
    plant_mapping: Optional[
        Union[PlantMappingDisclosure, PlantRegistryDisclosure, CanopySegmentDisclosure]
    ]
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
    def _superseded(event: dict) -> dict | None:
        event_id = event.get("event_id")
        assert isinstance(event_id, str), "delivery event record is missing its own event_id"
        return supersessions.get(event_id)

    return [{**event, "superseded": _superseded(event)} for event in events]


def validation_error_detail(exc: ValidationError) -> str:
    """``exc``'s errors rendered as one line, the one rendering both the results route's refusal
    and ``scripts/conform_delivery_events.py``'s outcome lines share, so a record's shape errors
    never break a caller's one-line-per-outcome rendering with pydantic's own multi-line dump."""
    return "; ".join(
        f"{'.'.join(str(p) for p in error['loc']) or 'record'}: {error['msg']}"
        for error in exc.errors()
    )
