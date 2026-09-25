"""The ``delivery_events`` record's declared shape.

``DeliveryEventRecord.plant_mapping`` carries one of three disclosure shapes, or ``None``: a walked
capture mapping's :class:`PlantMappingDisclosure`, a whole-raster frame's
:class:`PlantRegistryDisclosure` (the nearest-neighbor regime of
``deliver_orthomosaic_plant_counts``), or that door's canopy-segment regime's
:class:`CanopySegmentDisclosure`. No two of the three declare the same required key set (the
registry form has ``nn_tolerance_m``, the canopy form ``canopy_segments``, the mapping form
``name``), so with an extra key forbidden on every one a stored dict validates against at most one,
with no discriminator field.

Every model forbids an undeclared key, so a stored record or disclosure carrying one is refused by
name.
"""

from __future__ import annotations

from typing import Literal, Mapping, Optional, TypeGuard, Union

from pydantic import BaseModel, ConfigDict, ValidationError


class MatchTolerance(BaseModel):
    """A plant mapping's resolved match radius (meters) and the branch of
    ``plant_mapping.resolve_nn_tolerance_m`` that produced it."""

    model_config = ConfigDict(extra="forbid")

    value: float
    source: str


class PlantRegistryReference(BaseModel):
    """The registered plant registry a delivery read, by name and content digest."""

    model_config = ConfigDict(extra="forbid")

    name: str
    digest: str


class PlantMappingDisclosure(BaseModel):
    """The ``plant_mapping`` a phenology delivery attributed detections through, as
    :meth:`tcip_mcp.pipelines.postprocessing.plant_mapping.MappingBuild.delivery_disclosure`
    composes it: the mapping's own identity, ``verify_mapping_inputs``'s two unverified
    disclosures, and this delivery's own unattributed-capture count scoped to its delivered dates.
    Every key is required.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    project_root: str
    dataset_id: str
    dataset_root: str
    built_at: str
    record_sha256: str
    nn_tolerance_m: MatchTolerance
    capture_identity: dict[str, str]
    captures_unverified: list[str]
    plant_csvs_unverified: list[str]
    dates_delivered: list[str]
    images_unattributed: int
    images_unattributed_scope: Literal["delivered_dates"]
    plant_attribution: str


class PlantRegistryDisclosure(BaseModel):
    """The ``plant_mapping`` an orthomosaic delivery's nearest-neighbor regime attributed
    detections through: the plant registry it read, the raster identity every count in the delivery
    is attributed through, the tolerance it matched detections under, this delivery's own
    unattributed-detection count, and the registry plants the raster's own frame does not picture
    at all, by name. Every key is required.
    """

    model_config = ConfigDict(extra="forbid")

    plant_registry: PlantRegistryReference
    project_root: str
    raster_identity: dict
    nn_tolerance_m: MatchTolerance
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
    """The ``plant_mapping`` an orthomosaic delivery's canopy-segment regime attributed detections
    through.

    Names the registry and raster identity the same way :class:`PlantRegistryDisclosure` does, plus
    the canopy document this delivery read its boundaries from, the resolved segment-to-plant ties,
    and every plant this delivery's own rows do not cover, by name and by reason: outside the
    raster's frame, inside no segment, or inside a segment whose own detection was ambiguous
    (:data:`~tcip_mcp.pipelines.postprocessing.segment_attribution.SegmentAssignment`'s
    ``"overlapping_segments"`` source). Every key is required.
    """

    model_config = ConfigDict(extra="forbid")

    plant_registry: PlantRegistryReference
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
    the two whole-raster shapes (:class:`PlantRegistryDisclosure`,
    :class:`CanopySegmentDisclosure`) or neither.
    """
    return isinstance(pm, dict) and "name" in pm and "record_sha256" in pm


class DocumentBinding(BaseModel):
    """One bucket's binding evidence, exactly as ``record_delivery_binding_event`` renders a
    :class:`tcip_mcp.pipelines.resolution.StampBinding` into the stored record's ``documents``
    mapping and into each :class:`ReconciledDocument`'s own ``bindings``."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    claimed: bool
    experiment_id: Optional[str]
    producing_experiment_id: Optional[str]
    checkpoint_sha256: Optional[str]
    record_digest: Optional[str]
    note: str


class ReconciledDocument(BaseModel):
    """One sidecar document's reconciled validity, as
    :func:`tcip_mcp.pipelines.resolution._reconcile_validity` returns it, keyed by the document
    name.

    ``bound_validated`` and ``delivery_note`` are set only on the classifier entry: the
    delivery-level state ``bind_classifier_validity`` returned (the one the gate actually used),
    kept beside the reconciler's own ``validated`` since the two differ when the binding floors a
    validated stamp.
    """

    model_config = ConfigDict(extra="forbid")

    validated: str
    on_disk_validated: bool
    missing_sidecars: list[str]
    unvalidated_buckets: list[str]
    binding_notes: dict[str, str]
    bindings: dict[str, DocumentBinding]
    conf: Optional[float]
    confs: dict[str, Optional[float]]
    per_bucket: dict[str, str]
    bound_validated: Optional[str] = None
    delivery_note: Optional[str] = None


class ReconciledDimension(BaseModel):
    """One geometry or scope dimension's reconciled validity (``claim_scope``, ``tile_size`` or
    ``scale``), exactly as ``reconcile_claim_scope_validity``, ``reconcile_tile_size_validity``
    and ``reconcile_scale_validity`` (``resolution.py``) return it. ``validated`` is ``None``
    when ``operative`` is ``False``: never operative for this delivery, not a failed reference.
    """

    model_config = ConfigDict(extra="forbid")

    operative: bool
    validated: Optional[str]
    per_bucket: dict[str, str]
    unvalidated_buckets: list[str]
    binding_notes: dict[str, str]


class DeliveryEventRecord(BaseModel):
    """The stored per-delivery record: what shipped, under which trait and kind, and the real
    per-bucket verification evidence the delivering door reconciled at the time."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    trait: Optional[str]
    delivery_kind: Optional[str]
    door: str
    output_path: Optional[str]
    output_sha256: Optional[str]
    # Who acknowledged this delivery unvalidated, and why: null on both when nothing was
    # acknowledged, the same pair DeliveryGateResult carries.
    acknowledged_by: Optional[str]
    acknowledgment_reason: Optional[str]
    plant_mapping: Optional[
        Union[PlantMappingDisclosure, PlantRegistryDisclosure, CanopySegmentDisclosure]
    ]
    documents: dict[str, DocumentBinding]
    # Keyed by the reconciler the delivering door's gate ran.
    document_reconciliations: dict[str, ReconciledDocument]
    dimension_reconciliations: dict[str, ReconciledDimension]
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
    """Every one of ``events`` with its own supersession attached under ``superseded``: the record
    ``supersede_delivery`` filed against that event's id, or ``None`` when nothing supersedes it.

    ``supersessions`` maps a superseded event's id to its own stored ``delivery_supersessions``
    record (:func:`tcip_mcp.pipelines.resolution.load_delivery_supersessions`'s own shape).
    """
    def _superseded(event: dict) -> dict | None:
        event_id = event.get("event_id")
        assert isinstance(event_id, str), "delivery event record is missing its own event_id"
        return supersessions.get(event_id)

    return [{**event, "superseded": _superseded(event)} for event in events]


def validation_error_detail(exc: ValidationError) -> str:
    """``exc``'s errors rendered as one line."""
    return "; ".join(
        f"{'.'.join(str(p) for p in error['loc']) or 'record'}: {error['msg']}"
        for error in exc.errors()
    )
