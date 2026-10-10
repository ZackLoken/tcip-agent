"""The declared shapes of a delivery event record and of the acknowledgment it may ship under.

``DeliveryEventRecord.plant_mapping`` is one of three disclosures or ``None``; no two declare the
same required key set, so with an undeclared key forbidden on every model a stored dict validates
against at most one.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict

from tcip_mcp.pipelines.postprocessing.plant_mapping import (
    ToleranceSource, UnattributedSegmentSource,
)
from tcip_mcp.traits import Text


class MatchTolerance(BaseModel):
    """A plant mapping's resolved match radius (meters) and the branch of
    ``plant_mapping.resolve_nn_tolerance_m`` that produced it."""

    model_config = ConfigDict(extra="forbid")

    value: float
    source: ToleranceSource


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
    dataset_id: str
    dataset_root: str
    built_at: str
    record_sha256: str
    nn_tolerance_m: MatchTolerance
    capture_digests: dict[str, dict[str, str]]
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
    raster_identity: dict
    nn_tolerance_m: MatchTolerance
    detections_unattributed: int
    detections_unattributed_scope: Literal["delivered_raster"]
    plant_attribution: str
    plants_outside_raster: list[str]


class CanopySegmentsDocument(BaseModel):
    """The label document a canopy-segment delivery read its boundaries from, named by its capture
    and stem under the delivered bucket's dataset root and by the version it was read at."""

    model_config = ConfigDict(extra="forbid")

    capture: str
    stem: str
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


class CanopySegmentDisclosure(BaseModel):
    """The ``plant_mapping`` an orthomosaic delivery's canopy-segment regime attributed detections
    through.

    Names the registry and raster identity the same way :class:`PlantRegistryDisclosure` does, plus
    the canopy document this delivery read its boundaries from, the resolved segment-to-plant ties,
    the position error in meters the ties were held to, and the registry plants attribution
    excluded, by name and by reason: outside the raster's frame, inside no segment, within the
    position error of its segment's boundary, or inside a segment whose own detection was
    ambiguous (:data:`~tcip_mcp.pipelines.postprocessing.segment_attribution.SegmentAssignment`'s
    ``"overlapping_segments"`` source). Every key is required.
    """

    model_config = ConfigDict(extra="forbid")

    plant_registry: PlantRegistryReference
    raster_identity: dict
    canopy_segments: CanopySegmentsDocument
    segment_ties: list[SegmentTieDisclosure]
    position_error_m: float
    """The registry positions' error bound in meters the ties were held to, the project's
    stated value."""
    segments_without_plant: int
    plants_outside_raster: list[str]
    plants_without_segment: list[str]
    plants_within_position_error: list[str]
    plants_with_ambiguous_detections: list[str]
    detections_unattributed: int
    detections_unattributed_by_source: dict[UnattributedSegmentSource, int]
    """The unattributed detections by the segment source that left each one unattributed; their
    sum is ``detections_unattributed``."""
    detections_unattributed_scope: Literal["delivered_raster"]
    plant_attribution: str


class Producer(BaseModel):
    """The one checkpoint and run behind every delivered bucket
    (:attr:`~tcip_mcp.model_registry.VerifiedCheckpoint.producer`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_sha256: str
    experiment_id: Optional[str]


class BucketFinding(BaseModel):
    """What the gate found for one delivered bucket, named by its dataset root and its name:
    validated when no reason refuses it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_root: str
    bucket: str
    date: Optional[str]
    assessment_id: Optional[str]
    validated: bool
    reason: Optional[str]


class Acknowledgment(BaseModel):
    """The breeder's recorded act of shipping one unvalidated result: its id, who did it and why
    (both required non-empty; a blank one refuses, ``ValueError``), the digest of the result it was
    given for, and when it was recorded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    acknowledgment_id: str
    acknowledged_by: Text
    reason: Text
    result_sha256: str
    recorded_at: str


class DeliveryEventRecord(BaseModel):
    """The stored per-delivery record: what shipped, under which trait revision and kind, from
    which producer, what the gate found per bucket, and the acknowledgment it shipped under."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    door: str
    delivery_kind: str
    trait: str
    trait_revision: int
    trait_revision_sha256: str
    output_path: str
    output_sha256: str
    producer: Producer
    buckets: list[BucketFinding]
    scale_assessment_id: Optional[str]
    validated: bool
    acknowledgment: Optional[Acknowledgment]
    population: list[str]
    require_all_dates_complete: Optional[bool]
    plant_mapping: Optional[
        Union[PlantMappingDisclosure, PlantRegistryDisclosure, CanopySegmentDisclosure]
    ]
    produced_at: str
