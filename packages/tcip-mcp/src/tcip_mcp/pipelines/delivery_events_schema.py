"""The declared shapes of a delivery event record and of the acknowledgment it may ship under.

``DeliveryEventRecord.plant_mapping`` is one of three disclosures or ``None``; no two declare the
same required key set, so with an undeclared key forbidden on every model a stored dict validates
against at most one.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from tcip_annotation.text import Text
from tcip_mcp.model_registry import Producer
from tcip_mcp.pipelines.postprocessing.plant_mapping import (
    MatchRadius,
    MatchTolerance,
    PlantRegistryReference,
    UnattributedSegmentSource,
)
from tcip_mcp.pipelines.raster_source import RasterIdentity
from tcip_mcp.pipelines.schemas import TraitRevisionRef


class PlantMappingDisclosure(BaseModel):
    """The ``plant_mapping`` a phenology delivery attributed detections through.

    It holds the mapping's own identity, ``verify_mapping_inputs``'s two unverified disclosures,
    and this delivery's own unattributed-capture count scoped to its delivered dates. Every key is
    required.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    dataset_id: str
    dataset_root: str
    built_at: str
    record_sha256: str
    match_radius: dict[str, MatchRadius]
    capture_digests: dict[str, dict[str, str]]
    captures_unverified: list[str]
    plant_csvs_unverified: list[str]
    dates_delivered: list[str]
    images_unattributed: int
    images_unattributed_scope: Literal["delivered_dates"]
    plant_attribution: str


class PlantRegistryDisclosure(BaseModel):
    """The ``plant_mapping`` of an orthomosaic delivery's nearest-neighbor regime.

    It names the plant registry the regime read, the raster identity every count in the delivery
    is attributed through, the tolerance it matched detections under, this delivery's own
    unattributed-detection count, and the registry plants the raster's own frame does not picture
    at all, by name. Every key is required.
    """

    model_config = ConfigDict(extra="forbid")

    plant_registry: PlantRegistryReference
    raster_identity: RasterIdentity
    nn_tolerance_m: MatchTolerance
    detections_unattributed: int
    detections_unattributed_scope: Literal["delivered_raster"]
    plant_attribution: str
    plants_outside_raster: list[str]


class CanopySegmentsDocument(BaseModel):
    """The label document a canopy-segment delivery read its boundaries from.

    It is named by its capture and stem under the delivered bucket's dataset root and by the
    version it was read at.
    """

    model_config = ConfigDict(extra="forbid")

    capture: str
    stem: str
    sha256: str
    subject: str
    n_segments: int


class SegmentTieDisclosure(BaseModel):
    """One resolved segment-to-plant tie: the attribution claim itself, and its derived margin.

    The margin
    (:class:`~tcip_mcp.pipelines.postprocessing.segment_attribution.TiedSegment.clearance_m``) is
    what a displaced registry position would have to exceed to leave this segment.
    """

    model_config = ConfigDict(extra="forbid")

    segment_index: int
    plant_id: str
    clearance_m: float


class CanopySegmentDisclosure(BaseModel):
    """The ``plant_mapping`` of an orthomosaic delivery's canopy-segment regime.

    Names the plant registry and the raster identity, plus the canopy document this delivery read
    its boundaries from, the resolved segment-to-plant ties, the position error in meters the ties
    were held to, and the registry plants attribution excluded, by name and by reason: outside the
    raster's frame, inside no segment, within the position error of its segment's boundary, or
    inside a segment whose own detection was ambiguous
    (:data:`~tcip_mcp.pipelines.postprocessing.segment_attribution.SegmentAssignment`'s
    ``"overlapping_segments"`` source). Every key is required.
    """

    model_config = ConfigDict(extra="forbid")

    plant_registry: PlantRegistryReference
    raster_identity: RasterIdentity
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


class BucketFinding(BaseModel):
    """What the gate found for one delivered bucket, named by its dataset root and its name.

    The bucket is validated when no reason refuses it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_root: str
    bucket: str
    date: str | None
    assessment_id: str | None
    validated: bool
    reason: str | None


class Acknowledgment(BaseModel):
    """The breeder's recorded act of shipping one unvalidated result.

    It holds its id, who did it and why (both required non-empty; a blank one refuses,
    ``ValueError``), the digest of the result it was given for, and when it was recorded.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    acknowledgment_id: str
    acknowledged_by: Text
    reason: Text
    result_sha256: str
    recorded_at: str


RasterDisclosure = PlantRegistryDisclosure | CanopySegmentDisclosure
"""The disclosures of an orthomosaic delivery, which attributes through a plant registry."""

PlantMapping = PlantMappingDisclosure | RasterDisclosure
"""The disclosure a delivery's ``plant_mapping`` is: a walked mapping's or a raster one."""


class DeliveryEventRecord(TraitRevisionRef):
    """The stored per-delivery record.

    It holds the trait revision it shipped under (its
    :class:`~tcip_mcp.pipelines.schemas.TraitRevisionRef` fields), what shipped and of which kind,
    from which producer, what the gate found per bucket, and the acknowledgment it shipped under.
    """

    event_id: str
    door: str
    delivery_kind: str
    output_path: str
    output_sha256: str
    producer: Producer
    buckets: list[BucketFinding]
    scale_assessment_id: str | None
    validated: bool
    acknowledgment: Acknowledgment | None
    population: list[str]
    plant_mapping: PlantMapping | None
    produced_at: str
