"""The declared shapes of a delivery event record and of the acknowledgment it may ship under.

``DeliveryEventRecord.plant_mapping`` is one of three disclosures, told apart by its ``kind``, or
``None``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from tcip_annotation.text import Text
from tcip_mcp.model_registry import Producer
from tcip_mcp.pipelines.postprocessing.plant_mapping import (
    MatchTolerance,
    PlantMappingDisclosure,
    PlantRegistryReference,
    UnattributedSegmentSource,
)
from tcip_mcp.pipelines.postprocessing.segment_attribution import PositionError
from tcip_mcp.pipelines.raster_source import RasterIdentity
from tcip_mcp.pipelines.schemas import TraitRevisionRef


class PlantRegistryDisclosure(BaseModel):
    """The ``plant_mapping`` of an orthomosaic delivery's nearest-neighbor regime.

    It names the plant registry the regime read, the raster identity every count in the delivery
    is attributed through, the tolerance it matched detections under, this delivery's own count
    of detections attributed to no plant over the delivered raster, and the registry plants the
    raster's own frame does not picture at all, by name. Every key is required.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["plant_registry"] = "plant_registry"
    plant_registry: PlantRegistryReference
    raster_identity: RasterIdentity
    nn_tolerance_m: MatchTolerance
    detections_unattributed: int
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

    kind: Literal["canopy_segments"] = "canopy_segments"
    plant_registry: PlantRegistryReference
    raster_identity: RasterIdentity
    canopy_segments: CanopySegmentsDocument
    segment_ties: list[SegmentTieDisclosure]
    position_error_m: PositionError
    """The registry positions' error bound in meters the ties were held to, the project's
    stated value."""
    segments_without_plant: int
    plants_outside_raster: list[str]
    plants_without_segment: list[str]
    plants_within_position_error: list[str]
    plants_with_ambiguous_detections: list[str]
    detections_unattributed_by_source: dict[UnattributedSegmentSource, int]
    """The unattributed detections over the delivered raster, by the segment source that left each
    one unattributed."""
    plant_attribution: str


class ValidatedBucket(BaseModel):
    """A delivered bucket the assessment it was published under validates.

    The bucket is named by its dataset root, its name and its capture date (``None`` for an
    undated bucket).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["validated"] = "validated"
    dataset_root: str
    bucket: str
    date: str | None
    assessment_id: str


class UnvalidatedBucket(BaseModel):
    """A delivered bucket nothing validates, and why.

    The bucket is named as a :class:`ValidatedBucket` is, beside the assessment it was published
    under, ``None`` when it names none.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["unvalidated"] = "unvalidated"
    dataset_root: str
    bucket: str
    date: str | None
    assessment_id: str | None
    reason: str


type BucketFinding = Annotated[
    ValidatedBucket | UnvalidatedBucket, Field(discriminator="kind")
]
"""What the gate found for one delivered bucket, told apart by its ``kind``."""


class AnsweredScale(BaseModel):
    """The physical-scale assessment a delivery named, which answers for its unit and buckets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["answered"] = "answered"
    assessment_id: str


class UnansweredScale(BaseModel):
    """The physical-scale assessment a delivery named, which answers for nothing, and why.

    ``reason`` is the sentence the gate refused the delivery with for this assessment.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["unanswered"] = "unanswered"
    assessment_id: str
    reason: str


type ScaleRead = Annotated[
    AnsweredScale | UnansweredScale, Field(discriminator="kind")
]
"""What the gate found for the physical-scale assessment a delivery named, told apart by its
``kind``."""


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
    recorded_at: AwareDatetime


class ShippedValidated(BaseModel):
    """A delivery every finding of which the gate validated, shipped under no acknowledgment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["validated"] = "validated"


class ShippedAcknowledged(BaseModel):
    """A delivery shipped unvalidated under a breeder's acknowledgment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["acknowledged"] = "acknowledged"
    acknowledgment: Acknowledgment


type Shipped = Annotated[
    ShippedValidated | ShippedAcknowledged, Field(discriminator="kind")
]
"""How a delivery shipped, validated or under an acknowledgment, told apart by its ``kind``."""


type PlantMapping = Annotated[
    PlantMappingDisclosure | PlantRegistryDisclosure | CanopySegmentDisclosure,
    Field(discriminator="kind"),
]
"""The disclosure a delivery's ``plant_mapping`` is, a walked mapping's or one of an orthomosaic
delivery's two regimes, told apart by its ``kind``."""


class DeliveryEventRecord(TraitRevisionRef):
    """The stored per-delivery record.

    It holds the trait revision it shipped under (its
    :class:`~tcip_mcp.pipelines.schemas.TraitRevisionRef` fields), what shipped and of which kind,
    from which producer, what the gate found per bucket and for the physical-scale assessment the
    delivery named (:data:`ScaleRead`, ``None`` when it named none), and how it shipped
    (:data:`Shipped`).
    """

    event_id: str
    door: str
    delivery_kind: str
    output_path: str
    output_sha256: str
    producer: Producer
    buckets: list[BucketFinding]
    scale: ScaleRead | None
    shipped: Shipped
    population: list[str]
    plant_mapping: PlantMapping | None
    produced_at: AwareDatetime
