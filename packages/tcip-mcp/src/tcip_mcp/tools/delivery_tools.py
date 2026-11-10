"""The general per-plant CSV door, not owned by one trait or delivery kind."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from tcip_annotation.text import Text
from tcip_mcp.pipelines.postprocessing.aggregation import (
    deliver_per_plant_aggregate,
)
from tcip_mcp.server import tool
from tcip_mcp.traits import PlantAggregateKind


class ImageObservation(BaseModel):
    """One image's measurement, the image named by its capture date and its stem.

    ``value`` is what was measured on it, ``None`` for an image yielding none. The plant it
    counts toward is the delivery's plant mapping's assignment of the image, never the caller's.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    date: Text
    stem: Text
    value: int | float | None


@tool()
def deliver_per_plant_csv(
    project: Path,
    results: list[ImageObservation],
    output_path: str,
    delivered_phenotype: str,
    delivery_kind: PlantAggregateKind,
    plants: list[str],
    dataset_root: str,
    buckets: list[str],
    value_key: str,
    plant_mapping: str,
    scale_assessment_id: str | None = None,
    acknowledgment_id: str | None = None,
    *,
    actor: str | None,
) -> dict:
    """Deliver per-image ``results`` aggregated per plant as a CSV for the plants ``plants`` names.

    The delivery is ``actor``'s.

    The general per-plant CSV door: the results are aggregated under the trait's confirmed
    ``plant_aggregation`` and delivered over the published buckets they came from. The
    aggregation, every check a delivered number has to pass (the meaning, the population, the
    units, the gate over the buckets and, for a dimensional value, the physical-scale
    assessment) and the delivered schema are
    :func:`~tcip_mcp.pipelines.postprocessing.aggregation.deliver_per_plant_aggregate`'s. An
    unvalidated delivery ships only under ``acknowledgment_id``, a breeder's recorded
    acknowledgment of exactly this result, which this door executes and never records.

    ``plant_mapping`` is resolved and verified through ``plant_mapping.plant_mapping_disclosure``,
    which refuses an unstored mapping and a delivered plant the mapping assigned on no delivered
    date. Each result counts toward the plant that mapping assigned its image, and each row's
    ``n_images``, ``plant_id_source`` and ``plant_id_distance_m_max`` are its plant's images'
    assignments'; the rows' ``plant_attribution`` is the disclosure's, and their ``crop`` the crop
    of the plant registry the mapping verified.

    Args:
        results: Per-image observations, each naming its image by its capture ``date`` and its
            ``stem`` and carrying the image's measurement as ``value`` (``null`` for an image
            yielding none); a row naming any other key is refused. Every image the delivery reads
            of a plant in ``plants`` (attributed by the mapping, with a document in its date's
            bucket, on disk) needs exactly one result, and results naming any other image are
            refused in one answer naming every such image under its reason and that reason's
            next step.
        output_path: Where to write the CSV.
        delivered_phenotype: The crop-vocabulary phenotype this CSV ships under.
        delivery_kind: A kind whose per-plant value is the trait's ``plant_aggregation`` over
            the plant's records.
        plants: The delivery's population.
        dataset_root: The dataset root the buckets are published under.
        buckets: The names of the published buckets the values came from, one per capture date;
            each stands for the date its own record states.
        value_key: The quantity every value is, one the operationalization covers.
        plant_mapping: The persisted plant mapping that attributes the images to plants.
        scale_assessment_id: The physical-scale assessment a dimensional value rests on.
        acknowledgment_id: A breeder's recorded acknowledgment of this unvalidated result.

    """
    from tcip_mcp.buckets import by_recorded_date, read_bucket
    from tcip_mcp.pipelines.postprocessing import plant_mapping as mapping
    from tcip_mcp.pipelines.postprocessing.phenology import population
    from tcip_mcp.registry_paths import located
    from tcip_mcp.traits import TraitUnknownError

    try:
        root = located(dataset_root, project)
        delivered = [read_bucket(root, b) for b in buckets]
        wanted = population(plants)
        resolved = mapping.plant_mapping_disclosure(
            project, plant_mapping, by_recorded_date(delivered), wanted
        )
        return deliver_per_plant_aggregate(
            project,
            results,
            str(located(output_path, project)),
            value_key=value_key,
            delivered_phenotype=delivered_phenotype,
            delivery_kind=delivery_kind,
            buckets=delivered,
            plants=wanted,
            mapping=resolved,
            door="deliver_per_plant_csv",
            scale_assessment_id=scale_assessment_id,
            acknowledgment_id=acknowledgment_id,
            actor=actor,
        ).model_dump(mode="json")
    except (
        TraitUnknownError,
        mapping.MappingDeliveryError,
        ValueError,
    ) as exc:
        return {"error": str(exc)}
