"""The general per-plant CSV door, not owned by one trait or delivery kind."""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.pipelines.postprocessing.aggregation import (
    PlantObservation,
    deliver_per_plant_aggregate,
)
from tcip_mcp.server import tool


@tool()
def deliver_per_plant_csv(
    project: Path,
    results: list[PlantObservation],
    output_path: str,
    delivered_phenotype: str,
    delivery_kind: str,
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

    ``plant_mapping`` is the claim about where the rows' plant identities came from, disclosed
    through ``plant_mapping.plant_mapping_disclosure``, which refuses an unstored mapping and a
    delivered plant the mapping assigned on no delivered date; the rows' ``plant_attribution`` is
    that disclosure's, and their ``crop`` the crop of the plant registry the mapping verified.

    Args:
        results: Per-image observations, each the plant ``build_plant_mapping`` assigned its
            image as ``plant_id`` and the image's measurement as ``value`` (``null`` for an image
            yielding none), with the assignment's ``source`` and ``distance_m`` when it carries
            them; a row naming any other key, or no plant, is refused.
        output_path: Where to write the CSV.
        delivered_phenotype: The crop-vocabulary phenotype this CSV ships under.
        delivery_kind: ``per_plant_count_aggregate``, ``per_plant_ordinal_aggregate`` or
            ``per_plant_regression_aggregate``.
        plants: The delivery's population.
        dataset_root: The dataset root the buckets are published under.
        buckets: The names of the published buckets the values came from, one per capture date;
            each stands for the date its own record states.
        value_key: The quantity every value is, one the operationalization covers.
        plant_mapping: The persisted plant mapping the rows' plant ids came from.
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
            crop=resolved.verified.crop,
            door="deliver_per_plant_csv",
            plant_mapping=resolved.disclosure,
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
