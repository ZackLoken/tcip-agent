"""The general per-plant CSV door, not owned by one trait or delivery kind."""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.server import tool


@tool()
def deliver_per_plant_csv(
    project: Path,
    results: list[dict],
    output_path: str,
    delivered_phenotype: str,
    delivery_kind: str,
    plants: list[str],
    buckets: list[str],
    crop: str = "",
    pipeline_version: str = "",
    plant_mapping: str = "",
    scale_assessment_id: str | None = None,
    acknowledgment_id: str | None = None,
) -> dict:
    """The general per-plant CSV door: ``aggregate_per_plant``'s own output delivered over the
    published buckets it came from, for exactly the plants ``plants`` names.

    Every check a delivered number has to pass (the meaning, the population, the units, the one
    gate over the buckets and, for a dimensional value, the physical-scale assessment) and the
    delivered schema are
    :func:`~tcip_mcp.pipelines.postprocessing.aggregation.deliver_per_plant_aggregate`'s. An
    unvalidated delivery ships only under ``acknowledgment_id``, a breeder's recorded
    acknowledgment of exactly this result, which this door executes and never records.

    ``plant_mapping``, when named, is a claim about where the rows' plant identities came from,
    disclosed through ``plant_mapping.plant_mapping_disclosure``, which refuses a delivered plant
    the mapping assigned on no delivered date.

    Args:
        results: ``aggregate_per_plant``'s own output, one dict per plant.
        output_path: Where to write the CSV; a relative path is under the project.
        delivered_phenotype: The crop-vocabulary phenotype this CSV ships under.
        delivery_kind: ``per_plant_count_aggregate``, ``per_plant_ordinal_aggregate`` or
            ``per_plant_regression_aggregate``.
        plants: The delivery's population.
        buckets: The published buckets the values came from, one per capture date; each stands
            for the date its own record states.
        crop / pipeline_version: Written into every row.
        plant_mapping: The persisted plant mapping the rows' plant ids came from, or empty.
        scale_assessment_id: The physical-scale assessment a dimensional value rests on.
        acknowledgment_id: A breeder's recorded acknowledgment of this unvalidated result.
    """
    from tcip_mcp.buckets import by_recorded_date, read_bucket
    from tcip_mcp.pipelines.postprocessing import plant_mapping as mapping
    from tcip_mcp.pipelines.postprocessing.aggregation import deliver_per_plant_aggregate
    from tcip_mcp.pipelines.postprocessing.phenology import population
    from tcip_mcp.traits import TraitUnknownError

    try:
        delivered = [read_bucket(Path(project, b)) for b in buckets]
        wanted = population(plants)
        disclosure = (mapping.plant_mapping_disclosure(
            project, plant_mapping, by_recorded_date(delivered), wanted)
            if plant_mapping else None)
        return deliver_per_plant_aggregate(
            project, results, str(Path(project, output_path)),
            delivered_phenotype=delivered_phenotype, delivery_kind=delivery_kind,
            buckets=delivered, plants=wanted, crop=crop,
            pipeline_version=pipeline_version, door="deliver_per_plant_csv",
            plant_mapping=disclosure, scale_assessment_id=scale_assessment_id,
            acknowledgment_id=acknowledgment_id)
    except (TraitUnknownError, mapping.MappingDeliveryRefusal, ValueError) as exc:
        return {"error": str(exc)}
