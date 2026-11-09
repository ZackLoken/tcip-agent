"""The per-plant delivery: one value per plant, and per-image records aggregated into one.

Every per-plant delivery writes its values under what its trait's confirmed revision states
(:func:`plant_value_meaning`, :func:`deliver_plant_values`). Per-image records are aggregated
under the statistic that revision names among :data:`PLANT_AGGREGATIONS`
(:func:`deliver_per_plant_aggregate`).

``delivered_phenotype`` is a crop-vocabulary delivered-phenotype name (``stem_count`` is one, as an
example rather than as the shape), which is what the CSV column and the unit cross-check are about.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from tcip_annotation.text import Crop, Text

if TYPE_CHECKING:
    from tcip_mcp.buckets import Bucket
    from tcip_mcp.delivery import PerPlantDelivered
    from tcip_mcp.pipelines.delivery_events_schema import PlantMapping
    from tcip_mcp.traits import (
        PlantAggregateKind,
        PlantAggregation,
        TraitRevision,
    )

logger = logging.getLogger(__name__)


class PlantObservation(BaseModel):
    """One observation of one plant under the delivery's attribution, a per-plant delivery's row.

    ``plant_id`` is the plant the delivery's attribution assigned the observation to, never one
    read off a file name; ``value`` is what was observed, ``None`` for an observation that yields
    no value. ``source`` and ``distance_m`` are the plant assignment's own, when it carries them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    plant_id: Text
    value: int | float | None
    source: str | None = None
    distance_m: float | None = None


def _per_plant(
    records: list[PlantObservation], *, plant_aggregation: PlantAggregation
) -> list[dict]:
    """Return per-plant summaries of per-image ``records``, one per ``plant_id``, sorted by it.

    Each summary holds the plant, its record count as ``n_images``, and as ``value`` the
    :data:`PLANT_AGGREGATIONS` statistic ``plant_aggregation`` names over the records' values (a
    ``None`` value is a missing observation; ``None`` when every record of the plant misses one).

    Where records carry an assignment's ``source``/``distance_m``, they are summarized per plant as
    ``plant_id_source`` (``"mixed"`` where the plant's images disagree) and
    ``plant_id_distance_m_max``.
    """
    groups: dict[str, list[PlantObservation]] = defaultdict(list)
    for r in records:
        groups[r.plant_id].append(r)

    statistic = PLANT_AGGREGATIONS[plant_aggregation]
    results = []
    for plant_id, items in sorted(groups.items()):
        values = [r.value for r in items if r.value is not None]
        summary: dict[str, Any] = {
            "plant_id": plant_id,
            "n_images": len(items),
            "value": statistic(values) if values else None,
        }
        sources = {r.source for r in items if r.source is not None}
        if sources:
            summary["plant_id_source"] = (
                sources.pop() if len(sources) == 1 else "mixed"
            )
        distances = [r.distance_m for r in items if r.distance_m is not None]
        if distances:
            summary["plant_id_distance_m_max"] = max(distances)
        results.append(summary)

    return results


PLANT_AGGREGATIONS: dict[PlantAggregation, Callable[[list], Any]] = {
    "median": statistics.median,
    "mean": statistics.mean,
    "mode": statistics.mode,
    "sum": sum,
}
"""The per-plant statistics by the name a trait entry's ``plant_aggregation`` states."""


def _resolve_units(
    delivered_phenotype: str,
    value_key: str,
    scalar: bool,
) -> tuple[str, str | None, str | None]:
    """Return ``(display_unit, linear_basis, declared_unit)`` for a delivery's values.

    The first two are the units implied by the delivered values' ``value_key``, the third
    crops.yml's declared unit for ``delivered_phenotype`` or ``None``. Under a ``scalar`` head's
    delivery, a value_key with no unit suffix at all takes the declared unit; one that explicitly
    ends in ``_px`` (or is bare ``"px"``) never does. A value_key that implies a physical unit is
    cross-checked against the declared one either way, and a mismatch raises ``ValueError``.
    ``display_unit`` is squared for an area; ``linear_basis`` is always linear.
    """
    from tcip_mcp.pipelines.measurement.mask_geometry import (
        is_pixel_space_key,
        unit_from_value_key,
    )
    from tcip_mcp.traits import crops_units

    declared = crops_units().get(delivered_phenotype)
    pair = unit_from_value_key(value_key)
    if pair is None:
        if (
            scalar
            and declared is not None
            and not is_pixel_space_key(value_key)
        ):
            return declared, declared, declared
        return "", None, declared
    display, linear_basis = pair
    if declared is not None and linear_basis != declared:
        raise ValueError(
            f"delivered phenotype {delivered_phenotype!r} is declared units={declared!r} in "
            f"crops.yml, but the aggregated values' own key implies {linear_basis!r}; refusing to "
            "ship a mismatched unit label rather than guessing which one is right."
        )
    return display, linear_basis, declared


@dataclass(frozen=True)
class PlantValueMeaning:
    """What one per-plant delivery's values are, as its trait's confirmed revision states it.

    It holds the delivery's kind, phenotype and value key, the delivered plants' crop, the units
    the values are written in and their linear basis, and the revision the delivery ships under.
    """

    delivery_kind: str
    delivered_phenotype: str
    value_key: str
    crop: Crop
    units: str
    linear_basis: str | None
    revision: TraitRevision


def plant_value_meaning(
    project: Path,
    *,
    delivery_kind: str,
    delivered_phenotype: str,
    value_key: str,
    crop: Crop,
) -> PlantValueMeaning:
    """Return what a ``delivery_kind`` delivery's ``value_key`` values of a phenotype are.

    The units come from ``value_key``, cross-checked against crops.yml; a count kind's value key
    must itself imply the unit crops.yml declares for the phenotype, if any. The phenotype,
    ``delivered_phenotype``, resolves to the one trait whose latest confirmed revision delivers
    it, and that revision's ``delivery_kind`` operationalization must cover it and ``value_key``
    (:func:`~tcip_mcp.operationalization.confirmed_revision`); a revision delivering a phenotype
    crops.yml does not name for ``crop``, the delivered plants' registry's crop, refuses
    (:func:`~tcip_mcp.traits.require_delivered_for_crop`). Each refusal is a ``ValueError``.
    """
    from tcip_mcp.operationalization import confirmed_revision
    from tcip_mcp.traits import COUNT_KINDS, require_delivered_for_crop

    counted = delivery_kind in COUNT_KINDS
    units, linear_basis, declared = _resolve_units(
        delivered_phenotype, value_key, scalar=not counted
    )
    if counted and declared and not units:
        raise ValueError(
            f"delivered phenotype {delivered_phenotype!r} is declared "
            f"units={declared!r}, and the value key {value_key!r} implies no physical unit."
        )
    revision = confirmed_revision(
        delivery_kind,
        project=project,
        delivered_phenotype=delivered_phenotype,
        value_key=value_key,
    )
    require_delivered_for_crop(revision.entry, crop)
    return PlantValueMeaning(
        delivery_kind=delivery_kind,
        delivered_phenotype=delivered_phenotype,
        value_key=value_key,
        crop=crop,
        units=units,
        linear_basis=linear_basis,
        revision=revision,
    )


def deliver_plant_values(
    project: Path,
    meaning: PlantValueMeaning,
    values: Mapping[str, int | float],
    output_path: str,
    *,
    door_columns: Mapping[str, Mapping[str, object]],
    buckets: Sequence[Bucket],
    plants: list[str],
    door: str,
    plant_mapping: PlantMapping,
    scale_assessment_id: str | None = None,
    acknowledgment_id: str | None = None,
    actor: str | None,
) -> PerPlantDelivered:
    """Deliver one value per plant, ``values`` by plant id, as a CSV at ``output_path``.

    The values are what ``meaning`` states, each plant attributed at the granularity
    ``plant_mapping.plant_attribution`` names; the CSV is a delivery by ``actor`` of exactly the
    population ``plants`` (:func:`~tcip_mcp.pipelines.postprocessing.phenology.population`'s
    list), refusing (``ValueError``) values of any other plants, over the published ``buckets``
    the values came from. The buckets, the physical-scale assessment ``scale_assessment_id`` and
    the unit clear :func:`~tcip_mcp.delivery.gate` under the meaning's revision, the recorded
    acknowledgment ``acknowledgment_id`` shipping them unvalidated when it does not validate them.

    Each row carries the plant, the crop, its value, the units and value key, its granularity of
    attribution, the cells ``door_columns`` holds (each column only the door writes, by name, to
    its cell per plant), and the delivery's own cells, written with the delivery's one event under
    ``door`` with its population and ``plant_mapping`` disclosure
    (:func:`~tcip_mcp.delivery.deliver_csv`); returns what was delivered.
    """
    from tcip_mcp.delivery import (
        DELIVERY_COLUMNS,
        PerPlantDelivered,
        Result,
        deliver_csv,
        gate,
    )

    if sorted(values) != sorted(plants):
        raise ValueError(
            f"the records cover plants {sorted(values)} and the population is "
            f"{sorted(plants)}: a delivery ships exactly its population, one row each."
        )
    result = Result(
        [
            "plant_id",
            "crop",
            "delivered_phenotype",
            "value",
            "units",
            "value_key",
            "plant_attribution",
            *door_columns,
            *DELIVERY_COLUMNS,
        ],
        [
            {
                **{
                    column: cells[plant]
                    for column, cells in door_columns.items()
                },
                "plant_id": plant,
                "crop": meaning.crop,
                "delivered_phenotype": meaning.delivered_phenotype,
                "value": values[plant],
                "units": meaning.units,
                "value_key": meaning.value_key,
                "plant_attribution": plant_mapping.plant_attribution,
            }
            for plant in plants
        ],
        population=plants,
        plant_mapping=plant_mapping,
    )
    clearance = gate(
        project,
        buckets,
        delivery_kind=meaning.delivery_kind,
        revision=meaning.revision,
        result=result,
        acknowledgment_id=acknowledgment_id,
        scale_assessment_id=scale_assessment_id,
        unit=meaning.linear_basis,
    )
    delivered = deliver_csv(
        project,
        output_path,
        result,
        clearance=clearance,
        revision=meaning.revision,
        door=door,
        delivery_kind=meaning.delivery_kind,
        actor=actor,
    )
    return PerPlantDelivered(**dict(delivered), n_plants=len(plants))


PER_IMAGE_COLUMNS = ("n_images", "plant_id_source", "plant_id_distance_m_max")
"""The cells only a per-image aggregate writes: each plant's image count and the worst of its
images' plant assignments (:func:`_per_plant`), blank where the records carried none."""


def deliver_per_plant_aggregate(
    project: Path,
    records: list[PlantObservation],
    output_path: str,
    *,
    value_key: str,
    delivered_phenotype: str,
    delivery_kind: PlantAggregateKind,
    buckets: Sequence[Bucket],
    plants: list[str],
    crop: Crop,
    door: str,
    plant_mapping: PlantMapping,
    scale_assessment_id: str | None = None,
    acknowledgment_id: str | None = None,
    actor: str | None,
) -> PerPlantDelivered:
    """Deliver the per-plant aggregate of per-image ``records`` as a CSV at ``output_path``.

    Each record's value is a ``value_key`` quantity of ``delivered_phenotype`` for ``crop``, the
    meaning :func:`plant_value_meaning` resolves for ``delivery_kind`` and refuses as it does. The
    records are aggregated per plant under that revision's ``plant_aggregation``
    (:func:`_per_plant`); a plant whose every record carries no value refuses (``ValueError``).
    The aggregates, each plant's :data:`PER_IMAGE_COLUMNS` cells beside them, are delivered for
    the population ``plants`` over ``buckets`` under ``door`` by ``actor`` with the
    ``plant_mapping`` disclosure, ``scale_assessment_id`` and ``acknowledgment_id``, as
    :func:`deliver_plant_values` delivers them; returns what was delivered.
    """
    meaning = plant_value_meaning(
        project,
        delivery_kind=delivery_kind,
        delivered_phenotype=delivered_phenotype,
        value_key=value_key,
        crop=crop,
    )
    by_plant = {
        s["plant_id"]: s
        for s in _per_plant(
            records, plant_aggregation=meaning.revision.entry.plant_aggregation
        )
    }
    none_valued = [p for p, s in by_plant.items() if s["value"] is None]
    if none_valued:
        raise ValueError(
            f"plant(s) {none_valued} carry no {delivered_phenotype!r} observation: a "
            "missing measurement never ships as an empty cell; leave them out of "
            "the population."
        )
    return deliver_plant_values(
        project,
        meaning,
        {p: s["value"] for p, s in by_plant.items()},
        output_path,
        door_columns={
            column: {p: s.get(column, "") for p, s in by_plant.items()}
            for column in PER_IMAGE_COLUMNS
        },
        buckets=buckets,
        plants=plants,
        door=door,
        plant_mapping=plant_mapping,
        scale_assessment_id=scale_assessment_id,
        acknowledgment_id=acknowledgment_id,
        actor=actor,
    )
