"""The per-plant delivery: one value per plant, and per-image values aggregated into one.

Every per-plant delivery writes its values under what its trait's confirmed revision states
(:func:`plant_value_meaning`, :func:`deliver_plant_values`). Per-image values are attributed to
plants by the delivery's plant mapping and aggregated under the statistic that revision names
among :data:`PLANT_AGGREGATIONS` (:func:`deliver_per_plant_aggregate`).

``delivered_phenotype`` is a crop-vocabulary delivered-phenotype name (``stem_count`` is one, as an
example rather than as the shape), which is what the CSV column and the unit cross-check are about.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tcip_annotation.text import Crop

if TYPE_CHECKING:
    from tcip_mcp.buckets import Bucket
    from tcip_mcp.delivery import PerPlantDelivered
    from tcip_mcp.pipelines.delivery_events_schema import PlantMapping
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        Assignment,
        MappingResolution,
    )
    from tcip_mcp.tools.delivery_tools import ImageObservation
    from tcip_mcp.traits import (
        PlantAggregateKind,
        PlantAggregation,
        TraitRevision,
    )

logger = logging.getLogger(__name__)

type Observed = tuple[Assignment, int | float | None]
"""One image's assignment to a plant and the value measured on it."""


def _per_plant(
    observed: Sequence[Observed], *, plant_aggregation: PlantAggregation
) -> list[dict]:
    """Return per-plant summaries of ``observed`` images, one per plant, sorted by it.

    Each image counts toward the plant its assignment names. Each summary holds the plant, its
    image count as ``n_images``, and as ``value`` the :data:`PLANT_AGGREGATIONS` statistic
    ``plant_aggregation`` names over the images' values (a ``None`` value is a missing
    observation; ``None`` when every image of the plant misses one). The images' assignments are
    summarized as ``plant_id_source``, their one ``source`` or ``"mixed"`` where they disagree,
    and ``plant_id_distance_m_max``, the largest ``distance_m`` among them.
    """
    groups: dict[str | None, list[Observed]] = {}
    for a, v in observed:
        groups.setdefault(a.plant_id, []).append((a, v))

    statistic = PLANT_AGGREGATIONS[plant_aggregation]
    results = []
    for plant_id in sorted(groups, key=str):
        items = groups[plant_id]
        values = [v for _, v in items if v is not None]
        sources = {a.source for a, _ in items}
        distances = [
            a.distance_m for a, _ in items if a.distance_m is not None
        ]
        results.append(
            {
                "plant_id": plant_id,
                "n_images": len(items),
                "value": statistic(values) if values else None,
                "plant_id_source": sources.pop()
                if len(sources) == 1
                else "mixed",
                "plant_id_distance_m_max": max(distances, default=""),
            }
        )
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
    it, and that revision's ``delivery_kind`` operationalization must cover it and, for a kind
    whose operationalization bounds its value keys, ``value_key``
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
images' plant assignments (:func:`_per_plant`)."""


def _attributed(
    results: Sequence[ImageObservation],
    mapping: MappingResolution,
    plants: Sequence[str],
) -> list[Observed]:
    """Return each of ``results`` with the assignment ``mapping`` attributes its image to.

    Each result's image is looked up by its date and stem in the mapping's build
    (:meth:`~tcip_mcp.pipelines.postprocessing.plant_mapping.MappingBuild.attributed_assignment`,
    refusing as it does). Refuses (``ValueError``), naming the image, a result whose date is not a
    delivered date, whose image's plant is not one of ``plants``, whose image the mapping's
    verification did not read (no document in the delivered bucket, or no longer on disk), or
    whose image another result names too; and, naming each, a read image of a plant of ``plants``
    (:func:`~tcip_mcp.pipelines.postprocessing.plant_mapping.read_images`) no result names.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import read_images

    delivered = mapping.disclosure.dates_delivered
    read_stems = mapping.verified.read_stems
    seen: set[tuple[str, str]] = set()
    observed: list[Observed] = []
    for result in results:
        image = f"image {result.stem!r} of {result.date}"
        if result.date not in delivered:
            raise ValueError(
                f"{image}: {result.date} is not a delivered date ({delivered}); deliver its "
                "bucket or drop the result."
            )
        assignment = mapping.build.attributed_assignment(
            result.date, result.stem
        )
        if assignment.plant_id not in plants:
            raise ValueError(
                f"{image} is of plant {assignment.plant_id!r}, which the population "
                f"{list(plants)} does not name; deliver that plant or drop the result."
            )
        if result.stem not in read_stems[result.date]:
            raise ValueError(
                f"{image} was not read by the delivery: the delivered bucket holds no "
                "prediction document for it, or it is no longer on disk; drop the result."
            )
        if (result.date, result.stem) in seen:
            raise ValueError(
                f"{image} is named by two results; each image is delivered once."
            )
        seen.add((result.date, result.stem))
        observed.append((assignment, result.value))
    uncovered = sorted(
        f"{date}/{stem}"
        for images in read_images(mapping, plants).values()
        for date, stem in images
        if (date, stem) not in seen
    )
    if uncovered:
        raise ValueError(
            f"image(s) {uncovered} of the delivered plants were read by the delivery and no "
            "result names them: every image the delivery read counts toward its plant."
        )
    return observed


def deliver_per_plant_aggregate(
    project: Path,
    results: Sequence[ImageObservation],
    output_path: str,
    *,
    value_key: str,
    delivered_phenotype: str,
    delivery_kind: PlantAggregateKind,
    buckets: Sequence[Bucket],
    plants: list[str],
    mapping: MappingResolution,
    door: str,
    scale_assessment_id: str | None = None,
    acknowledgment_id: str | None = None,
    actor: str | None,
) -> PerPlantDelivered:
    """Deliver the per-plant aggregate of per-image ``results`` as a CSV at ``output_path``.

    Each result's value is a ``value_key`` quantity of ``delivered_phenotype`` for the crop of
    ``mapping``'s verified registry, the meaning :func:`plant_value_meaning` resolves for
    ``delivery_kind`` and refuses as it does. Each result counts toward the plant ``mapping``
    attributes its image to, refusing as :func:`_attributed` does, and the results are aggregated
    per plant under that revision's ``plant_aggregation`` (:func:`_per_plant`); a plant whose
    every image carries no value refuses (``ValueError``). The aggregates, each plant's
    :data:`PER_IMAGE_COLUMNS` cells beside them, are delivered for the population ``plants`` over
    ``buckets`` under ``door`` by ``actor`` with ``mapping``'s disclosure, ``scale_assessment_id``
    and ``acknowledgment_id``, as :func:`deliver_plant_values` delivers them; returns what was
    delivered.
    """
    meaning = plant_value_meaning(
        project,
        delivery_kind=delivery_kind,
        delivered_phenotype=delivered_phenotype,
        value_key=value_key,
        crop=mapping.verified.crop,
    )
    by_plant = {
        s["plant_id"]: s
        for s in _per_plant(
            _attributed(results, mapping, plants),
            plant_aggregation=meaning.revision.entry.plant_aggregation,
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
            column: {p: s[column] for p, s in by_plant.items()}
            for column in PER_IMAGE_COLUMNS
        },
        buckets=buckets,
        plants=plants,
        door=door,
        plant_mapping=mapping.disclosure,
        scale_assessment_id=scale_assessment_id,
        acknowledgment_id=acknowledgment_id,
        actor=actor,
    )
