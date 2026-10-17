"""The per-plant delivery of per-image records aggregated per plant.

Records are aggregated under the statistic the trait's confirmed revision names among
:data:`PLANT_AGGREGATIONS`.

``delivered_phenotype`` is a crop-vocabulary delivered-phenotype name (``stem_count`` is one, as an
example rather than as the shape), which is what the CSV column and the unit cross-check are about.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tcip_mcp.buckets import Bucket
    from tcip_mcp.traits import PlantAggregation

logger = logging.getLogger(__name__)


def _per_plant(
    records: list[dict], *, plant_aggregation: PlantAggregation, value_key: str
) -> list[dict]:
    """Return per-plant summaries of per-image ``records``, one per ``plant_id``, sorted by it.

    Each summary holds the plant, its record count as ``n_images``, and as ``value`` the
    :data:`PLANT_AGGREGATIONS` statistic ``plant_aggregation`` names over the records'
    ``value_key`` (a record without it is a missing observation; ``None`` when every record of the
    plant misses it).

    A record carrying no ``plant_id`` refuses (``ValueError``), naming ``build_plant_mapping``.
    Where records carry an assignment's ``source``/``distance_m``, they are summarized per plant as
    ``plant_id_source`` (``"mixed"`` where the plant's images disagree) and
    ``plant_id_distance_m_max``.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("plant_id") in (None, ""):
            raise ValueError(
                f"record {r.get('image', '<no image key>')!r} carries no plant_id. Plant "
                "identity is never guessed from a filename: attach the plant each image was "
                "assigned by tcip_mcp.pipelines.postprocessing.plant_mapping.build_plant_mapping."
            )
        groups[str(r["plant_id"])].append(r)

    statistic = PLANT_AGGREGATIONS[plant_aggregation]
    results = []
    for plant_id, items in sorted(groups.items()):
        values = [r[value_key] for r in items if value_key in r]
        summary: dict[str, Any] = {
            "plant_id": plant_id,
            "n_images": len(items),
            "value": statistic(values) if values else None,
        }
        sources = {r["source"] for r in items if r.get("source") is not None}
        if sources:
            summary["plant_id_source"] = (
                sources.pop() if len(sources) == 1 else "mixed"
            )
        distances = [
            r["distance_m"] for r in items if r.get("distance_m") is not None
        ]
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


def deliver_per_plant_aggregate(
    project: Path,
    records: list[dict],
    output_path: str,
    *,
    value_key: str,
    delivered_phenotype: str,
    delivery_kind: str,
    buckets: Sequence[Bucket],
    plants: list[str],
    crop: str = "",
    pipeline_version: str = "",
    door: str,
    plant_mapping: dict | None = None,
    scale_assessment_id: str | None = None,
    acknowledgment_id: str | None = None,
    actor: str | None,
) -> dict:
    """Deliver the per-plant aggregate of per-image ``records`` as a CSV at ``output_path``.

    Each record carries its ``plant_id`` and its value under ``value_key``; the CSV is a
    ``delivery_kind`` delivery by ``actor``, for exactly the population ``plants``
    (:func:`~tcip_mcp.pipelines.postprocessing.phenology.population`'s list), over the published
    ``buckets`` the values came from.

    ``delivered_phenotype`` resolves to the one trait whose latest confirmed revision delivers it,
    and that revision's ``delivery_kind`` operationalization must cover it and ``value_key``. The
    records are aggregated per plant under the revision's ``plant_aggregation``
    (:func:`_per_plant`); the plants must be exactly the population, every one carrying a value,
    and the records agree on ``plant_attribution``. The units come from ``value_key``,
    cross-checked against crops.yml. The buckets, the physical-scale assessment
    ``scale_assessment_id`` and the unit clear :func:`~tcip_mcp.delivery.gate`, the recorded
    acknowledgment ``acknowledgment_id`` shipping them unvalidated when it does not validate them.

    Each row carries the plant, its value and unit, its provenance of attribution, and the
    delivery's own cells, written with the delivery's one event under ``door`` with its
    population and ``plant_mapping`` disclosure (:func:`~tcip_mcp.delivery.deliver_csv`); returns
    what was delivered.
    """
    from tcip_mcp.delivery import DELIVERY_COLUMNS, Result, deliver_csv, gate
    from tcip_mcp.operationalization import confirmed_revision
    from tcip_mcp.traits import PER_PLANT_COUNT_AGGREGATE

    units, linear_basis, declared = _resolve_units(
        delivered_phenotype,
        value_key,
        scalar=delivery_kind != PER_PLANT_COUNT_AGGREGATE,
    )
    if delivery_kind == PER_PLANT_COUNT_AGGREGATE and declared and not units:
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
    by_plant = {
        s["plant_id"]: s
        for s in _per_plant(
            records,
            plant_aggregation=revision.entry.plant_aggregation,
            value_key=value_key,
        )
    }
    if sorted(by_plant) != sorted(plants):
        raise ValueError(
            f"the records cover plants {sorted(by_plant)} and the population is "
            f"{sorted(plants)}: a delivery ships exactly its population, one row each."
        )
    none_valued = [p for p in plants if by_plant[p]["value"] is None]
    if none_valued:
        raise ValueError(
            f"plant(s) {none_valued} carry no {delivered_phenotype!r} observation: a "
            "missing measurement never ships as an empty cell; leave them out of "
            "the population."
        )
    attributions = {r.get("plant_attribution") for r in records}
    if len(attributions) > 1 or None in attributions:
        raise ValueError(
            "the delivered records disagree on or omit plant_attribution "
            f"({sorted(str(a) for a in attributions)}); every record must state the same one."
        )
    (attribution,) = attributions
    result = Result(
        [
            "plant_id",
            "crop",
            "delivered_phenotype",
            "value",
            "units",
            "value_key",
            "n_images",
            "pipeline_version",
            "plant_id_source",
            "plant_attribution",
            "plant_id_distance_m_max",
            *DELIVERY_COLUMNS,
        ],
        [
            {
                "plant_id": plant,
                "crop": crop,
                "delivered_phenotype": delivered_phenotype,
                "value": by_plant[plant]["value"],
                "units": units,
                "value_key": value_key,
                "n_images": by_plant[plant]["n_images"],
                "pipeline_version": pipeline_version,
                "plant_id_source": by_plant[plant].get("plant_id_source", ""),
                "plant_attribution": attribution,
                "plant_id_distance_m_max": by_plant[plant].get(
                    "plant_id_distance_m_max", ""
                ),
            }
            for plant in plants
        ],
        population=plants,
        plant_mapping=plant_mapping,
    )
    clearance = gate(
        project,
        buckets,
        delivery_kind=delivery_kind,
        revision=revision,
        result=result,
        acknowledgment_id=acknowledgment_id,
        scale_assessment_id=scale_assessment_id,
        unit=linear_basis,
    )
    delivered = deliver_csv(
        project,
        output_path,
        result,
        clearance=clearance,
        revision=revision,
        door=door,
        delivery_kind=delivery_kind,
        actor=actor,
    )
    return {**delivered, "n_plants": len(plants)}
