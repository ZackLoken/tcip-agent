"""Per-plant aggregation, temporal/spatial aggregation of per-image results.

Aggregation strategies (per-plant summary of a per-image value):
  - count:    Median detection count across images per plant
  - mode:     Most frequent value (ordinal traits)
  - mean:     Arithmetic mean (continuous traits)
  - sum:      Sum of values (area traits)

A trait's phenology milestones (percentile-crossing dates of its positive-state fraction) are
``postprocessing/phenology.py``'s.

Usage:
    results = aggregate_per_plant(image_results, strategy="count")
    deliver_per_plant_aggregate(project, results, "output.csv", delivered_phenotype="stem_count", ...)

``delivered_phenotype`` is a crop-vocabulary delivered-phenotype name (``stem_count`` is one, as an
example rather than as the shape), which is what the CSV column and the unit cross-check are about.
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tcip_mcp.buckets import Bucket

logger = logging.getLogger(__name__)


def aggregate_per_plant(
    image_results: list[dict],
    strategy: str = "count",
    plant_id_key: str = "plant_id",
    value_key: str = "count",
    plant_id_fn: Any = None,
) -> list[dict]:
    """Aggregate per-image results to per-plant summaries.

    Every record's ``plant_id`` must come from an explicit ``plant_id_key`` value or
    ``plant_id_fn(image)``; a record for which neither resolves raises, naming
    ``build_plant_mapping`` (``tcip_mcp.pipelines.postprocessing.plant_mapping``).

    When a record carries an assignment's ``source``/``distance_m`` (as ``build_plant_mapping``'s
    ``Assignment`` rows do), they are summarized per plant as ``plant_id_source`` (``"mixed"``
    where the plant's images disagree) and ``plant_id_distance_m_max``. A record's ``plant_attribution`` (the
    granularity objects were attributed to plants at, e.g. ``plant_mapping.MappingBuild``'s
    ``"image"`` or ``orthomosaic_mapping.DetectionAssignment``'s ``"detection"``) is carried onto
    the summary; a plant whose own images disagree on it refuses (see :func:`_agreed`).

    Args:
        image_results: List of dicts, each with at least an 'image' key, a plant_id_key or a value
            plant_id_fn can resolve, and a value field (e.g., 'count', 'class', 'value').
        strategy: Aggregation strategy, 'count', 'mode', 'mean', 'sum'.
        plant_id_key: Key in each result dict for plant identification.
        value_key: Key in each result dict for the value to aggregate.
        plant_id_fn: Callable to derive plant_id from the image path when plant_id_key is absent.

    Returns:
        List of per-plant summary dicts.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in image_results:
        if plant_id_key in r and r[plant_id_key] not in (None, ""):
            pid = r[plant_id_key]
        elif plant_id_fn is not None:
            pid = plant_id_fn(r.get("image", ""))
        else:
            pid = None
        if pid in (None, ""):
            raise ValueError(
                f"aggregate_per_plant: record {r.get('image', '<no image key>')!r} has no "
                f"{plant_id_key!r} value and no plant_id_fn resolved one. Plant identity is never "
                "guessed from a filename. Pass plant_id_fn=... built from "
                "tcip_mcp.pipelines.postprocessing.plant_mapping.build_plant_mapping's real "
                f"GNSS+sequence resolution, or ensure every record carries a {plant_id_key!r} key."
            )
        groups[pid].append(r)

    from tcip_mcp.pipelines.model_build import resolve_named

    aggregator = resolve_named(strategy, _STRATEGIES, kind="aggregation strategy")

    results = []
    for plant_id, items in sorted(groups.items()):
        values = [r[value_key] for r in items if value_key in r]
        summary = aggregator(values, len(items) - len(values))
        summary["plant_id"] = plant_id
        summary["observations"] = len(items)
        summary["value_key"] = value_key
        summary["plant_attribution"] = _agreed(
            items, "plant_attribution", f"aggregate_per_plant, plant {plant_id!r}")
        sources = {r["source"] for r in items if r.get("source") is not None}
        if sources:
            summary["plant_id_source"] = sources.pop() if len(sources) == 1 else "mixed"
        distances = [r["distance_m"] for r in items if r.get("distance_m") is not None]
        if distances:
            summary["plant_id_distance_m_max"] = max(distances)
        results.append(summary)

    return results


def _agreed(items: list[dict], key: str, where: str, *, required: bool = False) -> Any:
    """The one value every item states for ``key``; ``None`` when every item omits an optional
    ``key``. Refuses (``ValueError`` naming ``where``) when the items disagree or, with
    ``required``, when any omits it."""
    values = {r.get(key) for r in items}
    if len(values) > 1 or (required and None in values):
        raise ValueError(
            f"{where}: records disagree on{' or omit' if required else ''} {key} "
            f"({sorted(str(v) for v in values)}); every record must state the same one."
        )
    return next(iter(values))


# ── Strategy implementations ────────────────────────────────────────────────


def _agg_count(values: list, n_missing: int) -> dict:
    """Median count across images. A missing value_key is a missing observation, not a measured 0."""
    return {
        "value": statistics.median(values) if values else None,
        "min_count": min(values) if values else None,
        "max_count": max(values) if values else None,
        "n_missing": n_missing,
    }


def _agg_mean(values: list, n_missing: int) -> dict:
    """Arithmetic mean of continuous values; ``None`` when every item omits ``value_key``."""
    if not values:
        return {"value": None, "n_observations_with_value": 0}
    return {
        "value": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "n_observations_with_value": len(values),
    }


def _agg_mode(values: list, n_missing: int) -> dict:
    """Most frequent value (for ordinal traits)."""
    if not values:
        return {"value": None}
    counter = Counter(values)
    mode_val, mode_count = counter.most_common(1)[0]
    return {
        "value": mode_val,
        "agreement": round(mode_count / len(values), 4),
        "distribution": dict(counter),
    }


def _agg_sum(values: list, n_missing: int) -> dict:
    """Sum of values (for area traits); ``None`` when every item omits ``value_key``."""
    if not values:
        return {"value": None, "n_observations_with_value": 0}
    return {"value": sum(values), "n_observations_with_value": len(values)}


_STRATEGIES = {
    "count": _agg_count,
    "mean": _agg_mean,
    "mode": _agg_mode,
    "sum": _agg_sum,
}


def _resolve_units(
    delivered_phenotype: str, results: list[dict], scalar: bool,
) -> tuple[str, str | None, str | None]:
    """``(display_unit, linear_basis, declared_unit)``: the units implied by the aggregated
    values' own value_key, and crops.yml's declared unit for ``delivered_phenotype`` or ``None``.

    Under a ``scalar`` head's delivery, a value_key with no unit suffix at all takes the declared
    unit; one that explicitly ends in ``_px`` (or is bare ``"px"``) never does. A value_key that
    implies a physical unit is cross-checked against the declared one either way.
    ``display_unit`` is squared for an area; ``linear_basis`` is always linear, and is what the
    cross-check and the scale's own unit compare against.
    """
    from tcip_mcp.pipelines.measurement.mask_geometry import (
        is_pixel_space_key,
        unit_from_value_key,
    )
    from tcip_mcp.traits import crops_units

    implied_pairs = {p for p in (unit_from_value_key(r.get("value_key", "")) for r in results) if p}
    if len(implied_pairs) > 1:
        implied_units = {display for display, _linear_basis in implied_pairs}
        raise ValueError(
            f"results for delivered phenotype {delivered_phenotype!r} imply more than one "
            f"physical unit ({sorted(implied_units)}) across rows, cannot label a single units "
            "column.")
    declared = crops_units().get(delivered_phenotype)
    pair = next(iter(implied_pairs), None)
    if pair is None:
        explicitly_px = any(is_pixel_space_key(r.get("value_key", "")) for r in results)
        if scalar and declared is not None and not explicitly_px:
            return declared, declared, declared
        return "", None, declared
    display, linear_basis = pair
    if declared is not None and linear_basis != declared:
        raise ValueError(
            f"delivered phenotype {delivered_phenotype!r} is declared units={declared!r} in "
            f"crops.yml, but the aggregated values' own key implies {linear_basis!r}; refusing to "
            "ship a mismatched unit label rather than guessing which one is right.")
    return display, linear_basis, declared


def deliver_per_plant_aggregate(
    project: Path, results: list[dict], output_path: str, *, delivered_phenotype: str,
    delivery_kind: str, buckets: Sequence[Bucket], plants: list[str], crop: str = "",
    pipeline_version: str = "", door: str, plant_mapping: dict | None = None,
    scale_assessment_id: str | None = None, acknowledgment_id: str | None = None,
    actor: str | None,
) -> dict:
    """Deliver per-plant aggregated ``results`` (:func:`aggregate_per_plant`'s own output) as a
    ``delivery_kind`` CSV at ``output_path``, by ``actor``, for exactly the population ``plants``
    (:func:`~tcip_mcp.pipelines.postprocessing.phenology.population`'s list), over the published
    ``buckets`` the values came from.

    ``delivered_phenotype`` resolves to the one trait whose latest confirmed revision delivers it,
    and that revision's ``delivery_kind`` operationalization must cover it and every row's value
    key. The rows must be exactly the population, every one carrying a value, and agree on
    ``plant_attribution``. The units come from the rows' own value key, cross-checked against
    crops.yml. The buckets, the physical-scale assessment ``scale_assessment_id`` and the unit
    clear the one gate, the recorded acknowledgment ``acknowledgment_id`` shipping them
    unvalidated when it does not validate them.

    Each row carries the plant, its value and unit, its provenance of attribution, and the
    delivery's own cells, written with the delivery's one event under ``door`` with its population
    and ``plant_mapping`` disclosure (:func:`~tcip_mcp.delivery.deliver_csv`); returns what was
    delivered.
    """
    from tcip_mcp.delivery import DELIVERY_COLUMNS, Result, deliver_csv, gate
    from tcip_mcp.operationalization import confirmed_revision
    from tcip_mcp.traits import PER_PLANT_COUNT_AGGREGATE

    by_plant = {str(r["plant_id"]): r for r in results}
    if sorted(by_plant) != sorted(plants):
        raise ValueError(
            f"the rows cover plants {sorted(by_plant)} and the population is {sorted(plants)}: a "
            "delivery ships exactly its population, one row each.")
    none_valued = [p for p in plants if by_plant[p].get("value") is None]
    if none_valued:
        raise ValueError(f"plant(s) {none_valued} carry no {delivered_phenotype!r} observation: a "
                         "missing measurement never ships as an empty cell; leave them out of "
                         "the population.")
    attribution = _agreed(results, "plant_attribution", "the delivered rows", required=True)
    units, linear_basis, declared = _resolve_units(
        delivered_phenotype, results, scalar=delivery_kind != PER_PLANT_COUNT_AGGREGATE)
    if delivery_kind == PER_PLANT_COUNT_AGGREGATE and declared and not units:
        raise ValueError(f"delivered phenotype {delivered_phenotype!r} is declared "
                         f"units={declared!r}, and the rows' value key implies no physical unit.")
    revision = confirmed_revision(delivery_kind, project=project,
                                  delivered_phenotype=delivered_phenotype,
                                  value_keys=[r.get("value_key", "") for r in results])
    result = Result(
        ["plant_id", "crop", "delivered_phenotype", "value", "units", "value_key", "n_images",
         "pipeline_version", "plant_id_source", "plant_attribution", "plant_id_distance_m_max",
         *DELIVERY_COLUMNS],
        [{"plant_id": plant, "crop": crop, "delivered_phenotype": delivered_phenotype,
          "value": by_plant[plant]["value"], "units": units,
          "value_key": by_plant[plant].get("value_key", ""),
          "n_images": by_plant[plant].get("observations", 0), "pipeline_version": pipeline_version,
          "plant_id_source": by_plant[plant].get("plant_id_source", ""),
          "plant_attribution": attribution,
          "plant_id_distance_m_max": by_plant[plant].get("plant_id_distance_m_max", "")}
         for plant in plants],
        population=plants, plant_mapping=plant_mapping)
    clearance = gate(project, buckets, delivery_kind=delivery_kind, revision=revision,
                     result=result, acknowledgment_id=acknowledgment_id,
                     scale_assessment_id=scale_assessment_id, unit=linear_basis)
    delivered = deliver_csv(project, output_path, result, clearance=clearance, revision=revision,
                            door=door, delivery_kind=delivery_kind, actor=actor)
    return {**delivered, "n_plants": len(plants)}
