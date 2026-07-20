"""Per-plant aggregation — temporal/spatial aggregation of per-image results.

Aggregation strategies (per-plant summary of a per-image value):
  - count:    Median detection count across images per plant
  - mode:     Most frequent value (ordinal traits)
  - mean:     Arithmetic mean (continuous traits)
  - sum:      Sum of values (area traits)

Bloom phenology (05/50/95-per-date) is intentionally not here — it is the elongated-
fraction crossing, implemented once in ``postprocessing/phenology.py``.

Usage:
    results = aggregate_per_plant(image_results, strategy="count")
    export_aggregated_csv(results, "output.csv", trait_name="count")
"""

from __future__ import annotations

import csv
import logging
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _extract_plant_id(image_name: str) -> str:
    """Best-effort guess of plant_id from an image filename.

    Heuristic: strip the last **two** underscore-separated tokens, treated as a
    trailing capture/flight suffix, and return the remainder. E.g.:

        'bush_42_flight_3'      → 'bush_42'         (strips 'flight_3')
        'PLANT_001_2024_05_15'  → 'PLANT_001_2024'  (strips '05_15' only)

    Note the second case: a full ``YYYY_MM_DD`` date is *three* tokens, so one
    date token (the year) is retained and one physical plant's temporal series
    fragments per plant-year. There is no filename-only rule that recovers the
    intended plant_id for every naming scheme (``001`` is part of the id but
    ``2024_05_15`` is a date — both look numeric), so this stays a deliberate,
    minimal heuristic rather than a fragile guesser.

    This is a *fallback* used only when neither an explicit ``plant_id`` key nor
    a ``plant_id_fn`` is supplied to ``aggregate_per_plant``; that caller emits a
    warning when it fires so silent mis-grouping does not reach the delivery CSV.
    Pass ``plant_id_fn`` or a ``plant_id`` key to control grouping precisely.

    Falls back to the full stem if there is nothing to strip (single token).
    """
    stem = Path(image_name).stem
    parts = stem.rsplit("_", 2)
    if len(parts) >= 2:
        return parts[0]
    return stem


def aggregate_per_plant(
    image_results: list[dict],
    strategy: str = "count",
    plant_id_key: str = "plant_id",
    value_key: str = "count",
    plant_id_fn: Any = None,
) -> list[dict]:
    """Aggregate per-image results to per-plant summaries.

    Args:
        image_results: List of dicts, each with at least an 'image' key
                      and a value field (e.g., 'count', 'class', 'value').
        strategy: Aggregation strategy — 'count', 'mode', 'mean', 'sum'.
        plant_id_key: Key in each result dict for plant identification.
                     If not present, extracts from image filename.
        value_key: Key in each result dict for the value to aggregate.
        plant_id_fn: Optional callable to extract plant_id from image filename.

    Returns:
        List of per-plant summary dicts.
    """
    # Group by plant_id
    groups: dict[str, list[dict]] = defaultdict(list)
    used_fallback = False
    for r in image_results:
        if plant_id_key in r:
            pid = r[plant_id_key]
        elif plant_id_fn is not None:
            pid = plant_id_fn(r.get("image", ""))
        else:
            pid = _extract_plant_id(r.get("image", "unknown"))
            used_fallback = True
        groups[pid].append(r)

    if used_fallback:
        logger.warning(
            "aggregate_per_plant grouped image(s) by a plant_id *guessed* from "
            "filenames (no %r key and no plant_id_fn). A trailing YYYY_MM_DD date "
            "is three tokens but only two are stripped, so multi-date series can "
            "fragment per plant-year in the delivery CSV. Supply plant_id_fn or a "
            "%r key to control grouping.",
            plant_id_key,
            plant_id_key,
        )

    aggregator = _STRATEGIES.get(strategy)
    if aggregator is None:
        raise ValueError(f"Unknown aggregation strategy: {strategy}. Available: {list(_STRATEGIES.keys())}")

    results = []
    for plant_id, items in sorted(groups.items()):
        summary = aggregator(items, value_key)
        summary["plant_id"] = plant_id
        summary["observations"] = len(items)
        results.append(summary)

    return results


# ── Strategy implementations ────────────────────────────────────────────────


def _agg_count(items: list[dict], value_key: str) -> dict:
    """Median count across images."""
    values = [r.get(value_key, 0) for r in items]
    return {
        "value": statistics.median(values) if values else 0,
        "min_count": min(values) if values else 0,
        "max_count": max(values) if values else 0,
    }


def _agg_mean(items: list[dict], value_key: str) -> dict:
    """Arithmetic mean of continuous values."""
    values = [r.get(value_key, 0.0) for r in items if value_key in r]
    if not values:
        return {"value": 0.0}
    return {
        "value": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
    }


def _agg_mode(items: list[dict], value_key: str) -> dict:
    """Most frequent value (for ordinal traits)."""
    values = [r.get(value_key) for r in items if value_key in r]
    if not values:
        return {"value": None}
    counter = Counter(values)
    mode_val, mode_count = counter.most_common(1)[0]
    return {
        "value": mode_val,
        "agreement": round(mode_count / len(values), 4),
        "distribution": dict(counter),
    }


def _agg_sum(items: list[dict], value_key: str) -> dict:
    """Sum of values (for area traits)."""
    values = [r.get(value_key, 0.0) for r in items if value_key in r]
    return {"value": sum(values)}


# note: catkin bloom phenology (05/50/95-per-date milestones) is not an aggregation
# strategy here. It is the elongated-fraction crossing, and its single canonical
# implementation lives in ``postprocessing/phenology.py`` (used by the web Results route
# and the ``compute_phenology`` MCP tool). A prior "sigmoid" strategy computed those
# milestones from raw count normalized to the season peak — a different, wrong definition
# of the same trait — and has been removed (see CLAUDE.md measurement integrity).


_STRATEGIES = {
    "count": _agg_count,
    "mean": _agg_mean,
    "mode": _agg_mode,
    "sum": _agg_sum,
}


_PROVENANCE_COLUMNS = ["producer_model_sha256", "experiment_id", "produced_at",
                       "measurement_validated"]


def export_aggregated_csv(
    results: list[dict],
    output_path: str,
    trait_name: str = "trait",
    crop: str = "",
    pipeline_version: str = "",
    provenance: dict | None = None,
    *,
    measurement_validated: str | None = None,
    pred_dirs: list[str] | None = None,
    acknowledge_unvalidated: bool = False,
) -> str:
    """Export per-plant aggregated results to a delivery CSV.

    Follows the per-plant CSV schema from the delivery skill:
    plant_id, crop, trait_name, value, confidence, n_images, pipeline_version — plus an optional
    producing-model provenance stamp (checkpoint sha, experiment id, timestamp) so the final
    per-plant value is traceable to the exact model that produced it.

    The final per-plant CSV is a delivery door: it refuses a *bare* write (an unvalidated phenotype
    with no acknowledgement) via the shared ``check_delivery_gate`` and stamps the reconciled validity
    into every row. For a count trait, pass ``pred_dirs`` (the prediction buckets the counts came from)
    so the count operating point's validity is read from each ``operating_point.json`` sidecar and
    floored against ``measurement_validated``. For a continuous/ordinal trait with no conf op-point,
    pass ``measurement_validated`` directly (its own validation reference). ``acknowledge_unvalidated``
    ships a clearly-flagged provisional CSV stamped ``validated=false``.

    Args:
        results: Output from aggregate_per_plant().
        output_path: Path for the output CSV file.
        trait_name: Name of the trait being measured.
        crop: Crop species name.
        pipeline_version: Pipeline identifier.
        provenance: Optional producing-model stamp added as trailing columns.
        measurement_validated: The trait measurement's validation reference (a shippable reference).
        pred_dirs: Prediction buckets to reconcile the count operating point's validity from (count
            traits); floored against ``measurement_validated``.
        acknowledge_unvalidated: Write an unvalidated phenotype as a flagged provisional CSV.

    Returns:
        Path to the written CSV file.
    """
    from tcip_mcp.pipelines.resolution import (
        check_delivery_gate,
        reconcile_operating_point_validity,
    )

    state = measurement_validated
    if pred_dirs:
        # A count trait: the measurement validity IS the count operating point's, read from the
        # buckets' sidecars and floored against any caller assertion (never trusted from the string).
        state = reconcile_operating_point_validity(pred_dirs, asserted=measurement_validated)["validated"]
    gate = check_delivery_gate({"measurement": state},
                               acknowledge_unvalidated=acknowledge_unvalidated)
    if not gate.ok:
        raise ValueError(gate.reason)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    stamp = {k: (provenance or {}).get(k) for k in _PROVENANCE_COLUMNS}
    stamp["measurement_validated"] = gate.stamp["measurement"]
    fieldnames = [
        "plant_id", "crop", "trait_name", "value",
        "confidence", "n_images", "pipeline_version",
    ] + _PROVENANCE_COLUMNS

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow({
                "plant_id": r["plant_id"],
                "crop": crop,
                "trait_name": trait_name,
                "value": r.get("value", ""),
                "confidence": r.get("confidence", ""),
                "n_images": r.get("observations", 0),
                "pipeline_version": pipeline_version,
                **stamp,
            })

    return output_path
