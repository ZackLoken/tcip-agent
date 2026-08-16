"""Per-plant aggregation, temporal/spatial aggregation of per-image results.

Aggregation strategies (per-plant summary of a per-image value):
  - count:    Median detection count across images per plant
  - mode:     Most frequent value (ordinal traits)
  - mean:     Arithmetic mean (continuous traits)
  - sum:      Sum of values (area traits)

A trait's own phenology milestones (percentile-crossing dates of its positive-state fraction,
e.g. a 5/50/95% schedule) are intentionally not here: that is interpolated crossing time, a
different shape of computation than a per-image summary statistic, implemented once in
``postprocessing/phenology.py`` for whichever trait is registered.

Usage:
    results = aggregate_per_plant(image_results, strategy="count")
    export_aggregated_csv(results, "output.csv", trait_name="stem_count")

``trait_name`` is a crop-vocabulary delivered-phenotype name (``stem_count`` is one, as an example
rather than as the shape), which is what the CSV column and the unit cross-check are about.
"""

from __future__ import annotations

import csv
import logging
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def aggregate_per_plant(
    image_results: list[dict],
    strategy: str = "count",
    plant_id_key: str = "plant_id",
    value_key: str = "count",
    plant_id_fn: Any = None,
) -> list[dict]:
    """Aggregate per-image results to per-plant summaries.

    Plant identity is never guessed from a filename. Every record's ``plant_id`` must come from an
    explicit ``plant_id_key`` value or ``plant_id_fn(image)``, a record for which neither resolves
    raises, naming ``build_plant_mapping`` (the real GNSS+capture-sequence resolver,
    ``tcip_mcp.pipelines.postprocessing.plant_mapping``) as what to use instead. Filenames rarely
    carry a plant id at all; real identity resolution in this domain is RTK/GNSS + capture sequence
    or a zbar/QR code physically tied to the plant, never a filename heuristic.

    When a record carries ``plant_id_source``/``plant_id_distance_m`` (as ``build_plant_mapping``'s
    own ``Assignment`` records do, the real, honest uncertainty signal that mapping already
    produces), it is summarized per plant so identity confidence reaches the delivery CSV instead of
    stopping at the mapping step's own boundary.

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

    aggregator = _STRATEGIES.get(strategy)
    if aggregator is None:
        raise ValueError(f"Unknown aggregation strategy: {strategy}. Available: {list(_STRATEGIES.keys())}")

    results = []
    for plant_id, items in sorted(groups.items()):
        summary = aggregator(items, value_key)
        summary["plant_id"] = plant_id
        summary["observations"] = len(items)
        summary["value_key"] = value_key
        sources = {r["plant_id_source"] for r in items if r.get("plant_id_source") is not None}
        if sources:
            summary["plant_id_source"] = sources.pop() if len(sources) == 1 else "mixed"
        distances = [r["plant_id_distance_m"] for r in items
                    if isinstance(r.get("plant_id_distance_m"), (int, float))]
        if distances:
            summary["plant_id_distance_m_max"] = max(distances)
        results.append(summary)

    return results


# ── Strategy implementations ────────────────────────────────────────────────


def _agg_count(items: list[dict], value_key: str) -> dict:
    """Median count across images. A missing value_key is a missing observation, not a measured 0."""
    values = [r[value_key] for r in items if value_key in r]
    n_missing = len(items) - len(values)
    return {
        "value": statistics.median(values) if values else None,
        "min_count": min(values) if values else None,
        "max_count": max(values) if values else None,
        "n_missing": n_missing,
    }


def _agg_mean(items: list[dict], value_key: str) -> dict:
    """Arithmetic mean of continuous values. All-absent yields None, never a fabricated 0.0, a
    missing measurement must never read as a measured zero."""
    values = [r.get(value_key, 0.0) for r in items if value_key in r]
    n_observations_with_value = len(values)
    if not values:
        return {"value": None, "n_observations_with_value": 0}
    return {
        "value": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "n_observations_with_value": n_observations_with_value,
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
    """Sum of values (for area traits). All-absent yields None, never a fabricated 0."""
    values = [r.get(value_key, 0.0) for r in items if value_key in r]
    if not values:
        return {"value": None, "n_observations_with_value": 0}
    return {"value": sum(values), "n_observations_with_value": len(values)}


# A trait's phenology milestones need interpolated crossing time, not a per-image aggregate;
# see ``postprocessing/phenology.py``, not a strategy entry here.


_STRATEGIES = {
    "count": _agg_count,
    "mean": _agg_mean,
    "mode": _agg_mode,
    "sum": _agg_sum,
}


_PROVENANCE_COLUMNS = ["producer_model_sha256", "experiment_id", "produced_at",
                       "measurement_validated", "validation_record"]

def _unit_from_value_key(value_key: str) -> tuple[str, str] | None:
    """``(display_unit, linear_basis)`` a value_key implies (``area_mm2`` -> ``("mm2", "mm")``,
    ``principal_axis_extent_cm`` -> ``("cm", "cm")``), or None for a key with no physical-unit suffix
    (``count``, a plain ``value``, a px-suffixed key, or a trailing token outside crops.yml's own
    declared unit vocabulary). Delegates to :func:`mask_geometry.unit_from_value_key`, the single
    owner of the naming convention, vocabulary-driven rather than a field-name whitelist, so a
    bespoke agent-composed measurement (an arc length, a landmark distance) is recognized the same
    way mask_geometry's own fields are, rather than re-deriving the pattern with a local regex (two
    independent parsers of the same convention is exactly the drift class that let a wrong pattern
    fabricate unit labels from unrelated keys like ``plant_id`` or ``detections_total``)."""
    from tcip_mcp.pipelines.measurement.mask_geometry import unit_from_value_key

    return unit_from_value_key(value_key)


def _resolve_units(trait_name: str, results: list[dict]) -> str:
    """The physical unit implied by the aggregated values' own value_key, crops.yml's declared unit
    is a cross-check only, never a fallback source. A value_key with no recognized physical-unit
    suffix (px, count, or a trailing token outside crops.yml's declared unit vocabulary) yields a
    blank units column, exactly like a count trait already does, it never inherits crops.yml's
    declared unit unopposed, which was the actual defect: a pixel-space value shipping labeled with
    the trait's declared mm/cm/m because nothing derived a unit to check it against.

    An area's returned unit is squared (``"mm2"``, not ``"mm"``), the cross-check itself still
    compares against crops.yml's declared linear unit (crops.yml has no squared-unit vocabulary)."""
    from tcip_mcp.traits import crops_units

    implied_pairs = {p for p in (_unit_from_value_key(r.get("value_key", "")) for r in results) if p}
    if len(implied_pairs) > 1:
        implied_units = {display for display, _linear_basis in implied_pairs}
        raise ValueError(
            f"export_aggregated_csv: results for trait {trait_name!r} imply more than one physical "
            f"unit ({sorted(implied_units)}) across rows, cannot label a single units column."
        )
    pair = next(iter(implied_pairs), None)
    if pair is None:
        return ""
    display, linear_basis = pair
    declared = crops_units().get(trait_name)
    if declared is not None and linear_basis != declared:
        raise ValueError(
            f"export_aggregated_csv: trait {trait_name!r} is declared units={declared!r} in "
            f"crops.yml, but the aggregated values' own key implies {linear_basis!r}, refusing to "
            "ship a mismatched unit label rather than guessing which one is right."
        )
    return display


def export_aggregated_csv(
    results: list[dict],
    output_path: str,
    trait_name: str,
    crop: str = "",
    pipeline_version: str = "",
    provenance: dict | None = None,
    *,
    measurement_validated: str | None = None,
    pred_dirs: list[str] | None = None,
    task: str | None = None,
    scale_capture_id: str | None = None,
    acknowledge_unvalidated: bool = False,
) -> str:
    """Export per-plant aggregated results to a delivery CSV.

    Follows the per-plant CSV schema from the delivery skill, the ``fieldnames`` list below is the
    authority for it: plant_id, crop, trait_name, value, units, value_key, confidence, n_images,
    pipeline_version, plant_id_source, plant_id_distance_m_max, then ``_PROVENANCE_COLUMNS``
    (producer_model_sha256, experiment_id, produced_at, measurement_validated, validation_record) so
    the final per-plant value is traceable to the exact model that produced it and carries its own
    validity stamp. Those cells are built by ``delivered_provenance`` from the verification the gate
    already ran, so a producer this delivery cannot corroborate is reported unknown rather than
    repeated from the stamp that asserted it, and ``validation_record`` names the record a reader
    can open to see what the claim was earned against.

    The final per-plant CSV is a delivery door: it refuses a *bare* write (an unvalidated phenotype
    with no acknowledgement) via the shared ``check_delivery_gate`` and stamps the reconciled validity
    into every row. For a count trait, pass ``pred_dirs`` (the prediction buckets the counts came from)
    so the count operating point's validity is read from each ``operating_point.json`` sidecar and
    floored against ``measurement_validated``. A bucket produced by a tiled run gates on its
    ``tile_size`` too, the same operating point's other gating dimension: the tile edge scales the
    per-image counts this per-plant value aggregates, so a run with no persisted training geometry
    and no explicit caller override refuses here. Untiled buckets are never gated on it.

    For an ordinal/regression trait, pass both ``pred_dirs`` (the buckets holding
    ``ordinal_operating_point.json``/``regression_operating_point.json``, written by
    ``calibrate_ordinal_regression_operating_point``) and ``task`` (``"ordinal"`` or
    ``"regression"``), which dimension's sidecar to reconcile from is nothing this function can
    infer from ``pred_dirs``/``value_key`` alone, the caller must state it explicitly rather than
    have this door guess. ``pred_dirs`` given with no ``task`` still means a count trait (the prior,
    unchanged behavior); ``pred_dirs`` empty/omitted with no ``task`` has no on-disk validity
    producer at all, and floors to unvalidated unconditionally, the only route to delivery is the
    explicit acknowledge below. An ordinal/regression delivery never gates on tile geometry or
    physical scale, neither dimension applies to a per-image scalar prediction.

    When the aggregated values' own ``value_key`` implies a physical unit (``area_mm2``, a real
    dimensional trait, see ``_resolve_units``) and ``pred_dirs`` is given with no ``task`` (a count
    trait), the delivery also gates on the physical-scale dimension: each bucket's
    ``resolve_scale.json`` (see ``tcip_mcp.pipelines.measurement.mask_geometry.resolve_scale``) is
    reconciled the same floor-from-disk way the count operating point and tile geometry are, so a
    dimensional CSV can no longer ship stamped validated while the scale that produced every
    mm/cm/m number in it was never checked against a real physical reference. A count trait (no
    dimensional value_key) never acquires this dimension, the "operative" framing tile_size already
    uses: a scale was never relevant to what is being delivered, so nothing here manufactures a
    refusal over it. ``scale_capture_id`` scopes that reconciliation to one capture when the
    delivery's physical scale is itself capture-scoped (a handheld standoff that can vary image to
    image). ``acknowledge_unvalidated`` ships a clearly-flagged provisional CSV stamped
    ``validated=false``.

    A bucket whose sidecar records a claim scope (which raster its predictions were produced on,
    written by the whole-raster export regime) also gates on that dimension, whatever the task: an
    operating point calibrated on one mosaic says nothing about a bucket produced on another, so a
    recorded scope that cleared nothing floors this delivery the same way an uncalibrated conf
    does. A bucket recording no claim scope never acquires the dimension.

    Meaning door: the delivered value has to have a recorded, breeder-confirmed meaning before it
    ships. ``trait_name`` is the crop-vocabulary phenotype the CSV column carries and the unit
    cross-check reads; the record that says what the number means is keyed by the registered trait
    whose spec delivers that phenotype, resolved here, refusing when no registered trait delivers it
    and when more than one does. Which of the three aggregate kinds is confirmed follows from
    ``task``, since a count, an ordinal and a regression aggregate rest on three different floors.
    Every delivered row carries a value key, and every one must be inside the confirmed set: a row
    with no stated quantity has nothing to check against what the breeder confirmed.

    Args:
        results: Output from aggregate_per_plant().
        output_path: Path for the output CSV file.
        trait_name: The crop-vocabulary delivered phenotype this CSV ships under. Required, and
            resolved to the registered trait whose spec delivers it.
        crop: Crop species name.
        pipeline_version: Pipeline identifier.
        provenance: Optional producing-model stamp added as trailing columns.
        measurement_validated: Honored only when ``pred_dirs`` is also given (floors the on-disk
            validity, never raises it). Ignored, not a delivery path, when ``pred_dirs`` is empty,
            see the note above.
        pred_dirs: Prediction buckets to reconcile validity from: the count operating point (no
            ``task``) or the ordinal/regression compensating-error gate (``task`` given); floored
            against ``measurement_validated``. Also the source buckets for the physical-scale
            dimension when the results are dimensional and no ``task`` is given.
        task: ``None`` (a count trait, the prior behavior) or ``"ordinal"``/``"regression"``,
            which on-disk sidecar dimension to reconcile ``pred_dirs`` against.
        scale_capture_id: The capture this delivery's physical scale must match, when the scale is
            capture-scoped; a bucket's sidecar recording a different capture floors to unvalidated.
        acknowledge_unvalidated: Write an unvalidated phenotype as a flagged provisional CSV.

    Returns:
        Path to the written CSV file.
    """
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE,
        check_delivery_gate,
        delivered_provenance,
        record_delivery_binding_event,
        reconcile_claim_scope_validity,
        reconcile_operating_point_validity,
        reconcile_ordinal_validity,
        reconcile_regression_validity,
        reconcile_scale_validity,
        reconcile_tile_size_validity,
    )

    if task is not None and task not in ("ordinal", "regression"):
        raise ValueError(f"task must be None, 'ordinal', or 'regression', got {task!r}")

    units = _resolve_units(trait_name, results)

    from tcip_mcp.operationalization import (
        aggregate_delivery_kind,
        check_operationalization,
        resolve_trait_and_record,
        resolve_trait_for_phenotype,
    )

    delivery_kind = aggregate_delivery_kind(task)
    trait = resolve_trait_for_phenotype(trait_name)
    value_keys = [r.get("value_key", "") for r in results]
    spec, record, _specs_dir = resolve_trait_and_record(trait, delivery_kind)
    stated = check_operationalization(
        spec, record, delivery_kind, delivered_phenotype=trait_name, value_keys=value_keys)
    if not stated.ok:
        raise ValueError(stated.message)

    tile_recon = {"operative": False, "validated": None}
    scale_recon = {"operative": False, "validated": None}
    # Which raster a bucket's predictions were produced on is a fact about the bucket, not about
    # the task being delivered, so this reconciles for any bucket that records one.
    claim_scope_recon = (reconcile_claim_scope_validity(pred_dirs) if pred_dirs
                         else {"operative": False, "validated": None})
    measurement_recon: dict = {"bindings": {}}
    if pred_dirs and task == "ordinal":
        measurement_recon = reconcile_ordinal_validity(pred_dirs, asserted=measurement_validated)
        state = measurement_recon["validated"]
    elif pred_dirs and task == "regression":
        measurement_recon = reconcile_regression_validity(pred_dirs, asserted=measurement_validated)
        state = measurement_recon["validated"]
    elif pred_dirs:
        # A count trait: the measurement validity is the count operating point's, read from the
        # buckets' sidecars and floored against any caller assertion (never trusted from the string).
        measurement_recon = reconcile_operating_point_validity(
            pred_dirs, asserted=measurement_validated)
        state = measurement_recon["validated"]
        # The tile scale is the same operating point's other gating dimension: it scales the
        # per-image counts this per-plant value aggregates, so a tiled bucket whose tile edge has no
        # persisted or caller-stated basis is exactly as untrustworthy here as an uncalibrated conf.
        tile_recon = reconcile_tile_size_validity(pred_dirs)
        if units:
            # A dimensional value is actually present in what's being delivered: the physical scale
            # that produced it is a real gating dimension, reconciled from the buckets' own sidecars.
            scale_recon = reconcile_scale_validity(pred_dirs, capture_id=scale_capture_id)
    else:
        # No on-disk source exists for a continuous/ordinal trait's measurement validity with no
        # pred_dirs given, a bare caller-asserted `measurement_validated` string is never trusted on
        # its own. The only route to delivery without a producer is the explicit acknowledge below;
        # this never auto-sets it on the writer's own initiative.
        state = VALIDATED_FALSE
    flags: dict[str, str | None] = {"measurement": state}
    if tile_recon["operative"]:
        flags["tile_size"] = tile_recon["validated"]
    if scale_recon["operative"]:
        flags["scale"] = scale_recon["validated"]
    if claim_scope_recon["operative"]:
        flags["claim_scope"] = claim_scope_recon["validated"]
    gate = check_delivery_gate(flags, acknowledge_unvalidated=acknowledge_unvalidated)
    if not gate.ok:
        raise ValueError(gate.reason)

    # A confirmation withdrawn or a field moved since the first check refuses here, before anything.
    spec_now, record_now, _ = resolve_trait_and_record(trait, delivery_kind)
    still_stated = check_operationalization(
        spec_now, record_now, delivery_kind, delivered_phenotype=trait_name,
        value_keys=value_keys, basis=stated.basis)
    if not still_stated.ok:
        raise ValueError(still_stated.message)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    stamp = delivered_provenance(provenance, measurement_recon["bindings"],
                                 columns=_PROVENANCE_COLUMNS)
    stamp["measurement_validated"] = gate.column_stamp("measurement")
    fieldnames = [
        "plant_id", "crop", "trait_name", "value", "units", "value_key",
        "confidence", "n_images", "pipeline_version",
        "plant_id_source", "plant_id_distance_m_max",
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
                "units": units,
                "value_key": r.get("value_key", ""),
                "confidence": r.get("confidence", ""),
                "n_images": r.get("observations", 0),
                "pipeline_version": pipeline_version,
                "plant_id_source": r.get("plant_id_source", ""),
                "plant_id_distance_m_max": r.get("plant_id_distance_m_max", ""),
                **stamp,
            })

    record_delivery_binding_event("export_aggregated_csv", output_path, pred_dirs,
                                  measurement_recon["bindings"])
    return output_path
