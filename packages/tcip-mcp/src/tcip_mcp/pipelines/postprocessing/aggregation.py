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
    export_aggregated_csv(results, "output.csv", delivered_phenotype="stem_count")

``delivered_phenotype`` is a crop-vocabulary delivered-phenotype name (``stem_count`` is one, as an
example rather than as the shape), which is what the CSV column and the unit cross-check are about.
"""

from __future__ import annotations

import csv
import logging
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from tcip_mcp.pipelines.resolution import Acknowledgment

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

    When a record carries ``plant_id_source``/``plant_id_distance_m`` (as ``build_plant_mapping``'s
    ``Assignment`` records do), it is summarized per plant.

    Every record also states, via ``measurement_document``, which sidecar document
    (``operating_point``, ``ordinal_operating_point`` or ``regression_operating_point``) answers
    for the value it carries, and optionally ``scale_document`` (``"resolve_scale"``) when a
    per-pixel physical scale produced it. Both are carried onto the per-plant summary, and so is a
    record's ``plant_attribution`` (the granularity objects were attributed to plants at, e.g.
    ``plant_mapping.MappingBuild``'s ``"image"`` or ``orthomosaic_mapping.DetectionAssignment``'s
    ``"detection"``); a plant whose own images disagree on any of the three refuses (see
    :func:`_agreed_statement_field`).

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
        summary["measurement_document"] = _agreed_statement_field(
            items, "measurement_document", plant_id)
        summary["scale_document"] = _agreed_statement_field(items, "scale_document", plant_id)
        summary["plant_attribution"] = _agreed_statement_field(
            items, "plant_attribution", plant_id)
        sources = {r["plant_id_source"] for r in items if r.get("plant_id_source") is not None}
        if sources:
            summary["plant_id_source"] = sources.pop() if len(sources) == 1 else "mixed"
        distances = [r["plant_id_distance_m"] for r in items
                    if isinstance(r.get("plant_id_distance_m"), (int, float))]
        if distances:
            summary["plant_id_distance_m_max"] = max(distances)
        results.append(summary)

    return results


def _agreed_statement_field(items: list[dict], key: str, plant_id: str) -> Any:
    """One plant's own value for a statement field (``measurement_document``, ``scale_document``,
    or ``plant_attribution``), refusing when its images disagree. ``None`` when every item omits
    the field.
    """
    values = {r.get(key) for r in items}
    if len(values) > 1:
        raise ValueError(
            f"aggregate_per_plant: plant {plant_id!r} carries disagreeing {key} values "
            f"{sorted(str(v) for v in values)} across its images; a statement that disagrees "
            "with itself is not a statement."
        )
    return next(iter(values))


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


_PROVENANCE_COLUMNS = ["producer_model_sha256", "producing_experiment_id", "produced_at",
                       "operating_point_validated", "unvalidated_dimensions", "validation_record",
                       "acknowledged_by", "acknowledgment_reason"]


def _resolve_units(
    delivered_phenotype: str, results: list[dict], measurement_document: str
) -> tuple[str, str | None]:
    """``(display_unit, linear_basis)`` implied by the aggregated values' own value_key;
    crops.yml's declared unit is a cross-check only under ``operating_point``. A value_key with no
    recognized physical-unit suffix (px, count, or a trailing token outside crops.yml's declared
    unit vocabulary) yields ``("", None)`` under ``operating_point``.

    Under a scalar head (``ordinal_operating_point``/``regression_operating_point``), a value_key
    with no unit suffix at all (a bare ``value`` or the trait's own bare name, e.g.
    ``fruit_diameter``) takes the trait's declared unit. A value_key that explicitly ends in
    ``_px`` (or is bare ``"px"``) never inherits the declared unit, under either document. A
    value_key that does imply a physical unit is cross-checked against the declared one under
    either document.

    ``display_unit`` is squared for an area (``"mm2"``, not ``"mm"``); ``linear_basis`` is always
    linear (crops.yml has no squared-unit vocabulary) and is what the cross-check, and the
    physical-scale reconciliation's own ``unit`` argument, compare against.
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
            f"export_aggregated_csv: results for delivered phenotype {delivered_phenotype!r} imply "
            f"more than one physical unit ({sorted(implied_units)}) across rows, cannot label a "
            "single units column."
        )
    declared = crops_units().get(delivered_phenotype)
    pair = next(iter(implied_pairs), None)
    if pair is None:
        explicitly_px = any(is_pixel_space_key(r.get("value_key", "")) for r in results)
        if measurement_document in ("ordinal_operating_point", "regression_operating_point") \
                and declared is not None and not explicitly_px:
            return declared, declared
        return "", None
    display, linear_basis = pair
    if declared is not None and linear_basis != declared:
        raise ValueError(
            f"export_aggregated_csv: delivered phenotype {delivered_phenotype!r} is declared "
            f"units={declared!r} in crops.yml, but the aggregated values' own key implies "
            f"{linear_basis!r}, refusing to ship a mismatched unit label rather than guessing which "
            "one is right."
        )
    return display, linear_basis


def export_aggregated_csv(
    results: list[dict],
    output_path: str,
    delivered_phenotype: str,
    crop: str = "",
    pipeline_version: str = "",
    provenance: dict | None = None,
    *,
    operating_point_validated: str | None = None,
    pred_dirs: list[str] | None = None,
    images_dir: str | None = None,
    scale_capture_id: str | None = None,
    door: str = "export_aggregated_csv",
    plant_mapping: dict | None = None,
    acknowledgment: Acknowledgment | None = None,
    project_root: str | Path | None = None,
) -> tuple[str, dict, bool]:
    """Export per-plant aggregated results to a delivery CSV.

    The ``fieldnames`` list below is the schema: plant_id, crop, delivered_phenotype, value, units,
    value_key, measurement_document, scale_document, confidence, n_images, pipeline_version,
    plant_id_source, plant_attribution, plant_id_distance_m_max, then ``_PROVENANCE_COLUMNS``
    (producer_model_sha256, producing_experiment_id, produced_at, operating_point_validated,
    unvalidated_dimensions, validation_record, acknowledged_by, acknowledgment_reason). Those
    cells are built by ``delivered_tail`` from the verification the gate ran: a producer this
    delivery cannot corroborate reads unknown, ``produced_at`` is the write's own timestamp, and
    ``validation_record`` names the record the claim was earned against.

    Each result (``aggregate_per_plant``'s own output) states ``measurement_document``, which
    sidecar document answers for the measurement that produced the value (``operating_point``,
    ``ordinal_operating_point`` or ``regression_operating_point``; ``classifier_operating_point``
    and ``resolve_scale`` are refused here), and optionally ``scale_document``
    (``"resolve_scale"``) when a per-pixel physical scale produced it. Every result must state the
    same ``measurement_document`` and agree on ``scale_document`` (present on all or none); a
    delivery whose records disagree, or state nothing, refuses naming the field. The statement
    decides which validity dimension the door reconciles.

    The door refuses a bare write (an unvalidated phenotype with no acknowledgment) via
    ``check_delivery_gate`` and stamps the reconciled validity into every row. ``pred_dirs`` (the
    prediction buckets the values came from) has the stated document's own sidecar reconciled from
    each bucket and floored against ``operating_point_validated``. Under ``operating_point``, a
    tiled bucket also gates on its ``tile_size``; untiled buckets and ordinal/regression deliveries
    never do. ``pred_dirs`` empty/omitted floors to unvalidated. ``acknowledgment`` is the
    breeder's own act of shipping this delivery unvalidated, or ``None``.

    The physical-scale dimension is reconciled when and only when the results state
    ``scale_document`` and ``pred_dirs`` is given. A stated scale with a value_key implying no
    physical unit refuses, whether or not ``pred_dirs`` is given; a stated scale with no
    ``pred_dirs`` also refuses, ahead of the gate. With ``pred_dirs`` given, a value_key implying a
    physical unit with no stated scale is admitted only under a scalar head and refuses under
    ``operating_point``. When operative, each bucket's ``resolve_scale.json`` is reconciled against
    the delivered unit and the delivered trait (``reconcile_scale_validity``), recomputing the
    claim's imagery digest from ``images_dir`` (required whenever ``scale_document`` is stated
    alongside ``pred_dirs``). ``scale_capture_id`` scopes that reconciliation to one capture.

    Under ``operating_point``, a trait declaring a physical unit (crops.yml) whose delivered
    value_key implies none refuses, naming both. A count trait with no declared unit is unaffected;
    a scalar-head delivery is exempt.

    A bucket whose sidecar records a claim scope (which raster its predictions were produced on)
    also gates on that dimension, whatever the measurement document; a bucket recording no claim
    scope never acquires the dimension.

    The delivered value needs a breeder-confirmed operationalization: ``delivered_phenotype``
    resolves to the registered trait whose spec delivers it (refusing when none or more than one
    does), and the aggregate kind confirmed follows from ``measurement_document``. Every delivered
    row carries a value key inside the confirmed set. A plant whose own value is ``None`` refuses,
    naming the plant.

    Args:
        results: Output from aggregate_per_plant().
        output_path: Path for the output CSV file.
        delivered_phenotype: The crop-vocabulary delivered phenotype this CSV ships under.
            Required, and resolved to the registered trait whose spec delivers it.
        crop: Crop species name.
        pipeline_version: Pipeline identifier.
        provenance: Optional producing-model stamp added as trailing columns.
        operating_point_validated: Honored only when ``pred_dirs`` is also given (floors the
            on-disk validity, never raises it).
        pred_dirs: Prediction buckets to reconcile validity from: the results' own stated
            ``measurement_document``, and ``scale_document`` when stated; floored against
            ``operating_point_validated``.
        images_dir: The buckets' own images directory, required when ``scale_document`` is stated
            (with ``pred_dirs`` given) to recompute a scale claim's imagery digest.
        scale_capture_id: The capture this delivery's physical scale must match, when the scale is
            capture-scoped; a bucket's sidecar recording a different capture floors to unvalidated.
        door: The name the recorded delivery event carries.
        plant_mapping: The delivery's own plant-mapping binding, recorded onto the delivery event
            unchanged: the shape ``plant_mapping.MappingBuild.delivery_disclosure`` composes for a
            walked capture mapping, or the shape ``deliver_orthomosaic_plant_counts`` composes for
            a whole-raster frame (validated whole by ``PlantMappingDisclosure`` /
            ``PlantRegistryDisclosure``). ``None`` when the caller names no verified mapping.
        acknowledgment: The breeder's own act of shipping this delivery unvalidated, or ``None``.
        project_root: The project this delivery's meaning-record reads and delivery event belong
            to. ``None`` resolves against this process's pinned platform root.

    Returns:
        ``(path, tail, event_recorded)``: the path to the written CSV, the ``_PROVENANCE_COLUMNS``
        tail ``delivered_tail`` composed and wrote into every row, and whether the best-effort
        delivery-event write landed (``record_delivery_binding_event``'s own return).

    Raises:
        DeliveryRefused: the gate refused (an unvalidated dimension with no acknowledgment that
            clears it); carries the ``DeliveryGateResult`` and every operative reconciler's binding
            notes.
        OperationalizationRefused (``tcip_mcp.operationalization``): ``delivered_phenotype``'s
            operationalization is unrecorded, not breeder-confirmed, or was withdrawn since the
            first check; carries the failed check and no counts.
        ValueError: any other refusal (a statement or unit problem the results carry); carries no
            gate result.
        AuditEntryNotWritten (``tcip_mcp.audit``): the dataset-scoped delivery-event audit line
            could not be appended, raised by ``record_delivery_binding_event`` after the CSV was
            already written to ``output_path``.
    """
    from tcip_mcp.pipelines.resolution import (
        MEASUREMENT_DOCUMENTS,
        VALIDATED_FALSE,
        DeliveryRefused,
        binding_notes_text,
        check_delivery_gate,
        delivered_tail,
        record_delivery_binding_event,
        reconcile_claim_scope_validity,
        reconcile_operating_point_validity,
        reconcile_ordinal_validity,
        reconcile_regression_validity,
        reconcile_scale_validity,
        reconcile_tile_size_validity,
    )

    measurement_document, scale_document = _resolve_statement(results, MEASUREMENT_DOCUMENTS)
    plant_attribution = _resolve_plant_attribution(results)
    units, linear_basis = _resolve_units(delivered_phenotype, results, measurement_document)

    from tcip_mcp.traits import crops_units

    declared_unit = crops_units().get(delivered_phenotype)
    if measurement_document == "operating_point" and declared_unit and not units:
        raise ValueError(
            f"export_aggregated_csv: delivered phenotype {delivered_phenotype!r} is declared "
            f"units={declared_unit!r} in crops.yml, but the aggregated values' own value_key "
            "implies no physical unit under an operating_point measurement document; a pixel-space "
            "value delivered under a unit-declared phenotype is not that phenotype's number."
        )
    if scale_document is not None and not units:
        raise ValueError(
            "export_aggregated_csv: results state scale_document but their own value_key implies "
            "no physical unit; a physical scale cannot answer for a non-dimensional value."
        )
    if scale_document is not None and not pred_dirs:
        raise ValueError(
            "export_aggregated_csv: results state scale_document with no pred_dirs; nothing on "
            "disk can answer for a physical-scale claim without a bucket to reconcile it from, "
            "and no acknowledgment clears a scale claim with nothing on disk to reconcile it "
            "from regardless. Pass pred_dirs for the buckets behind these results."
        )
    if pred_dirs and units and scale_document is None and measurement_document == "operating_point":
        raise ValueError(
            f"export_aggregated_csv: results imply physical unit {units!r} from an "
            "operating_point measurement document with no stated scale_document; a dimensional "
            "number from a detection or segmentation bucket has nothing answering for its unit "
            "without one. State scale_document='resolve_scale' once the bucket carries a "
            "calibrate_physical_scale claim, or deliver a non-dimensional value_key."
        )

    none_valued = sorted(r["plant_id"] for r in results if r.get("value") is None)
    if none_valued:
        raise ValueError(
            f"export_aggregated_csv: plant(s) {none_valued} carry no {delivered_phenotype!r} "
            "observation at all (value is None); a missing measurement must never ship as an empty "
            "cell beside a validated stamp. Filter these plants out before calling this door to "
            "deliver only the plants that do carry a value."
        )

    from tcip_mcp.operationalization import (
        OperationalizationRefused,
        aggregate_delivery_kind,
        check_operationalization,
        resolve_trait_and_record,
        resolve_trait_for_phenotype,
    )

    delivery_kind = aggregate_delivery_kind(measurement_document)
    trait = resolve_trait_for_phenotype(delivered_phenotype, project_root=project_root)
    value_keys = [r.get("value_key", "") for r in results]
    spec, record, _specs_dir = resolve_trait_and_record(
        trait, delivery_kind, project_root=project_root)
    # This door never delivers a crossing kind, so it has no registry to check a positive class against.
    stated = check_operationalization(
        spec, record, delivery_kind, delivered_phenotype=delivered_phenotype, value_keys=value_keys,
        registry=None)
    if not stated.ok:
        raise OperationalizationRefused(stated)

    _reconcilers = {
        "operating_point": reconcile_operating_point_validity,
        "ordinal_operating_point": reconcile_ordinal_validity,
        "regression_operating_point": reconcile_regression_validity,
    }
    tile_recon: dict | None = None
    scale_recon: dict | None = None
    # Claim scope is a fact about the bucket, not the document being delivered: any bucket recording
    # one reconciles, regardless of measurement_document.
    claim_scope_recon: dict | None = reconcile_claim_scope_validity(pred_dirs) if pred_dirs else None
    operating_point_recon: dict | None = None
    if pred_dirs:
        operating_point_recon = _reconcilers[measurement_document](
            pred_dirs, trait=trait, asserted=operating_point_validated)
        state = operating_point_recon["validated"]
        if measurement_document == "operating_point":
            tile_recon = reconcile_tile_size_validity(pred_dirs)
        if scale_document is not None:
            assert linear_basis is not None, (
                "scale_document is only ever stated alongside a value_key that implies a unit, "
                "checked above; linear_basis cannot be None here.")
            if images_dir is None:
                raise ValueError(
                    "export_aggregated_csv: results state scale_document with pred_dirs given but "
                    "no images_dir; a scale claim's imagery digest cannot be recomputed without it."
                )
            scale_recon = reconcile_scale_validity(
                pred_dirs, unit=linear_basis, trait=trait, images_dir=images_dir,
                capture_id=scale_capture_id)
    else:
        # No pred_dirs, no on-disk source; a bare caller-asserted string is never trusted alone.
        state = VALIDATED_FALSE
    flags: dict[str, str | None] = {"operating_point": state}
    if tile_recon is not None and tile_recon["operative"]:
        flags["tile_size"] = tile_recon["validated"]
    if scale_recon is not None and scale_recon["operative"]:
        flags["scale"] = scale_recon["validated"]
    if claim_scope_recon is not None and claim_scope_recon["operative"]:
        flags["claim_scope"] = claim_scope_recon["validated"]
    gate = check_delivery_gate(flags, acknowledgment=acknowledgment)
    if not gate.ok:
        notes = " ".join(filter(None, (
            binding_notes_text((operating_point_recon or {}).get("binding_notes", {})),
            binding_notes_text((tile_recon or {}).get("binding_notes", {})),
            binding_notes_text((scale_recon or {}).get("binding_notes", {})),
            binding_notes_text((claim_scope_recon or {}).get("binding_notes", {})),
        )))
        raise DeliveryRefused(gate, notes)

    # A confirmation withdrawn or a field moved since the first check refuses here, before anything.
    spec_now, record_now, _ = resolve_trait_and_record(
        trait, delivery_kind, project_root=project_root)
    still_stated = check_operationalization(
        spec_now, record_now, delivery_kind, delivered_phenotype=delivered_phenotype,
        value_keys=value_keys, registry=None, basis=stated.basis)
    if not still_stated.ok:
        raise OperationalizationRefused(still_stated)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    stamp = delivered_tail(provenance, (operating_point_recon or {}).get("bindings", {}), gate,
                           columns=_PROVENANCE_COLUMNS)
    fieldnames = [
        "plant_id", "crop", "delivered_phenotype", "value", "units", "value_key",
        "measurement_document", "scale_document",
        "confidence", "n_images", "pipeline_version",
        "plant_id_source", "plant_attribution", "plant_id_distance_m_max",
    ] + _PROVENANCE_COLUMNS

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow({
                "plant_id": r["plant_id"],
                "crop": crop,
                "delivered_phenotype": delivered_phenotype,
                "value": r.get("value", ""),
                "units": units,
                "value_key": r.get("value_key", ""),
                "measurement_document": measurement_document,
                "scale_document": scale_document or "",
                "confidence": r.get("confidence", ""),
                "n_images": r.get("observations", 0),
                "pipeline_version": pipeline_version,
                "plant_id_source": r.get("plant_id_source", ""),
                "plant_attribution": plant_attribution,
                "plant_id_distance_m_max": r.get("plant_id_distance_m_max", ""),
                **stamp,
            })

    dimension_reconciliations = {
        **({"claim_scope": claim_scope_recon} if claim_scope_recon is not None else {}),
        **({"tile_size": tile_recon} if tile_recon is not None else {}),
        **({"scale": scale_recon} if scale_recon is not None else {}),
    }
    event_recorded = record_delivery_binding_event(
        door, output_path, pred_dirs,
        document_reconciliations=(
            {measurement_document: operating_point_recon} if operating_point_recon is not None
            else {}
        ),
        dimension_reconciliations=dimension_reconciliations,
        measurement_documents=[measurement_document],
        acknowledgment=gate.effective_acknowledgment(), trait=trait, delivery_kind=delivery_kind,
        project_root=project_root, plant_mapping=plant_mapping)
    return output_path, stamp, event_recorded


def _resolve_statement(
    results: list[dict], measurement_documents: tuple[str, ...]
) -> tuple[str, str | None]:
    """The delivery's ``(measurement_document, scale_document)``, read off ``results`` and checked
    for agreement: every row states a ``measurement_document``, all rows state the same one, all
    rows agree on ``scale_document`` (present on all or on none), and the measurement document is
    one of ``measurement_documents`` (``classifier_operating_point`` and ``resolve_scale`` are
    refused as a measurement here). A delivery that states nothing refuses instead of falling
    through to any particular reconciler."""
    if not results:
        raise ValueError("export_aggregated_csv: results is empty, nothing to deliver")
    documents = {r.get("measurement_document") for r in results}
    if len(documents) > 1 or None in documents:
        raise ValueError(
            f"export_aggregated_csv: results disagree on or omit measurement_document "
            f"({sorted(str(d) for d in documents)}); every record aggregate_per_plant produces "
            "must state which sidecar document its value rests on."
        )
    measurement_document = documents.pop()
    if measurement_document not in measurement_documents:
        raise ValueError(
            f"export_aggregated_csv: measurement_document {measurement_document!r} is not one of "
            f"{measurement_documents}; a per-plant aggregate never rests on "
            "classifier_operating_point or resolve_scale alone."
        )
    scale_documents = {r.get("scale_document") for r in results}
    if len(scale_documents) > 1:
        raise ValueError(
            f"export_aggregated_csv: results disagree on scale_document "
            f"({sorted(str(d) for d in scale_documents)}); it must be stated on every row or none."
        )
    scale_document = scale_documents.pop()
    if scale_document not in (None, "resolve_scale"):
        raise ValueError(
            f"export_aggregated_csv: scale_document must be 'resolve_scale' or absent, got "
            f"{scale_document!r}."
        )
    return measurement_document, scale_document


def _resolve_plant_attribution(results: list[dict]) -> str:
    """The delivery's own ``plant_attribution``, read off ``results`` the way
    :func:`_resolve_statement` reads ``measurement_document``: every row states one, and they
    agree.
    """
    values = {r.get("plant_attribution") for r in results}
    if len(values) > 1 or None in values:
        raise ValueError(
            f"export_aggregated_csv: results disagree on or omit plant_attribution "
            f"({sorted(str(v) for v in values)}); every record must state the granularity "
            "objects were attributed to plants at."
        )
    value = values.pop()
    # The refusal above already ruled out None and a multi-value set; what remains is one string.
    return cast(str, value)
