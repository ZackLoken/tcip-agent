"""Orthomosaic MCP tools: per-plant delivery from a persisted whole-raster prediction bucket plus
a plant-locations CSV.

The map-and-deliver half of the same detect-and-persist / map-and-deliver shape
``phenology_tools``'s ``build_plant_mapping``/``compute_phenology`` already use for the
per-image-EXIF case: the detect-and-persist half (tiled inference over a raster too large to load
whole) lives in ``inference_tools.export_predictions`` (its ``raster_path`` regime), so a breeder
can review the persisted predictions (or simply trust a tens-of-minutes tiled run once) before
re-running this comparatively cheap plant-mapping + aggregation step, or re-running it against a
different plant CSV without repeating the raster pass.

``deliver_orthomosaic_plant_counts`` reads that bucket + plant CSV(s) into a per-plant count CSV,
gated by the same measurement-integrity door every other count delivery goes through.

See ``pipelines.postprocessing.orthomosaic_mapping`` (georeferencing) for the primitive this
composes; this tool implements no new CV capability, it wires already-built pieces together for an
agent (or the breeder via the GUI) to actually invoke.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tcip_mcp.audit import audited
from tcip_mcp.server import mcp

logger = logging.getLogger(__name__)

_PER_PLANT_VALUE_KEY = "count"
"""The quantity every row of this delivery holds, named once for the aggregation and the check."""


@mcp.tool()
@audited
def deliver_orthomosaic_plant_counts(
    predictions_dir: str,
    raster_path: str,
    plant_csv_paths: list[str],
    output_csv_path: str,
    delivered_phenotype: str,
    crop: str = "",
    pipeline_version: str = "",
    nn_tolerance_m: float | None = None,
    acknowledge_unvalidated: bool = False,
) -> dict:
    """Per-plant detection counts from a persisted orthomosaic prediction bucket plus plant CSV(s).

    A prediction bucket here is a directory of prediction documents, not a score bin, held
    immutable once a human reviews it.

    Reads back the whole-mosaic predictions ``export_predictions``'s ``raster_path`` regime
    persisted (never re-runs the expensive tiled pass), resolves each detection's box centroid
    to a real-world coordinate via the raster's own georeferencing tags, and matches it to the
    nearest plant
    (:func:`assign_detections_to_plants`). Every geolocated plant in ``plant_csv_paths`` gets
    exactly one row in the delivered CSV: a plant with one or more assigned detections gets their
    sum, a plant the scan covered but matched no detection near gets an explicit ``0`` (a real
    measured absence, not a missing observation, since the tiled scan covers the whole raster and
    so every plant's location), never a fabricated value for a plant this delivery never actually
    covered.  A detection farther than the tolerance from every plant is excluded from any
    plant's count (counted in ``n_unmapped``, never force-assigned to the nearest plant
    regardless of distance).

    Identity door: the georeferencing that decides which plant each detection belongs to comes
    from the caller's ``raster_path``, so a raster that is not the one the bucket was produced on
    silently re-attributes every count (a pixel-identical copy with a moved tiepoint shifts each
    detection onto a neighbouring plant; a far-shifted one reads every plant as zero). The supplied
    raster is therefore checked against the identity the bucket recorded at export time, content
    and georeferencing alike (:func:`~tcip_mcp.pipelines.raster_source.
    georeferenced_raster_identity_mismatch`), and a bucket carrying no recorded identity is refused
    rather than trusted: there is nothing to check against, and no per-plant attribution is
    trustworthy without one.

    Delivery gate: the count is the phenotype, so this refuses a bare write of an unvalidated
    count operating point (read from the bucket's own ``operating_point.json``, never trusted
    from a caller string) unless ``acknowledge_unvalidated=True`` ships a clearly-flagged
    provisional CSV stamped ``operating_point_validated=false``. A tiled bucket's ``tile_size`` gates
    the same way (the tile edge scales the per-image counts the per-plant value sums), reusing
    the identical ``export_aggregated_csv`` gate every other per-plant delivery goes through, not
    a second implementation of it.

    Meaning door: this runs the same per-plant-count-aggregate precondition its nested writer runs,
    and runs it first, before the raster identity is resolved or a single prediction is read. A
    number with no confirmed meaning has nothing for a raster identity to attribute, so the
    precondition reports on its own rather than behind a refusal about the mosaic.

    The producer identity this returns is read back from the tail ``export_aggregated_csv`` itself
    returns beside the CSV path, never re-derived here: a bucket naming an experiment the store
    cannot answer for reports its producer unknown, while a bespoke bucket carrying a real
    checkpoint hash and no experiment keeps that hash, the one shared derivation
    (``delivered_tail``/``delivered_provenance``) behind both the CSV's own cells and this
    response. ``validation_record`` names the record behind a validated count, and is empty
    otherwise.

    Args:
        predictions_dir: The bucket ``export_predictions``'s ``raster_path`` regime persisted.
        raster_path: The same georeferenced raster the bucket's predictions were produced from
            (needed to resolve each detection's pixel position to a real-world coordinate). Given
            by the caller rather than taken from the sidecar's recorded ``raster_path``, which
            names a location that may hold a different file by now, and checked against the
            bucket's recorded raster identity before anything is resolved through it.
        plant_csv_paths: One or more plant-locations CSVs (columns ``plot_name``,
            ``accession_name``, ``WGS84_centroid_x/y``, …).
        output_csv_path: Where to write the delivered per-plant CSV. A relative path resolves
            against the platform state root, never the server process's cwd.
        delivered_phenotype: The crop-vocabulary delivered phenotype this CSV ships under, resolved
            to the registered trait whose spec delivers it and whose confirmed operationalization
            this delivery rests on.
        crop: Crop species name.
        pipeline_version: Pipeline identifier.
        nn_tolerance_m: Nearest-neighbour match tolerance (m). ``None`` (default) derives it from
            the plant grid's own spacing (:func:`assign_detections_to_plants`'s own default),
            never a pinned constant.
        acknowledge_unvalidated: Write the CSV even when the count operating point or tile_size
            is unvalidated, stamping the un-validated dimension ``false`` so the
            un-trustworthiness travels downstream.
    """
    from tcip_annotation.json_io import prediction_documents
    from tcip_mcp.project_paths import resolve_output_path

    output_csv_path = str(resolve_output_path(output_csv_path))
    pred_dir = Path(predictions_dir)
    if not pred_dir.is_dir():
        return {"error": f"predictions_dir not found: {predictions_dir}"}
    if not Path(raster_path).is_file():
        return {"error": f"raster_path not found: {raster_path}"}
    missing = [p for p in plant_csv_paths if not Path(p).is_file()]
    if missing:
        return {"error": f"plant CSV(s) not found: {missing}"}

    from tcip_mcp.operationalization import (
        PER_PLANT_COUNT_AGGREGATE,
        check_operationalization,
        resolve_trait_and_record,
        resolve_trait_for_phenotype,
    )
    from tcip_mcp.traits import TraitUnknownError

    # First refusal in this body: how a number was attributed says nothing until it has a meaning.
    try:
        trait = resolve_trait_for_phenotype(delivered_phenotype)
        spec, record, _specs_dir = resolve_trait_and_record(trait, PER_PLANT_COUNT_AGGREGATE)
    except (TraitUnknownError, ValueError) as exc:
        return {"error": str(exc)}
    # This door never delivers a crossing kind, so it has no registry to check a positive class against.
    stated = check_operationalization(
        spec, record, PER_PLANT_COUNT_AGGREGATE, delivered_phenotype=delivered_phenotype,
        value_keys=[_PER_PLANT_VALUE_KEY], registry=None)
    if not stated.ok:
        return {"error": stated.message}

    pred_files = prediction_documents(pred_dir)
    if not pred_files:
        return {"error": f"no prediction file(s) found in {predictions_dir}"}

    from tcip_mcp.pipelines.raster_source import georeferenced_raster_identity_mismatch
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    sidecar = read_operating_point_sidecar(predictions_dir) or {}
    recorded_identity = sidecar.get("raster_content_identity")
    if recorded_identity is None:
        return {"error": (
            f"delivery refused: the bucket at {predictions_dir} records no raster content "
            "identity, so there is nothing to check raster_path against. Every count this "
            "delivery writes is attributed to a plant through the supplied raster's own "
            "georeferencing, so the bucket must carry the identity of the raster it was produced "
            "on: produce it with export_predictions's raster_path regime, which records that "
            "identity into the bucket's operating_point.json."
        )}
    try:
        identity_mismatch = georeferenced_raster_identity_mismatch(recorded_identity, raster_path)
    except ValueError as exc:
        return {"error": f"delivery refused: {exc}"}
    if identity_mismatch is not None:
        return {"error": (
            f"delivery refused: {raster_path} is not the raster the bucket at {predictions_dir} "
            f"was produced on. {identity_mismatch}"
        )}

    from tcip_annotation.json_io import UnreadableLabelDocument, detection_annotations
    from tcip_annotation.state import bbox_of

    boxes: list[list[float]] = []
    try:
        for f in pred_files:
            for a in detection_annotations(str(f)):
                b = bbox_of(a.geometry)
                boxes.append([b.x1, b.y1, b.x2, b.y2])
    except UnreadableLabelDocument as exc:
        return {"error": str(exc)}
    detections = {"boxes": boxes}

    from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import (
        GeoreferencingError,
        OrthomosaicGeoreference,
        RotatedRasterError,
        assign_detections_to_plants,
    )
    from tcip_mcp.pipelines.postprocessing.plant_mapping import read_plant_csvs

    try:
        georef = OrthomosaicGeoreference.from_file(raster_path)
    except (GeoreferencingError, RotatedRasterError) as exc:
        return {"error": str(exc)}

    plants = read_plant_csvs([Path(p) for p in plant_csv_paths])
    if not plants:
        return {"error": f"no georeferenced plants parsed from {plant_csv_paths}"}

    assignments = assign_detections_to_plants(
        detections, georef, plants, nn_tolerance_m=nn_tolerance_m)
    mapped = [a for a in assignments if a.plot_name is not None]
    n_unmapped = len(assignments) - len(mapped)

    from tcip_mcp.pipelines.postprocessing.aggregation import aggregate_per_plant, export_aggregated_csv

    mapped_plant_ids = {a.plot_name for a in mapped}
    records = [
        {"plant_id": a.plot_name, "plant_id_source": a.source,
         "plant_id_distance_m": a.distance_m, _PER_PLANT_VALUE_KEY: 1,
         "measurement_document": "operating_point"}
        for a in mapped
    ]
    # Every plant the scan covers gets a row: an explicit 0 for a plant that matched no
    # detection, never silently absent from the delivery (see the docstring above).
    records += [{"plant_id": p.plot_name, _PER_PLANT_VALUE_KEY: 0,
                "measurement_document": "operating_point"}
                for p in plants if p.plot_name not in mapped_plant_ids]

    agg = aggregate_per_plant(records, strategy="sum", plant_id_key="plant_id",
                              value_key=_PER_PLANT_VALUE_KEY)

    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE,
        DeliveryRefused,
        record_delivery_binding_event,
        reconcile_operating_point_validity,
    )

    recon = reconcile_operating_point_validity([predictions_dir], trait=trait)
    # The raw asserted identity; export_aggregated_csv's own delivered_tail corroborates it, the
    # one shared derivation, so nothing here re-derives that identity a second time.
    provenance = {"producer_model_sha256": sidecar.get("checkpoint_sha256"),
                 "producing_experiment_id": sidecar.get("experiment_id")}

    try:
        csv_path, tail = export_aggregated_csv(
            agg, output_csv_path, delivered_phenotype=delivered_phenotype, crop=crop,
            pipeline_version=pipeline_version, provenance=provenance,
            pred_dirs=[predictions_dir],
            acknowledge_unvalidated=acknowledge_unvalidated,
        )
    except DeliveryRefused as exc:
        # operating_point_validated is the operating_point dimension's own cleared reference;
        # unvalidated_dimensions names every refusing dimension (claim_scope and scale included).
        refusal = {
            "error": str(exc),
            "operating_point_validated": exc.gate.stamp.get("operating_point", VALIDATED_FALSE),
            "unvalidated_dimensions": exc.gate.unvalidated_cell(),
            "n_detections": len(assignments), "n_mapped": len(mapped),
            "n_unmapped": n_unmapped,
        }
        if "tile_size" in exc.gate.stamp:
            refusal["tile_size_validated"] = exc.gate.stamp["tile_size"]
        return refusal
    except ValueError as exc:
        # aggregation.py's own refusal already names the failing bucket and why, through the same
        # binding_notes_text helper this door's own (identically-scoped) recon would just repeat.
        return {"error": str(exc),
                "n_detections": len(assignments), "n_mapped": len(mapped),
                "n_unmapped": n_unmapped}

    # This door always delivers under the aggregate count kind; trait was already resolved above,
    # at the meaning-refusal check.
    delivery_kind = PER_PLANT_COUNT_AGGREGATE
    record_delivery_binding_event("deliver_orthomosaic_plant_counts", output_csv_path,
                                  [predictions_dir], recon["bindings"],
                                  measurement_documents=["operating_point"], scale_document=None,
                                  trait=trait, delivery_kind=delivery_kind)

    return {
        "csv_path": csv_path,
        "n_plants": len(agg),
        "n_plants_zero_count": sum(1 for r in agg if r.get("value") == 0),
        "n_detections": len(assignments),
        "n_mapped": len(mapped),
        "n_unmapped": n_unmapped,
        "operating_point_validated": tail["operating_point_validated"],
        "unvalidated_dimensions": tail["unvalidated_dimensions"],
        "checkpoint_sha256": tail["producer_model_sha256"],
        "producing_experiment_id": tail["producing_experiment_id"],
        "validation_record": tail["validation_record"],
    }
