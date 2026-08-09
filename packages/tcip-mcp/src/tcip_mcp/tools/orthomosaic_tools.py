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


@mcp.tool()
@audited
def deliver_orthomosaic_plant_counts(
    predictions_dir: str,
    raster_path: str,
    plant_csv_paths: list[str],
    output_csv_path: str,
    trait_name: str,
    crop: str = "",
    pipeline_version: str = "",
    nn_tolerance_m: float | None = None,
    measurement_validated: str | None = None,
    acknowledge_unvalidated: bool = False,
) -> dict:
    """Per-plant detection counts from a persisted orthomosaic prediction bucket + plant CSV(s).

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

    Delivery gate: the count is the phenotype, so this refuses a bare write of an unvalidated
    count operating point (read from the bucket's own ``operating_point.json``, never trusted
    from a caller string) unless ``acknowledge_unvalidated=True`` ships a clearly-flagged
    provisional CSV stamped ``measurement_validated=false``. A tiled bucket's ``tile_size`` gates
    the same way (the tile edge scales the per-image counts the per-plant value sums), reusing
    the identical ``export_aggregated_csv`` gate every other per-plant delivery goes through, not
    a second implementation of it.

    Args:
        predictions_dir: The bucket ``export_predictions``'s ``raster_path`` regime persisted.
        raster_path: The same georeferenced raster the bucket's predictions were produced from
            (needed to resolve each detection's pixel position to a real-world coordinate; not
            read from the bucket, since the sidecar's own ``raster_path`` is a provenance record,
            not a promise this tool re-trusts it against a different file on disk).
        plant_csv_paths: One or more plant-locations CSVs (columns ``plot_name``,
            ``accession_name``, ``WGS84_centroid_x/y``, …).
        output_csv_path: Where to write the delivered per-plant CSV. A relative path resolves
            against the project root, never the server process's cwd.
        trait_name: Name of the trait being measured (a CSV column value, not validated against
            the trait registry, count traits carry no physical unit to cross-check).
        crop: Crop species name.
        pipeline_version: Pipeline identifier.
        nn_tolerance_m: Nearest-neighbour match tolerance (m). ``None`` (default) derives it from
            the plant grid's own spacing (:func:`assign_detections_to_plants`'s own default),
            never a pinned constant.
        measurement_validated: An optional caller assertion of the count operating point's
            validity. It only lowers the result: the real state is read from the bucket's
            ``operating_point.json`` sidecar and floored against this.
        acknowledge_unvalidated: Write the CSV even when the count operating point or tile_size
            is unvalidated, stamping the un-validated dimension ``false`` so the
            un-trustworthiness travels downstream.
    """
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

    pred_files = sorted(f for f in pred_dir.glob("*.json") if f.name != "operating_point.json")
    if not pred_files:
        return {"error": f"no prediction file(s) found in {predictions_dir}"}

    from tcip_annotation import json_io
    from tcip_annotation.state import bbox_of

    boxes: list[list[float]] = []
    for f in pred_files:
        for a in json_io.read_annotations(str(f)):
            b = bbox_of(a.geometry)
            boxes.append([b.x1, b.y1, b.x2, b.y2])
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
         "plant_id_distance_m": a.distance_m, "count": 1}
        for a in mapped
    ]
    # Every plant the scan covers gets a row: an explicit 0 for a plant that matched no
    # detection, never silently absent from the delivery (see the docstring above).
    records += [{"plant_id": p.plot_name, "count": 0}
                for p in plants if p.plot_name not in mapped_plant_ids]

    agg = aggregate_per_plant(records, strategy="sum", plant_id_key="plant_id", value_key="count")

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    sidecar = read_operating_point_sidecar(predictions_dir) or {}
    provenance = {
        "producer_model_sha256": sidecar.get("checkpoint_sha256"),
        "experiment_id": sidecar.get("experiment_id"),
        "produced_at": sidecar.get("produced_at"),
    }

    try:
        csv_path = export_aggregated_csv(
            agg, output_csv_path, trait_name=trait_name, crop=crop,
            pipeline_version=pipeline_version, provenance=provenance,
            measurement_validated=measurement_validated, pred_dirs=[predictions_dir],
            acknowledge_unvalidated=acknowledge_unvalidated,
        )
    except ValueError as exc:
        return {"error": str(exc), "n_detections": len(assignments), "n_mapped": len(mapped),
                "n_unmapped": n_unmapped}

    # The CSV's own measurement_validated column is what export_aggregated_csv's gate actually
    # stamped; read it back rather than re-deriving the same decision a second time here.
    import csv as _csv

    with open(csv_path, newline="") as csv_f:
        rows = list(_csv.DictReader(csv_f))
    validated_stamp = rows[0]["measurement_validated"] if rows else None

    return {
        "csv_path": csv_path,
        "n_plants": len(agg),
        "n_plants_zero_count": sum(1 for r in agg if r.get("value") == 0),
        "n_detections": len(assignments),
        "n_mapped": len(mapped),
        "n_unmapped": n_unmapped,
        "measurement_validated": validated_stamp,
        "checkpoint_sha256": sidecar.get("checkpoint_sha256"),
        "experiment_id": sidecar.get("experiment_id"),
    }
