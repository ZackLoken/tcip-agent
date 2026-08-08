"""Orthomosaic MCP tools: tiled inference over a huge georeferenced raster, and per-plant
delivery from the resulting predictions plus a plant-locations CSV.

Two composable steps, the same detect-and-persist / map-and-deliver shape
``phenology_tools``'s ``build_plant_mapping``/``compute_phenology`` already use for the
per-image-EXIF case, so a breeder can review the persisted predictions (or simply trust a
tens-of-minutes tiled run once) before re-running the comparatively cheap plant-mapping +
aggregation step, or re-running that step against a different plant CSV without repeating the
raster pass:

    run_orthomosaic_inference        checkpoint + a raster too large to load whole -> a
                                      persisted prediction bucket (one prediction file for the
                                      whole mosaic, plus the same operating_point.json
                                      convention every other bucket carries)
    deliver_orthomosaic_plant_counts that bucket + plant CSV(s) -> a per-plant count CSV,
                                      gated by the same measurement-integrity door every other
                                      count delivery goes through

See ``pipelines.postprocessing.orthomosaic_mapping`` (georeferencing), ``pipelines.raster_source``
(windowed pixel reads) and ``pipelines.inference.generic_predictor`` (the tiled pass) for the
primitives this composes; neither tool implements new CV capability, both wire
already-built pieces together for an agent (or the breeder via the GUI) to actually invoke.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tcip_mcp.audit import audited
from tcip_mcp.pipelines.resolution import DEFAULT_CONF, DEFAULT_MAX_DETS, DEFAULT_NMS_IOU
from tcip_mcp.server import mcp

logger = logging.getLogger(__name__)


@mcp.tool()
@audited
def run_orthomosaic_inference(
    checkpoint_path: str,
    raster_path: str,
    output_dir: str,
    device: str | None = None,
    conf_threshold: float = DEFAULT_CONF,
    tile_size: int | None = None,
    overlap: float | None = None,
    tile_batch_size: int = 96,
    global_nms_iou: float = DEFAULT_NMS_IOU,
    max_dets: int = DEFAULT_MAX_DETS,
    postprocess: str = "nms",
    require_masks: bool = True,
    experiment_id: str | None = None,
    overwrite: bool = False,
    acknowledge_unvalidated: bool = False,
) -> dict:
    """Tiled detection/instance_seg inference over a whole georeferenced orthomosaic.

    Sources tiles from the windowed raster layer (:func:`~tcip_mcp.pipelines.raster_source.
    open_raster`, GDAL-backed for a GeoTIFF, its decoded blocks held in GDAL's own budgeted
    cache) rather than decoding the raster whole, so this reaches a raster too large to load
    into memory (a real 90 GB, 141130x239921px, 4-band mosaic). Always tiled: there is
    no untiled option, the whole point of this tool is a raster too large for one. Works for
    ``instance_seg`` the same as ``detection``: a checkpoint's masks thread through
    ``predict_tiled_from_reader`` in its tile-local-patch shape, and
    ``write_predictions_json`` already converts that into a real polygon at the mask's
    full-mosaic-space offset, no special case needed for this raster source.

    Persists one prediction bucket at ``output_dir``: since there is no natural
    directory-of-per-plant-images shape for a whole-mosaic capture, "one image" is the whole
    raster, so the bucket holds exactly one ``<raster stem>.json`` prediction file (in
    full-mosaic pixel space) plus the same ``operating_point.json`` sidecar convention every
    other bucket carries (producer identity, the resolved operating point, tile-geometry
    validity, mask-binarize provenance when masks are present). Downstream code that reads a
    bucket's sidecar generically (``reconcile_operating_point_validity``,
    ``reconcile_tile_size_validity``) needs no special case for this bucket shape.

    A bucket that already carries review verdicts is immutable, mirroring
    ``export_predictions``: by default a re-run is redirected to a fresh run-scoped bucket
    (``<dir>@r2``, next free) and the dir actually written is returned as ``output_dir``.
    ``overwrite=True`` writes in place only when the bucket has zero verdicts, else refuses.

    This is the raw, detect-and-persist step: the persisted operating point is never stamped
    validated here (mirrors ``export_predictions``'s own raw path), a validated per-plant count
    is earned later, at delivery (``deliver_orthomosaic_plant_counts``, via a calibrated
    conf and/or the breeder review-confirmation loop). The one dimension this step does gate on
    write is ``tile_size``: a tiled run whose tile edge fell back to the fabricated default (no
    persisted training geometry, no explicit override) refuses to write here unless
    ``acknowledge_unvalidated=True``, the same ``tile_size_gate_flag`` gate every other tiled
    delivery door already uses; conf is always allowed to persist unvalidated at this step, the
    same asymmetry ``export_predictions`` already has.

    Args:
        checkpoint_path: Path to a bespoke model .pt checkpoint (detection or instance_seg task).
        raster_path: Path to the georeferenced GeoTIFF orthomosaic.
        output_dir: Directory for the persisted prediction bucket.
        device: Device to use ('cuda' or 'cpu').
        conf_threshold: Minimum confidence score.
        tile_size: Sliding-window tile edge (px). ``None`` (default) derives it from the
            checkpoint's training tile geometry; a checkpoint with no persisted geometry falls
            back to the documented default with a warning, stamped unvalidated so the tile_size
            gate below refuses unless acknowledged.
        overlap: Fractional tile overlap. ``None`` derives from the checkpoint (else the
            documented default).
        tile_batch_size: Tiles per forward batch.
        global_nms_iou: Cross-tile global NMS IoU threshold.
        max_dets: Full-frame detection cap (after the cross-tile merge).
        postprocess: Cross-tile merge, "nms" suppresses overlaps, "nmm" unions boxes split
            across a tile seam.
        require_masks: Collect masks for an ``instance_seg`` checkpoint (ignored for
            ``detection``, which never carries masks).
        experiment_id: The run that produced the checkpoint, for provenance. Best-effort
            resolved (the checkpoint's own stamp, then the registry) when omitted.
        overwrite: Write into ``output_dir`` even if it exists. Refused if the bucket has review
            verdicts; the default (False) auto-redirects to a fresh bucket instead.
        acknowledge_unvalidated: Write the bucket even when tile_size has no real basis,
            stamping ``tile_size_validated=false`` on the sidecar so the un-trustworthiness
            travels with it rather than writing silently.
    """
    if not Path(checkpoint_path).is_file():
        return {"error": f"Checkpoint not found: {checkpoint_path}"}
    if not Path(raster_path).is_file():
        return {"error": f"raster_path not found: {raster_path}"}

    from tcip_mcp.prediction_buckets import BucketHasVerdicts, resolve_writable_bucket
    from tcip_mcp.project_paths import resolve_output_path, resolve_state

    # A relative output_dir resolves against the project root, never the server process's cwd.
    out_path = resolve_output_path(output_dir)
    parent, base_name = out_path.parent, out_path.name
    review_state_dir = resolve_state(Path(".tcip") / "state")
    try:
        resolution = resolve_writable_bucket(
            review_state_dir, base_name, lambda n: [parent / n], overwrite=overwrite)
    except BucketHasVerdicts as exc:
        return {"error": str(exc), "verdict_count": exc.count,
                "suggested_bucket": str(parent / exc.suggested)}
    out = parent / resolution.name

    from tcip_mcp.model_registry import resolve_model_identity
    from tcip_mcp.pipelines.inference.predictor import build_predictor, resolve_tile_geometry

    predictor = build_predictor(
        checkpoint_path=checkpoint_path, device=device, score_threshold=conf_threshold,
        nms_iou=global_nms_iou, max_dets=max_dets,
    )
    identity = resolve_model_identity(checkpoint_path, experiment_id=experiment_id)

    resolved_tile, tile_size_source, resolved_overlap, overlap_source = resolve_tile_geometry(
        predictor, tile_size=tile_size, overlap=overlap)
    geometry_warning = None
    if tile_size_source == "default":
        geometry_warning = (
            "checkpoint carries no training tile geometry; using default "
            f"{resolved_tile}, counts may not match training scale. Retrain (geometry now "
            "persisted) or pass tile_size explicitly."
        )
        logger.warning(geometry_warning)

    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE, check_delivery_gate, raw_operating_point, tile_size_gate_flag,
    )

    # This tool always tiles (a raster too large to load whole has no untiled alternative), so
    # tiled=True unconditionally; "default" source since it is not a caller choice at all, the
    # same vocabulary raw_operating_point already uses for a param no caller can override. Every
    # input this bundle needs (conf/tile geometry/max_dets) is already resolved, so the tile_size
    # gate is checked here, before the expensive tiled pass runs, not after: a refusal is cheap,
    # a wasted tens-of-minutes pass over the whole raster is not.
    op_bundle = raw_operating_point(
        conf=conf_threshold, cross_tile_nms=global_nms_iou, tiled=True, tile_size=resolved_tile,
        max_dets=max_dets, tile_size_source=tile_size_source, tiled_source="default",
    )
    op_provenance = op_bundle.to_provenance()["operating_point"]

    tile_ref = tile_size_gate_flag(op_provenance)
    tile_flags = {"tile_size": tile_ref} if tile_ref is not None else {}
    gate = check_delivery_gate(tile_flags, acknowledge_unvalidated=acknowledge_unvalidated)
    if not gate.ok:
        return {"error": gate.reason, "tile_size_validated": tile_ref}
    tile_size_validated = gate.stamp.get("tile_size")

    from tcip_mcp.pipelines.raster_source import open_raster

    # The model's own in_chans is the channel routing hint; the reader's real band count is
    # checked against it inside predict_tiled_from_reader before any tile is read.
    with open_raster(raster_path, predictor.in_chans) as reader:
        result = predictor.predict_tiled_from_reader(
            reader, tile_size=resolved_tile, overlap=resolved_overlap,
            tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou,
            postprocess=postprocess, require_masks=require_masks, source_label=str(raster_path),
        )

    from tcip_mcp.tools.inference_tools import resolve_decode_id_map

    # No images_dir for a raster source: a bespoke checkpoint with no recorded training id_map
    # decodes to the raw 0-indexed id as its name, the same honest degraded fallback
    # write_predictions_json already documents for that case.
    id_map = resolve_decode_id_map(predictor, None)

    from datetime import datetime, timezone

    from tcip_mcp.pipelines.postprocessing.export import (
        mask_binarize_provenance, write_predictions_json,
    )
    from tcip_mcp.utils.atomic_io import atomic_write_json

    out.mkdir(parents=True, exist_ok=True)
    sha = identity["sha256"]
    producer = f"model:{Path(checkpoint_path).stem}" + (f"@{sha[:12]}" if sha else "")
    pred_path = out / f"{Path(raster_path).stem}.json"
    write_predictions_json(pred_path, result, created_by=producer, id_map=id_map)
    has_masks = bool(result.get("masks"))

    produced_at = datetime.now(timezone.utc).isoformat()
    op_stamp = {
        "operating_point": op_provenance,
        "id_map": id_map,
        # is_shippable is always False here (conf is never validated at this raw-persist step),
        # tile_size_validated still floors it explicitly so the same "acknowledged but unvalidated
        # tile scale" case export_predictions guards against can't read back as validated.
        "validated": bool(op_bundle.is_shippable) and tile_size_validated != VALIDATED_FALSE,
        "tile_size_validated": tile_size_validated,
        "shippable_issues": op_bundle.shippable_issues(),
        "checkpoint": Path(checkpoint_path).stem,
        "checkpoint_sha256": sha,
        "experiment_id": identity["experiment_id"],
        "raster_path": str(raster_path),
        "produced_at": produced_at,
    }
    if has_masks:
        op_stamp["mask_binarize"] = mask_binarize_provenance()
    atomic_write_json(out / "operating_point.json", op_stamp)

    exp_id = identity["experiment_id"]
    if exp_id:
        try:
            from tcip_mcp.experiments import update_lineage

            update_lineage(exp_id, predictions=str(out))
        except Exception:
            logger.warning("could not link predictions into experiment lineage", exc_info=True)

    response = {
        "output_dir": str(out),
        "prediction_path": str(pred_path),
        "bucket_redirected": resolution.redirected,
        "requested_output_dir": output_dir if resolution.redirected else None,
        "detection_count": result.get("count", 0),
        "tiles": result.get("tiles"),
        "operating_point": op_provenance,
        "overlap": resolved_overlap,
        "overlap_source": overlap_source,
        "tile_size_validated": tile_size_validated,
        "checkpoint_sha256": sha,
        "experiment_id": exp_id,
        "produced_at": produced_at,
    }
    if geometry_warning:
        response["warning"] = geometry_warning
    return response


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

    Reads back the whole-mosaic predictions ``run_orthomosaic_inference`` persisted (never
    re-runs the expensive tiled pass), resolves each detection's box centroid to a real-world
    coordinate via the raster's own georeferencing tags, and matches it to the nearest plant
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
        predictions_dir: The bucket ``run_orthomosaic_inference`` persisted.
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
