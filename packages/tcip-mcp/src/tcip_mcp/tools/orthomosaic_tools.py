"""Orthomosaic MCP tools: per-plant delivery from a persisted whole-raster prediction bucket plus
a plant-locations CSV.

The map-and-deliver half of the same detect-and-persist / map-and-deliver shape
``phenology_tools``'s ``build_plant_mapping``/``deliver_phenology_milestones`` already use for the
per-image-EXIF case: the detect-and-persist half (tiled inference over a raster too large to load
whole) lives in ``inference_tools.run_inference`` (its ``raster_path`` regime), so a breeder
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
from typing import TYPE_CHECKING

from tcip_mcp.audit import audited
from tcip_mcp.server import mcp

if TYPE_CHECKING:
    from tcip_mcp.pipelines.resolution import Acknowledgement

logger = logging.getLogger(__name__)

_PER_PLANT_VALUE_KEY = "count"
"""The quantity every row of this delivery holds, named once for the aggregation and the check."""


def orthomosaic_plant_counts(
    predictions_dir: str,
    raster_path: str,
    plant_registry: str,
    output_csv_path: str,
    delivered_phenotype: str,
    crop: str = "",
    pipeline_version: str = "",
    nn_tolerance_m: float | None = None,
    canopy_subject: str = "",
    project_root: str | Path | None = None,
    acknowledgement: Acknowledgement | None = None,
) -> dict:
    """Per-plant detection counts from a persisted orthomosaic prediction bucket plus plant CSV(s).

    The core the MCP tool ``deliver_orthomosaic_plant_counts`` (no ``project_root``, no
    ``acknowledgement``: the process-pinned root, never a provisional delivery) and the web
    results route (a guarded project root, a real breeder acknowledgement) both call. Raises
    rather than returns an error dict: ``operationalization.OperationalizationRefused`` for an
    unrecorded, unconfirmed, or
    since-withdrawn ``per_plant_count_aggregate`` meaning (carrying the check, no counts);
    ``pipelines.resolution.DeliveryRefused`` for the writer's own gate refusal (carrying the gate,
    with this call's own counts-bearing facts attached); ``pipelines.resolution.
    CountDeliveryRefused`` for everything else this door refuses on today (a missing bucket or
    raster, a conflicting regime, an unregistered or rewritten plant registry, an empty bucket, a
    raster identity mismatch, a canopy-segment refusal), each carrying the same facts the tool's
    own ``{"error": ...}`` response carries.

    A prediction bucket here is a directory of prediction documents, not a score bin, held
    immutable once a human reviews it.

    Reads back the whole-mosaic predictions ``run_inference``'s ``raster_path`` regime
    persisted (never re-runs the expensive tiled pass), resolves each detection's box centroid
    to a real-world coordinate via the raster's own georeferencing tags, and, absent
    ``canopy_subject``, matches it to the nearest plant (:func:`assign_detections_to_plants`).
    Absent ``canopy_subject``, every in-frame geolocated plant ``plant_registry`` names (its own
    projected position inside the raster's own recorded frame, :func:`~tcip_mcp.pipelines.postprocessing.
    orthomosaic_mapping.plants_in_frame`) gets exactly one row in the delivered CSV: a plant with
    one or more assigned detections gets their sum, a plant the scan covered but matched no
    detection near gets an explicit ``0`` (a real measured absence, not a missing observation,
    since the tiled scan covers the whole raster and so every in-frame plant's location), never a
    fabricated value for a plant this delivery never actually covered. A registry plant outside
    the raster's own frame gets no row under either regime and is disclosed by name
    (``plants_outside_raster``): the raster never pictures it, so a zero there would be a
    fabricated absence, not a measured one. A detection farther than the tolerance from every
    in-frame plant is excluded from any plant's count (counted in ``n_unmapped``, never
    force-assigned to the nearest plant regardless of distance); this counts detections this
    delivery could not attribute, a different mechanism from the walked-image mapping's own
    per-capture unattributed count (``plant_mapping.MappingBuild.unattributed``).

    ``canopy_subject`` switches the door to its segment regime: a detection is attributed by
    containment in a canopy boundary a person has accepted into the raster's own label document
    (:mod:`~tcip_mcp.pipelines.postprocessing.segment_attribution`), the boundary itself tied to
    the one registry plant whose own projected position it contains, never by nearest-neighbour
    distance. ``segment`` means the detection's box centroid fell inside a boundary a person
    accepted, never a mask-level or area measurement, and the boundary is accepted, not validated
    the way a detection's own confidence is: the registry position's own error is bounded by
    nothing but the disclosed clearance
    (:class:`~tcip_mcp.pipelines.postprocessing.segment_attribution.TiedSegment.clearance_m`), so
    a position displaced by more than its clearance places the plant in a neighbour's canopy with
    every check here passing. The canopy document's own location is derived from the caller's
    ``raster_path`` itself (its canonical position under a registered dataset the bucket shares,
    via :func:`~tcip_mcp.dataset_layout.annotation_path_for_image`), never a path the caller
    names directly; the document's identity is bound to the raster's own content, geotransform
    and dataset id (the checks below), so two content-identical rasters registered at different
    canonical positions under the same dataset resolve to two different label documents. A tied
    plant whose segment's own detection also lies in another segment gets no row,
    since its count would be an undisclosed lower bound, and is disclosed by name
    (``plants_with_ambiguous_detections``); a tied plant whose segment holds no detection at all
    gets an explicit ``0``; an in-frame plant inside no segment gets no row and is disclosed by
    name (``plants_without_segment``). Stating both ``canopy_subject`` and ``nn_tolerance_m``
    refuses: the two regimes' own match parameters are never mixed.

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
    from a caller string), reusing the identical ``export_aggregated_csv`` gate every other
    per-plant delivery goes through, not a second implementation of it. A tiled bucket's
    ``tile_size`` gates the same way (the tile edge scales the per-image counts the per-plant
    value sums). ``acknowledgement`` is the breeder's own act of shipping this delivery
    unvalidated (the web results route's per-plant count export is the one surface that builds
    one); the MCP tool ``deliver_orthomosaic_plant_counts`` builds none, so an unvalidated
    dimension always refuses through it, and a mosaic whose training run reserved no calibration
    region has no route to a delivered CSV through that door alone.

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

    Delivery-event disclosure: every registry CSV named above is hashed against the byte digest
    ``register_plant_registry`` recorded for it (:func:`~tcip_mcp.pipelines.postprocessing.
    plant_mapping.verify_registry_csv_bytes`), and a missing or rewritten file refuses by name
    before any plant or prediction is read, so every CSV this delivery reads is verified or the
    delivery never happens. Absent ``canopy_subject``, the delivery event this door's own
    ``export_aggregated_csv`` call records carries a ``PlantRegistryDisclosure``
    (``delivery_events_schema.py``): the registry it read, the raster identity every count is
    attributed through, the matched tolerance and its source, and this delivery's own
    unattributed-detection count, the whole-raster counterpart of the walked mapping's own
    ``PlantMappingDisclosure`` the phenology doors record. Under ``canopy_subject``, the event
    carries a ``CanopySegmentDisclosure`` instead: the same registry and raster identity, the
    canopy document read and every segment tie resolved from it (each with its own clearance,
    never a match tolerance), and the three plant-name lists this door's own response also
    carries (``plants_outside_raster``, ``plants_without_segment``,
    ``plants_with_ambiguous_detections``).

    Args:
        predictions_dir: The bucket ``run_inference``'s ``raster_path`` regime persisted.
        raster_path: The same georeferenced raster the bucket's predictions were produced from
            (needed to resolve each detection's pixel position to a real-world coordinate). Given
            by the caller rather than taken from the sidecar's recorded ``raster_path``, which
            names a location that may hold a different file by now, and checked against the
            bucket's recorded raster identity before anything is resolved through it.
        plant_registry: The name of a plant registry already registered under this project by
            ``register_plant_registry``.
        output_csv_path: Where to write the delivered per-plant CSV. A relative path resolves
            against the platform state root, never the server process's cwd.
        delivered_phenotype: The crop-vocabulary delivered phenotype this CSV ships under, resolved
            to the registered trait whose spec delivers it and whose confirmed operationalization
            this delivery rests on.
        crop: Crop species name.
        pipeline_version: Pipeline identifier.
        nn_tolerance_m: Nearest-neighbour match tolerance (m). ``None`` (default) derives it from
            the plant grid's own spacing (:func:`~tcip_mcp.pipelines.postprocessing.
            orthomosaic_mapping.resolve_nn_tolerance_m`), never a pinned constant. Refused
            alongside ``canopy_subject``.
        canopy_subject: The class registry subject naming a canopy boundary in the raster's own
            label document. Empty (default) runs the nearest-neighbour regime; set, this door
            attributes by segment containment instead, and no model architecture is prescribed
            for how the boundary itself was produced (a hand trace, an accepted SAM proposal, or
            an accepted bespoke instance-segmentation output all admit the same way).
        project_root: The project this delivery's meaning-record reads and delivery event belong
            to, and where ``plant_registry`` is looked up. ``None`` (the MCP tool) resolves
            against this process's pinned platform root; a web route already holding its own
            guarded, resolved root passes it explicitly.
        acknowledgement: The breeder's own act of shipping this delivery unvalidated, or ``None``
            for an ordinary validated export or the MCP tool, which never builds one.
    """
    from tcip_annotation.json_io import prediction_documents
    from tcip_mcp.pipelines.resolution import CountDeliveryRefused
    from tcip_mcp.project_paths import resolve_output_path

    output_csv_path = str(resolve_output_path(output_csv_path))
    pred_dir = Path(predictions_dir)
    if not pred_dir.is_dir():
        raise CountDeliveryRefused(f"predictions_dir not found: {predictions_dir}")
    if not Path(raster_path).is_file():
        raise CountDeliveryRefused(f"raster_path not found: {raster_path}")
    if canopy_subject and nn_tolerance_m is not None:
        raise CountDeliveryRefused(
            "canopy_subject and nn_tolerance_m are refused together: the segment regime attributes "
            "by containment, never by nearest-neighbour distance, so it takes no match tolerance")

    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        load_registry,
        registry_csv_entries,
        verify_registry_csv_bytes,
    )
    from tcip_mcp.project_paths import platform_state_root

    resolved_project_root = Path(project_root) if project_root is not None else platform_state_root()
    registry_record = load_registry(resolved_project_root, plant_registry)
    if registry_record is None:
        raise CountDeliveryRefused(
            f"plant registry not found: {plant_registry!r}; register it with "
            "register_plant_registry before this door reads it")
    registry_entries = registry_csv_entries(registry_record)
    missing, rewritten_fact, verified_csv_bytes = verify_registry_csv_bytes(registry_entries)
    if missing:
        raise CountDeliveryRefused(f"plant CSV(s) not found: {missing}")
    if rewritten_fact:
        raise CountDeliveryRefused(
            f"{rewritten_fact}: restore the file's registered bytes, or register the current "
            "file under a new registry name and deliver under that name")
    plant_csv_paths = [e["path"] for e in registry_entries]

    from tcip_mcp.operationalization import (
        PER_PLANT_COUNT_AGGREGATE,
        OperationalizationRefused,
        check_operationalization,
        resolve_trait_and_record,
        resolve_trait_for_phenotype,
    )
    from tcip_mcp.traits import TraitUnknownError

    # First refusal in this body: how a number was attributed says nothing until it has a meaning.
    # project_root (the caller's own, unresolved) is what every meaning-record read resolves against.
    try:
        trait = resolve_trait_for_phenotype(delivered_phenotype, project_root=project_root)
        spec, record, _specs_dir = resolve_trait_and_record(
            trait, PER_PLANT_COUNT_AGGREGATE, project_root=project_root)
    except (TraitUnknownError, ValueError) as exc:
        raise CountDeliveryRefused(str(exc)) from exc
    # This door never delivers a crossing kind, so it has no registry to check a positive class against.
    stated = check_operationalization(
        spec, record, PER_PLANT_COUNT_AGGREGATE, delivered_phenotype=delivered_phenotype,
        value_keys=[_PER_PLANT_VALUE_KEY], registry=None)
    if not stated.ok:
        raise OperationalizationRefused(stated)

    pred_files = prediction_documents(pred_dir)
    if not pred_files:
        raise CountDeliveryRefused(f"no prediction file(s) found in {predictions_dir}")

    from tcip_mcp.pipelines.raster_source import georeferenced_raster_identity_mismatch
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    sidecar = read_operating_point_sidecar(predictions_dir) or {}
    recorded_identity = sidecar.get("raster_content_identity")
    if recorded_identity is None:
        raise CountDeliveryRefused(
            f"delivery refused: the bucket at {predictions_dir} records no raster content "
            "identity, so there is nothing to check raster_path against. Every count this "
            "delivery writes is attributed to a plant through the supplied raster's own "
            "georeferencing, so the bucket must carry the identity of the raster it was produced "
            "on: produce it with run_inference's raster_path regime, which records that "
            "identity into the bucket's operating_point.json.")
    try:
        identity_mismatch = georeferenced_raster_identity_mismatch(recorded_identity, raster_path)
    except ValueError as exc:
        raise CountDeliveryRefused(f"delivery refused: {exc}") from exc
    if identity_mismatch is not None:
        raise CountDeliveryRefused(
            f"delivery refused: {raster_path} is not the raster the bucket at {predictions_dir} "
            f"was produced on. {identity_mismatch}")

    document_bytes: bytes = b""
    document_path: Path | None = None
    if canopy_subject:
        import hashlib

        from tcip_mcp.dataset_layout import annotation_path_for_image, dataset_root_of, \
            require_dataset_identity
        from tcip_mcp.pipelines.postprocessing.segment_attribution import (
            CanopySegmentRefusal, load_canopy_segments,
        )

        raster_dataset_root = dataset_root_of(Path(raster_path))
        if raster_dataset_root is None:
            raise CountDeliveryRefused(
                f"canopy_subject delivery refused: {raster_path} does not lie under a "
                "registered dataset (no images/, predictions/, annotations/, or labels/ segment "
                "in its path); copy the verified raster to its ingested position under the "
                "bucket's dataset (the identity is content-based, so the tiled pass is not "
                "re-run)")
        try:
            raster_dataset_identity = require_dataset_identity(raster_dataset_root)
        except ValueError as exc:
            raise CountDeliveryRefused(f"canopy_subject delivery refused: {exc}") from exc

        bucket_dataset_root = dataset_root_of(pred_dir)
        if bucket_dataset_root is None:
            raise CountDeliveryRefused(
                f"canopy_subject delivery refused: {predictions_dir} does not lie under a "
                "registered dataset's predictions/ tree; the canopy document is resolved "
                "relative to the bucket's own registered dataset")
        try:
            bucket_dataset_identity = require_dataset_identity(bucket_dataset_root)
        except ValueError as exc:
            raise CountDeliveryRefused(f"canopy_subject delivery refused: {exc}") from exc

        if raster_dataset_identity["id"] != bucket_dataset_identity["id"]:
            raise CountDeliveryRefused(
                f"canopy_subject delivery refused: {raster_path} lies under dataset "
                f"{raster_dataset_root} (id {raster_dataset_identity['id']!r}), a different "
                f"dataset than the bucket at {predictions_dir} (under {bucket_dataset_root}, "
                f"id {bucket_dataset_identity['id']!r}); copy the verified raster to its "
                "ingested position under the bucket's dataset")

        try:
            document_path = annotation_path_for_image(raster_path)
        except ValueError as exc:
            raise CountDeliveryRefused(f"canopy_subject delivery refused: {exc}") from exc
        if not document_path.is_file():
            raise CountDeliveryRefused(
                f"canopy_subject delivery refused: no label document at {document_path} for "
                f"raster {raster_path}; author the canopy boundaries for subject "
                f"{canopy_subject!r} there first")
        document_bytes = document_path.read_bytes()
        try:
            segments = load_canopy_segments(
                document_bytes, subject=canopy_subject, raster_stem=Path(raster_path).stem,
                raster_identity=recorded_identity,
            )
        except CanopySegmentRefusal as exc:
            raise CountDeliveryRefused(f"canopy_subject delivery refused: {exc}") from exc

    from tcip_annotation.json_io import UnreadableLabelDocument, detection_annotations
    from tcip_annotation.state import bbox_of

    boxes: list[list[float]] = []
    try:
        for f in pred_files:
            for a in detection_annotations(str(f)):
                b = bbox_of(a.geometry)
                boxes.append([b.x1, b.y1, b.x2, b.y2])
    except UnreadableLabelDocument as exc:
        raise CountDeliveryRefused(str(exc)) from exc
    detections = {"boxes": boxes}

    from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import (
        DetectionAssignment,
        GeoreferencingError,
        OrthomosaicGeoreference,
        RotatedRasterError,
        assign_detections_to_plants,
        plants_in_frame,
        resolve_nn_tolerance_m,
    )
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        read_plant_csv_bytes, require_named_plants,
    )

    try:
        georef = OrthomosaicGeoreference.from_file(raster_path)
    except (GeoreferencingError, RotatedRasterError) as exc:
        raise CountDeliveryRefused(str(exc)) from exc

    # Parsed from the bytes verify_registry_csv_bytes already read and hashed, never a second open.
    plants = [
        plant
        for entry in registry_entries
        for plant in read_plant_csv_bytes(verified_csv_bytes[entry["path"]])
    ]
    if not plants:
        raise CountDeliveryRefused(f"no georeferenced plants parsed from {plant_csv_paths}")

    width, height = int(recorded_identity["width"]), int(recorded_identity["height"])

    extra_response_fields: dict = {}
    if canopy_subject:
        from tcip_mcp.pipelines.postprocessing.segment_attribution import (
            SEGMENT_ASSIGNMENT_SOURCES, SegmentAssignment, assign_detections_to_segments,
            tie_segments_to_plants,
        )

        try:
            tie = tie_segments_to_plants(segments, plants, georef, width=width, height=height)
        except ValueError as exc:
            raise CountDeliveryRefused(f"canopy_subject delivery refused: {exc}") from exc

        segment_assignments = assign_detections_to_segments(detections, tie)
        ambiguous_segment_indices = {
            idx for a in segment_assignments if a.source == SEGMENT_ASSIGNMENT_SOURCES.overlapping
            for idx in a.overlapping_segment_indices
        }
        tied_by_index = {t.segment_index: t for t in tie.tied}
        ambiguous_tied_indices = ambiguous_segment_indices & set(tied_by_index)
        plants_with_ambiguous_detections = sorted(
            tied_by_index[idx].plot_name for idx in ambiguous_tied_indices)

        counts_by_segment: dict[int, int] = {}
        for a in segment_assignments:
            if a.source == SEGMENT_ASSIGNMENT_SOURCES.containment:
                # a containment assignment always carries the segment it was contained in
                assert a.segment_index is not None
                counts_by_segment[a.segment_index] = counts_by_segment.get(a.segment_index, 0) + 1

        records = [
            {"plant_id": t.plot_name, "plant_id_source": SEGMENT_ASSIGNMENT_SOURCES.containment,
             _PER_PLANT_VALUE_KEY: counts_by_segment.get(t.segment_index, 0),
             "measurement_document": "operating_point",
             "plant_attribution": SegmentAssignment.plant_attribution}
            for t in tie.tied if t.segment_index not in ambiguous_tied_indices
        ]
        if not records:
            raise CountDeliveryRefused(
                "canopy_subject delivery refused: no plant would receive a row; every tied "
                f"segment's own detection is ambiguous ({plants_with_ambiguous_detections}), and "
                f"{len(tie.plants_without_segment)} in-frame plant(s) lie in no segment "
                f"({tie.plants_without_segment}) while {len(tie.untied)} segment(s) contain no "
                "plant")

        n_mapped = sum(1 for a in segment_assignments
                      if a.source == SEGMENT_ASSIGNMENT_SOURCES.containment
                      and a.segment_index not in ambiguous_tied_indices)
        n_detections = len(segment_assignments)
        n_unmapped = n_detections - n_mapped

        by_source = {
            "outside_segments": sum(
                1 for a in segment_assignments if a.source == SEGMENT_ASSIGNMENT_SOURCES.outside),
            "overlapping_segments": sum(
                1 for a in segment_assignments
                if a.source == SEGMENT_ASSIGNMENT_SOURCES.overlapping),
            "segment_without_plant": sum(
                1 for a in segment_assignments
                if a.source == SEGMENT_ASSIGNMENT_SOURCES.without_plant),
        }

        assert document_path is not None  # canopy_subject implies the document was resolved above
        canopy_segments_doc = {
            "path": str(document_path), "sha256": hashlib.sha256(document_bytes).hexdigest(),
            "subject": canopy_subject, "n_segments": len(segments),
        }
        segment_ties_disclosure = [
            {"segment_index": t.segment_index, "plot_name": t.plot_name,
             "clearance_m": t.clearance_m}
            for t in tie.tied
        ]
        plant_mapping_disclosure = {
            "plant_registry": {"name": registry_record["name"], "digest": registry_record["digest"]},
            "project_root": str(resolved_project_root),
            "raster_identity": recorded_identity,
            "canopy_segments": canopy_segments_doc,
            "segment_ties": segment_ties_disclosure,
            "segments_without_plant": len(tie.untied),
            "plants_outside_raster": tie.plants_outside_raster,
            "plants_without_segment": tie.plants_without_segment,
            "plants_with_ambiguous_detections": plants_with_ambiguous_detections,
            "detections_unattributed": sum(by_source.values()),
            "detections_unattributed_by_source": by_source,
            "detections_unattributed_scope": "delivered_raster",
            "plant_attribution": SegmentAssignment.plant_attribution,
        }
        extra_response_fields = {
            "n_segments": len(segments),
            "segment_ties": segment_ties_disclosure,
            "n_segments_without_plant": len(tie.untied),
            "plants_without_segment": tie.plants_without_segment,
            "plants_with_ambiguous_detections": plants_with_ambiguous_detections,
            "plants_outside_raster": tie.plants_outside_raster,
            "n_delivered_of_registered": {"delivered": len(records), "registered": len(plants)},
            "n_unmapped_by_source": by_source,
        }
    else:
        try:
            require_named_plants(plants)
        except ValueError as exc:
            raise CountDeliveryRefused(f"delivery refused: {exc}") from exc
        in_frame_plants, outside_plants = plants_in_frame(plants, georef, width=width, height=height)
        plants_outside_raster_names = sorted(p.plot_name for p in outside_plants)
        if not in_frame_plants:
            raise CountDeliveryRefused(
                f"no registered plant lies inside this raster's frame ({raster_path}); every "
                f"plant in {plant_csv_paths} projects outside it")
        resolved_tolerance = resolve_nn_tolerance_m(in_frame_plants, nn_tolerance_m)
        detection_assignments = assign_detections_to_plants(
            detections, georef, in_frame_plants, nn_tolerance_m=resolved_tolerance["value"])
        mapped = [a for a in detection_assignments if a.plot_name is not None]
        n_mapped = len(mapped)
        n_detections = len(detection_assignments)
        n_unmapped = n_detections - n_mapped

        mapped_plant_ids = {a.plot_name for a in mapped}
        records = [
            {"plant_id": a.plot_name, "plant_id_source": a.source,
             "plant_id_distance_m": a.distance_m, _PER_PLANT_VALUE_KEY: 1,
             "measurement_document": "operating_point", "plant_attribution": a.plant_attribution}
            for a in mapped
        ]
        # Every in-frame plant the scan covers gets a row, an explicit 0 for one matching no
        # detection, never silently absent (see the docstring above).
        records += [{"plant_id": p.plot_name, _PER_PLANT_VALUE_KEY: 0,
                    "measurement_document": "operating_point",
                    "plant_attribution": DetectionAssignment.plant_attribution}
                    for p in in_frame_plants if p.plot_name not in mapped_plant_ids]

        # No walked MappingBuild exists for a whole-raster frame, so this names the registry,
        # the raster identity and the tolerance instead.
        plant_mapping_disclosure = {
            "plant_registry": {"name": registry_record["name"], "digest": registry_record["digest"]},
            "project_root": str(resolved_project_root),
            "raster_identity": recorded_identity,
            "nn_tolerance_m": resolved_tolerance,
            "detections_unattributed": n_unmapped,
            "detections_unattributed_scope": "delivered_raster",
            "plant_attribution": DetectionAssignment.plant_attribution,
            "plants_outside_raster": plants_outside_raster_names,
        }
        extra_response_fields = {"plants_outside_raster": plants_outside_raster_names}

    from tcip_mcp.pipelines.postprocessing.aggregation import aggregate_per_plant, export_aggregated_csv

    agg = aggregate_per_plant(records, strategy="sum", plant_id_key="plant_id",
                              value_key=_PER_PLANT_VALUE_KEY)

    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, DeliveryRefused

    # The raw asserted identity; export_aggregated_csv's own delivered_tail corroborates it.
    provenance = {"producer_model_sha256": sidecar.get("checkpoint_sha256"),
                 "producing_experiment_id": sidecar.get("experiment_id")}
    counts_facts = {
        "n_detections": n_detections, "n_mapped": n_mapped, "n_unmapped": n_unmapped,
    }

    try:
        csv_path, tail, event_recorded = export_aggregated_csv(
            agg, output_csv_path, delivered_phenotype=delivered_phenotype, crop=crop,
            pipeline_version=pipeline_version, provenance=provenance,
            pred_dirs=[predictions_dir],
            door="deliver_orthomosaic_plant_counts",
            plant_mapping=plant_mapping_disclosure,
            acknowledgement=acknowledgement, project_root=project_root,
        )
    except DeliveryRefused as exc:
        # operating_point_validated is the operating_point dimension's own cleared reference;
        # unvalidated_dimensions names every refusing dimension (claim_scope and scale included).
        exc.facts = {
            "operating_point_validated": exc.gate.stamp.get("operating_point", VALIDATED_FALSE),
            "unvalidated_dimensions": exc.gate.unvalidated_cell(),
            **counts_facts,
        }
        if "tile_size" in exc.gate.stamp:
            exc.facts["tile_size_validated"] = exc.gate.stamp["tile_size"]
        raise
    except OperationalizationRefused:
        # The writer's own raise already carries the failed check and no counts; nothing to add.
        raise
    except ValueError as exc:
        # aggregation.py's own refusal already names what triggered it (a plant, a phenotype, a
        # unit); this door's own counts-scoped recon would only restate it under a different name.
        raise CountDeliveryRefused(str(exc), **counts_facts) from exc

    # export_aggregated_csv already recorded the delivery event under this door's own name; this
    # door does not record a second event for the one CSV it just wrote.
    return {
        "csv_path": csv_path,
        "n_plants": len(agg),
        "n_plants_zero_count": sum(1 for r in agg if r.get("value") == 0),
        **counts_facts,
        "operating_point_validated": tail["operating_point_validated"],
        "unvalidated_dimensions": tail["unvalidated_dimensions"],
        "acknowledged_by": tail["acknowledged_by"],
        "checkpoint_sha256": tail["producer_model_sha256"],
        "producing_experiment_id": tail["producing_experiment_id"],
        "validation_record": tail["validation_record"],
        "delivery_event_recorded": event_recorded,
        **extra_response_fields,
    }


@mcp.tool()
@audited
def deliver_orthomosaic_plant_counts(
    predictions_dir: str,
    raster_path: str,
    plant_registry: str,
    output_csv_path: str,
    delivered_phenotype: str,
    crop: str = "",
    pipeline_version: str = "",
    nn_tolerance_m: float | None = None,
    canopy_subject: str = "",
) -> dict:
    """Per-plant detection counts from a persisted orthomosaic prediction bucket plus plant CSV(s).

    The MCP door over :func:`orthomosaic_plant_counts`, which carries the full contract (the
    nearest-neighbour and canopy-segment regimes, the raster-identity check, the delivery gate,
    the meaning door, and every response field); read that function's docstring for the complete
    picture. This door passes no ``project_root`` (the process-pinned platform root) and builds no
    ``acknowledgement``, so an unvalidated delivery always refuses here; the web results route's
    count export is the one surface that can acknowledge and ship this kind unvalidated.

    Returns the core's own response dict unchanged on success. On refusal, returns
    ``{"error": ..., **facts}``: the meaning door's own message with no facts
    (``OperationalizationRefused``), the delivery gate's refusal with
    ``operating_point_validated``/``unvalidated_dimensions``/``n_detections``/``n_mapped``/
    ``n_unmapped`` (``DeliveryRefused``, its own message naming the calibration remedy and the
    Results tab's count export), or every other refusal this door raises on, with whatever facts
    it had in hand (``CountDeliveryRefused``).
    """
    from tcip_mcp.operationalization import OperationalizationRefused
    from tcip_mcp.pipelines.resolution import CountDeliveryRefused, DeliveryRefused

    try:
        return orthomosaic_plant_counts(
            predictions_dir, raster_path, plant_registry, output_csv_path, delivered_phenotype,
            crop=crop, pipeline_version=pipeline_version, nn_tolerance_m=nn_tolerance_m,
            canopy_subject=canopy_subject, project_root=None, acknowledgement=None,
        )
    except OperationalizationRefused as exc:
        return {"error": exc.check.message}
    except DeliveryRefused as exc:
        reason = (
            f"{exc}. Calibrate the operating point (or the tile geometry) that produced this "
            "bucket and re-run, or acknowledge and re-export through the Results tab's count "
            "export; a mosaic whose training run reserved no calibration region has no route to "
            "a delivered CSV through this tool alone."
        )
        return {"error": reason, **exc.facts}
    except CountDeliveryRefused as exc:
        return {"error": str(exc), **exc.facts}
