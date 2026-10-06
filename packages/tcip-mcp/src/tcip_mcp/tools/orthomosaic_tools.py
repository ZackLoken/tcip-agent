"""Orthomosaic MCP tools: per-plant delivery from a published whole-raster prediction bucket plus a
registered plant registry.

``deliver_orthomosaic_plant_counts`` reads that bucket and registry into a per-plant count CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from tcip_mcp.server import tool

if TYPE_CHECKING:
    from tcip_annotation.state import BBox, Polygon

logger = logging.getLogger(__name__)

_PER_PLANT_VALUE_KEY = "count"
"""The quantity every row of this delivery holds, named once for the aggregation and its rows."""


def orthomosaic_plant_counts(
    project: Path,
    dataset_root: str,
    bucket: str,
    registry_record: dict,
    output_csv_path: str,
    delivered_phenotype: str,
    plants: list[str],
    crop: str = "",
    pipeline_version: str = "",
    nn_tolerance_m: float | None = None,
    canopy_subject: str = "",
    acknowledgment_id: str | None = None,
    *,
    door: str,
    actor: str | None,
) -> dict:
    """Per-plant detection counts from the whole-raster prediction bucket ``bucket`` published under
    ``dataset_root``, for exactly the
    plants ``plants`` names, delivered as a ``per_plant_count_aggregate`` CSV under ``project`` by
    ``actor`` through the door ``door`` names, over the registered plant registry
    ``registry_record``
    (:func:`~tcip_mcp.pipelines.postprocessing.plant_mapping.load_registry`).

    The raster is the one the bucket's record names; it must still be the raster its recorded
    content identity and georeferencing describe
    (:func:`~tcip_mcp.pipelines.raster_source.georeferenced_raster_identity_mismatch`). Each
    detection's box centroid resolves to a real-world coordinate through the raster's own
    georeferencing and, absent ``canopy_subject``, is matched to the nearest registered plant inside
    the raster's frame within the tolerance (derived from the plant grid unless stated); with
    ``canopy_subject`` it is attributed by containment in a canopy boundary accepted into the
    raster's own label document, each boundary tied to the one registry plant its position lies in
    (:mod:`~tcip_mcp.pipelines.postprocessing.segment_attribution`). A population plant matched by
    no detection gets an explicit ``0``. A population plant this attribution cannot count (outside
    the frame; in the segment regime, inside no segment or in one whose detection is ambiguous)
    refuses, naming it and why. The registry's files must still hash to the bytes it registered.

    Delivered through
    :func:`~tcip_mcp.pipelines.postprocessing.aggregation.deliver_per_plant_aggregate`
    under the bucket's gate, the recorded acknowledgment ``acknowledgment_id`` shipping it
    unvalidated. The delivery event carries a
    ``PlantRegistryDisclosure`` or, under ``canopy_subject``, a ``CanopySegmentDisclosure``.
    Refuses (``ValueError``) everything named above and whatever the delivery refuses.
    """
    from tcip_annotation.json_io import detection_annotations, read_label_document
    from tcip_annotation.state import bbox_of

    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import (
        DetectionAssignment, OrthomosaicGeoreference, assign_detections_to_plants, plants_in_frame,
    )
    from tcip_mcp.pipelines.postprocessing.phenology import population
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        assignment_is_attributed, read_plant_csv_bytes, registry_csv_entries,
        require_named_plants, resolve_nn_tolerance_m, verify_registry_csv_bytes,
    )
    from tcip_mcp.pipelines.raster_source import georeferenced_raster_identity_mismatch

    if canopy_subject and nn_tolerance_m is not None:
        raise ValueError("canopy_subject and nn_tolerance_m are refused together: the segment "
                         "regime attributes by containment and takes no match tolerance.")
    wanted = population(plants)
    found = read_bucket(dataset_root, bucket)
    raster_path, recorded_identity = found.raster(project), found.raster_identity
    if raster_path is None or recorded_identity is None:
        raise ValueError(f"bucket {bucket!r} is a bucket of per-image predictions, not of one "
                         "raster: deliver its counts through deliver_per_image_counts.")
    mismatch = georeferenced_raster_identity_mismatch(recorded_identity, raster_path)
    if mismatch is not None:
        raise ValueError(f"{raster_path} is no longer the raster bucket {bucket!r} was predicted "
                         f"on. {mismatch}")
    registry_entries = registry_csv_entries(registry_record, project)
    missing, rewritten, csv_bytes = verify_registry_csv_bytes(registry_entries)
    if missing or rewritten:
        raise ValueError(f"{rewritten or f'plant CSV(s) not found: {missing}'}: restore the "
                         "registered bytes, or register the current file under a new name.")
    registered = [p for e in registry_entries for p in read_plant_csv_bytes(csv_bytes[e["path"]])]
    boxes = [[b.x1, b.y1, b.x2, b.y2] for key in found.document_keys
             for b in (bbox_of(cast("BBox | Polygon", a.geometry))
                       for a in detection_annotations(read_label_document(key).annotations))]
    georef = OrthomosaicGeoreference.from_file(raster_path)
    width, height = int(recorded_identity["width"]), int(recorded_identity["height"])
    registry_ref = {"name": registry_record["name"], "digest": registry_record["digest"]}

    uncountable: dict[str, str] = {}
    if canopy_subject:
        counts, disclosure, uncountable = _segment_counts(
            raster_path, recorded_identity, canopy_subject, registered, georef,
            width, height, boxes, registry_ref)
        attribution = disclosure["plant_attribution"]
    else:
        require_named_plants(registered)
        in_frame, outside = plants_in_frame(registered, georef, width=width, height=height)
        uncountable = {p.plot_name: "outside the raster's frame" for p in outside}
        tolerance = resolve_nn_tolerance_m(in_frame, nn_tolerance_m)
        assignments = assign_detections_to_plants(boxes, georef, in_frame,
                                                  nn_tolerance_m=tolerance["value"])
        attributed = [a for a in assignments if assignment_is_attributed(a)]
        counts = {p.plot_name: 0 for p in in_frame}
        for a in attributed:
            counts[cast(str, a.plot_name)] += 1
        attribution = DetectionAssignment.plant_attribution
        disclosure = {
            "plant_registry": registry_ref, "raster_identity": recorded_identity,
            "nn_tolerance_m": tolerance,
            "detections_unattributed": len(assignments) - len(attributed),
            "detections_unattributed_scope": "delivered_raster", "plant_attribution": attribution,
            "plants_outside_raster": sorted(p.plot_name for p in outside),
        }
    refused = {p: uncountable.get(p, "not in the registry") for p in wanted if p not in counts}
    if refused:
        raise ValueError(f"these population plants cannot be counted from this raster: {refused}.")
    results = [{"plant_id": p, "value": counts[p], "value_key": _PER_PLANT_VALUE_KEY,
                "observations": 1, "plant_attribution": attribution} for p in wanted]

    from tcip_mcp.pipelines.postprocessing.aggregation import deliver_per_plant_aggregate
    from tcip_mcp.traits import PER_PLANT_COUNT_AGGREGATE

    delivered = deliver_per_plant_aggregate(
        project, results, str(Path(project, output_csv_path)),
        delivered_phenotype=delivered_phenotype, delivery_kind=PER_PLANT_COUNT_AGGREGATE,
        buckets=[found], plants=wanted, crop=crop, pipeline_version=pipeline_version,
        door=door, plant_mapping=disclosure, acknowledgment_id=acknowledgment_id, actor=actor)
    return {**delivered, "n_detections": len(boxes),
            "detections_unattributed": disclosure["detections_unattributed"],
            "plants_outside_raster": disclosure["plants_outside_raster"]}


def _segment_counts(
    raster_path: str, recorded_identity: dict, canopy_subject: str,
    registered: list, georef: Any, width: int, height: int, boxes: list, registry_ref: dict,
) -> tuple[dict[str, int], dict, dict[str, str]]:
    """Per-plant counts under the segment regime: each detection counted to the tied plant whose
    canopy segment contains its centroid. Returns the counts, the ``CanopySegmentDisclosure`` and
    each registry plant that regime leaves uncounted, with why. The canopy document is the
    raster's own label document (:func:`~tcip_mcp.dataset_layout.label_key_of`)."""
    from tcip_annotation.json_io import UnreadableLabelDocumentError, document_at, read_stored

    from tcip_mcp.dataset_layout import label_key_of
    from tcip_mcp.pipelines.postprocessing.plant_mapping import UNATTRIBUTED_SEGMENT_SOURCES
    from tcip_mcp.pipelines.postprocessing.segment_attribution import (
        SegmentAssignment, assign_detections_to_segments, load_canopy_segments,
        tie_segments_to_plants,
    )

    stem = Path(raster_path).stem
    key = label_key_of(raster_path)
    try:
        stored = read_stored(key)
    except UnreadableLabelDocumentError as exc:
        raise ValueError(f"canopy_subject delivery refused: the canopy boundaries for "
                         f"{canopy_subject!r} on the raster {stem!r} do not read (author the "
                         f"canopy boundaries where none exist): {exc}") from exc
    document, version = document_at(key, stored), stored.version
    segments = load_canopy_segments(document, subject=canopy_subject, raster_stem=stem,
                                    raster_identity=recorded_identity)
    tie = tie_segments_to_plants(segments, registered, georef, width=width, height=height)
    assignments = assign_detections_to_segments(boxes, tie)
    ambiguous_segments = {i for a in assignments if a.source == "overlapping_segments"
                          for i in a.overlapping_segment_indices}
    tied = {t.segment_index: t for t in tie.tied}
    ambiguous = sorted(tied[i].plot_name for i in ambiguous_segments & set(tied))
    counts = {t.plot_name: 0 for t in tie.tied if t.plot_name not in ambiguous}
    for a in assignments:
        if a.source == "segment_containment" and a.plot_name in counts:
            counts[a.plot_name] += 1
    by_source = {source: sum(a.source == source for a in assignments)
                 for source in UNATTRIBUTED_SEGMENT_SOURCES}
    disclosure = {
        "plant_registry": registry_ref, "raster_identity": recorded_identity,
        "canopy_segments": {"capture": key.parts[0], "stem": stem, "sha256": version.token,
                            "subject": canopy_subject, "n_segments": len(segments)},
        "segment_ties": [{"segment_index": t.segment_index, "plot_name": t.plot_name,
                          "clearance_m": t.clearance_m} for t in tie.tied],
        "segments_without_plant": len(tie.untied),
        "plants_outside_raster": tie.plants_outside_raster,
        "plants_without_segment": tie.plants_without_segment,
        "plants_with_ambiguous_detections": ambiguous,
        "detections_unattributed": sum(by_source.values()),
        "detections_unattributed_by_source": by_source,
        "detections_unattributed_scope": "delivered_raster",
        "plant_attribution": SegmentAssignment.plant_attribution,
    }
    uncountable = {**{p: "outside the raster's frame" for p in tie.plants_outside_raster},
                   **{p: "inside no canopy segment" for p in tie.plants_without_segment},
                   **{p: "its segment's detection is ambiguous" for p in ambiguous}}
    return counts, disclosure, uncountable


@tool()
def deliver_orthomosaic_plant_counts(
    project: Path,
    dataset_root: str,
    bucket: str,
    plant_registry: str,
    output_csv_path: str,
    delivered_phenotype: str,
    plants: list[str],
    crop: str = "",
    pipeline_version: str = "",
    nn_tolerance_m: float | None = None,
    canopy_subject: str = "",
    acknowledgment_id: str | None = None,
) -> dict:
    """Per-plant detection counts from a published whole-raster prediction bucket plus a registered
    plant registry, for the plants ``plants`` names.

    The MCP door over :func:`orthomosaic_plant_counts`, which carries the full contract. An
    unvalidated bucket ships only under ``acknowledgment_id``, a breeder's recorded acknowledgment
    of exactly this result, which this door executes and never records.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import load_registry

    try:
        return orthomosaic_plant_counts(
            project, dataset_root, bucket, load_registry(project, plant_registry),
            output_csv_path,
            delivered_phenotype, plants, crop=crop, pipeline_version=pipeline_version,
            nn_tolerance_m=nn_tolerance_m, canopy_subject=canopy_subject,
            acknowledgment_id=acknowledgment_id, door="deliver_orthomosaic_plant_counts",
            actor=None)
    except ValueError as exc:
        return {"error": str(exc)}
