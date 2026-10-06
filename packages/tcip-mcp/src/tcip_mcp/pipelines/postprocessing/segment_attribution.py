"""Per-plant attribution by canopy segment: a detection attributed to a plant by containment in a
canopy boundary a person accepted, the segment itself tied to a registry plant by containment of
the plant's own projected position. A canopy boundary is one a person accepted
(:func:`load_canopy_segments`). Provisional: no validated position-error bound exists, so a
registry position displaced past its disclosed clearance ties the plant to a neighbor's canopy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, cast

from shapely.geometry import Point as ShapelyPoint
from shapely.ops import nearest_points

from tcip_annotation.json_io import (
    LabelDocument,
    is_unadjudicated_prediction,
    provenance_facts,
)
from tcip_annotation.matching import box_ring, point_in_polygon, shapely_geometry
from tcip_annotation.state import (
    BBox, Polygon, box_derivable, polygonal,
)

from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import (
    GeoTransform,
    OrthomosaicGeoreference,
    detection_location,
    plants_in_frame,
)
from tcip_mcp.pipelines.postprocessing.plant_mapping import (
    PlantRecord, SegmentSource, require_named_plants,
)


class CanopySegmentRefusal(ValueError):
    """A canopy-segment document, or one of its own annotations, cannot stand behind a segment
    tie: a document/raster identity mismatch, an absent subject, a ``Point`` naming no region, or
    a record not positively a person's."""


@dataclass
class CanopySegment:
    """One canopy boundary a person accepted, in the raster's own full-mosaic pixel space.

    ``segment_index`` is the boundary's position among the document's own annotations of the stated
    subject (stable within one load); ``polygon`` is the boundary itself, a box admitted as its
    rectangle through :func:`tcip_annotation.matching.box_ring`.
    """

    segment_index: int
    polygon: Polygon


def _polygon_of(geometry: BBox | Polygon) -> Polygon:
    """``geometry`` as a :class:`Polygon`: itself, or a box's own rectangle built through
    :func:`tcip_annotation.matching.box_ring`.
    """
    if polygonal(geometry):
        return geometry
    return Polygon(rings=[box_ring(cast(BBox, geometry))])


def load_canopy_segments(
    document: LabelDocument, *, subject: str, raster_stem: str, raster_identity: dict,
) -> list[CanopySegment]:
    """The canopy segments of ``subject`` in the raster's own label ``document`` (the caller read
    it by the raster's key), checked against the raster's frame and against the canopy rule's own
    provenance admissibility.

    Keeps the annotations whose ``subject`` is the stated one. Refuses by name: when the
    document's ``width``/``height`` differ from ``raster_identity``'s; when no annotation of
    ``subject`` exists; when an annotation of ``subject`` carries no geometry at all (an
    image-level label) or is a :class:`~tcip_annotation.state.Point`, naming the record; and when
    any annotation of ``subject`` is not positively a person's: a scored record (the model's own
    unreviewed output), a record with no ``created_by`` at all, or a record whose ``created_by``
    is not a person's unless its ``accepted_by`` is a person's, each refused naming the record. A
    person's own hand trace, and a machine-authored proposal a reviewer has accepted, both admit.
    """
    if (int(document.width or -1), int(document.height or -1)) != (
            int(raster_identity["width"]), int(raster_identity["height"])):
        raise CanopySegmentRefusal(
            f"the canopy segment document of raster stem {raster_stem!r} is "
            f"{document.width}x{document.height}, the raster is "
            f"{raster_identity['width']}x{raster_identity['height']}; the document does not "
            "describe this raster"
        )

    annotations = [a for a in document.annotations if a.subject == subject]
    if not annotations:
        raise CanopySegmentRefusal(
            f"no annotation of subject {subject!r} exists in the canopy segment document for "
            f"raster stem {raster_stem!r}; canopy_subject names a claim the data must positively "
            "carry"
        )

    for i, a in enumerate(annotations):
        if is_unadjudicated_prediction(a):
            raise CanopySegmentRefusal(
                f"canopy segment {i} of subject {subject!r} carries a prediction score, the "
                "model's own unreviewed output, and cannot stand behind a boundary a person has "
                "not accepted"
            )
    for i, a in enumerate(annotations):
        if not box_derivable(a.geometry):
            named = "an image-level label" if a.geometry is None else type(a.geometry).__name__
            raise CanopySegmentRefusal(
                f"canopy segment {i} of subject {subject!r} carries {named}, which names no "
                "region; delete this record or replace it with a boundary (a box or a traced "
                "polygon)"
            )

    facts = provenance_facts(annotations)
    if facts.no_created_by:
        i = facts.no_created_by[0]
        raise CanopySegmentRefusal(
            f"canopy segment {i} of subject {subject!r} carries no created_by at all; a canopy "
            "boundary must positively carry a person's authorship or acceptance"
        )
    if facts.not_positively_a_persons:
        i = facts.not_positively_a_persons[0]
        raise CanopySegmentRefusal(
            f"canopy segment {i} of subject {subject!r} is authored by "
            f"{annotations[i].created_by!r}, which names no person under this platform's "
            "user:<name> convention, and its own accepted_by is not a person's either; a canopy "
            "boundary must be positively a person's, a reviewer's acceptance included"
        )

    return [CanopySegment(segment_index=i, polygon=_polygon_of(cast("BBox | Polygon", a.geometry)))
            for i, a in enumerate(annotations)]


@dataclass
class TiedSegment:
    """A canopy segment tied to exactly one registry plant.

    ``clearance_m`` is the distance from the plant's own projected position to this segment's
    boundary, in the raster's native CRS units: the margin a displaced registry position would have
    to exceed to leave this segment.
    """

    segment_index: int
    polygon: Polygon
    plot_name: str
    accession_name: str
    clearance_m: float


@dataclass
class UntiedSegment:
    """A canopy segment containing no registry plant."""

    segment_index: int
    polygon: Polygon


@dataclass
class SegmentTie:
    """The result of tying every canopy segment in one raster to the registry plants it can be
    tied to: the segments actually tied, the segments containing no plant, and the two name lists
    that account for every in-frame or out-of-frame plant a tie does not cover."""

    tied: list[TiedSegment]
    untied: list[UntiedSegment]
    plants_without_segment: list[str]
    """Plot names of every in-frame plant that lies inside no canopy segment, by name."""
    plants_outside_raster: list[str]
    """Plot names of every registry plant whose own projected position lies outside the raster's
    frame (:func:`~tcip_mcp.pipelines.postprocessing.orthomosaic_mapping.plants_in_frame`), never
    tested for containment at all."""


def _clearance_m(px: float, py: float, polygon: Polygon, transform: GeoTransform) -> float:
    """The distance from pixel ``(px, py)`` to ``polygon``'s boundary in the raster's native CRS
    units, each axis's pixel delta scaled by that axis's own pixel scale."""
    boundary_point = nearest_points(ShapelyPoint(px, py),
                                    shapely_geometry(polygon.rings).boundary)[1]
    delta_native_x = (px - boundary_point.x) * transform.pixel_scale_x
    delta_native_y = (py - boundary_point.y) * transform.pixel_scale_y
    return math.hypot(delta_native_x, delta_native_y)


def tie_segments_to_plants(
    segments: list[CanopySegment], plants: list[PlantRecord], georef: OrthomosaicGeoreference,
    *, width: int, height: int,
) -> SegmentTie:
    """Tie every one of ``segments`` to the one registry plant, if any, whose own projected
    position it contains.

    Refuses by name: a registry with a blank or duplicate ``plot_name``
    (:func:`~tcip_mcp.pipelines.postprocessing.plant_mapping.require_named_plants`); a plant inside
    more than one segment; a segment containing more than one plant; no in-frame plant in the
    registry at all. Plants are partitioned first through
    :func:`~tcip_mcp.pipelines.postprocessing.orthomosaic_mapping.plants_in_frame`, so a plant
    outside the raster is never tested for containment and is disclosed by name on the returned
    :class:`SegmentTie`.
    """
    require_named_plants(plants)

    in_frame, outside = plants_in_frame(plants, georef, width=width, height=height)
    if not in_frame:
        raise ValueError(
            "no registry plant lies inside this raster's frame; canopy segments cannot be tied to "
            "any plant"
        )

    plant_pixel = {p.plot_name: georef.wgs84_to_pixel(p.lat, p.lon) for p in in_frame}
    plant_segments: dict[str, list[int]] = {p.plot_name: [] for p in in_frame}
    segment_plants: dict[int, list[PlantRecord]] = {s.segment_index: [] for s in segments}
    for s in segments:
        for p in in_frame:
            px, py = plant_pixel[p.plot_name]
            if point_in_polygon(px, py, s.polygon):
                segment_plants[s.segment_index].append(p)
                plant_segments[p.plot_name].append(s.segment_index)

    for p in in_frame:
        containing = plant_segments[p.plot_name]
        if len(containing) > 1:
            raise ValueError(
                f"plant {p.plot_name!r} lies inside more than one canopy segment "
                f"{sorted(containing)}; a segment tie requires exactly one containing segment per "
                "plant"
            )
    for s in segments:
        contained = segment_plants[s.segment_index]
        if len(contained) > 1:
            raise ValueError(
                f"canopy segment {s.segment_index} contains more than one plant "
                f"({sorted(p.plot_name for p in contained)}); a segment tied to more than one "
                "plant cannot attribute detections to a single identity"
            )

    tied: list[TiedSegment] = []
    untied: list[UntiedSegment] = []
    for s in segments:
        contained = segment_plants[s.segment_index]
        if len(contained) == 1:
            p = contained[0]
            px, py = plant_pixel[p.plot_name]
            clearance_m = _clearance_m(px, py, s.polygon, georef.transform)
            tied.append(TiedSegment(
                segment_index=s.segment_index, polygon=s.polygon, plot_name=p.plot_name,
                accession_name=p.accession_name, clearance_m=clearance_m,
            ))
        else:
            untied.append(UntiedSegment(segment_index=s.segment_index, polygon=s.polygon))

    plants_without_segment = sorted(
        p.plot_name for p in in_frame if not plant_segments[p.plot_name]
    )
    plants_outside_raster = sorted(p.plot_name for p in outside)

    return SegmentTie(
        tied=tied, untied=untied, plants_without_segment=plants_without_segment,
        plants_outside_raster=plants_outside_raster,
    )


@dataclass
class SegmentAssignment:
    """The canopy segment a single detection resolves to, by containment of its box centroid.

    ``detection_index``/``pixel_x``/``pixel_y`` mirror
    :class:`~tcip_mcp.pipelines.postprocessing.orthomosaic_mapping.DetectionAssignment`'s own
    fields. ``segment_index`` is ``None`` when the centroid lies inside no segment
    (``source="outside_segments"``) or inside more than one (``source="overlapping_segments"``,
    attributed to neither); ``overlapping_segment_indices`` carries every segment index the
    overlap touched in that case, empty otherwise. ``distance_m`` is always ``None``: containment
    measures no positional distance.
    """

    detection_index: int
    pixel_x: float
    pixel_y: float
    segment_index: int | None
    plot_name: str | None
    accession_name: str | None
    source: SegmentSource
    distance_m: None
    overlapping_segment_indices: tuple[int, ...] = ()

    plant_attribution: ClassVar[str] = "segment"
    """The granularity this mapper attributes objects to plants at: containment of a detection in
    a canopy boundary a person accepted, never a mask-level or area measurement."""


def assign_detections_to_segments(boxes: Sequence[Sequence[float]],
                                  tie: SegmentTie) -> list[SegmentAssignment]:
    """One :class:`SegmentAssignment` per xyxy box of ``boxes`` (full-mosaic pixel space), its
    centroid tested against every candidate segment ``tie`` holds, tied and untied both."""
    candidates: list[tuple[int, Polygon, TiedSegment | None]] = [
        (s.segment_index, s.polygon, s) for s in tie.tied
    ] + [
        (s.segment_index, s.polygon, None) for s in tie.untied
    ]

    out: list[SegmentAssignment] = []
    for i, box in enumerate(boxes):
        cx, cy = detection_location(box)
        hits = [
            (idx, tied) for idx, polygon, tied in candidates if point_in_polygon(cx, cy, polygon)
        ]
        tied = hits[0][1] if len(hits) == 1 else None
        source: SegmentSource = ("outside_segments" if not hits
                                 else "overlapping_segments" if len(hits) > 1
                                 else "segment_containment" if tied is not None
                                 else "segment_without_plant")
        out.append(SegmentAssignment(
            detection_index=i, pixel_x=cx, pixel_y=cy,
            segment_index=hits[0][0] if len(hits) == 1 else None,
            plot_name=tied.plot_name if tied else None,
            accession_name=tied.accession_name if tied else None, source=source, distance_m=None,
            overlapping_segment_indices=tuple(idx for idx, _ in hits) if len(hits) > 1 else ()))
    return out
