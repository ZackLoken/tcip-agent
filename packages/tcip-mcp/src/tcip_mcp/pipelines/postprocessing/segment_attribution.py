"""Per-plant attribution by canopy segment: a detection attributed to a plant by containment in
a canopy boundary a person accepted, the segment itself tied to a registry plant by containment
of the plant's own projected position.

A canopy boundary here is whatever the breeder accepted into a per-image label document: a hand
trace, a SAM proposal a reviewer accepted, or an instance-segmentation model's own output once a
reviewer has accepted it. What produced it is not this module's concern; what makes it usable is
that a person positively stands behind it (:func:`load_canopy_segments`). The tie from a segment
to a plant identity rests on the registry position's own accuracy: a position displaced by more
than its disclosed clearance places the plant in a neighbour's canopy with every check here
passing, since no breeder-confirmed tie or validated position-error bound exists yet (see the
design's own record for that open point).

Composes :mod:`tcip_mcp.pipelines.postprocessing.orthomosaic_mapping` (the pixel <-> real-world
mapping and the shared in-frame partition) and
:mod:`tcip_mcp.pipelines.postprocessing.plant_mapping` (the registry's own ``PlantRecord`` and
attribution rule), so nothing here reimplements either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

from shapely.geometry import Point as ShapelyPoint
from shapely.ops import nearest_points
from shapely.validation import make_valid

from tcip_annotation.json_io import (
    annotations_of_document,
    is_unadjudicated_prediction,
    parse_label_document,
    provenance_facts,
)
from tcip_annotation.matching import _rings_to_shapely, box_ring, point_in_polygon
from tcip_annotation.state import Annotation, BBox, Point as AnnotationPoint, Polygon

from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import (
    GeoTransform,
    OrthomosaicGeoreference,
    detection_location,
    plants_in_frame,
)
from tcip_mcp.pipelines.postprocessing.plant_mapping import PlantRecord, require_named_plants


class CanopySegmentRefusal(ValueError):
    """A canopy-segment document, or one of its own annotations, cannot stand behind a segment
    tie: a document/raster identity mismatch, an absent subject, a ``Point`` naming no region, or
    a record not positively a person's."""


class _SegmentAssignmentSources(NamedTuple):
    """The four values :class:`SegmentAssignment`'s own ``source`` field takes, named once so
    :func:`assign_detections_to_segments` builds each assignment against a name rather than
    repeating the vocabulary as a bare literal per branch."""

    containment: str = "segment_containment"
    outside: str = "outside_segments"
    overlapping: str = "overlapping_segments"
    without_plant: str = "segment_without_plant"


SEGMENT_ASSIGNMENT_SOURCES = _SegmentAssignmentSources()


@dataclass
class CanopySegment:
    """One canopy boundary a person accepted, in the raster's own full-mosaic pixel space.

    ``segment_index`` is the boundary's position among the document's own annotations of the
    stated subject (stable within one load, so a later reader joins a tie or an assignment back
    to the boundary it names); ``polygon`` is the boundary itself, a box admitted as the rectangle
    it is through :func:`tcip_annotation.matching.box_ring`, the one ring construction a box
    turns into everywhere it does.
    """

    segment_index: int
    polygon: Polygon


def _polygon_of(geometry: BBox | Polygon) -> Polygon:
    """``geometry`` as a :class:`Polygon`: itself, or a box's own rectangle built through
    :func:`tcip_annotation.matching.box_ring`, the one ring construction
    ``tcip_annotation.matching._to_shapely`` uses too, so the two conversions cannot
    independently drift on which corner comes first."""
    if isinstance(geometry, Polygon):
        return geometry
    return Polygon(rings=[box_ring(geometry)])


def load_canopy_segments(
    document_bytes: bytes, *, subject: str, raster_stem: str, raster_identity: dict,
) -> list[CanopySegment]:
    """The canopy segments of ``subject`` in one label document's own bytes, checked against the
    raster they claim to describe and against the canopy rule's own provenance admissibility.

    Parses ``document_bytes`` (a byte snapshot the caller already read and hashed once) and keeps
    the annotations whose ``subject`` is the stated one. Refuses by name: when the document's own
    ``image`` is not ``raster_stem`` or its ``width``/``height`` differ from ``raster_identity``'s
    (the document at this position does not describe this raster); when no annotation of
    ``subject`` exists (a stated subject is a claim the data must positively carry); when an
    annotation of ``subject`` carries no geometry at all (an image-level label) or is a
    :class:`~tcip_annotation.state.Point` (either way it names no region, refused naming the
    record so the breeder can delete it, rather than skipped silently or the whole document
    refused by implication); and when any annotation of ``subject`` is not
    positively a person's: a scored record (the model's own unreviewed output), a record with no
    ``created_by`` at all (the reference rule's own pre-provenance exception does not apply here),
    or a record whose ``created_by`` is not a person's unless its ``accepted_by`` is a person's,
    each refused naming the record. A person's own hand trace, and a machine-authored (SAM or a
    bespoke model's) proposal a reviewer has accepted, both admit; the refusal text never asserts
    that a record naming a reviewer was unreviewed.
    """
    try:
        text = document_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CanopySegmentRefusal(
            f"the canopy segment document for raster stem {raster_stem!r} is not valid UTF-8: {exc}"
        ) from exc
    document = parse_label_document(
        text, source=f"canopy segment document for raster stem {raster_stem!r}")

    if document.get("image") != raster_stem:
        raise CanopySegmentRefusal(
            f"the canopy segment document names image {document.get('image')!r}, not the raster's "
            f"own stem {raster_stem!r}; the document at this position does not describe this raster"
        )
    doc_width, doc_height = document.get("width"), document.get("height")
    if int(doc_width or -1) != int(raster_identity["width"]) or \
            int(doc_height or -1) != int(raster_identity["height"]):
        raise CanopySegmentRefusal(
            f"the canopy segment document is {doc_width}x{doc_height}, the raster is "
            f"{raster_identity['width']}x{raster_identity['height']}; the document at this "
            "position does not describe this raster"
        )

    annotations = [a for a in annotations_of_document(document) if a.subject == subject]
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
        if a.geometry is None:
            raise CanopySegmentRefusal(
                f"canopy segment {i} of subject {subject!r} carries no geometry (an image-level "
                "label), which names no region; delete this record or replace it with a boundary "
                "(a box or a traced polygon)"
            )
        if isinstance(a.geometry, AnnotationPoint):
            raise CanopySegmentRefusal(
                f"canopy segment {i} of subject {subject!r} is a Point, which names no region; "
                "delete this record or replace it with a boundary (a box or a traced polygon)"
            )

    facts = provenance_facts(annotations)
    if facts.no_created_by:
        i = facts.no_created_by[0]
        raise CanopySegmentRefusal(
            f"canopy segment {i} of subject {subject!r} carries no created_by at all; a canopy "
            "boundary must positively carry a person's authorship or acceptance, unlike the "
            "reference rule's own pre-provenance hand labels"
        )
    if facts.not_positively_a_persons:
        i = facts.not_positively_a_persons[0]
        raise CanopySegmentRefusal(
            f"canopy segment {i} of subject {subject!r} is authored by "
            f"{annotations[i].created_by!r}, which names no person under this platform's "
            "user:<name> convention, and its own accepted_by is not a person's either; a canopy "
            "boundary must be positively a person's, a reviewer's acceptance included"
        )

    def _checked_polygon(a: Annotation) -> Polygon:
        # Every refusal above already ran over the same annotations; reaching here means none
        # of them had a None or Point geometry.
        assert a.geometry is not None and not isinstance(a.geometry, AnnotationPoint)
        return _polygon_of(a.geometry)

    return [
        CanopySegment(segment_index=i, polygon=_checked_polygon(a))
        for i, a in enumerate(annotations)
    ]


@dataclass
class TiedSegment:
    """A canopy segment tied to exactly one registry plant.

    ``clearance_m`` is the distance from the plant's own projected position to this segment's
    boundary, in the raster's native CRS units: the derived margin a displaced registry position
    would have to exceed to leave this segment (this platform's own honest residual for the tie;
    see this module's own docstring for what it does not establish).
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
    """The distance from pixel ``(px, py)`` to ``polygon``'s own boundary, in the raster's native
    CRS units, through ``transform``'s own pixel scale.

    Computed from each axis's own pixel delta to the nearest boundary point, converted through
    ``pixel_scale_x``/``pixel_scale_y`` independently and combined by Pythagoras, so an
    anisotropic pixel scale (``pixel_scale_x != pixel_scale_y``) still converts exactly, never a
    single scalar multiply that assumes square pixels.
    """
    geom = _rings_to_shapely(polygon.rings)
    if not geom.is_valid:
        geom = make_valid(geom)
    boundary_point = nearest_points(ShapelyPoint(px, py), geom.boundary)[1]
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
    (:func:`~tcip_mcp.pipelines.postprocessing.plant_mapping.require_named_plants`, the same check
    the nearest-neighbour regime runs over its own registry); a plant inside more than one segment;
    a segment containing more than one plant; no in-frame plant in the registry at all. Plants are
    partitioned first through :func:`~tcip_mcp.pipelines.postprocessing.orthomosaic_mapping.
    plants_in_frame`, so a plant outside the raster is never tested for containment and is instead
    disclosed by name on the returned :class:`SegmentTie`.
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

    plants_without_segment = sorted(p.plot_name for p in in_frame if not plant_segments[p.plot_name])
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
    attributed to neither, never the nearer); ``overlapping_segment_indices`` carries every
    segment index the overlap touched in that case, empty otherwise, so a caller can still name
    which tied segments' plants an ambiguous detection implicated. ``distance_m`` is always
    ``None``: containment carries no positional distance, so the CSV's own
    ``plant_id_distance_m_max`` stays blank and means no positional bound was measured, never zero
    uncertainty.
    """

    detection_index: int
    pixel_x: float
    pixel_y: float
    segment_index: int | None
    plot_name: str | None
    accession_name: str | None
    source: str  # one of SEGMENT_ASSIGNMENT_SOURCES
    distance_m: None
    overlapping_segment_indices: tuple[int, ...] = ()

    plant_attribution: ClassVar[str] = "segment"
    """The granularity this mapper attributes objects to plants at: containment of a detection in
    a canopy boundary a person accepted, never a mask-level or area measurement."""


def assign_detections_to_segments(detections: dict, tie: SegmentTie) -> list[SegmentAssignment]:
    """One :class:`SegmentAssignment` per box in a ``predict_tiled``-shaped ``detections`` result
    (the same ``{"boxes": [[x1, y1, x2, y2], ...]}`` shape
    :func:`~tcip_mcp.pipelines.postprocessing.orthomosaic_mapping.assign_detections_to_plants`
    reads), taking only ``tie`` (never the plant registry or the raster directly): every candidate
    segment, tied and untied both, already lives on it.
    """
    boxes = detections.get("boxes") or []
    candidates: list[tuple[int, Polygon, TiedSegment | None]] = [
        (s.segment_index, s.polygon, s) for s in tie.tied
    ] + [
        (s.segment_index, s.polygon, None) for s in tie.untied
    ]

    out: list[SegmentAssignment] = []
    for i, box in enumerate(boxes):
        cx, cy = detection_location(box)
        hits = [(idx, tied) for idx, polygon, tied in candidates if point_in_polygon(cx, cy, polygon)]
        if not hits:
            out.append(SegmentAssignment(
                detection_index=i, pixel_x=cx, pixel_y=cy, segment_index=None, plot_name=None,
                accession_name=None, source=SEGMENT_ASSIGNMENT_SOURCES.outside, distance_m=None,
            ))
        elif len(hits) > 1:
            out.append(SegmentAssignment(
                detection_index=i, pixel_x=cx, pixel_y=cy, segment_index=None, plot_name=None,
                accession_name=None, source=SEGMENT_ASSIGNMENT_SOURCES.overlapping,
                distance_m=None, overlapping_segment_indices=tuple(idx for idx, _ in hits),
            ))
        else:
            idx, tied = hits[0]
            if tied is not None:
                out.append(SegmentAssignment(
                    detection_index=i, pixel_x=cx, pixel_y=cy, segment_index=idx,
                    plot_name=tied.plot_name, accession_name=tied.accession_name,
                    source=SEGMENT_ASSIGNMENT_SOURCES.containment, distance_m=None,
                ))
            else:
                out.append(SegmentAssignment(
                    detection_index=i, pixel_x=cx, pixel_y=cy, segment_index=idx, plot_name=None,
                    accession_name=None, source=SEGMENT_ASSIGNMENT_SOURCES.without_plant,
                    distance_m=None,
                ))
    return out
