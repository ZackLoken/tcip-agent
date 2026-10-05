"""The platform's one matcher of detections to ground truth, and geometry helpers over
:class:`~tcip_annotation.state.Annotation` geometries: box and polygon IoU as matrices and polygon
containment."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid

from tcip_annotation.state import (
    Annotation, BBox, Polygon, bbox_of, box_derivable, is_detection, object_rows, polygonal,
    prediction_score,
)

REVIEW_CONF_FLOOR = 0.25
"""The confidence below which a prediction is neither matched nor drawn when a visualization or a
scoring call states none: a viewing filter a person moves, never an execution record's operating
point."""

Rings = list[list[tuple[float, float]]]


def _boxes(boxes):
    """``boxes`` as an ``m x 4`` array: an array or tensor as given (so a tensor keeps its
    autograd), anything else as float64."""
    if getattr(boxes, "ndim", None) == 2:
        return boxes
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def xywh_corners(boxes) -> np.ndarray:
    """The ``[x, y, w, h]`` ``boxes`` as an ``m x 4`` float64 array of corner boxes."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.concatenate([b[:, :2], b[:, :2] + b[:, 2:]], axis=1)


def box_areas(boxes):
    """The area of each corner box of ``boxes`` (``m x 4``, an array or a tensor, in which the
    areas are answered)."""
    b = _boxes(boxes)
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def box_sizes(boxes) -> np.ndarray:
    """The characteristic size ``sqrt(w * h)`` of each corner box of ``boxes``, a negative
    extent counting as zero."""
    b = _boxes(boxes)
    return np.sqrt(np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0))


def box_centers(boxes) -> np.ndarray:
    """The center of each corner box of ``boxes`` (``m x 4``), as an ``m x 2`` array."""
    b = _boxes(boxes)
    return (b[:, :2] + b[:, 2:]) / 2.0


def iou_matrix(a, b, *, over: str = "union") -> np.ndarray:
    """The pairwise intersection of the corner boxes ``a`` (``m x 4``) and ``b`` (``k x 4``) over
    their union, or with ``over="smaller"`` over the smaller box, as an ``m x k`` float64 array;
    a pair whose denominator has no area is ``0``."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    iw = np.maximum(0.0, np.minimum(a[:, 2:3], b[None, :, 2]) - np.maximum(a[:, 0:1], b[None, :, 0]))
    ih = np.maximum(0.0, np.minimum(a[:, 3:4], b[None, :, 3]) - np.maximum(a[:, 1:2], b[None, :, 1]))
    inter = iw * ih
    area_a, area_b = box_areas(a)[:, None], box_areas(b)[None, :]
    denominator = area_a + area_b - inter if over == "union" else np.minimum(area_a, area_b)
    iou = np.zeros_like(denominator)
    np.divide(inter, denominator, out=iou, where=denominator > 0)
    return iou


def shapely_geometry(rings: Rings):
    """One or more closed rings as one valid Shapely geometry: a polygon for one ring, a
    multipolygon for several, every ring contributing, repaired by ``make_valid`` when invalid."""
    geom = (ShapelyPolygon(rings[0]) if len(rings) == 1
            else ShapelyMultiPolygon([ShapelyPolygon(r) for r in rings]))
    return geom if geom.is_valid else make_valid(geom)


def rings_iou_matrix(a: Sequence[Rings], b: Sequence[Rings]) -> np.ndarray:
    """The pairwise IoU of the regions ``a`` against ``b``, each a list of rings
    (:func:`shapely_geometry`), as a ``len(a) x len(b)`` float64 array; a region with no ring
    overlaps nothing."""
    ga = [shapely_geometry(r) if r else None for r in a]
    gb = [shapely_geometry(r) if r else None for r in b]
    out = np.zeros((len(ga), len(gb)))
    for i, g in enumerate(ga):
        for j, d in enumerate(gb):
            if g is not None and d is not None:
                inter = g.intersection(d).area
                union = g.area + d.area - inter
                out[i, j] = inter / union if union > 0 else 0.0
    return out


def _corners(records: list[dict]) -> np.ndarray:
    return xywh_corners([r["bbox"] for r in records])


def _cost_matrix(gt: list[dict], dt: list[dict], criterion: dict) -> tuple[np.ndarray, float]:
    """How far each ground-truth record is from each detection record under ``criterion``, and
    the farthest a match may be: the distance between box centers within ``tolerance`` for a
    center match; one minus the box IoU within one minus ``iou_threshold`` for an IoU match, or
    one minus the region IoU of the records' ``rings`` for a mask IoU match."""
    if criterion["kind"] == "center_match":
        cg, cd = box_centers(_corners(gt)), box_centers(_corners(dt))
        return (np.linalg.norm(cg[:, None, :] - cd[None, :, :], axis=-1),
                float(criterion["tolerance"]))
    iou = (rings_iou_matrix([r["rings"] for r in gt], [r["rings"] for r in dt])
           if criterion["kind"] == "mask_iou_match" else iou_matrix(_corners(gt), _corners(dt)))
    return 1.0 - iou, 1.0 - float(criterion["iou_threshold"])


def match_pairs(gt: list[dict], dt: list[dict], criterion: dict, *,
                policy: str) -> list[tuple[int, int]]:
    """A greedy 1:1 matcher of ``{"bbox": xywh}`` records (carrying ``rings`` too under a mask
    criterion) under ``criterion`` (a center match's ``tolerance`` or an IoU or mask IoU match's
    ``iou_threshold``), the limit inclusive, under one of two stated policies. Returns
    ``(gt_index, dt_index)`` pairs.

    ``policy="score_first"`` walks ``dt`` in the order given, each claiming its nearest unused
    ground truth; among equally near unused ground truths the last index wins.

    ``policy="distance_first"`` sorts every (gt, dt) pair within the limit by distance ascending
    and claims the closest first, ties broken by ``(gt index, dt index)`` ascending.
    """
    if policy not in ("score_first", "distance_first"):
        raise ValueError(f"match_pairs: unknown policy {policy!r}, expected 'score_first' or "
                         "'distance_first'")
    cost, limit = _cost_matrix(gt, dt, criterion)
    pairs: list[tuple[int, int]] = []
    if policy == "score_first":
        used = [False] * len(gt)
        for di in range(len(dt)):
            best_gi, best = -1, limit
            for gi in range(len(gt)):
                if not used[gi] and cost[gi, di] <= best:
                    best, best_gi = cost[gi, di], gi
            if best_gi >= 0:
                used[best_gi] = True
                pairs.append((best_gi, di))
        return pairs
    candidates = sorted((cost[gi, di], gi, di) for gi in range(len(gt)) for di in range(len(dt))
                        if cost[gi, di] <= limit)
    matched_gt: set[int] = set()
    matched_dt: set[int] = set()
    for _, gi, di in candidates:
        if gi not in matched_gt and di not in matched_dt:
            matched_gt.add(gi)
            matched_dt.add(di)
            pairs.append((gi, di))
    return pairs


@dataclass(frozen=True)
class Matching:
    """One image's detections matched to its ground truth, by index into the lists matched:
    ``pairs`` as ``(gt index, detection index)``, ``unpaired`` the detections that pair nothing
    and no crowd region ignores (the false positives), ``ignored`` those a crowd region ignores,
    and ``missed`` the ground-truth objects nothing pairs (the false negatives)."""

    pairs: list[tuple[int, int]]
    unpaired: list[int]
    ignored: set[int]
    missed: list[int]

    @property
    def counts(self) -> tuple[int, int, int]:
        """``(tp, fp, fn)``."""
        return len(self.pairs), len(self.unpaired), len(self.missed)


def objects_of(records: list[dict], *, crowd: bool = False) -> list[int]:
    """The indices of the ground-truth records (``{"iscrowd", ...}``) that are each one object:
    every one but a crowd region (:func:`~tcip_annotation.state.object_rows`); with ``crowd``,
    the crowd regions instead."""
    return [i for i, one in enumerate(object_rows([r["iscrowd"] for r in records]))
            if one is not crowd]


def pair_detections(gt: list[dict], dt: list[dict], criterion: dict, *,
                    policy: str = "score_first") -> Matching:
    """The platform's one matcher of detections to ground truth on one image: ``dt`` records
    (``{"bbox": xywh}``, pre-sorted by score descending) paired under ``policy``
    (:func:`match_pairs`) to ``gt``'s objects (``{"bbox": xywh, "iscrowd"}``,
    :func:`objects_of`) under ``criterion``. An unpaired detection whose center lies inside a
    crowd region's box is ignored, and a crowd region is no object to miss.
    """
    objects = objects_of(gt)
    crowds = [gt[i]["bbox"] for i in objects_of(gt, crowd=True)]
    pairs = [(objects[gi], di) for gi, di in match_pairs(
        [gt[i] for i in objects], dt, criterion, policy=policy)]
    paired_dt, paired_gt = {di for _, di in pairs}, {gi for gi, _ in pairs}
    ignored = {di for di, (cx, cy) in enumerate(box_centers(_corners(dt)).tolist())
               if di not in paired_dt and any(x <= cx <= x + w and y <= cy <= y + h
                                              for x, y, w, h in crowds)}
    return Matching(pairs=pairs,
                    unpaired=[di for di in range(len(dt)) if di not in paired_dt | ignored],
                    ignored=ignored, missed=[gi for gi in objects if gi not in paired_gt])


def match_record(a: Annotation) -> dict:
    """One annotation as the record the matcher reads: its box as ``[x, y, w, h]`` on the stored
    grid, its crowd flag, and its rings (a box's own rectangle, for a box)."""
    from tcip_annotation.json_io import xywh

    b = bbox_of(a.geometry)  # type: ignore[arg-type]
    return {"bbox": xywh(b.x1, b.y1, b.x2, b.y2), "iscrowd": a.iscrowd,
            "rings": geometry_rings(a.geometry)}  # type: ignore[arg-type]


def geometry_rings(geometry: BBox | Polygon) -> Rings:
    """The region a box or polygon covers, as the rings a mask match reads: a polygon's own rings,
    a box's rectangle as one ring."""
    return geometry.rings if polygonal(geometry) else [box_ring(bbox_of(geometry))]


_MATCHED_BY = ("kind", "iou_threshold", "tolerance")
"""The criterion fields :func:`match_pairs` reads; a criterion's provenance fields match nothing."""


def class_matchings(gt: list[dict], dt: list[dict], criterion: dict, *,
                    conf: float = -math.inf, policy: str = "score_first",
                    held: dict | None = None) -> dict[Any, Matching]:
    """One image's records matched class by class, the one class partition every count, curve,
    average precision, attribute pair and displayed pair derives from: for each ``category_id``
    ``gt`` or ``dt`` carries, that class's detections scoring at least ``conf``, highest first,
    paired to that class's ground truth (:func:`pair_detections` under ``policy``), indices into
    ``gt`` and ``dt``. ``held`` keeps each class's matching by its exact inputs, so a second
    request under the same criterion over the same detections takes the matching already
    made."""
    out: dict[Any, Matching] = {}
    for cid in sorted({r["category_id"] for r in (*gt, *dt)}):
        gi = [i for i, r in enumerate(gt) if r["category_id"] == cid]
        di = sorted((i for i, r in enumerate(dt) if r["category_id"] == cid and r["score"] >= conf),
                    key=lambda i: -dt[i]["score"])
        key = (id(gt), cid, tuple(sorted((k, v) for k, v in criterion.items() if k in _MATCHED_BY)),
               tuple(di), policy)
        if held is not None and key in held:
            out[cid] = held[key]
            continue
        m = pair_detections([gt[i] for i in gi], [dt[i] for i in di], criterion, policy=policy)
        out[cid] = Matching(pairs=[(gi[g], di[d]) for g, d in m.pairs],
                            unpaired=[di[d] for d in m.unpaired],
                            ignored={di[d] for d in m.ignored}, missed=[gi[g] for g in m.missed])
        if held is not None:
            held[key] = out[cid]
    return out


def merged(by_class: dict[Any, Matching]) -> Matching:
    """One image's per-class matchings (:func:`class_matchings`) as one matching, ordered by
    detection index."""
    ms = list(by_class.values())
    return Matching(pairs=sorted((p for m in ms for p in m.pairs), key=lambda pair: pair[1]),
                    unpaired=sorted(d for m in ms for d in m.unpaired),
                    ignored={d for m in ms for d in m.ignored},
                    missed=sorted(g for m in ms for g in m.missed))


def pair_proposals(annotations: list[Annotation], proposals: list[Annotation], criterion: dict,
                   *, conf_threshold: float | None = None) -> Matching:
    """``annotations`` and the ``proposals`` that are detections
    (:func:`~tcip_annotation.state.is_detection`) scoring at least ``conf_threshold`` (every one
    for ``None``) matched by subject (:func:`class_matchings`), indices into the two lists
    given."""
    gt = [i for i, a in enumerate(annotations) if box_derivable(a.geometry)]
    dt = [i for i, p in enumerate(proposals) if is_detection(p)]
    m = merged(class_matchings(
        [{**match_record(annotations[i]), "category_id": annotations[i].subject} for i in gt],
        [{**match_record(proposals[i]), "category_id": proposals[i].subject,
          "score": prediction_score(proposals[i])} for i in dt],
        criterion, conf=-math.inf if conf_threshold is None else conf_threshold))
    return Matching(pairs=[(gt[g], dt[d]) for g, d in m.pairs],
                    unpaired=[dt[d] for d in m.unpaired], ignored={dt[d] for d in m.ignored},
                    missed=[gt[g] for g in m.missed])


def box_ring(bbox: BBox) -> list[tuple[float, float]]:
    """``bbox``'s four corners as one closed ring, in a fixed order (x1,y1 -> x2,y1 -> x2,y2 ->
    x1,y2).
    """
    return [(bbox.x1, bbox.y1), (bbox.x2, bbox.y1), (bbox.x2, bbox.y2), (bbox.x1, bbox.y2)]


def point_in_polygon(x: float, y: float, polygon: Polygon) -> bool:
    """Whether a point lies inside any ring of ``polygon`` (:func:`shapely_geometry`)."""
    return shapely_geometry(polygon.rings).contains(ShapelyPoint(x, y))
