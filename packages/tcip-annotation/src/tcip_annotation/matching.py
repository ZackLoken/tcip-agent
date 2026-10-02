"""The platform's one matcher of detections to ground truth, and geometry helpers over
:class:`~tcip_annotation.state.Annotation` geometries: box IoU as a matrix and polygon
containment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid

from tcip_annotation.state import (
    Annotation, BBox, Polygon, bbox_of, box_derivable, is_detection, prediction_score,
)

REVIEW_CONF_FLOOR = 0.25
"""The confidence below which a prediction is neither matched nor drawn when a visualization or a
scoring call states none: a viewing filter a person moves, never an execution record's operating
point."""


def _centers_xywh(anns: list[dict]) -> list[tuple[float, float]]:
    return [(a["bbox"][0] + a["bbox"][2] / 2.0, a["bbox"][1] + a["bbox"][3] / 2.0) for a in anns]


def _match_cost(criterion: dict) -> tuple[Callable[[list[float], list[float]], float], float]:
    """How far one xywh box is from another under ``criterion``, and the farthest a match may be:
    the distance between centers within ``tolerance`` for a center match, one minus the IoU within
    one minus ``iou_threshold`` for an IoU match."""
    if criterion["kind"] == "center_match":
        def center_distance(g: list[float], d: list[float]) -> float:
            return (((g[0] + g[2] / 2) - (d[0] + d[2] / 2)) ** 2
                    + ((g[1] + g[3] / 2) - (d[1] + d[3] / 2)) ** 2) ** 0.5

        return center_distance, float(criterion["tolerance"])

    def iou_distance(g: list[float], d: list[float]) -> float:
        iw = max(0.0, min(g[0] + g[2], d[0] + d[2]) - max(g[0], d[0]))
        ih = max(0.0, min(g[1] + g[3], d[1] + d[3]) - max(g[1], d[1]))
        union = g[2] * g[3] + d[2] * d[3] - iw * ih
        return 1.0 - (iw * ih / union if union > 0 else 0.0)

    return iou_distance, 1.0 - float(criterion["iou_threshold"])


def match_pairs(gt_boxes: list[list[float]], dt_boxes: list[list[float]], criterion: dict, *,
                policy: str) -> list[tuple[int, int]]:
    """A greedy 1:1 matcher of xywh boxes under ``criterion`` (a center match's ``tolerance`` or
    an IoU match's ``iou_threshold``), the limit inclusive, under one of two stated policies.
    Returns ``(gt_index, dt_index)`` pairs.

    ``policy="score_first"`` walks ``dt_boxes`` in the order given, each claiming its nearest
    unused ground truth; among equally near unused ground truths the last index wins.

    ``policy="distance_first"`` sorts every (gt, dt) pair within the limit by distance ascending
    and claims the closest first, ties broken by ``(gt index, dt index)`` ascending.
    """
    cost, limit = _match_cost(criterion)
    pairs: list[tuple[int, int]] = []
    if policy == "score_first":
        used = [False] * len(gt_boxes)
        for di, d in enumerate(dt_boxes):
            best_gi, best = -1, limit
            for gi, g in enumerate(gt_boxes):
                if not used[gi] and cost(g, d) <= best:
                    best, best_gi = cost(g, d), gi
            if best_gi >= 0:
                used[best_gi] = True
                pairs.append((best_gi, di))
        return pairs
    if policy != "distance_first":
        raise ValueError(f"match_pairs: unknown policy {policy!r}, expected 'score_first' or "
                         "'distance_first'")
    candidates = sorted((cost(g, d), gi, di) for gi, g in enumerate(gt_boxes)
                        for di, d in enumerate(dt_boxes) if cost(g, d) <= limit)
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


def pair_detections(gt: list[dict], dt: list[dict], criterion: dict) -> Matching:
    """The platform's one matcher of detections to ground truth on one image: ``dt``
    (``{"bbox": xywh}`` records, pre-sorted by score descending) paired score-first to ``gt``'s
    objects (``{"bbox": xywh, "iscrowd"}``) under ``criterion``. An unpaired detection whose
    center lies inside a crowd region's box is ignored (COCO's crowd rule), and a crowd region is
    no object to miss.
    """
    objects = [i for i, a in enumerate(gt) if not a["iscrowd"]]
    crowds = [a["bbox"] for a in gt if a["iscrowd"]]
    pairs = [(objects[gi], di) for gi, di in match_pairs(
        [gt[i]["bbox"] for i in objects], [d["bbox"] for d in dt], criterion,
        policy="score_first")]
    paired_dt, paired_gt = {di for _, di in pairs}, {gi for gi, _ in pairs}
    ignored = {di for di, (cx, cy) in enumerate(_centers_xywh(dt)) if di not in paired_dt and any(
        x <= cx <= x + w and y <= cy <= y + h for x, y, w, h in crowds)}
    return Matching(pairs=pairs,
                    unpaired=[di for di in range(len(dt)) if di not in paired_dt | ignored],
                    ignored=ignored, missed=[gi for gi in objects if gi not in paired_gt])


def pair_proposals(annotations: list[Annotation], proposals: list[Annotation], criterion: dict,
                   *, conf_threshold: float | None = None) -> Matching:
    """:func:`pair_detections` run per subject over ``annotations`` and the ``proposals`` that are
    detections (:func:`~tcip_annotation.state.is_detection`) scoring at least ``conf_threshold``
    (every one for ``None``), highest score first, its indices into the two lists given."""
    from tcip_annotation.json_io import xywh

    def record(a: Annotation) -> dict:
        b = bbox_of(a.geometry)  # type: ignore[arg-type]
        return {"bbox": xywh(b.x1, b.y1, b.x2, b.y2), "iscrowd": a.iscrowd}

    out = Matching(pairs=[], unpaired=[], ignored=set(), missed=[])
    for subject in sorted({a.subject for a in (*annotations, *proposals)}):
        gt = [i for i, a in enumerate(annotations)
              if a.subject == subject and box_derivable(a.geometry)]
        dt = sorted((i for i, p in enumerate(proposals) if p.subject == subject and is_detection(p)
                     and (conf_threshold is None or prediction_score(p) >= conf_threshold)),
                    key=lambda i: -prediction_score(proposals[i]))
        m = pair_detections([record(annotations[i]) for i in gt],
                            [record(proposals[i]) for i in dt], criterion)
        out.pairs.extend((gt[gi], dt[di]) for gi, di in m.pairs)
        out.unpaired.extend(dt[di] for di in m.unpaired)
        out.ignored.update(dt[di] for di in m.ignored)
        out.missed.extend(gt[gi] for gi in m.missed)
    out.pairs.sort(key=lambda pair: pair[1])
    out.unpaired.sort()
    out.missed.sort()
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The pairwise IoU of the corner boxes ``a`` (``m x 4``) against ``b`` (``k x 4``), as an
    ``m x k`` float64 array; a pair whose union has no area is ``0``."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    iw = np.maximum(0.0, np.minimum(a[:, 2:3], b[None, :, 2]) - np.maximum(a[:, 0:1], b[None, :, 0]))
    ih = np.maximum(0.0, np.minimum(a[:, 3:4], b[None, :, 3]) - np.maximum(a[:, 1:2], b[None, :, 1]))
    inter = iw * ih
    union = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None] + (
        (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :] - inter
    iou = np.zeros_like(union)
    np.divide(inter, union, out=iou, where=union > 0)
    return iou


def _rings_to_shapely(rings: list[list[tuple[float, float]]]):
    """One or more simple closed rings -> a Shapely Polygon (one ring) or MultiPolygon (several);
    every ring contributes, never just the first/largest."""
    if len(rings) == 1:
        return ShapelyPolygon(rings[0])
    return ShapelyMultiPolygon([ShapelyPolygon(r) for r in rings])


def box_ring(bbox: BBox) -> list[tuple[float, float]]:
    """``bbox``'s four corners as one closed ring, in a fixed order (x1,y1 -> x2,y1 -> x2,y2 ->
    x1,y2).
    """
    return [(bbox.x1, bbox.y1), (bbox.x2, bbox.y1), (bbox.x2, bbox.y2), (bbox.x1, bbox.y2)]


def point_in_polygon(x: float, y: float, polygon: Polygon) -> bool:
    """Test whether a point lies inside any ring of a polygon using Shapely."""
    geom = _rings_to_shapely(polygon.rings)
    if not geom.is_valid:
        geom = make_valid(geom)
    return geom.contains(ShapelyPoint(x, y))
