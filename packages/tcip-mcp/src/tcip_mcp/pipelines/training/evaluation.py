"""Task-aware evaluation metrics + composite selection objective:
  * detection / instance_seg counts and average precision from the platform's one matcher
  (:mod:`tcip_annotation.matching`), by box or, for instance segmentation, by mask;
  * in-house scalar metrics for classification / ordinal / regression;
  * the composite selection objective (lower = better);
  * a task-agnostic two-pass ``evaluate()``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
import torch

from tcip_store import stored_number, stored_numbers

from tcip_annotation.json_io import xywh

if TYPE_CHECKING:
    from tcip_mcp.traits import TraitEntry

logger = logging.getLogger(__name__)

DEFAULT_SCORE_WEIGHTS: dict[str, float] = {"loss": 0.45, "f1": 0.35, "map50": 0.20}
"""Composite-objective weights, each acting on its term as :func:`compute_composite_objective`
scales it: a caller-owned selection policy, overridable through ``score_weights`` on every eval
surface; a documented default, no derivation."""

CENTER_MATCH_COMPARABILITY_KEYS: frozenset[str] = frozenset({
    "map50", "map", "iou_precision", "iou_recall", "iou_f1",
})
"""The metric keys reported comparability-only (``map50_role``) under a trait's own governing
criterion: the IoU-convention average precision and counts."""

VAL_METRIC_PREFIX = "val_"
"""The prefix every validation metric key carries in a run's metrics log and registry entry."""

VAL_LOSS_KEY = VAL_METRIC_PREFIX + "loss"
"""The validation loss's key in a run's metrics log."""

HIGHER_IS_BETTER_BY_METRIC: dict[str, bool] = {
    "loss": False,
    "objective": False,
    "mae": False,
    "rmse": False,
    "precision": True,
    "recall": True,
    "f1": True,
    "map": True,
    "map50": True,
    "iou_precision": True,
    "iou_recall": True,
    "iou_f1": True,
    "accuracy": True,
    "rank_acc": True,
    "quadratic_weighted_kappa": True,
    "r_squared": True,
    "mIoU": True,
    "dice": True,
    "pixel_acc": True,
}
"""Direction of a better value for each directed scalar metric, keyed by its bare
(un-``val_``-prefixed) name; a raw count, a signed bias and a non-finite state companion have no
direction."""


def _rounded(value):
    """One metric at the reported precision, a non-finite or non-numeric value unchanged."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return round(value, 6) if math.isfinite(value) else value


def _reported_metrics(values: dict) -> dict:
    """One task's metrics as the result record carries them.

    Scalars are rounded and a non-finite one becomes null beside a field naming its state;
    the per-class mappings some tasks return, which already carry None for an absent class,
    pass through untouched.
    """
    return stored_numbers({k: _rounded(v) for k, v in values.items()})


# ====================================================================
# Composite selection objective
# ====================================================================

def compute_composite_objective(
    val_loss: float, f1: float, map50: float, score_weights: dict | None = None
) -> float | None:
    """Lower-is-better selection/tuning score blending loss, F1 and mAP50, or ``None`` when
    the epoch has no useful score.

    ``w["loss"]*loss + w["f1"]*(1-f1)*10 + w["map50"]*(1-map50)*10``, ``w`` the default weights
    when ``score_weights`` is empty or ``None``. A non-positive or non-finite loss, or both
    quality terms
    below 0.01, answers ``None``.
    """
    w = score_weights or DEFAULT_SCORE_WEIGHTS
    vl = float(val_loss) if (val_loss is not None and math.isfinite(val_loss)) else float("inf")
    f1v = float(f1) if (f1 is not None and math.isfinite(f1)) else 0.0
    m50 = float(map50) if (map50 is not None and math.isfinite(map50)) else 0.0
    if vl <= 0 or not math.isfinite(vl):
        return None
    if f1v < 0.01 and m50 < 0.01:
        return None
    return w["loss"] * vl + w["f1"] * (1.0 - f1v) * 10 + w["map50"] * (1.0 - m50) * 10


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision, recall and F1 from counts: each ``0.0`` when its denominator is empty."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


AP_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
"""The IoU thresholds ``map`` averages average precision over, 0.50 to 0.95 by 0.05: the COCO
convention's own, so ``map`` stays comparable to a COCO-reported figure."""

_AP_RECALL_POINTS = 101
"""The recall points average precision is interpolated at, the COCO convention's own."""


def iou_criterion(iou_threshold: float, *, by_mask: bool) -> dict:
    """The matcher's IoU criterion at ``iou_threshold``: over each record's ``rings`` for a mask
    match (``by_mask``), else over its box."""
    return {"kind": "mask_iou_match" if by_mask else "iou_match",
            "iou_threshold": float(iou_threshold)}


def matched(per_image: list[dict], criterion: dict, *, conf: float = -math.inf,
            policy: str = "score_first", held: dict | None = None) -> list[dict[int, Any]]:
    """Each per-image record's class-by-class matching under ``criterion`` at ``conf``
    (:func:`~tcip_annotation.matching.class_matchings`, indices into the record's ``gt`` and
    ``dt``), each taken from ``held`` where the same inputs were already matched."""
    from tcip_annotation.matching import class_matchings

    return [class_matchings(rec["gt"], rec["dt"], criterion, conf=conf, policy=policy, held=held)
            for rec in per_image]


def average_precision(per_image: list[dict], criterion: dict, *,
                      held: dict | None = None) -> float:
    """The area under the precision-recall curve of every detection in ``per_image`` matched
    under ``criterion`` (:func:`matched`, every detection), averaged over the classes the ground
    truth's objects carry: per class, detections ranked by score across images, one a crowd
    region ignores counting neither way, precision made monotone from the right and read at
    :data:`_AP_RECALL_POINTS` recall points. ``0.0`` for a reference holding no object."""
    images = list(zip(per_image, matched(per_image, criterion, held=held)))
    per_class = []
    for cid in sorted({a["category_id"] for rec in per_image for a in gt_objects(rec)}):
        ranked: list[tuple[float, bool]] = []
        n_objects = 0
        for rec, by_class in images:
            if cid not in by_class:
                continue
            m = by_class[cid]
            n_objects += len(m.pairs) + len(m.missed)
            paired = {d for _, d in m.pairs}
            ranked += [(_dt_score(rec["dt"][d]), d in paired) for d in sorted(
                paired | set(m.unpaired), key=lambda d: (-_dt_score(rec["dt"][d]), d))]
        ranked.sort(key=lambda r: -r[0])
        hits = np.asarray([hit for _, hit in ranked], dtype=bool)
        tp, fp = np.cumsum(hits), np.cumsum(~hits)
        recall = tp / n_objects
        precision = np.maximum.accumulate((tp / np.maximum(tp + fp, 1))[::-1])[::-1]
        at = np.searchsorted(recall, np.linspace(0.0, 1.0, _AP_RECALL_POINTS), side="left")
        per_class.append(float(np.mean([precision[i] if i < len(precision) else 0.0
                                        for i in at])))
    return float(np.mean(per_class)) if per_class else 0.0


def gt_objects(rec: dict, *, crowd: bool = False, class_id: int | None = None) -> list[dict]:
    """A per-image record's ground-truth objects of ``class_id`` (every class for ``None``): every
    ``gt`` entry but a crowd region (:func:`~tcip_annotation.matching.objects_of`), which is never
    one object in a count, a size or a spacing; with ``crowd``, the crowd regions instead.
    """
    from tcip_annotation.matching import objects_of

    gt = rec["gt"]
    return [gt[i] for i in objects_of(gt, crowd=crowd) if _in_class(gt[i], class_id)]


def _in_class(record: dict, class_id: int | None) -> bool:
    """Whether an evaluation record is of ``class_id``; every record is, for ``None``."""
    return class_id is None or record["category_id"] == class_id


def gt_class_avg_size(per_image: list[dict], class_id: int | None = None) -> float:
    """Average characteristic size (:func:`~tcip_annotation.matching.box_sizes`) of the ground
    truth's objects of ``class_id``, every class for ``None``; ``0.0`` for none."""
    from tcip_annotation.matching import box_sizes, xywh_corners

    boxes = [a["bbox"] for rec in per_image for a in gt_objects(rec, class_id=class_id)]
    return float(np.mean(box_sizes(xywh_corners(boxes)))) if boxes else 0.0


def mean_of_present_counts(counts: Iterable[int]) -> float:
    """Mean of the positive entries in ``counts``; ``0.0`` when none is positive."""
    present = [c for c in counts if c > 0]
    return float(np.mean(present)) if present else 0.0


def gt_class_typical_count(per_image: list[dict], class_id: int | None = None) -> float:
    """Mean ground-truth count of ``class_id`` (all classes pooled when ``None``) over the images
    holding any, independent of detections and confidence; ``0.0`` when no image holds one."""
    counts = [len(gt_objects(rec, class_id=class_id)) for rec in per_image]
    return mean_of_present_counts(counts)


def localization_frac(trait: TraitEntry, boxes_per_image: list[list[list[float]]]
                      ) -> tuple[float, str]:
    """The center-match tolerance, as a fraction of the average object size, and its source: the
    ground truth's own nearest-neighbor spacing (``boxes_per_image`` one list of xywh boxes per
    image), else, when no two same-class objects sit close enough to derive it, the trait
    revision's stated ``localization_tolerance_frac``."""
    from tcip_mcp.pipelines.derivations import derive_localization_tolerance_frac

    frac = derive_localization_tolerance_frac(boxes_per_image)
    if frac is not None:
        return frac, "GT nearest-neighbor spacing (p10 + margin)"
    return trait.localization_tolerance_frac, (
        f"the trait's stated localization_tolerance_frac ({trait.localization_tolerance}); no "
        "same-class neighbor in this ground truth to derive one from")


def resolve_match_criterion(trait: TraitEntry | None, per_image: list[dict], *,
                            class_id: int | None = None, iou_threshold: float = 0.5,
                            by_mask: bool = False) -> dict:
    """The localization criterion that governs a trait's phenotype count and model selection, as
    ``{kind, tolerance_frac and tolerance | iou_threshold, derived_from, trait}``, resolved once
    over the reference ``per_image`` and carried whole to every count and match over it.

    With no trait it is IoU matching at ``iou_threshold``, the comparability convention. With one,
    its stated ``localization`` governs, an unauthored one refusing
    (:class:`~tcip_mcp.traits.UnauthoredFieldError`): a center match's tolerance is a
    fraction of the average object size (:func:`localization_frac`), scaled to ``per_image`` here
    and to another reference by :func:`scaled_to`; an IoU match's threshold is the one the ground
    truth's own box sizes derive under the trait's authored ``iou_jitter_px`` and ``iou_margin``
    (unauthored ones refusing the same way), and a reference with no box to derive it from, or
    whose boxes the jitter displaces past overlap, refuses. An IoU match is over masks when
    ``by_mask`` (:func:`iou_criterion`).
    """
    if trait is None:
        return {**iou_criterion(iou_threshold, by_mask=by_mask),
                "derived_from": "comparability convention", "trait": None}
    from tcip_mcp.pipelines.derivations import IOU_MATCH_DERIVATION, derive_iou_match_threshold
    from tcip_mcp.traits import CENTER_MATCH, IOU_MATCH, LOCALIZATION_FIELDS, authored

    authored(trait, ("localization",))
    boxes_per_image = [[a["bbox"] for a in gt_objects(rec, class_id=class_id)]
                       for rec in per_image]
    if trait.localization == CENTER_MATCH:
        frac, frac_source = localization_frac(trait, boxes_per_image)
        return scaled_to({"kind": "center_match", "tolerance_frac": frac,
                          "derived_from": frac_source, "trait": trait.name}, per_image, class_id)
    authored(trait, LOCALIZATION_FIELDS[IOU_MATCH])
    threshold = derive_iou_match_threshold(
        boxes_per_image, jitter_px=cast(float, trait.iou_jitter_px),
        margin=cast(float, trait.iou_margin))
    if threshold is None:
        raise ValueError(
            f"trait {trait.name!r} matches by IoU, and this reference holds no ground-truth box "
            "to derive the IoU a match must reach from; evaluate against a labeled reference.")
    return {**iou_criterion(threshold, by_mask=by_mask),
            "derived_from": IOU_MATCH_DERIVATION, "trait": trait.name}


def scaled_to(criterion: dict, per_image: list[dict], class_id: int | None = None) -> dict:
    """``criterion`` as it applies to ``per_image``: a center match's tolerance scaled to that
    reference's own average object size; an IoU match unchanged."""
    if criterion["kind"] != "center_match":
        return criterion
    return {**criterion, "tolerance": float(
        criterion["tolerance_frac"] * gt_class_avg_size(per_image, class_id=class_id))}


def _dt_score(d: dict) -> float:
    """A detection record's confidence: a record with no ``score``, or a ``score`` of ``None``,
    refuses by name.
    """
    if "score" not in d:
        raise ValueError(f"detection record has no 'score' field, cannot count it: {d!r}")
    if d["score"] is None:
        raise ValueError(f"detection record's 'score' is None, cannot count it: {d!r}")
    return float(d["score"])


def governing_counts(per_image: list[dict], criterion: dict, *, conf_threshold: float,
                     class_id: int | None = None) -> dict:
    """tp/fp/fn/precision/recall/f1 of ``per_image`` matched under ``criterion`` at
    ``conf_threshold`` (:func:`counted`), for ``class_id`` or every class."""
    return counted(matched(per_image, criterion, conf=conf_threshold), criterion, class_id)


def counted(by_image: list[dict[int, Any]], criterion: dict, class_id: int | None = None) -> dict:
    """tp/fp/fn/precision/recall/f1 from the per-image matchings ``by_image`` under
    ``criterion``, for ``class_id`` or every class."""
    m = _count_stats(by_image, class_id)
    return {"tp": int(m["tp"]), "fp": int(m["fp"]), "fn": int(m["fn"]),
            **{k: round(m[k], 6) for k in ("precision", "recall", "f1")}, "criterion": criterion}


def attribute_pairs(per_image: list[dict], criterion: dict, *, conf: float,
                    column: int) -> list[tuple[Any, int, int]]:
    """``(image_id, reference id, predicted id)`` under attribute ``column`` for each reference
    object matched to one detection of its class scoring at least ``conf``, distance first, under
    ``criterion`` scaled to ``per_image`` (:func:`matched`). A reference object unassessed for
    that attribute contributes no pair."""
    from tcip_annotation.json_io import UNASSESSED

    pairs = []
    for rec, by_class in zip(per_image, matched(per_image, scaled_to(criterion, per_image),
                                                conf=conf, policy="distance_first")):
        for g, d in (pair for m in by_class.values() for pair in m.pairs):
            truth = rec["gt"][g]["attributes"][column]
            if truth != UNASSESSED:
                pairs.append((rec["image_id"], truth, rec["dt"][d]["attributes"][column]))
    return pairs


def _count_stats_at_conf(per_image: list[dict], *, criterion: dict, conf: float,
                         class_id: int | None) -> dict:
    """:func:`_count_stats` of ``per_image`` matched once at ``conf`` (:func:`matched`)."""
    return _count_stats(matched(per_image, criterion, conf=conf), class_id)


def _count_stats(matched: list[dict[int, Any]], class_id: int | None) -> dict:
    """Counting statistics over each image's per-class matchings ``matched``, for one class or,
    for ``None``, each image's counts summed over its classes: a curve entry.

    Two scopes of the same per-image bias travel side by side. The whole-reference statistics
    (``count_bias_mean``/``count_bias_std`` over ``n_images``) are what a conf picker compares
    across the grid: ``n_images`` is the same at every conf, so minimizing ``count_bias_mean``
    minimizes the reference's total signed miscount. The present-scoped statistics
    (``count_bias_mean_present``/``count_bias_std_present`` over ``n_present``) are what an
    equivalence gate compares against a relative tolerance scaled by a density measured over
    present images only (:func:`mean_of_present_counts`): an image with no GT and no surviving
    detection is excluded from them.
    """
    tp = fp = fn = 0
    biases: list[int] = []
    present_biases: list[int] = []
    for by_class in matched:
        ms = [m for cid, m in by_class.items() if class_id is None or cid == class_id]
        t, f, n = (sum(m.counts[i] for m in ms) for i in range(3))
        tp += t
        fp += f
        fn += n
        biases.append(f - n)
        if t + n + f:  # a counted object or detection; one ignored in a crowd region is neither
            present_biases.append(f - n)
    abs_biases = [abs(b) for b in biases]
    return {
        "tp": tp, "fp": fp, "fn": fn, **precision_recall_f1(tp, fp, fn),
        "count_bias_mean": float(np.mean(biases)) if biases else 0.0,
        "abs_count_error_mean": float(np.mean(abs_biases)) if biases else 0.0,
        # Tail dispersion, a p90 of |bias|, not another mean, since a mean can hide one
        # badly-off image among many. Reference-sufficiency terms, computed here, once,
        # so the gate never re-derives a second matcher over the same per-image biases.
        "count_error_p90": float(np.quantile(abs_biases, 0.9)) if abs_biases else 0.0,
        "count_bias_std": float(np.std(biases, ddof=1)) if len(biases) > 1 else 0.0,
        "n_images": len(biases),
        # Images holding a counted object or detection of this class, distinct from n_images
        # (the whole holdout) because a scope scarce in the reference gets the denominator of both
        # its bias and its standard error from how much evidence there really is, not diluted by
        # images that say nothing about it.
        "n_present": len(present_biases),
        "count_bias_mean_present": float(np.mean(present_biases)) if present_biases else 0.0,
        "count_bias_std_present": (float(np.std(present_biases, ddof=1))
                                   if len(present_biases) > 1 else 0.0),
    }


def derive_operating_point_curve(per_image: list[dict], *, criterion: dict,
                                 class_id: int | None = None,
                                 conf_grid: list[float] | None = None,
                                 max_thresholds: int = 80) -> dict:
    """Sweep the confidence threshold over ``per_image`` records, matched under ``criterion``.

    One model pass produces ``per_image`` (unfiltered dt with scores); this sweeps conf in Python,
    no re-forwarding. For each conf: aggregate TP/FP/FN and per-image count bias (FP-FN). An
    explicit ``conf_grid`` (e.g. a single-element ``[conf]``) evaluates exactly those points.
    Returns ``{criterion, class_id, curve:[{conf, tp, fp, fn, precision, recall, f1,
    count_bias_mean, abs_count_error_mean, count_error_p90, count_bias_std, n_images, n_present,
    count_bias_mean_present, count_bias_std_present, per_class}]}``. See
    :func:`_count_stats_at_conf` for the two bias scopes.

    ``per_class`` carries the same statistics within each class the records carry, keyed by
    ``str(category_id)`` (string keys so an in-memory sweep and one round-tripped through a stored
    JSON record have the same shape), from the one matching per conf the pooled entry sums. Class
    ids come from the records themselves.
    """
    scores = sorted({_dt_score(d) for rec in per_image for d in rec["dt"]})
    if conf_grid is None:
        if len(scores) > max_thresholds:
            conf_grid = list(np.linspace(scores[0], scores[-1], max_thresholds))
        else:
            conf_grid = list(scores)
        conf_grid = sorted(set([0.0, *conf_grid]))
    curve: list[dict] = []
    for conf in conf_grid:
        by_image = matched(per_image, criterion, conf=conf)
        class_ids = ([class_id] if class_id is not None
                     else sorted({cid for by_class in by_image for cid in by_class}))
        curve.append({"conf": float(conf), **_count_stats(by_image, class_id),
                      "per_class": {str(cid): _count_stats(by_image, cid) for cid in class_ids}})
    return {"criterion": criterion, "class_id": class_id, "curve": curve}


def worst_class_count_bias(entry: dict) -> float:
    """The largest |mean per-image count bias| over the classes in one curve entry, the class this
    conf serves worst; the pooled bias for an entry over no class."""
    return max((abs(s["count_bias_mean"]) for s in entry["per_class"].values()),
               default=abs(entry["count_bias_mean"]))


def pick_count_unbiased(sweep: dict) -> float:
    """The conf that minimizes the worst per-class |mean per-image count bias| (tie-break: lower
    pooled |bias|, higher F1, lower |error|, higher conf): the count-trait operating point, where
    the model's totals match GT totals."""
    best = min(sweep["curve"],
               key=lambda c: (worst_class_count_bias(c), abs(c["count_bias_mean"]), -c["f1"],
                              c["abs_count_error_mean"], -c["conf"]))
    return best["conf"]


def classes_with_evidence(entry: dict) -> set[str]:
    """The classes one curve entry actually says something about: those with a GT object or a
    surviving detection at that conf (``tp + fp + fn > 0``). A class whose entry is all zeros
    carries no evidence of an unbiased count.
    """
    return {cid for cid, s in entry["per_class"].items() if s["tp"] + s["fp"] + s["fn"] > 0}


def pick_f1_max(sweep: dict) -> float:
    """The F1-max conf, reported alongside the count-unbiased point to show the trade-off."""
    return max(sweep["curve"], key=lambda c: c["f1"])["conf"]


def image_record(width: float, height: float, gt: list[dict], dt: list[dict],
                 image_id=None) -> dict:
    """One per-image evaluation record: ``{"width", "height", "gt", "dt"}`` and ``image_id`` when
    given. The one shape every per-image record is built in."""
    rec = {"width": int(width), "height": int(height), "gt": list(gt), "dt": list(dt)}
    if image_id is not None:
        rec["image_id"] = image_id
    return rec


def _mask_rings(mask: Any, threshold: float | None) -> list:
    """A binary (``threshold`` ``None``) or soft mask's regions as rings
    (:func:`~tcip_annotation.mask_contours.mask_to_polygon_rings`), for a mask match."""
    from tcip_annotation.mask_contours import mask_to_polygon_rings

    m = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    return mask_to_polygon_rings(m, threshold=threshold)


def gt_record(bbox: list[float], category_id: int, crowd: Any) -> dict:
    """One ground-truth evaluation record: its class, its ``[x, y, w, h]`` box and its crowd flag.
    The one shape every ground-truth record is built in."""
    return {"category_id": int(category_id), "bbox": bbox, "iscrowd": int(crowd)}


def dt_record(bbox: list[float], category_id: Any, score: Any) -> dict:
    """One detection's evaluation record: its ``[x, y, w, h]`` box as given, its class and its
    score. The one shape every detection record is built in, whatever coordinates it is on."""
    return {"category_id": int(category_id), "bbox": bbox, "score": float(score)}


def detection_record(box: Sequence[float], label: Any, score: Any) -> dict:
    """One detection's evaluation record from a predictor's corner ``box``, on the stored grid
    (:func:`~tcip_annotation.json_io.xywh`).
    """
    return dt_record(xywh(*box), label, score)


def _rows(values: Any) -> list:
    """A tensor's, array's or list's rows as a list."""
    return values.tolist() if hasattr(values, "tolist") else list(values)


def _with_attributes(records: list[dict], values: Any) -> list[dict]:
    """``records``, each carrying its row of ``values`` (one id per attribute) under
    ``attributes``; unchanged when ``values`` is ``None``."""
    if values is not None:
        for record, row in zip(records, _rows(values), strict=True):
            record["attributes"] = [int(v) for v in row]
    return records


def prediction_record(result: Mapping[str, Any], gt: list[dict], *, image_id: str) -> dict:
    """One per-image evaluation record from a detection result (its ``width``, ``height``,
    corner ``boxes``, ``scores``, ``labels``, ``attributes`` and ``masks`` where it carries them,
    each mask's polygons as its record's ``rings``, and ``cap_hit``) and the image's ground-truth
    records ``gt``, named ``image_id``. Boxes, scores, labels and masks differing in length refuse
    (``ValueError``)."""
    dt = _with_attributes([detection_record(box, label, score) for box, score, label
                           in zip(result["boxes"], result["scores"], result["labels"],
                                  strict=True)], result.get("attributes"))
    for record, mask in zip(dt, result.get("masks", ()), strict="masks" in result):
        record["rings"] = [list(zip(p[0::2], p[1::2])) for p in mask["segmentation"]]
    return {**image_record(int(result["width"]), int(result["height"]), gt, dt,
                           image_id=image_id),
            "cap_hit": result["cap_hit"]}


def gt_records(target: Mapping[str, Any]) -> list[dict]:
    """A target's rows, corner ``boxes`` beside ``labels``, the crowd flag, the ``attributes`` row
    and each row's ``geometry`` as its ``rings`` (:func:`~tcip_annotation.matching.geometry_rings`)
    where the target carries them, as evaluation ground-truth records on the stored grid
    (:func:`~tcip_annotation.json_io.xywh`), from a tensor, array or list target alike, its crowd
    flags read through :func:`~tcip_mcp.pipelines.data.datasets.crowd_of`.
    """
    from tcip_annotation.matching import geometry_rings

    from tcip_mcp.pipelines.data.datasets import crowd_of

    if not len(target["boxes"]):
        return []
    records = _with_attributes(
        [gt_record(xywh(*box), lab, crowd)
         for box, lab, crowd in zip(_rows(target["boxes"]), _rows(target["labels"]),
                                    _rows(crowd_of(target)))], target.get("attributes"))
    for record, geometry in zip(records, target.get("geometry", ()),
                                strict="geometry" in target):
        record["rings"] = geometry_rings(geometry)
    return records


def records_from_detector(target: dict, output: dict, *, width: int, height: int,
                          include_masks: bool = False) -> dict:
    """A torchvision target and a detector's output as one per-image record. With
    ``include_masks`` (instance segmentation) every ground-truth and detection record also
    carries its mask's ``rings``, the predicted masks cut at the platform's binarize threshold
    (:func:`~tcip_mcp.pipelines.measurement.mask_geometry.resolve_binarize_threshold`), for a mask
    match."""
    gt = gt_records(target)
    dt = _with_attributes([detection_record(box, c, s) for box, c, s in zip(
        output["boxes"].detach().cpu().tolist(), output["labels"].detach().cpu().tolist(),
        output["scores"].detach().cpu().tolist(), strict=True)], output.get("attributes"))
    if include_masks:
        from tcip_mcp.pipelines.measurement.mask_geometry import resolve_binarize_threshold

        cut = resolve_binarize_threshold()["value"]
        for ann, mask in zip(gt, target["masks"], strict=True):
            ann["rings"] = _mask_rings(mask, None)
        for res, mask in zip(dt, output["masks"], strict=True):
            res["rings"] = _mask_rings(mask, cut)
    return image_record(width, height, gt, dt, image_id=target.get("image_id"))


def subject_category_ids(annotations) -> dict[str, int]:
    """Each box-derivable annotation's subject mapped to a 1-indexed COCO category id (background
    0, like detector labels), in first-seen order; a geometry-less label and a ``Point`` mint
    none."""
    from tcip_annotation.state import box_derivable

    names: list[str] = []
    for a in annotations:
        if box_derivable(a.geometry) and a.subject not in names:
            names.append(a.subject)
    return {n: i + 1 for i, n in enumerate(names)}


def records_from_annotation(gt, preds, *, width: int, height: int,
                            name_id: dict[str, int] | None = None) -> dict:
    """Ground-truth and predicted :class:`Annotation` lists (a prediction carrying a ``score``)
    as one per-image record, every record carrying its ``rings``
    (:func:`~tcip_annotation.matching.match_record`) beside its box and the ``index`` of the
    annotation it was built from in its list. ``name_id`` maps each subject
    to the one category id it has in every image scored together
    (:func:`subject_category_ids`), this image's own map when ``None``. A geometry-less annotation
    and a :class:`~tcip_annotation.state.Point` contribute no record: neither has a region to
    score."""
    from tcip_annotation.matching import match_record
    from tcip_annotation.state import box_derivable, is_detection, prediction_score

    if name_id is None:
        name_id = subject_category_ids((*gt, *preds))

    gt_recs = [{**gt_record(m["bbox"], name_id[a.subject], a.iscrowd), "rings": m["rings"],
                "index": i}
               for i, a in enumerate(gt) if box_derivable(a.geometry) for m in [match_record(a)]]
    dt_recs = [{**dt_record(m["bbox"], name_id[a.subject], prediction_score(a)),
                "rings": m["rings"], "index": i}
               for i, a in enumerate(preds) if is_detection(a) for m in [match_record(a)]]
    return image_record(width, height, gt_recs, dt_recs)


def bucket_reads(images: Sequence[Any], bucket: Any) -> list[tuple[Any, list, list | None]]:
    """The one read a scoring of ``bucket``'s documents for the logical ``images`` makes
    (:func:`~tcip_mcp.pipelines.image_utils.resolve_image_paths`' answers): each image's
    ``(source, ground truth, predictions)``, its label document's annotations and, where the
    bucket's record names a document for it, that document's (``None`` where it names none), each
    read once. An image with no label document refuses
    (:class:`~tcip_annotation.json_io.UnreadableLabelDocumentError`)."""
    from pathlib import Path

    from tcip_annotation.json_io import read_label_document, read_predictions

    from tcip_mcp.dataset_layout import label_key_of
    from tcip_mcp.pipelines.image_utils import source_path_of

    out = []
    for source in images:
        named = Path(source_path_of(source))
        document = bucket.document_key(named.stem)
        out.append((source, read_label_document(label_key_of(named)).annotations,
                    read_predictions(document) if document is not None else None))
    return out


class ScoredImage(NamedTuple):
    """One image of a bucket's scoring: the image's :class:`~tcip_mcp.pipelines.raster_source.
    SourceHeader` (the logical image as its ``source``, its frame read off it, and a render
    opening the image through it), its ground truth, the bucket's predictions for it (``None``
    where the bucket names no document for it, which leaves it out of every aggregate), and its
    governing matching, indices into ``gt`` and ``preds`` (every object missed where
    unpredicted)."""

    header: Any
    gt: list
    preds: list | None
    matching: Any


def score_bucket(images: Sequence[Any], bucket: Any, *, iou_threshold: float,
                 conf_threshold: float, trait: TraitEntry | None) -> tuple[list[ScoredImage], dict]:
    """``bucket``'s documents for the logical ``images`` scored against their ground truth from
    their one read (:func:`bucket_reads`): :func:`detection_metrics` at ``conf_threshold`` over the
    predicted images' records, one subject-to-id map across every image, by mask when any
    annotation is a polygon, under the criterion that governs (``trait``'s own when given, the IoU
    convention at ``iou_threshold`` otherwise), and each image as a :class:`ScoredImage` carrying
    the matching that scoring made of it."""
    from tcip_annotation.matching import Matching, merged
    from tcip_annotation.state import polygonal

    from tcip_mcp.pipelines.raster_source import SourceHeader

    read = [(SourceHeader(src), gt, preds) for src, gt, preds in bucket_reads(images, bucket)]
    name_id = subject_category_ids(
        [a for _h, gt, preds in read for a in (*gt, *(preds or ()))]
    )
    records = [records_from_annotation(gt, preds or [], width=header.display_frame[0],
                                       height=header.display_frame[1], name_id=name_id)
               for header, gt, preds in read]
    predicted = [k for k, (_h, _gt, preds) in enumerate(read) if preds is not None]
    metrics = detection_metrics(
        [records[k] for k in predicted], trait=trait, conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        by_mask=any(polygonal(a.geometry) for k in predicted
                    for a in (*read[k][1], *cast(list, read[k][2]))))
    by_image = dict(zip(predicted, metrics["matchings"]))
    unpredicted = [k for k in range(len(read)) if k not in by_image]
    by_image.update(zip(unpredicted, matched([records[k] for k in unpredicted],
                                             metrics["governing_criterion"])))

    def annotated(rec: dict, by_class: dict) -> Matching:
        m = merged(by_class)
        gi, di = [r["index"] for r in rec["gt"]], [r["index"] for r in rec["dt"]]
        return Matching(pairs=[(gi[g], di[d]) for g, d in m.pairs],
                        unpaired=[di[d] for d in m.unpaired], ignored={di[d] for d in m.ignored},
                        missed=[gi[g] for g in m.missed])

    return [ScoredImage(header, gt, preds, annotated(records[k], by_image[k]))
            for k, (header, gt, preds) in enumerate(read)], metrics


def classification_metrics(
    pred_labels: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> dict:
    """Accuracy + macro-F1 + per-class precision/recall/f1/support/count_bias, each per-class
    mapping keyed by the class index as a string (a JSON object's key).

    ``count_bias[c] = (predicted count - true count) / true count``: the phenotype is the
    positive-state fraction, so a class the classifier over-predicts inflates the fraction even at
    high accuracy.
    """
    pred = pred_labels.detach().cpu().long()
    gt = targets.detach().cpu().long()
    if gt.numel() == 0:
        return {"accuracy": 0.0, "f1": 0.0, "per_class": {}, "count_bias": {}}
    accuracy = (pred == gt).float().mean().item()
    per_class: dict[str, dict] = {}
    f1s = []
    for c in range(num_classes):
        tp = int(((pred == c) & (gt == c)).sum())
        fp = int(((pred == c) & (gt != c)).sum())
        fn = int(((pred != c) & (gt == c)).sum())
        support = int((gt == c).sum())
        pred_count = int((pred == c).sum())
        prf = precision_recall_f1(tp, fp, fn)
        f1s.append(prf["f1"])
        per_class[str(c)] = {**prf, "support": support,
                        "count_bias": (pred_count - support) / support if support > 0 else 0.0}
    return {
        "accuracy": accuracy,
        "f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "per_class": per_class,
        "count_bias": {c: per_class[c]["count_bias"] for c in per_class},
    }


def quadratic_weighted_kappa(
    pred_ranks: torch.Tensor, gt_ranks: torch.Tensor, num_ranks: int,
) -> float | None:
    """Chance-corrected ordinal agreement over ``num_ranks``, the run's own rank count: squared
    rank-distance weights, expected agreement from the scored set's own observed rank marginals,
    the ordinal counterpart to :func:`tcip_mcp.pipelines.operating_point._classification_kappa`.
    ``None`` when undefined: no items, or expected disagreement is zero (every populated
    true/predicted pair shares one rank).
    """
    pred = pred_ranks.detach().cpu().round().long()
    gt = gt_ranks.detach().cpu().round().long()
    n = gt.numel()
    if n == 0:
        return None
    k = num_ranks
    if k < 2:
        return None
    observed = torch.zeros((k, k))
    for t, p in zip(gt.tolist(), pred.tolist()):
        observed[t][p] += 1
    row_marginal = observed.sum(dim=1)
    col_marginal = observed.sum(dim=0)
    expected = torch.outer(row_marginal, col_marginal) / n
    idx = torch.arange(k, dtype=torch.float32)
    weights = (idx.unsqueeze(1) - idx.unsqueeze(0)) ** 2
    expected_disagreement = (weights * expected).sum().item()
    if expected_disagreement == 0.0:
        return None
    observed_disagreement = (weights * observed).sum().item()
    return 1.0 - observed_disagreement / expected_disagreement


def r_squared(pred_values: torch.Tensor, gt_values: torch.Tensor) -> float | None:
    """Fraction of variance explained beyond trivially predicting the scored set's own mean, the
    regression counterpart to :func:`quadratic_weighted_kappa`'s chance-correction (both express
    "how much better than the trivial baseline achievable from this set's own distribution").
    ``None`` when undefined: no items, or the set's own values are constant (no variance to
    explain).
    """
    pred = pred_values.detach().cpu().float()
    gt = gt_values.detach().cpu().float()
    if gt.numel() == 0:
        return None
    ss_tot = ((gt - gt.mean()) ** 2).sum().item()
    if ss_tot == 0.0:
        return None
    ss_res = ((gt - pred) ** 2).sum().item()
    return 1.0 - ss_res / ss_tot


def concordance_correlation_coefficient(
    pred_values: torch.Tensor, gt_values: torch.Tensor
) -> float | None:
    """Lin's concordance correlation coefficient: agreement between ``pred_values`` and
    ``gt_values`` as precision (Pearson correlation) times an accuracy/bias penalty. A prediction
    perfectly correlated with GT but systematically offset (a constant bias, or a scale != 1)
    scores low here. ``None`` when undefined: no items, or either series has zero variance.

    ``CCC = 2*r*sigma_pred*sigma_gt / (sigma_pred^2 + sigma_gt^2 + (mean_pred - mean_gt)^2)``, with
    ``r`` the Pearson correlation and ``sigma`` the population (not sample) standard deviation, the
    convention :func:`r_squared` uses.
    """
    pred = pred_values.detach().cpu().float()
    gt = gt_values.detach().cpu().float()
    n = gt.numel()
    if n == 0:
        return None
    pred_mean, gt_mean = pred.mean(), gt.mean()
    pred_var = ((pred - pred_mean) ** 2).mean().item()
    gt_var = ((gt - gt_mean) ** 2).mean().item()
    if pred_var == 0.0 or gt_var == 0.0:
        return None
    covariance = ((pred - pred_mean) * (gt - gt_mean)).mean().item()
    return (2.0 * covariance) / (pred_var + gt_var + (pred_mean.item() - gt_mean.item()) ** 2)


def ordinal_metrics(pred_ranks: torch.Tensor, gt_ranks: torch.Tensor, num_ranks: int) -> dict:
    """Ordinal metrics over ``num_ranks``, the run's own rank count, which the scored half is
    never asked for: it may not reach every rank."""
    pred = pred_ranks.detach().cpu().float()
    gt = gt_ranks.detach().cpu().float()
    if gt.numel() == 0:
        return {"mae": 0.0, "rank_acc": 0.0, "quadratic_weighted_kappa": None}
    return {
        "mae": (pred - gt).abs().mean().item(),
        "rank_acc": (pred.round() == gt.round()).float().mean().item(),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(pred_ranks, gt_ranks, num_ranks),
    }


def regression_metrics(pred_values: torch.Tensor, gt_values: torch.Tensor) -> dict:
    pred = pred_values.detach().cpu().float()
    gt = gt_values.detach().cpu().float()
    if gt.numel() == 0:
        return {"mae": 0.0, "rmse": 0.0, "r_squared": None}
    return {
        "mae": (pred - gt).abs().mean().item(),
        "rmse": ((pred - gt) ** 2).mean().sqrt().item(),
        "r_squared": r_squared(pred_values, gt_values),
    }


def semantic_seg_metrics(preds: torch.Tensor, targets: torch.Tensor, num_classes: int,
                         ignore_index: int | None = None) -> dict:
    """Standard mean-IoU / Dice for semantic segmentation from per-pixel class maps.

    ``preds`` and ``targets`` are integer class-id tensors of matching shape (any shape;
    both are flattened). Per class: IoU = ``|P∩G| / |P∪G|`` and Dice = ``2|P∩G| / (|P|+|G|)``.
    ``mIoU`` / ``dice`` average only over classes present in preds or targets, a class absent
    from both has an undefined ratio (reported ``None`` per-class, excluded from the mean), the
    standard convention. ``pixel_acc`` is the fraction of correctly labeled pixels. Pixels equal
    to ``ignore_index`` in the GT are dropped before scoring.
    """
    pred = preds.detach().cpu().reshape(-1).long()
    gt = targets.detach().cpu().reshape(-1).long()
    if ignore_index is not None:
        keep = gt != ignore_index
        pred, gt = pred[keep], gt[keep]
    per_class_iou: dict[int, float | None] = {}
    per_class_dice: dict[int, float | None] = {}
    ious, dices = [], []
    for c in range(num_classes):
        if c == ignore_index:
            continue
        p = pred == c
        g = gt == c
        inter = int((p & g).sum())
        union = int((p | g).sum())
        denom = int(p.sum()) + int(g.sum())
        if union == 0:  # class absent from both, ratio undefined, exclude from the mean
            per_class_iou[c] = None
            per_class_dice[c] = None
            continue
        iou = inter / union
        dice = 2 * inter / denom if denom > 0 else 0.0
        per_class_iou[c] = iou
        per_class_dice[c] = dice
        ious.append(iou)
        dices.append(dice)
    pixel_acc = float((pred == gt).float().mean()) if gt.numel() else 0.0
    return {
        "mIoU": sum(ious) / len(ious) if ious else 0.0,
        "dice": sum(dices) / len(dices) if dices else 0.0,
        "pixel_acc": pixel_acc,
        "per_class_iou": per_class_iou,
        "per_class_dice": per_class_dice,
    }


# ====================================================================
# Task-agnostic evaluate(), loss pass + prediction pass
# ====================================================================

def detection_metrics(per_image: list[dict], *, trait: TraitEntry | None, conf_threshold: float,
                      iou_threshold: float, by_mask: bool) -> dict:
    """A detector's metrics over ``per_image``, every number from the one matcher: ``tp``/``fp``/
    ``fn``/``precision``/``recall``/``f1`` at ``conf_threshold`` (:func:`governing_counts`) under
    the criterion that governs (:func:`resolve_match_criterion`, ``trait``'s own when given, by
    mask when ``by_mask``), recorded under ``governing_criterion``; ``map50`` and ``map``, the
    average precision (:func:`average_precision`) at the IoU 0.5 convention and over
    :data:`AP_IOU_THRESHOLDS`. Under a trait's criterion the IoU convention's own counts ride
    beside them as ``iou_precision``/``iou_recall``/``iou_f1`` and ``map50_role`` labels the
    convention's numbers comparability only. ``matchings`` holds each image's governing
    matching (:func:`matched`), the one the counts sum; every matching is made once per distinct
    criterion and detection set."""
    criterion = resolve_match_criterion(trait, per_image, iou_threshold=iou_threshold,
                                        by_mask=by_mask)
    held: dict = {}
    governing = matched(per_image, criterion, conf=conf_threshold, held=held)
    counts = counted(governing, criterion)
    ap = {t: average_precision(per_image, iou_criterion(t, by_mask=by_mask), held=held)
          for t in AP_IOU_THRESHOLDS}
    out: dict = {**{k: counts[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")},
                 "governing_criterion": criterion, "map50": round(ap[0.5], 6),
                 "map": round(float(np.mean(list(ap.values()))), 6), "matchings": governing}
    if trait is not None:
        convention = counted(matched(per_image, iou_criterion(iou_threshold, by_mask=by_mask),
                                     conf=conf_threshold, held=held), criterion)
        out.update({"iou_precision": convention["precision"], "iou_recall": convention["recall"],
                    "iou_f1": convention["f1"], "map50_role": "comparability_only"})
    return out


@torch.no_grad()
def evaluate(
    model, loader, device, task: str, *, dims: Mapping[str, Any],
    conf_threshold: float | None, iou_threshold: float = 0.5,
    score_weights: dict | None = None, trait: TraitEntry | None = None,
) -> dict:
    """Compute per-task validation/test metrics. Returns bare metric keys. ``conf_threshold`` is
    the confidence a detector's boxes are counted at, read from the run's validated
    ``evaluation.conf_threshold`` or the pass's execution record; ``None`` for any other head.

    ``dims`` is what the model was built at (:func:`~tcip_mcp.pipelines.model_build.model_dims`);
    a class or rank count is read from it, never off the half being scored. A detector's metrics
    are :func:`detection_metrics` (by mask for instance segmentation) beside the composite
    ``objective``; one whose dims carry ``attributes`` also reports ``attribute_agreement``: per
    attribute name, the matched pairs' count and :func:`classification_metrics` over them
    (:func:`attribute_pairs`, under the governing criterion at ``conf_threshold``). ``trait`` is
    the trait's confirmed entry whose criterion governs the count; absent, the IoU convention at
    ``iou_threshold`` governs.
    """
    from tcip_mcp.pipelines.model_contract import DETECTION_TASKS

    is_detection = task in DETECTION_TASKS
    is_instance_seg = task == "instance_seg"

    model.eval()
    total_loss = 0.0
    n_loss = 0
    per_image: list[dict] = []
    cls_p, cls_g, ord_p, ord_g, reg_p, reg_g = [], [], [], [], [], []
    seg_p, seg_g = [], []
    detector = getattr(model, "detector", None)

    for batch in loader:
        if is_detection:
            images, targets = batch
            images = [img.to(device) for img in images]
            targets = [
                {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
                for t in targets
            ]
            # Loss pass over the full batch, all-negative (empty-box) images contribute their
            # background/objectness loss, so val_loss penalizes false positives on empty frames and
            # matches the train loop's distribution. BN stays eval via the train()+BN.eval() trick.
            # The heads get the train loop's own targets; scoring below reads every row.
            from tcip_mcp.pipelines.data.datasets import instance_targets

            head_targets = instance_targets(targets)
            if detector is not None:
                detector.train()
                for m in detector.modules():
                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                        m.eval()
                ld = detector(images, head_targets)
                total_loss += float(sum(ld.values()).item())
                n_loss += 1
            else:
                model.training = True
                for head in getattr(model, "heads", []):
                    head.training = True
                ld = model(images, head_targets)
                # sum() over an untyped model's loss dict resolves, by mypy's overload matching on
                # Any, to its int overload rather than the runtime Tensor the values actually are.
                total_loss += (
                    float(cast(Any, sum(ld.values())).item())
                    if isinstance(ld, dict) else float(ld)
                )
                n_loss += 1
            # Prediction pass.
            model.eval()
            outputs = model(images)
            for img, t, out in zip(images, targets, outputs):
                h, w = int(img.shape[-2]), int(img.shape[-1])
                per_image.append(records_from_detector(
                    t, out, width=w, height=h, include_masks=is_instance_seg))
        else:
            images, targets = batch
            images = images.to(device)
            targets = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in targets.items()
            }
            # Loss pass (BN stays in eval via the top-level training flag trick).
            model.training = True
            ld = model(images, targets)
            # sum() over an untyped model's loss dict resolves, by mypy's overload matching on
            # Any, to its int overload rather than the runtime Tensor the values actually are.
            total_loss += (
                float(cast(Any, sum(ld.values())).item())
                if isinstance(ld, dict) else float(ld)
            )
            n_loss += 1
            # Prediction pass.
            model.eval()
            out = model(images)
            if task == "classification" and "head0_labels" in out:
                cls_p.append(out["head0_labels"].detach().cpu())
                cls_g.append(targets["labels"].detach().cpu())
            elif task == "ordinal" and "head0_ranks" in out:
                ord_p.append(out["head0_ranks"].detach().cpu())
                ord_g.append(targets["ranks"].detach().cpu())
            elif task == "regression" and "head0_values" in out:
                reg_p.append(out["head0_values"].detach().cpu())
                reg_g.append(targets["values"].detach().cpu())
            elif task == "semantic_seg" and "head0_masks" in out:
                pm = out["head0_masks"].detach().cpu()
                gm = targets["masks"].detach().cpu()
                # decode() argmaxes feature-resolution logits; upsample the label map (nearest)
                # to the GT frame so metrics compare per-pixel at the annotation resolution.
                if pm.shape[-2:] != gm.shape[-2:]:
                    pm = torch.nn.functional.interpolate(
                        pm.unsqueeze(1).float(), size=gm.shape[-2:], mode="nearest"
                    ).squeeze(1).long()
                seg_p.append(pm.reshape(-1))
                seg_g.append(gm.reshape(-1))

    model.eval()
    loss = total_loss / max(n_loss, 1)
    result: dict = stored_number("loss", _rounded(loss))

    if is_detection:
        conf = cast(float, conf_threshold)
        m = detection_metrics(per_image, trait=trait, conf_threshold=conf,
                              iou_threshold=iou_threshold, by_mask=is_instance_seg)
        result.update({k: v for k, v in m.items() if k not in ("tp", "fp", "fn", "matchings")})
        criterion = m["governing_criterion"]
        result.update(stored_number(
            "objective",
            _rounded(compute_composite_objective(loss, m["f1"], m["map50"], score_weights)),
        ))
        if dims.get("attributes"):
            agreement = {}
            for column, attribute in enumerate(dims["attributes"]):
                pairs = attribute_pairs(per_image, criterion, conf=conf, column=column)
                agreement[attribute.name] = {"pairs": len(pairs), **_reported_metrics(
                    classification_metrics(torch.tensor([p for _i, _t, p in pairs]),
                                           torch.tensor([t for _i, t, _p in pairs]),
                                           len(attribute.values)))}
            result["attribute_agreement"] = agreement
    elif task == "classification" and cls_p:
        result.update(_reported_metrics(classification_metrics(
            torch.cat(cls_p), torch.cat(cls_g), dims["num_classes"])))
    elif task == "ordinal" and ord_p:
        result.update(_reported_metrics(ordinal_metrics(
            torch.cat(ord_p), torch.cat(ord_g), dims["num_ranks"])))
    elif task == "regression" and reg_p:
        result.update(_reported_metrics(regression_metrics(torch.cat(reg_p), torch.cat(reg_g))))
    elif task == "semantic_seg" and seg_p:
        result.update(_reported_metrics(semantic_seg_metrics(
            torch.cat(seg_p), torch.cat(seg_g), dims["num_classes"])))

    return result
