"""Annotation tools: load, save and score name-based annotations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tcip_annotation import Annotation
from tcip_annotation.json_io import UnreadableLabelDocument, client_annotation
from tcip_annotation.json_io import read_annotations as read_labels
from tcip_annotation.json_io import read_predictions

from tcip_annotation.matching import REVIEW_CONF_FLOOR, Matching

from tcip_mcp.buckets import Bucket, read_bucket
from tcip_mcp.dataset_layout import annotation_path_for_image, find_gt_label
from tcip_mcp.pipelines.image_utils import image_path_dimensions
from tcip_mcp.server import tool

if TYPE_CHECKING:
    from tcip_mcp.traits import TraitEntry


def read_annotations(image_path: str, predictions_dir: str | None = None) -> dict:
    """Load the ground-truth labels for a single image and, given a bucket, its predictions.

    Both are the name-based per-image schema, one file per image, all subjects. A present document
    this schema cannot read returns an ``error``.

    Args:
        image_path: Absolute path to the image file.
        predictions_dir: The published bucket whose document for this image to read, if any.
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    w, h = image_path_dimensions(image_path)
    result: dict = {"image": image_path, "width": w, "height": h}

    gt_path = find_gt_label(image_path)
    if gt_path is not None:
        try:
            anns = read_labels(str(gt_path))
        except UnreadableLabelDocument as exc:
            return {"error": str(exc)}
        result["labels"] = {
            "path": str(gt_path), "count": len(anns),
            "subjects": sorted({a.subject for a in anns}),
            "annotations": [client_annotation(a) for a in anns],
        }

    pred_path = read_bucket(predictions_dir).document(image_path) if predictions_dir else None
    if pred_path is not None:
        try:
            preds = read_predictions(str(pred_path))
        except UnreadableLabelDocument as exc:
            return {"error": str(exc)}
        result["predictions"] = {
            "path": str(pred_path), "count": len(preds),
            "subjects": sorted({a.subject for a in preds}),
            "annotations": [client_annotation(a) for a in preds],
        }

    return result


@tool()
def save_annotations(
    project: Path,
    workspace: Path,
    image_path: str,
    annotations: list[dict],
    date: str | None = None,
    path: str | None = None,
    created_by: str | None = None,
) -> dict:
    """Write an image's annotations to its single per-image label file (all subjects, one file).

    The label goes to ``<dataset_root>/annotations/<date>/<stem>.json`` (see
    :mod:`tcip_mcp.dataset_layout`); ``date`` is derived from the image path when not given. Pass
    ``path`` to write to an explicit location instead. Each annotation is a dict carrying a
    ``subject`` (required, refused when absent), an optional geometry (``bbox`` = [x1,y1,x2,y2],
    ``points`` = [[x,y],...] for a single-ring polygon contour, ``rings`` = [[[x,y],...], ...] for
    a multi-ring polygon, whose ring vertices may be ``{x,y}`` dicts or ``[x,y]`` pairs, ``point``
    = [x,y] for a single prompt/keypoint location, or none of them for an image-level label), and
    optional ``attributes`` (attribute name -> value name).

    Args:
        image_path: Absolute path to the image file.
        annotations: List of ``{subject, bbox?/points?/rings?/point?, attributes?}`` dicts (pixel
            coords); an empty list writes an empty document.
        date: Capture date; derived from the image path when omitted.
        path: Explicit label path (overrides the canonical location).
        created_by: Producer stamped on each written annotation. Omit to leave provenance unset.
    """
    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_mcp.dataset_layout import save_label_document
    from tcip_mcp.web_client import PANEL_EVENT_LABELS_WRITTEN, post_panel_event

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    w, h = image_path_dimensions(image_path)
    try:
        out_path = Path(path) if path else annotation_path_for_image(image_path, date=date)
        save_label_document(project, image_path, out_path, annotations, width=w, height=h,
                            author=created_by)
    except (ValueError, AuditEntryNotWritten) as exc:
        return {"error": str(exc)}

    post_panel_event(project, workspace, "annotate", PANEL_EVENT_LABELS_WRITTEN,
                     {"image_path": image_path, "stem": img.stem, "written": [str(out_path)]})
    return {"written": [str(out_path)], "count": len(annotations)}


def _scored(images: list[Path], bucket: Bucket, *, iou_threshold: float, conf_threshold: float,
            trait: TraitEntry | None) -> dict:
    """``bucket``'s documents for ``images`` scored against their ground truth: each predicted
    image's annotations read once, one subject-to-id map across all of them, one COCO record per
    image, the comparability metrics over those records, and each image's matching
    (:func:`~tcip_annotation.matching.pair_proposals`, its TP/FP/FN) under the one criterion
    that governs them (``governing_criterion``): the trait's own when ``trait`` is given, the IoU
    convention at ``iou_threshold`` otherwise. An image the bucket's record names no document for
    is not predicted, so it is left out of every count and listed under ``not_predicted``.
    Returns ``{"read", "iou_type", "metrics", "matchings", "governing_criterion",
    "not_predicted"}``, ``read`` each scored image's ``(image, gt, preds, width, height)``."""
    from tcip_annotation.matching import pair_proposals
    from tcip_annotation.state import polygonal

    from tcip_mcp.pipelines.training.evaluation import (
        coco_detection_metrics, records_from_annotation, resolve_match_criterion,
        subject_category_ids,
    )

    documents = {img: bucket.document(img) for img in images}
    read = []
    for img, document in documents.items():
        if document is not None:
            gt_path = find_gt_label(str(img))
            read.append((img, read_labels(str(gt_path)) if gt_path else [],
                         read_predictions(str(document)), *image_path_dimensions(img)))
    annotations = [a for _img, gt, preds, _w, _h in read for a in (*gt, *preds)]
    segm = any(polygonal(a.geometry) for a in annotations)
    name_id = subject_category_ids(annotations)
    records = [records_from_annotation(gt, preds, width=w, height=h, force_segm=segm,
                                       name_id=name_id)[1]
               for _img, gt, preds, w, h in read]
    iou_type = "segm" if segm else "bbox"
    metrics = coco_detection_metrics(records, iou_type=iou_type, iou_threshold=iou_threshold,
                                     conf_threshold=conf_threshold)
    criterion = (resolve_match_criterion(trait, records) if trait is not None
                 else resolve_match_criterion(None, [], iou_threshold=iou_threshold))
    matchings = [pair_proposals(gt, preds, criterion, conf_threshold=conf_threshold)
                 for _img, gt, preds, _w, _h in read]
    return {"read": read, "iou_type": iou_type, "metrics": metrics, "matchings": matchings,
            "governing_criterion": criterion,
            "not_predicted": [str(img) for img, document in documents.items() if document is None]}


def _counts(m: Matching) -> dict:
    """One matching's TP, FP and FN."""
    return {"tp": len(m.pairs), "fp": len(m.unpaired), "fn": len(m.missed)}


def _totals(scored: dict) -> dict:
    """The TP/FP/FN a scoring reports, summed over its images' matchings, with the precision,
    recall and F1 they give."""
    from tcip_mcp.pipelines.training.evaluation import precision_recall_f1

    per_image = [_counts(m) for m in scored["matchings"]]
    tp, fp, fn = (sum(c[k] for c in per_image) for k in ("tp", "fp", "fn"))
    return {"tp": tp, "fp": fp, "fn": fn, **precision_recall_f1(tp, fp, fn)}


def _detection_breakdown(m: Matching, gt: list[Annotation],
                         preds: list[Annotation]) -> list[dict]:
    """Per-detection records from the one matcher's matching ``m``: each projected as every read
    door projects one (:func:`~tcip_annotation.json_io.client_annotation`), tagged ``tp`` for a
    pair, ``fp`` for an unpaired detection and ``fn`` for a missed object, with its indices."""
    return (
        [{**client_annotation(preds[p]), "tag": "tp", "gt_idx": g, "pred_idx": p}
         for g, p in m.pairs]
        + [{**client_annotation(preds[p]), "tag": "fp", "pred_idx": p} for p in m.unpaired]
        + [{**client_annotation(gt[g]), "tag": "fn", "gt_idx": g} for g in m.missed])


def _rounded(totals: dict) -> dict:
    """``totals`` with its precision, recall and F1 rounded to four places."""
    return {k: round(v, 4) if k in ("precision", "recall", "f1") else v
            for k, v in totals.items()}


def _evaluate_image(image: Path, bucket: Bucket, iou_threshold: float, conf_threshold: float,
                    detail: bool, trait: TraitEntry | None) -> dict:
    """``bucket``'s predictions for one image matched against its ground truth (:func:`_scored`),
    with ``matches``, the ``[gt_idx, pred_idx]`` pairs the one matcher
    (:func:`~tcip_annotation.matching.pair_proposals`) draws per subject over the predictions at
    or above ``conf_threshold`` under the scoring's criterion, ordered by prediction index, the
    same matching its TP/FP/FN count. An image the bucket's record names no document for refuses
    (``ValueError``): it was not predicted, so there is nothing to score."""
    scored = _scored([image], bucket, iou_threshold=iou_threshold, conf_threshold=conf_threshold,
                     trait=trait)
    if scored["not_predicted"]:
        raise ValueError(f"{bucket.path} names no document for {image.name}: it was not "
                         "predicted, so there is nothing to score.")
    _img, gt, preds, w, h = scored["read"][0]
    (m,) = scored["matchings"]
    out = {"image": str(image), **_rounded(_totals(scored)),
           "map50": round(scored["metrics"]["map50"], 4), "iou_type": scored["iou_type"],
           "iou_threshold": iou_threshold, "conf_threshold": conf_threshold,
           "matches": [list(pair) for pair in m.pairs]}
    if detail:
        out.update(img_w=w, img_h=h, detections=_detection_breakdown(m, gt, preds))
    return out


def _evaluate_folder(images_dir: str, bucket: Bucket, iou_threshold: float,
                     conf_threshold: float, trait: TraitEntry | None) -> dict:
    """Aggregate detection metrics across the logical images of one images directory against
    ``bucket``'s documents for them (:func:`_scored`).

    A ``.bandgroup``-grouped capture scores as one logical image, and a nested folder is not
    descended. A directory holding two raw images under one case-folded stem is refused by
    ``list_logical_images``.
    """
    from tcip_mcp.pipelines.image_utils import BandGroupRef, list_logical_images

    images = sorted(src.manifest_path if isinstance(src, BandGroupRef) else src
                    for src in list_logical_images(images_dir).values())
    scored = _scored(images, bucket, iou_threshold=iou_threshold,
                     conf_threshold=conf_threshold, trait=trait)
    counts = {"tp": "total_tp", "fp": "total_fp", "fn": "total_fn"}
    totals = {counts.get(k, k): v for k, v in _rounded(_totals(scored)).items()}
    m = scored["metrics"]
    return {"path": images_dir, "image_count": len(scored["read"]),
            "not_predicted": scored["not_predicted"], "map": round(m["map"], 4),
            "map50": round(m["map50"], 4), **totals, "iou_type": scored["iou_type"],
            "per_image": [{"image": img.name, **_counts(match)}
                          for (img, *_rest), match in zip(scored["read"], scored["matchings"])]}


def score_predictions(
    path: str,
    predictions_dir: str,
    iou_threshold: float = 0.5,
    conf_threshold: float = REVIEW_CONF_FLOOR,
    detail: bool = False,
    trait: TraitEntry | None = None,
) -> dict:
    """Score a published bucket's predictions against on-disk ground truth (COCOeval).

    Dispatches on the input: a single image file returns per-box ``matches`` (plus an optional
    per-detection ``detections`` breakdown with ``img_w`` / ``img_h`` when ``detail=True``); an
    images directory returns aggregate metrics plus ``per_image`` TP/FP/FN. Both regimes share
    ``coco_detection_metrics``.

    A classified bucket's predictions carry the object class in ``subject``, so this scores the
    localization of the object class, never the classifier's own call.

    Args:
        path: Absolute path to an image file (single-image match) or an images directory
            (aggregate).
        predictions_dir: The bucket whose documents are scored.
        iou_threshold: IoU threshold for a positive match (the AP@0.5 comparability convention).
        conf_threshold: Minimum confidence to consider a prediction.
        detail: Single-image only, also return the per-detection ``detections`` breakdown: each
            entry the annotation it names as ``client_annotation`` projects it (corner ``bbox``,
            ``rings`` or ``point``, ``subject``, ``attributes``, ``iscrowd``, ``score`` and the
            provenance it holds) beside its ``tag`` and indices.
        trait: The trait's confirmed entry; when set, its own localization criterion governs the
            reported TP/FP/FN count; map50 stays a labeled comparability metric. Absent -> the IoU
            convention governs.
    """
    p = Path(path)
    try:
        bucket = read_bucket(predictions_dir)
        if p.is_file():
            return _evaluate_image(p, bucket, iou_threshold, conf_threshold, detail, trait)
        if p.is_dir():
            return _evaluate_folder(path, bucket, iou_threshold, conf_threshold, trait)
    except (UnreadableLabelDocument, ValueError) as exc:
        return {"error": str(exc)}
    return {"error": f"Path not found: {path}"}


@tool()
def write_subject_registry(
    project: Path, dataset_root: str, subjects: dict,
    allow_removals: bool = False, allow_type_changes: bool = False,
) -> dict:
    """Author the dataset's nested subject registry, a thin wrapper over ``subject_registry``.

    ``subjects`` is the nested registry mapping the expert defines, subjects to their
    ``description`` / provenance and zero or more ``attributes`` (each ``categorical`` |
    ``ordinal`` with ordered ``values``). It is validated through
    :func:`subject_registry.registry_from_dict` (a malformed shape refuses) and written to
    ``<dataset_root>/subjects.json`` via :func:`subject_registry.replace_registry`, which guards
    only the store's own window between its read and its put and refuses an empty registry. No
    numeric class ids, no colors, no id enumeration.

    A write that would drop a subject, attribute or attribute value the stored registry declares is
    refused unless ``allow_removals`` is set; the same flag also allows replacing a stored registry
    whose bytes will not decode. A write that keeps an attribute's name and values but changes its
    ``type`` (categorical to ordinal or back) is refused independently of ``allow_removals`` unless
    ``allow_type_changes`` is set.

    Args:
        dataset_root: Dataset root; the registry is written to ``<dataset_root>/subjects.json``.
        subjects: Nested ``{subject: {description?, defined_by?, defined_at?, attributes?}}`` dict.
        allow_removals: State a dropped name, or a stored registry that will not decode, as a
            deliberate removal/repair.
        allow_type_changes: State a same-values attribute type flip (categorical to ordinal or
            back) as deliberate.
    """
    from tcip_store import VersionConflict

    from tcip_mcp import subject_registry
    from tcip_mcp.audit import AuditEntryNotWritten

    try:
        registry = subject_registry.registry_from_dict(subjects)
    except subject_registry.RegistryError as exc:
        return {"error": f"invalid registry: {exc}"}

    try:
        result = subject_registry.replace_registry(
            dataset_root, registry, expect=None, allow_removals=allow_removals,
            allow_type_changes=allow_type_changes)
    except (subject_registry.RegistryError, VersionConflict, AuditEntryNotWritten) as exc:
        return {"error": str(exc)}
    return {"subjects_path": result["subjects_path"],
            "subjects": [s.name for s in registry.subjects]}
