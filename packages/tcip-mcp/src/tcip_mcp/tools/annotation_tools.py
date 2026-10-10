"""Annotation tools: load, save and score name-based annotations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import tcip_store

from tcip_annotation import Annotation
from tcip_annotation.json_io import (
    UnreadableLabelDocumentError, client_annotation, read_document_versioned, read_predictions,
)

from tcip_annotation.matching import REVIEW_CONF_FLOOR, Matching

from tcip_mcp.buckets import Bucket, read_bucket, source_root
from tcip_mcp.dataset_layout import label_key_of
from tcip_mcp.pipelines.image_utils import image_path_dimensions
from tcip_mcp.registry_paths import located
from tcip_mcp.server import tool
from tcip_mcp.workspace import BoundProject

if TYPE_CHECKING:
    from tcip_mcp.traits import TraitEntry


def _summary(annotations: list[Annotation]) -> dict:
    """A document's annotations as a read returns them: their count, subjects and records."""
    return {"count": len(annotations), "subjects": sorted({a.subject for a in annotations}),
            "annotations": [client_annotation(a) for a in annotations]}


def read_annotations(image_path: str, bucket: str | None = None) -> dict:
    """Load the ground-truth labels for a single image and, given a bucket published under the
    image's dataset root, its predictions.

    Both are the name-based per-image schema, one document per image, all subjects. A present
    document this schema cannot read returns an ``error``.

    Args:
        image_path: Absolute path to the image file (:func:`~tcip_mcp.registry_paths.located`
            with no project).
        bucket: The name of the published bucket whose document for this image to read, if any.
    """
    try:
        img = located(image_path, None)
    except ValueError as exc:
        return {"error": str(exc)}
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    w, h = image_path_dimensions(image_path)
    result: dict = {"image": image_path, "width": w, "height": h}
    try:
        key = label_key_of(image_path)
        labels, version = read_document_versioned(key)
        if version != tcip_store.Version.ABSENT:
            result["labels"] = _summary(labels.annotations)
        document = read_bucket(key.root, bucket).document_key(img.stem) if bucket else None
        if document is not None:
            result["predictions"] = _summary(read_predictions(document).annotations)
    except (UnreadableLabelDocumentError, ValueError) as exc:
        return {"error": str(exc)}
    return result


@tool()
def save_annotations(
    bound: BoundProject,
    image_path: str,
    annotations: list[dict],
) -> dict:
    """Write an image's annotations as its single per-image label document (all subjects, one
    document), keyed by the image's dataset root, capture and stem
    (:func:`~tcip_mcp.dataset_layout.label_key`), through the one label save
    (:func:`~tcip_mcp.dataset_layout.save_label_document`) as the author ``save_annotations``:
    each annotation takes the provenance of one stored record of the same content not yet
    claimed by another (:func:`~tcip_annotation.json_io.stamped`), and every other one is this
    tool's own. Each annotation is a dict carrying a ``subject`` (required, refused when
    absent), an
    optional geometry (``bbox`` = [x1,y1,x2,y2], ``points`` = [[x,y],...] for a single-ring
    polygon contour, ``rings`` = [[[x,y],...], ...] for a multi-ring polygon, whose ring vertices
    may be ``{x,y}`` dicts or ``[x,y]`` pairs, ``point`` = [x,y] for a single prompt/keypoint
    location, or none of them for an image-level label), and optional ``attributes`` (attribute
    name -> value name).

    Args:
        image_path: The image file (:func:`~tcip_mcp.registry_paths.located` against the
            project).
        annotations: List of ``{subject, bbox?/points?/rings?/point?, attributes?}`` dicts (pixel
            coords); an empty list writes an empty document.

    Returns ``{capture, written, count}``, ``written`` the one-element list of the stem
    written, or ``{error}`` with nothing written for a ``ValueError`` the save raises (a
    payload that does not parse; a removed annotation an open flag sits on, since a tool names
    no person and the person resolves the flag first). A stored document that does not decode
    raises ``UnreadableLabelDocumentError`` through this door.
    """
    from tcip_mcp.dataset_layout import save_label_document
    from tcip_mcp.web_client import PANEL_EVENT_LABELS_WRITTEN, post_panel_event

    image_path = str(located(image_path, bound.root))
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    w, h = image_path_dimensions(image_path)
    try:
        key = label_key_of(image_path)
        save_label_document(bound.root, key, annotations, width=w, height=h,
                            author="save_annotations", actor=None)  # the answer is for the editor
    except ValueError as exc:
        return {"error": str(exc)}

    post_panel_event(bound, "annotate", PANEL_EVENT_LABELS_WRITTEN,
                     {"image_path": image_path})
    capture, stem = key.parts
    return {"capture": capture, "written": [stem], "count": len(annotations)}


def _counts(m: Matching) -> dict:
    """One matching's TP, FP and FN."""
    return dict(zip(("tp", "fp", "fn"), m.counts))


def _totals(metrics: dict) -> dict:
    """The TP/FP/FN a scoring's ``metrics`` report with the precision, recall and F1 they give."""
    return {k: metrics[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")}


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
    """``bucket``'s predictions for one image matched against its ground truth
    (:func:`~tcip_mcp.pipelines.training.evaluation.score_bucket`), with ``matches``, the
    ``[gt_idx, pred_idx]`` pairs the one matcher draws per subject over the predictions at or above
    ``conf_threshold`` under the scoring's criterion, ordered by prediction index, the same
    matching its TP/FP/FN count. An image the bucket's record names no document for refuses
    (``ValueError``): it was not predicted, so there is nothing to score."""
    from tcip_mcp.pipelines.image_utils import resolve_image_path
    from tcip_mcp.pipelines.training.evaluation import score_bucket

    (one,), metrics = score_bucket([resolve_image_path(image)], bucket,
                                   iou_threshold=iou_threshold, conf_threshold=conf_threshold,
                                   trait=trait)
    if one.preds is None:
        raise ValueError(f"bucket {bucket.name!r} names no document for {image.name}: it was "
                         "not predicted, so there is nothing to score.")
    out = {"image": str(image), **_rounded(_totals(metrics)),
           "map50": round(metrics["map50"], 4),
           "governing_criterion": metrics["governing_criterion"],
           "iou_threshold": iou_threshold, "conf_threshold": conf_threshold,
           "matches": [list(pair) for pair in one.matching.pairs]}
    if detail:
        img_w, img_h = one.header.display_frame
        out.update(img_w=img_w, img_h=img_h,
                   detections=_detection_breakdown(one.matching, one.gt, one.preds))
    return out


def _evaluate_folder(images_dir: str, bucket: Bucket, iou_threshold: float,
                     conf_threshold: float, trait: TraitEntry | None) -> dict:
    """Aggregate detection metrics across the logical images of one images directory against
    ``bucket``'s documents for them
    (:func:`~tcip_mcp.pipelines.training.evaluation.score_bucket`), the images it names no
    document for listed under ``not_predicted`` and left out of every count.

    A ``.bandgroup``-grouped capture scores as one logical image, and a nested folder is not
    descended. A directory holding two raw images under one case-folded stem is refused by
    ``list_logical_images``.
    """
    from tcip_mcp.pipelines.image_utils import list_logical_images, source_path_of
    from tcip_mcp.pipelines.training.evaluation import score_bucket

    images = sorted(list_logical_images(images_dir).values(), key=source_path_of)
    scored, m = score_bucket(images, bucket, iou_threshold=iou_threshold,
                             conf_threshold=conf_threshold, trait=trait)
    predicted = [one for one in scored if one.preds is not None]
    counts = {"tp": "total_tp", "fp": "total_fp", "fn": "total_fn"}
    totals = {counts.get(k, k): v for k, v in _rounded(_totals(m)).items()}
    return {"path": images_dir, "image_count": len(predicted),
            "not_predicted": [source_path_of(one.header.source) for one in scored
                              if one.preds is None],
            "map": round(m["map"], 4), "map50": round(m["map50"], 4), **totals,
            "governing_criterion": m["governing_criterion"],
            "per_image": [{"image": Path(source_path_of(one.header.source)).name,
                           **_counts(one.matching)}
                          for one in predicted]}


def score_predictions(
    path: str,
    bucket: str,
    iou_threshold: float = 0.5,
    conf_threshold: float = REVIEW_CONF_FLOOR,
    detail: bool = False,
    trait: TraitEntry | None = None,
    *,
    project: Path | None = None,
) -> dict:
    """Score a published bucket's predictions against the images' own label documents through the
    platform's one matcher.

    Dispatches on the input: a single image file returns per-box ``matches`` (plus an optional
    per-detection ``detections`` breakdown with ``img_w`` / ``img_h`` when ``detail=True``); an
    images directory returns aggregate metrics plus ``per_image`` TP/FP/FN. Both regimes are
    :func:`~tcip_mcp.pipelines.training.evaluation.score_bucket`'s.

    A prediction carries its object class in ``subject``, so this scores the localization of the
    object class, never an attribute head's call.

    Args:
        path: An image file (single-image match) or an images directory (aggregate)
            (:func:`~tcip_mcp.registry_paths.located` against ``project``; with none held, a
            relative path refuses).
        bucket: The name of the bucket, published under ``path``'s dataset root, whose documents
            are scored.
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
    try:
        p = located(path, project)
    except ValueError as exc:
        return {"error": str(exc)}
    if not p.exists():
        return {"error": f"Path not found: {path}"}
    try:
        found = read_bucket(source_root([p]), bucket)
        if p.is_file():
            return _evaluate_image(p, found, iou_threshold, conf_threshold, detail, trait)
        return _evaluate_folder(str(p), found, iou_threshold, conf_threshold, trait)
    except (UnreadableLabelDocumentError, ValueError) as exc:
        return {"error": str(exc)}


@tool()
def write_subject_registry(
    project: Path, dataset_root: str, subjects: dict,
    allow_removals: bool = False, allow_type_changes: bool = False,
) -> dict:
    """Write the dataset's subject registry: ``subjects``, each subject's ``description`` and
    ``attributes`` (each ``categorical`` or ``ordinal`` with ordered ``values``), validated
    (:func:`subject_registry.registry_from_request`) and written
    (:func:`subject_registry.replace_registry`), whose refusals answer as errors: an empty
    registry; a dropped subject, attribute or value, or a stored registry that will not decode,
    without ``allow_removals``; an attribute's type changed without ``allow_type_changes``.

    Args:
        dataset_root: Dataset root; the registry is written to ``<dataset_root>/subjects.json``.
        subjects: Nested ``{subject: {description?, attributes?}}`` dict.
        allow_removals: State a dropped name, or a stored registry that will not decode, as a
            deliberate removal/repair.
        allow_type_changes: State a same-values attribute type flip (categorical to ordinal or
            back) as deliberate.
    """
    from tcip_store import VersionConflictError

    from tcip_mcp import subject_registry
    from tcip_mcp.audit import AuditEntryNotWrittenError

    try:
        registry = subject_registry.registry_from_request(subjects)
    except subject_registry.RegistryError as exc:
        return {"error": f"invalid registry: {exc}"}

    try:
        result = subject_registry.replace_registry(
            located(dataset_root, project), registry, expect=None, allow_removals=allow_removals,
            allow_type_changes=allow_type_changes, actor=None)
    except (subject_registry.RegistryError, VersionConflictError, AuditEntryNotWrittenError) as exc:
        return {"error": str(exc)}
    return {"subjects_path": result["subjects_path"],
            "subjects": [s.name for s in registry.subjects]}
