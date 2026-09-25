"""Annotation tools: load, save and score name-based annotations."""

from __future__ import annotations

from pathlib import Path

from tcip_annotation import Annotation, compute_matches
from tcip_annotation.json_io import (
    UnreadableLabelDocument, annotation_from_payload, client_annotation, write_annotations,
)
from tcip_annotation.json_io import read_annotations as read_labels
from tcip_annotation.json_io import read_predictions

from tcip_mcp.dataset_layout import (
    annotation_path_for_image,
    find_gt_label,
    find_prediction,
    image_root,
)
from tcip_mcp.pipelines.image_utils import image_dimensions, resolve_image_source
from tcip_mcp.pipelines.resolution import DEFAULT_CONF
from tcip_mcp.server import mcp
from tcip_mcp.audit import audited


def _dims_for(image_path: str) -> tuple[int, int]:
    """``(width, height)`` for ``image_path``, channel-aware, through ``resolve_image_source``."""
    img = Path(image_path)
    source = resolve_image_source(img.parent, img.stem)
    return image_dimensions(source)



def read_annotations(image_path: str) -> dict:
    """Load the ground-truth labels and predictions for a single image.

    Both are the name-based per-image schema, one file per image, all subjects. A present document
    this schema cannot read returns an ``error``.

    Args:
        image_path: Absolute path to the image file.
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    w, h = _dims_for(image_path)
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

    pred_path = find_prediction(image_path)
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


@mcp.tool()
@audited(scope_arg="image_path")
def save_annotations(
    image_path: str,
    annotations: list[dict] | None = None,
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
            coords).
        date: Capture date; derived from the image path when omitted.
        path: Explicit label path (overrides the canonical location).
        created_by: Producer stamped on each written annotation. Omit to leave provenance unset.
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    anns_in = annotations or []
    if not anns_in:
        return {"error": "provide at least one annotation to save (each carrying a subject)"}

    from tcip_mcp.workspace import is_valid_name

    if path is None and date is not None and not is_valid_name(date):
        return {"error": f"date must be a single safe path segment (no separators/'..'), got {date!r}"}

    w, h = _dims_for(image_path)

    from datetime import datetime, timezone
    _now = datetime.now(timezone.utc).isoformat()

    typed = []
    for i, a in enumerate(anns_in):
        try:
            typed.append(annotation_from_payload(a, author=created_by, now=_now))
        except ValueError as exc:
            return {"error": f"annotation {i} {exc}"}

    out_path = Path(path) if path else annotation_path_for_image(image_path, date=date)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_annotations(str(out_path), typed, w, h, keep_empty=True)

    try:
        from tcip_mcp.web_client import PANEL_EVENT_LABELS_WRITTEN, post_panel_event

        post_panel_event("annotate", PANEL_EVENT_LABELS_WRITTEN,
                         {"image_path": image_path, "stem": img.stem, "written": [str(out_path)]})
    except Exception:
        pass

    return {"written": [str(out_path)], "count": len(typed)}


def _load_image_annotations(image_path: str, *, _checked_bucket_dirs: set | None = None):
    """Load GT + predictions for one image and build a COCO per-image record.

    Returns ``(iou_type, record, (gt, preds), width, height)`` where ``gt`` / ``preds`` are
    :class:`Annotation` lists; ``None`` if unreadable. When a prediction file is found, its
    bucket's own recorded scope is read; an undecodable stamp propagates the seam's own
    ``StoreError``, and a bare directory or any readable scope scores by ``subject``.
    ``_checked_bucket_dirs`` skips a directory already read this pass.
    """
    from tcip_mcp.pipelines.resolution import bucket_scope
    from tcip_mcp.pipelines.training.evaluation import records_from_annotation

    img = Path(image_path)
    if not img.is_file():
        return None
    w, h = _dims_for(image_path)
    gt: list[Annotation] = []
    preds: list[Annotation] = []

    gt_path = find_gt_label(image_path)
    if gt_path:
        gt = read_labels(str(gt_path))
    pred_path = find_prediction(image_path)
    if pred_path:
        bucket_dir = Path(pred_path).parent
        if _checked_bucket_dirs is None or bucket_dir not in _checked_bucket_dirs:
            bucket_scope(bucket_dir)
            if _checked_bucket_dirs is not None:
                _checked_bucket_dirs.add(bucket_dir)
        preds = read_predictions(str(pred_path))

    iou_type, record = records_from_annotation(gt, preds, width=w, height=h)
    return iou_type, record, (gt, preds), w, h


def _detection_breakdown(matches: dict, gt: list[Annotation], preds: list[Annotation]) -> list[dict]:
    """Per-detection TP/FP/FN records from a match result: the annotation each names, projected
    as every read door projects one (:func:`~tcip_annotation.json_io.client_annotation`, which
    carries its ``subject`` and ``score``), with its tag, IoU and indices."""
    return (
        [{**client_annotation(preds[m["pred_idx"]]), "tag": "tp", "iou": m["iou"],
          "gt_idx": m["gt_idx"], "pred_idx": m["pred_idx"]} for m in matches["tp"]]
        + [{**client_annotation(preds[m["pred_idx"]]), "tag": "fp", "pred_idx": m["pred_idx"]}
           for m in matches["fp"]]
        + [{**client_annotation(gt[m["gt_idx"]]), "tag": "fn", "gt_idx": m["gt_idx"]}
           for m in matches["fn"]])


def _apply_governing_criterion(out: dict, records: list, *, trait: str | None,
                               iou_threshold: float, conf_threshold: float) -> dict:
    """Override the human-facing TP/FP/FN + P/R/F1 with the trait's derived criterion.

    A count trait using center-match governs the review count that feeds the phenotype; AP@0.5
    stays as a labeled comparability metric. With no trait, ``out`` is returned unchanged.
    """
    from tcip_mcp.pipelines.training.evaluation import governing_counts, resolve_match_criterion

    criterion = resolve_match_criterion(trait, records, iou_threshold=iou_threshold)
    if criterion["kind"] != "center_match":
        return out
    gc = governing_counts(records, criterion, conf_threshold=conf_threshold)
    out.update({
        "iou_tp": out.get("tp"), "iou_fp": out.get("fp"), "iou_fn": out.get("fn"),
        "tp": gc["tp"], "fp": gc["fp"], "fn": gc["fn"],
        "precision": round(gc["precision"], 4), "recall": round(gc["recall"], 4),
        "f1": round(gc["f1"], 4),
        "governing_criterion": criterion, "map50_role": "comparability_only",
    })
    return out


def _evaluate_image(
    image_path: str,
    iou_threshold: float = 0.5,
    conf_threshold: float = DEFAULT_CONF,
    detail: bool = False,
    trait: str | None = None,
) -> dict:
    """Match predictions against ground truth for a single image (COCOeval).

    mAP / TP / FP / FN come from pycocotools; the ``matches`` block is a per-box overlay the agent
    can render for review (``compute_matches``). With a count ``trait`` the reported count is
    governed by the trait's derived criterion, map50 kept as comparability.
    """
    loaded = _load_image_annotations(image_path)
    if loaded is None:
        return {"error": f"Image not found: {image_path}"}
    iou_type, record, (gt, preds), w, h = loaded

    from tcip_mcp.pipelines.training.evaluation import coco_detection_metrics
    m = coco_detection_metrics([record], iou_type=iou_type,
                               iou_threshold=iou_threshold, conf_threshold=conf_threshold)
    matches = compute_matches(gt, preds, iou_threshold=iou_threshold, conf_threshold=conf_threshold)
    out = {
        "image": image_path,
        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
        "map50": round(m["map50"], 4),
        "iou_type": iou_type,
        "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold,
        "matches": matches,
    }
    out = _apply_governing_criterion(out, [record], trait=trait,
                                     iou_threshold=iou_threshold, conf_threshold=conf_threshold)
    if detail:
        out["img_w"] = w
        out["img_h"] = h
        out["detections"] = _detection_breakdown(matches, gt, preds)
    return out


def _evaluate_folder(
    folder_path: str,
    iou_threshold: float = 0.5,
    conf_threshold: float = DEFAULT_CONF,
    trait: str | None = None,
) -> dict:
    """Aggregate detection metrics across all images in a dataset.

    Scores the logical images directly under ``images_dir`` plus those in each of its direct bucket
    subdirectories (``images/<bucket>/``, the dataset layout), one level: a loose image beside a
    dated bucket still scores, a ``.bandgroup``-grouped capture scores as one logical image, and a
    folder nested inside a bucket is not descended. A bucket holding two raw images under one
    case-folded stem is refused by ``list_logical_images``.
    """
    from tcip_mcp.pipelines.image_utils import BandGroupRef, list_logical_images

    root = Path(folder_path)
    images_dir = image_root(root)
    if not images_dir.is_dir():
        images_dir = root

    def _logical_paths(d: Path) -> list[Path]:
        return [src.manifest_path if isinstance(src, BandGroupRef) else src
                for src in list_logical_images(d).values()]

    images = _logical_paths(images_dir)
    if images_dir.is_dir():
        for bucket in sorted(p for p in images_dir.iterdir() if p.is_dir()):
            images.extend(_logical_paths(bucket))
    images.sort()

    from tcip_mcp.pipelines.training.evaluation import (
        coco_detection_metrics, records_from_annotation, subject_category_ids,
    )

    collected = []  # (iou_type, record, (gt, preds), w, h, img)
    checked_bucket_dirs: set = set()
    for img in images:
        loaded = _load_image_annotations(str(img), _checked_bucket_dirs=checked_bucket_dirs)
        if loaded is None:
            continue
        iou_type, record, raw, w, h = loaded
        collected.append((iou_type, record, raw, w, h, img))

    any_segm = any(c[0] == "segm" for c in collected)
    dataset_iou_type = "segm" if any_segm else "bbox"
    # One subject->id map across the whole scored set: coco_detection_metrics accumulates every
    # per-image record into a single eval, so a subject must carry the same category id in every
    # image. Rebuild all records with it (the per-image records built by _load_image_annotations
    # used a per-image-local map, which would pool distinct subjects into one class across images).
    name_id = subject_category_ids(
        a for (_it, _rec, (gt, preds), _w, _h, _img) in collected for a in (*gt, *preds))
    records = [records_from_annotation(gt, preds, width=w, height=h,
                                       force_segm=any_segm, name_id=name_id)[1]
               for (_it, _rec, (gt, preds), w, h, _img) in collected]
    valid_images = [c[5] for c in collected]

    m = coco_detection_metrics(records, iou_type=dataset_iou_type,
                               iou_threshold=iou_threshold, conf_threshold=conf_threshold)

    counts_by_id = {c["image_id"]: c for c in m["per_image_counts"]}
    per_image = []
    for idx, img in enumerate(valid_images, start=1):
        c = counts_by_id.get(idx, {"tp": 0, "fp": 0, "fn": 0})
        per_image.append({"image": img.name, "tp": c["tp"], "fp": c["fp"], "fn": c["fn"]})

    out = {
        "path": folder_path,
        "image_count": len(images),
        "map": round(m["map"], 4),
        "map50": round(m["map50"], 4),
        "total_tp": m["tp"], "total_fp": m["fp"], "total_fn": m["fn"],
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
        "iou_type": dataset_iou_type,
        "per_image": per_image,
    }
    from tcip_mcp.pipelines.training.evaluation import governing_counts, resolve_match_criterion
    criterion = resolve_match_criterion(trait, records, iou_threshold=iou_threshold)
    if criterion["kind"] == "center_match":
        gc = governing_counts(records, criterion, conf_threshold=conf_threshold)
        out.update({
            "iou_total_tp": out["total_tp"], "iou_total_fp": out["total_fp"],
            "iou_total_fn": out["total_fn"],
            "total_tp": gc["tp"], "total_fp": gc["fp"], "total_fn": gc["fn"],
            "precision": round(gc["precision"], 4), "recall": round(gc["recall"], 4),
            "f1": round(gc["f1"], 4),
            "governing_criterion": criterion, "map50_role": "comparability_only",
        })
        out["per_image"] = [
            {"image": img.name,
             **{k: governing_counts([rec], criterion, conf_threshold=conf_threshold)[k]
                for k in ("tp", "fp", "fn")}}
            for rec, img in zip(records, valid_images)
        ]
    return out


def score_predictions(
    path: str,
    iou_threshold: float = 0.5,
    conf_threshold: float = DEFAULT_CONF,
    detail: bool = False,
    trait: str | None = None,
) -> dict:
    """Score on-disk predictions against on-disk ground truth (COCOeval).

    Dispatches on the input: a single image file returns per-box ``matches`` (plus an optional
    per-detection ``detections`` breakdown with ``img_w`` / ``img_h`` when ``detail=True``); a
    dataset directory returns aggregate metrics plus ``per_image`` TP/FP/FN. Both regimes share
    ``coco_detection_metrics``.

    A classified bucket's predictions carry the object class in ``subject``, so this scores the
    localization of the object class, never the classifier's own call. A prediction bucket whose
    own recorded stamp will not decode, or decodes with no ``(subject, attribute)`` pair at all,
    refuses by name.

    Args:
        path: Absolute path to an image file (single-image match) or a dataset root (aggregate).
        iou_threshold: IoU threshold for a positive match (the AP@0.5 comparability convention).
        conf_threshold: Minimum confidence to consider a prediction.
        detail: Single-image only, also return the per-detection ``detections`` breakdown: each
            entry the annotation it names as ``client_annotation`` projects it (corner ``bbox``,
            ``rings`` or ``point``, ``subject``, ``attributes``, ``iscrowd``, ``score`` and the
            provenance it holds) beside its ``tag``, ``iou`` and indices.
        trait: When set, the trait's derived localization criterion governs the reported TP/FP/FN
            count; map50 stays a labeled comparability metric. Absent -> the IoU convention
            governs.
    """
    from tcip_store import StoreError

    p = Path(path)
    try:
        if p.is_file():
            return _evaluate_image(path, iou_threshold, conf_threshold, detail, trait)
        if p.is_dir():
            return _evaluate_folder(path, iou_threshold, conf_threshold, trait)
    except (UnreadableLabelDocument, StoreError) as exc:
        return {"error": str(exc)}
    return {"error": f"Path not found: {path}"}


@mcp.tool()
@audited(scope_arg="dataset_root")
def write_subject_registry(
    dataset_root: str, subjects: dict, output_path: str = "", allow_removals: bool = False,
    allow_type_changes: bool = False,
) -> dict:
    """Author the dataset's nested subject registry, a thin wrapper over ``subject_registry``.

    ``subjects`` is the nested registry mapping the expert defines, subjects to their
    ``description`` / provenance and zero or more ``attributes`` (each ``categorical`` |
    ``ordinal`` with ordered ``values``). It is validated through
    :func:`subject_registry.registry_from_dict` (a malformed shape refuses) and written to
    ``<dataset_root>/subjects.json`` via :func:`subject_registry.replace_registry`, which reads the
    current version and passes it back in as that same call's own ``expect``, so it guards only the
    store's own window between the read and the put. No numeric class ids, no colors, no id
    enumeration.

    A write that would drop a subject, attribute or attribute value the stored registry declares is
    refused unless ``allow_removals`` is set; the same flag also allows replacing a stored registry
    whose bytes will not decode. A write that keeps an attribute's name and values but changes its
    ``type`` (categorical to ordinal or back) is refused independently of ``allow_removals`` unless
    ``allow_type_changes`` is set; landing it quarantines the finished statuses under the subject
    the same way a value change does.

    Once a new registry that changes a subject's attribute vocabulary lands, the outgoing digest is
    recorded onto that subject's still-unstamped confirmations. What was stamped, and any warning
    if the sweep could not complete, comes back under ``schema_change_sweep``.

    Args:
        dataset_root: Dataset root; the registry is written to ``<dataset_root>/subjects.json``.
        subjects: Nested ``{subject: {description?, defined_by?, defined_at?, attributes?}}`` dict.
        output_path: Optional explicit path whose directory, not its file name, is what the write
            is keyed by (``_registry_key``); the write always lands at
            ``<directory>/subjects.json``.
        allow_removals: State a dropped name, or a stored registry that will not decode, as a
            deliberate removal/repair.
        allow_type_changes: State a same-values attribute type flip (categorical to ordinal or
            back) as deliberate.
    """
    from tcip_store import VersionConflict

    from tcip_mcp import subject_registry
    from tcip_mcp.dataset_layout import subjects_path

    if not isinstance(subjects, dict) or not subjects:
        return {"error": "subjects must be a non-empty nested registry mapping"}
    try:
        registry = subject_registry.registry_from_dict(subjects)
    except subject_registry.RegistryError as exc:
        return {"error": f"invalid registry: {exc}"}

    out = Path(output_path) if output_path else subjects_path(dataset_root)
    expect = subject_registry.read_version(out)
    try:
        result = subject_registry.replace_registry(
            out, registry, expect=expect, allow_removals=allow_removals,
            allow_type_changes=allow_type_changes)
    except (subject_registry.RegistryError, VersionConflict) as exc:
        return {"error": str(exc)}
    return {"subjects_path": str(subjects_path(out.parent)),
            "subjects": [s.name for s in registry.subjects],
            "schema_change_sweep": result["schema_change_sweep"]}
