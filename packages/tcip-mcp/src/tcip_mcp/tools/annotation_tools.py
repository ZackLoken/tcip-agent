"""Annotation tools: load, save, and evaluate name-based annotations via MCP.

The GUI-driving tools (push_panel_event, focus_human_attention) live in gui_tools.py; the proposal-workflow tools
(segment_prompt and stage_proposals, moved out of here, beside propose_annotations, moved out
of vision_tools.py) all live in proposal_tools.py. This module keeps label I/O and scoring.
"""

from __future__ import annotations

from pathlib import Path

from tcip_annotation import (
    Annotation,
    BBox,
    Point,
    Polygon,
    compute_matches,
    detect_format,
    load_annotations_any,
    save_annotations_any,
)
from tcip_annotation.json_io import (
    _PROV_KEYS, UnreadableLabelDocument, annotation_from_payload,
)
from tcip_annotation.json_io import read_annotations as read_labels

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
    """``(width, height)`` for ``image_path``, channel-aware, resolves through the same
    enumeration/resolution primitive every other reader now shares (``resolve_image_source``),
    so a ``.bandgroup``-grouped capture measures its real stacked frame instead of ``PIL``
    misreading a manifest file (or a genuinely multi-band raster) as a photograph.
    """
    img = Path(image_path)
    source = resolve_image_source(img.parent, img.stem)
    return image_dimensions(source)



def _ann_dict(a: Annotation) -> dict:
    """A name-based annotation as a plain JSON dict for a tool response.

    ``rings`` (not ``points``) for a polygon, a stored annotation can genuinely carry more than
    one ring (an occlusion-split instance_seg prediction), so the read side always represents every
    ring rather than silently reporting only the first. ``point`` is the same ``[x, y]`` key the
    on-disk schema uses, so a prompt/keypoint reads back as itself instead of as a geometry-less label.

    Provenance travels out under the schema's own key names, and only where the record holds it, so
    who authored a label and who accepted it are readable here rather than write-only. A reference's
    admissibility turns on exactly those two fields
    (:func:`tcip_annotation.json_io.require_reference_ground_truth`), so a reader that dropped them
    could not tell agent-authored ground truth from a person's.
    """
    d: dict = {"subject": a.subject, "attributes": dict(a.attributes)}
    if isinstance(a.geometry, BBox):
        d["bbox"] = [a.geometry.x1, a.geometry.y1, a.geometry.x2, a.geometry.y2]
    elif isinstance(a.geometry, Polygon):
        d["rings"] = [[[p[0], p[1]] for p in ring] for ring in a.geometry.rings]
    elif isinstance(a.geometry, Point):
        d["point"] = [a.geometry.x, a.geometry.y]
    if a.score is not None:
        d["score"] = a.score
    for k in _PROV_KEYS:
        v = getattr(a, k, None)
        if v is not None:
            d[k] = v
    return d


def read_annotations(image_path: str, fmt: str | None = None) -> dict:
    """Load the ground-truth labels and predictions for a single image.

    Not an MCP tool: no script wraps it, per the admission standard (packages/tcip-mcp/CLAUDE.md);
    an agent reads a label file through this function directly.

    Both are the name-based per-image schema, one file per image, all subjects. Reads the canonical
    per-image JSON (an ``annotations`` key) or an assembled dataset-level COCO (an
    ``images``/``categories`` key), detected from the file's own keys unless ``fmt`` is given. An
    unrecognized store returns an ``error`` rather than a guess.

    Args:
        image_path: Absolute path to the image file.
        fmt: Force annotation format ('json' or 'coco'). Detected from the file's keys if omitted.
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    w, h = _dims_for(image_path)
    result: dict = {"image": image_path, "width": w, "height": h}

    gt_path = find_gt_label(image_path, fmt=fmt)
    if gt_path is not None:
        try:
            file_fmt = fmt or detect_format(str(gt_path))
            anns = load_annotations_any(str(gt_path), fmt=file_fmt, file_name=img.name)
        except (ValueError, UnreadableLabelDocument) as exc:
            return {"error": str(exc)}
        result["labels"] = {
            "path": str(gt_path), "format": file_fmt, "count": len(anns),
            "subjects": sorted({a.subject for a in anns}),
            "annotations": [_ann_dict(a) for a in anns],
        }

    pred_path = find_prediction(image_path)
    if pred_path is not None:
        try:
            preds = read_labels(str(pred_path))
        except UnreadableLabelDocument as exc:
            return {"error": str(exc)}
        result["predictions"] = {
            "path": str(pred_path), "count": len(preds),
            "subjects": sorted({a.subject for a in preds}),
            "annotations": [_ann_dict(a) for a in preds],
        }

    return result


@mcp.tool()
@audited(scope_arg="image_path")
def save_annotations(
    image_path: str,
    annotations: list[dict] | None = None,
    fmt: str = "json",
    date: str | None = None,
    path: str | None = None,
    created_by: str | None = None,
) -> dict:
    """Write an image's annotations to its single per-image label file (all subjects, one file).

    The label goes to ``<dataset_root>/annotations/<date>/<stem>.json`` (see
    :mod:`tcip_mcp.dataset_layout`); ``date`` is derived from the image path when not given. Pass
    ``path`` to write to an explicit location instead. Each annotation is a dict carrying a
    ``subject`` (required, refused when absent, since a name-based label is undecodable without it),
    an optional geometry (``bbox`` = [x1,y1,x2,y2], ``points`` = [[x,y],...] for a single-ring polygon
    contour, ``rings`` = [[[x,y],...], ...] for a multi-ring polygon (an occlusion-split mask,
    e.g. ``segment_prompt``'s own output, whose ring vertices are ``{x,y}`` dicts and are accepted
    the same as ``[x,y]`` pairs), ``point`` = [x,y] for a single prompt/keypoint location, or none of
    them for an image-level label), and optional ``attributes`` (attribute name -> value name).

    Args:
        image_path: Absolute path to the image file.
        annotations: List of ``{subject, bbox?/points?/rings?/point?, attributes?}`` dicts (pixel
            coords).
        fmt: Output format, 'json' (canonical per-image, default) or 'coco'.
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
    for i, a in enumerate(anns_in):
        if not isinstance(a, dict) or not a.get("subject"):
            return {"error": f"annotation {i} needs a non-empty subject: {a!r}"}

    from tcip_mcp.workspace import is_valid_name

    if path is None and date is not None and not is_valid_name(date):
        return {"error": f"date must be a single safe path segment (no separators/'..'), got {date!r}"}

    w, h = _dims_for(image_path)

    from datetime import datetime, timezone
    _now = datetime.now(timezone.utc).isoformat()

    try:
        typed = [annotation_from_payload(a, author=created_by, now=_now) for a in anns_in]
    except ValueError as exc:
        return {"error": str(exc)}

    out_path = Path(path) if path else annotation_path_for_image(image_path, fmt, date=date)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_annotations_any(str(out_path), typed, w, h, fmt=fmt, file_name=img.name, keep_empty=True)

    try:
        from tcip_mcp.web_client import PANEL_EVENT_LABELS_WRITTEN, post_panel_event

        post_panel_event("annotate", PANEL_EVENT_LABELS_WRITTEN,
                         {"image_path": image_path, "stem": img.stem, "written": [str(out_path)]})
    except Exception:
        pass

    return {"written": [str(out_path)], "format": fmt, "count": len(typed)}


def _load_image_annotations(image_path: str, *, _checked_bucket_dirs: set | None = None):
    """Load GT + predictions for one image and build a COCO per-image record.

    Returns ``(iou_type, record, (gt, preds), width, height)`` where ``gt`` / ``preds`` are
    :class:`Annotation` lists; ``None`` if unreadable. When a prediction file is found, its
    bucket's own recorded scope is read (never trusted implicitly): a neither-key or undecodable
    stamp propagates by name (``StampScopeUnstated``, the seam's ``StoreError``) rather than
    letting a pre-conform classified bucket's value-keyed records score as object classes; a bare
    directory or any readable scope scores by ``subject`` as it does today. ``_checked_bucket_dirs``
    (a folder-scan caller's own set, threaded across its calls) skips a directory already read
    this pass, since many images share one prediction bucket and the read is for its raise alone.
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
        gt = load_annotations_any(str(gt_path), file_name=img.name)
    pred_path = find_prediction(image_path)
    if pred_path:
        bucket_dir = Path(pred_path).parent
        if _checked_bucket_dirs is None or bucket_dir not in _checked_bucket_dirs:
            bucket_scope(bucket_dir)
            if _checked_bucket_dirs is not None:
                _checked_bucket_dirs.add(bucket_dir)
        preds = read_labels(str(pred_path))

    iou_type, record = records_from_annotation(gt, preds, width=w, height=h)
    return iou_type, record, (gt, preds), w, h


def _add_geom(d: dict, a: Annotation) -> None:
    if isinstance(a.geometry, BBox):
        b = a.geometry
        d["box"] = [b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1]
    elif isinstance(a.geometry, Polygon):
        d["polygon_rings"] = [[[pt[0], pt[1]] for pt in ring] for ring in a.geometry.rings]
    elif isinstance(a.geometry, Point):
        d["point"] = [a.geometry.x, a.geometry.y]


def _detection_breakdown(matches: dict, gt: list[Annotation], preds: list[Annotation]) -> list[dict]:
    """Per-detection TP/FP/FN records (geometry + IoU + confidence + class name) from a match result."""
    detections: list[dict] = []
    for m in matches["tp"]:
        d = {"tag": "tp", "class_name": m["class_name"], "iou": m["iou"], "confidence": m["conf"],
             "gt_idx": m["gt_idx"], "pred_idx": m["pred_idx"]}
        _add_geom(d, preds[m["pred_idx"]])
        detections.append(d)
    for m in matches["fp"]:
        d = {"tag": "fp", "class_name": m["class_name"], "confidence": m["conf"],
             "pred_idx": m["pred_idx"]}
        _add_geom(d, preds[m["pred_idx"]])
        detections.append(d)
    for m in matches["fn"]:
        d = {"tag": "fn", "class_name": m["class_name"], "confidence": 0, "gt_idx": m["gt_idx"]}
        _add_geom(d, gt[m["gt_idx"]])
        detections.append(d)
    return detections


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
    can render for review (``compute_matches``); the Review tab's own GUI route reads
    ``compute_matches`` directly rather than through this tool. With a count ``trait`` the
    reported count is governed by the trait's derived criterion, map50 kept as comparability.
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

    Scores the logical images directly under ``images_dir`` plus those in each of its direct
    bucket subdirectories (``images/<bucket>/``, the dataset layout), one level: a loose image
    beside a dated bucket still scores, a ``.bandgroup``-grouped capture scores as one logical
    image, and a folder nested inside a bucket is not itself a bucket, so it is not descended.
    Two raw images sharing a case-folded stem in one bucket (different extensions, or a case
    variant) are not two identities collapsed to one: ``list_logical_images`` refuses the whole
    bucket for it, since one label document holds one record per stem and cannot represent two.
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

    from tcip_mcp.pipelines.training.evaluation import coco_detection_metrics, records_from_annotation

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
    global_names: list[str] = []
    for (_it, _rec, (gt, preds), _w, _h, _img) in collected:
        for a in (*gt, *preds):
            # Same membership records_from_annotation applies when it builds the records this map
            # keys: a geometry-less label and a Point produce no scorable box, so neither may mint a
            # COCO category that no annotation ever lands in.
            if a.geometry is None or isinstance(a.geometry, Point):
                continue
            if a.subject not in global_names:
                global_names.append(a.subject)
    name_id = {n: i + 1 for i, n in enumerate(global_names)}
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

    Not an MCP tool: run through ``scripts/score_predictions.py``, per the admission standard
    (packages/tcip-mcp/CLAUDE.md), while staying importable for its own tests.

    Dispatches on the input: a single image file returns per-box ``matches`` (plus an optional
    per-detection ``detections`` breakdown with ``img_w`` / ``img_h`` when ``detail=True``) for the
    agent to render for review; a dataset directory returns aggregate metrics plus ``per_image``
    TP/FP/FN. Both regimes share ``coco_detection_metrics``; no GUI route calls this function,
    since the Review tab's own backend route reads ``compute_matches`` directly.

    A classified bucket's predictions now carry the object class in ``subject`` (the same shape
    ground truth carries), so this scores the localization of the object class, never the
    classifier's own call: a valid number about finding the object, not about its confirmed state.
    A prediction bucket whose own recorded stamp will not decode, or decodes with no
    ``(subject, attribute)`` pair at all (a pre-conform classified bucket), refuses by name rather
    than silently scoring its value-keyed records as if they were object classes.

    Args:
        path: Absolute path to an image file (single-image match) or a dataset root (aggregate).
        iou_threshold: IoU threshold for a positive match (the AP@0.5 comparability convention).
        conf_threshold: Minimum confidence to consider a prediction.
        detail: Single-image only, also return the per-detection ``detections`` breakdown.
        trait: When set, the trait's derived localization criterion governs the reported TP/FP/FN
            count; map50 stays a labeled comparability metric. Absent -> the IoU convention governs.
    """
    from tcip_mcp.pipelines.resolution import StampScopeUnstated
    from tcip_store import StoreError

    p = Path(path)
    try:
        if p.is_file():
            return _evaluate_image(path, iou_threshold, conf_threshold, detail, trait)
        if p.is_dir():
            return _evaluate_folder(path, iou_threshold, conf_threshold, trait)
    except (UnreadableLabelDocument, StampScopeUnstated, StoreError) as exc:
        return {"error": str(exc)}
    return {"error": f"Path not found: {path}"}


@mcp.tool()
@audited(scope_arg="dataset_root")
def write_class_map(
    dataset_root: str, subjects: dict, output_path: str = "", allow_removals: bool = False,
) -> dict:
    """Author the dataset's nested class registry, a thin wrapper over ``class_registry``.

    ``subjects`` is the nested registry mapping the expert defines, subjects to their
    ``description`` / provenance and zero or more ``attributes`` (each ``categorical`` | ``ordinal``
    with ordered ``values``). It is validated through :func:`class_registry.registry_from_dict` (a
    malformed shape refuses loudly) and written to ``<dataset_root>/classes.json`` via
    :func:`class_registry.replace_registry`, which reads the current version and passes it straight
    back in as that same call's own ``expect``. This call holds no version of its own to carry, the
    way the GUI holds the one its last load returned: the read and the put happen back to back
    inside this one call, so it guards only the store's own window between them, never a window
    open before this call was made. A GUI edit landing in that earlier window that drops no
    declared name is not caught by the by-name refusal and is silently overwritten by this write;
    only a dropped name, an empty registry, or undecodable stored bytes are ever refused. No
    numeric class ids, no colors, no id enumeration: a label-scan cannot infer an attribute's type
    or rank, the expert's fact is the input here.

    A write that would drop a subject, attribute or attribute value the stored registry declares
    is refused (labels or confirmations may still reference the dropped name) unless
    ``allow_removals`` is set, which states the removal as deliberate; the same flag also allows
    replacing a stored registry whose bytes will not decode, since that is this tool's own repair
    door.

    Changing a subject's attribute vocabulary invalidates the confirmations made under the old one,
    so once the new registry lands, the outgoing digest is recorded onto that subject's still-
    unstamped confirmations; they then read as predating the change rather than as made under the
    new vocabulary. What was stamped, and any warning if the sweep could not complete, comes back
    under ``schema_change_sweep``.

    Args:
        dataset_root: Dataset root; the registry is written to ``<dataset_root>/classes.json``.
        subjects: Nested ``{subject: {description?, defined_by?, defined_at?, attributes?}}`` dict.
        output_path: Optional explicit path (overrides ``<dataset_root>/classes.json``).
        allow_removals: State a dropped name, or a stored registry that will not decode, as a
            deliberate removal/repair rather than refusing it.
    """
    from tcip_store import VersionConflict

    from tcip_mcp import class_registry
    from tcip_mcp.dataset_layout import classes_path

    if not isinstance(subjects, dict) or not subjects:
        return {"error": "subjects must be a non-empty nested registry mapping"}
    try:
        registry = class_registry.registry_from_dict(subjects)
    except class_registry.RegistryError as exc:
        return {"error": f"invalid registry: {exc}"}

    out = Path(output_path) if output_path else classes_path(dataset_root)
    expect = class_registry.read_version(out)
    try:
        result = class_registry.replace_registry(
            out, registry, expect=expect, allow_removals=allow_removals)
    except (class_registry.RegistryError, VersionConflict) as exc:
        return {"error": str(exc)}
    return {"classes_path": str(out), "subjects": [s.name for s in registry.subjects],
            "schema_change_sweep": result["schema_change_sweep"]}
