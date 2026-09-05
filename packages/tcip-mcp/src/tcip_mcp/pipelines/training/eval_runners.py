"""Orchestrates a checkpoint evaluation run (tile-level or delivery-grade full-frame) and writes
its scored result; ``evaluation.py`` keeps the metrics computation itself.
"""

from __future__ import annotations

from pathlib import Path

from tcip_store import RECORD_JSON, Key, StoreDescriptor, register_store, store
from tcip_store.file_backend import RootedFileLocator

# Every box handed to pycocotools goes through this, both sides of a match on the one stored grid.
from tcip_annotation.json_io import xywh

from tcip_mcp.pipelines.resolution import DEFAULT_CONF, DEFAULT_MAX_DETS, DEFAULT_NMS_IOU

_RESULTS_DOC = RootedFileLocator(suffix=".json")

EVALUATION_RESULTS_STORE = "evaluation_results"
register_store(
    StoreDescriptor(
        name=EVALUATION_RESULTS_STORE,
        kind="record",
        key_fields=("document",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        locator=_RESULTS_DOC,
    )
)

_RESULTS_DOCUMENT = "test_results"


def evaluation_results_key(output_dir: Path | str) -> Key:
    """Where one evaluation run's scored result lands, under the directory it was given.

    Measurement output, so it is durable and written whole. ``last_writer_wins``: an
    evaluation composes its entire result in memory and writes it once; a later evaluation
    into the same directory is a new measurement replacing the old one, not a merge into it.
    """
    return Key(EVALUATION_RESULTS_STORE, str(Path(output_dir).resolve()), (_RESULTS_DOCUMENT,))


def evaluation_results_path(output_dir: Path | str) -> Path:
    """The result record's own file, spelled the way the caller spelled ``output_dir``.

    A finished evaluation hands its caller this path and nothing else, so it is resolved
    through the store's locator rather than composed a second time.
    """
    relative = _RESULTS_DOC.relative_path("", (_RESULTS_DOCUMENT,))
    return Path(output_dir).joinpath(*relative.parts)


def _producer_identity(checkpoint) -> dict:
    """Producing-model identity for a test-results stamp (checkpoint sha + experiment id), off
    a ``VerifiedCheckpoint`` already loaded and matched against the registry."""
    from tcip_mcp.model_registry import resolve_model_identity

    identity = resolve_model_identity(checkpoint)
    return {"model_sha256": identity["sha256"], "experiment_id": identity["experiment_id"]}


# Pinned by name and presence only; a regime's own fields travel in the writer's `extra` instead.
_COMMON_EVAL_FIELDS = (
    "model_path", "task", "model_sha256", "experiment_id", "iou_type",
    "iou_threshold", "conf_threshold", "max_dets", "tiled", "eval_regime",
)


def write_evaluation_result(output_dir: Path | str, common: dict, extra: dict) -> dict:
    """Write one evaluation-result record: the one place ``run_test_evaluation`` and
    ``run_full_frame_evaluation`` call ``store.replace`` on ``evaluation_results_key``.

    ``common`` carries the identity tuple both regimes share (``_COMMON_EVAL_FIELDS``);
    ``experiment_id`` may legitimately be ``None``, but every key must be present, or this
    refuses rather than write a result silently missing part of its own identity. ``extra``
    carries this regime's own fields (metrics included) and is written through unmodified.
    A key present in both ``common`` and ``extra`` is a programming error, not a precedence rule
    to resolve silently: this refuses rather than let one shadow the other, either direction.
    """
    missing = [field for field in _COMMON_EVAL_FIELDS if field not in common]
    if missing:
        raise ValueError(
            f"write_evaluation_result: common is missing required field(s): {missing}")
    collisions = sorted(set(common) & set(extra))
    if collisions:
        raise ValueError(
            f"write_evaluation_result: extra collides with common on field(s): {collisions}")
    result = {**{field: common[field] for field in _COMMON_EVAL_FIELDS}, **extra}
    store.replace(evaluation_results_key(output_dir), result)
    return result


def run_test_evaluation(
    checkpoint, loader, device, task: str, output_dir: str, *,
    conf_threshold: float = DEFAULT_CONF, iou_threshold: float = 0.5,  # report at the ship point
    iou_type: str | None = None, max_dets: int = 100, score_weights: dict | None = None,
    tiling: dict | None = None, trait: str | None = None,
    split_manifest_dir: str | None = None, evaluated_stem_count: int | None = None,
) -> dict:
    """Evaluate ``loader`` against ``checkpoint``, write ``test_results.json``.

    ``checkpoint`` is a ``VerifiedCheckpoint`` (``model_registry.load_registered_checkpoint``);
    this function reads no file itself.

    ``tiling`` describes the eval dataset regime for provenance only (the loader is built by the
    caller): a tile-level run scores per-tile predictions against per-tile GT (a diagnostic that
    matches the training-run val mAP), not the delivery regime, the stamp keeps the two from being
    silently conflated. See ``run_full_frame_evaluation`` for a delivery-grade metric.

    ``split_manifest_dir``/``evaluated_stem_count`` are the caller's own record of a manifest the
    loader was narrowed to (``evaluate_model``'s own binding, resolved before the loader was
    built, refusing rather than calling here when the loader admitted fewer stems than the
    universe the manifest drew): recorded verbatim when given, absent otherwise, never
    re-derived here.
    """
    from tcip_mcp.pipelines.model_build import STATE_DICT_KEY, build_model
    from tcip_mcp.pipelines.training.evaluation import effective_iou_type, evaluate

    ckpt = checkpoint.payload
    model = build_model(ckpt)
    model.load_state_dict(ckpt[STATE_DICT_KEY])
    model.to(device)

    metrics = evaluate(model, loader, device, task, conf_threshold=conf_threshold,
                       iou_threshold=iou_threshold, iou_type=iou_type, max_dets=max_dets,
                       score_weights=score_weights, trait=trait)
    tiled = bool(tiling and tiling.get("enabled", True) and task == "detection")
    producer = _producer_identity(checkpoint)
    common = {
        "model_path": checkpoint.path, "task": task,
        "model_sha256": producer["model_sha256"], "experiment_id": producer["experiment_id"],
        "iou_type": effective_iou_type(task, iou_type),
        "iou_threshold": iou_threshold, "conf_threshold": conf_threshold, "max_dets": max_dets,
        "tiled": tiled,
        "eval_regime": "tile-level" if tiled else "full-frame-single-pass",
    }
    extra = dict(metrics)
    if split_manifest_dir is not None:
        extra["split_manifest_dir"] = split_manifest_dir
        extra["evaluated_stem_count"] = evaluated_stem_count
    result = write_evaluation_result(output_dir, common, extra)
    result["results_path"] = str(evaluation_results_path(output_dir))
    return result


def run_full_frame_evaluation(
    checkpoint, images_dir: str, labels_dir: str, output_dir: str, *,
    subject: str | None = None, attribute: str | None = None,
    conf_threshold: float = DEFAULT_CONF, iou_threshold: float = 0.5,
    tile_size: int | None = None, overlap: float | None = None,
    global_nms_iou: float = DEFAULT_NMS_IOU,
    max_dets: int = DEFAULT_MAX_DETS, postprocess: str = "nms", device: str | None = None,
    trait: str | None = None, date: str | None = None,
) -> dict:
    """Delivery-grade detection eval: tiled inference reconstructed to full frame,
    matched to full-frame GT.

    Unlike tile-level eval this exercises the cross-tile merge and scores against un-fragmented GT,
    so it answers "how well does the shipped full-frame count match ground truth", the number that
    gates a phenotype delivery. Tile-level (``run_test_evaluation`` with ``tiling``) is a diagnostic
    that matches the training-run val mAP; it must not be reported as the delivery metric.

    Only call this with a checkpoint that was actually trained tiled (``predictor.train_tile_size``
    persisted), a foreign checkpoint whose geometry you can independently derive and state, or one
    where you intend to state a tile scale yourself. A checkpoint trained without tiling has no
    "regime mismatch" to reconcile in the first place, ``evaluate_model``'s default
    (``use_tiled_inference=False``) full-frame single-pass path is that model's correct delivery
    gate (same untiled regime end to end), and is the one to call instead of this function.

    ``tile_size``/``overlap`` are resolved by the same precedence ``run_inference`` uses, explicit >
    the checkpoint's own persisted training geometry > a native-ratio tier (a checkpoint's own
    recorded uniform untiled training frame) > no real basis at all, via the shared
    ``resolve_tile_regime``. Unlike the exploratory ``run_inference``, this is the delivery-gating
    call: ``tile_size`` is gated through the same shared ``resolve_tile_size_param`` every other
    door resolves through, and a scale with no real basis at all (``"unavailable"``: no explicit
    value and nothing persisted or derivable from the checkpoint) raises rather than silently
    fabricating one, since a wrong tile scale here is a wrong number that gates a phenotype, not
    just a wrong preview. ``overlap`` alone falling back to a default does not raise, a checkpoint
    trained with no tiling overlap convention at all has no persisted overlap analog, which is a
    legitimate fact, not a missing derivation; only ``tile_size``'s absence changes the object
    count's scale.

    ``date`` is the capture date the GT's confirmed negatives were recorded under, the key the
    same negative rail reads them by. It is stated by the caller, never taken from ``labels_dir``:
    a key a writer stated and a date a path spells are different facts, and a delivery-grade number
    scored against a set that silently lost its human-confirmed empties is a wrong number.

    This is a box metric (``iou_type="bbox"``): it requests boxes-only tiled inference
    (``predict_tiled(require_masks=False)``), so an instance_seg checkpoint is gated here on its
    boxes/counts, never on its masks. A mask-quality gate is separate work; do not report this
    number as one.
    """
    from tcip_mcp.pipelines.data.label_queries import json_det_targets, resolve_registry_id_map
    from tcip_mcp.pipelines.inference.predictor import (
        build_predictor, explicit_edge_provenance, resolve_tile_regime,
    )
    from tcip_mcp.pipelines.operating_point import _cap_saturated_frac
    from tcip_mcp.pipelines.resolution import resolve_tile_size_param
    from tcip_mcp.pipelines.training.evaluation import (
        build_coco_image_record, coco_detection_metrics, governing_counts, resolve_match_criterion,
    )

    predictor = build_predictor(
        checkpoint, device=device,
        score_threshold=conf_threshold, nms_iou=global_nms_iou, max_dets=max_dets)

    # A stated edge contradicting the checkpoint's own recorded geometry raises here and propagates
    # to this function's own caller, the same as every other refusal on this delivery-gating path.
    resolved_tile, tile_size_source, resolved_overlap, overlap_source, tile_resize = (
        resolve_tile_regime(predictor, tiled=True, tile_size=tile_size, overlap=overlap))
    tile_size_derived_from = (
        explicit_edge_provenance(predictor, resolved_tile)
        if tile_size_source == "explicit" and resolved_tile is not None else None)
    # The shared gate every other door resolves through: acceptance is that vocabulary and nothing
    # else, never a hand-written tuple of source labels.
    tile_param = resolve_tile_size_param(
        resolved_tile, tiled=True, tile_size_source=tile_size_source,
        tile_size_derived_from=tile_size_derived_from)
    if not tile_param.is_shippable:
        raise ValueError(
            f"Cannot resolve a trustworthy tile_size for {checkpoint.path}: no explicit tile_size was "
            "passed and the checkpoint carries no persisted or native-frame training tile geometry. "
            "This is the delivery-grade gating path (report this to gate a delivery); it refuses to "
            "silently score at a fabricated scale rather than the model's real training scale. If "
            "this checkpoint was trained without tiling, call evaluate_model with "
            "use_tiled_inference=False instead, since that untiled regime is its correct delivery "
            "gate, with no scale to reconcile. If you have genuinely derived (or intend to derive, "
            "e.g. from this dataset's object-size distribution vs. image resolution, per "
            "'Parameters: derive, don't pin') a tile scale for this checkpoint, pass it explicitly "
            "via the tiling= dict (and overlap, if known); it is not cross-checked against the "
            "checkpoint's actual training scale, so state it deliberately, not as a guess."
        )
    tile_size, overlap = tile_param.value, resolved_overlap

    img_dir, lbl_dir = Path(images_dir), Path(labels_dir)
    # GT category ids come from the run's single assign_class_ids map (json_det_targets), so
    # delivery-grade GT never diverges from training; not caught here, so its own ValueError propagates.
    _gt_id_map = None
    if subject:
        _reg, _gt_id_map = resolve_registry_id_map(lbl_dir, subject, attribute)
    # Same negative rail training uses: an image with no label record has no GT, so scoring it
    # would turn every correct detection into a false positive and drag down this delivery number.
    from tcip_mcp.pipelines.data.label_queries import image_name_map, trainable_stems

    names = image_name_map(img_dir)
    contradicted_negatives: set[str] = set()
    if lbl_dir.is_dir():
        keep, sample_counts = trainable_stems(
            lbl_dir, img_dir, subject=subject, date=date, contradicted_out=contradicted_negatives)
        paths = [img_dir / names[s] for s in keep if s in names]
    else:
        # No label store, so no rail to apply and no ground truth either; reuses names (already
        # the shared bucket enumeration) rather than a second, raw directory walk.
        sample_counts = {}
        paths = sorted(img_dir / filename for filename in names.values())
    per_image: list[dict] = []
    n_excluded_incomplete = 0
    for p in paths:
        gt = []
        gt_file = lbl_dir / f"{p.stem}.json"
        if gt_file.is_file() and _gt_id_map is not None:
            # Same loader-side reader + id map the training targets use (1-indexed to match the
            # predictor's torchvision labels), so this delivery-grade GT can't diverge from training.
            gboxes, glabels, n_unlabeled = json_det_targets(str(gt_file), subject, attribute, _gt_id_map)
            # An instance unlabeled for `attribute` gives this image incomplete GT for this scope;
            # excluded from delivery-grade scoring entirely, never scored against its labeled subset.
            if n_unlabeled:
                n_excluded_incomplete += 1
                continue
            for (x1, y1, x2, y2), lab in zip(gboxes, glabels):
                gt.append({"category_id": int(lab),
                           "bbox": xywh(x1, y1, x2, y2), "iscrowd": 0})
        # require_masks=False: this gate matches boxes to full-frame GT and never reads masks, so
        # a tile-trained instance_seg checkpoint evaluates exactly as a detector does here.
        r = predictor.predict_tiled(str(p), tile_size=tile_size, overlap=overlap,
                                    global_nms_iou=global_nms_iou, postprocess=postprocess,
                                    require_masks=False, tile_resize=tile_resize)
        w, h = int(r["width"]), int(r["height"])
        dt = [{"category_id": int(lab), "bbox": xywh(*b), "score": float(s)}
              for b, s, lab in zip(r["boxes"], r["scores"], r["labels"])]
        rec = build_coco_image_record(w, h, gt, dt, image_id=p.stem)
        # predict_tiled stamps cap_hit itself now; read it rather than re-deriving, falling back
        # to the equivalent direct computation for a predictor stub that predates that stamp.
        rec["cap_hit"] = r.get("cap_hit", len(dt) >= max_dets)
        per_image.append(rec)

    m = coco_detection_metrics(per_image, iou_threshold=iou_threshold,
                               conf_threshold=conf_threshold, max_dets=max_dets)
    keys = ("map", "map50", "map75", "map_at_maxdets", "map50_at_maxdets",
            "precision", "recall", "f1", "tp", "fp", "fn", "n_images", "n_gt", "n_pred")
    producer = _producer_identity(checkpoint)
    # task: the predictor's own real task, never a hardcoded "detection". iou_type stays the
    # literal "bbox": this gate always computes a box-only metric by design, see the docstring.
    common = {
        "model_path": checkpoint.path, "task": getattr(predictor, "task", "detection"),
        "iou_type": "bbox",
        "model_sha256": producer["model_sha256"], "experiment_id": producer["experiment_id"],
        "iou_threshold": iou_threshold, "conf_threshold": conf_threshold, "max_dets": max_dets,
        "tiled": True, "eval_regime": "full-frame-tiled-inference",
    }
    # scored_images/sample_counts/n_excluded_incomplete_attribute: which images this number was
    # computed over and which were held out, so a reviewer can reconstruct the denominator.
    extra: dict = {
        **{k: m[k] for k in keys},
        "max_dets_cap_saturated_frac": _cap_saturated_frac(per_image),
        "tile_size": tile_size, "tile_size_source": tile_size_source,
        "overlap": overlap, "overlap_source": overlap_source,
        "scored_images": len(per_image), "sample_counts": sample_counts,
        "n_excluded_incomplete_attribute": n_excluded_incomplete,
        # Names recorded negative whose label file now holds subject content; scored on that
        # content, not filtered out, but the stale confirmation needs re-review.
        "contradicted_negatives": sorted(contradicted_negatives),
    }
    # For a count trait, the delivery-grade count that gates the phenotype is the derived
    # criterion's tp/fp/fn (center-match, for a trait so configured), not AP@0.5, kept alongside, clearly labeled.
    criterion = resolve_match_criterion(trait, per_image, iou_threshold=iou_threshold)
    if criterion["kind"] == "center_match":
        gc = governing_counts(per_image, criterion, conf_threshold=conf_threshold, max_dets=max_dets)
        extra.update({
            "governing_counts": gc, "governing_criterion": criterion,
            "map50_role": "comparability_only",
            "iou_tp": m["tp"], "iou_fp": m["fp"], "iou_fn": m["fn"],
            "iou_precision": m["precision"], "iou_recall": m["recall"], "iou_f1": m["f1"],
            "tp": gc["tp"], "fp": gc["fp"], "fn": gc["fn"],
            "precision": gc["precision"], "recall": gc["recall"], "f1": gc["f1"],
        })
    result = write_evaluation_result(output_dir, common, extra)
    result["results_path"] = str(evaluation_results_path(output_dir))
    return result
