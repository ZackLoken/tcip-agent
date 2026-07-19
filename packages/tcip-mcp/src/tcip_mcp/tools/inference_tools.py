"""Inference MCP tools — run models on images, export results."""

from __future__ import annotations

import logging
from pathlib import Path

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited
from tcip_mcp.pipelines.postprocessing.export import export_detection_csv, write_predictions_json
from tcip_mcp.pipelines.resolution import (
    DEFAULT_CONF,
    DEFAULT_MAX_DETS,
    DEFAULT_NMS_IOU,
    DEFAULT_TILE_SIZE,
    DEFAULT_TILED,
)

logger = logging.getLogger(__name__)


def _calibrate_operating_point(predictor, trait, labels_dir, images_dir, *,
                               tile, tile_size, overlap, tile_batch_size,
                               global_nms_iou, postprocess, cross_tile_nms, max_dets):
    """Resolve a per-dataset operating point from a labeled split (CV0). Returns (bundle, hash).

    The count-unbiased center-match sweep + held-out bias check run the SAME predictor path the
    delivery will use (same tile/tile_size/overlap/nms/postprocess) over a disjoint cal/holdout split
    of the labeled dir, at a floor conf so hesitant detections survive to be swept — so the resolved
    conf is validated in the regime it ships through, not an untiled full-frame model pass.
    """
    from tcip_annotation import json_io

    from tcip_mcp.pipelines.operating_point import (
        resolve_operating_point, set_detector_operating_point,
    )
    from tcip_mcp.pipelines.resolution import dataset_hash
    from tcip_mcp.pipelines.training.evaluation import build_coco_image_record

    labels_p, images_p = Path(labels_dir), Path(images_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    label_stems = {p.stem for p in labels_p.glob("*.json")}
    stem_to_image = {p.stem: p for p in images_p.iterdir()
                     if p.suffix.lower() in image_exts and p.stem in label_stems}
    stems = sorted(stem_to_image)
    mid = max(1, len(stems) // 2)
    cal_stems, hold_stems = stems[:mid], stems[mid:]

    # Floor the in-model + predictor conf so hesitant detections survive to be swept.
    set_detector_operating_point(predictor.model, score_thresh=0.01)
    predictor.score_threshold = 0.01

    def _records(sub_stems):
        if not sub_stems:
            return []
        results = predictor.predict_batch(
            [str(stem_to_image[s]) for s in sub_stems], tile=tile, tile_size=tile_size,
            overlap=overlap, tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou,
            postprocess=postprocess,
        )
        recs = []
        for s, r in zip(sub_stems, results):
            dt = [{"category_id": int(lab), "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(sc)}
                  for (x1, y1, x2, y2), sc, lab in zip(r["boxes"], r["scores"], r["labels"])]
            # lift GT to the predictor's 1-indexed labels so the record shape matches the model pass
            gt = [{"category_id": int(b.class_id) + 1,
                   "bbox": [b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1], "iscrowd": 0}
                  for b in json_io.read_detect(str(labels_p / f"{s}.json"))[0]]
            recs.append(build_coco_image_record(int(r["width"]), int(r["height"]), gt, dt, image_id=s))
        return recs

    cal_records = _records(cal_stems)
    hold_records = _records(hold_stems)
    dh = dataset_hash(labels_dir)
    bundle = resolve_operating_point(
        trait, dataset_hash=dh, calibration_records=cal_records,
        holdout_records=hold_records or None, tile_size=tile_size,
        cross_tile_nms=cross_tile_nms, max_dets=max_dets,
    )
    return bundle, dh


def _sweep_summary(conf_param) -> dict:
    """Compact, response-safe view of a calibration sweep (the full curve is written to disk)."""
    sweep = conf_param.sweep or {}
    hb = sweep.get("holdout_bias") or {}
    return {
        "count_unbiased_conf": conf_param._raw,
        "f1_max_conf": sweep.get("f1_max_conf"),
        "holdout_bias": hb.get("count_bias_mean") if isinstance(hb, dict) else None,
        "passed_holdout": sweep.get("passed_holdout"),
        "count_bias_tolerance": sweep.get("count_bias_tolerance"),
    }


@mcp.tool()
@audited
def run_inference(
    checkpoint_path: str,
    image_paths: list[str] | None = None,
    images_dir: str | None = None,
    conf_threshold: float = DEFAULT_CONF,
    device: str | None = None,
    tile: bool = DEFAULT_TILED,
    tile_size: int | None = None,
    overlap: float | None = None,
    tile_batch_size: int = 96,
    global_nms_iou: float = DEFAULT_NMS_IOU,
    max_dets: int = DEFAULT_MAX_DETS,
    postprocess: str = "nms",
    dry_run: bool = False,
    trait: str | None = None,
    calibration_labels_dir: str | None = None,
    calibration_images_dir: str | None = None,
) -> dict:
    """Run a trained model on images.

    Provide either image_paths (specific images) or images_dir (all images
    in a directory). Set ``tile=True`` for SAHI-style sliding-window detection on
    high-resolution imagery with many small objects (detection heads only).

    Args:
        checkpoint_path: Path to model .pt checkpoint.
        image_paths: List of specific image paths.
        images_dir: Directory containing images to process.
        conf_threshold: Minimum confidence score.
        device: Device to use ('cuda' or 'cpu').
        tile: Enable tiled (SAHI-style) detection inference.
        tile_size: Sliding-window tile edge (px). ``None`` (default) derives it from the
            checkpoint's training tile geometry so inference matches the trained scale; a value
            overrides. Foreign/legacy checkpoints with no geometry fall back to 640 with a warning.
        overlap: Fractional tile overlap (stride = tile_size*(1-overlap)). ``None`` derives from the
            checkpoint (else 0.2).
        tile_batch_size: Tiles per forward batch.
        global_nms_iou: Cross-tile global NMS IoU threshold.
        max_dets: Full-frame detection cap (after any tiled merge).
        postprocess: Cross-tile merge — "nms" suppresses overlaps, "nmm" unions boxes split
            across a tile seam (better for an object straddling a boundary).
        trait: Trait name (with ``calibration_labels_dir``) to DERIVE the confidence operating point
            per dataset instead of pinning a default — the count is the phenotype, so conf must be
            calibrated (CV0). Absent -> the byte-identical raw path (conf=score_threshold, unvalidated).
        calibration_labels_dir: Labeled dir for a disjoint cal/holdout split to calibrate + held-out
            validate the operating point. Its GT identity scopes the resolved conf (dataset firewall).
        calibration_images_dir: Images for the calibration labels (defaults to ``images_dir``).
    """
    if not Path(checkpoint_path).is_file():
        return {"error": f"Checkpoint not found: {checkpoint_path}"}

    if dry_run:
        # Report the effective operating point without loading the model or running inference, so the
        # agent can see what conf/NMS/tiling will govern the object count before committing to a run.
        return {
            "dry_run": True,
            "checkpoint_path": checkpoint_path,
            "operating_point": {
                "conf": conf_threshold,
                "cross_tile_nms": global_nms_iou if tile else None,
                "tiled": tile,
                "tile_size": tile_size if tile_size is not None else "derived-from-checkpoint",
                "overlap": overlap if overlap is not None else "derived-from-checkpoint",
                "max_dets": max_dets,
                "postprocess": postprocess,
            },
            "note": ("These operating-point values govern the object count (the phenotype for count "
                     "traits). For a trait with a labeled subset, resolve them per dataset "
                     "(resolve_operating_point) so the count is calibrated, not a default."),
        }

    # Lazy import to avoid torch import at module level
    from tcip_mcp.pipelines.inference.predictor import build_predictor
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point
    from tcip_mcp.pipelines.resolution import dataset_hash, raw_operating_point

    # Thread NMS IoU + the full-frame detection cap into the model so they govern which boxes exist
    # (torchvision in-model thresholds / ultralytics overrides), not just cross-tile merge — else
    # nms_iou has no effect on an untiled run and dense scenes truncate at the framework default.
    predictor = build_predictor(
        checkpoint_path=checkpoint_path,
        device=device,
        score_threshold=conf_threshold,
        nms_iou=global_nms_iou,
        max_dets=max_dets,
    )

    # CV2: derive the tile geometry from the checkpoint's training geometry unless the caller pinned
    # it, so a tiled run doesn't silently infer at a different scale than it trained at (which shifts
    # the object count — the phenotype). None sentinel keeps an explicit 640 distinct from the default.
    geometry_warning = None
    if tile_size is not None:
        resolved_tile, tile_size_source = int(tile_size), "explicit"
    elif getattr(predictor, "train_tile_size", None):
        resolved_tile, tile_size_source = int(predictor.train_tile_size), "derived"
        if tile and resolved_tile != DEFAULT_TILE_SIZE:
            # Loud, not just provenance: counts change vs the old pinned 640 for this checkpoint.
            logger.info("tile_size %d derived from the checkpoint's training geometry "
                        "(was pinned %d before derivation existed)", resolved_tile, DEFAULT_TILE_SIZE)
    else:
        resolved_tile, tile_size_source = DEFAULT_TILE_SIZE, "default"
        if tile:
            geometry_warning = (
                "checkpoint carries no training tile geometry; using default "
                f"{DEFAULT_TILE_SIZE} — counts may not match training scale. Retrain (geometry now "
                "persisted) or pass tile_size explicitly."
            )
            logger.warning(geometry_warning)
    resolved_overlap = (
        float(overlap) if overlap is not None
        else (float(predictor.train_overlap) if getattr(predictor, "train_overlap", None) is not None else 0.2)
    )

    if image_paths is None:
        if images_dir is None:
            return {"error": "Provide either image_paths or images_dir"}
        p = Path(images_dir)
        image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        image_paths = sorted(str(f) for f in p.iterdir() if f.suffix.lower() in image_exts)

    # Resolve the confidence operating point. CV0: with a trait + labeled calibration dir, DERIVE it
    # per dataset (count-unbiased + held-out validated); otherwise the byte-identical raw path.
    extra: dict = {}
    if trait and calibration_labels_dir:
        cal_images = calibration_images_dir or images_dir
        bundle, cal_hash = _calibrate_operating_point(
            predictor, trait, calibration_labels_dir, cal_images,
            tile=tile, tile_size=resolved_tile, overlap=resolved_overlap,
            tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou, postprocess=postprocess,
            cross_tile_nms=(global_nms_iou if global_nms_iou != DEFAULT_NMS_IOU else None),
            max_dets=(max_dets if max_dets != DEFAULT_MAX_DETS else None),
        )
        conf_param = bundle.get("conf")
        conf = (conf_param.value if conf_param.is_shippable
                else conf_param.unvalidated_value(acknowledge_unvalidated=True))
        max_dets = int(bundle.get("max_dets")._raw)
        global_nms_iou = float(bundle.get("cross_tile_nms")._raw or global_nms_iou)
        # Apply the resolved operating point to the model so it governs which boxes exist.
        predictor.score_threshold = conf
        set_detector_operating_point(predictor.model, score_thresh=conf, detections_per_img=max_dets)
        op_bundle = bundle
        # Dataset-scope firewall: the conf is scoped to the calibration GT. The inference target is
        # usually UNLABELED, so its GT identity (a content hash) is undefined — pass None and record
        # 'not-comparable-unlabeled-target'. Only when inferencing the SAME labeled set it calibrated
        # on can we compare real hashes and flag cross-dataset inheritance.
        inf_stems = [Path(pp).stem for pp in image_paths]
        cal_label_stems = {pp.stem for pp in Path(calibration_labels_dir).glob("*.json")}
        same_images = calibration_images_dir is None or (
            images_dir is not None and Path(calibration_images_dir) == Path(images_dir))
        if same_images and inf_stems and set(inf_stems) == cal_label_stems:
            target_hash, cross_dataset_check = dataset_hash(calibration_labels_dir, stems=inf_stems), "same-labeled-set"
        else:
            target_hash, cross_dataset_check = None, "not-comparable-unlabeled-target"
        issues = bundle.shippable_issues(target_dataset_hash=target_hash)
        # validated only when held-out passed AND nothing is un-shippable under the target actually used.
        extra = {
            "validated": bool(bundle.is_shippable and not issues),
            "shippable_issues": issues,
            "cross_dataset_check": cross_dataset_check,
            "conf_source": "calibration",
            "dataset_hash": cal_hash,
            "sweep_summary": _sweep_summary(conf_param),
        }
        # The full sweep can be large — persist it and return the path (provenance emits has_sweep).
        try:
            from tcip_mcp.project_paths import project_root
            from tcip_mcp.utils.atomic_io import atomic_write_json
            art = project_root() / ".tcip" / "artifacts"
            art.mkdir(parents=True, exist_ok=True)
            sweep_path = art / f"operating_point_sweep_{cal_hash}.json"
            atomic_write_json(sweep_path, {"trait": trait, "dataset_hash": cal_hash,
                                           "sweep": conf_param.sweep})
            extra["sweep_path"] = str(sweep_path)
        except Exception:
            logger.warning("could not persist operating-point sweep", exc_info=True)
    else:
        # Raw inference has no per-dataset calibration: the model already carries score_threshold as
        # its in-model conf; the bundle stamps it validated_vs_gt=false so the un-trustworthiness of
        # this uncalibrated operating point (the count is the phenotype) travels with the result.
        op_bundle = raw_operating_point(
            conf=conf_threshold, cross_tile_nms=global_nms_iou, tiled=tile,
            tile_size=resolved_tile, max_dets=max_dets, tile_size_source=tile_size_source,
        )
        extra = {"validated": False, "conf_source": "default"}

    # Preflight: warn (don't fail) when a slow workload will run on CPU because CUDA isn't
    # available — full tiled inference over thousands of images is hours on CPU vs minutes on
    # a GPU. Install a CUDA torch build (see environment.yml) to use the card.
    cpu_warning = None
    if device != "cpu" and (tile or len(image_paths) > 8):
        import torch

        if not torch.cuda.is_available():
            cpu_warning = (
                f"CUDA not available — running {len(image_paths)} image(s)"
                f"{' tiled' if tile else ''} on CPU, which is much slower. Install a CUDA torch "
                "build (see environment.yml) to use the GPU."
            )
            logger.warning(cpu_warning)

    results = predictor.predict_batch(
        image_paths, tile=tile, tile_size=resolved_tile, overlap=resolved_overlap,
        tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou, postprocess=postprocess,
    )
    total_detections = sum(r["count"] for r in results)

    out = {
        "checkpoint": checkpoint_path,
        "image_count": len(results),
        "total_detections": total_detections,
        "tiled": tile,
        "operating_point": op_bundle.to_provenance()["operating_point"],
        "results": results,
        **extra,
    }
    warning = "; ".join(w for w in (geometry_warning, cpu_warning) if w)
    if warning:
        out["warning"] = warning
    return out


@mcp.tool()
@audited
def export_predictions(
    checkpoint_path: str,
    images_dir: str,
    output_dir: str,
    conf_threshold: float = DEFAULT_CONF,
    device: str | None = None,
    tile: bool = DEFAULT_TILED,
    tile_size: int | None = None,
    overlap: float | None = None,
    tile_batch_size: int = 96,
    global_nms_iou: float = DEFAULT_NMS_IOU,
    max_dets: int = DEFAULT_MAX_DETS,
    postprocess: str = "nms",
    trait: str | None = None,
    calibration_labels_dir: str | None = None,
    calibration_images_dir: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Run inference and save predictions as per-image COCO/JSON files.

    Routes through ``run_inference`` so this delivery door resolves the same firewalled
    operating point (conf/NMS/tiling/max_dets) — earlier it built its own bare predictor and
    so truncated the count at the framework default and shipped labels with no provenance.
    Writes ``<stem>.json`` per image plus an ``operating_point.json`` stamp beside them.

    A prediction bucket (``output_dir``) that already carries review verdicts is immutable: by
    default the export is redirected to a fresh run-scoped bucket (``<dir>@r2``, ``@r3`` — next
    free) and the dir actually written is returned as ``output_dir``. Pass ``overwrite=True`` to
    write in place only when the bucket has zero verdicts; with verdicts present it is refused
    (error names the count and a suggested dir) so a re-run never orphans recorded verdicts.

    Args:
        checkpoint_path: Path to model .pt checkpoint.
        images_dir: Directory containing input images.
        output_dir: Directory for output .json prediction files.
        conf_threshold: Minimum confidence score.
        device: Device to use.
        tile: Tiled (SAHI-style) inference for small dense objects.
        tile_size: Sliding-window tile edge (px).
        overlap: Fractional tile overlap.
        tile_batch_size: Tiles per forward batch.
        global_nms_iou: Cross-tile NMS IoU.
        max_dets: Full-frame detection cap.
        postprocess: Cross-tile merge — "nms" or "nmm".
        trait: Trait to calibrate the operating point per dataset (with ``calibration_labels_dir``).
        calibration_labels_dir: Labeled dir for calibrating + held-out validating the operating point.
        calibration_images_dir: Images for the calibration labels (defaults to ``images_dir``).
        overwrite: Write into ``output_dir`` even if it exists. Refused if the bucket has review
            verdicts; the default (False) auto-redirects to a fresh bucket instead.
    """
    from tcip_mcp.prediction_buckets import BucketHasVerdicts, resolve_writable_bucket
    from tcip_mcp.project_paths import resolve_state

    # Resolve the writable bucket before the (expensive) inference so a verdict-blocked overwrite
    # fails fast; verdicts live under the pinned project's ``.tcip/state``.
    out_path = Path(output_dir)
    parent, base_name = out_path.parent, out_path.name
    review_state_dir = resolve_state(Path(".tcip") / "state")
    try:
        resolution = resolve_writable_bucket(
            review_state_dir, base_name, lambda n: [parent / n], overwrite=overwrite)
    except BucketHasVerdicts as exc:
        return {"error": str(exc), "verdict_count": exc.count,
                "suggested_bucket": str(parent / exc.suggested)}
    out = parent / resolution.name

    result = run_inference(
        checkpoint_path=checkpoint_path, images_dir=images_dir, conf_threshold=conf_threshold,
        device=device, tile=tile, tile_size=tile_size, overlap=overlap,
        tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou, max_dets=max_dets,
        postprocess=postprocess, trait=trait,
        calibration_labels_dir=calibration_labels_dir, calibration_images_dir=calibration_images_dir,
    )
    if "error" in result:
        return result

    from tcip_mcp.utils.atomic_io import atomic_write_json

    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    producer = f"model:{Path(checkpoint_path).stem}"
    for r in result["results"]:
        out_json = out / f"{Path(r['image']).stem}.json"
        write_predictions_json(out_json, r, created_by=producer)
        written.append(str(out_json))

    # Stamp the operating point beside the delivered labels. ``validated`` is derived from the run's
    # resolved bundle (true only when a held-out calibration passed) — never hardcoded, or a passing
    # calibration would be dishonestly recorded as unvalidated (and vice versa).
    atomic_write_json(out / "operating_point.json",
                      {"operating_point": result.get("operating_point"),
                       "validated": bool(result.get("validated", False)),
                       "shippable_issues": result.get("shippable_issues", [])})

    return {"image_count": len(written), "output_dir": str(out), "files": written,
            "bucket_redirected": resolution.redirected,
            "requested_output_dir": output_dir if resolution.redirected else None,
            "operating_point": result.get("operating_point"),
            "validated": bool(result.get("validated", False)),
            "conf_source": result.get("conf_source")}


@mcp.tool()
@audited
def tabulate_counts(
    checkpoint_path: str,
    images_dir: str,
    output_path: str,
    conf_threshold: float = DEFAULT_CONF,
    device: str | None = None,
    tile: bool = DEFAULT_TILED,
    tile_size: int | None = None,
    overlap: float | None = None,
    tile_batch_size: int = 96,
    global_nms_iou: float = DEFAULT_NMS_IOU,
    max_dets: int = DEFAULT_MAX_DETS,
    postprocess: str = "nms",
    trait: str | None = None,
    calibration_labels_dir: str | None = None,
    calibration_images_dir: str | None = None,
) -> dict:
    """Run inference and export a CSV summary of detection counts per image.

    Routes through ``run_inference`` so the per-image counts resolve the same firewalled
    operating point (conf/NMS/tiling/max_dets) as ``run_inference``/``export_predictions`` —
    the CSV is a count-bearing deliverable (the count is the phenotype for count traits), so it
    must not be produced at a different, untiled, truncating operating point. Earlier this door
    hardcoded ``conf=0.5`` and passed no tiling/max_dets, under-reporting dense
    small-object counts relative to the other two doors.

    Args:
        checkpoint_path: Path to model .pt checkpoint.
        images_dir: Directory containing input images.
        output_path: Path for the output CSV file.
        conf_threshold: Minimum confidence score.
        device: Device to use.
        tile: Tiled (SAHI-style) inference for small dense objects.
        tile_size: Sliding-window tile edge (px).
        overlap: Fractional tile overlap.
        tile_batch_size: Tiles per forward batch.
        global_nms_iou: Cross-tile NMS IoU.
        max_dets: Full-frame detection cap.
        postprocess: Cross-tile merge — "nms" or "nmm".
        trait: Trait to calibrate the operating point per dataset (with ``calibration_labels_dir``).
        calibration_labels_dir: Labeled dir for calibrating + held-out validating the operating point.
        calibration_images_dir: Images for the calibration labels (defaults to ``images_dir``).
    """
    result = run_inference(
        checkpoint_path=checkpoint_path,
        images_dir=images_dir,
        conf_threshold=conf_threshold,
        device=device,
        tile=tile,
        tile_size=tile_size,
        overlap=overlap,
        tile_batch_size=tile_batch_size,
        global_nms_iou=global_nms_iou,
        max_dets=max_dets,
        postprocess=postprocess,
        trait=trait,
        calibration_labels_dir=calibration_labels_dir,
        calibration_images_dir=calibration_images_dir,
    )
    if "error" in result:
        return result

    csv_path = export_detection_csv(result["results"], output_path)
    return {
        "csv_path": csv_path,
        "image_count": result["image_count"],
        "total_detections": result["total_detections"],
        # Carry the operating point that produced these counts — the CSV is a count-bearing
        # deliverable and the numbers are only as trustworthy as the operating point behind them.
        "operating_point": result.get("operating_point"),
        "validated": bool(result.get("validated", False)),
        "conf_source": result.get("conf_source"),
    }
