"""Built-in pipeline phase runners.

The five runners (training, inference, cropping, aggregation, export) the orchestrator ships
with. Extracted from :mod:`tcip_mcp.pipelines.orchestrator` so the engine (validation, dispatch,
retry, checkpoint/resume) stays readable; the orchestrator imports these and registers them in
``_PHASE_RUNNERS``. Each runner has the signature ``(phase, context, work_dir) -> PhaseResult``.

``PhaseResult`` is imported from the orchestrator; this module is imported by the orchestrator
only after ``PhaseResult`` is defined, so the reference resolves without a circular-import hazard.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from tcip_mcp.pipelines.orchestrator import PhaseResult

logger = logging.getLogger(__name__)


def _run_training_phase(phase: dict, context: dict[str, Any], work_dir: Path) -> PhaseResult:
    """Train a model for this phase."""
    from tcip_mcp.experiments import (
        create_experiment, log_metrics, register_model_from_experiment, update_status,
    )
    from tcip_mcp.pipelines.training.generic_trainer import (
        create_run, train, task_collate,
    )
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from torch.utils.data import DataLoader

    result = PhaseResult(phase_name=phase["name"], status="running")
    t0 = time.perf_counter()

    if "model_source" not in phase:
        result.status = "failed"
        result.error = "Training phase needs 'model_source'"
        return result

    try:
        task = phase["task"]
        ds_cfg = phase.get("dataset", {})

        # Use input artifacts from prior phase if referenced
        input_ref = phase.get("input")
        if input_ref and input_ref in context:
            prev = context[input_ref]
            if "images_dir" in prev:
                ds_cfg.setdefault("images_dir", prev["images_dir"])
            if "labels_dir" in prev:
                ds_cfg.setdefault("labels_dir", prev["labels_dir"])

        # train_stems/val_stems are split directives for this phase, not dataset
        # constructor kwargs — strip them before spreading (each build maps its
        # split onto the dataset's 'stems' parameter).
        src = {k: v for k, v in ds_cfg.items() if k not in ("train_stems", "val_stems")}
        train_ds = build_dataset(task, **{**src, "stems": ds_cfg.get("train_stems")})
        collate = task_collate(task)
        train_loader = DataLoader(train_ds, batch_size=phase.get("batch_size", 4), collate_fn=collate, shuffle=True)

        val_loader = None
        if ds_cfg.get("val_stems"):
            val_ds = build_dataset(task, **{**src, "stems": ds_cfg["val_stems"]})
            val_loader = DataLoader(val_ds, batch_size=phase.get("batch_size", 4), collate_fn=collate)

        out = work_dir / phase["name"]
        training_cfg = phase.get("training", {})
        run_config = {
            "model_source": phase["model_source"],
            "data": ds_cfg,
            "stages": phase.get("stages", [{"freeze_to": -1, "epochs": 5}, {"freeze_to": 0, "epochs": 5}]),
            "optimizer": phase.get("optimizer", {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 1e-4}),
            "early_stopping": phase.get("early_stopping", {"enabled": True, "patience": 7}),
            "mixed_precision": phase.get("mixed_precision", True),
            "batch_size": phase.get("batch_size", 4),
            "seed": phase.get("seed", training_cfg.get("seed")),
            "deterministic": phase.get("deterministic", training_cfg.get("deterministic", False)),
        }
        run = create_run(run_config, str(out))

        # Track the run like launch_training does (experiment + registry) so
        # pipeline-trained models show up in .tcip/experiments, the GUI run
        # list, and get_best_model. Best-effort: work_dir may not live inside
        # a .tcip project, and tracking failure must not fail the phase.
        try:
            create_experiment(run.run_id, run_config, data_source=ds_cfg.get("images_dir"))
            update_status(run.run_id, "running")
        except Exception as exc:
            logger.warning("Experiment tracking failed for %s: %s", run.run_id, exc)

        def _epoch_cb(epoch: int, epoch_metrics: dict) -> None:
            try:
                log_metrics(run.run_id, epoch, epoch_metrics)
            except Exception as exc:
                logger.warning("Experiment metric log failed (%s epoch %s): %s", run.run_id, epoch, exc)

        train(run, train_loader, val_loader, task=task, epoch_callback=_epoch_cb)

        try:
            if run.status == "completed":
                best = out / "model_best.pt"
                weights = str(best if best.is_file() else out / "model_final.pt")
                update_status(run.run_id, "completed")
                register_model_from_experiment(run.run_id, weights)
            else:
                update_status(run.run_id, run.status or "failed")
        except Exception as exc:
            logger.warning("Experiment completion wiring failed for %s: %s", run.run_id, exc)

        result.status = "completed" if run.status == "completed" else "failed"
        result.artifacts = {"checkpoint": str(out / "model_best.pt"), "output_dir": str(out), "run_id": run.run_id}
        result.metrics = run.metrics_history[-1] if run.metrics_history else {}
        result.error = run.error

    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        logger.exception("Training phase '%s' failed", phase["name"])

    result.elapsed_seconds = time.perf_counter() - t0
    return result


def _run_inference_phase(phase: dict, context: dict[str, Any], work_dir: Path) -> PhaseResult:
    """Run inference using a trained checkpoint."""
    result = PhaseResult(phase_name=phase["name"], status="running")
    t0 = time.perf_counter()

    try:
        from tcip_mcp.pipelines.inference.predictor import build_predictor
        from tcip_mcp.pipelines.postprocessing.export import write_predictions_json

        ckpt = phase.get("checkpoint")
        input_ref = phase.get("input")
        if not ckpt and input_ref and input_ref in context:
            ckpt = context[input_ref].get("checkpoint")

        if not ckpt:
            raise ValueError("Inference phase needs a checkpoint (from prior training or explicit)")

        predictor = build_predictor(ckpt)
        images_dir = phase.get("images_dir")
        if not images_dir and input_ref and input_ref in context:
            images_dir = context[input_ref].get("images_dir")

        if not images_dir:
            raise ValueError("Inference phase needs images_dir")

        preds_dir = work_dir / phase["name"] / "predictions"
        preds_dir.mkdir(parents=True, exist_ok=True)

        img_paths = sorted(Path(images_dir).glob("*"))
        img_paths = [p for p in img_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif"}]

        producer = f"model:{Path(ckpt).stem}"
        results_list = []
        for ip in img_paths:
            pred = predictor.predict(str(ip))
            results_list.append(pred)
            # Save predictions as the canonical per-image COCO/JSON that
            # parse_detect_predictions / the Review tab consume.
            write_predictions_json(preds_dir / f"{ip.stem}.json", pred, created_by=producer)

        result.status = "completed"
        result.artifacts = {"predictions_dir": str(preds_dir), "images_dir": images_dir, "count": len(results_list)}
        result.metrics = {"num_images": len(results_list)}

    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        logger.exception("Inference phase '%s' failed", phase["name"])

    result.elapsed_seconds = time.perf_counter() - t0
    return result


def _run_cropping_phase(phase: dict, context: dict[str, Any], work_dir: Path) -> PhaseResult:
    """Crop detected/segmented regions into sub-images for the next phase."""
    result = PhaseResult(phase_name=phase["name"], status="running")
    t0 = time.perf_counter()

    try:
        from tcip_annotation import json_io
        from tcip_mcp.pipelines.image_utils import load_image

        input_ref = phase.get("input")
        if not input_ref or input_ref not in context:
            raise ValueError("Cropping phase needs an input reference to a detection/seg phase")

        prev = context[input_ref]
        preds_dir = Path(prev.get("predictions_dir", ""))
        images_dir = Path(prev.get("images_dir", ""))

        crops_dir = work_dir / phase["name"] / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        crop_count = 0
        for pred_path in sorted(preds_dir.glob("*.json")):
            stem = pred_path.stem
            # Find original image
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".tif"):
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                continue

            # EXIF-oriented so crops align with predictions (which the predictor emits in the
            # same upright frame via load_image), not the raw sensor frame.
            img = load_image(img_path, 3)
            w, h = img.size
            # Canonical per-image COCO/JSON predictions — geometry is already pixel-xyxy.
            boxes, _ = json_io.read_detect_pred(pred_path, w, h)
            for i, b in enumerate(boxes):
                x1 = max(0, int(b.x1))
                y1 = max(0, int(b.y1))
                x2 = min(w, int(b.x2))
                y2 = min(h, int(b.y2))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                crop = img.crop((x1, y1, x2, y2))
                crop_path = crops_dir / f"{stem}_crop{i}.jpg"
                crop.save(crop_path, quality=95)
                crop_count += 1

        result.status = "completed"
        result.artifacts = {"images_dir": str(crops_dir), "count": crop_count}

    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        logger.exception("Cropping phase '%s' failed", phase["name"])

    result.elapsed_seconds = time.perf_counter() - t0
    return result


def _run_aggregation_phase(phase: dict, context: dict[str, Any], work_dir: Path) -> PhaseResult:
    """Temporal/spatial aggregation — per-image results to per-plant CSV."""
    result = PhaseResult(phase_name=phase["name"], status="running")
    t0 = time.perf_counter()

    try:
        from tcip_annotation import json_io
        from tcip_mcp.pipelines.postprocessing.aggregation import (
            aggregate_per_plant,
            export_aggregated_csv,
        )

        strategy = phase.get("strategy", "count")
        value_key = phase.get("value_key", "count")
        trait_name = phase.get("trait_name", "trait")
        crop = phase.get("crop", "")

        input_ref = phase.get("input")

        # Collect per-image results from the prior phase
        image_results: list[dict] = []
        if input_ref and input_ref in context:
            prev = context[input_ref]
            preds_dir = prev.get("predictions_dir")
            if preds_dir:
                preds_path = Path(preds_dir)
                for pred_path in sorted(preds_path.glob("*.json")):
                    boxes, _ = json_io.read_detect_pred(pred_path)
                    image_results.append({
                        "image": pred_path.stem,
                        "count": len(boxes),
                    })
            # Also check for pre-computed results list
            if "results" in prev:
                image_results = prev["results"]

        if not image_results:
            # No input data — produce empty CSV and complete gracefully
            per_plant = []
        else:
            # Aggregate
            per_plant = aggregate_per_plant(image_results, strategy=strategy, value_key=value_key)

        # Export CSV
        out_csv = work_dir / phase["name"] / "aggregated.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        export_aggregated_csv(per_plant, str(out_csv), trait_name=trait_name, crop=crop)

        result.status = "completed"
        result.artifacts = {
            "csv_path": str(out_csv),
            "strategy": strategy,
            "n_plants": len(per_plant),
        }
        result.metrics = {
            "strategy": strategy,
            "n_plants": len(per_plant),
            "n_images": len(image_results),
        }

    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        logger.exception("Aggregation phase '%s' failed", phase["name"])

    result.elapsed_seconds = time.perf_counter() - t0
    return result


def _run_export_phase(phase: dict, context: dict[str, Any], work_dir: Path) -> PhaseResult:
    """Export results to CSV or other delivery formats."""
    result = PhaseResult(phase_name=phase["name"], status="running")
    t0 = time.perf_counter()

    try:
        from tcip_annotation import json_io
        from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

        input_ref = phase.get("input")
        if not input_ref or input_ref not in context:
            raise ValueError("Export phase needs an input reference")

        prev = context[input_ref]
        out_path = work_dir / phase["name"] / "results.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # If previous phase produced a CSV, just copy/reference it
        if "csv_path" in prev:
            import shutil
            shutil.copy2(prev["csv_path"], str(out_path))
        elif "predictions_dir" in prev:
            # Build image_results from predictions dir
            preds_path = Path(prev["predictions_dir"])
            image_results = []
            for pred_path in sorted(preds_path.glob("*.json")):
                # Canonical per-image COCO/JSON predictions — count + per-object confidence.
                boxes, _ = json_io.read_detect_pred(pred_path)
                image_results.append({
                    "image": pred_path.stem,
                    "count": len(boxes),
                    "scores": [b.confidence for b in boxes],
                })
            export_detection_csv(image_results, str(out_path))
        else:
            raise ValueError("Export phase: no compatible input artifacts")

        result.status = "completed"
        result.artifacts = {"csv_path": str(out_path)}
        result.metrics = {"output_path": str(out_path)}

    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        logger.exception("Export phase '%s' failed", phase["name"])

    result.elapsed_seconds = time.perf_counter() - t0
    return result
