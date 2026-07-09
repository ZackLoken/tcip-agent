"""Multi-phase pipeline orchestrator.

Chains phases that pass artifacts to each other. The agent designs a
PipelineSpec; the orchestrator validates structure (not content) and executes it.

Five built-in phase runners are provided: training, inference, cropping,
aggregation, export. These cover common multi-phase ML shapes but are NOT
the only allowed phase types. Pipelines can declare custom phase types and
register runners for them via ``register_phase_runner``. The orchestrator
does not gate on a closed set of phase or task names.

Example pipeline (hazelnut catkin phenology — uses built-in runners):
  1. isolate_bushes  (instance_seg) → bush_crops
  2. detect_catkins  (detection)    → catkin_detections (input: bush_crops)
  3. classify_catkins (classification) → catkin_classes (input: catkin_detections)
  4. temporal_aggregation (aggregation) → phenology_csv

For patterns that don't fit a multi-phase ML structure at all (e.g. a single
whole-plant classifier, or an NIR spectral regression), skip the orchestrator
entirely and write a Python script. The orchestrator is a convenience for
multi-phase artifact passing, not a required scaffold.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ====================================================================
# Data types
# ====================================================================

@dataclass
class PhaseResult:
    phase_name: str
    status: str = "pending"  # pending | running | completed | failed
    artifacts: dict = field(default_factory=dict)   # e.g. {"predictions_dir": "...", "checkpoint": "..."}
    metrics: dict = field(default_factory=dict)
    error: str = ""
    elapsed_seconds: float = 0.0


@dataclass
class PipelineResult:
    pipeline_name: str
    phases: list[PhaseResult] = field(default_factory=list)
    status: str = "pending"  # pending | running | completed | failed
    start_time: float = 0.0
    end_time: float = 0.0


# ====================================================================
# Validation
# ====================================================================

# Reference list of tasks the built-in phase runners know how to handle.
# This is a hint for autocomplete and documentation, NOT a gate — custom
# phase runners registered via register_phase_runner() can use any task name.
KNOWN_TASKS = {
    "detection",
    "instance_seg",
    "semantic_seg",
    "classification",
    "ordinal",
    "regression",
    "aggregation",
}


def validate_pipeline(spec: dict) -> list[str]:
    """Check a PipelineSpec for structural issues before execution.

    Only catches wiring errors (missing names, dangling input references,
    duplicate output names). Does NOT gate on a closed set of phase types
    or tasks — runners can be extended via register_phase_runner().

    Returns list of human-readable issue strings (empty = valid).
    """
    issues: list[str] = []

    if "name" not in spec:
        issues.append("Pipeline spec missing 'name'")

    phases = spec.get("phases", [])
    if not phases:
        issues.append("Pipeline has no phases")
        return issues

    output_names: set[str] = set()
    for i, phase in enumerate(phases):
        prefix = f"Phase {i} ({phase.get('name', '?')})"

        if "name" not in phase:
            issues.append(f"{prefix}: missing 'name'")
        if "task" not in phase:
            issues.append(f"{prefix}: missing 'task'")

        # Check input references exist
        input_ref = phase.get("input")
        if input_ref and input_ref not in output_names:
            issues.append(f"{prefix}: input '{input_ref}' not produced by any prior phase")

        output_ref = phase.get("output")
        if output_ref:
            if output_ref in output_names:
                issues.append(f"{prefix}: duplicate output name '{output_ref}'")
            output_names.add(output_ref)

        # Validate a dict model_spec's components/channel compatibility (composer)
        # and the phase config's types/structure (pydantic schema), deduplicated —
        # a single validation path so the orchestrator doesn't skip type checks (1.3).
        model_spec = phase.get("model_spec")
        if isinstance(model_spec, dict):
            from tcip_mcp.pipelines.composer import validate_model_spec
            from tcip_mcp.pipelines.schemas import validate_train_config_schema
            phase_issues = list(validate_model_spec(model_spec))
            phase_issues += validate_train_config_schema(phase)
            for m in dict.fromkeys(phase_issues):
                issues.append(f"{prefix}: {m}")

        # Phase-type-specific requirements (model_spec for training, etc.)
        # are enforced by the runners themselves, not here — custom phase
        # types may not need either of those fields.

    return issues


# ====================================================================
# Phase executors
# ====================================================================

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
    t0 = time.time()

    if "model_spec" not in phase:
        result.status = "failed"
        result.error = "Training phase needs 'model_spec'"
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
            "model_spec": phase["model_spec"],
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

    result.elapsed_seconds = time.time() - t0
    return result


def _run_inference_phase(phase: dict, context: dict[str, Any], work_dir: Path) -> PhaseResult:
    """Run inference using a trained checkpoint."""
    result = PhaseResult(phase_name=phase["name"], status="running")
    t0 = time.time()

    try:
        from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
        from tcip_mcp.pipelines.postprocessing.export import result_to_yolo_lines

        ckpt = phase.get("checkpoint")
        input_ref = phase.get("input")
        if not ckpt and input_ref and input_ref in context:
            ckpt = context[input_ref].get("checkpoint")

        if not ckpt:
            raise ValueError("Inference phase needs a checkpoint (from prior training or explicit)")

        predictor = GenericPredictor(ckpt)
        images_dir = phase.get("images_dir")
        if not images_dir and input_ref and input_ref in context:
            images_dir = context[input_ref].get("images_dir")

        if not images_dir:
            raise ValueError("Inference phase needs images_dir")

        preds_dir = work_dir / phase["name"] / "predictions"
        preds_dir.mkdir(parents=True, exist_ok=True)

        img_paths = sorted(Path(images_dir).glob("*"))
        img_paths = [p for p in img_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif"}]

        results_list = []
        for ip in img_paths:
            pred = predictor.predict(str(ip))
            results_list.append(pred)
            # Save predictions in the canonical "cls conf cx cy w h" format
            # that parse_detect_predictions / the Review tab consume.
            txt_path = preds_dir / f"{ip.stem}.txt"
            txt_path.write_text("\n".join(result_to_yolo_lines(pred)))

        result.status = "completed"
        result.artifacts = {"predictions_dir": str(preds_dir), "images_dir": images_dir, "count": len(results_list)}
        result.metrics = {"num_images": len(results_list)}

    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        logger.exception("Inference phase '%s' failed", phase["name"])

    result.elapsed_seconds = time.time() - t0
    return result


def _run_cropping_phase(phase: dict, context: dict[str, Any], work_dir: Path) -> PhaseResult:
    """Crop detected/segmented regions into sub-images for the next phase."""
    result = PhaseResult(phase_name=phase["name"], status="running")
    t0 = time.time()

    try:
        from PIL import Image

        input_ref = phase.get("input")
        if not input_ref or input_ref not in context:
            raise ValueError("Cropping phase needs an input reference to a detection/seg phase")

        prev = context[input_ref]
        preds_dir = Path(prev.get("predictions_dir", ""))
        images_dir = Path(prev.get("images_dir", ""))

        crops_dir = work_dir / phase["name"] / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        crop_count = 0
        for txt_path in sorted(preds_dir.glob("*.txt")):
            stem = txt_path.stem
            # Find original image
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".tif"):
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                continue

            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            for i, line in enumerate(txt_path.read_text().splitlines()):
                # Canonical prediction format: "cls conf cx cy w h" (normalized).
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                cx, cy, bw, bh = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
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

    result.elapsed_seconds = time.time() - t0
    return result


def _run_aggregation_phase(phase: dict, context: dict[str, Any], work_dir: Path) -> PhaseResult:
    """Temporal/spatial aggregation — per-image results to per-plant CSV."""
    result = PhaseResult(phase_name=phase["name"], status="running")
    t0 = time.time()

    try:
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
                for txt_path in sorted(preds_path.glob("*.txt")):
                    lines = txt_path.read_text().strip().splitlines()
                    image_results.append({
                        "image": txt_path.stem,
                        "count": len(lines),
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

    result.elapsed_seconds = time.time() - t0
    return result


def _run_export_phase(phase: dict, context: dict[str, Any], work_dir: Path) -> PhaseResult:
    """Export results to CSV or other delivery formats."""
    result = PhaseResult(phase_name=phase["name"], status="running")
    t0 = time.time()

    try:
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
            for txt_path in sorted(preds_path.glob("*.txt")):
                lines = txt_path.read_text().strip().splitlines()
                scores = []
                for line in lines:
                    # Canonical prediction format: "cls conf cx cy w h".
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            scores.append(float(parts[1]))
                        except ValueError:
                            pass
                image_results.append({
                    "image": txt_path.stem,
                    "count": len(lines),
                    "scores": scores,
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

    result.elapsed_seconds = time.time() - t0
    return result


_PHASE_RUNNERS: dict[str, Any] = {
    "training": _run_training_phase,
    "inference": _run_inference_phase,
    "cropping": _run_cropping_phase,
    "aggregation": _run_aggregation_phase,
    "export": _run_export_phase,
}


def register_phase_runner(phase_type: str, runner) -> None:
    """Register a custom phase runner.

    Lets pipelines use phase types beyond the five built-in ones. The runner
    signature is ``(phase: dict, context: dict, work_dir: Path) -> PhaseResult``.

    Example:
        def run_spectral_regression(phase, context, work_dir):
            ...
            return PhaseResult(phase_name=phase["name"], status="completed", ...)

        register_phase_runner("spectral_regression", run_spectral_regression)
    """
    _PHASE_RUNNERS[phase_type] = runner


def _infer_phase_type(phase: dict) -> str:
    """Infer phase type from explicit type field, then task, then contents.

    An explicit ``type`` field always wins — this is the escape hatch for
    custom phase types registered via register_phase_runner().
    """
    if phase.get("type"):
        return phase["type"]
    task = phase.get("task", "")
    if task == "aggregation":
        return "aggregation"
    if task == "export":
        return "export"
    if "model_spec" in phase:
        return "training"
    if "checkpoint" in phase:
        return "inference"
    return "training"


# ====================================================================
# Orchestrator
# ====================================================================

class PipelineOrchestrator:
    """Executes multi-phase pipelines with artifact passing, checkpoint/resume, and retry."""

    MAX_RETRIES = 2
    RETRY_BACKOFF = [2.0, 5.0]  # seconds between retries

    def __init__(self, work_dir: str = "./pipeline_runs") -> None:
        self.work_dir = Path(work_dir)

    def run_pipeline(self, spec: dict, resume_from: str | None = None) -> PipelineResult:
        """Execute all phases sequentially, passing artifacts between them.

        Args:
            spec: Pipeline specification dict.
            resume_from: If set, skip phases up to and including this phase name,
                         restoring context from the checkpoint file.
        """
        issues = validate_pipeline(spec)
        if issues:
            pr = PipelineResult(pipeline_name=spec.get("name", "unknown"), status="failed")
            pr.phases = [PhaseResult(phase_name="validation", status="failed", error="; ".join(issues))]
            return pr

        name = spec["name"]
        pr = PipelineResult(pipeline_name=name, status="running", start_time=time.time())
        context: dict[str, Any] = {}

        # Resume: load checkpoint and skip completed phases
        skip_until_after: str | None = None
        if resume_from:
            checkpoint = self._load_checkpoint(spec, resume_from)
            if checkpoint is None:
                # Fail loudly instead of silently re-running from scratch —
                # expensive phases (training) must not re-execute unannounced.
                logger.warning(
                    "Resume point '%s' not found in any checkpoint for pipeline '%s' under %s",
                    resume_from, name, self.work_dir,
                )
                pr.status = "failed"
                pr.phases = [PhaseResult(
                    phase_name="resume",
                    status="failed",
                    error=(
                        f"Resume point '{resume_from}' not found in any checkpoint "
                        f"under {self.work_dir}. Run without resume_from to start fresh."
                    ),
                )]
                pr.end_time = time.time()
                return pr
            context = checkpoint.get("context", {})
            skip_until_after = resume_from
            for phase_info in checkpoint.get("completed_phases", []):
                pr.phases.append(PhaseResult(
                    phase_name=phase_info["name"],
                    status="completed",
                    artifacts=phase_info.get("artifacts", {}),
                    metrics=phase_info.get("metrics", {}),
                ))
            logger.info("Resuming pipeline '%s' after phase '%s'", name, resume_from)

        run_dir = self.work_dir / f"{name}_{int(time.time())}"
        run_dir.mkdir(parents=True, exist_ok=True)

        for phase in spec["phases"]:
            # Skip already-completed phases on resume
            if skip_until_after:
                if phase["name"] == skip_until_after:
                    skip_until_after = None
                continue

            phase_type = _infer_phase_type(phase)
            runner = _PHASE_RUNNERS.get(phase_type)
            if runner is None:
                known = sorted(_PHASE_RUNNERS.keys())
                result = PhaseResult(
                    phase_name=phase["name"],
                    status="failed",
                    error=(
                        f"No runner for phase type '{phase_type}'. "
                        f"Known types: {known}. "
                        f"Register a custom runner via register_phase_runner() "
                        f"or omit 'type' to let it be inferred."
                    ),
                )
            else:
                logger.info("Running phase '%s' (type=%s)", phase["name"], phase_type)
                result = self._run_with_retry(runner, phase, context, run_dir)

            pr.phases.append(result)

            # Store artifacts for downstream phases
            output_name = phase.get("output")
            if output_name and result.status == "completed":
                context[output_name] = result.artifacts

            # Checkpoint after each completed phase
            if result.status == "completed":
                self._save_checkpoint(spec, pr, context, run_dir)

            # Stop on failure
            if result.status == "failed":
                pr.status = "failed"
                logger.error("Pipeline '%s' failed at phase '%s': %s", name, phase["name"], result.error)
                # Save checkpoint so user can resume
                self._save_checkpoint(spec, pr, context, run_dir)
                break

        if pr.status != "failed":
            pr.status = "completed"
        pr.end_time = time.time()

        # Save pipeline result
        result_path = run_dir / "pipeline_result.json"
        self._save_result(pr, result_path)

        return pr

    def _run_with_retry(
        self,
        runner,
        phase: dict,
        context: dict[str, Any],
        run_dir: Path,
    ) -> PhaseResult:
        """Run a phase with retry for transient failures."""
        last_result: PhaseResult | None = None

        for attempt in range(1 + self.MAX_RETRIES):
            result = runner(phase, context, run_dir)
            if result.status != "failed":
                return result

            last_result = result

            # Don't retry if we've exhausted attempts
            if attempt >= self.MAX_RETRIES:
                break

            # Only retry on likely-transient errors
            if not self._is_transient(result.error):
                break

            wait = self.RETRY_BACKOFF[min(attempt, len(self.RETRY_BACKOFF) - 1)]
            logger.warning(
                "Phase '%s' failed (attempt %d/%d), retrying in %.1fs: %s",
                phase["name"], attempt + 1, 1 + self.MAX_RETRIES, wait, result.error,
            )
            time.sleep(wait)

        return last_result or PhaseResult(phase_name=phase.get("name", "?"), status="failed", error="No result")

    @staticmethod
    def _is_transient(error: str) -> bool:
        """Heuristic: is this error likely transient (worth retrying)?"""
        transient_patterns = ["CUDA out of memory", "ConnectionError", "TimeoutError", "OSError"]
        return any(p.lower() in error.lower() for p in transient_patterns)

    def _save_checkpoint(
        self,
        spec: dict,
        pr: PipelineResult,
        context: dict[str, Any],
        run_dir: Path,
    ) -> None:
        """Save checkpoint for resume capability."""
        checkpoint_path = run_dir / "checkpoint.json"
        completed = [
            {
                "name": p.phase_name,
                "status": p.status,
                "artifacts": p.artifacts,
                "metrics": p.metrics,
            }
            for p in pr.phases
            if p.status == "completed"
        ]
        data = {
            "pipeline_name": spec.get("name"),
            "spec": spec,
            "completed_phases": completed,
            "context": context,
            "last_completed": completed[-1]["name"] if completed else None,
            "timestamp": time.time(),
        }
        checkpoint_path.write_text(json.dumps(data, indent=2, default=str))

    def _load_checkpoint(self, spec: dict, resume_from: str) -> dict | None:
        """Find the latest checkpoint for a pipeline."""
        name = spec.get("name", "")
        if not self.work_dir.exists():
            return None

        # Find most recent run directory for this pipeline
        candidates = sorted(
            [d for d in self.work_dir.iterdir() if d.is_dir() and d.name.startswith(name)],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        for d in candidates:
            cp = d / "checkpoint.json"
            if cp.exists():
                data = json.loads(cp.read_text())
                completed_names = [p["name"] for p in data.get("completed_phases", [])]
                if resume_from in completed_names:
                    return data
        return None

    def run_phase(self, phase: dict, context: dict[str, Any] | None = None) -> PhaseResult:
        """Run a single phase (for testing or re-runs)."""
        context = context or {}
        phase_type = _infer_phase_type(phase)
        runner = _PHASE_RUNNERS.get(phase_type)
        if runner is None:
            known = sorted(_PHASE_RUNNERS.keys())
            return PhaseResult(
                phase_name=phase.get("name", "?"),
                status="failed",
                error=f"No runner for phase type '{phase_type}'. Known types: {known}. Use register_phase_runner() to add custom runners.",
            )
        run_dir = self.work_dir / f"single_{int(time.time())}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return runner(phase, context, run_dir)

    @staticmethod
    def _save_result(pr: PipelineResult, path: Path) -> None:
        data = {
            "pipeline_name": pr.pipeline_name,
            "status": pr.status,
            "elapsed_seconds": pr.end_time - pr.start_time,
            "phases": [
                {
                    "name": p.phase_name,
                    "status": p.status,
                    "artifacts": p.artifacts,
                    "metrics": p.metrics,
                    "error": p.error,
                    "elapsed_seconds": p.elapsed_seconds,
                }
                for p in pr.phases
            ],
        }
        path.write_text(json.dumps(data, indent=2))
