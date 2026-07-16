"""Multi-phase pipeline orchestrator.

Chains phases that pass artifacts to each other. The agent designs a
PipelineSpec; the orchestrator validates structure (not content) and executes it.

Five built-in phase runners are provided: training, inference, cropping,
aggregation, export. These cover common multi-phase ML shapes but are not
the only allowed phase types. Pipelines can declare custom phase types and
register runners for them via ``register_phase_runner``. The orchestrator
does not gate on a closed set of phase or task names.

The runner *implementations* live in :mod:`tcip_mcp.pipelines.phases` (imported below);
this module keeps the engine — validation, dispatch, retry, checkpoint/resume.

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
# This is a hint for autocomplete and documentation, not a gate — custom
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
    duplicate output names). Does not gate on a closed set of phase types
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
# Phase runners (implementations in pipelines/phases.py)
# ====================================================================
# Imported after PhaseResult is defined so the runners' `from orchestrator import PhaseResult`
# resolves without a circular-import hazard.
from tcip_mcp.pipelines.phases import (  # noqa: E402
    _run_aggregation_phase,
    _run_cropping_phase,
    _run_export_phase,
    _run_inference_phase,
    _run_training_phase,
)

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
