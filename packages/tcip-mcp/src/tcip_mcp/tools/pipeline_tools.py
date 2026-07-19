"""Pipeline orchestration MCP tools — run and monitor multi-phase pipelines."""

from __future__ import annotations

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited


@mcp.tool()
@audited
def run_pipeline(spec: dict, work_dir: str = "./pipeline_runs", resume_from: str = "") -> dict:
    """Execute a multi-phase pipeline.

    Phases run sequentially, passing artifacts between them. If a phase fails,
    a checkpoint is saved so you can resume later with resume_from.

    Args:
        spec: Pipeline spec dict (name + phases).
        work_dir: Directory for pipeline artifacts.
        resume_from: Optional phase name to resume after (skips completed phases).
    """
    from tcip_mcp.pipelines.orchestrator import PipelineOrchestrator
    orch = PipelineOrchestrator(work_dir=work_dir)
    result = orch.run_pipeline(spec, resume_from=resume_from or None)
    return {
        "pipeline": result.pipeline_name,
        "status": result.status,
        "phases": [
            {
                "name": p.phase_name,
                "status": p.status,
                "metrics": p.metrics,
                "artifacts": p.artifacts,
                "error": p.error,
                "elapsed": p.elapsed_seconds,
            }
            for p in result.phases
        ],
        "total_elapsed": result.end_time - result.start_time if result.end_time else 0,
    }
