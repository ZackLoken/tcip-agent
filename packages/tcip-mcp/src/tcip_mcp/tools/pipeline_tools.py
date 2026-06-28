"""Pipeline orchestration MCP tools — design, run, and monitor multi-phase pipelines."""

from __future__ import annotations

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited


@mcp.tool()
@audited
def list_components() -> dict:
    """List all registered ML components across all registries.

    Returns backbones, necks, heads, losses, and optimizers with metadata.
    The agent uses this to reason about available primitives for model design.
    """
    from tcip_mcp.pipelines.registry import BACKBONES, NECKS, HEADS, LOSSES, OPTIMIZERS

    # Import component modules to trigger registration
    import tcip_mcp.pipelines.components.backbones  # noqa: F401
    import tcip_mcp.pipelines.components.necks  # noqa: F401
    import tcip_mcp.pipelines.components.heads  # noqa: F401
    import tcip_mcp.pipelines.components.losses  # noqa: F401

    try:
        import tcip_mcp.pipelines.components.temporal  # noqa: F401
    except ImportError:
        pass
    try:
        import tcip_mcp.pipelines.components.backbones_3d  # noqa: F401
    except ImportError:
        pass
    try:
        import tcip_mcp.pipelines.training.optimizer_factory  # noqa: F401
    except ImportError:
        pass

    return {
        "backbones": BACKBONES.list(),
        "necks": NECKS.list(),
        "heads": HEADS.list(),
        "losses": LOSSES.list(),
        "optimizers": OPTIMIZERS.list(),
    }


@mcp.tool()
@audited
def recommend_model(
    task: str,
    dataset_size: int = 500,
    sensor: str = "rgb",
    num_classes: int = 1,
    num_ranks: int = 5,
    object_size: str = "medium",
) -> dict:
    """Recommend a model spec for a given task and dataset characteristics.

    The agent can use this as a starting point and then customize backbone,
    neck, head, loss, etc.

    Args:
        task: Task type (detection, classification, ordinal, regression, semantic_seg, instance_seg).
        dataset_size: Number of training images.
        sensor: Sensor type (rgb, multispectral, lidar).
        num_classes: Number of output classes.
        num_ranks: Number of ordinal ranks (only for ordinal task).
        object_size: Detection object scale (tiny/small/medium/large) — tiny/small pick
            the anchor-free FCOS detector with smaller anchors; medium/large keep Faster R-CNN.
    """
    from tcip_mcp.pipelines.composer import recommend_model_spec
    return recommend_model_spec(task, dataset_size, sensor, num_classes, num_ranks, object_size)


@mcp.tool()
@audited
def validate_model_spec(spec: dict) -> dict:
    """Validate a model specification before building.

    Checks that backbone, neck, and heads exist and are compatible.

    Args:
        spec: Model spec dict with backbone, neck, heads keys.
    """
    from tcip_mcp.pipelines.composer import validate_model_spec
    issues = validate_model_spec(spec)
    return {"valid": len(issues) == 0, "issues": issues}


@mcp.tool()
@audited
def validate_pipeline_spec(spec: dict) -> dict:
    """Validate a multi-phase pipeline specification.

    Checks phase dependencies, input/output compatibility, and that
    all referenced components exist.

    Args:
        spec: Pipeline spec with name and phases list.
    """
    from tcip_mcp.pipelines.orchestrator import validate_pipeline
    issues = validate_pipeline(spec)
    return {"valid": len(issues) == 0, "issues": issues}


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


@mcp.tool()
@audited
def compose_and_summarize(spec: dict) -> dict:
    """Build a ComposedModel from spec and return its summary.

    Useful for the agent to verify the model architecture before training.

    Args:
        spec: Model spec dict.
    """
    from tcip_mcp.pipelines.composer import compose_model, validate_model_spec

    issues = validate_model_spec(spec)
    if issues:
        return {"valid": False, "issues": issues}

    model = compose_model(spec)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "valid": True,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_M": round(total_params / 1e6, 2),
        "backbone": spec.get("backbone", {}).get("name", "?"),
        "neck": spec.get("neck", {}).get("name", "?"),
        "heads": [h.get("name", "?") for h in spec.get("heads", [])],
    }
