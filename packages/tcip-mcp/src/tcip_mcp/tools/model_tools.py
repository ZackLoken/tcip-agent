"""Model management tools — registry, listing, comparison."""

from __future__ import annotations

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited
from tcip_mcp.model_registry import ModelRegistry


@mcp.tool()
@audited
def list_available_models() -> dict:
    """List all available model architectures and configurations.

    Queries the composable component registries for backbones, necks,
    heads, and losses.
    """
    from tcip_mcp.pipelines.registry import (
        BACKBONES, NECKS, HEADS, LOSSES,
    )
    # Trigger registration side-effects
    import tcip_mcp.pipelines.components.backbones  # noqa: F401
    import tcip_mcp.pipelines.components.necks  # noqa: F401
    import tcip_mcp.pipelines.components.heads  # noqa: F401
    import tcip_mcp.pipelines.components.losses  # noqa: F401

    return {
        "backbones": sorted(BACKBONES.list()),
        "necks": sorted(NECKS.list()),
        "heads": sorted(HEADS.list()),
        "losses": sorted(LOSSES.list()),
    }


@mcp.tool()
@audited
def register_model(
    project_path: str,
    name: str,
    checkpoint_path: str,
    config: dict,
    metrics: dict | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Register a trained model in the project model registry.

    Args:
        project_path: Project root directory.
        name: Model name (e.g. 'hazelnut_catkin_v1').
        checkpoint_path: Path to the .pt checkpoint.
        config: Training configuration used.
        metrics: Evaluation metrics.
        tags: Tags for filtering.
    """
    registry = ModelRegistry(project_path)
    return registry.register_model(name, checkpoint_path, config, metrics, tags)


@mcp.tool()
@audited
def list_registered_models(project_path: str, tag: str | None = None) -> dict:
    """List models in the project registry.

    Args:
        project_path: Project root directory.
        tag: Optional tag filter.
    """
    registry = ModelRegistry(project_path)
    models = registry.list_models(tag)
    return {"models": models, "count": len(models)}


@mcp.tool()
@audited
def get_best_model(project_path: str, metric: str = "mAP") -> dict:
    """Get the best model by a specific metric.

    Args:
        project_path: Project root directory.
        metric: Metric key to sort by.
    """
    registry = ModelRegistry(project_path)
    best = registry.best_model(metric)
    if best is None:
        return {"error": "No models registered"}
    return best
