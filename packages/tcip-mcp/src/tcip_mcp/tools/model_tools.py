"""Model management tools — registry, listing, comparison."""

from __future__ import annotations

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited
from tcip_mcp.model_registry import ModelRegistry
from tcip_mcp.project_paths import project_root


def _registry_root(project_path: str) -> str:
    """Explicit path wins; empty falls back to the platform root (the adopted project)."""
    return project_path or str(project_root())


@mcp.tool()
@audited
def list_available_models() -> dict:
    """List available model architectures (backbones / necks / heads / losses).

    A focused view over the component registries. Delegates to ``list_components`` (the
    full set — also optimizers/detectors with metadata) so there is one source of truth;
    the previous standalone version crashed on ``sorted()`` over metadata dicts.
    """
    from tcip_mcp.tools.pipeline_tools import list_components
    full = list_components()
    return {key: full.get(key, []) for key in ("backbones", "necks", "heads", "losses")}


@mcp.tool()
@audited
def register_model(
    name: str,
    checkpoint_path: str,
    config: dict,
    project_path: str = "",
    metrics: dict | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Register a trained model in the project model registry.

    Args:
        name: Model name (e.g. 'hazelnut_catkin_v1').
        checkpoint_path: Path to the .pt checkpoint.
        config: Training configuration used.
        project_path: Project root directory. Empty defaults to the platform state root.
        metrics: Evaluation metrics.
        tags: Tags for filtering.
    """
    registry = ModelRegistry(_registry_root(project_path))
    return registry.register_model(name, checkpoint_path, config, metrics, tags)


@mcp.tool()
@audited
def list_registered_models(project_path: str = "", tag: str | None = None) -> dict:
    """List models in the project registry.

    Args:
        project_path: Project root directory. Empty defaults to the platform state root.
        tag: Optional tag filter.
    """
    registry = ModelRegistry(_registry_root(project_path))
    models = registry.list_models(tag)
    return {"models": models, "count": len(models)}


@mcp.tool()
@audited
def get_best_model(project_path: str = "", metric: str = "val_map50") -> dict:
    """Get the best registered model by a metric.

    Args:
        project_path: Project root directory. Empty defaults to the platform state root.
        metric: Metric key to rank by (default ``val_map50``; loss/error keys rank ascending).
    """
    registry = ModelRegistry(_registry_root(project_path))
    best = registry.best_model(metric)
    if best is None:
        models = registry.list_models()
        if not models:
            return {"error": "No models registered"}
        available = sorted({k for m in models for k in m.get("metrics", {})})
        return {
            "error": f"No registered model has metric '{metric}'.",
            "available_metrics": available,
            "n_models": len(models),
        }
    return best


@mcp.tool()
@audited
def register_model_from_experiment(
    experiment_id: str,
    checkpoint_path: str,
    project_path: str = "",
    name: str | None = None,
) -> dict:
    """Register a completed experiment's model into the project registry.

    Pulls the experiment's config + final metrics, registers the checkpoint with an
    ``experiment:<id>`` back-reference, and records it in the experiment's lineage.
    (Training already does this automatically on completion; use this for manual /
    re-registration.)

    Args:
        experiment_id: The experiment to register from.
        checkpoint_path: Path to the model checkpoint (e.g. model_best.pt).
        project_path: Project root (registry at ``<path>/.tcip/models``); empty defaults to the platform state root.
        name: Registry name (defaults to the experiment id).
    """
    from tcip_mcp.experiments import register_model_from_experiment as _reg
    return _reg(experiment_id, checkpoint_path, project_path=project_path, name=name)
