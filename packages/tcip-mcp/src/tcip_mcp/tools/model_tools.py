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
def register_model(
    name: str = "",
    checkpoint_path: str = "",
    config: dict | None = None,
    project_path: str = "",
    metrics: dict | None = None,
    tags: list[str] | None = None,
    experiment_id: str = "",
) -> dict:
    """Register a trained model in the project model registry.

    Two modes:
      - Explicit: pass ``config`` (and optionally ``metrics``/``tags``) directly.
      - From experiment: pass ``experiment_id`` to pull that experiment's config + the
        checkpoint's own metrics, register with an ``experiment:<id>`` back-reference, and
        record the checkpoint in the experiment's lineage. ``name`` then defaults to the
        experiment id. (Training already does this on completion; use it for manual /
        re-registration.)

    Args:
        name: Model name (e.g. 'hazelnut_catkin_v1'); defaults to the experiment id in experiment mode.
        checkpoint_path: Path to the .pt checkpoint.
        config: Training configuration used (explicit mode; ignored when ``experiment_id`` is set).
        project_path: Project root directory. Empty defaults to the platform state root.
        metrics: Evaluation metrics (explicit mode).
        tags: Tags for filtering (explicit mode).
        experiment_id: Register from this experiment instead of an explicit config.
    """
    if experiment_id:
        from tcip_mcp.experiments import register_model_from_experiment as _reg
        return _reg(experiment_id, checkpoint_path, project_path=project_path, name=name or None)
    # Record the model kind so the GUI + agent know how to run it; best-effort — a checkpoint
    # that can't be sniffed still registers, and build_predictor re-sniffs at inference time.
    kind = None
    try:
        from tcip_mcp.pipelines.inference.predictor import detect_kind
        kind = detect_kind(checkpoint_path)
    except Exception:
        kind = None
    registry = ModelRegistry(_registry_root(project_path))
    return registry.register_model(name, checkpoint_path, config or {}, metrics, tags, kind=kind)


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


def _labeled_available_metrics(models: list[dict]) -> list[dict]:
    from tcip_mcp.pipelines.training.evaluation import CENTER_MATCH_COMPARABILITY_KEYS

    keys = sorted({k for m in models for k in m.get("metrics", {})})
    return [
        {"metric": k, "role": "comparability_only"
         if k.removeprefix("val_") in CENTER_MATCH_COMPARABILITY_KEYS else "unlabeled"}
        for k in keys
    ]


@mcp.tool()
@audited
def select_best_model(project_path: str = "", metric: str = "") -> dict:
    """Get the best registered model by an explicit metric — no default is assumed (K9).

    map50-family metrics (and, once a center-match trait is in play, the IoU-convention
    precision/recall/F1 relabeled ``iou_*``) are a labeled comparability convention, not
    necessarily what governs a trait's phenotype (see the evaluation skill /
    ``resolve_match_criterion``) — silently ranking by ``val_map50`` could promote a model that is
    worse on the trait's own governing criterion. Call with ``metric=""`` (or an unknown metric) to
    get ``available_metrics`` instead of guessing, each labeled ``comparability_only`` vs
    ``unlabeled``.

    Args:
        project_path: Project root directory. Empty defaults to the platform state root.
        metric: Metric key to rank by — required. loss/error keys rank ascending, others descending.
    """
    registry = ModelRegistry(_registry_root(project_path))
    models = registry.list_models()
    if not models:
        return {"error": "No models registered"}
    if not metric:
        return {
            "error": "metric is required — select_best_model no longer defaults to 'val_map50' "
                     "(a labeled comparability metric, not necessarily what governs a trait's "
                     "phenotype). Pick one of available_metrics.",
            "available_metrics": _labeled_available_metrics(models),
            "n_models": len(models),
        }
    best = registry.best_model(metric)
    if best is None:
        return {
            "error": f"No registered model has metric '{metric}'.",
            "available_metrics": _labeled_available_metrics(models),
            "n_models": len(models),
        }
    return {**best, "ranking_basis": metric}
