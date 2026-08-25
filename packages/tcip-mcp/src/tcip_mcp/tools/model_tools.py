"""Model management tools, registry, listing, comparison."""

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
      - Explicit: pass ``config`` (and optionally ``metrics``/``tags``) directly; the entry's
        ``metrics_source`` becomes ``"caller"`` when ``metrics`` is non-empty, ``None`` otherwise.
        Nothing here verifies a caller-asserted metric.
      - From experiment: pass ``experiment_id`` alone to pull that experiment's config + the
        checkpoint's own metrics, register with an ``experiment:<id>`` back-reference, and
        record the checkpoint in the experiment's lineage. ``name`` then defaults to the
        experiment id. (Training already does this on completion; use it for manual /
        re-registration.) ``metrics``/``config``/``tags`` are refused alongside
        ``experiment_id``, since which path produced the numbers is what decides
        ``metrics_source``, and this mode always decides it from the experiment itself.

    Args:
        name: Model name (e.g. '<crop>_<trait>_v1'); defaults to the experiment id in experiment mode.
        checkpoint_path: Path to the .pt checkpoint.
        config: Training configuration used (explicit mode; refused when ``experiment_id`` is set).
        project_path: Project root directory. Empty defaults to the platform state root.
        metrics: Evaluation metrics (explicit mode; refused when ``experiment_id`` is set).
        tags: Tags for filtering (explicit mode; refused when ``experiment_id`` is set).
        experiment_id: Register from this experiment instead of an explicit config.
    """
    if experiment_id:
        if config or metrics or tags:
            raise ValueError(
                "register_model: experiment_id registers from the experiment's own config and "
                "checkpoint metrics; pass config/metrics/tags only in explicit mode (no "
                "experiment_id), not alongside it."
            )
        from tcip_mcp.experiments import register_model_from_experiment as _reg
        return _reg(experiment_id, checkpoint_path, project_path=project_path, name=name or None)
    # Record the model kind so the GUI + agent know how to run it; best-effort, a checkpoint
    # that can't be sniffed still registers, and build_predictor re-sniffs at inference time.
    kind = None
    try:
        from tcip_mcp.pipelines.inference.predictor import detect_kind
        kind = detect_kind(checkpoint_path)
    except Exception:
        kind = None
    registry = ModelRegistry(_registry_root(project_path))
    metrics_source = "caller" if metrics else None
    return registry.register_model(name, checkpoint_path, config or {}, metrics, tags, kind=kind,
                                    metrics_source=metrics_source)


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
    """Every metric key any registered model carries, each with what is known about it.

    ``direction`` is ``"higher"``/``"lower"`` when :mod:`evaluation`'s declaration names the bare
    (``val_``-stripped) key, else ``None``: an undeclared metric still shows up here, it just
    needs a stated direction to be ranked. ``sources`` names the distinct ``metrics_source``
    values among entries that carry the key, so a caller can see whether ranking it would mean
    trusting an unverified number. Never refuses: this is the "here is what you can pick from"
    surface for both the missing- and unknown-metric refusals below.
    """
    from tcip_mcp.pipelines.training.evaluation import (
        CENTER_MATCH_COMPARABILITY_KEYS,
        HIGHER_IS_BETTER_BY_METRIC,
        VAL_METRIC_PREFIX,
    )

    keys = sorted({k for m in models for k in m.get("metrics", {})})
    result = []
    for k in keys:
        bare = k.removeprefix(VAL_METRIC_PREFIX)
        higher = HIGHER_IS_BETTER_BY_METRIC.get(bare)
        sources = sorted({
            m["metrics_source"] for m in models
            if k in m.get("metrics", {}) and m.get("metrics_source") is not None
        })
        result.append({
            "metric": k,
            "role": "comparability_only" if bare in CENTER_MATCH_COMPARABILITY_KEYS else "unlabeled",
            "direction": None if higher is None else ("higher" if higher else "lower"),
            "sources": sources,
        })
    return result


@mcp.tool()
@audited
def select_best_model(
    project_path: str = "", metric: str = "",
    higher_is_better: bool | None = None, include_unverified: bool = False,
) -> dict:
    """Get the best registered model by an explicit metric, no default is assumed.

    map50-family metrics (and, once a center-match trait is in play, the IoU-convention
    precision/recall/F1 relabeled ``iou_*``) are a labeled comparability convention, not
    necessarily what governs a trait's phenotype (see the evaluation skill /
    ``resolve_match_criterion``), silently ranking by ``val_map50`` could promote a model that is
    worse on the trait's own governing criterion. Call with ``metric=""`` (or an unknown metric) to
    get ``available_metrics`` instead of guessing, each labeled ``comparability_only`` vs
    ``unlabeled`` and, now, its declared ``direction`` and the ``metrics_source`` values it
    appears under.

    Ranks only ``metrics_source="trainer"`` entries by default, the platform's own
    ``default_train`` is the one path anything here measured; ``include_unverified=True`` also
    ranks ``"training_source"``/``"caller"`` entries, whose numbers were asserted, not verified.
    Entries the ranking left out for being unverified are named in ``excluded_unverified`` when
    ``include_unverified`` is false; when true, nothing is left out on that basis, so the list is
    always empty.

    Args:
        project_path: Project root directory. Empty defaults to the platform state root.
        metric: Metric key to rank by, required.
        higher_is_better: Overrides the declared direction (``evaluation.HIGHER_IS_BETTER_BY_METRIC``,
            keyed by the ``val_``-stripped name) when given; required when ``metric`` is undeclared.
        include_unverified: Also rank entries whose ``metrics_source`` is not ``"trainer"``.
    """
    from tcip_store.values import NOT_FINITE_SUFFIX

    from tcip_mcp.pipelines.training.evaluation import HIGHER_IS_BETTER_BY_METRIC, VAL_METRIC_PREFIX

    registry = ModelRegistry(_registry_root(project_path))
    models = registry.list_models()
    if not models:
        return {"error": "No models registered"}
    if not metric:
        return {
            "error": "metric is required: select_best_model has no default (a labeled "
                     "comparability metric like val_map50 doesn't necessarily govern a trait's "
                     "phenotype). Pick one of available_metrics.",
            "available_metrics": _labeled_available_metrics(models),
            "n_models": len(models),
        }
    if metric.endswith(NOT_FINITE_SUFFIX):
        companion = metric[: -len(NOT_FINITE_SUFFIX)]
        return {
            "error": f"{metric!r} records why {companion!r} is not a number (nan/positive_"
                     "infinity/negative_infinity), it is not itself a ranking.",
            "available_metrics": _labeled_available_metrics(models),
            "n_models": len(models),
        }

    bare = metric.removeprefix(VAL_METRIC_PREFIX)
    declared = HIGHER_IS_BETTER_BY_METRIC.get(bare)
    if higher_is_better is not None:
        resolved_direction, direction_source = higher_is_better, "stated"
    elif declared is not None:
        resolved_direction, direction_source = declared, "declared"
    else:
        return {
            "error": f"'{metric}' has no declared ranking direction (evaluation."
                     "HIGHER_IS_BETTER_BY_METRIC names no entry for it). Pass higher_is_better "
                     "explicitly to rank by it anyway, or pick one of available_metrics.",
            "available_metrics": _labeled_available_metrics(models),
            "n_models": len(models),
        }

    unverified = [m for m in models if m.get("metrics_source") != "trainer"]
    excluded_unverified = [] if include_unverified else [
        {"name": m["name"], "metrics_source": m.get("metrics_source")} for m in unverified
    ]
    best = registry.best_model(metric, higher_is_better=resolved_direction,
                               include_unverified=include_unverified)
    if best is None:
        carriers = [m for m in models if metric in m.get("metrics", {})]
        if carriers and not include_unverified and all(m in unverified for m in carriers):
            return {
                "error": f"every registered model carrying '{metric}' is unverified "
                         "(metrics_source is not 'trainer'); pass include_unverified=True to "
                         "rank them, or register a verified run.",
                "excluded_unverified": excluded_unverified,
                "n_models": len(models),
            }
        return {
            "error": f"No registered model has metric '{metric}'.",
            "available_metrics": _labeled_available_metrics(models),
            "n_models": len(models),
        }
    return {
        **best, "ranking_basis": metric, "higher_is_better": resolved_direction,
        "direction_source": direction_source, "unverified_included": include_unverified,
        "excluded_unverified": excluded_unverified,
    }
