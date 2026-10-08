"""Model management tools, registry, listing, comparison."""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.server import tool
from tcip_mcp.model_registry import ModelRegistry, best_model, verified


@tool()
def register_model(
    project: Path,
    name: str = "",
    checkpoint_path: str = "",
    metrics: dict | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Register a foreign checkpoint in the project model registry: one no run of this project
    produced (a run registers its own final weights on completion), or a second checkpoint of a
    run under a name of its own. The entry's ``metrics_source`` is ``"caller"`` when ``metrics``
    is non-empty, ``None`` otherwise; nothing here verifies a caller-asserted metric. Returns the
    entry that owns the checkpoint's bytes: this one, or the completed run whose final status
    names the same bytes.

    Args:
        name: Model name (e.g. '<crop>_<trait>_v1').
        checkpoint_path: Path to the .pt checkpoint; its config is the one its payload carries.
        metrics: Evaluation metrics.
        tags: Tags for filtering.
    """
    registry = ModelRegistry(str(project))
    return registry.register_model(name, checkpoint_path, metrics=metrics, tags=tags)


def _labeled_available_metrics(models: list[dict]) -> list[dict]:
    """Every metric key any registered model carries, each with what is known about it.

    ``direction`` is ``"higher"``/``"lower"`` when :mod:`evaluation`'s declaration names the bare
    (``val_``-stripped) key, else ``None``: an undeclared metric still shows up here, it just needs
    a stated direction to be ranked. ``sources`` names the distinct ``metrics_source`` values among
    entries that carry the key. Never refuses.
    """
    from tcip_mcp.pipelines.training.evaluation import (
        CENTER_MATCH_COMPARABILITY_KEYS,
        HIGHER_IS_BETTER_BY_METRIC,
        VAL_METRIC_PREFIX,
    )

    keys = sorted({k for m in models for k in m["metrics"]})
    result = []
    for k in keys:
        bare = k.removeprefix(VAL_METRIC_PREFIX)
        higher = HIGHER_IS_BETTER_BY_METRIC.get(bare)
        sources = sorted({
            m["metrics_source"] for m in models
            if k in m["metrics"] and m["metrics_source"] is not None
        })
        result.append({
            "metric": k,
            "role": (
                "comparability_only" if bare in CENTER_MATCH_COMPARABILITY_KEYS else "unlabeled"
            ),
            "direction": None if higher is None else ("higher" if higher else "lower"),
            "sources": sources,
        })
    return result


@tool()
def rank_registered_models(
    project: Path, metric: str = "",
    higher_is_better: bool | None = None, include_unverified: bool = False,
    experiment_ids: list[str] | None = None, tag: str | None = None,
) -> dict:
    """List the project's registered models, or rank them by an explicit metric.

    ``metric=""`` (the default) lists rather than ranks: returns ``{"models", "count",
    "available_metrics"}`` over the registry (optionally ``tag``-filtered and
    ``experiment_ids``-narrowed), no error, no default ranking assumed. map50-family metrics (and,
    once a center-match trait is in play, the IoU-convention precision/recall/F1 relabeled
    ``iou_*``) are a labeled comparability convention, not necessarily what governs a trait's
    phenotype (see the evaluation skill / ``resolve_match_criterion``). ``available_metrics``
    labels each key ``comparability_only`` vs ``unlabeled``, its declared ``direction``, and the
    ``metrics_source`` values it appears under.

    A stated ``metric`` ranks instead: only ``metrics_source="trainer"`` entries by default;
    ``include_unverified=True`` also ranks ``"training_source"``/``"caller"`` entries, whose
    numbers were asserted, not verified. Entries the ranking left out for being unverified are
    named in ``excluded_unverified`` when ``include_unverified`` is false; when true, the list is
    always empty. The undeclared-direction refusal carries ``needs_direction: True`` and the
    every-carrier-unverified refusal carries ``all_unverified: True``.

    Args:
        metric: Metric key to rank by; empty lists instead of ranking.
        higher_is_better: Overrides the declared direction
            (``evaluation.HIGHER_IS_BETTER_BY_METRIC``, keyed by the ``val_``-stripped name) when
            given; required when ``metric`` is undeclared.
        include_unverified: Also rank entries whose ``metrics_source`` is not ``"trainer"``.
        experiment_ids: Narrow to entries produced by one of these experiments, applied before
            ``available_metrics`` or the unverified exclusions are derived, so both describe only
            the marked set. ``None`` (the default) covers the whole registry. With ``metric=""``, a
            narrowing that leaves nothing returns an empty listing rather than refusing.
        tag: Optional tag filter, applied to both the listing and the ranking.
    """
    if not metric:
        return registered_listing(project, tag=tag, experiment_ids=experiment_ids)
    return ranked_registered_model(
        project, metric, higher_is_better=higher_is_better,
        include_unverified=include_unverified, experiment_ids=experiment_ids, tag=tag)


def _narrowed(models: list[dict], experiment_ids: list[str] | None) -> list[dict]:
    """``models`` narrowed to ``experiment_ids`` when given."""
    if experiment_ids is None:
        return models
    return [m for m in models if m["experiment_id"] in set(experiment_ids)]


def registered_listing(project: Path, *, tag: str | None = None,
                       experiment_ids: list[str] | None = None) -> dict:
    """``{"models", "count", "available_metrics"}`` over the project's registered models under
    ``tag``, narrowed to ``experiment_ids`` when given; an empty registry or narrowing lists
    nothing."""
    models = _narrowed(ModelRegistry(str(project)).list_models(tag), experiment_ids)
    return {"models": models, "count": len(models),
            "available_metrics": _labeled_available_metrics(models)}


def ranked_registered_model(
    project: Path, metric: str, *, higher_is_better: bool | None, include_unverified: bool,
    experiment_ids: list[str] | None, tag: str | None,
) -> dict:
    """The registered model ``metric`` ranks best under ``tag``, among ``experiment_ids`` when
    given, with the direction used, its source and the unverified exclusions; or the error dict
    naming why none ranks: an empty ``metric``, nothing registered, nothing registered by the
    marked experiments, an undeclared direction, every carrier unverified, no carrier."""
    from tcip_store.values import NOT_FINITE_SUFFIX

    from tcip_mcp.pipelines.training.evaluation import HIGHER_IS_BETTER_BY_METRIC, VAL_METRIC_PREFIX

    if not metric:
        return {"error": "a ranking names the metric it ranks by; list the registered models "
                         "and their available_metrics to pick one."}
    registered = ModelRegistry(str(project)).list_models(tag)
    if not registered:
        return {"error": "No models registered"}
    models = _narrowed(registered, experiment_ids)
    if not models:
        return {"error": "none of the marked experiments registered a checkpoint"}
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
                     "HIGHER_IS_BETTER_BY_METRIC names no entry for it). State a direction to "
                     "rank by it anyway, or pick one of the available metrics.",
            "needs_direction": True,
            "available_metrics": _labeled_available_metrics(models),
            "n_models": len(models),
        }

    unverified = [m for m in models if not verified(m)]
    excluded_unverified = [] if include_unverified else [
        {"name": m["name"], "metrics_source": m["metrics_source"]} for m in unverified
    ]
    best = best_model(models, metric, higher_is_better=resolved_direction,
                      include_unverified=include_unverified)
    if best is None:
        carriers = [m for m in models if metric in m["metrics"]]
        if carriers and not include_unverified and all(m in unverified for m in carriers):
            return {
                "error": f"every registered model carrying '{metric}' is unverified "
                         "(metrics_source is not 'trainer'); include unverified models to "
                         "rank them, or register a verified run.",
                "all_unverified": True,
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
