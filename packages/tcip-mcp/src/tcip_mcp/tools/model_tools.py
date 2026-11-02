"""Model management tools, registry, listing, comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from tcip_mcp.model_registry import (
    Listed,
    MetricsSource,
    ModelRegistry,
    Skipped,
    best_model,
    metric_keys,
    stamped,
)
from tcip_mcp.pipelines.schemas import Direction, StatedDirection
from tcip_mcp.server import tool


class AvailableMetric(BaseModel):
    """One metric key registered models are stamped with: its role, declared direction, sources.

    ``role`` is ``comparability_only`` for a center-match comparability key, ``unlabeled``
    otherwise; ``direction`` is ``schemas.declared_direction``'s, and ``sources`` the distinct
    ``metrics_source`` values among the entries stamped with it.
    """

    metric: str
    role: Literal["comparability_only", "unlabeled"]
    direction: Direction
    sources: list[MetricsSource]


class RegisteredModels(BaseModel):
    """A project's registered models as listed, with the metrics they carry."""

    kind: Literal["listing"] = "listing"
    models: list[Listed]
    available_metrics: list[AvailableMetric]


class Ranking(Listed):
    """The registered model a metric ranks best, as listed, with how the ranking was made.

    ``metric`` is the key it was ranked by, ``direction`` the direction used and
    ``direction_source`` whether it was stated or declared; ``unregistered`` is each marked
    experiment id that registered no checkpoint, and ``skipped`` each entry
    :func:`~tcip_mcp.model_registry.best_model` left out.
    """

    kind: Literal["ranking"] = "ranking"
    metric: str
    direction: StatedDirection
    direction_source: Literal["stated", "declared"]
    unregistered: list[str]
    skipped: list[Skipped]


@tool()
def register_model(
    project: Path,
    name: str = "",
    checkpoint_path: str = "",
    metrics: dict | None = None,
    tags: list[str] | None = None,
    *,
    actor: str | None,
) -> Listed:
    """Register a foreign checkpoint in the project model registry, the registration ``actor``'s.

    A foreign checkpoint is one no run of this project produced, or a second checkpoint of a run
    under a name of its own. The entry's ``metrics_source`` is ``"caller"`` when ``metrics``
    is non-empty, ``None`` otherwise; nothing here verifies a caller-asserted metric. Returns the
    listed entry that owns the checkpoint's bytes (its ``entry`` and its checkpoint's ``facts``):
    this one, or the completed run whose final status names the same bytes.

    Args:
        name: Model name (e.g. '<crop>_<trait>_v1').
        checkpoint_path: Path to the .pt checkpoint; its config is the one its payload carries.
        metrics: Evaluation metrics.
        tags: Tags for filtering.

    """
    registry = ModelRegistry(str(project))
    return registry.register_model(
        name, checkpoint_path, metrics=metrics, tags=tags, actor=actor
    )


def _labeled_available_metrics(models: list[Listed]) -> list[AvailableMetric]:
    """List the metric keys of ``models`` (:func:`metric_keys`), with roles, directions, sources.

    ``direction`` is ``schemas.declared_direction``'s: an undeclared metric still shows up here,
    it just needs a stated direction to be ranked. ``sources`` names the distinct
    ``metrics_source`` values among entries :func:`stamped` with the key. Never refuses.
    """
    from tcip_mcp.pipelines.schemas import (
        CENTER_MATCH_COMPARABILITY_KEYS,
        VAL_METRIC_PREFIX,
        declared_direction,
    )

    result = []
    for k in metric_keys(models):
        sources = sorted(
            {
                m.facts.metrics_source
                for m in models
                if stamped(m, k) is not None
                and m.facts.metrics_source is not None
            }
        )
        result.append(
            AvailableMetric(
                metric=k,
                role="comparability_only"
                if k.removeprefix(VAL_METRIC_PREFIX)
                in CENTER_MATCH_COMPARABILITY_KEYS
                else "unlabeled",
                direction=declared_direction(k),
                sources=sources,
            )
        )
    return result


def _dumped_available_metrics(models: list[Listed]) -> list[dict]:
    """Return :func:`_labeled_available_metrics` as a refusal carries them."""
    return [m.model_dump() for m in _labeled_available_metrics(models)]


@tool()
def rank_registered_models(
    project: Path,
    metric: str = "",
    direction: Direction = "unstated",
    include_unverified: bool = False,
    experiment_ids: list[str] | None = None,
    tag: str | None = None,
) -> Annotated[RegisteredModels | Ranking, Field(discriminator="kind")] | dict:
    """List the project's registered models, or rank them by an explicit metric.

    ``metric=""`` (the default) lists rather than ranks: returns the :class:`RegisteredModels`
    of the registry (optionally ``tag``-filtered and ``experiment_ids``-narrowed), each model its
    ``entry`` and its checkpoint's ``facts``, no error, no default ranking assumed. map50-family
    metrics (and,
    once a center-match trait is in play, the IoU-convention precision/recall/F1 relabeled
    ``iou_*``) are a labeled comparability convention, not necessarily what governs a trait's
    phenotype (see the evaluation skill / ``resolve_match_criterion``). ``available_metrics``
    labels each key ``comparability_only`` vs ``unlabeled``, its declared ``direction``, and
    the ``metrics_source`` values it appears under.

    A stated ``metric`` ranks instead, answering the :class:`Ranking`: only
    ``metrics_source="trainer"`` entries by default; ``include_unverified=True`` also ranks
    ``"training_source"``/``"caller"`` entries, whose numbers were asserted, not verified. The
    undeclared-direction refusal carries ``needs_direction: True`` and the refusal that every
    entry stamped with the metric is unverified carries ``all_unverified: True``.

    Args:
        metric: Metric key to rank by; empty lists instead of ranking.
        direction: ``higher`` or ``lower`` overrides the declared direction
            (``schemas.DIRECTION_BY_METRIC``, keyed by the ``val_``-stripped name);
            required when ``metric`` is undeclared.
        include_unverified: Also rank entries whose ``metrics_source`` is not ``"trainer"``.
        experiment_ids: Narrow to entries produced by one of these experiments, applied before
            ``available_metrics`` or what the ranking skipped is derived, so both describe only
            the marked set. ``None`` (the default) covers the whole registry. With ``metric=""``, a
            narrowing that leaves nothing returns an empty listing rather than refusing.
        tag: Optional tag filter, applied to both the listing and the ranking.

    """
    if not metric:
        return registered_listing(
            project, tag=tag, experiment_ids=experiment_ids
        )
    return ranked_registered_model(
        project,
        metric,
        direction=direction,
        include_unverified=include_unverified,
        experiment_ids=experiment_ids,
        tag=tag,
    )


def _narrowed(
    models: list[Listed], experiment_ids: list[str] | None
) -> list[Listed]:
    """Return ``models`` narrowed to ``experiment_ids`` when given."""
    if experiment_ids is None:
        return models
    return [m for m in models if _run_ids([m]) & set(experiment_ids)]


def _run_ids(models: list[Listed]) -> set[str]:
    """Return the id of the run each of ``models`` came from, a foreign one naming none."""
    return {m.entry.name for m in models if m.entry.origin == "run"}


def registered_listing(
    project: Path,
    *,
    tag: str | None = None,
    experiment_ids: list[str] | None = None,
) -> RegisteredModels:
    """List the project's registered models under ``tag`` with their available metrics.

    The listing is narrowed to ``experiment_ids`` when given; an empty registry or narrowing
    lists nothing.
    """
    models = _narrowed(
        ModelRegistry(str(project)).list_models(tag), experiment_ids
    )
    return RegisteredModels(
        models=models, available_metrics=_labeled_available_metrics(models)
    )


def ranked_registered_model(
    project: Path,
    metric: str,
    *,
    direction: Direction,
    include_unverified: bool,
    experiment_ids: list[str] | None,
    tag: str | None,
) -> Ranking | dict:
    """Return the registered model ``metric`` ranks best under ``tag``, or why none ranks.

    The ranking is among ``experiment_ids`` when given, answered as its :class:`Ranking`; or the
    error dict naming why none ranks: an empty
    ``metric``, nothing registered, nothing registered by the marked experiments, an undeclared
    direction, every entry stamped with ``metric`` unverified, none stamped with it.
    """
    from tcip_mcp.pipelines.schemas import declared_direction
    from tcip_store.values import NOT_FINITE_SUFFIX

    if not metric:
        return {
            "error": "a ranking names the metric it ranks by; list the registered models "
            "and their available_metrics to pick one."
        }
    registered = ModelRegistry(str(project)).list_models(tag)
    if not registered:
        return {"error": "No models registered"}
    models = _narrowed(registered, experiment_ids)
    if not models:
        return {
            "error": "none of the marked experiments registered a checkpoint"
        }
    if metric.endswith(NOT_FINITE_SUFFIX):
        companion = metric[: -len(NOT_FINITE_SUFFIX)]
        return {
            "error": f"{metric!r} records why {companion!r} is not a number (nan/positive_"
            "infinity/negative_infinity), it is not itself a ranking.",
            "available_metrics": _dumped_available_metrics(models),
            "n_models": len(models),
        }

    direction_source: Literal["stated", "declared"] = "stated"
    if direction == "unstated":
        direction, direction_source = declared_direction(metric), "declared"
    if direction == "unstated":
        return {
            "error": f"'{metric}' has no declared ranking direction (schemas."
            "DIRECTION_BY_METRIC names no entry for it). State a direction to "
            "rank by it anyway, or pick one of the available metrics.",
            "needs_direction": True,
            "available_metrics": _dumped_available_metrics(models),
            "n_models": len(models),
        }

    best, left_out = best_model(
        models,
        metric,
        direction=direction,
        include_unverified=include_unverified,
    )
    if best is None:
        if any(s.unverified and not s.unstamped for s in left_out):
            return {
                "error": f"every registered model stamped with '{metric}' is unverified "
                "(metrics_source is not 'trainer'); include unverified models to "
                "rank them, or register a verified run.",
                "all_unverified": True,
                "skipped": [s.model_dump() for s in left_out],
                "n_models": len(models),
            }
        return {
            "error": f"No registered model is stamped with '{metric}'.",
            "available_metrics": _dumped_available_metrics(models),
            "n_models": len(models),
        }
    registered_ids = _run_ids(models)
    return Ranking(
        **dict(best),
        metric=metric,
        direction=direction,
        direction_source=direction_source,
        unregistered=[
            e for e in experiment_ids or [] if e not in registered_ids
        ],
        skipped=left_out,
    )
