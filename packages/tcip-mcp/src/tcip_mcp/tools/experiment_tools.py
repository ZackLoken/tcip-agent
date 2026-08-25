"""Experiment tracking MCP tools: create, list, log, compare, and trace experiments."""

from __future__ import annotations

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited


@mcp.tool()
@audited
def create_experiment(
    experiment_id: str,
    config: dict,
    parent_experiment: str = "",
    data_source: str = "",
) -> dict:
    """Create a new experiment to track a training run.

    Use this tool when starting a new training run to track config,
    metrics, artifacts, and data lineage. The experiment_id should be
    descriptive, e.g. 'exp-001-<crop>-<trait>-det'.

    Args:
        experiment_id: Unique experiment name (e.g. 'exp-001-<crop>-<trait>-det').
        config: Full training configuration to snapshot.
        parent_experiment: Optional parent experiment ID for transfer learning lineage.
        data_source: Optional description of the data source.
    """
    from tcip_mcp.experiments import create_experiment as _create

    return _create(
        experiment_id,
        config,
        parent_experiment=parent_experiment or None,
        data_source=data_source or None,
    )


@mcp.tool()
@audited
def get_experiment(
    experiment_id: str, view: str = "full",
    metrics_limit: int | None = None, metrics_offset: int = 0,
) -> dict:
    """Read an experiment record.

    With ``view='full'`` (default) returns the full state: config, status, artifacts, lineage,
    and metrics, the run's own logged rows, in order, oldest first, and not a verified result.
    ``n_epochs`` is the number of distinct epoch values logged; ``n_rows`` is the row count and
    the bound ``metrics_limit``/``metrics_offset`` page against (a bespoke loop may log more than
    one row per epoch, so the two counts can differ). With ``view='lineage'`` returns only the
    data → model → predictions chain (data source, parent model, model weights path, predictions
    path), enriched with the config's data-source block; ``metrics_limit``/``metrics_offset``
    apply only to ``view='full'`` and are refused with a non-default value under ``view='lineage'``,
    which has no metrics rows to page.

    Args:
        experiment_id: Experiment to retrieve.
        view: 'full' for the complete record, 'lineage' for the traced chain only.
        metrics_limit: Maximum metrics rows to return, view='full' only. None returns all.
        metrics_offset: Row offset into the metrics log to start from, view='full' only.
    """
    from tcip_mcp.experiments import get_experiment as _get
    from tcip_mcp.experiments import get_experiment_lineage as _lineage

    if view == "lineage":
        if metrics_limit is not None or metrics_offset != 0:
            return {"error": "metrics_limit/metrics_offset apply only to view='full'; "
                             "view='lineage' has no metrics rows to page."}
        return _lineage(experiment_id)
    if view != "full":
        return {"error": f"Invalid view: {view!r} (expected 'full' or 'lineage')"}
    return _get(experiment_id, metrics_limit=metrics_limit, metrics_offset=metrics_offset)


@mcp.tool()
@audited
def list_experiments() -> dict:
    """Enumerate every experiment the store holds a status record for.

    Covers every experiment, not only a training run: a calibration experiment (its id is
    derived from a claim's content and cannot otherwise be reconstructed), a review-feedback
    lineage, a pre-created experiment never launched, and a launched one whose ``run_id`` stamp
    was lost. Use this to rediscover what the store holds after a session is lost, before
    reaching for ``list_training_runs`` (launched runs only) or ``get_experiment`` (one record's
    full detail).

    Returns:
        ``experiments``: a list of ``{experiment_id, state, created, run_id, has_model_source}``,
        one per experiment. ``run_id`` is ``None`` when no launch stamped one;
        ``has_model_source`` is whether the config carries a ``model_source`` (a training run)
        versus an experiment tracking something else.
    """
    from tcip_mcp.experiments import list_experiments as _list

    return {"experiments": _list()}


@mcp.tool()
@audited
def compare_experiments(experiment_ids: list[str]) -> dict:
    """Side-by-side comparison of multiple experiments.

    Returns, per experiment: ``recorded_state`` and the heartbeat-derived ``state``,
    ``log_locked`` (whether the metrics lock refuses further rows), ``last_logged_metrics`` (the
    last logged row, not a verified final result), ``rows_after_end`` (rows landed after the
    record's own ``ended``), ``n_epochs``/``n_rows``, ``refused_mutations`` (every refused write
    the audit log recorded against it, absent when that log can't be read), the model builder,
    and dataset identity. Use this to compare different model architectures or hyperparameters.

    Args:
        experiment_ids: List of experiment IDs to compare.
    """
    from tcip_mcp.experiments import compare_experiments as _compare

    return _compare(experiment_ids)
