"""Experiment MCP tools: read one run's directory, and list the project's runs."""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.server import tool


@tool()
def get_experiment(
    project: Path, experiment_id: str, view: str = "full",
    metrics_limit: int | None = None, metrics_offset: int = 0,
) -> dict:
    """Read one run's directory.

    With ``view='full'`` (default) returns ``experiments.get_experiment``'s read: the launch
    record, the resolved record and the final status (each ``None`` until written), the derived
    state, and the metrics rows the run appended, oldest first; the last row is only the last one
    logged, not a verified result. ``n_epochs`` is the number of distinct epoch values logged;
    ``n_rows`` is the row count and the bound ``metrics_limit``/``metrics_offset`` page against.
    With ``view='lineage'`` returns only the data-to-model chain
    (``experiments.get_experiment_lineage``); ``metrics_limit``/``metrics_offset`` are refused with
    a non-default value under ``view='lineage'``.

    Args:
        experiment_id: The run to read.
        view: 'full' for the whole directory, 'lineage' for the traced chain only.
        metrics_limit: Maximum metrics rows to return, view='full' only. None returns all.
        metrics_offset: Row offset into the metrics log to start from, view='full' only.
    """
    from tcip_mcp.experiments import get_experiment as _get
    from tcip_mcp.experiments import get_experiment_lineage as _lineage

    if view == "lineage":
        if metrics_limit is not None or metrics_offset != 0:
            return {"error": "metrics_limit/metrics_offset apply only to view='full'; "
                             "view='lineage' has no metrics rows to page."}
        return _lineage(experiment_id, project=project)
    if view != "full":
        return {"error": f"Invalid view: {view!r} (expected 'full' or 'lineage')"}
    return _get(experiment_id, project=project, metrics_limit=metrics_limit,
                metrics_offset=metrics_offset)


@tool()
def list_experiments(project: Path, launched_only: bool = False) -> dict:
    """Enumerate every run directory of the project: a training run and a calibration run of a
    checkpoint no run produced alike. Use this to rediscover the project's runs after a session is
    lost, before reaching for ``get_experiment`` (one run's full detail).

    ``launched_only=True`` switches to the other view this door serves: every training run
    directory, in id order, calibration runs and HPO trials excluded (a trial belongs to its
    sweep).

    Returns:
        With ``launched_only=False`` (default), ``experiments``: a list of
        ``{experiment_id, state, created, has_model_source}``, one per run directory
        (``experiments.list_experiments``). With ``launched_only=True``, ``runs``: the training
        run rows themselves, see :func:`tcip_mcp.tools.training_tools._all_training_runs`.
    """
    if launched_only:
        from tcip_mcp.tools.training_tools import _all_training_runs

        return {"runs": _all_training_runs(project)}

    from tcip_mcp.experiments import list_experiments as _list

    return {"experiments": _list(project)}
