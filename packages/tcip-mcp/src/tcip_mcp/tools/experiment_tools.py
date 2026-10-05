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
    record, the final status (``None`` until written), the derived state, and the run's metrics
    as one row per epoch, oldest first; the last row is only the last one logged, not a verified
    result. ``n_epochs`` is the number of epoch rows and the bound
    ``metrics_limit``/``metrics_offset`` page against. With ``view='lineage'`` returns only the data-to-model chain
    (``experiments.get_experiment_lineage``); ``metrics_limit``/``metrics_offset`` are refused with
    a non-default value under ``view='lineage'``.

    Args:
        experiment_id: The run to read.
        view: 'full' for the whole directory, 'lineage' for the traced chain only.
        metrics_limit: Maximum epoch rows to return, view='full' only. None returns all.
        metrics_offset: Epoch row offset to start from, view='full' only.
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
def list_experiments(project: Path) -> dict:
    """Every run and sweep of the project (``experiments.training_listing``):
    ``{"runs", "sweeps"}``, each run launched on its own as its row and each sweep with its trial
    rows under it."""
    from tcip_mcp.experiments import training_listing

    return training_listing(project).model_dump()
