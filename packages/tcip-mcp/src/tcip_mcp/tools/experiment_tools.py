"""Experiment tracking MCP tools: create, log, compare, and trace experiments."""

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
def get_experiment(experiment_id: str, view: str = "full") -> dict:
    """Read an experiment record.

    With ``view='full'`` (default) returns the full state: config, metrics, artifacts,
    and lineage. With ``view='lineage'`` returns only the data → model → predictions
    chain (data source, parent model, model weights path, predictions path), enriched
    with the config's data-source block.

    Args:
        experiment_id: Experiment to retrieve.
        view: 'full' for the complete record, 'lineage' for the traced chain only.
    """
    from tcip_mcp.experiments import get_experiment as _get
    from tcip_mcp.experiments import get_experiment_lineage as _lineage

    if view == "lineage":
        return _lineage(experiment_id)
    if view != "full":
        return {"error": f"Invalid view: {view!r} (expected 'full' or 'lineage')"}
    return _get(experiment_id)


@mcp.tool()
@audited
def compare_experiments(experiment_ids: list[str]) -> dict:
    """Side-by-side comparison of multiple experiments.

    Returns final metrics, backbone, and epoch count for each experiment.
    Use this to compare different model architectures or hyperparameters.

    Args:
        experiment_ids: List of experiment IDs to compare.
    """
    from tcip_mcp.experiments import compare_experiments as _compare

    return _compare(experiment_ids)
