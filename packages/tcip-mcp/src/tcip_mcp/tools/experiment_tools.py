"""Experiment tracking MCP tools — create, log, compare, and trace experiments."""

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
    descriptive, e.g. 'exp-001-hazelnut-catkin-det'.

    Args:
        experiment_id: Unique experiment name (e.g. 'exp-001-hazelnut-catkin-det').
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
def get_experiment(experiment_id: str) -> dict:
    """Read the full state of an experiment including config, metrics, artifacts, and lineage.

    Args:
        experiment_id: Experiment to retrieve.
    """
    from tcip_mcp.experiments import get_experiment as _get

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


@mcp.tool()
@audited
def get_experiment_lineage(experiment_id: str) -> dict:
    """Trace the full data → model → predictions chain for an experiment.

    Returns data source, parent model, model weights path, and predictions path.

    Args:
        experiment_id: Experiment to trace lineage for.
    """
    from tcip_mcp.experiments import get_experiment_lineage as _lineage

    return _lineage(experiment_id)
