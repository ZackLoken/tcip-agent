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
def log_metrics(experiment_id: str, epoch: int, metrics: dict) -> dict:
    """Log epoch metrics to an experiment's metrics.jsonl file.

    Call this after each training epoch to record loss, mAP, precision, recall, etc.

    Args:
        experiment_id: Experiment to log to.
        epoch: Epoch number (0-indexed).
        metrics: Dict of metric values (e.g. {"train_loss": 0.5, "val_loss": 0.3, "mAP50": 0.72}).
    """
    from tcip_mcp.experiments import log_metrics as _log

    return _log(experiment_id, epoch, metrics)


@mcp.tool()
@audited
def record_artifact(experiment_id: str, name: str, path: str) -> dict:
    """Register an artifact (model weights, predictions dir, etc.) with an experiment.

    Args:
        experiment_id: Experiment to record artifact for.
        name: Artifact name (e.g. 'model_weights', 'predictions', 'best_checkpoint').
        path: Path to the artifact file or directory.
    """
    from tcip_mcp.experiments import record_artifact as _record

    return _record(experiment_id, name, path)


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
