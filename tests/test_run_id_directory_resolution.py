"""A run named by its own id is monitored, canceled and streamed from its directory alone.

A training run's id is its directory's name, and a process other than its launcher reaches it by
that id, off the directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests._verified_checkpoint_fixtures import (
    detection_config,
    log_epoch,
    opened_run,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_a_custom_named_run_is_monitored_by_its_own_id(tmp_path: Path) -> None:
    from tcip_mcp.tools.training_tools import monitor_training

    run_dir = opened_run(
        tmp_path,
        detection_config(tmp_path / "data", project=tmp_path),
        experiment_id="exp-001-bud-det",
    )
    log_epoch(run_dir, 3, {"val_map50": 0.4})

    result = monitor_training(tmp_path, "exp-001-bud-det")["run"]
    assert result["state"] == "running"
    assert result["current_epoch"] == 3
    assert result["output_dir"] == str(run_dir)


def test_a_custom_named_run_is_canceled_by_its_own_id(tmp_path: Path) -> None:
    from tcip_mcp.experiments import cancel_requested
    from tcip_mcp.tools.training_tools import cancel_training

    run_dir = opened_run(
        tmp_path,
        detection_config(tmp_path / "data", project=tmp_path),
        experiment_id="exp-002-bud-det",
    )

    result = cancel_training(tmp_path, "exp-002-bud-det", actor=None)
    assert result["cancel_requested"] is True
    assert result["experiment_id"] == "exp-002-bud-det"
    assert cancel_requested(run_dir)


def test_an_id_naming_no_run_resolves_to_nothing(tmp_path: Path) -> None:
    from tcip_mcp.experiments import find_run
    from tcip_mcp.tools.training_tools import cancel_training, monitor_training

    opened_run(
        tmp_path,
        detection_config(tmp_path / "data", project=tmp_path),
        experiment_id="exp-004-bud-det",
    )

    assert find_run("exp-never-launched", project=tmp_path) is None
    assert "error" in monitor_training(tmp_path, "exp-never-launched")
    assert "error" in cancel_training(
        tmp_path, "exp-never-launched", actor=None
    )
