"""A run with no final status reads 'running' while its heartbeat is fresh.

It reads 'interrupted' when the heartbeat is past the stale window.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from tests._verified_checkpoint_fixtures import detection_config, opened_run

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _beat(run_dir: Path, seconds_ago: float) -> float:
    """Touch ``run_dir``'s heartbeat through its own writer, then age it ``seconds_ago``.

    Returns the instant the heartbeat now reads.
    """
    from tcip_mcp.experiments import HEARTBEAT_FILE, touch_heartbeat

    touch_heartbeat(run_dir)
    instant = time.time() - seconds_ago
    os.utime(run_dir / HEARTBEAT_FILE, (instant, instant))
    return instant


def _listed(project: Path) -> dict:
    """Return the project's runs as ``list_experiments`` lists them, by id."""
    from tcip_mcp.tools.experiment_tools import list_experiments

    return {r["experiment_id"]: r for r in list_experiments(project)["runs"]}


def test_the_heartbeat_window_decides_running_against_interrupted(
    tmp_path: Path,
) -> None:
    from tcip_mcp.experiments import HEARTBEAT_STALE_SECONDS, observe

    run_dir = opened_run(
        tmp_path, detection_config(tmp_path / "data", project=tmp_path)
    )
    _beat(run_dir, 0)
    assert observe(run_dir).state == "running"
    _beat(run_dir, HEARTBEAT_STALE_SECONDS - 30)
    assert observe(run_dir).state == "running"
    _beat(run_dir, HEARTBEAT_STALE_SECONDS + 60)
    assert observe(run_dir).state == "interrupted"


def test_listed_runs_read_running_against_interrupted(tmp_path: Path) -> None:
    _beat(
        opened_run(
            tmp_path,
            detection_config(tmp_path / "data", project=tmp_path),
            experiment_id="live",
        ),
        0,
    )
    _beat(
        opened_run(
            tmp_path,
            detection_config(tmp_path / "data", project=tmp_path),
            experiment_id="dead",
        ),
        2 * 3600,
    )

    by_id = _listed(tmp_path)
    assert by_id["live"]["state"] == "running"
    assert by_id["dead"]["state"] == "interrupted"


def test_a_listed_row_carries_the_heartbeat_instant(tmp_path: Path) -> None:
    """A listed row carries the heartbeat instant itself.

    A 'running' row carries the heartbeat's instant itself beside the derived state.
    """
    instant = _beat(
        opened_run(
            tmp_path,
            detection_config(tmp_path / "data", project=tmp_path),
            experiment_id="beating",
        ),
        180,
    )

    by_id = _listed(tmp_path)
    assert by_id["beating"]["state"] == "running"
    assert datetime.fromisoformat(
        by_id["beating"]["heartbeat"]
    ) == datetime.fromtimestamp(instant, timezone.utc)


def test_configured_stale_window_agrees_across_run_list_compare_and_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured stale window reads alike in the run list, the compare and the status.

    list_experiments, compare_experiments and monitor_training each read a 300s-old heartbeat
    as interrupted under a 30s ``experiments.HEARTBEAT_STALE_SECONDS``.
    """
    from tcip_mcp import experiments
    from tcip_mcp.experiments import compare_experiments as compare_tool
    from tcip_mcp.tools.training_tools import monitor_training

    monkeypatch.setattr(experiments, "HEARTBEAT_STALE_SECONDS", 30.0)
    _beat(
        opened_run(
            tmp_path,
            detection_config(tmp_path / "data", project=tmp_path),
            experiment_id="exp-window",
        ),
        300,
    )

    assert _listed(tmp_path)["exp-window"]["state"] == "interrupted"

    cmp = compare_tool(["exp-window"], project=tmp_path).model_dump()
    assert cmp["experiments"][0]["run"]["state"] == "interrupted"

    assert (
        monitor_training(tmp_path, "exp-window")["run"]["state"]
        == "interrupted"
    )
