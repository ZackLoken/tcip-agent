"""A run directory with no final status reads as 'running' while its heartbeat is fresh (still
training in another process, e.g. the MCP agent) and only 'interrupted' once the heartbeat goes
stale. Prevents the GUI mislabeling an agent-launched run as dead and inviting a duplicate
launch."""

import os
import time
from datetime import datetime, timezone

from tests._verified_checkpoint_fixtures import detection_config, opened_run


def _config(tmp_path) -> dict:
    """A detector run's config over two frames of its own under ``tmp_path``."""
    return detection_config(tmp_path / "data",
                            model_source={"builder": "my_models:build_detector",
                                          "task": "detection"})


def _beat(run_dir, seconds_ago: float) -> float:
    """Touch ``run_dir``'s heartbeat through its own writer, then age it ``seconds_ago``; the
    instant it now reads."""
    from tcip_mcp.experiments import HEARTBEAT_FILE, touch_heartbeat

    touch_heartbeat(run_dir)
    instant = time.time() - seconds_ago
    os.utime(run_dir / HEARTBEAT_FILE, (instant, instant))
    return instant


def test_the_heartbeat_window_decides_running_against_interrupted(tmp_path):
    from tcip_mcp.experiments import HEARTBEAT_STALE_SECONDS, observe

    run_dir = opened_run(tmp_path, _config(tmp_path))
    _beat(run_dir, 0)
    assert observe(run_dir).state == "running"
    _beat(run_dir, HEARTBEAT_STALE_SECONDS - 30)
    assert observe(run_dir).state == "running"
    _beat(run_dir, HEARTBEAT_STALE_SECONDS + 60)
    assert observe(run_dir).state == "interrupted"


def test_listed_runs_read_running_against_interrupted(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.tools.experiment_tools import list_experiments

    _beat(opened_run(None, _config(tmp_path), experiment_id="live"), 0)
    _beat(opened_run(None, _config(tmp_path), experiment_id="dead"), 2 * 3600)

    by_id = {r["experiment_id"]: r for r in list_experiments(launched_only=True)["runs"]}
    assert by_id["live"]["status"] == "running"
    assert by_id["dead"]["status"] == "interrupted"


def test_a_listed_row_carries_the_heartbeat_instant(tmp_path, monkeypatch):
    """No process id is recorded anywhere a listed row could check, so a client showing a
    'running' row as live needs the heartbeat instant itself, not just the derived state, to
    say how stale that liveness claim already is."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.tools.experiment_tools import list_experiments

    instant = _beat(opened_run(None, _config(tmp_path), experiment_id="beating"), 180)

    by_id = {r["experiment_id"]: r for r in list_experiments(launched_only=True)["runs"]}
    assert by_id["beating"]["status"] == "running"
    assert by_id["beating"]["heartbeat"] == datetime.fromtimestamp(
        instant, timezone.utc).isoformat()


def test_configured_stale_window_agrees_across_run_list_compare_and_status(tmp_path, monkeypatch):
    """list_experiments(launched_only=True), compare_experiments and monitor_training derive
    "interrupted" the same way under a configured heartbeat window, one constant
    (``experiments.HEARTBEAT_STALE_SECONDS``) read by every consumer: a 300s-old heartbeat reads
    stale under a 30s window even though it would read fresh under the 600s default."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp import experiments
    from tcip_mcp.experiments import compare_experiments as compare_tool
    from tcip_mcp.tools.experiment_tools import list_experiments
    from tcip_mcp.tools.training_tools import monitor_training

    monkeypatch.setattr(experiments, "HEARTBEAT_STALE_SECONDS", 30.0)
    _beat(opened_run(None, _config(tmp_path), experiment_id="exp-window"), 300)

    by_id = {r["experiment_id"]: r for r in list_experiments(launched_only=True)["runs"]}
    assert by_id["exp-window"]["status"] == "interrupted"

    cmp = compare_tool(["exp-window"])
    assert cmp["experiments"][0]["state"] == "interrupted"

    assert monitor_training("exp-window")["status"] == "interrupted"
