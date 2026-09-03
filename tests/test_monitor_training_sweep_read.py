"""``monitor_training(sweep_id=)``: the sweep read a non-Claude host reaches through the same
tool used for training runs, over the identical disk-only reader ``routes.tuning`` composes on
(``training_tools.read_sweep_from_disk``), so the two never drift into two answers for one
question.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app

from tests.test_tcip_web_tuning_routes import _write_sweep, _write_trial


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture
def hpo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the platform state root at a tmp dir so the tool and the route read the same sweeps."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    root = tmp_path / ".tcip" / "hpo"
    root.mkdir(parents=True)
    return root


def test_monitor_training_refuses_both_run_id_and_sweep_id() -> None:
    from tcip_mcp.tools.training_tools import monitor_training

    res = monitor_training(run_id="run-1", sweep_id="sweep-1")
    assert "error" in res
    assert "run_id" in res["error"]
    assert "sweep_id" in res["error"]


def test_monitor_training_refuses_neither_run_id_nor_sweep_id() -> None:
    from tcip_mcp.tools.training_tools import monitor_training

    res = monitor_training()
    assert "error" in res
    assert "run_id" in res["error"]
    assert "sweep_id" in res["error"]


def test_monitor_training_sweep_id_matches_the_tuning_route_disk_read(client, hpo_root) -> None:
    from tcip_mcp.tools.training_tools import monitor_training

    sweep = _write_sweep(hpo_root, "hpo_sweep0001", status="completed",
                         result={"best_params": {"lr": 0.01}, "best_value": 0.2})
    _write_trial(sweep, "0", params={"lr": 0.01}, metrics=[{"epoch": 1, "loss": 0.1}])

    tool_result = monitor_training(sweep_id="hpo_sweep0001")
    route_result = client.get("/api/tuning/sweeps/hpo_sweep0001").json()

    assert tool_result["status"] == "completed"
    assert tool_result["status"] == route_result["status"]
    assert tool_result["manifest"] == route_result["manifest"]
    assert tool_result["trials"] == route_result["trials"]
    assert tool_result["result"] == route_result["result"]
    assert tool_result["trials"][0]["trial_id"] == "0"
    assert tool_result["trials"][0]["params"] == {"lr": 0.01}


def test_monitor_training_sweep_id_not_found_names_the_sweep() -> None:
    from tcip_mcp.tools.training_tools import monitor_training

    res = monitor_training(sweep_id="no-such-sweep")
    assert "error" in res
    assert "no-such-sweep" in res["error"]
