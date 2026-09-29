"""A run named by its own id is monitored, canceled and streamed from its directory alone.

There is one id: a training run's id is its directory's name, so any process reaches a run it
never launched by that id, straight off the directory, with no resolver and no second id.
"""

from __future__ import annotations

from tests._verified_checkpoint_fixtures import detection_config, log_epoch, opened_run


def _config(tmp_path) -> dict:
    """A detector run's config over two frames of its own under ``tmp_path``."""
    return detection_config(tmp_path / "data",
                            model_source={"builder": "my_models:build_detector",
                                          "task": "detection"})


def test_a_custom_named_run_is_monitored_by_its_own_id(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.tools.training_tools import monitor_training

    run_dir = opened_run(None, _config(tmp_path), experiment_id="exp-001-bud-det")
    log_epoch(run_dir, 3, {"val_map50": 0.4})

    result = monitor_training("exp-001-bud-det")
    assert result["status"] == "running"
    assert result["epoch"] == 3
    assert result["output_dir"] == str(run_dir)


def test_a_custom_named_run_is_canceled_by_its_own_id(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import cancel_requested
    from tcip_mcp.tools.training_tools import cancel_training

    run_dir = opened_run(None, _config(tmp_path), experiment_id="exp-002-bud-det")

    result = cancel_training("exp-002-bud-det")
    assert result["cancel_requested"] is True
    assert result["experiment_id"] == "exp-002-bud-det"
    assert cancel_requested(run_dir)


def test_an_id_naming_no_run_resolves_to_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import find_run
    from tcip_mcp.tools.training_tools import cancel_training, monitor_training

    opened_run(None, _config(tmp_path), experiment_id="exp-004-bud-det")

    assert find_run("exp-never-launched") is None
    assert "error" in monitor_training("exp-never-launched")
    assert "error" in cancel_training("exp-never-launched")
