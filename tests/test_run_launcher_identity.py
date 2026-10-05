"""Who launched a run, as its launch event states it (the agent identity the ``launch_training``
line naming the run carries, empty outside an MCP handshake), and the launch's refusals before
and after its directory exists.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

import pytest

from tests._producer_fixtures import fake_popen as _fake_popen
from tests._producer_fixtures import seed_bud_images, small_detection_config

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


@pytest.fixture(autouse=True)
def _forget_agent_identity_between_tests():
    from tcip_mcp import agent_identity

    agent_identity.end()
    yield
    agent_identity.end()


def _launch_record(project, experiment_id: str) -> dict:
    from tcip_mcp.experiments import RUN_FILE, experiment_dir, read_record

    return read_record(experiment_dir(experiment_id, project=project) / RUN_FILE)


def _row(project, experiment_id: str) -> dict:
    from tcip_mcp.experiments import run_rows

    return next(r for r in run_rows(project) if r.experiment_id == experiment_id).model_dump()


def test_a_launch_no_agent_declared_itself_to_shows_an_empty_declaration(tmp_path, monkeypatch):
    """A launch outside an MCP handshake leaves its launch event carrying no identity, and that
    is what the run's row shows: the fact available, never a guess about who."""
    from tcip_mcp.tools import training_tools

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    seed_bud_images(images_dir, n=2, size=32, box=(1, 1, 9, 9))
    _fake_popen(monkeypatch, [])

    result = training_tools.launch_training(
        tmp_path, small_detection_config(images_dir), actor=None)
    assert "error" not in result, result
    assert _row(tmp_path, result["experiment_id"])["launch"] == {}
    assert "launched_by" not in _launch_record(tmp_path, result["experiment_id"])


def test_a_launch_inside_an_mcp_handshake_shows_the_agents_declaration_from_its_event(
    tmp_path, monkeypatch,
):
    """A launch made while an MCP handshake is in force shows the connected agent's identity,
    read off the launch event that names the run, never off the run's own record."""
    monkeypatch.delenv("TCIP_TERMINAL_SESSION", raising=False)
    from tcip_mcp import agent_identity
    from tcip_mcp.tools import training_tools

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    seed_bud_images(images_dir, n=2, size=32, box=(1, 1, 9, 9))
    _fake_popen(monkeypatch, [])

    identity = agent_identity.begin("claude-code", "2.1.238")
    result = training_tools.launch_training(
        tmp_path, small_detection_config(images_dir), actor=None)
    agent_identity.end()
    assert "error" not in result, result
    assert _row(tmp_path, result["experiment_id"])["launch"] == {
        "agent_client_name": "claude-code", "agent_client_version": "2.1.238",
        "agent_session": identity.agent_session,
    }
    assert "launched_by" not in _launch_record(tmp_path, result["experiment_id"])


def test_launch_refuses_when_the_launch_record_cannot_be_written(tmp_path, monkeypatch):
    """A launch record that cannot be written refuses the launch by name before the subprocess,
    and leaves no directory that reads as a run."""
    from tcip_store import StoreError

    from tcip_mcp import experiments as experiments_mod
    from tcip_mcp.experiments import run_dirs
    from tcip_mcp.tools import training_tools

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    seed_bud_images(images_dir, n=2, size=32, box=(1, 1, 9, 9))
    captured: list[list[str]] = []
    _fake_popen(monkeypatch, captured)

    def _raise(*args, **kwargs):
        raise StoreError("disk full")

    monkeypatch.setattr(experiments_mod, "write_once", _raise)

    result = training_tools.launch_training(
        tmp_path, small_detection_config(images_dir), actor=None)

    assert result["error"] == "launch_training: disk full"
    assert not any("tcip_mcp.pipelines.training.subprocess_worker" in argv for argv in captured)
    assert run_dirs(tmp_path) == []


def test_a_launch_records_the_seed_it_draws(tmp_path, monkeypatch):
    """No seed in the caller's config: the launch draws one onto the config its ``run.json``
    records, the one record the child trains from."""
    from tcip_mcp.tools import training_tools

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    seed_bud_images(images_dir, n=2, size=32, box=(1, 1, 9, 9))
    _fake_popen(monkeypatch, [])

    cfg = small_detection_config(images_dir)
    result = training_tools.launch_training(tmp_path, cfg, actor=None)
    assert "error" not in result, result
    assert "seed" not in cfg
    assert isinstance(_launch_record(tmp_path, result["experiment_id"])["config"]["seed"], int)


def test_a_spawn_failure_leaves_a_directory_that_reads_interrupted(tmp_path, monkeypatch):
    """A Popen failure after the launch record propagates out of launch_training, and the one
    directory it opened reads interrupted once its heartbeat window has passed."""
    import subprocess

    from tcip_mcp import experiments as experiments_mod
    from tcip_mcp.experiments import observe, run_dirs
    from tcip_mcp.tools import training_tools

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    seed_bud_images(images_dir, n=2, size=32, box=(1, 1, 9, 9))

    def _raise_popen(*args, **kwargs):
        raise OSError("no such executable")

    monkeypatch.setattr(subprocess, "Popen", _raise_popen)

    with pytest.raises(OSError):
        training_tools.launch_training(
            tmp_path, small_detection_config(images_dir), actor=None)

    monkeypatch.setattr(experiments_mod, "HEARTBEAT_STALE_SECONDS", -1.0)
    (opened,) = run_dirs(tmp_path)
    assert observe(opened).state == "interrupted"
