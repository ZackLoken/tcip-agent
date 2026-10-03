"""Who launched a run, as its launch event states it (the agent identity the ``launch_training``
line naming the run carries, empty outside an MCP handshake), and the launch's refusals before
and after its directory exists.
"""

from __future__ import annotations

import pytest

from tests._producer_fixtures import seed_two_bud_images, small_detection_config

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


@pytest.fixture(autouse=True)
def _forget_agent_identity_between_tests():
    from tcip_mcp import agent_identity

    agent_identity.end()
    yield
    agent_identity.end()


def _fake_popen(monkeypatch: pytest.MonkeyPatch, captured: list[list[str]]) -> None:
    import subprocess

    class _FakeProc:
        pid = 424242

    def _popen(argv, **kwargs):
        captured.append(argv)
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _popen)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})


def _launch_record(project, experiment_id: str) -> dict:
    from tcip_mcp.experiments import RUN_FILE, experiment_dir, read_record

    return read_record(experiment_dir(experiment_id, project=project) / RUN_FILE)


def _row(project, experiment_id: str) -> dict:
    from tcip_mcp.tools import training_tools

    return next(r for r in training_tools._all_training_runs(project)
                if r["experiment_id"] == experiment_id)


def test_a_launch_no_agent_declared_itself_to_shows_an_empty_declaration(tmp_path, monkeypatch):
    """A launch outside an MCP handshake leaves its launch event carrying no identity, and that
    is what the run's row shows: the fact available, never a guess about who."""
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    seed_two_bud_images(images_dir, labels_dir)
    _fake_popen(monkeypatch, [])

    result = training_tools.launch_training(
        tmp_path, small_detection_config(images_dir, labels_dir, "exp-bare-launch"), actor=None)
    assert "error" not in result, result
    assert _row(tmp_path, "exp-bare-launch")["launch"] == {}
    assert "launched_by" not in _launch_record(tmp_path, "exp-bare-launch")


def test_a_launch_inside_an_mcp_handshake_shows_the_agents_declaration_from_its_event(
    tmp_path, monkeypatch,
):
    """A launch made while an MCP handshake is in force shows the connected agent's identity,
    read off the launch event that names the run, never off the run's own record."""
    monkeypatch.delenv("TCIP_TERMINAL_SESSION", raising=False)
    from tcip_mcp import agent_identity
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    seed_two_bud_images(images_dir, labels_dir)
    _fake_popen(monkeypatch, [])

    identity = agent_identity.begin("claude-code", "2.1.238")
    result = training_tools.launch_training(
        tmp_path, small_detection_config(images_dir, labels_dir, "exp-agent-launch"), actor=None)
    agent_identity.end()
    assert "error" not in result, result
    assert _row(tmp_path, "exp-agent-launch")["launch"] == {
        "agent_client_name": "claude-code", "agent_client_version": "2.1.238",
        "agent_session": identity.agent_session,
    }
    assert "launched_by" not in _launch_record(tmp_path, "exp-agent-launch")


def test_launch_refuses_a_dataset_identity_above_the_readers_ceiling(tmp_path, monkeypatch):
    """A version-refused dataset identity document refuses the launch by name, before the run's
    directory exists and before Popen is ever reached. The document is written through the
    store's own put_blob, the platform's own producer for a schema_version this reader does not
    accept. The admitting half is
    test_a_launch_no_agent_declared_itself_to_shows_an_empty_declaration above."""
    import tcip_store as ts

    from tcip_mcp.dataset_layout import dataset_identity_key
    from tcip_mcp.experiments import find_run
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    seed_two_bud_images(images_dir, labels_dir)
    captured: list[list[str]] = []
    _fake_popen(monkeypatch, captured)

    key = dataset_identity_key(tmp_path)
    document = {"crop": "test-crop", "id": "abc123", "fingerprint": "v1:deadbeef",
                "schema_version": 2}
    ts.put_blob(key, ts.RECORD_JSON.encode(document))

    result = training_tools.launch_training(
        tmp_path, small_detection_config(images_dir, labels_dir, "exp-identity-refused"), actor=None)

    assert result["error"].startswith("launch_training:")
    assert "schema_version" in result["error"]
    assert captured == []
    assert find_run("exp-identity-refused", project=tmp_path) is None


def test_launch_refuses_an_experiment_id_that_is_not_a_legal_directory_name(tmp_path, monkeypatch):
    """The id becomes the run's own directory: a path separator refuses by name, before any
    directory or process exists."""
    from tcip_mcp.experiments import run_dirs
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    seed_two_bud_images(images_dir, labels_dir)
    captured: list[list[str]] = []
    _fake_popen(monkeypatch, captured)

    result = training_tools.launch_training(
        tmp_path, small_detection_config(images_dir, labels_dir, "not/legal"), actor=None)

    assert result["error"].startswith("launch_training:")
    assert "not/legal" in result["error"]
    assert captured == []
    assert run_dirs(tmp_path) == []


def test_launch_refuses_when_the_launch_record_cannot_be_written(tmp_path, monkeypatch):
    """A launch record that cannot be written refuses the launch by name before the subprocess,
    and leaves no directory that reads as a run."""
    from tcip_store import StoreError

    from tcip_mcp import experiments as experiments_mod
    from tcip_mcp.experiments import find_run
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    seed_two_bud_images(images_dir, labels_dir)
    captured: list[list[str]] = []
    _fake_popen(monkeypatch, captured)

    def _raise(*args, **kwargs):
        raise StoreError("disk full")

    monkeypatch.setattr(experiments_mod, "write_once", _raise)

    result = training_tools.launch_training(
        tmp_path, small_detection_config(images_dir, labels_dir, "exp-write-raises"), actor=None)

    assert result["error"] == "launch_training: disk full"
    assert not any("tcip_mcp.pipelines.training.subprocess_worker" in argv for argv in captured)
    assert find_run("exp-write-raises", project=tmp_path) is None


def test_a_launch_records_the_seed_it_draws(tmp_path, monkeypatch):
    """No seed in the caller's config: the launch draws one onto the config its ``run.json``
    records, the one record the child trains from."""
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    seed_two_bud_images(images_dir, labels_dir)
    _fake_popen(monkeypatch, [])

    cfg = small_detection_config(images_dir, labels_dir, "exp-fresh-seed")
    result = training_tools.launch_training(tmp_path, cfg, actor=None)
    assert "error" not in result, result
    assert "seed" not in cfg
    assert isinstance(_launch_record(tmp_path, "exp-fresh-seed")["config"]["seed"], int)


def test_a_spawn_failure_leaves_a_directory_that_reads_interrupted(tmp_path, monkeypatch):
    """A Popen failure after the launch record propagates out of launch_training, the directory
    it opened reads interrupted once its heartbeat window has passed, and a relaunch under the
    same id refuses naming the existing directory."""
    import subprocess

    from tcip_mcp import experiments as experiments_mod
    from tcip_mcp.experiments import experiment_dir, observe
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    seed_two_bud_images(images_dir, labels_dir)

    def _raise_popen(*args, **kwargs):
        raise OSError("no such executable")

    monkeypatch.setattr(subprocess, "Popen", _raise_popen)

    with pytest.raises(OSError):
        training_tools.launch_training(
            tmp_path, small_detection_config(images_dir, labels_dir, "exp-spawn-fails"), actor=None)

    monkeypatch.setattr(experiments_mod, "HEARTBEAT_STALE_SECONDS", -1.0)
    assert observe(experiment_dir("exp-spawn-fails", project=tmp_path)).state == "interrupted"

    _fake_popen(monkeypatch, [])
    relaunch = training_tools.launch_training(
        tmp_path, small_detection_config(images_dir, labels_dir, "exp-spawn-fails"), actor=None)
    assert "already exists" in relaunch["error"]
