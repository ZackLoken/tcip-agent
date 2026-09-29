"""Who launched a run, written once into its ``run.json``: ``launch_training``'s resolution of
``launched_by`` (a declared launcher, an MCP agent's identity, or a bare process), and the launch's
refusals before and after its directory exists.
"""

from __future__ import annotations

import pytest

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


def _detection_cfg(images_dir, labels_dir, experiment_id: str) -> dict:
    return {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"min_size": 64, "max_size": 128},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                 "scope": {"subject": "bud"}},
        "batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
        "mixed_precision": False, "device": "cpu",
        "experiment_id": experiment_id,
    }


def _seed_one_image(images_dir, labels_dir) -> None:
    """Two labeled images, so a drawn split holds one out for validation."""
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir.mkdir()
    labels_dir.mkdir()
    for i in range(2):
        Image.new("RGB", (32, 32), color=(10 * i, 0, 0)).save(images_dir / f"img{i}.png")
        json_io.write_annotations(str(labels_dir / f"img{i}.json"),
                                  [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)


def _launch_record(experiment_id: str) -> dict:
    from tcip_mcp.experiments import RUN_FILE, experiment_dir, read_record

    return read_record(experiment_dir(experiment_id) / RUN_FILE)


def _launched_by(experiment_id: str) -> dict:
    return _launch_record(experiment_id)["launched_by"]


def test_a_bare_launch_writes_launcher_process(tmp_path, monkeypatch):
    """A launch through neither a declared launcher nor an MCP handshake records ``process``: the
    fact available, never a guess about who."""
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    _fake_popen(monkeypatch, [])

    result = training_tools.launch_training(
        _detection_cfg(images_dir, labels_dir, "exp-bare-launch"))
    assert "error" not in result, result
    assert _launched_by(result["experiment_id"]) == {"launcher": "process"}


def test_a_launch_inside_an_mcp_handshake_writes_launcher_agent_with_identity_fields(
    tmp_path, monkeypatch,
):
    """A launch made while an MCP handshake is in force records the connected agent's own
    identity, the same fields its audit line already carries."""
    monkeypatch.delenv("TCIP_TERMINAL_SESSION", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    monkeypatch.delenv("CLAUDE_EFFORT", raising=False)
    from tcip_mcp import agent_identity
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    _fake_popen(monkeypatch, [])

    identity = agent_identity.begin("claude-code", "2.1.238")
    result = training_tools.launch_training(
        _detection_cfg(images_dir, labels_dir, "exp-agent-launch"))
    assert "error" not in result, result
    assert _launched_by(result["experiment_id"]) == {
        "launcher": "agent", "agent_client_name": "claude-code", "agent_client_version": "2.1.238",
        "agent_session": identity.session,
    }


def test_declare_launcher_stamps_the_declared_name(tmp_path, monkeypatch):
    """A launch made inside ``declare_launcher(name)`` records that name whatever the connected
    identity: an identity is begun on this thread first, so the precedence branch of
    ``_resolve_launched_by`` is the one exercised."""
    from tcip_mcp import agent_identity
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    _fake_popen(monkeypatch, [])

    agent_identity.begin("claude-code", "2.1.238")
    try:
        with training_tools.declare_launcher("gui"):
            result = training_tools.launch_training(
                _detection_cfg(images_dir, labels_dir, "exp-gui-launch"))
    finally:
        agent_identity.end()
    assert "error" not in result, result
    assert _launched_by(result["experiment_id"]) == {"launcher": "gui"}


def test_all_training_runs_reads_launched_by_from_the_runs_own_record(tmp_path, monkeypatch):
    """The Training view's row for a launched run carries the ``launched_by`` its ``run.json``
    holds."""
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    _fake_popen(monkeypatch, [])

    with training_tools.declare_launcher("gui"):
        result = training_tools.launch_training(
            _detection_cfg(images_dir, labels_dir, "exp-row-launcher"))
    assert "error" not in result, result

    row = next(r for r in training_tools._all_training_runs()
               if r["experiment_id"] == "exp-row-launcher")
    assert row["launched_by"] == {"launcher": "gui"}


def test_launch_refuses_a_dataset_identity_above_the_readers_ceiling(tmp_path, monkeypatch):
    """A version-refused dataset identity document refuses the launch by name, before the run's
    directory exists and before Popen is ever reached. The document is written through the
    store's own put_blob, the platform's own producer for a schema_version this reader does not
    accept. The admitting half is test_a_bare_launch_writes_launcher_process above."""
    import tcip_store as ts

    from tcip_mcp.dataset_layout import dataset_identity_key
    from tcip_mcp.experiments import find_run
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    captured: list[list[str]] = []
    _fake_popen(monkeypatch, captured)

    key = dataset_identity_key(tmp_path)
    document = {"crop": "test-crop", "id": "abc123", "fingerprint": "v1:deadbeef",
                "schema_version": 2}
    ts.put_blob(key, ts.RECORD_JSON.encode(document))

    result = training_tools.launch_training(
        _detection_cfg(images_dir, labels_dir, "exp-identity-refused"))

    assert result["error"].startswith("launch_training:")
    assert "schema_version" in result["error"]
    assert captured == []
    assert find_run("exp-identity-refused") is None


def test_launch_refuses_an_experiment_id_that_is_not_a_legal_directory_name(tmp_path, monkeypatch):
    """The id becomes the run's own directory: a path separator refuses by name, before any
    directory or process exists."""
    from tcip_mcp.experiments import run_dirs
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    captured: list[list[str]] = []
    _fake_popen(monkeypatch, captured)

    result = training_tools.launch_training(_detection_cfg(images_dir, labels_dir, "not/legal"))

    assert result["error"].startswith("launch_training:")
    assert "not/legal" in result["error"]
    assert captured == []
    assert run_dirs() == []


def test_launch_refuses_when_the_launch_record_cannot_be_written(tmp_path, monkeypatch):
    """A launch record that cannot be written refuses the launch by name before the subprocess,
    and leaves no directory that reads as a run."""
    from tcip_store import StoreError

    from tcip_mcp import experiments as experiments_mod
    from tcip_mcp.experiments import find_run
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    captured: list[list[str]] = []
    _fake_popen(monkeypatch, captured)

    def _raise(*args, **kwargs):
        raise StoreError("disk full")

    monkeypatch.setattr(experiments_mod, "write_once", _raise)

    result = training_tools.launch_training(
        _detection_cfg(images_dir, labels_dir, "exp-write-raises"))

    assert result["error"] == "launch_training: disk full"
    assert not any("tcip_mcp.pipelines.training.subprocess_worker" in argv for argv in captured)
    assert find_run("exp-write-raises") is None


def test_a_launch_records_the_seed_it_draws(tmp_path, monkeypatch):
    """No seed in the caller's config: the launch draws one onto the config its ``run.json``
    records, the one record the child trains from."""
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    _fake_popen(monkeypatch, [])

    cfg = _detection_cfg(images_dir, labels_dir, "exp-fresh-seed")
    result = training_tools.launch_training(cfg)
    assert "error" not in result, result
    assert "seed" not in cfg
    assert isinstance(_launch_record("exp-fresh-seed")["config"]["seed"], int)


def test_a_spawn_failure_leaves_a_directory_that_reads_interrupted(tmp_path, monkeypatch):
    """A Popen failure after the launch record propagates out of launch_training, the directory
    it opened reads interrupted once its heartbeat window has passed, and a relaunch under the
    same id refuses naming the existing directory."""
    import subprocess

    from tcip_mcp import experiments as experiments_mod
    from tcip_mcp.experiments import experiment_dir, observe
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)

    def _raise_popen(*args, **kwargs):
        raise OSError("no such executable")

    monkeypatch.setattr(subprocess, "Popen", _raise_popen)

    with pytest.raises(OSError):
        training_tools.launch_training(_detection_cfg(images_dir, labels_dir, "exp-spawn-fails"))

    monkeypatch.setattr(experiments_mod, "HEARTBEAT_STALE_SECONDS", -1.0)
    assert observe(experiment_dir("exp-spawn-fails")).state == "interrupted"

    _fake_popen(monkeypatch, [])
    relaunch = training_tools.launch_training(
        _detection_cfg(images_dir, labels_dir, "exp-spawn-fails"))
    assert "already exists" in relaunch["error"]
