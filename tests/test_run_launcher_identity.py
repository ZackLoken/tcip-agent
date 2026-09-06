"""Who launched a run, stamped on the record: ``stamp_run_identity``'s ``launched_by`` and
``launch_training``'s resolution of it (a declared launcher, an MCP agent's identity, or a bare
process), read back through ``reconstruct_from_status``.
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


def test_stamp_run_identity_requires_launched_by(tmp_path, monkeypatch):
    """The keyword is required: a caller with nothing to declare must still say so explicitly,
    naming the fact available (``process``) rather than leaving the field to a default."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, stamp_run_identity

    create_experiment("exp-launcher-required", {"model_source": {"builder": "x:y"}})

    with pytest.raises(TypeError):
        stamp_run_identity("exp-launcher-required", "run-1", "out")  # type: ignore[call-arg]

    stamp_run_identity("exp-launcher-required", "run-1", "out", launched_by={"launcher": "process"})


def test_a_record_with_no_stamp_reconstructs_with_launched_by_none(tmp_path, monkeypatch):
    """A record built through create_experiment/update_status alone, never stamp_run_identity
    (the pre-field shape, or one whose best-effort stamp was dropped), reconstructs with
    launched_by absent rather than a guessed value."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import (
        create_experiment, is_launched, read_member, reconstruct_from_status, status_key,
        update_status,
    )

    create_experiment("exp-pre-field", {"model_source": {"builder": "x:y"}})
    update_status("exp-pre-field", "running")

    status = read_member(status_key("exp-pre-field"), {})
    assert is_launched(status)  # a launched run by state alone, with no run_id ever stamped

    row = reconstruct_from_status("exp-pre-field", status, stale_seconds=600.0, read_progress=False)
    assert row["launched_by"] is None


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
                         "builder_kwargs": {"num_classes": 1, "min_size": 64, "max_size": 128},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud"},
        "training": {"batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                     "mixed_precision": False, "device": "cpu"},
        "experiment_id": experiment_id,
    }


def _seed_one_image(images_dir, labels_dir) -> None:
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (32, 32)).save(images_dir / "img0.png")
    json_io.write_annotations(str(labels_dir / "img0.json"),
                              [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)


def test_a_bare_launch_writes_launcher_process(tmp_path, monkeypatch):
    """A launch through neither a declared launcher nor an MCP handshake stamps ``process``: the
    fact available, never a guess about who."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import read_member, status_key
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    _fake_popen(monkeypatch, [])

    result = training_tools.launch_training(
        _detection_cfg(images_dir, labels_dir, "exp-bare-launch"), str(tmp_path / "out"))
    assert "error" not in result, result

    status = read_member(status_key(result["experiment_id"]))
    assert status["launched_by"] == {"launcher": "process"}


def test_a_launch_inside_an_mcp_handshake_writes_launcher_agent_with_identity_fields(
    tmp_path, monkeypatch,
):
    """A launch made while an MCP handshake is in force stamps the connected agent's own
    identity, the same fields its audit line already carries."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TCIP_TERMINAL_SESSION", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    monkeypatch.delenv("CLAUDE_EFFORT", raising=False)
    from tcip_mcp import agent_identity
    from tcip_mcp.experiments import read_member, status_key
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    _fake_popen(monkeypatch, [])

    identity = agent_identity.begin("claude-code", "2.1.238")
    result = training_tools.launch_training(
        _detection_cfg(images_dir, labels_dir, "exp-agent-launch"), str(tmp_path / "out"))
    assert "error" not in result, result

    status = read_member(status_key(result["experiment_id"]))
    assert status["launched_by"] == {
        "launcher": "agent", "agent_client_name": "claude-code", "agent_client_version": "2.1.238",
        "agent_session": identity.session,
    }


def test_declare_launcher_stamps_the_declared_name(tmp_path, monkeypatch):
    """The mechanism the web route uses: a launch made inside ``declare_launcher(name)`` stamps
    that name whatever the connected identity, since a declaration on the calling thread takes
    precedence over the handshake fallback."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import read_member, status_key
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    _fake_popen(monkeypatch, [])

    with training_tools.declare_launcher("gui"):
        result = training_tools.launch_training(
            _detection_cfg(images_dir, labels_dir, "exp-gui-launch"), str(tmp_path / "out"))
    assert "error" not in result, result

    status = read_member(status_key(result["experiment_id"]))
    assert status["launched_by"] == {"launcher": "gui"}


def test_every_ensure_experiment_branch_stamps_the_same_launched_by(tmp_path, monkeypatch):
    """The fresh-creation, pristine-reuse and fresh-id-conflict branches of _ensure_experiment
    each stamp the launch's own resolved declaration, not only the fresh-creation branch."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, read_member, status_key
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    captured: list[list[str]] = []
    _fake_popen(monkeypatch, captured)

    with training_tools.declare_launcher("gui"):
        # Fresh creation.
        fresh = training_tools.launch_training(
            _detection_cfg(images_dir, labels_dir, "exp-branch-fresh"), str(tmp_path / "out1"))
        assert "error" not in fresh, fresh

        # Pristine pre-created reuse.
        create_experiment("exp-branch-pristine", {"a": 1})
        pristine = training_tools.launch_training(
            _detection_cfg(images_dir, labels_dir, "exp-branch-pristine"), str(tmp_path / "out2"))
        assert "error" not in pristine, pristine
        assert pristine["experiment_id"] == "exp-branch-pristine"

        # Fresh-id conflict: relaunching a config that already has a run mints <id>_<run_id>.
        forked = training_tools.launch_training(
            _detection_cfg(images_dir, labels_dir, "exp-branch-pristine"), str(tmp_path / "out3"))
        assert "error" not in forked, forked
        assert forked["experiment_id"] != "exp-branch-pristine"

    for result in (fresh, pristine, forked):
        status = read_member(status_key(result["experiment_id"]))
        assert status["launched_by"] == {"launcher": "gui"}


def test_all_training_runs_reads_launched_by_from_the_record_not_the_live_row(tmp_path, monkeypatch):
    """A live row's launched_by is read fresh from the experiment's own status record, never
    carried on the in-memory TrainRun: a run whose stamp failed (or was never made) renders the
    same way any other unstamped record does, launcher not recorded, rather than a caller's
    in-memory intent that the record itself never held."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.tools.training_tools import _all_training_runs

    run = create_run({"model_source": {"builder": "x:y"}}, str(tmp_path / "out"))
    run.experiment_id = "exp-stamp-write-failed"

    # create_experiment/update_status alone, never stamp_run_identity: the shape a failed or
    # dropped stamp leaves behind.
    create_experiment("exp-stamp-write-failed", {"model_source": {"builder": "x:y"}})
    update_status("exp-stamp-write-failed", "running")

    rows = _all_training_runs(read_progress=False)
    row = next(r for r in rows if r["run_id"] == run.run_id)
    assert row["launched_by"] is None
