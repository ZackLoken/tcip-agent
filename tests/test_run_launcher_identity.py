"""Who launched a run, as its launch event states it, and the launch's refusals.

The launch event carries the agent identity on the ``launch_training`` line naming the run, empty
outside an MCP handshake; a launch record that cannot be written leaves no run directory, and a
spawn failure leaves one that reads interrupted.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tests._producer_fixtures import (
    fake_popen as _fake_popen,
    seed_bud_images,
    small_detection_config,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


@pytest.fixture(autouse=True)
def _forget_agent_identity_between_tests() -> Iterator[None]:
    from tcip_mcp import agent_identity

    agent_identity.end()
    yield
    agent_identity.end()


class NoExecutableChild:
    """A ``Popen`` stand-in whose construction raises ``OSError``, as a missing executable does."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise OSError("no such executable")


def _launch_record(project: Path, experiment_id: str) -> dict:
    from tcip_mcp.experiments import RUN_FILE, experiment_dir, read_record

    return read_record(
        experiment_dir(experiment_id, project=project) / RUN_FILE
    )


def _row(project: Path, experiment_id: str) -> dict:
    from tcip_mcp.experiments import run_rows

    return next(
        r for r in run_rows(project) if r.experiment_id == experiment_id
    ).model_dump()


def test_a_launch_no_agent_declared_itself_to_shows_an_empty_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launch outside an MCP handshake shows an empty identity on the run's row.

    The run's row shows an empty launch identity, and the launch record carries no
    ``launched_by``.
    """
    from tcip_mcp.tools import training_tools

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    seed_bud_images(images_dir, n=2, size=64, box=(1, 1, 9, 9))
    _fake_popen(monkeypatch, [])

    result = training_tools.launch_training(
        tmp_path,
        small_detection_config(images_dir, project=tmp_path),
        actor=None,
    )
    assert isinstance(result, training_tools.RunLaunched), result
    assert _row(tmp_path, result.experiment_id)["launch"] == {}
    assert "launched_by" not in _launch_record(tmp_path, result.experiment_id)


def test_a_launch_inside_an_mcp_handshake_shows_the_agents_declaration_from_its_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A launch inside an MCP handshake shows the connected agent's identity from its event.

    The run's row shows the agent's name, version and session, and the run's own record carries
    no ``launched_by``.
    """
    monkeypatch.delenv("TCIP_TERMINAL_SESSION", raising=False)
    from tcip_mcp import agent_identity
    from tcip_mcp.tools import training_tools

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    seed_bud_images(images_dir, n=2, size=64, box=(1, 1, 9, 9))
    _fake_popen(monkeypatch, [])

    identity = agent_identity.begin("claude-code", "2.1.238")
    result = training_tools.launch_training(
        tmp_path,
        small_detection_config(images_dir, project=tmp_path),
        actor=None,
    )
    agent_identity.end()
    assert isinstance(result, training_tools.RunLaunched), result
    assert _row(tmp_path, result.experiment_id)["launch"] == {
        "agent_client_name": "claude-code",
        "agent_client_version": "2.1.238",
        "agent_session": identity.agent_session,
    }
    assert "launched_by" not in _launch_record(tmp_path, result.experiment_id)


def test_launch_refuses_when_the_launch_record_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launch whose launch record cannot be written is refused, spawning no worker.

    The refusal names the failure and leaves no directory that reads as a run.
    """
    from tcip_mcp import experiments as experiments_mod
    from tcip_mcp.experiments import run_dirs
    from tcip_mcp.tools import training_tools
    from tcip_store import StoreError

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    seed_bud_images(images_dir, n=2, size=64, box=(1, 1, 9, 9))
    captured: list[list[str]] = []
    _fake_popen(monkeypatch, captured)

    def _raise(*args: object, **kwargs: object) -> None:
        raise StoreError("disk full")

    monkeypatch.setattr(experiments_mod, "write_once", _raise)

    result = training_tools.launch_training(
        tmp_path,
        small_detection_config(images_dir, project=tmp_path),
        actor=None,
    )

    assert result == {"error": "launch_training: disk full"}
    assert not any(
        "tcip_mcp.pipelines.training.subprocess_worker" in argv
        for argv in captured
    )
    assert run_dirs(tmp_path) == []


def test_a_launch_records_the_seed_its_config_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launch records the seed its config states in ``run.json``."""
    from tcip_mcp.tools import training_tools

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    seed_bud_images(images_dir, n=2, size=64, box=(1, 1, 9, 9))
    _fake_popen(monkeypatch, [])

    cfg = {
        **small_detection_config(images_dir, project=tmp_path),
        "seed": 1234,
    }
    result = training_tools.launch_training(tmp_path, cfg, actor=None)
    assert isinstance(result, training_tools.RunLaunched), result
    recorded = _launch_record(tmp_path, result.experiment_id)["config"]
    assert recorded["seed"] == 1234


def test_a_spawn_failure_leaves_a_directory_that_reads_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spawn failure propagates and leaves a directory that reads interrupted.

    A Popen failure after the launch record propagates out of launch_training, and the one
    directory it opened reads interrupted when its heartbeat window has passed.
    """
    from tcip_mcp import experiments as experiments_mod
    from tcip_mcp.experiments import observe, run_dirs
    from tcip_mcp.tools import training_tools

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    seed_bud_images(images_dir, n=2, size=64, box=(1, 1, 9, 9))

    monkeypatch.setattr(subprocess, "Popen", NoExecutableChild)

    with pytest.raises(OSError):
        training_tools.launch_training(
            tmp_path,
            small_detection_config(images_dir, project=tmp_path),
            actor=None,
        )

    monkeypatch.setattr(experiments_mod, "HEARTBEAT_STALE_SECONDS", -1.0)
    (opened,) = run_dirs(tmp_path)
    assert observe(opened).state == "interrupted"
