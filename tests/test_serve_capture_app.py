"""tools/serve_capture_app.py: the environment it builds, the roots it refuses, and its server.

The environment and refusal tests start no server and run on every host. The tests marked
``ON_WINDOWS`` start real backends over scratch harness roots through the tool's command line,
since the tool ends one with ``taskkill /T``; each ends whatever backend it left running.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import psutil
import pytest

from tcip_mcp.web_client import (
    free_port,
    record_backend_port,
    resolve_web_port,
)
from tests._scratch_server_fixtures import (
    ON_WINDOWS,
    answered_workspace,
    end_served_processes,
    serve,
    served_processes,
    started,
    stopped,
)
from tests._web_fixtures import new_project
from tools import serve_capture_app

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def make_root(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Callable[[], Path]]:
    """Yield a maker of fresh harness roots outside the test's own workspace.

    Every backend still running over a root it made is ended after the test.
    """
    made: list[Path] = []

    def make() -> Path:
        made.append(tmp_path_factory.mktemp("harness").resolve())
        return made[-1]

    yield make
    for root in made:
        end_served_processes(root)


@pytest.fixture
def harness(make_root: Callable[[], Path]) -> Path:
    """Return one fresh harness root (:func:`make_root`)."""
    return make_root()


def test_the_workspace_and_port_point_under_the_given_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The built environment sets the workspace under the root, the requested port, unbuffered.

    ``TCIP_WORKSPACE`` is ``<root>/workspace``, ``TCIP_WEB_PORT`` is ``"8799"`` and
    ``PYTHONUNBUFFERED`` is ``"1"``.
    """
    monkeypatch.delenv("TCIP_WORKSPACE", raising=False)
    root = tmp_path / "harness"
    env = serve_capture_app.build_environ(root, 8799)

    assert env["TCIP_WORKSPACE"] == str(root / "workspace")
    assert env["TCIP_WEB_PORT"] == "8799"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_pythonpath_carries_the_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The built ``PYTHONPATH`` leads with the repository root and keeps its existing entries."""
    monkeypatch.delenv("TCIP_WORKSPACE", raising=False)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "existing"))
    root = tmp_path / "harness"

    env = serve_capture_app.build_environ(root, 8799)

    parts = env["PYTHONPATH"].split(os.pathsep)
    assert str(serve_capture_app.REPO_ROOT) == parts[0]
    assert str(tmp_path / "existing") in parts


def test_a_root_under_the_repository_is_refused() -> None:
    root = serve_capture_app.REPO_ROOT / "scratch_harness"
    with pytest.raises(SystemExit, match="is under the repository"):
        serve_capture_app.build_environ(root, 8799)


def test_the_repository_root_itself_is_refused() -> None:
    with pytest.raises(SystemExit, match="is under the repository"):
        serve_capture_app.build_environ(serve_capture_app.REPO_ROOT, 8799)


def test_a_root_under_the_callers_own_active_workspace_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active_workspace = tmp_path / "S" / "real-project"
    monkeypatch.setenv("TCIP_WORKSPACE", str(active_workspace))
    root = active_workspace / "harness"

    with pytest.raises(SystemExit, match="own TCIP_WORKSPACE"):
        serve_capture_app.build_environ(root, 8799)


def test_a_root_outside_both_is_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "elsewhere"))
    root = tmp_path / "harness"

    env = serve_capture_app.build_environ(root, 8799)

    assert env["TCIP_WORKSPACE"] == str(root / "workspace")


def test_a_root_containing_the_callers_own_active_workspace_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root containing the process's own active workspace is refused.

    A root holding the process's own TCIP_WORKSPACE as a subdirectory is refused naming that
    workspace variable.
    """
    root = tmp_path / "harness"
    active_workspace = root / "nested" / "real-project"
    monkeypatch.setenv("TCIP_WORKSPACE", str(active_workspace))

    with pytest.raises(SystemExit, match="own TCIP_WORKSPACE"):
        serve_capture_app.build_environ(root, 8799)


def test_a_root_that_already_carries_its_own_tcip_state_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TCIP_WORKSPACE", raising=False)
    root = tmp_path / "harness"
    (root / ".tcip").mkdir(parents=True)

    with pytest.raises(SystemExit, match="already looks like a workspace"):
        serve_capture_app.build_environ(root, 8799)


def test_a_root_whose_child_already_carries_tcip_state_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TCIP_WORKSPACE", raising=False)
    root = tmp_path / "harness"
    (root / "existing_project" / ".tcip").mkdir(parents=True)

    with pytest.raises(SystemExit, match="already looks like a workspace"):
        serve_capture_app.build_environ(root, 8799)


def test_a_fresh_root_with_no_tcip_workspace_bound_is_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh root with no workspace bound and no ``.tcip`` under it is admitted."""
    monkeypatch.delenv("TCIP_WORKSPACE", raising=False)
    root = tmp_path / "harness"

    env = serve_capture_app.build_environ(root, 8799)

    assert env["TCIP_WORKSPACE"] == str(root / "workspace")


def test_stop_refuses_outright_on_a_non_windows_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(serve_capture_app.sys, "platform", "linux")

    with pytest.raises(SystemExit, match="Windows-only"):
        serve_capture_app.stop(tmp_path)


def test_stop_is_not_refused_for_platform_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a Windows host, stop passes the platform check and reaches its real work.

    It reaches the missing server_info.json refusal.
    """
    monkeypatch.setattr(serve_capture_app.sys, "platform", "win32")

    with pytest.raises(SystemExit, match="no server_info.json"):
        serve_capture_app.stop(tmp_path)


def test_the_callers_image_roots_do_not_reach_the_scratch_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``TCIP_IMAGE_ROOTS`` in the caller's environment is absent from the server's."""
    monkeypatch.delenv("TCIP_WORKSPACE", raising=False)
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(tmp_path))

    env = serve_capture_app.build_environ(tmp_path / "harness", 8799)

    assert "TCIP_IMAGE_ROOTS" not in env


def test_a_workspace_linked_outside_the_root_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TCIP_WORKSPACE", raising=False)
    root = tmp_path / "harness"
    root.mkdir()
    (tmp_path / "elsewhere").mkdir()
    os.symlink(
        tmp_path / "elsewhere", root / "workspace", target_is_directory=True
    )

    with pytest.raises(SystemExit, match="resolves outside"):
        serve_capture_app.build_environ(root, 8799)


def test_a_project_linked_into_the_workspace_from_outside_the_root_is_refused(
    project: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A project made by ``initialize_project`` outside the root, linked into its workspace."""
    root = tmp_path_factory.mktemp("harness").resolve()
    (root / "workspace").mkdir()
    os.symlink(
        project, root / "workspace" / "linked", target_is_directory=True
    )

    with pytest.raises(SystemExit, match="resolves outside"):
        serve_capture_app.build_environ(root, 8799)


def test_a_root_whose_workspace_holds_the_backends_own_port_record_is_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What a first start leaves, the backend's port record under the workspace, refuses nothing.

    The record is written through the backend's own writer, ``record_backend_port``.
    """
    monkeypatch.delenv("TCIP_WORKSPACE", raising=False)
    root = tmp_path / "harness"
    (root / "workspace").mkdir(parents=True)
    record_backend_port(root / "workspace", 8799)
    assert (root / "workspace" / ".tcip").is_dir()

    env = serve_capture_app.build_environ(root, 8799)

    assert env["TCIP_WORKSPACE"] == str(root / "workspace")


def test_start_refuses_a_host_it_cannot_stop_on_before_starting_anything(
    harness: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(serve_capture_app.sys, "platform", "linux")

    with pytest.raises(SystemExit, match="Windows-only"):
        serve_capture_app.start(harness, free_port(0))
    assert not (harness / "server.log").exists()
    assert served_processes(harness) == []


@ON_WINDOWS
def test_a_start_naming_no_port_serves_its_own_workspace_on_the_port_the_backend_recorded(
    harness: Path,
) -> None:
    info = started(str(harness))
    workspace = harness / "workspace"

    assert info["port"] == resolve_web_port(workspace)
    assert os.path.samefile(answered_workspace(info["port"]), workspace)
    stopped(str(harness))
    assert served_processes(harness) == []


@ON_WINDOWS
def test_a_start_naming_a_free_port_serves_its_own_workspace_there(
    harness: Path,
) -> None:
    port = free_port(0)

    info = started(str(harness), "--port", str(port))

    assert info["port"] == port
    assert os.path.samefile(answered_workspace(port), harness / "workspace")
    stopped(str(harness))


@ON_WINDOWS
def test_a_port_another_backend_holds_is_never_taken_for_the_started_server(
    harness: Path, make_root: Callable[[], Path]
) -> None:
    """Asked for the port a second scratch backend holds, the start serves on its own port.

    The second backend still answers for its own workspace afterwards.
    """
    other = make_root()
    taken = started(str(other), "--port", str(free_port(0)))["port"]

    info = started(str(harness), "--port", str(taken))

    assert info["port"] != taken
    assert os.path.samefile(
        answered_workspace(info["port"]), harness / "workspace"
    )
    assert os.path.samefile(answered_workspace(taken), other / "workspace")
    stopped(str(harness))
    stopped(str(other))


@ON_WINDOWS
def test_an_answer_for_another_workspace_is_refused_and_leaves_that_backend_running(
    harness: Path, make_root: Callable[[], Path]
) -> None:
    other = make_root()
    taken = started(str(other), "--port", str(free_port(0)))["port"]
    (harness / "workspace").mkdir()

    with pytest.raises(SystemExit, match="answers for"):
        serve_capture_app.answers_for(taken, harness / "workspace")
    assert os.path.samefile(answered_workspace(taken), other / "workspace")
    stopped(str(other))


@ON_WINDOWS
def test_a_root_named_by_a_relative_path_serves_that_roots_workspace(
    harness: Path,
) -> None:
    info = started(
        harness.name, "--port", str(free_port(0)), cwd=harness.parent
    )

    assert os.path.samefile(
        answered_workspace(info["port"]), harness / "workspace"
    )
    stopped(harness.name, cwd=harness.parent)


@ON_WINDOWS
def test_a_start_whose_server_exits_early_leaves_no_process(
    harness: Path,
) -> None:
    """A port no socket can bind ends the backend at its start; the refusal says so."""
    done = serve("start", str(harness), "--port", "70000")

    assert done.returncode != 0
    assert "exited early" in done.stderr
    assert served_processes(harness) == []
    assert not (harness / "server_info.json").exists()


@ON_WINDOWS
def test_a_start_that_times_out_ends_the_process_it_started(
    harness: Path,
) -> None:
    with pytest.raises(SystemExit, match="did not answer"):
        serve_capture_app.start(harness, None, timeout_s=0)
    assert served_processes(harness) == []
    assert not (harness / "server_info.json").exists()


@ON_WINDOWS
def test_a_stop_whose_kill_fails_refuses_and_keeps_the_record(
    harness: Path,
) -> None:
    """A stop whose server is already gone cannot kill it: it refuses naming the process."""
    started(str(harness), "--port", str(free_port(0)))
    record = json.loads(
        (harness / "server_info.json").read_text(encoding="utf-8")
    )
    served = psutil.Process(record["pid"])
    served.kill()
    served.wait(timeout=30)

    done = serve("stop", str(harness))

    assert done.returncode != 0
    assert f"process {record['pid']}" in done.stderr
    assert (harness / "server_info.json").is_file()


@ON_WINDOWS
def test_a_second_start_and_stop_over_a_used_root_succeed(
    harness: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no workspace bound to the caller, a used root holding a project starts again."""
    monkeypatch.delenv("TCIP_WORKSPACE", raising=False)
    (harness / "workspace" / "orchard").mkdir(parents=True)
    new_project(harness / "workspace" / "orchard")
    for _ in range(2):
        info = started(str(harness), "--port", str(free_port(0)))
        assert os.path.samefile(
            answered_workspace(info["port"]), harness / "workspace"
        )
        stopped(str(harness))
    assert served_processes(harness) == []
