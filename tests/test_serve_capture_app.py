"""tools/serve_capture_app.py's environment builder.

Covers the scratch variables it sets and the roots it refuses. No server is started here.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from tools import serve_capture_app

if TYPE_CHECKING:
    from pathlib import Path


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
