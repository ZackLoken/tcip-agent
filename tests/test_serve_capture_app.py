"""tools/serve_capture_app.py's environment builder: the scratch variables it sets, and the
roots it refuses before setting anything. No server is started here."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "tools" / "serve_capture_app.py"


def _load():
    spec = importlib.util.spec_from_file_location("serve_capture_app", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["serve_capture_app"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def tool():
    return _load()


def test_the_two_variables_point_under_the_given_root(tool, tmp_path, monkeypatch):
    # The ambient test rail may itself bind TCIP_WORKSPACE under this very tmp_path; a root
    # this test controls directly must not accidentally trip that unrelated refusal.
    monkeypatch.delenv("TCIP_WORKSPACE", raising=False)
    root = tmp_path / "harness"
    env = tool.build_environ(root, "my_project", 8799)

    assert env["TCIP_WORKSPACE"] == str(root / "workspace")
    assert env["TCIP_STATE_ROOT"] == str(root / "workspace" / "my_project")
    assert env["TCIP_WEB_PORT"] == "8799"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_pythonpath_carries_the_repository_root(tool, tmp_path, monkeypatch):
    """A seeded run's subprocess imports a fixture module (a tiny trainer, say) by dotted name
    against the repository, the same way the day-3 harness's own environment did."""
    monkeypatch.delenv("TCIP_WORKSPACE", raising=False)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "existing"))
    root = tmp_path / "harness"

    env = tool.build_environ(root, "my_project", 8799)

    parts = env["PYTHONPATH"].split(os.pathsep)
    assert str(tool.REPO_ROOT) == parts[0]
    assert str(tmp_path / "existing") in parts


def test_a_root_under_the_repository_is_refused(tool):
    root = tool.REPO_ROOT / "scratch_harness"
    with pytest.raises(SystemExit, match="is under the repository"):
        tool.build_environ(root, "my_project", 8799)


def test_the_repository_root_itself_is_refused(tool):
    with pytest.raises(SystemExit, match="is under the repository"):
        tool.build_environ(tool.REPO_ROOT, "my_project", 8799)


def test_a_root_under_the_callers_own_active_workspace_is_refused(tool, tmp_path, monkeypatch):
    active_workspace = tmp_path / "S" / "real-project"
    monkeypatch.setenv("TCIP_WORKSPACE", str(active_workspace))
    root = active_workspace / "harness"

    with pytest.raises(SystemExit, match="own TCIP_WORKSPACE"):
        tool.build_environ(root, "my_project", 8799)


def test_a_root_outside_both_is_admitted(tool, tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "elsewhere"))
    root = tmp_path / "harness"

    env = tool.build_environ(root, "my_project", 8799)

    assert env["TCIP_STATE_ROOT"] == str(root / "workspace" / "my_project")


def test_a_root_containing_the_callers_own_active_workspace_is_refused(tool, tmp_path, monkeypatch):
    """The old check only refused the root-under-workspace direction; a root that itself
    contains the caller's own active TCIP_WORKSPACE as a subdirectory aliases it just as much."""
    root = tmp_path / "harness"
    active_workspace = root / "nested" / "real-project"
    monkeypatch.setenv("TCIP_WORKSPACE", str(active_workspace))

    with pytest.raises(SystemExit, match="own TCIP_WORKSPACE"):
        tool.build_environ(root, "my_project", 8799)


def test_a_root_that_already_carries_its_own_tcip_state_is_refused(tool, tmp_path, monkeypatch):
    monkeypatch.delenv("TCIP_WORKSPACE", raising=False)
    root = tmp_path / "harness"
    (root / ".tcip").mkdir(parents=True)

    with pytest.raises(SystemExit, match="already looks like a workspace"):
        tool.build_environ(root, "my_project", 8799)


def test_a_root_whose_child_already_carries_tcip_state_is_refused(tool, tmp_path, monkeypatch):
    monkeypatch.delenv("TCIP_WORKSPACE", raising=False)
    root = tmp_path / "harness"
    (root / "existing_project" / ".tcip").mkdir(parents=True)

    with pytest.raises(SystemExit, match="already looks like a workspace"):
        tool.build_environ(root, "my_project", 8799)


def test_a_fresh_root_with_no_tcip_workspace_bound_is_admitted(tool, tmp_path, monkeypatch):
    """A legitimate call still succeeds: a brand-new root, with no TCIP_WORKSPACE bound and no
    .tcip anywhere under it, is not refused."""
    monkeypatch.delenv("TCIP_WORKSPACE", raising=False)
    root = tmp_path / "harness"

    env = tool.build_environ(root, "my_project", 8799)

    assert env["TCIP_STATE_ROOT"] == str(root / "workspace" / "my_project")
