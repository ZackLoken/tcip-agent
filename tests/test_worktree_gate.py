"""tools/worktree_gate.py's environment builder and its refusal before any gate runs when
tcip_mcp does not resolve inside the given worktree. No real gate (ruff/mypy/pytest) runs here."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "tools" / "worktree_gate.py"


def _load():
    spec = importlib.util.spec_from_file_location("worktree_gate", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["worktree_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def tool():
    return _load()


def _sep() -> str:
    return ";" if sys.platform == "win32" else ":"


def test_build_environ_sets_pythonpath_to_the_four_package_src_dirs_under_the_worktree(tool, tmp_path):
    worktree = tmp_path / "worktree"

    env = tool.build_environ(worktree, "sqlite")

    parts = env["PYTHONPATH"].split(_sep())
    expected = [str((worktree / "packages" / name / "src").resolve()) for name in tool.PACKAGE_SRC_DIRS]
    assert parts == expected
    assert "TCIP_STORE_BACKEND" not in env


def test_build_environ_sets_the_file_backend_when_requested(tool, tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STORE_BACKEND", "sqlite")

    env = tool.build_environ(tmp_path / "worktree", "file")

    assert env["TCIP_STORE_BACKEND"] == "file"


def test_the_proof_succeeds_when_tcip_mcp_resolves_under_the_worktree(tool, tmp_path, monkeypatch):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    inside = worktree / "packages" / "tcip-mcp" / "src" / "tcip_mcp" / "__init__.py"

    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=str(inside) + "\n", stderr="")

    monkeypatch.setattr(tool.subprocess, "run", _fake_run)
    env = tool.build_environ(worktree, "sqlite")

    resolved = tool.prove_resolution(worktree, env)

    assert resolved == inside.resolve()


def test_the_proof_refuses_when_tcip_mcp_resolves_outside_the_worktree(tool, tmp_path, monkeypatch):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        elsewhere = tmp_path / "elsewhere" / "tcip_mcp" / "__init__.py"
        return subprocess.CompletedProcess(cmd, 0, stdout=str(elsewhere) + "\n", stderr="")

    monkeypatch.setattr(tool.subprocess, "run", _fake_run)
    env = tool.build_environ(worktree, "sqlite")

    with pytest.raises(SystemExit, match="outside the worktree"):
        tool.prove_resolution(worktree, env)
    assert len(calls) == 1, "the proof itself must be the only subprocess call made"


def test_main_refuses_before_running_any_gate_when_resolution_is_outside(tool, tmp_path, monkeypatch):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        elsewhere = tmp_path / "elsewhere" / "tcip_mcp.py"
        return subprocess.CompletedProcess(cmd, 0, stdout=str(elsewhere) + "\n", stderr="")

    monkeypatch.setattr(tool.subprocess, "run", _fake_run)
    monkeypatch.setattr(sys, "argv", [
        "worktree_gate.py", str(worktree), "--ruff", "--mypy", "--pytest", "tests/test_x.py",
    ])

    with pytest.raises(SystemExit, match="outside the worktree"):
        tool.main()

    assert len(calls) == 1, "no gate (ruff/mypy/pytest) may run once the proof has refused"


def test_run_mypy_removes_its_cache_directory_afterward(tool, monkeypatch):
    created: list[str] = []

    def _fake_run(cmd, **kwargs):
        cache_dir = cmd[cmd.index("--cache-dir") + 1]
        assert Path(cache_dir).is_dir()
        created.append(cache_dir)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(tool.subprocess, "run", _fake_run)

    code = tool.run_mypy(Path("."), {})

    assert code == 0
    assert created and not Path(created[0]).exists()


def test_a_worktree_that_is_not_a_directory_is_refused_before_any_proof_or_gate(tool, tmp_path, monkeypatch):
    calls: list[object] = []

    def _fake_run(*args, **kwargs):
        calls.append(args)

    monkeypatch.setattr(tool.subprocess, "run", _fake_run)
    monkeypatch.setattr(sys, "argv", ["worktree_gate.py", str(tmp_path / "absent")])

    with pytest.raises(SystemExit, match="is not a directory"):
        tool.main()
    assert calls == []
