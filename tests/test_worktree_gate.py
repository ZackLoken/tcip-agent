"""tools/worktree_gate.py's environment builder and its refusal before any gate runs.

The refusal comes when tcip_mcp does not resolve inside the given worktree. No real gate
(ruff/mypy/pytest) runs here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tools import worktree_gate


def _sep() -> str:
    return ";" if sys.platform == "win32" else ":"


def test_build_environ_sets_pythonpath_to_the_four_package_src_dirs_under_the_worktree(
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "worktree"

    env = worktree_gate.build_environ(worktree)

    parts = env["PYTHONPATH"].split(_sep())
    expected = [
        str((worktree / "packages" / name / "src").resolve())
        for name in worktree_gate.PACKAGE_SRC_DIRS
    ]
    assert parts == expected


def test_the_proof_succeeds_when_tcip_mcp_resolves_under_the_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    inside = (
        worktree / "packages" / "tcip-mcp" / "src" / "tcip_mcp" / "__init__.py"
    )

    def _fake_run(
        cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            cmd, 0, stdout=str(inside) + "\n", stderr=""
        )

    monkeypatch.setattr(worktree_gate.subprocess, "run", _fake_run)
    env = worktree_gate.build_environ(worktree)

    resolved = worktree_gate.prove_resolution(worktree, env)

    assert resolved == inside.resolve()


def test_the_proof_refuses_when_tcip_mcp_resolves_outside_the_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    calls = []

    def _fake_run(
        cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        elsewhere = tmp_path / "elsewhere" / "tcip_mcp" / "__init__.py"
        return subprocess.CompletedProcess(
            cmd, 0, stdout=str(elsewhere) + "\n", stderr=""
        )

    monkeypatch.setattr(worktree_gate.subprocess, "run", _fake_run)
    env = worktree_gate.build_environ(worktree)

    with pytest.raises(SystemExit, match="outside the worktree"):
        worktree_gate.prove_resolution(worktree, env)
    assert len(calls) == 1, (
        "the proof itself must be the only subprocess call made"
    )


def test_main_refuses_before_running_any_gate_when_resolution_is_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    calls = []

    def _fake_run(
        cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        elsewhere = tmp_path / "elsewhere" / "tcip_mcp.py"
        return subprocess.CompletedProcess(
            cmd, 0, stdout=str(elsewhere) + "\n", stderr=""
        )

    monkeypatch.setattr(worktree_gate.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worktree_gate.py",
            str(worktree),
            "--ruff",
            "--mypy",
            "--pytest",
            "tests/test_x.py",
        ],
    )

    with pytest.raises(SystemExit, match="outside the worktree"):
        worktree_gate.main()

    assert len(calls) == 1, (
        "no gate (ruff/mypy/pytest) may run once the proof has refused"
    )


def test_run_mypy_removes_its_cache_directory_afterward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[str] = []

    def _fake_run(
        cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        cache_dir = cmd[cmd.index("--cache-dir") + 1]
        assert Path(cache_dir).is_dir()
        created.append(cache_dir)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(worktree_gate.subprocess, "run", _fake_run)

    code = worktree_gate.run_mypy(Path("."), {})

    assert code == 0
    assert created and not Path(created[0]).exists()


def test_a_worktree_that_is_not_a_directory_is_refused_before_any_proof_or_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []

    def _fake_run(*args: object, **kwargs: object) -> None:
        calls.append(args)

    monkeypatch.setattr(worktree_gate.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        sys, "argv", ["worktree_gate.py", str(tmp_path / "absent")]
    )

    with pytest.raises(SystemExit, match="is not a directory"):
        worktree_gate.main()
    assert calls == []
