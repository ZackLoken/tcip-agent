"""Tests for the platform state-root resolver (cwd-fragmentation fix)."""

from __future__ import annotations

import os
from pathlib import Path

import tcip_mcp.project_paths as pp


def test_project_root_defaults_to_cwd(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(pp.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    assert pp.project_root() == tmp_path


def test_project_root_honors_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(pp.ENV_VAR, str(tmp_path))
    # Even from a different cwd, the pinned root wins: the whole point (no fragmentation).
    other = tmp_path / "sub"
    other.mkdir()
    monkeypatch.chdir(other)
    assert pp.project_root() == tmp_path


def test_repo_root_finds_the_marker() -> None:
    root = pp.repo_root_from_here()
    # .mcp.json is the specific repo-root marker (unlike CLAUDE.md, which every package under
    # packages/ also carries): assert the real root, not just "some ancestor with a marker",
    # which a package-level CLAUDE.md would satisfy too.
    assert (root / ".mcp.json").is_file()


def test_repo_root_climbs_past_a_package_level_claude_md(tmp_path: Path, monkeypatch) -> None:
    """A package subdir with its own CLAUDE.md (every packages/* dir now has one) must not stop
    the climb before the true repo root's .mcp.json: regression for the packages/tcip-mcp
    CLAUDE.md landing and shadowing the real root for any module under that package."""
    root = tmp_path / "repo"
    pkg_src = root / "packages" / "pkg" / "src" / "pkg_module"
    pkg_src.mkdir(parents=True)
    (root / ".mcp.json").write_text("{}")
    (root / "CLAUDE.md").write_text("root")
    (root / "packages" / "pkg" / "CLAUDE.md").write_text("package-level")

    # repo_root_from_here() resolves Path(__file__): patch __file__ in the module under test to
    # a path under the fake package tree (need not exist; Path.resolve() only normalizes).
    monkeypatch.setattr(pp, "__file__", str(pkg_src / "somewhere.py"))
    assert pp.repo_root_from_here() == root


def test_pin_sets_env_to_repo_root_when_unset(monkeypatch) -> None:
    monkeypatch.delenv(pp.ENV_VAR, raising=False)
    try:
        root = pp.pin_project_root()
        assert os.environ.get(pp.ENV_VAR) == str(root)
        assert (root / ".mcp.json").is_file() or (root / "CLAUDE.md").is_file()
    finally:
        os.environ.pop(pp.ENV_VAR, None)


def test_pin_respects_a_preset_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(pp.ENV_VAR, str(tmp_path))
    # setdefault: an operator-provided root is not overridden.
    assert pp.pin_project_root() == tmp_path


def test_resolve_state_relative_stays_cwd_relative_when_unpinned(monkeypatch) -> None:
    # Unpinned: a relative state path is returned as-is → resolved against cwd at use, so a
    # test that chdir's to a tmp dir keeps its .tcip isolated (the regression this guards).
    monkeypatch.delenv(pp.ENV_VAR, raising=False)
    rel = Path(".tcip/experiments")
    assert pp.resolve_state(rel) == rel


def test_resolve_state_relative_prefixed_when_pinned(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(pp.ENV_VAR, str(tmp_path))
    assert pp.resolve_state(Path(".tcip/audit.jsonl")) == tmp_path / ".tcip" / "audit.jsonl"


def test_resolve_state_absolute_passes_through(monkeypatch, tmp_path: Path) -> None:
    # A rebound absolute path (test isolation) wins even when a root is pinned.
    monkeypatch.setenv(pp.ENV_VAR, str(tmp_path / "pinned"))
    abs_path = tmp_path / "isolated" / "audit.jsonl"
    assert pp.resolve_state(abs_path) == abs_path
