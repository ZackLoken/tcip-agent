"""Tests for the platform state-root resolver (cwd-fragmentation fix)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import tcip_mcp.project_paths as pp


def test_platform_state_root_defaults_to_cwd(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(pp.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    assert pp.platform_state_root() == tmp_path


def test_platform_state_root_honors_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(pp.ENV_VAR, str(tmp_path))
    # Even from a different cwd, the pinned root wins: the whole point (no fragmentation).
    other = tmp_path / "sub"
    other.mkdir()
    monkeypatch.chdir(other)
    assert pp.platform_state_root() == tmp_path


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
    binding = pp.pin_platform_root(from_marker=False)
    assert os.environ.get(pp.ENV_VAR) == str(binding.root)
    assert binding.source == "repo_root"
    assert binding.inherited_root is None
    assert binding.marker_problem is None
    assert (binding.root / ".mcp.json").is_file() or (binding.root / "CLAUDE.md").is_file()


def test_pin_respects_a_preset_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(pp.ENV_VAR, str(tmp_path))
    # An operator-provided root is not overridden when the process does not opt into the marker.
    binding = pp.pin_platform_root(from_marker=False)
    assert binding.root == tmp_path
    assert binding.source == "inherited"
    assert binding.inherited_root == str(tmp_path)
    assert binding.marker_problem is None


def test_pin_from_marker_prefers_the_marker_over_an_inherited_root(
    tmp_path: Path, monkeypatch
) -> None:
    from tcip_mcp import workspace

    proj_root = workspace.project_path("chestnut_demo")
    (proj_root / ".tcip").mkdir(parents=True)
    workspace.set_active_project("chestnut_demo")  # also repins the variable to proj_root

    inherited = tmp_path.parent / "elsewhere"
    inherited.mkdir()
    monkeypatch.setenv(pp.ENV_VAR, str(inherited))  # an inherited root predating this bind

    binding = pp.pin_platform_root(from_marker=True)
    assert binding.root == proj_root
    assert binding.source == "marker"
    assert binding.inherited_root == str(inherited)
    assert binding.marker_problem is None
    assert os.environ.get(pp.ENV_VAR) == str(proj_root)


def test_pin_from_marker_falls_back_to_repo_root_when_nothing_inherited(monkeypatch) -> None:
    # The per-test workspace holds no marker: no project to bind from and nothing inherited.
    monkeypatch.delenv(pp.ENV_VAR, raising=False)
    binding = pp.pin_platform_root(from_marker=True)
    assert binding.source == "repo_root"
    assert binding.inherited_root is None
    assert binding.marker_problem is None
    assert (binding.root / ".mcp.json").is_file() or (binding.root / "CLAUDE.md").is_file()


def test_pin_from_marker_records_a_dangling_marker_and_keeps_the_inherited_root(
    tmp_path: Path, monkeypatch
) -> None:
    from tcip_mcp import workspace

    proj_root = workspace.project_path("chestnut_demo")
    (proj_root / ".tcip").mkdir(parents=True)
    workspace.set_active_project("chestnut_demo")
    shutil.rmtree(proj_root / ".tcip")  # the marker now names a project with nothing there

    inherited = tmp_path.parent / "elsewhere"
    inherited.mkdir()
    monkeypatch.setenv(pp.ENV_VAR, str(inherited))

    binding = pp.pin_platform_root(from_marker=True)
    assert binding.source == "inherited"
    assert binding.root == inherited
    assert binding.marker_problem is not None
    assert "chestnut_demo" in binding.marker_problem


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


def test_pin_from_marker_keeps_a_transient_read_failure_even_when_a_retry_would_succeed(
    tmp_path: Path, monkeypatch
) -> None:
    """A lock timeout on the marker read must not be discarded by a second, independent read
    that happens to succeed: the transient failure is what the binding records."""
    from tcip_mcp import workspace

    proj_root = workspace.project_path("chestnut_demo")
    (proj_root / ".tcip").mkdir(parents=True)
    workspace.set_active_project("chestnut_demo")

    real_active = workspace.active_project_if_present
    calls = {"n": 0}

    def flaky(*, create=False):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("could not acquire the workspace lock within 30s")
        return real_active(create=create)

    monkeypatch.setattr(workspace, "active_project_if_present", flaky)

    binding = pp.pin_platform_root(from_marker=True)
    assert binding.marker_problem == "could not acquire the workspace lock within 30s"


def test_repin_platform_root_updates_the_root_binding(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(pp.ENV_VAR, raising=False)
    other = tmp_path.parent / "elsewhere"
    other.mkdir()

    pp.repin_platform_root(other)

    binding = pp.root_binding()
    assert binding is not None
    assert binding.root == other
    assert binding.source == "adopted"
    assert binding.inherited_root is None
    assert binding.marker_problem is None


def test_repin_platform_root_records_the_previous_root_as_inherited(
    monkeypatch, tmp_path: Path
) -> None:
    first = tmp_path.parent / "first"
    second = tmp_path.parent / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv(pp.ENV_VAR, str(first))

    pp.repin_platform_root(second)

    binding = pp.root_binding()
    assert binding.root == second
    assert binding.inherited_root == str(first)


def test_restore_binding_resets_the_root_binding_to_a_snapshot() -> None:
    original = pp.root_binding()
    pp.pin_platform_root(from_marker=False)
    assert pp.root_binding() is not original

    pp.restore_binding(original)
    assert pp.root_binding() is original
