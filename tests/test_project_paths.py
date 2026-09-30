"""The repository root this package infers from its own location."""

from __future__ import annotations

from pathlib import Path

import tcip_mcp.project_paths as pp


def test_repo_root_finds_the_marker() -> None:
    root = pp.repo_root_from_here()
    # .mcp.json is the repo-root marker; every package under packages/ also carries a CLAUDE.md,
    # so the assertion names the real root rather than any ancestor with a marker.
    assert (root / ".mcp.json").is_file()


def test_repo_root_climbs_past_a_package_level_claude_md(tmp_path: Path, monkeypatch) -> None:
    """A package subdir with its own CLAUDE.md must not stop the climb before the true repo
    root's .mcp.json."""
    root = tmp_path / "repo"
    pkg_src = root / "packages" / "pkg" / "src" / "pkg_module"
    pkg_src.mkdir(parents=True)
    (root / ".mcp.json").write_text("{}")
    (root / "CLAUDE.md").write_text("root")
    (root / "packages" / "pkg" / "CLAUDE.md").write_text("package-level")

    # repo_root_from_here() resolves Path(__file__), so __file__ is pointed into the fake tree.
    monkeypatch.setattr(pp, "__file__", str(pkg_src / "somewhere.py"))
    assert pp.repo_root_from_here() == root
