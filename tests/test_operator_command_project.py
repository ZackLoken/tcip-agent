"""An operator command that runs a tool function outside the MCP server acts on the project its
``--project`` names, and refuses one that names no project.

Exercised as the real process entry point (a subprocess, the way an operator runs it), through
overlay-reference-grid, a render that leaves no audit line.
"""

from __future__ import annotations

from pathlib import Path

from tests._cli_fixtures import run_tcip
from tests._producer_fixtures import write_image


def _image(tmp_path: Path) -> Path:
    return write_image(tmp_path / "images" / "field.png", (64, 48), (40, 90, 60))


def test_a_render_command_leaves_no_audit_line_anywhere(tmp_path):
    """A render is a read: the named project's log gains no line, and the operator's cwd and the
    image's directory get no state."""
    import tcip_store as ts

    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.tools.project_tools import initialize_project

    project = tmp_path / "project"
    assert "error" not in initialize_project(str(project), "Grid project", "east block")
    image = _image(tmp_path)
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = run_tcip("overlay-reference-grid", ["--image", str(image), "--project", str(project)], cwd=cwd)

    assert result.returncode == 0, result.stderr
    rows = ts.read_log(audit_log_key(project)).records
    assert [r["tool"] for r in rows] == ["project_created"], rows
    assert not (cwd / ".tcip").exists()
    assert not (image.parent / ".tcip").exists()


def test_a_command_naming_no_project_refuses_and_writes_nothing(tmp_path):
    not_a_project = tmp_path / "plain"
    not_a_project.mkdir()
    image = _image(tmp_path)

    result = run_tcip("overlay-reference-grid", ["--image", str(image), "--project", str(not_a_project)], cwd=tmp_path)

    assert result.returncode != 0, result.stdout
    assert "names no readable project" in result.stderr
    assert not (not_a_project / ".tcip").exists()
