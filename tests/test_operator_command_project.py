"""An operator command that runs a tool function outside the MCP server acts on the project its
``--project`` names, and refuses one that names no project.

Exercised as the real process entry point (a subprocess, the way an operator runs it), through
overlay-reference-grid, a door that writes an artifact and so leaves an audit line.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tcip_web.cli", "overlay-reference-grid", *args],
        cwd=str(cwd), capture_output=True, text=True, timeout=60,
    )


def _image(tmp_path: Path) -> Path:
    from PIL import Image

    images = tmp_path / "images"
    images.mkdir()
    image = images / "field.png"
    Image.new("RGB", (64, 48), color=(40, 90, 60)).save(image)
    return image


def test_a_writing_command_files_its_audit_line_under_the_named_project(tmp_path):
    """The audit line of a door that writes lands in the named project's own log, and the
    operator's cwd and the image's directory get nothing."""
    import tcip_store as ts

    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.tools.project_tools import initialize_project

    project = tmp_path / "project"
    assert "error" not in initialize_project(str(project), "Grid project", "east block")
    image = _image(tmp_path)
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = _run(["--image", str(image), "--project", str(project)], cwd=cwd)

    assert result.returncode == 0, result.stderr
    rows = ts.read_log(audit_log_key(project)).records
    assert "overlay_reference_grid" in [r["tool"] for r in rows], rows
    assert not (cwd / ".tcip").exists()
    assert not (image.parent / ".tcip").exists()


def test_a_command_naming_no_project_refuses_and_writes_nothing(tmp_path):
    not_a_project = tmp_path / "plain"
    not_a_project.mkdir()
    image = _image(tmp_path)

    result = _run(["--image", str(image), "--project", str(not_a_project)], cwd=tmp_path)

    assert result.returncode != 0, result.stdout
    assert "names no readable project" in result.stderr
    assert not (not_a_project / ".tcip").exists()
