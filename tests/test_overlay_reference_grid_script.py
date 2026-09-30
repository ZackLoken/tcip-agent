"""tcip overlay-reference-grid: the demoted door's own command-line entry point.

--project is required unconditionally and must name a project: the door writes an artifact and
carries an audit line, both under that project.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tcip_web.cli", "overlay-reference-grid", *args],
        cwd=str(cwd), capture_output=True, text=True, timeout=60,
    )


def test_refuses_without_a_project_and_plants_no_store(tmp_path):
    img = tmp_path / "a.jpg"
    Image.new("RGB", (640, 480), color=(90, 110, 70)).save(img)
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = _run(["--image", str(img)], cwd=cwd)

    assert result.returncode != 0, result.stdout
    assert "--project" in result.stderr
    assert not (cwd / ".tcip").exists()


def test_renders_the_overlay_and_echoes_grid_geometry_under_the_named_project(project, tmp_path):
    img = project / "images" / "a.jpg"
    img.parent.mkdir()
    Image.new("RGB", (640, 480), color=(90, 110, 70)).save(img)
    cwd = tmp_path.parent / "operator_cwd"
    cwd.mkdir()

    result = _run(["--image", str(img), "--project", str(project), "--tile-size", "80"], cwd=cwd)

    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert "error" not in body
    assert Path(body["image_path"]).parent == (project / ".tcip" / "artifacts" / "viz").resolve()
    assert body["tile_size"] == 80
    assert body["cols"] == 8 and body["rows"] == 6
    assert not (cwd / ".tcip").exists()
