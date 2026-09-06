"""tcip overlay-reference-grid: the demoted door's own command-line entry point.

Unlike tcip score-predictions, --project (or $TCIP_STATE_ROOT) is required unconditionally: the
door writes an artifact and carries a platform audit line (bare @audited, no scope_arg), so an
unpinned root is always the wrong root for both to land under.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _run(args: list[str], cwd: Path, platform_root: str | None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if platform_root is None:
        env.pop("TCIP_STATE_ROOT", None)
    else:
        env["TCIP_STATE_ROOT"] = platform_root
    return subprocess.run(
        [sys.executable, "-m", "tcip_web.cli", "overlay-reference-grid", *args],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=60,
    )


def test_refuses_from_an_unpinned_cwd_and_plants_no_store(tmp_path):
    img = tmp_path / "a.jpg"
    Image.new("RGB", (640, 480), color=(90, 110, 70)).save(img)
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = _run(["--image", str(img)], cwd=cwd, platform_root=None)

    assert result.returncode != 0, result.stdout
    assert "TCIP_STATE_ROOT" in result.stderr
    assert not (cwd / ".tcip").exists()


def test_renders_the_overlay_and_echoes_grid_geometry_over_a_fixture_root(tmp_path):
    img = tmp_path / "images" / "a.jpg"
    img.parent.mkdir()
    Image.new("RGB", (640, 480), color=(90, 110, 70)).save(img)
    project = tmp_path / "project"
    project.mkdir()
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = _run(
        ["--image", str(img), "--project", str(project), "--tile-size", "80"],
        cwd=cwd, platform_root=None,
    )

    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert "error" not in body
    assert Path(body["image_path"]).is_file()
    assert body["tile_size"] == 80
    assert body["cols"] == 8 and body["rows"] == 6
    assert (project / ".tcip").is_dir()
