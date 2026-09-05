"""scripts/visualize.py: the demoted door's own command-line entry point.

Like overlay_reference_grid.py, --project (or $TCIP_STATE_ROOT) is required unconditionally:
the door writes an artifact and carries a platform audit line (bare @audited, no scope_arg).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "visualize.py"


def _run(args: list[str], cwd: Path, platform_root: str | None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if platform_root is None:
        env.pop("TCIP_STATE_ROOT", None)
    else:
        env["TCIP_STATE_ROOT"] = platform_root
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=60,
    )


def _fixture(tmp_path: Path) -> Path:
    images = tmp_path / "images"
    images.mkdir()
    img = images / "a.jpg"
    Image.new("RGB", (100, 80), color=(120, 120, 120)).save(img)
    labels = tmp_path / "annotations"
    labels.mkdir()
    write_annotations(labels / "a.json",
                      [Annotation(subject="bud", geometry=BBox(1, 1, 40, 30))], 100, 80)
    return img


def test_refuses_from_an_unpinned_cwd_and_plants_no_store(tmp_path):
    img = _fixture(tmp_path)
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = _run(["--source", "annotations", "--path", str(img)], cwd=cwd, platform_root=None)

    assert result.returncode != 0, result.stdout
    assert "TCIP_STATE_ROOT" in result.stderr
    assert not (cwd / ".tcip").exists()


def test_renders_annotations_over_a_fixture_root(tmp_path):
    img = _fixture(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = _run(
        ["--source", "annotations", "--path", str(img), "--project", str(project)],
        cwd=cwd, platform_root=None,
    )

    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert "error" not in body
    assert Path(body["image_path"]).is_file()
    assert (project / ".tcip").is_dir()
