"""tcip visualize: the demoted door's own command-line entry point.

Like tcip overlay-reference-grid, --project is required unconditionally and must name a project:
the door writes an artifact and carries an audit line, both under that project.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from tcip_annotation.state import Annotation, BBox
from tests._producer_fixtures import label_image


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tcip_web.cli", "visualize", *args],
        cwd=str(cwd), capture_output=True, text=True, timeout=60,
    )


def _fixture(root: Path) -> Path:
    images = root / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
    img = images / "a.jpg"
    Image.new("RGB", (100, 80), color=(120, 120, 120)).save(img)
    label_image(img, [Annotation(subject="bud", geometry=BBox(1, 1, 40, 30))], 100, 80)
    return img


def test_refuses_without_a_project_and_plants_no_store(tmp_path):
    img = _fixture(tmp_path)
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = _run(["--source", "annotations", "--path", str(img)], cwd=cwd)

    assert result.returncode != 0, result.stdout
    assert "--project" in result.stderr
    assert not (cwd / ".tcip").exists()


def test_refuses_a_directory_holding_no_project_record(tmp_path):
    img = _fixture(tmp_path)
    bare = tmp_path / "bare"
    bare.mkdir()

    result = _run(["--source", "annotations", "--path", str(img), "--project", str(bare)],
                  cwd=tmp_path)

    assert result.returncode != 0, result.stdout
    assert "initialize_project" in result.stdout + result.stderr
    assert not (bare / ".tcip").exists()


def test_renders_annotations_under_the_named_project(project, tmp_path):
    img = _fixture(project)
    cwd = tmp_path.parent / "operator_cwd"
    cwd.mkdir()

    result = _run(["--source", "annotations", "--path", str(img), "--project", str(project)],
                  cwd=cwd)

    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert "error" not in body
    assert Path(body["image_path"]).parent == (project / ".tcip" / "artifacts" / "viz").resolve()
    assert not (cwd / ".tcip").exists()
