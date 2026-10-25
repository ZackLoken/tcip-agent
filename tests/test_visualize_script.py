"""tcip visualize: the render tool's command, through the console's one adapter.

The tool acts on a project, so a run naming no --project is refused naming it, a --project
holding no project record is refused, and the artifact lands under the project named.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tcip_annotation.state import Annotation, BBox
from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tests._cli_fixtures import run_tcip
from tests._producer_fixtures import label_image


def _fixture(root: Path) -> Path:
    images = root / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
    img = images / "a.jpg"
    Image.new("RGB", (100, 80), color=(120, 120, 120)).save(img)
    label_image(
        img, [Annotation(subject="bud", geometry=BBox(1, 1, 40, 30))], 100, 80
    )
    return img


def _annotations(img: Path) -> list[str]:
    """Return the command's statement of an annotations render of ``img``, by a person."""
    render = {"source": "annotations", "path": str(img)}
    return ["--render", json.dumps(render), "--user", "tester"]


def test_refuses_without_a_project_and_plants_no_store(tmp_path: Path) -> None:
    img = _fixture(tmp_path)
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = run_tcip("visualize", _annotations(img), cwd=cwd)

    assert result.returncode != 0, result.stdout
    assert "--project" in result.stderr
    assert not (cwd / ".tcip").exists()


def test_refuses_a_directory_holding_no_project_record(tmp_path: Path) -> None:
    img = _fixture(tmp_path)
    bare = tmp_path / "bare"
    bare.mkdir()

    result = run_tcip(
        "visualize", [*_annotations(img), "--project", str(bare)], cwd=tmp_path
    )

    assert result.returncode != 0, result.stdout
    assert "initialize_project" in result.stdout + result.stderr
    assert not (bare / ".tcip").exists()


def test_renders_annotations_under_the_named_project(
    project: Path, tmp_path: Path
) -> None:
    img = _fixture(project)
    cwd = tmp_path.parent / "operator_cwd"
    cwd.mkdir()

    result = run_tcip(
        "visualize", [*_annotations(img), "--project", str(project)], cwd=cwd
    )

    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert "error" not in body
    assert (
        Path(body["image_path"]).parent
        == (project / ".tcip" / "artifacts" / "viz").resolve()
    )
    assert not (cwd / ".tcip").exists()
