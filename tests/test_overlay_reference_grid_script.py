"""tcip overlay-reference-grid: the overlay tool's command, through the console's one adapter.

The tool acts on a project, so a run naming no --project is refused naming it; the render lands
under the project named.
"""

from __future__ import annotations

import json
from pathlib import Path

from tcip_mcp.dataset_layout import image_root
from tests._cli_fixtures import run_tcip
from tests._producer_fixtures import write_image


def test_refuses_without_a_project_and_plants_no_store(
    tmp_path: Path,
) -> None:
    img = write_image(tmp_path / "a.jpg", (640, 480), (90, 110, 70))
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = run_tcip(
        "overlay-reference-grid",
        ["--image-path", str(img), "--overlap", "0", "--user", "tester"],
        cwd=cwd,
    )

    assert result.returncode != 0, result.stdout
    assert "--project" in result.stderr
    assert not (cwd / ".tcip").exists()


def test_renders_the_overlay_and_echoes_grid_geometry_under_the_named_project(
    project: Path, tmp_path: Path
) -> None:
    img = write_image(image_root(project) / "a.jpg", (640, 480), (90, 110, 70))
    cwd = tmp_path.parent / "operator_cwd"
    cwd.mkdir()

    result = run_tcip(
        "overlay-reference-grid",
        [
            "--image-path",
            str(img),
            "--project",
            str(project),
            "--tile-size",
            "80",
            "--overlap",
            "0",
            "--user",
            "tester",
        ],
        cwd=cwd,
    )

    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert "error" not in body
    assert (
        Path(body["image_path"]).parent
        == (project / ".tcip" / "artifacts" / "viz").resolve()
    )
    assert body["tile_size"] == 80
    assert body["cols"] == 8 and body["rows"] == 6
    assert not (cwd / ".tcip").exists()
