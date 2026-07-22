"""Tests for project management tools."""

from __future__ import annotations

import json
from pathlib import Path

from tcip_mcp.tools.project_tools import (
    init_project,
    inspect_project,
    archive_project,
    import_project,
)


def test_init_project(tmp_path: Path):
    result = init_project(str(tmp_path))
    assert (tmp_path / ".tcip").is_dir()
    assert (tmp_path / ".tcip" / "artifacts").is_dir()
    assert (tmp_path / ".tcip" / "config.toml").is_file()
    assert ".tcip/" in result["created"]


def test_inspect_project(tmp_path: Path):
    status = inspect_project(str(tmp_path))
    assert status["initialized"] is False

    init_project(str(tmp_path))
    status = inspect_project(str(tmp_path))
    assert status["initialized"] is True
    assert status["has_config"] is True


def test_export_import_roundtrip(tmp_path: Path):
    """archive_project -> import_project -> inspect_project recovers the project."""
    from PIL import Image

    src = tmp_path / "src_project"
    date = "2-11-26"
    images = src / "images" / date
    labels = src / "annotations" / "default" / date / "detect"
    for d in (images, labels):
        d.mkdir(parents=True)
    init_project(str(src))

    from tcip_annotation import json_io
    from tcip_annotation.state import BBox

    Image.new("RGB", (64, 64)).save(images / "img_000.jpg")
    json_io.write_detect(str(labels / "img_000.json"), [BBox(10, 10, 30, 30, 0)], 64, 64)
    # The subject class registry decodes the labels' category_ids — a self-contained bundle must
    # carry it, or the archived annotations are unreadable on the other end.
    (src / "classes").mkdir()
    (src / "classes" / "default.json").write_text('{"0": {"name": "catkin", "color": "#FF0000"}}')

    zip_path = tmp_path / "export.zip"
    exported = archive_project(str(src), str(zip_path))
    assert "error" not in exported
    assert zip_path.is_file()

    dest = tmp_path / "restored"
    imported = import_project(str(zip_path), str(dest))
    assert "error" not in imported
    assert imported["files_extracted"] == exported["files_added"]

    status = inspect_project(str(dest))
    assert status["initialized"] is True
    assert status["has_config"] is True
    assert status["image_count"] == 1
    assert (dest / "annotations" / "default" / date / "detect" / "img_000.json").is_file()
    # The registry survived, so the restored labels are still decodable.
    restored_map = json.loads((dest / "classes" / "default.json").read_text())
    assert restored_map["0"]["name"] == "catkin"
