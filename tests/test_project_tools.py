"""Tests for project management tools."""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.tools.project_tools import (
    init_project,
    inspect_project,
    archive_project,
    import_project,
    read_datasets,
    register_dataset,
)


def _make_dataset(root: Path) -> None:
    """A minimal nested-schema dataset (image + label + registry) for identity tests."""
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import ClassRegistry, Subject

    (root / "images" / "2-11-26").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32)).save(root / "images" / "2-11-26" / "img_000.jpg")
    (root / "annotations" / "2-11-26").mkdir(parents=True, exist_ok=True)
    json_io.write_annotations(
        str(root / "annotations" / "2-11-26" / "img_000.json"),
        [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))], 32, 32)
    class_registry.write_registry(
        root / "classes.json", ClassRegistry(subjects=(Subject(name="catkin"),)))


def test_register_dataset_writes_identity_and_registers(tmp_path: Path):
    import json

    src = tmp_path / "proj"
    _make_dataset(src)

    res = register_dataset(str(src), crop="hazelnut")
    assert "error" not in res
    assert res["crop"] == "hazelnut" and res["id"] and res["fingerprint"]

    # dataset.json holds {crop, id, fingerprint}.
    ident = json.loads((src / "dataset.json").read_text())
    assert ident == {"crop": "hazelnut", "id": res["id"], "fingerprint": res["fingerprint"]}
    # the project registry knows the dataset.
    regs = read_datasets(src)
    assert len(regs) == 1 and regs[0]["id"] == res["id"] and regs[0]["crop"] == "hazelnut"


def test_register_dataset_requires_crop_and_keeps_id_stable(tmp_path: Path):
    src = tmp_path / "proj"
    _make_dataset(src)

    assert "error" in register_dataset(str(src), crop="")  # crop is the expert's fact, required

    first = register_dataset(str(src), crop="hazelnut")
    again = register_dataset(str(src), crop="hazelnut")
    assert again["id"] == first["id"]  # id minted once, preserved across re-runs
    assert len(read_datasets(src)) == 1  # not duplicated in the registry


def test_register_dataset_reconciles_a_move_by_id(tmp_path: Path):
    import shutil

    src = tmp_path / "orig"
    _make_dataset(src)
    reg = register_dataset(str(src), crop="hazelnut")

    moved = tmp_path / "moved"
    shutil.copytree(src, moved)  # same content, new path
    register_dataset(str(moved), crop="hazelnut", project_root=str(src))

    regs = read_datasets(src)
    same = [r for r in regs if r["id"] == reg["id"]]
    assert len(same) == 1  # one entry for the id — the move updated the path, not duplicated
    assert same[0]["path"] == str(moved)
    assert same[0]["fingerprint"] == reg["fingerprint"]  # unchanged content -> same fingerprint


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
    labels = src / "annotations" / date
    for d in (images, labels):
        d.mkdir(parents=True)
    init_project(str(src))

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import ClassRegistry, Subject

    Image.new("RGB", (64, 64)).save(images / "img_000.jpg")
    json_io.write_annotations(
        str(labels / "img_000.json"),
        [Annotation(subject="catkin", geometry=BBox(10, 10, 30, 30))], 64, 64,
    )
    # The class registry decodes the labels' names — a self-contained bundle must carry it, or the
    # archived annotations are unreadable on the other end. One nested classes.json at the root.
    class_registry.write_registry(
        src / "classes.json",
        ClassRegistry(subjects=(Subject(name="catkin", description="a hazelnut catkin"),)),
    )
    reg = register_dataset(str(src), crop="hazelnut")  # dataset.json identity travels with the data

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
    assert (dest / "annotations" / date / "img_000.json").is_file()
    # The registry survived, so the restored labels are still decodable.
    restored = class_registry.read_registry(dest / "classes.json")
    assert [s.name for s in restored.subjects] == ["catkin"]
    # dataset.json travelled with the data — identity (id/crop/fingerprint) survives the round-trip.
    import json

    restored_id = json.loads((dest / "dataset.json").read_text())
    assert restored_id == {"crop": "hazelnut", "id": reg["id"], "fingerprint": reg["fingerprint"]}
