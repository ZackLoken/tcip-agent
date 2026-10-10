"""``archive_project``'s directory-tree mode: the same bundle ``output_path`` zips, written
instead as a directory an ``import_project`` run reads back exactly like a ZIP.
"""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.tools.project_tools import archive_project, import_project
from tests._producer_fixtures import one_labeled_capture


def test_archive_project_refuses_a_destination_inside_the_project(tmp_path):
    root = one_labeled_capture(tmp_path / "project")

    result = archive_project(root, output_dir=str(root / "bundle"))

    assert "error" in result
    assert "inside the project" in result["error"]
    assert not (root / "bundle").exists()


def test_archive_project_refuses_a_non_empty_destination(tmp_path):
    root = one_labeled_capture(tmp_path / "project")
    dest = tmp_path / "bundle"
    dest.mkdir()
    (dest / "already_here.txt").write_text("x", encoding="utf-8")

    result = archive_project(root, output_dir=str(dest))

    assert "error" in result
    assert "already exists" in result["error"]
    assert [p.name for p in dest.iterdir()] == ["already_here.txt"]


def test_archive_project_refuses_a_zip_destination_inside_the_project_or_already_written(
    tmp_path, monkeypatch,
):
    """The ZIP form takes the directory form's refusals: a destination inside the project would
    bundle its own truncated self, and a written archive is never written over; a relative path
    names a location under the project, never under wherever the process runs."""
    root = one_labeled_capture(tmp_path / "project")
    outside = tmp_path / "bundle.zip"

    inside = archive_project(root, output_path=str(root / "archive.zip"))
    assert "inside the project" in inside["error"]
    assert not (root / "archive.zip").exists()

    assert "error" not in archive_project(root, output_path=str(outside))
    written = outside.read_bytes()
    again = archive_project(root, output_path=str(outside))
    assert "already exists" in again["error"]
    assert outside.read_bytes() == written

    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert "inside the project" in archive_project(root, output_path="archive.zip")["error"]
    assert "error" not in archive_project(root, output_path="../beside.zip")
    assert (tmp_path / "beside.zip").is_file()
    assert list(elsewhere.iterdir()) == []


def test_archive_project_refuses_both_output_path_and_output_dir(tmp_path):
    root = one_labeled_capture(tmp_path / "project")

    result = archive_project(
        root, output_path=str(tmp_path / "bundle.zip"), output_dir=str(tmp_path / "bundle"),
    )

    assert "error" in result
    assert "not both" in result["error"]


def test_archive_project_refuses_neither_output_path_nor_output_dir(tmp_path):
    root = one_labeled_capture(tmp_path / "project")

    result = archive_project(root)

    assert "error" in result
    assert "give either output_path" in result["error"]


def test_archive_project_directory_mode_admits_valid_work(tmp_path):
    root = one_labeled_capture(tmp_path / "project")
    dest = tmp_path / "bundle"

    result = archive_project(root, output_dir=str(dest))

    assert "error" not in result, result
    assert result["output_dir"] == str(dest)
    assert result["files_added"] > 0
    assert result["size_bytes"] > 0
    assert (dest / "images" / "2026-03-04" / "a_1.jpg").is_file()
    assert (dest / "subjects.json").is_file()


def test_directory_bundle_round_trip_yields_the_same_records_as_the_zip_round_trip(tmp_path):
    """archive_project(output_dir=...) -> import_project recovers a project, its band-group
    manifest and its multi-band-file capture included."""
    from tcip_mcp.tools.project_tools import initialize_project, inspect_project, register_dataset

    src = tmp_path / "src_project"
    initialize_project(str(src), "Source project", "north orchard")
    manifest, npz_image = _populate_project(src)
    reg = register_dataset(src, str(src), crop="currant")
    assert "error" not in reg, reg

    bundle_dir = tmp_path / "bundle"
    exported = archive_project(src, output_dir=str(bundle_dir))
    assert "error" not in exported, exported
    assert bundle_dir.is_dir()

    dest = tmp_path / "restored"
    imported = import_project(str(bundle_dir), str(dest))
    assert "error" not in imported, imported
    assert imported["files_extracted"] == exported["files_added"]

    from tcip_annotation.json_io import read_label_document
    from tcip_mcp.dataset_layout import label_key

    status = inspect_project(dest)
    assert status["display_name"] == "Source project"
    date = "2026-03-04"
    assert read_label_document(label_key(dest.resolve(), date, "a_1")) == read_label_document(
        label_key(src.resolve(), date, "a_1"))
    assert read_label_document(label_key(dest.resolve(), date, "a_1")).annotations
    restored_images = dest / "images" / date
    restored_manifest = restored_images / manifest.name
    assert restored_manifest.is_file()
    assert restored_manifest.read_bytes() == manifest.read_bytes()
    assert (restored_images / npz_image.name).read_bytes() == npz_image.read_bytes()

    from tcip_mcp.pipelines.image_utils import list_logical_images

    assert sorted(list_logical_images(restored_images)) == sorted(
        list_logical_images(src / "images" / date)
    ) == ["a_1", "cap_001", "cap_002"]

    from tcip_mcp import subject_registry

    restored = subject_registry.read_registry(dest)
    assert [s.name for s in restored.subjects] == ["bud"]

    import json

    restored_id = json.loads((dest / "dataset.json").read_text())
    assert restored_id == {"crop": "currant", "id": reg["id"], "fingerprint": reg["fingerprint"]}


def _populate_project(src: Path) -> tuple[Path, Path]:
    """Populate an already-initialized project with a minimal image/label/registry, a
    multispectral capture's band-group manifest, and a sensor's own multi-band file, without
    re-creating ``.tcip``. Returns the manifest path and the multi-band file path.
    """
    from PIL import Image

    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import SubjectRegistry, Subject
    from tests._producer_fixtures import label_image, registry_over

    date = "2026-03-04"
    images = src / "images" / date
    images.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(images / "a_1.jpg")
    label_image(images / "a_1.jpg", [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))],
                32, 32)
    registry_over(src, SubjectRegistry(subjects=(Subject(name="bud"),)))

    # A multispectral capture written one file per band: the manifest beside the bands is what
    # makes those files one logical image, so a directory bundle has to carry all of them too.
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    bands = {}
    for band, wavelength in (("green", 560.0), ("red", 668.0)):
        band_file = images / f"cap_001_{band}.tif"
        Image.new("L", (32, 32)).save(band_file)
        bands[band] = band_file
    manifest = write_band_group_manifest(
        images, "cap_001", bands, central_wavelength_nm={"green": 560.0, "red": 668.0},
        source="explicit-manifest",
    )
    # A sensor that writes one multi-band file per capture instead of one file per band.
    import numpy as np

    npz_image = images / "cap_002.npz"
    np.savez(npz_image, bands=np.zeros((2, 32, 32), dtype=np.uint16))
    return manifest, npz_image


def test_import_project_refuses_a_bundle_path_that_does_not_exist(tmp_path):
    result = import_project(str(tmp_path / "nope"), str(tmp_path / "dest"))

    assert "error" in result
    assert "bundle not found" in result["error"]
