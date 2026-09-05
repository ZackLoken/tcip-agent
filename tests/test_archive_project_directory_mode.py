"""``archive_project``'s directory-tree mode: the same bundle ``output_path`` zips, written
instead as a directory an ``import_project`` run reads back exactly like a ZIP.
"""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.tools.project_tools import archive_project, import_project


def _project(tmp_path: Path) -> Path:
    """A minimal project: one image, one label, and the registry that decodes it."""
    from tcip_annotation import json_io
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import ClassRegistry, Subject

    root = tmp_path / "project"
    (root / "images" / "2026-03-04").mkdir(parents=True)
    (root / "images" / "2026-03-04" / "a_1.jpg").write_bytes(b"\xff\xd8\xff")
    (root / "annotations" / "2026-03-04").mkdir(parents=True)
    json_io.write_annotations(
        str(root / "annotations" / "2026-03-04" / "a_1.json"), [], 10, 10, keep_empty=True)
    class_registry.write_registry(
        root / "classes.json", ClassRegistry(subjects=(Subject(name="bud"),))
    )
    return root


def test_archive_project_refuses_a_destination_inside_the_project(tmp_path):
    root = _project(tmp_path)

    result = archive_project(str(root), output_dir=str(root / "bundle"))

    assert "error" in result
    assert "inside the project" in result["error"]
    assert not (root / "bundle").exists()


def test_archive_project_refuses_a_non_empty_destination(tmp_path):
    root = _project(tmp_path)
    dest = tmp_path / "bundle"
    dest.mkdir()
    (dest / "already_here.txt").write_text("x", encoding="utf-8")

    result = archive_project(str(root), output_dir=str(dest))

    assert "error" in result
    assert "not empty" in result["error"]
    assert [p.name for p in dest.iterdir()] == ["already_here.txt"]


def test_archive_project_refuses_both_output_path_and_output_dir(tmp_path):
    root = _project(tmp_path)

    result = archive_project(
        str(root), output_path=str(tmp_path / "bundle.zip"), output_dir=str(tmp_path / "bundle"),
    )

    assert "error" in result
    assert "not both" in result["error"]


def test_archive_project_refuses_neither_output_path_nor_output_dir(tmp_path):
    root = _project(tmp_path)

    result = archive_project(str(root))

    assert "error" in result
    assert "give either output_path" in result["error"]


def test_archive_project_directory_mode_admits_valid_work(tmp_path):
    root = _project(tmp_path)
    dest = tmp_path / "bundle"

    result = archive_project(str(root), output_dir=str(dest))

    assert "error" not in result, result
    assert result["output_dir"] == str(dest)
    assert result["files_added"] > 0
    assert result["size_bytes"] > 0
    assert (dest / "images" / "2026-03-04" / "a_1.jpg").is_file()
    assert (dest / "classes.json").is_file()


def test_directory_bundle_round_trip_yields_the_same_records_as_the_zip_round_trip(tmp_path):
    """archive_project(output_dir=...) -> import_project recovers a project the way the ZIP
    round trip does (tests/test_project_tools.py::test_export_import_roundtrip), including its
    band-group manifest and its multi-band-file capture, not only the single-band image."""
    from tcip_mcp.tools.project_tools import initialize_project, inspect_project, register_dataset

    src = tmp_path / "src_project"
    initialize_project(str(src), site="north orchard")
    manifest, npz_image = _populate_project(src)
    reg = register_dataset(str(src), crop="currant")
    assert "error" not in reg, reg

    bundle_dir = tmp_path / "bundle"
    exported = archive_project(str(src), output_dir=str(bundle_dir))
    assert "error" not in exported, exported
    assert bundle_dir.is_dir()

    dest = tmp_path / "restored"
    imported = import_project(str(bundle_dir), str(dest))
    assert "error" not in imported, imported
    assert imported["files_extracted"] == exported["files_added"]

    from tcip_store.adoption import adopt_root
    from tcip_store.file_backend import database_file
    from tcip_store.layout_claims import ROOT

    dest_abs = str(Path(dest).absolute())
    if not database_file(dest_abs).is_file():
        adopt_root(dest_abs, ROOT, report=lambda line: None)

    status = inspect_project(str(dest))
    assert status["initialized"] is True
    assert (dest / "annotations" / "2026-03-04" / "a_1.json").is_file()

    date = "2026-03-04"
    restored_images = dest / "images" / date
    restored_manifest = restored_images / manifest.name
    assert restored_manifest.is_file()
    assert restored_manifest.read_bytes() == manifest.read_bytes()
    assert (restored_images / npz_image.name).read_bytes() == npz_image.read_bytes()

    from tcip_mcp.pipelines.image_utils import list_logical_images

    assert sorted(list_logical_images(restored_images)) == sorted(
        list_logical_images(src / "images" / date)
    ) == ["a_1", "cap_001", "cap_002"]

    from tcip_mcp import class_registry

    restored = class_registry.read_registry(dest / "classes.json")
    assert [s.name for s in restored.subjects] == ["bud"]

    import json

    restored_id = json.loads((dest / "dataset.json").read_text())
    assert restored_id == {"crop": "currant", "id": reg["id"], "fingerprint": reg["fingerprint"]}


def _populate_project(src: Path) -> tuple[Path, Path]:
    """Populate an already-initialized project with a minimal image/label/registry, a
    multispectral capture's band-group manifest, and a sensor's own multi-band file, without
    re-creating ``.tcip``. Returns the manifest path and the multi-band file path, the two
    ``test_export_import_roundtrip`` also asserts a bundle carries whole.
    """
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import ClassRegistry, Subject

    date = "2026-03-04"
    images = src / "images" / date
    labels = src / "annotations" / date
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(images / "a_1.jpg")
    json_io.write_annotations(
        str(labels / "a_1.json"), [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)
    class_registry.write_registry(
        src / "classes.json", ClassRegistry(subjects=(Subject(name="bud"),)))

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
