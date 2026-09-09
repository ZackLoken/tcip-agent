"""The retired classes.json: the three registry writers, register_dataset, the two named
absence refusals, the doctor's finding, dataset_scope_of's evidence and the two bundle doors,
all against a dataset still holding the pre-rename document. Each refusal is paired with what
still admits, per CLAUDE.md's rail rule: a bare dataset root with no registry file at all works
at every one of these sites exactly as before this family.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp.subject_registry import (
    Subject,
    SubjectRegistry,
    SubjectRegistryUnconformed,
    copy_registry,
    read_registry,
    replace_registry,
    retired_document,
    write_registry,
)


def _dataset(root: Path, *, with_registry: bool = True) -> Path:
    """One image, one label, and (unless suppressed) the registry that decodes it."""
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    (root / "images" / "2026-03-04").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32)).save(root / "images" / "2026-03-04" / "a_1.jpg")
    (root / "annotations" / "2026-03-04").mkdir(parents=True, exist_ok=True)
    json_io.write_annotations(
        str(root / "annotations" / "2026-03-04" / "a_1.json"),
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)
    if with_registry:
        write_registry(root / "subjects.json", SubjectRegistry(subjects=(Subject(name="bud"),)))
    return root


def _retire(root: Path) -> None:
    """Stand in for the old writer: rename the current registry to the retired filename, the
    same on-disk state a dataset that predates this family is in."""
    (root / "subjects.json").rename(root / "classes.json")


# ── the predicate itself ────────────────────────────────────────────────────────────────────


def test_retired_document_is_none_when_nothing_is_there(tmp_path):
    assert retired_document(tmp_path) is None


def test_retired_document_answers_the_path_when_it_decodes(tmp_path):
    _dataset(tmp_path)
    _retire(tmp_path)
    assert retired_document(tmp_path) == tmp_path / "classes.json"


def test_retired_document_answers_present_even_beside_a_fresh_subjects_json(tmp_path):
    """Whether or not subjects.json exists beside it: a stray write must not be able to hide the
    retired document by placing a fresh one next to it."""
    _dataset(tmp_path)
    root = tmp_path
    (root / "classes.json").write_bytes((root / "subjects.json").read_bytes())
    assert retired_document(root) == root / "classes.json"


def test_a_classes_json_that_does_not_decode_is_not_the_retired_document(tmp_path):
    """A third-party file of that name is no claim the platform ever wrote a registry there."""
    (tmp_path / "classes.json").write_text("not a registry", encoding="utf-8")
    assert retired_document(tmp_path) is None


# ── the three writers refuse beside it ──────────────────────────────────────────────────────


def test_write_registry_refuses_beside_the_retired_document(tmp_path):
    _dataset(tmp_path)
    _retire(tmp_path)
    with pytest.raises(SubjectRegistryUnconformed) as exc:
        write_registry(tmp_path / "subjects.json", SubjectRegistry(subjects=(Subject(name="bud"),)))
    assert "classes.json" in str(exc.value)
    assert "tcip rename-subject-registry" in str(exc.value)
    assert not (tmp_path / "subjects.json").exists()


def test_replace_registry_refuses_beside_the_retired_document(tmp_path):
    _dataset(tmp_path)
    _retire(tmp_path)
    with pytest.raises(SubjectRegistryUnconformed) as exc:
        replace_registry(
            tmp_path / "subjects.json", SubjectRegistry(subjects=(Subject(name="bud"),)),
            expect=None)
    assert "tcip rename-subject-registry" in str(exc.value)
    assert not (tmp_path / "subjects.json").exists()


def test_copy_registry_refuses_at_a_destination_carrying_the_retired_document(tmp_path):
    source = _dataset(tmp_path / "source")
    dest = _dataset(tmp_path / "dest", with_registry=False)
    (dest / "classes.json").write_bytes((source / "subjects.json").read_bytes())
    with pytest.raises(SubjectRegistryUnconformed) as exc:
        copy_registry(source / "subjects.json", dest / "subjects.json")
    assert "tcip rename-subject-registry" in str(exc.value)
    assert not (dest / "subjects.json").exists()


def test_write_registry_admits_valid_work_with_no_registry_at_all(tmp_path):
    """The rail's other half: nothing retired, nothing to refuse."""
    write_registry(tmp_path / "subjects.json", SubjectRegistry(subjects=(Subject(name="bud"),)))
    assert read_registry(tmp_path / "subjects.json").subjects[0].name == "bud"


# ── write_subject_registry's tool door answers {"error": ...} ──────────────────────────────


def test_write_class_map_tool_answers_error_beside_the_retired_document(tmp_path):
    from tcip_mcp.tools.annotation_tools import write_class_map

    _dataset(tmp_path)
    _retire(tmp_path)
    result = write_class_map(str(tmp_path), {"bud": {}})
    assert "error" in result
    assert "classes.json" in result["error"]


# ── register_dataset refuses, naming the path and the command ──────────────────────────────


def test_register_dataset_refuses_beside_the_retired_document(tmp_path):
    from tcip_mcp.tools.project_tools import register_dataset

    root = _dataset(tmp_path / "proj")
    _retire(root)
    result = register_dataset(str(root), crop="walnut")
    assert "error" in result
    assert str(root) in result["error"]
    assert "tcip rename-subject-registry" in result["error"]


def test_register_dataset_admits_valid_work_with_no_registry_at_all(tmp_path):
    from tcip_mcp.tools.project_tools import register_dataset

    root = _dataset(tmp_path / "proj", with_registry=False)
    result = register_dataset(str(root), crop="walnut")
    assert "error" not in result


# ── the named absence refusals ──────────────────────────────────────────────────────────────


def test_resolve_registry_id_map_names_the_retired_file_for_attribute_classification(tmp_path):
    from tcip_mcp.pipelines.data.label_queries import resolve_registry_id_map

    root = _dataset(tmp_path)
    _retire(root)
    with pytest.raises(ValueError) as exc:
        resolve_registry_id_map(root / "annotations" / "2026-03-04", "bud", "condition")
    assert "classes.json" in str(exc.value)
    assert "tcip rename-subject-registry" in str(exc.value)


def test_resolve_statement_registry_names_the_retired_file(tmp_path):
    from tcip_mcp.operationalization import resolve_statement_registry

    root = _dataset(tmp_path)
    _retire(root)
    with pytest.raises(ValueError) as exc:
        resolve_statement_registry(str(root), str(root))
    assert "classes.json" in str(exc.value)
    assert "tcip rename-subject-registry" in str(exc.value)


def test_unmapped_classified_run_names_the_retired_file(tmp_path):
    from tcip_mcp.tools.inference_tools import unmapped_classified_run

    root = _dataset(tmp_path)
    _retire(root)
    msg = unmapped_classified_run(
        {"subject": "bud", "attribute": "condition"}, None, images_dir=str(root / "images" / "2026-03-04"))
    assert msg is not None
    assert "classes.json" in msg
    assert "tcip rename-subject-registry" in msg


# ── the doctor reports it ───────────────────────────────────────────────────────────────────


def test_doctor_reports_the_retired_document(tmp_path):
    from tcip_mcp.cli.doctor import check_retired_subject_registry

    root = _dataset(tmp_path / "proj")
    (root / ".tcip").mkdir(parents=True, exist_ok=True)
    _retire(root)
    findings: list = []
    check_retired_subject_registry(root, findings)
    assert findings
    level, message = findings[0]
    assert level == "warn"
    assert str(root) in message
    assert "tcip rename-subject-registry" in message


def test_doctor_reports_a_stray_undecodable_classes_json_separately(tmp_path):
    from tcip_mcp.cli.doctor import check_retired_subject_registry

    root = _dataset(tmp_path / "proj", with_registry=False)
    (root / ".tcip").mkdir(parents=True, exist_ok=True)
    (root / "classes.json").write_text("not a registry", encoding="utf-8")
    findings: list = []
    check_retired_subject_registry(root, findings)
    assert findings
    level, message = findings[0]
    assert level == "warn"
    assert "stray file" in message


def test_doctor_warns_rather_than_crashes_when_roots_cannot_be_enumerated(tmp_path):
    """A project root holding an experiment's loose record files with no store.db, under the
    database backend, cannot be enumerated through the seam; the check reports it and stops
    rather than raising the StoreError out through the whole doctor run.

    Bound to the sqlite backend explicitly regardless of the ambient run: the refusal this
    proves is specific to a database backend meeting loose files with no database, the exact
    friction the file backend never has."""
    import json

    import tcip_store as ts
    from tcip_store.sqlite_backend import SqliteBackend
    from tcip_store.store import _backend

    from tcip_mcp.cli.doctor import check_retired_subject_registry

    root = _dataset(tmp_path / "proj")
    (root / ".tcip").mkdir(parents=True, exist_ok=True)
    manifest_dir = root / ".tcip" / "experiments" / "exp1" / "model_src"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps({"files": [], "missing": [], "snapshot_errors": []}), encoding="utf-8")

    previous = _backend()
    backend = SqliteBackend()
    ts.bind(backend)
    try:
        findings: list = []
        check_retired_subject_registry(root, findings)
    finally:
        ts.bind(previous)
        backend.close()
    assert findings
    level, message = findings[0]
    assert level == "warn"
    assert "could not enumerate" in message


def test_doctor_reports_nothing_with_no_registry_at_all(tmp_path):
    from tcip_mcp.cli.doctor import check_retired_subject_registry

    root = _dataset(tmp_path / "proj", with_registry=False)
    (root / ".tcip").mkdir(parents=True, exist_ok=True)
    findings: list = []
    check_retired_subject_registry(root, findings)
    assert findings == []


# ── dataset_scope_of still resolves the root ────────────────────────────────────────────────


def test_dataset_scope_of_resolves_a_root_holding_only_the_retired_document(tmp_path):
    from tcip_mcp.audit import dataset_scope_of

    root = _dataset(tmp_path / "proj")
    _retire(root)
    assert dataset_scope_of(str(root)) == root.resolve()


# ── the two bundle doors ────────────────────────────────────────────────────────────────────


def test_archive_project_refuses_naming_the_retired_document(tmp_path):
    from tcip_mcp.tools.project_tools import archive_project

    root = _dataset(tmp_path / "proj")
    _retire(root)
    result = archive_project(str(root), str(tmp_path / "bundle.zip"))
    assert "error" in result
    assert "classes.json" in result["error"]
    assert "tcip rename-subject-registry" in result["error"]
    assert not (tmp_path / "bundle.zip").exists()


def test_archive_project_admits_valid_work_with_no_registry_at_all(tmp_path):
    from tcip_mcp.tools.project_tools import archive_project

    root = _dataset(tmp_path / "proj", with_registry=False)
    result = archive_project(str(root), str(tmp_path / "bundle.zip"))
    assert "error" not in result
    assert (tmp_path / "bundle.zip").is_file()


def test_import_project_refuses_naming_the_retired_document_and_the_hand_recipe(tmp_path):
    import zipfile

    from tcip_mcp.tools.project_tools import archive_project, import_project

    root = _dataset(tmp_path / "proj")
    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(str(root), str(zip_path))
    # Simulate an archive made before the rename: the bundled member is still named classes.json.
    rebuilt = tmp_path / "rebuilt.zip"
    with zipfile.ZipFile(str(zip_path)) as src, zipfile.ZipFile(str(rebuilt), "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            name = "classes.json" if item.filename == "subjects.json" else item.filename
            dst.writestr(name, data)

    dest = tmp_path / "dest"
    result = import_project(str(rebuilt), str(dest))
    assert "error" in result
    assert "classes.json" in result["error"]
    assert "rename it to subjects.json" in result["error"]
    assert not dest.exists() or not any(dest.iterdir())
