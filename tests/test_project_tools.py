"""Tests for project management tools."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

import tcip_store
from tcip_mcp.tools.project_tools import (
    init_project,
    inspect_project,
    archive_project,
    import_project,
    read_datasets,
    register_dataset,
    upsert_dataset,
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
    assert len(same) == 1  # one entry for the id: the move updated the path, not duplicated
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


def test_inspect_project_folds_in_recent_activity(tmp_path: Path):
    from tcip_mcp.tools.meta_tools import claude_reports

    init_project(str(tmp_path))
    status = inspect_project(str(tmp_path))
    assert status["recent_activity"] == {}  # no history yet: genuinely empty, not corrupt

    claude_reports(str(tmp_path), category="missing_tool", detail="a")
    status = inspect_project(str(tmp_path))
    assert status["recent_activity"]["reports_since_last_retrospective"] == 1


def test_inspect_project_surfaces_corrupt_status_honestly(tmp_path: Path):
    init_project(str(tmp_path))
    status_path = tmp_path / ".tcip" / "state" / "project_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text("{not valid json", encoding="utf-8")

    status = inspect_project(str(tmp_path))
    assert "status_unavailable" in status["recent_activity"]
    # Live counts must stay unaffected by a corrupt status file: different store, different rail.
    assert status["initialized"] is True


def test_set_active_project_folds_in_recent_activity(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path))
    import tcip_mcp.web_client as web_client
    from tcip_mcp.tools.meta_tools import claude_reports
    from tcip_mcp.tools.project_tools import set_active_project
    from tcip_mcp.workspace import project_path

    # Stub the GUI notification so the result is deterministic regardless of whether a tcip-web
    # backend happens to be listening on this machine (matches test_set_active_project.py).
    monkeypatch.setattr(web_client, "post_panel_event", lambda *a, **k: {"delivered": False})

    init_project(str(project_path("proj_a")))
    claude_reports(str(project_path("proj_a")), category="missing_tool", detail="a")

    result = set_active_project("proj_a")
    assert result["recent_activity"]["reports_since_last_retrospective"] == 1


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
    # The class registry decodes the labels' names: a self-contained bundle must carry it, or the
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
    # dataset.json travelled with the data: identity (id/crop/fingerprint) survives the round-trip.
    import json

    restored_id = json.loads((dest / "dataset.json").read_text())
    assert restored_id == {"crop": "hazelnut", "id": reg["id"], "fingerprint": reg["fingerprint"]}


def test_archive_project_includes_bespoke_model_source(tmp_path: Path):
    """A bespoke run's snapshotted .py source (model_src/, written by snapshot_model_source) must
    travel with the archive, or a published/archived project bundles the provenance manifest
    without the code it describes and can't rerun its own pipeline from the archive alone."""
    src = tmp_path / "src_project"
    init_project(str(src))

    model_src = src / ".tcip" / "experiments" / "exp_001" / "model_src" / "abcd1234"
    model_src.mkdir(parents=True)
    (model_src / "my_model.py").write_text("def build(): ...\n", encoding="utf-8")
    manifest_dir = src / ".tcip" / "experiments" / "exp_001" / "model_src"
    (manifest_dir / "manifest.json").write_text("{}", encoding="utf-8")

    zip_path = tmp_path / "export.zip"
    exported = archive_project(str(src), str(zip_path), include_models=True)
    assert "error" not in exported

    import zipfile

    with zipfile.ZipFile(str(zip_path)) as zf:
        names = zf.namelist()
    py_entries = [n for n in names if n.endswith("my_model.py")]
    assert py_entries, f"model_src's .py source is missing from the archive: {names}"
    assert any(n.endswith("manifest.json") for n in names)


class _BarrierId(str):
    """A dataset id that parks at a barrier the first time it is rendered for the registry's sort.

    The rendering sits after the registry has been read and before it is written back, which is
    the window a lost update opens in, so a writer holding no lock across that pair waits there
    until every other writer has read the same state it did.
    """

    barrier: threading.Barrier

    def __str__(self) -> str:
        if not self.__dict__.get("parked"):
            self.__dict__["parked"] = True
            try:
                type(self).barrier.wait()
            except threading.BrokenBarrierError:
                pass
        return str.__str__(self)


def test_concurrent_registrations_both_survive_in_the_registry(tmp_path: Path):
    """Two writers adding different datasets at once both land. Each reads the whole list,
    drops one entry and writes the list back, so a pair that is not serialized writes lists
    assembled before the other's entry existed and one dataset's identity disappears."""
    project = tmp_path / "proj"
    project.mkdir()
    _BarrierId.barrier = threading.Barrier(2, timeout=1.0)
    failures: list[BaseException] = []

    def register(name: str) -> None:
        try:
            upsert_dataset(project, {"id": _BarrierId(name), "path": str(tmp_path / name),
                                     "crop": "hazelnut", "fingerprint": name})
        except BaseException as exc:  # recorded, never swallowed into a passing test
            failures.append(exc)

    threads = [threading.Thread(target=register, args=(name,)) for name in ("aaa", "bbb")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not failures, failures
    assert sorted(r["id"] for r in read_datasets(project)) == ["aaa", "bbb"]


def test_an_undecodable_dataset_registry_refuses_and_an_absent_one_reads_empty(tmp_path: Path):
    """A registry read as empty would make the next registration write a one-entry list and drop
    every other dataset identity the project had recorded, so corruption is not absence here."""
    project = tmp_path / "proj"
    (project / ".tcip").mkdir(parents=True)

    assert read_datasets(project) == []  # a project with nothing registered yet

    (project / ".tcip" / "datasets.json").write_bytes(b'[{"id": "aaa"')  # truncated mid-list
    with pytest.raises(tcip_store.DecodeError):
        read_datasets(project)


def test_an_undecodable_identity_document_refuses_rather_than_minting_a_fresh_id(tmp_path: Path):
    """Minting a new id over an identity that will not decode severs every experiment, split and
    delivered number citing the old one, so the tool refuses and names the document."""
    src = tmp_path / "proj"
    _make_dataset(src)
    truncated = b'{"id": "known_id"'
    (src / "dataset.json").write_bytes(truncated)

    refused = register_dataset(str(src), crop="hazelnut")

    assert "error" in refused and "dataset.json" in refused["error"]
    assert (src / "dataset.json").read_bytes() == truncated  # nothing written over it
    assert read_datasets(src) == []


def test_scaffolding_twice_keeps_the_config_the_first_run_wrote(tmp_path: Path):
    """Scaffolding is idempotent and never overwrites a config a human has since edited."""
    from tcip_mcp.tools.project_tools import _scaffold_project

    _scaffold_project(str(tmp_path))
    edited = b'[project]\nname = "edited by hand"\n'
    (tmp_path / ".tcip" / "config.toml").write_bytes(edited)

    _scaffold_project(str(tmp_path))

    assert (tmp_path / ".tcip" / "config.toml").read_bytes() == edited
    assert inspect_project(str(tmp_path))["has_config"] is True
