"""Tests for project management tools."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

import tcip_store
from tcip_mcp.registry_paths import stored_path
from tcip_mcp.tools.project_tools import (
    initialize_project,
    inspect_project,
    archive_project,
    import_project,
    read_datasets,
    register_dataset,
    upsert_dataset,
    _external_dataset_paths,
)
from tests._record_damage_fixtures import damage_record


def _make_dataset(root: Path) -> None:
    """A minimal nested-schema dataset (image + label + registry) for identity tests."""
    from PIL import Image

    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import SubjectRegistry, Subject
    from tests._producer_fixtures import label_image, registry_over

    image = root / "images" / "2-11-26" / "img_000.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32)).save(image)
    label_image(image, [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)
    registry_over(root, SubjectRegistry(subjects=(Subject(name="bud"),)))


def _initialized(path: Path) -> dict:
    """``path`` made a project through the platform's own creation door; its record."""
    result = initialize_project(str(path), "Test project", "north orchard")
    assert "error" not in result, result
    return result


def test_register_dataset_writes_identity_and_registers(tmp_path: Path):
    import json

    src = tmp_path / "proj"
    _make_dataset(src)

    res = register_dataset(tmp_path, str(src), crop="currant")
    assert "error" not in res
    assert res["crop"] == "currant" and res["id"] and res["fingerprint"]

    # dataset.json holds {crop, id, fingerprint}.
    ident = json.loads((src / "dataset.json").read_text())
    assert ident == {"crop": "currant", "id": res["id"], "fingerprint": res["fingerprint"]}
    # the project registry knows the dataset by id and path, its identity held by dataset.json.
    assert read_datasets(tmp_path) == [{"id": res["id"], "path": stored_path(src, tmp_path)}]


def test_register_dataset_requires_crop_and_keeps_id_stable(tmp_path: Path):
    src = tmp_path / "proj"
    _make_dataset(src)

    assert "error" in register_dataset(tmp_path, str(src), crop="")  # the expert's fact, required

    first = register_dataset(tmp_path, str(src), crop="currant")
    again = register_dataset(tmp_path, str(src), crop="currant")
    assert again["id"] == first["id"]  # id minted once, preserved across re-runs
    assert len(read_datasets(tmp_path)) == 1  # not duplicated in the registry


def test_register_dataset_reconciles_a_move_by_id(tmp_path: Path):
    import shutil

    src = tmp_path / "orig"
    _make_dataset(src)
    reg = register_dataset(tmp_path, str(src), crop="currant")

    moved = tmp_path / "moved"
    tcip_store.release_root(src)
    shutil.copytree(src, moved)  # same content, new path
    register_dataset(tmp_path, str(moved), crop="currant")

    regs = read_datasets(tmp_path)
    same = [r for r in regs if r["id"] == reg["id"]]
    assert len(same) == 1  # one entry for the id: the move updated the path, not duplicated
    assert same[0] == {"id": reg["id"], "path": stored_path(moved, tmp_path)}


def test_initialize_project(tmp_path: Path):
    result = _initialized(tmp_path)
    assert (tmp_path / ".tcip").is_dir()
    assert (tmp_path / ".tcip" / "artifacts").is_dir()
    assert (tmp_path / ".tcip" / "models").is_dir()
    assert result["display_name"] == "Test project" and result["id"]


def test_inspect_project(tmp_path: Path):
    status = inspect_project(tmp_path)
    assert status["id"] is None
    assert "initialize_project" in status["record_problem"]

    record = _initialized(tmp_path)
    status = inspect_project(tmp_path)
    assert (status["id"], status["display_name"]) == (record["id"], "Test project")
    assert status["record_problem"] is None


def test_inspect_project_folds_in_recent_activity(tmp_path: Path):
    from tcip_mcp.tools.meta_tools import report_friction

    _initialized(tmp_path)
    status = inspect_project(tmp_path)
    assert status["recent_activity"] == {  # no history yet: counted from the log, not corrupt
        "reports_since_last_retrospective": 0, "reports_since_last_distillation": 0,
        "retrospectives_since_last_distillation": 0}

    report_friction(tmp_path, category="missing_tool", detail="a")
    status = inspect_project(tmp_path)
    assert status["recent_activity"]["reports_since_last_retrospective"] == 1


def test_inspect_project_folds_in_last_retrospective_by_id_not_path(tmp_path: Path):
    from tcip_mcp.tools.meta_tools import write_retrospective

    _initialized(tmp_path)
    write_retrospective(
        tmp_path, project_id="p", task="t", worked="w", did_not_work="d",
    )

    status = inspect_project(tmp_path)
    last = status["recent_activity"]["last_retrospective"]
    assert last["project_id"] == "p"
    assert "path" not in last


def test_inspect_project_surfaces_an_undecodable_audit_log_honestly(tmp_path: Path):
    import sqlite3

    from tcip_mcp.tools.meta_tools import report_friction
    from tcip_store.file_backend import database_file

    _initialized(tmp_path)
    report_friction(tmp_path, category="missing_tool", detail="a")
    conn = sqlite3.connect(str(database_file(str(tmp_path))), isolation_level=None)
    try:
        conn.execute("update log_entries set entry = ? where id = "
                     "(select max(id) from log_entries)", (b"{not valid json",))
    finally:
        conn.close()

    status = inspect_project(tmp_path)
    assert "undecodable" in status["recent_activity"]["status_unavailable"]
    # The record reads unaffected by a damaged log: different store, different rail.
    assert status["record_problem"] is None


def test_inspect_project_against_a_nonexistent_workspace_creates_nothing(
    tmp_path: Path, monkeypatch
):
    ws = tmp_path / "no_such_workspace"
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    proj = tmp_path / "proj"
    proj.mkdir()

    inspect_project(proj)

    assert not ws.exists()


def test_export_import_roundtrip(tmp_path: Path):
    """archive_project -> import_project -> inspect_project recovers the project."""
    from PIL import Image

    src = tmp_path / "src_project"
    date = "2-11-26"
    images = src / "images" / date
    images.mkdir(parents=True)
    record = _initialized(src)

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp import subject_registry
    from tcip_mcp.dataset_layout import label_key
    from tcip_mcp.subject_registry import SubjectRegistry, Subject
    from tests._producer_fixtures import label_image, registry_over

    Image.new("RGB", (64, 64)).save(images / "img_000.jpg")
    label_image(images / "img_000.jpg", [Annotation(subject="bud", geometry=BBox(10, 10, 30, 30))],
                64, 64)
    # A multispectral capture the sensor wrote one file per band for: the manifest beside the
    # bands is what makes those files one logical image, so the bundle has to carry all of them.
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    bands = {}
    for band, wavelength in (("green", 560.0), ("red", 668.0)):
        band_file = images / f"cap_001_{band}.tif"
        Image.new("L", (64, 64)).save(band_file)
        bands[band] = band_file
    manifest = write_band_group_manifest(
        images, "cap_001", bands, central_wavelength_nm={"green": 560.0, "red": 668.0},
        source="explicit-manifest",
    )
    # A sensor that writes one multi-band file per capture instead of one file per band. It
    # enumerates as a logical image on its own, so a bundle that drops it drops that capture.
    import numpy as np

    npz_image = images / "cap_002.npz"
    np.savez(npz_image, bands=np.zeros((2, 64, 64), dtype=np.uint16))
    # The subject registry decodes the labels' names: a self-contained bundle must carry it, or the
    # archived annotations are unreadable on the other end. One nested subjects.json at the root.
    registry_over(src, SubjectRegistry(subjects=(
        Subject(name="bud", description="a currant bud"),)))
    reg = register_dataset(src, str(src), crop="currant")  # identity travels with the data

    zip_path = tmp_path / "export.zip"
    exported = archive_project(src, str(zip_path))
    assert "error" not in exported
    assert zip_path.is_file()

    dest = tmp_path / "restored"
    imported = import_project(str(zip_path), str(dest))
    assert "error" not in imported
    assert imported["files_extracted"] == exported["files_added"]

    status = inspect_project(dest)
    assert status["id"] == record["id"]
    # inspect_project counts raw image files, so the two sibling bands count separately here;
    # the logical-image count the band group folds them into is asserted below.
    assert status["image_count"] == 3
    (restored_label,) = json_io.read_label_document(label_key(dest, date, "img_000")).annotations
    assert restored_label.subject == "bud"
    # Comparing the enumeration rather than a file list pins the archive to the same notion of
    # "image" the platform reads the restored directory back with.
    from tcip_mcp.pipelines.image_utils import list_logical_images

    restored_images = dest / "images" / date
    restored_manifest = restored_images / manifest.name
    assert restored_manifest.is_file()
    assert restored_manifest.read_bytes() == manifest.read_bytes()
    assert (restored_images / npz_image.name).read_bytes() == npz_image.read_bytes()
    assert sorted(list_logical_images(restored_images)) == sorted(
        list_logical_images(images)
    ) == ["cap_001", "cap_002", "img_000"]
    # The registry survived, so the restored labels are still decodable.
    restored = subject_registry.read_registry(dest)
    assert [s.name for s in restored.subjects] == ["bud"]
    # dataset.json traveled with the data: identity (id/crop/fingerprint) survives the round-trip.
    import json

    restored_id = json.loads((dest / "dataset.json").read_text())
    assert restored_id == {"crop": "currant", "id": reg["id"], "fingerprint": reg["fingerprint"]}


def test_external_dataset_paths_names_an_external_registry_entry(tmp_path: Path):
    """A registered dataset outside the project is named as external."""
    project = tmp_path / "project"
    dataset = tmp_path / "dataset"  # a sibling of project, never nested under it: external
    project.mkdir()
    dataset.mkdir()
    _make_dataset(dataset)
    register_dataset(project, str(dataset), crop="currant")
    entry = read_datasets(project)[0]
    assert entry["path"] == str(dataset.resolve())  # external entries store absolute

    assert _external_dataset_paths(project) == [str(dataset.resolve())]


def test_archive_project_includes_bespoke_model_source(tmp_path: Path):
    """A bespoke run's snapshotted .py source (model_src/, written by snapshot_model_source) must
    travel with the archive, or a published/archived project bundles the provenance manifest
    without the code it describes and can't rerun its own pipeline from the archive alone."""
    src = tmp_path / "src_project"
    _initialized(src)

    model_src = src / ".tcip" / "experiments" / "exp_001" / "model_src" / "abcd1234"
    model_src.mkdir(parents=True)
    (model_src / "my_model.py").write_text("def build(): ...\n", encoding="utf-8")
    manifest_dir = src / ".tcip" / "experiments" / "exp_001" / "model_src"
    (manifest_dir / "manifest.json").write_text("{}", encoding="utf-8")

    zip_path = tmp_path / "export.zip"
    exported = archive_project(src, str(zip_path), include_models=True)
    assert "error" not in exported

    import zipfile

    with zipfile.ZipFile(str(zip_path)) as zf:
        names = zf.namelist()
    py_entries = [n for n in names if n.endswith("my_model.py")]
    assert py_entries, f"model_src's .py source is missing from the archive: {names}"
    assert any(n.endswith("manifest.json") for n in names)


def test_archive_project_reports_checkpoints_excluded_by_default(tmp_path: Path):
    """A checkpoint under .tcip/models is dropped by include_models=False, and counted."""
    src = tmp_path / "src_project"
    _initialized(src)
    (src / ".tcip" / "models" / "m.pt").write_bytes(b"weights")

    result = archive_project(src, str(tmp_path / "export.zip"))

    assert "error" not in result
    assert result["checkpoints_excluded"] == 1

    result_included = archive_project(src, str(tmp_path / "export2.zip"), include_models=True)
    assert result_included["checkpoints_excluded"] == 0


def test_archive_project_includes_a_registered_run_checkpoint_outside_tcip_models(tmp_path: Path):
    """A completed run's checkpoint sits in its own run directory,
    ``.tcip/experiments/<experiment_id>/model_final.pt``, not under ``.tcip/models/``.
    ``include_models=True`` must bundle it there too, or a breeder who trusts the flag gets an
    archive with no model in it at all."""
    from tests._verified_checkpoint_fixtures import finished_run

    src = tmp_path / "src_project"
    _initialized(src)
    finished_run(src, experiment_id="exp_ckpt_bundle")

    result = archive_project(src, str(tmp_path / "export.zip"), include_models=True)
    assert "error" not in result

    import zipfile

    with zipfile.ZipFile(str(tmp_path / "export.zip")) as zf:
        names = zf.namelist()
    assert any(n.endswith("model_final.pt") for n in names), (
        f"a registered run checkpoint outside .tcip/models/ is missing from the archive: {names}"
    )


def test_import_project_admits_a_bundle_holding_a_registered_run_checkpoint(tmp_path: Path):
    """archive_project(include_models=True) bundles a completed run's checkpoint from its run
    directory, and import_project admits it: the restored run's final status still names it,
    and the registry the restored project reads lists it."""
    from tcip_mcp.experiments import experiment_dir, observe
    from tcip_mcp.model_registry import ModelRegistry
    from tests._verified_checkpoint_fixtures import finished_run

    src = tmp_path / "src_project"
    _initialized(src)
    exp_id = "exp_roundtrip"
    finished_run(src, experiment_id=exp_id)

    zip_path = tmp_path / "export.zip"
    exported = archive_project(src, str(zip_path), include_models=True)
    assert "error" not in exported, exported

    dest = tmp_path / "restored"
    imported = import_project(str(zip_path), str(dest))

    assert "error" not in imported, imported
    restored = observe(experiment_dir(exp_id, project=dest)).checkpoint
    assert restored is not None and Path(restored["path"]).is_file()
    assert exp_id in {m["name"] for m in ModelRegistry(str(dest)).list_models()}


def _internal_foreign_checkpoint(src: Path) -> Path:
    """A foreign checkpoint registered from inside the project's own tree."""
    from tcip_mcp.model_registry import ModelRegistry
    from tests._verified_checkpoint_fixtures import checkpoint_file

    weights = src / ".tcip" / "models" / "internal.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file(weights, "weights registered from inside the project")
    ModelRegistry(str(src)).register_model("internal", str(weights), {})
    return weights


def test_import_project_admits_a_registered_checkpoint_with_no_disclosure(tmp_path: Path):
    """A checkpoint registered under the project's own tree comes back from an archive/import
    round trip with nothing to disclose: the writer already spelled it relative to the
    registry's scope root, so the moved tree's registry still resolves under it. The stored
    entry itself stays relative; the resolved response is absolute; weights load by digest
    either way, since loading never reads the stored path."""
    from tcip_mcp.model_registry import ModelRegistry, read_registry_index, registry_index_key

    src = tmp_path / "src_project"
    _initialized(src)
    _internal_foreign_checkpoint(src)

    zip_path = tmp_path / "export.zip"
    exported = archive_project(src, str(zip_path), include_models=True)
    assert "error" not in exported, exported

    dest = tmp_path / "restored"
    imported = import_project(str(zip_path), str(dest))

    assert "error" not in imported, imported
    assert imported["dataset_paths_unresolved"] == []
    assert imported["checkpoint_paths_unresolved"] == []
    assert imported["external_checkpoints"] == []

    raw = tcip_store.read(registry_index_key(dest))
    stored = raw["entries"][0]["checkpoint_path"]
    assert not Path(stored).is_absolute(), stored
    assert ".." not in Path(stored).parts

    entries = read_registry_index(dest)
    assert entries[0]["checkpoint_path"] == stored

    (resolved,) = [m["checkpoint_path"] for m in ModelRegistry(str(dest)).list_models()
                   if m["name"] == entries[0]["name"]]
    assert Path(resolved).is_absolute()
    assert Path(resolved).is_file()


def test_import_project_keeps_a_relative_entry_relative_when_the_archive_carries_no_checkpoint(
    tmp_path: Path,
):
    """A relative registry entry whose weights the archive legitimately dropped
    (``include_models=False``) must come back still relative and disclosed as unresolved: the
    staging conform's no-match fallback must never write the entry's own staging directory's
    absolute path over it, which would misfile an internal-but-absent entry as designed-external
    and leave a path into a directory the door is about to delete permanently in the registry."""
    from tcip_mcp.model_registry import read_registry_index, registry_index_key

    src = tmp_path / "src_project"
    _initialized(src)
    _internal_foreign_checkpoint(src)

    stored_before = read_registry_index(src)[0]["checkpoint_path"]
    assert not Path(stored_before).is_absolute(), stored_before

    zip_path = tmp_path / "export.zip"
    exported = archive_project(src, str(zip_path), include_models=False)
    assert "error" not in exported, exported

    dest = tmp_path / "restored"
    imported = import_project(str(zip_path), str(dest))

    assert "error" not in imported, imported
    stored_after = tcip_store.read(registry_index_key(dest))["entries"][0]["checkpoint_path"]
    assert stored_after == stored_before
    assert imported["checkpoint_paths_unresolved"] == [stored_before]
    assert imported["external_checkpoints"] == []


def test_import_project_discloses_a_designed_external_checkpoint_separately_from_unresolved(
    tmp_path: Path,
):
    """A registry entry that is a genuine designed-external claim (outside the project tree
    entirely) must appear in ``external_checkpoints``, never counted toward
    ``checkpoint_paths_unresolved``, which names only an entry expected to resolve under the
    tree that does not."""
    from tcip_mcp.model_registry import ModelRegistry
    from tests._verified_checkpoint_fixtures import checkpoint_file

    src = tmp_path / "src_project"
    _initialized(src)
    internal_dir = src / ".tcip" / "models"
    internal_dir.mkdir(parents=True, exist_ok=True)
    internal_ckpt = checkpoint_file(internal_dir / "internal.pt", "internal weights")
    external_dir = tmp_path / "elsewhere"
    external_dir.mkdir()
    external_ckpt = checkpoint_file(external_dir / "external.pt", "external weights")

    reg = ModelRegistry(str(src))
    reg.register_model("m_internal", str(internal_ckpt), {})
    reg.register_model("m_external", str(external_ckpt), {})

    zip_path = tmp_path / "export.zip"
    exported = archive_project(src, str(zip_path), include_models=True)
    assert "error" not in exported, exported

    dest = tmp_path / "restored"
    imported = import_project(str(zip_path), str(dest))

    assert "error" not in imported, imported
    assert imported["checkpoint_paths_unresolved"] == []
    assert imported["external_checkpoints"] == [
        {"checkpoint_path": str(external_ckpt), "exists": True},
    ]


def test_archive_project_bundles_a_registered_tcip_models_checkpoint_once(tmp_path: Path):
    """A checkpoint sitting under .tcip/models/ that is also a registry entry is one file to
    the two checkpoint arms that recognize it; it lands in the bundle once."""
    from tcip_mcp.model_registry import ModelRegistry, checkpoint_files
    from tests._verified_checkpoint_fixtures import checkpoint_file

    src = tmp_path / "src_project"
    _initialized(src)
    ckpt = checkpoint_file(src / ".tcip" / "models" / "m.pt", "weights")
    ModelRegistry(str(src)).register_model("m", str(ckpt), {})

    assert Path(ckpt).resolve() in checkpoint_files(src)

    result = archive_project(src, str(tmp_path / "export.zip"), include_models=True)
    assert "error" not in result, result

    import zipfile

    with zipfile.ZipFile(str(tmp_path / "export.zip")) as zf:
        names = zf.namelist()
    matching = [n for n in names if n.endswith("m.pt")]
    assert len(matching) == 1, f"m.pt bundled more than once: {names}"


def test_a_checkpoint_registered_from_anywhere_in_the_project_travels_only_with_models(
    tmp_path: Path,
):
    """A checkpoint registered from a directory of the project's own, neither ``.tcip/models``
    nor a run's, is a checkpoint: ``include_models=False`` leaves it out and counts it, and
    ``include_models=True`` carries it."""
    import zipfile

    from tcip_mcp.model_registry import ModelRegistry
    from tests._verified_checkpoint_fixtures import checkpoint_file

    src = tmp_path / "src_project"
    _initialized(src)
    (src / "weights").mkdir()
    ckpt = checkpoint_file(src / "weights" / "foreign.pt", "weights registered in the project")
    ModelRegistry(str(src)).register_model("foreign", str(ckpt), {})

    without = archive_project(src, str(tmp_path / "without.zip"), include_models=False)
    with_models = archive_project(src, str(tmp_path / "with.zip"), include_models=True)

    assert without["checkpoints_excluded"] == 1
    with zipfile.ZipFile(str(tmp_path / "without.zip")) as zf:
        assert "weights/foreign.pt" not in zf.namelist()
    with zipfile.ZipFile(str(tmp_path / "with.zip")) as zf:
        assert "weights/foreign.pt" in zf.namelist()
    assert with_models["checkpoints_excluded"] == 0


def test_archive_project_admits_a_symlink_spelled_project(tmp_path: Path):
    """A project reached through a symlink archives rather than raising ValueError out of the
    door: archive_project resolves the project once and uses that resolved root for both
    member.relative_to and the include_models comparison."""
    real = tmp_path / "real_project"
    _make_dataset(real)
    link = tmp_path / "linked_project"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks not available on this machine: {exc}")

    result = archive_project(link, str(tmp_path / "export.zip"))

    assert "error" not in result
    assert result["files_added"] > 0


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
            upsert_dataset(project, {"id": _BarrierId(name), "path": str(tmp_path / name)})
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
    from tcip_mcp.tools.project_tools import dataset_registry_key

    project = tmp_path / "proj"
    (project / ".tcip").mkdir(parents=True)

    assert read_datasets(project) == []  # a project with nothing registered yet

    upsert_dataset(project, {"id": "aaa", "path": str(project)})
    damage_record(dataset_registry_key(project), b'[{"id": "aaa"')  # truncated mid-list
    with pytest.raises(tcip_store.DecodeError):
        read_datasets(project)


def test_an_identity_minted_while_this_registration_ran_is_adopted_not_overwritten(
    tmp_path: Path, monkeypatch
):
    """A dataset's id is minted once. Two first-time registrations both read no identity, so the
    write has to be conditional on there still being none: the loser adopts what committed instead
    of stamping its own id over an id other records may already cite.

    The window is between reading the absent identity and writing the minted one, which is exactly
    where the id is drawn. A competing identity lands there, so the second write conflicts, the
    call re-reads the committed document, and the id it reports and registers is that one.
    """
    import json
    import uuid as uuid_module

    src = tmp_path / "proj"
    _make_dataset(src)
    committed = b'{"crop": "currant", "id": "committed_id", "fingerprint": "written first"}\n'
    real_uuid4 = uuid_module.uuid4
    raced: list[str] = []

    def racing_uuid4():
        if not raced:
            raced.append("minted")
            (src / "dataset.json").write_bytes(committed)
        return real_uuid4()

    monkeypatch.setattr(uuid_module, "uuid4", racing_uuid4)

    result = register_dataset(tmp_path, str(src), crop="currant")

    assert raced == ["minted"]
    assert "error" not in result
    assert result["id"] == "committed_id"
    assert json.loads((src / "dataset.json").read_text(encoding="utf-8"))["id"] == "committed_id"
    assert [r["id"] for r in read_datasets(tmp_path)] == ["committed_id"]


def test_an_undecodable_identity_document_refuses_rather_than_minting_a_fresh_id(tmp_path: Path):
    """Minting a new id over an identity that will not decode severs every experiment, split and
    delivered number citing the old one, so the tool refuses and names the document."""
    src = tmp_path / "proj"
    _make_dataset(src)
    truncated = b'{"id": "known_id"'
    (src / "dataset.json").write_bytes(truncated)

    refused = register_dataset(tmp_path, str(src), crop="currant")

    assert "error" in refused and "dataset.json" in refused["error"]
    assert (src / "dataset.json").read_bytes() == truncated  # nothing written over it
    assert read_datasets(tmp_path) == []


def test_initializing_twice_leaves_what_the_first_run_created(tmp_path: Path):
    """A second creation with the same display name and site keeps the record's id and touches
    nothing the first run created."""
    first = _initialized(tmp_path)
    (tmp_path / ".tcip" / "artifacts" / "kept.txt").write_text("kept", encoding="utf-8")

    again = _initialized(tmp_path)

    assert again["id"] == first["id"]
    assert (tmp_path / ".tcip" / "models").is_dir()
    assert (tmp_path / ".tcip" / "artifacts" / "kept.txt").read_text(encoding="utf-8") == "kept"


# ── the project record ────────────────────────────────────────────────────────


def test_initialize_project_records_the_id_display_name_and_site(tmp_path: Path):
    from tcip_mcp.project_record import read_record

    result = _initialized(tmp_path)

    assert read_record(tmp_path) == {"id": result["id"], "display_name": "Test project",
                                     "site": "north orchard"}


def test_initialize_project_refuses_a_different_site_than_the_one_already_recorded(
    tmp_path: Path,
):
    from tcip_mcp.project_record import read_record

    _initialized(tmp_path)

    result = initialize_project(str(tmp_path), "Test project", "south orchard")

    assert "error" in result
    assert "north orchard" in result["error"]
    assert "south orchard" in result["error"]
    assert read_record(tmp_path)["site"] == "north orchard"


def test_initialize_project_refuses_a_different_display_name_than_the_one_already_recorded(
    tmp_path: Path,
):
    from tcip_mcp.project_record import read_record

    _initialized(tmp_path)

    result = initialize_project(str(tmp_path), "Another name", "north orchard")

    assert "error" in result
    assert "rename" in result["error"]
    assert read_record(tmp_path)["display_name"] == "Test project"


def test_initialize_project_refuses_an_empty_site_leaving_nothing_on_disk(tmp_path: Path):
    """A refused site is validated before anything is created: the destination is left exactly
    as it was, not half-scaffolded."""
    dest = tmp_path / "fresh_project"

    result = initialize_project(str(dest), "Test project", "   ")

    assert "error" in result
    assert not dest.exists()


def test_initialize_project_refuses_an_empty_display_name_leaving_nothing_on_disk(
    tmp_path: Path,
):
    dest = tmp_path / "fresh_project"

    result = initialize_project(str(dest), "  ", "north orchard")

    assert "error" in result and "display name" in result["error"]
    assert not dest.exists()


def test_initialize_project_scaffolds_a_relative_path_where_it_resolves(tmp_path: Path,
                                                                        monkeypatch):
    """A relative project_path scaffolds and records at the absolute location it resolves to,
    rather than the record write refusing a relative root after ``.tcip`` already exists."""
    from tcip_mcp.project_record import read_record

    monkeypatch.chdir(tmp_path)

    result = initialize_project("relative_proj", "Test project", "north orchard")

    assert "error" not in result
    assert (tmp_path / "relative_proj" / ".tcip").is_dir()
    assert read_record(tmp_path / "relative_proj")["site"] == "north orchard"


def test_initialize_project_refuses_a_present_but_invalid_record(tmp_path: Path):
    """The door surfaces the reader's own refusal rather than the store's raw exception."""
    from tcip_mcp.project_record import project_record_key

    key = project_record_key(tmp_path)
    tcip_store.replace(key, {"not_site": "whatever"}, expect=tcip_store.Version.ABSENT)

    result = initialize_project(str(tmp_path), "Test project", "north orchard")

    assert "error" in result
    assert "does not hold an id, a display name and a site" in result["error"]


def test_initialize_project_refuses_an_undecodable_record(tmp_path: Path):
    """The store's own DecodeError is a StoreError, caught and returned as the door's error."""
    from tcip_mcp.project_record import project_record_key

    record = _initialized(tmp_path)
    damage_record(project_record_key(tmp_path), b"{not valid json")

    result = initialize_project(str(tmp_path), record["display_name"], record["site"])

    assert "error" in result
    assert "does not decode" in result["error"]


def test_initialize_project_records_on_a_directory_that_gained_tcip_with_no_creating_door(
    tmp_path: Path,
):
    """The reachable state a store write with no door leaves (``report_friction`` on a bare
    directory): ``initialize_project`` on it afterward records the project the same way it would
    on a truly fresh directory, since the writer's create-only write does not distinguish the
    two."""
    from tcip_mcp.tools.meta_tools import report_friction

    report_friction(tmp_path, category="missing_tool", detail="a")
    assert (tmp_path / ".tcip").is_dir()

    result = _initialized(tmp_path)

    assert result["site"] == "north orchard"


def test_inspect_project_reports_the_record_fields_across_project_states(tmp_path: Path):
    """A path with no record carries the absent-record problem text; a project with a record
    carries its fields and no problem."""
    from tcip_mcp.tools.meta_tools import report_friction

    bare = tmp_path / "no_tcip"
    bare.mkdir()
    status = inspect_project(bare)
    assert status["site"] is None
    assert "initialize_project" in status["record_problem"]

    recordless = tmp_path / "recordless"
    recordless.mkdir()
    report_friction(recordless, category="missing_tool", detail="a")
    status = inspect_project(recordless)
    assert status["site"] is None
    assert "initialize_project" in status["record_problem"]

    recorded = tmp_path / "recorded"
    _initialized(recorded)
    status = inspect_project(recorded)
    assert status["site"] == "north orchard"
    assert status["record_problem"] is None


def test_inspect_project_reports_an_invalid_record(tmp_path: Path):
    from tcip_mcp.project_record import project_record_key

    _initialized(tmp_path)
    key = project_record_key(tmp_path)
    current = tcip_store.read_versioned(key).version
    tcip_store.replace(key, {"not_site": "x"}, expect=current)

    status = inspect_project(tmp_path)

    assert status["site"] is None
    assert "does not hold an id" in status["record_problem"]


def test_archive_and_import_carry_the_project_record(tmp_path: Path):
    """initialize_project -> archive_project -> import_project round-trips a project's record,
    its id included, inside the database the archive copies, with no operator step between the
    two doors."""
    src = tmp_path / "src_project"
    record = _initialized(src)

    zip_path = tmp_path / "export.zip"
    exported = archive_project(src, str(zip_path))
    assert "error" not in exported

    import zipfile

    with zipfile.ZipFile(str(zip_path)) as zf:
        assert ".tcip/store.db" in zf.namelist()

    dest = tmp_path / "restored"
    imported = import_project(str(zip_path), str(dest))
    assert "error" not in imported

    from tcip_mcp.project_record import read_record

    restored = read_record(dest)
    assert (restored["id"], restored["site"]) == (record["id"], "north orchard")


def test_an_archive_of_an_open_database_carries_every_record_and_entry_it_holds(tmp_path: Path):
    """The project's database is open, its latest commits still in its write-ahead log, when the
    archive runs; the imported project's dump holds every record the source's did, byte for
    byte, and every log entry, the import's own line after them."""
    from tcip_mcp.cli.dump_store import dump_store
    from tcip_mcp.tools.meta_tools import report_friction, write_retrospective

    src = tmp_path / "src_project"
    _initialized(src)
    for i in range(3):
        report_friction(src, category="missing_tool", detail=f"report {i}")
    write_retrospective(src, project_id="p", task="t", worked="w", did_not_work="d")
    before_dir = tmp_path / "before"
    before = dump_store(src, before_dir)
    assert (src / ".tcip" / "store.db-wal").stat().st_size > 0

    assert "error" not in archive_project(src, str(tmp_path / "export.zip"))
    dest = tmp_path / "restored"
    assert "error" not in import_project(str(tmp_path / "export.zip"), str(dest))
    after_dir = tmp_path / "after"
    dump_store(dest, after_dir)

    assert len(before) == 6
    for written in before:
        restored = after_dir / written.relative_to(before_dir)
        if written.suffix == ".jsonl":
            assert restored.read_bytes().startswith(written.read_bytes()), written
        else:
            assert restored.read_bytes() == written.read_bytes(), written
