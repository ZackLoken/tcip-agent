"""Tests for project management tools."""

from __future__ import annotations

import os
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
from tcip_store.binding import BACKEND_ENV, DEFAULT_BACKEND, FILE_BACKEND, SQLITE_BACKEND


def _damage_record(key: tcip_store.Key, data: bytes) -> None:
    """Put ``data`` behind a record, wherever the bound backend keeps it.

    A record must already exist at the key; this corrupts the bytes behind it in place, on the
    same path the bound backend actually reads, so the case is genuine on both backends rather
    than reporting absence on one and corruption on the other.
    """
    from tcip_store.store import _backend

    name = os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND
    if name == FILE_BACKEND:
        _backend().path_for(key).write_bytes(data)
        return
    if name != SQLITE_BACKEND:
        raise ValueError(f"no bytes-corruption path for backend {name!r}")
    import sqlite3

    from tcip_store.sqlite_backend import database_path, encode_parts

    conn = sqlite3.connect(str(database_path(str(key.root))), isolation_level=None)
    try:
        conn.execute(
            "update records set value = ? where store = ? and parts = ?",
            (data, key.store, encode_parts(key.parts)),
        )
    finally:
        conn.close()


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


def test_init_project(tmp_path: Path, monkeypatch):
    # tmp_path sits directly under this test's workspace; point the workspace elsewhere so
    # init_project's naming rail (which only holds under the workspace) doesn't apply here.
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    result = init_project(str(tmp_path), site="north orchard")
    assert (tmp_path / ".tcip").is_dir()
    assert (tmp_path / ".tcip" / "artifacts").is_dir()
    assert (tmp_path / ".tcip" / "models").is_dir()
    assert ".tcip/" in result["created"]


def test_inspect_project(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    status = inspect_project(str(tmp_path))
    assert status["initialized"] is False

    init_project(str(tmp_path), site="north orchard")
    status = inspect_project(str(tmp_path))
    assert status["initialized"] is True


def test_inspect_project_folds_in_recent_activity(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    from tcip_mcp.tools.meta_tools import claude_reports

    init_project(str(tmp_path), site="north orchard")
    status = inspect_project(str(tmp_path))
    assert status["recent_activity"] == {}  # no history yet: genuinely empty, not corrupt

    claude_reports(str(tmp_path), category="missing_tool", detail="a")
    status = inspect_project(str(tmp_path))
    assert status["recent_activity"]["reports_since_last_retrospective"] == 1


def test_inspect_project_folds_in_last_retrospective_by_id_not_path(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    from tcip_mcp.tools.meta_tools import project_retrospective

    init_project(str(tmp_path), site="north orchard")
    project_retrospective(
        str(tmp_path), project_id="p", task="t", worked="w", did_not_work="d",
    )

    status = inspect_project(str(tmp_path))
    last = status["recent_activity"]["last_retrospective"]
    assert last["project_id"] == "p"
    assert "path" not in last


def test_inspect_project_surfaces_corrupt_status_honestly(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    from tcip_mcp.project_status import project_status_key, record_report

    init_project(str(tmp_path), site="north orchard")
    record_report(tmp_path)  # seed a real record so a damaged one has somewhere to overwrite
    _damage_record(project_status_key(tmp_path), b"{not valid json")

    status = inspect_project(str(tmp_path))
    assert "status_unavailable" in status["recent_activity"]
    # Live counts must stay unaffected by a corrupt status file: different store, different rail.
    assert status["initialized"] is True


def test_inspect_project_surfaces_version_refused_status_distinctly_from_corrupt(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    from tcip_mcp.project_status import (
        PROJECT_STATUS_STORE, project_status_key, record_report,
    )

    init_project(str(tmp_path), site="north orchard")
    record_report(tmp_path)  # seed a real record so a poisoned one has somewhere to overwrite
    poisoned = tcip_store.get_descriptor(PROJECT_STATUS_STORE).codec.encode(
        {"reports_since_last_retrospective": 1, "schema_version": 99})
    _damage_record(project_status_key(tmp_path), poisoned)

    status = inspect_project(str(tmp_path))
    assert "schema_version" in status["recent_activity"]["status_unavailable"]
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

    # A directory made outside the platform (init_project itself now refuses a non-conforming
    # name under the workspace); set_active_project must still adopt it by its existing name.
    (project_path("proj_a") / ".tcip").mkdir(parents=True)
    claude_reports(str(project_path("proj_a")), category="missing_tool", detail="a")

    result = set_active_project("proj_a")
    assert result["recent_activity"]["reports_since_last_retrospective"] == 1


def test_inspect_project_reports_platform_root_divergence_from_marker(tmp_path: Path, monkeypatch):
    """Adoption repins only the adopting process; a stale process's own root can keep naming a
    different project than the marker until it deliberately adopts too, and inspect_project must
    say so rather than answering as if the two agreed."""
    from tcip_mcp import workspace

    ws = tmp_path / "ws"
    proj = ws / "hazelnut_catkin_valley"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    workspace.set_active_project("hazelnut_catkin_valley")

    stale_root = tmp_path / "stale"
    stale_root.mkdir()
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(stale_root))

    status = inspect_project(str(proj))
    divergence = status["platform_root_diverges_from_marker"]
    assert divergence["marker_project"] == str(proj)
    assert divergence["platform_root"] == str(stale_root)
    assert divergence["action"] == "set_active_project"


def test_inspect_project_reports_no_divergence_when_root_matches_the_marker(
    tmp_path: Path, monkeypatch
):
    from tcip_mcp import workspace

    ws = tmp_path / "ws"
    proj = ws / "hazelnut_catkin_valley"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    workspace.set_active_project("hazelnut_catkin_valley")  # also repins TCIP_PROJECT_ROOT

    status = inspect_project(str(proj))
    assert "platform_root_diverges_from_marker" not in status


def test_inspect_project_reports_the_current_platform_root_binding_after_a_repin(
    tmp_path: Path, monkeypatch
):
    """platform_root_binding is the substitute for a log line no process here emits: it must
    name the just-adopted root, not whatever pin_project_root last decided at process startup."""
    from tcip_mcp import workspace

    ws = tmp_path / "ws"
    proj = ws / "hazelnut_catkin_valley"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))

    workspace.set_active_project("hazelnut_catkin_valley")
    status = inspect_project(str(proj))

    binding = status["platform_root_binding"]
    assert binding["root"] == str(proj)
    assert binding["source"] == "adopted"


def test_inspect_project_reports_marker_problem_for_a_dangling_marker(
    tmp_path: Path, monkeypatch
):
    """A marker naming a project whose ``.tcip`` is gone is not adoptable: the divergence
    report must say so rather than naming ``set_active_project`` as if adopting it would work."""
    import shutil

    from tcip_mcp import workspace

    ws = tmp_path / "ws"
    proj = ws / "chestnut_burr_valley"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    workspace.set_active_project("chestnut_burr_valley")
    shutil.rmtree(proj / ".tcip")

    status = inspect_project(str(tmp_path / "elsewhere"))
    divergence = status["platform_root_diverges_from_marker"]
    assert "marker_problem" in divergence
    assert "chestnut_burr_valley" in divergence["marker_problem"]


def test_inspect_project_against_a_nonexistent_workspace_creates_nothing(
    tmp_path: Path, monkeypatch
):
    ws = tmp_path / "no_such_workspace"
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    proj = tmp_path / "proj"
    proj.mkdir()

    inspect_project(str(proj))

    assert not ws.exists()


def test_inspect_project_reports_the_workspace_store_refusal_for_a_loose_marker(
    tmp_path: Path, monkeypatch
):
    """A workspace holding a loose ``.active`` with no database is what precedes
    ``python scripts/adopt_store.py``; the divergence check must name that refusal rather
    than let it raise out of ``inspect_project``."""
    from tcip_store.sqlite_backend import SqliteBackend

    from tcip_mcp import workspace

    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    tcip_store.bind(SqliteBackend())
    (ws / workspace.ACTIVE_MARKER).write_text("some_project\n")

    proj = tmp_path / "proj"
    proj.mkdir()

    status = inspect_project(str(proj))

    divergence = status["platform_root_diverges_from_marker"]
    assert "marker_problem" in divergence
    assert not (ws / ".tcip").exists()


def test_init_project_refuses_a_non_conforming_name_under_the_workspace(tmp_path: Path, monkeypatch):
    ws = tmp_path / "ws"
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))

    result = init_project(str(ws / "two_segments"), site="north orchard")

    assert "error" in result
    assert not (ws / "two_segments").exists()


def test_init_project_admits_a_conforming_name_under_the_workspace(tmp_path: Path, monkeypatch):
    ws = tmp_path / "ws"
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))

    result = init_project(str(ws / "hazelnut_catkin_elongation"), site="north orchard")

    assert "error" not in result
    assert (ws / "hazelnut_catkin_elongation" / ".tcip").is_dir()


def test_init_project_admits_a_non_conforming_name_outside_the_workspace(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))

    result = init_project(str(tmp_path / "two_segments"), site="north orchard")

    assert "error" not in result
    assert (tmp_path / "two_segments" / ".tcip").is_dir()


def test_import_project_refuses_a_non_conforming_destination_under_the_workspace(
    tmp_path: Path, monkeypatch
):
    ws = tmp_path / "ws"
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    src = tmp_path / "src_project"
    init_project(str(src), site="north orchard")
    zip_path = tmp_path / "export.zip"
    exported = archive_project(str(src), str(zip_path))
    assert "error" not in exported

    dest = ws / "two_segments"
    imported = import_project(str(zip_path), str(dest))

    assert "error" in imported
    assert not dest.exists()


def test_import_project_admits_a_conforming_destination_under_the_workspace(
    tmp_path: Path, monkeypatch
):
    ws = tmp_path / "ws"
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    src = tmp_path / "src_project"
    init_project(str(src), site="north orchard")
    zip_path = tmp_path / "export.zip"
    exported = archive_project(str(src), str(zip_path))
    assert "error" not in exported

    dest = ws / "hazelnut_catkin_elongation"
    imported = import_project(str(zip_path), str(dest))

    assert "error" not in imported
    assert dest.is_dir()


def test_export_import_roundtrip(tmp_path: Path):
    """archive_project -> import_project -> inspect_project recovers the project."""
    from PIL import Image

    src = tmp_path / "src_project"
    date = "2-11-26"
    images = src / "images" / date
    labels = src / "annotations" / date
    for d in (images, labels):
        d.mkdir(parents=True)
    init_project(str(src), site="north orchard")

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import ClassRegistry, Subject

    Image.new("RGB", (64, 64)).save(images / "img_000.jpg")
    json_io.write_annotations(
        str(labels / "img_000.json"),
        [Annotation(subject="catkin", geometry=BBox(10, 10, 30, 30))], 64, 64,
    )
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

    # A restored bundle is files, not a database: a database backend refuses to touch it until
    # its own record/log files are moved in, the same conform step real usage runs.
    from tcip_store.adoption import adopt_root
    from tcip_store.file_backend import database_file
    from tcip_store.layout_claims import ROOT

    dest_abs = str(Path(dest).absolute())
    if not database_file(dest_abs).is_file():
        adopt_root(dest_abs, ROOT, report=lambda line: None)

    status = inspect_project(str(dest))
    assert status["initialized"] is True
    # inspect_project counts raw image files, so the two sibling bands count separately here;
    # the logical-image count the band group folds them into is asserted below.
    assert status["image_count"] == 3
    assert (dest / "annotations" / date / "img_000.json").is_file()
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
    init_project(str(src), site="north orchard")

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


def test_archive_project_reports_checkpoints_excluded_by_default(tmp_path: Path):
    """A checkpoint under .tcip/models/*.pt is dropped by include_models=False; left_behind
    names that count separately from unaccounted and bookkeeping, rather than folding it in."""
    src = tmp_path / "src_project"
    init_project(str(src), site="north orchard")
    (src / ".tcip" / "models" / "m.pt").write_bytes(b"weights")

    result = archive_project(str(src), str(tmp_path / "export.zip"))

    assert "error" not in result
    assert result["left_behind"]["checkpoints_excluded"] == 1
    assert result["left_behind"]["unaccounted"] == 0

    result_included = archive_project(
        str(src), str(tmp_path / "export2.zip"), include_models=True
    )
    assert result_included["left_behind"]["checkpoints_excluded"] == 0


def test_archive_project_admits_a_symlink_spelled_project(tmp_path: Path):
    """A project reached through a symlink must archive rather than raising ValueError out of
    the door: archive_project resolves project_path once and uses that resolved root for both
    member.relative_to and the include_models comparison."""
    real = tmp_path / "real_project"
    _make_dataset(real)
    link = tmp_path / "linked_project"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks not available on this machine: {exc}")

    result = archive_project(str(link), str(tmp_path / "export.zip"))

    assert "error" not in result
    assert result["files_added"] > 0


def test_archive_project_admits_a_relative_spelled_project(tmp_path: Path, monkeypatch):
    """A relative project_path must archive rather than raising ValueError out of the door: the
    resolved root, not the literal relative spelling, is what member paths are relative to."""
    real = tmp_path / "rel_project"
    _make_dataset(real)
    monkeypatch.chdir(tmp_path)

    result = archive_project("rel_project", str(tmp_path / "export.zip"))

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
    from tcip_mcp.tools.project_tools import dataset_registry_key

    project = tmp_path / "proj"
    (project / ".tcip").mkdir(parents=True)

    assert read_datasets(project) == []  # a project with nothing registered yet

    upsert_dataset(project, {"id": "aaa", "path": str(project), "crop": "hazelnut"})
    _damage_record(dataset_registry_key(project), b'[{"id": "aaa"')  # truncated mid-list
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
    committed = b'{"crop": "hazelnut", "id": "committed_id", "fingerprint": "written first"}\n'
    real_uuid4 = uuid_module.uuid4
    raced: list[str] = []

    def racing_uuid4():
        if not raced:
            raced.append("minted")
            (src / "dataset.json").write_bytes(committed)
        return real_uuid4()

    monkeypatch.setattr(uuid_module, "uuid4", racing_uuid4)

    result = register_dataset(str(src), crop="hazelnut")

    assert raced == ["minted"]
    assert "error" not in result
    assert result["id"] == "committed_id"
    assert json.loads((src / "dataset.json").read_text(encoding="utf-8"))["id"] == "committed_id"
    assert [r["id"] for r in read_datasets(src)] == ["committed_id"]


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


def test_scaffolding_twice_leaves_what_the_first_run_created(tmp_path: Path):
    """Scaffolding is idempotent: a second run re-creates the directories and touches nothing."""
    from tcip_mcp.tools.project_tools import _scaffold_project

    _scaffold_project(str(tmp_path), "north orchard")
    (tmp_path / ".tcip" / "artifacts" / "kept.txt").write_text("kept", encoding="utf-8")

    _scaffold_project(str(tmp_path), "north orchard")

    assert (tmp_path / ".tcip" / "artifacts").is_dir()
    assert (tmp_path / ".tcip" / "models").is_dir()
    assert (tmp_path / ".tcip" / "artifacts" / "kept.txt").read_text(encoding="utf-8") == "kept"


# ── the project record's authored site ────────────────────────────────────────


def test_init_project_records_the_site(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    result = init_project(str(tmp_path), site="north orchard")

    assert result["site"] == "north orchard"
    from tcip_mcp.project_record import read_record

    assert read_record(str(tmp_path)) == {"site": "north orchard"}


def test_init_project_run_twice_with_the_same_site_is_idempotent(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    init_project(str(tmp_path), site="north orchard")

    result = init_project(str(tmp_path), site="north orchard")

    assert "error" not in result
    assert result["site"] == "north orchard"


def test_init_project_refuses_a_different_site_than_the_one_already_recorded(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    init_project(str(tmp_path), site="north orchard")

    result = init_project(str(tmp_path), site="south orchard")

    assert "error" in result
    assert "north orchard" in result["error"]
    assert "south orchard" in result["error"]
    from tcip_mcp.project_record import read_record

    assert read_record(str(tmp_path))["site"] == "north orchard"


def test_init_project_refuses_an_empty_site(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))

    result = init_project(str(tmp_path), site="   ")

    assert "error" in result
    from tcip_mcp.project_record import site_fields

    assert site_fields(str(tmp_path))["site"] is None


def test_init_project_refuses_an_empty_site_leaving_nothing_on_disk(tmp_path: Path, monkeypatch):
    """A refused site is validated before anything is created, the same as the name-scheme
    refusal: the destination is left exactly as it was, not half-scaffolded."""
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    dest = tmp_path / "fresh_project"

    result = init_project(str(dest), site="   ")

    assert "error" in result
    assert not dest.exists()


def test_init_project_scaffolds_a_relative_path_where_it_resolves(tmp_path: Path, monkeypatch):
    """A relative project_path scaffolds and records at the same absolute location the
    workspace-name check itself resolved, rather than the record write refusing a relative
    root after ``.tcip`` already exists."""
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    monkeypatch.chdir(tmp_path)

    result = init_project("relative_proj", site="north orchard")

    assert "error" not in result
    assert (tmp_path / "relative_proj" / ".tcip").is_dir()
    from tcip_mcp.project_record import read_record

    assert read_record(str(tmp_path / "relative_proj")) == {"site": "north orchard"}


def test_init_project_refuses_a_present_but_invalid_record(tmp_path: Path, monkeypatch):
    """The door surfaces the reader's own refusal rather than the store's raw exception."""
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    from tcip_mcp.project_record import project_record_key

    key = project_record_key(str(tmp_path))
    tcip_store.replace(key, {"not_site": "whatever"}, expect=tcip_store.Version.ABSENT)

    result = init_project(str(tmp_path), site="north orchard")

    assert "error" in result
    assert "does not hold a site" in result["error"]


def test_init_project_refuses_an_undecodable_record(tmp_path: Path, monkeypatch):
    """The store's own DecodeError is a StoreError, caught and returned as the door's error."""
    from tcip_mcp.project_record import project_record_key
    from tests._record_damage_fixtures import damage_record

    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    key = project_record_key(str(tmp_path))
    tcip_store.replace(key, {"site": "north orchard"}, expect=tcip_store.Version.ABSENT)
    damage_record(key, b"{not valid json")

    result = init_project(str(tmp_path), site="north orchard")

    assert "error" in result
    assert "does not decode" in result["error"]


def test_init_project_refuses_an_unadopted_root(tmp_path: Path, monkeypatch):
    """A root whose records are still loose files: the store's conform rail refuses
    init_project's site write there until scripts/adopt_store.py has run, the same rule every
    other record store under that root already obeys. The file backend legitimately produces
    that state (import_project no longer does: it adopts a fresh root under the database
    backend), so the unadopted root here is built by writing through the file backend directly
    and then judged under the database backend."""
    from tcip_store.file_backend import FileBackend
    from tcip_store.sqlite_backend import SqliteBackend
    from tcip_store.store import _backend

    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    dest = tmp_path / "unadopted"
    previous = _backend()
    # init_project's own audit entry lands at the platform root, not dest; a throwaway root here
    # keeps it off tmp_path, which stage two's own audit write below needs to find pristine.
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path / "scratch_platform_root"))
    file_backend = FileBackend()
    tcip_store.bind(file_backend)
    try:
        init_project(str(dest), site="north orchard")
    finally:
        tcip_store.bind(previous)
        file_backend.close()

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    backend = SqliteBackend()
    tcip_store.bind(backend)
    try:
        result = init_project(str(dest), site="north orchard")
    finally:
        tcip_store.bind(previous)
        backend.close()

    assert "error" in result
    assert "scripts/adopt_store.py" in result["error"]


def test_init_project_records_the_site_on_a_directory_that_gained_tcip_with_no_creating_door(
    tmp_path: Path, monkeypatch
):
    """The reachable state a store write with no door leaves (``claude_reports`` on a bare
    directory): ``init_project`` on it afterward records the site the same way it would on a
    truly fresh directory, since the writer's create-only write does not distinguish the two."""
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    from tcip_mcp.tools.meta_tools import claude_reports

    claude_reports(str(tmp_path), category="missing_tool", detail="a")
    assert (tmp_path / ".tcip").is_dir()

    result = init_project(str(tmp_path), site="north orchard")

    assert "error" not in result
    assert result["site"] == "north orchard"


def test_inspect_project_reports_site_fields_across_project_states(tmp_path: Path, monkeypatch):
    """A path with no ``.tcip`` carries neither key; a project with a record carries the site; a
    project with ``.tcip`` and no record carries the absent-record problem text."""
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    from tcip_mcp.tools.meta_tools import claude_reports

    bare = tmp_path / "no_tcip"
    bare.mkdir()
    status = inspect_project(str(bare))
    assert "site" not in status
    assert "site_problem" not in status

    recordless = tmp_path / "recordless"
    recordless.mkdir()
    claude_reports(str(recordless), category="missing_tool", detail="a")
    status = inspect_project(str(recordless))
    assert status["site"] is None
    assert "init_project" in status["site_problem"]

    recorded = tmp_path / "recorded"
    recorded.mkdir()
    init_project(str(recorded), site="north orchard")
    status = inspect_project(str(recorded))
    assert status["site"] == "north orchard"
    assert status["site_problem"] is None


def test_inspect_project_reports_an_invalid_record(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    from tcip_mcp.project_record import project_record_key

    init_project(str(tmp_path), site="north orchard")
    key = project_record_key(str(tmp_path))
    current = tcip_store.read_versioned(key).version
    tcip_store.replace(key, {"not_site": "x"}, expect=current)

    status = inspect_project(str(tmp_path))

    assert status["site"] is None
    assert "does not hold a site" in status["site_problem"]


def test_archive_and_import_carry_the_project_record(tmp_path: Path, monkeypatch):
    """init_project -> archive_project -> import_project round-trips a project whose record is
    on disk in the archive: the record travels with the project like every other ``.tcip``
    document, and archive_project exports it itself, so no operator step sits between the two
    doors."""
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    src = tmp_path / "src_project"
    init_project(str(src), site="north orchard")

    zip_path = tmp_path / "export.zip"
    exported = archive_project(str(src), str(zip_path))
    assert "error" not in exported

    import zipfile

    with zipfile.ZipFile(str(zip_path)) as zf:
        assert ".tcip/project.json" in zf.namelist()

    dest = tmp_path / "restored"
    imported = import_project(str(zip_path), str(dest))
    assert "error" not in imported

    from tcip_store.adoption import adopt_root
    from tcip_store.file_backend import database_file
    from tcip_store.layout_claims import ROOT

    dest_abs = str(Path(dest).absolute())
    if not database_file(dest_abs).is_file():
        adopt_root(dest_abs, ROOT, report=lambda line: None)

    from tcip_mcp.project_record import read_record

    assert read_record(str(dest))["site"] == "north orchard"
