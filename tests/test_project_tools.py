"""Tests for project management tools."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

import tcip_store
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
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)
    class_registry.write_registry(
        root / "classes.json", ClassRegistry(subjects=(Subject(name="bud"),)))


def test_register_dataset_writes_identity_and_registers(tmp_path: Path):
    import json

    src = tmp_path / "proj"
    _make_dataset(src)

    res = register_dataset(str(src), crop="currant")
    assert "error" not in res
    assert res["crop"] == "currant" and res["id"] and res["fingerprint"]

    # dataset.json holds {crop, id, fingerprint}.
    ident = json.loads((src / "dataset.json").read_text())
    assert ident == {"crop": "currant", "id": res["id"], "fingerprint": res["fingerprint"]}
    # the project registry knows the dataset.
    regs = read_datasets(src)
    assert len(regs) == 1 and regs[0]["id"] == res["id"] and regs[0]["crop"] == "currant"


def test_register_dataset_requires_crop_and_keeps_id_stable(tmp_path: Path):
    src = tmp_path / "proj"
    _make_dataset(src)

    assert "error" in register_dataset(str(src), crop="")  # crop is the expert's fact, required

    first = register_dataset(str(src), crop="currant")
    again = register_dataset(str(src), crop="currant")
    assert again["id"] == first["id"]  # id minted once, preserved across re-runs
    assert len(read_datasets(src)) == 1  # not duplicated in the registry


def test_register_dataset_reconciles_a_move_by_id(tmp_path: Path):
    import shutil

    src = tmp_path / "orig"
    _make_dataset(src)
    reg = register_dataset(str(src), crop="currant")

    moved = tmp_path / "moved"
    shutil.copytree(src, moved)  # same content, new path
    register_dataset(str(moved), crop="currant", project_root=str(src))

    regs = read_datasets(src)
    same = [r for r in regs if r["id"] == reg["id"]]
    assert len(same) == 1  # one entry for the id: the move updated the path, not duplicated
    assert same[0]["path"] == str(moved)
    assert same[0]["fingerprint"] == reg["fingerprint"]  # unchanged content -> same fingerprint


def test_initialize_project(tmp_path: Path, monkeypatch):
    # tmp_path sits directly under this test's workspace; point the workspace elsewhere so
    # initialize_project's naming rail (which only holds under the workspace) doesn't apply here.
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    result = initialize_project(str(tmp_path), site="north orchard")
    assert (tmp_path / ".tcip").is_dir()
    assert (tmp_path / ".tcip" / "artifacts").is_dir()
    assert (tmp_path / ".tcip" / "models").is_dir()
    assert ".tcip/" in result["created"]


def test_inspect_project(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    status = inspect_project(str(tmp_path))
    assert status["initialized"] is False

    initialize_project(str(tmp_path), site="north orchard")
    status = inspect_project(str(tmp_path))
    assert status["initialized"] is True


def test_inspect_project_folds_in_recent_activity(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    from tcip_mcp.tools.meta_tools import report_friction

    initialize_project(str(tmp_path), site="north orchard")
    status = inspect_project(str(tmp_path))
    assert status["recent_activity"] == {}  # no history yet: genuinely empty, not corrupt

    report_friction(str(tmp_path), category="missing_tool", detail="a")
    status = inspect_project(str(tmp_path))
    assert status["recent_activity"]["reports_since_last_retrospective"] == 1


def test_inspect_project_folds_in_last_retrospective_by_id_not_path(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    from tcip_mcp.tools.meta_tools import write_retrospective

    initialize_project(str(tmp_path), site="north orchard")
    write_retrospective(
        str(tmp_path), project_id="p", task="t", worked="w", did_not_work="d",
    )

    status = inspect_project(str(tmp_path))
    last = status["recent_activity"]["last_retrospective"]
    assert last["project_id"] == "p"
    assert "path" not in last


def test_inspect_project_surfaces_corrupt_status_honestly(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    from tcip_mcp.project_status import project_status_key, record_report

    initialize_project(str(tmp_path), site="north orchard")
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

    initialize_project(str(tmp_path), site="north orchard")
    record_report(tmp_path)  # seed a real record so a poisoned one has somewhere to overwrite
    poisoned = tcip_store.get_descriptor(PROJECT_STATUS_STORE).codec.encode(
        {"reports_since_last_retrospective": 1, "schema_version": 99})
    _damage_record(project_status_key(tmp_path), poisoned)

    status = inspect_project(str(tmp_path))
    assert "schema_version" in status["recent_activity"]["status_unavailable"]
    assert status["initialized"] is True


def test_activate_project_folds_in_recent_activity(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path))
    import tcip_mcp.web_client as web_client
    from tcip_mcp.tools.meta_tools import report_friction
    from tcip_mcp.tools.project_tools import activate_project
    from tcip_mcp.workspace import project_path

    # Stub the GUI notification so the result is deterministic regardless of whether a tcip-web
    # backend happens to be listening on this machine (matches test_activate_project.py).
    monkeypatch.setattr(web_client, "post_panel_event", lambda *a, **k: {"delivered": False})

    # A directory made outside the platform (initialize_project itself now refuses a non-conforming
    # name under the workspace); activate_project must still adopt it by its existing name.
    (project_path("proj_a") / ".tcip").mkdir(parents=True)
    report_friction(str(project_path("proj_a")), category="missing_tool", detail="a")

    result = activate_project("proj_a")
    assert result["recent_activity"]["reports_since_last_retrospective"] == 1


def test_inspect_project_reports_platform_root_divergence_from_marker(tmp_path: Path, monkeypatch):
    """Adoption repins only the adopting process; a stale process's own root can keep naming a
    different project than the marker until it deliberately adopts too, and inspect_project must
    say so rather than answering as if the two agreed."""
    from tcip_mcp import workspace

    ws = tmp_path / "ws"
    proj = ws / "currant_bud_valley"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    workspace.activate_project("currant_bud_valley")

    stale_root = tmp_path / "stale"
    stale_root.mkdir()
    monkeypatch.setenv("TCIP_STATE_ROOT", str(stale_root))

    status = inspect_project(str(proj))
    divergence = status["platform_root_diverges_from_marker"]
    assert divergence["marker_project"] == str(proj)
    assert divergence["platform_root"] == str(stale_root)
    assert divergence["action"] == "activate_project"


def test_inspect_project_reports_no_divergence_when_root_matches_the_marker(
    tmp_path: Path, monkeypatch
):
    from tcip_mcp import workspace

    ws = tmp_path / "ws"
    proj = ws / "currant_bud_valley"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    workspace.activate_project("currant_bud_valley")  # also repins TCIP_STATE_ROOT

    status = inspect_project(str(proj))
    assert "platform_root_diverges_from_marker" not in status


def test_inspect_project_reports_the_current_platform_root_binding_after_a_repin(
    tmp_path: Path, monkeypatch
):
    """platform_root_binding is the substitute for a log line no process here emits: it must
    name the just-adopted root, not whatever pin_platform_root last decided at process startup."""
    from tcip_mcp import workspace

    ws = tmp_path / "ws"
    proj = ws / "currant_bud_valley"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))

    workspace.activate_project("currant_bud_valley")
    status = inspect_project(str(proj))

    binding = status["platform_root_binding"]
    assert binding["root"] == str(proj)
    assert binding["source"] == "adopted"


def test_inspect_project_reports_marker_problem_for_a_dangling_marker(
    tmp_path: Path, monkeypatch
):
    """A marker naming a project whose ``.tcip`` is gone is not adoptable: the divergence
    report must say so rather than naming ``activate_project`` as if adopting it would work."""
    import shutil

    from tcip_mcp import workspace

    ws = tmp_path / "ws"
    proj = ws / "chestnut_burr_valley"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    workspace.activate_project("chestnut_burr_valley")
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
    ``tcip adopt-store``; the divergence check must name that refusal rather
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


def test_initialize_project_refuses_a_non_conforming_name_under_the_workspace(tmp_path: Path, monkeypatch):
    ws = tmp_path / "ws"
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))

    result = initialize_project(str(ws / "two_segments"), site="north orchard")

    assert "error" in result
    assert not (ws / "two_segments").exists()


def test_initialize_project_admits_a_conforming_name_under_the_workspace(tmp_path: Path, monkeypatch):
    ws = tmp_path / "ws"
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))

    result = initialize_project(str(ws / "currant_bud_opening"), site="north orchard")

    assert "error" not in result
    assert (ws / "currant_bud_opening" / ".tcip").is_dir()


def test_initialize_project_admits_a_non_conforming_name_outside_the_workspace(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))

    result = initialize_project(str(tmp_path / "two_segments"), site="north orchard")

    assert "error" not in result
    assert (tmp_path / "two_segments" / ".tcip").is_dir()


def test_import_project_refuses_a_non_conforming_destination_under_the_workspace(
    tmp_path: Path, monkeypatch
):
    ws = tmp_path / "ws"
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    src = tmp_path / "src_project"
    initialize_project(str(src), site="north orchard")
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
    initialize_project(str(src), site="north orchard")
    zip_path = tmp_path / "export.zip"
    exported = archive_project(str(src), str(zip_path))
    assert "error" not in exported

    dest = ws / "currant_bud_opening"
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
    initialize_project(str(src), site="north orchard")

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import ClassRegistry, Subject

    Image.new("RGB", (64, 64)).save(images / "img_000.jpg")
    json_io.write_annotations(
        str(labels / "img_000.json"),
        [Annotation(subject="bud", geometry=BBox(10, 10, 30, 30))], 64, 64,
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
        ClassRegistry(subjects=(Subject(name="bud", description="a currant bud"),)),
    )
    reg = register_dataset(str(src), crop="currant")  # dataset.json identity travels with the data

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
    assert [s.name for s in restored.subjects] == ["bud"]
    # dataset.json travelled with the data: identity (id/crop/fingerprint) survives the round-trip.
    import json

    restored_id = json.loads((dest / "dataset.json").read_text())
    assert restored_id == {"crop": "currant", "id": reg["id"], "fingerprint": reg["fingerprint"]}


def _bare_the_registry_entry(project_root: Path, entry: dict) -> str:
    """Overwrite a real registration's project-registry entry with its bare-hex fingerprint,
    standing in for a dataset registered under a project before the formula-version prefix
    existed: the project-registry counterpart to
    test_dataset_identity_fingerprint_formula.py's own _bare_the_identity, which does this to
    the dataset's own identity document instead."""
    bare = entry["fingerprint"].split(":", 1)[1]
    upsert_dataset(project_root, {**entry, "fingerprint": bare})
    return bare


def test_store_bootstrap_project_roots_admits_a_bare_fingerprint_registry_entry(tmp_path: Path):
    """project_roots is the path adopt_store.py/export_store.py use to reach a project; it must
    not itself be blocked by the identity problem register_dataset re-registration exists to
    fix, so it reads locations through read_datasets_raw rather than read_datasets."""
    from tcip_store.layout_claims import ROOT

    from tcip_mcp.store_catalogue import project_roots

    project = tmp_path / "project"
    dataset = tmp_path / "dataset"
    project.mkdir()
    dataset.mkdir()
    _make_dataset(dataset)
    register_dataset(str(dataset), crop="currant", project_root=str(project))
    _bare_the_registry_entry(project, read_datasets(project)[0])

    with pytest.raises(ValueError, match="register_dataset"):
        read_datasets(project)

    roots = project_roots(project)
    assert (str(dataset.resolve()), ROOT) in roots


def test_project_roots_names_a_run_output_dir_a_split_manifest_and_a_prediction_bucket(
    tmp_path: Path, monkeypatch,
):
    """project_roots reaches every layout a project's own records name it under, not only the
    registered dataset roots: an experiment's own recorded run output directory, the split
    manifest a run bound to (its split.json's manifest_binding.manifest_dir), and a prediction
    bucket under a registered dataset's own predictions/ tree."""
    from types import SimpleNamespace

    from tcip_store.layout_claims import PREDICTION_BUCKET, RUN, SPLITS

    from tcip_mcp.store_catalogue import project_roots
    from tcip_mcp import experiments
    from tcip_mcp.dataset_layout import prediction_dir
    from tcip_mcp.pipelines.data.split_construction import persist_split_manifest

    project = tmp_path / "project"
    dataset = tmp_path / "dataset"
    project.mkdir()
    dataset.mkdir()
    _make_dataset(dataset)
    register_dataset(str(dataset), crop="currant", project_root=str(project))

    # The producers below resolve every member key against the pinned platform root.
    monkeypatch.setenv("TCIP_STATE_ROOT", str(project))

    experiments.create_experiment("exp-1", {"model_source": {}})
    run_dir = tmp_path / "runs" / "exp-1"
    run_dir.mkdir(parents=True)
    experiments.stamp_run_identity("exp-1", "exp-1", str(run_dir))

    split_dir = tmp_path / "splits" / "frozen-exp-1"
    split_dir.mkdir(parents=True)
    persist_split_manifest(
        "exp-1", SimpleNamespace(stems=["a"]), SimpleNamespace(stems=["b"]),
        {"labels_dir": "", "split": {"manifest_binding": {"manifest_dir": str(split_dir)}}},
    )

    bucket = prediction_dir(dataset, "modelA", "2-11-26")
    bucket.mkdir(parents=True)

    roots = project_roots(project)

    assert (str(run_dir.absolute()), RUN) in roots
    assert (str(split_dir.absolute()), SPLITS) in roots
    assert (str(bucket.absolute()), PREDICTION_BUCKET) in roots


def test_project_roots_names_the_hpo_root_and_its_sweeps(tmp_path: Path):
    """project_roots names the project's own HPO root, the fixed convention
    training_tools.hpo_root resolves to, and every sweep directory found under it, the same
    directory training_tools.sweep_dir names a study's own sweep at."""
    from tcip_store.layout_claims import HPO_ROOT, SWEEP

    from tcip_mcp.store_catalogue import project_roots
    from tcip_mcp.tools import training_tools

    project = tmp_path / "project"
    project.mkdir()

    hpo_dir = training_tools.hpo_root(root=project)
    sweep = training_tools.sweep_dir("study-1", root=project)
    sweep.mkdir(parents=True)

    roots = project_roots(project)

    assert (str(hpo_dir), HPO_ROOT) in roots
    assert (str(sweep), SWEEP) in roots


def test_project_roots_names_a_curated_artifact_and_a_lineage_prediction_bucket(
    tmp_path: Path, monkeypatch,
):
    """project_roots names a curated-dataset artifact recorded through record_artifact, the way
    feedback_tools.py's materialize_review_dataset records one, and a prediction bucket an
    inference run recorded through update_lineage, for a bucket written outside any registered
    dataset's own tree."""
    from tcip_store.layout_claims import CURATED, PREDICTION_BUCKET

    from tcip_mcp.store_catalogue import project_roots
    from tcip_mcp import experiments

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("TCIP_STATE_ROOT", str(project))

    experiments.create_experiment("exp-artifacts", {"model_source": {}})

    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    experiments.record_artifact("exp-artifacts", "curated_dataset", str(curated_dir))

    lineage_bucket = tmp_path / "external_predictions" / "run1"
    lineage_bucket.mkdir(parents=True)
    experiments.update_lineage("exp-artifacts", predictions=str(lineage_bucket))

    roots = project_roots(project)

    assert (str(curated_dir.absolute()), CURATED) in roots
    assert (str(lineage_bucket.absolute()), PREDICTION_BUCKET) in roots


def test_project_roots_keeps_both_layouts_when_one_directory_is_two_kinds_of_root(
    tmp_path: Path, monkeypatch,
):
    """A directory recorded as a curated-dataset artifact that is also registered as a project
    dataset keeps both layouts: _add is keyed on the (path, layout) pair, not the path alone, so
    the dataset-registry add is not silently dropped because the artifact add already claimed
    that path."""
    from tcip_store.layout_claims import CURATED, ROOT

    from tcip_mcp.store_catalogue import project_roots
    from tcip_mcp import experiments

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("TCIP_STATE_ROOT", str(project))

    experiments.create_experiment("exp-shared", {"model_source": {}})

    shared = tmp_path / "shared"
    shared.mkdir()
    _make_dataset(shared)
    experiments.record_artifact("exp-shared", "curated_dataset", str(shared))
    register_dataset(str(shared), crop="currant", project_root=str(project))

    roots = project_roots(project)

    assert (str(shared.absolute()), CURATED) in roots
    assert (str(shared.absolute()), ROOT) in roots


def test_project_roots_skips_a_recorded_run_output_dir_that_no_longer_exists(
    tmp_path: Path, monkeypatch,
):
    """A run's own status.json can still name an output directory that has since been moved or
    deleted; project_roots skips it rather than handing adopt_store.py a path to recreate from
    nothing."""
    from tcip_store.layout_claims import RUN

    from tcip_mcp.store_catalogue import project_roots
    from tcip_mcp import experiments

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("TCIP_STATE_ROOT", str(project))

    experiments.create_experiment("exp-stale", {"model_source": {}})
    run_dir = tmp_path / "runs" / "exp-stale"
    run_dir.mkdir(parents=True)
    experiments.stamp_run_identity("exp-stale", "exp-stale", str(run_dir))
    run_dir.rmdir()

    roots = project_roots(project)

    assert not any(layout == RUN for _, layout in roots)


def test_external_dataset_paths_admits_a_bare_fingerprint_registry_entry(tmp_path: Path):
    """import_project calls this after extraction to disclose which registered datasets stayed
    external; a bare pre-prefix fingerprint left over from before the restamp family existed
    must not make the door raise after the extraction it is reporting on has already run, so it
    reads through read_datasets_raw rather than read_datasets."""
    project = tmp_path / "project"
    dataset = tmp_path / "dataset"  # a sibling of project, never nested under it: external
    project.mkdir()
    dataset.mkdir()
    _make_dataset(dataset)
    register_dataset(str(dataset), crop="currant", project_root=str(project))
    entry = read_datasets(project)[0]
    assert entry["path"] == str(dataset.resolve())  # external entries store absolute
    _bare_the_registry_entry(project, entry)

    with pytest.raises(ValueError, match="register_dataset"):
        read_datasets(project)

    assert _external_dataset_paths(project) == [str(dataset.resolve())]


def test_archive_project_includes_bespoke_model_source(tmp_path: Path):
    """A bespoke run's snapshotted .py source (model_src/, written by snapshot_model_source) must
    travel with the archive, or a published/archived project bundles the provenance manifest
    without the code it describes and can't rerun its own pipeline from the archive alone."""
    src = tmp_path / "src_project"
    initialize_project(str(src), site="north orchard")

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
    initialize_project(str(src), site="north orchard")
    (src / ".tcip" / "models" / "m.pt").write_bytes(b"weights")

    result = archive_project(str(src), str(tmp_path / "export.zip"))

    assert "error" not in result
    assert result["left_behind"]["checkpoints_excluded"] == 1
    assert result["left_behind"]["unaccounted"] == 0

    result_included = archive_project(
        str(src), str(tmp_path / "export2.zip"), include_models=True
    )
    assert result_included["left_behind"]["checkpoints_excluded"] == 0


def test_archive_project_includes_a_registered_run_checkpoint_outside_tcip_models(
    tmp_path: Path, monkeypatch,
):
    """A checkpoint registered through ``register_model_from_experiment`` sits wherever
    ``launch_training`` actually wrote it, ``.tcip/experiments/<experiment_id>/model_final.pt``
    under the platform's own default ``output_dir``, not under ``.tcip/models/``.
    ``include_models=True`` must bundle it there too, or a breeder who trusts the flag gets an
    archive with no model in it at all."""
    from tcip_mcp.experiments import (
        complete_run,
        create_experiment,
        experiment_dir,
        register_model_from_experiment,
        update_status,
    )

    src = tmp_path / "src_project"
    initialize_project(str(src), site="north orchard")
    monkeypatch.setenv("TCIP_STATE_ROOT", str(src))

    exp_id = "exp_ckpt_bundle"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")
    ckpt_dir = experiment_dir(exp_id)
    # The file backend materializes the experiment directory with the record; the database
    # backend does not, so tolerate either.
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights = ckpt_dir / "model_final.pt"
    weights.write_bytes(b"the real weights this run produced")
    completed = complete_run(exp_id, str(weights))
    assert "error" not in completed, completed
    registered = register_model_from_experiment(exp_id, str(weights), project_path=str(src))
    assert "error" not in registered, registered

    result = archive_project(str(src), str(tmp_path / "export.zip"), include_models=True)
    assert "error" not in result

    import zipfile

    with zipfile.ZipFile(str(tmp_path / "export.zip")) as zf:
        names = zf.namelist()
    assert any(n.endswith("model_final.pt") for n in names), (
        f"a registered run checkpoint outside .tcip/models/ is missing from the archive: {names}"
    )


def test_import_project_admits_a_bundle_holding_a_registered_run_checkpoint(
    tmp_path: Path, monkeypatch,
):
    """archive_project(include_models=True) bundles a run's registered checkpoint from wherever
    it actually sits; the staged tree's own registry still names the exporting project's absolute
    path, which does not resolve under staging, so import_project must still recognize and admit
    the checkpoint by its own shape rather than refusing its sibling door's own archive."""
    from tcip_mcp.experiments import (
        complete_run, create_experiment, experiment_dir, register_model_from_experiment,
        update_status,
    )

    src = tmp_path / "src_project"
    initialize_project(str(src), site="north orchard")
    monkeypatch.setenv("TCIP_STATE_ROOT", str(src))

    exp_id = "exp_roundtrip"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")
    ckpt_dir = experiment_dir(exp_id)
    # The file backend materializes the experiment directory; the database backend does not.
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights = ckpt_dir / "model_final.pt"
    weights.write_bytes(b"the real weights this run produced")
    completed = complete_run(exp_id, str(weights))
    assert "error" not in completed, completed
    registered = register_model_from_experiment(exp_id, str(weights), project_path=str(src))
    assert "error" not in registered, registered

    zip_path = tmp_path / "export.zip"
    exported = archive_project(str(src), str(zip_path), include_models=True)
    assert "error" not in exported, exported

    dest = tmp_path / "restored"
    imported = import_project(str(zip_path), str(dest))

    assert "error" not in imported, imported
    assert (dest / ".tcip" / "experiments" / exp_id / "model_final.pt").is_file()


def test_import_project_admits_a_registered_checkpoint_with_no_disclosure(
    tmp_path: Path, monkeypatch,
):
    """A run's own checkpoint, registered under the project's own tree, comes back from an
    archive/import round trip with nothing to disclose: the writer already spelled it relative
    to the registry's scope root, so the moved tree's registry still resolves under it, unlike
    the pre-family behavior this test used to pin (every checkpoint_path was stored absolute, so
    the imported registry always named the exporting root's stale path). The stored entry itself
    stays relative; the resolved response is absolute; weights load by digest either way, since
    loading never reads the stored path."""
    from tcip_mcp.experiments import (
        complete_run, create_experiment, experiment_dir, register_model_from_experiment,
        update_status,
    )
    from tcip_mcp.model_registry import ModelRegistry, read_registry_index, registry_index_key

    src = tmp_path / "src_project"
    initialize_project(str(src), site="north orchard")
    monkeypatch.setenv("TCIP_STATE_ROOT", str(src))

    exp_id = "exp_disclosure"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")
    ckpt_dir = experiment_dir(exp_id)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights = ckpt_dir / "model_final.pt"
    weights.write_bytes(b"the real weights this run produced")
    completed = complete_run(exp_id, str(weights))
    assert "error" not in completed, completed
    registered = register_model_from_experiment(exp_id, str(weights), project_path=str(src))
    assert "error" not in registered, registered
    assert Path(registered["checkpoint"]).is_absolute()

    zip_path = tmp_path / "export.zip"
    exported = archive_project(str(src), str(zip_path), include_models=True)
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

    resolved = ModelRegistry(str(dest)).get_model(entries[0]["name"])["checkpoint_path"]
    assert Path(resolved).is_absolute()
    assert Path(resolved).is_file()


def test_import_project_keeps_a_relative_entry_relative_when_the_archive_carries_no_checkpoint(
    tmp_path: Path, monkeypatch,
):
    """A relative registry entry whose weights the archive legitimately dropped
    (``include_models=False``) must come back still relative and disclosed as unresolved: the
    staging conform's no-match fallback used to write the entry's own staging directory's
    absolute path over it, misfiling an internal-but-absent entry as designed-external and
    leaving a path into a directory the door was about to delete permanently in the registry."""
    from tcip_mcp.experiments import (
        complete_run, create_experiment, experiment_dir, register_model_from_experiment,
        update_status,
    )
    from tcip_mcp.model_registry import read_registry_index, registry_index_key

    src = tmp_path / "src_project"
    initialize_project(str(src), site="north orchard")
    monkeypatch.setenv("TCIP_STATE_ROOT", str(src))

    exp_id = "exp_no_checkpoint"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")
    ckpt_dir = experiment_dir(exp_id)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights = ckpt_dir / "model_final.pt"
    weights.write_bytes(b"weights the archive will legitimately drop")
    completed = complete_run(exp_id, str(weights))
    assert "error" not in completed, completed
    registered = register_model_from_experiment(exp_id, str(weights), project_path=str(src))
    assert "error" not in registered, registered

    stored_before = read_registry_index(src)[0]["checkpoint_path"]
    assert not Path(stored_before).is_absolute(), stored_before

    zip_path = tmp_path / "export.zip"
    exported = archive_project(str(src), str(zip_path), include_models=False)
    assert "error" not in exported, exported

    dest = tmp_path / "restored"
    imported = import_project(str(zip_path), str(dest))

    assert "error" not in imported, imported
    stored_after = tcip_store.read(registry_index_key(dest))["entries"][0]["checkpoint_path"]
    assert stored_after == stored_before
    assert imported["checkpoint_paths_unresolved"] == [stored_before]
    assert imported["external_checkpoints"] == []


def test_import_project_conforms_a_genuinely_unconformed_registry_the_archive_carries(
    tmp_path: Path,
):
    """An archive made before this family existed carries a bare version-1 registry array; the
    import door's own on-disk conform (never exercised by a test whose registry the writer
    already spelled version-2 relative) must wrap and respell it so the weights load at the new
    location, not merely leave an already-conformed registry untouched."""
    import zipfile

    from tcip_mcp.model_registry import ModelRegistry, read_registry_index, registry_index_key

    src = tmp_path / "src_project"
    initialize_project(str(src), site="north orchard")
    ckpt_dir = src / ".tcip" / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    content = b"weights an archive made before the family carried"
    ckpt = ckpt_dir / "m.pt"
    ckpt.write_bytes(content)
    ModelRegistry(str(src)).register_model("m", str(ckpt), {}, metrics_source=None)

    zip_path = tmp_path / "export.zip"
    exported = archive_project(str(src), str(zip_path), include_models=True)
    assert "error" not in exported, exported

    v1_entries = tcip_store.read(registry_index_key(src))["entries"]
    downgraded = tmp_path / "downgraded.zip"
    with zipfile.ZipFile(zip_path) as src_zip, zipfile.ZipFile(downgraded, "w") as dst_zip:
        for item in src_zip.infolist():
            data = src_zip.read(item.filename)
            if item.filename.endswith(".tcip/models/registry.json"):
                data = tcip_store.RECORD_JSON.encode(v1_entries)
            dst_zip.writestr(item, data)

    dest = tmp_path / "restored"
    imported = import_project(str(downgraded), str(dest))

    assert "error" not in imported, imported
    assert imported["checkpoint_paths_unresolved"] == []
    assert imported["external_checkpoints"] == []
    entries = read_registry_index(dest)
    assert entries[0]["checkpoint_path"] == ".tcip/models/m.pt"
    resolved = ModelRegistry(str(dest)).get_model("m")["checkpoint_path"]
    assert Path(resolved).is_file()


def test_import_project_conforms_a_stray_schema_version_two_registry_the_archive_carries(
    tmp_path: Path,
):
    """An archive whose registry.json is ``{"schema_version": 2, "entries": [...]}``, the
    version-1 reset's own dev-era shape (planted by rewriting the bundled index's bytes, the
    reset's own raw-bytes technique), must still conform through the import door's own on-disk
    conform, and land with the field dropped rather than refusing the whole import."""
    import zipfile

    from tcip_mcp.model_registry import ModelRegistry, read_registry_index, registry_index_key

    src = tmp_path / "src_project"
    initialize_project(str(src), site="north orchard")
    ckpt_dir = src / ".tcip" / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    content = b"weights a dev-era writer stamped schema_version 2 onto"
    ckpt = ckpt_dir / "m.pt"
    ckpt.write_bytes(content)
    ModelRegistry(str(src)).register_model("m", str(ckpt), {}, metrics_source=None)

    zip_path = tmp_path / "export.zip"
    exported = archive_project(str(src), str(zip_path), include_models=True)
    assert "error" not in exported, exported

    entries = tcip_store.read(registry_index_key(src))["entries"]
    poisoned = tmp_path / "poisoned.zip"
    with zipfile.ZipFile(zip_path) as src_zip, zipfile.ZipFile(poisoned, "w") as dst_zip:
        for item in src_zip.infolist():
            data = src_zip.read(item.filename)
            if item.filename.endswith(".tcip/models/registry.json"):
                data = tcip_store.RECORD_JSON.encode({"schema_version": 2, "entries": entries})
            dst_zip.writestr(item, data)

    dest = tmp_path / "restored"
    imported = import_project(str(poisoned), str(dest))

    assert "error" not in imported, imported
    stored = tcip_store.read(registry_index_key(dest))
    assert "schema_version" not in stored
    conformed = read_registry_index(dest)
    assert conformed[0]["checkpoint_path"] == ".tcip/models/m.pt"
    resolved = ModelRegistry(str(dest)).get_model("m")["checkpoint_path"]
    assert Path(resolved).is_file()


def test_import_project_discloses_a_designed_external_checkpoint_separately_from_unresolved(
    tmp_path: Path,
):
    """A registry entry that is a genuine designed-external claim (outside the project tree
    entirely) must appear in ``external_checkpoints``, never counted toward
    ``checkpoint_paths_unresolved``, which names only an entry expected to resolve under the
    tree that does not."""
    from tcip_mcp.model_registry import ModelRegistry

    src = tmp_path / "src_project"
    initialize_project(str(src), site="north orchard")
    internal_dir = src / ".tcip" / "models"
    internal_dir.mkdir(parents=True, exist_ok=True)
    internal_ckpt = internal_dir / "internal.pt"
    internal_ckpt.write_bytes(b"internal weights")
    external_dir = tmp_path / "elsewhere"
    external_dir.mkdir()
    external_ckpt = external_dir / "external.pt"
    external_ckpt.write_bytes(b"external weights")

    reg = ModelRegistry(str(src))
    reg.register_model("m_internal", str(internal_ckpt), {}, metrics_source=None)
    reg.register_model("m_external", str(external_ckpt), {}, metrics_source=None)

    zip_path = tmp_path / "export.zip"
    exported = archive_project(str(src), str(zip_path), include_models=True)
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
    _blob_files' two homes (the models glob and the registered-checkpoint reader); it must land
    in the bundle once, not as a duplicate zip member neither door's own accounting predicts."""
    from tcip_mcp.model_registry import ModelRegistry
    from tcip_mcp.tools.bundle import account_for

    src = tmp_path / "src_project"
    initialize_project(str(src), site="north orchard")
    ckpt = src / ".tcip" / "models" / "m.pt"
    ckpt.write_bytes(b"weights")
    ModelRegistry(str(src)).register_model("m", str(ckpt), {}, metrics_source=None)

    accounting = account_for(src)
    blob_names = [os.path.normcase(str(p)) for p in accounting.blobs]
    assert blob_names.count(os.path.normcase(str(ckpt))) == 1

    result = archive_project(str(src), str(tmp_path / "export.zip"), include_models=True)
    assert "error" not in result, result

    import zipfile

    with zipfile.ZipFile(str(tmp_path / "export.zip")) as zf:
        names = zf.namelist()
    matching = [n for n in names if n.endswith("m.pt")]
    assert len(matching) == 1, f"m.pt bundled more than once: {names}"


def test_archive_project_carries_a_registered_checkpoint_inside_model_src_when_models_excluded(
    tmp_path: Path,
):
    """A bespoke run's model_src/ snapshot travels regardless of include_models. A checkpoint that
    happens to sit inside that snapshot, and is also registered, must classify as model_src, not
    as a checkpoint blob include_models=False is entitled to drop."""
    from tcip_mcp.model_registry import ModelRegistry

    src = tmp_path / "src_project"
    initialize_project(str(src), site="north orchard")
    model_src = src / ".tcip" / "experiments" / "exp_001" / "model_src" / "abcd1234"
    model_src.mkdir(parents=True)
    ckpt = model_src / "weights.pt"
    ckpt.write_bytes(b"snapshot-bundled weights")
    ModelRegistry(str(src)).register_model("snap", str(ckpt), {}, metrics_source=None)

    result = archive_project(str(src), str(tmp_path / "export.zip"), include_models=False)
    assert "error" not in result, result

    import zipfile

    with zipfile.ZipFile(str(tmp_path / "export.zip")) as zf:
        names = zf.namelist()
    assert any(n.endswith("weights.pt") for n in names), (
        f"a checkpoint inside a model_src snapshot must travel regardless of include_models: {names}"
    )


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
                                     "crop": "currant", "fingerprint": f"v1:{name}"})
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

    upsert_dataset(project, {"id": "aaa", "path": str(project), "crop": "currant"})
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
    committed = b'{"crop": "currant", "id": "committed_id", "fingerprint": "written first"}\n'
    real_uuid4 = uuid_module.uuid4
    raced: list[str] = []

    def racing_uuid4():
        if not raced:
            raced.append("minted")
            (src / "dataset.json").write_bytes(committed)
        return real_uuid4()

    monkeypatch.setattr(uuid_module, "uuid4", racing_uuid4)

    result = register_dataset(str(src), crop="currant")

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

    refused = register_dataset(str(src), crop="currant")

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


def test_initialize_project_records_the_site(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    result = initialize_project(str(tmp_path), site="north orchard")

    assert result["site"] == "north orchard"
    from tcip_mcp.project_record import read_record

    assert read_record(str(tmp_path)) == {"site": "north orchard"}


def test_initialize_project_run_twice_with_the_same_site_is_idempotent(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    initialize_project(str(tmp_path), site="north orchard")

    result = initialize_project(str(tmp_path), site="north orchard")

    assert "error" not in result
    assert result["site"] == "north orchard"


def test_initialize_project_refuses_a_different_site_than_the_one_already_recorded(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    initialize_project(str(tmp_path), site="north orchard")

    result = initialize_project(str(tmp_path), site="south orchard")

    assert "error" in result
    assert "north orchard" in result["error"]
    assert "south orchard" in result["error"]
    from tcip_mcp.project_record import read_record

    assert read_record(str(tmp_path))["site"] == "north orchard"


def test_initialize_project_refuses_an_empty_site(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))

    result = initialize_project(str(tmp_path), site="   ")

    assert "error" in result
    from tcip_mcp.project_record import site_fields

    assert site_fields(str(tmp_path))["site"] is None


def test_initialize_project_refuses_an_empty_site_leaving_nothing_on_disk(tmp_path: Path, monkeypatch):
    """A refused site is validated before anything is created, the same as the name-scheme
    refusal: the destination is left exactly as it was, not half-scaffolded."""
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    dest = tmp_path / "fresh_project"

    result = initialize_project(str(dest), site="   ")

    assert "error" in result
    assert not dest.exists()


def test_initialize_project_scaffolds_a_relative_path_where_it_resolves(tmp_path: Path, monkeypatch):
    """A relative project_path scaffolds and records at the same absolute location the
    workspace-name check itself resolved, rather than the record write refusing a relative
    root after ``.tcip`` already exists."""
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    monkeypatch.chdir(tmp_path)

    result = initialize_project("relative_proj", site="north orchard")

    assert "error" not in result
    assert (tmp_path / "relative_proj" / ".tcip").is_dir()
    from tcip_mcp.project_record import read_record

    assert read_record(str(tmp_path / "relative_proj")) == {"site": "north orchard"}


def test_initialize_project_refuses_a_present_but_invalid_record(tmp_path: Path, monkeypatch):
    """The door surfaces the reader's own refusal rather than the store's raw exception."""
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    from tcip_mcp.project_record import project_record_key

    key = project_record_key(str(tmp_path))
    tcip_store.replace(key, {"not_site": "whatever"}, expect=tcip_store.Version.ABSENT)

    result = initialize_project(str(tmp_path), site="north orchard")

    assert "error" in result
    assert "does not hold a site" in result["error"]


def test_initialize_project_refuses_an_undecodable_record(tmp_path: Path, monkeypatch):
    """The store's own DecodeError is a StoreError, caught and returned as the door's error."""
    from tcip_mcp.project_record import project_record_key
    from tests._record_damage_fixtures import damage_record

    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    key = project_record_key(str(tmp_path))
    tcip_store.replace(key, {"site": "north orchard"}, expect=tcip_store.Version.ABSENT)
    damage_record(key, b"{not valid json")

    result = initialize_project(str(tmp_path), site="north orchard")

    assert "error" in result
    assert "does not decode" in result["error"]


def test_initialize_project_refuses_an_unadopted_root(tmp_path: Path, monkeypatch):
    """A root whose records are still loose files: the store's conform rail refuses
    initialize_project's site write there until tcip adopt-store has run, the same rule every
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
    # initialize_project's own audit entry lands at the platform root, not dest; a throwaway root here
    # keeps it off tmp_path, which stage two's own audit write below needs to find pristine.
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "scratch_platform_root"))
    file_backend = FileBackend()
    tcip_store.bind(file_backend)
    try:
        initialize_project(str(dest), site="north orchard")
    finally:
        tcip_store.bind(previous)
        file_backend.close()

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    backend = SqliteBackend()
    tcip_store.bind(backend)
    try:
        result = initialize_project(str(dest), site="north orchard")
    finally:
        tcip_store.bind(previous)
        backend.close()

    assert "error" in result
    assert "tcip adopt-store" in result["error"]


def test_initialize_project_records_the_site_on_a_directory_that_gained_tcip_with_no_creating_door(
    tmp_path: Path, monkeypatch
):
    """The reachable state a store write with no door leaves (``report_friction`` on a bare
    directory): ``initialize_project`` on it afterward records the site the same way it would on a
    truly fresh directory, since the writer's create-only write does not distinguish the two."""
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    from tcip_mcp.tools.meta_tools import report_friction

    report_friction(str(tmp_path), category="missing_tool", detail="a")
    assert (tmp_path / ".tcip").is_dir()

    result = initialize_project(str(tmp_path), site="north orchard")

    assert "error" not in result
    assert result["site"] == "north orchard"


def test_inspect_project_reports_site_fields_across_project_states(tmp_path: Path, monkeypatch):
    """A path with no ``.tcip`` carries neither key; a project with a record carries the site; a
    project with ``.tcip`` and no record carries the absent-record problem text."""
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    from tcip_mcp.tools.meta_tools import report_friction

    bare = tmp_path / "no_tcip"
    bare.mkdir()
    status = inspect_project(str(bare))
    assert "site" not in status
    assert "site_problem" not in status

    recordless = tmp_path / "recordless"
    recordless.mkdir()
    report_friction(str(recordless), category="missing_tool", detail="a")
    status = inspect_project(str(recordless))
    assert status["site"] is None
    assert "initialize_project" in status["site_problem"]

    recorded = tmp_path / "recorded"
    recorded.mkdir()
    initialize_project(str(recorded), site="north orchard")
    status = inspect_project(str(recorded))
    assert status["site"] == "north orchard"
    assert status["site_problem"] is None


def test_inspect_project_reports_an_invalid_record(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    from tcip_mcp.project_record import project_record_key

    initialize_project(str(tmp_path), site="north orchard")
    key = project_record_key(str(tmp_path))
    current = tcip_store.read_versioned(key).version
    tcip_store.replace(key, {"not_site": "x"}, expect=current)

    status = inspect_project(str(tmp_path))

    assert status["site"] is None
    assert "does not hold a site" in status["site_problem"]


def test_archive_and_import_carry_the_project_record(tmp_path: Path, monkeypatch):
    """initialize_project -> archive_project -> import_project round-trips a project whose record is
    on disk in the archive: the record travels with the project like every other ``.tcip``
    document, and archive_project exports it itself, so no operator step sits between the two
    doors."""
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    src = tmp_path / "src_project"
    initialize_project(str(src), site="north orchard")

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
