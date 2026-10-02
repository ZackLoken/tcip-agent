"""Tests for project management tools."""

from __future__ import annotations

import os
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

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import SubjectRegistry, Subject
    from tests._producer_fixtures import registry_over

    (root / "images" / "2-11-26").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32)).save(root / "images" / "2-11-26" / "img_000.jpg")
    (root / "annotations" / "2-11-26").mkdir(parents=True, exist_ok=True)
    json_io.write_annotations(
        str(root / "annotations" / "2-11-26" / "img_000.json"),
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)
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
    # the project registry knows the dataset.
    regs = read_datasets(tmp_path)
    assert len(regs) == 1 and regs[0]["id"] == res["id"] and regs[0]["crop"] == "currant"


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
    shutil.copytree(src, moved)  # same content, new path
    register_dataset(tmp_path, str(moved), crop="currant")

    regs = read_datasets(tmp_path)
    same = [r for r in regs if r["id"] == reg["id"]]
    assert len(same) == 1  # one entry for the id: the move updated the path, not duplicated
    assert same[0]["path"] == stored_path(moved, tmp_path)
    assert same[0]["fingerprint"] == reg["fingerprint"]  # unchanged content -> same fingerprint


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
    assert status["recent_activity"] == {}  # no history yet: genuinely empty, not corrupt

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


def test_inspect_project_surfaces_corrupt_status_honestly(tmp_path: Path):
    from tcip_mcp.project_status import project_status_key, record_report

    _initialized(tmp_path)
    record_report(tmp_path)  # seed a real record so a damaged one has somewhere to overwrite
    damage_record(project_status_key(tmp_path), b"{not valid json")

    status = inspect_project(tmp_path)
    assert "status_unavailable" in status["recent_activity"]
    # The record reads unaffected by a corrupt status file: different store, different rail.
    assert status["record_problem"] is None


def test_inspect_project_surfaces_version_refused_status_distinctly_from_corrupt(tmp_path: Path):
    from tcip_mcp.project_status import (
        PROJECT_STATUS_STORE, project_status_key, record_report,
    )

    _initialized(tmp_path)
    record_report(tmp_path)  # seed a real record so a poisoned one has somewhere to overwrite
    poisoned = tcip_store.get_descriptor(PROJECT_STATUS_STORE).codec.encode(
        {"reports_since_last_retrospective": 1, "schema_version": 99})
    damage_record(project_status_key(tmp_path), poisoned)

    status = inspect_project(tmp_path)
    assert "schema_version" in status["recent_activity"]["status_unavailable"]
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
    labels = src / "annotations" / date
    for d in (images, labels):
        d.mkdir(parents=True)
    record = _initialized(src)

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp import subject_registry
    from tcip_mcp.subject_registry import SubjectRegistry, Subject
    from tests._producer_fixtures import registry_over

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
    # The subject registry decodes the labels' names: a self-contained bundle must carry it, or the
    # archived annotations are unreadable on the other end. One nested subjects.json at the root.
    registry_over(src, SubjectRegistry(subjects=(Subject(name="bud", description="a currant bud"),)))
    reg = register_dataset(src, str(src), crop="currant")  # identity travels with the data

    zip_path = tmp_path / "export.zip"
    exported = archive_project(src, str(zip_path))
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

    status = inspect_project(dest)
    assert status["id"] == record["id"]
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
    restored = subject_registry.read_registry(dest)
    assert [s.name for s in restored.subjects] == ["bud"]
    # dataset.json traveled with the data: identity (id/crop/fingerprint) survives the round-trip.
    import json

    restored_id = json.loads((dest / "dataset.json").read_text())
    assert restored_id == {"crop": "currant", "id": reg["id"], "fingerprint": reg["fingerprint"]}


def test_project_roots_names_a_run_output_dir_a_selection_and_a_prediction_bucket(
    tmp_path: Path,
):
    """project_roots reaches every layout a project's own records name it under, not only the
    registered dataset roots: each run directory and the selection a run bound to (its resolved
    partition's selection.selection_dir)."""
    from tcip_store.layout_claims import RUN, SPLITS

    from tcip_mcp.store_catalog import project_roots

    project = tmp_path / "project"
    dataset = tmp_path / "dataset"
    project.mkdir()
    dataset.mkdir()
    _make_dataset(dataset)
    register_dataset(project, str(dataset), crop="currant")

    split_dir = tmp_path / "splits" / "frozen-exp-1"
    split_dir.mkdir(parents=True)
    run_dir = _run_bound_to(project, split_dir)

    roots = project_roots(project)

    assert (str(run_dir.absolute()), RUN) in roots
    assert (str(split_dir.absolute()), SPLITS) in roots


def _run_bound_to(project: Path, selection_dir: Path) -> Path:
    """A run directory under ``project`` whose launch record's partition is bound to a selection
    ``draw_splits`` drew into ``selection_dir`` (over a dataset beside it), resolved and opened
    by the launcher's own producer and writer."""
    from tests._verified_checkpoint_fixtures import opened_run
    from tests.test_selection_binding import _draw, _two_subject_two_date_dataset

    _draw(project, _two_subject_two_date_dataset(
        selection_dir.parent / f"{selection_dir.name}-ds"), selection_dir)
    return opened_run(project, {"model_source": {"task": "detection"},
                                "data": {"split": {"selection_dir": str(selection_dir)}}})


def test_project_roots_keeps_both_layouts_when_one_directory_is_two_kinds_of_root(
    tmp_path: Path,
):
    """A directory a run bound to as its selection that is also registered as a project dataset
    keeps both layouts: _add is keyed on the (path, layout) pair, not the path alone, so the
    dataset-registry add is not silently dropped because the selection add already claimed that
    path."""
    from tcip_store.layout_claims import ROOT, SPLITS

    from tcip_mcp.store_catalog import project_roots

    project = tmp_path / "project"
    project.mkdir()

    shared = tmp_path / "shared"
    shared.mkdir()
    _make_dataset(shared)
    _run_bound_to(project, shared)
    register_dataset(project, str(shared), crop="currant")

    roots = project_roots(project)

    assert (str(shared.absolute()), SPLITS) in roots
    assert (str(shared.absolute()), ROOT) in roots


def test_project_roots_skips_a_bound_selection_that_no_longer_exists(tmp_path: Path):
    """A run's resolved record can still name a selection directory that has since been moved or
    deleted; project_roots skips it rather than handing ``tcip adopt-store`` a path to recreate
    from nothing."""
    import shutil

    from tcip_store.layout_claims import SPLITS

    from tcip_mcp.store_catalog import project_roots

    project = tmp_path / "project"
    project.mkdir()

    split_dir = tmp_path / "splits" / "gone"
    split_dir.mkdir(parents=True)
    _run_bound_to(project, split_dir)
    tcip_store.release_root(split_dir)  # the selection's own store lets go of its file
    shutil.rmtree(split_dir)

    roots = project_roots(project)

    assert not any(layout == SPLITS for _, layout in roots)


def test_external_dataset_paths_names_an_external_registry_entry(tmp_path: Path):
    """import_project calls this after extraction to disclose which registered datasets stayed
    external."""
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
    """A checkpoint under .tcip/models/*.pt is dropped by include_models=False; left_behind
    names that count separately from unaccounted and bookkeeping, rather than folding it in."""
    src = tmp_path / "src_project"
    _initialized(src)
    (src / ".tcip" / "models" / "m.pt").write_bytes(b"weights")

    result = archive_project(src, str(tmp_path / "export.zip"))

    assert "error" not in result
    assert result["left_behind"]["checkpoints_excluded"] == 1
    assert result["left_behind"]["unaccounted"] == 0

    result_included = archive_project(src, str(tmp_path / "export2.zip"), include_models=True)
    assert result_included["left_behind"]["checkpoints_excluded"] == 0


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
    _blob_files' two homes (the models glob and the registered-checkpoint reader); it must land
    in the bundle once, not as a duplicate zip member neither door's own accounting predicts."""
    from tcip_mcp.model_registry import ModelRegistry
    from tcip_mcp.tools.bundle import account_for
    from tests._verified_checkpoint_fixtures import checkpoint_file

    src = tmp_path / "src_project"
    _initialized(src)
    ckpt = checkpoint_file(src / ".tcip" / "models" / "m.pt", "weights")
    ModelRegistry(str(src)).register_model("m", str(ckpt), {})

    accounting = account_for(src)
    blob_names = [os.path.normcase(str(p)) for p in accounting.blobs]
    assert blob_names.count(os.path.normcase(str(ckpt))) == 1

    result = archive_project(src, str(tmp_path / "export.zip"), include_models=True)
    assert "error" not in result, result

    import zipfile

    with zipfile.ZipFile(str(tmp_path / "export.zip")) as zf:
        names = zf.namelist()
    matching = [n for n in names if n.endswith("m.pt")]
    assert len(matching) == 1, f"m.pt bundled more than once: {names}"


def test_archive_project_carries_a_registered_checkpoint_inside_model_src_when_models_excluded(
    tmp_path: Path,
):
    """A bespoke run's model_src/ snapshot travels regardless of include_models. A weights file
    that happens to sit inside that snapshot is a run file, not a checkpoint blob
    include_models=False is entitled to drop."""
    src = tmp_path / "src_project"
    _initialized(src)
    model_src = src / ".tcip" / "experiments" / "exp_001" / "model_src" / "abcd1234"
    model_src.mkdir(parents=True)
    ckpt = model_src / "weights.pt"
    ckpt.write_bytes(b"snapshot-bundled weights")

    result = archive_project(src, str(tmp_path / "export.zip"), include_models=False)
    assert "error" not in result, result

    import zipfile

    with zipfile.ZipFile(str(tmp_path / "export.zip")) as zf:
        names = zf.namelist()
    assert any(n.endswith("weights.pt") for n in names), (
        f"a checkpoint inside a model_src snapshot must travel regardless of include_models: {names}"
    )


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


def test_initialize_project_refuses_an_unadopted_root(tmp_path: Path):
    """A root whose records are still loose files: the store's conform rail refuses
    initialize_project's record write there until tcip adopt-store has run, the same rule every
    other record store under that root already obeys. The file backend legitimately produces
    that state, so the unadopted root here is built by writing through the file backend directly
    and then judged under the database backend."""
    from tcip_store.file_backend import FileBackend
    from tcip_store.sqlite_backend import SqliteBackend
    from tcip_store.store import _backend

    dest = tmp_path / "unadopted"
    previous = _backend()
    file_backend = FileBackend()
    tcip_store.bind(file_backend)
    try:
        _initialized(dest)
    finally:
        tcip_store.bind(previous)
        file_backend.close()

    backend = SqliteBackend()
    tcip_store.bind(backend)
    try:
        result = initialize_project(str(dest), "Test project", "north orchard")
    finally:
        tcip_store.bind(previous)
        backend.close()

    assert "error" in result
    assert "tcip adopt-store" in result["error"]


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
    """initialize_project -> archive_project -> import_project round-trips a project whose record
    is on disk in the archive: the record, its id included, travels with the project like every
    other ``.tcip`` document, and archive_project exports it itself, so no operator step sits
    between the two doors."""
    src = tmp_path / "src_project"
    record = _initialized(src)

    zip_path = tmp_path / "export.zip"
    exported = archive_project(src, str(zip_path))
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

    restored = read_record(dest)
    assert (restored["id"], restored["site"]) == (record["id"], "north orchard")
