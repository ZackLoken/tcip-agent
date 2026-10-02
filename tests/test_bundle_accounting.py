"""The shared membership accounting (tcip_mcp.tools.bundle): what a project bundle holds, one
implementation both archive_project and import_project compose from or judge by.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.subject_registry import SubjectRegistry, Subject
from tcip_mcp.tools.bundle import AnchorMisplaced, account_for
from tcip_store.sqlite_backend import SqliteBackend
from tests._producer_fixtures import registry_over
from tests.test_archive_store_gate import bound


def _dataset_tree(root: Path) -> None:
    (root / "images" / "2026-03-04").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16)).save(root / "images" / "2026-03-04" / "a_1.jpg")
    (root / "annotations" / "2026-03-04").mkdir(parents=True, exist_ok=True)
    json_io.write_annotations(
        str(root / "annotations" / "2026-03-04" / "a_1.json"),
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 16, 16)
    registry_over(root, SubjectRegistry(subjects=(Subject(name="bud"),)))
    (root / "dataset.json").write_text('{"crop": "currant", "id": "x", "fingerprint": "y"}',
                                       encoding="utf-8")


def _plan_paths(accounting) -> set[str]:
    return {str(entry.path) for plan in accounting.plans for entry in plan.entries}


def test_a_plain_dataset_tree_is_all_blob_and_nothing_unaccounted(tmp_path: Path):
    root = tmp_path / "proj"
    with bound(SqliteBackend()):
        _dataset_tree(root)

    accounting = account_for(root)

    assert not accounting.unaccounted
    # Lock residue (kept under Unix) and the registry save's audit-log database are bookkeeping.
    assert all(entry.name.endswith(".lock") or entry.name.startswith("store.db")
               for entry in accounting.bookkeeping), accounting.bookkeeping
    assert not _plan_paths(accounting)
    blobs = {p.name for p in accounting.blobs}
    assert {"a_1.jpg", "a_1.json", "subjects.json", "dataset.json"} <= blobs


def test_a_state_record_is_claimed_under_the_state_root(tmp_path: Path):
    root = tmp_path / "proj"
    _dataset_tree(root)
    trait_records = root / ".tcip" / "state" / "traits"
    trait_records.mkdir(parents=True)
    (trait_records / "bud.json").write_text("{}", encoding="utf-8")

    accounting = account_for(root)

    assert str(trait_records / "bud.json") in _plan_paths(accounting)
    assert not accounting.unaccounted


def test_every_file_of_a_run_and_a_sweeps_trial_is_a_run_blob(tmp_path: Path):
    """A run directory and a sweep's trial run directory travel whole as files, written by the
    launcher's own writer and the envelope's own sink: none is a store record and none is left
    unaccounted."""
    from tcip_mcp.experiments import sweeps_dir
    from tcip_mcp.pipelines.data.split_construction import resolve_run
    from tcip_mcp.tools.bundle import BLOB_RUNS, blob_home
    from tcip_mcp.tools.training_tools import open_run
    from tests._verified_checkpoint_fixtures import detection_config, log_epoch, opened_run

    root = tmp_path / "proj"
    _dataset_tree(root)
    config = detection_config(tmp_path / "ds")
    run_dir = opened_run(root, config, experiment_id="exp1")
    log_epoch(run_dir, 1, {"loss": 0.5})
    trial_dir = sweeps_dir(root) / "study1" / "trial_0"
    open_run(trial_dir, dict(config), resolve_run(config, project=root).record,
             trial_params={"lr": 0.1})

    accounting = account_for(root)

    blobs = {str(p) for p in accounting.blobs}
    for member in (run_dir / "run.json", run_dir / "metrics.jsonl", trial_dir / "run.json"):
        assert str(member) in blobs
        assert blob_home(accounting.tree, member) == BLOB_RUNS
    assert str(run_dir / "run.json") not in _plan_paths(accounting)
    assert not accounting.unaccounted


def test_a_project_relative_splits_manifest_is_derived_and_claimed(tmp_path: Path):
    root = tmp_path / "proj"
    _dataset_tree(root)
    splits_dir = root / "splits_out"
    splits_dir.mkdir()
    (splits_dir / "selection.json").write_text("{}", encoding="utf-8")

    accounting = account_for(root)

    assert str(splits_dir / "selection.json") in _plan_paths(accounting)
    assert not accounting.unaccounted


def test_a_selection_at_the_tree_root_refuses_by_name(tmp_path: Path):
    root = tmp_path / "proj"
    _dataset_tree(root)
    (root / "selection.json").write_text("{}", encoding="utf-8")

    with pytest.raises(AnchorMisplaced, match="selection.json"):
        account_for(root)


def test_a_selection_under_annotations_refuses_by_name(tmp_path: Path):
    root = tmp_path / "proj"
    _dataset_tree(root)
    (root / "annotations" / "selection.json").write_text("{}", encoding="utf-8")

    with pytest.raises(AnchorMisplaced, match="selection.json"):
        account_for(root)


def test_an_unclaimed_stray_under_tcip_state_is_unaccounted(tmp_path: Path):
    root = tmp_path / "proj"
    _dataset_tree(root)
    (root / ".tcip" / "state").mkdir(parents=True)
    probe = root / ".tcip" / "state" / "_write_probe.txt"
    probe.write_text("probe", encoding="utf-8")

    accounting = account_for(root)

    assert probe in accounting.unaccounted


def test_a_database_file_is_bookkeeping_not_unaccounted(tmp_path: Path):
    root = tmp_path / "proj"
    _dataset_tree(root)
    (root / ".tcip").mkdir(parents=True, exist_ok=True)
    db = root / ".tcip" / "store.db"
    db.write_bytes(b"not a real database, just bytes")

    accounting = account_for(root)

    assert db in accounting.bookkeeping
    assert db not in accounting.unaccounted


def test_model_src_files_under_a_run_are_blob(tmp_path: Path):
    root = tmp_path / "proj"
    _dataset_tree(root)
    model_src = root / ".tcip" / "experiments" / "exp1" / "model_src"
    model_src.mkdir(parents=True)
    (model_src / "model.py").write_text("class M: pass", encoding="utf-8")

    accounting = account_for(root)

    assert model_src / "model.py" in accounting.blobs
    assert not accounting.unaccounted


def test_a_checkpoint_directly_under_tcip_models_is_blob(tmp_path: Path):
    root = tmp_path / "proj"
    _dataset_tree(root)
    models = root / ".tcip" / "models"
    models.mkdir(parents=True)
    (models / "m.pt").write_bytes(b"weights")

    accounting = account_for(root)

    assert models / "m.pt" in accounting.blobs
    assert not accounting.unaccounted


def test_account_for_works_with_only_the_package_on_sys_path(tmp_path: Path):
    """Coverage: account_for's store-catalog import must not need the repository's own
    ``tools`` package, which exists only with the repo root on sys.path; an installed
    deployment never puts it there."""
    import os
    import subprocess
    import sys

    root = tmp_path / "proj"
    _dataset_tree(root)

    repo_root = Path(__file__).resolve().parent.parent
    src_dirs = [str(repo_root / "packages" / pkg / "src") for pkg in (
        "tcip-store", "tcip-annotation", "tcip-mcp", "tcip-web",
    )]
    script = f"from tcip_mcp.tools.bundle import account_for; account_for({str(root)!r})"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(src_dirs)

    result = subprocess.run(
        [sys.executable, "-c", script], cwd=str(tmp_path), env=env,
        capture_output=True, text=True, timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
