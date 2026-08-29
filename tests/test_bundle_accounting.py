"""The shared membership accounting (tcip_mcp.tools.bundle): what a project bundle holds, one
implementation both archive_project and import_project compose from or judge by.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp import class_registry
from tcip_mcp.class_registry import ClassRegistry, Subject
from tcip_mcp.tools.bundle import AnchorMisplaced, account_for


def _dataset_tree(root: Path) -> None:
    (root / "images" / "2026-03-04").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16)).save(root / "images" / "2026-03-04" / "a_1.jpg")
    (root / "annotations" / "2026-03-04").mkdir(parents=True, exist_ok=True)
    json_io.write_annotations(
        str(root / "annotations" / "2026-03-04" / "a_1.json"),
        [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))], 16, 16)
    class_registry.write_registry(root / "classes.json", ClassRegistry(subjects=(Subject(name="catkin"),)))
    (root / "dataset.json").write_text('{"crop": "hazelnut", "id": "x", "fingerprint": "y"}',
                                       encoding="utf-8")


def _plan_paths(accounting) -> set[str]:
    return {str(entry.path) for plan in accounting.plans for entry in plan.entries}


def test_a_plain_dataset_tree_is_all_blob_and_nothing_unaccounted(tmp_path: Path):
    root = tmp_path / "proj"
    _dataset_tree(root)

    accounting = account_for(root)

    assert not accounting.unaccounted
    assert not accounting.bookkeeping
    assert not _plan_paths(accounting)
    blobs = {p.name for p in accounting.blobs}
    assert {"a_1.jpg", "a_1.json", "classes.json", "dataset.json"} <= blobs


def test_a_state_record_is_claimed_under_the_state_root(tmp_path: Path):
    root = tmp_path / "proj"
    _dataset_tree(root)
    trait_specs = root / ".tcip" / "state" / "trait_specs"
    trait_specs.mkdir(parents=True)
    (trait_specs / "catkin.json").write_text("{}", encoding="utf-8")

    accounting = account_for(root)

    assert str(trait_specs / "catkin.json") in _plan_paths(accounting)
    assert not accounting.unaccounted


def test_an_experiment_member_and_a_run_only_launch_config_are_each_claimed(tmp_path: Path):
    root = tmp_path / "proj"
    _dataset_tree(root)
    run_dir = root / ".tcip" / "experiments" / "exp1"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text('{"epoch": 1}\n', encoding="utf-8")
    (run_dir / "launch_config.json").write_text("{}", encoding="utf-8")

    accounting = account_for(root)

    claimed = _plan_paths(accounting)
    assert str(run_dir / "config.json") in claimed
    assert str(run_dir / "metrics.jsonl") in claimed
    assert str(run_dir / "launch_config.json") in claimed
    assert not accounting.unaccounted


def test_an_hpo_study_result_and_sweep_manifest_and_trial_members_are_claimed(tmp_path: Path):
    root = tmp_path / "proj"
    _dataset_tree(root)
    hpo = root / ".tcip" / "hpo"
    hpo.mkdir(parents=True)
    (hpo / "study1.json").write_text("{}", encoding="utf-8")  # the unanchored study-result template
    study_dir = hpo / "study1"
    study_dir.mkdir()
    (study_dir / "manifest.json").write_text("{}", encoding="utf-8")
    trial_dir = study_dir / "trial_0"
    trial_dir.mkdir()
    (trial_dir / "resolved_config.json").write_text("{}", encoding="utf-8")
    (trial_dir / "metrics.jsonl").write_text('{"iter": 1}\n', encoding="utf-8")

    accounting = account_for(root)

    claimed = _plan_paths(accounting)
    assert str(hpo / "study1.json") in claimed
    assert str(study_dir / "manifest.json") in claimed
    assert str(trial_dir / "resolved_config.json") in claimed
    assert str(trial_dir / "metrics.jsonl") in claimed
    assert not accounting.unaccounted


def test_a_project_relative_splits_manifest_is_derived_and_claimed(tmp_path: Path):
    root = tmp_path / "proj"
    _dataset_tree(root)
    splits_dir = root / "splits_out"
    splits_dir.mkdir()
    (splits_dir / "split_manifest.json").write_text("{}", encoding="utf-8")

    accounting = account_for(root)

    assert str(splits_dir / "split_manifest.json") in _plan_paths(accounting)
    assert not accounting.unaccounted


def test_a_split_manifest_at_the_tree_root_refuses_by_name(tmp_path: Path):
    root = tmp_path / "proj"
    _dataset_tree(root)
    (root / "split_manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(AnchorMisplaced, match="split_manifest.json"):
        account_for(root)


def test_a_split_manifest_under_annotations_refuses_by_name(tmp_path: Path):
    root = tmp_path / "proj"
    _dataset_tree(root)
    (root / "annotations" / "split_manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(AnchorMisplaced, match="split_manifest.json"):
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
