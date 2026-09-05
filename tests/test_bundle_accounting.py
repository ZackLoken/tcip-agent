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
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 16, 16)
    class_registry.write_registry(root / "classes.json", ClassRegistry(subjects=(Subject(name="bud"),)))
    (root / "dataset.json").write_text('{"crop": "currant", "id": "x", "fingerprint": "y"}',
                                       encoding="utf-8")


def _plan_paths(accounting) -> set[str]:
    return {str(entry.path) for plan in accounting.plans for entry in plan.entries}


def test_a_plain_dataset_tree_is_all_blob_and_nothing_unaccounted(tmp_path: Path):
    root = tmp_path / "proj"
    _dataset_tree(root)

    accounting = account_for(root)

    assert not accounting.unaccounted
    # filelock keeps the released lock file under Unix and deletes it under Windows, so the
    # producers' own lock residue is the one bookkeeping content a plain tree may carry.
    assert all(entry.name.endswith(".lock") for entry in accounting.bookkeeping)
    assert not _plan_paths(accounting)
    blobs = {p.name for p in accounting.blobs}
    assert {"a_1.jpg", "a_1.json", "classes.json", "dataset.json"} <= blobs


def test_a_state_record_is_claimed_under_the_state_root(tmp_path: Path):
    root = tmp_path / "proj"
    _dataset_tree(root)
    trait_specs = root / ".tcip" / "state" / "trait_specs"
    trait_specs.mkdir(parents=True)
    (trait_specs / "bud.json").write_text("{}", encoding="utf-8")

    accounting = account_for(root)

    assert str(trait_specs / "bud.json") in _plan_paths(accounting)
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


def test_account_for_works_with_only_the_package_on_sys_path(tmp_path: Path):
    """Coverage: account_for's store-catalogue import must not need the repository's own
    ``scripts`` package, which exists only with the repo root on sys.path; an installed
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
