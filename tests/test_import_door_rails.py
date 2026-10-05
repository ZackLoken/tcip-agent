"""The import door's own rails: staging, the refusal of the store's own bookkeeping and of an
escaping member, and the move. Each refusal here is paired with the legitimate call the same rail
must still admit.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp.tools.project_tools import archive_project, import_project
from tcip_mcp.web_client import gui_snapshot_key
from tcip_store.file_backend import lock_file_for


def _project(root: Path) -> Path:
    """A dataset root with one image, one empty label, and the registry that decodes it."""
    from tcip_mcp.subject_registry import SubjectRegistry, Subject
    from tests._producer_fixtures import label_image, registry_over

    (root / "images" / "2026-03-04").mkdir(parents=True, exist_ok=True)
    (root / "images" / "2026-03-04" / "a_1.jpg").write_bytes(b"\xff\xd8\xff")
    label_image(root / "images" / "2026-03-04" / "a_1.jpg", [], 8, 8, keep_empty=True)
    registry_over(root, SubjectRegistry(subjects=(Subject(name="bud"),)))
    return root


def _hand_zip(path: Path, members: dict[str, bytes]) -> Path:
    """A zip carrying exactly ``members``, hand-built: no producer writes these shapes."""
    with zipfile.ZipFile(str(path), "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


# ── rail 2: a non-empty destination refuses and changes nothing ────────────────────────────


def test_import_refuses_a_non_empty_destination_and_changes_nothing(tmp_path, monkeypatch):
    root = _project(tmp_path / "source")
    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(root, str(zip_path))

    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "subjects.json").write_bytes(b"already here")
    ts.replace(gui_snapshot_key(dest), {"active_subject": "x"}, expect=ts.Version.ABSENT)

    result = import_project(str(zip_path), str(dest))

    assert "error" in result
    assert str(dest) in result["error"]
    assert (dest / "subjects.json").read_bytes() == b"already here"
    assert ts.read(gui_snapshot_key(dest)) == {"active_subject": "x"}


def test_import_admits_a_pre_existing_empty_destination(tmp_path):
    root = _project(tmp_path / "source")
    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(root, str(zip_path))

    dest = tmp_path / "dest"
    dest.mkdir()

    result = import_project(str(zip_path), str(dest))

    assert "error" not in result
    assert (dest / "subjects.json").is_file()


# ── rail 5: a database's sidecar, or its lock, refuses ──────────────────────────────────────


def test_import_refuses_a_member_named_as_a_database_sidecar(tmp_path):
    zip_path = _hand_zip(tmp_path / "bundle.zip", {".tcip/store.db-wal": b"not a log"})
    dest = tmp_path / "dest"

    result = import_project(str(zip_path), str(dest))

    assert "error" in result
    assert "store.db-wal" in result["error"]
    assert not dest.exists()


def test_import_refuses_a_member_named_store_db_lock(tmp_path):
    zip_path = _hand_zip(tmp_path / "bundle.zip", {".tcip/store.db.lock": b""})
    dest = tmp_path / "dest"

    result = import_project(str(zip_path), str(dest))

    assert "error" in result
    assert not dest.exists()


# ── rail 8: zip-slip refusal ─────────────────────────────────────────────────────────────────


def test_import_refuses_a_zip_slip_path(tmp_path):
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(str(zip_path), "w") as zf:
        zf.writestr("../../evil.txt", "escaped")
    dest = tmp_path / "dest"

    result = import_project(str(zip_path), str(dest))

    assert "error" in result
    assert "Unsafe path" in result["error"]
    assert not dest.exists()


def test_extract_zip_refuses_a_sibling_directory_that_shares_stagings_name_as_a_prefix(tmp_path):
    """A string-prefix escape check reads a member resolving to ``<staging.name>extra/`` as
    contained, since that sibling's path literally starts with staging's own path; containment
    by ``relative_to`` does not."""
    from tcip_mcp.tools.project_tools import _extract_zip

    staging = tmp_path / "abcd1234"
    staging.mkdir()
    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(str(evil_zip), "w") as zf:
        zf.writestr(f"../{staging.name}extra/evil.txt", "escaped")

    with pytest.raises(ValueError, match="Unsafe path"):
        _extract_zip(evil_zip, staging)


def test_import_refuses_a_corrupt_zip_without_stranding_a_staging_tree(tmp_path):
    root = _project(tmp_path / "source")
    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(root, str(zip_path))
    data = zip_path.read_bytes()
    zip_path.write_bytes(data[: len(data) // 2])  # truncated: no longer a readable zip

    dest = tmp_path / "dest"
    result = import_project(str(zip_path), str(dest))

    assert "error" in result
    assert not dest.exists()
    imports_root = dest.parent / ".imports"
    assert not imports_root.is_dir() or not any(imports_root.iterdir())


# ── rail 10: a record the database holds travels in its database ────────────────────────────


def test_a_record_travels_inside_the_database_the_archive_copies(tmp_path):
    record = {"active_subject": "x"}
    root = _project(tmp_path / "source")
    ts.replace(gui_snapshot_key(root), record, expect=ts.Version.ABSENT)
    assert "error" not in archive_project(root, str(tmp_path / "bundle.zip"))

    dest = tmp_path / "dest"
    result = import_project(str(tmp_path / "bundle.zip"), str(dest))

    assert "error" not in result
    assert (dest / ".tcip" / "store.db").is_file()
    assert ts.read(gui_snapshot_key(dest)) == record


# ── rail 13: concurrency ─────────────────────────────────────────────────────────────────────


def test_a_locked_staging_sibling_is_left_alone_while_another_import_completes(tmp_path):
    import filelock

    root = _project(tmp_path / "source")
    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(root, str(zip_path))

    dest = tmp_path / "dest"
    imports_root = dest.parent / ".imports"
    imports_root.mkdir(parents=True, exist_ok=True)
    live = imports_root / "concurrent-run"
    live.mkdir()
    (live / "marker.txt").write_text("live", encoding="utf-8")
    # A raw, separately constructed FileLock stands in for a concurrent process's own hold.
    raw_lock = filelock.FileLock(str(lock_file_for(live)))
    raw_lock.acquire()
    try:
        result = import_project(str(zip_path), str(dest))

        assert "error" not in result
        assert live.is_dir()
        assert (live / "marker.txt").is_file()
    finally:
        raw_lock.release()


def test_a_free_locked_leftover_staging_sibling_is_swept_with_its_lock_file(tmp_path):
    root = _project(tmp_path / "source")
    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(root, str(zip_path))

    dest = tmp_path / "dest"
    imports_root = dest.parent / ".imports"
    imports_root.mkdir(parents=True, exist_ok=True)
    leftover = imports_root / "crash-run"
    leftover.mkdir()
    (leftover / "marker.txt").write_text("stale", encoding="utf-8")

    result = import_project(str(zip_path), str(dest))

    assert "error" not in result
    assert not leftover.exists()
    assert not lock_file_for(leftover).exists()


# ── rail 7: the dataset registry travels ────────────────────────────────────────────────────


def test_dataset_registry_stores_the_relative_dot_after_import(tmp_path):
    """The project's own dataset registers to itself, and that entry's ``path`` survives the
    archive/import round trip as the project-relative ``"."`` rather than an absolute path baked
    in before the move."""
    from tcip_mcp.tools.project_tools import read_datasets, register_dataset

    root = _project(tmp_path / "source")
    registered = register_dataset(root, str(root), crop="currant")
    assert "error" not in registered

    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(root, str(zip_path))
    dest = tmp_path / "dest"
    imported = import_project(str(zip_path), str(dest))

    assert "error" not in imported
    entries = read_datasets(dest)
    assert entries[0]["path"] == "."
    assert imported["dataset_paths_unresolved"] == []


def test_dataset_registry_travels_with_nothing_rewritten(tmp_path):
    """The accessor resolves the stored "." against wherever the project was actually
    imported, so nothing about the registry needed rewriting for the move to survive."""
    from tcip_mcp.tools.project_tools import dataset_entry_path, read_datasets, register_dataset

    root = _project(tmp_path / "source")
    registered = register_dataset(root, str(root), crop="currant")
    assert "error" not in registered

    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(root, str(zip_path))
    dest = tmp_path / "dest"
    imported = import_project(str(zip_path), str(dest))

    assert "error" not in imported
    entries = read_datasets(dest)
    assert dataset_entry_path(dest, entries[0]).resolve() == dest.resolve()


def test_an_external_dataset_entry_stays_absolute_and_is_disclosed(tmp_path):
    from tcip_mcp.tools.project_tools import register_dataset

    root = _project(tmp_path / "source")
    external = _project(tmp_path / "external_dataset")
    registered = register_dataset(root, str(external), crop="currant")
    assert "error" not in registered

    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(root, str(zip_path))
    dest = tmp_path / "dest"
    imported = import_project(str(zip_path), str(dest))

    assert "error" not in imported
    assert str(external.resolve()) in imported["dataset_paths_unresolved"]


# ── rail 1 / 6 / 9 / 14: the full round trip through real producers ─────────────────────────


def _annotated_dataset(root: Path, n: int) -> None:
    """``n`` distinct single-tile foreground groups of one subject, enough to clear
    draw_splits' floor (one group each for train/val, two for calibration)."""
    from PIL import Image

    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import SubjectRegistry, Subject
    from tests._producer_fixtures import label_image, registry_over

    images_dir = root / "images" / "2026-03-04"
    images_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        stem = f"img_{i:03d}"
        Image.new("RGB", (640, 480), color=(128, 128, 128)).save(images_dir / f"{stem}.jpg")
        label_image(images_dir / f"{stem}.jpg",
                    [Annotation(subject="bud", geometry=BBox(288, 216, 352, 264))], 640, 480)
    registry_over(root, SubjectRegistry(subjects=(Subject(name="bud"),)))


def test_a_splits_root_nested_under_a_nested_dataset_root_archives_and_round_trips(tmp_path):
    """A dataset sits under a project, and draw_splits partitions it in place, landing
    selection.json under the dataset root rather than beside it. The cross-anchor constraint must
    admit that nesting rather than refusing the whole project."""
    from tcip_mcp.pipelines.data.selection import selection_key
    from tcip_mcp.tools.data_tools import draw_splits

    project = tmp_path / "project"
    curated = project / "curated"
    _annotated_dataset(curated, 4)

    splits_result = draw_splits(project, str(curated), subject="bud", output_path=str(curated / "splits"),
                                 seed=1, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)
    assert "error" not in splits_result, splits_result

    zip_path = tmp_path / "bundle.zip"
    archived = archive_project(project, str(zip_path))
    assert "error" not in archived, archived

    dest = tmp_path / "dest"
    imported = import_project(str(zip_path), str(dest))
    assert "error" not in imported, imported
    assert ts.read(selection_key(dest / "curated" / "splits"))["scope"]["subject"] == "bud"


def test_the_full_round_trip_reads_back_at_once_with_no_hand_adoption(tmp_path, monkeypatch):
    """initialize_project, register_dataset, a confirmed trait revision, a completed run's
    directory, an HPO sweep's directory and a project-relative splits manifest, all through their
    own real producers; archived, imported into a fresh destination, and read back at once. The
    sweep's trial body is stood in for by
    one that only resolves and opens the trial's run directory through the launcher's own
    producer and writer."""
    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp import experiments
    from tcip_mcp.pipelines.data.selection import selection_key
    from tcip_mcp.pipelines.data.split_construction import resolve_run
    from tcip_mcp.traits import read_trait
    from tcip_mcp.tools.data_tools import draw_splits
    from tcip_mcp.tools.project_tools import (
        dataset_entry_path, initialize_project, read_datasets, register_dataset,
    )

    from tests._trait_fixtures import COUNT_TRAIT, seed_confirmed_count

    root = tmp_path / "source"
    _annotated_dataset(root, 4)
    assert "error" not in initialize_project(str(root), "North orchard", site="north orchard")
    assert "error" not in register_dataset(root, str(root), crop="currant")
    confirmed = seed_confirmed_count(root)
    assert confirmed.confirmed

    from tests._verified_checkpoint_fixtures import finished_run

    finished_run(root, experiment_id="exp1", rows=[{"epoch": 1, "loss": 0.5}])

    def fake_trial(point, report, base_config, trial_dir, *, project, objective):
        config = tt._apply_hpo_params(base_config, point)
        tt.open_run(trial_dir, config,
                    resolve_run(config, project=project, objective=objective).record,
                    trial_params=point)
        report(0.2)

    def fake_search(**kw):
        kw["objective_fn"]({"lr": 0.1}, lambda value: None)
        return str(Path(kw["storage_path"]) / kw["study_name"])

    monkeypatch.setattr(tt, "_run_hpo_trial", fake_trial)
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)
    hpo_result = tt.run_hyperparameter_search(
        root, base_config={"model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                                      "task": "detection"},
                     "data": {"images_dir": str(root / "images" / "2026-03-04"),
                              "scope": {"subject": "bud"}, "split": {"seed": 0, "val_ratio": 0.15}}},
        n_trials=1, search_seed=0
    )
    study = hpo_result["study_name"]

    splits_result = draw_splits(root, str(root), output_path=str(root / "splits_out"), subject="bud",
                                seed=1, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)
    assert "error" not in splits_result, splits_result

    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(root, str(zip_path))

    dest = tmp_path / "restored"
    imported = import_project(str(zip_path), str(dest))
    assert "error" not in imported

    from tcip_mcp.project_record import read_record

    assert read_record(str(dest))["site"] == "north orchard"

    entries = read_datasets(dest)
    assert entries[0]["path"] == "."
    assert dataset_entry_path(dest, entries[0]).resolve() == dest.resolve()

    assert read_trait(COUNT_TRAIT, dest).latest_confirmed == confirmed

    run_dir = experiments.find_run("exp1", project=dest)
    assert run_dir is not None
    observation = experiments.observe(run_dir)
    assert observation.record["config"]["model_source"]["task"] == "detection"
    rows = experiments.read_rows(run_dir / experiments.METRICS_FILE)[0]
    assert rows[0]["loss"] == 0.5
    assert observation.checkpoint is not None

    sweep = tt.read_sweep(tt.sweep_observation(study, project=dest))
    assert sweep["status"] == "completed"
    assert [t["params"] for t in sweep["trials"]] == [{"lr": 0.1}]

    selection = ts.read(selection_key(dest / "splits_out"))
    assert selection["scope"]["subject"] == "bud"
