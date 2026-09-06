"""The import door's own rails: staging, the shared accounting's refusals, the backend-
conditional adoption, and the move. Each refusal here is paired with what still admits, per
CLAUDE.md's rail rule.
"""

from __future__ import annotations

import zipfile
from contextlib import contextmanager
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp import dataset_layout
from tcip_mcp.tools.project_tools import archive_project, import_project
from tcip_store.file_backend import FileBackend, lock_file_for
from tcip_store.sqlite_backend import SqliteBackend


@contextmanager
def bound(backend):
    """Bind one backend for a block, putting the suite's own backend back on the way out."""
    from tcip_store.store import _backend

    previous = _backend()
    ts.bind(backend)
    try:
        yield backend
    finally:
        ts.bind(previous)
        backend.close()


def _project(root: Path) -> Path:
    """A dataset root with one image, one empty label, and the registry that decodes it."""
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import ClassRegistry, Subject

    (root / "images" / "2026-03-04").mkdir(parents=True, exist_ok=True)
    (root / "images" / "2026-03-04" / "a_1.jpg").write_bytes(b"\xff\xd8\xff")
    (root / "annotations" / "2026-03-04").mkdir(parents=True, exist_ok=True)
    (root / "annotations" / "2026-03-04" / "a_1.json").write_text(
        '{"annotations": []}', encoding="utf-8"
    )
    class_registry.write_registry(
        root / "classes.json", ClassRegistry(subjects=(Subject(name="bud"),))
    )
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
    assert "error" not in archive_project(str(root), str(zip_path))

    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "classes.json").write_bytes(b"already here")
    # A scratch platform root for import_project's own audit entry, off tmp_path.
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "scratch_platform_root"))
    with bound(SqliteBackend()):
        ts.replace(
            dataset_layout.image_status_key(dest),
            {"bud/2026-03-04": {"a_1.jpg": {"status": "negative", "by": "user:x"}}},
            expect=ts.Version.ABSENT,
        )

        result = import_project(str(zip_path), str(dest))

        assert "error" in result
        assert str(dest) in result["error"]
        assert (dest / "classes.json").read_bytes() == b"already here"
        assert ts.read(dataset_layout.image_status_key(dest)) == {
            "bud/2026-03-04": {"a_1.jpg": {"status": "negative", "by": "user:x"}}
        }


def test_import_admits_a_pre_existing_empty_destination(tmp_path):
    root = _project(tmp_path / "source")
    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(str(root), str(zip_path))

    dest = tmp_path / "dest"
    dest.mkdir()

    result = import_project(str(zip_path), str(dest))

    assert "error" not in result
    assert (dest / "classes.json").is_file()


# ── rail 3: an unaccounted member refuses the whole import, naming it ──────────────────────


def test_import_refuses_an_unaccounted_member_naming_it(tmp_path):
    root = _project(tmp_path / "source")
    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(str(root), str(zip_path))
    with zipfile.ZipFile(str(zip_path), "a") as zf:
        zf.writestr(".tcip/state/_write_probe.txt", "probe")

    dest = tmp_path / "dest"
    result = import_project(str(zip_path), str(dest))

    assert "error" in result
    assert "_write_probe.txt" in result["error"]
    assert not dest.exists()


# ── rail 4: an undecodable claimed member refuses naming the file and the store ─────────────


def test_import_refuses_an_undecodable_claimed_member_on_both_backends(tmp_path, monkeypatch):
    zip_path = _hand_zip(tmp_path / "bundle.zip", {".tcip/project.json": b"{not valid json"})

    # import_project's own audit entry lands at the platform root; each stage gets its own so
    # one backend's write there never collides with the other's.
    dest = tmp_path / "dest_sqlite"
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "scratch_sqlite"))
    with bound(SqliteBackend()):
        result = import_project(str(zip_path), str(dest))
    assert "error" in result
    assert "project.json" in result["error"]
    assert not dest.exists()

    dest2 = tmp_path / "dest_file"
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "scratch_file"))
    with bound(FileBackend()):
        result2 = import_project(str(zip_path), str(dest2))
    assert "error" in result2
    assert "project.json" in result2["error"]
    assert not dest2.exists()


# ── rail 5: a member named store.db, or its lock, refuses ───────────────────────────────────


def test_import_refuses_a_member_named_store_db(tmp_path):
    zip_path = _hand_zip(tmp_path / "bundle.zip", {".tcip/store.db": b"not a database"})
    dest = tmp_path / "dest"

    result = import_project(str(zip_path), str(dest))

    assert "error" in result
    assert "store.db" in result["error"]
    assert not dest.exists()


def test_import_refuses_a_member_named_store_db_lock(tmp_path):
    zip_path = _hand_zip(tmp_path / "bundle.zip", {".tcip/store.db.lock": b""})
    dest = tmp_path / "dest"

    result = import_project(str(zip_path), str(dest))

    assert "error" in result
    assert not dest.exists()


# ── rail 12: cross-root collision ────────────────────────────────────────────────────────────

_COLLISION_PROBE_STORE = "rail_collision_probe"
_collision_probe_registered = False


_COLLISION_PROBE_RUN_ID = "rail_collision_probe_run"
"""A literal run id no other test's experiment ever uses, so this probe's claim collides only
with a zip built to trigger it, never with an ordinary experiment created anywhere else in the
same pytest session."""


def _register_collision_probe() -> None:
    """A test-only store whose declared claim collides with the shipped EXPERIMENTS claim: a
    ROOT-layout template that spells out .tcip/experiments/<the probe run id>/config.json in
    full, the exact path the shipped experiment_config claim already owns one directory level
    in. Registered once, process-wide, so the admitting test above this in file order must run
    first.
    """
    global _collision_probe_registered
    if _collision_probe_registered:
        return
    _collision_probe_registered = True
    from tcip_store.file_backend import RootedFileLocator
    from tcip_store.layout_claims import ROOT, Claim, Constant, Patterned, literal

    ts.register_store(
        ts.StoreDescriptor(
            name=_COLLISION_PROBE_STORE,
            kind="record",
            key_fields=("run_id", "document"),
            codec=ts.RECORD_JSON,
            concurrency="last_writer_wins",
            locator=RootedFileLocator(prefix=(".tcip", "experiments"), suffix=".json"),
            claim=Claim(
                ROOT,
                ((Constant(".tcip"), Constant("experiments"), Constant(_COLLISION_PROBE_RUN_ID),
                  Patterned(literal("config"), tail=".json")),),
            ),
        )
    )


def test_import_admits_a_config_json_before_any_collision_is_registered(tmp_path):
    """The admitting side: run before test_import_refuses_a_runtime_registered_claim_collision
    registers the colliding claim, since register_store cannot be undone within a process."""
    member = f".tcip/experiments/{_COLLISION_PROBE_RUN_ID}/config.json"
    zip_path = _hand_zip(tmp_path / "bundle.zip", {member: b"{}"})
    dest = tmp_path / "dest"

    result = import_project(str(zip_path), str(dest))

    assert "error" not in result
    assert (dest / ".tcip" / "experiments" / _COLLISION_PROBE_RUN_ID / "config.json").is_file()


def test_import_refuses_a_runtime_registered_claim_collision_by_name(tmp_path):
    _register_collision_probe()
    member = f".tcip/experiments/{_COLLISION_PROBE_RUN_ID}/config.json"
    zip_path = _hand_zip(tmp_path / "bundle.zip", {member: b"{}"})
    dest = tmp_path / "dest"

    result = import_project(str(zip_path), str(dest))

    assert "error" in result
    assert "config.json" in result["error"]
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
    assert "error" not in archive_project(str(root), str(zip_path))
    data = zip_path.read_bytes()
    zip_path.write_bytes(data[: len(data) // 2])  # truncated: no longer a readable zip

    dest = tmp_path / "dest"
    result = import_project(str(zip_path), str(dest))

    assert "error" in result
    assert not dest.exists()
    imports_root = dest.parent / ".imports"
    assert not imports_root.is_dir() or not any(imports_root.iterdir())


# ── rail 10: the file-backend leg ───────────────────────────────────────────────────────────


def test_import_under_file_backend_lands_files_and_builds_no_database(tmp_path):
    root = _project(tmp_path / "source")
    negative = {"bud/2026-03-04": {"a_1.jpg": {"status": "negative", "by": "user:x"}}}
    with bound(FileBackend()):
        ts.replace(dataset_layout.image_status_key(root), negative, expect=ts.Version.ABSENT)
        assert "error" not in archive_project(str(root), str(tmp_path / "bundle.zip"))

    dest = tmp_path / "dest"
    with bound(FileBackend()):
        result = import_project(str(tmp_path / "bundle.zip"), str(dest))

        assert "error" not in result
        assert result["database_built"] is False
        assert not (dest / ".tcip" / "store.db").exists()
        assert ts.read(dataset_layout.image_status_key(dest)) == negative


# ── rail 13: concurrency ─────────────────────────────────────────────────────────────────────


def test_a_locked_staging_sibling_is_left_alone_while_another_import_completes(tmp_path):
    import filelock

    root = _project(tmp_path / "source")
    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(str(root), str(zip_path))

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
    assert "error" not in archive_project(str(root), str(zip_path))

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
    in before the move. Bound to the file backend throughout (rather than the ambient default),
    so this reads what the door itself wrote on either side of the row that introduced the
    relative form, with no unrelated database-conform refusal in between."""
    from tcip_mcp.tools.project_tools import read_datasets, register_dataset

    with bound(FileBackend()):
        root = _project(tmp_path / "source")
        registered = register_dataset(str(root), crop="currant", project_root=str(root))
        assert "error" not in registered

        zip_path = tmp_path / "bundle.zip"
        assert "error" not in archive_project(str(root), str(zip_path))
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
    registered = register_dataset(str(root), crop="currant", project_root=str(root))
    assert "error" not in registered

    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(str(root), str(zip_path))
    dest = tmp_path / "dest"
    imported = import_project(str(zip_path), str(dest))

    assert "error" not in imported
    entries = read_datasets(dest)
    assert dataset_entry_path(dest, entries[0]).resolve() == dest.resolve()


def test_an_external_dataset_entry_stays_absolute_and_is_disclosed(tmp_path):
    from tcip_mcp.tools.project_tools import register_dataset

    root = _project(tmp_path / "source")
    external = _project(tmp_path / "external_dataset")
    registered = register_dataset(str(external), crop="currant", project_root=str(root))
    assert "error" not in registered

    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(str(root), str(zip_path))
    dest = tmp_path / "dest"
    imported = import_project(str(zip_path), str(dest))

    assert "error" not in imported
    assert str(external.resolve()) in imported["dataset_paths_unresolved"]


# ── rail 1 / 6 / 9 / 14: the full round trip through real producers ─────────────────────────


def _annotated_dataset(root: Path, n: int) -> None:
    """``n`` distinct single-tile foreground groups of one subject, enough to clear
    draw_splits' floor (one group each for train/val, two for calibration)."""
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import ClassRegistry, Subject

    images_dir = root / "images" / "2026-03-04"
    labels_dir = root / "annotations" / "2026-03-04"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        stem = f"img_{i:03d}"
        Image.new("RGB", (640, 480), color=(128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="bud", geometry=BBox(288, 216, 352, 264))], 640, 480,
        )
    class_registry.write_registry(
        root / "classes.json", ClassRegistry(subjects=(Subject(name="bud"),))
    )


def test_a_splits_root_nested_under_a_curated_root_archives_and_round_trips(tmp_path):
    """The producer chain the skills document: a curated dataset (materialize_review_dataset's
    own output shape, a curated_manifest.json at its root) sits under a project, and draw_splits
    partitions it in place with no output_path, landing split_manifest.json under the curated
    root rather than beside it. The cross-anchor constraint must admit that nesting rather than
    refusing the whole project."""
    from tcip_mcp.pipelines.feedback.materialize import curated_manifest_key
    from tcip_mcp.tools.data_tools import draw_splits, split_manifest_key

    project = tmp_path / "project"
    curated = project / "curated"
    _annotated_dataset(curated, 4)
    ts.replace(curated_manifest_key(curated), {"source": "review verdicts"}, expect=ts.Version.ABSENT)

    splits_result = draw_splits(str(curated), subject="bud", materialize=True,
                                 train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in splits_result, splits_result

    zip_path = tmp_path / "bundle.zip"
    archived = archive_project(str(project), str(zip_path))
    assert "error" not in archived, archived

    dest = tmp_path / "dest"
    imported = import_project(str(zip_path), str(dest))
    assert "error" not in imported, imported
    assert (dest / "curated" / "curated_manifest.json").is_file()
    assert ts.read(split_manifest_key(dest / "curated" / "splits"))["subject"] == "bud"


def test_the_full_round_trip_reads_back_at_once_with_no_hand_adoption(tmp_path, monkeypatch):
    """initialize_project, register_dataset, a confirmed operationalization, an experiment's members,
    an HPO sweep's members and a project-relative splits manifest, all through their own real
    producers; archived, imported into a fresh destination, and read back through the store
    under the default backend with no ``tcip adopt-store`` run.

    launch_config.json is hand-placed (launch_training itself needs a real training launch to
    produce one), stated as such; every other member here comes from the producer that writes it.
    """
    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp.experiments import config_key, create_experiment, log_metrics, metrics_key, status_key
    from tcip_mcp.operationalization import PER_IMAGE_COUNT, resolve_trait_and_record
    from tcip_mcp.tools.data_tools import draw_splits, split_manifest_key
    from tcip_mcp.tools.project_tools import (
        dataset_entry_path, initialize_project, read_datasets, register_dataset,
    )

    from tests._operationalization_fixtures import COUNT_TRAIT, seed_confirmed_count

    root = tmp_path / "source"
    monkeypatch.setenv("TCIP_STATE_ROOT", str(root))
    _annotated_dataset(root, 4)
    assert "error" not in initialize_project(str(root), site="north orchard")
    assert "error" not in register_dataset(str(root), crop="currant", project_root=str(root))
    confirmed = seed_confirmed_count(root)
    assert confirmed["confirmed_by"]

    create_experiment("exp1", {"trait": COUNT_TRAIT})
    log_metrics("exp1", 1, {"loss": 0.5})
    launch_config_path = root / ".tcip" / "experiments" / "exp1" / "launch_config.json"
    launch_config_path.parent.mkdir(parents=True, exist_ok=True)
    launch_config_path.write_text('{"data": {}}', encoding="utf-8")

    def fake_trial(config, report, base_config, trial_dir):
        trial_path = Path(trial_dir)
        sweep_root, name = trial_path.parent, trial_path.name
        ts.replace(tt.trial_config_key(sweep_root, name), config, expect=ts.Version.ABSENT)
        ts.append(tt.trial_metrics_key(sweep_root, name), {"epoch": 1, "loss": 0.2})
        report(0.2)

    def fake_search(**kw):
        kw["objective_fn"]({"lr": 0.1}, lambda value: None)
        return {"best_params": {"lr": 0.1}, "best_value": 0.2, "n_trials": 1,
                "study_name": kw["study_name"]}

    monkeypatch.setattr(tt, "_run_hpo_trial", fake_trial)
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)
    hpo_result = tt.run_hyperparameter_search(
        base_config={"model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                                      "builder_kwargs": {"num_classes": 1}, "task": "detection"},
                     "data": {"images_dir": str(root / "images"), "labels_dir": str(root / "annotations")}},
        n_trials=1,
    )
    study = hpo_result["study_name"]

    splits_result = draw_splits(str(root), output_path=str(root / "splits_out"), subject="bud",
                                train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in splits_result, splits_result

    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(str(root), str(zip_path))

    dest = tmp_path / "restored"
    # A scratch platform root for import_project's own audit entry, off root (whose own
    # conform state depends on whichever backend the suite happens to run this file on).
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "scratch_platform_root"))
    with bound(SqliteBackend()):
        imported = import_project(str(zip_path), str(dest))
        assert "error" not in imported
        assert imported["database_built"] is True

        from tcip_mcp.project_record import read_record

        assert read_record(str(dest))["site"] == "north orchard"

        entries = read_datasets(dest)
        assert entries[0]["path"] == "."
        assert dataset_entry_path(dest, entries[0]).resolve() == dest.resolve()

        _, record, _ = resolve_trait_and_record(COUNT_TRAIT, PER_IMAGE_COUNT, project_root=dest)
        assert record.value["confirmed_by"] == confirmed["confirmed_by"]
        assert record.value["confirmed_at"] == confirmed["confirmed_at"]
        assert record.value["confirmed_fields"] == confirmed["confirmed_fields"]

        assert ts.read(config_key("exp1", root=dest))["trait"] == COUNT_TRAIT
        assert ts.read(status_key("exp1", root=dest))["metrics_logged"] is True
        assert ts.read_log(metrics_key("exp1", root=dest)).records[0]["loss"] == 0.5
        assert (dest / ".tcip" / "experiments" / "exp1" / "launch_config.json").is_file()

        assert ts.read(tt.study_result_key(study, str(dest / ".tcip" / "hpo")))["best_params"] == {
            "lr": 0.1
        }
        assert ts.read(tt.sweep_manifest_key(study, str(dest / ".tcip" / "hpo")))["status"] == \
            "completed"
        trial_dirs = list((dest / ".tcip" / "hpo" / study).glob("trial_*"))
        assert trial_dirs
        assert ts.read(tt.trial_config_key(dest / ".tcip" / "hpo" / study, trial_dirs[0].name))

        manifest = ts.read(split_manifest_key(dest / "splits_out"))
        assert manifest["subject"] == "bud"
