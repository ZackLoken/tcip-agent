"""The import door's own rails: staging, its refusals and the move into place.

Staging, the refusal of the store's own bookkeeping and of an escaping member, and the move.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import tcip_store as ts
from tcip_mcp.tools.project_tools import (
    ZipBundle,
    archive_project,
    import_project,
)
from tcip_mcp.web_client import gui_snapshot_key
from tcip_store.file_backend import lock_file_for
from tests._chain_fixtures import BESPOKE_DETECTION, FRAME_STATISTICS
from tests._producer_fixtures import one_labeled_capture
from tests._training_values import drawn_split

if TYPE_CHECKING:
    from collections.abc import Callable


def _hand_zip(path: Path, members: dict[str, bytes]) -> Path:
    """Write a hand-built zip carrying exactly ``members`` and return its path."""
    with zipfile.ZipFile(str(path), "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


# ── rail 2: a non-empty destination refuses and changes nothing ────────────────────────────


def test_import_refuses_a_non_empty_destination_and_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = one_labeled_capture(tmp_path / "source")
    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(
        root, ZipBundle(output_path=str(zip_path))
    )

    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "subjects.json").write_bytes(b"already here")
    ts.replace(
        gui_snapshot_key(dest),
        {"active_subject": "x"},
        expect=ts.Version.ABSENT,
    )

    result = import_project(str(zip_path), str(dest))

    assert "error" in result
    assert str(dest) in result["error"]
    assert (dest / "subjects.json").read_bytes() == b"already here"
    assert ts.read(gui_snapshot_key(dest)) == {"active_subject": "x"}


def test_import_admits_a_pre_existing_empty_destination(
    tmp_path: Path,
) -> None:
    root = one_labeled_capture(tmp_path / "source")
    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(
        root, ZipBundle(output_path=str(zip_path))
    )

    dest = tmp_path / "dest"
    dest.mkdir()

    result = import_project(str(zip_path), str(dest))

    assert "error" not in result
    assert (dest / "subjects.json").is_file()


# ── rail 5: a database's sidecar, or its lock, refuses ──────────────────────────────────────


def test_import_refuses_a_member_named_as_a_database_sidecar(
    tmp_path: Path,
) -> None:
    zip_path = _hand_zip(
        tmp_path / "bundle.zip", {".tcip/store.db-wal": b"not a log"}
    )
    dest = tmp_path / "dest"

    result = import_project(str(zip_path), str(dest))

    assert "error" in result
    assert "store.db-wal" in result["error"]
    assert not dest.exists()


def test_import_refuses_a_member_named_store_db_lock(tmp_path: Path) -> None:
    zip_path = _hand_zip(tmp_path / "bundle.zip", {".tcip/store.db.lock": b""})
    dest = tmp_path / "dest"

    result = import_project(str(zip_path), str(dest))

    assert "error" in result
    assert not dest.exists()


# ── rail 8: zip-slip refusal ─────────────────────────────────────────────────────────────────


def test_import_refuses_a_zip_slip_path(tmp_path: Path) -> None:
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(str(zip_path), "w") as zf:
        zf.writestr("../../evil.txt", "escaped")
    dest = tmp_path / "dest"

    result = import_project(str(zip_path), str(dest))

    assert "error" in result
    assert "Unsafe path" in result["error"]
    assert not dest.exists()


def test_extract_zip_refuses_a_sibling_directory_that_shares_stagings_name_as_a_prefix(
    tmp_path: Path,
) -> None:
    """Zip extraction refuses a sibling directory sharing staging's name as a prefix.

    A member resolving to ``<staging.name>extra/`` lies outside staging and refuses as an unsafe
    path.
    """
    from tcip_mcp.tools.project_tools import _extract_zip

    staging = tmp_path / "abcd1234"
    staging.mkdir()
    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(str(evil_zip), "w") as zf:
        zf.writestr(f"../{staging.name}extra/evil.txt", "escaped")

    with pytest.raises(ValueError, match="Unsafe path"):
        _extract_zip(evil_zip, staging)


def test_import_refuses_a_corrupt_zip_without_stranding_a_staging_tree(
    tmp_path: Path,
) -> None:
    root = one_labeled_capture(tmp_path / "source")
    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(
        root, ZipBundle(output_path=str(zip_path))
    )
    data = zip_path.read_bytes()
    zip_path.write_bytes(
        data[: len(data) // 2]
    )  # truncated: no longer a readable zip

    dest = tmp_path / "dest"
    result = import_project(str(zip_path), str(dest))

    assert "error" in result
    assert not dest.exists()
    imports_root = dest.parent / ".imports"
    assert not imports_root.is_dir() or not any(imports_root.iterdir())


# ── rail 10: a record the database holds travels in its database ────────────────────────────


def test_a_record_travels_inside_the_database_the_archive_copies(
    tmp_path: Path,
) -> None:
    record = {"active_subject": "x"}
    root = one_labeled_capture(tmp_path / "source")
    ts.replace(gui_snapshot_key(root), record, expect=ts.Version.ABSENT)
    assert "error" not in archive_project(
        root, ZipBundle(output_path=str(tmp_path / "bundle.zip"))
    )

    dest = tmp_path / "dest"
    result = import_project(str(tmp_path / "bundle.zip"), str(dest))

    assert "error" not in result
    assert (dest / ".tcip" / "store.db").is_file()
    assert ts.read(gui_snapshot_key(dest)) == record


def test_the_database_the_archive_copies_is_already_in_wal(
    tmp_path: Path,
) -> None:
    """The store database a bundle carries reports WAL before anything opens it to read or write.

    The member is extracted as it lies in the bundle and its mode read without changing it.
    """
    from tests.test_store_contract import _journal_mode

    root = one_labeled_capture(tmp_path / "source")
    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(
        root, ZipBundle(output_path=str(zip_path))
    )
    with zipfile.ZipFile(str(zip_path)) as zf:
        extracted = Path(zf.extract(".tcip/store.db", tmp_path / "member"))

    assert _journal_mode(extracted) == "wal"


# ── rail 13: concurrency ─────────────────────────────────────────────────────────────────────


def test_a_locked_staging_sibling_is_left_alone_while_another_import_completes(
    tmp_path: Path,
) -> None:
    import filelock

    root = one_labeled_capture(tmp_path / "source")
    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(
        root, ZipBundle(output_path=str(zip_path))
    )

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


def test_a_free_locked_leftover_staging_sibling_is_swept_with_its_lock_file(
    tmp_path: Path,
) -> None:
    root = one_labeled_capture(tmp_path / "source")
    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(
        root, ZipBundle(output_path=str(zip_path))
    )

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


def test_dataset_registry_stores_the_relative_dot_after_import(
    tmp_path: Path,
) -> None:
    """The project's own dataset entry stores the relative ``"."`` after import.

    The project registered as its own dataset reads back after the archive/import round trip
    with ``path`` ``"."`` and no unresolved dataset path.
    """
    from tcip_mcp.tools.project_tools import read_datasets, register_dataset

    root = one_labeled_capture(tmp_path / "source")
    registered = register_dataset(root, str(root), crop="currant", actor=None)
    assert "error" not in registered

    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(
        root, ZipBundle(output_path=str(zip_path))
    )
    dest = tmp_path / "dest"
    imported = import_project(str(zip_path), str(dest))

    assert "error" not in imported
    entries = read_datasets(dest)
    assert entries[0].path == "."
    assert imported["dataset_paths_unresolved"] == []


def test_dataset_registry_travels_with_nothing_rewritten(
    tmp_path: Path,
) -> None:
    """The project's own dataset entry resolves to the import destination after an import."""
    from tcip_mcp.tools.project_tools import (
        dataset_entry_path,
        read_datasets,
        register_dataset,
    )

    root = one_labeled_capture(tmp_path / "source")
    registered = register_dataset(root, str(root), crop="currant", actor=None)
    assert "error" not in registered

    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(
        root, ZipBundle(output_path=str(zip_path))
    )
    dest = tmp_path / "dest"
    imported = import_project(str(zip_path), str(dest))

    assert "error" not in imported
    entries = read_datasets(dest)
    assert dataset_entry_path(dest, entries[0]).resolve() == dest.resolve()


def test_an_external_dataset_entry_stays_absolute_and_is_disclosed(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.project_tools import register_dataset

    root = one_labeled_capture(tmp_path / "source")
    external = one_labeled_capture(tmp_path / "external_dataset")
    registered = register_dataset(
        root, str(external), crop="currant", actor=None
    )
    assert "error" not in registered

    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(
        root, ZipBundle(output_path=str(zip_path))
    )
    dest = tmp_path / "dest"
    imported = import_project(str(zip_path), str(dest))

    assert "error" not in imported
    assert str(external.resolve()) in imported["dataset_paths_unresolved"]


# ── rail 1 / 6 / 9 / 14: the full round trip through real producers ─────────────────────────


def _annotated_dataset(root: Path, n: int) -> None:
    """Write ``n`` 640 x 480 images, each under its own stem, on one date, and a ``bud`` registry.

    Each image holds one bright ``bud`` box that its label document names.
    """
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import Subject, SubjectRegistry
    from tests._producer_fixtures import (
        BRIGHT,
        label_image,
        painted_frame,
        registry_over,
    )

    images_dir = root / "images" / "2026-03-04"
    images_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        stem = f"img_{i:03d}"
        painted_frame(
            640, 480, (128, 128, 128), [((288, 216, 352, 264), BRIGHT)]
        ).save(images_dir / f"{stem}.jpg")
        label_image(
            images_dir / f"{stem}.jpg",
            [Annotation(subject="bud", geometry=BBox(288, 216, 352, 264))],
            640,
            480,
        )
    registry_over(
        root, SubjectRegistry(subjects=(Subject(name="bud", description=""),))
    )


def test_a_splits_root_nested_under_a_nested_dataset_root_archives_and_round_trips(
    tmp_path: Path,
) -> None:
    """A splits root nested under a nested dataset root archives and round-trips.

    A dataset under a project, partitioned in place by draw_splits with selection.json under the
    dataset root, archives, imports, and reads its selection back.
    """
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.tools.data_tools import draw_splits

    project = tmp_path / "project"
    curated = project / "curated"
    _annotated_dataset(curated, 4)

    splits_result = draw_splits(
        project,
        str(curated),
        subject="bud",
        output_path=str(curated / "splits"),
        **drawn_split(
            seed=1,
            val_ratio=0.25,
            calibration_ratio=0.125,
            holdout_ratio=0.125,
            grouping="tile_prefix",
        ),
        actor=None,
    )
    assert "error" not in splits_result, splits_result

    zip_path = tmp_path / "bundle.zip"
    archived = archive_project(project, ZipBundle(output_path=str(zip_path)))
    assert "error" not in archived, archived

    dest = tmp_path / "dest"
    imported = import_project(str(zip_path), str(dest))
    assert "error" not in imported, imported
    assert (
        read_selection(dest / "curated" / "splits", project=dest).scope.subject
        == "bud"
    )


def test_the_full_round_trip_reads_back_at_once_with_no_hand_adoption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full round trip through real producers reads back with no hand adoption.

    A project record, a registered dataset, a confirmed trait revision, a completed run's
    directory, an HPO sweep's directory and a project-relative splits manifest, each written by
    its own producer, are archived, imported into a fresh destination and read back with no
    step between. The sweep's trial body is a stand-in that resolves and opens the trial's run
    directory through the launcher's own producer and writer, and trains nothing.
    """
    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp import experiments
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.tools.data_tools import draw_splits
    from tcip_mcp.tools.project_tools import (
        dataset_entry_path,
        initialize_project,
        read_datasets,
        register_dataset,
    )
    from tcip_mcp.traits import read_trait
    from tests._trait_fixtures import COUNT_TRAIT, seed_confirmed_count
    from tests._web_fixtures import INVARIANCES

    root = tmp_path / "source"
    _annotated_dataset(root, 4)
    assert "error" not in initialize_project(
        str(root),
        "North orchard",
        site="north orchard",
        invariances=INVARIANCES,
        actor=None,
    )
    assert "error" not in register_dataset(
        root, str(root), crop="currant", actor=None
    )
    confirmed = seed_confirmed_count(root)
    assert confirmed.confirmed

    from tests._verified_checkpoint_fixtures import finished_run

    finished_run(root, experiment_id="exp1", rows=[{"epoch": 1, "loss": 0.5}])

    def fake_trial(
        point: dict,
        report: Callable[[float], None],
        sweep: Path,
        trial_id: str,
    ) -> None:
        tt.open_trial(sweep, trial_id, point)
        report(0.2)

    def fake_search(
        **kw: Callable[[dict, Callable[[float], None]], object]
        | dict
        | str
        | int
        | Path
        | None,
    ) -> None:
        kw["objective_fn"]({"optimizer.head_lr": 0.1}, lambda value: None)

    from tests._chain_fixtures import BESPOKE_MODELS, training_config
    from tests._training_values import fifo_search, sweep_space

    monkeypatch.setattr(tt, "_run_hpo_trial", fake_trial)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", fake_search
    )
    hpo_result = tt.run_hyperparameter_search(
        root,
        base_config=training_config(
            root,
            {
                "builder": BESPOKE_DETECTION,
                "source_files": [BESPOKE_MODELS],
                "task": "detection",
                "normalization": FRAME_STATISTICS,
            },
            {
                "images_dir": str(root / "images" / "2026-03-04"),
                "scope": {"subject": "bud"},
                "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
            },
        ),
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        search_seed=0,
        actor=None,
    )
    sweep_id = hpo_result["sweep"]["sweep_id"]

    splits_result = draw_splits(
        root,
        str(root),
        output_path=str(root / "splits_out"),
        subject="bud",
        **drawn_split(
            seed=1,
            val_ratio=0.25,
            calibration_ratio=0.125,
            holdout_ratio=0.125,
            grouping="tile_prefix",
        ),
        actor=None,
    )
    assert "error" not in splits_result, splits_result

    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(
        root, ZipBundle(output_path=str(zip_path))
    )

    dest = tmp_path / "restored"
    imported = import_project(str(zip_path), str(dest))
    assert "error" not in imported

    from tcip_mcp.project_record import read_record

    assert read_record(str(dest)).site == "north orchard"

    entries = read_datasets(dest)
    assert entries[0].path == "."
    assert dataset_entry_path(dest, entries[0]).resolve() == dest.resolve()

    assert read_trait(COUNT_TRAIT, dest).latest_confirmed == confirmed

    run_dir = experiments.find_run("exp1", project=dest)
    assert run_dir is not None
    observation = experiments.observe_run(run_dir)
    assert observation.record.spec.model_source.task == "detection"
    rows = experiments.read_rows(run_dir / experiments.METRICS_FILE)[0]
    assert rows[0]["loss"] == 0.5
    assert observation.checkpoint is not None

    sweep = tt.monitor_training(dest, sweep_id)["sweep"]
    assert sweep["state"] == "completed"
    assert [t["trial_params"] for t in sweep["trials"]] == [
        {"optimizer.head_lr": 0.1}
    ]

    selection = read_selection(dest / "splits_out", project=dest)
    assert selection.scope.subject == "bud"
