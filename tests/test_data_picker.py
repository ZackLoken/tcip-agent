"""The data picker: choosing a recorded partition (or "As recorded") for a relaunch.

Covers the pieces the design record's second family adds: the shared
``manifest_compatibility`` check, the extracted ``narrow_manifest_to_date``, the checked manifest
reader, ``list_split_choices`` and its route, and the launch route's ``split_manifest_dir`` field.
The manifest-binding mechanics themselves (``bind_manifest_stems``, ``auto_train_val``'s manifest
branch, ``read_split_manifest_dir``) are ``test_split_manifest_binding.py``'s; this file reuses
its dataset fixture rather than restating it.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import tcip_store as ts
from tcip_web.app import app

from tests.test_split_manifest_binding import DATES, OTHER_SUBJECT, SUBJECT, _draw, \
    _two_subject_two_date_dataset


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _bespoke_config(images_dir: Path, labels_dir: Path, *, subject: str = SUBJECT) -> dict:
    return {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1, "min_size": 64, "max_size": 64},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                 "subject": subject},
        "training": {"batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                     "mixed_precision": False, "device": "cpu",
                     "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False}},
    }


# -- narrow_manifest_to_date ---------------------------------------------------


def test_narrow_manifest_to_date_matches_bind_manifest_stems_own_arithmetic(tmp_path: Path):
    """The extracted narrowing and the binding it now feeds must agree: this is the same
    arithmetic the picker shows before a launch and the launch itself applies after one."""
    from tcip_mcp.pipelines.data.splits import bind_manifest_stems, narrow_manifest_to_date

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    manifest = _draw(root, tmp_path / "m")

    narrowing = narrow_manifest_to_date(manifest, DATES[0])
    binding = bind_manifest_stems(
        manifest, DATES[0], SUBJECT, None, ["a", "b", "c", "d", "e", "f"],
        images_dir=root / "images" / DATES[0])

    assert len(narrowing.train_ids) == binding.train_bound
    assert len(narrowing.val_ids) == binding.val_bound
    assert len(narrowing.calibration_ids) == binding.calibration_bound
    assert narrowing.other_dates == binding.other_dates


# -- manifest_compatibility and the new absent-images-root refusal --------------


def _manifest_missing_images_root(root: Path, out: Path, date: str) -> dict:
    """A real ``draw_splits`` draw, hand-mutated to drop one date's ``images_root``: the shape a
    hand-edited or pre-images_root manifest would read as. Every other required field stays
    exactly what ``draw_splits`` wrote, so this exercises only the one field under test."""
    from tcip_mcp.pipelines.data.splits import manifest_date_key
    from tcip_mcp.tools.data_tools import split_manifest_key

    manifest = _draw(root, out)
    date_key = manifest_date_key(date)
    manifest["members"][date_key] = {
        k: v for k, v in manifest["members"][date_key].items() if k != "images_root"
    }
    ts.replace(split_manifest_key(out), manifest)
    return manifest


def test_manifest_compatibility_flags_a_members_block_with_no_images_root(tmp_path: Path):
    from tcip_mcp.tools.training_tools import manifest_compatibility

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _manifest_missing_images_root(root, out, DATES[0])
    config = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    config["data"]["split"] = {"manifest_dir": str(out)}

    issues = manifest_compatibility(config, manifest, str(out))

    assert any("images root" in i for i in issues)


def test_manifest_compatibility_reports_a_subject_mismatch_and_a_date_disagreement_together(
    tmp_path: Path,
):
    """A config wrong on two independent fronts (its subject disagrees with the manifest's own,
    and its own ``data.date`` disagrees with the date its ``data.labels_dir`` is under) reports
    both, not only the date one: the accumulator runs before the date check is appended, so
    neither issue hides the other."""
    from tcip_mcp.tools.training_tools import manifest_compatibility

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)
    config = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0],
                             subject=OTHER_SUBJECT)
    config["data"]["split"] = {"manifest_dir": str(out)}
    config["data"]["date"] = "2099-01-01"

    issues = manifest_compatibility(config, manifest, str(out))

    assert any("subject" in i for i in issues)
    assert any("data.date" in i for i in issues)


def test_preflight_names_the_manifest_directory_in_the_date_block_message(tmp_path: Path):
    """The date-block refusal names the directory it read, restoring what the message carried
    before the shared function existed, reached through preflight_config so the proof runs
    against a signature the fix under test did not itself change."""
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    config = _bespoke_config(root / "images" / DATES[0], root / "annotations" / "2099-01-01")
    config["data"]["split"] = {"manifest_dir": str(out)}

    result = preflight_config(config)

    expected = f"split manifest at {str(out)!r} holds no members under date"
    assert any(expected in i for i in result["issues"])


def _manifest_empty_val_side(root: Path, out: Path, date: str) -> dict:
    """A real ``draw_splits`` draw, hand-mutated to drop one date's ``val`` members: the shape a
    manifest binds to as leaving a run's own date with an empty side. Every other required field
    stays exactly what ``draw_splits`` wrote, so this exercises only the one refusal under test."""
    from tcip_mcp.pipelines.data.splits import member_identity_parts
    from tcip_mcp.tools.data_tools import split_manifest_key

    manifest = _draw(root, out)
    this_date_val = {i for i in manifest["splits"]["val"] if member_identity_parts(i)[0] == date}
    manifest["splits"]["val"] = sorted(set(manifest["splits"]["val"]) - this_date_val)
    ts.replace(split_manifest_key(out), manifest)
    return manifest


def test_manifest_compatibility_flags_an_empty_side_after_narrowing_to_the_runs_date(
    tmp_path: Path,
):
    from tcip_mcp.tools.training_tools import manifest_compatibility

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _manifest_empty_val_side(root, out, DATES[0])
    config = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    config["data"]["split"] = {"manifest_dir": str(out)}

    issues = manifest_compatibility(config, manifest, str(out))

    assert any("empty side" in i for i in issues)


def test_preflight_config_flags_a_manifest_that_leaves_an_empty_side_under_this_date(
    tmp_path: Path,
):
    """The same refusal, reached through preflight_config: bind_manifest_stems would raise this
    exact way at launch, so preflight names it before the child ever runs."""
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _manifest_empty_val_side(root, out, DATES[0])
    config = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    config["data"]["split"] = {"manifest_dir": str(out)}

    result = preflight_config(config)

    assert any("empty side" in i for i in result["issues"])


def test_manifest_compatibility_admits_a_draw_splits_manifest_with_no_empty_side(
    tmp_path: Path,
):
    """A manifest draw_splits itself drew, never hand-mutated: valid work still passes once the
    empty-side check lives in the shared function."""
    from tcip_mcp.tools.training_tools import manifest_compatibility

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)
    config = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    config["data"]["split"] = {"manifest_dir": str(out)}

    assert manifest_compatibility(config, manifest, str(out)) == []


def test_manifest_compatibility_scope_check_reaches_the_picker(tmp_path: Path, monkeypatch):
    """Marker proof that manifest_compatibility reaches manifest_scope_issues, the one
    accumulator every manifest-scope consumer shares: a site that stopped calling it would pass
    this test's own scenario silently instead of raising the marker below."""
    import tcip_mcp.pipelines.data.splits as splits_mod
    from tcip_mcp.tools.training_tools import manifest_compatibility

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)
    config = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    config["data"]["split"] = {"manifest_dir": str(out)}

    monkeypatch.setattr(
        splits_mod, "manifest_scope_issues",
        lambda *a, **k: (["MARKER-COMPATIBILITY-SCOPE-ISSUE"], None),
    )

    issues = manifest_compatibility(config, manifest, str(out))

    assert any("MARKER-COMPATIBILITY-SCOPE-ISSUE" in i for i in issues)


def test_preflight_config_flags_a_manifest_with_no_images_root_recorded(tmp_path: Path):
    """The same refusal, reached through preflight_config: a stated root must be positively
    carried, and draw_splits always writes one, so only a hand-edited manifest ever trips it."""
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _manifest_missing_images_root(root, out, DATES[0])
    config = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    config["data"]["split"] = {"manifest_dir": str(out)}

    result = preflight_config(config)

    assert any("images root" in i for i in result["issues"])


def test_preflight_reports_the_conflict_issues_even_when_the_manifest_is_unreadable(
    tmp_path: Path,
):
    """The val_images_dir and drawn-key conflicts don't need the manifest to answer, so an
    unreadable manifest_dir must never suppress them: preflight names both conflicts and the
    read failure, not only the read failure."""
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    config = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    config["data"]["val_images_dir"] = str(root / "images" / DATES[1])
    config["data"]["split"] = {"manifest_dir": str(tmp_path / "nope"), "seed": 7}

    result = preflight_config(config)

    conflict_issues = [i for i in result["issues"] if "conflicts with" in i]
    read_issues = [i for i in result["issues"] if "no split manifest recorded" in i]
    assert len(conflict_issues) == 2
    assert len(read_issues) == 1


def test_preflight_reports_the_task_cannot_bind_explanation_without_reading_the_manifest(
    tmp_path: Path,
):
    """The task check doesn't need the manifest either: a task outside detection/instance_seg
    names why it cannot bind a split manifest even when the manifest_dir names nothing
    readable."""
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    config = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    config["model_source"]["task"] = "classification"
    config["data"]["task"] = "classification"
    config["data"]["split"] = {"manifest_dir": str(tmp_path / "nope")}

    result = preflight_config(config)

    assert any("cannot bind to one" in i for i in result["issues"])


# -- read_split_manifest_dir_checked ---------------------------------------------


def test_read_split_manifest_dir_checked_tells_absence_apart_from_a_broken_record(tmp_path: Path):
    from tcip_mcp.tools.data_tools import read_split_manifest_dir_checked, split_manifest_key

    absent_manifest, absent_error = read_split_manifest_dir_checked(tmp_path / "nope")
    assert (absent_manifest, absent_error) == (None, None)

    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    ts.replace(split_manifest_key(broken_dir), {"seed": 1, "splits": {"train": [], "val": []}})
    broken_manifest, broken_error = read_split_manifest_dir_checked(broken_dir)
    assert broken_manifest is None
    assert broken_error is not None and "subject" in broken_error


# -- list_split_choices ------------------------------------------------------------


def test_list_split_choices_answers_only_as_recorded_without_images_or_labels_dir(
    tmp_path: Path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.training_tools import list_split_choices

    create_experiment("exp-no-data", {
        "model_source": {"builder": "m:f", "task": "detection"}, "data": {},
    })

    result = list_split_choices("exp-no-data")

    assert result["manifests"] == []
    assert result["as_recorded"]["case"] == "drawn"


def test_list_split_choices_route_404s_for_an_unknown_experiment(
    tmp_path: Path, monkeypatch, client: TestClient,
) -> None:
    """The route itself exists and answers a known id (proving the 404 below discriminates the
    one unknown id, never a route that failed to mount at all)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment

    create_experiment("exp-known", {
        "model_source": {"builder": "m:f", "task": "detection"}, "data": {},
    })

    known = client.get("/api/training/configs/exp-known/splits")
    assert known.status_code == 200
    assert known.json()["as_recorded"]["case"] == "drawn"

    resp = client.get("/api/training/configs/nope/splits")
    assert resp.status_code == 404


def test_list_split_choices_offers_every_recorded_partition_with_the_bindings_own_counts(
    tmp_path: Path, monkeypatch,
):
    """Through the platform's own producers: a manifest ``draw_splits`` writes under the
    dataset's own ``splits`` directory, a second one elsewhere that a run bound to through
    ``launch_training``, a third for another subject reached through a bound run (listed
    disabled with the compatibility text), and a directory whose record will not decode (listed
    disabled, never absent)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})
    import subprocess

    class _StubChild:
        def __init__(self, *a, **k) -> None:
            self.pid = 4242

        def __class_getitem__(cls, item):
            return cls

    monkeypatch.setattr(subprocess, "Popen", _StubChild)

    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.pipelines.data.splits import bind_manifest_stems
    from tcip_mcp.tools.data_tools import split_manifest_key
    from tcip_mcp.tools.training_tools import launch_training, list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    dataset_default = root / "splits"
    default_manifest = _draw(root, dataset_default)

    elsewhere = tmp_path / "elsewhere"
    _draw(root, elsewhere, seed=2)
    bound_cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    bound_cfg["data"]["split"] = {"manifest_dir": str(elsewhere)}
    launched = launch_training(bound_cfg, str(tmp_path / "out_bound"))
    assert "error" not in launched, launched

    other_subject_dir = tmp_path / "other_subject"
    _draw(root, other_subject_dir, subject=OTHER_SUBJECT, seed=3)
    create_experiment("exp-other-subject", {
        "data": {"split": {"manifest_dir": str(other_subject_dir)}},
    })

    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    ts.replace(split_manifest_key(broken_dir), {"seed": 1, "splits": {"train": [], "val": []}})
    create_experiment("exp-broken-manifest", {"data": {"split": {"manifest_dir": str(broken_dir)}}})

    picked_cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    create_experiment("exp-picked", picked_cfg)

    result = list_split_choices("exp-picked")
    by_dir = {m["manifest_dir"]: m for m in result["manifests"]}

    expected = bind_manifest_stems(
        default_manifest, DATES[0], SUBJECT, None, ["a", "b", "c", "d", "e", "f"],
        images_dir=root / "images" / DATES[0])
    default_entry = by_dir[str(dataset_default)]
    assert default_entry["enabled"] is True
    assert default_entry["train"] == expected.train_bound
    assert default_entry["val"] == expected.val_bound
    assert default_entry["calibration"] == expected.calibration_bound
    assert default_entry["other_dates"] == expected.other_dates

    elsewhere_entry = by_dir[str(elsewhere)]
    assert elsewhere_entry["enabled"] is True
    assert elsewhere_entry["train"] > 0 and elsewhere_entry["val"] > 0

    other_subject_entry = by_dir[str(other_subject_dir)]
    assert other_subject_entry["enabled"] is False
    assert "subject" in other_subject_entry["reason"]

    broken_entry = by_dir[str(broken_dir)]
    assert broken_entry["enabled"] is False
    assert broken_entry["reason"] is not None


def test_list_split_choices_offers_a_frozen_manifest_with_its_origin(tmp_path: Path):
    """A manifest freeze_split_manifest wrote one level under the dataset's own splits
    directory is offered the identical checked-then-compatibility way as any other candidate,
    with its origin carried on the row."""
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.data_tools import freeze_split_manifest
    from tcip_mcp.tools.training_tools import list_split_choices

    from tests.test_freeze_split_manifest import _real_drawn_experiment

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(root, "exp-src")
    frozen = freeze_split_manifest("exp-src")
    assert "error" not in frozen, frozen

    picked_cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    create_experiment("exp-picked-frozen", picked_cfg)

    result = list_split_choices("exp-picked-frozen")
    by_dir = {m["manifest_dir"]: m for m in result["manifests"]}

    entry = by_dir[frozen["manifest_dir"]]
    assert entry["enabled"] is True
    assert entry["origin"]["experiment_id"] == "exp-src"


def test_list_split_choices_offers_two_frozen_manifests_under_one_splits_directory(tmp_path: Path):
    """Two runs each freeze to their own default directory (no output_path given), landing
    under the dataset's one splits/ directory; list_split_choices offers both."""
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.data_tools import freeze_split_manifest
    from tcip_mcp.tools.training_tools import list_split_choices

    from tests.test_freeze_split_manifest import _real_drawn_experiment

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(root, "exp-first")
    _real_drawn_experiment(root, "exp-second")
    first = freeze_split_manifest("exp-first")
    second = freeze_split_manifest("exp-second")
    assert "error" not in first, first
    assert "error" not in second, second
    assert Path(first["manifest_dir"]).parent == root / "splits"
    assert Path(second["manifest_dir"]).parent == root / "splits"
    assert first["manifest_dir"] != second["manifest_dir"]

    picked_cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    create_experiment("exp-picked-both-frozen", picked_cfg)

    result = list_split_choices("exp-picked-both-frozen")
    by_dir = {m["manifest_dir"]: m for m in result["manifests"]}

    assert by_dir[first["manifest_dir"]]["enabled"] is True
    assert by_dir[first["manifest_dir"]]["origin"]["experiment_id"] == "exp-first"
    assert by_dir[second["manifest_dir"]]["enabled"] is True
    assert by_dir[second["manifest_dir"]]["origin"]["experiment_id"] == "exp-second"


def test_list_split_choices_as_recorded_reports_moved_directories_like_preflight(
    tmp_path: Path, monkeypatch,
):
    """The picker's own "As recorded" row checks the same directory presence preflight would
    refuse a launch on, so a snapshot whose recorded images_dir no longer exists reads disabled
    before Start rather than only failing after it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.training_tools import list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    create_experiment("exp-moved-dirs", cfg)
    shutil.rmtree(root / "images" / DATES[0])

    result = list_split_choices("exp-moved-dirs")

    assert result["as_recorded"]["compatible"] is False
    assert "Directory not found" in result["as_recorded"]["reason"]


def test_list_split_choices_as_recorded_reports_a_version_refused_own_binding(
    tmp_path: Path, monkeypatch,
):
    """A version-refused own binding reads as a disabled "As recorded" with the refusal text,
    never propagates as an uncaught StoreError: the plain reader's ValueError-only except would
    have let SchemaVersionRefused (a StoreError, not a ValueError) escape past it."""
    monkeypatch.setenv("TCIP_STORE_BACKEND", "file")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_store.binding import bind_default as _rebind

    _rebind()  # the autouse fixture already bound before this env var was set
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.training_tools import list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    manifest_dir = tmp_path / "m"
    manifest = _draw(root, manifest_dir)
    manifest_path = manifest_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps({**manifest, "schema_version": 99}), encoding="utf-8")

    cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    cfg["data"]["split"] = {"manifest_dir": str(manifest_dir)}
    create_experiment("exp-version-refused", cfg)

    result = list_split_choices("exp-version-refused")

    assert result["as_recorded"]["compatible"] is False
    assert "schema_version" in result["as_recorded"]["reason"]


def test_list_split_choices_reports_the_recorded_split_keys_a_partition_replaces(
    tmp_path: Path, monkeypatch,
):
    """Choosing a partition replaces ``data.split`` wholesale
    (:func:`candidate_config_with_manifest`); the listing discloses every recorded key other
    than ``manifest_dir`` that drops, per offered manifest."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.training_tools import candidate_config_with_manifest, list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    dataset_default = root / "splits"
    _draw(root, dataset_default)

    cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    cfg["data"]["split"] = {"seed": 7, "group_by": "tile_prefix"}
    create_experiment("exp-drawn-policy", cfg)

    result = list_split_choices("exp-drawn-policy")
    entry = next(m for m in result["manifests"] if m["manifest_dir"] == str(dataset_default))

    candidate = candidate_config_with_manifest(cfg, str(dataset_default))
    dropped = sorted(set(cfg["data"]["split"]) - set(candidate["data"]["split"]))
    assert entry["replaced_split_keys"] == dropped


def test_list_split_choices_reports_the_redraw_flag_among_the_keys_a_partition_replaces(
    tmp_path: Path, monkeypatch,
):
    """A config's own recorded ``data.split`` names ``redraw_within_manifest`` (bound and
    redrawing) as well as ``manifest_dir`` and ``seed``: offering a different partition drops
    all of them but ``manifest_dir`` itself, and the listing names every one dropped, the
    redraw flag included."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.training_tools import candidate_config_with_manifest, list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    dataset_default = root / "splits"
    _draw(root, dataset_default)
    own_manifest_dir = tmp_path / "own"
    _draw(root, own_manifest_dir, seed=3)

    cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    cfg["data"]["split"] = {
        "manifest_dir": str(own_manifest_dir), "seed": 11, "redraw_within_manifest": True,
    }
    create_experiment("exp-redrawn-policy", cfg)

    result = list_split_choices("exp-redrawn-policy")
    entry = next(m for m in result["manifests"] if m["manifest_dir"] == str(dataset_default))

    candidate = candidate_config_with_manifest(cfg, str(dataset_default))
    dropped = sorted(set(cfg["data"]["split"]) - set(candidate["data"]["split"]))
    assert entry["replaced_split_keys"] == dropped


def test_list_split_choices_never_names_a_null_valued_split_key_as_replaced(
    tmp_path: Path, monkeypatch,
):
    """A recorded ``data.split`` key present with a ``null`` value never configured anything a
    chosen partition could drop; it must not be named among ``replaced_split_keys``."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.training_tools import list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    dataset_default = root / "splits"
    _draw(root, dataset_default)

    cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    cfg["data"]["split"] = {"seed": 7, "group_by": None}
    create_experiment("exp-null-policy-key", cfg)

    result = list_split_choices("exp-null-policy-key")
    entry = next(m for m in result["manifests"] if m["manifest_dir"] == str(dataset_default))

    assert entry["replaced_split_keys"] == ["seed"]


def test_list_split_choices_reads_the_picked_experiments_own_config_only_once(
    tmp_path: Path, monkeypatch,
):
    """``experiment_ids_with_status`` enumerates every experiment including the one being
    listed for; its own manifest_dir is already known from the read taken above the loop, so
    the loop must skip it rather than reading its config a second time only to derive the
    identical fact and discard it as its own binding. ``images_dir``/``labels_dir`` must be
    recorded, or the function returns before ever reaching that loop."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    import tcip_mcp.experiments as experiments_module
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.training_tools import list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    picked_cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    create_experiment("exp-picked-once", picked_cfg)
    other_cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    create_experiment("exp-other-once", other_cfg)

    reads: list[str] = []
    original_read_member = experiments_module.read_member

    def _counting_read_member(key, *a, **k):
        reads.append(key.parts[0])
        return original_read_member(key, *a, **k)

    monkeypatch.setattr(experiments_module, "read_member", _counting_read_member)

    list_split_choices("exp-picked-once")

    assert reads.count("exp-picked-once") == 1
    assert reads.count("exp-other-once") == 1


def test_list_split_choices_does_not_offer_the_own_manifest_under_a_different_spelling(
    tmp_path: Path, monkeypatch,
):
    """A differently spelled path to the config's own bound directory (a trailing separator,
    dropped by ``Path(...).resolve()`` but not by a raw string compare) must not be offered as
    if it were a second, alternative partition: both spellings normalize to the identical
    picker identity."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.training_tools import list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    elsewhere = tmp_path / "elsewhere"
    _draw(root, elsewhere, seed=2)

    cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    cfg["data"]["split"] = {"manifest_dir": str(elsewhere)}
    create_experiment("exp-own-spelling", cfg)

    respelled = str(elsewhere) + os.sep
    other_cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    other_cfg["data"]["split"] = {"manifest_dir": respelled}
    create_experiment("exp-other-spelling", other_cfg)

    result = list_split_choices("exp-own-spelling")

    offered = {m["manifest_dir"] for m in result["manifests"]}
    assert str(elsewhere) not in offered
    assert respelled not in offered


def test_list_split_choices_offers_a_symlinked_manifest_directory_once(
    tmp_path: Path, monkeypatch,
):
    """A symlink under the dataset's ``splits/`` directory to a sibling manifest directory
    resolves to the identical directory the candidate dedupe must fold, or one partition is
    offered twice under two spellings."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.training_tools import list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    real_dir = splits_dir / "real"
    _draw(root, real_dir, seed=3)

    link_dir = splits_dir / "link"
    try:
        os.symlink(real_dir, link_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"this platform refuses directory symlink creation for this user: {exc}")

    cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    create_experiment("exp-symlinked-splits", cfg)

    result = list_split_choices("exp-symlinked-splits")

    offered = [m["manifest_dir"] for m in result["manifests"]]
    matches = [p for p in offered if Path(p).resolve() == real_dir.resolve()]
    assert len(matches) == 1


def test_list_split_choices_offers_a_case_respelled_duplicate_manifest_once(
    tmp_path: Path, monkeypatch,
):
    """Two candidate experiments bound to the identical directory under different case spelling
    must be offered once, not twice, on a case-insensitive filesystem."""
    if os.name != "nt":
        pytest.skip("case folding only matters on a case-insensitive filesystem")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.training_tools import list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    shared_dir = tmp_path / "shared_manifest"
    _draw(root, shared_dir, seed=4)

    own_cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    create_experiment("exp-picker", own_cfg)

    lower_cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    lower_cfg["data"]["split"] = {"manifest_dir": str(shared_dir)}
    create_experiment("exp-candidate-lower", lower_cfg)

    upper_cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    upper_cfg["data"]["split"] = {"manifest_dir": str(shared_dir).upper()}
    create_experiment("exp-candidate-upper", upper_cfg)

    result = list_split_choices("exp-picker")

    offered = [m["manifest_dir"] for m in result["manifests"]]
    matches = [p for p in offered if Path(p).resolve() == shared_dir.resolve()]
    assert len(matches) == 1


# -- POST /api/training/runs's split_manifest_dir --------------------------------


def test_relaunch_route_409s_for_a_split_manifest_dir_outside_the_enabled_set(
    tmp_path: Path, monkeypatch, client: TestClient,
) -> None:
    """The browser's string is verified against the platform's own listing: a string this
    listing never offered, and one it offered but disabled, both refuse the same way."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.data_tools import split_manifest_key

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    create_experiment("exp-picker-guard", cfg)

    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    ts.replace(split_manifest_key(broken_dir), {"seed": 1, "splits": {"train": [], "val": []}})
    create_experiment("exp-broken-source", {"data": {"split": {"manifest_dir": str(broken_dir)}}})

    resp_unknown = client.post("/api/training/runs", json={
        "experiment_id": "exp-picker-guard", "split_manifest_dir": str(tmp_path / "never-listed"),
    })
    assert resp_unknown.status_code == 409

    resp_disabled = client.post("/api/training/runs", json={
        "experiment_id": "exp-picker-guard", "split_manifest_dir": str(broken_dir),
    })
    assert resp_disabled.status_code == 409


def test_relaunch_route_leaves_the_snapshots_data_unchanged_when_no_partition_is_chosen(
    tmp_path: Path, monkeypatch, client: TestClient,
) -> None:
    """"As recorded" launches the stored config's own data section, whether it was bound or
    drawn, byte for byte: the route never rewrites data.split unless a partition is chosen."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    import tcip_mcp.tools.training_tools as training_tools_module
    from tcip_mcp.experiments import create_experiment

    captured: dict = {}

    def fake_launch_training(config, *a, **k):
        captured["data"] = config.get("data")
        return {"run_id": "r1", "experiment_id": config.get("experiment_id"), "status": "launched"}

    monkeypatch.setattr(training_tools_module, "launch_training", fake_launch_training)

    drawn_data = {"images_dir": "/data/images", "labels_dir": "/data/labels", "subject": SUBJECT}
    create_experiment("exp-drawn", {
        "model_source": {"builder": "m:f", "task": "detection"}, "data": dict(drawn_data),
    })
    resp = client.post("/api/training/runs", json={"experiment_id": "exp-drawn"})
    assert resp.status_code == 200, resp.json()
    assert captured["data"] == drawn_data

    bound_data = {"images_dir": "/data/images", "labels_dir": "/data/labels", "subject": SUBJECT,
                 "split": {"manifest_dir": "/some/manifest"}}
    create_experiment("exp-bound", {
        "model_source": {"builder": "m:f", "task": "detection"}, "data": dict(bound_data),
    })
    resp = client.post("/api/training/runs", json={"experiment_id": "exp-bound"})
    assert resp.status_code == 200, resp.json()
    assert captured["data"] == bound_data


def test_relaunch_route_launches_with_a_chosen_manifest_and_refreshes_the_pristine_config(
    tmp_path: Path, monkeypatch, client: TestClient,
) -> None:
    """A launch with a chosen partition submits a string the server itself listed; on the
    pristine branch the existing first-run refresh stores that candidate, data.split replaced
    wholesale and val_images_dir removed, as the experiment's own config."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})
    import subprocess

    class _StubChild:
        def __init__(self, *a, **k) -> None:
            self.pid = 4242

        def __class_getitem__(cls, item):
            return cls

    monkeypatch.setattr(subprocess, "Popen", _StubChild)

    from tcip_mcp.experiments import config_key, create_experiment, read_member

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    chosen = root / "splits"
    _draw(root, chosen)

    cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    cfg["data"]["val_images_dir"] = str(root / "images" / DATES[1])
    create_experiment("exp-choose-partition", cfg)

    resp = client.post("/api/training/runs", json={
        "experiment_id": "exp-choose-partition", "split_manifest_dir": str(chosen),
    })
    assert resp.status_code == 200, resp.json()

    snapshot = read_member(config_key("exp-choose-partition"))
    assert snapshot["data"]["split"] == {"manifest_dir": str(chosen)}
    assert "val_images_dir" not in snapshot["data"]


def test_relaunch_route_admits_a_symlinked_spelling_of_an_offered_split_directory(
    tmp_path: Path, monkeypatch, client: TestClient,
) -> None:
    """The picker's own dedupe offers one spelling of a symlinked directory; the relaunch route
    must still admit the other spelling of the identical directory, not just the one string it
    happened to list. split_dir_identity is the one comparison both sides now share, so a
    symlinked or differently cased spelling of an offered directory is accepted."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})
    import subprocess

    class _StubChild:
        def __init__(self, *a, **k) -> None:
            self.pid = 4242

        def __class_getitem__(cls, item):
            return cls

    monkeypatch.setattr(subprocess, "Popen", _StubChild)

    from tcip_mcp.experiments import config_key, create_experiment, read_member
    from tcip_mcp.tools.training_tools import list_split_choices

    def identity(p: str) -> str:
        return os.path.normcase(str(Path(p).resolve()))

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    real_dir = splits_dir / "real"
    _draw(root, real_dir, seed=5)

    link_dir = splits_dir / "link"
    try:
        os.symlink(real_dir, link_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"this platform refuses directory symlink creation for this user: {exc}")

    cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    create_experiment("exp-symlink-relaunch", cfg)

    choices = list_split_choices("exp-symlink-relaunch")
    offered = next(m["manifest_dir"] for m in choices["manifests"] if m["enabled"])
    assert offered in (str(real_dir), str(link_dir))
    other_spelling = str(real_dir) if offered == str(link_dir) else str(link_dir)
    assert identity(other_spelling) == identity(offered)

    resp = client.post("/api/training/runs", json={
        "experiment_id": "exp-symlink-relaunch", "split_manifest_dir": other_spelling,
    })
    assert resp.status_code == 200, resp.json()

    snapshot = read_member(config_key("exp-symlink-relaunch"))
    assert snapshot["data"]["split"] == {"manifest_dir": other_spelling}


def test_a_chosen_manifest_binds_and_the_runs_own_split_record_names_it(tmp_path: Path) -> None:
    """The candidate config the route builds for a chosen partition, bound the exact way the
    child binds it (auto_train_val then persist_split_manifest, the sequence
    subprocess_worker.run follows, called directly so no training subprocess is needed): the
    run's own split.json names the chosen manifest as what it bound to."""
    from tcip_mcp.experiments import create_experiment, read_split_manifest
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_split_manifest

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    chosen = tmp_path / "chosen"
    _draw(root, chosen, seed=5)

    data_cfg = {
        "images_dir": str(root / "images" / DATES[0]),
        "labels_dir": str(root / "annotations" / DATES[0]),
        "subject": SUBJECT, "split": {"manifest_dir": str(chosen)},
    }
    train_ds, val_ds, label_digests = auto_train_val("detection", data_cfg, None)
    create_experiment("exp-bound-split-record", {})
    persist_split_manifest("exp-bound-split-record", train_ds, val_ds, data_cfg,
                           label_digests=label_digests)

    split_record = read_split_manifest("exp-bound-split-record")
    assert split_record["manifest_binding"]["manifest_dir"] == str(chosen)
