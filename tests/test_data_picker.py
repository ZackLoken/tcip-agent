"""The data picker: choosing a recorded partition (or "As recorded") for a relaunch.

Covers the shared ``selection_compatibility`` check, the checked selection reader,
``list_split_choices`` and its route, and the launch route's ``selection_dir`` field.

The binding mechanics themselves (``auto_train_val``'s selection branch, ``read_selection``) are
``test_selection_binding.py``'s; this file reuses its dataset fixture rather than restating it.
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
from tcip_mcp.pipelines.data.selection import ClassScope
from tcip_web.app import app

from tests.test_selection_binding import DATES, OTHER_SUBJECT, SUBJECT, _draw, \
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
        "batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                     "mixed_precision": False, "device": "cpu",
                     "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False},
    }


# -- selection_compatibility ---------------------------------------------------


def _selection_with_an_empty_val_side(root: Path, out: Path):
    """A real ``draw_splits`` draw, rewritten without its ``val`` samples: the shape a run binds
    to as leaving it with an empty side. Every other recorded field stays exactly what the draw
    wrote, so this exercises only the one refusal under test."""
    from tcip_mcp.pipelines.data.selection import Selection, write_selection

    drawn = _draw(root, out)
    return write_selection(out, Selection(
        samples=tuple(s for s in drawn.samples if s.side != "val"),
        subject=drawn.subject, attribute=drawn.attribute, id_map=drawn.id_map,
        seed=drawn.seed, group_by=drawn.group_by,
    ))


def test_selection_compatibility_flags_an_empty_side(tmp_path: Path):
    from tcip_mcp.tools.training_tools import selection_compatibility

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    selection = _selection_with_an_empty_val_side(root, out)
    config = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    config["data"]["split"] = {"selection_dir": str(out)}

    issues = selection_compatibility(config, selection, str(out))

    assert any("empty side" in i for i in issues)


def test_preflight_config_flags_a_selection_that_leaves_an_empty_side(tmp_path: Path):
    """The same refusal, reached through preflight_config: the bind would raise this exact way at
    launch, so preflight names it before the child ever runs."""
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _selection_with_an_empty_val_side(root, out)
    config = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    config["data"]["split"] = {"selection_dir": str(out)}

    result = preflight_config(config)

    assert any("empty side" in i for i in result["issues"])


def test_selection_compatibility_admits_a_draw_splits_selection_with_no_empty_side(
    tmp_path: Path,
):
    """A selection draw_splits itself drew, never hand-mutated: valid work still passes once the
    empty-side check lives in the shared function."""
    from tcip_mcp.tools.training_tools import selection_compatibility

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    selection = _draw(root, out)
    config = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    config["data"]["split"] = {"selection_dir": str(out)}

    assert selection_compatibility(config, selection, str(out)) == []


def test_preflight_reports_the_conflict_issues_even_when_the_manifest_is_unreadable(
    tmp_path: Path,
):
    """The drawn-key conflicts don't need the selection to answer, so an
    unreadable selection_dir must never suppress them: preflight names the conflict and the
    read failure, not only the read failure."""
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    config = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    config["data"]["split"] = {"selection_dir": str(tmp_path / "nope"), "seed": 7,
                               "group_by": "stem"}

    result = preflight_config(config)

    conflict_issues = [i for i in result["issues"] if "conflicts with" in i]
    read_issues = [i for i in result["issues"] if "no selection recorded" in i]
    assert len(conflict_issues) == 1
    assert len(read_issues) == 1


# -- read_selection_checked ------------------------------------------------------


def test_read_selection_checked_tells_absence_apart_from_a_broken_record(tmp_path: Path):
    from tcip_mcp.pipelines.data.selection import read_selection_checked, selection_key

    absent, absent_error = read_selection_checked(tmp_path / "nope")
    assert (absent, absent_error) == (None, None)

    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    ts.replace(selection_key(broken_dir), {"seed": 1, "samples": []})
    broken, broken_error = read_selection_checked(broken_dir)
    assert broken is None
    assert broken_error is not None and "no samples" in broken_error


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

    assert result["selections"] == []
    assert result["as_recorded"]["case"] == "drawn"


def test_list_split_choices_offers_a_table_selection_for_a_table_configuration(
    tmp_path: Path, monkeypatch,
):
    """The picker anchors on the image location and resolves the ground truth through the task's
    own key, so an ordinary images-plus-table configuration finds the selections drawn over that
    dataset. Requiring a directory key would return nothing for every table run."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.data_tools import draw_splits
    from tcip_mcp.tools.training_tools import list_split_choices
    from tests.test_mask_and_table_membership import _table_dataset

    root = tmp_path / "ds"
    images_dir, csv_path = _table_dataset(root)
    drawn = draw_splits(str(root), output_path=str(root / "splits" / "rows"),
                        ground_truth=str(csv_path), seed=4, train_ratio=0.5, val_ratio=0.25,
                        calibration_ratio=0.25, group_by="stem")
    assert "error" not in drawn, drawn

    create_experiment("exp-table-picker", {
        "model_source": {"builder": "m:f", "task": "classification"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(csv_path),
                 "task": "classification"},
    })

    result = list_split_choices("exp-table-picker")

    offered = {choice["selection_dir"]: choice for choice in result["selections"]}
    assert str(root / "splits" / "rows") in offered, result
    assert offered[str(root / "splits" / "rows")]["enabled"] is True, offered


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
    """Through the platform's own producers: a selection ``draw_splits`` writes under the
    dataset's own ``splits`` directory, a second one elsewhere that a run bound to through
    ``launch_training``, a third for another subject reached through a bound run (offered on its
    own terms, since a selection states its own scope), and a directory whose record will not
    decode (listed disabled, never absent)."""
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
    from tcip_mcp.pipelines.data.selection import selection_key
    from tcip_mcp.tools.training_tools import (
        candidate_config_with_selection, launch_training, list_split_choices,
    )

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    dataset_default = root / "splits"
    default_selection = _draw(root, dataset_default)

    elsewhere = tmp_path / "elsewhere"
    _draw(root, elsewhere, seed=2)
    bound_cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    bound_cfg["data"]["split"] = {"selection_dir": str(elsewhere)}
    launched = launch_training(bound_cfg, str(tmp_path / "out_bound"))
    assert "error" not in launched, launched

    other_subject_dir = tmp_path / "other_subject"
    _draw(root, other_subject_dir, subject=OTHER_SUBJECT, seed=3)
    create_experiment("exp-other-subject", {
        "data": {"split": {"selection_dir": str(other_subject_dir)}},
    })

    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    ts.replace(selection_key(broken_dir), {"seed": 1, "samples": []})
    create_experiment("exp-broken-selection", {"data": {"split": {"selection_dir": str(broken_dir)}}})

    picked_cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    create_experiment("exp-picked", picked_cfg)

    result = list_split_choices("exp-picked")
    by_dir = {m["selection_dir"]: m for m in result["selections"]}

    expected = default_selection.counts()
    default_entry = by_dir[str(dataset_default)]
    assert default_entry["enabled"] is True
    assert default_entry["train"] == expected["train"]
    assert default_entry["val"] == expected["val"]
    assert default_entry["calibration"] == expected["calibration"]

    elsewhere_entry = by_dir[str(elsewhere)]
    assert elsewhere_entry["enabled"] is True
    assert elsewhere_entry["train"] > 0 and elsewhere_entry["val"] > 0

    # A partition drawn for another subject is a real offer: choosing it empties the recorded
    # scope, since a bound run reads subject, attribute and class map off the selection.
    other_subject_entry = by_dir[str(other_subject_dir)]
    assert other_subject_entry["enabled"] is True
    chosen = candidate_config_with_selection(picked_cfg, str(other_subject_dir))
    assert ClassScope.recorded_in(chosen["data"]) == ClassScope()

    broken_entry = by_dir[str(broken_dir)]
    assert broken_entry["enabled"] is False
    assert broken_entry["reason"] is not None


def test_list_split_choices_offers_a_frozen_manifest_with_its_origin(tmp_path: Path):
    """A selection freeze_selection wrote one level under the dataset's own splits directory is
    offered the identical checked-then-compatibility way as any other candidate, with its origin
    carried on the row."""
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.data_tools import freeze_selection
    from tcip_mcp.tools.training_tools import list_split_choices

    from tests.test_freeze_selection import _real_drawn_experiment

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(root, "exp-src")
    frozen = freeze_selection("exp-src")
    assert "error" not in frozen, frozen

    picked_cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    create_experiment("exp-picked-frozen", picked_cfg)

    result = list_split_choices("exp-picked-frozen")
    by_dir = {m["selection_dir"]: m for m in result["selections"]}

    entry = by_dir[frozen["selection_dir"]]
    assert entry["enabled"] is True
    assert entry["origin"]["experiment_id"] == "exp-src"


def test_list_split_choices_offers_two_frozen_manifests_under_one_splits_directory(tmp_path: Path):
    """Two runs each freeze to their own default directory (no output_path given), landing
    under the dataset's one splits/ directory; list_split_choices offers both."""
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.data_tools import freeze_selection
    from tcip_mcp.tools.training_tools import list_split_choices

    from tests.test_freeze_selection import _real_drawn_experiment

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(root, "exp-first")
    _real_drawn_experiment(root, "exp-second")
    first = freeze_selection("exp-first")
    second = freeze_selection("exp-second")
    assert "error" not in first, first
    assert "error" not in second, second
    assert Path(first["selection_dir"]).parent == root / "splits"
    assert Path(second["selection_dir"]).parent == root / "splits"
    assert first["selection_dir"] != second["selection_dir"]

    picked_cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    create_experiment("exp-picked-both-frozen", picked_cfg)

    result = list_split_choices("exp-picked-both-frozen")
    by_dir = {m["selection_dir"]: m for m in result["selections"]}

    assert by_dir[first["selection_dir"]]["enabled"] is True
    assert by_dir[first["selection_dir"]]["origin"]["experiment_id"] == "exp-first"
    assert by_dir[second["selection_dir"]]["enabled"] is True
    assert by_dir[second["selection_dir"]]["origin"]["experiment_id"] == "exp-second"


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
    assert "Not found: data.images_dir" in result["as_recorded"]["reason"]


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

    from tcip_mcp.pipelines.data.selection import selection_document, selection_path

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    selection_dir = tmp_path / "m"
    drawn = _draw(root, selection_dir)
    selection_path(selection_dir).write_text(
        json.dumps({**selection_document(drawn), "schema_version": 99}), encoding="utf-8")

    cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    cfg["data"]["split"] = {"selection_dir": str(selection_dir)}
    create_experiment("exp-version-refused", cfg)

    result = list_split_choices("exp-version-refused")

    assert result["as_recorded"]["compatible"] is False
    assert "schema_version" in result["as_recorded"]["reason"]


def test_list_split_choices_reports_the_recorded_split_keys_a_partition_replaces(
    tmp_path: Path, monkeypatch,
):
    """Choosing a partition replaces ``data.split`` wholesale
    (:func:`candidate_config_with_selection`); the listing discloses every recorded key other
    than ``selection_dir`` that drops, per offered selection."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.training_tools import candidate_config_with_selection, list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    dataset_default = root / "splits"
    _draw(root, dataset_default)

    cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    cfg["data"]["split"] = {"seed": 7, "group_by": "tile_prefix"}
    create_experiment("exp-drawn-policy", cfg)

    result = list_split_choices("exp-drawn-policy")
    entry = next(m for m in result["selections"] if m["selection_dir"] == str(dataset_default))

    candidate = candidate_config_with_selection(cfg, str(dataset_default))
    dropped = sorted(set(cfg["data"]["split"]) - set(candidate["data"]["split"]))
    assert entry["replaced_split_keys"] == dropped


def test_list_split_choices_reports_the_redraw_flag_among_the_keys_a_partition_replaces(
    tmp_path: Path, monkeypatch,
):
    """A config's own recorded ``data.split`` names ``redraw_within_selection`` (bound and
    redrawing) as well as ``selection_dir`` and ``seed``: offering a different partition drops
    all of them but ``selection_dir`` itself, and the listing names every one dropped, the
    redraw flag included."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.training_tools import candidate_config_with_selection, list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    dataset_default = root / "splits"
    _draw(root, dataset_default)
    own_manifest_dir = tmp_path / "own"
    _draw(root, own_manifest_dir, seed=3)

    cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    cfg["data"]["split"] = {
        "selection_dir": str(own_manifest_dir), "seed": 11, "redraw_within_selection": True,
    }
    create_experiment("exp-redrawn-policy", cfg)

    result = list_split_choices("exp-redrawn-policy")
    entry = next(m for m in result["selections"] if m["selection_dir"] == str(dataset_default))

    candidate = candidate_config_with_selection(cfg, str(dataset_default))
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
    entry = next(m for m in result["selections"] if m["selection_dir"] == str(dataset_default))

    assert entry["replaced_split_keys"] == ["seed"]


def test_list_split_choices_reads_the_picked_experiments_own_config_only_once(
    tmp_path: Path, monkeypatch,
):
    """``experiment_ids_with_status`` enumerates every experiment including the one being
    listed for; its own selection_dir is already known from the read taken above the loop, so
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


def test_list_split_choices_does_not_offer_the_own_selection_under_a_different_spelling(
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
    cfg["data"]["split"] = {"selection_dir": str(elsewhere)}
    create_experiment("exp-own-spelling", cfg)

    respelled = str(elsewhere) + os.sep
    other_cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    other_cfg["data"]["split"] = {"selection_dir": respelled}
    create_experiment("exp-other-spelling", other_cfg)

    result = list_split_choices("exp-own-spelling")

    offered = {m["selection_dir"] for m in result["selections"]}
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

    offered = [m["selection_dir"] for m in result["selections"]]
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
    lower_cfg["data"]["split"] = {"selection_dir": str(shared_dir)}
    create_experiment("exp-candidate-lower", lower_cfg)

    upper_cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    upper_cfg["data"]["split"] = {"selection_dir": str(shared_dir).upper()}
    create_experiment("exp-candidate-upper", upper_cfg)

    result = list_split_choices("exp-picker")

    offered = [m["selection_dir"] for m in result["selections"]]
    matches = [p for p in offered if Path(p).resolve() == shared_dir.resolve()]
    assert len(matches) == 1


# -- POST /api/training/runs's selection_dir --------------------------------


def test_relaunch_route_409s_for_a_selection_dir_outside_the_enabled_set(
    tmp_path: Path, monkeypatch, client: TestClient,
) -> None:
    """The browser's string is verified against the platform's own listing: a string this
    listing never offered, and one it offered but disabled, both refuse the same way."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.pipelines.data.selection import selection_key

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    create_experiment("exp-picker-guard", cfg)

    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    ts.replace(selection_key(broken_dir), {"seed": 1, "samples": []})
    create_experiment("exp-broken-source", {"data": {"split": {"selection_dir": str(broken_dir)}}})

    resp_unknown = client.post("/api/training/runs", json={
        "experiment_id": "exp-picker-guard", "selection_dir": str(tmp_path / "never-listed"),
    })
    assert resp_unknown.status_code == 409

    resp_disabled = client.post("/api/training/runs", json={
        "experiment_id": "exp-picker-guard", "selection_dir": str(broken_dir),
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
        return {"experiment_id": config.get("experiment_id"), "status": "launched"}

    monkeypatch.setattr(training_tools_module, "launch_training", fake_launch_training)

    drawn_data = {"images_dir": "/data/images", "labels_dir": "/data/labels", "subject": SUBJECT}
    create_experiment("exp-drawn", {
        "model_source": {"builder": "m:f", "task": "detection"}, "data": dict(drawn_data),
    })
    resp = client.post("/api/training/runs", json={"experiment_id": "exp-drawn"})
    assert resp.status_code == 200, resp.json()
    assert captured["data"] == drawn_data

    bound_data = {"images_dir": "/data/images", "labels_dir": "/data/labels", "subject": SUBJECT,
                 "split": {"selection_dir": "/some/manifest"}}
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
    wholesale and the recorded class scope dropped, as the experiment's own config."""
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
    cfg["data"]["subject"] = SUBJECT
    create_experiment("exp-choose-partition", cfg)

    resp = client.post("/api/training/runs", json={
        "experiment_id": "exp-choose-partition", "selection_dir": str(chosen),
    })
    assert resp.status_code == 200, resp.json()

    snapshot = read_member(config_key("exp-choose-partition"))
    assert snapshot["data"]["split"] == {"selection_dir": str(chosen)}
    assert ClassScope.recorded_in(snapshot["data"]) == ClassScope()


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
    offered = next(m["selection_dir"] for m in choices["selections"] if m["enabled"])
    assert offered in (str(real_dir), str(link_dir))
    other_spelling = str(real_dir) if offered == str(link_dir) else str(link_dir)
    assert identity(other_spelling) == identity(offered)

    resp = client.post("/api/training/runs", json={
        "experiment_id": "exp-symlink-relaunch", "selection_dir": other_spelling,
    })
    assert resp.status_code == 200, resp.json()

    snapshot = read_member(config_key("exp-symlink-relaunch"))
    assert snapshot["data"]["split"] == {"selection_dir": other_spelling}


def test_a_chosen_selection_binds_and_the_runs_own_split_record_names_it(tmp_path: Path) -> None:
    """The candidate config the route builds for a chosen partition, bound the exact way the
    child binds it (auto_train_val then persist_run_partition, the sequence
    subprocess_worker.run follows, called directly so no training subprocess is needed): the
    run's own split.json names the chosen selection as what it bound to."""
    from tcip_mcp.experiments import create_experiment, read_run_partition
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_run_partition

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    chosen = tmp_path / "chosen"
    _draw(root, chosen, seed=5)

    data_cfg = {"split": {"selection_dir": str(chosen)}}
    train_ds, val_ds, partition = auto_train_val("detection", data_cfg, None)
    create_experiment("exp-bound-split-record", {})
    persist_run_partition("exp-bound-split-record", data_cfg,
                          partition=partition)

    split_record = read_run_partition("exp-bound-split-record")
    assert split_record["selection_binding"]["selection_dir"] == str(chosen)
