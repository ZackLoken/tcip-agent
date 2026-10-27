"""The data picker: choosing a recorded partition (or "As recorded") for a relaunch.

Covers the shared ``selection_compatibility`` check, the checked selection reader,
``list_split_choices`` and its route, and the launch route's ``selection_dir`` field.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from httpx import Response
    from pydantic import JsonValue

    from tcip_mcp.experiments import RunRecord
    from tcip_mcp.pipelines.data.selection import Selection
    from tcip_mcp.pipelines.schemas import DataSpec

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import tcip_store as ts
from tcip_mcp.dataset_layout import image_dir
from tests._chain_fixtures import (
    FRAME_STATISTICS,
    training_config,
    weights_statistics,
)
from tests._producer_fixtures import fake_popen
from tests._training_values import RATIONALE, drawn_split
from tests._verified_checkpoint_fixtures import BUILT_DETECTOR, opened_run
from tests.test_selection_binding import (
    DATES,
    OTHER_SUBJECT,
    SUBJECT,
    _draw,
    _two_subject_two_date_dataset,
)


def _bespoke_config(
    project: Path,
    images_dir: Path,
    *,
    subject: str = SUBJECT,
    split: dict | None = None,
    model_source: dict = BUILT_DETECTOR,
) -> dict:
    """Return a config of ``project`` of ``model_source`` over ``images_dir`` scoped to ``subject``.

    The config draws ``split``, by default a tile-prefix draw holding 0.15 out.
    """
    return training_config(
        project,
        model_source,
        {
            "images_dir": str(images_dir),
            "scope": {"subject": subject},
            "split": split
            or drawn_split(val_ratio=0.15, grouping="tile_prefix"),
        },
    )


def _bound_config(
    root: Path,
    selection_dir: Path | str,
    project: Path,
    *,
    model_source: dict = BUILT_DETECTOR,
    **split: int | None,
) -> dict:
    """Return a config of ``project`` of ``model_source`` bound to a recorded selection.

    The selection is the one at ``selection_dir``; ``split``'s keys sit beside the binding, and
    the config states no scope of its own.
    """
    return training_config(
        project,
        model_source,
        {
            "images_dir": str(image_dir(root, DATES[0])),
            "split": {"selection_dir": str(selection_dir), **split},
        },
    )


def _data(config: dict) -> DataSpec:
    """Return ``config``'s data block as a run validates it."""
    from tcip_mcp.pipelines.schemas import train_config

    return train_config(config).data


def _chosen(config: dict, selection_dir: str, project: Path) -> dict:
    """Return ``config`` with the data block that choosing ``selection_dir`` builds.

    The block is built by ``training_tools.data_with_selection``.
    """
    from tcip_mcp.tools.training_tools import data_with_selection

    return {
        **config,
        "data": data_with_selection(
            _data(config), selection_dir, project
        ).record(),
    }


# -- selection_compatibility ---------------------------------------------------


def _selection_with_an_empty_val_side(
    project: Path, root: Path, out: Path
) -> Selection:
    """Write a real ``draw_splits`` draw under ``project`` rewritten without its ``val`` samples.

    Every other recorded field stays what the draw wrote.
    """
    from tcip_mcp.pipelines.data.selection import Selection, write_selection

    drawn = _draw(project, root, out)
    return write_selection(
        out,
        Selection(
            samples=tuple(s for s in drawn.samples if s.side != "val"),
            scope=drawn.scope,
            seed=drawn.seed,
            group_by=drawn.group_by,
            ratios=drawn.ratios,
            stratify_foreground=drawn.stratify_foreground,
            rationale=drawn.rationale,
        ),
        project=project,
    )


def test_selection_compatibility_flags_an_empty_side(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.split_construction import (
        selection_compatibility,
    )

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    selection = _selection_with_an_empty_val_side(tmp_path, root, out)

    data = _data(
        _bound_config(
            root,
            out,
            tmp_path,
            model_source=weights_statistics(BUILT_DETECTOR),
        )
    )
    issues = selection_compatibility(data, data.split, selection)

    assert any("empty side" in i for i in issues)


def test_preflight_config_flags_a_selection_that_leaves_an_empty_side(
    tmp_path: Path,
) -> None:
    """``preflight_config`` flags a bound selection that leaves an empty side."""
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _selection_with_an_empty_val_side(tmp_path, root, out)

    result = preflight_config(
        tmp_path,
        _bound_config(
            root,
            out,
            tmp_path,
            model_source=weights_statistics(BUILT_DETECTOR),
        ),
    )

    assert any("empty side" in i for i in result["issues"])


def test_selection_compatibility_admits_a_draw_splits_selection_with_no_empty_side(
    tmp_path: Path,
) -> None:
    """A selection ``draw_splits`` itself drew passes the compatibility check.

    The selection is the one ``draw_splits`` wrote, unmodified.
    """
    from tcip_mcp.pipelines.data.split_construction import (
        selection_compatibility,
    )

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    selection = _draw(tmp_path, root, out)
    data = _data(_bound_config(root, out, tmp_path))

    assert selection_compatibility(data, data.split, selection) == []


def test_preflight_refuses_a_draw_key_beside_a_selection_even_when_it_is_unreadable(
    tmp_path: Path,
) -> None:
    """Preflight refuses a draw key beside a selection even when the selection is unreadable.

    A split naming a selection is a bound split, which carries no draw key: the schema refuses
    one by name without reading the selection, so an unreadable selection_dir does not suppress
    it.
    """
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    config = _bespoke_config(
        tmp_path,
        image_dir(root, DATES[0]),
        split={
            "selection_dir": str(tmp_path / "nope"),
            "seed": 7,
            "grouping": "stem",
        },
        model_source=weights_statistics(BUILT_DETECTOR),
    )

    result = preflight_config(tmp_path, config)

    assert [i for i in result["issues"] if "grouping" in i], result["issues"]


# -- read_selection_checked ------------------------------------------------------


def test_read_selection_checked_tells_absence_apart_from_a_broken_record(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.data.selection import (
        read_selection_checked,
        selection_key,
    )

    absent, absent_error = read_selection_checked(
        tmp_path / "nope", project=tmp_path
    )
    assert (absent, absent_error) == (None, None)

    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    ts.replace(selection_key(broken_dir), {"seed": 1, "samples": []})
    broken, broken_error = read_selection_checked(broken_dir, project=tmp_path)
    assert broken is None
    assert broken_error is not None and "group_by" in broken_error


# -- list_split_choices ------------------------------------------------------------


def test_list_split_choices_offers_a_table_selection_for_a_table_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The split picker offers a selection drawn over a table to a table configuration.

    A selection drawn with the table as its ground truth is offered to a classification run
    configured by its images directory and that table.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.data_tools import draw_splits
    from tcip_mcp.tools.training_tools import list_split_choices
    from tests.test_mask_and_table_membership import _table_dataset

    root = tmp_path / "ds"
    images_dir, csv_path, _images, _masks = _table_dataset(root)
    drawn = draw_splits(
        tmp_path,
        str(root),
        output_path=str(root / "splits" / "rows"),
        ground_truth=str(csv_path),
        seed=4,
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
        grouping="stem",
        stratify_foreground=True,
        rationale=RATIONALE,
        actor=None,
    )
    assert "error" not in drawn, drawn

    from tests.tiny_trainer_fixtures import (
        MEAN_INTENSITY_CLASSIFIER,
        TINY_TRAINER_FILE,
    )

    opened_run(
        tmp_path,
        training_config(
            tmp_path,
            {
                "builder": MEAN_INTENSITY_CLASSIFIER,
                "source_files": [TINY_TRAINER_FILE],
                "task": "classification",
                "normalization": FRAME_STATISTICS,
            },
            {
                "images_dir": str(images_dir),
                "labels_dir": str(csv_path),
                "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
            },
        ),
        experiment_id="exp-table-picker",
    )

    result = list_split_choices(tmp_path, "exp-table-picker")

    offered = {
        choice["selection_dir"]: choice for choice in result["selections"]
    }
    assert str(root / "splits" / "rows") in offered, result
    assert offered[str(root / "splits" / "rows")]["enabled"] is True, offered


def test_list_split_choices_route_404s_for_an_unknown_experiment(
    tmp_path: Path,
    opened_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
) -> None:
    """The split-choices route answers 404 for an unknown experiment and 200 for a known one.

    The known id's answer is its drawn split, and the configs path answers 404 for it.
    """
    monkeypatch.chdir(tmp_path)

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    opened_run(
        tmp_path,
        _bespoke_config(tmp_path, image_dir(root, DATES[0])),
        experiment_id="exp-known",
    )

    known = client.get("/api/training/runs/exp-known/splits")
    assert known.status_code == 200
    assert known.json()["as_recorded"]["case"] == "drawn"

    resp = client.get("/api/training/runs/nope/splits")
    assert resp.status_code == 404
    assert (
        client.get("/api/training/configs/exp-known/splits").status_code == 404
    )


def test_list_split_choices_offers_every_recorded_partition_with_the_bindings_own_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The split picker offers each recorded partition, the default at its selection's counts.

    Through the platform's own producers: a selection ``draw_splits`` writes under the dataset's
    own ``splits`` directory, a second one elsewhere that a run bound to through
    ``launch_training``, a third for another subject reached through a bound run (enabled, and
    choosing it drops the stated scope), and a directory under the dataset's ``splits`` whose
    record will not decode (listed disabled with a reason).
    """
    monkeypatch.chdir(tmp_path)
    fake_popen(monkeypatch, [])

    from tcip_mcp.pipelines.data.selection import selection_key
    from tcip_mcp.tools.training_tools import (
        RunLaunched,
        launch_training,
        list_split_choices,
    )

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    dataset_default = root / "splits"
    default_selection = _draw(tmp_path, root, dataset_default)

    elsewhere = tmp_path / "elsewhere"
    _draw(tmp_path, root, elsewhere, seed=2)
    launched = launch_training(
        tmp_path, _bound_config(root, elsewhere, tmp_path), actor=None
    )
    assert isinstance(launched, RunLaunched), launched

    other_subject_dir = tmp_path / "other_subject"
    _draw(tmp_path, root, other_subject_dir, subject=OTHER_SUBJECT, seed=3)
    opened_run(
        tmp_path,
        _bound_config(root, other_subject_dir, tmp_path),
        experiment_id="exp-other-subject",
    )

    broken_dir = dataset_default / "broken"
    broken_dir.mkdir()
    ts.replace(selection_key(broken_dir), {"seed": 1, "samples": []})

    picked_cfg = _bespoke_config(tmp_path, image_dir(root, DATES[0]))
    opened_run(tmp_path, picked_cfg, experiment_id="exp-picked")

    result = list_split_choices(tmp_path, "exp-picked")
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

    # A partition drawn for another subject is a real offer: choosing it drops the stated
    # scope, since a bound run reads its scope off the selection.
    other_subject_entry = by_dir[str(other_subject_dir)]
    assert other_subject_entry["enabled"] is True
    chosen = _chosen(picked_cfg, str(other_subject_dir), tmp_path)
    assert "scope" not in chosen["data"]

    broken_entry = by_dir[str(broken_dir)]
    assert broken_entry["enabled"] is False
    assert broken_entry["reason"] is not None


def test_list_split_choices_offers_a_frozen_manifest(tmp_path: Path) -> None:
    """The split picker offers a frozen manifest as an enabled candidate.

    The selection is the one freeze_selection wrote one level under the dataset's own splits
    directory.
    """
    from tcip_mcp.tools.data_tools import freeze_selection
    from tcip_mcp.tools.training_tools import list_split_choices
    from tests.test_freeze_selection import _real_drawn_experiment

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(tmp_path, root, "exp-src")
    frozen = freeze_selection(tmp_path, "exp-src", actor=None)
    assert "error" not in frozen, frozen

    picked_cfg = _bespoke_config(tmp_path, image_dir(root, DATES[0]))
    opened_run(tmp_path, picked_cfg, experiment_id="exp-picked-frozen")

    result = list_split_choices(tmp_path, "exp-picked-frozen")
    by_dir = {m["selection_dir"]: m for m in result["selections"]}

    entry = by_dir[frozen["selection_dir"]]
    assert entry["enabled"] is True


def test_list_split_choices_offers_two_frozen_manifests_under_one_splits_directory(
    tmp_path: Path,
) -> None:
    """The split picker offers both of two frozen manifests under one splits directory.

    Two runs each freeze to their own default directory (no output_path given), landing under
    the dataset's one splits/ directory; list_split_choices offers both.
    """
    from tcip_mcp.tools.data_tools import freeze_selection
    from tcip_mcp.tools.training_tools import list_split_choices
    from tests.test_freeze_selection import _real_drawn_experiment

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(tmp_path, root, "exp-first")
    _real_drawn_experiment(tmp_path, root, "exp-second")
    first = freeze_selection(tmp_path, "exp-first", actor=None)
    second = freeze_selection(tmp_path, "exp-second", actor=None)
    assert "error" not in first, first
    assert "error" not in second, second
    assert Path(first["selection_dir"]).parent == root / "splits"
    assert Path(second["selection_dir"]).parent == root / "splits"
    assert first["selection_dir"] != second["selection_dir"]

    picked_cfg = _bespoke_config(tmp_path, image_dir(root, DATES[0]))
    opened_run(tmp_path, picked_cfg, experiment_id="exp-picked-both-frozen")

    result = list_split_choices(tmp_path, "exp-picked-both-frozen")
    by_dir = {m["selection_dir"]: m for m in result["selections"]}

    assert by_dir[first["selection_dir"]]["enabled"] is True
    assert by_dir[second["selection_dir"]]["enabled"] is True


def test_list_split_choices_as_recorded_reports_moved_directories_like_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The picker's "As recorded" row reads incompatible when its images directory is gone.

    The row's reason names ``data.images_dir`` as not found.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    cfg = _bespoke_config(tmp_path, image_dir(root, DATES[0]))
    opened_run(tmp_path, cfg, experiment_id="exp-moved-dirs")
    shutil.rmtree(image_dir(root, DATES[0]))

    result = list_split_choices(tmp_path, "exp-moved-dirs")

    assert result["as_recorded"]["compatible"] is False
    assert "Not found: data.images_dir" in result["as_recorded"]["reason"]


def test_list_split_choices_reports_the_recorded_split_keys_a_partition_replaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The split listing reports the recorded split keys a chosen partition replaces.

    Choosing a partition replaces ``data.split`` wholesale
    (``training_tools.data_with_selection``); the listing discloses, for an offered selection,
    the stated keys other than ``selection_dir`` that drop.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    dataset_default = root / "splits"
    _draw(tmp_path, root, dataset_default)

    cfg = _bespoke_config(
        tmp_path,
        image_dir(root, DATES[0]),
        split=drawn_split(seed=7, val_ratio=0.15, grouping="tile_prefix"),
    )
    opened_run(tmp_path, cfg, experiment_id="exp-drawn-policy")

    result = list_split_choices(tmp_path, "exp-drawn-policy")
    entry = next(
        m
        for m in result["selections"]
        if m["selection_dir"] == str(dataset_default)
    )

    candidate = _chosen(cfg, str(dataset_default), tmp_path)
    dropped = sorted(
        set(cfg["data"]["split"]) - set(candidate["data"]["split"])
    )
    assert entry["replaced_split_keys"] == dropped


def test_list_split_choices_reports_the_redraw_seed_among_the_keys_a_partition_replaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The split listing names a bound split's redraw seed among the keys a partition replaces.

    A config's own stated ``data.split`` names a ``seed`` (bound and redrawing at it) beside
    ``selection_dir``: offering a different partition drops it, and the listing names it.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    dataset_default = root / "splits"
    _draw(tmp_path, root, dataset_default)
    own_manifest_dir = tmp_path / "own"
    _draw(tmp_path, root, own_manifest_dir, seed=3)

    cfg = _bound_config(root, own_manifest_dir, tmp_path, seed=11)
    opened_run(tmp_path, cfg, experiment_id="exp-redrawn-policy")

    result = list_split_choices(tmp_path, "exp-redrawn-policy")
    entry = next(
        m
        for m in result["selections"]
        if m["selection_dir"] == str(dataset_default)
    )

    candidate = _chosen(cfg, str(dataset_default), tmp_path)
    dropped = sorted(
        set(cfg["data"]["split"]) - set(candidate["data"]["split"])
    )
    assert entry["replaced_split_keys"] == dropped


def test_list_split_choices_never_names_a_null_valued_split_key_as_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A null-valued split key is not named among the keys a partition replaces.

    A stated ``data.split`` key present with a ``null`` value configured nothing a chosen
    partition could drop, and ``replaced_split_keys`` is empty.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    dataset_default = root / "splits"
    _draw(tmp_path, root, dataset_default)
    own_manifest_dir = tmp_path / "own"
    _draw(tmp_path, root, own_manifest_dir, seed=3)

    cfg = _bound_config(root, own_manifest_dir, tmp_path, seed=None)
    opened_run(tmp_path, cfg, experiment_id="exp-null-policy-key")

    result = list_split_choices(tmp_path, "exp-null-policy-key")
    entry = next(
        m
        for m in result["selections"]
        if m["selection_dir"] == str(dataset_default)
    )

    assert entry["replaced_split_keys"] == []


def test_list_split_choices_reads_the_picked_runs_own_launch_record_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The split listing reads each run's launch record once, the picked run's own included."""
    monkeypatch.chdir(tmp_path)
    import tcip_mcp.experiments as experiments_module
    from tcip_mcp.tools.training_tools import list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    picked_cfg = _bespoke_config(tmp_path, image_dir(root, DATES[0]))
    opened_run(tmp_path, picked_cfg, experiment_id="exp-picked-once")
    other_cfg = _bespoke_config(tmp_path, image_dir(root, DATES[0]))
    opened_run(tmp_path, other_cfg, experiment_id="exp-other-once")

    reads: list[str] = []
    original_read_record = experiments_module.read_record

    def _counting_read_record(
        path: Path, *a: object, **k: object
    ) -> JsonValue:
        if Path(path).name == experiments_module.RUN_FILE:
            reads.append(Path(path).parent.name)
        return original_read_record(path, *a, **k)

    monkeypatch.setattr(
        experiments_module, "read_record", _counting_read_record
    )

    list_split_choices(tmp_path, "exp-picked-once")

    assert reads.count("exp-picked-once") == 1
    assert reads.count("exp-other-once") == 1


def test_list_split_choices_does_not_offer_the_own_selection_under_a_different_spelling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The split picker does not offer the run's own selection under a different spelling.

    Another run bound to the run's own directory spelled with a trailing separator leaves
    neither spelling offered.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    elsewhere = tmp_path / "elsewhere"
    _draw(tmp_path, root, elsewhere, seed=2)

    opened_run(
        tmp_path,
        _bound_config(root, elsewhere, tmp_path),
        experiment_id="exp-own-spelling",
    )

    respelled = str(elsewhere) + os.sep
    other_cfg = _bound_config(root, respelled, tmp_path)
    opened_run(tmp_path, other_cfg, experiment_id="exp-other-spelling")

    result = list_split_choices(tmp_path, "exp-own-spelling")

    offered = {m["selection_dir"] for m in result["selections"]}
    assert str(elsewhere) not in offered
    assert respelled not in offered


def test_list_split_choices_offers_a_symlinked_manifest_directory_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The split picker offers a symlinked manifest directory once.

    With a symlink under the dataset's ``splits/`` directory to a sibling manifest directory,
    one offer resolves to that directory.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    real_dir = splits_dir / "real"
    _draw(tmp_path, root, real_dir, seed=3)

    link_dir = splits_dir / "link"
    try:
        os.symlink(real_dir, link_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(
            f"this platform refuses directory symlink creation for this user: {exc}"
        )

    cfg = _bespoke_config(tmp_path, image_dir(root, DATES[0]))
    opened_run(tmp_path, cfg, experiment_id="exp-symlinked-splits")

    result = list_split_choices(tmp_path, "exp-symlinked-splits")

    offered = [m["selection_dir"] for m in result["selections"]]
    matches = [p for p in offered if Path(p).resolve() == real_dir.resolve()]
    assert len(matches) == 1


def test_list_split_choices_offers_a_case_respelled_duplicate_manifest_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The split picker offers a case-respelled duplicate manifest once.

    With two candidate runs bound to one directory under different case spellings, one offer
    resolves to it on a case-insensitive filesystem.
    """
    if os.name != "nt":
        pytest.skip(
            "case folding only matters on a case-insensitive filesystem"
        )
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    shared_dir = tmp_path / "shared_manifest"
    _draw(tmp_path, root, shared_dir, seed=4)

    own_cfg = _bespoke_config(tmp_path, image_dir(root, DATES[0]))
    opened_run(tmp_path, own_cfg, experiment_id="exp-picker")

    opened_run(
        tmp_path,
        _bound_config(root, shared_dir, tmp_path),
        experiment_id="exp-candidate-lower",
    )

    upper_cfg = _bound_config(root, str(shared_dir).upper(), tmp_path)
    opened_run(tmp_path, upper_cfg, experiment_id="exp-candidate-upper")

    result = list_split_choices(tmp_path, "exp-picker")

    offered = [m["selection_dir"] for m in result["selections"]]
    matches = [p for p in offered if Path(p).resolve() == shared_dir.resolve()]
    assert len(matches) == 1


# -- POST /api/training/runs's selection_dir --------------------------------


def test_relaunch_route_refuses_a_selection_dir_the_launch_cannot_bind(
    tmp_path: Path,
    opened_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
) -> None:
    """The relaunch route refuses a selection directory the launch cannot bind.

    The browser's string is checked by the launch it reaches: a directory nothing is recorded
    under, and one whose record is broken, both answer the launch's own refusal.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.pipelines.data.selection import selection_key

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    cfg = _bespoke_config(tmp_path, image_dir(root, DATES[0]))
    opened_run(tmp_path, cfg, experiment_id="exp-picker-guard")

    broken_dir = root / "splits" / "broken"
    broken_dir.mkdir(parents=True)
    ts.replace(selection_key(broken_dir), {"seed": 1, "samples": []})

    resp_unknown = client.post(
        "/api/training/runs",
        json={
            "relaunched_from": "exp-picker-guard",
            "selection_dir": str(tmp_path / "never-listed"),
            "user": "tester",
        },
    )
    assert resp_unknown.status_code == 422
    assert "no selection recorded" in str(
        resp_unknown.json()["detail"]["issues"]
    )

    resp_disabled = client.post(
        "/api/training/runs",
        json={
            "relaunched_from": "exp-picker-guard",
            "selection_dir": str(broken_dir),
            "user": "tester",
        },
    )
    assert resp_disabled.status_code == 422


def test_relaunch_route_launches_the_stated_data_unchanged_when_no_partition_is_chosen(
    tmp_path: Path,
    opened_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
) -> None:
    """The relaunch route launches the stated data unchanged when no partition is chosen.

    "As recorded" launches the stated config's own data section as it validates, whether it was
    bound or drawn: with no partition chosen the route does not rewrite data.split.
    """
    monkeypatch.chdir(tmp_path)
    fake_popen(monkeypatch, [])
    from tcip_mcp.pipelines.schemas import train_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    drawn = _bespoke_config(tmp_path, image_dir(root, DATES[0]))
    opened_run(tmp_path, drawn, experiment_id="exp-drawn")
    resp = client.post(
        "/api/training/runs",
        json={"relaunched_from": "exp-drawn", "user": "tester"},
    )
    assert resp.status_code == 200, resp.json()
    assert (
        _relaunched_config(tmp_path, resp).config["data"]
        == train_config(drawn).data.record()
    )

    _draw(tmp_path, root, tmp_path / "manifest")
    bound = _bound_config(root, tmp_path / "manifest", tmp_path)
    opened_run(tmp_path, bound, experiment_id="exp-bound")
    resp = client.post(
        "/api/training/runs",
        json={"relaunched_from": "exp-bound", "user": "tester"},
    )
    assert resp.status_code == 200, resp.json()
    assert (
        _relaunched_config(tmp_path, resp).config["data"]
        == train_config(bound).data.record()
    )


def _relaunched_config(project: Path, response: Response) -> RunRecord:
    """Return the launch record of the new run of ``project`` a relaunch response names."""
    from tcip_mcp.experiments import experiment_dir, observe_run

    return observe_run(
        experiment_dir(response.json()["experiment_id"], project=project)
    ).record


def test_relaunch_route_launches_a_new_run_with_the_chosen_partition(
    tmp_path: Path,
    opened_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
) -> None:
    """The relaunch route launches a new run bound to the chosen partition.

    A launch with a chosen partition submits a string the server itself listed; the new run's
    launch record states that candidate, data.split replaced wholesale and the stated class
    scope dropped, and names the run it was relaunched from, whose own record is untouched.
    """
    monkeypatch.chdir(tmp_path)
    fake_popen(monkeypatch, [])
    from tcip_mcp.experiments import RUN_FILE, experiment_dir, read_record

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    chosen = root / "splits"
    _draw(tmp_path, root, chosen)

    cfg = _bespoke_config(tmp_path, image_dir(root, DATES[0]))
    opened_run(tmp_path, cfg, experiment_id="exp-choose-partition")
    source_path = (
        experiment_dir("exp-choose-partition", project=tmp_path) / RUN_FILE
    )
    source_before = read_record(source_path)

    resp = client.post(
        "/api/training/runs",
        json={
            "relaunched_from": "exp-choose-partition",
            "selection_dir": str(chosen),
            "user": "tester",
        },
    )
    assert resp.status_code == 200, resp.json()

    launched = _relaunched_config(tmp_path, resp)
    data = launched.config["data"]
    assert isinstance(data, dict)
    assert data["split"] == {"selection_dir": str(chosen)}
    assert "scope" not in data
    assert launched.relaunched_from == "exp-choose-partition"
    assert read_record(source_path) == source_before


def test_relaunch_route_admits_a_symlinked_spelling_of_an_offered_split_directory(
    tmp_path: Path,
    opened_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
) -> None:
    """The relaunch route admits a symlinked spelling of an offered split directory.

    The picker's dedupe offers one spelling of a symlinked directory; the relaunch route admits
    the second spelling of the identical directory, and the new run records a selection whose
    canonical path is the offered one's.
    """
    monkeypatch.chdir(tmp_path)
    fake_popen(monkeypatch, [])
    from tcip_mcp.tools.training_tools import list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    real_dir = splits_dir / "real"
    _draw(tmp_path, root, real_dir, seed=5)

    link_dir = splits_dir / "link"
    try:
        os.symlink(real_dir, link_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(
            f"this platform refuses directory symlink creation for this user: {exc}"
        )

    cfg = _bespoke_config(tmp_path, image_dir(root, DATES[0]))
    opened_run(tmp_path, cfg, experiment_id="exp-symlink-relaunch")

    choices = list_split_choices(tmp_path, "exp-symlink-relaunch")
    offered = next(
        m["selection_dir"] for m in choices["selections"] if m["enabled"]
    )
    assert offered in (str(real_dir), str(link_dir))
    other_spelling = (
        str(real_dir) if offered == str(link_dir) else str(link_dir)
    )
    assert ts.canonical_path(other_spelling) == ts.canonical_path(offered)

    resp = client.post(
        "/api/training/runs",
        json={
            "relaunched_from": "exp-symlink-relaunch",
            "selection_dir": other_spelling,
            "user": "tester",
        },
    )
    assert resp.status_code == 200, resp.json()

    data = _relaunched_config(tmp_path, resp).config["data"]
    assert isinstance(data, dict)
    recorded = data["split"]
    assert isinstance(recorded, dict)
    assert set(recorded) == {"selection_dir"}
    assert ts.canonical_path(recorded["selection_dir"]) == ts.canonical_path(
        other_spelling
    )


def test_a_chosen_selection_binds_and_the_runs_own_resolved_record_names_it(
    tmp_path: Path,
) -> None:
    """A chosen selection binds, and the run's own resolved record names it.

    The candidate config the route builds for a chosen partition is resolved and opened by the
    launcher's own producer, and the run's launch record names the chosen selection as the
    partition it bound to.
    """
    from tcip_mcp.experiments import observe_run

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    chosen = tmp_path / "chosen"
    _draw(tmp_path, root, chosen, seed=5)
    stated = _bespoke_config(tmp_path, image_dir(root, DATES[0]))

    run_dir = opened_run(
        tmp_path,
        _chosen(stated, str(chosen), tmp_path),
        experiment_id="exp-bound-split-record",
    )

    binding = observe_run(run_dir).record.resolution.partition.selection
    assert binding is not None and binding.selection_dir == str(chosen)
