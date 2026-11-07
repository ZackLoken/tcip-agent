"""``tcip diagnose-project`` reads the split selections a project's ``splits`` directory holds.

Each selection is drawn through ``draw_splits``; the one an error finding must name is then
re-keyed by the store's own writer, so the project's records are never hand-built.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import tcip_store as ts
from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.pipelines.data.selection import selection_key, selection_place
from tcip_mcp.tools.training_tools import (
    UnreadSelectionChoice,
    list_split_choices,
)
from tests._cli_fixtures import run_tcip
from tests._verified_checkpoint_fixtures import opened_run
from tests._web_fixtures import new_project
from tests.test_data_picker import _bespoke_config
from tests.test_selection_binding import (
    DATES,
    _draw,
    _two_subject_two_date_dataset,
)


def _project_with_selections(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Return a project holding a dataset and two selections drawn under its ``splits``.

    The project is made through the creation door and is its own dataset root. The two
    selections are drawn by ``draw_splits`` into ``splits`` and one level under it; the second
    is the one a test re-keys.
    """
    project = tmp_path / "proj"
    new_project(project)
    _two_subject_two_date_dataset(project)
    intact = project / "splits"
    rekeyed = intact / "frozen-x"
    _draw(project, project, intact)
    _draw(project, project, rekeyed, seed=3)
    return project, intact, rekeyed


def _without_extent(selection_dir: Path) -> None:
    """Re-key the selection stored under ``selection_dir`` with no sample carrying ``extent``."""
    document = ts.read(selection_key(selection_dir))
    assert isinstance(document, dict)
    samples = document["samples"]
    assert isinstance(samples, list) and samples
    for sample in samples:
        assert isinstance(sample, dict)
        del sample["extent"]
    ts.replace(selection_key(selection_dir), document)


def _naming(found: list[list[str]], selection_dir: Path) -> list[str]:
    """Return the level of each finding naming the selection at ``selection_dir`` as its place."""
    place = f"{selection_place(selection_dir)}:"
    return [level for level, msg in found if place in msg]


def _findings(project: Path) -> list[list[str]]:
    """Return the findings ``tcip diagnose-project`` prints over ``project``."""
    res = run_tcip(
        "diagnose-project", ["--project", str(project), "--user", "tester"]
    )
    assert "Traceback" not in res.stderr, res.stderr
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)["findings"]


def test_the_doctor_command_reports_a_selection_that_will_not_read(
    tmp_path: Path,
) -> None:
    """The doctor's command reports a re-keyed selection once as an error naming its place.

    The selection beside it, untouched, adds no finding.
    """
    project, intact, rekeyed = _project_with_selections(tmp_path)
    _without_extent(rekeyed)

    found = _findings(project)

    assert _naming(found, rekeyed) == ["error"], found
    assert any("extent" in msg for _, msg in found), found
    assert not _naming(found, intact), found


def test_the_doctor_command_reports_no_selection_finding_for_readable_selections(
    tmp_path: Path,
) -> None:
    """The doctor's command names no selection when every selection under ``splits`` reads."""
    project, intact, rekeyed = _project_with_selections(tmp_path)

    found = _findings(project)

    assert not _naming(found, intact) and not _naming(found, rekeyed), found


def test_the_split_picker_lists_the_selection_the_doctor_reports(
    tmp_path: Path,
) -> None:
    """The picker and the doctor see the same selection directories under ``splits``.

    The re-keyed selection is an unreadable choice in the picker and an error finding in the
    doctor, and the untouched one is a readable choice and no finding.
    """
    project, intact, rekeyed = _project_with_selections(tmp_path)
    _without_extent(rekeyed)
    opened_run(
        project,
        _bespoke_config(project, image_dir(project, DATES[0])),
        experiment_id="exp-picked",
    )

    choices = {
        c.selection_dir: c
        for c in list_split_choices(project, "exp-picked").selections
    }
    found = _findings(project)

    assert isinstance(choices[str(rekeyed)], UnreadSelectionChoice)
    assert not isinstance(choices[str(intact)], UnreadSelectionChoice)
    assert _naming(found, rekeyed) == ["error"], found
    assert not _naming(found, intact), found
