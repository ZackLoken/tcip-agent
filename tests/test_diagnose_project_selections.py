"""``tcip diagnose-project`` reads the split selections a project's ``splits`` directory holds.

Each selection is drawn through ``draw_splits``; the ones an error finding must name are then
damaged by the store's own writer or by overwriting their database, so the project's records are
never hand-built.
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
from tcip_store.file_backend import database_file
from tests._cli_fixtures import run_tcip
from tests._verified_checkpoint_fixtures import opened_run
from tests._web_fixtures import new_project
from tests.test_data_picker import _bespoke_config
from tests.test_selection_binding import (
    DATES,
    _draw,
    _two_subject_two_date_dataset,
)

FINDING_PREFIX = "selection will not read: "
"""What the doctor's finding says before the refusal of a selection that will not read."""


def _project_with_selections(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Return a project holding a dataset and three selections drawn under its ``splits``.

    The project is made through the creation door and is its own dataset root. The selections
    are drawn by ``draw_splits`` into ``splits`` and, as ``frozen-x`` and ``damaged``, one level
    under it; a test damages the last two. Returns the project and the three directories.
    """
    project = tmp_path / "proj"
    new_project(project)
    _two_subject_two_date_dataset(project)
    intact = project / "splits"
    rekeyed = intact / "frozen-x"
    damaged = intact / "damaged"
    _draw(project, project, intact)
    _draw(project, project, rekeyed, seed=3)
    _draw(project, project, damaged, seed=4)
    return project, intact, rekeyed, damaged


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


def _unopenable(selection_dir: Path) -> None:
    """Overwrite the database of the selection under ``selection_dir`` with non-database bytes."""
    ts.release_root(selection_dir)
    database_file(str(selection_dir)).write_bytes(
        b"not a database, only bytes under its name" * 100
    )


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

    The selections beside it, untouched, add no finding.
    """
    project, intact, rekeyed, damaged = _project_with_selections(tmp_path)
    _without_extent(rekeyed)

    found = _findings(project)

    assert _naming(found, rekeyed) == ["error"], found
    assert not _naming(found, intact) and not _naming(found, damaged), found


def test_the_doctor_command_reports_a_selection_whose_database_will_not_open(
    tmp_path: Path,
) -> None:
    """The doctor's command answers, with an error finding, over a non-database store file.

    The finding names the selection's place, and the selections beside it add none.
    """
    project, intact, rekeyed, damaged = _project_with_selections(tmp_path)
    _unopenable(damaged)

    found = _findings(project)

    assert _naming(found, damaged) == ["error"], found
    assert not _naming(found, intact) and not _naming(found, rekeyed), found


def test_the_doctor_command_reports_no_selection_finding_for_readable_selections(
    tmp_path: Path,
) -> None:
    """The doctor's command names no selection when every selection under ``splits`` reads."""
    project, *selections = _project_with_selections(tmp_path)

    found = _findings(project)

    assert not [msg for _, msg in found if msg.startswith(FINDING_PREFIX)], (
        found
    )
    assert not [d for d in selections if _naming(found, d)], found


def test_the_split_picker_lists_the_selections_the_doctor_reports(
    tmp_path: Path,
) -> None:
    """The picker's unread reasons and the doctor's selection findings are the same set.

    One selection is re-keyed and one has a database that will not open; each is an
    ``UnreadSelectionChoice`` in the picker and an error finding in the doctor, with the same
    refusal behind its place.
    """
    project, intact, rekeyed, damaged = _project_with_selections(tmp_path)
    _without_extent(rekeyed)
    _unopenable(damaged)
    opened_run(
        project,
        _bespoke_config(project, image_dir(project, DATES[0])),
        experiment_id="exp-picked",
    )

    choices = list_split_choices(project, "exp-picked").selections
    unread = {
        f"{FINDING_PREFIX}{c.reason}"
        for c in choices
        if isinstance(c, UnreadSelectionChoice)
    }
    found = _findings(project)
    reported = {msg for _, msg in found if msg.startswith(FINDING_PREFIX)}

    assert len(unread) == 2
    assert reported == unread, found
    assert {level for level, msg in found if msg in reported} == {"error"}
