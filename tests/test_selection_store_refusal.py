"""The selection reader refuses a selection directory whose store will not open under its place.

Each selection is drawn through ``draw_splits``; its database is then overwritten with bytes that
are not a SQLite database.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import tcip_store as ts
from tcip_mcp.pipelines.data.selection import (
    read_selection,
    read_selection_checked,
    selection_place,
)
from tcip_mcp.tools.data_tools import freeze_selection
from tcip_store.file_backend import database_file
from tests.test_freeze_selection import _real_drawn_experiment
from tests.test_selection_binding import _draw, _two_subject_two_date_dataset

NOT_A_DATABASE = b"not a database, only bytes under its name" * 100


def _overwritten(selection_dir: Path) -> Path:
    """Overwrite the database under ``selection_dir`` with non-database bytes; return its path."""
    ts.release_root(selection_dir)
    path = database_file(str(selection_dir))
    path.write_bytes(NOT_A_DATABASE)
    return path


def test_both_readers_refuse_a_selection_whose_database_will_not_open(
    tmp_path: Path,
) -> None:
    """The checked reader answers the refusal and the strict reader raises it, each at the place."""
    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "selection"
    _draw(tmp_path, root, out)
    _overwritten(out)

    selection, refused = read_selection_checked(out, project=tmp_path)
    with pytest.raises(ts.ReadRefusedError) as strict:
        read_selection(out, project=tmp_path)

    assert selection is None and refused is not None
    assert refused.where == selection_place(out)
    assert "not a SQLite database" in str(refused)
    assert strict.value.served == refused


def test_a_drawn_selection_reads_back_through_both_readers(
    tmp_path: Path,
) -> None:
    """A selection ``draw_splits`` wrote reads as the same selection through both readers."""
    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "selection"
    drawn = _draw(tmp_path, root, out)

    checked, refused = read_selection_checked(out, project=tmp_path)

    assert refused is None
    assert checked == drawn == read_selection(out, project=tmp_path)


def test_an_absent_selection_is_no_answer_to_the_checked_reader_and_a_refusal_to_the_strict(
    tmp_path: Path,
) -> None:
    """A directory recording no selection reads as ``(None, None)``; the strict reader refuses."""
    absent = tmp_path / "nothing-here"

    assert read_selection_checked(absent, project=tmp_path) == (None, None)
    with pytest.raises(ts.ReadRefusedError) as strict:
        read_selection(absent, project=tmp_path)
    assert strict.value.served.where == selection_place(absent)
    assert "no selection recorded there" in str(strict.value)


def test_freeze_selection_answers_an_unreadable_destination_not_an_existing_selection(
    tmp_path: Path,
) -> None:
    """A destination whose database will not open is answered by its refusal and left as it is.

    A freeze of the same run into a fresh destination succeeds.
    """
    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(tmp_path, root, "exp-src")
    destination = tmp_path / "destination"
    destination.mkdir()
    ts.release_root(destination)
    database = database_file(str(destination))
    database.parent.mkdir(parents=True, exist_ok=True)
    database.write_bytes(NOT_A_DATABASE)

    refused = freeze_selection(
        tmp_path, "exp-src", output_path=str(destination), actor=None
    )
    frozen = freeze_selection(
        tmp_path, "exp-src", output_path=str(tmp_path / "fresh"), actor=None
    )

    assert selection_place(destination) in refused["error"], refused
    assert "not a SQLite database" in refused["error"], refused
    assert "already exists" not in refused["error"], refused
    again = freeze_selection(
        tmp_path, "exp-src", output_path=str(tmp_path / "fresh"), actor=None
    )

    assert database.read_bytes() == NOT_A_DATABASE
    assert "error" not in frozen, frozen
    assert "a selection already exists" in again["error"], again
