"""``diagnose_project`` reports a store that will not list as its own finding.

The project, its trait, its selections and its image are made by the platform's own writers; a
store's database is then overwritten, so the doctor meets a store the platform produced and a
later edit spoiled. The traits store is a database of its own under the project's state
directory, reached through the command; the project's own store holds the label, bucket, review
flags and review verdicts listings, which the command's bind reads first, so those are reached
by calling the tool.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import tcip_store as ts
from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.project_paths import project_state_dir
from tcip_mcp.tools.doctor_tools import diagnose_project
from tcip_store.file_backend import database_file
from tests import _trait_fixtures as fx
from tests._producer_fixtures import write_image
from tests._web_fixtures import new_project
from tests.test_diagnose_project_selections import (
    FINDING_PREFIX,
    _findings,
    _project_with_selections,
    _without_extent,
)

NOT_COMPLETED = "could not be completed"
"""What the doctor's finding says of a check whose listing the store refused."""

NON_DATABASE = b"not a database, only bytes under its name" * 100
"""Bytes that fail the store's identity check."""


def _overwritten(root: Path) -> None:
    """Replace the database of the store rooted at ``root`` with :data:`NON_DATABASE`."""
    ts.release_root(str(root))
    database = database_file(str(root))
    assert database.is_file()
    database.write_bytes(NON_DATABASE)


def _project_with_a_trait_and_a_selection_that_will_not_read(
    tmp_path: Path,
) -> Path:
    """Return a project holding a proposed trait and a selection that will not read."""
    project, _intact, rekeyed, _damaged = _project_with_selections(tmp_path)
    fx.propose(project, fx.entry("leaf", ("leaf_length",)))
    _without_extent(rekeyed)
    return project


def test_the_doctor_command_reports_a_traits_store_that_will_not_list_and_runs_the_other_checks(
    tmp_path: Path,
) -> None:
    """A traits database that fails the store's identity check is one error finding.

    The finding names the traits check as not completed, and the selections check, which runs
    after the traits check, still reports the selection that will not read.
    """
    project = _project_with_a_trait_and_a_selection_that_will_not_read(
        tmp_path
    )
    _overwritten(project_state_dir(project))

    found = _findings(project)

    assert [
        level
        for level, msg in found
        if "traits" in msg and NOT_COMPLETED in msg
    ] == ["error"], found
    assert [
        level for level, msg in found if msg.startswith(FINDING_PREFIX)
    ] == ["error"], found


def test_the_doctor_command_reports_no_incomplete_check_over_a_traits_store_that_lists(
    tmp_path: Path,
) -> None:
    """A readable traits store adds no incomplete-check finding and its trait is still checked."""
    project = _project_with_a_trait_and_a_selection_that_will_not_read(
        tmp_path
    )

    found = _findings(project)

    assert not [msg for _, msg in found if NOT_COMPLETED in msg], found
    assert [level for level, msg in found if "'leaf'" in msg] == ["warn"], (
        found
    )


def test_the_doctor_reports_each_listing_of_a_project_store_that_will_not_list(
    tmp_path: Path,
) -> None:
    """A project database that fails the identity check is one finding per listing it holds.

    The label census, the prediction buckets, the review flags and the review verdicts each
    report as not completed, and the census's count of images with no label record, which would
    rest on labels it could not list, is not reported.
    """
    project = tmp_path / "proj"
    new_project(project)
    write_image(image_dir(project, "2026-03-04") / "a.png", (32, 32))
    _overwritten(project)

    found = diagnose_project(project)["findings"]

    for check in (
        "label census",
        "prediction buckets",
        "review flags",
        "review verdicts",
    ):
        assert [
            level
            for level, msg in found
            if check in msg and NOT_COMPLETED in msg
        ] == ["error"], found
    assert not [msg for _, msg in found if "no label record" in msg], found
