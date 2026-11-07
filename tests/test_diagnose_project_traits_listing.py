"""``tcip diagnose-project`` reports a traits store that will not list as its own finding.

The project and its trait are made by the platform's own writers; the traits store's database,
a file of its own under the project's state directory, is then overwritten, so the doctor meets
a store the platform produced and a later edit spoiled.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import tcip_store as ts
from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.project_paths import project_state_dir
from tcip_store.file_backend import database_file
from tests import _trait_fixtures as fx
from tests._producer_fixtures import write_image
from tests._web_fixtures import new_project
from tests.test_diagnose_project_selections import _findings

NOT_COMPLETED = "could not be completed"
"""What the doctor's finding says of a check whose listing the store refused."""


def _project_with_a_trait_and_an_unlabeled_image(tmp_path: Path) -> Path:
    """Return a project holding one proposed trait and one image with no label record."""
    project = tmp_path / "proj"
    new_project(project)
    write_image(image_dir(project, "2026-03-04") / "a.png", (32, 32))
    fx.propose(project, fx.entry("leaf", ("leaf_length",)))
    return project


def test_the_doctor_command_reports_a_traits_store_that_will_not_list_and_runs_the_other_checks(
    tmp_path: Path,
) -> None:
    """A traits database that fails the store's identity check is one error finding.

    The finding names the traits check as not completed, and the census over the project's
    images still reports the image with no label record.
    """
    project = _project_with_a_trait_and_an_unlabeled_image(tmp_path)
    state = str(project_state_dir(project))
    ts.release_root(state)
    database = database_file(state)
    assert database.is_file()
    database.write_bytes(b"not a database, only bytes under its name" * 100)

    found = _findings(project)

    assert [
        level
        for level, msg in found
        if "traits" in msg and NOT_COMPLETED in msg
    ] == ["error"], found
    assert [level for level, msg in found if "no label record" in msg] == [
        "info"
    ], found


def test_the_doctor_command_reports_no_incomplete_check_over_a_traits_store_that_lists(
    tmp_path: Path,
) -> None:
    """A readable traits store adds no incomplete-check finding and its trait is still checked."""
    project = _project_with_a_trait_and_an_unlabeled_image(tmp_path)

    found = _findings(project)

    assert not [msg for _, msg in found if NOT_COMPLETED in msg], found
    assert [level for level, msg in found if "'leaf'" in msg] == ["warn"], (
        found
    )
