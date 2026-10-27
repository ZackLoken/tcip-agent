"""Tests for the outcomes of ``tcip check-dataset-identity``.

The command prints the verdict ``check_dataset_identity`` answers, its state one of unchanged,
unregistered, changed and never recorded, and exits 0 for each, the verdict being the answer;
each path the project's registry holds for the id reads moved or gone where the dataset is no
longer there.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import tcip_store as ts
from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.tools.project_tools import register_dataset
from tests._cli_fixtures import run_tcip
from tests._producer_fixtures import label_image, write_image
from tests._web_fixtures import new_project


def _real_dataset(root: Path) -> None:
    image = write_image(
        image_dir(root, "2024-01-01") / "a.png", (10, 10), (1, 2, 3)
    )
    label_image(image, [], 10, 10, keep_empty=True)


def _registered(
    tmp_path: Path, *, content: bool = True
) -> tuple[Path, Path, dict]:
    """Return a project and a dataset beside it registered to it through ``register_dataset``."""
    project = new_project(tmp_path / "project").root
    root = tmp_path / "dataset"
    root.mkdir()
    if content:
        _real_dataset(root)
    result = register_dataset(project, str(root), "chestnut", actor=None)
    assert "error" not in result, result
    return project, root, result


def _check(root: Path, project: Path) -> dict:
    """Run the command over ``root`` for ``project``; return the verdict it printed, exiting 0."""
    completed = run_tcip(
        "check-dataset-identity",
        [
            *("--dataset-root", str(root)),
            *("--project", str(project), "--user", "tester"),
        ],
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout)


def test_a_matching_fingerprint_reports_unchanged(tmp_path: Path) -> None:
    project, root, _ = _registered(tmp_path)

    verdict = _check(root, project)

    assert verdict["state"] == "unchanged"
    assert [entry["state"] for entry in verdict["registered"]] == ["in_place"]


def test_content_carrying_no_identity_reports_unregistered(
    tmp_path: Path,
) -> None:
    project = new_project(tmp_path / "project").root
    root = tmp_path / "dataset"
    _real_dataset(root)

    verdict = _check(root, project)

    assert verdict["state"] == "unregistered"
    assert verdict["reason"]


def test_a_real_content_change_still_reports_changed(tmp_path: Path) -> None:
    project, root, _ = _registered(tmp_path)
    image = write_image(
        image_dir(root, "2024-01-01") / "b.png", (10, 10), (4, 5, 6)
    )
    label_image(image, [], 10, 10, keep_empty=True)

    verdict = _check(root, project)

    assert verdict["state"] == "changed"
    assert verdict["fingerprint"] != verdict["identity"]["fingerprint"]


def test_a_never_recorded_fingerprint_is_its_own_outcome(
    tmp_path: Path,
) -> None:
    project, root, result = _registered(tmp_path, content=False)
    assert result["fingerprint"] is None
    # Real content shows up after the fingerprint-less registration.
    _real_dataset(root)

    verdict = _check(root, project)

    assert verdict["state"] == "never_recorded"


def test_the_same_identity_at_another_path_is_reported_moved(
    tmp_path: Path,
) -> None:
    """A copy carrying the identity read at the registered path is reported moved.

    The verdict names the path the id is registered at.
    """
    project, root, result = _registered(tmp_path)
    copy = tmp_path / "copied"
    ts.release_root(root)
    shutil.copytree(root, copy)

    verdict = _check(copy, project)

    (entry,) = verdict["registered"]
    assert (entry["path"], entry["state"]) == (str(root), "moved")
    assert verdict["identity"]["id"] == result["id"]


def test_a_registered_path_holding_no_identity_is_reported_gone(
    tmp_path: Path,
) -> None:
    """A registered path holding no identity is reported gone.

    A dataset moved away from its registered path is reported gone at that path, naming why.
    """
    project, root, _ = _registered(tmp_path)
    moved = tmp_path / "moved"
    ts.release_root(root)
    shutil.move(str(root), str(moved))

    verdict = _check(moved, project)

    (entry,) = verdict["registered"]
    assert (entry["path"], entry["state"]) == (str(root), "gone")
    assert entry["unreadable"]
