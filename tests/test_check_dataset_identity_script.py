"""Tests for tcip check-dataset-identity's outcomes: OK, CHANGED, NEVER-RECORDED, and the identity
read at the path the project's registry holds: MOVED and GONE."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

import tcip_store as ts
from tcip_mcp.tools.project_tools import register_dataset
from tests._web_fixtures import new_project


def _run_script(root: Path, project: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tcip_web.cli", "check-dataset-identity", str(root),
         "--project", str(project)],
        capture_output=True, text=True, timeout=60,
    )


def _real_dataset(root: Path) -> None:
    images = root / "images" / "2024-01-01"
    labels = root / "annotations" / "2024-01-01"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (10, 10), (1, 2, 3)).save(images / "a.png")
    (labels / "a.json").write_text(
        '{"image": "a", "width": 10, "height": 10, "annotations": []}', encoding="utf-8"
    )


def _registered(tmp_path: Path, *, content: bool = True) -> tuple[Path, Path, dict]:
    """A project and a dataset beside it registered to it through ``register_dataset``."""
    project = new_project(tmp_path / "project")
    root = tmp_path / "dataset"
    root.mkdir()
    if content:
        _real_dataset(root)
    result = register_dataset(project, str(root), "chestnut")
    assert "error" not in result, result
    return project, root, result


def test_a_matching_fingerprint_reports_ok(tmp_path):
    project, root, _ = _registered(tmp_path)

    completed = _run_script(root, project)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OK" in completed.stdout
    assert "MOVED" not in completed.stdout and "GONE" not in completed.stdout


def test_a_real_content_change_still_reports_changed(tmp_path):
    project, root, _ = _registered(tmp_path)
    (root / "annotations" / "2024-01-01" / "b.json").write_text(
        '{"image": "b", "width": 10, "height": 10, "annotations": []}', encoding="utf-8"
    )
    Image.new("RGB", (10, 10), (4, 5, 6)).save(root / "images" / "2024-01-01" / "b.png")

    completed = _run_script(root, project)

    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert "CHANGED" in completed.stdout


def test_a_never_recorded_fingerprint_is_its_own_outcome(tmp_path):
    project, root, result = _registered(tmp_path, content=False)
    assert result["fingerprint"] is None
    # Real content shows up after the fingerprint-less registration.
    _real_dataset(root)

    completed = _run_script(root, project)

    assert completed.returncode == 4, completed.stdout + completed.stderr
    assert "NEVER-RECORDED" in completed.stdout


def test_the_same_identity_at_another_path_is_reported_moved(tmp_path):
    """The identity read at the registered path is the one the checked copy carries, so the
    dataset the registry names now also lives here."""
    project, root, result = _registered(tmp_path)
    copy = tmp_path / "copied"
    shutil.copytree(root, copy)

    completed = _run_script(copy, project)

    assert f"MOVED: id {result['id']} is registered at {root}" in completed.stdout, (
        completed.stdout + completed.stderr)


def test_a_registered_path_holding_no_identity_is_reported_gone(tmp_path):
    """The registered path is read, not compared: a dataset moved away leaves nothing readable
    there, which is its own outcome."""
    project, root, result = _registered(tmp_path)
    moved = tmp_path / "moved"
    ts.release_root(root)
    shutil.move(str(root), str(moved))

    completed = _run_script(moved, project)

    assert f"GONE: id {result['id']} is registered at {root}" in completed.stdout, (
        completed.stdout + completed.stderr)
