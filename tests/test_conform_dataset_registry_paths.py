"""Tests for scripts/conform_dataset_registry_paths.py, run by subprocess against a project.

register_dataset no longer writes an absolute path for a project's own dataset, so the shape
this script conforms cannot be produced by any writer any more; the entries below are
hand-built, standing in for a project registered before that change, the same treatment
test_store_conform_rail.py's database-plus-never-held sibling gives a shape no producer makes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import tcip_store as ts

from tcip_mcp.tools.project_tools import dataset_registry_key

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "conform_dataset_registry_paths.py"


def _run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True, text=True, timeout=60,
    )


def test_conform_script_makes_the_projects_own_dataset_entry_relative(tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    ts.replace(
        dataset_registry_key(project),
        [{"id": "abc123", "path": str(project), "crop": "currant", "fingerprint": "f"}],
        expect=ts.Version.ABSENT,
    )

    result = _run_script(str(project))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "->" in result.stdout
    entries = ts.read(dataset_registry_key(project))
    assert entries[0]["path"] == "."


def test_conform_script_leaves_an_external_dataset_absolute(tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    external = tmp_path / "external_dataset"
    external.mkdir()
    ts.replace(
        dataset_registry_key(project),
        [{"id": "abc123", "path": str(external), "crop": "currant", "fingerprint": "f"}],
        expect=ts.Version.ABSENT,
    )

    result = _run_script(str(project))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "nothing to conform" in result.stdout
    entries = ts.read(dataset_registry_key(project))
    assert entries[0]["path"] == str(external)


def test_conform_script_is_idempotent(tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    ts.replace(
        dataset_registry_key(project),
        [{"id": "abc123", "path": str(project), "crop": "currant", "fingerprint": "f"}],
        expect=ts.Version.ABSENT,
    )

    first = _run_script(str(project))
    second = _run_script(str(project))

    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    assert "nothing to conform" in second.stdout
