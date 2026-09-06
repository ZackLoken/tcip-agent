"""Tests for tcip write-project-site, run by subprocess against a project on disk.

The command is the one door that records a site for a project whose name the scheme refuses, and
the sanctioned path for conforming a project that predates the record; every project here is
scaffolded through the platform's own doors first.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tcip_web.cli", "write-project-site", *args],
        capture_output=True, text=True, timeout=60,
    )


def test_conform_script_writes_a_fresh_site_for_a_project_with_no_record(tmp_path: Path):
    from tcip_mcp.tools.meta_tools import report_friction

    project = tmp_path / "bare"
    project.mkdir()
    report_friction(str(project), category="missing_tool", detail="probe")

    result = _run_script(str(project), "north orchard")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "written" in result.stdout


def test_conform_script_reports_already_recorded_the_same(tmp_path: Path):
    from tcip_mcp.tools.project_tools import initialize_project

    project = tmp_path / "proj"
    initialize_project(str(project), site="north orchard")

    result = _run_script(str(project), "north orchard")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "already recorded the same" in result.stdout


def test_conform_script_refuses_a_conflicting_site_without_replace(tmp_path: Path):
    from tcip_mcp.project_record import read_record
    from tcip_mcp.tools.project_tools import initialize_project

    project = tmp_path / "proj"
    initialize_project(str(project), site="north orchard")

    result = _run_script(str(project), "south orchard")

    assert result.returncode != 0
    assert "refused" in result.stdout
    assert read_record(str(project))["site"] == "north orchard"  # nothing written


def test_conform_script_replaces_a_conflicting_site(tmp_path: Path):
    from tcip_mcp.project_record import read_record
    from tcip_mcp.tools.project_tools import initialize_project

    project = tmp_path / "proj"
    initialize_project(str(project), site="north orchard")

    result = _run_script(str(project), "south orchard", "--replace")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "replaced" in result.stdout
    assert read_record(str(project))["site"] == "south orchard"


def test_conform_script_replaces_a_damaged_record_naming_it_replaced_not_written(
    tmp_path: Path,
):
    """A prior record that existed but could not be read as a site is a replacement, not a
    fresh write: the breeder's earlier value existed and this call is what corrected it."""
    import tcip_store

    from tcip_mcp.project_record import project_record_key, read_record
    from tcip_mcp.tools.project_tools import initialize_project

    project = tmp_path / "proj"
    initialize_project(str(project), site="north orchard")
    key = project_record_key(str(project))
    current = tcip_store.read_versioned(key).version
    tcip_store.replace(key, {"not_site": "x"}, expect=current)

    result = _run_script(str(project), "south orchard", "--replace")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "replaced" in result.stdout
    assert "written:" not in result.stdout
    assert "does not hold a site" in result.stdout
    assert read_record(str(project))["site"] == "south orchard"
