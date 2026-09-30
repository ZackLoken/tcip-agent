"""Tests for tcip write-project-site, run by subprocess against a project on disk.

The command corrects a site typed wrong once, keeping the record's id and display name; every
project here is created through ``initialize_project`` first.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_command(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tcip_web.cli", "write-project-site", *args],
        capture_output=True, text=True, timeout=60,
    )


def test_write_project_site_replaces_the_site_and_keeps_the_id_and_name(project: Path):
    from tcip_mcp.project_record import read_record

    before = read_record(project)

    result = _run_command(str(project), "south orchard")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "written" in result.stdout
    after = read_record(project)
    assert after == {**before, "site": "south orchard"}


def test_write_project_site_refuses_a_directory_holding_no_project_record(tmp_path: Path):
    bare = tmp_path / "bare"
    bare.mkdir()

    result = _run_command(str(bare), "north orchard")

    assert result.returncode == 2
    assert "refused" in result.stdout


def test_write_project_site_refuses_a_record_that_does_not_read(project: Path):
    import tcip_store

    from tcip_mcp.project_record import project_record_key

    key = project_record_key(str(project))
    current = tcip_store.read_versioned(key).version
    tcip_store.replace(key, {"not_site": "x"}, expect=current)

    result = _run_command(str(project), "south orchard")

    assert result.returncode == 2
    assert "does not hold an id, a display name and a site" in result.stdout
    assert tcip_store.read(key) == {"not_site": "x"}
