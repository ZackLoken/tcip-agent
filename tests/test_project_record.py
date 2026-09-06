"""Tests for the project record: the authored site, its create-only write, and its readers.

``tcip_mcp.project_record`` is new: every symbol from it is imported inside each test function
rather than at module level, so a fail-before run against a tree that predates the module fails
on that one test's own assertions rather than on collection for the whole file.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_store.file_backend import FileBackend
from tcip_store.sqlite_backend import SqliteBackend


@contextmanager
def bound(backend):
    """Bind one backend for a block, putting the suite's own back on the way out."""
    from tcip_store.store import _backend

    previous = _backend()
    ts.bind(backend)
    try:
        yield backend
    finally:
        ts.bind(previous)
        backend.close()


# ── site validation ─────────────────────────────────────────────────────────────


def test_record_site_refuses_a_non_string(tmp_path: Path):
    from tcip_mcp.project_record import record_site

    with pytest.raises(ValueError, match="must be a string"):
        record_site(str(tmp_path), 42)  # type: ignore[arg-type]


def test_record_site_refuses_an_empty_or_whitespace_only_site(tmp_path: Path):
    from tcip_mcp.project_record import record_site

    with pytest.raises(ValueError, match="empty"):
        record_site(str(tmp_path), "   ")


def test_record_site_strips_surrounding_whitespace_before_storing(tmp_path: Path):
    from tcip_mcp.project_record import read_record, record_site

    recorded = record_site(str(tmp_path), "  north orchard  ")
    assert recorded["site"] == "north orchard"
    assert read_record(str(tmp_path))["site"] == "north orchard"


def test_record_site_refuses_a_non_printable_character_naming_the_code_point_and_offset(
    tmp_path: Path,
):
    from tcip_mcp.project_record import record_site

    with pytest.raises(ValueError, match=r"U\+0009 at offset 5") as raised:
        record_site(str(tmp_path), "north\torchard")
    assert "U+0009" in str(raised.value)


def test_record_site_refuses_a_site_over_the_length_bound_naming_the_length(tmp_path: Path):
    from tcip_mcp.project_record import record_site

    long_site = "a" * 201
    with pytest.raises(ValueError, match="201 characters"):
        record_site(str(tmp_path), long_site)


def test_record_site_admits_a_site_at_exactly_the_length_bound(tmp_path: Path):
    from tcip_mcp.project_record import record_site

    site = "a" * 200
    recorded = record_site(str(tmp_path), site)
    assert recorded["site"] == site


# ── record_site: the states of the create-only write ────────────────────────────


def test_record_site_writes_an_absent_record(tmp_path: Path):
    from tcip_mcp.project_record import read_record, record_site

    recorded = record_site(str(tmp_path), "north orchard")
    assert recorded == {"site": "north orchard", "previous_site": None}
    assert read_record(str(tmp_path)) == {"site": "north orchard"}


def test_record_site_is_a_no_op_when_the_same_site_is_offered_again(tmp_path: Path):
    """A second season of the same site: the ordinary re-run, once with trailing whitespace."""
    from tcip_mcp.project_record import read_record, record_site

    record_site(str(tmp_path), "north orchard")

    again = record_site(str(tmp_path), "north orchard  ")

    assert again == {"site": "north orchard", "previous_site": "north orchard"}
    assert read_record(str(tmp_path)) == {"site": "north orchard"}


def test_record_site_refuses_a_different_site_naming_both_and_the_project(tmp_path: Path):
    from tcip_mcp.project_record import SiteConflict, read_record, record_site

    record_site(str(tmp_path), "north orchard")

    with pytest.raises(SiteConflict) as raised:
        record_site(str(tmp_path), "south orchard")

    message = str(raised.value)
    assert "north orchard" in message
    assert "south orchard" in message
    assert str(tmp_path) in message
    assert read_record(str(tmp_path))["site"] == "north orchard"  # nothing written over it


def test_record_site_refuses_a_present_record_that_is_not_a_site_record(tmp_path: Path):
    from tcip_mcp.project_record import ProjectRecordInvalid, project_record_key, record_site

    key = project_record_key(str(tmp_path))
    ts.replace(key, {"not_site": "whatever"}, expect=ts.Version.ABSENT)

    with pytest.raises(ProjectRecordInvalid, match="does not hold a site"):
        record_site(str(tmp_path), "north orchard")


def test_record_site_lets_a_decode_error_through_for_a_present_undecodable_record(
    tmp_path: Path,
):
    from tcip_mcp.project_record import project_record_key, record_site
    from tests._record_damage_fixtures import damage_record

    key = project_record_key(str(tmp_path))
    ts.replace(key, {"site": "north orchard"}, expect=ts.Version.ABSENT)
    damage_record(key, b"{not valid json")

    with pytest.raises(ts.DecodeError):
        record_site(str(tmp_path), "north orchard")


def test_record_site_refuses_writing_over_an_unadopted_root(tmp_path: Path):
    """A root whose project record is still a loose file (no database) refuses, naming the
    conform script, the same rule every other record store obeys under this root."""
    from tcip_mcp.project_record import record_site

    with bound(FileBackend()):
        record_site(str(tmp_path), "north orchard")

    with bound(SqliteBackend()):
        with pytest.raises(ts.StoreError, match="tcip adopt-store"):
            record_site(str(tmp_path), "north orchard")


# ── record_site(replace=True): the operator script's one deliberate overwrite ───


def test_record_site_replace_writes_fresh_when_nothing_was_there(tmp_path: Path):
    from tcip_mcp.project_record import record_site

    recorded = record_site(str(tmp_path), "north orchard", replace=True)
    assert recorded == {
        "site": "north orchard", "previous_site": None, "previous_record_problem": None,
    }


def test_record_site_replace_overwrites_a_valid_record_and_reports_the_previous_site(
    tmp_path: Path,
):
    from tcip_mcp.project_record import read_record, record_site

    record_site(str(tmp_path), "north orchard")

    recorded = record_site(str(tmp_path), "south orchard", replace=True)

    assert recorded == {
        "site": "south orchard", "previous_site": "north orchard",
        "previous_record_problem": None,
    }
    assert read_record(str(tmp_path))["site"] == "south orchard"


def test_record_site_replace_overwrites_an_invalid_record_naming_the_problem(tmp_path: Path):
    """A prior record that existed but could not be read as a site is a replacement, not a
    fresh write: ``previous_record_problem`` names it, distinct from ``previous_site`` being
    unset for a truly new project."""
    from tcip_mcp.project_record import project_record_key, read_record, record_site

    key = project_record_key(str(tmp_path))
    ts.replace(key, {"not_site": "whatever"}, expect=ts.Version.ABSENT)

    recorded = record_site(str(tmp_path), "north orchard", replace=True)

    assert recorded["site"] == "north orchard"
    assert recorded["previous_site"] is None
    assert "does not hold a site" in recorded["previous_record_problem"]
    assert read_record(str(tmp_path))["site"] == "north orchard"


# ── read_record ──────────────────────────────────────────────────────────────────


def test_read_record_raises_missing_and_publishes_no_database_for_a_root_with_no_store(
    tmp_path: Path,
):
    """Names both doors: ``initialize_project`` for a project whose name fits the workspace scheme,
    and the operator script for any project, since ``initialize_project`` cannot serve one that
    doesn't (``gui-smoke-scratch``-shaped names). The no-database assertion holds the store's
    own guarantee at this surface: a read of an absent record never publishes a database,
    which ``read_record`` relies on rather than re-checking itself."""
    from tcip_store.file_backend import database_file

    from tcip_mcp.project_record import ProjectRecordMissing, read_record

    with pytest.raises(ProjectRecordMissing, match="initialize_project") as raised:
        read_record(str(tmp_path))
    assert "tcip write-project-site" in str(raised.value)

    assert not database_file(str(tmp_path.absolute())).is_file()


def test_project_record_path_is_the_dotted_tcip_document(tmp_path: Path):
    from tcip_mcp.project_record import project_record_path

    assert project_record_path(str(tmp_path)) == tmp_path / ".tcip" / "project.json"


# ── site_fields: never raises, exactly one field set ─────────────────────────────


def test_site_fields_reports_the_recorded_site(tmp_path: Path):
    from tcip_mcp.project_record import record_site, site_fields

    record_site(str(tmp_path), "north orchard")

    fields = site_fields(str(tmp_path))

    assert fields == {"site": "north orchard", "site_problem": None}


def test_site_fields_names_the_absent_record(tmp_path: Path):
    from tcip_mcp.project_record import site_fields

    fields = site_fields(str(tmp_path))

    assert fields["site"] is None
    assert "initialize_project" in fields["site_problem"]
    assert "tcip write-project-site" in fields["site_problem"]


def test_site_fields_names_a_present_but_invalid_record(tmp_path: Path):
    from tcip_mcp.project_record import project_record_key, site_fields

    key = project_record_key(str(tmp_path))
    ts.replace(key, {"not_site": "whatever"}, expect=ts.Version.ABSENT)

    fields = site_fields(str(tmp_path))

    assert fields["site"] is None
    assert "does not hold a site" in fields["site_problem"]


def test_site_fields_names_an_undecodable_record(tmp_path: Path):
    from tcip_mcp.project_record import project_record_key, site_fields
    from tests._record_damage_fixtures import damage_record

    key = project_record_key(str(tmp_path))
    ts.replace(key, {"site": "north orchard"}, expect=ts.Version.ABSENT)
    damage_record(key, b"{not valid json")

    fields = site_fields(str(tmp_path))

    assert fields["site"] is None
    assert "does not decode" in fields["site_problem"]


def test_site_fields_names_a_root_the_store_refuses_to_read(tmp_path: Path):
    """A database already exists for this root (from some other store) but has never held
    ``project_record``, and ``project.json`` arrives beside it outside the seam: the conform
    rail refuses the file as a claimed document no export can explain, naming the operator
    script. A fresh backend instance is required to force re-verification against the file
    that arrived after the first connection was opened (test_store_conform_rail.py's own
    ``LATE_ARRIVAL`` cases use the same two-instance shape)."""
    from tcip_store.registry import RECORD_JSON

    from tcip_mcp.project_record import project_record_path, site_fields
    from tcip_mcp.project_status import record_report

    with bound(SqliteBackend()):
        record_report(tmp_path)  # forces a database to exist, holding project_status only

    raw_path = project_record_path(tmp_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(RECORD_JSON.encode({"site": "north orchard"}))

    with bound(SqliteBackend()):
        fields = site_fields(str(tmp_path))

    assert fields["site"] is None
    assert "tcip adopt-store" in fields["site_problem"]


def test_site_fields_on_an_unadopted_root_names_tcip_adopt_store(tmp_path: Path):
    """A root whose records are still loose files: ``tcip adopt-store`` is the state it
    conforms. The file backend legitimately produces that state (``import_project`` no longer
    does: it adopts a fresh root under the database backend), so the unadopted root here is
    built by writing through the file backend directly, through ``initialize_project``, never by
    hand-writing ``project.json``."""
    from tcip_mcp.project_record import site_fields
    from tcip_mcp.tools.project_tools import initialize_project

    dest = tmp_path / "unadopted"
    with bound(FileBackend()):
        initialize_project(str(dest), site="north orchard")

    with bound(SqliteBackend()):
        fields = site_fields(str(dest))

    assert fields["site"] is None
    assert "tcip adopt-store" in fields["site_problem"]


def test_site_fields_on_a_bare_directory_that_gained_tcip_with_no_creating_door(
    tmp_path: Path,
):
    """A store write with no door (``report_friction`` on a directory neither ``initialize_project`` nor
    ``ingest_images`` ever touched) leaves ``.tcip`` with no project record: a permanent,
    reachable state every reader has to name honestly rather than crash on."""
    from tcip_mcp.project_record import site_fields
    from tcip_mcp.tools.meta_tools import report_friction

    project = tmp_path / "bare"
    project.mkdir()
    result = report_friction(str(project), category="missing_tool", detail="probe")
    assert "error" not in result
    assert (project / ".tcip").is_dir()

    fields = site_fields(str(project))

    assert fields["site"] is None
    assert "initialize_project" in fields["site_problem"]
