"""Tests for the project record: its minted id, display name and site, its create-only write, and
its readers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts


# ── text validation ─────────────────────────────────────────────────────────────


def test_validate_text_refuses_a_non_string():
    from tcip_mcp.project_record import validate_text

    with pytest.raises(ValueError, match="must be a string"):
        validate_text("site", 42)


def test_validate_text_refuses_an_empty_or_whitespace_only_value():
    from tcip_mcp.project_record import validate_text

    with pytest.raises(ValueError, match="empty"):
        validate_text("display name", "   ")


def test_validate_text_strips_surrounding_whitespace():
    from tcip_mcp.project_record import validate_text

    assert validate_text("site", "  north orchard  ") == "north orchard"


def test_validate_text_refuses_a_non_printable_character_naming_the_code_point_and_offset():
    from tcip_mcp.project_record import validate_text

    with pytest.raises(ValueError, match=r"U\+0009 at offset 5"):
        validate_text("site", "north\torchard")


def test_validate_text_refuses_a_value_over_the_length_bound_and_admits_one_at_it():
    from tcip_mcp.project_record import validate_text

    with pytest.raises(ValueError, match="201 characters"):
        validate_text("site", "a" * 201)
    assert validate_text("site", "a" * 200) == "a" * 200


# ── create_record: the states of the create-only write ──────────────────────────


def test_create_record_mints_an_id_and_writes_an_absent_record(tmp_path: Path):
    from tcip_mcp.project_record import create_record, read_record

    record = create_record(tmp_path, "  Valley block ", "north orchard")

    assert record["display_name"] == "Valley block" and record["site"] == "north orchard"
    assert len(record["id"]) == 12
    assert read_record(tmp_path) == record


def test_create_record_answers_the_existing_record_when_offered_it_again(tmp_path: Path):
    """Creating a project that already records the same name and site keeps its id."""
    from tcip_mcp.project_record import create_record

    first = create_record(tmp_path, "Valley block", "north orchard")

    assert create_record(tmp_path, "Valley block", "north orchard  ") == first


def test_create_record_refuses_a_different_site_naming_both(tmp_path: Path):
    from tcip_mcp.project_record import SiteConflictError, create_record, read_record

    create_record(tmp_path, "Valley block", "north orchard")

    with pytest.raises(SiteConflictError) as raised:
        create_record(tmp_path, "Valley block", "south orchard")

    assert "north orchard" in str(raised.value) and "south orchard" in str(raised.value)
    assert read_record(tmp_path)["site"] == "north orchard"


def test_create_record_refuses_a_different_display_name_naming_rename(tmp_path: Path):
    from tcip_mcp.project_record import create_record, read_record

    create_record(tmp_path, "Valley block", "north orchard")

    with pytest.raises(ValueError, match="rename"):
        create_record(tmp_path, "Hill block", "north orchard")

    assert read_record(tmp_path)["display_name"] == "Valley block"


def test_create_record_refuses_a_present_record_missing_a_field(tmp_path: Path):
    from tcip_mcp.project_record import ProjectRecordInvalidError, create_record, project_record_key

    ts.replace(project_record_key(tmp_path), {"site": "north orchard"},
               expect=ts.Version.ABSENT)

    with pytest.raises(ProjectRecordInvalidError, match="does not hold an id"):
        create_record(tmp_path, "Valley block", "north orchard")


def test_create_record_lets_a_decode_error_through_for_an_undecodable_record(tmp_path: Path):
    from tcip_mcp.project_record import create_record, project_record_key
    from tests._record_damage_fixtures import damage_record

    create_record(tmp_path, "Valley block", "north orchard")
    damage_record(project_record_key(tmp_path), b"{not valid json")

    with pytest.raises(ts.DecodeError):
        create_record(tmp_path, "Valley block", "north orchard")


# ── replace_site: the one deliberate correction ─────────────────────────────────


def test_replace_site_corrects_the_site_keeping_the_id_and_display_name(tmp_path: Path):
    from tcip_mcp.project_record import create_record, read_record, replace_site

    record = create_record(tmp_path, "Valley block", "north orchard")

    assert replace_site(tmp_path, "south orchard") == {
        "site": "south orchard", "previous_site": "north orchard"}
    assert read_record(tmp_path) == {**record, "site": "south orchard"}


def test_replace_site_refuses_a_project_with_no_record(tmp_path: Path):
    from tcip_mcp.project_record import ProjectRecordMissingError, replace_site

    with pytest.raises(ProjectRecordMissingError):
        replace_site(tmp_path, "south orchard")


# ── read_record and record_fields ───────────────────────────────────────────────


def test_read_record_raises_missing_and_publishes_no_database_for_a_root_with_no_store(
    tmp_path: Path,
):
    """Names the creating door; a read of an absent record never publishes a database."""
    from tcip_store.file_backend import database_file

    from tcip_mcp.project_record import ProjectRecordMissingError, read_record

    with pytest.raises(ProjectRecordMissingError, match="initialize_project"):
        read_record(tmp_path)

    assert not database_file(str(tmp_path.absolute())).is_file()


def test_record_fields_reports_the_record(tmp_path: Path):
    from tcip_mcp.project_record import create_record, record_fields

    record = create_record(tmp_path, "Valley block", "north orchard")

    assert record_fields(tmp_path) == {**record, "record_problem": None}


def test_record_fields_names_the_absent_record(tmp_path: Path):
    from tcip_mcp.project_record import record_fields

    fields = record_fields(tmp_path)

    assert fields["id"] is None and fields["site"] is None
    assert "initialize_project" in fields["record_problem"]


def test_record_fields_names_an_undecodable_record(tmp_path: Path):
    from tcip_mcp.project_record import create_record, project_record_key, record_fields
    from tests._record_damage_fixtures import damage_record

    create_record(tmp_path, "Valley block", "north orchard")
    damage_record(project_record_key(tmp_path), b"{not valid json")

    fields = record_fields(tmp_path)

    assert fields["display_name"] is None
    assert "does not decode" in fields["record_problem"]


def test_existing_project_resolves_a_project_and_refuses_a_directory_with_no_record(
    tmp_path: Path,
):
    from tcip_mcp.project_record import create_record, existing_project

    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(ValueError, match="names no readable project"):
        existing_project(bare)

    created = create_record(tmp_path, "Valley block", "north orchard")
    root, record = existing_project(tmp_path)
    assert (root, record["id"]) == (tmp_path.resolve(), created["id"])
