"""Tests for the project record: its minted id, display name, site and capture invariances, its
create-only write, and its readers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp.project_record import CaptureInvariances
from tests._audit_fixtures import audit_rows
from tests._web_fixtures import INVARIANCES, named_project

# ── the record's text fields ────────────────────────────────────────────────────


def test_a_record_refuses_a_non_string_site():
    from pydantic import ValidationError

    from tcip_mcp.project_record import ProjectRecord

    with pytest.raises(ValidationError, match="site"):
        ProjectRecord.model_validate(
            {
                "id": "abc123def456",
                "display_name": "Valley block",
                "site": 42,
                "invariances": INVARIANCES.model_dump(),
            }
        )


def test_create_record_refuses_a_non_printable_character_naming_the_code_point_and_offset(
    tmp_path: Path,
):
    from tcip_mcp.project_record import create_record

    with pytest.raises(ValueError, match=r"U\+0009 at offset 5"):
        create_record(tmp_path, "Valley block", " north\torchard", INVARIANCES)


def test_create_record_refuses_a_site_over_the_length_bound_and_admits_one_at_it(
    tmp_path: Path,
):
    from tcip_mcp.project_record import create_record

    with pytest.raises(ValueError, match="at most 200 characters"):
        create_record(tmp_path, "Valley block", "a" * 201, INVARIANCES)
    assert (
        create_record(tmp_path, "Valley block", "a" * 200, INVARIANCES).site
        == "a" * 200
    )


# ── create_record: the states of the create-only write ──────────────────────────


def test_create_record_mints_an_id_and_writes_an_absent_record(tmp_path: Path):
    from tcip_mcp.project_record import create_record, read_record

    record = create_record(
        tmp_path, "  Valley block ", "north orchard", INVARIANCES
    )

    assert (record.display_name, record.site) == (
        "Valley block",
        "north orchard",
    )
    assert len(record.id) == 12
    assert read_record(tmp_path) == record


def test_create_record_answers_the_existing_record_when_offered_it_again(
    tmp_path: Path,
):
    """Creating a project that already records the same name, site and invariances keeps its
    id.
    """
    from tcip_mcp.project_record import create_record

    first = create_record(
        tmp_path, "Valley block", "north orchard", INVARIANCES
    )

    assert (
        create_record(tmp_path, "Valley block", "north orchard  ", INVARIANCES)
        == first
    )


def test_create_record_refuses_a_different_site_naming_both(tmp_path: Path):
    from tcip_mcp.project_record import (
        SiteConflictError,
        create_record,
        read_record,
    )

    create_record(tmp_path, "Valley block", "north orchard", INVARIANCES)

    with pytest.raises(SiteConflictError) as raised:
        create_record(tmp_path, "Valley block", "south orchard", INVARIANCES)

    assert "north orchard" in str(raised.value) and "south orchard" in str(
        raised.value
    )
    assert read_record(tmp_path).site == "north orchard"


def test_create_record_refuses_a_different_display_name_naming_rename(
    tmp_path: Path,
):
    from tcip_mcp.project_record import create_record, read_record

    create_record(tmp_path, "Valley block", "north orchard", INVARIANCES)

    with pytest.raises(ValueError, match="rename"):
        create_record(tmp_path, "Hill block", "north orchard", INVARIANCES)

    assert read_record(tmp_path).display_name == "Valley block"


def test_create_record_refuses_a_present_record_missing_a_field(
    tmp_path: Path,
):
    from tcip_mcp.project_record import (
        ProjectRecordInvalidError,
        create_record,
        project_record_key,
    )

    ts.replace(
        project_record_key(tmp_path),
        {"site": "north orchard"},
        expect=ts.Version.ABSENT,
    )

    with pytest.raises(
        ProjectRecordInvalidError, match="is not a project record"
    ):
        create_record(tmp_path, "Valley block", "north orchard", INVARIANCES)


def test_read_record_refuses_a_record_stating_no_invariances(tmp_path: Path):
    from tcip_mcp.project_record import (
        ProjectRecordInvalidError,
        create_record,
        project_record_key,
        read_record,
    )

    record = create_record(
        tmp_path, "Valley block", "north orchard", INVARIANCES
    )
    stored = ts.read_versioned(project_record_key(tmp_path))
    ts.replace(
        project_record_key(tmp_path),
        record.model_dump(exclude={"invariances"}),
        expect=stored.version,
    )

    with pytest.raises(ProjectRecordInvalidError, match="invariances"):
        read_record(tmp_path)


def test_create_record_lets_a_decode_error_through_for_an_undecodable_record(
    tmp_path: Path,
):
    from tcip_mcp.project_record import create_record, project_record_key
    from tests._record_damage_fixtures import damage_record

    create_record(tmp_path, "Valley block", "north orchard", INVARIANCES)
    damage_record(project_record_key(tmp_path), b"{not valid json")

    with pytest.raises(ts.DecodeError):
        create_record(tmp_path, "Valley block", "north orchard", INVARIANCES)


def test_create_record_refuses_different_invariances_naming_the_correction_door(
    tmp_path: Path,
):
    from tcip_mcp.project_record import create_record, read_record

    create_record(tmp_path, "Valley block", "north orchard", INVARIANCES)
    offered = CaptureInvariances.model_validate(
        INVARIANCES.model_dump() | {"rotation": 5.0}
    )

    with pytest.raises(ValueError) as raised:
        create_record(tmp_path, "Valley block", "north orchard", offered)

    assert "tcip write-project-record invariances" in str(raised.value)
    assert "Setup tab" in str(raised.value)
    assert read_record(tmp_path).invariances == INVARIANCES


# ── the record's corrections: one audited route, one line each ──────────────────


def _corrections(project: Path) -> list[dict]:
    return audit_rows(project, "project_record_corrected")


def test_replace_invariances_writes_the_block_and_records_one_correction_line(
    tmp_path: Path,
):
    from tcip_mcp.identity import actor
    from tcip_mcp.project_record import read_record, replace_invariances

    project = named_project(tmp_path / "valley_block", "Valley block").root
    before, lines = read_record(project), audit_rows(project)
    corrected = CaptureInvariances.model_validate(
        INVARIANCES.model_dump() | {"vertical_flip": False, "rotation": 5.0}
    )

    answer = replace_invariances(project, corrected, actor=actor("tester"))

    assert answer.model_dump() == {
        "id": before.id,
        "invariances": corrected.model_dump(),
        "previous_invariances": INVARIANCES.model_dump(),
    }
    assert read_record(project) == before.model_copy(
        update={"invariances": corrected}
    )
    assert audit_rows(project)[:-1] == lines
    (line,) = _corrections(project)
    assert line["actor"] == actor("tester")
    assert line["arguments"] == {
        "id": before.id,
        "field": "invariances",
        "value": corrected.model_dump(),
        "previous_value": INVARIANCES.model_dump(),
    }


def test_replace_invariances_refuses_a_block_the_type_refuses_writing_nothing(
    tmp_path: Path,
):
    from pydantic import ValidationError

    from tcip_mcp.identity import actor
    from tcip_mcp.project_record import read_record, replace_invariances

    project = named_project(tmp_path / "valley_block", "Valley block").root
    before, lines = read_record(project), audit_rows(project)
    unvalidated = INVARIANCES.model_copy(update={"rotation": 400.0})

    with pytest.raises(ValidationError, match="rotation"):
        replace_invariances(project, unvalidated, actor=actor("tester"))

    assert read_record(project) == before
    assert audit_rows(project) == lines


def test_replace_site_corrects_the_site_and_records_one_correction_line(
    tmp_path: Path,
):
    from tcip_mcp.identity import actor
    from tcip_mcp.project_record import read_record, replace_site

    project = named_project(tmp_path / "valley_block", "Valley block").root
    before = read_record(project)

    assert replace_site(project, "south orchard", actor=actor("tester")) == {
        "site": "south orchard",
        "previous_site": "north orchard",
    }
    assert read_record(project) == before.model_copy(
        update={"site": "south orchard"}
    )
    (line,) = _corrections(project)
    assert line["actor"] == actor("tester")
    assert line["arguments"] == {
        "id": before.id,
        "field": "site",
        "value": "south orchard",
        "previous_value": "north orchard",
    }


def test_replace_site_requires_an_actor_by_signature(tmp_path: Path):
    from tcip_mcp.project_record import replace_site

    project = named_project(tmp_path / "valley_block", "Valley block").root

    with pytest.raises(TypeError, match="actor"):
        replace_site(project, "south orchard")  # type: ignore[call-arg]
    assert _corrections(project) == []


def test_replace_site_refuses_a_project_with_no_record(tmp_path: Path):
    from tcip_mcp.identity import actor
    from tcip_mcp.project_record import ProjectRecordMissingError, replace_site

    with pytest.raises(ProjectRecordMissingError):
        replace_site(tmp_path, "south orchard", actor=actor("tester"))


# ── read_record and record_fields ───────────────────────────────────────────────


def test_read_record_raises_missing_and_publishes_no_database_for_a_root_with_no_store(
    tmp_path: Path,
):
    """Names the creating door; a read of an absent record never publishes a database."""
    from tcip_mcp.project_record import ProjectRecordMissingError, read_record
    from tcip_store.file_backend import database_file

    with pytest.raises(ProjectRecordMissingError, match="initialize_project"):
        read_record(tmp_path)

    assert not database_file(str(tmp_path.absolute())).is_file()


def test_record_fields_reports_the_record(tmp_path: Path):
    from tcip_mcp.project_record import create_record, record_fields

    record = create_record(
        tmp_path, "Valley block", "north orchard", INVARIANCES
    )

    assert record_fields(tmp_path) == {
        **record.model_dump(),
        "record_problem": None,
    }


def test_record_fields_names_the_absent_record(tmp_path: Path):
    from tcip_mcp.project_record import record_fields

    fields = record_fields(tmp_path)

    assert fields["id"] is None and fields["site"] is None
    assert "initialize_project" in fields["record_problem"]


def test_record_fields_names_an_undecodable_record(tmp_path: Path):
    from tcip_mcp.project_record import (
        create_record,
        project_record_key,
        record_fields,
    )
    from tests._record_damage_fixtures import damage_record

    create_record(tmp_path, "Valley block", "north orchard", INVARIANCES)
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

    created = create_record(
        tmp_path, "Valley block", "north orchard", INVARIANCES
    )
    root, record = existing_project(tmp_path)
    assert (root, record["id"]) == (tmp_path.resolve(), created.id)
