"""Tests for tcip write-project-record, run by subprocess against a project on disk.

The command corrects a mistyped site or capture invariances, keeping every other field of the
record, and names the person making the correction; the project is the suite's ``project``
fixture, made through the platform's own creation door.
"""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.identity import actor
from tests._audit_fixtures import audit_rows
from tests._cli_fixtures import run_tcip

CORRECTED_ANSWERS = [
    "--horizontal-flip",
    "--no-vertical-flip",
    "--rotation",
    "5",
    "--color",
    "--intensity",
    "--scale",
]
"""The sample answers with no vertical flip and rotation up to 5 degrees, as the command takes
them."""


def test_the_site_form_replaces_the_site_and_keeps_every_other_field(
    project: Path,
):
    from tcip_mcp.project_record import read_record

    before = read_record(project)

    result = run_tcip(
        "write-project-record",
        ["site", str(project), "south orchard", "--user", "tester"],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "written" in result.stdout
    assert read_record(project) == before.model_copy(
        update={"site": "south orchard"}
    )
    (line,) = audit_rows(project, "project_record_corrected")
    assert line["actor"] == actor("tester")


def test_the_invariances_form_writes_the_answers(project: Path):
    from tcip_mcp.project_record import read_record

    before = read_record(project)

    result = run_tcip(
        "write-project-record",
        ["invariances", str(project), *CORRECTED_ANSWERS, "--user", "tester"],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    after = read_record(project)
    corrected = {"vertical_flip": False, "rotation": 5.0}
    assert after.invariances.model_dump() == (
        before.invariances.model_dump() | corrected
    )
    assert after.model_dump(exclude={"invariances"}) == before.model_dump(
        exclude={"invariances"}
    )
    (line,) = audit_rows(project, "project_record_corrected")
    assert line["arguments"]["field"] == "invariances"


def test_a_correction_naming_no_one_is_refused_by_the_parser(project: Path):
    from tcip_mcp.project_record import read_record

    before = read_record(project)

    for args in (
        ["site", str(project), "south orchard"],
        ["invariances", str(project), *CORRECTED_ANSWERS],
    ):
        result = run_tcip("write-project-record", args)
        assert result.returncode == 2, result.stdout + result.stderr
        assert "--user" in result.stderr
    assert read_record(project) == before
    assert audit_rows(project, "project_record_corrected") == []


def test_a_directory_holding_no_project_record_is_refused(tmp_path: Path):
    bare = tmp_path / "bare"
    bare.mkdir()

    for args in (
        ["site", str(bare), "north orchard"],
        ["invariances", str(bare), *CORRECTED_ANSWERS],
    ):
        result = run_tcip("write-project-record", [*args, "--user", "tester"])
        assert result.returncode == 2
        assert "refused" in result.stdout


def test_a_record_that_does_not_read_is_refused(project: Path):
    import tcip_store
    from tcip_mcp.project_record import project_record_key

    key = project_record_key(str(project))
    current = tcip_store.read_versioned(key).version
    tcip_store.replace(key, {"not_site": "x"}, expect=current)

    result = run_tcip(
        "write-project-record",
        ["site", str(project), "south orchard", "--user", "tester"],
    )

    assert result.returncode == 2
    assert "is not a project record" in result.stdout
    assert tcip_store.read(key) == {"not_site": "x"}
