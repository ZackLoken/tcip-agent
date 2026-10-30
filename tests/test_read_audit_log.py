"""``read_audit_log``: the one tool reading a scope's own audit log back, on the record itself.

The entries read come from real ``@audited`` calls or the platform's own line writer
(``audit.record_event_or_raise``), except in the corrupt-page case, which damages a committed
entry's bytes in the database by hand. The bounds are read as an MCP client states them, through
the server's own argument schema.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import tcip_mcp.audit as audit_module
from tcip_mcp.tools.meta_tools import read_audit_log
from tests._server_calls import called_through_the_server

if TYPE_CHECKING:
    from tcip_mcp.workspace import BoundProject


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "orchard_dataset"
    root.mkdir()
    return root


def _subjects() -> dict:
    return {"bud": {"description": "a currant bud"}}


def _lines_at(
    scope: Path, instants: list[datetime], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write one ``scan_dataset`` line in ``scope``'s log at each of ``instants``, in order.

    Each is the platform's own line (:func:`audit.record_event_or_raise`), its clock read at the
    instant given.
    """
    for instant in instants:
        monkeypatch.setattr(audit_module, "now", lambda at=instant: at)
        audit_module.record_event_or_raise(
            "scan_dataset", {}, actor=None, scope=scope
        )


def _read(bound: BoundProject, **arguments: str) -> dict | str:
    """Return ``read_audit_log``'s answer to ``arguments`` as a client calls it, or its refusal."""
    answered = called_through_the_server(
        "read_audit_log", arguments, bound=bound
    )
    assert isinstance(answered, dict | str), answered
    return answered


def test_read_audit_log_filters_by_tool_and_status_newest_first(
    project: Path,
    dataset_root: Path,
) -> None:
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    ok = write_subject_registry(
        project, str(dataset_root), subjects=_subjects(), actor=None
    )
    assert "error" not in ok, ok
    failed = write_subject_registry(
        project, str(dataset_root), subjects={}, actor=None
    )
    assert "error" in failed, failed
    ok2 = write_subject_registry(
        project,
        str(dataset_root),
        subjects=_subjects(),
        allow_removals=True,
        actor=None,
    )
    assert "error" not in ok2, ok2

    result = read_audit_log(
        project, scope=str(dataset_root), tool="replace_registry", status="ok"
    )
    assert "error" not in result, result
    # The two writes each leave their line; the refused call, no act, leaves none.
    assert result["count"] == 2
    assert [e["status"] for e in result["entries"]] == ["ok", "ok"]
    # Newest first: the last call's own timestamp sorts ahead of the first's.
    assert (
        result["entries"][0]["timestamp"] >= result["entries"][-1]["timestamp"]
    )
    assert result["scope_resolved"] == str(dataset_root.resolve())
    assert result["skipped"] == 0
    assert (
        read_audit_log(
            project,
            scope=str(dataset_root),
            tool="replace_registry",
            status="exception",
        )["count"]
        == 0
    )


def test_read_audit_log_limit_states_what_it_truncated(
    project: Path,
    dataset_root: Path,
) -> None:
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    for _ in range(3):
        res = write_subject_registry(
            project,
            str(dataset_root),
            subjects=_subjects(),
            allow_removals=True,
            actor=None,
        )
        assert "error" not in res, res

    result = read_audit_log(
        project, scope=str(dataset_root), tool="replace_registry", limit=1
    )
    assert result["count"] == 1
    assert result["skipped"] == 2


def test_read_audit_log_project_default_scope_excludes_dataset_scoped_entries(
    project: Path,
    dataset_root: Path,
) -> None:
    from tcip_mcp.tools.annotation_tools import write_subject_registry
    from tcip_mcp.tools.meta_tools import report_friction

    write_subject_registry(
        project, str(dataset_root), subjects=_subjects(), actor=None
    )
    report_friction(
        project, "unexpected_behavior", "a project-scoped mutation", actor=None
    )

    project_result = read_audit_log(project, tool="report_friction")
    assert project_result["count"] == 1
    assert (
        Path(project_result["scope_resolved"]).resolve() == project.resolve()
    )

    dataset_result = read_audit_log(
        project, scope=str(dataset_root), tool="report_friction"
    )
    assert dataset_result["count"] == 0


def test_read_audit_log_refuses_on_a_page_carrying_an_undecodable_entry(
    project: Path,
    dataset_root: Path,
) -> None:
    import sqlite3

    from tcip_mcp.tools.annotation_tools import write_subject_registry
    from tcip_store.file_backend import database_file

    assert "error" not in write_subject_registry(
        project, str(dataset_root), subjects=_subjects(), actor=None
    )
    conn = sqlite3.connect(
        str(database_file(str(dataset_root.resolve()))), isolation_level=None
    )
    try:
        conn.execute(
            "update log_entries set entry = ? where id = (select max(id) from "
            "log_entries)",
            (b'{"tool": "write_subject_registry", bro',),
        )
    finally:
        conn.close()

    result = read_audit_log(project, scope=str(dataset_root))

    assert "error" in result
    assert "undecodable" in result["error"]
    assert "entries" not in result


def test_read_audit_log_refuses_a_scope_naming_no_dataset_or_project(
    project: Path,
    tmp_path: Path,
) -> None:
    stray = tmp_path / "not_a_dataset_or_project"
    stray.mkdir()

    result = read_audit_log(project, scope=str(stray))

    assert "error" in result
    assert str(stray) in result["error"]
    assert "entries" not in result


def test_read_audit_log_resolves_an_inner_path_to_its_dataset_root(
    project: Path,
    dataset_root: Path,
) -> None:
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    assert "error" not in write_subject_registry(
        project, str(dataset_root), subjects=_subjects(), actor=None
    )

    inner = str(image_dir(dataset_root, "2026-03-02"))
    result = read_audit_log(project, scope=inner, tool="replace_registry")

    assert "error" not in result, result
    assert result["count"] == 1
    assert result["scope_resolved"] == str(dataset_root.resolve())


def test_read_audit_log_treats_a_date_only_until_as_the_end_of_that_day(
    bound: BoundProject,
    dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    last_of_the_day = datetime(2026, 3, 2, 23, 59, 59, tzinfo=UTC)
    _lines_at(
        dataset_root,
        [last_of_the_day, datetime(2026, 3, 3, 0, 0, 1, tzinfo=UTC)],
        monkeypatch,
    )

    result = _read(bound, scope=str(dataset_root), until="2026-03-02")

    assert isinstance(result, dict), result
    assert result["count"] == 1
    (entry,) = result["entries"]
    assert datetime.fromisoformat(entry["timestamp"]) == last_of_the_day


def test_read_audit_log_accepts_a_z_suffixed_bound(
    bound: BoundProject,
    dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _lines_at(
        dataset_root, [datetime(2026, 3, 2, 10, tzinfo=UTC)], monkeypatch
    )

    result = _read(
        bound,
        scope=str(dataset_root),
        since="2026-03-02T00:00:00Z",
        until="2026-03-02T23:59:59Z",
    )

    assert isinstance(result, dict), result
    assert result["count"] == 1


@pytest.mark.parametrize(
    "value",
    ["2026-03-02Z", "2026-03-02T10Z:00:00", "not-a-timestamp", "2026-03-02T10:00:00"],
)
def test_read_audit_log_refuses_a_since_bound_that_is_no_instant_or_date(
    bound: BoundProject,
    dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    """A bound that parses as neither an instant with its offset nor a date is refused by name.

    A naive instant states no offset and is refused with the rest.
    """
    _lines_at(
        dataset_root, [datetime(2026, 3, 2, 10, tzinfo=UTC)], monkeypatch
    )

    refused = _read(bound, scope=str(dataset_root), since=value)

    assert isinstance(refused, str), refused
    assert "validation error" in refused and "since" in refused
