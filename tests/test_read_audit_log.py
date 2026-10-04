"""``read_audit_log``: the one tool reading a scope's own audit log back, on the record itself.

Coverage, not a guard: this tool has no prior behavior to prove absent.

Every entry read here comes from a real ``@audited`` call, never a hand-built dict, except the
corrupt-page case, which damages a committed entry's bytes in the database by hand, never
something a writer through the seam could produce.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_mcp.audit as audit_module
import tcip_store as ts
from tcip_mcp.tools.meta_tools import read_audit_log


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "orchard_dataset"
    root.mkdir()
    return root


def _subjects() -> dict:
    return {"bud": {"description": "a currant bud"}}


def test_read_audit_log_filters_by_tool_and_status_newest_first(
    project: Path, dataset_root: Path,
) -> None:
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    ok = write_subject_registry(project, str(dataset_root), subjects=_subjects())
    assert "error" not in ok, ok
    failed = write_subject_registry(project, str(dataset_root), subjects={})
    assert "error" in failed, failed
    ok2 = write_subject_registry(project, str(dataset_root), subjects=_subjects(), allow_removals=True)
    assert "error" not in ok2, ok2

    result = read_audit_log(project, scope=str(dataset_root), tool="replace_registry", status="ok")
    assert "error" not in result, result
    # The two writes each leave their line; the refused call, no act, leaves none.
    assert result["count"] == 2
    assert [e["status"] for e in result["entries"]] == ["ok", "ok"]
    # Newest first: the last call's own timestamp sorts ahead of the first's.
    assert result["entries"][0]["timestamp"] >= result["entries"][-1]["timestamp"]
    assert result["scope_resolved"] == str(dataset_root.resolve())
    assert result["skipped"] == 0
    assert read_audit_log(project, scope=str(dataset_root), tool="replace_registry",
                          status="exception")["count"] == 0


def test_read_audit_log_limit_states_what_it_truncated(
    project: Path, dataset_root: Path,
) -> None:
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    for _ in range(3):
        res = write_subject_registry(project, str(dataset_root), subjects=_subjects(), allow_removals=True)
        assert "error" not in res, res

    result = read_audit_log(project, scope=str(dataset_root), tool="replace_registry", limit=1)
    assert result["count"] == 1
    assert result["skipped"] == 2


def test_read_audit_log_project_default_scope_excludes_dataset_scoped_entries(
    project: Path, dataset_root: Path,
) -> None:
    from tcip_mcp.tools.annotation_tools import write_subject_registry
    from tcip_mcp.tools.meta_tools import report_friction

    write_subject_registry(project, str(dataset_root), subjects=_subjects())
    report_friction(project, "unexpected_behavior", "a project-scoped mutation")

    project_result = read_audit_log(project, tool="report_friction")
    assert project_result["count"] == 1
    assert Path(project_result["scope_resolved"]).resolve() == project.resolve()

    dataset_result = read_audit_log(project, scope=str(dataset_root), tool="report_friction")
    assert dataset_result["count"] == 0


def test_read_audit_log_refuses_on_a_page_carrying_an_undecodable_entry(
    project: Path, dataset_root: Path,
) -> None:
    import sqlite3

    from tcip_mcp.tools.annotation_tools import write_subject_registry
    from tcip_store.file_backend import database_file

    assert "error" not in write_subject_registry(project, str(dataset_root), subjects=_subjects())
    conn = sqlite3.connect(str(database_file(str(dataset_root.resolve()))), isolation_level=None)
    try:
        conn.execute("update log_entries set entry = ? where id = (select max(id) from "
                     "log_entries)", (b'{"tool": "write_subject_registry", bro',))
    finally:
        conn.close()

    result = read_audit_log(project, scope=str(dataset_root))

    assert "error" in result
    assert "undecodable" in result["error"]
    assert "entries" not in result


def test_read_audit_log_refuses_a_scope_naming_no_dataset_or_project(
    project: Path, tmp_path: Path,
) -> None:
    stray = tmp_path / "not_a_dataset_or_project"
    stray.mkdir()

    result = read_audit_log(project, scope=str(stray))

    assert "error" in result
    assert str(stray) in result["error"]
    assert "entries" not in result


def test_read_audit_log_resolves_an_inner_path_to_its_dataset_root(
    project: Path, dataset_root: Path,
) -> None:
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    assert "error" not in write_subject_registry(project, str(dataset_root), subjects=_subjects())

    inner = str(dataset_root / "annotations" / "2026-03-02")
    result = read_audit_log(project, scope=inner, tool="replace_registry")

    assert "error" not in result, result
    assert result["count"] == 1
    assert result["scope_resolved"] == str(dataset_root.resolve())


def test_read_audit_log_treats_a_date_only_until_as_the_end_of_that_day(
    project: Path, dataset_root: Path,
) -> None:
    key = audit_module.audit_log_key(dataset_root)
    ts.append(key, {"tool": "write_subject_registry", "status": "ok",
                     "timestamp": "2026-03-02T23:59:59+00:00"})
    ts.append(key, {"tool": "write_subject_registry", "status": "ok",
                     "timestamp": "2026-03-03T00:00:01+00:00"})

    result = read_audit_log(project, scope=str(dataset_root), until="2026-03-02")

    assert result["count"] == 1
    assert result["entries"][0]["timestamp"] == "2026-03-02T23:59:59+00:00"


def test_read_audit_log_accepts_a_z_suffixed_bound(
    project: Path, dataset_root: Path,
) -> None:
    key = audit_module.audit_log_key(dataset_root)
    ts.append(key, {"tool": "write_subject_registry", "status": "ok",
                     "timestamp": "2026-03-02T10:00:00+00:00"})

    result = read_audit_log(
        project, scope=str(dataset_root), since="2026-03-02T00:00:00Z", until="2026-03-02T23:59:59Z",
    )

    assert result["count"] == 1


def test_read_audit_log_refuses_an_unparseable_since_bound(
    project: Path, dataset_root: Path,
) -> None:
    key = audit_module.audit_log_key(dataset_root)
    ts.append(key, {"tool": "write_subject_registry", "status": "ok",
                     "timestamp": "2026-03-02T10:00:00+00:00"})

    result = read_audit_log(project, scope=str(dataset_root), since="not-a-timestamp")

    assert "error" in result
    assert "since" in result["error"]
    assert "entries" not in result
