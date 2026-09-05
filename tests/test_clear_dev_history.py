"""``scripts/clear_dev_history.py``: clearing a root's development-era audit lines, friction
reports, retrospectives and learning-capture lines before a root reaches an alpha tester, and
leaving every other store untouched.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import tcip_store as ts
from tcip_store.binding import bind_default
from tcip_store.file_backend import database_file

from tcip_mcp.audit import AuditEntryNotWritten, audit_log_key, record_event_or_raise
from tcip_mcp.dataset_layout import image_status_key, record_image_statuses, status_bucket
from tcip_mcp.tools.meta_tools import (
    FRICTION_REPORT_STORE,
    RETROSPECTIVE_STORE,
    report_friction,
    write_retrospective,
)
from tcip_mcp.tools.project_tools import dataset_registry_key, upsert_dataset
from tcip_web.agent_learning_capture import learning_capture_key

SCRIPT = Path(__file__).parent.parent / "scripts" / "clear_dev_history.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("clear_dev_history_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_dev_history(root: Path) -> None:
    """One friction report, one retrospective, one learning-capture line, and three audit
    lines: ``report_friction`` and ``write_retrospective`` each write their own (they are
    ``@audited`` tools), and one more is written explicitly through ``record_event_or_raise``,
    the way a non-tool caller records to the audit log."""
    (root / ".tcip").mkdir(exist_ok=True)
    report_friction(str(root), "unexpected_behavior", "the shoot count looked off")
    write_retrospective(
        str(root), "phase0", task="bud survey", worked="the split held",
        did_not_work="the first model overfit",
    )
    record_event_or_raise("some_tool", {"n": 1}, scope=root)
    from tcip_store import append

    append(learning_capture_key(root), {"ts": "2026-01-01T00:00:00+00:00", "note": "session ended"})


def _seed_other_stores(root: Path) -> None:
    """One annotation status and one dataset registry row, so the test can prove they survive."""
    record_image_statuses(
        root, status_bucket("shoot", "2026-01-01"), {"img_0001": "complete"}, recorded_by="user:test",
    )
    upsert_dataset(root, {"id": "ds-1", "path": ".", "crop": "chestnut"})


def test_a_plan_removes_nothing_and_lists_the_counts(tmp_path: Path):
    bind_default()
    module = _load_script()
    _seed_dev_history(tmp_path)

    outcomes = module.plan_root(tmp_path)

    assert any("would remove 1 friction_reports record(s)" in o for o in outcomes)
    assert any("would remove 1 retrospectives record(s)" in o for o in outcomes)
    assert any("would remove 3 audit_log line(s)" in o for o in outcomes)
    assert any("would remove 1 learning_capture line(s)" in o for o in outcomes)
    assert any("nothing was written" in o for o in outcomes)
    assert len(ts.keys(FRICTION_REPORT_STORE, str(tmp_path))) == 1
    assert len(ts.keys(RETROSPECTIVE_STORE, str(tmp_path))) == 1
    assert len(ts.read_log(audit_log_key(tmp_path)).records) == 3
    assert len(ts.read_log(learning_capture_key(tmp_path)).records) == 1


def test_an_apply_removes_every_record_and_line_and_leaves_other_stores_untouched(tmp_path: Path):
    bind_default()
    module = _load_script()
    _seed_dev_history(tmp_path)
    _seed_other_stores(tmp_path)

    outcomes = module.clear_root(tmp_path, by="user:zack", reason="alpha handoff")

    assert ts.keys(FRICTION_REPORT_STORE, str(tmp_path)) == []
    assert ts.keys(RETROSPECTIVE_STORE, str(tmp_path)) == []
    audit_page = ts.read_log(audit_log_key(tmp_path))
    assert len(audit_page.records) == 1
    assert audit_page.records[0]["tool"] == "clear_dev_history"
    assert audit_page.records[0]["arguments"]["by"] == "user:zack"
    assert audit_page.records[0]["arguments"]["reason"] == "alpha handoff"
    assert audit_page.records[0]["arguments"]["friction_reports_removed"] == 1
    assert audit_page.records[0]["arguments"]["retrospectives_removed"] == 1
    assert audit_page.records[0]["arguments"]["audit_log_lines_removed"] == 3
    assert audit_page.records[0]["arguments"]["learning_capture_lines_removed"] == 1
    assert ts.read_log(learning_capture_key(tmp_path)).records == []
    assert any("closing audit line recorded" in o for o in outcomes)

    status_doc = ts.read(image_status_key(tmp_path))
    assert status_doc["shoot/2026-01-01"]["img_0001"]["status"] == "complete"
    registry = ts.read(dataset_registry_key(tmp_path))
    assert [r["id"] for r in registry] == ["ds-1"]


def test_main_plan_over_a_root_with_no_tcip_directory_exits_two_and_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    bind_default()
    module = _load_script()

    monkeypatch.setattr(sys, "argv", ["clear_dev_history.py", str(tmp_path)])
    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 2
    assert f"{tmp_path.resolve()}: refused, no .tcip directory found" in output


def test_main_apply_without_by_or_reason_refuses_before_touching_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    bind_default()
    module = _load_script()
    _seed_dev_history(tmp_path)

    monkeypatch.setattr(sys, "argv", ["clear_dev_history.py", "--apply", str(tmp_path)])
    with pytest.raises(SystemExit) as raised:
        module.main()
    assert raised.value.code == 2

    assert len(ts.keys(FRICTION_REPORT_STORE, str(tmp_path))) == 1
    assert len(ts.read_log(audit_log_key(tmp_path)).records) == 3


def test_main_apply_with_by_but_no_reason_refuses_before_touching_anything(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bind_default()
    module = _load_script()
    _seed_dev_history(tmp_path)

    monkeypatch.setattr(
        sys, "argv", ["clear_dev_history.py", "--apply", "--by", "user:zack", str(tmp_path)]
    )
    with pytest.raises(SystemExit) as raised:
        module.main()
    assert raised.value.code == 2
    assert len(ts.keys(FRICTION_REPORT_STORE, str(tmp_path))) == 1


def test_main_apply_over_a_seeded_root_exits_zero_and_records_the_closing_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    bind_default()
    module = _load_script()
    _seed_dev_history(tmp_path)

    monkeypatch.setattr(
        sys, "argv",
        ["clear_dev_history.py", "--apply", "--by", "user:zack", "--reason", "alpha handoff",
         str(tmp_path)],
    )
    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "closing audit line recorded" in output
    assert len(ts.read_log(audit_log_key(tmp_path)).records) == 1


def test_a_failed_audit_append_after_clearing_exits_nonzero_naming_audit_entry_not_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    bind_default()
    module = _load_script()
    _seed_dev_history(tmp_path)

    def _boom(*args, **kwargs):
        raise AuditEntryNotWritten("clear_dev_history", RuntimeError("disk full"))

    monkeypatch.setattr(module, "record_event_or_raise", _boom)
    monkeypatch.setattr(
        sys, "argv",
        ["clear_dev_history.py", "--apply", "--by", "user:zack", "--reason", "alpha handoff",
         str(tmp_path)],
    )
    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "AuditEntryNotWritten" in output
    # Everything preceding the closing append already committed.
    assert ts.keys(FRICTION_REPORT_STORE, str(tmp_path)) == []
    assert ts.keys(RETROSPECTIVE_STORE, str(tmp_path)) == []
    assert ts.read_log(learning_capture_key(tmp_path)).records == []


def test_stale_exported_loose_copies_are_removed_under_the_database_backend(
    tmp_path: Path,
) -> None:
    # This case is about the sqlite backend's own exported-copy mechanics, so it binds the
    # backend directly rather than through the ambient TCIP_STORE_BACKEND.
    from tcip_store import bind
    from tcip_store.sqlite_backend import SqliteBackend

    bind(SqliteBackend())
    module = _load_script()
    _seed_dev_history(tmp_path)

    from tcip_store.export import export_root

    export_root(str(tmp_path))
    reports_dir = tmp_path / ".tcip" / "reports"
    retros_dir = tmp_path / ".tcip" / "retrospectives"
    audit_file = tmp_path / ".tcip" / "audit.jsonl"
    capture_file = tmp_path / ".tcip" / "learning_capture.jsonl"
    assert any(reports_dir.iterdir())
    assert any(retros_dir.iterdir())
    assert audit_file.is_file()
    assert capture_file.is_file()

    module.clear_root(tmp_path, by="user:zack", reason="alpha handoff")

    assert not any(reports_dir.iterdir()) if reports_dir.is_dir() else True
    assert not any(retros_dir.iterdir()) if retros_dir.is_dir() else True
    assert not audit_file.is_file()
    assert not capture_file.is_file()
    assert database_file(str(tmp_path)).is_file()
