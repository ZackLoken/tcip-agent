"""``scripts/clear_dev_history.py``: clearing a root's development-era audit lines, friction
reports, retrospectives and learning-capture lines before a root reaches an alpha tester, and
leaving every other store untouched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import tcip_store as ts
from tcip_store.binding import BACKEND_ENV, SQLITE_BACKEND, bind_default
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


def test_a_plan_removes_nothing_and_lists_the_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A database-backed root, pinned rather than left to the ambient default, since the
    # stale-export outcome line below only appears once a database exists to leave one behind.
    monkeypatch.setenv(BACKEND_ENV, SQLITE_BACKEND)
    bind_default()
    module = _load_script()
    _seed_dev_history(tmp_path)

    outcomes = module.plan_root(tmp_path)

    assert any("would remove 1 friction_reports record(s)" in o for o in outcomes)
    assert any("would remove 1 retrospectives record(s)" in o for o in outcomes)
    assert any("would remove 3 audit_log line(s)" in o for o in outcomes)
    assert any("would remove 1 learning_capture line(s)" in o for o in outcomes)
    assert any("nothing was written" in o for o in outcomes)
    assert any(
        "a database backs this root" in o and "python scripts/export_store.py" in o
        for o in outcomes
    )
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


def test_main_apply_and_plan_together_refuses_before_touching_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bind_default()
    module = _load_script()
    _seed_dev_history(tmp_path)

    monkeypatch.setattr(
        sys, "argv",
        ["clear_dev_history.py", "--apply", "--plan", "--by", "user:zack",
         "--reason", "alpha handoff", str(tmp_path)],
    )
    with pytest.raises(SystemExit) as raised:
        module.main()
    assert raised.value.code == 2

    assert len(ts.keys(FRICTION_REPORT_STORE, str(tmp_path))) == 1
    assert len(ts.read_log(audit_log_key(tmp_path)).records) == 3


def test_main_apply_over_a_seeded_root_exits_zero_and_records_the_closing_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # A database-backed root, pinned rather than left to the ambient default, since the
    # stale-export outcome line below only appears once a database exists to leave one behind.
    monkeypatch.setenv(BACKEND_ENV, SQLITE_BACKEND)
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
    assert "a database backs this root" in output
    assert "python scripts/export_store.py" in output
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


def test_a_store_error_on_one_root_does_not_abort_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A second root a mis-bound backend refuses must not stop the first from clearing, and
    must not surface as an uncaught traceback: main() names the refusal and moves on."""
    from tcip_store.binding import BACKEND_ENV, FILE_BACKEND, SQLITE_BACKEND

    root1 = tmp_path / "project1"
    root2 = tmp_path / "project2"
    root1.mkdir()
    root2.mkdir()

    (root2 / ".tcip").mkdir()
    monkeypatch.setenv(BACKEND_ENV, SQLITE_BACKEND)
    sqlite_backend = bind_default()
    try:
        # An explicit scope=root2 append, not _seed_dev_history's tools (bare @audited, so
        # platform-scoped): those would write against TCIP_STATE_ROOT, not root2 itself.
        record_event_or_raise("seed", {}, scope=root2)
    finally:
        sqlite_backend.close()

    monkeypatch.setenv(BACKEND_ENV, FILE_BACKEND)
    file_backend = bind_default()
    try:
        module = _load_script()
        _seed_dev_history(root1)

        monkeypatch.setattr(
            sys, "argv",
            ["clear_dev_history.py", "--apply", "--by", "user:zack", "--reason", "alpha handoff",
             str(root1), str(root2)],
        )
        exit_code = module.main()
        output = capsys.readouterr().out

        assert exit_code == 2
        assert f"{root1}: closing audit line recorded" in output
        assert f"{root2}: refused, StoreError:" in output
        assert len(ts.read_log(audit_log_key(root1)).records) == 1
    finally:
        file_backend.close()


def test_a_plan_refusal_on_the_first_root_still_plans_the_second(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A plan (the default, no --apply) over two roots where the first refuses must still plan
    the second and exit nonzero, not traceback out of the loop: plan_root's own StoreError (a
    sqlite conform-rail refusal here, the read-side counterpart to the write-side one
    test_a_store_error_on_one_root_does_not_abort_the_others exercises under --apply) shares
    main()'s refusal handling rather than escaping uncaught from the --plan branch."""
    from tcip_store.binding import BACKEND_ENV, FILE_BACKEND, SQLITE_BACKEND

    root1 = tmp_path / "project1"
    root2 = tmp_path / "project2"
    root1.mkdir()
    root2.mkdir()

    monkeypatch.setenv("TCIP_STATE_ROOT", str(root1))
    monkeypatch.setenv(BACKEND_ENV, FILE_BACKEND)
    file_backend = bind_default()
    try:
        _seed_dev_history(root1)
    finally:
        file_backend.close()

    monkeypatch.setenv("TCIP_STATE_ROOT", str(root2))
    monkeypatch.setenv(BACKEND_ENV, SQLITE_BACKEND)
    sqlite_backend = bind_default()
    try:
        module = _load_script()
        _seed_dev_history(root2)

        monkeypatch.setattr(sys, "argv", ["clear_dev_history.py", str(root1), str(root2)])
        exit_code = module.main()
        output = capsys.readouterr().out

        assert exit_code == 2
        assert f"{root1}: refused, StoreError:" in output
        assert f"{root2}: would remove 1 friction_reports record(s)" in output
    finally:
        sqlite_backend.close()


def test_stale_exported_loose_copies_are_removed_by_the_next_export_after_clearing(
    tmp_path: Path,
) -> None:
    """clear_root leaves an already-exported loose copy on disk; a later export_store.py run
    is what removes it, from the tombstone the seam's own deletes now leave behind. Binds a
    backend of its own (this case is about sqlite's exported-copy mechanics, not the ambient
    TCIP_STORE_BACKEND) and closes it itself, since the per-test fixture closes only its own.
    """
    from tcip_store import bind
    from tcip_store.sqlite_backend import SqliteBackend

    backend = SqliteBackend()
    try:
        bind(backend)
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
        assert audit_file.is_file(), "clear_root itself must not touch the export"

        export_root(str(tmp_path))

        assert not any(reports_dir.iterdir()) if reports_dir.is_dir() else True
        assert not any(retros_dir.iterdir()) if retros_dir.is_dir() else True
        assert not capture_file.is_file()
        # audit_log itself carries the closing line clear_root just wrote, so its exported
        # file is re-materialized with that one line rather than deleted outright.
        lines = audit_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["tool"] == "clear_dev_history"
        assert database_file(str(tmp_path)).is_file()
    finally:
        backend.close()
