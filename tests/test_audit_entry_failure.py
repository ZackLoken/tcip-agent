"""What an audited tool does when its own audit entry cannot be appended: a body that returned
raises the unwritten line, and a body that raised keeps its own exception."""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_mcp.audit as audit_module
from tcip_mcp.audit import audited
from tests._audit_fixtures import audit_rows, refuse_audit_appends


class _AppendRefusedError(RuntimeError):
    """Stands in for whatever stops a real append: a busy lock, a refused root, a bad key."""


def _refuse_append(monkeypatch: pytest.MonkeyPatch) -> None:
    refuse_audit_appends(
        monkeypatch, error=_AppendRefusedError("the audit log could not be appended to")
    )


def test_append_failure_after_a_successful_body_refuses_and_names_the_committed_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A committed mutation with no audit line is told to the caller, not left to a log line."""

    @audited
    def stage_something(project: Path, count: int) -> dict:
        return {"ok": True, "count": count}

    _refuse_append(monkeypatch)

    # Raising at all is the property; the type is asserted after, so a decorator that returns
    # normally here fails on the behavior rather than on a name it does not carry.
    with pytest.raises(RuntimeError) as caught:
        stage_something(tmp_path, 3)

    assert type(caught.value) is audit_module.MutationCommittedWithoutAuditLineError
    assert caught.value.tool == "stage_something"
    assert "do not retry it blind" in str(caught.value)
    assert isinstance(caught.value.__cause__, _AppendRefusedError)


def test_a_failed_body_keeps_its_own_exception_when_the_audit_of_it_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit-of-failure is the decorator's business; the failure itself is the caller's."""

    @audited
    def stage_something(project: Path) -> dict:
        raise KeyError("the body's own failure")

    _refuse_append(monkeypatch)

    with pytest.raises(KeyError, match="the body's own failure"):
        stage_something(tmp_path)


def test_an_ordinary_call_against_a_healthy_log_returns_its_result_and_writes_one_entry(
    tmp_path: Path,
) -> None:
    """The rail admits the work it exists beside: nothing about a healthy call changes."""

    @audited
    def stage_something(project: Path, count: int) -> dict:
        return {"ok": True, "count": count}

    assert stage_something(tmp_path, 3) == {"ok": True, "count": 3}

    rows = audit_rows(tmp_path, "stage_something")
    assert len(rows) == 1, rows
    assert rows[0]["status"] == "ok"
    assert rows[0]["arguments"] == {"count": 3}


def test_a_failing_body_against_a_healthy_log_still_records_the_call_and_re_raises(
    tmp_path: Path,
) -> None:
    """The other half of admitting valid work: a refusing tool is audited and stays refusing."""

    @audited
    def stage_something(project: Path) -> dict:
        raise ValueError("the body refused")

    with pytest.raises(ValueError, match="the body refused"):
        stage_something(tmp_path)

    rows = audit_rows(tmp_path, "stage_something")
    assert len(rows) == 1, rows
    assert rows[0]["status"] == "exception"
    assert rows[0]["error"] == "the body refused"


def test_a_door_with_no_project_parameter_is_refused_at_decoration() -> None:
    with pytest.raises(ValueError, match="needs a project parameter"):
        @audited
        def stage_something(count: int) -> dict:
            return {"count": count}


# ── record_event_or_raise: the line for a mutation that has already committed ──


def test_record_event_or_raise_raises_audit_entry_not_written_when_the_append_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A confirmation that already committed must not be reported as silently unrecorded."""
    _refuse_append(monkeypatch)

    with pytest.raises(audit_module.AuditEntryNotWrittenError) as caught:
        audit_module.record_event_or_raise("confirm_something", {"trait": "bloom"}, actor=None,
                                           scope=tmp_path)

    assert caught.value.tool == "confirm_something"
    assert "do not retry it blind" in str(caught.value)
    assert isinstance(caught.value.__cause__, _AppendRefusedError)


def test_record_event_or_raise_against_a_healthy_log_writes_one_entry_and_returns_silently(
    tmp_path: Path,
) -> None:
    """The rail admits the work it exists beside: a healthy append lands one line."""
    audit_module.record_event_or_raise(
        "confirm_something", {"trait": "bloom"}, actor=None, status="ok", scope=tmp_path,
        note="confirmed",
    )

    rows = audit_rows(tmp_path, "confirm_something")
    assert len(rows) == 1, rows
    assert rows[0]["status"] == "ok"
    assert rows[0]["arguments"] == {"trait": "bloom"}
    assert rows[0]["note"] == "confirmed"
