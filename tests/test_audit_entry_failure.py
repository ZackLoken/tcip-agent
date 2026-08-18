"""What an audited tool does when its own audit entry cannot be appended.

The entry is written after the tool body, so a failed append is not one situation but two. A body
that returned has already committed whatever it changed, and a caller that saw only a warning
would read the missing entry as a failed call and retry a mutation that already happened. A body
that raised owns the exception the caller gets, and a failed audit-of-failure must never displace
it. Both are properties of the decorator rather than of any one tool, so they are checked against
a tool defined here, and on whichever backend the suite is bound to.
"""

from __future__ import annotations

import pytest

import tcip_mcp.audit as audit_module
import tcip_store as ts
from tcip_mcp.audit import audited


class _AppendRefused(RuntimeError):
    """Stands in for whatever stops a real append: a busy lock, a refused root, a bad key."""


def _refuse_append(*args: object, **kwargs: object) -> None:
    raise _AppendRefused("the audit log could not be appended to")


def _rows_for(tool: str) -> list[dict]:
    """Every platform-log row a call left behind, read through the seam."""
    key = audit_module.audit_log_key(audit_module.platform_audit_scope())
    return [row for row in ts.read_log(key).records if row["tool"] == tool]


def test_append_failure_after_a_successful_body_refuses_and_names_the_committed_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A committed mutation with no audit line is told to the caller, not left to a log line."""

    @audited
    def stage_something(count: int) -> dict:
        return {"ok": True, "count": count}

    monkeypatch.setattr(audit_module, "append", _refuse_append)

    # Raising at all is the property; the type is asserted after, so a decorator that returns
    # normally here fails on the behavior rather than on a name it does not carry.
    with pytest.raises(RuntimeError) as caught:
        stage_something(3)

    assert type(caught.value) is audit_module.MutationCommittedWithoutAuditLine
    assert caught.value.tool == "stage_something"
    assert "do not retry it blind" in str(caught.value)
    assert isinstance(caught.value.__cause__, _AppendRefused)


def test_a_failed_body_keeps_its_own_exception_when_the_audit_of_it_cannot_be_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit-of-failure is the decorator's business; the failure itself is the caller's."""

    @audited
    def stage_something() -> dict:
        raise KeyError("the body's own failure")

    monkeypatch.setattr(audit_module, "append", _refuse_append)

    with pytest.raises(KeyError, match="the body's own failure"):
        stage_something()


def test_an_ordinary_call_against_a_healthy_log_returns_its_result_and_writes_one_entry() -> None:
    """The rail admits the work it exists beside: nothing about a healthy call changes."""

    @audited
    def stage_something(count: int) -> dict:
        return {"ok": True, "count": count}

    assert stage_something(3) == {"ok": True, "count": 3}

    rows = _rows_for("stage_something")
    assert len(rows) == 1, rows
    assert rows[0]["status"] == "ok"
    assert rows[0]["arguments"] == {"count": 3}


def test_a_failing_body_against_a_healthy_log_still_records_the_call_and_re_raises() -> None:
    """The other half of admitting valid work: a refusing tool is audited and stays refusing."""

    @audited
    def stage_something() -> dict:
        raise ValueError("the body refused")

    with pytest.raises(ValueError, match="the body refused"):
        stage_something()

    rows = _rows_for("stage_something")
    assert len(rows) == 1, rows
    assert rows[0]["status"] == "exception"
    assert rows[0]["error"] == "the body refused"
