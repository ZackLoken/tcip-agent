"""``deliver_csv`` writes one ``delivery_events`` record and one audit line naming it.

The record is project-scoped. The line names the record's ``event_id`` and nothing else, and a
failed append raises ``AuditEntryNotWrittenError`` with the record in place.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp import delivery
from tcip_mcp.audit import AuditEntryNotWrittenError
from tests._audit_fixtures import audit_rows, refuse_audit_appends
from tests._trait_fixtures import seed_confirmed_count
from tests.test_delivery_events import _bucket, _record


class _AppendRefusedError(RuntimeError):
    """An append failure standing in for a busy lock, a refused root or a bad key."""


def test_a_failed_audit_append_raises_with_the_delivery_events_record_already_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed audit append raises, with the delivery's events record in place.

    The delivery raises ``AuditEntryNotWrittenError`` for ``delivery_event``, caused by the append
    failure, and the one events record is in place.
    """
    revision = seed_confirmed_count(tmp_path / "trait_project")
    bucket = _bucket(tmp_path)
    # The acknowledgment the unassessed bucket ships under lands its own line first.
    refuse_audit_appends(
        monkeypatch,
        landing=1,
        error=_AppendRefusedError("the audit log could not be appended to"),
    )

    with pytest.raises(AuditEntryNotWrittenError) as caught:
        _record(tmp_path, revision, bucket)

    assert caught.value.tool == "delivery_event"
    assert isinstance(caught.value.__cause__, _AppendRefusedError)
    assert len(delivery.read_delivery_events(tmp_path.resolve())) == 1


def test_the_audit_line_names_the_records_event_id_and_nothing_else(
    tmp_path: Path,
) -> None:
    """A delivery's audit line names the record's ``event_id`` and nothing else.

    An ordinary delivery writes one record, naming its door, and one audit line, a
    ``delivery_event`` operation whose arguments hold the record's ``event_id`` alone.
    """
    from tcip_mcp import agent_identity

    revision = seed_confirmed_count(tmp_path / "trait_project")
    bucket = _bucket(tmp_path)
    dataset = tmp_path / "ds"  # the dataset the delivered bucket sits in
    before = len(audit_rows(dataset))

    _record(tmp_path, revision, bucket)

    records = delivery.read_delivery_events(tmp_path.resolve())
    assert len(records) == 1 and records[0].door == "test_door"
    lines = audit_rows(dataset)[before:]
    assert len(lines) == 1
    stated = {
        k: v
        for k, v in lines[0].items()
        if k not in {"timestamp", *agent_identity.RECORD_FIELDS}
    }
    assert stated == {
        "tool": "delivery_event",
        "status": "ok",
        "arguments": {"event_id": records[0].event_id},
    }


def test_the_record_carries_its_audit_lines_instant(tmp_path: Path) -> None:
    """A delivery's record and its audit line state one instant, the line's."""
    from tcip_mcp.audit import AuditLine

    revision = seed_confirmed_count(tmp_path / "trait_project")
    bucket = _bucket(tmp_path)
    dataset = tmp_path / "ds"
    before = len(audit_rows(dataset))

    _record(tmp_path, revision, bucket)

    (record,) = delivery.read_delivery_events(tmp_path.resolve())
    (line,) = audit_rows(dataset)[before:]
    assert AuditLine.model_validate(line).timestamp == record.produced_at
