"""``deliver_csv`` writes the project-scoped ``delivery_events`` record first and the
delivery's one audit line after it: the line names the record's ``event_id`` and nothing else,
and a failed append raises ``AuditEntryNotWritten`` with the record already written.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp import delivery
from tcip_mcp.audit import AuditEntryNotWritten, audit_log_key
from tests._audit_fixtures import refuse_audit_appends
from tests._trait_fixtures import seed_confirmed_count
from tests.test_delivery_events import _bucket, _record


class _AppendRefused(RuntimeError):
    """Stands in for whatever stops a real append: a busy lock, a refused root, a bad key."""


def test_a_failed_audit_append_raises_with_the_delivery_events_record_already_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The record is the delivery's evidence and is written before the audit line; a dropped
    append still reaches the caller as a raise."""
    revision = seed_confirmed_count(tmp_path / "trait_project")
    bucket = _bucket(tmp_path)
    # The acknowledgment the unassessed bucket ships under lands its own line first.
    refuse_audit_appends(monkeypatch, landing=1,
                         error=_AppendRefused("the audit log could not be appended to"))

    with pytest.raises(AuditEntryNotWritten) as caught:
        _record(tmp_path, revision, bucket)

    assert caught.value.tool == "delivery_event"
    assert isinstance(caught.value.__cause__, _AppendRefused)
    assert len(delivery.read_delivery_events(tmp_path.resolve())) == 1


def test_the_audit_line_names_the_records_event_id_and_nothing_else(tmp_path: Path) -> None:
    """An ordinary delivery writes one record and one audit line, a generic ``delivery_event``
    operation whose only fact is the record's ``event_id``: the door lives in the record alone."""
    from tcip_mcp import agent_identity

    revision = seed_confirmed_count(tmp_path / "trait_project")
    bucket = _bucket(tmp_path)
    log = audit_log_key(tmp_path / "ds")  # the dataset the delivered bucket sits in
    before = len(ts.read_log(log).records)

    _record(tmp_path, revision, bucket)

    records = delivery.read_delivery_events(tmp_path.resolve())
    assert len(records) == 1 and records[0].door == "test_door"
    lines = ts.read_log(log).records[before:]
    assert len(lines) == 1
    stated = {k: v for k, v in lines[0].items()
              if k not in {"timestamp", *agent_identity.RECORD_FIELDS}}
    assert stated == {"tool": "delivery_event", "status": "ok",
                      "arguments": {"event_id": records[0].event_id}}
