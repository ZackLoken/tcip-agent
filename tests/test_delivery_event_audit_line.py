"""``record_delivery_binding_event`` writes the project-scoped ``delivery_events`` record first and
the delivery's one audit line after it: the line names the record's ``event_id`` and nothing else,
and a failed append raises ``AuditEntryNotWritten`` with the record already written.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_mcp.audit as audit_module
import tcip_store as ts
from tcip_mcp.audit import AuditEntryNotWritten, audit_log_key
from tcip_mcp.pipelines import resolution


class _AppendRefused(RuntimeError):
    """Stands in for whatever stops a real append: a busy lock, a refused root, a bad key."""


def _refuse_append(*args: object, **kwargs: object) -> None:
    raise _AppendRefused("the audit log could not be appended to")


def _delivery_event_records(project_root: Path) -> list[dict]:
    scope = resolution.delivery_events_scope(project_root)
    keys = ts.keys(resolution.DELIVERY_EVENTS_STORE, str(scope))
    return [ts.read(key) for key in keys]


def _record(project_root: Path) -> None:
    resolution.record_delivery_binding_event(
        "test_door", None, [], document_reconciliations={}, dimension_reconciliations={},
        acknowledgment=None, trait="bud_opening",
        delivery_kind="test_kind", project_root=project_root, plant_mapping=None,
    )


def test_a_failed_audit_append_raises_with_the_delivery_events_record_already_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The record is the delivery's evidence and is written before the audit line; a dropped
    append still reaches the caller as a raise."""
    monkeypatch.setattr(audit_module, "append", _refuse_append)

    with pytest.raises(AuditEntryNotWritten) as caught:
        _record(tmp_path)

    assert caught.value.tool == "delivery_event"
    assert isinstance(caught.value.__cause__, _AppendRefused)
    assert len(_delivery_event_records(tmp_path.resolve())) == 1


def test_the_audit_line_names_the_records_event_id_and_nothing_else(tmp_path: Path) -> None:
    """An ordinary delivery writes one record and one audit line, a generic ``delivery_event``
    operation whose only fact is the record's ``event_id``: the door lives in the record alone."""
    from tcip_mcp import agent_identity

    _record(tmp_path)

    records = _delivery_event_records(tmp_path.resolve())
    assert len(records) == 1 and records[0]["door"] == "test_door"
    lines = list(ts.read_log(audit_log_key(tmp_path)).records)
    assert len(lines) == 1
    stated = {k: v for k, v in lines[0].items()
              if k not in {"timestamp", *agent_identity.RECORD_FIELDS}}
    assert stated == {"tool": "delivery_event", "status": "ok",
                      "arguments": {"event_id": records[0]["event_id"]}}
