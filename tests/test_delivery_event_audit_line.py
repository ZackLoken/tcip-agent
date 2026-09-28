"""``record_delivery_binding_event`` writes the project-scoped ``delivery_events`` record first and
the delivery's one audit line after it: the line names the record's ``event_id`` and nothing else,
and a failed append raises ``AuditEntryNotWritten`` with the record already written.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp.audit import AuditEntryNotWritten, audit_log_key
from tcip_mcp.pipelines import resolution
from tcip_mcp.traits import TraitRevision
from tests._audit_fixtures import refuse_audit_appends
from tests._trait_fixtures import seed_confirmed_count


class _AppendRefused(RuntimeError):
    """Stands in for whatever stops a real append: a busy lock, a refused root, a bad key."""


def _record(project_root: Path, revision: TraitRevision) -> None:
    resolution.record_delivery_binding_event(
        "test_door", None, [], document_reconciliations={}, dimension_reconciliations={},
        acknowledgment=None, revision=revision,
        delivery_kind="per_image_count", project_root=project_root, plant_mapping=None,
    )


def test_a_failed_audit_append_raises_with_the_delivery_events_record_already_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The record is the delivery's evidence and is written before the audit line; a dropped
    append still reaches the caller as a raise."""
    revision = seed_confirmed_count(tmp_path / "trait_project")
    refuse_audit_appends(monkeypatch, error=_AppendRefused("the audit log could not be appended to"))

    with pytest.raises(AuditEntryNotWritten) as caught:
        _record(tmp_path, revision)

    assert caught.value.tool == "delivery_event"
    assert isinstance(caught.value.__cause__, _AppendRefused)
    assert len(resolution.read_delivery_events(tmp_path.resolve())) == 1


def test_the_audit_line_names_the_records_event_id_and_nothing_else(tmp_path: Path) -> None:
    """An ordinary delivery writes one record and one audit line, a generic ``delivery_event``
    operation whose only fact is the record's ``event_id``: the door lives in the record alone."""
    from tcip_mcp import agent_identity

    _record(tmp_path, seed_confirmed_count(tmp_path / "trait_project"))

    records = resolution.read_delivery_events(tmp_path.resolve())
    assert len(records) == 1 and records[0]["door"] == "test_door"
    lines = list(ts.read_log(audit_log_key(tmp_path)).records)
    assert len(lines) == 1
    stated = {k: v for k, v in lines[0].items()
              if k not in {"timestamp", *agent_identity.RECORD_FIELDS}}
    assert stated == {"tool": "delivery_event", "status": "ok",
                      "arguments": {"event_id": records[0]["event_id"]}}
