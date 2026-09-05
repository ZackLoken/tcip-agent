"""``record_delivery_binding_event``'s dataset-scoped audit line used to go through
``record_event`` (never raises, a dropped append is only logged) even though the mutation it
records already committed by the time this call runs and no ``@audited`` tool body brackets it.
It now goes through ``record_event_or_raise``, so a failed append surfaces to the delivering
door as ``AuditEntryNotWritten`` rather than passing as if the platform's canonical log recorded
the delivery when it did not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_mcp.audit as audit_module
from tcip_mcp.audit import AuditEntryNotWritten
from tcip_mcp.pipelines import resolution


class _AppendRefused(RuntimeError):
    """Stands in for whatever stops a real append: a busy lock, a refused root, a bad key."""


def _refuse_append(*args: object, **kwargs: object) -> None:
    raise _AppendRefused("the audit log could not be appended to")


def _delivery_event_records(project_root: Path) -> list[dict]:
    scope = resolution.delivery_events_scope(project_root)
    import tcip_store as ts

    keys = ts.keys(resolution.DELIVERY_EVENTS_STORE, str(scope))
    return [ts.read(key) for key in keys]


def test_a_failed_audit_append_raises_and_writes_no_delivery_events_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit line is attempted first; a failed append raises before the project-scoped
    delivery_events record is ever built or written, since the artifact already shipped and the
    canonical log missing the event must reach the caller, not the record's own True/False."""
    monkeypatch.setattr(audit_module, "append", _refuse_append)

    with pytest.raises(AuditEntryNotWritten) as caught:
        resolution.record_delivery_binding_event(
            "test_door", None, [], {}, measurement_documents=["operating_point"],
            scale_document=None, acknowledgement=None, trait="bud_opening",
            delivery_kind="test_kind", project_root=tmp_path, plant_mapping=None,
        )

    assert caught.value.tool == "test_door"
    assert isinstance(caught.value.__cause__, _AppendRefused)
    assert _delivery_event_records(tmp_path.resolve()) == []


def test_an_ordinary_call_still_records_both_the_audit_line_and_the_delivery_event(
    tmp_path: Path,
) -> None:
    """The admitting case: a delivery whose audit append succeeds is unaffected by the switch to
    record_event_or_raise, both writes land exactly as they did on record_event."""
    import tcip_store as ts

    recorded = resolution.record_delivery_binding_event(
        "test_door", None, [], {}, measurement_documents=["operating_point"],
        scale_document=None, acknowledgement=None, trait="bud_opening",
        delivery_kind="test_kind", project_root=tmp_path, plant_mapping=None,
    )

    assert recorded is True
    assert len(_delivery_event_records(tmp_path.resolve())) == 1

    from tcip_mcp.audit import audit_log_key

    events = ts.read_log(audit_log_key(tmp_path)).records
    assert any(e["tool"] == "test_door" and "verified_buckets" in e for e in events)
