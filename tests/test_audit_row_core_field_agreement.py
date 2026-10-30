"""A GUI route's audit row and the platform writer's carry their core fields alike.

One row from each writer lands in one scope's log and reads back as an audit line; each carries a
UTC timestamp, the tool that acted and its arguments as a mapping, and the two rows' statuses are
equal. A stored line whose timestamp is no instant is refused by the reader.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import tcip_mcp.audit as audit_module
import tcip_store


def test_gui_route_rows_and_platform_rows_agree_on_their_core_fields(
    client: TestClient, tmp_path: Path
) -> None:
    dataset_root = tmp_path / "shared_dataset"
    dataset_root.mkdir()

    resp = client.post(
        "/api/subjects/save",
        json={
            "dataset_root": str(dataset_root),
            "subjects": {"bud": {"description": "one bud"}},
            "version": None,
            "user": "tester",
        },
    )
    assert resp.status_code == 200, resp.text
    audit_module.record_event_or_raise(
        "scan_dataset",
        {"dataset_root": str(dataset_root)},
        actor=None,
        scope=dataset_root,
    )

    rows, _cursor = audit_module.audit_entries(dataset_root)
    assert [row.tool for row in rows] == ["replace_registry", "scan_dataset"]
    assert [row.timestamp.utcoffset() for row in rows] == [timedelta(0)] * 2

    gui_row, platform_row = rows
    assert gui_row.status == platform_row.status, (
        "a successful GUI write and a successful platform event must report success the same way"
    )


def test_an_extra_fact_spelled_as_a_field_of_the_line_is_dropped() -> None:
    """An extra fact keyed by one of the line's own fields is no extra fact; any other is kept."""
    line = audit_module.audit_line(
        "scan_dataset",
        {},
        None,
        "ok",
        extra={"duration_ms": 5.0, "error": "spelled", "note": "kept"},
    )

    assert (line.duration_ms, line.error) == (None, None)
    assert line.stored()["note"] == "kept"


def test_a_stored_line_whose_timestamp_is_no_instant_is_refused_by_the_reader(
    tmp_path: Path,
) -> None:
    line = audit_module.audit_line("scan_dataset", {}, None, "ok").stored()
    tcip_store.append(
        audit_module.audit_log_key(tmp_path), {**line, "timestamp": "yesterday"}
    )

    with pytest.raises(ValidationError) as refused:
        audit_module.audit_entries(tmp_path)

    (error,) = refused.value.errors()
    assert error["loc"] == ("timestamp",)
