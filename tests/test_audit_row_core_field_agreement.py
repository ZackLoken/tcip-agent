"""A GUI route's audit row and the platform writer's carry their core fields alike.

One row from each writer lands in one scope's log; each carries a UTC timestamp, the tool that
acted and its arguments as a mapping, and the two rows' statuses are equal.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

import tcip_mcp.audit as audit_module
from tests._audit_fixtures import audit_rows


def test_gui_route_rows_and_platform_rows_agree_on_their_core_fields(
    client: TestClient, tmp_path: Path
) -> None:
    dataset_root = tmp_path / "shared_dataset"
    dataset_root.mkdir()

    resp = client.post(
        "/api/subjects/save",
        json={
            "dataset_root": str(dataset_root),
            "subjects": {"bud": {}},
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

    rows = audit_rows(dataset_root)
    assert len(rows) == 2
    gui_row, platform_row = rows
    assert gui_row["tool"] == "replace_registry"
    assert platform_row["tool"] == "scan_dataset"

    for row in rows:
        assert isinstance(row["arguments"], dict), row
        stamp = datetime.fromisoformat(row["timestamp"])
        assert stamp.utcoffset() == timedelta(0), row

    assert gui_row["status"] == platform_row["status"], (
        "a successful GUI write and a successful platform event must report success the same way"
    )
