"""Rows the GUI routes append to an audit log parse the same way as the platform writer's rows.

Both writers append to the same scope's log, so a consumer reads a single stream: every row is
a mapping carrying a timezone-aware UTC timestamp, the tool that acted, its arguments, and a
status drawn from one vocabulary, whichever writer produced it.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import tcip_mcp.audit as audit_module
import tcip_store as ts
from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def test_gui_route_rows_and_platform_rows_agree_on_their_core_fields(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "shared_dataset"
    dataset_root.mkdir()
    monkeypatch.setattr(audit_module, "AUDIT_ROOT", dataset_root)

    resp = client.post(
        "/api/classes/image_status",
        json={"project_root": str(tmp_path / "project"), "dataset_root": str(dataset_root),
              "image_name": "IMG_0001.JPG", "status": "complete", "subject": "bud"},
    )
    assert resp.status_code == 200, resp.text
    audit_module.record_event("scan_dataset", {"dataset_root": str(dataset_root)})

    rows = ts.read_log(audit_module.audit_log_key(dataset_root)).records
    assert len(rows) == 2
    gui_row, platform_row = rows
    assert gui_row["tool"] == "gui_set_image_status"
    assert platform_row["tool"] == "scan_dataset"

    for row in rows:
        assert isinstance(row["arguments"], dict), row
        stamp = datetime.fromisoformat(row["timestamp"])
        assert stamp.utcoffset() == timedelta(0), row

    assert gui_row["status"] == platform_row["status"], (
        "a successful GUI write and a successful platform event must report success the same way"
    )
