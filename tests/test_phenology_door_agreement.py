"""The two phenology delivery doors, the web export route and ``deliver_phenology_milestones``,
deliver the same majority-crossing column and name the same trait revision on their events."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import pytest

pytest.importorskip("torch")


def test_the_web_and_mcp_deliveries_agree_on_the_majority_column_and_the_revision(
    tmp_path: Path,
) -> None:
    from fastapi.testclient import TestClient

    from tcip_mcp.delivery import read_delivery_events
    from tcip_mcp.operationalization import latest_confirmed
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones
    from tcip_web.app import app
    from tests._chain_fixtures import attributed_series

    series = attributed_series(tmp_path)
    client = TestClient(app, base_url="http://127.0.0.1")
    body = series.body()
    resp = client.post(
        "/api/results/export_csv",
        json={**body, "payload": "milestones", "filename": "delivery.csv"})
    assert resp.status_code == 200, resp.text[:300]
    web_row = next(iter(csv.DictReader(StringIO(resp.text))))

    out_csv = tmp_path / "mcp_delivery.csv"
    result = deliver_phenology_milestones(
        tmp_path, trait=body["trait"], mapping_name=body["mapping_name"], plants=body["plants"],
        buckets=body["buckets"], output_csv_path=str(out_csv))
    assert "error" not in result, result
    with out_csv.open(newline="", encoding="utf-8") as fh:
        mcp_row = next(iter(csv.DictReader(fh)))

    column = f"bud_{latest_confirmed('bud_opening', tmp_path).entry.majority_label}_date"
    assert web_row[column] == mcp_row[column] != ""
    assert web_row["validated"] == mcp_row["validated"] == "True"
    revisions = {(e.trait_revision, e.trait_revision_sha256)
                 for e in read_delivery_events(tmp_path)}
    assert len(revisions) == 1, revisions
