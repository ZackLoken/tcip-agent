"""The delivered majority-crossing marker answers a different question from the delivery gate.

A trait's ``majority_provisional`` records whether the breeders have confirmed that the trait's
"most objects in state" phrase maps to the crossing key its spec names. The delivery gate records
whether the measurement dimensions behind the numbers were validated against a reference. Both reach
the breeder in one CSV, so the web export door must fill the marker column from the trait's own spec
and never from the gate it just cleared: the two answers are independent, and one delivery can carry
a settled gate beside an unconfirmed reading.
"""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_mcp.pipelines.postprocessing import phenology
from tcip_mcp.pipelines.resolution import VALIDATED_HELD_OUT
from tcip_mcp.traits import get_trait
from tcip_web.app import app

from tests.test_tcip_web_results_routes import _phenology_fixture

pytestmark = pytest.mark.usefixtures("seed_bud_operationalization")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _export_row(client: TestClient, body: dict) -> dict:
    resp = client.post(
        "/api/results/export_csv",
        json={**body, "payload": "milestones", "filename": "delivery.csv"})
    assert resp.status_code == 200, resp.text[:300]
    rows = list(csv.DictReader(StringIO(resp.text)))
    assert rows, "the delivery must carry at least one plant row"
    return rows[0]


def test_the_majority_marker_reports_the_specs_reading_not_the_gates_verdict(
    client: TestClient, tmp_path: Path,
) -> None:
    """One delivery, two answers. The trait's majority phrase maps to its 95% crossing on a reading
    the breeders have not confirmed, while every measurement dimension behind the dates cleared the
    gate. The marker column must say the reading is unconfirmed even though nothing about this
    delivery is provisional in the gate's sense."""
    spec = get_trait("bud_opening")
    assert spec.majority_provisional is True
    body = _phenology_fixture(tmp_path, validated=True, detections=100)

    disclosure = client.post("/api/results/phenology_measurement", json=body).json()
    assert disclosure["has_unvalidated_dimensions"] is False
    assert disclosure["validated"]["operating_point"] == VALIDATED_HELD_OUT
    assert disclosure["validated"]["classifier"] == VALIDATED_HELD_OUT

    row = _export_row(client, body)
    marker = phenology.majority_crossing_unconfirmed_column(spec)
    assert marker in row
    assert row[marker] == "true"
    assert row[marker] == str(spec.majority_provisional).lower()
    assert row["operating_point_validated"] == VALIDATED_HELD_OUT
    assert row["positive_state_classifier_validated"] == VALIDATED_HELD_OUT


def test_the_web_and_mcp_deliveries_agree_on_the_majority_marker(
    client: TestClient, tmp_path: Path,
) -> None:
    """Two delivery doors, one trait registry: the CSV the Results tab downloads and the CSV
    ``deliver_phenology_milestones`` writes must carry the same marker under the same column name for the same
    trait, rather than each door deciding for itself what the reading's status is."""
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    body = _phenology_fixture(tmp_path, validated=True, detections=100)
    web_row = _export_row(client, body)

    out_csv = tmp_path / "mcp_delivery.csv"
    result = deliver_phenology_milestones(
        trait=body["trait"], mapping_name=body["mapping_name"],
        predictions_by_date=body["predictions_by_date"], output_csv_path=str(out_csv),
        classifier_pred_dirs=list(body["predictions_by_date"].values()),
    )
    assert "error" not in result, result
    with out_csv.open(newline="", encoding="utf-8") as fh:
        mcp_row = next(iter(csv.DictReader(fh)))

    marker = phenology.majority_crossing_unconfirmed_column(get_trait("bud_opening"))
    assert marker in mcp_row
    assert web_row[marker] == mcp_row[marker]
