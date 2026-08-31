"""``record_delivery_binding_event``'s project-scoped document axis (``delivery_events``).

``compute_phenology`` already writes an audit-log line naming which buckets stood behind a
delivery (``test_phenology_tools.py``'s own
``test_compute_phenology_records_what_verification_found_in_the_datasets_own_log``); this
persistence step is a second, enumerable write beside that unchanged one, carrying the real
per-bucket ``StampBinding`` evidence the door already computed rather than a coarse gate stamp.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp.operationalization import STATE_CROSSING_DATES
from tcip_mcp.pipelines import resolution
from tcip_mcp.tools.phenology_tools import compute_phenology

from tests._binding_fixtures import record_producing_run
from tests.test_phenology_tools import _delivery_setup, _ds_root

pytestmark = pytest.mark.usefixtures("seed_catkin_operationalization")


def _delivery_event_records(project_root: Path | None = None) -> list[dict]:
    scope = resolution.delivery_events_scope(project_root)
    keys = ts.keys(resolution.DELIVERY_EVENTS_STORE, str(scope))
    return [ts.read(key) for key in keys]


def test_a_completed_crossing_delivery_writes_a_delivery_events_record_with_the_real_bindings(
    tmp_path: Path,
) -> None:
    sha = record_producing_run(tmp_path, "exp-producer")
    mapping_name, d1, d2 = _delivery_setup(
        tmp_path, experiment_id="exp-producer", checkpoint_sha256=sha)
    out_csv = tmp_path / "out" / "catkin_phenology.csv"

    res = compute_phenology(
        trait="catkin", mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv), classifier_pred_dirs=[str(d1)],
        operating_point_conf=0.4, operating_point_validated="held_out_annotations",
    )
    assert "error" not in res, res

    # The existing dataset-scoped audit-log binding event still lands, unchanged.
    from tcip_mcp.audit import audit_log_key

    page = ts.read_log(audit_log_key(_ds_root(tmp_path)))
    audit_events = [e for e in page.records
                    if e["tool"] == "compute_phenology" and "verified_buckets" in e]
    assert len(audit_events) == 1, page.records

    # The new project-scoped delivery_events record lands beside it.
    records = [r for r in _delivery_event_records() if r["door"] == "compute_phenology"]
    assert len(records) == 1, records
    record = records[0]
    assert record["trait"] == "catkin"
    assert record["delivery_kind"] == STATE_CROSSING_DATES
    assert record["output_path"] == str(out_csv)

    # The documents are the real StampBinding data this delivery computed, not a placeholder.
    audit_verified = audit_events[0]["verified_buckets"]
    assert set(record["documents"]) == set(audit_verified)
    for bucket, doc in record["documents"].items():
        assert doc["ok"] is True
        assert doc["claimed"] is True
        assert doc["experiment_id"] == "exp-record-" + Path(bucket).name
        assert doc["producing_experiment_id"] == "exp-producer"
        assert doc["checkpoint_sha256"] == sha
        assert doc["record_digest"]
        assert doc["record_digest"] == audit_verified[bucket]["record"].split(":")[-1]


def test_two_deliveries_of_the_same_trait_and_kind_both_enumerate_distinctly(
    tmp_path: Path,
) -> None:
    sha = record_producing_run(tmp_path, "exp-producer")
    mapping_name, d1, d2 = _delivery_setup(
        tmp_path, experiment_id="exp-producer", checkpoint_sha256=sha)

    first_csv = tmp_path / "out" / "first.csv"
    second_csv = tmp_path / "out" / "second.csv"
    for out_csv in (first_csv, second_csv):
        res = compute_phenology(
            trait="catkin", mapping_name=mapping_name,
            predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
            output_csv_path=str(out_csv), classifier_pred_dirs=[str(d1)],
            operating_point_conf=0.4, operating_point_validated="held_out_annotations",
        )
        assert "error" not in res, res

    records = [r for r in _delivery_event_records() if r["door"] == "compute_phenology"]
    assert len(records) == 2, records
    assert records[0]["event_id"] != records[1]["event_id"]
    assert {r["output_path"] for r in records} == {str(first_csv), str(second_csv)}
    assert {r["trait"] for r in records} == {"catkin"}
    assert {r["delivery_kind"] for r in records} == {STATE_CROSSING_DATES}


def test_a_web_route_writes_its_delivery_event_under_the_payloads_root_not_the_pinned_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A web-backend process can serve more than one project, so its process-pinned root can
    diverge from the project a specific request names, the same divergence D11 already closes
    for the operationalization record. A request naming a different project than the process
    pin must still land its delivery_events record under the project it named."""
    from fastapi.testclient import TestClient

    from tcip_web.app import app

    from tests.test_tcip_web_results_routes import _phenology_fixture

    pinned_root = tmp_path / "pinned"
    pinned_root.mkdir()
    monkeypatch.setenv("TCIP_STATE_ROOT", str(pinned_root))

    payload_root = tmp_path / "payload_project"
    body = _phenology_fixture(payload_root, validated=True, fractions=(0.75, 1.0), detections=4)

    resp = TestClient(app, base_url="http://127.0.0.1").post(
        "/api/results/phenology_measurement", json=body)
    assert resp.status_code == 200, resp.text

    payload_records = [
        r for r in _delivery_event_records(payload_root.resolve())
        if r["door"] == "results.per_plant_curves"
    ]
    assert len(payload_records) == 1, payload_records

    pinned_records = [
        r for r in _delivery_event_records(pinned_root) if r["door"] == "results.per_plant_curves"
    ]
    assert pinned_records == []


def test_phenology_measurement_records_onset_dates_only_when_the_positive_class_was_assessed(
    tmp_path: Path,
) -> None:
    """The deleted onset_dates door itself recorded its event unconditionally, but ResultsTab
    never called that door at all when positive_class_assessed was false; the merged door now
    runs both projections every time, so it must enforce that same condition itself rather than
    recording an onset-projection event the pre-merge flow never produced."""
    from fastapi.testclient import TestClient

    from tcip_web.app import app

    from tests.test_tcip_web_results_routes import _phenology_fixture

    client = TestClient(app, base_url="http://127.0.0.1")

    unclassified = _phenology_fixture(
        tmp_path, validated=True, fractions=(0.0,), id_map={"catkin": 0}, detections=2)
    resp = client.post("/api/results/phenology_measurement", json=unclassified)
    assert resp.status_code == 200, resp.text
    assert resp.json()["positive_class_assessed"] is False

    doors = {r["door"] for r in _delivery_event_records(tmp_path.resolve())}
    assert "results.per_plant_curves" in doors
    assert "results.onset_dates" not in doors


def test_phenology_measurement_records_onset_dates_when_the_positive_class_was_assessed(
    tmp_path: Path,
) -> None:
    """The parity counterpart: a run that did assess the positive class still records both
    projection events, the merged door's baseline behavior."""
    from fastapi.testclient import TestClient

    from tcip_web.app import app

    from tests.test_tcip_web_results_routes import _phenology_fixture

    client = TestClient(app, base_url="http://127.0.0.1")

    assessed = _phenology_fixture(tmp_path, validated=True, detections=100)
    resp = client.post("/api/results/phenology_measurement", json=assessed)
    assert resp.status_code == 200, resp.text
    assert resp.json()["positive_class_assessed"] is True

    doors = {r["door"] for r in _delivery_event_records(tmp_path.resolve())}
    assert {"results.per_plant_curves", "results.onset_dates"} <= doors
