"""``record_delivery_binding_event``'s project-scoped document axis (``delivery_events``).

``deliver_phenology_milestones`` already writes an audit-log line naming which buckets stood behind a
delivery (``test_phenology_tools.py``'s own
``test_deliver_phenology_milestones_records_what_verification_found_in_the_datasets_own_log``); this
persistence step is a second, enumerable write beside that unchanged one, carrying the real
per-bucket ``StampBinding`` evidence the door already computed rather than a coarse gate stamp.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp.operationalization import STATE_CROSSING_DATES
from tcip_mcp.pipelines import resolution
from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

from tests._binding_fixtures import record_producing_run
from tests.test_phenology_tools import _delivery_setup, _ds_root

pytestmark = pytest.mark.usefixtures("seed_bud_operationalization")

from tests._population import mapped_plants


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
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        trait="bud_opening", mapping_name=mapping_name, plants=mapped_plants(mapping_name),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv), classifier_pred_dirs=[str(d1)],
    )
    assert "error" not in res, res

    # The existing dataset-scoped audit-log binding event still lands, unchanged.
    from tcip_mcp.audit import audit_log_key

    page = ts.read_log(audit_log_key(_ds_root(tmp_path)))
    audit_events = [e for e in page.records
                    if e["tool"] == "deliver_phenology_milestones" and "verified_buckets" in e]
    assert len(audit_events) == 1, page.records

    # The new project-scoped delivery_events record lands beside it.
    records = [r for r in _delivery_event_records() if r["door"] == "deliver_phenology_milestones"]
    assert len(records) == 1, records
    record = records[0]
    assert record["trait"] == "bud_opening"
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


def test_a_record_predating_the_reconciliation_fields_validates_and_reads_back_with_both_keys_absent(
    tmp_path: Path,
) -> None:
    """A record written before document_reconciliations/dimension_reconciliations existed still
    validates against today's model and reads back through read_delivery_events with both keys
    absent, never as None: read_delivery_events returns the stored dict itself, so a key this
    record never carried is missing from it rather than present and null."""
    event_id = "pre-field-record"
    key = resolution.delivery_event_key(resolution.delivery_events_scope(tmp_path), event_id)
    ts.replace(
        key,
        {
            "event_id": event_id, "trait": "astringency", "delivery_kind": "state_crossing_dates",
            "door": "test_door", "output_path": None, "output_sha256": None,
            "measurement_documents": ["operating_point"], "scale_document": None,
            "acknowledged_by": None, "acknowledgement_reason": None, "plant_mapping": None,
            "documents": {}, "produced_at": "2026-02-03T12:00:00+00:00",
        },
        expect=ts.Version.ABSENT,
    )

    records = resolution.read_delivery_events(tmp_path)
    assert len(records) == 1, records
    record = records[0]
    assert "document_reconciliations" not in record
    assert "dimension_reconciliations" not in record


def test_a_completed_crossing_delivery_reads_back_through_read_delivery_events_with_its_reconciliations(
    tmp_path: Path,
) -> None:
    """The healthy round trip goes through the reader every other consumer uses
    (list_delivery_events, _citing_delivery_event_ids), not the raw store this module's own
    _delivery_event_records helper reads through: a real delivery's document_reconciliations and
    dimension_reconciliations come back exactly as the door computed them."""
    sha = record_producing_run(tmp_path, "exp-producer")
    mapping_name, d1, d2 = _delivery_setup(
        tmp_path, experiment_id="exp-producer", checkpoint_sha256=sha)
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        trait="bud_opening", mapping_name=mapping_name, plants=mapped_plants(mapping_name),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv), classifier_pred_dirs=[str(d1)],
    )
    assert "error" not in res, res

    records = [
        r for r in resolution.read_delivery_events(tmp_path)
        if r["door"] == "deliver_phenology_milestones"
    ]
    assert len(records) == 1, records
    record = records[0]

    assert set(record["document_reconciliations"]) == {
        "operating_point", "classifier_operating_point"}
    assert set(record["document_reconciliations"]["operating_point"]["bindings"]) == (
        set(record["documents"]))
    classifier_entry = record["document_reconciliations"]["classifier_operating_point"]
    assert classifier_entry["bound_validated"] == classifier_entry["validated"]
    assert record["dimension_reconciliations"]["tile_size"]["operative"] is False


def test_two_deliveries_of_the_same_trait_and_kind_both_enumerate_distinctly(
    tmp_path: Path,
) -> None:
    sha = record_producing_run(tmp_path, "exp-producer")
    mapping_name, d1, d2 = _delivery_setup(
        tmp_path, experiment_id="exp-producer", checkpoint_sha256=sha)

    first_csv = tmp_path / "out" / "first.csv"
    second_csv = tmp_path / "out" / "second.csv"
    for out_csv in (first_csv, second_csv):
        res = deliver_phenology_milestones(
            trait="bud_opening", mapping_name=mapping_name, plants=mapped_plants(mapping_name),
            predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
            output_csv_path=str(out_csv), classifier_pred_dirs=[str(d1)],
        )
        assert "error" not in res, res

    records = [r for r in _delivery_event_records() if r["door"] == "deliver_phenology_milestones"]
    assert len(records) == 2, records
    assert records[0]["event_id"] != records[1]["event_id"]
    assert {r["output_path"] for r in records} == {str(first_csv), str(second_csv)}
    assert {r["trait"] for r in records} == {"bud_opening"}
    assert {r["delivery_kind"] for r in records} == {STATE_CROSSING_DATES}


def test_a_web_route_writes_its_delivery_event_under_the_payloads_root_not_the_pinned_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A web-backend process can serve more than one project, so its process-pinned root can
    diverge from the project a specific request names, the same divergence already closed
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
        "/api/results/export_csv", json={**body, "payload": "milestones", "filename": "x.csv"})
    assert resp.status_code == 200, resp.text

    payload_records = [
        r for r in _delivery_event_records(payload_root.resolve())
        if r["door"] == "results.export_csv"
    ]
    assert len(payload_records) == 1, payload_records

    pinned_records = [
        r for r in _delivery_event_records(pinned_root) if r["door"] == "results.export_csv"
    ]
    assert pinned_records == []


def test_phenology_measurement_records_no_delivery_event_for_an_unclassified_look(
    tmp_path: Path,
) -> None:
    """Looking at a number on screen is not delivering it: phenology_measurement records no
    delivery event and no audit line either way, since nothing here changed any state."""
    from fastapi.testclient import TestClient

    from tcip_mcp.audit import audit_log_key

    from tcip_web.app import app

    from tests.test_tcip_web_results_routes import _phenology_fixture

    client = TestClient(app, base_url="http://127.0.0.1")

    unclassified = _phenology_fixture(
        tmp_path, validated=True, fractions=(0.0,), id_map={"bud": 0}, detections=2)
    before = list(ts.read_log(audit_log_key(tmp_path)).records)
    resp = client.post("/api/results/phenology_measurement", json=unclassified)
    assert resp.status_code == 200, resp.text
    assert resp.json()["positive_class_assessed"] is False

    assert _delivery_event_records(tmp_path.resolve()) == []
    assert list(ts.read_log(audit_log_key(tmp_path)).records) == before


def test_record_delivery_binding_event_reports_a_failed_store_write_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The event write is best effort: a store failure after the artifact already shipped is a
    provenance gap the caller can disclose, never a reason to raise on an already-completed
    delivery. The return value says whether the write actually landed."""
    def _boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(ts, "replace", _boom)

    recorded = resolution.record_delivery_binding_event(
        "test_door", None, [], document_reconciliations={}, dimension_reconciliations={},
        measurement_documents=["operating_point"],
        scale_document=None, acknowledgement=None, trait="astringency",
        delivery_kind=STATE_CROSSING_DATES, project_root=tmp_path, plant_mapping=None,
    )

    assert recorded is False
    assert _delivery_event_records(tmp_path.resolve()) == []


def test_record_delivery_binding_event_raises_and_writes_nothing_when_plant_mapping_fails_validation(
    tmp_path: Path,
) -> None:
    """A caller's ``plant_mapping`` missing a required disclosure key is a shape violation in the
    caller, never an environmental failure, so it raises into the delivering door rather than
    falling into this function's own best-effort warning path (reserved for the store write)."""
    from pydantic import ValidationError

    bad_mapping = {
        "name": "valley", "project_root": str(tmp_path), "dataset_id": "ds-1",
        "dataset_root": "C:/data", "built_at": "2026-02-01T00:00:00+00:00",
        "record_sha256": "0" * 64, "nn_tolerance_m": {"value": 3, "source": "stated"},
        "capture_identity": {}, "captures_unverified": [], "plant_csvs_unverified": [],
        "images_unattributed_scope": "delivered_dates",
        # dates_delivered, images_unattributed and plant_attribution are missing.
    }

    with pytest.raises(ValidationError):
        resolution.record_delivery_binding_event(
            "test_door", None, [], document_reconciliations={}, dimension_reconciliations={},
            measurement_documents=["operating_point"],
            scale_document=None, acknowledgement=None, trait="astringency",
            delivery_kind=STATE_CROSSING_DATES, project_root=tmp_path, plant_mapping=bad_mapping,
        )

    assert _delivery_event_records(tmp_path) == []


def test_record_delivery_binding_event_refuses_when_the_declared_documents_entry_is_absent(
    tmp_path: Path,
) -> None:
    """A caller stating what its gate reconciled in measurement_documents but not reconciling it
    is a deterministic defect in the caller: pred_dirs is non-empty, the sole declared document
    has no entry in document_reconciliations, and the writer refuses by name with the audit line
    already on the log and no record built."""
    d = str(tmp_path / "bucket")

    with pytest.raises(ValueError, match="operating_point"):
        resolution.record_delivery_binding_event(
            "test_door", None, [d], document_reconciliations={}, dimension_reconciliations={},
            measurement_documents=["operating_point"], scale_document=None, acknowledgement=None,
            trait="astringency", delivery_kind=STATE_CROSSING_DATES, project_root=tmp_path,
            plant_mapping=None,
        )

    from tcip_mcp.audit import audit_log_key

    events = ts.read_log(audit_log_key(tmp_path)).records
    assert any(e["tool"] == "test_door" for e in events)
    assert _delivery_event_records(tmp_path.resolve()) == []


def test_record_delivery_binding_event_refuses_naming_the_classifier_document_when_only_the_count_is_given(
    tmp_path: Path,
) -> None:
    """Every document measurement_documents declares is checked, not only the first: a two-
    document delivery with the count entry alone still refuses, naming the missing classifier
    document specifically."""
    from tcip_mcp.pipelines.resolution import StampBinding

    from tests._binding_fixtures import document_reconciliation

    d = str(tmp_path / "bucket")
    binding = StampBinding(ok=True, claimed=True, experiment_id="exp-1",
                           producing_experiment_id="exp-1", checkpoint_sha256="0" * 64,
                           record_digest="digest-1")
    recon = document_reconciliation(
        {d: binding}, validated="held_out_annotations", per_bucket={d: "held_out_annotations"},
        unvalidated_buckets=[], missing_sidecars=[], on_disk_validated=True)

    with pytest.raises(ValueError, match="classifier_operating_point"):
        resolution.record_delivery_binding_event(
            "test_door", None, [d], document_reconciliations={"operating_point": recon},
            dimension_reconciliations={},
            measurement_documents=["operating_point", "classifier_operating_point"],
            scale_document=None, acknowledgement=None, trait="astringency",
            delivery_kind=STATE_CROSSING_DATES, project_root=tmp_path, plant_mapping=None,
        )

    assert _delivery_event_records(tmp_path.resolve()) == []


def test_record_delivery_binding_event_refuses_a_document_entry_missing_a_key_the_reconciler_returns(
    tmp_path: Path,
) -> None:
    """Every key _reconcile_validity always returns is required of a document entry: one that
    dropped per_bucket raises naming the key, with the audit line already on the log and no
    record built, rather than landing a record whose empty per_bucket reads as a reconciliation
    of nothing. Guards the renderer reading the key as required instead of defaulting it."""
    from tcip_mcp.pipelines.resolution import StampBinding

    from tests._binding_fixtures import document_reconciliation

    d = str(tmp_path / "bucket")
    binding = StampBinding(ok=True, claimed=True, experiment_id="exp-1",
                           producing_experiment_id="exp-1", checkpoint_sha256="0" * 64,
                           record_digest="digest-1")
    recon = document_reconciliation(
        {d: binding}, validated="held_out_annotations", per_bucket={d: "held_out_annotations"},
        unvalidated_buckets=[], missing_sidecars=[], on_disk_validated=True)
    del recon["per_bucket"]

    with pytest.raises(KeyError, match="per_bucket"):
        resolution.record_delivery_binding_event(
            "test_door", None, [d], document_reconciliations={"operating_point": recon},
            dimension_reconciliations={}, measurement_documents=["operating_point"],
            scale_document=None, acknowledgement=None, trait="astringency",
            delivery_kind=STATE_CROSSING_DATES, project_root=tmp_path, plant_mapping=None,
        )

    assert _delivery_event_records(tmp_path.resolve()) == []


def test_record_delivery_binding_event_refuses_a_dimension_entry_missing_a_key_the_reconciler_returns(
    tmp_path: Path,
) -> None:
    """The dimension renderer reads its five keys as required the same way: a tile_size entry
    that dropped binding_notes raises naming the key and builds no record."""
    from tcip_mcp.pipelines.resolution import StampBinding

    from tests._binding_fixtures import document_reconciliation

    d = str(tmp_path / "bucket")
    binding = StampBinding(ok=True, claimed=True, experiment_id="exp-1",
                           producing_experiment_id="exp-1", checkpoint_sha256="0" * 64,
                           record_digest="digest-1")
    recon = document_reconciliation(
        {d: binding}, validated="held_out_annotations", per_bucket={d: "held_out_annotations"},
        unvalidated_buckets=[], missing_sidecars=[], on_disk_validated=True)
    tile = {"operative": False, "validated": None, "per_bucket": {}, "unvalidated_buckets": []}

    with pytest.raises(KeyError, match="binding_notes"):
        resolution.record_delivery_binding_event(
            "test_door", None, [d], document_reconciliations={"operating_point": recon},
            dimension_reconciliations={"tile_size": tile},
            measurement_documents=["operating_point"], scale_document=None,
            acknowledgement=None, trait="astringency", delivery_kind=STATE_CROSSING_DATES,
            project_root=tmp_path, plant_mapping=None,
        )

    assert _delivery_event_records(tmp_path.resolve()) == []


def test_plant_mapping_union_resolves_each_shape_and_refuses_a_hybrid(tmp_path: Path) -> None:
    """No two of the three ``plant_mapping`` disclosure shapes share a required key set: a dict
    validates against exactly the one model whose keys it carries, and a hybrid combining keys
    from two shapes resolves to none."""
    from pydantic import ValidationError

    from tcip_mcp.pipelines.delivery_events_schema import (
        CanopySegmentDisclosure,
        DeliveryEventRecord,
        PlantMappingDisclosure,
        PlantRegistryDisclosure,
    )

    mapping = {
        "name": "valley", "project_root": str(tmp_path), "dataset_id": "ds-1",
        "dataset_root": "C:/data", "built_at": "2026-02-01T00:00:00+00:00",
        "record_sha256": "0" * 64, "nn_tolerance_m": {"value": 3, "source": "stated"},
        "capture_identity": {}, "captures_unverified": [], "plant_csvs_unverified": [],
        "dates_delivered": [], "images_unattributed": 0,
        "images_unattributed_scope": "delivered_dates", "plant_attribution": "image",
    }
    registry = {
        "plant_registry": {"name": "reg", "digest": "0" * 64}, "project_root": str(tmp_path),
        "raster_identity": {"width": 10, "height": 10},
        "nn_tolerance_m": {"value": 1, "source": "stated"}, "detections_unattributed": 0,
        "detections_unattributed_scope": "delivered_raster", "plant_attribution": "detection",
        "plants_outside_raster": [],
    }
    canopy = {
        "plant_registry": {"name": "reg", "digest": "0" * 64}, "project_root": str(tmp_path),
        "raster_identity": {"width": 10, "height": 10},
        "canopy_segments": {"path": "x", "sha256": "0" * 64, "subject": "canopy", "n_segments": 1},
        "segment_ties": [], "segments_without_plant": 0, "plants_outside_raster": [],
        "plants_without_segment": [], "plants_with_ambiguous_detections": [],
        "detections_unattributed": 0,
        "detections_unattributed_by_source": {
            "outside_segments": 0, "overlapping_segments": 0, "segment_without_plant": 0},
        "detections_unattributed_scope": "delivered_raster", "plant_attribution": "segment",
    }

    def _resolved(pm: dict) -> object:
        record = {
            "event_id": "e", "trait": None, "delivery_kind": None, "door": "d",
            "output_path": None, "output_sha256": None, "measurement_documents": [],
            "scale_document": None, "acknowledged_by": None, "acknowledgement_reason": None,
            "plant_mapping": pm, "documents": {}, "produced_at": "t",
        }
        return DeliveryEventRecord.model_validate(record).plant_mapping

    assert isinstance(_resolved(mapping), PlantMappingDisclosure)
    assert isinstance(_resolved(registry), PlantRegistryDisclosure)
    assert isinstance(_resolved(canopy), CanopySegmentDisclosure)

    hybrid = {**registry, "canopy_segments": canopy["canopy_segments"]}
    with pytest.raises(ValidationError):
        _resolved(hybrid)


def test_phenology_measurement_records_no_delivery_event_for_an_assessed_look(
    tmp_path: Path,
) -> None:
    """The parity counterpart: a run that did assess the positive class still records no delivery
    event and no audit line, since a look on screen never becomes a shipped artifact."""
    from fastapi.testclient import TestClient

    from tcip_mcp.audit import audit_log_key

    from tcip_web.app import app

    from tests.test_tcip_web_results_routes import _phenology_fixture

    client = TestClient(app, base_url="http://127.0.0.1")

    assessed = _phenology_fixture(tmp_path, validated=True, detections=100)
    before = list(ts.read_log(audit_log_key(tmp_path)).records)
    resp = client.post("/api/results/phenology_measurement", json=assessed)
    assert resp.status_code == 200, resp.text
    assert resp.json()["positive_class_assessed"] is True

    assert _delivery_event_records(tmp_path.resolve()) == []
    assert list(ts.read_log(audit_log_key(tmp_path)).records) == before
