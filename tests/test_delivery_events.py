"""``record_delivery_binding_event``'s project-scoped document axis (``delivery_events``).

The record carries the real per-bucket ``StampBinding`` evidence the delivering door computed, and
the delivery's audit line names only the record's ``event_id``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp.pipelines import resolution
from tcip_mcp.pipelines.resolution import read_delivery_events
from tcip_mcp.project_paths import project_state_dir
from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones
from tcip_mcp.traits import STATE_CROSSING_DATES, read_trait

from tests._binding_fixtures import record_producing_run
from tests._trait_fixtures import seed_confirmed_count
from tests.test_phenology_tools import _delivery_setup, _ds_root

pytestmark = pytest.mark.usefixtures("seed_bud_operationalization")

from tests._population import mapped_plants


def test_a_completed_crossing_delivery_writes_a_delivery_events_record_with_the_real_bindings(
    tmp_path: Path,
) -> None:
    sha = record_producing_run(tmp_path, "exp-producer")
    mapping_name, d1, d2 = _delivery_setup(
        tmp_path, experiment_id="exp-producer", checkpoint_sha256=sha)
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        tmp_path, trait="bud_opening", mapping_name=mapping_name, plants=mapped_plants(tmp_path, mapping_name),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv), classifier_pred_dirs=[str(d1)],
    )
    assert "error" not in res, res

    from tcip_mcp.audit import audit_log_key

    records = [r for r in read_delivery_events(tmp_path) if r["door"] == "deliver_phenology_milestones"]
    assert len(records) == 1, records
    record = records[0]
    assert record["trait"] == "bud_opening"
    shipped_under = read_trait("bud_opening", tmp_path).latest_confirmed
    assert shipped_under is not None
    assert (record["trait_revision"], record["trait_revision_sha256"]) == (
        shipped_under.number, shipped_under.entry_sha256)
    assert record["delivery_kind"] == STATE_CROSSING_DATES
    assert record["output_path"] == str(out_csv)

    page = ts.read_log(audit_log_key(_ds_root(tmp_path)))
    audit_events = [e for e in page.records if e["tool"] == "delivery_event"
                    and e.get("arguments") == {"event_id": record["event_id"]}]
    assert len(audit_events) == 1, page.records

    bindings = record["document_reconciliations"]["operating_point"]["bindings"]
    assert set(bindings) == {str(d1), str(d2)}
    for bucket, doc in bindings.items():
        assert doc["ok"] is True
        assert doc["claimed"] is True
        assert doc["experiment_id"] == "exp-record-" + Path(bucket).name
        assert doc["producing_experiment_id"] == "exp-producer"
        assert doc["checkpoint_sha256"] == sha
        assert doc["record_digest"]


def test_a_completed_crossing_delivery_reads_back_through_read_delivery_events_with_its_reconciliations(
    tmp_path: Path,
) -> None:
    """A real delivery's document_reconciliations and dimension_reconciliations come back
    through read_delivery_events exactly as the door computed them."""
    sha = record_producing_run(tmp_path, "exp-producer")
    mapping_name, d1, d2 = _delivery_setup(
        tmp_path, experiment_id="exp-producer", checkpoint_sha256=sha)
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        tmp_path, trait="bud_opening", mapping_name=mapping_name, plants=mapped_plants(tmp_path, mapping_name),
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
    assert set(record["document_reconciliations"]["operating_point"]["bindings"]) == {
        str(d1), str(d2)}
    assert "documents" not in record
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
            tmp_path, trait="bud_opening", mapping_name=mapping_name, plants=mapped_plants(tmp_path, mapping_name),
            predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
            output_csv_path=str(out_csv), classifier_pred_dirs=[str(d1)],
        )
        assert "error" not in res, res

    records = [r for r in read_delivery_events(tmp_path) if r["door"] == "deliver_phenology_milestones"]
    assert len(records) == 2, records
    assert records[0]["event_id"] != records[1]["event_id"]
    assert {r["output_path"] for r in records} == {str(first_csv), str(second_csv)}
    assert {r["trait"] for r in records} == {"bud_opening"}
    assert {r["delivery_kind"] for r in records} == {STATE_CROSSING_DATES}


def test_a_web_route_writes_its_delivery_event_under_the_open_project_only(
    tmp_path: Path,
) -> None:
    """A delivery through the web backend lands its delivery_events record under the project the
    backend has open, and nowhere else."""
    from fastapi.testclient import TestClient

    from tcip_web.app import app

    from tests.test_tcip_web_results_routes import _phenology_fixture

    other_project = tmp_path / "other_project"
    other_project.mkdir()

    body = _phenology_fixture(tmp_path, validated=True, fractions=(0.75, 1.0), detections=4)

    resp = TestClient(app, base_url="http://127.0.0.1").post(
        "/api/results/export_csv", json={**body, "payload": "milestones", "filename": "x.csv"})
    assert resp.status_code == 200, resp.text

    open_records = [
        r for r in read_delivery_events(tmp_path) if r["door"] == "results.export_csv"
    ]
    assert len(open_records) == 1, open_records
    assert read_delivery_events(other_project) == []


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

    assert read_delivery_events(tmp_path.resolve()) == []
    assert list(ts.read_log(audit_log_key(tmp_path)).records) == before


def test_record_delivery_binding_event_raises_on_a_failed_store_write_before_any_audit_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delivery whose record cannot be written raises into the delivering door, and no audit
    line names an event that was never recorded."""
    from tcip_mcp.audit import audit_log_key

    def _boom(*a, **kw):
        raise OSError("disk full")

    revision = seed_confirmed_count(tmp_path)
    monkeypatch.setattr(ts, "replace", _boom)

    with pytest.raises(OSError, match="disk full"):
        resolution.record_delivery_binding_event(
            "test_door", None, [], document_reconciliations={}, dimension_reconciliations={},
            acknowledgment=None, revision=revision,
            delivery_kind=STATE_CROSSING_DATES, project=tmp_path, plant_mapping=None,
        )

    monkeypatch.undo()
    assert read_delivery_events(tmp_path.resolve()) == []
    events = ts.read_log(audit_log_key(tmp_path)).records
    assert not any(e["tool"] == "delivery_event" for e in events)


def test_record_delivery_binding_event_raises_and_writes_nothing_when_plant_mapping_fails_validation(
    tmp_path: Path,
) -> None:
    """A caller's ``plant_mapping`` missing a required disclosure key is a shape violation in the
    caller, never an environmental failure, so it raises into the delivering door rather than
    falling into this function's own best-effort warning path (reserved for the store write)."""
    from pydantic import ValidationError

    bad_mapping = {
        "name": "valley", "dataset_id": "ds-1", "dataset_root": "data", "built_at": "2026-02-01T00:00:00+00:00",
        "record_sha256": "0" * 64, "nn_tolerance_m": {"value": 3, "source": "stated"},
        "capture_identity": {}, "captures_unverified": [], "plant_csvs_unverified": [],
        "images_unattributed_scope": "delivered_dates",
        # dates_delivered, images_unattributed and plant_attribution are missing.
    }

    with pytest.raises(ValidationError):
        resolution.record_delivery_binding_event(
            "test_door", None, [], document_reconciliations={}, dimension_reconciliations={},
            acknowledgment=None, revision=seed_confirmed_count(tmp_path),
            delivery_kind=STATE_CROSSING_DATES, project=tmp_path, plant_mapping=bad_mapping,
        )

    assert read_delivery_events(tmp_path) == []


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
            dimension_reconciliations={}, acknowledgment=None,
            revision=seed_confirmed_count(tmp_path),
            delivery_kind=STATE_CROSSING_DATES, project=tmp_path, plant_mapping=None,
        )

    assert read_delivery_events(tmp_path.resolve()) == []


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
            acknowledgment=None, revision=seed_confirmed_count(tmp_path),
            delivery_kind=STATE_CROSSING_DATES, project=tmp_path, plant_mapping=None,
        )

    assert read_delivery_events(tmp_path.resolve()) == []


def test_plant_mapping_union_resolves_each_shape_and_refuses_a_hybrid() -> None:
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
        "name": "valley", "dataset_id": "ds-1", "dataset_root": "data", "built_at": "2026-02-01T00:00:00+00:00",
        "record_sha256": "0" * 64, "nn_tolerance_m": {"value": 3, "source": "stated"},
        "capture_identity": {}, "captures_unverified": [], "plant_csvs_unverified": [],
        "dates_delivered": [], "images_unattributed": 0,
        "images_unattributed_scope": "delivered_dates", "plant_attribution": "image",
    }
    registry = {
        "plant_registry": {"name": "reg", "digest": "0" * 64},
        "raster_identity": {"width": 10, "height": 10},
        "nn_tolerance_m": {"value": 1, "source": "stated"}, "detections_unattributed": 0,
        "detections_unattributed_scope": "delivered_raster", "plant_attribution": "detection",
        "plants_outside_raster": [],
    }
    canopy = {
        "plant_registry": {"name": "reg", "digest": "0" * 64},
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
            "event_id": "e", "trait": "t", "trait_revision": 1, "trait_revision_sha256": "0" * 64,
            "delivery_kind": STATE_CROSSING_DATES, "door": "d",
            "output_path": None, "output_sha256": None,
            "acknowledged_by": None, "acknowledgment_reason": None,
            "plant_mapping": pm, "document_reconciliations": {},
            "dimension_reconciliations": {}, "produced_at": "t",
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

    assert read_delivery_events(tmp_path.resolve()) == []
    assert list(ts.read_log(audit_log_key(tmp_path)).records) == before


# ── the delivery-events route: read-only ─────────────────────────────────────

DELIVERY_EVENTS_ROUTE = "/api/results/delivery-events"


def _client():
    from fastapi.testclient import TestClient

    from tcip_web.app import app

    return TestClient(app, base_url="http://127.0.0.1")


def test_delivery_events_route_lists_a_recorded_event_with_its_revision(
    opened_project: Path,
) -> None:
    revision = seed_confirmed_count(opened_project)
    resolution.record_delivery_binding_event(
        "test_door", None, [], document_reconciliations={}, dimension_reconciliations={},
        acknowledgment=None, revision=revision, delivery_kind="per_image_count",
        project=opened_project,
    )

    resp = _client().get(DELIVERY_EVENTS_ROUTE)

    assert resp.status_code == 200
    (record,) = resp.json()["records"]
    assert record["door"] == "test_door"
    assert (record["trait"], record["trait_revision"], record["trait_revision_sha256"]) == (
        revision.entry.name, revision.number, revision.entry_sha256)
    assert record["delivery_kind"] == "per_image_count"


def test_delivery_events_route_refuses_while_no_project_is_open() -> None:
    resp = _client().get(DELIVERY_EVENTS_ROUTE)

    assert resp.status_code == 409


def test_delivery_events_route_refuses_a_stored_event_whose_plant_mapping_lacks_the_disclosure(
    opened_project: Path,
) -> None:
    """A record whose ``plant_mapping`` lacks ``dates_delivered``, ``images_unattributed`` and
    ``plant_attribution`` refuses the whole listing by event_id rather than being served with
    the gap silently absent."""
    event_id = "lacking-disclosure"
    key = resolution.delivery_event_key(project_state_dir(opened_project), event_id)
    ts.replace(
        key,
        {
            "event_id": event_id,
            "trait": "astringency",
            "delivery_kind": "state_crossing_dates",
            "door": "deliver_phenology_milestones",
            "output_path": None,
            "plant_mapping": {
                "name": "valley",
                "dataset_id": "ds-1",
                "dataset_root": "data",
                "built_at": "2026-02-01T00:00:00+00:00",
                "record_sha256": "0" * 64,
                "nn_tolerance_m": {"value": 3, "source": "stated"},
                "capture_identity": {},
                "captures_unverified": [],
                "plant_csvs_unverified": [],
                "images_unattributed_scope": "delivered_dates",
            },
            "produced_at": "2026-02-03T12:00:00+00:00",
        },
        expect=ts.Version.ABSENT,
    )

    resp = _client().get(DELIVERY_EVENTS_ROUTE)

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert event_id in detail
    assert "does not validate as a DeliveryEventRecord" in detail
    assert "dates_delivered" in detail


def test_delivery_events_route_serves_a_real_plant_mapping_disclosure_with_all_three_keys(
    opened_project: Path,
) -> None:
    """A ``plant_mapping`` built the way a real delivery builds it (``MappingBuild.
    delivery_disclosure``) carries all three keys, and the route serves them through unchanged."""
    from datetime import datetime, timezone

    from tcip_mcp.pipelines.postprocessing.plant_mapping import MappingBuild

    dates = ["2026-01-01", "2026-01-08"]
    build = MappingBuild(
        name="valley", dataset_root="data",
        dataset_id="ds-1", built_by="build_plant_mapping",
        built_at=datetime.now(timezone.utc).isoformat(),
        dates_requested=None, dates=dates,
        nn_tolerance_m={"value": 3.0, "source": "stated"},
        plant_registry={"name": "unregistered", "digest": "0" * 64},
        capture_identity={d: "0" * 16 for d in dates},
        capture_digests={d: {} for d in dates}, unreadable={d: [] for d in dates},
        assignments={}, record_sha256="0" * 64,
    )
    disclosure = build.delivery_disclosure(
        {"captures_unverified": [], "plant_csvs_unverified": []}, dates)
    revision = read_trait("bud_opening", opened_project).latest_confirmed
    assert revision is not None

    resolution.record_delivery_binding_event(
        "test_door", None, [], document_reconciliations={}, dimension_reconciliations={},
        acknowledgment=None, revision=revision,
        delivery_kind=STATE_CROSSING_DATES, project=opened_project, plant_mapping=disclosure,
    )

    resp = _client().get(DELIVERY_EVENTS_ROUTE)

    assert resp.status_code == 200
    (record,) = resp.json()["records"]
    mapping = record["plant_mapping"]
    assert mapping["dates_delivered"] == dates
    assert mapping["images_unattributed"] == 0
    assert mapping["plant_attribution"] == "image"
