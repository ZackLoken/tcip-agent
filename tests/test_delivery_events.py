"""The project-scoped ``delivery_events`` record ``deliver_csv`` writes once per delivery.

The record carries the gate's own finding for each delivered bucket and the producer once; it is
validated against :class:`~tcip_mcp.pipelines.delivery_events_schema.DeliveryEventRecord` before
it is written, and a look on screen records none.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp import delivery
from tcip_mcp.delivery import read_delivery_events
from tcip_mcp.traits import PER_IMAGE_COUNT, STATE_CROSSING_DATES, read_trait
from tests._chain_fixtures import deliver_milestones
from tests._mapping_fixtures import POSITION_ERROR_M
from tests._trait_fixtures import COUNT_SUBJECT, seed_confirmed_count


def _bucket(project: Path):
    """An unassessed bucket of one document detecting nothing, counting the count trait's
    measured subject."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import predicted, published

    return published(project, "m/2026-01-01",
                     [predicted(project / "ds" / "images" / "2026-01-01" / "a.png", [])],
                     scope={"subject": COUNT_SUBJECT})


def _cleared(project: Path, revision, bucket, result):
    """The gate's clearance of ``result``, a per-image count delivery under ``revision`` over the
    unassessed ``bucket``, shipped under a breeder's acknowledgment of it."""
    from tests._chain_fixtures import acknowledged

    return acknowledged(project, lambda ack: delivery.gate(
        project, [bucket], delivery_kind=PER_IMAGE_COUNT, revision=revision, result=result,
        acknowledgment_id=ack), reason="an unassessed fixture bucket")


def _record(project: Path, revision, bucket=None, **disclosed) -> dict:
    """``deliver_csv`` of an empty per-image count delivery under ``revision`` over ``bucket``
    (by default a fresh :func:`_bucket`), shipped under a breeder's acknowledgment."""
    result = delivery.Result(delivery.DELIVERY_COLUMNS, [], population=[], **disclosed)
    clearance = _cleared(project, revision, bucket if bucket is not None else _bucket(project),
                         result)
    return delivery.deliver_csv(project, project / "out" / "counts.csv", result,
                                clearance=clearance, revision=revision, door="test_door",
                                delivery_kind=PER_IMAGE_COUNT, actor=None)


def test_a_completed_crossing_delivery_writes_the_gates_finding_for_every_bucket(tmp_path):
    pytest.importorskip("torch")
    from tests._chain_fixtures import attributed_series

    series = attributed_series(tmp_path, fractions=(0.0, 1.0))
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_milestones(tmp_path, series.body(), out_csv)

    assert "error" not in res, res
    (record,) = read_delivery_events(tmp_path)
    shipped_under = read_trait("bud_opening", tmp_path).latest_confirmed
    assert shipped_under is not None
    assert (record.trait, record.trait_revision, record.trait_revision_sha256) == (
        "bud_opening", shipped_under.number, shipped_under.entry_sha256)
    assert record.delivery_kind == STATE_CROSSING_DATES
    assert record.output_path == str(out_csv)
    assert record.validated is True and record.acknowledgment is None
    assert record.producer.model_dump() == series.assessment["producer"]
    assert {b.bucket for b in record.buckets} == set(series.buckets.values())
    assert {b.dataset_root for b in record.buckets} == {str(series.root)}
    for finding in record.buckets:
        assert (finding.assessment_id, finding.validated, finding.reason) == (
            series.assessment["assessment_id"], True, None)
    assert record.population == ["PLANT_A", "PLANT_B"]
    assert record.plant_mapping is not None and record.plant_mapping.name == series.mapping_name


def test_two_deliveries_of_the_same_trait_and_kind_both_enumerate_distinctly(tmp_path):
    pytest.importorskip("torch")
    from tests._chain_fixtures import attributed_series

    body = attributed_series(tmp_path, fractions=(0.0, 1.0)).body()
    first_csv, second_csv = tmp_path / "out" / "first.csv", tmp_path / "out" / "second.csv"
    for out_csv in (first_csv, second_csv):
        assert "error" not in deliver_milestones(tmp_path, body, out_csv)

    records = read_delivery_events(tmp_path)
    assert len(records) == 2, records
    assert records[0].event_id != records[1].event_id
    assert {r.output_path for r in records} == {str(first_csv), str(second_csv)}


def test_a_web_route_writes_its_delivery_event_under_the_open_project_only(tmp_path, client):
    """A delivery through the web backend lands its record under the project the backend has
    open, and nowhere else."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import attributed_series

    other_project = tmp_path / "other_project"
    other_project.mkdir()
    body = attributed_series(tmp_path, fractions=(0.0, 1.0)).body()

    resp = client.post(
        "/api/results/export_csv",
        json={**body, "payload": "milestones", "filename": "x.csv", "user": "tester"})

    assert resp.status_code == 200, resp.text
    assert [r.door for r in read_delivery_events(tmp_path)] == ["results.export_csv"]
    assert read_delivery_events(other_project) == []


def test_a_look_on_screen_records_no_delivery_event_and_no_audit_line(tmp_path, client):
    """Looking at a number is not delivering it: phenology_measurement changes no state."""
    pytest.importorskip("torch")
    from tcip_mcp.audit import audit_log_key
    from tests._chain_fixtures import attributed_series

    series = attributed_series(tmp_path, fractions=(0.0, 1.0))
    logs = (audit_log_key(tmp_path), audit_log_key(series.root))
    before = [list(ts.read_log(k).records) for k in logs]

    resp = client.post(
        "/api/results/phenology_measurement", json=series.body())

    assert resp.status_code == 200, resp.text
    assert resp.json()["positive_class_assessed"] is True
    assert read_delivery_events(tmp_path) == []
    assert [list(ts.read_log(k).records) for k in logs] == before


def test_a_record_that_cannot_be_written_raises_before_any_audit_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delivery event that cannot be written raises into the delivering door, and no audit line
    names an event that was never recorded."""
    from tcip_mcp.audit import audit_log_key

    revision = seed_confirmed_count(tmp_path)
    result = delivery.Result(delivery.DELIVERY_COLUMNS, [], population=[])
    clearance = _cleared(tmp_path, revision, _bucket(tmp_path), result)
    replace = ts.replace

    def _event_write_fails(key, *a, **kw):
        if key.store == delivery.DELIVERY_EVENTS_STORE:
            raise OSError("disk full")
        return replace(key, *a, **kw)

    monkeypatch.setattr(ts, "replace", _event_write_fails)

    with pytest.raises(OSError, match="disk full"):
        delivery.deliver_csv(tmp_path, tmp_path / "out" / "counts.csv", result,
                             clearance=clearance, revision=revision, door="test_door",
                             delivery_kind=PER_IMAGE_COUNT, actor=None)

    monkeypatch.undo()
    assert read_delivery_events(tmp_path.resolve()) == []
    events = ts.read_log(audit_log_key(tmp_path)).records
    assert not any(e["tool"] == "delivery_event" for e in events)


def test_a_plant_mapping_missing_a_disclosure_key_raises_and_writes_nothing(tmp_path: Path):
    """A ``plant_mapping`` missing a required disclosure key refuses before the CSV or its event
    is written."""
    from pydantic import ValidationError

    bad_mapping = {
        "name": "valley", "dataset_id": "ds-1", "dataset_root": "data",
        "built_at": "2026-02-01T00:00:00+00:00", "record_sha256": "0" * 64,
        "nn_tolerance_m": {"value": 3, "source": "stated"}, "capture_identity": {},
        "captures_unverified": [], "plant_csvs_unverified": [],
        "images_unattributed_scope": "delivered_dates",
    }

    with pytest.raises(ValidationError):
        _record(tmp_path, seed_confirmed_count(tmp_path), plant_mapping=bad_mapping)

    assert read_delivery_events(tmp_path) == []
    assert not (tmp_path / "out" / "counts.csv").exists()


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
        "name": "valley", "dataset_id": "ds-1", "dataset_root": "data",
        "built_at": "2026-02-01T00:00:00+00:00",
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
        "canopy_segments": {"capture": "2026-01-01", "stem": "x", "sha256": "0" * 64,
                            "subject": "canopy", "n_segments": 1},
        "position_error_m": POSITION_ERROR_M,
        "segment_ties": [], "segments_without_plant": 0, "plants_outside_raster": [],
        "plants_without_segment": [], "plants_within_position_error": [],
        "plants_with_ambiguous_detections": [], "detections_unattributed": 0,
        "detections_unattributed_by_source": {
            "outside_segments": 0, "overlapping_segments": 0, "segment_without_plant": 0,
            "segment_plant_within_position_error": 0},
        "detections_unattributed_scope": "delivered_raster", "plant_attribution": "segment",
    }

    def _resolved(pm: dict) -> object:
        record = {
            "event_id": "e", "door": "d", "delivery_kind": STATE_CROSSING_DATES, "trait": "t",
            "trait_revision": 1, "trait_revision_sha256": "0" * 64,
            "output_path": "out.csv", "output_sha256": "0" * 64,
            "producer": {"checkpoint_sha256": "0" * 64, "experiment_id": None}, "buckets": [],
            "scale_assessment_id": None, "validated": True, "acknowledgment": None,
            "population": [], "require_all_dates_complete": None,
            "plant_mapping": pm, "produced_at": "t",
        }
        return DeliveryEventRecord.model_validate(record).plant_mapping

    assert isinstance(_resolved(mapping), PlantMappingDisclosure)
    assert isinstance(_resolved(registry), PlantRegistryDisclosure)
    assert isinstance(_resolved(canopy), CanopySegmentDisclosure)

    hybrid = {**registry, "canopy_segments": canopy["canopy_segments"]}
    with pytest.raises(ValidationError):
        _resolved(hybrid)


# ── the delivery-events route: read-only ─────────────────────────────────────

DELIVERY_EVENTS_ROUTE = "/api/results/delivery-events"


def test_delivery_events_route_lists_a_recorded_event_with_its_revision(
    opened_project: Path, client,
) -> None:
    revision = seed_confirmed_count(opened_project)
    _record(opened_project, revision)

    resp = client.get(DELIVERY_EVENTS_ROUTE)

    assert resp.status_code == 200
    (record,) = resp.json()["records"]
    assert record["door"] == "test_door"
    assert (record["trait"], record["trait_revision"], record["trait_revision_sha256"]) == (
        revision.entry.name, revision.number, revision.entry_sha256)
    assert record["delivery_kind"] == PER_IMAGE_COUNT


def test_delivery_events_route_refuses_while_no_project_is_open(client) -> None:
    resp = client.get(DELIVERY_EVENTS_ROUTE)

    assert resp.status_code == 409


def test_delivery_events_route_serves_a_delivered_plant_mapping_disclosure_unchanged(
    tmp_path: Path, client,
) -> None:
    """The disclosure a phenology delivery records over a built mapping is the one the route
    serves."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import attributed_series

    series = attributed_series(tmp_path, fractions=(0.0, 1.0))
    assert "error" not in deliver_milestones(tmp_path, series.body(),
                                             tmp_path / "out" / "bud.csv")
    (recorded,) = read_delivery_events(tmp_path)

    resp = client.get(DELIVERY_EVENTS_ROUTE)

    assert resp.status_code == 200
    (record,) = resp.json()["records"]
    assert recorded.plant_mapping is not None
    assert record["plant_mapping"] == recorded.plant_mapping.model_dump(mode="json")
    assert record["plant_mapping"]["dates_delivered"] == sorted(series.buckets)
