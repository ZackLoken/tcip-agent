"""Coverage for the delivered-file digest (``output_sha256``) and ``supersede_delivery``: a
delivery event records the bytes it shipped, and a supersession states that a delivered number is
withdrawn or replaced without touching the file or the event.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp.pipelines import resolution
from tcip_mcp.pipelines.delivery_events_schema import DeliveryEventRecord, with_supersessions
from tcip_mcp.tools.delivery_tools import supersede_delivery
from tcip_mcp.tools.phenology_tools import build_plant_mapping, deliver_phenology_milestones

from tests._binding_fixtures import register_plant_registry_for
from tests.test_plant_mapping_binding import DATES, _dataset, _init, _validate_buckets
from tests.test_second_trait_acceptance import _seed_currant_bloom_trait


def _delivered_scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """A real phenology delivery, through the platform's own doors, whose delivery event this
    module's tests then supersede."""
    from tests.test_plant_mapping_binding import _write_scene

    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])
    registry = register_plant_registry_for([plant_csv])
    build_res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)
    _validate_buckets(preds_by_date, dataset_root)
    return preds_by_date


def _one_event(tmp_path: Path, door: str = "deliver_phenology_milestones") -> dict:
    scope = resolution.delivery_events_scope(tmp_path)
    keys = ts.keys(resolution.DELIVERY_EVENTS_STORE, str(scope))
    events = [ts.read(k) for k in keys if ts.read(k)["door"] == door]
    assert len(events) == 1, events
    return events[0]


def test_a_delivered_csv_carries_the_written_files_own_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coverage: the same digest, end to end through the real phenology delivery door (the
    isolated GUARDS proof for the writer itself lives in test_delivery_output_digest_guard.py)."""
    preds_by_date = _delivered_scene(tmp_path, monkeypatch)
    out_csv = tmp_path / "out.csv"
    res = deliver_phenology_milestones(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv), classifier_pred_dirs=list(preds_by_date.values()))
    assert "error" not in res, res

    event = _one_event(tmp_path)
    assert event["output_sha256"] == hashlib.sha256(out_csv.read_bytes()).hexdigest()
    DeliveryEventRecord.model_validate(event)


def test_a_fileless_event_carries_no_digest(tmp_path: Path) -> None:
    resolution.record_delivery_binding_event(
        "test_door", None, [], {}, measurement_documents=["operating_point"],
        scale_document=None, acknowledgement=None, trait="astringency",
        delivery_kind="state_crossing_dates", project_root=tmp_path, plant_mapping=None,
    )
    event = _one_event(tmp_path, door="test_door")
    assert event["output_sha256"] is None


def test_supersede_delivery_over_a_real_event_records_the_withdrawal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admits valid work: supersede_delivery over an event the platform's own writer recorded."""
    preds_by_date = _delivered_scene(tmp_path, monkeypatch)
    out_csv = tmp_path / "out.csv"
    res = deliver_phenology_milestones(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv), classifier_pred_dirs=list(preds_by_date.values()))
    assert "error" not in res, res
    event = _one_event(tmp_path)

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    outcome = supersede_delivery(event["event_id"], "a mis-stated crop was corrected upstream")

    assert "error" not in outcome, outcome
    assert outcome["superseded_event_id"] == event["event_id"]
    assert outcome["output_sha256"] == event["output_sha256"]
    assert outcome["replacement_event_id"] is None

    stored = ts.read(resolution.delivery_supersession_key(
        resolution.delivery_events_scope(tmp_path), event["event_id"]))
    assert stored["reason"] == "a mis-stated crop was corrected upstream"
    assert stored["output_sha256"] == event["output_sha256"]
    assert out_csv.exists(), "a supersession never deletes or rewrites the delivered file"


def test_supersede_delivery_names_a_replacement_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    preds_by_date = _delivered_scene(tmp_path, monkeypatch)
    first_csv = tmp_path / "first.csv"
    assert "error" not in deliver_phenology_milestones(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(first_csv), classifier_pred_dirs=list(preds_by_date.values()))
    first_event = _one_event(tmp_path)

    second_csv = tmp_path / "second.csv"
    assert "error" not in deliver_phenology_milestones(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(second_csv), classifier_pred_dirs=list(preds_by_date.values()))
    scope = resolution.delivery_events_scope(tmp_path)
    events = [ts.read(k) for k in ts.keys(resolution.DELIVERY_EVENTS_STORE, str(scope))
             if ts.read(k)["door"] == "deliver_phenology_milestones"]
    second_event = next(e for e in events if e["event_id"] != first_event["event_id"])

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    outcome = supersede_delivery(
        first_event["event_id"], "re-delivered with a corrected mapping",
        replacement_event_id=second_event["event_id"])

    assert "error" not in outcome, outcome
    assert outcome["replacement_event_id"] == second_event["event_id"]


def test_supersede_delivery_refuses_an_unknown_event_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    (tmp_path / ".tcip").mkdir(parents=True, exist_ok=True)

    res = supersede_delivery("does-not-exist", "some reason")

    assert "error" in res
    assert "not found" in res["error"]


def test_supersede_delivery_refuses_an_empty_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))

    res = supersede_delivery("whatever", "   ")

    assert "error" in res
    assert "reason" in res["error"]


def test_supersede_delivery_refuses_an_unknown_replacement_event_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    preds_by_date = _delivered_scene(tmp_path, monkeypatch)
    out_csv = tmp_path / "out.csv"
    assert "error" not in deliver_phenology_milestones(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv), classifier_pred_dirs=list(preds_by_date.values()))
    event = _one_event(tmp_path)

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    res = supersede_delivery(
        event["event_id"], "some reason", replacement_event_id="also-does-not-exist")

    assert "error" in res
    assert "replacement_event_id" in res["error"]


def test_supersede_delivery_refuses_a_second_supersession_of_the_same_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    preds_by_date = _delivered_scene(tmp_path, monkeypatch)
    out_csv = tmp_path / "out.csv"
    assert "error" not in deliver_phenology_milestones(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv), classifier_pred_dirs=list(preds_by_date.values()))
    event = _one_event(tmp_path)

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    assert "error" not in supersede_delivery(event["event_id"], "first reason")

    res = supersede_delivery(event["event_id"], "second reason")

    assert "error" in res
    assert "already carries a supersession" in res["error"]


def test_supersede_delivery_refuses_an_event_missing_the_acknowledgement_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record stored before ``acknowledged_by``/``acknowledgement_reason`` existed does not
    validate against ``DeliveryEventRecord``, the same shape the Results tab's panel route
    refuses to list; ``supersede_delivery`` reads the event it supersedes through the identical
    check, so it refuses too rather than quietly superseding a record the panel would reject."""
    scope = resolution.delivery_events_scope(tmp_path)
    event_id = "pre-acknowledgement-event"
    ts.replace(resolution.delivery_event_key(scope, event_id), {
        "event_id": event_id, "trait": "currant_bloom", "delivery_kind": "state_crossing_dates",
        "door": "deliver_phenology_milestones", "output_path": str(tmp_path / "out.csv"),
        "output_sha256": "0" * 64, "measurement_documents": ["operating_point"],
        "scale_document": None, "plant_mapping": None, "documents": {},
        "produced_at": "2026-02-11T00:00:00+00:00",
    })

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    res = supersede_delivery(event_id, "some reason")

    assert "error" in res
    assert "does not validate" in res["error"]
    assert "no operator door rewrites an existing delivery_events record" in res["error"]
    assert not ts.exists(resolution.delivery_supersession_key(scope, event_id))


def test_supersede_delivery_refuses_a_replacement_event_missing_the_acknowledgement_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The replacement event is read through the identical validating check as the superseded
    one, so a malformed replacement refuses by name rather than being cited unread."""
    preds_by_date = _delivered_scene(tmp_path, monkeypatch)
    out_csv = tmp_path / "out.csv"
    assert "error" not in deliver_phenology_milestones(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv), classifier_pred_dirs=list(preds_by_date.values()))
    event = _one_event(tmp_path)

    scope = resolution.delivery_events_scope(tmp_path)
    replacement_id = "pre-acknowledgement-replacement"
    ts.replace(resolution.delivery_event_key(scope, replacement_id), {
        "event_id": replacement_id, "trait": "currant_bloom",
        "delivery_kind": "state_crossing_dates", "door": "deliver_phenology_milestones",
        "output_path": str(tmp_path / "out2.csv"), "output_sha256": "1" * 64,
        "measurement_documents": ["operating_point"], "scale_document": None,
        "plant_mapping": None, "documents": {}, "produced_at": "2026-02-11T00:00:00+00:00",
    })

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    res = supersede_delivery(
        event["event_id"], "some reason", replacement_event_id=replacement_id)

    assert "error" in res
    assert "does not validate" in res["error"]
    assert not ts.exists(resolution.delivery_supersession_key(scope, event["event_id"]))


def test_with_supersessions_attaches_only_the_matching_events_own_record() -> None:
    events = [{"event_id": "a"}, {"event_id": "b"}]
    supersessions = {"a": {"reason": "withdrawn"}}

    joined = with_supersessions(events, supersessions)

    assert joined[0]["superseded"] == {"reason": "withdrawn"}
    assert joined[1]["superseded"] is None
