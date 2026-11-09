"""The project-scoped ``delivery_events`` record ``deliver_csv`` writes for a delivery.

The record carries the gate's own finding for each delivered bucket and the producer; a record
:class:`~tcip_mcp.pipelines.delivery_events_schema.DeliveryEventRecord` refuses is not written,
and a look on screen records none.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import tcip_store as ts
from tcip_mcp import delivery
from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.delivery import read_delivery_events
from tcip_mcp.traits import PER_IMAGE_COUNT, STATE_CROSSING_DATES, read_trait
from tests._audit_fixtures import audit_rows
from tests._chain_fixtures import deliver_milestones
from tests._mapping_fixtures import POSITION_ERROR_M
from tests._trait_fixtures import COUNT_SUBJECT, seed_confirmed_count

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi.testclient import TestClient
    from pydantic import JsonValue

    from tcip_mcp.buckets import Bucket
    from tcip_mcp.delivery import Delivered
    from tcip_mcp.traits import TraitRevision
    from tcip_store import Key, Version


def _bucket(project: Path) -> Bucket:
    """Publish an unassessed bucket of one document detecting nothing and return it.

    The bucket counts the count trait's measured subject.
    """
    pytest.importorskip("torch")
    from tests._chain_fixtures import predicted, published

    return published(
        project,
        "m/2026-01-01",
        [predicted(image_dir(project / "ds", "2026-01-01") / "a.png", [])],
        scope={"subject": COUNT_SUBJECT},
    )


def _cleared(
    project: Path,
    revision: TraitRevision,
    bucket: Bucket,
    result: delivery.Result,
) -> delivery.Clearance:
    """Return the gate's clearance of ``result``, shipped under a breeder's acknowledgment of it.

    ``result`` is a per-image count delivery under ``revision`` over the unassessed ``bucket``.
    """
    from tests._chain_fixtures import acknowledged

    return acknowledged(
        project,
        lambda ack: delivery.gate(
            project,
            [bucket],
            delivery_kind=PER_IMAGE_COUNT,
            revision=revision,
            result=result,
            acknowledgment_id=ack,
        ),
        reason="an unassessed fixture bucket",
    )


def _record(
    project: Path,
    revision: TraitRevision,
    bucket: Bucket | None = None,
) -> Delivered:
    """Run ``deliver_csv`` of an empty per-image count delivery under ``revision``.

    The delivery is over ``bucket`` (by default a fresh :func:`_bucket`), shipped under a
    breeder's acknowledgment.
    """
    result = delivery.Result(delivery.DELIVERY_COLUMNS, [], population=[])
    clearance = _cleared(
        project,
        revision,
        bucket if bucket is not None else _bucket(project),
        result,
    )
    return delivery.deliver_csv(
        project,
        project / "out" / "counts.csv",
        result,
        clearance=clearance,
        revision=revision,
        door="test_door",
        delivery_kind=PER_IMAGE_COUNT,
        actor=None,
    )


def _written(project: Path) -> Delivered:
    """Deliver an empty per-image count CSV through ``deliver_csv`` (:func:`_record`)."""
    return _record(project, seed_confirmed_count(project))


def _counted(project: Path) -> Delivered:
    """Deliver per-image counts of :func:`_bucket`, under a breeder's acknowledgment."""
    from tcip_mcp.pipelines.postprocessing.export import (
        deliver_per_image_counts_csv,
    )
    from tests._chain_fixtures import acknowledged

    revision = seed_confirmed_count(project)
    bucket = _bucket(project)
    return acknowledged(
        project,
        lambda ack: deliver_per_image_counts_csv(
            project,
            bucket.root,
            bucket.name,
            str(project / "out" / "counts.csv"),
            trait=revision.entry.name,
            acknowledgment_id=ack,
            door="test_door",
            actor=None,
        ),
        reason="an unassessed fixture bucket",
    )


def _aggregated(project: Path) -> Delivered:
    """Deliver one plant's per-image counts as a per-plant aggregate, under an acknowledgment."""
    from tcip_mcp.traits import PER_PLANT_COUNT_AGGREGATE
    from tests import _trait_fixtures as fx
    from tests._chain_fixtures import deliver_acknowledged

    fx.seed_delivery_traits(project)
    fx.seed_confirmed_aggregate(
        project, "stem_count", value_keys=["count"], measured_subject="bud"
    )
    return deliver_acknowledged(
        project,
        {"P1": [3, 3]},
        project / "out" / "plants.csv",
        "stem_count",
        value_key="count",
        delivery_kind=PER_PLANT_COUNT_AGGREGATE,
    )


def _measured(project: Path) -> Delivered:
    """Deliver a mapped phenology's milestones, under a breeder's acknowledgment.

    The scene is ``test_plant_mapping_binding``'s: a registered dataset, its published buckets
    and a mapping built over them, measured for the second trait.
    """
    from tcip_mcp.pipelines.postprocessing import phenology
    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tests import test_plant_mapping_binding as binding
    from tests._chain_fixtures import acknowledged
    from tests._mapping_fixtures import register_plant_registry_for

    binding._init(project)
    dataset_root = binding._dataset(project)
    plant_csv, preds_by_date = binding._write_scene(dataset_root)
    built = build_plant_mapping(
        project,
        name="valley",
        dataset_root=str(dataset_root),
        plant_registry=register_plant_registry_for(project, [plant_csv]),
        actor=None,
    )
    assert "error" not in built, built
    binding._seed_currant_bloom_trait(project)
    measurement = phenology.measure_phenology(
        project,
        trait="currant_bloom",
        mapping_name="valley",
        dataset_root=dataset_root,
        buckets=list(preds_by_date.values()),
        plants=binding.POPULATION,
    )
    return acknowledged(
        project,
        lambda ack: phenology.deliver_phenology(
            project,
            measurement,
            projection="milestones",
            output_path=project / "out" / "milestones.csv",
            acknowledgment_id=ack,
            door="test_door",
            actor=None,
        ),
        reason="an unassessed fixture bucket",
    )


def _counted_per_plant(project: Path) -> Delivered:
    """Deliver per-plant counts from a whole-raster bucket, under a breeder's acknowledgment.

    The scene is ``test_orthomosaic_tools``'s plant grid over one detection.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import load_registry
    from tcip_mcp.tools.orthomosaic_tools import orthomosaic_plant_counts
    from tests import _trait_fixtures as fx, test_orthomosaic_tools as ortho
    from tests._chain_fixtures import acknowledged

    fx.seed_delivery_traits(project)
    fx.seed_confirmed_orthomosaic_count(project, "stem_count")
    raster_path = ortho._raster(project)
    ortho._write_geo_raster(raster_path)
    bucket = ortho._raster_bucket(
        project, raster_path, [(8.0, 8.0, 12.0, 12.0)]
    )
    registry = ortho._plant_registry(
        project,
        ortho._plant_grid_csv(project, raster_path, ortho._PLANT_PIXELS),
    )
    return acknowledged(
        project,
        lambda ack: orthomosaic_plant_counts(
            project,
            bucket.root,
            bucket.name,
            load_registry(project, registry),
            project / "out" / "counts.csv",
            "stem_count",
            ortho._GRID,
            acknowledgment_id=ack,
            door="test_door",
            actor=None,
        ),
        reason="an unassessed fixture bucket",
    )


@pytest.mark.parametrize(
    "deliver",
    [_written, _counted, _aggregated, _measured, _counted_per_plant],
    ids=lambda deliver: deliver.__name__.strip("_"),
)
def test_a_delivery_answers_the_fields_of_the_event_it_wrote(
    tmp_path: Path, deliver: Callable[[Path], Delivered]
) -> None:
    """Each delivery's answer is its one delivery event's own fields as the event is served.

    Every field ``Delivered`` declares equals the event's field of the same name that the reader
    serves for that id, each path in the served form and neither side normalized.
    """
    pytest.importorskip("torch")

    delivered = deliver(tmp_path)

    (event,) = (
        e
        for e in read_delivery_events(tmp_path)
        if e.event_id == delivered.event_id
    )
    served = event.model_dump(mode="json")
    answered = delivered.model_dump(mode="json")
    fields = delivery.Delivered.model_fields
    assert {key: answered[key] for key in fields} == {
        key: served[key] for key in fields
    }


def _acknowledged_for(project: Path, refused: Callable[[], object]) -> str:
    """Return the id of a breeder's acknowledgment of the delivery ``refused()`` refuses."""
    from tcip_mcp.delivery import DeliveryRefusedError, record_acknowledgment
    from tests._chain_fixtures import digest_of

    with pytest.raises(DeliveryRefusedError) as refusal:
        refused()
    return record_acknowledgment(
        project,
        acknowledged_by="user:breeder",
        reason="an unassessed fixture bucket",
        result_sha256=digest_of(refusal.value),
    ).acknowledgment_id


def _per_image_counts_call(project: Path) -> tuple[str, dict]:
    """Return ``deliver_per_image_counts``'s call over :func:`_bucket`, acknowledged."""
    from tcip_mcp.pipelines.postprocessing.export import (
        deliver_per_image_counts_csv,
    )
    from tests._web_fixtures import named_project

    named_project(project, "Counts")
    trait = seed_confirmed_count(project).entry.name
    bucket = _bucket(project)
    out = str(project / "out" / "counts.csv")
    return "deliver_per_image_counts", {
        "dataset_root": str(bucket.root),
        "bucket": bucket.name,
        "output_path": out,
        "trait": trait,
        "acknowledgment_id": _acknowledged_for(
            project,
            lambda: deliver_per_image_counts_csv(
                project,
                bucket.root,
                bucket.name,
                out,
                trait=trait,
                acknowledgment_id=None,
                door="deliver_per_image_counts",
                actor=None,
            ),
        ),
    }


def _per_plant_csv_arguments(project: Path) -> dict:
    """Return ``deliver_per_plant_csv``'s arguments over one mapped plant's image, unacknowledged.

    The scene is ``test_deliver_per_plant_csv``'s; the row states every field the record holds.
    """
    from tests import _trait_fixtures as fx, test_deliver_per_plant_csv as door

    fx.seed_delivery_traits(project)
    fx.seed_confirmed_aggregate(project, "stem_count", value_keys=["count"])
    dataset_root, date = door._mapped(project)
    return {
        "results": [{"date": date, "stem": "P1", "value": 2}],
        "output_path": str(project / "out" / "plants.csv"),
        "delivered_phenotype": "stem_count",
        "delivery_kind": door.KIND,
        "plants": ["P1"],
        "dataset_root": str(dataset_root),
        "buckets": [door._bucket(project, dataset_root, date, {"P1": 2})],
        "value_key": "count",
        "plant_mapping": "valley",
    }


def _per_plant_csv_call(project: Path) -> tuple[str, dict]:
    """Return ``deliver_per_plant_csv``'s call over its mapped scene's arguments, acknowledged."""
    return _acknowledged_per_plant_csv(
        project, _per_plant_csv_arguments(project)
    )


def _acknowledged_per_plant_csv(
    project: Path, arguments: dict
) -> tuple[str, dict]:
    """Return ``deliver_per_plant_csv``'s call over ``arguments``, acknowledged for exactly them."""
    from tcip_mcp.buckets import by_recorded_date, read_bucket
    from tcip_mcp.pipelines.postprocessing.aggregation import (
        deliver_per_plant_aggregate,
    )
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        plant_mapping_disclosure,
    )
    from tests.test_deliver_per_plant_csv import door_results

    published = [
        read_bucket(Path(arguments["dataset_root"]), name)
        for name in arguments["buckets"]
    ]
    resolved = plant_mapping_disclosure(
        project,
        arguments["plant_mapping"],
        by_recorded_date(published),
        arguments["plants"],
    )
    return "deliver_per_plant_csv", {
        **arguments,
        "acknowledgment_id": _acknowledged_for(
            project,
            lambda: deliver_per_plant_aggregate(
                project,
                door_results(arguments["results"]),
                arguments["output_path"],
                value_key=arguments["value_key"],
                delivered_phenotype=arguments["delivered_phenotype"],
                delivery_kind=arguments["delivery_kind"],
                buckets=published,
                plants=arguments["plants"],
                mapping=resolved,
                door="deliver_per_plant_csv",
                actor=None,
            ),
        ),
    }


def _orthomosaic_call(project: Path) -> tuple[str, dict]:
    """Return ``deliver_orthomosaic_plant_counts``'s call over a plant grid, acknowledged.

    The scene is ``test_orthomosaic_tools``'s plant grid over one detection.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import load_registry
    from tcip_mcp.tools.orthomosaic_tools import orthomosaic_plant_counts
    from tests import _trait_fixtures as fx, test_orthomosaic_tools as ortho
    from tests._web_fixtures import named_project

    named_project(project, "Orchard")
    fx.seed_delivery_traits(project)
    fx.seed_confirmed_orthomosaic_count(project, "stem_count")
    raster_path = ortho._raster(project)
    ortho._write_geo_raster(raster_path)
    bucket = ortho._raster_bucket(
        project, raster_path, [(8.0, 8.0, 12.0, 12.0)]
    )
    registry = ortho._plant_registry(
        project,
        ortho._plant_grid_csv(project, raster_path, ortho._PLANT_PIXELS),
    )
    out = project / "out" / "counts.csv"
    return "deliver_orthomosaic_plant_counts", {
        "dataset_root": str(bucket.root),
        "bucket": bucket.name,
        "plant_registry": registry,
        "output_csv_path": str(out),
        "delivered_phenotype": "stem_count",
        "plants": ortho._GRID,
        "acknowledgment_id": _acknowledged_for(
            project,
            lambda: orthomosaic_plant_counts(
                project,
                bucket.root,
                bucket.name,
                load_registry(project, registry),
                out,
                "stem_count",
                ortho._GRID,
                door="deliver_orthomosaic_plant_counts",
                actor=None,
            ),
        ),
    }


def _milestones_call(project: Path) -> tuple[str, dict]:
    """Return ``deliver_phenology_milestones``'s call over a mapped series, acknowledged.

    The scene is ``test_plant_mapping_binding``'s, measured for the second trait.
    """
    from tcip_mcp.pipelines.postprocessing import phenology
    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tests import test_plant_mapping_binding as binding
    from tests._mapping_fixtures import register_plant_registry_for

    binding._init(project)
    dataset_root = binding._dataset(project)
    plant_csv, preds_by_date = binding._write_scene(dataset_root)
    built = build_plant_mapping(
        project,
        name="valley",
        dataset_root=str(dataset_root),
        plant_registry=register_plant_registry_for(project, [plant_csv]),
        actor=None,
    )
    assert "error" not in built, built
    binding._seed_currant_bloom_trait(project)
    out = project / "out" / "milestones.csv"
    measurement = phenology.measure_phenology(
        project,
        trait="currant_bloom",
        mapping_name="valley",
        dataset_root=dataset_root,
        buckets=list(preds_by_date.values()),
        plants=binding.POPULATION,
    )
    return "deliver_phenology_milestones", {
        "trait": "currant_bloom",
        "mapping_name": "valley",
        "dataset_root": str(dataset_root),
        "buckets": list(preds_by_date.values()),
        "output_csv_path": str(out),
        "plants": binding.POPULATION,
        "acknowledgment_id": _acknowledged_for(
            project,
            lambda: phenology.deliver_phenology(
                project,
                measurement,
                projection="milestones",
                output_path=out,
                acknowledgment_id=None,
                door="deliver_phenology_milestones",
                actor=None,
            ),
        ),
    }


@pytest.mark.parametrize(
    "call",
    [
        _per_image_counts_call,
        _per_plant_csv_call,
        _orthomosaic_call,
        _milestones_call,
    ],
    ids=lambda call: call.__name__.strip("_").removesuffix("_call"),
)
def test_a_delivery_tool_answers_the_fields_of_the_event_it_wrote(
    tmp_path: Path, call: Callable[[Path], tuple[str, dict]]
) -> None:
    """Each delivery tool, called through the server's door, answers its event's own fields.

    The tool runs as an MCP client calls it, under the project it is bound to and a breeder's
    recorded acknowledgment; every field ``Delivered`` declares equals the event's field of the
    same name that the reader serves for that id, neither side normalized.
    """
    from tcip_mcp.server import binding_for
    from tests._server_calls import called_through_the_server

    pytest.importorskip("torch")
    tool, arguments = call(tmp_path)
    bound = binding_for(tmp_path)

    answer = called_through_the_server(tool, arguments, bound=bound)

    assert isinstance(answer, dict) and "error" not in answer, answer
    (event,) = (
        e
        for e in read_delivery_events(bound.root)
        if e.event_id == answer["event_id"]
    )
    served = event.model_dump(mode="json")
    fields = delivery.Delivered.model_fields
    assert {key: answer[key] for key in fields} == {
        key: served[key] for key in fields
    }


@pytest.mark.parametrize(
    "call",
    [_per_plant_csv_call, _milestones_call],
    ids=lambda call: call.__name__.strip("_").removesuffix("_call"),
)
def test_a_mapped_delivery_through_either_door_discloses_no_plant_csv_list(
    tmp_path: Path, call: Callable[[Path], tuple[str, dict]]
) -> None:
    """A mapped delivery through either door ships disclosing nothing it could not verify.

    The tool is called through the server under a breeder's recorded acknowledgment, over a
    mapping whose every registered CSV and capture is intact; the served event's disclosure lists
    no unverified capture and holds no list of plant CSVs.
    """
    from tcip_mcp.server import binding_for
    from tests._server_calls import called_through_the_server

    pytest.importorskip("torch")
    tool, arguments = call(tmp_path)
    bound = binding_for(tmp_path)

    answer = called_through_the_server(tool, arguments, bound=bound)

    assert isinstance(answer, dict) and "error" not in answer, answer
    (event,) = read_delivery_events(bound.root)
    disclosure = event.model_dump(mode="json")["plant_mapping"]
    assert disclosure["captures_unverified"] == []
    assert "plant_csvs_unverified" not in disclosure


def test_a_per_image_counts_answer_leaves_its_bucket_to_the_event(
    tmp_path: Path,
) -> None:
    """The per-image counts answer names no bucket root or name: the event it wrote holds them.

    The door is called through the server; its answer carries the event's id and the counts, and
    the served event's first bucket is the one the call named.
    """
    from tcip_mcp.server import binding_for
    from tests._server_calls import called_through_the_server

    pytest.importorskip("torch")
    tool, arguments = _per_image_counts_call(tmp_path)
    bound = binding_for(tmp_path)

    answer = called_through_the_server(tool, arguments, bound=bound)

    assert isinstance(answer, dict) and "error" not in answer, answer
    assert {"dataset_root", "bucket"}.isdisjoint(answer), answer
    (event,) = (
        e
        for e in read_delivery_events(bound.root)
        if e.event_id == answer["event_id"]
    )
    (finding,) = event.buckets
    assert finding.bucket == arguments["bucket"]
    assert finding.dataset_root == str(
        Path(arguments["dataset_root"]).resolve()
    )


def test_a_general_door_delivery_states_the_mappings_attribution_once(
    tmp_path: Path,
) -> None:
    """A general-door delivery's rows carry the attribution its event's disclosure states.

    The door is called through the server over the mapping ``build_plant_mapping`` persisted and
    names no attribution; every row's ``plant_attribution`` equals the event's disclosure field
    and the mapper's own constant.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import MappingBuild
    from tcip_mcp.server import binding_for
    from tests import csv_rows
    from tests._server_calls import called_through_the_server

    pytest.importorskip("torch")
    unacknowledged = _per_plant_csv_arguments(tmp_path)
    bound = binding_for(tmp_path)
    refused = called_through_the_server(
        "deliver_per_plant_csv", unacknowledged, bound=bound
    )
    assert isinstance(refused, dict), refused
    assert "no assessment answers" in refused["error"], refused
    tool, arguments = _acknowledged_per_plant_csv(tmp_path, unacknowledged)

    answer = called_through_the_server(tool, arguments, bound=bound)

    assert isinstance(answer, dict) and "error" not in answer, answer
    (event,) = read_delivery_events(bound.root)
    assert event.plant_mapping is not None
    rows = csv_rows(Path(arguments["output_path"]))
    assert rows
    assert (
        {row["plant_attribution"] for row in rows}
        == {event.plant_mapping.plant_attribution}
        == {MappingBuild.plant_attribution}
    )


def test_a_completed_crossing_delivery_writes_the_gates_finding_for_every_bucket(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from tests._chain_fixtures import attributed_series

    series = attributed_series(tmp_path, fractions=(0.0, 1.0))
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_milestones(tmp_path, series.body(), out_csv)

    assert "error" not in res, res
    (record,) = read_delivery_events(tmp_path)
    shipped_under = read_trait("bud_opening", tmp_path).latest_confirmed
    assert shipped_under is not None
    assert (
        record.trait,
        record.trait_revision,
        record.trait_revision_sha256,
    ) == ("bud_opening", shipped_under.number, shipped_under.entry_sha256)
    assert record.delivery_kind == STATE_CROSSING_DATES
    assert record.output_path == str(out_csv)
    assert record.shipped.kind == "validated"
    assert record.producer.model_dump() == series.assessment["producer"]
    assert {b.bucket for b in record.buckets} == set(series.buckets.values())
    assert {b.dataset_root for b in record.buckets} == {str(series.root)}
    for finding in record.buckets:
        assert finding.kind == "validated", finding
        assert finding.assessment_id == series.assessment["assessment_id"]
    assert record.population == ["PLANT_A", "PLANT_B"]
    assert (
        record.plant_mapping is not None
        and record.plant_mapping.name == series.mapping_name
    )


def test_two_deliveries_of_the_same_trait_and_kind_both_enumerate_distinctly(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from tests._chain_fixtures import attributed_series

    body = attributed_series(tmp_path, fractions=(0.0, 1.0)).body()
    first_csv, second_csv = (
        tmp_path / "out" / "first.csv",
        tmp_path / "out" / "second.csv",
    )
    for out_csv in (first_csv, second_csv):
        assert "error" not in deliver_milestones(tmp_path, body, out_csv)

    records = read_delivery_events(tmp_path)
    assert len(records) == 2, records
    assert records[0].event_id != records[1].event_id
    assert {r.output_path for r in records} == {
        str(first_csv),
        str(second_csv),
    }


def test_a_web_route_writes_its_delivery_event_under_the_open_project_only(
    tmp_path: Path, client: TestClient
) -> None:
    """A web delivery lands its event record under the open project and none under a second one."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import attributed_series

    other_project = tmp_path / "other_project"
    other_project.mkdir()
    body = attributed_series(tmp_path, fractions=(0.0, 1.0)).body()

    resp = client.post(
        "/api/results/export_csv",
        json={
            **body,
            "payload": "milestones",
            "user": "tester",
        },
    )

    assert resp.status_code == 200, resp.text
    assert [r.door for r in read_delivery_events(tmp_path)] == [
        "results.export_csv"
    ]
    assert read_delivery_events(other_project) == []


def test_a_look_on_screen_records_no_delivery_event_and_no_audit_line(
    tmp_path: Path, client: TestClient
) -> None:
    """``phenology_measurement`` answers a number with no delivery event and no new log line."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import attributed_series

    series = attributed_series(tmp_path, fractions=(0.0, 1.0))
    scopes = (tmp_path, series.root)
    before = [audit_rows(scope) for scope in scopes]

    resp = client.post(
        "/api/results/phenology_measurement", json=series.body()
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["positive_class_assessed"] is True
    assert read_delivery_events(tmp_path) == []
    assert [audit_rows(scope) for scope in scopes] == before


def test_a_record_that_cannot_be_written_raises_before_any_audit_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delivery event that cannot be written raises, and no audit line names one.

    No event is recorded.
    """
    revision = seed_confirmed_count(tmp_path)
    result = delivery.Result(delivery.DELIVERY_COLUMNS, [], population=[])
    clearance = _cleared(tmp_path, revision, _bucket(tmp_path), result)
    replace = ts.replace

    def _event_write_fails(
        key: Key, *a: JsonValue, **kw: Version | None
    ) -> Version:
        if key.store == delivery.DELIVERY_EVENTS_STORE:
            raise OSError("disk full")
        return replace(key, *a, **kw)

    monkeypatch.setattr(ts, "replace", _event_write_fails)

    with pytest.raises(OSError, match="disk full"):
        delivery.deliver_csv(
            tmp_path,
            tmp_path / "out" / "counts.csv",
            result,
            clearance=clearance,
            revision=revision,
            door="test_door",
            delivery_kind=PER_IMAGE_COUNT,
            actor=None,
        )

    monkeypatch.undo()
    assert read_delivery_events(tmp_path.resolve()) == []
    assert audit_rows(tmp_path, "delivery_event") == []


def _delivered_disclosure(project: Path, deliver: Callable[[], dict]) -> dict:
    """Return the ``plant_mapping`` disclosure the one delivery ``deliver()`` records, as JSON.

    The delivery records under ``project``, its delivered ``stem_count`` confirmed there first as
    an orthomosaic count.
    """
    from tcip_mcp.delivery import read_delivery_events
    from tests import _trait_fixtures as fx

    fx.seed_delivery_traits(project)
    fx.seed_confirmed_orthomosaic_count(project, "stem_count")
    delivered = deliver()
    assert "error" not in delivered, delivered
    (event,) = read_delivery_events(project)
    return event.model_dump(mode="json")["plant_mapping"]


def test_plant_mapping_union_resolves_each_shape_and_refuses_a_hybrid(
    tmp_path: Path,
) -> None:
    """The ``plant_mapping`` union resolves each disclosure shape and refuses a hybrid.

    The disclosures the three producers record each validate, answering three distinct kinds,
    and a hybrid combining keys from two shapes is refused.
    """
    from pydantic import TypeAdapter, ValidationError

    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.pipelines.delivery_events_schema import PlantMapping
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        plant_mapping_disclosure,
    )
    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tests import (
        test_orthomosaic_tools as ortho,
        test_plant_mapping_binding as binding,
    )
    from tests._mapping_fixtures import register_plant_registry_for

    mapped = tmp_path / "mapped"
    binding._init(mapped)
    dataset_root = binding._dataset(mapped)
    plant_csv, preds_by_date = binding._write_scene(dataset_root)
    built = build_plant_mapping(
        mapped,
        name="valley",
        dataset_root=str(dataset_root),
        plant_registry=register_plant_registry_for(mapped, [plant_csv]),
        actor=None,
    )
    assert "error" not in built, built
    buckets = {
        date: read_bucket(dataset_root, name)
        for date, name in preds_by_date.items()
    }
    mapping = plant_mapping_disclosure(
        mapped, "valley", buckets, binding.POPULATION
    ).disclosure

    nearest = tmp_path / "nearest"
    raster_path = ortho._raster(nearest)
    ortho._write_geo_raster(raster_path)
    bucket = ortho._raster_bucket(
        nearest, raster_path, [(8.0, 8.0, 12.0, 12.0)]
    )
    reg = ortho._plant_registry(
        nearest,
        ortho._plant_grid_csv(nearest, raster_path, ortho._PLANT_PIXELS),
    )
    registry = _delivered_disclosure(
        nearest, lambda: ortho._deliver(nearest, bucket, reg, ortho._GRID)
    )

    segmented = tmp_path / "segmented"
    _root, raster_path, bucket = ortho._canopy_setup(
        segmented, [(8.0, 8.0, 12.0, 12.0)]
    )
    reg = ortho._plant_registry(
        segmented,
        ortho._plants_csv_at(segmented, raster_path, [("plot0", 10.0, 10.0)]),
    )
    ortho._write_canopy_document(raster_path, [(5.0, 5.0, 15.0, 15.0)])
    canopy = _delivered_disclosure(
        segmented,
        lambda: ortho._deliver(
            segmented,
            bucket,
            reg,
            ["plot0"],
            canopy_subject="canopy",
            position_error_m=POSITION_ERROR_M,
        ),
    )

    union: TypeAdapter[PlantMapping] = TypeAdapter(PlantMapping)

    def _resolved(pm: object) -> PlantMapping:
        return union.validate_python(pm)

    produced: list[object] = [mapping, registry, canopy]
    assert len({_resolved(pm).kind for pm in produced}) == len(produced)

    hybrid = {**registry, "canopy_segments": canopy["canopy_segments"]}
    with pytest.raises(ValidationError):
        _resolved(hybrid)


# ── the delivery-events route: read-only ─────────────────────────────────────

DELIVERY_EVENTS_ROUTE = "/api/results/delivery-events"


def test_delivery_events_route_lists_a_recorded_event_with_its_revision(
    opened_project: Path,
    client: TestClient,
) -> None:
    revision = seed_confirmed_count(opened_project)
    _record(opened_project, revision)

    resp = client.get(DELIVERY_EVENTS_ROUTE)

    assert resp.status_code == 200
    (served,) = resp.json()["records"]
    record = served["event"]
    assert record["door"] == "test_door"
    assert (
        record["trait"],
        record["trait_revision"],
        record["trait_revision_sha256"],
    ) == (revision.entry.name, revision.number, revision.entry_sha256)
    assert record["delivery_kind"] == PER_IMAGE_COUNT


def test_delivery_events_route_refuses_a_record_its_model_refuses_with_the_models_issues(
    opened_project: Path,
    client: TestClient,
) -> None:
    """A stored event its model refuses refuses the listing (400) with the model's issues.

    The detail names the event, and the issues are served each by its place in the record and
    its message: a record that is a list where an event belongs is one issue, placed at the record
    as a whole.
    """
    from tests._record_damage_fixtures import damage_record

    _record(opened_project, seed_confirmed_count(opened_project))
    (event,) = read_delivery_events(opened_project)
    damage_record(
        delivery.delivery_event_key(opened_project, event.event_id), b"[]"
    )

    resp = client.get(DELIVERY_EVENTS_ROUTE)

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert event.event_id in detail["where"]
    assert detail["refusal"]["kind"] == "invalid"
    assert [issue["loc"] for issue in detail["refusal"]["issues"]] == [[]]


def test_delivery_events_route_refuses_while_no_project_is_open(
    client: TestClient,
) -> None:
    resp = client.get(DELIVERY_EVENTS_ROUTE)

    assert resp.status_code == 409


def test_delivery_events_route_serves_a_delivered_plant_mapping_disclosure_unchanged(
    tmp_path: Path,
    client: TestClient,
) -> None:
    """The delivery-events route serves a phenology delivery's mapping disclosure unchanged.

    The disclosure is the one the delivery records over a built mapping, served beside the key
    its cited record loads under, the mapping's own name while no rebuild has moved past it.
    """
    pytest.importorskip("torch")
    from tests._chain_fixtures import attributed_series

    series = attributed_series(tmp_path, fractions=(0.0, 1.0))
    assert "error" not in deliver_milestones(
        tmp_path, series.body(), tmp_path / "out" / "bud.csv"
    )
    (recorded,) = read_delivery_events(tmp_path)

    resp = client.get(DELIVERY_EVENTS_ROUTE)

    assert resp.status_code == 200
    (record,) = resp.json()["records"]
    assert recorded.plant_mapping is not None
    assert recorded.plant_mapping.kind == "plant_mapping"
    assert record == {
        "event": recorded.model_dump(mode="json"),
        "resolved_key": recorded.plant_mapping.name,
    }
    assert record["event"]["plant_mapping"]["dates_delivered"] == sorted(
        series.buckets
    )
