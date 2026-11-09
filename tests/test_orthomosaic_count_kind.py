"""The orthomosaic count's own delivery kind, from its proposal to the rows its doors write.

Covered: an entry stating only that kind is proposed with no ``plant_aggregation`` and is held to
the count fields; the tool and the export route write its rows with no per-image cell under both
attribution regimes; a revision stating only the per-image aggregate refuses it; an
acknowledgment binds one of its results; a unit-declared phenotype refuses at it; and its shared
columns and provenance agree with the per-image aggregate door's under one revision and one
producer.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_mcp import traits
from tcip_mcp.delivery import DELIVERY_COLUMNS
from tcip_mcp.pipelines.postprocessing.aggregation import PER_IMAGE_COLUMNS
from tests import _trait_fixtures as fx, csv_rows
from tests._mapping_fixtures import POSITION_ERROR_M

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from fastapi.testclient import TestClient

    from tcip_mcp.buckets import Bucket
    from tcip_mcp.workspace import BoundProject

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tests import test_orthomosaic_tools as ortho  # noqa: E402

KIND = "per_plant_orthomosaic_count"
"""The orthomosaic count's delivery kind, as a trait revision and a delivery event store it."""

SHARED_COLUMNS = (
    "plant_id",
    "crop",
    "delivered_phenotype",
    "value",
    "units",
    "value_key",
    "plant_attribution",
)
"""The cells every per-plant delivery writes beside its delivery cells, in their order."""

THREE_DETECTIONS = [
    (8.0, 8.0, 12.0, 12.0),
    (9.0, 9.0, 11.0, 11.0),
    (48.0, 8.0, 52.0, 12.0),
]
"""Two detections at the plant at pixel (10, 10), one at the plant at (50, 10), none elsewhere."""

type Scene = tuple[Bucket, str, list[str], dict]


def _nearest(project: Path) -> Scene:
    """Return the 2x2 plant grid's bucket of :data:`THREE_DETECTIONS`, nearest-plant attributed.

    The return is the bucket, the registry's name, the population and the regime's arguments.
    """
    raster_path = ortho._raster(project)
    ortho._write_geo_raster(raster_path)
    bucket = ortho._raster_bucket(project, raster_path, THREE_DETECTIONS)
    registry = ortho._plant_registry(
        project,
        ortho._plant_grid_csv(project, raster_path, ortho._PLANT_PIXELS),
    )
    return bucket, registry, ortho._GRID, {}


def _segment(project: Path) -> Scene:
    """Return a bucket of :data:`THREE_DETECTIONS` attributed by containment in canopy segments.

    Three accepted segments each hold one plant; the third holds no detection.
    """
    _root, raster_path, bucket = ortho._canopy_setup(project, THREE_DETECTIONS)
    registry = ortho._plant_registry(
        project,
        ortho._plants_csv_at(
            project,
            raster_path,
            [
                ("plot0", 10.0, 10.0),
                ("plot1", 10.0, 50.0),
                ("plot2", 50.0, 10.0),
            ],
        ),
    )
    ortho._write_canopy_document(
        raster_path,
        [
            (5.0, 5.0, 15.0, 15.0),
            (5.0, 45.0, 15.0, 55.0),
            (45.0, 5.0, 55.0, 15.0),
        ],
    )
    return (
        bucket,
        registry,
        ["plot0", "plot1", "plot2"],
        {"canopy_subject": "canopy", "position_error_m": POSITION_ERROR_M},
    )


def _tool(
    project: Path,
    scene: Scene,
    *,
    plants: list[str] | None = None,
    phenotype: str = "stem_count",
    acknowledgment_id: str | None = None,
) -> dict:
    """Return ``deliver_orthomosaic_plant_counts``'s answer over ``scene`` into ``counts.csv``.

    ``plants`` names the population, the scene's own by default.
    """
    from tcip_mcp.tools.orthomosaic_tools import (
        deliver_orthomosaic_plant_counts,
    )

    bucket, registry, population, regime = scene
    return deliver_orthomosaic_plant_counts(
        project,
        str(bucket.root),
        bucket.name,
        registry,
        str(project / "counts.csv"),
        phenotype,
        population if plants is None else plants,
        acknowledgment_id=acknowledgment_id,
        actor=None,
        **regime,
    )


def _acknowledged_tool(
    project: Path, scene: Scene, *, plants: list[str] | None = None
) -> dict:
    """Return :func:`_tool`'s answer under a breeder's acknowledgment of the result it computes.

    The acknowledgment is recorded for the digest the library door's refusal of the same
    delivery names.
    """
    from tcip_mcp.delivery import DeliveryRefusedError, record_acknowledgment
    from tcip_mcp.pipelines.postprocessing.plant_mapping import load_registry
    from tcip_mcp.tools.orthomosaic_tools import orthomosaic_plant_counts
    from tests._chain_fixtures import digest_of

    bucket, registry, population, regime = scene
    with pytest.raises(DeliveryRefusedError) as refused:
        orthomosaic_plant_counts(
            project,
            bucket.root,
            bucket.name,
            load_registry(project, registry),
            project / "counts.csv",
            "stem_count",
            population if plants is None else plants,
            door="test_orthomosaic",
            actor=None,
            **regime,
        )
    act = record_acknowledgment(
        project,
        acknowledged_by="user:breeder",
        reason="a look at the mosaic before its assessment",
        result_sha256=digest_of(refused.value),
    )
    return _tool(
        project, scene, plants=plants, acknowledgment_id=act.acknowledgment_id
    )


def _header(path: Path) -> list[str]:
    with path.open(newline="") as f:
        return next(csv.reader(f))


# ── the proposal ────────────────────────────────────────────────────────


def test_an_entry_stating_only_the_orthomosaic_count_is_proposed_with_no_plant_aggregation(
    bound: BoundProject,
) -> None:
    """An entry stating only the orthomosaic count is admitted with ``plant_aggregation`` empty.

    The same entry stating the per-image aggregate instead is refused asking the breeder the
    statistic's question, and appends nothing; both go through ``propose_trait``'s server door.
    """
    admitted = _proposed(bound, KIND)
    refused = _proposed(
        bound, traits.PER_PLANT_COUNT_AGGREGATE, value_keys=["count"]
    )

    assert isinstance(admitted, dict) and "error" not in admitted, admitted
    assert (
        admitted["number"],
        admitted["entry"]["plant_aggregation"],
        list(admitted["entry"]["operationalizations"]),
    ) == (1, "", [KIND])
    question = traits.QUESTIONS["plant_aggregation"]
    assert f"plant_aggregation: {question}" in str(refused), refused
    assert len(traits.read_trait(fx.COUNT_TRAIT, bound.root).revisions) == 1


def test_an_orthomosaic_count_leaving_a_count_field_unauthored_is_refused_asking_for_it(
    bound: BoundProject,
) -> None:
    """An entry stating only the orthomosaic count rests on the count fields.

    With ``count_bias_tolerance_frac`` unauthored it is refused through ``propose_trait``'s server
    door, asking the breeder that field's question, and nothing is appended.
    """
    from tcip_mcp.traits import trait_names

    refused = _proposed(bound, KIND, count_bias_tolerance_frac=None)

    question = traits.QUESTIONS["count_bias_tolerance_frac"]
    assert f"count_bias_tolerance_frac: {question}" in str(refused), refused
    assert fx.COUNT_TRAIT not in trait_names(bound.root)


def test_an_orthomosaic_count_names_no_value_keys(bound: BoundProject) -> None:
    """An entry stating only the orthomosaic count states no value keys; its door fixes ``count``.

    Through ``propose_trait``'s server door, the entry stating the set ``count`` is refused naming
    ``delivered_value_keys``, and the same entry stating none is admitted as the one revision.
    """
    refused = _proposed(bound, KIND, value_keys=["count"])
    admitted = _proposed(bound, KIND)

    assert "delivered_value_keys must be empty" in str(refused), refused
    assert isinstance(admitted, dict) and "error" not in admitted, admitted
    assert (
        admitted["entry"]["operationalizations"][KIND]["delivered_value_keys"]
        == []
    )
    assert len(traits.read_trait(fx.COUNT_TRAIT, bound.root).revisions) == 1


def _proposed(
    bound: BoundProject,
    kind: str,
    *,
    value_keys: Sequence[str] = (),
    **fields: object,
) -> object:
    """Return ``propose_trait``'s server answer for the count trait stating ``kind`` alone.

    The entry is the fixture count trait with ``plant_aggregation`` unauthored and ``fields``
    over it, its one operationalization covering ``stem_count`` and the value keys ``value_keys``.
    """
    from tests._server_calls import called_through_the_server

    entry = fx.with_fields(
        fx.COUNT_SPEC, plant_aggregation="", **fields
    ).model_dump(mode="json")
    stated = fx.operationalization(
        delivered_phenotypes=["stem_count"],
        delivered_value_keys=list(value_keys),
    )
    return called_through_the_server(
        "propose_trait",
        {
            "entry": {**entry, "operationalizations": {kind: stated}},
            "rationale": "the breeder counts each plant on the mosaic",
        },
        bound=bound,
    )


# ── the delivered rows ──────────────────────────────────────────────────


@pytest.mark.usefixtures("confirmed_orthomosaic_count")
@pytest.mark.parametrize(
    ("scene", "counts", "attribution"),
    [
        (
            _nearest,
            {"plot0": "2", "plot1": "0", "plot2": "1", "plot3": "0"},
            "detection",
        ),
        (_segment, {"plot0": "2", "plot1": "0", "plot2": "1"}, "segment"),
    ],
    ids=["nearest", "segment"],
)
def test_the_orthomosaic_door_writes_each_plants_count_and_no_per_image_cell(
    tmp_path: Path,
    scene: Callable[[Path], Scene],
    counts: dict[str, str],
    attribution: str,
) -> None:
    """The tool's CSV carries the shared cells alone, each plant's count, under its own kind.

    The revision states the orthomosaic count alone with ``plant_aggregation`` unauthored; a
    population plant no detection reaches carries 0, every row the regime's attribution, and the
    delivery event the orthomosaic count's kind.
    """
    from tcip_mcp.delivery import read_delivery_events

    delivered = _acknowledged_tool(tmp_path, scene(tmp_path))

    assert "error" not in delivered, delivered
    assert _header(tmp_path / "counts.csv") == [
        *SHARED_COLUMNS,
        *DELIVERY_COLUMNS,
    ]
    rows = csv_rows(tmp_path / "counts.csv")
    assert {r["plant_id"]: r["value"] for r in rows} == counts
    assert {(r["value_key"], r["plant_attribution"]) for r in rows} == {
        ("count", attribution)
    }
    (event,) = read_delivery_events(tmp_path)
    assert event.delivery_kind == KIND


@pytest.mark.usefixtures("confirmed_orthomosaic_count")
def test_the_export_route_writes_the_header_the_tool_writes(
    client: TestClient, tmp_path: Path
) -> None:
    """The count export route's orthomosaic CSV has the tool's header, each under acknowledgment."""
    from tests._web_fixtures import acknowledged_post, open_new_project
    from tests.test_tcip_web_results_routes import COUNT_ROUTE, _per_plant

    scene = _nearest(tmp_path)
    tool = _acknowledged_tool(tmp_path, scene)
    open_new_project(tmp_path)
    route = acknowledged_post(
        client,
        COUNT_ROUTE,
        _per_plant(scene[0], scene[1]),
        reason="a look at the mosaic before its assessment",
    )

    assert "error" not in tool, tool
    assert route.status_code == 200, route.text[:300]
    assert _header(tmp_path / "results_export" / "plant_counts.csv") == (
        _header(tmp_path / "counts.csv")
    )


@pytest.mark.usefixtures("confirmed_count_aggregate")
def test_a_revision_stating_only_the_per_image_aggregate_refuses_the_orthomosaic_door(
    tmp_path: Path,
) -> None:
    """A revision whose one operationalization is the per-image aggregate refuses the mosaic count.

    The refusal names the orthomosaic count's kind as unstated, and no CSV is written.
    """
    refused = _tool(tmp_path, _nearest(tmp_path))

    assert (
        f"states no operationalization for a {KIND} delivery"
        in refused.get("error", "")
    ), refused
    assert not (tmp_path / "counts.csv").exists()


def test_a_unit_declared_phenotype_counted_on_a_mosaic_refuses_for_its_unit(
    tmp_path: Path,
) -> None:
    """A mosaic count of a phenotype crops.yml declares in ``mm`` refuses, naming the unit.

    The phenotype's revision states the orthomosaic count of it; the count's value key implies
    no physical unit, so nothing ships.
    """
    assert traits.crops_units()["bark_thickness"] == "mm"
    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_orthomosaic_count(tmp_path, "bark_thickness")

    refused = _tool(tmp_path, _nearest(tmp_path), phenotype="bark_thickness")

    error = refused.get("error", "")
    assert "units='mm'" in error and "implies no physical unit" in error, (
        refused
    )
    assert not (tmp_path / "counts.csv").exists()


# ── the acknowledgment ──────────────────────────────────────────────────


@pytest.mark.usefixtures("confirmed_orthomosaic_count")
def test_an_acknowledgment_of_one_mosaic_count_refuses_another_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """An acknowledgment of one orthomosaic result refuses a changed one, which a fresh one ships.

    The changed result is a smaller population; its refusal leaves the CSV the first delivery
    wrote and its one event as they were.
    """
    from tcip_mcp.delivery import read_delivery_events

    scene = _nearest(tmp_path)
    first = _acknowledged_tool(tmp_path, scene)
    assert "error" not in first, first
    acknowledgment = first["shipped"]["acknowledgment"]
    written = (tmp_path / "counts.csv").read_bytes()

    refused = _tool(
        tmp_path,
        scene,
        plants=ortho._GRID[:2],
        acknowledgment_id=acknowledgment["acknowledgment_id"],
    )

    assert "was given for another result" in refused.get("error", ""), refused
    assert (tmp_path / "counts.csv").read_bytes() == written
    assert len(read_delivery_events(tmp_path)) == 1
    fresh = _acknowledged_tool(tmp_path, scene, plants=ortho._GRID[:2])
    assert fresh["shipped"]["kind"] == "acknowledged", fresh
    assert [r["plant_id"] for r in csv_rows(tmp_path / "counts.csv")] == (
        ortho._GRID[:2]
    )


# ── agreement with the per-image aggregate door ─────────────────────────


def test_the_two_per_plant_doors_write_their_shared_columns_and_provenance_alike(
    tmp_path: Path,
) -> None:
    """Both per-plant doors in one project, under one revision and one producer, agree.

    The revision states both count kinds. ``deliver_per_plant_csv``, called through the server,
    delivers a mapped capture's records over a bucket ``build_plant_mapping`` mapped; the
    orthomosaic tool delivers a raster bucket of another dataset; the fixture checkpoint
    predicted both. The orthomosaic header is the per-image header without that door's own
    columns, and the two events name one producer and one revision, each read from its own event.
    """
    from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
    from tcip_mcp.delivery import read_delivery_events
    from tcip_mcp.server import binding_for
    from tests import (
        test_deliver_per_plant_csv as per_plant,
        test_delivery_events as delivery_events,
    )
    from tests._server_calls import called_through_the_server

    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_aggregate(tmp_path, "stem_count", value_keys=["count"])
    fx.propose_and_confirm(
        tmp_path,
        fx.with_operationalization(
            fx.count_revision(tmp_path).entry,
            traits.PER_PLANT_ORTHOMOSAIC_COUNT,
            delivered_phenotypes=("stem_count",),
        ),
    )
    dataset_root, date = per_plant._mapped(tmp_path)
    tool, arguments = delivery_events._acknowledged_per_plant_csv(
        tmp_path,
        {
            "results": [{"date": date, "stem": "P1", "value": 2}],
            "output_path": str(tmp_path / "per_image.csv"),
            "delivered_phenotype": "stem_count",
            "delivery_kind": traits.PER_PLANT_COUNT_AGGREGATE,
            "plants": ["P1"],
            "dataset_root": str(dataset_root),
            "buckets": [
                per_plant._bucket(tmp_path, dataset_root, date, {"P1": 2})
            ],
            "value_key": "count",
            "plant_mapping": "valley",
        },
    )
    answered = called_through_the_server(
        tool, arguments, bound=binding_for(tmp_path)
    )
    assert isinstance(answered, dict) and "error" not in answered, answered
    raster_path = (
        image_dir(tmp_path / "mosaic_ds", UNDATED_BUCKET) / "mosaic.tif"
    )
    ortho._write_geo_raster(raster_path)
    mosaic_scene = (
        ortho._raster_bucket(tmp_path, raster_path, THREE_DETECTIONS),
        ortho._plant_registry(
            tmp_path,
            ortho._plant_grid_csv(tmp_path, raster_path, ortho._PLANT_PIXELS),
            name="mosaic-grid",
        ),
        ortho._GRID,
        {},
    )
    delivered = _acknowledged_tool(tmp_path, mosaic_scene)
    assert "error" not in delivered, delivered

    events = {e.door: e for e in read_delivery_events(tmp_path)}
    mosaic = events["deliver_orthomosaic_plant_counts"]
    per_image = events["deliver_per_plant_csv"]
    mosaic_header = _header(tmp_path / "counts.csv")
    per_image_header = _header(tmp_path / "per_image.csv")
    assert mosaic_header == [
        c for c in per_image_header if c not in PER_IMAGE_COLUMNS
    ]
    assert set(per_image_header) - set(mosaic_header) == set(PER_IMAGE_COLUMNS)
    provenance = (
        "producer",
        "trait",
        "trait_revision",
        "trait_revision_sha256",
    )
    assert [getattr(mosaic, f) for f in provenance] == [
        getattr(per_image, f) for f in provenance
    ]
    assert (mosaic.delivery_kind, per_image.delivery_kind) == (
        KIND,
        traits.PER_PLANT_COUNT_AGGREGATE,
    )
