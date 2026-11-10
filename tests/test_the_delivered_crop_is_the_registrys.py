"""A delivered row's crop is its plant registry's, and every crop a delivery rests on agrees.

Through the doors an agent or a breeder calls: the build's tool and route, the per-plant tool, the
phenology tool and its two routes, and the orthomosaic tool and its route. A dataset and a plant
registry of different crops refuse at the build and at every mapped delivery, naming both; an
orthomosaic count is tied to the dataset its bucket recorded; a trait revision delivering a
phenotype crops.yml does not name for the registry's crop refuses, naming both; and every
delivered row's ``crop`` cell is the registry's, with no door taking one. The delivered CSV also
carries no ``pipeline_version`` column, what produced it being the producer its delivery event
records, and no door takes one.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests import _trait_fixtures as fx, csv_rows
from tests._mapping_fixtures import (
    COUNT_CROP,
    CROP,
    register_plant_registry_for,
    registered_copy,
)
from tests._producer_fixtures import register_dataset_at

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from tcip_mcp.buckets import Bucket

pytest.importorskip("torch")
pytest.importorskip("torchvision")

OFF_CROP_PHENOTYPE = "astringency"
"""A phenotype crops.yml names for persimmon alone, never for :data:`COUNT_CROP` or :data:`CROP`."""


def _crops_named(error: str, *crops: str) -> None:
    """Assert ``error`` names each of ``crops`` as a crop."""
    for crop in crops:
        assert f"crop {crop!r}" in error, error


# ── the build ──────────────────────────────────────────────────────────


def test_a_build_over_a_dataset_and_a_registry_of_different_crops_refuses_at_both_doors(
    client: TestClient, tmp_path: Path
) -> None:
    """The build's tool and route refuse a registry of another crop than the dataset's, naming both.

    The dataset is registered at :data:`CROP` and the plants under a second registry at
    :data:`COUNT_CROP`; no mapping is stored.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import load_mapping
    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tests.test_results_mapping_summary_and_audit_anchoring import (
        _capture_fixture,
    )

    payload = _capture_fixture(tmp_path)
    other = register_plant_registry_for(
        tmp_path, [payload["csv_path"]], name="other-plots", crop=COUNT_CROP
    )

    tool = build_plant_mapping(
        tmp_path,
        name=payload["name"],
        dataset_root=payload["dataset_root"],
        plant_registry=other,
        actor=None,
    )
    route = client.post(
        "/api/results/plant_mapping/build",
        json={**payload, "plant_registry": other},
    )

    assert "error" in tool, tool
    _crops_named(tool["error"], CROP, COUNT_CROP)
    assert other in tool["error"]
    assert route.status_code == 400, route.text
    _crops_named(route.json()["detail"], CROP, COUNT_CROP)
    assert load_mapping(tmp_path, payload["name"]) is None


# ── a dataset registered again with another crop after the build ────────


@pytest.mark.usefixtures("confirmed_count_aggregate")
def test_a_per_plant_delivery_over_a_dataset_registered_again_with_another_crop_refuses(
    tmp_path: Path,
) -> None:
    """A per-plant delivery whose dataset now states another crop than its registry refuses.

    The mapping was built with the dataset and the registry both at :data:`COUNT_CROP`; the
    dataset is then registered again at :data:`CROP`, which keeps its id, so the id the mapping
    recorded still matches. The refusal names both crops, and no CSV is written.
    """
    from tests.test_deliver_per_plant_csv import (
        _bucket,
        _deliver,
        _mapped,
        _results,
    )

    dataset_root, date = _mapped(tmp_path)
    bucket = _bucket(tmp_path, dataset_root, date, {"P1": 1})
    register_dataset_at(tmp_path, dataset_root, crop=CROP)

    refused = _deliver(
        tmp_path,
        _results(date, ("P1", 1)),
        ["P1"],
        dataset_root,
        [bucket],
        plant_mapping="valley",
    )

    assert "error" in refused, refused
    _crops_named(refused["error"], CROP, COUNT_CROP)
    assert not (tmp_path / "o.csv").exists()


def _phenology_refusals(
    client: TestClient, project: Path, body: dict
) -> list[str]:
    """Return what the phenology tool and its two routes answer for ``body``, each refused.

    The routes are ``/phenology_measurement`` and ``/export_csv``; each must answer 400.
    """
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    tool = deliver_phenology_milestones(
        project,
        trait=body["trait"],
        mapping_name=body["mapping_name"],
        dataset_root=body["dataset_root"],
        buckets=body["buckets"],
        output_csv_path=str(project / "milestones.csv"),
        plants=body["plants"],
        actor=None,
    )
    assert "error" in tool, tool
    answers = [
        client.post("/api/results/phenology_measurement", json=body),
        client.post(
            "/api/results/export_csv",
            json={**body, "payload": "milestones", "user": "breeder"},
        ),
    ]
    for answer in answers:
        assert answer.status_code == 400, answer.text
    assert not (project / "milestones.csv").exists()
    return [tool["error"], *(str(a.json()["detail"]) for a in answers)]


def test_a_phenology_delivery_over_a_dataset_registered_again_with_another_crop_refuses(
    client: TestClient, tmp_path: Path
) -> None:
    """The phenology tool and both routes refuse a dataset now of another crop than its registry.

    The mapping was built with both at :data:`CROP`; the dataset is registered again at
    :data:`COUNT_CROP`. Each door's refusal names both crops.
    """
    from tests.test_second_trait_acceptance import _currant_bloom_body

    body = _currant_bloom_body(tmp_path)
    register_dataset_at(tmp_path, Path(body["dataset_root"]), crop=COUNT_CROP)

    for refusal in _phenology_refusals(client, tmp_path, body):
        _crops_named(refusal, CROP, COUNT_CROP)


# ── the orthomosaic count's dataset ─────────────────────────────────────


def _orthomosaic_scene(project: Path) -> Bucket:
    """Return a whole-raster bucket over the project's dataset ``ds``, the count trait confirmed.

    The raster's dataset is registered at :data:`COUNT_CROP`.
    """
    from tests.test_orthomosaic_tools import (
        _raster,
        _raster_bucket,
        _write_geo_raster,
    )

    fx.seed_delivery_traits(project)
    fx.seed_confirmed_orthomosaic_count(project, "stem_count")
    raster_path = _raster(project)
    _write_geo_raster(raster_path)
    return _raster_bucket(project, raster_path, [(8.0, 8.0, 12.0, 12.0)])


def _grid_registry(project: Path, *, crop: str = COUNT_CROP) -> str:
    """Register the raster's 2x2 plant grid at ``crop`` and return the registry's name."""
    from tests.test_orthomosaic_tools import (
        _PLANT_PIXELS,
        _plant_grid_csv,
        _raster,
    )

    return register_plant_registry_for(
        project,
        [_plant_grid_csv(project, _raster(project), _PLANT_PIXELS)],
        name=f"grid-{crop.replace('_', '-')}",
        crop=crop,
    )


def _orthomosaic_refusals(
    client: TestClient,
    project: Path,
    dataset_root: Path,
    bucket: str,
    registry: str,
) -> list[str]:
    """Return what the orthomosaic tool and its route answer, each refused, and assert no CSV.

    The route answers 400 with its delivery refusal's message.
    """
    from tcip_mcp.tools.orthomosaic_tools import (
        deliver_orthomosaic_plant_counts,
    )
    from tests._web_fixtures import open_new_project
    from tests.test_orthomosaic_tools import _GRID
    from tests.test_tcip_web_results_routes import COUNT_ROUTE

    tool = deliver_orthomosaic_plant_counts(
        project,
        str(dataset_root),
        bucket,
        registry,
        str(project / "counts.csv"),
        "stem_count",
        _GRID,
        actor=None,
    )
    assert "error" in tool, tool
    open_new_project(project)
    route = client.post(
        COUNT_ROUTE,
        json={
            "delivery": {
                "kind": "orthomosaic_plant_counts",
                "dataset_root": str(dataset_root),
                "bucket": bucket,
                "plant_registry": registry,
                "delivered_phenotype": "stem_count",
                "plants": _GRID,
            },
            "filename": "counts.csv",
            "user": "breeder",
        },
    )
    assert route.status_code == 400, route.text
    assert not (project / "counts.csv").exists()
    assert not (project / "results_export" / "counts.csv").exists()
    return [tool["error"], route.json()["detail"]["message"]]


def test_an_orthomosaic_count_over_a_registry_of_another_crop_than_its_dataset_refuses(
    client: TestClient, tmp_path: Path
) -> None:
    """The orthomosaic tool and route refuse a registry of another crop than the bucket's dataset.

    The bucket's dataset is at :data:`COUNT_CROP`, the registry at :data:`CROP`; each refusal
    names both.
    """
    bucket = _orthomosaic_scene(tmp_path)
    registry = _grid_registry(tmp_path, crop=CROP)

    for refusal in _orthomosaic_refusals(
        client, tmp_path, bucket.root, bucket.name, registry
    ):
        _crops_named(refusal, CROP, COUNT_CROP)


def test_an_orthomosaic_count_over_a_bucket_recording_another_dataset_than_its_root_refuses(
    client: TestClient, tmp_path: Path
) -> None:
    """A bucket under a root whose identity is another dataset's refuses, naming both ids.

    The dataset ``ds`` is copied with its store but without its identity document and the copy
    registered, which mints it its own id; the copy's bucket still records the original's.
    """
    bucket = _orthomosaic_scene(tmp_path)
    copy, minted = registered_copy(
        tmp_path, bucket.root, "ds-copy", crop=COUNT_CROP
    )
    assert minted != bucket.dataset_id

    for refusal in _orthomosaic_refusals(
        client, tmp_path, copy, bucket.name, _grid_registry(tmp_path)
    ):
        assert bucket.dataset_id in refusal, refusal
        assert minted in refusal, refusal


# ── a revision delivering a phenotype the registry's crop does not carry ──


def _count_trait_delivering_off_crop(project: Path) -> None:
    """Confirm the count trait delivering ``stem_count`` and :data:`OFF_CROP_PHENOTYPE`.

    The revision states no operationalization; the caller confirms the one its door ships under,
    covering ``stem_count`` alone.
    """
    fx.propose_and_confirm(
        project,
        fx.with_fields(
            fx.COUNT_SPEC, delivers=("stem_count", OFF_CROP_PHENOTYPE)
        ),
    )


def _names_the_off_crop_phenotype(error: str, crop: str) -> None:
    """Assert ``error`` names :data:`OFF_CROP_PHENOTYPE` alone of the phenotypes, and ``crop``."""
    assert f"[{OFF_CROP_PHENOTYPE!r}]" in error, error
    _crops_named(error, crop)


def test_a_per_plant_delivery_of_a_revision_with_a_phenotype_off_the_registrys_crop_refuses(
    tmp_path: Path,
) -> None:
    """The per-plant tool refuses a revision one of whose phenotypes is not of the registry's crop.

    The dataset and the registry are at :data:`COUNT_CROP`, which crops.yml names for
    ``stem_count`` and not for :data:`OFF_CROP_PHENOTYPE`.
    """
    from tests.test_deliver_per_plant_csv import (
        _bucket,
        _deliver,
        _mapped,
        _results,
    )

    _count_trait_delivering_off_crop(tmp_path)
    fx.seed_confirmed_aggregate(tmp_path, "stem_count", value_keys=["count"])
    dataset_root, date = _mapped(tmp_path)

    refused = _deliver(
        tmp_path,
        _results(date, ("P1", 1)),
        ["P1"],
        dataset_root,
        [_bucket(tmp_path, dataset_root, date, {"P1": 1})],
        plant_mapping="valley",
    )

    assert "error" in refused, refused
    _names_the_off_crop_phenotype(refused["error"], COUNT_CROP)
    assert not (tmp_path / "o.csv").exists()


def test_a_phenology_delivery_of_a_revision_with_a_phenotype_off_the_registrys_crop_refuses(
    client: TestClient, tmp_path: Path
) -> None:
    """The phenology tool and both routes refuse a crossing trait delivering a phenotype off-crop.

    The trait delivers two bloom dates crops.yml names for :data:`CROP` and
    :data:`OFF_CROP_PHENOTYPE`, which it does not; its crossing operationalization covers the
    two dates.
    """
    import asyncio

    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tcip_web.state import store
    from tests._mapping_fixtures import PLANTS
    from tests.test_plant_mapping_binding import _dataset, _init, _write_scene
    from tests.test_second_trait_acceptance import BLOOM_STATE

    project = _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    plant_csv, preds_by_date = _write_scene(dataset_root)
    built = build_plant_mapping(
        tmp_path,
        name="valley",
        dataset_root=str(dataset_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
        actor=None,
    )
    assert "error" not in built, built
    dates = ("bloom_05per_date", "bloom_50per_date")
    fx.propose(
        tmp_path,
        fx.entry(
            "bloom_and_more",
            (*dates, OFF_CROP_PHENOTYPE),
            positive_state={"attribute": BLOOM_STATE.name, "value": "open"},
            milestone_fractions=(0.05, 0.50),
            milestone_on="positive_fraction",
            phenology_prefix="bloom",
        ),
    )
    fx.seed_confirmed_crossing(
        tmp_path,
        "bloom_and_more",
        measured_subject="flower",
        delivered_phenotypes=dates,
    )
    asyncio.run(store.open_project(project))
    body = {
        "mapping_name": "valley",
        "dataset_root": str(dataset_root),
        "buckets": list(preds_by_date.values()),
        "trait": "bloom_and_more",
        "plants": [p["plot"] for p in PLANTS],
    }

    for refusal in _phenology_refusals(client, tmp_path, body):
        _names_the_off_crop_phenotype(refusal, CROP)


def test_an_orthomosaic_delivery_of_a_revision_with_a_phenotype_off_the_registrys_crop_refuses(
    client: TestClient, tmp_path: Path
) -> None:
    """The orthomosaic tool and route refuse a revision one of whose phenotypes is off-crop.

    The bucket's dataset and the registry are at :data:`COUNT_CROP`; the count trait also
    delivers :data:`OFF_CROP_PHENOTYPE`.
    """
    from tests.test_orthomosaic_tools import (
        _raster,
        _raster_bucket,
        _write_geo_raster,
    )

    _count_trait_delivering_off_crop(tmp_path)
    fx.seed_confirmed_orthomosaic_count(tmp_path, "stem_count")
    raster_path = _raster(tmp_path)
    _write_geo_raster(raster_path)
    bucket = _raster_bucket(tmp_path, raster_path, [(8.0, 8.0, 12.0, 12.0)])

    for refusal in _orthomosaic_refusals(
        client, tmp_path, bucket.root, bucket.name, _grid_registry(tmp_path)
    ):
        _names_the_off_crop_phenotype(refusal, COUNT_CROP)


# ── the delivered crop cell ─────────────────────────────────────────────


def test_a_per_plant_delivery_derives_its_crop_and_writes_no_pipeline_version(
    tmp_path: Path,
) -> None:
    """A validated per-plant delivery through the tool writes the registry's crop in every row.

    The bucket is published under a per-plant count assessment and mapped by ``valley`` at
    :data:`COUNT_CROP`, read back from the written CSV, whose header has no ``pipeline_version``.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import load_registry
    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv
    from tcip_mcp.traits import PER_PLANT_COUNT_AGGREGATE
    from tests._chain_fixtures import mapped_results
    from tests.test_aggregation_multi_plant_delivery import (
        _PLANTS,
        _VALUES,
        _assessed,
        _resolved,
    )

    fx.seed_delivery_traits(tmp_path)
    bucket = _assessed(tmp_path, PER_PLANT_COUNT_AGGREGATE, "exp-crop-cell")
    out = tmp_path / "per_plant.csv"

    delivered = deliver_per_plant_csv(
        tmp_path,
        mapped_results(_resolved(tmp_path, bucket), _VALUES),
        str(out),
        "stem_count",
        PER_PLANT_COUNT_AGGREGATE,
        _PLANTS,
        str(bucket.root),
        [bucket.name],
        value_key="count",
        plant_mapping="valley",
        actor=None,
    )

    assert "error" not in delivered, delivered
    registry = load_registry(tmp_path, "valley-registry")
    assert registry["crop"] == COUNT_CROP
    assert {row["crop"] for row in csv_rows(out)} == {registry["crop"]}
    assert "pipeline_version" not in csv_rows(out)[0]


def test_an_orthomosaic_delivery_derives_its_crop_and_writes_no_pipeline_version_at_both_doors(
    client: TestClient, tmp_path: Path
) -> None:
    """The orthomosaic tool and route each write the registry's crop and no ``pipeline_version``.

    Each ships under an acknowledgment of the digest its own refusal named, the route's through
    a breeder's acknowledgment from the page, read back from the CSV each wrote.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import load_registry
    from tests._web_fixtures import acknowledged_post, open_new_project
    from tests.test_orthomosaic_tools import _GRID, _deliver
    from tests.test_tcip_web_results_routes import COUNT_ROUTE, _per_plant

    bucket = _orthomosaic_scene(tmp_path)
    registry = _grid_registry(tmp_path)
    crop = load_registry(tmp_path, registry)["crop"]

    tool = _deliver(tmp_path, bucket, registry, _GRID)
    open_new_project(tmp_path)
    route = acknowledged_post(
        client,
        COUNT_ROUTE,
        _per_plant(bucket, registry),
        reason="a look before assessment",
    )

    assert "error" not in tool, tool
    written = csv_rows(tmp_path / "counts.csv")
    assert {row["crop"] for row in written} == {crop}
    assert "pipeline_version" not in written[0]
    assert route.status_code == 200, route.text[:300]
    served = csv_rows(tmp_path / "results_export" / "plant_counts.csv")
    assert {row["crop"] for row in served} == {crop}
    assert "pipeline_version" not in served[0]


# ── no door takes a derived column ──────────────────────────────────────


@pytest.mark.parametrize(
    ("key", "value"), [("crop", CROP), ("pipeline_version", "v1")]
)
def test_neither_delivering_tool_nor_the_count_payload_takes_the_key(
    client: TestClient, opened_project: Path, key: str, value: str
) -> None:
    """The two per-plant tools list no ``key`` parameter, and the count payload refuses one.

    ``key`` is a column the delivered CSV derives or drops: ``crop`` (the registry's) or
    ``pipeline_version`` (the delivery event's producer). The listing is the MCP server's own;
    the payload's refusal is its schema's, 422.
    """
    import anyio

    from tcip_mcp.server import Session, build_server
    from tests.test_tcip_web_results_routes import COUNT_ROUTE

    listed = {
        t.name: t.input_schema["properties"]
        for t in anyio.run(build_server(Session()).list_tools)
    }
    refused = client.post(
        COUNT_ROUTE,
        json={
            "delivery": {
                "kind": "orthomosaic_plant_counts",
                "dataset_root": str(opened_project),
                "bucket": "preds/2026-01-01",
                "plant_registry": "reg",
                "delivered_phenotype": "stem_count",
                "plants": ["plot0"],
                key: value,
            },
            "filename": "counts.csv",
            "user": "breeder",
        },
    )

    for name in ("deliver_per_plant_csv", "deliver_orthomosaic_plant_counts"):
        assert {"plant_registry", "plant_mapping"} & set(listed[name])
        assert key not in listed[name], name
    assert refused.status_code == 422, refused.text
    assert key in refused.text
