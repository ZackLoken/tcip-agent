"""A registry's plants are read from the bytes it registered, by both doors that read them.

``build_plant_mapping`` and ``deliver_orthomosaic_plant_counts`` refuse a registry whose CSV is
missing or rewritten, each naming every such file in one sentence.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest

import tcip_store
from tcip_mcp.pipelines.postprocessing import plant_mapping
from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts
from tcip_mcp.tools.phenology_tools import build_plant_mapping
from tests._mapping_fixtures import (
    COUNT_CROP,
    geolocated_captures,
    register_plant_registry_for,
    write_plant_csv,
)
from tests.test_orthomosaic_tools import (
    _PLANT_PIXELS,
    _pixel_to_wgs84,
    _raster,
    _raster_bucket,
    _write_geo_raster,
)

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from tcip_mcp.buckets import Bucket

pytestmark = pytest.mark.usefixtures("confirmed_orthomosaic_count")

DATE = "2026-02-11"
PLANTS = ["plot0", "plot1", "plot2"]

CHANGES = [
    pytest.param([], [0], id="removed"),
    pytest.param([0], [], id="rewritten"),
    pytest.param([1, 2], [0], id="removed-and-two-rewritten"),
]
"""Each change as the indices of the registered CSVs it rewrites, then the ones it removes."""


class Scene(NamedTuple):
    """A project holding one registry, a dataset it maps and a raster bucket it counts."""

    project: Path
    photos: Path
    bucket: Bucket
    csvs: list[Path]
    registry: str


def _scene(project: Path) -> Scene:
    """Register three one-plant CSVs and return the project built around them.

    The plants stand on the raster's own grid; a dataset of geolocated photographs maps them and a
    whole-raster bucket counts them.
    """
    from tests._producer_fixtures import register_dataset_at

    raster_path = _raster(project)
    _write_geo_raster(raster_path)
    bucket = _raster_bucket(project, raster_path, [(8.0, 8.0, 12.0, 12.0)])
    plants = []
    for plot, (px, py) in zip(PLANTS, _PLANT_PIXELS, strict=False):
        lat, lon = _pixel_to_wgs84(raster_path, px, py)
        plants.append(
            {"plot": plot, "accession": f"acc-{plot}", "lat": lat, "lon": lon}
        )
    csvs = [
        write_plant_csv(project / f"registered_{i}.csv", [plant])
        for i, plant in enumerate(plants)
    ]
    photos = project / "photos"
    photos.mkdir()
    for root in (photos, bucket.root):
        register_dataset_at(project, root, crop=COUNT_CROP)
    geolocated_captures(photos, [DATE], plants)
    return Scene(
        project,
        photos,
        bucket,
        csvs,
        register_plant_registry_for(
            project, csvs, name="reg", crop=COUNT_CROP
        ),
    )


def _change(
    scene: Scene, rewritten: list[int], removed: list[int]
) -> list[Path]:
    """Rewrite the CSVs at ``rewritten`` and remove those at ``removed``; return them all."""
    for i in rewritten:
        scene.csvs[i].write_bytes(scene.csvs[i].read_bytes() + b"\n")
    for i in removed:
        scene.csvs[i].unlink()
    return [scene.csvs[i] for i in (*rewritten, *removed)]


def _build(scene: Scene, name: str = "valley") -> dict:
    return build_plant_mapping(
        scene.project,
        name=name,
        dataset_root=str(scene.photos),
        plant_registry=scene.registry,
        actor=None,
    )


def _count(scene: Scene) -> dict:
    return deliver_orthomosaic_plant_counts(
        scene.project,
        str(scene.bucket.root),
        scene.bucket.name,
        scene.registry,
        str(scene.project / "counts.csv"),
        "stem_count",
        PLANTS,
        actor=None,
    )


def _names_exactly(error: str, scene: Scene, changed: list[Path]) -> None:
    """Assert ``error`` names every CSV in ``changed`` and none of the other registered ones."""
    for csv_path in scene.csvs:
        assert (csv_path.name in error) == (csv_path in changed), error


@pytest.mark.parametrize(("rewritten", "removed"), CHANGES)
def test_a_build_over_a_changed_registry_refuses_naming_each_changed_csv(
    opened_project: Path, rewritten: list[int], removed: list[int]
) -> None:
    scene = _scene(opened_project)
    changed = _change(scene, rewritten, removed)

    refused = _build(scene)

    _names_exactly(refused["error"], scene, changed)
    assert not tcip_store.exists(
        plant_mapping.plant_mapping_key(scene.project, "valley")
    )


@pytest.mark.parametrize(("rewritten", "removed"), CHANGES)
def test_an_orthomosaic_count_over_a_changed_registry_refuses_naming_each_changed_csv(
    opened_project: Path, rewritten: list[int], removed: list[int]
) -> None:
    scene = _scene(opened_project)
    changed = _change(scene, rewritten, removed)

    refused = _count(scene)

    _names_exactly(refused["error"], scene, changed)
    assert not (scene.project / "counts.csv").exists()


@pytest.mark.parametrize(("rewritten", "removed"), CHANGES)
def test_the_build_and_the_orthomosaic_count_refuse_a_changed_registry_in_one_sentence(
    opened_project: Path, rewritten: list[int], removed: list[int]
) -> None:
    scene = _scene(opened_project)
    changed = _change(scene, rewritten, removed)

    built, counted = _build(scene), _count(scene)

    _names_exactly(built["error"], scene, changed)
    _names_exactly(counted["error"], scene, changed)
    assert built["error"] == counted["error"]


def test_the_build_route_over_a_removed_csv_answers_400_with_the_tools_sentence(
    client: TestClient, opened_project: Path
) -> None:
    scene = _scene(opened_project)
    (removed,) = _change(scene, [], [0])

    resp = client.post(
        "/api/results/plant_mapping/build",
        json={
            "name": "valley",
            "dataset_root": str(scene.photos),
            "plant_registry": scene.registry,
            "user": "tester",
        },
    )

    assert resp.status_code == 400, resp.text
    assert removed.name in resp.json()["detail"]
    assert resp.json()["detail"] == _build(scene)["error"]


def test_a_build_over_an_intact_registry_maps_its_plants(
    opened_project: Path,
) -> None:
    scene = _scene(opened_project)

    built = _build(scene)

    assert "error" not in built, built
    mapping = plant_mapping.load_mapping(scene.project, "valley")
    assert {a.plant_id for a in mapping.assignments[DATE]} == set(PLANTS)


def test_the_build_route_over_an_intact_registry_maps_its_plants(
    client: TestClient, opened_project: Path
) -> None:
    scene = _scene(opened_project)

    resp = client.post(
        "/api/results/plant_mapping/build",
        json={
            "name": "valley",
            "dataset_root": str(scene.photos),
            "plant_registry": scene.registry,
            "user": "tester",
        },
    )

    assert resp.status_code == 200, resp.text
    mapping = plant_mapping.load_mapping(scene.project, "valley")
    assert {a.plant_id for a in mapping.assignments[DATE]} == set(PLANTS)


def test_an_orthomosaic_count_over_an_intact_registry_reaches_the_delivery_gate(
    opened_project: Path,
) -> None:
    scene = _scene(opened_project)

    refused = _count(scene)

    assert "no assessment answers" in refused["error"], refused
    assert not any(c.name in refused["error"] for c in scene.csvs)


def test_the_export_route_over_an_intact_registry_delivers_the_counts(
    client: TestClient, opened_project: Path
) -> None:
    from tests._web_fixtures import acknowledged_post

    scene = _scene(opened_project)

    resp = acknowledged_post(
        client,
        "/api/results/export_count_csv",
        {
            "delivery": {
                "kind": "orthomosaic_plant_counts",
                "dataset_root": str(scene.bucket.root),
                "bucket": scene.bucket.name,
                "plant_registry": scene.registry,
                "delivered_phenotype": "stem_count",
                "plants": PLANTS,
            },
            "filename": "plant_counts.csv",
        },
        reason="a look before assessment",
    )

    assert resp.status_code == 200, resp.text[:300]
    assert "plot0" in resp.text
