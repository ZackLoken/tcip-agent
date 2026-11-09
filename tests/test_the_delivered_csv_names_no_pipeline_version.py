"""A delivered per-plant CSV carries no ``pipeline_version`` column, and no door takes one.

What produced a result is the producer its delivery event records, reached through the row's
``delivery_event_id``. Through the per-plant tool, the orthomosaic tool and the count route, each
read back from the file it wrote, and through the tool listing and the count payload's schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests import _trait_fixtures as fx

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def _header(path: Path) -> list[str]:
    """Return the column names of the UTF-8 CSV file at ``path``."""
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def _columns_without_a_version(path: Path) -> None:
    """Assert the CSV at ``path`` has its neighboring columns and no ``pipeline_version``."""
    header = _header(path)
    assert {"n_images", "plant_id_source", "delivery_event_id"} <= set(
        header
    ), header
    assert "pipeline_version" not in header, header


def test_a_per_plant_delivery_writes_no_pipeline_version_column(
    tmp_path: Path,
) -> None:
    """A validated per-plant delivery through the tool writes a header without the column.

    The bucket is published under a per-plant count assessment and mapped by ``valley``, read
    back from the written file.
    """
    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv
    from tcip_mcp.traits import PER_PLANT_COUNT_AGGREGATE
    from tests.test_aggregation_multi_plant_delivery import (
        _PLANTS,
        _RECORDS,
        _assessed,
    )

    fx.seed_delivery_traits(tmp_path)
    bucket = _assessed(tmp_path, PER_PLANT_COUNT_AGGREGATE, "exp-no-version")
    out = tmp_path / "per_plant.csv"

    delivered = deliver_per_plant_csv(
        tmp_path,
        _RECORDS,
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
    _columns_without_a_version(out)


def test_an_orthomosaic_delivery_writes_no_pipeline_version_column_at_both_doors(
    client: TestClient, tmp_path: Path
) -> None:
    """The orthomosaic tool and the count route each write a header without the column.

    Each ships under an acknowledgment of the digest its own refusal named, the route's through
    a breeder's acknowledgment from the page, read back from the CSV each wrote.
    """
    from tests._web_fixtures import acknowledged_post, open_new_project
    from tests.test_orthomosaic_tools import _GRID, _deliver
    from tests.test_tcip_web_results_routes import COUNT_ROUTE, _per_plant
    from tests.test_the_delivered_crop_is_the_registrys import (
        _grid_registry,
        _orthomosaic_scene,
    )

    bucket = _orthomosaic_scene(tmp_path)
    registry = _grid_registry(tmp_path)

    tool = _deliver(tmp_path, bucket, registry, _GRID)
    open_new_project(tmp_path)
    route = acknowledged_post(
        client,
        COUNT_ROUTE,
        _per_plant(bucket, registry),
        reason="a look before assessment",
    )

    assert "error" not in tool, tool
    _columns_without_a_version(tmp_path / "counts.csv")
    assert route.status_code == 200, route.text[:300]
    _columns_without_a_version(
        tmp_path / "results_export" / "plant_counts.csv"
    )


def test_no_door_takes_a_pipeline_version(
    client: TestClient, opened_project: Path
) -> None:
    """The two per-plant tools list no ``pipeline_version``, and the count payload refuses one.

    The listing is the MCP server's own; the payload's refusal is its schema's, 422.
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
                "pipeline_version": "v1",
            },
            "filename": "counts.csv",
            "user": "breeder",
        },
    )

    for name in ("deliver_per_plant_csv", "deliver_orthomosaic_plant_counts"):
        assert {"plant_registry", "plant_mapping"} & set(listed[name])
        assert "pipeline_version" not in listed[name], name
    assert refused.status_code == 422, refused.text
    assert "pipeline_version" in refused.text
