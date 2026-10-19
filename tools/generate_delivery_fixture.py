"""Write served delivery events as a fixture file, produced by the platform itself.

In a scratch session (``tools/fixture_session.py``), in projects each created through
``tests._web_fixtures.named_project``, deliveries are made through the platform's own producers
and the test suite's own scenes for them: a per-image count delivery
over a bucket its assessment validates, a phenology delivery through a plant mapping shipped
under a breeder's acknowledgment, and the orthomosaic door's nearest-plant and canopy-segment
deliveries. Each project is opened through ``POST /api/projects/open`` and its records served by
``GET /api/results/delivery-events``; the phenology project's record is served again after a
superseding rebuild archives the mapping it cites. The served records are written,
:func:`rebased`, to ``frontend/src/test/deliveryEvents.json``, keyed by fixture name.

    python -m tools.generate_delivery_fixture
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tools.fixture_session import (
    opened,
    replaced,
    scratch_session,
    write_fixture,
)

if TYPE_CHECKING:
    from fastapi.testclient import TestClient
    from pydantic import JsonValue

    from tcip_web.state import OpenProject

_REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATED_PATH = (
    _REPO_ROOT
    / "packages"
    / "tcip-web"
    / "frontend"
    / "src"
    / "test"
    / "deliveryEvents.json"
)

SCRATCH_ROOT = "/scratch"
"""What the scratch directory the deliveries were made under reads as in every served path."""


def rebased(
    value: dict[str, JsonValue], scratch: Path
) -> dict[str, JsonValue]:
    """Return ``value`` with every spelling of ``scratch`` in its strings made :data:`SCRATCH_ROOT`.

    The scratch directory differs with where a run was made.
    """

    def leaf(_key: str | None, text: JsonValue) -> JsonValue:
        if not isinstance(text, str):
            return text
        for spelling in (str(scratch), scratch.as_posix()):
            text = text.replace(spelling, SCRATCH_ROOT)
        return text.replace("\\", "/") if SCRATCH_ROOT in text else text

    return replaced(value, leaf)


def deliveries(workspace: Path, client: TestClient) -> dict[str, JsonValue]:
    """Return the served delivery events of the scenes this module names, by fixture name.

    The scenes are made in projects created under ``workspace`` and served through ``client``.
    """
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts
    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tests import (
        _trait_fixtures as fx,
        test_orthomosaic_tools as ortho,
        test_plant_mapping_binding as binding,
    )
    from tests._chain_fixtures import run_the_chain
    from tests._mapping_fixtures import (
        POSITION_ERROR_M,
        register_plant_registry_for,
    )
    from tests._web_fixtures import named_project
    from tests.test_second_trait_acceptance import _seed_currant_bloom_trait

    def served(project: OpenProject) -> JsonValue:
        opened(client, project)
        response = client.get("/api/results/delivery-events")
        response.raise_for_status()
        (record,) = response.json()["records"]
        return record

    def delivered(result: dict) -> None:
        if "error" in result:
            raise SystemExit(
                f"generate_delivery_fixture.py: a scene's delivery refused: {result}"
            )

    out: dict[str, JsonValue] = {}

    root = workspace / "counts"
    project = named_project(root, "Counts")
    chain = run_the_chain(root, experiment_id="exp-fixture")
    delivered(
        deliver_per_image_counts(
            root,
            str(chain.root),
            chain.bucket,
            output_path=str(root / "counts.csv"),
            trait=fx.COUNT_TRAIT,
        )
    )
    out["validated"] = served(project)

    root = workspace / "phenology"
    project = binding._init(root)
    dataset_root = binding._dataset(root)
    images_root, plant_csv, preds_by_date = binding._write_scene(dataset_root)
    registry = register_plant_registry_for(root, [plant_csv])
    delivered(
        build_plant_mapping(
            root,
            name="valley",
            images_root=str(images_root),
            plant_registry=registry,
        )
    )
    _seed_currant_bloom_trait(root)
    delivered(
        binding._deliver(
            root,
            trait="currant_bloom",
            mapping_name="valley",
            plants=[str(plot) for plot in binding.POPULATION],
            dataset_root=dataset_root,
            buckets=preds_by_date.values(),
            output_csv_path=str(root / "phenology.csv"),
        )
    )
    out["acknowledged_mapping"] = served(project)
    delivered(
        build_plant_mapping(
            root,
            name="valley",
            images_root=str(images_root),
            plant_registry=registry,
            supersede=True,
        )
    )
    out["archived_mapping"] = served(project)

    root = workspace / "nearest"
    project = named_project(root, "Nearest")
    fx.seed_delivery_traits(root)
    fx.seed_confirmed_aggregate(root, "stem_count", value_keys=["count"])
    raster_path = ortho._raster(root)
    ortho._write_geo_raster(raster_path)
    bucket = ortho._raster_bucket(
        root, raster_path, [(8.0, 8.0, 12.0, 12.0), (48.0, 8.0, 52.0, 12.0)]
    )
    registry = ortho._plant_registry(
        root,
        ortho._plants_csv_at(
            root,
            raster_path,
            [
                ("plot0", 10.0, 10.0),
                ("plot2", 50.0, 10.0),
                ("plot9", 200.0, 200.0),
            ],
        ),
    )
    delivered(ortho._deliver(root, bucket, registry, ["plot0", "plot2"]))
    out["registry"] = served(project)

    root = workspace / "segmented"
    project = named_project(root, "Segmented")
    fx.seed_delivery_traits(root)
    fx.seed_confirmed_aggregate(root, "stem_count", value_keys=["count"])
    _dataset_root, raster_path, bucket = ortho._canopy_setup(
        root,
        [
            (8.0, 8.0, 12.0, 12.0),
            (48.0, 8.0, 52.0, 12.0),
            (49.0, 49.0, 51.0, 51.0),
            (30.0, 30.0, 32.0, 32.0),
        ],
    )
    registry = ortho._plant_registry(
        root,
        ortho._plants_csv_at(
            root,
            raster_path,
            [
                ("plot0", 10.0, 10.0),
                ("plot1", 10.0, 50.0),
                ("plot2", 50.0, 10.0),
                ("plot3", 20.5, 45.0),
                ("plot4", 44.0, 44.0),
                ("plot9", 200.0, 200.0),
            ],
        ),
    )
    ortho._write_canopy_document(
        raster_path,
        [
            (5.0, 5.0, 15.0, 15.0),
            (45.0, 5.0, 55.0, 15.0),
            (20.0, 40.0, 30.0, 50.0),
            (40.0, 40.0, 52.0, 52.0),
            (48.0, 48.0, 60.0, 60.0),
        ],
    )
    delivered(
        ortho._deliver(
            root,
            bucket,
            registry,
            ["plot0", "plot2"],
            canopy_subject="canopy",
            position_error_m=POSITION_ERROR_M,
        )
    )
    out["canopy"] = served(project)
    return out


def produce() -> dict[str, JsonValue]:
    """Return the fixture's content, made in a scratch session.

    The content is :func:`rebased` against the scratch directory holding its workspace.
    """
    with scratch_session() as (workspace, client):
        served = deliveries(workspace, client)
    return rebased(served, workspace.parent)


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
