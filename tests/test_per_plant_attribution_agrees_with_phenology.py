"""The per-plant CSV door and the phenology curves attribute each image to the same plant.

Both routes read one mapping ``build_plant_mapping`` built and the same published buckets; each
is read from the file its own delivery wrote.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests import _trait_fixtures as fx, csv_rows
from tests._mapping_fixtures import CROP, PHOTO_POSITION_ERROR_M, PLOTS

if TYPE_CHECKING:
    from tcip_mcp.pipelines.postprocessing.phenology import Projection

torch = pytest.importorskip("torch")

DATES = ("2026-02-11", "2026-02-25")
DETECTIONS = {
    (DATES[0], "a"): ("P1", 1),
    (DATES[0], "b"): ("P1", 2),
    (DATES[0], "c"): ("P2", 4),
    (DATES[1], "a"): ("P2", 8),
    (DATES[1], "c"): ("P1", 16),
    (DATES[1], "d"): ("P2", 32),
}
"""Each image by ``(date, stem)``: the plot it is taken at and its number of open flowers.

The counts are distinct powers of two, so each set of images has its own sum, and stems ``a``
and ``c`` stand at a different plot on each date."""
OFF_DISK = (DATES[0], "b")
"""The one documented image whose file leaves the disk after the buckets are published."""
PHENOTYPE = "bloom_per_axil"
"""A currant phenotype the fixture's count trait delivers as the sum of its plant's images."""


def _read_images() -> dict[tuple[str, str], tuple[str, int]]:
    """Return :data:`DETECTIONS` without :data:`OFF_DISK`, the images a delivery reads."""
    return {key: v for key, v in DETECTIONS.items() if key != OFF_DISK}


def _scene(project: Path) -> tuple[Path, list[str]]:
    """Map :data:`DETECTIONS`' photographs and publish one bucket per date; return both.

    The dataset is registered at :data:`~tests._mapping_fixtures.CROP` with the bloom subject
    registry, mapped by ``build_plant_mapping`` against :data:`PLOTS`, and each date's bucket
    holds each image's open flowers, a document for every image. :data:`OFF_DISK`'s file is then
    removed, so the mapping and its bucket still name it and the disk no longer holds it.
    """
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tests._chain_fixtures import detected, detection_pass, published
    from tests._mapping_fixtures import (
        register_plant_registry_for,
        write_photos,
        write_plots_csv,
    )
    from tests.test_plant_mapping_binding import _dataset, _init
    from tests.test_second_trait_acceptance import BLOOM_STATE, FLOWERS

    _init(project)
    dataset_root = _dataset(project)
    for date in DATES:
        write_photos(
            image_dir(dataset_root, date),
            {
                f"{stem}.jpg": (*PLOTS[plot], PHOTO_POSITION_ERROR_M)
                for (d, stem), (plot, _) in DETECTIONS.items()
                if d == date
            },
            datetime.fromisoformat(date).replace(hour=9),
            timedelta(minutes=1),
        )
    built = build_plant_mapping(
        project,
        name="valley",
        dataset_root=str(dataset_root),
        plant_registry=register_plant_registry_for(
            project, [write_plots_csv(project / "plants.csv")], crop=CROP
        ),
        actor=None,
    )
    assert "error" not in built, built
    open_flower = detected(
        (1.0, 1.0, 3.0, 3.0),
        0.9,
        attributes=[BLOOM_STATE.values.index("open")],
    )
    buckets = [
        published(
            project,
            f"run/{date}",
            [
                detection_pass(
                    image_dir(dataset_root, date) / f"{stem}.jpg",
                    (8, 8),
                    [open_flower] * n,
                    cap=n + 1,
                )
                for (d, stem), (_, n) in DETECTIONS.items()
                if d == date
            ],
            scope={"subject": "flower"},
            registry=FLOWERS,
        ).name
        for date in DATES
    ]
    (image_dir(dataset_root, OFF_DISK[0]) / f"{OFF_DISK[1]}.jpg").unlink()
    return dataset_root, buckets


def _confirm_traits(project: Path) -> None:
    """Confirm the bloom crossing trait and a count trait summing each plant's open flowers."""
    from tests.test_second_trait_acceptance import _seed_currant_bloom_trait

    _seed_currant_bloom_trait(project)
    fx.propose(
        project,
        fx.entry(
            "axil_bloom",
            (PHENOTYPE,),
            count_objective="count_unbiased",
            localization="center_match",
            plant_aggregation="sum",
            notes="Test-only, provisional: a plant's open flowers summed over its images.",
        ),
    )
    fx.seed_confirmed_aggregate(
        project, PHENOTYPE, value_keys=["count"], measured_subject="flower"
    )


def _per_plant_csv(
    project: Path, dataset_root: Path, buckets: list[str]
) -> Path:
    """Deliver each read image's open flowers through ``deliver_per_plant_csv``; return the CSV.

    The records leave :data:`OFF_DISK` out, since the door refuses a record of an image it did not
    read.
    """
    from tcip_mcp.server import binding_for
    from tests._server_calls import called_through_the_server
    from tests.test_delivery_events import _acknowledged_per_plant_csv

    out = project / "per_plant.csv"
    tool, arguments = _acknowledged_per_plant_csv(
        project,
        {
            "results": [
                {"date": date, "stem": stem, "value": n}
                for (date, stem), (_, n) in _read_images().items()
            ],
            "output_path": str(out),
            "delivered_phenotype": PHENOTYPE,
            "delivery_kind": "per_plant_count_aggregate",
            "plants": sorted(PLOTS),
            "dataset_root": str(dataset_root),
            "buckets": buckets,
            "value_key": "count",
            "plant_mapping": "valley",
        },
    )
    answer = called_through_the_server(
        tool, arguments, bound=binding_for(project)
    )
    assert isinstance(answer, dict) and "error" not in answer, answer
    return out


def _phenology_csv(
    project: Path,
    dataset_root: Path,
    buckets: list[str],
    projection: Projection,
) -> Path:
    """Measure the bloom trait over the same buckets and deliver ``projection``; return the CSV."""
    from tcip_mcp.pipelines.postprocessing import phenology
    from tests._chain_fixtures import acknowledged

    out = project / f"{projection}.csv"
    measurement = phenology.measure_phenology(
        project,
        trait="currant_bloom",
        mapping_name="valley",
        dataset_root=dataset_root,
        buckets=buckets,
        plants=sorted(PLOTS),
    )
    acknowledged(
        project,
        lambda ack: phenology.deliver_phenology(
            project,
            measurement,
            projection=projection,
            output_path=out,
            acknowledgment_id=ack,
            door=f"test_{projection}",
            actor=None,
        ),
    )
    return out


def test_each_plants_delivered_sum_is_its_curve_rows_total_over_the_same_images(
    tmp_path: Path,
) -> None:
    """Each plant's summed value is the total its curve rows count, over the same images.

    Every mapping date is delivered and every attributed image has a bucket document, but
    :data:`OFF_DISK`'s file is gone, so both routes treat it as unread: the per-plant door takes
    no record of it and the curves count it as a missing image. Each image carries its own power of
    two, so a plant's sum names exactly its read images: an image of one plant counted toward the
    other changes both plants' sums. The per-plant ``n_images`` equals the plant's curve rows'
    images less their missing ones, and the plant of the unread image has no milestone while the
    other plant, every image read, has them.
    """
    _confirm_traits(tmp_path)
    dataset_root, buckets = _scene(tmp_path)

    per_plant = {
        row["plant_id"]: row
        for row in csv_rows(_per_plant_csv(tmp_path, dataset_root, buckets))
    }
    curves = csv_rows(
        _phenology_csv(tmp_path, dataset_root, buckets, "curves")
    )
    milestones = {
        row["plant_id"]: row
        for row in csv_rows(
            _phenology_csv(tmp_path, dataset_root, buckets, "milestones")
        )
    }

    totals = {
        plant: sum(int(r["n_total"]) for r in curves if r["plant_id"] == plant)
        for plant in PLOTS
    }
    images = {
        plant: sum(
            int(r["n_images"]) - int(r["n_missing"])
            for r in curves
            if r["plant_id"] == plant
        )
        for plant in PLOTS
    }
    assert {p: int(float(r["value"])) for p, r in per_plant.items()} == totals
    assert {p: int(r["n_images"]) for p, r in per_plant.items()} == images
    assert totals == {
        plant: sum(n for p, n in _read_images().values() if p == plant)
        for plant in PLOTS
    }
    unread_plant = DETECTIONS[OFF_DISK][0]
    for plant, row in milestones.items():
        assert bool(row["bloom_50per_date"]) is (plant != unread_plant), plant
