"""Tests for the general per-plant CSV door.

Per-image records are aggregated and delivered over published buckets, their plant identities
checked against a mapping built by ``build_plant_mapping``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tcip_mcp.dataset_layout import image_dir
from tests import _trait_fixtures as fx
from tests._mapping_fixtures import PLOTS, write_plots_csv

torch = pytest.importorskip("torch")

KIND = "per_plant_count_aggregate"
SCOPE = {"subject": fx.COUNT_SUBJECT}
pytestmark = pytest.mark.usefixtures("confirmed_count_aggregate")


def _scene(
    tmp_path: Path, plants: dict[str, tuple[float, float]]
) -> tuple[Path, Path, str]:
    """Write a registered dataset of one geolocated image per plant and a plant CSV naming them.

    The images share one date. Returns ``(dataset root, plant csv, date)``.
    """
    from tcip_mcp.tools.project_tools import (
        initialize_project,
        register_dataset,
    )
    from tcip_mcp.traits import registered_crops
    from tests._mapping_fixtures import PHOTO_POSITION_ERROR_M, write_photos
    from tests._web_fixtures import INVARIANCES

    assert "error" not in initialize_project(
        str(tmp_path),
        "Orchard",
        site="orchard block",
        invariances=INVARIANCES,
        actor=None,
    )
    dataset_root, date = tmp_path / "ds", "2026-02-11"
    write_photos(
        image_dir(dataset_root, date),
        {
            f"{plant}.jpg": (lat, lon, PHOTO_POSITION_ERROR_M)
            for plant, (lat, lon) in plants.items()
        },
        datetime(2026, 2, 11, 9, 30),
        timedelta(minutes=5),
    )
    result = register_dataset(
        tmp_path,
        str(dataset_root),
        crop=sorted(registered_crops())[0],
        actor=None,
    )
    assert "error" not in result, result
    plant_csv = write_plots_csv(tmp_path / "plants.csv", plants)
    return dataset_root, plant_csv, date


ONE_PLANT = {"P1": (43.19670, -90.058000)}
TWO_PLANTS = {**ONE_PLANT, "P2": (43.19680, -90.057000)}


def _mapped(
    tmp_path: Path, plants: dict[str, tuple[float, float]] = ONE_PLANT
) -> tuple[Path, str]:
    """Build :func:`_scene` with the mapping ``valley`` over it; return ``(dataset root, date)``."""
    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tests._mapping_fixtures import register_plant_registry_for

    dataset_root, plant_csv, date = _scene(tmp_path, plants)
    mapped = build_plant_mapping(
        tmp_path,
        name="valley",
        dataset_root=str(dataset_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
        actor=None,
    )
    assert "error" not in mapped, mapped
    return dataset_root, date


def _bucket(
    project: Path,
    dataset_root: Path,
    date: str,
    counts: dict[str, int],
    name: str = "manual",
) -> str:
    """Publish a bucket named ``<name>/<date>`` under ``project`` and return its name.

    The bucket lies under ``dataset_root`` and holds ``counts[stem]`` detections on each image
    ``images/<date>/<stem>.jpg``.
    """
    from tests._chain_fixtures import published

    results = [
        {
            "image": str(image_dir(dataset_root, date) / f"{stem}.jpg"),
            "width": 8,
            "height": 8,
            "boxes": [[1.0, 1.0, 3.0, 3.0]] * n,
            "scores": [0.9] * n,
            "labels": [1] * n,
            "cap": n + 1,
        }
        for stem, n in counts.items()
    ]
    return published(project, f"{name}/{date}", results, scope=SCOPE).name


def _records(*plant_counts: tuple[str, int | None]) -> list[dict]:
    """Return one per-image record per ``(plant_id, count)``, omitting a ``None`` count."""
    return [
        {
            "plant_id": plant,
            "plant_attribution": "image",
            **({"count": count} if count is not None else {}),
        }
        for plant, count in plant_counts
    ]


def _deliver(
    tmp_path: Path,
    records: list[dict],
    plants: list[str],
    dataset_root: Path,
    buckets: list[str],
    **kwargs: str | None,
) -> dict:
    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv

    return deliver_per_plant_csv(
        tmp_path,
        records,
        str(tmp_path / "o.csv"),
        "stem_count",
        KIND,
        plants,
        str(dataset_root),
        buckets,
        value_key="count",
        **kwargs,
        actor=None,
    )


def test_a_verified_mapping_over_an_unassessed_bucket_reaches_the_gate_and_refuses_there(
    tmp_path: Path,
) -> None:
    """A verified mapping over an unassessed bucket reaches the gate and refuses there.

    Over a mapping built on a registered registry, a published bucket and per-image records, the
    delivery reaches the one gate; the bucket was published under no assessment, so it refuses
    there, writing no output CSV.
    """
    dataset_root, date = _mapped(tmp_path, TWO_PLANTS)
    bucket = _bucket(
        tmp_path, dataset_root, date, {"P1": 2, "P2": 1}, name="run"
    )

    refused = _deliver(
        tmp_path,
        _records(("P1", 2), ("P2", 1)),
        ["P1", "P2"],
        dataset_root,
        [bucket],
        crop="currant",
        plant_mapping="valley",
    )

    assert "no assessment answers" in refused["error"]
    assert not (tmp_path / "o.csv").exists()


def test_an_unknown_plant_mapping_refuses_by_name(tmp_path: Path) -> None:
    """A ``plant_mapping`` with no stored record is refused by name, and no CSV is written.

    The refusal names the mapping and ``build_plant_mapping``.
    """
    dataset_root, date = _mapped(tmp_path)
    bucket = _bucket(tmp_path, dataset_root, date, {"P1": 1})

    res = _deliver(
        tmp_path,
        _records(("P1", 3)),
        ["P1"],
        dataset_root,
        [bucket],
        plant_mapping="no-such-mapping",
    )

    assert "no-such-mapping" in res["error"]
    assert "build_plant_mapping" in res["error"]
    assert not (tmp_path / "o.csv").exists()


def test_a_delivered_date_the_mapping_does_not_cover_refuses(
    tmp_path: Path,
) -> None:
    from tests._image_fixtures import write_geo_image

    dataset_root, date = _mapped(tmp_path)
    uncovered = "2026-03-01"
    write_geo_image(
        image_dir(dataset_root, uncovered) / "P1.jpg",
        *ONE_PLANT["P1"],
        datetime(2026, 3, 1, 9, 30),
    )

    res = _deliver(
        tmp_path,
        _records(("P1", 1)),
        ["P1"],
        dataset_root,
        [
            _bucket(tmp_path, dataset_root, date, {"P1": 1}),
            _bucket(tmp_path, dataset_root, uncovered, {"P1": 1}),
        ],
        plant_mapping="valley",
    )

    assert "does not cover" in res["error"] and uncovered in res["error"]
    assert not (tmp_path / "o.csv").exists()


def test_predictions_under_another_dataset_than_the_mappings_refuse(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.traits import registered_crops
    from tests._image_fixtures import write_geo_image

    _dataset_root, date = _mapped(tmp_path)
    other = tmp_path / "ds2"
    write_geo_image(
        image_dir(other, date) / "P1.jpg",
        *ONE_PLANT["P1"],
        datetime(2026, 2, 11, 9, 30),
    )
    assert "error" not in register_dataset(
        tmp_path, str(other), crop=sorted(registered_crops())[0], actor=None
    )

    res = _deliver(
        tmp_path,
        _records(("P1", 1)),
        ["P1"],
        other,
        [_bucket(tmp_path, other, date, {"P1": 1})],
        plant_mapping="valley",
    )

    assert "different dataset" in res["error"]
    assert not (tmp_path / "o.csv").exists()


def test_a_mapping_with_no_capture_at_all_for_a_delivered_date_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mapping with no capture for a delivered date refuses on the record's own evidence.

    The mapping is recorded with no capture at all for that date.
    """
    from tests.test_ungeoreferenced_capture_refusal import (
        DATE,
        _delivery_scene,
        _persist_synthetic_mapping,
    )

    _, dataset_root, preds_by_date = _delivery_scene(tmp_path, monkeypatch)
    plant_csv = write_plots_csv(tmp_path / "plants.csv")
    _persist_synthetic_mapping(
        tmp_path,
        dataset_root,
        "valley",
        plant_csvs=[
            {
                "path": str(plant_csv),
                "sha256": "0" * 64,
                "n_plants": len(PLOTS),
            }
        ],
        assignments={DATE: []},
    )

    res = _deliver(
        tmp_path,
        _records(("P1", 3)),
        ["P1"],
        dataset_root,
        list(preds_by_date.values()),
        plant_mapping="valley",
    )

    assert "recorded no capture at all" in res["error"]
    assert not (tmp_path / "o.csv").exists()


def test_a_delivered_plant_the_mapping_never_assigned_refuses_by_name(
    tmp_path: Path,
) -> None:
    """A delivered plant the mapping never assigned on a delivered date is refused by name.

    The refusal names the unassigned plant and not the assigned one, and no output CSV is
    written.
    """
    dataset_root, date = _mapped(tmp_path)
    bucket = _bucket(tmp_path, dataset_root, date, {"P1": 1})

    res = _deliver(
        tmp_path,
        _records(("P1", 1), ("GHOST", 2)),
        ["P1", "GHOST"],
        dataset_root,
        [bucket],
        plant_mapping="valley",
    )

    assert "GHOST" in res["error"] and "'P1'" not in res["error"]
    assert not (tmp_path / "o.csv").exists()


def test_a_capture_added_since_the_mapping_was_built_refuses(
    tmp_path: Path,
) -> None:
    """A capture added since the mapping was built refuses the delivery.

    The refusal says the mapping does not cover what is on disk now, and no CSV is written.
    """
    from tests._image_fixtures import write_geo_image

    dataset_root, date = _mapped(tmp_path)
    bucket = _bucket(tmp_path, dataset_root, date, {"P1": 1})
    write_geo_image(
        image_dir(dataset_root, date) / "P2.jpg",
        43.19680,
        -90.057000,
        datetime(2026, 2, 11, 9, 35),
    )

    res = _deliver(
        tmp_path,
        _records(("P1", 1)),
        ["P1"],
        dataset_root,
        [bucket],
        plant_mapping="valley",
    )

    assert "does not cover what is on disk now" in res["error"]
    assert not (tmp_path / "o.csv").exists()
