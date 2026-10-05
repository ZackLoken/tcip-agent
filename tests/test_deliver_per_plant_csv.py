"""Tests for the general per-plant CSV door: ``aggregate_per_plant``'s own output delivered over
published buckets, its plant identities checked against a mapping built by
``build_plant_mapping``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from tests import _trait_fixtures as fx
from tests._mapping_fixtures import write_plant_csv

torch = pytest.importorskip("torch")

KIND = "per_plant_count_aggregate"
SCOPE = {"subject": fx.COUNT_SUBJECT}
pytestmark = pytest.mark.usefixtures("confirmed_count_aggregate")


def _scene(tmp_path: Path, plants: dict[str, tuple[float, float]]) -> tuple[Path, Path, str]:
    """A registered dataset of one geolocated image per plant on one date and a plant CSV
    naming them; ``(dataset root, plant csv, date)``."""
    from tcip_mcp.tools.project_tools import initialize_project, register_dataset
    from tcip_mcp.traits import registered_crops
    from tests._image_fixtures import write_geo_image

    assert "error" not in initialize_project(str(tmp_path), "Orchard", site="orchard block")
    dataset_root, date = tmp_path / "ds", "2026-02-11"
    for i, (plant, (lat, lon)) in enumerate(plants.items()):
        write_geo_image(dataset_root / "images" / date / f"{plant}.jpg", lat, lon,
                        datetime(2026, 2, 11, 9, 30 + 5 * i))
    result = register_dataset(tmp_path, str(dataset_root), crop=sorted(registered_crops())[0])
    assert "error" not in result, result
    plant_csv = write_plant_csv(tmp_path / "plants.csv", [
        {"plot": plant, "accession": f"acc-{plant}", "lat": lat, "lon": lon}
        for plant, (lat, lon) in plants.items()])
    return dataset_root, plant_csv, date


ONE_PLANT = {"P1": (43.19670, -90.058000)}
TWO_PLANTS = {**ONE_PLANT, "P2": (43.19680, -90.057000)}


def _mapped(tmp_path: Path, plants=ONE_PLANT, **stated) -> tuple[Path, str]:
    """:func:`_scene` with the mapping ``valley`` built over it; ``(dataset root, date)``."""
    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tests._mapping_fixtures import register_plant_registry_for

    dataset_root, plant_csv, date = _scene(tmp_path, plants)
    mapped = build_plant_mapping(
        tmp_path, name="valley", images_root=str(dataset_root / "images"),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]), **stated)
    assert "error" not in mapped, mapped
    return dataset_root, date


def _bucket(project: Path, dataset_root: Path, date: str, counts: dict[str, int],
            name: str = "manual") -> str:
    """A bucket named ``<name>/<date>`` under ``dataset_root`` holding ``counts[stem]``
    detections on each image ``images/<date>/<stem>.jpg``, published under ``project``; its
    name."""
    from tests._chain_fixtures import published

    results = [{"image": str(dataset_root / "images" / date / f"{stem}.jpg"), "width": 8,
                "height": 8, "boxes": [[1.0, 1.0, 3.0, 3.0]] * n, "scores": [0.9] * n,
                "labels": [1] * n} for stem, n in counts.items()]
    return published(project, f"{name}/{date}", results, scope=SCOPE).name


def _results(*plant_counts: tuple[str, int | None]) -> list[dict]:
    from tcip_mcp.pipelines.postprocessing.aggregation import aggregate_per_plant

    records = [{"plant_id": plant, "plant_attribution": "image",
                **({"count": count} if count is not None else {})}
               for plant, count in plant_counts]
    return aggregate_per_plant(records, strategy="count", value_key="count")


def _deliver(tmp_path: Path, results: list[dict], plants: list[str], dataset_root: Path,
             buckets: list[str], **kwargs) -> dict:
    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv

    return deliver_per_plant_csv(tmp_path, results, str(tmp_path / "o.csv"), "stem_count", KIND,
                                 plants, str(dataset_root), buckets, **kwargs)


def test_a_verified_mapping_over_an_unassessed_bucket_reaches_the_gate_and_refuses_there(
    tmp_path,
):
    """The whole producer chain: a mapping built over a registered registry, a published bucket
    and records from ``aggregate_per_plant``. The mapping verifies and every delivered plant is
    one it assigned, so the delivery reaches the one gate; the bucket was published under no
    assessment and this door takes no acknowledgment, so it refuses there, writing nothing."""
    dataset_root, date = _mapped(tmp_path, TWO_PLANTS)
    bucket = _bucket(tmp_path, dataset_root, date, {"P1": 2, "P2": 1}, name="run")

    refused = _deliver(tmp_path, _results(("P1", 2), ("P2", 1)), ["P1", "P2"], dataset_root,
                       [bucket], crop="currant", plant_mapping="valley")

    assert "no assessment answers" in refused["error"]
    assert not (tmp_path / "o.csv").exists()


def test_an_unknown_plant_mapping_refuses_by_name(tmp_path):
    """A stated ``plant_mapping`` is a claim the data must positively carry: naming one with no
    stored record refuses by name, before the writer ever runs."""
    dataset_root, date = _mapped(tmp_path, nn_tolerance_m=10.0)
    bucket = _bucket(tmp_path, dataset_root, date, {"P1": 1})

    res = _deliver(tmp_path, _results(("P1", 3)), ["P1"], dataset_root, [bucket],
                   plant_mapping="no-such-mapping")

    assert "no-such-mapping" in res["error"]
    assert "build_plant_mapping" in res["error"]
    assert not (tmp_path / "o.csv").exists()


def test_a_delivered_date_the_mapping_does_not_cover_refuses(tmp_path):
    from tests._image_fixtures import write_geo_image

    dataset_root, date = _mapped(tmp_path, nn_tolerance_m=10.0)
    uncovered = "2026-03-01"
    write_geo_image(dataset_root / "images" / uncovered / "P1.jpg", *ONE_PLANT["P1"],
                    datetime(2026, 3, 1, 9, 30))

    res = _deliver(tmp_path, _results(("P1", 1)), ["P1"], dataset_root, [
        _bucket(tmp_path, dataset_root, date, {"P1": 1}),
        _bucket(tmp_path, dataset_root, uncovered, {"P1": 1})], plant_mapping="valley")

    assert "does not cover" in res["error"] and uncovered in res["error"]
    assert not (tmp_path / "o.csv").exists()


def test_predictions_under_another_dataset_than_the_mappings_refuse(tmp_path):
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.traits import registered_crops
    from tests._image_fixtures import write_geo_image

    _dataset_root, date = _mapped(tmp_path, nn_tolerance_m=10.0)
    other = tmp_path / "ds2"
    write_geo_image(other / "images" / date / "P1.jpg", *ONE_PLANT["P1"],
                    datetime(2026, 2, 11, 9, 30))
    assert "error" not in register_dataset(tmp_path, str(other),
                                           crop=sorted(registered_crops())[0])

    res = _deliver(tmp_path, _results(("P1", 1)), ["P1"], other,
                   [_bucket(tmp_path, other, date, {"P1": 1})], plant_mapping="valley")

    assert "different dataset" in res["error"]
    assert not (tmp_path / "o.csv").exists()


def test_a_mapping_with_no_capture_at_all_for_a_delivered_date_refuses(tmp_path, monkeypatch):
    """A mapping recorded with no capture at all for a delivered date refuses on the record's own
    evidence."""
    from tests.test_plant_mapping_binding import PLANTS
    from tests.test_ungeoreferenced_capture_refusal import (
        DATE, _delivery_scene, _persist_synthetic_mapping,
    )

    dataset_root, preds_by_date = _delivery_scene(tmp_path, monkeypatch)
    plant_csv = write_plant_csv(tmp_path / "plants.csv", PLANTS)
    _persist_synthetic_mapping(
        tmp_path, dataset_root, "valley",
        plant_csvs=[{"path": str(plant_csv), "sha256": "0" * 64, "n_plants": len(PLANTS)}],
        assignments={DATE: []})

    res = _deliver(tmp_path, _results(("P1", 3)), ["P1"], dataset_root,
                   list(preds_by_date.values()), plant_mapping="valley")

    assert "recorded no capture at all" in res["error"]
    assert not (tmp_path / "o.csv").exists()


def test_a_delivered_plant_the_mapping_never_assigned_refuses_by_name(tmp_path):
    """This door's one added claim: every delivered plant must be one the named mapping assigned
    on the delivered dates."""
    dataset_root, date = _mapped(tmp_path, nn_tolerance_m=10.0)
    bucket = _bucket(tmp_path, dataset_root, date, {"P1": 1})

    res = _deliver(tmp_path, _results(("P1", 1), ("GHOST", 2)), ["P1", "GHOST"], dataset_root,
                   [bucket], plant_mapping="valley")

    assert "GHOST" in res["error"] and "'P1'" not in res["error"]
    assert not (tmp_path / "o.csv").exists()


def test_a_capture_added_since_the_mapping_was_built_refuses(tmp_path):
    """A capture the mapping's own assignments do not name means the mapping does not cover what
    is on disk now."""
    from tests._image_fixtures import write_geo_image

    dataset_root, date = _mapped(tmp_path, nn_tolerance_m=10.0)
    bucket = _bucket(tmp_path, dataset_root, date, {"P1": 1})
    write_geo_image(dataset_root / "images" / date / "P2.jpg", 43.19680, -90.057000,
                    datetime(2026, 2, 11, 9, 35))

    res = _deliver(tmp_path, _results(("P1", 1)), ["P1"], dataset_root, [bucket],
                   plant_mapping="valley")

    assert "does not cover what is on disk now" in res["error"]
    assert not (tmp_path / "o.csv").exists()
