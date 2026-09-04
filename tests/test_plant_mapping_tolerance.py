"""build_mapping's own four nn_tolerance_m sources: derived from the grid pitch, the fallback
constant when the layout carries too few positioned plants to derive one, a stated value at or
under the derived cap, and a stated value above it capped down to it.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pytest

from tcip_mcp.pipelines.postprocessing.plant_mapping import (
    NN_TOLERANCE_METERS,
    build_mapping,
    grid_pitch_m,
    read_plant_csvs,
)

from tests.test_plant_mapping_binding import _write_geo_image

PLANTS = [
    {"plot": "P1", "accession": "acc-A", "lat": 43.19670, "lon": -90.058000},
    {"plot": "P2", "accession": "acc-B", "lat": 43.19670, "lon": -90.058037},
]


def _write_plant_csv(path: Path, plants: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["plot_name", "accession_name", "WGS84_centroid_x", "WGS84_centroid_y"])
        for p in plants:
            w.writerow([p["plot"], p["accession"], p["lon"], p["lat"]])


def _write_one_image(dataset_root: Path) -> Path:
    images_root = dataset_root / "images"
    _write_geo_image(
        images_root / "2026-02-11" / "P1_20260211.jpg",
        PLANTS[0]["lat"], PLANTS[0]["lon"], datetime(2026, 2, 11, 9, 30),
    )
    return images_root


def _build(tmp_path: Path, plants: list[dict], nn_tolerance_m: float | None) -> tuple:
    images_root = _write_one_image(tmp_path)
    plant_csv = tmp_path / "plants.csv"
    _write_plant_csv(plant_csv, plants)
    build = build_mapping(
        images_root, [plant_csv], name="valley", dataset_root=tmp_path / "ds",
        dataset_id="ds-1", project_root=tmp_path, built_by="build_plant_mapping",
        plant_registry={"name": "unregistered", "digest": "0" * 64},
        nn_tolerance_m=nn_tolerance_m,
    )
    return build, grid_pitch_m(read_plant_csvs([plant_csv]))


def test_derives_from_grid_pitch_when_unstated(tmp_path: Path) -> None:
    build, pitch = _build(tmp_path, PLANTS, None)
    assert pitch > 0
    assert build.nn_tolerance_m["source"] == "grid_pitch"
    assert build.nn_tolerance_m["value"] == pytest.approx(pitch / 6)


def test_falls_back_when_fewer_than_two_plants_have_positions(tmp_path: Path) -> None:
    build, pitch = _build(tmp_path, PLANTS[:1], None)
    assert pitch == 0.0
    assert build.nn_tolerance_m["source"] == "fallback"
    assert build.nn_tolerance_m["value"] == pytest.approx(NN_TOLERANCE_METERS)


def test_honors_a_stated_value_at_or_under_the_cap(tmp_path: Path) -> None:
    build, pitch = _build(tmp_path, PLANTS, 0.1)
    assert pitch / 6 > 0.1
    assert build.nn_tolerance_m["source"] == "stated"
    assert build.nn_tolerance_m["value"] == pytest.approx(0.1)


def test_caps_a_stated_value_above_the_grid_pitch_ceiling(tmp_path: Path) -> None:
    build, pitch = _build(tmp_path, PLANTS, 5.0)
    cap = pitch / 6
    assert 5.0 > cap
    assert build.nn_tolerance_m["source"] == "stated_capped"
    assert build.nn_tolerance_m["value"] == pytest.approx(cap)
