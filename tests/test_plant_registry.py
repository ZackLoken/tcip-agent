"""Coverage for ``register_plant_registry`` (additions-design section 1): a named,
project-scoped record of a plant-locations CSV set, read back by ``build_plant_mapping`` and
``deliver_orthomosaic_plant_counts`` in place of a re-asserted path list on every call.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp.pipelines.postprocessing import plant_mapping
from tcip_mcp.tools.phenology_tools import build_plant_mapping, register_plant_registry

from tests.test_plant_mapping_binding import PLANTS, _dataset, _init, _write_scene


def _plant_csv(path: Path, plants: list[dict] | None = None) -> Path:
    plants = PLANTS if plants is None else plants
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["plot_name", "accession_name", "WGS84_centroid_x", "WGS84_centroid_y"])
        for p in plants:
            w.writerow([p["plot"], p["accession"], p["lon"], p["lat"]])
    return path


def test_registers_a_csv_and_persists_the_frozen_record(tmp_path: Path) -> None:
    """The recorded shape: per-file {path, sha256, n_plants}, crop, site, registered_by,
    registered_at, and a digest over the parsed rows."""
    csv_path = _plant_csv(tmp_path / "plants.csv")

    res = register_plant_registry(
        name="valley-plants", csv_paths=[str(csv_path)], crop="hazelnut", site="north orchard")

    assert "error" not in res, res
    assert res["name"] == "valley-plants"
    assert res["n_plants"] == len(PLANTS)
    assert res["registered_by"] == "agent:register_plant_registry"
    assert len(res["digest"]) == 64
    assert res["csvs"] == [{
        "path": str(csv_path),
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "n_plants": len(PLANTS),
    }]

    stored = plant_mapping.load_registry(tmp_path, "valley-plants")
    assert stored is not None
    assert stored["digest"] == res["digest"]
    assert stored["csvs"] == res["csvs"]


def test_refuses_a_name_outside_name_segment(tmp_path: Path) -> None:
    csv_path = _plant_csv(tmp_path / "plants.csv")

    res = register_plant_registry(
        name="Not Legal!", csv_paths=[str(csv_path)], crop="hazelnut", site="orchard")

    assert "error" in res
    assert "lowercase" in res["error"]


def test_refuses_a_missing_file(tmp_path: Path) -> None:
    res = register_plant_registry(
        name="valley", csv_paths=[str(tmp_path / "missing.csv")], crop="hazelnut", site="orchard")

    assert "error" in res
    assert "plant CSV" in res["error"]


def test_refuses_naming_a_file_that_parses_no_georeferenced_plant(tmp_path: Path) -> None:
    """A per-file check: one good file beside one that parses to nothing still refuses, naming
    only the failing file."""
    good = _plant_csv(tmp_path / "good.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("plot_name,accession_name,WGS84_centroid_x,WGS84_centroid_y\n", encoding="utf-8")

    res = register_plant_registry(
        name="valley", csv_paths=[str(good), str(bad)], crop="hazelnut", site="orchard")

    assert "error" in res
    assert bad.name in res["error"]
    assert good.name not in res["error"]
    assert not ts.exists(plant_mapping.plant_registry_key(tmp_path, "valley"))


def test_a_second_registration_under_the_same_name_and_content_is_a_no_op(tmp_path: Path) -> None:
    csv_path = _plant_csv(tmp_path / "plants.csv")

    first = register_plant_registry(
        name="valley", csv_paths=[str(csv_path)], crop="hazelnut", site="orchard")
    assert "error" not in first, first

    second = register_plant_registry(
        name="valley", csv_paths=[str(csv_path)], crop="hazelnut", site="orchard")

    assert "error" not in second, second
    assert second["digest"] == first["digest"]


def test_a_second_registration_under_the_same_name_and_different_plants_refuses(
    tmp_path: Path,
) -> None:
    csv_path = _plant_csv(tmp_path / "plants.csv")
    other_csv = _plant_csv(
        tmp_path / "other.csv", [{"plot": "P9", "accession": "acc-Z", "lat": 1.0, "lon": 1.0}])

    first = register_plant_registry(
        name="valley", csv_paths=[str(csv_path)], crop="hazelnut", site="orchard")
    assert "error" not in first, first

    conflict = register_plant_registry(
        name="valley", csv_paths=[str(other_csv)], crop="hazelnut", site="orchard")

    assert "error" in conflict
    assert first["digest"] in conflict["error"]
    stored = plant_mapping.load_registry(tmp_path, "valley")
    assert stored["digest"] == first["digest"], "the taken name's record must not have moved"


def test_build_plant_mapping_reads_a_registered_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admits valid work: a mapping built by build_plant_mapping over a name
    register_plant_registry minted, through both platform doors."""
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root, dates=["2026-02-11"])

    reg = register_plant_registry(
        name="valley-plants", csv_paths=[str(plant_csv)], crop="hazelnut", site="orchard")
    assert "error" not in reg, reg

    res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_registry="valley-plants")

    assert "error" not in res, res
    build = plant_mapping.load_mapping(tmp_path, "valley")
    assert build is not None
    assert build.plant_registry == {"name": "valley-plants", "digest": reg["digest"]}


def test_build_plant_mapping_refuses_naming_register_plant_registry_when_registry_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, _, _ = _write_scene(dataset_root, dates=["2026-02-11"])

    res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_registry="nope")

    assert "error" in res
    assert "register_plant_registry" in res["error"]


