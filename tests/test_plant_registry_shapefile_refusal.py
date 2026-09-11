"""register_plant_registry refuses a shapefile's own parts by suffix, naming the conversion
command, rather than letting an unguarded UnicodeDecodeError escape the door as an exception; and
refuses a binary file under any other suffix (a ``.csv`` included) the same way. The admits-valid-
work case: a shapefile converted first, then registered, answers the same record a hand-authored
CSV would.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp.tools.phenology_tools import register_plant_registry

UTM_15N_EPSG = 32615
POINT_NATIVE = (500_000.0, 4_649_776.0)


def _point_shapefile(tmp_path: Path) -> Path:
    import fiona

    path = tmp_path / "plants.shp"
    schema = {"geometry": "Point", "properties": {"plot_name": "str", "accession_name": "str"}}
    with fiona.open(str(path), "w", driver="ESRI Shapefile", crs=f"EPSG:{UTM_15N_EPSG}",
                    schema=schema) as dst:
        dst.write({"geometry": {"type": "Point", "coordinates": POINT_NATIVE},
                   "properties": {"plot_name": "P1", "accession_name": "acc-A"}})
        dst.write({"geometry": {"type": "Point", "coordinates": (POINT_NATIVE[0] + 5, POINT_NATIVE[1])},
                   "properties": {"plot_name": "P2", "accession_name": "acc-B"}})
    return path


def test_register_plant_registry_refuses_a_shp_naming_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    shp = _point_shapefile(tmp_path)

    res = register_plant_registry(
        name="reg", csv_paths=[str(shp)], crop="black locust", site="block")

    assert "error" in res
    assert "tcip shp-to-plant-csv" in res["error"]


def test_register_plant_registry_refuses_a_binary_csv_as_not_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    binary_path = tmp_path / "plants.csv"
    binary_path.write_bytes(bytes(range(256)))

    res = register_plant_registry(
        name="reg", csv_paths=[str(binary_path)], crop="black locust", site="block")

    assert "error" in res
    assert "not UTF-8 text" in res["error"]


def test_register_plant_registry_admits_a_shapefile_converted_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shapefile is never registered directly; converted through the platform's own command
    first, the resulting CSV registers exactly as a hand-authored one would."""
    from tcip_mcp.cli.shp_to_plant_csv import convert_shp_to_plant_csv

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    shp = _point_shapefile(tmp_path)
    csv_path = tmp_path / "plants.csv"
    converted = convert_shp_to_plant_csv(shp, csv_path)

    res = register_plant_registry(
        name="reg", csv_paths=[str(csv_path)], crop="black locust", site="block")

    assert "error" not in res, res
    assert res["n_plants"] == converted["n_features"] == 2
