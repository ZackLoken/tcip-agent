"""register_plant_registry refuses a shapefile or binary file, naming the conversion tool.

It refuses a shapefile's own parts by suffix, and a binary file under any other suffix (a
``.csv`` included), as a returned error. A shapefile converted first, then registered, registers
its plants.
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
    schema = {
        "geometry": "Point",
        "properties": {"plant_id": "str", "accession_name": "str"},
    }
    with fiona.open(
        str(path),
        "w",
        driver="ESRI Shapefile",
        crs=f"EPSG:{UTM_15N_EPSG}",
        schema=schema,
    ) as dst:
        dst.write(
            {
                "geometry": {"type": "Point", "coordinates": POINT_NATIVE},
                "properties": {"plant_id": "P1", "accession_name": "acc-A"},
            }
        )
        dst.write(
            {
                "geometry": {
                    "type": "Point",
                    "coordinates": (POINT_NATIVE[0] + 5, POINT_NATIVE[1]),
                },
                "properties": {"plant_id": "P2", "accession_name": "acc-B"},
            }
        )
    return path


def test_register_plant_registry_refuses_a_shp_naming_the_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shp = _point_shapefile(tmp_path)

    res = register_plant_registry(
        tmp_path,
        name="reg",
        csv_paths=[str(shp)],
        crop="black locust",
        site="block",
        actor=None,
    )

    match res:
        case {"error": str() as error}:
            assert "shp_to_plant_csv" in error
        case _:
            raise AssertionError(res)


def test_register_plant_registry_refuses_a_binary_csv_as_not_utf8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary_path = tmp_path / "plants.csv"
    binary_path.write_bytes(bytes(range(256)))

    res = register_plant_registry(
        tmp_path,
        name="reg",
        csv_paths=[str(binary_path)],
        crop="black locust",
        site="block",
        actor=None,
    )

    match res:
        case {"error": str() as error}:
            assert "not UTF-8 text" in error
        case _:
            raise AssertionError(res)


def test_register_plant_registry_admits_a_shapefile_converted_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shapefile converted first through the platform's own tool registers its plants.

    The CSV the conversion writes registers with one plant per converted feature.
    """
    from tcip_mcp.tools.phenology_tools import shp_to_plant_csv

    shp = _point_shapefile(tmp_path)
    csv_path = tmp_path / "plants.csv"
    converted = shp_to_plant_csv(tmp_path, str(shp), str(csv_path), actor=None)

    res = register_plant_registry(
        tmp_path,
        name="reg",
        csv_paths=[str(csv_path)],
        crop="black locust",
        site="block",
        actor=None,
    )

    assert "error" not in res, res
    assert res["n_plants"] == converted["n_features"] == 2
