"""``shp_to_plant_csv``: a shapefile converted to a ``read_plant_csvs`` CSV.

The conversion keeps correct WGS84 axis order and handles point and polygon geometry.

EPSG:4326's authority-declared axis order is (lat, lon). These tests build real synthetic
shapefiles in a real projected CRS (UTM 15N, via fiona, no binary fixture committed) and check
the output against an independently-computed transform.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tcip_mcp.pipelines.postprocessing.plant_mapping import ShapefileField
from tcip_mcp.tools.phenology_tools import (
    _validate_round_trip,
    shp_to_plant_csv,
)
from tests import csv_rows


def _convert(shp: Path, csv_path: Path, project: Path) -> int:
    """Run the conversion command from ``shp`` to ``csv_path`` for ``project``; return its exit."""
    from tcip_web.cli import main

    return main(
        [
            *("shp-to-plant-csv", "--shp-path", str(shp)),
            *("--csv-path", str(csv_path), "--project", str(project)),
            *("--user", "tester"),
        ]
    )


UTM_15N_EPSG = 32615

# A single UTM point whose WGS84 equivalent is independently hand-verifiable: 500 000 mE is UTM's
# own false easting (the central meridian, -93 degrees exactly) for every northern zone.
POINT_NATIVE = (500_000.0, 4_649_776.0)


def _reference_lonlat(native_x: float, native_y: float) -> tuple[float, float]:
    import pyproj

    transformer = pyproj.Transformer.from_crs(
        f"EPSG:{UTM_15N_EPSG}", "EPSG:4326", always_xy=True
    )
    return transformer.transform(native_x, native_y)


_SCHEMA_PROPERTIES = {
    "plant_id": "str",
    "accession_name": "str",
    "plot_number": "float",
    "row_number": "float",
    "col_number": "float",
}

# The ESRI Shapefile driver truncates a field name over 10 characters when it writes the DBF, so a
# fixture's properties are keyed by the same truncated form the script's own resolver looks for.
_ATTRS_POINT = {
    "plant_id": "P1",
    "accession_name": "acc-A",
    "plot_number": 1.0,
    "row_number": 2.0,
    "col_number": 3.0,
}
_ATTRS_POLYGON = {
    "plant_id": "P2",
    "accession_name": "acc-B",
    "plot_number": 4.0,
    "row_number": 5.0,
    "col_number": 6.0,
}


def _write_shapefile(
    path: Path, geom_type: str, geometry: dict, attrs: dict, epsg: int | None
) -> Path:
    import fiona

    schema = {"geometry": geom_type, "properties": dict(_SCHEMA_PROPERTIES)}
    crs = f"EPSG:{epsg}" if epsg is not None else None
    with fiona.open(
        str(path), "w", driver="ESRI Shapefile", crs=crs, schema=schema
    ) as dst:
        dst.write({"geometry": geometry, "properties": dict(attrs)})
    return path


def _point_shapefile(
    tmp_path: Path, *, epsg: int | None = UTM_15N_EPSG
) -> Path:
    return _write_shapefile(
        tmp_path / "points.shp",
        "Point",
        {"type": "Point", "coordinates": tuple(POINT_NATIVE)},
        _ATTRS_POINT,
        epsg,
    )


def _polygon_shapefile(tmp_path: Path) -> Path:
    """Write ``tmp_path / "polys.shp"`` in UTM 15N holding one polygon; return its path.

    The polygon's ring runs through ``POINT_NATIVE`` plus and minus 10 on each axis, and its
    attributes are ``_ATTRS_POLYGON``.
    """
    cx, cy = POINT_NATIVE
    half = 10.0
    ring = [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
        (cx - half, cy - half),
    ]
    return _write_shapefile(
        tmp_path / "polys.shp",
        "Polygon",
        {"type": "Polygon", "coordinates": [ring]},
        _ATTRS_POLYGON,
        UTM_15N_EPSG,
    )


# ── axis order + point geometry ─────────────────────────────────────────


def test_point_geometry_reprojects_to_correct_unswapped_wgs84(
    tmp_path: Path,
) -> None:
    shp = _point_shapefile(tmp_path)
    csv_path = tmp_path / "plants.csv"

    result = shp_to_plant_csv(tmp_path, str(shp), str(csv_path), actor=None)

    assert result["geometry_kinds"] == ["point"]
    rows = csv_rows(csv_path)
    assert len(rows) == 1
    lon, lat = (
        float(rows[0]["WGS84_centroid_x"]),
        float(rows[0]["WGS84_centroid_y"]),
    )

    # Not swapped: UTM zone 15N sits in the western hemisphere, well north of the equator.
    assert -100 < lon < -85
    assert 35 < lat < 50

    # Matches an independently-constructed transform to float precision, not just a plausible
    # range.
    expected_lon, expected_lat = _reference_lonlat(*POINT_NATIVE)
    assert lon == pytest.approx(expected_lon, abs=1e-6)
    assert lat == pytest.approx(expected_lat, abs=1e-6)

    assert rows[0]["plant_id"] == "P1"
    assert rows[0]["accession_name"] == "acc-A"
    assert rows[0]["plot_number"] == "1.0"


def test_swapped_axis_order_would_fail_this_test(tmp_path: Path) -> None:
    """Without ``always_xy``, pyproj's UTM 15N transform answers latitude first.

    With ``always_xy`` omitted, the default (authority-compliant, lat/lon) axis order puts the
    first coordinate in latitude's range for UTM 15N, outside longitude's.
    """
    import pyproj

    # Deliberately no always_xy=True, the parameter the script applies.
    transform = pyproj.Transformer.from_crs(
        f"EPSG:{UTM_15N_EPSG}", "EPSG:4326"
    )
    a, b = transform.transform(*POINT_NATIVE)

    # With no axis correction pyproj returns (lat, lon): the first coordinate lands in latitude's
    # plausible range, not longitude's, i.e. exactly the swap ``always_xy=True`` prevents.
    assert 35 < a < 50
    assert not (-100 < a < -85)


# ── polygon geometry (centroid) ─────────────────────────────────────────


def test_polygon_geometry_uses_centroid(tmp_path: Path) -> None:
    shp = _polygon_shapefile(tmp_path)
    csv_path = tmp_path / "plants.csv"

    result = shp_to_plant_csv(tmp_path, str(shp), str(csv_path), actor=None)

    assert result["geometry_kinds"] == ["polygon"]
    rows = csv_rows(csv_path)
    lon, lat = (
        float(rows[0]["WGS84_centroid_x"]),
        float(rows[0]["WGS84_centroid_y"]),
    )
    expected_lon, expected_lat = _reference_lonlat(*POINT_NATIVE)
    assert lon == pytest.approx(expected_lon, abs=1e-6)
    assert lat == pytest.approx(expected_lat, abs=1e-6)


# ── CRS refusal ──────────────────────────────────────────────────────────


def test_refuses_a_shapefile_with_no_resolvable_crs(tmp_path: Path) -> None:
    shp = _point_shapefile(tmp_path, epsg=None)
    csv_path = tmp_path / "plants.csv"

    result = shp_to_plant_csv(tmp_path, str(shp), str(csv_path), actor=None)

    assert "resolvable coordinate reference system" in result["error"]
    assert not csv_path.exists()


# ── round-trip validation ───────────────────────────────────────────────


def test_round_trip_validation_names_the_real_cause_on_a_broken_header(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "broken.csv"
    with broken.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["plant_id", "accession_name", "lon", "lat"]
        )  # wrong column names
        w.writerow(["P1", "acc-A", "-93.0", "42.0"])

    with pytest.raises(ValueError, match="parsed 0 plant records"):
        _validate_round_trip(broken, n_written=1)


def test_round_trip_validation_passes_on_a_well_formed_csv(
    tmp_path: Path,
) -> None:
    from tests._mapping_fixtures import write_plant_csv

    good = write_plant_csv(
        tmp_path / "good.csv",
        [{"plot": "P1", "accession": "acc-A", "lat": 42.0, "lon": -93.0}],
    )

    assert _validate_round_trip(good, n_written=1) == 1


# ── field-name overrides for a source's own (possibly truncated) attribute names ────────────────


def test_missing_optional_field_is_reported_and_written_empty(
    tmp_path: Path,
) -> None:
    shp = _point_shapefile(tmp_path)
    csv_path = tmp_path / "plants.csv"

    result = shp_to_plant_csv(
        tmp_path,
        str(shp),
        str(csv_path),
        field_map={ShapefileField("accession_name"): "no_such_field"},
        actor=None,
    )

    assert "accession_name" in result["missing_fields"]
    rows = csv_rows(csv_path)
    assert rows[0]["accession_name"] == ""


def test_field_map_override_picks_up_a_renamed_source_field(
    tmp_path: Path,
) -> None:
    import fiona

    path = tmp_path / "renamed.shp"
    # ESRI Shapefile DBF field names are capped at 10 characters: a real source's "accession_name"
    # is commonly truncated like this.
    schema = {
        "geometry": "Point",
        "properties": {"plant_id": "str", "ACC_NAME": "str"},
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
                "geometry": {
                    "type": "Point",
                    "coordinates": tuple(POINT_NATIVE),
                },
                "properties": {"plant_id": "P1", "ACC_NAME": "acc-Z"},
            }
        )

    csv_path = tmp_path / "plants.csv"
    result = shp_to_plant_csv(
        tmp_path,
        str(path),
        str(csv_path),
        field_map={ShapefileField("accession_name"): "ACC_NAME"},
        actor=None,
    )

    assert "accession_name" not in result["missing_fields"]
    rows = csv_rows(csv_path)
    assert rows[0]["accession_name"] == "acc-Z"


# ── zero-feature refusal ─────────────────────────────────────────────────


def test_refuses_a_shapefile_with_zero_features(tmp_path: Path) -> None:
    import fiona

    path = tmp_path / "empty.shp"
    schema = {"geometry": "Point", "properties": dict(_SCHEMA_PROPERTIES)}
    with fiona.open(
        str(path),
        "w",
        driver="ESRI Shapefile",
        crs=f"EPSG:{UTM_15N_EPSG}",
        schema=schema,
    ):
        pass

    result = shp_to_plant_csv(tmp_path, str(path), "plants.csv", actor=None)

    assert "zero features" in result["error"]
    assert not (tmp_path / "plants.csv").exists()


# ── CLI end to end ───────────────────────────────────────────────────────


def test_main_cli_end_to_end(tmp_path: Path, project: Path) -> None:
    shp = _point_shapefile(tmp_path)
    csv_path = tmp_path / "plants.csv"

    rc = _convert(shp, csv_path, project)

    assert rc == 0
    assert csv_path.is_file()
    rows = csv_rows(csv_path)
    assert len(rows) == 1


def test_main_cli_prints_the_refusal_and_writes_nothing(
    tmp_path: Path, project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    shp = _point_shapefile(tmp_path, epsg=None)
    csv_path = tmp_path / "plants.csv"

    rc = _convert(shp, csv_path, project)

    assert rc == 0
    refusal = json.loads(capsys.readouterr().out)["error"]
    assert "resolvable coordinate reference system" in refusal
    assert not csv_path.exists()


# ── read_plant_shapefile itself ──────────────────────────────────────────


def _alnum_point_shapefile(tmp_path: Path) -> Path:
    """Write a point shapefile whose ``plot_number`` is a non-numeric string field."""
    import fiona

    path = tmp_path / "alnum.shp"
    schema = {
        "geometry": "Point",
        "properties": {
            "plant_id": "str",
            "accession_name": "str",
            "plot_number": "str",
        },
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
                "geometry": {
                    "type": "Point",
                    "coordinates": tuple(POINT_NATIVE),
                },
                "properties": {
                    "plant_id": "P9",
                    "accession_name": "acc-Z",
                    "plot_number": "A1",
                },
            }
        )
    return path


def test_read_plant_shapefile_point_rows_match_reference_and_attrs_verbatim(
    tmp_path: Path,
) -> None:
    """A point feature reads its own coordinate, reprojected to WGS84.

    It carries its DBF attributes through as the values' own strings.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        read_plant_shapefile,
    )

    result = read_plant_shapefile(_point_shapefile(tmp_path))

    assert result.geometry_kinds == ["point"]
    assert result.skipped_null_geometry == 0
    row = result.rows[0]
    expected_lon, expected_lat = _reference_lonlat(*POINT_NATIVE)
    assert row["WGS84_centroid_x"] == pytest.approx(expected_lon, abs=1e-6)
    assert row["WGS84_centroid_y"] == pytest.approx(expected_lat, abs=1e-6)
    assert row["plant_id"] == "P1"
    assert row["accession_name"] == "acc-A"
    assert row["plot_number"] == "1.0"


def test_read_plant_shapefile_polygon_rows_match_reference_and_attrs_verbatim(
    tmp_path: Path,
) -> None:
    """A polygon feature reads its centroid, reprojected to WGS84.

    It is reported under its own geometry kind.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        read_plant_shapefile,
    )

    result = read_plant_shapefile(_polygon_shapefile(tmp_path))

    assert result.geometry_kinds == ["polygon"]
    row = result.rows[0]
    expected_lon, expected_lat = _reference_lonlat(*POINT_NATIVE)
    assert row["WGS84_centroid_x"] == pytest.approx(expected_lon, abs=1e-6)
    assert row["WGS84_centroid_y"] == pytest.approx(expected_lat, abs=1e-6)
    assert row["plant_id"] == "P2"
    assert row["accession_name"] == "acc-B"


def test_read_plant_shapefile_non_numeric_plot_number_carried_verbatim(
    tmp_path: Path,
) -> None:
    """A plot number held as a string field survives the reader verbatim."""
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        read_plant_shapefile,
    )

    result = read_plant_shapefile(_alnum_point_shapefile(tmp_path))
    assert result.rows[0]["plot_number"] == "A1"


def test_read_plant_shapefile_refuses_a_crs_it_cannot_resolve(
    tmp_path: Path,
) -> None:
    """A shapefile with no .prj is refused for naming no resolvable coordinate reference system."""
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        ShapefileCrsUnknownError,
        read_plant_shapefile,
    )

    shp = _point_shapefile(tmp_path, epsg=None)
    with pytest.raises(
        ShapefileCrsUnknownError,
        match="resolvable coordinate reference system",
    ):
        read_plant_shapefile(shp)


# ── a .prj that exists but does not parse ────────────────────────────────


def _garbage_prj_shapefile(tmp_path: Path) -> Path:
    """Write a ``.shp`` with a valid geometry beside a ``.prj`` holding unparseable WKT."""
    shp = _point_shapefile(tmp_path, epsg=UTM_15N_EPSG)
    (tmp_path / "points.prj").write_text(
        "this is not valid WKT at all !!! ###"
    )
    return shp


def test_garbage_prj_refuses_through_the_conversion(tmp_path: Path) -> None:
    """A .prj holding unparseable text is refused through the missing-CRS check.

    The conversion answers the CRS refusal and writes no CSV.
    """
    shp = _garbage_prj_shapefile(tmp_path)
    csv_path = tmp_path / "plants.csv"

    result = shp_to_plant_csv(tmp_path, str(shp), str(csv_path), actor=None)

    assert "resolvable coordinate reference system" in result["error"]
    assert not csv_path.exists()


# ── a geometry type this reader does not centroid or read a point from ──


def _line_string_shapefile(tmp_path: Path) -> Path:
    import fiona

    path = tmp_path / "lines.shp"
    schema = {"geometry": "LineString", "properties": dict(_SCHEMA_PROPERTIES)}
    with fiona.open(
        str(path),
        "w",
        driver="ESRI Shapefile",
        crs=f"EPSG:{UTM_15N_EPSG}",
        schema=schema,
    ) as dst:
        dst.write(
            {
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        tuple(POINT_NATIVE),
                        (POINT_NATIVE[0] + 10, POINT_NATIVE[1] + 10),
                    ],
                },
                "properties": dict(_ATTRS_POINT),
            }
        )
    return path


def test_line_string_geometry_refuses_named_through_the_conversion(
    tmp_path: Path,
) -> None:
    """A line string geometry is refused, naming its type."""
    shp = _line_string_shapefile(tmp_path)

    result = shp_to_plant_csv(tmp_path, str(shp), "plants.csv", actor=None)

    assert "LineString" in result["error"]


# ── null-geometry features ───────────────────────────────────────────────


def _point_shapefile_with_null_feature(tmp_path: Path) -> Path:
    import fiona

    path = tmp_path / "with_null.shp"
    schema = {"geometry": "Point", "properties": dict(_SCHEMA_PROPERTIES)}
    with fiona.open(
        str(path),
        "w",
        driver="ESRI Shapefile",
        crs=f"EPSG:{UTM_15N_EPSG}",
        schema=schema,
    ) as dst:
        dst.write(
            {
                "geometry": {
                    "type": "Point",
                    "coordinates": tuple(POINT_NATIVE),
                },
                "properties": dict(_ATTRS_POINT),
            }
        )
        dst.write({"geometry": None, "properties": dict(_ATTRS_POLYGON)})
    return path


def test_null_geometry_feature_is_skipped_and_counted(tmp_path: Path) -> None:
    """A feature with no geometry is skipped and counted, with no raise."""
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        read_plant_shapefile,
    )

    result = read_plant_shapefile(_point_shapefile_with_null_feature(tmp_path))
    assert len(result.rows) == 1
    assert result.skipped_null_geometry == 1


def test_the_conversion_reports_skipped_null_geometry(tmp_path: Path) -> None:
    """The conversion reports the reader's skipped-null count beside its own row count."""
    shp = _point_shapefile_with_null_feature(tmp_path)
    result = shp_to_plant_csv(tmp_path, str(shp), "plants.csv", actor=None)
    assert result["n_features"] == 1
    assert result["skipped_null_geometry"] == 1


# ── PLANT_CSV_COLUMNS: one schema both sides own ─────────────────────────


def test_plant_csv_columns_matches_written_header(tmp_path: Path) -> None:
    """The written header equals ``PLANT_CSV_COLUMNS``, the values of its field mapping."""
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        PLANT_CSV_COLUMN_FOR_FIELD,
        PLANT_CSV_COLUMNS,
    )

    csv_path = tmp_path / "plants.csv"
    shp_to_plant_csv(
        tmp_path, str(_point_shapefile(tmp_path)), str(csv_path), actor=None
    )
    with csv_path.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))

    assert header == PLANT_CSV_COLUMNS
    assert PLANT_CSV_COLUMNS == list(PLANT_CSV_COLUMN_FOR_FIELD.values())


def test_every_written_column_survives_read_plant_csv_bytes(
    tmp_path: Path,
) -> None:
    """A written CSV reads back through ``read_plant_csv_bytes`` with id, accession and point."""
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        read_plant_csv_bytes,
    )

    csv_path = tmp_path / "plants.csv"
    shp_to_plant_csv(
        tmp_path,
        str(_alnum_point_shapefile(tmp_path)),
        str(csv_path),
        actor=None,
    )

    records = read_plant_csv_bytes(csv_path.read_bytes())

    assert len(records) == 1
    record = records[0]
    assert record.plant_id == "P9"
    assert record.accession_name == "acc-Z"
    expected_lon, expected_lat = _reference_lonlat(*POINT_NATIVE)
    assert record.lon == pytest.approx(expected_lon, abs=1e-6)
    assert record.lat == pytest.approx(expected_lat, abs=1e-6)


def test_converter_csv_carries_non_numeric_plot_number_verbatim(
    tmp_path: Path,
) -> None:
    """The written CSV carries a non-numeric plot number through unchanged."""
    csv_path = tmp_path / "plants.csv"
    shp_to_plant_csv(
        tmp_path,
        str(_alnum_point_shapefile(tmp_path)),
        str(csv_path),
        actor=None,
    )
    rows = csv_rows(csv_path)
    assert rows[0]["plot_number"] == "A1"
