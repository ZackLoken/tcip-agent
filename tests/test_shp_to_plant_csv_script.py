"""``tcip shp-to-plant-csv``: shapefile -> ``read_plant_csvs`` CSV, with correct WGS84 axis
order and point/polygon geometry handling.

EPSG:4326's authority-declared axis order is (lat, lon); a converter that reprojects without
``always_xy`` silently writes latitude into the ``WGS84_centroid_x`` (longitude) column. These
tests build real synthetic shapefiles in a real projected CRS (UTM 15N, via fiona, no binary
fixture committed) and check the output against an independently-computed transform, not just a
plausible range.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tcip_mcp.cli.shp_to_plant_csv import _validate_round_trip, convert_shp_to_plant_csv, main

UTM_15N_EPSG = 32615

# A single UTM point whose WGS84 equivalent is independently hand-verifiable: 500 000 mE is UTM's
# own false easting (the central meridian, -93 degrees exactly) for every northern zone.
POINT_NATIVE = (500_000.0, 4_649_776.0)


def _reference_lonlat(native_x: float, native_y: float) -> tuple[float, float]:
    import pyproj

    transformer = pyproj.Transformer.from_crs(f"EPSG:{UTM_15N_EPSG}", "EPSG:4326", always_xy=True)
    return transformer.transform(native_x, native_y)


_SCHEMA_PROPERTIES = {
    "plot_name": "str", "accession_name": "str",
    "plot_number": "float", "row_number": "float", "col_number": "float",
}

# The ESRI Shapefile driver truncates a field name over 10 characters when it writes the DBF, so a
# fixture's properties are keyed by the same truncated form the script's own resolver looks for.
_ATTRS_POINT = {"plot_name": "P1", "accession_name": "acc-A",
                "plot_number": 1.0, "row_number": 2.0, "col_number": 3.0}
_ATTRS_POLYGON = {"plot_name": "P2", "accession_name": "acc-B",
                  "plot_number": 4.0, "row_number": 5.0, "col_number": 6.0}


def _write_shapefile(path: Path, geom_type: str, geometry: dict, attrs: dict,
                     epsg: int | None) -> Path:
    import fiona

    schema = {"geometry": geom_type, "properties": dict(_SCHEMA_PROPERTIES)}
    crs = f"EPSG:{epsg}" if epsg is not None else None
    with fiona.open(str(path), "w", driver="ESRI Shapefile", crs=crs, schema=schema) as dst:
        dst.write({"geometry": geometry, "properties": dict(attrs)})
    return path


def _point_shapefile(tmp_path: Path, *, epsg: int | None = UTM_15N_EPSG) -> Path:
    return _write_shapefile(
        tmp_path / "points.shp", "Point",
        {"type": "Point", "coordinates": tuple(POINT_NATIVE)}, _ATTRS_POINT, epsg,
    )


def _polygon_shapefile(tmp_path: Path) -> Path:
    """A square centered exactly on ``POINT_NATIVE``, so its centroid is the center by
    construction, not something this test would need to recompute geometrically."""
    cx, cy = POINT_NATIVE
    half = 10.0
    ring = [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half),
            (cx - half, cy + half), (cx - half, cy - half)]
    return _write_shapefile(
        tmp_path / "polys.shp", "Polygon",
        {"type": "Polygon", "coordinates": [ring]}, _ATTRS_POLYGON, UTM_15N_EPSG,
    )


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── axis order + point geometry ─────────────────────────────────────────


def test_point_geometry_reprojects_to_correct_unswapped_wgs84(tmp_path: Path) -> None:
    shp = _point_shapefile(tmp_path)
    csv_path = tmp_path / "plants.csv"

    result = convert_shp_to_plant_csv(shp, csv_path)

    assert result["geometry_kinds"] == ["point"]
    rows = _read_csv_rows(csv_path)
    assert len(rows) == 1
    lon, lat = float(rows[0]["WGS84_centroid_x"]), float(rows[0]["WGS84_centroid_y"])

    # Not swapped: UTM zone 15N sits in the western hemisphere, well north of the equator.
    assert -100 < lon < -85
    assert 35 < lat < 50

    # Matches an independently-constructed transform to float precision, not just a plausible
    # range.
    expected_lon, expected_lat = _reference_lonlat(*POINT_NATIVE)
    assert lon == pytest.approx(expected_lon, abs=1e-6)
    assert lat == pytest.approx(expected_lat, abs=1e-6)

    assert rows[0]["plot_name"] == "P1"
    assert rows[0]["accession_name"] == "acc-A"
    assert rows[0]["plot_number"] == "1.0"


def test_swapped_axis_order_would_fail_this_test(tmp_path: Path) -> None:
    """Pins the failure mode this converter guards against: transforming with the default
    (authority-compliant, lat/lon) axis order instead of ``always_xy`` lands outside a plausible
    longitude range for UTM 15N. Exercises the same pyproj call the script makes, with
    ``always_xy=True`` deliberately omitted, so a regression that drops it would be caught by
    the assertion above failing, not by this test passing either way."""
    import pyproj

    # Deliberately no always_xy=True, the parameter the script applies.
    transform = pyproj.Transformer.from_crs(f"EPSG:{UTM_15N_EPSG}", "EPSG:4326")
    a, b = transform.transform(*POINT_NATIVE)

    # With no axis correction pyproj returns (lat, lon): the first coordinate lands in latitude's
    # plausible range, not longitude's, i.e. exactly the swap ``always_xy=True`` prevents.
    assert 35 < a < 50
    assert not (-100 < a < -85)


# ── polygon geometry (centroid) ─────────────────────────────────────────


def test_polygon_geometry_uses_centroid(tmp_path: Path) -> None:
    shp = _polygon_shapefile(tmp_path)
    csv_path = tmp_path / "plants.csv"

    result = convert_shp_to_plant_csv(shp, csv_path)

    assert result["geometry_kinds"] == ["polygon"]
    rows = _read_csv_rows(csv_path)
    lon, lat = float(rows[0]["WGS84_centroid_x"]), float(rows[0]["WGS84_centroid_y"])
    expected_lon, expected_lat = _reference_lonlat(*POINT_NATIVE)
    assert lon == pytest.approx(expected_lon, abs=1e-6)
    assert lat == pytest.approx(expected_lat, abs=1e-6)


# ── CRS refusal ──────────────────────────────────────────────────────────


def test_refuses_a_shapefile_with_no_resolvable_crs(tmp_path: Path) -> None:
    shp = _point_shapefile(tmp_path, epsg=None)
    csv_path = tmp_path / "plants.csv"

    with pytest.raises(ValueError, match="resolvable coordinate reference system"):
        convert_shp_to_plant_csv(shp, csv_path)

    assert not csv_path.exists()


# ── round-trip validation ───────────────────────────────────────────────


def test_round_trip_validation_names_the_real_cause_on_a_broken_header(tmp_path: Path) -> None:
    broken = tmp_path / "broken.csv"
    with broken.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["plot_name", "accession_name", "lon", "lat"])  # wrong column names
        w.writerow(["P1", "acc-A", "-93.0", "42.0"])

    with pytest.raises(ValueError, match="parsed 0 plant records"):
        _validate_round_trip(broken, n_written=1)


def test_round_trip_validation_passes_on_a_well_formed_csv(tmp_path: Path) -> None:
    good = tmp_path / "good.csv"
    with good.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["plot_name", "accession_name", "WGS84_centroid_x", "WGS84_centroid_y"])
        w.writerow(["P1", "acc-A", "-93.0", "42.0"])

    assert _validate_round_trip(good, n_written=1) == 1


# ── field-name overrides for a source's own (possibly truncated) attribute names ────────────────


def test_missing_optional_field_is_reported_and_written_empty(tmp_path: Path) -> None:
    shp = _point_shapefile(tmp_path)
    csv_path = tmp_path / "plants.csv"

    result = convert_shp_to_plant_csv(shp, csv_path, field_map={"accession_name": "no_such_field"})

    assert "accession_name" in result["missing_fields"]
    rows = _read_csv_rows(csv_path)
    assert rows[0]["accession_name"] == ""


def test_field_map_override_picks_up_a_renamed_source_field(tmp_path: Path) -> None:
    import fiona

    path = tmp_path / "renamed.shp"
    # ESRI Shapefile DBF field names are capped at 10 characters: a real source's "accession_name"
    # is commonly truncated like this.
    schema = {"geometry": "Point", "properties": {"plot_name": "str", "ACC_NAME": "str"}}
    with fiona.open(str(path), "w", driver="ESRI Shapefile",
                    crs=f"EPSG:{UTM_15N_EPSG}", schema=schema) as dst:
        dst.write({"geometry": {"type": "Point", "coordinates": tuple(POINT_NATIVE)},
                   "properties": {"plot_name": "P1", "ACC_NAME": "acc-Z"}})

    csv_path = tmp_path / "plants.csv"
    result = convert_shp_to_plant_csv(path, csv_path, field_map={"accession_name": "ACC_NAME"})

    assert "accession_name" not in result["missing_fields"]
    rows = _read_csv_rows(csv_path)
    assert rows[0]["accession_name"] == "acc-Z"


# ── zero-feature refusal ─────────────────────────────────────────────────


def test_refuses_a_shapefile_with_zero_features(tmp_path: Path) -> None:
    import fiona

    path = tmp_path / "empty.shp"
    schema = {"geometry": "Point", "properties": dict(_SCHEMA_PROPERTIES)}
    with fiona.open(str(path), "w", driver="ESRI Shapefile",
                    crs=f"EPSG:{UTM_15N_EPSG}", schema=schema):
        pass

    with pytest.raises(ValueError, match="zero features"):
        convert_shp_to_plant_csv(path, tmp_path / "plants.csv")


# ── CLI end to end ───────────────────────────────────────────────────────


def test_main_cli_end_to_end(tmp_path: Path) -> None:
    shp = _point_shapefile(tmp_path)
    csv_path = tmp_path / "plants.csv"

    rc = main([str(shp), str(csv_path)])

    assert rc == 0
    assert csv_path.is_file()
    rows = _read_csv_rows(csv_path)
    assert len(rows) == 1


def test_main_cli_refusal_returns_nonzero(tmp_path: Path) -> None:
    shp = _point_shapefile(tmp_path, epsg=None)
    csv_path = tmp_path / "plants.csv"

    rc = main([str(shp), str(csv_path)])

    assert rc == 1
    assert not csv_path.exists()


# ── read_plant_shapefile itself ──────────────────────────────────────────


def _alnum_point_shapefile(tmp_path: Path) -> Path:
    """A ``plot_number`` held as a string field with a non-numeric value, the shape a real plot
    naming scheme can take; the reader carries it through verbatim, unlike ``PlantRecord``'s own
    ``Optional[float]`` field."""
    import fiona

    path = tmp_path / "alnum.shp"
    schema = {"geometry": "Point",
              "properties": {"plot_name": "str", "accession_name": "str", "plot_number": "str"}}
    with fiona.open(str(path), "w", driver="ESRI Shapefile", crs=f"EPSG:{UTM_15N_EPSG}",
                    schema=schema) as dst:
        dst.write({"geometry": {"type": "Point", "coordinates": tuple(POINT_NATIVE)},
                   "properties": {"plot_name": "P9", "accession_name": "acc-Z",
                                  "plot_number": "A1"}})
    return path


def test_read_plant_shapefile_point_rows_match_reference_and_attrs_verbatim(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.postprocessing.plant_mapping import read_plant_shapefile

    result = read_plant_shapefile(_point_shapefile(tmp_path))

    assert result.geometry_kinds == ["point"]
    assert result.skipped_null_geometry == 0
    row = result.rows[0]
    expected_lon, expected_lat = _reference_lonlat(*POINT_NATIVE)
    assert row["WGS84_centroid_x"] == pytest.approx(expected_lon, abs=1e-6)
    assert row["WGS84_centroid_y"] == pytest.approx(expected_lat, abs=1e-6)
    assert row["plot_name"] == "P1"
    assert row["accession_name"] == "acc-A"
    assert row["plot_number"] == "1.0"


def test_read_plant_shapefile_polygon_rows_match_reference_and_attrs_verbatim(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.postprocessing.plant_mapping import read_plant_shapefile

    result = read_plant_shapefile(_polygon_shapefile(tmp_path))

    assert result.geometry_kinds == ["polygon"]
    row = result.rows[0]
    expected_lon, expected_lat = _reference_lonlat(*POINT_NATIVE)
    assert row["WGS84_centroid_x"] == pytest.approx(expected_lon, abs=1e-6)
    assert row["WGS84_centroid_y"] == pytest.approx(expected_lat, abs=1e-6)
    assert row["plot_name"] == "P2"
    assert row["accession_name"] == "acc-B"


def test_read_plant_shapefile_non_numeric_plot_number_carried_verbatim(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.postprocessing.plant_mapping import read_plant_shapefile

    result = read_plant_shapefile(_alnum_point_shapefile(tmp_path))
    assert result.rows[0]["plot_number"] == "A1"


def test_read_plant_shapefile_refuses_a_crs_it_cannot_resolve(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        ShapefileCrsUnknown,
        read_plant_shapefile,
    )

    shp = _point_shapefile(tmp_path, epsg=None)
    with pytest.raises(ShapefileCrsUnknown, match="resolvable coordinate reference system"):
        read_plant_shapefile(shp)


# ── a .prj that exists but does not parse ────────────────────────────────


def _garbage_prj_shapefile(tmp_path: Path) -> Path:
    """A ``.shp`` with a valid geometry beside a ``.prj`` holding unparseable WKT: fiona answers
    the same falsy ``layer.crs`` a missing ``.prj`` would (probed directly against this
    environment's fiona), never an exception at open or at read."""
    shp = _point_shapefile(tmp_path, epsg=UTM_15N_EPSG)
    (tmp_path / "points.prj").write_text("this is not valid WKT at all !!! ###")
    return shp


def test_garbage_prj_refuses_through_convert_shp_to_plant_csv(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.postprocessing.plant_mapping import ShapefileCrsUnknown

    shp = _garbage_prj_shapefile(tmp_path)
    csv_path = tmp_path / "plants.csv"
    with pytest.raises(ShapefileCrsUnknown, match="resolvable coordinate reference system"):
        convert_shp_to_plant_csv(shp, csv_path)
    assert not csv_path.exists()


# ── a geometry type this reader does not centroid or read a point from ──


def _line_string_shapefile(tmp_path: Path) -> Path:
    import fiona

    path = tmp_path / "lines.shp"
    schema = {"geometry": "LineString", "properties": dict(_SCHEMA_PROPERTIES)}
    with fiona.open(str(path), "w", driver="ESRI Shapefile", crs=f"EPSG:{UTM_15N_EPSG}",
                    schema=schema) as dst:
        dst.write({
            "geometry": {
                "type": "LineString",
                "coordinates": [tuple(POINT_NATIVE), (POINT_NATIVE[0] + 10, POINT_NATIVE[1] + 10)],
            },
            "properties": dict(_ATTRS_POINT),
        })
    return path


def test_line_string_geometry_refuses_named_through_convert_shp_to_plant_csv(
    tmp_path: Path,
) -> None:
    """Today a line string is centroided silently and labelled "polygon"; the reader refuses it
    by name instead, since a line's centroid is not a plant's location."""
    shp = _line_string_shapefile(tmp_path)
    with pytest.raises(ValueError, match="LineString"):
        convert_shp_to_plant_csv(shp, tmp_path / "plants.csv")


# ── null-geometry features ───────────────────────────────────────────────


def _point_shapefile_with_null_feature(tmp_path: Path) -> Path:
    import fiona

    path = tmp_path / "with_null.shp"
    schema = {"geometry": "Point", "properties": dict(_SCHEMA_PROPERTIES)}
    with fiona.open(str(path), "w", driver="ESRI Shapefile", crs=f"EPSG:{UTM_15N_EPSG}",
                    schema=schema) as dst:
        dst.write({"geometry": {"type": "Point", "coordinates": tuple(POINT_NATIVE)},
                   "properties": dict(_ATTRS_POINT)})
        dst.write({"geometry": None, "properties": dict(_ATTRS_POLYGON)})
    return path


def test_null_geometry_feature_is_skipped_and_counted(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.postprocessing.plant_mapping import read_plant_shapefile

    result = read_plant_shapefile(_point_shapefile_with_null_feature(tmp_path))
    assert len(result.rows) == 1
    assert result.skipped_null_geometry == 1


def test_convert_shp_to_plant_csv_reports_skipped_null_geometry(tmp_path: Path) -> None:
    shp = _point_shapefile_with_null_feature(tmp_path)
    result = convert_shp_to_plant_csv(shp, tmp_path / "plants.csv")
    assert result["n_features"] == 1
    assert result["skipped_null_geometry"] == 1


# ── PLANT_CSV_COLUMNS: one schema both sides own ─────────────────────────


def test_plant_csv_columns_matches_written_header(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.postprocessing.plant_mapping import PLANT_CSV_COLUMNS

    csv_path = tmp_path / "plants.csv"
    convert_shp_to_plant_csv(_point_shapefile(tmp_path), csv_path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))

    assert header == PLANT_CSV_COLUMNS
    assert PLANT_CSV_COLUMNS == [
        "plot_name", "accession_name", "plot_number", "row_number", "col_number",
        "WGS84_centroid_x", "WGS84_centroid_y",
    ]


def test_converter_csv_carries_non_numeric_plot_number_verbatim(tmp_path: Path) -> None:
    csv_path = tmp_path / "plants.csv"
    convert_shp_to_plant_csv(_alnum_point_shapefile(tmp_path), csv_path)
    rows = _read_csv_rows(csv_path)
    assert rows[0]["plot_number"] == "A1"
