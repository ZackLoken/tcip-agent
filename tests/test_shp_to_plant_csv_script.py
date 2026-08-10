"""``scripts/shp_to_plant_csv.py``: shapefile -> ``read_plant_csvs`` CSV, with correct WGS84 axis
order and point/polygon geometry handling.

GDAL 3's authority-compliant axis order for EPSG:4326 is (lat, lon); a converter that reprojects
without pinning ``OAMS_TRADITIONAL_GIS_ORDER`` on both the source and target SRS silently writes
latitude into the ``WGS84_centroid_x`` (longitude) column. These tests build real synthetic
shapefiles in a real projected CRS (UTM 15N, via ``osgeo.ogr``, no binary fixture committed) and
check the output against an independently-computed pyproj transform, not just a plausible range.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.shp_to_plant_csv import _validate_round_trip, convert_shp_to_plant_csv, main

UTM_15N_EPSG = 32615

# A single UTM point whose WGS84 equivalent is independently hand-verifiable: 500 000 mE is UTM's
# own false easting (the central meridian, -93 degrees exactly) for every northern zone.
POINT_NATIVE = (500_000.0, 4_649_776.0)


def _reference_lonlat(native_x: float, native_y: float) -> tuple[float, float]:
    import pyproj

    transformer = pyproj.Transformer.from_crs(f"EPSG:{UTM_15N_EPSG}", "EPSG:4326", always_xy=True)
    return transformer.transform(native_x, native_y)


def _new_shapefile_layer(ds, geom_type, epsg: int | None):
    from osgeo import ogr, osr

    srs = None
    if epsg is not None:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
    layer = ds.CreateLayer("plants", srs=srs, geom_type=geom_type)
    for name, kind in (
        ("plot_name", ogr.OFTString), ("accession_name", ogr.OFTString),
        ("plot_number", ogr.OFTReal), ("row_number", ogr.OFTReal), ("col_number", ogr.OFTReal),
    ):
        layer.CreateField(ogr.FieldDefn(name, kind))
    return layer


def _set_field(feature, layer, name: str, value) -> None:
    """Set a feature field by its logical name, following the same truncated-name convention the
    ESRI Shapefile DBF driver applies to a field name over 10 characters (the script's own
    ``convert_shp_to_plant_csv`` resolves a source's schema the same way)."""
    defn = layer.GetLayerDefn()
    if defn.GetFieldIndex(name) >= 0:
        feature.SetField(name, value)
    else:
        feature.SetField(name[:10], value)


def _add_point_feature(layer, x: float, y: float, **attrs) -> None:
    from osgeo import ogr

    feature = ogr.Feature(layer.GetLayerDefn())
    for k, v in attrs.items():
        _set_field(feature, layer, k, v)
    geom = ogr.Geometry(ogr.wkbPoint)
    geom.AddPoint(x, y)
    feature.SetGeometry(geom)
    layer.CreateFeature(feature)


def _add_square_polygon_feature(layer, cx: float, cy: float, half: float, **attrs) -> None:
    """A square centered exactly on ``(cx, cy)``, so its centroid is the center by construction,
    not something this test would need to recompute geometrically."""
    from osgeo import ogr

    feature = ogr.Feature(layer.GetLayerDefn())
    for k, v in attrs.items():
        _set_field(feature, layer, k, v)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half), (-half, -half)):
        ring.AddPoint(cx + dx, cy + dy)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    feature.SetGeometry(poly)
    layer.CreateFeature(feature)


@pytest.fixture
def _ogr():
    from osgeo import ogr, osr

    ogr.UseExceptions()
    osr.UseExceptions()
    return ogr


def _point_shapefile(tmp_path: Path, ogr_mod, *, epsg: int | None = UTM_15N_EPSG) -> Path:
    path = tmp_path / "points.shp"
    driver = ogr_mod.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(str(path))
    layer = _new_shapefile_layer(ds, ogr_mod.wkbPoint, epsg)
    _add_point_feature(
        layer, *POINT_NATIVE,
        plot_name="P1", accession_name="acc-A", plot_number=1.0, row_number=2.0, col_number=3.0,
    )
    ds = None
    return path


def _polygon_shapefile(tmp_path: Path, ogr_mod) -> Path:
    path = tmp_path / "polys.shp"
    driver = ogr_mod.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(str(path))
    layer = _new_shapefile_layer(ds, ogr_mod.wkbPolygon, UTM_15N_EPSG)
    _add_square_polygon_feature(
        layer, *POINT_NATIVE, half=10.0,
        plot_name="P2", accession_name="acc-B", plot_number=4.0, row_number=5.0, col_number=6.0,
    )
    ds = None
    return path


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── axis order + point geometry ─────────────────────────────────────────


def test_point_geometry_reprojects_to_correct_unswapped_wgs84(tmp_path: Path, _ogr) -> None:
    shp = _point_shapefile(tmp_path, _ogr)
    csv_path = tmp_path / "plants.csv"

    result = convert_shp_to_plant_csv(shp, csv_path)

    assert result["geometry_kinds"] == ["point"]
    rows = _read_csv_rows(csv_path)
    assert len(rows) == 1
    lon, lat = float(rows[0]["WGS84_centroid_x"]), float(rows[0]["WGS84_centroid_y"])

    # Not swapped: UTM zone 15N sits in the western hemisphere, well north of the equator.
    assert -100 < lon < -85
    assert 35 < lat < 50

    # Matches an independent oracle (pyproj, not osgeo.osr) to float precision, not just a
    # plausible range.
    expected_lon, expected_lat = _reference_lonlat(*POINT_NATIVE)
    assert lon == pytest.approx(expected_lon, abs=1e-6)
    assert lat == pytest.approx(expected_lat, abs=1e-6)

    assert rows[0]["plot_name"] == "P1"
    assert rows[0]["accession_name"] == "acc-A"
    assert rows[0]["plot_number"] == "1.0"


def test_swapped_axis_order_would_fail_this_test(tmp_path: Path) -> None:
    """Pins the failure mode this converter guards against: transforming with the default
    (authority-compliant, lat/lon) axis order instead of TRADITIONAL_GIS_ORDER lands outside a
    plausible longitude range for UTM 15N. Exercises the same osgeo.osr call the script makes,
    with the fix deliberately omitted, so a regression that drops the fix would be caught by the
    assertion above failing, not by this test passing either way."""
    from osgeo import osr

    src = osr.SpatialReference()
    src.ImportFromEPSG(UTM_15N_EPSG)
    wgs = osr.SpatialReference()
    wgs.ImportFromEPSG(4326)
    # Deliberately no SetAxisMappingStrategy call on either SRS.
    transform = osr.CoordinateTransformation(src, wgs)
    a, b, _z = transform.TransformPoint(*POINT_NATIVE)

    # With no axis fix, GDAL 3 returns (lat, lon): the first coordinate lands in latitude's
    # plausible range, not longitude's, i.e. exactly the swap the fix prevents.
    assert 35 < a < 50
    assert not (-100 < a < -85)


# ── polygon geometry (centroid) ─────────────────────────────────────────


def test_polygon_geometry_uses_centroid(tmp_path: Path, _ogr) -> None:
    shp = _polygon_shapefile(tmp_path, _ogr)
    csv_path = tmp_path / "plants.csv"

    result = convert_shp_to_plant_csv(shp, csv_path)

    assert result["geometry_kinds"] == ["polygon"]
    rows = _read_csv_rows(csv_path)
    lon, lat = float(rows[0]["WGS84_centroid_x"]), float(rows[0]["WGS84_centroid_y"])
    expected_lon, expected_lat = _reference_lonlat(*POINT_NATIVE)
    assert lon == pytest.approx(expected_lon, abs=1e-6)
    assert lat == pytest.approx(expected_lat, abs=1e-6)


# ── CRS refusal ──────────────────────────────────────────────────────────


def test_refuses_a_shapefile_with_no_resolvable_crs(tmp_path: Path, _ogr) -> None:
    shp = _point_shapefile(tmp_path, _ogr, epsg=None)
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


def test_missing_optional_field_is_reported_and_written_empty(tmp_path: Path, _ogr) -> None:
    shp = _point_shapefile(tmp_path, _ogr)
    csv_path = tmp_path / "plants.csv"

    result = convert_shp_to_plant_csv(shp, csv_path, field_map={"accession_name": "no_such_field"})

    assert "accession_name" in result["missing_fields"]
    rows = _read_csv_rows(csv_path)
    assert rows[0]["accession_name"] == ""


def test_field_map_override_picks_up_a_renamed_source_field(tmp_path: Path, _ogr) -> None:
    path = tmp_path / "renamed.shp"
    driver = _ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(str(path))
    layer = _new_shapefile_layer(ds, _ogr.wkbPoint, UTM_15N_EPSG)
    # ESRI Shapefile DBF field names are capped at 10 characters: a real source's "accession_name"
    # is commonly truncated like this.
    layer.CreateField(_ogr.FieldDefn("ACC_NAME", _ogr.OFTString))
    feature = _ogr.Feature(layer.GetLayerDefn())
    feature.SetField("plot_name", "P1")
    feature.SetField("ACC_NAME", "acc-Z")
    geom = _ogr.Geometry(_ogr.wkbPoint)
    geom.AddPoint(*POINT_NATIVE)
    feature.SetGeometry(geom)
    layer.CreateFeature(feature)
    ds = None

    csv_path = tmp_path / "plants.csv"
    result = convert_shp_to_plant_csv(path, csv_path, field_map={"accession_name": "ACC_NAME"})

    assert "accession_name" not in result["missing_fields"]
    rows = _read_csv_rows(csv_path)
    assert rows[0]["accession_name"] == "acc-Z"


# ── zero-feature refusal ─────────────────────────────────────────────────


def test_refuses_a_shapefile_with_zero_features(tmp_path: Path, _ogr) -> None:
    path = tmp_path / "empty.shp"
    driver = _ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(str(path))
    _new_shapefile_layer(ds, _ogr.wkbPoint, UTM_15N_EPSG)
    ds = None

    with pytest.raises(ValueError, match="zero features"):
        convert_shp_to_plant_csv(path, tmp_path / "plants.csv")


# ── CLI end to end ───────────────────────────────────────────────────────


def test_main_cli_end_to_end(tmp_path: Path, _ogr) -> None:
    shp = _point_shapefile(tmp_path, _ogr)
    csv_path = tmp_path / "plants.csv"

    rc = main([str(shp), str(csv_path)])

    assert rc == 0
    assert csv_path.is_file()
    rows = _read_csv_rows(csv_path)
    assert len(rows) == 1


def test_main_cli_refusal_returns_nonzero(tmp_path: Path, _ogr) -> None:
    shp = _point_shapefile(tmp_path, _ogr, epsg=None)
    csv_path = tmp_path / "plants.csv"

    rc = main([str(shp), str(csv_path)])

    assert rc == 1
    assert not csv_path.exists()
