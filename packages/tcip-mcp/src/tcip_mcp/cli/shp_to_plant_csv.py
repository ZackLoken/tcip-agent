"""Convert a plant-locations shapefile into ``read_plant_csvs``' CSV schema.

GDAL 3's authority-compliant axis order for EPSG:4326 returns (lat, lon); reprojecting without
``SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER)`` silently writes latitude into the
``WGS84_centroid_x`` (longitude) column instead. Refuses a source ``.shp`` with no resolvable CRS
(missing/unreadable ``.prj``) rather than guessing one, handles both point and polygon source
geometry (a polygon's own centroid), and validates its own output by reading it back through
``read_plant_csvs`` before reporting success.

Usage:
    tcip shp-to-plant-csv <plants.shp> <plants.csv> \
        [--plot-name-field FIELD] [--accession-name-field FIELD] \
        [--plot-number-field FIELD] [--row-number-field FIELD] [--col-number-field FIELD]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

#: read_plant_csvs' exact column schema (plant_mapping.py's read_plant_csvs/PlantRecord).
CSV_FIELDS = [
    "plot_name", "accession_name", "plot_number", "row_number", "col_number",
    "WGS84_centroid_x", "WGS84_centroid_y",
]

# Optional pass-through columns' default source-attribute field names, overridable per shapefile:
# an ESRI Shapefile DBF caps field names at 10 characters, so a source rarely matches these exactly.
_DEFAULT_FIELD_MAP = {
    "plot_name": "plot_name",
    "accession_name": "accession_name",
    "plot_number": "plot_number",
    "row_number": "row_number",
    "col_number": "col_number",
}


def _field_value(properties: dict, field_name: str | None) -> str:
    if field_name is None:
        return ""
    value = properties.get(field_name)
    return "" if value is None else str(value)


def _validate_round_trip(csv_path: Path, n_written: int) -> int:
    """Read ``csv_path`` back through ``read_plant_csvs`` and fail loudly, naming the real cause,
    rather than letting a schema/column mismatch surface later as ``draw_splits``' generic
    "group_key_map is missing N stems"."""
    from tcip_mcp.pipelines.postprocessing.plant_mapping import read_plant_csvs

    parsed = read_plant_csvs([csv_path])
    if len(parsed) == 0:
        raise ValueError(
            f"{csv_path}: wrote {n_written} row(s) but read_plant_csvs parsed 0 plant records back "
            f"from it. The likely cause is a WGS84_centroid_x/WGS84_centroid_y value that failed "
            f"float() parsing, or a header not matching the expected columns {CSV_FIELDS}."
        )
    if len(parsed) != n_written:
        raise ValueError(
            f"{csv_path}: wrote {n_written} row(s) but read_plant_csvs parsed only {len(parsed)} "
            "back; one or more rows' WGS84_centroid_x/WGS84_centroid_y failed float() parsing."
        )
    return len(parsed)


def convert_shp_to_plant_csv(
    shp_path: str | Path,
    csv_path: str | Path,
    *,
    field_map: dict[str, str] | None = None,
) -> dict:
    """Convert ``shp_path`` (a point or polygon plant-locations shapefile) to ``csv_path`` in
    ``read_plant_csvs``' schema, reprojecting every feature's own coordinate to WGS84.

    Returns ``{csv_path, n_features, n_parsed, geometry_kinds, missing_fields}``. Raises
    ``ValueError`` if the source has no resolvable CRS, has zero features with readable geometry,
    or if the written CSV fails to round-trip through ``read_plant_csvs``.
    """
    import fiona
    from pyproj import Transformer
    from shapely.geometry import shape

    shp_path = Path(shp_path)
    csv_path = Path(csv_path)
    fields = dict(_DEFAULT_FIELD_MAP)
    if field_map:
        fields.update(field_map)

    geom_kinds: set[str] = set()
    rows: list[dict] = []
    with fiona.open(str(shp_path)) as layer:
        # fiona reports a missing .prj as an empty CRS rather than None, so this tests for
        # emptiness; a bare `is None` would let an unprojected source through to pyproj.
        if not layer.crs:
            raise ValueError(
                f"{shp_path}: no resolvable coordinate reference system (missing or unreadable "
                ".prj). Refusing to guess a CRS; supply a .prj alongside the .shp."
            )
        # always_xy keeps both sides in (lon, lat) order; EPSG:4326's authority-declared order is
        # (lat, lon), and without this every coordinate lands in the wrong CSV column.
        transform = Transformer.from_crs(layer.crs, "EPSG:4326", always_xy=True)

        available = set(layer.schema["properties"])
        resolved_fields: dict[str, str | None] = {}
        for csv_col, shp_field in fields.items():
            if shp_field in available:
                resolved_fields[csv_col] = shp_field
            elif shp_field[:10] in available:
                # The ESRI Shapefile DBF driver truncates a field name over 10 chars (e.g.
                # "accession_name" -> "accession_"); try that truncated form before giving up.
                resolved_fields[csv_col] = shp_field[:10]
            else:
                resolved_fields[csv_col] = None
        missing = sorted(csv_col for csv_col, f in resolved_fields.items() if f is None)

        for feature in layer:
            if feature.geometry is None:
                continue
            geom = shape(feature.geometry)
            if geom.geom_type == "Point":
                geom_kinds.add("point")
                x, y = geom.x, geom.y
            else:
                geom_kinds.add("polygon")
                centroid = geom.centroid
                x, y = centroid.x, centroid.y
            lon, lat = transform.transform(x, y)
            properties = dict(feature.properties)
            rows.append({
                "plot_name": _field_value(properties, resolved_fields["plot_name"]),
                "accession_name": _field_value(properties, resolved_fields["accession_name"]),
                "plot_number": _field_value(properties, resolved_fields["plot_number"]),
                "row_number": _field_value(properties, resolved_fields["row_number"]),
                "col_number": _field_value(properties, resolved_fields["col_number"]),
                "WGS84_centroid_x": lon,
                "WGS84_centroid_y": lat,
            })

    if not rows:
        raise ValueError(f"{shp_path}: zero features with readable geometry; nothing to convert.")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_parsed = _validate_round_trip(csv_path, len(rows))

    return {
        "csv_path": str(csv_path),
        "n_features": len(rows),
        "n_parsed": n_parsed,
        "geometry_kinds": sorted(geom_kinds),
        "missing_fields": missing,
    }


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument("shp_path")
    parser.add_argument("csv_path")
    parser.add_argument("--plot-name-field", default=None)
    parser.add_argument("--accession-name-field", default=None)
    parser.add_argument("--plot-number-field", default=None)
    parser.add_argument("--row-number-field", default=None)
    parser.add_argument("--col-number-field", default=None)
    args = parser.parse_args(argv)

    field_map = {}
    for csv_col, cli_value in (
        ("plot_name", args.plot_name_field),
        ("accession_name", args.accession_name_field),
        ("plot_number", args.plot_number_field),
        ("row_number", args.row_number_field),
        ("col_number", args.col_number_field),
    ):
        if cli_value:
            field_map[csv_col] = cli_value

    try:
        result = convert_shp_to_plant_csv(args.shp_path, args.csv_path, field_map=field_map or None)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {result['n_features']} plant record(s) to {result['csv_path']} "
          f"(source geometry: {', '.join(result['geometry_kinds'])}); "
          f"read_plant_csvs parsed {result['n_parsed']} back.")
    if result["missing_fields"]:
        print(f"Note: no source attribute field found for {result['missing_fields']}; "
              "written empty in the CSV.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
