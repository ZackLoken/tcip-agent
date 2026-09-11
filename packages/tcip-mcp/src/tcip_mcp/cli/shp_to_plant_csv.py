"""Convert a plant-locations shapefile into ``read_plant_csvs``' CSV schema.

Composes ``plant_mapping.read_plant_shapefile`` for the actual read (the CRS refusal, the
point/polygon/multipolygon geometry rule, the DBF field resolution and truncation) and writes its
rows through ``PLANT_CSV_COLUMNS``, the same header ``read_plant_csv_bytes`` reads, so the two
sides cannot drift; ``n_features`` is the rows written, which excludes the null-geometry features
``skipped_null_geometry`` counts. Validates its own output by reading it back through
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


def _validate_round_trip(csv_path: Path, n_written: int) -> int:
    """Read ``csv_path`` back through ``read_plant_csvs`` and fail loudly, naming the real cause,
    rather than letting a schema/column mismatch surface later as ``draw_splits``' generic
    "group_key_map is missing N stems"."""
    from tcip_mcp.pipelines.postprocessing.plant_mapping import PLANT_CSV_COLUMNS, read_plant_csvs

    parsed = read_plant_csvs([csv_path])
    if len(parsed) == 0:
        raise ValueError(
            f"{csv_path}: wrote {n_written} row(s) but read_plant_csvs parsed 0 plant records back "
            f"from it. The likely cause is a WGS84_centroid_x/WGS84_centroid_y value that failed "
            f"float() parsing, or a header not matching the expected columns {PLANT_CSV_COLUMNS}."
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

    Returns ``{csv_path, n_features, n_parsed, geometry_kinds, missing_fields,
    skipped_null_geometry}``. Raises ``ValueError`` if the source has no resolvable CRS
    (``ShapefileCrsUnknown``), a feature's geometry is neither point, polygon nor multipolygon,
    has zero features with readable geometry, or if the written CSV fails to round-trip through
    ``read_plant_csvs``.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        PLANT_CSV_COLUMNS,
        read_plant_shapefile,
    )

    shp_path = Path(shp_path)
    csv_path = Path(csv_path)

    result = read_plant_shapefile(shp_path, field_map=field_map)
    if not result.rows:
        raise ValueError(f"{shp_path}: zero features with readable geometry; nothing to convert.")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PLANT_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(result.rows)

    n_parsed = _validate_round_trip(csv_path, len(result.rows))

    return {
        "csv_path": str(csv_path),
        "n_features": len(result.rows),
        "n_parsed": n_parsed,
        "geometry_kinds": result.geometry_kinds,
        "missing_fields": result.missing_fields,
        "skipped_null_geometry": result.skipped_null_geometry,
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
