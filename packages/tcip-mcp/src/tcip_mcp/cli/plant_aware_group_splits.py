r"""Plant-aware group-key derivation for ``draw_splits``, over per-stem georeferenced rasters.

When a dataset's images are themselves individually georeferenced rasters (e.g.
per-plant/per-region GeoTIFF chips cut from a larger orthomosaic, which keep the parent's
georeferencing tags), this script derives the group key from location: each image's own center
pixel resolves to a (lat, lon) via its GeoTIFF tags, then to the nearest plant in a plant-locations
CSV, so every capture of one physical plant across every date lands in the same split side.

Uses ``read_plant_csvs``, ``OrthomosaicGeoreference.pixel_to_wgs84`` and ``_nearest_plant``, then
hands the resulting ``{identity: group_key}`` map to ``draw_splits(group_key_map=...)``.
``identity`` is ``<date>/<stem>``, since a stem is unique only within one capture date.

``--subject`` is required: ``draw_splits`` draws its samples through the platform's own per-subject
admission and refuses to write a selection without one.

Usage:
    tcip plant-aware-group-splits <dataset_root> --plant-csv <plants.csv> \
        [--plant-csv <more_plants.csv> ...] --subject <subject> [--attribute <attribute>] \
        --train-ratio <ratio> --val-ratio <ratio> --calibration-ratio <ratio> [--seed 42] \
        [--tolerance-m 5.0] [--output-path <dir>]

``--train-ratio``, ``--val-ratio`` and ``--calibration-ratio`` all have no default and are
required: the three must sum to 1.0, and a selection write (``--output-path``) additionally refuses
any of them being zero, by name.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _raster_pixel_extent(path: Path) -> tuple[int, int]:
    """``(width, height)`` of a GeoTIFF's first page, without decoding any pixel data."""
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        shape = tif.pages[0].shape
    height, width = shape[0], shape[1]
    return width, height


def derive_plant_group_key_map(
    stem_to_raster: dict[str, Path],
    plants: list,
    *,
    nn_tolerance_m: float | None = None,
) -> dict[str, str]:
    """``{identity: plot_name}`` for every key in ``stem_to_raster``, resolved by GPS
    nearest-neighbor.

    Each raster's own center pixel is its representative location, converted to WGS84 via that
    raster's own :class:`OrthomosaicGeoreference` (built per file, since two stems can carry
    different tiepoints or even different CRSes), then matched to the nearest plant in ``plants``.

    ``nn_tolerance_m`` resolves through ``plant_mapping.resolve_nn_tolerance_m``.

    Every stem must resolve: raises ``ValueError`` naming every stem whose raster can't be
    georeferenced or whose nearest plant falls outside tolerance, with its cause, once all stems
    have been checked.
    """
    import tifffile

    from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import (
        GeoreferencingError,
        OrthomosaicGeoreference,
        RotatedRasterError,
    )
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        _nearest_plant,
        resolve_nn_tolerance_m,
    )

    if not plants:
        raise ValueError("no plant records to match against (the plant CSV(s) parsed to zero rows)")

    tolerance_m = resolve_nn_tolerance_m(plants, nn_tolerance_m)["value"]

    group_key_map: dict[str, str] = {}
    failures: list[str] = []
    for stem in sorted(stem_to_raster):
        path = stem_to_raster[stem]
        try:
            georef = OrthomosaicGeoreference.from_file(path)
            width, height = _raster_pixel_extent(path)
            lat, lon = georef.pixel_to_wgs84(width / 2.0, height / 2.0)
        except (RotatedRasterError, GeoreferencingError, OSError, tifffile.TiffFileError) as exc:
            failures.append(f"{stem} ({path}): could not georeference - {exc}")
            continue

        plant, distance_m = _nearest_plant(lat, lon, plants)
        if plant is None or distance_m is None or distance_m > tolerance_m:
            observed = f"{distance_m:.1f}m" if distance_m is not None else "no plants in the CSV"
            failures.append(
                f"{stem} ({path}): nearest plant is {observed} away, outside tolerance "
                f"{tolerance_m:.1f}m (lat={lat:.6f}, lon={lon:.6f})"
            )
            continue
        if not plant.plot_name:
            failures.append(f"{stem} ({path}): the matched plant record has no plot_name")
            continue

        group_key_map[stem] = plant.plot_name

    if failures:
        preview = "\n  ".join(failures[:20])
        more = f"\n  (+{len(failures) - 20} more)" if len(failures) > 20 else ""
        raise ValueError(
            f"{len(failures)} of {len(stem_to_raster)} stem(s) could not be resolved to a "
            f"plant/plot group key:\n  {preview}{more}"
        )
    return group_key_map


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument("dataset_root", help="Dataset root (canonical images/, annotations/ layout).")
    parser.add_argument("--plant-csv", action="append", required=True, dest="plant_csv_paths",
                         help="Plant-locations CSV (read_plant_csvs schema); repeatable.")
    parser.add_argument("--train-ratio", type=float, required=True,
                         help="Fraction for the training side; --train-ratio, --val-ratio and "
                              "--calibration-ratio must sum to 1.0. No default.")
    parser.add_argument("--val-ratio", type=float, required=True,
                         help="Fraction for the validation side; the three ratios must sum to "
                              "1.0. No default.")
    parser.add_argument("--calibration-ratio", type=float, required=True,
                         help="Fraction held out as the calibration universe; the three ratios "
                              "must sum to 1.0. No default. A selection write (--output-path) "
                              "refuses a zero ratio on any of the three, naming it.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance-m", type=float, default=None,
                         help="Max GPS distance (m) to the nearest plant. Defaults to "
                              "grid_pitch_m(plants)/6, the same derivation build_mapping/"
                              "assign_detections_to_plants already use.")
    parser.add_argument("--output-path", default=None,
                         help="Where draw_splits writes the selection.")
    parser.add_argument("--subject", required=True,
                         help="The object class draw_splits draws its members for and the "
                              "confirmed negatives are keyed under.")
    parser.add_argument("--attribute", default=None,
                         help="Scope the draw to instances already assessed for this attribute "
                              "of --subject; omitted, every instance of --subject counts.")
    args = parser.parse_args(argv)

    # Its own process entry point, so it binds the storage backend the seam has no default for.
    from tcip_store.binding import bind_default

    from tcip_mcp.dataset_layout import parse_image_path
    from tcip_mcp.pipelines.data.splits import member_identity
    from tcip_mcp.pipelines.postprocessing.plant_mapping import read_plant_csvs
    from tcip_mcp.tools.data_tools import _scan_dataset, draw_splits

    bind_default()

    scan = _scan_dataset(args.dataset_root)
    stem_to_raster: dict[str, Path] = {}
    for p in scan["images"]:
        _root, date, stem = parse_image_path(p)
        stem_to_raster[member_identity(date, stem)] = Path(p)
    if not stem_to_raster:
        print(f"error: no images found under {args.dataset_root}", file=sys.stderr)
        return 1

    plants = read_plant_csvs(Path(p) for p in args.plant_csv_paths)
    if not plants:
        print(f"error: {args.plant_csv_paths} parsed to zero plant records", file=sys.stderr)
        return 1

    try:
        group_key_map = derive_plant_group_key_map(
            stem_to_raster, plants, nn_tolerance_m=args.tolerance_m,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    n_groups = len(set(group_key_map.values()))
    print(f"Resolved {len(group_key_map)} stem(s) to {n_groups} plant/plot group(s).")

    result = draw_splits(
        folder_path=args.dataset_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        calibration_ratio=args.calibration_ratio,
        seed=args.seed,
        group_key_map=group_key_map,
        output_path=args.output_path,
        subject=args.subject,
        attribute=args.attribute,
    )
    if "error" in result:
        print(f"error: draw_splits refused: {result['error']}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
