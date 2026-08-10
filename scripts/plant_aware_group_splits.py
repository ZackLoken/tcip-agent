"""Plant-aware group-key derivation for ``make_splits``, over per-stem georeferenced rasters.

A train/val/test split that groups only by tile prefix (``make_splits``' default) never notices
that two differently-named images from two different capture dates are photos of the *same*
physical plant: nothing in either stem says so. When a dataset's images are themselves individually
georeferenced rasters (e.g. per-plant/per-region GeoTIFF chips cut from a larger orthomosaic, which
keep the parent's georeferencing tags), this script derives the real group key instead: each image's
own center pixel resolves to a (lat, lon) via its GeoTIFF tags, then to the nearest plant in a
plant-locations CSV, so every capture of one physical plant across every date lands in the same
split side.

Composes three existing pieces without reimplementing any of them: ``read_plant_csvs`` (the plant
CSV parser), ``OrthomosaicGeoreference.pixel_to_wgs84`` (GeoTIFF pixel -> WGS84), and
``_nearest_plant`` (GPS nearest-neighbour match) -- then hands the resulting ``{stem: group_key}``
map to ``make_splits(group_key_map=...)``, which already refuses loudly (via
``resolve_group_key_fn``) if the map doesn't cover every stem it needs.

Usage:
    python scripts/plant_aware_group_splits.py <dataset_root> --plant-csv <plants.csv> \
        [--plant-csv <more_plants.csv> ...] [--train-ratio 0.7] [--val-ratio 0.2] \
        [--test-ratio 0.1] [--seed 42] [--tolerance-m 5.0] [--output-path <dir>] \
        [--materialize] [--no-copy] [--subject <subject>]
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
    """``{stem: plot_name}`` for every stem in ``stem_to_raster``, resolved by GPS nearest-neighbour.

    Each raster's own center pixel is its representative location (mirroring how
    ``assign_detections_to_plants`` uses a detection box's own centroid), converted to WGS84 via
    that raster's own :class:`OrthomosaicGeoreference` -- built per file, since two stems can carry
    different tiepoints or even different CRSes -- then matched to the nearest plant in ``plants``.

    ``nn_tolerance_m`` defaults to the same derivation ``plant_mapping.build_mapping`` and
    ``orthomosaic_mapping.assign_detections_to_plants`` already use: ``grid_pitch_m(plants) / 6``,
    or the honest ``NN_TOLERANCE_METERS`` fallback when the plant layout has too few georeferenced
    plants to derive a pitch from.

    Every stem must resolve: a stem whose raster can't be georeferenced, or whose nearest plant
    falls outside tolerance, is a real gap in this map, not something to silently drop and let
    ``make_splits``' generic "group_key_map is missing N stems" surface downstream instead. Raises
    ``ValueError`` naming every such stem and its specific cause once all stems have been checked,
    rather than stopping at the first.
    """
    import tifffile

    from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import (
        GeoreferencingError,
        OrthomosaicGeoreference,
        RotatedRasterError,
    )
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        NN_TOLERANCE_METERS,
        _nearest_plant,
        grid_pitch_m,
    )

    if not plants:
        raise ValueError("no plant records to match against (the plant CSV(s) parsed to zero rows)")

    if nn_tolerance_m is None:
        pitch = grid_pitch_m(plants)
        nn_tolerance_m = (pitch / 6) if pitch > 0 else NN_TOLERANCE_METERS

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
        if plant is None or distance_m is None or distance_m > nn_tolerance_m:
            observed = f"{distance_m:.1f}m" if distance_m is not None else "no plants in the CSV"
            failures.append(
                f"{stem} ({path}): nearest plant is {observed} away, outside tolerance "
                f"{nn_tolerance_m:.1f}m (lat={lat:.6f}, lon={lon:.6f})"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Dataset root (canonical images/, annotations/ layout).")
    parser.add_argument("--plant-csv", action="append", required=True, dest="plant_csv_paths",
                         help="Plant-locations CSV (read_plant_csvs schema); repeatable.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance-m", type=float, default=None,
                         help="Max GPS distance (m) to the nearest plant. Defaults to "
                              "grid_pitch_m(plants)/6, the same derivation build_mapping/"
                              "assign_detections_to_plants already use.")
    parser.add_argument("--output-path", default=None, help="Where make_splits writes manifests.")
    parser.add_argument("--materialize", action="store_true",
                         help="Also lay out a {train,val,test}/{images,labels}/ tree.")
    parser.add_argument("--no-copy", action="store_true",
                         help="Symlink instead of copy when materializing.")
    parser.add_argument("--subject", default=None,
                         help="Subject confirmed negatives are keyed under (materialize only).")
    args = parser.parse_args(argv)

    from tcip_mcp.pipelines.postprocessing.plant_mapping import read_plant_csvs
    from tcip_mcp.tools.data_tools import _scan_dataset, make_splits

    scan = _scan_dataset(args.dataset_root)
    stem_to_raster = {Path(p).stem: Path(p) for p in scan["images"]}
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

    result = make_splits(
        folder_path=args.dataset_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        group_key_map=group_key_map,
        output_path=args.output_path,
        materialize=args.materialize,
        copy_files=not args.no_copy,
        subject=args.subject,
    )
    if "error" in result:
        print(f"error: make_splits refused: {result['error']}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
