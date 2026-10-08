r"""Plant-aware group-key derivation for ``draw_splits``, over per-stem georeferenced rasters.

When a dataset's images are themselves individually georeferenced rasters (e.g.
per-plant/per-region GeoTIFF chips cut from a larger orthomosaic, which keep the parent's
georeferencing tags), this script derives the group key from location: each image's own center
pixel resolves to a (lat, lon) via its GeoTIFF tags, then to the nearest plant in a plant-locations
CSV, so every capture of one physical plant across every date lands in the same split side.

Uses ``read_plant_csvs``, ``OrthomosaicGeoreference.pixel_to_wgs84`` and ``nearest_plant``, then
hands the resulting ``{identity: group_key}`` map to ``draw_splits(group_key_map=...)``.
``identity`` is ``<date>/<stem>``, since a stem is unique only within one capture date.

``--subject`` is required: ``draw_splits`` draws its samples through the platform's own per-subject
admission and refuses to write a selection without one.

Usage:
    tcip plant-aware-group-splits <dataset_root> --project <project> --plant-csv <plants.csv> \
        [--plant-csv <more_plants.csv> ...] --subject <subject> --seed <seed> \
        [--val-ratio <ratio>] [--calibration-ratio <ratio>] [--holdout-ratio <ratio>] \
        [--tolerance-m <meters>] [--output-path <dir>]

The ratios default to ``draw_splits``'s own (``splits.DEFAULT_SHARES``), train the remainder; the
seed has no default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tcip_mcp.pipelines.data.splits import DEFAULT_SHARES


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

    from tcip_mcp.pipelines.image_utils import display_frame
    from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import (
        GeoreferencingError,
        OrthomosaicGeoreference,
        RotatedRasterError,
    )
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        nearest_plant,
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
            width, height = display_frame(path)
            lat, lon = georef.pixel_to_wgs84(width / 2.0, height / 2.0)
        except (RotatedRasterError, GeoreferencingError, OSError, ValueError,
                tifffile.TiffFileError) as exc:
            failures.append(f"{stem} ({path}): could not georeference - {exc}")
            continue

        index, distance_m = nearest_plant((lat, lon), plants, within_m=tolerance_m)
        if index is None:
            failures.append(
                f"{stem} ({path}): nearest plant is {distance_m:.1f}m away, outside tolerance "
                f"{tolerance_m:.1f}m (lat={lat:.6f}, lon={lon:.6f})"
            )
            continue
        plant = plants[index]
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
    parser.add_argument("dataset_root", help="Dataset root (canonical images/ layout).")
    parser.add_argument("--project", required=True,
                        help="The project the draw acts on and records its audit line under.")
    parser.add_argument("--plant-csv", action="append", required=True, dest="plant_csv_paths",
                         help="Plant-locations CSV (read_plant_csvs schema); repeatable.")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_SHARES["val"],
                         help="Fraction for the validation side; train takes the remainder.")
    parser.add_argument("--calibration-ratio", type=float, default=DEFAULT_SHARES["calibration"],
                         help="Fraction an assessment fits its operating point on.")
    parser.add_argument("--holdout-ratio", type=float, default=DEFAULT_SHARES["holdout"],
                         help="Fraction an assessment checks its operating point against. A side "
                              "whose ratio is zero is not drawn.")
    parser.add_argument("--seed", type=int, required=True,
                         help="The seed the draw is reproduced by. No default.")
    parser.add_argument("--tolerance-m", type=float, default=None,
                         help="Max GPS distance (m) to the nearest plant. Unstated, it derives "
                              "from the plant grid (plant_mapping.resolve_nn_tolerance_m).")
    parser.add_argument("--output-path", default=None,
                         help="Where draw_splits writes the selection.")
    parser.add_argument("--subject", required=True,
                         help="The object class draw_splits draws its members for and the "
                              "confirmed negatives are keyed under.")
    args = parser.parse_args(argv)

    from tcip_mcp.cli import bound_project
    from tcip_mcp.dataset_layout import parse_image_path
    from tcip_mcp.pipelines.data.splits import member_identity
    from tcip_mcp.pipelines.postprocessing.plant_mapping import read_plant_csvs
    from tcip_mcp.tools.data_tools import _scan_dataset, draw_splits

    project = bound_project(args.project)

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
        project,
        folder_path=args.dataset_root,
        val_ratio=args.val_ratio,
        calibration_ratio=args.calibration_ratio,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
        group_key_map=group_key_map,
        output_path=args.output_path,
        subject=args.subject,
    )
    if "error" in result:
        print(f"error: draw_splits refused: {result['error']}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
