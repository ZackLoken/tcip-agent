"""``scripts/plant_aware_group_splits.py``: plant/plot-identity group keys for ``make_splits``.

Builds synthetic per-stem georeferenced GeoTIFFs (the same tiepoint + pixel-scale + GeoKeyDirectory
tag pattern ``test_orthomosaic_mapping.py`` uses) at known real-world offsets from two plants, and
checks the derived ``{stem: group_key}`` map: same physical plant -> same group key regardless of
which "date" (source raster) the stem came from, a plant far outside tolerance -> a named refusal,
an ungeoreferenced source -> a named refusal, and the map ``make_splits(group_key_map=...)`` actually
accepts, keeping every group's stems on one split side.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import tifffile

from scripts.plant_aware_group_splits import derive_plant_group_key_map, main

UTM_15N_EPSG = 32615
PIXEL_SCALE = 1.0
SHAPE = (4, 4, 3)  # (height, width, channels): center pixel is (2.0, 2.0)


def _geokeys() -> tuple[int, ...]:
    # GTModelTypeGeoKey=1 (Projected), ProjectedCSTypeGeoKey=UTM_15N_EPSG.
    return (1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, UTM_15N_EPSG)


def _write_geotiff(path: Path, native_x: float, native_y: float) -> None:
    """A small GeoTIFF whose pixel (0, 0) sits at real-world ``(native_x, native_y)`` in UTM 15N."""
    arr = np.zeros(SHAPE, dtype=np.uint8)
    tifffile.imwrite(
        str(path), arr, photometric="rgb",
        extratags=[
            (33550, "d", 3, (PIXEL_SCALE, PIXEL_SCALE, 0.0), False),
            (33922, "d", 6, (0.0, 0.0, 0.0, native_x, native_y, 0.0), False),
            (34735, "H", len(_geokeys()), _geokeys(), False),
        ],
    )


def _write_ungeoreferenced_tiff(path: Path) -> None:
    tifffile.imwrite(str(path), np.zeros(SHAPE, dtype=np.uint8), photometric="rgb")


def _center_latlon(native_x: float, native_y: float) -> tuple[float, float]:
    """The (lat, lon) a center-pixel GeoTIFF at this tiepoint resolves to, computed independently
    via pyproj (not through ``OrthomosaicGeoreference``) so a fixture plant sits exactly at the
    raster's own center."""
    import pyproj

    height, width, _ = SHAPE
    center_native_x = native_x + (width / 2.0) * PIXEL_SCALE
    center_native_y = native_y - (height / 2.0) * PIXEL_SCALE
    transformer = pyproj.Transformer.from_crs(f"EPSG:{UTM_15N_EPSG}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(center_native_x, center_native_y)
    return lat, lon


# Two plants ~100 m apart in UTM easting: far outside any NN tolerance a 2-plant, ~100 m grid
# pitch would derive (pitch/6 ~= 16.7 m), so a stem can only match its own true plant.
P1_TIEPOINT = (500_000.0, 4_800_000.0)
P2_TIEPOINT = (500_100.0, 4_800_000.0)
FAR_TIEPOINT = (600_000.0, 4_800_000.0)  # ~100 km away: outside tolerance for both plants


def _write_plant_csv(path: Path, plants: list[tuple[str, str, float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["plot_name", "accession_name", "WGS84_centroid_x", "WGS84_centroid_y"])
        for plot_name, accession_name, lat, lon in plants:
            w.writerow([plot_name, accession_name, lon, lat])


@pytest.fixture
def two_plant_csv(tmp_path: Path) -> Path:
    p1_lat, p1_lon = _center_latlon(*P1_TIEPOINT)
    p2_lat, p2_lon = _center_latlon(*P2_TIEPOINT)
    csv_path = tmp_path / "plants.csv"
    _write_plant_csv(csv_path, [("P1", "acc-A", p1_lat, p1_lon), ("P2", "acc-B", p2_lat, p2_lon)])
    return csv_path


def _plants(csv_path: Path) -> list:
    from tcip_mcp.pipelines.postprocessing.plant_mapping import read_plant_csvs

    return read_plant_csvs([csv_path])


# ── derive_plant_group_key_map ──────────────────────────────────────────


def test_resolves_each_stem_to_its_own_nearest_plant(tmp_path: Path, two_plant_csv: Path) -> None:
    r1 = tmp_path / "r1.tif"
    r2 = tmp_path / "r2.tif"
    _write_geotiff(r1, *P1_TIEPOINT)
    _write_geotiff(r2, *P2_TIEPOINT)

    result = derive_plant_group_key_map({"r1": r1, "r2": r2}, _plants(two_plant_csv))

    assert result == {"r1": "P1", "r2": "P2"}


def test_same_plant_two_capture_dates_share_one_group_key(
    tmp_path: Path, two_plant_csv: Path,
) -> None:
    """The actual value this script exists for: two differently-named stems (different capture
    dates) of the same physical plant must land under the identical group key, so a split can
    never put one date's photo of a plant in train and another date's photo of the same plant in
    val/test."""
    date1 = tmp_path / "p1_2026-02-01.tif"
    date2 = tmp_path / "p1_2026-03-01.tif"
    other = tmp_path / "p2_2026-02-01.tif"
    _write_geotiff(date1, *P1_TIEPOINT)
    _write_geotiff(date2, *P1_TIEPOINT)
    _write_geotiff(other, *P2_TIEPOINT)

    result = derive_plant_group_key_map(
        {"p1_2026-02-01": date1, "p1_2026-03-01": date2, "p2_2026-02-01": other},
        _plants(two_plant_csv),
    )

    assert result["p1_2026-02-01"] == result["p1_2026-03-01"]
    assert result["p1_2026-02-01"] != result["p2_2026-02-01"]


def test_refuses_naming_the_stem_when_no_plant_is_within_tolerance(
    tmp_path: Path, two_plant_csv: Path,
) -> None:
    far = tmp_path / "far_away.tif"
    _write_geotiff(far, *FAR_TIEPOINT)

    with pytest.raises(ValueError, match="far_away"):
        derive_plant_group_key_map({"far_away": far}, _plants(two_plant_csv))


def test_refuses_naming_the_stem_when_the_raster_has_no_georeferencing(
    tmp_path: Path, two_plant_csv: Path,
) -> None:
    plain = tmp_path / "plain.tif"
    _write_ungeoreferenced_tiff(plain)

    with pytest.raises(ValueError, match="plain"):
        derive_plant_group_key_map({"plain": plain}, _plants(two_plant_csv))


def test_a_malformed_tiff_is_named_and_does_not_kill_the_whole_batch(
    tmp_path: Path, two_plant_csv: Path,
) -> None:
    """``_raster_pixel_extent`` reads through ``tifffile.TiffFile``, which raises
    ``tifffile.TiffFileError`` (not an ``OSError`` subclass) on a non-TIFF/malformed file. Mixing
    one good stem with one malformed one proves the malformed stem is named specifically in the
    refusal, matching the function's own "name every failing stem" promise, rather than an
    uncaught TiffFileError killing the whole batch (including the good stem)."""
    good = tmp_path / "good.tif"
    _write_geotiff(good, *P1_TIEPOINT)
    malformed = tmp_path / "malformed.tif"
    malformed.write_bytes(b"not a real tiff")

    with pytest.raises(ValueError, match="malformed"):
        derive_plant_group_key_map(
            {"good": good, "malformed": malformed}, _plants(two_plant_csv))


def test_refuses_on_empty_plant_list(tmp_path: Path) -> None:
    r1 = tmp_path / "r1.tif"
    _write_geotiff(r1, *P1_TIEPOINT)

    with pytest.raises(ValueError, match="zero rows"):
        derive_plant_group_key_map({"r1": r1}, [])


# ── make_splits(group_key_map=...) admits the derived map (rail-admits-valid-work) ─────────────


def _write_dataset_stem(dataset_root: Path, date: str, stem: str, tiepoint: tuple[float, float]) -> None:
    from tcip_annotation import json_io

    images_dir = dataset_root / "images" / date
    labels_dir = dataset_root / "annotations" / date
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    _write_geotiff(images_dir / f"{stem}.tif", *tiepoint)
    json_io.write_annotations(labels_dir / f"{stem}.json", [], SHAPE[1], SHAPE[0])


def test_make_splits_keeps_every_plants_stems_on_one_split_side(
    tmp_path: Path, two_plant_csv: Path,
) -> None:
    from tcip_mcp.tools.data_tools import _scan_dataset, make_splits

    dataset_root = tmp_path / "dataset"
    # Two plants, two capture dates each: four stems total, two groups of two.
    for date, tiepoint in (("2026-02-01", P1_TIEPOINT), ("2026-03-01", P1_TIEPOINT)):
        _write_dataset_stem(dataset_root, date, f"p1_{date}", tiepoint)
    for date, tiepoint in (("2026-02-01", P2_TIEPOINT), ("2026-03-01", P2_TIEPOINT)):
        _write_dataset_stem(dataset_root, date, f"p2_{date}", tiepoint)

    scan = _scan_dataset(str(dataset_root))
    stem_to_raster = {Path(p).stem: Path(p) for p in scan["images"]}
    group_key_map = derive_plant_group_key_map(stem_to_raster, _plants(two_plant_csv))

    out_dir = tmp_path / "splits_out"
    result = make_splits(
        folder_path=str(dataset_root), train_ratio=0.5, val_ratio=0.5, test_ratio=0.0,
        seed=0, group_key_map=group_key_map, output_path=str(out_dir),
    )

    assert "error" not in result
    assert result["group_by"] == "explicit_map"

    import json as _json

    manifest = _json.loads((out_dir / "split_manifest.json").read_text(encoding="utf-8"))
    stem_side = {s: side for side, stems in manifest["splits"].items() for s in stems}
    # Every group's stems (same plot_name) land on the identical side.
    for group_stems in (
        [f"p1_{d}" for d in ("2026-02-01", "2026-03-01")],
        [f"p2_{d}" for d in ("2026-02-01", "2026-03-01")],
    ):
        sides = {stem_side[s] for s in group_stems}
        assert len(sides) == 1, f"{group_stems} split across sides: {stem_side}"


# ── CLI end to end ───────────────────────────────────────────────────────


def test_main_cli_end_to_end(tmp_path: Path, two_plant_csv: Path) -> None:
    dataset_root = tmp_path / "dataset"
    for date, tiepoint in (("2026-02-01", P1_TIEPOINT), ("2026-03-01", P1_TIEPOINT)):
        _write_dataset_stem(dataset_root, date, f"p1_{date}", tiepoint)
    for date, tiepoint in (("2026-02-01", P2_TIEPOINT), ("2026-03-01", P2_TIEPOINT)):
        _write_dataset_stem(dataset_root, date, f"p2_{date}", tiepoint)

    out_dir = tmp_path / "cli_splits_out"
    rc = main([
        str(dataset_root), "--plant-csv", str(two_plant_csv),
        "--train-ratio", "0.5", "--val-ratio", "0.5", "--test-ratio", "0.0",
        "--seed", "0", "--output-path", str(out_dir),
    ])

    assert rc == 0
    assert (out_dir / "split_manifest.json").is_file()


def test_main_cli_reports_refusal_and_nonzero_exit(tmp_path: Path, two_plant_csv: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset_stem(dataset_root, "2026-02-01", "far_stem", FAR_TIEPOINT)

    rc = main([str(dataset_root), "--plant-csv", str(two_plant_csv)])

    assert rc == 1
