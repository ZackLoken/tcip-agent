"""Unit tests for the orthomosaic georeferencing and windowed-reading pipeline module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import (
    GeoreferencingError,
    GeoTransform,
    OrthomosaicGeoreference,
    OrthomosaicWindowReader,
    RotatedRasterError,
    UnsupportedRasterLayout,
    pixel_to_native,
    read_geotransform,
)

# UTM zone 15N: real projected CRS the confirmed orthomosaic uses. Central meridian is
# -93 degrees exactly, which gives an independently hand-verifiable reference point below.
UTM_15N_EPSG = 32615
TIEPOINT_NATIVE_X = 500_000.0  # UTM zone 15N's own false easting = the central meridian
TIEPOINT_NATIVE_Y = 4_800_000.0
PIXEL_SCALE = 0.5  # native-CRS units (m) per pixel


def _geokeys(*, model_type: int = 1, projected_epsg: int | None = UTM_15N_EPSG) -> tuple[int, ...]:
    entries: list[int] = [1024, 0, 1, model_type]
    if projected_epsg is not None:
        entries += [3072, 0, 1, projected_epsg]
    num_keys = len(entries) // 4
    return (1, 1, 0, num_keys, *entries)


def _write_geotiff(
    path: Path,
    *,
    pixel_scale: tuple[float, float, float] = (PIXEL_SCALE, PIXEL_SCALE, 0.0),
    tiepoint: tuple[float, ...] = (0.0, 0.0, 0.0, TIEPOINT_NATIVE_X, TIEPOINT_NATIVE_Y, 0.0),
    geokeys: tuple[int, ...] | None = None,
    include_transformation_tag: bool = False,
    shape: tuple[int, int, int] = (5, 5, 4),
) -> None:
    if geokeys is None:
        geokeys = _geokeys()
    arr = np.zeros(shape, dtype=np.uint8)
    extratags = [
        (33550, "d", 3, pixel_scale, False),
        (33922, "d", len(tiepoint), tiepoint, False),
        (34735, "H", len(geokeys), geokeys, False),
    ]
    if include_transformation_tag:
        # Identity-ish 4x4 model transformation matrix (row-major, 16 doubles): present at
        # all is what this module refuses on, regardless of the values it carries.
        transform = (
            pixel_scale[0], 0.0, 0.0, tiepoint[3],
            0.0, -pixel_scale[1], 0.0, tiepoint[4],
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        extratags.append((34264, "d", 16, transform, False))
    tifffile.imwrite(
        str(path),
        arr,
        photometric="rgb",
        extrasamples=["unassalpha"] * (shape[-1] - 3),
        extratags=extratags,
    )


# ── Pixel -> native affine (including the y-axis flip) ────────────────────


def test_pixel_to_native_at_tiepoint_pixel(tmp_path: Path) -> None:
    path = tmp_path / "mosaic.tif"
    _write_geotiff(path)
    transform = read_geotransform(path)
    x, y = pixel_to_native(transform, 0.0, 0.0)
    assert x == pytest.approx(TIEPOINT_NATIVE_X)
    assert y == pytest.approx(TIEPOINT_NATIVE_Y)


def test_pixel_to_native_x_increases_with_pixel_column(tmp_path: Path) -> None:
    path = tmp_path / "mosaic.tif"
    _write_geotiff(path)
    transform = read_geotransform(path)
    x, y = pixel_to_native(transform, 10.0, 0.0)
    assert x == pytest.approx(TIEPOINT_NATIVE_X + 10 * PIXEL_SCALE)
    assert y == pytest.approx(TIEPOINT_NATIVE_Y)


def test_pixel_to_native_y_flips_with_pixel_row(tmp_path: Path) -> None:
    """A pixel further down the raster (larger row) must resolve to a *smaller* northing:
    GeoTIFF rows increase downward while northing increases upward. Two pixels on the same
    column but different rows must disagree in the direction a symmetric bug wouldn't catch."""
    path = tmp_path / "mosaic.tif"
    _write_geotiff(path)
    transform = read_geotransform(path)
    _x_top, y_top = pixel_to_native(transform, 0.0, 0.0)
    _x_bottom, y_bottom = pixel_to_native(transform, 0.0, 20.0)
    assert y_bottom == pytest.approx(TIEPOINT_NATIVE_Y - 20 * PIXEL_SCALE)
    assert y_bottom < y_top


# ── EPSG extraction ─────────────────────────────────────────────────────


def test_epsg_read_from_geokey(tmp_path: Path) -> None:
    path = tmp_path / "mosaic.tif"
    _write_geotiff(path)
    transform = read_geotransform(path)
    assert transform.epsg == UTM_15N_EPSG


def test_epsg_missing_projected_cs_key_refuses(tmp_path: Path) -> None:
    path = tmp_path / "mosaic.tif"
    _write_geotiff(path, geokeys=_geokeys(projected_epsg=None))
    with pytest.raises(GeoreferencingError):
        read_geotransform(path)


def test_non_projected_crs_refuses(tmp_path: Path) -> None:
    path = tmp_path / "mosaic.tif"
    # GTModelTypeGeoKey = 2 (Geographic), not 1 (Projected).
    _write_geotiff(path, geokeys=_geokeys(model_type=2))
    with pytest.raises(GeoreferencingError):
        read_geotransform(path)


def test_missing_geokey_directory_refuses(tmp_path: Path) -> None:
    path = tmp_path / "mosaic.tif"
    arr = np.zeros((5, 5, 4), dtype=np.uint8)
    tifffile.imwrite(
        str(path),
        arr,
        photometric="rgb",
        extrasamples=["unassalpha"],
        extratags=[
            (33550, "d", 3, (PIXEL_SCALE, PIXEL_SCALE, 0.0), False),
            (33922, "d", 6, (0.0, 0.0, 0.0, TIEPOINT_NATIVE_X, TIEPOINT_NATIVE_Y, 0.0), False),
        ],
    )
    with pytest.raises(GeoreferencingError):
        read_geotransform(path)


# ── Rotation refusal ────────────────────────────────────────────────────


def test_model_transformation_tag_refuses(tmp_path: Path) -> None:
    path = tmp_path / "mosaic_rotated.tif"
    _write_geotiff(path, include_transformation_tag=True)
    with pytest.raises(RotatedRasterError):
        read_geotransform(path)


# ── UTM -> WGS84 reprojection ───────────────────────────────────────────


def test_pixel_to_wgs84_matches_independent_pyproj_transform(tmp_path: Path) -> None:
    import pyproj

    path = tmp_path / "mosaic.tif"
    _write_geotiff(path)
    georef = OrthomosaicGeoreference.from_file(path)
    lat, lon = georef.pixel_to_wgs84(30.0, 40.0)

    # A second, independently-constructed Transformer: exercises whether OrthomosaicGeoreference
    # wired the EPSG code, argument order, and lat/lon swap correctly, not whether pyproj itself
    # is correct.
    reference = pyproj.Transformer.from_crs(f"EPSG:{UTM_15N_EPSG}", "EPSG:4326", always_xy=True)
    native_x, native_y = pixel_to_native(read_geotransform(path), 30.0, 40.0)
    expected_lon, expected_lat = reference.transform(native_x, native_y)

    assert lat == pytest.approx(expected_lat)
    assert lon == pytest.approx(expected_lon)


def test_pixel_to_wgs84_central_meridian_known_reference(tmp_path: Path) -> None:
    """UTM zone 15N's central meridian is exactly -93 degrees longitude; a pixel resolving to
    native easting 500000 (the zone's false easting) must land on that meridian regardless of
    northing, a hand-verifiable check independent of trusting pyproj's own round-trip."""
    path = tmp_path / "mosaic.tif"
    _write_geotiff(path)
    georef = OrthomosaicGeoreference.from_file(path)
    _lat, lon = georef.pixel_to_wgs84(0.0, 0.0)  # tiepoint pixel -> native (500000, 4800000)
    assert lon == pytest.approx(-93.0, abs=1e-6)


# ── Windowed reads ──────────────────────────────────────────────────────


def _distinctive_array(height: int, width: int, channels: int = 3) -> np.ndarray:
    """Each pixel encodes its own row/col so a returned sub-array can be checked exactly."""
    arr = np.zeros((height, width, channels), dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            arr[row, col, 0] = row % 256
            arr[row, col, 1] = col % 256
            if channels > 2:
                arr[row, col, 2] = (row + col) % 256
    return arr


def _write_striped_tiff(path: Path, arr: np.ndarray, *, rowsperstrip: int) -> None:
    extrasamples = ["unassalpha"] * (arr.shape[-1] - 3) if arr.shape[-1] > 3 else None
    kwargs = {"photometric": "rgb", "rowsperstrip": rowsperstrip}
    if extrasamples:
        kwargs["extrasamples"] = extrasamples
    tifffile.imwrite(str(path), arr, **kwargs)


def test_read_window_full_extent_matches_source(tmp_path: Path) -> None:
    arr = _distinctive_array(23, 17)
    path = tmp_path / "win.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)
    with OrthomosaicWindowReader(path) as reader:
        assert reader.height == 23
        assert reader.width == 17
        assert reader.num_channels == 3
        window = reader.read_window(0, 23, 0, 17)
    assert np.array_equal(window, arr)


def test_read_window_spans_multiple_internal_strips(tmp_path: Path) -> None:
    arr = _distinctive_array(23, 17)
    path = tmp_path / "win.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)  # strips at rows 0,4,8,12,16,20
    with OrthomosaicWindowReader(path) as reader:
        window = reader.read_window(3, 13, 2, 10)  # spans strip boundaries at rows 4,8,12
    assert np.array_equal(window, arr[3:13, 2:10])


def test_read_window_partial_edge_window(tmp_path: Path) -> None:
    arr = _distinctive_array(23, 17)  # 23 rows, rowsperstrip=4 -> last strip is a partial 3-row strip
    path = tmp_path / "win.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)
    with OrthomosaicWindowReader(path) as reader:
        window = reader.read_window(19, 23, 10, 17)
    assert np.array_equal(window, arr[19:23, 10:17])


def test_read_window_out_of_bounds_raises(tmp_path: Path) -> None:
    arr = _distinctive_array(10, 10)
    path = tmp_path / "win.tif"
    _write_striped_tiff(path, arr, rowsperstrip=4)
    with OrthomosaicWindowReader(path) as reader:
        with pytest.raises(ValueError):
            reader.read_window(0, 11, 0, 10)


def test_window_reader_refuses_tiled_tiff(tmp_path: Path) -> None:
    arr = _distinctive_array(64, 64)
    path = tmp_path / "tiled.tif"
    tifffile.imwrite(str(path), arr, photometric="rgb", tile=(16, 16))
    with pytest.raises(UnsupportedRasterLayout):
        OrthomosaicWindowReader(path)


# ── GeoTransform composes with plain floats (no bespoke point type) ─────


def test_pixel_to_native_returns_plain_floats(tmp_path: Path) -> None:
    path = tmp_path / "mosaic.tif"
    _write_geotiff(path)
    transform = read_geotransform(path)
    assert isinstance(transform, GeoTransform)
    x, y = pixel_to_native(transform, 5.0, 5.0)
    assert isinstance(x, float)
    assert isinstance(y, float)


# ── Windowed tiled inference: agrees with the full-array predict_tiled path ──

TILE = 32


def _windowed_multiband_tiff(path: Path, *, height: int = 96, width: int = 96,
                              channels: int = 4, rowsperstrip: int = 12) -> np.ndarray:
    """A small multi-band raster with real pixel content (not all-zero, so a from-scratch
    detector's convolutions see something), decodable by both ``load_multiband`` (the full-array
    path) and :class:`OrthomosaicWindowReader` (the windowed path): strip-based, contiguous
    samples, no compression.
    """
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(height, width, channels), dtype=np.uint8)
    tifffile.imwrite(str(path), arr, rowsperstrip=rowsperstrip)
    return arr


def _bespoke_detection_checkpoint(tmp_path: Path, raster_path: Path, *, in_chans: int,
                                  tile_size: int = TILE) -> str:
    """A real bespoke detector checkpoint for a raster with ``in_chans`` bands.

    At ``in_chans != 3``, ``build_detector`` refuses without per-band ``image_mean``/``image_std``
    (torchvision's 3-element ImageNet default doesn't describe an N-band raster); derived here from
    ``raster_path`` itself via ``derivations.band_normalization_stats``, mirroring how a real
    multispectral model_source is built, not a pinned placeholder.
    """
    import torch

    from tcip_mcp.pipelines.derivations import band_normalization_stats
    from tcip_mcp.pipelines.model_build import build_model

    builder_kwargs = {"num_classes": 1, "in_chans": in_chans,
                       "min_size": tile_size, "max_size": tile_size * 2}
    if in_chans != 3:
        stats = band_normalization_stats([str(raster_path)], in_chans)
        assert stats is not None
        mean, std = stats
        builder_kwargs["image_mean"] = mean
        builder_kwargs["image_std"] = std

    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": builder_kwargs, "task": "detection", "in_chans": in_chans}
    model = build_model({"model_source": model_source})
    ckpt = tmp_path / "model_best.pt"
    torch.save({"model_source": model_source, "model_state_dict": model.state_dict()}, str(ckpt))
    return str(ckpt)


def test_predict_tiled_from_reader_matches_full_array_predict_tiled(tmp_path: Path) -> None:
    """The load-bearing correctness check: the windowed-read tiling path and the existing
    full-array ``predict_tiled`` path must produce bit-identical full-mosaic-pixel-space
    detections for the same checkpoint and the same raster, not merely both run without error.
    """
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    path = tmp_path / "mosaic.tif"
    arr = _windowed_multiband_tiff(path)
    ckpt = _bespoke_detection_checkpoint(tmp_path, path, in_chans=arr.shape[-1])

    full = GenericPredictor(ckpt, device="cpu", score_threshold=0.0)
    full_result = full.predict_tiled(str(path), tile_size=TILE, overlap=0.2)

    windowed = GenericPredictor(ckpt, device="cpu", score_threshold=0.0)
    with OrthomosaicWindowReader(path) as reader:
        win_result = windowed.predict_tiled_from_reader(
            reader, tile_size=TILE, overlap=0.2, source_label=str(path))

    assert win_result["width"] == full_result["width"] == arr.shape[1]
    assert win_result["height"] == full_result["height"] == arr.shape[0]
    assert win_result["tiles"] == full_result["tiles"] > 1  # really tiled, not one pass
    assert win_result["image"] == str(path)
    np.testing.assert_allclose(win_result["boxes"], full_result["boxes"], rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(win_result["scores"], full_result["scores"], rtol=1e-5, atol=1e-6)
    assert win_result["labels"] == full_result["labels"]


def test_predict_tiled_from_reader_channel_mismatch_refuses() -> None:
    """A model's declared ``in_chans`` disagreeing with the raster's own band count must refuse
    rather than silently truncate/pad the band count the model was trained on. A bare predictor
    (no real checkpoint) is enough: the refusal happens before any tile is read or any forward
    pass runs, mirroring ``test_instance_seg_masks.py``'s own ``_bare_predictor`` pattern for a
    rail check that doesn't need a real model.
    """
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    class _FakeReader:
        height, width, num_channels = 64, 64, 5

        def read_window(self, y0, y1, x0, x1):
            raise AssertionError("a channel mismatch must refuse before any tile is read")

    p = GenericPredictor.__new__(GenericPredictor)
    p.task = "detection"
    p.score_threshold = 0.0
    p.max_dets = None
    p.in_chans = 3

    with pytest.raises(ValueError, match="channel"):
        p.predict_tiled_from_reader(_FakeReader())


def test_predict_tiled_from_reader_refuses_instance_seg_masks() -> None:
    """The same instance_seg mask-through-tiling refusal ``predict_tiled`` already has (masks
    can't yet cross ``reconstruct_core``/``global_nms``/``global_merge``) must hold for the
    windowed-read path too, for the same reason; not silently dropping masks either way."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    class _FakeReader:
        height, width, num_channels = 64, 64, 3

        def read_window(self, y0, y1, x0, x1):
            raise AssertionError("the mask refusal must happen before any tile is read")

    p = GenericPredictor.__new__(GenericPredictor)
    p.task = "instance_seg"
    p.score_threshold = 0.0
    p.max_dets = None
    p.in_chans = 3
    p.model_source = None

    with pytest.raises(NotImplementedError, match="mask"):
        p.predict_tiled_from_reader(_FakeReader())

    # The rail admits valid work: the same predictor with require_masks=False reaches past the
    # refusal (into the real tile loop, which then fails on the fake reader's own AssertionError,
    # proving it got there rather than raising NotImplementedError again).
    with pytest.raises(AssertionError):
        p.predict_tiled_from_reader(_FakeReader(), require_masks=False)


def test_predict_tiled_from_reader_refuses_non_detection_task() -> None:
    """Unlike ``predict_tiled``, there is no untiled ``predict()`` fallback for a windowed
    reader (the whole point is the raster can't be decoded whole), so a non-detection task must
    refuse outright rather than attempt one."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    class _FakeReader:
        height, width, num_channels = 64, 64, 3

        def read_window(self, y0, y1, x0, x1):
            raise AssertionError("a non-detection task must refuse before any tile is read")

    p = GenericPredictor.__new__(GenericPredictor)
    p.task = "classification"
    p.score_threshold = 0.0
    p.max_dets = None
    p.in_chans = 3

    with pytest.raises(ValueError, match="detection"):
        p.predict_tiled_from_reader(_FakeReader())


# ── Per-detection plant assignment ───────────────────────────────────────


def _write_plant_csv(path: Path, rows: list[dict]) -> None:
    import csv

    fieldnames = ["plot_name", "accession_name", "plot_number", "row_number", "col_number",
                  "WGS84_centroid_y", "WGS84_centroid_x"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plant_grid_csv(tmp_path: Path, georef: OrthomosaicGeoreference,
                    plant_pixels: list[tuple[float, float]]) -> Path:
    rows = []
    for i, (px, py) in enumerate(plant_pixels):
        lat, lon = georef.pixel_to_wgs84(px, py)
        rows.append({
            "plot_name": f"plot{i}", "accession_name": f"acc{i}",
            "plot_number": i, "row_number": i // 2, "col_number": i % 2,
            "WGS84_centroid_y": lat, "WGS84_centroid_x": lon,
        })
    csv_path = tmp_path / "plants.csv"
    _write_plant_csv(csv_path, rows)
    return csv_path


# A 2x2 plant grid, 40px apart (20 m at PIXEL_SCALE=0.5 m/px): a small but real layout to derive
# grid_pitch_m from, laid out identically to the georeferenced raster the detections resolve
# their own pixel positions against.
_PLANT_PIXELS = [(10.0, 10.0), (10.0, 50.0), (50.0, 10.0), (50.0, 50.0)]


def test_assign_detections_to_plants_maps_near_and_leaves_far_unmapped(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import assign_detections_to_plants
    from tcip_mcp.pipelines.postprocessing.plant_mapping import read_plant_csvs

    path = tmp_path / "mosaic.tif"
    _write_geotiff(path)
    georef = OrthomosaicGeoreference.from_file(path)
    plants = read_plant_csvs([_plant_grid_csv(tmp_path, georef, _PLANT_PIXELS)])
    assert len(plants) == 4

    detections = {
        "boxes": [
            [8.0, 8.0, 12.0, 12.0],              # centroid (10, 10): sits on plant 0
            [3990.0, 3990.0, 4010.0, 4010.0],    # ~2 km away: no plant anywhere near
        ],
        "scores": [0.9, 0.8],
        "labels": [1, 1],
    }
    assignments = assign_detections_to_plants(detections, georef, plants)
    assert len(assignments) == 2

    near, far = assignments
    assert near.detection_index == 0
    assert near.source == "nearest_neighbour"
    assert near.plot_name == "plot0"
    assert near.accession_name == "acc0"
    assert near.distance_m is not None and near.distance_m < 1.0

    assert far.detection_index == 1
    assert far.source == "unmapped"
    assert far.plot_name is None
    assert far.accession_name is None
    assert far.distance_m is not None  # honest distance even when unmapped, never fabricated


def test_assign_detections_to_plants_default_tolerance_is_pitch_derived(tmp_path: Path) -> None:
    """The default ``nn_tolerance_m`` must be this plot's own ``grid_pitch_m(plants) / 6``, the
    same derivation ``plant_mapping.build_mapping`` uses, not the hardcoded ``NN_TOLERANCE_METERS``
    fallback: a detection placed between the two values is mapped only when the tolerance passed
    is the looser fallback, never with the derived (tighter) default.
    """
    from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import assign_detections_to_plants
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        NN_TOLERANCE_METERS, grid_pitch_m, haversine_m, read_plant_csvs,
    )

    path = tmp_path / "mosaic.tif"
    _write_geotiff(path)
    georef = OrthomosaicGeoreference.from_file(path)
    plants = read_plant_csvs([_plant_grid_csv(tmp_path, georef, _PLANT_PIXELS)])

    pitch = grid_pitch_m(plants)
    derived_tol = pitch / 6
    assert 0 < derived_tol < NN_TOLERANCE_METERS  # the scenario must actually tell them apart

    # ~5 m from plant 0 (10 px at 0.5 m/px): farther than the derived tolerance but closer than
    # the old hardcoded 10 m fallback.
    det_px = (20.0, 10.0)
    lat, lon = georef.pixel_to_wgs84(*det_px)
    plant0_lat, plant0_lon = georef.pixel_to_wgs84(*_PLANT_PIXELS[0])
    dist = haversine_m(lat, lon, plant0_lat, plant0_lon)
    assert derived_tol < dist < NN_TOLERANCE_METERS

    detections = {"boxes": [[det_px[0] - 1, det_px[1] - 1, det_px[0] + 1, det_px[1] + 1]]}

    default_result = assign_detections_to_plants(detections, georef, plants)
    assert default_result[0].source == "unmapped"

    explicit_result = assign_detections_to_plants(
        detections, georef, plants, nn_tolerance_m=NN_TOLERANCE_METERS)
    assert explicit_result[0].source == "nearest_neighbour"
    assert explicit_result[0].plot_name == "plot0"


def test_assign_detections_to_plants_no_boxes_returns_empty(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import assign_detections_to_plants
    from tcip_mcp.pipelines.postprocessing.plant_mapping import read_plant_csvs

    path = tmp_path / "mosaic.tif"
    _write_geotiff(path)
    georef = OrthomosaicGeoreference.from_file(path)
    plants = read_plant_csvs([_plant_grid_csv(tmp_path, georef, _PLANT_PIXELS)])

    assert assign_detections_to_plants({"boxes": []}, georef, plants) == []
