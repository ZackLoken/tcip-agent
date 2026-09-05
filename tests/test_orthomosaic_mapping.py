"""Orthomosaic georeferencing, per-detection plant assignment, and the tiled inference pass that
sources its tiles from a windowed raster read (the reading layer itself: test_raster_source.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import (
    GeoreferencingError,
    GeoTransform,
    OrthomosaicGeoreference,
    RotatedRasterError,
    native_to_pixel,
    pixel_to_native,
    plants_in_frame,
    read_geotransform,
)
from tcip_mcp.pipelines.raster_source import open_raster

from tests._geotiff_fixtures import (
    PIXEL_SCALE,
    TIEPOINT_NATIVE_X,
    TIEPOINT_NATIVE_Y,
    UTM_15N_EPSG,
    build_geokeys as _geokeys,
    write_geotiff as _write_geotiff,
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


# ── WGS84 -> pixel: the exact inverse of pixel -> WGS84 ─────────────────


def test_wgs84_to_pixel_matches_independent_pyproj_inverse_at_asymmetric_pixel(tmp_path: Path) -> None:
    """An asymmetric pixel (distinct nonzero x and y, off the tiepoint): the round-trip and an
    independently-built pyproj inverse transform must both land back on it."""
    import pyproj

    path = tmp_path / "mosaic.tif"
    _write_geotiff(path)
    georef = OrthomosaicGeoreference.from_file(path)
    px, py = 30.0, 47.0
    lat, lon = georef.pixel_to_wgs84(px, py)

    reference = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{UTM_15N_EPSG}", always_xy=True)
    expected_native_x, expected_native_y = reference.transform(lon, lat)
    transform = read_geotransform(path)
    expected_px, expected_py = native_to_pixel(transform, expected_native_x, expected_native_y)

    result_px, result_py = georef.wgs84_to_pixel(lat, lon)
    assert result_px == pytest.approx(expected_px, abs=1e-6)
    assert result_py == pytest.approx(expected_py, abs=1e-6)
    assert result_px == pytest.approx(px, abs=1e-6)
    assert result_py == pytest.approx(py, abs=1e-6)


def test_native_to_pixel_round_trips_with_pixel_to_native_at_asymmetric_pixel(tmp_path: Path) -> None:
    path = tmp_path / "mosaic.tif"
    _write_geotiff(path)
    transform = read_geotransform(path)
    px, py = 30.0, 47.0
    native_x, native_y = pixel_to_native(transform, px, py)
    round_px, round_py = native_to_pixel(transform, native_x, native_y)
    assert round_px == pytest.approx(px)
    assert round_py == pytest.approx(py)


def test_native_to_pixel_zero_pixel_scale_x_refuses_naming_the_scale(tmp_path: Path) -> None:
    path = tmp_path / "mosaic.tif"
    _write_geotiff(path)
    transform = read_geotransform(path)
    degenerate = GeoTransform(**{**transform.__dict__, "pixel_scale_x": 0.0})
    with pytest.raises(ValueError, match="pixel_scale_x"):
        native_to_pixel(degenerate, 500_000.0, 4_800_000.0)


def test_native_to_pixel_zero_pixel_scale_y_refuses_naming_the_scale(tmp_path: Path) -> None:
    path = tmp_path / "mosaic.tif"
    _write_geotiff(path)
    transform = read_geotransform(path)
    degenerate = GeoTransform(**{**transform.__dict__, "pixel_scale_y": 0.0})
    with pytest.raises(ValueError, match="pixel_scale_y"):
        native_to_pixel(degenerate, 500_000.0, 4_800_000.0)


# ── plants_in_frame: the one in-frame partition both attribution regimes share ──


def test_plants_in_frame_partitions_by_the_rasters_own_recorded_dimensions(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.postprocessing.plant_mapping import PlantRecord

    path = tmp_path / "mosaic.tif"
    _write_geotiff(path, width=64, height=64, shape=(64, 64, 3))
    georef = OrthomosaicGeoreference.from_file(path)

    inside_lat, inside_lon = georef.pixel_to_wgs84(10.0, 10.0)
    outside_lat, outside_lon = georef.pixel_to_wgs84(-5.0, 10.0)  # negative column: outside
    plants = [
        PlantRecord(plot_name="in", accession_name="a", plot_number=0, row_number=0,
                   col_number=0, lat=inside_lat, lon=inside_lon),
        PlantRecord(plot_name="out", accession_name="b", plot_number=1, row_number=0,
                   col_number=1, lat=outside_lat, lon=outside_lon),
    ]

    in_frame, outside = plants_in_frame(plants, georef, width=64, height=64)
    assert [p.plot_name for p in in_frame] == ["in"]
    assert [p.plot_name for p in outside] == ["out"]


def test_plants_in_frame_edge_pixel_at_width_or_height_is_outside(tmp_path: Path) -> None:
    """The half-open test: a plant projecting to exactly the raster's own width or height row or
    column is outside, since a valid pixel index runs only 0..width-1 / 0..height-1, on both axes."""
    from tcip_mcp.pipelines.postprocessing.plant_mapping import PlantRecord

    path = tmp_path / "mosaic.tif"
    _write_geotiff(path, width=64, height=64, shape=(64, 64, 3))
    georef = OrthomosaicGeoreference.from_file(path)
    width_edge_lat, width_edge_lon = georef.pixel_to_wgs84(64.0, 10.0)
    height_edge_lat, height_edge_lon = georef.pixel_to_wgs84(10.0, 64.0)
    plants = [
        PlantRecord(plot_name="width_edge", accession_name="a", plot_number=0, row_number=0,
                    col_number=0, lat=width_edge_lat, lon=width_edge_lon),
        PlantRecord(plot_name="height_edge", accession_name="b", plot_number=1, row_number=0,
                    col_number=1, lat=height_edge_lat, lon=height_edge_lon),
    ]

    in_frame, outside = plants_in_frame(plants, georef, width=64, height=64)
    assert in_frame == []
    assert {p.plot_name for p in outside} == {"width_edge", "height_edge"}


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


def _register_checkpoint(tmp_path: Path, ckpt_path: str, *, name: str) -> None:
    """Register a checkpoint the platform's own producer wrote against tmp_path as project root,
    the same root each test's own load_registered_checkpoint call resolves the registry from."""
    from tcip_mcp.tools.model_tools import register_model

    result = register_model(name=name, checkpoint_path=ckpt_path, config={},
                            project_path=str(tmp_path))
    assert "error" not in result, result


def _windowed_multiband_tiff(path: Path, *, height: int = 96, width: int = 96,
                              channels: int = 4, rowsperstrip: int = 12) -> np.ndarray:
    """A small multi-band raster with real pixel content (not all-zero, so a from-scratch
    detector's convolutions see something), decodable by both ``load_multiband`` (the full-array
    path) and the windowed raster layer (``open_raster``): strip-based, contiguous samples, no
    compression.
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
        mean, std, paths_read = stats
        builder_kwargs["image_mean"] = mean
        builder_kwargs["image_std"] = std

    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": builder_kwargs, "task": "detection", "in_chans": in_chans}
    model = build_model({"model_source": model_source})
    ckpt = tmp_path / "model_best.pt"
    torch.save({"model_source": model_source, "model_state_dict": model.state_dict()}, str(ckpt))
    return str(ckpt)


def test_predict_tiled_windowed_source_matches_full_array_predict_tiled(tmp_path: Path) -> None:
    """The load-bearing correctness check: the windowed-read tiling path and the existing
    full-array ``predict_tiled`` path must produce bit-identical full-mosaic-pixel-space
    detections for the same checkpoint and the same raster, not merely both run without error.
    """
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    path = tmp_path / "mosaic.tif"
    arr = _windowed_multiband_tiff(path)
    ckpt = _bespoke_detection_checkpoint(tmp_path, path, in_chans=arr.shape[-1])
    _register_checkpoint(tmp_path, ckpt, name="ortho-detection")

    full = GenericPredictor(load_registered_checkpoint(ckpt, project_path=str(tmp_path)),
                            device="cpu", score_threshold=0.0)
    full_result = full.predict_tiled(str(path), tile_size=TILE, overlap=0.2)

    windowed = GenericPredictor(load_registered_checkpoint(ckpt, project_path=str(tmp_path)),
                                device="cpu", score_threshold=0.0)
    with open_raster(path, arr.shape[-1]) as reader:
        win_result = windowed.predict_tiled(
            reader, tile_size=TILE, overlap=0.2, source_label=str(path))

    assert win_result["width"] == full_result["width"] == arr.shape[1]
    assert win_result["height"] == full_result["height"] == arr.shape[0]
    assert win_result["tiles"] == full_result["tiles"] > 1  # really tiled, not one pass
    assert win_result["image"] == str(path)
    np.testing.assert_allclose(win_result["boxes"], full_result["boxes"], rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(win_result["scores"], full_result["scores"], rtol=1e-5, atol=1e-6)
    assert win_result["labels"] == full_result["labels"]


def _bespoke_instance_seg_checkpoint(tmp_path: Path, *, in_chans: int = 3, tile_size: int = TILE) -> str:
    """A real bespoke Mask R-CNN checkpoint (RGB only, no multispectral norm derivation needed
    for this end-to-end mask-shape test)."""
    import torch

    from tcip_mcp.pipelines.model_build import build_model

    model_source = {"builder": "tests.bespoke_models:build_bespoke_instance_seg",
                    "builder_kwargs": {"num_classes": 1, "in_chans": in_chans,
                                      "min_size": tile_size, "max_size": tile_size * 2},
                    "task": "instance_seg", "in_chans": in_chans}
    model = build_model({"model_source": model_source})
    ckpt = tmp_path / "instance_seg_best.pt"
    torch.save({"model_source": model_source, "model_state_dict": model.state_dict()}, str(ckpt))
    return str(ckpt)


def test_predict_tiled_windowed_source_and_predict_tiled_produce_matching_tiled_masks(tmp_path: Path) -> None:
    """Both tiled entry points carry masks for a multi-tile instance_seg case, in the tiled
    (tile-local-patch + full-image-offset) shape documented on ``predict_tiled``, and the windowed
    path agrees with the full-array path detection-for-detection, masks included."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    path = tmp_path / "mosaic.tif"
    _windowed_multiband_tiff(path, channels=3)
    ckpt = _bespoke_instance_seg_checkpoint(tmp_path)
    _register_checkpoint(tmp_path, ckpt, name="ortho-instance-seg")

    full = GenericPredictor(load_registered_checkpoint(ckpt, project_path=str(tmp_path)),
                            device="cpu", score_threshold=0.0)
    full_result = full.predict_tiled(str(path), tile_size=TILE, overlap=0.2)

    windowed = GenericPredictor(load_registered_checkpoint(ckpt, project_path=str(tmp_path)),
                                device="cpu", score_threshold=0.0)
    with open_raster(path, 3) as reader:
        win_result = windowed.predict_tiled(
            reader, tile_size=TILE, overlap=0.2, source_label=str(path))

    assert "masks" in full_result and "masks" in win_result
    assert len(full_result["masks"]) == len(win_result["masks"]) == full_result["count"]
    for m in full_result["masks"] + win_result["masks"]:
        assert set(m) == {"mask_patch", "offset_x", "offset_y"}
        patch = np.asarray(m["mask_patch"])
        assert patch.shape == (TILE, TILE)  # tile-local, never a full-mosaic-sized array
        assert 0 <= m["offset_x"] <= win_result["width"]
        assert 0 <= m["offset_y"] <= win_result["height"]
    for fm, wm in zip(full_result["masks"], win_result["masks"]):
        assert fm["offset_x"] == wm["offset_x"] and fm["offset_y"] == wm["offset_y"]
        np.testing.assert_allclose(fm["mask_patch"], wm["mask_patch"], rtol=1e-5, atol=1e-6)


def test_predict_tiled_windowed_source_require_masks_false_carries_no_masks_key(tmp_path: Path) -> None:
    """The boxes-only opt-out still works on the windowed path: no ``masks`` key at all, not an
    empty one, mirroring ``predict_tiled``'s own opt-out contract."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    path = tmp_path / "mosaic.tif"
    _windowed_multiband_tiff(path, channels=3)
    ckpt = _bespoke_instance_seg_checkpoint(tmp_path)
    _register_checkpoint(tmp_path, ckpt, name="ortho-instance-seg")

    checkpoint = load_registered_checkpoint(ckpt, project_path=str(tmp_path))
    predictor = GenericPredictor(checkpoint, device="cpu", score_threshold=0.0)
    with open_raster(path, 3) as reader:
        result = predictor.predict_tiled(
            reader, tile_size=TILE, overlap=0.2, require_masks=False)
    assert "masks" not in result
    assert {"boxes", "scores", "labels", "count", "tiles"} <= set(result)


def test_predict_tiled_windowed_source_tiled_mask_polygon_exports_at_correct_offset(tmp_path: Path) -> None:
    """A tiled instance_seg detection's mask round-trips through ``write_predictions_json`` to a
    polygon positioned in full-mosaic pixel space, not left at its tile-local offset."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from tcip_annotation import json_io
    from tcip_annotation.state import Polygon
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json

    path = tmp_path / "mosaic.tif"
    _windowed_multiband_tiff(path, channels=3)
    ckpt = _bespoke_instance_seg_checkpoint(tmp_path)
    _register_checkpoint(tmp_path, ckpt, name="ortho-instance-seg")

    checkpoint = load_registered_checkpoint(ckpt, project_path=str(tmp_path))
    predictor = GenericPredictor(checkpoint, device="cpu", score_threshold=0.0)
    with open_raster(path, 3) as reader:
        result = predictor.predict_tiled(
            reader, tile_size=TILE, overlap=0.2, source_label=str(path))
    assert result["count"] > 0, "fixture assumes at least one surviving detection"

    # A synthetic mask with a clean blob, positioned at a non-zero tile offset, replaces whatever
    # the untrained model actually predicted: the point of this test is the offset plumbing through
    # export, not the (meaningless, from-scratch-weights) mask content itself.
    offset_x, offset_y = result["masks"][0]["offset_x"], result["masks"][0]["offset_y"]
    patch = np.zeros((TILE, TILE), dtype=np.float32)
    patch[4:10, 4:10] = 0.9
    result["masks"][0] = {"mask_patch": patch.tolist(), "offset_x": offset_x, "offset_y": offset_y}

    out = tmp_path / "pred.json"
    write_predictions_json(out, result, subject="leaf", attribute=None)
    anns = json_io.read_annotations(str(out))
    assert isinstance(anns[0].geometry, Polygon)
    xs = [x for ring in anns[0].geometry.rings for x, _ in ring]
    ys = [y for ring in anns[0].geometry.rings for _, y in ring]
    # The blob sits at local [4:10, 4:10]; the exported polygon must be shifted into full-mosaic
    # pixel space by the patch's own offset, not left at its tile-local coordinates.
    assert min(xs) == pytest.approx(4 + offset_x, abs=1.0)
    assert min(ys) == pytest.approx(4 + offset_y, abs=1.0)


def test_predict_tiled_windowed_source_channel_mismatch_refuses() -> None:
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
        p.predict_tiled(_FakeReader())


def test_predict_tiled_windowed_source_reaches_real_tiling_for_instance_seg_with_and_without_masks() -> None:
    """instance_seg masks thread through the windowed-read path the same as the full-array
    ``predict_tiled`` path: neither ``require_masks=True`` (the default, masks collected) nor
    ``require_masks=False`` (the boxes-only opt-out) refuses outright, both reach the real tile
    loop, which then fails on the fake reader's own ``AssertionError`` rather than anything raised
    by ``predict_tiled`` itself."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    class _FakeReader:
        height, width, num_channels = 64, 64, 3

        def read_window(self, y0, y1, x0, x1):
            raise AssertionError("reached the real tile loop")

    p = GenericPredictor.__new__(GenericPredictor)
    p.task = "instance_seg"
    p.score_threshold = 0.0
    p.max_dets = None
    p.in_chans = 3
    p.model_source = None

    with pytest.raises(AssertionError):
        p.predict_tiled(_FakeReader(), tile_size=32)

    with pytest.raises(AssertionError):
        p.predict_tiled(_FakeReader(), tile_size=32, require_masks=False)


def test_predict_tiled_windowed_source_refuses_non_detection_task() -> None:
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
        p.predict_tiled(_FakeReader())


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
