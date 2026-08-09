"""uint8 pixels reach the PIL augmentation chain whatever container they decoded from.

to_pil_if_faithful is the single conversion; load_image applies it on the array-backend return
and the tiled dataset applies it on windowed tiles, so an augmentation configured for a run
covers a uint8 GeoTIFF the same as a JPEG, while dtypes PIL has no faithful mode for stay
ndarray and keep the warn-once unaugmented path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tcip_mcp.pipelines.image_utils import load_image, pil_to_tensor


def _grid(height: int, width: int, channels: int, dtype=np.uint8) -> np.ndarray:
    arr = np.zeros((height, width, channels), dtype=dtype)
    for c in range(channels):
        arr[:, :, c] = (np.add.outer(np.arange(height) * (c + 2), np.arange(width))) % 199
    return arr


# ── The helper itself ────────────────────────────────────────────────────


def test_uint8_channel_counts_convert_to_their_pil_mode():
    from tcip_mcp.pipelines.image_utils import to_pil_if_faithful

    for channels, mode in ((1, "L"), (3, "RGB"), (4, "RGBA")):
        arr = _grid(6, 5, channels)
        img = to_pil_if_faithful(arr)
        assert isinstance(img, Image.Image) and img.mode == mode
        back = np.asarray(img)
        assert np.array_equal(back, arr[:, :, 0] if channels == 1 else arr)


def test_unfaithful_inputs_return_unchanged():
    from tcip_mcp.pipelines.image_utils import to_pil_if_faithful

    for arr in (_grid(6, 5, 3, dtype=np.uint16), _grid(6, 5, 2), _grid(6, 5, 5),
                _grid(6, 5, 3).astype(np.float32), np.zeros((6, 5), dtype=np.uint8)):
        assert to_pil_if_faithful(arr) is arr
    pil = Image.new("RGB", (5, 6))
    assert to_pil_if_faithful(pil) is pil


# ── load_image's array-backend return ────────────────────────────────────


def test_load_image_returns_pil_for_uint8_tiffs(tmp_path: Path):
    import tifffile

    rgb = _grid(20, 16, 3)
    tifffile.imwrite(str(tmp_path / "rgb.tif"), rgb, photometric="rgb", rowsperstrip=8)
    got = load_image(tmp_path / "rgb.tif", 3)
    assert isinstance(got, Image.Image) and got.mode == "RGB"
    assert np.array_equal(np.asarray(got), rgb)

    gray = _grid(20, 16, 1)[:, :, 0]
    tifffile.imwrite(str(tmp_path / "gray.tif"), gray, rowsperstrip=8)
    got = load_image(tmp_path / "gray.tif", 1)
    assert isinstance(got, Image.Image) and got.mode == "L"
    assert np.array_equal(np.asarray(got), gray)


def test_load_image_keeps_ndarray_where_pil_has_no_faithful_mode(tmp_path: Path):
    import tifffile

    u16 = _grid(20, 16, 3, dtype=np.uint16)
    tifffile.imwrite(str(tmp_path / "deep.tif"), u16, photometric="rgb", rowsperstrip=8)
    got = load_image(tmp_path / "deep.tif", 3)
    assert isinstance(got, np.ndarray) and got.dtype == np.uint16

    five = _grid(20, 16, 5)
    tifffile.imwrite(str(tmp_path / "five.tif"), five)
    got = load_image(tmp_path / "five.tif", 5)
    assert isinstance(got, np.ndarray) and got.shape == (20, 16, 5)


# ── The augmentation chain applies end to end ────────────────────────────


def _flip_transform():
    def transform(img, target):
        return pil_to_tensor(img.transpose(Image.FLIP_LEFT_RIGHT)), target

    return transform


def _detection_project(tmp_path: Path, arr: np.ndarray):
    import tifffile
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    tifffile.imwrite(str(images_dir / "img0.tif"), arr, photometric="rgb", rowsperstrip=8)
    h, w = arr.shape[:2]
    json_io.write_annotations(str(labels_dir / "img0.json"),
                              [Annotation(subject="catkin", geometry=BBox(2, 2, 10, 10))],
                              w, h, keep_empty=True)
    return images_dir, labels_dir


def test_a_uint8_tiff_trains_augmented_through_detection(tmp_path: Path):
    pytest.importorskip("torch")
    import torch
    from tcip_mcp.pipelines.data.datasets import build_dataset

    arr = _grid(40, 32, 3)
    images_dir, labels_dir = _detection_project(tmp_path, arr)
    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                       subject="catkin", transforms=_flip_transform())
    got, _target = ds[0]
    flipped = torch.from_numpy(arr[:, ::-1].astype(np.float32) / 255.0).permute(2, 0, 1)
    assert torch.equal(got, flipped)


def test_a_uint8_windowed_tile_trains_augmented_through_tiling(tmp_path: Path):
    pytest.importorskip("torch")
    import torch
    from tcip_mcp.pipelines import raster_source
    from tcip_mcp.pipelines.data.datasets import build_dataset

    raster_source.close_source_pool()
    try:
        arr = _grid(96, 96, 3)
        images_dir, labels_dir = _detection_project(tmp_path, arr)
        ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                           subject="catkin", transforms=_flip_transform(),
                           tiling={"enabled": True, "tile_size": 64, "overlap": 0.2})
        got, _target = ds[0]  # tile at (0, 0)
        tile = arr[0:64, 0:64]
        flipped = torch.from_numpy(tile[:, ::-1].astype(np.float32) / 255.0).permute(2, 0, 1)
        assert torch.equal(got, flipped)
        assert ds.source_frames["img0"]["windowed"] is True
    finally:
        raster_source.close_source_pool()
