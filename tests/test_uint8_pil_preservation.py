"""uint8 pixels reach the PIL augmentation chain whatever container they decoded from.

to_pil_if_faithful is the single conversion; load_image applies it on the array-backend return
and the tiled dataset applies it on windowed tiles, so an augmentation configured for a run
covers a uint8 GeoTIFF the same as a JPEG, while dtypes PIL has no faithful mode for stay
ndarray and keep the warn-once unaugmented path. A 4-channel array only converts to RGBA when
the source's own band_interpretations names the 4th band alpha; with no such signal, or a
signal naming something else (a genuine spectral band), it stays on that same unaugmented path
rather than guess.
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

    for channels, mode in ((1, "L"), (3, "RGB")):
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


# ── The 4-channel alpha-vs-spectral-band ambiguity ───────────────────────


def test_four_channel_converts_to_rgba_only_when_the_source_declares_alpha():
    """RGBA is a faithful *pixel* round-trip for any 4-channel uint8 array, but PIL's
    augmentation chain treats an alpha channel differently than a color one -- correct only when
    the 4th channel really is transparency. band_interpretations (the same GDAL-color-
    interpretation fact raster_source's other consumers already read) is the only signal that
    settles it; with no signal, or a signal naming something other than alpha, the array must
    stay ndarray rather than guess."""
    from tcip_mcp.pipelines.image_utils import to_pil_if_faithful

    arr = _grid(6, 5, 4)

    img = to_pil_if_faithful(arr, band_interpretations=("red", "green", "blue", "alpha"))
    assert isinstance(img, Image.Image) and img.mode == "RGBA"
    assert np.array_equal(np.asarray(img), arr)

    assert to_pil_if_faithful(arr) is arr  # no signal at all (.npy/.npz, a band group)
    assert to_pil_if_faithful(arr, band_interpretations=None) is arr
    # a real signal that names the 4th band something other than alpha (a genuinely spectral one,
    # or a file whose color interpretation is simply undeclared)
    assert to_pil_if_faithful(arr, band_interpretations=("red", "green", "blue", "undefined")) is arr
    assert to_pil_if_faithful(arr, band_interpretations=("gray", "undefined", "undefined", "undefined")) is arr


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


def test_load_image_converts_a_real_alpha_tagged_geotiff_to_rgba(tmp_path: Path):
    """A GDAL-served 4-band GeoTIFF whose 4th band the file itself declares alpha (TIFF
    ExtraSamples=2, unassociated alpha) is the real signal load_image now reads through
    raster_source.GdalSource.band_interpretations, not a guess from the channel count."""
    import tifffile

    rgba = _grid(20, 16, 4)
    tifffile.imwrite(str(tmp_path / "rgba.tif"), rgba, photometric="rgb", rowsperstrip=8,
                     extrasamples=["unassalpha"])
    got = load_image(tmp_path / "rgba.tif", 4)
    assert isinstance(got, Image.Image) and got.mode == "RGBA"
    assert np.array_equal(np.asarray(got), rgba)


def test_load_image_keeps_a_declared_spectral_fourth_band_as_ndarray(tmp_path: Path):
    """A 4-band GeoTIFF whose file tags the 4th band ExtraSamples=0 (unspecified, GDAL's
    'undefined' color interpretation) is a genuine 4th channel, never alpha; load_image must not
    guess it into RGBA, where PIL's augmentation chain would silently leave it untouched."""
    import tifffile

    ms = _grid(20, 16, 4)
    tifffile.imwrite(str(tmp_path / "ms.tif"), ms, photometric="rgb", rowsperstrip=8,
                     extrasamples=["unspecified"])
    got = load_image(tmp_path / "ms.tif", 4)
    assert isinstance(got, np.ndarray)
    assert np.array_equal(got, ms)


def test_load_image_keeps_an_undeclared_fourth_band_npy_as_ndarray(tmp_path: Path):
    """A .npy stack carries no color-interpretation metadata at all: no signal ever exists to
    call its 4th channel alpha, so load_image must never guess it into RGBA."""
    stack = _grid(20, 16, 4)
    np.save(str(tmp_path / "stack.npy"), stack)
    got = load_image(tmp_path / "stack.npy", 4)
    assert isinstance(got, np.ndarray)
    assert np.array_equal(got, stack)


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


def _detection_project(tmp_path: Path, arr: np.ndarray, *, extrasamples: list[str] | None = None):
    import tifffile
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    kwargs = {"photometric": "rgb", "rowsperstrip": 8}
    if extrasamples:
        kwargs["extrasamples"] = extrasamples
    tifffile.imwrite(str(images_dir / "img0.tif"), arr, **kwargs)
    h, w = arr.shape[:2]
    json_io.write_annotations(str(labels_dir / "img0.json"),
                              [Annotation(subject="bud", geometry=BBox(2, 2, 10, 10))],
                              w, h, keep_empty=True)
    return images_dir, labels_dir


def test_a_uint8_tiff_trains_augmented_through_detection(tmp_path: Path):
    pytest.importorskip("torch")
    import torch
    from tcip_mcp.pipelines.data.datasets import build_dataset

    arr = _grid(40, 32, 3)
    images_dir, labels_dir = _detection_project(tmp_path, arr)
    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                       subject="bud", transforms=_flip_transform())
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
                           subject="bud", transforms=_flip_transform(),
                           tiling={"enabled": True, "tile_size": 64, "overlap": 0.2})
        got, _target = ds[0]  # tile at (0, 0)
        tile = arr[0:64, 0:64]
        flipped = torch.from_numpy(tile[:, ::-1].astype(np.float32) / 255.0).permute(2, 0, 1)
        assert torch.equal(got, flipped)
        assert ds.source_frames["img0"]["windowed"] is True
    finally:
        raster_source.close_source_pool()


def test_a_declared_alpha_windowed_tile_trains_augmented(tmp_path: Path):
    """The windowed tile path reads the same real band_interpretations signal load_image does:
    a 4-band GeoTIFF whose file declares its 4th band alpha still gets the PIL augmentation
    chain, byte-identical to the whole-decode case above."""
    pytest.importorskip("torch")
    import torch
    from tcip_mcp.pipelines import raster_source
    from tcip_mcp.pipelines.data.datasets import build_dataset

    raster_source.close_source_pool()
    try:
        arr = _grid(96, 96, 4)
        images_dir, labels_dir = _detection_project(tmp_path, arr, extrasamples=["unassalpha"])
        ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                           subject="bud", num_channels=4, transforms=_flip_transform(),
                           tiling={"enabled": True, "tile_size": 64, "overlap": 0.2})
        got, _target = ds[0]  # tile at (0, 0)
        tile = arr[0:64, 0:64]
        flipped = torch.from_numpy(tile[:, ::-1].astype(np.float32) / 255.0).permute(2, 0, 1)
        assert torch.equal(got, flipped)
    finally:
        raster_source.close_source_pool()


def test_a_declared_spectral_fourth_band_windowed_tile_trains_unaugmented(tmp_path: Path):
    """A 4-band GeoTIFF whose file declares its 4th band something other than alpha (a genuine
    spectral band) must not be guessed into RGBA: the transform never runs (BaseImageDataset's
    own PIL-only skip, the same path uint16/5-band tiles already take), so the served tile is
    the tile's own raw, unflipped pixels."""
    pytest.importorskip("torch")
    import torch
    from tcip_mcp.pipelines import raster_source
    from tcip_mcp.pipelines.data.datasets import build_dataset

    raster_source.close_source_pool()
    try:
        arr = _grid(96, 96, 4)
        images_dir, labels_dir = _detection_project(tmp_path, arr, extrasamples=["unspecified"])
        ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                           subject="bud", num_channels=4, transforms=_flip_transform(),
                           tiling={"enabled": True, "tile_size": 64, "overlap": 0.2})
        got, _target = ds[0]  # tile at (0, 0)
        tile = arr[0:64, 0:64]
        unflipped = torch.from_numpy(tile.astype(np.float32) / 255.0).permute(2, 0, 1)
        assert torch.equal(got, unflipped)
    finally:
        raster_source.close_source_pool()
