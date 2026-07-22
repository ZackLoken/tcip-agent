"""Shared image utilities for the composable ML pipeline (channel-aware)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def image_dimensions(path: str | Path, num_channels: int = 3) -> tuple[int, int]:
    """``(width, height)`` as ``load_image`` will decode it, without decoding pixels where possible.

    ``tcip_annotation.get_image_dimensions`` reads through PIL, which is right for the photographic
    formats it was written for and wrong for a multi-band raster — PIL reports a 5-band 40x24
    GeoTIFF as 5x40. Labels clipped against one frame and tiles cropped from another displace every
    box silently, so anything that measures a frame it will later decode must route the same way
    ``load_image`` does.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if num_channels in (1, 3, 4) and ext not in (".npy", ".npz", ".tif", ".tiff"):
        from tcip_annotation.utils import get_image_dimensions

        return get_image_dimensions(str(path))  # header-only, EXIF-aware
    if ext in (".tif", ".tiff"):
        try:
            import tifffile

            with tifffile.TiffFile(str(path)) as tif:
                # The series shape, not pages[0]: a channel-last TIFF stores each row-block as its
                # own page, so pages[0] of a 24x40x5 raster is (40, 5).
                shape = tif.series[0].shape  # header-only
            if len(shape) == 2:
                return int(shape[1]), int(shape[0])
            if len(shape) == 3:
                # Same channel-first heuristic load_multiband applies, so both agree.
                if shape[0] == num_channels and shape[2] != num_channels:
                    return int(shape[2]), int(shape[1])
                return int(shape[1]), int(shape[0])
        except Exception:  # noqa: BLE001 — fall through to a full read rather than guess
            pass
    arr = load_multiband(path, num_channels)
    return int(arr.shape[1]), int(arr.shape[0])


def crop_pad_tile(img, x: int, y: int, tile_size: int, w: int, h: int):
    """Crop a ``tile_size`` window at (x, y) and zero-pad short (edge) tiles.

    Channel-generic: PIL for 1/3/4-channel images, numpy ``[H, W, C]`` for multi-band rasters
    (which have no ``.crop``). Shared by the training tiler and the inference tiler so the two ends
    of the reproduce-a-number chain cannot crop differently.
    """
    x2, y2 = min(x + tile_size, w), min(y + tile_size, h)
    if isinstance(img, Image.Image):
        crop = img.crop((x, y, x2, y2))
        if crop.size != (tile_size, tile_size):
            padded = Image.new(img.mode, (tile_size, tile_size))  # 0-fill for the image's mode
            padded.paste(crop, (0, 0))
            crop = padded
        return crop
    crop = img[y:y2, x:x2]
    ph, pw = tile_size - crop.shape[0], tile_size - crop.shape[1]
    if ph or pw:
        pad_width = [(0, ph), (0, pw)] + ([(0, 0)] if crop.ndim == 3 else [])
        crop = np.pad(crop, pad_width, mode="constant")
    return crop


def pil_to_tensor(img) -> torch.Tensor:
    """Convert a PIL Image or H×W[×C] array to a float32 ``[C, H, W]`` tensor in ``[0, 1]``.

    Channel-aware: a 2-D grayscale array becomes ``[1, H, W]``; any channel count is
    supported. Integer inputs are scaled by their dtype max (uint8→/255, uint16→/65535);
    float inputs are assumed already normalized.
    """
    arr = np.asarray(img)
    if arr.ndim == 2:  # grayscale [H, W] -> [H, W, 1]
        arr = arr[:, :, None]
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    else:
        arr = arr.astype(np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_image(path: str | Path, num_channels: int = 3):
    """Open an image honoring ``num_channels``.

    Returns a ``PIL.Image`` for 1/3/4-channel raster images (so the PIL augmentation
    pipeline keeps working), or an ``[H, W, C]`` ndarray for multi-band inputs
    (``.npy`` / ``.npz`` / multi-band GeoTIFF). An RGB file requested as 1 channel is
    converted to grayscale; as 3, kept RGB.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if num_channels in (1, 3, 4) and ext not in (".npy", ".npz", ".tif", ".tiff"):
        mode = {1: "L", 3: "RGB", 4: "RGBA"}[num_channels]
        # EXIF-orient before convert so the returned frame matches get_image_dimensions()
        # (both apply auto_orient_image). Labels are authored in this upright frame; without
        # this the loader would denormalize upright coords against the raw sensor frame and
        # scatter every box (Orientation-6 JPEGs differ 5712×4284 ↔ 4284×5712).
        from tcip_annotation.utils import auto_orient_image

        return auto_orient_image(Image.open(path)).convert(mode)
    # >4 channels, or a numpy/GeoTIFF container -> multi-band array.
    return load_multiband(path, num_channels)


def load_multiband(path: str | Path, num_channels: int) -> np.ndarray:
    """Load a multi-band image as ``[H, W, C]`` (NPY/NPZ natively; GeoTIFF via tifffile)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(str(path))
    elif ext == ".npz":
        npz = np.load(str(path))
        arr = npz[npz.files[0]]
    elif ext in (".tif", ".tiff"):
        import tifffile

        arr = tifffile.imread(str(path))
    else:
        raise ValueError(
            f"Cannot load a {num_channels}-channel image from '{ext}'. "
            "Use .npy/.npz or a multi-band GeoTIFF (.tif/.tiff)."
        )
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.ndim == 3 and arr.shape[0] == num_channels and arr.shape[2] != num_channels:
        arr = np.transpose(arr, (1, 2, 0))  # channel-first -> channel-last
    return arr
