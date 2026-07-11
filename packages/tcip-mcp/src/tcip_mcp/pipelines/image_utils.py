"""Shared image utilities for the composable ML pipeline (channel-aware)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


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
        try:
            import tifffile
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Reading multi-band GeoTIFF needs tifffile (pip install tifffile)."
            ) from exc
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
