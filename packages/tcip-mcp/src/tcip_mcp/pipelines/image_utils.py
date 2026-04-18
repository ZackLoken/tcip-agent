"""Shared image utilities for the composable ML pipeline."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a float32 [C, H, W] tensor in [0, 1]."""
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)
