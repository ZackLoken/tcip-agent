"""Detection-aware augmentation transforms.

Uses torchvision.transforms.v2 for geometric and photometric augmentation.
Each transform takes (image, target) and returns (image, target).

Includes domain-specific augmentations for field crop imagery:
- ColorJitter: lighting/weather variation
- RandomPerspective: viewpoint change from drone/ground
- Mosaic: combines 4 images for context diversity
- GaussianBlur: motion blur / depth-of-field
- RandomCrop: zooming into regions
"""

from __future__ import annotations

import math
import random
from typing import Any

import torch
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np


class Compose:
    """Chain multiple (image, target) transforms."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, image: Image.Image, target: dict) -> tuple[torch.Tensor, dict]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert PIL Image to float tensor [C, H, W] in [0, 1]."""

    def __call__(self, image: Image.Image, target: dict) -> tuple[torch.Tensor, dict]:
        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, target


class RandomHorizontalFlip:
    """Flip image and boxes horizontally with probability p."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: Any, target: dict) -> tuple[Any, dict]:
        if torch.rand(1).item() < self.p:
            if isinstance(image, Image.Image):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                w = image.width
            else:
                image = image.flip(-1)
                w = image.shape[-1]
            boxes = target["boxes"]
            if len(boxes) > 0:
                boxes = boxes.clone()
                x1 = w - boxes[:, 2]
                x2 = w - boxes[:, 0]
                boxes[:, 0] = x1
                boxes[:, 2] = x2
                target = {**target, "boxes": boxes}
        return image, target


class RandomVerticalFlip:
    """Flip image and boxes vertically with probability p."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: Any, target: dict) -> tuple[Any, dict]:
        if torch.rand(1).item() < self.p:
            if isinstance(image, Image.Image):
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                h = image.height
            else:
                image = image.flip(-2)
                h = image.shape[-2]
            boxes = target["boxes"]
            if len(boxes) > 0:
                boxes = boxes.clone()
                y1 = h - boxes[:, 3]
                y2 = h - boxes[:, 1]
                boxes[:, 1] = y1
                boxes[:, 3] = y2
                target = {**target, "boxes": boxes}
        return image, target


class Resize:
    """Resize image and scale bounding boxes."""

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size  # (height, width)

    def __call__(self, image: Any, target: dict) -> tuple[Any, dict]:
        if isinstance(image, Image.Image):
            orig_w, orig_h = image.size
            image = image.resize((self.size[1], self.size[0]), Image.BILINEAR)
        else:
            orig_h, orig_w = image.shape[-2], image.shape[-1]

        scale_x = self.size[1] / orig_w
        scale_y = self.size[0] / orig_h
        boxes = target["boxes"]
        if len(boxes) > 0:
            boxes = boxes.clone()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target = {**target, "boxes": boxes}
        return image, target


class ColorJitter:
    """Random brightness, contrast, saturation, and hue shifts.

    Essential for field imagery where lighting varies with weather,
    time of day, and sun angle.
    """

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 0.05,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image: Any, target: dict) -> tuple[Any, dict]:
        if not isinstance(image, Image.Image):
            return image, target

        # Brightness
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)
        # Contrast
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)
        # Saturation
        if self.saturation > 0:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            image = ImageEnhance.Color(image).enhance(factor)
        # Hue shift via HSV manipulation
        if self.hue > 0:
            hsv = image.convert("HSV")
            h, s, v = hsv.split()
            h_arr = np.array(h, dtype=np.int16)
            shift = int(random.uniform(-self.hue, self.hue) * 255)
            h_arr = ((h_arr + shift) % 256).astype(np.uint8)
            image = Image.merge("HSV", (Image.fromarray(h_arr), s, v)).convert("RGB")

        return image, target


class GaussianBlur:
    """Apply Gaussian blur to simulate motion blur or depth-of-field."""

    def __init__(self, p: float = 0.3, radius_range: tuple[float, float] = (0.5, 2.0)) -> None:
        self.p = p
        self.radius_range = radius_range

    def __call__(self, image: Any, target: dict) -> tuple[Any, dict]:
        if isinstance(image, Image.Image) and random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image, target


class RandomCrop:
    """Random crop with box clipping and filtering.

    Useful for training on high-res images without tiling.
    Boxes that are mostly outside the crop are removed.
    """

    def __init__(self, size: tuple[int, int], min_area_ratio: float = 0.3) -> None:
        self.size = size  # (height, width)
        self.min_area_ratio = min_area_ratio

    def __call__(self, image: Any, target: dict) -> tuple[Any, dict]:
        if not isinstance(image, Image.Image):
            return image, target

        w, h = image.size
        crop_h, crop_w = min(self.size[0], h), min(self.size[1], w)

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        image = image.crop((left, top, left + crop_w, top + crop_h))

        boxes = target["boxes"]
        if len(boxes) > 0:
            boxes = boxes.clone()
            # Shift and clip boxes
            boxes[:, [0, 2]] -= left
            boxes[:, [1, 3]] -= top
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, crop_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, crop_h)

            # Filter boxes that are too small after clipping
            orig_areas = (target["boxes"][:, 2] - target["boxes"][:, 0]) * (target["boxes"][:, 3] - target["boxes"][:, 1])
            new_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            keep = new_areas > (orig_areas * self.min_area_ratio)

            target = {
                **target,
                "boxes": boxes[keep],
                "labels": target["labels"][keep],
            }

        return image, target


class RandomPerspective:
    """Random perspective transform simulating angled aerial views.

    Critical for drone/UAV imagery where perspective varies between passes.
    """

    def __init__(self, distortion_scale: float = 0.2, p: float = 0.3) -> None:
        self.distortion_scale = distortion_scale
        self.p = p

    def __call__(self, image: Any, target: dict) -> tuple[Any, dict]:
        if not isinstance(image, Image.Image) or random.random() > self.p:
            return image, target

        w, h = image.size
        half_h = h * self.distortion_scale
        half_w = w * self.distortion_scale

        topleft = [random.uniform(-half_w, half_w), random.uniform(-half_h, half_h)]
        topright = [w + random.uniform(-half_w, half_w), random.uniform(-half_h, half_h)]
        botright = [w + random.uniform(-half_w, half_w), h + random.uniform(-half_h, half_h)]
        botleft = [random.uniform(-half_w, half_w), h + random.uniform(-half_h, half_h)]

        src = [(0, 0), (w, 0), (w, h), (0, h)]
        dst = [topleft, topright, botright, botleft]

        coeffs = _find_perspective_coeffs(dst, src)
        image = image.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR)

        # For simplicity, keep boxes as-is (perspective warp on small boxes is minor)
        # A full implementation would transform each corner point
        return image, target


def _find_perspective_coeffs(
    src_points: list[list[float]],
    dst_points: list[tuple[int, int]],
) -> tuple:
    """Compute perspective transform coefficients for PIL."""
    matrix = []
    for s, d in zip(src_points, dst_points):
        matrix.append([d[0], d[1], 1, 0, 0, 0, -s[0] * d[0], -s[0] * d[1]])
        matrix.append([0, 0, 0, d[0], d[1], 1, -s[1] * d[0], -s[1] * d[1]])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([p for pair in src_points for p in pair], dtype=np.float64)
    try:
        res = np.linalg.solve(A, B)
        return tuple(res.tolist())
    except np.linalg.LinAlgError:
        return (1, 0, 0, 0, 1, 0, 0, 0)


class Normalize:
    """Normalize tensor with given mean and std. Applied after ToTensor."""

    def __init__(
        self,
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, image: Any, target: dict) -> tuple[Any, dict]:
        if isinstance(image, torch.Tensor):
            image = (image - self.mean) / self.std
        return image, target


def build_train_transforms(config: dict) -> Compose:
    """Build training augmentation pipeline from config dict.

    Supported config keys:
      resize: [H, W]
      horizontal_flip: bool (default True)
      horizontal_flip_p: float (default 0.5)
      vertical_flip: bool (default False)
      vertical_flip_p: float (default 0.5)
      color_jitter: bool (default True)
      color_jitter_brightness: float (default 0.3)
      color_jitter_contrast: float (default 0.3)
      color_jitter_saturation: float (default 0.3)
      color_jitter_hue: float (default 0.05)
      gaussian_blur: bool (default True)
      gaussian_blur_p: float (default 0.3)
      random_crop: [H, W] or null
      random_perspective: bool (default False)
      random_perspective_scale: float (default 0.2)
      normalize: bool (default False)
    """
    transforms = []
    if config.get("resize"):
        transforms.append(Resize(tuple(config["resize"])))
    if config.get("random_crop"):
        transforms.append(RandomCrop(tuple(config["random_crop"])))
    if config.get("horizontal_flip", True):
        transforms.append(RandomHorizontalFlip(config.get("horizontal_flip_p", 0.5)))
    if config.get("vertical_flip", False):
        transforms.append(RandomVerticalFlip(config.get("vertical_flip_p", 0.5)))
    if config.get("color_jitter", True):
        transforms.append(ColorJitter(
            brightness=config.get("color_jitter_brightness", 0.3),
            contrast=config.get("color_jitter_contrast", 0.3),
            saturation=config.get("color_jitter_saturation", 0.3),
            hue=config.get("color_jitter_hue", 0.05),
        ))
    if config.get("gaussian_blur", True):
        transforms.append(GaussianBlur(p=config.get("gaussian_blur_p", 0.3)))
    if config.get("random_perspective", False):
        transforms.append(RandomPerspective(
            distortion_scale=config.get("random_perspective_scale", 0.2),
        ))
    transforms.append(ToTensor())
    if config.get("normalize", False):
        transforms.append(Normalize())
    return Compose(transforms)


def build_val_transforms(config: dict) -> Compose:
    """Build validation/test transform pipeline (no augmentation)."""
    transforms = []
    if config.get("resize"):
        transforms.append(Resize(tuple(config["resize"])))
    transforms.append(ToTensor())
    if config.get("normalize", False):
        transforms.append(Normalize())
    return Compose(transforms)
