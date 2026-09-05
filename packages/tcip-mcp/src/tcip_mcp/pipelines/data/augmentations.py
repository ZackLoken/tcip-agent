"""Data augmentation transforms for all task types.

Provides composable augmentation transforms that work with both detection
(image, target dict with 'boxes') and classification (image, target dict)
pipelines. Implemented directly on PIL and torch, with no torchvision
dependency.

Usage:
    transforms = build_augmentation(config)
    img_tensor, target = transforms(pil_image, target_dict)
"""

from __future__ import annotations

import random

import torch
from PIL import Image, ImageEnhance, ImageFilter

from tcip_mcp.pipelines.image_utils import pil_to_tensor


class Compose:
    """Chain multiple (image, target) transforms."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, img: Image.Image, target: dict) -> tuple[torch.Tensor | Image.Image, dict]:
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

    def __repr__(self) -> str:
        lines = [f"  {t}" for t in self.transforms]
        return "Compose([\n" + "\n".join(lines) + "\n])"


class ToTensor:
    """Convert PIL Image to tensor (final step)."""

    def __call__(self, img: Image.Image, target: dict) -> tuple[torch.Tensor, dict]:
        return pil_to_tensor(img), target


def _resize_masks(masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Nearest-resize instance ``[N, H, W]`` or semantic ``[H, W]`` masks to ``(w, h)``."""
    if masks.ndim == 2:
        return _resize_masks(masks.unsqueeze(0), size).squeeze(0)
    if len(masks) == 0:
        return masks.new_zeros((0, size[1], size[0]))
    resized = torch.nn.functional.interpolate(
        masks.unsqueeze(1).float(), size=(size[1], size[0]), mode="nearest"
    )
    return resized.squeeze(1).to(masks.dtype)


class RandomHorizontalFlip:
    """Flip image, boxes, and masks horizontally with probability p."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Image.Image, target: dict) -> tuple[Image.Image, dict]:
        if random.random() < self.p:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            w = img.width
            if "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    # [x1, y1, x2, y2] format
                    new_boxes = boxes.clone()
                    new_boxes[:, 0] = w - boxes[:, 2]
                    new_boxes[:, 2] = w - boxes[:, 0]
                    target["boxes"] = new_boxes
            if torch.is_tensor(target.get("masks")):
                # Works for both [N, H, W] instance and [H, W] semantic masks
                target["masks"] = torch.flip(target["masks"], dims=[-1])
        return img, target


class RandomVerticalFlip:
    """Flip image, boxes, and masks vertically with probability p."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Image.Image, target: dict) -> tuple[Image.Image, dict]:
        if random.random() < self.p:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            h = img.height
            if "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    new_boxes = boxes.clone()
                    new_boxes[:, 1] = h - boxes[:, 3]
                    new_boxes[:, 3] = h - boxes[:, 1]
                    target["boxes"] = new_boxes
            if torch.is_tensor(target.get("masks")):
                target["masks"] = torch.flip(target["masks"], dims=[-2])
        return img, target


class ColorJitter:
    """Random brightness, contrast, saturation, hue perturbations."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img: Image.Image, target: dict) -> tuple[Image.Image, dict]:
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)
        if self.saturation > 0:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            img = ImageEnhance.Color(img).enhance(factor)
        # Hue shift via HSV conversion
        if self.hue > 0:
            # Simple hue shift isn't trivial with PIL alone; skip (no-op for now).
            pass
        return img, target


class RandomResizedCrop:
    """Crop a random region and resize to target size. Adjusts boxes and masks accordingly."""

    def __init__(self, size: tuple[int, int] = (640, 640), min_scale: float = 0.5, max_scale: float = 1.0) -> None:
        self.size = size
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img: Image.Image, target: dict) -> tuple[Image.Image, dict]:
        w, h = img.size
        scale = random.uniform(self.min_scale, self.max_scale)
        crop_w = int(w * scale)
        crop_h = int(h * scale)
        x1 = random.randint(0, max(0, w - crop_w))
        y1 = random.randint(0, max(0, h - crop_h))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        img = img.crop((x1, y1, x2, y2)).resize(self.size, Image.Resampling.BILINEAR)

        # Crop + nearest-resize masks in lockstep with the image
        if torch.is_tensor(target.get("masks")):
            target["masks"] = _resize_masks(target["masks"][..., y1:y2, x1:x2], self.size)

        # Adjust boxes
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor):
                # Shift by crop origin
                boxes = boxes.clone()
                boxes[:, 0] = (boxes[:, 0] - x1) * self.size[0] / crop_w
                boxes[:, 1] = (boxes[:, 1] - y1) * self.size[1] / crop_h
                boxes[:, 2] = (boxes[:, 2] - x1) * self.size[0] / crop_w
                boxes[:, 3] = (boxes[:, 3] - y1) * self.size[1] / crop_h
                # Clamp to image bounds
                boxes[:, 0].clamp_(min=0, max=self.size[0])
                boxes[:, 1].clamp_(min=0, max=self.size[1])
                boxes[:, 2].clamp_(min=0, max=self.size[0])
                boxes[:, 3].clamp_(min=0, max=self.size[1])
                # Filter out degenerate boxes
                valid = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
                target["boxes"] = boxes[valid]
                if "labels" in target:
                    target["labels"] = target["labels"][valid]
                masks = target.get("masks")
                if torch.is_tensor(masks) and masks.ndim == 3:
                    target["masks"] = masks[valid]

        return img, target


class GaussianBlur:
    """Apply Gaussian blur with probability p."""

    def __init__(self, p: float = 0.1, radius: float = 2.0) -> None:
        self.p = p
        self.radius = radius

    def __call__(self, img: Image.Image, target: dict) -> tuple[Image.Image, dict]:
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.radius))
        return img, target


class Resize:
    """Resize image to fixed size. Adjusts boxes and masks accordingly."""

    def __init__(self, size: tuple[int, int] = (640, 640)) -> None:
        self.size = size

    def __call__(self, img: Image.Image, target: dict) -> tuple[Image.Image, dict]:
        w, h = img.size
        img = img.resize(self.size, Image.Resampling.BILINEAR)
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor):
                scale_x = self.size[0] / w
                scale_y = self.size[1] / h
                boxes = boxes.clone()
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                target["boxes"] = boxes
        if torch.is_tensor(target.get("masks")):
            target["masks"] = _resize_masks(target["masks"], self.size)
        return img, target


class RandomRotation:
    """Rotate image (and detection boxes/masks) by a uniform angle in
    ``[-degrees, degrees]`` with probability ``p``.

    Nadir/aerial imagery has no canonical "up", so free rotation is a valid
    augmentation. Boxes are rotated as 4 corners about the image center and taken
    as their axis-aligned envelope, then clamped + degenerate-filtered exactly like
    ``RandomResizedCrop`` (filtering ``labels``/``masks`` in lockstep). Semantic
    ``[H, W]`` masks are rotated with nearest resampling. Non-detection targets
    (classification/ordinal/regression) pass through untouched.
    """

    def __init__(self, degrees: float = 180.0, p: float = 1.0) -> None:
        self.degrees = degrees
        self.p = p

    def __call__(self, img: Image.Image, target: dict) -> tuple[Image.Image, dict]:
        if random.random() >= self.p:
            return img, target
        import math

        angle = random.uniform(-self.degrees, self.degrees)
        w, h = img.size
        out = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)

        boxes = target.get("boxes")
        if isinstance(boxes, torch.Tensor) and len(boxes) > 0:
            cx, cy = w / 2.0, h / 2.0
            theta = math.radians(angle)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            new_boxes = []
            for x1, y1, x2, y2 in boxes.tolist():
                xs, ys = [], []
                for px, py in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
                    dx, dy = px - cx, py - cy
                    xs.append(cx + dx * cos_t - dy * sin_t)
                    ys.append(cy + dx * sin_t + dy * cos_t)
                new_boxes.append([min(xs), min(ys), max(xs), max(ys)])
            nb = torch.tensor(new_boxes, dtype=boxes.dtype)
            nb[:, [0, 2]] = nb[:, [0, 2]].clamp(min=0, max=w)
            nb[:, [1, 3]] = nb[:, [1, 3]].clamp(min=0, max=h)
            valid = (nb[:, 2] - nb[:, 0] > 1) & (nb[:, 3] - nb[:, 1] > 1)
            target["boxes"] = nb[valid]
            if "labels" in target and torch.is_tensor(target["labels"]):
                target["labels"] = target["labels"][valid]
            masks = target.get("masks")
            if torch.is_tensor(masks) and masks.ndim == 3 and len(masks) == len(valid):
                import numpy as np
                rotated = [
                    torch.tensor(
                        np.array(Image.fromarray(m.cpu().numpy().astype("uint8")).rotate(
                            angle, resample=Image.Resampling.NEAREST, expand=False)),
                        dtype=masks.dtype,
                    )
                    for m in masks
                ]
                target["masks"] = torch.stack(rotated)[valid] if rotated else masks[valid]
        masks = target.get("masks")
        if torch.is_tensor(masks) and masks.ndim == 2:
            import numpy as np
            target["masks"] = torch.tensor(
                np.array(Image.fromarray(masks.cpu().numpy().astype("uint8")).rotate(
                    angle, resample=Image.Resampling.NEAREST, expand=False)),
                dtype=masks.dtype,
            )
        return out, target


# ── Registry and builder ────────────────────────────────────────────────


_AUGMENTATION_REGISTRY: dict[str, type] = {
    "horizontal_flip": RandomHorizontalFlip,
    "vertical_flip": RandomVerticalFlip,
    "color_jitter": ColorJitter,
    "random_crop": RandomResizedCrop,
    "gaussian_blur": GaussianBlur,
    "resize": Resize,
    "rotation": RandomRotation,
}


def get_augmentation_preset(name: str, image_size: tuple[int, int] = (640, 640)) -> dict:
    """Return a ``build_augmentation``-ready config dict for a named preset.

    ``nadir_rotation`` mirrors the chestnut-burr small-object policy (training.py
    317-320): free rotation + h/v flips + mild jitter, with mosaic/copy-paste/mixup
    intentionally omitted (they shrink small objects / stitch unnatural composites).
    """
    presets: dict[str, dict] = {
        "nadir_rotation": {
            "rotation": {"degrees": 180, "p": 1.0},
            "horizontal_flip": 0.5,
            "vertical_flip": 0.5,
            "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.0},
            "resize": list(image_size),
        },
        "default": {
            "horizontal_flip": 0.5,
            "color_jitter": {"brightness": 0.2, "contrast": 0.2},
            "resize": list(image_size),
        },
        "none": {"resize": list(image_size)},
    }
    if name not in presets:
        raise ValueError(f"Unknown augmentation preset '{name}'. Available: {sorted(presets)}")
    return presets[name]


def recorded_resize(config: dict | str | None) -> tuple[int, int] | None:
    """The fixed ``(width, height)`` an augmentation config resizes every sample to, or ``None``
    when it pins no size.

    Resolved by building the config's own chain (:func:`build_augmentation`, which resolves a preset
    name string through :func:`get_augmentation_preset` and applies the same float/list/dict/bool
    parameter conventions) and reading the last :class:`Resize` in it, never by re-reading the config
    here: a preset name is not a ``[w, h]`` pair, and a ``resize`` entry can legitimately be a list,
    a kwargs dict, or ``True``. An unbuildable config raises from the builder rather than being
    reported as "no resize", so a caller about to reproduce a training input geometry hears about it.

    Only the deterministic :class:`Resize` counts. :class:`RandomResizedCrop` also fixes the tensor
    size it emits, but its per-sample crop scale is drawn at random, so the input geometry it
    produced is not reproducible outside training and is not reported here.
    """
    if not config:
        return None
    sizes = [t.size for t in build_augmentation(config).transforms if isinstance(t, Resize)]
    if not sizes:
        return None
    width, height = sizes[-1]
    return int(width), int(height)


def build_augmentation(config: dict | str) -> Compose:
    """Build an augmentation pipeline from a config dict or a preset name.

    ``config`` may be a dict (as below) or a preset-name string
    (e.g. ``"nadir_rotation"``) resolved via :func:`get_augmentation_preset`.

    Config format:
        {
            "rotation": {"degrees": 180, "p": 1.0},
            "horizontal_flip": 0.5,       # probability
            "vertical_flip": 0.3,
            "color_jitter": {"brightness": 0.3, "contrast": 0.3},
            "random_crop": {"min_scale": 0.5, "size": [640, 640]},
            "gaussian_blur": 0.1,
            "resize": [640, 640],
        }

    Values can be:
      - float: interpreted as probability (p=value)
      - list/tuple: interpreted as size
      - dict: passed as kwargs to the transform constructor
      - bool: True → use defaults

    Returns a Compose([..., ToTensor()]) pipeline.
    """
    if isinstance(config, str):
        config = get_augmentation_preset(config)

    transforms = []

    for name, params in config.items():
        cls = _AUGMENTATION_REGISTRY.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown augmentation '{name}'. Available: {sorted(_AUGMENTATION_REGISTRY)}"
            )

        if isinstance(params, bool) and params:
            transforms.append(cls())
        elif isinstance(params, (int, float)):
            transforms.append(cls(p=params))
        elif isinstance(params, (list, tuple)):
            transforms.append(cls(size=tuple(params)))
        elif isinstance(params, dict):
            transforms.append(cls(**params))

    # Always end with ToTensor
    transforms.append(ToTensor())
    return Compose(transforms)
