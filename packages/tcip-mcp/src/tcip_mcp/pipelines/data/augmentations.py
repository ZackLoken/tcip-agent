"""Data augmentation transforms for all task types.

Provides composable augmentation transforms that work with both detection
(image, target dict with 'boxes') and classification (image, target dict)
pipelines, implemented on PIL and torch.

Usage:
    transforms = build_augmentation(config)
    img_tensor, target = transforms(pil_image, target_dict)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated

import torch
from PIL import Image, ImageEnhance, ImageFilter
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveInt,
    ValidationError,
    model_validator,
)

from tcip_mcp.pipelines.image_utils import pil_to_tensor
from tcip_mcp.project_record import CaptureInvariances, RotationDegrees

if TYPE_CHECKING:
    from tcip_mcp.pipelines.schemas import AugmentationSpec

Probability = Annotated[float, Field(ge=0, le=1)]
CropFraction = Annotated[float, Field(gt=0, le=1)]
Size = tuple[PositiveInt, PositiveInt]
Refusals = dict[str, bool | float]
"""Capture invariances (fields of ``CaptureInvariances``), each mapped to the value a record
states for it."""


class _Transform(BaseModel, ABC):
    """A transform built from exactly the values an augmentation entry states.

    Each value is in its type's range and finite.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @abstractmethod
    def refused_under(self, invariances: CaptureInvariances) -> Refusals:
        """Return each capture invariance the measurement does not hold under for this transform.

        ``invariances`` says so for the transform as stated; each maps to the value
        ``invariances`` states for it.
        """


class Compose:
    """Chain multiple (image, target) transforms."""

    def __init__(self, transforms: list) -> None:
        """Hold ``transforms``, applied in their order."""
        self.transforms = transforms

    def __call__(
        self, img: Image.Image, target: dict
    ) -> tuple[torch.Tensor | Image.Image, dict]:
        """Apply each transform to ``(img, target)`` in order and return the resulting pair."""
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

    def __repr__(self) -> str:
        """Return the chain as ``Compose([...])``, one transform per line."""
        lines = [f"  {t!r}" for t in self.transforms]
        return "Compose([\n" + "\n".join(lines) + "\n])"


class ToTensor:
    """Convert a PIL image to a tensor, ``target`` unchanged."""

    def __call__(
        self, img: Image.Image, target: dict
    ) -> tuple[torch.Tensor, dict]:
        """Return ``img`` as a tensor (:func:`pil_to_tensor`) beside ``target`` unchanged."""
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


def _keep_rows(target: dict, valid: torch.Tensor) -> None:
    """Keep the rows ``valid`` marks in every per-box tensor of ``target``.

    The per-box keys are :data:`~tcip_mcp.pipelines.data.datasets.PER_BOX_KEYS`.
    """
    from tcip_mcp.pipelines.data.datasets import PER_BOX_KEYS

    for key in PER_BOX_KEYS:
        value = target.get(key)
        if torch.is_tensor(value) and len(value) == len(valid):
            target[key] = value[valid]


class RandomHorizontalFlip(_Transform):
    """Flip image, boxes, and masks horizontally with probability p."""

    p: Probability

    def refused_under(self, invariances: CaptureInvariances) -> Refusals:
        """Return the horizontal flip's refusal unless ``p`` is 0 or ``invariances`` holds it."""
        held = invariances.horizontal_flip
        return {} if held or not self.p else {"horizontal_flip": held}

    def __call__(
        self, img: Image.Image, target: dict
    ) -> tuple[Image.Image, dict]:
        """Flip ``img``, its boxes and its masks horizontally with probability ``p``."""
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


class RandomVerticalFlip(_Transform):
    """Flip image, boxes, and masks vertically with probability p."""

    p: Probability

    def refused_under(self, invariances: CaptureInvariances) -> Refusals:
        """Return the vertical flip's refusal unless ``p`` is 0 or ``invariances`` holds it."""
        held = invariances.vertical_flip
        return {} if held or not self.p else {"vertical_flip": held}

    def __call__(
        self, img: Image.Image, target: dict
    ) -> tuple[Image.Image, dict]:
        """Flip ``img``, its boxes and its masks vertically with probability ``p``."""
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


class ColorJitter(_Transform):
    """Random brightness, contrast and saturation perturbations.

    Each is a factor drawn uniformly from ``[1 - value, 1 + value]``; a value of 0 leaves that
    property unchanged, and each value is at most 1, where the factor's lower end reaches 0 and
    stays non-negative.
    """

    brightness: Annotated[float, Field(ge=0, le=1)]
    contrast: Annotated[float, Field(ge=0, le=1)]
    saturation: Annotated[float, Field(ge=0, le=1)]

    def refused_under(self, invariances: CaptureInvariances) -> Refusals:
        """Return the refusals of color and intensity, each unless ``invariances`` holds it.

        Color is refused when the jitter shifts saturation, intensity when it shifts brightness
        or contrast.
        """
        refused: Refusals = {}
        if self.saturation and not invariances.color:
            refused["color"] = invariances.color
        if (self.brightness or self.contrast) and not invariances.intensity:
            refused["intensity"] = invariances.intensity
        return refused

    def __call__(
        self, img: Image.Image, target: dict
    ) -> tuple[Image.Image, dict]:
        """Perturb ``img``'s brightness, contrast and saturation by factors drawn per call."""
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)
        if self.saturation > 0:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            img = ImageEnhance.Color(img).enhance(factor)
        return img, target


class RandomResizedCrop(_Transform):
    """Crop a random region and resize to target size, adjusting boxes and masks alike.

    ``min_scale`` is at most ``max_scale``.
    """

    size: Size
    min_scale: CropFraction
    max_scale: CropFraction

    @model_validator(mode="after")
    def _ordered(self) -> RandomResizedCrop:
        if self.min_scale > self.max_scale:
            raise ValueError(
                f"min_scale {self.min_scale} exceeds max_scale {self.max_scale}"
            )
        return self

    def refused_under(self, invariances: CaptureInvariances) -> Refusals:
        """Return the scale refusal unless ``invariances`` holds under scale.

        Every crop the transform states resizes a region to ``size``.
        """
        return {} if invariances.scale else {"scale": invariances.scale}

    def __call__(
        self, img: Image.Image, target: dict
    ) -> tuple[Image.Image, dict]:
        """Crop a random region of ``img`` and resize it to ``size``, with its boxes and masks.

        A box left one pixel wide or tall or less is dropped, its row in every per-box tensor
        with it.
        """
        w, h = img.size
        scale = random.uniform(self.min_scale, self.max_scale)
        crop_w = int(w * scale)
        crop_h = int(h * scale)
        x1 = random.randint(0, max(0, w - crop_w))
        y1 = random.randint(0, max(0, h - crop_h))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        img = img.crop((x1, y1, x2, y2)).resize(
            self.size, Image.Resampling.BILINEAR
        )

        # Crop + nearest-resize masks in lockstep with the image
        if torch.is_tensor(target.get("masks")):
            target["masks"] = _resize_masks(
                target["masks"][..., y1:y2, x1:x2], self.size
            )

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
                valid = (boxes[:, 2] - boxes[:, 0] > 1) & (
                    boxes[:, 3] - boxes[:, 1] > 1
                )
                target["boxes"] = boxes
                _keep_rows(target, valid)

        return img, target


class GaussianBlur(_Transform):
    """Apply Gaussian blur with probability p."""

    p: Probability
    radius: NonNegativeFloat
    """The Gaussian's radius in pixels."""

    def refused_under(self, invariances: CaptureInvariances) -> Refusals:
        """Return no capture-invariance refusals."""
        return {}

    def __call__(
        self, img: Image.Image, target: dict
    ) -> tuple[Image.Image, dict]:
        """Blur ``img`` at ``radius`` with probability ``p``, ``target`` unchanged."""
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.radius))
        return img, target


class Resize(_Transform):
    """Resize image to fixed size. Adjusts boxes and masks accordingly."""

    size: Size

    def refused_under(self, invariances: CaptureInvariances) -> Refusals:
        """Return no capture-invariance refusals."""
        return {}

    def __call__(
        self, img: Image.Image, target: dict
    ) -> tuple[Image.Image, dict]:
        """Resize ``img`` to ``size``, scaling its boxes and resizing its masks with it."""
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


class RandomRotation(_Transform):
    """Rotate image (and detection boxes/masks) by a uniform angle with probability ``p``.

    The angle is drawn from ``[-degrees, degrees]``. Boxes are rotated as 4 corners about the
    image center and taken as their axis-aligned envelope, then clamped to the frame; a box one
    pixel wide or tall or less is dropped, its row in every per-box tensor with it. Semantic
    ``[H, W]`` masks are rotated with nearest resampling. Non-detection targets
    (classification/ordinal/regression) pass through untouched.
    """

    degrees: RotationDegrees
    p: Probability

    def refused_under(self, invariances: CaptureInvariances) -> Refusals:
        """Return the rotation refusal when ``p`` is not 0 and ``degrees`` passes what is held.

        What is held is the largest rotation ``invariances`` holds under.
        """
        held = invariances.rotation
        return {"rotation": held} if self.p and self.degrees > held else {}

    def __call__(
        self, img: Image.Image, target: dict
    ) -> tuple[Image.Image, dict]:
        """Rotate ``img`` and its target by one drawn angle with probability ``p``."""
        if random.random() >= self.p:
            return img, target
        import math

        angle = random.uniform(-self.degrees, self.degrees)
        w, h = img.size
        out = img.rotate(
            angle, resample=Image.Resampling.BILINEAR, expand=False
        )

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
            target["boxes"] = nb
            masks = target.get("masks")
            if torch.is_tensor(masks) and masks.ndim == 3 and len(masks):
                import numpy as np

                target["masks"] = torch.stack(
                    [
                        torch.tensor(
                            np.array(
                                Image.fromarray(
                                    m.cpu().numpy().astype("uint8")
                                ).rotate(
                                    angle,
                                    resample=Image.Resampling.NEAREST,
                                    expand=False,
                                )
                            ),
                            dtype=masks.dtype,
                        )
                        for m in masks
                    ]
                )
            _keep_rows(target, valid)
        masks = target.get("masks")
        if torch.is_tensor(masks) and masks.ndim == 2:
            import numpy as np

            target["masks"] = torch.tensor(
                np.array(
                    Image.fromarray(
                        masks.cpu().numpy().astype("uint8")
                    ).rotate(
                        angle, resample=Image.Resampling.NEAREST, expand=False
                    )
                ),
                dtype=masks.dtype,
            )
        return out, target


# ── Registry and builder ────────────────────────────────────────────────


_AUGMENTATION_REGISTRY: dict[str, type[_Transform]] = {
    "horizontal_flip": RandomHorizontalFlip,
    "vertical_flip": RandomVerticalFlip,
    "color_jitter": ColorJitter,
    "random_crop": RandomResizedCrop,
    "gaussian_blur": GaussianBlur,
    "resize": Resize,
    "rotation": RandomRotation,
}
"""Every transform an augmentation block names."""


def _stated_transforms(
    config: AugmentationSpec,
) -> list[tuple[str, _Transform]]:
    """Return each transform ``config`` states, in its order, beside its name.

    Each is the transform its stated values validate to. Refuses (``ValueError``) an unknown
    transform, and, naming the entry, one whose values its transform's type refuses.
    """
    from tcip_mcp.pipelines.model_build import resolve_named

    stated = []
    for name, params in (config.model_extra or {}).items():
        cls = resolve_named(name, _AUGMENTATION_REGISTRY, kind="augmentation")
        try:
            stated.append((name, cls.model_validate(params)))
        except ValidationError as exc:
            raise ValueError(f"augmentation {name!r}: {exc}") from exc
    return stated


def forbidden_transforms(
    config: AugmentationSpec, invariances: CaptureInvariances
) -> list[str]:
    """Return one issue per invariance under which a transform ``config`` states is refused.

    ``invariances`` is a project record's, and the refusals are each transform's
    ``refused_under``. Each issue names the transform, the invariance, the value the record
    states for it and ``initialize_project``, where it is stated. :func:`_stated_transforms`'
    refusals propagate.
    """
    return [
        f"augmentation {name!r} relies on the capture invariance {fact!r} beyond what this "
        f"project's record states for it ({stated!r}): the trait's measurement does not hold "
        "under the transform as stated (the invariances are stated through initialize_project)"
        for name, transform in _stated_transforms(config)
        for fact, stated in transform.refused_under(invariances).items()
    ]


def recorded_resize(config: AugmentationSpec | None) -> tuple[int, int] | None:
    """Return the fixed ``(width, height)`` an augmentation config resizes every sample to.

    The answer is ``None`` when it pins no size. It is resolved by building the config's own
    chain (:func:`build_augmentation`) and reading the last :class:`Resize` in it. An
    unbuildable config raises. :class:`RandomResizedCrop` is not reported.
    """
    if config is None:
        return None
    sizes = [
        t.size
        for t in build_augmentation(config).transforms
        if isinstance(t, Resize)
    ]
    if not sizes:
        return None
    width, height = sizes[-1]
    return int(width), int(height)


def build_augmentation(config: AugmentationSpec) -> Compose:
    """Build the augmentation chain ``config`` states, in its order, ending in :class:`ToTensor`.

    ``config`` maps each transform's name (:data:`_AUGMENTATION_REGISTRY`) to a dict stating
    every value that transform's constructor takes, e.g. ``{"rationale": "...",
    "horizontal_flip": {"p": 0.5}, "resize": {"size": [512, 512]}}``. Refuses what
    :func:`_stated_transforms` refuses.
    """
    return Compose(
        [transform for _name, transform in _stated_transforms(config)]
        + [ToTensor()]
    )
