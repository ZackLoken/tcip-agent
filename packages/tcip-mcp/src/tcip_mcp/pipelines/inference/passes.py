"""What a checkpoint's pass returns for one image: a detector's detections or a head's outputs."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, JsonValue


class Detection(BaseModel):
    """One detection in full-frame pixels.

    It holds its xyxy ``box``, ``score``, 1-indexed ``label``, ``attributes`` (one value id per
    attribute the checkpoint's recorded dimensions declare, none where they declare none) and
    ``mask``, its flat ``[x0, y0, x1, y1, ...]`` polygons: ``None`` for a detection carrying no
    mask, empty for a mask that binarized to nothing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    box: tuple[float, float, float, float]
    score: float
    label: int
    attributes: list[int]
    mask: list[list[float]] | None


class DetectionPass(BaseModel):
    """A detector's pass over one ``width`` x ``height`` frame, the source ``image``.

    ``detections`` are the frame's, kept to its ``cap``; each carries a mask where the pass's
    execution cuts masks. ``soft_masks`` is each detection's uncut mask as the model kept it
    (the part of the frame its nonzero pixels span and that part's ``(x, y)`` origin), for an
    :attr:`~tcip_mcp.pipelines.execution.Execution.uncut` pass alone.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, arbitrary_types_allowed=True
    )

    image: str
    width: int
    height: int
    cap: int
    detections: list[Detection]
    soft_masks: list[tuple[np.ndarray, tuple[int, int]]] | None


class HeadPass(BaseModel):
    """A classification, ordinal, regression or semantic segmentation pass over one image.

    ``outputs`` are the model's own outputs by name, each tensor as nested lists.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    image: str
    width: int
    height: int
    outputs: dict[str, JsonValue]
