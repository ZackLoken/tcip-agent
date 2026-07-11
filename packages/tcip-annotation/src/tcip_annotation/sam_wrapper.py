"""SAM2 wrapper for interactive segmentation.

Uses SAM2 (Hiera backbones). Supports point prompts, box prompts, and automatic
mask generation. Caches image embeddings for fast repeated predictions on the
same image.

Install:
    pip install "sam-2 @ git+https://github.com/facebookresearch/sam2.git"
    # (the distribution is named `sam-2`; the import path is still `sam2`)
    # checkpoints download into ~/.cache/tcip/sam2/
    # see https://github.com/facebookresearch/sam2#model-description

Public API:
    predict_from_point(image_path, x, y, label, model_type)
    predict_from_points(image_path, points, labels, model_type)
    predict_from_box(image_path, x1, y1, x2, y2, model_type)
    auto_mask(image_path, model_type, ...)
    grid_to_pixel(cell, img_w, img_h, cols, rows)

`model_type` values: "hiera_t", "hiera_s", "hiera_b+", "hiera_l".
"""

from __future__ import annotations

import functools
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_predictor: Any = None
_current_model_type: str | None = None
_current_image_path: str | None = None

# Serializes the shared predictor + current-image globals. Concurrent callers (e.g. the
# web review engine handling multiple requests on different threads) must not interleave
# set-image / predict, which would return masks computed against the wrong image.
_SAM_LOCK = threading.RLock()


def _serialized(fn):
    """Run ``fn`` while holding the SAM predictor lock."""
    @functools.wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        with _SAM_LOCK:
            return fn(*args, **kwargs)

    return _wrapped

# SAM 2.1 model variants: (hydra config, checkpoint filename).
_MODEL_MAP = {
    "hiera_t":  ("configs/sam2.1/sam2.1_hiera_t.yaml",  "sam2.1_hiera_tiny.pt"),
    "hiera_s":  ("configs/sam2.1/sam2.1_hiera_s.yaml",  "sam2.1_hiera_small.pt"),
    "hiera_b+": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "hiera_l":  ("configs/sam2.1/sam2.1_hiera_l.yaml",  "sam2.1_hiera_large.pt"),
}


def _get_predictor(model_type: str = "hiera_b+") -> Any:
    """Load or reuse the SAM2 predictor singleton."""
    global _predictor, _current_model_type, _current_image_path

    if _predictor is not None and _current_model_type == model_type:
        return _predictor

    if model_type not in _MODEL_MAP:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Valid: {sorted(_MODEL_MAP.keys())}"
        )

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import torch

    config_file, ckpt_filename = _MODEL_MAP[model_type]

    checkpoint_dir = Path.home() / ".cache" / "tcip" / "sam2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / ckpt_filename
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found at {ckpt_path}. "
            f"Download from https://github.com/facebookresearch/sam2#model-description "
            f"(looking for {ckpt_filename})."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading SAM2 %s on %s", model_type, device)
    sam2_model = build_sam2(config_file, str(ckpt_path), device=device)
    _predictor = SAM2ImagePredictor(sam2_model)
    _current_model_type = model_type
    # A fresh predictor has no image embedding: invalidate the image cache so
    # _set_image recomputes it even when the same image path is requested again.
    _current_image_path = None
    return _predictor


def _set_image(predictor: Any, image_path: str) -> None:
    """Set the image on the predictor, using cache when possible."""
    global _current_image_path
    if _current_image_path == image_path:
        return

    import numpy as np
    from PIL import Image

    from tcip_annotation.utils import auto_orient_image

    # EXIF-orient so SAM's embedding, box prompts, and returned polygons all live in the
    # same upright frame the annotations are authored in (cv2.imread ignores EXIF).
    with Image.open(image_path) as im:
        img_rgb = np.asarray(auto_orient_image(im).convert("RGB"))

    predictor.set_image(img_rgb)
    _current_image_path = image_path
    logger.info("SAM2 image embedding computed for %s", image_path)


def mask_to_polygon(mask: Any) -> list[tuple[float, float]]:
    """Convert a binary mask to a single polygon (largest contour).

    Args:
        mask: 2D boolean or uint8 array.

    Returns:
        List of (x, y) tuples in pixel coordinates.
    """
    import cv2
    import numpy as np
    mask_uint8 = (np.asarray(mask).astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    if not contours:
        return []

    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.5 * cv2.arcLength(largest, True) / max(len(largest), 1)
    epsilon = max(epsilon, 1.0)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    return [(float(pt[0][0]), float(pt[0][1])) for pt in approx]


@_serialized
def predict_from_point(
    image_path: str,
    x: float,
    y: float,
    label: int = 1,
    model_type: str = "hiera_b+",
) -> list[tuple[float, float]]:
    """Run SAM2 prediction from a single point prompt.

    Args:
        image_path: Absolute path to the image.
        x: X coordinate in pixel space.
        y: Y coordinate in pixel space.
        label: 1 for foreground, 0 for background.
        model_type: hiera_t, hiera_s, hiera_b+, or hiera_l.

    Returns:
        List of (x, y) polygon vertices in pixel coordinates.
    """
    import numpy as np
    predictor = _get_predictor(model_type)
    _set_image(predictor, image_path)

    masks, scores, _ = predictor.predict(
        point_coords=np.array([[x, y]]),
        point_labels=np.array([label]),
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    return mask_to_polygon(masks[best_idx])


@_serialized
def predict_from_points(
    image_path: str,
    points: list[tuple[float, float]],
    labels: list[int],
    model_type: str = "hiera_b+",
) -> list[tuple[float, float]]:
    """Run SAM2 prediction from multiple point prompts."""
    import numpy as np
    predictor = _get_predictor(model_type)
    _set_image(predictor, image_path)

    masks, scores, _ = predictor.predict(
        point_coords=np.array(points),
        point_labels=np.array(labels),
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    return mask_to_polygon(masks[best_idx])


@_serialized
def predict_from_box(
    image_path: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    model_type: str = "hiera_b+",
) -> list[tuple[float, float]]:
    """Run SAM2 prediction from a box prompt."""
    import numpy as np
    predictor = _get_predictor(model_type)
    _set_image(predictor, image_path)

    masks, scores, _ = predictor.predict(
        box=np.array([x1, y1, x2, y2]),
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    return mask_to_polygon(masks[best_idx])


@_serialized
def auto_mask(
    image_path: str,
    model_type: str = "hiera_b+",
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.86,
    stability_score_thresh: float = 0.92,
    min_mask_region_area: int = 100,
) -> list[dict]:
    """Generate all candidate masks in an image using SAM2 auto-mask generator.

    Returns a list of candidate dicts sorted by area (largest first), each with:
      - candidate_id: int (0-based index)
      - bbox: [x1, y1, x2, y2] in pixel coordinates
      - area: int (pixel count)
      - stability_score: float
      - predicted_iou: float
      - polygon: list of (x, y) tuples in pixel coordinates

    Args:
        image_path: Absolute path to the image.
        model_type: hiera_t, hiera_s, hiera_b+, or hiera_l.
        points_per_side: Grid density for auto-mask generation.
        pred_iou_thresh: Minimum predicted IoU to keep a mask.
        stability_score_thresh: Minimum stability score to keep a mask.
        min_mask_region_area: Minimum mask area in pixels.
    """
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    predictor = _get_predictor(model_type)
    sam2_model = predictor.model

    generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )

    import numpy as np
    from PIL import Image

    from tcip_annotation.utils import auto_orient_image

    # EXIF-orient so returned mask bboxes/polygons land in the upright annotation frame.
    with Image.open(image_path) as im:
        img_rgb = np.asarray(auto_orient_image(im).convert("RGB"))

    logger.info("Running SAM2 auto-mask generation on %s", image_path)
    masks = generator.generate(img_rgb)
    masks.sort(key=lambda m: m["area"], reverse=True)

    candidates: list[dict] = []
    for i, m in enumerate(masks):
        polygon = mask_to_polygon(m["segmentation"])
        if len(polygon) < 3:
            continue
        x, y, w, h = m["bbox"]
        candidates.append({
            "candidate_id": i,
            "bbox": [float(x), float(y), float(x + w), float(y + h)],
            "area": int(m["area"]),
            "stability_score": float(m["stability_score"]),
            "predicted_iou": float(m["predicted_iou"]),
            "polygon": polygon,
        })

    logger.info("Auto-mask generated %d candidates for %s", len(candidates), image_path)
    return candidates


def grid_to_pixel(
    cell: str,
    img_w: int,
    img_h: int,
    cols: int = 8,
    rows: int = 6,
) -> tuple[float, float]:
    """Convert a grid cell reference (e.g. 'B3') to pixel coordinates (cell center).

    Grid uses letter columns (A-H) and number rows (1-6) by default.

    Args:
        cell: Grid reference like 'B3', 'D5'.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        cols: Number of grid columns.
        rows: Number of grid rows.

    Returns:
        (x, y) center of the referenced cell in pixel coordinates.
    """
    cell = cell.strip().upper()
    if len(cell) < 2:
        raise ValueError(f"Invalid cell reference: {cell!r}. Expected format like 'B3'.")

    col_letter = cell[0]
    row_str = cell[1:]

    col_idx = ord(col_letter) - ord("A")
    if col_idx < 0 or col_idx >= cols:
        raise ValueError(
            f"Column '{col_letter}' out of range. Use A-{chr(ord('A') + cols - 1)}."
        )

    try:
        row_idx = int(row_str) - 1
    except ValueError:
        raise ValueError(f"Invalid row number: {row_str!r}. Expected integer.")
    if row_idx < 0 or row_idx >= rows:
        raise ValueError(f"Row {row_str} out of range. Use 1-{rows}.")

    cell_w = img_w / cols
    cell_h = img_h / rows
    return (col_idx + 0.5) * cell_w, (row_idx + 0.5) * cell_h
