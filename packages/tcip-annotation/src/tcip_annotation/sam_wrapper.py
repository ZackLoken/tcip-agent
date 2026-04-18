"""SAM (Segment Anything Model) wrapper for interactive segmentation.

Supports point prompts and box prompts. Caches image embeddings for fast
repeated predictions on the same image. Uses MobileSAM by default (small
and fast); supports SAM-ViT-B/H via model_type parameter.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded model cache
_predictor: object | None = None
_current_model_type: str | None = None
_current_image_path: str | None = None


def _get_predictor(model_type: str = "vit_b") -> object:
    """Load or reuse the SAM predictor singleton."""
    global _predictor, _current_model_type

    if _predictor is not None and _current_model_type == model_type:
        return _predictor

    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        raise ImportError(
            "segment-anything is not installed. "
            "Install with: pip install segment-anything"
        )

    # Resolve checkpoint path
    checkpoint_dir = Path.home() / ".cache" / "tcip" / "sam"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_map = {
        "vit_b": "sam_vit_b_01ec64.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_h": "sam_vit_h_4b8939.pth",
    }

    ckpt_name = checkpoint_map.get(model_type)
    if ckpt_name is None:
        raise ValueError(f"Unknown model_type: {model_type}. Use vit_b, vit_l, or vit_h.")

    ckpt_path = checkpoint_dir / ckpt_name
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"SAM checkpoint not found at {ckpt_path}. "
            f"Download from https://github.com/facebookresearch/segment-anything#model-checkpoints"
        )

    logger.info("Loading SAM model %s from %s", model_type, ckpt_path)
    sam = sam_model_registry[model_type](checkpoint=str(ckpt_path))

    # Use CUDA if available
    import torch
    if torch.cuda.is_available():
        sam = sam.to("cuda")
        logger.info("SAM model loaded on CUDA")
    else:
        logger.info("SAM model loaded on CPU")

    _predictor = SamPredictor(sam)
    _current_model_type = model_type
    return _predictor


def _set_image(predictor: object, image_path: str) -> None:
    """Set the image on the predictor, using cache when possible."""
    global _current_image_path

    if _current_image_path == image_path:
        return  # embedding already cached

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)  # type: ignore[union-attr]
    _current_image_path = image_path
    logger.info("SAM image embedding computed for %s", image_path)


def mask_to_polygon(mask: np.ndarray) -> list[tuple[float, float]]:
    """Convert a binary mask to a single polygon (largest contour).

    Args:
        mask: 2D boolean/uint8 array.

    Returns:
        List of (x, y) tuples in pixel coordinates.
    """
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    if not contours:
        return []

    # Take the largest contour by area
    largest = max(contours, key=cv2.contourArea)

    # Simplify to reduce vertex count
    epsilon = 0.5 * cv2.arcLength(largest, True) / max(len(largest), 1)
    epsilon = max(epsilon, 1.0)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    points: list[tuple[float, float]] = []
    for pt in approx:
        points.append((float(pt[0][0]), float(pt[0][1])))

    return points


def predict_from_point(
    image_path: str,
    x: float,
    y: float,
    label: int = 1,
    model_type: str = "vit_b",
) -> list[tuple[float, float]]:
    """Run SAM prediction from a single point prompt.

    Args:
        image_path: Absolute path to the image.
        x: X coordinate in pixel space.
        y: Y coordinate in pixel space.
        label: 1 for foreground, 0 for background.
        model_type: SAM model variant (vit_b, vit_l, vit_h).

    Returns:
        List of (x, y) polygon vertices in pixel coordinates.
    """
    predictor = _get_predictor(model_type)
    _set_image(predictor, image_path)

    points = np.array([[x, y]])
    labels = np.array([label])

    masks, scores, _ = predictor.predict(  # type: ignore[union-attr]
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )

    # Select the mask with highest score
    best_idx = int(np.argmax(scores))
    return mask_to_polygon(masks[best_idx])


def predict_from_box(
    image_path: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    model_type: str = "vit_b",
) -> list[tuple[float, float]]:
    """Run SAM prediction from a box prompt.

    Args:
        image_path: Absolute path to the image.
        x1, y1, x2, y2: Box corners in pixel coordinates.
        model_type: SAM model variant.

    Returns:
        List of (x, y) polygon vertices in pixel coordinates.
    """
    predictor = _get_predictor(model_type)
    _set_image(predictor, image_path)

    box = np.array([x1, y1, x2, y2])

    masks, scores, _ = predictor.predict(  # type: ignore[union-attr]
        box=box,
        multimask_output=True,
    )

    best_idx = int(np.argmax(scores))
    return mask_to_polygon(masks[best_idx])


def predict_from_points(
    image_path: str,
    points: list[tuple[float, float]],
    labels: list[int],
    model_type: str = "vit_b",
) -> list[tuple[float, float]]:
    """Run SAM prediction from multiple point prompts.

    Args:
        image_path: Absolute path to the image.
        points: List of (x, y) coordinates.
        labels: List of labels (1=foreground, 0=background) per point.
        model_type: SAM model variant.

    Returns:
        List of (x, y) polygon vertices in pixel coordinates.
    """
    predictor = _get_predictor(model_type)
    _set_image(predictor, image_path)

    pts = np.array(points)
    lbls = np.array(labels)

    masks, scores, _ = predictor.predict(  # type: ignore[union-attr]
        point_coords=pts,
        point_labels=lbls,
        multimask_output=True,
    )

    best_idx = int(np.argmax(scores))
    return mask_to_polygon(masks[best_idx])
