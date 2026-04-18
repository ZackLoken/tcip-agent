"""SAM (Segment Anything Model) wrapper for interactive segmentation.

Supports point prompts and box prompts. Caches image embeddings for fast
repeated predictions on the same image. Uses MobileSAM by default (small
and fast); supports SAM-ViT-B/H via model_type parameter.
"""

from __future__ import annotations

import logging
from pathlib import Path

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

    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # noqa: F811
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
    import cv2
    import numpy as np
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
    import numpy as np
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
    import numpy as np
    predictor = _get_predictor(model_type)
    _set_image(predictor, image_path)

    box = np.array([x1, y1, x2, y2])

    masks, scores, _ = predictor.predict(  # type: ignore[union-attr]
        box=box,
        multimask_output=True,
    )

    best_idx = int(np.argmax(scores))
    return mask_to_polygon(masks[best_idx])


def auto_mask(
    image_path: str,
    model_type: str = "vit_b",
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.86,
    stability_score_thresh: float = 0.92,
    min_mask_region_area: int = 100,
) -> list[dict]:
    """Generate all candidate masks in an image using SAM auto-mask generator.

    Returns a list of candidate dicts sorted by area (largest first), each with:
      - candidate_id: int (0-based index)
      - bbox: [x1, y1, x2, y2] in pixel coordinates
      - area: int (pixel count)
      - stability_score: float
      - predicted_iou: float
      - polygon: list of (x, y) tuples in pixel coordinates

    Args:
        image_path: Absolute path to the image.
        model_type: SAM model variant (vit_b, vit_l, vit_h).
        points_per_side: Grid density for auto-mask generation.
        pred_iou_thresh: Minimum predicted IoU to keep a mask.
        stability_score_thresh: Minimum stability score to keep a mask.
        min_mask_region_area: Minimum mask area in pixels.
    """
    try:
        from segment_anything import SamAutomaticMaskGenerator
    except ImportError:
        raise ImportError(
            "segment-anything is not installed. "
            "Install with: pip install segment-anything"
        )

    # Get the underlying SAM model from the predictor
    predictor = _get_predictor(model_type)
    sam_model = predictor.model  # type: ignore[union-attr]

    generator = SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )

    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    logger.info("Running SAM auto-mask generation on %s", image_path)
    masks = generator.generate(img_rgb)

    # Sort by area descending
    masks.sort(key=lambda m: m["area"], reverse=True)

    candidates = []
    for i, m in enumerate(masks):
        polygon = mask_to_polygon(m["segmentation"])
        if len(polygon) < 3:
            continue
        x, y, w, h = m["bbox"]  # XYWH format from SAM
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
    """Convert a grid cell reference (e.g. 'B3') to pixel coordinates (center of cell).

    Grid uses letter columns (A-H) and number rows (1-6) by default.

    Args:
        cell: Grid reference like 'B3', 'D5', etc.
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
        row_idx = int(row_str) - 1  # 1-based to 0-based
    except ValueError:
        raise ValueError(f"Invalid row number: {row_str!r}. Expected integer.")
    if row_idx < 0 or row_idx >= rows:
        raise ValueError(f"Row {row_str} out of range. Use 1-{rows}.")

    cell_w = img_w / cols
    cell_h = img_h / rows
    cx = (col_idx + 0.5) * cell_w
    cy = (row_idx + 0.5) * cell_h
    return cx, cy


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
    import numpy as np
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
