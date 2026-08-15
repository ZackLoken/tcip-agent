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
    grid_to_rect(cell, cells) / grid_to_pixel(cell, cells) / cell_fields(cell)
    column_label(index) / column_index(label)

The prompted predictors return polygon rings (a mask with two disjoint regions is two rings),
via the shared :func:`tcip_annotation.mask_contours.mask_to_polygon_rings`, the same extractor the
model-prediction export path uses, so SAM-assisted GT and a model's prediction describe an
occlusion-split object identically.

`model_type` values: "hiera_t", "hiera_s", "hiera_b+", "hiera_l".
"""

from __future__ import annotations

import functools
import logging
import re
import threading
from pathlib import Path
from typing import Any

from tcip_annotation.mask_contours import mask_to_polygon_rings

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

def checkpoint_path(model_type: str = "hiera_b+") -> Path:
    """The local checkpoint path ``_get_predictor`` loads for ``model_type``: the single source of
    truth for the filename/location, so callers (and tests that gate on availability) never drift.
    ``Path.home()`` is read at call time (not cached at import) so a test that redirects it works."""
    if model_type not in _MODEL_MAP:
        raise ValueError(f"Unknown model_type '{model_type}'. Valid: {sorted(_MODEL_MAP)}")
    return Path.home() / ".cache" / "tcip" / "sam2" / _MODEL_MAP[model_type][1]


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

    config_file, _ = _MODEL_MAP[model_type]

    ckpt_path = checkpoint_path(model_type)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found at {ckpt_path}. "
            f"Download from https://github.com/facebookresearch/sam2#model-description "
            f"(looking for {ckpt_path.name})."
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


@_serialized
def predict_from_point(
    image_path: str,
    x: float,
    y: float,
    label: int = 1,
    model_type: str = "hiera_b+",
) -> list[list[tuple[float, float]]]:
    """Run SAM2 prediction from a single point prompt.

    Args:
        image_path: Absolute path to the image.
        x: X coordinate in pixel space.
        y: Y coordinate in pixel space.
        label: 1 for foreground, 0 for background.
        model_type: hiera_t, hiera_s, hiera_b+, or hiera_l.

    Returns:
        Polygon rings: one list of (x, y) pixel vertices per connected region of the mask.
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
    return mask_to_polygon_rings(masks[best_idx])


@_serialized
def predict_from_points(
    image_path: str,
    points: list[tuple[float, float]],
    labels: list[int],
    model_type: str = "hiera_b+",
) -> list[list[tuple[float, float]]]:
    """Run SAM2 prediction from multiple point prompts (returns polygon rings)."""
    import numpy as np
    predictor = _get_predictor(model_type)
    _set_image(predictor, image_path)

    masks, scores, _ = predictor.predict(
        point_coords=np.array(points),
        point_labels=np.array(labels),
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    return mask_to_polygon_rings(masks[best_idx])


@_serialized
def predict_from_box(
    image_path: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    model_type: str = "hiera_b+",
) -> list[list[tuple[float, float]]]:
    """Run SAM2 prediction from a box prompt (returns polygon rings)."""
    import numpy as np
    predictor = _get_predictor(model_type)
    _set_image(predictor, image_path)

    masks, scores, _ = predictor.predict(
        box=np.array([x1, y1, x2, y2]),
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    return mask_to_polygon_rings(masks[best_idx])


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
      - rings: polygon rings, one list of (x, y) pixel tuples per connected region of the mask
        (an occlusion-split object carries more than one; none is dropped)

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
        rings = mask_to_polygon_rings(m["segmentation"])
        if not rings:
            continue
        x, y, w, h = m["bbox"]
        candidates.append({
            "candidate_id": i,
            "bbox": [float(x), float(y), float(x + w), float(y + h)],
            "area": int(m["area"]),
            "stability_score": float(m["stability_score"]),
            "predicted_iou": float(m["predicted_iou"]),
            "rings": rings,
        })

    logger.info("Auto-mask generated %d candidates for %s", len(candidates), image_path)
    return candidates


def column_label(index: int) -> str:
    """Spreadsheet-style label for 0-based column ``index``: A-Z, then AA, AB, ...

    Bijective base-26, the scheme the grid overlay renders with; :func:`column_index` is the
    inverse. A single ``chr`` past 'Z' walks into punctuation and then lowercase letters, and
    case-insensitive parsing folds lowercase back onto columns 0-25, so labels beyond 26
    columns must widen instead.
    """
    if index < 0:
        raise ValueError(f"Column index must be non-negative, got {index}")
    label = ""
    n = index + 1
    while n:
        n, rem = divmod(n - 1, 26)
        label = chr(ord("A") + rem) + label
    return label


def column_index(label: str) -> int:
    """0-based column index for a spreadsheet-style ``label``; inverse of :func:`column_label`."""
    if not label or not all("A" <= ch <= "Z" for ch in label):
        raise ValueError(f"Invalid column label: {label!r}. Expected letters A-Z.")
    n = 0
    for ch in label:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def cell_fields(cell: Any) -> tuple[str, float, float, float, float]:
    """``(name, x0, y0, x1, y1)`` from one grid cell, however the caller shipped it.

    The cell shape both packages exchange: a mapping (what a JSON route serves) or an
    object with attributes (tcip-mcp's ``reference_grid.Cell``), carrying ``name`` plus
    the half-open native-pixel rect ``x0, y0, x1, y1``. One accessor so the renderer and
    the name lookup can never read the shape differently.
    """
    if isinstance(cell, dict):
        return (str(cell["name"]), float(cell["x0"]), float(cell["y0"]),
                float(cell["x1"]), float(cell["y1"]))
    return (str(cell.name), float(cell.x0), float(cell.y0),
            float(cell.x1), float(cell.y1))


def grid_to_rect(cell: str, cells: "list[Any]") -> tuple[float, float, float, float]:
    """Resolve a grid cell reference (e.g. 'B3') to the named cell's half-open native-pixel rect
    ``(x0, y0, x1, y1)``.

    The one cell-name lookup: every consumer of a cell name resolves through this, whether it
    wants the cell's center (:func:`grid_to_pixel`, for a point prompt) or its rect (a region a
    caller crops or bounds). ``cells`` is the caller's own cell list (see :func:`cell_fields` for
    the accepted shapes), the same list the overlay was rendered with: a cell name means nothing
    without the grid that produced it. Matching is case-insensitive and whitespace-stripped. A
    reference that is not a cell name at all raises ValueError naming the expected format; one
    that names no cell in this grid raises ValueError naming the grid's valid range.
    """
    wanted = cell.strip().upper()
    if re.fullmatch(r"[A-Z]+[0-9]+", wanted) is None:
        raise ValueError(f"Invalid cell reference: {cell!r}. Expected format like 'B3'.")
    if not cells:
        raise ValueError("cells is empty: there is no grid to resolve a cell name against")

    max_col = max_row = -1
    for c in cells:
        name, x0, y0, x1, y1 = cell_fields(c)
        if name.strip().upper() == wanted:
            return x0, y0, x1, y1
        parsed = re.fullmatch(r"([A-Z]+)([0-9]+)", name.strip().upper())
        if parsed is not None:
            max_col = max(max_col, column_index(parsed.group(1)))
            max_row = max(max_row, int(parsed.group(2)))
    hint = f" Use A1 through {column_label(max_col)}{max_row}." if max_col >= 0 else ""
    raise ValueError(f"Cell '{wanted}' is not in this grid.{hint}")


def grid_to_pixel(cell: str, cells: "list[Any]") -> tuple[float, float]:
    """Resolve a grid cell reference (e.g. 'B3') to the named cell's center in native pixels.

    The center of the rect :func:`grid_to_rect` resolves; see it for the accepted ``cells``
    shapes, the matching rules and the refusals.
    """
    x0, y0, x1, y1 = grid_to_rect(cell, cells)
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0
