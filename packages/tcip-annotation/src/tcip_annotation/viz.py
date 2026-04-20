"""Visualization rendering — draws annotations and predictions on images.

All functions return the output file path for consumption by view_image.
Default output directory: .tcip/artifacts/viz/

Coordinates: functions accept pixel coordinates. Use yolo_to_pixel() for
normalized YOLO coords.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont
from tcip_annotation.utils import auto_orient_image, get_image_dimensions

# 20-class color palette (RGB) — consistent with webview annotation canvas
COLOR_PALETTE: list[tuple[int, int, int]] = [
    (255, 0, 0),       # red
    (0, 255, 0),       # green
    (0, 0, 255),       # blue
    (255, 255, 0),     # yellow
    (255, 0, 255),     # magenta
    (0, 255, 255),     # cyan
    (255, 128, 0),     # orange
    (128, 0, 255),     # purple
    (0, 128, 255),     # sky blue
    (255, 0, 128),     # rose
    (128, 255, 0),     # lime
    (0, 255, 128),     # spring green
    (128, 128, 0),     # olive
    (128, 0, 128),     # dark purple
    (0, 128, 128),     # teal
    (255, 128, 128),   # salmon
    (128, 255, 128),   # light green
    (128, 128, 255),   # light blue
    (255, 255, 128),   # cream
    (255, 128, 255),   # pink
]

MAX_RENDER_EDGE = 1024


def _default_output(func_name: str, suffix: str = ".png") -> str:
    """Generate default output path in .tcip/artifacts/viz/."""
    viz_dir = Path(".tcip") / "artifacts" / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    # Include microseconds to avoid collision when many images render in one second
    us = time.time_ns() % 1_000_000
    return str(viz_dir / f"{ts}_{us:06d}_{func_name}{suffix}")


def _load_and_resize(image_path: str) -> Image.Image:
    """Load image, apply EXIF orientation, and resize if larger than MAX_RENDER_EDGE."""
    img = Image.open(image_path)
    img = auto_orient_image(img)
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_RENDER_EDGE:
        scale = MAX_RENDER_EDGE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _get_scale(orig_w: int, orig_h: int, render_w: int, render_h: int) -> tuple[float, float]:
    """Get scale factors from original image to render dimensions."""
    return render_w / orig_w, render_h / orig_h


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    return COLOR_PALETTE[class_id % len(COLOR_PALETTE)]


def _try_font(size: int = 12) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a TTF font, fall back to default bitmap font."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def yolo_to_pixel(
    cx: float, cy: float, bw: float, bh: float, img_w: int, img_h: int,
) -> tuple[float, float, float, float]:
    """Convert normalized YOLO center coords to pixel (x1, y1, x2, y2)."""
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h
    return x1, y1, x2, y2


def render_detections(
    image_path: str,
    boxes: list[dict],
    class_names: dict[int, str] | None = None,
    output_path: str | None = None,
    line_width: int = 2,
    conf_key: str | None = "confidence",
) -> str:
    """Draw bounding boxes on image. Returns output path.

    Args:
        image_path: Path to the source image.
        boxes: List of dicts with x1, y1, x2, y2, class_id (pixel coords).
               Optionally include 'confidence' for score display.
        class_names: Mapping from class_id to display name.
        output_path: Where to save. Defaults to .tcip/artifacts/viz/.
        line_width: Box outline width in pixels.
        conf_key: Key for confidence score in box dicts. None to hide scores.
    """
    output_path = output_path or _default_output("detections")
    class_names = class_names or {}

    orig_w, orig_h = get_image_dimensions(image_path)
    img = _load_and_resize(image_path)
    draw = ImageDraw.Draw(img)
    font = _try_font(12)
    sx, sy = _get_scale(orig_w, orig_h, img.size[0], img.size[1])

    for box in boxes:
        cid = box.get("class_id", 0)
        color = _color_for_class(cid)
        x1 = box["x1"] * sx
        y1 = box["y1"] * sy
        x2 = box["x2"] * sx
        y2 = box["y2"] * sy
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        label = class_names.get(cid, str(cid))
        if conf_key and conf_key in box:
            label += f" {box[conf_key]:.2f}"

        # Label background
        bbox = font.getbbox(label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = max(y1 - th - 4, 0)
        draw.rectangle([x1, label_y, x1 + tw + 4, label_y + th + 4], fill=color)
        draw.text((x1 + 2, label_y + 2), label, fill=(255, 255, 255), font=font)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def render_segmentations(
    image_path: str,
    polygons: list[dict],
    class_names: dict[int, str] | None = None,
    output_path: str | None = None,
    alpha: float = 0.3,
) -> str:
    """Draw filled polygons on image. Returns output path.

    Args:
        image_path: Path to the source image.
        polygons: List of dicts with 'points' (list of (x,y) pixel tuples)
                  and 'class_id'.
        class_names: Mapping from class_id to display name.
        output_path: Where to save.
        alpha: Fill transparency (0=transparent, 1=opaque).
    """
    output_path = output_path or _default_output("segmentations")
    class_names = class_names or {}

    orig_w, orig_h = get_image_dimensions(image_path)
    img = _load_and_resize(image_path)
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    font = _try_font(12)
    sx, sy = _get_scale(orig_w, orig_h, img.size[0], img.size[1])

    for poly in polygons:
        cid = poly.get("class_id", 0)
        color = _color_for_class(cid)
        pts = [(x * sx, y * sy) for x, y in poly["points"]]
        if len(pts) >= 3:
            draw.polygon(pts, fill=color + (int(255 * alpha),), outline=color)

            # Label at centroid
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            label = class_names.get(cid, str(cid))
            draw.text((cx, cy), label, fill=(255, 255, 255), font=font)

    img = Image.blend(img, overlay.convert("RGB"), alpha)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def render_comparison(
    image_path: str,
    gt_boxes: list[dict],
    pred_boxes: list[dict],
    matches: list[dict] | None = None,
    class_names: dict[int, str] | None = None,
    output_path: str | None = None,
) -> str:
    """Overlay GT (green) vs predictions (red) with optional match lines.

    Args:
        image_path: Path to the source image.
        gt_boxes: Ground truth boxes (x1, y1, x2, y2, class_id).
        pred_boxes: Prediction boxes (x1, y1, x2, y2, class_id, confidence).
        matches: Output from run_matching — list of matched pairs.
        class_names: Mapping from class_id to display name.
        output_path: Where to save.
    """
    output_path = output_path or _default_output("comparison")
    class_names = class_names or {}

    orig_w, orig_h = get_image_dimensions(image_path)
    img = _load_and_resize(image_path)
    draw = ImageDraw.Draw(img)
    font = _try_font(11)
    sx, sy = _get_scale(orig_w, orig_h, img.size[0], img.size[1])

    gt_color = (0, 255, 0)      # green for ground truth
    pred_color = (255, 0, 0)    # red for predictions
    match_color = (255, 255, 0) # yellow for match lines

    # Draw GT boxes
    for box in gt_boxes:
        x1, y1 = box["x1"] * sx, box["y1"] * sy
        x2, y2 = box["x2"] * sx, box["y2"] * sy
        draw.rectangle([x1, y1, x2, y2], outline=gt_color, width=2)
        label = "GT:" + class_names.get(box.get("class_id", 0), str(box.get("class_id", 0)))
        draw.text((x1, max(y1 - 14, 0)), label, fill=gt_color, font=font)

    # Draw pred boxes
    for box in pred_boxes:
        x1, y1 = box["x1"] * sx, box["y1"] * sy
        x2, y2 = box["x2"] * sx, box["y2"] * sy
        draw.rectangle([x1, y1, x2, y2], outline=pred_color, width=2)
        label = "P:" + class_names.get(box.get("class_id", 0), str(box.get("class_id", 0)))
        if "confidence" in box:
            label += f" {box['confidence']:.2f}"
        draw.text((x1, y2 + 2), label, fill=pred_color, font=font)

    # Draw match lines (center-to-center)
    if matches:
        for m in matches:
            gt = m.get("gt", m.get("ground_truth", {}))
            pred = m.get("pred", m.get("prediction", {}))
            if gt and pred:
                gt_cx = (gt.get("x1", 0) + gt.get("x2", 0)) / 2 * sx
                gt_cy = (gt.get("y1", 0) + gt.get("y2", 0)) / 2 * sy
                pr_cx = (pred.get("x1", 0) + pred.get("x2", 0)) / 2 * sx
                pr_cy = (pred.get("y1", 0) + pred.get("y2", 0)) / 2 * sy
                draw.line([(gt_cx, gt_cy), (pr_cx, pr_cy)], fill=match_color, width=1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def render_grid(
    image_paths: list[str],
    titles: list[str] | None = None,
    cols: int = 4,
    cell_size: int = 256,
    output_path: str | None = None,
) -> str:
    """Tile multiple images into a grid. Returns output path.

    Args:
        image_paths: List of image file paths.
        titles: Optional per-image titles.
        cols: Number of columns in the grid.
        cell_size: Size of each cell (images resized to fit).
        output_path: Where to save.
    """
    output_path = output_path or _default_output("grid")
    n = len(image_paths)
    if n == 0:
        # Empty grid — create a small placeholder
        img = Image.new("RGB", (cell_size, cell_size), (64, 64, 64))
        draw = ImageDraw.Draw(img)
        draw.text((10, cell_size // 2), "No images", fill=(200, 200, 200))
        img.save(output_path)
        return output_path

    rows = (n + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell_size, rows * cell_size), (32, 32, 32))
    draw = ImageDraw.Draw(grid)
    font = _try_font(11)

    for i, path in enumerate(image_paths):
        row, col = divmod(i, cols)
        try:
            thumb = Image.open(path)
            thumb = auto_orient_image(thumb)
            thumb = thumb.convert("RGB")
            thumb.thumbnail((cell_size, cell_size), Image.LANCZOS)
            # Center in cell
            x_off = col * cell_size + (cell_size - thumb.size[0]) // 2
            y_off = row * cell_size + (cell_size - thumb.size[1]) // 2
            grid.paste(thumb, (x_off, y_off))
        except Exception:
            # Draw error placeholder
            x_off = col * cell_size
            y_off = row * cell_size
            draw.rectangle([x_off, y_off, x_off + cell_size, y_off + cell_size], fill=(64, 0, 0))
            draw.text((x_off + 4, y_off + cell_size // 2), "Error", fill=(255, 128, 128))

        if titles and i < len(titles):
            tx = col * cell_size + 4
            ty = row * cell_size + 2
            # Dark background for readability
            bbox = font.getbbox(titles[i])
            tw = bbox[2] - bbox[0]
            draw.rectangle([tx, ty, tx + tw + 4, ty + 16], fill=(0, 0, 0, 180))
            draw.text((tx + 2, ty + 1), titles[i], fill=(255, 255, 255), font=font)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return output_path


def render_confusion_examples(
    worst_predictions: list[dict],
    images_dir: str | None = None,
    output_dir: str | None = None,
) -> list[str]:
    """Render the worst prediction cases for visual failure analysis.

    Args:
        worst_predictions: From get_worst_predictions — list of dicts
            with 'image', 'gt_boxes', 'pred_boxes', optional 'error_type'.
        images_dir: Directory containing source images.
        output_dir: Where to save renders. Defaults to .tcip/artifacts/viz/failures/.

    Returns:
        List of output paths (one per failure case).
    """
    output_dir = output_dir or str(Path(".tcip") / "artifacts" / "viz" / "failures")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    for i, wp in enumerate(worst_predictions):
        img_name = wp.get("image", f"case_{i}")
        img_path = wp.get("image_path")
        if not img_path and images_dir:
            # Try common extensions
            for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                candidate = Path(images_dir) / f"{img_name}{ext}"
                if candidate.is_file():
                    img_path = str(candidate)
                    break

        if not img_path or not Path(img_path).is_file():
            continue

        out = str(Path(output_dir) / f"failure_{i:03d}_{Path(img_name).stem}.png")
        render_comparison(
            image_path=img_path,
            gt_boxes=wp.get("gt_boxes", []),
            pred_boxes=wp.get("pred_boxes", []),
            matches=wp.get("matches"),
            output_path=out,
        )
        paths.append(out)

    return paths


def render_candidates(
    image_path: str,
    candidates: list[dict],
    output_path: str | None = None,
    alpha: float = 0.35,
) -> str:
    """Render numbered SAM candidate masks on image for agent review.

    Each candidate is drawn as a semi-transparent colored polygon with a
    large numbered label. Colors cycle through the palette.

    Args:
        image_path: Path to the source image.
        candidates: List of dicts from auto_mask(), each with:
            candidate_id, bbox, polygon, area, stability_score, predicted_iou.
        output_path: Where to save. Defaults to .tcip/artifacts/viz/.
        alpha: Fill transparency (0=transparent, 1=opaque).

    Returns:
        Output path to the rendered image.
    """
    output_path = output_path or _default_output("candidates")

    orig_w, orig_h = get_image_dimensions(image_path)
    img = _load_and_resize(image_path)
    rw, rh = img.size
    sx, sy = _get_scale(orig_w, orig_h, rw, rh)

    # Draw semi-transparent polygons on overlay
    overlay = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    font_label = _try_font(16)
    font_small = _try_font(10)

    for cand in candidates:
        cid = cand["candidate_id"]
        color = COLOR_PALETTE[cid % len(COLOR_PALETTE)]
        pts = [(x * sx, y * sy) for x, y in cand["polygon"]]
        if len(pts) < 3:
            continue

        # Fill polygon
        fill = color + (int(255 * alpha),)
        overlay_draw.polygon(pts, fill=fill, outline=color)

        # Draw candidate number at centroid
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)

        # White circle background for number
        num_text = str(cid)
        bbox = font_label.getbbox(num_text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        radius = max(tw, th) // 2 + 6
        overlay_draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(0, 0, 0, 200),
        )
        overlay_draw.text(
            (cx - tw // 2, cy - th // 2), num_text,
            fill=(255, 255, 255), font=font_label,
        )

        # Stability + IoU info below the number
        info = f"s={cand.get('stability_score', 0):.2f}"
        overlay_draw.text((cx - 12, cy + radius + 2), info, fill=color, font=font_small)

    # Composite overlay onto image
    img_rgba = img.convert("RGBA")
    composited = Image.alpha_composite(img_rgba, overlay)
    result = composited.convert("RGB")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    return output_path


def render_grid_overlay(
    image_path: str,
    cols: int = 8,
    rows: int = 6,
    output_path: str | None = None,
) -> str:
    """Render image with a labeled grid overlay for spatial referencing.

    The grid uses letter columns (A-H) and number rows (1-6). The agent
    can reference cells like 'B3' or 'F5' to indicate object locations,
    which are converted to pixel coordinates via grid_to_pixel().

    Args:
        image_path: Path to the source image.
        cols: Number of grid columns (default 8, labeled A-H).
        rows: Number of grid rows (default 6, labeled 1-6).
        output_path: Where to save.

    Returns:
        Output path to the rendered image.
    """
    output_path = output_path or _default_output("grid_overlay")

    img = _load_and_resize(image_path)
    rw, rh = img.size
    draw = ImageDraw.Draw(img)

    cell_w = rw / cols
    cell_h = rh / rows

    grid_color = (255, 255, 0, 128)  # semi-transparent yellow
    label_color = (255, 255, 0)

    font = _try_font(14)

    # Draw vertical lines
    for c in range(cols + 1):
        x = int(c * cell_w)
        draw.line([(x, 0), (x, rh)], fill=grid_color, width=1)

    # Draw horizontal lines
    for r in range(rows + 1):
        y = int(r * cell_h)
        draw.line([(0, y), (rw, y)], fill=grid_color, width=1)

    # Label each cell at top-left corner
    for c in range(cols):
        for r in range(rows):
            label = f"{chr(ord('A') + c)}{r + 1}"
            x = int(c * cell_w) + 3
            y = int(r * cell_h) + 2
            # Dark background for readability
            bbox = font.getbbox(label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([x - 1, y - 1, x + tw + 3, y + th + 3], fill=(0, 0, 0, 180))
            draw.text((x + 1, y), label, fill=label_color, font=font)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path
