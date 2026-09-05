"""Visualization rendering: draws annotations and predictions on images.

All functions return the output file path for the agent's own image-capable read tool to consume.
Default output directory: .tcip/artifacts/viz/

The renderers take display pixels, never a path: whoever holds the raster decodes it, bounds its
resolution and composites its bands, so a renderer never has to know how a multi-band or
overview-bearing raster reads. Annotation coordinates stay in the raster's own full-resolution
frame, so a renderer handed reduced pixels also takes the ``native_size`` those coordinates are in
and scales them itself. ``render_grid`` is the exception: it tiles already-rendered artifacts and
so takes their paths.

Coordinates: functions accept pixel coordinates.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont
from tcip_annotation.utils import auto_orient_image

if TYPE_CHECKING:
    import numpy as np

# 20-class color palette (RGB), indexed by class id modulo its length; the same palette across
# every renderer in this module, not the GUI annotation canvas's own colors
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

_PLATFORM_ROOT_ENV = "TCIP_STATE_ROOT"
"""Mirrors ``tcip_mcp.project_paths.ENV_VAR``. This package must not import ``tcip_mcp`` (see
packages/tcip-annotation/CLAUDE.md), so the platform-state-root variable name is restated here
rather than imported; a test holds the two strings equal."""


def _viz_base() -> Path:
    """The ``.tcip`` base for viz output. Honors ``TCIP_STATE_ROOT`` (the platform-state root the
    MCP server / web backend pin to the active project) so renders land under the project, not the
    process CWD (the agent's CWD is often the repo, which fragmented artifacts away from the
    project and returned a CWD-relative path callers couldn't resolve). Falls back to CWD-relative
    for standalone ``tcip_annotation`` use."""
    import os

    root = os.environ.get(_PLATFORM_ROOT_ENV)
    return (Path(root) if root else Path(".")) / ".tcip"


def _default_output(func_name: str, suffix: str = ".png") -> str:
    """Generate an absolute default output path under ``<root>/.tcip/artifacts/viz/``."""
    viz_dir = (_viz_base() / "artifacts" / "viz").resolve()
    viz_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    # Include microseconds to avoid collision when many images render in one second
    us = time.time_ns() % 1_000_000
    return str(viz_dir / f"{ts}_{us:06d}_{func_name}{suffix}")


def _rgb_frame(image: "Image.Image | np.ndarray") -> Image.Image:
    """The caller's display pixels as an RGB frame this module can draw on.

    A ``uint8 [H, W, 3]`` array or any PIL image; the returned frame is always a new one, so
    drawing on it can never mutate what the caller passed. An array of another dtype or channel
    count is refused rather than coerced: the only reading that would rescue it is a display
    stretch or a band selection, and those belong to whoever read the raster and knows its bounds.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    import numpy as np

    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"a renderer takes uint8 [H, W, 3] display pixels or a PIL image, got "
            f"{arr.shape} {arr.dtype}"
        )
    return Image.fromarray(arr, mode="RGB")


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


def render_detections(
    image: "Image.Image | np.ndarray",
    boxes: list[dict],
    *,
    native_size: tuple[int, int],
    class_names: dict[int, str] | None = None,
    output_path: str | None = None,
    line_width: int = 2,
    conf_key: str | None = "confidence",
) -> str:
    """Draw bounding boxes on display pixels. Returns output path.

    Args:
        image: Display pixels (uint8 RGB array or PIL image).
        boxes: List of dicts with x1, y1, x2, y2, class_id (pixel coords in the native frame).
               Optionally include 'confidence' for score display.
        native_size: ``(width, height)`` of the frame ``boxes`` are measured in.
        class_names: Mapping from class_id to display name.
        output_path: Where to save. Defaults to .tcip/artifacts/viz/.
        line_width: Box outline width in pixels.
        conf_key: Key for confidence score in box dicts. None to hide scores.
    """
    output_path = output_path or _default_output("detections")
    class_names = class_names or {}

    orig_w, orig_h = native_size
    img = _rgb_frame(image)
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
    image: "Image.Image | np.ndarray",
    polygons: list[dict],
    *,
    native_size: tuple[int, int],
    class_names: dict[int, str] | None = None,
    output_path: str | None = None,
    alpha: float = 0.3,
) -> str:
    """Draw filled polygons on display pixels. Returns output path.

    Args:
        image: Display pixels (uint8 RGB array or PIL image).
        polygons: List of dicts with 'rings' (list of rings, each a list of (x,y) pixel tuples in
                  the native frame; an occlusion-split instance is more than one ring, drawn as one
                  instance) and 'class_id'.
        native_size: ``(width, height)`` of the frame the rings are measured in.
        class_names: Mapping from class_id to display name.
        output_path: Where to save.
        alpha: Fill transparency (0=transparent, 1=opaque).
    """
    output_path = output_path or _default_output("segmentations")
    class_names = class_names or {}

    orig_w, orig_h = native_size
    img = _rgb_frame(image)
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    font = _try_font(12)
    sx, sy = _get_scale(orig_w, orig_h, img.size[0], img.size[1])

    for poly in polygons:
        cid = poly.get("class_id", 0)
        color = _color_for_class(cid)
        all_pts: list[tuple[float, float]] = []
        for ring in poly.get("rings", []):
            pts = [(x * sx, y * sy) for x, y in ring]
            if len(pts) >= 3:
                draw.polygon(pts, fill=color + (int(255 * alpha),), outline=color)
                all_pts.extend(pts)
        if all_pts:
            # Label once per instance, at the centroid of every drawn ring combined.
            cx = sum(p[0] for p in all_pts) / len(all_pts)
            cy = sum(p[1] for p in all_pts) / len(all_pts)
            label = class_names.get(cid, str(cid))
            draw.text((cx, cy), label, fill=(255, 255, 255), font=font)

    img = Image.blend(img, overlay.convert("RGB"), alpha)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def render_comparison(
    image: "Image.Image | np.ndarray",
    gt_boxes: list[dict],
    pred_boxes: list[dict],
    *,
    native_size: tuple[int, int],
    matches: list[dict] | None = None,
    class_names: dict[int, str] | None = None,
    output_path: str | None = None,
) -> str:
    """Overlay GT (green) vs predictions (red) with optional match lines.

    Args:
        image: Display pixels (uint8 RGB array or PIL image).
        gt_boxes: Ground truth boxes (x1, y1, x2, y2, class_id) in the native frame.
        pred_boxes: Prediction boxes (x1, y1, x2, y2, class_id, confidence) in the native frame.
        native_size: ``(width, height)`` of the frame the boxes are measured in.
        matches: ``compute_matches``'s own ``tp`` entries (``{gt_idx, pred_idx, ...}``), indexing
            into ``gt_boxes``/``pred_boxes`` in the order they were built from the same lists.
        class_names: Mapping from class_id to display name.
        output_path: Where to save.
    """
    output_path = output_path or _default_output("comparison")
    class_names = class_names or {}

    orig_w, orig_h = native_size
    img = _rgb_frame(image)
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

    # Draw match lines (center-to-center), resolved from the gt/pred lists already in hand.
    if matches:
        for m in matches:
            gt = gt_boxes[m["gt_idx"]]
            pred = pred_boxes[m["pred_idx"]]
            gt_cx = (gt["x1"] + gt["x2"]) / 2 * sx
            gt_cy = (gt["y1"] + gt["y2"]) / 2 * sy
            pr_cx = (pred["x1"] + pred["x2"]) / 2 * sx
            pr_cy = (pred["y1"] + pred["y2"]) / 2 * sy
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
        # Empty grid: create a small placeholder
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
            thumb: Image.Image = Image.open(path)
            thumb = auto_orient_image(thumb)
            thumb = thumb.convert("RGB")
            thumb.thumbnail((cell_size, cell_size), Image.Resampling.LANCZOS)
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


def render_candidates(
    image: "Image.Image | np.ndarray",
    candidates: list[dict],
    *,
    native_size: tuple[int, int],
    output_path: str | None = None,
    alpha: float = 0.35,
) -> str:
    """Render numbered proposal-engine candidate masks on display pixels for agent review.

    Each candidate is drawn as a semi-transparent colored polygon with a
    large numbered label. Colors cycle through the palette. Engine-agnostic.
    Every ring of a candidate is drawn: an occlusion-split proposal must look split, not whole.

    Args:
        image: Display pixels (uint8 RGB array or PIL image).
        candidates: Neutral candidate dicts, each with candidate_id, bbox, rings, area, score
            (SAM populates these via its proposer adapter); coordinates in the native frame.
        native_size: ``(width, height)`` of the frame the candidates are measured in.
        output_path: Where to save. Defaults to .tcip/artifacts/viz/.
        alpha: Fill transparency (0=transparent, 1=opaque).

    Returns:
        Output path to the rendered image.
    """
    output_path = output_path or _default_output("candidates")

    orig_w, orig_h = native_size
    img = _rgb_frame(image)
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
        rings = [[(x * sx, y * sy) for x, y in ring] for ring in cand["rings"]]
        rings = [r for r in rings if len(r) >= 3]
        if not rings:
            continue

        # Fill every ring; one number for the candidate as a whole
        fill = color + (int(255 * alpha),)
        for ring in rings:
            overlay_draw.polygon(ring, fill=fill, outline=color)

        pts = [p for ring in rings for p in ring]
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

        # Proposal score below the number (neutral across engines)
        info = f"s={cand.get('score', 0):.2f}"
        overlay_draw.text((cx - 12, cy + radius + 2), info, fill=color, font=font_small)

    # Composite overlay onto image
    img_rgba = img.convert("RGBA")
    composited = Image.alpha_composite(img_rgba, overlay)
    result = composited.convert("RGB")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    return output_path


def render_grid_overlay(
    image: "Image.Image | np.ndarray",
    cells: list,
    *,
    native_size: tuple[int, int],
    output_path: str | None = None,
) -> str:
    """Render display pixels with the caller's labeled reference-grid cells overlaid.

    ``cells`` is the caller's own cell list, each entry a mapping or an object carrying
    ``name`` plus the half-open native-pixel rect ``x0, y0, x1, y1`` (see
    ``sam_wrapper.cell_fields``, the shape tcip-mcp's reference grid computes and its
    coverage route serves). Rects scale by the rendered/native ratio like the other
    renderers, so the lines land on the true cell boundaries, which under a clamped grid
    are non-uniform at the edges.

    Boundaries always draw. A cell's name draws only when the rendered cell is at least
    24 px on its short edge and wide enough to hold the label's own backing box: below
    the edge floor the 14 px label covers the cell instead of labeling it, and a label
    wider than its cell runs into the neighbor's. The floor is a display concern of this
    renderer alone, not a platform constant.

    Args:
        image: Display pixels (uint8 RGB array or PIL image).
        cells: The grid's cells, names plus native-pixel rects.
        native_size: The raster's own (width, height) the cell rects are measured in.
        output_path: Where to save.

    Returns:
        Output path to the rendered image.
    """
    from tcip_annotation.sam_wrapper import cell_fields

    if not cells:
        raise ValueError("cells is empty: there is no grid to render")
    output_path = output_path or _default_output("grid_overlay")

    img = _rgb_frame(image)
    rw, rh = img.size
    sx, sy = _get_scale(native_size[0], native_size[1], rw, rh)
    draw = ImageDraw.Draw(img)

    grid_color = (255, 255, 0)
    label_color = (255, 255, 0)
    label_min_edge = 24

    font = _try_font(14)

    scaled = [(name, x0 * sx, y0 * sy, x1 * sx, y1 * sy)
              for name, x0, y0, x1, y1 in map(cell_fields, cells)]

    for _name, x0, y0, x1, y1 in scaled:
        # A far edge that scales to the frame size lands one past the last pixel row or
        # column and PIL clips the whole line; pin it inside so the outer boundary renders.
        rx1 = min(x1, rw - 1)
        ry1 = min(y1, rh - 1)
        draw.line([(x0, y0), (x0, ry1)], fill=grid_color, width=1)
        draw.line([(rx1, y0), (rx1, ry1)], fill=grid_color, width=1)
        draw.line([(x0, y0), (rx1, y0)], fill=grid_color, width=1)
        draw.line([(x0, ry1), (rx1, ry1)], fill=grid_color, width=1)

    for name, x0, y0, x1, y1 in scaled:
        if min(x1 - x0, y1 - y0) < label_min_edge:
            continue
        bbox = font.getbbox(name)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw + 6 > x1 - x0:
            continue
        x = int(x0) + 3
        y = int(y0) + 2
        # Dark background for readability
        draw.rectangle([x - 1, y - 1, x + tw + 3, y + th + 3], fill=(0, 0, 0))
        draw.text((x + 1, y), name, fill=label_color, font=font)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


# ── Live GUI canvas render (display-resolved shapes from the canvas-state push) ──


def _hex_rgb(color: str, fallback: tuple[int, int, int] = (255, 255, 255)) -> tuple[int, int, int]:
    """``#RRGGBB`` (or ``#RGB``) → RGB tuple; anything unparsable falls back to white."""
    c = (color or "").lstrip("#")
    try:
        if len(c) == 3:
            return tuple(int(ch * 2, 16) for ch in c)  # type: ignore[return-value]
        if len(c) >= 6:
            return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    except ValueError:
        pass
    return fallback


def _dashed_segment(draw, p1, p2, fill, width: int, dash: float, gap: float) -> None:
    """Draw p1→p2 as dashes (PIL has no native dashed lines)."""
    x1, y1 = p1
    x2, y2 = p2
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if length < 1e-6:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        draw.line([(x1 + ux * pos, y1 + uy * pos), (x1 + ux * end, y1 + uy * end)],
                  fill=fill, width=width)
        pos = end + gap


def _draw_path(draw, pts, color, width: int, dashed: bool, closed: bool) -> None:
    seg_pts = list(pts) + ([pts[0]] if closed and len(pts) > 2 else [])
    if dashed:
        for a, b in zip(seg_pts, seg_pts[1:]):
            _dashed_segment(draw, a, b, color, width, dash=8.0, gap=4.0)
    elif len(seg_pts) > 1:
        draw.line(seg_pts, fill=color, width=width, joint="curve")


def render_canvas_state(
    image: "Image.Image | np.ndarray",
    shapes: list[dict],
    *,
    origin: tuple[float, float],
    scale: float,
    output_path: str | None = None,
) -> str:
    """Render the live GUI canvas: display-resolved shapes over the pixels the human is viewing.

    ``shapes`` come from the canvas-state push, each already carrying the exact symbology the
    GUI rendered: ``{kind: box|polygon|polyline|point, xyxy|points (pixel), color '#hex', fill?,
    dashed?, label?}``, so this draws what the annotator sees rather than re-deriving colors. A
    ``point`` carries one coordinate in ``points`` and draws as the GUI's mark (a core with radial
    ticks); it is never widened into a box, which would show the agent an extent the annotation
    does not claim.

    ``image`` is whatever region of the raster the caller read (the human's viewport, or the whole
    frame), ``origin`` is that region's top-left corner in the raster's own full-resolution grid
    and ``scale`` is the served resolution as a fraction of native. Shape coordinates arrive in
    the native grid and are placed by those two, so the caller reads exactly the pixels it wants
    shown rather than this decoding a whole raster to crop it.
    """
    if output_path is None:
        output_path = _default_output("canvas", suffix=".jpg")

    img = _rgb_frame(image)
    ox, oy = float(origin[0]), float(origin[1])
    k = float(scale)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    out_w = img.size[0]
    lw = max(1, round(out_w / 700))
    font = _try_font(max(11, out_w // 90))
    dot_r = max(2.0, out_w / 450)

    def tx(p) -> tuple[float, float]:
        return ((float(p[0]) - ox) * k, (float(p[1]) - oy) * k)

    # Two passes over one overlay: all fills first, then all outlines/vertices. ImageDraw
    # replaces pixels (it does not composite), so a later shape's translucent fill would
    # otherwise punch its silhouette out of earlier shapes' opaque outlines.
    parsed: list[tuple[dict, list[tuple[float, float]], tuple[int, int, int], bool]] = []
    labels: list[tuple[tuple[float, float], str, tuple[int, int, int]]] = []
    for s in shapes:
        if not isinstance(s, dict):
            continue
        color = _hex_rgb(str(s.get("color", "")))
        kind = s.get("kind")
        try:
            if kind == "box" and s.get("xyxy"):
                x1, y1 = tx(s["xyxy"][:2])
                x2, y2 = tx(s["xyxy"][2:4])
                pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                closed = True
            elif kind == "point" and s.get("points"):
                pts = [tx(s["points"][0])]
                closed = False
            elif kind in ("polygon", "polyline") and s.get("points"):
                pts = [tx(p) for p in s["points"]]
                if len(pts) < 2:
                    continue
                closed = kind == "polygon"
            else:
                continue
        except (TypeError, ValueError, IndexError):
            continue  # a malformed shape must never sink the whole render
        parsed.append((s, pts, color, closed))
        label = s.get("label")
        if label:
            labels.append((pts[0], str(label), color))

    for s, pts, color, closed in parsed:  # pass 1: fills
        if s.get("fill") and closed and len(pts) >= 3:
            draw.polygon(pts, fill=color + (38,))
    for s, pts, color, closed in parsed:  # pass 2: outlines + vertices
        if s.get("kind") == "point":
            # The GUI's reticle: a core plus four radial ticks converging on the coordinate, the
            # mark that distinguishes a location from a very small box on the same canvas.
            px, py = pts[0]
            core = dot_r * 1.6
            inner, outer = core * 1.9, core * 3.2
            draw.ellipse([px - core, py - core, px + core, py + core], fill=color + (255,))
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                draw.line(
                    [(px + dx * inner, py + dy * inner), (px + dx * outer, py + dy * outer)],
                    fill=color + (255,), width=lw,
                )
            continue
        _draw_path(draw, pts, color + (255,), lw, bool(s.get("dashed")), closed=closed)
        if s.get("kind") == "polyline":  # in-progress drawing: show the laid vertices
            for px, py in pts:
                draw.ellipse([px - dot_r, py - dot_r, px + dot_r, py + dot_r],
                             fill=color + (255,))

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw2 = ImageDraw.Draw(img)
    for (ax, ay), text, color in labels:  # halo text, drawn after compositing so it stays crisp
        x, y = ax + 2, max(0.0, ay - (out_w // 90) - 4)
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            draw2.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
        draw2.text((x, y), text, fill=color, font=font)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=88)
    return output_path
