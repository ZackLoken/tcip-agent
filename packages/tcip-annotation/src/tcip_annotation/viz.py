"""Visualization rendering: annotations and predictions drawn on images.

Each render is written to the ``output_path`` its caller names and that path returned. The
renderers take display pixels, never a path, and pixel coordinates in the raster's own
full-resolution frame, so a renderer handed reduced pixels also takes the ``native_size`` those
coordinates are in. ``render_grid`` tiles already-rendered artifacts and so takes their paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, NamedTuple

from PIL import Image, ImageDraw, ImageFont
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    Strict,
    StringConstraints,
)

from tcip_annotation.utils import auto_orient_image

if TYPE_CHECKING:
    import numpy as np

    from tcip_annotation.grid import Cell

type Pixels = Image.Image | np.ndarray
"""Display pixels: a PIL image or a ``uint8 [H, W, 3]`` array."""

type FrameSize = tuple[int, int]
"""A frame's ``(width, height)`` in pixels."""

type Coord = tuple[float, float]
"""One ``(x, y)`` pixel position."""

type Dash = tuple[float, float]
"""A stroke's ``(on, off)`` dash lengths in pixels."""

type Color = tuple[int, int, int]
"""An RGB color."""

type ColorAlpha = tuple[int, int, int, int]
"""An RGBA color."""

type Font = ImageFont.FreeTypeFont | ImageFont.ImageFont
"""A loaded font, TrueType or PIL's bitmap default."""

# 20-class color palette (RGB), indexed by class id modulo its length; the same palette across
# every renderer in this module, not the GUI annotation canvas's own colors
COLOR_PALETTE: list[Color] = [
    (255, 0, 0),  # red
    (0, 255, 0),  # green
    (0, 0, 255),  # blue
    (255, 255, 0),  # yellow
    (255, 0, 255),  # magenta
    (0, 255, 255),  # cyan
    (255, 128, 0),  # orange
    (128, 0, 255),  # purple
    (0, 128, 255),  # sky blue
    (255, 0, 128),  # rose
    (128, 255, 0),  # lime
    (0, 255, 128),  # spring green
    (128, 128, 0),  # olive
    (128, 0, 128),  # dark purple
    (0, 128, 128),  # teal
    (255, 128, 128),  # salmon
    (128, 255, 128),  # light green
    (128, 128, 255),  # light blue
    (255, 255, 128),  # cream
    (255, 128, 255),  # pink
]


def _rgb_frame(image: Pixels) -> Image.Image:
    """Return the caller's display pixels as an RGB frame this module can draw on.

    A ``uint8 [H, W, 3]`` array or any PIL image; the returned frame is always a new one. An array
    of another dtype or channel count is refused.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    import numpy as np

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"a renderer takes uint8 [H, W, 3] display pixels or a PIL image, got "
            f"{image.shape} {image.dtype}"
        )
    return Image.fromarray(image, mode="RGB")


def _get_scale(
    orig_w: int, orig_h: int, render_w: int, render_h: int
) -> tuple[float, float]:
    """Get scale factors from original image to render dimensions."""
    return render_w / orig_w, render_h / orig_h


def _color_for_class(class_id: int) -> Color:
    """Return the palette color of ``class_id``, cycling through :data:`COLOR_PALETTE`."""
    return COLOR_PALETTE[class_id % len(COLOR_PALETTE)]


def _saved(
    img: Image.Image, output_path: str, *, quality: int | None = None
) -> str:
    """Write ``img`` to ``output_path``, making its directory first, and return the path.

    ``quality`` is the JPEG quality ``Image.save`` takes; omitted, the writer's own default holds.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if quality is None:
        img.save(output_path)
    else:
        img.save(output_path, quality=quality)
    return output_path


def _label(
    draw: ImageDraw.ImageDraw,
    xy: Coord,
    text: str,
    font: Font,
    background: Color,
    fill: Color = (255, 255, 255),
) -> None:
    """Draw ``text`` at ``xy`` over a ``background`` box padded two pixels each side."""
    x, y = xy
    left, top, right, bottom = font.getbbox(text)
    draw.rectangle(
        [x, y, x + right - left + 4, y + bottom - top + 4], fill=background
    )
    draw.text((x + 2, y + 2), text, fill=fill, font=font)


def _try_font(size: int = 12) -> Font:
    """Try to load a TTF font, fall back to default bitmap font."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def render_detections(
    image: Pixels,
    boxes: list[dict],
    *,
    native_size: FrameSize,
    class_names: dict[int, str] | None = None,
    output_path: str,
    line_width: int = 2,
) -> str:
    """Draw bounding boxes on display pixels. Returns output path.

    Args:
        image: Display pixels (uint8 RGB array or PIL image).
        boxes: List of dicts with x1, y1, x2, y2, class_id (pixel coords in the native frame);
               a prediction's also carries its 'score', which its label shows.
        native_size: ``(width, height)`` of the frame ``boxes`` are measured in.
        class_names: Mapping from class_id to display name.
        output_path: Where to save.
        line_width: Box outline width in pixels.

    """
    class_names = class_names or {}

    orig_w, orig_h = native_size
    img = _rgb_frame(image)
    draw = ImageDraw.Draw(img)
    font = _try_font(12)
    sx, sy = _get_scale(orig_w, orig_h, img.size[0], img.size[1])

    for box in boxes:
        cid = box["class_id"]
        color = _color_for_class(cid)
        x1 = box["x1"] * sx
        y1 = box["y1"] * sy
        x2 = box["x2"] * sx
        y2 = box["y2"] * sy
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        label = class_names.get(cid, str(cid))
        if "score" in box:
            label += f" {box['score']:.2f}"
        top, bottom = font.getbbox(label)[1::2]
        _label(draw, (x1, max(y1 - (bottom - top) - 4, 0)), label, font, color)

    return _saved(img, output_path)


def render_segmentations(
    image: Pixels,
    polygons: list[dict],
    *,
    native_size: FrameSize,
    class_names: dict[int, str] | None = None,
    output_path: str,
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
    class_names = class_names or {}

    orig_w, orig_h = native_size
    img = _rgb_frame(image)
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    font = _try_font(12)
    sx, sy = _get_scale(orig_w, orig_h, img.size[0], img.size[1])

    for poly in polygons:
        cid = poly["class_id"]
        color = _color_for_class(cid)
        all_pts: list[Coord] = []
        for ring in poly["rings"]:
            pts = [(x * sx, y * sy) for x, y in ring]
            draw.polygon(pts, fill=color + (int(255 * alpha),), outline=color)
            all_pts.extend(pts)
        if all_pts:
            # Label once per instance, at the centroid of every drawn ring combined.
            cx = sum(p[0] for p in all_pts) / len(all_pts)
            cy = sum(p[1] for p in all_pts) / len(all_pts)
            label = class_names.get(cid, str(cid))
            draw.text((cx, cy), label, fill=(255, 255, 255), font=font)

    return _saved(Image.blend(img, overlay.convert("RGB"), alpha), output_path)


def render_comparison(
    image: Pixels,
    gt_boxes: list[dict],
    pred_boxes: list[dict],
    *,
    native_size: FrameSize,
    matches: list[tuple[int, int]] | None = None,
    class_names: dict[int, str] | None = None,
    output_path: str,
) -> str:
    """Overlay GT (green) vs predictions (red) with optional match lines.

    Args:
        image: Display pixels (uint8 RGB array or PIL image).
        gt_boxes: Ground truth boxes (x1, y1, x2, y2, class_id) in the native frame.
        pred_boxes: Prediction boxes (x1, y1, x2, y2, class_id, score) in the native frame.
        native_size: ``(width, height)`` of the frame the boxes are measured in.
        matches: ``(gt_idx, pred_idx)`` pairs indexing into ``gt_boxes``/``pred_boxes`` in the
            order they were built from the same lists.
        class_names: Mapping from class_id to display name.
        output_path: Where to save.

    """
    class_names = class_names or {}

    orig_w, orig_h = native_size
    img = _rgb_frame(image)
    draw = ImageDraw.Draw(img)
    font = _try_font(11)
    sx, sy = _get_scale(orig_w, orig_h, img.size[0], img.size[1])

    gt_color = (0, 255, 0)  # green for ground truth
    pred_color = (255, 0, 0)  # red for predictions
    match_color = (255, 255, 0)  # yellow for match lines

    # Draw GT boxes
    for box in gt_boxes:
        x1, y1 = box["x1"] * sx, box["y1"] * sy
        x2, y2 = box["x2"] * sx, box["y2"] * sy
        draw.rectangle([x1, y1, x2, y2], outline=gt_color, width=2)
        label = "GT:" + class_names.get(box["class_id"], str(box["class_id"]))
        draw.text((x1, max(y1 - 14, 0)), label, fill=gt_color, font=font)

    # Draw pred boxes
    for box in pred_boxes:
        x1, y1 = box["x1"] * sx, box["y1"] * sy
        x2, y2 = box["x2"] * sx, box["y2"] * sy
        draw.rectangle([x1, y1, x2, y2], outline=pred_color, width=2)
        label = "P:" + class_names.get(box["class_id"], str(box["class_id"]))
        label += f" {box['score']:.2f}"
        draw.text((x1, y2 + 2), label, fill=pred_color, font=font)

    # Draw match lines (center-to-center), resolved from the gt/pred lists already in hand.
    if matches:
        from tcip_annotation.matching import box_centers

        for gt_idx, pred_idx in matches:
            ends = box_centers(
                [
                    [b["x1"], b["y1"], b["x2"], b["y2"]]
                    for b in (gt_boxes[gt_idx], pred_boxes[pred_idx])
                ]
            ) * (sx, sy)
            draw.line(
                [tuple(end) for end in ends.tolist()],
                fill=match_color,
                width=1,
            )

    return _saved(img, output_path)


def render_grid(
    image_paths: list[str],
    titles: list[str] | None = None,
    cols: int = 4,
    cell_size: int = 256,
    *,
    output_path: str,
) -> str:
    """Tile multiple images into a grid. Returns output path.

    Args:
        image_paths: List of image file paths.
        titles: Optional per-image titles.
        cols: Number of columns in the grid.
        cell_size: Size of each cell (images resized to fit).
        output_path: Where to save.

    """
    n = len(image_paths)
    if n == 0:
        # Empty grid: create a small placeholder
        img = Image.new("RGB", (cell_size, cell_size), (64, 64, 64))
        draw = ImageDraw.Draw(img)
        draw.text((10, cell_size // 2), "No images", fill=(200, 200, 200))
        return _saved(img, output_path)

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
            draw.rectangle(
                [x_off, y_off, x_off + cell_size, y_off + cell_size],
                fill=(64, 0, 0),
            )
            draw.text(
                (x_off + 4, y_off + cell_size // 2),
                "Error",
                fill=(255, 128, 128),
            )

        if titles and i < len(titles):
            _label(
                draw,
                (col * cell_size + 4, row * cell_size + 2),
                titles[i],
                font,
                (0, 0, 0),
            )

    return _saved(grid, output_path)


def render_candidates(
    image: Pixels,
    candidates: list[dict],
    *,
    native_size: FrameSize,
    output_path: str,
    alpha: float = 0.35,
) -> str:
    """Render numbered proposal candidates on display pixels.

    Every ring of each is a semi-transparent polygon in its palette color, with one number and
    score per candidate.

    Args:
        image: Display pixels (uint8 RGB array or PIL image).
        candidates: Candidate dicts, each with candidate_id, rings and score, coordinates in the
            native frame.
        native_size: ``(width, height)`` of the frame the candidates are measured in.
        output_path: Where to save.
        alpha: Fill transparency (0=transparent, 1=opaque).

    Returns:
        Output path to the rendered image.

    """
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
        color = _color_for_class(cid)
        rings = [[(x * sx, y * sy) for x, y in ring] for ring in cand["rings"]]

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
            (cx - tw // 2, cy - th // 2),
            num_text,
            fill=(255, 255, 255),
            font=font_label,
        )

        # Proposal score below the number (neutral across engines)
        info = f"s={cand['score']:.2f}"
        overlay_draw.text(
            (cx - 12, cy + radius + 2), info, fill=color, font=font_small
        )

    return _saved(
        Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"),
        output_path,
    )


def render_grid_overlay(
    image: Pixels,
    cells: list[Cell],
    *,
    native_size: FrameSize,
    output_path: str,
) -> str:
    """Render display pixels with the caller's labeled reference-grid cells overlaid.

    ``cells`` is the caller's own cell list, each a :class:`~tcip_annotation.grid.Cell`: its
    ``name`` plus the half-open native-pixel rect ``x0, y0, x1, y1``.
    Rects scale by the rendered/native ratio, so the lines land on the true cell boundaries, which
    under a clamped grid are non-uniform at the edges.

    Boundaries always draw. A cell's name draws only when the rendered cell is at least 24 px on
    its short edge and wide enough to hold the label's own backing box.

    Args:
        image: Display pixels (uint8 RGB array or PIL image).
        cells: The grid's cells, names plus native-pixel rects.
        native_size: The raster's own (width, height) the cell rects are measured in.
        output_path: Where to save.

    Returns:
        Output path to the rendered image.

    """
    from tcip_annotation.grid import cell_fields

    if not cells:
        raise ValueError("cells is empty: there is no grid to render")

    img = _rgb_frame(image)
    rw, rh = img.size
    sx, sy = _get_scale(native_size[0], native_size[1], rw, rh)
    draw = ImageDraw.Draw(img)

    grid_color = (255, 255, 0)
    label_color = (255, 255, 0)
    label_min_edge = 24

    font = _try_font(14)

    scaled = [
        (name, x0 * sx, y0 * sy, x1 * sx, y1 * sy)
        for name, x0, y0, x1, y1 in map(cell_fields, cells)
    ]

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
        left, _top, right, _bottom = font.getbbox(name)
        if right - left + 6 > x1 - x0:
            continue
        _label(
            draw,
            (int(x0) + 2, int(y0) + 1),
            name,
            font,
            (0, 0, 0),
            fill=label_color,
        )

    return _saved(img, output_path)


def _hex_rgb(color: HexColor) -> Color:
    """Return a :data:`HexColor` as an RGB tuple."""
    from PIL import ImageColor

    return ImageColor.getrgb(color)[:3]  # type: ignore[return-value]


def _dashed_segment(
    draw: ImageDraw.ImageDraw,
    p1: Coord,
    p2: Coord,
    fill: ColorAlpha,
    width: int,
    dash: float,
    gap: float,
) -> None:
    """Draw p1→p2 as dashes."""
    x1, y1 = p1
    x2, y2 = p2
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if length < 1e-6:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        draw.line(
            [(x1 + ux * pos, y1 + uy * pos), (x1 + ux * end, y1 + uy * end)],
            fill=fill,
            width=width,
        )
        pos = end + gap


HexColor = Annotated[
    str, StringConstraints(pattern=r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
]
"""A color as ``#RGB`` or ``#RRGGBB``."""

_FINITE = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class CanvasHalo(BaseModel):
    """The wider translucent stroke the GUI draws under its focused shape."""

    model_config = _FINITE

    color: HexColor
    opacity: Annotated[float, Field(ge=0, le=1)]
    width_factor: PositiveFloat


class _CanvasStroke(BaseModel):
    """What every pushed canvas shape carries beside its geometry.

    ``color`` is its stroke, ``tag`` what it stands for, ``dash`` its ``(on, off)`` pattern in
    stroke widths (``None`` for a solid stroke), ``halo`` its focus halo and ``label`` its text,
    each ``None`` when the GUI draws none.
    """

    model_config = _FINITE

    color: HexColor
    tag: str
    dash: (
        tuple[
            Annotated[float, Strict(), Field(gt=0)],
            Annotated[float, Strict(), Field(ge=0)],
        ]
        | None
    ) = None
    halo: CanvasHalo | None = None
    label: str | None = None


class CanvasBox(_CanvasStroke):
    """A box, its pixel ``xyxy``, filled when ``fill``."""

    kind: Literal["box"]
    xyxy: tuple[float, float, float, float]
    fill: bool = False


class CanvasPoint(_CanvasStroke):
    """A point mark at the one pixel coordinate ``points`` holds."""

    kind: Literal["point"]
    points: tuple[tuple[float, float]]


class CanvasPolygon(_CanvasStroke):
    """One closed ring of two or more pixel coordinates, filled when ``fill``."""

    kind: Literal["polygon"]
    points: Annotated[list[tuple[float, float]], Field(min_length=2)]
    fill: bool = False


class CanvasPolyline(_CanvasStroke):
    """A drawing in progress, its first laid vertex alone included.

    Each vertex is dotted unless ``vertices`` is false.
    """

    kind: Literal["polyline"]
    points: Annotated[list[tuple[float, float]], Field(min_length=1)]
    vertices: bool = True


CanvasShape = Annotated[
    CanvasBox | CanvasPoint | CanvasPolygon | CanvasPolyline,
    Field(discriminator="kind"),
]
"""One display-resolved shape the GUI's canvas pushes, as it draws it."""


class _CanvasShape(NamedTuple):
    """One pushed canvas shape as placed.

    It holds its record, its points in output pixels, its color, whether its outline closes and
    its dash as pixel lengths (None for a solid stroke).
    """

    spec: CanvasShape
    pts: list[Coord]
    color: Color
    closed: bool
    dash: Dash | None


def _draw_path(
    draw: ImageDraw.ImageDraw,
    pts: list[Coord],
    color: ColorAlpha,
    width: int,
    dash: Dash | None,
    closed: bool,
) -> None:
    """Draw ``pts`` as one path, solid or as ``dash`` (on, off) lengths in pixels."""
    seg_pts = pts + ([pts[0]] if closed and len(pts) > 2 else [])
    if dash is not None:
        for a, b in zip(seg_pts, seg_pts[1:]):
            _dashed_segment(
                draw, a, b, color, width, dash=dash[0], gap=dash[1]
            )
    elif len(seg_pts) > 1:
        draw.line(seg_pts, fill=color, width=width, joint="curve")


def render_canvas_state(
    image: Pixels,
    shapes: Sequence[CanvasShape],
    *,
    region: tuple[int, int, int, int],
    output_path: str,
) -> str:
    """Render the live GUI canvas: display-resolved shapes over the pixels the human is viewing.

    ``shapes`` come from the canvas-state push, each already carrying the exact symbology the GUI
    rendered. A point draws as the GUI's mark (a core with radial ticks); a shape carrying a halo
    draws that wider translucent stroke under its own, and one carrying a dash draws its outline
    dashed to that pattern.

    ``image`` is whatever region of the raster the caller read (the human's viewport, or the whole
    frame), and ``region`` is that region's half-open ``(x0, y0, x1, y1)`` in the raster's own
    full-resolution grid. Shape coordinates arrive in the native grid and are placed by the
    region's origin and the image's size against the region's, one scale per axis. The render is
    written to ``output_path`` as JPEG.
    """
    img = _rgb_frame(image)
    ox, oy = region[0], region[1]
    sx, sy = _get_scale(
        region[2] - ox, region[3] - oy, img.size[0], img.size[1]
    )

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    out_w = img.size[0]
    lw = max(1, round(out_w / 700))
    font = _try_font(max(11, out_w // 90))
    dot_r = max(2.0, out_w / 450)

    def tx(p: Sequence[float]) -> Coord:
        return ((float(p[0]) - ox) * sx, (float(p[1]) - oy) * sy)

    # Two passes over one overlay: all fills first, then all outlines/vertices. ImageDraw
    # replaces pixels (it does not composite), so a later shape's translucent fill would
    # otherwise punch its silhouette out of earlier shapes' opaque outlines.
    parsed: list[_CanvasShape] = []
    labels: list[tuple[Coord, str, Color]] = []
    for s in shapes:
        color = _hex_rgb(s.color)
        if isinstance(s, CanvasBox):
            (x1, y1), (x2, y2) = tx(s.xyxy[:2]), tx(s.xyxy[2:])
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        else:
            pts = [tx(p) for p in s.points]
        dash = None if s.dash is None else (s.dash[0] * lw, s.dash[1] * lw)
        closed = isinstance(s, (CanvasBox, CanvasPolygon))
        parsed.append(_CanvasShape(s, pts, color, closed, dash))
        if s.label:
            labels.append((pts[0], s.label, color))

    for shape in parsed:  # pass 1: fills
        if (
            isinstance(shape.spec, (CanvasBox, CanvasPolygon))
            and shape.spec.fill
            and len(shape.pts) >= 3
        ):
            draw.polygon(shape.pts, fill=shape.color + (38,))
    for shape in parsed:  # pass 2: halos, under every outline
        s, pts, closed = shape.spec, shape.pts, shape.closed
        if s.halo is None:
            continue
        halo = _hex_rgb(s.halo.color) + (round(255 * s.halo.opacity),)
        halo_w = max(1, round(lw * s.halo.width_factor))
        if isinstance(s, CanvasPoint):
            px, py = pts[0]
            reach = dot_r * 1.6 * 3.2
            draw.ellipse(
                [px - reach, py - reach, px + reach, py + reach],
                outline=halo,
                width=halo_w,
            )
        else:
            _draw_path(draw, pts, halo, halo_w, None, closed=closed)
    for shape in parsed:  # pass 3: outlines + vertices
        s, pts, color, closed, dash = shape
        if isinstance(s, CanvasPoint):
            # The GUI's reticle: a core plus four radial ticks converging on the coordinate, the
            # mark that distinguishes a location from a very small box on the same canvas.
            px, py = pts[0]
            core = dot_r * 1.6
            inner, outer = core * 1.9, core * 3.2
            draw.ellipse(
                [px - core, py - core, px + core, py + core],
                fill=color + (255,),
            )
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                _draw_path(
                    draw,
                    [
                        (px + dx * inner, py + dy * inner),
                        (px + dx * outer, py + dy * outer),
                    ],
                    color + (255,),
                    lw,
                    dash,
                    closed=False,
                )
            continue
        _draw_path(draw, pts, color + (255,), lw, dash, closed=closed)
        if (
            isinstance(s, CanvasPolyline) and s.vertices
        ):  # a drawing's laid vertices
            for px, py in pts:
                draw.ellipse(
                    [px - dot_r, py - dot_r, px + dot_r, py + dot_r],
                    fill=color + (255,),
                )

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw2 = ImageDraw.Draw(img)
    for (
        (ax, ay),
        text,
        color,
    ) in labels:  # halo text, drawn after compositing so it stays crisp
        x, y = ax + 2, max(0.0, ay - (out_w // 90) - 4)
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            draw2.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
        draw2.text((x, y), text, fill=color, font=font)

    return _saved(img, output_path, quality=88)
