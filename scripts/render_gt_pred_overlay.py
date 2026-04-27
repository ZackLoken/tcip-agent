"""Render GT + predictions overlay on an image at a viewable resolution.

Green = GT bboxes. Red = predictions at conf >= CONF_THRESHOLD.
Applies EXIF orientation so the image renders upright.
Can crop to the GT-extent bounding box ("foreground region") with margin,
or to a fixed quadrant, or render the full image.

Usage:
    python render_gt_pred_overlay.py <stem> --mode fg [--pad 0.05]
    python render_gt_pred_overlay.py <stem> --mode quad --crop q1|q2|q3|q4|center
    python render_gt_pred_overlay.py <stem> --mode full
    python render_gt_pred_overlay.py <stem> --mode box --box cx cy w h
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(r"c:/Users/exx/Documents/GitHub/tcip-agent")
VF = ROOT / "data" / "hazelnut" / "catkin_05-50-95-per_date" / "Valley_Farm"
IMG_DIR = VF / "images" / "2-11-26"
GT_DIR = VF / "annotations" / "catkin" / "2-11-26" / "detect"
PRED_DIR = VF / "models" / "baseline" / "predictions_unfiltered" / "detect"
OUT_DIR = ROOT / ".tcip" / "artifacts" / "review"

CONF_THRESHOLD = 0.3
VIEW_WIDTH = 2400
GT_COLOR = (0, 255, 0)
PRED_COLOR = (255, 40, 40)
GT_WIDTH = 3
PRED_WIDTH = 2


def load_yolo(path: Path, has_conf: bool) -> list[tuple[float, float, float, float, float]]:
    rows: list[tuple[float, float, float, float, float]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if has_conf:
            _, conf, cx, cy, w, h = parts
            rows.append((float(cx), float(cy), float(w), float(h), float(conf)))
        else:
            _, cx, cy, w, h = parts
            rows.append((float(cx), float(cy), float(w), float(h), 1.0))
    return rows


def crop_region(
    im: Image.Image,
    mode: str,
    region: str | None,
    gt_rows: list[tuple[float, float, float, float, float]],
    pad: float,
    custom_box: tuple[float, float, float, float] | None,
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Return (cropped, (x0n, y0n, x1n, y1n))."""
    W, H = im.size
    if mode == "full":
        return im, (0.0, 0.0, 1.0, 1.0)
    if mode == "quad":
        regions = {
            "q1": (0.0, 0.0, 0.5, 0.5),
            "q2": (0.5, 0.0, 1.0, 0.5),
            "q3": (0.0, 0.5, 0.5, 1.0),
            "q4": (0.5, 0.5, 1.0, 1.0),
            "center": (0.25, 0.25, 0.75, 0.75),
        }
        x0, y0, x1, y1 = regions[region or "center"]
    elif mode == "fg":
        if not gt_rows:
            raise SystemExit("No GT rows; cannot compute foreground extent. Use --mode full or quad.")
        xs = [r[0] for r in gt_rows]
        ys = [r[1] for r in gt_rows]
        x0 = max(0.0, min(xs) - pad)
        y0 = max(0.0, min(ys) - pad)
        x1 = min(1.0, max(xs) + pad)
        y1 = min(1.0, max(ys) + pad)
    elif mode == "box":
        if not custom_box:
            raise SystemExit("--mode box requires --box cx cy w h")
        cx, cy, w, h = custom_box
        x0 = max(0.0, cx - w / 2)
        y0 = max(0.0, cy - h / 2)
        x1 = min(1.0, cx + w / 2)
        y1 = min(1.0, cy + h / 2)
    else:
        raise SystemExit(f"Unknown mode: {mode}")

    px0, py0, px1, py1 = int(W * x0), int(H * y0), int(W * x1), int(H * y1)
    return im.crop((px0, py0, px1, py1)), (x0, y0, x1, y1)


def render(stem: str, mode: str, region: str | None, pad: float, custom_box: tuple[float, float, float, float] | None) -> Path:
    img_path = IMG_DIR / f"{stem}.JPG"
    gt_rows = load_yolo(GT_DIR / f"{stem}.txt", has_conf=False)
    pred_rows = load_yolo(PRED_DIR / f"{stem}.txt", has_conf=True)

    im = ImageOps.exif_transpose(Image.open(img_path).convert("RGB"))
    im, (x0n, y0n, x1n, y1n) = crop_region(im, mode, region, gt_rows, pad, custom_box)

    scale = VIEW_WIDTH / im.size[0]
    target = (VIEW_WIDTH, int(im.size[1] * scale))
    im = im.resize(target, Image.LANCZOS)
    W, H = im.size

    draw = ImageDraw.Draw(im)
    region_w = x1n - x0n
    region_h = y1n - y0n

    def to_pixel_box(cx: float, cy: float, w: float, h: float) -> tuple[int, int, int, int] | None:
        if not (x0n <= cx <= x1n and y0n <= cy <= y1n):
            return None
        rcx = (cx - x0n) / region_w
        rcy = (cy - y0n) / region_h
        rw = w / region_w
        rh = h / region_h
        x1 = int((rcx - rw / 2) * W)
        y1 = int((rcy - rh / 2) * H)
        x2 = int((rcx + rw / 2) * W)
        y2 = int((rcy + rh / 2) * H)
        return x1, y1, x2, y2

    gt_drawn = 0
    for cx, cy, w, h, _ in gt_rows:
        box = to_pixel_box(cx, cy, w, h)
        if box:
            draw.rectangle(box, outline=GT_COLOR, width=GT_WIDTH)
            gt_drawn += 1

    pred_drawn = 0
    for cx, cy, w, h, conf in pred_rows:
        if conf < CONF_THRESHOLD:
            continue
        box = to_pixel_box(cx, cy, w, h)
        if box:
            draw.rectangle(box, outline=PRED_COLOR, width=PRED_WIDTH)
            pred_drawn += 1

    caption = (
        f"{stem} mode={mode} region={region or 'n/a'}  "
        f"GT(green)={gt_drawn}/{len(gt_rows)}  "
        f"PRED(red,conf>={CONF_THRESHOLD})={pred_drawn}/{sum(1 for r in pred_rows if r[4] >= CONF_THRESHOLD)}"
    )
    draw.rectangle((0, 0, W, 40), fill=(0, 0, 0))
    draw.text((10, 10), caption, fill=(255, 255, 255))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = mode if mode != "quad" else f"quad_{region}"
    if mode == "box":
        tag = f"box_{custom_box[0]:.2f}_{custom_box[1]:.2f}_{custom_box[2]:.2f}_{custom_box[3]:.2f}"
    out_path = OUT_DIR / f"{stem}_{tag}.jpg"
    im.save(out_path, quality=90)
    print(f"saved {out_path}")
    print(caption)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stem")
    p.add_argument("--mode", choices=["full", "quad", "fg", "box"], default="fg")
    p.add_argument("--crop", default=None, help="q1|q2|q3|q4|center for mode=quad")
    p.add_argument("--pad", type=float, default=0.03, help="padding around GT extent for mode=fg")
    p.add_argument("--box", nargs=4, type=float, default=None, metavar=("CX", "CY", "W", "H"))
    args = p.parse_args()
    render(args.stem, args.mode, args.crop, args.pad, tuple(args.box) if args.box else None)


if __name__ == "__main__":
    main()
