"""Render GT boxes on the EXIF-aligned image to prove alignment.

Annotations were created on EXIF-aligned (upright) images, so labels are in the
rotated canvas's coord space. We apply exif_transpose to the image and draw
YOLO boxes at their given normalized coords directly — no coord rotation.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(r"c:/Users/exx/Documents/GitHub/tcip-agent")
VF = ROOT / "data" / "hazelnut" / "catkin_05-50-95-per_date" / "Valley_Farm"
GT_DIR = VF / "annotations" / "catkin" / "2-11-26" / "detect"
IMG_DIR = VF / "images" / "2-11-26"
OUT_DIR = ROOT / ".tcip" / "artifacts" / "review"

VIEW_WIDTH = 2000
GT_COLOR = (0, 255, 0)
GT_WIDTH = 3


def load_gt(path: Path) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        _, cx, cy, w, h = parts
        rows.append((float(cx), float(cy), float(w), float(h)))
    return rows


def render(stem: str) -> Path:
    im = ImageOps.exif_transpose(Image.open(IMG_DIR / f"{stem}.JPG").convert("RGB"))
    W, H = im.size
    rows = load_gt(GT_DIR / f"{stem}.txt")
    draw = ImageDraw.Draw(im)
    for cx, cy, w, h in rows:
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)
        draw.rectangle((x1, y1, x2, y2), outline=GT_COLOR, width=GT_WIDTH)
    draw.rectangle((0, 0, W, 60), fill=(0, 0, 0))
    draw.text((10, 20), f"{stem}  exif-transposed {W}x{H}  GT={len(rows)}  (no coord rotation)", fill=(255, 255, 255))

    scale = VIEW_WIDTH / W
    view = im.resize((VIEW_WIDTH, int(H * scale)), Image.LANCZOS)
    out = OUT_DIR / f"{stem}_aligned.jpg"
    view.save(out, quality=90)
    print(f"saved {out}")
    return out


def main() -> None:
    stems = sys.argv[1:] or ["IMG_0134", "IMG_0208"]
    for s in stems:
        render(s)


if __name__ == "__main__":
    main()
