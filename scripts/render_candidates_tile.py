"""Render a tile showing GT (green) and only the FN candidates (numbered red).

Reads fn_candidates.json from scripts/foreground_fn_candidates.py. Candidates
outside the requested tile are not drawn. Numbers the candidates so each can be
individually classified (catkin/not_catkin/ambiguous) during review.

Usage:
    python render_candidates_tile.py <stem> --box cx cy w h
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, ImageFont

from _paths import CATKIN_DATE, repo_root, vf_root

VF = vf_root()
IMG_DIR = VF / "images" / CATKIN_DATE
GT_DIR = VF / "annotations" / "catkin" / CATKIN_DATE / "detect"
_REVIEW = repo_root() / ".tcip" / "artifacts" / "review"
CANDS_PATH = _REVIEW / "fn_candidates.json"
OUT_DIR = _REVIEW

VIEW_WIDTH = 2400
GT_COLOR = (0, 220, 0)
CAND_COLOR = (255, 40, 40)
NUM_COLOR = (255, 255, 0)
GT_WIDTH = 2
CAND_WIDTH = 3


def load_gt(path: Path) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        _, cx, cy, w, h = parts
        rows.append((float(cx), float(cy), float(w), float(h)))
    return rows


def render(stem: str, box: tuple[float, float, float, float]) -> Path:
    cands_all = json.loads(CANDS_PATH.read_text())
    cands = cands_all.get(stem, {}).get("candidates", [])
    gt_rows = load_gt(GT_DIR / f"{stem}.txt")

    im = ImageOps.exif_transpose(Image.open(IMG_DIR / f"{stem}.JPG").convert("RGB"))
    W_orig, H_orig = im.size

    cx, cy, w, h = box
    x0n = max(0.0, cx - w / 2)
    y0n = max(0.0, cy - h / 2)
    x1n = min(1.0, cx + w / 2)
    y1n = min(1.0, cy + h / 2)
    px0, py0, px1, py1 = int(W_orig * x0n), int(H_orig * y0n), int(W_orig * x1n), int(H_orig * y1n)
    crop = im.crop((px0, py0, px1, py1))

    scale = VIEW_WIDTH / crop.size[0]
    crop = crop.resize((VIEW_WIDTH, int(crop.size[1] * scale)), Image.Resampling.LANCZOS)
    W, H = crop.size
    region_w = x1n - x0n
    region_h = y1n - y0n

    draw = ImageDraw.Draw(crop)
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    def to_pix(cx_n, cy_n, w_n, h_n):
        if not (x0n <= cx_n <= x1n and y0n <= cy_n <= y1n):
            return None
        rcx = (cx_n - x0n) / region_w
        rcy = (cy_n - y0n) / region_h
        rw = w_n / region_w
        rh = h_n / region_h
        return (
            int((rcx - rw / 2) * W),
            int((rcy - rh / 2) * H),
            int((rcx + rw / 2) * W),
            int((rcy + rh / 2) * H),
        )

    gt_in = 0
    for gcx, gcy, gw, gh in gt_rows:
        b = to_pix(gcx, gcy, gw, gh)
        if b:
            draw.rectangle(b, outline=GT_COLOR, width=GT_WIDTH)
            gt_in += 1

    cand_in = 0
    for idx, c in enumerate(cands):
        b = to_pix(c["cx"], c["cy"], c["w"], c["h"])
        if b:
            draw.rectangle(b, outline=CAND_COLOR, width=CAND_WIDTH)
            draw.text((b[0] + 4, b[1] - 28), str(idx), fill=NUM_COLOR, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
            cand_in += 1

    draw.rectangle((0, 0, W, 50), fill=(0, 0, 0))
    draw.text(
        (10, 15),
        f"{stem} box={box}  GT(green)={gt_in}/{len(gt_rows)}  CANDIDATES(red #num)={cand_in}/{len(cands)}",
        fill=(255, 255, 255),
    )

    out = OUT_DIR / f"{stem}_cands_{box[0]:.2f}_{box[1]:.2f}_{box[2]:.2f}_{box[3]:.2f}.jpg"
    crop.save(out, quality=90)
    print(f"saved {out}  GT_in_tile={gt_in}  candidates_in_tile={cand_in}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stem")
    p.add_argument("--box", nargs=4, type=float, required=True, metavar=("CX", "CY", "W", "H"))
    args = p.parse_args()
    render(args.stem, tuple(args.box))


if __name__ == "__main__":
    main()
