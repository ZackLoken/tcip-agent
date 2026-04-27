"""Empirically determine which way exif_transpose rotates an orientation=6 image.

Plant a distinctive red dot at raw pixel (1500, 2000) in IMG_0208. Apply
exif_transpose. Find the red dot in the rotated image. That tells us the actual
coordinate transform, independent of any theory about PIL conventions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

IMG = Path(r"c:/Users/exx/Documents/GitHub/tcip-agent/data/hazelnut/catkin_05-50-95-per_date/Valley_Farm/images/2-11-26/IMG_0208.JPG")

DOT_X_RAW = 1500
DOT_Y_RAW = 2000
DOT_RADIUS = 40
DOT_COLOR = (255, 0, 0)


def main() -> None:
    raw = Image.open(IMG).convert("RGB")
    W_raw, H_raw = raw.size
    print(f"raw size: {W_raw} x {H_raw}")

    marked = raw.copy()
    ImageDraw.Draw(marked).ellipse(
        (DOT_X_RAW - DOT_RADIUS, DOT_Y_RAW - DOT_RADIUS, DOT_X_RAW + DOT_RADIUS, DOT_Y_RAW + DOT_RADIUS),
        fill=DOT_COLOR,
    )

    rotated = ImageOps.exif_transpose(marked)
    W_rot, H_rot = rotated.size
    print(f"rotated size: {W_rot} x {H_rot}")

    arr = np.array(rotated)
    red_mask = (arr[..., 0] > 200) & (arr[..., 1] < 60) & (arr[..., 2] < 60)
    ys, xs = np.where(red_mask)
    if len(xs) == 0:
        print("ERROR: no red pixels found in rotated image")
        return
    cx_pix = int(xs.mean())
    cy_pix = int(ys.mean())
    print(f"red dot center in rotated image: pixel ({cx_pix}, {cy_pix})")
    print(f"normalized: ({cx_pix / W_rot:.4f}, {cy_pix / H_rot:.4f})")

    print("\nCandidate transforms from raw (1500, 2000) in 5712x4284 canvas:")
    print(f"  raw_normalized:       ({DOT_X_RAW/W_raw:.4f}, {DOT_Y_RAW/H_raw:.4f})")
    print(f"  90 CW (1-cy, cx):     ({1 - DOT_Y_RAW/H_raw:.4f}, {DOT_X_RAW/W_raw:.4f})")
    print(f"  90 CCW (cy, 1-cx):    ({DOT_Y_RAW/H_raw:.4f}, {1 - DOT_X_RAW/W_raw:.4f})")
    print(f"  180   (1-cx, 1-cy):   ({1 - DOT_X_RAW/W_raw:.4f}, {1 - DOT_Y_RAW/H_raw:.4f})")


if __name__ == "__main__":
    main()
