"""Render SAM segment annotations on IMG_0133 for visual QA."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "tcip-annotation" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "tcip-mcp" / "src"))

from tcip_annotation.label_io import parse_segment_labels
from tcip_annotation.viz import render_segmentations

IMAGE_NAME = "IMG_0133"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGE_PATH = DATA_DIR / "images" / f"{IMAGE_NAME}.JPG"
SEGMENT_LABEL = DATA_DIR / "labels" / "segment" / f"{IMAGE_NAME}.txt"

import cv2
img = cv2.imread(str(IMAGE_PATH))
h, w = img.shape[:2]

polys, class_ids = parse_segment_labels(str(SEGMENT_LABEL), w, h)
print(f"Loaded {len(polys)} segment polygons")

poly_dicts = [
    {"points": [(pt[0], pt[1]) for pt in p.points], "class_id": p.class_id}
    for p in polys
]

out = render_segmentations(str(IMAGE_PATH), poly_dicts)
print(f"Rendered to: {out}")
