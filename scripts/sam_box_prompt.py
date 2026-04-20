"""Run SAM segmentation using existing detection boxes as prompts.

Downscales the image to fit GPU memory, runs SAM box-prompted prediction
for each detection box, then saves polygons in YOLO segment format.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

# --- Config ---
IMAGE_NAME = "IMG_0133"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGE_PATH = DATA_DIR / "images" / f"{IMAGE_NAME}.JPG"
DETECT_LABEL = DATA_DIR / "labels" / "detect" / f"{IMAGE_NAME}.txt"
OUTPUT_PATH = DATA_DIR / "labels" / "segment" / f"{IMAGE_NAME}.txt"
CHECKPOINT = Path.home() / ".cache" / "tcip" / "sam" / "sam_vit_b_01ec64.pth"
MAX_LONG_SIDE = 1024  # downscale target


def load_yolo_boxes(label_path: Path, img_w: int, img_h: int):
    """Load YOLO detection labels and convert to pixel coords."""
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def mask_to_polygon(mask: np.ndarray) -> list[tuple[float, float]]:
    """Convert binary mask to simplified polygon (largest contour)."""
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.5 * cv2.arcLength(largest, True) / max(len(largest), 1)
    epsilon = max(epsilon, 1.0)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return [(float(pt[0][0]), float(pt[0][1])) for pt in approx]


def main():
    if not IMAGE_PATH.is_file():
        print(f"Image not found: {IMAGE_PATH}")
        sys.exit(1)
    if not DETECT_LABEL.is_file():
        print(f"Detection labels not found: {DETECT_LABEL}")
        sys.exit(1)
    if not CHECKPOINT.is_file():
        print(f"SAM checkpoint not found: {CHECKPOINT}")
        sys.exit(1)

    # Load image
    img = cv2.imread(str(IMAGE_PATH))
    orig_h, orig_w = img.shape[:2]
    print(f"Original image: {orig_w}x{orig_h}")

    # Compute downscale factor
    long_side = max(orig_w, orig_h)
    scale = MAX_LONG_SIDE / long_side
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    print(f"Downscaled to: {new_w}x{new_h} (scale={scale:.4f})")

    img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    # Load detection boxes (in original pixel coords)
    boxes = load_yolo_boxes(DETECT_LABEL, orig_w, orig_h)
    print(f"Loaded {len(boxes)} detection boxes")

    # Load SAM
    torch.cuda.empty_cache()
    print("Loading SAM vit_b...")
    sam = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT))
    sam = sam.to("cuda")
    predictor = SamPredictor(sam)

    # Set downscaled image
    predictor.set_image(img_rgb)
    print("Image embedding computed")

    # Run box-prompted prediction for each detection
    output_lines = []
    success = 0
    for i, (cls_id, x1, y1, x2, y2) in enumerate(boxes):
        # Scale box to downscaled image
        box_scaled = np.array([
            x1 * scale, y1 * scale,
            x2 * scale, y2 * scale,
        ])

        masks, scores, _ = predictor.predict(
            box=box_scaled,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        polygon = mask_to_polygon(masks[best_idx])

        if len(polygon) < 3:
            # Fallback: use the detection box as a rectangle polygon
            polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            # Normalize directly
            parts = [str(cls_id)]
            for px, py in polygon:
                parts.append(f"{px / orig_w:.6f}")
                parts.append(f"{py / orig_h:.6f}")
            output_lines.append(" ".join(parts))
            continue

        # Scale polygon back to original resolution and normalize
        parts = [str(cls_id)]
        for px, py in polygon:
            orig_px = px / scale
            orig_py = py / scale
            parts.append(f"{orig_px / orig_w:.6f}")
            parts.append(f"{orig_py / orig_h:.6f}")
        output_lines.append(" ".join(parts))
        success += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(boxes)} boxes...")

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"\nDone! {success}/{len(boxes)} boxes got SAM polygons")
    print(f"Saved to: {OUTPUT_PATH}")

    # Cleanup
    del predictor, sam
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
