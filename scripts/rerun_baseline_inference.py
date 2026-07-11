"""Re-run baseline YOLO inference on the labeled catkin images via SAHI (tiled).

Full-image inference destroys recall on small catkins, so we tile. The slice size is the
model's own training ``imgsz`` (read from the checkpoint, not hardcoded) — slicing at that
scale keeps catkins the size the network learned. Tiles overlap so boundary catkins aren't
cut. ``conf`` defaults near zero (unfiltered) to preserve low-confidence "hesitant"
detections for false-negative mining.

SAHI reads images EXIF-upright (``exif_fix=True``) and we normalize by the upright
dimensions, so predictions land in the same frame the GT labels are authored in.

Output: YOLO detect ``class conf cx cy w h`` (normalized, full-image, upright frame).

    python rerun_baseline_inference.py [--tile N] [--overlap R] [--conf C] [--device D]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from tcip_annotation.utils import auto_orient_image, get_image_dimensions

from _paths import CATKIN_DATE, vf_root

VF = vf_root()
WEIGHTS = VF / "models" / "baseline" / "weights.pt"
IMAGES_DIR = VF / "images" / CATKIN_DATE
LABEL_STEMS = sorted(p.stem for p in (VF / "annotations" / "catkin" / CATKIN_DATE / "detect").glob("*.txt"))
OUT_DIR = VF / "models" / "baseline" / "predictions_unfiltered" / "detect"


def training_imgsz(weights, default: int = 640) -> int:
    """The size the model was trained at — read from the checkpoint so the tile tracks the
    actual model instead of a guessed constant. Falls back to ``default`` if unavailable."""
    try:
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        imgsz = (ckpt.get("train_args") or {}).get("imgsz", default) if isinstance(ckpt, dict) else default
        return int(imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz)
    except Exception:
        return default


def main() -> None:
    ap = argparse.ArgumentParser(description="Tiled SAHI inference for the baseline detector.")
    ap.add_argument("--tile", type=int, default=None, help="slice size in px (default: model's training imgsz)")
    ap.add_argument("--overlap", type=float, default=0.2, help="tile overlap ratio, both axes (default: 0.2)")
    ap.add_argument("--conf", type=float, default=0.001, help="confidence threshold; ~0 keeps hesitant dets for FN mining")
    ap.add_argument("--device", default=None, help="torch device (default: cuda:0 if available, else cpu)")
    args = ap.parse_args()

    tile = args.tile or training_imgsz(WEIGHTS)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"tile={tile}px overlap={args.overlap} conf={args.conf} device={device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = [IMAGES_DIR / f"{stem}.JPG" for stem in LABEL_STEMS]
    missing = [p for p in image_paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing images: {missing}")

    det_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(WEIGHTS),
        confidence_threshold=args.conf,
        device=device,
    )

    total = 0
    for path in image_paths:
        # SAHI reads EXIF-upright; we hand it the already-oriented array and normalize by the
        # upright dims (get_image_dimensions), so predictions share the GT's upright frame.
        W, H = get_image_dimensions(str(path))
        image = np.asarray(auto_orient_image(Image.open(path)).convert("RGB"))
        result = get_sliced_prediction(
            image=image,
            detection_model=det_model,
            slice_height=tile,
            slice_width=tile,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            verbose=0,
            postprocess_type="NMS",
            postprocess_match_threshold=0.5,
        )

        lines: list[str] = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox
            x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            cls = int(pred.category.id)
            conf = float(pred.score.value)
            lines.append(f"{cls} {conf:.6f} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        (OUT_DIR / f"{path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        total += len(lines)
        print(f"  {path.name}: {len(lines)} predictions")

    print(f"\nWrote {total} predictions across {len(image_paths)} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
