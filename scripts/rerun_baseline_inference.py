"""Re-run baseline YOLOv8x inference on the 18 Feb-11 labeled images via SAHI.

The baseline was trained on tiles, so full-image inference at imgsz=640 destroys
recall on small catkins. PD's original predictions came from tiled/SAHI inference.
We match that here, at conf=0.001 (effectively unfiltered) so low-confidence
"hesitant" detections are preserved for false-negative mining.

Output format: YOLO detect `class conf cx cy w h` (normalized, full-image coords).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

ROOT = Path(r"c:/Users/exx/Documents/GitHub/tcip-agent/data/hazelnut/catkin_05-50-95-per_date/Valley_Farm")
WEIGHTS = ROOT / "models" / "baseline" / "weights.pt"
IMAGES_DIR = ROOT / "images" / "2-11-26"
LABEL_STEMS = sorted(p.stem for p in (ROOT / "annotations" / "catkin" / "2-11-26" / "detect").glob("*.txt"))
OUT_DIR = ROOT / "models" / "baseline" / "predictions_unfiltered" / "detect"

TILE_H = 640
TILE_W = 640
OVERLAP_H = 0.2
OVERLAP_W = 0.2
CONF = 0.001


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = [IMAGES_DIR / f"{stem}.JPG" for stem in LABEL_STEMS]
    missing = [p for p in image_paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing images: {missing}")

    det_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(WEIGHTS),
        confidence_threshold=CONF,
        device="cuda:0",
    )

    total = 0
    for path in image_paths:
        with Image.open(path) as im:
            W, H = im.size
        result = get_sliced_prediction(
            image=str(path),
            detection_model=det_model,
            slice_height=TILE_H,
            slice_width=TILE_W,
            overlap_height_ratio=OVERLAP_H,
            overlap_width_ratio=OVERLAP_W,
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
