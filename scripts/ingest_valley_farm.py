"""Idempotent ingest for Valley_Farm hazelnut catkin dataset.

Does three things:
  1. Strips ' iPhone' suffix from date-named image folders.
  2. Derives YOLO detect bboxes from bush segmentation polygons
     (bbox = axis-aligned min/max of polygon vertices).
  3. Remaps bush class 2 -> 0 in both segment and derived detect files.

Safe to re-run. Prints a summary at the end.
"""
from __future__ import annotations

from _paths import BUSH_DATE, CATKIN_DATE, vf_root

VF = vf_root()
IMAGES = VF / "images"
BUSH_ANN = VF / "annotations" / "bush" / BUSH_DATE
CATKIN_ANN = VF / "annotations" / "catkin" / CATKIN_DATE


def flatten_iphone_suffix() -> list[str]:
    actions: list[str] = []
    for child in sorted(IMAGES.iterdir()):
        if child.is_dir() and child.name.endswith(" iPhone"):
            new_name = child.name.replace(" iPhone", "")
            target = child.parent / new_name
            if target.exists():
                actions.append(f"SKIP (target exists): {child.name} -> {new_name}")
                continue
            child.rename(target)
            actions.append(f"RENAMED: {child.name} -> {new_name}")
        elif child.is_dir():
            actions.append(f"OK: {child.name}")
    return actions


def polygon_to_bbox(values: list[float]) -> tuple[float, float, float, float]:
    """Convert YOLO polygon (x1 y1 x2 y2 ...) to YOLO bbox (cx cy w h)."""
    xs = values[0::2]
    ys = values[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return cx, cy, w, h


def derive_bush_detect_and_remap() -> dict:
    """Create bush detect/ dir from segment/, and remap class 2->0 in both."""
    seg_dir = BUSH_ANN / "segment"
    det_dir = BUSH_ANN / "detect"
    det_dir.mkdir(parents=True, exist_ok=True)

    stats = {"segment_remapped": 0, "detect_written": 0, "segment_total": 0}

    for seg_path in sorted(seg_dir.glob("*.txt")):
        stats["segment_total"] += 1
        lines_seg_out: list[str] = []
        lines_det_out: list[str] = []
        changed_seg = False
        for line in seg_path.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            new_cls = 0 if cls == 2 else cls
            if new_cls != cls:
                changed_seg = True
            lines_seg_out.append(" ".join([str(new_cls), *parts[1:]]))
            cx, cy, w, h = polygon_to_bbox(coords)
            lines_det_out.append(f"{new_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if changed_seg:
            seg_path.write_text("\n".join(lines_seg_out) + "\n")
            stats["segment_remapped"] += 1

        det_path = det_dir / seg_path.name
        det_path.write_text("\n".join(lines_det_out) + "\n")
        stats["detect_written"] += 1

    return stats


def summarize_counts() -> dict:
    out: dict = {"images_per_date": {}, "catkin_ann": {}, "bush_ann": {}}
    for d in sorted(IMAGES.iterdir()):
        if d.is_dir():
            n = sum(1 for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".heic"})
            out["images_per_date"][d.name] = n
    if (CATKIN_ANN / "detect").exists():
        out["catkin_ann"]["detect"] = sum(1 for _ in (CATKIN_ANN / "detect").glob("*.txt"))
    if (CATKIN_ANN / "segment").exists():
        out["catkin_ann"]["segment"] = sum(1 for _ in (CATKIN_ANN / "segment").glob("*.txt"))
    if (BUSH_ANN / "detect").exists():
        out["bush_ann"]["detect"] = sum(1 for _ in (BUSH_ANN / "detect").glob("*.txt"))
    if (BUSH_ANN / "segment").exists():
        out["bush_ann"]["segment"] = sum(1 for _ in (BUSH_ANN / "segment").glob("*.txt"))
    return out


def main() -> None:
    print("=== flatten ' iPhone' suffix ===")
    for a in flatten_iphone_suffix():
        print(f"  {a}")

    print("\n=== derive bush detect/ + remap class 2 -> 0 ===")
    stats = derive_bush_detect_and_remap()
    print(f"  segment files total:     {stats['segment_total']}")
    print(f"  segment files remapped:  {stats['segment_remapped']}")
    print(f"  detect files written:    {stats['detect_written']}")

    print("\n=== post-ingest counts ===")
    summary = summarize_counts()
    print(f"  images per date:      {summary['images_per_date']}")
    print(f"  catkin annotations:   {summary['catkin_ann']}")
    print(f"  bush annotations:     {summary['bush_ann']}")


if __name__ == "__main__":
    main()
