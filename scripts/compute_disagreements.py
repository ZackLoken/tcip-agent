"""Summarize GT-vs-prediction disagreements per image at several conf thresholds.

A prediction is a candidate FN if its center lies outside all GT bbox expansions
(no GT within 1.0 * max(w,h) radius of any prediction center). Using center-based
matching rather than IoU because catkin bboxes are tiny (~40 px) and IoU is noisy
at that scale.

A GT is a candidate FP if its center is not covered by any low-conf prediction
expansion — i.e., the model didn't even consider this region a catkin.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(r"c:/Users/exx/Documents/GitHub/tcip-agent/data/hazelnut/catkin_05-50-95-per_date/Valley_Farm")
GT_DIR = ROOT / "annotations" / "catkin" / "2-11-26" / "detect"
PRED_DIR = ROOT / "models" / "baseline" / "predictions_unfiltered" / "detect"

CONF_THRESHOLDS = [0.3, 0.5, 0.7]
FP_EXPAND = 1.5  # GT counts as covered if any pred center is within this * bbox_half_size


def load_gt(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    if not path.exists():
        return np.zeros((0, 4), dtype=float)
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        _, cx, cy, w, h = parts
        rows.append([float(cx), float(cy), float(w), float(h)])
    return np.asarray(rows, dtype=float) if rows else np.zeros((0, 4), dtype=float)


def load_pred(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    confs: list[float] = []
    if not path.exists():
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=float)
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        _, conf, cx, cy, w, h = parts
        confs.append(float(conf))
        rows.append([float(cx), float(cy), float(w), float(h)])
    if not rows:
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=float)
    return np.asarray(rows, dtype=float), np.asarray(confs, dtype=float)


def center_covered(points: np.ndarray, boxes: np.ndarray, expand: float = 1.0) -> np.ndarray:
    """For each point (N,2), check if it lies inside any expanded box (M,4 cx,cy,w,h)."""
    if len(boxes) == 0 or len(points) == 0:
        return np.zeros(len(points), dtype=bool)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - (w * expand) / 2
    x2 = cx + (w * expand) / 2
    y1 = cy - (h * expand) / 2
    y2 = cy + (h * expand) / 2
    px = points[:, 0][:, None]
    py = points[:, 1][:, None]
    covered = (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)
    return covered.any(axis=1)


def main() -> None:
    print(f"{'image':<14} {'n_gt':>6} {'n_pred':>7}   " + "   ".join(f"FN@{c:.1f}" for c in CONF_THRESHOLDS) + "   FP(no_lowconf)")
    print("-" * 80)

    totals = {"gt": 0, "pred_all": 0}
    fn_totals = {c: 0 for c in CONF_THRESHOLDS}
    fp_total = 0

    for gt_path in sorted(GT_DIR.glob("*.txt")):
        stem = gt_path.stem
        gt = load_gt(gt_path)
        preds, confs = load_pred(PRED_DIR / f"{stem}.txt")

        row = f"{stem:<14} {len(gt):>6} {len(preds):>7}   "

        for thr in CONF_THRESHOLDS:
            sel = confs >= thr
            p_sel = preds[sel]
            if len(p_sel) == 0:
                fn = 0
            else:
                pred_centers = p_sel[:, :2]
                covered = center_covered(pred_centers, gt, expand=FP_EXPAND)
                fn = int((~covered).sum())
            fn_totals[thr] += fn
            row += f"  {fn:>5}"

        if len(gt) > 0:
            gt_centers = gt[:, :2]
            covered = center_covered(gt_centers, preds, expand=FP_EXPAND)
            fp = int((~covered).sum())
        else:
            fp = 0
        fp_total += fp
        row += f"   {fp:>5}"

        totals["gt"] += len(gt)
        totals["pred_all"] += len(preds)
        print(row)

    print("-" * 80)
    tot_row = f"{'TOTAL':<14} {totals['gt']:>6} {totals['pred_all']:>7}   "
    for thr in CONF_THRESHOLDS:
        tot_row += f"  {fn_totals[thr]:>5}"
    tot_row += f"   {fp_total:>5}"
    print(tot_row)


if __name__ == "__main__":
    main()
