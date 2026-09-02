"""Compute foreground-only high-confidence FN candidates per image.

A candidate FN is a prediction that:
  - has conf >= CONF_MIN
  - lies inside the foreground region (convex hull of GT centers, dilated)
  - has no GT center within MATCH_RADIUS (normalized)

Prints per-image counts and totals. Also writes a JSON sidecar with candidate
coordinates per image so the review workflow can consume them directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from tcip_annotation import Polygon, point_in_polygon

from _paths import CATKIN_DATE, repo_root, vf_root

VF = vf_root()
GT_DIR = VF / "annotations" / "catkin" / CATKIN_DATE / "detect"
PRED_DIR = VF / "models" / "baseline" / "predictions_unfiltered" / "detect"
OUT_PATH = repo_root() / ".tcip" / "artifacts" / "review" / "fn_candidates.json"

CONF_MIN = 0.5
HULL_DILATE = 0.02
MATCH_RADIUS_SCALE = 1.5  # times median GT diagonal


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


def foreground_mask(centers: np.ndarray, gt_centers: np.ndarray, dilate: float) -> np.ndarray:
    """Points inside the (dilated) convex hull of GT centers."""
    if len(gt_centers) < 3 or len(centers) == 0:
        return np.zeros(len(centers), dtype=bool)
    try:
        hull = ConvexHull(gt_centers)
    except Exception:
        return np.zeros(len(centers), dtype=bool)
    hull_pts = gt_centers[hull.vertices]
    centroid = hull_pts.mean(axis=0)
    dilated = hull_pts + (hull_pts - centroid) * (dilate / np.linalg.norm(hull_pts - centroid, axis=1, keepdims=True).clip(min=1e-6))
    ring = [(float(x), float(y)) for x, y in dilated]
    polygon = Polygon(rings=[ring])
    return np.array([point_in_polygon(float(cx), float(cy), polygon) for cx, cy in centers], dtype=bool)


def main() -> None:
    results = {}
    total_candidates = 0
    print(f"{'image':<14} {'n_gt':>6} {'pred_highconf':>14} {'in_foreground':>14} {'fn_candidates':>14}")
    print("-" * 70)

    for gt_path in sorted(GT_DIR.glob("*.txt")):
        stem = gt_path.stem
        gt = load_gt(gt_path)
        preds, confs = load_pred(PRED_DIR / f"{stem}.txt")

        if len(gt) == 0:
            # Confirmed negative image; no foreground region -> no FN candidates valid here
            print(f"{stem:<14} {len(gt):>6} {'-':>14} {'-':>14} {'n/a (neg)':>14}")
            results[stem] = {"n_gt": 0, "candidates": []}
            continue

        highconf = confs >= CONF_MIN
        pred_centers = preds[highconf][:, :2]
        pred_boxes = preds[highconf]
        pred_confs = confs[highconf]
        n_highconf = len(pred_centers)

        if n_highconf == 0:
            print(f"{stem:<14} {len(gt):>6} {n_highconf:>14} {0:>14} {0:>14}")
            results[stem] = {"n_gt": int(len(gt)), "candidates": []}
            continue

        in_fg = foreground_mask(pred_centers, gt[:, :2], HULL_DILATE)
        n_in_fg = int(in_fg.sum())

        gt_diag = np.sqrt(gt[:, 2] ** 2 + gt[:, 3] ** 2)
        radius = float(np.median(gt_diag) * MATCH_RADIUS_SCALE)
        gt_centers = gt[:, :2]
        dists = np.linalg.norm(pred_centers[:, None, :] - gt_centers[None, :, :], axis=2)
        min_dist = dists.min(axis=1)
        unmatched = min_dist > radius
        is_candidate = in_fg & unmatched
        n_cand = int(is_candidate.sum())

        cands = []
        for idx in np.where(is_candidate)[0]:
            cx, cy, w, h = pred_boxes[idx]
            cands.append({
                "cx": float(cx),
                "cy": float(cy),
                "w": float(w),
                "h": float(h),
                "conf": float(pred_confs[idx]),
            })
        results[stem] = {"n_gt": int(len(gt)), "candidates": cands, "match_radius_norm": radius}
        total_candidates += n_cand

        print(f"{stem:<14} {len(gt):>6} {n_highconf:>14} {n_in_fg:>14} {n_cand:>14}")

    print("-" * 70)
    print(f"{'TOTAL':<14} {'':>6} {'':>14} {'':>14} {total_candidates:>14}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote candidate list to {OUT_PATH}")


if __name__ == "__main__":
    main()
