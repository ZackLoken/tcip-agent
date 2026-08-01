"""Sliding-window tiling geometry for small-object detection (SAHI-style).

Pure numpy; the only torch touchpoint is an optional ``torchvision.ops`` import in
``global_nms`` with a numpy greedy-NMS fallback, so the geometry is unit-testable
without torch. Ported from the chestnut-burr ``CanopyTiler`` /
``reconstruct_detections_core`` / ``_dedup_boxes`` (verified against that source):

  * ``stride = int(tile_size * (1 - overlap))``  (224, 0.2 -> 179)
  * pad H,W up to the next multiple of ``tile_size``; iterate origins on a regular
    grid; skip origins that lie entirely in the padding region
  * reconstruction core margin ``= (tile_size - stride) / 2``; keep a detection only
    if its box center lies in the tile's non-overlapping core (inset on interior
    sides, flush on image-boundary sides), then a single low-IoU global NMS
"""

from __future__ import annotations

import numpy as np

EMPTY_BOXES = np.zeros((0, 4), dtype=np.float32)
EMPTY_LABELS = np.zeros((0,), dtype=np.int64)
EMPTY_SCORES = np.zeros((0,), dtype=np.float32)


def compute_stride(tile_size: int, overlap: float) -> int:
    """``int(tile_size * (1 - overlap))``, floored to >= 1. (224, 0.2) -> 179."""
    return max(1, int(tile_size * (1.0 - overlap)))


def tile_positions(height: int, width: int, tile_size: int, stride: int) -> list[tuple[int, int]]:
    """Sliding-window tile origins ``(tile_x, tile_y)`` over a (padded) HxW image.

    Pads H,W up to the next multiple of ``tile_size`` and drops origins whose tile
    is entirely in the bottom/right padding region (``tile_x >= width`` or
    ``tile_y >= height``).
    """
    def _padded(dim: int) -> int:
        return ((dim + tile_size - 1) // tile_size) * tile_size

    padded_h, padded_w = _padded(height), _padded(width)
    positions: list[tuple[int, int]] = []
    for ty in range(0, padded_h - tile_size + 1, stride):
        for tx in range(0, padded_w - tile_size + 1, stride):
            if tx >= width or ty >= height:
                continue  # tile is pure padding
            positions.append((tx, ty))
    return positions


def clip_boxes_to_tile(
    boxes: np.ndarray, labels: np.ndarray, tile_x: int, tile_y: int,
    tile_size: int, min_box_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect full-image-px boxes with a tile; drop seam slivers; emit tile-local xyxy.

    A box clipped by the tile edge is dropped only when the *visible* (clipped) part is a sliver:
    its characteristic size ``sqrt(iw*ih) < min_box_size``. ``min_box_size`` is derived per dataset
    from the class's average box size (a partial catkin counts unless it's a tiny sliver; see
    ``TiledDetectionDataset``), not a fixed fraction. Boxes fully inside the tile are always kept.
    """
    if len(boxes) == 0:
        return EMPTY_BOXES.copy(), EMPTY_LABELS.copy()
    tx2, ty2 = tile_x + tile_size, tile_y + tile_size
    out_boxes, out_labels = [], []
    for (bx1, by1, bx2, by2), lab in zip(boxes, labels):
        ix1, iy1 = max(bx1, tile_x), max(by1, tile_y)
        ix2, iy2 = min(bx2, tx2), min(by2, ty2)
        iw, ih = ix2 - ix1, iy2 - iy1
        if iw <= 0 or ih <= 0:
            continue
        clipped = bx1 < tile_x or by1 < tile_y or bx2 > tx2 or by2 > ty2
        if clipped and (iw * ih) ** 0.5 < min_box_size:
            continue
        out_boxes.append([ix1 - tile_x, iy1 - tile_y, ix2 - tile_x, iy2 - tile_y])
        out_labels.append(int(lab))
    if not out_boxes:
        return EMPTY_BOXES.copy(), EMPTY_LABELS.copy()
    return np.asarray(out_boxes, dtype=np.float32), np.asarray(out_labels, dtype=np.int64)


def _iou(a, b, area_a, area_b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _ios(a, b, area_a, area_b) -> float:
    """Intersection over the smaller box area: the NMM match metric. A partial fragment mostly
    inside a fuller detection scores high here even when its IoU is low, which is exactly the
    seam-split case merging is meant to catch (IoU would leave the two boxes separate)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def dedup_boxes(
    boxes: np.ndarray, labels: np.ndarray, iou_thresh: float, class_aware: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy largest-first dedup: drop a box overlapping a kept box by >= iou_thresh."""
    n = len(boxes)
    if not iou_thresh or iou_thresh >= 1.0 or n < 2:
        return boxes, labels
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = sorted(range(n), key=lambda i: -areas[i])  # largest first
    kept: list[int] = []
    for i in order:
        dup = False
        for k in kept:
            if class_aware and labels[i] != labels[k]:
                continue
            if _iou(boxes[i], boxes[k], areas[i], areas[k]) >= iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(i)
    keep_idx = sorted(kept)
    return boxes[keep_idx], labels[keep_idx]


def reconstruct_core(
    per_tile_boxes: list[np.ndarray], per_tile_scores: list[np.ndarray],
    per_tile_labels: list[np.ndarray], tile_info: list[dict], tile_size: int, stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift tile-local detections to full-image coords, keep center-in-core, clip.

    ``tile_info[i]`` = ``{'tile_x','tile_y','original_width','original_height'}``.
    """
    margin = (tile_size - stride) / 2.0
    out_b, out_s, out_l = [], [], []
    for tb, ts, tl, info in zip(per_tile_boxes, per_tile_scores, per_tile_labels, tile_info):
        if len(tb) == 0:
            continue
        tx, ty = info["tile_x"], info["tile_y"]
        img_w, img_h = info["original_width"], info["original_height"]
        core_x0 = tx + (margin if tx > 0 else 0)
        core_y0 = ty + (margin if ty > 0 else 0)
        core_x1 = (tx + tile_size) - (margin if (tx + tile_size) < img_w else 0)
        core_y1 = (ty + tile_size) - (margin if (ty + tile_size) < img_h else 0)
        boxes = np.asarray(tb, dtype=np.float64).reshape(-1, 4).copy()
        boxes[:, [0, 2]] += tx
        boxes[:, [1, 3]] += ty
        for i in range(len(boxes)):
            box = boxes[i]
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            if not (core_x0 <= cx <= core_x1 and core_y0 <= cy <= core_y1):
                continue
            clipped = box.copy()
            clipped[[0, 2]] = np.clip(clipped[[0, 2]], 0, img_w)
            clipped[[1, 3]] = np.clip(clipped[[1, 3]], 0, img_h)
            out_b.append(clipped)
            out_s.append(float(ts[i]))
            out_l.append(int(tl[i]))
    if not out_b:
        return EMPTY_BOXES.copy(), EMPTY_SCORES.copy(), EMPTY_LABELS.copy()
    return (np.asarray(out_b, dtype=np.float32),
            np.asarray(out_s, dtype=np.float32),
            np.asarray(out_l, dtype=np.int64))


def _numpy_nms(boxes, scores, labels, iou_thresh, class_aware) -> np.ndarray:
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])  # highest score first
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    kept: list[int] = []
    for i in order:
        suppress = False
        for k in kept:
            if class_aware and labels[i] != labels[k]:
                continue
            if _iou(boxes[i], boxes[k], areas[i], areas[k]) >= iou_thresh:
                suppress = True
                break
        if not suppress:
            kept.append(i)
    return np.asarray(kept, dtype=np.int64)


def global_nms(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
    iou_thresh: float, class_aware: bool = True,
) -> np.ndarray:
    """Cross-tile NMS. Returns kept indices. Uses torchvision when available."""
    if len(boxes) == 0:
        return EMPTY_LABELS.copy()
    try:
        import torch
        from torchvision.ops import batched_nms, nms

        b = torch.as_tensor(np.asarray(boxes), dtype=torch.float32)
        s = torch.as_tensor(np.asarray(scores), dtype=torch.float32)
        if class_aware:
            idxs = torch.as_tensor(np.asarray(labels), dtype=torch.int64)
            keep = batched_nms(b, s, idxs, iou_thresh)
        else:
            keep = nms(b, s, iou_thresh)
        return keep.cpu().numpy()
    except Exception:  # noqa: BLE001 (torch/torchvision absent -> numpy fallback)
        return _numpy_nms(np.asarray(boxes), np.asarray(scores), np.asarray(labels), iou_thresh, class_aware)


def global_merge(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
    iou_thresh: float, class_aware: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-tile Non-Max *Merging*: union overlapping same-class boxes (bbox hull, max score)
    instead of suppressing the lower-score one, recovering an object split across a tile seam
    into two partial boxes (SAHI's NMM). Returns merged ``(boxes, scores, labels)``, new boxes,
    not a subset of indices like :func:`global_nms`, so callers consume the arrays directly.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(boxes) == 0:
        return EMPTY_BOXES.copy(), EMPTY_SCORES.copy(), EMPTY_LABELS.copy()

    order = sorted(range(len(boxes)), key=lambda i: -scores[i])  # highest score seeds each cluster
    used = [False] * len(boxes)
    m_boxes, m_scores, m_labels = [], [], []
    for i in order:
        if used[i]:
            continue
        used[i] = True
        cur = boxes[i].copy()
        cur_score = float(scores[i])
        merged = True
        while merged:  # keep absorbing boxes overlapping the growing hull (transitive seams)
            merged = False
            cur_area = (cur[2] - cur[0]) * (cur[3] - cur[1])
            for j in order:
                if used[j] or (class_aware and labels[j] != labels[i]):
                    continue
                aj = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
                if _ios(cur, boxes[j], cur_area, aj) >= iou_thresh:
                    used[j] = True
                    cur = np.array([min(cur[0], boxes[j][0]), min(cur[1], boxes[j][1]),
                                    max(cur[2], boxes[j][2]), max(cur[3], boxes[j][3])])
                    cur_score = max(cur_score, float(scores[j]))
                    merged = True
                    break
        m_boxes.append(cur)
        m_scores.append(cur_score)
        m_labels.append(int(labels[i]))
    return (np.asarray(m_boxes, dtype=np.float32),
            np.asarray(m_scores, dtype=np.float32),
            np.asarray(m_labels, dtype=np.int64))
