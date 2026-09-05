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

import math
from typing import NamedTuple, overload

import numpy as np

EMPTY_BOXES = np.zeros((0, 4), dtype=np.float32)
EMPTY_LABELS = np.zeros((0,), dtype=np.int64)
EMPTY_SCORES = np.zeros((0,), dtype=np.float32)


class MaskPatch(NamedTuple):
    """One detection's instance-seg mask, kept tile-local rather than expanded to a full-raster
    canvas: ``patch`` is a small dense soft-mask array (today, always tile-sized: the tile-local
    array a per-tile model forward already produced), ``offset_x``/``offset_y`` place its ``[0, 0]``
    pixel in full-image (or full-raster) pixel space. A consumer that needs full-image pixel
    coordinates (a polygon for export, a composited canvas) adds the offset at the point of use,
    the same "defer the expansion" convention ``export.py`` already uses for the untiled path's own
    dense masks (see ``mask_to_polygon_points``): this representation just makes that deferral
    mandatory instead of optional, since a tiled source raster can be too large to ever hold one
    full-size mask per detection.
    """

    patch: np.ndarray
    offset_x: int
    offset_y: int


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


def tile_within_extent(tile_x: int, tile_y: int, tile_size: int, width: int, height: int) -> bool:
    """Whether a tile's full rect fits inside the image's real (unpadded) extent.

    ``tile_positions`` only excludes an origin that falls entirely in the padding region; a
    tile whose origin is in-bounds can still extend past ``width``/``height`` on its far edge
    (the case ``crop_pad_tile`` zero-pads). A caller that needs every kept tile fully real
    (no synthetic padding pixels), such as a spatial train/val split, filters on this first.
    """
    return tile_x + tile_size <= width and tile_y + tile_size <= height


def rect_contains_rect(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int],
) -> bool:
    """Whether half-open pixel rect ``inner`` lies fully inside half-open pixel rect ``outer``."""
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    return ox0 <= ix0 and oy0 <= iy0 and ix1 <= ox1 and iy1 <= oy1


def rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Whether two half-open pixel rects share any pixel; sharing only an edge does not count."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def rect_contains_tile(rect: tuple[int, int, int, int], tile_x: int, tile_y: int, tile_size: int) -> bool:
    """Whether a tile's rect lies fully inside a half-open pixel rect ``(x0, y0, x1, y1)``."""
    return rect_contains_rect(rect, (tile_x, tile_y, tile_x + tile_size, tile_y + tile_size))


def region_halo(
    rect: tuple[int, int, int, int], mosaic_width: int, mosaic_height: int,
    tile_size: int, overlap: float,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """The ``(haloed_rect, inner_rect)`` pair for tiled inference over one sub-region of a
    larger mosaic (a block-aware calibration/holdout region), each a half-open pixel rect.

    ``halo = ceil((tile_size - stride) / 2)`` is :func:`reconstruct_core`'s own ``margin``,
    rounded up to a whole pixel: the minimum context production tiling already guarantees at
    every tile's edge from its neighbor, never a separately chosen, larger buffer. Expanding
    ``rect`` by ``halo`` on every side (clipped to the mosaic's own ``(0, 0, mosaic_width,
    mosaic_height)`` bounds) restores exactly that minimum around the sub-region's own boundary.
    A caller runs tiled inference (``predict_tiled``) over a ``_RegionView`` on the returned
    haloed rect, then keeps only detections whose box center lands in the returned inner rect
    (``rect`` unchanged) and clips GT the same way, discarding the halo band from the scored
    result, never from the pixels the model actually saw.

    This bounds, but does not eliminate, a small perimeter bias at the inner rect's own edge (see
    :func:`reconstruct_core`'s own docstring): a detection near that edge still sees only
    production's minimum guaranteed context, not the fuller context an interior detection sees
    from tiles further beyond it. Kept as designed; shrinking it further is not this function's
    job.
    """
    x0, y0, x1, y1 = rect
    halo = math.ceil((tile_size - compute_stride(tile_size, overlap)) / 2.0)
    haloed = (
        max(0, x0 - halo), max(0, y0 - halo),
        min(mosaic_width, x1 + halo), min(mosaic_height, y1 + halo),
    )
    return haloed, (x0, y0, x1, y1)


def clip_boxes_to_tile(
    boxes: np.ndarray, labels: np.ndarray, tile_x: int, tile_y: int,
    tile_w: int, tile_h: int, min_box_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect full-image-px boxes with a ``tile_w`` x ``tile_h`` tile; drop seam slivers; emit
    tile-local xyxy.

    A box clipped by the tile edge is dropped only when the *visible* (clipped) part is a sliver:
    its characteristic size ``sqrt(iw*ih) < min_box_size``. ``min_box_size`` is derived per dataset
    from the class's average box size (a partial object counts unless it's a tiny sliver; see
    ``TiledDetectionDataset``), not a fixed fraction. Boxes fully inside the tile are always kept.

    ``tile_w``/``tile_h`` need not be equal: every square caller (a sliding-window training/
    inference tile) passes ``tile_w == tile_h == tile_size``, and the rectangular case exists for
    a haloed calibration/holdout block, whose own extent need not be square.
    """
    boxes = np.asarray(boxes)
    labels = np.asarray(labels)
    if len(boxes) == 0:
        return EMPTY_BOXES.copy(), EMPTY_LABELS.copy()
    tx2, ty2 = tile_x + tile_w, tile_y + tile_h
    ix1 = np.maximum(boxes[:, 0], tile_x)
    iy1 = np.maximum(boxes[:, 1], tile_y)
    ix2 = np.minimum(boxes[:, 2], tx2)
    iy2 = np.minimum(boxes[:, 3], ty2)
    iw, ih = ix2 - ix1, iy2 - iy1
    visible = (iw > 0) & (ih > 0)
    was_clipped = ((boxes[:, 0] < tile_x) | (boxes[:, 1] < tile_y)
                   | (boxes[:, 2] > tx2) | (boxes[:, 3] > ty2))
    # Negative extents are clamped before the size so an empty intersection never feeds sqrt;
    # the clamp changes nothing kept, since only visible boxes survive the mask below.
    char_size = (np.maximum(iw, 0) * np.maximum(ih, 0)) ** 0.5
    keep = visible & ~(was_clipped & (char_size < min_box_size))
    if not keep.any():
        return EMPTY_BOXES.copy(), EMPTY_LABELS.copy()
    out = np.stack([ix1[keep] - tile_x, iy1[keep] - tile_y,
                    ix2[keep] - tile_x, iy2[keep] - tile_y], axis=1)
    return out.astype(np.float32), labels[keep].astype(np.int64)


def clipped_boxes_per_tile(
    boxes: np.ndarray, labels: np.ndarray, positions: list[tuple[int, int]],
    tile_size: int, min_box_size: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """:func:`clip_boxes_to_tile` for every position at once.

    Result ``i`` equals ``clip_boxes_to_tile(boxes, labels, *positions[i], ...)`` exactly
    (values, dtypes, ordering, fresh arrays), but the cost scales with the box-tile incidences
    rather than positions times boxes: each box's candidate tile rows/columns come from one
    ``searchsorted`` over the distinct origins, and only tiles with a candidate pay a clip call.
    The candidate range is a strict superset of visibility, so :func:`clip_boxes_to_tile` stays
    the one authority on what a tile keeps. ``positions`` is a :func:`tile_positions` result:
    distinct origins on a regular grid.
    """
    n_pos = len(positions)
    boxes = np.asarray(boxes)
    labels = np.asarray(labels)
    if n_pos == 0 or len(boxes) == 0:
        return [(EMPTY_BOXES.copy(), EMPTY_LABELS.copy()) for _ in range(n_pos)]
    xs = np.unique(np.asarray([p[0] for p in positions], dtype=np.int64))
    ys = np.unique(np.asarray([p[1] for p in positions], dtype=np.int64))
    # float64 holds every float32/float64 coordinate and every origin exactly, so these strict
    # bounds agree with the clip's own visibility arithmetic instead of re-rounding it.
    bx1 = boxes[:, 0].astype(np.float64)
    by1 = boxes[:, 1].astype(np.float64)
    bx2 = boxes[:, 2].astype(np.float64)
    by2 = boxes[:, 3].astype(np.float64)
    col_lo = np.searchsorted(xs, bx1 - tile_size, side="right")
    col_hi = np.searchsorted(xs, bx2, side="left")
    row_lo = np.searchsorted(ys, by1 - tile_size, side="right")
    row_hi = np.searchsorted(ys, by2, side="left")
    candidates: dict[tuple[int, int], list[int]] = {}
    for i in range(len(boxes)):
        for k in range(row_lo[i], row_hi[i]):
            ty = int(ys[k])
            for j in range(col_lo[i], col_hi[i]):
                candidates.setdefault((int(xs[j]), ty), []).append(i)
    results: list[tuple[np.ndarray, np.ndarray] | None] = [None] * n_pos
    index_of = {(int(tx), int(ty)): m for m, (tx, ty) in enumerate(positions)}
    for key, idxs in candidates.items():
        m = index_of.get(key)
        if m is None:
            continue  # a grid cell tile_positions never emitted (dropped as pure padding)
        results[m] = clip_boxes_to_tile(
            boxes[idxs], labels[idxs], key[0], key[1], tile_size, tile_size, min_box_size)
    return [(EMPTY_BOXES.copy(), EMPTY_LABELS.copy()) if r is None else r for r in results]


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


@overload
def reconstruct_core(
    per_tile_boxes: list[np.ndarray], per_tile_scores: list[np.ndarray],
    per_tile_labels: list[np.ndarray], tile_info: list[dict], tile_size: int, stride: int,
    per_tile_masks: None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
@overload
def reconstruct_core(
    per_tile_boxes: list[np.ndarray], per_tile_scores: list[np.ndarray],
    per_tile_labels: list[np.ndarray], tile_info: list[dict], tile_size: int, stride: int,
    per_tile_masks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[MaskPatch]]: ...
def reconstruct_core(
    per_tile_boxes: list[np.ndarray], per_tile_scores: list[np.ndarray],
    per_tile_labels: list[np.ndarray], tile_info: list[dict], tile_size: int, stride: int,
    per_tile_masks: list[np.ndarray] | None = None,
) -> (tuple[np.ndarray, np.ndarray, np.ndarray]
      | tuple[np.ndarray, np.ndarray, np.ndarray, list[MaskPatch]]):
    """Shift tile-local detections to full-image coords, keep center-in-core, clip.

    ``tile_info[i]`` = ``{'tile_x','tile_y','original_width','original_height'}``.

    ``per_tile_masks[i]`` (optional), when given, is the ``[N_i, tile_size, tile_size]`` tile-local
    soft-mask stack for tile ``i``, same order/length as ``per_tile_boxes[i]``. A surviving
    detection's mask travels with it (as a :class:`MaskPatch`, still tile-local, offset by the
    tile's own full-image origin, never expanded to a full-image canvas here); a detection dropped
    by the center-in-core check or never returned from a tile drops its mask too, so there is never
    an orphaned mask for a box that did not survive. Returns a 4-tuple (boxes, scores, labels,
    masks) when ``per_tile_masks`` is given, else the original 3-tuple, byte-identical to the
    boxes-only behavior every existing caller of this function already depends on.
    """
    margin = (tile_size - stride) / 2.0
    collect_masks = per_tile_masks is not None
    out_b, out_s, out_l = [], [], []
    out_m: list[MaskPatch] = []
    mask_stream: list[np.ndarray] | list[None] = (
        per_tile_masks if per_tile_masks is not None else [None] * len(per_tile_boxes))
    for tb, ts, tl, info, tm in zip(per_tile_boxes, per_tile_scores, per_tile_labels, tile_info, mask_stream):
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
            if collect_masks:
                assert tm is not None, "mask_stream carries only ndarrays when collect_masks is True"
                out_m.append(MaskPatch(patch=np.asarray(tm[i]), offset_x=int(tx), offset_y=int(ty)))
    if not out_b:
        empty = (EMPTY_BOXES.copy(), EMPTY_SCORES.copy(), EMPTY_LABELS.copy())
        return (*empty, []) if collect_masks else empty
    result = (np.asarray(out_b, dtype=np.float32),
              np.asarray(out_s, dtype=np.float32),
              np.asarray(out_l, dtype=np.int64))
    return (*result, out_m) if collect_masks else result


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


def _composite_mask_patches(merged_box: np.ndarray, patches: list[MaskPatch]) -> MaskPatch:
    """Paste every tile-local mask patch absorbed into one merged detection onto a canvas sized to
    the merged box's own (small) hull, not the source raster: a merged cluster is a handful of
    nearby tile-local detections, so this stays cheap even behind a huge orthomosaic. Overlapping
    patches take the pixelwise max (masks are soft probabilities; a plain boolean OR would discard
    the confidence signal NMM otherwise preserves via ``max score kept``)."""
    x0, y0 = int(np.floor(merged_box[0])), int(np.floor(merged_box[1]))
    x1, y1 = int(np.ceil(merged_box[2])), int(np.ceil(merged_box[3]))
    canvas_w, canvas_h = max(1, x1 - x0), max(1, y1 - y0)
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    for mp in patches:
        patch = np.asarray(mp.patch)
        ph, pw = patch.shape[-2:]
        dy, dx = mp.offset_y - y0, mp.offset_x - x0  # patch-local -> canvas-local shift
        r0, r1 = max(0, -dy), min(ph, canvas_h - dy)
        c0, c1 = max(0, -dx), min(pw, canvas_w - dx)
        if r0 >= r1 or c0 >= c1:
            continue  # patch falls entirely outside the merged hull (shouldn't happen, defensive)
        canvas[r0 + dy:r1 + dy, c0 + dx:c1 + dx] = np.maximum(
            canvas[r0 + dy:r1 + dy, c0 + dx:c1 + dx], patch[r0:r1, c0:c1])
    return MaskPatch(patch=canvas, offset_x=x0, offset_y=y0)


@overload
def global_merge(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
    iou_thresh: float, class_aware: bool = True,
    per_det_masks: None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
@overload
def global_merge(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
    iou_thresh: float, class_aware: bool = True,
    *, per_det_masks: list[MaskPatch],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[MaskPatch]]: ...
def global_merge(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
    iou_thresh: float, class_aware: bool = True,
    per_det_masks: list[MaskPatch] | None = None,
) -> (tuple[np.ndarray, np.ndarray, np.ndarray]
      | tuple[np.ndarray, np.ndarray, np.ndarray, list[MaskPatch]]):
    """Cross-tile Non-Max *Merging*: union overlapping same-class boxes (bbox hull, max score)
    instead of suppressing the lower-score one, recovering an object split across a tile seam
    into two partial boxes (SAHI's NMM). Returns merged ``(boxes, scores, labels)``, new boxes,
    not a subset of indices like :func:`global_nms`, so callers consume the arrays directly.

    ``per_det_masks`` (optional), parallel to ``boxes``, carries each input detection's tile-local
    :class:`MaskPatch`. When given, every merged cluster's absorbed patches are composited (see
    :func:`_composite_mask_patches`) into one new :class:`MaskPatch` sized to the merged box's own
    hull, and returned as a 4th value parallel to the merged boxes; the cluster membership used to
    do that compositing is exactly the ``used``/absorption walk this function already performs, so
    a caller never has to reconstruct it from a separately-exposed group index. Omitting
    ``per_det_masks`` returns the original 3-tuple, byte-identical to the boxes-only behavior every
    existing caller of this function already depends on.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    collect_masks = per_det_masks is not None
    if len(boxes) == 0:
        empty = (EMPTY_BOXES.copy(), EMPTY_SCORES.copy(), EMPTY_LABELS.copy())
        return (*empty, []) if collect_masks else empty

    order = sorted(range(len(boxes)), key=lambda i: -scores[i])  # highest score seeds each cluster
    used = [False] * len(boxes)
    m_boxes, m_scores, m_labels = [], [], []
    m_masks: list[MaskPatch] = []
    for i in order:
        if used[i]:
            continue
        used[i] = True
        cur = boxes[i].copy()
        cur_score = float(scores[i])
        members = [i] if collect_masks else None
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
                    if collect_masks:
                        assert members is not None, "members is a list whenever collect_masks is True"
                        members.append(j)
                    break
        m_boxes.append(cur)
        m_scores.append(cur_score)
        m_labels.append(int(labels[i]))
        if collect_masks:
            assert per_det_masks is not None and members is not None, (
                "collect_masks is True only when per_det_masks was given, and members is a "
                "list then too")
            m_masks.append(_composite_mask_patches(cur, [per_det_masks[k] for k in members]))
    result = (np.asarray(m_boxes, dtype=np.float32),
              np.asarray(m_scores, dtype=np.float32),
              np.asarray(m_labels, dtype=np.int64))
    return (*result, m_masks) if collect_masks else result
