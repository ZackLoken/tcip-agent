"""P3 integrity fold-ins: cross-tile NMM merge + channel-generic tiled crop/pad."""

from __future__ import annotations

import numpy as np


def test_global_merge_unions_seam_split_boxes():
    from tcip_mcp.pipelines.data.tiling import global_merge

    # Two same-class boxes overlapping near a seam -> merged into one hull box, max score kept.
    boxes = np.array([[0, 0, 10, 10], [8, 0, 18, 10]], dtype=np.float32)  # IoU 20/180 ≈ 0.11
    scores = np.array([0.9, 0.7], dtype=np.float32)
    labels = np.array([1, 1], dtype=np.int64)

    mb, ms, ml = global_merge(boxes, scores, labels, iou_thresh=0.05)
    assert len(mb) == 1
    assert list(mb[0]) == [0.0, 0.0, 18.0, 10.0]  # bounding hull of both
    assert ms[0] == 0.9  # max score of the cluster
    assert ml[0] == 1


def test_global_merge_keeps_different_classes_apart():
    from tcip_mcp.pipelines.data.tiling import global_merge

    boxes = np.array([[0, 0, 10, 10], [8, 0, 18, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.7], dtype=np.float32)
    labels = np.array([1, 2], dtype=np.int64)  # different classes must not merge
    mb, _, _ = global_merge(boxes, scores, labels, iou_thresh=0.05)
    assert len(mb) == 2


def test_global_merge_absorbs_low_iou_fragment_via_ios():
    from tcip_mcp.pipelines.data.tiling import global_merge

    # A small partial fragment mostly inside a fuller detection: IoU ≈ 0.04 (NMS would keep both),
    # IoS = 1.0 (fully contained). NMM must merge it — the seam-split case merging exists for.
    boxes = np.array([[0, 0, 10, 10], [4, 4, 6, 6]], dtype=np.float32)
    scores = np.array([0.9, 0.6], dtype=np.float32)
    labels = np.array([1, 1], dtype=np.int64)
    mb, _, _ = global_merge(boxes, scores, labels, iou_thresh=0.3)  # 0.3 would not merge on IoU
    assert len(mb) == 1
    assert list(mb[0]) == [0.0, 0.0, 10.0, 10.0]


def test_global_merge_leaves_disjoint_boxes():
    from tcip_mcp.pipelines.data.tiling import global_merge

    boxes = np.array([[0, 0, 5, 5], [50, 50, 55, 55]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    labels = np.array([1, 1], dtype=np.int64)
    mb, _, _ = global_merge(boxes, scores, labels, iou_thresh=0.3)
    assert len(mb) == 2  # no overlap -> untouched


def test_crop_pad_tile_numpy_multichannel_pads_edges():
    from tcip_mcp.pipelines.inference.generic_predictor import _crop_pad_tile

    arr = np.arange(5 * 5 * 2).reshape(5, 5, 2).astype(np.float32)  # 2-channel raster
    # Tile of size 4 at (3, 3): only a 2x2 window exists -> zero-pad bottom/right to 4x4x2.
    tile = _crop_pad_tile(arr, 3, 3, 4, 5, 5)
    assert tile.shape == (4, 4, 2)
    assert np.array_equal(tile[:2, :2], arr[3:5, 3:5])
    assert np.all(tile[2:, :] == 0) and np.all(tile[:, 2:] == 0)


def test_crop_pad_tile_pil_path():
    from PIL import Image

    from tcip_mcp.pipelines.inference.generic_predictor import _crop_pad_tile

    tile = _crop_pad_tile(Image.new("RGB", (5, 5), (7, 7, 7)), 3, 3, 4, 5, 5)
    assert isinstance(tile, Image.Image) and tile.size == (4, 4)
