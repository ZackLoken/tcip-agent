"""clip_boxes_to_tile and clipped_boxes_per_tile semantics.

The per-box reference below restates the clip/sliver rules one box at a time, independent of
the shipped vectorized code, so the equivalence tests catch a vectorization that drifts from
the semantics and the property test states those semantics on their own terms.
"""

from __future__ import annotations

import time

import numpy as np

from tcip_mcp.pipelines.data import tiling


def _clip_reference(boxes, labels, tile_x, tile_y, tile_size, min_box_size):
    """One box at a time: intersect, drop empty, drop clipped slivers, emit tile-local xyxy."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
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
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(out_boxes, dtype=np.float32), np.asarray(out_labels, dtype=np.int64)


def _assert_identical(got, want):
    gb, gl = got
    wb, wl = want
    assert gb.dtype == wb.dtype and gl.dtype == wl.dtype
    assert np.array_equal(gb, wb), f"boxes differ: {gb} vs {wb}"
    assert np.array_equal(gl, wl), f"labels differ: {gl} vs {wl}"


# Tile at (200, 200), size 64, so edges sit at 200 and 264.
_ADVERSARIAL = [
    # fully inside
    np.array([[210.0, 210.0, 230.0, 230.0]]),
    # straddling each of the four tile edges
    np.array([[180.0, 210.0, 220.0, 230.0], [210.0, 180.0, 230.0, 220.0],
              [250.0, 210.0, 290.0, 230.0], [210.0, 250.0, 230.0, 290.0]]),
    # visible part exactly at the threshold (3x3, char 3.0 with min_box_size 3.0: kept, rule is <)
    np.array([[261.0, 261.0, 290.0, 290.0]]),
    # visible part just under the threshold (2x2, char 2.0: dropped)
    np.array([[262.0, 262.0, 290.0, 290.0]]),
    # duplicates
    np.array([[210.0, 210.0, 230.0, 230.0], [210.0, 210.0, 230.0, 230.0]]),
    # no overlap at all, and a box merely touching the tile edge (zero-width intersection)
    np.array([[0.0, 0.0, 10.0, 10.0], [150.0, 210.0, 200.0, 230.0]]),
    # degenerate zero-area box inside the tile
    np.array([[220.0, 220.0, 220.0, 220.0]]),
    # box equal to the tile itself, and one covering the whole tile plus margin
    np.array([[200.0, 200.0, 264.0, 264.0], [190.0, 190.0, 274.0, 274.0]]),
    # negative coordinates
    np.array([[-50.0, -50.0, 210.0, 210.0]]),
    # empty
    np.zeros((0, 4)),
]


def test_clip_matches_the_per_box_reference_on_adversarial_geometries():
    for boxes in _ADVERSARIAL:
        for dtype in (np.float32, np.float64):
            b = boxes.astype(dtype)
            labels = np.arange(1, len(b) + 1, dtype=np.int64)
            got = tiling.clip_boxes_to_tile(b, labels, 200, 200, 64, 3.0)
            want = _clip_reference(b, labels, 200, 200, 64, 3.0)
            _assert_identical(got, want)


def test_clip_matches_the_per_box_reference_on_random_geometries():
    rng = np.random.default_rng(11)
    for _ in range(40):
        n = int(rng.integers(0, 200))
        xy = rng.uniform(-40, 460, size=(n, 2))
        wh = rng.uniform(0, 80, size=(n, 2))
        boxes = np.concatenate([xy, xy + wh], axis=1)
        labels = rng.integers(1, 4, size=n).astype(np.int64)
        tile_x, tile_y = int(rng.integers(0, 300)), int(rng.integers(0, 300))
        min_box_size = float(rng.uniform(0, 20))
        for dtype in (np.float32, np.float64):
            got = tiling.clip_boxes_to_tile(boxes.astype(dtype), labels, tile_x, tile_y, 128,
                                            min_box_size)
            want = _clip_reference(boxes.astype(dtype), labels, tile_x, tile_y, 128,
                                   min_box_size)
            _assert_identical(got, want)


def test_clip_semantics_stated_as_properties():
    """The rules themselves, restated in plain python floats over the float64 inputs the
    arithmetic then matches exactly: a visible intersection is kept unless the box was clipped
    and its visible part's characteristic size falls below the cutoff; outputs are tile-local,
    in input order."""
    rng = np.random.default_rng(23)
    tile_size = 100
    for _ in range(25):
        n = int(rng.integers(1, 120))
        xy = rng.uniform(-60, 400, size=(n, 2))
        wh = rng.uniform(0, 90, size=(n, 2))
        boxes = np.concatenate([xy, xy + wh], axis=1)
        labels = np.arange(1, n + 1, dtype=np.int64)
        tx, ty = int(rng.integers(0, 250)), int(rng.integers(0, 250))
        cutoff = float(rng.uniform(0, 25))
        out_boxes, out_labels = tiling.clip_boxes_to_tile(boxes, labels, tx, ty, tile_size,
                                                          cutoff)
        expected = []
        for i, (bx1, by1, bx2, by2) in enumerate(boxes.tolist()):
            iw = min(bx2, tx + tile_size) - max(bx1, tx)
            ih = min(by2, ty + tile_size) - max(by1, ty)
            if iw <= 0 or ih <= 0:
                continue
            clipped = bx1 < tx or by1 < ty or bx2 > tx + tile_size or by2 > ty + tile_size
            if clipped and (iw * ih) ** 0.5 < cutoff:
                continue
            expected.append((i, (max(bx1, tx) - tx, max(by1, ty) - ty,
                                 min(bx2, tx + tile_size) - tx, min(by2, ty + tile_size) - ty)))
        assert [int(lab) for lab in out_labels] == [labels[i] for i, _ in expected]
        assert np.allclose(out_boxes, np.array([r for _, r in expected]).reshape(-1, 4),
                           atol=1e-4)
        if len(out_boxes):
            assert out_boxes.min() >= 0 and out_boxes.max() <= tile_size


def test_bulk_clip_equals_the_per_tile_calls_over_a_grid():
    rng = np.random.default_rng(31)
    height, width, tile_size = 500, 700, 128
    stride = tiling.compute_stride(tile_size, 0.2)
    positions = tiling.tile_positions(height, width, tile_size, stride)
    for _ in range(5):
        n = int(rng.integers(0, 150))
        xy = rng.uniform(-40, 700, size=(n, 2))
        wh = rng.uniform(0, 100, size=(n, 2))
        boxes = np.concatenate([xy, xy + wh], axis=1).astype(np.float32)
        labels = rng.integers(1, 3, size=n).astype(np.int64)
        results = tiling.clipped_boxes_per_tile(boxes, labels, positions, tile_size, 6.0)
        assert len(results) == len(positions)
        for (tx, ty), got in zip(positions, results):
            _assert_identical(got, _clip_reference(boxes, labels, tx, ty, tile_size, 6.0))


def test_bulk_clip_returns_fresh_arrays_per_tile():
    """Empty tiles must not share one mutable array: a caller mutating one tile's boxes must
    never see the change in another's."""
    positions = [(0, 0), (100, 0), (0, 100)]
    results = tiling.clipped_boxes_per_tile(
        np.array([[10.0, 10.0, 20.0, 20.0]]), np.array([1]), positions, 64, 2.0)
    empties = [b for b, _lab in results if len(b) == 0]
    assert len(empties) == 2
    assert empties[0] is not empties[1]
    empties[0].resize((1, 4), refcheck=False)
    assert empties[1].shape == (0, 4)


def test_bulk_clip_cost_scales_with_incidences_not_positions_times_boxes():
    """A generous wall-clock rail on a large geometry: the cost must follow the box-tile
    incidences, so a dense grid over many small boxes finishes in seconds."""
    rng = np.random.default_rng(5)
    width = height = 56_000
    tile_size, stride = 224, 179
    positions = tiling.tile_positions(height, width, tile_size, stride)
    assert len(positions) > 90_000
    n = 50_000
    xy = rng.uniform(0, width - 60, size=(n, 2))
    wh = rng.uniform(8, 48, size=(n, 2))
    boxes = np.concatenate([xy, xy + wh], axis=1).astype(np.float32)
    labels = rng.integers(1, 3, size=n).astype(np.int64)
    start = time.perf_counter()
    results = tiling.clipped_boxes_per_tile(boxes, labels, positions, tile_size, 6.0)
    elapsed = time.perf_counter() - start
    assert len(results) == len(positions)
    assert sum(len(lab) for _b, lab in results) >= n  # every box lands in at least one tile
    assert elapsed < 60.0
