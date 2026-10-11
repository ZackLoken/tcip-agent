"""slice_candidates and clip_rows_to_tile over box rows.

The per-box reference below restates the clip and sliver rules one box at a time, independent of
the shipped code, so the equivalence tests catch a clip that drifts from the semantics and the
property test states those semantics on their own terms.
"""

from __future__ import annotations

import time

import numpy as np

from tcip_annotation.matching import box_ring, shapely_geometry
from tcip_annotation.state import BBox
from tcip_mcp.pipelines.data.datasets import (
    clip_rows_to_tile,
    slice_candidates,
)
from tcip_mcp.pipelines.slicing import slice_lattice


def _clip_reference(boxes, tile_x, tile_y, tile_w, tile_h, min_box_size):
    """One box at a time: intersect, drop empty, drop clipped slivers; the rows kept and their
    tile-local xyxy.
    """
    tx2, ty2 = tile_x + tile_w, tile_y + tile_h
    rows, out = [], []
    for i, (bx1, by1, bx2, by2) in enumerate(
        np.asarray(boxes, dtype=np.float64).tolist()
    ):
        ix1, iy1 = max(bx1, tile_x), max(by1, tile_y)
        ix2, iy2 = min(bx2, tx2), min(by2, ty2)
        iw, ih = ix2 - ix1, iy2 - iy1
        if iw <= 0 or ih <= 0:
            continue
        clipped = bx1 < tile_x or by1 < tile_y or bx2 > tx2 or by2 > ty2
        if clipped and (iw * ih) ** 0.5 < min_box_size:
            continue
        rows.append(i)
        out.append([ix1 - tile_x, iy1 - tile_y, ix2 - tile_x, iy2 - tile_y])
    return np.asarray(out, dtype=np.float32).reshape(-1, 4), np.asarray(
        rows, dtype=np.int64
    )


def _clip(boxes, tile, min_box_size, rows=None):
    """``boxes`` as box rows clipped to ``tile`` (:func:`clip_rows_to_tile`): kept boxes, rows."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    shapes = [shapely_geometry([box_ring(BBox(*b))]) for b in boxes.tolist()]
    rows = np.arange(len(boxes)) if rows is None else rows
    kept, kept_rows, _regions = clip_rows_to_tile(
        shapes, boxes, rows, tile, min_box_size
    )
    return kept, kept_rows


def _assert_identical(got, want):
    (gb, gr), (wb, wr) = got, want
    assert gb.dtype == wb.dtype
    assert np.array_equal(gb, wb), f"boxes differ: {gb} vs {wb}"
    assert np.array_equal(gr, wr), f"rows differ: {gr} vs {wr}"


# Tile at (200, 200), size 64, so edges sit at 200 and 264.
_ADVERSARIAL = [
    # fully inside
    np.array([[210.0, 210.0, 230.0, 230.0]]),
    # straddling each of the four tile edges
    np.array(
        [
            [180.0, 210.0, 220.0, 230.0],
            [210.0, 180.0, 230.0, 220.0],
            [250.0, 210.0, 290.0, 230.0],
            [210.0, 250.0, 230.0, 290.0],
        ]
    ),
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
        _assert_identical(
            _clip(boxes, (200, 200, 264, 264), 3.0),
            _clip_reference(boxes, 200, 200, 64, 64, 3.0),
        )


def test_clip_matches_the_per_box_reference_on_random_geometries():
    rng = np.random.default_rng(11)
    for _ in range(40):
        n = int(rng.integers(0, 200))
        xy = rng.uniform(-40, 460, size=(n, 2))
        wh = rng.uniform(0.5, 80, size=(n, 2))
        boxes = np.concatenate([xy, xy + wh], axis=1)
        tile_x, tile_y = int(rng.integers(0, 300)), int(rng.integers(0, 300))
        min_box_size = float(rng.uniform(0, 20))
        _assert_identical(
            _clip(
                boxes,
                (tile_x, tile_y, tile_x + 128, tile_y + 128),
                min_box_size,
            ),
            _clip_reference(boxes, tile_x, tile_y, 128, 128, min_box_size),
        )


def test_a_rectangular_tile_clips_against_its_own_extent():
    """A box's visible part is a 54x40 sliver against a 64-wide square tile (dropped at
    min_box_size=100), but the same box sits entirely inside a 300-wide tile at the same origin,
    unclipped and kept regardless of size.
    """
    boxes = np.array([[210.0, 210.0, 350.0, 250.0]])
    square = _clip(boxes, (200, 200, 264, 264), 100.0)
    rect = _clip(boxes, (200, 200, 500, 264), 100.0)
    assert len(square[0]) == 0
    assert len(rect[0]) == 1
    assert np.allclose(rect[0][0], [10.0, 10.0, 150.0, 50.0])


def test_the_candidates_clipped_equal_every_row_clipped_over_a_lattice():
    rng = np.random.default_rng(31)
    height, width, tile_size = 500, 700, 128
    slices = slice_lattice(height, width, tile_size, 0.2)
    for _ in range(5):
        n = int(rng.integers(0, 150))
        xy = rng.uniform(-40, 700, size=(n, 2))
        wh = rng.uniform(0.5, 100, size=(n, 2))
        boxes = np.concatenate([xy, xy + wh], axis=1).astype(np.float32)
        candidates = slice_candidates(boxes, slices)
        assert len(candidates) == len(slices)
        for (tx, ty, x1, y1), rows in zip(slices, candidates):
            _assert_identical(
                _clip(boxes, (tx, ty, x1, y1), 6.0, rows),
                _clip_reference(boxes, tx, ty, x1 - tx, y1 - ty, 6.0),
            )


def test_each_slice_gets_its_own_candidate_array():
    """Empty slices must not share one mutable array: a caller mutating one slice's candidates
    must never see the change in another's.
    """
    slices = [(0, 0, 64, 64), (100, 0, 164, 64), (0, 100, 64, 164)]
    candidates = slice_candidates(np.array([[10.0, 10.0, 20.0, 20.0]]), slices)
    empties = [rows for rows in candidates if len(rows) == 0]
    assert len(empties) == 2
    assert empties[0] is not empties[1]
    empties[0].resize((1,), refcheck=False)
    assert empties[1].shape == (0,)


def test_candidate_cost_scales_with_incidences_not_positions_times_boxes():
    """A generous wall-clock rail on a large geometry: the cost must follow the box-slice
    incidences, so a dense lattice over many small boxes finishes in seconds.
    """
    rng = np.random.default_rng(5)
    width = height = 56_000
    tile_size = 224
    positions = slice_lattice(height, width, tile_size, 0.2)
    assert len(positions) > 90_000
    n = 50_000
    xy = rng.uniform(0, width - 60, size=(n, 2))
    wh = rng.uniform(8, 48, size=(n, 2))
    boxes = np.concatenate([xy, xy + wh], axis=1).astype(np.float32)
    start = time.perf_counter()
    candidates = slice_candidates(boxes, positions)
    elapsed = time.perf_counter() - start
    assert len(candidates) == len(positions)
    assert (
        sum(len(rows) for rows in candidates) >= n
    )  # every box lands in at least one tile
    assert elapsed < 60.0
