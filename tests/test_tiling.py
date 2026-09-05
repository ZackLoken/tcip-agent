"""Sliding-window tiling geometry + TiledDetectionDataset wrapper.

The geometry tests are pure numpy (no torch). The wrapper tests skip if torch is
absent (they go through ``build_dataset``).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tcip_mcp.pipelines.data import tiling


# --------------------------------------------------------------------------
# Pure geometry
# --------------------------------------------------------------------------

def test_tile_positions_pad_and_stride():
    assert tiling.compute_stride(224, 0.2) == 179
    pos = set(tiling.tile_positions(300, 300, 224, 179))
    assert pos == {(0, 0), (179, 0), (0, 179), (179, 179)}
    assert all(tx < 300 and ty < 300 for tx, ty in pos)  # no pure-padding origin


def test_clip_boxes_to_tile_sliver_drop_and_remap():
    # min_box_size=12: a clipped box counts unless its visible part is a sliver (< 12px char-size).
    # Fully-inside box -> always kept, remapped to tile-local (minus origin 200,200).
    tb, tl = tiling.clip_boxes_to_tile(np.array([[210., 210., 230., 230.]]), np.array([1]), 200, 200, 64, 64, 12.0)
    assert tb.shape == (1, 4)
    assert np.allclose(tb[0], [10, 10, 30, 30])
    # Straddling box whose visible part is substantial (clipped to 14x14, char 14 >= 12) -> kept.
    tb2, _ = tiling.clip_boxes_to_tile(np.array([[250., 250., 290., 290.]]), np.array([1]), 200, 200, 64, 64, 12.0)
    assert len(tb2) == 1
    # Straddling box whose visible part is a sliver (clipped to 9x9, char 9 < 12) -> dropped.
    tb3, _ = tiling.clip_boxes_to_tile(np.array([[255., 255., 265., 265.]]), np.array([1]), 200, 200, 64, 64, 12.0)
    assert len(tb3) == 0
    # Non-overlapping box -> no output.
    tb4, _ = tiling.clip_boxes_to_tile(np.array([[0., 0., 10., 10.]]), np.array([1]), 200, 200, 64, 64, 12.0)
    assert len(tb4) == 0


def test_dedup_boxes_class_aware():
    boxes = np.array([[0., 0., 20., 20.], [1., 1., 19., 19.]])  # IoU 0.81, same label
    db, _ = tiling.dedup_boxes(boxes, np.array([1, 1]), 0.8)
    assert len(db) == 1 and np.allclose(db[0], [0, 0, 20, 20])  # larger kept
    db2, _ = tiling.dedup_boxes(np.array([[0., 0., 10., 10.], [50., 50., 60., 60.]]), np.array([1, 1]), 0.8)
    assert len(db2) == 2  # distinct boxes both survive
    db3, _ = tiling.dedup_boxes(boxes, np.array([1, 2]), 0.8, class_aware=True)
    assert len(db3) == 2  # same geometry, different labels -> both survive
    db4, _ = tiling.dedup_boxes(boxes, np.array([1, 1]), 1.0)
    assert len(db4) == 2  # iou_thresh >= 1.0 is a no-op


def test_reconstruct_core_dedup_seam():
    tile_size, img_w, img_h = 64, 120, 64
    stride = tiling.compute_stride(64, 0.2)  # 51, margin 6.5
    # Same object at full-image center x=55: its center is in tile A's core, not tile B's.
    per_tile_boxes = [np.array([[50., 20., 60., 40.]]), np.array([[-1., 20., 9., 40.]])]
    per_tile_scores = [np.array([0.9]), np.array([0.8])]
    per_tile_labels = [np.array([1]), np.array([1])]
    tile_info = [
        {"tile_x": 0, "tile_y": 0, "original_width": img_w, "original_height": img_h},
        {"tile_x": 51, "tile_y": 0, "original_width": img_w, "original_height": img_h},
    ]
    b, s, ll = tiling.reconstruct_core(per_tile_boxes, per_tile_scores, per_tile_labels, tile_info, tile_size, stride)
    assert len(b) == 1
    assert np.allclose(b[0], [50, 20, 60, 40])
    assert s[0] == pytest.approx(0.9) and ll[0] == 1


def test_region_halo_expands_by_ceil_margin_and_clips_to_mosaic():
    haloed, inner = tiling.region_halo((100, 100, 300, 300), 1000, 800, 224, 0.2)
    assert inner == (100, 100, 300, 300)
    assert haloed == (77, 77, 323, 323)  # halo=ceil((224-179)/2)=23 on every side


def test_region_halo_clips_to_mosaic_bounds_at_an_edge():
    haloed, inner = tiling.region_halo((0, 0, 200, 50), 1000, 60, 224, 0.2)
    assert inner == (0, 0, 200, 50)
    assert haloed == (0, 0, 223, 60)  # y1 clips at the mosaic's own height (60), not 50+23=73


def test_region_halo_is_bounded_by_half_the_tile_for_every_overlap():
    """halo <= tile_size/2 for every overlap in [0, 1) (equality only at stride=1, even
    tile_size): well inside a spatial_strip_split buffer (>= tile_size), the invariant that
    guarantees a haloed region never reaches into a genuinely different split's tiles."""
    for tile_size in (64, 100, 224, 512):
        for overlap in (0.0, 0.1, 0.2, 0.5, 0.75, 0.99):
            stride = tiling.compute_stride(tile_size, overlap)
            halo = math.ceil((tile_size - stride) / 2.0)
            assert halo <= tile_size / 2.0
            haloed, inner = tiling.region_halo(
                (0, 0, tile_size, tile_size), tile_size * 4, tile_size * 4, tile_size, overlap)
            assert inner == (0, 0, tile_size, tile_size)
            assert haloed == (0, 0, tile_size + halo, tile_size + halo)


def test_global_nms_collapses_duplicates():
    boxes = np.array([[0., 0., 20., 20.], [1., 1., 21., 21.], [100., 100., 120., 120.]])
    keep = tiling.global_nms(boxes, np.array([0.9, 0.8, 0.95]), np.array([1, 1, 1]), 0.3)
    kept = set(int(i) for i in keep)
    assert len(kept) == 2 and 0 in kept and 2 in kept and 1 not in kept  # higher score of overlap kept


# --------------------------------------------------------------------------
# TiledDetectionDataset wrapper (needs torch)
# --------------------------------------------------------------------------

def _det_dataset(tmp_path: Path, n: int = 1, size: int = 128):
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (size, size), (120, 120, 120)).save(images_dir / f"img{i}.jpg")
        # YOLO "0 0.5 0.5 0.1 0.1" (normalized) -> pixel xyxy in a size×size image
        box = BBox(0.45 * size, 0.45 * size, 0.55 * size, 0.55 * size)
        json_io.write_annotations(str(labels_dir / f"img{i}.json"),
                                  [Annotation(subject="bud", geometry=box)], size, size, keep_empty=True)
    return images_dir, labels_dir


def test_tiled_detection_dataset_wrapper(tmp_path):
    torch = pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.datasets import build_dataset

    images_dir, labels_dir = _det_dataset(tmp_path)
    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                       subject="bud", tiling={"enabled": True, "tile_size": 64, "overlap": 0.2})
    assert len(ds) >= 1  # 128px image -> multiple tiles
    img, target = ds[0]
    assert tuple(img.shape) == (3, 64, 64)
    assert target["boxes"].shape[1] == 4
    assert (target["boxes"] >= 0).all() and (target["boxes"] <= 64).all()
    assert target["labels"].dtype == torch.int64


def test_tiled_dataset_derives_sliver_and_keeps_empty_tiles(tmp_path):
    pytest.importorskip("torch")
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.data.datasets import build_dataset

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (256, 256), (120, 120, 120)).save(images_dir / "a.jpg")
    # YOLO "0 0.1 0.1 0.1 0.1" in a 256×256 image -> a 25.6px box in the top-left corner
    json_io.write_annotations(str(labels_dir / "a.json"),
                              [Annotation(subject="bud", geometry=BBox(12.8, 12.8, 38.4, 38.4))],
                              256, 256, keep_empty=True)
    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                       subject="bud", tiling={"enabled": True, "tile_size": 64, "overlap": 0.2})
    # class_avg_size is derived from the class-average box size. sliver_frac would be too, but a
    # single box is too few to measure a size spread from (derive_sliver_frac's own min_samples
    # guard), an honest "underivable", not a value dressed as derived, so it falls back to 0.5.
    assert ds.class_avg_size == pytest.approx(25.6, abs=1.0)
    assert ds.sliver_frac == pytest.approx(0.5)
    assert ds.sliver_frac_source == "documented default (underivable: too few GT boxes to measure a spread)"
    assert ds.min_box_size == pytest.approx(0.5 * ds.class_avg_size)
    # skip_empty now defaults False -> tiles far from the object are kept as valid negatives.
    empties = sum(1 for i in range(len(ds)) if ds[i][1]["boxes"].shape[0] == 0)
    assert empties > 0


def test_tiled_dataset_derives_sliver_frac_with_enough_boxes(tmp_path):
    # With real spread and enough boxes (>= min_samples), the sliver cutoff is genuinely derived,
    # end to end through build_dataset, not just the pure function in test_derivations.py.
    pytest.importorskip("torch")
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.derivations import derive_sliver_frac

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (256, 256), (120, 120, 120)).save(images_dir / "a.jpg")
    # 6 boxes with real size spread (10..90px char size), all near the top-left so tiling geometry
    # is not the point of this test.
    sizes = [10.0, 26.0, 42.0, 58.0, 74.0, 90.0]
    anns = [Annotation(subject="bud", geometry=BBox(2.0, 2.0 + 5.0 * i, 2.0 + s, 2.0 + 5.0 * i + s))
            for i, s in enumerate(sizes)]
    json_io.write_annotations(str(labels_dir / "a.json"), anns, 256, 256, keep_empty=True)
    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                       subject="bud", tiling={"enabled": True, "tile_size": 64, "overlap": 0.2})
    assert ds.sliver_frac_source == "GT characteristic-size spread (p10 / mean)"
    assert ds.sliver_frac == pytest.approx(derive_sliver_frac(sizes))
    assert ds.sliver_frac != 0.5  # genuinely derived, not the pinned constant it replaces


def test_tiled_dataset_collate_roundtrip(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torch.utils.data import DataLoader
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.training.collation import task_collate

    images_dir, labels_dir = _det_dataset(tmp_path)
    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                       subject="bud", tiling={"enabled": True, "tile_size": 64, "overlap": 0.2})
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("detection"))
    imgs, targets = next(iter(loader))
    assert isinstance(imgs, list) and isinstance(targets, list)
    assert len(imgs) == len(targets)
    for t in targets:
        assert t["boxes"].shape[0] == t["labels"].shape[0]


def test_build_dataset_no_tiling_unchanged(tmp_path):
    pytest.importorskip("torch")
    import csv
    from PIL import Image
    from tcip_mcp.pipelines.data.datasets import build_dataset, DetectionDataset

    images_dir, labels_dir = _det_dataset(tmp_path, n=3, size=64)
    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir), subject="bud")
    assert isinstance(ds, DetectionDataset)
    assert len(ds) == 3  # no tiling -> one sample per image

    # tiling passed to a non-detection task is ignored (no raise).
    cls_dir = tmp_path / "cls"
    cls_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(2):
        Image.new("RGB", (32, 32), (100, 100, 100)).save(cls_dir / f"c{i}.png")
        rows.append((f"c{i}", i % 2))
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("stem", "label"))
        w.writerows(rows)
    ds2 = build_dataset("classification", images_dir=str(cls_dir), csv_path=str(csv_path),
                        num_classes=2, tiling={"enabled": True})
    assert ds2.num_samples == 2  # plain classification dataset


# TiledDetectionDataset keep_regions
# --------------------------------------------------------------------------

def test_tile_within_extent_and_rect_contains_tile():
    assert tiling.tile_within_extent(0, 0, 64, 100, 100) is True
    assert tiling.tile_within_extent(50, 0, 64, 100, 100) is False  # 50+64 > 100
    assert tiling.rect_contains_tile((0, 0, 128, 128), 0, 0, 64) is True
    assert tiling.rect_contains_tile((0, 0, 128, 128), 100, 0, 64) is False  # 100+64 > 128


def test_keep_regions_none_is_byte_identical_to_before(tmp_path):
    """The default (no keep_regions) builds the exact same index as before this parameter
    existed: every current caller of TiledDetectionDataset omits it."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.datasets import DetectionDataset, TiledDetectionDataset

    images_dir, labels_dir = _det_dataset(tmp_path, n=1, size=256)
    base = DetectionDataset(str(images_dir), str(labels_dir), subject="bud")
    plain = TiledDetectionDataset(base, tile_size=64, overlap=0.2)
    explicit_none = TiledDetectionDataset(base, tile_size=64, overlap=0.2, keep_regions=None)
    assert plain.tile_entries == explicit_none.tile_entries
    assert plain.tiles_dropped_past_extent == explicit_none.tiles_dropped_past_extent == 0
    assert plain.tiles_dropped_outside_regions == explicit_none.tiles_dropped_outside_regions == 0


def test_keep_regions_restricts_to_fully_inside_tiles(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.datasets import DetectionDataset, TiledDetectionDataset

    images_dir, labels_dir = _det_dataset(tmp_path, n=1, size=256)
    base = DetectionDataset(str(images_dir), str(labels_dir), subject="bud")
    full = TiledDetectionDataset(base, tile_size=64, overlap=0.2)
    left_half = TiledDetectionDataset(base, tile_size=64, overlap=0.2, keep_regions=[(0, 0, 128, 256)])

    assert 0 < left_half.num_samples < full.num_samples
    for _stem, tx, ty in left_half.tile_entries:
        assert tx + 64 <= 128  # fully inside the left-half region, never straddling it
    assert left_half.tiles_dropped_outside_regions == full.num_samples - left_half.num_samples
    assert left_half.tiles_dropped_past_extent == 0  # every tile in a 256x256 image is in-extent


def test_keep_regions_two_views_share_one_base_and_partition_disjointly(tmp_path):
    """The mechanism a spatial train/val split uses: two TiledDetectionDataset instances over one
    shared base, complementary keep_regions, no tile common to both."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.datasets import DetectionDataset, TiledDetectionDataset

    images_dir, labels_dir = _det_dataset(tmp_path, n=1, size=256)
    base = DetectionDataset(str(images_dir), str(labels_dir), subject="bud")
    left = TiledDetectionDataset(base, tile_size=64, overlap=0.2, keep_regions=[(0, 0, 128, 256)])
    right = TiledDetectionDataset(base, tile_size=64, overlap=0.2, keep_regions=[(128, 0, 256, 256)])

    assert left.num_samples > 0 and right.num_samples > 0
    assert set(left.tile_entries).isdisjoint(set(right.tile_entries))
    assert left.base is base and right.base is base  # one shared base, not two separate builds
