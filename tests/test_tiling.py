"""TiledDetectionDataset: SAHI's lattice over each image, labels clipped to each slice.

The label tests are pure numpy (no torch). The wrapper tests skip if torch is absent (they go
through ``build_dataset``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tcip_mcp.pipelines.data.datasets import clip_boxes_to_tile, dedup_boxes
from tests._producer_fixtures import dataset_over  # noqa: E402

SLIVER = 0.5
"""The sliver cutoff the wrapper tests state: their one-box fixtures derive no size spread."""


def test_clip_boxes_to_tile_sliver_drop_and_remap():
    # min_box_size=12: a clipped box counts unless its visible part is a sliver (< 12px char-size).
    # Fully-inside box -> always kept, remapped to tile-local (minus origin 200,200).
    tb, tl = clip_boxes_to_tile(np.array([[210., 210., 230., 230.]]), np.array([1]), 200, 200, 64, 64, 12.0)
    assert tb.shape == (1, 4)
    assert np.allclose(tb[0], [10, 10, 30, 30])
    # Straddling box whose visible part is substantial (clipped to 14x14, char 14 >= 12) -> kept.
    tb2, _ = clip_boxes_to_tile(np.array([[250., 250., 290., 290.]]), np.array([1]), 200, 200, 64, 64, 12.0)
    assert len(tb2) == 1
    # Straddling box whose visible part is a sliver (clipped to 9x9, char 9 < 12) -> dropped.
    tb3, _ = clip_boxes_to_tile(np.array([[255., 255., 265., 265.]]), np.array([1]), 200, 200, 64, 64, 12.0)
    assert len(tb3) == 0
    # Non-overlapping box -> no output.
    tb4, _ = clip_boxes_to_tile(np.array([[0., 0., 10., 10.]]), np.array([1]), 200, 200, 64, 64, 12.0)
    assert len(tb4) == 0


def test_dedup_boxes_class_aware():
    boxes = np.array([[0., 0., 20., 20.], [1., 1., 19., 19.]])  # IoU 0.81, same label
    assert dedup_boxes(boxes, np.array([1, 1]), 0.8) == [0]  # larger kept
    distinct = np.array([[0., 0., 10., 10.], [50., 50., 60., 60.]])
    assert dedup_boxes(distinct, np.array([1, 1]), 0.8) == [0, 1]  # both survive
    # same geometry, different labels -> both survive
    assert dedup_boxes(boxes, np.array([1, 2]), 0.8) == [0, 1]
    assert dedup_boxes(boxes, np.array([1, 1]), 1.0) == [0, 1]  # iou_thresh >= 1.0: no-op


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


def test_a_dataset_too_sparse_to_derive_its_sliver_cutoff_refuses_naming_the_field(tmp_path):
    """No fraction stands in for a cutoff the ground truth cannot derive: one box per image over
    two images measures no size spread, so an unstated cutoff refuses naming what to state, and
    the same dataset with the field stated tiles."""
    pytest.importorskip("torch")

    images_dir, labels_dir = _det_dataset(tmp_path, n=2)
    with pytest.raises(ValueError, match="tiling.sliver_frac"):
        dataset_over("detection", str(images_dir), str(labels_dir), subject="bud",
                     tiling={"enabled": True, "tile_size": 64, "overlap": 0.2})
    stated = dataset_over("detection", str(images_dir), str(labels_dir), subject="bud",
                          tiling={"enabled": True, "tile_size": 64, "overlap": 0.2,
                                  "sliver_frac": SLIVER})
    assert len(stated) > 0


def test_tiled_detection_dataset_wrapper(tmp_path):
    torch = pytest.importorskip("torch")

    images_dir, labels_dir = _det_dataset(tmp_path)
    ds = dataset_over("detection", str(images_dir), str(labels_dir), subject="bud", tiling={"enabled": True, "tile_size": 64, "overlap": 0.2, "sliver_frac": SLIVER})
    assert len(ds) >= 1  # 128px image -> multiple tiles
    img, target = ds[0]
    assert tuple(img.shape) == (3, 64, 64)
    assert target["boxes"].shape[1] == 4
    assert (target["boxes"] >= 0).all() and (target["boxes"] <= 64).all()
    assert target["labels"].dtype == torch.int64


def test_tiled_dataset_keeps_empty_tiles(tmp_path):
    pytest.importorskip("torch")
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (256, 256), (120, 120, 120)).save(images_dir / "a.jpg")
    # YOLO "0 0.1 0.1 0.1 0.1" in a 256×256 image -> a 25.6px box in the top-left corner
    json_io.write_annotations(str(labels_dir / "a.json"),
                              [Annotation(subject="bud", geometry=BBox(12.8, 12.8, 38.4, 38.4))],
                              256, 256, keep_empty=True)
    ds = dataset_over("detection", str(images_dir), str(labels_dir), subject="bud", tiling={"enabled": True, "tile_size": 64, "overlap": 0.2, "sliver_frac": SLIVER})
    # Tiles far from the object are kept as valid negatives.
    empties = sum(1 for i in range(len(ds)) if ds[i][1]["boxes"].shape[0] == 0)
    assert empties > 0


def test_tiled_dataset_collate_roundtrip(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torch.utils.data import DataLoader
    from tcip_mcp.pipelines.training.collation import task_collate

    images_dir, labels_dir = _det_dataset(tmp_path)
    ds = dataset_over("detection", str(images_dir), str(labels_dir), subject="bud", tiling={"enabled": True, "tile_size": 64, "overlap": 0.2, "sliver_frac": SLIVER})
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
    from tcip_mcp.pipelines.data.datasets import DetectionDataset

    images_dir, labels_dir = _det_dataset(tmp_path, n=3, size=64)
    ds = dataset_over("detection", str(images_dir), str(labels_dir), subject="bud")
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
    ds2 = dataset_over("classification", str(cls_dir), str(csv_path), tiling={"enabled": True})
    assert ds2.num_samples == 2  # plain classification dataset


# TiledDetectionDataset keep_regions
# --------------------------------------------------------------------------

def test_keep_regions_none_indexes_every_slice(tmp_path):
    """No keep_regions and keep_regions=None build the same index, every slice of the lattice."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset

    images_dir, labels_dir = _det_dataset(tmp_path, n=1, size=256)
    base = dataset_over('detection', str(images_dir), str(labels_dir), subject="bud")
    plain = TiledDetectionDataset(base, tile_size=64, overlap=0.2, sliver_frac=SLIVER)
    explicit_none = TiledDetectionDataset(base, tile_size=64, overlap=0.2, sliver_frac=SLIVER, keep_regions=None)
    from tcip_mcp.pipelines.slicing import slice_lattice

    assert plain.tile_entries == explicit_none.tile_entries
    assert [box for _s, box in plain.tile_entries] == slice_lattice(256, 256, 64, 0.2)
    assert plain.tiles_dropped_past_extent == explicit_none.tiles_dropped_past_extent == 0
    assert plain.tiles_dropped_outside_regions == explicit_none.tiles_dropped_outside_regions == 0


def test_keep_regions_restricts_to_fully_inside_tiles(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset

    images_dir, labels_dir = _det_dataset(tmp_path, n=1, size=256)
    base = dataset_over('detection', str(images_dir), str(labels_dir), subject="bud")
    full = TiledDetectionDataset(base, tile_size=64, overlap=0.2, sliver_frac=SLIVER)
    left_half = TiledDetectionDataset(base, tile_size=64, overlap=0.2, sliver_frac=SLIVER, keep_regions=[(0, 0, 128, 256)])

    assert 0 < left_half.num_samples < full.num_samples
    for _stem, box in left_half.tile_entries:
        assert box[2] <= 128  # fully inside the left-half region, never straddling it
    assert left_half.tiles_dropped_outside_regions == full.num_samples - left_half.num_samples
    assert left_half.tiles_dropped_past_extent == 0  # every slice of a 256x256 image is full


def test_keep_regions_two_views_share_one_base_and_partition_disjointly(tmp_path):
    """The mechanism a spatial train/val split uses: two TiledDetectionDataset instances over one
    shared base, complementary keep_regions, no tile common to both."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset

    images_dir, labels_dir = _det_dataset(tmp_path, n=1, size=256)
    base = dataset_over('detection', str(images_dir), str(labels_dir), subject="bud")
    left = TiledDetectionDataset(base, tile_size=64, overlap=0.2, sliver_frac=SLIVER, keep_regions=[(0, 0, 128, 256)])
    right = TiledDetectionDataset(base, tile_size=64, overlap=0.2, sliver_frac=SLIVER, keep_regions=[(128, 0, 256, 256)])

    assert left.num_samples > 0 and right.num_samples > 0
    assert set(left.tile_entries).isdisjoint(set(right.tile_entries))
