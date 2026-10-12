"""TiledDetectionDataset: SAHI's lattice over each image, each row clipped once to each slice.

The clip tests need no torch. The wrapper tests skip if torch is absent (they go through
``build_dataset``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tcip_annotation.matching import geometry_rings, shapely_geometry
from tcip_annotation.state import BBox, Polygon
from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tcip_mcp.pipelines.data.datasets import clip_rows_to_tile
from tests._producer_fixtures import dataset_over  # noqa: E402
from tests._training_values import tiling_block

SLIVER = 0.5
"""The sliver cutoff the wrapper tests state: their one-box fixtures derive no size spread."""
TRIANGLE = Polygon(rings=[[(4.0, 4.0), (124.0, 60.0), (4.0, 60.0)]])
"""A region no box stands in for, crossing the seam of a frame two 64 px tiles wide: within the
second tile its diagonal edge starts at y=32, where its bounding box starts at y=4."""


def _clipped(geometries, tile, min_box_size):
    """``geometries`` (each a box or a polygon) clipped to ``tile`` (:func:`clip_rows_to_tile`)."""
    boxes = np.asarray(
        [
            [g.x1, g.y1, g.x2, g.y2]
            if isinstance(g, BBox)
            else [
                min(x for r in g.rings for x, _ in r),
                min(y for r in g.rings for _, y in r),
                max(x for r in g.rings for x, _ in r),
                max(y for r in g.rings for _, y in r),
            ]
            for g in geometries
        ]
    )
    return clip_rows_to_tile(
        [shapely_geometry(geometry_rings(g)) for g in geometries],
        boxes,
        np.arange(len(geometries)),
        tile,
        min_box_size,
    )


def test_a_row_is_clipped_once_and_carries_the_bounds_of_what_the_tile_shows():
    """min_box_size 12 on the tile (200, 200, 264, 264): a row inside the tile is kept, remapped
    to tile-local; one crossing the edge with a 14 px visible part is kept and one with a 9 px
    sliver is dropped; one not meeting the tile is gone. A polygon's box is the bounds of its
    clipped region, never its bounding box cut at the tile, and its geometry is that region,
    tile-local.
    """
    tile = (200, 200, 264, 264)
    boxes, rows, regions = _clipped(
        [
            BBox(210, 210, 230, 230),
            BBox(250, 250, 290, 290),
            BBox(255, 255, 265, 265),
            BBox(0, 0, 10, 10),
        ],
        tile,
        12.0,
    )
    assert rows.tolist() == [0, 1]
    assert np.allclose(boxes, [[10, 10, 30, 30], [50, 50, 64, 64]])
    assert sorted(regions[1].rings[0]) == [
        (50.0, 50.0),
        (50.0, 64.0),
        (64.0, 50.0),
        (64.0, 64.0),
    ]

    boxes, rows, regions = _clipped([TRIANGLE], (64, 0, 128, 64), 0.0)
    assert rows.tolist() == [0]
    assert np.allclose(boxes, [[0.0, 32.0, 60.0, 60.0]])
    assert sorted(regions[0].rings[0]) == [
        (0.0, 32.0),
        (0.0, 60.0),
        (60.0, 60.0),
    ]


# --------------------------------------------------------------------------
# TiledDetectionDataset wrapper (needs torch)
# --------------------------------------------------------------------------


def _det_dataset(tmp_path: Path, n: int = 1, size: int = 128) -> Path:
    """``n`` square ``size``px images, each holding one centered ``bud`` box a tenth as wide."""
    from tests._producer_fixtures import seed_bud_images

    return seed_bud_images(
        tmp_path / "images" / UNDATED_BUCKET,
        n=n,
        size=size,
        box=(0.45 * size, 0.45 * size, 0.55 * size, 0.55 * size),
    )


def test_a_dataset_too_sparse_to_derive_its_sliver_cutoff_refuses_naming_the_field(
    tmp_path,
):
    """No fraction stands in for a cutoff the ground truth cannot derive: one box per image over
    two images measures no size spread, so an unstated cutoff refuses naming what to state, and
    the same dataset with the field stated tiles.
    """
    pytest.importorskip("torch")

    images_dir = _det_dataset(tmp_path, n=2)
    with pytest.raises(ValueError, match="tiling.sliver_frac"):
        dataset_over(
            "detection",
            str(images_dir),
            subject="bud",
            tiling=tiling_block(tile_size=64, overlap=0.2),
        )
    stated = dataset_over(
        "detection",
        str(images_dir),
        subject="bud",
        tiling={
            **tiling_block(),
            "tile_size": 64,
            "overlap": 0.2,
            "sliver_frac": SLIVER,
        },
    )
    assert len(stated) > 0


def test_tiled_detection_dataset_wrapper(tmp_path):
    torch = pytest.importorskip("torch")

    images_dir = _det_dataset(tmp_path)
    ds = dataset_over(
        "detection",
        str(images_dir),
        subject="bud",
        tiling={
            **tiling_block(),
            "tile_size": 64,
            "overlap": 0.2,
            "sliver_frac": SLIVER,
        },
    )
    assert len(ds) >= 1  # 128px image -> multiple tiles
    img, target = ds[0]
    assert tuple(img.shape) == (3, 64, 64)
    assert target["boxes"].shape[1] == 4
    assert (target["boxes"] >= 0).all() and (target["boxes"] <= 64).all()
    assert target["labels"].dtype == torch.int64


def test_detection_and_instance_seg_tiles_of_one_frame_agree(tmp_path):
    """One frame of polygons, several crossing a seam and one of them a triangle no box stands in
    for, tiled at one lattice by the detection loader and the instance-segmentation loader: the
    two agree on every tile's rect, boxes and labels, every instance mask is the untiled
    instance-segmentation loader's frame mask over its tile's rect up to the rasterizer's
    rounding at the region's edge, and every box is the bounds of its mask's pixels, so a box
    cut from the unclipped region, or a mask cut at another offset than its box, disagrees.
    """
    torch = pytest.importorskip("torch")
    from PIL import Image
    from shapely.geometry import Point as ShapelyPoint

    from tcip_annotation.state import Annotation
    from tests._producer_fixtures import label_image

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    Image.new("RGB", (200, 200), (120, 120, 120)).save(images_dir / "a.png")
    regions = [
        Polygon(rings=[[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]])
        for x0, y0, x1, y1 in [
            (40, 40, 80, 80),
            (100, 30, 150, 70),
            (20, 120, 60, 170),
        ]
    ]
    regions.append(
        Polygon(rings=[[(100.0, 100.0), (196.0, 190.0), (100.0, 190.0)]])
    )
    label_image(
        images_dir / "a.png",
        [Annotation(subject="bud", geometry=region) for region in regions],
        200,
        200,
    )
    edges = [shapely_geometry(region.rings).boundary for region in regions]
    # A tenth of the objects' size keeps the triangle's corner pieces, whose boxes are the point.
    tiling = {
        **tiling_block(),
        "tile_size": 64,
        "overlap": 0.25,
        "sliver_frac": 0.1,
    }
    detection = dataset_over(
        "detection", str(images_dir), subject="bud", tiling=tiling
    )
    instances = dataset_over(
        "instance_seg", str(images_dir), subject="bud", tiling=tiling
    )
    frame = dataset_over("instance_seg", str(images_dir), subject="bud")[0][1][
        "masks"
    ]

    assert detection.tile_entries == instances.tile_entries
    straddled = 0
    for index, (_stem, (x0, y0, x1, y1)) in enumerate(instances.tile_entries):
        boxes, masked = detection[index][1], instances[index][1]
        assert torch.equal(boxes["boxes"], masked["boxes"])
        assert torch.equal(boxes["labels"], masked["labels"])
        assert masked["masks"].shape == (
            len(masked["boxes"]),
            y1 - y0,
            x1 - x0,
        )
        cropped = frame[:, y0:y1, x0:x1]
        for mask, box in zip(masked["masks"], masked["boxes"].tolist()):
            # The frame's raster of the same region: the two differ only where the rasterizer
            # rounds the region's edge, never a pixel farther from it.
            which = max(
                range(len(cropped)),
                key=lambda i: int((mask & cropped[i]).sum()),
            )
            assert all(
                edges[which].distance(ShapelyPoint(x0 + x + 0.5, y0 + y + 0.5))
                <= 1.0
                for y, x in torch.nonzero(mask != cropped[which]).tolist()
            )
            ys, xs = torch.nonzero(mask, as_tuple=True)
            pixels = [
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1,
            ]
            assert all(abs(p - b) <= 1 for p, b in zip(pixels, box)), (
                box,
                pixels,
            )
        straddled += int(
            any(
                0 < int(c.sum()) < int(f.sum()) for c, f in zip(cropped, frame)
            )
        )
    assert straddled > 0


def test_tiled_dataset_keeps_empty_tiles(tmp_path):
    pytest.importorskip("torch")
    from PIL import Image

    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import label_image

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    Image.new("RGB", (256, 256), (120, 120, 120)).save(images_dir / "a.jpg")
    # YOLO "0 0.1 0.1 0.1 0.1" in a 256×256 image -> a 25.6px box in the top-left corner
    label_image(
        images_dir / "a.jpg",
        [Annotation(subject="bud", geometry=BBox(12.8, 12.8, 38.4, 38.4))],
        256,
        256,
        keep_empty=True,
    )
    ds = dataset_over(
        "detection",
        str(images_dir),
        subject="bud",
        tiling={
            **tiling_block(),
            "tile_size": 64,
            "overlap": 0.2,
            "sliver_frac": SLIVER,
        },
    )
    # Tiles far from the object are kept as valid negatives.
    empties = sum(1 for i in range(len(ds)) if ds[i][1]["boxes"].shape[0] == 0)
    assert empties > 0


def test_tiled_dataset_collate_roundtrip(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.training.collation import task_collate

    images_dir = _det_dataset(tmp_path)
    ds = dataset_over(
        "detection",
        str(images_dir),
        subject="bud",
        tiling={
            **tiling_block(),
            "tile_size": 64,
            "overlap": 0.2,
            "sliver_frac": SLIVER,
        },
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("detection"))
    imgs, targets = next(iter(loader))
    assert isinstance(imgs, list) and isinstance(targets, list)
    assert len(imgs) == len(targets)
    for t in targets:
        assert t["boxes"].shape[0] == t["labels"].shape[0]


def test_build_dataset_no_tiling_unchanged(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.datasets import DetectionDataset
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir = _det_dataset(tmp_path, n=3, size=64)
    ds = dataset_over("detection", str(images_dir), subject="bud")
    assert isinstance(ds, DetectionDataset)
    assert len(ds) == 3  # no tiling -> one sample per image

    # An enabled tiling handed to a task that trains whole frames refuses by name; untiled, the
    # same samples build the plain classification dataset.
    cls_dir, csv_path = write_regression_dataset(
        tmp_path / "cls", [0.4, 0.4], [0, 1], height=32, width=32
    )
    with pytest.raises(
        ValueError, match="data.tiling is stated for a run of task 'classific"
    ):
        dataset_over(
            "classification",
            str(cls_dir),
            str(csv_path),
            tiling=tiling_block(),
        )
    ds2 = dataset_over("classification", str(cls_dir), str(csv_path))
    assert ds2.num_samples == 2  # plain classification dataset


# TiledDetectionDataset keep_regions
# --------------------------------------------------------------------------


def test_keep_regions_none_indexes_every_slice(tmp_path):
    """No keep_regions and keep_regions=None build the same index, every slice of the lattice."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset

    images_dir = _det_dataset(tmp_path, n=1, size=256)
    base = dataset_over("detection", str(images_dir), subject="bud")
    plain = TiledDetectionDataset(
        base, tile_size=64, overlap=0.2, sliver_frac=SLIVER, skip_empty=False
    )
    explicit_none = TiledDetectionDataset(
        base,
        tile_size=64,
        overlap=0.2,
        sliver_frac=SLIVER,
        skip_empty=False,
        keep_regions=None,
    )
    from tcip_mcp.pipelines.slicing import slice_lattice

    assert plain.tile_entries == explicit_none.tile_entries
    assert [box for _s, box in plain.tile_entries] == slice_lattice(
        256, 256, 64, 0.2
    )
    assert (
        plain.tiles_dropped_past_extent
        == explicit_none.tiles_dropped_past_extent
        == 0
    )
    assert (
        plain.tiles_dropped_outside_regions
        == explicit_none.tiles_dropped_outside_regions
        == 0
    )


def test_keep_regions_restricts_to_fully_inside_tiles(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset

    images_dir = _det_dataset(tmp_path, n=1, size=256)
    base = dataset_over("detection", str(images_dir), subject="bud")
    full = TiledDetectionDataset(
        base, tile_size=64, overlap=0.2, sliver_frac=SLIVER, skip_empty=False
    )
    left_half = TiledDetectionDataset(
        base,
        tile_size=64,
        overlap=0.2,
        sliver_frac=SLIVER,
        skip_empty=False,
        keep_regions=[(0, 0, 128, 256)],
    )

    assert 0 < left_half.num_samples < full.num_samples
    for _stem, box in left_half.tile_entries:
        assert (
            box[2] <= 128
        )  # fully inside the left-half region, never straddling it
    assert (
        left_half.tiles_dropped_outside_regions
        == full.num_samples - left_half.num_samples
    )
    assert (
        left_half.tiles_dropped_past_extent == 0
    )  # every slice of a 256x256 image is full


def test_keep_regions_two_views_share_one_base_and_partition_disjointly(
    tmp_path,
):
    """The mechanism a spatial train/val split uses: two TiledDetectionDataset instances over one
    shared base, complementary keep_regions, no tile common to both.
    """
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset

    images_dir = _det_dataset(tmp_path, n=1, size=256)
    base = dataset_over("detection", str(images_dir), subject="bud")
    left = TiledDetectionDataset(
        base,
        tile_size=64,
        overlap=0.2,
        sliver_frac=SLIVER,
        skip_empty=False,
        keep_regions=[(0, 0, 128, 256)],
    )
    right = TiledDetectionDataset(
        base,
        tile_size=64,
        overlap=0.2,
        sliver_frac=SLIVER,
        skip_empty=False,
        keep_regions=[(128, 0, 256, 256)],
    )

    assert left.num_samples > 0 and right.num_samples > 0
    assert set(left.tile_entries).isdisjoint(set(right.tile_entries))
