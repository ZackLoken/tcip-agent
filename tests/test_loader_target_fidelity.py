"""What a loader hands the trainer for one sample.

These tests cover its box extents, the tile index's boxes and the fields that identify the
sample, over asymmetric fixtures: boxes wider than tall and taller than wide, a non-square frame,
and sparse ordinal ranks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

torch = pytest.importorskip("torch")

from tcip_annotation.state import Annotation, BBox  # noqa: E402
from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir  # noqa: E402
from tests._producer_fixtures import (  # noqa: E402
    dataset_over,
    label_image,
    run_over,
    write_image,
)

BUD = "bud"


def _make_images(
    images_dir: Path, stems: list[str], size: tuple[int, int]
) -> None:
    for stem in stems:
        write_image(images_dir / f"{stem}.jpg", size)


def _write(
    images_dir: Path,
    stem: str,
    boxes: list[tuple[int, int, int, int]],
    size: tuple[int, int],
) -> None:
    label_image(
        images_dir / f"{stem}.jpg",
        [Annotation(subject=BUD, geometry=BBox(*b)) for b in boxes],
        size[0],
        size[1],
    )


def test_a_boxs_width_and_height_reach_the_loader_unswapped(
    tmp_path: Path,
) -> None:
    """A box's width and height reach the loader unswapped.

    One box wider than tall and one taller than wide reach the target as the pixel xyxy the
    document states.
    """
    size = (120, 80)
    images = image_dir(tmp_path, UNDATED_BUCKET)
    _make_images(images, ["img0"], size)
    _write(images, "img0", [(10, 20, 70, 40), (80, 5, 95, 75)], size)

    ds = dataset_over("detection", str(images), subject=BUD)
    _img, target = ds[0]
    assert target["boxes"].tolist() == [
        [10.0, 20.0, 70.0, 40.0],
        [80.0, 5.0, 95.0, 75.0],
    ]


# A 320x256 frame sliced at 64 with no overlap is an exact lattice (columns 0/64/128/192/256,
# rows 0/64/128/192); two boxes cross a seam by two pixels, the rest sit inside one tile each.
SEAM_BOXES = [
    (
        10,
        20,
        66,
        47,
    ),  # 56x27, crosses the x=64 seam, leaving a 2x27 stub in the next column
    (70, 10, 110, 34),  # 40x24, inside column 64
    (150, 100, 190, 127),  # 40x27, inside column 128, row 64
    (200, 20, 236, 60),  # 36x40, inside column 192, row 0
    (30, 140, 60, 190),  # 30x50, inside column 0, row 128
    (
        200,
        126,
        244,
        156,
    ),  # 44x30, crosses the y=128 seam, leaving a 44x2 stub in the row above
]


@pytest.mark.parametrize("sliver_frac", [None, 0.5])
def test_a_seam_fragment_is_not_indexed_as_a_whole_object(
    tmp_path: Path, sliver_frac: float | None
) -> None:
    """A seam fragment is not indexed as a whole object.

    A box a tile boundary cuts down to a two-pixel stub is indexed in no tile, whether the
    sliver cutoff is derived from the dataset or supplied, and the tile holding the rest of
    each box keeps it.
    """
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset

    size = (320, 256)
    images = image_dir(tmp_path, UNDATED_BUCKET)
    _make_images(images, ["img0"], size)
    _write(images, "img0", SEAM_BOXES, size)

    base = dataset_over("detection", str(images), subject=BUD)
    ds = TiledDetectionDataset(
        base,
        tile_size=64,
        overlap=0.0,
        sliver_frac=sliver_frac,
        skip_empty=False,
    )

    per_tile = {e["slice"][:2]: len(e["boxes"]) for e in ds._index}
    assert len(per_tile) == 20, (
        "five slice columns by four rows over the 320x256 frame"
    )
    assert per_tile[(64, 0)] == 1, (
        "the stub beside the neighboring box was indexed as an object"
    )
    assert per_tile[(192, 64)] == 0, (
        "the stub above the seam was indexed as an object"
    )
    assert per_tile[(0, 0)] == 1, "the clipped box's own tile lost it"
    assert per_tile[(192, 128)] == 1, "the clipped box's own tile lost it"
    assert sum(per_tile.values()) == len(SEAM_BOXES)


def test_each_detection_sample_carries_its_own_index_as_image_id(
    tmp_path: Path,
) -> None:
    """Each detection sample carries its own index as its image id, beside its own boxes."""
    size = (120, 80)
    images = image_dir(tmp_path, UNDATED_BUCKET)
    stems = ["a", "b", "c"]
    _make_images(images, stems, size)
    for i, stem in enumerate(stems):
        _write(
            images,
            stem,
            [(5 + 20 * k, 6, 25 + 20 * k, 44) for k in range(i + 1)],
            size,
        )

    ds = dataset_over("detection", str(images), subject=BUD)
    assert len(ds) == 3
    samples = [ds[i][1] for i in range(len(ds))]
    assert [t["image_id"] for t in samples] == [0, 1, 2]
    assert [t["boxes"].shape[0] for t in samples] == [1, 2, 3]


def test_an_ordinal_sample_carries_the_rank_count_its_head_decodes_with(
    tmp_path: Path,
) -> None:
    """An ordinal run's rank count is the scale's reach, not the number of samples.

    Over three samples ranked 0, 2 and 5, the run's rank count is six and each sample carries
    its own rank.
    """
    images = image_dir(tmp_path, UNDATED_BUCKET)
    _make_images(images, ["s0", "s1", "s2"], (48, 32))
    csv_path = tmp_path / "ranks.csv"
    csv_path.write_text("stem,rank\ns0,0\ns1,2\ns2,5\n", encoding="utf-8")

    ds, data = run_over("ordinal", str(images), str(csv_path))
    assert len(ds) == 3
    assert (
        data["num_ranks"] == 6
    )  # the run's own rank count, once, off the table it was handed
    for idx, expected_rank in enumerate([0, 2, 5]):
        _img, target = ds[idx]
        assert target["ranks"] == expected_rank
