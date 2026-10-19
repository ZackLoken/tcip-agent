"""Channel-generic tiled reads: multi-band tiles carry their boxes over bright pixels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tests._producer_fixtures import dataset_over, label_image, painted_array  # noqa: E402
from tests._training_values import tiling_block

if TYPE_CHECKING:
    from pathlib import Path


def _patched(
    width: int = 40,
    height: int = 24,
    bands: int = 5,
    patch: tuple[int, int] = (28, 12),
) -> np.ndarray:
    """Return ``bands`` copies, stacked last, of one ``painted_array`` band.

    The band is ``width`` by ``height``, painted 255 over the 6 by 6 rectangle whose corner is
    ``patch``.
    """
    px, py = patch
    band = painted_array(width, height, [((px, py, px + 6, py + 6), 255)])
    return np.stack([band] * bands, axis=-1)


def _multiband_detection_fixture(
    tmp_path: Path,
    width: int = 40,
    height: int = 24,
    bands: int = 5,
    patch: tuple[int, int] = (28, 12),
) -> Path:
    """Write :func:`_patched`'s array as ``a.tif`` with one ``bud`` box over the patch rectangle.

    The TIFF sits under ``images/<undated bucket>``, written by ``tifffile`` with no
    georeferencing tags. Returns its images directory.
    """
    import tifffile

    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    px, py = patch
    tifffile.imwrite(
        images_dir / "a.tif", _patched(width, height, bands, patch)
    )
    label_image(
        images_dir / "a.tif",
        [Annotation(subject="bud", geometry=BBox(px, py, px + 6, py + 6))],
        width,
        height,
        keep_empty=True,
    )
    return images_dir


def test_tiled_detection_reads_multiband_and_keeps_boxes_on_their_pixels(
    tmp_path: Path,
) -> None:
    """Tiles of a five-band raster keep five bands and each box over its bright patch.

    Each tile is 5 by 16 by 16, and each kept box's region of its tile holds a pixel above 0.5.
    """
    import torch

    images_dir = _multiband_detection_fixture(tmp_path)
    ds = dataset_over(
        "detection",
        str(images_dir),
        subject="bud",
        stated={"num_channels": 5},
        tiling={
            **tiling_block(),
            "tile_size": 16,
            "overlap": 0.0,
            "sliver_frac": 0.5,
        },
    )
    assert ds.expected_channels == 5

    hits = [ds[i] for i in range(len(ds))]
    assert hits, "tiled index is empty"
    assert all(t.shape == (5, 16, 16) for t, _ in hits)

    with_boxes = [(t, tgt) for t, tgt in hits if len(tgt["boxes"])]
    assert with_boxes, "the GT box did not survive tiling"
    for tile, target in with_boxes:
        for x1, y1, x2, y2 in target["boxes"].tolist():
            region = tile[
                :,
                int(y1) : int(torch.ceil(torch.tensor(y2))),
                int(x1) : int(torch.ceil(torch.tensor(x2))),
            ]
            assert region.numel() and float(region.max()) > 0.5, (
                "the box does not sit on the bright patch: tile and labels are in different frames"
            )


def test_tiled_dataset_refuses_labels_authored_in_a_different_frame(
    tmp_path: Path,
) -> None:
    """The tiled dataset refuses labels authored in a different frame, naming both frames.

    The label document is authored in a frame other than the one the raster decodes in (here
    the 5x40 PIL reports for a 40x24x5 TIFF).
    """
    import pytest
    import tifffile
    from PIL import Image

    from tcip_annotation.state import Annotation, BBox
    from tcip_annotation.utils import oriented_size

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    tifffile.imwrite(images_dir / "a.tif", _patched())

    with Image.open(images_dir / "a.tif") as opened:
        pil_w, pil_h = oriented_size(opened)
    assert (pil_w, pil_h) == (5, 40), (
        "fixture assumes PIL misreads this multi-band raster"
    )
    label_image(
        images_dir / "a.tif",
        [Annotation(subject="bud", geometry=BBox(1, 12, 4, 18))],
        pil_w,
        pil_h,
        keep_empty=True,
    )

    with pytest.raises(
        ValueError,
        match="the labels record a 5x40 image but it decodes as 40x24",
    ):
        dataset_over(
            "detection",
            str(images_dir),
            subject="bud",
            stated={"num_channels": 5},
            tiling={
                **tiling_block(),
                "tile_size": 16,
                "overlap": 0.0,
                "sliver_frac": 0.5,
            },
        )


def test_the_authored_frame_raises_on_a_corrupt_label_rather_than_reading_as_no_frame(
    tmp_path: Path,
) -> None:
    """A present, unreadable label raises when the tiled dataset is built.

    The error is ``UnreadableLabelDocumentError``.
    """
    import pytest

    from tcip_annotation.json_io import UnreadableLabelDocumentError
    from tests._producer_fixtures import image_label_key
    from tests._record_damage_fixtures import damage_record

    images_dir = _multiband_detection_fixture(tmp_path)
    damage_record(image_label_key(images_dir / "a.tif"), b"{not json")

    with pytest.raises(UnreadableLabelDocumentError):
        dataset_over(
            "detection",
            str(images_dir),
            subject="bud",
            stated={"num_channels": 5},
            tiling={
                **tiling_block(),
                "tile_size": 16,
                "overlap": 0.0,
                "sliver_frac": 0.5,
            },
        )


def test_ctx_tiled_dataset_inherits_the_band_count(tmp_path: Path) -> None:
    """A ``TiledDetectionDataset`` built over a five-band base expects five channels."""
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset

    images_dir = _multiband_detection_fixture(tmp_path)
    base = dataset_over(
        "detection", str(images_dir), subject="bud", stated={"num_channels": 5}
    )
    assert (
        TiledDetectionDataset(
            base, tile_size=16, overlap=0.0, sliver_frac=0.5, skip_empty=False
        ).expected_channels
        == 5
    )


def test_tiled_detection_handles_channel_first_rasters(tmp_path: Path) -> None:
    """Tiles of a channel-first five-band raster are 5 by 16 by 16, each box over a bright pixel."""
    import tifffile

    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    tifffile.imwrite(
        images_dir / "a.tif", np.transpose(_patched(), (2, 0, 1))
    )  # [C, H, W]
    label_image(
        images_dir / "a.tif",
        [Annotation(subject="bud", geometry=BBox(28, 12, 34, 18))],
        40,
        24,
        keep_empty=True,
    )

    ds = dataset_over(
        "detection",
        str(images_dir),
        subject="bud",
        stated={"num_channels": 5},
        tiling={
            **tiling_block(),
            "tile_size": 16,
            "overlap": 0.0,
            "sliver_frac": 0.5,
        },
    )
    with_boxes = [
        (t, tgt)
        for t, tgt in (ds[i] for i in range(len(ds)))
        if len(tgt["boxes"])
    ]
    assert with_boxes, (
        "the GT box did not survive tiling on a channel-first raster"
    )
    for tile, target in with_boxes:
        assert tile.shape == (5, 16, 16)
        for x1, y1, x2, y2 in target["boxes"].tolist():
            region = tile[:, int(y1) : int(y2) + 1, int(x1) : int(x2) + 1]
            assert region.numel() and float(region.max()) > 0.5
