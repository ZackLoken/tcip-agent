"""Channel-generic tiled training reads: multi-band slices keep their boxes on their pixels."""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

import numpy as np
from tests._producer_fixtures import dataset_over, label_image, painted_array  # noqa: E402


def _patched(width=40, height=24, bands=5, patch=(28, 12)) -> np.ndarray:
    """A black ``height`` by ``width`` array of ``bands`` bands, bright in every band over the
    6 px square at ``patch``."""
    px, py = patch
    band = painted_array(width, height, [((px, py, px + 6, py + 6), 255)])
    return np.stack([band] * bands, axis=-1)


def _multiband_detection_fixture(tmp_path, width=40, height=24, bands=5, patch=(28, 12)):
    """A non-square multi-band GeoTIFF with a bright patch, and a GT box on that patch; its
    images directory."""
    import tifffile
    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    px, py = patch
    tifffile.imwrite(images_dir / "a.tif", _patched(width, height, bands, patch))
    label_image(images_dir / "a.tif",
                [Annotation(subject="bud", geometry=BBox(px, py, px + 6, py + 6))],
                width, height, keep_empty=True)
    return images_dir


def test_tiled_detection_reads_multiband_and_keeps_boxes_on_their_pixels(tmp_path):
    """Channel count is not enough: a frame mismatch passes a shape check but displaces every box.

    The bright patch and the GT box are placed together, so this fails if the tile is cut from a
    differently-oriented frame than the labels were clipped against.
    """
    import torch

    images_dir = _multiband_detection_fixture(tmp_path)
    ds = dataset_over("detection", str(images_dir), subject="bud",
                      stated={"num_channels": 5},
                      tiling={"enabled": True, "tile_size": 16, "overlap": 0.0, "sliver_frac": 0.5})
    assert ds.expected_channels == 5

    hits = [ds[i] for i in range(len(ds))]
    assert hits, "tiled index is empty"
    assert all(t.shape == (5, 16, 16) for t, _ in hits)

    with_boxes = [(t, tgt) for t, tgt in hits if len(tgt["boxes"])]
    assert with_boxes, "the GT box did not survive tiling"
    for tile, target in with_boxes:
        for x1, y1, x2, y2 in target["boxes"].tolist():
            region = tile[:, int(y1):int(torch.ceil(torch.tensor(y2))),
                          int(x1):int(torch.ceil(torch.tensor(x2)))]
            assert region.numel() and float(region.max()) > 0.5, (
                "the box does not sit on the bright patch: tile and labels are in different frames")


def test_tiled_dataset_refuses_labels_authored_in_a_different_frame(tmp_path):
    """A label document authored in a frame other than the one the raster decodes in (here
    the 5x40 PIL reports for a 40x24x5 TIFF) is refused by the tiled dataset, naming both
    frames."""
    import pytest
    import tifffile
    from tcip_annotation.state import Annotation, BBox
    from PIL import Image
    from tcip_annotation.utils import oriented_size

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    tifffile.imwrite(images_dir / "a.tif", _patched())

    with Image.open(images_dir / "a.tif") as opened:
        pil_w, pil_h = oriented_size(opened)
    assert (pil_w, pil_h) == (5, 40), "fixture assumes PIL misreads this multi-band raster"
    label_image(images_dir / "a.tif", [Annotation(subject="bud", geometry=BBox(1, 12, 4, 18))],
                pil_w, pil_h, keep_empty=True)

    with pytest.raises(ValueError, match="the labels record a 5x40 image but it decodes as 40x24"):
        dataset_over("detection", str(images_dir), subject="bud",
                     stated={"num_channels": 5},
                     tiling={"enabled": True, "tile_size": 16, "overlap": 0.0, "sliver_frac": 0.5})


def test_the_authored_frame_raises_on_a_corrupt_label_rather_than_reading_as_no_frame(tmp_path):
    """The tiled dataset reads each label's authored frame off its stored document, so a present,
    unreadable label raises rather than silently disabling the frame check for that sample."""
    import pytest
    from tcip_annotation.json_io import UnreadableLabelDocumentError

    from tests._producer_fixtures import image_label_key
    from tests._record_damage_fixtures import damage_record

    images_dir = _multiband_detection_fixture(tmp_path)
    damage_record(image_label_key(images_dir / "a.tif"), b"{not json")

    with pytest.raises(UnreadableLabelDocumentError):
        dataset_over("detection", str(images_dir), subject="bud",
                     stated={"num_channels": 5},
                     tiling={"enabled": True, "tile_size": 16, "overlap": 0.0, "sliver_frac": 0.5})


def test_ctx_tiled_dataset_inherits_the_band_count(tmp_path):
    """ctx.tiled_dataset constructs the tiler directly: it must not fall back to 3 channels."""
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset

    images_dir = _multiband_detection_fixture(tmp_path)
    base = dataset_over("detection", str(images_dir), subject="bud",
                        stated={"num_channels": 5})
    assert TiledDetectionDataset(base, tile_size=16, overlap=0.0,
                                 sliver_frac=0.5).expected_channels == 5


def test_tiled_detection_handles_channel_first_rasters(tmp_path):
    """The other common GeoTIFF layout, where the axis-order heuristic is observable."""
    import tifffile
    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    tifffile.imwrite(images_dir / "a.tif", np.transpose(_patched(), (2, 0, 1)))  # [C, H, W]
    label_image(images_dir / "a.tif", [Annotation(subject="bud", geometry=BBox(28, 12, 34, 18))],
                40, 24, keep_empty=True)

    ds = dataset_over("detection", str(images_dir), subject="bud",
                      stated={"num_channels": 5},
                      tiling={"enabled": True, "tile_size": 16, "overlap": 0.0, "sliver_frac": 0.5})
    with_boxes = [(t, tgt) for t, tgt in (ds[i] for i in range(len(ds))) if len(tgt["boxes"])]
    assert with_boxes, "the GT box did not survive tiling on a channel-first raster"
    for tile, target in with_boxes:
        assert tile.shape == (5, 16, 16)
        for x1, y1, x2, y2 in target["boxes"].tolist():
            region = tile[:, int(y1):int(y2) + 1, int(x1):int(x2) + 1]
            assert region.numel() and float(region.max()) > 0.5
