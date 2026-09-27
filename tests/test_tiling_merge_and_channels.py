"""Channel-generic tiled training reads: multi-band slices keep their boxes on their pixels."""

from __future__ import annotations

import numpy as np
from tests._producer_fixtures import dataset_over  # noqa: E402


def _multiband_detection_fixture(tmp_path, width=40, height=24, bands=5, patch=(28, 12)):
    """A non-square multi-band GeoTIFF with a bright patch, and a GT box on that patch."""
    import tifffile
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    arr = np.zeros((height, width, bands), dtype=np.uint8)
    px, py = patch
    arr[py:py + 6, px:px + 6, :] = 255
    tifffile.imwrite(images_dir / "a.tif", arr)
    json_io.write_annotations(str(labels_dir / "a.json"),
                              [Annotation(subject="bud", geometry=BBox(px, py, px + 6, py + 6))],
                              width, height, keep_empty=True)
    return images_dir, labels_dir


def test_tiled_detection_reads_multiband_and_keeps_boxes_on_their_pixels(tmp_path):
    """Channel count is not enough: a frame mismatch passes a shape check but displaces every box.

    The bright patch and the GT box are placed together, so this fails if the tile is cut from a
    differently-oriented frame than the labels were clipped against.
    """
    import torch


    images_dir, labels_dir = _multiband_detection_fixture(tmp_path)
    ds = dataset_over("detection", str(images_dir), str(labels_dir), subject="bud",
                      stated={"num_channels": 5},
                      tiling={"enabled": True, "tile_size": 16, "overlap": 0.0})
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
    """Refuse when the labels' own frame disagrees with the decode: the real scramble case.

    The annotation stack measures with PIL, which reports a 40x24x5 GeoTIFF as 5x40, so labels
    authored through it genuinely disagree with the multi-band decode. Comparing two decoders
    instead would prove nothing: they share a branch and agree by construction.
    """
    import pytest
    import tifffile
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_annotation.utils import get_image_dimensions


    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    arr = np.zeros((24, 40, 5), dtype=np.uint8)
    arr[12:18, 28:34, :] = 255
    tifffile.imwrite(images_dir / "a.tif", arr)

    pil_w, pil_h = get_image_dimensions(str(images_dir / "a.tif"))
    assert (pil_w, pil_h) == (5, 40), "fixture assumes PIL misreads this multi-band raster"
    json_io.write_annotations(str(labels_dir / "a.json"),
                              [Annotation(subject="bud", geometry=BBox(1, 12, 4, 18))], pil_w, pil_h,
                              keep_empty=True)

    with pytest.raises(ValueError, match="the labels record a 5x40 image but it decodes as 40x24"):
        dataset_over("detection", str(images_dir), str(labels_dir), subject="bud",
                     stated={"num_channels": 5},
                     tiling={"enabled": True, "tile_size": 16, "overlap": 0.0})


def test_authored_frame_raises_on_a_corrupt_label_rather_than_reading_as_no_frame(tmp_path):
    """authored_frame's per-image branch reads through the one label reader
    (splits.label_document_extent), so a present, unreadable label raises rather than
    silently disabling the tiled dataset's frame-mismatch check for that sample."""
    import pytest
    from tcip_annotation.json_io import UnreadableLabelDocument

    from tcip_mcp.pipelines.data.label_queries import authored_frame

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "a.json").write_bytes(b"{not json")

    with pytest.raises(UnreadableLabelDocument):
        authored_frame(labels_dir / "a.json")


def test_ctx_tiled_dataset_inherits_the_band_count(tmp_path):
    """ctx.tiled_dataset constructs the tiler directly: it must not fall back to 3 channels."""
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset

    images_dir, labels_dir = _multiband_detection_fixture(tmp_path)
    base = dataset_over("detection", str(images_dir), str(labels_dir), subject="bud",
                        stated={"num_channels": 5})
    assert TiledDetectionDataset(base, tile_size=16).expected_channels == 5


def test_tiled_detection_handles_channel_first_rasters(tmp_path):
    """The other common GeoTIFF layout, where the axis-order heuristic is observable."""
    import tifffile
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox


    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    arr = np.zeros((24, 40, 5), dtype=np.uint8)
    arr[12:18, 28:34, :] = 255
    tifffile.imwrite(images_dir / "a.tif", np.transpose(arr, (2, 0, 1)))  # [C, H, W]
    json_io.write_annotations(str(labels_dir / "a.json"),
                              [Annotation(subject="bud", geometry=BBox(28, 12, 34, 18))], 40, 24,
                              keep_empty=True)

    ds = dataset_over("detection", str(images_dir), str(labels_dir), subject="bud",
                      stated={"num_channels": 5},
                      tiling={"enabled": True, "tile_size": 16, "overlap": 0.0})
    with_boxes =[(t, tgt) for t, tgt in (ds[i] for i in range(len(ds))) if len(tgt["boxes"])]
    assert with_boxes, "the GT box did not survive tiling on a channel-first raster"
    for tile, target in with_boxes:
        assert tile.shape == (5, 16, 16)
        for x1, y1, x2, y2 in target["boxes"].tolist():
            region = tile[:, int(y1):int(y2) + 1, int(x1):int(x2) + 1]
            assert region.numel() and float(region.max()) > 0.5
