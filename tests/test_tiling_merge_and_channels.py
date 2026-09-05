"""Cross-tile NMM merge + channel-generic tiled crop/pad."""

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
    # IoS = 1.0 (fully contained). NMM must merge it: the seam-split case merging exists for.
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


def test_train_and_inference_tilers_are_the_same_function():
    """One cropper, so the two ends of the reproduce-a-number chain cannot drift apart."""
    from tcip_mcp.pipelines.data import datasets
    from tcip_mcp.pipelines.image_utils import crop_pad_tile
    from tcip_mcp.pipelines.inference.generic_predictor import _crop_pad_tile

    assert _crop_pad_tile is crop_pad_tile
    assert datasets.crop_pad_tile is crop_pad_tile


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

    from tcip_mcp.pipelines.data.datasets import build_dataset

    images_dir, labels_dir = _multiband_detection_fixture(tmp_path)
    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                       subject="bud", num_channels=5,
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

    from tcip_mcp.pipelines.data.datasets import build_dataset

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
        build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                      subject="bud", num_channels=5,
                      tiling={"enabled": True, "tile_size": 16, "overlap": 0.0})


def test_authored_frame_raises_on_a_corrupt_label_rather_than_reading_as_no_frame(tmp_path):
    """authored_frame's json branch reads through the one label reader
    (splits.image_extent_from_labels), so a present, unreadable label raises rather than
    silently disabling the tiled dataset's frame-mismatch check for that stem."""
    import pytest
    from tcip_annotation.json_io import UnreadableLabelDocument

    from tcip_mcp.pipelines.data.label_queries import authored_frame

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "a.json").write_bytes(b"{not json")

    with pytest.raises(UnreadableLabelDocument):
        authored_frame("a", labels_dir, "json")


def test_ctx_tiled_dataset_inherits_the_band_count(tmp_path):
    """ctx.tiled_dataset constructs the tiler directly: it must not fall back to 3 channels."""
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset, build_dataset

    images_dir, labels_dir = _multiband_detection_fixture(tmp_path)
    base = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                         subject="bud", num_channels=5)
    assert TiledDetectionDataset(base, tile_size=16).expected_channels == 5


def test_tiled_detection_handles_channel_first_rasters(tmp_path):
    """The other common GeoTIFF layout, where the axis-order heuristic is observable."""
    import tifffile
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    from tcip_mcp.pipelines.data.datasets import build_dataset

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    arr = np.zeros((24, 40, 5), dtype=np.uint8)
    arr[12:18, 28:34, :] = 255
    tifffile.imwrite(images_dir / "a.tif", np.transpose(arr, (2, 0, 1)))  # [C, H, W]
    json_io.write_annotations(str(labels_dir / "a.json"),
                              [Annotation(subject="bud", geometry=BBox(28, 12, 34, 18))], 40, 24,
                              keep_empty=True)

    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                       subject="bud", num_channels=5,
                       tiling={"enabled": True, "tile_size": 16, "overlap": 0.0})
    with_boxes = [(t, tgt) for t, tgt in (ds[i] for i in range(len(ds))) if len(tgt["boxes"])]
    assert with_boxes, "the GT box did not survive tiling on a channel-first raster"
    for tile, target in with_boxes:
        assert tile.shape == (5, 16, 16)
        for x1, y1, x2, y2 in target["boxes"].tolist():
            region = tile[:, int(y1):int(y2) + 1, int(x1):int(x2) + 1]
            assert region.numel() and float(region.max()) > 0.5
