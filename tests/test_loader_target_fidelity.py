"""What a loader hands the trainer for one sample: box extents, the tile index's own boxes, and
the fields that identify the sample.

Fixtures here are deliberately asymmetric (boxes wider than they are tall and taller than they are
wide, a non-square frame, sparse ordinal ranks) so a transposed extent, an unfiltered tile index or
a collapsed identity produces different numbers than the correct behavior rather than the same
ones.
"""

import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402

BUD = "bud"


def _make_images(images_dir, stems, size):
    images_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        Image.new("RGB", size).save(images_dir / f"{stem}.jpg")


def _write(labels_dir, stem, boxes, size):
    json_io.write_annotations(
        labels_dir / f"{stem}.json",
        [Annotation(subject=BUD, geometry=BBox(*b)) for b in boxes],
        size[0], size[1], keep_empty=True)


def test_a_boxs_width_and_height_survive_the_assembled_coco_round_trip(tmp_path):
    """Training assembles the per-image JSON into COCO xywh and the loader converts it back to
    pixel xyxy. A box wider than it is tall must stay wider than it is tall: swapping the two keeps
    the origin and moves the far corner, so every object trains against the wrong extent while the
    box count and its position both look right.
    """
    from tcip_mcp.pipelines.data.datasets import build_dataset

    size = (120, 80)
    images, labels = tmp_path / "images", tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["img0"], size)
    _write(labels, "img0", [(10, 20, 70, 40), (80, 5, 95, 75)], size)

    ds = build_dataset("detection", images_dir=str(images), labels_dir=str(labels), subject=BUD)
    assert ds.label_format == "coco"
    _img, target = ds[0]
    assert target["boxes"].tolist() == [[10.0, 20.0, 70.0, 40.0], [80.0, 5.0, 95.0, 75.0]]


# A 300x200 frame tiled at 64 with no overlap: columns 0/64/128/192/256, rows 0/64/128/192.
# Two boxes cross a seam by two pixels; the rest sit inside one tile each.
SEAM_BOXES = [
    (10, 20, 66, 47),     # 56x27, crosses the x=64 seam, leaving a 2x27 stub in the next column
    (70, 10, 110, 34),    # 40x24, inside column 64
    (150, 100, 190, 127),  # 40x27, inside column 128, row 64
    (200, 20, 236, 60),   # 36x40, inside column 192, row 0
    (30, 140, 60, 190),   # 30x50, inside column 0, row 128
    (200, 126, 244, 156),  # 44x30, crosses the y=128 seam, leaving a 44x2 stub in the row above
]


@pytest.mark.parametrize("sliver_frac", [None, 0.5])
def test_a_seam_fragment_is_not_indexed_as_a_whole_object(tmp_path, sliver_frac):
    """A box a tile boundary cuts down to a two-pixel stub is a fragment, not an object. The
    cutoff the dataset resolves, derived from this dataset's own size spread or supplied by the
    caller, has to reach the index: otherwise the model is trained to call the stub a whole object,
    and the recorded fraction describes a filter the data never went through. The tiles holding the
    rest of each box keep it.
    """
    from tcip_mcp.pipelines.data.datasets import DetectionDataset, TiledDetectionDataset

    size = (300, 200)
    images, labels = tmp_path / "images", tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["img0"], size)
    _write(labels, "img0", SEAM_BOXES, size)

    base = DetectionDataset(str(images), str(labels), subject=BUD)
    ds = TiledDetectionDataset(base, tile_size=64, overlap=0.0, sliver_frac=sliver_frac)

    assert ds.min_box_size > 0
    per_tile = {(e["tile_x"], e["tile_y"]): len(e["boxes"]) for e in ds._index}
    assert len(per_tile) == 20, "five tile columns by four rows over the 300x200 frame"
    assert per_tile[(64, 0)] == 1, "the stub beside the neighbouring box was indexed as an object"
    assert per_tile[(192, 64)] == 0, "the stub above the seam was indexed as an object"
    assert per_tile[(0, 0)] == 1, "the clipped box's own tile lost it"
    assert per_tile[(192, 128)] == 1, "the clipped box's own tile lost it"
    assert sum(per_tile.values()) == len(SEAM_BOXES)


def test_each_detection_sample_carries_its_own_index_as_image_id(tmp_path):
    """The target dict says which sample it came from, and a consumer joins per-sample results back
    through that field. A shared value silently attributes every sample's boxes to one image.
    """
    from tcip_mcp.pipelines.data.datasets import build_dataset

    size = (120, 80)
    images, labels = tmp_path / "images", tmp_path / "annotations"
    labels.mkdir(parents=True)
    stems = ["a", "b", "c"]
    _make_images(images, stems, size)
    for i, stem in enumerate(stems):
        _write(labels, stem, [(5 + 20 * k, 6, 25 + 20 * k, 44) for k in range(i + 1)], size)

    ds = build_dataset("detection", images_dir=str(images), labels_dir=str(labels), subject=BUD)
    assert len(ds) == 3
    samples = [ds[i][1] for i in range(len(ds))]
    assert [t["image_id"] for t in samples] == [0, 1, 2]
    assert [t["boxes"].shape[0] for t in samples] == [1, 2, 3]


def test_an_ordinal_sample_carries_the_rank_count_its_head_decodes_with(tmp_path):
    """The CORN loss and decode both loop over the rank count the target carries, so it is how many
    ranks the scale reaches, never how many samples were loaded. The ranks here are sparse, so the
    two are different numbers.
    """
    from tcip_mcp.pipelines.data.datasets import build_dataset

    images = tmp_path / "images"
    _make_images(images, ["s0", "s1", "s2"], (48, 32))
    csv_path = tmp_path / "ranks.csv"
    csv_path.write_text("stem,rank\ns0,0\ns1,2\ns2,5\n", encoding="utf-8")

    ds = build_dataset("ordinal", images_dir=str(images), csv_path=str(csv_path))
    assert len(ds) == 3
    for idx, expected_rank in enumerate([0, 2, 5]):
        _img, target = ds[idx]
        assert target["ranks"] == expected_rank
        assert target["num_ranks"] == 6
