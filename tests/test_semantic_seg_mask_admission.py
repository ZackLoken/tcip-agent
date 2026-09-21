"""A semantic-segmentation sample is an image plus exactly ``<stem>.png`` in the labels
directory. A same-stem entry in any other format refuses by name, and no sample ever trains
with a fabricated all-background mask.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from PIL import Image  # noqa: E402
from tests._producer_fixtures import dataset_over  # noqa: E402


def _dataset(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    for i in range(2):
        Image.new("RGB", (8, 8), (i * 40, 0, 0)).save(images_dir / f"img{i}.png")
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    Image.fromarray(mask, mode="L").save(masks_dir / "img0.png")
    return images_dir, masks_dir


def test_a_png_mask_admits_its_image_and_serves_that_mask(tmp_path: Path) -> None:

    images_dir, masks_dir = _dataset(tmp_path)
    Image.fromarray(np.ones((8, 8), dtype=np.uint8), mode="L").save(masks_dir / "img1.png")

    ds = dataset_over("semantic_seg", str(images_dir), str(masks_dir), num_classes=2)

    assert ds.record_stems == ["img0", "img1"]
    _img, target = ds[0]
    assert int(target["masks"].sum()) == 16
    _img, target = ds[1]
    assert int(target["masks"].sum()) == 64


def test_a_same_stem_non_png_entry_refuses(tmp_path: Path) -> None:

    images_dir, masks_dir = _dataset(tmp_path)
    Image.fromarray(np.ones((8, 8), dtype=np.uint8), mode="L").save(masks_dir / "img1.tif")

    with pytest.raises(ValueError, match="img1.tif"):
        dataset_over("semantic_seg", str(images_dir), str(masks_dir), num_classes=2)


def test_the_class_count_is_derived_from_the_masks_the_run_was_handed(tmp_path: Path) -> None:
    """The head is sized for the classes this run's own masks hold, never a pinned two: masks
    reaching id 2 need three logits, a caller may state more than they hold, and a caller stating
    fewer refuses by name rather than indexing past the head a target would then reach."""
    images_dir, masks_dir = _dataset(tmp_path)
    three_classes = np.zeros((8, 8), dtype=np.uint8)
    three_classes[0:4, 0:4] = 1
    three_classes[4:8, 4:8] = 2
    Image.fromarray(three_classes, mode="L").save(masks_dir / "img1.png")

    assert dataset_over("semantic_seg", str(images_dir), str(masks_dir)).num_classes == 3
    assert dataset_over(
        "semantic_seg", str(images_dir), str(masks_dir), num_classes=5).num_classes == 5

    with pytest.raises(ValueError, match="num_classes"):
        dataset_over("semantic_seg", str(images_dir), str(masks_dir), num_classes=2)


def test_an_image_with_no_mask_is_skipped_never_served_as_background(tmp_path: Path) -> None:

    images_dir, masks_dir = _dataset(tmp_path)

    from tests._producer_fixtures import admit_over

    ds = dataset_over("semantic_seg", str(images_dir), str(masks_dir), num_classes=2)
    assert ds.record_stems == ["img0"]
    assert admit_over(images_dir, masks_dir).counts == {
        "annotated": 1, "skipped_unannotated": 1}
