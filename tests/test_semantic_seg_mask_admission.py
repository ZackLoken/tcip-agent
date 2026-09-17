"""A semantic-segmentation sample is an image plus exactly ``<stem>.png`` in the masks
directory. A same-stem entry in any other format refuses by name, and no sample ever trains
with a fabricated all-background mask.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from PIL import Image  # noqa: E402


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
    from tcip_mcp.pipelines.data.datasets import build_dataset

    images_dir, masks_dir = _dataset(tmp_path)
    Image.fromarray(np.ones((8, 8), dtype=np.uint8), mode="L").save(masks_dir / "img1.png")

    ds = build_dataset("semantic_seg", images_dir=str(images_dir), masks_dir=str(masks_dir),
                       num_classes=2)

    assert ds.stems == ["img0", "img1"]
    _img, target = ds[0]
    assert int(target["masks"].sum()) == 16
    _img, target = ds[1]
    assert int(target["masks"].sum()) == 64


def test_a_same_stem_non_png_entry_refuses(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.datasets import build_dataset

    images_dir, masks_dir = _dataset(tmp_path)
    Image.fromarray(np.ones((8, 8), dtype=np.uint8), mode="L").save(masks_dir / "img1.tif")

    with pytest.raises(ValueError, match="img1.tif"):
        build_dataset("semantic_seg", images_dir=str(images_dir), masks_dir=str(masks_dir),
                      num_classes=2)


def test_an_image_with_no_mask_is_skipped_never_served_as_background(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.datasets import build_dataset

    images_dir, masks_dir = _dataset(tmp_path)

    ds = build_dataset("semantic_seg", images_dir=str(images_dir), masks_dir=str(masks_dir),
                       num_classes=2)

    assert ds.stems == ["img0"]
    assert ds.sample_counts == {"annotated": 1, "skipped_unannotated": 1}
