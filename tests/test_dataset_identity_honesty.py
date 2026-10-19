"""A dataset that has lost its imagery or its labels recomputes to no identity.

The recompute over the surviving half is ``None``.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

import tcip_store as ts
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.dataset_layout import capture_label_keys
from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint
from tests._producer_fixtures import label_image

_DATE = "2026-02-11"


def _dataset(root: Path) -> None:
    """Write two images of different sizes and colors under one dated capture, each labeled."""
    (root / "images" / _DATE).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 32), color=(120, 90, 40)).save(
        root / "images" / _DATE / "IMG_1.png"
    )
    Image.new("RGB", (64, 40), color=(20, 160, 70)).save(
        root / "images" / _DATE / "IMG_2.png"
    )
    label_image(
        root / "images" / _DATE / "IMG_1.png",
        [Annotation(subject="bud", geometry=BBox(4, 6, 22, 31))],
        48,
        32,
    )
    label_image(
        root / "images" / _DATE / "IMG_2.png",
        [Annotation(subject="bud", geometry=BBox(9, 3, 60, 28))],
        64,
        40,
    )


def test_a_dataset_whose_images_were_all_removed_reports_no_identity(
    tmp_path: Path,
) -> None:
    """A dataset whose images were all removed reports no identity, its labels left in place."""
    _dataset(tmp_path)
    before = dataset_fingerprint(tmp_path)
    assert before is not None
    assert len(before) == 16

    for img in (tmp_path / "images" / _DATE).glob("*.png"):
        img.unlink()
    assert dataset_fingerprint(tmp_path) is None


def test_a_dataset_whose_labels_were_all_removed_reports_no_identity(
    tmp_path: Path,
) -> None:
    """A dataset whose labels were all removed reports no identity from its imagery alone."""
    _dataset(tmp_path)
    for key in capture_label_keys(tmp_path, _DATE):
        ts.delete(key)
    assert dataset_fingerprint(tmp_path) is None


def test_the_identity_comes_back_when_the_imagery_does(tmp_path: Path) -> None:
    """Imagery rewritten byte for byte under its names recomputes to the identity it had.

    While the images are gone the identity is ``None``.
    """
    _dataset(tmp_path)
    before = dataset_fingerprint(tmp_path)
    for img in (tmp_path / "images" / _DATE).glob("*.png"):
        img.unlink()
    assert dataset_fingerprint(tmp_path) is None

    _dataset(tmp_path)
    assert dataset_fingerprint(tmp_path) == before
