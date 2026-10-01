"""What each count scan_dataset reports actually counts.

The tool's whole output is a census, so a count that quietly measures a different collection
than its name claims (labels counted as images, a bucket's record counted as a prediction, the
unlabeled remainder taken off the wrong total) is invisible to the reader.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox

from tcip_mcp.tools.data_tools import scan_dataset

DATE = "2-11-26"
SUBJECT = "bud"


def _write_image(path: Path, width: int, height: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(90, 120, 60)).save(path)


def _lopsided_dataset(root: Path) -> Path:
    """Five images, three label files, two of which pair with an image.

    Deliberately asymmetric: the image count, the label count, the paired count and the
    unlabeled remainder are four different numbers, and the frame is not square, so a count
    computed off the wrong collection cannot coincide with the right one.
    """
    images_dir = root / "images" / DATE
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    for stem in ("plotA_0_0", "plotA_0_1", "plotB_0_0", "plotB_0_1", "plotC_0_0"):
        _write_image(images_dir / f"{stem}.jpg", 96, 64)
    for stem in ("plotA_0_0", "plotA_0_1", "plotZ_9_9"):
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject=SUBJECT, geometry=BBox(11, 7, 39, 51))],
            96, 64,
        )
    return root


def test_scan_reports_four_distinct_counts_over_a_lopsided_dataset(tmp_path: Path):
    """image_count, labels_count, paired_images and unlabeled_images each measure their own
    collection: the unlabeled remainder is the images no label pairs with, never the labels
    no image pairs with."""
    root = _lopsided_dataset(tmp_path / "ds")

    result = scan_dataset(str(root))

    assert result["image_count"] == 5
    assert result["labels_count"] == 3
    assert result["paired_images"] == 2
    assert result["unlabeled_images"] == 3
    assert result["image_stems_sample"] == [
        "plotA_0_0", "plotA_0_1", "plotB_0_0", "plotB_0_1", "plotC_0_0",
    ]


def test_a_buckets_record_is_not_counted_as_a_prediction(tmp_path: Path):
    """A published bucket holds its per-image documents plus its own record; only the documents
    are predictions, and documents in no bucket are none."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    root = _lopsided_dataset(tmp_path / "ds")
    bucket = root / "predictions" / "run_a" / DATE
    published(tmp_path, bucket, [
        {"image": str(root / "images" / DATE / f"{stem}.jpg"), "width": 96, "height": 64,
         "boxes": [[12.0, 8.0, 40.0, 52.0]], "scores": [0.8], "labels": [1]}
        for stem in ("plotA_0_0", "plotB_0_0")], scope={"subject": SUBJECT, "id_map": {SUBJECT: 0}})
    staged = root / "predictions" / "staged" / DATE
    staged.mkdir(parents=True)
    json_io.write_annotations(
        staged / "plotC_0_0.json",
        [Annotation(subject=SUBJECT, geometry=BBox(12, 8, 40, 52), score=0.8)], 96, 64)

    result = scan_dataset(str(root))

    assert (bucket / "bucket.json").is_file()
    assert result["predictions_count"] == 2


def test_scan_writes_nothing_into_the_dataset_tree(tmp_path: Path):
    """The census is a read of the tree, so the tree is byte-identical afterwards."""
    root = _lopsided_dataset(tmp_path / "ds")
    before = {
        str(p.relative_to(root)): p.stat().st_size for p in sorted(root.rglob("*")) if p.is_file()
    }
    assert before

    scan_dataset(str(root))

    after = {
        str(p.relative_to(root)): p.stat().st_size for p in sorted(root.rglob("*")) if p.is_file()
    }
    assert after == before
