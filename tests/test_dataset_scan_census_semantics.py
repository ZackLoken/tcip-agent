"""What each count scan_dataset reports actually counts.

The tool's whole output is a census, so a count that quietly measures a different collection
than its name claims (labels counted as images, a stamp counted as a prediction, the
unlabelled remainder taken off the wrong total) is invisible to the reader.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    unlabelled remainder are four different numbers, and the frame is not square, so a count
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
    """image_count, labels_count, paired_images and unlabelled_images each measure their own
    collection: the unlabelled remainder is the images no label pairs with, never the labels
    no image pairs with."""
    root = _lopsided_dataset(tmp_path / "ds")

    result = scan_dataset(str(root))

    assert result["image_count"] == 5
    assert result["labels_count"] == 3
    assert result["paired_images"] == 2
    assert result["unlabelled_images"] == 3
    assert result["image_stems_sample"] == [
        "plotA_0_0", "plotA_0_1", "plotB_0_0", "plotB_0_1", "plotC_0_0",
    ]


def test_operating_point_stamp_is_not_counted_as_a_prediction(tmp_path: Path):
    """predictions/ holds per-image prediction files plus the run's operating-point stamp; only
    the per-image files are predictions."""
    root = _lopsided_dataset(tmp_path / "ds")
    pred_dir = root / "predictions" / "run_a" / DATE
    pred_dir.mkdir(parents=True)
    for stem in ("plotA_0_0", "plotB_0_0"):
        json_io.write_annotations(
            pred_dir / f"{stem}.json",
            [Annotation(subject=SUBJECT, geometry=BBox(12, 8, 40, 52), score=0.8)],
            96, 64,
        )
    (pred_dir / "operating_point.json").write_text(
        json.dumps({"conf": 0.41, "nms_iou": 0.5}), encoding="utf-8"
    )

    result = scan_dataset(str(root))

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
