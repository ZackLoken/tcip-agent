"""Tests for the canonical dataset-layout resolver."""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.dataset_layout import (
    annotation_dir,
    annotation_path_for_image,
    find_gt_label,
    parse_image_path,
    prediction_dir,
)


def test_parse_image_path_date_nested() -> None:
    root, date, stem = parse_image_path("/ds/images/2-11-26/IMG_1.JPG")
    assert Path(root) == Path("/ds")
    assert date == "2-11-26"
    assert stem == "IMG_1"


def test_parse_image_path_flat() -> None:
    root, date, stem = parse_image_path("/ds/images/IMG_1.JPG")
    assert Path(root) == Path("/ds")
    assert date is None
    assert stem == "IMG_1"


def test_annotation_dir_with_and_without_date() -> None:
    assert annotation_dir("/ds", "catkin", "2-11-26", "detect") == Path(
        "/ds/annotations/catkin/2-11-26/detect"
    )
    assert annotation_dir("/ds", "catkin", None, "detect") == Path("/ds/annotations/catkin/detect")


def test_prediction_dir_date_nested() -> None:
    assert prediction_dir("/ds", "m1", "2-11-26", "segment") == Path(
        "/ds/predictions/m1/2-11-26/segment"
    )


def test_annotation_path_for_image_derives_date() -> None:
    p = annotation_path_for_image("/ds/images/2-11-26/IMG_1.JPG", "detect", "yolo", trait="catkin")
    assert p == Path("/ds/annotations/catkin/2-11-26/detect/IMG_1.txt")


def test_find_gt_label_prefers_canonical(tmp_path: Path) -> None:
    img = tmp_path / "images" / "2-11-26" / "IMG_1.JPG"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"x")
    ann = tmp_path / "annotations" / "catkin" / "2-11-26" / "detect"
    ann.mkdir(parents=True)
    (ann / "IMG_1.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    assert find_gt_label(str(img), "detect") == ann / "IMG_1.txt"


def test_find_gt_label_missing_returns_none(tmp_path: Path) -> None:
    img = tmp_path / "images" / "IMG_1.JPG"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"x")
    assert find_gt_label(str(img), "detect") is None
