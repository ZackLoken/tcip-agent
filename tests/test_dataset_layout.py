"""Tests for the canonical dataset-layout resolver."""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.dataset_layout import (
    annotation_dir,
    annotation_path_for_image,
    find_gt_label,
    models_with_predictions,
    parse_image_path,
    prediction_dir,
    traits_with_labels,
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
    assert p == Path("/ds/annotations/catkin/2-11-26/detect/IMG_1.json")


def test_find_gt_label_prefers_canonical(tmp_path: Path) -> None:
    img = tmp_path / "images" / "2-11-26" / "IMG_1.JPG"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"x")
    ann = tmp_path / "annotations" / "catkin" / "2-11-26" / "detect"
    ann.mkdir(parents=True)
    (ann / "IMG_1.json").write_text("0 0.5 0.5 0.1 0.1\n")
    assert find_gt_label(str(img), "detect") == ann / "IMG_1.json"


def test_find_gt_label_missing_returns_none(tmp_path: Path) -> None:
    img = tmp_path / "images" / "IMG_1.JPG"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"x")
    assert find_gt_label(str(img), "detect") is None


def _touch(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_traits_with_labels_is_per_date(tmp_path: Path) -> None:
    root = tmp_path
    # catkin labelled on 2026-02-11 only; bush labelled on 2026-03-02 only.
    _touch(annotation_dir(root, "catkin", "2026-02-11", "detect") / "IMG_1.json", "0 0.5 0.5 0.1 0.1\n")
    _touch(annotation_dir(root, "bush", "2026-03-02", "detect") / "IMG_9.json", "0 0.5 0.5 0.2 0.2\n")
    # An empty label file is a confirmed negative — still counts as "labelled".
    _touch(annotation_dir(root, "catkin", "2026-03-02", "segment") / "IMG_5.json", "")

    assert traits_with_labels(root, "2026-02-11") == ["catkin"]
    # 2026-03-02 has bush (detect) and catkin (empty negative in segment) → both, sorted.
    assert traits_with_labels(root, "2026-03-02") == ["bush", "catkin"]
    # A date with images but no labels for any trait → nothing to offer.
    assert traits_with_labels(root, "2026-03-24") == []


def test_models_with_predictions_is_per_date(tmp_path: Path) -> None:
    root = tmp_path
    _touch(prediction_dir(root, "baseline", "2026-02-11", "detect") / "IMG_1.json", "0 0.9 0.5 0.5 0.1 0.1\n")
    # 'baseline' has a predictions dir on 03-24 but no files in it → not offered.
    (prediction_dir(root, "baseline", "2026-03-24", "detect")).mkdir(parents=True)

    assert models_with_predictions(root, "2026-02-11") == ["baseline"]
    assert models_with_predictions(root, "2026-03-24") == []
    assert models_with_predictions(root, "2026-03-02") == []
