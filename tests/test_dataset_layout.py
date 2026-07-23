"""Tests for the canonical dataset-layout resolver."""

from __future__ import annotations

from pathlib import Path

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox

from tcip_mcp.dataset_layout import (
    annotation_dir,
    annotation_path_for_image,
    find_gt_label,
    models_with_predictions,
    parse_image_path,
    prediction_dir,
    subjects_with_labels,
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
    # Labels are one file per image; the path no longer carries a subject or task segment.
    assert annotation_dir("/ds", "2-11-26") == Path("/ds/annotations/2-11-26")
    assert annotation_dir("/ds", None) == Path("/ds/annotations")


def test_prediction_dir_date_nested() -> None:
    assert prediction_dir("/ds", "m1", "2-11-26") == Path("/ds/predictions/m1/2-11-26")


def test_annotation_path_for_image_derives_date() -> None:
    p = annotation_path_for_image("/ds/images/2-11-26/IMG_1.JPG")
    assert p == Path("/ds/annotations/2-11-26/IMG_1.json")


def test_find_gt_label_prefers_canonical(tmp_path: Path) -> None:
    img = tmp_path / "images" / "2-11-26" / "IMG_1.JPG"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"x")
    ann = tmp_path / "annotations" / "2-11-26"
    ann.mkdir(parents=True)
    (ann / "IMG_1.json").write_text('{"annotations": []}')
    assert find_gt_label(str(img)) == ann / "IMG_1.json"


def test_find_gt_label_missing_returns_none(tmp_path: Path) -> None:
    img = tmp_path / "images" / "IMG_1.JPG"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"x")
    assert find_gt_label(str(img)) is None


def _touch(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_subjects_with_labels_is_per_date(tmp_path: Path) -> None:
    root = tmp_path
    # Subjects are read from the per-image label records (the path no longer encodes them).
    # catkin labelled on 2026-02-11 only; bush labelled on 2026-03-02 only.
    json_io.write_annotations(
        str(annotation_dir(root, "2026-02-11") / "IMG_1.json"),
        [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))], 100, 100)
    json_io.write_annotations(
        str(annotation_dir(root, "2026-03-02") / "IMG_9.json"),
        [Annotation(subject="bush", geometry=BBox(1, 1, 9, 9))], 100, 100)
    # A second image on 2026-03-02 carries catkin, so that date offers both subjects.
    json_io.write_annotations(
        str(annotation_dir(root, "2026-03-02") / "IMG_5.json"),
        [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))], 100, 100)

    assert subjects_with_labels(root, "2026-02-11") == ["catkin"]
    # 2026-03-02 has bush and catkin → both, sorted.
    assert subjects_with_labels(root, "2026-03-02") == ["bush", "catkin"]
    # A date with no labels for any subject → nothing to offer.
    assert subjects_with_labels(root, "2026-03-24") == []


def test_models_with_predictions_is_per_date(tmp_path: Path) -> None:
    root = tmp_path
    _touch(prediction_dir(root, "baseline", "2026-02-11") / "IMG_1.json", '{"annotations": []}')
    # 'baseline' has a predictions dir on 03-24 but no files in it → not offered.
    (prediction_dir(root, "baseline", "2026-03-24")).mkdir(parents=True)

    assert models_with_predictions(root, "2026-02-11") == ["baseline"]
    assert models_with_predictions(root, "2026-03-24") == []
    assert models_with_predictions(root, "2026-03-02") == []


def test_classes_path_is_the_single_dataset_registry():
    from tcip_mcp.dataset_layout import classes_path

    # One nested registry at the dataset root — no per-subject classes/<x>.json anymore.
    assert classes_path("/ds") == Path("/ds/classes.json")


def test_dataset_root_of_recovers_the_root_from_any_layout_dir():
    from tcip_mcp.dataset_layout import dataset_root_of

    assert dataset_root_of("/ds/annotations/2026-03-02") == Path("/ds")
    assert dataset_root_of("/ds/predictions/live/2026-03-02") == Path("/ds")
    assert dataset_root_of("/ds/images/2026-03-02") == Path("/ds")
    assert dataset_root_of("/some/where/else") is None
    # Anchors on the LAST dataset segment: a dataset nested under an ancestor named 'annotations'
    # (or any other segment) still resolves to the real root, not the ancestor.
    assert dataset_root_of("/data/annotations/proj/predictions/live") == Path("/data/annotations/proj")
    assert dataset_root_of("/data/images/proj/annotations/2026-03-02") == Path("/data/images/proj")
    # A bare segment with nothing above it is not inside a dataset.
    assert dataset_root_of("annotations/2026-03-02") is None
