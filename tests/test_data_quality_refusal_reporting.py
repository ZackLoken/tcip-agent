"""What validate_data_quality reports, and what it must never quietly claim.

Two standing facts the caller relies on. First, a label store whose format the detector refused
is a different state from a dataset that simply has no labels, and the report has to keep the two
apart rather than collapsing them into one reassuring shape. Second, the report's own vocabulary
is load bearing: a warning and an error are not interchangeable, and the subjects listed are the
ones present in the label files, not the ones a registry declares.
"""

from __future__ import annotations

import json
from pathlib import Path

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox

from tcip_mcp.tools.data_tools import validate_data_quality

DATE = "2-11-26"


def _write_image(path: Path, width: int, height: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(90, 120, 60)).save(path)


def test_undetectable_label_store_is_reported_as_labels_present_without_a_format(tmp_path: Path):
    """Labels present but undetectable reports the labels it found and no format; a dataset with
    no annotations dir at all reports no format and no labels. The two states stay distinguishable
    from the report alone."""
    root = tmp_path / "undetectable"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    for stem in ("plotA_0_0", "plotA_0_1"):
        _write_image(root / "images" / DATE / f"{stem}.jpg", 96, 64)
        (labels_dir / f"{stem}.json").write_text(
            json.dumps({"shapes": [{"label": "catkin", "points": [[3, 5], [40, 52]]}]}),
            encoding="utf-8",
        )

    refused = validate_data_quality(str(root))
    assert refused["format"] is None
    assert refused["total_labels"] == 2
    assert refused["total_images"] == 2
    assert refused["subjects"] == []

    bare = tmp_path / "unlabelled"
    for stem in ("plotA_0_0", "plotA_0_1"):
        _write_image(bare / "images" / DATE / f"{stem}.jpg", 96, 64)

    no_labels = validate_data_quality(str(bare))
    assert no_labels["format"] is None
    assert no_labels["total_labels"] == 0
    assert no_labels["total_images"] == 2


def test_a_label_with_no_matching_image_is_an_error_that_denies_validity(tmp_path: Path):
    """An orphan label is an error-level issue, and an error is what makes the dataset invalid."""
    root = tmp_path / "ds"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    for stem in ("plotA_0_0", "plotB_0_0"):
        _write_image(root / "images" / DATE / f"{stem}.jpg", 96, 64)
    for stem in ("plotA_0_0", "plotB_0_0", "plotZ_9_9"):
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="catkin", geometry=BBox(11, 7, 39, 51))],
            96, 64,
        )

    result = validate_data_quality(str(root))

    errors = [i for i in result["issues"] if i["level"] == "error"]
    assert len(errors) == 1
    assert Path(errors[0]["file"]).stem == "plotZ_9_9"
    assert result["is_valid"] is False


def test_a_coco_image_missing_from_the_images_dir_is_a_warning_that_leaves_the_dataset_valid(
    tmp_path: Path,
):
    """A COCO entry pointing at an absent image file is reported at warning level, and a warning
    on its own never denies validity."""
    root = tmp_path / "ds"
    for stem in ("plotA_0_0", "plotB_0_0"):
        _write_image(root / "images" / DATE / f"{stem}.jpg", 96, 64)
    (root / "annotations.json").write_text(json.dumps({
        "images": [
            {"id": 1, "file_name": "plotA_0_0.jpg", "width": 96, "height": 64},
            {"id": 2, "file_name": "plotB_0_0.jpg", "width": 96, "height": 64},
            {"id": 3, "file_name": "plotC_0_0.jpg", "width": 96, "height": 64},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [11, 7, 28, 44], "area": 1232,
             "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "catkin"}],
    }), encoding="utf-8")

    result = validate_data_quality(str(root))

    assert result["format"] == "coco"
    warnings = [i for i in result["issues"] if i["level"] == "warning"]
    assert len(warnings) == 1
    assert "plotC_0_0.jpg" in warnings[0]["message"]
    assert [i for i in result["issues"] if i["level"] == "error"] == []
    assert result["is_valid"] is True


def test_reported_subjects_are_the_ones_present_in_the_label_files(tmp_path: Path):
    """The subject list is a census of every label file's contents, not a reading of the
    registry: a subject declared but never annotated is absent, and a subject annotated only in
    the last file is present."""
    from tcip_mcp.class_registry import ClassRegistry, Subject, write_registry

    root = tmp_path / "ds"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    write_registry(root / "classes.json", ClassRegistry(subjects=(
        Subject(name="catkin", description="a hazelnut catkin"),
        Subject(name="leaf", description="a leaf"),
        Subject(name="bush", description="a whole plant"),
        Subject(name="nut", description="a nut, never annotated here"),
    )))
    by_stem = {
        "plotA_0_0": ["catkin"],
        "plotB_0_0": ["catkin", "leaf"],
        "plotC_0_0": ["bush"],
    }
    for stem, subjects in by_stem.items():
        _write_image(root / "images" / DATE / f"{stem}.jpg", 96, 64)
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject=s, geometry=BBox(11, 7, 39, 51)) for s in subjects],
            96, 64,
        )

    result = validate_data_quality(str(root))

    assert result["format"] == "json"
    assert result["subjects"] == ["bush", "catkin", "leaf"]
    assert result["is_valid"] is True
