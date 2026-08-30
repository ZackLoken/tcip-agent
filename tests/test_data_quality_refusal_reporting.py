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


def test_a_store_mixing_shapes_is_reported_invalid_even_when_a_coco_file_sorts_first(
    tmp_path: Path,
):
    """Format is decided per label file, not once for the whole dataset: a COCO-shaped file
    sorting first must not make every other file in the same directory get silently parsed as
    COCO too, which would hide a real defect (here, an orphan per-image label) behind a report
    of nothing wrong."""
    root = tmp_path / "ds"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    _write_image(root / "images" / DATE / "plotA_0_0.jpg", 96, 64)

    (labels_dir / "0_coco.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "plotA_0_0.jpg", "width": 96, "height": 64}],
        "annotations": [],
        "categories": [{"id": 1, "name": "catkin"}],
    }), encoding="utf-8")
    json_io.write_annotations(
        labels_dir / "1_orphan.json",
        [Annotation(subject="catkin", geometry=BBox(1, 1, 8, 8))], 100, 100,
    )

    result = validate_data_quality(str(root))

    # format names every distinct shape present, not the first-sorted file's shape alone.
    assert sorted(result["format"]) == ["coco", "json"]
    errors = [i for i in result["issues"] if i["level"] == "error"]
    assert len(errors) == 1
    assert Path(errors[0]["file"]).stem == "1_orphan"
    assert result["is_valid"] is False


def test_an_empty_label_not_confirmed_negative_is_an_error_that_denies_validity(tmp_path: Path):
    """A platform-written empty document is not a zero-byte file, so a size check never catches
    it; an empty label with no human confirmation is unannotated, not a negative, and reporting
    it as a mere warning would let is_valid stay true over exactly the state that corrupts
    training."""
    root = tmp_path / "ds"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    _write_image(root / "images" / DATE / "plotA_0_0.jpg", 96, 64)
    json_io.write_annotations(labels_dir / "plotA_0_0.json", [], 96, 64, keep_empty=True)

    result = validate_data_quality(str(root))

    errors = [i for i in result["issues"] if i["level"] == "error"]
    assert len(errors) == 1
    assert Path(errors[0]["file"]).stem == "plotA_0_0"
    assert "confirmed negative" in errors[0]["message"]
    assert result["is_valid"] is False


def test_a_confirmed_negative_empty_label_stays_valid(tmp_path: Path):
    """The rail this suppression exists for: a human's Complete-with-nothing must not be flagged
    as though nobody had looked."""
    from tcip_mcp.dataset_layout import replace_image_status_store, status_bucket, status_records

    root = tmp_path / "ds"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    _write_image(root / "images" / DATE / "plotA_0_0.jpg", 96, 64)
    json_io.write_annotations(labels_dir / "plotA_0_0.json", [], 96, 64, keep_empty=True)
    replace_image_status_store(root, {
        status_bucket("catkin", DATE): status_records(
            {"plotA_0_0.jpg": "negative"}, recorded_by="user:breeder"),
    })

    result = validate_data_quality(str(root))

    assert result["issues"] == []
    assert result["is_valid"] is True


def test_a_malformed_root_label_candidate_is_reported_instead_of_discarded(tmp_path: Path):
    """A root-level candidate whose format cannot be determined is a present label file, not
    evidence the dataset carries none; it must be counted and flagged, not silently dropped by a
    caught detection error."""
    root = tmp_path / "ds"
    _write_image(root / "images" / DATE / "plotA_0_0.jpg", 96, 64)
    (root / "annotations.json").write_text(json.dumps({"foo": "bar"}), encoding="utf-8")

    result = validate_data_quality(str(root))

    assert result["total_labels"] == 1
    errors = [i for i in result["issues"] if i["level"] == "error"]
    assert len(errors) == 1
    assert Path(errors[0]["file"]).name == "annotations.json"
    assert result["is_valid"] is False


def test_a_root_coco_candidate_sits_beside_the_per_image_tree_not_in_place_of_it(tmp_path: Path):
    """A root-level assembled label document is one more present label, never a replacement: two
    unconfirmed empty per-image labels stay reported even when a root candidate is also present."""
    root = tmp_path / "ds"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    for stem in ("plotA_0_0", "plotB_0_0"):
        _write_image(root / "images" / DATE / f"{stem}.jpg", 96, 64)
        json_io.write_annotations(labels_dir / f"{stem}.json", [], 96, 64, keep_empty=True)
    (root / "annotations.json").write_text(json.dumps({
        "images": [], "annotations": [], "categories": [{"id": 1, "name": "catkin"}],
    }), encoding="utf-8")

    result = validate_data_quality(str(root))

    assert result["total_labels"] == 3
    errors = [i for i in result["issues"] if i["level"] == "error"]
    assert {Path(e["file"]).stem for e in errors} == {"plotA_0_0", "plotB_0_0"}
    assert result["is_valid"] is False


def test_an_npz_captures_confirmed_negative_is_recognized(tmp_path: Path):
    """The confirmed-negative name is resolved through the layout's own extension set, not the
    six-extension list an ``.npz`` capture falls outside of."""
    from tcip_mcp.dataset_layout import replace_image_status_store, status_bucket, status_records

    root = tmp_path / "ds"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    (root / "images" / DATE).mkdir(parents=True)
    (root / "images" / DATE / "plotA_0_0.npz").write_bytes(b"\x00")
    json_io.write_annotations(labels_dir / "plotA_0_0.json", [], 8, 8, keep_empty=True)
    replace_image_status_store(root, {
        status_bucket("catkin", DATE): status_records(
            {"plotA_0_0.npz": "negative"}, recorded_by="user:breeder"),
    })

    result = validate_data_quality(str(root))

    assert not any("confirmed negative" in i["message"] for i in result["issues"])
    assert result["total_images"] == 1
    assert not any("No matching image" in i["message"] for i in result["issues"])


def test_an_undecodable_label_is_a_finding_beside_a_readable_json_and_a_readable_coco_file(
    tmp_path: Path,
):
    """The reader refusal a genuinely undecodable document raises (as opposed to a decodable but
    unrecognized shape, covered above) must surface as a per-file error finding, never propagate
    out of the walk, and must not stop the readable json and coco candidates in the same
    directory from being read and reported normally."""
    root = tmp_path / "ds"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    _write_image(root / "images" / DATE / "plotA_0_0.jpg", 96, 64)
    _write_image(root / "images" / DATE / "plotC_0_0.jpg", 96, 64)

    json_io.write_annotations(
        labels_dir / "plotA_0_0.json",
        [Annotation(subject="catkin", geometry=BBox(11, 7, 39, 51))], 96, 64,
    )
    (labels_dir / "plotB_0_0.json").write_bytes(b"{not json")
    (root / "annotations.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "plotC_0_0.jpg", "width": 96, "height": 64}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 10, 10], "area": 100,
             "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "leaf"}],
    }), encoding="utf-8")

    result = validate_data_quality(str(root))

    assert result["total_labels"] == 3
    assert sorted(result["format"]) == ["coco", "json"]
    errors = [i for i in result["issues"] if i["level"] == "error"]
    assert len(errors) == 1
    assert Path(errors[0]["file"]).stem == "plotB_0_0"
    assert "will not read" in errors[0]["message"]
    assert [i for i in result["issues"] if i["level"] == "warning"] == []
    assert result["subjects"] == ["catkin", "leaf"]
    assert result["is_valid"] is False


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
