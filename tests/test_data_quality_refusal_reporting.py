"""scripts/doctor.py's check_data_quality: what it reports, and what it must never quietly claim.

Folded in from the retired per-file quality tool. Two standing facts the caller relies on.
First, a label store whose format the detector refused is a distinct finding from a dataset that
simply has no labels of that shape. Second, the finding's own vocabulary is load bearing: a
warning and an error are not interchangeable.
"""

from __future__ import annotations

import json
from pathlib import Path

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox

from scripts import doctor

DATE = "2-11-26"


def _write_image(path: Path, width: int, height: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(90, 120, 60)).save(path)


def _check(root: Path) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    doctor.check_data_quality(root, findings)
    return findings


def test_an_undetectable_label_store_is_an_error_naming_the_format_failure(tmp_path: Path):
    """Labels present but undetectable is an error per file, naming the detection failure; a
    dataset with no annotations dir at all yields no finding from this check at all."""
    root = tmp_path / "undetectable"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    for stem in ("plotA_0_0", "plotA_0_1"):
        _write_image(root / "images" / DATE / f"{stem}.jpg", 96, 64)
        (labels_dir / f"{stem}.json").write_text(
            json.dumps({"shapes": [{"label": "bud", "points": [[3, 5], [40, 52]]}]}),
            encoding="utf-8",
        )

    findings = _check(root)
    assert len(findings) == 2
    assert all(level == "error" and "cannot determine annotation format" in msg
              for level, msg in findings)

    bare = tmp_path / "unlabelled"
    for stem in ("plotA_0_0", "plotA_0_1"):
        _write_image(bare / "images" / DATE / f"{stem}.jpg", 96, 64)

    assert _check(bare) == []


def test_a_label_with_no_matching_image_is_an_error(tmp_path: Path):
    """An orphan label is an error-level finding."""
    root = tmp_path / "ds"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    for stem in ("plotA_0_0", "plotB_0_0"):
        _write_image(root / "images" / DATE / f"{stem}.jpg", 96, 64)
    for stem in ("plotA_0_0", "plotB_0_0", "plotZ_9_9"):
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="bud", geometry=BBox(11, 7, 39, 51))],
            96, 64,
        )

    findings = _check(root)

    errors = [msg for level, msg in findings if level == "error"]
    assert len(errors) == 1
    assert "plotZ_9_9" in errors[0] and "no matching image" in errors[0]


def test_a_coco_image_missing_from_the_images_dir_is_a_warning(tmp_path: Path):
    """A COCO entry pointing at an absent image file is a warning, never an error."""
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
        "categories": [{"id": 1, "name": "bud"}],
    }), encoding="utf-8")

    findings = _check(root)

    assert [level for level, _ in findings] == ["warn"]
    assert "plotC_0_0.jpg" in findings[0][1]


def test_format_is_decided_per_file_not_once_for_the_whole_directory(tmp_path: Path):
    """A COCO-shaped file sorting first must not make every other file in the same directory get
    silently parsed as COCO too, which would hide a real defect (here, an orphan per-image label)
    behind a report of nothing wrong."""
    root = tmp_path / "ds"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    _write_image(root / "images" / DATE / "plotA_0_0.jpg", 96, 64)

    (labels_dir / "0_coco.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "plotA_0_0.jpg", "width": 96, "height": 64}],
        "annotations": [],
        "categories": [{"id": 1, "name": "bud"}],
    }), encoding="utf-8")
    json_io.write_annotations(
        labels_dir / "1_orphan.json",
        [Annotation(subject="bud", geometry=BBox(1, 1, 8, 8))], 100, 100,
    )

    findings = _check(root)

    errors = [msg for level, msg in findings if level == "error"]
    assert len(errors) == 1
    assert "1_orphan" in errors[0] and "no matching image" in errors[0]


def test_an_empty_label_not_confirmed_negative_is_an_error(tmp_path: Path):
    """A platform-written empty document is not a zero-byte file, so a size check never catches
    it; an empty label with no human confirmation is unannotated, not a negative."""
    root = tmp_path / "ds"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    _write_image(root / "images" / DATE / "plotA_0_0.jpg", 96, 64)
    json_io.write_annotations(labels_dir / "plotA_0_0.json", [], 96, 64, keep_empty=True)

    findings = _check(root)

    errors = [msg for level, msg in findings if level == "error"]
    assert len(errors) == 1
    assert "plotA_0_0" in errors[0] and "confirmed negative" in errors[0]


def test_a_confirmed_negative_empty_label_stays_clean(tmp_path: Path):
    """The rail this suppression exists for: a human's Complete-with-nothing must not be flagged
    as though nobody had looked."""
    from tcip_mcp.dataset_layout import replace_image_status_store, status_bucket, status_records

    root = tmp_path / "ds"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    _write_image(root / "images" / DATE / "plotA_0_0.jpg", 96, 64)
    json_io.write_annotations(labels_dir / "plotA_0_0.json", [], 96, 64, keep_empty=True)
    replace_image_status_store(root, {
        status_bucket("bud", DATE): status_records(
            {"plotA_0_0.jpg": "negative"}, recorded_by="user:breeder"),
    })

    assert _check(root) == []


def test_a_malformed_root_label_candidate_is_reported_instead_of_discarded(tmp_path: Path):
    """A root-level candidate whose format cannot be determined is a present label file, not
    evidence the dataset carries none; it must be counted and flagged, not silently dropped by a
    caught detection error."""
    root = tmp_path / "ds"
    _write_image(root / "images" / DATE / "plotA_0_0.jpg", 96, 64)
    (root / "annotations.json").write_text(json.dumps({"foo": "bar"}), encoding="utf-8")

    findings = _check(root)

    errors = [msg for level, msg in findings if level == "error"]
    assert len(errors) == 1
    assert "annotations.json" in errors[0]


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
        "images": [], "annotations": [], "categories": [{"id": 1, "name": "bud"}],
    }), encoding="utf-8")

    findings = _check(root)

    errors = [msg for level, msg in findings if level == "error"]
    assert len(errors) == 2
    assert sum("plotA_0_0" in e for e in errors) == 1
    assert sum("plotB_0_0" in e for e in errors) == 1


def test_an_npz_capture_confirmed_negative_is_recognized(tmp_path: Path):
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
        status_bucket("bud", DATE): status_records(
            {"plotA_0_0.npz": "negative"}, recorded_by="user:breeder"),
    })

    assert _check(root) == []


def test_an_undecodable_label_is_a_finding_beside_a_readable_json_and_a_readable_coco_file(
    tmp_path: Path,
):
    """Coverage, not a guard: an undecodable document is already refused by ``detect_format``'s
    own decode before the per-format read this test's fixture exercises is ever reached, so this
    passes unchanged whichever reader the json branch calls. It records that the refusal still
    surfaces as a per-file finding beside a readable json and coco candidate in the same
    directory, never propagating out of the walk."""
    root = tmp_path / "ds"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    _write_image(root / "images" / DATE / "plotA_0_0.jpg", 96, 64)
    _write_image(root / "images" / DATE / "plotC_0_0.jpg", 96, 64)

    json_io.write_annotations(
        labels_dir / "plotA_0_0.json",
        [Annotation(subject="bud", geometry=BBox(11, 7, 39, 51))], 96, 64,
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

    findings = _check(root)

    errors = [msg for level, msg in findings if level == "error"]
    assert len(errors) == 1
    assert "plotB_0_0" in errors[0] and "will not read" in errors[0]
    assert [level for level, _ in findings if level == "warn"] == []
