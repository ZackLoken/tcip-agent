"""tcip doctor's check_data_quality: what it reports, and what it must never quietly claim.

Two standing facts the caller relies on. First, a label document the one per-image reader refuses
is a named finding for that file, never a reason to stop looking at the others. Second, the
finding's own vocabulary is load bearing: a warning and an error are not interchangeable.
"""

from __future__ import annotations

import json
from pathlib import Path

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox

from tcip_mcp.cli import doctor

DATE = "2-11-26"

COCO = {
    "images": [{"id": 1, "file_name": "plotA_0_0.jpg", "width": 96, "height": 64}],
    "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [11, 7, 28, 44]}],
    "categories": [{"id": 1, "name": "bud"}],
}


def _write_image(path: Path, width: int, height: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(90, 120, 60)).save(path)


def _check(root: Path) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    doctor.check_data_quality(root, findings)
    return findings


def _errors(findings: list[tuple[str, str]]) -> list[str]:
    return [msg for level, msg in findings if level == "error"]


def test_an_unreadable_label_store_is_an_error_per_file(tmp_path: Path):
    """Labels present but of a shape the reader refuses are an error per file, naming the refusal;
    a dataset with no annotations dir at all yields no finding from this check at all."""
    root = tmp_path / "unreadable"
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
    assert all(level == "error" and "label file will not read" in msg
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

    errors = _errors(_check(root))

    assert len(errors) == 1
    assert "plotZ_9_9" in errors[0] and "no matching image" in errors[0]


def test_a_coco_document_at_a_label_path_is_an_error_naming_the_import(tmp_path: Path):
    """A dataset-level COCO sitting where an image's label document belongs is refused by the one
    reader, and the finding carries the reader's own remedy."""
    root = tmp_path / "ds"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    _write_image(root / "images" / DATE / "plotA_0_0.jpg", 96, 64)
    (labels_dir / "plotA_0_0.json").write_text(json.dumps(COCO), encoding="utf-8")

    findings = _check(root)

    assert [level for level, _ in findings] == ["error"]
    assert "plotA_0_0" in findings[0][1] and "import_coco" in findings[0][1]


def test_each_file_is_read_on_its_own_not_once_for_the_whole_directory(tmp_path: Path):
    """A COCO-shaped file sorting first must not decide how every other file in the same
    directory is read, which would hide a real defect (here, an orphan per-image label) behind
    the finding the COCO file earns."""
    root = tmp_path / "ds"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    _write_image(root / "images" / DATE / "plotA_0_0.jpg", 96, 64)

    (labels_dir / "0_coco.json").write_text(json.dumps(COCO), encoding="utf-8")
    json_io.write_annotations(
        labels_dir / "1_orphan.json",
        [Annotation(subject="bud", geometry=BBox(1, 1, 8, 8))], 100, 100,
    )

    errors = _errors(_check(root))

    assert any("0_coco" in e and "import_coco" in e for e in errors)
    assert any("1_orphan" in e and "no matching image" in e for e in errors)


def test_an_undecodable_first_label_hides_no_later_finding(tmp_path: Path):
    """The census reads no document, so an undecodable label sorting first is one finding and
    the orphan after it is still reported."""
    root = tmp_path / "ds"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    _write_image(root / "images" / DATE / "0_bad.jpg", 96, 64)
    (labels_dir / "0_bad.json").write_bytes(b"{not json")
    json_io.write_annotations(
        labels_dir / "zzz_orphan.json",
        [Annotation(subject="bud", geometry=BBox(1, 1, 8, 8))], 96, 64,
    )

    errors = _errors(_check(root))

    assert any("0_bad" in e and "will not read" in e for e in errors)
    assert any("zzz_orphan" in e and "no matching image" in e for e in errors)


def test_an_empty_label_not_confirmed_negative_is_an_error(tmp_path: Path):
    """A platform-written empty document is not a zero-byte file, so a size check never catches
    it; an empty label with no human confirmation is unannotated, not a negative."""
    root = tmp_path / "ds"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    _write_image(root / "images" / DATE / "plotA_0_0.jpg", 96, 64)
    json_io.write_annotations(labels_dir / "plotA_0_0.json", [], 96, 64, keep_empty=True)

    errors = _errors(_check(root))

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


def test_a_coco_at_the_dataset_root_is_not_one_of_the_datasets_labels(tmp_path: Path):
    """An external COCO document left at the dataset root is not a label store: the per-image
    tree's own findings are reported and nothing is reported about the root document."""
    root = tmp_path / "ds"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    for stem in ("plotA_0_0", "plotB_0_0"):
        _write_image(root / "images" / DATE / f"{stem}.jpg", 96, 64)
        json_io.write_annotations(labels_dir / f"{stem}.json", [], 96, 64, keep_empty=True)
    (root / "annotations.json").write_text(json.dumps(COCO), encoding="utf-8")

    errors = _errors(_check(root))

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


def test_an_undecodable_label_is_a_finding_beside_a_readable_json_file(tmp_path: Path):
    """Coverage, not a guard: the refusal surfaces as a per-file finding beside a readable
    document in the same directory, never propagating out of the walk."""
    root = tmp_path / "ds"
    labels_dir = root / "annotations" / DATE
    labels_dir.mkdir(parents=True)
    _write_image(root / "images" / DATE / "plotA_0_0.jpg", 96, 64)
    _write_image(root / "images" / DATE / "plotB_0_0.jpg", 96, 64)

    json_io.write_annotations(
        labels_dir / "plotA_0_0.json",
        [Annotation(subject="bud", geometry=BBox(11, 7, 39, 51))], 96, 64,
    )
    (labels_dir / "plotB_0_0.json").write_bytes(b"{not json")

    findings = _check(root)

    errors = _errors(findings)
    assert len(errors) == 1
    assert "plotB_0_0" in errors[0] and "will not read" in errors[0]
    assert [level for level, _ in findings if level == "warn"] == []
