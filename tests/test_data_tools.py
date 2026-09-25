"""Tests for data management tools."""

from __future__ import annotations

import pytest
import tcip_store as ts
from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox

from pathlib import Path

from tcip_mcp.cli import doctor
from tcip_mcp.pipelines.data.selection import (
    Sample, Selection, read_selection, selection_key, write_selection,
)
from tcip_mcp.tools.data_tools import scan_dataset, draw_splits


def _quality_findings(root) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    doctor.check_data_quality(Path(root), findings, census=doctor._census(Path(root), findings, set()))
    return findings


def test_scan_dataset(data_dir: Path):
    result = scan_dataset(str(data_dir))
    assert result["image_count"] == 3
    assert result["labels_count"] == 3
    assert result["paired_images"] == 3
    assert result["unlabeled_images"] == 0


def test_scan_dataset_not_found():
    result = scan_dataset("/nonexistent/path")
    assert "error" in result


def test_doctor_check_data_quality(data_dir: Path):
    findings = _quality_findings(data_dir)
    assert not [f for f in findings if f[0] == "error"]


def test_doctor_check_data_quality_missing_dir(tmp_path: Path):
    # _scan_dataset's own rglob over a nonexistent path yields nothing rather than raising; a
    # missing project root is doctor.main()'s own upfront refusal, not this check's job.
    missing = tmp_path / "nonexistent" / "xyz123"
    findings = _quality_findings(missing)
    assert not [f for f in findings if f[0] == "error"]


def test_scan_dataset_reports_a_reserved_stem_the_census_still_counted(tmp_path: Path):
    """The census walks with a raw glob and counts a label named like a bucket's own provenance
    stamp, unlike every bucket walk through prediction_documents; reserved_name_labels names it so
    a caller does not read the difference as a disagreement."""
    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)
    labels_dir = root / "annotations" / "2-11-26"
    labels_dir.mkdir(parents=True)
    json_io.write_annotations(
        labels_dir / "operating_point.json",
        [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))], 32, 32,
    )
    reserved_label = str(labels_dir / "operating_point.json")

    scan_result = scan_dataset(str(root))

    assert scan_result["reserved_name_labels"] == [reserved_label]


def test_scan_dataset_reports_a_reserved_stem_image_with_no_label(tmp_path: Path):
    """An image whose own stem is reserved for a bucket's own provenance stamp must be named,
    not folded into unlabeled_images with no signal that its label can never be read through
    any bucket walk."""
    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)
    (images_dir / "operating_point.jpg").write_bytes(b"\xff\xd8\xff")
    (images_dir / "ordinary.jpg").write_bytes(b"\xff\xd8\xff")
    reserved_image = str(images_dir / "operating_point.jpg")

    scan_result = scan_dataset(str(root))

    assert scan_result["reserved_name_images"] == [reserved_image]
    assert scan_result["unlabeled_images"] == 2


def test_scan_dataset_drops_a_cleared_bucket_from_the_prediction_count(tmp_path: Path):
    """A document ``clear_prediction_bucket`` has moved into the cleared archive is never counted
    as a live prediction: the census walks ``predictions/`` with a raw ``rglob``, which would
    otherwise see it exactly like a live bucket's own document."""
    from tcip_mcp.dataset_layout import cleared_prediction_dir, prediction_dir

    root = tmp_path / "ds"
    live = prediction_dir(root, "modelA", "2-11-26")
    live.mkdir(parents=True)
    json_io.write_annotations(
        live / "imgA.json", [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))], 32, 32,
    )

    cleared = cleared_prediction_dir(root, "modelB", "2-11-26", "20260906T120000Z")
    cleared.mkdir(parents=True)
    json_io.write_annotations(
        cleared / "imgB.json", [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))], 32, 32,
    )

    scan_result = scan_dataset(str(root))

    assert scan_result["predictions_count"] == 1


def _add_extra_bud_groups(data_dir: Path, count: int) -> None:
    """Adds ``count`` more single-tile foreground groups under ``data_dir``'s own date, for its
    own subject, without touching the shared ``data_dir`` fixture other tests depend on: a
    manifest write needs at least four foreground groups to clear ``draw_splits``' floor, one
    more than the fixture's own three."""
    from PIL import Image

    images_dir = data_dir / "images" / "2-11-26"
    labels_dir = data_dir / "annotations" / "2-11-26"
    for i in range(count):
        stem = f"extra_{i:03d}"
        Image.new("RGB", (640, 480), color=(128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="bud", geometry=BBox(288, 216, 352, 264))], 640, 480,
        )

def test_draw_splits_basic(data_dir: Path, tmp_path: Path):
    _add_extra_bud_groups(data_dir, 1)
    out = tmp_path / "manifests"
    # The fixture's 4 stems (img_001..003 plus one grown group) are 4 distinct foreground
    # groups, exactly the manifest floor (one each for train/val, two for calibration).
    result = draw_splits(str(data_dir), output_path=str(out), subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in result, result
    assert result["total_stems"] == 4
    assert result["groups"] == 4
    assert sum(result["splits"].values()) == 4
    assert result["stratified"] is True

    drawn = read_selection(out)
    assert drawn.counts() == {k: v for k, v in result["splits"].items()}
    assert all(drawn.counts()[side] for side in ("train", "val", "calibration"))
    assert drawn.subject == "bud"
    assert drawn.attribute is None
    assert drawn.dataset_fingerprint is not None
    for sample in drawn.samples:
        assert Path(sample.source).is_file()
        assert Path(sample.ground_truth).is_file()
        assert sample.ground_truth_digest


def test_draw_splits_refuses_a_version_refused_subject_registry_as_an_error(data_dir: Path, tmp_path: Path):
    from tcip_mcp.dataset_layout import subject_registry_key

    _add_extra_bud_groups(data_dir, 1)
    ts.put_blob(
        subject_registry_key(data_dir), ts.RECORD_JSON.encode({"schema_version": 99})
    )
    result = draw_splits(str(data_dir), output_path=str(tmp_path / "manifests"), subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" in result, result
    assert "schema_version" in result["error"]


def test_draw_splits_stats_only_admits_a_nonzero_calibration_ratio(data_dir: Path):
    """A stats-only call (no output_path) may pass any calibration_ratio; only writing a
    selection requires a non-zero one."""
    result = draw_splits(str(data_dir), train_ratio=0.7, val_ratio=0.2, calibration_ratio=0.1)
    assert "error" not in result, result
    assert result["splits"]["calibration"] > 0


def test_draw_splits_reports_an_unreadable_label_by_name(data_dir: Path, tmp_path: Path):
    """A present, unreadable label among the candidates is an error naming the file, never a
    raise through the tool boundary."""
    bad = next((data_dir / "annotations" / "2-11-26").glob("*.json"))
    bad.write_bytes(b"{not json")

    result = draw_splits(str(data_dir), output_path=str(tmp_path / "manifests"), subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    assert "error" in result
    assert str(bad) in result["error"]


def test_draw_splits_reports_an_unreadable_label_sorted_last(
    data_dir: Path, tmp_path: Path,
):
    """A corrupt label reached last in sort order is caught by the same per-stem admission read
    as the first-sorted case above, regardless of where in the candidate order it falls, and
    answers the same error dict, never a raw raise."""
    bad = sorted((data_dir / "annotations" / "2-11-26").glob("*.json"))[-1]
    bad.write_bytes(b"{not json")

    result = draw_splits(str(data_dir), output_path=str(tmp_path / "manifests"), subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    assert "error" in result
    assert str(bad) in result["error"]


def test_draw_splits_writes_nothing_when_a_confirmed_negative_will_not_read(
    data_dir: Path, tmp_path: Path,
):
    """An unreadable label on an image a human confirmed negative refuses the whole draw and
    leaves no selection behind: a partial record would claim a partition nobody drew, and the
    confirmation cannot be checked against a document that will not parse."""
    from tcip_mcp.dataset_layout import record_image_statuses, status_bucket

    bad = data_dir / "annotations" / "2-11-26" / "img_002.json"
    record_image_statuses(data_dir, status_bucket("bud", "2-11-26"),
                          {bad.with_suffix(".jpg").name: "negative"}, recorded_by="user:tester")
    bad.write_bytes(b"{not json")
    out = tmp_path / "selection"

    result = draw_splits(str(data_dir), output_path=str(out), subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    assert "error" in result
    assert str(bad) in result["error"]
    assert not ts.exists(selection_key(out))


def test_draw_splits_stats_only_reports_an_unreadable_first_sorted_label(data_dir: Path):
    """A stats-only call (no output_path) draws no subject-scoped admission at all: its own scan
    raises on the first-sorted candidate, the same as scan_dataset would."""
    bad = data_dir / "annotations" / "2-11-26" / "img_001.json"
    bad.write_bytes(b"{not json")

    result = draw_splits(str(data_dir))

    assert "error" in result
    assert str(bad) in result["error"]


def test_draw_splits_stats_only_reports_an_unreadable_label_during_stratification(data_dir: Path):
    """A stats-only call still reads every stem's label to count its annotations for stratified
    balancing, so a corrupt label reached after a readable first candidate is an error naming the
    file, not a raw raise."""
    bad = data_dir / "annotations" / "2-11-26" / "img_003.json"
    bad.write_bytes(b"{not json")

    result = draw_splits(str(data_dir))

    assert "error" in result
    assert str(bad) in result["error"]


def test_draw_splits_manifest_answers_an_ambiguous_image_stem_as_an_error(tmp_path: Path):
    """A raw file colliding with a band group's own canonical stem is an error naming the
    directory, never a raise through the tool boundary, the same contract the unreadable-label
    cases above state."""
    import numpy as np

    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)
    (root / "annotations" / "2-11-26").mkdir(parents=True)

    band_a, band_b = images_dir / "plotA_B1.npy", images_dir / "plotA_B2.npy"
    np.save(band_a, np.zeros((4, 4), dtype=np.uint8))
    np.save(band_b, np.zeros((4, 4), dtype=np.uint8))
    write_band_group_manifest(images_dir, "plotA", {"B1": band_a, "B2": band_b})
    (images_dir / "plotA.jpg").write_bytes(b"\xff\xd8\xff")

    result = draw_splits(str(root), output_path=str(tmp_path / "manifests"), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    assert "error" in result
    assert "plotA" in result["error"]


def test_draw_splits_stats_only_answers_an_ambiguous_image_stem_as_an_error(tmp_path: Path):
    """The stats-only branch (no output_path) answers the identical stem collision the
    selection-writing branch already reports as an error, never a raw raise: both branches route
    their image census through ``_scan_dataset``."""
    import numpy as np

    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)

    band_a, band_b = images_dir / "plotA_B1.npy", images_dir / "plotA_B2.npy"
    np.save(band_a, np.zeros((4, 4), dtype=np.uint8))
    np.save(band_b, np.zeros((4, 4), dtype=np.uint8))
    write_band_group_manifest(images_dir, "plotA", {"B1": band_a, "B2": band_b})
    (images_dir / "plotA.jpg").write_bytes(b"\xff\xd8\xff")

    result = draw_splits(str(root))

    assert "error" in result
    assert "plotA" in result["error"]


def test_scan_dataset_answers_a_newer_written_bandgroup_manifest_as_an_error(tmp_path: Path):
    """A ``.bandgroup`` manifest above this reader's ceiling propagates as
    ``tcip_store.SchemaVersionRefused`` out of ``list_logical_images``, uncaught by design;
    ``scan_dataset`` must answer that as a named error rather than letting it raise through the
    tool boundary, the same contract it already holds for an ambiguous stem."""
    from tcip_store.file_backend import FileBackend
    import tcip_store as ts

    from tcip_mcp.pipelines.data.band_groups import band_group_manifest_key

    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)

    key = band_group_manifest_key(images_dir, "plotA")
    path = FileBackend().path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ts.RECORD_JSON.encode(
        {"schema_version": 2, "bands": {"B1": "plotA_B1.npy", "B2": "plotA_B2.npy"}}))

    result = scan_dataset(str(root))

    assert "error" in result
    assert "schema_version 2, above the 1 this reader knows" in result["error"]


def test_scan_dataset_answers_an_ambiguous_image_stem_as_an_error(tmp_path: Path):
    """``scan_dataset`` answers the same stem collision as an error rather than letting
    ``AmbiguousImageStem`` escape uncaught through the tool boundary."""
    import numpy as np

    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)

    band_a, band_b = images_dir / "plotA_B1.npy", images_dir / "plotA_B2.npy"
    np.save(band_a, np.zeros((4, 4), dtype=np.uint8))
    np.save(band_b, np.zeros((4, 4), dtype=np.uint8))
    write_band_group_manifest(images_dir, "plotA", {"B1": band_a, "B2": band_b})
    (images_dir / "plotA.jpg").write_bytes(b"\xff\xd8\xff")

    result = scan_dataset(str(root))

    assert "error" in result
    assert "plotA" in result["error"]


def _add_extra_leaf_groups(images_dir: Path, labels_dir: Path, count: int) -> None:
    """Adds ``count`` more plain single-tile foreground groups (subject ``leaf``) beside a
    band-group fixture, so a manifest write over it clears the four-foreground-group floor."""
    from PIL import Image

    letters = "BCDEFGH"
    for i in range(count):
        stem = f"plot{letters[i]}"
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
        )


def test_draw_splits_refuses_an_incomplete_band_group_before_writing(tmp_path: Path):
    """A band group whose manifest names a missing sibling is refused before the selection is
    written, never a raise through the tool after it lands."""
    import numpy as np

    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)
    labels_dir = root / "annotations" / "2-11-26"
    labels_dir.mkdir(parents=True)

    band_g, band_r = images_dir / "plotA_G.npy", images_dir / "plotA_R.npy"
    np.save(band_g, np.zeros((4, 4), dtype=np.uint8))
    write_band_group_manifest(images_dir, "plotA", {"G": band_g, "R": band_r})  # R never created
    json_io.write_annotations(
        labels_dir / "plotA.json",
        [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
    )
    _add_extra_leaf_groups(images_dir, labels_dir, 3)
    out = tmp_path / "m"

    result = draw_splits(str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    assert "error" in result
    assert "plotA" in result["error"] and "R" in result["error"]
    assert not out.exists()


def test_place_logical_image_leaves_an_existing_destination_alone_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The leave-alone path is a cheap ``dst.exists()`` check in front of the store write, not a
    read-and-hash of the whole existing destination: a destination already present is skipped
    before ``put_blob_from_path`` is even attempted."""
    from tcip_mcp.pipelines import image_utils

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    source = src_dir / "img.jpg"
    source.write_bytes(b"\xff\xd8\xff")
    (dest_dir / "img.jpg").write_bytes(b"already there")

    calls: list = []
    real_put = image_utils.tcip_store.put_blob_from_path

    def _spy(key, src_path, **kwargs):
        calls.append(key)
        return real_put(key, src_path, **kwargs)

    monkeypatch.setattr(image_utils.tcip_store, "put_blob_from_path", _spy)

    name = image_utils.place_logical_image(
        source, dest_dir, copy_files=True,
        dest_key=lambda filename: image_utils.flat_image_key(dest_dir, filename),
    )

    assert name == "img.jpg"
    assert (dest_dir / "img.jpg").read_bytes() == b"already there"
    assert calls == []


def test_draw_splits_stats_only_carries_dataset_hash(data_dir: Path):
    """A stats-only call's answer identifies the labels it partitioned, the same as a manifest
    call's own per-date record."""
    result = draw_splits(str(data_dir))
    assert "error" not in result
    assert result["dataset_hash"]
    assert result["dataset_hashes_by_date"] == {"2-11-26": result["dataset_hash"]}


def test_draw_splits_stats_only_over_two_dates_names_both_hashes_and_no_single_hash(
    tmp_path: Path,
):
    """A stats-only call over a tree with more than one labels directory names each date's own
    hash and carries no single dataset_hash, which would be blind to every other date's
    content: the same hash implementation the manifest write calls per date."""
    from PIL import Image

    root = tmp_path / "ds"
    for date, stems in (("2-11-26", ("a", "b")), ("2-12-01", ("c", "d"))):
        images_dir = root / "images" / date
        labels_dir = root / "annotations" / date
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        for stem in stems:
            Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
            json_io.write_annotations(
                labels_dir / f"{stem}.json",
                [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
            )

    result = draw_splits(str(root))

    assert "error" not in result, result
    assert result["dataset_hash"] is None
    assert set(result["dataset_hashes_by_date"]) == {"2-11-26", "2-12-01"}
    assert result["dataset_hashes_by_date"]["2-11-26"] != result["dataset_hashes_by_date"]["2-12-01"]

def test_draw_splits_bad_ratios(data_dir: Path):
    result = draw_splits(str(data_dir), train_ratio=0.5, val_ratio=0.5, calibration_ratio=0.5)
    assert "error" in result


def test_draw_splits_train_val_calibration_not_summing_to_one_names_all_three(data_dir: Path):
    """The sum-check message names all three standing constraints, not just the raw sum."""
    result = draw_splits(str(data_dir), train_ratio=0.7, val_ratio=0.2, calibration_ratio=0.2)
    assert "error" in result
    assert "calibration_ratio" in result["error"]
    assert "train_ratio" in result["error"] and "val_ratio" in result["error"]


def test_draw_splits_manifest_write_refuses_a_zero_calibration_ratio(tmp_path: Path):
    """A selection's calibration side is the universe every calibration under it draws from, so
    writing one states a non-zero calibration_ratio; the keyword names the missing input."""
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"

    result = draw_splits(str(root), output_path=str(out), subject="bud")

    assert "error" in result
    assert "calibration_ratio" in result["error"]
    assert not out.exists()


def test_draw_splits_selection_carries_all_three_split_sides(tmp_path: Path):
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"

    result = draw_splits(str(root), output_path=str(out), seed=1, subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    assert "error" not in result, result
    counts = read_selection(out).counts()
    assert set(counts) == {"train", "val", "calibration"}
    assert all(counts.values())


def test_draw_splits_floor_refuses_before_any_write_regardless_of_stratify_foreground(
    tmp_path: Path,
):
    """The foreground floor is over the draw's own subject-scoped counter on every selection
    draw, whether or not stratify_foreground toggles the balancing pass: a tree with only three
    foreground groups refuses before anything is written, with stratify_foreground off."""
    root = _multi_source_dataset(tmp_path / "ds", prefixes=("srcA", "srcB", "srcC"))
    out = tmp_path / "m"

    result = draw_splits(str(root), output_path=str(out), subject="bud", seed=1,
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25,
                         stratify_foreground=False)

    assert "error" in result
    assert "foreground group" in result["error"]
    assert not out.exists()


def test_draw_splits_manifest_write_refuses_a_zero_ratio_on_any_side_by_name(tmp_path: Path):
    """Writing a selection requires all three ratios non-zero, refused by name naming the zero
    one, before the foreground floor is ever reached: no side is dropped by zeroing its ratio."""
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"

    result = draw_splits(str(root), output_path=str(out), subject="bud", seed=1,
                         train_ratio=0.75, val_ratio=0.0, calibration_ratio=0.25)

    assert "error" in result
    assert "val_ratio" in result["error"] and "must be non-zero" in result["error"]
    assert "foreground group" not in result["error"]  # never reaches the floor
    assert not out.exists()


def test_draw_splits_floor_ignores_a_groups_only_annotations_of_another_subject(tmp_path: Path):
    """A confirmed-negative-for-the-draws-subject group whose label file happens to carry
    another subject's annotation is not this draw's foreground: the subject-scoped counter
    reads it as zero, so three real ``leaf`` groups plus one such group still refuse (below the
    floor of four), the same tree an unscoped counter would have read as four and written."""
    from PIL import Image

    from tcip_mcp.subject_registry import SubjectRegistry, Subject, write_registry
    from tcip_mcp.dataset_layout import record_image_statuses, status_bucket

    root = tmp_path / "ds"
    date = "2-11-26"
    images_dir, labels_dir = root / "images" / date, root / "annotations" / date
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    write_registry(root / "subjects.json", SubjectRegistry(subjects=(
        Subject(name="leaf"), Subject(name="bud"),
    )))
    for stem in ("p1", "p2", "p3"):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
        )
    Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / "p4.jpg")
    json_io.write_annotations(
        labels_dir / "p4.json",
        [Annotation(subject="bud", geometry=BBox(4, 4, 12, 12))], 100, 80, keep_empty=True,
    )
    record_image_statuses(root, status_bucket("leaf", date), {"p4.jpg": "negative"},
                          recorded_by="user:tester")

    out = tmp_path / "m"
    result = draw_splits(str(root), output_path=str(out), subject="leaf", seed=1,
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    assert "error" in result
    assert "foreground group" in result["error"]
    assert not out.exists()


def _leaf_dataset_with_negatives(root: Path, date: str, n_foreground: int, n_negative: int) -> None:
    """``n_foreground`` single-tile ``leaf`` groups plus ``n_negative`` confirmed-negative
    images, all under one capture date."""
    from PIL import Image

    from tcip_mcp.dataset_layout import record_image_statuses, status_bucket

    images_dir, labels_dir = root / "images" / date, root / "annotations" / date
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_foreground):
        stem = f"{date}_fg{i}"
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
        )
    negative_names = []
    for i in range(n_negative):
        stem = f"{date}_bg{i}"
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(labels_dir / f"{stem}.json", [], 100, 80, keep_empty=True)
        negative_names.append(f"{stem}.jpg")
    record_image_statuses(root, status_bucket("leaf", date),
                          {n: "negative" for n in negative_names}, recorded_by="user:tester")


def test_draw_splits_calibration_side_holds_real_foreground_regardless_of_stratify_foreground(
    tmp_path: Path,
):
    """The minimum-foreground pass sees the draw's subject-scoped foreground counts on every
    selection draw, not only when stratify_foreground also balances by them: the calibration
    side's stated minimum of two foreground groups is met with real foreground even with
    balancing off, across every seed."""
    root = tmp_path / "ds"
    date = "2-11-26"
    _leaf_dataset_with_negatives(root, date, n_foreground=4, n_negative=6)

    for seed in range(1, 21):
        out = tmp_path / f"m{seed}"
        result = draw_splits(str(root), output_path=str(out), subject="leaf", seed=seed,
                             train_ratio=0.8, val_ratio=0.1, calibration_ratio=0.1,
                             stratify_foreground=False)
        assert "error" not in result, (seed, result)
        assert result["calibration_foreground_groups"] >= 2, (seed, result)

def _multi_source_dataset(root: Path, prefixes=("srcA", "srcB", "srcC", "srcD"), tiles=3) -> Path:
    from PIL import Image

    date = "2-11-26"
    images_dir = root / "images" / date
    images_dir.mkdir(parents=True)
    labels_dir = root / "annotations" / date
    labels_dir.mkdir(parents=True)
    for pref in prefixes:
        for t in range(tiles):
            stem = f"{pref}_{t}_0"
            Image.new("RGB", (64, 64), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
            json_io.write_annotations(labels_dir / f"{stem}.json",
                                      [Annotation(subject="bud", geometry=BBox(19, 13, 45, 51))],
                                      64, 64)
    return root


def test_draw_splits_groups_tiles_together(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import default_group_key

    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = draw_splits(str(root), output_path=str(out), seed=1, subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert result["groups"] == 4  # 4 source prefixes, not 12 tiles

    # No source prefix may appear in more than one split.
    seen: dict[str, str] = {}
    for sample in read_selection(out).samples:
        g = default_group_key(Path(sample.ground_truth).stem)
        assert seen.get(g, sample.side) == sample.side, f"group {g} spans splits"
        seen[g] = sample.side


def test_draw_splits_group_key_map_never_straddles(tmp_path: Path):
    """An agent-derived group_key_map (5 samples, 4 groups) is honored: the two same-group
    samples never land in different splits."""
    root = _multi_source_dataset(tmp_path / "ds", prefixes=("x", "y", "z", "w", "v"), tiles=1)
    out = tmp_path / "m"
    group_key_map = {
        "2-11-26/x_0_0": "gA", "2-11-26/y_0_0": "gA", "2-11-26/z_0_0": "gB",
        "2-11-26/w_0_0": "gC", "2-11-26/v_0_0": "gD",
    }
    result = draw_splits(str(root), output_path=str(out), seed=1,
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25,
                         group_by="tile_prefix", group_key_map=group_key_map, subject="bud")
    assert "error" not in result, result
    assert result["group_by"] == "explicit_map"

    drawn = read_selection(out)
    by_stem = {Path(s.ground_truth).stem: s for s in drawn.samples}
    assert by_stem["x_0_0"].side == by_stem["y_0_0"].side  # gA never straddles
    assert by_stem["x_0_0"].group == by_stem["y_0_0"].group == "gA"
    assert drawn.group_by == "explicit_map"


def test_draw_splits_unrecognized_group_by_refuses_without_writing(tmp_path: Path):
    """An unrecognized ``group_by`` must refuse loudly and write nothing, never fall back to
    ``GROUP_KEY_FNS.get(group_by, default_group_key)`` and mis-group a dataset silently."""
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = draw_splits(str(root), output_path=str(out), group_by="not_a_real_key", subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" in result
    assert not out.exists() or not (out / "selection.json").is_file()


def test_draw_splits_refuses_to_write_a_selection_with_no_subject(tmp_path: Path):
    """A selection with no subject would be a partition of images, not of a run's admissible
    samples; draw_splits refuses to write one rather than guessing what a run would admit."""
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = draw_splits(str(root), output_path=str(out),
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" in result and "subject" in result["error"]
    assert not out.exists()


def _two_date_collision_dataset(root: Path, subject: str) -> Path:
    """One stem name, ``shared``, present under two capture dates with different content, plus
    one more distinct stem per date so a selection drawn over this tree clears the foreground
    floor: a record keyed by bare stem could only ever hold one of the two ``shared`` images."""
    from PIL import Image

    for date, box_x in (("2-11-26", 4), ("2-12-01", 40)):
        images_dir = root / "images" / date
        labels_dir = root / "annotations" / date
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / "shared.jpg")
        json_io.write_annotations(
            labels_dir / "shared.json",
            [Annotation(subject=subject, geometry=BBox(box_x, 4, box_x + 8, 12))], 100, 80,
        )
        extra_stem = f"extra_{date}"
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{extra_stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{extra_stem}.json",
            [Annotation(subject=subject, geometry=BBox(4, 4, 12, 12))], 100, 80,
        )
    return root


def test_two_dates_sharing_a_filename_stay_distinct_samples(tmp_path: Path):
    """The two ``shared`` images are two samples, each naming its own source and label under its
    own capture date, and each reads back as distinct pixels: a selection carries no bare-stem
    identity a second date could collide with."""
    root = _two_date_collision_dataset(tmp_path / "ds", subject="leaf")
    out = tmp_path / "m"
    result = draw_splits(str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25, seed=1)
    assert "error" not in result, result
    assert result["total_stems"] == 4

    drawn = read_selection(out)
    shared = [s for s in drawn.samples if Path(s.ground_truth).stem == "shared"]
    assert len(shared) == 2
    assert {Path(s.source).parent.name for s in shared} == {"2-11-26", "2-12-01"}
    assert {Path(s.ground_truth).parent.name for s in shared} == {"2-11-26", "2-12-01"}
    assert len({s.identity for s in shared}) == 2
    # Each sample's own label document is the one under its own date, never the other's.
    boxes = {
        json_io.read_annotations(s.ground_truth)[0].geometry.x1 for s in shared  # type: ignore[union-attr]
    }
    assert boxes == {4.0, 40.0}


def _two_subject_dataset(root: Path) -> Path:
    """Six stems on one date: four carry ``leaf``, two carry the unrelated subject ``bud``, no
    stem carries both; four ``leaf`` stems clear a leaf-scoped draw's foreground floor."""
    from PIL import Image

    from tcip_mcp.subject_registry import SubjectRegistry, Subject, write_registry

    date = "2-11-26"
    images_dir = root / "images" / date
    labels_dir = root / "annotations" / date
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    write_registry(root / "subjects.json", SubjectRegistry(subjects=(
        Subject(name="leaf"), Subject(name="bud"),
    )))
    for stem, subject in (
        ("leaf_a", "leaf"), ("leaf_b", "leaf"), ("leaf_c", "leaf"), ("leaf_d", "leaf"),
        ("bud_a", "bud"), ("bud_b", "bud"),
    ):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject=subject, geometry=BBox(4, 4, 12, 12))], 100, 80,
        )
    return root


def test_draw_splits_holds_only_the_named_subjects_admitted_samples(tmp_path: Path):
    root = _two_subject_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = draw_splits(str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25, seed=1)
    assert "error" not in result, result
    assert result["total_stems"] == 4
    drawn = read_selection(out)
    assert {Path(s.ground_truth).stem for s in drawn.samples} == {
        "leaf_a", "leaf_b", "leaf_c", "leaf_d"}


def _attribute_scoped_dataset(root: Path) -> Path:
    """Five stems on one date, one subject: four have their instance assessed for ``condition``
    (clearing an attribute-scoped draw's foreground floor), one carries an instance never
    assessed for it."""
    from PIL import Image

    from tcip_mcp.subject_registry import Attribute, SubjectRegistry, Subject, write_registry

    date = "2-11-26"
    images_dir = root / "images" / date
    labels_dir = root / "annotations" / date
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    write_registry(root / "subjects.json", SubjectRegistry(subjects=(
        Subject(name="leaf", attributes=(
            Attribute(name="condition", type="categorical", values=("healthy", "damaged")),
        )),
    )))
    for stem, condition in (
        ("assessed_a", "healthy"), ("assessed_b", "damaged"),
        ("assessed_c", "healthy"), ("assessed_d", "damaged"),
    ):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12),
                       attributes={"condition": condition})], 100, 80,
        )
    Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / "unassessed.jpg")
    json_io.write_annotations(
        labels_dir / "unassessed.json",
        [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
    )
    return root


def test_draw_splits_attribute_scoped_selection_holds_only_assessed_samples(tmp_path: Path):
    root = _attribute_scoped_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = draw_splits(str(root), output_path=str(out), subject="leaf", attribute="condition",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25, seed=1)
    assert "error" not in result, result
    assert result["total_stems"] == 4
    drawn = read_selection(out)
    assert drawn.attribute == "condition"
    assert {Path(s.ground_truth).stem for s in drawn.samples} == {
        "assessed_a", "assessed_b", "assessed_c", "assessed_d"}


def _two_date_flat_images_dataset(root: Path, subject: str) -> Path:
    """Two dated label directories whose images were never split into date buckets: both
    entries fall back to the same flat images/ root."""
    from PIL import Image

    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    for date in ("2-11-26", "2-12-01"):
        labels_dir = root / "annotations" / date
        labels_dir.mkdir(parents=True)
        for stem in ("w", "x", "y", "z"):
            dst = images_dir / f"{stem}.jpg"
            if not dst.exists():
                Image.new("RGB", (100, 80), (128, 128, 128)).save(dst)
            json_io.write_annotations(
                labels_dir / f"{stem}.json",
                [Annotation(subject=subject, geometry=BBox(4, 4, 12, 12))], 100, 80,
            )
    return root


def test_draw_splits_refuses_two_dated_label_dirs_sharing_a_flat_images_root(tmp_path: Path):
    """Two label dates whose images were never split into date buckets both resolve to the
    same flat images/ root: a manifest keyed by <date>/<stem> would admit one image file once
    per date and could place the same pixels on both sides of the split."""
    root = _two_date_flat_images_dataset(tmp_path / "ds", subject="leaf")
    out = tmp_path / "m"

    result = draw_splits(str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    assert "error" in result
    assert "2-11-26" in result["error"] and "2-12-01" in result["error"]
    assert not out.exists()


def test_draw_splits_refuses_a_dated_dir_and_loose_labels_sharing_a_flat_images_root(
    tmp_path: Path,
):
    """A dated label directory with no images/<date>/ bucket of its own and a loose label
    beside it both fall back to the same flat images/ root: the same leak, mirrored."""
    from PIL import Image

    root = tmp_path / "ds"
    images_dir = root / "images"
    dated_labels = root / "annotations" / "2-11-26"
    images_dir.mkdir(parents=True)
    dated_labels.mkdir(parents=True)
    for stem in ("a", "b", "c", "d"):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
    for stem in ("b", "c", "d"):
        json_io.write_annotations(
            dated_labels / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
        )
    json_io.write_annotations(
        root / "annotations" / "a.json",
        [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
    )
    out = tmp_path / "m"

    result = draw_splits(str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    assert "error" in result
    assert "2-11-26" in result["error"]
    assert "annotations/ (loose labels)" in result["error"]
    assert not out.exists()


def test_draw_splits_nothing_admitted_names_the_searched_directories_and_the_unpaired_move(
    tmp_path: Path,
):
    """A tree whose labels sit flat while its images were split into a date bucket admits
    nothing: the refusal names each entry's searched directory and the unpaired bucket."""
    from PIL import Image

    root = tmp_path / "ds"
    dated_images = root / "images" / "2-11-26"
    dated_images.mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    Image.new("RGB", (100, 80), (128, 128, 128)).save(dated_images / "a.jpg")
    json_io.write_annotations(
        root / "annotations" / "a.json",
        [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
    )
    out = tmp_path / "m"

    result = draw_splits(str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    assert "error" in result
    assert "annotations/ (loose labels)" in result["error"]
    assert str(root / "images") in result["error"]
    assert str(dated_images) in result["error"]
    assert not out.exists()


def _dated_labels_flat_images_dataset(root: Path, stems: tuple[str, ...]) -> Path:
    """Labels dated but images never split into date buckets: a layout the platform's other
    readers already resolve (``annotation_tools.py``'s stage-shape door)."""
    from PIL import Image

    images_dir = root / "images"
    labels_dir = root / "annotations" / "2-11-26"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for stem in stems:
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
        )
    return root


def test_draw_splits_manifest_admits_dated_labels_over_flat_images(tmp_path: Path):
    root = _dated_labels_flat_images_dataset(tmp_path / "ds", ("p0", "p1", "p2", "p3", "p4"))
    out = tmp_path / "m"
    result = draw_splits(str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in result
    assert result["total_stems"] == 5
    assert result["admission_counts"]["annotated"] == 5


def test_draw_splits_manifest_admits_a_loose_label_beside_a_dated_one(tmp_path: Path):
    """A label sitting loose in ``annotations/`` beside a dated bucket enters the draw as a
    dateless member, rather than being invisible to both the manifest and its own counts."""
    from PIL import Image

    root = tmp_path / "ds"
    dated_images = root / "images" / "2-11-26"
    dated_labels = root / "annotations" / "2-11-26"
    dated_images.mkdir(parents=True)
    dated_labels.mkdir(parents=True)
    for stem in ("a", "b", "c"):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(dated_images / f"{stem}.jpg")
        json_io.write_annotations(
            dated_labels / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
        )
    for stem in ("loose1", "loose2"):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(root / "images" / f"{stem}.jpg")
        json_io.write_annotations(
            root / "annotations" / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
        )

    out = tmp_path / "m"
    result = draw_splits(str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    assert "error" not in result
    assert result["total_stems"] == 5
    assert result["admission_counts"]["annotated"] == 5
    drawn = read_selection(out)
    assert {Path(s.ground_truth).stem for s in drawn.samples} == {"a", "b", "c", "loose1", "loose2"}
    assert {str(Path(s.ground_truth).parent.relative_to(root)) for s in drawn.samples} == {
        str(Path("annotations") / "2-11-26"), "annotations"}


def test_split_date_dirs_ignores_a_stray_stamp_named_document(tmp_path: Path):
    """A bucket's own provenance stamp sitting loose directly under ``annotations/`` is not a
    loose label: it must never mint a dateless entry the way a real loose label would, the same
    exclusion every bucket walk through ``prediction_documents`` already applies."""
    from tcip_mcp.tools.data_tools import _split_date_dirs

    root = tmp_path / "ds"
    (root / "annotations" / "2-11-26").mkdir(parents=True)
    (root / "images" / "2-11-26").mkdir(parents=True)
    (root / "annotations" / "operating_point.json").write_text('{"trait": null}', encoding="utf-8")

    entries = _split_date_dirs(root)

    assert [date for date, _, _ in entries] == ["2-11-26"]


def test_split_date_dirs_still_admits_a_real_loose_label(tmp_path: Path):
    from tcip_mcp.tools.data_tools import _split_date_dirs

    root = tmp_path / "ds"
    (root / "annotations" / "2-11-26").mkdir(parents=True)
    (root / "images" / "2-11-26").mkdir(parents=True)
    (root / "images").mkdir(parents=True, exist_ok=True)
    json_io.write_annotations(
        root / "annotations" / "loose.json",
        [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
    )

    entries = _split_date_dirs(root)

    assert {date for date, _, _ in entries} == {None, "2-11-26"}


def test_draw_splits_holds_no_sample_for_a_date_that_admits_nothing(tmp_path: Path):
    """A capture date whose only label resolves to no image anywhere contributes no sample, so a
    selection never carries a member whose pixels are gone."""
    from PIL import Image

    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    labels_dir = root / "annotations" / "2-11-26"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for stem in ("a", "b", "c", "d"):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
        )
    orphan_labels = root / "annotations" / "2-12-26"
    orphan_labels.mkdir(parents=True)
    json_io.write_annotations(
        orphan_labels / "orphan.json",
        [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
    )

    out = tmp_path / "m"
    result = draw_splits(str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in result
    drawn = read_selection(out)

    assert all(Path(s.ground_truth).parent.name == "2-11-26" for s in drawn.samples)
    assert "orphan" not in {Path(s.ground_truth).stem for s in drawn.samples}


def test_doctor_check_data_quality_admits_a_confirmed_negative_under_dated_labels_flat_images(
    tmp_path: Path,
):
    """A human-confirmed negative resolves the same way the doctor's check reads it as the
    draw that admits it: labels dated, images never split into date buckets."""
    from tcip_mcp.dataset_layout import record_image_statuses, status_bucket

    root = _dated_labels_flat_images_dataset(tmp_path / "ds", ("p0",))
    (root / "annotations" / "2-11-26" / "p0.json").unlink()
    json_io.write_annotations(
        root / "annotations" / "2-11-26" / "p0.json", [], 100, 80, keep_empty=True,
    )
    record_image_statuses(
        root, status_bucket("leaf", "2-11-26"), {"p0.jpg": "negative"}, recorded_by="user:right",
    )

    assert _quality_findings(root) == []


def _one_sample_selection(source: str = "images/a.jpg", label: str = "annotations/a.json",
                          group: str = "a", side: str = "train") -> Selection:
    return Selection(
        samples=(Sample(member=Path(label).stem, source=source, ground_truth=label, group=group,
                        side=side, confirmation_bucket="leaf/2-11-26"),),
        subject="leaf", id_map={"leaf": 0}, seed=1, group_by="stem",
    )


def test_read_selection_admits_the_writers_own_record(tmp_path: Path):
    """The reader accepts exactly what the writer wrote, through the platform's own producer."""
    out = tmp_path / "m"
    write_selection(out, _one_sample_selection())

    drawn = read_selection(out)

    assert drawn.subject == "leaf"
    assert drawn.seed == 1
    assert [s.identity for s in drawn.samples] == ["images/a.jpg"]


def test_read_selection_refuses_an_absent_record_by_name(tmp_path: Path):
    with pytest.raises(ValueError, match="no selection recorded"):
        read_selection(tmp_path / "nothing")


def test_read_selection_refuses_a_sample_missing_its_own_ground_truth(tmp_path: Path):
    """A sample naming no ground truth binds nothing: a selection's samples each name their own
    rather than sharing a directory the reader could reconstruct one from."""
    out = tmp_path / "m"
    write_selection(out, _one_sample_selection())
    document = ts.read(selection_key(out))
    document["samples"][0].pop("ground_truth")
    ts.replace(selection_key(out), document)

    with pytest.raises(ValueError, match=r"carries no \['ground_truth'\]"):
        read_selection(out)


def test_read_selection_refuses_a_label_document_sample_with_no_confirmation_bucket(
    tmp_path: Path,
):
    """Which human confirmations admitted a sample is a per-sample fact a later admission check
    reads back, and a label document's admission always reads one, so a sample naming its own
    document and no bucket names nothing to re-check it against. A mask or a table row is
    admitted by existing, so neither carries one and neither is refused for it."""
    out = tmp_path / "m"
    write_selection(out, _one_sample_selection())
    document = ts.read(selection_key(out))
    document["samples"][0].pop("confirmation_bucket")
    ts.replace(selection_key(out), document)

    with pytest.raises(ValueError, match="no confirmation_bucket"):
        read_selection(out)


def test_read_selection_refuses_an_empty_sample_list(tmp_path: Path):
    out = tmp_path / "m"
    write_selection(out, _one_sample_selection())
    document = ts.read(selection_key(out))
    document["samples"] = []
    ts.replace(selection_key(out), document)

    with pytest.raises(ValueError, match="lists no samples"):
        read_selection(out)


def test_read_selection_refuses_one_source_on_two_sides(tmp_path: Path):
    """The same pixels on train and calibration would be trained on and measured on at once."""
    out = tmp_path / "m"
    write_selection(out, _one_sample_selection())
    document = ts.read(selection_key(out))
    document["samples"].append(
        {"member": "a", "source": "images/a.jpg", "ground_truth": "annotations/a.json",
         "group": "b",
         "side": "calibration", "confirmation_bucket": "leaf/2-11-26"})
    ts.replace(selection_key(out), document)

    with pytest.raises(ValueError, match="on more than one side"):
        read_selection(out)


def test_read_selection_refuses_one_group_on_two_sides(tmp_path: Path):
    """Crops of one parent, or captures of one subject, share a group key: splitting them across
    sides leaks one side into the other, so the reader refuses the partition outright."""
    out = tmp_path / "m"
    write_selection(out, _one_sample_selection())
    document = ts.read(selection_key(out))
    document["samples"].append(
        {"member": "a_0_1", "source": "images/a_0_1.jpg",
         "ground_truth": "annotations/a_0_1.json", "group": "a",
         "side": "val", "confirmation_bucket": "leaf/2-11-26"})
    ts.replace(selection_key(out), document)

    with pytest.raises(ValueError, match="group"):
        read_selection(out)
