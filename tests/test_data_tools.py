"""Tests for data management tools."""

from __future__ import annotations

import pytest
import tcip_store as ts
from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox

from pathlib import Path

from tcip_mcp.tools.data_tools import (
    read_split_manifest_dir,
    scan_dataset,
    validate_data_quality,
    make_splits,
    split_manifest_key,
    split_stem_list_key,
)


def test_scan_dataset(data_dir: Path):
    result = scan_dataset(str(data_dir))
    assert result["image_count"] == 3
    assert result["labels_count"] == 3
    assert result["paired_images"] == 3
    assert result["unlabelled_images"] == 0


def test_scan_dataset_not_found():
    result = scan_dataset("/nonexistent/path")
    assert "error" in result


def test_validate_data_quality(data_dir: Path):
    result = validate_data_quality(str(data_dir))
    assert result["total_images"] == 3
    assert result["total_labels"] == 3
    assert result["is_valid"] is True
    assert "catkin" in result["subjects"]


def test_validate_data_quality_missing_dir():
    assert "error" in validate_data_quality("/nonexistent/path/xyz123")


def test_scan_dataset_and_validate_data_quality_count_the_same_labels(tmp_path: Path):
    """scan_dataset's labels_count and validate_data_quality's total_labels are the same list
    over one root: the per-image tree plus a present root candidate, a review baseline excluded
    from both the same way."""
    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)
    labels_dir = root / "annotations" / "2-11-26"
    labels_dir.mkdir(parents=True)
    for stem in ("plotA_0_0", "plotB_0_0"):
        (images_dir / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")
        json_io.write_annotations(labels_dir / f"{stem}.json", [], 32, 32, keep_empty=True)
    baselines = labels_dir / ".original"
    baselines.mkdir()
    json_io.write_annotations(baselines / "plotA_0_0.json", [], 32, 32, keep_empty=True)
    (root / "annotations.json").write_text(
        '{"images": [], "annotations": [], "categories": []}', encoding="utf-8"
    )

    scan_result = scan_dataset(str(root))
    quality_result = validate_data_quality(str(root))

    assert scan_result["labels_count"] == quality_result["total_labels"] == 3


def test_scan_and_validate_report_a_reserved_stem_the_census_still_counted(tmp_path: Path):
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
        [Annotation(subject="catkin", geometry=BBox(1, 1, 5, 5))], 32, 32,
    )
    reserved_label = str(labels_dir / "operating_point.json")

    scan_result = scan_dataset(str(root))
    quality_result = validate_data_quality(str(root))

    assert scan_result["reserved_name_labels"] == [reserved_label]
    assert quality_result["reserved_name_labels"] == [reserved_label]


def test_scan_and_validate_report_a_reserved_stem_image_with_no_label(tmp_path: Path):
    """An image whose own stem is reserved for a bucket's own provenance stamp must be named,
    not folded into unlabelled_images with no signal that its label can never be read through
    any bucket walk."""
    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)
    (images_dir / "operating_point.jpg").write_bytes(b"\xff\xd8\xff")
    (images_dir / "ordinary.jpg").write_bytes(b"\xff\xd8\xff")
    reserved_image = str(images_dir / "operating_point.jpg")

    scan_result = scan_dataset(str(root))
    quality_result = validate_data_quality(str(root))

    assert scan_result["reserved_name_images"] == [reserved_image]
    assert quality_result["reserved_name_images"] == [reserved_image]
    assert scan_result["unlabelled_images"] == 2


def _add_extra_catkin_groups(data_dir: Path, count: int) -> None:
    """Adds ``count`` more single-tile foreground groups under ``data_dir``'s own date, for its
    own subject, without touching the shared ``data_dir`` fixture other tests depend on: a
    manifest write needs at least four foreground groups to clear ``make_splits``' floor, one
    more than the fixture's own three."""
    from PIL import Image

    images_dir = data_dir / "images" / "2-11-26"
    labels_dir = data_dir / "annotations" / "2-11-26"
    for i in range(count):
        stem = f"extra_{i:03d}"
        Image.new("RGB", (640, 480), color=(128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="catkin", geometry=BBox(288, 216, 352, 264))], 640, 480,
        )


def test_make_splits_materialize(data_dir: Path, tmp_path: Path):
    _add_extra_catkin_groups(data_dir, 1)
    out = tmp_path / "splits"
    result = make_splits(str(data_dir), output_path=str(out), materialize=True, subject="catkin",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in result, result
    assert result["total_stems"] == 4
    assert sum(result["splits"].values()) == 4
    assert result["output_dir"] == str(out)
    for split in ("train", "val", "calibration"):
        assert ts.exists(split_stem_list_key(out, split))
        assert (out / split / "images").is_dir()
        assert (out / split / "labels").is_dir()
    assert not (out / "test").exists()
    # Every image landed under exactly one split's images/ dir.
    placed = sorted(p.stem for p in out.rglob("images/*") if p.is_file())
    assert placed == ["extra_000", "img_001", "img_002", "img_003"]


def test_make_splits_basic(data_dir: Path, tmp_path: Path):
    _add_extra_catkin_groups(data_dir, 1)
    out = tmp_path / "manifests"
    # The fixture's 4 stems (img_001..003 plus one grown group) are 4 distinct foreground
    # groups, exactly the manifest floor (one each for train/val, two for calibration).
    result = make_splits(str(data_dir), output_path=str(out), subject="catkin",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in result, result
    assert result["total_stems"] == 4
    assert result["groups"] == 4
    assert sum(result["splits"].values()) == 4
    assert result["stratified"] is True
    for split in ("train", "val", "calibration"):
        assert ts.exists(split_stem_list_key(out, split))

    manifest = ts.read(split_manifest_key(out))
    assert manifest["subject"] == "catkin"
    assert manifest["attribute"] is None
    date_block = manifest["members"]["2-11-26"]
    assert Path(date_block["labels_root"]).is_dir()
    assert date_block["dataset_hash"]
    assert manifest["dataset_fingerprint"] is not None
    assert set(manifest["splits"]) == {"train", "val", "calibration"}


def test_make_splits_stats_only_admits_a_nonzero_calibration_ratio(data_dir: Path):
    """A stats-only call (no output_path, no materialize) may pass any calibration_ratio; only a
    manifest write requires a non-zero one."""
    result = make_splits(str(data_dir), train_ratio=0.7, val_ratio=0.2, calibration_ratio=0.1)
    assert "error" not in result, result
    assert result["splits"]["calibration"] >= 0


def test_make_splits_reports_an_unreadable_label_by_name(data_dir: Path, tmp_path: Path):
    """A present, unreadable label among the candidates is an error naming the file, never a
    raise through the tool boundary."""
    bad = next((data_dir / "annotations" / "2-11-26").glob("*.json"))
    bad.write_bytes(b"{not json")

    result = make_splits(str(data_dir), output_path=str(tmp_path / "manifests"), subject="catkin",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    assert "error" in result
    assert str(bad) in result["error"]


def test_make_splits_reports_an_unreadable_label_sorted_last(
    data_dir: Path, tmp_path: Path,
):
    """A corrupt label reached last in sort order is caught by the same per-stem admission read
    as the first-sorted case above, regardless of where in the candidate order it falls, and
    answers the same error dict, never a raw raise."""
    bad = sorted((data_dir / "annotations" / "2-11-26").glob("*.json"))[-1]
    bad.write_bytes(b"{not json")

    result = make_splits(str(data_dir), output_path=str(tmp_path / "manifests"), subject="catkin",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    assert "error" in result
    assert str(bad) in result["error"]


def test_make_splits_stats_only_reports_an_unreadable_first_sorted_label(data_dir: Path):
    """A stats-only call (no output_path, no materialize) draws no subject-scoped admission at
    all: its own scan raises on the first-sorted candidate, the same as scan_dataset would."""
    bad = data_dir / "annotations" / "2-11-26" / "img_001.json"
    bad.write_bytes(b"{not json")

    result = make_splits(str(data_dir))

    assert "error" in result
    assert str(bad) in result["error"]


def test_make_splits_stats_only_reports_an_unreadable_label_during_stratification(data_dir: Path):
    """A stats-only call still reads every stem's label to count its annotations for stratified
    balancing, so a corrupt label reached after a readable first candidate is an error naming the
    file, not a raw raise."""
    bad = data_dir / "annotations" / "2-11-26" / "img_003.json"
    bad.write_bytes(b"{not json")

    result = make_splits(str(data_dir))

    assert "error" in result
    assert str(bad) in result["error"]


def test_make_splits_manifest_answers_an_ambiguous_image_stem_as_an_error(tmp_path: Path):
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

    result = make_splits(str(root), output_path=str(tmp_path / "manifests"), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

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


def test_make_splits_materialize_refuses_an_incomplete_band_group_before_writing(tmp_path: Path):
    """A band group whose manifest names a missing sibling is refused before any stem list, the
    manifest or a split tree is written, never a raise through the tool after they land."""
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

    result = make_splits(str(root), output_path=str(out), materialize=True, subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    assert "error" in result
    assert "plotA" in result["error"] and "R" in result["error"]
    assert not out.exists()


def test_make_splits_materialize_places_a_complete_band_group(tmp_path: Path):
    """A complete band group still materializes, copying or symlinking every sibling band plus
    its manifest."""
    import numpy as np

    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    for copy_files in (True, False):
        root = tmp_path / f"ds_{copy_files}"
        images_dir = root / "images" / "2-11-26"
        images_dir.mkdir(parents=True)
        labels_dir = root / "annotations" / "2-11-26"
        labels_dir.mkdir(parents=True)
        band_g, band_r = images_dir / "plotA_G.npy", images_dir / "plotA_R.npy"
        np.save(band_g, np.zeros((4, 4), dtype=np.uint8))
        np.save(band_r, np.zeros((4, 4), dtype=np.uint8))
        write_band_group_manifest(images_dir, "plotA", {"G": band_g, "R": band_r})
        json_io.write_annotations(
            labels_dir / "plotA.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
        )
        _add_extra_leaf_groups(images_dir, labels_dir, 3)
        out = tmp_path / f"m_{copy_files}"

        result = make_splits(str(root), output_path=str(out), materialize=True, subject="leaf",
                             copy_files=copy_files, train_ratio=0.5, val_ratio=0.25,
                             calibration_ratio=0.25)

        assert "error" not in result, result
        placed = {p.name for split in ("train", "val", "calibration")
                 for p in (out / split / "images").glob("plotA*")}
        assert placed == {"plotA.bandgroup", "plotA_G.npy", "plotA_R.npy"}


def test_make_splits_stats_only_carries_dataset_hash(data_dir: Path):
    """A stats-only call's answer identifies the labels it partitioned, the same as a manifest
    call's own per-date record."""
    result = make_splits(str(data_dir))
    assert "error" not in result
    assert result["dataset_hash"]
    assert result["dataset_hashes_by_date"] == {"2-11-26": result["dataset_hash"]}


def test_make_splits_stats_only_over_two_dates_names_both_hashes_and_no_single_hash(
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

    result = make_splits(str(root))

    assert "error" not in result, result
    assert result["dataset_hash"] is None
    assert set(result["dataset_hashes_by_date"]) == {"2-11-26", "2-12-01"}
    assert result["dataset_hashes_by_date"]["2-11-26"] != result["dataset_hashes_by_date"]["2-12-01"]


def test_make_splits_manifest_answer_carries_each_dates_hash(data_dir: Path, tmp_path: Path):
    """A manifest call's answer identifies the labels it partitioned per capture date, the same
    hashes the written manifest's ``members`` blocks record, so the caller need not open the
    record to cite what the draw covered."""
    _add_extra_catkin_groups(data_dir, 1)
    out = tmp_path / "manifests"
    result = make_splits(str(data_dir), output_path=str(out), subject="catkin",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in result, result

    members = ts.read(split_manifest_key(out))["members"]
    assert result["dataset_hashes_by_date"] == {
        key: block["dataset_hash"] for key, block in members.items()}
    assert result["dataset_hashes_by_date"]["2-11-26"]


def test_make_splits_bad_ratios(data_dir: Path):
    result = make_splits(str(data_dir), train_ratio=0.5, val_ratio=0.5, calibration_ratio=0.5)
    assert "error" in result


def test_make_splits_train_val_calibration_not_summing_to_one_names_all_three(data_dir: Path):
    """The sum-check message names all three standing constraints, not just the raw sum."""
    result = make_splits(str(data_dir), train_ratio=0.7, val_ratio=0.2, calibration_ratio=0.2)
    assert "error" in result
    assert "calibration_ratio" in result["error"]
    assert "train_ratio" in result["error"] and "val_ratio" in result["error"]


def test_make_splits_manifest_write_refuses_a_zero_calibration_ratio(tmp_path: Path):
    """A manifest's calibration side is the universe every calibration under it draws from, so a
    manifest write states a non-zero calibration_ratio; the keyword names the missing input."""
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"

    result = make_splits(str(root), output_path=str(out), subject="catkin")

    assert "error" in result
    assert "calibration_ratio" in result["error"]
    assert not out.exists()


def test_make_splits_writes_three_stem_lists(tmp_path: Path):
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"

    result = make_splits(str(root), output_path=str(out), seed=1, subject="catkin",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    assert "error" not in result, result
    for split in ("train", "val", "calibration"):
        assert ts.exists(split_stem_list_key(out, split))
    manifest = ts.read(split_manifest_key(out))
    assert set(manifest["splits"]) == {"train", "val", "calibration"}
    assert manifest["splits"]["calibration"]


def test_make_splits_floor_refuses_before_any_write_regardless_of_stratify_foreground(
    tmp_path: Path,
):
    """The foreground floor is over the draw's own subject-scoped counter on every manifest
    draw, whether or not stratify_foreground toggles the balancing pass: a tree with only three
    foreground groups refuses before anything is written, with stratify_foreground off."""
    root = _multi_source_dataset(tmp_path / "ds", prefixes=("srcA", "srcB", "srcC"))
    out = tmp_path / "m"

    result = make_splits(str(root), output_path=str(out), subject="catkin", seed=1,
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25,
                         stratify_foreground=False)

    assert "error" in result
    assert "foreground group" in result["error"]
    assert not out.exists()


def test_make_splits_floor_ignores_a_groups_only_annotations_of_another_subject(tmp_path: Path):
    """A confirmed-negative-for-the-draws-subject group whose label file happens to carry
    another subject's annotation is not this draw's foreground: the subject-scoped counter
    reads it as zero, so three real ``leaf`` groups plus one such group still refuse (below the
    floor of four), the same tree an unscoped counter would have read as four and written."""
    from PIL import Image

    from tcip_mcp.class_registry import ClassRegistry, Subject, write_registry
    from tcip_mcp.dataset_layout import record_image_statuses, status_bucket

    root = tmp_path / "ds"
    date = "2-11-26"
    images_dir, labels_dir = root / "images" / date, root / "annotations" / date
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    write_registry(root / "classes.json", ClassRegistry(subjects=(
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
    result = make_splits(str(root), output_path=str(out), subject="leaf", seed=1,
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    assert "error" in result
    assert "foreground group" in result["error"]
    assert not out.exists()


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
                                      [Annotation(subject="catkin", geometry=BBox(19, 13, 45, 51))],
                                      64, 64)
    return root


def test_make_splits_groups_tiles_together(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import default_group_key

    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), seed=1, subject="catkin",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert result["groups"] == 4  # 4 source prefixes, not 12 tiles

    # No source prefix may appear in more than one split.
    seen: dict[str, str] = {}
    for split in ("train", "val", "calibration"):
        for stem in ts.read(split_stem_list_key(out, split)):
            g = default_group_key(stem)
            assert seen.get(g, split) == split, f"group {g} spans splits"
            seen[g] = split


def test_make_splits_group_key_map_never_straddles(tmp_path: Path):
    """An agent-derived group_key_map (5 members, 4 groups) is honored: the two same-group
    members never land in different splits."""
    root = _multi_source_dataset(tmp_path / "ds", prefixes=("x", "y", "z", "w", "v"), tiles=1)
    out = tmp_path / "m"
    group_key_map = {
        "2-11-26/x_0_0": "gA", "2-11-26/y_0_0": "gA", "2-11-26/z_0_0": "gB",
        "2-11-26/w_0_0": "gC", "2-11-26/v_0_0": "gD",
    }
    result = make_splits(str(root), output_path=str(out), seed=1,
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25,
                         group_by="tile_prefix", group_key_map=group_key_map, subject="catkin")
    assert "error" not in result, result
    assert result["group_by"] == "explicit_map"

    membership: dict[str, str] = {}
    for split in ("train", "val", "calibration"):
        for identity in ts.read(split_stem_list_key(out, split)):
            membership[identity] = split
    assert membership["2-11-26/x_0_0"] == membership["2-11-26/y_0_0"]  # gA never straddles
    manifest = ts.read(split_manifest_key(out))
    assert manifest["group_by"] == "explicit_map"
    assert manifest["group_key_map"] == group_key_map


def test_make_splits_unrecognized_group_by_refuses_without_writing(tmp_path: Path):
    """A silent ``GROUP_KEY_FNS.get(group_by, default_group_key)`` fallback used to mis-group a
    dataset without anyone noticing; it must refuse loudly and write nothing instead."""
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), group_by="not_a_real_key", subject="catkin",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" in result
    assert not out.exists() or not (out / "split_manifest.json").is_file()


def test_make_splits_refuses_to_write_a_manifest_with_no_subject(tmp_path: Path):
    """A manifest with no subject would be a partition of images, not of a run's admissible
    samples; make_splits refuses to write one rather than guessing what a run would admit."""
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out),
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" in result and "subject" in result["error"]
    assert not out.exists()


def _two_date_collision_dataset(root: Path, subject: str) -> Path:
    """One stem name, ``shared``, present under two capture dates with different content, plus
    one more distinct stem per date so a manifest write over this tree clears the foreground
    floor: a manifest keyed by bare stem could only ever hold one of the two ``shared`` images."""
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


def test_two_dates_sharing_a_filename_produce_two_members(tmp_path: Path):
    root = _two_date_collision_dataset(tmp_path / "ds", subject="leaf")
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25, seed=1)
    assert "error" not in result, result
    assert result["total_stems"] == 4
    manifest = ts.read(split_manifest_key(out))
    members = {identity for identities in manifest["splits"].values() for identity in identities}
    assert {"2-11-26/shared", "2-12-01/shared"} <= members
    assert set(manifest["members"]) == {"2-11-26", "2-12-01"}


def _two_subject_dataset(root: Path) -> Path:
    """Six stems on one date: four carry ``leaf``, two carry the unrelated subject ``bud``, no
    stem carries both; four ``leaf`` stems clear a leaf-scoped manifest write's foreground floor."""
    from PIL import Image

    from tcip_mcp.class_registry import ClassRegistry, Subject, write_registry

    date = "2-11-26"
    images_dir = root / "images" / date
    labels_dir = root / "annotations" / date
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    write_registry(root / "classes.json", ClassRegistry(subjects=(
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


def test_make_splits_holds_only_the_named_subjects_admitted_samples(tmp_path: Path):
    root = _two_subject_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25, seed=1)
    assert "error" not in result, result
    assert result["total_stems"] == 4
    manifest = ts.read(split_manifest_key(out))
    members = {identity for identities in manifest["splits"].values() for identity in identities}
    assert members == {"2-11-26/leaf_a", "2-11-26/leaf_b", "2-11-26/leaf_c", "2-11-26/leaf_d"}


def _attribute_scoped_dataset(root: Path) -> Path:
    """Five stems on one date, one subject: four have their instance assessed for ``condition``
    (clearing an attribute-scoped manifest write's foreground floor), one carries an instance
    never assessed for it."""
    from PIL import Image

    from tcip_mcp.class_registry import Attribute, ClassRegistry, Subject, write_registry

    date = "2-11-26"
    images_dir = root / "images" / date
    labels_dir = root / "annotations" / date
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    write_registry(root / "classes.json", ClassRegistry(subjects=(
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


def test_make_splits_attribute_scoped_manifest_holds_only_assessed_samples(tmp_path: Path):
    root = _attribute_scoped_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), subject="leaf", attribute="condition",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25, seed=1)
    assert "error" not in result, result
    assert result["total_stems"] == 4
    manifest = ts.read(split_manifest_key(out))
    assert manifest["attribute"] == "condition"
    members = {identity for identities in manifest["splits"].values() for identity in identities}
    assert members == {
        "2-11-26/assessed_a", "2-11-26/assessed_b", "2-11-26/assessed_c", "2-11-26/assessed_d",
    }


def test_make_splits_refuses_to_materialize_a_multi_date_manifest(tmp_path: Path):
    root = _two_date_collision_dataset(tmp_path / "ds", subject="leaf")
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), materialize=True, subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25, seed=1)
    assert "error" in result
    assert "2-11-26" in result["error"] and "2-12-01" in result["error"]
    assert not out.exists()


def test_make_splits_multi_date_refusal_names_the_loose_label_bucket(tmp_path: Path):
    """A loose-label entry beside a dated bucket is named in the multi-date materialize
    refusal, not dropped from the list of spanned dates it claims to name."""
    from PIL import Image

    root = tmp_path / "ds"
    dated_images = root / "images" / "2-11-26"
    dated_labels = root / "annotations" / "2-11-26"
    dated_images.mkdir(parents=True)
    dated_labels.mkdir(parents=True)
    for stem in ("a", "b"):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(dated_images / f"{stem}.jpg")
        json_io.write_annotations(
            dated_labels / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
        )
    for stem in ("loose", "loose2"):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(root / "images" / f"{stem}.jpg")
        json_io.write_annotations(
            root / "annotations" / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
        )

    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), materialize=True, subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25, seed=1)

    assert "error" in result
    assert "2-11-26" in result["error"]
    assert "annotations/ (loose labels)" in result["error"]
    assert not out.exists()


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


def test_make_splits_refuses_two_dated_label_dirs_sharing_a_flat_images_root(tmp_path: Path):
    """Two label dates whose images were never split into date buckets both resolve to the
    same flat images/ root: a manifest keyed by <date>/<stem> would admit one image file once
    per date and could place the same pixels on both sides of the split."""
    root = _two_date_flat_images_dataset(tmp_path / "ds", subject="leaf")
    out = tmp_path / "m"

    result = make_splits(str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    assert "error" in result
    assert "2-11-26" in result["error"] and "2-12-01" in result["error"]
    assert not out.exists()


def test_make_splits_refuses_a_dated_dir_and_loose_labels_sharing_a_flat_images_root(
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

    result = make_splits(str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    assert "error" in result
    assert "2-11-26" in result["error"]
    assert "annotations/ (loose labels)" in result["error"]
    assert not out.exists()


def test_make_splits_nothing_admitted_names_the_searched_directories_and_the_unpaired_move(
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

    result = make_splits(str(root), output_path=str(out), subject="leaf",
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


def test_make_splits_manifest_admits_dated_labels_over_flat_images(tmp_path: Path):
    root = _dated_labels_flat_images_dataset(tmp_path / "ds", ("p0", "p1", "p2", "p3", "p4"))
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in result
    assert result["total_stems"] == 5
    assert result["admission_counts"]["annotated"] == 5


def test_make_splits_manifest_admits_a_loose_label_beside_a_dated_one(tmp_path: Path):
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
    result = make_splits(str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    assert "error" not in result
    assert result["total_stems"] == 5
    assert result["admission_counts"]["annotated"] == 5
    manifest = ts.read(split_manifest_key(out))
    identities = {i for ids in manifest["splits"].values() for i in ids}
    assert identities == {
        "2-11-26/a", "2-11-26/b", "2-11-26/c", "loose1", "loose2",
    }


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


def test_make_splits_writes_no_member_block_for_a_date_that_admits_nothing(tmp_path: Path):
    """A capture date whose only label resolves to no image anywhere writes no ``members`` block
    for it, so binding a run to that date names it as one the manifest never held, not one it
    holds empty."""
    from PIL import Image

    from tcip_mcp.pipelines.data.splits import bind_manifest_stems

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
    result = make_splits(str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in result
    manifest = ts.read(split_manifest_key(out))

    assert "2-12-26" not in manifest["members"]
    with pytest.raises(ValueError, match=r"holds members under \['2-11-26'\]"):
        bind_manifest_stems(manifest, "2-12-26", "leaf", None, [])


def test_make_splits_materialize_negative_carry_reads_only_the_materializing_dates_bucket(
    tmp_path: Path,
):
    """A confirmed negative recorded under a different date's bucket for a same-named image is
    not this split's to carry: the carry reads the one bucket the materializing date's own
    admission read, never a merge across every bucket the store names the subject under."""
    from PIL import Image

    from tcip_mcp.dataset_layout import (
        read_image_status_store, record_image_statuses, status_bucket,
    )

    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    labels_dir = root / "annotations" / "2-11-26"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / "neg.jpg")
    json_io.write_annotations(labels_dir / "neg.json", [], 100, 80, keep_empty=True)
    # A pure-negative draw has zero foreground groups; four more real annotations clear the
    # manifest floor without changing what this test is about (the negative carry's own bucket).
    for stem in ("pos_a", "pos_b", "pos_c", "pos_d"):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
        )

    record_image_statuses(
        root, status_bucket("leaf", "2-11-26"), {"neg.jpg": "negative"}, recorded_by="user:right",
    )
    record_image_statuses(
        root, status_bucket("leaf", "2-99-99"), {"neg.jpg": "negative"}, recorded_by="user:wrong",
    )

    out = tmp_path / "splits"
    result = make_splits(str(root), output_path=str(out), materialize=True, subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in result, result

    split_dir = next(
        out / s for s in ("train", "val", "calibration")
        if (out / s / "images" / "neg.jpg").is_file()
    )
    store = read_image_status_store(split_dir)
    record = store[status_bucket("leaf", None)]["neg.jpg"]
    assert record["recorded_by"] == "user:right"


def test_validate_data_quality_admits_a_confirmed_negative_under_dated_labels_flat_images(
    tmp_path: Path,
):
    """A human-confirmed negative resolves the same way validate_data_quality reads it as the
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

    result = validate_data_quality(str(root))

    assert result["is_valid"] is True
    assert result["issues"] == []


def test_read_split_manifest_dir_admits_the_writers_own_record(tmp_path: Path):
    """A manifest make_splits actually wrote carries every key the reader requires: the
    required set never rejects the writer's own output."""
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    write_result = make_splits(str(root), output_path=str(out), seed=1, subject="catkin",
                               train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in write_result

    manifest = read_split_manifest_dir(out)

    assert manifest["subject"] == "catkin"
    assert manifest["seed"] == 1


def test_read_split_manifest_dir_refuses_each_missing_required_key_by_name(tmp_path: Path):
    """A manifest missing any key make_splits writes is refused before a bind ever reads it,
    naming the missing key, rather than reaching a downstream cast with nothing to fall back to."""
    keys_a_written_manifest_carries = (
        "seed", "group_by", "dataset_fingerprint", "subject", "attribute", "id_map",
        "members", "splits", "admission_counts",
    )
    root = _multi_source_dataset(tmp_path / "ds")
    for missing_key in keys_a_written_manifest_carries:
        out = tmp_path / f"m_{missing_key}"
        result = make_splits(str(root), output_path=str(out), seed=1, subject="catkin",
                             train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
        assert "error" not in result

        full = ts.read(split_manifest_key(out))
        del full[missing_key]
        ts.replace(split_manifest_key(out), full)

        with pytest.raises(ValueError, match=missing_key):
            read_split_manifest_dir(out)


def test_read_split_manifest_dir_refuses_a_two_sided_record(tmp_path: Path):
    """A manifest drawn before the platform held out a calibration side (here simulated by
    dropping the third side from a record the writer actually produced) binds nothing: the
    reader refuses it by name rather than silently reading it as a two-sided manifest."""
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), seed=1, subject="catkin",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in result, result

    full = ts.read(split_manifest_key(out))
    full["splits"] = {"train": full["splits"]["train"], "val": full["splits"]["val"]}
    ts.replace(split_manifest_key(out), full)

    with pytest.raises(ValueError, match="calibration"):
        read_split_manifest_dir(out)


def test_read_split_manifest_dir_refuses_overlapping_sides(tmp_path: Path):
    """A record whose sides are not pairwise disjoint is refused, naming the identities two
    sides claim: the reader's contract, checked once here for every consumer that reads through
    it, since a member on train and calibration would be trained on and one it was held out to
    calibrate against."""
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), seed=1, subject="catkin",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in result, result

    full = ts.read(split_manifest_key(out))
    straddler = full["splits"]["calibration"][0]
    full["splits"]["train"] = [*full["splits"]["train"], straddler]
    ts.replace(split_manifest_key(out), full)

    with pytest.raises(ValueError, match="train/calibration"):
        read_split_manifest_dir(out)
