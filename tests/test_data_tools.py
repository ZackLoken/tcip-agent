"""Tests for data management tools."""

from __future__ import annotations

import tcip_store as ts
from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox

from pathlib import Path

from tcip_mcp.tools.data_tools import (
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


def test_make_splits_materialize(data_dir: Path, tmp_path: Path):
    out = tmp_path / "splits"
    result = make_splits(str(data_dir), output_path=str(out), materialize=True, subject="catkin")
    assert result["total_stems"] == 3
    assert sum(result["splits"].values()) == 3
    assert result["output_dir"] == str(out)
    for split in ("train", "val"):
        assert ts.exists(split_stem_list_key(out, split))
        assert (out / split / "images").is_dir()
        assert (out / split / "labels").is_dir()
    assert "test" not in result["splits"]
    assert not (out / "test").exists()
    # Every image landed under exactly one split's images/ dir.
    placed = sorted(p.stem for p in out.rglob("images/*") if p.is_file())
    assert placed == ["img_001", "img_002", "img_003"]


def test_make_splits_basic(data_dir: Path, tmp_path: Path):
    out = tmp_path / "manifests"
    # The 3 fixture stems (img_001..003) are 3 distinct foreground groups.
    result = make_splits(str(data_dir), output_path=str(out), subject="catkin")
    assert result["total_stems"] == 3
    assert result["groups"] == 3
    assert sum(result["splits"].values()) == 3
    assert result["stratified"] is True
    for split in ("train", "val"):
        assert ts.exists(split_stem_list_key(out, split))
    assert "test" not in result["splits"]

    manifest = ts.read(split_manifest_key(out))
    assert manifest["subject"] == "catkin"
    assert manifest["attribute"] is None
    date_block = manifest["members"]["2-11-26"]
    assert Path(date_block["labels_root"]).is_dir()
    assert date_block["dataset_hash"]
    assert manifest["dataset_fingerprint"] is not None
    assert "test" not in manifest["splits"]


def test_make_splits_refuses_a_nonzero_test_ratio(data_dir: Path):
    """No launch path honours a held-out test list: make_splits refuses one rather than writing
    a partition nothing downstream reads."""
    result = make_splits(str(data_dir), train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    assert "error" in result
    assert "test_ratio" in result["error"]


def test_make_splits_reports_an_unreadable_label_by_name(data_dir: Path, tmp_path: Path):
    """A present, unreadable label among the candidates is an error naming the file, never a
    raise through the tool boundary."""
    bad = next((data_dir / "annotations" / "2-11-26").glob("*.json"))
    bad.write_bytes(b"{not json")

    result = make_splits(str(data_dir), output_path=str(tmp_path / "manifests"), subject="catkin")

    assert "error" in result
    assert str(bad) in result["error"]


def test_make_splits_reports_an_unreadable_label_reached_only_through_stratification(
    data_dir: Path, tmp_path: Path,
):
    """A corrupt label sorted after the scan's own format-probe file is caught by the
    stratification count, not the scan: a different site than the first-sorted case above, and
    it must answer the same error dict, never a raw raise."""
    bad = sorted((data_dir / "annotations" / "2-11-26").glob("*.json"))[-1]
    bad.write_bytes(b"{not json")

    result = make_splits(str(data_dir), output_path=str(tmp_path / "manifests"), subject="catkin")

    assert "error" in result
    assert str(bad) in result["error"]


def test_make_splits_bad_ratios(data_dir: Path):
    result = make_splits(str(data_dir), train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)
    assert "error" in result


def test_make_splits_train_and_val_not_summing_to_one_names_both_constraints(data_dir: Path):
    """A caller carrying over the old three-way train_ratio with test_ratio at its new default
    of 0 gets a message naming both standing constraints, not just the raw sum."""
    result = make_splits(str(data_dir), train_ratio=0.7, val_ratio=0.2, test_ratio=0.0)
    assert "error" in result
    assert "test_ratio" in result["error"]
    assert "train_ratio" in result["error"] and "val_ratio" in result["error"]


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
    result = make_splits(str(root), output_path=str(out), seed=1, subject="catkin")
    assert result["groups"] == 4  # 4 source prefixes, not 12 tiles

    # No source prefix may appear in more than one split.
    seen: dict[str, str] = {}
    for split in ("train", "val"):
        for stem in ts.read(split_stem_list_key(out, split)):
            g = default_group_key(stem)
            assert seen.get(g, split) == split, f"group {g} spans splits"
            seen[g] = split


def test_make_splits_group_key_map_never_straddles(tmp_path: Path):
    """An agent-derived group_key_map (3 members, 2 groups) is honored: the two same-group
    members never land in different splits."""
    root = _multi_source_dataset(tmp_path / "ds", prefixes=("x", "y", "z"), tiles=1)
    out = tmp_path / "m"
    group_key_map = {"2-11-26/x_0_0": "gA", "2-11-26/y_0_0": "gA", "2-11-26/z_0_0": "gB"}
    result = make_splits(str(root), output_path=str(out), seed=1,
                         train_ratio=0.5, val_ratio=0.5, test_ratio=0.0,
                         group_by="tile_prefix", group_key_map=group_key_map, subject="catkin")
    assert "error" not in result
    assert result["group_by"] == "explicit_map"

    membership: dict[str, str] = {}
    for split in ("train", "val"):
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
    result = make_splits(str(root), output_path=str(out), group_by="not_a_real_key", subject="catkin")
    assert "error" in result
    assert not out.exists() or not (out / "split_manifest.json").is_file()


def test_make_splits_refuses_to_write_a_manifest_with_no_subject(tmp_path: Path):
    """A manifest with no subject would be a partition of images, not of a run's admissible
    samples; make_splits refuses to write one rather than guessing what a run would admit."""
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out))
    assert "error" in result and "subject" in result["error"]
    assert not out.exists()


def _two_date_collision_dataset(root: Path, subject: str) -> Path:
    """One stem name, ``shared``, present under two capture dates with different content: a
    manifest keyed by bare stem could only ever hold one of the two."""
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
    return root


def test_two_dates_sharing_a_filename_produce_two_members(tmp_path: Path):
    root = _two_date_collision_dataset(tmp_path / "ds", subject="leaf")
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.5, test_ratio=0.0, seed=1)
    assert "error" not in result
    assert result["total_stems"] == 2
    manifest = ts.read(split_manifest_key(out))
    members = {identity for identities in manifest["splits"].values() for identity in identities}
    assert members == {"2-11-26/shared", "2-12-01/shared"}
    assert set(manifest["members"]) == {"2-11-26", "2-12-01"}


def _two_subject_dataset(root: Path) -> Path:
    """Four stems on one date: two carry ``leaf``, two carry the unrelated subject ``bud``, no
    stem carries both."""
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
    for stem, subject in (("leaf_a", "leaf"), ("leaf_b", "leaf"),
                         ("bud_a", "bud"), ("bud_b", "bud")):
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
                         train_ratio=0.5, val_ratio=0.5, test_ratio=0.0, seed=1)
    assert "error" not in result
    assert result["total_stems"] == 2
    manifest = ts.read(split_manifest_key(out))
    members = {identity for identities in manifest["splits"].values() for identity in identities}
    assert members == {"2-11-26/leaf_a", "2-11-26/leaf_b"}


def _attribute_scoped_dataset(root: Path) -> Path:
    """Three stems on one date, one subject: two have their instance assessed for ``condition``,
    one carries an instance never assessed for it."""
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
    for stem, condition in (("assessed_a", "healthy"), ("assessed_b", "damaged")):
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
                         train_ratio=0.5, val_ratio=0.5, test_ratio=0.0, seed=1)
    assert "error" not in result
    assert result["total_stems"] == 2
    manifest = ts.read(split_manifest_key(out))
    assert manifest["attribute"] == "condition"
    members = {identity for identities in manifest["splits"].values() for identity in identities}
    assert members == {"2-11-26/assessed_a", "2-11-26/assessed_b"}


def test_make_splits_refuses_to_materialize_a_multi_date_manifest(tmp_path: Path):
    root = _two_date_collision_dataset(tmp_path / "ds", subject="leaf")
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), materialize=True, subject="leaf",
                         train_ratio=0.5, val_ratio=0.5, test_ratio=0.0, seed=1)
    assert "error" in result
    assert "2-11-26" in result["error"] and "2-12-01" in result["error"]
    assert not out.exists()
