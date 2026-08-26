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


def test_make_splits_materialize(data_dir: Path, tmp_path: Path):
    out = tmp_path / "splits"
    result = make_splits(str(data_dir), output_path=str(out), materialize=True)
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
    result = make_splits(str(data_dir), output_path=str(out))
    assert result["total_stems"] == 3
    assert result["groups"] == 3
    assert sum(result["splits"].values()) == 3
    assert result["stratified"] is True
    for split in ("train", "val"):
        assert ts.exists(split_stem_list_key(out, split))
    assert "test" not in result["splits"]

    manifest = ts.read(split_manifest_key(out))
    assert manifest["labels_root"] is not None
    assert Path(manifest["labels_root"]).is_dir()
    assert manifest["dataset_fingerprint"] is not None
    assert "test" not in manifest["splits"]


def test_make_splits_refuses_a_nonzero_test_ratio(data_dir: Path):
    """No launch path honours a held-out test list: make_splits refuses one rather than writing
    a partition nothing downstream reads."""
    result = make_splits(str(data_dir), train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    assert "error" in result
    assert "test_ratio" in result["error"]


def test_make_splits_bad_ratios(data_dir: Path):
    result = make_splits(str(data_dir), train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)
    assert "error" in result


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
    result = make_splits(str(root), output_path=str(out), seed=1)
    assert result["groups"] == 4  # 4 source prefixes, not 12 tiles

    # No source prefix may appear in more than one split.
    seen: dict[str, str] = {}
    for split in ("train", "val"):
        for stem in ts.read(split_stem_list_key(out, split)):
            g = default_group_key(stem)
            assert seen.get(g, split) == split, f"group {g} spans splits"
            seen[g] = split


def test_make_splits_group_key_map_never_straddles(tmp_path: Path):
    """An agent-derived group_key_map (3 stems, 2 groups) is honored: the two same-group stems
    never land in different splits."""
    root = _multi_source_dataset(tmp_path / "ds", prefixes=("x", "y", "z"), tiles=1)
    out = tmp_path / "m"
    group_key_map = {"x_0_0": "gA", "y_0_0": "gA", "z_0_0": "gB"}
    result = make_splits(str(root), output_path=str(out), seed=1,
                         train_ratio=0.5, val_ratio=0.5, test_ratio=0.0,
                         group_by="tile_prefix", group_key_map=group_key_map)
    assert "error" not in result
    assert result["group_by"] == "explicit_map"

    membership: dict[str, str] = {}
    for split in ("train", "val"):
        for stem in ts.read(split_stem_list_key(out, split)):
            membership[stem] = split
    assert membership["x_0_0"] == membership["y_0_0"]  # gA never straddles
    manifest = ts.read(split_manifest_key(out))
    assert manifest["group_by"] == "explicit_map"
    assert manifest["group_key_map"] == group_key_map


def test_make_splits_unrecognized_group_by_refuses_without_writing(tmp_path: Path):
    """A silent ``GROUP_KEY_FNS.get(group_by, default_group_key)`` fallback used to mis-group a
    dataset without anyone noticing; it must refuse loudly and write nothing instead."""
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), group_by="not_a_real_key")
    assert "error" in result
    assert not out.exists() or not (out / "split_manifest.json").is_file()


def _single_source_dataset(root: Path, width: int, height: int) -> Path:
    from PIL import Image

    images_dir = root / "images"
    labels_dir = root / "annotations"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    Image.new("RGB", (width, height), (128, 128, 128)).save(images_dir / "mosaic.jpg")
    json_io.write_annotations(
        labels_dir / "mosaic.json", [Annotation(subject="catkin", geometry=BBox(19, 13, 45, 51))],
        width, height,
    )
    return root


def test_make_splits_spatial_requires_tile_size_and_overlap(tmp_path: Path):
    root = _single_source_dataset(tmp_path / "ds", 1600, 1200)
    result = make_splits(str(root), spatial=True)
    assert "error" in result and "tile_size" in result["error"]


def test_make_splits_spatial_refuses_multi_source_folder(tmp_path: Path):
    root = _multi_source_dataset(tmp_path / "ds")
    result = make_splits(str(root), spatial=True, tile_size=128, overlap=0.2)
    assert "error" in result and "single-stem" in result["error"]


def test_make_splits_spatial_refuses_materialize(tmp_path: Path):
    root = _single_source_dataset(tmp_path / "ds", 1600, 1200)
    result = make_splits(str(root), spatial=True, tile_size=128, overlap=0.2, materialize=True)
    assert "error" in result and "materialize" in result["error"]


def test_make_splits_spatial_writes_strip_identity_manifest(tmp_path: Path):
    root = _single_source_dataset(tmp_path / "ds", 1600, 1200)
    out = tmp_path / "m"
    result = make_splits(str(root), spatial=True, tile_size=128, overlap=0.2,
                         train_ratio=0.75, val_ratio=0.25, test_ratio=0.0,
                         seed=1, output_path=str(out))
    assert "error" not in result
    assert result["group_by"] == "spatial_strip"
    assert result["splits"]["train"] > 0 and result["splits"]["val"] > 0

    train_ids = ts.read(split_stem_list_key(out, "train"))
    val_ids = ts.read(split_stem_list_key(out, "val"))
    assert train_ids and val_ids
    assert set(train_ids).isdisjoint(set(val_ids))
    assert all(i.startswith("mosaic::strip_") for i in train_ids + val_ids)

    manifest = ts.read(split_manifest_key(out))
    assert manifest["group_by"] == "spatial_strip"
    assert manifest["spatial"]["train_identities"] == train_ids
    assert manifest["labels_root"] is not None
    assert manifest["dataset_fingerprint"] is not None


def test_make_splits_spatial_refuses_a_nonzero_test_ratio(tmp_path: Path):
    """The spatial branch refuses a held-out test fraction the same way the grouped path does:
    no launch path honours a held-out test list."""
    root = _single_source_dataset(tmp_path / "ds", 4000, 3000)
    out = tmp_path / "m3"
    result = make_splits(str(root), spatial=True, tile_size=128, overlap=0.2,
                         train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                         seed=1, output_path=str(out))
    assert "error" in result
    assert "test_ratio" in result["error"]
    assert not out.exists()
