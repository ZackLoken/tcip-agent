"""Tests for data management tools."""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.tools.data_tools import (
    load_dataset,
    validate_data_quality,
    make_splits,
)


def test_load_dataset(data_dir: Path):
    result = load_dataset(str(data_dir))
    assert result["image_count"] == 3
    assert result["labels_detect_count"] == 3
    assert result["paired_images"] == 3
    assert result["unlabelled_images"] == 0


def test_load_dataset_not_found():
    result = load_dataset("/nonexistent/path")
    assert "error" in result


def test_validate_data_quality(data_dir: Path):
    result = validate_data_quality(str(data_dir))
    assert result["total_images"] == 3
    assert result["total_labels"] == 3
    assert result["is_valid"] is True
    assert 0 in result["class_ids"]


def test_make_splits_materialize(data_dir: Path, tmp_path: Path):
    out = tmp_path / "splits"
    result = make_splits(str(data_dir), output_path=str(out), materialize=True)
    assert result["total_stems"] == 3
    assert sum(result["splits"].values()) == 3
    assert result["output_dir"] == str(out)
    for split in ("train", "val", "test"):
        assert (out / f"{split}.json").is_file()
        assert (out / split / "images").is_dir()
        assert (out / split / "labels").is_dir()
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
    for split in ("train", "val", "test"):
        assert (out / f"{split}.json").is_file()


def test_make_splits_bad_ratios(data_dir: Path):
    result = make_splits(str(data_dir), train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)
    assert "error" in result


def _multi_source_dataset(root: Path, prefixes=("srcA", "srcB", "srcC", "srcD"), tiles=3) -> Path:
    from PIL import Image

    date = "2-11-26"
    images_dir = root / "images" / date
    images_dir.mkdir(parents=True)
    labels_dir = root / "annotations" / "default" / date / "detect"
    labels_dir.mkdir(parents=True)
    for pref in prefixes:
        for t in range(tiles):
            stem = f"{pref}_{t}_0"
            Image.new("RGB", (64, 64), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
            (labels_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    return root


def test_make_splits_groups_tiles_together(tmp_path: Path):
    import json

    from tcip_mcp.pipelines.data.splits import default_group_key

    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), seed=1)
    assert result["groups"] == 4  # 4 source prefixes, not 12 tiles

    # No source prefix may appear in more than one split.
    seen: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for stem in json.loads((out / f"{split}.json").read_text()):
            g = default_group_key(stem)
            assert seen.get(g, split) == split, f"group {g} spans splits"
            seen[g] = split
