"""Tests for data management tools."""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.tools.data_tools import load_dataset, validate_data_quality, split_dataset


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


def test_split_dataset(data_dir: Path, tmp_path: Path):
    out = tmp_path / "splits"
    result = split_dataset(str(data_dir), output_path=str(out))
    assert result["total"] == 3
    assert sum(result["splits"].values()) == 3
    assert (out / "train.json").is_file()
    assert (out / "val.json").is_file()
    assert (out / "test.json").is_file()


def test_split_dataset_bad_ratios(data_dir: Path):
    result = split_dataset(str(data_dir), train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)
    assert "error" in result
