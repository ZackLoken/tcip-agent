"""The fingerprint formula-version prefix (Part 14 Q2): two values computed under different
formulas must never compare as equal or unequal by accident, so dataset_fingerprint stamps its
own formula version and every comparator that reads a stored value checks for it first.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import tcip_mcp.experiments as exp
from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint, fingerprint_formula_version


@pytest.fixture
def experiments_dir(tmp_path: Path):
    original = exp.EXPERIMENTS_DIR
    exp.EXPERIMENTS_DIR = tmp_path / "experiments"
    yield tmp_path
    exp.EXPERIMENTS_DIR = original


def _real_dataset(tmp_path: Path) -> Path:
    """A tiny dataset with one image and one label, the platform's own producer's shape."""
    root = tmp_path / "dataset"
    images = root / "images" / "2024-01-01"
    labels = root / "annotations" / "2024-01-01"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (10, 10), (1, 2, 3)).save(images / "a.png")
    (labels / "a.json").write_text(
        '{"image": "a", "width": 10, "height": 10, "annotations": []}', encoding="utf-8"
    )
    return root


def test_dataset_fingerprint_carries_its_formula_version_as_a_prefix(tmp_path):
    fp = dataset_fingerprint(_real_dataset(tmp_path))
    assert fp is not None
    assert fp.startswith("v1:")
    assert fingerprint_formula_version(fp) == 1


def test_fingerprint_formula_version_names_none_for_a_bare_legacy_value():
    assert fingerprint_formula_version("deadbeefcafef00d") is None
    assert fingerprint_formula_version(None) is None
    assert fingerprint_formula_version(42) is None


def test_compare_experiments_reports_same_dataset_for_two_matching_prefixed_fingerprints(
    experiments_dir,
):
    exp.create_experiment("e1", {}, dataset_fingerprint="v1:aaaa")
    exp.create_experiment("e2", {}, dataset_fingerprint="v1:aaaa")

    result = exp.compare_experiments(["e1", "e2"])
    assert result["same_dataset_fingerprint"] is True
    assert all("fingerprint_formula_unrecorded" not in c for c in result["experiments"])


def test_compare_experiments_forces_none_and_flags_a_bare_legacy_fingerprint(experiments_dir):
    exp.create_experiment("e1", {}, dataset_fingerprint="v1:aaaa")
    exp.create_experiment("e2", {}, dataset_fingerprint="aaaa")  # bare, pre-family value

    result = exp.compare_experiments(["e1", "e2"])
    assert result["same_dataset_fingerprint"] is None
    by_id = {c["experiment_id"]: c for c in result["experiments"]}
    assert by_id["e2"].get("fingerprint_formula_unrecorded") is True
    assert "fingerprint_formula_unrecorded" not in by_id["e1"]


def test_compare_experiments_flags_a_non_current_formula_version_not_only_a_bare_value(
    experiments_dir,
):
    # Not bare (fingerprint_formula_version parses it): must compare against the current formula.
    exp.create_experiment("e1", {}, dataset_fingerprint="v1:aaaa")
    exp.create_experiment("e2", {}, dataset_fingerprint="v2:aaaa")

    result = exp.compare_experiments(["e1", "e2"])
    assert result["same_dataset_fingerprint"] is None
    by_id = {c["experiment_id"]: c for c in result["experiments"]}
    assert by_id["e2"].get("fingerprint_formula_unrecorded") is True
    assert "fingerprint_formula_unrecorded" not in by_id["e1"]
