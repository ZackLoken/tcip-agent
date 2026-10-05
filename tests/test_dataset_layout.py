"""Tests for the canonical dataset-layout resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts
from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox

from tcip_mcp.dataset_layout import (
    UNDATED_BUCKET,
    label_key,
    capture_subjects,
    parse_image_path,
)


def test_parse_image_path_date_nested() -> None:
    root, date, stem = parse_image_path("/ds/images/2-11-26/IMG_1.JPG")
    assert Path(root) == Path("/ds")
    assert date == "2-11-26"
    assert stem == "IMG_1"


@pytest.mark.parametrize("path", ["/ds/images/IMG_1.JPG", "/somewhere/random/IMG_1.JPG"],
                         ids=["flat", "no-image-tree"])
def test_parse_image_path_refuses_an_image_of_no_capture(path: str) -> None:
    with pytest.raises(ValueError, match="is not a capture"):
        parse_image_path(path)


def test_a_flat_image_beside_an_undated_one_of_its_stem_is_no_second_image(tmp_path: Path) -> None:
    """``images/same.png`` beside ``images/undated/same.png``: the census lists the bucketed one
    alone, keyed by its own capture, and the flat one has no label key."""
    from tcip_mcp.dataset_layout import label_key_of
    from tcip_mcp.tools.data_tools import scan_dataset

    (tmp_path / "images" / UNDATED_BUCKET).mkdir(parents=True)
    for place in (tmp_path / "images", tmp_path / "images" / UNDATED_BUCKET):
        (place / "same.png").write_bytes(b"\x89PNG")

    scan = scan_dataset(str(tmp_path))

    assert scan["image_count"] == 1, scan
    assert label_key_of(tmp_path / "images" / UNDATED_BUCKET / "same.png").parts == (
        UNDATED_BUCKET, "same")
    with pytest.raises(ValueError, match="is not a capture"):
        label_key_of(tmp_path / "images" / "same.png")


def test_an_images_label_key_is_its_capture_and_stem() -> None:
    # One document per image, keyed by its capture and stem; no subject or task in the key.
    assert label_key("/ds", "2-11-26", "IMG_1").parts == ("2-11-26", "IMG_1")
    assert label_key(*parse_image_path("/ds/images/2-11-26/IMG_1.JPG")) == label_key(
        "/ds", "2-11-26", "IMG_1")


def test_capture_subjects_is_per_capture(tmp_path: Path) -> None:
    root = tmp_path
    # Subjects are read from the per-image label records (the key never encodes them).
    # bud labeled on 2026-02-11 only; bush labeled on 2026-03-02 only.
    json_io.write_label_document(label_key(root, "2026-02-11", "IMG_1"),
                                 [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 100, 100)
    json_io.write_label_document(label_key(root, "2026-03-02", "IMG_9"),
                                 [Annotation(subject="bush", geometry=BBox(1, 1, 9, 9))], 100, 100)
    # A second image on 2026-03-02 carries bud, so that date offers both subjects.
    json_io.write_label_document(label_key(root, "2026-03-02", "IMG_5"),
                                 [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 100, 100)

    assert capture_subjects(root, "2026-02-11") == (["bud"], [])
    # 2026-03-02 has bush and bud → both, sorted.
    assert capture_subjects(root, "2026-03-02") == (["bud", "bush"], [])
    # A date with no labels for any subject → nothing to offer.
    assert capture_subjects(root, "2026-03-24") == ([], [])


def test_capture_subjects_names_an_unreadable_label_and_lists_the_rest(tmp_path: Path) -> None:
    ts.replace(label_key(tmp_path, "2026-02-11", "IMG_1"), ["not", "a", "document"])
    json_io.write_label_document(label_key(tmp_path, "2026-02-11", "IMG_2"),
                                 [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 100, 100)

    assert capture_subjects(tmp_path, "2026-02-11") == (["bud"], ["IMG_1"])


def test_subjects_path_is_the_single_dataset_registry():
    from tcip_mcp.dataset_layout import subjects_path

    # One nested registry at the dataset root: no per-subject classes/<x>.json anymore.
    assert subjects_path("/ds") == Path("/ds/subjects.json")


def test_dataset_root_of_recovers_the_root_from_its_image_tree():
    from tcip_mcp.dataset_layout import dataset_root_of

    assert dataset_root_of("/ds/images/2026-03-02") == Path("/ds")
    assert dataset_root_of("/ds/images/2026-03-02/IMG_1.JPG") == Path("/ds")
    assert dataset_root_of("/some/where/else") is None
    # Anchors on the last images segment: a dataset nested under an ancestor named 'images'
    # still resolves to the real root, not the ancestor.
    assert dataset_root_of("/data/images/proj/images/2026-03-02") == Path("/data/images/proj")
    # A bare segment with nothing above it is not inside a dataset.
    assert dataset_root_of("images/2026-03-02") is None


def test_a_capture_directory_sits_under_the_image_root() -> None:
    from tcip_mcp.dataset_layout import image_dir, image_root

    assert image_dir("/ds", "2026-03-02") == image_root("/ds") / "2026-03-02"
    with pytest.raises(ValueError, match="single safe path segment"):
        image_dir("/ds", "../escape")


def test_buckets_under_lists_every_published_bucket_and_nothing_else(tmp_path: Path) -> None:
    """A published bucket is a record, its name dated or not: the one listing the doctor command
    reads through. A prediction document no record names is none."""
    pytest.importorskip("torch")
    from tcip_mcp.buckets import buckets_under
    from tcip_mcp.dataset_layout import prediction_key
    from tests._chain_fixtures import predicted, published

    root = tmp_path
    for name in ("modelA/2026-03-02", "modelB"):
        published(root, name, [predicted(root / "images" / "IMG_1.png", ["bud"])],
                  scope={"subject": "bud"})
    json_io.write_label_document(prediction_key(root, "modelC/2026-03-02", "IMG_1"), [], 8, 8,
                                 keep_empty=True)

    assert [b.name for b in buckets_under(root)] == ["modelA/2026-03-02", "modelB"]
