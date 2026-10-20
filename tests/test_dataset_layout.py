"""Tests for the canonical dataset-layout resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts
from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.dataset_layout import (
    UNDATED_BUCKET,
    capture_subjects,
    label_key,
    parse_image_path,
)


def test_parse_image_path_date_nested() -> None:
    root, date, stem = parse_image_path("/ds/images/2-11-26/IMG_1.JPG")
    assert Path(root) == Path("/ds")
    assert date == "2-11-26"
    assert stem == "IMG_1"


@pytest.mark.parametrize(
    "path",
    ["/ds/images/IMG_1.JPG", "/somewhere/random/IMG_1.JPG"],
    ids=["flat", "no-image-tree"],
)
def test_parse_image_path_refuses_an_image_of_no_capture(path: str) -> None:
    with pytest.raises(ValueError, match="is not a capture"):
        parse_image_path(path)


def test_a_flat_image_beside_an_undated_one_of_its_stem_is_no_second_image(
    tmp_path: Path,
) -> None:
    """A flat image beside an undated one of its stem is no second image.

    With ``images/same.png`` beside ``images/undated/same.png``, the census counts one image, the
    bucketed one's label key is its capture and stem, and the flat one is refused a label key.
    """
    from tcip_mcp.dataset_layout import label_key_of
    from tcip_mcp.tools.data_tools import scan_dataset

    (tmp_path / "images" / UNDATED_BUCKET).mkdir(parents=True)
    for place in (tmp_path / "images", tmp_path / "images" / UNDATED_BUCKET):
        (place / "same.png").write_bytes(b"\x89PNG")

    scan = scan_dataset(str(tmp_path))

    assert scan["image_count"] == 1, scan
    assert label_key_of(
        tmp_path / "images" / UNDATED_BUCKET / "same.png"
    ).parts == (UNDATED_BUCKET, "same")
    with pytest.raises(ValueError, match="is not a capture"):
        label_key_of(tmp_path / "images" / "same.png")


def test_an_images_label_key_is_its_capture_and_stem() -> None:
    # One document per image, keyed by its capture and stem; no subject or task in the key.
    assert label_key("/ds", "2-11-26", "IMG_1").parts == ("2-11-26", "IMG_1")
    assert label_key(
        *parse_image_path("/ds/images/2-11-26/IMG_1.JPG")
    ) == label_key("/ds", "2-11-26", "IMG_1")


def test_capture_subjects_is_per_capture(tmp_path: Path) -> None:
    root = tmp_path
    # Subjects are read from the per-image label records (the key never encodes them).
    # bud labeled on 2026-02-11 only; bush labeled on 2026-03-02 only.
    json_io.write_label_document(
        label_key(root, "2026-02-11", "IMG_1"),
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))],
        100,
        100,
    )
    json_io.write_label_document(
        label_key(root, "2026-03-02", "IMG_9"),
        [Annotation(subject="bush", geometry=BBox(1, 1, 9, 9))],
        100,
        100,
    )
    # A second image on 2026-03-02 carries bud, so that date offers both subjects.
    json_io.write_label_document(
        label_key(root, "2026-03-02", "IMG_5"),
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))],
        100,
        100,
    )

    assert capture_subjects(root, "2026-02-11") == (["bud"], [])
    # 2026-03-02 has bush and bud → both, sorted.
    assert capture_subjects(root, "2026-03-02") == (["bud", "bush"], [])
    # A date with no labels for any subject → nothing to offer.
    assert capture_subjects(root, "2026-03-24") == ([], [])


def test_capture_subjects_names_an_unreadable_label_and_lists_the_rest(
    tmp_path: Path,
) -> None:
    ts.replace(
        label_key(tmp_path, "2026-02-11", "IMG_1"), ["not", "a", "document"]
    )
    json_io.write_label_document(
        label_key(tmp_path, "2026-02-11", "IMG_2"),
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))],
        100,
        100,
    )

    assert capture_subjects(tmp_path, "2026-02-11") == (["bud"], ["IMG_1"])


def test_subjects_path_is_the_single_dataset_registry() -> None:
    from tcip_mcp.dataset_layout import subjects_path

    # One nested registry at the dataset root: no per-subject classes/<x>.json anymore.
    assert subjects_path("/ds") == Path("/ds/subjects.json")


def test_every_reader_of_a_dataset_root_answers_the_root_the_layout_wrote(
    tmp_path: Path,
) -> None:
    """Every reader of a dataset root answers the root its image and capture were written under.

    For a capture named ``images``, and for a dataset nested under an ancestor named ``images``,
    each registered, a bucket's source root, an image's capture, a capture directory's parse, the
    root of a capture directory or image tree, and an audit scope each answer the root
    :func:`image_path` was handed. For a registered root itself named ``images``, a raster in its
    image tree answers that root and no capture; a directory neither root claims refuses naming
    both, and a path under no image tree answers none.
    """
    from tcip_mcp.audit import dataset_scope_of
    from tcip_mcp.buckets import source_root
    from tcip_mcp.dataset_layout import (
        dataset_root_of,
        image_capture,
        image_dir,
        image_path,
        image_root,
        parse_capture_dir,
    )
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.traits import registered_crops

    crop = sorted(registered_crops())[0]
    for root, capture in (
        (tmp_path / "data", "images"),
        (tmp_path / "images" / "proj", "2026-03-02"),
    ):
        image = image_path(root, capture, "a", ".png")
        image.parent.mkdir(parents=True)
        image.write_bytes(b"\x89PNG")
        assert "error" not in register_dataset(tmp_path, str(root), crop=crop)
        assert image_capture(image) == (root, capture)
        assert parse_capture_dir(image_dir(root, capture)) == (root, capture)
        assert source_root([image]) == root
        for directory in (image_dir(root, capture), image_root(root)):
            assert dataset_root_of(directory) == root
            assert dataset_scope_of(directory) == root
        assert dataset_scope_of(image) == root

    named = tmp_path / "scratch" / "images"
    raster = image_root(named) / "raster.tif"
    raster.parent.mkdir(parents=True)
    raster.write_bytes(b"II*\x00")
    assert "error" not in register_dataset(tmp_path, str(named), crop=crop)
    assert source_root([raster]) == named
    assert image_capture(raster) == (None, None)
    assert dataset_scope_of(raster) == named
    assert dataset_root_of(image_root(named)) == named
    assert dataset_scope_of(image_root(named)) == named
    with pytest.raises(ValueError, match="is not a capture"):
        parse_capture_dir(image_root(named))

    unclaimed = image_root(tmp_path / "bare" / "images")
    with pytest.raises(ValueError, match="neither carries") as refused:
        dataset_root_of(unclaimed)
    assert str(tmp_path / "bare") in str(refused.value)
    assert str(unclaimed.parent) in str(refused.value)
    assert image_capture("/some/where/else.png") == (None, None)
    assert dataset_root_of("/some/where/else") is None


def test_a_capture_directory_sits_under_the_image_root() -> None:
    from tcip_mcp.dataset_layout import image_dir, image_root

    assert image_dir("/ds", "2026-03-02") == image_root("/ds") / "2026-03-02"
    with pytest.raises(ValueError, match="single safe path segment"):
        image_dir("/ds", "../escape")


def test_buckets_under_lists_every_published_bucket_and_nothing_else(
    tmp_path: Path,
) -> None:
    """``buckets_under`` lists every published bucket, its name dated or not, and nothing else.

    A published bucket is a record. A prediction document no record names is none.
    """
    pytest.importorskip("torch")
    from tcip_mcp.buckets import buckets_under
    from tcip_mcp.dataset_layout import prediction_key
    from tests._chain_fixtures import predicted, published

    root = tmp_path
    for name in ("modelA/2026-03-02", "modelB"):
        published(
            root,
            name,
            [predicted(root / "images" / "IMG_1.png", ["bud"])],
            scope={"subject": "bud"},
        )
    json_io.write_label_document(
        prediction_key(root, "modelC/2026-03-02", "IMG_1"),
        [],
        8,
        8,
        keep_empty=True,
    )

    assert [b.name for b in buckets_under(root)] == [
        "modelA/2026-03-02",
        "modelB",
    ]
