"""Carrying human-confirmed negatives into a materialized split tree.

A split tree is ``{train,val,test}/{images,labels}/`` and cannot recover a subject or a date from
its own path, so every confirmation it inherits has to be re-attributed explicitly: to the right
split, under the key the resolver computes for a dateless tree, and with the status token that
means "a human looked and found nothing" rather than the token that means "a human finished an
image that has content on it".
"""

from __future__ import annotations

import json
from pathlib import Path

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox

from tcip_mcp.dataset_layout import (
    image_status_digest_path,
    image_status_path,
    status_bucket,
)
from tcip_mcp.tools.data_tools import make_splits

DATE = "2-11-26"
SUBJECT = "catkin"
NEGATIVE_STEM = "plotF_0_0"
POPULATED_STEMS = ("plotA_0_0", "plotB_0_0", "plotC_0_0", "plotD_0_0", "plotE_0_0")


def _dataset_with_one_confirmed_negative(root: Path) -> Path:
    """Six sources on a non-square frame: five carrying annotations, one an empty label a human
    confirmed negative for ``catkin``. The counts per source differ so the split is not symmetric.
    """
    from PIL import Image

    from tcip_mcp.class_registry import ClassRegistry, Subject, write_registry

    images_dir = root / "images" / DATE
    labels_dir = root / "annotations" / DATE
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    write_registry(root / "classes.json", ClassRegistry(subjects=(
        Subject(name=SUBJECT, description="a hazelnut catkin"),
    )))

    for i, stem in enumerate(POPULATED_STEMS):
        Image.new("RGB", (96, 64), (90, 120, 60)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject=SUBJECT, geometry=BBox(5 + 3 * k, 7, 25 + 3 * k, 51))
             for k in range(i + 1)],
            96, 64,
        )
    Image.new("RGB", (96, 64), (90, 120, 60)).save(images_dir / f"{NEGATIVE_STEM}.jpg")
    json_io.write_annotations(labels_dir / f"{NEGATIVE_STEM}.json", [], 96, 64, keep_empty=True)

    store = image_status_path(root)
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps(
        {status_bucket(SUBJECT, DATE): {f"{NEGATIVE_STEM}.jpg": "negative"}}, indent=2,
    ), encoding="utf-8")
    return root


def _materialize(tmp_path: Path, *, subject: str | None) -> tuple[Path, dict]:
    root = _dataset_with_one_confirmed_negative(tmp_path / "ds")
    out = tmp_path / "splits"
    result = make_splits(
        str(root), output_path=str(out), materialize=True, subject=subject,
        train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=3,
    )
    assert "error" not in result
    assert all(result["splits"][name] > 0 for name in ("train", "val", "test"))
    return out, result


def _split_holding(out: Path, image_name: str) -> str:
    holders = [name for name in ("train", "val", "test")
               if (out / name / "images" / image_name).is_file()]
    assert len(holders) == 1, f"{image_name} landed in {holders}"
    return holders[0]


def test_confirmation_is_carried_only_into_the_split_that_holds_its_image(tmp_path: Path):
    """One confirmed negative, one status store: the splits that do not hold the image inherit
    no confirmation for it."""
    out, _ = _materialize(tmp_path, subject=SUBJECT)
    holder = _split_holding(out, f"{NEGATIVE_STEM}.jpg")

    with_store = [name for name in ("train", "val", "test")
                  if image_status_path(out / name).is_file()]

    assert with_store == [holder]


def test_carried_confirmation_is_stored_under_the_dateless_bucket_as_a_negative(tmp_path: Path):
    """The store lands where ``image_status_path`` resolves it, keyed by the bucket
    ``status_bucket`` computes for a tree that carries no date, holding the negative token."""
    out, _ = _materialize(tmp_path, subject=SUBJECT)
    holder = _split_holding(out, f"{NEGATIVE_STEM}.jpg")

    stored = json.loads(image_status_path(out / holder).read_text(encoding="utf-8"))

    assert list(stored) == [status_bucket(SUBJECT, None)]
    assert stored[status_bucket(SUBJECT, None)] == {f"{NEGATIVE_STEM}.jpg": "negative"}


def test_carried_confirmation_round_trips_through_the_confirmed_negative_reader(tmp_path: Path):
    """The reader every training path uses recovers the confirmation from the split tree, so the
    image trains as a hard negative instead of an unconfirmed empty."""
    from tcip_mcp.pipelines.data.datasets import confirmed_negative_names

    out, _ = _materialize(tmp_path, subject=SUBJECT)
    holder = _split_holding(out, f"{NEGATIVE_STEM}.jpg")

    recovered = confirmed_negative_names(out / holder / "labels", subject=SUBJECT)

    assert recovered == {f"{NEGATIVE_STEM}.jpg"}


def test_carried_schema_stamp_is_recorded_per_image_not_per_bucket(tmp_path: Path):
    """The stamp travels alongside the confirmation, per image, and matches the digest of the
    registry copied in beside it. A bucket-wide stamp would be overwritten by the next unrelated
    write to that bucket."""
    from tcip_mcp.class_registry import attribute_schema_digest, read_registry

    out, _ = _materialize(tmp_path, subject=SUBJECT)
    holder = _split_holding(out, f"{NEGATIVE_STEM}.jpg")
    split_root = out / holder

    assert (split_root / "classes.json").is_file()
    expected = attribute_schema_digest(read_registry(split_root / "classes.json"), SUBJECT)
    assert expected is not None
    stamps = json.loads(image_status_digest_path(split_root).read_text(encoding="utf-8"))

    assert stamps == {status_bucket(SUBJECT, None): {f"{NEGATIVE_STEM}.jpg": expected}}


def test_no_subject_threaded_carries_no_confirmation(tmp_path: Path):
    """With no subject to attribute them to, no confirmation is carried and no split gets a
    status store: an unattributed confirmation would apply one subject's judgement to every
    other subject."""
    out, _ = _materialize(tmp_path, subject=None)

    assert [name for name in ("train", "val", "test")
            if image_status_path(out / name).is_file()] == []
