"""What the resolver reports a dataset contains: capture-date buckets, the published prediction
buckets of each date, and the difference between the subjects a registry declares and the
subjects a date's labels hold."""

from __future__ import annotations

from pathlib import Path

import pytest
from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox

from tcip_mcp.subject_registry import SubjectRegistry, RegistryError, Subject
from tests._producer_fixtures import registry_over
from tcip_mcp.dataset_layout import (
    annotation_dir,
    subjects_path,
    list_dates,
    list_subjects,
    prediction_root,
    subjects_with_labels,
)


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_only_published_buckets_are_listed_under_the_date_their_record_states(
    tmp_path: Path,
) -> None:
    """A directory under ``predictions/`` holding artifacts that are not a publication (a
    rendered overlay, a half-written temp file, a staged document) is no bucket: listing it would
    offer the breeder a model with nothing published behind it."""
    pytest.importorskip("torch")
    from tcip_mcp.buckets import buckets_by_date
    from tests._chain_fixtures import predicted, published

    root = tmp_path
    scope = {"subject": "bud"}
    _write(prediction_root(root) / "baseline" / "2026-02-11" / "overlay.png", "not a label")
    _write(prediction_root(root) / "baseline" / "2026-02-11" / "IMG_1.json",
           '{"annotations": []}')
    for model, date, stem in (("candidate", "2026-02-11", "IMG_1"),
                              ("baseline", "2026-03-02", "IMG_2")):
        published(root, prediction_root(root) / model / date,
                  [{**predicted(stem, ["bud"]),
                    "image": str(root / "images" / date / f"{stem}.png")}], scope=scope)

    listed = buckets_by_date(root, ["2026-02-11", "2026-03-02"])

    assert {date: sorted(names) for date, names in listed.items()} == {
        "2026-02-11": ["candidate/2026-02-11"], "2026-03-02": ["baseline/2026-03-02"]}


def test_capture_dates_are_the_bucket_directories_not_loose_images(tmp_path: Path) -> None:
    """A date bucket is a directory under ``images/``; an image sitting directly there is not one.

    The flat layout (``images/<stem>.<ext>``) is a legitimate non-dated dataset, so a loose image
    file must never be reported as a capture date to anything that iterates dates.
    """
    root = tmp_path
    (root / "images" / "2026-02-11").mkdir(parents=True)
    (root / "images" / "2026-03-02").mkdir()
    (root / "images" / "flat_capture.JPG").write_bytes(b"x")

    assert list_dates(root) == ["2026-02-11", "2026-03-02"]
    assert list_dates(tmp_path / "no_such_dataset") == []


def _registry(root: Path) -> None:
    """A registry declaring three subjects in an order that is not alphabetical."""
    registry_over(
        root,
        SubjectRegistry(subjects=(Subject(name="leaf"), Subject(name="bush"), Subject(name="bud"))),
    )


def test_registry_subjects_keep_their_declared_order(tmp_path: Path) -> None:
    """``list_subjects`` reports the registry's declared order, which is the order class ids are
    assigned in; re-ordering it would reindex a training run's classes against its labels."""
    root = tmp_path
    _registry(root)
    assert list_subjects(root) == ["leaf", "bush", "bud"]
    assert list_subjects(tmp_path / "no_such_dataset") == []


def test_a_registry_that_will_not_decode_is_not_reported_as_no_subjects(tmp_path: Path) -> None:
    """Absence and corruption are different answers on the subject registry.

    Reading unreadable bytes as an empty registry hands every caller a dataset that declares
    no subjects, which is indistinguishable from one that genuinely has none: the labels under
    it are name-based and undecodable without it, so the read refuses instead.
    """
    root = tmp_path
    _registry(root)
    subjects_path(root).write_bytes(b"{ this is not a registry")

    with pytest.raises(RegistryError):
        list_subjects(root)


def test_declared_subjects_and_subjects_labeled_on_a_date_are_not_interchangeable(
    tmp_path: Path,
) -> None:
    """The date's own labels, not the registry, decide what a date offers.

    A subject the registry declares but that nobody has labeled on the selected date lands the
    user on an empty canvas, so the two lists are different facts and each has its own source.
    """
    root = tmp_path
    _registry(root)
    json_io.write_annotations(
        str(annotation_dir(root, "2026-02-11") / "IMG_1.json"),
        [Annotation(subject="bud", geometry=BBox(1, 2, 7, 19))], 120, 90)
    json_io.write_annotations(
        str(annotation_dir(root, "2026-03-02") / "IMG_2.json"),
        [Annotation(subject="bush", geometry=BBox(3, 4, 40, 11))], 120, 90)

    assert list_subjects(root) == ["leaf", "bush", "bud"]
    assert subjects_with_labels(root, "2026-02-11") == ["bud"]
    assert subjects_with_labels(root, "2026-03-02") == ["bush"]
    assert subjects_with_labels(root, "2026-04-01") == []
