"""Carrying human-confirmed negatives into a materialized split tree.

A split tree is ``{train,val,calibration}/{images,labels}/`` and cannot recover a subject or a date from
its own path, so every confirmation it inherits has to be re-attributed explicitly: to the right
split, under the key the resolver computes for a dateless tree, and with the status token that
means "a human looked and found nothing" rather than the token that means "a human finished an
image that has content on it".
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tcip_store as ts
from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox

from tcip_mcp.dataset_layout import (
    image_status_digest_key,
    image_status_key,
    status_bucket,
)
from tcip_mcp.tools.data_tools import draw_splits

DATE = "2-11-26"
SUBJECT = "bud"
NEGATIVE_STEM = "plotF_0_0"
CONFIRMED_BY = "user:breeder"
POPULATED_STEMS = ("plotA_0_0", "plotB_0_0", "plotC_0_0", "plotD_0_0", "plotE_0_0")


def _dataset_with_one_confirmed_negative(root: Path) -> Path:
    """Six sources on a non-square frame: five carrying annotations, one an empty label a human
    confirmed negative for ``bud``. The counts per source differ so the split is not symmetric.
    """
    from PIL import Image

    from tcip_mcp.class_registry import ClassRegistry, Subject, write_registry

    images_dir = root / "images" / DATE
    labels_dir = root / "annotations" / DATE
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    write_registry(root / "classes.json", ClassRegistry(subjects=(
        Subject(name=SUBJECT, description="a currant bud"),
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

    from tcip_mcp.dataset_layout import record_image_statuses

    record_image_statuses(root, status_bucket(SUBJECT, DATE),
                          {f"{NEGATIVE_STEM}.jpg": "negative"}, recorded_by=CONFIRMED_BY)
    return root


def _materialize(tmp_path: Path, *, subject: str | None) -> tuple[Path, dict]:
    root = _dataset_with_one_confirmed_negative(tmp_path / "ds")
    out = tmp_path / "splits"
    result = draw_splits(
        str(root), output_path=str(out), materialize=True, subject=subject,
        train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25, seed=3,
    )
    assert "error" not in result, result
    assert all(result["splits"][name] > 0 for name in ("train", "val", "calibration"))
    return out, result


def _split_holding(out: Path, image_name: str) -> str:
    holders = [name for name in ("train", "val", "calibration")
               if (out / name / "images" / image_name).is_file()]
    assert len(holders) == 1, f"{image_name} landed in {holders}"
    return holders[0]


def test_confirmation_is_carried_only_into_the_split_that_holds_its_image(tmp_path: Path):
    """One confirmed negative, one status store: the splits that do not hold the image inherit
    no confirmation for it."""
    out, _ = _materialize(tmp_path, subject=SUBJECT)
    holder = _split_holding(out, f"{NEGATIVE_STEM}.jpg")

    with_store = [name for name in ("train", "val", "calibration")
                  if ts.exists(image_status_key(out / name))]

    assert with_store == [holder]


def test_carried_confirmation_is_stored_under_the_dateless_bucket_as_a_negative(tmp_path: Path):
    """The store lands where ``image_status_path`` resolves it, keyed by the bucket
    ``status_bucket`` computes for a tree that carries no date, holding the negative token."""
    out, _ = _materialize(tmp_path, subject=SUBJECT)
    holder = _split_holding(out, f"{NEGATIVE_STEM}.jpg")

    stored = ts.read(image_status_key(out / holder))

    assert list(stored) == [status_bucket(SUBJECT, None)]
    assert list(stored[status_bucket(SUBJECT, None)]) == [f"{NEGATIVE_STEM}.jpg"]
    assert stored[status_bucket(SUBJECT, None)][f"{NEGATIVE_STEM}.jpg"]["status"] == "negative"


def test_the_carried_confirmation_still_names_who_made_it(tmp_path: Path):
    """The split records the person the source dataset recorded, not the tool that copied it:
    re-attributing a carried confirmation would credit a human's work to a split writer."""
    out, _ = _materialize(tmp_path, subject=SUBJECT)
    holder = _split_holding(out, f"{NEGATIVE_STEM}.jpg")

    stored = ts.read(image_status_key(out / holder))
    carried = stored[status_bucket(SUBJECT, None)][f"{NEGATIVE_STEM}.jpg"]
    source = ts.read(image_status_key(tmp_path / "ds"))

    assert carried["recorded_by"] == CONFIRMED_BY
    assert carried == source[status_bucket(SUBJECT, DATE)][f"{NEGATIVE_STEM}.jpg"]


def test_carried_confirmation_round_trips_through_the_confirmed_negative_reader(tmp_path: Path):
    """The reader every training path uses recovers the confirmation from the split tree, so the
    image trains as a hard negative instead of an unconfirmed empty."""
    from tcip_mcp.pipelines.data.label_queries import confirmed_negative_names

    out, _ = _materialize(tmp_path, subject=SUBJECT)
    holder = _split_holding(out, f"{NEGATIVE_STEM}.jpg")

    recovered = confirmed_negative_names(out / holder / "labels", subject=SUBJECT, date=None)

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
    stamps = ts.read(image_status_digest_key(split_root))

    assert stamps == {status_bucket(SUBJECT, None): {f"{NEGATIVE_STEM}.jpg": expected}}


def test_materialize_with_no_subject_is_refused(tmp_path: Path):
    """Materializing with no subject to attribute confirmations to would silently drop every
    human-confirmed negative from the drawn membership as well as from the carry, so draw_splits
    refuses rather than materializing an unattributed split tree."""
    root = _dataset_with_one_confirmed_negative(tmp_path / "ds")
    out = tmp_path / "splits"

    result = draw_splits(
        str(root), output_path=str(out), materialize=True, subject=None,
        train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25, seed=3,
    )

    assert "error" in result
    assert not out.exists()


def test_carried_registry_declares_the_same_document_as_the_source(tmp_path: Path):
    """The split's registry declares what the source declares: a digest or an id assignment
    taken against either one has to read the same declared order.

    Every record now shares one spelling, so equal bytes no longer distinguish the carried
    document from a re-serialization of it; what this pins is that nothing was dropped,
    re-ordered or respelled on the way into the split.
    """
    root = _dataset_with_one_confirmed_negative(tmp_path / "ds")
    out = tmp_path / "splits"
    result = draw_splits(
        str(root), output_path=str(out), materialize=True, subject=SUBJECT,
        train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25, seed=3,
    )
    assert "error" not in result
    holder = _split_holding(out, f"{NEGATIVE_STEM}.jpg")

    assert (out / holder / "classes.json").read_bytes() == (root / "classes.json").read_bytes()


def test_a_contradicted_negative_is_excluded_from_the_carry_and_named_in_the_result(
    tmp_path: Path,
):
    """A stored negative whose label file now holds the subject is not carried into the split as a
    negative (it lands there as an ordinary annotated stem, through its real content), and the
    contradiction is named in the result rather than left for a separate doctor pass."""
    root = _dataset_with_one_confirmed_negative(tmp_path / "ds")
    labels_dir = root / "annotations" / DATE
    json_io.write_annotations(
        labels_dir / f"{NEGATIVE_STEM}.json",
        [Annotation(subject=SUBJECT, geometry=BBox(5, 7, 25, 51))], 96, 64,
    )
    out = tmp_path / "splits"

    result = draw_splits(
        str(root), output_path=str(out), materialize=True, subject=SUBJECT,
        train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25, seed=3,
    )

    assert result.get("contradicted_negatives") == [f"{NEGATIVE_STEM}.jpg"]
    holder = _split_holding(out, f"{NEGATIVE_STEM}.jpg")
    assert not ts.exists(image_status_key(out / holder))


def test_a_failed_carry_computation_writes_no_split_tree(tmp_path: Path, monkeypatch):
    """The carry is computed before any split-tree file is written, so a failure while computing
    it must never leave a partial tree on disk."""
    import tcip_mcp.tools.data_tools as data_tools_mod

    root = _dataset_with_one_confirmed_negative(tmp_path / "ds")
    out = tmp_path / "splits"

    def _boom(*_a, **_kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(data_tools_mod, "_compute_negative_carry", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        draw_splits(str(root), output_path=str(out), materialize=True, subject=SUBJECT,
                    train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25, seed=3)

    assert not (out / "train").exists()
    assert not (out / "val").exists()
    assert not (out / "calibration").exists()


def test_an_unreadable_confirmed_negative_persists_nothing(tmp_path: Path):
    """A confirmed negative whose label file will not read is caught by the admission's own read,
    before the split's own manifest or tree is written: the call answers an error dict
    naming the file, and nothing from this call is left on disk."""
    from tcip_mcp.tools.data_tools import split_manifest_key

    root = _dataset_with_one_confirmed_negative(tmp_path / "ds")
    bad = root / "annotations" / DATE / f"{NEGATIVE_STEM}.json"
    bad.write_bytes(b"{not json")
    out = tmp_path / "splits"

    result = draw_splits(
        str(root), output_path=str(out), materialize=True, subject=SUBJECT,
        train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25, seed=3, stratify_foreground=False,
    )

    assert "error" in result
    assert str(bad) in result["error"]
    assert not ts.exists(split_manifest_key(out))
    assert not out.exists()
