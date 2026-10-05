"""tcip doctor's check_data_quality: what it reports, and what it must never quietly claim.

Two standing facts the caller relies on. First, a label document the one per-image reader refuses
is a named finding for that document, never a reason to stop looking at the others. Second, the
finding's own vocabulary is load bearing: a warning and an error are not interchangeable.
"""

from __future__ import annotations

from pathlib import Path

import tcip_store as ts
from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox

from tcip_mcp.cli import doctor
from tcip_mcp.dataset_layout import label_key
from tests._producer_fixtures import write_image

DATE = "2-11-26"


def _check(root: Path) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    doctor.check_data_quality(root, findings, census=doctor._census(root, findings))
    return findings


def _errors(findings: list[tuple[str, str]]) -> list[str]:
    return [msg for level, msg in findings if level == "error"]


def _label(root: Path, stem: str, annotations, width: int = 96, height: int = 64,
           **kwargs) -> None:
    json_io.write_label_document(label_key(root, DATE, stem), annotations, width, height, **kwargs)


def test_an_unreadable_label_record_is_an_error_per_document(tmp_path: Path):
    """Label records of a shape the reader refuses are an error per document, naming the refusal;
    a dataset with no label documents at all yields no error, only the count of unlabeled
    images."""
    root = tmp_path / "unreadable"
    for stem in ("plotA_0_0", "plotA_0_1"):
        write_image(root / "images" / DATE / f"{stem}.jpg", (96, 64))
        ts.replace(label_key(root, DATE, stem),
                   {"shapes": [{"label": "bud", "points": [[3, 5], [40, 52]]}]})

    findings = _check(root)
    assert len(findings) == 2
    assert all(level == "error" and "label document will not read" in msg
               for level, msg in findings)

    bare = tmp_path / "unlabeled"
    for stem in ("plotA_0_0", "plotA_0_1"):
        write_image(bare / "images" / DATE / f"{stem}.jpg", (96, 64))

    findings = _check(bare)
    assert _errors(findings) == []
    assert [level for level, _ in findings] == ["info"]


def test_a_label_with_no_matching_image_is_an_error(tmp_path: Path):
    """An orphan label is an error-level finding."""
    root = tmp_path / "ds"
    for stem in ("plotA_0_0", "plotB_0_0"):
        write_image(root / "images" / DATE / f"{stem}.jpg", (96, 64))
    for stem in ("plotA_0_0", "plotB_0_0", "plotZ_9_9"):
        _label(root, stem, [Annotation(subject="bud", geometry=BBox(11, 7, 39, 51))])

    errors = _errors(_check(root))

    assert len(errors) == 1
    assert "plotZ_9_9" in errors[0] and "no matching image" in errors[0]


def test_an_unreadable_first_label_hides_no_later_finding(tmp_path: Path):
    """The census reads each document on its own, so an unreadable label sorting first is one
    finding and the orphan after it is still reported."""
    root = tmp_path / "ds"
    write_image(root / "images" / DATE / "0_bad.jpg", (96, 64))
    ts.replace(label_key(root, DATE, "0_bad"), ["not", "a", "document"])
    _label(root, "zzz_orphan", [Annotation(subject="bud", geometry=BBox(1, 1, 8, 8))])

    errors = _errors(_check(root))

    assert any("0_bad" in e and "will not read" in e for e in errors)
    assert any("zzz_orphan" in e and "no matching image" in e for e in errors)


def test_an_empty_label_not_marked_complete_is_an_error(tmp_path: Path):
    """A platform-written empty document is not an absent one, so a presence check never catches
    it; an empty label no person marked complete is unannotated, not a negative."""
    root = tmp_path / "ds"
    write_image(root / "images" / DATE / "plotA_0_0.jpg", (96, 64))
    _label(root, "plotA_0_0", [], keep_empty=True)

    errors = _errors(_check(root))

    assert len(errors) == 1
    assert "plotA_0_0" in errors[0] and "not marked complete" in errors[0]


def test_a_confirmed_negative_empty_label_stays_clean(tmp_path: Path):
    """The rail this suppression exists for: a human's Complete-with-nothing must not be flagged
    as though nobody had looked."""
    from tests._producer_fixtures import mark_complete

    root = tmp_path / "ds"
    image = root / "images" / DATE / "plotA_0_0.jpg"
    write_image(image, (96, 64))
    _label(root, "plotA_0_0", [], keep_empty=True)
    mark_complete(image, "bud", project=root)

    assert _check(root) == []


def test_an_unreadable_label_is_a_finding_beside_a_readable_one(tmp_path: Path):
    """Coverage, not a guard: the refusal surfaces as a per-document finding beside a readable
    document of the same capture, never propagating out of the walk."""
    root = tmp_path / "ds"
    write_image(root / "images" / DATE / "plotA_0_0.jpg", (96, 64))
    write_image(root / "images" / DATE / "plotB_0_0.jpg", (96, 64))

    _label(root, "plotA_0_0", [Annotation(subject="bud", geometry=BBox(11, 7, 39, 51))])
    ts.replace(label_key(root, DATE, "plotB_0_0"), ["not", "a", "document"])

    findings = _check(root)

    errors = _errors(findings)
    assert len(errors) == 1
    assert "plotB_0_0" in errors[0] and "will not read" in errors[0]
    assert [level for level, _ in findings if level == "warn"] == []
