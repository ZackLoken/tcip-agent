"""The doctor's check_data_quality: the findings it reports and the level of each.

A label document the per-image reader refuses is a named finding for that document, and the
check goes on to the remaining documents. A warning and an error are distinct levels.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import tcip_store as ts
from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.dataset_layout import image_dir, label_key
from tcip_mcp.tools import doctor_tools as doctor
from tests._producer_fixtures import write_image

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from tcip_annotation.json_io import CompletionMark
    from tcip_store import Version

DATE = "2-11-26"


def _check(root: Path) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    census = doctor._census(root, findings)
    assert census is not None, findings
    doctor.check_data_quality(root, findings, census=census)
    return findings


def _errors(findings: list[tuple[str, str]]) -> list[str]:
    return [msg for level, msg in findings if level == "error"]


def _label(
    root: Path,
    stem: str,
    annotations: Iterable[Annotation],
    width: int = 96,
    height: int = 64,
    **kwargs: bool | Version | Mapping[str, list[CompletionMark]] | None,
) -> None:
    json_io.write_label_document(
        label_key(root, DATE, stem), annotations, width, height, **kwargs
    )


def test_an_unreadable_label_record_is_an_error_per_document(
    tmp_path: Path,
) -> None:
    """An unreadable label record is an error per document, naming the refusal.

    Two label records of a shape the reader refuses are two errors; a dataset with no label
    documents yields one info-level finding and no error.
    """
    root = tmp_path / "unreadable"
    for stem in ("plotA_0_0", "plotA_0_1"):
        write_image(image_dir(root, DATE) / f"{stem}.jpg", (96, 64))
        ts.replace(
            label_key(root, DATE, stem),
            {"shapes": [{"label": "bud", "points": [[3, 5], [40, 52]]}]},
        )

    findings = _check(root)
    assert len(findings) == 2
    assert all(
        level == "error" and "label document will not read" in msg
        for level, msg in findings
    )

    bare = tmp_path / "unlabeled"
    for stem in ("plotA_0_0", "plotA_0_1"):
        write_image(image_dir(bare, DATE) / f"{stem}.jpg", (96, 64))

    findings = _check(bare)
    assert _errors(findings) == []
    assert [level for level, _ in findings] == ["info"]


def test_a_label_with_no_matching_image_is_an_error(tmp_path: Path) -> None:
    """An orphan label is an error-level finding."""
    root = tmp_path / "ds"
    for stem in ("plotA_0_0", "plotB_0_0"):
        write_image(image_dir(root, DATE) / f"{stem}.jpg", (96, 64))
    for stem in ("plotA_0_0", "plotB_0_0", "plotZ_9_9"):
        _label(
            root,
            stem,
            [Annotation(subject="bud", geometry=BBox(11, 7, 39, 51))],
        )

    errors = _errors(_check(root))

    assert len(errors) == 1
    assert "plotZ_9_9" in errors[0] and "no matching image" in errors[0]


def test_an_unreadable_first_label_hides_no_later_finding(
    tmp_path: Path,
) -> None:
    """An unreadable first label hides no later finding.

    An unreadable label sorting first and the orphan after it are each reported.
    """
    root = tmp_path / "ds"
    write_image(image_dir(root, DATE) / "0_bad.jpg", (96, 64))
    ts.replace(label_key(root, DATE, "0_bad"), ["not", "a", "document"])
    _label(
        root,
        "zzz_orphan",
        [Annotation(subject="bud", geometry=BBox(1, 1, 8, 8))],
    )

    errors = _errors(_check(root))

    assert any("0_bad" in e and "will not read" in e for e in errors)
    assert any("zzz_orphan" in e and "no matching image" in e for e in errors)


def test_an_empty_label_not_marked_complete_is_an_error(
    tmp_path: Path,
) -> None:
    """An empty label no person marked complete is the one error, naming it not marked complete."""
    root = tmp_path / "ds"
    write_image(image_dir(root, DATE) / "plotA_0_0.jpg", (96, 64))
    _label(root, "plotA_0_0", [], keep_empty=True)

    errors = _errors(_check(root))

    assert len(errors) == 1
    assert "plotA_0_0" in errors[0] and "not marked complete" in errors[0]


def test_a_confirmed_negative_empty_label_stays_clean(tmp_path: Path) -> None:
    """An empty label a person marked complete stays clean, a confirmed negative.

    A human's Complete-with-nothing yields no finding at all.
    """
    from tests._producer_fixtures import mark_complete

    root = tmp_path / "ds"
    image = image_dir(root, DATE) / "plotA_0_0.jpg"
    write_image(image, (96, 64))
    _label(root, "plotA_0_0", [], keep_empty=True)
    mark_complete(image, "bud", project=root)

    assert _check(root) == []


def test_an_unreadable_label_is_a_finding_beside_a_readable_one(
    tmp_path: Path,
) -> None:
    """An unreadable label is one finding beside a readable label of the same capture.

    The refusal surfaces as one error-level finding and no warning.
    """
    root = tmp_path / "ds"
    write_image(image_dir(root, DATE) / "plotA_0_0.jpg", (96, 64))
    write_image(image_dir(root, DATE) / "plotB_0_0.jpg", (96, 64))

    _label(
        root,
        "plotA_0_0",
        [Annotation(subject="bud", geometry=BBox(11, 7, 39, 51))],
    )
    ts.replace(label_key(root, DATE, "plotB_0_0"), ["not", "a", "document"])

    findings = _check(root)

    errors = _errors(findings)
    assert len(errors) == 1
    assert "plotB_0_0" in errors[0] and "will not read" in errors[0]
    assert [level for level, _ in findings if level == "warn"] == []
