"""Tests for data management tools."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import tcip_store as ts
from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.dataset_layout import UNDATED_BUCKET, label_key
from tcip_mcp.pipelines.data.label_queries import registry_scope
from tcip_mcp.pipelines.data.selection import (
    ClassScope,
    Sample,
    Selection,
    read_selection,
    selection_key,
    write_selection,
)
from tcip_mcp.tools import doctor_tools as doctor
from tcip_mcp.tools.data_tools import draw_splits, scan_dataset
from tests._audit_fixtures import audit_rows
from tests._producer_fixtures import label_image, registry_over
from tests._training_values import RATIONALE, drawn_split

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from pydantic import JsonValue

    from tcip_mcp.pipelines.data.split_construction import Membership

SHARES = {"val_ratio": 0.15, "calibration_ratio": 0.1, "holdout_ratio": 0.05}
"""The sample shares a draw here states unless it states its own."""


def _draw(
    project: Path,
    folder_path: str,
    **stated: str | float | bool | dict[str, str] | None,
) -> dict:
    """Run ``draw_splits`` over ``folder_path`` at :data:`SHARES` and return its result.

    The draw is grouped by tile prefix and stratified, with ``stated`` over those.
    """
    return draw_splits(
        project,
        folder_path,
        **drawn_split(**{**SHARES, "grouping": "tile_prefix", **stated}),
        actor=None,
    )


def _quality_findings(root: Path) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    doctor.check_data_quality(
        Path(root), findings, census=doctor._census(Path(root), findings)
    )
    return findings


def test_scan_dataset(data_dir: Path) -> None:
    result = scan_dataset(data_dir, str(data_dir))
    assert result["image_count"] == 3
    assert result["labels_count"] == 3
    assert result["paired_images"] == 3
    assert result["unlabeled_images"] == 0


def test_scan_dataset_not_found(tmp_path: Path) -> None:
    result = scan_dataset(tmp_path, "/nonexistent/path")
    assert "error" in result


def test_doctor_check_data_quality(data_dir: Path) -> None:
    findings = _quality_findings(data_dir)
    assert not [f for f in findings if f[0] == "error"]


def test_doctor_check_data_quality_missing_dir(tmp_path: Path) -> None:
    # _scan_dataset's own rglob over a nonexistent path yields nothing rather than raising.
    missing = tmp_path / "nonexistent" / "xyz123"
    findings = _quality_findings(missing)
    assert not [f for f in findings if f[0] == "error"]


def test_scan_dataset_counts_only_the_documents_a_published_bucket_names(
    tmp_path: Path,
) -> None:
    """``scan_dataset`` counts only the prediction documents a published bucket names.

    Beside one published document, a prediction document no bucket record names leaves the count
    at one.
    """
    pytest.importorskip("torch")
    from tcip_mcp.dataset_layout import prediction_key
    from tests._chain_fixtures import predicted, published

    root = tmp_path / "ds"
    published(
        tmp_path,
        "modelA/2-11-26",
        [predicted(root / "images" / "2-11-26" / "imgA.png", ["bud"])],
        scope={"subject": "bud"},
    )
    json_io.write_label_document(
        prediction_key(root, "modelB/2-11-26", "imgB"),
        [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5), score=0.9)],
        32,
        32,
    )

    scan_result = scan_dataset(tmp_path, str(root))

    assert scan_result["predictions_count"] == 1


def _add_extra_bud_groups(data_dir: Path, count: int) -> None:
    """Add ``count`` more single-tile foreground groups under ``data_dir``'s own date.

    The groups carry the fixture's own subject.
    """
    from PIL import Image

    images_dir = data_dir / "images" / "2-11-26"
    for i in range(count):
        stem = f"extra_{i:03d}"
        Image.new("RGB", (640, 480), color=(128, 128, 128)).save(
            images_dir / f"{stem}.jpg"
        )
        label_image(
            images_dir / f"{stem}.jpg",
            [Annotation(subject="bud", geometry=BBox(288, 216, 352, 264))],
            640,
            480,
        )


def test_draw_splits_basic(data_dir: Path, tmp_path: Path) -> None:
    _add_extra_bud_groups(data_dir, 1)
    out = tmp_path / "manifests"
    # The fixture's 4 stems (img_001..003 plus one grown group) are 4 distinct foreground
    # groups, exactly the manifest floor (one each for train/val, two for calibration).
    result = _draw(
        data_dir,
        str(data_dir),
        output_path=str(out),
        subject="bud",
        seed=1,
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
    )
    assert "error" not in result, result
    assert result["total_stems"] == 4
    assert result["groups"] == 4
    assert sum(result["splits"].values()) == 4
    assert result["stratify_foreground"] is True

    drawn = read_selection(out, project=data_dir)
    assert drawn.counts() == {k: v for k, v in result["splits"].items()}
    assert all(
        drawn.counts()[side] for side in ("train", "val", "calibration")
    )
    assert drawn.scope.subject == "bud"
    assert drawn.scope.attributes == ()
    assert drawn.dataset_fingerprint is not None
    for sample in drawn.samples:
        assert Path(sample.source).is_file()
        assert ts.exists(sample.ground_truth)
        assert sample.ground_truth_digest


LEAF_DATE = "2-11-26"


def _leaf_scene(root: Path, n: int = 6) -> Path:
    """Write ``n`` captures of one date under ``root`` and return ``root``.

    The ``i``-th label holds ``i + 1`` leaves and ``5 * (n - i)`` buds.
    """
    from PIL import Image

    images = root / "images" / LEAF_DATE
    images.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images / f"s{i}.jpg")
        label_image(
            images / f"s{i}.jpg",
            [
                *(
                    Annotation(subject="leaf", geometry=BBox(2, 2, 6, 6))
                    for _ in range(i + 1)
                ),
                *(
                    Annotation(subject="bud", geometry=BBox(8, 8, 12, 12))
                    for _ in range(5 * (n - i))
                ),
            ],
            100,
            80,
        )
    return root


def test_a_draw_without_stratification_reports_the_counted_annotations(
    tmp_path: Path,
) -> None:
    """A draw without stratification reports the subject's annotations it counted.

    With balancing off the draw reports the subject's own annotation total, and the per-side
    foreground sums to that total.
    """
    root = _leaf_scene(tmp_path / "ds")
    result = _draw(
        tmp_path, str(root), seed=1, subject="leaf", stratify_foreground=False
    )
    assert "error" not in result, result
    assert result["stratify_foreground"] is False
    assert result["total_annotations"] == sum(range(1, 7))
    assert (
        sum(result["foreground_annotations"].values())
        == result["total_annotations"]
    )


def test_draw_splits_and_a_runs_own_draw_side_the_same_members(
    tmp_path: Path,
) -> None:
    """``draw_splits`` and a run's own draw hold the same per-side sizes and foreground.

    The draws share ground truth, seed, policy and val share, and the run's foreground is
    recounted here from each member's own leaves.
    """
    pytest.importorskip("torch")
    from tests._producer_fixtures import train_val

    root = _leaf_scene(tmp_path / "ds")
    drawn = _draw(
        tmp_path,
        str(root),
        subject="leaf",
        val_ratio=0.5,
        calibration_ratio=0,
        holdout_ratio=0,
        grouping="stem",
        seed=7,
    )
    assert "error" not in drawn, drawn
    data_cfg = {
        "images_dir": str(root / "images" / LEAF_DATE),
        "scope": {"subject": "leaf"},
        "split": drawn_split(val_ratio=0.5, seed=7),
    }
    _train, _val, partition, _data = train_val(tmp_path, "detection", data_cfg)

    by_side: dict[str, list[int]] = {"train": [], "val": []}
    for sample in partition.samples:
        by_side[sample.side].append(int(sample.member.removeprefix("s")) + 1)
    assert {side: len(v) for side, v in by_side.items()} == {
        side: drawn["splits"][side] for side in by_side
    }
    assert {side: sum(v) for side, v in by_side.items()} == {
        side: drawn["foreground_annotations"][side] for side in by_side
    }


def test_every_draw_refuses_a_tree_short_of_its_floor_the_same_way(
    tmp_path: Path,
) -> None:
    """``draw_splits`` and a run's own draw both refuse a tree short of the floor in its words.

    Two captures mapped into one group are one foreground group, short of a train and a val
    side, and both draws refuse with that floor's words.
    """
    pytest.importorskip("torch")
    from tests._producer_fixtures import train_val

    root = _leaf_scene(tmp_path / "ds", n=2)
    one_group = {f"{LEAF_DATE}/s{i}": "plot" for i in range(2)}
    floor = "1 foreground group(s), fewer than the 2 the requested sides need"

    drawn = _draw(
        tmp_path,
        str(root),
        subject="leaf",
        seed=1,
        val_ratio=0.5,
        calibration_ratio=0,
        holdout_ratio=0,
        grouping=one_group,
    )
    assert floor in drawn["error"]
    data_cfg = {
        "images_dir": str(root / "images" / LEAF_DATE),
        "scope": {"subject": "leaf"},
        "split": drawn_split(val_ratio=0.5, seed=1, grouping=one_group),
    }
    with pytest.raises(
        ValueError, match=floor.replace("(", r"\(").replace(")", r"\)")
    ):
        train_val(tmp_path, "detection", data_cfg)


def test_draw_splits_stats_only_admits_a_nonzero_calibration_ratio(
    data_dir: Path,
) -> None:
    """A stats-only draw admits a nonzero calibration ratio beside a zero holdout.

    A stats-only call (no output_path) draws a calibration side beside a holdout stated at zero,
    which it does not draw.
    """
    result = _draw(
        data_dir,
        str(data_dir),
        seed=1,
        val_ratio=0.2,
        calibration_ratio=0.1,
        holdout_ratio=0,
        subject="bud",
    )
    assert "error" not in result, result
    assert result["splits"]["calibration"] > 0
    assert result["splits"]["holdout"] == 0


def _spoil(data_dir: Path, stem: str) -> None:
    """Replace the label document of the fixture's image ``stem`` with a non-document record.

    The replacement goes past the label writer.
    """
    ts.replace(label_key(data_dir, "2-11-26", stem), ["not", "a", "document"])


def test_draw_splits_reports_an_unreadable_label_by_name(
    data_dir: Path, tmp_path: Path
) -> None:
    """``draw_splits`` reports an unreadable label among its candidates by name.

    A present, unreadable label among the candidates is an error naming the document.
    """
    _spoil(data_dir, "img_001")

    result = _draw(
        data_dir,
        str(data_dir),
        output_path=str(tmp_path / "manifests"),
        subject="bud",
        seed=1,
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
    )

    assert "error" in result
    assert "'img_001'" in result["error"]


def test_draw_splits_reports_an_unreadable_label_sorted_last(
    data_dir: Path,
    tmp_path: Path,
) -> None:
    """``draw_splits`` reports an unreadable label sorted last by name.

    A corrupt label last in sort order answers an error dict naming it.
    """
    _spoil(data_dir, "img_003")

    result = _draw(
        data_dir,
        str(data_dir),
        output_path=str(tmp_path / "manifests"),
        subject="bud",
        seed=1,
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
    )

    assert "error" in result
    assert "'img_003'" in result["error"]


def test_draw_splits_writes_nothing_when_a_marked_document_will_not_read(
    data_dir: Path,
    tmp_path: Path,
) -> None:
    """``draw_splits`` writes no selection when a document marked complete will not read.

    An unreadable label on an image a human marked complete refuses the draw naming it, and no
    selection is written.
    """
    from tcip_mcp.dataset_layout import image_dir
    from tests._producer_fixtures import mark_complete

    mark_complete(
        image_dir(data_dir, "2-11-26") / "img_002.jpg", "bud", project=data_dir
    )
    _spoil(data_dir, "img_002")
    out = tmp_path / "selection"

    result = _draw(
        data_dir,
        str(data_dir),
        output_path=str(out),
        subject="bud",
        seed=1,
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
    )

    assert "error" in result
    assert "'img_002'" in result["error"]
    assert not ts.exists(selection_key(out))


def test_draw_splits_stats_only_reports_an_unreadable_first_sorted_label(
    data_dir: Path,
) -> None:
    """A stats-only draw reports an unreadable first-sorted label by name.

    In a stats-only call (no output_path), an unreadable first-sorted candidate is an error
    naming it.
    """
    _spoil(data_dir, "img_001")

    result = _draw(data_dir, str(data_dir), seed=1, subject="bud")

    assert "error" in result
    assert "'img_001'" in result["error"]


def test_draw_splits_stats_only_reports_an_unreadable_label_during_stratification(
    data_dir: Path,
) -> None:
    """A stats-only draw reports a corrupt label reached after a readable one by name.

    In a stats-only call, a corrupt label sorted after a readable first candidate is an error
    naming the document.
    """
    _spoil(data_dir, "img_003")

    result = _draw(data_dir, str(data_dir), seed=1, subject="bud")

    assert "error" in result
    assert "'img_003'" in result["error"]


def test_draw_splits_manifest_answers_an_ambiguous_image_stem_as_an_error(
    tmp_path: Path,
) -> None:
    """A selection-writing draw answers an ambiguous image stem as an error naming the stem.

    The ambiguity is a raw file colliding with a band group's own canonical stem.
    """
    import numpy as np

    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)

    band_a, band_b = images_dir / "plotA_B1.npy", images_dir / "plotA_B2.npy"
    np.save(band_a, np.zeros((4, 4), dtype=np.uint8))
    np.save(band_b, np.zeros((4, 4), dtype=np.uint8))
    write_band_group_manifest(
        images_dir,
        "plotA",
        {"B1": band_a, "B2": band_b},
        central_wavelength_nm=None,
        identity_tolerance={},
    )
    (images_dir / "plotA.jpg").write_bytes(b"\xff\xd8\xff")
    label_image(
        images_dir / "plotA.jpg",
        [Annotation(subject="leaf", geometry=BBox(1, 1, 3, 3))],
        4,
        4,
    )

    result = _draw(
        tmp_path,
        str(root),
        output_path=str(tmp_path / "manifests"),
        subject="leaf",
        seed=1,
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
    )

    assert "error" in result
    assert "plotA" in result["error"]


def test_draw_splits_stats_only_answers_an_ambiguous_image_stem_as_an_error(
    tmp_path: Path,
) -> None:
    """A stats-only draw answers an ambiguous image stem as an error naming the stem."""
    import numpy as np

    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)

    band_a, band_b = images_dir / "plotA_B1.npy", images_dir / "plotA_B2.npy"
    np.save(band_a, np.zeros((4, 4), dtype=np.uint8))
    np.save(band_b, np.zeros((4, 4), dtype=np.uint8))
    write_band_group_manifest(
        images_dir,
        "plotA",
        {"B1": band_a, "B2": band_b},
        central_wavelength_nm=None,
        identity_tolerance={},
    )
    (images_dir / "plotA.jpg").write_bytes(b"\xff\xd8\xff")
    label_image(
        images_dir / "plotA.jpg",
        [Annotation(subject="leaf", geometry=BBox(1, 1, 3, 3))],
        4,
        4,
    )

    result = _draw(tmp_path, str(root), seed=1, subject="leaf")

    assert "error" in result
    assert "plotA" in result["error"]


def test_scan_dataset_answers_an_ambiguous_image_stem_as_an_error(
    tmp_path: Path,
) -> None:
    """``scan_dataset`` answers an ambiguous image stem as an error naming the stem."""
    import numpy as np

    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)

    band_a, band_b = images_dir / "plotA_B1.npy", images_dir / "plotA_B2.npy"
    np.save(band_a, np.zeros((4, 4), dtype=np.uint8))
    np.save(band_b, np.zeros((4, 4), dtype=np.uint8))
    write_band_group_manifest(
        images_dir,
        "plotA",
        {"B1": band_a, "B2": band_b},
        central_wavelength_nm=None,
        identity_tolerance={},
    )
    (images_dir / "plotA.jpg").write_bytes(b"\xff\xd8\xff")

    result = scan_dataset(tmp_path, str(root))

    assert "error" in result
    assert "plotA" in result["error"]


def _add_extra_leaf_groups(images_dir: Path, count: int) -> None:
    """Add ``count`` more plain single-tile foreground groups of subject ``leaf`` to a directory."""
    from PIL import Image

    letters = "BCDEFGH"
    for i in range(count):
        stem = f"plot{letters[i]}"
        Image.new("RGB", (100, 80), (128, 128, 128)).save(
            images_dir / f"{stem}.jpg"
        )
        label_image(
            images_dir / f"{stem}.jpg",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))],
            100,
            80,
        )


def test_draw_splits_refuses_an_incomplete_band_group_before_writing(
    tmp_path: Path,
) -> None:
    """A band group whose manifest names a missing sibling refuses the draw, writing no selection.

    The refusal names the group and its missing band.
    """
    import numpy as np

    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)

    band_g, band_r = images_dir / "plotA_G.npy", images_dir / "plotA_R.npy"
    np.save(band_g, np.zeros((4, 4), dtype=np.uint8))
    write_band_group_manifest(
        images_dir,
        "plotA",
        {"G": band_g, "R": band_r},
        central_wavelength_nm=None,
        identity_tolerance={},
    )  # R never created
    label_image(
        images_dir / "plotA.bandgroup",
        [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))],
        100,
        80,
    )
    _add_extra_leaf_groups(images_dir, 3)
    out = tmp_path / "m"

    result = _draw(
        tmp_path,
        str(root),
        output_path=str(out),
        subject="leaf",
        seed=1,
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
    )

    assert "error" in result
    assert "plotA" in result["error"] and "R" in result["error"]
    assert not ts.exists(selection_key(out))


def test_draw_splits_bad_ratios(data_dir: Path) -> None:
    result = _draw(
        data_dir,
        str(data_dir),
        seed=1,
        val_ratio=0.5,
        calibration_ratio=0.25,
        holdout_ratio=0.25,
    )
    assert "error" in result


def test_shares_leaving_train_nothing_refuse_naming_every_share(
    tmp_path: Path,
) -> None:
    """Shares that leave ``train`` no remainder refuse naming every share they were given."""
    root = _multi_source_dataset(tmp_path / "ds")
    result = _draw(
        tmp_path,
        str(root),
        seed=1,
        subject="bud",
        val_ratio=0.5,
        calibration_ratio=0.3,
        holdout_ratio=0.3,
    )
    assert "the shares must leave it some" in result["error"]
    assert all(
        side in result["error"]
        for side in ("train", "val", "calibration", "holdout")
    )


def test_a_dry_draw_leaves_no_audit_line_and_a_written_selection_leaves_one(
    tmp_path: Path,
) -> None:
    root = _multi_source_dataset(tmp_path / "ds")

    assert "error" not in _draw(tmp_path, str(root), seed=1, subject="bud")
    assert audit_rows(tmp_path, "draw_splits") == []
    out = tmp_path / "m"
    assert "error" not in _draw(
        tmp_path, str(root), seed=1, subject="bud", output_path=str(out)
    )
    assert [
        row["arguments"]["output_path"]
        for row in audit_rows(tmp_path, "draw_splits")
    ] == [str(out)]


def test_draw_splits_selection_carries_all_four_sides(tmp_path: Path) -> None:
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"

    result = _draw(
        tmp_path,
        str(root),
        output_path=str(out),
        seed=1,
        subject="bud",
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
    )

    assert "error" not in result, result
    counts = read_selection(out, project=tmp_path).counts()
    assert set(counts) == {"train", "val", "calibration", "holdout"}
    assert all(counts.values())


@pytest.mark.parametrize(
    "unstated",
    [
        "val_ratio",
        "calibration_ratio",
        "holdout_ratio",
        "stratify_foreground",
        "rationale",
        "grouping",
    ],
)
def test_a_draw_stating_no_share_grouping_stratification_or_reason_refuses_naming_it(
    tmp_path: Path, unstated: str
) -> None:
    """A draw stating no share, grouping, stratification or reason is refused naming it.

    A draw leaving a share, the grouping, the stratification or its reason unstated refuses
    naming it and writes no selection, and the same draw stating each writes a selection, and an
    audit line, recording the shares, the stratification and the reason.
    """
    root = _multi_source_dataset(tmp_path / "ds")
    stated = drawn_split(
        seed=1,
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
        grouping="tile_prefix",
    )
    out = tmp_path / "unstated"
    try:
        refusal = draw_splits(
            tmp_path,
            str(root),
            output_path=str(out),
            subject="bud",
            **{k: v for k, v in stated.items() if k != unstated},
            actor=None,
        )["error"]
    except TypeError as exc:
        refusal = str(exc)
    assert unstated in refusal
    assert not ts.exists(selection_key(out))

    out = tmp_path / "stated"
    result = draw_splits(
        tmp_path,
        str(root),
        output_path=str(out),
        subject="bud",
        **stated,
        actor=None,
    )
    assert "error" not in result, result
    drawn = read_selection(out, project=tmp_path)
    shares = {
        "train": 0.5,
        "val": 0.25,
        "calibration": 0.125,
        "holdout": 0.125,
    }
    assert (drawn.ratios, drawn.stratify_foreground, drawn.rationale) == (
        shares,
        True,
        RATIONALE,
    )
    (line,) = [row["arguments"] for row in audit_rows(tmp_path, "draw_splits")]
    assert isinstance(line, dict)
    assert (
        line["ratios"],
        line["stratify_foreground"],
        line["rationale"],
    ) == (shares, True, RATIONALE)


def test_draw_splits_floor_refuses_before_any_write_regardless_of_stratify_foreground(
    tmp_path: Path,
) -> None:
    """The foreground floor refuses a draw with ``stratify_foreground`` off, writing no selection.

    A tree with three foreground groups refuses naming the floor.
    """
    root = _multi_source_dataset(
        tmp_path / "ds", prefixes=("srcA", "srcB", "srcC")
    )
    out = tmp_path / "m"

    result = _draw(
        tmp_path,
        str(root),
        output_path=str(out),
        subject="bud",
        seed=1,
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
        stratify_foreground=False,
    )

    assert "error" in result
    assert "foreground group" in result["error"]
    assert not ts.exists(selection_key(out))


def test_draw_splits_answers_one_draw_whether_or_not_it_writes(
    tmp_path: Path,
) -> None:
    """``draw_splits`` answers one draw whether or not it writes.

    A zero ratio draws no side of that name and a negative one refuses, with or without
    ``output_path``: what is drawn does not depend on whether it is written.
    """
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    ratios = {
        "val_ratio": 0.0,
        "calibration_ratio": 0.125,
        "holdout_ratio": 0.125,
    }

    stats = _draw(tmp_path, str(root), subject="bud", seed=1, **ratios)
    written = _draw(
        tmp_path,
        str(root),
        output_path=str(out),
        subject="bud",
        seed=1,
        **ratios,
    )

    assert "error" not in written, written
    assert {
        key: value for key, value in written.items() if key != "selection_dir"
    } == {key: value for key, value in stats.items() if key != "selection_dir"}
    assert written["splits"]["val"] == 0
    for output_path in (None, str(tmp_path / "negative")):
        refused = _draw(
            tmp_path,
            str(root),
            output_path=output_path,
            subject="bud",
            seed=1,
            val_ratio=-0.25,
            calibration_ratio=0.25,
            holdout_ratio=0.125,
        )
        assert "share in [0, 1)" in refused["error"]
    assert not ts.exists(selection_key(tmp_path / "negative"))


def test_draw_splits_floor_ignores_a_groups_only_annotations_of_another_subject(
    tmp_path: Path,
) -> None:
    """The draw's floor ignores a group whose only annotations are of another subject.

    Three ``leaf`` groups plus one group holding only a ``bud`` annotation, marked complete for
    ``leaf``, refuse below the floor of four, and no selection is written.
    """
    from PIL import Image

    from tcip_mcp.subject_registry import Subject, SubjectRegistry
    from tests._producer_fixtures import mark_complete

    root = tmp_path / "ds"
    date = "2-11-26"
    images_dir = root / "images" / date
    images_dir.mkdir(parents=True)
    registry_over(
        root,
        SubjectRegistry(
            subjects=(
                Subject(name="leaf", description=""),
                Subject(name="bud", description=""),
            )
        ),
    )
    for stem in ("p1", "p2", "p3"):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(
            images_dir / f"{stem}.jpg"
        )
        label_image(
            images_dir / f"{stem}.jpg",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))],
            100,
            80,
        )
    Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / "p4.jpg")
    label_image(
        images_dir / "p4.jpg",
        [Annotation(subject="bud", geometry=BBox(4, 4, 12, 12))],
        100,
        80,
        keep_empty=True,
    )
    mark_complete(images_dir / "p4.jpg", "leaf", project=root)

    out = tmp_path / "m"
    result = _draw(
        tmp_path,
        str(root),
        output_path=str(out),
        subject="leaf",
        seed=1,
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
    )

    assert "error" in result
    assert "foreground group" in result["error"]
    assert not ts.exists(selection_key(out))


def _leaf_dataset_with_negatives(
    root: Path, date: str, n_foreground: int, n_negative: int
) -> None:
    """Write ``n_foreground`` single-tile ``leaf`` groups and ``n_negative`` confirmed negatives.

    All of them lie under one capture date.
    """
    from PIL import Image

    from tests._producer_fixtures import mark_complete

    images_dir = root / "images" / date
    images_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_foreground):
        stem = f"{date}_fg{i}"
        Image.new("RGB", (100, 80), (128, 128, 128)).save(
            images_dir / f"{stem}.jpg"
        )
        label_image(
            images_dir / f"{stem}.jpg",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))],
            100,
            80,
        )
    for i in range(n_negative):
        stem = f"{date}_bg{i}"
        Image.new("RGB", (100, 80), (128, 128, 128)).save(
            images_dir / f"{stem}.jpg"
        )
        label_image(images_dir / f"{stem}.jpg", [], 100, 80, keep_empty=True)
        mark_complete(images_dir / f"{stem}.jpg", "leaf", project=root)


def test_draw_splits_calibration_side_holds_real_foreground_regardless_of_stratify_foreground(
    tmp_path: Path,
) -> None:
    """The calibration side holds real foreground with ``stratify_foreground`` off.

    Over four foreground groups and six confirmed negatives, for each of seeds 1 to 20 the
    calibration side holds at least two foreground groups.
    """
    root = tmp_path / "ds"
    date = "2-11-26"
    _leaf_dataset_with_negatives(root, date, n_foreground=4, n_negative=6)

    for seed in range(1, 21):
        out = tmp_path / f"m{seed}"
        result = _draw(
            tmp_path,
            str(root),
            output_path=str(out),
            subject="leaf",
            seed=seed,
            val_ratio=0.1,
            calibration_ratio=0.05,
            holdout_ratio=0.05,
            stratify_foreground=False,
        )
        assert "error" not in result, (seed, result)
        assert result["calibration_foreground_groups"] >= 2, (seed, result)


def _multi_source_dataset(
    root: Path,
    prefixes: tuple[str, ...] = ("srcA", "srcB", "srcC", "srcD"),
    tiles: int = 3,
) -> Path:
    from PIL import Image

    date = "2-11-26"
    images_dir = root / "images" / date
    images_dir.mkdir(parents=True)
    for pref in prefixes:
        for t in range(tiles):
            stem = f"{pref}_{t}_0"
            Image.new("RGB", (64, 64), (128, 128, 128)).save(
                images_dir / f"{stem}.jpg"
            )
            label_image(
                images_dir / f"{stem}.jpg",
                [Annotation(subject="bud", geometry=BBox(19, 13, 45, 51))],
                64,
                64,
            )
    return root


def _bud_membership(root: Path) -> Membership:
    from tcip_mcp.pipelines.data.split_construction import admitted_membership

    date = "2-11-26"
    return admitted_membership(
        [(date, root / "images" / date, None)],
        scope=ClassScope(subject="bud"),
        grouping="tile_prefix",
    )


def _spied_draws(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Spy on the split algorithm and return each ``weighted`` the draw hands it, in call order."""
    import tcip_mcp.pipelines.data.splits as splits

    calls: list[bool] = []
    real = splits.group_balanced_split

    def spy(
        stems: Sequence[str],
        **kwargs: Mapping[str, float] | bool | int | Callable[[str], str],
    ) -> dict[str, list[str]]:
        calls.append(kwargs["weighted"])
        return real(stems, **kwargs)

    monkeypatch.setattr(splits, "group_balanced_split", spy)
    return calls


@pytest.mark.parametrize(
    "sides",
    [
        {"train": 0.5},
        {"val": 1.0},
        {"val": -0.25},
        {"val": 0.5, "holdout": 0.5},
        {"test": 0.1},
    ],
    ids=["train_stated", "one", "negative", "no_remainder", "unknown_side"],
)
def test_check_shares_refuses_shares_that_are_not_a_draw(
    sides: dict[str, float],
) -> None:
    from tcip_mcp.pipelines.data.split_construction import check_shares

    with pytest.raises(ValueError, match=r"share in \[0, 1\)"):
        check_shares(sides)


def test_check_shares_gives_train_the_remainder_and_draws_no_zero_side() -> (
    None
):
    from tcip_mcp.pipelines.data.split_construction import check_shares

    assert check_shares(
        {"val": 0.25, "calibration": 0.0, "holdout": 0.125}
    ) == {"train": 0.625, "val": 0.25, "holdout": 0.125}


@pytest.mark.parametrize("stratify", [True, False])
def test_draw_sides_cuts_the_reference_under_the_balancing_it_was_given(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stratify: bool,
) -> None:
    from tcip_mcp.pipelines.data.split_construction import draw_sides

    membership = _bud_membership(_multi_source_dataset(tmp_path / "ds"))
    draws = _spied_draws(monkeypatch)
    drawn, _counted = draw_sides(
        membership.samples,
        membership.scope,
        seed=1,
        stratify=stratify,
        ratios={
            "train": 0.5,
            "val": 0.25,
            "calibration": 0.125,
            "holdout": 0.125,
        },
    )
    assert draws == [stratify, stratify]
    assert {sample.side for sample in drawn.values()} == {
        "train",
        "val",
        "calibration",
        "holdout",
    }


def test_draw_splits_groups_tiles_together(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.splits import default_group_key

    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = _draw(
        tmp_path,
        str(root),
        output_path=str(out),
        seed=1,
        subject="bud",
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
    )
    assert result["groups"] == 4  # 4 source prefixes, not 12 tiles

    # No source prefix may appear in more than one split.
    seen: dict[str, str] = {}
    for sample in read_selection(out, project=tmp_path).samples:
        g = default_group_key(sample.member)
        assert seen.get(g, sample.side) == sample.side, (
            f"group {g} spans splits"
        )
        seen[g] = sample.side


def test_draw_splits_grouping_map_never_straddles(tmp_path: Path) -> None:
    """A draw's grouping map is honored, its two same-group samples landing on one side.

    In an agent-derived grouping map of 5 samples and 4 groups, the two samples of group ``gA``
    land on one side under that group, and the draw records the explicit map.
    """
    root = _multi_source_dataset(
        tmp_path / "ds", prefixes=("x", "y", "z", "w", "v"), tiles=1
    )
    out = tmp_path / "m"
    group_key_map = {
        "2-11-26/x_0_0": "gA",
        "2-11-26/y_0_0": "gA",
        "2-11-26/z_0_0": "gB",
        "2-11-26/w_0_0": "gC",
        "2-11-26/v_0_0": "gD",
    }
    result = _draw(
        tmp_path,
        str(root),
        output_path=str(out),
        seed=1,
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
        grouping=group_key_map,
        subject="bud",
    )
    assert "error" not in result, result
    assert result["group_by"] == "explicit_map"

    drawn = read_selection(out, project=tmp_path)
    by_stem = {s.member: s for s in drawn.samples}
    assert by_stem["x_0_0"].side == by_stem["y_0_0"].side  # gA never straddles
    assert by_stem["x_0_0"].group == by_stem["y_0_0"].group == "gA"
    assert drawn.group_by == "explicit_map"


def test_draw_splits_unrecognized_grouping_policy_refuses_without_writing(
    tmp_path: Path,
) -> None:
    """An unrecognized grouping policy refuses the draw and writes no selection."""
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = _draw(
        tmp_path,
        str(root),
        output_path=str(out),
        grouping="not_a_real_key",
        subject="bud",
        seed=1,
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
    )
    assert "error" in result
    assert not ts.exists(selection_key(out))


def test_draw_splits_refuses_to_write_a_selection_with_no_subject(
    tmp_path: Path,
) -> None:
    """``draw_splits`` given no subject refuses naming ``subject`` and writes no selection."""
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = _draw(
        tmp_path,
        str(root),
        output_path=str(out),
        seed=1,
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
    )
    assert "error" in result and "subject" in result["error"]
    assert not ts.exists(selection_key(out))


def _two_date_collision_dataset(root: Path, subject: str) -> Path:
    """Write one stem name, ``shared``, under two capture dates with different label boxes.

    Each date also holds one more labeled stem, ``extra_<date>``. Returns ``root``.
    """
    from PIL import Image

    for date, box_x in (("2-11-26", 4), ("2-12-01", 40)):
        images_dir = root / "images" / date
        images_dir.mkdir(parents=True)
        Image.new("RGB", (100, 80), (128, 128, 128)).save(
            images_dir / "shared.jpg"
        )
        label_image(
            images_dir / "shared.jpg",
            [
                Annotation(
                    subject=subject, geometry=BBox(box_x, 4, box_x + 8, 12)
                )
            ],
            100,
            80,
        )
        extra_stem = f"extra_{date}"
        Image.new("RGB", (100, 80), (128, 128, 128)).save(
            images_dir / f"{extra_stem}.jpg"
        )
        label_image(
            images_dir / f"{extra_stem}.jpg",
            [Annotation(subject=subject, geometry=BBox(4, 4, 12, 12))],
            100,
            80,
        )
    return root


def test_two_dates_sharing_a_filename_stay_distinct_samples(
    tmp_path: Path,
) -> None:
    """Two dates sharing a filename stay two distinct samples.

    The two ``shared`` images are two samples, each naming its own source and label under its
    own capture date, and each label reads back with its own date's box.
    """
    root = _two_date_collision_dataset(tmp_path / "ds", subject="leaf")
    out = tmp_path / "m"
    result = _draw(
        tmp_path,
        str(root),
        output_path=str(out),
        subject="leaf",
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
        seed=1,
    )
    assert "error" not in result, result
    assert result["total_stems"] == 4

    drawn = read_selection(out, project=tmp_path)
    shared = [s for s in drawn.samples if s.member == "shared"]
    assert len(shared) == 2
    assert {Path(s.source).parent.name for s in shared} == {
        "2-11-26",
        "2-12-01",
    }
    assert {s.ground_truth.parts[0] for s in shared} == {"2-11-26", "2-12-01"}
    assert len({s.location for s in shared}) == 2
    # Each sample's own label document is the one under its own date, never the other's.
    boxes = {
        json_io.read_label_document(s.ground_truth).annotations[0].geometry.x1  # type: ignore[union-attr,arg-type]
        for s in shared
    }
    assert boxes == {4.0, 40.0}


def _two_subject_dataset(root: Path) -> Path:
    """Write six stems on one date, four carrying ``leaf`` and two the unrelated subject ``bud``.

    No stem carries both, and the registry declares the two subjects. Returns ``root``.
    """
    from PIL import Image

    from tcip_mcp.subject_registry import Subject, SubjectRegistry

    date = "2-11-26"
    images_dir = root / "images" / date
    images_dir.mkdir(parents=True)
    registry_over(
        root,
        SubjectRegistry(
            subjects=(
                Subject(name="leaf", description=""),
                Subject(name="bud", description=""),
            )
        ),
    )
    for stem, subject in (
        ("leaf_a", "leaf"),
        ("leaf_b", "leaf"),
        ("leaf_c", "leaf"),
        ("leaf_d", "leaf"),
        ("bud_a", "bud"),
        ("bud_b", "bud"),
    ):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(
            images_dir / f"{stem}.jpg"
        )
        label_image(
            images_dir / f"{stem}.jpg",
            [Annotation(subject=subject, geometry=BBox(4, 4, 12, 12))],
            100,
            80,
        )
    return root


def test_draw_splits_holds_only_the_named_subjects_admitted_samples(
    tmp_path: Path,
) -> None:
    root = _two_subject_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = _draw(
        tmp_path,
        str(root),
        output_path=str(out),
        subject="leaf",
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
        seed=1,
    )
    assert "error" not in result, result
    assert result["total_stems"] == 4
    drawn = read_selection(out, project=tmp_path)
    assert {s.member for s in drawn.samples} == {
        "leaf_a",
        "leaf_b",
        "leaf_c",
        "leaf_d",
    }


def _attribute_scoped_dataset(root: Path) -> Path:
    """Write five stems on one date of one subject declaring ``condition``.

    Four have their instance assessed for it, one carries an instance never assessed for it.
    Returns ``root``.
    """
    from PIL import Image

    from tcip_mcp.subject_registry import Attribute, Subject, SubjectRegistry

    date = "2-11-26"
    images_dir = root / "images" / date
    images_dir.mkdir(parents=True)
    registry_over(
        root,
        SubjectRegistry(
            subjects=(
                Subject(
                    name="leaf",
                    description="",
                    attributes=(
                        Attribute(
                            name="condition",
                            type="categorical",
                            values=("healthy", "damaged"),
                        ),
                    ),
                ),
            )
        ),
    )
    for stem, condition in (
        ("assessed_a", "healthy"),
        ("assessed_b", "damaged"),
        ("assessed_c", "healthy"),
        ("assessed_d", "damaged"),
    ):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(
            images_dir / f"{stem}.jpg"
        )
        label_image(
            images_dir / f"{stem}.jpg",
            [
                Annotation(
                    subject="leaf",
                    geometry=BBox(4, 4, 12, 12),
                    attributes={"condition": condition},
                )
            ],
            100,
            80,
        )
    Image.new("RGB", (100, 80), (128, 128, 128)).save(
        images_dir / "unassessed.jpg"
    )
    label_image(
        images_dir / "unassessed.jpg",
        [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))],
        100,
        80,
    )
    return root


def test_draw_splits_records_every_declared_attribute_and_keeps_an_unassessed_sample(
    tmp_path: Path,
) -> None:
    root = _attribute_scoped_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = _draw(
        tmp_path,
        str(root),
        output_path=str(out),
        subject="leaf",
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
        seed=1,
    )
    assert "error" not in result, result
    assert result["total_stems"] == 5
    drawn = read_selection(out, project=tmp_path)
    assert [a.name for a in drawn.scope.attributes] == ["condition"]
    assert {s.member for s in drawn.samples} == {
        "assessed_a",
        "assessed_b",
        "assessed_c",
        "assessed_d",
        "unassessed",
    }


def test_draw_splits_nothing_admitted_names_the_searched_images(
    tmp_path: Path,
) -> None:
    """A draw that admits nothing names the images directory it searched and writes no selection.

    A capture whose only label document names an image that is gone admits nothing.
    """
    root = tmp_path / "ds"
    (root / "images" / "2-11-26").mkdir(parents=True)
    json_io.write_label_document(
        label_key(root, "2-11-26", "a"),
        [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))],
        100,
        80,
    )
    out = tmp_path / "m"

    result = _draw(
        tmp_path,
        str(root),
        output_path=str(out),
        subject="leaf",
        seed=1,
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
    )

    assert "error" in result
    assert str(root / "images" / "2-11-26") in result["error"]
    assert not ts.exists(selection_key(out))


def test_draw_splits_draws_an_undated_capture_beside_a_dated_one(
    tmp_path: Path,
) -> None:
    """Both captures' label documents enter one draw, each sample naming its own."""
    from PIL import Image

    root = tmp_path / "ds"
    for capture, stems in (
        ("2-11-26", ("a", "b", "c")),
        (UNDATED_BUCKET, ("loose1", "loose2")),
    ):
        images = root / "images" / capture
        images.mkdir(parents=True)
        for stem in stems:
            Image.new("RGB", (100, 80), (128, 128, 128)).save(
                images / f"{stem}.jpg"
            )
            label_image(
                images / f"{stem}.jpg",
                [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))],
                100,
                80,
            )

    out = tmp_path / "m"
    result = _draw(
        tmp_path,
        str(root),
        output_path=str(out),
        subject="leaf",
        seed=1,
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
    )

    assert "error" not in result, result
    assert result["total_stems"] == 5
    assert result["tallies"] == {"partial": 5}
    drawn = read_selection(out, project=tmp_path)
    assert {s.member for s in drawn.samples} == {
        "a",
        "b",
        "c",
        "loose1",
        "loose2",
    }
    assert {s.ground_truth.parts[0] for s in drawn.samples} == {
        "2-11-26",
        UNDATED_BUCKET,
    }


def test_draw_splits_holds_no_sample_for_a_date_that_admits_nothing(
    tmp_path: Path,
) -> None:
    """A capture date that admits nothing contributes no sample to the draw.

    A capture whose only label document names no image anywhere contributes no sample: every
    drawn sample comes from the capture holding images, and the orphan is not among them.
    """
    from PIL import Image

    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)
    for stem in ("a", "b", "c", "d"):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(
            images_dir / f"{stem}.jpg"
        )
        label_image(
            images_dir / f"{stem}.jpg",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))],
            100,
            80,
        )
    json_io.write_label_document(
        label_key(root, "2-12-26", "orphan"),
        [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))],
        100,
        80,
    )

    out = tmp_path / "m"
    result = _draw(
        tmp_path,
        str(root),
        output_path=str(out),
        subject="leaf",
        seed=1,
        val_ratio=0.25,
        calibration_ratio=0.125,
        holdout_ratio=0.125,
    )
    assert "error" not in result, result
    drawn = read_selection(out, project=tmp_path)

    assert all(s.ground_truth.parts[0] == "2-11-26" for s in drawn.samples)
    assert "orphan" not in {s.member for s in drawn.samples}


def _write_one_sample_selection(root: Path, out: Path) -> None:
    """Write a one-sample selection of the project ``root`` under ``out``.

    Its source and label document lie under ``root``.
    """
    write_selection(
        out,
        Selection(
            samples=(
                Sample(
                    member="a",
                    source=str(root / "images" / UNDATED_BUCKET / "a.jpg"),
                    ground_truth=label_key(root, UNDATED_BUCKET, "a"),
                    group="a",
                    side="train",
                ),
            ),
            scope=registry_scope(root / "images", "leaf"),
            seed=1,
            group_by="stem",
            ratios={"train": 1.0},
            stratify_foreground=True,
            rationale=RATIONALE,
        ),
        project=root,
    )


def _stored_selection(
    out: Path,
) -> tuple[dict[str, JsonValue], list[dict[str, JsonValue]]]:
    """Return the selection document stored under ``out`` and its sample documents, as stored."""
    document = ts.read(selection_key(out))
    assert isinstance(document, dict)
    samples = document["samples"]
    assert isinstance(samples, list)
    stored = [s for s in samples if isinstance(s, dict)]
    assert len(stored) == len(samples)
    return document, stored


def test_read_selection_admits_the_writers_own_record(tmp_path: Path) -> None:
    """``read_selection`` admits the record the selection writer wrote.

    The stored source is project-relative, and the reader answers the scope, the seed and each
    sample's resolved location and label key.
    """
    out = tmp_path / "m"
    _write_one_sample_selection(tmp_path, out)

    _document, samples = _stored_selection(out)
    assert samples[0]["source"] == f"images/{UNDATED_BUCKET}/a.jpg"
    drawn = read_selection(out, project=tmp_path)

    assert drawn.scope == registry_scope(tmp_path / "images", "leaf")
    assert drawn.seed == 1
    assert [s.location for s in drawn.samples] == [
        str(tmp_path.resolve() / "images" / UNDATED_BUCKET / "a.jpg")
    ]
    assert [s.ground_truth for s in drawn.samples] == [
        label_key(tmp_path, UNDATED_BUCKET, "a")
    ]


def test_read_selection_refuses_an_absent_record_by_name(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="no selection recorded"):
        read_selection(tmp_path / "nothing", project=tmp_path)


def test_read_selection_refuses_a_sample_missing_its_own_ground_truth(
    tmp_path: Path,
) -> None:
    """``read_selection`` refuses a sample naming no ground truth of its own.

    A sample whose ``ground_truth`` is removed is refused naming that key.
    """
    out = tmp_path / "m"
    _write_one_sample_selection(tmp_path, out)
    document, samples = _stored_selection(out)
    samples[0].pop("ground_truth")
    ts.replace(selection_key(out), document)

    with pytest.raises(ValueError, match=r"ground_truth\s+Field required"):
        read_selection(out, project=tmp_path)


def test_read_selection_refuses_an_empty_sample_list(tmp_path: Path) -> None:
    out = tmp_path / "m"
    _write_one_sample_selection(tmp_path, out)
    document, _samples = _stored_selection(out)
    ts.replace(selection_key(out), {**document, "samples": []})

    with pytest.raises(ValueError, match="lists no samples"):
        read_selection(out, project=tmp_path)


def test_read_selection_refuses_one_source_on_two_sides(
    tmp_path: Path,
) -> None:
    """``read_selection`` refuses one source on two sides.

    A sample copied onto the calibration side under another group is refused as on more than
    one side.
    """
    out = tmp_path / "m"
    _write_one_sample_selection(tmp_path, out)
    document, samples = _stored_selection(out)
    ts.replace(
        selection_key(out),
        {
            **document,
            "samples": [
                *samples,
                {**samples[0], "group": "b", "side": "calibration"},
            ],
        },
    )

    with pytest.raises(ValueError, match="on more than one side"):
        read_selection(out, project=tmp_path)


def test_read_selection_refuses_one_group_on_two_sides(tmp_path: Path) -> None:
    """``read_selection`` refuses a selection placing one group on two sides.

    Two samples of group ``a``, one on each of two sides, are refused naming the group.
    """
    out = tmp_path / "m"
    _write_one_sample_selection(tmp_path, out)
    document, samples = _stored_selection(out)
    ground_truth, source = samples[0]["ground_truth"], samples[0]["source"]
    assert isinstance(ground_truth, dict) and isinstance(source, str)
    ts.replace(
        selection_key(out),
        {
            **document,
            "samples": [
                *samples,
                {
                    "member": "a_0_1",
                    "source": Path(source).with_name("a_0_1.jpg").as_posix(),
                    "ground_truth": {**ground_truth, "stem": "a_0_1"},
                    "group": "a",
                    "side": "val",
                },
            ],
        },
    )

    with pytest.raises(ValueError, match="group"):
        read_selection(out, project=tmp_path)
