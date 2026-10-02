"""Tests for data management tools."""

from __future__ import annotations

import pytest
import tcip_store as ts
from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox

from pathlib import Path

from tcip_mcp.cli import doctor
from tcip_mcp.pipelines.data.label_queries import registry_scope
from tcip_mcp.pipelines.data.selection import (
    ClassScope, Sample, Selection, read_selection, selection_key, write_selection,
)
from tcip_mcp.tools.data_tools import scan_dataset, draw_splits
from tests._producer_fixtures import registry_over


def _quality_findings(root) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    doctor.check_data_quality(Path(root), findings, census=doctor._census(Path(root), findings, set()))
    return findings


def test_scan_dataset(data_dir: Path):
    result = scan_dataset(str(data_dir))
    assert result["image_count"] == 3
    assert result["labels_count"] == 3
    assert result["paired_images"] == 3
    assert result["unlabeled_images"] == 0


def test_scan_dataset_not_found():
    result = scan_dataset("/nonexistent/path")
    assert "error" in result


def test_doctor_check_data_quality(data_dir: Path):
    findings = _quality_findings(data_dir)
    assert not [f for f in findings if f[0] == "error"]


def test_doctor_check_data_quality_missing_dir(tmp_path: Path):
    # _scan_dataset's own rglob over a nonexistent path yields nothing rather than raising; a
    # missing project root is doctor.main()'s own upfront refusal, not this check's job.
    missing = tmp_path / "nonexistent" / "xyz123"
    findings = _quality_findings(missing)
    assert not [f for f in findings if f[0] == "error"]


def test_scan_dataset_reports_a_reserved_stem_the_census_still_counted(tmp_path: Path):
    """The census walks with a raw glob and counts a label named like a bucket's own record,
    unlike every bucket walk through prediction_documents; reserved_name_labels names it so a
    caller does not read the difference as a disagreement."""
    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)
    labels_dir = root / "annotations" / "2-11-26"
    labels_dir.mkdir(parents=True)
    # Written by hand: the platform's own encoder refuses a document under a reserved stem.
    (labels_dir / "bucket.json").write_text('{"annotations": []}', encoding="utf-8")
    reserved_label = str(labels_dir / "bucket.json")

    scan_result = scan_dataset(str(root))

    assert scan_result["reserved_name_labels"] == [reserved_label]


def test_scan_dataset_reports_a_reserved_stem_image_with_no_label(tmp_path: Path):
    """An image whose own stem is reserved for a bucket's own record must be named, not folded
    into unlabeled_images with no signal that its label can never be read through any bucket
    walk."""
    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)
    (images_dir / "bucket.jpg").write_bytes(b"\xff\xd8\xff")
    (images_dir / "ordinary.jpg").write_bytes(b"\xff\xd8\xff")
    reserved_image = str(images_dir / "bucket.jpg")

    scan_result = scan_dataset(str(root))

    assert scan_result["reserved_name_images"] == [reserved_image]
    assert scan_result["unlabeled_images"] == 2


def test_scan_dataset_counts_only_the_documents_of_published_buckets(tmp_path: Path):
    """A directory of documents under ``predictions/`` holding no ``bucket.json`` is no
    bucket, so the census counts only a published bucket's documents."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import predicted, published

    root = tmp_path / "ds"
    published(tmp_path, root / "predictions" / "modelA" / "2-11-26",
              [predicted("imgA", ["bud"])], scope={"subject": "bud"})
    staged = root / "predictions" / "modelB" / "2-11-26"
    staged.mkdir(parents=True)
    json_io.write_annotations(
        staged / "imgB.json", [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))], 32, 32,
    )

    scan_result = scan_dataset(str(root))

    assert scan_result["predictions_count"] == 1


def _add_extra_bud_groups(data_dir: Path, count: int) -> None:
    """Adds ``count`` more single-tile foreground groups under ``data_dir``'s own date, for its
    own subject, without touching the shared ``data_dir`` fixture other tests depend on: a
    manifest write needs at least four foreground groups to clear ``draw_splits``' floor, one
    more than the fixture's own three."""
    from PIL import Image

    images_dir = data_dir / "images" / "2-11-26"
    labels_dir = data_dir / "annotations" / "2-11-26"
    for i in range(count):
        stem = f"extra_{i:03d}"
        Image.new("RGB", (640, 480), color=(128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="bud", geometry=BBox(288, 216, 352, 264))], 640, 480,
        )

def test_draw_splits_basic(data_dir: Path, tmp_path: Path):
    _add_extra_bud_groups(data_dir, 1)
    out = tmp_path / "manifests"
    # The fixture's 4 stems (img_001..003 plus one grown group) are 4 distinct foreground
    # groups, exactly the manifest floor (one each for train/val, two for calibration).
    result = draw_splits(data_dir, str(data_dir), output_path=str(out), subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)
    assert "error" not in result, result
    assert result["total_stems"] == 4
    assert result["groups"] == 4
    assert sum(result["splits"].values()) == 4
    assert result["stratified"] is True

    drawn = read_selection(out, project=data_dir)
    assert drawn.counts() == {k: v for k, v in result["splits"].items()}
    assert all(drawn.counts()[side] for side in ("train", "val", "calibration"))
    assert drawn.scope.subject == "bud"
    assert drawn.scope.attributes == ()
    assert drawn.dataset_fingerprint is not None
    for sample in drawn.samples:
        assert Path(sample.source).is_file()
        assert Path(sample.ground_truth).is_file()
        assert sample.ground_truth_digest


LEAF_DATE = "2-11-26"


def _leaf_scene(root: Path, n: int = 6) -> Path:
    """``n`` captures of one date, the ``i``-th label holding ``i + 1`` leaves and ``5 * (n - i)``
    buds, so a count scoped to the leaf and one over every record disagree."""
    from PIL import Image

    images, labels = root / "images" / LEAF_DATE, root / "annotations" / LEAF_DATE
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images / f"s{i}.jpg")
        json_io.write_annotations(labels / f"s{i}.json", [
            *(Annotation(subject="leaf", geometry=BBox(2, 2, 6, 6)) for _ in range(i + 1)),
            *(Annotation(subject="bud", geometry=BBox(8, 8, 12, 12)) for _ in range(5 * (n - i))),
        ], 100, 80)
    return root


def test_a_draw_without_stratification_reports_the_counted_annotations(tmp_path: Path):
    """Turning the balancing off changes how the draw sides its members, never what it counts:
    every side's foreground is the subject's own annotations it holds."""
    root = _leaf_scene(tmp_path / "ds")
    result = draw_splits(tmp_path, str(root), subject="leaf", stratify_foreground=False)
    assert "error" not in result, result
    assert result["stratified"] is False
    assert result["total_annotations"] == sum(range(1, 7))
    assert sum(result["foreground_annotations"].values()) == result["total_annotations"]


def test_draw_splits_and_a_runs_own_draw_side_the_same_members(tmp_path: Path):
    """``draw_splits`` and a run's own train/val draw over the same ground truth, seed, policy
    and val share answer the same sides: the tool's per-side sizes and foreground equal what the
    run's partition holds, its foreground recounted here from each member's own leaves."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _leaf_scene(tmp_path / "ds")
    drawn = draw_splits(tmp_path, str(root), subject="leaf", train_ratio=0.5, val_ratio=0.5,
                        group_by="stem", seed=7)
    assert "error" not in drawn, drawn
    data_cfg = {"images_dir": str(root / "images" / LEAF_DATE),
                "labels_dir": str(root / "annotations" / LEAF_DATE), "scope": {"subject": "leaf"},
                "split": {"val_ratio": 0.5, "seed": 7, "group_by": "stem"}}
    _train, _val, partition = auto_train_val(tmp_path, "detection", data_cfg, None)

    by_side: dict[str, list[int]] = {"train": [], "val": []}
    for sample in partition["samples"]:
        by_side[sample["side"]].append(int(sample["member"].removeprefix("s")) + 1)
    assert {side: len(v) for side, v in by_side.items()} == {
        side: drawn["splits"][side] for side in by_side}
    assert {side: sum(v) for side, v in by_side.items()} == {
        side: drawn["foreground_annotations"][side] for side in by_side}


def test_every_draw_refuses_a_tree_short_of_its_floor_the_same_way(tmp_path: Path):
    """``draw_splits`` and a run's own draw hold one floor: two captures mapped into one group
    are one foreground group, short of a train and a val side, and both refuse with that floor's
    words rather than one of them training without validation."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _leaf_scene(tmp_path / "ds", n=2)
    one_group = {f"{LEAF_DATE}/s{i}": "plot" for i in range(2)}
    floor = "1 foreground group(s), fewer than the 2 the requested sides need"

    drawn = draw_splits(tmp_path, str(root), subject="leaf", train_ratio=0.5, val_ratio=0.5,
                        group_key_map=one_group)
    assert floor in drawn["error"]
    data_cfg = {"images_dir": str(root / "images" / LEAF_DATE),
                "labels_dir": str(root / "annotations" / LEAF_DATE), "scope": {"subject": "leaf"},
                "split": {"val_ratio": 0.5, "group_key_map": one_group}}
    with pytest.raises(ValueError, match=floor.replace("(", r"\(").replace(")", r"\)")):
        auto_train_val(tmp_path, "detection", data_cfg, None)


def test_draw_splits_refuses_a_version_refused_subject_registry_as_an_error(data_dir: Path, tmp_path: Path):
    from tcip_mcp.dataset_layout import subject_registry_key

    _add_extra_bud_groups(data_dir, 1)
    ts.put_blob(
        subject_registry_key(data_dir), ts.RECORD_JSON.encode({"schema_version": 99})
    )
    result = draw_splits(data_dir, str(data_dir), output_path=str(tmp_path / "manifests"), subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)
    assert "error" in result, result
    assert "schema_version" in result["error"]


def test_draw_splits_stats_only_admits_a_nonzero_calibration_ratio(data_dir: Path):
    """A stats-only call (no output_path) may pass any calibration_ratio, a zero holdout_ratio
    included; only writing a selection requires every ratio non-zero."""
    result = draw_splits(data_dir, str(data_dir), train_ratio=0.7, val_ratio=0.2,
                         calibration_ratio=0.1, subject="bud")
    assert "error" not in result, result
    assert result["splits"]["calibration"] > 0
    assert result["splits"]["holdout"] == 0


def test_draw_splits_reports_an_unreadable_label_by_name(data_dir: Path, tmp_path: Path):
    """A present, unreadable label among the candidates is an error naming the file, never a
    raise through the tool boundary."""
    bad = next((data_dir / "annotations" / "2-11-26").glob("*.json"))
    bad.write_bytes(b"{not json")

    result = draw_splits(data_dir, str(data_dir), output_path=str(tmp_path / "manifests"), subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)

    assert "error" in result
    assert str(bad) in result["error"]


def test_draw_splits_reports_an_unreadable_label_sorted_last(
    data_dir: Path, tmp_path: Path,
):
    """A corrupt label reached last in sort order is caught by the same per-stem admission read
    as the first-sorted case above, regardless of where in the candidate order it falls, and
    answers the same error dict, never a raw raise."""
    bad = sorted((data_dir / "annotations" / "2-11-26").glob("*.json"))[-1]
    bad.write_bytes(b"{not json")

    result = draw_splits(data_dir, str(data_dir), output_path=str(tmp_path / "manifests"), subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)

    assert "error" in result
    assert str(bad) in result["error"]


def test_draw_splits_writes_nothing_when_a_marked_document_will_not_read(
    data_dir: Path, tmp_path: Path,
):
    """An unreadable label on an image a human marked complete refuses the whole draw and leaves
    no selection behind: a partial record would claim a partition nobody drew, and the mark cannot
    be read from a document that will not parse."""
    from tcip_mcp.dataset_layout import image_dir
    from tests._producer_fixtures import mark_complete

    bad = data_dir / "annotations" / "2-11-26" / "img_002.json"
    mark_complete(image_dir(data_dir, "2-11-26") / "img_002.jpg", bad, "bud", project=data_dir)
    bad.write_bytes(b"{not json")
    out = tmp_path / "selection"

    result = draw_splits(data_dir, str(data_dir), output_path=str(out), subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)

    assert "error" in result
    assert str(bad) in result["error"]
    assert not ts.exists(selection_key(out))


def test_draw_splits_stats_only_reports_an_unreadable_first_sorted_label(data_dir: Path):
    """A stats-only call (no output_path) draws through the same admission a written one does, so
    an unreadable first-sorted candidate is an error naming it."""
    bad = data_dir / "annotations" / "2-11-26" / "img_001.json"
    bad.write_bytes(b"{not json")

    result = draw_splits(data_dir, str(data_dir), subject="bud")

    assert "error" in result
    assert str(bad) in result["error"]


def test_draw_splits_stats_only_reports_an_unreadable_label_during_stratification(data_dir: Path):
    """A stats-only call still reads every stem's label, so a corrupt label reached after a
    readable first candidate is an error naming the file, not a raw raise."""
    bad = data_dir / "annotations" / "2-11-26" / "img_003.json"
    bad.write_bytes(b"{not json")

    result = draw_splits(data_dir, str(data_dir), subject="bud")

    assert "error" in result
    assert str(bad) in result["error"]


def test_draw_splits_manifest_answers_an_ambiguous_image_stem_as_an_error(tmp_path: Path):
    """A raw file colliding with a band group's own canonical stem is an error naming the
    directory, never a raise through the tool boundary, the same contract the unreadable-label
    cases above state."""
    import numpy as np

    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)
    (root / "annotations" / "2-11-26").mkdir(parents=True)

    band_a, band_b = images_dir / "plotA_B1.npy", images_dir / "plotA_B2.npy"
    np.save(band_a, np.zeros((4, 4), dtype=np.uint8))
    np.save(band_b, np.zeros((4, 4), dtype=np.uint8))
    write_band_group_manifest(images_dir, "plotA", {"B1": band_a, "B2": band_b})
    (images_dir / "plotA.jpg").write_bytes(b"\xff\xd8\xff")

    result = draw_splits(tmp_path, str(root), output_path=str(tmp_path / "manifests"), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)

    assert "error" in result
    assert "plotA" in result["error"]


def test_draw_splits_stats_only_answers_an_ambiguous_image_stem_as_an_error(tmp_path: Path):
    """A stats-only call (no output_path) answers the identical stem collision a selection-writing
    call reports as an error, never a raw raise: both draw through the one admission."""
    import numpy as np

    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)
    (root / "annotations" / "2-11-26").mkdir(parents=True)

    band_a, band_b = images_dir / "plotA_B1.npy", images_dir / "plotA_B2.npy"
    np.save(band_a, np.zeros((4, 4), dtype=np.uint8))
    np.save(band_b, np.zeros((4, 4), dtype=np.uint8))
    write_band_group_manifest(images_dir, "plotA", {"B1": band_a, "B2": band_b})
    (images_dir / "plotA.jpg").write_bytes(b"\xff\xd8\xff")

    result = draw_splits(tmp_path, str(root), subject="leaf")

    assert "error" in result
    assert "plotA" in result["error"]


def test_scan_dataset_answers_a_newer_written_bandgroup_manifest_as_an_error(tmp_path: Path):
    """A ``.bandgroup`` manifest above this reader's ceiling propagates as
    ``tcip_store.SchemaVersionRefused`` out of ``list_logical_images``, uncaught by design;
    ``scan_dataset`` must answer that as a named error rather than letting it raise through the
    tool boundary, the same contract it already holds for an ambiguous stem."""
    from tcip_store.file_backend import FileBackend
    import tcip_store as ts

    from tcip_mcp.pipelines.data.band_groups import band_group_manifest_key

    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)

    key = band_group_manifest_key(images_dir, "plotA")
    path = FileBackend().path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ts.RECORD_JSON.encode(
        {"schema_version": 2, "bands": {"B1": "plotA_B1.npy", "B2": "plotA_B2.npy"}}))

    result = scan_dataset(str(root))

    assert "error" in result
    assert "schema_version 2, above the 1 this reader knows" in result["error"]


def test_scan_dataset_answers_an_ambiguous_image_stem_as_an_error(tmp_path: Path):
    """``scan_dataset`` answers the same stem collision as an error rather than letting
    ``AmbiguousImageStem`` escape uncaught through the tool boundary."""
    import numpy as np

    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)

    band_a, band_b = images_dir / "plotA_B1.npy", images_dir / "plotA_B2.npy"
    np.save(band_a, np.zeros((4, 4), dtype=np.uint8))
    np.save(band_b, np.zeros((4, 4), dtype=np.uint8))
    write_band_group_manifest(images_dir, "plotA", {"B1": band_a, "B2": band_b})
    (images_dir / "plotA.jpg").write_bytes(b"\xff\xd8\xff")

    result = scan_dataset(str(root))

    assert "error" in result
    assert "plotA" in result["error"]


def _add_extra_leaf_groups(images_dir: Path, labels_dir: Path, count: int) -> None:
    """Adds ``count`` more plain single-tile foreground groups (subject ``leaf``) beside a
    band-group fixture, so a manifest write over it clears the four-foreground-group floor."""
    from PIL import Image

    letters = "BCDEFGH"
    for i in range(count):
        stem = f"plot{letters[i]}"
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
        )


def test_draw_splits_refuses_an_incomplete_band_group_before_writing(tmp_path: Path):
    """A band group whose manifest names a missing sibling is refused before the selection is
    written, never a raise through the tool after it lands."""
    import numpy as np

    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)
    labels_dir = root / "annotations" / "2-11-26"
    labels_dir.mkdir(parents=True)

    band_g, band_r = images_dir / "plotA_G.npy", images_dir / "plotA_R.npy"
    np.save(band_g, np.zeros((4, 4), dtype=np.uint8))
    write_band_group_manifest(images_dir, "plotA", {"G": band_g, "R": band_r})  # R never created
    json_io.write_annotations(
        labels_dir / "plotA.json",
        [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
    )
    _add_extra_leaf_groups(images_dir, labels_dir, 3)
    out = tmp_path / "m"

    result = draw_splits(tmp_path, str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)

    assert "error" in result
    assert "plotA" in result["error"] and "R" in result["error"]
    assert not out.exists()


def test_draw_splits_bad_ratios(data_dir: Path):
    result = draw_splits(data_dir, str(data_dir), train_ratio=0.5, val_ratio=0.5,
                         calibration_ratio=0.25, holdout_ratio=0.25)
    assert "error" in result


def test_draw_splits_ratios_not_summing_to_one_names_every_share(tmp_path: Path):
    """The sum refusal names every share it was given, not just the raw sum."""
    root = _multi_source_dataset(tmp_path / "ds")
    result = draw_splits(tmp_path, str(root), subject="bud", train_ratio=0.7, val_ratio=0.2,
                         calibration_ratio=0.1, holdout_ratio=0.1)
    assert "summing to exactly 1" in result["error"]
    assert all(side in result["error"] for side in ("train", "val", "calibration", "holdout"))


def test_draw_splits_selection_carries_all_four_sides(tmp_path: Path):
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"

    result = draw_splits(tmp_path, str(root), output_path=str(out), seed=1, subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)

    assert "error" not in result, result
    counts = read_selection(out, project=tmp_path).counts()
    assert set(counts) == {"train", "val", "calibration", "holdout"}
    assert all(counts.values())


def test_draw_splits_floor_refuses_before_any_write_regardless_of_stratify_foreground(
    tmp_path: Path,
):
    """The foreground floor is over the draw's own subject-scoped counter on every selection
    draw, whether or not stratify_foreground toggles the balancing pass: a tree with only three
    foreground groups refuses before anything is written, with stratify_foreground off."""
    root = _multi_source_dataset(tmp_path / "ds", prefixes=("srcA", "srcB", "srcC"))
    out = tmp_path / "m"

    result = draw_splits(tmp_path, str(root), output_path=str(out), subject="bud", seed=1,
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125,
                         stratify_foreground=False)

    assert "error" in result
    assert "foreground group" in result["error"]
    assert not out.exists()


def test_draw_splits_answers_one_draw_whether_or_not_it_writes(tmp_path: Path):
    """A zero ratio draws no side of that name and a negative one refuses, with or without
    ``output_path``: what is drawn never depends on whether it is written."""
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    ratios = {"train_ratio": 0.75, "val_ratio": 0.0, "calibration_ratio": 0.125,
              "holdout_ratio": 0.125}

    stats = draw_splits(tmp_path, str(root), subject="bud", seed=1, **ratios)
    written = draw_splits(tmp_path, str(root), output_path=str(out), subject="bud", seed=1,
                          **ratios)

    assert "error" not in written, written
    assert {key: value for key, value in written.items() if key != "selection_dir"} == {
        key: value for key, value in stats.items() if key != "selection_dir"}
    assert written["splits"]["val"] == 0
    for output_path in (None, str(tmp_path / "negative")):
        refused = draw_splits(tmp_path, str(root), output_path=output_path, subject="bud",
                              seed=1, train_ratio=0.875, val_ratio=-0.25,
                              calibration_ratio=0.25, holdout_ratio=0.125)
        assert "share in (0, 1)" in refused["error"]
    assert not (tmp_path / "negative").exists()


def test_draw_splits_floor_ignores_a_groups_only_annotations_of_another_subject(tmp_path: Path):
    """A confirmed-negative-for-the-draws-subject group whose label file happens to carry
    another subject's annotation is not this draw's foreground: the subject-scoped counter
    reads it as zero, so three real ``leaf`` groups plus one such group still refuse (below the
    floor of four), the same tree an unscoped counter would have read as four and written."""
    from PIL import Image

    from tcip_mcp.subject_registry import SubjectRegistry, Subject
    from tests._producer_fixtures import mark_complete

    root = tmp_path / "ds"
    date = "2-11-26"
    images_dir, labels_dir = root / "images" / date, root / "annotations" / date
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    registry_over(root,SubjectRegistry(subjects=(
        Subject(name="leaf"), Subject(name="bud"),
    )))
    for stem in ("p1", "p2", "p3"):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
        )
    Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / "p4.jpg")
    json_io.write_annotations(
        labels_dir / "p4.json",
        [Annotation(subject="bud", geometry=BBox(4, 4, 12, 12))], 100, 80, keep_empty=True,
    )
    mark_complete(images_dir / "p4.jpg", labels_dir / "p4.json", "leaf", project=root)

    out = tmp_path / "m"
    result = draw_splits(tmp_path, str(root), output_path=str(out), subject="leaf", seed=1,
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)

    assert "error" in result
    assert "foreground group" in result["error"]
    assert not out.exists()


def _leaf_dataset_with_negatives(root: Path, date: str, n_foreground: int, n_negative: int) -> None:
    """``n_foreground`` single-tile ``leaf`` groups plus ``n_negative`` confirmed-negative
    images, all under one capture date."""
    from PIL import Image

    from tests._producer_fixtures import mark_complete

    images_dir, labels_dir = root / "images" / date, root / "annotations" / date
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_foreground):
        stem = f"{date}_fg{i}"
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
        )
    for i in range(n_negative):
        stem = f"{date}_bg{i}"
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(labels_dir / f"{stem}.json", [], 100, 80, keep_empty=True)
        mark_complete(images_dir / f"{stem}.jpg", labels_dir / f"{stem}.json", "leaf",
                      project=root)


def test_draw_splits_calibration_side_holds_real_foreground_regardless_of_stratify_foreground(
    tmp_path: Path,
):
    """The minimum-foreground pass sees the draw's subject-scoped foreground counts on every
    selection draw, not only when stratify_foreground also balances by them: the calibration
    side's stated minimum of two foreground groups is met with real foreground even with
    balancing off, across every seed."""
    root = tmp_path / "ds"
    date = "2-11-26"
    _leaf_dataset_with_negatives(root, date, n_foreground=4, n_negative=6)

    for seed in range(1, 21):
        out = tmp_path / f"m{seed}"
        result = draw_splits(tmp_path, str(root), output_path=str(out), subject="leaf", seed=seed,
                             train_ratio=0.8, val_ratio=0.1, calibration_ratio=0.05,
                             holdout_ratio=0.05, stratify_foreground=False)
        assert "error" not in result, (seed, result)
        assert result["calibration_foreground_groups"] >= 2, (seed, result)

def _multi_source_dataset(root: Path, prefixes=("srcA", "srcB", "srcC", "srcD"), tiles=3) -> Path:
    from PIL import Image

    date = "2-11-26"
    images_dir = root / "images" / date
    images_dir.mkdir(parents=True)
    labels_dir = root / "annotations" / date
    labels_dir.mkdir(parents=True)
    for pref in prefixes:
        for t in range(tiles):
            stem = f"{pref}_{t}_0"
            Image.new("RGB", (64, 64), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
            json_io.write_annotations(labels_dir / f"{stem}.json",
                                      [Annotation(subject="bud", geometry=BBox(19, 13, 45, 51))],
                                      64, 64)
    return root


def _bud_membership(root: Path):
    from tcip_mcp.pipelines.data.split_construction import admitted_membership

    date = "2-11-26"
    return admitted_membership(
        [(date, root / "images" / date, str(root / "annotations" / date))],
        scope=ClassScope(subject="bud"), group_by="tile_prefix", group_key_map=None)


def _spied_draws(monkeypatch) -> list[bool]:
    """Each ``weighted`` the draw hands the split algorithm, in call order."""
    import tcip_mcp.pipelines.data.splits as splits

    calls: list[bool] = []
    real = splits.group_balanced_split

    def spy(stems, **kwargs):
        calls.append(kwargs["weighted"])
        return real(stems, **kwargs)

    monkeypatch.setattr(splits, "group_balanced_split", spy)
    return calls


@pytest.mark.parametrize("ratios", [
    {"train": 1.0}, {"train": 0.75, "val": 0.0, "calibration": 0.25},
    {"train": 1.25, "val": -0.25}, {"train": 0.5, "val": 0.495}],
    ids=["one", "zero", "negative", "short_of_one"])
def test_draw_sides_refuses_a_share_outside_zero_to_one_before_any_draw(
    tmp_path: Path, monkeypatch, ratios,
):
    from tcip_mcp.pipelines.data.split_construction import draw_sides

    membership = _bud_membership(_multi_source_dataset(tmp_path / "ds"))
    draws = _spied_draws(monkeypatch)
    with pytest.raises(ValueError, match=r"share in \(0, 1\)"):
        draw_sides(membership.samples, membership.scope, ratios=ratios, seed=1, stratify=True)
    assert draws == []


@pytest.mark.parametrize("stratify", [True, False])
def test_draw_sides_cuts_the_reference_under_the_balancing_it_was_given(
    tmp_path: Path, monkeypatch, stratify,
):
    from tcip_mcp.pipelines.data.split_construction import draw_sides

    membership = _bud_membership(_multi_source_dataset(tmp_path / "ds"))
    draws = _spied_draws(monkeypatch)
    drawn, _counted = draw_sides(
        membership.samples, membership.scope, seed=1, stratify=stratify,
        ratios={"train": 0.5, "val": 0.25, "calibration": 0.125, "holdout": 0.125})
    assert draws == [stratify, stratify]
    assert {sample.side for sample in drawn.values()} == {"train", "val", "calibration", "holdout"}


def test_draw_splits_groups_tiles_together(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import default_group_key

    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = draw_splits(tmp_path, str(root), output_path=str(out), seed=1, subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)
    assert result["groups"] == 4  # 4 source prefixes, not 12 tiles

    # No source prefix may appear in more than one split.
    seen: dict[str, str] = {}
    for sample in read_selection(out, project=tmp_path).samples:
        g = default_group_key(Path(sample.ground_truth).stem)
        assert seen.get(g, sample.side) == sample.side, f"group {g} spans splits"
        seen[g] = sample.side


def test_draw_splits_group_key_map_never_straddles(tmp_path: Path):
    """An agent-derived group_key_map (5 samples, 4 groups) is honored: the two same-group
    samples never land in different splits."""
    root = _multi_source_dataset(tmp_path / "ds", prefixes=("x", "y", "z", "w", "v"), tiles=1)
    out = tmp_path / "m"
    group_key_map = {
        "2-11-26/x_0_0": "gA", "2-11-26/y_0_0": "gA", "2-11-26/z_0_0": "gB",
        "2-11-26/w_0_0": "gC", "2-11-26/v_0_0": "gD",
    }
    result = draw_splits(tmp_path, str(root), output_path=str(out), seed=1,
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125,
                         group_by="tile_prefix", group_key_map=group_key_map, subject="bud")
    assert "error" not in result, result
    assert result["group_by"] == "explicit_map"

    drawn = read_selection(out, project=tmp_path)
    by_stem = {Path(s.ground_truth).stem: s for s in drawn.samples}
    assert by_stem["x_0_0"].side == by_stem["y_0_0"].side  # gA never straddles
    assert by_stem["x_0_0"].group == by_stem["y_0_0"].group == "gA"
    assert drawn.group_by == "explicit_map"


def test_draw_splits_unrecognized_group_by_refuses_without_writing(tmp_path: Path):
    """An unrecognized ``group_by`` must refuse loudly and write nothing, never fall back to
    ``GROUP_KEY_FNS.get(group_by, default_group_key)`` and mis-group a dataset silently."""
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = draw_splits(tmp_path, str(root), output_path=str(out), group_by="not_a_real_key", subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)
    assert "error" in result
    assert not out.exists() or not (out / "selection.json").is_file()


def test_draw_splits_refuses_to_write_a_selection_with_no_subject(tmp_path: Path):
    """A selection with no subject would be a partition of images, not of a run's admissible
    samples; draw_splits refuses to write one rather than guessing what a run would admit."""
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = draw_splits(tmp_path, str(root), output_path=str(out),
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)
    assert "error" in result and "subject" in result["error"]
    assert not out.exists()


def _two_date_collision_dataset(root: Path, subject: str) -> Path:
    """One stem name, ``shared``, present under two capture dates with different content, plus
    one more distinct stem per date so a selection drawn over this tree clears the foreground
    floor: a record keyed by bare stem could only ever hold one of the two ``shared`` images."""
    from PIL import Image

    for date, box_x in (("2-11-26", 4), ("2-12-01", 40)):
        images_dir = root / "images" / date
        labels_dir = root / "annotations" / date
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / "shared.jpg")
        json_io.write_annotations(
            labels_dir / "shared.json",
            [Annotation(subject=subject, geometry=BBox(box_x, 4, box_x + 8, 12))], 100, 80,
        )
        extra_stem = f"extra_{date}"
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{extra_stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{extra_stem}.json",
            [Annotation(subject=subject, geometry=BBox(4, 4, 12, 12))], 100, 80,
        )
    return root


def test_two_dates_sharing_a_filename_stay_distinct_samples(tmp_path: Path):
    """The two ``shared`` images are two samples, each naming its own source and label under its
    own capture date, and each reads back as distinct pixels: a selection carries no bare-stem
    identity a second date could collide with."""
    root = _two_date_collision_dataset(tmp_path / "ds", subject="leaf")
    out = tmp_path / "m"
    result = draw_splits(tmp_path, str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125, seed=1)
    assert "error" not in result, result
    assert result["total_stems"] == 4

    drawn = read_selection(out, project=tmp_path)
    shared = [s for s in drawn.samples if Path(s.ground_truth).stem == "shared"]
    assert len(shared) == 2
    assert {Path(s.source).parent.name for s in shared} == {"2-11-26", "2-12-01"}
    assert {Path(s.ground_truth).parent.name for s in shared} == {"2-11-26", "2-12-01"}
    assert len({s.location for s in shared}) == 2
    # Each sample's own label document is the one under its own date, never the other's.
    boxes = {
        json_io.read_annotations(s.ground_truth)[0].geometry.x1 for s in shared  # type: ignore[union-attr]
    }
    assert boxes == {4.0, 40.0}


def _two_subject_dataset(root: Path) -> Path:
    """Six stems on one date: four carry ``leaf``, two carry the unrelated subject ``bud``, no
    stem carries both; four ``leaf`` stems clear a leaf-scoped draw's foreground floor."""
    from PIL import Image

    from tcip_mcp.subject_registry import SubjectRegistry, Subject

    date = "2-11-26"
    images_dir = root / "images" / date
    labels_dir = root / "annotations" / date
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    registry_over(root,SubjectRegistry(subjects=(
        Subject(name="leaf"), Subject(name="bud"),
    )))
    for stem, subject in (
        ("leaf_a", "leaf"), ("leaf_b", "leaf"), ("leaf_c", "leaf"), ("leaf_d", "leaf"),
        ("bud_a", "bud"), ("bud_b", "bud"),
    ):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject=subject, geometry=BBox(4, 4, 12, 12))], 100, 80,
        )
    return root


def test_draw_splits_holds_only_the_named_subjects_admitted_samples(tmp_path: Path):
    root = _two_subject_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = draw_splits(tmp_path, str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125, seed=1)
    assert "error" not in result, result
    assert result["total_stems"] == 4
    drawn = read_selection(out, project=tmp_path)
    assert {Path(s.ground_truth).stem for s in drawn.samples} == {
        "leaf_a", "leaf_b", "leaf_c", "leaf_d"}


def _attribute_scoped_dataset(root: Path) -> Path:
    """Five stems on one date, one subject declaring ``condition``: four have their instance
    assessed for it, one carries an instance never assessed for it."""
    from PIL import Image

    from tcip_mcp.subject_registry import Attribute, SubjectRegistry, Subject

    date = "2-11-26"
    images_dir = root / "images" / date
    labels_dir = root / "annotations" / date
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    registry_over(root,SubjectRegistry(subjects=(
        Subject(name="leaf", attributes=(
            Attribute(name="condition", type="categorical", values=("healthy", "damaged")),
        )),
    )))
    for stem, condition in (
        ("assessed_a", "healthy"), ("assessed_b", "damaged"),
        ("assessed_c", "healthy"), ("assessed_d", "damaged"),
    ):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12),
                       attributes={"condition": condition})], 100, 80,
        )
    Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / "unassessed.jpg")
    json_io.write_annotations(
        labels_dir / "unassessed.json",
        [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
    )
    return root


def test_draw_splits_records_every_declared_attribute_and_keeps_an_unassessed_sample(
    tmp_path: Path,
):
    root = _attribute_scoped_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = draw_splits(tmp_path, str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125, seed=1)
    assert "error" not in result, result
    assert result["total_stems"] == 5
    drawn = read_selection(out, project=tmp_path)
    assert [a.name for a in drawn.scope.attributes] == ["condition"]
    assert {Path(s.ground_truth).stem for s in drawn.samples} == {
        "assessed_a", "assessed_b", "assessed_c", "assessed_d", "unassessed"}


def _two_date_flat_images_dataset(root: Path, subject: str) -> Path:
    """Two dated label directories whose images were never split into date buckets: both
    entries fall back to the same flat images/ root."""
    from PIL import Image

    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    for date in ("2-11-26", "2-12-01"):
        labels_dir = root / "annotations" / date
        labels_dir.mkdir(parents=True)
        for stem in ("w", "x", "y", "z"):
            dst = images_dir / f"{stem}.jpg"
            if not dst.exists():
                Image.new("RGB", (100, 80), (128, 128, 128)).save(dst)
            json_io.write_annotations(
                labels_dir / f"{stem}.json",
                [Annotation(subject=subject, geometry=BBox(4, 4, 12, 12))], 100, 80,
            )
    return root


def test_draw_splits_refuses_two_dated_label_dirs_sharing_a_flat_images_root(tmp_path: Path):
    """Two label dates whose images were never split into date buckets both resolve to the
    same flat images/ root: a manifest keyed by <date>/<stem> would admit one image file once
    per date and could place the same pixels on both sides of the split."""
    root = _two_date_flat_images_dataset(tmp_path / "ds", subject="leaf")
    out = tmp_path / "m"

    result = draw_splits(tmp_path, str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)

    assert "error" in result
    assert "2-11-26" in result["error"] and "2-12-01" in result["error"]
    assert not out.exists()


def test_draw_splits_refuses_a_dated_dir_and_loose_labels_sharing_a_flat_images_root(
    tmp_path: Path,
):
    """A dated label directory with no images/<date>/ bucket of its own and a loose label
    beside it both fall back to the same flat images/ root: the same leak, mirrored."""
    from PIL import Image

    root = tmp_path / "ds"
    images_dir = root / "images"
    dated_labels = root / "annotations" / "2-11-26"
    images_dir.mkdir(parents=True)
    dated_labels.mkdir(parents=True)
    for stem in ("a", "b", "c", "d"):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
    for stem in ("b", "c", "d"):
        json_io.write_annotations(
            dated_labels / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
        )
    json_io.write_annotations(
        root / "annotations" / "a.json",
        [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
    )
    out = tmp_path / "m"

    result = draw_splits(tmp_path, str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)

    assert "error" in result
    assert "2-11-26" in result["error"]
    assert "annotations/ (loose labels)" in result["error"]
    assert not out.exists()


def test_draw_splits_nothing_admitted_names_the_searched_directories(tmp_path: Path):
    """A tree whose labels sit flat while its images were split into a date bucket admits
    nothing: the refusal names each entry's searched directory."""
    from PIL import Image

    root = tmp_path / "ds"
    dated_images = root / "images" / "2-11-26"
    dated_images.mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    Image.new("RGB", (100, 80), (128, 128, 128)).save(dated_images / "a.jpg")
    json_io.write_annotations(
        root / "annotations" / "a.json",
        [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
    )
    out = tmp_path / "m"

    result = draw_splits(tmp_path, str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)

    assert "error" in result
    assert "annotations/ (loose labels)" in result["error"]
    assert str(root / "images") in result["error"]
    assert not out.exists()


def _dated_labels_flat_images_dataset(root: Path, stems: tuple[str, ...]) -> Path:
    """Labels dated but images never split into date buckets: a layout the platform's other
    readers already resolve (``annotation_tools.py``'s stage-shape door)."""
    from PIL import Image

    images_dir = root / "images"
    labels_dir = root / "annotations" / "2-11-26"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for stem in stems:
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
        )
    return root


def test_draw_splits_manifest_admits_dated_labels_over_flat_images(tmp_path: Path):
    root = _dated_labels_flat_images_dataset(tmp_path / "ds", ("p0", "p1", "p2", "p3", "p4"))
    out = tmp_path / "m"
    result = draw_splits(tmp_path, str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)
    assert "error" not in result
    assert result["total_stems"] == 5
    assert result["tallies"]["annotated"] == 5


def test_draw_splits_manifest_admits_a_loose_label_beside_a_dated_one(tmp_path: Path):
    """A label sitting loose in ``annotations/`` beside a dated bucket enters the draw as a
    dateless member, rather than being invisible to both the manifest and its own counts."""
    from PIL import Image

    root = tmp_path / "ds"
    dated_images = root / "images" / "2-11-26"
    dated_labels = root / "annotations" / "2-11-26"
    dated_images.mkdir(parents=True)
    dated_labels.mkdir(parents=True)
    for stem in ("a", "b", "c"):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(dated_images / f"{stem}.jpg")
        json_io.write_annotations(
            dated_labels / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
        )
    for stem in ("loose1", "loose2"):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(root / "images" / f"{stem}.jpg")
        json_io.write_annotations(
            root / "annotations" / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
        )

    out = tmp_path / "m"
    result = draw_splits(tmp_path, str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)

    assert "error" not in result
    assert result["total_stems"] == 5
    assert result["tallies"]["annotated"] == 5
    drawn = read_selection(out, project=tmp_path)
    assert {Path(s.ground_truth).stem for s in drawn.samples} == {"a", "b", "c", "loose1", "loose2"}
    assert {str(Path(s.ground_truth).parent.relative_to(root)) for s in drawn.samples} == {
        str(Path("annotations") / "2-11-26"), "annotations"}


def test_a_stray_bucket_record_under_annotations_is_not_searched_as_loose_labels(
    tmp_path: Path,
):
    """A bucket's own record sitting loose directly under ``annotations/`` is not a loose label,
    so a draw that admits nothing names only the dated place it searched."""
    root = tmp_path / "ds"
    (root / "annotations" / "2-11-26").mkdir(parents=True)
    (root / "images" / "2-11-26").mkdir(parents=True)
    (root / "annotations" / "bucket.json").write_text('{"checkpoint": "m"}', encoding="utf-8")

    result = draw_splits(tmp_path, str(root), subject="leaf")

    assert "error" in result
    assert "2-11-26 ->" in result["error"]
    assert "loose labels" not in result["error"]


def test_draw_splits_holds_no_sample_for_a_date_that_admits_nothing(tmp_path: Path):
    """A capture date whose only label resolves to no image anywhere contributes no sample, so a
    selection never carries a member whose pixels are gone."""
    from PIL import Image

    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    labels_dir = root / "annotations" / "2-11-26"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for stem in ("a", "b", "c", "d"):
        Image.new("RGB", (100, 80), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
        )
    orphan_labels = root / "annotations" / "2-12-26"
    orphan_labels.mkdir(parents=True)
    json_io.write_annotations(
        orphan_labels / "orphan.json",
        [Annotation(subject="leaf", geometry=BBox(4, 4, 12, 12))], 100, 80,
    )

    out = tmp_path / "m"
    result = draw_splits(tmp_path, str(root), output_path=str(out), subject="leaf",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)
    assert "error" not in result
    drawn = read_selection(out, project=tmp_path)

    assert all(Path(s.ground_truth).parent.name == "2-11-26" for s in drawn.samples)
    assert "orphan" not in {Path(s.ground_truth).stem for s in drawn.samples}


def test_doctor_check_data_quality_admits_a_confirmed_negative_under_dated_labels_flat_images(
    tmp_path: Path,
):
    """A human-confirmed negative resolves the same way the doctor's check reads it as the
    draw that admits it: labels dated, images never split into date buckets."""
    from tests._producer_fixtures import mark_complete

    root = _dated_labels_flat_images_dataset(tmp_path / "ds", ("p0",))
    label = root / "annotations" / "2-11-26" / "p0.json"
    label.unlink()
    json_io.write_annotations(label, [], 100, 80, keep_empty=True)
    mark_complete(root / "images" / "p0.jpg", label, "leaf", project=root)

    assert _quality_findings(root) == []


def _write_one_sample_selection(root: Path, out: Path) -> None:
    """A one-sample selection of the project ``root`` written under ``out``, its source and
    label under ``root``."""
    write_selection(out, Selection(
        samples=(Sample(member="a", source=str(root / "images" / "a.jpg"),
                        ground_truth=str(root / "annotations" / "a.json"), group="a",
                        side="train"),),
        scope=registry_scope(root, "leaf"), seed=1, group_by="stem",
    ), project=root)


def test_read_selection_admits_the_writers_own_record(tmp_path: Path):
    """The reader accepts exactly what the writer wrote, through the platform's own producer,
    each path stored under the project and read back where it lies."""
    out = tmp_path / "m"
    _write_one_sample_selection(tmp_path, out)

    assert ts.read(selection_key(out))["samples"][0]["source"] == "images/a.jpg"
    drawn = read_selection(out, project=tmp_path)

    assert drawn.scope == registry_scope(tmp_path, "leaf")
    assert drawn.seed == 1
    assert [s.location for s in drawn.samples] == [str(tmp_path.resolve() / "images" / "a.jpg")]


def test_read_selection_refuses_an_absent_record_by_name(tmp_path: Path):
    with pytest.raises(ValueError, match="no selection recorded"):
        read_selection(tmp_path / "nothing", project=tmp_path)


def test_read_selection_refuses_a_sample_missing_its_own_ground_truth(tmp_path: Path):
    """A sample naming no ground truth binds nothing: a selection's samples each name their own
    rather than sharing a directory the reader could reconstruct one from."""
    out = tmp_path / "m"
    _write_one_sample_selection(tmp_path, out)
    document = ts.read(selection_key(out))
    document["samples"][0].pop("ground_truth")
    ts.replace(selection_key(out), document)

    with pytest.raises(ValueError, match=r"carries no \['ground_truth'\]"):
        read_selection(out, project=tmp_path)


def test_read_selection_refuses_an_empty_sample_list(tmp_path: Path):
    out = tmp_path / "m"
    _write_one_sample_selection(tmp_path, out)
    document = ts.read(selection_key(out))
    document["samples"] = []
    ts.replace(selection_key(out), document)

    with pytest.raises(ValueError, match="lists no samples"):
        read_selection(out, project=tmp_path)


def test_read_selection_refuses_one_source_on_two_sides(tmp_path: Path):
    """The same pixels on train and calibration would be trained on and measured on at once."""
    out = tmp_path / "m"
    _write_one_sample_selection(tmp_path, out)
    document = ts.read(selection_key(out))
    document["samples"].append(
        {"member": "a", "source": "images/a.jpg", "ground_truth": "annotations/a.json",
         "group": "b", "side": "calibration"})
    ts.replace(selection_key(out), document)

    with pytest.raises(ValueError, match="on more than one side"):
        read_selection(out, project=tmp_path)


def test_read_selection_refuses_one_group_on_two_sides(tmp_path: Path):
    """Crops of one parent, or captures of one subject, share a group key: splitting them across
    sides leaks one side into the other, so the reader refuses the partition outright."""
    out = tmp_path / "m"
    _write_one_sample_selection(tmp_path, out)
    document = ts.read(selection_key(out))
    document["samples"].append(
        {"member": "a_0_1", "source": "images/a_0_1.jpg",
         "ground_truth": "annotations/a_0_1.json", "group": "a",
         "side": "val"})
    ts.replace(selection_key(out), document)

    with pytest.raises(ValueError, match="group"):
        read_selection(out, project=tmp_path)
