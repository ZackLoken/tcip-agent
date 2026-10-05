"""One read of each document, each bucket record and each ground-truth table per act: the
admission's read is the one its loaders, their targets, their class counts, every item and the
assessment's retained copy consume, and the scoring's read is the one its renderers draw."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


_READ_STORES = ("label_documents", "prediction_documents", "prediction_buckets")
"""The stores whose records hold ground truth, predictions and the buckets naming them."""


def _counting_reads(monkeypatch, files: tuple[str, ...] = ()) -> Counter:
    """Every read at the store boundary of a record of :data:`_READ_STORES`, and every open for
    reading of a file whose suffix is among ``files`` by whatever route opens it, counted by what
    was read."""
    import builtins
    import io

    from tcip_store.sqlite_backend import SqliteBackend

    counts: Counter = Counter()
    real_record = SqliteBackend.read_versioned

    def counted_record(self, key, **default):
        if key.store in _READ_STORES:
            counts[str(key)] += 1
        return real_record(self, key, **default)

    monkeypatch.setattr(SqliteBackend, "read_versioned", counted_record)
    real_open = builtins.open

    def counted_open(file, mode="r", *args, **kwargs):
        if (isinstance(file, (str, Path)) and mode.startswith("r")
                and Path(file).suffix in files):
            counts[str(Path(file))] += 1
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counted_open)
    monkeypatch.setattr(io, "open", counted_open)
    return counts


def _consume(*datasets) -> None:
    """Every class count and every item of each loader, the reads a run makes of it."""
    for ds in datasets:
        ds.class_distribution
        for i in range(len(ds)):
            ds[i]


def test_a_drawn_detection_run_reads_each_document_once(tmp_path: Path, monkeypatch):
    from tcip_mcp.pipelines.data.split_construction import auto_train_val
    from tests.test_training_autoval import _detection_dataset

    images_dir, stems = _detection_dataset(tmp_path / "ds")
    counts = _counting_reads(monkeypatch)

    train, val, _partition = auto_train_val(tmp_path, "detection", {
        "images_dir": str(images_dir), "scope": {"subject": "bud"},
        "split": {"val_ratio": 0.4, "seed": 1}}, None)
    _consume(train, val)

    assert len(counts) == len(stems)
    assert set(counts.values()) == {1}, counts


def test_a_drawn_table_run_reads_its_table_once(tmp_path: Path, monkeypatch):
    from tcip_mcp.pipelines.data.split_construction import auto_train_val
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path, intensities=[0.1, 0.3, 0.5, 0.7], values=[0.2, 0.6, 1.0, 1.4])
    counts = _counting_reads(monkeypatch, files=(".csv",))

    train, val, _partition = auto_train_val(tmp_path, "regression", {
        "images_dir": str(images_dir), "labels_dir": str(csv_path),
        "split": {"val_ratio": 0.5, "seed": 0}}, None)
    _consume(train, val)

    assert counts == Counter({str(csv_path): 1})


def test_a_written_draw_reads_each_document_once_with_its_fingerprint(tmp_path: Path,
                                                                     monkeypatch):
    from tcip_mcp.tools.data_tools import draw_splits
    from tests.test_training_autoval import _detection_dataset

    images_dir, stems = _detection_dataset(tmp_path / "ds")
    counts = _counting_reads(monkeypatch)

    result = draw_splits(tmp_path, str(tmp_path / "ds"), seed=1, subject="bud", val_ratio=0.25,
                         calibration_ratio=0, holdout_ratio=0, group_by="stem",
                         output_path=str(tmp_path / "selection"))

    assert "error" not in result, result
    assert len(counts) == len(stems)
    assert set(counts.values()) == {1}, counts


def test_a_recorded_table_changed_under_its_digest_refuses_by_name(tmp_path: Path):
    from dataclasses import replace

    from tcip_mcp.pipelines.data.label_queries import acquired, admit
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir, csv_path = write_regression_dataset(tmp_path, intensities=[0.1, 0.3],
                                                    values=[0.2, 0.6])
    admitted = admit(images_dir, str(csv_path)).every_sample()
    recorded = [replace(s, read=None, stored=None) for s in admitted]
    # Admits valid work: a replay of the unchanged table reads what the admission read.
    assert [s.read for s in acquired(recorded)] == [s.read for s in admitted]

    csv_path.write_bytes(csv_path.read_bytes().replace(b"0.6", b"9.9"))
    with pytest.raises(ValueError, match="the version recorded"):
        acquired(recorded)


def test_an_assessment_reads_each_reference_document_once(tmp_path: Path, monkeypatch):
    """The assessment measures the snapshot its own re-admission read, at the version that read
    answered: a document emptied before that re-admission refuses as unannotated, and one emptied
    after it is measured as read, from the copy retained at that version."""
    import tcip_mcp.pipelines.data.label_queries as label_queries
    from tcip_annotation import json_io
    from tcip_mcp.assessment import read_assessment
    from tests._chain_fixtures import IMG, assess, confirm_count_trait
    from tests.test_assessment_guards import _trained

    _images_dir, drawn, selection_dir, checkpoint = _trained(tmp_path, "exp-reads")
    confirm_count_trait(tmp_path)
    reference = sorted((s.ground_truth for s in drawn.samples
                        if s.side in ("calibration", "holdout")), key=str)
    readmitted: dict = {}
    real_readmit = label_queries.readmitted_samples

    def emptied_after_its_read(samples, scope):
        out = real_readmit(samples, scope)
        readmitted.update((s.ground_truth, s.ground_truth_digest) for s in out)
        json_io.write_label_document(reference[0], [], IMG, IMG, keep_empty=True)
        return out

    monkeypatch.setattr(label_queries, "readmitted_samples", emptied_after_its_read)
    counts = _counting_reads(monkeypatch)

    record = assess(tmp_path, checkpoint, selection_dir)

    assert "error" not in record, record
    assert {str(key) for key in reference} <= set(counts), counts
    assert set(counts.values()) == {1}, counts
    retained = read_assessment(tmp_path, record["assessment_id"]).reference.ground_truth
    assert {f.ground_truth: f.digest for f in retained} == {
        key: readmitted[key] for key in reference}
    assert json_io.read_label_document(reference[0]).annotations == []


def test_a_reference_document_emptied_before_the_assessment_reads_it_refuses(tmp_path: Path):
    from tcip_annotation import json_io
    from tests._chain_fixtures import IMG, assess, confirm_count_trait
    from tests.test_assessment_guards import _trained

    _images_dir, drawn, selection_dir, checkpoint = _trained(tmp_path, "exp-emptied")
    confirm_count_trait(tmp_path)
    emptied = next(s.ground_truth for s in drawn.samples if s.side == "holdout")
    json_io.write_label_document(emptied, [], IMG, IMG, keep_empty=True)

    record = assess(tmp_path, checkpoint, selection_dir)

    assert "no longer admissible" in record["error"], record


def test_a_semantic_run_reads_each_mask_once(tmp_path: Path, monkeypatch):
    import numpy as np
    from PIL import Image

    from tcip_mcp.dataset_layout import UNDATED_BUCKET
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    images_dir, masks_dir = tmp_path / "images" / UNDATED_BUCKET, tmp_path / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir()
    for i in range(4):
        Image.new("RGB", (8, 8), (i * 40, 0, 0)).save(images_dir / f"tree{i}.png")
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        Image.fromarray(mask, mode="L").save(masks_dir / f"tree{i}.png")
    counts = _counting_reads(monkeypatch, files=(".png",))

    train, val, _partition = auto_train_val(tmp_path, "semantic_seg", {
        "images_dir": str(images_dir), "labels_dir": str(masks_dir),
        "split": {"val_ratio": 0.5, "seed": 0}}, None)
    _consume(train, val)

    masks = {k: n for k, n in counts.items() if Path(k).parent == masks_dir}
    assert len(masks) == 4 and set(masks.values()) == {1}, counts


def test_a_freeze_reads_each_document_once_with_its_fingerprint(tmp_path: Path, monkeypatch):
    from tcip_mcp.tools.data_tools import freeze_selection
    from tests.test_freeze_selection import _real_drawn_experiment
    from tests.test_selection_binding import _two_subject_two_date_dataset

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(tmp_path, root, "exp-frozen")
    counts = _counting_reads(monkeypatch)

    result = freeze_selection(tmp_path, "exp-frozen")

    assert "error" not in result, result
    assert counts and set(counts.values()) == {1}, counts


def test_a_physical_scale_assessment_reads_its_references_and_table_once(tmp_path: Path,
                                                                         monkeypatch):
    from tests import _trait_fixtures as fx
    from tests.test_physical_scale_assessment import _author_tolerance, _calibrate, _reference

    fx.seed_delivery_traits(tmp_path)
    _author_tolerance(tmp_path)
    selection, csv_path = _reference(tmp_path)
    counts = _counting_reads(monkeypatch, files=(".csv",))

    scale = _calibrate(tmp_path, selection, csv_path)

    assert "error" not in scale, scale
    assert counts[str(csv_path)] == 1
    assert set(counts.values()) == {1}, counts


def test_a_reserved_regions_assessment_reads_its_mosaic_document_once(tmp_path: Path,
                                                                      monkeypatch):
    from tests import _trait_fixtures as fx
    from tests.test_block_calibration import _assess, _attested

    fx.seed_confirmed_count(tmp_path, measured_subject="bud")
    exp = _attested(tmp_path)
    counts = _counting_reads(monkeypatch)

    record = _assess(exp)

    assert "error" not in record, record
    assert counts[str(exp["label"])] == 1
    assert set(counts.values()) == {1}, counts


def _published_frames(root: Path) -> list[Path]:
    """Two labeled frames of one capture under ``root``, each with its predictions published as
    the bucket ``published``; their image paths."""
    from PIL import Image

    from tcip_annotation.state import Annotation, BBox

    from tcip_mcp.dataset_layout import UNDATED_BUCKET
    from tests._chain_fixtures import published
    from tests._producer_fixtures import label_image

    images_dir = root / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    frames = []
    for name in ("leaf_a", "leaf_b"):
        frame = images_dir / f"{name}.png"
        Image.new("RGB", (64, 48), color=(90, 110, 70)).save(frame)
        label_image(frame, [Annotation(subject="leaf", geometry=BBox(8, 8, 24, 24))], 64, 48)
        frames.append(frame)
    published(root, "published", [
        {"image": str(frame), "width": 64, "height": 48, "boxes": [[8, 8, 24, 24], [40, 30, 50, 40]],
         "scores": [0.9, 0.7], "labels": [1, 1]} for frame in frames], scope={"subject": "leaf"})
    return frames


def _counting_scans(monkeypatch) -> Counter:
    """Every scan of an image directory for its logical images, counted by directory."""
    from tcip_mcp.pipelines import image_utils

    scans: Counter = Counter()
    real_scan = image_utils._scan_identities

    def counted_scan(directory):
        scans[str(directory)] += 1
        return real_scan(directory)

    monkeypatch.setattr(image_utils, "_scan_identities", counted_scan)
    return scans


def _counting_resolutions(monkeypatch) -> Counter:
    """Every path resolved to its logical image, counted by path."""
    from tcip_mcp.pipelines import image_utils

    resolutions: Counter = Counter()
    real_resolve = image_utils.resolve_image_paths

    def counted_resolve(paths):
        paths = list(paths)
        resolutions.update(str(Path(p)) for p in paths)
        return real_resolve(paths)

    monkeypatch.setattr(image_utils, "resolve_image_paths", counted_resolve)
    return resolutions


def test_an_assessment_resolves_each_image_once(tmp_path: Path, monkeypatch):
    from tests._chain_fixtures import assess, confirm_count_trait
    from tests.test_assessment_guards import _trained

    _images_dir, drawn, selection_dir, checkpoint = _trained(tmp_path, "exp-resolved")
    confirm_count_trait(tmp_path)
    reference = {str(Path(s.source)) for s in drawn.samples
                 if s.side in ("calibration", "holdout")}
    resolutions = _counting_resolutions(monkeypatch)

    record = assess(tmp_path, checkpoint, selection_dir)

    assert "error" not in record, record
    assert reference <= set(resolutions), resolutions
    assert set(resolutions.values()) == {1}, resolutions


def test_a_selection_bound_run_resolves_each_image_once(tmp_path: Path, monkeypatch):
    from tcip_mcp.pipelines.data.split_construction import auto_train_val
    from tests.test_selection_binding import (
        _draw, _run_data_cfg, _two_subject_two_date_dataset,
    )

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    drawn = _draw(tmp_path, root, tmp_path / "m")
    resolutions = _counting_resolutions(monkeypatch)

    train, val, _partition = auto_train_val(tmp_path, "detection",
                                            _run_data_cfg(root, tmp_path / "m"), None)
    _consume(train, val)

    trained = {str(Path(s.source)) for s in drawn.on("train") + drawn.on("val")}
    assert trained <= set(resolutions), resolutions
    assert set(resolutions.values()) == {1}, resolutions


def test_a_comparison_render_reads_its_documents_and_bucket_once(tmp_path: Path, monkeypatch):
    from tcip_mcp.tools.vision_tools import visualize

    frame, _other = _published_frames(tmp_path)
    counts = _counting_reads(monkeypatch)
    scans = _counting_scans(monkeypatch)

    result = visualize(tmp_path, "comparison", str(frame), bucket="published",
                       conf_threshold=0.0)

    assert "error" not in result, result
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)
    assert len(counts) == 3, counts
    assert set(counts.values()) == {1}, counts
    assert scans == Counter({str(frame.parent): 1}), scans


def test_a_folder_scoring_resolves_its_images_once(tmp_path: Path, monkeypatch):
    from tcip_mcp.tools.annotation_tools import score_predictions

    frames = _published_frames(tmp_path)
    scans = _counting_scans(monkeypatch)

    result = score_predictions(str(frames[0].parent), "published", conf_threshold=0.0)

    assert "error" not in result, result
    assert result["image_count"] == len(frames)
    assert scans == Counter({str(frames[0].parent): 1}), scans


def test_a_failure_render_reads_each_document_and_its_bucket_once(tmp_path: Path, monkeypatch):
    from tcip_mcp.tools.vision_tools import render_failure_cases

    frames = _published_frames(tmp_path)
    counts = _counting_reads(monkeypatch)

    result = render_failure_cases(tmp_path, str(tmp_path), "published")

    assert "error" not in result, result
    assert len(result["case_images"]) == len(frames)
    assert len(counts) == 2 * len(frames) + 1, counts
    assert set(counts.values()) == {1}, counts
