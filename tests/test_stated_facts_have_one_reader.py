"""Each fact here has one producer and one path its readers share: a prediction's score, a
ground-truth record's content, the target a loader builds from annotations, a box's stored grid,
a stamp write's audit line, a split lock's recorded fields, a completeness digest's stored grid
and a prediction bucket's dataset root. A record that does not state the fact is refused by name where it is read, never given a
stand-in.

Fixtures are written through the platform's own label writer and read back through its reader.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox, Polygon

SUBJECT = "strig"
IMG = 100
BOX = BBox(10.0, 10.0, 30.0, 30.0)


def _read_back(path: Path, anns: list[Annotation]) -> list[Annotation]:
    json_io.write_annotations(path, anns, IMG, IMG)
    return json_io.read_annotations(path)


def test_a_prediction_document_stating_no_score_is_refused_where_it_is_read(tmp_path: Path):
    path = tmp_path / "p.json"
    json_io.write_annotations(path, [Annotation(subject=SUBJECT, geometry=BOX, score=0.8),
                                     Annotation(subject=SUBJECT, geometry=BOX)], IMG, IMG)

    with pytest.raises(json_io.UnreadableLabelDocument, match="record 1 .*'score'"):
        json_io.read_predictions(path)


def test_a_scored_prediction_document_reads_whole(tmp_path: Path):
    path = tmp_path / "p.json"
    json_io.write_annotations(path, [Annotation(subject=SUBJECT, geometry=BOX, score=0.8)],
                              IMG, IMG)

    assert [a.score for a in json_io.read_predictions(path)] == [0.8]


@pytest.mark.parametrize("reader", ["match", "evaluation_record"])
def test_no_reader_stands_in_a_score_for_a_prediction_stating_none(tmp_path: Path, reader: str):
    from tcip_annotation.matching import compute_matches

    from tcip_mcp.pipelines.training.evaluation import records_from_annotation

    gt = _read_back(tmp_path / "g.json", [Annotation(subject=SUBJECT, geometry=BOX)])
    preds = _read_back(tmp_path / "p.json", [Annotation(subject=SUBJECT, geometry=BOX)])

    with pytest.raises(ValueError, match="'score'"):
        if reader == "match":
            compute_matches(gt, preds)
        else:
            records_from_annotation(gt, preds, width=IMG, height=IMG)


def test_a_built_in_engines_candidate_carries_the_score_its_engine_reported():
    from tcip_mcp.pipelines.proposal import neutral_candidate

    raw = {"candidate_id": 0, "bbox": [1.0, 1.0, 2.0, 2.0], "area": 1,
           "rings": [[(1, 1), (2, 1), (2, 2)]], "predicted_iou": 0.7}

    assert neutral_candidate(raw, engine="sam", score_key="predicted_iou",
                             meta_keys=())["score"] == 0.7
    with pytest.raises(KeyError, match="predicted_iou"):
        neutral_candidate({k: v for k, v in raw.items() if k != "predicted_iou"}, engine="sam",
                          score_key="predicted_iou", meta_keys=())


def test_both_content_identities_of_ground_truth_see_its_crowd_flag(tmp_path: Path):
    """The review reference's hash and the holdout's content-overlap identity hash one
    projection of ground truth, so neither can call a crowd region and one object the same."""
    from tcip_mcp.pipelines.feedback.review_calibration import review_reference_hash
    from tcip_mcp.pipelines.operating_point import _record_content_hash
    from tcip_mcp.pipelines.training.evaluation import records_from_annotation

    records = []
    for crowd in (False, True):
        gt = _read_back(tmp_path / f"g{crowd}.json",
                        [Annotation(subject=SUBJECT, geometry=BOX, iscrowd=crowd)])
        record = records_from_annotation(gt, [], width=IMG, height=IMG)[1]
        records.append({**record, "image_id": "a"})

    assert _record_content_hash(records[0]) != _record_content_hash(records[1])
    assert review_reference_hash([records[0]]) != review_reference_hash([records[1]])


def test_the_instance_loader_builds_its_target_from_the_polygons_it_reads(tmp_path: Path):
    """The instance loader's target comes from the one target builder, over the geometry the
    loader declares it reads: a box beside the polygon is no instance and no mask."""
    from PIL import Image

    from tests._producer_fixtures import dataset_over

    images, labels = tmp_path / "images", tmp_path / "annotations"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (IMG, IMG), (40, 40, 40)).save(images / "p0.png")
    json_io.write_annotations(labels / "p0.json", [
        Annotation(subject=SUBJECT, geometry=BOX),
        Annotation(subject=SUBJECT,
                   geometry=Polygon([[(50.0, 50.0), (70.0, 50.0), (70.0, 70.0)]]))], IMG, IMG)

    _image, target = dataset_over("instance_seg", images, labels, subject=SUBJECT)[0]

    assert target["boxes"].tolist() == [[50.0, 50.0, 70.0, 70.0]]
    assert target["labels"].tolist() == [1]
    assert len(target["masks"]) == 1 and int(target["masks"][0].sum()) > 0


def test_the_completeness_digest_reads_a_box_on_the_writers_own_grid(tmp_path: Path):
    """The digest hashes each annotation as the label writer stores it, so a box off the stored
    grid digests the same before it is written as after it is read back."""
    from tcip_mcp.pipelines.reference_grid import reference_cells
    from tcip_mcp.pipelines.region_completeness import cell_annotation_digest

    (cell,) = reference_cells(IMG, IMG, IMG, clamp=True)
    anns = [Annotation(subject=SUBJECT, geometry=BBox(0.005, 10.0, 10.004, 20.0))]

    assert cell_annotation_digest(anns, SUBJECT, cell) == cell_annotation_digest(
        _read_back(tmp_path / "c.json", anns), SUBJECT, cell)


def test_a_review_box_is_put_on_the_stored_grid_only_where_it_has_pixels():
    """A verdict's box scaled to its image lands on the 2-decimal grid its stored ground truth
    lives on; one kept on the unit square (no image dimensions) keeps its full precision."""
    from tcip_mcp.pipelines.feedback.review_calibration import _to_xywh

    assert _to_xywh((1 / 3, 0.5, 0.1, 0.1), (600, 400)) == [170.0, 180.0, 60.0, 40.0]
    assert _to_xywh((1 / 3, 0.5, 0.1, 0.1), None)[0] == 1 / 3 - 0.05


def test_every_stamp_write_leaves_its_one_line_and_a_merge_that_writes_nothing_leaves_none(
        tmp_path: Path):
    """The stamp library records its own write, whichever door called it, under the bucket's
    dataset root; a merge that leaves the stamp as it was is no act."""
    import tcip_store as ts

    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.pipelines.resolution import update_sidecar, write_sidecar

    root = tmp_path / "orchard"
    bucket = root / "predictions" / "m" / "2025-09-14"
    write_sidecar(bucket, {"validated": False, "subject": SUBJECT, "attribute": None})
    assert update_sidecar(bucket, lambda stored: {**stored, "shippable_issues": ["merged"]})
    assert not update_sidecar(bucket, lambda stored: None)

    rows = ts.read_log(audit_log_key(root)).records
    assert [(r["tool"], r["arguments"]["pred_dir"]) for r in rows] == [
        ("stamp_written", str(bucket)), ("stamp_written", str(bucket))]
    assert rows[1]["stamp"]["shippable_issues"] == ["merged"]
    assert ts.read_log(audit_log_key()).records == []


def test_a_split_lock_missing_a_recorded_field_raises_naming_it(tmp_path: Path):
    """A lock is read as its draw stated it: a field it lacks is never read as empty history."""
    import tcip_store as ts

    from tcip_mcp.pipelines.data.splits import (
        cal_holdout_lock_key,
        resolve_locked_cal_holdout_split,
    )

    stems = [f"plot_{i}" for i in range(6)]
    resolve_locked_cal_holdout_split(stems, identity_hash="h", scope_root=tmp_path)
    key = cal_holdout_lock_key("h", scope_root=tmp_path)
    lock = ts.read(key)
    del lock["redraw_history"]
    ts.replace(key, lock)

    with pytest.raises(KeyError, match="redraw_history"):
        resolve_locked_cal_holdout_split(stems, identity_hash="h", scope_root=tmp_path,
                                         force_redraw=True)


def test_every_reader_of_a_buckets_dataset_root_answers_what_bucket_dataset_root_answers(
    tmp_path: Path, monkeypatch,
):
    """One bucket spelled two ways, absolute and relative through a ``..`` segment, is one bucket
    under one root to the verdict key and to a delivery's single-dataset check; a bucket under no
    dataset root has none to either."""
    from tcip_mcp.dataset_layout import bucket_dataset_root
    from tcip_mcp.prediction_buckets import bucket_key_of
    from tcip_mcp.subject_registry import _distinct_dataset_root

    dataset = tmp_path / "orchard"
    bucket = dataset / "predictions" / "detector" / "2026-05-01"
    bucket.mkdir(parents=True)
    (dataset / "predictions" / "other").mkdir()
    monkeypatch.chdir(tmp_path)
    detour = Path("orchard", "predictions", "other", "..", "detector", "2026-05-01")

    assert bucket_dataset_root(detour) == bucket_dataset_root(bucket) == dataset.resolve()
    assert bucket_key_of(detour) == bucket_key_of(bucket) == "predictions/detector/2026-05-01"
    assert _distinct_dataset_root([bucket, detour]) == dataset.resolve()

    loose = tmp_path / "scratch" / "run"
    loose.mkdir(parents=True)
    assert bucket_dataset_root(loose) is None
    assert bucket_key_of(loose) == loose.resolve().as_posix()
    assert _distinct_dataset_root([loose]) is None
