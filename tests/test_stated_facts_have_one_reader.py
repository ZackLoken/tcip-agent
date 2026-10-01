"""A prediction's score, a ground-truth record's content, the target a loader builds from
annotations, a box's stored grid, a stamp write's audit line, a split lock's recorded fields, a
completeness digest's stored grid and a prediction bucket's dataset root: a record that does not
state one of these is refused by name where it is read.
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


def test_every_reader_of_a_buckets_dataset_root_answers_what_dataset_root_of_answers(
    tmp_path: Path, monkeypatch,
):
    """One bucket spelled two ways, absolute and relative through a ``..`` segment, is one bucket
    under one root to the verdict key and to a delivery's single-dataset check; a bucket under no
    dataset root has none to either."""
    from tcip_mcp.buckets import bucket_key_of
    from tcip_mcp.dataset_layout import dataset_root_of as bucket_dataset_root
    from tcip_mcp.subject_registry import distinct_dataset_root

    dataset = tmp_path / "orchard"
    bucket = dataset / "predictions" / "detector" / "2026-05-01"
    bucket.mkdir(parents=True)
    (dataset / "predictions" / "other").mkdir()
    monkeypatch.chdir(tmp_path)
    detour = Path("orchard", "predictions", "other", "..", "detector", "2026-05-01")

    assert (bucket_dataset_root(detour).resolve() == bucket_dataset_root(bucket).resolve()
            == dataset.resolve())
    assert bucket_key_of(detour) == bucket_key_of(bucket) == "predictions/detector/2026-05-01"
    assert distinct_dataset_root([bucket, detour]) == dataset.resolve()

    loose = tmp_path / "scratch" / "run"
    loose.mkdir(parents=True)
    assert bucket_dataset_root(loose) is None
    assert bucket_key_of(loose) == loose.resolve().as_posix()
    assert distinct_dataset_root([loose]) is None
