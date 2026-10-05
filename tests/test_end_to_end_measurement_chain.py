"""The measurement chain end to end: ingest, train, assess, publish, confirm, deliver.

Three detectors over one flow. The first delivers a CSV whose validated column reads true, and
is the admitting half for the two refusals beside it: the same flow with the reference labels
edited after the assessment refuses, and the same flow with the bucket published twice refuses
the second publish.

The subject is the chain, not the fit: the model is tiny and its predictions are stable, so a
run that stops delivering a validated number says the chain broke rather than that a fit
wandered.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402

from tests._chain_fixtures import (  # noqa: E402
    IMG, STEMS, SUBJECT, draw_reference_selection, object_at, run_the_chain, synthetic_capture,
    train_on,
)


def test_a_drawn_reference_selection_records_each_samples_ground_truth_digest(tmp_path: Path):
    """The fact the edited-reference refusal rests on: a selection records, per sample, the
    digest of the ground truth the draw held out, so a reader can say that document moved since
    without re-reading the draw."""
    root = tmp_path / "ds"
    synthetic_capture(root)
    selection = draw_reference_selection(tmp_path, root, tmp_path / "selection")

    calibration = selection.on("calibration")
    assert calibration, selection.counts()
    assert all(s.ground_truth_digest for s in calibration), [
        (s.ground_truth, s.ground_truth_digest) for s in calibration
    ]


def test_the_draw_and_the_delivery_check_digest_a_ground_truth_the_same_way(tmp_path: Path):
    """The digest the draw records for each sample's ground truth is the one the delivery-time
    check recomputes for an untouched reference."""
    from tcip_mcp.pipelines.data.selection import ground_truth_digest

    root = tmp_path / "ds"
    synthetic_capture(root)
    selection = draw_reference_selection(tmp_path, root, tmp_path / "selection")

    for sample in selection.on("calibration"):
        assert ground_truth_digest(sample.ground_truth) == sample.ground_truth_digest, (
            sample.ground_truth)


def test_the_tiny_detector_trains_and_finds_one_object_per_frame(tmp_path: Path):
    """The model the chain rests on: a real training pass, and predictions stable enough that a
    count measured over them is a fact about the chain rather than about a fit."""
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import Stated, prepare_pass

    root = tmp_path / "ds"
    images_dir = synthetic_capture(root)
    selection_dir = tmp_path / "selection"
    draw_reference_selection(tmp_path, root, selection_dir)

    checkpoint_path = train_on(selection_dir, tmp_path, "exp-chain-train")

    checkpoint = load_registered_checkpoint(checkpoint_path, project=tmp_path)
    p = prepare_pass(checkpoint, Stated(tile=False, conf=0.5), device="cpu")
    results = p.predict([str(images_dir / f"{stem}.png") for stem in STEMS[:3]])

    assert [r["count"] for r in results] == [1, 1, 1], results
    for index, result in enumerate(results):
        x0, y0, size = object_at(index)
        assert result["boxes"][0] == pytest.approx(
            [float(x0), float(y0), float(x0 + size), float(y0 + size)], abs=1.0)


def test_a_reference_document_edited_after_its_read_is_never_measured(tmp_path: Path, monkeypatch):
    """The loaders measure each reference document as the assessment's one read answered it, the
    version it retained: an edit landing after that read is never measured, and the record names
    the version read, not the edit."""
    from tcip_mcp import assessment as assessment_mod
    from tcip_mcp.pipelines.data.selection import ground_truth_digest
    from tests._chain_fixtures import assess, confirm_count_trait

    root = tmp_path / "ds"
    synthetic_capture(root)
    selection_dir = tmp_path / "selection"
    draw_reference_selection(tmp_path, root, selection_dir)
    checkpoint_path = train_on(selection_dir, tmp_path, "exp-chain-moved")
    confirm_count_trait(tmp_path)
    real_retained = assessment_mod._retained
    edited: list = []

    def edited_first(run_dir, samples, reads):
        edited.append(samples[0].ground_truth)
        json_io.write_label_document(samples[0].ground_truth, [], IMG, IMG, keep_empty=True)
        return real_retained(run_dir, samples, reads)

    monkeypatch.setattr(assessment_mod, "_retained", edited_first)
    assessed = assess(tmp_path, checkpoint_path, selection_dir)

    (moved,) = edited
    assert "error" not in assessed, assessed
    recorded = assessment_mod.read_assessment(tmp_path, assessed["assessment_id"])
    (retained,) = [f for f in recorded.reference.ground_truth if f.ground_truth == moved]
    assert retained.digest != ground_truth_digest(moved)


def test_the_assessment_passes_and_the_bucket_published_under_it_names_it(tmp_path: Path):
    """Assess and publish: the assessment fits the operating point on the selection's calibration
    side, measures the count over its holdout side and passes; the bucket published under it runs
    its execution record and names it, and no training run is opened for it."""
    from tcip_mcp.assessment import read_assessment
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.experiments import run_rows

    chain = run_the_chain(tmp_path, experiment_id="exp-chain-publish")

    record = read_assessment(tmp_path, chain.assessment["assessment_id"])
    assert record.passed is True and record.failures == []
    assert record.producer["experiment_id"] == "exp-chain-publish"
    disjointness = record.disjointness
    assert disjointness["holdout_shares_calibration"] == [], disjointness
    for side in ("training", "selection"):
        assert disjointness[side] == {"groups": [], "source_digests": []}, disjointness
    bucket = read_bucket(chain.root, chain.bucket)
    assert bucket.assessment_id == record.assessment_id
    assert (bucket.producer, bucket.execution) == (record.producer, record.execution)
    assert [e.experiment_id for e in run_rows(tmp_path)] == ["exp-chain-publish"]


# -- the three detectors -------------------------------------------------------


def test_the_chain_delivers_a_csv_whose_validated_column_reads_true(tmp_path: Path):
    """The admitting flow, and the half that makes the two refusals below mean something.

    Ingest synthetic images, train a tiny model, publish a bucket, assess it against a reference
    selection, confirm a trait revision, deliver a CSV whose validated column reads true. Nothing
    is edited and nothing is republished, so the delivery stands; the same predictions published
    under no assessment refuse.
    """
    from tests import csv_rows

    from tests import _trait_fixtures as fx

    from tcip_mcp.delivery import read_delivery_events
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts, run_inference

    chain = run_the_chain(tmp_path, experiment_id="exp-chain-delivers")

    out_csv = tmp_path / "per_image_counts.csv"
    delivered = deliver_per_image_counts(tmp_path, str(chain.root), chain.bucket, str(out_csv),
                                         trait=fx.COUNT_TRAIT)

    assert "error" not in delivered, delivered
    assert delivered["validated"] is True
    rows = csv_rows(out_csv)
    assert len(rows) == len(STEMS), len(rows)
    for row in rows:
        assert row["validated"] == "True", row
        assert row["delivery_event_id"] == delivered["delivery_event_id"], row
        assert int(row["detection_count"]) == 1, row
    (event,) = read_delivery_events(tmp_path)
    assert event.buckets[0].assessment_id == chain.assessment["assessment_id"]

    unassessed = "unassessed/2026-01-01"
    published = run_inference(tmp_path, checkpoint_path=chain.checkpoint_path,
                              images_dir=str(chain.images_dir), bucket=unassessed)
    assert "error" not in published, published
    refused_csv = tmp_path / "unassessed_counts.csv"
    refused = deliver_per_image_counts(tmp_path, str(chain.root), unassessed, str(refused_csv),
                                       trait=fx.COUNT_TRAIT)
    assert "no assessment answers" in refused["error"], refused
    assert not refused_csv.exists()


@pytest.mark.parametrize("edit", ["calibration", "holdout", "retained_copy"])
def test_editing_the_reference_labels_after_the_assessment_refuses_the_delivery(
    tmp_path: Path, edit: str,
):
    """The same flow with a reference label edited after the assessment, on either side of the
    reference or in the copy the assessment retained of it: the delivery refuses naming the
    document and writes no CSV.
    """
    import tcip_store

    from tests import _trait_fixtures as fx

    from tcip_mcp.assessment import assessment_dir, read_assessment
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts

    chain = run_the_chain(tmp_path, experiment_id="exp-chain-edited")
    reference = read_assessment(tmp_path, chain.assessment["assessment_id"]).reference
    side = "calibration" if edit == "retained_copy" else edit
    member = next(s for s in reference.samples if s.side == side)
    annotations = [Annotation(subject=SUBJECT, geometry=BBox(1, 1, 9, 9))]
    if edit == "retained_copy":
        retained = next(f for f in reference.ground_truth
                        if f.ground_truth == member.ground_truth)
        copy = assessment_dir(tmp_path, chain.assessment["assessment_id"]) / retained.copy
        copy.write_bytes(tcip_store.encode_record(json_io.document_payload(
            annotations, IMG, IMG)))
    else:
        json_io.write_label_document(member.ground_truth, annotations, IMG, IMG)

    out_csv = tmp_path / "per_image_counts.csv"
    delivered = deliver_per_image_counts(tmp_path, str(chain.root), chain.bucket, str(out_csv),
                                         trait=fx.COUNT_TRAIT)

    assert "error" in delivered, delivered
    assert member.ground_truth.parts[-1] in str(delivered["error"]), delivered["error"]
    assert not out_csv.exists(), "a refused delivery writes no CSV"


def test_publishing_the_same_bucket_twice_refuses_the_second_publish(tmp_path: Path):
    """The same flow with the bucket published twice, through the inference door and through
    ``publish`` itself: each refuses, and the bucket's record and every one of its documents is
    exactly what the first publication left.
    """
    import tcip_store

    from tcip_mcp.buckets import BucketExists, Document, publish, read_bucket
    from tcip_mcp.dataset_layout import PREDICTION_BUCKETS, PREDICTION_DOCUMENTS
    from tcip_mcp.tools.inference_tools import run_inference

    chain = run_the_chain(tmp_path, experiment_id="exp-chain-republish")

    def records() -> dict:
        root = str(chain.root)
        return {key: tcip_store.read_versioned(key).version
                for store in (PREDICTION_BUCKETS, PREDICTION_DOCUMENTS)
                for key in tcip_store.keys(store, root)}

    before = records()
    assert len(before) == len(STEMS) + 1

    republished = run_inference(
        tmp_path, checkpoint_path=chain.checkpoint_path, images_dir=str(chain.images_dir),
        bucket=chain.bucket, assessment_id=chain.assessment["assessment_id"])

    assert "error" in republished, republished
    assert "already" in republished["error"], republished
    assert records() == before, "the refused publish must leave the bucket exactly as it was"

    first = read_bucket(chain.root, chain.bucket)
    documents = [Document(str(chain.images_dir / f"{stem}.png"), {"annotations": []})
                 for stem in STEMS]
    with pytest.raises(BucketExists):
        publish(tmp_path, first.root, chain.bucket, documents, producer=first.producer,
                scope=first.scope,
                execution=first.execution, raster_path=None, raster_identity=None,
                assessment_id=first.assessment_id, actor=None)
    assert records() == before
