"""A prediction bucket cannot vouch for itself.

A bucket's record names the assessment it was published under, and a delivery reads that
assessment and compares what it measured (the checkpoint, the execution record, the captures of
its reference) against what the bucket states. A record edited after publication, one naming an
assessment nobody ran, and a capture the reference never covered each refuse with the sentence
naming why; the untouched flow beside them delivers validated
(``tests/test_end_to_end_measurement_chain.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from tests._chain_fixtures import run_the_chain, synthetic_capture  # noqa: E402


def _deliver(project: Path, bucket: Path, out: Path) -> dict:
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts
    from tests import _trait_fixtures as fx

    return deliver_per_image_counts(project, predictions_dir=str(bucket), output_path=str(out),
                                    trait=fx.COUNT_TRAIT)


def _edit_record(bucket: Path, edit) -> None:
    """Rewrite ``bucket``'s record on disk through ``edit``, the way a hand edit after publication
    reaches it."""
    path = bucket / "bucket.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    edit(record)
    path.write_text(json.dumps(record), encoding="utf-8")


def test_a_bucket_record_edited_to_another_execution_answers_for_nothing(tmp_path: Path):
    """The assessment measured one execution record; a bucket stating another, under the same
    assessment id, was not produced by what the assessment measured."""
    chain = run_the_chain(tmp_path, experiment_id="exp-binding-execution")
    assert _deliver(tmp_path, chain.bucket, tmp_path / "before.csv")["validated"] is True

    _edit_record(chain.bucket, lambda record: record["execution"].update(conf=0.05))

    out = tmp_path / "after.csv"
    delivered = _deliver(tmp_path, chain.bucket, out)
    assert "was not produced by the checkpoint and execution record" in delivered["error"]
    assert chain.assessment["assessment_id"] in delivered["error"]
    assert not out.exists()


def test_a_bucket_record_naming_an_assessment_nobody_ran_answers_for_nothing(tmp_path: Path):
    chain = run_the_chain(tmp_path, experiment_id="exp-binding-forged-id")

    _edit_record(chain.bucket, lambda record: record.update(assessment_id="assessment-never-run"))

    out = tmp_path / "counts.csv"
    delivered = _deliver(tmp_path, chain.bucket, out)
    assert "no assessment 'assessment-never-run' is recorded" in delivered["error"]
    assert not out.exists()


def test_a_capture_outside_the_assessments_reference_answers_for_nothing(tmp_path: Path):
    """The checkpoint and execution are the assessed ones, but the capture is not among those the
    reference held out, so the assessment says nothing about it."""
    from tcip_mcp.tools.inference_tools import run_inference

    chain = run_the_chain(tmp_path, experiment_id="exp-binding-coverage")
    other_images, _labels = synthetic_capture(chain.root, date="2-25-26")
    bucket = chain.root / "predictions" / "chain" / "2-25-26"
    published = run_inference(tmp_path, checkpoint_path=chain.checkpoint_path,
                              images_dir=str(other_images), output_dir=str(bucket),
                              assessment_id=chain.assessment["assessment_id"])
    assert "error" not in published, published

    out = tmp_path / "counts.csv"
    delivered = _deliver(tmp_path, bucket, out)
    assert "reference does not cover" in delivered["error"]
    assert "2-25-26" in delivered["error"]
    assert not out.exists()
