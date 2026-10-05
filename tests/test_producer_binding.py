"""A prediction bucket cannot vouch for itself.

A bucket's record names the assessment it was published under, and a delivery reads that
assessment and compares what it measured (the checkpoint, the execution record, the captures of
its reference) against what the bucket states. A record edited after publication, one naming an
assessment nobody ran, and a capture the reference never covered each refuse with the sentence
naming why; the untouched flow beside them delivers validated
(``tests/test_end_to_end_measurement_chain.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from tests._chain_fixtures import run_the_chain, synthetic_capture  # noqa: E402


def _deliver(project: Path, root: Path, bucket: str, out: Path) -> dict:
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts
    from tests import _trait_fixtures as fx

    return deliver_per_image_counts(project, str(root), bucket, str(out), trait=fx.COUNT_TRAIT)


def _edit_record(root: Path, bucket: str, edit) -> None:
    """Rewrite ``bucket``'s record through ``edit`` past the publication, the way a hand edit
    after publication reaches it."""
    import tcip_store

    from tcip_mcp.dataset_layout import bucket_key

    key = bucket_key(root, bucket)
    record = tcip_store.read(key)
    edit(record)
    tcip_store.replace(key, record)


def test_a_bucket_record_edited_to_another_execution_answers_for_nothing(tmp_path: Path):
    """The assessment measured one execution record; a bucket stating another, under the same
    assessment id, was not produced by what the assessment measured."""
    chain = run_the_chain(tmp_path, experiment_id="exp-binding-execution")
    assert _deliver(tmp_path, chain.root, chain.bucket,
                    tmp_path / "before.csv")["validated"] is True

    _edit_record(chain.root, chain.bucket, lambda record: record["execution"].update(conf=0.05))

    out = tmp_path / "after.csv"
    delivered = _deliver(tmp_path, chain.root, chain.bucket, out)
    assert "was not produced by the checkpoint and execution record" in delivered["error"]
    assert chain.assessment["assessment_id"] in delivered["error"]
    assert not out.exists()


def test_a_bucket_record_naming_an_assessment_nobody_ran_answers_for_nothing(tmp_path: Path):
    chain = run_the_chain(tmp_path, experiment_id="exp-binding-forged-id")

    _edit_record(chain.root, chain.bucket,
                 lambda record: record.update(assessment_id="assessment-never-run"))

    out = tmp_path / "counts.csv"
    delivered = _deliver(tmp_path, chain.root, chain.bucket, out)
    assert "no assessment 'assessment-never-run' is recorded" in delivered["error"]
    assert not out.exists()


def test_a_capture_outside_the_assessments_reference_answers_for_nothing(tmp_path: Path):
    """The checkpoint and execution are the assessed ones, but the capture is not among those the
    reference held out, so the assessment says nothing about it."""
    from tcip_mcp.tools.inference_tools import run_inference

    chain = run_the_chain(tmp_path, experiment_id="exp-binding-coverage")
    other_images = synthetic_capture(chain.root, date="2-25-26")
    bucket = "chain/2-25-26"
    published = run_inference(tmp_path, checkpoint_path=chain.checkpoint_path,
                              images_dir=str(other_images), bucket=bucket,
                              assessment_id=chain.assessment["assessment_id"])
    assert "error" not in published, published

    out = tmp_path / "counts.csv"
    delivered = _deliver(tmp_path, chain.root, bucket, out)
    assert "reference does not cover" in delivered["error"]
    assert "2-25-26" in delivered["error"]
    assert not out.exists()
