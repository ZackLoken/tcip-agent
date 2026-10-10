"""A published prediction bucket is readable by the reader every delivery uses.

This drives the real writer (``run_inference``'s publication) and the real reader
(``buckets.read_bucket``) against each other, so a record the writer produces but the reader cannot
take back whole is a failure here rather than downstream.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _publish(tmp_path, monkeypatch, **stated) -> dict:
    """``run_inference`` over one dated capture of the dataset ``dataset``, its predictor
    stubbed."""
    from PIL import Image

    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._predictor_fixtures import StubPredictor, install
    from tests._verified_checkpoint_fixtures import project_checkpoint

    images_dir = tmp_path / "dataset" / "images" / "2026-03-01"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (160, 120), color=(70, 90, 110)).save(images_dir / "capture_a.png")
    install(monkeypatch, StubPredictor(width=160, height=120, scores=(0.95,), task="detection",
                                       in_chans=3))
    return run_inference(tmp_path, project_checkpoint(tmp_path), images_dir=str(images_dir),
                         bucket="baseline/2026-03-01", device="cpu", stated=Stated(**stated))


def test_the_record_the_publication_writes_is_the_record_the_reader_takes_back(
    tmp_path, monkeypatch,
):
    from tcip_mcp.buckets import read_bucket

    from tests._verified_checkpoint_fixtures import SAMPLE_CROSS_TILE_NMS, SAMPLE_OVERLAP

    result = _publish(tmp_path, monkeypatch, tile=True, tile_size=64, overlap=SAMPLE_OVERLAP,
                      conf=0.3, cross_tile_nms=SAMPLE_CROSS_TILE_NMS)

    assert "error" not in result, result
    bucket = read_bucket(result["dataset_root"], result["bucket"])
    assert bucket.execution.record() == result["execution"]
    assert (bucket.execution.tile_size, bucket.execution.sources["tile_size"]) == (64, "explicit")
    assert bucket.date == "2026-03-01" and bucket.documents == {"capture_a": "capture_a.png"}
    assert bucket.producer["checkpoint_sha256"] == result["checkpoint_sha256"]
