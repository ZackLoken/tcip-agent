"""A published prediction bucket is readable by the reader every delivery uses, and a publication
that dies part way leaves a directory no reader takes for a bucket.

These drive the real writer (``run_inference``'s publication) and the real reader
(``buckets.read_bucket``) against each other, so a record the writer produces but the reader cannot
take back whole is a failure here rather than downstream.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class _OneBoxPredictor:
    task = "detection"
    in_chans = 3

    def predict_batch(self, paths, execution=None, **kw):
        return [{"image": str(p), "width": 160, "height": 120, "boxes": [[10, 10, 30, 30]],
                 "scores": [0.95], "labels": [1], "count": 1, "cap_hit": False} for p in paths]


def _publish(tmp_path, monkeypatch, **stated) -> dict:
    """``run_inference`` over one dated capture of the dataset ``dataset``, its predictor
    stubbed."""
    from PIL import Image

    import tcip_mcp.pipelines.inference.generic_predictor as predictor_mod
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._verified_checkpoint_fixtures import project_checkpoint

    images_dir = tmp_path / "dataset" / "images" / "2026-03-01"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (160, 120), color=(70, 90, 110)).save(images_dir / "capture_a.png")
    monkeypatch.setattr(predictor_mod, "GenericPredictor", lambda *a, **kw: _OneBoxPredictor())
    return run_inference(tmp_path, project_checkpoint(tmp_path), images_dir=str(images_dir),
                         output_dir=str(tmp_path / "dataset" / "predictions" / "baseline"
                                        / "2026-03-01"), device="cpu", stated=Stated(**stated))


def test_the_record_the_publication_writes_is_the_record_the_reader_takes_back(
    tmp_path, monkeypatch,
):
    from tcip_mcp.buckets import read_bucket

    result = _publish(tmp_path, monkeypatch, tile=True, tile_size=64, conf=0.3)

    assert "error" not in result, result
    bucket = read_bucket(result["output_dir"])
    assert bucket.execution.record() == result["execution"]
    assert (bucket.execution.tile_size, bucket.execution.sources["tile_size"]) == (64, "explicit")
    assert bucket.date == "2026-03-01" and bucket.documents == {"capture_a": "capture_a.png"}
    assert bucket.producer["checkpoint_sha256"] == result["checkpoint_sha256"]


def test_a_publication_that_dies_before_its_record_leaves_no_bucket(tmp_path, monkeypatch):
    """Documents written and no ``bucket.json`` is a directory every reader refuses, with the
    publication's failed line naming what it wrote: the safe direction."""
    import tcip_mcp.experiments as experiments
    import tcip_store as ts
    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.buckets import read_bucket
    from tests._verified_checkpoint_fixtures import project_checkpoint

    project_checkpoint(tmp_path)  # made before the record write is broken; answered again after

    def _die(*a, **kw):
        raise RuntimeError("the process died between the documents and the record")

    monkeypatch.setattr(experiments, "write_once", _die)
    with pytest.raises(RuntimeError, match="the process died"):
        _publish(tmp_path, monkeypatch, tile=False)

    bucket = tmp_path / "dataset" / "predictions" / "baseline" / "2026-03-01"
    assert (bucket / "capture_a.json").is_file()
    with pytest.raises(ValueError, match="holds no bucket.json"):
        read_bucket(bucket)
    (line,) = [r for r in ts.read_log(audit_log_key(tmp_path / "dataset")).records
               if r["tool"] == "prediction_bucket_published"]
    assert line["status"] == "failed" and line["arguments"]["written"] == ["capture_a"]
