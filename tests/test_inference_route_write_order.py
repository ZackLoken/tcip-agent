"""The GUI inference worker's bucket never carries a provenance stamp for predictions that are
not on disk: the stamp lands after the pass, the order both export doors already write in."""

import json
from pathlib import Path

import pytest


def _two_images(tmp_path):
    from PIL import Image

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for stem in ("a", "b"):
        Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / f"{stem}.jpg")
    return images_dir


class _FakePredictor:
    """A detector that returns one detection per image, enough for a real prediction write."""

    def __init__(self, checkpoint_path=None, **kwargs):
        pass

    def predict_batch(self, paths, tile=False, tile_size=224, overlap=0.2, **kw):
        return [{"image": p, "width": 100, "height": 100,
                 "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1}
                for p in paths]


def _job(job_id, images_dir, out_dir, ckpt, platform_root):
    from tcip_web.routes.inference import InferenceJob

    return InferenceJob(
        job_id=job_id, checkpoint_path=str(ckpt), images_dir=str(images_dir),
        output_dir=str(out_dir), tile=False, conf=0.25, iou=0.7,
        slice_hw=(640, 640), overlap=0.2, postprocess="nms",
        platform_root=str(platform_root),
    )


def test_worker_leaves_no_sidecar_when_a_prediction_write_fails_partway(tmp_path, monkeypatch):
    """A pass that dies between images leaves the predictions it managed to write and no stamp:
    an operating_point.json beside a half-filled bucket reads as provenance for counts that were
    never produced."""
    pytest.importorskip("fastapi")
    from tcip_mcp.pipelines.postprocessing import export
    from tcip_web.routes.inference import _worker
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images_dir = _two_images(tmp_path)
    out_dir = tmp_path / "out"
    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", _FakePredictor)

    real_write = export.write_predictions_json
    calls = []

    def failing_write(json_path, result, **kwargs):
        calls.append(json_path)
        if len(calls) > 1:
            raise OSError("disk full")
        return real_write(json_path, result, **kwargs)

    monkeypatch.setattr(export, "write_predictions_json", failing_write)

    job = _job("write-fails-partway", images_dir, out_dir, ckpt, tmp_path)
    _worker(job)

    assert job.status == "failed"
    assert len(calls) == 2                            # the run really did die inside the pass
    assert (out_dir / "a.json").is_file()             # the prediction written before the failure
    assert not (out_dir / "b.json").exists()
    assert not (out_dir / "operating_point.json").exists()


def test_worker_writes_every_prediction_file_and_the_sidecar_on_a_full_pass(tmp_path, monkeypatch):
    """The ordering costs a completed run nothing: every prediction file and the stamp that
    certifies them are all on disk, in the order the pass walked the images, and the job still
    reports what it reported before."""
    pytest.importorskip("fastapi")
    from tcip_mcp.pipelines.postprocessing import export
    from tcip_web.routes.inference import _summary, _worker
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images_dir = _two_images(tmp_path)
    out_dir = tmp_path / "out"
    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path, filename="m.pt")

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", _FakePredictor)

    real_write = export.write_predictions_json
    written = []

    def recording_write(json_path, result, **kwargs):
        written.append(json_path)
        return real_write(json_path, result, **kwargs)

    monkeypatch.setattr(export, "write_predictions_json", recording_write)

    job = _job("full-pass", images_dir, out_dir, ckpt, tmp_path)
    _worker(job)

    assert job.status == "completed"
    assert job.done == 2 and job.total == 2
    assert job.error is None
    for stem in ("a", "b"):
        assert json.loads((out_dir / f"{stem}.json").read_text())["annotations"][0]["score"] \
            == pytest.approx(0.9)

    import tcip_store as ts
    from tcip_mcp.pipelines.resolution import sidecar_key
    sidecar = ts.read(sidecar_key(out_dir))
    assert sidecar["operating_point"]["conf"]["value"] == pytest.approx(0.25)
    assert sidecar["checkpoint"] == "m"
    assert sidecar["checkpoint_sha256"] and sidecar["produced_at"]
    assert sidecar["images_dir"] == str(images_dir)
    assert sidecar["validated"] is False
    assert sidecar["image_filenames"] == {"a": "a.jpg", "b": "b.jpg"}

    assert set(_summary(job)) == {"job_id", "status", "done", "total", "images_dir", "output_dir",
                                 "error", "warning", "dropped_nonpositive_boxes", "platform_root"}
    assert [Path(p).stem for p in written] == ["a", "b"]
