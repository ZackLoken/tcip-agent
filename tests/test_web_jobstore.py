"""Phase 4.1b — web job lifecycle: memory cap, persistence, and inference cancellation."""

import json

import pytest


def test_evict_terminal_caps_and_keeps_running():
    from tcip_web.jobstore import evict_terminal

    class J:
        def __init__(self, status):
            self.status = status

    jobs = {f"done{i}": J("completed") for i in range(5)}
    jobs["live"] = J("running")
    evict_terminal(jobs, max_jobs=3)

    assert len(jobs) == 3
    assert "live" in jobs           # running jobs are never evicted
    assert "done0" not in jobs      # oldest terminal evicted first


def test_persist_writes_state_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_web.jobstore import persist
    persist("inference_jobs", [{"job_id": "a", "status": "completed"}])
    data = json.loads((tmp_path / ".tcip" / "state" / "inference_jobs.json").read_text())
    assert data == [{"job_id": "a", "status": "completed"}]


def test_inference_cancel_endpoint_and_worker(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.chdir(tmp_path)
    from PIL import Image

    from tcip_web.routes.inference import InferenceJob, _register, _worker, cancel_job

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (16, 16)).save(images_dir / "img.jpg")
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")

    class FakePredictor:
        def __init__(self, **kw):
            pass

        def predict_batch(self, paths, **kw):
            return [{"boxes": [], "scores": [], "labels": [], "width": 16, "height": 16}]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)

    job = InferenceJob(job_id="j1", checkpoint_path=str(ckpt), images_dir=str(images_dir),
                       output_dir=str(tmp_path / "out"), sahi=False, conf=0.25, iou=0.7,
                       slice_hw=(640, 640), overlap=0.2)
    _register(job)

    res = cancel_job("j1")
    assert res["cancel_requested"] is True and job.cancel_event.is_set()
    with pytest.raises(Exception):
        cancel_job("missing")  # 404 -> HTTPException

    _worker(job)  # honors the pre-set cancel
    assert job.status == "cancelled"
    assert job.done == 0

    data = json.loads((tmp_path / ".tcip" / "state" / "inference_jobs.json").read_text())
    assert any(s["job_id"] == "j1" and s["status"] == "cancelled" for s in data)
