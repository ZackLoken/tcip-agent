"""Web job lifecycle: memory cap, persistence, and inference cancellation."""

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


def test_load_roundtrips_persist_and_defaults_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_web.jobstore import load, persist

    assert load("inference_jobs") == []  # nothing persisted yet -> clean start
    persist("inference_jobs", [{"job_id": "a", "status": "running"}])
    assert load("inference_jobs") == [{"job_id": "a", "status": "running"}]


def test_inference_rehydrate_marks_dead_jobs_interrupted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_web.jobstore import persist
    from tcip_web.routes import inference

    persist("inference_jobs", [
        {"job_id": "done", "status": "completed", "done": 3, "total": 3,
         "images_dir": "i", "output_dir": "o", "error": None},
        {"job_id": "dead", "status": "running", "done": 1, "total": 5,
         "images_dir": "i", "output_dir": "o", "error": None},
    ])
    inference._jobs.clear()
    try:
        inference.rehydrate()
        jobs = {j["job_id"]: j for j in inference.list_jobs()["jobs"]}
        assert jobs["done"]["status"] == "completed"      # terminal preserved
        assert jobs["dead"]["status"] == "interrupted"    # thread gone -> not resumable
    finally:
        inference._jobs.clear()


def test_tuning_rehydrate_marks_dead_sweeps_interrupted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_web.jobstore import persist
    from tcip_web.routes import tuning

    persist("hpo_sweeps", [
        {"sweep_id": "s_done", "status": "completed", "error": None, "has_result": True},
        {"sweep_id": "s_dead", "status": "running", "error": None, "has_result": False},
    ])
    tuning._sweeps.clear()
    try:
        tuning.rehydrate()
        got = {s["sweep_id"]: s for s in tuning.list_sweeps()["sweeps"]}
        assert got["s_done"]["status"] == "completed"
        assert got["s_dead"]["status"] == "interrupted"
    finally:
        tuning._sweeps.clear()


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
                       output_dir=str(tmp_path / "out"), tile=False, conf=0.25, iou=0.7,
                       slice_hw=(640, 640), overlap=0.2)
    _register(job)

    res = cancel_job("j1")
    assert res["cancel_requested"] is True and job.cancel_event.is_set()
    # Cancelling a job that was never registered is a client-side miss, so it has to reach the
    # browser as a 404 and name the id: any other status reads to the caller as a real outcome.
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as cancel_miss:
        cancel_job("missing")
    assert cancel_miss.value.status_code == 404
    assert "missing" in cancel_miss.value.detail

    _worker(job)  # honors the pre-set cancel
    assert job.status == "cancelled"
    assert job.done == 0

    data = json.loads((tmp_path / ".tcip" / "state" / "inference_jobs.json").read_text())
    assert any(s["job_id"] == "j1" and s["status"] == "cancelled" for s in data)
