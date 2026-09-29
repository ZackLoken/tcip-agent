"""Web job lifecycle: the in-memory registry, its memory cap, and inference cancellation."""

import pytest


def test_evict_terminal_caps_and_keeps_running():
    from tcip_web.jobstore import evict_terminal

    class J:
        def __init__(self, status):
            self.status = status
            self.platform_root = "root-a"

    jobs = {f"done{i}": J("completed") for i in range(5)}
    jobs["live"] = J("running")
    evict_terminal(jobs, "root-a", max_jobs=3)

    assert len(jobs) == 3
    assert "live" in jobs           # running jobs are never evicted
    assert "done0" not in jobs      # oldest terminal evicted first


def test_evict_terminal_prefers_the_overflowing_roots_own_oldest_jobs():
    """One root's own overflow is trimmed first; the whole dict is then trimmed to max_jobs
    too (oldest terminal job of any root), so a lone recent job from a different root survives
    as long as an older job from the overflowing root is still there to take its place."""
    from tcip_web.jobstore import evict_terminal

    class J:
        def __init__(self, status, root):
            self.status = status
            self.platform_root = root

    jobs = {f"a{i}": J("completed", "root-a") for i in range(5)}
    jobs["b_done"] = J("completed", "root-b")
    evict_terminal(jobs, "root-a", max_jobs=3)

    assert "b_done" in jobs
    assert sum(1 for j in jobs.values() if j.platform_root == "root-a") == 2


def test_evict_terminal_bounds_the_whole_dict_across_roots():
    """The whole dict stays bounded at max_jobs even as more roots register their own jobs,
    not just each root's own share: a root that has stopped receiving launches is trimmed too,
    the leak this helper exists to close."""
    from tcip_web.jobstore import evict_terminal

    class J:
        def __init__(self, status, root):
            self.status = status
            self.platform_root = root

    jobs: dict[str, J] = {}
    for i in range(7):
        jobs[f"a{i}"] = J("completed", "root-a")
        evict_terminal(jobs, "root-a", max_jobs=5)
    for i in range(7):
        jobs[f"b{i}"] = J("completed", "root-b")
        evict_terminal(jobs, "root-b", max_jobs=5)

    assert len(jobs) <= 5
    assert not any(j.platform_root == "root-a" for j in jobs.values())


def test_job_registry_registers_gets_lists_by_root_and_finds_or_registers():
    """The one dict-plus-lock registry every route adopts: a job registered under a root is got
    by id from any root, listed only under its own, and found again rather than made twice."""
    from tcip_web.jobstore import JobRegistry

    class J:
        def __init__(self, job_id, root):
            self.job_id = job_id
            self.status = "pending"
            self.platform_root = root

    registry = JobRegistry()
    job = J("j1", "root-a")
    registry.register(job.job_id, job, job_root=job.platform_root)

    assert registry.get("j1") is job
    assert registry.list("root-a") == [job]
    assert registry.list("root-b") == []

    found, created = registry.find_or_register(lambda j: j.job_id == "j1",
                                               lambda: J("j2", "root-a"))
    assert (found, created) == (job, False)
    made, created = registry.find_or_register(lambda j: j.job_id == "j3",
                                              lambda: J("j3", "root-b"), job_root="root-b")
    assert created is True and registry.get("j3") is made


def _fake_predictor(monkeypatch) -> None:
    class FakePredictor:
        def __init__(self, checkpoint_path=None, **kw):
            pass

        def predict_batch(self, paths, **kw):
            return [{"image": p, "boxes": [], "scores": [], "labels": [], "width": 16,
                     "height": 16} for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)


def _one_image(tmp_path):
    from PIL import Image

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (16, 16)).save(images_dir / "img.jpg")
    return images_dir


def test_inference_cancel_endpoint_and_worker(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.chdir(tmp_path)
    from fastapi import HTTPException

    from tcip_web.routes._body_common import EmptyBodyPayload
    from tcip_web.routes.inference import InferenceJob, _register, _worker, cancel_job
    from tests._verified_checkpoint_fixtures import project_checkpoint

    images_dir = _one_image(tmp_path)
    _fake_predictor(monkeypatch)

    job = InferenceJob(job_id="j1", checkpoint_path=project_checkpoint(),
                       images_dir=str(images_dir), output_dir=str(tmp_path / "out"), tile=False,
                       conf=0.25, cross_tile_nms=0.7, overlap=0.2)
    _register(job)

    res = cancel_job("j1", EmptyBodyPayload())
    assert res["cancel_requested"] is True and job.cancel_event.is_set()
    # Canceling a job that was never registered is a client-side miss, so it has to reach the
    # browser as a 404 and name the id: any other status reads to the caller as a real outcome.
    with pytest.raises(HTTPException) as cancel_miss:
        cancel_job("missing", EmptyBodyPayload())
    assert cancel_miss.value.status_code == 404
    assert "missing" in cancel_miss.value.detail

    _worker(job)  # honors the pre-set cancel
    assert job.status == "canceled"
    assert job.done == 0


def test_inference_worker_sets_audit_warning_on_a_lost_audit_line(tmp_path, monkeypatch):
    """The run's own predictions land regardless; a failed append must not vanish as a silent
    warning, and it must not change the run's own terminal status either."""
    pytest.importorskip("fastapi")
    monkeypatch.chdir(tmp_path)
    import tcip_mcp.audit as audit_module
    from tcip_web.routes import inference
    from tcip_web.routes.inference import InferenceJob, _register, _worker
    from tests._verified_checkpoint_fixtures import project_checkpoint

    images_dir = _one_image(tmp_path)
    ckpt = project_checkpoint()
    _fake_predictor(monkeypatch)

    def _refuse_append(*args: object, **kwargs: object) -> None:
        raise RuntimeError("audit log unwritable")

    monkeypatch.setattr(audit_module, "append", _refuse_append)

    output_dir = tmp_path / "ds" / "predictions" / "model" / "2026-01-01"
    job = InferenceJob(job_id="j-audit", checkpoint_path=ckpt, images_dir=str(images_dir),
                       output_dir=str(output_dir), tile=False, conf=0.25, cross_tile_nms=0.7,
                       overlap=0.2)
    _register(job)

    _worker(job)
    served = {j["job_id"]: j for j in inference.list_jobs()["jobs"]}.get("j-audit", {})
    assert served.get("status") == "completed"
    warning = served.get("audit_warning")
    assert warning is not None
    assert "stamp_written" in warning
    assert (output_dir / "img.json").exists()


def test_inference_worker_healthy_run_serves_audit_warning_none(tmp_path, monkeypatch):
    """Coverage: a run whose own audit line lands carries no gap on the served body."""
    pytest.importorskip("fastapi")
    monkeypatch.chdir(tmp_path)
    from tcip_web.routes import inference
    from tcip_web.routes.inference import InferenceJob, _register, _worker
    from tests._verified_checkpoint_fixtures import project_checkpoint

    images_dir = _one_image(tmp_path)
    _fake_predictor(monkeypatch)

    output_dir = tmp_path / "ds" / "predictions" / "model" / "2026-01-01"
    job = InferenceJob(job_id="j-healthy", checkpoint_path=project_checkpoint(),
                       images_dir=str(images_dir), output_dir=str(output_dir), tile=False,
                       conf=0.25, cross_tile_nms=0.7, overlap=0.2)
    _register(job)

    _worker(job)
    served = {j["job_id"]: j for j in inference.list_jobs()["jobs"]}.get("j-healthy", {})
    assert served.get("status") == "completed"
    assert served.get("audit_warning") is None


def test_inference_stream_final_frame_never_precedes_the_audit_attempt(tmp_path, monkeypatch):
    """The worker's terminal status must not become visible to the stream before the audit
    attempt for this run resolves: a frame that read the status while the append was still in
    flight would carry a terminal status with no ``audit_warning`` yet, and the stream closes on
    any terminal status, so that frame would be the last one the client ever sees."""
    pytest.importorskip("fastapi")
    monkeypatch.chdir(tmp_path)
    import threading

    from fastapi.testclient import TestClient

    import tcip_mcp.audit as audit_module
    from tcip_web.app import app
    from tcip_web.routes.inference import InferenceJob, _register, _worker
    from tests._verified_checkpoint_fixtures import project_checkpoint

    images_dir = _one_image(tmp_path)
    ckpt = project_checkpoint()
    _fake_predictor(monkeypatch)

    about_to_append = threading.Event()
    release_append = threading.Event()

    def _blocking_refusal(*args: object, **kwargs: object) -> None:
        about_to_append.set()
        release_append.wait(10)
        raise RuntimeError("audit log unwritable")

    monkeypatch.setattr(audit_module, "append", _blocking_refusal)

    output_dir = tmp_path / "ds" / "predictions" / "model" / "2026-01-01"
    job = InferenceJob(job_id="j-stream-order", checkpoint_path=ckpt,
                       images_dir=str(images_dir), output_dir=str(output_dir), tile=False,
                       conf=0.25, cross_tile_nms=0.7, overlap=0.2)
    _register(job)

    worker_thread = threading.Thread(target=_worker, args=(job,))
    worker_thread.start()
    try:
        assert about_to_append.wait(10)
        assert job.status == "running"

        client = TestClient(app, base_url="http://127.0.0.1")
        with client.websocket_connect(
            f"ws://127.0.0.1/api/inference/jobs/{job.job_id}/stream"
        ) as ws:
            first = ws.receive_json()
            assert first["type"] == "progress"
            assert first["status"] == "running"
            release_append.set()
            frame = None
            for _ in range(50):
                frame = ws.receive_json()
                if frame["type"] == "final":
                    break
                assert frame["status"] == "running"
            assert frame is not None and frame["type"] == "final"
            assert frame["audit_warning"] is not None
            assert "stamp_written" in frame["audit_warning"]
    finally:
        release_append.set()
        worker_thread.join(10)


def test_inference_cancel_reaches_a_job_launched_under_a_previous_root(tmp_path, monkeypatch):
    """Canceling a run one launched is legitimate work: a repin to another project must not
    make the job invisible to cancel or stream, only to the list route."""
    from fastapi import HTTPException

    from tcip_mcp import workspace
    from tcip_web.routes._body_common import EmptyBodyPayload
    from tcip_web.routes.inference import InferenceJob, _get, _register, _registry, cancel_job

    job = InferenceJob(
        job_id="launched-under-a", checkpoint_path="c", images_dir="i", output_dir="o",
        tile=False, conf=0.25, cross_tile_nms=0.7, overlap=0.2,
    )
    _register(job)

    try:
        proj_b = workspace.project_path("chestnut_burr_other")
        (proj_b / ".tcip").mkdir(parents=True)
        workspace.activate_project("chestnut_burr_other")

        assert _get("launched-under-a") is job

        res = cancel_job("launched-under-a", EmptyBodyPayload())
        assert res["cancel_requested"] is True
        assert job.cancel_event.is_set()

        with pytest.raises(HTTPException) as miss:
            cancel_job("never-launched", EmptyBodyPayload())
        assert miss.value.status_code == 404
    finally:
        _registry.jobs.clear()


def test_priority_queue_by_id_reaches_a_job_launched_under_a_previous_root(tmp_path, monkeypatch):
    """Answering a ranked queue one launched is legitimate work, the same contract inference
    already holds: a repin to another project must not make the job invisible by id, only to
    the list route (which has none of its own for the priority queue)."""
    from fastapi import HTTPException

    from tcip_mcp import workspace
    from tcip_web.routes.review import (
        PriorityQueueJob, _pq_get, _pq_register, _pq_registry, get_priority_queue_job,
    )

    job = PriorityQueueJob(
        job_id="pq-under-a", checkpoint_path="c", images_dir="i", dataset_root="d",
        status="completed", queue=[{"image": "a.jpg", "score": 0.9}],
    )
    _pq_register(job)

    try:
        proj_b = workspace.project_path("chestnut_burr_other")
        (proj_b / ".tcip").mkdir(parents=True)
        workspace.activate_project("chestnut_burr_other")

        assert _pq_get("pq-under-a") is job
        assert get_priority_queue_job("pq-under-a")["job_id"] == "pq-under-a"

        with pytest.raises(HTTPException) as miss:
            get_priority_queue_job("pq-never-launched")
        assert miss.value.status_code == 404
    finally:
        _pq_registry.jobs.clear()
