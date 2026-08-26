"""Web job lifecycle: memory cap, persistence, and inference cancellation."""

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


def test_evict_terminal_never_touches_another_roots_jobs():
    """One root's overflow must not push a different root's own jobs out of memory."""
    from tcip_web.jobstore import evict_terminal

    class J:
        def __init__(self, status, root):
            self.status = status
            self.platform_root = root

    jobs = {f"a{i}": J("completed", "root-a") for i in range(5)}
    jobs["b_done"] = J("completed", "root-b")
    evict_terminal(jobs, "root-a", max_jobs=3)

    assert "b_done" in jobs
    assert sum(1 for j in jobs.values() if j.platform_root == "root-a") == 3


def test_persist_grouped_writes_state_that_reads_back(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_web.jobstore import load, persist_grouped
    persist_grouped("inference_jobs", [{"job_id": "a", "status": "completed"}])
    assert load("inference_jobs") == [{"job_id": "a", "status": "completed"}]


def test_load_roundtrips_persist_grouped_and_defaults_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_web.jobstore import load, persist_grouped

    assert load("inference_jobs") == []  # nothing persisted yet -> clean start
    persist_grouped("inference_jobs", [{"job_id": "a", "status": "running"}])
    assert load("inference_jobs") == [{"job_id": "a", "status": "running"}]


def test_persist_grouped_writes_each_root_under_its_own_key(tmp_path, monkeypatch):
    """A snapshot spanning two roots must land two files, not one mixed one."""
    monkeypatch.chdir(tmp_path)
    from tcip_web.jobstore import job_registry_key, persist_grouped
    from tcip_store import read

    here_root = tmp_path.resolve()
    other_root = (tmp_path.parent / "other_root").resolve()
    other_root.mkdir()
    persist_grouped("inference_jobs", [
        {"job_id": "here", "status": "completed", "platform_root": str(here_root)},
        {"job_id": "there", "status": "completed", "platform_root": str(other_root)},
    ])
    assert read(job_registry_key("inference_jobs"), default=[]) == [
        {"job_id": "here", "status": "completed", "platform_root": str(here_root)},
    ]
    assert read(job_registry_key("inference_jobs", root=other_root), default=[]) == [
        {"job_id": "there", "status": "completed", "platform_root": str(other_root)},
    ]


def test_inference_rehydrate_marks_dead_jobs_interrupted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_web.jobstore import persist_grouped
    from tcip_web.routes import inference

    persist_grouped("inference_jobs", [
        {"job_id": "done", "status": "completed", "done": 3, "total": 3,
         "images_dir": "i", "output_dir": "o", "error": None},
        {"job_id": "dead", "status": "running", "done": 1, "total": 5,
         "images_dir": "i", "output_dir": "o", "error": None},
    ])
    inference._jobs.clear()
    try:
        inference.rehydrate_for_current_root()
        jobs = {j["job_id"]: j for j in inference.list_jobs()["jobs"]}
        assert jobs["done"]["status"] == "completed"      # terminal preserved
        assert jobs["dead"]["status"] == "interrupted"    # thread gone -> not resumable
    finally:
        inference._jobs.clear()


def test_tuning_rehydrate_marks_dead_sweeps_interrupted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_web.jobstore import persist_grouped
    from tcip_web.routes import tuning

    persist_grouped("hpo_sweeps", [
        {"sweep_id": "s_done", "status": "completed", "error": None, "has_result": True},
        {"sweep_id": "s_dead", "status": "running", "error": None, "has_result": False},
    ])
    tuning._sweeps.clear()
    try:
        tuning.rehydrate_for_current_root()
        got = {s["sweep_id"]: s for s in tuning.list_sweeps()["sweeps"]}
        assert got["s_done"]["status"] == "completed"
        assert got["s_dead"]["status"] == "interrupted"
    finally:
        tuning._sweeps.clear()


def test_inference_jobs_persist_list_and_rehydrate_per_root_across_a_repin(tmp_path, monkeypatch):
    """Two jobs launched under two roots (a repin between them) persist each under its own
    root and neither under the other's; the list route after the repin answers the second
    root's jobs only; rehydrate after the repin loads only that root's persisted job."""
    from tcip_store import read

    from tcip_mcp import workspace
    from tcip_web.jobstore import job_registry_key
    from tcip_web.routes import inference

    def _job(job_id: str) -> inference.InferenceJob:
        return inference.InferenceJob(
            job_id=job_id, checkpoint_path="c", images_dir="i", output_dir="o",
            tile=False, conf=0.25, iou=0.7, slice_hw=(640, 640), overlap=0.2,
        )

    job_a = _job("a1")
    inference._register(job_a)

    proj_b = workspace.project_path("chestnut_burr_other")
    (proj_b / ".tcip").mkdir(parents=True)
    workspace.set_active_project("chestnut_burr_other")

    job_b = _job("b1")
    inference._register(job_b)

    try:
        assert [j["job_id"] for j in inference.list_jobs()["jobs"]] == ["b1"]

        docs_a = read(job_registry_key("inference_jobs", root=job_a.platform_root), default=[])
        docs_b = read(job_registry_key("inference_jobs", root=job_b.platform_root), default=[])
        assert [d["job_id"] for d in docs_a] == ["a1"]
        assert [d["job_id"] for d in docs_b] == ["b1"]

        inference._jobs.clear()
        inference.rehydrate_for_current_root()
        assert [j["job_id"] for j in inference.list_jobs()["jobs"]] == ["b1"]
    finally:
        inference._jobs.clear()


def test_review_priority_queue_persists_lists_and_rehydrates_per_root_across_a_repin(
    tmp_path, monkeypatch
):
    """The review priority-queue registry gets the same per-root treatment as inference/tuning."""
    from tcip_store import read

    from tcip_mcp import workspace
    from tcip_web.jobstore import job_registry_key
    from tcip_web.routes import review

    def _job(job_id: str) -> review.PriorityQueueJob:
        return review.PriorityQueueJob(
            job_id=job_id, checkpoint_path="c", images_dir="i", dataset_root="d",
        )

    job_a = _job("pq-a1")
    review._pq_register(job_a)

    proj_b = workspace.project_path("chestnut_burr_other")
    (proj_b / ".tcip").mkdir(parents=True)
    workspace.set_active_project("chestnut_burr_other")

    job_b = _job("pq-b1")
    review._pq_register(job_b)

    try:
        assert review._pq_get("pq-a1") is None       # not this root's job
        assert review._pq_get("pq-b1") is not None

        docs_a = read(
            job_registry_key("review_priority_jobs", root=job_a.platform_root), default=[])
        docs_b = read(
            job_registry_key("review_priority_jobs", root=job_b.platform_root), default=[])
        assert [d["job_id"] for d in docs_a] == ["pq-a1"]
        assert [d["job_id"] for d in docs_b] == ["pq-b1"]

        review._pq_jobs.clear()
        review.rehydrate_for_current_root()
        assert set(review._pq_jobs) == {"pq-b1"}
    finally:
        review._pq_jobs.clear()


def test_tuning_sweeps_persist_list_and_rehydrate_per_root_across_a_repin(tmp_path, monkeypatch):
    """The same per-root treatment as inference, for the live HPO registry."""
    from tcip_store import read

    from tcip_mcp import workspace
    from tcip_web.jobstore import job_registry_key
    from tcip_web.routes import tuning

    job_a = tuning.HPOJob(sweep_id="hpo-a1")
    with tuning._lock:
        tuning._sweeps[job_a.sweep_id] = job_a
    tuning._persist()

    proj_b = workspace.project_path("chestnut_burr_other")
    (proj_b / ".tcip").mkdir(parents=True)
    workspace.set_active_project("chestnut_burr_other")

    job_b = tuning.HPOJob(sweep_id="hpo-b1")
    with tuning._lock:
        tuning._sweeps[job_b.sweep_id] = job_b
    tuning._persist()

    try:
        assert [s["sweep_id"] for s in tuning.list_sweeps()["sweeps"]] == ["hpo-b1"]

        docs_a = read(job_registry_key("hpo_sweeps", root=job_a.platform_root), default=[])
        docs_b = read(job_registry_key("hpo_sweeps", root=job_b.platform_root), default=[])
        assert [d["sweep_id"] for d in docs_a] == ["hpo-a1"]
        assert [d["sweep_id"] for d in docs_b] == ["hpo-b1"]

        tuning._sweeps.clear()
        tuning.rehydrate_for_current_root()
        assert set(tuning._sweeps) == {"hpo-b1"}
    finally:
        tuning._sweeps.clear()


def test_inference_cancel_endpoint_and_worker(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.chdir(tmp_path)
    from PIL import Image

    from tcip_web.routes._body_common import EmptyBodyPayload
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

    res = cancel_job("j1", EmptyBodyPayload())
    assert res["cancel_requested"] is True and job.cancel_event.is_set()
    # Cancelling a job that was never registered is a client-side miss, so it has to reach the
    # browser as a 404 and name the id: any other status reads to the caller as a real outcome.
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as cancel_miss:
        cancel_job("missing", EmptyBodyPayload())
    assert cancel_miss.value.status_code == 404
    assert "missing" in cancel_miss.value.detail

    _worker(job)  # honors the pre-set cancel
    assert job.status == "cancelled"
    assert job.done == 0

    from tcip_web.jobstore import load
    data = load("inference_jobs")
    assert any(s["job_id"] == "j1" and s["status"] == "cancelled" for s in data)
