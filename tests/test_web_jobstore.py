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


def test_review_priority_queue_rehydrate_restores_the_persisted_queue(tmp_path, monkeypatch):
    """A completed job's ranked queue is persisted (_pq_summary carries it); a rehydrate must
    restore it rather than leaving the dataclass field's empty default."""
    from tcip_web.routes import review

    job = review.PriorityQueueJob(
        job_id="pq-done", checkpoint_path="c", images_dir="i", dataset_root="d",
        status="completed", queue=[{"image": "a.jpg", "score": 0.9}],
    )
    review._pq_register(job)
    review._pq_jobs.clear()

    try:
        review.rehydrate_for_current_root()
        assert review._pq_jobs["pq-done"].queue == [{"image": "a.jpg", "score": 0.9}]
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


def test_inference_cancel_reaches_a_job_launched_under_a_previous_root(tmp_path, monkeypatch):
    """Cancelling a run one launched is legitimate work: a repin to another project must not
    make the job invisible to preview, cancel or stream, only to the list route."""
    from fastapi import HTTPException

    from tcip_mcp import workspace
    from tcip_web.routes._body_common import EmptyBodyPayload
    from tcip_web.routes.inference import InferenceJob, _get, _jobs, _register, cancel_job

    job = InferenceJob(
        job_id="launched-under-a", checkpoint_path="c", images_dir="i", output_dir="o",
        tile=False, conf=0.25, iou=0.7, slice_hw=(640, 640), overlap=0.2,
    )
    _register(job)

    try:
        proj_b = workspace.project_path("chestnut_burr_other")
        (proj_b / ".tcip").mkdir(parents=True)
        workspace.set_active_project("chestnut_burr_other")

        assert _get("launched-under-a") is job

        res = cancel_job("launched-under-a", EmptyBodyPayload())
        assert res["cancel_requested"] is True
        assert job.cancel_event.is_set()

        with pytest.raises(HTTPException) as miss:
            cancel_job("never-launched", EmptyBodyPayload())
        assert miss.value.status_code == 404
    finally:
        _jobs.clear()


def test_rehydrate_never_displaces_a_job_still_live_from_another_root(tmp_path, monkeypatch):
    """The merge every rehydrate performs (job id already live -> skip) must not overwrite a
    job that is still running under a different root with the interrupted record its own
    persisted file carries."""
    from tcip_mcp import workspace
    from tcip_web.routes import inference

    job_a = inference.InferenceJob(
        job_id="live-a", checkpoint_path="c", images_dir="i", output_dir="o",
        tile=False, conf=0.25, iou=0.7, slice_hw=(640, 640), overlap=0.2,
    )
    job_a.status = "running"
    job_a.done, job_a.total = 2, 5
    inference._register(job_a)

    try:
        proj_b = workspace.project_path("chestnut_burr_other")
        (proj_b / ".tcip").mkdir(parents=True)
        workspace.set_active_project("chestnut_burr_other")

        job_b = inference.InferenceJob(
            job_id="done-b", checkpoint_path="c", images_dir="i", output_dir="o",
            tile=False, conf=0.25, iou=0.7, slice_hw=(640, 640), overlap=0.2,
        )
        job_b.status = "completed"
        inference._register(job_b)

        # job_a is never cleared from _jobs: it is still live in memory when the rehydrate
        # for root B's own registry runs, the shape a repin takes in the running process.
        inference.rehydrate_for_current_root()

        assert inference._jobs["live-a"] is job_a
        assert job_a.status == "running"
        assert job_a.done == 2 and job_a.total == 5
    finally:
        inference._jobs.clear()
