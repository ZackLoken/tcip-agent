"""Web job lifecycle: memory cap, persistence, and inference cancellation."""

from pathlib import Path

import pytest


def test_job_registry_documents_each_match_the_job_registry_claim():
    """tcip-store cannot import tcip-web, so the ``job_registry`` claim in
    ``tcip_store.layout_claims`` cannot enumerate ``JOB_REGISTRY_DOCUMENTS`` itself; this test
    holds the agreement from this side, so a document added to the tuple without a matching
    claim template fails here rather than going unclaimed by the conform rail."""
    from tcip_store.layout_claims import PLATFORM_CLAIMS, matches_template
    from tcip_web.jobstore import JOB_REGISTRY_DOCUMENTS

    templates = PLATFORM_CLAIMS["job_registry"].templates
    for name in JOB_REGISTRY_DOCUMENTS:
        segments = (".tcip", "state", f"{name}.json")
        assert any(matches_template(t, segments) for t in templates), name


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
    root = str(tmp_path.resolve())
    persist_grouped("inference_jobs", [{"job_id": "a", "status": "completed", "platform_root": root}])
    assert load("inference_jobs") == [{"job_id": "a", "status": "completed", "platform_root": root}]


def test_persist_grouped_refuses_a_summary_carrying_no_platform_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_web.jobstore import persist_grouped

    with pytest.raises(ValueError, match="no operator door"):
        persist_grouped("inference_jobs", [{"job_id": "a", "status": "completed"}])


def test_load_roundtrips_persist_grouped_and_defaults_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_web.jobstore import load, persist_grouped
    root = str(tmp_path.resolve())

    assert load("inference_jobs") == []  # nothing persisted yet -> clean start
    persist_grouped("inference_jobs", [{"job_id": "a", "status": "running", "platform_root": root}])
    assert load("inference_jobs") == [{"job_id": "a", "status": "running", "platform_root": root}]


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

    root = str(tmp_path.resolve())
    persist_grouped("inference_jobs", [
        {"job_id": "done", "status": "completed", "done": 3, "total": 3,
         "images_dir": "i", "output_dir": "o", "error": None, "platform_root": root},
        {"job_id": "dead", "status": "running", "done": 1, "total": 5,
         "images_dir": "i", "output_dir": "o", "error": None, "platform_root": root},
    ])
    inference._registry.jobs.clear()
    try:
        inference.rehydrate_for_current_root()
        jobs = {j["job_id"]: j for j in inference.list_jobs()["jobs"]}
        assert jobs["done"]["status"] == "completed"      # terminal preserved
        assert jobs["dead"]["status"] == "interrupted"    # thread gone -> not resumable
    finally:
        inference._registry.jobs.clear()


def test_tuning_rehydrate_marks_dead_sweeps_interrupted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_web.jobstore import persist_grouped
    from tcip_web.routes import tuning

    root = str(tmp_path.resolve())
    persist_grouped("hpo_sweeps", [
        {"sweep_id": "s_done", "status": "completed", "error": None, "has_result": True,
         "platform_root": root},
        {"sweep_id": "s_dead", "status": "running", "error": None, "has_result": False,
         "platform_root": root},
    ])
    tuning._registry.jobs.clear()
    try:
        tuning.rehydrate_for_current_root()
        got = {s["sweep_id"]: s for s in tuning.list_sweeps()["sweeps"]}
        assert got["s_done"]["status"] == "completed"
        assert got["s_dead"]["status"] == "interrupted"
    finally:
        tuning._registry.jobs.clear()


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
    workspace.activate_project("chestnut_burr_other")

    job_b = _job("b1")
    inference._register(job_b)

    try:
        assert [j["job_id"] for j in inference.list_jobs()["jobs"]] == ["b1"]

        docs_a = read(job_registry_key("inference_jobs", root=job_a.platform_root), default=[])
        docs_b = read(job_registry_key("inference_jobs", root=job_b.platform_root), default=[])
        assert [d["job_id"] for d in docs_a] == ["a1"]
        assert [d["job_id"] for d in docs_b] == ["b1"]

        inference._registry.jobs.clear()
        inference.rehydrate_for_current_root()
        assert [j["job_id"] for j in inference.list_jobs()["jobs"]] == ["b1"]
    finally:
        inference._registry.jobs.clear()


def test_inference_rehydrate_restores_dropped_nonpositive_boxes(tmp_path, monkeypatch):
    """``dropped_nonpositive_boxes`` is written on the persisted row (``_summary``); a restart
    must serve the recorded count back, not the field's own zero default."""
    from tcip_web.routes import inference

    job = inference.InferenceJob(
        job_id="j-dropped", checkpoint_path="c", images_dir="i", output_dir="o",
        tile=False, conf=0.25, iou=0.7, slice_hw=(640, 640), overlap=0.2,
    )
    job.status = "completed"
    job.dropped_boxes = 3
    inference._register(job)

    inference._registry.jobs.clear()
    try:
        inference.rehydrate_for_current_root()
        jobs = {j["job_id"]: j for j in inference.list_jobs()["jobs"]}
        assert jobs["j-dropped"]["dropped_nonpositive_boxes"] == 3
    finally:
        inference._registry.jobs.clear()


def test_review_priority_queue_persists_lists_and_rehydrates_per_root_across_a_repin(
    tmp_path, monkeypatch
):
    """The review priority-queue registry persists, lists and rehydrates per root the same way
    inference/tuning do; only its by-id lookup differs from those two, spanning every root this
    process holds, the same contract inference and tuning already hold for theirs."""
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
    workspace.activate_project("chestnut_burr_other")

    job_b = _job("pq-b1")
    review._pq_register(job_b)

    try:
        assert review._pq_get("pq-a1") is job_a      # reachable by id across the repin
        assert review._pq_get("pq-b1") is not None

        docs_a = read(
            job_registry_key("review_priority_jobs", root=job_a.platform_root), default=[])
        docs_b = read(
            job_registry_key("review_priority_jobs", root=job_b.platform_root), default=[])
        assert [d["job_id"] for d in docs_a] == ["pq-a1"]
        assert [d["job_id"] for d in docs_b] == ["pq-b1"]

        review._pq_registry.jobs.clear()
        review.rehydrate_for_current_root()
        assert set(review._pq_registry.jobs) == {"pq-b1"}
    finally:
        review._pq_registry.jobs.clear()


def test_review_priority_queue_rehydrate_restores_the_persisted_queue(tmp_path, monkeypatch):
    """A completed job's ranked queue is persisted (_pq_summary carries it); a rehydrate must
    restore it rather than leaving the dataclass field's empty default."""
    from tcip_web.routes import review

    job = review.PriorityQueueJob(
        job_id="pq-done", checkpoint_path="c", images_dir="i", dataset_root="d",
        status="completed", queue=[{"image": "a.jpg", "score": 0.9}],
    )
    review._pq_register(job)
    review._pq_registry.jobs.clear()

    try:
        review.rehydrate_for_current_root()
        assert review._pq_registry.jobs["pq-done"].queue == [{"image": "a.jpg", "score": 0.9}]
    finally:
        review._pq_registry.jobs.clear()


def test_review_priority_queue_rehydrate_restores_calibration_marks_fields(tmp_path, monkeypatch):
    """A bound run's per-candidate calibration_member marks ride inside the persisted queue
    dicts already (asserted above); a manifest that could not be read instead carries the reason
    on marks_unresolved, its own dataclass field a rehydrate must restore rather than the empty
    default."""
    from tcip_web.routes import review

    job = review.PriorityQueueJob(
        job_id="pq-unresolved", checkpoint_path="c", images_dir="i", dataset_root="d",
        status="completed",
        queue=[{"image": "a.jpg", "score": 0.9}],
        marks_unresolved="this run is bound to split manifest 'nope', but it could not be read",
    )
    review._pq_register(job)
    review._pq_registry.jobs.clear()

    try:
        review.rehydrate_for_current_root()
        restored = review._pq_registry.jobs["pq-unresolved"]
        assert restored.marks_unresolved == (
            "this run is bound to split manifest 'nope', but it could not be read")
    finally:
        review._pq_registry.jobs.clear()


def test_tuning_sweeps_persist_list_and_rehydrate_per_root_across_a_repin(tmp_path, monkeypatch):
    """The same per-root treatment as inference, for the live HPO registry."""
    from tcip_store import read

    from tcip_mcp import workspace
    from tcip_web.jobstore import job_registry_key
    from tcip_web.routes import tuning

    job_a = tuning.HPOJob(sweep_id="hpo-a1")
    with tuning._lock:
        tuning._registry.jobs[job_a.sweep_id] = job_a
    tuning._persist()

    proj_b = workspace.project_path("chestnut_burr_other")
    (proj_b / ".tcip").mkdir(parents=True)
    workspace.activate_project("chestnut_burr_other")

    job_b = tuning.HPOJob(sweep_id="hpo-b1")
    with tuning._lock:
        tuning._registry.jobs[job_b.sweep_id] = job_b
    tuning._persist()

    try:
        assert [s["sweep_id"] for s in tuning.list_sweeps()["sweeps"]] == ["hpo-b1"]

        docs_a = read(job_registry_key("hpo_sweeps", root=job_a.platform_root), default=[])
        docs_b = read(job_registry_key("hpo_sweeps", root=job_b.platform_root), default=[])
        assert [d["sweep_id"] for d in docs_a] == ["hpo-a1"]
        assert [d["sweep_id"] for d in docs_b] == ["hpo-b1"]

        tuning._registry.jobs.clear()
        tuning.rehydrate_for_current_root()
        assert set(tuning._registry.jobs) == {"hpo-b1"}
    finally:
        tuning._registry.jobs.clear()


def test_inference_cancel_endpoint_and_worker(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.chdir(tmp_path)
    from PIL import Image

    from tcip_web.routes._body_common import EmptyBodyPayload
    from tcip_web.routes.inference import InferenceJob, _register, _worker, cancel_job
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (16, 16)).save(images_dir / "img.jpg")
    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)

    class FakePredictor:
        def __init__(self, checkpoint_path=None, **kw):
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
    make the job invisible to cancel or stream, only to the list route."""
    from fastapi import HTTPException

    from tcip_mcp import workspace
    from tcip_web.routes._body_common import EmptyBodyPayload
    from tcip_web.routes.inference import InferenceJob, _get, _register, _registry, cancel_job

    job = InferenceJob(
        job_id="launched-under-a", checkpoint_path="c", images_dir="i", output_dir="o",
        tile=False, conf=0.25, iou=0.7, slice_hw=(640, 640), overlap=0.2,
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


def test_rehydrate_bounds_the_whole_dict_across_every_root_it_adopts(tmp_path, monkeypatch):
    """Adopting three roots that each hold a full persisted registry, without registering a
    single job here, must not grow this process's memory by MAX_JOBS per root: rehydrate
    bounds the dict the same way registering a job already does."""
    from tcip_mcp import workspace
    from tcip_web.jobstore import MAX_JOBS, job_registry_key, persist_to
    from tcip_web.routes import inference

    names = ("root_x", "root_y", "root_z")
    for name in names:
        proj = workspace.project_path(name)
        (proj / ".tcip").mkdir(parents=True)
        summaries = [
            {"job_id": f"{name}-{i}", "status": "completed", "done": 1, "total": 1,
             "images_dir": "i", "output_dir": "o", "error": None,
             "platform_root": str(proj.resolve())}
            for i in range(MAX_JOBS)
        ]
        persist_to(job_registry_key("inference_jobs", root=proj), summaries)

    inference._registry.jobs.clear()
    try:
        for name in names:
            workspace.activate_project(name)
            inference.rehydrate_for_current_root()
        assert len(inference._registry.jobs) <= MAX_JOBS
    finally:
        inference._registry.jobs.clear()


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
        workspace.activate_project("chestnut_burr_other")

        job_b = inference.InferenceJob(
            job_id="done-b", checkpoint_path="c", images_dir="i", output_dir="o",
            tile=False, conf=0.25, iou=0.7, slice_hw=(640, 640), overlap=0.2,
        )
        job_b.status = "completed"
        inference._register(job_b)

        # job_a is never cleared from the registry: still live when root B's rehydrate runs.
        inference.rehydrate_for_current_root()

        assert inference._registry.jobs["live-a"] is job_a
        assert job_a.status == "running"
        assert job_a.done == 2 and job_a.total == 5
    finally:
        inference._registry.jobs.clear()


def test_job_registry_register_get_persist_rehydrate_match_the_module_shape(tmp_path, monkeypatch):
    """jobstore.JobRegistry is the one home for the dict-plus-lock register/get/persist/rehydrate
    shape review.py's priority queue, inference.py and tuning.py each adopt: a bare registry
    constructed directly (no route, no module-specific dataclass) exercises the same four
    operations the adopting modules now call through."""
    monkeypatch.chdir(tmp_path)
    from tcip_web.jobstore import JobRegistry, TERMINAL_STATUSES

    root = str(tmp_path.resolve())  # rehydrate/load are keyed by the current root; a fake root
                                     # string would persist under a key rehydrate never reads back.

    class J:
        def __init__(self, job_id, status="pending", platform_root=root):
            self.job_id = job_id
            self.status = status
            self.platform_root = platform_root

    def to_summary(j):
        return {"job_id": j.job_id, "status": j.status, "platform_root": j.platform_root}

    def factory(s, root):
        return J(s["job_id"], status=s["status"], platform_root=s.get("platform_root") or root)

    registry = JobRegistry("inference_jobs", to_summary=to_summary, from_summary=factory)
    job = J("j1", status="completed")
    registry.register(job.job_id, job, job_root=job.platform_root)

    assert registry.get("j1") is job
    assert registry.list(root) == [job]
    assert registry.list("root-b") == []

    registry.jobs.clear()
    registry.rehydrate()
    assert registry.get("j1").status == "completed"
    assert registry.get("j1").status in TERMINAL_STATUSES


def test_job_registry_named_registry_refuses_without_a_summary_codec():
    """A registry that persists must not be able to skip its persist (or its rehydrate)
    silently by a caller simply omitting the codec at one call site: the codec is required at
    construction, once, so a named registry with none refuses to exist rather than register."""
    from tcip_web.jobstore import JobRegistry

    with pytest.raises(ValueError, match="inference_jobs"):
        JobRegistry("inference_jobs")
    with pytest.raises(ValueError):
        JobRegistry("inference_jobs", to_summary=lambda j: {})
    with pytest.raises(ValueError):
        JobRegistry("inference_jobs", from_summary=lambda s, root: s)


def test_job_registry_persist_is_a_no_op_for_an_unpersisted_registry(tmp_path, monkeypatch):
    """images.py's overview-build registry carries no root concept and persists nothing:
    JobRegistry(None) must not write or read anything through jobstore's own store, and needs
    neither codec to be constructed."""
    monkeypatch.chdir(tmp_path)
    from tcip_web.jobstore import JobRegistry

    registry = JobRegistry(None)
    registry.register("ovr1", object(), job_root=None)
    registry.persist()  # no-op: no store binding required
    registry.rehydrate()  # no-op
    assert list(registry.jobs) == ["ovr1"]


def test_registered_job_summaries_persist_byte_stable_through_job_registry(tmp_path, monkeypatch):
    """The persisted job_registry record must not change shape when a registry adopts
    jobstore.JobRegistry. persist_grouped/load are unchanged by the reshape, so a summary
    written directly through persist_grouped (the pre-adoption path every route's own ``_persist``
    called) and the identical summary written through JobRegistry.persist (the post-adoption call
    the adopting routes now make) must decode back to the identical record: value/structural
    equality of the decoded JSON document, the idiom test_persist_grouped_writes_state_that_reads_back
    above already uses for this store. A byte-for-byte comparison of the underlying storage would
    additionally depend on which backend (sqlite/file) is bound, which the persisted shape itself
    does not.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_web.jobstore import JobRegistry, load, persist_grouped

    root = str(tmp_path.resolve())
    summary = {"job_id": "a", "status": "completed", "done": 3, "total": 3,
               "images_dir": "i", "output_dir": "o", "error": None,
               "warning": None, "dropped_nonpositive_boxes": 0, "platform_root": root}

    persist_grouped("inference_jobs", [summary])
    before = load("inference_jobs")

    class J:
        pass

    job = J()
    for k, v in summary.items():
        setattr(job, k, v)

    registry = JobRegistry(
        "inference_jobs",
        to_summary=lambda j: {k: getattr(j, k) for k in summary},
        from_summary=lambda s, root: s,
    )
    registry.jobs["a"] = job
    registry.persist()

    after = load("inference_jobs")
    assert after == before


def test_inference_rehydrate_refuses_a_summary_carrying_no_platform_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_store import replace
    from tcip_web.jobstore import job_registry_key
    from tcip_web.routes import inference

    replace(job_registry_key("inference_jobs"), [
        {"job_id": "old", "status": "completed", "done": 1, "total": 1,
         "images_dir": "i", "output_dir": "o", "error": None},
    ], expect=None)

    with pytest.raises(ValueError, match="no operator door"):
        inference.rehydrate_for_current_root()


def test_review_priority_queue_rehydrate_refuses_a_summary_carrying_no_platform_root(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    from tcip_store import replace
    from tcip_web.jobstore import job_registry_key
    from tcip_web.routes import review

    replace(job_registry_key("review_priority_jobs"), [
        {"job_id": "old", "status": "completed", "error": None, "queue": [],
         "total_candidates": 0, "reviewed_skipped": 0, "marks_unresolved": None},
    ], expect=None)

    with pytest.raises(ValueError, match="no operator door"):
        review.rehydrate_for_current_root()


def test_tuning_rehydrate_refuses_a_summary_carrying_no_platform_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_store import replace
    from tcip_web.jobstore import job_registry_key
    from tcip_web.routes import tuning

    replace(job_registry_key("hpo_sweeps"), [
        {"sweep_id": "old", "status": "completed", "error": None, "has_result": False},
    ], expect=None)

    with pytest.raises(ValueError, match="no operator door"):
        tuning.rehydrate_for_current_root()


def test_review_priority_queue_summaries_persist_byte_stable_through_job_registry(
    tmp_path, monkeypatch,
):
    """The same byte-stability check as inference's own, fed by review's real _pq_summary
    producer, so the priority-queue registry's persisted shape is pinned too, not only
    inference's."""
    monkeypatch.chdir(tmp_path)
    from tcip_web.jobstore import JobRegistry, load, persist_grouped
    from tcip_web.routes import review

    root = str(tmp_path.resolve())
    job = review.PriorityQueueJob(
        job_id="pq-byte", checkpoint_path="c", images_dir="i", dataset_root="d",
        status="completed", queue=[{"image": "a.jpg", "score": 0.9}],
        total_candidates=4, reviewed_skipped=1, platform_root=root,
    )
    persist_grouped("review_priority_jobs", [review._pq_summary(job)])
    before = load("review_priority_jobs")

    registry = JobRegistry(
        "review_priority_jobs", to_summary=review._pq_summary, from_summary=review._pq_from_summary,
    )
    registry.jobs[job.job_id] = job
    registry.persist()

    assert load("review_priority_jobs") == before


def test_tuning_sweep_summaries_persist_byte_stable_through_job_registry(tmp_path, monkeypatch):
    """The same byte-stability check, fed by tuning's real persisted-summary producer, so the
    HPO registry's persisted shape is pinned too."""
    monkeypatch.chdir(tmp_path)
    from tcip_web.jobstore import JobRegistry, load, persist_grouped
    from tcip_web.routes import tuning

    root = str(tmp_path.resolve())
    job = tuning.HPOJob(sweep_id="hpo-byte", status="completed", platform_root=root)
    persist_grouped("hpo_sweeps", [tuning._persisted_summary(job)])
    before = load("hpo_sweeps")

    registry = JobRegistry(
        "hpo_sweeps", to_summary=tuning._persisted_summary, from_summary=tuning._from_summary,
        id_field="sweep_id",
    )
    registry.jobs[job.sweep_id] = job
    registry.persist()

    assert load("hpo_sweeps") == before


def test_job_registry_persist_refuses_to_overwrite_a_document_it_could_not_fully_rehydrate(
    tmp_path, monkeypatch,
):
    """A rehydrate refused by one bad summary must not let ordinary new-job registration
    silently rewrite the document down to just the summaries that did load: the stored document
    survives byte-for-byte until the conform script has stamped the missing key and this
    process is restarted against a conformed document."""
    import importlib.util

    monkeypatch.chdir(tmp_path)
    from tcip_store import read, replace
    from tcip_web.jobstore import JobRegistry, job_registry_key, require_platform_root

    root = str(tmp_path.resolve())
    (tmp_path / ".tcip").mkdir(exist_ok=True)

    class J:
        def __init__(self, job_id, status, platform_root):
            self.job_id = job_id
            self.status = status
            self.platform_root = platform_root

    def to_summary(j):
        return {"job_id": j.job_id, "status": j.status, "platform_root": j.platform_root}

    def from_summary(s, root):
        return J(s["job_id"], s["status"], require_platform_root(s, name="inference_jobs", root=root))

    stored = [
        {"job_id": "old-good", "status": "completed", "platform_root": root},
        {"job_id": "old-bad", "status": "completed"},
    ]
    replace(job_registry_key("inference_jobs"), stored, expect=None)

    registry = JobRegistry("inference_jobs", to_summary=to_summary, from_summary=from_summary)
    with pytest.raises(ValueError, match="no operator door"):
        registry.rehydrate()
    assert read(job_registry_key("inference_jobs"), default=[]) == stored

    with pytest.raises(ValueError, match="no operator door"):
        registry.register("new", J("new", "pending", root), job_root=root)
    assert read(job_registry_key("inference_jobs"), default=[]) == stored

    script = Path(__file__).parent.parent / "scripts" / "conform_job_registry_roots.py"
    spec = importlib.util.spec_from_file_location("conform_job_registry_roots_under_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outcomes, refused = module.conform_root(tmp_path, plan=False)
    assert refused is False

    fresh = JobRegistry("inference_jobs", to_summary=to_summary, from_summary=from_summary)
    fresh.rehydrate()  # no longer raises: this instance never marked the root refused
    fresh.register("new2", J("new2", "pending", root), job_root=root)

    final = read(job_registry_key("inference_jobs"), default=[])
    assert {d["job_id"] for d in final} == {"old-good", "old-bad", "new2"}


def test_a_conformed_summary_rehydrates(tmp_path, monkeypatch):
    """conform_job_registry_roots.py's stamp is what makes a pre-field document rehydratable
    again: the same summary that refuses above, once the conform script has stamped its
    platform_root, loads back into the live registry."""
    import importlib.util

    monkeypatch.chdir(tmp_path)
    from tcip_store import replace
    from tcip_web.jobstore import job_registry_key
    from tcip_web.routes import inference

    (tmp_path / ".tcip").mkdir()
    replace(job_registry_key("inference_jobs"), [
        {"job_id": "old", "status": "completed", "done": 1, "total": 1,
         "images_dir": "i", "output_dir": "o", "error": None},
    ], expect=None)

    script = Path(__file__).parent.parent / "scripts" / "conform_job_registry_roots.py"
    spec = importlib.util.spec_from_file_location("conform_job_registry_roots_under_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outcomes, refused = module.conform_root(tmp_path, plan=False)
    assert refused is False

    inference._registry.jobs.clear()
    try:
        inference.rehydrate_for_current_root()
        jobs = {j["job_id"]: j for j in inference.list_jobs()["jobs"]}
        assert jobs["old"]["status"] == "completed"
    finally:
        inference._registry.jobs.clear()
