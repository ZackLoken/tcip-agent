"""Web job lifecycle: the in-memory registry, its memory cap, and inference cancellation."""

import pytest

from tcip_mcp.pipelines.execution import Stated

STATED = Stated(tile=False, conf=0.25, cross_tile_nms=0.7, overlap=0.2)
"""The execution values every inference job here states."""


class J:
    """A job as the registry reads one: its status, the project it runs for, and its id."""

    def __init__(self, status: str, root: str = "root-a", job_id: str = "") -> None:
        self.status = status
        self.project = root
        self.job_id = job_id


def test_evict_terminal_caps_and_keeps_running():
    from tcip_web.jobstore import evict_terminal

    jobs = {f"done{i}": J("completed") for i in range(5)}
    jobs["live"] = J("running")
    evict_terminal(jobs, max_jobs=3)

    assert len(jobs) == 3
    assert "live" in jobs           # running jobs are never evicted
    assert "done0" not in jobs      # oldest terminal evicted first


def test_evict_terminal_drops_the_oldest_terminal_jobs_whichever_root_they_belong_to():
    """The whole dict is trimmed to max_jobs, oldest terminal job of any root first, so a lone
    recent job from a different root survives while older ones are still there to go."""
    from tcip_web.jobstore import evict_terminal

    jobs = {f"a{i}": J("completed", "root-a") for i in range(5)}
    jobs["b_done"] = J("completed", "root-b")
    evict_terminal(jobs, max_jobs=3)

    assert "b_done" in jobs
    assert sum(1 for j in jobs.values() if j.project == "root-a") == 2


def test_evict_terminal_bounds_the_whole_dict_across_roots():
    """The whole dict stays bounded at max_jobs even as more roots register their own jobs,
    not just each root's own share: a root that has stopped receiving launches is trimmed too,
    the leak this helper exists to close."""
    from tcip_web.jobstore import evict_terminal

    jobs: dict[str, J] = {}
    for i in range(7):
        jobs[f"a{i}"] = J("completed", "root-a")
        evict_terminal(jobs, max_jobs=5)
    for i in range(7):
        jobs[f"b{i}"] = J("completed", "root-b")
        evict_terminal(jobs, max_jobs=5)

    assert len(jobs) <= 5
    assert not any(j.project == "root-a" for j in jobs.values())


def test_job_registry_registers_gets_lists_by_root_and_finds_or_registers():
    """The one dict-plus-lock registry every route adopts: a job registered under a root is got
    by id from any root, listed only under its own, and found again rather than made twice."""
    from tcip_web.jobstore import JobRegistry

    registry = JobRegistry()
    job = J("pending", "root-a", "j1")
    registry.register(job.job_id, job)

    assert registry.get("j1") is job
    assert registry.list("root-a") == [job]
    assert registry.list("root-b") == []

    found, created = registry.find_or_register(lambda j: j.job_id == "j1",
                                               lambda: J("pending", "root-a", "j2"))
    assert (found, created) == (job, False)
    made, created = registry.find_or_register(lambda j: j.job_id == "j3",
                                              lambda: J("pending", "root-b", "j3"))
    assert created is True and registry.get("j3") is made


def _fake_predictor(monkeypatch) -> None:
    from tests._predictor_fixtures import StubPredictor, install

    install(monkeypatch, StubPredictor(width=16, height=16, boxes=(), scores=()))


def _one_image(tmp_path):
    from PIL import Image

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (16, 16)).save(images_dir / "img.jpg")
    return images_dir


def test_inference_cancel_endpoint_and_worker(tmp_path, opened_project, monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.chdir(tmp_path)
    from fastapi import HTTPException

    from tcip_web.routes._body_common import PersonPayload
    from tcip_web.routes.inference import InferenceJob, _register, _worker, cancel_job
    from tests._audit_fixtures import audit_rows
    from tests._verified_checkpoint_fixtures import project_checkpoint

    images_dir = _one_image(tmp_path)
    _fake_predictor(monkeypatch)

    job = InferenceJob(job_id="j1", actor="user:tester", project=str(tmp_path),
                       checkpoint_path=project_checkpoint(tmp_path), dataset_root=str(tmp_path),
                       images_dir=str(images_dir), bucket="out", stated=STATED)
    _register(job)

    res = cancel_job("j1", PersonPayload(user="Alice"))
    assert res["cancel_requested"] is True and job.cancel_event.is_set()
    (line,) = audit_rows(tmp_path, "inference_canceled")
    assert (line["actor"], line["arguments"]["job_id"]) == ("user:Alice", "j1")
    # Canceling a job that was never registered is a client-side miss, so it has to reach the
    # browser as a 404 and name the id: any other status reads to the caller as a real outcome.
    with pytest.raises(HTTPException) as cancel_miss:
        cancel_job("missing", PersonPayload(user="Alice"))
    assert cancel_miss.value.status_code == 404
    assert "missing" in cancel_miss.value.detail

    _worker(job)  # honors the pre-set cancel
    assert job.status == "canceled"
    assert job.done == 0


def test_inference_cancel_reaches_a_job_launched_for_a_previously_open_project(
    tmp_path, opened_project,
):
    """Canceling a run one launched is legitimate work: opening another project must not make
    the job invisible to cancel or stream, only to the list route."""
    from fastapi import HTTPException

    from tcip_web.routes._body_common import PersonPayload
    from tcip_web.routes.inference import (
        InferenceJob, _get, _register, _registry, cancel_job, list_jobs,
    )
    from tests._web_fixtures import open_new_project

    job = InferenceJob(
        job_id="launched-under-a", actor="user:tester", project=str(opened_project),
        checkpoint_path="c", dataset_root="d", images_dir="i", bucket="b", stated=STATED,
    )
    _register(job)

    try:
        open_new_project(tmp_path / "other")

        assert _get("launched-under-a") is job
        assert list_jobs()["jobs"] == []

        res = cancel_job("launched-under-a", PersonPayload(user="Alice"))
        assert res["cancel_requested"] is True
        assert job.cancel_event.is_set()

        with pytest.raises(HTTPException) as miss:
            cancel_job("never-launched", PersonPayload(user="Alice"))
        assert miss.value.status_code == 404
    finally:
        _registry.jobs.clear()


def test_priority_queue_by_id_reaches_a_job_launched_for_a_previously_open_project(
    tmp_path, opened_project,
):
    """Answering a ranked queue one launched is legitimate work, the same contract inference
    already holds: opening another project must not make the job invisible by id."""
    from fastapi import HTTPException

    from tcip_web.routes.annotate import PriorityQueueJob, _pq_registry, get_priority_queue_job
    from tests._web_fixtures import open_new_project

    job = PriorityQueueJob(
        job_id="pq-under-a", project=str(opened_project), checkpoint_path="c", images_dir="i",
        subject=None, method="combined", budget=50, status="completed",
        queue=[{"image": "a.jpg", "score": 0.9}],
    )
    _pq_registry.register(job.job_id, job)

    try:
        open_new_project(tmp_path / "other")

        assert _pq_registry.get("pq-under-a") is job
        assert get_priority_queue_job("pq-under-a")["job_id"] == "pq-under-a"

        with pytest.raises(HTTPException) as miss:
            get_priority_queue_job("pq-never-launched")
        assert miss.value.status_code == 404
    finally:
        _pq_registry.jobs.clear()
