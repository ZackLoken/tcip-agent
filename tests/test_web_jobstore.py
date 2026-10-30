"""Web job lifecycle: the in-memory registry, its memory cap, and inference cancellation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.pipelines.execution import Stated
from tcip_web.jobstore import JobStatus, ProjectJob
from tests._producer_fixtures import write_image
from tests._training_values import RATIONALE

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient

STATED = Stated(
    tile=False, conf=0.25, cross_tile_nms=0.7, overlap=0.2, rationale=RATIONALE
)
"""The execution values every inference job here states."""


def _job(
    status: JobStatus, root: str = "root-a", job_id: str = ""
) -> ProjectJob:
    """Return a job of ``root`` holding ``status`` under ``job_id``, over a capture of ``root``."""
    return ProjectJob(
        job_id=job_id,
        status=status,
        project=root,
        checkpoint_path="model.pt",
        images_dir=str(image_dir(root, "2026-01-01")),
    )


def test_evict_terminal_caps_and_keeps_running() -> None:
    from tcip_web.jobstore import evict_terminal

    jobs = {f"done{i}": _job("completed") for i in range(5)}
    jobs["live"] = _job("running")
    evict_terminal(jobs, max_jobs=3)

    assert len(jobs) == 3
    assert "live" in jobs  # running jobs are never evicted
    assert "done0" not in jobs  # oldest terminal evicted first


def test_evict_terminal_drops_the_oldest_terminal_jobs_whichever_root_they_belong_to() -> (
    None
):
    """Trimming to max_jobs keeps a lone recent job of another root beside the first root's.

    Six terminal jobs across two roots trimmed to three keep the second root's one job and two
    of the first root's.
    """
    from tcip_web.jobstore import evict_terminal

    jobs = {f"a{i}": _job("completed", "root-a") for i in range(5)}
    jobs["b_done"] = _job("completed", "root-b")
    evict_terminal(jobs, max_jobs=3)

    assert "b_done" in jobs
    assert sum(1 for j in jobs.values() if j.project == "root-a") == 2


def test_evict_terminal_bounds_the_whole_dict_across_roots() -> None:
    """Eviction bounds the whole job dict across roots.

    The whole dict stays at max_jobs or fewer as a second root registers its jobs, and the
    first root, which receives no more launches, is trimmed away.
    """
    from tcip_web.jobstore import evict_terminal

    jobs: dict[str, ProjectJob] = {}
    for i in range(7):
        jobs[f"a{i}"] = _job("completed", "root-a")
        evict_terminal(jobs, max_jobs=5)
    for i in range(7):
        jobs[f"b{i}"] = _job("completed", "root-b")
        evict_terminal(jobs, max_jobs=5)

    assert len(jobs) <= 5
    assert not any(j.project == "root-a" for j in jobs.values())


def test_job_registry_answers_snapshots_and_hands_the_live_job_to_its_worker() -> (
    None
):
    """The job registry launches, gets, lists by root, finds or launches, and signals.

    A job launched under a root is got by id and listed under its own root and not another,
    each answer a snapshot equal to the job and never the object its worker holds; a find
    matching it answers it with nothing created, a find matching none launches the job it makes,
    and a signal runs over the live job and answers its snapshot after.
    """
    from tcip_web.jobstore import JobRegistry

    registry: JobRegistry[ProjectJob] = JobRegistry()
    worked: list[ProjectJob] = []
    job = _job("pending", "root-a", "j1")
    launched = registry.launch(job, worked.append)
    assert job.thread is not None
    job.thread.join(timeout=10)

    assert worked == [job] and worked[0] is job
    assert launched.model_dump() == job.model_dump() and launched is not job
    got = registry.get("j1")
    assert got == job and got is not job
    assert registry.list("root-a") == [job]
    assert registry.list("root-b") == []

    found, created = registry.find_or_launch(
        lambda j: j.job_id == "j1",
        lambda: _job("pending", "root-a", "j2"),
        worked.append,
    )
    assert (found, created) == (job, False)
    made, created = registry.find_or_launch(
        lambda j: j.job_id == "j3",
        lambda: _job("pending", "root-b", "j3"),
        lambda _job: None,
    )
    j3 = registry.get("j3")
    assert created is True and j3 is not None
    assert j3.model_dump() == made.model_dump()

    signaled = registry.signal("j1", lambda live: setattr(live, "error", "x"))
    assert job.error == "x" and signaled == job and signaled is not job
    assert registry.signal("missing", worked.append) is None


def _fake_predictor(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests._predictor_fixtures import StubPredictor, install

    install(
        monkeypatch, StubPredictor(width=16, height=16, boxes=(), scores=())
    )


def test_inference_cancel_endpoint_and_worker(
    tmp_path: Path, opened_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    monkeypatch.chdir(tmp_path)
    from fastapi import HTTPException

    from tcip_web.routes._body_common import PersonPayload
    from tcip_web.routes.inference import _registry, _worker, cancel_job
    from tests._audit_fixtures import audit_rows
    from tests._verified_checkpoint_fixtures import project_checkpoint
    from tests._web_fixtures import inference_job

    images_dir = image_dir(tmp_path, "2026-01-01")
    write_image(images_dir / "img.jpg", (16, 16))
    _fake_predictor(monkeypatch)

    job = inference_job(
        images_dir,
        tmp_path,
        job_id="j1",
        checkpoint_path=project_checkpoint(tmp_path),
        stated=STATED,
    )
    _registry.launch(job, lambda _job: None)

    res = cancel_job("j1", PersonPayload(user="Alice"))
    assert res == job and job.cancel_event.is_set()
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
    tmp_path: Path,
    opened_project: Path,
) -> None:
    """An inference cancel reaches a job launched for a previously open project.

    After another project is opened the job is found by id and canceled, the list route lists
    no job, and an unknown id answers 404.
    """
    from fastapi import HTTPException

    from tcip_web.routes._body_common import PersonPayload
    from tcip_web.routes.inference import (
        _get,
        _registry,
        cancel_job,
        list_jobs,
    )
    from tests._web_fixtures import inference_job, open_new_project

    job = inference_job(
        image_dir(tmp_path / "ds", "2026-01-01"),
        opened_project,
        job_id="launched-under-a",
    )
    _registry.launch(job, lambda _job: None)

    try:
        open_new_project(tmp_path / "other")

        assert _get("launched-under-a") == job
        assert list_jobs().jobs == []

        assert (
            cancel_job("launched-under-a", PersonPayload(user="Alice")) == job
        )
        assert job.cancel_event.is_set()

        with pytest.raises(HTTPException) as miss:
            cancel_job("never-launched", PersonPayload(user="Alice"))
        assert miss.value.status_code == 404
    finally:
        _registry.jobs.clear()


def test_priority_queue_by_id_reaches_a_job_launched_for_a_previously_open_project(
    tmp_path: Path,
    opened_project: Path,
    opened_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A priority queue reached by id answers a job launched for a previously open project.

    After another project is opened the job the launch route made is found by id, its ranking
    held, and an unknown id answers 404.
    """
    from fastapi import HTTPException

    from tcip_web.routes.annotate import _pq_registry, get_priority_queue_job
    from tests._verified_checkpoint_fixtures import registered_checkpoint
    from tests._web_fixtures import open_new_project
    from tests.test_review_priority_queue import (
        QUEUE,
        ended,
        select,
        stub_scorer,
    )

    ckpt = registered_checkpoint(opened_project)
    select(opened_client, opened_project, None)
    stub_scorer(monkeypatch)
    launched = opened_client.post(
        "/api/annotate/queue/launch", json={"checkpoint_path": ckpt, **QUEUE}
    ).json()
    job_id = launched["job_id"]

    try:
        held = ended(opened_client, job_id)
        open_new_project(tmp_path / "other")

        assert get_priority_queue_job(job_id).model_dump(mode="json") == held
        assert held["ranked"]["queue"]

        with pytest.raises(HTTPException) as miss:
            get_priority_queue_job("pq-never-launched")
        assert miss.value.status_code == 404
    finally:
        _pq_registry.jobs.clear()
