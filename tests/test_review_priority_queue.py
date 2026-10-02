"""The breeder-browsable priority queue surfaced from prioritize_review_queue.

The route (`/api/annotate/queue/launch` + `/api/annotate/queue/{job_id}`) must never
reimplement `prioritize_review_queue`'s own scoring/filtering: its own tests
(test_feedback_tools.py) already cover checkpoint-missing / non-composed-kind /
unresolvable-scorer. These tests pin the route's own job-lifecycle wiring: it calls the same tool
function on a background thread and maps its result (or its soft {"error": ...}) onto the job,
handing the tool the open project and the subject the request named.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest


@pytest.fixture
def client(opened_project):
    from fastapi.testclient import TestClient

    import tcip_web.routes.annotate as annotate_mod
    from tcip_web.app import app

    annotate_mod._pq_registry.jobs.clear()  # a stale job from another test must not leak into this one
    return TestClient(app, base_url="http://127.0.0.1")


def _wait_for_terminal(client, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/annotate/queue/{job_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached a terminal status")


def test_launch_404s_on_missing_checkpoint(client, tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    resp = client.post("/api/annotate/queue/launch", json={
        "checkpoint_path": str(tmp_path / "nope.pt"), "images_dir": str(images)})
    assert resp.status_code == 404
    assert "checkpoint not found" in resp.json()["detail"]


def test_launch_404s_on_missing_images_dir(client, tmp_path: Path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"not a real checkpoint")
    resp = client.post("/api/annotate/queue/launch", json={
        "checkpoint_path": str(ckpt), "images_dir": str(tmp_path / "nope")})
    assert resp.status_code == 404
    assert "images_dir not found" in resp.json()["detail"]


def test_unknown_job_id_404s(client):
    resp = client.get("/api/annotate/queue/does-not-exist")
    assert resp.status_code == 404


def test_job_completes_and_carries_the_tool_s_own_queue(
    client, tmp_path: Path, opened_project: Path, monkeypatch,
):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"not a real checkpoint")
    images = tmp_path / "images"
    images.mkdir()

    calls: list[dict] = []

    def fake_prioritize_review_queue(project, **kwargs):
        calls.append({"project": project, **kwargs})
        return {
            "method": "combined", "task": "detection",
            "total_candidates": 3, "reviewed_skipped": 1, "selected_count": 2,
            "queue": [{"image": "b.jpg", "score": 0.9}, {"image": "a.jpg", "score": 0.4}],
        }

    import tcip_mcp.tools.feedback_tools as feedback_tools_mod
    monkeypatch.setattr(feedback_tools_mod, "prioritize_review_queue", fake_prioritize_review_queue)

    resp = client.post("/api/annotate/queue/launch", json={
        "checkpoint_path": str(ckpt), "images_dir": str(images), "subject": "bud"})
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    body = _wait_for_terminal(client, job_id)
    assert body["status"] == "completed"
    assert body["queue"] == [{"image": "b.jpg", "score": 0.9}, {"image": "a.jpg", "score": 0.4}]
    assert body["total_candidates"] == 3
    assert body["reviewed_skipped"] == 1

    # The route hands over the open project and the subject the request named.
    assert len(calls) == 1
    assert Path(calls[0]["project"]).resolve() == Path(opened_project).resolve()
    assert calls[0]["subject"] == "bud"
    # The route never sends a strategy kwarg: prioritize_review_queue accepts none since the split.
    assert "strategy" not in calls[0]


def test_job_fails_honestly_on_the_tool_s_own_refusal(client, tmp_path: Path, monkeypatch):
    # prioritize_review_queue returns a soft {"error": ...} dict (never raises) for e.g. an
    # unresolvable scorer name: the job must surface that as status=failed with the same message,
    # not swallow it or report completed with an empty queue.
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"not a real checkpoint")
    images = tmp_path / "images"
    images.mkdir()

    def fake_prioritize_review_queue(project, **kwargs):
        return {"error": "no scorer registered as 'nonsense'"}

    import tcip_mcp.tools.feedback_tools as feedback_tools_mod
    monkeypatch.setattr(feedback_tools_mod, "prioritize_review_queue", fake_prioritize_review_queue)

    resp = client.post("/api/annotate/queue/launch", json={
        "checkpoint_path": str(ckpt), "images_dir": str(images)})
    job_id = resp.json()["job_id"]

    body = _wait_for_terminal(client, job_id)
    assert body["status"] == "failed"
    assert body["error"] == "no scorer registered as 'nonsense'"
    assert body["queue"] == []
