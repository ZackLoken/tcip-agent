"""The review queue routes: a launched job runs ``prioritize_review_queue`` over the open project
and the subject the request named, and its status reports the tool's result or refusal."""

from __future__ import annotations

import time
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _no_queue_jobs():
    """Start each test with no queue job held by the backend."""
    import tcip_web.routes.annotate as annotate_mod

    annotate_mod._pq_registry.jobs.clear()


def _wait_for_terminal(client, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/annotate/queue/{job_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached a terminal status")


def test_launch_404s_on_missing_checkpoint(opened_client, tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    resp = opened_client.post("/api/annotate/queue/launch", json={
        "checkpoint_path": str(tmp_path / "nope.pt"), "images_dir": str(images)})
    assert resp.status_code == 404
    assert "checkpoint not found" in resp.json()["detail"]


def test_launch_404s_on_missing_images_dir(opened_client, tmp_path: Path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"not a real checkpoint")
    resp = opened_client.post("/api/annotate/queue/launch", json={
        "checkpoint_path": str(ckpt), "images_dir": str(tmp_path / "nope")})
    assert resp.status_code == 404
    assert "images_dir not found" in resp.json()["detail"]


def test_unknown_job_id_404s(opened_client):
    resp = opened_client.get("/api/annotate/queue/does-not-exist")
    assert resp.status_code == 404


def test_job_completes_and_carries_the_tool_s_own_queue(
    opened_client, tmp_path: Path, opened_project: Path, monkeypatch,
):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"not a real checkpoint")
    images = tmp_path / "images"
    images.mkdir()

    calls: list[dict] = []

    def fake_ranked_review_queue(project, checkpoint_path, images_dir, **kwargs):
        calls.append({"project": project, "checkpoint_path": checkpoint_path,
                      "images_dir": images_dir, **kwargs})
        return {
            "method": "combined", "task": "detection",
            "total_candidates": 3, "reviewed_skipped": 1, "selected_count": 2,
            "queue": [{"image": "b.jpg", "score": 0.9}, {"image": "a.jpg", "score": 0.4}],
        }

    import tcip_mcp.registry_paths as registry_paths
    import tcip_mcp.tools.feedback_tools as feedback_tools_mod
    import tcip_web.paths as web_paths
    monkeypatch.setattr(feedback_tools_mod, "ranked_review_queue", fake_ranked_review_queue)
    real_located = registry_paths.located
    located_paths: list = []

    def located(path, project):
        located_paths.append(path)
        return real_located(path, project)

    for module in (registry_paths, web_paths, feedback_tools_mod):
        monkeypatch.setattr(module, "located", located)

    resp = opened_client.post("/api/annotate/queue/launch", json={
        "checkpoint_path": str(ckpt), "images_dir": str(images), "subject": "bud"})
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    body = _wait_for_terminal(opened_client, job_id)
    assert body["status"] == "completed"
    assert body["queue"] == [{"image": "b.jpg", "score": 0.9}, {"image": "a.jpg", "score": 0.4}]
    assert body["total_candidates"] == 3
    assert body["reviewed_skipped"] == 1

    # The route hands over the open project and the subject the request named.
    assert len(calls) == 1
    assert Path(calls[0]["project"]).resolve() == Path(opened_project).resolve()
    assert calls[0]["subject"] == "bud"
    # The operation is handed the locations the route's own arrival established, each located
    # once, at that arrival.
    assert (calls[0]["checkpoint_path"], calls[0]["images_dir"]) == (ckpt.resolve(),
                                                                    images.resolve())
    assert located_paths == [str(ckpt), str(images)]
    assert "strategy" not in calls[0]


def test_job_fails_honestly_on_the_tool_s_own_refusal(opened_client, tmp_path: Path, monkeypatch):
    """The operation's soft ``{"error": ...}`` (an unresolvable scorer name, say) surfaces as the
    job's ``failed`` status with the same message, never swallowed or completed empty."""
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"not a real checkpoint")
    images = tmp_path / "images"
    images.mkdir()

    def fake_ranked_review_queue(project, checkpoint_path, images_dir, **kwargs):
        return {"error": "no scorer registered as 'nonsense'"}

    import tcip_mcp.tools.feedback_tools as feedback_tools_mod
    monkeypatch.setattr(feedback_tools_mod, "ranked_review_queue", fake_ranked_review_queue)

    resp = opened_client.post("/api/annotate/queue/launch", json={
        "checkpoint_path": str(ckpt), "images_dir": str(images)})
    job_id = resp.json()["job_id"]

    body = _wait_for_terminal(opened_client, job_id)
    assert body["status"] == "failed"
    assert body["error"] == "no scorer registered as 'nonsense'"
    assert body["queue"] == []
