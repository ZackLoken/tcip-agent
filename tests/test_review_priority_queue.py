"""The review queue routes.

A launched job runs ``prioritize_review_queue``'s ranking over the open project's selected capture
and its subject, and its status reports the ranking's result or refusal.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_mcp.dataset_layout import image_dir
from tests._producer_fixtures import write_image

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from tcip_mcp.pipelines.active_learning import ScorerSpec
    from tcip_mcp.tools.feedback_tools import ReviewQueue

QUEUE = {
    "scorer": {"name": "uncertainty"},
    "budget": 10,
    "rationale": "a sample reason the test states",
}
"""What a launch here states: its scorer block, budget and their reason."""

DATE = "2026-02-11"
"""The capture a launch here ranks."""


def _select(client: TestClient, root: Path, subject: str | None) -> Path:
    """Select the capture :data:`DATE` of ``root`` for ``subject`` and return its directory.

    Its two images are written first; the selection is made through the select route.
    """
    images = image_dir(root, DATE)
    for name in ("a.jpg", "b.jpg"):
        write_image(images / name, (8, 8))
    resp = client.post(
        "/api/dataset/select",
        json={"dataset_root": str(root), "subject": subject, "date": DATE},
    )
    assert resp.status_code == 200, resp.text
    return images


def _ranked(
    project: Path,
    checkpoint_path: Path,
    images_dir: Path,
    **kwargs: ScorerSpec | int | str | None,
) -> ReviewQueue:
    """Answer the ranking's own answer over two entries, as ``ranked_review_queue`` does."""
    from tcip_mcp.pipelines.active_learning import ScorerSpec
    from tcip_mcp.tools.feedback_tools import QueueEntry, ReviewQueue

    scorer, budget, rationale = (
        kwargs[k] for k in ("scorer", "budget", "rationale")
    )
    assert isinstance(scorer, ScorerSpec)
    assert isinstance(budget, int) and isinstance(rationale, str)
    return ReviewQueue(
        scorer=scorer,
        budget=budget,
        rationale=rationale,
        task="detection",
        total_candidates=3,
        reviewed_skipped=1,
        queue=[
            QueueEntry(image="b.jpg", score=0.9, reference_member=None),
            QueueEntry(image="a.jpg", score=0.4, reference_member=True),
        ],
    )


@pytest.fixture(autouse=True)
def _no_queue_jobs() -> None:
    """Clear the queue jobs the backend's annotate routes hold."""
    import tcip_web.routes.annotate as annotate_mod

    annotate_mod._pq_registry.jobs.clear()


def _wait_for_terminal(
    client: TestClient, job_id: str, timeout: float = 5.0
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/annotate/queue/{job_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached a terminal status")


def test_launch_404s_on_missing_checkpoint(
    opened_client: TestClient, opened_project: Path
) -> None:
    _select(opened_client, opened_project, "bud")
    resp = opened_client.post(
        "/api/annotate/queue/launch",
        json={"checkpoint_path": str(opened_project / "nope.pt"), **QUEUE},
    )
    assert resp.status_code == 404
    assert "checkpoint not found" in resp.json()["detail"]


def test_launch_refuses_a_queue_stating_no_scorer_budget_or_reason(
    opened_client: TestClient, opened_project: Path
) -> None:
    _select(opened_client, opened_project, "bud")
    for missing in QUEUE:
        resp = opened_client.post(
            "/api/annotate/queue/launch",
            json={
                "checkpoint_path": str(opened_project / "nope.pt"),
                **{k: v for k, v in QUEUE.items() if k != missing},
            },
        )
        assert resp.status_code == 422, (missing, resp.text)
        assert missing in resp.text


def test_launch_with_no_capture_selected_answers_400_naming_the_select_route(
    opened_client: TestClient, opened_project: Path
) -> None:
    ckpt = opened_project / "model.pt"
    ckpt.write_bytes(b"not a real checkpoint")
    resp = opened_client.post(
        "/api/annotate/queue/launch",
        json={"checkpoint_path": str(ckpt), **QUEUE},
    )
    assert resp.status_code == 400, resp.text
    assert "/api/dataset/select" in resp.json()["detail"]


def test_unknown_job_id_404s(opened_client: TestClient) -> None:
    resp = opened_client.get("/api/annotate/queue/does-not-exist")
    assert resp.status_code == 404


def test_job_completes_and_carries_the_tool_s_own_queue(
    opened_client: TestClient,
    opened_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job over the selected capture answers one record, the ranking's entries in its queue.

    The launch and the status answer the same record; the ranking runs over the selected
    capture's directory and its subject; the job's queue is the tool's own dumped answer's.
    """
    import pydantic_core

    ckpt = opened_project / "model.pt"
    ckpt.write_bytes(b"not a real checkpoint")
    images = _select(opened_client, opened_project, "bud")

    calls: list[dict] = []

    def fake_ranked_review_queue(
        project: Path,
        checkpoint_path: Path,
        images_dir: Path,
        **kwargs: ScorerSpec | int | str | None,
    ) -> ReviewQueue:
        calls.append(
            {
                "project": project,
                "checkpoint_path": checkpoint_path,
                "images_dir": images_dir,
                **kwargs,
            }
        )
        return _ranked(project, checkpoint_path, images_dir, **kwargs)

    import tcip_mcp.registry_paths as registry_paths
    import tcip_mcp.tools.feedback_tools as feedback_tools_mod
    import tcip_web.paths as web_paths

    monkeypatch.setattr(
        feedback_tools_mod, "ranked_review_queue", fake_ranked_review_queue
    )
    real_located = registry_paths.located
    located_paths: list = []

    def located(path: str | Path, project: str | Path | None) -> Path:
        located_paths.append(path)
        return real_located(path, project)

    for module in (registry_paths, web_paths, feedback_tools_mod):
        monkeypatch.setattr(module, "located", located)

    resp = opened_client.post(
        "/api/annotate/queue/launch",
        json={"checkpoint_path": str(ckpt), **QUEUE},
    )
    assert resp.status_code == 200, resp.text
    launched = resp.json()
    job_id = launched["job_id"]

    body = _wait_for_terminal(opened_client, job_id)
    assert body["status"] == "completed"
    # Both routes answer the one job record; the launch's has no ranking yet.
    assert launched.keys() == body.keys()
    assert launched["ranked"] is None
    ranked = body["ranked"]
    assert {name: ranked[name] for name in QUEUE} == QUEUE
    assert ranked["total_candidates"] == 3
    assert ranked["reviewed_skipped"] == 1
    # The checkpoint is located once, at the route's arrival.
    assert located_paths == [str(ckpt)]
    # The job's ranking is the tool's own answer, dumped as the server dumps it.
    from tcip_mcp.pipelines.active_learning import ScorerSpec

    tool_answer = feedback_tools_mod.prioritize_review_queue(
        opened_project,
        str(ckpt),
        str(images),
        scorer=ScorerSpec.model_validate(QUEUE["scorer"]),
        budget=QUEUE["budget"],
        rationale=QUEUE["rationale"],
        subject="bud",
    )
    assert ranked == pydantic_core.to_jsonable_python(tool_answer)

    # The route hands over the open project and the selected capture and its subject.
    route_call = calls[0]
    assert (
        Path(route_call["project"]).resolve() == Path(opened_project).resolve()
    )
    assert route_call["subject"] == "bud"
    assert route_call["images_dir"] == images
    assert (
        route_call["scorer"].model_dump(),
        route_call["budget"],
        route_call["rationale"],
    ) == (QUEUE["scorer"], QUEUE["budget"], QUEUE["rationale"])
    assert route_call["checkpoint_path"] == ckpt.resolve()
    assert "strategy" not in route_call


def test_job_fails_honestly_on_the_tool_s_own_refusal(
    opened_client: TestClient,
    opened_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A queue job reports the operation's own refusal as its ``failed`` status.

    The operation's soft ``{"error": ...}`` (an unresolvable scorer name, say) surfaces as the
    job's ``failed`` status with the same message and no ranking.
    """
    ckpt = opened_project / "model.pt"
    ckpt.write_bytes(b"not a real checkpoint")
    _select(opened_client, opened_project, None)

    def fake_ranked_review_queue(
        project: Path,
        checkpoint_path: Path,
        images_dir: Path,
        **kwargs: ScorerSpec | int | str | None,
    ) -> dict:
        return {"error": "no scorer registered as 'nonsense'"}

    import tcip_mcp.tools.feedback_tools as feedback_tools_mod

    monkeypatch.setattr(
        feedback_tools_mod, "ranked_review_queue", fake_ranked_review_queue
    )

    resp = opened_client.post(
        "/api/annotate/queue/launch",
        json={"checkpoint_path": str(ckpt), **QUEUE},
    )
    job_id = resp.json()["job_id"]

    body = _wait_for_terminal(opened_client, job_id)
    assert body["status"] == "failed"
    assert body["error"] == "no scorer registered as 'nonsense'"
    assert body["ranked"] is None
