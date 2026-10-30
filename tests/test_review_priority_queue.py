"""The review queue routes.

A launched job runs ``prioritize_review_queue``'s ranking over the open project's selected capture
and its subject, and its status reports the ranking's result or refusal and the capture it ranked.
"""

from __future__ import annotations

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


def select(client: TestClient, root: Path, subject: str | None) -> dict:
    """Select the capture :data:`DATE` of ``root`` for ``subject``; return the served selection.

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
    return resp.json()["selection"]


def stub_scorer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Have every scorer block build a scorer ranking the candidates in reverse name order."""
    from types import SimpleNamespace

    import tcip_mcp.pipelines.active_learning.scorer as al_scorer

    monkeypatch.setattr(
        al_scorer,
        "resolve_scorer",
        lambda block, task: SimpleNamespace(
            score=lambda sources, labeled, predictor: [
                (s, float(i)) for i, s in enumerate(sources)
            ][::-1]
        ),
    )


def ended(client: TestClient, job_id: str) -> dict:
    """Wait for the queue job ``job_id`` to end; return its record as the status route answers.

    Ended is the record's own ``live`` read false.
    """
    from tcip_web.routes.annotate import _pq_registry

    job = _pq_registry.get(job_id)
    assert job is not None and job.thread is not None, job_id
    job.thread.join(timeout=60)
    body = client.get(f"/api/annotate/queue/{job_id}").json()
    assert body["live"] is False, body
    return body


@pytest.fixture(autouse=True)
def _no_queue_jobs() -> None:
    """Clear the queue jobs the backend's annotate routes hold."""
    import tcip_web.routes.annotate as annotate_mod

    annotate_mod._pq_registry.jobs.clear()


def test_launch_404s_on_missing_checkpoint(
    opened_client: TestClient, opened_project: Path
) -> None:
    select(opened_client, opened_project, "bud")
    resp = opened_client.post(
        "/api/annotate/queue/launch",
        json={"checkpoint_path": str(opened_project / "nope.pt"), **QUEUE},
    )
    assert resp.status_code == 404
    assert "checkpoint not found" in resp.json()["detail"]


def test_launch_refuses_a_queue_stating_no_scorer_budget_or_reason(
    opened_client: TestClient, opened_project: Path
) -> None:
    select(opened_client, opened_project, "bud")
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


def test_launch_refuses_a_scorer_block_its_factory_s_types_refuse_registering_no_job(
    opened_client: TestClient,
    opened_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A value of the wrong type, one not taken and a required one unstated each answer 422.

    Each refusal names its parameter at the scorer block's place, and no job is registered. The
    factory is the queue fixture's typed sample (a boolean, a list of floats, an optional float).
    """
    import tcip_mcp.pipelines.active_learning.scorer as al_scorer
    from tcip_web.routes.annotate import _pq_registry
    from tests._verified_checkpoint_fixtures import registered_checkpoint
    from tools.generate_queue_fixture import TYPED, typed_scorer

    monkeypatch.setitem(al_scorer.SCORER_REGISTRY, TYPED, typed_scorer)
    ckpt = registered_checkpoint(opened_project)
    select(opened_client, opened_project, "bud")
    stated = {"flag": False, "weights": [1, 2.5], "cap": 0.5}

    for change, refused in (
        ({"cap": "half"}, ("cap", "float_type")),
        ({"unstated": 1}, ("unstated", "extra_forbidden")),
        ({"flag": None}, ("flag", "missing")),
    ):
        values = {
            k: v for k, v in {**stated, **change}.items() if v is not None
        }
        resp = opened_client.post(
            "/api/annotate/queue/launch",
            json={
                "checkpoint_path": ckpt,
                **QUEUE,
                "scorer": {"name": TYPED, **values},
            },
        )
        assert resp.status_code == 422, resp.text
        assert [
            (issue["loc"], issue["type"]) for issue in resp.json()["detail"]
        ] == [(["body", "scorer", refused[0]], refused[1])]
        assert _pq_registry.jobs == {}


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


def test_the_scorer_route_answers_the_catalog(
    opened_client: TestClient,
) -> None:
    """``GET /api/annotate/scorers`` answers ``scorer_catalog``'s catalog."""
    from tcip_mcp.pipelines.active_learning import scorer_catalog

    resp = opened_client.get("/api/annotate/scorers")
    assert resp.status_code == 200, resp.text
    assert resp.json() == scorer_catalog().model_dump(mode="json")


def test_job_completes_and_carries_the_tool_s_own_queue_and_the_capture_it_ranked(
    opened_client: TestClient,
    opened_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job over the selected capture answers one record, the ranking's entries in its queue.

    The launch and the status answer the same record, naming the capture the selection names;
    the ranking runs over the selected capture's directory and its subject; the job's queue is
    the tool's own answer over the same capture.
    """
    import pydantic_core

    import tcip_mcp.registry_paths as registry_paths
    import tcip_mcp.tools.feedback_tools as feedback_tools_mod
    import tcip_web.paths as web_paths
    from tcip_mcp.pipelines.active_learning import ScorerSpec
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(opened_project)
    selection = select(opened_client, opened_project, "bud")
    images = image_dir(opened_project, DATE)
    stub_scorer(monkeypatch)

    ranking = feedback_tools_mod.ranked_review_queue
    calls: list[dict] = []

    def ranked_review_queue(
        project: Path,
        checkpoint_path: Path,
        images_dir: Path,
        **kwargs: ScorerSpec | int | str | None,
    ) -> ReviewQueue | dict[str, str]:
        calls.append(
            {
                "project": project,
                "checkpoint_path": checkpoint_path,
                "images_dir": images_dir,
                **kwargs,
            }
        )
        return ranking(project, checkpoint_path, images_dir, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        feedback_tools_mod, "ranked_review_queue", ranked_review_queue
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
        json={"checkpoint_path": ckpt, **QUEUE},
    )
    assert resp.status_code == 200, resp.text
    launched = resp.json()

    body = ended(opened_client, launched["job_id"])
    assert body["status"] == "completed", body
    # Both routes answer the one job record; the launch's has no ranking yet.
    assert launched.keys() == body.keys()
    assert launched["ranked"] is None
    # The record names the capture as the selection does.
    assert (body["dataset_root"], body["capture"]) == (
        selection["dataset_root"],
        selection["date"],
    )
    ranked = body["ranked"]
    assert {name: ranked[name] for name in QUEUE} == QUEUE
    assert [entry["image"] for entry in ranked["queue"]] == ["b.jpg", "a.jpg"]
    # The checkpoint is located once, at the route's arrival.
    assert located_paths[0] == ckpt
    assert located_paths.count(ckpt) == 1

    tool_answer = feedback_tools_mod.prioritize_review_queue(
        opened_project,
        ckpt,
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
    assert route_call["checkpoint_path"] == Path(ckpt).resolve()


def test_job_fails_on_the_ranking_s_own_refusal(
    opened_client: TestClient, opened_project: Path
) -> None:
    """A queue job reports the ranking's own refusal as its error, ranking nothing.

    A diversity ranking over a selection naming no subject has no labeled image to measure
    against, and the job fails with the scorer's refusal as the ranking wrote it.
    """
    from tcip_mcp.pipelines.active_learning import ScorerSpec
    from tcip_mcp.tools.feedback_tools import prioritize_review_queue
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(opened_project)
    select(opened_client, opened_project, None)
    queue = {**QUEUE, "scorer": {"name": "diversity"}}

    resp = opened_client.post(
        "/api/annotate/queue/launch", json={"checkpoint_path": ckpt, **queue}
    )
    body = ended(opened_client, resp.json()["job_id"])

    refusal = prioritize_review_queue(
        opened_project,
        ckpt,
        str(image_dir(opened_project, DATE)),
        scorer=ScorerSpec(name="diversity"),
        budget=QUEUE["budget"],
        rationale=QUEUE["rationale"],
    )
    assert isinstance(refusal, dict), refusal
    assert body["status"] == "failed"
    assert body["error"] == refusal["error"]
    assert "no image finished" in body["error"]
    assert body["ranked"] is None
