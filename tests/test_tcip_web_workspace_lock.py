"""A held workspace lock delays the routes that need it and never the event loop."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

    from fastapi.testclient import TestClient

    from tcip_web.state import OpenProject

DISPLAY_NAME = "Valley block"


class _SignaledLock:
    """The workspace lock, setting ``reached`` as each acquire begins."""

    def __init__(self, lock: threading.Lock, reached: threading.Event) -> None:
        self._lock = lock
        self._reached = reached

    def __enter__(self) -> bool:
        self._reached.set()
        return self._lock.__enter__()

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self._lock.__exit__(kind, error, trace)


def _request(
    route: str, client: TestClient, project: OpenProject
) -> tuple[str, dict]:
    """Return the URL and body of a request of ``route`` for ``project``, set up through routes.

    ``image_event`` first opens ``project`` and records a contribution, so the request it
    returns names that session.
    """
    named = {"id": project.id, "user": "g"}
    if route == "open":
        return "/api/projects/open", named
    if route == "remove":
        return "/api/projects/remove", {**named, "confirm_name": DISPLAY_NAME}
    assert client.post("/api/projects/open", json=named).status_code == 200
    event = {
        "contribution_id": "c-1",
        "image_name": "IMG_A",
        "seconds": 3.0,
        "annotations_added": 1,
        "activity": "new_annotation",
        "user": "alice",
        "project_id": project.id,
        "started": None,
    }
    first = client.post("/api/sessions/image_event", json=event)
    assert first.status_code == 200, first.text
    return "/api/sessions/image_event", {
        **event,
        "contribution_id": "c-2",
        "started": first.json()["started"],
    }


@pytest.mark.parametrize("route", ["open", "remove", "image_event"])
def test_a_route_waiting_on_the_workspace_lock_leaves_the_event_loop_free(
    tmp_path: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
) -> None:
    """While the lock is held, another request answers; the waiting route answers on release.

    The client is entered, so one event loop serves both requests. The state route takes no
    lock; it answers within five seconds of the route reaching the lock, and the route itself
    answers 200 once the lock is released.
    """
    from tcip_web.routes import projects
    from tests._web_fixtures import named_project

    project = named_project(tmp_path.parent / "valley_block", DISPLAY_NAME)
    answers: dict[str, int] = {}

    def call(
        name: str, method: str, url: str, **kwargs: dict
    ) -> threading.Thread:
        def send() -> None:
            answers[name] = client.request(method, url, **kwargs).status_code

        thread = threading.Thread(target=send, daemon=True)
        thread.start()
        return thread

    with client:
        url, body = _request(route, client, project)
        lock = projects._WORKSPACE_TURN
        reached = threading.Event()
        monkeypatch.setattr(
            projects, "_WORKSPACE_TURN", _SignaledLock(lock, reached)
        )
        with lock:
            waiting = call("route", "POST", url, json=body)
            assert reached.wait(10), (
                f"the {route} route never reached the lock"
            )
            probe = call("state", "GET", "/api/state")
            probe.join(5)
            assert not probe.is_alive(), (
                f"/api/state did not answer while the {route} route waited on the lock"
            )
        waiting.join(10)

    assert answers == {"state": 200, "route": 200}
