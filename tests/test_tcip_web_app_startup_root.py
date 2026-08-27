"""Tests for tcip_web.app's platform-state root pin: a served app's own responsibility,
never an importer's, and off the event loop when the marker read can block."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

import httpx
from fastapi.testclient import TestClient

from tcip_mcp import project_paths, workspace
from tcip_mcp.web_client import PANEL_EVENT_ACTIVE_PROJECT_CHANGED
from tcip_web.app import app

_IMPORT_ONLY = """\
import os
before = os.environ.get("TCIP_PROJECT_ROOT")
from tcip_web.app import app  # noqa: F401
after = os.environ.get("TCIP_PROJECT_ROOT")
print(before)
print(after)
"""


def test_importing_the_app_alone_pins_nothing(tmp_path, monkeypatch):
    """A repo script, or a test module at collection, that only imports tcip_web.app must not
    silently repin TCIP_PROJECT_ROOT to whatever the machine's workspace marker names."""
    ws = tmp_path / "ws"
    proj = ws / "elderberry_cyme_bloom"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    workspace.set_active_project("elderberry_cyme_bloom")

    env = dict(os.environ)
    env.pop("TCIP_PROJECT_ROOT", None)
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_ONLY],
        capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    before, after = result.stdout.splitlines()
    assert before == "None"
    assert after == "None"


def test_first_request_pins_from_the_marker(tmp_path, monkeypatch):
    import tcip_store

    from tcip_mcp.project_paths import project_root

    ws = tmp_path / "ws"
    proj = ws / "elderberry_cyme_bloom"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    monkeypatch.delenv("TCIP_PROJECT_ROOT", raising=False)
    project_paths.restore_binding(None)
    tcip_store.replace(workspace.active_project_key(), "elderberry_cyme_bloom")

    resp = TestClient(app, base_url="http://127.0.0.1").get("/health")

    assert resp.status_code == 200
    assert project_root().resolve() == proj.resolve()


def test_a_binding_set_before_the_first_request_is_not_replaced(tmp_path, monkeypatch):
    """``set_active_project`` (source ``adopted``) can run before this process has served
    its first request; the middleware's own bind must leave that binding alone rather than
    resolving the marker itself and overwriting it with source ``marker``."""
    ws = tmp_path / "ws"
    proj = ws / "elderberry_cyme_bloom"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    monkeypatch.delenv("TCIP_PROJECT_ROOT", raising=False)
    project_paths.restore_binding(None)

    workspace.set_active_project("elderberry_cyme_bloom")
    assert project_paths.root_binding().source == "adopted"

    resp = TestClient(app, base_url="http://127.0.0.1").get("/health")

    assert resp.status_code == 200
    assert project_paths.root_binding().source == "adopted"
    assert project_paths.root_binding().root.resolve() == proj.resolve()


def test_lifespan_binds_before_rehydrate_reads_a_registry(tmp_path, monkeypatch):
    import tcip_store

    from tcip_web import jobstore
    from tcip_web.routes import inference

    ws = tmp_path / "ws"
    proj = ws / "elderberry_cyme_bloom"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    monkeypatch.delenv("TCIP_PROJECT_ROOT", raising=False)
    project_paths.restore_binding(None)
    tcip_store.replace(workspace.active_project_key(), "elderberry_cyme_bloom")

    seen_roots = []
    real_rehydrate = inference.rehydrate_for_current_root

    def _record_then_rehydrate():
        seen_roots.append(jobstore.current_root())
        real_rehydrate()

    monkeypatch.setattr(inference, "rehydrate_for_current_root", _record_then_rehydrate)

    with TestClient(app, base_url="http://127.0.0.1"):
        pass

    assert seen_roots == [str(proj.resolve())]


async def _largest_loop_gap(coro):
    """Run ``coro`` alongside a ticker that measures the largest gap, in seconds, between
    consecutive event-loop turns while it is in flight.

    Blind to which thread does the work inside ``coro``; sensitive only to whether the loop
    itself stayed free to run other coroutines meanwhile. A gap near the duration of a slow
    synchronous read inside ``coro`` means the loop was blocked for that read; a gap near zero
    means the read ran off the loop.
    """
    gaps: list[float] = []
    last = time.monotonic()

    async def _tick() -> None:
        nonlocal last
        while True:
            await asyncio.sleep(0)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    ticker = asyncio.create_task(_tick())
    # A task, unlike a bare coroutine, always goes through the loop's ready queue.
    work = asyncio.create_task(coro)
    try:
        result = await work
    finally:
        ticker.cancel()
    return result, max(gaps, default=0.0)


def test_active_project_changed_event_does_not_stall_a_concurrent_request(monkeypatch):
    """The event branch's marker read runs off the event loop, so a slow one lengthens this
    request's own wall time but never blocks the loop from running its other turns."""
    from tcip_mcp.project_paths import pin_project_root

    pin_project_root(from_marker=False)

    def slow_read(*, create=False):
        time.sleep(1.0)
        return None

    monkeypatch.setattr(workspace, "active_project_if_present", slow_read)

    async def _post() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            return await client.post(
                "/api/events/app",
                json={"event_type": PANEL_EVENT_ACTIVE_PROJECT_CHANGED, "data": {}},
            )

    resp, gap = asyncio.run(_largest_loop_gap(_post()))
    assert resp.status_code == 200
    assert gap < 0.5


def test_first_requests_bind_does_not_stall_the_loop(monkeypatch):
    """bind_startup_root's own marker read is the same store-bound read the event branch
    makes; a slow one must not stall the loop on the process's very first request either."""
    project_paths.restore_binding(None)

    def slow_read(*, create=False):
        time.sleep(1.0)
        return None

    monkeypatch.setattr(workspace, "active_project_if_present", slow_read)

    async def _get() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            return await client.get("/health")

    resp, gap = asyncio.run(_largest_loop_gap(_get()))
    assert resp.status_code == 200
    assert gap < 0.5


def test_event_branch_reports_the_shared_marker_problem_text(monkeypatch):
    """The event branch answers a non-adoptable marker through the one predicate every other
    reader of it shares, rather than re-encoding its own copy of the same fold."""
    monkeypatch.setattr(workspace, "active_project_if_present", lambda create=False: None)
    monkeypatch.setattr(
        workspace, "marker_problem", lambda create=False: "a distinctive marker problem"
    )

    resp = TestClient(app, base_url="http://127.0.0.1").post(
        "/api/events/app",
        json={"event_type": PANEL_EVENT_ACTIVE_PROJECT_CHANGED, "data": {}},
    )
    assert resp.json()["platform_root_problem"] == "a distinctive marker problem"


def test_concurrent_first_requests_on_separate_loops_bind_once(tmp_path, monkeypatch):
    """Eight clients on eight threads, each request on its own event loop, all reach the
    startup bind at once: every one is answered and the root is bound once, so the
    serialization the middleware applies holds across loops and threads, not only within
    one loop."""
    from concurrent.futures import ThreadPoolExecutor

    import tcip_store

    from tcip_mcp.project_paths import project_root

    ws = tmp_path / "ws"
    proj = ws / "elderberry_cyme_bloom"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    monkeypatch.delenv("TCIP_PROJECT_ROOT", raising=False)
    project_paths.restore_binding(None)
    tcip_store.replace(workspace.active_project_key(), "elderberry_cyme_bloom")

    def get(_):
        return TestClient(app, base_url="http://127.0.0.1").get("/health").status_code

    with ThreadPoolExecutor(max_workers=8) as ex:
        statuses = list(ex.map(get, range(8)))

    assert statuses == [200] * 8
    assert project_root().resolve() == proj.resolve()
