"""Tests for tcip_web.app's platform-state root pin: a served app's own responsibility,
never an importer's, and off the event loop when the marker read can block.

The marker-seeding tests below (``test_first_request_pins_from_the_marker``,
``test_a_binding_set_before_the_first_request_is_not_replaced``,
``test_lifespan_binds_before_rehydrate_reads_a_registry``,
``test_concurrent_first_requests_on_separate_loops_bind_once``), each setting
``TCIP_WORKSPACE`` and asserting a 200 response, are the proof that the startup-under-test
rail admits a bound workspace; no separate admits-valid-work test duplicates that here."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from tcip_mcp import project_paths, workspace
from tcip_mcp.web_client import PANEL_EVENT_ACTIVE_PROJECT_CHANGED
from tcip_web.app import app

_IMPORT_ONLY = """\
import os
before = os.environ.get("TCIP_STATE_ROOT")
from tcip_web.app import app  # noqa: F401
after = os.environ.get("TCIP_STATE_ROOT")
print(before)
print(after)
"""


def test_importing_the_app_alone_pins_nothing(tmp_path, monkeypatch):
    """A repo script, or a test module at collection, that only imports tcip_web.app must not
    silently repin TCIP_STATE_ROOT to whatever the machine's workspace marker names."""
    ws = tmp_path / "ws"
    proj = ws / "elderberry_cyme_bloom"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    workspace.activate_project("elderberry_cyme_bloom")

    env = dict(os.environ)
    env.pop("TCIP_STATE_ROOT", None)
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_ONLY],
        capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    before, after = result.stdout.splitlines()
    assert before == "None"
    assert after == "None"


def _unbind_workspace_default(tmp_path, monkeypatch):
    """Make the workspace's default (``TCIP_WORKSPACE`` unset) resolve under ``tmp_path``, so
    a refusal test run against code without the rail still touches only tmp and never a real
    marker.

    ``tcip_mcp.workspace.DEFAULT_WORKSPACE`` is ``Path.home() / "tcip-projects"`` computed once
    at import, so ``HOME``/``USERPROFILE`` alone do not move it; the module attribute is patched
    directly.
    """
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    monkeypatch.delenv("TCIP_WORKSPACE", raising=False)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setattr(workspace, "DEFAULT_WORKSPACE", fake_home / "tcip-projects")


def test_workspace_unset_under_pytest_refuses_the_first_request(tmp_path, monkeypatch):
    """A pytest process with no ``TCIP_WORKSPACE`` bound must never pin the operator's real
    workspace. Asserts the exception by name so the test also runs against code lacking the
    class.
    """
    _unbind_workspace_default(tmp_path, monkeypatch)
    project_paths.restore_binding(None)

    with pytest.raises(Exception) as excinfo:
        TestClient(app, base_url="http://127.0.0.1").get("/health")
    assert type(excinfo.value).__name__ == "WorkspaceUnsetUnderTest"


def test_workspace_unset_under_pytest_refuses_entering_the_lifespan(tmp_path, monkeypatch):
    """The same refusal fires from ``bind_startup_root``'s own call inside the lifespan, not
    only from the middleware's scope check: entering ``with TestClient(app):`` runs
    ``_lifespan``, which calls ``bind_startup_root`` before any request is served."""
    _unbind_workspace_default(tmp_path, monkeypatch)
    project_paths.restore_binding(None)

    with pytest.raises(Exception) as excinfo:
        with TestClient(app, base_url="http://127.0.0.1"):
            pass
    assert type(excinfo.value).__name__ == "WorkspaceUnsetUnderTest"


_TEST_CLIENT_WITHOUT_PYTEST = """\
from fastapi.testclient import TestClient
from tcip_web.app import app

try:
    TestClient(app, base_url="http://127.0.0.1").get("/health")
except Exception as exc:
    print(type(exc).__name__)
else:
    print("no-exception")
"""


def test_workspace_unset_test_client_signal_without_pytest_refuses(tmp_path):
    """A bare ``TestClient`` request with no ``TCIP_WORKSPACE`` bound is refused even outside a
    pytest process: this subprocess never imports pytest, so only the process-level signal that
    ``starlette.testclient`` has been imported, or the middleware's own scope check, can catch
    it, proving the refusal does not depend on the pytest signal. The two later tests each
    isolate one of those two signals on its own: the module-import signal alone (entering the
    lifespan with no request) and the scope-check signal alone (an ``httpx.ASGITransport``
    client, which never imports ``starlette.testclient``)."""
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    scratch_state = tmp_path / "state"
    scratch_state.mkdir()
    env = dict(os.environ)
    env.pop("TCIP_WORKSPACE", None)
    env["HOME"] = str(fake_home)
    env["USERPROFILE"] = str(fake_home)
    env["TCIP_STATE_ROOT"] = str(scratch_state)

    result = subprocess.run(
        [sys.executable, "-c", _TEST_CLIENT_WITHOUT_PYTEST],
        capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "WorkspaceUnsetUnderTest"


_TEST_CLIENT_CONTEXT_WITHOUT_PYTEST = """\
from fastapi.testclient import TestClient
from tcip_web.app import app

try:
    with TestClient(app, base_url="http://127.0.0.1"):
        pass
except Exception as exc:
    print(type(exc).__name__)
else:
    print("no-exception")
"""


def test_workspace_unset_test_client_import_signal_without_pytest_refuses_the_lifespan(tmp_path):
    """Entering ``with TestClient(app):`` with no request yet runs the lifespan, which calls
    ``bind_startup_root`` with no ASGI scope available: neither the pytest signal nor the
    middleware's scope check can catch that case, so only the process-level signal that
    ``starlette.testclient`` has been imported does. This subprocess never imports pytest,
    proving that signal works on its own; the baseline (code without it) prints
    "no-exception"."""
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    scratch_state = tmp_path / "state"
    scratch_state.mkdir()
    env = dict(os.environ)
    env.pop("TCIP_WORKSPACE", None)
    env["HOME"] = str(fake_home)
    env["USERPROFILE"] = str(fake_home)
    env["TCIP_STATE_ROOT"] = str(scratch_state)

    result = subprocess.run(
        [sys.executable, "-c", _TEST_CLIENT_CONTEXT_WITHOUT_PYTEST],
        capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "WorkspaceUnsetUnderTest"


_HTTPX_ASGI_TRANSPORT_WITHOUT_PYTEST = """\
import asyncio

import httpx
from tcip_web.app import app


async def _get():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        await client.get("/health")


try:
    asyncio.run(_get())
except Exception as exc:
    print(type(exc).__name__)
else:
    print("no-exception")
"""


def test_workspace_unset_asgi_transport_signal_without_pytest_refuses(tmp_path):
    """A request through a bare ``httpx.ASGITransport`` client with no ``TCIP_WORKSPACE`` bound
    is refused even outside a pytest process and with starlette's ``TestClient`` never imported:
    this subprocess uses only ``httpx.AsyncClient`` (``ASGITransport`` implements only the async
    transport interface, verified against the installed httpx), so only the middleware's scope
    check recognising ``ASGITransport``'s default client identity (``("127.0.0.1", 123)``,
    verified against the installed httpx) can catch it."""
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    scratch_state = tmp_path / "state"
    scratch_state.mkdir()
    env = dict(os.environ)
    env.pop("TCIP_WORKSPACE", None)
    env["HOME"] = str(fake_home)
    env["USERPROFILE"] = str(fake_home)
    env["TCIP_STATE_ROOT"] = str(scratch_state)

    result = subprocess.run(
        [sys.executable, "-c", _HTTPX_ASGI_TRANSPORT_WITHOUT_PYTEST],
        capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "WorkspaceUnsetUnderTest"


def test_first_request_pins_from_the_marker(tmp_path, monkeypatch):
    import tcip_store

    from tcip_mcp.project_paths import platform_state_root

    ws = tmp_path / "ws"
    proj = ws / "elderberry_cyme_bloom"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    monkeypatch.delenv("TCIP_STATE_ROOT", raising=False)
    project_paths.restore_binding(None)
    tcip_store.replace(workspace.active_project_key(), "elderberry_cyme_bloom")

    resp = TestClient(app, base_url="http://127.0.0.1").get("/health")

    assert resp.status_code == 200
    assert platform_state_root().resolve() == proj.resolve()


def test_a_binding_set_before_the_first_request_is_not_replaced(tmp_path, monkeypatch):
    """``activate_project`` (source ``adopted``) can run before this process has served
    its first request; the middleware's own bind must leave that binding alone rather than
    resolving the marker itself and overwriting it with source ``marker``."""
    ws = tmp_path / "ws"
    proj = ws / "elderberry_cyme_bloom"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    monkeypatch.delenv("TCIP_STATE_ROOT", raising=False)
    project_paths.restore_binding(None)

    workspace.activate_project("elderberry_cyme_bloom")
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
    monkeypatch.delenv("TCIP_STATE_ROOT", raising=False)
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


def test_a_refused_rehydrate_does_not_block_the_other_two_registries(tmp_path, monkeypatch):
    """One registry's refused rehydrate (an unconformed document) must not skip the other two
    the way one shared try around all three used to: each gets its own try, and the refusal is
    recorded for the workspace status route rather than only logged."""
    import tcip_store

    from tcip_web import jobstore
    from tcip_web.routes import inference, review, tuning

    ws = tmp_path / "ws"
    proj = ws / "elderberry_cyme_bloom"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    monkeypatch.delenv("TCIP_STATE_ROOT", raising=False)
    project_paths.restore_binding(None)
    tcip_store.replace(workspace.active_project_key(), "elderberry_cyme_bloom")
    jobstore._startup_refusals.clear()

    called: list[str] = []
    real_tuning, real_review = tuning.rehydrate_for_current_root, review.rehydrate_for_current_root

    def _fail_inference():
        raise ValueError("boom; no operator door stamps the missing field onto this root")

    def _record_tuning():
        called.append("tuning")
        real_tuning()

    def _record_review():
        called.append("review")
        real_review()

    monkeypatch.setattr(inference, "rehydrate_for_current_root", _fail_inference)
    monkeypatch.setattr(tuning, "rehydrate_for_current_root", _record_tuning)
    monkeypatch.setattr(review, "rehydrate_for_current_root", _record_review)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        resp = client.get("/api/projects")

    assert called == ["tuning", "review"]
    refusals = jobstore.startup_refusals()
    assert len(refusals) == 1
    assert refusals[0]["registry"] == jobstore.INFERENCE_JOBS
    assert "no operator door" in refusals[0]["error"]
    assert resp.json()["job_registry_startup_refusals"] == refusals


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
    from tcip_mcp.project_paths import pin_platform_root

    pin_platform_root(from_marker=False)

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

    from tcip_mcp.project_paths import platform_state_root

    ws = tmp_path / "ws"
    proj = ws / "elderberry_cyme_bloom"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    monkeypatch.delenv("TCIP_STATE_ROOT", raising=False)
    project_paths.restore_binding(None)
    tcip_store.replace(workspace.active_project_key(), "elderberry_cyme_bloom")

    def get(_):
        return TestClient(app, base_url="http://127.0.0.1").get("/health").status_code

    with ThreadPoolExecutor(max_workers=8) as ex:
        statuses = list(ex.map(get, range(8)))

    assert statuses == [200] * 8
    assert platform_state_root().resolve() == proj.resolve()
