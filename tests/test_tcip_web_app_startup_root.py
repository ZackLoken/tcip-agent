"""Tests for tcip_web.app's platform-state root pin: a served app's own responsibility,
never an importer's, and off the event loop when the marker read can block."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from tcip_mcp import workspace
from tcip_mcp.web_client import PANEL_EVENT_ACTIVE_PROJECT_CHANGED
from tcip_web.app import app


@pytest.fixture(autouse=True)
def _reset_startup_bound(monkeypatch):
    """Every test here exercises the once-per-process startup pin from a clean slate,
    regardless of whether an earlier test in the same worker already bound it."""
    import tcip_web.app as app_module

    monkeypatch.setattr(app_module, "_startup_root_bound", False)


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
    tcip_store.replace(workspace.active_project_key(), "elderberry_cyme_bloom")

    resp = TestClient(app, base_url="http://127.0.0.1").get("/health")

    assert resp.status_code == 200
    assert project_root().resolve() == proj.resolve()


def test_lifespan_binds_before_rehydrate_reads_a_registry(tmp_path, monkeypatch):
    import tcip_store

    from tcip_web import jobstore
    from tcip_web.routes import inference

    ws = tmp_path / "ws"
    proj = ws / "elderberry_cyme_bloom"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    monkeypatch.delenv("TCIP_PROJECT_ROOT", raising=False)
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


def test_active_project_changed_event_does_not_stall_a_concurrent_request(monkeypatch):
    """The marker read and its rehydrates run off the event loop, so a slow one stalls no
    other request served by the same process."""

    def slow_read(*, create=False):
        time.sleep(1.0)
        return None

    monkeypatch.setattr(workspace, "active_project_if_present", slow_read)

    async def _drive() -> tuple[httpx.Response, float]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            event_task = asyncio.create_task(
                client.post(
                    "/api/events/app",
                    json={"event_type": PANEL_EVENT_ACTIVE_PROJECT_CHANGED, "data": {}},
                )
            )
            await asyncio.sleep(0.1)
            t0 = time.monotonic()
            health = await client.get("/health")
            elapsed = time.monotonic() - t0
            await event_task
            return health, elapsed

    health, elapsed = asyncio.run(_drive())
    assert health.status_code == 200
    assert elapsed < 0.5


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
