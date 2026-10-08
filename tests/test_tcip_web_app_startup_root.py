"""Tests for the web backend's start: ``python -m tcip_web``'s ``main`` resolves the workspace once,
publishes the port under it and configures the backend with it, refusing an unset
``TCIP_WORKSPACE``; the lifespan opens the project the last-opened pointer names; every test runs
its backend on its own scratch workspace."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tcip_mcp import workspace
from tcip_web.app import app
from tcip_web.state import store


def test_every_test_starts_the_backend_on_its_own_scratch_workspace(tmp_path):
    assert store.workspace == tmp_path.parent
    assert store.image_roots == ()


def test_the_backend_refuses_to_start_with_no_workspace(monkeypatch):
    import tcip_web.__main__ as entry

    served: list = []
    monkeypatch.setattr(entry.uvicorn, "run", lambda *a, **k: served.append(a))
    monkeypatch.delenv("TCIP_WORKSPACE")

    with pytest.raises(ValueError, match="TCIP_WORKSPACE"):
        entry.main()
    assert served == []


def test_main_publishes_the_port_and_configures_the_backend_on_one_workspace(
    tmp_path_factory, monkeypatch,
):
    """The port the backend binds is the port a tool process resolves under the same workspace."""
    import tcip_web.__main__ as entry
    import tcip_store
    from tcip_mcp.web_client import resolve_web_port

    elsewhere = tmp_path_factory.mktemp("elsewhere")
    extra = tmp_path_factory.mktemp("extra")
    served: list = []
    monkeypatch.setattr(entry.uvicorn, "run", lambda *a, **k: served.append(k["port"]))
    monkeypatch.setenv("TCIP_WORKSPACE", str(elsewhere))
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(extra))
    monkeypatch.setenv("TCIP_WEB_PORT", "0")

    entry.main()

    tcip_store.bind()
    assert store.workspace == elsewhere.resolve()
    assert store.image_roots == (extra.resolve(),)
    assert served[0] != 0
    assert resolve_web_port(elsewhere.resolve()) == served[0]


def test_the_lifespan_opens_the_project_the_last_opened_pointer_names(made):
    workspace.write_last_opened(made.root.parent, made.id)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/projects").json()["open_id"] == made.id
        assert store.opened == made
