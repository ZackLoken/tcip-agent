"""Tests for the web backend's start: ``python -m tcip_web``'s ``main`` resolves the workspace once,
publishes the port under it and configures the backend with it, refusing an unset
``TCIP_WORKSPACE``; the lifespan opens the project the last-opened pointer names; every test runs
its backend on its own scratch workspace.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlencode

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


def _answer(url: str) -> tuple[int, dict]:
    """The status and JSON body ``url`` answers with, a refusal included."""
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as refused:
        return refused.code, json.loads(refused.read())


def test_main_publishes_the_port_and_configures_the_backend_on_one_workspace(
    tmp_path_factory,
):
    """``python -m tcip_web``, run as an operator runs it, records under its workspace the port it
    then serves on, and serves the backend configured with that workspace and its image roots: a
    tool process resolving the port under the same workspace reaches it, the listing names the
    workspace, and an image under the image root is read while one under neither root is
    refused.
    """
    from PIL import Image

    from tcip_mcp.web_client import NoBackendPortError, resolve_web_port

    elsewhere = tmp_path_factory.mktemp("elsewhere").resolve()
    extra = tmp_path_factory.mktemp("extra").resolve()
    outside = tmp_path_factory.mktemp("outside").resolve()
    for root in (extra, outside):
        Image.new("RGB", (8, 8)).save(root / "IMG_0001.png")
    env = {
        **os.environ,
        "TCIP_WORKSPACE": str(elsewhere),
        "TCIP_IMAGE_ROOTS": str(extra),
        "TCIP_WEB_PORT": "0",
    }
    served = subprocess.Popen(
        [sys.executable, "-m", "tcip_web"],
        env=env,
        cwd=str(elsewhere),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 120
        while True:
            assert served.poll() is None, "python -m tcip_web exited early"
            assert time.monotonic() < deadline, "the backend never came up"
            try:
                port = resolve_web_port(elsewhere)
                if _answer(f"http://127.0.0.1:{port}/health")[0] == 200:
                    break
            except (NoBackendPortError, OSError):
                pass
            time.sleep(0.5)
        base = f"http://127.0.0.1:{port}"

        listing = _answer(f"{base}/api/projects")
        admitted, refused = (
            _answer(f"{base}/api/images/bands?{urlencode({'path': image})}")
            for image in (extra / "IMG_0001.png", outside / "IMG_0001.png")
        )
    finally:
        served.terminate()
        served.wait(timeout=30)

    assert port != 0
    assert listing[1]["workspace"] == str(elsewhere)
    assert admitted[0] == 200, admitted
    assert refused[0] != 200, refused


def test_the_lifespan_opens_the_project_the_last_opened_pointer_names(made):
    workspace.write_last_opened(made.root.parent, made.id)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/projects").json()["open_id"] == made.id
        assert store.opened == made
