"""G5b trust-boundary hardening: Host allow-list, WS Origin check, insecure-bind refusal."""

from __future__ import annotations

import socket

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_trusted_host_rejects_foreign_host(client: TestClient) -> None:
    # Default TestClient Host ("testserver") is allowed; a foreign Host (DNS-rebinding) is not.
    assert client.get("/health").status_code == 200
    assert client.get("/health", headers={"host": "evil.example.com"}).status_code == 400


def test_ws_state_allows_missing_origin(client: TestClient) -> None:
    # A non-browser client sends no Origin — allowed, and gets the initial snapshot.
    with client.websocket_connect("/ws/state") as ws:
        assert ws.receive_json()["type"] == "state_snapshot"


def test_ws_state_allows_local_origin(client: TestClient) -> None:
    with client.websocket_connect("/ws/state", headers={"origin": "http://127.0.0.1:8765"}) as ws:
        assert ws.receive_json()["type"] == "state_snapshot"


def test_ws_state_rejects_cross_site_origin(client: TestClient) -> None:
    # A page on another site must not be able to open a state socket and read paths.
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/state", headers={"origin": "http://evil.example.com"}):
            pass


def test_is_loopback_host() -> None:
    from tcip_web.paths import is_loopback_host

    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("localhost")
    assert is_loopback_host("::1")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("192.168.1.5")


def test_refuse_insecure_bind() -> None:
    import tcip_web.__main__ as m

    m._refuse_insecure_bind("127.0.0.1")  # loopback -> no refusal
    with pytest.raises(SystemExit):
        m._refuse_insecure_bind("0.0.0.0")  # exposed, no opt-in -> refused


def test_refuse_insecure_bind_override(monkeypatch) -> None:
    import tcip_web.__main__ as m

    monkeypatch.setenv("TCIP_WEB_ALLOW_INSECURE", "1")
    m._refuse_insecure_bind("0.0.0.0")  # explicit opt-in -> allowed


def test_pick_port_finds_free_when_taken() -> None:
    import tcip_web.__main__ as m

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        taken = s.getsockname()[1]
        got = m._pick_port("127.0.0.1", taken)  # requested port is occupied
        assert got != taken and got > 0


def test_ws_inference_stream_rejects_cross_site_origin(client: TestClient) -> None:
    # The inference progress stream must reject a cross-site opener (it echoes job state).
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/api/inference/jobs/does-not-exist/stream",
            headers={"origin": "http://evil.example.com"},
        ):
            pass


def test_ws_training_stream_rejects_cross_site_origin(client: TestClient) -> None:
    # The training metrics stream takes a project_root and tails metrics.jsonl — a
    # path-shaped cross-site file read if left ungated.
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/api/training/runs/does-not-exist/stream?project_root=/tmp",
            headers={"origin": "http://evil.example.com"},
        ):
            pass
