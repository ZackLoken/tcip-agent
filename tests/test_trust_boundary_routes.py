"""The network trust boundary as the app applies it: locality decided per connection, the Host
check and the Origin check.

The TestClient sets the ASGI server address from its base URL (HTTP) or from an absolute
WebSocket URL, which is how a connection through a routable address is simulated in-process. Every
refusal is paired with the legitimate connection the same boundary must still serve.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tcip_web.app import app
from tcip_web.trust_boundary import TrustBoundaryMiddleware

LAN = "http://192.168.1.23:8765"
TERMINAL_WORDS = "interactive agent terminal"


def test_a_routable_arrival_is_refused_whatever_the_environment_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No setting serves a network client: a routable arrival is refused over HTTP and
    WebSocket with every variable an exposure mode once read set, while a loopback request with a
    matching Origin is served."""
    monkeypatch.setenv("TCIP_WEB_ALLOW_INSECURE", "1")
    monkeypatch.setenv("TCIP_WEB_ADVERTISED_HOSTS", "192.168.1.23:8765")
    lan = TestClient(app, base_url=LAN)
    resp = lan.get("/health")
    assert resp.status_code == 403
    assert TERMINAL_WORDS in resp.text
    assert lan.post("/api/subjects/save", json={}, headers={"origin": LAN}).status_code == 403
    with pytest.raises(WebSocketDisconnect) as closed:
        with lan.websocket_connect("ws://192.168.1.23:8765/ws/state"):
            pass
    assert closed.value.code == 1008

    local = TestClient(app, base_url="http://127.0.0.1:8765")
    with local.websocket_connect("ws://127.0.0.1:8765/ws/state",
                                 headers={"origin": "http://127.0.0.1:8765"}) as ws:
        assert ws.receive_json()["type"] == "state_snapshot"


def test_a_connection_from_this_machine_is_served() -> None:
    for base in ("http://127.0.0.1", "http://localhost:8765"):
        assert TestClient(app, base_url=base).get("/health").status_code == 200, base
    # The test client cannot form an IPv6 base URL; the mapped spelling is exercised through the
    # Host header instead, which the same canonical parser reads.
    local = TestClient(app, base_url="http://127.0.0.1")
    assert local.get("/health", headers={"host": "[::ffff:127.0.0.1]:80"}).status_code == 200
    assert local.get("/health", headers={"host": "LOCALHOST."}).status_code == 200


def test_an_arrival_the_backend_cannot_classify_is_refused() -> None:
    assert TestClient(app, base_url="http://testserver").get("/health").status_code == 403


def test_a_refused_arrival_is_named_once_to_the_operator(
    caplog: pytest.LogCaptureFixture,
) -> None:
    lan = TestClient(app, base_url="http://10.9.8.7:8765")
    with caplog.at_level(logging.WARNING, logger="tcip_web.trust_boundary"):
        lan.get("/health")
        lan.get("/health")
    named = [r for r in caplog.records if TERMINAL_WORDS in r.getMessage()]
    assert len(named) == 1


def test_the_host_must_be_a_loopback_name_at_the_arrival_port() -> None:
    local = TestClient(app, base_url="http://127.0.0.1:8765")
    assert local.get("/health", headers={"host": "localhost:8765"}).status_code == 200
    assert local.get("/health", headers={"host": "evil.example.com"}).status_code == 400
    assert local.get("/health", headers={"host": "localhost:9999"}).status_code == 400
    assert local.get("/health", headers={"host": "gui.example:443"}).status_code == 400


def test_a_duplicate_host_header_is_refused() -> None:
    local = TestClient(app, base_url="http://127.0.0.1:8765")
    resp = local.get("/health", headers=[("host", "127.0.0.1:8765"), ("host", "127.0.0.1:8765")])
    assert resp.status_code == 400


def test_the_lifespan_runs_with_no_arrival_to_classify() -> None:
    with TestClient(app, base_url="http://127.0.0.1:8765") as running:
        assert running.get("/health").status_code == 200


async def _accepting_ws_app(scope, receive, send) -> None:
    """A minimal ASGI app with no origin check of its own, so wrapping it in the
    middleware isolates the middleware's own Origin enforcement from any route."""
    await send({"type": "websocket.accept"})
    await send({"type": "websocket.send", "text": "hello"})


def test_the_middleware_itself_refuses_a_foreign_origin_websocket() -> None:
    wrapped = TrustBoundaryMiddleware(_accepting_ws_app)
    client = TestClient(wrapped, base_url="http://127.0.0.1")
    with pytest.raises(WebSocketDisconnect) as closed:
        with client.websocket_connect("ws://127.0.0.1/anything", headers={"origin": "http://evil.example"}):
            pass
    assert closed.value.code == 1008
    assert closed.value.reason == "origin not allowed"
