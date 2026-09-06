"""The network trust boundary as the app applies it: exposure decided per connection, the Host
check, the WebSocket Origin check, and the operator's opt-in.

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


def test_a_connection_through_a_routable_address_is_refused_until_the_operator_opts_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TCIP_WEB_ALLOW_INSECURE", raising=False)
    lan = TestClient(app, base_url=LAN)
    resp = lan.get("/health")
    assert resp.status_code == 403
    assert TERMINAL_WORDS in resp.text
    with pytest.raises(WebSocketDisconnect) as closed:
        with lan.websocket_connect("ws://192.168.1.23:8765/ws/state"):
            pass
    assert closed.value.code == 1008

    monkeypatch.setenv("TCIP_WEB_ALLOW_INSECURE", "1")
    assert lan.get("/health").status_code == 200
    with lan.websocket_connect("ws://192.168.1.23:8765/ws/state") as ws:
        assert ws.receive_json()["type"] == "state_snapshot"


def test_a_connection_from_this_machine_is_served_without_any_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TCIP_WEB_ALLOW_INSECURE", raising=False)
    for base in ("http://127.0.0.1", "http://localhost:8765"):
        assert TestClient(app, base_url=base).get("/health").status_code == 200, base
    # The test client cannot form an IPv6 base URL; the mapped spelling is exercised through the
    # Host header instead, which the same canonical parser reads.
    local = TestClient(app, base_url="http://127.0.0.1")
    assert local.get("/health", headers={"host": "[::ffff:127.0.0.1]:80"}).status_code == 200
    assert local.get("/health", headers={"host": "LOCALHOST."}).status_code == 200


def test_an_arrival_the_backend_cannot_classify_is_refused() -> None:
    assert TestClient(app, base_url="http://testserver").get("/health").status_code == 403


def test_a_refused_exposed_arrival_is_named_once_to_the_operator(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("TCIP_WEB_ALLOW_INSECURE", raising=False)
    lan = TestClient(app, base_url="http://10.9.8.7:8765")
    with caplog.at_level(logging.WARNING, logger="tcip_web.trust_boundary"):
        lan.get("/health")
        lan.get("/health")
    named = [r for r in caplog.records if TERMINAL_WORDS in r.getMessage()]
    assert len(named) == 1


def test_the_host_must_name_this_backend_as_reached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TCIP_WEB_ALLOW_INSECURE", "1")
    lan = TestClient(app, base_url=LAN)
    assert lan.get("/health").status_code == 200
    assert lan.get("/health", headers={"host": "evil.example.com"}).status_code == 400
    assert lan.get("/health", headers={"host": "192.168.1.23:9999"}).status_code == 400
    local = TestClient(app, base_url="http://127.0.0.1:8765")
    assert local.get("/health", headers={"host": "localhost:8765"}).status_code == 200
    assert local.get("/health", headers={"host": "evil.example.com"}).status_code == 400


def test_an_advertised_name_is_served_only_under_the_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A same-machine reverse proxy arrives on loopback with the proxy's name as Host: advertising
    that name declares network exposure, so it is inert without the opt-in."""
    monkeypatch.setenv("TCIP_WEB_ADVERTISED_HOSTS", "gui.example:443, orchardbox.local")
    monkeypatch.delenv("TCIP_WEB_ALLOW_INSECURE", raising=False)
    local = TestClient(app, base_url="http://127.0.0.1:8765")
    assert local.get("/health", headers={"host": "gui.example:443"}).status_code == 400
    monkeypatch.setenv("TCIP_WEB_ALLOW_INSECURE", "1")
    assert local.get("/health", headers={"host": "gui.example:443"}).status_code == 200
    assert local.get("/health", headers={"host": "orchardbox.local:8765"}).status_code == 200
    assert local.get("/health", headers={"host": "orchardbox.local:9999"}).status_code == 400


def test_a_websocket_origin_must_be_the_requests_own_origin_on_an_exposed_arrival(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TCIP_WEB_ALLOW_INSECURE", "1")
    lan = TestClient(app, base_url=LAN)
    with lan.websocket_connect("ws://192.168.1.23:8765/ws/state",
                               headers={"origin": "http://192.168.1.23:8765"}) as ws:
        assert ws.receive_json()["type"] == "state_snapshot"
    for foreign in ("http://192.168.1.23:3000", "http://evil.example.com", "null"):
        with pytest.raises(WebSocketDisconnect):
            with lan.websocket_connect("ws://192.168.1.23:8765/ws/state",
                                       headers={"origin": foreign}):
                pass


def test_a_duplicate_host_header_is_refused() -> None:
    local = TestClient(app, base_url="http://127.0.0.1:8765")
    resp = local.get("/health", headers=[("host", "127.0.0.1:8765"), ("host", "127.0.0.1:8765")])
    assert resp.status_code == 400


def test_the_lifespan_runs_with_no_arrival_to_classify(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TCIP_WEB_ALLOW_INSECURE", raising=False)
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
