"""The route-level face of the trust boundary: Host check, WS Origin check, path confinement
on the launch and stream routes. The boundary's own policy is covered in test_trust_boundary.py."""

from __future__ import annotations

import socket

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


class _RawHeaderList(list):
    """A duplicate-preserving stand-in for the mapping ``websocket_connect`` expects.

    Passing a plain list of (name, value) pairs keeps two same-named header lines distinct
    through the request encoding, where an ``httpx.Headers`` instance built the same way is
    merged into one comma-joined value before it reaches the ASGI scope. ``setdefault`` is the
    one mapping method the test transport calls, to fill in the WebSocket upgrade headers.
    """

    def setdefault(self, key: str, value: str) -> str:
        low = key.lower()
        for k, v in self:
            if k.lower() == low:
                return v
        self.append((key, value))
        return value


def test_trusted_host_rejects_foreign_host(client: TestClient) -> None:
    # The Host naming the arrival (127.0.0.1) is served; a foreign Host (DNS-rebinding) is not.
    assert client.get("/health").status_code == 200
    assert client.get("/health", headers={"host": "evil.example.com"}).status_code == 400


def test_inference_launch_confines_checkpoint_to_image_roots(
    client, tmp_path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # A checkpoint outside every allowed root must be rejected before it reaches
    # torch.load(weights_only=False), an arbitrary-pickle sink.
    outside = tmp_path_factory.mktemp("outside") / "evil.pt"
    outside.write_bytes(b"x")
    resp = client.post("/api/inference/launch", json={
        "checkpoint_path": str(outside), "dataset_root": str(tmp_path),
        "model_name": "baseline", "date": "2026-02-11",
    })
    assert resp.status_code == 403


def test_inference_launch_unconfined_for_a_checkpoint_inside_the_workspace(client, tmp_path) -> None:
    # With no additive TCIP_IMAGE_ROOTS, a checkpoint under the workspace still clears the guard:
    # a missing checkpoint reaches its own 404, never a 403 from the path check.
    resp = client.post("/api/inference/launch", json={
        "checkpoint_path": str(tmp_path / "nope.pt"),
        "dataset_root": str(tmp_path), "model_name": "baseline", "date": "2026-02-11",
    })
    assert resp.status_code == 404


def test_ws_state_allows_missing_origin(client: TestClient) -> None:
    # A non-browser client sends no Origin: allowed, and gets the initial snapshot.
    with client.websocket_connect("ws://127.0.0.1/ws/state") as ws:
        assert ws.receive_json()["type"] == "state_snapshot"


def test_ws_state_allows_local_origin(client: TestClient) -> None:
    with client.websocket_connect("ws://127.0.0.1/ws/state", headers={"origin": "http://127.0.0.1:8765"}) as ws:
        assert ws.receive_json()["type"] == "state_snapshot"


def test_ws_state_rejects_cross_site_origin(client: TestClient) -> None:
    # A page on another site must not be able to open a state socket and read paths.
    with pytest.raises(WebSocketDisconnect) as closed:
        with client.websocket_connect("ws://127.0.0.1/ws/state", headers={"origin": "http://evil.example.com"}):
            pass
    assert closed.value.code == 1008
    assert closed.value.reason == "origin not allowed"


def test_ws_state_rejects_duplicate_origin(client: TestClient) -> None:
    # A duplicated Origin header is refused outright, the same way a duplicated Host is.
    duplicated = _RawHeaderList([("origin", "http://127.0.0.1"), ("origin", "http://127.0.0.1")])
    with pytest.raises(WebSocketDisconnect) as closed:
        with client.websocket_connect("ws://127.0.0.1/ws/state", headers=duplicated):
            pass
    assert closed.value.code == 1008
    assert closed.value.reason == "origin not allowed"


def test_ws_panel_rejects_cross_site_origin(client: TestClient) -> None:
    # The panel event socket has no origin test today; it is Origin-checked like every other.
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "ws://127.0.0.1/ws/panel/meta", headers={"origin": "http://evil.example.com"}
        ):
            pass


def test_is_loopback_host() -> None:
    from tcip_web.trust_boundary import is_loopback_host

    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("localhost")
    assert is_loopback_host("::1")
    assert is_loopback_host("[::ffff:127.0.0.1]")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("192.168.1.5")


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
            "ws://127.0.0.1/api/inference/jobs/does-not-exist/stream",
            headers={"origin": "http://evil.example.com"},
        ):
            pass


def test_ws_training_stream_rejects_cross_site_origin(client: TestClient, tmp_path) -> None:
    # project_root is confined so the origin, not the path guard, refuses this connect.
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            f"ws://127.0.0.1/api/training/runs/does-not-exist/stream?project_root={tmp_path}",
            headers={"origin": "http://evil.example.com"},
        ):
            pass


def test_ws_training_stream_confines_project_root_to_allowed_roots(
    client: TestClient, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # training_stream_ws's project_root must be confined the same way meta.py's report
    # routes confine their own project_root parameter.
    outside = tmp_path_factory.mktemp("outside")
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            f"ws://127.0.0.1/api/training/runs/does-not-exist/stream?project_root={outside}",
        ):
            pass


def test_ws_training_stream_unconfined_for_a_project_root_inside_the_workspace(
    client: TestClient, tmp_path
) -> None:
    # The rail must admit valid work: a project_root under the workspace still opens the socket
    # and reaches its normal "unknown run" error message rather than being refused by the guard.
    with client.websocket_connect(
        f"ws://127.0.0.1/api/training/runs/does-not-exist/stream?project_root={tmp_path}",
    ) as ws:
        msg = ws.receive_json()
        assert msg["type"] == "status"
        assert msg["status"] is None
        assert msg["error"]
