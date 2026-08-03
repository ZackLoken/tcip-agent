"""Trust-boundary hardening: Host allow-list, WS Origin check, insecure-bind refusal."""

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


def test_inference_launch_confines_checkpoint_to_image_roots(client, tmp_path, monkeypatch) -> None:
    # A locked-down server (TCIP_IMAGE_ROOTS set) must reject a checkpoint outside the allowed
    # roots before it reaches torch.load(weights_only=False), an arbitrary-pickle sink.
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(allowed))
    outside = tmp_path / "evil.pt"
    outside.write_bytes(b"x")
    resp = client.post("/api/inference/launch", json={
        "checkpoint_path": str(outside), "dataset_root": str(allowed),
        "model_name": "baseline", "date": "2026-02-11",
    })
    assert resp.status_code == 403


def test_inference_launch_unconfined_when_no_image_roots(client, tmp_path, monkeypatch) -> None:
    # With no allow-list the guard is a no-op: a missing checkpoint is a 404, never a 403.
    monkeypatch.delenv("TCIP_IMAGE_ROOTS", raising=False)
    resp = client.post("/api/inference/launch", json={
        "checkpoint_path": str(tmp_path / "nope.pt"),
        "dataset_root": str(tmp_path), "model_name": "baseline", "date": "2026-02-11",
    })
    assert resp.status_code == 404


def test_ws_state_allows_missing_origin(client: TestClient) -> None:
    # A non-browser client sends no Origin: allowed, and gets the initial snapshot.
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
    # The training metrics stream takes a project_root and tails metrics.jsonl: a
    # path-shaped cross-site file read if left ungated.
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/api/training/runs/does-not-exist/stream?project_root=/tmp",
            headers={"origin": "http://evil.example.com"},
        ):
            pass


def test_ws_training_stream_confines_project_root_to_allowed_roots(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    # training_stream_ws takes project_root as a query param and passes it straight to
    # _metrics_path; it must be confined the same way get_run_metrics' identical parameter is.
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(allowed))
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            f"/api/training/runs/does-not-exist/stream?project_root={outside}",
        ):
            pass


def test_ws_training_stream_unconfined_when_no_image_roots(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    # The rail must admit valid work: with TCIP_IMAGE_ROOTS unset, the socket still opens and
    # reaches its normal "unknown run" error message rather than being refused by the new guard.
    monkeypatch.delenv("TCIP_IMAGE_ROOTS", raising=False)
    with client.websocket_connect(
        f"/api/training/runs/does-not-exist/stream?project_root={tmp_path}",
    ) as ws:
        msg = ws.receive_json()
        assert msg["type"] == "status"
        assert msg["status"]["error"]
