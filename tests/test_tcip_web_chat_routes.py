"""Integration tests for the chat routes, driving the scripted fake sidecar.

The fake sidecar (``tests/fake_chat_sidecar.py``) is a test double at the process
boundary — injected via ``TCIP_CHAT_SIDECAR_CMD`` — so the spawn → reader-thread →
transcript/WS plumbing is exercised without the real ``claude`` CLI.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app
from tcip_web.routes import chat

FAKE = Path(__file__).parent / "fake_chat_sidecar.py"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _isolate_chat(tmp_path, monkeypatch):
    """Route transcripts to tmp and point the sidecar at the fake process."""
    chat_dir = tmp_path / "chat"
    chat_dir.mkdir()
    monkeypatch.setattr(chat, "_chat_dir", lambda project_root: chat_dir)
    monkeypatch.setenv("TCIP_CHAT_SIDECAR_CMD", f"{sys.executable} {FAKE}")
    yield
    chat.shutdown_all()


def _types(messages: list[dict]) -> list[str]:
    return [m.get("type") for m in messages]


def _poll_for(client: TestClient, sid: str, event_type: str, timeout: float = 15.0) -> list[dict]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        messages = client.get(f"/api/chat/sessions/{sid}/messages").json()["messages"]
        if any(m.get("type") == event_type for m in messages):
            return messages
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {event_type!r}; got {_types(messages)}")


def test_status_available_with_fake(client):
    assert client.get("/api/chat/status").json() == {"available": True}


def test_status_unavailable_without_cli(client, monkeypatch):
    monkeypatch.delenv("TCIP_CHAT_SIDECAR_CMD", raising=False)
    monkeypatch.setenv("TCIP_CHAT_CLI", "definitely-not-a-real-cli-xyz-123")
    body = client.get("/api/chat/status").json()
    assert body["available"] is False
    assert "reason" in body


def test_create_and_list_session(client):
    sid = client.post("/api/chat/sessions").json()["session_id"]
    assert sid.startswith("chat_")
    listed = client.get("/api/chat/sessions").json()["sessions"]
    assert any(s["id"] == sid for s in listed)


def test_message_streams_full_turn(client):
    sid = client.post("/api/chat/sessions").json()["session_id"]
    resp = client.post(f"/api/chat/sessions/{sid}/messages", json={"text": "hello agent"})
    assert resp.status_code == 202

    messages = _poll_for(client, sid, "turn_done")
    types = _types(messages)
    # The user echo comes first, then the agent's stream, ending in turn_done.
    assert "user_message" in types
    assert "assistant_delta" in types
    assert "tool_use" in types
    assert "tool_result" in types
    assert types[-1] == "turn_done"
    # The assistant echoed our text.
    assert any(m.get("text") == "You said: hello agent" for m in messages if m.get("type") == "assistant_delta")


def test_permission_flow_allow(client):
    sid = client.post("/api/chat/sessions").json()["session_id"]
    client.post(f"/api/chat/sessions/{sid}/messages", json={"text": "do something danger"})

    messages = _poll_for(client, sid, "permission_request")
    req = next(m for m in messages if m["type"] == "permission_request")
    assert req["tool"] == "cancel_training"

    decision = client.post(
        f"/api/chat/sessions/{sid}/permission",
        json={"request_id": req["request_id"], "decision": "allow"},
    )
    assert decision.status_code == 200
    # After approval the turn completes.
    messages = _poll_for(client, sid, "turn_done")
    assert "tool_result" in _types(messages)


def test_permission_flow_deny(client):
    sid = client.post("/api/chat/sessions").json()["session_id"]
    client.post(f"/api/chat/sessions/{sid}/messages", json={"text": "do something danger"})
    messages = _poll_for(client, sid, "permission_request")
    req = next(m for m in messages if m["type"] == "permission_request")

    client.post(
        f"/api/chat/sessions/{sid}/permission",
        json={"request_id": req["request_id"], "decision": "deny", "note": "no thanks"},
    )
    messages = _poll_for(client, sid, "turn_done")
    # Denied path: the agent acknowledges without a tool_result.
    assert any("won't do that" in (m.get("text") or "") for m in messages)


def test_message_to_unknown_session_404(client):
    resp = client.post("/api/chat/sessions/nope/messages", json={"text": "hi"})
    assert resp.status_code == 404


def test_empty_message_rejected(client):
    sid = client.post("/api/chat/sessions").json()["session_id"]
    resp = client.post(f"/api/chat/sessions/{sid}/messages", json={"text": "   "})
    assert resp.status_code == 400


def test_message_503_when_unavailable(client, monkeypatch):
    sid = client.post("/api/chat/sessions").json()["session_id"]
    monkeypatch.delenv("TCIP_CHAT_SIDECAR_CMD", raising=False)
    monkeypatch.setenv("TCIP_CHAT_CLI", "definitely-not-a-real-cli-xyz-123")
    resp = client.post(f"/api/chat/sessions/{sid}/messages", json={"text": "hi"})
    assert resp.status_code == 503


def test_ws_replays_transcript(client):
    sid = client.post("/api/chat/sessions").json()["session_id"]
    client.post(f"/api/chat/sessions/{sid}/messages", json={"text": "hello agent"})
    _poll_for(client, sid, "turn_done")

    with client.websocket_connect(f"/api/chat/ws/{sid}") as ws:
        seen: list[str] = []
        # Replay is bounded — read until we see turn_done or run out.
        for _ in range(50):
            ev = ws.receive_json()
            seen.append(ev.get("type"))
            if ev.get("type") == "turn_done":
                break
    assert "user_message" in seen
    assert "assistant_delta" in seen
    assert "turn_done" in seen


def test_ws_rejects_unknown_session(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/api/chat/ws/nonexistent"):
            pass


def test_include_context_prepends_gui_state(client, monkeypatch):
    # Capture what gets written to the sidecar to confirm the context block is prepended.
    captured: list[str] = []
    real_encode = chat.agent_host.encode_user_message
    monkeypatch.setattr(
        chat.agent_host,
        "encode_user_message",
        lambda text: captured.append(text) or real_encode(text),
    )
    sid = client.post("/api/chat/sessions").json()["session_id"]
    client.post(
        f"/api/chat/sessions/{sid}/messages",
        json={"text": "what tab am I on?", "include_context": True},
    )
    _poll_for(client, sid, "turn_done")
    assert captured, "encode_user_message was not called"
    assert "[Current GUI context]" in captured[0]
    assert "what tab am I on?" in captured[0]
