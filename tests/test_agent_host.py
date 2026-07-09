"""Unit tests for the chat sidecar protocol layer (pure functions)."""

from __future__ import annotations

import json

from tcip_web import agent_host


def test_resolve_command_uses_override(monkeypatch):
    monkeypatch.setenv("TCIP_CHAT_SIDECAR_CMD", "python fake.py")
    assert agent_host.resolve_sidecar_command()[-1] == "fake.py"


def test_resolve_command_none_when_cli_absent(monkeypatch):
    monkeypatch.delenv("TCIP_CHAT_SIDECAR_CMD", raising=False)
    monkeypatch.setenv("TCIP_CHAT_CLI", "definitely-not-a-real-cli-xyz-123")
    assert agent_host.resolve_sidecar_command() is None


def test_chat_status_available_with_override(monkeypatch):
    monkeypatch.setenv("TCIP_CHAT_SIDECAR_CMD", "python fake.py")
    assert agent_host.chat_status() == {"available": True}


def test_chat_status_unavailable(monkeypatch):
    monkeypatch.delenv("TCIP_CHAT_SIDECAR_CMD", raising=False)
    monkeypatch.setenv("TCIP_CHAT_CLI", "definitely-not-a-real-cli-xyz-123")
    status = agent_host.chat_status()
    assert status["available"] is False
    assert "reason" in status


def test_encoders_round_trip():
    user = json.loads(agent_host.encode_user_message("hello"))
    assert user == {"type": "user", "message": {"role": "user", "content": "hello"}}

    perm = json.loads(agent_host.encode_permission_response("perm-1", True, "ok"))
    assert perm == {"type": "permission_response", "tool_use_id": "perm-1", "approved": True, "reason": "ok"}

    assert json.loads(agent_host.encode_interrupt()) == {"type": "interrupt"}


def test_translate_init_to_running():
    out = agent_host.translate_event({"type": "system", "subtype": "init"})
    assert out == [{"type": "session_state", "state": "running"}]


def test_translate_assistant_text_and_tool_use_real_nested_shape():
    # The real CLI nests blocks under message.content.
    raw = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "tool_use", "name": "launch_training", "input": {"epochs": 10}},
            ]
        },
    }
    out = agent_host.translate_event(raw)
    assert out[0] == {"type": "assistant_delta", "text": "hi"}
    assert out[1]["type"] == "tool_use"
    assert out[1]["name"] == "launch_training"
    assert "epochs" in out[1]["input_summary"]


def test_translate_assistant_flat_content_still_works():
    # Backward-compatible with a flat top-level content array.
    out = agent_host.translate_event(
        {"type": "assistant", "content": [{"type": "text", "text": "hi"}]}
    )
    assert out == [{"type": "assistant_delta", "text": "hi"}]


def test_translate_user_event_tool_result():
    # Tool results come back as blocks in a user-type event.
    raw = {
        "type": "user",
        "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": True}]},
    }
    assert agent_host.translate_event(raw) == [{"type": "tool_result", "name": None, "ok": False}]


def test_translate_user_text_echo_is_ignored():
    # A user event that's just the echoed prompt (string content) yields nothing.
    assert agent_host.translate_event({"type": "user", "message": {"content": "hello"}}) == []


def test_translate_stream_text_delta():
    raw = {
        "type": "stream_event",
        "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "tok"}},
    }
    assert agent_host.translate_event(raw) == [{"type": "assistant_delta", "text": "tok"}]


def test_translate_top_level_tool_events():
    tu = agent_host.translate_event(
        {"type": "tool_use", "tool_name": "run_inference", "tool_input": {"x": 1}}
    )
    assert tu[0]["name"] == "run_inference"
    tr = agent_host.translate_event({"type": "tool_result", "tool_name": "run_inference", "is_error": True})
    assert tr[0] == {"type": "tool_result", "name": "run_inference", "ok": False}


def test_translate_permission_request():
    raw = {
        "type": "control_request",
        "request_type": "permission",
        "tool_name": "cancel_training",
        "tool_input": {"run_id": "abc"},
        "tool_use_id": "perm-9",
    }
    out = agent_host.translate_event(raw)
    assert out[0]["type"] == "permission_request"
    assert out[0]["request_id"] == "perm-9"
    assert out[0]["tool"] == "cancel_training"


def test_translate_result_to_turn_done():
    assert agent_host.translate_event({"type": "result", "subtype": "success"}) == [
        {"type": "turn_done", "stop_reason": "success"}
    ]


def test_translate_unknown_event_is_ignored():
    assert agent_host.translate_event({"type": "something_new"}) == []
