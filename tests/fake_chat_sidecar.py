"""A scripted fake agent sidecar for the chat tests — a test double at the PROCESS
boundary, not a code path shipped in the product.

It speaks the same newline-delimited stream-json protocol the real ``claude`` CLI does
(the subset ``tcip_web.agent_host.translate_event`` understands), so the chat session's
spawn → reader-thread → WS/transcript plumbing is exercised end-to-end without the real
CLI (which CI doesn't have). Point ``TCIP_CHAT_SIDECAR_CMD`` at ``python <this file>``.

Behaviour, per user message read from stdin:
  * emit ``system/init`` (once),
  * if the message contains "danger", emit a ``control_request`` permission and BLOCK
    until a ``permission_response`` arrives on stdin, then continue,
  * emit an ``assistant`` text block, a ``tool_use``, a ``tool_result``, and a ``result``.
An ``interrupt`` control message ends the current turn with ``result/interrupted``.
"""

from __future__ import annotations

import json
import sys


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def read_line() -> dict | None:
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return {}
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {}


def _user_text(msg: dict) -> str:
    inner = msg.get("message", {})
    content = inner.get("content", "")
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def main() -> None:
    initialised = False
    while True:
        msg = read_line()
        if msg is None:
            break  # EOF → parent closed stdin
        mtype = msg.get("type")
        if mtype == "interrupt":
            emit({"type": "result", "subtype": "interrupted", "is_error": False})
            continue
        if mtype != "user":
            continue

        if not initialised:
            emit({"type": "system", "subtype": "init", "session_id": "fake", "tools": []})
            initialised = True

        text = _user_text(msg)

        if "danger" in text.lower():
            emit(
                {
                    "type": "control_request",
                    "request_type": "permission",
                    "tool_name": "cancel_training",
                    "tool_input": {"run_id": "abc"},
                    "tool_use_id": "perm-1",
                }
            )
            # Block until the decision arrives (ignore anything that isn't the response).
            approved = False
            while True:
                resp = read_line()
                if resp is None:
                    return
                if resp.get("type") == "permission_response":
                    approved = bool(resp.get("approved"))
                    break
            if not approved:
                emit(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [{"type": "text", "text": "Understood — I won't do that."}]
                        },
                    }
                )
                emit({"type": "result", "subtype": "success", "is_error": False})
                continue

        # Real-CLI shapes: assistant text + tool_use are blocks under message.content;
        # the tool RESULT comes back as a block inside a user-type event.
        emit(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": f"You said: {text}"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "get_project_status",
                            "input": {"project_path": "."},
                        },
                    ]
                },
            }
        )
        emit(
            {
                "type": "user",
                "message": {
                    "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": False}]
                },
            }
        )
        emit({"type": "result", "subtype": "success", "is_error": False})


if __name__ == "__main__":
    main()
