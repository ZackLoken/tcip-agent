"""Agent sidecar — drive the local Claude Code runtime for the in-app chat.

Per ``docs/chat-popup-design.md`` (Option A), the chat agent *is* the same agent the
project is designed around: we spawn the locally installed ``claude`` CLI headless, in
newline-delimited-JSON streaming mode (``--input-format/--output-format stream-json``),
with its working directory set to the active project so it picks up ``.mcp.json``,
``CLAUDE.md``, and ``.github/skills/`` exactly like an interactive session. It reuses the
operator's existing Claude Code auth (subscription or ``ANTHROPIC_API_KEY``) — tcip-web
never holds a key of its own.

This module owns the transport-level concerns that are worth testing in isolation:

* **preflight** — is a sidecar available? (:func:`resolve_sidecar_command`, :func:`chat_status`)
* **protocol translation** — sidecar stdout events → the minimal chat WS envelope
  (:func:`translate_event`), and the stdin encoders for user messages / permission
  decisions / interrupts.

The spawn command is injectable via ``TCIP_CHAT_SIDECAR_CMD`` so tests point it at a
scripted fake process (a test double at the process boundary, not a prod code path) and
so CI — where the ``claude`` CLI is absent — cleanly reports "unavailable" instead of
failing. The session lifecycle (spawn, reader thread, WS fan-out, kill-on-shutdown) lives
in ``routes/chat.py``.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
from typing import Any, Optional

# Env overrides — the full command wins (tests / power users); otherwise the CLI name.
SIDECAR_CMD_ENV = "TCIP_CHAT_SIDECAR_CMD"
CLI_NAME_ENV = "TCIP_CHAT_CLI"
DEFAULT_CLI = "claude"

# Flags for the real CLI: headless print mode, streaming JSON both ways, verbose so every
# turn event is emitted, and partial messages so assistant text streams token-by-token.
# Kept here (not scattered) so they're easy to tune; override the whole command with
# TCIP_CHAT_SIDECAR_CMD if a future CLI version changes them.
_CLI_STREAM_ARGS = [
    "--print",
    "--input-format",
    "stream-json",
    "--output-format",
    "stream-json",
    "--verbose",
    "--include-partial-messages",
]

_UNAVAILABLE_REASON = (
    "Claude Code is not available. Install the `claude` CLI and sign in "
    "(subscription or ANTHROPIC_API_KEY) to enable the in-app agent chat."
)


def resolve_sidecar_command() -> Optional[list[str]]:
    """Return the argv to spawn the agent sidecar, or ``None`` if none is available.

    Order: an explicit ``TCIP_CHAT_SIDECAR_CMD`` override (used by tests and power
    users), then the ``claude`` CLI if it's on ``PATH``.
    """
    override = os.environ.get(SIDECAR_CMD_ENV, "").strip()
    if override:
        if os.name == "nt":
            # posix=False keeps backslash paths intact; strip the quotes it leaves on
            # quoted (spaces-in-path) tokens so Popen gets clean argv entries.
            return [tok.strip('"') for tok in shlex.split(override, posix=False)]
        return shlex.split(override)
    cli = os.environ.get(CLI_NAME_ENV, DEFAULT_CLI)
    exe = shutil.which(cli)
    if exe:
        return [exe, *_CLI_STREAM_ARGS]
    return None


def chat_status() -> dict:
    """Preflight: ``{available: bool, reason?: str}`` for ``GET /api/chat/status``."""
    if resolve_sidecar_command() is not None:
        return {"available": True}
    return {"available": False, "reason": _UNAVAILABLE_REASON}


# ── stdin encoders (backend → sidecar) ──────────────────────────────────


def encode_user_message(text: str) -> str:
    """A user turn as a stream-json stdin line."""
    return json.dumps({"type": "user", "message": {"role": "user", "content": text}}) + "\n"


def encode_permission_response(request_id: str, approved: bool, note: str = "") -> str:
    """A permission decision as a stream-json stdin line."""
    payload: dict[str, Any] = {
        "type": "permission_response",
        "tool_use_id": request_id,
        "approved": approved,
    }
    if note:
        payload["reason"] = note
    return json.dumps(payload) + "\n"


def encode_interrupt() -> str:
    """An interrupt control message as a stream-json stdin line."""
    return json.dumps({"type": "interrupt"}) + "\n"


# ── stdout translation (sidecar → chat WS envelope) ─────────────────────


def _summarize_input(tool_input: Any, limit: int = 160) -> str:
    """One-line, bounded summary of a tool's input for a compact transcript chip."""
    try:
        text = json.dumps(tool_input, default=str) if not isinstance(tool_input, str) else tool_input
    except (TypeError, ValueError):
        text = str(tool_input)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _content_blocks(raw: dict) -> list:
    """Content blocks of an assistant/user event.

    The real ``claude`` CLI nests them under ``message.content``
    (``{"type":"assistant","message":{"content":[...]}}``); accept a flat top-level
    ``content`` too for robustness.
    """
    msg = raw.get("message")
    if isinstance(msg, dict) and isinstance(msg.get("content"), list):
        return msg["content"]
    content = raw.get("content")
    return content if isinstance(content, list) else []


def translate_event(raw: dict) -> list[dict]:
    """Map one sidecar stdout event to zero or more chat WS envelope events.

    Envelope (per ``chat-popup-design.md`` §3): ``assistant_delta`` / ``tool_use`` /
    ``tool_result`` / ``permission_request`` / ``turn_done`` / ``session_state``. Unknown
    event types translate to nothing (forward-compatible with CLI additions).

    Matches the real CLI's shapes (empirically): assistant text + tool_use arrive as
    blocks under ``assistant.message.content``; tool RESULTS come back as blocks under a
    ``user``-type event's ``message.content``.
    """
    etype = raw.get("type")
    out: list[dict] = []

    if etype == "system" and raw.get("subtype") == "init":
        out.append({"type": "session_state", "state": "running"})

    elif etype == "assistant":
        for block in _content_blocks(raw):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text"):
                out.append({"type": "assistant_delta", "text": block["text"]})
            elif block.get("type") == "tool_use":
                out.append(
                    {
                        "type": "tool_use",
                        "name": block.get("name", "tool"),
                        "input_summary": _summarize_input(block.get("input", {})),
                    }
                )

    elif etype == "user":
        # Tool results are delivered as blocks inside a user-type event (the user's own
        # text echo is ignored — we already surface it via our own user_message event).
        for block in _content_blocks(raw):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                out.append(
                    {
                        "type": "tool_result",
                        "name": None,
                        "ok": not bool(block.get("is_error", False)),
                    }
                )

    elif etype == "stream_event":
        # Token-level deltas when the CLI runs with --include-partial-messages.
        ev = raw.get("event", {}) or {}
        if ev.get("type") == "content_block_delta":
            delta = ev.get("delta", {}) or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                out.append({"type": "assistant_delta", "text": delta["text"]})

    elif etype == "tool_use":  # top-level form (fallback / other clients)
        out.append(
            {
                "type": "tool_use",
                "name": raw.get("tool_name", raw.get("name", "tool")),
                "input_summary": _summarize_input(raw.get("tool_input", raw.get("input", {}))),
            }
        )

    elif etype == "tool_result":  # top-level form (fallback / other clients)
        out.append(
            {
                "type": "tool_result",
                "name": raw.get("tool_name", raw.get("name")),
                "ok": not bool(raw.get("is_error", False)),
            }
        )

    elif etype == "control_request" and raw.get("request_type") == "permission":
        request_id = raw.get("tool_use_id") or raw.get("request_id") or ""
        tool = raw.get("tool_name", raw.get("tool", "tool"))
        out.append(
            {
                "type": "permission_request",
                "request_id": request_id,
                "tool": tool,
                "detail": _summarize_input(raw.get("tool_input", raw.get("detail", {}))),
            }
        )

    elif etype == "result":
        out.append({"type": "turn_done", "stop_reason": raw.get("subtype", "end_turn")})

    return out
