"""Report an agent harness's session start to the agent terminal and print what its agent is told.

The harness hands the hook its ``SessionStart`` event as JSON on stdin: Claude Code's and Codex's
name the conversation ``session_id`` and how the session started ``source``, and Antigravity's,
whose hook fires for a new conversation only, name it ``conversationId``. The report goes to the
session start route the argument names or, with none, to that of the terminal session
``TCIP_TERMINAL_SESSION`` names on the backend serving the workspace ``TCIP_WORKSPACE`` names. The
orientation the route answers is printed in the shape that harness reads from a hook's stdout,
and nothing is printed when it answers none.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, TypeAdapter

from tcip_web.terminal import SessionStartSource

if TYPE_CHECKING:
    from pathlib import Path


class ClaudeOrCodexStart(BaseModel):
    """The part of Claude Code's and Codex's ``SessionStart`` hook input this command reads."""

    conversation: str = Field(alias="session_id", min_length=1)
    source: SessionStartSource

    def output(self, text: str) -> dict:
        """Return ``text`` as the ``additionalContext`` either harness gives its agent."""
        return {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": text,
            }
        }


class AntigravityStart(BaseModel):
    """The part of Antigravity's ``SessionStart`` hook input this command reads."""

    conversation: str = Field(alias="conversationId", min_length=1)

    @property
    def source(self) -> Literal["startup"]:
        """How the session started: Antigravity's hook fires for a new conversation only."""
        return "startup"

    def output(self, text: str) -> dict:
        """Return ``text`` as the system message Antigravity injects at the session's start."""
        return {"injectSteps": [{"systemMessage": {"systemMessage": text}}]}


_START: TypeAdapter[ClaudeOrCodexStart | AntigravityStart] = TypeAdapter(
    ClaudeOrCodexStart | AntigravityStart
)


def session_start_url(workspace: Path, session: str) -> str:
    """Return the URL of terminal session ``session``'s start route on ``workspace``'s backend.

    Raises what :func:`~tcip_mcp.web_client.backend_url` raises for a workspace whose backend
    recorded no port.
    """
    from tcip_mcp.web_client import backend_url
    from tcip_web.routes.terminal import router

    return backend_url(
        workspace, router.url_path_for("session_start", session_id=session)
    )


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    """Report the session start the hook's stdin states and print the orientation answered.

    Returns 0 once the backend answered. Raises what ``httpx.post`` raises for an unreachable
    backend, ``httpx.HTTPStatusError`` for an answer that is not 2xx, ``ValidationError`` for an
    input neither harness shape reads, and ``KeyError`` naming ``TCIP_TERMINAL_SESSION`` when no
    URL is given and it is unset.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], prog=prog
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="the session start route to report to; omitted, the route of the "
        "session TCIP_TERMINAL_SESSION names on the backend serving TCIP_WORKSPACE",
    )
    url = parser.parse_args(argv).url
    start = _START.validate_json(sys.stdin.read())

    import httpx

    from tcip_web.routes.terminal import SessionOrientation, SessionStartReport

    if url is None:
        from tcip_mcp.agent_identity import TERMINAL_SESSION_ENV
        from tcip_mcp.workspace import workspace_from_environment
        from tcip_store import bind

        bind()
        url = session_start_url(
            workspace_from_environment(), os.environ[TERMINAL_SESSION_ENV]
        )
    report = SessionStartReport(
        conversation=start.conversation, source=start.source
    )
    answer = httpx.post(url, json=report.model_dump(), timeout=None)
    answer.raise_for_status()
    text = SessionOrientation.model_validate_json(answer.text).text
    if text is not None:
        print(json.dumps(start.output(text)))
    return 0
