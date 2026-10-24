"""Report the conversation an agent harness's ``SessionStart`` hook names to the agent terminal.

The harness hands the hook its event as JSON on stdin, its ``session_id`` the conversation's id;
the terminal session is the one ``TCIP_TERMINAL_SESSION`` names, and the backend the one serving
the workspace ``TCIP_WORKSPACE`` names. Nothing is printed on stdout, which a harness adds to its
agent's context.
"""

from __future__ import annotations

import argparse
import os
import sys

from pydantic import BaseModel, Field


class SessionStartEvent(BaseModel):
    """The part of a ``SessionStart`` hook's input this command reads: the conversation's id."""

    session_id: str = Field(min_length=1)


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    """Post the conversation the hook's stdin names to the backend for this terminal session.

    Returns 0 once the backend took it. Raises what ``httpx.post`` raises for an unreachable
    backend, ``httpx.HTTPStatusError`` for an answer that is not 2xx, and ``KeyError`` naming
    ``TCIP_TERMINAL_SESSION`` when it is unset.
    """
    argparse.ArgumentParser(
        description=__doc__.splitlines()[0], prog=prog
    ).parse_args(argv)
    event = SessionStartEvent.model_validate_json(sys.stdin.read())

    import httpx

    from tcip_mcp.agent_identity import TERMINAL_SESSION_ENV
    from tcip_mcp.web_client import backend_url
    from tcip_mcp.workspace import workspace_from_environment
    from tcip_store import bind
    from tcip_web.routes.terminal import ConversationReport, router

    bind()
    path = router.url_path_for(
        "report_conversation", session_id=os.environ[TERMINAL_SESSION_ENV]
    )
    httpx.post(
        backend_url(workspace_from_environment(), path),
        json=ConversationReport(conversation=event.session_id).model_dump(),
    ).raise_for_status()
    return 0
