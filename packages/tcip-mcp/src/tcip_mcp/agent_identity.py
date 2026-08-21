"""Which agent harness this MCP server process is serving, and the session it minted for it.

A stdio MCP server runs one process per connected client, and the client declares what it is in
the initialize handshake (``client_info``: a name and a version, such as ``claude-code 2.1.238``
or ``codex-mcp-client 0.147.0``). This module keeps that declaration for the life of one server
run, beside a session id the server mints itself, and projects the pair onto every record the
process writes: the audit line, the statement records, and the headers of the one HTTP push the
tools make. The in-app terminal's own session id, when the web backend passed it down through
``TCIP_TERMINAL_SESSION``, rides along as a correlation.

Every value here is a declaration by software, recorded as declared. Nothing verifies it,
nothing refuses on it, and none of it says who the person at the keyboard was; that stays a
declared name until authentication exists. What it does say, honestly, is which harness a record
came in through in ordinary use and which records one session wrote. A process that never
completed a handshake (the web backend importing the tools, a script, a test) has no identity,
and its records carry none rather than a guessed one.
"""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote

if TYPE_CHECKING:
    from mcp.server.context import CallNext, HandlerResult, ServerRequestContext

logger = logging.getLogger(__name__)

TERMINAL_SESSION_ENV = "TCIP_TERMINAL_SESSION"
"""The in-app terminal's session id, set by the web backend on the agent process it spawns and
inherited by the MCP server that agent launches. A correlation, not a credential: any launcher can
set it, so a record carrying it says only what the launcher declared."""

RECORD_FIELDS = (
    "agent_client_name", "agent_client_version", "agent_session", "terminal_session",
    "harness_session", "harness_effort_at_connect",
)
"""The fields a record gains, in the order a statement's field tuple lists them."""

HEADERS = {
    "agent_client_name": "X-TCIP-Agent-Client-Name",
    "agent_client_version": "X-TCIP-Agent-Client-Version",
    "agent_session": "X-TCIP-Agent-Session",
    "terminal_session": "X-TCIP-Terminal-Session",
    "harness_session": "X-TCIP-Harness-Session",
    "harness_effort_at_connect": "X-TCIP-Harness-Effort-At-Connect",
}
"""The header each field travels under on the tools' HTTP push; the backend reads the same map."""

HARNESS_EXPORTS: dict[str, dict[str, str]] = {
    "claude-code": {
        "harness_session": "CLAUDE_CODE_SESSION_ID",
        "harness_effort_at_connect": "CLAUDE_EFFORT",
    },
}
"""What each harness, by the name it declares, exports to the MCP servers it spawns: a session id
(for Claude Code, the key to the transcript it keeps, where the model that actually ran is
written) and the effort it was running at when it spawned the server. Both are read once, at the
handshake: a child's environment does not follow a later change in the harness, so the effort is
a connect-time snapshot and is named as one. Read only under the declaring harness's own entry,
so a harness that declares another name and passes an enclosing harness's variables through is
not attributed them; a client declaring the same name is, which is the trusted-user residual.
Observed by execution for a fresh Claude Code 2.1.238 launch (nested launches, ``--resume`` and
the SDK entry point were not probed); Codex 0.147.0 scrubs the environment its servers receive
and exports nothing of the kind, so a Codex session records these as absent."""


@dataclass(frozen=True)
class AgentIdentity:
    """What the connecting harness declared, the session this server minted, the terminal session
    the launcher declared, and what the harness exported about itself, each absent when not
    declared."""

    client_name: str
    client_version: str
    session: str
    terminal_session: str | None
    harness_session: str | None = None
    harness_effort_at_connect: str | None = None

    def fields(self) -> dict[str, Any]:
        return {
            "agent_client_name": self.client_name,
            "agent_client_version": self.client_version,
            "agent_session": self.session,
            "terminal_session": self.terminal_session,
            "harness_session": self.harness_session,
            "harness_effort_at_connect": self.harness_effort_at_connect,
        }


_current: AgentIdentity | None = None


def begin(client_name: str, client_version: str) -> AgentIdentity:
    """Record the handshake's declaration and mint this run's session id.

    The holder is process-wide because the server runs on stdio, one connection per process, and
    a record may be written from any thread of it (a tool body on a worker thread, the training
    envelope on its own background thread). The first handshake of a run therefore holds for the
    whole run: a record written by a thread that started under it is never re-attributed
    midway. A later handshake in the same run has no meaning on stdio and is logged, not applied.
    """
    global _current
    if _current is not None:
        logger.warning(
            "a second MCP handshake (%s %s) reached a server already serving %s %s; the identity "
            "of the first connection stands for this run",
            client_name, client_version, _current.client_name, _current.client_version,
        )
        return _current
    exports = HARNESS_EXPORTS.get(client_name, {})
    _current = AgentIdentity(
        client_name=client_name,
        client_version=client_version,
        session="mcp_" + secrets.token_hex(8),
        terminal_session=os.environ.get(TERMINAL_SESSION_ENV) or None,
        harness_session=os.environ.get(exports.get("harness_session", "")) or None,
        harness_effort_at_connect=os.environ.get(exports.get("harness_effort_at_connect", ""))
        or None,
    )
    return _current


def end() -> None:
    """Forget the identity; called when the server run that established it ends."""
    global _current
    _current = None


def current() -> AgentIdentity | None:
    """The identity this run established, or ``None`` outside a completed handshake."""
    return _current


def audit_fields() -> dict[str, Any]:
    """What an audit line carries: nothing outside a session, so the entry keeps its shape, else
    the identity with ``terminal_session`` present only when one was declared."""
    if _current is None:
        return {}
    return {key: value for key, value in _current.fields().items() if value is not None}


def statement_fields() -> dict[str, Any]:
    """What a statement record carries: every field, ``None`` where nothing was declared, because
    a statement's field set is fixed by its tuple and hashed by the confirmation."""
    if _current is None:
        return {field: None for field in RECORD_FIELDS}
    return _current.fields()


def http_headers() -> dict[str, str]:
    """The identity as request headers for the tools' HTTP push; empty outside a session.

    A declared name or version is any string, and a header value must be ASCII without control
    characters, so each value travels percent-encoded (letters, digits and ``-._~`` bare) and the
    backend decodes it; a push never fails on what a harness chose to call itself.
    """
    return {HEADERS[key]: quote(str(value), safe="") for key, value in audit_fields().items()}


def fields_from_headers(headers: Mapping[str, str]) -> dict[str, Any]:
    """The identity a request declared through :data:`HEADERS`, ``None`` per field not sent,
    decoded from the percent-encoding :func:`http_headers` applies.

    Read by the backend. A header is a declaration by the sender like everything else here; the
    backend records it as sent and never acts on it.
    """
    return {
        field: unquote(headers.get(header) or "") or None for field, header in HEADERS.items()
    }


async def record_connecting_client(
    ctx: ServerRequestContext[Any, Any], call_next: CallNext
) -> HandlerResult:
    """Server middleware: take the client's declaration on the first message that carries it.

    The session carries ``client_params`` from the end of ``initialize`` on. The first message
    after that is ``notifications/initialized`` for a client that follows the protocol's ordering,
    but the SDK serves a request that arrives before the notification (the spec says SHOULD NOT,
    not MUST NOT), and Antigravity was observed to make its tool calls without the identity ever
    being taken by a notification-only capture. So the capture is keyed on the declaration being
    present and not yet taken, whatever the method, and is in place before any record is written.
    A client that initialized without declaring itself leaves no identity.
    """
    if _current is None and ctx.session.client_params is not None:
        info = ctx.session.client_params.client_info
        begin(info.name, info.version)
    return await call_next(ctx)


@asynccontextmanager
async def session_lifespan(server: object) -> AsyncIterator[dict[str, Any]]:
    """Scope the identity to one server run, so nothing outlives the connection it came from."""
    try:
        yield {}
    finally:
        end()
