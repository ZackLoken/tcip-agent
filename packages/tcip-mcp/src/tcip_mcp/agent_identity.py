"""Which agent harness this MCP server process is serving, and the session it minted for it.

A stdio MCP server runs one process per connected client, and the client declares what it is in the
initialize handshake (``client_info``: a name and a version, such as ``claude-code 2.1.238`` or
``codex-mcp-client 0.147.0``). This module keeps that declaration for the life of one server run,
beside a session id the server mints itself, and projects the pair onto its audit-line fields and
its HTTP push headers. The in-app terminal's own session id, when the web backend passed it down
through ``TCIP_TERMINAL_SESSION``, rides along as a correlation.

Every value here is a declaration by software, recorded as declared; nothing verifies it or refuses
on it. A process that never completed a handshake has no identity, and its records carry none.
"""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from mcp.server.context import (
        CallNext,
        HandlerResult,
        ServerRequestContext,
    )

logger = logging.getLogger(__name__)

TERMINAL_SESSION_ENV = "TCIP_TERMINAL_SESSION"
"""The in-app terminal's session id, set by the web backend on the agent process it spawns and
inherited by the MCP server that agent launches. A correlation, not a credential: any launcher can
set it, so a record carrying it says only what the launcher declared."""


class RecordedIdentity(BaseModel):
    """The agent identity a record carries, each field absent where nothing declared it.

    Its fields are the ones a record gains, in order.
    """

    model_config = ConfigDict(frozen=True)

    agent_client_name: str | None = None
    agent_client_version: str | None = None
    agent_session: str | None = None
    terminal_session: str | None = None


class AgentIdentity(RecordedIdentity):
    """One server run's declared agent identity and the session it minted.

    It holds what the connecting harness declared and the session this server minted, each
    required, and the terminal session the launcher declared, absent when not declared.
    """

    agent_client_name: str
    agent_client_version: str
    agent_session: str


RECORD_FIELDS = tuple(RecordedIdentity.model_fields)
"""The fields a record gains, in declaration order."""

HEADERS = {
    field: "X-TCIP-" + "-".join(w.capitalize() for w in field.split("_"))
    for field in RECORD_FIELDS
}
"""The header each field travels under on an HTTP push."""

_current: AgentIdentity | None = None


def begin(client_name: str, client_version: str) -> AgentIdentity:
    """Record the handshake's declaration and mint this run's session id.

    The first handshake of a run holds for the whole run; a later handshake in the same run is
    logged, not applied.
    """
    global _current
    if _current is not None:
        logger.warning(
            "a second MCP handshake (%s %s) reached a server already serving %s %s; the identity "
            "of the first connection stands for this run",
            client_name,
            client_version,
            _current.agent_client_name,
            _current.agent_client_version,
        )
        return _current
    _current = AgentIdentity(
        agent_client_name=client_name,
        agent_client_version=client_version,
        agent_session="mcp_" + secrets.token_hex(8),
        terminal_session=os.environ.get(TERMINAL_SESSION_ENV) or None,
    )
    return _current


def end() -> None:
    """Forget the identity."""
    global _current
    _current = None


def current() -> AgentIdentity | None:
    """Return the identity this run established, or ``None`` outside a completed handshake."""
    return _current


def http_headers() -> dict[str, str]:
    """Return the identity as request headers for an HTTP push; empty outside a session.

    Each declared field travels, ``terminal_session`` only when one was declared. A declared name
    or version is any string, and a header value must be ASCII without control characters, so
    each value travels percent-encoded (letters, digits and ``-._~`` bare).
    """
    return {
        HEADERS[key]: quote(value, safe="")
        for key, value in (_current or RecordedIdentity())
        if value is not None
    }


def identity_from_headers(headers: Mapping[str, str]) -> RecordedIdentity:
    """Return the identity a request declared through :data:`HEADERS`.

    Each field is decoded from the percent-encoding :func:`http_headers` applies, and is ``None``
    when not sent or empty.
    """
    return RecordedIdentity(
        **{
            field: unquote(headers.get(header) or "") or None
            for field, header in HEADERS.items()
        }
    )


async def record_connecting_client(
    ctx: ServerRequestContext[Any, Any], call_next: CallNext
) -> HandlerResult:
    """Take the client's declaration on the first message that carries it, then pass the call on.

    Server middleware: the declaration is taken whatever the method, before the handler runs. A
    client that initialized without declaring itself leaves no identity.
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
