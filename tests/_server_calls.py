"""A tool called as an MCP client calls it, through the server's own argument schema."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anyio
import mcp.types as mcp_types
from mcp.client.session import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

if TYPE_CHECKING:
    from mcp.server import MCPServer
    from pydantic import JsonValue

    from tcip_mcp.workspace import BoundProject

DECLARED = mcp_types.Implementation(name="reviewing-harness", version="1.2.3")
"""The client a handshake declares unless the caller declares another."""


def client_server(bound: BoundProject | None) -> MCPServer:
    """Return the server an MCP client calls: its session bound to ``bound``, for no person."""
    from tcip_mcp.server import Session, build_server

    session = Session()
    session.bind(bound, None)
    return build_server(session)


def called_through_the_server(
    tool: str, arguments: dict, *, bound: BoundProject | None = None
) -> JsonValue:
    """Call ``tool`` with ``arguments`` through :func:`client_server` for ``bound``.

    The server is started for ``bound`` or, by default, for no project. Returns the tool error's
    text when the call is refused, or the answer the server sends, decoded from its JSON text,
    when it is admitted.
    """
    import asyncio
    import json

    from mcp.server.mcpserver.exceptions import ToolError

    try:
        result = asyncio.run(client_server(bound).call_tool(tool, arguments))
    except ToolError as exc:
        return str(exc)
    (content,) = result.content
    return json.loads(content.text)


def results_through_handshake(
    calls: list[tuple[str, dict]],
    declared: mcp_types.Implementation = DECLARED,
    *,
    bound: BoundProject | None,
) -> list[Any]:
    """Make ``calls`` in order through a handshake as ``declared`` and return each call's result.

    The real server runs in memory, built for ``bound`` (``None``: for no project). Each result
    is returned as the server sent it, a refused call's error included.
    """

    async def run() -> list[Any]:
        bodies: list[Any] = []
        async with create_client_server_memory_streams() as (
            client_streams,
            server_streams,
        ):
            async with anyio.create_task_group() as tg:
                # The run loop; MCPServer.run is stdio-only.
                server = client_server(bound)._lowlevel_server

                async def serve() -> None:
                    await server.run(
                        server_streams[0],
                        server_streams[1],
                        server.create_initialization_options(),
                        raise_exceptions=True,
                    )

                tg.start_soon(serve)
                async with ClientSession(
                    client_streams[0], client_streams[1], client_info=declared
                ) as session:
                    await session.initialize()
                    for name, arguments in calls:
                        bodies.append(await session.call_tool(name, arguments))
                tg.cancel_scope.cancel()
        return bodies

    return anyio.run(run)
