"""A tool called as an MCP client calls it, through the server's own argument schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server import MCPServer
    from pydantic import JsonValue

    from tcip_mcp.workspace import BoundProject


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
