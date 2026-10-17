"""A tool called as an MCP client calls it, through the server's own argument schema."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tcip_mcp.workspace import BoundProject


def called_through_the_server(
    tool: str, arguments: dict, *, bound: BoundProject | None = None
) -> Any:
    """``tool`` called with ``arguments`` through ``build_server(bound)``, a server started for
    ``bound`` or, by default, for no project: the tool error's text when the call is refused, or
    the answer the server sends, decoded from its JSON text, when it is admitted.
    """
    import asyncio
    import json

    from mcp.server.mcpserver.exceptions import ToolError

    from tcip_mcp.server import build_server

    try:
        result = asyncio.run(build_server(bound).call_tool(tool, arguments))
    except ToolError as exc:
        return str(exc)
    (content,) = result.content
    return json.loads(content.text)
